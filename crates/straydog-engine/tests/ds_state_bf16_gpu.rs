//! DeepSeek-V4.1 keeps its compressed rows and index keys as BF16, read through tagged pointers
//! (gpu/ops.hpp state_tag). A kernel turns each BF16 value into the F32 it stands for, so its
//! output from tagged BF16 rows is bit for bit its output from F32 rows holding the same values:
//! checked for the sparse attention (decode's one row and a prompt's many, F32 and WMMA kernels)
//! and the indexer, beside the op that rounds F32 into BF16.
//!
//! cargo test --release -p straydog-engine --features hip --test ds_state_bf16_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn floats(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| ((self.next() % 2_000_000) as f32 / 1_000_000.0 - 1.0) * scale).collect()
    }
}

/// The nearest BF16 (ties to even), as the host reference.
fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn widen(half: u16) -> f32 {
    f32::from_bits(u32::from(half) << 16)
}

fn upload<T: Copy>(gpu: &mut Backend, values: &[T]) -> Buffer {
    let buffer = gpu.buffer(std::mem::size_of_val(values).max(4) as u64).unwrap();
    gpu.write_buffer(&buffer, 0, values).unwrap();
    buffer
}

fn tagged(buffer: &Buffer) -> *mut std::ffi::c_void {
    buffer.base().wrapping_byte_add(1)
}

#[test]
fn rounding_into_bf16_is_to_the_nearest_even() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut values = Random(0x243f_6a88_85a3_08d3).floats(10_000, 1e3);
    values.extend([0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MAX, f32::MIN_POSITIVE, 1.0 + 1.0 / 256.0,
                   1.0 + 3.0 / 256.0, f32::from_bits(0x3f80_8000), f32::from_bits(0x3f81_8000)]);
    let input = upload(&mut gpu, &values);
    let out = gpu.buffer((values.len() * 2) as u64).unwrap();
    gpu.to_bf16(input.base(), values.len() as u64, out.base()).unwrap();
    let mut got = vec![0u16; values.len()];
    gpu.read_buffer(&out, 0, &mut got).unwrap();
    for (index, (&value, &half)) in values.iter().zip(&got).enumerate() {
        if value.is_nan() {
            assert!(widen(half).is_nan(), "a NaN became {half:#x}");
        } else {
            assert_eq!(half, bf16(value), "value {index} ({value:e})");
        }
    }
}

/// The sparse attention over `rows` rows from F32 compressed rows and from the same as tagged BF16.
fn attention(rows: usize, heads: usize, half: bool) {
    let (dim, window, raw_rows, compressed_rows, selections, rope_dims) = (512usize, 128usize, 256usize, 300usize, 512usize, 64usize);
    let mut gpu = Backend::new(0, 4 << 30).unwrap();
    let mut random = Random(0x9e37_79b9 ^ (rows * 131 + heads) as u64);
    let q = random.floats(rows * heads * dim, 1.0);
    let raw = random.floats(raw_rows * dim, 1.0);
    let halves: Vec<u16> = random.floats(compressed_rows * dim, 1.0).into_iter().map(bf16).collect();
    let compressed: Vec<f32> = halves.iter().map(|&half| widen(half)).collect();
    let sinks = random.floats(heads, 2.0);
    let frequencies: Vec<f32> = (0..rope_dims / 2).map(|i| 10000f32.powf(-(2.0 * i as f32) / rope_dims as f32)).collect();
    let positions: Vec<i32> = (0..rows as i32).map(|row| 1000 + row).collect();
    let selected: Vec<i32> = (0..rows * selections)
        .map(|at| if at % 37 == 5 { -1 } else { (random.next() % compressed_rows as u64) as i32 }).collect();
    let (q_b, raw_b, sinks_b, freq_b) = (upload(&mut gpu, &q), upload(&mut gpu, &raw), upload(&mut gpu, &sinks), upload(&mut gpu, &frequencies));
    let (f32_rows, bf16_rows) = (upload(&mut gpu, &compressed), upload(&mut gpu, &halves));
    let (positions_b, selected_b) = (upload(&mut gpu, &positions), upload(&mut gpu, &selected));
    gpu.attention_turns_reserve(rows as u64, rope_dims as u64).unwrap();
    let scratch = gpu.buffer(256 << 20).unwrap();
    gpu.configure_gemm(8, half).unwrap();
    let mut run = |compressed: *const std::ffi::c_void| -> Vec<u32> {
        let out = gpu.buffer((rows * heads * dim * 4) as u64).unwrap();
        gpu.sparse_attention(q_b.base(), raw_b.base(), raw_rows as u64, positions_b.base(), window as u64,
                             Some((compressed, compressed_rows as u64)), Some((selected_b.base(), selections as u64)), sinks_b.base(),
                             rows as u64, heads as u64, dim as u64, 1.0 / (dim as f32).sqrt(), Some((rope_dims as u64, &freq_b)),
                             Some(&scratch), out.base()).unwrap();
        let mut values = vec![0u32; rows * heads * dim];
        gpu.read_buffer(&out, 0, &mut values).unwrap();
        values
    };
    let from_f32 = run(f32_rows.base());
    let from_bf16 = run(tagged(&bf16_rows));
    assert!(from_f32.iter().any(|&bits| bits != 0), "no output");
    assert_eq!(from_f32, from_bf16, "{rows} rows, {heads} heads, FP16 {half}");
}

#[test]
fn the_attention_of_bf16_rows_is_that_of_their_f32_values() {
    for (rows, heads, half) in [(1, 32, false), (4, 32, false), (64, 32, false), (64, 32, true), (64, 64, true)] {
        attention(rows, heads, half);
    }
}

#[test]
fn the_indexer_of_bf16_keys_is_that_of_their_f32_values() {
    let (groups, heads, dim, k, rope_dims) = (3000usize, 32usize, 128usize, 512usize, 64usize);
    for (rows, half) in [(1usize, false), (4, false), (64, false), (64, true)] {
        let mut gpu = Backend::new(0, 2 << 30).unwrap();
        let mut random = Random(0x6a09_e667 ^ rows as u64);
        let (q, weights) = (random.floats(rows * heads * dim, 1.0), random.floats(rows * heads, 1.0));
        let halves: Vec<u16> = random.floats(groups * dim, 1.0).into_iter().map(bf16).collect();
        let keys: Vec<f32> = halves.iter().map(|&half| widen(half)).collect();
        let frequencies: Vec<f32> = (0..rope_dims / 2).map(|i| 10000f32.powf(-(2.0 * i as f32) / rope_dims as f32)).collect();
        let positions: Vec<i32> = (0..rows as i32).map(|row| 4000 + 37 * row).collect();
        let visible: Vec<i32> = (0..rows as i32).map(|row| 1000 + 30 * row).collect();
        let (q_b, weights_b, freq_b) = (upload(&mut gpu, &q), upload(&mut gpu, &weights), upload(&mut gpu, &frequencies));
        let (f32_keys, bf16_keys) = (upload(&mut gpu, &keys), upload(&mut gpu, &halves));
        let (positions_b, visible_b) = (upload(&mut gpu, &positions), upload(&mut gpu, &visible));
        gpu.configure_gemm(8, half).unwrap();
        let mut run = |keys: *const std::ffi::c_void| -> (Vec<u32>, Vec<i32>) {
            let (scores, out) = (gpu.buffer((rows * groups * 4) as u64).unwrap(), gpu.buffer((rows * k * 4) as u64).unwrap());
            gpu.index_select(q_b.base(), keys, weights_b.base(), visible_b.base(), rows as u64, groups as u64, heads as u64, dim as u64,
                             0.01, (rope_dims as u64, &positions_b, &freq_b), scores.base(), k as u64, out.base()).unwrap();
            let (mut s, mut o) = (vec![0u32; rows * groups], vec![0i32; rows * k]);
            gpu.read_buffer(&scores, 0, &mut s).unwrap();
            gpu.read_buffer(&out, 0, &mut o).unwrap();
            (s, o)
        };
        let from_f32 = run(f32_keys.base());
        let from_bf16 = run(tagged(&bf16_keys));
        assert_eq!(from_f32, from_bf16, "{rows} rows, FP16 {half}");
    }
}
