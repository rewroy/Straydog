//! DeepSeek-V4.1's sparse attention of an FP16 prompt pass (the WMMA kernels, ds_attention_wmma.hpp)
//! against the F32 kernels, on generated rows at its shapes: a 512-wide latent shared by every head
//! (32 on each of two devices, 64 on one), a 128-row window ring, 512 selected compressed rows (some
//! unselected), rope on the last 64 columns and a sink per head. Each head's output is within FP16's
//! rounding of the F32 one's, and the WMMA kernels ran (the outputs differ in some bits).
//!
//! cargo test --release -p straydog-engine --features hip --test ds_attention_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::Backend;

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn value(&mut self) -> f32 {
        (self.next() % 2_000_000) as f32 / 1_000_000.0 - 1.0
    }
}

fn attention(heads: usize) {
    let (rows, dim, window, raw_rows, compressed_rows, selections, rope_dims) = (64usize, 512usize, 128usize, 256usize, 300usize, 512usize, 64usize);
    let mut gpu = Backend::new(0, 4 << 30).unwrap();
    let mut random = Random(0x9e37_79b9 + heads as u64);
    let mut floats = |count: usize, scale: f32| -> Vec<f32> { (0..count).map(|_| random.value() * scale).collect() };
    let q = floats(rows * heads * dim, 1.0);
    let raw = floats(raw_rows * dim, 1.0);
    let compressed = floats(compressed_rows * dim, 1.0);
    let sinks = floats(heads, 2.0);
    let frequencies: Vec<f32> = (0..rope_dims / 2).map(|i| 10000f32.powf(-(2.0 * i as f32) / rope_dims as f32)).collect();
    // positions from 1000 on: every row sees a full window
    let positions: Vec<i32> = (0..rows as i32).map(|row| 1000 + row).collect();
    let mut chooser = Random(0x51ed_270b);
    let selected: Vec<i32> = (0..rows * selections)
        .map(|at| if at % 37 == 5 { -1 } else { (chooser.next() % compressed_rows as u64) as i32 }).collect();
    let upload = |gpu: &mut Backend, values: &[f32]| {
        let buffer = gpu.buffer((values.len() * 4) as u64).unwrap();
        gpu.write_buffer(&buffer, 0, values).unwrap();
        buffer
    };
    let (q_buffer, raw_buffer, compressed_buffer, sinks_buffer, frequencies_buffer) =
        (upload(&mut gpu, &q), upload(&mut gpu, &raw), upload(&mut gpu, &compressed), upload(&mut gpu, &sinks), upload(&mut gpu, &frequencies));
    let positions_buffer = gpu.buffer((rows * 4) as u64).unwrap();
    gpu.write_buffer(&positions_buffer, 0, &positions).unwrap();
    let selected_buffer = gpu.buffer((selected.len() * 4) as u64).unwrap();
    gpu.write_buffer(&selected_buffer, 0, &selected).unwrap();
    gpu.attention_turns_reserve(rows as u64, rope_dims as u64).unwrap();
    let scratch = gpu.buffer(256 << 20).unwrap();
    let run = |gpu: &mut Backend, half: bool| -> Vec<f32> {
        gpu.configure_gemm(8, half).unwrap();
        let out = gpu.buffer((rows * heads * dim * 4) as u64).unwrap();
        gpu.sparse_attention(q_buffer.base(), raw_buffer.base(), raw_rows as u64, positions_buffer.base(), window as u64,
                             Some((compressed_buffer.base(), compressed_rows as u64)), Some((selected_buffer.base(), selections as u64)),
                             sinks_buffer.base(), rows as u64, heads as u64, dim as u64, 1.0 / (dim as f32).sqrt(),
                             Some((rope_dims as u64, &frequencies_buffer)), Some(&scratch), out.base()).unwrap();
        let mut values = vec![0f32; rows * heads * dim];
        gpu.read_buffer(&out, 0, &mut values).unwrap();
        values
    };
    let exact = run(&mut gpu, false);
    let wmma = run(&mut gpu, true);
    let differ = exact.iter().zip(&wmma).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert!(differ > 0, "{heads} heads: the WMMA kernels did not run");
    let mut worst = 0f64;
    for (a, b) in exact.chunks_exact(dim).zip(wmma.chunks_exact(dim)) {
        assert!(b.iter().all(|value| value.is_finite()), "{heads} heads: a non-finite output");
        let (mut difference, mut norm) = (0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            difference += ((x - y) as f64).powi(2);
            norm += (x as f64).powi(2);
        }
        worst = worst.max((difference / norm.max(f64::MIN_POSITIVE)).sqrt());
    }
    eprintln!("{heads} heads: worst head's relative difference {worst:.3e}, {differ} values differ");
    assert!(worst < 5e-3, "{heads} heads: a head's output is {worst:.3e} from F32's");
}

#[test]
fn the_wmma_attention_of_an_fp16_pass_agrees_with_f32_across_two_devices_heads() {
    attention(32);
}

#[test]
fn the_wmma_attention_of_an_fp16_pass_agrees_with_f32_on_one_device_s_heads() {
    attention(64);
}
