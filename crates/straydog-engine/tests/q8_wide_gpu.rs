//! The wide Q8 decode kernels (Backend::wide_q8_decode: Qwen3.8's projections) against a host
//! reference on random data, as q8_decode_gpu.rs checks DeepSeek-V4.1's:
//! - Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 GEMVs: within F32 rounding of the rows' dots with the quantized
//!   input (a K-quant's min term against the input itself);
//! - two to four vectors: each vector's result the bits of its own one-vector GEMV;
//! - a list mixing them with BF16 rows (a DeltaNet layer's qkv, gate, beta and alpha): each row the
//!   bits of its matrix alone, for one to four vectors.
//!
//! cargo test --release -p straydog-engine --features hip --test q8_wide_gpu (or --features cuda), under the GPU lock.
// (RDNA4's v_dot4 and CUDA's dp4a: both take the kernels; other devices run these ops in F32, with nothing of theirs to compare)
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::formats;
use straydog_engine::gpu::Backend;
use straydog_engine::ops::{Buffer, Linear};

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn floats(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale).collect()
    }
    /// Random blocks of `kind` with a small FP16 scale (and a K-quant's dmin); BF16 values of a few units.
    fn blocks(&mut self, kind: u32, values: usize) -> Vec<u8> {
        if kind == 30 {
            return self.floats(values, 0.05).iter().flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes()).collect();
        }
        let (per, bytes) = match kind {
            8 => (32, 34),
            12 => (256, 144),
            13 => (256, 176),
            14 => (256, 210),
            23 => (256, 136),
            _ => panic!("no format {kind}"),
        };
        let mut out: Vec<u8> = (0..values / per * bytes).map(|_| self.next() as u8).collect();
        for block in out.chunks_exact_mut(bytes) {
            let at = if kind == 14 { 208 } else { 0 };
            block[at..at + 2].copy_from_slice(&0x2c00u16.to_le_bytes());
            if kind == 12 || kind == 13 {
                block[2..4].copy_from_slice(&0x2400u16.to_le_bytes());
            }
        }
        out
    }
}

fn upload(gpu: &mut Backend, values: &[f32]) -> Buffer {
    let buffer = gpu.buffer((values.len() * 4) as u64).unwrap();
    gpu.write_buffer(&buffer, 0, values).unwrap();
    buffer
}

fn read(gpu: &mut Backend, buffer: &Buffer, count: usize) -> Vec<f32> {
    let mut values = vec![0f32; count];
    gpu.read_buffer(buffer, 0, &mut values).unwrap();
    values
}

fn quantized(x: &[f32]) -> Vec<f32> {
    x.chunks_exact(32).flat_map(|group| {
        let top = group.iter().fold(0f32, |top, value| top.max(value.abs()));
        let inverse = if top > 0.0 { 127.0 / top } else { 0.0 };
        let scale = top / 127.0;
        group.iter().map(move |value| (value * inverse).round_ties_even() * scale).collect::<Vec<_>>()
    }).collect()
}

fn dots(weights: &[f32], cols: usize, x: &[f32]) -> Vec<(f64, f64)> {
    weights.chunks_exact(cols).map(|row| {
        row.iter().zip(x).fold((0.0, 0.0), |(sum, size), (&w, &v)| (sum + f64::from(w) * f64::from(v), size + (f64::from(w) * f64::from(v)).abs()))
    }).collect()
}

fn decoded(kind: u32, bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    formats::decode_into(kind, bytes, &mut out).unwrap();
    out
}

/// The rows' dots as the Q8 kernels take them: the quantized input, a K-quant's mins against the input itself.
fn expected(kind: u32, bytes: &[u8], cols: usize, x: &[f32]) -> Vec<(f64, f64)> {
    if kind != 12 && kind != 13 {
        return dots(&decoded(kind, bytes), cols, &quantized(x));
    }
    let size = if kind == 12 { 144 } else { 176 };
    let zeroed = |at: usize| {
        let mut copy = bytes.to_vec();
        for block in copy.chunks_exact_mut(size) { block[at..at + 2].copy_from_slice(&[0, 0]); }
        decoded(kind, &copy)
    };
    dots(&zeroed(2), cols, &quantized(x)).iter().zip(dots(&zeroed(0), cols, x)).map(|(&(a, sa), (b, sb))| (a + b, sa + sb)).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn wide_q8_gemvs_match_the_quantized_input_and_each_vector_alone() {
    straydog_engine::gpu::load_iq_tables().unwrap();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    gpu.wide_q8_decode().unwrap();
    let mut random = Random(0x51f1_5a0e_7c2d_93b1);
    let scratch = gpu.buffer(1 << 20).unwrap();
    for (kind, rows, cols) in [(14u32, 10240usize, 2560usize), (14, 2560, 6144), (12, 6144, 2560), (12, 2560, 6144), (13, 10240, 2560),
                               (13, 2560, 6144), (8, 6144, 2560), (8, 2560, 6144), (23, 6144, 2560), (23, 640, 2560), (14, 124160, 2560)] {
        let bytes = random.blocks(kind, rows * cols);
        let matrix = gpu.upload_bytes(&bytes).unwrap();
        let x = random.floats(4 * cols, 2.0);
        let input = upload(&mut gpu, &x);
        let output = gpu.buffer((4 * rows * 4) as u64).unwrap();
        let mut alone = Vec::new();
        for v in 0..4 {
            gpu.linear(&matrix, kind, rows as u64, cols as u64, 0, input.at(v * cols), 0, 1, output.base(), 0, &scratch).unwrap();
            let got = read(&mut gpu, &output, rows);
            for (row, (&value, &(sum, size))) in got.iter().zip(&expected(kind, &bytes, cols, &x[v * cols..(v + 1) * cols])).enumerate() {
                assert!((f64::from(value) - sum).abs() <= 2e-6 * size + 1e-6,
                        "kind {kind} {rows} x {cols}, vector {v}, row {row}: {value} against {sum} (products {size})");
            }
            alone.push(bits(&got));
        }
        for vectors in 2..=4usize {
            gpu.linear(&matrix, kind, rows as u64, cols as u64, 0, input.base(), cols as u64, vectors as u64, output.base(), rows as u64,
                       &scratch).unwrap();
            let got = read(&mut gpu, &output, vectors * rows);
            for v in 0..vectors {
                assert_eq!(bits(&got[v * rows..(v + 1) * rows]), alone[v], "kind {kind} {rows} x {cols}: vector {v} of {vectors}");
            }
        }
    }
}

#[test]
fn a_mixed_wide_list_gives_each_matrix_its_own_bits() {
    // (a DeltaNet layer's qkv, gate, beta and alpha, decoded alone and as a verification's rows; then the Q5_K and Q8_0 bundles' lists)
    mixed_list(&[(14u32, 10240usize), (12, 6144), (30, 48), (30, 48)]);
    mixed_list(&[(13, 10240), (14, 6144), (30, 48), (30, 48)]);
    mixed_list(&[(8, 10240), (8, 6144), (30, 48), (30, 48)]);
    mixed_list(&[(14, 12288), (14, 512), (14, 512), (30, 512), (30, 128)]);
}

fn mixed_list(shapes: &[(u32, usize)]) {
    straydog_engine::gpu::load_iq_tables().unwrap();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    gpu.wide_q8_decode().unwrap();
    let mut random = Random(0x3c6e_f372_fe94_f82b);
    let scratch = gpu.buffer(1 << 20).unwrap();
    let cols = 2560usize;
    let matrices: Vec<Buffer> = shapes.iter().map(|&(kind, rows)| gpu.upload_bytes(&random.blocks(kind, rows * cols)).unwrap()).collect();
    let x = random.floats(4 * cols, 2.0);
    let input = upload(&mut gpu, &x);
    let total: usize = shapes.iter().map(|&(_, rows)| rows).sum();
    let output = gpu.buffer((4 * total * 4) as u64).unwrap();
    // each matrix alone, each vector alone
    let mut alone = vec![Vec::new(); 4];
    for (v, row) in alone.iter_mut().enumerate() {
        for (&(kind, rows), matrix) in shapes.iter().zip(&matrices) {
            gpu.linear(matrix, kind, rows as u64, cols as u64, 0, input.at(v * cols), 0, 1, output.base(), 0, &scratch).unwrap();
            row.extend(bits(&read(&mut gpu, &output, rows)));
        }
    }
    for vectors in 1..=4usize {
        let mut offset = 0;
        let items: Vec<Linear> = shapes.iter().zip(&matrices).map(|(&(kind, rows), matrix)| {
            let item = Linear { weights: *matrix, kind, rows: rows as u64, cols: cols as u64, first_row: 0, input: input.base(),
                                output: output.at(offset) };
            offset += rows;
            item
        }).collect();
        gpu.linears(&items, vectors as u64, cols as u64, total as u64, &scratch).unwrap();
        let got = read(&mut gpu, &output, vectors * total);
        for (v, expected) in alone.iter().enumerate().take(vectors) {
            assert_eq!(&bits(&got[v * total..(v + 1) * total]), expected, "vector {v} of {vectors}");
        }
    }
}
