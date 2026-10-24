//! Full DSpark injection dimensions, independent F64 oracle, and deterministic ring overwrite.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::Backend;
use straydog_engine::ops::{Buffer, DsparkInjectStage};

const INPUT: usize = 15360;
const WIDTH: usize = 5120;
const KV: usize = 512;
const WINDOW: usize = 128;

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0
    }
    fn floats(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale).collect()
    }
    fn norms(&mut self, n: usize) -> Vec<f32> {
        self.floats(n, 0.5).iter().map(|x| x + 1.0).collect()
    }
    fn q8(&mut self, rows: usize, cols: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows * cols / 32 * 34);
        for _ in 0..rows * cols / 32 {
            // Exact positive power-of-two FP16 scales; signed random Q8 quants.
            bytes.extend_from_slice(&((6 + self.next() % 4) as u16 * 1024).to_le_bytes());
            for _ in 0..32 { bytes.push(((self.next() % 17) as u8).wrapping_sub(8)); }
        }
        bytes
    }
}

fn upload<T: Copy>(gpu: &mut Backend, values: &[T]) -> Buffer {
    let b = gpu.buffer(std::mem::size_of_val(values) as u64).unwrap();
    gpu.write_buffer(&b, 0, values).unwrap(); b
}
fn read(gpu: &mut Backend, b: &Buffer, n: usize) -> Vec<f32> {
    let mut v = vec![0.0; n]; gpu.read_buffer(b, 0, &mut v).unwrap(); v
}
fn dequantize(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(34).flat_map(|block| {
        let half = u16::from_le_bytes([block[0], block[1]]);
        let scale = 2f32.powi(i32::from(half >> 10) - 15);
        block[2..].iter().map(move |&q| (q as i8) as f32 * scale)
    }).collect()
}
fn matmul(weights: &[f32], cols: usize, inputs: &[f64]) -> Vec<f64> {
    inputs.chunks_exact(cols).flat_map(|x| {
        weights.chunks_exact(cols).map(move |row| row.iter().zip(x).map(|(&a, &b)| a as f64 * b).sum())
    }).collect()
}
fn norm(values: &mut [f64], weights: &[f32]) {
    for row in values.chunks_exact_mut(weights.len()) {
        let rms = (row.iter().map(|x| x * x).sum::<f64>() / row.len() as f64 + 1e-20).sqrt();
        for (x, &w) in row.iter_mut().zip(weights) { *x = *x / rms * w as f64; }
    }
}
fn check(label: &str, actual: &[f32], reference: &[f64]) {
    assert_eq!(actual.len(), reference.len());
    let mut worst = 0f64;
    for (i, (&a, &b)) in actual.iter().zip(reference).enumerate() {
        let error = (a as f64 - b).abs() / (1.0 + b.abs());
        assert!(error <= 1e-4, "{label} value {i}: {a} vs {b}, error {error}");
        worst = worst.max(error);
    }
    eprintln!("{label}: {} values, max abs/(1+abs(ref)) {worst:e}", actual.len());
}
fn expected_ring(kv: &[f64], positions: &[i32], frequencies: &[f32], sentinel: &[f32]) -> Vec<f64> {
    let mut ring: Vec<f64> = sentinel.iter().map(|&x| x as f64).collect();
    for (&position, row) in positions.iter().zip(kv.chunks_exact(KV)) {
        let at = position.rem_euclid(WINDOW as i32) as usize * KV;
        ring[at..at + KV].copy_from_slice(row);
        for (pair, &frequency) in frequencies.iter().enumerate() {
            // Use the actual F32 frequency table supplied to the existing RoPE op,
            // with independent F64 trig / normalization / matrix accumulation.
            let (s, c) = (position as f64 * frequency as f64).sin_cos();
            let i = KV - 64 + 2 * pair;
            ring[at + i] = row[i] * c - row[i + 1] * s;
            ring[at + i + 1] = row[i] * s + row[i + 1] * c;
        }
    }
    ring
}

#[test]
fn full_dimensions_n1_n7_n130_match_f64_and_last_writer() {
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let features = random.floats(130 * INPUT, 0.1);
    let fc_bytes = random.q8(WIDTH, INPUT);
    let encoder_norm = random.norms(WIDTH);
    let kv_bytes: [Vec<u8>; 3] = std::array::from_fn(|_| random.q8(KV, WIDTH));
    let kv_norms: [Vec<f32>; 3] = std::array::from_fn(|_| random.norms(KV));
    let frequencies: Vec<f32> = (0..32).map(|i| 10000f64.powf(-((2 * i) as f64) / 64.0) as f32).collect();
    let sentinel: [Vec<f32>; 3] = std::array::from_fn(|_| random.floats(WINDOW * KV, 0.2));
    let inputs: Vec<f64> = features.iter().map(|&x| x as f64).collect();
    let mut expected_encoded = matmul(&dequantize(&fc_bytes), INPUT, &inputs);
    norm(&mut expected_encoded, &encoder_norm);
    let expected_kv: [Vec<f64>; 3] = std::array::from_fn(|i| {
        let mut kv = matmul(&dequantize(&kv_bytes[i]), WIDTH, &expected_encoded);
        norm(&mut kv, &kv_norms[i]); kv
    });
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let features = upload(&mut gpu, &features);
    let fc = gpu.upload_bytes(&fc_bytes).unwrap();
    let encoder_norm = upload(&mut gpu, &encoder_norm);
    let kv = kv_bytes.each_ref().map(|b| gpu.upload_bytes(b).unwrap());
    let norms = kv_norms.each_ref().map(|n| upload(&mut gpu, n));
    let rings = sentinel.each_ref().map(|s| upload(&mut gpu, s));
    let frequencies_gpu = upload(&mut gpu, &frequencies);
    let encoded = gpu.buffer((130 * WIDTH * 4) as u64).unwrap();
    let scratch = gpu.buffer(16 << 20).unwrap();
    for n in [1usize, 7, 130] {
        let mut positions: Vec<i32> = (97..97 + n as i32).collect();
        if n == 7 { positions[6] = positions[2]; }
        let positions_gpu = upload(&mut gpu, &positions);
        for i in 0..3 { gpu.write_buffer(&rings[i], 0, &sentinel[i]).unwrap(); }
        let stages = std::array::from_fn(|i| DsparkInjectStage { kv: &kv[i], norm: &norms[i], ring: &rings[i] });
        gpu.dspark_inject(features.base(), &positions_gpu, n as u64, &fc, &encoder_norm, &stages,
            &frequencies_gpu, encoded.base(), &scratch).unwrap();
        check(&format!("N={n} encoded"), &read(&mut gpu, &encoded, n * WIDTH), &expected_encoded[..n * WIDTH]);
        for i in 0..3 {
            let expected = expected_ring(&expected_kv[i], &positions, &frequencies, &sentinel[i]);
            let actual = read(&mut gpu, &rings[i], WINDOW * KV);
            check(&format!("N={n} stage={i} ring"), &actual, &expected);
            for row in 0..WINDOW {
                if !positions.iter().any(|&p| p.rem_euclid(WINDOW as i32) as usize == row) {
                    let range = row * KV..(row + 1) * KV;
                    assert_eq!(actual[range.clone()], sentinel[i][range], "untouched N={n} stage={i} row={row}");
                }
            }
        }
        assert!(gpu.dspark_inject(features.base(), &positions_gpu, 0, &fc, &encoder_norm, &stages,
            &frequencies_gpu, encoded.base(), &scratch).is_err());
        let short = scratch.slice(0, 4096 * n as u64 - 1);
        assert!(gpu.dspark_inject(features.base(), &positions_gpu, n as u64, &fc, &encoder_norm, &stages,
            &frequencies_gpu, encoded.base(), &short).is_err());
    }
}
