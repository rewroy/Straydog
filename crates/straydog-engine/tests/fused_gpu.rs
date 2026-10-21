//! Fused decode ops against the ops they stand in for, on random data:
//! - sd_op_delta_step against sd_op_conv_silu, sd_op_delta_rule and sd_op_gated_norm (Qwen3.6's
//!   DeltaNet shape: 8 key heads, 16 value heads of 128, a 4-tap convolution): the same history,
//!   states and outputs within F32 rounding;
//! - sd_op_norm_linears over Q8_0 and F32 rows in one launch against sd_op_rms_norm and a
//!   sd_op_linears launch per format: the same bits.
//!
//! cargo test --release -p straydog-engine --features hip --test fused_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

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
    fn float(&mut self, scale: f32) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
    }
    fn floats(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| self.float(scale)).collect()
    }
}

fn upload(gpu: &mut Backend, values: &[f32]) -> Buffer {
    let buffer = gpu.buffer((values.len() * 4) as u64).unwrap();
    gpu.write_buffer(&buffer, 0, values).unwrap();
    buffer
}

fn read(gpu: &mut Backend, buffer: &Buffer, count: usize) -> Vec<u32> {
    let mut values = vec![0f32; count];
    gpu.read_buffer(buffer, 0, &mut values).unwrap();
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn a_fused_delta_step_matches_the_three_ops() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let (k_heads, v_heads, dim, kernel) = (8usize, 16usize, 128usize, 4usize);
    let channels = (2 * k_heads + v_heads) * dim;
    let qkv = random.floats(channels, 2.0);
    let history = random.floats((kernel - 1) * channels, 2.0);
    let taps = random.floats(kernel * channels, 0.5);
    let (beta, alpha) = (random.floats(v_heads, 2.0), random.floats(v_heads, 2.0));
    let (dt, a) = (random.floats(v_heads, 1.0), random.floats(v_heads, 1.0).iter().map(|x| -x.abs()).collect::<Vec<_>>());
    let state = random.floats(v_heads * dim * dim, 0.1);
    let gate = random.floats(v_heads * dim, 3.0);
    let weight = random.floats(dim, 1.0);
    let eps = 1e-6;
    let shared = [upload(&mut gpu, &beta), upload(&mut gpu, &alpha), upload(&mut gpu, &dt), upload(&mut gpu, &a), upload(&mut gpu, &gate),
                  upload(&mut gpu, &weight), upload(&mut gpu, &taps)];
    let [beta, alpha, dt, a, gate, weight, taps] = shared;
    // the three ops
    let (qkv_1, history_1, state_1) = (upload(&mut gpu, &qkv), upload(&mut gpu, &history), upload(&mut gpu, &state));
    let (core, out_1) = (gpu.buffer((v_heads * dim * 4) as u64).unwrap(), gpu.buffer((v_heads * dim * 4) as u64).unwrap());
    gpu.conv_silu(qkv_1.base(), history_1.base(), taps.base(), channels as u64, kernel as u64, 1).unwrap();
    gpu.delta_rule(qkv_1.base(), beta.base(), alpha.base(), dt.base(), a.base(), state_1.base(), k_heads as u64, v_heads as u64, dim as u64,
                   1, eps, core.base()).unwrap();
    gpu.gated_norm(core.base(), gate.base(), &weight, v_heads as u64, dim as u64, eps, out_1.base()).unwrap();
    // fused
    let (qkv_2, history_2, state_2) = (upload(&mut gpu, &qkv), upload(&mut gpu, &history), upload(&mut gpu, &state));
    let out_2 = gpu.buffer((v_heads * dim * 4) as u64).unwrap();
    assert!(gpu.delta_step(qkv_2.base(), history_2.base(), taps.base(), kernel as u64, beta.base(), alpha.base(), dt.base(), a.base(),
                           state_2.base(), k_heads as u64, v_heads as u64, dim as u64, eps, gate.base(), &weight, out_2.base()).unwrap());
    assert_eq!(read(&mut gpu, &history_1, history.len()), read(&mut gpu, &history_2, history.len()), "histories");
    // The states and outputs agree to within F32 rounding (a key head's normalized q and k can
    // round apart by an ulp between the two kernels, as their compiled code differs).
    for (label, first, second, count) in [("states", &state_1, &state_2, state.len()), ("outputs", &out_1, &out_2, v_heads * dim)] {
        let (a, b) = (read(&mut gpu, first, count), read(&mut gpu, second, count));
        let (a, b): (Vec<f32>, Vec<f32>) = (a.into_iter().map(f32::from_bits).collect(), b.into_iter().map(f32::from_bits).collect());
        let scale = a.iter().fold(0f32, |m, x| m.max(x.abs()));
        let worst = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        let differ = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        eprintln!("{label}: {differ} of {count} differ, at most {worst:e} (largest value {scale:e})");
        assert!(worst <= 1e-5 * scale, "{label} differ by {worst:e}");
    }
}

#[test]
fn norm_linears_of_two_formats_match_a_launch_per_format() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    let width = 2048usize;
    let x = upload(&mut gpu, &random.floats(width, 3.0));
    let weight = upload(&mut gpu, &random.floats(width, 1.0));
    // Q8_0 rows: random quants and scales
    let q8 = |random: &mut Random, rows: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows * width / 32 * 34);
        for _ in 0..rows * width / 32 {
            // an FP16 scale between 2^-7 and 2^-3, either sign
            let bits = ((random.next() & 1) << 15) | ((8 + random.next() % 5) << 10) | (random.next() & 0x3ff);
            bytes.extend_from_slice(&(bits as u16).to_le_bytes());
            for _ in 0..32 {
                bytes.push(random.next() as u8);
            }
        }
        bytes
    };
    let f32_rows = |random: &mut Random, rows: usize| -> Vec<u8> {
        random.floats(rows * width, 0.05).iter().flat_map(|value| value.to_le_bytes()).collect()
    };
    let shapes = [(8u32, 1024usize), (8, 512), (0, 256), (0, 1)];
    let mut matrices = Vec::new();
    for &(kind, rows) in &shapes {
        let bytes = if kind == 8 { q8(&mut random, rows) } else { f32_rows(&mut random, rows) };
        matrices.push((kind, rows, gpu.upload_bytes(&bytes).unwrap()));
    }
    let outputs = |gpu: &mut Backend| -> Vec<Buffer> { shapes.iter().map(|&(_, rows)| gpu.buffer((rows * 4) as u64).unwrap()).collect() };
    let linears = |normed: &Buffer, outs: &[Buffer], kinds: &[u32]| -> Vec<Linear> {
        matrices.iter().zip(outs).filter(|((kind, _, _), _)| kinds.contains(kind)).map(|((kind, rows, data), out)| Linear {
            weights: *data, kind: *kind, rows: *rows as u64, cols: width as u64, first_row: 0, input: normed.base(), output: out.base(),
        }).collect()
    };
    let scratch = gpu.buffer(64 << 20).unwrap();
    // a norm, then a launch per format
    let (normed_1, outs_1) = (gpu.buffer((width * 4) as u64).unwrap(), outputs(&mut gpu));
    gpu.rms_norm(x.base(), &weight, 1, width as u64, 1e-6, normed_1.base()).unwrap();
    for kind in [8u32, 0] {
        gpu.linears(&linears(&normed_1, &outs_1, &[kind]), 1, 0, 0, &scratch).unwrap();
    }
    // one launch
    let (normed_2, outs_2) = (gpu.buffer((width * 4) as u64).unwrap(), outputs(&mut gpu));
    gpu.norm_linears(x.base(), &weight, width as u64, 1e-6, normed_2.base(), &linears(&normed_2, &outs_2, &[8, 0])).unwrap();
    assert_eq!(read(&mut gpu, &normed_1, width), read(&mut gpu, &normed_2, width), "the norm");
    for (index, &(kind, rows)) in shapes.iter().enumerate() {
        assert_eq!(read(&mut gpu, &outs_1[index], rows), read(&mut gpu, &outs_2[index], rows), "kind {kind}, {rows} rows");
    }
}
