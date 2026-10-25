//! The Q8 decode kernels (Backend::configure_decode) against a host reference on random data: a
//! decode row's input quantized per 32 values to INT8 (round(x * 127 / max |x|), the scale
//! max |x| / 127), then the rows' dots with the host-decoded weights in F64:
//! - Q5_K and Q8_0 GEMVs (DeepSeek-V4.1's decode projection widths, and rows that leave a wave
//!   part empty): within F32 rounding of that reference (a Q5_K group's min term takes the input's
//!   own sum, as the kernel does);
//! - a GEMV of two to four vectors (a verify pass's rows): each vector's result the bits of its own
//!   one-vector GEMV;
//! - the IQ2_XXS, IQ2_XS and IQ2_S experts' gate/up (two routes, SwiGLU clamped, times the routes'
//!   weights): the same;
//! - with the Q8 kernels off, the F32 kernels' results are the unquantized input's within F32
//!   rounding, as before.
//!
//! cargo test --release -p straydog-engine --features hip --test q8_decode_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::formats;
use straydog_engine::gpu::Backend;
use straydog_engine::ops::{Buffer, ExpertSplit};

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
    /// Random blocks of `kind` with a small leading FP16 scale (and Q5_K's dmin).
    fn blocks(&mut self, kind: u32, values: usize) -> Vec<u8> {
        let (per, bytes) = match kind {
            8 => (32, 34),
            13 => (256, 176),
            16 => (256, 66),
            17 => (256, 74),
            22 => (256, 82),
            _ => panic!("no format {kind}"),
        };
        let mut out: Vec<u8> = (0..values / per * bytes).map(|_| self.next() as u8).collect();
        for block in out.chunks_exact_mut(bytes) {
            block[..2].copy_from_slice(&0x2c00u16.to_le_bytes());
            if kind == 13 {
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

/// The input as the Q8 kernels take it: each 32 values' INT8 codes times their scale.
fn quantized(x: &[f32]) -> Vec<f32> {
    x.chunks_exact(32).flat_map(|group| {
        let top = group.iter().fold(0f32, |top, value| top.max(value.abs()));
        let inverse = if top > 0.0 { 127.0 / top } else { 0.0 };
        let scale = top / 127.0;
        group.iter().map(move |value| (value * inverse).round_ties_even() * scale).collect::<Vec<_>>()
    }).collect()
}

/// Each row's dot with `x` in F64, and the sum of its products' magnitudes (the rounding's scale).
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

#[test]
fn q8_gemvs_match_the_quantized_input_on_the_host() {
    straydog_engine::gpu::load_iq_tables().unwrap();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x9b05_688c_2b3e_6c1f);
    let scratch = gpu.buffer(1 << 20).unwrap();
    for (kind, rows, cols) in [(13u32, 1280usize, 5120usize), (13, 300, 4096), (13, 16384, 1280), (8, 1792, 5120), (8, 77, 1024)] {
        let bytes = random.blocks(kind, rows * cols);
        let weights = decoded(kind, &bytes);
        // Q5_K's values d * scale * q - dmin * min, apart: the first part (dmin zeroed) meets the
        // quantized input, the second (d zeroed) the input itself
        let parts = (kind == 13).then(|| {
            let zeroed = |at: usize| {
                let mut copy = bytes.clone();
                for block in copy.chunks_exact_mut(176) { block[at..at + 2].copy_from_slice(&[0, 0]); }
                decoded(kind, &copy)
            };
            (zeroed(2), zeroed(0))
        });
        let matrix = gpu.upload_bytes(&bytes).unwrap();
        let x = random.floats(cols, 2.0);
        let input = upload(&mut gpu, &x);
        let output = gpu.buffer((rows * 4) as u64).unwrap();
        for q8 in [true, false] {
            gpu.configure_decode(q8).unwrap();
            gpu.linear(&matrix, kind, rows as u64, cols as u64, 0, input.base(), 0, 1, output.base(), 0, &scratch).unwrap();
            let got = read(&mut gpu, &output, rows);
            let expected: Vec<(f64, f64)> = match (&parts, q8) {
                (Some((scaled, mins)), true) => dots(scaled, cols, &quantized(&x)).iter().zip(dots(mins, cols, &x))
                    .map(|(&(a, sa), (b, sb))| (a + b, sa + sb)).collect(),
                _ => dots(&weights, cols, &if q8 { quantized(&x) } else { x.clone() }),
            };
            for (row, (&value, &(sum, size))) in got.iter().zip(&expected).enumerate() {
                assert!((f64::from(value) - sum).abs() <= 2e-6 * size + 1e-6,
                        "kind {kind} {rows} x {cols}, Q8 {q8}, row {row}: {value} against {sum} (products {size})");
            }
        }
        gpu.configure_decode(false).unwrap();
    }
}

#[test]
fn a_q8_gemv_of_several_vectors_gives_each_one_alones_bits() {
    // (a verify pass's rows must be those of the tokens fed one at a time)
    straydog_engine::gpu::load_iq_tables().unwrap();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x2f0b_7f1e_33aa_91c5);
    let scratch = gpu.buffer(1 << 20).unwrap();
    gpu.configure_decode(true).unwrap();
    for (kind, rows, cols) in [(13u32, 1280usize, 5120usize), (8, 1792, 5120), (13, 300, 4096)] {
        let matrix = gpu.upload_bytes(&random.blocks(kind, rows * cols)).unwrap();
        let x = random.floats(4 * cols, 2.0);
        let input = upload(&mut gpu, &x);
        let output = gpu.buffer((4 * rows * 4) as u64).unwrap();
        let mut alone = Vec::new();
        for v in 0..4 {
            gpu.linear(&matrix, kind, rows as u64, cols as u64, 0, input.at(v * cols), 0, 1, output.base(), 0, &scratch).unwrap();
            alone.push(read(&mut gpu, &output, rows).iter().map(|value| value.to_bits()).collect::<Vec<_>>());
        }
        for vectors in 2..=4usize {
            gpu.linear(&matrix, kind, rows as u64, cols as u64, 0, input.base(), cols as u64, vectors as u64, output.base(), rows as u64,
                       &scratch).unwrap();
            let got = read(&mut gpu, &output, vectors * rows);
            for v in 0..vectors {
                let bits: Vec<u32> = got[v * rows..(v + 1) * rows].iter().map(|value| value.to_bits()).collect();
                assert_eq!(bits, alone[v], "kind {kind} {rows} x {cols}: vector {v} of {vectors}");
            }
        }
    }
}

#[test]
fn q8_expert_gate_up_matches_the_quantized_input_on_the_host() {
    straydog_engine::gpu::load_iq_tables().unwrap();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x5be0_cd19_137e_2179);
    let (hidden, ffn, routes, clamp) = (2048usize, 512usize, 2usize, 10.0f32);
    for kind in [16u32, 17, 22] {
        // two experts, each its gate, up and down (down unused: the gate/up stage alone)
        let gate_bytes = random.blocks(kind, ffn * hidden).len();
        let down_bytes = random.blocks(16, hidden * ffn).len();
        let slot = (2 * gate_bytes + down_bytes).next_multiple_of(256);
        let mut bank = Vec::new();
        let mut matrices = Vec::new();
        for _ in 0..routes {
            let start = bank.len();
            let (gate, up) = (random.blocks(kind, ffn * hidden), random.blocks(kind, ffn * hidden));
            matrices.push((decoded(kind, &gate), decoded(kind, &up)));
            bank.extend(gate);
            bank.extend(up);
            bank.extend(random.blocks(16, hidden * ffn));
            bank.resize(start + slot, 0);
        }
        let bank = gpu.upload_bytes(&bank).unwrap();
        let bases: Vec<u64> = (0..routes).map(|route| bank.base() as u64 + (route * slot) as u64).collect();
        let x = random.floats(hidden, 1.5);
        let route_weights = [0.7f32, 0.3];
        let (input, weights) = (upload(&mut gpu, &x), upload(&mut gpu, &route_weights));
        let product = gpu.buffer((routes * ffn * 4) as u64).unwrap();
        let out = gpu.buffer((hidden * 4) as u64).unwrap();
        let split = ExpertSplit { ffn_first: 0, ffn_rows: ffn as u64, hidden_first: 0, hidden_rows: hidden as u64, stages: 1, send: 0, more: 0,
                                  send_at: 0 };
        for q8 in [true, false] {
            gpu.configure_decode(q8).unwrap();
            gpu.experts(Some(&bases), slot as u64, [0, gate_bytes as u64, 2 * gate_bytes as u64], [kind, kind, 16], hidden as u64,
                        ffn as u64, 1, routes as u64, input.base(), 0, Some(weights.base()), clamp, false, product.base(), out.base(), 0,
                        Some(split)).unwrap();
            let got = read(&mut gpu, &product, routes * ffn);
            let v = if q8 { quantized(&x) } else { x.clone() };
            for (route, (gate, up)) in matrices.iter().enumerate() {
                for (row, (&(g, gs), &(u, us))) in dots(gate, hidden, &v).iter().zip(&dots(up, hidden, &v)).enumerate() {
                    let (g, u) = (g.min(f64::from(clamp)), u.clamp(-f64::from(clamp), f64::from(clamp)));
                    let expected = g / (1.0 + (-g).exp()) * u * f64::from(route_weights[route]);
                    // (the rounding of the dots, carried through SwiGLU: at most the product's slope times it)
                    let tolerance = 2e-6 * (gs * (u.abs() * 1.2) + us * g.abs().max(1.0)) * f64::from(route_weights[route]) + 1e-6;
                    let value = f64::from(got[route * ffn + row]);
                    assert!((value - expected).abs() <= tolerance, "kind {kind}, Q8 {q8}, route {route} row {row}: {value} against {expected}");
                }
            }
        }
        gpu.configure_decode(false).unwrap();
    }
}
