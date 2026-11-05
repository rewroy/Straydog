//! The Q8 decode kernels (Backend::configure_decode) against a host reference on random data: a
//! decode row's input quantized per 32 values to INT8 (round(x * 127 / max |x|), the scale
//! max |x| / 127), then the rows' dots with the host-decoded weights in F64:
//! - Q5_K and Q8_0 GEMVs (DeepSeek-V4.1's decode projection widths, and rows that leave a wave
//!   part empty): within F32 rounding of that reference (a Q5_K group's min term takes the input's
//!   own sum, as the kernel does);
//! - a GEMV of two to four vectors (a verify pass's rows): each vector's result the bits of its own
//!   one-vector GEMV;
//! - the IQ2_XXS, IQ2_XS, IQ2_S and Q5_K experts' gate/up (two routes, SwiGLU clamped, times the routes'
//!   weights): the same;
//! - the IQ2_XXS, IQ2_XS, IQ2_S and Q5_K experts' down (three routes of three tokens' product rows):
//!   the same, and each token's row the bits of the token fed alone; so IQ1_M's, which the F32 kernels
//!   take whatever the decode mode;
//! - a GEMV's row is the bits it gets in a launch of few rows, whatever the launch holds (q_b's 16384 rows
//!   take several to a wave, a smaller launch one);
//! - with the Q8 kernels off, the F32 kernels' results are the unquantized input's within F32
//!   rounding, as before.
//!
//! cargo test --release -p straydog-engine --features hip --test q8_decode_gpu, under the GPU lock.
// (RDNA4's kernels: kernels.cpp takes them only on an RDNA4 device, and a CUDA build runs these ops in F32, with nothing of theirs to compare)
#![cfg(feature = "hip")]

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
            29 => (256, 56),
            _ => panic!("no format {kind}"),
        };
        let mut out: Vec<u8> = (0..values / per * bytes).map(|_| self.next() as u8).collect();
        for block in out.chunks_exact_mut(bytes) {
            if kind == 29 {
                // (IQ1_M's FP16 scale is spread over the top nibbles of four of its scale words)
                block[49] &= 0x0f;
                block[51] &= 0x0f;
                block[53] = block[53] & 0x0f | 0xc0;
                block[55] = block[55] & 0x0f | 0x20;
                continue;
            }
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

/// Each row's dot with `x` as the kernels take it: the F32 kernels' with the input itself, the Q8 kernels' with the quantized input (Q5_K's mins against the input itself), and
/// the sum of the products' magnitudes.
fn expected_dots(kind: u32, bytes: &[u8], cols: usize, x: &[f32], q8: bool) -> Vec<(f64, f64)> {
    if !q8 {
        return dots(&decoded(kind, bytes), cols, x);
    }
    if kind != 13 {
        return dots(&decoded(kind, bytes), cols, &quantized(x));
    }
    // Q5_K's values d * scale * q - dmin * min, apart: the first part (dmin zeroed) meets the quantized input, the second (d zeroed) the input
    let zeroed = |at: usize| {
        let mut copy = bytes.to_vec();
        for block in copy.chunks_exact_mut(176) { block[at..at + 2].copy_from_slice(&[0, 0]); }
        decoded(kind, &copy)
    };
    dots(&zeroed(2), cols, &quantized(x)).iter().zip(dots(&zeroed(0), cols, x)).map(|(&(a, sa), (b, sb))| (a + b, sa + sb)).collect()
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
    // (q_b's 16384 rows: four a wave for one or two vectors, eight for three or four; and a count that leaves the last waves part empty)
    for (kind, rows, cols) in [(13u32, 1280usize, 5120usize), (8, 1792, 5120), (13, 300, 4096), (13, 2048, 1280), (13, 16384, 1280), (13, 16301, 1280)] {
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
fn a_q8_gemv_row_is_the_same_whatever_launch_it_is_in() {
    // (q_b's 16384 rows meet the kernel that takes several rows to a wave; a launch of a few of them meets the one that takes one)
    straydog_engine::gpu::load_iq_tables().unwrap();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x6a09_e667_f3bc_c908);
    let scratch = gpu.buffer(1 << 20).unwrap();
    gpu.configure_decode(true).unwrap();
    let (rows, cols, chunk) = (16384usize, 1280usize, 1024usize);
    let matrix = gpu.upload_bytes(&random.blocks(13, rows * cols)).unwrap();
    let input = upload(&mut gpu, &random.floats(4 * cols, 2.0));
    let (whole, parts) = (gpu.buffer((4 * rows * 4) as u64).unwrap(), gpu.buffer((4 * rows * 4) as u64).unwrap());
    for vectors in 1..=4usize {
        gpu.linear(&matrix, 13, rows as u64, cols as u64, 0, input.base(), cols as u64, vectors as u64, whole.base(), rows as u64, &scratch).unwrap();
        for first in (0..rows).step_by(chunk) {
            gpu.linear(&matrix, 13, chunk as u64, cols as u64, first as u64, input.base(), cols as u64, vectors as u64, parts.at(first), rows as u64,
                       &scratch).unwrap();
        }
        let (whole, parts) = (read(&mut gpu, &whole, vectors * rows), read(&mut gpu, &parts, vectors * rows));
        for vector in 0..vectors {
            for row in 0..rows {
                let at = vector * rows + row;
                assert_eq!(whole[at].to_bits(), parts[at].to_bits(), "{vectors} vectors: vector {vector} row {row}: {} in 16384 rows, {} in {chunk}", whole[at], parts[at]);
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
    for kind in [16u32, 17, 22, 13] {
        // two experts, each its gate, up and down (down unused: the gate/up stage alone)
        let gate_bytes = random.blocks(kind, ffn * hidden).len();
        let down_bytes = random.blocks(16, hidden * ffn).len();
        let slot = (2 * gate_bytes + down_bytes).next_multiple_of(256);
        let mut bank = Vec::new();
        let mut matrices = Vec::new();
        for _ in 0..routes {
            let start = bank.len();
            let (gate, up) = (random.blocks(kind, ffn * hidden), random.blocks(kind, ffn * hidden));
            matrices.push((gate.clone(), up.clone()));
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
            for (route, (gate, up)) in matrices.iter().enumerate() {
                let (gate, up) = (expected_dots(kind, gate, hidden, &x, q8), expected_dots(kind, up, hidden, &x, q8));
                for (row, (&(g, gs), &(u, us))) in gate.iter().zip(&up).enumerate() {
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

#[test]
fn q8_expert_down_matches_the_quantized_product_on_the_host() {
    straydog_engine::gpu::load_iq_tables().unwrap();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x3c6e_f372_fe94_f82b);
    let (hidden, ffn, routes, tokens, experts) = (1536usize, 768usize, 3usize, 3usize, 4usize);
    // (IQ2_S's and IQ1_M's downs stay on the F32 kernels)
    for kind in [16u32, 17, 22, 13, 29] {
        // an expert's slot: gate and up (IQ2_XXS, unused: the down stage alone), then its down matrix
        let gate_bytes = random.blocks(16, ffn * hidden).len();
        let down_bytes = random.blocks(kind, hidden * ffn).len();
        let slot = (2 * gate_bytes + down_bytes).next_multiple_of(256);
        let (mut bank, mut downs) = (Vec::new(), Vec::new());
        for _ in 0..experts {
            let start = bank.len();
            bank.extend(random.blocks(16, 2 * ffn * hidden));
            let down = random.blocks(kind, hidden * ffn);
            bank.extend(&down);
            downs.push(down);
            bank.resize(start + slot, 0);
        }
        let bank = gpu.upload_bytes(&bank).unwrap();
        let expert_of = |token: usize, route: usize| (token + 2 * route) % experts;
        let bases: Vec<u64> = (0..tokens * routes).map(|r| bank.base() as u64 + (expert_of(r / routes, r % routes) * slot) as u64).collect();
        let products = random.floats(tokens * routes * ffn, 3.0);
        let (product, input) = (upload(&mut gpu, &products), upload(&mut gpu, &random.floats(tokens * hidden, 1.0)));
        let out = gpu.buffer((tokens * hidden * 4) as u64).unwrap();
        let split = ExpertSplit { ffn_first: 0, ffn_rows: ffn as u64, hidden_first: 0, hidden_rows: hidden as u64, stages: 2, send: 0, more: 0,
                                  send_at: 0 };
        let offsets = [0, gate_bytes as u64, 2 * gate_bytes as u64];
        for q8 in [true, false] {
            gpu.configure_decode(q8).unwrap();
            gpu.experts(Some(&bases), slot as u64, offsets, [16, 16, kind], hidden as u64, ffn as u64, tokens as u64, routes as u64, input.base(),
                        0, None, 10.0, false, product.base(), out.base(), 0, Some(split)).unwrap();
            let got = read(&mut gpu, &out, tokens * hidden);
            // (the F32 kernels, and IQ2_S's with the Q8 ones on, take the product itself)
            let quantize = q8 && kind != 22 && kind != 29;
            for token in 0..tokens {
                let mut sums = vec![(0.0f64, 0.0f64); hidden];
                for route in 0..routes {
                    let row = &products[(token * routes + route) * ffn..][..ffn];
                    for (sum, (value, size)) in sums.iter_mut().zip(expected_dots(kind, &downs[expert_of(token, route)], ffn, row, quantize)) {
                        *sum = (sum.0 + value, sum.1 + size);
                    }
                }
                for (row, (&(sum, size), &value)) in sums.iter().zip(&got[token * hidden..(token + 1) * hidden]).enumerate() {
                    assert!((f64::from(value) - sum).abs() <= 2e-6 * size + 1e-6,
                            "kind {kind}, Q8 {q8}, token {token} row {row}: {value} against {sum} (products {size})");
                }
            }
            // each token's row is the bits of the token fed alone (a verify pass's rows are those of the tokens one at a time)
            for token in 0..tokens {
                let (alone_product, alone_input) = (upload(&mut gpu, &products[token * routes * ffn..(token + 1) * routes * ffn]),
                                                    upload(&mut gpu, &random.floats(hidden, 1.0)));
                let alone_out = gpu.buffer((hidden * 4) as u64).unwrap();
                gpu.experts(Some(&bases[token * routes..(token + 1) * routes]), slot as u64, offsets, [16, 16, kind], hidden as u64, ffn as u64, 1,
                            routes as u64, alone_input.base(), 0, None, 10.0, false, alone_product.base(), alone_out.base(), 0, Some(split)).unwrap();
                let bits: Vec<u32> = read(&mut gpu, &alone_out, hidden).iter().map(|value| value.to_bits()).collect();
                let together: Vec<u32> = got[token * hidden..(token + 1) * hidden].iter().map(|value| value.to_bits()).collect();
                assert_eq!(together, bits, "kind {kind}, Q8 {q8}: token {token} of {tokens}");
            }
        }
        gpu.configure_decode(false).unwrap();
    }
}
