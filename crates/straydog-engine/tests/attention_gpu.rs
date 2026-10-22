//! Model-free causal GQA against an independent F64 softmax reference.
//! In particular, rows >= 16 and dim = 256 launch the tiled prefill kernel;
//! CUDA must opt in to its 50240 bytes of dynamic shared memory.
//! Run with --features hip or cuda, under the host's GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;

struct Random(u64);
impl Random {
    fn floats(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }).collect()
    }
}

fn upload(gpu: &mut Backend, values: &[f32]) -> Buffer {
    let buffer = gpu.buffer((values.len() * 4) as u64).unwrap();
    gpu.write_buffer(&buffer, 0, values).unwrap();
    buffer
}

fn gamma(operations: usize) -> f64 {
    let nu = operations as f64 * 2f64.powi(-24);
    nu / (1.0 - nu)
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    rows: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    context: usize,
}

// F64 arithmetic over exactly the uploaded F32 operands, including the F32 scale.
// A logit error <= e changes normalized positive weights by at most exp(2e)-1.
// Bound the F32 dot (even without FMA) by gamma(2*dim+1)*sum|q*k|*|scale|.
// For the weighted value/normalizer accumulation and online rescaling we budget
// gamma(8*length+32)*max|v|, plus 16 F32 unit roundoffs for exp/divide/sigmoid.
// That last allowance is a regression tolerance for device transcendental math,
// not a formal promise about every backend's exp implementation. Bounded inputs
// avoid overflow/underflow; no relative-to-output tolerance hides cancellation.
fn reference(shape: Shape, q: &[f32], keys: &[f32], values: &[f32], positions: &[i32], scale: f32,
             gate: Option<&[f32]>) -> Vec<(f64, f64)> {
    let Shape { rows, heads, kv_heads, dim, context } = shape;
    let mut result = vec![(0.0, 0.0); rows * heads * dim];
    for row in 0..rows {
        let length = (positions[row] as usize + 1).min(context);
        for head in 0..heads {
            let kh = head / (heads / kv_heads);
            let query = &q[(row * heads + head) * dim..][..dim];
            let mut error = 0f64;
            let scores: Vec<f64> = (0..length).map(|key| {
                let key = &keys[(key * kv_heads + kh) * dim..][..dim];
                let (sum, absolute) = query.iter().zip(key).fold((0.0, 0.0), |(sum, absolute), (&q, &k)| {
                    let product = f64::from(q) * f64::from(k);
                    (sum + product, absolute + product.abs())
                });
                error = error.max(gamma(2 * dim + 1) * absolute * f64::from(scale).abs());
                sum * f64::from(scale)
            }).collect();
            let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - top).exp()).collect();
            let total = weights.iter().sum::<f64>();
            for d in 0..dim {
                let (mut sum, mut largest) = (0.0, 0f64);
                for (key, weight) in weights.iter().enumerate() {
                    let value = f64::from(values[(key * kv_heads + kh) * dim + d]);
                    sum += weight * value;
                    largest = largest.max(value.abs());
                }
                let at = (row * heads + head) * dim + d;
                let gating = gate.map_or(1.0, |gate| 1.0 / (1.0 + (-f64::from(gate[at])).exp()));
                let bound = largest * ((2.0 * error).exp_m1() + gamma(8 * length + 32) + 16.0 * 2f64.powi(-24));
                result[at] = (sum / total * gating, bound);
            }
        }
    }
    result
}

fn check(shape: Shape, zero_queries: bool) {
    let Shape { rows, heads, kv_heads, dim, context } = shape;
    let mut random = Random(0x9e3779b97f4a7c15 ^ rows as u64 ^ (dim as u64) << 16);
    let mut q = random.floats(rows * heads * dim);
    if zero_queries { q.fill(0.0); }
    let keys = random.floats(context * kv_heads * dim);
    let values = random.floats(context * kv_heads * dim);
    // Nonmonotone, short/long rows in the same tile, tile boundaries, and a
    // position past context (the public API clips it to the cache).
    let special = [0, 1, 15, 16, context - 1, context + 7];
    let positions: Vec<i32> = (0..rows).map(|r| {
        if r < special.len() { special[r] as i32 } else { ((r * 17 + 3) % context) as i32 }
    }).collect();
    let gate: Vec<f32> = (0..q.len()).map(|i| [-12.0, -1.0, 0.0, 1.0, 12.0][i % 5]).collect();
    let scale = 1.0 / (dim as f32).sqrt();
    let mut gpu = Backend::new(0, 128 << 20).unwrap();
    let (qs, ks, vs, gs) = (upload(&mut gpu, &q), upload(&mut gpu, &keys), upload(&mut gpu, &values), upload(&mut gpu, &gate));
    let ps = gpu.buffer((rows * 4) as u64).unwrap();
    gpu.write_buffer(&ps, 0, &positions).unwrap();
    // 64 is the public header's maximum split count, sufficient for either path.
    let scratch = gpu.buffer((rows * heads * 64 * (dim + 2) * 4) as u64).unwrap();
    const GUARD: usize = 7;
    const CANARY: f32 = 12345.125;
    let output = upload(&mut gpu, &vec![CANARY; q.len() + 2 * GUARD]);
    for gated in [false, true] {
        gpu.attention(qs.base(), ks.base(), vs.base(), context as u64, ps.base(), rows as u64, heads as u64,
                      kv_heads as u64, dim as u64, scale, gated.then(|| gs.base().cast_const()), &scratch, output.at(GUARD))
            .unwrap_or_else(|error| panic!("{shape:?} gated={gated}: {error}"));
        let mut actual = vec![0f32; q.len() + 2 * GUARD];
        gpu.read_buffer(&output, 0, &mut actual).unwrap();
        assert!(actual[..GUARD].iter().chain(&actual[GUARD + q.len()..]).all(|&x| x == CANARY), "{shape:?}: output guard");
        let expected = reference(shape, &q, &keys, &values, &positions, scale, gated.then_some(gate.as_slice()));
        let mut worst = 0f64;
        for (i, (&got, &(wanted, bound))) in actual[GUARD..][..q.len()].iter().zip(&expected).enumerate() {
            let error = (f64::from(got) - wanted).abs();
            assert!(got.is_finite() && error <= bound, "{shape:?} zero_q={zero_queries} gated={gated} at={i}: {got} vs {wanted}, error={error:e} bound={bound:e}");
            worst = worst.max(error / bound);
        }
        eprintln!("{shape:?} zero_q={zero_queries} gated={gated}: all outputs within F64 bounds, max error/bound={worst:e}");
    }
}

#[test]
fn causal_gqa_prefill_and_split_match_f64_softmax() {
    // 15/16/17 cross the kernel dispatch boundary; 17/31 test partial row tiles.
    // Nonmultiples of 16/64 exercise the last key tile and unequal split tails.
    for (rows, heads, kv_heads, dim, context) in [
        (15, 4, 2, 256, 33), (16, 4, 2, 256, 33), (17, 4, 2, 256, 35),
        (31, 6, 2, 256, 65), (32, 4, 4, 256, 129), (16, 4, 2, 128, 35),
        (1, 2, 1, 256, 137), (17, 1, 1, 256, 1), (15, 16, 2, 256, 137),
    ] {
        check(Shape { rows, heads, kv_heads, dim, context }, false);
    }
    check(Shape { rows: 17, heads: 4, kv_heads: 2, dim: 256, context: 35 }, true);
}
