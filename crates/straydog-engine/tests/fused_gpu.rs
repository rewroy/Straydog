//! Fused decode ops against the ops they stand in for, on random data:
//! - sd_op_delta_step against sd_op_conv_silu, sd_op_delta_rule and sd_op_gated_norm (Qwen3.6's
//!   DeltaNet shape: 8 key heads, 16 value heads of 128, a 4-tap convolution): the same history,
//!   states and outputs within F32 rounding;
//! - sd_op_norm_linears over Q8_0 and F32 rows in one launch against sd_op_rms_norm and a
//!   sd_op_linears launch per format: the same bits;
//! - sd_op_norm_linears_rope (DeepSeek-V4.1's decode q_a norm and q_b on one of two GPUs, with
//!   the kv row's norm and rope into a ring row beside them) against sd_op_rms_norm, sd_op_linears
//!   and sd_op_norm_rope: the same bits;
//! - sd_op_index_select (DeepSeek-V4.1's indexer: 32 heads of 128, 64 rotated, top 512 of a few
//!   thousand groups, one row and four) against sd_op_rope, sd_op_index_scores and sd_op_top_k: the
//!   same scores and indices, run twice (its row counters start over);
//! - sd_op_index_select_proj (one row, the indexer's projection of x in the same launch, BF16 and
//!   Q8_0 rows) against sd_op_linear and sd_op_index_select: the same weights, scores and indices;
//! - sd_op_attention_prepare against sd_op_split, sd_op_rms_norm, sd_op_rope_halves and
//!   sd_op_scatter (Qwen3.6's attention shape on one of two GPUs: 8 query heads, a KV head of 256,
//!   64 rotated dimensions): the same bits.
//! - shapes whose sizes overflow, or whose launch would not fit, are refused;
//! - sd_op_swiglu_linear_route against sd_op_swiglu_linear and sd_op_route (Qwen3.6's shared expert
//!   down on one of two GPUs, 256 experts, top 8): the same bits.
//!
//! cargo test --release -p straydog-engine --features hip --test fused_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::Backend;
use straydog_engine::ops::{Buffer, Linear, NormRope};

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

#[test]
fn norm_linears_with_a_rope_row_match_the_separate_ops() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0xbb67_ae85_84ca_a73b);
    let (width, rows, dim, rope_dims, ring_rows, position, ring_row) = (1024usize, 4096usize, 512usize, 64usize, 8usize, 70_001i32, 5i32);
    let x = upload(&mut gpu, &random.floats(width, 3.0));
    let weight = upload(&mut gpu, &random.floats(width, 1.0));
    let mut bytes = Vec::with_capacity(rows * width / 32 * 34);
    for _ in 0..rows * width / 32 {
        let bits = ((random.next() & 1) << 15) | ((8 + random.next() % 5) << 10) | (random.next() & 0x3ff);
        bytes.extend_from_slice(&(bits as u16).to_le_bytes());
        for _ in 0..32 {
            bytes.push(random.next() as u8);
        }
    }
    let matrix = gpu.upload_bytes(&bytes).unwrap();
    let kv = upload(&mut gpu, &random.floats(dim, 3.0));
    let kv_weight = upload(&mut gpu, &random.floats(dim, 1.0));
    let frequencies: Vec<f32> = (0..rope_dims / 2).map(|i| 1e4f32.powf(-(2.0 * i as f32) / rope_dims as f32)).collect();
    let frequencies = upload(&mut gpu, &frequencies);
    let positions = gpu.buffer(4).unwrap();
    gpu.write_buffer(&positions, 0, &[position]).unwrap();
    let ring_index = gpu.buffer(4).unwrap();
    gpu.write_buffer(&ring_index, 0, &[ring_row]).unwrap();
    let ring = random.floats(ring_rows * dim, 1.0);
    let scratch = gpu.buffer(64 << 20).unwrap();
    let linear = |normed: &Buffer, out: &Buffer| Linear {
        weights: matrix, kind: 8, rows: rows as u64, cols: width as u64, first_row: 0, input: normed.base(), output: out.base(),
    };
    // the separate ops
    let (normed_1, out_1, ring_1) = (gpu.buffer((width * 4) as u64).unwrap(), gpu.buffer((rows * 4) as u64).unwrap(), upload(&mut gpu, &ring));
    gpu.rms_norm(x.base(), &weight, 1, width as u64, 1e-6, normed_1.base()).unwrap();
    gpu.linears(&[linear(&normed_1, &out_1)], 1, 0, 0, &scratch).unwrap();
    gpu.norm_rope(kv.base(), &kv_weight, 1, dim as u64, 1e-6, Some((rope_dims as u64, &positions, &frequencies)), Some(ring_index.base()),
                  ring_1.base(), ring_rows as u64).unwrap();
    // one launch
    let (normed_2, out_2, ring_2) = (gpu.buffer((width * 4) as u64).unwrap(), gpu.buffer((rows * 4) as u64).unwrap(), upload(&mut gpu, &ring));
    let side = NormRope {
        input: kv.base(), weight: &kv_weight, width: dim as u64, eps: 1e-6, rope: Some((rope_dims as u64, &positions, &frequencies)),
        out_rows: Some(ring_index.base()), out: ring_2.base(), out_count: ring_rows as u64,
    };
    gpu.norm_linears_rope(x.base(), &weight, width as u64, 1e-6, normed_2.base(), &[linear(&normed_2, &out_2)], &side).unwrap();
    assert_eq!(read(&mut gpu, &normed_1, width), read(&mut gpu, &normed_2, width), "the norm");
    assert_eq!(read(&mut gpu, &out_1, rows), read(&mut gpu, &out_2, rows), "the linear");
    assert_eq!(read(&mut gpu, &ring_1, ring_rows * dim), read(&mut gpu, &ring_2, ring_rows * dim), "the ring");
    // a side row outside its ring, or a rope wider than its row, is refused
    let wide = NormRope { rope: Some((2 * dim as u64 + 2, &positions, &frequencies)), ..side };
    assert!(gpu.norm_linears_rope(x.base(), &weight, width as u64, 1e-6, normed_2.base(), &[linear(&normed_2, &out_2)], &wide).is_err());
    let outside = NormRope { out_count: u64::MAX, ..side };
    assert!(gpu.norm_linears_rope(x.base(), &weight, width as u64, 1e-6, normed_2.base(), &[linear(&normed_2, &out_2)], &outside).is_err());
}

#[test]
fn a_prepared_attention_step_matches_its_ops() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x6a09_e667_f3bc_c908);
    let (rows, heads, kv_heads, dim, rope_dims, context) = (3usize, 8usize, 1usize, 256usize, 64usize, 16usize);
    let kv_width = kv_heads * dim;
    let q_full = upload(&mut gpu, &random.floats(rows * heads * 2 * dim, 4.0));
    let k = random.floats(rows * kv_width, 4.0);
    let v = upload(&mut gpu, &random.floats(rows * kv_width, 4.0));
    let (q_weight, k_weight) = (upload(&mut gpu, &random.floats(dim, 2.0)), upload(&mut gpu, &random.floats(dim, 2.0)));
    let frequencies: Vec<f32> = (0..rope_dims / 2).map(|i| 1e7f32.powf(-(2.0 * i as f32) / rope_dims as f32)).collect();
    let frequencies = upload(&mut gpu, &frequencies);
    // positions out of order, and the rotation's angles large
    let positions = [9i32, 2, 13];
    let position_buffer = gpu.buffer((rows * 4) as u64).unwrap();
    gpu.write_buffer(&position_buffer, 0, &positions).unwrap();
    let cache = random.floats(context * kv_width, 1.0);
    let eps = 1e-6;
    let outputs = |gpu: &mut Backend| -> [Buffer; 5] {
        [gpu.buffer((rows * heads * dim * 4) as u64).unwrap(), gpu.buffer((rows * heads * dim * 4) as u64).unwrap(), upload(gpu, &k),
         upload(gpu, &cache), upload(gpu, &cache)]
    };
    // the ops
    let [q_1, gate_1, k_1, keys_1, values_1] = outputs(&mut gpu);
    gpu.split(q_full.base(), (rows * heads) as u64, dim as u64, q_1.base(), gate_1.base()).unwrap();
    gpu.rms_norm(q_1.base(), &q_weight, (rows * heads) as u64, dim as u64, eps, q_1.base()).unwrap();
    gpu.rms_norm(k_1.base(), &k_weight, (rows * kv_heads) as u64, dim as u64, eps, k_1.base()).unwrap();
    gpu.rope_halves(q_1.base(), rows as u64, heads as u64, dim as u64, rope_dims as u64, &position_buffer, &frequencies).unwrap();
    gpu.rope_halves(k_1.base(), rows as u64, kv_heads as u64, dim as u64, rope_dims as u64, &position_buffer, &frequencies).unwrap();
    gpu.scatter(k_1.base(), position_buffer.base(), rows as u64, kv_width as u64, keys_1.base(), context as u64).unwrap();
    gpu.scatter(v.base(), position_buffer.base(), rows as u64, kv_width as u64, values_1.base(), context as u64).unwrap();
    // one launch
    let [q_2, gate_2, k_2, keys_2, values_2] = outputs(&mut gpu);
    gpu.attention_prepare(q_full.base(), k_2.base(), v.base(), rows as u64, heads as u64, kv_heads as u64, dim as u64, &q_weight, &k_weight,
                          eps, rope_dims as u64, &position_buffer, &frequencies, q_2.base(), gate_2.base(), keys_2.base(), values_2.base(),
                          context as u64).unwrap();
    for (label, first, second, count) in [("queries", &q_1, &q_2, rows * heads * dim), ("gates", &gate_1, &gate_2, rows * heads * dim),
                                          ("keys", &k_1, &k_2, rows * kv_width), ("key cache", &keys_1, &keys_2, context * kv_width),
                                          ("value cache", &values_1, &values_2, context * kv_width)] {
        assert_eq!(read(&mut gpu, first, count), read(&mut gpu, second, count), "{label}");
    }
}

#[test]
fn routes_picked_beside_the_shared_expert_match_the_route_op() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0xbb67_ae85_84ca_a73b);
    let (rows, cols, experts, k) = (2048usize, 256usize, 256usize, 8usize);
    // Q8_0 rows (34-byte blocks: an FP16 scale between 2^-7 and 2^-3, then 32 quants)
    let mut bytes = Vec::with_capacity(rows * cols / 32 * 34);
    for _ in 0..rows * cols / 32 {
        let bits = ((random.next() & 1) << 15) | ((8 + random.next() % 5) << 10) | (random.next() & 0x3ff);
        bytes.extend_from_slice(&(bits as u16).to_le_bytes());
        for _ in 0..32 {
            bytes.push(random.next() as u8);
        }
    }
    let down = gpu.upload_bytes(&bytes).unwrap();
    let (gate, up) = (upload(&mut gpu, &random.floats(cols, 3.0)), upload(&mut gpu, &random.floats(cols, 3.0)));
    let scale = upload(&mut gpu, &random.floats(1, 2.0));
    let logits = upload(&mut gpu, &random.floats(experts, 4.0));
    let buffers = |gpu: &mut Backend| -> [Buffer; 4] {
        [gpu.buffer((cols * 4) as u64).unwrap(), gpu.buffer((rows * 4) as u64).unwrap(), gpu.buffer((k * 4) as u64).unwrap(),
         gpu.buffer((k * 4) as u64).unwrap()]
    };
    let [hidden_1, out_1, ids_1, weights_1] = buffers(&mut gpu);
    gpu.swiglu_linear(&down, 8, rows as u64, cols as u64, gate.base(), up.base(), Some(scale.base()), hidden_1.base(), out_1.base()).unwrap();
    gpu.route(logits.base(), None, 1, experts as u64, k as u64, 0, true, 1.0, ids_1.base(), weights_1.base()).unwrap();
    let [hidden_2, out_2, ids_2, weights_2] = buffers(&mut gpu);
    gpu.swiglu_linear_route(&down, 8, rows as u64, cols as u64, gate.base(), up.base(), Some(scale.base()), hidden_2.base(), out_2.base(),
                            logits.base(), k as u64, ids_2.base(), weights_2.base()).unwrap();
    assert_eq!(read(&mut gpu, &out_1, rows), read(&mut gpu, &out_2, rows), "outputs");
    assert_eq!(read(&mut gpu, &ids_1, k), read(&mut gpu, &ids_2, k), "routes");
    assert_eq!(read(&mut gpu, &weights_1, k), read(&mut gpu, &weights_2, k), "weights");
}

#[test]
fn shapes_that_overflow_are_refused() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let buffer = gpu.buffer(1 << 20).unwrap();
    let floats = upload(&mut gpu, &[0.0; 256]);
    let huge = 1u64 << 62;
    // counts whose products wrap past the address space
    assert!(gpu.add(buffer.base(), huge, buffer.base()).is_err());
    assert!(gpu.split(buffer.base(), huge, 8, buffer.base(), buffer.base()).is_err());
    assert!(gpu.gated_norm(buffer.base(), buffer.base(), &floats, huge, 1 << 4, 1e-6, buffer.base()).is_err());
    assert!(gpu.rope_halves(buffer.base(), huge, 4, 8, 4, &floats, &floats).is_err());
    assert!(gpu.scale_rows(buffer.base(), buffer.base(), 4, huge).is_err());
    // heads beyond a launch's dimensions, a context whose cache size wraps
    assert!(gpu.attention_prepare(buffer.base(), buffer.base(), buffer.base(), 1, 1 << 40, 1, 64, &floats, &floats, 1e-6, 0, &floats,
                                  &floats, buffer.base(), buffer.base(), buffer.base(), buffer.base(), 16).is_err());
    assert!(gpu.attention_prepare(buffer.base(), buffer.base(), buffer.base(), 1, 1, 1 << 40, 64, &floats, &floats, 1e-6, 0, &floats,
                                  &floats, buffer.base(), buffer.base(), buffer.base(), buffer.base(), 16).is_err());
    assert!(gpu.attention_prepare(buffer.base(), buffer.base(), buffer.base(), 1, 1, 1, 64, &floats, &floats, 1e-6, 0, &floats, &floats,
                                  buffer.base(), buffer.base(), buffer.base(), buffer.base(), huge).is_err());
    assert!(gpu.swiglu_linear(&buffer, 8, huge, 256, buffer.base(), buffer.base(), None, buffer.base(), buffer.base()).is_err());
}

#[test]
fn an_index_select_matches_rope_scores_and_top_k() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0xa54f_f53a_5f1d_36f1);
    let (heads, dim, rope_dims, k) = (32usize, 128usize, 64usize, 512usize);
    let frequencies: Vec<f32> = (0..rope_dims / 2).map(|i| 1.6e5f32.powf(-(2.0 * i as f32) / rope_dims as f32)).collect();
    let frequencies = upload(&mut gpu, &frequencies);
    for (rows, groups) in [(1usize, 3000usize), (4, 2000), (1, 300)] {
        let q = random.floats(rows * heads * dim, 2.0);
        let keys = upload(&mut gpu, &random.floats(groups * dim, 1.0));
        let weights = upload(&mut gpu, &random.floats(rows * heads, 1.0));
        let visible: Vec<i32> = (0..rows).map(|row| (groups - 3 * row) as i32).collect();
        let (positions, visible) = (gpu.buffer((rows * 4) as u64).unwrap(), {
            let buffer = gpu.buffer((rows * 4) as u64).unwrap();
            gpu.write_buffer(&buffer, 0, &visible).unwrap();
            buffer
        });
        let at: Vec<i32> = (0..rows).map(|row| 20_000 + 7 * row as i32).collect();
        gpu.write_buffer(&positions, 0, &at).unwrap();
        let scale = 1.0 / ((dim as f32).sqrt() * (heads as f32).sqrt());
        // apart
        let (q_1, scores_1, out_1) = (upload(&mut gpu, &q), gpu.buffer((rows * groups * 4) as u64).unwrap(), gpu.buffer((rows * k * 4) as u64).unwrap());
        gpu.rope(q_1.base(), rows as u64, heads as u64, dim as u64, rope_dims as u64, &positions, &frequencies, false).unwrap();
        gpu.index_scores(q_1.base(), keys.base(), weights.base(), visible.base(), rows as u64, groups as u64, heads as u64, dim as u64,
                         scale, scores_1.base()).unwrap();
        gpu.top_k(scores_1.base(), rows as u64, groups as u64, k as u64, out_1.base()).unwrap();
        // one launch, twice
        let q_2 = upload(&mut gpu, &q);
        for run in 0..2 {
            let (scores_2, out_2) = (gpu.buffer((rows * groups * 4) as u64).unwrap(), gpu.buffer((rows * k * 4) as u64).unwrap());
            gpu.index_select(q_2.base(), keys.base(), weights.base(), visible.base(), rows as u64, groups as u64, heads as u64, dim as u64,
                             scale, (rope_dims as u64, &positions, &frequencies), scores_2.base(), k as u64, out_2.base()).unwrap();
            assert_eq!(read(&mut gpu, &scores_1, rows * groups), read(&mut gpu, &scores_2, rows * groups), "{rows} rows, run {run}: scores");
            assert_eq!(read(&mut gpu, &out_1, rows * k), read(&mut gpu, &out_2, rows * k), "{rows} rows, run {run}: indices");
        }
        assert_eq!(read(&mut gpu, &q_2, rows * heads * dim), q.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), "q unchanged");
    }
}

#[test]
fn an_index_select_with_its_projection_matches_the_two_ops() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x510e_527f_ade6_82d1);
    let (heads, dim, rope_dims, k, width, groups) = (32usize, 128usize, 64usize, 512usize, 5120usize, 2500usize);
    let frequencies: Vec<f32> = (0..rope_dims / 2).map(|i| 1.6e5f32.powf(-(2.0 * i as f32) / rope_dims as f32)).collect();
    let frequencies = upload(&mut gpu, &frequencies);
    let x = upload(&mut gpu, &random.floats(width, 2.0));
    let q = upload(&mut gpu, &random.floats(heads * dim, 2.0));
    let keys = upload(&mut gpu, &random.floats(groups * dim, 1.0));
    let (positions, visible) = (gpu.buffer(4).unwrap(), gpu.buffer(4).unwrap());
    gpu.write_buffer(&positions, 0, &[31_337i32]).unwrap();
    gpu.write_buffer(&visible, 0, &[(groups - 5) as i32]).unwrap();
    let scale = 1.0 / ((dim as f32).sqrt() * (heads as f32).sqrt());
    let scratch = gpu.buffer(64 << 20).unwrap();
    // BF16 rows (the fused launch), and Q8_0 rows (a GEMV, then the scores)
    let bf16: Vec<u8> = random.floats(heads * width, 0.05).iter().flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes()).collect();
    let mut q8 = Vec::with_capacity(heads * width / 32 * 34);
    for _ in 0..heads * width / 32 {
        let bits = ((random.next() & 1) << 15) | ((8 + random.next() % 5) << 10) | (random.next() & 0x3ff);
        q8.extend_from_slice(&(bits as u16).to_le_bytes());
        for _ in 0..32 {
            q8.push(random.next() as u8);
        }
    }
    for (kind, bytes) in [(30u32, bf16), (8, q8)] {
        let matrix = gpu.upload_bytes(&bytes).unwrap();
        let fresh = |gpu: &mut Backend| (gpu.buffer((heads * 4) as u64).unwrap(), gpu.buffer((groups * 4) as u64).unwrap(),
                                         gpu.buffer((k * 4) as u64).unwrap());
        let (weights_1, scores_1, out_1) = fresh(&mut gpu);
        gpu.linear(&matrix, kind, heads as u64, width as u64, 0, x.base(), 0, 1, weights_1.base(), 0, &scratch).unwrap();
        gpu.index_select(q.base(), keys.base(), weights_1.base(), visible.base(), 1, groups as u64, heads as u64, dim as u64, scale,
                         (rope_dims as u64, &positions, &frequencies), scores_1.base(), k as u64, out_1.base()).unwrap();
        for run in 0..2 {
            let (weights_2, scores_2, out_2) = fresh(&mut gpu);
            gpu.index_select_proj((&matrix, kind, width as u64), x.base(), q.base(), keys.base(), weights_2.base(), visible.base(),
                                  groups as u64, heads as u64, dim as u64, scale, (rope_dims as u64, &positions, &frequencies),
                                  scores_2.base(), k as u64, out_2.base()).unwrap();
            assert_eq!(read(&mut gpu, &weights_1, heads), read(&mut gpu, &weights_2, heads), "kind {kind}, run {run}: weights");
            assert_eq!(read(&mut gpu, &scores_1, groups), read(&mut gpu, &scores_2, groups), "kind {kind}, run {run}: scores");
            assert_eq!(read(&mut gpu, &out_1, k), read(&mut gpu, &out_2, k), "kind {kind}, run {run}: indices");
        }
    }
}
