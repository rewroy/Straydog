//! Fused decode ops against the ops they stand in for, on random data:
//! - sd_op_delta_step against sd_op_conv_silu, sd_op_delta_rule and sd_op_gated_norm (Qwen3.6's
//!   DeltaNet shape: 8 key heads, 16 value heads of 128, a 4-tap convolution): the same history,
//!   states and outputs within F32 rounding;
//! - sd_op_norm_linears over Q8_0 and F32 rows in one launch against sd_op_rms_norm and a
//!   sd_op_linears launch per format: the same bits;
//! - sd_op_norm_linears_rope (DeepSeek-V4.1's decode q_a norm and q_b on one of two GPUs, with
//!   the kv row's norm and rope into a ring row beside them) against sd_op_rms_norm, sd_op_linears
//!   and sd_op_norm_rope: the same bits (Q8_0 rows on the F32 and on the Q8 decode kernels);
//! - sd_op_top_k against a reference of its rule (ties keep the lower index, -inf never kept, -1 for
//!   empty slots) on scores with many ties, 511 to 131072 groups, top 512: the same indices;
//! - sd_op_index_select (DeepSeek-V4.1's indexer: 32 heads of 128, 64 rotated, top 512 of a few
//!   thousand groups, one row and four) against sd_op_rope, sd_op_index_scores and sd_op_top_k: the
//!   same scores and indices, run twice (its row counters start over);
//! - sd_op_index_select_proj (one row, the indexer's projection of x in the same launch, BF16 and
//!   Q8_0 rows) against sd_op_linear and sd_op_index_select: the same weights, scores and indices;
//! - sd_op_index_candidates and sd_op_index_select(_proj) among them (DeepSeek-V4.1's two-level
//!   indexer: 2048 blocks of 8 of up to 23001 groups; one row and three, F32 and FP16 (WMMA), and the
//!   FP16 prompt pass's 32)
//!   against the reference's rule on the scores of every group: the same blocks and indices;
//! - sd_op_attention_prepare against sd_op_split, sd_op_rms_norm, sd_op_rope_halves and
//!   sd_op_scatter (Qwen3.6's attention shape on one of two GPUs: 8 query heads, a KV head of 256,
//!   64 rotated dimensions): the same bits.
//! - shapes whose sizes overflow, or whose launch would not fit, are refused;
//! - sd_op_swiglu_linear_route against sd_op_swiglu_linear and sd_op_route (Qwen3.6's shared expert
//!   down on one of two GPUs, 256 experts, top 8): the same bits.
//! - sd_op_linear_route over BF16 rows (DeepSeek-V4.1's router) against sd_op_linear and sd_op_route
//!   on its logits: the same routes, the logits within F32 rounding.
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

/// One test at a time on the device: a test that captures a graph (linears_norm_tail_graph_matches_separate_ops) fails under CUDA, and
/// fails the test beside it, when another thread of this process launches on the device while the capture is open (codes 901 and 900;
/// the capture mode is shared with HIP and stays as it is). As link_gpu does.
fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    DEVICE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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
    let _one_at_a_time = one_at_a_time();
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
    let _one_at_a_time = one_at_a_time();
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
    let _one_at_a_time = one_at_a_time();
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
    let _one_at_a_time = one_at_a_time();
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
    let _one_at_a_time = one_at_a_time();
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
    let _one_at_a_time = one_at_a_time();
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

/// top_k_row's order-preserving key (-0 ties +0).
fn top_k_key(value: f32) -> u32 {
    let bits = if value == 0.0 { 0.0f32.to_bits() } else { value.to_bits() };
    if bits & 0x8000_0000 != 0 { !bits } else { bits | 0x8000_0000 }
}

/// The k highest scores' indices in ascending index order (ties keep the lower index; -inf is never
/// kept), -1 for empty slots: what sd_op_top_k must give.
fn top_k_reference(scores: &[f32], k: usize) -> Vec<u32> {
    let floor = top_k_key(f32::NEG_INFINITY);
    let keys: Vec<u32> = scores.iter().map(|&score| top_k_key(score)).collect();
    let live: Vec<usize> = (0..scores.len()).filter(|&g| keys[g] > floor).collect();
    let mut out: Vec<i32> = if live.len() <= k {
        live.iter().map(|&g| g as i32).collect()
    } else {
        let mut sorted: Vec<u32> = live.iter().map(|&g| keys[g]).collect();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let threshold = sorted[k - 1];
        let mut need = k - live.iter().filter(|&&g| keys[g] > threshold).count();
        let mut kept = Vec::with_capacity(k);
        for &g in &live {
            if keys[g] > threshold {
                kept.push(g as i32);
            } else if keys[g] == threshold && need > 0 {
                kept.push(g as i32);
                need -= 1;
            }
        }
        kept
    };
    out.resize(k, -1);
    out.into_iter().map(|index| index as u32).collect()
}

#[test]
fn top_k_matches_its_reference_with_many_ties() {
    let _one_at_a_time = one_at_a_time();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x2f6b_91d3_c0a5_4e77);
    let k = 512usize;
    for groups in [511usize, 512, 513, 600, 4096, 30000, 65536, 131072] {
        let rows = 2usize;
        // quarter steps in [-2, 2] (many ties), -0 beside +0, some -inf (never kept) and a few +inf;
        // with 600 groups most are -inf, fewer than k finite
        let scores: Vec<f32> = (0..rows * groups).map(|i| {
            let draw = random.next();
            if groups == 600 && !draw.is_multiple_of(8) { return f32::NEG_INFINITY; }
            match draw % 97 {
                0..=6 => f32::NEG_INFINITY,
                7 => f32::INFINITY,
                8 => -0.0,
                _ => ((draw >> 8) % 17) as f32 * 0.25 - 2.0 + if i % 1000 == 999 { 0.125 } else { 0.0 },
            }
        }).collect();
        let (input, out) = (upload(&mut gpu, &scores), gpu.buffer((rows * k * 4) as u64).unwrap());
        gpu.top_k(input.base(), rows as u64, groups as u64, k as u64, out.base()).unwrap();
        let got = read(&mut gpu, &out, rows * k);
        for row in 0..rows {
            assert_eq!(got[row * k..(row + 1) * k], top_k_reference(&scores[row * groups..(row + 1) * groups], k)[..],
                       "{groups} groups, row {row}");
        }
    }
}

#[test]
fn an_index_select_matches_rope_scores_and_top_k() {
    let _one_at_a_time = one_at_a_time();
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
                             scale, (rope_dims as u64, &positions, &frequencies), scores_2.base(), k as u64, out_2.base(), None).unwrap();
            assert_eq!(read(&mut gpu, &scores_1, rows * groups), read(&mut gpu, &scores_2, rows * groups), "{rows} rows, run {run}: scores");
            assert_eq!(read(&mut gpu, &out_1, rows * k), read(&mut gpu, &out_2, rows * k), "{rows} rows, run {run}: indices");
        }
        assert_eq!(read(&mut gpu, &q_2, rows * heads * dim), q.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), "q unchanged");
    }
}

#[test]
fn an_index_select_with_its_projection_matches_the_two_ops() {
    let _one_at_a_time = one_at_a_time();
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
                         (rope_dims as u64, &positions, &frequencies), scores_1.base(), k as u64, out_1.base(), None).unwrap();
        for run in 0..2 {
            let (weights_2, scores_2, out_2) = fresh(&mut gpu);
            gpu.index_select_proj((&matrix, kind, width as u64), x.base(), q.base(), keys.base(), weights_2.base(), visible.base(),
                                  groups as u64, heads as u64, dim as u64, scale, (rope_dims as u64, &positions, &frequencies),
                                  scores_2.base(), k as u64, out_2.base(), None).unwrap();
            assert_eq!(read(&mut gpu, &weights_1, heads), read(&mut gpu, &weights_2, heads), "kind {kind}, run {run}: weights");
            assert_eq!(read(&mut gpu, &scores_1, groups), read(&mut gpu, &scores_2, groups), "kind {kind}, run {run}: scores");
            assert_eq!(read(&mut gpu, &out_1, k), read(&mut gpu, &out_2, k), "kind {kind}, run {run}: indices");
        }
    }
}

#[test]
fn the_two_level_indexer_matches_its_reference() {
    let _one_at_a_time = one_at_a_time();
    let (heads, dim, rope_dims, k, blocks, block) = (32usize, 128usize, 64usize, 512usize, 2048usize, 8usize);
    let mut random = Random(0x1f83_d9ab_fb41_bd6b);
    let floats = |bits: Vec<u32>| bits.into_iter().map(f32::from_bits).collect::<Vec<_>>();
    // one row, a few (one of them with fewer blocks than candidates: every one kept), F32 and with the
    // FP16 prompt pass's products (WMMA scores), and that pass's 32 (its scores, then the top k over the
    // candidates)
    for (rows, groups, half) in [(1usize, 20000usize, false), (3, 23001, false), (1, 20000, true), (3, 23001, true), (32, 17000, true)] {
        let mut gpu = Backend::new(0, 1 << 30).unwrap();
        if half {
            gpu.configure_gemm(32, true).unwrap();
        }
        let frequencies: Vec<f32> = (0..rope_dims / 2).map(|i| 1.6e5f32.powf(-(2.0 * i as f32) / rope_dims as f32)).collect();
        let frequencies = upload(&mut gpu, &frequencies);
        let keys = upload(&mut gpu, &random.floats(groups * dim, 1.0));
        let seen: Vec<i32> = (0..rows).map(|row| if row == 1 { 16000 } else { (groups - 13 * row) as i32 }).collect();
        let visible = gpu.buffer((rows * 4) as u64).unwrap();
        gpu.write_buffer(&visible, 0, &seen).unwrap();
        let positions = gpu.buffer((rows * 4) as u64).unwrap();
        gpu.write_buffer(&positions, 0, &seen).unwrap();
        let scale = 1.0 / ((dim as f32).sqrt() * (heads as f32).sqrt());
        let rope = (rope_dims as u64, &positions, &frequencies);
        let fresh = |gpu: &mut Backend| (gpu.buffer((rows * groups * 4) as u64).unwrap(), gpu.buffer((rows * k * 4) as u64).unwrap());
        // the candidate source: every group, then its blocks
        let (q, weights) = (upload(&mut gpu, &random.floats(rows * heads * dim, 2.0)), upload(&mut gpu, &random.floats(rows * heads, 1.0)));
        let (scores, out) = fresh(&mut gpu);
        gpu.index_select(q.base(), keys.base(), weights.base(), visible.base(), rows as u64, groups as u64, heads as u64, dim as u64, scale,
                         rope, scores.base(), k as u64, out.base(), None).unwrap();
        let candidates = gpu.buffer((rows * blocks * 4) as u64).unwrap();
        let scratch = gpu.buffer((rows * groups.div_ceil(block) * 4) as u64).unwrap();
        gpu.index_candidates(scores.base(), visible.base(), rows as u64, groups as u64, blocks as u64, block as u64, candidates.base(),
                             &scratch).unwrap();
        let source = floats(read(&mut gpu, &scores, rows * groups));
        let chosen = read(&mut gpu, &candidates, rows * blocks);
        let mut keep = vec![false; rows * groups];
        for row in 0..rows {
            let row_scores = &source[row * groups..(row + 1) * groups];
            // each block's highest (in top_k_row's order), the newest visible one's +inf
            let newest = (seen[row] as usize - 1) / block;
            let best: Vec<f32> = (0..groups.div_ceil(block)).map(|b| if b == newest { f32::INFINITY } else {
                row_scores[b * block..((b + 1) * block).min(groups)].iter().copied().max_by_key(|&score| top_k_key(score)).unwrap()
            }).collect();
            let expected = top_k_reference(&best, blocks);
            assert_eq!(chosen[row * blocks..(row + 1) * blocks], expected[..], "{rows} rows, row {row}: candidate blocks");
            for &b in expected.iter().filter(|&&b| b != u32::MAX) {
                keep[row * groups + b as usize * block..(row * groups + (b as usize + 1) * block).min((row + 1) * groups)].fill(true);
            }
        }
        // a later indexer: every group's scores, and its choice among the candidates against their top k
        let (q, weights) = (upload(&mut gpu, &random.floats(rows * heads * dim, 2.0)), upload(&mut gpu, &random.floats(rows * heads, 1.0)));
        let (every, _) = fresh(&mut gpu);
        gpu.index_select(q.base(), keys.base(), weights.base(), visible.base(), rows as u64, groups as u64, heads as u64, dim as u64, scale,
                         rope, every.base(), k as u64, out.base(), None).unwrap();
        let masked: Vec<f32> = floats(read(&mut gpu, &every, rows * groups)).into_iter().zip(&keep)
            .map(|(score, &kept)| if kept { score } else { f32::NEG_INFINITY }).collect();
        let within = Some((candidates.base().cast_const(), blocks as u64, block as u64));
        for run in 0..2 {
            let (scores, out) = fresh(&mut gpu);
            gpu.index_select(q.base(), keys.base(), weights.base(), visible.base(), rows as u64, groups as u64, heads as u64, dim as u64, scale,
                             rope, scores.base(), k as u64, out.base(), within).unwrap();
            let got = read(&mut gpu, &out, rows * k);
            for row in 0..rows {
                assert_eq!(got[row * k..(row + 1) * k], top_k_reference(&masked[row * groups..(row + 1) * groups], k)[..],
                           "{rows} rows, run {run}, row {row}: indices among the candidates");
            }
        }
        if rows == 1 {
            // the decode launch: the indexer's projection of x into its weights, then the same choice
            let width = 5120usize;
            let x = upload(&mut gpu, &random.floats(width, 2.0));
            let bf16: Vec<u8> = random.floats(heads * width, 0.05).iter().flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes()).collect();
            let matrix = gpu.upload_bytes(&bf16).unwrap();
            let scratch = gpu.buffer(64 << 20).unwrap();
            let formed = gpu.buffer((heads * 4) as u64).unwrap();
            gpu.linear(&matrix, 30, heads as u64, width as u64, 0, x.base(), 0, 1, formed.base(), 0, &scratch).unwrap();
            gpu.index_select(q.base(), keys.base(), formed.base(), visible.base(), 1, groups as u64, heads as u64, dim as u64, scale, rope,
                             every.base(), k as u64, out.base(), None).unwrap();
            let masked: Vec<f32> = floats(read(&mut gpu, &every, groups)).into_iter().zip(&keep)
                .map(|(score, &kept)| if kept { score } else { f32::NEG_INFINITY }).collect();
            for run in 0..2 {
                let (scores, out) = fresh(&mut gpu);
                let weights = gpu.buffer((heads * 4) as u64).unwrap();
                gpu.index_select_proj((&matrix, 30, width as u64), x.base(), q.base(), keys.base(), weights.base(), visible.base(), groups as u64,
                                      heads as u64, dim as u64, scale, rope, scores.base(), k as u64, out.base(), within).unwrap();
                assert_eq!(read(&mut gpu, &out, k), top_k_reference(&masked, k), "run {run}: the decode launch's indices among the candidates");
            }
        }
    }
}

#[test]
fn f32_router_prefetch_keeps_odd_units_and_routes_bit_exact() {
    let _one_at_a_time = one_at_a_time();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x510e_527f_ade6_82d1);
    let scratch = gpu.buffer(1 << 20).unwrap();
    // One-unit tails around both a wave and a pair, plus the actual DS router shape.
    for cols in [4usize, 124, 128, 132, 252, 256, 260, 5120] {
        for experts in [1usize, 33, 384] {
            let k = experts.min(6);
            let matrix = upload(&mut gpu, &random.floats(experts * cols, 0.02));
            let input = upload(&mut gpu, &random.floats(cols, 3.0));
            let bias = upload(&mut gpu, &random.floats(experts, 0.1));
            let baseline = gpu.buffer((experts * 4) as u64).unwrap();
            let fused = gpu.buffer((experts * 4) as u64).unwrap();
            let ids_a = gpu.buffer((k * 4) as u64).unwrap();
            let ids_b = gpu.buffer((k * 4) as u64).unwrap();
            let weights_a = gpu.buffer((k * 4) as u64).unwrap();
            let weights_b = gpu.buffer((k * 4) as u64).unwrap();
            gpu.linear(&matrix, 0, experts as u64, cols as u64, 0, input.base(), 0, 1, baseline.base(), 0, &scratch).unwrap();
            for gating in 0..3 {
                gpu.route(baseline.base(), Some(bias.base()), 1, experts as u64, k as u64, gating, true, 1.3,
                          ids_a.base(), weights_a.base()).unwrap();
                // Run twice to check that the launch's completion counter resets.
                for _ in 0..2 {
                    gpu.linear_route(&matrix, 0, experts as u64, cols as u64, input.base(), 1, fused.base(), Some(bias.base()),
                                     k as u64, gating, true, 1.3, ids_b.base(), weights_b.base(), &scratch).unwrap();
                    assert_eq!(read(&mut gpu, &baseline, experts), read(&mut gpu, &fused, experts), "logits {cols}/{experts}");
                    assert_eq!(read(&mut gpu, &ids_a, k), read(&mut gpu, &ids_b, k), "ids {cols}/{experts}/{gating}");
                    assert_eq!(read(&mut gpu, &weights_a, k), read(&mut gpu, &weights_b, k), "weights {cols}/{experts}/{gating}");
                }
            }
        }
    }
}

#[test]
fn bf16_router_routes_its_own_logits_in_one_launch() {
    let _one_at_a_time = one_at_a_time();
    // DeepSeek-V4.1's router (384 experts of 5120 BF16 values) and smaller shapes: the fused launch's
    // routes are the ones sd_op_route takes from its logits, and those are sd_op_linear's within
    // F32 rounding (another kernel may sum them).
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x1f83_d9ab_fb41_bd6b);
    let scratch = gpu.buffer(1 << 20).unwrap();
    for (cols, experts) in [(5120usize, 384usize), (128, 33), (1024, 256)] {
        let k = experts.min(6);
        let bytes: Vec<u8> = random.floats(experts * cols, 0.02).iter().flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let matrix = gpu.upload_bytes(&bytes).unwrap();
        let input = upload(&mut gpu, &random.floats(cols, 3.0));
        let bias = upload(&mut gpu, &random.floats(experts, 0.1));
        let baseline = gpu.buffer((experts * 4) as u64).unwrap();
        let fused = gpu.buffer((experts * 4) as u64).unwrap();
        let ids_a = gpu.buffer((k * 4) as u64).unwrap();
        let ids_b = gpu.buffer((k * 4) as u64).unwrap();
        let weights_a = gpu.buffer((k * 4) as u64).unwrap();
        let weights_b = gpu.buffer((k * 4) as u64).unwrap();
        gpu.linear(&matrix, 30, experts as u64, cols as u64, 0, input.base(), 0, 1, baseline.base(), 0, &scratch).unwrap();
        let mut expected = vec![0f32; experts];
        gpu.read_buffer(&baseline, 0, &mut expected).unwrap();
        for gating in 0..3 {
            // Run twice to check that the launch's completion counter resets.
            for _ in 0..2 {
                gpu.linear_route(&matrix, 30, experts as u64, cols as u64, input.base(), 1, fused.base(), Some(bias.base()),
                                 k as u64, gating, true, 1.3, ids_b.base(), weights_b.base(), &scratch).unwrap();
                let mut logits = vec![0f32; experts];
                gpu.read_buffer(&fused, 0, &mut logits).unwrap();
                for (expert, (&a, &b)) in expected.iter().zip(&logits).enumerate() {
                    assert!((a - b).abs() <= 1e-5 * (1.0 + a.abs()), "logit {expert} of {cols}/{experts}: {a} against {b}");
                }
                gpu.route(fused.base(), Some(bias.base()), 1, experts as u64, k as u64, gating, true, 1.3, ids_a.base(),
                          weights_a.base()).unwrap();
                assert_eq!(read(&mut gpu, &ids_a, k), read(&mut gpu, &ids_b, k), "ids {cols}/{experts}/{gating}");
                assert_eq!(read(&mut gpu, &weights_a, k), read(&mut gpu, &weights_b, k), "weights {cols}/{experts}/{gating}");
            }
        }
    }
}

#[test]
fn a_verify_pass_routes_every_row_in_the_fused_launch() {
    let _one_at_a_time = one_at_a_time();
    // A verify pass's 2-4 rows through DeepSeek-V4.1's BF16 router: each row's routes in the one
    // launch are sd_op_route's of that row's logits (every thread takes every row's scores).
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x9b05_688c_2b3e_6c1f);
    let scratch = gpu.buffer(1 << 20).unwrap();
    let (cols, experts, k) = (5120usize, 384usize, 6usize);
    let bytes: Vec<u8> = random.floats(experts * cols, 0.02).iter().flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let matrix = gpu.upload_bytes(&bytes).unwrap();
    let bias = upload(&mut gpu, &random.floats(experts, 0.1));
    for steps in 2..=4usize {
        let input = upload(&mut gpu, &random.floats(steps * cols, 3.0));
        let logits = gpu.buffer((steps * experts * 4) as u64).unwrap();
        let (ids_a, ids_b) = (gpu.buffer((steps * k * 4) as u64).unwrap(), gpu.buffer((steps * k * 4) as u64).unwrap());
        let (weights_a, weights_b) = (gpu.buffer((steps * k * 4) as u64).unwrap(), gpu.buffer((steps * k * 4) as u64).unwrap());
        for gating in 0..3 {
            gpu.linear_route(&matrix, 30, experts as u64, cols as u64, input.base(), steps as u64, logits.base(), Some(bias.base()),
                             k as u64, gating, true, 1.3, ids_b.base(), weights_b.base(), &scratch).unwrap();
            gpu.route(logits.base(), Some(bias.base()), steps as u64, experts as u64, k as u64, gating, true, 1.3, ids_a.base(),
                      weights_a.base()).unwrap();
            let (want, got) = (read(&mut gpu, &ids_a, steps * k), read(&mut gpu, &ids_b, steps * k));
            assert_eq!(want, got, "ids of {steps} rows, gating {gating}");
            // (the rows' routes differ: a row that took row 0's scores would show here)
            assert_ne!(got[..k], got[k..2 * k], "rows 0 and 1 of {steps}, gating {gating}");
            assert_eq!(read(&mut gpu, &weights_a, steps * k), read(&mut gpu, &weights_b, steps * k), "weights of {steps} rows, gating {gating}");
        }
    }
}

#[test]
fn linears_norm_tail_graph_matches_separate_ops() {
    let _one_at_a_time = one_at_a_time();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x3c6e_f372_fe94_f82b);
    let (rank, dim, rope, context) = (1280usize, 512usize, 64usize, 16usize);
    // (Q8_0 rows also on the Q8 decode kernels: the same rows in either launch)
    for (kind, cols, q8) in [(0u32, 128usize, false), (8, 5120, false), (8, 5120, true)] {
        gpu.configure_decode(q8).unwrap();
        let key = 121 + u64::from(q8);
        let matrices: Vec<Buffer> = [rank, dim].into_iter().map(|rows| {
            if kind == 0 { upload(&mut gpu, &random.floats(rows * cols, 0.02)) }
            else {
                let mut bytes = Vec::new();
                for _ in 0..rows * cols / 32 {
                    bytes.extend_from_slice(&0x2000u16.to_le_bytes());
                    for _ in 0..32 { bytes.push(random.next() as u8); }
                }
                gpu.upload_bytes(&bytes).unwrap()
            }
        }).collect();
        let input = upload(&mut gpu, &vec![0.0; cols]);
        let qweight = upload(&mut gpu, &random.floats(rank, 1.0));
        let kvweight = upload(&mut gpu, &random.floats(dim, 1.0));
        let frequencies = upload(&mut gpu, &(0..rope / 2).map(|i| 1e4f32.powf(-(2.0 * i as f32) / rope as f32)).collect::<Vec<_>>());
        let positions = gpu.buffer(4).unwrap();
        let out_rows = gpu.buffer(4).unwrap();
        gpu.write_buffer(&positions, 0, &[70_001i32]).unwrap();
        gpu.write_buffer(&out_rows, 0, &[5i32]).unwrap();
        let outputs = |gpu: &mut Backend| [gpu.buffer((rank * 4) as u64).unwrap(), gpu.buffer((dim * 4) as u64).unwrap(),
            gpu.buffer((rank * 4) as u64).unwrap(), upload(gpu, &vec![0.0; context * dim])];
        let a = outputs(&mut gpu); let b = outputs(&mut gpu);
        let scratch = gpu.buffer(16 << 20).unwrap();
        let items = |out: &[Buffer; 4]| [rank, dim].into_iter().enumerate().map(|(i, rows)| Linear {
            weights: matrices[i], kind, rows: rows as u64, cols: cols as u64, first_row: 0, input: input.base(), output: out[i].base()
        }).collect::<Vec<_>>();
        let side = NormRope { input: b[1].base(), weight: &kvweight, width: dim as u64, eps: 1e-6,
            rope: Some((rope as u64, &positions, &frequencies)), out_rows: Some(out_rows.base()), out: b[3].base(), out_count: context as u64 };
        // The first invocation is captured: no prior launch initializes its counter.
        gpu.graph_begin().unwrap();
        gpu.linears_norm_rope(&items(&b), b[0].base(), &qweight, rank as u64, 1e-6, b[2].base(), &side).unwrap();
        gpu.graph_end(key, true).unwrap();
        for replay in 0..4 {
            gpu.write_buffer(&input, 0, &random.floats(cols, 2.0)).unwrap();
            gpu.write_buffer(&positions, 0, &[70_001i32 + replay]).unwrap();
            gpu.write_buffer(&out_rows, 0, &[5i32 + replay]).unwrap();
            gpu.linears(&items(&a), 1, 0, 0, &scratch).unwrap();
            gpu.rms_norm(a[0].base(), &qweight, 1, rank as u64, 1e-6, a[2].base()).unwrap();
            gpu.norm_rope(a[1].base(), &kvweight, 1, dim as u64, 1e-6, Some((rope as u64, &positions, &frequencies)),
                Some(out_rows.base()), a[3].base(), context as u64).unwrap();
            assert!(gpu.graph_run(key).unwrap());
            for (i, count) in [rank, dim, rank, context * dim].into_iter().enumerate() {
                assert_eq!(read(&mut gpu, &a[i], count), read(&mut gpu, &b[i], count), "kind {kind}, Q8 {q8}, replay {replay}, output {i}");
            }
        }
    }
}

/// DeepSeek-V4.1's routing (384 experts, square-root softplus, a bias, top 6), whose tail runs in
/// one wave: the picks a host reference makes (the highest biased score, the lowest index on a tie,
/// none twice) and their scores normalized and scaled.
#[test]
fn a_wave_routes_384_experts_as_the_reference_does() {
    let _one_at_a_time = one_at_a_time();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x3c6e_f372_fe94_f82b);
    let (experts, k, scale) = (384usize, 6usize, 1.5f32);
    for trial in 0..20 {
        // (values apart by far more than an ulp, and runs of equal ones: ties)
        let logits: Vec<f32> = (0..experts).map(|e| if trial % 2 == 0 && e % 7 == 0 { 0.5 } else { (random.next() % 4000) as f32 / 500.0 - 4.0 })
            .collect();
        let bias: Vec<f32> = (0..experts).map(|e| if trial % 3 == 0 { 0.0 } else { ((e * 37 + trial) % 11) as f32 / 1000.0 }).collect();
        let (logits_buffer, bias_buffer) = (upload(&mut gpu, &logits), upload(&mut gpu, &bias));
        let (ids, weights) = (gpu.buffer(64).unwrap(), gpu.buffer(64).unwrap());
        gpu.route(logits_buffer.base(), Some(bias_buffer.base()), 1, experts as u64, k as u64, 2, true, scale, ids.base(), weights.base())
            .unwrap();
        let mut got_ids = vec![0i32; k];
        gpu.read_buffer(&ids, 0, &mut got_ids).unwrap();
        let got_weights: Vec<f32> = read(&mut gpu, &weights, k).into_iter().map(f32::from_bits).collect();
        let scores: Vec<f32> = logits.iter().map(|&x| if x > 20.0 { x.sqrt() } else { x.exp().ln_1p().sqrt() }).collect();
        let mut picked: Vec<usize> = Vec::new();
        for _ in 0..k {
            let best = (0..experts).filter(|e| !picked.contains(e))
                .reduce(|best, e| if scores[e] + bias[e] > scores[best] + bias[best] { e } else { best }).unwrap();
            picked.push(best);
        }
        assert_eq!(got_ids, picked.iter().map(|&e| e as i32).collect::<Vec<_>>(), "trial {trial}");
        let sum: f32 = picked.iter().map(|&e| scores[e]).sum();
        for (slot, &e) in picked.iter().enumerate() {
            let want = scores[e] / (sum + 1e-20) * scale;
            assert!((got_weights[slot] - want).abs() <= 1e-5 * want.abs(), "trial {trial} slot {slot}: {} vs {want}", got_weights[slot]);
        }
    }
}
