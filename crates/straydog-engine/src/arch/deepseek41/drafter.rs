//! DSpark, DeepSeek-V4.1's three-stage block drafter (llama.cpp be83657 src/models/dflash.cpp's
//! graph_dsv41), on one device: its weights resident there, the target's token embedding read on
//! the host and the target's LM head copied whole. It only proposes: nothing here changes what
//! the target computes.
//!
//! State: three K-only rings of the last [`RING`] positions' injected target features (by position
//! modulo the ring). A draft at `position` (where the anchor token goes) reads positions
//! [position - history, position), so those must have been injected (Drafter::inject) first.

use crate::arch::Tensors;
use crate::formats;
use crate::gpu::{Backend, Result};
use crate::ops::{Buffer, DsparkInjectStage, Linear};

/// The rows of history a block attends to (and each ring's rows).
pub const RING: usize = 128;
/// The most rows a draft block takes here (the target's GEMV takes up to four vectors).
pub const MAX_SLOTS: usize = 4;
const WIDTH: usize = 5120;
const HC: usize = 4;
const HEADS: usize = 64;
const HEAD_DIM: usize = 512;
const ROPE: usize = 64;
const Q_RANK: usize = 1280;
const GROUPS: usize = 8;
const OUT_RANK: usize = 1024;
const EXPERTS: usize = 128;
const USED: usize = 3;
const FFN: usize = 2304;
const MARKOV: usize = 256;
const MIXES: usize = 2 * HC + HC * HC;
const NOISE: u32 = 128_799;
const NORM_EPS: f32 = 1e-20;
const HC_EPS: f32 = 1e-6;
const SINKHORN: u64 = 20;
const CLAMP: f32 = 10.0;
const SCALE: f32 = 1.5;
const FEATURES: usize = 3 * WIDTH;

struct Stage {
    q_a: Buffer,
    q_b: Buffer,
    kv: Buffer,
    out_a: Buffer,
    out_b: Buffer,
    attn_norm: Buffer,
    ffn_norm: Buffer,
    q_a_norm: Buffer,
    kv_norm: Buffer,
    sinks: Buffer,
    hc_attn: [Buffer; 3],
    hc_ffn: [Buffer; 3],
    router: Buffer,
    bias: Buffer,
    /// The table of the routed experts' slots, [gate | up | down] (MXFP4), on the device.
    table: Buffer,
    /// The shared expert as one slot, [gate | up | down] (Q8_0).
    shared: Buffer,
    ring: Buffer,
}

/// The drafter's weights and buffers on its device.
pub struct Drafter {
    stages: Vec<Stage>,
    fc: Buffer,
    encoder_norm: Buffer,
    output_norm: Buffer,
    markov: [Buffer; 2],
    confidence: Buffer,
    head: Buffer,
    head_kind: u32,
    vocabulary: usize,
    frequencies: Buffer,
    expert_bytes: u64,
    shared_bytes: u64,
    // a block's buffers
    stream: [Buffer; 2],
    x: Buffer,
    collapsed: Buffer,
    q_a: Buffer,
    q: Buffer,
    kv: Buffer,
    attended: Buffer,
    low: Buffer,
    out: Buffer,
    mixes: Buffer,
    /// (pre, post, comb): the stage's attention's, then two FFNs' in turn (a stage's attention
    /// collapses by the previous stage's FFN pre-mix).
    coefficients: [[Buffer; 3]; 3],
    /// Pre coefficients that collapse a stream to its copy 0 (rows of [1, 0, 0, 0]).
    first: Buffer,
    logits_router: Buffer,
    ids: Buffer,
    weights: Buffer,
    product: Buffer,
    logits: Buffer,
    corrected: Buffer,
    tokens: Buffer,
    confidences: Buffer,
    count: Buffer,
    positions: Buffer,
    inject_positions: Buffer,
    encoded: Buffer,
    scratch: Buffer,
    /// Per ring row, the position injected there (-1: none).
    held: Vec<i64>,
    /// The target's token embedding rows, decoded on the host.
    embed_kind: u32,
}

fn upload(gpu: &mut Backend, tensors: &impl Tensors, name: &str, kind: u32, dims: &[usize]) -> Result<Buffer> {
    let tensor = tensors.tensor(name)?;
    if tensor.kind != kind || tensor.dims != dims {
        return Err(format!("DSpark tensor {name}: type {} shape {:?}, expected {kind} {dims:?}", tensor.kind, tensor.dims));
    }
    gpu.upload_bytes(tensor.data)
}

/// Each expert's gate, up and down (one row of each tensor's experts) back to back in a slot.
fn slots(gpu: &mut Backend, tensors: &impl Tensors, names: [&str; 3], count: usize, kind: u32) -> Result<(Buffer, u64)> {
    let parts: Vec<_> = names.iter().map(|name| tensors.tensor(name)).collect::<Result<_>>()?;
    if parts.iter().any(|part| part.kind != kind || !part.data.len().is_multiple_of(count)) {
        return Err(format!("DSpark experts {names:?} are not {count} experts of type {kind}"));
    }
    let each = parts[0].data.len() / count;
    if parts.iter().any(|part| part.data.len() / count != each) {
        return Err(format!("DSpark experts {names:?} differ in size"));
    }
    let slot = 3 * each as u64;
    let buffer = gpu.buffer(slot * count as u64)?;
    for expert in 0..count {
        for (role, part) in parts.iter().enumerate() {
            gpu.write_buffer(&buffer, expert as u64 * slot + (role * each) as u64, &part.data[expert * each..(expert + 1) * each])?;
        }
    }
    Ok((buffer, slot))
}

impl Drafter {
    /// Load DSpark (`draft`, checked by [`super::dspark::Dspark`]) onto `gpu`, with the target's
    /// LM head (`target`'s output.weight, whole).
    pub fn load(gpu: &mut Backend, draft: &impl Tensors, target: &impl Tensors, vocabulary: usize) -> Result<Drafter> {
        let mut stages = Vec::new();
        let (mut expert_bytes, mut shared_bytes) = (0, 0);
        for stage in 0..3 {
            let name = |suffix: &str| format!("blk.{stage}.{suffix}");
            let (experts, slot) = slots(gpu, draft, [&name("ffn_gate_exps.weight"), &name("ffn_up_exps.weight"),
                                                      &name("ffn_down_exps.weight")], EXPERTS, 39)?;
            let (shared, shared_slot) = slots(gpu, draft, [&name("ffn_gate_shexp.weight"), &name("ffn_up_shexp.weight"),
                                                            &name("ffn_down_shexp.weight")], 1, 8)?;
            (expert_bytes, shared_bytes) = (slot, shared_slot);
            let table = gpu.buffer(EXPERTS as u64 * 8)?;
            let bases: Vec<u64> = (0..EXPERTS as u64).map(|expert| experts.base() as u64 + expert * slot).collect();
            gpu.expert_table(&table, &bases, slot)?;
            let ring = gpu.buffer((RING * HEAD_DIM * 4) as u64)?;
            let f32s = |gpu: &mut Backend, suffix: &str, dims: &[usize]| upload(gpu, draft, &name(suffix), 0, dims);
            stages.push(Stage {
                q_a: upload(gpu, draft, &name("attn_q_a.weight"), 8, &[WIDTH, Q_RANK])?,
                q_b: upload(gpu, draft, &name("attn_q_b.weight"), 8, &[Q_RANK, HEADS * HEAD_DIM])?,
                kv: upload(gpu, draft, &name("attn_kv.weight"), 8, &[WIDTH, HEAD_DIM])?,
                out_a: upload(gpu, draft, &name("attn_output_a.weight"), 8, &[HEADS / GROUPS * HEAD_DIM, GROUPS * OUT_RANK])?,
                out_b: upload(gpu, draft, &name("attn_output_b.weight"), 8, &[GROUPS * OUT_RANK, WIDTH])?,
                attn_norm: f32s(gpu, "attn_norm.weight", &[WIDTH])?,
                ffn_norm: f32s(gpu, "ffn_norm.weight", &[WIDTH])?,
                q_a_norm: f32s(gpu, "attn_q_a_norm.weight", &[Q_RANK])?,
                kv_norm: f32s(gpu, "attn_kv_a_norm.weight", &[HEAD_DIM])?,
                sinks: f32s(gpu, "attn_sinks.weight", &[HEADS])?,
                hc_attn: [f32s(gpu, "hc_attn_fn.weight", &[HC * WIDTH, MIXES])?, f32s(gpu, "hc_attn_base.weight", &[MIXES])?,
                          f32s(gpu, "hc_attn_scale.weight", &[3])?],
                hc_ffn: [f32s(gpu, "hc_ffn_fn.weight", &[HC * WIDTH, MIXES])?, f32s(gpu, "hc_ffn_base.weight", &[MIXES])?,
                         f32s(gpu, "hc_ffn_scale.weight", &[3])?],
                router: f32s(gpu, "ffn_gate_inp.weight", &[WIDTH, EXPERTS])?,
                bias: upload(gpu, draft, &name("exp_probs_b.bias"), 0, &[EXPERTS])?,
                table, shared, ring,
            });
        }
        let head = target.tensor("output.weight")?;
        if head.dims != [WIDTH, vocabulary] {
            return Err(format!("the target's output.weight is {:?}, not {WIDTH} x {vocabulary}", head.dims));
        }
        let head_kind = head.kind;
        let head = gpu.upload_bytes(head.data)?;
        let embed_kind = target.tensor("token_embd.weight")?.kind;
        let frequencies: Vec<f32> = (0..ROPE / 2).map(|i| 10000f32.powf(-(2.0 * i as f32) / ROPE as f32)).collect();
        let frequencies_buffer = gpu.buffer((frequencies.len() * 4) as u64)?;
        gpu.write_buffer(&frequencies_buffer, 0, &frequencies)?;
        let floats = |gpu: &mut Backend, count: usize| gpu.buffer((count * 4) as u64);
        let s = MAX_SLOTS;
        let drafter = Drafter {
            fc: upload(gpu, draft, "fc.weight", 8, &[FEATURES, WIDTH])?,
            encoder_norm: upload(gpu, draft, "enc.output_norm.weight", 0, &[WIDTH])?,
            output_norm: upload(gpu, draft, "output_norm.weight", 0, &[WIDTH])?,
            markov: [upload(gpu, draft, "markov_w1.weight", 30, &[MARKOV, vocabulary])?,
                     upload(gpu, draft, "markov_w2.weight", 30, &[MARKOV, vocabulary])?],
            confidence: upload(gpu, draft, "conf_proj.weight", 0, &[WIDTH + MARKOV, 1])?,
            stages, head, head_kind, vocabulary, frequencies: frequencies_buffer, expert_bytes, shared_bytes,
            stream: [floats(gpu, s * HC * WIDTH)?, floats(gpu, s * HC * WIDTH)?],
            x: floats(gpu, s * WIDTH)?,
            collapsed: floats(gpu, s * WIDTH)?,
            q_a: floats(gpu, s * Q_RANK)?,
            q: floats(gpu, s * HEADS * HEAD_DIM)?,
            kv: floats(gpu, s * HEAD_DIM)?,
            attended: floats(gpu, s * HEADS * HEAD_DIM)?,
            low: floats(gpu, s * GROUPS * OUT_RANK)?,
            out: floats(gpu, s * WIDTH)?,
            mixes: floats(gpu, s * MIXES)?,
            coefficients: [[floats(gpu, s * HC)?, floats(gpu, s * HC)?, floats(gpu, s * HC * HC)?],
                           [floats(gpu, s * HC)?, floats(gpu, s * HC)?, floats(gpu, s * HC * HC)?],
                           [floats(gpu, s * HC)?, floats(gpu, s * HC)?, floats(gpu, s * HC * HC)?]],
            first: floats(gpu, s * HC)?,
            logits_router: floats(gpu, s * EXPERTS)?,
            ids: floats(gpu, s * USED)?,
            weights: floats(gpu, s * USED)?,
            product: floats(gpu, s * USED * FFN)?,
            logits: floats(gpu, s * vocabulary)?,
            corrected: floats(gpu, s * vocabulary)?,
            tokens: floats(gpu, s)?,
            confidences: floats(gpu, s)?,
            count: floats(gpu, 1)?,
            positions: floats(gpu, s)?,
            inject_positions: floats(gpu, RING)?,
            encoded: floats(gpu, RING * WIDTH)?,
            scratch: gpu.buffer(64 << 20)?,
            held: vec![-1; RING],
            embed_kind,
        };
        let first: Vec<f32> = (0..s * HC).map(|at| if at % HC == 0 { 1.0 } else { 0.0 }).collect();
        gpu.write_buffer(&drafter.first, 0, &first)?;
        Ok(drafter)
    }

    /// Inject the target features of `positions` (rows of 3 x width F32 at `features`, on this
    /// device) into the rings: the history of later drafts.
    pub fn inject(&mut self, gpu: &mut Backend, features: *const std::ffi::c_void, positions: &[i32]) -> Result<()> {
        if positions.is_empty() || positions.len() > RING {
            return Err(format!("DSpark injects 1 to {RING} rows, not {}", positions.len()));
        }
        gpu.write_buffer(&self.inject_positions, 0, positions)?;
        let stages: [DsparkInjectStage<'_>; 3] = std::array::from_fn(|stage| {
            let s = &self.stages[stage];
            DsparkInjectStage { kv: &s.kv, norm: &s.kv_norm, ring: &s.ring }
        });
        let inject_positions = self.inject_positions.slice(0, (positions.len() * 4) as u64);
        gpu.dspark_inject(features, &inject_positions, positions.len() as u64, &self.fc, &self.encoder_norm, &stages, &self.frequencies,
                          self.encoded.base(), &self.scratch)?;
        for &position in positions {
            self.held[position as usize % RING] = position as i64;
        }
        Ok(())
    }

    /// Forget every injected position from `position` on (a rewind of the target to `position`).
    pub fn forget_from(&mut self, position: usize) {
        for held in &mut self.held {
            if *held >= position as i64 {
                *held = -1;
            }
        }
    }

    /// Draft `slots` tokens after `anchor`, the token at `position` (chosen, not yet fed to the
    /// target): the block [anchor, noise...] through the three stages, then the Markov head. The
    /// tokens for positions position + 1.. and their usable prefix (confidence 0: all of them).
    pub fn draft(&mut self, gpu: &mut Backend, embed: &impl Tensors, anchor: u32, position: usize, slots: usize) -> Result<Vec<u32>> {
        if slots == 0 || slots > MAX_SLOTS || anchor as usize >= self.vocabulary {
            return Err(format!("a DSpark draft of {slots} slots after token {anchor}"));
        }
        let history = position.min(RING);
        for at in position - history..position {
            if self.held[at % RING] != at as i64 {
                return Err(format!("DSpark's history lacks position {at} (a draft at {position})"));
            }
        }
        let rows = slots as u64;
        let (w, hc) = (WIDTH as u64, HC as u64);
        // the block's embeddings, repeated into the hc copies
        let table = embed.tensor("token_embd.weight")?;
        let bytes = table.data.len() / self.vocabulary;
        let (mut stream, mut row) = (Vec::with_capacity(slots * HC * WIDTH), Vec::new());
        for slot in 0..slots {
            let token = if slot == 0 { anchor } else { NOISE } as usize;
            row.clear();
            formats::decode_into(self.embed_kind, &table.data[token * bytes..(token + 1) * bytes], &mut row)?;
            for _ in 0..HC {
                stream.extend_from_slice(&row);
            }
        }
        gpu.write_buffer(&self.stream[0], 0, &stream)?;
        let positions: Vec<i32> = (position..position + slots).map(|at| at as i32).collect();
        gpu.write_buffer(&self.positions, 0, &positions)?;
        let block_positions = self.positions.slice(0, rows * 4);
        let scratch = self.scratch;
        let linear = |gpu: &mut Backend, weights: &Buffer, kind: u32, out_rows: usize, cols: usize, input: *mut std::ffi::c_void,
                      output: *mut std::ffi::c_void| {
            gpu.linear(weights, kind, out_rows as u64, cols as u64, 0, input, 0, rows, output, 0, &scratch)
        };
        // (the attention's pre-mix: copy 0 for the first stage, the previous stage's FFN pre after)
        let mut previous: Option<usize> = None;
        for (index, stage) in self.stages.iter().enumerate() {
            let [stream, expanded] = self.stream;
            // attention coefficients from the stage's input; its pre-mix collapses for the FFN
            linear(gpu, &stage.hc_attn[0], 0, MIXES, HC * WIDTH, stream.base(), self.mixes.base())?;
            let [pre_a, post_a, comb_a] = self.coefficients[0];
            gpu.hc_coefficients(self.mixes.base(), stream.base(), rows, hc, w, stage.hc_attn[1].base(), stage.hc_attn[2].base(), SINKHORN,
                                HC_EPS, NORM_EPS, pre_a.base(), post_a.base(), comb_a.base())?;
            let pre = previous.map_or(self.first, |ffn| self.coefficients[ffn][0]);
            gpu.hc_collapse(stream.base(), pre.base(), rows, hc, w, self.collapsed.base())?;
            gpu.rms_norm(self.collapsed.base(), &stage.attn_norm, rows, w, NORM_EPS, self.x.base())?;
            // attention: q (no per-head norm) and the block's kv, rotated at the block's positions
            linear(gpu, &stage.q_a, 8, Q_RANK, WIDTH, self.x.base(), self.q_a.base())?;
            gpu.rms_norm(self.q_a.base(), &stage.q_a_norm, rows, Q_RANK as u64, NORM_EPS, self.q_a.base())?;
            linear(gpu, &stage.q_b, 8, HEADS * HEAD_DIM, Q_RANK, self.q_a.base(), self.q.base())?;
            gpu.rope(self.q.base(), rows, HEADS as u64, HEAD_DIM as u64, ROPE as u64, &block_positions, &self.frequencies, false)?;
            linear(gpu, &stage.kv, 8, HEAD_DIM, WIDTH, self.x.base(), self.kv.base())?;
            gpu.rms_norm(self.kv.base(), &stage.kv_norm, rows, HEAD_DIM as u64, NORM_EPS, self.kv.base())?;
            gpu.rope(self.kv.base(), rows, 1, HEAD_DIM as u64, ROPE as u64, &block_positions, &self.frequencies, false)?;
            gpu.dspark_attention(self.q.base(), &stage.ring, self.kv.base(), &stage.sinks, HEADS as u64, rows, position as u32, history as u64,
                                 self.attended.base(), &scratch)?;
            gpu.rope(self.attended.base(), rows, HEADS as u64, HEAD_DIM as u64, ROPE as u64, &block_positions, &self.frequencies, true)?;
            let group_width = HEADS / GROUPS * HEAD_DIM;
            let projections: Vec<Linear> = (0..GROUPS).map(|g| Linear {
                weights: stage.out_a, kind: 8, rows: OUT_RANK as u64, cols: group_width as u64, first_row: (g * OUT_RANK) as u64,
                input: self.attended.at(g * group_width) as *const std::ffi::c_void, output: self.low.at(g * OUT_RANK),
            }).collect();
            gpu.linears(&projections, rows, (HEADS * HEAD_DIM) as u64, (GROUPS * OUT_RANK) as u64, &scratch)?;
            linear(gpu, &stage.out_b, 8, WIDTH, GROUPS * OUT_RANK, self.low.base(), self.out.base())?;
            gpu.hc_expand(self.out.base(), stream.base(), post_a.base(), comb_a.base(), rows, hc, w, expanded.base())?;
            // FFN: its coefficients (the next stage's pre-mix), collapsed by the attention's pre-mix
            let other = 1 + index % 2;
            linear(gpu, &stage.hc_ffn[0], 0, MIXES, HC * WIDTH, expanded.base(), self.mixes.base())?;
            let [pre_f, post_f, comb_f] = self.coefficients[other];
            gpu.hc_coefficients(self.mixes.base(), expanded.base(), rows, hc, w, stage.hc_ffn[1].base(), stage.hc_ffn[2].base(), SINKHORN,
                                HC_EPS, NORM_EPS, pre_f.base(), post_f.base(), comb_f.base())?;
            gpu.hc_collapse(expanded.base(), pre_a.base(), rows, hc, w, self.collapsed.base())?;
            gpu.rms_norm(self.collapsed.base(), &stage.ffn_norm, rows, w, NORM_EPS, self.x.base())?;
            linear(gpu, &stage.router, 0, EXPERTS, WIDTH, self.x.base(), self.logits_router.base())?;
            gpu.route(self.logits_router.base(), Some(stage.bias.base().cast_const()), rows, EXPERTS as u64, USED as u64, 2, true, SCALE,
                      self.ids.base(), self.weights.base())?;
            let offsets = |slot: u64| [0, slot / 3, 2 * slot / 3];
            gpu.experts_table(&stage.table, EXPERTS as u64, self.ids.base(), self.expert_bytes, offsets(self.expert_bytes), [39; 3], w,
                              FFN as u64, rows, USED as u64, self.x.base(), Some(self.weights.base().cast_const()), CLAMP, false, self.product.base(),
                              self.out.base())?;
            let shared = vec![stage.shared.base() as u64; slots];
            gpu.experts(Some(shared.as_slice()), self.shared_bytes, offsets(self.shared_bytes), [8; 3], w, FFN as u64, rows, 1, self.x.base(), 0,
                        None, CLAMP, true, self.product.base(), self.out.base(), 0, None)?;
            gpu.hc_expand(self.out.base(), expanded.base(), post_f.base(), comb_f.base(), rows, hc, w, stream.base())?;
            previous = Some(other);
        }
        // the last FFN's pre-mix collapses the block (the confidence head reads it un-normed)
        let last = previous.unwrap();
        gpu.hc_collapse(self.stream[0].base(), self.coefficients[last][0].base(), rows, hc, w, self.collapsed.base())?;
        gpu.rms_norm(self.collapsed.base(), &self.output_norm, rows, w, NORM_EPS, self.x.base())?;
        linear(gpu, &self.head, self.head_kind, self.vocabulary, WIDTH, self.x.base(), self.logits.base())?;
        gpu.dspark_head(self.logits.base(), self.collapsed.base(), &self.markov[0], &self.markov[1], &self.confidence, rows,
                        self.vocabulary as u64, MARKOV as u64, w, anchor, 0.0, 0, self.corrected.base(), self.tokens.base(),
                        self.confidences.base(), self.count.base(), &scratch)?;
        let mut tokens = vec![0i32; slots];
        gpu.read_buffer(&self.tokens, 0, &mut tokens)?;
        let mut count = [0u32; 1];
        gpu.read_buffer(&self.count, 0, &mut count)?;
        Ok(tokens[..(count[0] as usize).min(slots)].iter().map(|&token| token as u32).collect())
    }
}
