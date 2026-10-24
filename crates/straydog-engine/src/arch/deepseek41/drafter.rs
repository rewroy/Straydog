//! DSpark, DeepSeek-V4.1's three-stage block drafter (llama.cpp be83657 src/models/dflash.cpp's
//! graph_dsv41), beside a target engine on its devices ([`Beside`]) and split among them as the
//! target splits itself: each device runs its share of the attention heads (its rows of q_b, its
//! sinks, its groups of wo_a and their columns of wo_b) and of every expert's width, routed and
//! shared (its rows of gate and up, its columns of down); every device sums the devices' parts in
//! device order after the attention and after the experts, and computes the rest alike. The
//! block's logits come from the target's own head (each device its rows), gathered on the first
//! device for the Markov head; the token embedding is the target's, read on the host. It only
//! proposes: nothing here changes what the target computes.
//!
//! State: on every device three K-only rings of the last [`RING`] positions' injected target
//! features (by position modulo the ring). A draft at `position` (where the anchor token goes)
//! reads positions [position - history, position), so those must have been injected
//! (Drafter::inject) first.

use std::ffi::c_void;

use super::engine::{columns, Beside};
use crate::arch::{Tensor, Tensors};
use crate::formats;
use crate::gpu::{Backend, Result};
use crate::ops::{self, Buffer, DsparkInjectStage, Linear};

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

/// One device's part of a stage.
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
    /// The table of this device's rows of the routed experts' slots, [gate | up | down] (MXFP4).
    table: Buffer,
    /// This device's rows of the shared expert as a slot (Q8_0).
    shared: Buffer,
    ring: Buffer,
}

/// An expert's (this device's rows of it) slot: its bytes and its matrices' offsets.
#[derive(Clone, Copy)]
struct Slot {
    bytes: u64,
    offsets: [u64; 3],
}

/// The drafter on one device: its part of the weights, its rings and its block's buffers.
struct Part {
    stages: Vec<Stage>,
    /// This device's shares: attention heads, wo_a groups, expert width.
    heads: usize,
    groups: usize,
    ffn: usize,
    /// Whether other devices share the sums (else the parts are the sums).
    shared_sums: bool,
    experts: Slot,
    shared_expert: Slot,
    fc: Buffer,
    encoder_norm: Buffer,
    output_norm: Buffer,
    frequencies: Buffer,
    stream: [Buffer; 2],
    x: Buffer,
    collapsed: Buffer,
    q_a: Buffer,
    q: Buffer,
    kv: Buffer,
    attended: Buffer,
    low: Buffer,
    out: Buffer,
    /// This device's parts of the attention's and of the experts' sums.
    parts: [Buffer; 2],
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
    /// This device's rows of the block's logits (the first device's go into the logits).
    head_out: Buffer,
    positions: Buffer,
    inject_positions: Buffer,
    encoded: Buffer,
    scratch: Buffer,
}

/// The rows of a draft block, where its anchor goes and how far back it reads.
struct Block {
    rows: u64,
    position: usize,
    history: usize,
}

/// The drafter's weights and buffers on its devices.
pub struct Drafter {
    parts: Vec<Part>,
    // on the first device: the Markov head and the gathered logits
    markov: [Buffer; 2],
    confidence: Buffer,
    logits: Buffer,
    corrected: Buffer,
    tokens: Buffer,
    confidences: Buffer,
    count: Buffer,
    vocabulary: usize,
    /// Per ring row, the position injected there (-1: none).
    held: Vec<i64>,
    /// The target's token embedding rows, decoded on the host.
    embed_kind: u32,
}

/// A draft: its tokens and DSpark's confidence in each.
pub struct Draft {
    pub tokens: Vec<u32>,
    pub confidences: Vec<f32>,
}

fn tensor<'t>(tensors: &'t impl Tensors, name: &str, kind: u32, dims: &[usize]) -> Result<Tensor<'t>> {
    let tensor = tensors.tensor(name)?;
    if tensor.kind != kind || tensor.dims != dims {
        return Err(format!("DSpark tensor {name}: type {} shape {:?}, expected {kind} {dims:?}", tensor.kind, tensor.dims));
    }
    Ok(tensor)
}

fn upload(gpu: &mut Backend, tensors: &impl Tensors, name: &str, kind: u32, dims: &[usize]) -> Result<Buffer> {
    gpu.upload_bytes(tensor(tensors, name, kind, dims)?.data)
}

/// Rows `first..first + count` of a matrix (rows of dims[0] values).
fn rows<'t>(tensor: &Tensor<'t>, first: usize, count: usize) -> Result<&'t [u8]> {
    let bytes = formats::matrix_bytes(tensor.kind, 1, tensor.dims[0]).ok_or(format!("unsupported tensor type {}", tensor.kind))?;
    tensor.data.get(first * bytes..(first + count) * bytes).ok_or_else(|| "DSpark rows past a tensor's end".to_string())
}

/// This device's rows (`first..first + ffn` of the width) of each of `count` experts as slots: of
/// gate and up those rows, of down those columns.
fn expert_slots(gpu: &mut Backend, [gate, up, down]: &[Tensor<'_>; 3], count: usize, first: usize, ffn: usize) -> Result<(Buffer, Slot)> {
    let (width, hidden) = (gate.dims[1], down.dims[1]);
    let slot = |expert: usize| -> Result<Vec<u8>> {
        let down = columns(down.kind, hidden, width, rows(down, expert * hidden, hidden)?, first, ffn)?;
        Ok([rows(gate, expert * width + first, ffn)?, rows(up, expert * width + first, ffn)?, down.as_slice()].concat())
    };
    let gate_bytes = formats::matrix_bytes(gate.kind, ffn, gate.dims[0]).ok_or("unsupported expert type")? as u64;
    let up_bytes = formats::matrix_bytes(up.kind, ffn, up.dims[0]).ok_or("unsupported expert type")? as u64;
    let first_slot = slot(0)?;
    let bytes = first_slot.len() as u64;
    let buffer = gpu.buffer(bytes * count as u64)?;
    gpu.write_buffer(&buffer, 0, &first_slot)?;
    for expert in 1..count {
        gpu.write_buffer(&buffer, expert as u64 * bytes, &slot(expert)?)?;
    }
    Ok((buffer, Slot { bytes, offsets: [0, gate_bytes, gate_bytes + up_bytes] }))
}

impl Part {
    /// Load this device's (`index` of `count`) part of DSpark.
    fn load(gpu: &mut Backend, draft: &impl Tensors, index: usize, count: usize, head_rows: usize) -> Result<Part> {
        let (heads, groups, ffn) = (HEADS / count, GROUPS / count, FFN / count);
        let mut stages = Vec::new();
        let (mut experts, mut shared_expert) = (None, None);
        for stage in 0..3 {
            let name = |suffix: &str| format!("blk.{stage}.{suffix}");
            let t = |suffix: &str, kind: u32, dims: &[usize]| tensor(draft, &name(suffix), kind, dims);
            let whole = |gpu: &mut Backend, suffix: &str, kind: u32, dims: &[usize]| upload(gpu, draft, &name(suffix), kind, dims);
            let routed = [t("ffn_gate_exps.weight", 39, &[WIDTH, FFN, EXPERTS])?, t("ffn_up_exps.weight", 39, &[WIDTH, FFN, EXPERTS])?,
                          t("ffn_down_exps.weight", 39, &[FFN, WIDTH, EXPERTS])?];
            let (slots, slot) = expert_slots(gpu, &routed, EXPERTS, index * ffn, ffn)?;
            let table = gpu.buffer(EXPERTS as u64 * 8)?;
            let bases: Vec<u64> = (0..EXPERTS as u64).map(|expert| slots.base() as u64 + expert * slot.bytes).collect();
            gpu.expert_table(&table, &bases, slot.bytes)?;
            let shared = [t("ffn_gate_shexp.weight", 8, &[WIDTH, FFN])?, t("ffn_up_shexp.weight", 8, &[WIDTH, FFN])?,
                          t("ffn_down_shexp.weight", 8, &[FFN, WIDTH])?];
            let (shared, shared_slot) = expert_slots(gpu, &shared, 1, index * ffn, ffn)?;
            (experts, shared_expert) = (Some(slot), Some(shared_slot));
            // this device's heads of q_b and of the sinks, its groups of wo_a and their columns of wo_b
            let q_b = t("attn_q_b.weight", 8, &[Q_RANK, HEADS * HEAD_DIM])?;
            let out_a = t("attn_output_a.weight", 8, &[HEADS / GROUPS * HEAD_DIM, GROUPS * OUT_RANK])?;
            let out_b = t("attn_output_b.weight", 8, &[GROUPS * OUT_RANK, WIDTH])?;
            let sinks = t("attn_sinks.weight", 0, &[HEADS])?;
            stages.push(Stage {
                q_a: whole(gpu, "attn_q_a.weight", 8, &[WIDTH, Q_RANK])?,
                q_b: gpu.upload_bytes(rows(&q_b, index * heads * HEAD_DIM, heads * HEAD_DIM)?)?,
                kv: whole(gpu, "attn_kv.weight", 8, &[WIDTH, HEAD_DIM])?,
                out_a: gpu.upload_bytes(rows(&out_a, index * groups * OUT_RANK, groups * OUT_RANK)?)?,
                out_b: gpu.upload_bytes(&columns(8, WIDTH, GROUPS * OUT_RANK, out_b.data, index * groups * OUT_RANK, groups * OUT_RANK)?)?,
                attn_norm: whole(gpu, "attn_norm.weight", 0, &[WIDTH])?,
                ffn_norm: whole(gpu, "ffn_norm.weight", 0, &[WIDTH])?,
                q_a_norm: whole(gpu, "attn_q_a_norm.weight", 0, &[Q_RANK])?,
                kv_norm: whole(gpu, "attn_kv_a_norm.weight", 0, &[HEAD_DIM])?,
                sinks: gpu.upload_bytes(&sinks.data[index * heads * 4..(index + 1) * heads * 4])?,
                hc_attn: [whole(gpu, "hc_attn_fn.weight", 0, &[HC * WIDTH, MIXES])?, whole(gpu, "hc_attn_base.weight", 0, &[MIXES])?,
                          whole(gpu, "hc_attn_scale.weight", 0, &[3])?],
                hc_ffn: [whole(gpu, "hc_ffn_fn.weight", 0, &[HC * WIDTH, MIXES])?, whole(gpu, "hc_ffn_base.weight", 0, &[MIXES])?,
                         whole(gpu, "hc_ffn_scale.weight", 0, &[3])?],
                router: whole(gpu, "ffn_gate_inp.weight", 0, &[WIDTH, EXPERTS])?,
                bias: upload(gpu, draft, &name("exp_probs_b.bias"), 0, &[EXPERTS])?,
                table,
                shared,
                ring: gpu.buffer((RING * HEAD_DIM * 4) as u64)?,
            });
        }
        let frequencies: Vec<f32> = (0..ROPE / 2).map(|i| 10000f32.powf(-(2.0 * i as f32) / ROPE as f32)).collect();
        let frequencies_buffer = gpu.buffer((frequencies.len() * 4) as u64)?;
        gpu.write_buffer(&frequencies_buffer, 0, &frequencies)?;
        let floats = |gpu: &mut Backend, count: usize| gpu.buffer((count * 4) as u64);
        let coefficients = |gpu: &mut Backend| -> Result<[Buffer; 3]> {
            Ok([floats(gpu, MAX_SLOTS * HC)?, floats(gpu, MAX_SLOTS * HC)?, floats(gpu, MAX_SLOTS * HC * HC)?])
        };
        let s = MAX_SLOTS;
        let part = Part {
            stages, heads, groups, ffn,
            shared_sums: count > 1,
            experts: experts.ok_or("DSpark has no stages")?,
            shared_expert: shared_expert.ok_or("DSpark has no stages")?,
            fc: upload(gpu, draft, "fc.weight", 8, &[FEATURES, WIDTH])?,
            encoder_norm: upload(gpu, draft, "enc.output_norm.weight", 0, &[WIDTH])?,
            output_norm: upload(gpu, draft, "output_norm.weight", 0, &[WIDTH])?,
            frequencies: frequencies_buffer,
            stream: [floats(gpu, s * HC * WIDTH)?, floats(gpu, s * HC * WIDTH)?],
            x: floats(gpu, s * WIDTH)?,
            collapsed: floats(gpu, s * WIDTH)?,
            q_a: floats(gpu, s * Q_RANK)?,
            q: floats(gpu, s * heads * HEAD_DIM)?,
            kv: floats(gpu, s * HEAD_DIM)?,
            attended: floats(gpu, s * heads * HEAD_DIM)?,
            low: floats(gpu, s * groups * OUT_RANK)?,
            out: floats(gpu, s * WIDTH)?,
            parts: [floats(gpu, s * WIDTH)?, floats(gpu, s * WIDTH)?],
            mixes: floats(gpu, s * MIXES)?,
            coefficients: [coefficients(gpu)?, coefficients(gpu)?, coefficients(gpu)?],
            first: floats(gpu, s * HC)?,
            logits_router: floats(gpu, s * EXPERTS)?,
            ids: floats(gpu, s * USED)?,
            weights: floats(gpu, s * USED)?,
            product: floats(gpu, s * USED * ffn)?,
            head_out: floats(gpu, s * head_rows)?,
            positions: floats(gpu, s)?,
            inject_positions: floats(gpu, RING)?,
            encoded: floats(gpu, RING * WIDTH)?,
            scratch: gpu.buffer(64 << 20)?,
        };
        let first: Vec<f32> = (0..s * HC).map(|at| if at % HC == 0 { 1.0 } else { 0.0 }).collect();
        gpu.write_buffer(&part.first, 0, &first)?;
        Ok(part)
    }

    /// Where this device's part of sum `which` goes: its part, or the sum itself alone.
    fn target(&self, which: usize) -> *mut c_void {
        if self.shared_sums { self.parts[which].base() } else { self.out.base() }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(&self, gpu: &mut Backend, weights: &Buffer, kind: u32, rows: usize, cols: usize, input: *mut c_void, output: *mut c_void,
              block: &Block) -> Result<()> {
        gpu.linear(weights, kind, rows as u64, cols as u64, 0, input, 0, block.rows, output, 0, &self.scratch)
    }

    /// A stage's attention: its coefficients from the stream, collapsed by `pre`, this device's heads
    /// over the ring and the block, into this device's part of the attention's sum.
    fn attention(&self, gpu: &mut Backend, stage: usize, pre: Buffer, block: &Block) -> Result<()> {
        let (s, rows, (w, hc)) = (&self.stages[stage], block.rows, (WIDTH as u64, HC as u64));
        let [stream, _] = self.stream;
        let [pre_a, post_a, comb_a] = self.coefficients[0];
        self.linear(gpu, &s.hc_attn[0], 0, MIXES, HC * WIDTH, stream.base(), self.mixes.base(), block)?;
        gpu.hc_coefficients(self.mixes.base(), stream.base(), rows, hc, w, s.hc_attn[1].base(), s.hc_attn[2].base(), SINKHORN, HC_EPS,
                            NORM_EPS, pre_a.base(), post_a.base(), comb_a.base())?;
        gpu.hc_collapse(stream.base(), pre.base(), rows, hc, w, self.collapsed.base())?;
        gpu.rms_norm(self.collapsed.base(), &s.attn_norm, rows, w, NORM_EPS, self.x.base())?;
        // q (no per-head norm) and the block's kv, rotated at the block's positions
        let positions = self.positions.slice(0, rows * 4);
        let q_width = self.heads * HEAD_DIM;
        self.linear(gpu, &s.q_a, 8, Q_RANK, WIDTH, self.x.base(), self.q_a.base(), block)?;
        gpu.rms_norm(self.q_a.base(), &s.q_a_norm, rows, Q_RANK as u64, NORM_EPS, self.q_a.base())?;
        self.linear(gpu, &s.q_b, 8, q_width, Q_RANK, self.q_a.base(), self.q.base(), block)?;
        gpu.rope(self.q.base(), rows, self.heads as u64, HEAD_DIM as u64, ROPE as u64, &positions, &self.frequencies, false)?;
        self.linear(gpu, &s.kv, 8, HEAD_DIM, WIDTH, self.x.base(), self.kv.base(), block)?;
        gpu.rms_norm(self.kv.base(), &s.kv_norm, rows, HEAD_DIM as u64, NORM_EPS, self.kv.base())?;
        gpu.rope(self.kv.base(), rows, 1, HEAD_DIM as u64, ROPE as u64, &positions, &self.frequencies, false)?;
        gpu.dspark_attention(self.q.base(), &s.ring, self.kv.base(), &s.sinks, self.heads as u64, rows, block.position as u32,
                             block.history as u64, self.attended.base(), &self.scratch)?;
        gpu.rope(self.attended.base(), rows, self.heads as u64, HEAD_DIM as u64, ROPE as u64, &positions, &self.frequencies, true)?;
        let group_width = HEADS / GROUPS * HEAD_DIM;
        let projections: Vec<Linear> = (0..self.groups).map(|g| Linear {
            weights: s.out_a, kind: 8, rows: OUT_RANK as u64, cols: group_width as u64, first_row: (g * OUT_RANK) as u64,
            input: self.attended.at(g * group_width) as *const c_void, output: self.low.at(g * OUT_RANK),
        }).collect();
        gpu.linears(&projections, rows, q_width as u64, (self.groups * OUT_RANK) as u64, &self.scratch)?;
        self.linear(gpu, &s.out_b, 8, WIDTH, self.groups * OUT_RANK, self.low.base(), self.target(0), block)
    }

    /// A stage's FFN after the attention's sum: the stream expanded, the FFN's coefficients (set
    /// `ffn`), collapsed by the attention's pre-mix, this device's rows of the routed and shared
    /// experts into its part of the experts' sum.
    fn ffn(&self, gpu: &mut Backend, stage: usize, ffn: usize, block: &Block) -> Result<()> {
        let (s, rows, (w, hc)) = (&self.stages[stage], block.rows, (WIDTH as u64, HC as u64));
        let [stream, expanded] = self.stream;
        let [pre_a, post_a, comb_a] = self.coefficients[0];
        gpu.hc_expand(self.out.base(), stream.base(), post_a.base(), comb_a.base(), rows, hc, w, expanded.base())?;
        self.linear(gpu, &s.hc_ffn[0], 0, MIXES, HC * WIDTH, expanded.base(), self.mixes.base(), block)?;
        let [pre_f, post_f, comb_f] = self.coefficients[ffn];
        gpu.hc_coefficients(self.mixes.base(), expanded.base(), rows, hc, w, s.hc_ffn[1].base(), s.hc_ffn[2].base(), SINKHORN, HC_EPS,
                            NORM_EPS, pre_f.base(), post_f.base(), comb_f.base())?;
        gpu.hc_collapse(expanded.base(), pre_a.base(), rows, hc, w, self.collapsed.base())?;
        gpu.rms_norm(self.collapsed.base(), &s.ffn_norm, rows, w, NORM_EPS, self.x.base())?;
        self.linear(gpu, &s.router, 0, EXPERTS, WIDTH, self.x.base(), self.logits_router.base(), block)?;
        gpu.route(self.logits_router.base(), Some(s.bias.base().cast_const()), rows, EXPERTS as u64, USED as u64, 2, true, SCALE,
                  self.ids.base(), self.weights.base())?;
        let (experts, shared) = (self.experts, self.shared_expert);
        gpu.experts_table(&s.table, EXPERTS as u64, self.ids.base(), experts.bytes, experts.offsets, [39; 3], w, self.ffn as u64, rows,
                          USED as u64, self.x.base(), Some(self.weights.base().cast_const()), CLAMP, false, self.product.base(),
                          self.target(1))?;
        let bases = vec![s.shared.base() as u64; rows as usize];
        gpu.experts(Some(bases.as_slice()), shared.bytes, shared.offsets, [8; 3], w, self.ffn as u64, rows, 1, self.x.base(), 0, None, CLAMP,
                    true, self.product.base(), self.target(1), 0, None)
    }

    /// After the experts' sum: the stream expanded by the FFN's coefficients (set `ffn`).
    fn expand(&self, gpu: &mut Backend, ffn: usize, block: &Block) -> Result<()> {
        let [stream, expanded] = self.stream;
        let [_, post_f, comb_f] = self.coefficients[ffn];
        gpu.hc_expand(self.out.base(), expanded.base(), post_f.base(), comb_f.base(), block.rows, HC as u64, WIDTH as u64, stream.base())
    }
}

impl Drafter {
    /// Load DSpark (`draft`, checked by [`super::dspark::Dspark`]) onto a target engine's devices,
    /// split among them; its head is theirs ([`Beside`]).
    pub fn load(devices: &mut [Beside<'_>], draft: &impl Tensors, target: &impl Tensors, vocabulary: usize) -> Result<Drafter> {
        let count = devices.len();
        if count == 0 || !HEADS.is_multiple_of(count) || !GROUPS.is_multiple_of(count) || !FFN.is_multiple_of(32 * count) {
            return Err(format!("DSpark does not split among {count} devices"));
        }
        let mut first_row = 0;
        for device in devices.iter() {
            if device.head_first != first_row || device.head_kind != devices[0].head_kind {
                return Err("the target's head is not split in order".into());
            }
            first_row += device.head_rows;
        }
        if first_row != vocabulary {
            return Err(format!("the target's head has {first_row} rows, not {vocabulary}"));
        }
        let mut parts = Vec::new();
        for (index, device) in devices.iter_mut().enumerate() {
            parts.push(Part::load(device.gpu, draft, index, count, device.head_rows)?);
        }
        let gpu = &mut *devices[0].gpu;
        let floats = |gpu: &mut Backend, count: usize| gpu.buffer((count * 4) as u64);
        Ok(Drafter {
            parts,
            markov: [upload(gpu, draft, "markov_w1.weight", 30, &[MARKOV, vocabulary])?,
                     upload(gpu, draft, "markov_w2.weight", 30, &[MARKOV, vocabulary])?],
            confidence: upload(gpu, draft, "conf_proj.weight", 0, &[WIDTH + MARKOV, 1])?,
            logits: floats(gpu, MAX_SLOTS * vocabulary)?,
            corrected: floats(gpu, MAX_SLOTS * vocabulary)?,
            tokens: floats(gpu, MAX_SLOTS)?,
            confidences: floats(gpu, MAX_SLOTS)?,
            count: floats(gpu, 1)?,
            vocabulary,
            held: vec![-1; RING],
            embed_kind: target.tensor("token_embd.weight")?.kind,
        })
    }

    /// Inject the target features of `positions` (each device's tap's first rows) into the rings:
    /// the history of later drafts.
    pub fn inject(&mut self, devices: &mut [Beside<'_>], positions: &[i32]) -> Result<()> {
        if positions.is_empty() || positions.len() > RING || devices.len() != self.parts.len() {
            return Err(format!("DSpark injects 1 to {RING} rows on its devices, not {}", positions.len()));
        }
        for (part, device) in self.parts.iter().zip(devices.iter_mut()) {
            let gpu = &mut *device.gpu;
            gpu.write_buffer(&part.inject_positions, 0, positions)?;
            let stages: [DsparkInjectStage<'_>; 3] = std::array::from_fn(|stage| {
                let s = &part.stages[stage];
                DsparkInjectStage { kv: &s.kv, norm: &s.kv_norm, ring: &s.ring }
            });
            let inject_positions = part.inject_positions.slice(0, (positions.len() * 4) as u64);
            gpu.dspark_inject(device.tap.base(), &inject_positions, positions.len() as u64, &part.fc, &part.encoder_norm, &stages,
                              &part.frequencies, part.encoded.base(), &part.scratch)?;
        }
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

    /// Every device's sum `which` of the devices' parts.
    fn sum(&self, devices: &[Beside<'_>], which: usize, block: &Block) -> Result<()> {
        if devices.len() == 1 {
            return Ok(());
        }
        let gpus: Vec<&Backend> = devices.iter().map(|device| &*device.gpu).collect();
        let outs: Vec<*mut c_void> = self.parts.iter().map(|part| part.out.base()).collect();
        let parts: Vec<*const c_void> = self.parts.iter().map(|part| part.parts[which].base() as *const c_void).collect();
        ops::sum_devices(&gpus, &outs, &parts, block.rows * WIDTH as u64)
    }

    /// Draft `slots` tokens after `anchor`, the token at `position` (chosen, not yet fed to the
    /// target): the block [anchor, noise...] through the three stages, then the Markov head. The
    /// tokens for positions position + 1.. (as many as the head keeps: confidence 0, all) and
    /// DSpark's confidence in each.
    pub fn draft(&mut self, devices: &mut [Beside<'_>], embed: &impl Tensors, anchor: u32, position: usize, slots: usize)
                 -> Result<Draft> {
        if slots == 0 || slots > MAX_SLOTS || anchor as usize >= self.vocabulary || devices.len() != self.parts.len() {
            return Err(format!("a DSpark draft of {slots} slots after token {anchor}"));
        }
        let history = position.min(RING);
        for at in position - history..position {
            if self.held[at % RING] != at as i64 {
                return Err(format!("DSpark's history lacks position {at} (a draft at {position})"));
            }
        }
        let block = Block { rows: slots as u64, position, history };
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
        let positions: Vec<i32> = (position..position + slots).map(|at| at as i32).collect();
        for (part, device) in self.parts.iter().zip(devices.iter_mut()) {
            device.gpu.write_buffer(&part.stream[0], 0, &stream)?;
            device.gpu.write_buffer(&part.positions, 0, &positions)?;
        }
        // (the attention's pre-mix: copy 0 for the first stage, the previous stage's FFN pre after)
        let mut previous: Option<usize> = None;
        for stage in 0..self.parts[0].stages.len() {
            let ffn = 1 + stage % 2;
            for (part, device) in self.parts.iter().zip(devices.iter_mut()) {
                let pre = previous.map_or(part.first, |set| part.coefficients[set][0]);
                part.attention(device.gpu, stage, pre, &block)?;
            }
            self.sum(devices, 0, &block)?;
            for (part, device) in self.parts.iter().zip(devices.iter_mut()) {
                part.ffn(device.gpu, stage, ffn, &block)?;
            }
            self.sum(devices, 1, &block)?;
            for (part, device) in self.parts.iter().zip(devices.iter_mut()) {
                part.expand(device.gpu, ffn, &block)?;
            }
            previous = Some(ffn);
        }
        // the last FFN's pre-mix collapses the block (the confidence head reads it un-normed); each
        // device's rows of the logits, gathered on the first
        let last = previous.ok_or("DSpark has no stages")?;
        let (vocabulary, rows) = (self.vocabulary, block.rows);
        for (index, (part, device)) in self.parts.iter().zip(devices.iter_mut()).enumerate() {
            let gpu = &mut *device.gpu;
            gpu.hc_collapse(part.stream[0].base(), part.coefficients[last][0].base(), rows, HC as u64, WIDTH as u64, part.collapsed.base())?;
            gpu.rms_norm(part.collapsed.base(), &part.output_norm, rows, WIDTH as u64, NORM_EPS, part.x.base())?;
            let (output, stride) = if index == 0 { (self.logits.at(device.head_first), vocabulary) } else { (part.head_out.base(), 0) };
            gpu.linear(&device.head, device.head_kind, device.head_rows as u64, WIDTH as u64, 0, part.x.base(), 0, rows, output, stride as u64,
                       &part.scratch)?;
        }
        let (first, rest) = devices.split_at_mut(1);
        for (part, device) in self.parts[1..].iter().zip(rest.iter_mut()) {
            for slot in 0..slots {
                device.gpu.copy_to(part.head_out.at(slot * device.head_rows), first[0].gpu, self.logits.at(slot * vocabulary + device.head_first),
                                   (device.head_rows * 4) as u64)?;
            }
        }
        let gpu = &mut *first[0].gpu;
        gpu.dspark_head(self.logits.base(), self.parts[0].collapsed.base(), &self.markov[0], &self.markov[1], &self.confidence, rows,
                        vocabulary as u64, MARKOV as u64, WIDTH as u64, anchor, 0.0, 0, self.corrected.base(), self.tokens.base(),
                        self.confidences.base(), self.count.base(), &self.parts[0].scratch)?;
        let (mut tokens, mut confidences, mut count) = (vec![0i32; slots], vec![0f32; slots], [0u32; 1]);
        gpu.read_buffer(&self.tokens, 0, &mut tokens)?;
        gpu.read_buffer(&self.confidences, 0, &mut confidences)?;
        gpu.read_buffer(&self.count, 0, &mut count)?;
        let kept = (count[0] as usize).min(slots);
        Ok(Draft { tokens: tokens[..kept].iter().map(|&token| token as u32).collect(), confidences: confidences[..kept].to_vec() })
    }
}

/// How many of a draft's tokens a round verifies: the most expected tokens per millisecond, by
/// DSpark's confidences calibrated against teacher-forced match rates ([`CALIBRATION`]), a target
/// pass's cost by its rows and the draft's. After [`PAUSE_AFTER`] rounds in a row that verify
/// none it stops drafting for [`PAUSE_ROUNDS`] rounds (the draft's cost is paid either way).
pub struct Policy {
    /// A target pass's milliseconds by its rows (1, 2, ...): a first guess, then running means of
    /// [`Policy::passed`]'s (the drafter's own memory makes every pass dearer than without it).
    verify_ms: Vec<f64>,
    /// A draft's milliseconds (a running mean of [`Policy::drafted`]'s).
    draft_ms: f64,
    zeros: usize,
    paused: usize,
}

/// Rounds in a row that verify no draft before a pause, and the rounds a pause skips.
pub const PAUSE_AFTER: usize = 4;
pub const PAUSE_ROUNDS: usize = 16;

/// Per slot, the rate at which a draft's token matches the target's greedy one, given that every
/// token before it did, at DSpark's confidence 0, 0.1, ... 1 (linear between): a monotone fit of
/// 750 teacher-forced blocks of three on the six probe prompts (ds41_quality --dspark
/// --show-drafts; a fit on three prompts gives the others' rate within 3%). The fourth slot
/// takes the third's.
const CALIBRATION: [[f64; 11]; MAX_SLOTS] = [
    [0.0, 0.0, 0.001, 0.25, 0.344, 0.533, 0.562, 0.672, 0.755, 0.951, 1.0],
    [0.0, 0.0, 0.072, 0.227, 0.394, 0.51, 0.605, 0.758, 0.808, 0.967, 1.0],
    [0.0, 0.0, 0.06, 0.303, 0.414, 0.54, 0.708, 0.8, 0.856, 1.0, 1.0],
    [0.0, 0.0, 0.06, 0.303, 0.414, 0.54, 0.708, 0.8, 0.856, 1.0, 1.0],
];

fn calibrated(slot: usize, confidence: f32) -> f64 {
    let at = (confidence.clamp(0.0, 1.0) as f64) * 10.0;
    let (low, high) = (at.floor() as usize, at.ceil() as usize);
    let table = &CALIBRATION[slot.min(MAX_SLOTS - 1)];
    table[low] + (table[high] - table[low]) * (at - low as f64)
}

impl Policy {
    /// With a target pass's milliseconds by its rows and a first guess at a draft's.
    pub fn new(verify_ms: &[f64], draft_ms: f64) -> Policy {
        Policy { verify_ms: verify_ms.to_vec(), draft_ms, zeros: 0, paused: 0 }
    }

    /// Whether this round drafts (not during a pause).
    pub fn drafting(&mut self) -> bool {
        if self.paused > 0 {
            self.paused -= 1;
            return false;
        }
        true
    }

    /// A draft took `ms`.
    pub fn drafted(&mut self, ms: f64) {
        self.draft_ms = 0.8 * self.draft_ms + 0.2 * ms;
    }

    /// A target pass of `rows` rows took `ms`.
    pub fn passed(&mut self, rows: usize, ms: f64) {
        if let Some(mean) = rows.checked_sub(1).and_then(|at| self.verify_ms.get_mut(at)) {
            *mean = 0.9 * *mean + 0.1 * ms;
        }
    }

    /// A new context: no pause.
    pub fn reset(&mut self) {
        (self.zeros, self.paused) = (0, 0);
    }

    /// The draft's tokens to verify (0 to `most`, at most a pass's rows less one): the width with
    /// the most expected tokens (the matched prefix and the target's next) per millisecond.
    pub fn width(&mut self, confidences: &[f32], most: usize) -> usize {
        let (mut reach, mut expected) = (1.0, 1.0);
        let mut best = (0, expected / (self.verify_ms[0] + self.draft_ms));
        let most = most.min(self.verify_ms.len() - 1);
        for (slot, &confidence) in confidences.iter().enumerate().take(most) {
            reach *= calibrated(slot, confidence);
            expected += reach;
            let rate = expected / (self.verify_ms[slot + 1] + self.draft_ms);
            if rate > best.1 {
                best = (slot + 1, rate);
            }
        }
        if best.0 == 0 {
            self.zeros += 1;
            if self.zeros >= PAUSE_AFTER {
                (self.zeros, self.paused) = (0, PAUSE_ROUNDS);
            }
        } else {
            self.zeros = 0;
        }
        best.0
    }
}
