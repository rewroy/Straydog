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

use std::collections::HashSet;
use std::ffi::c_void;

use super::engine::{columns, Beside, TAP_LAYERS, TAP_ROWS};
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
/// Each stage's routed experts by how often DSpark's drafts routed to them, most first: route dumps
/// of nine prompts (six probes, an essay, a tool call, code), generated and teacher forced, 26K
/// routes a stage (2026-10-25). A stage uses few: 98% of its routes go to its first 87, 50 and 29.
const RANKING: [[u8; EXPERTS]; 3] = [
    [97, 43, 36, 85, 8, 121, 92, 19, 93, 95, 109, 33, 61, 17, 72, 45, 73, 5, 101, 3, 110, 47, 71, 24, 124, 27, 67, 69, 117, 42, 68, 96, 14, 59, 87, 23, 39, 65, 113, 77, 115, 26, 15, 76, 51, 34, 58, 6, 13, 75, 123, 100, 31, 63, 62, 120, 57, 83, 49, 46, 35, 60, 21, 119, 66, 112, 64, 125, 70, 4, 99, 108, 50, 111, 53, 118, 89, 82, 98, 122, 20, 22, 79, 48, 28, 114, 106, 78, 90, 11, 102, 107, 54, 74, 86, 7, 38, 105, 104, 2, 88, 41, 52, 126, 84, 16, 30, 56, 81, 80, 32, 37, 29, 0, 1, 9, 10, 12, 18, 25, 40, 44, 55, 91, 94, 103, 116, 127],
    [121, 109, 61, 98, 78, 113, 126, 56, 58, 82, 71, 26, 29, 114, 5, 49, 24, 32, 10, 35, 79, 45, 100, 77, 53, 122, 66, 52, 74, 42, 76, 90, 97, 111, 39, 123, 63, 27, 21, 89, 28, 3, 62, 36, 22, 1, 18, 102, 115, 31, 25, 60, 116, 55, 73, 37, 7, 11, 125, 101, 68, 64, 118, 107, 110, 41, 19, 46, 95, 80, 96, 23, 47, 120, 30, 6, 124, 57, 51, 65, 43, 0, 2, 4, 8, 9, 12, 13, 14, 15, 16, 17, 20, 33, 34, 38, 40, 44, 48, 50, 54, 59, 67, 69, 70, 72, 75, 81, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 99, 103, 104, 105, 106, 108, 112, 117, 119, 127],
    [19, 58, 97, 98, 81, 127, 103, 16, 9, 17, 78, 38, 102, 12, 116, 6, 52, 101, 39, 83, 53, 27, 118, 45, 122, 88, 21, 120, 84, 67, 96, 93, 47, 110, 14, 51, 111, 61, 13, 80, 54, 85, 40, 15, 82, 49, 59, 77, 11, 66, 79, 94, 5, 91, 119, 112, 0, 1, 2, 3, 4, 7, 8, 10, 18, 20, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 46, 48, 50, 55, 56, 57, 60, 62, 63, 64, 65, 68, 69, 70, 71, 72, 73, 74, 75, 76, 86, 87, 89, 90, 92, 95, 99, 100, 104, 105, 106, 107, 108, 109, 113, 114, 115, 117, 121, 123, 124, 125, 126]
];

/// The routed experts each stage keeps resident unless told otherwise: their first 87, 50 and 29 of
/// [`RANKING`], the experts of 98% of its routes. Of the 7.2 GB of routed experts that is 3.1 GB, and
/// DSpark's acceptance is what it is with all of them (1.554 against 1.562 tokens of a draft of three
/// accepted, teacher forced); a decode with the drafter ran 11% faster for the slots it freed.
pub const RESIDENT: [usize; 3] = [87, 50, 29];

/// How many of each stage's routed experts are kept: [`RESIDENT`], or with DS41_DSPARK_RESIDENT=A,B,C
/// each stage's first A, B, C of [`RANKING`], or `all` for all of them.
pub fn resident_counts() -> Result<[usize; 3]> {
    parse_counts(std::env::var_os("DS41_DSPARK_RESIDENT").as_deref().map(|value| value.to_string_lossy()).as_deref())
}

fn parse_counts(setting: Option<&str>) -> Result<[usize; 3]> {
    let Some(setting) = setting else { return Ok(RESIDENT) };
    if setting.trim() == "all" {
        return Ok([EXPERTS; 3]);
    }
    let counts: Vec<usize> = setting.split(',').map(|count| count.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>().map_err(|_| "DS41_DSPARK_RESIDENT takes three counts, A,B,C, or all")?;
    match counts[..] {
        [a, b, c] if counts.iter().all(|count| (USED..=EXPERTS).contains(count)) => Ok([a, b, c]),
        _ => Err(format!("DS41_DSPARK_RESIDENT takes three counts from {USED} to {EXPERTS}, or all")),
    }
}

/// The routed experts a stage keeps resident (`count` of them), in slot order: its first `count` of
/// [`RANKING`]. A draft routes among those alone (the router's bias keeps the others out of its
/// top 3), so the drafter takes the target's expert store only their bytes.
fn resident(stage: usize, count: usize) -> Vec<usize> {
    RANKING[stage][..count].iter().map(|&expert| expert as usize).collect()
}
const SCALE: f32 = 1.5;
const FEATURES: usize = 3 * WIDTH;
/// The drafter's graph keys on a device: this, the part of a draft (a stage's attention, routes and
/// experts, then the head and the Markov head) << 8, the block's slots (apart from the target's,
/// layer << 8 and below).
const GRAPHS: u64 = 0xd5 << 56;

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
    /// Each stage's routes (kept apart, read back by [`Drafter::routes`]).
    ids: [Buffer; 3],
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
    /// The block's anchor token (i32), for the Markov head's graph.
    anchor: Buffer,
    vocabulary: usize,
    /// The positions the rings hold.
    held: History,
    /// The graph keys whose parts have run once (the next time records them).
    ran: HashSet<u64>,
    /// The target's token embedding rows, decoded on the host.
    embed_kind: u32,
}

/// A draft: its tokens and DSpark's confidence in each.
#[derive(Debug, Clone, PartialEq)]
pub struct Draft {
    pub tokens: Vec<u32>,
    pub confidences: Vec<f32>,
}

/// DSpark's rings of injected features as a checkpoint kept them ([`Drafter::save_rings`]).
pub struct Rings {
    /// Per device, per stage, a ring's [`RING`] rows.
    rings: Vec<Vec<f32>>,
    held: History,
}

/// Which positions the rings hold: per ring row, the position injected there (-1: none).
#[derive(Clone)]
struct History(Vec<i64>);

impl History {
    fn new() -> History {
        History(vec![-1; RING])
    }

    /// The features of `position` were injected.
    fn note(&mut self, position: usize) {
        self.0[position % RING] = position as i64;
    }

    /// Every position from `position` on is forgotten (a rewind of the target to `position`).
    fn forget_from(&mut self, position: usize) {
        for held in &mut self.0 {
            if *held >= position as i64 {
                *held = -1;
            }
        }
    }

    /// The first of the last [`RING`] positions before `position` (or of all of them) that the
    /// rings lack: what a draft at `position` reads is there when it is none.
    fn lacking(&self, position: usize) -> Option<usize> {
        (position - position.min(RING)..position).find(|&at| self.0[at % RING] != at as i64)
    }
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

/// This device's rows (`first..first + ffn` of the width) of each of `experts` as slots, in that
/// order: of gate and up those rows, of down those columns.
fn expert_slots(gpu: &mut Backend, [gate, up, down]: &[Tensor<'_>; 3], experts: &[usize], first: usize, ffn: usize) -> Result<(Buffer, Slot)> {
    let (width, hidden) = (gate.dims[1], down.dims[1]);
    let slot = |expert: usize| -> Result<Vec<u8>> {
        let down = columns(down.kind, hidden, width, rows(down, expert * hidden, hidden)?, first, ffn)?;
        Ok([rows(gate, expert * width + first, ffn)?, rows(up, expert * width + first, ffn)?, down.as_slice()].concat())
    };
    let gate_bytes = formats::matrix_bytes(gate.kind, ffn, gate.dims[0]).ok_or("unsupported expert type")? as u64;
    let up_bytes = formats::matrix_bytes(up.kind, ffn, up.dims[0]).ok_or("unsupported expert type")? as u64;
    let first_slot = slot(experts[0])?;
    let bytes = first_slot.len() as u64;
    let buffer = gpu.buffer(bytes * experts.len() as u64)?;
    gpu.write_buffer(&buffer, 0, &first_slot)?;
    for (index, &expert) in experts.iter().enumerate().skip(1) {
        gpu.write_buffer(&buffer, index as u64 * bytes, &slot(expert)?)?;
    }
    Ok((buffer, Slot { bytes, offsets: [0, gate_bytes, gate_bytes + up_bytes] }))
}

/// `ops` on `gpu`: from the graph under `key` when there is one, recorded into it when `key` is given
/// and its part ran before (`ran`: the first run sets up what a recording may not), else as they are.
fn replay(gpu: &mut Backend, key: Option<u64>, ran: bool, ops: impl FnOnce(&mut Backend) -> Result<()>) -> Result<()> {
    let Some(key) = key.filter(|_| ran) else { return ops(gpu) };
    if gpu.graph_run(key)? {
        return Ok(());
    }
    gpu.graph_begin()?;
    let result = ops(gpu);
    gpu.graph_end(key, result.is_ok())?;
    result?;
    if gpu.graph_run(key)? { Ok(()) } else { Err("a recorded DSpark graph went missing".into()) }
}

impl Part {
    /// Load this device's (`index` of `count`) part of DSpark.
    fn load(gpu: &mut Backend, draft: &impl Tensors, index: usize, count: usize, head_rows: usize, counts: [usize; 3]) -> Result<Part> {
        let (heads, groups, ffn) = (HEADS / count, GROUPS / count, FFN / count);
        let mut stages = Vec::new();
        let (mut experts, mut shared_expert) = (None, None);
        for (stage, &resident_count) in counts.iter().enumerate() {
            let name = |suffix: &str| format!("blk.{stage}.{suffix}");
            let t = |suffix: &str, kind: u32, dims: &[usize]| tensor(draft, &name(suffix), kind, dims);
            let whole = |gpu: &mut Backend, suffix: &str, kind: u32, dims: &[usize]| upload(gpu, draft, &name(suffix), kind, dims);
            let routed = [t("ffn_gate_exps.weight", 39, &[WIDTH, FFN, EXPERTS])?, t("ffn_up_exps.weight", 39, &[WIDTH, FFN, EXPERTS])?,
                          t("ffn_down_exps.weight", 39, &[FFN, WIDTH, EXPERTS])?];
            let kept = resident(stage, resident_count);
            let (slots, slot) = expert_slots(gpu, &routed, &kept, index * ffn, ffn)?;
            let table = gpu.buffer(EXPERTS as u64 * 8)?;
            // (an expert not kept is never routed to: its entry is the first slot's)
            let mut bases = vec![slots.base() as u64; EXPERTS];
            for (at, &expert) in kept.iter().enumerate() {
                bases[expert] = slots.base() as u64 + at as u64 * slot.bytes;
            }
            gpu.expert_table(&table, &bases, slot.bytes)?;
            let shared = [t("ffn_gate_shexp.weight", 8, &[WIDTH, FFN])?, t("ffn_up_shexp.weight", 8, &[WIDTH, FFN])?,
                          t("ffn_down_shexp.weight", 8, &[FFN, WIDTH])?];
            let (shared, shared_slot) = expert_slots(gpu, &shared, &[0], index * ffn, ffn)?;
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
                bias: {
                    // (an expert not kept out of every top 3: the bias steers the choice alone)
                    let data = tensor(draft, &name("exp_probs_b.bias"), 0, &[EXPERTS])?.data;
                    let mut bias: Vec<f32> = data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                    for (expert, value) in bias.iter_mut().enumerate() {
                        if !kept.contains(&expert) {
                            *value = -1e30;
                        }
                    }
                    gpu.upload_bytes(&bias.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<u8>>())?
                },
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
            ids: [floats(gpu, s * USED)?, floats(gpu, s * USED)?, floats(gpu, s * USED)?],
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
        // (the anchor's position read from the block's positions, as a replayed graph needs it)
        gpu.dspark_attention(self.q.base(), &s.ring, self.kv.base(), &s.sinks, self.heads as u64, rows, block.position as u32,
                             Some(self.positions.base().cast_const()), block.history as u64, self.attended.base(), &self.scratch)?;
        gpu.rope(self.attended.base(), rows, self.heads as u64, HEAD_DIM as u64, ROPE as u64, &positions, &self.frequencies, true)?;
        let group_width = HEADS / GROUPS * HEAD_DIM;
        let projections: Vec<Linear> = (0..self.groups).map(|g| Linear {
            weights: s.out_a, kind: 8, rows: OUT_RANK as u64, cols: group_width as u64, first_row: (g * OUT_RANK) as u64,
            input: self.attended.at(g * group_width) as *const c_void, output: self.low.at(g * OUT_RANK),
        }).collect();
        gpu.linears(&projections, rows, q_width as u64, (self.groups * OUT_RANK) as u64, &self.scratch)?;
        self.linear(gpu, &s.out_b, 8, WIDTH, self.groups * OUT_RANK, self.low.base(), self.target(0), block)
    }

    /// A stage's FFN up to its routes, after the attention's sum: the stream expanded, the FFN's
    /// coefficients (set `ffn`), collapsed by the attention's pre-mix, normed, routed.
    fn route(&self, gpu: &mut Backend, stage: usize, ffn: usize, block: &Block) -> Result<()> {
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
                  self.ids[stage].base(), self.weights.base())
    }

    /// A stage's experts after its routes: this device's rows of the routed (through the stage's
    /// table) and shared experts into its part of the experts' sum.
    fn experts(&self, gpu: &mut Backend, stage: usize, block: &Block) -> Result<()> {
        let (s, rows, w) = (&self.stages[stage], block.rows, WIDTH as u64);
        let (experts, shared) = (self.experts, self.shared_expert);
        gpu.experts_table(&s.table, EXPERTS as u64, self.ids[stage].base(), experts.bytes, experts.offsets, [39; 3], w, self.ffn as u64, rows,
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

/// Whether DSpark splits among `devices` devices as the target does (heads, output groups and
/// expert width in equal shares).
pub fn splits(devices: usize) -> bool {
    devices > 0 && HEADS.is_multiple_of(devices) && GROUPS.is_multiple_of(devices) && FFN.is_multiple_of(32 * devices)
}

/// Bytes of VRAM DSpark takes on device `index` of `devices` ([`Drafter::load`], with `resident`
/// experts kept in each stage and the target's head of `vocabulary` rows split evenly): the
/// weights it cuts for the device, its experts' slots, rings and work buffers, the first device's
/// Markov head and logits too, and the target's feature tap ([`TAP_ROWS`]). A stage's graphs and
/// the runtime's own memory are the placement's reserve. Placement counts this before it sizes
/// the expert store; a test holds it to what a load allocates.
pub fn footprint(index: usize, devices: usize, vocabulary: usize, resident: [usize; 3]) -> u64 {
    let bytes = |kind: u32, rows: usize, cols: usize| formats::matrix_bytes(kind, rows, cols).expect("DSpark's types") as u64;
    let floats = |count: usize| 4 * count as u64;
    let (heads, groups, ffn, s) = (HEADS / devices, GROUPS / devices, FFN / devices, MAX_SLOTS);
    let head_rows = vocabulary / devices;
    let slot = 2 * bytes(39, ffn, WIDTH) + bytes(39, WIDTH, ffn);
    let shared = 2 * bytes(8, ffn, WIDTH) + bytes(8, WIDTH, ffn);
    let mut total = 0;
    for kept in resident {
        total += bytes(8, Q_RANK, WIDTH) + bytes(8, heads * HEAD_DIM, Q_RANK) + bytes(8, HEAD_DIM, WIDTH)
            + bytes(8, groups * OUT_RANK, HEAD_DIM * HEADS / GROUPS) + bytes(8, WIDTH, groups * OUT_RANK);
        total += floats(2 * WIDTH + Q_RANK + HEAD_DIM + heads + 2 * (HC * WIDTH * MIXES + MIXES + 3) + WIDTH * EXPERTS + EXPERTS);
        total += EXPERTS as u64 * 8 + kept as u64 * slot + shared + floats(RING * HEAD_DIM);
    }
    let coefficients = 3 * floats(2 * s * HC + s * HC * HC);
    total += bytes(8, WIDTH, FEATURES) + floats(2 * WIDTH + ROPE / 2) + coefficients + floats(s * HC);
    total += floats(2 * s * HC * WIDTH + 5 * s * WIDTH + s * Q_RANK + 2 * s * heads * HEAD_DIM + s * HEAD_DIM + s * groups * OUT_RANK);
    total += floats(s * MIXES + s * EXPERTS + 4 * s * USED + s * USED * ffn + s * head_rows + s + RING + RING * WIDTH) + (64 << 20);
    if index == 0 {
        total += 2 * bytes(30, MARKOV, vocabulary) + floats(WIDTH + MARKOV + 2 * s * vocabulary + 2 * s + 2);
    }
    total + floats(TAP_ROWS * TAP_LAYERS.len() * WIDTH)
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
        let counts = resident_counts()?;
        let mut parts = Vec::new();
        for (index, device) in devices.iter_mut().enumerate() {
            parts.push(Part::load(device.gpu, draft, index, count, device.head_rows, counts)?);
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
            anchor: floats(gpu, 1)?,
            vocabulary,
            held: History::new(),
            ran: HashSet::new(),
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
            self.held.note(position as usize);
        }
        Ok(())
    }

    /// The last draft's routes, for each stage its `slots` rows of experts (read back: not in a
    /// draft's timing).
    pub fn routes(&mut self, devices: &mut [Beside<'_>], slots: usize) -> Result<Vec<Vec<i32>>> {
        let (part, gpu) = (&self.parts[0], &mut *devices[0].gpu);
        let mut all = Vec::with_capacity(part.stages.len());
        for ids in &part.ids[..part.stages.len()] {
            let mut routes = vec![0i32; slots.min(MAX_SLOTS) * USED];
            gpu.read_buffer(ids, 0, &mut routes)?;
            all.push(routes);
        }
        Ok(all)
    }

    /// Forget every injected position from `position` on (a rewind of the target to `position`).
    pub fn forget_from(&mut self, position: usize) {
        self.held.forget_from(position);
    }

    /// Whether a draft at `position` finds the history it reads (the last [`RING`] positions
    /// before it, or all of them) in the rings.
    pub fn has_history(&self, position: usize) -> bool {
        self.held.lacking(position).is_none()
    }

    /// A copy of the rings (every device's, every stage's) and of which positions they hold, for a
    /// checkpoint of the target's sequence ([`Drafter::restore_rings`] puts them back).
    pub fn save_rings(&self, devices: &mut [Beside<'_>]) -> Result<Rings> {
        if devices.len() != self.parts.len() {
            return Err("DSpark's rings saved from other devices than it runs on".into());
        }
        let mut rings = Vec::with_capacity(devices.len() * self.parts[0].stages.len());
        for (part, device) in self.parts.iter().zip(devices.iter_mut()) {
            for stage in &part.stages {
                let mut ring = vec![0f32; RING * HEAD_DIM];
                device.gpu.read_buffer(&stage.ring, 0, &mut ring)?;
                rings.push(ring);
            }
        }
        Ok(Rings { rings, held: self.held.clone() })
    }

    /// Put rings back as [`Drafter::save_rings`] took them: the history of the sequence then.
    pub fn restore_rings(&mut self, devices: &mut [Beside<'_>], saved: &Rings) -> Result<()> {
        let stages = self.parts[0].stages.len();
        if devices.len() != self.parts.len() || saved.rings.len() != devices.len() * stages {
            return Err("DSpark's rings restored to other devices than they were saved from".into());
        }
        for ((part, device), rings) in self.parts.iter().zip(devices.iter_mut()).zip(saved.rings.chunks(stages)) {
            for (stage, ring) in part.stages.iter().zip(rings) {
                device.gpu.write_buffer(&stage.ring, 0, ring)?;
            }
        }
        self.held.clone_from(&saved.held);
        Ok(())
    }

    /// Whether the part under `key` has run before (and so records a graph now), noting that it has.
    fn ran(&mut self, key: Option<u64>) -> bool {
        key.is_some_and(|key| !self.ran.insert(key))
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
        if let Some(at) = self.held.lacking(position) {
            return Err(format!("DSpark's history lacks position {at} (a draft at {position})"));
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
        // A block after a full history repeats its shape: each part between the devices' sums (and a
        // stage's routes apart from its experts) replays from a graph (recorded the second time it
        // runs), reading the anchor's position and token from the devices.
        let full = history == RING;
        let key = |segment: u64| full.then_some(GRAPHS | segment << 8 | slots as u64);
        // (the attention's pre-mix: copy 0 for the first stage, the previous stage's FFN pre after)
        let mut previous: Option<usize> = None;
        for stage in 0..self.parts[0].stages.len() {
            let ffn = 1 + stage % 2;
            let (attention, routes, experts) = (key(3 * stage as u64), key(3 * stage as u64 + 1), key(3 * stage as u64 + 2));
            let ran = self.ran(attention);
            for (part, device) in self.parts.iter().zip(devices.iter_mut()) {
                replay(device.gpu, attention, ran, |gpu| {
                    if let Some(set) = previous {
                        part.expand(gpu, set, &block)?;
                    }
                    let pre = previous.map_or(part.first, |set| part.coefficients[set][0]);
                    part.attention(gpu, stage, pre, &block)
                })?;
            }
            self.sum(devices, 0, &block)?;
            // (the routes and the experts apart: a seam where experts that are not resident would be
            // fetched)
            let ran = self.ran(routes);
            for (part, device) in self.parts.iter().zip(devices.iter_mut()) {
                replay(device.gpu, routes, ran, |gpu| part.route(gpu, stage, ffn, &block))?;
            }
            let ran = self.ran(experts);
            for (part, device) in self.parts.iter().zip(devices.iter_mut()) {
                replay(device.gpu, experts, ran, |gpu| part.experts(gpu, stage, &block))?;
            }
            self.sum(devices, 1, &block)?;
            previous = Some(ffn);
        }
        // the last FFN's pre-mix collapses the block (the confidence head reads it un-normed); each
        // device's rows of the logits, gathered on the first
        let last = previous.ok_or("DSpark has no stages")?;
        let (vocabulary, rows) = (self.vocabulary, block.rows);
        let (head, ran) = (key(9), self.ran(key(9)));
        for (index, (part, device)) in self.parts.iter().zip(devices.iter_mut()).enumerate() {
            let (weights, kind, head_rows, head_first) = (device.head, device.head_kind, device.head_rows, device.head_first);
            let (output, stride) = if index == 0 { (self.logits.at(head_first), vocabulary) } else { (part.head_out.base(), 0) };
            replay(device.gpu, head, ran, |gpu| {
                part.expand(gpu, last, &block)?;
                gpu.hc_collapse(part.stream[0].base(), part.coefficients[last][0].base(), rows, HC as u64, WIDTH as u64, part.collapsed.base())?;
                gpu.rms_norm(part.collapsed.base(), &part.output_norm, rows, WIDTH as u64, NORM_EPS, part.x.base())?;
                gpu.linear(&weights, kind, head_rows as u64, WIDTH as u64, 0, part.x.base(), 0, rows, output, stride as u64, &part.scratch)
            })?;
        }
        let (first, rest) = devices.split_at_mut(1);
        for (part, device) in self.parts[1..].iter().zip(rest.iter_mut()) {
            for slot in 0..slots {
                device.gpu.copy_to(part.head_out.at(slot * device.head_rows), first[0].gpu, self.logits.at(slot * vocabulary + device.head_first),
                                   (device.head_rows * 4) as u64)?;
            }
        }
        let gpu = &mut *first[0].gpu;
        gpu.write_buffer(&self.anchor, 0, &[anchor as i32])?;
        let (markov, ran) = (key(10), self.ran(key(10)));
        replay(gpu, markov, ran, |gpu| {
            gpu.dspark_head(self.logits.base(), self.parts[0].collapsed.base(), &self.markov[0], &self.markov[1], &self.confidence, rows,
                            vocabulary as u64, MARKOV as u64, WIDTH as u64, anchor, Some(self.anchor.base().cast_const()), 0.0, 0,
                            self.corrected.base(), self.tokens.base(), self.confidences.base(), self.count.base(), &self.parts[0].scratch)
        })?;
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

/// A target pass's milliseconds on two GPUs by its rows, 1 to 4 (T133), and a draft's: where a
/// [`Policy`] starts before it has seen the passes of the run.
pub const VERIFY_MS: [f64; 4] = [25.4, 41.4, 57.0, 69.5];
pub const DRAFT_MS: f64 = 5.0;

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

impl Default for Policy {
    fn default() -> Policy {
        Policy::new(&VERIFY_MS, DRAFT_MS)
    }
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

    /// A target pass of `rows` rows took `ms` (counted at most twice the mean so far: one pass that
    /// waited on the drive must not make a width look too dear ever to be tried again).
    pub fn passed(&mut self, rows: usize, ms: f64) {
        if let Some(mean) = rows.checked_sub(1).and_then(|at| self.verify_ms.get_mut(at)) {
            *mean = 0.9 * *mean + 0.1 * ms.min(2.0 * *mean);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn noted(positions: std::ops::Range<usize>) -> History {
        let mut history = History::new();
        positions.for_each(|at| history.note(at));
        history
    }

    /// A draft at a position reads the positions before it, the last ring of them: they are all
    /// there only when each was injected and not overwritten or forgotten since.
    #[test]
    fn a_draft_finds_its_history_only_where_every_position_was_injected() {
        let empty = History::new();
        assert_eq!((empty.lacking(0), empty.lacking(1)), (None, Some(0)));
        let some = noted(0..10);
        assert_eq!((some.lacking(10), some.lacking(11)), (None, Some(10)));
        // (past a ring the oldest positions are overwritten, and a draft reads only the last ring)
        let long = noted(0..300);
        assert_eq!((long.lacking(300), long.lacking(301)), (None, Some(300)));
        let wrapped = noted(0..129);
        assert_eq!((wrapped.lacking(129), wrapped.lacking(130)), (None, Some(129)));
        // a position injected a ring ago does not stand for the one that shares its row
        let mut stale = noted(0..128);
        stale.forget_from(100);
        stale.note(228);
        assert_eq!(stale.lacking(229), Some(101));
        assert_eq!(History(vec![5; RING]).lacking(6), Some(0));
    }

    /// A rewind of the target takes back the positions from where it went, and nothing before; a
    /// new sequence takes back all. A position forgotten that had overwritten an older one takes
    /// that one with it.
    #[test]
    fn forgetting_takes_back_the_positions_from_one_on() {
        let mut history = noted(0..100);
        history.forget_from(60);
        assert_eq!((history.lacking(60), history.lacking(61)), (None, Some(60)));
        history.note(60);
        assert_eq!((history.lacking(61), history.lacking(100)), (None, Some(61)));
        history.forget_from(0);
        assert_eq!((history.lacking(0), history.lacking(1), history.lacking(300)), (None, Some(0), Some(172)));
        let mut wrapped = noted(0..140);
        wrapped.forget_from(135);
        assert_eq!((wrapped.lacking(135), wrapped.lacking(140)), (Some(7), Some(135)));
        // forgetting past what was injected changes nothing
        let mut kept = noted(0..10);
        kept.forget_from(10);
        assert_eq!(kept.lacking(10), None);
    }

    #[test]
    fn the_resident_experts_are_a_default_a_list_or_all() {
        assert_eq!(parse_counts(None).unwrap(), [87, 50, 29]);
        assert_eq!(parse_counts(Some("all")).unwrap(), [128; 3]);
        assert_eq!(parse_counts(Some(" 128, 3 ,10")).unwrap(), [128, 3, 10]);
        for wrong in ["", "87,50", "87,50,29,1", "2,50,29", "87,50,129", "a,b,c", "-1,50,29"] {
            assert!(parse_counts(Some(wrong)).is_err(), "{wrong}");
        }
        // every stage ranks each of its experts once, and the default keeps the ones most used
        for ranking in RANKING {
            let mut seen = [false; EXPERTS];
            ranking.iter().for_each(|&expert| seen[expert as usize] = true);
            assert!(seen.iter().all(|&expert| expert));
        }
        assert_eq!(resident(2, 29).len(), 29);
        assert_eq!(resident(0, 128).len(), EXPERTS);
        assert_eq!(&resident(1, 3), &[121, 109, 61]);
    }

    /// The VRAM placement counts for DSpark: all its experts and one device is the file's size and
    /// a little more (its work buffers and the target's feature tap); a device of two holds half
    /// the split weights; the resident set takes the experts it leaves out off, each of
    /// a device's share 3 x 1152 rows of MXFP4 blocks.
    #[test]
    fn the_drafters_footprint_is_its_file_its_experts_kept_and_its_buffers() {
        let file = crate::bundle::dspark_source("GSQ-RCO-IQ2_XS").unwrap().expect("a pinned DSpark").asset.bytes;
        let vocabulary = 129_280;
        let whole = footprint(0, 1, vocabulary, [EXPERTS; 3]);
        assert!(whole > file && whole < file + (100 << 20), "{whole} for a file of {file}");
        let slot = 3 * 1152 * (5120 / 32 * 17) as u64;
        let (all, kept) = (footprint(0, 2, vocabulary, [EXPERTS; 3]), footprint(0, 2, vocabulary, RESIDENT));
        assert_eq!(all - kept, (3 * 128 - 166) * slot);
        // (the first device also holds the Markov heads and the logits)
        let second = footprint(1, 2, vocabulary, RESIDENT);
        assert!(kept > second && kept - second > 2 * 256 * vocabulary as u64 * 2, "{kept} {second}");
        assert!(kept < 2200 << 20 && second > 1800 << 20, "{kept} {second}");
        assert!(splits(1) && splits(2) && !splits(0) && !splits(3) && !splits(16));
    }

    fn round(policy: &mut Policy, confidence: f32) -> usize {
        policy.width(&[confidence; 3], 3)
    }

    /// The most expected tokens a millisecond: a confident draft is verified whole, a hopeless one
    /// not at all, and rounds that verify none pause the drafting for a while, which a draft that
    /// is worth it ends.
    #[test]
    fn the_policy_verifies_confident_drafts_and_pauses_after_hopeless_ones() {
        let mut policy = Policy::default();
        assert_eq!(round(&mut policy, 1.0), 3);
        assert_eq!(policy.width(&[1.0, 1.0, 1.0], 2), 2, "no more than the pass takes");
        for _ in 0..PAUSE_AFTER - 1 {
            assert!(policy.drafting());
            assert_eq!(round(&mut policy, 0.1), 0);
        }
        // (a round worth it in between starts the count again)
        assert_eq!(round(&mut policy, 1.0), 3);
        for _ in 0..PAUSE_AFTER {
            assert_eq!(round(&mut policy, 0.1), 0);
        }
        for _ in 0..PAUSE_ROUNDS {
            assert!(!policy.drafting());
        }
        assert!(policy.drafting());
        // a new context ends a pause
        for _ in 0..PAUSE_AFTER {
            round(&mut policy, 0.0);
        }
        assert!(!policy.drafting());
        policy.reset();
        assert!(policy.drafting());
    }

    /// What the policy learns: a draft's cost a running mean; a pass's by its rows, one that waited
    /// (on the drive) counted at most twice the mean; a pass of other rows than it plans for is
    /// no one's.
    #[test]
    fn the_policy_learns_the_costs_it_sees_without_being_misled_by_one_slow_pass() {
        let mut policy = Policy::new(&[20.0, 30.0], 5.0);
        policy.drafted(10.0);
        assert!((policy.draft_ms - 6.0).abs() < 1e-9);
        policy.passed(2, 1000.0);
        assert!((policy.verify_ms[1] - (0.9 * 30.0 + 0.1 * 60.0)).abs() < 1e-9);
        policy.passed(1, 10.0);
        assert!((policy.verify_ms[0] - 19.0).abs() < 1e-9);
        for rows in [0, 3, 9] {
            policy.passed(rows, 1.0);
        }
        assert_eq!(policy.verify_ms.len(), 2);
        assert!((policy.verify_ms[1] - 33.0).abs() < 1e-9);
        // the width is cut to the rows the policy knows
        assert_eq!(policy.width(&[1.0, 1.0, 1.0], 3), 1);
    }

    /// A confidence maps to the rate a slot's token matches, linearly between the table's tenths.
    #[test]
    fn a_confidence_is_calibrated_to_a_match_rate_per_slot() {
        assert_eq!((calibrated(0, 0.0), calibrated(0, 1.0), calibrated(3, 1.0)), (0.0, 1.0, 1.0));
        assert!((calibrated(0, 0.25) - (0.001 + 0.5 * (0.25 - 0.001))).abs() < 1e-9);
        assert_eq!(calibrated(1, 2.0), 1.0, "past 1 is 1");
        assert_eq!(calibrated(9, 0.5), calibrated(3, 0.5), "a slot past the table takes the last");
    }
}
