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
use crate::arch::served::Draw;
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

/// The routed experts each stage keeps resident unless told otherwise: their first 48, 28 and 16 of
/// [`RANKING`], 1.7 of the 7.2 GB of routed experts. The first 87, 50 and 29 (98% of its routes, 3.1 GB)
/// left the acceptance as it was with all of them, but their VRAM is the target's expert slots: served
/// (2026-11-08, both R9700s), Unsloth UD-IQ2_XXS's code went from 51.7 to 55.7 tokens/s and five agent
/// turns from 26.6 to 25.5 s with 48/28/16 (a code draft's tokens kept 81-82% where 85-89%), and
/// UD-IQ1_M's were the same within their spread (code 76.7 and 78.5, the turns 18.9 and 19.3 s).
pub const RESIDENT: [usize; 3] = [48, 28, 16];

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
/// The most rows a matrix product takes as GEMV (F32 sums of the F32 inputs, whatever the device's
/// products): the engine's decode and verify passes, and the pieces features are injected in.
const GEMV_ROWS: usize = 4;
/// The drafter's graph keys on a device: this, the part of a draft (a stage's attention, routes and
/// experts, then the head and the Markov head) << 8, the block's slots (apart from the target's,
/// layer << 8 and below).
const GRAPHS: u64 = 0xd5 << 56;
/// The F32 values hc_step keeps per row of a block (its counter, then the sums of each 64 columns
/// of the stream: the mixes and two sums of squares).
const HC_STEP_VALUES: usize = 1 + WIDTH.div_ceil(64) * (MIXES + 2);

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
    /// The workspace of hc_step ([`HC_STEP_VALUES`] a row).
    hc_workspace: Buffer,
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
    /// The logits' columns the other devices computed (from the second device's first), gathered here:
    /// every slot's, [`MAX_SLOTS`] rows.
    tail: Buffer,
    corrected: Buffer,
    /// The head's results, read back whole: its [`MAX_SLOTS`] tokens (i32), their confidences (F32),
    /// the count of them that are kept (u32) and, for drawn tokens, their [`MAX_SLOTS`] probabilities (F32).
    results: Buffer,
    /// The block's anchor token (i32), for the Markov head's graph.
    anchor: Buffer,
    /// How the head's tokens are chosen (four u32: the inverse temperature's bits, 0 for the argmax, and a
    /// 64-bit seed), read on the device, so that the head's graph follows it.
    sample: Buffer,
    vocabulary: usize,
    /// The positions the rings hold.
    held: History,
    /// The graph keys whose parts have run once (the next time records them).
    ran: HashSet<u64>,
    /// The target's token embedding rows, decoded on the host.
    embed_kind: u32,
    /// The embedding of the noise token every slot after the anchor's starts from (decoded once), and
    /// the block's rows as the last draft laid them out (kept for the allocation).
    noise: Vec<f32>,
    stream_rows: Vec<f32>,
}

/// A draft: its tokens and DSpark's confidence in each; for tokens the head drew, each one's probability under the
/// softmax it was drawn from (none for the argmax).
#[derive(Debug, Clone, PartialEq)]
pub struct Draft {
    pub tokens: Vec<u32>,
    pub confidences: Vec<f32>,
    pub probabilities: Vec<f32>,
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
            hc_workspace: floats(gpu, s * HC_STEP_VALUES)?,
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

    /// A stage's attention: the previous stage's FFN output (`previous`, the set of its coefficients)
    /// expanded into the stream, the attention's coefficients from it, the stream collapsed by the
    /// previous FFN's pre-mix (copy 0 for the first stage) and normed, this device's heads over the
    /// ring and the block, into this device's part of the attention's sum.
    fn attention(&self, gpu: &mut Backend, stage: usize, previous: Option<usize>, block: &Block) -> Result<()> {
        let (s, rows, (w, hc)) = (&self.stages[stage], block.rows, (WIDTH as u64, HC as u64));
        let [stream, expanded] = self.stream;
        let [pre_a, post_a, comb_a] = self.coefficients[0];
        let pre = previous.map_or(self.first, |set| self.coefficients[set][0]);
        let expansion = previous.map(|set| {
            let [_, post_f, comb_f] = self.coefficients[set];
            (self.out.base().cast_const(), expanded.base().cast_const(), post_f.base().cast_const(), comb_f.base().cast_const())
        });
        gpu.hc_step(expansion, stream.base(), Some((s.hc_attn[0].base().cast_const(), s.hc_attn[1].base().cast_const(),
                    s.hc_attn[2].base().cast_const(), pre_a.base(), post_a.base(), comb_a.base())), SINKHORN, HC_EPS, NORM_EPS,
                    pre.base().cast_const(), s.attn_norm.base().cast_const(), rows, hc, w, &self.hc_workspace, self.x.base(), None)?;
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
                             Some(self.positions.base().cast_const()), block.history as u64, self.attended.base())?;
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
    /// coefficients (set `ffn`), collapsed by the attention's pre-mix and normed, routed.
    fn route(&self, gpu: &mut Backend, stage: usize, ffn: usize, block: &Block) -> Result<()> {
        let (s, rows, (w, hc)) = (&self.stages[stage], block.rows, (WIDTH as u64, HC as u64));
        let [stream, expanded] = self.stream;
        let [pre_a, post_a, comb_a] = self.coefficients[0];
        let [pre_f, post_f, comb_f] = self.coefficients[ffn];
        let expansion = (self.out.base().cast_const(), stream.base().cast_const(), post_a.base().cast_const(), comb_a.base().cast_const());
        gpu.hc_step(Some(expansion), expanded.base(), Some((s.hc_ffn[0].base().cast_const(), s.hc_ffn[1].base().cast_const(),
                    s.hc_ffn[2].base().cast_const(), pre_f.base(), post_f.base(), comb_f.base())), SINKHORN, HC_EPS, NORM_EPS,
                    pre_a.base().cast_const(), s.ffn_norm.base().cast_const(), rows, hc, w, &self.hc_workspace, self.x.base(), None)?;
        gpu.linear_route(&s.router, 0, EXPERTS as u64, WIDTH as u64, self.x.base(), rows, self.logits_router.base(),
                         Some(s.bias.base().cast_const()), USED as u64, 2, true, SCALE, self.ids[stage].base(), self.weights.base(),
                         &self.scratch)
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
    total += floats(s * HC_STEP_VALUES + s * EXPERTS + 4 * s * USED + s * USED * ffn + s * head_rows + s + RING + RING * WIDTH) + (64 << 20);
    if index == 0 {
        total += 2 * bytes(30, MARKOV, vocabulary) + floats(WIDTH + MARKOV + 2 * s * vocabulary + 2 * s + 2);
        total += floats(s * (vocabulary - head_rows).max(1));
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
        let head_rows = devices[0].head_rows;
        let floats = |gpu: &mut Backend, count: usize| gpu.buffer((count * 4) as u64);
        Ok(Drafter {
            parts,
            markov: [upload(gpu, draft, "markov_w1.weight", 30, &[MARKOV, vocabulary])?,
                     upload(gpu, draft, "markov_w2.weight", 30, &[MARKOV, vocabulary])?],
            confidence: upload(gpu, draft, "conf_proj.weight", 0, &[WIDTH + MARKOV, 1])?,
            logits: floats(gpu, MAX_SLOTS * vocabulary)?,
            tail: floats(gpu, MAX_SLOTS * (vocabulary - head_rows).max(1))?,
            corrected: floats(gpu, MAX_SLOTS * vocabulary)?,
            results: floats(gpu, 3 * MAX_SLOTS + 1)?,
            anchor: floats(gpu, 1)?,
            sample: floats(gpu, 4)?,
            vocabulary,
            held: History::new(),
            ran: HashSet::new(),
            embed_kind: target.tensor("token_embd.weight")?.kind,
            noise: Vec::new(),
            stream_rows: Vec::new(),
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
            // In pieces of at most GEMV_ROWS rows, in order (the last row to a ring row wins, as in one call):
            // a longer one's encoder is a GEMM, and a GEMM on a device set to FP16 products (what a
            // prompt pass of 32 rows or more leaves it at) casts the features to FP16, where the rows of
            // the sequence's attention sinks (1e5 and more, FP16 ends at 65504) become infinite and
            // their ring rows NaN: drafts then go wrong for as long as such a row is in the window.
            for (piece, rows) in positions.chunks(GEMV_ROWS).enumerate() {
                let first = piece * GEMV_ROWS;
                let inject_positions = part.inject_positions.slice((first * 4) as u64, (rows.len() * 4) as u64);
                gpu.dspark_inject(device.tap.at(first * FEATURES), &inject_positions, rows.len() as u64, &part.fc, &part.encoder_norm, &stages,
                                  &part.frequencies, part.encoded.base(), &part.scratch)?;
            }
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

    /// The head's logits (corrected by the Markov head) of `slot` of the last draft, the vocabulary wide, read into `into`: the
    /// row a drawn draft's token was drawn from the softmax of (they stay as the last draft left them until the next).
    pub fn drawn_row(&mut self, devices: &mut [Beside<'_>], slot: usize, into: &mut Vec<f32>) -> Result<()> {
        if slot >= MAX_SLOTS {
            return Err(format!("a DSpark draft has no slot {slot}"));
        }
        let gpu = &mut *devices[0].gpu;
        into.resize(self.vocabulary, 0.0);
        gpu.read_buffer(&self.corrected, slot * self.vocabulary, into)
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
        self.draft_drawn(devices, embed, anchor, position, slots, None)
    }

    /// [`Drafter::draft`], its tokens drawn from the softmax of the head's logits at the temperature `draw` gives
    /// instead of the argmax when it is given (each slot's Markov features then follow the token drawn before it);
    /// [`Draft::probabilities`] has the probabilities of the draws and [`Drafter::drawn_row`] the logits they were from.
    pub fn draft_drawn(&mut self, devices: &mut [Beside<'_>], embed: &impl Tensors, anchor: u32, position: usize, slots: usize,
                       draw: Option<Draw>) -> Result<Draft> {
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
        let decode = |token: usize, row: &mut Vec<f32>| {
            row.clear();
            formats::decode_into(self.embed_kind, &table.data[token * bytes..(token + 1) * bytes], row)
        };
        if self.noise.is_empty() {
            decode(NOISE as usize, &mut self.noise)?;
        }
        let (mut stream, mut row) = (std::mem::take(&mut self.stream_rows), Vec::with_capacity(WIDTH));
        stream.clear();
        decode(anchor as usize, &mut row)?;
        for _ in 0..HC {
            stream.extend_from_slice(&row);
        }
        for _ in 1..slots {
            for _ in 0..HC {
                stream.extend_from_slice(&self.noise);
            }
        }
        let positions: Vec<i32> = (position..position + slots).map(|at| at as i32).collect();
        for (part, device) in self.parts.iter().zip(devices.iter_mut()) {
            device.gpu.write_buffer(&part.stream[0], 0, &stream)?;
            device.gpu.write_buffer(&part.positions, 0, &positions)?;
        }
        self.stream_rows = stream;
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
                replay(device.gpu, attention, ran, |gpu| part.attention(gpu, stage, previous, &block))?;
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
        // The others' columns come over whole, into a buffer of their own that the Markov head reads
        // beside the first device's (one copy for the second device alone: its rows are the columns
        // after the first's, slot after slot).
        let (first, rest) = devices.split_at_mut(1);
        let split = rest.first().map_or(vocabulary, |device| device.head_first);
        match rest {
            [only] => only.gpu.copy_to(self.parts[1].head_out.base(), first[0].gpu, self.tail.base(), (slots * only.head_rows * 4) as u64)?,
            _ => {
                for (part, device) in self.parts[1..].iter().zip(rest.iter_mut()) {
                    for slot in 0..slots {
                        device.gpu.copy_to(part.head_out.at(slot * device.head_rows), first[0].gpu,
                                           self.tail.at(slot * (vocabulary - split) + device.head_first - split), (device.head_rows * 4) as u64)?;
                    }
                }
            }
        }
        let tail = (split < vocabulary).then(|| (self.tail.base().cast_const(), split as u64));
        let gpu = &mut *first[0].gpu;
        gpu.write_buffer(&self.anchor, 0, &[anchor as i32])?;
        // (the head's graph is recorded once, with the probabilities in it: they are left unread for a draft that draws nothing)
        let drawn = draw.is_some();
        let draw = draw.map_or([0u32; 4], |draw| [draw.inverse_temperature.to_bits(), draw.seed as u32, (draw.seed >> 32) as u32, 0]);
        gpu.write_buffer(&self.sample, 0, &draw)?;
        let (markov, ran) = (key(10), self.ran(key(10)));
        replay(gpu, markov, ran, |gpu| {
            gpu.dspark_head(self.logits.base(), tail, self.parts[0].collapsed.base(), &self.markov[0], &self.markov[1], &self.confidence, rows,
                            vocabulary as u64, MARKOV as u64, WIDTH as u64, anchor, Some(self.anchor.base().cast_const()),
                            Some(self.sample.base().cast_const()), 0.0, 0,
                            self.corrected.base(), self.results.at(0), self.results.at(MAX_SLOTS), self.results.at(2 * MAX_SLOTS),
                            Some(self.results.at(2 * MAX_SLOTS + 1)), &self.parts[0].scratch)
        })?;
        let mut results = [0u32; 3 * MAX_SLOTS + 1];
        gpu.read_buffer(&self.results, 0, &mut results)?;
        let kept = (results[2 * MAX_SLOTS] as usize).min(slots);
        Ok(Draft {
            tokens: results[..kept].to_vec(),
            confidences: results[MAX_SLOTS..MAX_SLOTS + kept].iter().map(|&bits| f32::from_bits(bits)).collect(),
            probabilities: if drawn {
                results[2 * MAX_SLOTS + 1..2 * MAX_SLOTS + 1 + kept].iter().map(|&bits| f32::from_bits(bits)).collect()
            } else {
                Vec::new()
            },
        })
    }
}

/// How many of a draft's tokens a round verifies: the most expected tokens per millisecond, by
/// DSpark's confidences calibrated against teacher-forced match rates ([`CALIBRATION`]) and then
/// against the drafts of this run ([`Policy::verified`]: on a prompt the drafter reads badly its
/// confidences promise too much), a target pass's cost by its rows ([`EXTRA_COST`], scaled by what
/// the run's passes cost, times the cost of a one-row pass now, which its one-row passes tell) and
/// the draft's.
///
/// Whether to draft at all is a matter of what drafting has paid so far: a draft costs a few
/// milliseconds whatever the round makes of it, and a round that verifies none, or tokens that cost
/// as much as decoding them one at a time, saves nothing. The policy keeps a running mean of what
/// each round that drafted saved against decoding its tokens without (a draft that was not
/// verified: its cost lost); when that is below nothing it stops drafting for [`PAUSE_ROUNDS`]
/// rounds, then tries again.
pub struct Policy {
    /// A one-row pass's milliseconds now: a running mean of the one-row passes seen (what a pass
    /// costs depends on how many of the experts it reads the stores hold, which changes as a
    /// conversation goes on).
    unit_ms: f64,
    /// How much of [`EXTRA_COST`] the passes of more rows cost now: a running mean of what every
    /// such pass showed, whatever its rows, so the cost of a width not chosen for a while follows
    /// the ones that were (it cannot go stale).
    extra: f64,
    /// A draft's milliseconds (a running mean of [`Policy::drafted`]'s).
    draft_ms: f64,
    /// How many of the drafts calibration expects are accepted, over those that were (a running
    /// ratio, drifting back to 1 while no draft is verified).
    skill: f64,
    /// The last round's width and the accepted drafts calibration expected of it.
    chosen: (usize, f64),
    /// What a round that drafted saved, in milliseconds (a running mean).
    saved: f64,
    /// The last target pass's milliseconds.
    pass_ms: f64,
    paused: usize,
    /// What the rounds since [`Policy::take_stats`] did.
    stats: Stats,
}

/// What a [`Policy`]'s rounds did: the drafts it was asked for and, of those, how many tokens each
/// verified (index 0: none, the draft thrown away).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub runs: usize,
    pub widths: [usize; MAX_SLOTS],
}

/// What a pass of 2, 3 and 4 rows costs over one of a single row, in one-row passes, before the
/// run's own measure ([`Policy::extra`]). Served on two R9700s a verify pass cost 1.45 / 1.8 / 2.7
/// one-row passes on code (30.2 / 37.1 / 56.4 ms over 20.9) and 1.7 / 2.2 / 2.8 on prose (29.9 /
/// 39.0 / 49.8 over 17.8): the fourth row costs most, the second and third less than a row each.
/// The run's passes scale the whole shape, so no width's cost is left to go stale.
const EXTRA_COST: [f64; MAX_SLOTS - 1] = [0.6, 1.1, 1.8];
/// The scale of [`EXTRA_COST`] a run may show (clamped, so one pass that waited on the drive does
/// not make a width look too dear ever to be tried again).
const EXTRA_RANGE: (f64, f64) = (0.4, 2.0);

/// Where a [`Policy`] starts before it has seen the passes of the run: a one-row pass's
/// milliseconds and a draft's.
pub const PASS_MS: f64 = 20.0;
pub const DRAFT_MS: f64 = 2.4;

/// The rounds a pause skips drafting for.
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
        Policy::new(PASS_MS, DRAFT_MS)
    }
}

impl Policy {
    /// With a first guess at a one-row pass's and a draft's milliseconds.
    pub fn new(pass_ms: f64, draft_ms: f64) -> Policy {
        Policy { unit_ms: pass_ms, extra: 1.0, draft_ms, skill: 1.0, chosen: (0, 0.0), saved: 1.0, pass_ms, paused: 0, stats: Stats::default() }
    }

    /// A pass of `rows` rows (1 to 4) in one-row passes now.
    fn cost(&self, rows: usize) -> f64 {
        1.0 + rows.checked_sub(2).map_or(0.0, |at| self.extra * EXTRA_COST[at])
    }

    /// Whether this round drafts: not during a pause, which begins when drafts have not paid.
    pub fn drafting(&mut self) -> bool {
        if self.saved < 0.0 && self.paused == 0 {
            // (the first draft after a pause starts from nothing saved: it must pay or the pause comes back)
            (self.saved, self.paused) = (0.0, PAUSE_ROUNDS);
        }
        if self.paused > 0 {
            self.paused -= 1;
            self.forgive();
            return false;
        }
        true
    }

    /// A draft took `ms`.
    pub fn drafted(&mut self, ms: f64) {
        self.draft_ms = 0.8 * self.draft_ms + 0.2 * ms;
    }

    /// A target pass of `rows` rows took `ms`. A pass of one row tells what one costs now; a pass
    /// of more tells how much of [`EXTRA_COST`] the extra rows cost (whatever its rows, and at most
    /// the range a run may show: one pass that waited on the drive must not make a width look too
    /// dear ever to be tried again). A pass of other rows than the policy plans for is no one's.
    pub fn passed(&mut self, rows: usize, ms: f64) {
        self.pass_ms = ms;
        match rows {
            1 => self.unit_ms = 0.85 * self.unit_ms + 0.15 * ms.min(2.0 * self.unit_ms),
            2..=MAX_SLOTS => {
                let seen = ((ms / self.unit_ms - 1.0) / EXTRA_COST[rows - 2]).clamp(EXTRA_RANGE.0, EXTRA_RANGE.1);
                self.extra = 0.85 * self.extra + 0.15 * seen;
            }
            _ => {}
        }
    }

    /// The last round's draft was verified and `accepted` of its tokens matched: the drafter's
    /// confidences are worth what the drafts of this run say from here on, and the round saved
    /// what decoding its tokens one at a time would have cost over what it did.
    pub fn verified(&mut self, accepted: usize) {
        let (width, expected) = std::mem::take(&mut self.chosen);
        if width == 0 {
            return;
        }
        if expected > 0.05 {
            let seen = (accepted as f64 / expected).min(1.5);
            self.skill += (seen - self.skill) * (0.1 * expected).min(0.25);
        }
        self.save((1 + accepted) as f64 * self.unit_ms - self.pass_ms - self.draft_ms);
    }

    fn save(&mut self, milliseconds: f64) {
        self.saved = 0.9 * self.saved + 0.1 * milliseconds;
    }

    /// A round with no verified draft: what was learned of the drafter's confidences fades
    /// (a prompt where it read badly may be followed by one where it reads well).
    fn forgive(&mut self) {
        self.skill += (1.0 - self.skill) * 0.01;
    }

    /// What the rounds did since the last call.
    pub fn take_stats(&mut self) -> Stats {
        std::mem::take(&mut self.stats)
    }

    /// A new context: no pause, no verdict on the drafter.
    pub fn reset(&mut self) {
        (self.paused, self.skill, self.chosen, self.saved) = (0, 1.0, (0, 0.0), 1.0);
    }

    /// The draft's tokens to verify (0 to `most`, at most a pass's rows less one): the width with
    /// the most expected tokens (the matched prefix and the target's next) per millisecond.
    pub fn width(&mut self, confidences: &[f32], most: usize) -> usize {
        let (mut reach, mut raw, mut expected, mut expected_raw) = (1.0, 1.0, 1.0, 0.0);
        let mut best = (0, expected / (self.unit_ms + self.draft_ms), 0.0);
        for (slot, &confidence) in confidences.iter().enumerate().take(most.min(MAX_SLOTS - 1)) {
            let matched = calibrated(slot, confidence);
            raw *= matched;
            reach *= (matched * self.skill).min(1.0);
            expected += reach;
            expected_raw += raw;
            let rate = expected / (self.unit_ms * self.cost(slot + 2) + self.draft_ms);
            if rate > best.1 {
                best = (slot + 1, rate, expected_raw);
            }
        }
        self.chosen = (best.0, best.2);
        self.stats.runs += 1;
        self.stats.widths[best.0] += 1;
        if best.0 == 0 {
            // (the draft's cost lost)
            self.forgive();
            self.save(-self.draft_ms);
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
        assert_eq!(parse_counts(None).unwrap(), [48, 28, 16]);
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
        let file = crate::bundle::dspark_source(crate::bundle::Id::find(Some(crate::bundle::DEEPSEEK), None, "IQ2_XS").unwrap()).unwrap().expect("a pinned DSpark").asset.bytes;
        let vocabulary = 129_280;
        let whole = footprint(0, 1, vocabulary, [EXPERTS; 3]);
        assert!(whole > file && whole < file + (100 << 20), "{whole} for a file of {file}");
        let slot = 3 * 1152 * (5120 / 32 * 17) as u64;
        let (all, kept) = (footprint(0, 2, vocabulary, [EXPERTS; 3]), footprint(0, 2, vocabulary, RESIDENT));
        assert_eq!(all - kept, (3 * 128 - 92) * slot);
        // (the first device also holds the Markov heads and the logits)
        let second = footprint(1, 2, vocabulary, RESIDENT);
        assert!(kept > second && kept - second > 2 * 256 * vocabulary as u64 * 2, "{kept} {second}");
        assert!(kept < 1500 << 20 && second > 1200 << 20, "{kept} {second}");
        assert!(splits(1) && splits(2) && !splits(0) && !splits(3) && !splits(16));
    }

    fn round(policy: &mut Policy, confidence: f32) -> usize {
        policy.width(&[confidence; 3], 3)
    }

    /// The rounds the drafting is paused for, from now to the next that drafts (a plain pass each).
    fn silence(policy: &mut Policy) -> usize {
        let mut rounds = 0;
        while !policy.drafting() {
            policy.passed(1, PASS_MS);
            rounds += 1;
            assert!(rounds < 1000);
        }
        rounds
    }

    /// The most expected tokens a millisecond: a confident draft is verified (to the width where
    /// another row costs more than the token it is expected to bring), a hopeless one not at all.
    #[test]
    fn the_policy_verifies_confident_drafts_as_wide_as_a_row_pays() {
        let mut policy = Policy::default();
        // (a pass of four rows costs 2.8 one-row passes before the run has shown its own: the third draft pays it)
        assert_eq!(round(&mut policy, 1.0), 3);
        assert_eq!(policy.width(&[1.0, 1.0, 1.0], 2), 2, "no more than the pass takes");
        assert_eq!(policy.width(&[1.0, 1.0, 1.0], 1), 1);
        assert_eq!(round(&mut policy, 0.1), 0);
        // a draft that is likely right in its first token only brings that one
        assert_eq!(policy.width(&[0.95, 0.1, 0.1], 3), 1);
    }

    /// Drafts that save nothing pause the drafting for a while, then one is tried, which pays or
    /// brings the pause back; one that pays does not.
    #[test]
    fn drafts_that_do_not_pay_pause_the_drafting() {
        let mut policy = Policy::default();
        // drafts that bring nothing cost their time: the mean saved goes below nothing
        let mut asked = 0;
        while policy.saved >= 0.0 {
            silence(&mut policy);
            assert_eq!(round(&mut policy, 0.1), 0);
            policy.passed(1, PASS_MS);
            asked += 1;
            assert!(asked < 20);
        }
        assert!((3..8).contains(&asked), "{asked}");
        // the pause, then a draft that brings nothing either: the pause is back
        for _ in 0..3 {
            assert_eq!(silence(&mut policy), PAUSE_ROUNDS);
            assert_eq!(round(&mut policy, 0.1), 0);
            policy.passed(1, PASS_MS);
        }
        // a draft after a pause that pays (three tokens for a pass of four rows and a draft) ends them
        silence(&mut policy);
        assert_eq!(round(&mut policy, 1.0), 3);
        policy.passed(4, 56.0);
        policy.verified(3);
        assert!(policy.drafting() && policy.saved > 0.0);
        // ... and one verified with nothing accepted costs its pass: the pause comes
        assert_eq!(round(&mut policy, 1.0), 3);
        policy.passed(4, 56.0);
        policy.verified(0);
        assert_eq!(silence(&mut policy), PAUSE_ROUNDS);
        policy.reset();
        assert!(policy.drafting());
    }

    /// Every draft the policy is asked about is counted with the width it was verified at, the
    /// hopeless ones as width 0; the count starts over each time it is taken.
    #[test]
    fn the_policy_counts_its_draft_runs_by_width() {
        let mut policy = Policy::default();
        assert_eq!(policy.take_stats(), Stats::default());
        assert_eq!(round(&mut policy, 1.0), 3);
        assert_eq!(round(&mut policy, 0.1), 0);
        assert_eq!(policy.width(&[1.0; 3], 2), 2);
        assert_eq!(policy.take_stats(), Stats { runs: 3, widths: [1, 0, 1, 1] });
        assert_eq!(policy.take_stats(), Stats::default());
    }

    /// What the policy learns of the costs: a draft's a running mean; a one-row pass's from the
    /// passes of one row; the scale of the extra rows' cost from every pass of more, whatever its
    /// rows, at most the range a run may show; a pass of other rows than it plans for is no one's.
    #[test]
    fn the_policy_learns_the_costs_it_sees_without_being_misled_by_one_slow_pass() {
        let mut policy = Policy::new(20.0, 5.0);
        policy.drafted(10.0);
        assert!((policy.draft_ms - 6.0).abs() < 1e-9);
        // (a pass of more rows says nothing of what one row costs)
        policy.passed(2, 1000.0);
        assert_eq!(policy.unit_ms, 20.0);
        assert!((policy.extra - (0.85 + 0.15 * EXTRA_RANGE.1)).abs() < 1e-9, "one slow pass counts as the most a run may show");
        policy.passed(1, 10.0);
        let unit = 0.85 * 20.0 + 0.15 * 10.0;
        assert!((policy.unit_ms - unit).abs() < 1e-9);
        policy.passed(1, 1000.0);
        assert!((policy.unit_ms - (0.85 * unit + 0.15 * 2.0 * unit)).abs() < 1e-9, "a slow one-row pass counts at most twice the mean");
        // (passes of three rows that cost what the policy expects show the scale it has)
        let mut policy = Policy::new(20.0, 5.0);
        for _ in 0..60 {
            policy.passed(3, 20.0 * (1.0 + 0.5 * EXTRA_COST[1]));
        }
        assert!((policy.extra - 0.5).abs() < 0.01, "{}", policy.extra);
        let extra = policy.extra;
        for rows in [0, 5, 9] {
            policy.passed(rows, 1.0);
        }
        assert_eq!((policy.unit_ms, policy.extra), (20.0, extra));
        // (a pass cannot cost less than the range allows)
        for _ in 0..60 {
            policy.passed(2, 1.0);
        }
        assert!((policy.extra - EXTRA_RANGE.0).abs() < 0.01);
    }

    /// Passes that cost little beside their extra rows (served warm: 18.9, 28.7, 34.5 and 40.4 ms
    /// for one to four rows) make a confident draft worth the widest verify; dear ones (a cold
    /// store: four rows 3.6 one-row passes) not.
    #[test]
    fn a_warm_shape_verifies_three_drafts_at_high_confidence_and_a_dear_one_two() {
        let mut policy = Policy::default();
        for _ in 0..40 {
            for (rows, ms) in [(1, 18.9), (2, 28.7), (3, 34.5), (4, 40.4)] {
                policy.passed(rows, ms);
            }
        }
        assert_eq!(policy.width(&[0.9; 3], 3), 3);
        let mut dear = Policy::default();
        for _ in 0..40 {
            for (rows, ms) in [(1, 17.8), (2, 31.0), (3, 41.0), (4, 64.0)] {
                dear.passed(rows, ms);
            }
        }
        assert_eq!(dear.width(&[0.9; 3], 3), 2);
        // (the same confidence, the cost the only difference)
        assert_eq!(policy.width(&[0.9; 3], 3), 3);
    }

    /// A width's cost is not a number of its own that only its passes revise: every pass of more
    /// rows revises the scale of them all. Passes of four rows that were dear make three drafts too
    /// dear, and then they are not made, yet cheap passes of two rows alone bring the fourth row's
    /// cost down until three drafts are made again.
    #[test]
    fn the_cost_of_a_width_not_chosen_does_not_go_stale() {
        let mut policy = Policy::new(20.0, 4.2);
        for _ in 0..20 {
            policy.passed(1, 20.0);
            policy.passed(4, 20.0 * (1.0 + 1.6 * EXTRA_COST[2]));
        }
        assert_eq!(policy.width(&[0.9; 3], 3), 2, "four rows cost too much");
        for _ in 0..40 {
            policy.passed(2, 20.0 * (1.0 + 0.4 * EXTRA_COST[0]));
        }
        assert_eq!(policy.width(&[0.9; 3], 3), 3, "two rows cost what four will");
    }

    /// Rounds of a drafter whose confidence is 0.9 throughout, `accepted` of a draft's tokens right;
    /// passes cost what the policy first expects. The policy after them, and each round's width.
    fn play(rounds: usize, accepted: impl Fn(usize) -> usize) -> (Policy, Vec<usize>) {
        let mut policy = Policy::default();
        let mut widths = Vec::new();
        for _ in 0..rounds {
            if !policy.drafting() {
                policy.passed(1, PASS_MS);
                continue;
            }
            let width = policy.width(&[0.9; 3], 3);
            policy.passed(width + 1, PASS_MS * (1.0 + width.checked_sub(1).map_or(0.0, |at| EXTRA_COST[at])));
            if width > 0 {
                policy.verified(accepted(width));
            }
            widths.push(width);
        }
        (policy, widths)
    }

    /// The drafter's confidences are trusted as far as its drafts bear them out: none accepted, and
    /// drafts are rarer and narrower (a draft that was not worth it costs its time, and the pause);
    /// all accepted, and the skill is above 1 and every draft is verified as wide as it can be worth.
    #[test]
    fn the_policy_trusts_the_confidence_as_far_as_the_drafts_bear_it_out() {
        let (policy, widths) = play(400, |_| 0);
        let drafted = widths.iter().filter(|&&width| width > 0).count();
        assert!(widths[..3].iter().all(|&width| width > 0), "{widths:?}");
        assert!(drafted < 40 && policy.skill < 0.8, "{drafted} drafted, skill {}", policy.skill);
        let (policy, widths) = play(400, |width| width);
        assert!(widths.iter().all(|&width| width == 3), "{widths:?}");
        assert!(policy.skill > 1.0 && policy.skill <= 1.5 && policy.saved > 5.0, "skill {} saved {}", policy.skill, policy.saved);
        // half of the tokens right: a draft costs what it saves, and few are made
        let (policy, widths) = play(400, |width| width / 2);
        assert!(widths.iter().filter(|&&width| width > 0).count() < 40 && policy.skill < 0.9, "skill {}", policy.skill);
        // a new context starts with the drafter trusted again
        let mut policy = policy;
        policy.reset();
        assert_eq!(policy.skill, 1.0);
        assert!(policy.drafting() && policy.width(&[0.9; 3], 3) > 0);
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
