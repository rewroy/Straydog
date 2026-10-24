//! DeepSeek-V4.1 on one GPU or across several, composed from the generic ops: dense weights
//! resident, routed experts in the expert store's tiers (VRAM slots, a host tier, the files;
//! src/experts.rs), and the sequence state a forward pass continues (window rings, compressed
//! rows and index keys, compression tails, the Engram history).
//! Checked against the CPU reference by examples/ds41_gpu and examples/ds41_chat.
//!
//! Across several devices every layer is split (tensor parallel): a device runs its share of the
//! attention heads (its rows of q_b, its sinks, its groups of wo_a and their columns of wo_b), its
//! rows of the head, and its rows of every expert, shared and routed: of gate and up its share of
//! the expert's width, of down its share of the output. Each device's store holds its rows of the
//! experts, so a missing expert comes over every device's bus at once (each a share of it) and the
//! devices' drive reads run side by side; every route runs on every device, so the work is even.
//! The rest is replicated, computed alike on every device. Every device sums the devices' parts
//! in device order after the attention, of the experts' activations (each device's share of the
//! width) and after the experts, so all hold the same values.
//!
//! A pass runs layer by layer over all its tokens (up to `Settings::rows`): a layer's attention a
//! batch of rows at a time, then its experts over all the rows, each routed expert read once for
//! them all (in waves, the next wave's fills behind the one computing). Only the last row's output
//! leaves a pass, and the decoder (the layers from the last kv source on) reaches back only through
//! its sliding windows: of a pass of n rows layer k computes the last 1 + (layers - 1 - k) x
//! (window - 1) and the K/V of the window - 1 rows before them, while the kv source still
//! compresses every row (the decoder's compressed rows and index keys). The state a later pass
//! continues (window rings, compressed rows, their tails and index keys) is that of every row
//! computed; DS41_FULL_DECODER computes every row, for comparisons.
//!
//! A pass of up to [`VERIFY_ROWS`] tokens can be taken back but for its first few ([`Engine::rewind`]):
//! it keeps the counters of the sequence state from before it, and each ratio-2 kv source's
//! compressor rows of its tokens (one of them waits in the source's tail after a rewind). A
//! verify pass ([`Engine::verify`]) is such a pass computing every row in every layer, with every
//! token's logits: those decode gives the tokens one at a time, bit for bit (every op of a few
//! rows computes each row as it would alone), so drafted tokens can be checked and the rejected
//! ones taken back.
//!
//! A decode row can predict the next layer's experts (its router on this layer's input) and
//! prefetch the DS41_PREDICT best (up to 8) from the host tier, DS41_PREDICT_DRIVE also from the
//! drive. It is off: on two GPUs most of the prefetches were never used, and each one cost an
//! upload and, the tiers being exclusive, a write-back over the GPU's bus.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::mpsc::channel;
use std::sync::Arc;

use super::{Gating, Hparams, Rope};
use crate::arch::{Tensor, Tensors};
use crate::experts::{select, taken_up_to, Counts, DeviceExperts, Layout, Reader, Role};
use crate::formats;
use crate::gpu::{Backend, Result};
use crate::moe;
use crate::ops::{self, Buffer, Linear};
use crate::transfer::Stream;

mod vision;
pub use vision::{encode_image_rows, IMAGE_TOKEN};

type P = *const c_void;
type M = *mut c_void;

/// The error of a pass whose cross-device wait gave up. Its results are void but the devices are
/// sound (the next linked pass starts the links over), so a reset engine takes the next request.
pub const VOID_PASS: &str = "a DeepSeek-V4.1 part link's wait gave up: the pass is void";

/// DS41_PROFILE=host+stamps's points of a decode layer, in the order device 0 reaches them.
#[derive(Clone, Copy)]
enum Stamp {
    LayerStart,
    /// The Engram (if any) is in; the attention's hyper-connection step is next.
    AttentionStart,
    /// The hyper-connection step before the attention is done (device 0, recorded with it).
    HcDone,
    /// The attention's output is complete on this device (summed over the part link in decode).
    AttentionDone,
    /// The routes are out (the shared expert's gate and up are next).
    Routed,
    /// The shared expert's gate and up are done.
    SharedDone,
    /// The host has the routes (queued right after its wait).
    RoutesTaken,
    /// The routed experts' first launch.
    ExpertsStart,
    /// The experts VRAM held: their gate and up are done (decode with misses).
    HitsDone,
    /// The missed experts' gate and up are done (they waited for their reads).
    MissesDone,
    /// The activations are summed across the devices.
    ActivationsSummed,
    LayerEnd,
}

impl Stamp {
    const COUNT: usize = 12;
}

/// DS41_PROFILE=host+stamps's points of a decode pass apart from its layers (after them in the
/// buffer): the pass's first launch, and the head's end.
const PASS_STAMPS: usize = 2;

/// The stamps' device buffer (a u64 a point of a layer) and its clock's rate.
struct Stamps {
    buffer: Buffer,
    khz: f64,
    /// The last decode pass's last layer's end and its head's (0: none), for the gap to the next.
    last_end: Option<(u64, u64)>,
}

pub struct Settings {
    /// Longest sequence (positions) the state holds.
    pub context: usize,
    /// Most tokens one forward pass takes (at least the batch). A pass runs layer by layer over all
    /// of them, and its buffers of every row hold them all: the stream and its coefficients, the
    /// attention's and the experts' inputs and outputs, ~0.3 MB a row on each device ([`pass_row_bytes`]).
    pub rows: usize,
    /// Rows the attention and the hyper-connections run at a time within a pass (the window rings
    /// hold this many beyond the window).
    pub batch: usize,
    /// Experts the VRAM cache holds, per device (with experts split across n devices, each holds
    /// n times as many shares of experts in the same memory).
    pub expert_slots: usize,
    /// Bytes of host tier (page-locked memory) for experts VRAM evicts, per device.
    pub host_expert_bytes: u64,
    pub scratch_bytes: u64,
    /// FP16 operands (F32 sums and output) for the products of a pass of at least 32 rows: the
    /// shared/routed FFN GEMMs, and through the devices' GEMM policy every other projection over
    /// its rows (attention, index, router, Engram). Fewer rows stay F32; up to four rows GEMV.
    pub half_gemm: bool,
    /// The layers this engine runs: the first embeds the tokens, the last has the head, and an
    /// engine between takes the stream the one before hands over (see [`forward_split`]).
    pub layers: std::ops::Range<usize>,
}

#[derive(Clone)]
struct Dense {
    buffer: Buffer,
    kind: u32,
    dims: Vec<usize>,
}

/// An expert (or a device's rows of it) laid out as a slot: gate, up and down at byte offsets of
/// one buffer.
#[derive(Clone, Copy)]
struct Slot {
    buffer: Buffer,
    offsets: [u64; 3],
    kinds: [u32; 3],
}

/// A device's rows of every expert: gate and up rows [ffn.0, ffn.0 + ffn.1), down rows
/// [hidden.0, hidden.0 + hidden.1).
#[derive(Clone, Copy)]
struct Rows {
    ffn: (usize, usize),
    hidden: (usize, usize),
}

impl Rows {
    fn of(part: usize, parts: usize, ffn: usize, hidden: usize) -> Rows {
        Rows { ffn: (part * ffn / parts, ffn / parts), hidden: (part * hidden / parts, hidden / parts) }
    }

    fn split(&self, stages: i32) -> ops::ExpertSplit {
        ops::ExpertSplit { ffn_first: self.ffn.0 as u64, ffn_rows: self.ffn.1 as u64, hidden_first: self.hidden.0 as u64,
                           hidden_rows: self.hidden.1 as u64, stages, send: 0, more: 0, send_at: 0 }
    }

    /// [`Rows::split`] whose down stage completes this device's part of a part link's sum on
    /// `channel`, sending it.
    fn sending(&self, stages: i32, channel: u32) -> ops::ExpertSplit {
        ops::ExpertSplit { send: channel as i32 + 1, ..self.split(stages) }
    }

    /// The gate/up stage sending its columns of the product as part of a part link's sum on
    /// `channel`, at `at` floats into the inbox; with `more`, a later call sends the sum's other
    /// values and completes it.
    fn sending_product(&self, channel: u32, at: usize, more: bool) -> ops::ExpertSplit {
        ops::ExpertSplit { send: channel as i32 + 1, more: i32::from(more), send_at: at as u64, ..self.split(1) }
    }
}

/// The part links' channels (see `Engine::linked`): the attention's output, the experts', and the
/// experts' activations (their gate and up, the shared expert's too, each device its columns).
const ATTENTION: u32 = 0;
const EXPERTS: u32 = 1;
const ACTIVATIONS: u32 = 2;

/// A kv source layer's compressed rows so far (one per group of `ratio` positions), and the rows
/// of its current incomplete group (`tail`).
struct Source {
    ratio: usize,
    groups: usize,
    tail: usize,
}

/// A device's copy of a kv source layer's compressed rows and index keys (BF16: the readers take
/// them through tagged pointers, [`state_tagged`]), and of the rows of its incomplete group (at most
/// one: the ratio is 1 or 2).
struct SourceRows {
    rows: Buffer,
    keys: Buffer,
    tail_kv: Buffer,
    tail_gate: Buffer,
    /// The compressor rows of the last pass that can be taken back (of more than one token, the
    /// ratio 2), after the row waiting before it: what a rewind leaves in the tail.
    pass_kv: Buffer,
    pass_gate: Buffer,
}

/// A kv source's compressed rows or index keys lent to the device's expert store past the
/// sequence's reach ([`Engine::reserve_state`]): the store's region, the source's ratio, a group's
/// bytes there, the buffer's bytes, and how far from its start the store has given them back.
#[derive(Clone, Copy)]
struct StateRegion {
    region: usize,
    ratio: usize,
    row_bytes: u64,
    bytes: u64,
    reclaimed: u64,
}

/// Tokens past a pass's reach its kv source buffers are taken back for (the next passes'
/// groups), and past which, the sequence shorter, they go back to the expert store.
const STATE_AHEAD: usize = 4096;
const STATE_KEEP: usize = 16384;

/// A BF16 state buffer as the kernels take it: its address with bit 0 set (gpu/ops.hpp state_tag).
fn state_tagged(buffer: &Buffer) -> M {
    buffer.base().wrapping_byte_add(1)
}

/// The address `values` BF16 values into a state buffer.
fn halves_at(buffer: &Buffer, values: usize) -> M {
    buffer.base().wrapping_byte_add(values * 2)
}

/// The sequence state before the last pass that can be taken back ([`Engine::rewind`]): its first
/// position, its tokens, and each kv source's groups and rows waiting.
struct Rewind {
    start: usize,
    rows: usize,
    sources: BTreeMap<usize, (usize, usize)>,
}

/// A pass's work buffers. Those of every row (the stream and its coefficients, the attention's and
/// the experts' inputs and outputs, positions, selections and routes) hold a pass's rows and start
/// at a row in a [`view`](Work::view), so a batch of rows uses its own; the attention's scratch
/// holds a batch, the experts' a few rows over their slots or many rows' routes expert by expert.
#[derive(Clone)]
struct Work {
    // of every row
    stream: Buffer,
    stream2: Buffer,
    pre_a: Buffer,
    post_a: Buffer,
    comb_a: Buffer,
    pre_f: [Buffer; 2],
    post_f: Buffer,
    comb_f: Buffer,
    identity: Buffer,
    x: Buffer,
    out: Buffer,
    /// With several devices: this device's parts of the attention's and the experts' outputs (of
    /// the experts', its columns; the others stay zero).
    parts: [Buffer; 2],
    /// positions, ring rows and visible[1..=2]: four segments of a pass's own rows, one allocation.
    /// A longer pass's arena keeps the four apart and uploads them one by one.
    schedule: Buffer,
    positions: Buffer,
    ring_rows: Buffer,
    visible: [Buffer; 3],
    /// Per index source layer: its top-k compressed rows of each row.
    chosen: BTreeMap<usize, Buffer>,
    /// Each row's candidate blocks, from the candidate source's indexer for the later ones (a slot
    /// when the layers have none).
    candidates: Buffer,
    /// The routes and their weights (after a decode row's routes, its predictions of the next
    /// layer's experts).
    ids: Buffer,
    weights: Buffer,
    // of a batch
    qr: Buffer,
    /// A decode row's normed q_a (its norm fused into q_b's launch: see attention_ops).
    qn: Buffer,
    q: Buffer,
    kv: Buffer,
    attended: Buffer,
    low: Buffer,
    pool_kv: Buffer,
    pool_gate: Buffer,
    group_positions: Buffer,
    index_q: Buffer,
    index_w: Buffer,
    scores: Buffer,
    router: Buffer,
    all_rows: Buffer,
    engram_rows: Buffer,
    engram_kv: Buffer,
    head: Buffer,
    /// hc_step's partial sums and counters.
    hc_workspace: Buffer,
    /// The experts of a few rows over their slots: with several devices, this device's share of
    /// the experts' activations (per route a row of the expert width, its share filled; two, used
    /// by waves in turn) and their sum; on one device, the activations.
    products: [Buffer; 2],
    product: Buffer,
    hidden: Buffer,
    /// The experts of many rows, expert by expert: the routes grouped by expert (rows and
    /// weights), an expert's rows of the input, its gate and up (this device's columns; the others
    /// stay zero), the activations of a wave's routes and (several devices) their sum, and an
    /// expert's output (this device's columns).
    route_rows: Buffer,
    route_weights: Buffer,
    xe: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    act_sum: Buffer,
    y: Buffer,
    /// For the grouped WMMA kernels (an FP16 pass on RDNA4, see `Engine::experts_grouped`): the
    /// input rows in FP16, a wave's FP16 products (the device's share of the expert width) and,
    /// with several devices, their summed activations in FP16; and each launch's group and tile
    /// lists.
    xh: Buffer,
    ph: Buffer,
    ah: Buffer,
    tables: Buffer,
    /// Values a row of the buffers of every row: hc, width, routes, selections, candidate blocks.
    widths: (usize, usize, usize, usize, usize),
}

/// Experts acquired at a time by a pass of many rows (each wave's reads overlap the one before's
/// compute), fewer when a size of experts has few slots (DeviceExperts::wave_limit).
const WAVE: usize = 32;
/// Experts of the next layer a decode row predicts from its router applied to this layer's input:
/// the 8 best hold 79% of the 6 it takes.
const PREDICTED: usize = 8;
/// A pass whose next layer routes at least this many times as many times as there are experts
/// prefetches every expert of the next layer (then each used with near certainty: of 4 x 384 routes
/// all but 1.8% of the experts).
const AHEAD: usize = 4;
/// Passes of up to this many rows run the routed experts over their slots (decode); longer ones
/// expert by expert.
const SLOT_ROWS: usize = 4;
/// Most tokens a pass that can be taken back takes, a verify pass among them: its experts run over
/// their slots, and every op of it computes each row as it would alone.
pub const VERIFY_ROWS: usize = SLOT_ROWS;

/// One device's share of the engine: its weights (its parts of the split ones), its copy of the
/// sequence state and work buffers, and its expert store.
struct Shard {
    /// Seconds starting the GPU, its buffers/store and the F32/FP16 GEMM warm-ups.
    startup: [f64; 5],
    /// This device's rows of the experts.
    rows: Rows,
    gpu: Backend,
    dense: HashMap<String, Dense>,
    /// Per layer: this device's rows of the shared expert, as a slot.
    shared: HashMap<usize, Slot>,
    scratch: Buffer,
    rings: BTreeMap<usize, Buffer>,
    sources: BTreeMap<usize, SourceRows>,
    /// A batch's new compressed rows, then their index keys, in F32 (none without kv sources).
    staging: Option<Buffer>,
    plain: Buffer,
    compressed_rope: Buffer,
    /// The pass's buffers (see [`Engine::select_work`]).
    work: Work,
    /// Its own buffers, with those of every row for [`OWN_ROWS`] (a pass of more takes `arena`'s).
    own: Work,
    /// The buffers of every row of a pass of more than [`OWN_ROWS`] (laid out for its rows by
    /// [`Work::in_arena`]), lent to the expert store between such passes (DS41_LEND=0: not lent).
    arena: Option<Buffer>,
    shape: RowShape,
    /// The bytes of `arena` the store gave back for the passes since the last of a few rows, and
    /// the store's region `arena` is (none: not lent).
    reclaimed: u64,
    arena_region: Option<usize>,
    /// The kv sources' buffers lent to the store too.
    state_regions: Vec<StateRegion>,
    experts: DeviceExperts,
    checks: Option<SlotChecks>,
    /// The routers' image biases (every layer's, one after another), uploaded with the first image.
    image_bias: Option<Buffer>,
    /// With a DSpark drafter (on every device): the target features of a pass's last
    /// [`TAP_ROWS`] rows, rows of [`TAP_LAYERS`] x width (see [`tap_features`]).
    tap: Option<Buffer>,
}

/// VRAM a device keeps a prompt pass's FP16 attention weights in (Backend::keep_expansions), for its
/// batches after a layer's first (ATTENTION_ROWS): a layer's on two GPUs take ~170 MB. Expanded for
/// every batch, they held a GPU ~0.23 s of an agent's 1.6K-token turn (2.9 s).
pub const EXPANSIONS: u64 = 192 << 20;

/// The target layers whose input streams DSpark reads (the mean of their hc copies), in order.
pub const TAP_LAYERS: [usize; 3] = [37, 38, 39];
/// The rows of a pass whose features are kept: DSpark's window of history (and the rows every
/// tapped layer computes after a prefill's pruning, [`Engine::rows_of`]).
pub const TAP_ROWS: usize = 128;

/// A drafter's view of one of an engine's devices ([`Engine::beside`]): the device; its tap, the
/// last pass's last min(n, [`TAP_ROWS`]) rows of [`TAP_LAYERS`] x width, in order; and its rows of
/// the head (output.weight), `head_rows` rows from `head_first`.
pub struct Beside<'e> {
    pub gpu: &'e mut Backend,
    pub tap: Buffer,
    pub head: Buffer,
    pub head_kind: u32,
    pub head_first: usize,
    pub head_rows: usize,
}

/// Rows of a pass on a device's own buffers of every row: decode, verify, a short prompt. A
/// longer pass takes the arena lent to the expert store back as far as its rows reach (T8).
pub const OWN_ROWS: usize = 64;
/// The least rows a product takes GEMM for (the devices' default).
const GEMM_MIN_BATCH: u64 = 8;
/// A wave's group and tile lists for the grouped WMMA kernels (32 experts' groups and their 64-row
/// tiles of up to a pass's routes take ~11 KB).
const SLOT_TABLE_BYTES: u64 = 64 << 10;

/// What sizes a device's buffers of every row: hc, width, routes a row, the experts' width,
/// selections a row, the index source layers (in order), whether the device has parts of sums,
/// and the routes a wave of a pass's experts takes at most.
#[derive(Clone)]
struct RowShape {
    hc: usize,
    w: usize,
    k: usize,
    ff: usize,
    top_k: usize,
    chosen: Vec<usize>,
    blocks: usize,
    split: bool,
    capacity: usize,
}

impl RowShape {
    /// The floats of each buffer of every row but identity for a pass of `rows`, in the order
    /// [`Work::in_arena`] lays them out.
    fn floats(&self, rows: usize) -> Vec<usize> {
        let (hc, w, k, ff) = (self.hc, self.w, self.k, self.ff);
        let split = if self.split { rows } else { 0 };
        // a wave's activations: at most the pass's routes and a wave's
        let routes = (rows * k).min(self.capacity).max(rows);
        let mut floats = vec![rows * hc * w, rows * hc * w, rows * hc, rows * hc, rows * hc * hc, rows * hc, rows * hc, rows * hc,
                              rows * hc * hc, rows * w, rows * w, split * w, split * w, rows, rows, rows, rows, rows];
        floats.extend(self.chosen.iter().map(|_| rows * self.top_k));
        floats.push(rows * self.blocks);
        // (y: a wave's routes' outputs; xh, ph and ah: halves, two to a float)
        floats.extend([rows * k + PREDICTED, rows * k, rows * k, (rows * k).max(PREDICTED), rows * w, rows * ff, rows * ff,
                       routes * ff, if self.split { routes } else { 0 } * ff, routes * w, rows * w / 2, routes * ff / 2, routes * ff / 2]);
        floats
    }

    /// The bytes those buffers take one after another, each from a 256-byte boundary.
    fn bytes(&self, rows: usize) -> u64 {
        self.floats(rows).iter().fold(0, |at: u64, &floats| at.next_multiple_of(256) + (floats.max(1) * 4) as u64)
    }
}

/// Checksums a pass queues at most (DS41_CHECK_SLOTS).
const CHECKS: usize = 32768;

/// DS41_CHECK_SLOTS (development): each routed expert's slot is checksummed on the compute stream
/// when acquired and again just before its release; when the pass ends, the two are compared (the
/// slot changed while in use) and the first with the expert's first checksum on this device
/// (another copy of it, from the host tier or the drive, differs).
struct SlotChecks {
    partials: Buffer,
    /// Each queued checksum's layer, expert, slot address and whether it follows the use.
    queued: Vec<(usize, usize, u64, bool)>,
    first: HashMap<(usize, usize), u64>,
    mismatches: u64,
}

/// Queue checksums of `experts`' slots, `bytes` each (DS41_CHECK_SLOTS; nothing otherwise).
fn check_slots(shard: &mut Shard, layer: usize, experts: &[usize], slots: &[Buffer], bytes: u64, after: bool) -> Result<()> {
    let Some(checks) = shard.checks.as_mut() else { return Ok(()) };
    for (&expert, slot) in experts.iter().zip(slots) {
        if checks.queued.len() == CHECKS {
            break;
        }
        let partials = checks.partials.base().wrapping_byte_add(checks.queued.len() * 64 * 8);
        shard.gpu.debug_checksum(slot.base() as P, bytes & !3, partials)?;
        checks.queued.push((layer, expert, slot.base() as u64, after));
    }
    Ok(())
}

/// The sequence state a later pass continues, in host memory ([`Engine::snapshot`]): the window
/// rings' rows of the last `window` positions (oldest first), each kv source's compressed rows,
/// index keys and tail rows, and the counters.
#[derive(Default)]
pub struct Snapshot {
    position: usize,
    history: Vec<u32>,
    /// (layer, groups, tail) of each kv source
    sources: Vec<(usize, usize, usize)>,
    windows: Vec<(usize, Vec<f32>)>,
    /// (layer, [compressed rows, index keys, tail kv, tail gate])
    rows: Vec<(usize, [Vec<f32>; 4])>,
    /// Compressed rows and index keys still in the devices' buffers ([`Engine::finish_snapshot`]).
    pending: bool,
}

impl Snapshot {
    /// Whether its compressed rows and index keys are still to be copied.
    pub fn pending(&self) -> bool {
        self.pending
    }

    /// Host bytes it holds (pending ones too: their room is taken).
    pub fn bytes(&self) -> u64 {
        let floats: usize = self.windows.iter().map(|(_, rows)| rows.capacity()).sum::<usize>()
            + self.rows.iter().flat_map(|(_, kept)| kept.iter().map(Vec::capacity)).sum::<usize>();
        (floats * 4 + self.history.capacity() * 4) as u64
    }
}

/// `values` at `len` and no larger: a buffer taken again neither keeps an earlier, longer
/// checkpoint's memory nor doubles as it grows (the budget counts what is held).
fn fit<T: Clone + Default>(values: &mut Vec<T>, len: usize) {
    values.reserve_exact(len.saturating_sub(values.len()));
    values.resize(len, T::default());
    values.shrink_to_fit();
}

/// `values` empty, with room for `len` and no more (as [`fit`]), for a copy to fill later.
fn fit_room<T>(values: &mut Vec<T>, len: usize) {
    values.clear();
    values.shrink_to(len);
    values.reserve_exact(len);
}

/// Host bytes a [`Snapshot`] holds, known before it is taken: `layers` window rings' rows of the
/// last `window` of `position` positions, each kv source's compressed rows and index keys (its
/// `groups`) and its two tail rows, of `dim` and `index_dim` floats, and `history` tokens.
pub fn snapshot_bytes(position: usize, window: usize, layers: usize, dim: usize, index_dim: usize, groups: &[usize], history: usize) -> u64 {
    // (the window rows and tails F32, the compressed rows and index keys BF16)
    let floats = layers * position.min(window) * dim + groups.iter().map(|&groups| groups * (dim + index_dim) / 2 + 2 * dim).sum::<usize>();
    ((floats + history) * 4) as u64
}

/// Positions `first..end` in a ring of `ring` rows: (ring row, rows, rows before it), at most two.
fn ring_runs(first: usize, end: usize, ring: usize) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    let mut position = first;
    while position < end {
        let at = position % ring;
        let run = (ring - at).min(end - position);
        runs.push((at, run, position - first));
        position += run;
    }
    runs
}

pub struct Engine<'a, T: Tensors> {
    /// Shared, so a pass takes the parameters without copying the Engram tables' metadata.
    hp: std::rc::Rc<Hparams>,
    tensors: &'a T,
    settings: Settings,
    /// One per device; a layer's work is split across them.
    shards: Vec<Shard>,
    position: usize,
    history: Vec<u32>,
    ring: usize,
    sources: BTreeMap<usize, Source>,
    /// The first layer of the decoder (the last kv source), whose layers run only the rows the last
    /// row's output needs (see the module documentation); None: every layer runs every row.
    decoder: Option<usize>,
    /// The last rows of the running pass whose outputs leave it: 1, every row of a verify pass.
    outputs: usize,
    /// The state before the last pass, when it can be taken back.
    rewind: Option<Rewind>,
    /// How many of a decode row's predicted next-layer experts are prefetched (DS41_PREDICT, up to
    /// [`PREDICTED`]; 0, the default, predicts nothing).
    prefetches: usize,
    /// Next-layer experts predicted by this pass, prefetched after the layer's own.
    predicted: Vec<usize>,
    /// Decode runs the experts VRAM holds while the others land (DS41_HITS_FIRST=0: all at once).
    hits_first: bool,
    /// Decode's fills start their gate and up once those are in, their downs landing meanwhile
    /// (DS41_GATE_UP_FIRST=0: once all of them is in).
    gate_up_first: bool,
    /// An FP16 pass's waves of experts take the grouped WMMA kernels where the device has them
    /// (DS41_SLOT_WMMA=0: an expanded GEMM per expert).
    slot_wmma: bool,
    /// A pass of a few rows has the first device's router publish its routes to page-locked host
    /// memory, for the host to take without a readback (DS41_ROUTE_READBACK: read them back).
    mailbox: bool,
    /// On two devices, a decode pass (one row) sums the parts of the attention's and the experts'
    /// outputs over part links: the kernels forming them send them to the other device, and the
    /// hyper-connection step after them adds the two as it reads them, in place of a sum's launch
    /// and its waits (DS41_LINK=0: a sum's launch).
    linked: bool,
    /// The running pass uses the part links.
    link_pass: bool,
    /// A pass using the part links did not complete: the next starts their counts over.
    link_dirty: bool,
    /// The devices' GEMM policy as last set: FP16 operands (see [`Settings::half_gemm`]).
    half_policy: Option<bool>,
    /// With DS41_PROFILE set: seconds per phase, the GPUs synchronized at each phase's end
    /// (DS41_PROFILE=host: the host's time alone, nothing synchronized and graphs kept;
    /// DS41_PROFILE=host+stamps: device 0's time of each decode layer's steps as well).
    profile: Option<(std::time::Instant, BTreeMap<&'static str, f64>)>,
    profile_sync: bool,
    /// DS41_PROFILE=host+stamps: device 0's clock at points of every decode layer
    /// ([`Stamp`]), turned after each pass into "gpu: …" phases of the profile.
    stamps: Option<Stamps>,
    /// A pass's host-side rows, kept between passes: the embeddings repeated into the hc copies,
    /// and a layer's Engram rows decoded. New memory every pass (hundreds of MB) waited on the
    /// kernel's direct reclaim whenever the host ran short of free pages, seconds at a time.
    stream_rows: Vec<f32>,
    engram_values: Vec<f32>,
    /// The rows passes embed at image positions (ids from [`vision::IMAGE_TOKEN`]): each image
    /// span's first position and its rows ([`Engine::set_images`]).
    images: Vec<(usize, Arc<Vec<f32>>)>,
    /// The running pass's rows at image positions (ranges), which route with the image biases.
    image_rows: Vec<(usize, usize)>,
    /// Work::view calls the last decode token did not make because a graph replayed
    /// (DS41_COUNT_VIEWS; off, so it stays 0, unless that is set).
    views: ViewCounter,
    /// Position whose one-token schedule was queued after the previous logits' copy, before that
    /// copy was waited for. The next one-token pass there skips the upload.
    primed: Option<usize>,
    /// Seconds each step of its start took (each device's, the dense weights'), for the load steps
    /// a server reports.
    pub startup: Vec<(String, f64)>,
}

/// A resident tensor (or a layer's shared expert: gate, up and down) for a device's upload thread.
enum Resident<'a> {
    Tensor(String, Tensor<'a>),
    Shared(usize, [Tensor<'a>; 3]),
    /// A tensor made at loading (name, kind, dims, bytes).
    Made(String, u32, Vec<usize>, Vec<u8>),
}

/// Engram's gate weights from its q and k norm weights apart: their product, F32 bytes.
fn engram_gate(q: &Tensor<'_>, k: &Tensor<'_>) -> Result<Vec<u8>> {
    if q.dims != k.dims {
        return Err("DeepSeek-V4.1 engram_q and engram_k differ in shape".into());
    }
    let (mut qs, mut ks) = (Vec::new(), Vec::new());
    formats::decode_into(q.kind, q.data, &mut qs)?;
    formats::decode_into(k.kind, k.data, &mut ks)?;
    Ok(qs.iter().zip(&ks).flat_map(|(a, b)| (a * b).to_le_bytes()).collect())
}

fn frequencies(dims: usize, rope: &Rope) -> Vec<f32> {
    let mut freqs: Vec<f64> = (0..dims / 2).map(|i| 1.0 / rope.base.powf((2 * i) as f64 / dims as f64)).collect();
    if let Some((factor, original, beta_fast, beta_slow)) = rope.yarn {
        let corrected =
            |rotations: f64| dims as f64 * (original / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * rope.base.ln());
        let low = corrected(beta_fast).floor().max(0.0);
        let high = corrected(beta_slow).ceil().min(dims as f64 - 1.0);
        for (i, freq) in freqs.iter_mut().enumerate() {
            let smooth = 1.0 - ((i as f64 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    freqs.into_iter().map(|f| f as f32).collect()
}

fn row_bytes(kind: u32, cols: usize) -> Result<usize> {
    formats::matrix_bytes(kind, 1, cols).ok_or_else(|| format!("unsupported tensor type {kind}"))
}

/// Positions, ring rows, visible[1] and visible[2] of a pass of `n` tokens from `start`.
fn pass_schedule(start: usize, n: usize, ring: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut positions = Vec::with_capacity(n);
    let mut ring_rows = Vec::with_capacity(n);
    let mut visible1 = Vec::with_capacity(n);
    let mut visible2 = Vec::with_capacity(n);
    for p in start..start + n {
        positions.push(p as i32);
        ring_rows.push((p % ring) as i32);
        visible1.push((p + 1) as i32);
        #[allow(clippy::manual_div_ceil)] // visible groups of ratio 2
        visible2.push(((p + 1) / 2) as i32);
    }
    (positions, ring_rows, visible1, visible2)
}

/// Own-row buffers are four contiguous segments of one schedule. A longer pass's arena is not.
fn schedule_is_packed(work: &Work, rows: usize) -> bool {
    let stride = (work.positions.bytes / 4) as usize;
    stride > 0 && rows <= stride && work.ring_rows.bytes == work.positions.bytes
        && work.visible[1].bytes == work.positions.bytes && work.visible[2].bytes == work.positions.bytes
        && work.positions.bytes.checked_mul(4) == Some(work.schedule.bytes)
        && work.ring_rows.base() == work.positions.at(stride)
        && work.visible[1].base() == work.positions.at(stride * 2)
        && work.visible[2].base() == work.positions.at(stride * 3)
}

fn upload_rows(shard: &mut Shard, positions: &[i32], ring_rows: &[i32], visible1: &[i32], visible2: &[i32]) -> Result<()> {
    let work = &shard.work;
    if schedule_is_packed(work, positions.len()) {
        let stride = (work.positions.bytes / 4) as usize;
        let image = pack_schedule(stride, positions, ring_rows, visible1, visible2);
        shard.gpu.write_buffer(&work.schedule, 0, &image)
    } else {
        shard.gpu.write_buffer(&work.positions, 0, positions)?;
        shard.gpu.write_buffer(&work.ring_rows, 0, ring_rows)?;
        shard.gpu.write_buffer(&work.visible[1], 0, visible1)?;
        shard.gpu.write_buffer(&work.visible[2], 0, visible2)
    }
}

/// Host image of one upload into `schedule`: `stride` i32s each of positions, ring rows,
/// visible[1] and visible[2]. A pass's rows sit at the start of each segment.
fn pack_schedule(stride: usize, positions: &[i32], ring_rows: &[i32], visible1: &[i32], visible2: &[i32]) -> Vec<i32> {
    let n = positions.len();
    let mut packed = vec![0i32; stride * 4];
    packed[..n].copy_from_slice(positions);
    packed[stride..stride + n].copy_from_slice(ring_rows);
    packed[stride * 2..stride * 2 + n].copy_from_slice(visible1);
    packed[stride * 3..stride * 3 + n].copy_from_slice(visible2);
    packed
}

fn name(layer: usize, tensor: &str) -> String {
    format!("blk.{layer}.{tensor}.weight")
}

/// Columns [first, first + count) of every row of a `rows` x `cols` matrix, whole blocks only.
pub(super) fn columns(kind: u32, rows: usize, cols: usize, data: &[u8], first: usize, count: usize) -> Result<Vec<u8>> {
    let whole = row_bytes(kind, cols)?;
    let (skip, take) = (formats::matrix_bytes(kind, 1, first), formats::matrix_bytes(kind, 1, count));
    let (Some(skip), Some(take)) = (skip, take) else {
        return Err(format!("columns {first}..{} of a type {kind} matrix split a block", first + count));
    };
    let mut out = Vec::with_capacity(rows * take);
    for row in data.chunks_exact(whole).take(rows) {
        out.extend_from_slice(&row[skip..skip + take]);
    }
    Ok(out)
}

impl Shard {
    fn new(hp: &Hparams, settings: &Settings, device: usize, budget: u64, layout: Arc<Layout>, rows: Rows, parts: usize)
           -> Result<Shard> {
        let range = settings.layers.clone();
        let started = std::time::Instant::now();
        let mut gpu = Backend::new(device, budget)?;
        if std::env::var("DS41_FP16_CHECKS").is_ok_and(|value| value == "1") {
            gpu.enable_gemm_checks()?;
        }
        let initialized = started.elapsed().as_secs_f64();
        let scratch = gpu.buffer(settings.scratch_bytes)?;
        // (a prompt pass's attention weights expanded once a layer, not a batch: ATTENTION_ROWS)
        if settings.half_gemm && settings.rows >= 32 {
            gpu.keep_expansions(std::env::var("DS41_EXPANSIONS").map_or(EXPANSIONS, |value| if value == "0" { 0 } else { EXPANSIONS }))?;
        }
        // The normal four-token warm takes GEMV; move BLAS's lazy setup ahead of requests. Initialize both policies: batches below 32 still use F32.
        // Synthetic zero matrices touch no model, expert store or sequence state. Release
        // them before the long-lived buffers/cache are allocated (no residency change).
        let mut gemm_warm = [0.0; 2];
        if settings.half_gemm && settings.rows >= 32 {
            let warming = std::time::Instant::now();
            let weights = gpu.buffer(256 * 256 * 4)?;
            let input = gpu.buffer(32 * 256 * 4)?;
            let output = gpu.buffer(32 * 256 * 4)?;
            let mut finished = [0.0f32];
            for (index, half) in [false, true].into_iter().enumerate() {
                let started = if index == 0 { warming } else { std::time::Instant::now() };
                gpu.linear_gemm(&weights, 0, 256, 256, 0, input.base(), 0, 32, output.base(), 0, &scratch, half)?;
                // This read waits for the stream; startup timing includes completed GPU work.
                gpu.read_buffer(&output, 0, &mut finished)?;
                gemm_warm[index] = started.elapsed().as_secs_f64();
            }
            for buffer in [weights, input, output] { gpu.release_buffer(buffer)?; }
        }
        let (b, hc, w) = (settings.batch, hp.hc, hp.width);
        // One table for every attention of this pass, before any graph records a pointer into it.
        if hp.rope_dims > 0 {
            gpu.attention_turns_reserve(b as u64, hp.rope_dims as u64)?;
        }
        let (r, k, ff) = (settings.rows.max(b), hp.experts_used, hp.expert_ff);
        // the buffers of every row for this device's own rows (a longer pass's in the arena)
        let own = OWN_ROWS.min(r);
        let ring = hp.window + b;
        let floats = |gpu: &mut Backend, count: usize| gpu.buffer((count.max(1) * 4) as u64);
        let halves = |gpu: &mut Backend, count: usize| gpu.buffer((count.max(1) * 2) as u64);
        let rings = range.clone().map(|layer| Ok((layer, floats(&mut gpu, ring * hp.head_dim)?))).collect::<Result<BTreeMap<_, _>>>()?;
        let mut sources = BTreeMap::new();
        for (layer, plan) in hp.layers.iter().enumerate().filter(|(layer, _)| range.contains(layer)) {
            if let Some(c) = plan.compressed.filter(|c| c.kv_source == layer) {
                let groups = settings.context / c.ratio + 1;
                sources.insert(layer, SourceRows {
                    rows: halves(&mut gpu, groups * hp.head_dim)?,
                    keys: halves(&mut gpu, groups * hp.index_dim)?,
                    tail_kv: floats(&mut gpu, hp.head_dim)?,
                    tail_gate: floats(&mut gpu, hp.head_dim)?,
                    pass_kv: floats(&mut gpu, (VERIFY_ROWS + 1) * hp.head_dim)?,
                    pass_gate: floats(&mut gpu, (VERIFY_ROWS + 1) * hp.head_dim)?,
                });
            }
        }
        // A batch's new compressed rows and index keys in F32 before they are rounded into the state
        let staging = if sources.is_empty() { None } else { Some(floats(&mut gpu, (b + 2) * (hp.head_dim + hp.index_dim))?) };
        let max_groups = settings.context + 1;
        let identity: Vec<f32> = (0..r).flat_map(|_| (0..hc).map(|j| f32::from(u8::from(j == 0)))).collect();
        let all_rows: Vec<i32> = (0..b as i32).collect();
        let mut chosen = BTreeMap::new();
        let blocks = candidate_blocks(hp, range.clone())?;
        for (layer, plan) in hp.layers.iter().enumerate().filter(|(layer, _)| range.contains(layer)) {
            if plan.compressed.is_some_and(|c| c.index_source == layer) {
                chosen.insert(layer, floats(&mut gpu, own * hp.index_top_k)?);
            }
        }
        // with one device, no parts and no sums
        let split = if parts > 1 { own } else { 0 };
        // a wave's activations: at most the pass's routes and a wave's
        let routes = (own * k).min(r).max(own);
        // Four i32 vectors of a pass's own rows, contiguous so one upload fills them.
        let schedule_rows = own.max(1);
        let schedule = floats(&mut gpu, schedule_rows * 4)?;
        let segment = (schedule_rows * 4) as u64;
        let positions = schedule.slice(0, segment);
        let ring_rows = schedule.slice(segment, segment);
        let visible_1 = schedule.slice(segment * 2, segment);
        let visible_2 = schedule.slice(segment * 3, segment);
        let work = Work {
            stream: floats(&mut gpu, own * hc * w)?,
            stream2: floats(&mut gpu, own * hc * w)?,
            pre_a: floats(&mut gpu, own * hc)?,
            post_a: floats(&mut gpu, own * hc)?,
            comb_a: floats(&mut gpu, own * hc * hc)?,
            pre_f: [floats(&mut gpu, own * hc)?, floats(&mut gpu, own * hc)?],
            post_f: floats(&mut gpu, own * hc)?,
            comb_f: floats(&mut gpu, own * hc * hc)?,
            identity: floats(&mut gpu, r * hc)?,
            x: floats(&mut gpu, own * w)?,
            out: floats(&mut gpu, own * w)?,
            parts: [floats(&mut gpu, split * w)?, floats(&mut gpu, split * w)?],
            schedule,
            positions,
            ring_rows,
            visible: [floats(&mut gpu, own)?, visible_1, visible_2],
            chosen,
            candidates: floats(&mut gpu, own * blocks)?,
            ids: floats(&mut gpu, own * k + PREDICTED)?,
            weights: floats(&mut gpu, own * k)?,
            qr: floats(&mut gpu, b * hp.q_rank)?,
            qn: floats(&mut gpu, hp.q_rank)?,
            q: floats(&mut gpu, b * hp.heads * hp.head_dim)?,
            kv: floats(&mut gpu, b * hp.head_dim)?,
            attended: floats(&mut gpu, b * hp.heads * hp.head_dim)?,
            low: floats(&mut gpu, b * hp.out_groups * hp.out_rank)?,
            pool_kv: floats(&mut gpu, (b + 1) * hp.head_dim)?,
            pool_gate: floats(&mut gpu, (b + 1) * hp.head_dim)?,
            group_positions: floats(&mut gpu, b + 1)?,
            index_q: floats(&mut gpu, b * hp.index_heads * hp.index_dim)?,
            index_w: floats(&mut gpu, b * hp.index_heads)?,
            scores: floats(&mut gpu, b * max_groups)?,
            router: floats(&mut gpu, (b + 1) * hp.experts)?,
            all_rows: floats(&mut gpu, b)?,
            engram_rows: floats(&mut gpu, b * hp.engram_columns() * hp.engram.as_ref().map_or(1, |e| e.head_dim))?,
            engram_kv: floats(&mut gpu, b * (hc + 1) * w)?,
            // (and a part link status after the logits)
            head: floats(&mut gpu, if range.end == hp.layers.len() { VERIFY_ROWS * hp.vocabulary / parts + 1 } else { 1 })?,
            hc_workspace: floats(&mut gpu, b * w.div_ceil(64) * ((2 + hc) * hc + 2) + b)?,
            products: [floats(&mut gpu, SLOT_ROWS * (k + 1) * ff)?, floats(&mut gpu, SLOT_ROWS * (k + 1) * ff)?],
            product: floats(&mut gpu, SLOT_ROWS * (k + 1) * ff)?,
            hidden: floats(&mut gpu, SLOT_ROWS * k * ff)?,
            route_rows: floats(&mut gpu, own * k)?,
            route_weights: floats(&mut gpu, (own * k).max(PREDICTED))?,
            xe: floats(&mut gpu, own * w)?,
            gate: floats(&mut gpu, own * ff)?,
            up: floats(&mut gpu, own * ff)?,
            act: floats(&mut gpu, routes * ff)?,
            act_sum: floats(&mut gpu, if parts > 1 { routes } else { 0 } * ff)?,
            y: floats(&mut gpu, routes * w)?,
            xh: floats(&mut gpu, own * w / 2)?,
            ph: floats(&mut gpu, routes * ff / 2)?,
            ah: floats(&mut gpu, routes * ff / 2)?,
            tables: gpu.buffer(SLOT_TABLE_BYTES)?,
            widths: (hc, w, k, hp.index_top_k, blocks),
        };
        gpu.write_buffer(&work.identity, 0, &identity)?;
        gpu.write_buffer(&work.all_rows, 0, &all_rows)?;
        let plain = frequencies(hp.rope_dims, &hp.rope);
        let compressed = frequencies(hp.rope_dims, &hp.compressed_rope);
        let plain_buffer = floats(&mut gpu, plain.len())?;
        gpu.write_buffer(&plain_buffer, 0, &plain)?;
        let compressed_buffer = floats(&mut gpu, compressed.len())?;
        gpu.write_buffer(&compressed_buffer, 0, &compressed)?;
        let building = std::time::Instant::now();
        let buffers = (building - started).as_secs_f64() - initialized - gemm_warm.iter().sum::<f64>();
        let mut experts = DeviceExperts::new(&mut gpu, layout, Reader::shared(), settings.expert_slots * parts,
                                             settings.host_expert_bytes)?;
        let store = building.elapsed().as_secs_f64();
        let shape = RowShape {
            hc, w, k, ff, top_k: hp.index_top_k, chosen: work.chosen.keys().copied().collect(), blocks, split: parts > 1, capacity: r,
        };
        let arena = if r > own { Some(gpu.buffer(shape.bytes(r))?) } else { None };
        let arena_region = match arena.filter(|_| std::env::var("DS41_LEND").map_or(true, |value| value != "0")) {
            Some(arena) => Some(experts.lend_region(arena)?.0),
            None => None,
        };
        // The kv sources' compressed rows and index keys past the sequence's reach hold experts
        // too (DS41_LEND_STATE=0: not lent): a pass takes back what it reaches, a shorter sequence
        // gives the rest back. Sized for the whole context, they sit mostly unused (a 75K-token
        // sequence of 262144 reaches 29%).
        let mut state_regions = Vec::new();
        if std::env::var("DS41_LEND_STATE").map_or(true, |value| value != "0") {
            for (&layer, rows) in &sources {
                let ratio = hp.layers[layer].compressed.map_or(1, |c| c.ratio);
                for (buffer, dim) in [(rows.rows, hp.head_dim), (rows.keys, hp.index_dim)] {
                    let (region, _) = experts.lend_region(buffer)?;
                    state_regions.push(StateRegion { region, ratio, row_bytes: (dim * 2) as u64, bytes: buffer.bytes, reclaimed: 0 });
                }
            }
        }
        let checks = match std::env::var_os("DS41_CHECK_SLOTS") {
            Some(_) => Some(SlotChecks { partials: gpu.buffer((CHECKS * 64 * 8) as u64)?, queued: Vec::new(), first: HashMap::new(), mismatches: 0 }),
            None => None,
        };
        Ok(Shard {
            startup: [initialized, buffers, store, gemm_warm[0], gemm_warm[1]],
            tap: None,
            checks,
            rows,
            gpu, dense: HashMap::new(), shared: HashMap::new(), scratch, rings, sources, staging, plain: plain_buffer,
            compressed_rope: compressed_buffer, own: work.clone(), work, arena, shape, reclaimed: 0, arena_region, state_regions, experts,
            image_bias: None,
        })
    }

    /// This device's part of a resident tensor over `parts` devices (see [`divided`]), or all of it.
    fn place(&mut self, part: usize, parts: usize, name: String, tensor: &Tensor<'_>) -> Result<()> {
        if parts == 1 || !divided(&name) {
            return self.upload(name, tensor.kind, tensor.dims.clone(), tensor.data);
        }
        // (rows: every dimension past the first; GGUFs that give wo_a as [cols, rank, groups] hold
        // its groups' rows one group after another, as [cols, groups x rank] does)
        let (cols, rows) = (tensor.dims[0], tensor.dims[1..].iter().product::<usize>().max(1));
        if name.ends_with(".attn_output_b.weight") {
            // its groups' columns: a part of every output
            let count = cols / parts;
            let data = columns(tensor.kind, rows, cols, tensor.data, part * count, count)?;
            self.upload(name, tensor.kind, vec![count, rows], &data)
        } else if name.ends_with(".attn_sinks.weight") {
            let count = cols / parts;
            self.upload(name, tensor.kind, vec![count], &tensor.data[part * count * 4..(part + 1) * count * 4])
        } else {
            // its rows: of q_b its heads, of wo_a its groups, of the head its tokens
            let (count, bytes) = (rows / parts, row_bytes(tensor.kind, cols)?);
            self.upload(name, tensor.kind, vec![cols, count], &tensor.data[part * count * bytes..(part + 1) * count * bytes])
        }
    }

    /// This device's rows of a layer's shared expert (gate, up and down) as a slot.
    fn place_shared(&mut self, layer: usize, [gate, up, down]: &[Tensor<'_>; 3]) -> Result<()> {
        let rows = |tensor: &Tensor<'_>, (first, count): (usize, usize)| -> Result<Vec<u8>> {
            let bytes = row_bytes(tensor.kind, tensor.dims[0])?;
            Ok(tensor.data[first * bytes..(first + count) * bytes].to_vec())
        };
        let pieces = [rows(gate, self.rows.ffn)?, rows(up, self.rows.ffn)?, rows(down, self.rows.hidden)?];
        let offsets = [0, pieces[0].len() as u64, (pieces[0].len() + pieces[1].len()) as u64];
        let buffer = self.gpu.upload_bytes(&pieces.concat())?;
        self.shared.insert(layer, Slot { buffer, offsets, kinds: [gate.kind, up.kind, down.kind] });
        Ok(())
    }

    /// A routed expert of a layer as this device's slots hold it (see [`moe::Shape`]).
    fn expert_shape(&self, hp: &Hparams, gate_bytes: u64, down_bytes: u64, kinds: [u32; 3]) -> moe::Shape {
        moe::Shape { gate_bytes, down_bytes, kinds, width: hp.width, ffn: hp.expert_ff, ffn_rows: self.rows.ffn, hidden_rows: self.rows.hidden }
    }

    fn dense(&self, name: &str) -> Result<Dense> {
        self.dense.get(name).cloned().ok_or_else(|| format!("tensor {name} is not resident"))
    }

    fn upload(&mut self, name: String, kind: u32, dims: Vec<usize>, data: &[u8]) -> Result<()> {
        let buffer = self.gpu.upload_bytes(data)?;
        self.dense.insert(name, Dense { buffer, kind, dims });
        Ok(())
    }

    /// out = W . input over `steps` rows (a whole resident 2D tensor).
    fn linear(&mut self, name: &str, input: P, steps: usize, out: M) -> Result<()> {
        let weight = self.dense(name)?;
        let scratch = self.scratch;
        self.gpu.linear(&weight.buffer, weight.kind, weight.dims[1] as u64, weight.dims[0] as u64, 0, input, 0, steps as u64, out, 0,
                        &scratch)
    }

    /// [`Shard::linear`] of a row into this device's part of a part link's sum on `channel`, sent as
    /// it forms.
    fn linear_send(&mut self, channel: u32, name: &str, input: P, steps: usize, out: M) -> Result<()> {
        let weight = self.dense(name)?;
        self.gpu.linear_send(channel, &weight.buffer, weight.kind, weight.dims[1] as u64, weight.dims[0] as u64, input, steps as u64, out)
    }

    fn norm(&mut self, name: &str, x: M, rows: usize, width: usize, eps: f32) -> Result<()> {
        let weight = self.dense(name)?;
        self.gpu.rms_norm(x, &weight.buffer, rows as u64, width as u64, eps, x)
    }
}

impl<'a, T: Tensors> Engine<'a, T> {
    /// An engine over `devices`: one, or several splitting every layer (tensor parallel).
    pub fn new(hp: Hparams, tensors: &'a T, devices: &[usize], budget: u64, settings: Settings) -> Result<Engine<'a, T>> {
        let range = settings.layers.clone();
        if range.is_empty() || range.end > hp.layers.len() {
            return Err("DeepSeek-V4.1 engine layers outside the model".into());
        }
        for plan in &hp.layers[range.clone()] {
            if let Some(c) = plan.compressed.filter(|c| c.kv_source < range.start || c.index_source < range.start) {
                return Err(format!("DeepSeek-V4.1 layers {range:?} read layer {}'s compressed rows", c.kv_source.min(c.index_source)));
            }
        }
        if settings.batch == 0 || settings.rows == 0 {
            return Err("a DeepSeek-V4.1 pass takes rows".into());
        }
        let parts = devices.len();
        if parts == 0 || !hp.heads.is_multiple_of(hp.out_groups) || !hp.out_groups.is_multiple_of(parts)
            || !hp.vocabulary.is_multiple_of(parts) || !hp.expert_ff.is_multiple_of(parts) || !hp.width.is_multiple_of(parts)
        {
            return Err(format!("DeepSeek-V4.1 does not split over {parts} devices"));
        }
        // Each device's rows of each layer's experts as three roles of the files (a slot holds them
        // back to back).
        let files = tensors.files()?;
        let mut shards = Vec::with_capacity(parts);
        let mut startup = Vec::new();
        for (part, &device) in devices.iter().enumerate() {
            let rows = Rows::of(part, parts, hp.expert_ff, hp.width);
            let mut roles = vec![None; hp.layers.len()];
            for layer in range.clone() {
                let mut layer_roles = Vec::new();
                for (role, (first, count)) in [("ffn_gate_exps", rows.ffn), ("ffn_up_exps", rows.ffn), ("ffn_down_exps", rows.hidden)] {
                    let tensor_name = name(layer, role);
                    let tensor = tensors.tensor(&tensor_name)?;
                    let (file, base) = tensors.location(&tensor_name).ok_or("the expert store needs the model's files")?;
                    let bytes = row_bytes(tensor.kind, tensor.dims[0])?;
                    let stride = (bytes * tensor.dims[1]) as u64;
                    layer_roles.push(Role { file, base: base + (first * bytes) as u64, stride, bytes: (count * bytes) as u64 });
                }
                roles[layer] = Some(layer_roles);
            }
            let laying = std::time::Instant::now();
            let layout = Arc::new(Layout::new(&files, roles, tensors.verifier())?.with_experts(hp.experts));
            let laid = laying.elapsed().as_secs_f64();
            let shard = Shard::new(&hp, &settings, device, budget, layout, rows, parts)?;
            let [init, buffers, store, f32_warm, f16_warm] = shard.startup;
            // (the wait before a server starts, by step: T161)
            startup.extend([(format!("GPU {device} expert layout"), laid), (format!("GPU {device} init"), init),
                            (format!("GPU {device} buffers"), buffers), (format!("GPU {device} expert store"), store),
                            (format!("GPU {device} F32 GEMM warm"), f32_warm), (format!("GPU {device} FP16 GEMM warm"), f16_warm)]);
            shards.push(shard);
        }
        let mut sources = BTreeMap::new();
        for (layer, plan) in hp.layers.iter().enumerate().filter(|(layer, _)| range.contains(layer)) {
            if let Some(c) = plan.compressed.filter(|c| c.kv_source == layer) {
                sources.insert(layer, Source { ratio: c.ratio, groups: 0, tail: 0 });
            }
        }
        let decoder = hp.layers.iter().enumerate().filter_map(|(layer, plan)| plan.compressed.filter(|c| c.kv_source == layer).map(|_| layer))
            .max().filter(|_| std::env::var_os("DS41_FULL_DECODER").is_none());
        let linked = shards.len() == 2 && std::env::var("DS41_LINK").map_or(true, |value| value != "0");
        // A decode row's Q5_K/Q8_0 projections, IQ2 experts' gate/up and down (IQ2_S's down excepted) and
        // Q5_K shared expert take INT8 activations (the Q8 kernels: DS41_Q8=0 keeps them F32); with the FP16
        // prompt pass, a decode or verify pass's index and attention scores take FP16 operands (the GEMM policy
        // is F32 for those passes).
        let q8 = std::env::var("DS41_Q8").map_or(true, |value| value != "0");
        for shard in &shards {
            shard.gpu.decode_options(q8, settings.half_gemm)?;
        }
        if let [first, second] = &mut shards[..] {
            if linked {
                // (a row's output, or a decode row's activations: every route's and the shared expert's)
                Backend::link_parts(&mut first.gpu, &mut second.gpu, hp.width.max((hp.experts_used + 1) * hp.expert_ff) as u64)?;
            }
        }
        let mut engine = Engine {
            ring: hp.window + settings.batch,
            decoder,
            outputs: 1,
            rewind: None,
            hp: std::rc::Rc::new(hp),
            tensors,
            settings,
            shards,
            position: 0,
            history: Vec::new(),
            sources,
            prefetches: std::env::var("DS41_PREDICT").ok().and_then(|n| n.parse::<usize>().ok()).unwrap_or(0).min(PREDICTED),
            predicted: Vec::new(),
            hits_first: std::env::var("DS41_HITS_FIRST").map_or(true, |value| value != "0"),
            gate_up_first: std::env::var("DS41_GATE_UP_FIRST").map_or(true, |value| value != "0"),
            slot_wmma: std::env::var("DS41_SLOT_WMMA").map_or(true, |value| value != "0"),
            mailbox: std::env::var_os("DS41_ROUTE_READBACK").is_none(),
            linked,
            link_pass: false,
            link_dirty: false,
            half_policy: None,
            profile: std::env::var_os("DS41_PROFILE").map(|_| (std::time::Instant::now(), BTreeMap::new())),
            profile_sync: std::env::var("DS41_PROFILE").is_ok_and(|value| !value.starts_with("host")),
            stamps: None,
            stream_rows: Vec::new(),
            engram_values: Vec::new(),
            images: Vec::new(),
            image_rows: Vec::new(),
            views: ViewCounter::new(std::env::var_os("DS41_COUNT_VIEWS").is_some()),
            primed: None,
            startup,
        };
        if std::env::var("DS41_PROFILE").is_ok_and(|value| value == "host+stamps") {
            // (made before any graph is recorded: a recorded stamp keeps its address)
            let gpu = &mut engine.shards[0].gpu;
            let buffer = gpu.buffer(((engine.hp.layers.len() * Stamp::COUNT + PASS_STAMPS) * 8) as u64)?;
            let khz = gpu.debug_clock_khz()? as f64;
            engine.stamps = Some(Stamps { buffer, khz, last_end: None });
        }
        let (reading, uploading) = engine.upload_dense()?;
        engine.startup.extend([("dense read".to_string(), reading), ("dense upload after".to_string(), uploading)]);
        let staging = std::time::Instant::now();
        for shard in &mut engine.shards {
            shard.experts.prepare_staging()?;
        }
        engine.startup.push(("host staging ready".to_string(), staging.elapsed().as_secs_f64()));
        Ok(engine)
    }

    /// Every tensor of this engine's layers but the routed experts and the Engram tables, resident
    /// (with the head on the last engine): each device's parts of the divided ones (see the module
    /// documentation and [`divided`]), the rest whole; each shared expert's part as a slot. This
    /// thread takes the tensors from the source (which may check them as it hands them over), a
    /// thread per device uploads its parts of those taken before.
    /// Returns its seconds taking the tensors from the source (their authentication) and waiting
    /// for the uploads after.
    fn upload_dense(&mut self) -> Result<(f64, f64)> {
        let hp = self.hp.clone();
        let range = self.settings.layers.clone();
        let parts = self.shards.len();
        let mut names = Vec::new();
        if range.end == hp.layers.len() {
            names.extend(["output.weight".to_string(), "output_norm.weight".to_string()]);
        }
        for layer in range.clone() {
            for tensor in ["attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv", "attn_kv_a_norm", "attn_sinks",
                           "attn_output_a", "attn_output_b", "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn",
                           "hc_ffn_base", "hc_ffn_scale", "ffn_norm", "ffn_gate_inp", "attn_compressor_kv", "attn_compressor_gate",
                           "attn_compressor_norm", "indexer.attn_k", "indexer.k_norm", "indexer.attn_q_b", "indexer.proj",
                           "engram_kv", "engram_wkv", "engram_gate"] {
                names.push(name(layer, tensor));
            }
            names.push(format!("blk.{layer}.exp_probs_b.bias"));
        }
        // GGUFs that keep Engram's two norm weights apart (engram_q, engram_k): the gate the op
        // takes is their product (the gate others store; a product of BF16s is exact in F32)
        let mut gates = Vec::new();
        for layer in range.clone() {
            if let (Ok(q), Ok(k)) = (self.tensors.tensor(&name(layer, "engram_q")), self.tensors.tensor(&name(layer, "engram_k"))) {
                gates.push((name(layer, "engram_gate"), engram_gate(&q, &k)?, q.dims.clone()));
            }
        }
        let tensors = self.tensors;
        let copy = |tensor: &Tensor<'a>| Tensor { kind: tensor.kind, dims: tensor.dims.clone(), data: tensor.data };
        let mut authentication = names.clone();
        for layer in range.clone() {
            authentication.extend(["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"].map(|role| name(layer, role)));
        }
        // Include planning/worker setup in dense read, rather than hiding work before its timer.
        let preparing = std::time::Instant::now();
        tensors.with_authentication(&authentication, |tensors| std::thread::scope(|scope| {
            let prepared = preparing.elapsed().as_secs_f64();
            let (mut senders, mut threads) = (Vec::new(), Vec::new());
            for (part, shard) in self.shards.iter_mut().enumerate() {
                let (sender, receiver) = channel::<Resident<'a>>();
                senders.push(sender);
                threads.push(scope.spawn(move || -> Result<()> {
                    for resident in receiver {
                        match resident {
                            Resident::Tensor(name, tensor) => shard.place(part, parts, name, &tensor)?,
                            Resident::Shared(layer, roles) => shard.place_shared(layer, &roles)?,
                            Resident::Made(name, kind, dims, data) => shard.place(part, parts, name, &Tensor { kind, dims, data: &data })?,
                        }
                    }
                    Ok(())
                }));
            }
            // A device's thread that failed takes no more; its error is the one reported.
            let send = |resident: &dyn Fn() -> Resident<'a>| senders.iter().all(|sender| sender.send(resident()).is_ok());
            let mut taken = Ok(());
            // (the time the source takes to hand the dense tensors over: their authentication)
            let mut reading = prepared;
            for name in names {
                let read = std::time::Instant::now();
                let Ok(tensor) = tensors.tensor(&name) else { continue };
                reading += read.elapsed().as_secs_f64();
                // (engram_wkv is what others call engram_kv)
                let placed = name.replace(".engram_wkv.", ".engram_kv.");
                // The hyper-connection weights go to the device as BF16: half of hc_step's reads.
                if tensor.kind == 0 && (placed.ends_with(".hc_attn_fn.weight") || placed.ends_with(".hc_ffn_fn.weight")) {
                    let halves: Vec<u8> = tensor.data.chunks_exact(4)
                        .flat_map(|bytes| crate::vision::bf16(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])).to_le_bytes())
                        .collect();
                    if !send(&|| Resident::Made(placed.clone(), 30, tensor.dims.clone(), halves.clone())) {
                        break;
                    }
                    continue;
                }
                if !send(&|| Resident::Tensor(placed.clone(), copy(&tensor))) {
                    break;
                }
            }
            for (name, gate, dims) in &gates {
                if !send(&|| Resident::Made(name.clone(), 0, dims.clone(), gate.clone())) {
                    break;
                }
            }
            for layer in range {
                let read = std::time::Instant::now();
                let roles = ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"].map(|role| tensors.tensor(&name(layer, role)));
                reading += read.elapsed().as_secs_f64();
                let [gate, up, down] = match roles {
                    [Ok(gate), Ok(up), Ok(down)] => [gate, up, down],
                    [gate, up, down] => {
                        taken = gate.and(up).and(down).map(drop);
                        break;
                    }
                };
                if !send(&|| Resident::Shared(layer, [copy(&gate), copy(&up), copy(&down)])) {
                    break;
                }
            }
            drop(senders);
            let waiting = std::time::Instant::now();
            for thread in threads {
                thread.join().unwrap_or_else(|_| Err("a device's upload thread panicked".into()))?;
            }
            taken.map(|()| (reading, waiting.elapsed().as_secs_f64()))
        }))
    }

    /// Attribute the time since the last mark to `phase` (profiling only: waits for the GPUs, but
    /// for the host's time alone).
    fn mark(&mut self, phase: &'static str) -> Result<()> {
        if self.profile.is_none() {
            return Ok(());
        }
        for shard in self.shards.iter_mut().filter(|_| self.profile_sync) {
            let mut one = [0f32; 1];
            let x = shard.work.x;
            shard.gpu.read_buffer(&x, 0, &mut one)?;
        }
        let (since, phases) = self.profile.as_mut().unwrap();
        *phases.entry(phase).or_default() += since.elapsed().as_secs_f64();
        *since = std::time::Instant::now();
        Ok(())
    }

    /// DS41_PROFILE=host+stamps: queue device 0's clock into the slot of `point` of `layer` (a decode pass's).
    fn stamp(&mut self, layer: usize, rows: usize, point: Stamp) -> Result<()> {
        match &self.stamps {
            Some(stamps) if rows <= VERIFY_ROWS => {
                let at = stamps.buffer.base();
                self.shards[0].gpu.debug_stamp(at, (layer * Stamp::COUNT + point as usize) as u64)
            }
            _ => Ok(()),
        }
    }

    /// DS41_PROFILE=host+stamps: queue device 0's clock into pass point `point` (< PASS_STAMPS).
    fn stamp_pass(&mut self, rows: usize, point: usize) -> Result<()> {
        let slot = (self.hp.layers.len() * Stamp::COUNT + point) as u64;
        match &self.stamps {
            Some(stamps) if rows <= VERIFY_ROWS => {
                let at = stamps.buffer.base();
                self.shards[0].gpu.debug_stamp(at, slot)
            }
            _ => Ok(()),
        }
    }

    /// DS41_PROFILE=host+stamps: after a decode pass, its layers' device time into the profile ("gpu: …").
    fn read_stamps(&mut self, layers: std::ops::Range<usize>, n: usize) -> Result<()> {
        let Some(stamps) = self.stamps.as_mut().filter(|_| n <= VERIFY_ROWS) else { return Ok(()) };
        let mut ticks = vec![0u64; self.hp.layers.len() * Stamp::COUNT + PASS_STAMPS];
        self.shards[0].gpu.read_buffer(&stamps.buffer, 0, &mut ticks)?;
        let seconds = |from: u64, to: u64| to.saturating_sub(from) as f64 / (stamps.khz * 1e3);
        let at = |layer: usize, point: Stamp| ticks[layer * Stamp::COUNT + point as usize];
        let mut phases: Vec<(&'static str, f64)> = Vec::new();
        for layer in layers.clone() {
            let (start, end) = (at(layer, Stamp::LayerStart), at(layer, Stamp::LayerEnd));
            let routed = at(layer, Stamp::Routed);
            if routed == 0 {
                // (a layer whose experts took another path: its time as a whole)
                phases.push(("gpu: layer without stamps", seconds(start, end)));
            } else {
                let (shared, taken, experts) = (at(layer, Stamp::SharedDone), at(layer, Stamp::RoutesTaken), at(layer, Stamp::ExpertsStart));
                let (attending, hc, attended) = (at(layer, Stamp::AttentionStart), at(layer, Stamp::HcDone), at(layer, Stamp::AttentionDone));
                phases.push(("gpu: engram", seconds(start, attending)));
                phases.push(("gpu: attention hc", seconds(attending, hc)));
                phases.push(("gpu: attention", seconds(hc, attended)));
                phases.push(("gpu: ffn hc and router", seconds(attended, routed)));
                phases.push(("gpu: shared gate/up", seconds(routed, shared)));
                phases.push(("gpu: routes to the host and back", seconds(shared, taken)));
                phases.push(("gpu: host routing", seconds(taken, experts)));
                let (hits, misses, summed) = (at(layer, Stamp::HitsDone), at(layer, Stamp::MissesDone), at(layer, Stamp::ActivationsSummed));
                if hits > 0 && misses > 0 && summed > 0 {
                    phases.push(("gpu: experts: hits' gate/up", seconds(experts, hits)));
                    phases.push(("gpu: experts: misses' gate/up", seconds(hits, misses)));
                    phases.push(("gpu: experts: activations summed", seconds(misses, summed)));
                    phases.push(("gpu: experts: down", seconds(summed, end)));
                } else {
                    phases.push(("gpu: experts: no misses", seconds(experts, end)));
                }
            }
            if layer > layers.start {
                phases.push(("gpu: between layers", seconds(at(layer - 1, Stamp::LayerEnd), start)));
            }
        }
        let pass = |point: usize| ticks[self.hp.layers.len() * Stamp::COUNT + point];
        let (pass_start, head_end) = (pass(0), pass(1));
        if let Some((last_end, last_head)) = stamps.last_end {
            if last_head > 0 {
                phases.push(("gpu: boundary: head", seconds(last_end, last_head)));
                phases.push(("gpu: boundary: host between passes", seconds(last_head, pass_start)));
            } else {
                phases.push(("gpu: boundary: head and host between passes", seconds(last_end, pass_start)));
            }
            phases.push(("gpu: boundary: pass start to layer", seconds(pass_start, at(layers.start, Stamp::LayerStart))));
        }
        // (an engine before the last runs no head: its pass ends at its last layer)
        let pass_end = if head_end > 0 { head_end } else { at(layers.end - 1, Stamp::LayerEnd) };
        let pass_ms = seconds(pass_start, pass_end) * 1e3;
        stamps.last_end = Some((at(layers.end - 1, Stamp::LayerEnd), head_end));
        // (every pass of a few rows, one line: the sums of its layers' steps, in ms)
        let mut sums: BTreeMap<&'static str, f64> = BTreeMap::new();
        for (phase, value) in &phases {
            *sums.entry(phase).or_default() += value * 1e3;
        }
        eprintln!("Straydog stamps: rows {n}, pass {pass_ms:.3} ms; {}",
                  sums.iter().map(|(phase, ms)| format!("{} {ms:.3}", phase.trim_start_matches("gpu: "))).collect::<Vec<_>>().join(", "));
        ticks.fill(0);
        self.shards[0].gpu.write_buffer(&stamps.buffer, 0, &ticks)?;
        if n > 1 {
            return Ok(());
        }
        if let Some((_, profile)) = self.profile.as_mut() {
            for (phase, value) in phases {
                *profile.entry(phase).or_default() += value;
            }
        }
        Ok(())
    }

    /// Seconds per phase so far (DS41_PROFILE), and restart the count.
    pub fn take_profile(&mut self) -> BTreeMap<&'static str, f64> {
        match self.profile.as_mut() {
            Some((since, phases)) => {
                *since = std::time::Instant::now();
                std::mem::take(phases)
            }
            None => BTreeMap::new(),
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// The devices this engine runs on, one per shard (memory and identity for reports).
    pub fn gpus(&self) -> impl Iterator<Item = &Backend> {
        self.shards.iter().map(|shard| &shard.gpu)
    }

    /// How many experts the devices hold in VRAM and can keep in the host tier (each device its
    /// rows of the same experts).
    pub fn expert_capacity(&self) -> (usize, usize) {
        self.shards.first().map_or((0, 0), |shard| shard.experts.capacity())
    }

    /// All shards' scheduled/minimum staging/currently pinned host expert bytes.
    pub fn expert_host_memory(&self) -> (u64, u64, u64) {
        self.shards.iter().map(|shard| shard.experts.host_memory())
            .fold((0, 0, 0), |sum, value| (sum.0 + value.0, sum.1 + value.1, sum.2 + value.2))
    }

    /// The experts the devices hold, least recently used first: in VRAM, and only in the host
    /// tier (the first device's; each holds its rows of the same experts).
    pub fn expert_contents(&self) -> crate::experts::Contents {
        self.shards.first().map_or_else(Default::default, |shard| shard.experts.contents())
    }

    /// Warm the expert stores up with an earlier run's contents of this engine's layers (see
    /// `DeviceExperts::warm`), each device its rows; the reads still in flight.
    pub fn warm_experts(&mut self, vram: &[(usize, usize)], host: &[(usize, usize)]) -> Result<(usize, usize, usize)> {
        let (layers, count) = (self.settings.layers.clone(), self.hp.experts);
        let mine = |layer: usize, expert: usize| layers.contains(&layer) && expert < count;
        let ((my_vram, vram_at), (my_host, host_at)) = (select(vram, mine), select(host, mine));
        let (mut reading, mut placed, mut hosted) = (0, my_vram.len(), my_host.len());
        for shard in &mut self.shards {
            let (read, shard_placed, shard_hosted) = shard.experts.warm_bulk(&mut shard.gpu, &my_vram, &my_host)?;
            (reading, placed, hosted) = (reading + read, placed.min(shard_placed), hosted.min(shard_hosted));
        }
        Ok((reading, taken_up_to(placed, &vram_at, vram.len()), taken_up_to(hosted, &host_at, host.len())))
    }

    /// Of a warm-up's lists, this engine's experts that its stores' slots hold (every device's
    /// store has the same slots).
    pub fn fit_warm(&self, vram: &[(usize, usize)], host: &[(usize, usize)]) -> crate::experts::Contents {
        let (layers, count) = (self.settings.layers.clone(), self.hp.experts);
        let mine = |layer: usize, expert: usize| layers.contains(&layer) && expert < count;
        let ((vram, _), (host, _)) = (select(vram, mine), select(host, mine));
        self.shards.first().map_or((Vec::new(), Vec::new()), |shard| shard.experts.fit_warm(&vram, &host))
    }

    /// Take the expert reads that have landed, without waiting; those still in flight.
    pub fn poll_experts(&mut self) -> Result<usize> {
        let mut reading = 0;
        for shard in &mut self.shards {
            reading += shard.experts.poll(&mut shard.gpu)?;
        }
        Ok(reading)
    }

    /// The sequence state a later pass continues, as the first device holds it (for comparing
    /// runs): per layer, its window ring's rows of the last `window` positions (oldest first), and
    /// per kv source layer its compressed rows, index keys and the rows waiting in its tail.
    pub fn state(&mut self) -> Result<Vec<(String, Vec<f32>)>> {
        self.state_on(0)
    }

    /// [`Engine::state`] as device `device` of the engine's holds it (every device holds the same).
    pub fn state_on(&mut self, device: usize) -> Result<Vec<(String, Vec<f32>)>> {
        let (window, dim, index_dim, ring) = (self.hp.window, self.hp.head_dim, self.hp.index_dim, self.ring);
        let shard = self.shards.get_mut(device).ok_or("no such device in the engine")?;
        let mut state = Vec::new();
        for (&layer, buffer) in &shard.rings {
            let mut rows = vec![0f32; ring * dim];
            shard.gpu.read_buffer(buffer, 0, &mut rows)?;
            let last: Vec<f32> = (self.position.saturating_sub(window)..self.position)
                .flat_map(|position| rows[position % ring * dim..(position % ring + 1) * dim].iter().copied()).collect();
            state.push((format!("layer {layer} window"), last));
        }
        for (&layer, buffers) in &shard.sources {
            let source = &self.sources[&layer];
            let mut rows = vec![0u16; source.groups * dim];
            let mut keys = vec![0u16; source.groups * index_dim];
            let mut tail = vec![0f32; 2 * source.tail.min(1) * dim];
            if source.groups > 0 {
                shard.gpu.read_buffer(&buffers.rows, 0, &mut rows)?;
                shard.gpu.read_buffer(&buffers.keys, 0, &mut keys)?;
            }
            let widen = |halves: Vec<u16>| -> Vec<f32> { halves.into_iter().map(|half| f32::from_bits(u32::from(half) << 16)).collect() };
            let (rows, keys) = (widen(rows), widen(keys));
            if !tail.is_empty() {
                shard.gpu.read_buffer(&buffers.tail_kv, 0, &mut tail[..dim])?;
                shard.gpu.read_buffer(&buffers.tail_gate, 0, &mut tail[dim..])?;
            }
            state.extend([(format!("layer {layer} compressed"), rows), (format!("layer {layer} keys"), keys),
                          (format!("layer {layer} tail"), tail)]);
        }
        Ok(state)
    }

    /// Host bytes [`Engine::snapshot`] would take now.
    pub fn snapshot_bytes(&self) -> u64 {
        let groups: Vec<usize> = self.sources.values().map(|source| source.groups).collect();
        snapshot_bytes(self.position, self.hp.window, self.shards[0].rings.len(), self.hp.head_dim, self.hp.index_dim, &groups,
                       self.history.len())
    }

    /// The sequence state a later pass continues, copied to host memory: every device holds the
    /// same (see the module documentation), so from the first. [`Engine::restore`] starts that
    /// sequence over from it, whatever ran in between.
    pub fn snapshot(&mut self) -> Result<Snapshot> {
        let mut snapshot = Snapshot::default();
        self.snapshot_into(&mut snapshot)?;
        self.finish_snapshot(&mut snapshot, usize::MAX)?;
        Ok(snapshot)
    }

    /// [`Engine::snapshot`] into `snapshot`, its buffers taken again (resized, not made anew): a
    /// checkpoint taken over an earlier one touches no new memory but for the sequence's growth,
    /// where new memory every turn (hundreds of MB at long context) waited on the kernel's direct
    /// reclaim whenever the host ran short of free pages. The compressed rows and index keys, nearly
    /// all of it, are left [`pending`](Snapshot::pending) for [`Engine::finish_snapshot`]: the
    /// sequence only grows past them until a reset, a restore or a rewind below the snapshot (a
    /// served checkpoint copied them before the first token, ~0.2 s at 110K tokens and 2-4 s when
    /// the host had to find the pages).
    pub fn snapshot_into(&mut self, snapshot: &mut Snapshot) -> Result<()> {
        let (window, dim, index_dim, ring, position) = (self.hp.window, self.hp.head_dim, self.hp.index_dim, self.ring, self.position);
        let shard = &mut self.shards[0];
        let first = position.saturating_sub(window);
        snapshot.windows.resize_with(shard.rings.len(), Default::default);
        for ((&layer, buffer), (kept_layer, rows)) in shard.rings.iter().zip(&mut snapshot.windows) {
            *kept_layer = layer;
            fit(rows, (position - first) * dim);
            for (at, run, done) in ring_runs(first, position, ring) {
                shard.gpu.read_buffer(buffer, at * dim, &mut rows[done * dim..(done + run) * dim])?;
            }
        }
        snapshot.rows.resize_with(shard.sources.len(), Default::default);
        for ((&layer, buffers), (kept_layer, kept)) in shard.sources.iter().zip(&mut snapshot.rows) {
            *kept_layer = layer;
            let groups = self.sources[&layer].groups;
            // (the compressed rows and index keys as they are, BF16 pairs in F32 words; the dims are even)
            let [rows, keys, tail_kv, tail_gate] = kept;
            fit_room(rows, groups * dim / 2);
            fit_room(keys, groups * index_dim / 2);
            for (buffer, values) in [(&buffers.tail_kv, tail_kv), (&buffers.tail_gate, tail_gate)] {
                fit(values, dim);
                shard.gpu.read_buffer(buffer, 0, values)?;
            }
        }
        snapshot.pending = snapshot.rows.iter().any(|(_, kept)| kept[0].capacity() + kept[1].capacity() > 0);
        snapshot.sources.clear();
        snapshot.sources.extend(self.sources.iter().map(|(&layer, source)| (layer, source.groups, source.tail)));
        fit(&mut snapshot.history, self.history.len());
        snapshot.history.copy_from_slice(&self.history);
        snapshot.position = position;
        Ok(())
    }

    /// Copy up to `most` floats of a [`Engine::snapshot_into`]'s pending compressed rows and index
    /// keys, while the sequence has not gone below it; the floats copied.
    pub fn finish_snapshot(&mut self, snapshot: &mut Snapshot, most: usize) -> Result<usize> {
        if !snapshot.pending {
            return Ok(0);
        }
        let (dim, index_dim) = (self.hp.head_dim, self.hp.index_dim);
        let shard = &mut self.shards[0];
        let (mut copied, mut pending) = (0, false);
        for (layer, kept) in &mut snapshot.rows {
            let groups = snapshot.sources.iter().find(|source| source.0 == *layer).ok_or("a snapshot of another engine's layers")?.1;
            let buffers = shard.sources.get(layer).ok_or("a snapshot of another engine's layers")?;
            for ((buffer, len), values) in [(&buffers.rows, groups * dim / 2), (&buffers.keys, groups * index_dim / 2)].into_iter().zip(kept.iter_mut()) {
                let (done, take) = (values.len(), (len - values.len()).min(most - copied));
                if take > 0 {
                    values.resize(done + take, 0.0);
                    shard.gpu.read_buffer(buffer, done, &mut values[done..])?;
                    copied += take;
                }
                pending |= values.len() < len;
            }
        }
        snapshot.pending = pending;
        Ok(copied)
    }

    /// Put a [`Engine::snapshot`] back on every device: the sequence as it was then.
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        if snapshot.pending {
            return Err("a snapshot restored before its copy finished".into());
        }
        let (dim, ring, position) = (self.hp.head_dim, self.ring, snapshot.position);
        // (the kv sources' buffers back from the expert stores before the snapshot's rows go in)
        self.reserve_state(position)?;
        for shard in &mut self.shards {
            for (layer, rows) in &snapshot.windows {
                let buffer = shard.rings.get(layer).ok_or("a snapshot of another engine's layers")?;
                for (at, run, done) in ring_runs(position - rows.len() / dim, position, ring) {
                    shard.gpu.write_buffer(buffer, (at * dim * 4) as u64, &rows[done * dim..(done + run) * dim])?;
                }
            }
            for (layer, kept) in &snapshot.rows {
                let buffers = shard.sources.get(layer).ok_or("a snapshot of another engine's layers")?;
                for (buffer, values) in [&buffers.rows, &buffers.keys, &buffers.tail_kv, &buffers.tail_gate].into_iter().zip(kept) {
                    if !values.is_empty() {
                        shard.gpu.write_buffer(buffer, 0, values)?;
                    }
                }
            }
        }
        for &(layer, groups, tail) in &snapshot.sources {
            let source = self.sources.get_mut(&layer).ok_or("a snapshot of another engine's layers")?;
            (source.groups, source.tail) = (groups, tail);
        }
        self.position = position;
        self.history.clone_from(&snapshot.history);
        self.rewind = None;
        self.primed = None;
        Ok(())
    }

    /// The expert stores' counters, summed over the devices.
    pub fn cache_counts(&self) -> Counts {
        self.shards.iter().map(|shard| shard.experts.counts()).fold(Counts::default(), |sum, counts| sum + counts)
    }

    /// Between requests: room in each device's store for the next prefill's first waves (see
    /// [`DeviceExperts::make_room`]), at most `budget` write-backs a device. Whether there is.
    pub fn make_room(&mut self, budget: usize) -> Result<bool> {
        let mut done = true;
        for shard in &mut self.shards {
            let slots = 2 * shard.experts.wave_limit().min(WAVE);
            done &= shard.experts.make_room(&mut shard.gpu, slots, budget)?;
        }
        Ok(done)
    }

    /// Wait for the expert stores' reads and copies (before dropping the engine).
    pub fn settle(&mut self) -> Result<()> {
        for shard in &mut self.shards {
            shard.experts.settle(&mut shard.gpu)?;
        }
        Ok(())
    }

    /// Forget the sequence (the caches' contents stay, as warm experts do).
    pub fn reset(&mut self) {
        self.position = 0;
        self.history.clear();
        self.rewind = None;
        self.primed = None;
        for source in self.sources.values_mut() {
            source.groups = 0;
            source.tail = 0;
        }
    }

    /// With several devices: every device's `out` = the sum of the devices' parts `part`, of `rows`
    /// rows from row `first`.
    fn sum_parts(&mut self, part: usize, first: usize, rows: usize) -> Result<()> {
        if self.shards.len() == 1 || rows == 0 {
            return Ok(());
        }
        let w = self.hp.width;
        let devices: Vec<&Backend> = self.shards.iter().map(|shard| &shard.gpu).collect();
        let outs: Vec<M> = self.shards.iter().map(|shard| shard.work.out.at(first * w)).collect();
        let parts: Vec<P> = self.shards.iter().map(|shard| shard.work.parts[part].at(first * w) as P).collect();
        ops::sum_devices(&devices, &outs, &parts, (rows * w) as u64)
    }

    /// Where a device's share of a sublayer's output for a view's rows goes: `out` itself on one
    /// device, else its part.
    fn target(&self, work: &Work, part: usize) -> M {
        if self.shards.len() == 1 { work.out.base() } else { work.parts[part].base() }
    }

    /// Advance the sequence over `tokens` (at most the rows of a pass); the logits after the last
    /// one. The engine runs every layer.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.pass(tokens)?.ok_or_else(|| "this DeepSeek-V4.1 engine runs only some of the layers".into())
    }

    /// Run this engine's layers over `tokens`: the first engine embeds them, a later one continues
    /// the stream handed over to it ([`Engine::hand_over`]); the last returns the logits after the
    /// last token.
    pub fn pass(&mut self, tokens: &[u32]) -> Result<Option<Vec<f32>>> {
        self.run(tokens, 1)
    }

    /// Verify `tokens` (at most [`Engine::verify_rows`], continuing the sequence): one pass
    /// computing every row in every layer, the logits after each token, one row after another, each
    /// bit for bit those of a pass of that token alone. [`Engine::rewind`] takes back the tokens not
    /// accepted. The engine runs every layer.
    pub fn verify(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.verify_pass(tokens)?.ok_or_else(|| "this DeepSeek-V4.1 engine runs only some of the layers".into())
    }

    /// [`Engine::verify`]'s pass of this engine's layers ([`Engine::pass`] otherwise).
    pub fn verify_pass(&mut self, tokens: &[u32]) -> Result<Option<Vec<f32>>> {
        let n = tokens.len();
        // (a pass of more than a sequence's first token has compressed rows, and the first token's
        // attention would run over a selection of none of them, its sums split otherwise than with
        // none: not bit for bit)
        if n == 0 || n > self.verify_rows() || (self.position == 0 && n > 1) {
            return Err(format!("a DeepSeek-V4.1 verify pass takes 1-{} tokens continuing a sequence", self.verify_rows()));
        }
        self.run(tokens, n)
    }

    /// Keep DSpark's target features on every device from now on (the hyper-connections are alike
    /// on all): each pass's last [`TAP_ROWS`] rows of [`TAP_LAYERS`]' inputs, read through
    /// [`Engine::beside`]. Call before the first pass (a recorded decode graph holds whether it taps).
    pub fn enable_tap(&mut self) -> Result<()> {
        let bytes = (TAP_ROWS * TAP_LAYERS.len() * self.hp.width * 4) as u64;
        for shard in &mut self.shards {
            if shard.tap.is_none() {
                shard.tap = Some(shard.gpu.buffer(bytes)?);
            }
        }
        Ok(())
    }

    /// What a drafter beside this engine uses on each device, in device order (after
    /// [`Engine::enable_tap`]): the device, its tap and its rows of the head.
    pub fn beside(&mut self) -> Result<Vec<Beside<'_>>> {
        let mut devices = Vec::with_capacity(self.shards.len());
        for (part, shard) in self.shards.iter_mut().enumerate() {
            let tap = shard.tap.ok_or("no DeepSeek-V4.1 feature tap (Engine::enable_tap)")?;
            let head = shard.dense("output.weight")?;
            let rows = head.dims[1];
            devices.push(Beside { gpu: &mut shard.gpu, tap, head: head.buffer, head_kind: head.kind, head_first: part * rows, head_rows: rows });
        }
        Ok(devices)
    }

    /// Most tokens a verify pass takes: at most [`VERIFY_ROWS`], in a batch, and as many as the
    /// stores hold every route's expert for at once. Several devices acquire a pass's experts in
    /// waves (experts_split), each wave's down summed apart: a verify pass's routes must fit one
    /// wave, or a row's down would be summed otherwise than decode sums it. One device acquires
    /// them all together (routed).
    pub fn verify_rows(&self) -> usize {
        let split = self.shards.len() > 1;
        let experts = self.shards.iter()
            .map(|shard| if split { shard.experts.wave_limit().min(WAVE) } else { shard.experts.pass_limit() }).min().unwrap_or(0);
        VERIFY_ROWS.min(self.settings.batch).min(experts / self.hp.experts_used).max(1)
    }

    /// Each device's kv source buffers taken back from its expert store as far as a sequence of
    /// `tokens` reaches and [`STATE_AHEAD`] more, before a pass writes or reads them (between
    /// passes: no slot held); past [`STATE_KEEP`] more, a shorter sequence's, given back (their
    /// fills waiting for the work queued so far, the last to touch those bytes).
    fn reserve_state(&mut self, tokens: usize) -> Result<()> {
        for shard in &mut self.shards {
            for state in &mut shard.state_regions {
                let reach = |tokens: usize| ((tokens / state.ratio + 1) as u64 * state.row_bytes).min(state.bytes);
                let need = reach(tokens + STATE_AHEAD);
                if need > state.reclaimed {
                    shard.experts.reclaim_region(&mut shard.gpu, state.region, need)?;
                    state.reclaimed = need;
                } else if reach(tokens + STATE_KEEP) < state.reclaimed {
                    let keep = reach(tokens + STATE_KEEP);
                    let after = shard.gpu.mark(Stream::Compute)?;
                    shard.experts.restore_region(state.region, after, keep);
                    state.reclaimed = keep;
                }
            }
        }
        Ok(())
    }

    /// Each device's buffers for a pass of `n`: its own for at most [`OWN_ROWS`] (the arena given
    /// back to its expert store, whose fills into it wait for the passes before), else the arena's,
    /// laid out for `n`, the store giving back the lent slots as far as they reach.
    fn select_work(&mut self, n: usize) -> Result<()> {
        let split = self.shards.len() > 1;
        for shard in &mut self.shards {
            match shard.arena {
                Some(arena) if n > OWN_ROWS => {
                    let work = shard.own.in_arena(arena, n, &shard.shape)?;
                    let reach = shard.shape.bytes(n);
                    if let Some(region) = shard.arena_region.filter(|_| reach > shard.reclaimed) {
                        shard.experts.reclaim_region(&mut shard.gpu, region, reach)?;
                        shard.reclaimed = reach;
                    }
                    // With several devices, each writes only its columns of an expert's gate, up and
                    // output and of the experts' part, the others zero: the arena held other passes'
                    // buffers and lent slots' experts.
                    if split {
                        for buffer in [work.gate, work.up, work.y, work.parts[1]] {
                            shard.gpu.zero(&buffer)?;
                        }
                    }
                    shard.work = work;
                }
                _ => {
                    if let Some(region) = shard.arena_region.filter(|_| shard.reclaimed > 0) {
                        let after = shard.gpu.mark(Stream::Compute)?;
                        shard.experts.restore_region(region, after, 0);
                        shard.reclaimed = 0;
                    }
                    shard.work = shard.own.clone();
                }
            }
        }
        Ok(())
    }

    /// A pass over `tokens` whose last `outputs` rows leave it: on the last engine, the logits
    /// after each of those tokens, one row after another.
    fn run(&mut self, tokens: &[u32], outputs: usize) -> Result<Option<Vec<f32>>> {
        let n = tokens.len();
        if n == 0 || n > self.settings.rows.max(self.settings.batch) || self.position + n > self.settings.context {
            return Err("DeepSeek-V4.1 forward exceeds the rows of a pass or the context".into());
        }
        self.reserve_state(self.position + n)?;
        self.select_work(n)?;
        self.stamp_pass(n, 0)?;
        // A prompt pass's projections take FP16 operands as its FFN's do: in F32, its attention's
        // alone held each device ~6 of a 10.7K-token prompt's 36 s (F32 GEMMs and their weights
        // expanded to F32, every 256 rows).
        let half = self.settings.half_gemm && n >= 32;
        if self.half_policy != Some(half) {
            for shard in &self.shards {
                shard.gpu.configure_gemm(GEMM_MIN_BATCH, half)?;
            }
            self.half_policy = Some(half);
        }
        let hp = self.hp.clone();
        let (hc, w) = (hp.hc, hp.width);
        let range = self.settings.layers.clone();
        self.outputs = outputs;
        self.views.reset_token();
        // (a pass of several rows sums with a kernel: its parts' values, stored one by one into the
        // other's inbox as they form, cost more than the sum's launch saves)
        self.link_pass = self.linked && n == 1;
        if self.link_pass && std::mem::replace(&mut self.link_dirty, true) {
            self.shards[0].gpu.part_reset()?;
        }
        if let Some((since, _)) = self.profile.as_mut() {
            // another engine's pass ran since this one's last mark
            *since = std::time::Instant::now();
        }
        let start = self.position;
        // A pass of a few tokens can be taken back (Engine::rewind): the state from before it.
        let rewind = self.rewindable(n).then(|| Rewind {
            start,
            rows: n,
            sources: self.sources.iter().map(|(&layer, source)| (layer, (source.groups, source.tail))).collect(),
        });
        self.rewind = None;
        let (positions, ring_rows, visible1, visible2) = pass_schedule(start, n, self.ring);
        // The previous pass queued this one-token schedule behind its logit copy.
        let skip = n == 1 && self.primed.take() == Some(start);
        // embeddings repeated into the hc copies (in the buffer kept between passes)
        let mut stream = std::mem::take(&mut self.stream_rows);
        stream.clear();
        if range.start == 0 {
            let embed = self.tensors.tensor("token_embd.weight")?;
            let bytes = row_bytes(embed.kind, embed.dims[0])?;
            stream.reserve(n * hc * w);
            let mut row = Vec::new();
            for (i, &token) in tokens.iter().enumerate() {
                row.clear();
                if token >= vision::IMAGE_TOKEN {
                    row.extend_from_slice(self.image_row(self.position + i)?);
                } else {
                    formats::decode_into(embed.kind, &embed.data[token as usize * bytes..(token as usize + 1) * bytes], &mut row)?;
                }
                for _ in 0..hc {
                    stream.extend_from_slice(&row);
                }
            }
        }
        for shard in &mut self.shards {
            if !(skip && schedule_is_packed(&shard.work, positions.len())) {
                upload_rows(shard, &positions, &ring_rows, &visible1, &visible2)?;
            }
            if !stream.is_empty() {
                shard.gpu.write_buffer(&shard.work.stream, 0, &stream)?;
            }
        }
        self.stream_rows = stream;
        if hp.layers[range.clone()].iter().any(|plan| plan.engram.is_some()) {
            self.prefetch_engram(tokens)?;
        }
        self.image_rows = vision::runs(tokens);
        if !self.image_rows.is_empty() {
            self.upload_image_bias()?;
        }
        self.mark("embedding")?;
        // A handed-over stream comes expanded, with the collapse coefficients of the layer before.
        let mut pre_previous = range.start.checked_sub(1).map(|layer| layer % 2);
        for layer in range.clone() {
            self.stamp(layer, n, Stamp::LayerStart)?;
            self.layer(layer, tokens, n, pre_previous, layer > range.start)?;
            self.stamp(layer, n, Stamp::LayerEnd)?;
            pre_previous = Some(layer % 2);
        }
        let last_layer = range.end - 1;
        let logits = if range.end < hp.layers.len() {
            // the rows the next engine's layers read
            let first = self.rows_of(last_layer, n).2;
            for shard in &mut self.shards {
                let work = shard.work.view(first);
                if self.link_pass {
                    shard.gpu.part_receive(EXPERTS, work.parts[1].base() as P, ((n - first) * w) as u64, work.out.base())?;
                }
                shard.gpu.hc_expand(work.out.base(), work.stream2.base(), work.post_f.base(), work.comb_f.base(), (n - first) as u64,
                                    hc as u64, w as u64, work.stream.base())?;
            }
            if self.link_pass {
                for shard in &mut self.shards {
                    if shard.gpu.part_failed()? {
                        return Err(VOID_PASS.into());
                    }
                }
            }
            None
        } else {
            // The output rows' expansions of the last FFN output, collapsed and normed for the head;
            // each device has its rows of the head.
            let (first, part_rows) = (n - outputs, hp.vocabulary / self.shards.len());
            let mut logits = vec![0f32; outputs * hp.vocabulary];
            let link = self.link_pass;
            for shard in &mut self.shards {
                let work = shard.work.clone();
                let pre = work.pre_f[last_layer % 2];
                let norm = shard.dense("output_norm.weight")?;
                let x = if link { work.parts[1] } else { work.out };
                let expand = (x.at(first * w) as P, work.stream2.at(first * hc * w) as P, work.post_f.at(first * hc) as P,
                              work.comb_f.at(first * hc * hc) as P);
                shard.gpu.hc_step(Some(expand), work.stream.at(first * hc * w), None, 0, 0.0, hp.norm_eps, pre.at(first * hc),
                                  norm.buffer.base(), outputs as u64, hc as u64, w as u64, &work.hc_workspace, work.x.base(),
                                  link.then_some(EXPERTS))?;
                shard.linear("output.weight", work.x.base(), outputs, work.head.base())?;
                if link {
                    // read with the logits, after them
                    shard.gpu.part_status(work.head.at(outputs * part_rows))?;
                }
            }
            self.stamp_pass(n, 1)?;
            let mut failed = false;
            // One read outstanding per device (at most 1 MiB). Queue every device's copy
            // before waiting for any, so the second device's logits move while the first
            // wait runs. A longer read keeps the stream-synchronizing path.
            let read = outputs * part_rows + usize::from(link);
            let bytes = (read * std::mem::size_of::<f32>()) as u64;
            let parallel = bytes > 0 && bytes <= 1 << 20;
            if parallel {
                for shard in &mut self.shards {
                    let head = shard.work.head;
                    shard.gpu.read_start(&head, 0, bytes)?;
                }
                // The next one-token schedule depends only on the position. Queue it on this stream
                // before the copy is waited for. The embedding still waits for the token.
                self.prime_next_row(start + n)?;
            }
            let mut rows = vec![0f32; read];
            for (part, shard) in self.shards.iter_mut().enumerate() {
                let head = shard.work.head;
                if parallel {
                    shard.gpu.read_finish(&mut rows)?;
                } else {
                    shard.gpu.read_buffer(&head, 0, &mut rows)?;
                }
                for row in 0..outputs {
                    let at = row * hp.vocabulary + part * part_rows;
                    logits[at..at + part_rows].copy_from_slice(&rows[row * part_rows..(row + 1) * part_rows]);
                }
                failed |= link && rows[outputs * part_rows].to_bits() != 0;
            }
            if failed {
                return Err(VOID_PASS.into());
            }
            self.mark("head")?;
            Some(logits)
        };
        self.verify_slot_checks()?;
        self.read_stamps(range.clone(), n)?;
        if self.link_pass {
            self.link_dirty = false;
        }
        self.history.extend_from_slice(tokens);
        self.position += n;
        self.rewind = rewind;
        Ok(logits)
    }

    /// Compare the pass's slot checksums (DS41_CHECK_SLOTS), reporting each mismatch.
    fn verify_slot_checks(&mut self) -> Result<()> {
        let position = self.position;
        for (device, shard) in self.shards.iter_mut().enumerate() {
            let Some(checks) = shard.checks.as_mut() else { continue };
            if checks.queued.is_empty() {
                continue;
            }
            let mut partials = vec![0u64; checks.queued.len() * 64];
            shard.gpu.read_buffer(&checks.partials, 0, &mut partials)?;
            let mut before = HashMap::new();
            for (index, &(layer, expert, slot, after)) in checks.queued.iter().enumerate() {
                let sum = partials[index * 64..(index + 1) * 64].iter().fold(0u64, |total, &part| total.wrapping_add(part));
                let what = if after {
                    before.get(&(layer, expert, slot)).filter(|&&was| was != sum).map(|_| "changed while in use")
                } else {
                    before.insert((layer, expert, slot), sum);
                    match checks.first.get(&(layer, expert)) {
                        Some(&first) if first != sum => Some("differs from its first copy"),
                        Some(_) => None,
                        None => {
                            checks.first.insert((layer, expert), sum);
                            None
                        }
                    }
                };
                if let Some(what) = what {
                    checks.mismatches += 1;
                    eprintln!("DS41_CHECK_SLOTS: device {device}, the pass from position {position}: layer {layer} expert {expert} \
                               (slot {slot:#x}) {what}");
                }
            }
            checks.queued.clear();
        }
        Ok(())
    }

    /// Slot checksums that disagreed so far (DS41_CHECK_SLOTS).
    pub fn slot_mismatches(&self) -> u64 {
        self.shards.iter().filter_map(|shard| shard.checks.as_ref()).map(|checks| checks.mismatches).sum()
    }

    /// Queue the next one-token schedule behind the logit copy, when those buffers are the packed
    /// own-row schedule and another token fits. A longer pass uploads its own rows later.
    fn prime_next_row(&mut self, next: usize) -> Result<()> {
        if next >= self.settings.context || self.shards.iter().any(|shard| !schedule_is_packed(&shard.work, 1)) {
            return Ok(());
        }
        let (positions, ring_rows, visible1, visible2) = pass_schedule(next, 1, self.ring);
        for shard in &mut self.shards {
            upload_rows(shard, &positions, &ring_rows, &visible1, &visible2)?;
        }
        self.primed = Some(next);
        Ok(())
    }

    /// [`Work::view`] calls the last pass did not make because a recorded graph replayed
    /// (DS41_COUNT_VIEWS). A decode token's graph of the attention skips two and its graph of the
    /// router skips one, per device; the count stays 0 when it is off.
    pub fn views_avoided(&self) -> u64 {
        self.views.skipped()
    }

    /// Whether a pass of `n` rows has the first device's router publish its routes (see `mailbox`):
    /// a fused router's, of a few rows, with no prediction behind them.
    fn publishes(&self, layer: usize, n: usize, predict: bool) -> bool {
        self.mailbox && !predict && n <= SLOT_ROWS && self.image_rows.is_empty()
            && self.shards[0].dense.get(&name(layer, "ffn_gate_inp")).is_some_and(|router| router.kind == 0 || router.kind == 1 || router.kind == 30)
    }

    /// The routes of the layer's router on the first device: published, or read back (read_start).
    fn take_routes(&mut self, published: bool, routes: &mut [i32]) -> Result<()> {
        if published { self.shards[0].gpu.route_mailbox_wait(routes) } else { self.shards[0].gpu.read_finish(routes) }
    }

    /// Whether a pass of `n` tokens can be taken back: at most [`VERIFY_ROWS`], in one batch.
    fn rewindable(&self, n: usize) -> bool {
        n <= VERIFY_ROWS && n <= self.settings.batch
    }

    /// Take back the last pass but its first `keep` tokens: the sequence state as if only those had
    /// been fed. Only a pass of at most [`VERIFY_ROWS`] tokens can be, and only until the next pass
    /// or a reset. The window rings need nothing: a row of a token taken back sits where no later
    /// token's window reaches before it is written again; nor do the compressed rows and index
    /// keys beyond the groups kept, written before any pass reads them.
    pub fn rewind(&mut self, keep: usize) -> Result<()> {
        let rows = self.rewind.as_ref().ok_or("no pass of a few tokens to take back")?.rows;
        if keep > rows {
            return Err(format!("cannot keep {keep} tokens of a pass of {rows}"));
        }
        let point = self.rewind.take().ok_or("no pass of a few tokens to take back")?;
        let dim = self.hp.head_dim;
        self.position = point.start + keep;
        self.primed = None;
        self.history.truncate(self.history.len() - (point.rows - keep));
        for (layer, (groups, tail)) in point.sources {
            let source = self.sources.get_mut(&layer).ok_or("a kv source went missing")?;
            source.groups = groups + (tail + keep) / source.ratio;
            source.tail = (tail + keep) % source.ratio;
            // The row now waiting is the pass's compressor row `keep` (0: the one waiting before it,
            // where a pass of one token leaves it).
            if source.tail > 0 && keep < point.rows && point.rows > 1 {
                for shard in &mut self.shards {
                    let rows = &shard.sources[&layer];
                    let (pass_kv, pass_gate, tail_kv, tail_gate) = (rows.pass_kv, rows.pass_gate, rows.tail_kv, rows.tail_gate);
                    shard.gpu.copy(pass_kv.at(keep * dim) as P, tail_kv.base(), (dim * 4) as u64)?;
                    shard.gpu.copy(pass_gate.at(keep * dim) as P, tail_gate.base(), (dim * 4) as u64)?;
                }
            }
        }
        Ok(())
    }

    /// Hand the stream of the pass that just ran over `rows` tokens to the engine of the next layers.
    pub fn hand_over(&mut self, next: &mut Engine<'a, T>, rows: usize) -> Result<()> {
        let (hc, w) = (self.hp.hc, self.hp.width);
        let pre = (self.settings.layers.end - 1) % 2;
        if next.settings.layers.start != self.settings.layers.end {
            return Err("DeepSeek-V4.1 engines hand over between adjacent layers only".into());
        }
        next.select_work(rows)?;
        let from = &mut self.shards[0];
        let (stream, coefficients) = (from.work.stream, from.work.pre_f[pre]);
        for to in &mut next.shards {
            from.gpu.copy_to(stream.base(), &mut to.gpu, to.work.stream.base(), (rows * hc * w * 4) as u64)?;
            from.gpu.copy_to(coefficients.base(), &mut to.gpu, to.work.pre_f[pre].base(), (rows * hc * 4) as u64)?;
        }
        Ok(())
    }

    /// The rows of a pass of `n` that `layer` runs, as the first of those it compresses (a kv
    /// source), of those whose K/V it appends to its window ring, and of those it computes: every
    /// row before the decoder; from the decoder on, the rows the outputs (the last row's, every
    /// row's of a verify pass) need and the window before them (see the module documentation).
    fn rows_of(&self, layer: usize, n: usize) -> (usize, usize, usize) {
        match self.decoder {
            Some(decoder) if layer >= decoder => {
                let reach = self.hp.window - 1;
                let full = (n - self.outputs).saturating_sub((self.hp.layers.len() - 1 - layer) * reach);
                let kv = full.saturating_sub(reach);
                (if layer == decoder { 0 } else { kv }, kv, full)
            }
            _ => (0, 0, 0),
        }
    }

    /// One layer over its rows of a pass of `n` ([`Engine::rows_of`]): the attention a batch at a
    /// time, then the experts over all the rows. With `expand`, the previous layer's FFN output (in
    /// `out`) is expanded into `stream` on the way in.
    fn layer(&mut self, layer: usize, tokens: &[u32], n: usize, pre_previous: Option<usize>, expand: bool) -> Result<()> {
        let hp = self.hp.clone();
        let (hc, w) = (hp.hc as u64, hp.width as u64);
        let plan = &hp.layers[layer];
        let (from, kv_from, full_from) = self.rows_of(layer, n);
        // Decode repeats a layer's launches every token: a device queues each run of them that
        // takes no host data at once, from a graph recorded the first time (not while profiling).
        let record = n == 1 && (self.profile.is_none() || !self.profile_sync) && std::env::var_os("DS41_NO_GRAPHS").is_none();
        // a layer that compresses or indexes changes its launches from token to token
        let fixed = plan.compressed.is_none_or(|c| c.kv_source != layer && c.index_source != layer);
        let predict = n == 1 && self.settings.layers.contains(&(layer + 1)) && self.prefetches > 0;
        // A pass that can be taken back keeps its compressor rows (its one batch); of one token it
        // needs none (Engine::rewind).
        let saves = n > 1 && self.rewindable(n);
        // (its rows all in one batch, from the first: rows_of)
        let link = self.link_pass;
        // The Engram rows of every token, before the layer (each batch takes its own), in the
        // buffer kept between passes.
        let mut gathered = std::mem::take(&mut self.engram_values);
        let engram = match plan.engram {
            Some(module) => {
                self.engram_rows(module, tokens, &mut gathered)?;
                true
            }
            None => false,
        };
        let engram_values = hp.engram_columns() * hp.engram.as_ref().map_or(0, |engram| engram.head_dim);
        for first in (from..n).step_by(self.settings.batch) {
            let end = (first + self.settings.batch).min(n);
            let rows = (end - first) as u64;
            let mut expand = expand;
            if engram {
                let expanding = std::mem::take(&mut expand);
                for shard in &mut self.shards {
                    let work = shard.work.view(first);
                    if expanding {
                        if link {
                            shard.gpu.part_receive(EXPERTS, work.parts[1].base() as P, rows * w, work.out.base())?;
                        }
                        shard.gpu.hc_expand(work.out.base(), work.stream2.base(), work.post_f.base(), work.comb_f.base(), rows, hc, w,
                                            work.stream.base())?;
                    }
                    shard.gpu.write_buffer(&work.engram_rows, 0, &gathered[first * engram_values..end * engram_values])?;
                    shard.linear(&name(layer, "engram_kv"), work.engram_rows.base(), rows as usize, work.engram_kv.base())?;
                    let gate = shard.dense(&name(layer, "engram_gate"))?;
                    shard.gpu.engram_gate(work.stream.base(), work.engram_kv.base(), gate.buffer.base(), rows, hc, w, hp.norm_eps)?;
                }
                self.mark("engram")?;
            }
            let attention = self.attention_plan(layer, first, end, kv_from, full_from, saves);
            let key = |segment: u64| {
                (layer as u64) << 8 | u64::from(link) << 5 | segment << 4 | u64::from(expand) << 1 | u64::from(attention.groups > 0)
            };
            // the rows of the batch that attend, from this row
            let attending = first + attention.full_from;
            // A cached graph replays before any view is built (two per device: the batch and the
            // rows that attend). A miss builds them inside the recording. The other path is unchanged.
            let mut avoided = 0u64;
            let devices = self.shards.len();
            self.stamp(layer, n, Stamp::AttentionStart)?;
            // (device 0's stamp between the two, recorded with them)
            let hc_stamp = self.stamps.as_ref().filter(|_| n <= VERIFY_ROWS)
                .map(|stamps| (stamps.buffer.base(), (layer * Stamp::COUNT + Stamp::HcDone as usize) as u64));
            for index in 0..devices {
                if record && fixed {
                    if recorded(&mut self.shards[index], key(0), |shard| {
                        let work = shard.work.view(first);
                        let attend = shard.work.view(attending);
                        let target = if devices == 1 { attend.out.base() } else { attend.parts[0].base() };
                        hc_attention(shard, &work, &hp, layer, rows as usize, expand, pre_previous, link)?;
                        if let Some((at, slot)) = hc_stamp.filter(|_| index == 0) {
                            shard.gpu.debug_stamp(at, slot)?;
                        }
                        tap_features(shard, &work, &hp, layer, first, rows as usize, n)?;
                        attention_ops(shard, &work, &hp, &attention, target, link)
                    })? {
                        avoided += 2;
                    }
                } else {
                    let work = self.shards[index].work.view(first);
                    hc_attention(&mut self.shards[index], &work, &hp, layer, rows as usize, expand, pre_previous, link)?;
                    tap_features(&mut self.shards[index], &work, &hp, layer, first, rows as usize, n)?;
                }
            }
            self.views.replay(avoided);
            if !(record && fixed) {
                self.mark("hyper-connections")?;
                self.stamp(layer, n, Stamp::HcDone)?;
                for index in 0..self.shards.len() {
                    let (work, target) = (self.shards[index].work.view(first), self.target(&self.shards[index].work.view(attending), 0));
                    attention_ops(&mut self.shards[index], &work, &hp, &attention, target, link)?;
                }
            }
            if let Some((ratio, tail, _)) = attention.appended {
                let source = self.sources.get_mut(&layer).unwrap();
                source.groups += (tail + attention.rows) / ratio;
                source.tail = (tail + attention.rows) % ratio;
            }
            if attending == end {
                continue;
            }
            let attended = end - attending;
            if !link {
                self.sum_parts(0, attending, attended)?;
            }
            self.stamp(layer, n, Stamp::AttentionDone)?;
            self.mark("attention")?;
            // (the flag is lowered before the router is queued: it may run at once)
            let publish = self.publishes(layer, n, predict);
            if publish {
                self.shards[0].gpu.route_mailbox_clear()?;
            }
            // One view per device, built only when this graph is not already stored.
            let mut avoided = 0u64;
            for (index, shard) in self.shards.iter_mut().enumerate() {
                // the next layer's experts are predicted on the first device alone, which publishes the routes
                let (predict, publish) = (predict && index == 0, publish && index == 0);
                if record {
                    if recorded(shard, key(1) | u64::from(predict) << 2 | u64::from(publish) << 3, |shard| {
                        let work = shard.work.view(attending);
                        hc_ffn(shard, &work, &hp, layer, attended, link)?;
                        router_ops(shard, &work, &hp, layer, attended, predict, publish)
                    })? {
                        avoided += 1;
                    }
                } else {
                    let work = shard.work.view(attending);
                    hc_ffn(shard, &work, &hp, layer, attended, link)?;
                }
            }
            self.views.replay(avoided);
            if !record {
                self.mark("hyper-connections")?;
                for (index, shard) in self.shards.iter_mut().enumerate() {
                    let work = shard.work.view(attending);
                    router_ops(shard, &work, &hp, layer, attended, predict && index == 0, publish && index == 0)?;
                }
            }
            self.route_images(layer, attending, end)?;
            self.mark("router")?;
        }
        self.engram_values = gathered;
        if full_from < n {
            self.moe(layer, full_from, n, predict)?;
            if !link {
                self.sum_parts(1, full_from, n - full_from)?;
            }
        }
        Ok(())
    }

    /// What every device's attention of `layer` needs to know of a batch of a pass (its rows
    /// `first..end`, those the layer runs from `kv_from` and `full_from` on; with `saves`, the
    /// pass's one) and of the sequence.
    fn attention_plan(&self, layer: usize, first: usize, end: usize, kv_from: usize, full_from: usize, saves: bool) -> AttentionPlan {
        let compressed = self.hp.layers[layer].compressed;
        AttentionPlan {
            layer,
            rows: end - first,
            kv_from: kv_from.clamp(first, end) - first,
            full_from: full_from.clamp(first, end) - first,
            parts: self.shards.len(),
            compressed,
            appended: compressed.filter(|c| c.kv_source == layer).map(|_| {
                let source = &self.sources[&layer];
                (source.ratio, source.tail, source.groups)
            }),
            // the kv source has compressed (at least) through the batch's last row
            groups: compressed.map_or(0, |c| (self.position + end) / c.ratio),
            ring_rows: self.ring as u64,
            saves,
        }
    }

    /// Start reading the Engram rows passes over `tokens` (continuing the held sequence) will
    /// gather, when this engine's layers hold a module ([`Engine::prefetch_engram`]): the session
    /// asks for the next pass's before each pass, so its reads land while that pass runs.
    pub fn prefetch_ahead(&self, tokens: &[u32]) -> Result<()> {
        if self.hp.layers[self.settings.layers.clone()].iter().any(|plan| plan.engram.is_some()) {
            self.prefetch_engram(tokens)?;
        }
        Ok(())
    }

    /// Start reading every Engram row these tokens will gather, while the layers before
    /// the modules run.
    fn prefetch_engram(&self, tokens: &[u32]) -> Result<()> {
        let Some(engram) = self.hp.engram.as_ref() else { return Ok(()) };
        let mut history = self.history[self.history.len().saturating_sub(engram.ngram - 1)..].to_vec();
        let mut rows = vec![Vec::new(); engram.tables.len()];
        for (i, &token) in tokens.iter().enumerate() {
            let before = history[history.len().saturating_sub(engram.ngram - 1)..].to_vec();
            // (an image position gathers nothing: Engram passes it by)
            if token < IMAGE_TOKEN {
                for (module, rows) in rows.iter_mut().enumerate() {
                    engram.rows(module, token, &before, self.position + i, rows);
                }
            }
            history.push(token);
        }
        for (table, rows) in engram.tables.iter().zip(&rows) {
            self.tensors.prefetch_rows(table, rows)?;
        }
        Ok(())
    }

    /// The rows Engram module `module` gathers for `tokens`, decoded (a token's after another's)
    /// into `gathered`.
    fn engram_rows(&self, module: usize, tokens: &[u32], gathered: &mut Vec<f32>) -> Result<()> {
        let engram = self.hp.engram.as_ref().ok_or("no Engram metadata")?;
        let name = &engram.tables[module];
        let table = self.tensors.tensor(name)?;
        let (kind, width) = (table.kind, row_bytes(table.kind, table.dims[0])?);
        let mut history = self.history[self.history.len().saturating_sub(engram.ngram - 1)..].to_vec();
        let mut rows = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let before = &history[history.len().saturating_sub(engram.ngram - 1)..];
            if token < IMAGE_TOKEN {
                engram.rows(module, token, before, self.position + i, &mut rows);
            }
            history.push(token);
        }
        let bytes = self.tensors.rows(name, &rows)?;
        let values = self.hp.engram_columns() * engram.head_dim;
        gathered.clear();
        gathered.reserve(tokens.len() * values);
        // An image position's values are zeros: its gate's key and value are then zero, and the
        // stream passes it unchanged (the reference shuts its gate).
        let mut decoded = bytes.chunks_exact(width);
        for &token in tokens {
            if token >= IMAGE_TOKEN {
                gathered.resize(gathered.len() + values, 0.0);
                continue;
            }
            for row in decoded.by_ref().take(self.hp.engram_columns()) {
                formats::decode_into(kind, row, gathered)?;
            }
        }
        Ok(())
    }

    /// The experts of `layer` over the rows `first..n` of a pass, after the router (see
    /// [`router_ops`]); with `predict`, the first device also predicted the next layer's.
    fn moe(&mut self, layer: usize, first: usize, n: usize, predict: bool) -> Result<()> {
        let rows = n - first;
        if rows > SLOT_ROWS {
            return self.experts_grouped(layer, first, n);
        }
        let hp = self.hp.clone();
        let (w, ff, k) = (hp.width, hp.expert_ff, hp.experts_used);
        // The routes (alike on every device) travel to the host while the shared expert runs:
        // published by the router, or read back.
        let read = rows * k + if predict { PREDICTED } else { 0 };
        let published = self.publishes(layer, n, predict);
        if !published {
            let ids = self.shards[0].work.view(first).ids;
            self.shards[0].gpu.read_start(&ids, 0, (read * 4) as u64)?;
        }
        if self.shards.len() > 1 {
            return self.experts_split(layer, first, rows, read, published);
        }
        // One device: the shared expert writes `out`, the routed ones add to it.
        let shard = &mut self.shards[0];
        let work = shard.work.view(first);
        let shared = shard.shared[&layer];
        shard.gpu.experts(Some(&vec![shared.buffer.base() as u64; rows]), shared.buffer.bytes, shared.offsets, shared.kinds, w as u64,
                          ff as u64, rows as u64, 1, work.x.base(), 0, None, hp.layers[layer].clamp_shared, false, work.hidden.base(),
                          work.out.base(), 0, None)?;
        self.mark("shared expert")?;
        let mut routes = vec![0i32; read];
        self.take_routes(published, &mut routes)?;
        self.predicted = routes.split_off(rows * k).into_iter().take(self.prefetches).map(|expert| expert as usize).collect();
        self.routed(layer, first, rows, &routes)
    }

    /// The routed experts of a few rows (`rows` from `first`) on one device (decode): every
    /// route's expert made resident, then both stages over the slots in one launch each, added to
    /// `out`.
    fn routed(&mut self, layer: usize, first: usize, rows: usize, routes: &[i32]) -> Result<()> {
        let (w, ff) = (self.hp.width, self.hp.expert_ff);
        let (gate_bytes, down_bytes, kinds) = self.expert_kinds(layer, self.shards[0].rows)?;
        let shard = &mut self.shards[0];
        let work = shard.work.view(first);
        let experts: Vec<usize> = routes.iter().map(|&expert| expert as usize).collect::<std::collections::BTreeSet<_>>()
            .into_iter().collect();
        let slots = shard.experts.acquire(&mut shard.gpu, layer, &experts)?;
        check_slots(shard, layer, &experts, &slots, 2 * gate_bytes + down_bytes, false)?;
        // The next layer's predicted experts start their reads and copies behind this layer's.
        let predicted = std::mem::take(&mut self.predicted);
        if !predicted.is_empty() {
            let drive = std::env::var_os("DS41_PREDICT_DRIVE").is_some();
            shard.experts.prefetch(&mut shard.gpu, layer + 1, &predicted, drive)?;
        }
        self.mark("expert fills")?;
        let shard = &mut self.shards[0];
        let bases: Vec<u64> = routes.iter()
            .map(|expert| slots[experts.binary_search(&(*expert as usize)).unwrap()].base() as u64).collect();
        let slot_bytes = slots.first().map_or(0, |slot| slot.bytes);
        let offsets = [0, gate_bytes, 2 * gate_bytes];
        shard.gpu.experts(Some(&bases), slot_bytes, offsets, kinds, w as u64, ff as u64, rows as u64, self.hp.experts_used as u64,
                          work.x.base(), 0, Some(work.weights.base()), self.hp.layers[layer].clamp_routed, true, work.hidden.base(),
                          work.out.base(), 0, None)?;
        check_slots(shard, layer, &experts, &slots, 2 * gate_bytes + down_bytes, true)?;
        shard.experts.release(&mut shard.gpu)?;
        self.mark("routed experts")
    }

    /// A layer's routed expert formats, and the bytes of a device's rows of an expert's gate (or
    /// up) and of its down in a slot.
    fn expert_kinds(&self, layer: usize, rows: Rows) -> Result<(u64, u64, [u32; 3])> {
        let gate = self.tensors.tensor(&name(layer, "ffn_gate_exps"))?;
        let up = self.tensors.tensor(&name(layer, "ffn_up_exps"))?;
        let down = self.tensors.tensor(&name(layer, "ffn_down_exps"))?;
        Ok(((rows.ffn.1 * row_bytes(gate.kind, self.hp.width)?) as u64, (rows.hidden.1 * row_bytes(down.kind, self.hp.expert_ff)?) as u64,
            [gate.kind, up.kind, down.kind]))
    }

    /// The experts of a few rows (`rows` from `first`) on several devices, each with its rows of
    /// every expert: on every device, the shared expert's and every route's gate and up (its share
    /// of the width), the activations summed across devices, then its rows of down into its part of
    /// the output. The routes' experts come in waves (one for decode): a wave's fills start on
    /// every device before any waits, and the next wave's while this one computes.
    fn experts_split(&mut self, layer: usize, first: usize, rows: usize, read: usize, published: bool) -> Result<()> {
        let hp = self.hp.clone();
        let gate_up_first = self.gate_up_first;
        let (w, ff, k) = (hp.width as u64, hp.expert_ff as u64, hp.experts_used);
        let routed_rows = rows * k;
        // Decode sums the activations over a part link (one wave of experts: else with a kernel):
        // the shared expert's row goes first, and the routes' gate and up complete the sum. (With
        // more waves, the shared row's values wait unraised in the inbox slot the next sum fills.)
        let link = self.link_pass;
        self.stamp(layer, rows, Stamp::Routed)?;
        for shard in &mut self.shards {
            let work = shard.work.view(first);
            let shared = shard.shared[&layer];
            let split = if link { shard.rows.sending_product(ACTIVATIONS, routed_rows * ff as usize, true) } else { shard.rows.split(1) };
            shard.gpu.experts(Some(&vec![shared.buffer.base() as u64; rows]), shared.buffer.bytes, shared.offsets, shared.kinds, w, ff,
                              rows as u64, 1, work.x.base(), 0, None, hp.layers[layer].clamp_shared, false,
                              work.products[0].at(routed_rows * ff as usize), work.parts[1].base(), 0, Some(split))?;
        }
        self.mark("shared expert")?;
        self.stamp(layer, rows, Stamp::SharedDone)?;
        let mut routes = vec![0i32; read];
        self.take_routes(published, &mut routes)?;
        self.stamp(layer, rows, Stamp::RoutesTaken)?;
        self.mark("route readback")?;
        let predicted: Vec<usize> = routes.split_off(routed_rows).into_iter().take(self.prefetches).map(|expert| expert as usize).collect();
        let experts: Vec<usize> = routes.iter().map(|&expert| expert as usize).collect::<std::collections::BTreeSet<_>>()
            .into_iter().collect();
        let wave = self.shards.iter().map(|shard| shard.experts.wave_limit()).fold(WAVE, usize::min);
        let waves: Vec<&[usize]> = experts.chunks(wave).collect();
        let drive = std::env::var_os("DS41_PREDICT_DRIVE").is_some();
        if waves.len() == 1 && self.hits_first {
            let (hits, misses): (Vec<usize>, Vec<usize>) = experts.iter().partition(|&&expert| self.shards[0].experts.in_vram(layer, expert));
            if !hits.is_empty() && !misses.is_empty() {
                return self.experts_hits_first(layer, first, rows, &routes, &hits, &misses, &predicted, drive);
            }
        }
        if let Some(first) = waves.first() {
            for shard in &mut self.shards {
                shard.experts.request(&mut shard.gpu, layer, first)?;
            }
        }
        // every device holds as many rows of an expert
        let (gate_bytes, down_bytes, kinds) = self.expert_kinds(layer, self.shards[0].rows)?;
        let expert_bytes = 2 * gate_bytes + down_bytes;
        for (index, wave) in waves.iter().enumerate() {
            let buffer = index % 2;
            let mut acquired = Vec::with_capacity(self.shards.len());
            for shard in &mut self.shards {
                // gate and up start once theirs are in (as in experts_hits_first)
                acquired.push(if shard.checks.is_some() || !gate_up_first { shard.experts.acquire(&mut shard.gpu, layer, wave)? }
                              else { shard.experts.acquire_gate_up(&mut shard.gpu, layer, wave)? });
                check_slots(shard, layer, wave, acquired.last().unwrap(), expert_bytes, false)?;
                if let Some(next) = waves.get(index + 1) {
                    shard.experts.prefetch(&mut shard.gpu, layer, next, true)?;
                } else if !predicted.is_empty() {
                    // the next layer's predicted experts start their reads and copies behind this layer's
                    shard.experts.prefetch(&mut shard.gpu, layer + 1, &predicted, drive)?;
                }
            }
            self.mark("expert fills")?;
            // A route of another wave has no slot (a zero base: skipped).
            let bases = |slots: &[Buffer]| -> Vec<u64> {
                routes.iter().map(|&expert| wave.binary_search(&(expert as usize)).map_or(0, |at| slots[at].base() as u64)).collect()
            };
            let linked = link && waves.len() == 1;
            if index == 0 {
                self.stamp(layer, rows, Stamp::ExpertsStart)?;
            }
            for (shard, slots) in self.shards.iter_mut().zip(&acquired) {
                let work = shard.work.view(first);
                let split = if linked { shard.rows.sending_product(ACTIVATIONS, 0, false) } else { shard.rows.split(1) };
                shard.gpu.experts(Some(&bases(slots)), slots[0].bytes, [0, gate_bytes, 2 * gate_bytes], kinds, w, ff, rows as u64,
                                  k as u64, work.x.base(), 0, Some(work.weights.base()), hp.layers[layer].clamp_routed, true,
                                  work.products[buffer].base(), work.parts[1].base(), 0, Some(split))?;
            }
            // Every device's share of the activations (of the shared expert's too, in the first wave).
            let summed = routed_rows + if index == 0 { rows } else { 0 };
            if linked {
                for shard in &mut self.shards {
                    let work = shard.work.view(first);
                    shard.gpu.part_receive(ACTIVATIONS, work.products[buffer].base() as P, summed as u64 * ff, work.product.base())?;
                }
            } else {
                let devices: Vec<&Backend> = self.shards.iter().map(|shard| &shard.gpu).collect();
                let outs: Vec<M> = self.shards.iter().map(|shard| shard.work.product.base()).collect();
                let parts: Vec<P> = self.shards.iter().map(|shard| shard.work.products[buffer].base() as P).collect();
                ops::sum_devices(&devices, &outs, &parts, summed as u64 * ff)?;
            }
            for (shard, slots) in self.shards.iter_mut().zip(&acquired) {
                let work = shard.work.view(first);
                if index == 0 {
                    let shared = shard.shared[&layer];
                    shard.gpu.experts(Some(&vec![shared.buffer.base() as u64; rows]), shared.buffer.bytes, shared.offsets, shared.kinds,
                                      w, ff, rows as u64, 1, work.x.base(), 0, None, hp.layers[layer].clamp_shared, false,
                                      work.product.at(routed_rows * ff as usize), work.parts[1].base(), 0, Some(shard.rows.split(2)))?;
                }
                shard.experts.await_rest(&mut shard.gpu)?;
                // (the last wave completes the device's part of the output: sent over the part link)
                let split = if self.link_pass && index + 1 == waves.len() { shard.rows.sending(2, EXPERTS) } else { shard.rows.split(2) };
                shard.gpu.experts(None, slots[0].bytes, [0, gate_bytes, 2 * gate_bytes], kinds, w, ff, rows as u64, k as u64,
                                  work.x.base(), 0, None, hp.layers[layer].clamp_routed, true, work.product.base(), work.parts[1].base(),
                                  0, Some(split))?;
                check_slots(shard, layer, wave, slots, expert_bytes, true)?;
            }
            // Queue every down before slot release can prepare write-backs on the host.
            for shard in &mut self.shards {
                shard.experts.release(&mut shard.gpu)?;
            }
        }
        self.mark("routed experts")
    }

    /// [`Engine::experts_split`]'s one wave when VRAM holds some of its experts and not others:
    /// the others' fills start on every device, the held ones' gate and up run while they land,
    /// then the others'; the activations are summed across devices and every route's down runs at
    /// once. A route's gate and up are its own rows, and down sums the routes in route order, so
    /// the result is that of one launch for every route, whatever VRAM held.
    #[allow(clippy::too_many_arguments)]
    fn experts_hits_first(&mut self, layer: usize, first: usize, rows: usize, routes: &[i32], hits: &[usize], misses: &[usize],
                          predicted: &[usize], drive: bool) -> Result<()> {
        let hp = self.hp.clone();
        let gate_up_first = self.gate_up_first;
        let (w, ff, k) = (hp.width as u64, hp.expert_ff as u64, hp.experts_used);
        let routed_rows = rows * k;
        let (gate_bytes, down_bytes, kinds) = self.expert_kinds(layer, self.shards[0].rows)?;
        let (offsets, expert_bytes) = ([0, gate_bytes, 2 * gate_bytes], 2 * gate_bytes + down_bytes);
        for shard in &mut self.shards {
            shard.experts.request(&mut shard.gpu, layer, misses)?;
        }
        self.mark("misses requested")?;
        // A route to an expert of the other list has no slot here (a zero base: skipped).
        let bases = |list: &[usize], slots: &[Buffer]| -> Vec<u64> {
            routes.iter().map(|&expert| list.binary_search(&(expert as usize)).map_or(0, |at| slots[at].base() as u64)).collect()
        };
        let mut held = Vec::with_capacity(self.shards.len());
        self.stamp(layer, rows, Stamp::ExpertsStart)?;
        for shard in &mut self.shards {
            let slots = shard.experts.acquire(&mut shard.gpu, layer, hits)?;
            check_slots(shard, layer, hits, &slots, expert_bytes, false)?;
            let work = shard.work.view(first);
            let split = if self.link_pass { shard.rows.sending_product(ACTIVATIONS, 0, true) } else { shard.rows.split(1) };
            shard.gpu.experts(Some(&bases(hits, &slots)), slots[0].bytes, offsets, kinds, w, ff, rows as u64, k as u64, work.x.base(), 0,
                              Some(work.weights.base()), hp.layers[layer].clamp_routed, true, work.products[0].base(),
                              work.parts[1].base(), 0, Some(split))?;
            held.push(slots);
        }
        self.stamp(layer, rows, Stamp::HitsDone)?;
        self.mark("hits' gate/up")?;
        let mut every = Vec::with_capacity(self.shards.len());
        let mut filled = Vec::with_capacity(self.shards.len());
        for (shard, held) in self.shards.iter_mut().zip(&held) {
            // the misses' gate and up start once theirs are in, their downs landing meanwhile
            // (whole fills under DS41_CHECK_SLOTS, whose checksums read every byte)
            let slots = if shard.checks.is_some() || !gate_up_first { shard.experts.acquire(&mut shard.gpu, layer, misses)? }
                        else { shard.experts.acquire_gate_up(&mut shard.gpu, layer, misses)? };
            check_slots(shard, layer, misses, &slots, expert_bytes, false)?;
            if !predicted.is_empty() {
                shard.experts.prefetch(&mut shard.gpu, layer + 1, predicted, drive)?;
            }
            let work = shard.work.view(first);
            // (the misses' activations complete the device's part of the sum)
            let split = if self.link_pass { shard.rows.sending_product(ACTIVATIONS, 0, false) } else { shard.rows.split(1) };
            shard.gpu.experts(Some(&bases(misses, &slots)), slots[0].bytes, offsets, kinds, w, ff, rows as u64, k as u64, work.x.base(), 0,
                              Some(work.weights.base()), hp.layers[layer].clamp_routed, true, work.products[0].base(),
                              work.parts[1].base(), 0, Some(split))?;
            let both: Vec<u64> = bases(hits, held).into_iter().zip(bases(misses, &slots)).map(|(hit, miss)| hit | miss).collect();
            every.push((both, slots[0].bytes));
            filled.push(slots);
        }
        self.stamp(layer, rows, Stamp::MissesDone)?;
        self.mark("expert fills")?;
        // Every device's share of the activations, the shared expert's too.
        if self.link_pass {
            for shard in &mut self.shards {
                let work = shard.work.view(first);
                shard.gpu.part_receive(ACTIVATIONS, work.products[0].base() as P, (routed_rows + rows) as u64 * ff, work.product.base())?;
            }
        } else {
            let devices: Vec<&Backend> = self.shards.iter().map(|shard| &shard.gpu).collect();
            let outs: Vec<M> = self.shards.iter().map(|shard| shard.work.product.base()).collect();
            let parts: Vec<P> = self.shards.iter().map(|shard| shard.work.products[0].base() as P).collect();
            ops::sum_devices(&devices, &outs, &parts, (routed_rows + rows) as u64 * ff)?;
        }
        self.stamp(layer, rows, Stamp::ActivationsSummed)?;
        for (((shard, (both, slot_bytes)), held), filled) in self.shards.iter_mut().zip(&every).zip(&held).zip(&filled) {
            let work = shard.work.view(first);
            let shared = shard.shared[&layer];
            shard.gpu.experts(Some(&vec![shared.buffer.base() as u64; rows]), shared.buffer.bytes, shared.offsets, shared.kinds, w, ff,
                              rows as u64, 1, work.x.base(), 0, None, hp.layers[layer].clamp_shared, false,
                              work.product.at(routed_rows * ff as usize), work.parts[1].base(), 0, Some(shard.rows.split(2)))?;
            shard.experts.await_rest(&mut shard.gpu)?;
            let split = if self.link_pass { shard.rows.sending(2, EXPERTS) } else { shard.rows.split(2) };
            shard.gpu.experts(Some(both), *slot_bytes, offsets, kinds, w, ff, rows as u64, k as u64, work.x.base(), 0, None,
                              hp.layers[layer].clamp_routed, true, work.product.base(), work.parts[1].base(), 0, Some(split))?;
            check_slots(shard, layer, hits, held, expert_bytes, true)?;
            check_slots(shard, layer, misses, filled, expert_bytes, true)?;
        }
        // Queue every down before slot release can prepare write-backs on the host.
        for shard in &mut self.shards {
            shard.experts.release(&mut shard.gpu)?;
        }
        self.mark("routed experts")
    }

    /// The experts of many rows (`first..n` of a pass), expert by expert: the shared expert over
    /// all of them, then the routed experts in waves, each over every row routed to it (read once
    /// for them all), the next wave's fills started while one computes. On several devices each
    /// runs its rows of every expert: of gate and up into its columns of the activations (the
    /// others zero), summed across devices once a wave, then of down into its columns of its part
    /// of the output.
    fn experts_grouped(&mut self, layer: usize, first: usize, n: usize) -> Result<()> {
        let hp = self.hp.clone();
        let (w, ff, k) = (hp.width, hp.expert_ff, hp.experts_used);
        let (rows, split) = (n - first, self.shards.len() > 1);
        // Small prompts keep F32 after the half path regressed served cold TTFT.
        // Each routed GEMV still keeps its <=4 rule inside the linear op.
        let half_gemm = self.settings.half_gemm && rows >= 32;
        let capacity = self.settings.rows.max(self.settings.batch);
        // The routes (alike on every device), grouped by expert, in waves of at most the stores'
        // limit of experts and as many routes as a pass has rows.
        let (mut ids, mut weights) = (vec![0i32; rows * k], vec![0f32; rows * k]);
        {
            let shard = &mut self.shards[0];
            let work = shard.work.view(first);
            shard.gpu.read_buffer(&work.ids, 0, &mut ids)?;
            shard.gpu.read_buffer(&work.weights, 0, &mut weights)?;
        }
        let grouped = moe::Grouped::new(&ids, &weights, k);
        let limit = self.shards.iter().map(|shard| shard.experts.wave_limit()).fold(WAVE, usize::min);
        let waves = grouped.waves(limit, capacity);
        let experts = |wave: &[moe::Span]| -> Vec<usize> { wave.iter().map(|span| span.expert).collect() };
        for shard in &mut self.shards {
            let work = shard.work.clone();
            shard.gpu.write_buffer(&work.route_rows, 0, &grouped.rows)?;
            shard.gpu.write_buffer(&work.route_weights, 0, &grouped.weights)?;
            // (the layer's misses stream through, leaving the VRAM's experts to decoding)
            shard.experts.set_streaming(&mut shard.gpu, true)?;
            if let Some(wave) = waves.first() {
                shard.experts.request_bulk(&mut shard.gpu, layer, &experts(wave))?;
            }
        }
        // The shared expert over every row (its down writes this device's columns of the output).
        let (clamp_shared, clamp_routed) = (hp.layers[layer].clamp_shared, hp.layers[layer].clamp_routed);
        for shard in &mut self.shards {
            let (work, scratch, shared) = (shard.work.view(first), shard.scratch, shard.shared[&layer]);
            let (f0, fs) = shard.rows.ffn;
            let role = |r: usize, end: u64| shared.buffer.slice(shared.offsets[r], end - shared.offsets[r]);
            let (gate, up) = (role(0, shared.offsets[1]), role(1, shared.offsets[2]));
            shard.gpu.linear_gemm(&gate, shared.kinds[0], fs as u64, w as u64, 0, work.x.base(), 0, rows as u64, work.gate.at(f0), ff as u64,
                             &scratch, half_gemm)?;
            shard.gpu.linear_gemm(&up, shared.kinds[1], fs as u64, w as u64, 0, work.x.base(), 0, rows as u64, work.up.at(f0), ff as u64,
                             &scratch, half_gemm)?;
            shard.gpu.swiglu(work.gate.base(), work.up.base(), rows as u64, ff as u64, clamp_shared, None, work.act.base())?;
        }
        self.sum_activations(rows)?;
        for shard in &mut self.shards {
            let (work, scratch, shared) = (shard.work.view(first), shard.scratch, shard.shared[&layer]);
            let (h0, hs) = shard.rows.hidden;
            let down = shared.buffer.slice(shared.offsets[2], shared.buffer.bytes - shared.offsets[2]);
            let (input, target) = (if split { work.act_sum } else { work.act }, if split { work.parts[1] } else { work.out });
            shard.gpu.linear_gemm(&down, shared.kinds[2], hs as u64, ff as u64, 0, input.base(), 0, rows as u64, target.at(h0), w as u64,
                             &scratch, half_gemm)?;
        }
        self.mark("shared expert")?;
        let (gate_bytes, down_bytes, kinds) = self.expert_kinds(layer, self.shards[0].rows)?;
        // When the next layer's rows route to nearly every expert, its first wave (its first experts)
        // starts its reads behind this layer's last wave, through the next layer's attention.
        let next_rows = if self.settings.layers.contains(&(layer + 1)) { n - self.rows_of(layer + 1, n).2 } else { 0 };
        let ahead: Vec<usize> = if next_rows * k >= AHEAD * hp.experts { (0..limit.min(hp.experts)).collect() } else { Vec::new() };
        // An FP16 pass on RDNA4 runs a wave's experts in one grouped launch a stage, decoding their
        // slots' quantized weights straight into WMMA fragments. Each expert's weights expanded
        // into VRAM for a GEMM of its own held a GPU ~4.6 s of a 10.7K-token prompt.
        let wmma = half_gemm && self.slot_wmma && {
            let shard = &self.shards[0];
            let ((_, fs), (_, hs)) = (shard.rows.ffn, shard.rows.hidden);
            shard.gpu.slot_moe_supported(kinds, w as u64, fs as u64, ff as u64, hs as u64)?
        };
        if wmma {
            for shard in &mut self.shards {
                let work = shard.work.view(first);
                shard.gpu.half_values(work.x.base(), (rows * w) as u64, work.xh.base())?;
            }
        }
        // (a wave's experts as the grouped kernels take them: slots, grouped rows from the wave's
        // first, route counts)
        let slot_wave = |slots: &[Buffer], wave: &[moe::Span]| -> (Vec<u64>, Vec<u32>, Vec<u32>) {
            (slots.iter().map(|slot| slot.base() as u64).collect(), wave.iter().map(|span| (span.first - wave[0].first) as u32).collect(),
             wave.iter().map(|span| span.routes as u32).collect())
        };
        for (index, wave) in waves.iter().enumerate() {
            // A wave's gate and up write the activations again: each device first waits until the
            // others' last sum (the shared expert's, or the wave before's) has read its activations,
            // which a sum reads where they lie.
            if split {
                let devices: Vec<&Backend> = self.shards.iter().map(|shard| &shard.gpu).collect();
                ops::devices_settle(&devices)?;
            }
            let mut acquired = Vec::with_capacity(self.shards.len());
            // (under DS41_PROFILE: the host's share of the fills, apart from the devices' wait)
            let (mut acquiring, mut prefetching) = (0f64, 0f64);
            for shard in &mut self.shards {
                let started = std::time::Instant::now();
                acquired.push(shard.experts.acquire(&mut shard.gpu, layer, &experts(wave))?);
                check_slots(shard, layer, &experts(wave), acquired.last().unwrap(), 2 * gate_bytes + down_bytes, false)?;
                let issued = std::time::Instant::now();
                if let Some(next) = waves.get(index + 1) {
                    shard.experts.prefetch_bulk(&mut shard.gpu, layer, &experts(next), true)?;
                } else if !ahead.is_empty() {
                    shard.experts.prefetch_bulk(&mut shard.gpu, layer + 1, &ahead, true)?;
                }
                acquiring += (issued - started).as_secs_f64();
                prefetching += issued.elapsed().as_secs_f64();
            }
            if let Some((_, phases)) = self.profile.as_mut() {
                *phases.entry("fills: acquire on the host").or_default() += acquiring;
                *phases.entry("fills: prefetch on the host").or_default() += prefetching;
                *phases.entry("fills: waves (count)").or_default() += 1.0;
            }
            self.mark("expert fills")?;
            let base = wave[0].first;
            let routes = wave.iter().map(|span| span.routes).sum::<usize>();
            for (shard, slots) in self.shards.iter_mut().zip(&acquired) {
                let (work, shape) = (shard.work.view(first), shard.expert_shape(&hp, gate_bytes, down_bytes, kinds));
                if wmma {
                    // the wave's FP16 products (this device's columns)
                    let ((_, fs), (_, hs)) = (shard.rows.ffn, shard.rows.hidden);
                    let (bases, firsts, counts) = slot_wave(slots, wave);
                    let grouped = ops::SlotWave { bases: &bases, firsts: &firsts, counts: &counts, gate_bytes, kinds, width: w as u64,
                                                  ffn_share: fs as u64, ffn: ff as u64, hidden_share: hs as u64, tables: &work.tables };
                    shard.gpu.slot_moe_gate_up(&grouped, work.route_rows.at(base), rows as u64, work.xh.base(), clamp_routed,
                                               work.route_weights.at(base), work.ph.base())?;
                    continue;
                }
                let scratch = moe::Scratch { gemm: &shard.scratch, half_gemm, input: &work.xe, gate: &work.gate, up: &work.up, output: &work.y };
                for (span, slot) in wave.iter().zip(slots) {
                    moe::gate_up(&mut shard.gpu, slot, &shape, &scratch, work.x.base(), rows, work.route_rows.at(span.first),
                                 work.route_weights.at(span.first), span.routes, clamp_routed, work.act.at((span.first - base) * ff))?;
                }
            }
            if wmma && split {
                // every device's columns of the wave's FP16 products, gathered from each into every
                // one: the activations whole, as down takes them. (The F32 sum across devices of each
                // one's columns and zeros moved four times the bytes over PCIe: ~1.1 s of a 10.7K-token
                // prompt's 1.5 s of sums on each GPU.)
                let devices: Vec<&Backend> = self.shards.iter().map(|shard| &shard.gpu).collect();
                let outs: Vec<M> = self.shards.iter().map(|shard| shard.work.ah.base()).collect();
                let parts: Vec<P> = self.shards.iter().map(|shard| shard.work.ph.base() as P).collect();
                let firsts: Vec<u64> = self.shards.iter().map(|shard| shard.rows.ffn.0 as u64).collect();
                ops::gather_columns(&devices, &outs, &parts, &firsts, routes as u64, self.shards[0].rows.ffn.1 as u64, ff as u64)?;
            } else if !wmma {
                self.sum_activations(routes)?;
            }
            for (shard, slots) in self.shards.iter_mut().zip(&acquired) {
                let (work, shape) = (shard.work.view(first), shard.expert_shape(&hp, gate_bytes, down_bytes, kinds));
                let (input, target) = (if split { work.act_sum } else { work.act }, if split { work.parts[1] } else { work.out });
                if wmma {
                    // the whole width's FP16 activations (one device: its products are the whole width),
                    // down into this device's columns of the wave's rows, added to their rows' output
                    let ((_, fs), (h0, hs)) = (shard.rows.ffn, shard.rows.hidden);
                    let (bases, firsts, counts) = slot_wave(slots, wave);
                    let grouped = ops::SlotWave { bases: &bases, firsts: &firsts, counts: &counts, gate_bytes, kinds, width: w as u64,
                                                  ffn_share: fs as u64, ffn: ff as u64, hidden_share: hs as u64, tables: &work.tables };
                    let product = if split { work.ah.base() } else { work.ph.base() };
                    shard.gpu.slot_moe_down(&grouped, product, work.y.at(h0), w as u64)?;
                    // (an expert's routes at a time: a row routes to several experts of a wave, and a
                    // scatter's adds to one row race; an expert's routes are of distinct rows)
                    for span in wave.iter() {
                        shard.gpu.scatter_add(work.y.at((span.first - base) * w), work.route_rows.at(span.first), span.routes as u64, w as u64,
                                              target.base(), rows as u64)?;
                    }
                } else {
                    let scratch = moe::Scratch { gemm: &shard.scratch, half_gemm, input: &work.xe, gate: &work.gate, up: &work.up, output: &work.y };
                    for (span, slot) in wave.iter().zip(slots) {
                        moe::down(&mut shard.gpu, slot, &shape, &scratch, input.at((span.first - base) * ff), span.routes,
                                  work.route_rows.at(span.first), target.base(), rows)?;
                    }
                }
                check_slots(shard, layer, &experts(wave), slots, 2 * gate_bytes + down_bytes, true)?;
                shard.experts.release(&mut shard.gpu)?;
            }
        }
        for shard in &mut self.shards {
            shard.experts.set_streaming(&mut shard.gpu, false)?;
        }
        self.mark("routed experts")
    }

    /// With several devices: every device's `act_sum` = the sum of the devices' `act` (each its
    /// columns of `routes` rows of the expert width, the others zero).
    fn sum_activations(&mut self, routes: usize) -> Result<()> {
        if self.shards.len() == 1 || routes == 0 {
            return Ok(());
        }
        let devices: Vec<&Backend> = self.shards.iter().map(|shard| &shard.gpu).collect();
        let outs: Vec<M> = self.shards.iter().map(|shard| shard.work.act_sum.base()).collect();
        let parts: Vec<P> = self.shards.iter().map(|shard| shard.work.act.base() as P).collect();
        ops::sum_devices(&devices, &outs, &parts, (routes * self.hp.expert_ff) as u64)
    }
}

/// What a device's attention of a layer needs to know of a batch of rows and of the sequence: of
/// the batch's rows, the first only compressed (a kv source's beyond the decoder's reach), from
/// `kv_from` those whose K/V go to the window ring, from `full_from` those that attend; the
/// compressed rows a kv source appends (ratio, rows waiting, first group), and those the batch's
/// last row sees; whether a kv source keeps its compressor rows for a rewind.
struct AttentionPlan {
    layer: usize,
    rows: usize,
    kv_from: usize,
    full_from: usize,
    /// devices the heads are split across
    parts: usize,
    compressed: Option<super::Compressed>,
    appended: Option<(usize, usize, usize)>,
    groups: usize,
    ring_rows: u64,
    saves: bool,
}

/// Work::view calls skipped while a decode token's graphs replay (DS41_COUNT_VIEWS; otherwise unused).
struct ViewCounter {
    enabled: bool,
    skipped: u64,
}

impl ViewCounter {
    fn new(enabled: bool) -> Self {
        Self { enabled, skipped: 0 }
    }

    fn reset_token(&mut self) {
        if self.enabled {
            self.skipped = 0;
        }
    }

    fn replay(&mut self, built: u64) {
        if self.enabled {
            self.skipped += built;
        }
    }

    fn skipped(&self) -> u64 {
        self.skipped
    }
}

/// Queue `ops` from the graph stored under `key`, recording it the first time.
/// True when that graph replayed, so `ops` (which builds the views) did not run.
/// The replay is one lookup. Only the first recording looks the key up again, to launch it.
fn recorded(shard: &mut Shard, key: u64, ops: impl FnOnce(&mut Shard) -> Result<()>) -> Result<bool> {
    if shard.gpu.graph_run(key)? {
        return Ok(true);
    }
    shard.gpu.graph_begin()?;
    let result = ops(shard);
    shard.gpu.graph_end(key, result.is_ok())?;
    result?;
    if shard.gpu.graph_run(key)? { Ok(false) } else { Err("a recorded graph went missing".into()) }
}

/// A sublayer's hyper-connection weights as hc_step takes them: BF16 ones tagged (see state_tagged).
fn hc_weights(fn_: &Dense) -> M {
    if fn_.kind == 30 { state_tagged(&fn_.buffer) } else { fn_.buffer.base() }
}

/// The hyper-connection weights of a sublayer and its input norm.
fn sublayer(shard: &Shard, layer: usize, side: &str, norm: &str) -> Result<(Dense, Dense, Dense, Dense)> {
    Ok((shard.dense(&name(layer, &format!("hc_{side}_fn")))?, shard.dense(&name(layer, &format!("hc_{side}_base")))?,
        shard.dense(&name(layer, &format!("hc_{side}_scale")))?, shard.dense(&name(layer, norm))?))
}

/// The hyper-connection step into a layer's attention; with `expand`, the previous layer's FFN
/// output (in `out`, or with `link` the sum of the devices' parts over the part link) is expanded
/// into `stream` on the way in.
#[allow(clippy::too_many_arguments)]
fn hc_attention(shard: &mut Shard, work: &Work, hp: &Hparams, layer: usize, n: usize, expand: bool, pre_previous: Option<usize>,
                link: bool) -> Result<()> {
    let (fn_, base, scale, norm) = sublayer(shard, layer, "attn", "attn_norm")?;
    let pre = pre_previous.map_or(work.identity, |slot| work.pre_f[slot]);
    let x = if link { work.parts[1] } else { work.out };
    let expansion = expand.then(|| (x.base() as P, work.stream2.base() as P, work.post_f.base() as P, work.comb_f.base() as P));
    shard.gpu.hc_step(expansion, work.stream.base(), Some((hc_weights(&fn_), base.buffer.base(), scale.buffer.base(), work.pre_a.base(),
                      work.post_a.base(), work.comb_a.base())), hp.sinkhorn_iterations as u64, hp.hc_eps, hp.norm_eps, pre.base(),
                      norm.buffer.base(), n as u64, hp.hc as u64, hp.width as u64, &work.hc_workspace, work.x.base(),
                      (link && expand).then_some(EXPERTS))
}

/// The hyper-connection step from a layer's attention (in `out`, or with `link` the sum of the
/// devices' parts over the part link) into its FFN.
fn hc_ffn(shard: &mut Shard, work: &Work, hp: &Hparams, layer: usize, n: usize, link: bool) -> Result<()> {
    let (fn_, base, scale, norm) = sublayer(shard, layer, "ffn", "ffn_norm")?;
    let x = if link { work.parts[0] } else { work.out };
    let expansion = (x.base() as P, work.stream.base() as P, work.post_a.base() as P, work.comb_a.base() as P);
    shard.gpu.hc_step(Some(expansion), work.stream2.base(), Some((hc_weights(&fn_), base.buffer.base(), scale.buffer.base(),
                      work.pre_f[layer % 2].base(), work.post_f.base(), work.comb_f.base())), hp.sinkhorn_iterations as u64, hp.hc_eps,
                      hp.norm_eps, work.pre_a.base(), norm.buffer.base(), n as u64, hp.hc as u64, hp.width as u64, &work.hc_workspace,
                      work.x.base(), link.then_some(ATTENTION))
}

/// A device's attention of a batch of rows (`work` a view from its first): q_a, kv, the compressed
/// rows and the indexer (as every device), and its heads of q, the attention, wo_a and wo_b into
/// `target` (from the first row that attends; with `link`, this device's part of the part link's
/// sum, sent). Rows before `kv_from` are only compressed, rows before `full_from` only append their
/// K/V (see [`AttentionPlan`]).
fn attention_ops(shard: &mut Shard, work: &Work, hp: &Hparams, plan: &AttentionPlan, target: M, link: bool) -> Result<()> {
    let layer = plan.layer;
    let (kv_rows, n) = (plan.rows - plan.kv_from, plan.rows - plan.full_from);
    let (heads, dim, rope) = (hp.heads / plan.parts, hp.head_dim, hp.rope_dims as u64);
    let groups = hp.out_groups / plan.parts;
    let (kv_work, q_work) = (work.view(plan.kv_from), work.view(plan.full_from));
    let freqs = if plan.compressed.is_some() { shard.compressed_rope } else { shard.plain };
    let scratch = shard.scratch;
    let whole = |weight: &Dense, input: M, output: M| Linear {
        weights: weight.buffer, kind: weight.kind, rows: weight.dims[1] as u64, cols: weight.dims[0] as u64, first_row: 0,
        input: input as P, output,
    };
    let (q_a, kv) = (shard.dense(&name(layer, "attn_q_a"))?, shard.dense(&name(layer, "attn_kv"))?);
    let ring = shard.rings[&layer];
    // q (this device's heads) stays unrotated: the attention rotates it as it reads it. A decode
    // row's q_a norm (into qn, for q_b and the indexer) and its kv row's norm and rope run in the
    // last block of q_a's and kv's launch.
    // q is not rotated here. The KV norm uses this layer's frequencies and, when its rows are the
    // queries, their positions: it writes the rope table, so attention adds no launch. A KV prefix
    // that does not attend keeps attention's own launch.
    let same_rows = plan.kv_from == plan.full_from && rope > 0 && n > 0 && kv_rows > 0;
    shard.gpu.attention_turns_arm(if same_rows { kv_rows as u64 } else { 0 })?;
    let fused = n == 1 && kv_rows == 1;
    let normed_q = if fused { work.qn } else { work.qr };
    if fused {
        let (q_a_norm, kv_norm) = (shard.dense(&name(layer, "attn_q_a_norm"))?, shard.dense(&name(layer, "attn_kv_a_norm"))?);
        let side = ops::NormRope {
            input: work.kv.base() as P, weight: &kv_norm.buffer, width: dim as u64, eps: hp.norm_eps,
            rope: Some((rope, &kv_work.positions, &freqs)), out_rows: Some(kv_work.ring_rows.base() as P), out: ring.base(),
            out_count: plan.ring_rows,
        };
        shard.gpu.linears_norm_rope(&[whole(&q_a, q_work.x.base(), work.qr.base()), whole(&kv, q_work.x.base(), work.kv.base())],
                                    work.qr.base(), &q_a_norm.buffer, hp.q_rank as u64, hp.norm_eps, work.qn.base(), &side)?;
        shard.linear(&name(layer, "attn_q_b"), work.qn.base(), 1, work.q.base())?;
    } else if kv_rows == n {
        if n > 0 {
            shard.gpu.linears(&[whole(&q_a, q_work.x.base(), work.qr.base()), whole(&kv, q_work.x.base(), work.kv.base())], n as u64, 0,
                              0, &scratch)?;
        }
    } else {
        shard.gpu.linears(&[whole(&kv, kv_work.x.base(), work.kv.base())], kv_rows as u64, 0, 0, &scratch)?;
        if n > 0 {
            shard.gpu.linears(&[whole(&q_a, q_work.x.base(), work.qr.base())], n as u64, 0, 0, &scratch)?;
        }
    }
    if !fused {
        if n > 0 {
            shard.norm(&name(layer, "attn_q_a_norm"), work.qr.base(), n, hp.q_rank, hp.norm_eps)?;
            shard.linear(&name(layer, "attn_q_b"), work.qr.base(), n, work.q.base())?;
        }
        if kv_rows > 0 {
            let kv_norm = shard.dense(&name(layer, "attn_kv_a_norm"))?;
            shard.gpu.norm_rope(work.kv.base(), &kv_norm.buffer, kv_rows as u64, dim as u64, hp.norm_eps,
                                Some((rope, &kv_work.positions, &freqs)), Some(kv_work.ring_rows.base()), ring.base(), plan.ring_rows)?;
        }
    }
    let mut compressed = None;
    if let Some(c) = plan.compressed {
        if let Some((ratio, tail, first_group)) = plan.appended {
            compress(shard, work, hp, layer, plan.rows, freqs, ratio, tail, first_group, plan.saves)?;
        }
        if c.index_source == layer && plan.groups > 0 && n > 0 {
            let keys = shard.sources[&c.kv_source].keys;
            shard.linear(&name(layer, "indexer.attn_q_b"), normed_q.base(), n, work.index_q.base())?;
            // index_q rotated as the scores read it, and each row's top groups, in one launch (a decode
            // row's with the indexer's projection of x into its weights too)
            let scale = 1.0 / ((hp.index_dim as f32).sqrt() * (hp.index_heads as f32).sqrt());
            let chosen = q_work.chosen[&layer];
            let rotation = (rope, &q_work.positions, &freqs);
            // The two-level indexer (the reference's): the candidate source's scores pick each row's
            // candidate blocks, and a later indexer chooses among their groups only. While a row's groups
            // fit in its candidate blocks every one of them is a candidate: then neither is needed.
            let two_level = hp.candidates.filter(|&(_, blocks, block)| plan.groups > blocks * block);
            let within = two_level.filter(|&(source, _, _)| source < layer)
                .map(|(_, blocks, block)| (q_work.candidates.base().cast_const(), blocks as u64, block as u64));
            if n == 1 {
                let proj = shard.dense(&name(layer, "indexer.proj"))?;
                if proj.dims[1] != hp.index_heads {
                    return Err(format!("layer {layer}'s indexer projection has {} rows for {} heads", proj.dims[1], hp.index_heads));
                }
                shard.gpu.index_select_proj((&proj.buffer, proj.kind, proj.dims[0] as u64), q_work.x.base(), work.index_q.base(), state_tagged(&keys),
                                            work.index_w.base(), q_work.visible[c.ratio].base(), plan.groups as u64, hp.index_heads as u64,
                                            hp.index_dim as u64, scale, rotation, work.scores.base(), hp.index_top_k as u64, chosen.base(),
                                            within)?;
            } else {
                shard.linear(&name(layer, "indexer.proj"), q_work.x.base(), n, work.index_w.base())?;
                shard.gpu.index_select(work.index_q.base(), state_tagged(&keys), work.index_w.base(), q_work.visible[c.ratio].base(), n as u64,
                                       plan.groups as u64, hp.index_heads as u64, hp.index_dim as u64, scale, rotation, work.scores.base(),
                                       hp.index_top_k as u64, chosen.base(), within)?;
            }
            if let Some((_, blocks, block)) = two_level.filter(|&(source, _, _)| source == layer) {
                shard.gpu.index_candidates(work.scores.base(), q_work.visible[c.ratio].base(), n as u64, plan.groups as u64, blocks as u64,
                                           block as u64, q_work.candidates.base(), &scratch)?;
            }
        }
        if plan.groups > 0 {
            compressed = Some((shard.sources[&c.kv_source].rows, plan.groups, q_work.chosen[&c.index_source]));
        }
    }
    if n == 0 {
        return Ok(());
    }
    let sinks = shard.dense(&name(layer, "attn_sinks"))?;
    shard.gpu.sparse_attention(work.q.base(), ring.base(), plan.ring_rows, q_work.positions.base(), hp.window as u64,
                               compressed.map(|(rows, groups, _)| (state_tagged(&rows) as P, groups as u64)),
                               compressed.map(|(_, _, chosen)| (chosen.base() as P, hp.index_top_k as u64)), sinks.buffer.base(),
                               n as u64, heads as u64, dim as u64, 1.0 / (dim as f32).sqrt(), Some((rope, &freqs)), Some(&scratch),
                               work.attended.base())?;
    // wo_a is block-diagonal: group g projects its heads' slice alone.
    let wo_a = shard.dense(&name(layer, "attn_output_a"))?;
    let group_width = hp.heads * dim / hp.out_groups;
    let projections: Vec<Linear> = (0..groups).map(|g| Linear {
        weights: wo_a.buffer, kind: wo_a.kind, rows: hp.out_rank as u64, cols: wo_a.dims[0] as u64, first_row: (g * hp.out_rank) as u64,
        input: work.attended.at(g * group_width) as P, output: work.low.at(g * hp.out_rank),
    }).collect();
    shard.gpu.linears(&projections, n as u64, (heads * dim) as u64, (groups * hp.out_rank) as u64, &scratch)?;
    if link {
        return shard.linear_send(ATTENTION, &name(layer, "attn_output_b"), work.low.base(), n, target);
    }
    shard.linear(&name(layer, "attn_output_b"), work.low.base(), n, target)
}

/// With a feature tap (a drafter beside the engine), a tapped layer's input after its hyper-connections
/// (Engram applied, before attention) into the tap: the mean of each row's hc copies, for the
/// pass's last [`TAP_ROWS`] rows (`n` rows; this batch's `rows` from row `first`). Only the tap
/// is written: the layer's results are those without it.
fn tap_features(shard: &mut Shard, work: &Work, hp: &Hparams, layer: usize, first: usize, rows: usize, n: usize) -> Result<()> {
    let (Some(tap), Some(slot)) = (shard.tap, TAP_LAYERS.iter().position(|&tapped| tapped == layer)) else { return Ok(()) };
    let keep = n.saturating_sub(TAP_ROWS);
    if first + rows <= keep {
        return Ok(());
    }
    let (w, skip) = (hp.width, keep.saturating_sub(first));
    let stride = TAP_LAYERS.len() * w;
    shard.gpu.hc_mean(work.stream.base(), rows as u64, hp.hc as u64, w as u64, skip as u64, tap.at((first + skip - keep) * stride),
                      stride as u64, (slot * w) as u64)
}

/// A layer's router on a device (every device routes alike); with `predict`, also the next
/// layer's router on this layer's input, its best PREDICTED experts after the routes in `ids`.
fn router_ops(shard: &mut Shard, work: &Work, hp: &Hparams, layer: usize, n: usize, predict: bool, publish: bool) -> Result<()> {
    let k = hp.experts_used;
    let gating = match hp.gating { Gating::Softmax => 0, Gating::Sigmoid => 1, Gating::SqrtSoftplus => 2 };
    // (the router and its routes in one launch where it can)
    let router = shard.dense(&name(layer, "ffn_gate_inp"))?;
    let bias = shard.dense(&format!("blk.{layer}.exp_probs_b.bias"))?;
    let scratch = shard.scratch;
    if publish {
        // (the routes also to the device's route mailbox: Engine::take_routes)
        shard.gpu.linear_route_published(&router.buffer, router.kind, router.dims[1] as u64, router.dims[0] as u64, work.x.base(),
                                         n as u64, work.router.base(), Some(bias.buffer.base()), k as u64, gating, hp.route_normalize,
                                         hp.route_scale, work.ids.base(), work.weights.base(), &scratch)?;
    } else {
        shard.gpu.linear_route(&router.buffer, router.kind, router.dims[1] as u64, router.dims[0] as u64, work.x.base(), n as u64,
                               work.router.base(), Some(bias.buffer.base()), k as u64, gating, hp.route_normalize, hp.route_scale,
                               work.ids.base(), work.weights.base(), &scratch)?;
    }
    if predict {
        let next_bias = shard.dense(&format!("blk.{}.exp_probs_b.bias", layer + 1))?;
        shard.linear(&name(layer + 1, "ffn_gate_inp"), work.x.base(), 1, work.router.at(hp.experts))?;
        shard.gpu.route(work.router.at(hp.experts), Some(next_bias.buffer.base()), 1, hp.experts as u64, PREDICTED as u64, gating,
                        hp.route_normalize, hp.route_scale, work.ids.at(k), work.route_weights.base())?;
    }
    Ok(())
}

/// Append this pass's complete groups of a kv source layer to a device's compressed rows and index
/// keys (from `first_group`, with `tail` rows of an incomplete group waiting); the rows of a still
/// incomplete group wait in its tail. With `save` (a pass that can be taken back) and groups of 2,
/// the compressor rows (the one waiting first) are kept for a rewind. The groups appended and the
/// rows left over.
#[allow(clippy::too_many_arguments)]
fn compress(shard: &mut Shard, work: &Work, hp: &Hparams, layer: usize, n: usize, freqs: Buffer, ratio: usize, tail: usize,
            first_group: usize, save: bool) -> Result<(usize, usize)> {
    let dim = hp.head_dim;
    // rows 1..=n take this pass; row 0 the waiting one, when there is one
    shard.linear(&name(layer, "attn_compressor_kv"), work.x.base(), n, work.pool_kv.at(dim))?;
    if ratio > 1 {
        shard.linear(&name(layer, "attn_compressor_gate"), work.x.base(), n, work.pool_gate.at(dim))?;
    }
    let (rows, keys, tail_kv, tail_gate, pass_kv, pass_gate) = {
        let source = &shard.sources[&layer];
        (source.rows, source.keys, source.tail_kv, source.tail_gate, source.pass_kv, source.pass_gate)
    };
    if tail > 0 {
        shard.gpu.gather(tail_kv.base(), 1, work.all_rows.base(), 1, dim as u64, work.pool_kv.base())?;
        shard.gpu.gather(tail_gate.base(), 1, work.all_rows.base(), 1, dim as u64, work.pool_gate.base())?;
    }
    if save && ratio > 1 {
        // (the next kv source's pass reuses the pool)
        let bytes = ((n + 1) * dim * 4) as u64;
        shard.gpu.copy(work.pool_kv.base() as P, pass_kv.base(), bytes)?;
        shard.gpu.copy(work.pool_gate.base() as P, pass_gate.base(), bytes)?;
    }
    let first_row = 1 - tail;
    let groups = (tail + n) / ratio;
    let left = (tail + n) % ratio;
    if groups > 0 {
        // (formed in F32, the indexer's keys from the rows before their rope, then both rounded into the state)
        let staging = shard.staging.ok_or("no staging for a kv source's rows")?;
        let target = staging.base();
        shard.gpu.compress(work.pool_kv.at(first_row * dim), (ratio > 1).then(|| work.pool_gate.at(first_row * dim) as P),
                           groups as u64, ratio as u64, dim as u64, target)?;
        shard.norm(&name(layer, "attn_compressor_norm"), target, groups, dim, hp.norm_eps)?;
        let key_target = staging.at(groups * dim);
        let weight = shard.dense(&name(layer, "indexer.attn_k"))?;
        let scratch = shard.scratch;
        shard.gpu.linear(&weight.buffer, weight.kind, weight.dims[1] as u64, weight.dims[0] as u64, 0, target, 0, groups as u64,
                         key_target, 0, &scratch)?;
        let starts: Vec<i32> = (first_group..first_group + groups).map(|g| (g * ratio) as i32).collect();
        shard.gpu.write_buffer(&work.group_positions, 0, &starts)?;
        let key_norm = shard.dense(&name(layer, "indexer.k_norm"))?;
        shard.gpu.norm_rope(key_target, &key_norm.buffer, groups as u64, hp.index_dim as u64, hp.norm_eps,
                            Some((hp.rope_dims as u64, &work.group_positions, &freqs)), None, key_target, groups as u64)?;
        shard.gpu.rope(target, groups as u64, 1, dim as u64, hp.rope_dims as u64, &work.group_positions, &freqs, false)?;
        shard.gpu.to_bf16(target, (groups * dim) as u64, halves_at(&rows, first_group * dim))?;
        shard.gpu.to_bf16(key_target, (groups * hp.index_dim) as u64, halves_at(&keys, first_group * hp.index_dim))?;
    }
    if left > 0 {
        let last = first_row + groups * ratio;
        shard.gpu.gather(work.pool_kv.at(last * dim), 1, work.all_rows.base(), 1, dim as u64, tail_kv.base())?;
        shard.gpu.gather(work.pool_gate.at(last * dim), 1, work.all_rows.base(), 1, dim as u64, tail_gate.base())?;
    }
    Ok((groups, left))
}

/// Whether an engine over several devices divides this resident tensor among them (each holds its
/// part: of the head and the attention's q_b, sinks and output its heads' or groups', of a shared
/// expert its rows); every other resident tensor is whole on each device (the indexer's among
/// them: its q_b is not the attention's).
pub fn divided(name: &str) -> bool {
    match name.strip_prefix("blk.").and_then(|rest| rest.split_once('.')) {
        Some((_, role)) => ["attn_q_b", "attn_sinks", "attn_output_a", "attn_output_b", "ffn_gate_shexp", "ffn_up_shexp",
                            "ffn_down_shexp"].iter().any(|divided| role.strip_suffix(".weight") == Some(divided)),
        None => name == "output.weight",
    }
}

/// The rows the attention of a pass runs at a time when a pass takes many (a server's prefill):
/// each batch expands the attention's weights for its GEMMs once (an 8K pass on two GPUs: 64 rows
/// 32 s, 256 rows 26 s, 512 rows 25 s), and its scores take a row of the context each.
pub const ATTENTION_ROWS: usize = 256;

/// The bytes a device's buffers of every row take per row of a pass (see [`Settings::rows`]), on
/// an engine of `parts` devices over `layers`: as `Shard::new` allocates them.
pub fn pass_row_bytes(hp: &Hparams, parts: usize, layers: std::ops::Range<usize>) -> u64 {
    let (hc, w, ff, k) = (hp.hc, hp.width, hp.expert_ff, hp.experts_used);
    let indexes = hp.layers.iter().enumerate().filter(|(layer, plan)| layers.contains(layer)
                                                                          && plan.compressed.is_some_and(|c| c.index_source == *layer)).count();
    let split = usize::from(parts > 1);
    // the stream twice, its coefficients, x, out, an expert's input and output rows, the parts,
    // positions and visible counts, selections, routes, gate, up and the activations (and their sum)
    // (and the grouped WMMA kernels' FP16 input row, products and summed activations: w / 2 + ff)
    let values = 2 * hc * w + 6 * hc + 2 * hc * hc + 4 * w + 2 * split * w + 5 + indexes * hp.index_top_k + 4 * k + (3 + split) * ff
        + w / 2 + ff + candidate_blocks(hp, layers).unwrap_or(0);
    (values * 4) as u64
}

/// The candidate blocks a row keeps for the layers `layers` (the candidate source's indexer picks them
/// for the later ones): none without a candidate source there; an error when a later indexer there
/// would read them from another engine's.
fn candidate_blocks(hp: &Hparams, layers: std::ops::Range<usize>) -> Result<usize> {
    let Some((source, blocks, _)) = hp.candidates else { return Ok(0) };
    let users = hp.layers.iter().enumerate().any(|(layer, plan)| layers.contains(&layer) && layer > source
                                                                 && plan.compressed.is_some_and(|c| c.index_source == layer));
    match (layers.contains(&source), users) {
        (true, _) => Ok(blocks),
        (false, false) => Ok(0),
        (false, true) => Err(format!("DeepSeek-V4.1 layers {layers:?} choose among layer {source}'s candidates, held by another engine")),
    }
}

/// One pass over engines that run consecutive layers (on their own devices): each hands the stream
/// to the next; the logits after the last token.
pub fn forward_split<T: Tensors>(engines: &mut [Engine<'_, T>], tokens: &[u32]) -> Result<Vec<f32>> {
    split_pass(engines, tokens, false)
}

/// [`Engine::verify`] over engines that run consecutive layers: the logits after each token.
pub fn verify_split<T: Tensors>(engines: &mut [Engine<'_, T>], tokens: &[u32]) -> Result<Vec<f32>> {
    split_pass(engines, tokens, true)
}

fn split_pass<T: Tensors>(engines: &mut [Engine<'_, T>], tokens: &[u32], verify: bool) -> Result<Vec<f32>> {
    for i in 0..engines.len().saturating_sub(1) {
        if verify { engines[i].verify_pass(tokens)? } else { engines[i].pass(tokens)? };
        let (done, rest) = engines.split_at_mut(i + 1);
        done[i].hand_over(&mut rest[0], tokens.len())?;
    }
    let last = engines.last_mut().ok_or("no DeepSeek-V4.1 engines")?;
    if verify { last.verify(tokens) } else { last.forward(tokens) }
}

impl Work {
    /// These buffers with those of every row but identity for a pass of `rows` laid out in
    /// `arena`, one after another from its start ([`RowShape::floats`]).
    fn in_arena(&self, arena: Buffer, rows: usize, shape: &RowShape) -> Result<Work> {
        if shape.bytes(rows) > arena.bytes {
            return Err("a DeepSeek-V4.1 pass's rows exceed their arena".into());
        }
        let mut at = 0u64;
        let mut buffers = shape.floats(rows).into_iter().map(|floats| {
            let (start, bytes) = (at.next_multiple_of(256), (floats.max(1) * 4) as u64);
            at = start + bytes;
            arena.slice(start, bytes)
        });
        let mut next = || buffers.next().expect("a buffer of every row for each in RowShape::floats");
        // (fields are taken in the order written, as RowShape::floats lists them)
        Ok(Work {
            stream: next(), stream2: next(), pre_a: next(), post_a: next(), comb_a: next(), pre_f: [next(), next()], post_f: next(),
            comb_f: next(), x: next(), out: next(), parts: [next(), next()], positions: next(), ring_rows: next(),
            visible: [next(), next(), next()], chosen: self.chosen.keys().map(|&layer| (layer, next())).collect(), candidates: next(),
            ids: next(),
            weights: next(), route_rows: next(), route_weights: next(), xe: next(), gate: next(), up: next(), act: next(),
            act_sum: next(), y: next(), xh: next(), ph: next(), ah: next(), ..self.clone()
        })
    }

    /// The buffers with those of every row starting at row `first` (the rest as they are).
    fn view(&self, first: usize) -> Work {
        let (hc, w, k, top_k, blocks) = self.widths;
        let at = |buffer: Buffer, values: usize| {
            let offset = (first * values * 4) as u64;
            buffer.slice(offset.min(buffer.bytes), buffer.bytes.saturating_sub(offset))
        };
        Work {
            stream: at(self.stream, hc * w), stream2: at(self.stream2, hc * w), pre_a: at(self.pre_a, hc), post_a: at(self.post_a, hc),
            comb_a: at(self.comb_a, hc * hc), pre_f: self.pre_f.map(|buffer| at(buffer, hc)), post_f: at(self.post_f, hc),
            comb_f: at(self.comb_f, hc * hc), identity: at(self.identity, hc), x: at(self.x, w), out: at(self.out, w),
            parts: self.parts.map(|buffer| at(buffer, w)), positions: at(self.positions, 1), ring_rows: at(self.ring_rows, 1),
            visible: self.visible.map(|buffer| at(buffer, 1)),
            chosen: self.chosen.iter().map(|(&layer, &buffer)| (layer, at(buffer, top_k))).collect(),
            candidates: at(self.candidates, blocks), ids: at(self.ids, k),
            weights: at(self.weights, k), ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::divided;

    #[test]
    fn a_view_count_stays_off_until_a_replay_is_counted() {
        let mut off = super::ViewCounter::new(false);
        off.replay(2);
        off.replay(1);
        assert_eq!(off.skipped(), 0);
        let mut on = super::ViewCounter::new(true);
        on.reset_token();
        on.replay(2);
        on.replay(1);
        assert_eq!(on.skipped(), 3, "a device's attention graph skips two views and its router graph one");
        on.reset_token();
        assert_eq!(on.skipped(), 0);
    }

    #[test]
    fn the_next_token_s_schedule_is_the_row_after_this_pass() {
        let (positions, ring_rows, visible1, visible2) = super::pass_schedule(5, 2, 8);
        assert_eq!((&positions[..], &ring_rows[..], &visible1[..], &visible2[..]), (&[5, 6][..], &[5, 6][..], &[6, 7][..], &[3, 3][..]));
        let (positions, ring_rows, visible1, visible2) = super::pass_schedule(7, 1, 8);
        assert_eq!((positions, ring_rows, visible1, visible2), (vec![7], vec![7], vec![8], vec![4]));
    }

    #[test]
    fn a_pass_packs_its_four_position_vectors_into_one_image() {
        let packed = super::pack_schedule(4, &[10, 11], &[2, 3], &[7, 8], &[1, 1]);
        assert_eq!(packed, vec![10, 11, 0, 0, 2, 3, 0, 0, 7, 8, 0, 0, 1, 1, 0, 0]);
    }

    #[test]
    fn only_the_attention_and_the_shared_experts_and_the_head_are_divided() {
        for name in ["blk.2.attn_q_b.weight", "blk.39.attn_output_b.weight", "blk.0.ffn_down_shexp.weight", "output.weight"] {
            assert!(divided(name), "{name}");
        }
        for name in ["blk.2.indexer.attn_q_b.weight", "blk.2.indexer.attn_k.weight", "blk.2.attn_q_a.weight", "output_norm.weight",
                     "blk.2.attn_q_b.bias", "token_embd.weight"] {
            assert!(!divided(name), "{name}");
        }
    }
}
