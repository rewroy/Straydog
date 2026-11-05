//! The Qwen3.8-Flash-Next decoder on resident GPU layers, with MTP speculative decoding and
//! conversation checkpoints.
//!
//! Every layer runs in native spans on the GPUs; the host gathers embedding rows, takes the
//! first maximum of the logits and samples. Speculative decoding drafts with the model's own
//! MTP layer and verifies up to eight tokens per pass with the one-token arithmetic, so greedy
//! output is identical to token-by-token decode.
//!
//! A prompt resumes from the longest prefix it shares with what the model holds: the live
//! sequence when the prompt extends it, else the deepest conversation checkpoint it matches
//! (the end of the system prompt, and where the last user-initiated and the last tool-result
//! request began generating).

use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::constraint::{self, Constraint, Suppression};
use crate::drafts::{Ngrams, Policy, LONG_MATCH};
use crate::gpu::{expert_bytes, Columns, CpuExperts, Data, Operand, Piece, Result, Scores, HIDDEN, WIDTH};
use crate::group::{row_count, Group, Positions};
use crate::qwen_vision::{self, ImageSpan, IMAGE_PAD};
use crate::sampling::{self, Generated, Sampling};
use crate::vision::IMAGE_TOKEN;
use crate::weights::{ngram_rows, Progress, Weights, CONTEXT_LIMIT};

/// The most drafts a pass can verify. MTP drafts stay at three (--draft-max): at 20K, 2048 tokens greedy 154.8 tok/s at three, 153.9
/// at four and five, 153.3 at seven; sampled 141.2 at three, 137.6 at five, 125.9 at seven. The rows past four are for drafts
/// that cost no drafting and are nearly always accepted: all accepted, 20K, milliseconds a token by the pass's rows 4.21 at four,
/// 3.47 at eight.
pub const MAX_DRAFTS: usize = crate::gpu::VERIFY_MAX - 1;
/// The per-layer (PLE) n-gram embedding table, read by rows from the host.
const PLE_TABLE: &str = "per_layer_token_embd.weight";
/// Most rows a verification pass takes: a token and its drafts.
const VERIFY_ROWS: usize = crate::gpu::VERIFY_MAX;
/// Most rows a partially resident model's pass takes (gpu/hybrid_slots.hpp, hybrid_rows).
const PARTIAL_ROWS: usize = 4;
/// Rows a pass of a short prompt takes: every row is kept, and four cost less per row than a wider pass.
const PROMPT_ROWS: usize = 4;
/// Longest fully resident prompt that runs as verification passes (about 27 ms per four rows)
/// rather than as a batched prefill, whose fixed cost is about 110 ms on two R9700s.
const VERIFIED_PREFILL: usize = 16;
/// The tokens that end a reply.
pub const END_TOKENS: [u32; 2] = [248044, 248046];
const IM_START: u32 = 248045;
const IM_END: u32 = 248046;
/// Drafts stop where the product of their probabilities under the drafter falls below
/// this. Replayed over 16 prompts, with and without reasoning, greedy and with the model
/// card's sampling (every pass drafting three), 0.6 was best or within 1% everywhere: 3-9%
/// more tokens per second than an acceptance-rate estimate. A draft row costs about 3.5 ms of
/// verification and 1.35 ms of drafting against a 15 ms pass.
const DRAFT_CUT: f32 = 0.6;
/// [`DRAFT_CUT`] of a partially resident model: its pass costs 17 ms with a second row at 23, nearly all of it the row's CPU experts,
/// so a draft pays from about 0.35. At 0.3 (IQ2_XS, 12 GiB) a token took 13.9 ms, against 14.3 at 0.6.
const PARTIAL_DRAFT_CUT: f32 = 0.3;
/// [`DRAFT_CUT`] of drafts drawn from the drafter's top, under the product of each drawn distribution's greatest probability: a draft
/// is kept with the overlap of the two distributions, which a flat row lowers less than it lowers that. Served IQ3_S on two GPUs,
/// tok/s of four requests: 132 with every pass at four rows, 137 at 0.3, 134 at 0.6 (the most likely token at 0.6: 130).
const DRAWN_DRAFT_CUT: f32 = 0.3;
/// A draft's vocabulary with the head split across two GPUs: the head's first 100000 rows (English, code and the common CJK
/// characters: every token of the served cases' replies) and the rows of the tokens past them that the sequence has held (a
/// Japanese text's words: 97-99.6% of its tokens with the text before it, 82% without). A draft step's head reads 40% of the
/// rows: a pass 0.6 ms shorter, the same drafts on the served cases. Verification reads the whole head.
/// `STRAYDOG_DRAFT_VOCABULARY=0` drafts over the whole head.
const DRAFT_ROWS: u64 = 100_000;
/// Tokens past DRAFT_ROWS a draft's vocabulary holds at most (sd_draft_vocabulary's space).
const DRAFT_EXTRAS: usize = 16384;
/// Conversation checkpoints, each a page-locked host copy of the recurrent state (~118 MB):
/// root (the end of the system prompt), turn (where the reply to the last user message
/// began), prior (the same for the message before) and step (the reply to the last tool result).
const CHECKPOINTS: [&str; 4] = ["root", "turn", "prior", "step"];
/// Recurrent-state slots after the checkpoints' for parked sequences: one parked, one for the
/// live sequence it trades places with.
const PARK_SLOTS: [usize; 2] = [CHECKPOINTS.len(), CHECKPOINTS.len() + 1];
/// A live sequence is parked when a prompt departs from it at least this far below its end.
const PARK_MINIMUM: usize = 1024;
/// Only a system prompt this long is worth a copy.
const ROOT_MINIMUM: usize = 2048;
const VOCABULARY: usize = 248320;

/// Host bytes of one sequence's state before the GPU takes its attention and recurrence: each
/// attention layer's (the MTP layer's too, `drafting`) QSA K/V, pooled indexer keys and the
/// indexer's rolling history (eight inputs), DeltaNet state and convolution, PLE history.
pub fn context_state_bytes(context: u64, drafting: bool) -> u64 {
    let state = attention_layers(drafting) * (context * 2 * 2 * 256 + (context / 4 + 1) * 128 + 8 * 128) * 4;
    state + 36 * (48 * 128 * 128 + 4 * 10240) * 4 + 10 * 10240 * 4
}

/// The layers with QSA attention: twelve, and the MTP layer when drafting.
fn attention_layers(drafting: bool) -> u64 {
    12 + u64::from(drafting)
}

/// The end of a K/V growth's error when page-locking found no memory (`sd_kv_grow`): every state
/// is sound, and a shorter sequence can still run.
pub const KV_NO_MEMORY: &str = "not enough host memory for the K/V of this sequence";

/// The end of the error of what an image asked and the devices could not give, or of a prompt whose image rows would run in a
/// pass of one row, which no pass of its kind takes (the chat template cannot render one: its images are followed by text enough).
/// The devices are sound: the request fails, and the engine resets.
pub const IMAGE_REFUSED: &str = "(a request with an image, refused)";

/// The PLE table's row cache of a partially resident model with less memory than twice its banks.
pub const SCARCE_ROW_CACHE: u64 = 1 << 30;
/// The commit a K/V growth leaves the machine.
const KV_COMMIT_MARGIN: u64 = 2 << 30;
/// The positions a partially resident model's host K/V starts with: page-locked, 12 GiB at the
/// whole context take from a small machine what the expert banks need, and few sequences reach it.
pub const STARTING_KV: usize = 32768;

/// A verification's buffers beside partially resident layers (`verify_reserve`), and the VRAM for expert slots a drafter must
/// leave to stay: a pass of rows the CPU would mostly compute gains little.
const VERIFY_BYTES: u64 = 130 << 20;
const DRAFTER_SLOT_ROOM: u64 = 128 << 20;
/// Speed first for a partially resident model: STRAYDOG_EXPERT_FLOOR, a router weight below which a route the CPU would run is
/// left out of its token (its weight taken as 0). Unset or 0: every route runs.
pub fn expert_floor() -> Result<Option<f32>> {
    match std::env::var("STRAYDOG_EXPERT_FLOOR") {
        Ok(value) if !value.is_empty() => {
            let floor: f32 = value.parse().map_err(|_| format!("STRAYDOG_EXPERT_FLOOR {value:?}: a router weight from 0 to 1"))?;
            if !(0.0..1.0).contains(&floor) {
                return Err(format!("STRAYDOG_EXPERT_FLOOR {value:?}: a router weight from 0 to 1").into());
            }
            Ok((floor > 0.0).then_some(floor))
        }
        _ => Ok(None),
    }
}

/// The share of the routed experts the slots hold at load from which a partially resident model drafts two tokens a pass. Routing is uneven:
/// IQ3_S on an R9700, the slots of `--vram 24` (30% at load, 39% once the K/V has settled) serve 87% of the routes, and the second draft
/// costs less CPU. Two drafts against one, same seeds, tok/s of a second and third request: `--vram 24` 69.5, 70.5 / 67.7, 61.4; 16 (17%) 58.7,
/// 57.3 / 57.4, 56.6; 12 (under 15%) 50.1, 49.1 / 50.7, 48.0. HIP only: CUDA has not been measured.
const TWO_DRAFTS_SHARE: f64 = 0.15;
/// The experts' slots a partial placement keeps out of its pool for staging, its VRAM margin (`Group::create_expert_slots`), and the room its gate
/// asks beyond them (`Model::prepare_hybrid_moe`).
pub(crate) const STAGING_SLOTS: u64 = 32;
pub(crate) const SLOTS_MARGIN: u64 = 64 << 20;
const STAGING_GATE: u64 = 128 << 20;

/// The VRAM a partially resident Qwen3.8 without a drafter leaves its expert slots in a `budget` on one GPU, with `dense` bytes of pinned
/// weights ([`crate::weights::Weights::dense_requirements`]), experts of up to `expert` bytes and a context of `context` tokens that its
/// K/V may grow to; None if the placement would refuse it ([`Model::prepare_hybrid_moe`]'s gate at the K/V a model starts with), or a
/// growth to the whole context find no room for the longer indexers. What `straydog setup` tells of a machine rests on this.
pub fn partial_slot_bytes(dense: u64, expert: u64, context: u64, prefill: u64, budget: u64) -> Option<u64> {
    let decoder = crate::gpu::decoder_workspace_bytes(context.min(prefill));
    let kv = context.min(STARTING_KV as u64);
    let starting = crate::gpu::partial_state_bytes(kv, context, prefill, budget);
    let stage = STAGING_SLOTS * expert;
    let gate = starting + dense + stage + STAGING_GATE + decoder + crate::gpu::Backend::slack(budget);
    if gate > budget {
        return None;
    }
    budget.checked_sub(crate::gpu::partial_state_bytes(context, context, prefill, budget) + dense + decoder + stage + SLOTS_MARGIN)
}

/// Page-locked host bytes of the recurrent state's copies: [`CHECKPOINTS`] and [`PARK_SLOTS`].
pub fn checkpoint_bytes() -> u64 {
    (CHECKPOINTS.len() + PARK_SLOTS.len()) as u64 * (36 * (48 * 128 * 128 + 4 * 10240) * 4 + 10 * 10240 * 4)
}
/// New tokens from which a partially resident model's batches take the slots' VRAM: past one batch, the second batch's streaming costs more
/// than putting the slots back, a fraction of a second.
const LONG_PREFILL: usize = 1536;
/// A partially resident model's passes in a row with a draft: one alone then keeps [`Model::partial_drawn_cut`]'s measure current.
const PASSES_BETWEEN_LONE_ROWS: u32 = 64;

/// Expert bank bytes to page-lock given the host memory (0 if unknown): all of them with
/// memory twice their size; with at least 1.25 times, half the memory less the page-locked
/// sequence state; else none (the unlocked layers and the PLE table would thrash).
pub fn lock_budget(memory: u64, banks: u64, pinned_state: u64) -> u64 {
    if memory == 0 || 4 * memory < 5 * banks {
        return 0;
    }
    if memory >= 2 * banks {
        return banks;
    }
    (memory / 2).saturating_sub(pinned_state)
}

/// With DS41_PROFILE set: seconds per phase of a request's decode, and of its passes by rows
/// (count, seconds, phases). The host's clock alone: every call a phase covers returns after its
/// device work, so no extra synchronization is added.
#[derive(Default)]
struct Profile {
    phases: BTreeMap<&'static str, f64>,
    /// The phases of the pass under way, until its rows are known.
    pass: BTreeMap<&'static str, f64>,
    rows: [(u64, f64, BTreeMap<&'static str, f64>); VERIFY_ROWS + 1],
    /// A partially resident model's export wait and CPU seconds so far, for what each pass adds.
    hybrid: (f64, f64),
}

impl Profile {
    /// The pass under way ends: it had `rows` rows and took `seconds`.
    fn end_pass(&mut self, rows: usize, seconds: f64) {
        self.rows[rows].0 += 1;
        self.rows[rows].1 += seconds;
        for (name, spent) in std::mem::take(&mut self.pass) {
            *self.phases.entry(name).or_default() += spent;
            *self.rows[rows].2.entry(name).or_default() += spent;
        }
    }
}

struct Checkpoint {
    slot: usize,
    tokens: Vec<u32>,
    final_hidden: Option<Vec<f32>>,
    /// With the MTP layer, its indexer's rolling history then: resuming here, the MTP layer's
    /// pair before completes its pooled block from the three inputs before it.
    history: Option<Positions>,
}

/// A sequence a prompt departed from, kept to come back to: its recurrent state in a slot and
/// its indexers' rolling history, taken when it departed, and of its positional state only what
/// later prompts would overwrite, copied out before they write it (`saved`, covering `covered`).
/// Below the covered positions the live sequence holds the same tokens; above them, to the
/// sequence's end, the positional state is still where the parked sequence left it (borrowed:
/// any write there first extends the coverage, [`Model::protect`]), so a short prompt after a
/// long conversation copies a few hundred positions, not the conversation.
struct Parked {
    tokens: Vec<u32>,
    final_hidden: Option<Vec<f32>>,
    slot: usize,
    history: Option<Positions>,
    saved: Vec<Positions>,
    covered: Option<(usize, usize)>,
}

impl Parked {
    /// Its lowest position copied out (its end while none is).
    fn from(&self) -> usize {
        self.covered.map_or(self.tokens.len(), |(from, _)| from)
    }
}

/// `tokens` with `held`'s tokens over its text, as far as the two spell the same text: only the split into tokens differs there.
fn same_text<'a>(tokens: &[u32], held: &[u32], bytes: &dyn Fn(u32) -> Option<&'a [u8]>) -> Vec<u32> {
    /// Tokens on either side of a difference: past them the two are taken to have parted.
    const REACH: usize = 32;
    let mut agreed = Vec::with_capacity(tokens.len());
    let (mut at, mut from) = (0, 0);
    while at < held.len() && from < tokens.len() {
        if held[at] == tokens[from] {
            agreed.push(held[at]);
            at += 1;
            from += 1;
            continue;
        }
        let (mut one, mut other) = (at, from);
        let (mut first, mut second) = (Vec::new(), Vec::new());
        let met = loop {
            let of_held = first.len() <= second.len();
            let token = if of_held { held.get(one) } else { tokens.get(other) };
            // (a special token is itself, whatever other tokens spell the same)
            let Some(piece) = token.and_then(|token| bytes(*token)) else { break false };
            if of_held {
                first.extend_from_slice(piece);
                one += 1;
            } else {
                second.extend_from_slice(piece);
                other += 1;
            }
            let shorter = first.len().min(second.len());
            if first[..shorter] != second[..shorter] || one - at > REACH || other - from > REACH {
                break false;
            }
            if first.len() == second.len() && !first.is_empty() {
                break true;
            }
        };
        if !met {
            break;
        }
        agreed.extend_from_slice(&held[at..one]);
        at = one;
        from = other;
    }
    agreed.extend_from_slice(&tokens[from..]);
    agreed
}

/// How many leading tokens two sequences share.
fn shared(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

pub struct Settings {
    pub context: usize,
    pub state_budget_bytes: u64,
    pub prefill_batch_size: usize,
    pub speculative: usize,
    /// CPU expert threads for partial residency (0: one per physical core).
    pub cpu_threads: u32,
    /// Bytes of the row cache the PLE table is read through (`crate::rows`); 0 reads its rows
    /// through the mapping and the page cache instead; none, as much as the host memory leaves
    /// ([`crate::rows::auto_budget`]).
    pub row_cache_bytes: Option<u64>,
}

/// A small deterministic generator for sampling (xoshiro256**, seeded by SplitMix64).
pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Rng([next(), next(), next(), next()])
    }

    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    pub fn random(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// An index drawn with the given (not necessarily normalized) weights.
    pub fn choice(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut target = self.random() * total;
        for (index, weight) in weights.iter().enumerate() {
            if target < *weight {
                return index;
            }
            target -= weight;
        }
        weights.iter().rposition(|weight| *weight > 0.0).unwrap_or(0)
    }
}

/// Softmax of `logits / temperature` in double precision.
pub fn softmax(logits: &[f32], temperature: f64) -> Vec<f64> {
    let peak = logits.iter().fold(f32::NEG_INFINITY, |peak, value| peak.max(*value)) as f64;
    let weights: Vec<f64> = logits.iter().map(|value| ((*value as f64 - peak) / temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|weight| weight / total).collect()
}

/// The first maximum.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as u32
}

/// Each embedding row repeated over the four residual streams.
fn tile(embeddings: &[f32]) -> Vec<f32> {
    let mut hidden = Vec::with_capacity(embeddings.len() * 4);
    for row in embeddings.chunks_exact(HIDDEN) {
        for _ in 0..4 {
            hidden.extend_from_slice(row);
        }
    }
    hidden
}

/// A verification pass: the first layer's rows, the PLE embeddings, the final rows, each
/// row's argmax and, when sampling, the rows' logits.
type Verified = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<i32>, Option<Scores>);

/// Seconds since `clock`, which restarts.
fn lap(clock: &mut Instant) -> f64 {
    let seconds = clock.elapsed().as_secs_f64();
    *clock = Instant::now();
    seconds
}

fn last_row(values: &[f32]) -> &[f32] {
    &values[values.len() - WIDTH..]
}

pub struct Model {
    // Field order is drop order: the devices go before the CPU engine they use, and both
    // before the mappings their native plans point into.
    group: Group,
    cpu: Option<CpuExperts>,
    pub weights: Weights,
    pub context: usize,
    pub prefill_batch_size: usize,
    pub state_bytes: u64,
    pub speculative: usize,
    /// The most drafts a long match's copy verifies (drafts::Policy, by the match's length).
    pub copy_drafts: usize,
    pub resident: bool,
    pub partial: bool,
    pub host_kv: bool,
    pub history: Vec<u32>,
    final_hidden: Option<Vec<f32>>,
    checkpoints: IndexMap<&'static str, Checkpoint>,
    parked: Option<Parked>,
    /// Host bytes parked positions may take.
    park_budget: u64,
    /// Park a departing sequence whole at once ([`Model::park_whole`]).
    park_whole: bool,
    pub last_metrics: Map<String, Value>,
    /// When set, the SHA-256 of every logits vector the model returns (a determinism check).
    pub logits_digests: Option<Vec<String>>,
    /// When set, every logits vector the model returns (validation tooling).
    pub logits_capture: Option<Vec<Vec<f32>>>,
    /// When set, a speculative decode verifies a known sequence at given widths, with no drafter (see [`ForcedRows`]).
    pub forced_rows: Option<ForcedRows>,
    /// Inside a pipelined prefill: batches before the last return no rows.
    deferred: bool,
    /// Rows of the last batch the MTP layer's device kept (0: none, see `mtp_prompt_kept`).
    mtp_kept: u64,
    /// A draft's vocabulary (DRAFT_ROWS) in use; the tokens past DRAFT_ROWS it holds, and those noted but not yet on the device.
    draft_vocabulary: bool,
    draft_extras: HashSet<u32>,
    draft_pending: Vec<u32>,
    /// Seconds each step of loading took, in order.
    pub load_seconds: Vec<(&'static str, f64)>,
    profile: Option<Profile>,
    /// Running means of a decode pass's seconds with one row and with two (0: none yet), for [`Model::partial_drawn_cut`].
    pass_seconds: [f64; 2],
    passes_with_draft: u32,
    /// This request's passes by rows, and this process's memory as it began, for its log line.
    pass_counts: [u64; 2],
    memory_before: Option<(u64, u64, u64)>,
    /// The CPU engine's read-ahead hints (taken, left) when the request began.
    read_aheads_before: (u64, u64),
    /// Positions the K/V holds now (the context, unless it started small: `ensure_kv`).
    kv_capacity: usize,
    /// What is page-locked, for the thread that keeps the working set's limit.
    pinned_now: std::sync::Arc<std::sync::atomic::AtomicU64>,
    watching_memory: bool,
    /// Page-locked host bytes besides the K/V.
    pinned_other: u64,
    /// The expert banks a partially resident model reads from host memory.
    banks_bytes: u64,
    /// The group's counters when the request began (a request's partial residency counts).
    hybrid_counts: Value,
    /// The PLE table's row counts when the request began, and the seconds it has waited for rows since.
    ple_counts: (u64, u64, u64),
    ple_wait: f64,
    /// The request's images, by their first positions: their rows and what they do to the rope positions.
    images: Vec<ImageSpan>,
    /// What the devices hold for a sequence with images: the delta of its decode and verification rows (`sd_rope_delta`) and
    /// whether a table of rope positions is up (`sd_rope_rows`). A sequence without any costs a test of these two.
    rope_delta: u32,
    rope_table: bool,
}

/// A measurement of what a verification pass costs by its rows (examples/prefill --force-rows): the rows are the sequence's next
/// tokens, the widths taken in turn, with the drafter out of the pass. Given the greedy tokens of the prompt, every draft is
/// accepted: no replay, and the sequence's kept rows hashed show that a width changes no bit of them.
#[derive(Default)]
pub struct ForcedRows {
    /// The reply as a greedy decode gave it.
    pub tokens: Vec<u32>,
    /// A pass's rows, taken in turn.
    pub widths: Vec<usize>,
    /// (rows, milliseconds) of every pass.
    pub passes: Vec<(usize, f64)>,
    /// The kept rows of the final hidden states, and with `read_logits` their logits, in order.
    pub digest: sha2::Sha256,
    pub logits_digest: sha2::Sha256,
    pub read_logits: bool,
    /// Passes with a draft that was not accepted (without `reject`, not zero means the rows are not the one-token arithmetic's).
    pub rejected: usize,
    /// Every this many passes (0: none), a draft made wrong, its place going round: what follows must not see the rows taken back.
    pub reject: usize,
}

impl Model {
    /// Place the model on the group's GPUs, then read and authenticate the weights into the
    /// placement (see `crate::load`); `progress` follows the reading.
    pub fn new(weights: Weights, mut group: Group, settings: &Settings, progress: Progress) -> Result<Model> {
        let mut clock = Instant::now();
        let context = settings.context;
        if !(1..=CONTEXT_LIMIT).contains(&context) {
            return Err("Straydog context must be within the native model limit".into());
        }
        if !(1..=4096).contains(&settings.prefill_batch_size) {
            return Err("Straydog prefill batch must be between 1 and 4096".into());
        }
        if settings.speculative > MAX_DRAFTS {
            return Err(format!("Straydog speculative drafts must be between 0 and {MAX_DRAFTS}"));
        }
        crate::host::commit_note("start");
        let (layer_bytes, tail) = weights.placement_bytes(group.expert_parallel()?);
        group.balance_layers(&layer_bytes, tail)?;
        let drafting = settings.speculative > 0 && weights.drafter;
        validate_geometry(&weights)?;
        if settings.prefill_batch_size > group.logical_capacity() {
            return Err("Straydog prefill exceeds this GPU workspace capacity".into());
        }
        let context64 = context as u64;
        let mut state_bytes = context_state_bytes(context64, drafting);
        let attention_capable = group.batch_capacity() == 256;
        let kv_bytes = attention_layers(drafting) * context64 * 1024 * 4;
        let fits = attention_capable && kv_bytes <= group.memory()?.1 / 4;
        let host_capable = attention_capable && group.host_state_supported()?;
        let recurrent_bytes = 36 * 48 * 128 * 128 * 4;
        // K/V beyond a quarter of the VRAM budget stays in mapped host memory, which the
        // GPU kernels read over the bus.
        let (mut host, mut batch) = (attention_capable && !fits && host_capable, settings.prefill_batch_size);
        if host_capable && state_bytes - recurrent_bytes <= settings.state_budget_bytes {
            // The layers in parts on two GPUs that hold them so (Group::enable_parts), else split by layers.
            let mut placed = None;
            if group.enable_parts()? {
                placed = resident_placement(&weights, &group, context64, settings.prefill_batch_size, drafting, fits)?;
                if placed.is_none() {
                    group.disable_parts()?;
                }
            }
            if placed.is_none() {
                placed = working_split(&weights, &mut group, context64, settings.prefill_batch_size, drafting, fits)?;
            }
            if let Some((_, resident_host, resident_batch)) = placed {
                (host, batch) = (resident_host, resident_batch);
            }
        }
        let gpu_attention = fits || host;
        if !gpu_attention {
            return Err("the GPU budget cannot hold the attention state; raise --vram".into());
        }
        if !host {
            state_bytes -= kv_bytes;
        }
        state_bytes -= recurrent_bytes;
        if state_bytes > settings.state_budget_bytes {
            return Err(format!("Straydog context state needs {state_bytes} bytes, over its RAM budget"));
        }
        group.reserve_model_state(context64, batch as u64, drafting, host)?;
        let checkpoint_bytes = group.reserve_checkpoints(CHECKPOINTS.len() + PARK_SLOTS.len())?;
        let mut load_seconds = vec![("state", lap(&mut clock))];
        crate::host::commit_note("states and checkpoints");
        let mut model = Model {
            group,
            cpu: None,
            weights,
            context,
            prefill_batch_size: batch,
            state_bytes,
            speculative: 0,
            copy_drafts: 0,
            resident: false,
            partial: false,
            host_kv: host,
            history: Vec::new(),
            final_hidden: None,
            checkpoints: IndexMap::new(),
            parked: None,
            park_budget: crate::host::host_memory_bytes() / 8,
            park_whole: false,
            last_metrics: Map::new(),
            logits_digests: None,
            logits_capture: None,
            forced_rows: None,
            deferred: false,
            mtp_kept: 0,
            draft_vocabulary: false,
            draft_extras: HashSet::new(),
            draft_pending: Vec::new(),
            load_seconds: Vec::new(),
            profile: std::env::var_os("DS41_PROFILE").map(|_| Profile::default()),
            pass_seconds: [0.0; 2],
            passes_with_draft: 0,
            pass_counts: [0; 2],
            memory_before: None,
            read_aheads_before: (0, 0),
            kv_capacity: context,
            pinned_now: Default::default(),
            watching_memory: false,
            pinned_other: checkpoint_bytes,
            banks_bytes: 0,
            hybrid_counts: Value::Null,
            ple_counts: (0, 0, 0),
            ple_wait: 0.0,
            images: Vec::new(),
            rope_delta: 0,
            rope_table: false,
        };
        // Placement allocates every weight before any is read; reading then fills them.
        model.group.defer_uploads();
        model.resident = model.prepare_resident_moe()?;
        if !model.resident {
            // (before the expert placement, whose gates read what the indexers hold: at the whole context, a third of the slots)
            if model.group.batch_capacity() == 256 && host && !model.group.parts() && context > STARTING_KV {
                model.kv_capacity = STARTING_KV;
                model.group.start_kv(STARTING_KV as u64)?;
            }
            model.partial = model.group.batch_capacity() == 256 && model.prepare_hybrid_moe(settings.cpu_threads)?;
        }
        if !model.resident && !model.partial {
            return Err("the GPU budget cannot hold the model's dense weights; raise --vram".into());
        }
        crate::host::commit_note("experts placed");
        model.prepare_decoder(context64, false)?;
        crate::host::commit_note("decoder placed");
        // SAFETY of the borrow: the operand points into the weights, which outlive the group.
        let head: Operand<'static> = unsafe { std::mem::transmute(model.projection("output.weight")?) };
        model.group.pin_head(&head)?;
        let keep = model.first_use_tensors()?;
        load_seconds.push(("place", lap(&mut clock)));
        crate::host::commit_note("placed");
        let pending = model.group.take_pending();
        crate::load::load(&mut model.weights, &mut model.group, pending, &keep, progress)?;
        load_seconds.push(("read", lap(&mut clock)));
        crate::host::commit_note("read");
        // Host K/V page-locked on a thread of its own meanwhile (see `Backend::prefetch_host_states`).
        model.group.finish_model_state()?;
        load_seconds.push(("host_state", lap(&mut clock)));
        crate::host::commit_note("host K/V");
        model.prepare_decoder(model.kv_capacity as u64, true)?;
        if model.partial && model.weights.drafter {
            model.group.verify_reserve()?;
        }
        let mut two_drafts = false;
        if model.partial {
            let pinned = checkpoint_bytes + if host { attention_layers(model.weights.drafter) * model.kv_capacity as u64 * 4096 } else { 0 };
            model.lock_host_banks(pinned)?;
            crate::host::commit_note("banks locked");
            let slots = model.group.create_expert_slots()?;
            let experts = 48 * model.weights.tensor("blk.0.ffn_gate_exps.weight")?.dims.last().copied().unwrap_or(0) as u64;
            two_drafts = cfg!(feature = "hip") && slots as f64 >= TWO_DRAFTS_SHARE * experts as f64;
            crate::host::commit_note("expert slots");
        }
        if drafting && model.resident && model.weights.drafter
            && std::env::var("STRAYDOG_DRAFT_VOCABULARY").map_or(true, |value| value != "0")
        {
            model.draft_vocabulary = model.group.draft_vocabulary(&head, DRAFT_ROWS, &[], true)?;
        }
        // The K/V stages take the VRAM left after the decoder, any expert slots and a draft's vocabulary.
        model.group.grow_kv_stages(context64)?;
        let row_cache = match settings.row_cache_bytes {
            Some(bytes) => bytes,
            None => {
                let [_, rows] = model.weights.tensor(PLE_TABLE)?.dims[..] else { return Err("Straydog PLE table is not rows".into()) };
                let width = model.weights.tensor(PLE_TABLE)?.bytes / rows;
                let budget = crate::rows::auto_budget(rows, width, model.weights.host_memory_bytes, model.state_bytes);
                // Partially resident with less memory than the banks want: a token's rows are few, and what
                // the cache would hold serves the banks the CPU reads hundreds of experts from a token.
                let scarce = model.partial && {
                    let banks = (0..48).try_fold(0, |sum, layer| Self::host_banks(&model.weights, layer).map(|banks| sum + banks.iter().map(|bank| bank.len() as u64).sum::<u64>()))?;
                    model.weights.host_memory_bytes < 2 * banks
                };
                if scarce { budget.min(SCARCE_ROW_CACHE) } else { budget }
            }
        };
        if row_cache > 0 {
            model.weights.cache_rows(PLE_TABLE, row_cache)?;
        }
        load_seconds.push(("plans", lap(&mut clock)));
        crate::host::commit_note("plans");
        model.load_seconds = load_seconds;
        // A partially resident model drafts one token a pass: IQ2_XS --vram 12 on an R9700, 64 tok/s without drafts, 70 with one,
        // 64 with two and 65 with three. The widest pass costs far more than the rest (16 ms for one row, 24 for two, 53 for four).
        // Two once the slots hold enough ([`TWO_DRAFTS_SHARE`]).
        let drafts = if model.partial { settings.speculative.min(if two_drafts { 2 } else { 1 }) } else { settings.speculative };
        model.speculative = if drafting && model.weights.drafter { drafts } else { 0 };
        // (a partially resident model's rows cost it far more: as many as it drafts)
        model.copy_drafts = if model.partial || model.speculative == 0 { model.speculative } else { VERIFY_ROWS - 1 };
        Ok(model)
    }

    /// The most rows a pass of this placement takes.
    pub fn pass_rows(&self) -> usize {
        if self.partial { PARTIAL_ROWS } else { VERIFY_ROWS }
    }

    pub fn info(&self) -> Result<Value> {
        self.group.info()
    }

    // ---- operands ----

    fn projection(&self, name: &str) -> Result<Operand<'_>> {
        let tensor = self.weights.tensor(name)?;
        let [cols, rows] = tensor.dims[..] else { return Err(format!("Straydog matrix rank mismatch: {name}")) };
        if crate::formats::block(tensor.kind).is_none() {
            return Err(format!("unsupported Straydog weight format {}: {name}", tensor.kind));
        }
        let (data, kind) = self.weights.raw(name, 0, None)?;
        Ok(Operand::bytes(format!("{name}:None"), data, kind, rows, cols))
    }

    /// A projection uploaded quantized to Q8_0 when it is a HyperConnection's down or up (BF16 in the bundle: half the bytes a
    /// decode pass reads, a quarter of a row's; `STRAYDOG_HC_Q8=0` keeps it BF16) or a float matrix of the drafter's
    /// (crate::weights::drafter_quantized); as it is otherwise.
    fn q8_projection(&self, name: &str) -> Result<Operand<'_>> {
        let operand = self.projection(name)?;
        let float = matches!(operand.kind, 0 | 1 | 30);
        if !(crate::weights::hc_quantized(name) || crate::weights::drafter_quantized(name) && float) || !operand.cols.is_multiple_of(32) {
            return Ok(operand);
        }
        let Data::Bytes(bytes) = operand.data else { return Ok(operand) };
        Ok(Operand { key: format!("{name}:q8"), data: Data::Quantize(bytes, operand.kind), kind: 8, rows: operand.rows, cols: operand.cols })
    }

    /// A vector, uploaded decoded to F32.
    fn vector(&self, name: &str, key: Option<&str>) -> Result<Operand<'_>> {
        let key = key.map_or_else(|| format!("@vector:{name}"), str::to_string);
        let (rows, cols) = self.weights.vector_shape(name)?;
        let (data, kind) = self.weights.raw(name, 0, None)?;
        Ok(Operand { key, data: Data::Decode(data, kind), kind: 0, rows: rows as u64, cols: cols as u64 })
    }

    /// A MoE's shared-expert mix as a one-row matrix of its own bytes (BF16): a decode row's router and mix are then one launch of
    /// one kind, where an F32 mix took a launch of its own.
    fn mix_row(&self, name: &str) -> Result<Operand<'_>> {
        let (rows, cols) = self.weights.vector_shape(name)?;
        let (data, kind) = self.weights.raw(name, 0, None)?;
        if rows != 1 || kind != 30 {
            return self.vector(name, Some(&format!("{name}:vector")));
        }
        Ok(Operand::bytes(format!("{name}:row"), data, kind, 1, cols as u64))
    }

    /// A HyperConnection's norm (keyed by its prefix) and its down, up and inject projections.
    fn hc_operands(&self, prefix: &str) -> Result<Vec<Operand<'_>>> {
        let mut operands = vec![self.vector(&format!("{prefix}_norm.weight"), Some(&format!("@norm:{prefix}")))?];
        for suffix in ["_down.weight", "_up.weight"] {
            operands.push(self.q8_projection(&format!("{prefix}{suffix}"))?);
        }
        operands.push(self.projection(&format!("{prefix}_inject.weight"))?);
        Ok(operands)
    }

    /// Rows of a matrix (or a vector's values) as runs: (first, count) of `name`'s rows (values), placed one after the other.
    fn part_rows(&self, name: &str, part: usize, runs: &[(u64, u64)], rows: u64, cols: u64) -> Result<Operand<'_>> {
        let tensor = self.weights.tensor(name)?;
        let (bytes, kind) = self.weights.raw(name, 0, None)?;
        let row_bytes = crate::formats::matrix_bytes(kind, 1, cols as usize).ok_or("unsupported Straydog weight format")? as u64;
        let mut pieces = Vec::with_capacity(runs.len());
        let mut at = 0;
        for &(first, count) in runs {
            pieces.push(Piece { source: &bytes[(first * row_bytes) as usize..((first + count) * row_bytes) as usize], at, columns: None });
            at += count * row_bytes;
        }
        if at != rows * row_bytes || tensor.bytes % row_bytes != 0 {
            return Err(format!("Straydog part rows do not fit {name}"));
        }
        Ok(Operand { key: format!("{name}:part{part}"), data: Data::Gather(pieces), kind, rows, cols })
    }

    /// Runs (first, count) of an F32 vector's values, as one row.
    fn part_values(&self, name: &str, part: usize, runs: &[(u64, u64)]) -> Result<Operand<'_>> {
        let (bytes, kind) = self.weights.raw(name, 0, None)?;
        if kind != 0 {
            return Err(format!("Straydog part values of {name} are not F32"));
        }
        let mut pieces = Vec::with_capacity(runs.len());
        let mut at = 0;
        for &(first, count) in runs {
            pieces.push(Piece { source: &bytes[(first * 4) as usize..((first + count) * 4) as usize], at, columns: None });
            at += count * 4;
        }
        Ok(Operand { key: format!("{name}:part{part}"), data: Data::Gather(pieces), kind, rows: 1, cols: at / 4 })
    }

    /// Column runs (first, count) of every row of matrix `name` (`rows` x `cols`), placed one after the other in each row.
    fn part_columns(&self, name: &str, part: usize, runs: &[(u64, u64)], rows: u64, cols: u64) -> Result<Operand<'_>> {
        let (bytes, kind) = self.weights.raw(name, 0, None)?;
        let span = |count: u64| crate::formats::matrix_bytes(kind, 1, count as usize).map(|bytes| bytes as u64);
        let (pitch, width) = (span(cols).ok_or("unsupported Straydog weight format")?,
                              span(runs.iter().map(|run| run.1).sum()).ok_or("Straydog part columns do not fill blocks")?);
        let mut pieces = Vec::with_capacity(runs.len());
        let mut at = 0;
        for &(first, count) in runs {
            let (Some(first_bytes), Some(run)) = (span(first), span(count)) else { return Err("Straydog part columns do not fill blocks".into()) };
            pieces.push(Piece { source: bytes, at, columns: Some(Columns { pitch, first: first_bytes, width: run, to_pitch: width }) });
            at += run;
        }
        if bytes.len() as u64 != rows * pitch || at != width {
            return Err(format!("Straydog part columns do not fit {name}"));
        }
        let columns = runs.iter().map(|run| run.1).sum();
        Ok(Operand { key: format!("{name}:part{part}"), data: Data::Gather(pieces), kind, rows, cols: columns })
    }

    /// A layer's operands as decoder registration takes them for device `part`'s half of the heads (gpu/qwen_parts.hpp): a
    /// DeltaNet's key heads 8p..8p+7 with the value heads that read them (16c + 8p.., c = 0, 1, 2), an attention layer's query
    /// heads 12p..12p+11 with K/V head p; their rows of the projections and columns of the output projection.
    fn part_operands(&self, layer: usize, part: usize) -> Result<Vec<Operand<'_>>> {
        let prefix = format!("blk.{layer}.");
        let name = |suffix: &str| format!("{prefix}{suffix}");
        let p = part as u64;
        let mut operands = self.hc_operands(&name("hc_attn"))?;
        // (the MTP layer's float projections go up as Q8_0, as whole: crate::weights::drafter_quantized)
        fn q8(operand: Operand<'_>) -> Operand<'_> {
            let float = matches!(operand.kind, 0 | 1 | 30);
            match operand.data {
                Data::Gather(pieces) if float && crate::weights::drafter_quantized(operand.key.split(':').next().unwrap_or(""))
                    && operand.cols.is_multiple_of(32) =>
                    Operand { key: format!("{}:q8", operand.key), data: Data::QuantizeGather(pieces, operand.kind), kind: 8,
                              rows: operand.rows, cols: operand.cols },
                data => Operand { data, ..operand },
            }
        }
        if layer % 4 != 3 && layer < 48 {
            let values = |width: u64| -> Vec<(u64, u64)> { (0..3).map(|c| ((16 * c + 8 * p) * width, 8 * width)).collect() };
            let mut qkv = vec![(1024 * p, 1024), (2048 + 1024 * p, 1024)];
            qkv.extend(values(128).into_iter().map(|(first, count)| (4096 + first, count)));
            operands.push(self.part_rows(&name("attn_qkv.weight"), part, &qkv, 5120, 2560)?);
            operands.push(self.part_rows(&name("attn_gate.weight"), part, &values(128), 3072, 2560)?);
            operands.push(self.part_rows(&name("ssm_beta.weight"), part, &values(1), 24, 2560)?);
            operands.push(self.part_rows(&name("ssm_alpha.weight"), part, &values(1), 24, 2560)?);
            operands.push(self.part_columns(&name("ssm_out.weight"), part, &values(128), 2560, 6144)?);
            operands.push(self.part_rows(&name("ssm_conv1d.weight"), part, &qkv, 5120, 4)?);
            operands.push(self.part_values(&name("ssm_dt.bias"), part, &values(1))?);
            operands.push(self.part_values(&name("ssm_a"), part, &values(1))?);
            operands.push(self.vector(&name("ssm_norm.weight"), None)?);
        } else {
            operands.push(q8(self.part_rows(&name("attn_q.weight"), part, &[(6144 * p, 6144)], 6144, 2560)?));
            operands.push(q8(self.part_rows(&name("attn_k.weight"), part, &[(256 * p, 256)], 256, 2560)?));
            operands.push(q8(self.part_rows(&name("attn_v.weight"), part, &[(256 * p, 256)], 256, 2560)?));
            operands.push(self.q8_projection(&name("indexer.q_proj.weight"))?);
            operands.push(self.q8_projection(&name("indexer.k_proj.weight"))?);
            operands.push(q8(self.part_columns(&name("attn_output.weight"), part, &[(3072 * p, 3072)], 2560, 6144)?));
            for norm in ["attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"] {
                operands.push(self.vector(&name(norm), None)?);
            }
        }
        operands.extend(self.hc_operands(&name("hc_ffn"))?);
        Ok(operands)
    }

    fn ple_operands(&self) -> Result<Vec<Operand<'_>>> {
        let mut operands = vec![self.q8_projection("blk.1.ple_key.weight")?, self.q8_projection("blk.1.ple_value.weight")?];
        for name in ["ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight"] {
            operands.push(self.vector(&format!("blk.1.{name}"), None)?);
        }
        Ok(operands)
    }

    fn head_operands(&self) -> Result<Vec<Operand<'_>>> {
        Ok(vec![self.vector("output_hc_norm.weight", None)?, self.q8_projection("output_hc_down.weight")?,
                self.q8_projection("output_hc_up.weight")?, self.projection("output.weight")?])
    }

    /// The drafter's mixer before the shared vocabulary head.
    fn mtp_head_operands(&self) -> Result<Vec<Operand<'_>>> {
        Ok(vec![self.vector("mtp.hc_out_norm.weight", None)?, self.q8_projection("mtp.hc_out_down.weight")?,
                self.q8_projection("mtp.hc_out_up.weight")?, self.projection("output.weight")?])
    }

    /// The tensors behind operands the passes upload at first use (the PLE, the head's mixers,
    /// the drafter's projections) and the token embeddings: loading keeps host copies of them.
    fn first_use_tensors(&self) -> Result<Vec<String>> {
        let mut operands = self.ple_operands()?;
        operands.extend(self.head_operands()?);
        if self.weights.drafter {
            operands.extend(self.mtp_operands()?);
            operands.extend(self.mtp_head_operands()?);
        }
        let mut names = vec!["token_embd.weight".to_string()];
        for operand in &operands {
            if let Data::Bytes(bytes) | Data::Decode(bytes, _) | Data::Quantize(bytes, _) = operand.data {
                let (file, offset) =
                    self.weights.locate(bytes.as_ptr(), bytes.len()).ok_or("Straydog operand is not mapped")?;
                names.push(self.weights.tensor_at(file, offset)?.to_string());
            }
        }
        Ok(names)
    }

    fn resident_requirements(&self, drafter: bool) -> Vec<(String, u64)> {
        self.weights.resident_requirements(drafter && self.weights.drafter)
    }

    // ---- placement ----

    /// Full residency: pin every layer's banks, router and shared expert, once the whole
    /// weight footprint fits every GPU.
    fn prepare_resident_moe(&mut self) -> Result<bool> {
        if !self.group.can_pin_model(&self.resident_requirements(false), None)? {
            return Ok(false);
        }
        if self.weights.drafter && !self.group.can_pin_model(&self.resident_requirements(true), None)? {
            self.weights.drop_drafter();
        }
        for layer in self.weights.layers() {
            self.register_resident_layer(layer)?;
        }
        Ok(true)
    }

    /// A layer's banks, router and shared expert pinned.
    fn register_resident_layer(&mut self, layer: usize) -> Result<()> {
        let prefix = format!("blk.{layer}.");
        let mut operands = Vec::new();
        for role in ["gate", "up", "down"] {
            let name = format!("{prefix}ffn_{role}_exps.weight");
            let tensor = self.weights.tensor(&name)?;
            let (cols, rows, experts) = (tensor.dims[0], tensor.dims[1], tensor.dims[2]);
            let (data, kind) = self.weights.raw(&name, 0, None)?;
            operands.push(Operand::bytes(format!("{name}:bank"), data, kind, rows * experts, cols));
        }
        for role in ["gate", "up", "down"] {
            operands.push(self.q8_projection(&format!("{prefix}ffn_{role}_shexp.weight"))?);
        }
        operands.push(self.projection(&format!("{prefix}ffn_gate_inp.weight"))?);
        operands.push(self.mix_row(&format!("{prefix}ffn_gate_inp_shexp.weight"))?);
        // SAFETY of the borrow: operands point into the weights, which outlive the group.
        let operands: Vec<Operand<'static>> = unsafe { std::mem::transmute(operands) };
        self.group.register_resident_moe(&prefix, &operands, 512)
    }

    /// The MTP layer as one more partially resident layer: its most used experts in the slots, by the same scores as every layer's,
    /// the rest on the CPU. Which ones a draft routes to follows the text (the 256 most used of one set of prompts cover 78% of
    /// another's routes), so no fixed list would do, and the whole bank cost the other layers 16% more CPU routes.
    fn register_hybrid_drafter(&mut self) -> Result<()> {
        let prefix = "blk.48.";
        let mut operands = Vec::new();
        for role in ["gate", "up", "down"] {
            operands.push(self.q8_projection(&format!("{prefix}ffn_{role}_shexp.weight"))?);
        }
        operands.push(self.projection(&format!("{prefix}ffn_gate_inp.weight"))?);
        operands.push(self.mix_row(&format!("{prefix}ffn_gate_inp_shexp.weight"))?);
        let banks = Self::host_banks(&self.weights, 48)?;
        let kinds = self.weights.expert_kinds(48)?;
        // SAFETY of the borrows: banks and operands point into the weights, which outlive the group.
        let (operands, banks): (Vec<Operand<'static>>, [&'static [u8]; 3]) = unsafe {
            (std::mem::transmute::<Vec<Operand<'_>>, Vec<Operand<'static>>>(operands),
             std::mem::transmute::<[&[u8]; 3], [&'static [u8]; 3]>(banks))
        };
        self.group.register_hybrid_moe(prefix, &operands, banks, 512, kinds)
    }

    /// Partial residency: pin every dense operand and bind the host expert banks for the CPU
    /// engine and prefill streaming (page-locked once read, see [`Model::lock_host_banks`]).
    fn prepare_hybrid_moe(&mut self, cpu_threads: u32) -> Result<bool> {
        if !self.group.hybrid_supported()? {
            return Ok(false);
        }
        let kinds = (0..48).map(|layer| self.weights.expert_kinds(layer)).collect::<Result<Vec<_>>>()?;
        let staging = STAGING_SLOTS * kinds.iter().map(|kinds| expert_bytes(*kinds)).max().unwrap() + STAGING_GATE;
        let mut requirements = self.weights.dense_requirements();
        requirements.push(("@hybrid:staging".into(), staging));
        // The drafter when its layer and a verification's buffers leave the expert slots their share.
        let mut with_drafter = requirements.clone();
        // (its routed experts take expert slots like any layer's: register_hybrid_drafter)
        with_drafter.extend(self.weights.drafter_requirements().into_iter().filter(|(name, _)| !name.ends_with("_exps.weight")));
        with_drafter.push(("@verify:buffers".into(), VERIFY_BYTES));
        with_drafter.push(("@hybrid:slots".into(), DRAFTER_SLOT_ROOM));
        if !self.weights.drafter || !self.group.can_pin_model(&with_drafter, None)? {
            self.weights.drop_drafter();
            self.group.drop_mtp_layer_state()?;
        }
        if !self.group.can_pin_model(&requirements, None)? {
            return Ok(false);
        }
        let cpu = CpuExperts::new(cpu_threads)?;
        // (the same files as the mapping's, checked as it is: `Weights::check_unchanged` before every generation)
        #[cfg(windows)]
        for (map, file) in self.weights.mapped_files() {
            cpu.gather_file(map, file)?;
        }
        for (layer, kinds) in kinds.iter().enumerate() {
            let prefix = format!("blk.{layer}.");
            let banks = Self::host_banks(&self.weights, layer)?;
            let mut operands = Vec::new();
            for role in ["gate", "up", "down"] {
                operands.push(self.projection(&format!("{prefix}ffn_{role}_shexp.weight"))?);
            }
            operands.push(self.projection(&format!("{prefix}ffn_gate_inp.weight"))?);
            let name = format!("{prefix}ffn_gate_inp_shexp.weight");
            operands.push(self.vector(&name, Some(&format!("{name}:vector")))?);
            // SAFETY of the borrows: banks and operands point into the weights, which outlive the group.
            let (operands, banks): (Vec<Operand<'static>>, [&'static [u8]; 3]) = unsafe {
                (std::mem::transmute::<Vec<Operand<'_>>, Vec<Operand<'static>>>(operands),
                 std::mem::transmute::<[&[u8]; 3], [&'static [u8]; 3]>(banks))
            };
            self.group.register_hybrid_moe(&prefix, &operands, banks, 512, *kinds)?;
        }
        if self.weights.drafter {
            self.register_hybrid_drafter()?;
        }
        // The head and its mixing are not decoder-layer operands; keep them resident too.
        let head: Vec<Operand<'static>> = unsafe {
            std::mem::transmute(vec![self.projection("output.weight")?, self.q8_projection("output_hc_down.weight")?,
                                     self.q8_projection("output_hc_up.weight")?])
        };
        self.group.pin(&head)?;
        self.cpu = Some(cpu);
        Ok(true)
    }

    /// A layer's mapped gate, up and down expert banks.
    fn host_banks(weights: &Weights, layer: usize) -> Result<[&[u8]; 3]> {
        let bank = |role: &str| weights.raw(&format!("blk.{layer}.ffn_{role}_exps.weight"), 0, None).map(|raw| raw.0);
        Ok([bank("gate")?, bank("up")?, bank("down")?])
    }

    /// Partial residency, once the banks are authenticated: page-lock whole layers within the
    /// lock budget and hand absent experts to the CPU engine.
    fn lock_host_banks(&mut self, pinned_state: u64) -> Result<()> {
        let layer_banks = (0..48).map(|layer| Self::host_banks(&self.weights, layer)).collect::<Result<Vec<_>>>()?;
        let banks_total = layer_banks.iter().flatten().map(|bank| bank.len() as u64).sum();
        let (vram_used, vram_budget) = self.group.memory()?;
        let vram_room = vram_budget.saturating_sub(vram_used);
        let (by_memory, by_free) = (lock_budget(self.weights.host_memory_bytes, banks_total, pinned_state),
                                    crate::host::lockable(self.weights.host_memory_bytes, vram_room));
        let budget = by_memory.min(by_free);
        // Whole layers from the last: reading went through those most recently, so they are the
        // likeliest to be cached; the rest of the locked range is read ahead meanwhile.
        let (mut layers, mut locked) = (Vec::new(), 0u64);
        for layer in (0..48).rev() {
            let size: u64 = layer_banks[layer].iter().map(|bank| bank.len() as u64).sum();
            if locked + size > budget {
                break;
            }
            layers.push(layer);
            locked += size;
        }
        let mut registered = 0;
        // The layer whose banks the runtime would not page-lock, and why.
        let mut refused: Option<(usize, String)> = None;
        if !layers.is_empty() {
            let weights = &self.weights;
            let ranges: Vec<(String, (u64, u64))> = layers
                .iter()
                .flat_map(|layer| ["gate", "up", "down"].map(|role| format!("blk.{layer}.ffn_{role}_exps.weight")))
                .map(|name| {
                    let tensor = weights.tensor(&name).unwrap();
                    (name, (tensor.offset, tensor.bytes))
                })
                .collect();
            let group = &mut self.group;
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    for (name, range) in &ranges {
                        if let Ok(file) = weights.file_of(name) {
                            crate::digest::will_need(file, std::slice::from_ref(range));
                        }
                    }
                });
                for layer in &layers {
                    if let Some(error) = layer_banks[*layer].iter().find_map(|bank| group.register_host(bank).err()) {
                        refused = Some((*layer, error));
                        break;
                    }
                    registered += 1;
                }
            });
            // Locked, the banks' pages stay as they are now: with the shards' verity trees, check
            // them (they may have been read again since loading verified their chunks).
            for layer in &layers[..registered] {
                for role in ["gate", "up", "down"] {
                    weights.verify_mapped(&format!("blk.{layer}.ffn_{role}_exps.weight"))?;
                }
            }
        }
        // What the start did with the banks and why it did no more, for a log that says where the CPU's experts are read from.
        let gib = |bytes: u64| if bytes == u64::MAX { "no bound".to_string() } else { format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64) };
        let locked = layers[..registered].iter().map(|layer| layer_banks[*layer].iter().map(|bank| bank.len() as u64).sum::<u64>()).sum::<u64>();
        let free = crate::host::left_free(self.weights.host_memory_bytes);
        // (on Windows the commit, which locked memory and VRAM are charged to 1 to 1, can bind before the physical memory does)
        let commit = crate::host::commit_left();
        let why = match &refused {
            Some((layer, error)) => format!("registering layer {layer}'s banks failed ({error})"),
            None if registered == 48 => "none left".to_string(),
            None if by_memory <= by_free => format!("the lock budget of {} was used", gib(by_memory)),
            None if commit.is_some_and(|(available, commit)| commit < available) => {
                format!("the commit left allows {} (host::lockable: less the VRAM slots to come and {} kept free)", gib(by_free), gib(free))
            }
            None => format!("the available memory allows {} (host::lockable: less {} kept free)", gib(by_free), gib(free)),
        };
        let (available, commit) = match commit {
            Some((available, commit)) => (gib(available), format!(", commit left {}", gib(commit))),
            None => (crate::host::memory_note().map_or("unknown".to_string(), |(_, available, _)| gib(available)), String::new()),
        };
        eprintln!("Straydog host banks: {registered} of 48 layers page-locked, {} of {}; stopped because {why}; lock_budget {}, lockable {}, \
                   host memory {}, available {available}{commit}, VRAM slots to come {}", gib(locked), gib(banks_total), gib(by_memory),
                  gib(by_free), gib(self.weights.host_memory_bytes), gib(vram_room));
        self.banks_bytes = banks_total;
        crate::host::keep_resident(self.weights.host_memory_bytes, pinned_state, banks_total);
        self.pinned_now.store(pinned_state, std::sync::atomic::Ordering::Relaxed);
        if !self.watching_memory {
            self.watching_memory = true;
            crate::host::watch_working_set(self.weights.host_memory_bytes, self.pinned_now.clone(), banks_total);
        }
        if registered > 0 {
            // The GPU reads a few absent experts per layer over PCIe while the CPU computes the rest,
            // up to 5 of every 12 (expert_source). On an R9700 at 6 GiB beside a Ryzen 5 7500F: 1 route
            // a layer 30.7 tokens/s, 2 33.7, 3 35.7, 4 36.7; a higher share than 5 in 12 holds the
            // layers up (7 in 12: 33.7). CUDA takes two, staged (sd_hybrid_assist).
            self.group.enable_assist(4)?;
        }
        // Speed first (STRAYDOG_EXPERT_FLOOR, a router weight from 0 to 1): the CPU's routes below it are left out.
        if let Some(floor) = expert_floor()? {
            self.group.route_floor(floor)?;
            eprintln!("Straydog speed first: the CPU's routes below a router weight of {floor} are left out");
        }
        let cpu = self.cpu.as_ref().ok_or("Straydog partial residency has no CPU engine")?;
        cpu.watch_paging(&crate::host::PAGING)?;
        self.group.attach_cpu(cpu)
    }

    /// Pin each complete decoder layer's operands after the residency preflight; with
    /// `register`, once the sequence state exists, also create the layers' native plans.
    fn prepare_decoder(&mut self, context: u64, register: bool) -> Result<()> {
        for layer in self.weights.layers() {
            if self.group.parts() && (layer < 48 || self.group.mtp_parts()) {
                for part in 0..2 {
                    // SAFETY of the borrow: operands point into the weights, which outlive the group.
                    let operands: Vec<Operand<'static>> = unsafe { std::mem::transmute(self.part_operands(layer, part)?) };
                    if register {
                        self.group.register_decoder_part(layer, part, &operands, context)?;
                    } else {
                        self.group.pin_part(part, &operands)?;
                    }
                }
                continue;
            }
            let prefix = format!("blk.{layer}.");
            let mut operands = self.hc_operands(&format!("{prefix}hc_attn"))?;
            let (names, norms): (&[&str], &[&str]) = if layer % 4 != 3 && layer < 48 {
                (&["attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_out.weight"],
                 &["ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight"])
            } else {
                (&["attn_q.weight", "attn_k.weight", "attn_v.weight", "indexer.q_proj.weight", "indexer.k_proj.weight",
                   "attn_output.weight"],
                 &["attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"])
            };
            for name in names {
                operands.push(self.q8_projection(&format!("{prefix}{name}"))?);
            }
            for name in norms {
                operands.push(self.vector(&format!("{prefix}{name}"), None)?);
            }
            operands.extend(self.hc_operands(&format!("{prefix}hc_ffn"))?);
            // SAFETY of the borrow: operands point into the weights, which outlive the group.
            let operands: Vec<Operand<'static>> = unsafe { std::mem::transmute(operands) };
            if register {
                self.group.register_decoder_layer(layer, &operands, context)?;
            } else {
                self.group.pin_layer(layer, &operands)?;
            }
        }
        Ok(())
    }

    // ---- state ----

    pub fn reset(&mut self) -> Result<()> {
        self.history.clear();
        self.final_hidden = None;
        self.group.reset_state()
    }

    fn save_checkpoint(&mut self, kind: &'static str) -> Result<()> {
        if kind == "turn" {
            if let Some(turn) = self.checkpoints.shift_remove("turn") {
                if let Some(prior) = self.checkpoints.shift_remove("prior") {
                    self.release_checkpoint(prior)?;
                }
                self.checkpoints.insert("prior", turn);
            }
        }
        let used: Vec<usize> =
            self.checkpoints.iter().filter(|(name, _)| **name != kind).map(|(_, checkpoint)| checkpoint.slot).collect();
        let slot = (0..CHECKPOINTS.len()).find(|slot| !used.contains(slot)).unwrap();
        self.group.checkpoint(slot, true)?;
        // (inside a pipelined prompt, the row its last batch kept: a prompt resuming here pairs
        // its first token with it)
        let final_hidden = match self.final_hidden.clone() {
            None => self.kept_hidden()?,
            hidden => hidden,
        };
        // (the MTP layer's alone: the target's layers resume at a block boundary and read none)
        let history = if self.speculative > 0 { Some(self.group.save_history(self.history.len() as u64, Some("blk.48."))?) } else { None };
        if let Some(replaced) = self.checkpoints.insert(kind, Checkpoint { slot, tokens: self.history.clone(), final_hidden, history }) {
            self.release_checkpoint(replaced)?;
        }
        Ok(())
    }

    /// Free what a checkpoint holds besides its slot.
    fn release_checkpoint(&mut self, checkpoint: Checkpoint) -> Result<()> {
        checkpoint.history.map_or(Ok(()), |history| self.group.release_positions(history))
    }

    /// Inside a pipelined prompt with the MTP layer: its last batch's last row, which only the
    /// MTP layer's device kept (None outside one or before its first batch).
    fn kept_hidden(&mut self) -> Result<Option<Vec<f32>>> {
        if self.speculative == 0 || !self.deferred || self.mtp_kept == 0 {
            return Ok(None);
        }
        self.group.mtp_kept_row(self.mtp_kept).map(Some)
    }

    /// Park the live sequence whole: its positions from `from` (with the indexers' history) and
    /// its recurrent state (in `slot`). For the live sequence a restored one replaces, whose
    /// positions the restore overwrites at once.
    fn park(&mut self, from: usize, slot: usize) -> Result<Parked> {
        let positions = self.group.save_positions(from as u64, self.history.len() as u64, true)?;
        if let Err(error) = self.group.checkpoint(slot, true) {
            let _ = self.group.release_positions(positions);
            return Err(error);
        }
        Ok(Parked { tokens: self.history.clone(), final_hidden: self.final_hidden.clone(), slot, history: None,
                    saved: vec![positions], covered: Some((from, self.history.len())) })
    }

    /// Park the live sequence as a prompt departs from it: its recurrent state (in `slot`) and its
    /// indexers' history now, while they are still its own; its positions only as the prompt
    /// (and later ones) would overwrite them ([`Model::protect`]).
    fn depart(&mut self, slot: usize) -> Result<Parked> {
        let history = self.group.save_history(self.history.len() as u64, None)?;
        if let Err(error) = self.group.checkpoint(slot, true) {
            let _ = self.group.release_positions(history);
            return Err(error);
        }
        Ok(Parked { tokens: self.history.clone(), final_hidden: self.final_hidden.clone(), slot, history: Some(history),
                    saved: Vec::new(), covered: None })
    }

    /// Free what a parked sequence holds (once nothing restores it).
    fn discard(&mut self, parked: Parked) -> Result<()> {
        let mut result = Ok(());
        for positions in parked.saved.into_iter().chain(parked.history) {
            let released = self.group.release_positions(positions);
            if result.is_ok() {
                result = released;
            }
        }
        result
    }

    /// Put `parked` back as the live sequence's state: its copies lowest first (a pooled block two
    /// of them hold ends as the higher one has it; above the copies its positions are where it
    /// left them), then its indexers' history, its recurrent state.
    fn restore(&mut self, parked: &mut Parked) -> Result<()> {
        parked.saved.sort_by_key(|positions| positions.from);
        for positions in parked.saved.iter_mut().chain(parked.history.as_mut()) {
            self.group.restore_positions(positions)?;
        }
        self.group.checkpoint(parked.slot, false)
    }

    /// Before positions `from..to` are written: the parked sequence's state there copied out,
    /// what of it its coverage lacks (extended to multiples of four, but at its end), so the
    /// coverage stays one run. Every writer of positional state asks first: a prompt's whole
    /// range when it starts (from where it resumes, the MTP layer's pair before that, to its
    /// last possible output), then each prefill, decode, verification and draft its own.
    fn protect(&mut self, from: usize, to: usize) -> Result<()> {
        let Some(parked) = &self.parked else { return Ok(()) };
        let end = parked.tokens.len();
        let (from, to) = (from / 4 * 4, to.min(end));
        if from >= to {
            return Ok(());
        }
        let to = to.next_multiple_of(4).min(end);
        let pieces = match parked.covered {
            None => vec![(from, to)],
            Some((low, high)) => [(from, low), (high, to)].into_iter().filter(|(a, b)| a < b).collect(),
        };
        if pieces.is_empty() {
            return Ok(());
        }
        let grown = parked.covered.map_or((from, to), |(low, high)| (low.min(from), high.max(to)));
        if (grown.1 - grown.0) as u64 * self.group.position_bytes() > self.park_budget {
            // Its copies would outgrow the budget: the parked sequence is dropped, overwritten.
            let parked = self.parked.take().unwrap();
            return self.discard(parked);
        }
        let started = Instant::now();
        let mut saved = Vec::new();
        for (low, high) in &pieces {
            match self.group.save_positions(*low as u64, *high as u64, false) {
                Ok(positions) => saved.push(positions),
                Err(error) => {
                    // (the coverage stays as it was: nothing writes past it)
                    for positions in saved {
                        let _ = self.group.release_positions(positions);
                    }
                    return Err(error);
                }
            }
        }
        let parked = self.parked.as_mut().unwrap();
        parked.covered = Some(grown);
        parked.saved.extend(saved);
        let copied: usize = pieces.iter().map(|(low, high)| high - low).sum();
        self.metric_count("park_copied_positions", copied as u64);
        self.metric_add("park_seconds", started.elapsed().as_secs_f64());
        Ok(())
    }

    /// Whether parking the live sequence from `from` fits the host budget.
    fn parkable(&self, from: usize) -> bool {
        let count = self.history.len().saturating_sub(from);
        count >= PARK_MINIMUM && count as u64 * self.group.position_bytes() <= self.park_budget
    }

    /// Before `tokens` overwrite the live sequence. A parked sequence the prompt resumes further
    /// than the live one (it continues the parked sequence, or one of its checkpoints, past what
    /// it shares with the live one) becomes the live sequence again, the live one parked in its
    /// place when long. Otherwise a long live sequence the prompt departs from is parked; or the
    /// parked one takes the positions the prompt will overwrite below its own.
    fn park_or_restore(&mut self, tokens: &[u32]) -> Result<Option<&'static str>> {
        let limit = tokens.len() - 1;
        let live = shared(&self.history, &tokens[..limit]);
        if let Some(parked) = &self.parked {
            let back = shared(&parked.tokens, &tokens[..limit]);
            let resumable = back == parked.tokens.len()
                || self.checkpoints.values().any(|checkpoint| {
                    let position = checkpoint.tokens.len();
                    position > live && position <= back && checkpoint.tokens[..] == tokens[..position]
                });
            // Below its lowest saved position the live sequence must hold the parked tokens.
            let common = shared(&self.history, &parked.tokens);
            if back > live && resumable && common >= parked.from() {
                let mut parked = self.parked.take().unwrap();
                let slot = if parked.slot == PARK_SLOTS[0] { PARK_SLOTS[1] } else { PARK_SLOTS[0] };
                // The live sequence whole over what the restore overwrites: from the lowest copy too
                // (the same tokens below `common`, but perhaps not the same bits once rewritten), or
                // not at all when that does not fit the budget.
                let from = common.min(parked.from());
                let swapped = if self.parkable(common) && self.parkable(from) {
                    match self.park(from, slot) {
                        Ok(swapped) => Some(swapped),
                        Err(error) => {
                            // (nothing overwritten yet: the parked sequence stays parked)
                            self.parked = Some(parked);
                            return Err(error);
                        }
                    }
                } else {
                    None
                };
                if let Err(error) = self.restore(&mut parked) {
                    // Neither sequence whole any more: both copies freed, the live one reset.
                    let _ = self.discard(parked);
                    if let Some(swapped) = swapped {
                        let _ = self.discard(swapped);
                    }
                    let _ = self.reset();
                    return Err(error);
                }
                let (tokens, final_hidden) = (std::mem::take(&mut parked.tokens), parked.final_hidden.take());
                self.draft_note(&tokens);
                self.history = tokens;
                self.final_hidden = final_hidden;
                self.parked = swapped;
                self.discard(parked)?;
                return Ok(Some("restored"));
            }
        }
        if live < self.history.len() && self.parkable(live) {
            let slot = match self.parked.take() {
                Some(parked) => {
                    let slot = parked.slot;
                    self.discard(parked)?;
                    slot
                }
                None => PARK_SLOTS[0],
            };
            self.parked = Some(if self.park_whole { self.park(0, slot)? } else { self.depart(slot)? });
            return Ok(Some("saved"));
        }
        // (a parked sequence keeps the positions this prompt overwrites below its copies: the
        // request protects its range once it knows where it resumes)
        Ok(None)
    }

    /// Prompt tokens already held, after restoring the state they need; at least one token is
    /// left to run. K/V below the restored position comes from the live sequence, which
    /// matches the prompt there; the checkpoint restores the recurrent state.
    fn resume(&mut self, tokens: &[u32]) -> Result<(usize, &'static str)> {
        let limit = tokens.len() - 1;
        let held = &self.history;
        if held.len() <= limit && held[..] == tokens[..held.len()] {
            return Ok((held.len(), "live"));
        }
        let span = held.len().min(limit);
        let common = (0..span).find(|index| held[*index] != tokens[*index]).unwrap_or(span);
        let mut best: Option<&'static str> = None;
        for (kind, checkpoint) in &self.checkpoints {
            let position = checkpoint.tokens.len();
            if position <= common
                && best.is_none_or(|best| position > self.checkpoints[best].tokens.len())
                && checkpoint.tokens[..] == tokens[..position]
            {
                best = Some(kind);
            }
        }
        let Some(kind) = best else {
            self.reset()?;
            return Ok((0, "none"));
        };
        let checkpoint = self.checkpoints.get_mut(kind).unwrap();
        let (slot, saved, hidden) = (checkpoint.slot, checkpoint.tokens.clone(), checkpoint.final_hidden.clone());
        self.group.checkpoint(slot, false)?;
        if let Some(history) = checkpoint.history.as_mut() {
            self.group.restore_positions(history)?;
        }
        self.final_hidden = hidden;
        self.draft_note(&saved);
        self.history = saved;
        Ok((self.history.len(), kind))
    }

    /// Positions (multiples of four) this prefill checkpoints at, with their kinds. A request
    /// no later one continues (`keep` false) saves only the system-prompt root.
    fn checkpoint_plan(&self, tokens: &[u32], reused: usize, turn: bool, keep: bool) -> Vec<(usize, Vec<&'static str>)> {
        let mut plan: Vec<(usize, Vec<&'static str>)> = Vec::new();
        let mut add = |position: usize, kind: &'static str| match plan.iter_mut().find(|(at, _)| *at == position) {
            Some((_, kinds)) => kinds.push(kind),
            None => plan.push((position, vec![kind])),
        };
        let root_held = self.checkpoints.get("root").is_some_and(|root| {
            root.tokens.len() <= tokens.len() && root.tokens[..] == tokens[..root.tokens.len()]
        });
        if !root_held {
            if let Some(end) = tokens.iter().position(|token| *token == IM_END) {
                let root = (end + 1) / 4 * 4;
                if root >= ROOT_MINIMUM && reused < root && root < tokens.len() {
                    add(root, "root");
                }
            }
        }
        if keep {
            // The reply starts at the prompt's last <|im_start|> (the generation prompt).
            let lowest = tokens.len().saturating_sub(63);
            if let Some(index) = (lowest..tokens.len()).rev().find(|index| tokens[*index] == IM_START) {
                let start = index / 4 * 4;
                if reused < start {
                    add(start, if turn { "turn" } else { "step" });
                }
            }
        }
        plan.sort_by_key(|(position, _)| *position);
        plan
    }

    // ---- forward ----

    /// The PLE table rows of `tokens`, which follow `before` (its last two tokens count). An image's positions hash as
    /// `<|image_pad|>`, as the reference's PLE reads the ids before their rows are put in.
    fn ple_indices(&self, before: &[u32], tokens: &[u32]) -> Result<Vec<i64>> {
        let id = |token: &u32| if *token >= IMAGE_TOKEN { i64::from(IMAGE_PAD) } else { i64::from(*token) };
        let history: Vec<i64> = before[before.len().saturating_sub(2)..].iter().map(id).collect();
        let tokens: Vec<i64> = tokens.iter().map(id).collect();
        ngram_rows(&history, &tokens, &self.weights.integer_arrays)
    }

    fn ple_embeddings(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let started = Instant::now();
        let rows = self.weights.rows(PLE_TABLE, &self.ple_indices(&self.history, tokens)?);
        self.ple_wait += started.elapsed().as_secs_f64();
        rows
    }

    /// Add the time since `clock` to the profile's `name` and restart the clock (with DS41_PROFILE).
    fn phase(&mut self, name: &'static str, clock: &mut Instant) {
        if let Some(profile) = self.profile.as_mut() {
            *profile.pass.entry(name).or_default() += lap(clock);
        }
    }

    /// Start reading the PLE rows of `tokens` (after `before`) as soon as they are known: the
    /// pass that decodes them then waits only for rows still on their way.
    fn prefetch_ple(&self, before: &[u32], tokens: &[u32]) -> Result<()> {
        // A batch is at most 4096 tokens (the n-gram rows' limit); longer runs go in pieces.
        for start in (0..tokens.len()).step_by(4096) {
            let before = if start == 0 { before } else { &tokens[..start] };
            let part = &tokens[start..(start + 4096).min(tokens.len())];
            self.weights.prefetch_rows(PLE_TABLE, &self.ple_indices(before, part)?)?;
        }
        Ok(())
    }

    fn token_embeddings(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let ids: Vec<i64> = tokens.iter().map(|token| *token as i64).collect();
        self.weights.rows("token_embd.weight", &ids)
    }

    /// The rows the first layer reads for `tokens` from `position`: the token table's, an image's own where its span is.
    fn rows_at(&self, position: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        if !tokens.iter().any(|token| *token >= IMAGE_TOKEN) {
            return self.token_embeddings(tokens);
        }
        let ids: Vec<u32> = tokens.iter().map(|token| if *token >= IMAGE_TOKEN { IMAGE_PAD } else { *token }).collect();
        let mut rows = self.token_embeddings(&ids)?;
        for span in &self.images {
            let (from, to) = (span.start.max(position), (span.start + span.grid.0 * span.grid.1).min(position + tokens.len()));
            if from < to {
                let source = span.rows.get((from - span.start) * HIDDEN..(to - span.start) * HIDDEN)
                    .ok_or_else(|| format!("Straydog image rows are short {IMAGE_REFUSED}"))?;
                rows[(from - position) * HIDDEN..(to - position) * HIDDEN].copy_from_slice(source);
            }
        }
        Ok(rows)
    }

    /// The next request's images (none for a prompt without any).
    pub fn set_images(&mut self, images: Vec<ImageSpan>) {
        self.images = images;
    }

    /// The devices' rope positions for a pass over `tokens` from `start`: as the sequence's images say, or, as ever, the rows' own.
    /// Whether the rows or the block the first of them is in touch an image: only a batch's table has their positions then.
    fn set_rope(&mut self, start: usize, tokens: usize) -> Result<bool> {
        let end = start + tokens;
        if self.images.first().is_none_or(|first| first.start >= end) {
            if self.rope_delta != 0 {
                self.rope_delta = 0;
                self.group.rope_delta(0)?;
            }
            if std::mem::take(&mut self.rope_table) {
                self.group.rope_rows(0, &[])?;
            }
            return Ok(false);
        }
        // (the table reaches back to the block of the MTP layer's pair before these rows, which is at the row before them)
        let block = start.saturating_sub(1) / 4 * 4;
        let touched = self.images.iter().any(|span| span.start < end && span.start + span.grid.0 * span.grid.1 > block);
        if touched && tokens < 2 {
            return Err(format!("an image in a pass of one row {IMAGE_REFUSED}"));
        }
        self.set_rope_delta(start)?;
        if tokens > 1 {
            let axes: Vec<u32> = (block..end).flat_map(|position| qwen_vision::rope_of(&self.images, position)).collect();
            self.group.rope_rows(block as u64, &axes).map_err(|error| format!("{error} {IMAGE_REFUSED}"))?;
            self.rope_table = true;
        }
        Ok(touched)
    }

    /// The devices' delta of the decode and verification rows at `position`, as the images before it say.
    fn set_rope_delta(&mut self, position: usize) -> Result<()> {
        let delta = qwen_vision::rope_delta(&self.images, position);
        if delta != self.rope_delta {
            self.rope_delta = delta;
            self.group.rope_delta(delta)?;
        }
        Ok(())
    }

    fn with_operands<T>(&mut self, build: impl FnOnce(&Model) -> Result<Vec<Operand<'_>>>,
                        run: impl FnOnce(&mut Group, &[Operand]) -> Result<T>) -> Result<T> {
        let operands = build(self)?;
        // SAFETY of the borrow: operands point into the weights, which the group never frees.
        let operands: Vec<Operand<'static>> = unsafe { std::mem::transmute(operands) };
        run(&mut self.group, &operands)
    }

    /// The working set's limit, which follows what is free ([`crate::host::watch_working_set`]), after what a K/V growth has just locked.
    fn hold_memory(&self) {
        if self.partial {
            let pinned = self.pinned_other + attention_layers(self.weights.drafter) * self.kv_capacity as u64 * 4096;
            self.pinned_now.store(pinned, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The K/V's next size for `positions`: twice over where the commit and the memory have room for that, else a quarter more, else what the
    /// sequence needs, which only the commit can refuse: a sequence at 66K locked 6 GiB for 131072 positions. The commit is read
    /// with the slots' VRAM still charged to it: they give it up for the growth and take it back after, so what the growth takes of it
    /// beyond this they would not find (IQ3_XXS at auto: an hour's conversation past 65536 ended the worker).
    fn kv_capacity_for(&self, positions: usize) -> Result<usize> {
        let doubled = (positions + 4096).next_power_of_two().min(self.context);
        let Some((free, left)) = crate::host::commit_left() else { return Ok(doubled) };
        let position = attention_layers(self.weights.drafter) * 4096;
        // (a layer's old K/V and new one are both held while it is copied)
        let cost = |capacity: usize| (capacity - self.kv_capacity) as u64 * position + capacity as u64 * 4096 + KV_COMMIT_MARGIN;
        let least = (positions + 4096).div_ceil(4096).saturating_mul(4096).min(self.context);
        let quarter = least.max((self.kv_capacity + self.kv_capacity / 4).div_ceil(4096) * 4096).min(self.context);
        let spare = free.saturating_sub(crate::host::left_free(self.weights.host_memory_bytes));
        for capacity in [doubled, quarter] {
            let locked = (capacity - self.kv_capacity) as u64 * position;
            if cost(capacity) <= left && locked <= spare {
                return Ok(capacity);
            }
        }
        if cost(least) <= left {
            return Ok(least);
        }
        Err(KV_NO_MEMORY.into())
    }

    /// `tokens`, with the tokens of the sequence it goes on from wherever both spell the same text. A reply read back is tokenized anew, and a
    /// model does not always write the tokens the tokenizer would: one such place in a long reply cost all of it, read again from the last
    /// checkpoint (25000 tokens, 139 s, for a reply with two files in it).
    pub fn agree_with_held(&self, tokens: &[u32], text: &crate::constraint::TokenBytes) -> Vec<u32> {
        let parked = self.parked.as_ref().map_or(&[][..], |parked| &parked.tokens[..]);
        let held = if shared(parked, tokens) > shared(&self.history, tokens) { parked } else { &self.history[..] };
        same_text(tokens, held, &|token| text.text_of(token))
    }

    /// Room in the K/V for `positions`. A parked sequence's copies have the old pitches and go; the
    /// live sequence and the checkpoints below it stay.
    fn ensure_kv(&mut self, positions: usize) -> Result<()> {
        if positions <= self.kv_capacity {
            return Ok(());
        }
        if let Some(parked) = self.parked.take() {
            self.discard(parked)?;
        }
        let capacity = self.kv_capacity_for(positions)?;
        let pinned = self.pinned_other + attention_layers(self.weights.drafter) * capacity as u64 * 4096;
        crate::host::keep_resident(self.weights.host_memory_bytes, pinned, self.banks_bytes);
        self.group.grow_kv(capacity as u64, self.history.len() as u64)?;
        self.kv_capacity = capacity;
        self.hold_memory();
        // (a checkpoint's history of the MTP layer's indexer, which keeps it after its blocks)
        for checkpoint in self.checkpoints.values_mut() {
            if let Some(history) = checkpoint.history.as_mut() {
                history.move_history(capacity as u64);
            }
        }
        Ok(())
    }

    /// Advance the model over `tokens`; the logits after the last one when asked (None on cancellation).
    fn forward(&mut self, tokens: &[u32], logits: bool, cancelled: &dyn Fn() -> bool) -> Result<Option<Option<Vec<f32>>>> {
        if self.history.len() + tokens.len() > self.context
            || tokens.iter().any(|token| *token as usize >= VOCABULARY && (*token < IMAGE_TOKEN || self.images.is_empty()))
        {
            return Err("Straydog token or context is out of range".into());
        }
        // Every write below lies in these positions: the layers' rows of the tokens, and the MTP
        // layer's pairs (from the row before them), whichever path they take (a verified short
        // prefill, a resident or pipelined one with the MTP rows on the host, stashed or kept, a
        // decode step), all from here.
        let start = self.history.len();
        let touches_image = self.set_rope(start, tokens.len())?;
        self.ensure_kv(start + tokens.len())?;
        self.protect(start.saturating_sub(usize::from(self.speculative > 0)), start + tokens.len())?;
        self.prefetch_ple(&self.history, tokens)?;
        let embeddings = self.rows_at(start, tokens)?;
        let previous = self.final_hidden.clone();
        let position = self.history.len() as u64;
        // Inside a pipelined prompt the last batch's rows stayed on the GPUs: a batch on the
        // verified or the one-token path pairs its first token with that batch's last row, read
        // back from where the MTP layer's device kept it (the next batch pairs from this one's).
        let verified = tokens.len() > 1 && self.resident && tokens.len() <= VERIFIED_PREFILL && !touches_image;
        let previous = match previous {
            None if verified || tokens.len() == 1 => {
                let kept = self.kept_hidden()?;
                self.mtp_kept = 0;
                kept
            }
            previous => previous,
        };
        let hidden = if verified {
            let start = self.history.len();
            let rows = self.prefill_verified(tokens)?;
            // forward records the tokens below, after the MTP layer has read the prompt.
            self.history.truncate(start);
            if self.speculative > 0 {
                self.mtp_prompt(tokens, &rows, previous)?;
            }
            rows
        } else if tokens.len() > 1 {
            // With another device to keep them on, the prompt's rows never leave the GPUs.
            let stash = self.speculative > 0 && !self.deferred && self.group.mtp_stash();
            let Some(rows) = self.resident_prefill(&embeddings, tokens, cancelled, stash, logits || !self.deferred)? else {
                return Ok(None);
            };
            if self.speculative > 0 {
                if self.deferred || self.group.mtp_parts() {
                    // (the MTP layer in parts reads the rows both devices kept, deferred or not)
                    self.mtp_prompt_kept(tokens, previous)?;
                    if !self.deferred {
                        self.mtp_kept = 0;
                    }
                } else if stash {
                    self.mtp_prompt_resident(tokens, previous)?;
                } else {
                    self.mtp_prompt(tokens, &rows, previous)?;
                }
            }
            rows
        } else {
            let mut clock = Instant::now();
            let mut hidden = self.group.decode_span(0, 1, &tile(&embeddings), position)?;
            self.phase("step: first layer", &mut clock);
            if cancelled() {
                self.reset()?;
                return Ok(None);
            }
            let embeddings = self.ple_embeddings(tokens)?;
            self.phase("step: PLE rows wait", &mut clock);
            hidden = self.with_operands(Model::ple_operands, |group, operands| {
                group.ple("blk.1.", &hidden, &embeddings, operands)
            })?;
            self.phase("step: PLE layer", &mut clock);
            for (index, (begin, end)) in self.group.decoder_spans(1).into_iter().enumerate() {
                if cancelled() {
                    self.reset()?;
                    return Ok(None);
                }
                hidden = self.group.decode_span(begin, end, &hidden, position)?;
                self.phase(["step: layers (first device)", "step: layers (second device)"][index.min(1)], &mut clock);
            }
            if self.speculative > 0 {
                // The MTP layer's pair of the row before, its token now known (the longer
                // prompts' paths write theirs from there too).
                self.mtp_prompt(tokens, &hidden, previous)?;
            }
            hidden
        };
        self.history.extend_from_slice(tokens);
        self.draft_note(tokens);
        self.final_hidden = (!hidden.is_empty()).then(|| last_row(&hidden).to_vec());
        if !logits {
            return Ok(Some(None));
        }
        let mut clock = Instant::now();
        let row = last_row(&hidden).to_vec();
        let mixed = self.with_operands(
            |model| Ok(model.head_operands()?[..3].to_vec()),
            |group, operands| group.hyperconnection("output_hc", &row, operands[0].clone(), &operands[1..]),
        )?;
        let output = self.with_operands(|model| Ok(vec![model.projection("output.weight")?]), |group, operands| {
            group.project_many(&[(operands[0].clone(), &mixed[..])])
        })?;
        let logits = output.into_iter().next().unwrap();
        if tokens.len() == 1 {
            self.phase("step: head", &mut clock);
            if let Some(profile) = self.profile.as_mut() {
                profile.end_pass(1, 0.0);
            }
        }
        if let Some(digests) = self.logits_digests.as_mut() {
            let bytes: Vec<u8> = logits.iter().flat_map(|value| value.to_le_bytes()).collect();
            digests.push(format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&bytes)));
        }
        if let Some(capture) = self.logits_capture.as_mut() {
            capture.push(logits.clone());
        }
        Ok(Some(Some(logits)))
    }

    /// The last processed token's four-stream hidden state.
    pub fn final_hidden(&self) -> Option<&[f32]> {
        self.final_hidden.as_deref()
    }

    /// Diagnostics: a digest of the live sequence's state that is valid (its positions' K/V and
    /// complete pooled blocks, the indexers' history, the recurrent state).
    pub fn state_digest(&mut self) -> Result<Vec<String>> {
        self.group.state_digest(self.history.len() as u64)
    }

    /// Tests only: let parked sequences hold at most `positions` positions, to run the paths of a
    /// budget too low for a swap or for a copy (model_gpu's parking scenarios).
    #[doc(hidden)]
    pub fn limit_parking(&mut self, positions: usize) {
        self.park_budget = positions as u64 * self.group.position_bytes();
    }

    /// Tests only: park each departing sequence whole at once, every position copied before any
    /// later prompt runs and all of them restored: the reference that parking only what later
    /// prompts overwrite must equal bit for bit (model_gpu's parking scenarios).
    #[doc(hidden)]
    pub fn park_whole(&mut self) {
        self.park_whole = true;
    }

    /// Diagnostics: the parked sequence's positions copied out (`from..to`, if any) and its length.
    pub fn parked_coverage(&self) -> Option<(Option<(usize, usize)>, usize)> {
        self.parked.as_ref().map(|parked| (parked.covered, parked.tokens.len()))
    }

    /// Keep prompt rows on each GPU between spans; control returns after at most four layers
    /// for cancellation, and rows cross the host only at device boundaries.
    /// `stash`: the MTP layer reads every final row from the first layer's workspace, so only
    /// the last comes back; otherwise speculation takes them all.
    /// `keep_rows`: whether the last span returns rows (a deferred batch before the last does not).
    fn resident_prefill(&mut self, embeddings: &[f32], tokens: &[u32], cancelled: &dyn Fn() -> bool, stash: bool,
                        keep_rows: bool) -> Result<Option<Vec<f32>>> {
        let (steps, position) = (tokens.len() as u64, self.history.len() as u64);
        let boundaries = self.group.decoder_spans(1);
        let mut spans = vec![(0, 1)];
        for (begin, end) in &boundaries {
            spans.extend((*begin..*end).step_by(4).map(|first| (first, (first + 4).min(*end))));
        }
        // The prompt's embeddings start the first span; later spans continue on the device, and
        // rows move GPU to GPU where the next span is on the other device.
        self.group.prefill_embed(0, embeddings)?;
        let mut rows_out = Vec::new();
        let mut previous: Option<usize> = None;
        for (index, (first, last)) in spans.into_iter().enumerate() {
            if let Some(previous) = previous {
                self.group.prefill_handoff(previous, first, steps)?;
            }
            previous = Some(last - 1);
            if cancelled() {
                self.reset()?;
                return Ok(None);
            }
            if index == 1 {
                let embeddings = self.ple_embeddings(tokens)?;
                self.with_operands(Model::ple_operands, |group, operands| {
                    group.prefill_ple("blk.1.", &embeddings, operands)
                })?;
            }
            // Speculation keeps every final row: the MTP layer reads the prompt's hidden states.
            let rows = match last {
                48 if !keep_rows => 0,
                48 => if self.speculative > 0 && !stash && !self.deferred && !self.group.mtp_parts() { steps } else { 1 },
                _ => 0,
            };
            let output = self.group.prefill_layers(first, last, None, position, steps, rows)?;
            if rows > 0 {
                rows_out = output;
            }
        }
        if stash && !self.group.prefill_handoff(47, 0, steps)? {
            return Err("Straydog could not keep the prompt rows for the MTP layer".into());
        }
        Ok(Some(rows_out))
    }

    /// Advance the MTP layer's K/V over prologue rows starting at `position`.
    fn mtp_layer(&mut self, rows: &[f32], position: u64) -> Result<()> {
        let steps = row_count(rows) as u64;
        if steps == 1 {
            self.group.verify_span(48, 49, rows, position)?;
        } else {
            self.group.prefill_layers(48, 49, Some(rows), position, steps, 0)?;
        }
        Ok(())
    }

    fn mtp_operands(&self) -> Result<Vec<Operand<'_>>> {
        Ok(vec![self.vector("mtp.embed_norm.weight", None)?, self.q8_projection("mtp.fc_embedding.weight")?,
                self.vector("mtp.hidden_norm.weight", None)?, self.q8_projection("mtp.fc_hidden.weight")?])
    }

    fn mtp_prologue(&mut self, embeddings: &[f32], hidden: &[f32]) -> Result<Vec<f32>> {
        self.with_operands(Model::mtp_operands, |group, operands| group.mtp_prologue(operands, embeddings, hidden))
    }

    /// A drafting step's operands: the MTP layer's prologue, then the head's mixer and the shared vocabulary head.
    fn draft_operands(&self, head: bool) -> Result<Vec<Operand<'_>>> {
        let mut operands = self.mtp_operands()?;
        if head {
            operands.extend(self.mtp_head_operands()?);
        }
        Ok(operands)
    }

    /// One drafting step with the host out of its middle: the prologue over `embeddings` and `hidden` (`None`: the last row
    /// the layer's last step left on the device, of `previous` rows), the MTP layer, and the drafter's token for the last
    /// row through the layer's own mixer and the shared vocabulary head with its probability under the drafter.
    fn draft_step(&mut self, embeddings: &[f32], hidden: Option<&[f32]>, previous: u64, position: u64, head: bool, top: usize)
                  -> Result<Option<crate::gpu::Draft>> {
        self.with_operands(|model| model.draft_operands(head), |group, operands| {
            group.draft_step(operands, position, embeddings, hidden, previous, top)
        })
    }

    /// [`Model::mtp_prompt`] with the prompt's rows kept on the GPUs (see [`Group::mtp_stash`]):
    /// the same pairs, in the same batches of 512.
    fn mtp_prompt_resident(&mut self, tokens: &[u32], previous: Option<Vec<f32>>) -> Result<()> {
        let start = self.history.len();
        let previous = previous.filter(|_| start > 0);
        let (following, first) = if previous.is_some() { (tokens, start - 1) } else { (&tokens[1..], start) };
        for offset in (0..following.len()).step_by(512) {
            let end = (offset + 512).min(following.len());
            let embedded = self.rows_at(first + 1 + offset, &following[offset..end])?;
            // Pair j's hidden row is the prompt's row j, or row j - 1 after the previous row.
            let (row, earlier) = match previous.as_deref() {
                Some(row) if offset == 0 => (0, Some(row)),
                Some(_) => (offset - 1, None),
                None => (offset, None),
            };
            self.with_operands(Model::mtp_operands, |group, operands| {
                group.mtp_prompt_rows(operands, row, end - offset, earlier, &embedded, (first + offset) as u64)
            })?;
        }
        Ok(())
    }

    /// [`Model::mtp_prompt_resident`] inside a pipelined prefill: the batch's rows stay on the
    /// MTP layer's device after the row before them (carried there from the last batch, else
    /// `previous` from the host), and its pairs run on that device's stream with no host wait.
    fn mtp_prompt_kept(&mut self, tokens: &[u32], previous: Option<Vec<f32>>) -> Result<()> {
        let start = self.history.len();
        let host = if self.mtp_kept == 0 { previous.filter(|_| start > 0) } else { None };
        let before = self.mtp_kept > 0 || host.is_some();
        self.group.mtp_keep(tokens.len() as u64, self.mtp_kept, host.as_deref())?;
        self.mtp_kept = tokens.len() as u64;
        let (following, first, row) = if before { (tokens, start - 1, 0) } else { (&tokens[1..], start, 1) };
        if following.is_empty() {
            return Ok(());
        }
        let embedded = self.rows_at(first + 1, following)?;
        self.group.mtp_embed(&embedded)?;
        for offset in (0..following.len()).step_by(512) {
            let count = (following.len() - offset).min(512) as u64;
            self.with_operands(Model::mtp_operands, |group, operands| {
                group.mtp_prompt_kept(operands, (row + offset) as u64, offset as u64, count, (first + offset) as u64)
            })?;
        }
        Ok(())
    }

    /// Fill the MTP layer's K/V for a prompt batch: pair (hidden_t, token_t+1) sits at position t. The token's row at an image's
    /// position is the image's own (llama.cpp's MTP graph takes `inp->embd` there), through the layer's prologue as any.
    fn mtp_prompt(&mut self, tokens: &[u32], rows: &[f32], previous: Option<Vec<f32>>) -> Result<()> {
        let start = self.history.len();
        let mut hidden: Vec<&[f32]> = rows.chunks_exact(WIDTH).take(row_count(rows) - 1).collect();
        let mut following: Vec<u32> = tokens[1..].to_vec();
        let mut first = start;
        if start > 0 {
            if let Some(previous) = previous.as_deref() {
                hidden.insert(0, previous);
                following.insert(0, tokens[0]);
                first = start - 1;
            }
        }
        let mut offset = 0;
        while offset < hidden.len() {
            let mut end = (offset + 512).min(hidden.len());
            // (a last pass of one row reads the delta alone: in a sequence with images it stays in the table's)
            if !self.images.is_empty() && hidden.len() - end == 1 {
                end -= 1;
            }
            let chunk: Vec<f32> = hidden[offset..end].concat();
            let embedded = self.rows_at(first + 1 + offset, &following[offset..end])?;
            let prologue = self.mtp_prologue(&embedded, &chunk)?;
            self.mtp_layer(&prologue, (first + offset) as u64)?;
            offset = end;
        }
        Ok(())
    }

    /// Note the tokens past DRAFT_ROWS among `tokens` (entering the history) for the draft's vocabulary.
    fn draft_note(&mut self, tokens: &[u32]) {
        if self.draft_vocabulary {
            for &token in tokens {
                if u64::from(token) >= DRAFT_ROWS && (token as usize) < VOCABULARY && self.draft_extras.insert(token) {
                    self.draft_pending.push(token);
                }
            }
        }
    }

    /// The noted tokens' rows into the draft's vocabulary; when it is full, the history's tokens alone (the latest first).
    fn draft_vocabulary_update(&mut self) -> Result<()> {
        if self.draft_pending.is_empty() {
            return Ok(());
        }
        let reset = self.draft_extras.len() > DRAFT_EXTRAS;
        if reset {
            self.draft_extras.clear();
            self.draft_pending.clear();
            for &token in self.history.iter().rev() {
                if u64::from(token) >= DRAFT_ROWS && self.draft_extras.insert(token) {
                    self.draft_pending.push(token);
                    if self.draft_pending.len() == DRAFT_EXTRAS {
                        break;
                    }
                }
            }
        }
        let pending = std::mem::take(&mut self.draft_pending);
        self.with_operands(|model| Ok(vec![model.projection("output.weight")?]), |group, operands| {
            group.draft_vocabulary(&operands[0], DRAFT_ROWS, &pending, reset)
        })?;
        Ok(())
    }

    /// The cut of a partially resident model's drawn draft, from what a second row costs this machine at this budget, measured: one
    /// pass with a draft costs c passes without. A draft is kept with the overlap of the two distributions, which 2319 rows of a served
    /// run put near 0.45 + 0.4 * the drawn distribution's greatest probability (0.95 past 0.9). Weighing that overlap against c - 1 at the
    /// rate a pass already earns gives the line below, and the sweeps agree, tok/s by cut (0, 0.3, 0.45, 0.6): R9700 at --vram 8, c 1.36:
    /// 67.4, 67.1, 65.6, 64.4; at --vram 5, c 1.5: 52.1, 53.4, 52.9, 52.2; RTX 5070 Ti at --vram 12, c 1.65: 66.6, 68.1 at 0.4, 69.1 at 0.6.
    fn partial_drawn_cut(&self) -> f32 {
        match self.pass_seconds {
            [one, two] if one > 0.0 && two > 0.0 => (2.75 * (two / one - 1.0) - 1.125).clamp(0.0, 0.9) as f32,
            _ => PARTIAL_DRAFT_CUT,
        }
    }

    /// Catch the MTP layer up on the accepted pairs, then chain up to `limit` greedy drafts
    /// while the product of their probabilities under the drafter stays at the cut.
    /// Each step is one call: a draft's row stays on the device for the next step to read.
    ///
    /// A sampled request whose draw reads a row's greatest logits ([`Sampling::top_candidates`]) has its drafts drawn from the
    /// drafter's own: they come back with those rows, for [`sampling::accept_top_drawn`], and the probability is that of the drawn
    /// distribution's most likely token.
    ///
    /// With `suppression` a drafter's row is that distribution without what it refuses, and a chain ends where nothing is left of
    /// it, or where its most likely token is refused: a draft that could only be rejected takes no row.
    #[allow(clippy::too_many_arguments)]
    fn draft(&mut self, hidden: &[f32], tokens: &[u32], position: u64, limit: usize, sampling: &Sampling, rng: &mut Rng,
             mut suppression: Option<Suppression>) -> Result<(Vec<u32>, Vec<Vec<(f32, u32)>>)> {
        // (the catch-up's pairs, then a row for each draft tried, a failed one too)
        let mut clock = Instant::now();
        self.ensure_kv(position as usize + tokens.len() + limit + 1)?;
        self.protect(position as usize, position as usize + tokens.len() + limit)?;
        self.draft_vocabulary_update()?;
        let embedded = self.token_embeddings(tokens)?;
        self.phase("draft: embeddings", &mut clock);
        let top = sampling.top_candidates().unwrap_or(0);
        let mut proposed = Vec::new();
        let mut take = |mut draft: crate::gpu::Draft, rng: &mut Rng| -> Result<Option<(u32, f32)>> {
            if let Some(suppression) = &suppression {
                draft.top.retain(|&(_, token)| suppression.allows(token));
            }
            let taken = if top == 0 {
                (u32::try_from(draft.token).map_err(|_| "Straydog drafter head returned no token")?, draft.confidence)
            } else if draft.top.is_empty() {
                return Ok(None);
            } else {
                let drawn = sampling::draw_top(&draft.top, sampling, rng);
                proposed.push(draft.top);
                drawn
            };
            if let Some(suppression) = &mut suppression {
                if !suppression.allows(taken.0) {
                    return Ok(None);
                }
                suppression.advance(taken.0);
            }
            Ok(Some(taken))
        };
        let step = self.draft_step(&embedded, Some(hidden), 0, position, limit > 0, top)?;
        self.phase("draft: catch-up and first draft", &mut clock);
        let Some(step) = step else { return Ok((Vec::new(), Vec::new())) };
        let Some((token, mut chain)) = take(step, rng)? else { return Ok((Vec::new(), Vec::new())) };
        let cut = match (top > 0, self.partial) {
            (true, true) => self.partial_drawn_cut(),
            (true, false) => DRAWN_DRAFT_CUT,
            (false, true) => PARTIAL_DRAFT_CUT,
            (false, false) => DRAFT_CUT,
        };
        // (at a cut of 0 no pass would run alone, and what a lone row costs would never be measured again)
        if chain < cut || (self.partial && self.passes_with_draft >= PASSES_BETWEEN_LONE_ROWS) {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut drafts = vec![token];
        let (mut rows, mut position) = (tokens.len() as u64, position + tokens.len() as u64);
        while drafts.len() < limit {
            let embedded = self.token_embeddings(&drafts[drafts.len() - 1..])?;
            self.phase("draft: embeddings", &mut clock);
            let step = self.draft_step(&embedded, None, rows, position, true, top)?
                .ok_or("Straydog drafter head returned no probability")?;
            let Some((token, probability)) = take(step, rng)? else { break };
            self.phase("draft: next draft", &mut clock);
            chain *= probability;
            if chain < cut {
                break;
            }
            drafts.push(token);
            (rows, position) = (1, position + 1);
        }
        proposed.truncate(drafts.len());
        Ok((drafts, proposed))
    }

    /// The next token and its drafts through every layer with the one-token arithmetic.
    #[allow(clippy::type_complexity)]
    /// The prompt's batches from `position`, with the checkpoints of `plan`: the last batch's
    /// logits, or None when cancelled.
    fn prefill(&mut self, tokens: &[u32], position: usize, plan: &[(usize, Vec<&'static str>)],
               cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize)) -> Result<Option<Option<Vec<f32>>>> {
        // A long prompt's batches take the slots' VRAM until its end, however it ends.
        let mut batch = self.prefill_batch_size;
        // (a pass over an image's rows is one of at least two)
        if !self.images.is_empty() {
            batch = batch.max(2);
        }
        // (a batch that rests narrow widens sooner: 256 rows at a time would take a thousand tokens in four passes)
        let long = self.partial && tokens.len() - position >= LONG_PREFILL.min(self.prefill_batch_size * 2);
        if long {
            self.ensure_kv(tokens.len())?;
            if let Some(rows) = self.group.begin_long_prefill()? {
                batch = batch.max(rows as usize);
            }
        }
        let output = self.prefill_batches(tokens, position, plan, batch, cancelled, progress);
        if long {
            self.group.end_long_prefill()?;
        }
        output
    }

    fn prefill_batches(&mut self, tokens: &[u32], mut position: usize, plan: &[(usize, Vec<&'static str>)], batch: usize,
                       cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize)) -> Result<Option<Option<Vec<f32>>>> {
        let mut output = None;
        let stops: Vec<usize> = plan.iter().map(|(at, _)| *at).chain([tokens.len()]).collect();
        for stop in stops {
            while position < stop {
                if cancelled() {
                    return Ok(None);
                }
                let mut end = (position + batch).min(stop);
                // (none is left to a batch of one row, which may be an image's or end a block begun in one)
                if !self.images.is_empty() && stop - end == 1 && end - position > 2 {
                    end -= 1;
                }
                // The next batch's PLE rows arrive while this one runs.
                let next = (end + batch).min(tokens.len());
                self.prefetch_ple(&tokens[..end], &tokens[end..next])?;
                // A one-token batch runs whole, as decode does; a longer one checks between spans.
                let check: &dyn Fn() -> bool = if self.prefill_batch_size == 1 { &|| false } else { cancelled };
                let Some(logits) = self.forward(&tokens[position..end], end == tokens.len(), check)? else {
                    return Ok(None);
                };
                output = logits;
                if cancelled() {
                    return Ok(None);
                }
                position = end;
                progress(end, tokens.len());
            }
            if let Some((_, kinds)) = plan.iter().find(|(at, _)| *at == stop) {
                for kind in kinds.clone() {
                    self.save_checkpoint(kind)?;
                }
            }
        }
        Ok(Some(output))
    }

    /// Whether a prompt's batches overlap on two GPUs: the first device starts batch n+1 on its
    /// layers while the second finishes batch n on its own, nothing waiting on the host between
    /// (with experts split, each also helps the other's batch on its helper stream).
    fn pipelined_prefill(&self, tokens: usize) -> Result<bool> {
        Ok(self.images.is_empty() && self.resident && (self.speculative == 0 || self.group.keeps_mtp_rows()) && self.group.backends.len() == 2
           && tokens > self.prefill_batch_size)
    }

    /// The rows through every layer and the head: with `sampling`, each row's logits for the draw, or its `top` greatest
    /// when the draw reads only those (sampling::Sampling::top_candidates: the GPU picks them, the host reads a few pairs a row).
    fn verify(&mut self, tokens: &[u32], sampling: bool, top: usize) -> Result<Verified> {
        if !self.partial && self.group.chains_verification() {
            let embeddings = self.verify_chained(tokens)?;
            let mut clock = Instant::now();
            let ((targets, scores, _), hidden) = self.with_operands(Model::head_operands, |group, operands| {
                group.output_head_chained(operands, 47, tokens.len(), sampling, top, false)
            })?;
            self.phase("verify: head", &mut clock);
            return Ok((Vec::new(), embeddings, hidden, targets, scores));
        }
        let (first, embeddings, hidden) = self.verify_rows(tokens)?;
        let mut clock = Instant::now();
        let (targets, scores, _) = self.with_operands(Model::head_operands, |group, operands| {
            group.output_head(operands, &hidden, sampling, top, false)
        })?;
        self.phase("verify: head", &mut clock);
        Ok((first, embeddings, hidden, targets, scores))
    }

    /// A short prompt through the verification spans, up to [`PROMPT_ROWS`] rows per pass and every row
    /// kept: its fixed cost is that of a few decode steps rather than of a batched prefill.
    fn prefill_verified(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut rows = Vec::with_capacity(tokens.len() * WIDTH);
        for chunk in tokens.chunks(PROMPT_ROWS) {
            let (first, embeddings, hidden) = self.verify_rows(chunk)?;
            self.commit(chunk, chunk.len(), &first, &embeddings)?;
            rows.extend_from_slice(&hidden);
        }
        Ok(rows)
    }

    /// [`Model::verify_rows`] with the host out of the pass (Group::chains_verification): the first layer, the PLE layer and
    /// every span queued on their devices one after the other, the rows handed GPU to GPU between them, for the head to read
    /// where the last span left them. The PLE rows' embeddings come back (a partial acceptance replays them).
    fn verify_chained(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut clock = Instant::now();
        self.protect(self.history.len(), self.history.len() + tokens.len())?;
        let position = self.history.len() as u64;
        self.prefetch_ple(&self.history, tokens)?;
        let hidden = tile(&self.token_embeddings(tokens)?);
        self.phase("verify: protect, PLE prefetch, embeddings", &mut clock);
        self.group.verify_queue(0, 1, &hidden, position)?;
        self.phase("verify: first layer", &mut clock);
        let embeddings = self.ple_embeddings(tokens)?;
        self.phase("verify: PLE rows wait", &mut clock);
        self.with_operands(Model::ple_operands, |group, operands| group.ple_front_queue("blk.1.", &embeddings, operands))?;
        self.phase("verify: PLE layer", &mut clock);
        let mut after = 1;
        for (begin, end) in self.group.decoder_spans(1) {
            self.group.verify_chain(after, begin, end, tokens.len(), position)?;
            after = end;
        }
        self.phase("verify: layers (queued)", &mut clock);
        Ok(embeddings)
    }

    fn verify_rows(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        // (every row verified is written, kept or not)
        let mut clock = Instant::now();
        self.ensure_kv(self.history.len() + tokens.len())?;
        self.protect(self.history.len(), self.history.len() + tokens.len())?;
        let position = self.history.len() as u64;
        self.prefetch_ple(&self.history, tokens)?;
        let hidden = tile(&self.token_embeddings(tokens)?);
        self.phase("verify: protect, PLE prefetch, embeddings", &mut clock);
        let (first, embeddings, mut hidden) = if !self.partial && self.group.front_chained() {
            // The first layer runs on the device while the host waits for the PLE rows; the PLE layer then follows it
            // there, and one wait reads back both layers' rows.
            self.group.verify_queue(0, 1, &hidden, position)?;
            self.phase("verify: first layer", &mut clock);
            let embeddings = self.ple_embeddings(tokens)?;
            self.phase("verify: PLE rows wait", &mut clock);
            let (first, hidden) = self.with_operands(Model::ple_operands, |group, operands| {
                group.ple_front("blk.1.", &embeddings, operands)
            })?;
            self.phase("verify: PLE layer", &mut clock);
            (first, embeddings, hidden)
        } else {
            let first = self.group.verify_span(0, 1, &hidden, position)?;
            self.phase("verify: first layer", &mut clock);
            let embeddings = self.ple_embeddings(tokens)?;
            self.phase("verify: PLE rows wait", &mut clock);
            self.group.copy_state("@ple-snapshot:blk.1.", "@ple:blk.1.", "blk.1.")?;
            let hidden = self.with_operands(Model::ple_operands, |group, operands| {
                group.ple_rows("blk.1.", &first, &embeddings, operands)
            })?;
            self.phase("verify: PLE layer", &mut clock);
            (first, embeddings, hidden)
        };
        for (index, (begin, end)) in self.group.decoder_spans(1).into_iter().enumerate() {
            hidden = self.group.verify_span(begin, end, &hidden, position)?;
            self.phase(["verify: layers (first device)", "verify: layers (second device)"][index.min(1)], &mut clock);
        }
        Ok((first, embeddings, hidden))
    }

    /// Keep the first `kept` verified tokens; replay DeltaNet and PLE state for the rest.
    fn commit(&mut self, tokens: &[u32], kept: usize, first: &[f32], embeddings: &[f32]) -> Result<()> {
        let mut clock = Instant::now();
        for (begin, end) in std::iter::once((0, 1)).chain(self.group.decoder_spans(1)) {
            self.group.verify_commit(begin, end, kept as u64)?;
        }
        self.phase("commit: DeltaNet replay", &mut clock);
        if kept < tokens.len() {
            let embeddings = &embeddings[..kept * HIDDEN];
            if !self.partial && self.group.front_chained() {
                self.with_operands(Model::ple_operands, |group, operands| group.ple_replay("blk.1.", embeddings, operands))?;
            } else {
                self.group.copy_state("@ple:blk.1.", "@ple-snapshot:blk.1.", "blk.1.")?;
                let first = &first[..kept * WIDTH];
                self.with_operands(Model::ple_operands, |group, operands| {
                    group.ple_rows("blk.1.", first, embeddings, operands)
                })?;
            }
            self.phase("commit: PLE replay", &mut clock);
        }
        self.history.extend_from_slice(&tokens[..kept]);
        self.draft_note(&tokens[..kept]);
        Ok(())
    }

    /// Accepted draft count and the next token: a match with the GPU's argmax when greedy,
    /// else decided on the rows' logits (see [`sampling::accept`]).
    #[allow(clippy::too_many_arguments)]
    fn accept(drafts: &[u32], proposed: &[Vec<(f32, u32)>], targets: &[i32], scores: Option<&Scores>, sampling: &Sampling, generated: &mut Generated,
              rng: &mut Rng) -> (usize, u32) {
        let scores = match scores.filter(|_| sampling.needs_logits()) {
            Some(Scores::Top(top)) if proposed.len() == drafts.len() => return sampling::accept_top_drawn(drafts, proposed, top, sampling, generated, rng),
            Some(Scores::Top(top)) => return sampling::accept_top(drafts, top, sampling, generated, rng),
            Some(Scores::Logits(logits)) => Some(logits.as_slice()),
            None => None,
        };
        let Some(scores) = scores else {
            let accepted = drafts.iter().zip(targets).take_while(|(draft, target)| **draft == **target as u32).count();
            for draft in &drafts[..accepted] {
                generated.insert(*draft);
            }
            return (accepted, targets[accepted] as u32);
        };
        sampling::accept(drafts, scores, VOCABULARY, sampling, generated, rng)
    }

    fn metric_add(&mut self, key: &str, amount: f64) {
        let value = self.last_metrics.get(key).and_then(Value::as_f64).unwrap_or(0.0) + amount;
        self.last_metrics.insert(key.into(), json!(value));
    }

    fn metric_count(&mut self, key: &str, amount: u64) {
        let value = self.last_metrics.get(key).and_then(Value::as_u64).unwrap_or(0) + amount;
        self.last_metrics.insert(key.into(), json!(value));
    }

    fn update_rate(&mut self) {
        let forwards = self.last_metrics["decode_forward_tokens"].as_u64().unwrap_or(0) as f64;
        let seconds = self.last_metrics["decode_seconds"].as_f64().unwrap_or(0.0);
        self.last_metrics.insert("decode_tokens_per_second".into(), json!(forwards / seconds));
    }

    /// `held` (a constraint that only suppresses): the head gives every row's whole logits, even greedy, whatever the sampling
    /// would otherwise read of them on the GPU (its argmax, its top k), for the mask on the host: 1 MB a row. (IQ3_XXS at --vram 8.5 with
    /// the drafter on CUDA, each order once: 0.7 ms a token more at the card's sampling, 2.8%; 1.3 ms greedy, 5%.)
    #[allow(clippy::too_many_arguments)]
    fn generate_speculative(&mut self, logits: &[f32], max_new: usize, sampling: &Sampling, rng: &mut Rng,
                            cancelled: &dyn Fn() -> bool, started: Instant, emit: &mut dyn FnMut(u32), mut held: Option<&mut Constraint>)
                            -> Result<()> {
        let mut generated = Generated::new(VOCABULARY);
        let mut token = constraint::pick(logits, sampling, &generated, rng, &mut held)?;
        generated.insert(token);
        if END_TOKENS.contains(&token) {
            return Ok(());
        }
        self.last_metrics.insert("ttft_seconds".into(), json!(started.elapsed().as_secs_f64()));
        for (key, value) in [("generated_tokens", 1), ("speculative_passes", 0), ("drafted_tokens", 0), ("accepted_tokens", 0)] {
            self.last_metrics.insert(key.into(), json!(value));
        }
        emit(token);
        let mut produced = 1usize;
        // The MTP layer pairs each hidden state with the token after it, at the hidden's position.
        let mut mtp_hidden = self.final_hidden.clone().ok_or("Straydog has no hidden state to draft from")?;
        let mut mtp_tokens = vec![token];
        let mut mtp_position = self.history.len() as u64 - 1;
        // A long match with the sequence so far is copied rather than drafted (an edit repeats its source): those drafts are
        // accepted nearly always, and the drafter's steps are saved. The MTP layer still takes the kept rows' pairs.
        let mut ngrams = Ngrams::new();
        ngrams.extend(&self.history);
        let mut policy = Policy::new(self.copy_drafts + 1);
        while produced < max_new {
            if cancelled() {
                break;
            }
            let pass_started = Instant::now();
            let limit = self.speculative.min(max_new - produced - 1).min(self.context.saturating_sub(self.history.len() + 1));
            // The token's PLE rows read while the drafter runs; the drafts' while layer 0 does.
            let mut clock = Instant::now();
            self.prefetch_ple(&self.history, &[token])?;
            self.phase("PLE prefetch", &mut clock);
            ngrams.push(token);
            let copy_limit = self.copy_drafts.min(max_new - produced - 1).min(self.context.saturating_sub(self.history.len() + 1));
            let (copy, matched) = ngrams.propose(copy_limit);
            let width = if matched >= LONG_MATCH { policy.width(matched).min(copy.len()) } else { 0 };
            // (a copied token the request refuses could only be rejected: no row is spent on it)
            let width = held.as_deref().map_or(width, |held| held.allowed_prefix(&copy[..width]));
            let copied = width > 0 && self.forced_rows.is_none();
            let suppression = held.as_deref().and_then(Constraint::suppression).cloned();
            let (drafts, proposed) = if let Some(forced) = self.forced_rows.as_ref() {
                let (pass, width) = (forced.passes.len(), forced.widths[forced.passes.len() % forced.widths.len()]);
                let most = (width - 1).min(max_new - produced - 1).min(self.context.saturating_sub(self.history.len() + 1));
                let mut drafts: Vec<u32> = forced.tokens.iter().skip(produced).take(most).copied().collect();
                if forced.reject > 0 && pass % forced.reject == 0 && !drafts.is_empty() {
                    let at = (pass / forced.reject) % drafts.len();
                    drafts[at] = (drafts[at] + 1) % VOCABULARY as u32;
                }
                self.ensure_kv(self.history.len() + most + 1)?;
                (drafts, Vec::new())
            } else if copied {
                self.draft(&mtp_hidden, &mtp_tokens, mtp_position, 0, sampling, rng, suppression)?;
                self.ensure_kv(self.history.len() + 1 + width)?;
                self.protect(self.history.len(), self.history.len() + 1 + width)?;
                (copy[..width].to_vec(), Vec::new())
            } else {
                self.draft(&mtp_hidden, &mtp_tokens, mtp_position, limit, sampling, rng, suppression)?
            };
            let rows: Vec<u32> = std::iter::once(token).chain(drafts.iter().copied()).collect();
            let mut clock = Instant::now();
            self.prefetch_ple(&self.history, &rows)?;
            self.phase("PLE prefetch", &mut clock);
            if self.history.len() + rows.len() > self.context {
                return Err("Straydog prompt plus output exceeds the allocated context".into());
            }
            let position = self.history.len() as u64;
            let top = if held.is_some() { 0 } else { sampling.top_candidates().unwrap_or(0) };
            let read_logits = self.forced_rows.as_ref().is_some_and(|forced| forced.read_logits);
            let (first, embeddings, hidden, targets, mut scores) = self.verify(&rows, sampling.needs_logits() || held.is_some() || read_logits, top)?;
            let mut clock = Instant::now();
            let (accepted, following) = match (held.as_deref_mut(), scores.as_mut()) {
                (Some(held), Some(Scores::Logits(logits))) => held.accept_rows(&drafts, &proposed, logits, VOCABULARY, sampling, &mut generated, rng)?,
                _ => Self::accept(&drafts, &proposed, &targets, scores.as_ref(), sampling, &mut generated, rng),
            };
            self.phase("accept", &mut clock);
            generated.insert(following);
            if copied {
                policy.record(drafts.len(), accepted);
            }
            ngrams.extend(&drafts[..accepted]);
            let mut emitted: Vec<u32> = drafts[..accepted].iter().copied().chain([following]).collect();
            let mut kept = accepted + 1;
            if let Some(index) = emitted.iter().position(|value| END_TOKENS.contains(value)) {
                kept = kept.min(index + 1);
                emitted.truncate(index);
            }
            self.commit(&rows, kept, &first, &embeddings)?;
            if let Some(forced) = self.forced_rows.as_mut() {
                let bytes = |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|value| value.to_le_bytes()).collect() };
                sha2::Digest::update(&mut forced.digest, bytes(&hidden[..kept * WIDTH]));
                if let Some(Scores::Logits(logits)) = scores.as_ref() {
                    sha2::Digest::update(&mut forced.logits_digest, bytes(&logits[..kept * VOCABULARY]));
                }
                forced.rejected += usize::from(accepted < drafts.len());
                forced.passes.push((rows.len(), 1e3 * pass_started.elapsed().as_secs_f64()));
            }
            self.final_hidden = Some(hidden[(kept - 1) * WIDTH..kept * WIDTH].to_vec());
            self.metric_count("speculative_passes", 1);
            self.metric_count("drafted_tokens", drafts.len() as u64);
            self.metric_count("accepted_tokens", accepted as u64);
            self.metric_add("decode_seconds", pass_started.elapsed().as_secs_f64());
            let seconds = pass_started.elapsed().as_secs_f64();
            if let Some(mean) = self.pass_seconds.get_mut(rows.len().wrapping_sub(1)) {
                // (a pass that waited for the drive says nothing of a row's cost)
                *mean = if *mean > 0.0 { *mean + (seconds.min(3.0 * *mean) - *mean) / 16.0 } else { seconds };
            }
            self.passes_with_draft = if rows.len() == 1 { 0 } else { self.passes_with_draft + 1 };
            if let Some(count) = self.pass_counts.get_mut(rows.len().wrapping_sub(1)) {
                *count += 1;
            }
            let hybrid = if self.partial && self.profile.is_some() {
                let info = self.group.info().unwrap_or_default();
                Some((info["wait_ns"].as_u64().unwrap_or(0) as f64 / 1e9, info["cpu_ns"].as_u64().unwrap_or(0) as f64 / 1e9))
            } else {
                None
            };
            if let Some(profile) = self.profile.as_mut() {
                if let Some((wait, cpu)) = hybrid {
                    let (last_wait, last_cpu) = std::mem::replace(&mut profile.hybrid, (wait, cpu));
                    if last_wait > 0.0 {
                        profile.pass.insert("export wait", wait - last_wait);
                        profile.pass.insert("CPU", cpu - last_cpu);
                    }
                }
                profile.end_pass(rows.len(), seconds);
            }
            for value in &emitted {
                produced += 1;
                self.metric_count("generated_tokens", 1);
                self.metric_count("decode_forward_tokens", 1);
                self.update_rate();
                emit(*value);
            }
            // The kept rows' pairs, for the next pass to catch up on (but an end token's row:
            // the next prompt pairs it, from the row before its own).
            let ended = emitted.len() < accepted + 1;
            token = following;
            mtp_hidden = hidden[..emitted.len().min(kept) * WIDTH].to_vec();
            mtp_tokens = emitted;
            mtp_position = position;
            if ended {
                break;
            }
        }
        // The pairs no pass caught up on, now: else the MTP layer's rows from the last pass up
        // to the sequence's last would keep what was there before (another sequence's, or a
        // draft's), which every later draft attends to. (The draft does nothing more at 0.)
        if !mtp_tokens.is_empty() && self.forced_rows.is_none() {
            self.draft(&mtp_hidden, &mtp_tokens, mtp_position, 0, sampling, rng, None)?;
        }
        Ok(())
    }

    /// Generate without truncation or hidden retry, resuming from the longest prefix already
    /// held. `turn`: the prompt ends with a user message rather than a tool result;
    /// `checkpoint` false marks a one-off request whose reply no later request continues.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(&mut self, tokens: &[u32], max_new: usize, sampling: &Sampling, seed: u64,
                    cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize), turn: bool, checkpoint: bool,
                    emit: &mut dyn FnMut(u32)) -> Result<()> {
        self.generate_constrained(tokens, max_new, sampling, seed, cancelled, progress, turn, checkpoint, emit, None)
    }

    /// [`Model::generate`] with the reply held to `constraint` ([`crate::constraint`]): every token
    /// chosen against it, so without MTP drafts, unless it only suppresses.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_constrained(&mut self, tokens: &[u32], max_new: usize, sampling: &Sampling, seed: u64,
                                cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize), turn: bool,
                                checkpoint: bool, emit: &mut dyn FnMut(u32), constraint: Option<&mut Constraint>)
                                -> Result<()> {
        if let Some(profile) = self.profile.as_mut() {
            *profile = Profile::default();
        }
        self.ple_counts = self.weights.row_counts(PLE_TABLE).unwrap_or_default();
        self.ple_wait = 0.0;
        if self.partial {
            self.hybrid_counts = self.group.info().unwrap_or_default();
            self.pass_counts = [0; 2];
            self.memory_before = crate::host::memory_note();
            self.read_aheads_before = self.cpu.as_ref().map_or((0, 0), CpuExperts::read_aheads);
        }
        let result = self.generate_request(tokens, max_new, sampling, seed, cancelled, progress, turn, checkpoint, emit, constraint);
        self.report_profile();
        result
    }

    /// The request's PLE row counts and wait, and with DS41_PROFILE its decode phases (milliseconds a
    /// generated token) and its passes by rows, into the metrics.
    fn report_profile(&mut self) {
        let (asked, read, blocks) = self.weights.row_counts(PLE_TABLE).map_or((0, 0, 0), |now| {
            (now.0 - self.ple_counts.0, now.1 - self.ple_counts.1, now.2 - self.ple_counts.2)
        });
        self.last_metrics.insert("ple_rows".into(), json!({"used": asked, "read": read, "blocks": blocks,
                                                            "wait_seconds": self.ple_wait}));
        let generated = self.last_metrics.get("generated_tokens").and_then(Value::as_u64).unwrap_or(0).max(1) as f64;
        if self.partial {
            // Where a partially resident model's request went, on every request: the log alone must say what held it up.
            let info = self.group.info().unwrap_or_default();
            let spent = |key: &str| (info[key].as_u64().unwrap_or(0) - self.hybrid_counts[key].as_u64().unwrap_or(0)) as f64;
            let memory = crate::host::memory_note();
            let faults = memory.zip(self.memory_before).map_or(0, |(now, before)| now.2.saturating_sub(before.2));
            let (taken, left) = self.cpu.as_ref().map_or((0, 0), CpuExperts::read_aheads);
            self.last_metrics.insert("partial_detail".into(), json!({
                "wait_ms": spent("wait_ns") / 1e6 / generated, "cpu_ms": spent("cpu_ns") / 1e6 / generated,
                "gpu_routes": spent("gpu_experts") / generated, "cpu_routes": spent("cpu_experts") / generated,
                "assisted_routes": spent("assisted_experts") / generated, "swaps": spent("swapped_experts") / generated,
                "slots": info["cached_experts"], "streamed": spent("streamed_experts"),
                "passes": self.pass_counts, "pass_ms": [1e3 * self.pass_seconds[0], 1e3 * self.pass_seconds[1]],
                "resident_gib": memory.map(|note| note.0 as f64 / (1u64 << 30) as f64),
                "available_gib": memory.map(|note| note.1 as f64 / (1u64 << 30) as f64),
                "page_faults": faults,
                "read_aheads": [taken - self.read_aheads_before.0, left - self.read_aheads_before.1],
                "graphs": info["devices"][0]["cached_graphs"], "replays": info["devices"][0]["graph_replays"].as_u64().unwrap_or(0)
                    .saturating_sub(self.hybrid_counts["devices"][0]["graph_replays"].as_u64().unwrap_or(0)),
            }));
        }
        let Some(profile) = self.profile.as_ref() else { return };
        let mut phases: Map<String, Value> = profile.phases.iter().map(|(name, seconds)| ((*name).to_string(), json!(1e3 * seconds / generated))).collect();
        if self.partial {
            // A generated token's routes by where they ran and its swaps, the slots they ran in, and
            // what of "step: layers" the host spent waiting for the routes and computing them.
            let info = self.group.info().unwrap_or_default();
            let spent = |key: &str| (info[key].as_u64().unwrap_or(0) - self.hybrid_counts[key].as_u64().unwrap_or(0)) as f64;
            for (name, key) in [("GPU routes", "gpu_experts"), ("CPU routes", "cpu_experts"), ("assisted routes", "assisted_experts"), ("swaps", "swapped_experts")] {
                phases.insert(format!("hybrid: {name} (count)"), json!(spent(key) / generated));
            }
            phases.insert("hybrid: slots (count)".into(), info["cached_experts"].clone());
            phases.insert("of step: layers: export wait".into(), json!(spent("wait_ns") / 1e6 / generated));
            phases.insert("of step: layers: CPU".into(), json!(spent("cpu_ns") / 1e6 / generated));
        }
        let rows: Map<String, Value> = profile.rows.iter().enumerate().filter(|(_, (passes, _, _))| *passes > 0)
            .map(|(rows, (passes, seconds, phases))| {
                let phases: Map<String, Value> = phases.iter().map(|(name, spent)| ((*name).to_string(), json!(1e3 * spent / *passes as f64))).collect();
                (rows.to_string(), json!({"passes": passes, "ms": 1e3 * seconds / *passes as f64, "phases": phases}))
            })
            .collect();
        self.last_metrics.insert("decode_profile".into(), Value::Object(phases));
        self.last_metrics.insert("decode_pass_rows".into(), Value::Object(rows));
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_request(&mut self, tokens: &[u32], max_new: usize, sampling: &Sampling, seed: u64,
                        cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize), turn: bool,
                        checkpoint: bool, emit: &mut dyn FnMut(u32), mut constraint: Option<&mut Constraint>)
                        -> Result<()> {
        let started = Instant::now();
        self.last_metrics = Map::new();
        if tokens.is_empty() || max_new == 0 || max_new > self.context || tokens.len() + max_new > self.context {
            return Err("Straydog prompt plus output exceeds the allocated context".into());
        }
        sampling.check()?;
        self.weights.check_unchanged()?;
        self.group.ensure_slots()?;
        let parking = Instant::now();
        let parked = self.park_or_restore(tokens)?;
        let park_seconds = parking.elapsed().as_secs_f64();
        let (reused, resumed) = self.resume(tokens)?;
        self.last_metrics = json!({
            "cached_tokens": reused, "resumed_from": resumed, "prefill_tokens": tokens.len() - reused,
            "parked": parked, "park_seconds": park_seconds, "park_copied_positions": 0,
            "decode_forward_tokens": 0, "decode_seconds": 0.0, "decode_tokens_per_second": null,
            "generated_tokens": 0, "ttft_seconds": null,
        })
        .as_object()
        .unwrap()
        .clone();
        // The parked sequence's positions this request may write, copied out before anything
        // runs: from where it resumes (the MTP layer's pair before that) to its last possible
        // output (the rows of a verification later rejected among them).
        self.protect(reused.saturating_sub(usize::from(self.speculative > 0)), tokens.len() + max_new - 1)?;
        if let Some(parked) = &self.parked {
            let borrowed = parked.covered.map_or(0, |(_, high)| parked.tokens.len() - high);
            self.last_metrics.insert("park_borrowed_positions".into(), json!(borrowed));
        }
        let prefill_started = Instant::now();
        let plan = self.checkpoint_plan(tokens, reused, turn, checkpoint);
        let pipelined = self.pipelined_prefill(tokens.len() - reused)?;
        if pipelined {
            self.group.prefill_defer(true)?;
            self.deferred = true;
            self.mtp_kept = 0;
        }
        let prefilled = self.prefill(tokens, reused, &plan, cancelled, progress);
        if pipelined {
            self.deferred = false;
            self.mtp_kept = 0;
            let settled = self.group.prefill_settle();
            if prefilled.is_ok() {
                settled?;
            }
        }
        let Some(output) = prefilled? else { return Ok(()) };
        let elapsed = prefill_started.elapsed().as_secs_f64();
        self.last_metrics.insert("prefill_seconds".into(), json!(elapsed));
        self.last_metrics.insert("prefill_tokens_per_second".into(), json!((tokens.len() - reused) as f64 / elapsed));
        if self.partial {
            // Re-pick cached experts from this prompt's routing before decoding.
            let rebalance = Instant::now();
            self.group.rebalance(0)?;
            self.last_metrics.insert("rebalance_seconds".into(), json!(rebalance.elapsed().as_secs_f64()));
        }
        // (the last batch's delta was of where it began: the reply is past all of its images)
        self.set_rope_delta(self.history.len())?;
        let mut rng = Rng::new(seed);
        let Some(mut logits) = output else { return Ok(()) };
        if self.speculative > 0 && !constraint.as_deref().is_some_and(Constraint::chooses_every_token) {
            return self.generate_speculative(&logits, max_new, sampling, &mut rng, cancelled, started, emit, constraint);
        }
        let mut generated = Generated::new(VOCABULARY);
        for index in 0..max_new {
            if cancelled() {
                return Ok(());
            }
            let token = constraint::pick(&logits, sampling, &generated, &mut rng, &mut constraint)?;
            generated.insert(token);
            if END_TOKENS.contains(&token) {
                return Ok(());
            }
            if self.last_metrics["ttft_seconds"].is_null() {
                self.last_metrics.insert("ttft_seconds".into(), json!(started.elapsed().as_secs_f64()));
            }
            self.metric_count("generated_tokens", 1);
            emit(token);
            if index + 1 < max_new {
                let decode_started = Instant::now();
                let Some(next) = self.forward(&[token], true, &|| false)? else { return Ok(()) };
                logits = next.unwrap();
                let seconds = decode_started.elapsed().as_secs_f64();
                self.metric_add("decode_seconds", seconds);
                if let Some(profile) = self.profile.as_mut() {
                    profile.rows[1].1 += seconds;
                }
                // (for the request's line: a pass without drafts is a lone row)
                self.pass_counts[0] += 1;
                self.pass_seconds[0] = if self.pass_seconds[0] > 0.0 { self.pass_seconds[0] + (seconds.min(3.0 * self.pass_seconds[0]) - self.pass_seconds[0]) / 16.0 } else { seconds };
                self.metric_count("decode_forward_tokens", 1);
                self.update_rate();
            }
        }
        Ok(())
    }
}

/// The first layer of the second GPU when experts are split, as far as the placement allows:
/// a pipelined prompt runs both GPUs' layers at once, so the one owning more (attention and
/// dense work on top of its expert half) sets the pace. Balancing bytes gives the first GPU the
/// layers the second's head and MTP layer make room for (33 of 48 at IQ3_S on two R9700s);
/// 25 balanced their work there (20000 tokens 10.2 -> 9.0 s, decode unchanged).
const EVEN_SPLIT: usize = 25;

/// The layer split and its full-residency placement (see [`resident_placement`]): the split
/// balancing bytes moves toward [`EVEN_SPLIT`] while the placement stays as good.
fn working_split(weights: &Weights, group: &mut Group, context: u64, prefill: usize, drafting: bool, vram_kv: bool)
                 -> Result<Option<(bool, bool, usize)>> {
    let placed = resident_placement(weights, group, context, prefill, drafting, vram_kv)?;
    let (Some(baseline), Some(balanced)) = (placed, group.split()) else { return Ok(placed) };
    if !group.expert_parallel()? || balanced <= EVEN_SPLIT {
        return Ok(placed);
    }
    let as_good = |(keep, host, batch): (bool, bool, usize)| keep >= baseline.0 && (!host || baseline.1) && batch >= baseline.2;
    for split in EVEN_SPLIT..balanced {
        group.set_split(split)?;
        if let Some(placement) = resident_placement(weights, group, context, prefill, drafting, vram_kv)?.filter(|p| as_good(*p)) {
            return Ok(Some(placement));
        }
    }
    group.set_split(balanced)?;
    Ok(placed)
}

/// (drafter kept, K/V in host memory, prefill batch) that lets every weight live on the GPUs,
/// if any. Full residency (and with it MTP) is worth more than K/V in VRAM; a placement
/// keeping the drafter wins over one that drops it; the configured batch and VRAM K/V first.
fn resident_placement(weights: &Weights, group: &Group, context: u64, prefill: usize, drafting: bool, vram_kv: bool)
                      -> Result<Option<(bool, bool, usize)>> {
    let mut batches = vec![prefill];
    while *batches.last().unwrap() > 1024 {
        batches.push((batches.last().unwrap() / 2).max(1024));
    }
    let mut candidates: Vec<(bool, usize)> = if vram_kv { vec![(false, prefill)] } else { Vec::new() };
    candidates.extend(batches.iter().map(|batch| (true, *batch)));
    let keeps: &[bool] = if drafting { &[true, false] } else { &[false] };
    for keep in keeps {
        for (host, batch) in &candidates {
            let planned = group.planned_state_bytes(context, *batch as u64, drafting, *host)?;
            let requirements = weights.resident_requirements(*keep && weights.drafter);
            if group.can_pin_model(&requirements, Some(&planned))? {
                return Ok(Some((*keep, *host, *batch)));
            }
        }
    }
    Ok(None)
}

fn validate_geometry(weights: &Weights) -> Result<()> {
    let mut expected: Vec<(String, Vec<u64>)> = vec![
        ("token_embd.weight".into(), vec![2560, 248320]),
        ("output.weight".into(), vec![2560, 248320]),
        ("per_layer_token_embd.weight".into(), vec![160, 320001536]),
        ("output_hc_norm.weight".into(), vec![10240]),
        ("output_hc_down.weight".into(), vec![10240, 320]),
        ("output_hc_up.weight".into(), vec![320, 10240]),
    ];
    for layer in 0..48 {
        let prefix = format!("blk.{layer}.");
        for operation in ["hc_attn", "hc_ffn"] {
            expected.push((format!("{prefix}{operation}_norm.weight"), vec![10240]));
            expected.push((format!("{prefix}{operation}_down.weight"), vec![10240, 320]));
            expected.push((format!("{prefix}{operation}_up.weight"), vec![320, 10240]));
            expected.push((format!("{prefix}{operation}_inject.weight"), vec![10240, 4]));
        }
        expected.push((format!("{prefix}ffn_gate_inp.weight"), vec![2560, 512]));
        expected.push((format!("{prefix}ffn_gate_inp_shexp.weight"), vec![2560]));
        expected.push((format!("{prefix}ffn_gate_exps.weight"), vec![2560, 640, 512]));
        expected.push((format!("{prefix}ffn_up_exps.weight"), vec![2560, 640, 512]));
        expected.push((format!("{prefix}ffn_down_exps.weight"), vec![640, 2560, 512]));
        let shapes: &[(&str, [u64; 2])] = if layer % 4 != 3 {
            &[("attn_qkv.weight", [2560, 10240]), ("attn_gate.weight", [2560, 6144]), ("ssm_conv1d.weight", [4, 10240]),
              ("ssm_out.weight", [6144, 2560])]
        } else {
            &[("attn_q.weight", [2560, 12288]), ("attn_k.weight", [2560, 512]), ("attn_v.weight", [2560, 512]),
              ("attn_output.weight", [6144, 2560])]
        };
        for (name, shape) in shapes {
            expected.push((format!("{prefix}{name}"), shape.to_vec()));
        }
    }
    for (name, shape) in expected {
        if weights.shape(&name)? != shape.as_slice() {
            return Err(format!("Straydog model geometry mismatch: {name}"));
        }
    }
    if weights.integer_arrays.get("qwen4exp.ple.layers") != Some(&vec![1]) {
        return Err("Straydog model PLE placement mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod same_text_tests {
    use super::same_text;

    fn spelled(token: u32) -> Option<&'static [u8]> {
        match token {
            1 => Some(b"ab"),
            2 => Some(b"a"),
            3 => Some(b"b"),
            4 => Some(b"c"),
            5 => Some(b"abc"),
            _ => None,
        }
    }

    #[test]
    fn a_reply_tokenized_anew_keeps_the_tokens_it_was_written_in() {
        // (held "ab" where the prompt has "a" "b", and "a" "b" "c" where it has "abc")
        assert_eq!(same_text(&[4, 2, 3, 4, 5, 4], &[4, 1, 4, 2, 3, 4], &spelled), [4, 1, 4, 2, 3, 4, 4]);
    }

    #[test]
    fn other_text_and_special_tokens_part_the_two() {
        assert_eq!(same_text(&[4, 2, 4], &[4, 1, 4], &spelled), [4, 2, 4]);
        // (9 spells nothing: it is a special token)
        assert_eq!(same_text(&[9, 4], &[2, 4], &spelled), [9, 4]);
        assert_eq!(same_text(&[4, 2], &[4, 1], &spelled), [4, 2]);
    }

    #[test]
    fn what_follows_the_held_sequence_stays() {
        assert_eq!(same_text(&[1, 4, 5], &[1], &spelled), [1, 4, 5]);
        assert_eq!(same_text(&[2, 3, 5], &[1], &spelled), [1, 5]);
    }
}
