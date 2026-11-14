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
use std::sync::Arc;
use std::time::Instant;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::assist;
use crate::banks::{BankLocks, Copies, GaveBack, LockMode, Locked, PinnedMemory, Refusal};
use crate::constraint::{self, Constraint, Suppression};
use crate::drafts::{Ngrams, Policy, LONG_MATCH};
use crate::gpu::{expert_bytes, Columns, CpuExperts, Data, Operand, Piece, Result, Scores, HIDDEN, HOST_NO_MEMORY, WIDTH};
use crate::group::{row_count, Group, Positions};
use crate::qwen_vision::{self, ImageSpan, IMAGE_PAD};
use crate::routes::{self, Banks, Hints, Routes};
use crate::sampling::{self, Generated, Sampling};
use crate::vision::IMAGE_TOKEN;
use crate::weights::{ngram_rows, Progress, Weights, CONTEXT_LIMIT};
use crate::width;

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
/// A draft's vocabulary with the head split across two GPUs (a fully resident model's, and a partially resident one's on CUDA): the head's first 100000 rows (English, code and the common CJK
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
    let state = attention_layers(drafting) * (context * 4 * crate::gpu::KV_ROW_FLOATS_F16 + (context / 4 + 1) * 128 + 8 * 128) * 4;
    state + 36 * (48 * 128 * 128 + 4 * 10240) * 4 + 10 * 10240 * 4
}

/// The bytes one more token of context adds to [`context_state_bytes`]: every attention layer's K/V position and its share of
/// the pooled indexer keys (28,288 with the MTP layer: thirteen layers of 2,048 and 128). What a bundle's footprint counts of the
/// context ([`crate::bundle::footprint`]): the formula's own, so that it follows the K/V as it is held.
pub fn state_per_token(drafting: bool) -> u64 {
    const CONTEXT: u64 = 262_144;
    (context_state_bytes(2 * CONTEXT, drafting) - context_state_bytes(CONTEXT, drafting)) / CONTEXT
}

/// A whole attention layer's K/V bytes a position in FP16: the conservative estimate in
/// [`context_state_bytes`], also subtracted from it when the K/V stays in VRAM.
const KV_POSITION_BYTES: u64 = 4 * crate::gpu::KV_ROW_FLOATS_F16 * 4;

/// A whole attention layer's K/V bytes a position as it is held ([`crate::gpu::KvFormat::selected`]): what its memory takes.
fn kv_position_bytes() -> u64 {
    4 * crate::gpu::KvFormat::selected().row_bytes()
}

/// The layers with QSA attention: twelve, and the MTP layer when drafting.
fn attention_layers(drafting: bool) -> u64 {
    12 + u64::from(drafting)
}

/// Whether the whole context's K/V, in its held format, takes at most a quarter of the
/// GPUs' total budget. Weights, workspaces and each GPU's share are checked by placement.
fn kv_fits_budget(context: u64, drafting: bool, format: crate::gpu::KvFormat, budget: u64) -> bool {
    u128::from(attention_layers(drafting)) * u128::from(context) * 4 * u128::from(format.row_bytes()) <= u128::from(budget / 4)
}

/// The end of a K/V growth's error when page-locking found no memory (`sd_kv_grow`): every state
/// is sound, and a shorter sequence can still run.
pub const KV_NO_MEMORY: &str = "not enough host memory for the K/V of this sequence";

/// The end of a K/V growth's error when the VRAM had no room for the longer K/V beside what the device holds (`sd_kv_grow`): as [`KV_NO_MEMORY`], every
/// state is sound, and a shorter sequence can still run.
pub const KV_NO_VRAM: &str = "not enough VRAM for the K/V of this sequence";

/// The model is intact, but its fixed allocations leave too little GPU memory for a request.
pub const VRAM_NO_MEMORY: &str = "not enough GPU memory to serve this request";

/// Combine completed device work without hiding a device fault behind a recovered wait.
/// Both calls have already finished, so choosing an error never skips another device's cleanup.
fn finish_device_work<T>(work: Result<T>, finished: Result<()>) -> Result<T> {
    match (work, finished) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(first), Err(second)) if first.ends_with(crate::gpu::HYBRID_WAIT_RECOVERED) => Err(second),
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
    }
}

/// The end of the error of what an image asked and the devices could not give, or of a prompt whose image rows would run in a
/// pass of one row, which no pass of its kind takes (the chat template cannot render one: its images are followed by text enough).
/// The devices are sound: the request fails, and the engine resets.
pub const IMAGE_REFUSED: &str = "(a request with an image, refused)";

/// The error of a pass that the warm-up behind "ready" stopped in the middle, for a request that came ([`Model::set_warm_stop`]): no device is
/// left in a state the warm-up's reset does not take back.
pub const WARM_STOPPED: &str = "Straydog warm-up stopped for a request";

/// Whether `error` is [`WARM_STOPPED`] (it may be joined to the error of restoring the expert slots after it).
pub fn is_warm_stop(error: &str) -> bool {
    error.contains(WARM_STOPPED)
}

/// The chooser of a model of `most` drafts at most that begins at `saved`, the width an earlier run chose, in place of `current`: None where `current`
/// stays. That is where the drafts are `held`, `saved` is not a width the model offers (1 to `most`), the chooser has made its first choice, or it has
/// begun a probe: what it chose or measured is not thrown away. The passes it has seen before that are the warm-up's two (each of its prompts
/// decodes one token through the chooser, as cold passes of one row with no draft: nothing in its windows) and a request's that came first.
fn chooser_started_at(current: &width::Chooser, most: usize, held: bool, saved: usize) -> Option<width::Chooser> {
    let (_, probed) = current.passes();
    (!held && (1..=most).contains(&saved) && !current.decided() && probed == 0).then(|| width::Chooser::new(most, saved))
}

/// The rows a layer-order batch needs to beat the ordinary path when the banks fit in RAM, on Windows.
const LAYER_PREFILL_FIT_ROWS: u64 = 512;
/// The rows it needs where they do not: its MoE takes time inversely with them (a 3.8K follow-up under memory pressure on the RTX 5070 Ti:
/// 12.7 s at 256 rows against the ordinary path's 16.0 s; about even at 128 rows by the 32.6K fits, losing below).
const LAYER_PREFILL_SHORT_ROWS: u64 = 128;

/// Whether the ordinary path's transfer of the routed experts costs enough every batch for layer order to win where the banks fit:
/// Windows (IQ2_XS on an RTX 5070 Ti at 5.5 GiB, banks in RAM: 8.2K new 17.5 s at 512 rows against 25.4 s; 32.6K at 256 rows 73.7 s
/// both; 3.8K at 128 rows 16.1 s against 10.0 s). Linux reads them faster (the same banks and budget on a Radeon AI PRO R9700: 14.9K new 22.4 s
/// at 512 rows against 23.6 s, 3.7K 10.5 s against 6.4 s), and keeps the ordinary path.
const LAYER_PREFILL_FITS: bool = cfg!(windows);

/// The memory the banks may hold for a prefill: what RAM leaves beside `pinned` and [`crate::host::left_free`], bounded by what the system
/// has for them now (`now`, from [`crate::host::bank_room_now`]) when other programs have taken it. None where RAM is unknown.
fn layer_prefill_room(memory: u64, pinned: u64, now: Option<u64>) -> Option<u64> {
    if memory == 0 { return None; }
    let usable = memory.saturating_sub(pinned).saturating_sub(crate::host::left_free(memory));
    Some(now.map_or(usable, |now| usable.min(now)))
}

/// Layer ordering pays for loading every routed bank once when the ordinary path would reread banks that do not fit in RAM (`room`, from
/// [`layer_prefill_room`]): under memory pressure, the ordinary path reads them from the drive again every batch (32.6K new on the
/// RTX 5070 Ti with 4.1 GiB available: 161.8 s, layer order 73.0 s at 256 rows). Large shortages amortize it over two ordinary batches; a
/// slight shortage needs three full maximum-sized batches, and either [`LAYER_PREFILL_SHORT_ROWS`] (`rows`, from
/// [`Group::layer_prefill_rows`]). Where the banks fit, `fits` (see [`LAYER_PREFILL_FITS`]) takes layer order from three ordinary batches
/// when it can take [`LAYER_PREFILL_FIT_ROWS`].
fn layer_prefill_needed(new_tokens: usize, batch: usize, ordinary: usize, banks: u64, room: Option<u64>, rows: u64, fits: bool) -> bool {
    if fits && rows >= LAYER_PREFILL_FIT_ROWS && new_tokens / ordinary.max(batch).max(1) >= 3 {
        return true;
    }
    let Some(room) = room else { return false };
    if banks <= room || rows < LAYER_PREFILL_SHORT_ROWS { return false; }
    if u128::from(banks) * 10 >= u128::from(room) * 13 {
        new_tokens > ordinary.max(batch).max(1)
    } else {
        new_tokens / batch.max(crate::gpu::LONG_PREFILL_ROWS as usize) >= 3
    }
}

/// The spans a resident prefill runs: the first layer alone, then each device's decoder layers (`boundaries`, from layer 1) in spans of at most
/// `width` layers.
fn prefill_spans(boundaries: &[(usize, usize)], width: usize) -> Vec<(usize, usize)> {
    let mut spans = vec![(0, 1)];
    for (begin, end) in boundaries {
        spans.extend((*begin..*end).step_by(width).map(|first| (first, (first + width).min(*end))));
    }
    spans
}

/// The PLE table's row cache of a partially resident model with less memory than twice its banks.
pub const SCARCE_ROW_CACHE: u64 = 1 << 30;
/// The commit a K/V growth leaves the machine.
const KV_COMMIT_MARGIN: u64 = 2 << 30;
/// What a copy of a conversation's state into host memory leaves the machine: the stacks, heaps and page-locked buffers the rest of the engine
/// asks for as it goes are charged to the same commit, and a process that finds it gone ends there, whatever it was doing (Windows: STATUS_NO_MEMORY).
const COPY_COMMIT_MARGIN: u64 = KV_COMMIT_MARGIN;
/// The positions a partially resident model's K/V starts with, in host memory or in VRAM: page-locked, 12 GiB at the
/// whole context take from a small machine what the expert banks need, and few sequences reach it; in VRAM, what a short conversation does not
/// use is expert slots lost (Qwen3.8-Flash-Next at 262144: 3.45 GiB, some 1,800 slots), which a growth takes back from them as a sequence needs it.
pub const STARTING_KV: usize = 32768;

/// The positions the K/V of a model of `context` starts with: [`STARTING_KV`], or STRAYDOG_KV_START's (whole thousands of positions; 0, or the
/// context or more, keeps the whole context from the start: what a growth is measured against).
fn starting_kv(context: usize) -> usize {
    starting_kv_named(std::env::var("STRAYDOG_KV_START").ok().as_deref(), context)
}

fn starting_kv_named(value: Option<&str>, context: usize) -> usize {
    match value.and_then(|positions| positions.trim().parse::<usize>().ok()) {
        Some(0) => context,
        Some(positions) => positions.next_multiple_of(4096).min(context),
        None => STARTING_KV.min(context),
    }
}

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

/// The share of the routed experts the slots hold at load from which a partially resident model on HIP starts with two drafts a pass, not one
/// ([`width::Chooser`] takes it from there, by what its passes cost: the starting width is where today's rule was measured right). Routing is
/// uneven: IQ3_S on an R9700, the slots of `--vram 24` (30% at load, 39% once the K/V has settled) serve 87% of the routes, and the second draft
/// costs less CPU. Two drafts against one, same seeds, tok/s of a second and third request: `--vram 24` 69.5, 70.5 / 67.7, 61.4; 16 (17%) 58.7,
/// 57.3 / 57.4, 56.6; 12 (under 15%) 50.1, 49.1 / 50.7, 48.0.
const TWO_DRAFTS_START: f64 = 0.15;
/// The experts' slots a partial placement keeps out of its pool for staging, its VRAM margin (`Group::create_expert_slots`), and the room its gate
/// asks beyond them (`Model::prepare_hybrid_moe`).
pub(crate) const STAGING_SLOTS: u64 = 32;
/// CUDA starts with missed routes on the CPU. Its tuner changes this level only after
/// the same candidate saves at least 3% in two consecutive rounds. STRAYDOG_ASSIST=N holds N instead.
const CUDA_ASSIST: u32 = 0;

/// The staging slots a wide partial read (batches of [`PRESTAGE_BATCH`] rows and more, on two NVIDIA GPUs) adds for the experts of the layer under
/// way that its slots do not hold: they come over the bus beside the layer's own work (gpu/hybrid_slots.hpp, prestage). STRAYDOG_PRESTAGE
/// names another count (0: none), to measure.
const PRESTAGE_SLOTS: u64 = 192;
const PRESTAGE_BATCH: usize = 2048;

fn prestage_slots() -> u64 {
    std::env::var("STRAYDOG_PRESTAGE").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|slots| *slots <= 512).unwrap_or(PRESTAGE_SLOTS)
}
pub(crate) const SLOTS_MARGIN: u64 = 64 << 20;
const STAGING_GATE: u64 = 128 << 20;

/// Target dense weights and the staging room a partially resident placement must keep.
fn partial_requirements(mut dense: Vec<(String, u64)>, kinds: &[[u32; 3]], prestage: u64) -> Vec<(String, u64)> {
    let staging = (STAGING_SLOTS + prestage) * kinds.iter().map(|kinds| expert_bytes(*kinds)).max().unwrap() + STAGING_GATE;
    dense.push(("@hybrid:staging".into(), staging));
    dense
}

/// The requested drafter's dense weights and verification room beside the target requirements.
fn partial_drafter_requirements(mut target: Vec<(String, u64)>, drafter: Vec<(String, u64)>) -> Vec<(String, u64)> {
    // Its routed experts use the same slots as the target's, not a separate resident bank.
    target.extend(drafter.into_iter().filter(|(name, _)| !name.ends_with("_exps.weight")));
    target.push(("@verify:buffers".into(), VERIFY_BYTES));
    target.push(("@hybrid:slots".into(), DRAFTER_SLOT_ROOM));
    target
}

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
/// Refilling multiple GPUs' expert caches costs several seconds; shorter prompts keep their
/// original workspaces instead of paying that cost to borrow slots.
const MULTI_GPU_LONG_PREFILL: usize = 65_536;
/// New tokens from which a read's DeltaNet layers scan in chunks of tokens (gpu/gated_delta_chunk.hpp) where the token-by-token scan
/// ran: a read this long spends a tenth of its time in the recurrence, and the chunked scan takes about 2.5 s off a 242K one. Its sums are in
/// another order (README, Numerics says how far that moves the outputs). Shorter reads (a conversation's turns, the pinned tests'
/// prompts) keep the scan, so their bits stay as they were.
pub const CHUNKED_SCAN_READ: usize = 16_384;
/// A partially resident model's passes in a row with a draft: one alone then keeps [`Model::partial_drawn_cut`]'s measure current.
const PASSES_BETWEEN_LONE_ROWS: u32 = 64;

/// Expert bank bytes to page-lock given the host memory (0 if unknown): all of them with
/// memory twice their size; with at least 1.25 times, half the memory less the page-locked
/// sequence state; else none (the unlocked layers and the PLE table would thrash).
pub fn lock_budget(memory: u64, banks: u64, pinned_state: u64) -> u64 {
    // (development: STRAYDOG_LOCK_BUDGET=all, or a number of GiB, in place of the rule -- to measure what the layers the rule leaves
    // unlocked cost: a layer that is not page-locked has no route staged to the GPU and no layer launched ahead of its CPU experts)
    if let Ok(asked) = std::env::var("STRAYDOG_LOCK_BUDGET") {
        if asked == "all" {
            return banks;
        }
        if let Ok(gib) = asked.parse::<u64>() {
            return (gib << 30).min(banks);
        }
    }
    if memory == 0 || 4 * memory < 5 * banks {
        return 0;
    }
    if memory >= 2 * banks {
        return banks;
    }
    (memory / 2).saturating_sub(pinned_state)
}

/// The part of a partially resident request's "other" (the request line's ms a token apart from the GPU wait and the CPU) that a decode
/// phase is, as the line names it. A lone row and a pass of drafts share one short list: the draft, the head, the first layer together
/// with the PLE, the acceptance and the commit, and the layers.
fn decode_part(phase: &str) -> Option<&'static str> {
    Some(match phase {
        "draft: embeddings" | "draft: catch-up and first draft" | "draft: next draft" | "draft: forced steps" => "draft",
        "step: head" | "verify: head" => "head",
        "step: first layer" | "step: PLE rows wait" | "step: PLE layer"
            | "verify: protect, PLE prefetch, embeddings" | "verify: first layer" | "verify: PLE rows wait" | "verify: PLE layer"
            | "PLE prefetch" => "first/PLE",
        "accept" | "commit: DeltaNet replay" | "commit: PLE replay" => "accept",
        "step: layers (first device)" | "step: layers (second device)"
            | "verify: layers (queued)" | "verify: layers (first device)" | "verify: layers (second device)" => "layers",
        _ => return None,
    })
}

/// Milliseconds a generated token of each [`decode_part`], from the phases already timed. `generated` is at least 1.
fn decode_steps(phases: &BTreeMap<&'static str, f64>, generated: f64) -> BTreeMap<&'static str, f64> {
    let mut steps = BTreeMap::new();
    for (name, seconds) in phases {
        if let Some(part) = decode_part(name) {
            *steps.entry(part).or_default() += 1e3 * seconds / generated;
        }
    }
    steps
}

/// STRAYDOG_FRONT_CHAIN: 0 keeps a partially resident verification's first layer and PLE layer going through the host; `check` chains
/// them and also runs the PLE layer again as the host path does, counting the floats that differ (reported with the row proofs).
fn front_chain_mode() -> Option<&'static str> {
    static MODE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    MODE.get_or_init(|| std::env::var("STRAYDOG_FRONT_CHAIN").ok()).as_deref()
}

fn front_chain_wanted() -> bool {
    front_chain_mode() != Some("0")
}

fn front_chain_checked() -> bool {
    front_chain_mode() == Some("check")
}

/// [`Model::partial_drawn_cut`] of the draft at position `at` (1 the first) from the running means of a pass's seconds by rows (0: none yet).
fn drawn_cut(pass_seconds: &[f64], at: usize) -> f32 {
    let seconds = |rows: usize| pass_seconds.get(rows - 1).copied().filter(|seconds| *seconds > 0.0);
    match (seconds(1), seconds(at), seconds(at + 1)) {
        (Some(one), Some(before), Some(after)) => (2.75 * ((after - before) / one) - 1.125).clamp(0.0, 0.9) as f32,
        _ if at > 1 => drawn_cut(pass_seconds, at - 1),
        _ => PARTIAL_DRAFT_CUT,
    }
}

/// Seconds per phase of a request's decode, and of its passes by rows (count, seconds, phases): every
/// phase with DS41_PROFILE set, else a partially resident model's [`decode_part`]s alone. The host's
/// clock alone: every call a phase covers returns after its device work, so no extra synchronization
/// is added.
#[derive(Default)]
struct Profile {
    /// Every phase is timed and reported (DS41_PROFILE).
    every: bool,
    phases: BTreeMap<&'static str, f64>,
    /// The phases of the pass under way, until its rows are known.
    pass: BTreeMap<&'static str, f64>,
    rows: [(u64, f64, BTreeMap<&'static str, f64>); VERIFY_ROWS + 1],
    /// A partially resident model's export wait and CPU seconds so far, for what each pass adds.
    hybrid: (f64, f64),
}

impl Profile {
    fn new(every: bool) -> Profile {
        Profile { every, ..Default::default() }
    }

    /// `seconds` of the pass under way went to the phase `name`: kept if every phase is, else if the request line names it.
    fn add(&mut self, name: &'static str, seconds: f64) {
        if self.every || decode_part(name).is_some() {
            *self.pass.entry(name).or_default() += seconds;
        }
    }

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

/// The assist levels of the next decode passes of one row, the last going on, and where each pass's routes went (see [`Model::assist_script`]).
struct AssistScript {
    next: std::collections::VecDeque<u32>,
    /// The level in force, and the routes assisted, on the GPU and on the CPU by the end of the pass before.
    level: u32,
    routed: [u64; 3],
    trace: Vec<(u32, [u64; 3], f64)>,
}

/// The rows a layer-order pass takes through every layer at most: the prompt's hidden rows it holds (40 KB a row, F32), and the passes a
/// prompt takes, each of which reads every layer's banks (a 250K prompt on Windows with 64 GB: 16 passes read some 36 GB of banks from the
/// drive each). STRAYDOG_LAYER_PREFILL_ROWS changes it, to measure ([`layer_prefill_rows`]).
const LAYER_PREFILL_ROWS: usize = 16384;
const LAYER_PREFILL_MIN: usize = 512;

/// The rows a layer-order pass takes: [`LAYER_PREFILL_ROWS`], or STRAYDOG_LAYER_PREFILL_ROWS when the process starts (from
/// [`LAYER_PREFILL_MIN`] to the context's limit; a value outside them, or not a number, keeps the default).
fn layer_prefill_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| layer_prefill_rows_named(std::env::var("STRAYDOG_LAYER_PREFILL_ROWS").ok().as_deref()))
}

/// Where a layer-order pass loads a layer's banks from.
#[derive(Debug, PartialEq, Eq)]
enum BankSource {
    /// The reader's checked copy, read ahead.
    Copy,
    /// The mapping, whose pages the layer's registration keeps checked (held for the load).
    Registered,
    /// A checked copy read now: the reader passed the layer by while it was registered, and its registration was given up since.
    ReadNow,
    /// The mapping, as with no reader at all.
    Mapping,
}

/// [`BankSource`] of a layer: whether the reader `copied` it ahead, whether the model's marks hold it `registered` now, and whether there is a
/// `reader`. A layer the reader skipped for its mark and that is no longer registered is not read from the mapping it no longer keeps.
fn bank_source(copied: bool, registered: bool, reader: bool) -> BankSource {
    match (copied, registered, reader) {
        (true, _, _) => BankSource::Copy,
        (false, true, _) => BankSource::Registered,
        (false, false, true) => BankSource::ReadNow,
        (false, false, false) => BankSource::Mapping,
    }
}

/// Whether a prompt is going through the layers in layer order now: the locking behind "ready" may take its turns meanwhile
/// (STRAYDOG_LOCK_DURING_PREFILL), as the prompt reads every layer's banks and no CPU expert of a generating request is in the way.
pub static LAYER_PREFILL_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets [`LAYER_PREFILL_ACTIVE`] for its life.
struct LayerPrefillActive;

impl LayerPrefillActive {
    fn begin() -> LayerPrefillActive {
        LAYER_PREFILL_ACTIVE.store(true, std::sync::atomic::Ordering::SeqCst);
        LayerPrefillActive
    }
}

impl Drop for LayerPrefillActive {
    fn drop(&mut self) {
        LAYER_PREFILL_ACTIVE.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

/// STRAYDOG_LAYER_ROWS_VRAM: anything but 0 (or unset) holds a layer-order pass's rows in VRAM where they fit.
fn layer_rows_vram_named(value: Option<&str>) -> bool {
    value.map_or(true, |value| value.trim() != "0")
}

/// STRAYDOG_HC=blas: every read keeps its HyperConnections' up projection (hipblasGemmEx) and finish as two kernels (anything else: one kernel
/// from [`CHUNKED_SCAN_READ`] new tokens, gpu/hc_fused.hpp).
fn hc_fused_from_named(value: Option<&str>) -> usize {
    if value.is_some_and(|value| value.trim() == "blas") { usize::MAX } else { CHUNKED_SCAN_READ }
}

/// STRAYDOG_DELTA_SCAN=token: every read scans its DeltaNet layers token by token (anything else: in chunks from [`CHUNKED_SCAN_READ`] new tokens).
fn chunked_scan_from_named(value: Option<&str>) -> usize {
    if value.is_some_and(|value| value.trim() == "token") { usize::MAX } else { CHUNKED_SCAN_READ }
}

fn layer_prefill_rows_named(value: Option<&str>) -> usize {
    match value.map(|value| value.trim().replace('_', "").parse::<usize>()) {
        None => LAYER_PREFILL_ROWS,
        Some(Ok(rows)) if (LAYER_PREFILL_MIN..=CONTEXT_LIMIT).contains(&rows) => rows,
        Some(_) => {
            eprintln!("WARN STRAYDOG_LAYER_PREFILL_ROWS must be a count from {LAYER_PREFILL_MIN} to {CONTEXT_LIMIT}; {LAYER_PREFILL_ROWS} rows a pass");
            LAYER_PREFILL_ROWS
        }
    }
}

/// Layer order's outcome: fall back to the ordinary path, cancelled, the logits, or done up to a position from which the ordinary path
/// finishes the prompt (a short last segment, after the layer order has given its VRAM back).
enum LayerPrefill { Fallback, Cancelled, Ready(Vec<f32>), Continue(usize) }

/// How a layer-order prefill of `begin..end` splits: the position up to which it runs in layer order, and whether it keeps the staging
/// slots. A segment under [`LAYER_PREFILL_MIN`] runs the ordinary way, which needs them; a short last segment (the reply's opening tokens
/// after a conversation's checkpoint, as most chat requests end) is left to the ordinary path after the layer order ends, so the layer
/// order's rows may have the staging slots' VRAM. A short segment before a long one keeps them.
fn layer_prefill_split(begin: usize, end: usize, plan: &[(usize, Vec<&'static str>)], segment: usize) -> (usize, bool) {
    let checkpoints: Vec<usize> = plan.iter().map(|(at, _)| *at).collect();
    let stops = layer_prefill_stops(begin, end, &checkpoints, segment);
    let starts: Vec<usize> = [begin].into_iter().chain(stops.iter().copied()).collect();
    let short: Vec<bool> = stops.iter().zip(&starts).map(|(stop, start)| stop - start < LAYER_PREFILL_MIN).collect();
    match short.iter().position(|short| *short) {
        None => (end, false),
        Some(at) if at + 1 == stops.len() && at > 0 => (starts[at], false),
        Some(_) => (end, true),
    }
}

/// Common positions at which a layer-ordered prompt may stop: checkpoints plus segments of at most `segment` rows (a pass's).
fn layer_prefill_stops(begin: usize, end: usize, checkpoints: &[usize], segment: usize) -> Vec<usize> {
    let mut stops: Vec<usize> = checkpoints.iter().copied().chain([end]).filter(|at| *at > begin && *at <= end).collect();
    stops.sort_unstable();
    stops.dedup();
    let mut position = begin;
    let mut out = Vec::new();
    for stop in stops {
        while position < stop {
            position = position.saturating_add(segment.max(1)).min(stop);
            out.push(position);
        }
    }
    out
}

/// Layer spans and their MTP pairs use the same token chunks, including a tail of two instead of a one-row decode.
fn layer_prefill_chunks(begin: usize, end: usize, batch: usize) -> Vec<(usize, usize)> {
    let mut chunks = Vec::new();
    let mut first = begin;
    while first < end {
        let mut next = (first + batch).min(end);
        if end - next == 1 && next - first > 2 { next -= 1; }
        chunks.push((first, next));
        first = next;
    }
    chunks
}

/// STRAYDOG_PREFILL_OVERLAP: 0 reads a partially resident prompt through the two devices one after the other (the comparison's other side);
/// anything else, or unset, lets them take different batches at once ([`Model::prefill_overlapped`]; a CUDA build).
fn prefill_overlap_named(value: Option<&str>) -> bool {
    value.is_none_or(|value| value.trim() != "0")
}

fn prefill_overlap() -> bool {
    cfg!(feature = "cuda") && prefill_overlap_named(std::env::var("STRAYDOG_PREFILL_OVERLAP").ok().as_deref())
}

/// The batches `position..stop` of a read in batches of `batch` rows that two devices can take at once: all of them but a last of one row,
/// which takes the decode path as it does serially. A batch of fewer than two rows is the decode path's too (`Backend::prefill_layers_into` takes
/// two rows or more), so a read in batches of one row takes none at once.
fn overlap_batches(position: usize, stop: usize, batch: usize) -> Vec<(usize, usize)> {
    if batch < 2 {
        return Vec::new();
    }
    let mut batches = Vec::new();
    let mut at = position;
    while at < stop {
        let end = (at + batch.max(1)).min(stop);
        batches.push((at, end));
        at = end;
    }
    if batches.last().is_some_and(|(first, end)| end - first < 2) {
        batches.pop();
    }
    batches
}

/// Whether an overlapped read has been cancelled: the first device's loop saw it before a batch (`cut`), or it came while the second device drained
/// the last batches and the MTP pairs.
fn overlap_stopped(cut: bool, cancelled: &dyn Fn() -> bool) -> bool {
    cut || cancelled()
}

/// The batches a segment `position..stop` takes at once, where the overlap can be taken at all (`eligible`): none, without a list built, where it
/// cannot (HIP, one GPU, the layers in parts: the segment is then read a batch at a time, which must not cost a list of what remains each time).
fn overlap_segment(eligible: bool, position: usize, stop: usize, batch: usize) -> Option<Vec<(usize, usize)>> {
    if !eligible {
        return None;
    }
    let batches = overlap_batches(position, stop, batch);
    (batches.len() >= 2).then_some(batches)
}

/// The first device's share of a batch of the overlapped read: the embeddings, the PLE layer and the layers up to the split, whose rows come back to
/// `rows` for the second device (the serial path's hand-off from one device's workspace to the other's, through host memory). Over the device alone and
/// the PLE layer's operands, nothing of the Model: the second device's thread has the Model's other parts.
#[allow(clippy::too_many_arguments)]
fn overlap_first_half(device: &mut crate::gpu::Backend, ple_operands: &[Operand], tokens: &[u32], start: usize, end: usize,
                      spans: &[(usize, (usize, usize))], rows: &mut [f32], (embeddings, ple_rows): (Arc<Vec<f32>>, Vec<f32>),
                      times: &mut StageSeconds) -> Result<()> {
    let steps = (end - start) as u64;
    debug_assert_eq!(steps as usize, tokens[start..end].len());
    let clock = Instant::now();
    // (the rows the serial path reads here, `rows_at` and `ple_embeddings`, were read ahead by the batch's thread)
    device.prefill_embed(0, &embeddings)?;
    times.prep += clock.elapsed().as_secs_f64();
    for (at, (index, (first, last))) in spans.iter().enumerate() {
        if *index == 1 {
            let clock = Instant::now();
            device.prefill_ple("blk.1.", &ple_rows, ple_operands, 1)?;
            times.ple += clock.elapsed().as_secs_f64();
        }
        let clock = Instant::now();
        if at + 1 == spans.len() {
            device.prefill_layers_into(*first, *last, None, start as u64, steps, rows)?;
            times.last_span += clock.elapsed().as_secs_f64();
        } else {
            device.prefill_layers(*first, *last, None, start as u64, steps, 0)?;
            times.span_calls += clock.elapsed().as_secs_f64();
        }
    }
    Ok(())
}

/// Whether the layer split leaves the first layer and the PLE layer (the one after it, on the device of layer 1) both on the first device, as the
/// overlapped read's stages need: the first device's thread applies the PLE layer after layer 0 and the second device's thread has no such step, so
/// a split at layer 1 (`STRAYDOG_SPLIT=1`) reads serially.
fn overlap_split_ok(split: Option<usize>) -> bool {
    split.is_some_and(|split| split >= 2)
}

/// A matrix of the weights as an operand of its own bytes ([`Model::projection`]). The operands of the overlapped read are built by these functions over
/// the weights alone: the read holds the weights shared beside the two devices (each a `&mut` for its own thread), and an operand borrows from the
/// weights and from nothing else, which the compiler checks.
fn projection_of<'a>(weights: &'a Weights, name: &str) -> Result<Operand<'a>> {
    let tensor = weights.tensor(name)?;
    let [cols, rows] = tensor.dims[..] else { return Err(format!("Straydog matrix rank mismatch: {name}")) };
    if crate::formats::block(tensor.kind).is_none() {
        return Err(format!("unsupported Straydog weight format {}: {name}", tensor.kind));
    }
    let (data, kind) = weights.raw(name, 0, None)?;
    Ok(Operand::bytes(format!("{name}:None"), data, kind, rows, cols))
}

/// [`Model::q8_projection`] over the weights alone.
fn q8_projection_of<'a>(weights: &'a Weights, name: &str) -> Result<Operand<'a>> {
    let operand = projection_of(weights, name)?;
    let float = matches!(operand.kind, 0 | 1 | 30);
    if !(crate::weights::hc_quantized(name) || crate::weights::drafter_quantized(name) && float) || !operand.cols.is_multiple_of(32) {
        return Ok(operand);
    }
    let Data::Bytes(bytes) = operand.data else { return Ok(operand) };
    Ok(Operand { key: format!("{name}:q8"), data: Data::Quantize(bytes, operand.kind), kind: 8, rows: operand.rows, cols: operand.cols })
}

/// [`Model::vector`] over the weights alone.
fn vector_of<'a>(weights: &'a Weights, name: &str, key: Option<&str>) -> Result<Operand<'a>> {
    let key = key.map_or_else(|| format!("@vector:{name}"), str::to_string);
    let (rows, cols) = weights.vector_shape(name)?;
    let (data, kind) = weights.raw(name, 0, None)?;
    Ok(Operand { key, data: Data::Decode(data, kind), kind: 0, rows: rows as u64, cols: cols as u64 })
}

/// The PLE layer's operands ([`Model::ple_operands`]) over the weights alone.
fn ple_operands_of(weights: &Weights) -> Result<Vec<Operand<'_>>> {
    let mut operands = vec![q8_projection_of(weights, "blk.1.ple_key.weight")?, q8_projection_of(weights, "blk.1.ple_value.weight")?];
    for name in ["ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight"] {
        operands.push(vector_of(weights, &format!("blk.1.{name}"), None)?);
    }
    Ok(operands)
}

/// The MTP layer's prologue operands ([`Model::mtp_operands`]) over the weights alone.
fn mtp_operands_of(weights: &Weights) -> Result<Vec<Operand<'_>>> {
    Ok(vec![vector_of(weights, "mtp.embed_norm.weight", None)?, q8_projection_of(weights, "mtp.fc_embedding.weight")?,
            vector_of(weights, "mtp.hidden_norm.weight", None)?, q8_projection_of(weights, "mtp.fc_hidden.weight")?])
}

/// The PLE table rows of `tokens`, which follow `before` (see [`Model::ple_indices`]), from the weights alone.
fn ple_indices_of(weights: &Weights, before: &[u32], tokens: &[u32]) -> Result<Vec<i64>> {
    let id = |token: &u32| if *token >= IMAGE_TOKEN { i64::from(IMAGE_PAD) } else { i64::from(*token) };
    let history: Vec<i64> = before[before.len().saturating_sub(2)..].iter().map(id).collect();
    let tokens: Vec<i64> = tokens.iter().map(id).collect();
    ngram_rows(&history, &tokens, &weights.integer_arrays)
}

/// A batch's first-layer rows and PLE rows, read from the weights ahead of the batch (the overlapped read's thread for it): what
/// [`Model::rows_at`] and [`Model::ple_embeddings`] make for a text batch, and the PLE rows of the batch after it asked for.
fn prepare_batch(weights: &Weights, tokens: &[u32], start: usize, end: usize, next: usize) -> Result<(Arc<Vec<f32>>, Vec<f32>)> {
    for from in (end..next).step_by(4096) {
        let part = &tokens[from..(from + 4096).min(next)];
        let before = if from == end { &tokens[..end] } else { &tokens[end..from] };
        weights.prefetch_rows(PLE_TABLE, &ple_indices_of(weights, before, part)?)?;
    }
    let ids: Vec<i64> = tokens[start..end].iter().map(|token| i64::from(*token)).collect();
    let rows = weights.rows("token_embd.weight", &ids)?;
    let ple = weights.rows(PLE_TABLE, &ple_indices_of(weights, &tokens[..start], &tokens[start..end])?)?;
    Ok((Arc::new(rows), ple))
}

/// The rows of a pass of the MTP layer over a prompt batch's pairs on the overlapped read: the batch's all (the device runs the prologue in pieces
/// of 512 and the layer once over them, `sd_mtp_prompt_kept`), where 512 rows at a time made each pass's fixed costs (the CPU's routes, the
/// experts it streams, the host's reads) eight times a batch.
const MTP_PASS_ROWS: usize = 4096;

/// The MTP layer's pairs of a prompt batch on the second device, from the batch's final rows in the layer 47's workspace there: kept on the device
/// beside the row before them (`carried` rows of the batch before, else the host's `previous` row), and fed to the layer in passes of
/// [`MTP_PASS_ROWS`] pairs; the same pairs and the same device copies, with the device as an argument ([`Model::mtp_prompt_kept`]). `token_rows` are the
/// batch's tokens' rows of the embedding table (the rows the first device's embeddings are, read once for both devices).
fn mtp_prompt_pairs(backend: &mut crate::gpu::Backend, operands: &[Operand], token_rows: &[f32], tokens: &[u32], start: usize,
                    host: Option<&[f32]>, carried: u64) -> Result<()> {
    backend.mtp_keep(tokens.len() as u64, carried, host)?;
    let before = carried > 0 || host.is_some();
    let (following, first, row) = if before { (tokens, start - 1, 0) } else { (&tokens[1..], start, 1) };
    if following.is_empty() {
        return Ok(());
    }
    if token_rows.len() != tokens.len() * HIDDEN {
        return Err("Straydog MTP pairs: the batch's token rows are not its tokens'".into());
    }
    backend.mtp_embed(if before { token_rows } else { &token_rows[HIDDEN..] })?;
    for offset in (0..following.len()).step_by(MTP_PASS_ROWS) {
        let count = (following.len() - offset).min(MTP_PASS_ROWS) as u64;
        backend.mtp_prompt_kept(operands, (row + offset) as u64, offset as u64, count, (first + offset) as u64)?;
    }
    Ok(())
}

/// How an overlapped read of a segment ended ([`Model::prefill_overlapped`]).
enum Overlapped {
    /// No buffers for the rows between the devices: nothing was read.
    Declined,
    Cancelled,
    /// Every batch was read; the logits where the last of them ends the prompt.
    Read(Option<Vec<f32>>),
}

/// Seconds a stage of an overlapped read spent: the first device's thread (`first` working, `blocked` waiting for a buffer or for the second
/// device to take a batch) and the second's (`second` on its layers and the rows' upload, `mtp` on the MTP pairs, `idle` waiting for a batch).
#[derive(Default)]
struct StageSeconds {
    first: f64,
    blocked: f64,
    second: f64,
    mtp: f64,
    idle: f64,
    /// The first device's thread within `first`: the embeddings' upload, the PLE layer, the span calls that queue and the last span's (with the rows' copy).
    prep: f64,
    ple: f64,
    span_calls: f64,
    last_span: f64,
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
    /// The state of a system block held at its root, imported ([`Model::import_prefix`]): once restored, the sequence is the root's too.
    root: bool,
    /// The state of a conversation held at one of its checkpoints, imported ([`Model::import_conversation`]): once restored, the sequence is
    /// the checkpoint's too.
    kept: bool,
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

/// What decoding a system block's state needs of the model that will take it ([`Model::prefix_plan`]).
pub struct PrefixPlan {
    tokens: usize,
    hidden: usize,
    /// The sections its bytes must have, to the byte.
    sections: Vec<(u64, u64)>,
    /// The parts to fill: the K/V's and the indexer's blocks, and the MTP layer's indexer history.
    saved: Positions,
    kept: Option<Positions>,
    /// Each device's recurrent bytes, and the K/V's capacity the parts' places are for.
    recurrent: Vec<u64>,
    capacity: usize,
}

/// A system block's state decoded, ready for [`Model::install_prefix`].
pub struct PrefixImage {
    ids: Vec<u32>,
    final_hidden: Option<Vec<f32>>,
    saved: Positions,
    kept: Option<Positions>,
    /// Where each device's recurrent state is in the bytes that were decoded.
    recurrent: Vec<std::ops::Range<usize>>,
    capacity: usize,
}

impl PrefixPlan {
    /// Decode the state `parsed` (of the bytes `Parsed::start` and `payload` came from) as this plan lays it out: its header must be the layout's
    /// to the byte, its tokens must be tokens of the vocabulary. Takes no part of the model, and a few hundred megabytes of copying: the work of an
    /// import that is done where nothing waits for it.
    pub fn decode(self, parsed: &crate::prefix::Parsed) -> Result<PrefixImage> {
        let refused = |why: &str| Err(format!("Straydog prefix state refused: {why}"));
        crate::prefix::check(&parsed.sections, &self.sections)?;
        let ids = crate::prefix::tokens_of(parsed.payload, self.tokens);
        if ids.iter().any(|token| *token as usize >= VOCABULARY) {
            return refused("a token outside the vocabulary");
        }
        let (hidden_bytes, rest) = parsed.payload[4 * self.tokens..].split_at(4 * self.hidden);
        let final_hidden = (self.hidden > 0).then(|| hidden_bytes.chunks_exact(4).map(|word| f32::from_le_bytes(word.try_into().unwrap())).collect());
        let (mut saved, mut kept) = (self.saved, self.kept);
        let places = Group::fill_prefix(&mut saved, kept.as_mut(), &self.recurrent, rest)?;
        // (the places in the bytes parsed: past the header, the tokens and the hidden state)
        let base = parsed.start + 4 * self.tokens + 4 * self.hidden;
        let recurrent = places.into_iter().map(|place| base + place.start..base + place.end).collect();
        Ok(PrefixImage { ids, final_hidden, saved, kept, recurrent, capacity: self.capacity })
    }
}

/// A system block's state as bytes, with what names it ([`Model::export_prefix`]).
pub struct Prefix {
    /// The tokens it is held at.
    pub tokens: usize,
    /// A digest of those tokens ([`crate::prefix::tokens_digest`]).
    pub digest: String,
    pub bytes: Vec<u8>,
}

pub struct Settings {
    pub context: usize,
    pub state_budget_bytes: u64,
    /// The rows of a prefill batch asked for, kept as asked where the placement allows; [`DERIVED_BATCH`] (0): the batch derived from the GPU
    /// budgets ([`resident_batch`]), which the placement may narrow where it does not pay.
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
    /// The generator for one position of a seeded sequence (a draw that depends on the seed and the position alone: see
    /// [`sampling::accept_keyed`]).
    pub fn keyed(seed: u64, position: u64) -> Rng {
        Rng::new(seed ^ position.wrapping_add(1).wrapping_mul(0xd6e8_feb8_6659_fd93))
    }

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

/// The layers a partially resident model's route counts cover (its decoder's 48 and the drafter's one) and the experts of each.
pub const ROUTED_LAYERS: usize = 49;
pub const ROUTED_EXPERTS: usize = 512;

/// What a warm start planned ([`Model::warm_start`]): the expert slots there are, what they held and how many experts had been swapped into them
/// at that moment, and what to ask the OS for.
pub struct WarmStart {
    pub slots: u64,
    pub held: u64,
    pub swapped: u64,
    pub hints: Hints,
}

/// Where the slot fill of a warm start stands ([`Model::warm_fill`]): the swaps still on their way (0: no more wanted), the slots, what they hold,
/// and how many experts have been swapped into them since the start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarmFill {
    pub pending: u64,
    pub slots: u64,
    pub held: u64,
    pub swapped: u64,
}

pub struct Model {
    // Field order is drop order: the devices go before the CPU engine they use, and both
    // before the mappings their native plans point into.
    group: Group,
    cpu: Option<CpuExperts>,
    /// Pipelined verification (STRAYDOG_PIPELINE=1): the second device's CPU engine, the first device's being `cpu` (each host thread has its own).
    cpu_second: Option<CpuExperts>,
    /// The pipelined verification's settings (STRAYDOG_PIPELINE and the rest, read at the start).
    pipeline: pipeline::PipelineConfig,
    /// The seed of the request under way (the keyed draws of `sampling::accept_keyed`).
    sample_seed: u64,
    /// Whether the request under way holds what the pipeline holds (the expert slots' swaps, the assist level, the head on one GPU): the requests
    /// the pipeline does not take (thinking ones, a constraint) run the ordinary loop as it is.
    pipeline_live: bool,
    /// The page-locked copies of the banks of the layers that a GPU which cannot register host memory read-only holds in place of the mapping
    /// ([`LockMode::Copy`]). The devices' plans and the CPU engine read them, and the model has no `Drop`: this field is declared after `group` and `cpu`
    /// so that they are dropped first (the devices drain their copy thread and destroy their plans, the CPU's workers stop) and nothing reads a copy when
    /// it is freed. Keep it after them.
    copies: Copies,
    pub weights: std::sync::Arc<Weights>,
    pub context: usize,
    pub prefill_batch_size: usize,
    pub state_bytes: u64,
    /// The most drafts a pass offers (a partially resident model offers as many as [`Model::drafts`] has in force).
    pub speculative: usize,
    /// The most drafts a long match's copy verifies (drafts::Policy, by the match's length).
    pub copy_drafts: usize,
    pub resident: bool,
    pub partial: bool,
    /// Layer-ordered long prefill; STRAYDOG_LAYER_PREFILL=0 selects the ordinary batch order.
    layer_prefill: bool,
    // Explicit numerical comparisons retain the developer API's forced path.
    layer_prefill_forced: bool,
    /// A layer-order pass holds its hidden rows in VRAM where they fit (STRAYDOG_LAYER_ROWS_VRAM=0: on the host, moved to the GPU and back
    /// for every layer, as before; [`Model::layer_rows_in_vram`] for a test).
    layer_rows_vram: bool,
    /// New tokens from which a read scans its DeltaNet layers in chunks ([`CHUNKED_SCAN_READ`]; [`Model::chunked_scan_from`] for a test).
    chunked_scan_from: usize,
    /// New tokens from which a read's HyperConnections run their up projection and finish as one kernel ([`CHUNKED_SCAN_READ`], never with
    /// STRAYDOG_HC=blas; [`Model::hc_fused_from`] for a test).
    hc_fused_from: usize,
    layer_prefill_restore_failure: Option<String>,
    /// A failed GPU state restoration leaves this model unavailable for further generations.
    recovery_failure: Option<String>,
    pub host_kv: bool,
    /// The rows of a long read's batches where the workspaces hold more than `prefill_batch_size` (0: they do not, or the batch was given: see
    /// [`wide_placement`]), and what the widest wanted would have needed when they do not.
    pub wide_batch_size: usize,
    pub wide_note: Option<String>,
    /// What the placement's first choice would have needed, when the VRAM left did not hold it (see `resident_placement`).
    pub placement_note: Option<String>,
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
    /// A logits vector the draw is done with, for the next step's head to fill (no allocation, no fresh pages to fault in).
    logits_spare: Vec<f32>,
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
    /// Running means of a decode pass's seconds by its rows (0: none yet), for [`Model::partial_drawn_cut`] and the request line.
    pass_seconds: [f64; PARTIAL_ROWS],
    passes_with_draft: u32,
    /// The drafts a pass offers now: `speculative` where the whole model is resident, else the width in force of `width`.
    drafts: usize,
    /// A partially resident model with a drafter chooses how many drafts a pass offers by what its passes cost ([`width`]); `speculative` is
    /// the most.
    width: Option<width::Chooser>,
    /// The drafts were held by [`Model::pin_drafts`] (a test, or STRAYDOG_PIPELINE_DRAFTS): no width an earlier run chose is taken then.
    drafts_held: bool,
    /// Tests only: the seconds a pass is taken to have taken by its rows, in place of the clock ([`Model::script_pass_seconds`]).
    pass_clock: Option<Box<dyn Fn(usize) -> f64 + Send + Sync>>,
    /// The assist level's tuner, where banks are page-locked and the GPU can take some of the CPU's routes ([`crate::assist`]).
    assist: Option<assist::Tuner>,
    /// Tests only: the levels the decode passes of one row are to run at, in place of the tuner's rounds ([`Model::assist_script`]).
    assist_script: Option<AssistScript>,
    /// This request's passes by rows, and this process's memory as it began, for its log line.
    pass_counts: [u64; PARTIAL_ROWS],
    memory_before: Option<(u64, u64, u64)>,
    /// The pages read from the drive for page faults when the request began ([`crate::host::fault_reads`]).
    fault_reads_before: Option<crate::host::FaultReads>,
    /// The CPU engine's read-ahead hints (taken, left) when the request began.
    read_aheads_before: (u64, u64),
    /// The CPU engine's phase times when the request began ([`CpuExperts::phase_ns`]).
    cpu_phases_before: [u64; 3],
    /// Positions the K/V holds now (the context, unless it started small: `ensure_kv`).
    kv_capacity: usize,
    /// What is page-locked, for the thread that keeps the working set's limit.
    pinned_now: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// The bytes of the banks page-locked in place now (not the layers held as copies, which are the process's own memory): what the working set's
    /// limit is never under, with that memory ([`crate::host::keep_resident`]).
    in_place_now: std::sync::Arc<std::sync::atomic::AtomicU64>,
    memory_watch: std::sync::Arc<crate::host::WorkingSetWatch>,
    /// Page-locked host bytes besides the K/V.
    pinned_other: u64,
    /// The expert banks a partially resident model reads from host memory.
    banks_bytes: u64,
    /// The layers whose banks are page-locked (resident until the process ends): a warm start asks the OS for no others' pages.
    locked_layers: Vec<usize>,
    /// The plans the GPU's assist had mapped when it was last started ([`Model::start_assist`]): a layer locked since is mapped when the count grows.
    assist_mapped: u64,
    /// The layers the start planned to page-lock behind "ready" and that are not locked yet, in the order they are taken ([`Model::bank_locks`]):
    /// a warm start asks the OS for none of their pages either.
    lock_pending: Vec<usize>,
    /// A bank that did not match the file's tree when it was locked behind "ready": no request is served from here on.
    bank_failure: Option<String>,
    /// How the banks are page-locked ([`LockMode`]): decided once, when the plan is made ([`Model::plan_host_banks`]), from what the GPUs say of registering
    /// host memory read-only, and kept for every later plan, so that a mapping the runtime refused is never offered to it again.
    lock_mode: LockMode,
    /// The layers registered in place and the reads of them under way ([`crate::banks::Registered`]): this model's alone.
    registered: std::sync::Arc<crate::banks::Registered>,
    /// A refusal of the very first layer, with nothing locked, that ends the planning of more ([`Model::more_bank_locks`]); and whether the log said so.
    lock_refusal: Refusal,
    refusal_told: bool,
    /// The start's plan was handed out (or found nothing to lock), and a plan is out that has not ended ([`Model::bank_locks`],
    /// [`Model::more_bank_locks`], [`Model::banks_lock_ended`]): one locking at a time, and more only after the start's.
    locks_planned: bool,
    locking: bool,
    /// Set when locked layers were given back to a request ([`Model::give_back`]): the locking behind "ready" locks no more (its plan holds this too).
    lock_halt: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The bytes of host K/V that the context's first 64K tokens still have to grow into (see [`Model::refresh_kv_ahead`]): the locking behind "ready" leaves them.
    kv_ahead: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// What the engine has given back to requests since it started, in order.
    pub gave_back: Vec<GaveBack>,
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
    /// Set only while the warm-up behind "ready" runs its passes ([`Model::set_warm_stop`]): the flag a request's arrival sets.
    warm_stop: Option<&'static std::sync::atomic::AtomicBool>,
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
        if settings.prefill_batch_size != DERIVED_BATCH && !(1..=4096).contains(&settings.prefill_batch_size) {
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
        let derived = settings.prefill_batch_size == DERIVED_BATCH;
        let asked = if derived { resident_batch(group.memory()?.1) } else { settings.prefill_batch_size };
        if asked > group.logical_capacity() {
            return Err("Straydog prefill exceeds this GPU workspace capacity".into());
        }
        let context64 = context as u64;
        let mut state_bytes = context_state_bytes(context64, drafting);
        let attention_capable = group.batch_capacity() == 256;
        let kv_bytes = attention_layers(drafting) * context64 * KV_POSITION_BYTES;
        let fits = attention_capable && kv_fits_budget(context64, drafting, crate::gpu::KvFormat::selected(), group.memory()?.1);
        let host_capable = attention_capable && group.host_state_supported()?;
        let recurrent_bytes = 36 * 48 * 128 * 128 * 4;
        // K/V beyond a quarter of the VRAM budget stays in mapped host memory, which the
        // GPU kernels read over the bus.
        let (mut host, mut batch) = (attention_capable && !fits && host_capable, asked);
        let mut placement_note = None;
        let mut resident = false;
        let (mut wide, mut wide_note) = (0, None);
        if host_capable && state_bytes - recurrent_bytes <= settings.state_budget_bytes {
            // The layers in parts on two GPUs that hold them so (Group::enable_parts), else split by layers.
            let mut placed = (None, None);
            if group.enable_parts()? {
                placed = resident_placement(&weights, &group, context64, asked, derived, drafting, fits)?;
                if placed.0.is_none() {
                    group.disable_parts()?;
                }
            }
            if placed.0.is_none() {
                // (none fitting either way, the note stays the parts' when they were tried: the placement a little more VRAM gives)
                let split = working_split(&weights, &mut group, context64, asked, derived, drafting, fits)?;
                placed = (split.0, if split.0.is_some() { split.1 } else { placed.1.or(split.1) });
            }
            placement_note = placed.1;
            if let Some((keep, resident_host, resident_batch)) = placed.0 {
                (host, batch) = (resident_host, resident_batch);
                resident = true;
                // The workspaces of a long read's wider batches, where the layers are in parts and what the placement left holds them.
                // (the rows named by STRAYDOG_PREFILL_WIDE are tried whatever batch was given: a measurement of what a width is worth)
                let named = std::env::var("STRAYDOG_PREFILL_WIDE").is_ok_and(|rows| rows != "off" && rows != "0");
                if (derived || named) && group.parts() {
                    (wide, wide_note) = wide_placement(&weights, &group, context64, batch, keep, drafting, host)?;
                }
            }
        }
        // STRAYDOG_SPLIT=N: the second device's first layer, in place of the balance of weight bytes (two GPUs, a model that is not fully
        // resident; nothing is placed yet). A pass's time on a device is not its dense bytes: this moves the boundary to balance by time.
        if !resident && group.split().is_some() {
            if let Some(split) = std::env::var("STRAYDOG_SPLIT").ok().and_then(|value| value.parse::<usize>().ok()) {
                group.set_split(split)?;
            } else if cfg!(feature = "cuda") {
                // A layer's routed experts live in its own device's slots here (no expert halves), so the balance counts them: by dense
                // bytes alone the first device had 34 layers with slots for half their experts while the second's slots stood empty.
                let (layer_bytes, tail) = weights.placement_bytes(false);
                group.balance_layers(&layer_bytes, tail)?;
            }
            eprintln!("Straydog layer split: device 0 has layers 0..{0}, device 1 {0}..48", group.split().unwrap_or(0));
        }
        // (on 48 GiB of budgets and more, where the derived batch is the widest, a partial placement keeps what it had at 2048: its room goes to
        // its expert slots. A batch given is taken as it is, and a smaller budget's derived one is not wider than that.)
        if derived && !resident && group.memory()?.1 >= WIDE_BATCH_BUDGET {
            batch = batch.min(NARROW_BATCH);
        } else if derived && !resident && partial_pair(cfg!(feature = "cuda"), group.backends.len(), group.memory()?.1) {
            batch = PARTIAL_PAIR_BATCH.min(group.logical_capacity());
        }
        if !resident && batch >= PRESTAGE_BATCH && partial_pair(cfg!(feature = "cuda"), group.backends.len(), group.memory()?.1) {
            group.set_prestage(prestage_slots());
        }
        if !resident && !host && host_capable {
            let kinds = (0..48).map(|layer| weights.expert_kinds(layer)).collect::<Result<Vec<_>>>()?;
            let requirements = partial_requirements(weights.dense_requirements(), &kinds, group.prestage_slots());
            let requirements = if drafting { partial_drafter_requirements(requirements, weights.drafter_requirements()) } else { requirements };
            let planned = group.planned_state_bytes(context64, batch as u64, drafting, false)?;
            // Before any state is allocated: keep the host path if VRAM K/V crowds out dense weights
            // or a requested drafter. Its indexers start small before the partial placement's gates.
            host = !group.can_pin_model(&requirements, Some(&planned))?;
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
        group.reserve_model_state(context64, batch.max(wide) as u64, drafting, host)?;
        let checkpoint_bytes = group.reserve_checkpoints(CHECKPOINTS.len() + PARK_SLOTS.len())?;
        let mut load_seconds = vec![("state", lap(&mut clock))];
        crate::host::commit_note("states and checkpoints");
        let devices = group.backends.len();
        let mut model = Model {
            group,
            cpu: None,
            cpu_second: None,
            pipeline: pipeline::PipelineConfig::from_env().for_devices(devices),
            sample_seed: 0,
            pipeline_live: true,
            copies: Copies::default(),
            weights: std::sync::Arc::new(weights),
            context,
            prefill_batch_size: batch,
            state_bytes,
            speculative: 0,
            copy_drafts: 0,
            resident: false,
            partial: false,
            layer_prefill: !std::env::var("STRAYDOG_LAYER_PREFILL").is_ok_and(|v| v == "0"),
            // (STRAYDOG_LAYER_PREFILL=force: every eligible prompt takes the layer order whatever the memory, to measure it where the
            // choice would not make it, e.g. a machine whose RAM holds the banks)
            layer_prefill_forced: std::env::var("STRAYDOG_LAYER_PREFILL").is_ok_and(|v| v == "force"),
            layer_rows_vram: layer_rows_vram_named(std::env::var("STRAYDOG_LAYER_ROWS_VRAM").ok().as_deref()),
            chunked_scan_from: chunked_scan_from_named(std::env::var("STRAYDOG_DELTA_SCAN").ok().as_deref()),
            hc_fused_from: hc_fused_from_named(std::env::var("STRAYDOG_HC").ok().as_deref()),
            layer_prefill_restore_failure: None,
            recovery_failure: None,
            host_kv: host,
            wide_batch_size: wide,
            wide_note,
            placement_note,
            history: Vec::new(),
            final_hidden: None,
            checkpoints: IndexMap::new(),
            parked: None,
            park_budget: crate::host::host_memory_bytes() / 8,
            park_whole: false,
            last_metrics: Map::new(),
            logits_digests: None,
            logits_capture: None,
            logits_spare: Vec::new(),
            forced_rows: None,
            deferred: false,
            mtp_kept: 0,
            draft_vocabulary: false,
            draft_extras: HashSet::new(),
            draft_pending: Vec::new(),
            load_seconds: Vec::new(),
            profile: std::env::var_os("DS41_PROFILE").map(|_| Profile::new(true)),
            pass_seconds: [0.0; PARTIAL_ROWS],
            passes_with_draft: 0,
            drafts: 0,
            width: None,
            drafts_held: false,
            pass_clock: None,
            assist: None,
            assist_script: None,
            pass_counts: [0; PARTIAL_ROWS],
            memory_before: None,
            fault_reads_before: None,
            read_aheads_before: (0, 0),
            cpu_phases_before: [0; 3],
            kv_capacity: context,
            pinned_now: Default::default(),
            in_place_now: Default::default(),
            memory_watch: Default::default(),
            pinned_other: checkpoint_bytes,
            banks_bytes: 0,
            locked_layers: Vec::new(),
            assist_mapped: 0,
            lock_pending: Vec::new(),
            bank_failure: None,
            lock_mode: LockMode::Register,
            registered: Default::default(),
            lock_refusal: Refusal::default(),
            refusal_told: false,
            locks_planned: false,
            locking: false,
            lock_halt: Default::default(),
            kv_ahead: Default::default(),
            gave_back: Vec::new(),
            hybrid_counts: Value::Null,
            ple_counts: (0, 0, 0),
            ple_wait: 0.0,
            images: Vec::new(),
            rope_delta: 0,
            rope_table: false,
            warm_stop: None,
        };
        // Placement allocates every weight before any is read; reading then fills them.
        model.group.defer_uploads();
        model.resident = model.prepare_resident_moe()?;
        if !model.resident {
            let start = starting_kv(context);
            let growing = model.group.batch_capacity() == 256 && !model.group.parts() && context > start;
            // (before the expert placement, whose gates read what the indexers hold: at the whole context, a third of the slots)
            if growing && host {
                model.kv_capacity = start;
                model.group.start_kv(start as u64)?;
            }
            model.partial = model.group.batch_capacity() == 256 && model.prepare_hybrid_moe(settings.cpu_threads)?;
            // A K/V in VRAM starts small after the placement's gates, which counted it at the whole context: the placement is one that the K/V can
            // grow to fill (the slots, made below, give the VRAM back for it as it grows), and the slots start with what a short conversation leaves them.
            if growing && !host && model.partial {
                model.kv_capacity = start;
                model.group.start_kv(start as u64)?;
            }
        }
        if !model.resident && !model.partial {
            return Err("the GPU budget cannot hold the model's dense weights; raise --vram".into());
        }
        if model.partial && model.profile.is_none() {
            // (the request line's "other" is broken down into the decode step's phases: crate::model::decode_part)
            model.profile = Some(Profile::new(false));
        }
        crate::host::commit_note("experts placed");
        model.prepare_decoder(context64, false)?;
        crate::host::commit_note("decoder placed");
        if model.partial {
            // Size the expert pool beside the PLE and head mixers otherwise uploaded by the
            // first request. These operands join the verified load before slots are sized.
            model.with_operands(Model::ple_operands, Group::pin)?;
            model.with_operands(Model::head_operands, Group::pin)?;
        }
        // SAFETY of the borrow: the operand points into the weights, which outlive the group.
        let head: Operand<'static> = unsafe { std::mem::transmute(model.projection("output.weight")?) };
        model.group.pin_head(&head)?;
        let keep = model.first_use_tensors()?;
        load_seconds.push(("read and place", lap(&mut clock)));
        crate::host::commit_note("placed");
        let pending = model.group.take_pending();
        // (the load changes the weights and the group at once: disjoint fields, so not through `weights_mut`)
        let loading = std::sync::Arc::get_mut(&mut model.weights).expect("the weights are held by the model alone while it loads");
        crate::load::load(loading, &mut model.group, pending, &keep, progress)?;
        load_seconds.push(("read", lap(&mut clock)));
        model.group.experts_finalize()?;
        load_seconds.push(("expert layout", lap(&mut clock)));
        crate::host::commit_note("read");
        // Host K/V page-locked on a thread of its own meanwhile (see `Backend::prefetch_host_states`).
        model.group.finish_model_state()?;
        load_seconds.push(("host_state", lap(&mut clock)));
        crate::host::commit_note("host K/V");
        model.prepare_decoder(model.kv_capacity as u64, true)?;
        if model.partial && model.pipeline.frozen_slots() {
            // (before the verification buffers: the first device keeps a second window of them)
            model.group.pipeline_setup(model.pipeline.windows(model.group.backends.len()), true)?;
        }
        if model.partial && model.weights.drafter {
            model.group.verify_reserve()?;
        }
        let mut two_drafts = false;
        if model.partial {
            let pinned = checkpoint_bytes + if host { attention_layers(model.weights.drafter) * model.kv_capacity as u64 * kv_position_bytes() } else { 0 };
            // (the banks are locked behind "ready", by the worker: [`Model::bank_locks`]; the slots below are sized as they were with
            // the locks taken, and the plan has left the commit the locks need out of what the VRAM takes)
            model.plan_host_banks(pinned)?;
            crate::host::commit_note("banks planned");
            // A draft's vocabulary (DRAFT_ROWS) on CUDA as well: the head's split across the two GPUs is the same, and it takes its rows' room
            // before the slots take the rest (34 MB: 17 slots of 17855).
            if drafting && model.weights.drafter && cfg!(feature = "cuda")
                && std::env::var("STRAYDOG_DRAFT_VOCABULARY").map_or(true, |value| value != "0")
            {
                model.draft_vocabulary = model.group.draft_vocabulary(&head, DRAFT_ROWS, &[], true)?;
            }
            let slots = model.group.create_expert_slots()?;
            if model.group.backends.len() == 2 {
                let per_device: Vec<String> = model.group.backends.iter().map(|backend| {
                    let used = backend.memory().map_or(0, |(used, _)| used) as f64 / (1u64 << 30) as f64;
                    format!("{} slots, {used:.1} GiB used", backend.slots().map_or(0, |(slots, _)| slots))
                }).collect();
                eprintln!("Straydog expert slots by device: {}", per_device.join(" | "));
            }
            // No staging mirror: layer order gathers through it, so a prompt keeps the ordinary batches.
            if !model.group.expert_slot_mirrors() {
                model.layer_prefill = false;
                model.layer_prefill_forced = false;
            }
            let experts = 48 * model.weights.tensor("blk.0.ffn_gate_exps.weight")?.dims.last().copied().unwrap_or(0) as u64;
            two_drafts = cfg!(feature = "hip") && slots as f64 >= TWO_DRAFTS_START * experts as f64;
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
            model.weights_mut().cache_rows(PLE_TABLE, row_cache)?;
        }
        load_seconds.push(("plans", lap(&mut clock)));
        crate::host::commit_note("plans");
        model.load_seconds = load_seconds;
        // A partially resident model starts at one draft a pass: IQ2_XS --vram 12 on an R9700, 64 tok/s without drafts, 70 with one,
        // 64 with two and 65 with three. The widest pass costs far more than the rest there (16 ms for one row, 24 for two, 53 for four),
        // and far less on a machine whose GPU is the limit of a layer: how many drafts pay is for the passes to say ([`width::Chooser`]).
        // It starts with two once the slots hold enough on HIP ([`TWO_DRAFTS_START`]).
        let most = if drafting && model.weights.drafter { settings.speculative } else { 0 };
        let (most, start) = if model.partial { (most.min(PARTIAL_ROWS - 1), most.min(if two_drafts { 2 } else { 1 })) } else { (most, most) };
        model.speculative = most;
        model.drafts = start;
        model.width = (model.partial && most > 0).then(|| width::Chooser::new(most, start));
        // (a partially resident model's rows cost it far more: as many as it drafts)
        model.copy_drafts = if model.partial || most == 0 { start } else { VERIFY_ROWS - 1 };
        // (STRAYDOG_PIPELINE_DRAFTS with the pipeline or `freeze`: the drafts of every pass held, not chosen by the passes' times)
        if model.pipeline.frozen_slots() {
            if let Some(drafts) = model.pipeline.drafts() {
                model.pin_drafts(Some(drafts.min(most)));
            }
        }
        model.refresh_kv_ahead();
        // A read of wide batches grows the rows' stagings as its batches ask (0.4 s of pinning, and a growth waits for the batch queued on its device):
        // they are page-locked from here on a thread of each device, at the size that read would pin them at, which stay pinned after it all the same
        // (beside the weights' reading it held the load up by 0.3 s).
        if model.wide_batch_size > model.prefill_batch_size {
            model.group.prefetch_row_staging(model.wide_batch_size as u64)?;
        }
        Ok(model)
    }

    /// The most rows a pass of this placement takes.
    pub fn pass_rows(&self) -> usize {
        if self.partial { PARTIAL_ROWS } else { VERIFY_ROWS }
    }

    /// The drafts a pass offers now: a partially resident model's width in force ([`width::Chooser`]).
    pub fn drafts(&self) -> usize {
        self.drafts
    }

    /// Tests only: hold the drafts a partially resident model offers at `drafts` (None: choose by the passes' costs again).
    pub fn pin_drafts(&mut self, drafts: Option<usize>) {
        if let Some(chooser) = self.width.as_mut() {
            chooser.pin(drafts);
            self.drafts = chooser.drafts();
            self.copy_drafts = self.drafts;
            self.drafts_held = drafts.is_some();
        }
    }

    /// The width in force of a partially resident model's drafts, once the passes have chosen it (the chooser's first table was given): the width a
    /// later start on this placement may begin at. None before, and where nothing chooses.
    pub fn chosen_drafts(&self) -> Option<usize> {
        self.width.as_ref().filter(|chooser| chooser.decided()).map(width::Chooser::drafts)
    }

    /// A partially resident model begins at `drafts`, the width an earlier run on this placement chose, in place of where it started: its chooser is built
    /// again at it, as at the load, and goes on choosing from there. Only while the width is the chooser's to take: not where the drafts are held
    /// (the pipeline's or a test's), nor once it has made its choice or begun a probe ([`chooser_started_at`]). Whether it did.
    pub fn start_drafts_at(&mut self, drafts: usize) -> bool {
        let held = self.drafts_held || self.pipeline.frozen_slots();
        let Some(chooser) = self.width.as_ref().and_then(|chooser| chooser_started_at(chooser, self.speculative, held, drafts)) else { return false };
        self.drafts = chooser.drafts();
        self.copy_drafts = self.drafts;
        self.width = Some(chooser);
        true
    }

    /// Tests only: take a decode pass of `rows` rows to have cost `seconds(rows)`, in place of its time (None: the clock again). The chooser
    /// of the drafts and the cut of drawn drafts then read the script's costs, with the acceptance as it was.
    pub fn script_pass_seconds(&mut self, seconds: Option<Box<dyn Fn(usize) -> f64 + Send + Sync>>) {
        self.pass_clock = seconds;
    }

    pub fn info(&self) -> Result<Value> {
        self.group.info()
    }

    // ---- operands ----

    fn projection(&self, name: &str) -> Result<Operand<'_>> {
        projection_of(&self.weights, name)
    }

    /// A projection uploaded quantized to Q8_0 when it is a HyperConnection's down or up (BF16 in the bundle: half the bytes a
    /// decode pass reads, a quarter of a row's; `STRAYDOG_HC_Q8=0` keeps it BF16) or a float matrix of the drafter's
    /// (crate::weights::drafter_quantized); as it is otherwise.
    fn q8_projection(&self, name: &str) -> Result<Operand<'_>> {
        q8_projection_of(&self.weights, name)
    }

    /// A MoE's router. Unsloth's bundles store Qwen3.8's in F32 though every value is a BF16 value (the model's own): it goes up as BF16,
    /// the same values in half the bytes a decode step reads (251.7 of a token's MB on its 48 layers), as GSQ's BF16 routers already do.
    fn router(&self, name: &str) -> Result<Operand<'_>> {
        let operand = self.projection(name)?;
        match operand.data {
            Data::Bytes(bytes) if self.weights.narrowed(name) => {
                Ok(Operand { key: format!("{name}:bf16"), data: Data::Narrow(bytes), kind: 30, rows: operand.rows, cols: operand.cols })
            }
            _ => Ok(operand),
        }
    }

    /// A vector, uploaded decoded to F32.
    fn vector(&self, name: &str, key: Option<&str>) -> Result<Operand<'_>> {
        vector_of(&self.weights, name, key)
    }

    /// A MoE's shared-expert mix as a one-row matrix of its own bytes (BF16): a decode row's router and mix are then one launch of
    /// one kind, where an F32 mix took a launch of its own.
    fn mix_row(&self, name: &str) -> Result<Operand<'_>> {
        let (rows, cols) = self.weights.vector_shape(name)?;
        let (data, kind) = self.weights.raw(name, 0, None)?;
        // (an F32 mix of BF16 values, as Unsloth stores Qwen3.8's, goes up as BF16 beside its router: Model::router)
        if rows == 1 && kind == 0 && crate::gpu::bf16_exact(data) {
            return Ok(Operand { key: format!("{name}:row"), data: Data::Narrow(data), kind: 30, rows: 1, cols: cols as u64 });
        }
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
        ple_operands_of(&self.weights)
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
            if let Data::Bytes(bytes) | Data::Decode(bytes, _) | Data::Quantize(bytes, _) | Data::Narrow(bytes) = operand.data {
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
            self.weights_mut().drop_drafter();
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
        operands.push(self.router(&format!("{prefix}ffn_gate_inp.weight"))?);
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
        operands.push(self.router(&format!("{prefix}ffn_gate_inp.weight"))?);
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
    /// engine and prefill streaming (page-locked behind "ready", see [`Model::bank_locks`]).
    fn prepare_hybrid_moe(&mut self, cpu_threads: u32) -> Result<bool> {
        if !self.group.hybrid_supported()? {
            return Ok(false);
        }
        let kinds = (0..48).map(|layer| self.weights.expert_kinds(layer)).collect::<Result<Vec<_>>>()?;
        let requirements = partial_requirements(self.weights.dense_requirements(), &kinds, self.group.prestage_slots());
        // The drafter when its layer and a verification's buffers leave the expert slots their share.
        let with_drafter = partial_drafter_requirements(requirements.clone(), self.weights.drafter_requirements());
        if !self.weights.drafter || !self.group.can_pin_model(&with_drafter, None)? {
            self.weights_mut().drop_drafter();
            self.group.drop_mtp_layer_state()?;
        }
        if !self.group.can_pin_model(&requirements, None)? {
            return Ok(false);
        }
        // (pipelined verification: an engine for each device's host thread, half the threads each)
        let pipelined = self.pipeline.windows(self.group.backends.len());
        let threads = if pipelined { self.pipeline.engine_threads(if cpu_threads == 0 { CpuExperts::default_threads()? } else { cpu_threads }) } else { cpu_threads };
        let cpu = CpuExperts::new(threads)?;
        if pipelined {
            // (the second engine's workers on the cores after the first's, not on the same ones)
            self.cpu_second = Some(CpuExperts::new_at(threads, threads)?);
            eprintln!("Straydog pipelined verification: two CPU engines of {threads} threads");
        }
        eprintln!("Straydog CPU experts: {} kernels", cpu.isa());
        // (the same files as the mapping's, checked as it is: `Weights::check_unchanged` before every generation)
        #[cfg(windows)]
        for (map, file) in self.weights.mapped_files() {
            cpu.gather_file(map, file)?;
            // (the second device's engine gathers from the files too: else its prefill gathers copy from the mapping)
            if let Some(second) = self.cpu_second.as_ref() {
                second.gather_file(map, file)?;
            }
        }
        for (layer, kinds) in kinds.iter().enumerate() {
            let prefix = format!("blk.{layer}.");
            let banks = Self::host_banks(&self.weights, layer)?;
            let mut operands = Vec::new();
            for role in ["gate", "up", "down"] {
                operands.push(self.projection(&format!("{prefix}ffn_{role}_shexp.weight"))?);
            }
            operands.push(self.router(&format!("{prefix}ffn_gate_inp.weight"))?);
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

    /// What a later start warms up with: the decode and verification rows routed to each expert of each layer, on the GPU and on the host, since the model
    /// started or its counts were last cleared ([`routes`](crate::routes)). Empty of counts for a model that holds all its experts.
    pub fn route_counts(&self) -> Result<Routes> {
        let mut routes = Routes::new(ROUTED_LAYERS, ROUTED_EXPERTS);
        for (layer, gpu, host) in self.group.route_counts(false)? {
            routes.set_layer(layer, &gpu, &host);
        }
        Ok(routes)
    }

    /// Forget the routes counted so far (the start-up's own passes say nothing of a later run's).
    pub fn clear_route_counts(&mut self) -> Result<()> {
        self.group.route_counts(true).map(drop)
    }

    /// The first step of a warm start of a partially resident model from an earlier run's counts, before its first request: the scores the slots are ranked
    /// by are set from them (so that the slots fill with the experts used most, and a request's own routes then weigh in on top), and the bank pages of the
    /// likeliest experts of layers that are not page-locked are planned for the OS, as far as the physical memory available allows less
    /// [`crate::host::left_free`]. The slots fill by [`Model::warm_fill`] and the pages are asked for by [`crate::host::prefetch`], both in the background.
    /// The pages begin at the likeliest expert, those the slots will take too: a slot is copied from its bank's pages, and a prompt streams every expert
    /// the slots do not hold yet, so the likeliest are the first that are read, by the fill or the prompt, whichever comes first.
    pub fn warm_start(&mut self, prior: &Routes) -> Result<WarmStart> {
        if !self.partial {
            return Err("a model that holds all its experts has no slots to warm up".into());
        }
        if prior.total() > 0 {
            for layer in 0..ROUTED_LAYERS {
                let rates = prior.rates(layer);
                if rates.iter().any(|&rate| rate > 0.0) {
                    self.group.seed_scores(layer, &rates)?;
                }
            }
        }
        let (slots, held) = self.group.slots()?;
        let banks = |layer: usize| -> Option<Banks> {
            // (a layer to be locked has its pages read by the locking, which a hint would only compete with)
            if self.locked_layers.contains(&layer) || self.lock_pending.contains(&layer) || (layer >= 48 && !self.weights.drafter) {
                return None;
            }
            let banks = Self::host_banks(&self.weights, layer).ok()?;
            if banks.iter().any(|bank| bank.is_empty() || bank.len() % ROUTED_EXPERTS != 0) {
                return None;
            }
            Some(Banks { start: banks.map(|bank| bank.as_ptr() as usize), bytes: banks.map(|bank| bank.len() / ROUTED_EXPERTS) })
        };
        let memory = self.weights.host_memory_bytes;
        let budget = crate::host::within_free(u64::MAX, 0, crate::host::prefetch_available_bytes(), memory);
        let hints = routes::plan_hints(&prior.ranked(), 0, budget, &banks);
        Ok(WarmStart { slots, held, swapped: self.group.swapped_experts()?, hints })
    }

    /// One bounded step of filling the expert slots by the scores a warm start set, never waiting for its copies.
    pub fn warm_fill(&mut self, max_swaps: u64) -> Result<WarmFill> {
        let (pending, slots, held) = self.group.refresh_slots(max_swaps)?;
        Ok(WarmFill { pending, slots, held, swapped: self.group.swapped_experts()? })
    }

    /// Let the GPU assist from the page-locked banks: CUDA starts at zero, HIP at the cap, unless a level is held explicitly. The plans mapped
    /// (none, and nothing is tuned, when no bank is page-locked). Called again as layers are locked, it maps those that are new at the level in force.
    fn start_assist(&mut self) -> Result<u64> {
        let cap = crate::gpu::Backend::assist_cap();
        // On CUDA the level is the most distinct experts a layer's pass may miss for the GPU to take them all over the bus (with more, the CPU
        // takes them all: staging costs the GPU about 95 us an expert, in series, where the CPU computes a layer's misses in parallel). CUDA
        // starts at CUDA_ASSIST and requires two consecutive rounds' confirmed 3% saving before adopting another level.
        // STRAYDOG_ASSIST=N holds the level at N on any build (0: every missed route to the CPU; the cap, 8, is the most). A CUDA build's MTP draft
        // layer takes none of it (hybrid_level in gpu/resident_moe.hpp).
        // (STRAYDOG_PIPELINE_ASSIST: the level the pipeline and `freeze` hold, which no tuner chooses there)
        let held = if self.pipeline.frozen_slots() { self.pipeline.assist() } else { None };
        let forced = held.or_else(|| std::env::var("STRAYDOG_ASSIST").ok().and_then(|value| value.parse::<u32>().ok())).map(|value| value.min(cap));
        // STRAYDOG_ASSIST=tune remains the automatic policy, as is an unset value.
        let fixed = forced;
        let first = fixed.unwrap_or(if cfg!(feature = "cuda") { CUDA_ASSIST.min(cap) } else { cap });
        let level = self.assist.as_ref().map_or(first, assist::Tuner::level);
        let mapped = self.group.enable_assist(level)?;
        if mapped > 0 && self.assist.is_none() {
            let levels = if fixed.is_some() { Vec::new() } else { assist::levels(cap) };
            self.assist = Some(if cfg!(feature = "cuda") { assist::Tuner::new_cuda(levels, first) } else { assist::Tuner::new(levels, first) });
        }
        self.assist_mapped = mapped;
        Ok(mapped)
    }

    /// The page locks for `layers`, to be taken one layer at a time ([`BankLocks::lock_next`]).
    fn lock_job(&self, layers: &[usize]) -> Result<BankLocks> {
        self.refresh_kv_ahead();
        let mut planned = Vec::with_capacity(layers.len());
        for &layer in layers {
            let bank = |role: &str| -> Result<crate::banks::Bank> {
                let name = format!("blk.{layer}.ffn_{role}_exps.weight");
                let bank = self.weights.mapped_bank(&name)?;
                // (the copy mode reads each bank's copy from its file)
                Ok(match self.lock_mode {
                    LockMode::Register => bank,
                    LockMode::Copy => bank.reading(self.weights.bank_file(&name)?),
                })
            };
            planned.push((layer, [bank("gate")?, bank("up")?, bank("down")?]));
        }
        let job = BankLocks::new(planned, self.group.host_ordinal(), crate::host::left_free(self.weights.host_memory_bytes))
            .tied_to(self.lock_halt.clone(), self.kv_ahead.clone()).remembering(self.lock_refusal.clone()).marking(self.registered.clone());
        // (a GPU that cannot register the read-only mapping is not asked to: each layer's turn copies its banks, and checks the copies)
        Ok(match self.lock_mode {
            LockMode::Register => job,
            LockMode::Copy => job.copying(std::sync::Arc::new(PinnedMemory { ordinal: self.group.host_ordinal() })),
        })
    }

    /// The layers a start planned to page-lock, as a plan to take layer by layer behind "ready" ([`BankLocks::lock_next`], then
    /// [`Model::banks_locked`] for each, then [`Model::banks_lock_ended`]): none for a model that has none to lock. With an earlier run's counts
    /// (`prior`) the layers the host ran most routes of come first, since locking a layer lets the GPU take some of its absent routes from the CPU;
    /// without, they keep the order the plan had (from the last layer).
    pub fn bank_locks(&mut self, prior: Option<&Routes>) -> Option<BankLocks> {
        self.locks_planned = true;
        if self.lock_pending.is_empty() || self.locking {
            return None;
        }
        if let Some(prior) = prior.filter(|prior| prior.total() > 0) {
            crate::banks::order_by_host_routes(&mut self.lock_pending, |layer| prior.host_routes(layer));
        }
        match self.lock_job(&self.lock_pending) {
            Ok(job) => {
                self.locking = true;
                Some(job)
            }
            Err(error) => {
                eprintln!("WARN Straydog host banks: no layer is page-locked ({error})");
                self.lock_pending.clear();
                None
            }
        }
    }

    /// Each GPU's ordinal, allocated bytes and budget, in the group's order.
    pub fn vram_budgets(&self) -> Result<Vec<(usize, u64, u64)>> {
        self.group.backends.iter().map(|backend| backend.memory().map(|(used, budget)| (backend.ordinal, used, budget))).collect()
    }

    /// The free memory of the group's `index`th GPU, other programs' share counted out.
    pub fn vram_free(&self, index: usize) -> Result<u64> {
        self.group.backends.get(index).ok_or("no such Straydog GPU")?.free_bytes()
    }

    /// The free memory or signed Windows process-budget headroom of this GPU.
    pub fn vram_headroom(&self, index: usize) -> Result<(i64, bool)> {
        self.group.backends.get(index).ok_or("no such Straydog GPU")?.headroom()
    }

    /// Raise the group's `index`th GPU's budget to `budget`, its expert slots placed again at what fits ([`crate::gpu::Backend::raise_budget`]):
    /// the slots before and after.
    pub fn raise_vram(&mut self, index: usize, budget: u64) -> Result<(u64, u64)> {
        self.group.backends.get_mut(index).ok_or("no such Straydog GPU")?.raise_budget(budget)
    }

    /// Resize this GPU's expert pool between requests, preserving the fixed model state.
    pub fn resize_vram(&mut self, index: usize, budget: u64) -> Result<(u64, u64)> {
        self.group.backends.get_mut(index).ok_or("no such Straydog GPU")?.resize_budget(budget)
    }

    /// The layers that memory freed since the start's plan ([`Model::plan_host_banks`]) has room for: a program that let go of its memory, or a
    /// previous worker that still held its banks when the start looked (RTX 4060, 63.6 GiB: 38 of 48 layers planned with 33.1 GiB available).
    /// Planned as the start's are, within the lock budget and what the commit left allows (`host::lockable`), each layer leaving the margin and
    /// the K/V still to grow ([`crate::banks::locking_room`]) of the memory available now; taken the same way, behind the requests. The layers
    /// the host ran most routes of (`prior`, asked only when a layer fits) come first. None while a plan is out, before the start's was, and while
    /// not one more layer fits; a line in the log says what was planned and why.
    pub fn more_bank_locks(&mut self, prior: impl FnOnce(&Model) -> Option<Routes>) -> Option<BankLocks> {
        if !self.partial || !self.locks_planned || self.locking || self.bank_failure.is_some() || self.locked_layers.len() >= 48 {
            return None;
        }
        // (the runtime refused the first layer of a plan with nothing locked: it would refuse the first of this one, and again after every request)
        match self.lock_refusal.blocks_more(self.locked_layers.len(), crate::host::available_bytes()) {
            Some(why) => {
                if !std::mem::replace(&mut self.refusal_told, true) {
                    eprintln!("Straydog host banks: no more layers are planned to be page-locked: the first layer's turn was refused with none locked ({why})");
                }
                return None;
            }
            // (the memory has grown by a layer's banks since, or nothing was refused: a refusal that comes later is told as its own)
            None => self.refusal_told = false,
        }
        let sizes = (0..48)
            .map(|layer| Self::host_banks(&self.weights, layer).map(|banks| banks.iter().map(|bank| bank.len() as u64).sum::<u64>()))
            .collect::<Result<Vec<_>>>().ok()?;
        let memory = self.weights.host_memory_bytes;
        let locked = self.locked_layers.iter().map(|&layer| sizes[layer]).sum::<u64>();
        let pinned = self.pinned_now.load(std::sync::atomic::Ordering::Relaxed);
        let by_memory = lock_budget(memory, sizes.iter().sum(), pinned).saturating_sub(locked);
        let mut budget = by_memory.min(crate::host::lockable(memory, 0));
        let mut left: Vec<usize> = (0..48).rev().filter(|layer| sizes[*layer] > 0 && !self.locked_layers.contains(layer)).collect();
        self.refresh_kv_ahead();
        let (margin, kv_ahead) = (crate::host::left_free(memory), self.kv_ahead.load(std::sync::atomic::Ordering::Relaxed));
        let available = crate::host::available_bytes();
        // (the least of them first: on most requests' ends nothing fits, and the routes are not asked for)
        let least = left.iter().map(|&layer| sizes[layer]).min()?;
        if least > budget || crate::banks::locking_room(available, least, margin, kv_ahead).is_err() {
            return None;
        }
        if let Some(prior) = prior(self).filter(|prior| prior.total() > 0) {
            crate::banks::order_by_host_routes(&mut left, |layer| prior.host_routes(layer));
        }
        let mut room = available;
        let mut layers = Vec::new();
        for layer in left {
            if sizes[layer] > budget || crate::banks::locking_room(room, sizes[layer], margin, kv_ahead).is_err() {
                break;
            }
            budget -= sizes[layer];
            room -= sizes[layer];
            layers.push(layer);
        }
        if layers.is_empty() {
            return None;
        }
        let gib = |bytes: u64| format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64);
        eprintln!("Straydog host banks: {} more of 48 layers planned to be page-locked behind the requests, {}, with {} available ({} kept free, K/V to \
                   grow {}); {} locked already", layers.len(), gib(available - room), gib(available), gib(margin), gib(kv_ahead), self.locked_layers.len());
        // (a give-back's halt was for the plan it stopped)
        self.lock_halt = Default::default();
        let job = match self.lock_job(&layers) {
            Ok(job) => job,
            Err(error) => {
                eprintln!("WARN Straydog host banks: no more layers are page-locked ({error})");
                return None;
            }
        };
        self.lock_pending = layers;
        self.locking = true;
        Some(job)
    }

    /// A layer whose banks were page-locked and checked against the file's tree ([`BankLocks::lock_next`]), taken under the model's lock: its locks
    /// are the device's from here (released when the model ends), its banks are mapped for the GPU's assist, which the tuner then times anew,
    /// and a warm start no longer plans the OS's reads of its pages. The plans the GPU can read now.
    pub fn banks_locked(&mut self, mut locked: Locked) -> Result<u64> {
        if self.lock_halt.load(std::sync::atomic::Ordering::SeqCst) {
            // (its locks were given back to a request while this layer's turn ran: the layer is let go, not taken)
            // SAFETY: the model keeps its weights mapped.
            unsafe { locked.unlock() };
            return Err("its page locks were given back to a K/V growth".into());
        }
        let layer = locked.layer;
        let copied = locked.is_copy();
        if let Some(copy) = locked.take_copy() {
            // The layer's banks are page-locked copies, checked: every device's plan of the layer reads them (one copy for all) from here, and the assist
            // below maps them. The mapping is registered nowhere.
            let banks = Self::host_banks(&self.weights, layer)?;
            self.copies.take_over(&mut self.group, banks, copy)?;
        } else {
            for bank in Self::host_banks(&self.weights, layer)? {
                self.group.adopt_host(bank);
            }
        }
        self.lock_pending.retain(|pending| *pending != layer);
        self.locked_layers.push(layer);
        self.count_in_place();
        let before = self.assist_mapped;
        let mapped = self.start_assist()?;
        // (a copy the runtime would not map for the GPU is a locked layer that assists nothing, and its prompts gather it through the CPU: say so for that
        // layer alone, the one whose turn did not add a mapped plan, since nothing else shows it)
        if copied && mapped <= before {
            eprintln!("WARN Straydog host banks: layer {layer}'s page-locked copy could not be mapped for the GPU's assist ({mapped} of {} locked layers are mapped); \
                       the GPU takes no routes from it and the CPU gathers its experts for a prompt", self.locked_layers.len());
        }
        // (what a round measured was of the layers locked then)
        if self.assist_tuning() {
            self.assist_restart()?;
        }
        Ok(mapped)
    }

    /// How many layers' banks are page-locked and mapped for the GPU's assist now.
    pub fn locked_layers(&self) -> usize {
        self.locked_layers.len()
    }

    /// The locking behind "ready" is over, with every layer taken or not (`failure`: a bank did not match the file's tree, which ends the model's
    /// service): what is left is left to the page cache, and the working set's limit, which follows what is free, is kept from here on.
    pub fn banks_lock_ended(&mut self, failure: Option<String>) {
        self.lock_pending.clear();
        self.locking = false;
        if failure.is_some() {
            self.bank_failure = failure;
        }
        self.hold_banks();
    }

    /// Count the bytes of the banks page-locked in place (the locked layers that are not held as copies) for the working set's limit.
    fn count_in_place(&mut self) {
        let bytes: u64 = self.locked_layers.iter().filter(|layer| !self.copies.holds(**layer))
            .filter_map(|&layer| Self::host_banks(&self.weights, layer).ok()).map(|banks| banks.iter().map(|bank| bank.len() as u64).sum::<u64>()).sum();
        self.in_place_now.store(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Keep the working set's limit ([`crate::host::keep_resident`]) from here on, now and as memory moves. It comes once the banks that are to be
    /// locked are: locked pages are not trimmed, but the mapped pages around them are, and a limit that came first would have trimmed what
    /// the locking reads next.
    fn hold_banks(&mut self) {
        let pinned = self.pinned_now.load(std::sync::atomic::Ordering::Relaxed);
        crate::host::keep_resident(self.weights.host_memory_bytes, pinned, self.banks_bytes, self.in_place_now.load(std::sync::atomic::Ordering::Relaxed));
        self.memory_watch.allow_quota();
    }

    /// Page-lock every layer the start planned, here and now, as the worker does behind "ready" (for what is not served: examples and tools).
    pub fn lock_banks_now(&mut self) -> Result<()> {
        let Some(mut job) = self.bank_locks(None) else { return Ok(()) };
        let mut failure = None;
        // SAFETY: the plan is of this model's own weights, which it keeps mapped.
        while let Some(next) = unsafe { job.lock_next() } {
            match next {
                Ok(locked) => {
                    self.banks_locked(locked)?;
                }
                Err(stop) => {
                    eprintln!("Straydog host banks: locking stopped because {}", stop.message());
                    failure = matches!(stop, crate::banks::Stop::Unverified { .. }).then(|| stop.message());
                    break;
                }
            }
        }
        self.banks_lock_ended(failure.clone());
        failure.map_or(Ok(()), Err)
    }

    /// Whether a prompt batch's experts beyond the slots with the fewest rows go to the CPU (on unless STRAYDOG_PROMPT_CPU=0). Off, a
    /// partially resident prompt is computed alike from run to run; on, which experts go to the CPU depends on timing.
    pub fn prompt_share(&mut self, on: bool) -> Result<()> {
        self.group.prompt_share(on)
    }

    /// Tests only: page-lock the banks of `layers` whatever the commit left says (as the start does for the layers it can) and let the GPU
    /// assist from them. The plans mapped.
    #[doc(hidden)]
    pub fn lock_layers_for_test(&mut self, layers: &[usize]) -> Result<u64> {
        // (a layer already locked: registering its banks again would fail)
        let wanted: Vec<usize> = layers.iter().copied().filter(|layer| !self.locked_layers.contains(layer)).collect();
        let mut job = self.lock_job(&wanted)?;
        // SAFETY: the plan is of this model's own weights, which it keeps mapped.
        while let Some(next) = unsafe { job.lock_next() } {
            self.banks_locked(next.map_err(|stop| stop.message())?)?;
        }
        self.start_assist()
    }

    /// Tests only: a plan for `layers` to lock the way the worker does, off the model's thread ([`BankLocks::lock_next`], then
    /// [`Model::banks_locked`]).
    #[doc(hidden)]
    pub fn lock_job_for_test(&self, layers: &[usize]) -> Result<BankLocks> {
        self.lock_job(layers)
    }

    /// Tests only: the assist level of each next decode pass of one row, the last going on, in place of the tuner's rounds. The first is in
    /// force now.
    #[doc(hidden)]
    pub fn assist_script(&mut self, levels: &[u32]) -> Result<()> {
        let (first, rest) = levels.split_first().ok_or("an empty assist script")?;
        let tuner = self.assist.as_mut().ok_or("no layer is page-locked: there is no assist to script")?;
        self.group.set_assist_level(*first)?;
        tuner.set(*first);
        let routed = self.route_totals();
        self.assist_script = Some(AssistScript { next: rest.iter().copied().collect(), level: *first, routed, trace: Vec::new() });
        Ok(())
    }

    /// Tests only: the level each scripted pass ran at, its routes that the GPU assisted, that were on the GPU, and that the CPU ran, and the
    /// seconds of its layers.
    #[doc(hidden)]
    pub fn assist_trace(&self) -> Vec<(u32, [u64; 3], f64)> {
        self.assist_script.as_ref().map_or_else(Vec::new, |script| script.trace.clone())
    }

    /// The routes the GPUs have assisted, run themselves and left to the CPU so far.
    fn route_totals(&self) -> [u64; 3] {
        let info = self.group.info().unwrap_or_default();
        ["assisted_experts", "gpu_experts", "cpu_experts"].map(|key| info[key].as_u64().unwrap_or(0))
    }

    /// A layer's mapped gate, up and down expert banks.
    fn host_banks(weights: &Weights, layer: usize) -> Result<[&[u8]; 3]> {
        let bank = |role: &str| weights.raw(&format!("blk.{layer}.ffn_{role}_exps.weight"), 0, None).map(|raw| raw.0);
        Ok([bank("gate")?, bank("up")?, bank("down")?])
    }

    /// Partial residency, once the banks are authenticated: plan which whole layers to page-lock within the lock budget, and hand absent experts to
    /// the CPU engine. The locking is not done here: it reads the banks from the drive, a third of a start on a machine that locks most of them, and
    /// the worker does it behind "ready" ([`Model::bank_locks`]). The plan is made at the point it always was, before the expert slots are placed,
    /// so that what it leaves of the commit for the locks is out of what the slots' VRAM will take.
    fn plan_host_banks(&mut self, pinned_state: u64) -> Result<()> {
        let sizes = (0..48)
            .map(|layer| Self::host_banks(&self.weights, layer).map(|banks| banks.iter().map(|bank| bank.len() as u64).sum::<u64>()))
            .collect::<Result<Vec<_>>>()?;
        let banks_total = sizes.iter().sum();
        let (vram_used, vram_budget) = self.group.memory()?;
        let vram_room = vram_budget.saturating_sub(vram_used);
        let (by_memory, by_free) = (lock_budget(self.weights.host_memory_bytes, banks_total, pinned_state),
                                    crate::host::lockable(self.weights.host_memory_bytes, vram_room));
        let budget = by_memory.min(by_free);
        // Whole layers from the last: reading went through those most recently, so they are the likeliest to be cached.
        let layers = crate::banks::layers_within(&sizes, budget);
        // What the start planned and why it planned no more, for a log that says where the CPU's experts are read from.
        let gib = |bytes: u64| if bytes == u64::MAX { "no bound".to_string() } else { format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64) };
        let planned = layers.iter().map(|layer| sizes[*layer]).sum::<u64>();
        let free = crate::host::left_free(self.weights.host_memory_bytes);
        // (on Windows the commit, which locked memory and VRAM are charged to 1 to 1, can bind before the physical memory does)
        let commit = crate::host::commit_left();
        let why = if layers.len() == 48 {
            "none left".to_string()
        } else if by_memory <= by_free {
            format!("the lock budget of {} was used", gib(by_memory))
        } else if commit.is_some_and(|(available, commit)| commit < available) {
            format!("the commit left allows {} (host::lockable: less the VRAM slots to come and {} kept free)", gib(by_free), gib(free))
        } else {
            format!("the available memory allows {} (host::lockable: less {} kept free)", gib(by_free), gib(free))
        };
        let (available, commit) = match commit {
            Some((available, commit)) => (gib(available), format!(", commit left {}", gib(commit))),
            None => (crate::host::memory_note().map_or("unknown".to_string(), |(_, available, _)| gib(available)), String::new()),
        };
        // A GPU that cannot register host memory read-only cannot page-lock the read-only mapping in place (two RTX 3090s on Windows 10: every layer's lock
        // failed and none was ever locked): its layers are copied, once, into page-locked memory and the copies are checked. Decided here for every plan.
        let devices: Vec<(usize, Option<bool>)> = self.group.ordinals().into_iter().map(|ordinal| (ordinal, crate::gpu::host_registers_read_only(ordinal))).collect();
        let (mode, copy_why) = crate::banks::lock_mode(std::env::var(crate::banks::READ_ONLY_REGISTRATION_ENV).ok().as_deref(), &devices);
        self.lock_mode = mode;
        if let Some(copy_why) = copy_why {
            eprintln!("Straydog host banks: {copy_why}, so each layer's banks are page-locked as one verified copy in portable host memory, not in the model's mapping");
        }
        eprintln!("Straydog host banks: {} of 48 layers planned to be page-locked behind ready, {} of {}; no more planned because {why}; lock_budget {}, \
                   lockable {}, host memory {}, available {available}{commit}, VRAM slots to come {}", layers.len(), gib(planned), gib(banks_total),
                  gib(by_memory), gib(by_free), gib(self.weights.host_memory_bytes), gib(vram_room));
        self.banks_bytes = banks_total;
        self.lock_pending = layers;
        self.pinned_now.store(pinned_state, std::sync::atomic::Ordering::Relaxed);
        // Paging must be observed even while a request holds up bank locking. The quota alone waits for the locks.
        crate::host::watch_working_set(self.weights.host_memory_bytes, self.pinned_now.clone(), self.banks_bytes, self.in_place_now.clone(), &self.memory_watch);
        // (with nothing to lock the limit comes now; else when the locking is over: `Model::banks_lock_ended`)
        if self.lock_pending.is_empty() {
            self.hold_banks();
        }
        // The GPU reads a few absent experts per layer over PCIe while the CPU computes the rest, up to 5 of every 12 (expert_source), from the
        // banks that are page-locked, as they are locked ([`Model::banks_locked`]). On an R9700 at 6 GiB beside a Ryzen 5 7500F: 1 route
        // a layer 30.7 tokens/s, 2 33.7, 3 35.7, 4 36.7; a higher share than 5 in 12 holds the layers up (7 in 12: 33.7). CUDA takes two,
        // staged (sd_hybrid_assist). That is the most a layer is given: how many it is worth on this machine, the engine measures in decode
        // (crate::assist), from the most at the first layer locked.
        // Speed first (STRAYDOG_EXPERT_FLOOR, a router weight from 0 to 1): the CPU's routes below it are left out.
        if let Some(floor) = expert_floor()? {
            self.group.route_floor(floor)?;
            eprintln!("Straydog speed first: the CPU's routes below a router weight of {floor} are left out");
        }
        // A prompt batch's experts beyond the slots with the fewest rows go to the CPU while the bus streams the rest (resident_moe.hpp,
        // prompt_cpu_share). STRAYDOG_PROMPT_CPU=0 streams them all (a comparison's other side).
        if std::env::var("STRAYDOG_PROMPT_CPU").is_ok_and(|value| value == "0") {
            self.group.prompt_share(false)?;
            eprintln!("Straydog prompt batches: every expert beyond the slots streams over the bus (STRAYDOG_PROMPT_CPU=0)");
        }
        let cpu = self.cpu.as_ref().ok_or("Straydog partial residency has no CPU engine")?;
        cpu.watch_paging(&crate::host::PAGING)?;
        if let Some(second) = self.cpu_second.as_ref() {
            second.watch_paging(&crate::host::PAGING)?;
            self.group.attach_cpu_for(0, cpu)?;
            return self.group.attach_cpu_for(1, second);
        }
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
        if let Some(failure) = &self.recovery_failure { return Err(failure.clone()); }
        if let Some(failure) = &self.layer_prefill_restore_failure { return Err(failure.clone()); }
        self.history.clear();
        self.final_hidden = None;
        self.group.reset_state()
    }

    /// Discard every saved sequence state after a pass whose native wait was recovered.
    fn recover_hybrid_wait(&mut self) -> Result<()> {
        let mut recovered = Ok(());
        for (_, checkpoint) in std::mem::take(&mut self.checkpoints) {
            let released = self.release_checkpoint(checkpoint);
            recovered = recovered.and(released);
        }
        if let Some(parked) = self.parked.take() {
            let released = self.discard(parked);
            recovered = recovered.and(released);
        }
        self.mtp_kept = 0;
        // These vocabulary rows may not have reached the GPU. Let a later sequence
        // note them again instead of mistaking them for rows already installed.
        for token in self.draft_pending.drain(..) { self.draft_extras.remove(&token); }
        let reset = self.reset();
        recovered = recovered.and(reset);
        if let Err(error) = recovered {
            let failure = format!("could not reset the GPU after its wait timed out: {error}; model state is unusable");
            self.recovery_failure = Some(failure.clone());
            return Err(failure);
        }
        Ok(())
    }

    /// Let the warm-up behind "ready" be stopped in the middle of a pass: while `stop` is set and true, a prompt batch ends between two layers
    /// with [`WARM_STOPPED`] (and runs in spans of one layer for it to be seen, not four), instead of at its end. A request's own prefill
    /// never does (it is read between batches, [`Model::prefill_batches`]: a stopped batch would take the held sequence's state with it), so
    /// the warm-up sets this for its passes alone and clears it (`None`) before its [`Model::reset`], which takes back what a pass stopped
    /// half way leaves: the recurrent state (zeroed), `history` (not yet extended by the batch) and the K/V rows past it (read only at positions
    /// a sequence has written, as after any reset). Experts that reached the slots stay.
    pub fn set_warm_stop(&mut self, stop: Option<&'static std::sync::atomic::AtomicBool>) {
        self.warm_stop = stop;
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
        let positions = self.save_positions(from, self.history.len(), true)?;
        if let Err(error) = self.group.checkpoint(slot, true) {
            let _ = self.group.release_positions(positions);
            return Err(error);
        }
        Ok(Parked { tokens: self.history.clone(), final_hidden: self.final_hidden.clone(), slot, history: None,
                    saved: vec![positions], covered: Some((from, self.history.len())), root: false, kept: false })
    }

    /// `group.save_positions` of `from..to`, with the room for the copy asked first: host memory (pageable, charged to the commit) for hundreds of MB to
    /// some GB. Where the system charges the commit (Windows) and what it has left is less than the copy and [`COPY_COMMIT_MARGIN`], or where it refuses the
    /// copy's memory ([`HOST_NO_MEMORY`]), locked layers are given back for it, newest first ([`Model::give_back`]), as for a K/V growth; with none left the
    /// error is [`HOST_NO_MEMORY`], which the caller survives by going without the copy.
    fn save_positions(&mut self, from: usize, to: usize, history: bool) -> Result<Positions> {
        let wanted = to.saturating_sub(from) as u64 * self.group.position_bytes() + COPY_COMMIT_MARGIN;
        loop {
            let short = crate::host::commit_left().is_some_and(|(_, left)| left < wanted);
            let refused = if short {
                HOST_NO_MEMORY.to_string()
            } else {
                match self.group.save_positions(from as u64, to as u64, history) {
                    Err(error) if error.ends_with(HOST_NO_MEMORY) => error,
                    other => return other,
                }
            };
            if self.give_back(wanted, "a copy of the earlier conversation")?.is_none() {
                return Err(refused);
            }
        }
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
                    saved: Vec::new(), covered: None, root: false, kept: false })
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
            match self.save_positions(*low, *high, false) {
                Ok(positions) => saved.push(positions),
                Err(error) => {
                    // (the coverage stays as it was: nothing writes past it)
                    for positions in saved {
                        let _ = self.group.release_positions(positions);
                    }
                    if error.ends_with(HOST_NO_MEMORY) {
                        // No memory to keep the parked sequence's positions that the prompt overwrites: it is dropped, as when its copies would outgrow the budget.
                        eprintln!("Straydog: the earlier conversation is dropped, there is no host memory to keep the part of it that this request overwrites");
                        let parked = self.parked.take().unwrap();
                        return self.discard(parked);
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
                let (root, kept) = (parked.root, parked.kept);
                let slot = if parked.slot == PARK_SLOTS[0] { PARK_SLOTS[1] } else { PARK_SLOTS[0] };
                // The live sequence whole over what the restore overwrites: from the lowest copy too
                // (the same tokens below `common`, but perhaps not the same bits once rewritten), or
                // not at all when that does not fit the budget.
                let from = common.min(parked.from());
                let swapped = if self.parkable(common) && self.parkable(from) {
                    match self.park(from, slot) {
                        Ok(swapped) => Some(swapped),
                        Err(error) if error.ends_with(HOST_NO_MEMORY) => {
                            // (no memory to keep the live sequence beside the parked one it gives way to: it is overwritten, as when it does not fit the budget)
                            eprintln!("Straydog: the live conversation is not kept, there is no host memory for a copy of it");
                            None
                        }
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
                if root {
                    // (the sequence is a system block at its root again: its checkpoint, which the prompt that made the state would have left)
                    self.save_checkpoint("root")?;
                } else if kept {
                    // (and a conversation at the place its reply began: the checkpoint the request that made the state would have left)
                    self.save_checkpoint("turn")?;
                }
                return Ok(Some(if root { "prefix" } else if kept { "kept" } else { "restored" }));
            }
        }
        // A system block's imported state that this prompt does not begin with is of no use to it, and holds its host memory until something
        // takes its place.
        if self.parked.as_ref().is_some_and(|parked| (parked.root || parked.kept) && shared(&parked.tokens, &tokens[..limit]) < parked.tokens.len()) {
            let parked = self.parked.take().unwrap();
            self.discard(parked)?;
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
            let parked = if self.park_whole { self.park(0, slot) } else { self.depart(slot) };
            match parked {
                Ok(parked) => self.parked = Some(parked),
                Err(error) if error.ends_with(HOST_NO_MEMORY) => {
                    // (no memory to keep what the prompt leaves of the live sequence: the prompt overwrites it, and no copy of it is kept)
                    eprintln!("Straydog: the live conversation is not kept, there is no host memory for a copy of it");
                    return Ok(None);
                }
                Err(error) => return Err(error),
            }
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
        ple_indices_of(&self.weights, before, tokens)
    }

    fn ple_embeddings(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let started = Instant::now();
        let rows = self.weights.rows(PLE_TABLE, &self.ple_indices(&self.history, tokens)?);
        self.ple_wait += started.elapsed().as_secs_f64();
        rows
    }

    /// Add the time since `clock` to the profile's `name` and restart the clock (with DS41_PROFILE; a partially resident model's
    /// profile without it keeps the [`decode_part`]s alone, but the clock restarts at every phase all the same).
    fn phase(&mut self, name: &'static str, clock: &mut Instant) {
        if let Some(profile) = self.profile.as_mut() {
            profile.add(name, lap(clock));
        }
    }

    /// Start reading the PLE rows of `tokens` (after `before`) as soon as they are known: the
    /// pass that decodes them then waits only for rows still on their way.
    fn prefetch_ple(&self, before: &[u32], tokens: &[u32]) -> Result<()> {
        // The PLE rows are read 4096 tokens at a time: a wider batch's go in pieces.
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
            let pinned = self.pinned_other + attention_layers(self.weights.drafter) * self.kv_capacity as u64 * kv_position_bytes();
            self.pinned_now.store(pinned, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// What the locking behind "ready" leaves room for: the host K/V of the context's first 64K tokens ([`crate::banks::KV_ROOM_POSITIONS`], all of the context
    /// if it is smaller) that is not yet grown, in bytes. Kept up for the locking's thread as the K/V grows; none where the K/V is in the VRAM.
    fn refresh_kv_ahead(&self) {
        let ahead = if self.host_kv {
            let reach = self.context.min(crate::banks::KV_ROOM_POSITIONS);
            attention_layers(self.weights.drafter) * reach.saturating_sub(self.kv_capacity) as u64 * kv_position_bytes()
        } else {
            0
        };
        self.kv_ahead.store(ahead, std::sync::atomic::Ordering::Relaxed);
    }

    /// The least the K/V grows to for `positions`: what the sequence needs and a little more, whole thousands of positions.
    fn kv_least(&self, positions: usize) -> usize {
        (positions + 4096).div_ceil(4096).saturating_mul(4096).min(self.context)
    }

    /// The bytes a growth of the K/V to `capacity` positions asks of the memory: the layers' new positions, a layer's old K/V and new one both held while
    /// it is copied, and the margin the commit keeps.
    fn kv_growth_cost(&self, capacity: usize) -> u64 {
        let position = attention_layers(self.weights.drafter) * kv_position_bytes();
        (capacity - self.kv_capacity) as u64 * position + capacity as u64 * kv_position_bytes() + KV_COMMIT_MARGIN
    }

    /// A request found no memory for `wanted` bytes of what it asks (`asked_by` says what): the page locks of the expert banks give way. The locking
    /// behind "ready" ends first if it is under way (it must not lock again what is given back, and a layer its thread has locked meanwhile is let go by
    /// [`Model::banks_locked`]); then locked layers are taken back, newest first, as many as their bytes cover what the memory available now lacks
    /// ([`crate::banks::give_back_count`]) and at least one. Each is taken out of the GPU's assist (its plan reads the bank in place no more, and the layers'
    /// graphs that held the view are captured again), its banks unregistered, and it is counted no more in [`Model::locked_layers`]: the CPU's experts and
    /// the slots read it as they read any layer that is not locked. One line in the log says what was given. None when there is nothing left to give back
    /// (no layer is locked and the locking was ended already): the request is refused as before.
    fn give_back(&mut self, wanted: u64, asked_by: &str) -> Result<Option<GaveBack>> {
        if self.locked_layers.is_empty() && self.lock_pending.is_empty() {
            return Ok(None);
        }
        self.lock_halt.store(true, std::sync::atomic::Ordering::SeqCst);
        self.lock_pending.clear();
        let available = crate::host::room_now();
        let gib = |bytes: u64| format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64);
        let mut gave = GaveBack { layers: Vec::new(), bytes: 0, left: 0, left_bytes: 0, asked_by: asked_by.into(), wanted, available };
        if self.locked_layers.is_empty() {
            // (the locking was under way and the model had taken no layer of it: ending it is all there is to give, once)
            eprintln!("Straydog host banks: the locking behind ready ended with no layer taken, for {asked_by} (it asked for {}, {} available)",
                      gib(wanted), available.map_or("not known".to_string(), gib));
            return Ok(Some(gave));
        }
        let mut sizes = Vec::with_capacity(self.locked_layers.len());
        for &layer in &self.locked_layers {
            sizes.push((layer, Self::host_banks(&self.weights, layer)?.iter().map(|bank| bank.len() as u64).sum::<u64>()));
        }
        // (what the memory available lacks of what is asked; where it is not known, all of it)
        let shortfall = available.map_or(wanted, |available| wanted.saturating_sub(available));
        let count = crate::banks::give_back_count(&sizes, shortfall);
        for &(layer, bytes) in sizes.iter().rev().take(count) {
            let banks = Self::host_banks(&self.weights, layer)?;
            // (a layer held as copies: its plans read the mapping again first, which waits for everything that reads the copies, and only then are they let
            // go: nothing reads them, by the engine's lock that this runs under and the device's waits)
            // (the mark first, and a read under way waited for: then the locks; a layer held as copies has none)
            self.registered.release(layer);
            if !self.copies.give_back(&mut self.group, layer, banks)? {
                self.group.release_host(banks)?;
            }
            self.locked_layers.retain(|locked| *locked != layer);
            gave.layers.push(layer);
            gave.bytes += bytes;
        }
        // (each layer given back took its mapped plan with it, if it had one: a count kept too high would make the next copy's turn look unmapped)
        self.assist_mapped = if self.locked_layers.is_empty() { 0 } else { self.assist_mapped.saturating_sub(gave.layers.len() as u64) };
        self.count_in_place();
        gave.left = self.locked_layers.len();
        gave.left_bytes = sizes.iter().filter(|(layer, _)| self.locked_layers.contains(layer)).map(|(_, bytes)| bytes).sum();
        if self.locked_layers.is_empty() {
            // (no bank is mapped: nothing is assisted or tuned, as before the first layer was locked)
            self.assist = None;
        } else if self.assist_tuning() {
            // (what a round measured was of the layers locked then)
            self.assist_restart()?;
        }
        eprintln!("{}", crate::banks::give_back_line(&gave));
        self.gave_back.push(gave.clone());
        Ok(Some(gave))
    }

    /// Tests only: room in the K/V for `positions`, as a request asks for it.
    #[doc(hidden)]
    pub fn ensure_kv_for_test(&mut self, positions: usize) -> Result<()> {
        self.ensure_kv(positions)
    }

    /// Tests only: the positions the K/V holds now.
    #[doc(hidden)]
    pub fn kv_positions(&self) -> usize {
        self.kv_capacity
    }

    /// Tests and measurements: the expert slots the devices have, and the experts they hold (0, 0 where there are none).
    #[doc(hidden)]
    pub fn slot_counts(&self) -> (u64, u64) {
        self.group.slots().unwrap_or((0, 0))
    }

    /// The K/V's next size for `positions`: twice over where the commit and the memory have room for that, else a quarter more, else what the
    /// sequence needs, which only the commit can refuse: a sequence at 66K locked 6 GiB for 131072 positions. The commit is read
    /// with the slots' VRAM still charged to it: they give it up for the growth and take it back after, so what the growth takes of it
    /// beyond this they would not find (IQ3_XXS at auto: an hour's conversation past 65536 ended the worker).
    fn kv_capacity_for(&self, positions: usize) -> Result<usize> {
        let doubled = (positions + 4096).next_power_of_two().min(self.context);
        let Some((free, left)) = crate::host::commit_left() else { return Ok(doubled) };
        let position = attention_layers(self.weights.drafter) * kv_position_bytes();
        let cost = |capacity: usize| self.kv_growth_cost(capacity);
        let least = self.kv_least(positions);
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
    ///
    /// A growth that finds no memory gives locked layers back and asks again ([`Model::give_back`]): the page locks are an optimisation of the CPU's
    /// experts and the K/V of a conversation is not. Only with no layer left to give back does the request fail, with [`KV_NO_MEMORY`].
    ///
    /// A K/V in VRAM grows the same way and takes its bytes from the expert slots for good ([`Model::grow_vram_kv`]).
    fn ensure_kv(&mut self, positions: usize) -> Result<()> {
        if positions <= self.kv_capacity {
            return Ok(());
        }
        let growing = Instant::now();
        if let Some(parked) = self.parked.take() {
            self.discard(parked)?;
        }
        if !self.host_kv {
            self.grow_vram_kv(positions, growing)?;
            self.after_kv_growth();
            return Ok(());
        }
        // The size a growth that has begun is for: a layer is grown or it is not, and asking again finishes what was begun (a layer grown is skipped,
        // and layers of two sizes would be left if the size were chosen anew).
        let mut begun: Option<usize> = None;
        let capacity = loop {
            let attempt = begun.map_or_else(|| self.kv_capacity_for(positions), Ok).and_then(|capacity| {
                let pinned = self.pinned_other + attention_layers(self.weights.drafter) * capacity as u64 * kv_position_bytes();
                // (the limit waits for the locking behind "ready", whose reads it would cut short)
                if self.lock_pending.is_empty() {
                    crate::host::keep_resident(self.weights.host_memory_bytes, pinned, self.banks_bytes, self.in_place_now.load(std::sync::atomic::Ordering::Relaxed));
                }
                begun = Some(capacity);
                self.group.grow_kv(capacity as u64, self.history.len() as u64).map(|()| capacity)
            });
            let refused = match attempt {
                Ok(capacity) => break capacity,
                Err(error) if error.ends_with(KV_NO_MEMORY) => error,
                Err(error) => return Err(error),
            };
            let capacity = begun.unwrap_or_else(|| self.kv_least(positions));
            if self.give_back(self.kv_growth_cost(capacity), &format!("a K/V growth to {capacity} positions"))?.is_none() {
                return Err(refused);
            }
        };
        self.kv_capacity = capacity;
        self.hold_memory();
        self.refresh_kv_ahead();
        // (what a growth costs a request: the case for growing by less than doubling)
        self.metric_add("kv_growth_seconds", growing.elapsed().as_secs_f64());
        self.after_kv_growth();
        Ok(())
    }

    /// What follows a growth of the K/V to [`Model::kv_capacity`], in host memory or in VRAM: the capacity as a request's metric, and the checkpoints'
    /// histories of the MTP layer's indexer, which keeps it after its blocks, moved to where the blocks end now.
    fn after_kv_growth(&mut self) {
        self.last_metrics.insert("kv_capacity".into(), json!(self.kv_capacity));
        for checkpoint in self.checkpoints.values_mut() {
            if let Some(history) = checkpoint.history.as_mut() {
                history.move_history(self.kv_capacity as u64);
            }
        }
    }

    /// The K/V in VRAM grown for `positions`: twice over (to the context at most), as the host K/V is where the memory allows. Every device's
    /// expert slots give up their VRAM for the layers' longer K/V and take what is left after ([`crate::gpu::Backend::grow_kv`]), so the K/V takes
    /// its bytes from the slots for good, a slot for each 1.95 MiB of IQ3_S experts, and the pool is filled again from the experts' scores
    /// (a read of the slots' experts from host memory, whose time is the growth's, the line below says it). A device with no VRAM for the longer
    /// K/V refuses the request ([`KV_NO_VRAM`]) with every state as it was: what had grown stays at its size, and a later request asks again.
    fn grow_vram_kv(&mut self, positions: usize, began: Instant) -> Result<()> {
        let capacity = (positions + 4096).next_power_of_two().min(self.context);
        let slots_before = self.group.slots().map_or(0, |(slots, _)| slots);
        self.group.grow_kv(capacity as u64, self.history.len() as u64)?;
        let (from, seconds) = (self.kv_capacity, began.elapsed().as_secs_f64());
        self.kv_capacity = capacity;
        self.metric_add("kv_growth_seconds", seconds);
        let slots_after = self.group.slots().map_or(0, |(slots, _)| slots);
        self.last_metrics.insert("kv_growth_slots".into(), json!([slots_before, slots_after]));
        eprintln!("Straydog K/V growth: {from} -> {capacity} positions in {seconds:.2} s, {slots_before} -> {slots_after} expert slots");
        Ok(())
    }

    /// Advance the model over `tokens`; the logits after the last one when asked. It runs whole: a request's cancellation is read
    /// between its batches ([`Model::prefill_batches`]), so the held sequence ends where a whole batch ended, and the next request that
    /// shares it goes on from there. (Stopped between a batch's spans, the recurrent state the batch had begun could only be reset, with
    /// everything the batches before it read: a 90K prompt read for 17 minutes on an 8 GB GPU went whole, and its next request read it all
    /// again.)
    fn forward(&mut self, tokens: &[u32], logits: bool) -> Result<Option<Vec<f32>>> {
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
            let rows = self.resident_prefill(&embeddings, tokens, stash, logits || !self.deferred)?;
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
            let embeddings = self.ple_embeddings(tokens)?;
            self.phase("step: PLE rows wait", &mut clock);
            hidden = self.with_operands(Model::ple_operands, |group, operands| {
                group.ple("blk.1.", &hidden, &embeddings, operands)
            })?;
            self.phase("step: PLE layer", &mut clock);
            let layers = Instant::now();
            for (index, (begin, end)) in self.group.decoder_spans(1).into_iter().enumerate() {
                hidden = self.group.decode_span(begin, end, &hidden, position)?;
                self.phase(["step: layers (first device)", "step: layers (second device)"][index.min(1)], &mut clock);
            }
            self.assist_pass(layers.elapsed().as_secs_f64())?;
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
            return Ok(None);
        }
        self.head_logits(last_row(&hidden), tokens.len() == 1).map(Some)
    }

    fn head_logits(&mut self, row: &[f32], step: bool) -> Result<Vec<f32>> {
        let mut clock = Instant::now();
        // (CUDA: the head's mix and projection in one wait, into the last step's allocation; elsewhere a wait and a copy to the host
        // between them)
        let mut logits = std::mem::take(&mut self.logits_spare);
        let fused = cfg!(feature = "cuda")
            && self.with_operands(Model::head_operands, |group, operands| group.decode_head(operands, row, &mut logits))?;
        if !fused {
            let mixed = self.with_operands(
                |model| Ok(model.head_operands()?[..3].to_vec()),
                |group, operands| group.hyperconnection("output_hc", row, operands[0].clone(), &operands[1..]),
            )?;
            let output = self.with_operands(|model| Ok(vec![model.projection("output.weight")?]), |group, operands| {
                group.project_many(&[(operands[0].clone(), &mixed[..])])
            })?;
            logits = output.into_iter().next().unwrap();
        }
        if step {
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
        Ok(logits)
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

    // ---- a system block's state, handed from one start of the engine to the next ----

    /// What a state of `tokens` tokens and a `hidden`-float hidden state is made of, section by section ([`crate::prefix`]).
    fn prefix_sections(&self, tokens: usize, hidden: usize) -> Vec<(u64, u64)> {
        let mut plan = vec![(crate::gpu::section_tag("tokens"), 4 * tokens as u64), (crate::gpu::section_tag("hidden"), 4 * hidden as u64)];
        plan.extend(self.group.prefix_sections(tokens as u64, self.speculative > 0));
        plan
    }

    /// A digest of the layout of this model's prefix states: the same for models that read each other's, whatever decided it (the devices and
    /// how the layers are split, the K/V's placement, the drafter's layer).
    pub fn prefix_layout(&self) -> String {
        crate::prefix::layout_digest(&self.prefix_sections(4, 0))
    }

    /// The state of the system block this model holds the root of, as bytes ([`crate::prefix`]): the tokens, the last token's hidden state,
    /// and for every device the recurrent state the root checkpoint saved, the K/V and the indexer's blocks of its positions (which the held
    /// sequence still has as the root left them), and the MTP layer's indexer history. None unless the root exists and the held sequence still
    /// begins with its tokens.
    pub fn export_prefix(&mut self) -> Result<Option<Prefix>> {
        self.export_checkpoint("root")
    }

    /// The state a checkpoint of the held sequence is, as bytes: [`export_prefix`](Self::export_prefix)'s, of the checkpoint `kind`.
    fn export_checkpoint(&mut self, kind: &str) -> Result<Option<Prefix>> {
        let Some(root) = self.checkpoints.get(kind) else { return Ok(None) };
        let tokens = root.tokens.len();
        if tokens == 0 || tokens % 4 != 0 || shared(&self.history, &root.tokens) < tokens || (self.speculative > 0) != root.history.is_some() {
            return Ok(None);
        }
        let hidden = root.final_hidden.as_deref().unwrap_or(&[]);
        let plan = self.prefix_sections(tokens, hidden.len());
        let header = crate::prefix::header(tokens, hidden.len(), &plan);
        let size = header.len() as u64 + plan.iter().map(|(_, bytes)| *bytes).sum::<u64>();
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size as usize).map_err(|_| format!("not enough host memory for {} MB of prefix state", size >> 20))?;
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&crate::prefix::token_bytes(&root.tokens));
        bytes.extend(hidden.iter().flat_map(|value| value.to_le_bytes()));
        self.group.export_prefix(root.slot, tokens as u64, root.history.as_ref(), &mut bytes)?;
        if bytes.len() as u64 != size {
            return Err(format!("Straydog prefix state is {} bytes where its layout has {size}", bytes.len()));
        }
        Ok(Some(Prefix { tokens, digest: crate::prefix::tokens_digest(&root.tokens), bytes }))
    }

    /// The checkpoint of the held conversation that is worth keeping for the next start: the deepest of the places a reply began (the last user
    /// message's, the one before's, the last tool result's) that the held sequence still begins with, at least [`PARK_MINIMUM`] tokens in. The
    /// next prompt of the same conversation begins with it, whatever the reply was.
    fn conversation_checkpoint(&self) -> Option<&'static str> {
        ["turn", "step", "prior"].into_iter()
            .filter_map(|kind| self.checkpoints.get(kind).map(|checkpoint| (kind, checkpoint)))
            .filter(|(_, checkpoint)| checkpoint.tokens.len() >= PARK_MINIMUM && shared(&self.history, &checkpoint.tokens) == checkpoint.tokens.len())
            // (a conversation with an image in it is not kept: the rows of the image are not among the tokens, and a state read back could not tell them)
            .filter(|(_, checkpoint)| checkpoint.tokens.iter().all(|token| (*token as usize) < VOCABULARY))
            .max_by_key(|(_, checkpoint)| checkpoint.tokens.len())
            .map(|(kind, _)| kind)
    }

    /// The tokens, and the bytes, of the state [`export_conversation`](Self::export_conversation) would give: None when the held conversation has
    /// no checkpoint worth keeping. For a caller that decides, before the copy is made, whether the room for it is there.
    pub fn conversation_size(&self) -> Option<(usize, u64)> {
        let checkpoint = self.checkpoints.get(self.conversation_checkpoint()?)?;
        Some((checkpoint.tokens.len(), self.state_size(checkpoint.tokens.len(), checkpoint.final_hidden.as_ref().map_or(0, Vec::len))))
    }

    /// The bytes of a state of `tokens` tokens and a `hidden`-float hidden state: its header and sections.
    pub fn state_size(&self, tokens: usize, hidden: usize) -> u64 {
        let plan = self.prefix_sections(tokens, hidden);
        crate::prefix::header(tokens, hidden, &plan).len() as u64 + plan.iter().map(|(_, bytes)| *bytes).sum::<u64>()
    }

    /// The state of the conversation the model holds, at its last checkpoint ([`crate::prefix`], the same bytes as a system block's): the tokens
    /// up to the place the last reply began, and the sequence's recurrent state, K/V and indexer's blocks there. A later start that is handed it
    /// ([`import_conversation`](Self::import_conversation)) takes the next request of the conversation from there, whatever the reply was. None
    /// when the model holds no conversation with such a checkpoint.
    pub fn export_conversation(&mut self) -> Result<Option<Prefix>> {
        match self.conversation_checkpoint() {
            Some(kind) => self.export_checkpoint(kind),
            None => Ok(None),
        }
    }

    /// What an import of a state of `tokens` tokens and a `hidden`-float hidden state needs of this model, to be taken at once, under the lock that
    /// guards the model: the layout its sections must match, the parts to fill, the K/V's capacity they are made for. The state's bytes are then
    /// decoded ([`PrefixPlan::decode`]) without the model, and [`install_prefix`](Self::install_prefix) takes the result in another moment: a request
    /// is held up by neither for more than a few milliseconds.
    pub fn prefix_plan(&self, tokens: usize, hidden: usize) -> Result<PrefixPlan> {
        let refused = |why: &str| Err(format!("Straydog prefix state refused: {why}"));
        if tokens == 0 || tokens % 4 != 0 || tokens > self.kv_capacity || tokens + 1 >= self.context {
            return refused(&format!("{tokens} tokens do not fit this model's K/V and context"));
        }
        if hidden != 0 && hidden != WIDTH {
            return refused(&format!("a hidden state of {hidden} floats"));
        }
        let (saved, kept) = self.group.prefix_templates(tokens as u64, self.speculative > 0);
        Ok(PrefixPlan { tokens, hidden, sections: self.prefix_sections(tokens, hidden), saved, kept, recurrent: self.group.recurrent_sections(),
                        capacity: self.kv_capacity })
    }

    /// Take a decoded state ([`PrefixPlan::decode`] of `bytes`) as a parked sequence: a prompt that begins with its tokens resumes from them (the
    /// request's `resumed_from` is `prefix`), and one that does not drops it. Refused, with nothing changed, if the K/V grew since the plan was made
    /// (the places of its parts changed with it). The tokens held.
    pub fn install_prefix(&mut self, image: PrefixImage, bytes: &[u8]) -> Result<usize> {
        self.install_state(image, bytes, false)
    }

    fn install_state(&mut self, image: PrefixImage, bytes: &[u8], kept: bool) -> Result<usize> {
        if image.capacity != self.kv_capacity {
            return Err("Straydog prefix state refused: the K/V grew while it was read".into());
        }
        let tokens = image.ids.len();
        // (what is parked gives way: it is a copy of a sequence this one is no longer than)
        if let Some(parked) = self.parked.take() {
            self.discard(parked)?;
        }
        self.group.write_recurrent(PARK_SLOTS[0], bytes, &image.recurrent)?;
        self.parked = Some(Parked { tokens: image.ids, final_hidden: image.final_hidden, slot: PARK_SLOTS[0], history: image.kept, saved: vec![image.saved],
                                    covered: Some((0, tokens)), root: !kept, kept });
        Ok(tokens)
    }

    /// Take a state [`export_prefix`](Self::export_prefix) made as a parked sequence, whole: [`prefix_plan`](Self::prefix_plan), decoding and
    /// [`install_prefix`](Self::install_prefix) in a row. Every size is checked against this model's own layout first, and anything that differs by a
    /// byte is refused with nothing changed.
    pub fn import_prefix(&mut self, bytes: &[u8]) -> Result<usize> {
        let parsed = crate::prefix::parse(bytes)?;
        let image = self.prefix_plan(parsed.tokens, parsed.hidden)?.decode(&parsed)?;
        self.install_prefix(image, bytes)
    }

    /// Take a state [`export_conversation`](Self::export_conversation) made, as the sequence the next request that begins with its tokens
    /// continues (its `resumed_from` is `kept`): checked against this model's own layout to the byte, and refused with nothing changed if it
    /// differs by one. The K/V first has room for the state and for `positions` (the prompt that is to go on from it), so that the request does not
    /// grow it again and drop the state with its old places. The tokens held. Whatever the model held is parked, as for any request.
    pub fn import_conversation(&mut self, bytes: &[u8], positions: usize) -> Result<usize> {
        let parsed = crate::prefix::parse(bytes)?;
        if parsed.tokens == 0 || parsed.tokens % 4 != 0 || parsed.tokens + 1 >= self.context || positions >= self.context {
            return Err(format!("Straydog prefix state refused: {} tokens do not fit this model's context", parsed.tokens));
        }
        let plan = self.prefix_plan_for(parsed.tokens, parsed.hidden, positions)?;
        let image = plan.decode(&parsed)?;
        self.install_state(image, bytes, true)
    }

    /// [`prefix_plan`](Self::prefix_plan) with the K/V grown to `positions` first, where it is short of them.
    fn prefix_plan_for(&mut self, tokens: usize, hidden: usize, positions: usize) -> Result<PrefixPlan> {
        if tokens <= self.kv_capacity && positions <= self.kv_capacity {
            return self.prefix_plan(tokens, hidden);
        }
        // (a check of what the K/V's size does not decide, before it is grown for the state)
        if hidden != 0 && hidden != WIDTH {
            return Err(format!("Straydog prefix state refused: a hidden state of {hidden} floats"));
        }
        self.ensure_kv(tokens.max(positions))?;
        self.prefix_plan(tokens, hidden)
    }

    /// Of `tokens`, how many the model holds already, in the live sequence or the parked one.
    pub fn held_prefix(&self, tokens: &[u32]) -> usize {
        let parked = self.parked.as_ref().map_or(&[][..], |parked| &parked.tokens[..]);
        shared(&self.history, tokens).max(shared(parked, tokens))
    }

    /// The error of a warm-up pass stopped between two spans ([`WARM_STOPPED`]). A span is waited for at its end, except where it is only
    /// queued (the layers in parts on two devices, a deferred prompt): with more than one device what they queued is waited for here, so that
    /// the reset after it finds every stream idle.
    fn warm_stopped(&mut self) -> String {
        if self.group.backends.len() > 1 {
            let _ = self.group.prefill_settle();
        }
        WARM_STOPPED.to_string()
    }

    /// Keep prompt rows on each GPU between spans of at most four layers (one while the warm-up can be stopped, [`Model::set_warm_stop`]);
    /// rows cross the host only at device boundaries.
    /// `stash`: the MTP layer reads every final row from the first layer's workspace, so only
    /// the last comes back; otherwise speculation takes them all.
    /// `keep_rows`: whether the last span returns rows (a deferred batch before the last does not).
    fn resident_prefill(&mut self, embeddings: &[f32], tokens: &[u32], stash: bool, keep_rows: bool) -> Result<Vec<f32>> {
        let (steps, position) = (tokens.len() as u64, self.history.len() as u64);
        // (the warm-up's pass is thrown away, and a request that comes meanwhile is seen at every layer; any other pass runs in spans of four)
        let stop = self.warm_stop;
        let stopped = || stop.is_some_and(|stop| stop.load(std::sync::atomic::Ordering::SeqCst));
        let spans = prefill_spans(&self.group.decoder_spans(1), if stop.is_some() { 1 } else { 4 });
        if stopped() {
            return Err(self.warm_stopped());
        }
        // The prompt's embeddings start the first span; later spans continue on the device, and
        // rows move GPU to GPU where the next span is on the other device.
        self.group.prefill_embed(0, embeddings)?;
        let mut rows_out = Vec::new();
        let mut previous: Option<usize> = None;
        for (index, (first, last)) in spans.into_iter().enumerate() {
            // (before the rows are handed to the next device and the PLE layer is queued, so that no half of a step is left on a stream)
            if index > 0 && stopped() {
                return Err(self.warm_stopped());
            }
            if let Some(previous) = previous {
                self.group.prefill_handoff(previous, first, steps)?;
            }
            previous = Some(last - 1);
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
        Ok(rows_out)
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
        mtp_operands_of(&self.weights)
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
        if self.group.mtp_parts() {
            self.group.mtp_prompt_layer(first as u64, following.len() as u64)?;
        }
        Ok(())
    }

    /// Fill the MTP layer's K/V for a prompt batch: pair (hidden_t, token_t+1) sits at position t. The token's row at an image's
    /// position is the image's own (llama.cpp's MTP graph takes `inp->embd` there), through the layer's prologue as any.
    fn mtp_prompt(&mut self, tokens: &[u32], rows: &[f32], previous: Option<Vec<f32>>) -> Result<()> {
        self.mtp_prompt_batched(tokens, rows, previous, 512, self.history.len())
    }

    fn mtp_prompt_batched(&mut self, tokens: &[u32], rows: &[f32], previous: Option<Vec<f32>>, batch: usize, start: usize) -> Result<()> {
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
            let mut end = (offset + batch).min(hidden.len());
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
        let Some((pending, reset)) = self.draft_vocabulary_pending() else { return Ok(()) };
        self.with_operands(|model| Ok(vec![model.projection("output.weight")?]), |group, operands| {
            group.draft_vocabulary(&operands[0], DRAFT_ROWS, &pending, reset)
        })?;
        Ok(())
    }

    /// The tokens noted since the last update and whether the vocabulary starts again from the history's (taken: noted no more), or none.
    pub(super) fn draft_vocabulary_pending(&mut self) -> Option<(Vec<u32>, bool)> {
        if self.draft_pending.is_empty() {
            return None;
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
        Some((std::mem::take(&mut self.draft_pending), reset))
    }

    /// The cut of a partially resident model's drawn draft at position `at` (1 the first), from what the row it adds costs this machine at
    /// this budget, measured: for the first, one pass with a draft costs c passes without. A draft is kept with the overlap of the two
    /// distributions, which 2319 rows of a served run put near 0.45 + 0.4 * the drawn distribution's greatest probability (0.95 past 0.9).
    /// Weighing that overlap against c - 1 at the rate a pass already earns gives the line below, and the sweeps agree, tok/s by cut (0, 0.3,
    /// 0.45, 0.6): R9700 at --vram 8, c 1.36: 67.4, 67.1, 65.6, 64.4; at --vram 5, c 1.5: 52.1, 53.4, 52.9, 52.2; RTX 5070 Ti at --vram 12,
    /// c 1.65: 66.6, 68.1 at 0.4, 69.1 at 0.6. A later draft's row costs what the pass of one row more costs over the one before it, in
    /// passes of one row (c - 1 is that for the first); until that is known it is cut as the draft before it.
    fn partial_drawn_cut(&self, at: usize) -> f32 {
        drawn_cut(&self.pass_seconds, at)
    }

    /// The cut of the draft at position `at` (1 the first): the chain of drafts ends before one whose product of probabilities falls below it.
    fn draft_cut(&self, drawn: bool, at: usize) -> f32 {
        match (drawn, self.partial) {
            (true, true) => self.partial_drawn_cut(at),
            (true, false) => DRAWN_DRAFT_CUT,
            (false, true) => PARTIAL_DRAFT_CUT,
            (false, false) => DRAFT_CUT,
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
        let cut = self.draft_cut(top > 0, 1);
        // (at a cut of 0 no pass would run alone, and what a lone row costs would never be measured again)
        if chain < cut || (self.partial && self.passes_with_draft >= PASSES_BETWEEN_LONE_ROWS && !self.pipeline.no_copy()) {
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
            if chain < self.draft_cut(top > 0, drafts.len() + 1) {
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
    /// logits, or None when cancelled. A read of [`CHUNKED_SCAN_READ`] new tokens or more runs its DeltaNet layers' recurrence in
    /// chunks of tokens on every device (the same sums in another order), a shorter read token by token.
    fn prefill(&mut self, tokens: &[u32], position: usize, plan: &[(usize, Vec<&'static str>)],
               cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize)) -> Result<Option<Option<Vec<f32>>>> {
        let new_tokens = tokens.len().saturating_sub(position);
        self.group.prompt_scan(new_tokens >= self.chunked_scan_from, new_tokens >= self.hc_fused_from)?;
        let output = self.prefill_read(tokens, position, plan, cancelled, progress);
        let cleared = self.group.prompt_scan(false, false);
        let output = output?;
        cleared?;
        Ok(output)
    }

    #[allow(clippy::type_complexity)]
    fn prefill_read(&mut self, tokens: &[u32], position: usize, plan: &[(usize, Vec<&'static str>)],
                    cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize)) -> Result<Option<Option<Vec<f32>>>> {
        // A long prompt's batches take the slots' VRAM until its end, however it ends.
        let mut batch = self.prefill_batch_size;
        // (a pass over an image's rows is one of at least two)
        if !self.images.is_empty() {
            batch = batch.max(2);
        }
        // A long read's batches are as wide as the workspaces were made (the batch asked for is the default's: a short read, an image, the warm-up
        // pass and a batch given are not widened).
        if self.wide_batch_size > batch && tokens.len() - position >= WIDE_READ_TOKENS && self.images.is_empty() && self.warm_stop.is_none() {
            batch = self.wide_batch_size;
        }
        // (a batch that rests narrow widens sooner: 256 rows at a time would take a thousand tokens in four passes)
        let threshold = if self.group.backends.len() > 1 { MULTI_GPU_LONG_PREFILL } else { LONG_PREFILL.min(self.prefill_batch_size * 2) };
        let long = self.partial && tokens.len() - position >= threshold;
        let eligible = long && self.layer_prefill && self.group.backends.len() == 1 && self.images.is_empty() && self.warm_stop.is_none();
        let layer_order = eligible && {
            self.ensure_kv(tokens.len())?;
            let ordinary = self.group.backends[0].ordinary_prefill_rows(batch as u64)? as usize;
            let (_, staging) = layer_prefill_split(position, tokens.len(), plan, layer_prefill_rows());
            let rows = self.group.layer_prefill_rows(self.layer_bank_bytes()?, staging)?;
            let pinned = self.pinned_now.load(std::sync::atomic::Ordering::Relaxed);
            let memory = self.weights.host_memory_bytes;
            // (the banks share the memory with the hidden rows a pass holds)
            let hidden = ((tokens.len() - position).min(layer_prefill_rows()) * WIDTH * 4) as u64;
            let room = layer_prefill_room(memory, pinned, crate::host::bank_room_now(pinned, self.locked_bank_bytes()?))
                .map(|room| room.saturating_sub(hidden));
            let selected = self.layer_prefill_forced
                || layer_prefill_needed(tokens.len() - position, batch, ordinary, self.banks_bytes, room, rows, LAYER_PREFILL_FITS);
            self.last_metrics.insert("layer_prefill_policy".into(), json!({"ordinary_batch_rows": ordinary, "layer_batch_rows": rows,
                                     "staging_kept": staging, "bank_bytes": self.banks_bytes, "bank_room": room, "selected": selected}));
            selected
        };
        let mut position = position;
        let mut long = long;
        let mut layer_rows = None;
        let tail_plan: Vec<(usize, Vec<&'static str>)>;
        let mut plan = plan;
        if layer_order {
            match self.try_layer_prefill(tokens, position, plan, cancelled, progress)? {
                LayerPrefill::Fallback => {},
                LayerPrefill::Cancelled => return Ok(None),
                LayerPrefill::Ready(logits) => return Ok(Some(Some(logits))),
                // (the short last segment, the ordinary way: only the checkpoints past the layer order's are left to save)
                LayerPrefill::Continue(at) => {
                    position = at;
                    long = self.partial && tokens.len() - position >= threshold;
                    layer_rows = self.last_metrics.get("prefill_batch_rows").cloned();
                    tail_plan = plan.iter().filter(|(stop, _)| *stop > at).cloned().collect();
                    plan = &tail_plan;
                }
            }
        }
        let preparing = Instant::now();
        let empty_before_prepare = long && {
            let (slots, held) = self.group.slots()?;
            slots > 0 && held == 0
        };
        if long {
            self.ensure_kv(tokens.len())?;
            if let Some(rows) = self.group.begin_long_prefill(batch as u64)? {
                batch = batch.max(rows as usize);
            }
        }
        self.metric_add("prefill_prepare_seconds", preparing.elapsed().as_secs_f64());
        match layer_rows {
            Some(rows) => {
                self.last_metrics.insert("prefill_batch_rows".into(), rows);
                self.last_metrics.insert("prefill_tail_tokens".into(), json!(tokens.len() - position));
            }
            None => { self.last_metrics.insert("prefill_batch_rows".into(), json!(batch)); }
        }
        let output = self.prefill_batches(tokens, position, plan, batch, empty_before_prepare, cancelled, progress);
        let restoring = Instant::now();
        let restored = if long { self.group.end_long_prefill() } else { Ok(()) };
        self.metric_add("prefill_restore_seconds", restoring.elapsed().as_secs_f64());
        match (output, restored) {
            (Err(error), Err(restore)) => Err(format!("{error}; restoring expert slots failed: {restore}")),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(output), Ok(())) => Ok(output),
        }
    }

    /// Development/validation: force the traversal order for identical-prompt comparisons, bypassing the automatic cost guard.
    #[doc(hidden)]
    pub fn layer_prefill(&mut self, enabled: bool) {
        self.layer_prefill = enabled;
        self.layer_prefill_forced = enabled;
    }

    /// Whether a layer-order pass holds its hidden rows in VRAM where they fit (a test compares the two ways).
    pub fn layer_rows_in_vram(&mut self, held: bool) {
        self.layer_rows_vram = held;
    }

    /// The read length (new tokens) from which the DeltaNet layers scan in chunks: `usize::MAX` for never, 0 for always (a test compares
    /// the two scans on a prompt shorter than the rule's).
    #[doc(hidden)]
    pub fn chunked_scan_from(&mut self, tokens: usize) {
        self.chunked_scan_from = tokens;
    }

    /// The read length (new tokens) from which the HyperConnections' up projection and finish are one kernel: `usize::MAX` for never, 0 for always.
    #[doc(hidden)]
    pub fn hc_fused_from(&mut self, tokens: usize) {
        self.hc_fused_from = tokens;
    }

    #[allow(clippy::too_many_arguments)]
    /// The bytes of the banks page-locked so far.
    fn locked_bank_bytes(&self) -> Result<u64> {
        let mut bytes = 0;
        for &layer in &self.locked_layers {
            bytes += Self::host_banks(&self.weights, layer)?.iter().map(|bank| bank.len() as u64).sum::<u64>();
        }
        Ok(bytes)
    }

    /// The layer-order bank pool: the largest layer's three routed banks (the drafter's included).
    fn layer_bank_bytes(&self) -> Result<u64> {
        let mut bank_bytes = 0;
        for layer in 0..48 + usize::from(self.speculative > 0) {
            let bytes = Self::host_banks(&self.weights, layer)?.iter().map(|bank| bank.len() as u64).sum::<u64>();
            bank_bytes = bank_bytes.max(bytes);
        }
        Ok(bank_bytes)
    }

    fn try_layer_prefill(&mut self, tokens: &[u32], position: usize, plan: &[(usize, Vec<&'static str>)],
                         cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize)) -> Result<LayerPrefill> {
        if tokens.iter().any(|token| *token as usize >= VOCABULARY) { return Err("Straydog token or context is out of range".into()); }
        let _trace_total = crate::layer_trace::Span::new("total", None, position, tokens.len(), 0);
        // (the K/V grown to the prompt first: its growth asks the commit that the hidden rows would take, more of it at more rows a pass)
        self.ensure_kv(tokens.len())?;
        let segment = layer_prefill_rows();
        let count = (tokens.len() - position).min(segment);
        let Some(mut hidden) = crate::gpu::zeroed_floats(count * WIDTH) else { return Ok(LayerPrefill::Fallback) };
        let bank_bytes = self.layer_bank_bytes()?;
        let (layer_end, staging) = layer_prefill_split(position, tokens.len(), plan, segment);
        let preparing = Instant::now();
        let trace_prepare = crate::layer_trace::Span::new("prepare", None, position, tokens.len(), 0);
        self.group.lend_prefill_head()?;
        let output: Result<Option<bool>> = (|| {
            let Some(rows) = self.group.begin_layer_prefill(bank_bytes, staging)? else { return Ok(None) };
            self.last_metrics.insert("prefill_order".into(), json!("layer"));
            self.last_metrics.insert("prefill_batch_rows".into(), json!(rows));
            self.last_metrics.insert("prefill_layer_rows".into(), json!(segment));
            self.last_metrics.insert("prefill_layer_bank_bytes".into(), json!(bank_bytes));
            self.metric_add("prefill_prepare_seconds", preparing.elapsed().as_secs_f64());
            drop(trace_prepare);
            let _active = LayerPrefillActive::begin();
            // (the pass's hidden rows in VRAM where they fit beside what layer order holds; else on the host, and the line says why)
            let host = if self.layer_rows_vram { self.group.hold_layer_rows(count as u64)? } else { Some("STRAYDOG_LAYER_ROWS_VRAM=0".to_string()) };
            match &host {
                Some(reason) => { self.last_metrics.insert("prefill_rows_host_reason".into(), json!(reason)); }
                None => { self.last_metrics.insert("prefill_rows_vram".into(), json!(true)); }
            }
            let moved = self.group.prefill_row_transfers().ok();
            let done = self.layer_prefill_segments(tokens, position, layer_end, staging, plan, rows as usize, &mut hidden, host.is_none(),
                                                   cancelled, progress);
            // (the hidden rows to the GPU and back, every layer of every pass: what holding them in VRAM would save)
            if let (Some(before), Ok(after)) = (moved, self.group.prefill_row_transfers()) {
                self.metric_add("prefill_rows_up_seconds", after[0].saturating_sub(before[0]) as f64 / 1e9);
                self.metric_add("prefill_rows_up_bytes", after[1].saturating_sub(before[1]) as f64);
                self.metric_add("prefill_rows_down_seconds", after[2].saturating_sub(before[2]) as f64 / 1e9);
                self.metric_add("prefill_rows_down_bytes", after[3].saturating_sub(before[3]) as f64);
            }
            done.map(Some)
        })();
        let restoring = Instant::now();
        let trace_restore = crate::layer_trace::Span::new("restore", None, position, tokens.len(), 0);
        // Always attempt every restoration step. The head must be back before the cache takes the remaining VRAM.
        let bank = self.group.finish_layer_prefill();
        let head = self.with_operands(Model::head_operands, Group::pin);
        let cache = self.group.end_long_prefill();
        self.metric_add("prefill_restore_seconds", restoring.elapsed().as_secs_f64());
        drop(trace_restore);
        let errors: Vec<String> = [bank, head, cache].into_iter().filter_map(Result::err).collect();
        if !errors.is_empty() {
            let restore = errors.join("; ");
            let failure = match output { Err(error) => format!("{error}; restoring layer prefill failed: {restore}"),
                                         _ => format!("restoring layer prefill failed: {restore}") };
            self.layer_prefill_restore_failure = Some(failure.clone());
            return Err(failure);
        }
        match output? {
            None => Ok(LayerPrefill::Fallback),
            Some(false) => Ok(LayerPrefill::Cancelled),
            Some(true) if layer_end < tokens.len() => Ok(LayerPrefill::Continue(layer_end)),
            Some(true) => {
                let row = self.final_hidden.clone().ok_or("layer prefill returned no final hidden row")?;
                self.head_logits(&row, false).map(LayerPrefill::Ready)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// The segments of `position..end` ([`layer_prefill_split`]'s end); without `staging` the pool has none, and a segment that would run
    /// the ordinary way is refused.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn layer_prefill_segments(&mut self, tokens: &[u32], position: usize, end: usize, staging: bool, plan: &[(usize, Vec<&'static str>)],
                              batch: usize, hidden: &mut [f32], held: bool, cancelled: &dyn Fn() -> bool,
                              progress: &mut dyn FnMut(usize, usize)) -> Result<bool> {
        let checkpoints: Vec<usize> = plan.iter().map(|(at, _)| *at).collect();
        let layers = 48 + usize::from(self.speculative > 0);
        let mut bank_reader = crate::layer_bank::Prefetch::new(&self.weights, layers, self.registered.clone());
        for stop in layer_prefill_stops(position, end, &checkpoints, layer_prefill_rows()) {
            if cancelled() { return Ok(false); }
            let begin = self.history.len();
            let count = stop - begin;
            let _trace_segment = crate::layer_trace::Span::new("segment", None, begin, stop, 0);
            self.group.clear_layer_bank()?;
            if count < LAYER_PREFILL_MIN && !staging {
                return Err(format!("a layer-order segment of {count} tokens has no staging slots to run the ordinary way"));
            }
            if count < LAYER_PREFILL_MIN {
                let _trace_tail = crate::layer_trace::Span::new("ordinary_tail", None, begin, stop, 0);
                for first in (begin..stop).step_by(batch) {
                    self.forward(&tokens[first..(first + batch).min(stop)], false)?;
                }
            } else {
                // (a pass through every layer, reading each one's banks)
                self.metric_count("prefill_layer_passes", 1);
                self.set_rope(begin, count)?;
                let rows = &mut hidden[..count * WIDTH];
                if let Some(reader) = &mut bank_reader { reader.start(0); }
                let trace_embedding = crate::layer_trace::Span::new("embedding", None, begin, stop, 0);
                for first in (begin..stop).step_by(batch) {
                    let end = (first + batch).min(stop);
                    let embedded = self.rows_at(first, &tokens[first..end])?;
                    for (row, embedded) in rows[(first - begin) * WIDTH..(end - begin) * WIDTH].chunks_exact_mut(WIDTH)
                        .zip(embedded.chunks_exact(HIDDEN)) {
                        for stream in row.chunks_exact_mut(HIDDEN) { stream.copy_from_slice(embedded); }
                    }
                }
                drop(trace_embedding);
                // (held in VRAM: the pass's rows go up once, and come back once after its last layer)
                if held {
                    self.group.put_layer_rows(0, rows)?;
                }
                let previous = self.final_hidden.clone();
                for layer in 0..48 {
                    let bank_bytes = if cfg!(feature = "prefill-observer") { Self::host_banks(&self.weights, layer)?.iter().map(|b| b.len() as u64).sum() } else { 0 };
                    let trace_bank = crate::layer_trace::Span::new("bank", Some(layer), begin, stop, bank_bytes);
                    self.load_prefill_bank_copy(layer, layers, &mut bank_reader)?;
                    drop(trace_bank);
                    let _trace_compute = crate::layer_trace::Span::new("compute", Some(layer), begin, stop, 0);
                    // Avoid a final prefill call of one row; the existing one-row path is a different API.
                    for (first, end) in layer_prefill_chunks(begin, stop, batch) {
                        let low = (first - begin) * WIDTH;
                        let high = (end - begin) * WIDTH;
                        if held {
                            self.group.prefill_rows_held(layer, (first - begin) as u64, (end - first) as u64)?;
                        } else {
                            self.group.prefill_rows(layer, &rows[low..high])?;
                        }
                        if layer == 1 {
                            let _trace_ple = crate::layer_trace::Span::new("ple", Some(layer), first, end, 0);
                            let next = (end + batch).min(stop);
                            self.prefetch_ple(&tokens[..end], &tokens[end..next])?;
                            self.prefetch_ple(&tokens[..first], &tokens[first..end])?;
                            let reading = Instant::now();
                            let embeddings = self.weights.rows(PLE_TABLE, &self.ple_indices(&tokens[..first], &tokens[first..end])?)?;
                            self.ple_wait += reading.elapsed().as_secs_f64();
                            self.with_operands(Model::ple_operands, |group, operands| group.prefill_ple("blk.1.", &embeddings, operands))?;
                        }
                        if held {
                            self.group.prefill_span_held(layer, (first - begin) as u64, first as u64, (end - first) as u64)?;
                        } else {
                            // (in place: a new vector a chunk faulted every row's pages in again for each layer)
                            self.group.prefill_layers_into(layer, layer + 1, None, first as u64, (end - first) as u64, &mut rows[low..high])?;
                        }
                    }
                }
                if held {
                    self.group.get_layer_rows(0, rows)?;
                }
                if self.speculative > 0 {
                    let bank_bytes = if cfg!(feature = "prefill-observer") { Self::host_banks(&self.weights, 48)?.iter().map(|b| b.len() as u64).sum() } else { 0 };
                    let trace_bank = crate::layer_trace::Span::new("bank", Some(48), begin, stop, bank_bytes);
                    self.load_prefill_bank_copy(48, layers, &mut bank_reader)?;
                    drop(trace_bank);
                    let _trace_compute = crate::layer_trace::Span::new("compute", Some(48), begin, stop, 0);
                    // Preserve the ordinary chunk order's pair positions and prologue batch shapes. The bank stays loaded once.
                    let mut previous = previous;
                    for (first, end) in layer_prefill_chunks(begin, stop, batch) {
                        let chunk = &rows[(first - begin) * WIDTH..(end - begin) * WIDTH];
                        self.mtp_prompt_batched(&tokens[first..end], chunk, previous, batch.min(512), first)?;
                        previous = Some(last_row(chunk).to_vec());
                    }
                }
                self.final_hidden = Some(last_row(rows).to_vec());
                self.history.extend_from_slice(&tokens[begin..stop]);
                self.draft_note(&tokens[begin..stop]);
            }
            self.group.clear_layer_bank()?;
            if let Some((_, kinds)) = plan.iter().find(|(at, _)| *at == stop) {
                let _trace_checkpoint = crate::layer_trace::Span::new("checkpoint", None, begin, stop, 0);
                for kind in kinds.clone() { self.save_checkpoint(kind)?; }
            }
            progress(stop, tokens.len());
            if cancelled() { return Ok(false); }
        }
        Ok(true)
    }

    fn load_prefill_bank_copy(&mut self, layer: usize, layers: usize, reader: &mut Option<crate::layer_bank::Prefetch>) -> Result<()> {
        let waiting = Instant::now();
        let mut copy = reader.as_mut().map(|reader| reader.take(layer)).transpose()?.flatten();
        self.metric_add("prefill_bank_wait_seconds", waiting.elapsed().as_secs_f64());
        // (a layer registered in place: the read holds the registration until the load's copies are done, which it waits for)
        let held = if copy.is_none() { self.registered.read(layer) } else { None };
        match bank_source(copy.is_some(), held.is_some(), reader.is_some()) {
            BankSource::Copy => {}
            // (the mapping's pages, which its locks keep checked, not the file's again)
            BankSource::Registered => {
                self.metric_add("prefill_bank_registered_layers", 1.0);
                self.group.load_layer_bank(layer)?;
            }
            // (passed by while it was registered, and given up since: its pages are no longer kept checked, so a checked copy now)
            BankSource::ReadNow => {
                let reading = Instant::now();
                copy = reader.as_mut().map(|reader| reader.read_now(layer)).transpose()?.flatten();
                self.metric_add("prefill_bank_wait_seconds", reading.elapsed().as_secs_f64());
                self.metric_add("prefill_bank_read_late_layers", 1.0);
                if copy.is_none() {
                    self.group.load_layer_bank(layer)?;
                }
            }
            BankSource::Mapping => self.group.load_layer_bank(layer)?,
        }
        drop(held);
        if let Some(copy) = &copy {
            self.group.load_layer_bank_from(layer, copy.sources())?;
            self.metric_add("prefill_bank_copied_layers", 1.0);
        }
        if let Some(reader) = reader {
            // The GPU now owns the entire bank: the same host buffer can receive the
            // next layer while this layer computes, without keeping two host copies.
            reader.recycle(copy);
            if layer + 1 < layers { reader.start(layer + 1); }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_batches(&mut self, tokens: &[u32], mut position: usize, plan: &[(usize, Vec<&'static str>)], batch: usize,
                       empty_before_prepare: bool, cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize)) -> Result<Option<Option<Vec<f32>>>> {
        let mut output = None;
        let beginning = position;
        let mut fill_empty_slots = if self.partial && !self.deferred && tokens.len() - position > batch {
            let (slots, held) = self.group.slots()?;
            slots > 0 && held < slots && (held == 0 || empty_before_prepare)
        } else { false };
        let stops: Vec<usize> = plan.iter().map(|(at, _)| *at).chain([tokens.len()]).collect();
        for stop in stops {
            // (Both devices at once where the segment has batches to overlap: the checkpoints between segments, and the end of the read, drain it. A read
            // that fills empty slots takes its first batch alone and fills them (below), and the batches after it overlap: the cold first request of a
            // process was read serially to its end.)
            let mut declined = false;
            let eligible = self.overlap_possible();
            while position < stop {
                if cancelled() {
                    return Ok(None);
                }
                if !fill_empty_slots && !declined && self.history.len() == position {
                    // (eligibility first: where the overlap cannot be taken the list of what remains is not built at each step)
                    if let Some(overlapped) = overlap_segment(eligible, position, stop, batch) {
                        match self.prefill_overlapped(tokens, &overlapped, batch, cancelled, progress)? {
                            Overlapped::Cancelled => return Ok(None),
                            Overlapped::Declined => declined = true,
                            Overlapped::Read(logits) => {
                                position = overlapped.last().map_or(position, |(_, end)| *end);
                                if logits.is_some() {
                                    output = logits;
                                }
                            }
                        }
                        if cancelled() {
                            return Ok(None);
                        }
                        continue;
                    }
                }
                let mut end = (position + batch).min(stop);
                // (none is left to a batch of one row, which may be an image's or end a block begun in one)
                if !self.images.is_empty() && stop - end == 1 && end - position > 2 {
                    end -= 1;
                }
                // The next batch's PLE rows arrive while this one runs.
                let next = (end + batch).min(tokens.len());
                self.prefetch_ple(&tokens[..end], &tokens[end..next])?;
                output = self.forward(&tokens[position..end], end == tokens.len())?;
                self.metric_count("prefill_batches", 1);
                if cancelled() {
                    return Ok(None);
                }
                position = end;
                progress(end, tokens.len());
                if fill_empty_slots && position < tokens.len() && position - beginning >= batch && !cancelled() {
                    // Borrowing may refill only some slots from an interrupted warm-up
                    // although the request began with an empty cache. Learn from a batch before
                    // filling the rest. A cache already populated before preparation, or filled
                    // completely by it, keeps its existing policy.
                    let filling = Instant::now();
                    self.group.rebalance(0)?;
                    self.metric_add("prefill_cache_fill_seconds", filling.elapsed().as_secs_f64());
                    fill_empty_slots = false;
                }
            }
            if let Some((_, kinds)) = plan.iter().find(|(at, _)| *at == stop) {
                for kind in kinds.clone() {
                    self.save_checkpoint(kind)?;
                }
            }
        }
        Ok(Some(output))
    }

    /// Whether a partially resident read may take both devices' layers at once (see [`Model::prefill_overlapped`]): two NVIDIA GPUs, a text prompt,
    /// no warm-up that a request may stop, no deferred pass, and STRAYDOG_PREFILL_OVERLAP not 0.
    fn overlap_possible(&self) -> bool {
        prefill_overlap() && self.partial && !self.group.parts() && self.group.backends.len() == 2 && self.images.is_empty()
            && self.warm_stop.is_none() && !self.deferred && overlap_split_ok(self.group.split())
    }

    /// A read of `batches` through both devices at the same time: the first device runs the first layers of a batch and hands its rows to the
    /// second through host memory, which runs the rest (and the MTP layer's pairs) while the first device already has the next batch. Each device
    /// still runs its own layers in order, one batch at a time, with the kernels, rows and sums of the serial path; only which batch a device
    /// holds meanwhile differs. The caller drains it (this returns after the second device has finished every batch) where the serial code must
    /// run: a checkpoint, the end of the read, a cancellation. The rows between the devices travel in two buffers made once for the segment and
    /// handed back and forth. Where the buffers cannot be had the read is declined and the caller reads serially.
    #[allow(clippy::type_complexity)]
    fn prefill_overlapped(&mut self, tokens: &[u32], batches: &[(usize, usize)], batch: usize, cancelled: &dyn Fn() -> bool,
                          progress: &mut dyn FnMut(usize, usize)) -> Result<Overlapped> {
        let (position, stop) = (batches[0].0, batches[batches.len() - 1].1);
        if stop > self.context || tokens[position..stop].iter().any(|token| *token as usize >= VOCABULARY) {
            return Err("Straydog token or context is out of range".into());
        }
        let speculative = self.speculative > 0;
        let widest = batches.iter().map(|(first, end)| end - first).max().unwrap_or(0);
        let (Some(first_buffer), Some(second_buffer)) = (crate::gpu::zeroed_floats(widest * WIDTH), crate::gpu::zeroed_floats(widest * WIDTH)) else {
            return Ok(Overlapped::Declined);
        };
        let began = Instant::now();
        // Everything the batches write lies in `position..stop` (see `forward`): asked for once, before the second thread starts.
        self.set_rope(position, stop - position)?;
        self.ensure_kv(stop)?;
        self.protect(position.saturating_sub(usize::from(speculative)), stop)?;
        let split = self.group.split().ok_or("Straydog overlapped read: no layer split")?;
        let spans = prefill_spans(&self.group.decoder_spans(1), 4);
        let (spans_a, spans_b): (Vec<(usize, (usize, usize))>, Vec<(usize, (usize, usize))>) =
            spans.iter().copied().enumerate().partition(|(_, (first, _))| *first < split);
        if spans_a.is_empty() || spans_b.is_empty() || spans_a.iter().any(|(_, (_, last))| *last > split) {
            return Err("Straydog overlapped read: the spans do not follow the split".into());
        }
        let completed = std::sync::atomic::AtomicUsize::new(position);
        let before = self.final_hidden.clone();
        // The borrows are split before the threads start, each part by itself: the weights shared (the second device's thread decodes the MTP pairs'
        // token rows and a third reads each batch's rows ahead, both with `&Weights`, and the operands of the PLE layer and the MTP layer are
        // borrowed from them: they live no longer than the borrow, as the compiler sees), the two devices each as a `&mut` of its own (this thread
        // takes the first, the second's thread the second), and nothing of the Model else is touched until the scope has ended: what the first
        // thread would have noted in it (the history, the draft vocabulary's tokens, the metrics) it keeps by the way and notes after.
        let weights: &Weights = &self.weights;
        let mtp_operands = if speculative { mtp_operands_of(weights)? } else { Vec::new() };
        let ple_operands = ple_operands_of(weights)?;
        let (first_device, second_device) = self.group.backend_pair()?;
        let mut cut = false;
        let mut failure: Option<String> = None;
        let mut first_half = StageSeconds::default();
        let mut dispatched = position;
        let outcome: Result<(Option<Vec<f32>>, StageSeconds)> = std::thread::scope(|scope| {
            let (sender, receiver) = std::sync::mpsc::sync_channel::<(usize, usize, Vec<f32>, Arc<Vec<f32>>)>(1);
            let (give, take) = std::sync::mpsc::channel::<Vec<f32>>();
            let _ = give.send(first_buffer);
            let _ = give.send(second_buffer);
            let (spans_b, mtp_operands, completed, ple_operands, spans_a) = (&spans_b, &mtp_operands, &completed, &ple_operands, &spans_a);
            let worker = scope.spawn(move || -> Result<(Option<Vec<f32>>, StageSeconds)> {
                let backend = second_device;
                let mut previous = before;
                // (the rows the MTP layer's device has kept from the batch before, for the pair that crosses two batches)
                let mut carried = 0u64;
                let mut times = StageSeconds::default();
                let mut idle = Instant::now();
                for (start, end, rows, token_rows) in receiver {
                    times.idle += idle.elapsed().as_secs_f64();
                    let busy = Instant::now();
                    let batch_tokens = &tokens[start..end];
                    let steps = batch_tokens.len();
                    // The rows go up and their buffer goes back to the first device's thread at once: the upload has copied them out.
                    backend.prefill_rows(spans_b[0].1 .0, &rows[..steps * WIDTH])?;
                    let _ = give.send(rows);
                    let mut final_row = Vec::new();
                    for (_, (first, last)) in spans_b.iter() {
                        // (the last row of the last span back: the MTP layer reads the rows from the device)
                        let wanted = u64::from(*last == 48);
                        let output = backend.prefill_layers(*first, *last, None, start as u64, steps as u64, wanted)?;
                        if wanted > 0 {
                            final_row = output;
                        }
                    }
                    times.second += busy.elapsed().as_secs_f64();
                    if speculative {
                        let pairing = Instant::now();
                        let host = if carried == 0 { previous.as_deref().filter(|_| start > 0) } else { None };
                        mtp_prompt_pairs(backend, mtp_operands, &token_rows, batch_tokens, start, host, carried)?;
                        carried = steps as u64;
                        times.mtp += pairing.elapsed().as_secs_f64();
                    }
                    previous = Some(final_row);
                    completed.store(end, std::sync::atomic::Ordering::Release);
                    idle = Instant::now();
                }
                Ok((previous, times))
            });
            // A thread reads each batch's first-layer rows and PLE rows ahead (one batch queued), where the first device's thread read them between
            // its batches: the device waited for them, and for the batch's PLE rows beside the first layer.
            let (prepared_send, prepared) = std::sync::mpsc::sync_channel::<Result<(Arc<Vec<f32>>, Vec<f32>)>>(1);
            scope.spawn(move || {
                for &(start, end) in batches {
                    let next = (end + batch).min(tokens.len());
                    if prepared_send.send(prepare_batch(weights, tokens, start, end, next)).is_err() {
                        break;
                    }
                }
            });
            for &(start, end) in batches {
                if cancelled() {
                    cut = true;
                    break;
                }
                let waiting = Instant::now();
                let Ok(mut rows) = take.recv() else { break };
                first_half.blocked += waiting.elapsed().as_secs_f64();
                let busy = Instant::now();
                let mut token_rows = None;
                let staged = prepared.recv().map_err(|_| "the overlapped read's batch preparation ended early".to_string()).and_then(|read| read)
                    .and_then(|read| {
                        token_rows = Some(read.0.clone());
                        overlap_first_half(first_device, ple_operands, tokens, start, end, spans_a, &mut rows[..(end - start) * WIDTH], read, &mut first_half)
                    });
                first_half.first += busy.elapsed().as_secs_f64();
                match staged {
                    Ok(()) => {
                        dispatched = end;
                        let waiting = Instant::now();
                        let Some(token_rows) = token_rows else { break };
                        if sender.send((start, end, rows, token_rows)).is_err() {
                            break;
                        }
                        first_half.blocked += waiting.elapsed().as_secs_f64();
                        progress(completed.load(std::sync::atomic::Ordering::Acquire), tokens.len());
                    }
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
            drop(sender);
            drop(prepared);
            worker.join().unwrap_or_else(|_| Err("a Straydog stage thread panicked".into()))
        });
        // (the batches the first device took are in the history, whatever became of the rest: as the serial path has them after each batch)
        self.history.extend_from_slice(&tokens[position..dispatched]);
        self.draft_note(&tokens[position..dispatched]);
        self.metric_count("prefill_batches", batches.iter().filter(|(_, end)| *end <= dispatched).count() as u64);
        let (row, second_half) = finish_device_work(outcome, failure.map_or(Ok(()), Err))?;
        self.metric_count("prefill_overlap_batches", batches.len() as u64);
        self.metric_add("prefill_overlap_seconds", began.elapsed().as_secs_f64());
        self.metric_add("prefill_overlap_first_seconds", first_half.first);
        self.metric_add("prefill_overlap_first_blocked_seconds", first_half.blocked);
        self.metric_add("prefill_overlap_first_prep_seconds", first_half.prep);
        self.metric_add("prefill_overlap_first_ple_seconds", first_half.ple);
        self.metric_add("prefill_overlap_first_span_calls_seconds", first_half.span_calls);
        self.metric_add("prefill_overlap_first_last_span_seconds", first_half.last_span);
        self.metric_add("prefill_overlap_second_seconds", second_half.second);
        self.metric_add("prefill_overlap_second_mtp_seconds", second_half.mtp);
        self.metric_add("prefill_overlap_second_idle_seconds", second_half.idle);
        if row.is_some() {
            self.final_hidden = row.clone();
        }
        // (a cancel that came while the second device drained the last batches and its MTP pairs is the request's: the serial path looks right after each batch)
        if overlap_stopped(cut, cancelled) {
            return Ok(Overlapped::Cancelled);
        }
        progress(completed.load(std::sync::atomic::Ordering::Acquire), tokens.len());
        match row {
            Some(row) if stop == tokens.len() => Ok(Overlapped::Read(Some(self.head_logits(&row, false)?))),
            _ => Ok(Overlapped::Read(None)),
        }
    }

    /// Whether a prompt's batches overlap on two GPUs: the first device starts batch n+1 on its
    /// layers while the second finishes batch n on its own, nothing waiting on the host between
    /// (with experts split, each also helps the other's batch on its helper stream).
    fn pipelined_prefill(&self, tokens: usize) -> Result<bool> {
        Ok(self.images.is_empty() && self.resident && (self.speculative == 0 || self.group.keeps_mtp_rows()) && self.group.backends.len() == 2
           && tokens > self.prefill_batch_size)
    }

    /// The rows through every layer and the head: with `sampling`, each row's logits for the draw, or its `top` greatest
    /// when the draw can use those (Sampling::target_read_top: penalty candidates still need a host-side completeness proof).
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
        let (first, embeddings, rows) = self.verify_rows(tokens, true)?;
        let mut clock = Instant::now();
        // (STRAYDOG_PIPELINE: the head on the second GPU alone, as the pipelined passes run it, so that two runs are the same arithmetic)
        let alone = self.frozen_now() && self.group.backends.len() == 2;
        let (targets, scores, hidden) = match rows {
            Some(hidden) => {
                let (targets, scores, _) = self.with_operands(Model::head_operands, |group, operands| {
                    if alone { group.output_head_alone(operands, &hidden, sampling, top, false) } else { group.output_head(operands, &hidden, sampling, top, false) }
                })?;
                (targets, scores, hidden)
            }
            // (the last span left its rows in its device's workspace: the head reads them there, and hands them back with its answer)
            None => {
                let ((targets, scores, _), hidden) = self.with_operands(Model::head_operands, |group, operands| {
                    if alone {
                        group.output_head_chained_alone(operands, 47, tokens.len(), sampling, top, false)
                    } else {
                        group.output_head_chained(operands, 47, tokens.len(), sampling, top, false)
                    }
                })?;
                (targets, scores, hidden)
            }
        };
        self.phase("verify: head", &mut clock);
        Ok((first, embeddings, hidden, targets, scores))
    }

    /// Re-read only the verification head from the retained, unmodified hidden rows.
    /// Keep the same split/single-device arithmetic as the head in `verify`; no decoder layer runs again.
    fn verified_full_scores(&mut self, hidden: &[f32]) -> Result<Scores> {
        let chained = !self.partial && self.group.chains_verification();
        let alone = !chained && self.frozen_now() && self.group.backends.len() == 2;
        let (_, scores, _) = self.with_operands(Model::head_operands, |group, operands| {
            if alone { group.output_head_alone(operands, hidden, true, 0, false) }
            else { group.output_head(operands, hidden, true, 0, false) }
        })?;
        match scores {
            Some(scores @ Scores::Logits(_)) => Ok(scores),
            _ => Err("Straydog verification head returned no full logits".into()),
        }
    }

    /// A short prompt through the verification spans, up to [`PROMPT_ROWS`] rows per pass and every row
    /// kept: its fixed cost is that of a few decode steps rather than of a batched prefill.
    fn prefill_verified(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut rows = Vec::with_capacity(tokens.len() * WIDTH);
        for chunk in tokens.chunks(PROMPT_ROWS) {
            let (first, embeddings, hidden) = self.verify_rows(chunk, false)?;
            let hidden = hidden.ok_or("Straydog verification has no rows")?;
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

    /// Whether a verification's first layer and its PLE layer follow each other on the device (see [`Group::front_chained`]): a fully resident
    /// model's, and a partially resident one's on CUDA, where the first layer's hybrid handshake ends in a wait and nothing is read back or
    /// uploaded between the layers. `STRAYDOG_FRONT_CHAIN=0` keeps the rows going through the host there.
    fn front_chained(&self) -> bool {
        self.group.front_chained() && (!self.partial || (cfg!(feature = "cuda") && front_chain_wanted()))
    }

    /// The rows through the first layer, the PLE layer and the spans. With `keep`, a partially resident model's rows may stay on the device
    /// after the last span (`None`: in its workspace, for [`Group::output_head_chained`]).
    fn verify_rows(&mut self, tokens: &[u32], keep: bool) -> Result<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)> {
        // (every row verified is written, kept or not)
        let mut clock = Instant::now();
        self.ensure_kv(self.history.len() + tokens.len())?;
        self.protect(self.history.len(), self.history.len() + tokens.len())?;
        let position = self.history.len() as u64;
        self.prefetch_ple(&self.history, tokens)?;
        let hidden = tile(&self.token_embeddings(tokens)?);
        self.phase("verify: protect, PLE prefetch, embeddings", &mut clock);
        let chained = self.front_chained();
        // (`None`: the PLE layer's rows are in the first span's workspace, which continues from them)
        let (first, embeddings, mut hidden) = if chained {
            // The first layer runs on the device while the host waits for the PLE rows; the PLE layer then follows it
            // there, and one wait reads back both layers' rows. (A partially resident model's: the first layer's rows are
            // not read back, the PLE layer follows them in place, and the first span continues from the workspace.)
            self.group.verify_queue(0, 1, &hidden, position)?;
            self.phase("verify: first layer", &mut clock);
            let embeddings = self.ple_embeddings(tokens)?;
            self.phase("verify: PLE rows wait", &mut clock);
            if self.partial && !front_chain_checked() {
                self.with_operands(Model::ple_operands, |group, operands| group.ple_front_queue("blk.1.", &embeddings, operands))?;
                self.phase("verify: PLE layer", &mut clock);
                (Vec::new(), embeddings, None)
            } else {
                let (first, hidden) = self.with_operands(Model::ple_operands, |group, operands| {
                    group.ple_front("blk.1.", &embeddings, operands)
                })?;
                if self.partial {
                    // The check: the PLE state back to its snapshot and the layer again as the rows through the host take it.
                    self.group.copy_state("@ple:blk.1.", "@ple-snapshot:blk.1.", "blk.1.")?;
                    let again = self.with_operands(Model::ple_operands, |group, operands| {
                        group.ple_rows("blk.1.", &first, &embeddings, operands)
                    })?;
                    let differing = hidden.iter().zip(&again).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    static PROOF: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));
                    let mut proof = PROOF.lock().unwrap();
                    proof.0 += differing as u64;
                    proof.1 += 1;
                    if proof.1 % 100 == 0 {
                        eprintln!("Straydog row proofs: first layer chain: {} floats differ in {} passes", proof.0, proof.1);
                    }
                }
                self.phase("verify: PLE layer", &mut clock);
                // (a partially resident model's first span continues from the workspace either way)
                (first, embeddings, if self.partial { None } else { Some(hidden) })
            }
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
            (first, embeddings, Some(hidden))
        };
        let layers = Instant::now();
        // The spans chained: each continues from its device's workspace, the rows crossing from the first device to the second GPU to GPU
        // (the fully resident model's route), and the rows after the last span staying there for the head (a partially resident model's
        // spans end in a wait, their layers' hand-off with the host being a host loop, but read nothing back).
        let chain = keep && hidden.is_none() && self.group.chains_verification();
        let mut after = 1;
        for (index, (begin, end)) in self.group.decoder_spans(1).into_iter().enumerate() {
            if chain {
                self.group.verify_chain(after, begin, end, tokens.len(), position)?;
                after = end;
            } else {
                hidden = Some(match hidden.take() {
                    Some(rows) => self.group.verify_span(begin, end, &rows, position)?,
                    None if index == 0 => self.group.verify_span_in_place(begin, end, tokens.len(), position)?,
                    None => return Err("Straydog verification lost its rows".into()),
                });
            }
            self.phase(["verify: layers (first device)", "verify: layers (second device)"][index.min(1)], &mut clock);
        }
        if !chain && hidden.is_none() {
            return Err("Straydog verification has no rows".into());
        }
        if tokens.len() == 1 {
            self.assist_pass(layers.elapsed().as_secs_f64())?;
        }
        Ok((first, embeddings, if chain { None } else { hidden }))
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
            if self.front_chained() {
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

    #[allow(clippy::too_many_arguments)]
    fn try_penalty_top(&mut self, drafts: &[u32], proposed: &[Vec<(f32, u32)>], top: &[Vec<(f32, u32)>], sampling: &Sampling,
                       generated: &mut Generated, rng: &mut Rng, position: u64, keyed: bool) -> Option<(usize, u32)> {
        self.metric_count("penalty_top_passes", 1);
        // Sampling proves every reachable accepted prefix before touching the real Generated/RNG.
        // None preserves their complete pre-pass state, so a full-logit retry needs no second snapshot.
        let accepted = if !proposed.is_empty() {
            None // This optimization preserves the existing point-draft policy for penalized requests.
        } else if keyed {
            let seed = self.sample_seed;
            sampling::try_accept_top_keyed(drafts, top, VOCABULARY, sampling, generated,
                                           &|index| Rng::keyed(seed, position + index as u64))
        } else {
            sampling::try_accept_top(drafts, top, VOCABULARY, sampling, generated, rng)
        };
        if accepted.is_none() {
            self.metric_count("penalty_top_fallback_passes", 1);
        }
        accepted
    }

    fn metric_add(&mut self, key: &str, amount: f64) {
        let value = self.last_metrics.get(key).and_then(Value::as_f64).unwrap_or(0.0) + amount;
        self.last_metrics.insert(key.into(), json!(value));
    }

    fn metric_count(&mut self, key: &str, amount: u64) {
        let value = self.last_metrics.get(key).and_then(Value::as_u64).unwrap_or(0) + amount;
        self.last_metrics.insert(key.into(), json!(value));
    }

    /// Invalidate measurements after the mapped layers change. A CUDA probe must be restored
    /// on the GPU before the tuner acknowledges the held level.
    fn assist_restart(&mut self) -> Result<()> {
        if let Some(tuner) = self.assist.as_mut() {
            if let Some(level) = tuner.restart() {
                if let Err(error) = self.group.set_assist_level(level) {
                    let failure = format!("Straydog assist: could not restore held level {level}: {error}");
                    self.recovery_failure = Some(failure.clone());
                    return Err(failure);
                }
                tuner.set(level);
            }
        }
        Ok(())
    }

    /// Put in force the assist level a round of [`crate::assist`] asks for and say what it
    /// found. CUDA invalidates a failed round and restores its held level; HIP tries the next best.
    fn assist_apply(&mut self, mut next: Option<u32>) -> Result<()> {
        let Some(tuner) = self.assist.as_mut() else { return Ok(()) };
        while let Some(level) = next.take() {
            match self.group.set_assist_level(level) {
                Ok(()) => tuner.set(level),
                Err(error) if tuner.restoring(level) => {
                    let failure = format!("Straydog assist: could not restore held level {level}: {error}");
                    self.recovery_failure = Some(failure.clone());
                    return Err(failure);
                }
                Err(error) => {
                    eprintln!("Straydog assist: level {level} cannot be set ({error}); it is left out");
                    next = tuner.cannot_set(level);
                }
            }
        }
        if let Some(outcome) = tuner.outcome().filter(assist::Outcome::worth_saying) {
            eprintln!("{}", outcome.line());
        }
        Ok(())
    }

    /// A decode pass of one row took `seconds` over its layers: a window of a round, if one is under way.
    fn assist_pass(&mut self, seconds: f64) -> Result<()> {
        // (STRAYDOG_PIPELINE: the assist level and the slots stay as they are, so that two runs are the same arithmetic)
        if self.frozen_now() {
            return Ok(());
        }
        // (a request whose passes the assist's round timed: decode rates of such a request are no other's, its drafts left out)
        if self.assist_probing() {
            self.metric_count("assist_round_passes", 1);
        }
        if self.assist_script.is_some() {
            self.assist_scripted(seconds);
            return Ok(());
        }
        if let Some(next) = self.assist.as_mut().map(|tuner| tuner.pass(seconds)) {
            self.assist_apply(next)?;
        }
        Ok(())
    }

    /// A scripted pass ended: what it assisted, and the next level of the script.
    fn assist_scripted(&mut self, seconds: f64) {
        let totals = self.route_totals();
        let Some(script) = self.assist_script.as_mut() else { return };
        script.trace.push((script.level, [0, 1, 2].map(|at| totals[at] - script.routed[at]), seconds));
        script.routed = totals;
        if let Some(level) = script.next.pop_front() {
            match self.group.set_assist_level(level) {
                Ok(()) => {
                    script.level = level;
                    if let Some(tuner) = self.assist.as_mut() {
                        tuner.set(level);
                    }
                }
                Err(error) => eprintln!("Straydog assist: the scripted level {level} cannot be set ({error})"),
            }
        }
    }

    /// Whether the tuner has the passes: not a model that verifies known rows, nor one whose levels a test scripts.
    fn assist_tuning(&self) -> bool {
        self.forced_rows.is_none() && self.assist_script.is_none() && !self.frozen_now()
    }

    /// Whether the passes now are timed as windows of a round: those of a speculating model have no drafts, so that each has one row.
    fn assist_probing(&self) -> bool {
        self.assist_tuning() && self.assist.as_ref().is_some_and(assist::Tuner::probing)
    }

    /// `count` decode tokens were generated.
    fn assist_tokens(&mut self, count: u64) {
        if self.assist_tuning() {
            if let Some(tuner) = self.assist.as_mut() {
                tuner.tokens(count);
            }
        }
    }

    /// A request begins to decode.
    fn assist_request(&mut self) {
        if self.assist_tuning() {
            if let Some(tuner) = self.assist.as_mut() {
                tuner.request();
            }
        }
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
    /// `resume`: the request taken over from the pipelined loop where it stands (the token chosen and its pairs for the drafter, the tokens
    /// produced so far), and `stretch`: the produced count at which to hand it back at a pass's end, with its state, instead of ending it.
    #[allow(clippy::too_many_arguments)]
    fn generate_speculative(&mut self, logits: &[f32], max_new: usize, sampling: &Sampling, rng: &mut Rng,
                            cancelled: &dyn Fn() -> bool, started: Instant, emit: &mut dyn FnMut(u32), mut held: Option<&mut Constraint>,
                            resume: Option<pipeline::Hand>, stretch: Option<usize>) -> Result<Option<pipeline::Hand>> {
        let resumed = resume.is_some();
        let (mut generated, mut token, mut produced, mut mtp_hidden, mut mtp_tokens, mut mtp_position);
        if let Some(hand) = resume {
            (generated, token, produced) = (hand.generated, hand.token, hand.produced);
            (mtp_hidden, mtp_tokens, mtp_position) = (hand.mtp_hidden, hand.mtp_tokens, hand.mtp_position);
        } else {
            generated = Generated::new(VOCABULARY);
            token = constraint::pick(logits, sampling, &generated, rng, &mut held)?;
            generated.insert(token);
            if END_TOKENS.contains(&token) {
                return Ok(None);
            }
            self.last_metrics.insert("ttft_seconds".into(), json!(started.elapsed().as_secs_f64()));
            for (key, value) in [("generated_tokens", 1), ("speculative_passes", 0), ("drafted_tokens", 0), ("accepted_tokens", 0)] {
                self.last_metrics.insert(key.into(), json!(value));
            }
            emit(token);
            produced = 1usize;
            // The MTP layer pairs each hidden state with the token after it, at the hidden's position.
            mtp_hidden = self.final_hidden.clone().ok_or("Straydog has no hidden state to draft from")?;
            mtp_tokens = vec![token];
            mtp_position = self.history.len() as u64 - 1;
        }
        // A long match with the sequence so far is copied rather than drafted (an edit repeats its source): those drafts are
        // accepted nearly always, and the drafter's steps are saved. The MTP layer still takes the kept rows' pairs.
        let mut ngrams = Ngrams::new();
        ngrams.extend(&self.history);
        let mut policy = Policy::new(self.copy_drafts + 1);
        if let Some(chooser) = self.width.as_mut().filter(|_| !resumed) {
            chooser.begin_request();
        }
        while produced < max_new {
            if cancelled() {
                break;
            }
            if stretch.is_some_and(|stop| produced >= stop) {
                // (handed back at a pass's end: the pairs the drafter has not caught up on go with it)
                return Ok(Some(pipeline::Hand { generated, token, produced, mtp_hidden, mtp_tokens, mtp_position }));
            }
            let pass_started = Instant::now();
            // (while a round of the assist's tuner times decode passes of one row, none has drafts: crate::assist)
            let probing = self.assist_probing();
            // How many drafts this pass offers: the width in force, or a stretch of a probe that keeps what the windows hold of the others
            // current. The passes of a verified known sequence and of an assist round are not the chooser's.
            let plan = match self.width.as_mut() {
                Some(chooser) if !probing && self.forced_rows.is_none() => chooser.plan(),
                _ => width::Plan { drafts: self.drafts, probe: false },
            };
            // (a pass of a probe at a width not in force: the request's decode rate is not the width's in force)
            if plan.probe {
                self.metric_count("drafts_probe_passes", 1);
            }
            let limit = if probing { 0 } else { plan.drafts.min(self.speculative).min(max_new - produced - 1).min(self.context.saturating_sub(self.history.len() + 1)) };
            // The token's PLE rows read while the drafter runs; the drafts' while layer 0 does.
            let mut clock = Instant::now();
            self.prefetch_ple(&self.history, &[token])?;
            self.phase("PLE prefetch", &mut clock);
            ngrams.push(token);
            let copy_limit = if probing || plan.probe || self.pipeline.no_copy() { 0 } else { self.copy_drafts.min(max_new - produced - 1).min(self.context.saturating_sub(self.history.len() + 1)) };
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
            let read_logits = self.forced_rows.as_ref().is_some_and(|forced| forced.read_logits);
            let top = if held.is_some() || (sampling.presence_penalty != 0.0 && read_logits) { 0 }
                      else { sampling.target_read_top().unwrap_or(0) };
            let (first, embeddings, hidden, targets, mut scores) = self.verify(&rows, sampling.needs_logits() || held.is_some() || read_logits, top)?;
            let mut clock = Instant::now();
            let penalty_top = sampling.presence_penalty != 0.0 && matches!(scores.as_ref(), Some(Scores::Top(_)));
            let certified = if let Some(Scores::Top(top)) = scores.as_ref().filter(|_| penalty_top) {
                let keyed = self.frozen_now() && proposed.is_empty();
                self.try_penalty_top(&drafts, &proposed, top, sampling, &mut generated, rng, position, keyed)
            } else { None };
            if penalty_top && certified.is_none() {
                self.phase("accept", &mut clock);
                scores = Some(self.verified_full_scores(&hidden)?);
                self.phase("verify: head", &mut clock);
            }
            let (accepted, following) = match certified {
                Some(accepted) => accepted,
                None => match (held.as_deref_mut(), scores.as_mut()) {
                    (Some(held), Some(Scores::Logits(logits))) => held.accept_rows(&drafts, &proposed, logits, VOCABULARY, sampling, &mut generated, rng)?,
                    // (STRAYDOG_PIPELINE: the tokens are the target's draws keyed by position, the scheme the pipelined passes use too)
                    (None, Some(Scores::Logits(logits))) if self.frozen_now() && proposed.is_empty() => {
                        let seed = self.sample_seed;
                        sampling::accept_keyed(&drafts, logits, VOCABULARY, sampling, &mut generated, &|index| Rng::keyed(seed, position + index as u64))
                    }
                    _ => Self::accept(&drafts, &proposed, &targets, scores.as_ref(), sampling, &mut generated, rng),
                },
            };
            self.phase("accept", &mut clock);
            if self.pipeline.pass_log() {
                eprintln!("Straydog pass: position {position} rows {rows:?} accepted {accepted} following {following}");
            }
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
            let seconds = self.pass_clock.as_ref().map_or(seconds, |clock| clock(rows.len()));
            if let Some(mean) = self.pass_seconds.get_mut(rows.len().wrapping_sub(1)) {
                // (a pass that waited for the drive says nothing of a row's cost)
                *mean = if *mean > 0.0 { *mean + (seconds.min(3.0 * *mean) - *mean) / 16.0 } else { seconds };
            }
            self.passes_with_draft = if rows.len() == 1 { 0 } else { self.passes_with_draft + 1 };
            // What the pass cost and kept goes to the windows the drafts are chosen by; the end of a probe may change them.
            if !probing && !copied && self.forced_rows.is_none() {
                let outcome = self.width.as_mut().and_then(|chooser| chooser.record(plan, limit, rows.len(), accepted, seconds));
                if let Some(outcome) = outcome {
                    if outcome.worth_saying() {
                        eprintln!("{}", outcome.line());
                    }
                    if outcome.chosen != self.drafts {
                        self.drafts = outcome.chosen;
                        self.copy_drafts = outcome.chosen;
                        policy.set_rows(self.copy_drafts + 1);
                    }
                }
            }
            self.assist_tokens(emitted.len() as u64);
            if let Some(count) = self.pass_counts.get_mut(rows.len().wrapping_sub(1)) {
                *count += 1;
            }
            let hybrid = if self.partial && self.profile.as_ref().is_some_and(|profile| profile.every) {
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
        Ok(None)
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
        if let Some(failure) = &self.recovery_failure { return Err(failure.clone()); }
        if let Some(profile) = self.profile.as_mut() {
            *profile = Profile::new(profile.every);
        }
        self.ple_counts = self.weights.row_counts(PLE_TABLE).unwrap_or_default();
        self.ple_wait = 0.0;
        if self.partial {
            self.hybrid_counts = self.group.info().unwrap_or_default();
            self.pass_counts = [0; PARTIAL_ROWS];
            self.memory_before = crate::host::memory_note();
            self.fault_reads_before = crate::host::fault_reads();
            self.read_aheads_before = self.cpu_read_aheads();
            self.cpu_phases_before = self.cpu_phase_ns();
        }
        let result = self.generate_request(tokens, max_new, sampling, seed, cancelled, progress, turn, checkpoint, emit, constraint);
        self.report_profile();
        if result.is_ok() && self.partial {
            self.report_state_digest();
        }
        if result.as_ref().is_err_and(|error| error.ends_with(crate::gpu::HYBRID_WAIT_RECOVERED)) {
            self.recover_hybrid_wait()?;
        }
        result
    }

    /// The weights for a change (the drafter dropped, a table's row cache, the load): where nothing else holds them, which is everywhere but
    /// while a pipelined request runs (its relay thread holds a counted reference to read the embeddings).
    fn weights_mut(&mut self) -> &mut Weights {
        std::sync::Arc::get_mut(&mut self.weights).expect("the weights are shared only while a pipelined request runs")
    }

    /// The read-aheads taken and left, and the phase times, of the CPU engines: the second device's engine (pipelined verification) adds its own.
    fn cpu_read_aheads(&self) -> (u64, u64) {
        [self.cpu.as_ref(), self.cpu_second.as_ref()].into_iter().flatten().map(CpuExperts::read_aheads).fold((0, 0), |sum, own| (sum.0 + own.0, sum.1 + own.1))
    }

    fn cpu_phase_ns(&self) -> [u64; 3] {
        [self.cpu.as_ref(), self.cpu_second.as_ref()].into_iter().flatten().map(CpuExperts::phase_ns)
            .fold([0; 3], |sum, own| [sum[0] + own[0], sum[1] + own[1], sum[2] + own[2]])
    }

    /// The request's PLE row counts and wait, and a partially resident model's decode step (milliseconds a generated token, its
    /// layers, head and PLE work apart), and with DS41_PROFILE all its decode phases and its passes by rows, into the metrics.
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
            // (the pages the drive gave back for faults while it ran, the working set's limit and the K/V's capacity: what the memory line
            // needs to say why a request waited on the drive)
            let read_back = crate::host::fault_reads().zip(self.fault_reads_before).map(|(now, before)| now.since(before));
            let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
            let kv_bytes = attention_layers(self.weights.drafter) * self.kv_capacity as u64 * kv_position_bytes();
            let (taken, left) = self.cpu_read_aheads();
            // (the CPU's time a token by the engine's phases: where a faster kernel shows, or does not)
            let phases = self.cpu_phase_ns();
            let cpu_phases: Vec<f64> = (0..3).map(|i| phases[i].saturating_sub(self.cpu_phases_before[i]) as f64 / 1e6 / generated).collect();
            // What the host spent outside the layers' routes in a decode step (the line's "other" is made of it and the layers' own time), on
            // every request, DS41_PROFILE or not, including a pass that verified drafts. The clocks were already taken; this only folds them.
            let steps = self.profile.as_ref().map(|profile| decode_steps(&profile.phases, generated)).unwrap_or_default();
            let slot_counts = self.group.slots().ok();
            self.last_metrics.insert("partial_detail".into(), json!({
                "steps_ms": steps,
                "wait_ms": spent("wait_ns") / 1e6 / generated, "cpu_ms": spent("cpu_ns") / 1e6 / generated, "cpu_phases_ms": cpu_phases,
                "layer_ms": spent("layer_ns") / 1e6 / generated, "gpu_experts_ms": spent("gpu_experts_ns") / 1e6 / generated,
                "gpu_routes": spent("gpu_experts") / generated, "cpu_routes": spent("cpu_experts") / generated,
                "assisted_routes": spent("assisted_experts") / generated, "swaps": spent("swapped_experts") / generated,
                "assist": self.assist.as_ref().map(assist::Tuner::level), "locked_layers": self.locked_layers.len(),
                "slots": info["cached_experts"], "slot_capacity": slot_counts.map(|(capacity, _)| capacity),
                "slots_held": slot_counts.map(|(_, held)| held),
                "streamed": spent("streamed_experts"), "prompt_cpu_routes": spent("prompt_cpu_routes"),
                "passes": self.pass_counts, "pass_ms": self.pass_seconds.map(|seconds| 1e3 * seconds),
                "drafts": (self.speculative > 0).then_some(self.drafts),
                "resident_gib": memory.map(|note| note.0 as f64 / (1u64 << 30) as f64),
                "available_gib": memory.map(|note| note.1 as f64 / (1u64 << 30) as f64),
                "page_faults": faults,
                "fault_read_pages": read_back.map(|(pages, _)| pages), "fault_reads_system": read_back.map(|(_, system)| system),
                "working_set_limit_gib": crate::host::working_set_limit_now().map(gib),
                "kv_positions": self.kv_capacity, "kv_gib": gib(kv_bytes), "kv_host": self.host_kv,
                "read_aheads": [taken - self.read_aheads_before.0, left - self.read_aheads_before.1],
                "graphs": info["devices"][0]["cached_graphs"], "replays": info["devices"][0]["graph_replays"].as_u64().unwrap_or(0)
                    .saturating_sub(self.hybrid_counts["devices"][0]["graph_replays"].as_u64().unwrap_or(0)),
            }));
        }
        let Some(profile) = self.profile.as_ref().filter(|profile| profile.every) else { return };
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
        if let Some(failure) = &self.layer_prefill_restore_failure { return Err(failure.clone()); }
        // (banks locked behind "ready" that did not match the file's tree: nothing more is served from these bytes)
        if let Some(failure) = &self.bank_failure {
            return Err(failure.clone());
        }
        self.group.ensure_slots()?;
        let parking = Instant::now();
        let parked = self.park_or_restore(tokens)?;
        let park_seconds = parking.elapsed().as_secs_f64();
        let (reused, resumed) = self.resume(tokens)?;
        // (the sequence a start was handed, and this prompt began with)
        let resumed = match parked {
            Some(origin @ ("prefix" | "kept")) => origin,
            _ => resumed,
        };
        self.last_metrics = json!({
            "cached_tokens": reused, "resumed_from": resumed, "prefill_tokens": tokens.len() - reused,
            "parked": parked, "park_seconds": park_seconds, "park_copied_positions": 0,
            "decode_forward_tokens": 0, "decode_seconds": 0.0, "decode_tokens_per_second": null,
            "generated_tokens": 0, "ttft_seconds": null,
        })
        .as_object()
        .unwrap()
        .clone();
        if sampling.presence_penalty != 0.0 && sampling.target_read_top().is_some() {
            self.metric_count("penalty_top_passes", 0);
            self.metric_count("penalty_top_fallback_passes", 0);
        }
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
            // A prompt stopped between batches: its last batch's last row, which only the MTP layer's device kept, is the held
            // sequence's final hidden state, and the next request's first MTP pair starts from it (taken before the batch flags go).
            let kept = if matches!(prefilled, Ok(None)) { self.kept_hidden() } else { Ok(None) };
            self.deferred = false;
            self.mtp_kept = 0;
            let settled = self.group.prefill_settle();
            if prefilled.is_ok() {
                settled?;
            }
            if let Some(row) = kept? {
                self.final_hidden = Some(row);
            }
        }
        let Some(output) = prefilled? else { return Ok(()) };
        // (a root this prompt made: the tokens it holds, and a digest that names it)
        if let Some((root, _)) = plan.iter().find(|(_, kinds)| kinds.contains(&"root")) {
            self.last_metrics.insert("root_checkpoint".into(), json!(root));
            self.last_metrics.insert("root_digest".into(), json!(crate::prefix::tokens_digest(&tokens[..*root])));
        }
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
        self.sample_seed = seed;
        let Some(mut logits) = output else { return Ok(()) };
        self.assist_request();
        let applies = self.speculative > 0 && !constraint.as_deref().is_some_and(Constraint::chooses_every_token)
            && self.pipeline_applies(sampling, constraint.as_deref());
        self.pipeline_live_for(applies)?;
        if self.speculative > 0 && !constraint.as_deref().is_some_and(Constraint::chooses_every_token) {
            // (greedy requests of a partially resident model on two GPUs, STRAYDOG_PIPELINE=1: the first GPU runs the next pass's
            // layers while the second finishes this one's; any other takes the ordinary loop)
            if applies {
                return self.generate_pipelined(&logits, max_new, sampling, &mut rng, cancelled, started, emit);
            }
            return self.generate_speculative(&logits, max_new, sampling, &mut rng, cancelled, started, emit, constraint, None, None).map(|_| ());
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
                let next = self.forward(&[token], true)?.ok_or("Straydog decode step returned no logits")?;
                self.logits_spare = std::mem::replace(&mut logits, next);
                let seconds = decode_started.elapsed().as_secs_f64();
                self.metric_add("decode_seconds", seconds);
                if let Some(profile) = self.profile.as_mut() {
                    profile.rows[1].1 += seconds;
                }
                // (for the request's line: a pass without drafts is a lone row)
                self.pass_counts[0] += 1;
                self.pass_seconds[0] = if self.pass_seconds[0] > 0.0 { self.pass_seconds[0] + (seconds.min(3.0 * self.pass_seconds[0]) - self.pass_seconds[0]) / 16.0 } else { seconds };
                self.metric_count("decode_forward_tokens", 1);
                self.assist_tokens(1);
                self.update_rate();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod device_failure_tests {
    use super::finish_device_work;
    use crate::gpu::HYBRID_WAIT_RECOVERED;

    #[test]
    fn a_device_fault_wins_over_a_recovered_wait_in_either_order() {
        let fault = "an illegal memory access was encountered";
        assert_eq!(finish_device_work::<()>(Err(HYBRID_WAIT_RECOVERED.into()), Err(fault.into())), Err(fault.into()));
        assert_eq!(finish_device_work::<()>(Err(fault.into()), Err(HYBRID_WAIT_RECOVERED.into())), Err(fault.into()));
        let closed = "the first device's thread ended without the device";
        assert_eq!(finish_device_work::<()>(Err(HYBRID_WAIT_RECOVERED.into()), Err(closed.into())), Err(closed.into()));
    }

    #[test]
    fn a_clean_finish_preserves_the_recovered_wait_or_the_work() {
        assert_eq!(finish_device_work::<()>(Err(HYBRID_WAIT_RECOVERED.into()), Ok(())), Err(HYBRID_WAIT_RECOVERED.into()));
        assert_eq!(finish_device_work(Ok(42), Ok(())), Ok(42));
        assert_eq!(finish_device_work(Ok(42), Err(HYBRID_WAIT_RECOVERED.into())), Err(HYBRID_WAIT_RECOVERED.into()));
    }
}

#[cfg(all(test, any(feature = "hip", feature = "cuda")))]
mod wait_recovery_gpu {
    use super::*;

    #[test]
    #[ignore = "requires model files and an exclusive GPU"]
    fn a_recovered_hybrid_wait_discards_saved_sequences_and_serves_the_next_request() {
        let root = std::env::var_os("STRAYDOG_MODEL").expect("STRAYDOG_MODEL must name the model directory");
        let quant = std::env::var("STRAYDOG_MODEL_QUANT").unwrap_or_else(|_| "UD-IQ1_S".into());
        let drafter = std::env::var_os("STRAYDOG_DRAFTER").map(std::path::PathBuf::from);
        let bundle = crate::bundle::Id::find(Some(crate::bundle::MODEL), None, &quant).expect("a known Qwen bundle");
        let budget = 6 << 30;
        let mut group = Group::new(&[0], &[budget], crate::group::SplitMode::Experts).unwrap();
        group.configure_gemm(2, true).unwrap();
        let weights = Weights::open(&std::path::PathBuf::from(root), bundle, drafter.as_deref(),
                                    crate::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
        let text = crate::text::Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap();
        let settings = Settings { context: 65536, state_budget_bytes: 8 << 30, prefill_batch_size: 256,
                                  speculative: if drafter.is_some() { 3 } else { 0 }, cpu_threads: 8, row_cache_bytes: None };
        let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
        assert!(model.partial);
        if drafter.is_some() { assert!(model.speculative > 0, "the test must keep its MTP layer"); }
        model.prompt_share(false).unwrap();
        // A display GPU prelaunches waits only for spans containing mapped banks.
        assert!(model.lock_layers_for_test(&[0]).unwrap() > 0);
        let prompt = text.encode("<|im_start|>user\nCount from one to thirty in English.<|im_end|>\n<|im_start|>assistant\n").unwrap();
        let generate = |model: &mut Model, tokens: &[u32]| {
            model.generate(tokens, 32, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true, &mut |_| {})
        };
        generate(&mut model, &prompt).unwrap();
        assert!(!model.history.is_empty());
        model.save_checkpoint("root").unwrap();
        model.save_checkpoint("turn").unwrap();
        if let Some(old) = model.parked.take() { model.discard(old).unwrap(); }
        model.parked = Some(model.park(0, PARK_SLOTS[0]).unwrap());
        let mut next = model.history.clone();
        next.extend(text.encode("\nContinue counting: ").unwrap());
        struct ClearFault;
        impl Drop for ClearFault {
            fn drop(&mut self) { unsafe { straydog_gpu::ffi::sd_test_hybrid_timeout(0); } }
        }
        let _clear = ClearFault;
        // SAFETY: a development-only counter; one GPU job holds the test's process.
        unsafe { straydog_gpu::ffi::sd_test_hybrid_timeout(1); }
        let error = generate(&mut model, &next).unwrap_err();
        unsafe { straydog_gpu::ffi::sd_test_hybrid_timeout(0); }
        assert!(error.ends_with(crate::gpu::HYBRID_WAIT_RECOVERED), "{error}");
        assert!(model.history.is_empty() && model.final_hidden.is_none());
        assert!(model.checkpoints.is_empty() && model.parked.is_none());
        assert_eq!(model.mtp_kept, 0);
        assert!(model.draft_pending.is_empty() && model.recovery_failure.is_none());
        generate(&mut model, &prompt).unwrap();
        assert!(model.history.starts_with(&prompt));
        let before = model.vram_budgets().unwrap()[0].2;
        model.resize_vram(0, before - (512 << 20)).unwrap();
        let after = model.vram_budgets().unwrap()[0].2;
        assert!(after < before, "the budget must actually fall");
        generate(&mut model, &next).unwrap();
        model.raise_vram(0, before).unwrap();
        generate(&mut model, &prompt).unwrap();
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
fn working_split(weights: &Weights, group: &mut Group, context: u64, prefill: usize, derived: bool, drafting: bool, vram_kv: bool)
                 -> Result<(Option<(bool, bool, usize)>, Option<String>)> {
    let placed = resident_placement(weights, group, context, prefill, derived, drafting, vram_kv)?;
    let (Some(baseline), Some(balanced)) = (placed.0, group.split()) else { return Ok(placed) };
    if !group.expert_parallel()? || balanced <= EVEN_SPLIT {
        return Ok(placed);
    }
    let as_good = |(keep, host, batch): (bool, bool, usize)| keep >= baseline.0 && (!host || baseline.1) && batch >= baseline.2;
    for split in EVEN_SPLIT..balanced {
        group.set_split(split)?;
        let placement = resident_placement(weights, group, context, prefill, derived, drafting, vram_kv)?;
        if placement.0.is_some_and(as_good) {
            return Ok(placement);
        }
    }
    group.set_split(balanced)?;
    Ok(placed)
}

/// The batch a model that is not fully resident keeps of the wider one derived (the 4096 a resident Qwen3.8 on two R9700s is derived), and the
/// batch a fully resident one's K/V stays in VRAM at before it moves to host memory at the wider one: what was derived before 4096.
const NARROW_BATCH: usize = 2048;
/// The GPU budgets in all from which the derived prefill batch is the widest.
const WIDE_BATCH_BUDGET: u64 = 48 << 30;
/// A partially resident model on two GPUs with this much budget in all (and under [`WIDE_BATCH_BUDGET`], where the batch is kept at 2048) reads in
/// batches of [`PARTIAL_PAIR_BATCH`] rows: every batch streams the experts the slots do not hold again, whatever its rows, and the workspaces of
/// 4096 rows take some 0.8 GiB a GPU from the slots, where borrowing them for a read (`Group::begin_long_prefill`) refilled the slots twice.
const PARTIAL_PAIR_BUDGET: u64 = 40 << 30;
const PARTIAL_PAIR_BATCH: usize = 4096;

/// Whether a build on NVIDIA GPUs (`cuda`) with `devices` GPUs and `budget` bytes of VRAM budget in all is the pair of 24 GB cards that
/// [`PARTIAL_PAIR_BATCH`] and [`PRESTAGE_SLOTS`] are for.
fn partial_pair(cuda: bool, devices: usize, budget: u64) -> bool {
    cuda && devices == 2 && (PARTIAL_PAIR_BUDGET..WIDE_BATCH_BUDGET).contains(&budget)
}
/// The rows a long read's batches may be, widest first, where the workspaces hold them (`wide_placement`).
/// A multiple of the experts' chunk (4096 rows, `MOE_CHUNK_ROWS`): between two, the experts of a batch are read from VRAM in two chunks that are each
/// narrower than 4096 and the batch pays them over more rows (two R9700s, `--vram 30`, a fresh 242K read: 50.0 s at 4096 rows, 5120 rows (2 x 2560)
/// 51.0 s, 6144 (2 x 3072) 50.05 s, 8192 (2 x 4096) 49.1 s).
const WIDE_ROWS: [usize; 2] = [12288, 8192];
/// The new tokens a read has from which its batches are the wide ones: two wide batches (a 20K read in 3.78 s against 3.81 at 4096 rows).
pub const WIDE_READ_TOKENS: usize = 16_384;
/// [`Settings::prefill_batch_size`] when the batch is not given: derived from the GPU budgets.
pub const DERIVED_BATCH: usize = 0;

/// The rows of a prefill batch derived for a resident model on this much GPU budget in all. Under 8 GiB the batches' workspaces rest at 256 rows, with
/// some 200 MiB more for decode's slots, and widen for a long prompt ([`Group::begin_long_prefill`]). From 48 GiB (two R9700s) it is the widest, 4096,
/// which the placement narrows (only a batch not given is narrowed): the model that is not fully resident keeps 2048, and one that fits resident only
/// at 2048 takes it ([`resident_placement`]).
pub fn resident_batch(total: u64) -> usize {
    match total {
        total if total >= WIDE_BATCH_BUDGET => 4096,
        total if total >= 8 << 30 => 1024,
        _ => 256,
    }
}

/// The placements `resident_placement` tries in order, as (K/V in host memory, prefill batch): K/V in VRAM (when it fits the budget's quarter)
/// at the asked batch, and for a derived batch at `NARROW_BATCH` if that was wider (a model that fit with its K/V in VRAM at 2048 does not give
/// it up for a wider batch with the K/V in host memory; a batch given is not narrowed to keep the K/V in VRAM), then K/V in host memory at
/// batches halved to 1024.
fn placement_candidates(prefill: usize, derived: bool, vram_kv: bool) -> Vec<(bool, usize)> {
    let mut batches = vec![prefill];
    while *batches.last().unwrap() > 1024 {
        batches.push((batches.last().unwrap() / 2).max(1024));
    }
    let mut candidates: Vec<(bool, usize)> = Vec::new();
    if vram_kv {
        candidates.push((false, prefill));
        if derived && prefill > NARROW_BATCH {
            candidates.push((false, NARROW_BATCH));
        }
    }
    candidates.extend(batches.iter().map(|batch| (true, *batch)));
    candidates
}

/// (drafter kept, K/V in host memory, prefill batch) that lets every weight live on the GPUs,
/// if any. Full residency (and with it MTP) is worth more than K/V in VRAM; a placement
/// keeping the drafter wins over one that drops it; the configured batch and VRAM K/V first.
/// The first placement that fits, of `placement_candidates`; with the MTP drafter, then without. With it, what the first choice lacked, when it was not taken:
/// a placement depends on the VRAM left, and the start's line says what a little more would have bought. None fitting, the note says
/// what the least of them (the last) lacked: what full residency needs.
fn resident_placement(weights: &Weights, group: &Group, context: u64, prefill: usize, derived: bool, drafting: bool, vram_kv: bool)
                      -> Result<(Option<(bool, bool, usize)>, Option<String>)> {
    let candidates = placement_candidates(prefill, derived, vram_kv);
    let keeps: &[bool] = if drafting { &[true, false] } else { &[false] };
    let (mut first, mut last): (Option<String>, Option<String>) = (None, None);
    for keep in keeps {
        for (host, batch) in &candidates {
            let planned = group.planned_state_bytes(context, *batch as u64, drafting, *host)?;
            let requirements = weights.resident_requirements(*keep && weights.drafter);
            let lack = group.pin_lack_model(&requirements, Some(&planned))?;
            if lack == 0 {
                return Ok((Some((*keep, *host, *batch)), first));
            }
            last = (lack != u64::MAX).then(|| {
                format!("K/V in {}, prefill batch {batch}{} would need {:.1} GiB more VRAM", if *host { "host memory" } else { "VRAM" },
                        if *keep && weights.drafter { ", MTP drafter" } else { "" }, lack as f64 / (1u64 << 30) as f64)
            });
            if first.is_none() {
                first = last.clone();
            }
        }
    }
    Ok((None, last.map(|least| format!("fully resident: {least}"))))
}

/// The rows of a long read's batches wider than `batch`, for a fully resident placement in parts whose workspaces hold them (the first of
/// `WIDE_ROWS` that fits, with the same measure as `resident_placement`: the weights and the state a placement still allocates, with the margin kept for
/// what the runtime allocates outside the budget), else 0 and what the widest wanted would have needed. The K/V stage grows into the VRAM the placement
/// leaves, so what the wider workspaces take is taken from its growth (beyond the 64th of the budget it starts at). `STRAYDOG_PREFILL_WIDE` names the rows
/// to try instead (`off`: none). The workspaces are allocated at that width once (a wide batch runs the experts in chunks of `MOE_CHUNK_ROWS`, so what
/// grows with the rows is the decoder's rows and the MTP layer's), and decode never sees them.
fn wide_placement(weights: &Weights, group: &Group, context: u64, batch: usize, keep: bool, drafting: bool, host: bool)
                  -> Result<(usize, Option<String>)> {
    let rows: Vec<usize> = match std::env::var("STRAYDOG_PREFILL_WIDE").ok().as_deref() {
        Some("off" | "0") => return Ok((0, None)),
        Some(rows) => rows.parse().ok().into_iter().collect(),
        None => WIDE_ROWS.to_vec(),
    };
    let requirements = weights.resident_requirements(keep && weights.drafter);
    let mut note = None;
    for rows in rows.into_iter().filter(|rows| *rows > batch && *rows as u64 <= crate::gpu::PREFILL_ROWS_MAX) {
        let planned = group.planned_state_bytes(context, rows as u64, drafting, host)?;
        let lack = group.pin_lack_model(&requirements, Some(&planned))?;
        if lack == 0 {
            return Ok((rows, note));
        }
        if note.is_none() && lack != u64::MAX {
            note = Some(format!("a long read's batches of {rows} rows would need {:.2} GiB more VRAM", lack as f64 / (1u64 << 30) as f64));
        }
    }
    Ok((0, note))
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
mod placement_candidate_tests {
    use super::{context_state_bytes, kv_fits_budget, partial_drafter_requirements, partial_requirements, placement_candidates, resident_batch, state_per_token,
                KV_POSITION_BYTES};
    use crate::gpu::KvFormat;

    #[test]
    fn a_whole_contexts_q8_kv_uses_its_actual_bytes_and_keeps_the_budgets_quarter() {
        // At 262144 positions Q8 needs 3.1875 GiB, or 3.453125 GiB with MTP.
        // FP16 still needs 6 / 6.5 GiB. Budgets here are totals, also for a pair of GPUs.
        for (budget_gib, q8, f16) in [
            (8, [false, false], [false, false]),
            (10, [false, false], [false, false]),
            (16, [true, true], [false, false]),
            (20, [true, true], [false, false]),
            (24, [true, true], [true, false]),
            (32, [true, true], [true, true]),
            (48, [true, true], [true, true]),
        ] {
            for (index, drafting) in [false, true].into_iter().enumerate() {
                assert_eq!(kv_fits_budget(262144, drafting, KvFormat::Q8, budget_gib << 30), q8[index], "Q8, {budget_gib} GiB, MTP={drafting}");
                assert_eq!(kv_fits_budget(262144, drafting, KvFormat::F16, budget_gib << 30), f16[index], "FP16, {budget_gib} GiB, MTP={drafting}");
            }
        }
    }

    #[test]
    fn the_kv_quarter_includes_its_exact_threshold_but_not_one_byte_less() {
        // Whole-context byte counts times four: Q8's 272-byte rows, FP16's 512.
        for (format, drafting, threshold) in [
            (KvFormat::Q8, false, 13_690_208_256),
            (KvFormat::Q8, true, 14_831_058_944),
            (KvFormat::F16, false, 24u64 << 30),
            (KvFormat::F16, true, 26u64 << 30),
        ] {
            assert!(!kv_fits_budget(262144, drafting, format, threshold - 1));
            assert!(kv_fits_budget(262144, drafting, format, threshold));
            assert!(kv_fits_budget(262144, drafting, format, threshold + 1));
        }
    }

    #[test]
    fn a_small_budget_still_has_host_candidates_after_the_vram_candidate() {
        // The former host-only 5.5 GiB / 64K placement now admits Q8 in VRAM.
        // Full residency still tries host K/V after it; this does not promise the partial gate fits.
        let budget = 11u64 << 29;
        for drafting in [false, true] {
            assert!(kv_fits_budget(65536, drafting, KvFormat::Q8, budget));
            assert_eq!(placement_candidates(256, false, kv_fits_budget(65536, drafting, KvFormat::Q8, budget)),
                       [(false, 256), (true, 256)]);
            // 128K retains the host-only candidate on the same card.
            assert!(!kv_fits_budget(131072, drafting, KvFormat::Q8, budget));
            assert_eq!(placement_candidates(256, false, kv_fits_budget(131072, drafting, KvFormat::Q8, budget)), [(true, 256)]);
        }
    }

    #[test]
    fn empty_kv_costs_nothing_and_nonempty_kv_never_fits_a_zero_budget() {
        for format in [KvFormat::Q8, KvFormat::F16] {
            for drafting in [false, true] {
                assert!(kv_fits_budget(0, drafting, format, 0));
                assert!(!kv_fits_budget(1, drafting, format, 0));
                assert!(!kv_fits_budget(u64::MAX, drafting, format, u64::MAX));
            }
        }
        // Four planes, one position, twelve layers: quarter-budget rounding is exact.
        assert!(!kv_fits_budget(1, false, KvFormat::Q8, 52_223));
        assert!(kv_fits_budget(1, false, KvFormat::Q8, 52_224));
    }

    #[test]
    fn the_conservative_host_estimate_keeps_its_fp16_kv_subtraction() {
        assert_eq!(state_per_token(false), 26_112);
        assert_eq!(state_per_token(true), 28_288);
        // Indexers, convolution and PLE remain in the host estimate after VRAM KV and recurrence are removed.
        let recurrence = 36 * 48 * 128 * 128 * 4;
        assert_eq!(context_state_bytes(262144, false) - 12 * 262144 * KV_POSITION_BYTES - recurrence, 409_016_320);
        assert_eq!(context_state_bytes(262144, true) - 13 * 262144 * KV_POSITION_BYTES - recurrence, 442_575_360);
    }

    #[test]
    fn the_partial_preflight_keeps_the_same_dense_weights_and_largest_staging_requirement() {
        let dense = vec![("blk.0.attn_qkv.weight".into(), 12345), ("output.weight".into(), 67890)];
        // IQ2_S gate/up + IQ4_NL down is 1971200 bytes; IQ4_XS gate/up raises it to 2662400.
        assert_eq!(partial_requirements(dense.clone(), &[[22, 22, 20], [23, 23, 20]], 0),
                   [dense[0].clone(), dense[1].clone(), ("@hybrid:staging".into(), 219_414_528)]);
        // Wide reads retain their extra staging slots in the new preflight as in the partial gate.
        assert_eq!(partial_requirements(dense.clone(), &[[22, 22, 20], [23, 23, 20]], 192),
                   [dense[0].clone(), dense[1].clone(), ("@hybrid:staging".into(), 730_595_328)]);
    }

    #[test]
    fn the_partial_drafter_gate_keeps_dense_weights_and_verification_room_without_its_routed_banks() {
        let target = vec![("output.weight".into(), 67890)];
        let drafter = vec![("mtp.fc_hidden.weight".into(), 4096), ("blk.48.ffn_gate_exps.weight".into(), 1u64 << 30)];
        assert_eq!(partial_drafter_requirements(target.clone(), drafter),
                   [target[0].clone(), ("mtp.fc_hidden.weight".into(), 4096),
                    ("@verify:buffers".into(), 136_314_880), ("@hybrid:slots".into(), 134_217_728)]);
    }

    /// The same real placement gate used before reserve: a short context newly eligible for
    /// Q8 in VRAM would crowd out dense weights that the old host path can still keep.
    #[cfg(any(feature = "hip", feature = "cuda"))]
    #[test]
    #[ignore = "requires a GPU; run under the GPU lock with Q8 K/V"]
    fn a_partial_vram_kv_refusal_keeps_the_old_host_path_with_small_indexers() {
        use crate::gpu::Backend;
        use crate::group::{Group, SplitMode};

        assert_eq!(KvFormat::selected(), KvFormat::Q8, "this regression exercises Q8 placement");
        let budget = 4u64 << 30;
        let group = Group::new(&[0], &[budget], SplitMode::Layer).unwrap();
        let (used, _) = group.memory().unwrap();
        let kinds = [[22, 22, 20]; 48];
        let staging = partial_requirements(Vec::new(), &kinds, 0)[0].1;
        for drafting in [false, true] {
            // The old FP16 estimate rejected VRAM; Q8 makes it a candidate at this budget.
            assert!(!kv_fits_budget(65536, drafting, KvFormat::F16, budget));
            assert!(kv_fits_budget(65536, drafting, KvFormat::Q8, budget));
            let vram = group.planned_state_bytes(65536, 256, drafting, false).unwrap();
            let host_start = group.planned_state_bytes(32768, 256, drafting, true).unwrap();
            let host_full = group.planned_state_bytes(65536, 256, drafting, true).unwrap();
            // Fill the old host path's exact available dense room, not an arbitrary safety margin.
            let dense = budget.checked_sub(used + Backend::slack(budget) + host_start[0] + staging).unwrap();
            let requirements = partial_requirements(vec![("blk.0.attn_qkv.weight".into(), dense)], &kinds, 0);
            assert!(group.can_pin_model(&requirements, Some(&host_start)).unwrap());
            assert!(!group.can_pin_model(&requirements, Some(&vram)).unwrap());
            // Checking the fallback at full context would wrongly refuse the old viable start.
            assert!(!group.can_pin_model(&requirements, Some(&host_full)).unwrap());
        }
        // A separate boundary: target dense weights fit with VRAM K/V, but a requested
        // drafter only fits on the old host path. The preflight must include its dense gate too.
        let vram = group.planned_state_bytes(65536, 256, true, false).unwrap();
        let host_start = group.planned_state_bytes(32768, 256, true, true).unwrap();
        let dense = budget.checked_sub(used + Backend::slack(budget) + vram[0] + staging).unwrap();
        let target = partial_requirements(vec![("blk.0.attn_qkv.weight".into(), dense)], &kinds, 0);
        let with_drafter = partial_drafter_requirements(target.clone(), vec![("mtp.fc_hidden.weight".into(), 16 << 20)]);
        assert!(group.can_pin_model(&target, Some(&vram)).unwrap());
        assert!(!group.can_pin_model(&with_drafter, Some(&vram)).unwrap());
        assert!(group.can_pin_model(&with_drafter, Some(&host_start)).unwrap());
    }

    /// A real model exercises the newly admitted KV placement, not just the byte gate.
    /// STRAYDOG_MODEL_QUANT defaults to UD-IQ1_S; STRAYDOG_DRAFTER optionally adds MTP.
    #[cfg(any(feature = "hip", feature = "cuda"))]
    #[test]
    #[ignore = "requires STRAYDOG_MODEL and a GPU; run under the GPU lock with Q8 K/V"]
    fn a_partial_model_keeps_vram_kv_and_the_requested_drafter_across_requests() {
        use std::path::PathBuf;
        use super::{starting_kv, Model, Settings};
        use crate::group::{Group, SplitMode};
        use crate::sampling::Sampling;
        use crate::text::Text;
        use crate::weights::Weights;

        assert_eq!(KvFormat::selected(), KvFormat::Q8, "this regression requires Q8 K/V");
        let root = PathBuf::from(std::env::var_os("STRAYDOG_MODEL").expect("STRAYDOG_MODEL must name the model bundle directory"));
        let quant = std::env::var("STRAYDOG_MODEL_QUANT").unwrap_or_else(|_| "UD-IQ1_S".into());
        let drafter = std::env::var_os("STRAYDOG_DRAFTER").map(PathBuf::from);
        let drafting = drafter.is_some();
        let (budget, context) = (8u64 << 30, 131072);
        assert!(!kv_fits_budget(context, drafting, KvFormat::F16, budget), "the old FP16 estimate excludes this placement");
        assert!(kv_fits_budget(context, drafting, KvFormat::Q8, budget), "the held Q8 rows fit a quarter of the budget");
        let id = crate::bundle::Id::find(Some(crate::bundle::MODEL), None, &quant).unwrap();
        let weights = Weights::open(&root, id, drafter.as_deref(), crate::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
        let text = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap();
        let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
        group.configure_gemm(2, true).unwrap();
        let settings = Settings { context: context as usize, state_budget_bytes: 8 << 30, prefill_batch_size: 256,
                                  speculative: if drafting { 3 } else { 0 }, cpu_threads: 0, row_cache_bytes: None };
        let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
        assert!(model.partial && !model.resident && !model.host_kv,
                "8 GiB / 128K must keep partial VRAM K/V: partial={} resident={} host={}", model.partial, model.resident, model.host_kv);
        assert_eq!(model.kv_positions(), starting_kv(settings.context), "partial VRAM K/V starts small and grows with the sequence");
        assert_eq!(model.speculative, settings.speculative, "the requested MTP drafter must not be dropped");
        if drafting { assert!(model.weights.drafter, "the requested drafter remains authenticated and in use"); }
        eprintln!("{}", serde_json::json!({"case":"partial_vram_kv_placement", "quant":quant, "budget":budget,
            "context":context, "mtp_requested":drafting, "speculative":model.speculative, "partial":model.partial,
            "kv_host":model.host_kv, "kv_positions":model.kv_positions(), "devices":model.group.info().unwrap()}));

        let mut prompt = text.encode("<|im_start|>user\nWrite one complete sentence about why printed books are useful.<|im_end|>\n<|im_start|>assistant\n").unwrap();
        for request in 0..2 {
            let mut tokens = Vec::new();
            model.generate(&prompt, 8, &Sampling::GREEDY, 1, &|| false, &mut |_, _| {}, true, false, &mut |token| tokens.push(token)).unwrap();
            assert!(!tokens.is_empty(), "request {request} must emit at least one token");
            assert!(model.history.starts_with(&prompt), "request {request} keeps its prompt in the live sequence");
            assert!(model.final_hidden().is_some_and(|row| row.iter().all(|value| value.is_finite())), "request {request} leaves finite hidden state");
            assert!(!model.host_kv && model.speculative == settings.speculative, "request {request} keeps the tested placement");
            eprintln!("{}", serde_json::json!({"case":"partial_vram_kv_request", "request":request,
                "prompt_tokens":prompt.len(), "tokens":tokens, "metrics":model.last_metrics}));
            if request == 0 {
                prompt.extend_from_slice(&tokens);
                prompt.extend(text.encode("<|im_end|>\n<|im_start|>user\nGive another use for printed books.<|im_end|>\n<|im_start|>assistant\n").unwrap());
            }
        }
    }

    #[test]
    fn the_derived_wide_batch_keeps_vram_kv_at_the_narrow_one_before_host_kv_at_any() {
        // (the batch derived for a resident Qwen3.8 on two R9700s, 4096: K/V in VRAM at 2048 comes before K/V in host memory at 4096)
        assert_eq!(placement_candidates(4096, true, true), [(false, 4096), (false, 2048), (true, 4096), (true, 2048), (true, 1024)]);
        assert_eq!(placement_candidates(4096, true, false), [(true, 4096), (true, 2048), (true, 1024)]);
        // (the batch derived until 4096: what fit then still gets what it got)
        assert_eq!(placement_candidates(2048, true, true), [(false, 2048), (true, 2048), (true, 1024)]);
        assert_eq!(placement_candidates(2048, true, false), [(true, 2048), (true, 1024)]);
        assert_eq!(placement_candidates(1024, true, true), [(false, 1024), (true, 1024)]);
        assert_eq!(placement_candidates(256, true, true), [(false, 256), (true, 256)]);
        assert_eq!(placement_candidates(3000, true, true), [(false, 3000), (false, 2048), (true, 3000), (true, 1500), (true, 1024)]);
    }

    #[test]
    fn a_batch_given_is_not_narrowed_to_keep_vram_kv() {
        // (--batch-size 4096 is 4096 with the K/V in VRAM, else in host memory, as before the derived batch was 4096)
        assert_eq!(placement_candidates(4096, false, true), [(false, 4096), (true, 4096), (true, 2048), (true, 1024)]);
        assert_eq!(placement_candidates(4096, false, false), [(true, 4096), (true, 2048), (true, 1024)]);
        assert_eq!(placement_candidates(3000, false, true), [(false, 3000), (true, 3000), (true, 1500), (true, 1024)]);
        // (a batch of 2048 or less has nothing to narrow)
        for batch in [256, 1024, 2048] {
            assert_eq!(placement_candidates(batch, false, true), placement_candidates(batch, true, true));
        }
    }

    #[test]
    fn the_derived_batch_is_the_widest_from_48_gib_of_budgets() {
        assert_eq!((resident_batch(5 << 30), resident_batch(8 << 30)), (256, 1024), "a small card's batches rest narrow");
        assert_eq!((resident_batch(40 << 30), resident_batch((48 << 30) - 1), resident_batch(48 << 30), resident_batch(64 << 30)),
                   (1024, 1024, 4096, 4096));
    }
}

#[cfg(test)]
mod overlap_tests {
    use super::{overlap_batches, overlap_segment, overlap_split_ok, overlap_stopped, prefill_overlap_named};

    /// Decides: which batches of a segment two devices take at once. All of them in batches of the given rows, but a last of one row, which is
    /// the decode path's, and a segment of one batch is nothing to overlap.
    #[test]
    fn a_last_row_alone_stays_serial() {
        assert_eq!(overlap_batches(0, 10, 4), vec![(0, 4), (4, 8), (8, 10)]);
        assert_eq!(overlap_batches(0, 9, 4), vec![(0, 4), (4, 8)]);
        assert_eq!(overlap_batches(100, 100, 4096), vec![]);
        assert_eq!(overlap_batches(0, 4096, 4096).len(), 1);
        assert_eq!(overlap_batches(5, 12, 4), vec![(5, 9), (9, 12)]);
        assert_eq!(overlap_batches(0, 1, 4096), vec![]);
    }

    /// Decides: a read in batches of fewer than two rows takes none of them at once (a batch of one row is the decode path's, and the first
    /// device's span call takes two rows or more), where a batch of two rows is overlapped as any other, its odd last row apart.
    #[test]
    fn batches_of_one_row_stay_serial() {
        assert_eq!(overlap_batches(0, 5, 1), vec![]);
        assert_eq!(overlap_batches(3, 100, 1), vec![]);
        assert_eq!(overlap_batches(0, 5, 0), vec![]);
        assert_eq!(overlap_batches(0, 5, 2), vec![(0, 2), (2, 4)]);
        assert_eq!(overlap_batches(0, 4, 2), vec![(0, 2), (2, 4)]);
        assert_eq!(overlap_batches(0, 3, 2), vec![(0, 2)]);
    }

    /// Decides: where the overlap cannot be taken (HIP, one GPU, the layers in parts) no list of the segment's batches is built, whatever its size
    /// (the serial loop asks at each of its batches: a list of what remains each time was a quadratic cost on paths that must not change). A segment of
    /// 2^60 rows in batches of 2 would not fit in memory as a list; where the overlap is eligible the list is the segment's batches, and a segment of
    /// fewer than two batches is nothing to overlap.
    #[test]
    fn an_ineligible_segment_builds_no_list() {
        assert_eq!(overlap_segment(false, 0, 1usize << 60, 2), None);
        assert_eq!(overlap_segment(false, 0, 10, 4), None);
        assert_eq!(overlap_segment(true, 0, 10, 4), Some(vec![(0, 4), (4, 8), (8, 10)]));
        assert_eq!(overlap_segment(true, 0, 4, 4), None);
        assert_eq!(overlap_segment(true, 0, 10, 1), None);
    }

    /// Decides: a cancel that comes after the first device's loop has ended (while the second device drains the last batches) stops the read, as
    /// one before a batch does: the closure of a request that turns true after a number of looks.
    #[test]
    fn a_cancel_during_the_drain_is_seen() {
        let looks = std::cell::Cell::new(0);
        let cancelled = || { looks.set(looks.get() + 1); looks.get() > 2 };
        assert!(!overlap_stopped(false, &cancelled));
        assert!(!overlap_stopped(false, &cancelled));
        assert!(overlap_stopped(false, &cancelled));
        assert!(overlap_stopped(true, &|| false));
        assert!(!overlap_stopped(false, &|| false));
    }

    /// Decides: the overlap needs layer 0 and the PLE layer (layer 1) on the first device, so the second device's first layer is the second or
    /// later; a split at layer 1 or none (one device) reads serially.
    #[test]
    fn the_first_device_holds_layer_one() {
        assert!(!overlap_split_ok(None));
        assert!(!overlap_split_ok(Some(0)));
        assert!(!overlap_split_ok(Some(1)));
        assert!(overlap_split_ok(Some(2)));
        assert!(overlap_split_ok(Some(26)));
    }

    /// Decides: STRAYDOG_PREFILL_OVERLAP=0 turns the overlap off; unset, empty or any other text leaves it on.
    #[test]
    fn the_switch_is_off_only_at_zero() {
        assert!(prefill_overlap_named(None));
        assert!(prefill_overlap_named(Some("1")));
        assert!(prefill_overlap_named(Some("")));
        assert!(!prefill_overlap_named(Some("0")));
        assert!(!prefill_overlap_named(Some(" 0 ")));
    }
}

#[cfg(test)]
mod partial_pair_tests {
    use super::{partial_pair, PRESTAGE_BATCH, PARTIAL_PAIR_BATCH, WIDE_BATCH_BUDGET};

    /// Decides: the wide batch and the prestage are for two NVIDIA GPUs with 40 to 48 GiB of budget in all (two 24 GB cards); one GPU,
    /// three, a smaller or a larger pair (which keeps its 2048-row batch), and a build for another vendor keep what they had.
    #[test]
    fn two_cards_of_24_gb_only() {
        const G: u64 = 1 << 30;
        assert!(partial_pair(true, 2, 45 * G));
        assert!(partial_pair(true, 2, 40 * G));
        assert!(!partial_pair(true, 2, 40 * G - 1));
        assert!(!partial_pair(true, 2, WIDE_BATCH_BUDGET));
        assert!(!partial_pair(true, 1, 45 * G));
        assert!(!partial_pair(true, 3, 45 * G));
        assert!(!partial_pair(false, 2, 45 * G));
        assert!(PARTIAL_PAIR_BATCH >= PRESTAGE_BATCH);
    }
}

#[cfg(test)]
mod layer_prefill_policy_tests {
    use super::{layer_prefill_needed, layer_prefill_room};
    const G:u64=1<<30;
    // The T284 cases: RAM `memory` with `pinned` state and nothing else taking memory (`now` unknown), Linux (no fit rule).
    fn needed(new:usize,batch:usize,ordinary:usize,banks:u64,memory:u64,pinned:u64)->bool{
        layer_prefill_needed(new,batch,ordinary,banks,layer_prefill_room(memory,pinned,None),512,false)
    }
    #[test]
    fn a_small_shortage_needs_three_full_maximum_batches() {
        let needed=|n,b|needed(n,b,4096,31*G,32*G,G);
        assert!(!needed(2548,256));
        assert!(!needed(3*4096-1,256));assert!(needed(3*4096,256));
        assert!(!needed(3*8192-1,8192));assert!(needed(3*8192,8192));
    }
    #[test]
    fn a_large_shortage_keeps_long_iq2_and_a_follow_up_that_spans_two_actual_batches() {
        assert!(needed(14937,256,4096,39*G,32*G,G));
        assert!(needed(3232,256,2048,39*G,32*G,G));
        assert!(!needed(2048,256,2048,39*G,32*G,G));
        assert!(!needed(3232,256,4096,39*G,32*G,G));
        // A 1.3x shortage is the boundary for the two-batch choice: 16-2-4 = 10 GiB usable.
        assert!(needed(2049,256,2048,13*G,16*G,2*G));
        assert!(!needed(2049,256,2048,13*G-1,16*G,2*G));
    }
    #[test]
    fn fitting_banks_and_unknown_ram_keep_the_ordinary_path() {
        assert!(!needed(30000,256,4096,39*G,64*G,G));
        let usable=32*G-G-4*G;
        assert!(!needed(30000,256,4096,usable,32*G,G));
        assert!(needed(30000,256,4096,usable+1,32*G,G));
        assert!(!needed(30000,256,4096,39*G,0,G));
        assert!(!needed(30000,0,0,0,32*G,G));
    }
    #[test]
    fn memory_another_program_takes_counts_as_a_shortage() {
        // The RTX 5070 Ti at 62 GiB, IQ2_XS banks of 39.4 GiB. Under pressure (4.2 GiB available) the room was 7.9 GiB for a 32.6K prompt,
        // 20.0 GiB for its 3.8K follow-up: layer order 72.7 and 12.7 s at 256 rows, the ordinary path 161.2 and 16.0 s.
        let banks=42283827200;
        for (new,room) in [(32646,8472035328),(3832,21448052736)] {
            let room=layer_prefill_room(62*G,2*G,Some(room));
            assert!(layer_prefill_needed(new,2048,2048,banks,room,256,true));
            assert!(layer_prefill_needed(new,2048,2048,banks,room,256,false));
            // narrower than 128 rows its MoE loses what the drive costs the ordinary path
            assert!(layer_prefill_needed(new,2048,2048,banks,room,128,false));
            assert!(!layer_prefill_needed(new,2048,2048,banks,room,64,false));
        }
        // Unpressured the room was 46 GiB: the banks fit, and only Windows' fit rule takes layer order (8.2K at 512 rows: 17.5 s against
        // 25.4 s); 32.6K at 256 rows and 3.8K at 128 rows stayed ordinary (74.5 and 9.0 s; layer order 73.7 and 16.1 s).
        let room=layer_prefill_room(62*G,2*G,Some(49839136768));
        assert!(layer_prefill_needed(8210,2048,2048,banks,room,512,true));
        assert!(!layer_prefill_needed(32646,2048,2048,banks,room,256,true));
        assert!(!layer_prefill_needed(3832,2048,2048,banks,room,128,true));
        // what the system says never raises what the RAM leaves
        assert_eq!(layer_prefill_room(32*G,G,Some(100*G)),Some(32*G-G-4*G));
        assert_eq!(layer_prefill_room(0,G,Some(100*G)),None);
    }
    #[test]
    fn fitting_banks_take_layer_order_from_three_ordinary_batches_at_the_fit_rows_on_windows() {
        // Windows, banks in RAM, ordinary batch 2048: 8.2K new at 512 rows layer order, 3.8K follow-up at 128 rows and 32.6K at 256 rows
        // ordinary.
        let room=Some(60*G);
        assert!(layer_prefill_needed(8210,2048,2048,50*G,room,512,true));
        assert!(!layer_prefill_needed(3832,2048,2048,50*G,room,128,true));
        assert!(!layer_prefill_needed(32646,2048,2048,50*G,room,256,true));
        assert!(layer_prefill_needed(32646,2048,2048,50*G,room,1024,true));
        assert!(!layer_prefill_needed(3*2048-1,2048,2048,50*G,room,512,true));
        assert!(layer_prefill_needed(3*2048,2048,2048,50*G,room,512,true));
        // Linux keeps the ordinary path for banks that fit (14.9K new at 512 rows: 22.4 s against 23.6 s, 3.7K 10.5 against 6.4).
        assert!(!layer_prefill_needed(14937,2048,2048,50*G,Some(98*G),512,false));
        assert!(!layer_prefill_needed(30000,0,0,0,None,0,true));
    }
}

#[cfg(test)]
mod saved_width_tests {
    use super::chooser_started_at;
    use crate::width::{Chooser, Plan};

    /// One pass of a machine whose pass costs 16 ms and 4 more for each row past the first, every draft kept.
    fn pass(chooser: &mut Chooser) {
        let plan = chooser.plan();
        let rows = plan.drafts + 1;
        chooser.record(plan, plan.drafts, rows, plan.drafts, 0.016 + 0.004 * (rows - 1) as f64);
    }

    /// Decides: a chooser that has made no choice and begun no probe starts at the saved width, whatever passes it has seen (the warm-up's two among
    /// them), and the new one is as a chooser built at that width at the load: nothing seen, nothing chosen, that width in force.
    #[test]
    fn a_chooser_that_has_chosen_nothing_starts_at_the_saved_width() {
        let mut current = Chooser::new(3, 1);
        let started = chooser_started_at(&current, 3, false, 2).expect("a new chooser takes it");
        assert_eq!((started.drafts(), started.passes(), started.decided()), (2, (0, 0), false));
        assert_eq!(chooser_started_at(&current, 3, false, 1).map(|chooser| chooser.drafts()), Some(1), "the width it has");
        // (the warm-up: two requests of two tokens, a pass of one row and no draft each, the first pass of a request left out of the windows)
        for _ in 0..2 {
            current.begin_request();
            pass(&mut current);
        }
        assert_eq!(current.passes(), (2, 0));
        let started = chooser_started_at(&current, 3, false, 3).expect("the warm-up's passes are not a choice");
        assert_eq!((started.drafts(), started.passes()), (3, (0, 0)));
        // (a request's passes that came before the warm start, short of a probe: nothing chosen yet either)
        for _ in 0..100 {
            pass(&mut current);
        }
        assert_eq!(current.passes(), (102, 0));
        assert!(chooser_started_at(&current, 3, false, 2).is_some());
    }

    /// Decides: nothing is taken where the drafts are held, where the saved width is not one the model offers (1 to its most), or once the chooser
    /// has begun a probe or made its first choice, which the saved width would throw away.
    #[test]
    fn a_saved_width_is_not_taken_where_it_would_replace_what_is_held_or_chosen() {
        let current = Chooser::new(3, 1);
        assert!(chooser_started_at(&current, 3, true, 2).is_none(), "held");
        assert!(chooser_started_at(&current, 3, false, 0).is_none(), "no drafts is not a width to start at");
        assert!(chooser_started_at(&current, 3, false, 4).is_none(), "more than the model offers");
        assert!(chooser_started_at(&current, 0, false, 1).is_none(), "a model that offers none");
        let mut probing = Chooser::new(3, 1);
        probing.record(Plan { drafts: 2, probe: true }, 2, 3, 2, 0.02);
        assert_eq!(probing.passes(), (1, 1));
        assert!(chooser_started_at(&probing, 3, false, 2).is_none(), "a probe has begun");
        let mut chosen = Chooser::new(3, 1);
        let mut passes = 0;
        while !chosen.decided() && passes < 20_000 {
            pass(&mut chosen);
            passes += 1;
        }
        assert!(chosen.decided(), "the machine's passes were not enough to choose by");
        assert!(chooser_started_at(&chosen, 3, false, 2).is_none(), "the first choice was made");
        assert!(chosen.passes().1 > 0 && chosen.passes().0 >= crate::width::FIRST_PROBE_AFTER);
    }
}

#[cfg(test)]
mod warm_stop_tests {
    use super::{is_warm_stop, prefill_spans, WARM_STOPPED};

    /// Spans of four are the spans a prompt has always run in, on one device and on two.
    #[test]
    fn a_prompt_runs_in_spans_of_four_layers() {
        assert_eq!(prefill_spans(&[(1, 48)], 4).len(), 1 + 12);
        let spans = prefill_spans(&[(1, 30), (30, 48)], 4);
        assert_eq!(spans[..4], [(0, 1), (1, 5), (5, 9), (9, 13)]);
        assert_eq!(spans[7..], [(25, 29), (29, 30), (30, 34), (34, 38), (38, 42), (42, 46), (46, 48)]);
        // (every layer once, in order)
        let layers: Vec<usize> = spans.iter().flat_map(|(first, last)| *first..*last).collect();
        assert_eq!(layers, (0..48).collect::<Vec<_>>());
    }

    /// While the warm-up can be stopped, each layer is a span of its own, so that a request is seen at every layer.
    #[test]
    fn a_warm_up_pass_runs_in_spans_of_one_layer() {
        let spans = prefill_spans(&[(1, 30), (30, 48)], 1);
        assert_eq!(spans.len(), 48);
        assert!(spans.iter().enumerate().all(|(layer, span)| *span == (layer, layer + 1)));
        assert_eq!(prefill_spans(&[(1, 48)], 1).last(), Some(&(47, 48)));
    }

    /// The stop is known by its words, joined to another error or not, and no other error is taken for it.
    #[test]
    fn a_stopped_warm_up_pass_is_known_by_its_error() {
        assert!(is_warm_stop(WARM_STOPPED));
        assert!(is_warm_stop(&format!("{WARM_STOPPED}; restoring expert slots failed: no memory")));
        assert!(!is_warm_stop("Straydog token or context is out of range"));
        assert!(!is_warm_stop(""));
    }
}

#[cfg(test)]
mod profile_tests {
    use super::{decode_part, decode_steps, Profile};
    use std::collections::BTreeMap;

    /// Without DS41_PROFILE a partially resident model's profile keeps the phases the request line names, from a lone row or a pass of drafts.
    #[test]
    fn a_profile_without_every_phase_keeps_the_steps_parts_alone() {
        let timed = [("step: head", 1.5), ("step: first layer", 1.0), ("step: PLE rows wait", 0.5), ("step: PLE layer", 0.25),
                     ("step: layers (first device)", 30.0), ("verify: head", 2.0), ("draft: next draft", 3.0), ("accept", 0.125),
                     ("note: not a part", 4.0)];
        let kept = |every: bool| {
            let mut profile = Profile::new(every);
            for (name, seconds) in timed {
                profile.add(name, seconds);
            }
            profile.end_pass(1, 0.0);
            profile.phases.into_iter().collect::<Vec<_>>()
        };
        let parts = kept(false);
        assert_eq!(parts.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
                   ["accept", "draft: next draft", "step: PLE layer", "step: PLE rows wait", "step: first layer", "step: head",
                    "step: layers (first device)", "verify: head"]);
        assert!(parts.iter().all(|(name, _)| decode_part(name).is_some()));
        assert_eq!(kept(true).len(), timed.len());
    }

    /// The parts of "other": a lone row and a pass of drafts fold into the same few names. Both devices' layers are one part.
    #[test]
    fn the_steps_phases_are_the_lines_parts() {
        assert_eq!(decode_part("draft: embeddings"), Some("draft"));
        assert_eq!(decode_part("draft: catch-up and first draft"), Some("draft"));
        assert_eq!(decode_part("draft: next draft"), Some("draft"));
        assert_eq!(decode_part("step: head"), Some("head"));
        assert_eq!(decode_part("verify: head"), Some("head"));
        for phase in ["step: first layer", "step: PLE rows wait", "step: PLE layer", "verify: protect, PLE prefetch, embeddings",
                      "verify: first layer", "verify: PLE rows wait", "verify: PLE layer", "PLE prefetch"] {
            assert_eq!(decode_part(phase), Some("first/PLE"), "{phase}");
        }
        assert_eq!(decode_part("accept"), Some("accept"));
        assert_eq!(decode_part("commit: DeltaNet replay"), Some("accept"));
        assert_eq!(decode_part("commit: PLE replay"), Some("accept"));
        for phase in ["step: layers (first device)", "step: layers (second device)", "verify: layers (queued)",
                      "verify: layers (first device)", "verify: layers (second device)"] {
            assert_eq!(decode_part(phase), Some("layers"), "{phase}");
        }
        assert_eq!(decode_part("note: not a part"), None);
    }

    /// A pass of drafts fills the same `steps_ms` parts. The fold does not look at whether the request verified drafts.
    #[test]
    fn a_pass_of_drafts_folds_into_the_same_few_parts() {
        let mut phases = BTreeMap::new();
        for (name, seconds) in [("draft: embeddings", 0.001), ("draft: next draft", 0.002), ("verify: head", 0.0015),
                                ("verify: first layer", 0.0004), ("verify: PLE rows wait", 0.0003), ("PLE prefetch", 0.0002),
                                ("accept", 0.0008), ("commit: DeltaNet replay", 0.0002), ("verify: layers (first device)", 0.004),
                                ("note: not a part", 9.0)] {
            phases.insert(name, seconds);
        }
        let steps = decode_steps(&phases, 2.0);
        let close = |part: &str, ms: f64| assert!((steps[part] - ms).abs() < 1e-9, "{part} {}", steps[part]);
        close("draft", 1.5);
        close("head", 0.75);
        close("first/PLE", 0.45);
        close("accept", 0.5);
        close("layers", 2.0);
        assert_eq!(steps.len(), 5);
    }
}

#[cfg(test)]
mod cut_tests {
    use super::{drawn_cut, PARTIAL_DRAFT_CUT};

    /// The first draft's cut is the one the second row's cost has always given, in passes of one row: c 1.5 gives a quarter, c 1.65 two
    /// thirds, c 1.36 (or less, a row nearly free) none.
    #[test]
    fn the_first_drafts_cut_is_that_of_the_second_rows_cost() {
        let cut = |one: f64, two: f64| drawn_cut(&[one, two, 0.0, 0.0], 1);
        assert!((cut(20.0, 30.0) - 0.25).abs() < 1e-6, "{}", cut(20.0, 30.0));
        assert!((cut(20.0, 33.0) - 0.6625).abs() < 1e-6, "{}", cut(20.0, 33.0));
        assert_eq!(cut(20.0, 27.2), 0.0);
        assert_eq!(cut(20.0, 20.0), 0.0);
        assert_eq!(cut(20.0, 60.0), 0.9);
    }

    /// A later draft is cut by what its own row costs over the row before it, in passes of one row.
    #[test]
    fn a_later_drafts_cut_is_that_of_its_own_rows_cost() {
        // 20 ms a pass of one row, 30 of two (the first draft's row costs half a pass), 32 of three (the second's a tenth), 52 of four (a whole one)
        let seconds = [20.0, 30.0, 32.0, 52.0];
        assert!((drawn_cut(&seconds, 1) - 0.25).abs() < 1e-6);
        assert_eq!(drawn_cut(&seconds, 2), 0.0);
        assert_eq!(drawn_cut(&seconds, 3), 0.9);
    }

    /// Until a draft's row has been timed it is cut as the draft before it, and the first as the fixed cut.
    #[test]
    fn a_draft_whose_row_is_not_timed_is_cut_as_the_one_before() {
        assert_eq!(drawn_cut(&[0.0; 4], 1), PARTIAL_DRAFT_CUT);
        assert_eq!(drawn_cut(&[20.0, 0.0, 0.0, 0.0], 1), PARTIAL_DRAFT_CUT);
        assert_eq!(drawn_cut(&[20.0, 0.0, 0.0, 0.0], 3), PARTIAL_DRAFT_CUT);
        let seconds = [20.0, 30.0, 0.0, 0.0];
        assert_eq!(drawn_cut(&seconds, 2), drawn_cut(&seconds, 1));
        assert_eq!(drawn_cut(&seconds, 3), drawn_cut(&seconds, 1));
    }
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
#[cfg(test)]
mod layer_prefill_tests {
    use super::{layer_prefill_chunks, layer_prefill_rows_named, layer_prefill_split, layer_prefill_stops, LAYER_PREFILL_MIN, LAYER_PREFILL_ROWS as N};

    fn layer_prefill_stops_default(begin: usize, end: usize, checkpoints: &[usize]) -> Vec<usize> {
        layer_prefill_stops(begin, end, checkpoints, N)
    }

    fn layer_prefill_split_default(begin: usize, end: usize, plan: &[(usize, Vec<&'static str>)]) -> (usize, bool) {
        layer_prefill_split(begin, end, plan, N)
    }

    #[test]
    fn body_and_draft_share_the_same_odd_tail() {
        assert_eq!(layer_prefill_chunks(0, 1025, 512), [(0,512),(512,1023),(1023,1025)]);
        assert_eq!(layer_prefill_chunks(7, 16, 8), [(7,14),(14,16)]);
        assert_eq!(layer_prefill_chunks(0, 1024, 512), [(0,512),(512,1024)]);
    }

    /// Decides: where a layer's banks come from. A copy read ahead is used; else a layer the model's marks hold registered is read from the
    /// mapping its locks keep checked; one whose registration was given up between the reader passing it by and its load is read now as a
    /// checked copy, never from the mapping it no longer keeps; with no reader, the mapping as before.
    #[test]
    fn a_layer_whose_registration_went_after_it_was_passed_by_is_read_as_a_checked_copy() {
        use super::{bank_source, BankSource};
        use crate::banks::Registered;
        assert_eq!(bank_source(true, true, true), BankSource::Copy);
        assert_eq!(bank_source(true, false, true), BankSource::Copy);
        assert_eq!(bank_source(false, true, true), BankSource::Registered);
        assert_eq!(bank_source(false, true, false), BankSource::Registered);
        assert_eq!(bank_source(false, false, true), BankSource::ReadNow);
        assert_eq!(bank_source(false, false, false), BankSource::Mapping);
        // (the race: registered when the reader passes it by, given up before its load)
        let registered = std::sync::Arc::new(Registered::default());
        registered.mark(9);
        let passed_by = registered.is_registered(9);
        registered.release(9);
        let held = registered.read(9);
        assert!(passed_by && held.is_none());
        assert_eq!(bank_source(false, held.is_some(), true), BankSource::ReadNow);
    }

    /// Decides: reads of 16384 new tokens or more scan their DeltaNet layers in chunks, shorter ones (the pinned tests' prompts) token by token as
    /// they did, and STRAYDOG_DELTA_SCAN=token keeps every read on the token-by-token scan.
    #[test]
    fn long_reads_scan_in_chunks_unless_the_variable_says_token() {
        use super::{chunked_scan_from_named, CHUNKED_SCAN_READ};
        assert_eq!(chunked_scan_from_named(None), CHUNKED_SCAN_READ);
        assert_eq!(chunked_scan_from_named(Some("chunks")), CHUNKED_SCAN_READ);
        assert_eq!(chunked_scan_from_named(Some(" token ")), usize::MAX);
        assert!(CHUNKED_SCAN_READ > 8192 && CHUNKED_SCAN_READ <= 16_384);
    }

    /// Decides: reads of 16384 new tokens or more run the HyperConnections' up projection and finish as one kernel, shorter ones keep the pair
    /// (the pinned tests' prompts keep their bits), and STRAYDOG_HC=blas keeps every read on the pair.
    #[test]
    fn long_reads_fuse_the_hyperconnection_unless_the_variable_says_blas() {
        use super::{hc_fused_from_named, CHUNKED_SCAN_READ};
        assert_eq!(hc_fused_from_named(None), CHUNKED_SCAN_READ);
        assert_eq!(hc_fused_from_named(Some("fused")), CHUNKED_SCAN_READ);
        assert_eq!(hc_fused_from_named(Some(" blas ")), usize::MAX);
    }

    /// Decides: a K/V starts at 32768 positions (the context if that is smaller), at STRAYDOG_KV_START's in whole thousands of them, and at the whole
    /// context where it says 0 or a number of positions the context does not have; a value that is no number is not one.
    #[test]
    fn the_kv_starts_at_32768_unless_the_variable_names_another_start() {
        use super::{starting_kv_named, STARTING_KV};
        assert_eq!(starting_kv_named(None, 262_144), STARTING_KV);
        assert_eq!(starting_kv_named(None, 16_384), 16_384);
        assert_eq!(starting_kv_named(Some("65536"), 262_144), 65_536);
        assert_eq!(starting_kv_named(Some(" 40000 "), 262_144), 40_960);
        assert_eq!(starting_kv_named(Some("0"), 262_144), 262_144);
        assert_eq!(starting_kv_named(Some("999999"), 262_144), 262_144);
        assert_eq!(starting_kv_named(Some("many"), 262_144), STARTING_KV);
    }

    /// Decides: STRAYDOG_LAYER_ROWS_VRAM=0 keeps a pass's rows on the host; unset or anything else holds them in VRAM where they fit.
    #[test]
    fn the_rows_are_held_in_vram_unless_the_variable_says_0() {
        use super::layer_rows_vram_named;
        assert!(layer_rows_vram_named(None) && layer_rows_vram_named(Some("1")) && layer_rows_vram_named(Some("yes")));
        assert!(!layer_rows_vram_named(Some("0")) && !layer_rows_vram_named(Some(" 0 ")));
    }

    /// Decides: STRAYDOG_LAYER_PREFILL_ROWS sets the rows a pass takes, a count from the least a layer-order segment takes to the
    /// context's limit; anything else keeps 16,384.
    #[test]
    fn the_rows_a_pass_takes_are_the_variables_within_bounds() {
        assert_eq!(layer_prefill_rows_named(None), N);
        assert_eq!(layer_prefill_rows_named(Some("65536")), 65_536);
        assert_eq!(layer_prefill_rows_named(Some("131_072")), 131_072);
        assert_eq!(layer_prefill_rows_named(Some(" 262144 ")), 262_144);
        assert_eq!(layer_prefill_rows_named(Some(&LAYER_PREFILL_MIN.to_string())), LAYER_PREFILL_MIN);
        for refused in ["511", "0", "262145", "many", "", "-1"] {
            assert_eq!(layer_prefill_rows_named(Some(refused)), N, "{refused}");
        }
    }

    /// Decides: a pass of more rows takes a long prompt in fewer passes, each still stopping at a checkpoint: 249,925 tokens in 16 at 16,384,
    /// 4 at 65,536 and 1 at 262,144; a checkpoint inside a pass splits it there; the short last segment is left to the ordinary path the same.
    #[test]
    fn more_rows_a_pass_take_a_long_prompt_in_fewer_passes() {
        let passes = |rows: usize| layer_prefill_stops(0, 249_925, &[], rows).len();
        assert_eq!((passes(N), passes(65_536), passes(131_072), passes(262_144)), (16, 4, 2, 1));
        assert_eq!(layer_prefill_stops(0, 249_925, &[], 65_536), [65_536, 131_072, 196_608, 249_925]);
        assert_eq!(layer_prefill_stops(0, 200_000, &[100_000], 65_536), [65_536, 100_000, 165_536, 200_000]);
        assert_eq!(layer_prefill_stops(1_000, 250_000, &[249_800], 262_144), [249_800, 250_000]);
        let at = |stops: &[usize]| stops.iter().map(|at| (*at, Vec::new())).collect::<Vec<(usize, Vec<&'static str>)>>();
        assert_eq!(layer_prefill_split(1_000, 250_000, &at(&[249_800]), 262_144), (249_800, false));
        assert_eq!(layer_prefill_split(0, 65_537, &at(&[]), 65_536), (65_536, false));
        assert_eq!(layer_prefill_split(0, 70_000, &at(&[]), 65_536), (70_000, false));
        // (each pass covers its rows once and the passes cover the prompt)
        for rows in [LAYER_PREFILL_MIN, N, 65_536, 262_144] {
            let stops = layer_prefill_stops(5, 249_930, &[], rows);
            let starts: Vec<usize> = [5].into_iter().chain(stops.iter().copied()).collect();
            assert!(stops.iter().zip(&starts).all(|(stop, start)| stop > start && stop - start <= rows), "{rows}");
            assert_eq!(*stops.last().unwrap(), 249_930);
        }
    }

    #[test]
    fn layer_segments_bound_work_without_crossing_a_checkpoint() {
        assert_eq!(layer_prefill_stops_default(0, N, &[]), [N]);
        assert_eq!(layer_prefill_stops_default(0, N + 1, &[]), [N, N + 1]);
        assert_eq!(layer_prefill_stops_default(0, 2 * N, &[N]), [N, 2 * N]);
        assert_eq!(layer_prefill_stops_default(0, N + 1, &[N]), [N, N + 1]);
        assert_eq!(layer_prefill_stops_default(0, N + 2, &[N + 1]), [N, N + 1, N + 2]);
        // a short last segment is left to the ordinary path after the layer order; a short segment before a long one keeps the staging slots
        let at = |stops: &[usize]| stops.iter().map(|at| (*at, Vec::new())).collect::<Vec<(usize, Vec<&'static str>)>>();
        assert_eq!(layer_prefill_split_default(0, 2 * N, &at(&[N])), (2 * N, false));
        assert_eq!(layer_prefill_split_default(0, N + 1, &at(&[])), (N, false));
        assert_eq!(layer_prefill_split_default(0, 32646, &at(&[32640])), (32640, false));
        assert_eq!(layer_prefill_split_default(100, 5000, &at(&[400])), (5000, true));
        assert_eq!(layer_prefill_split_default(100, 5000, &at(&[612])), (5000, false));
        assert_eq!(layer_prefill_split_default(0, 5000, &at(&[4600])), (4600, false));
        assert_eq!(layer_prefill_split_default(0, 5000, &at(&[4488, 9000])), (5000, false));
        assert_eq!(layer_prefill_split_default(0, 5000, &at(&[300, 4990])), (5000, true));
        assert!(layer_prefill_stops_default(7, 7, &[7]).is_empty());
        assert_eq!(layer_prefill_stops_default(100, 2 * N + 100, &[N + 100]), [N + 100, 2 * N + 100]);
        assert_eq!(layer_prefill_stops_default(0, 20000, &[1000, 1000]), [1000, 17384, 20000]);
    }
}

#[cfg(all(test, feature = "prefill-observer", any(feature = "hip", feature = "cuda")))]
#[path = "layer_prefill_diagnostics.rs"]
mod layer_prefill_diagnostics;

#[path = "model_pipeline.rs"]
mod pipeline;
