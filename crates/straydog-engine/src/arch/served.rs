//! What a served model of any architecture does with its placed engines (docs/DESIGN.md §10.1,
//! layer 3): continue the sequence they hold when a prompt extends it (or, with engines that keep
//! checkpoints, from where an earlier prompt's reply began), sample, report a request's metrics,
//! and warm the expert stores up with an earlier run's contents. An architecture places its
//! engines (its `serve` module) and drives them through [`Engines`].

use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::constraint::{pick, Constraint};
use crate::drafts::{Ngrams, Policy};
use crate::experts::Contents;
use crate::gpu::Backend;
use crate::group::SplitMode;
use crate::model::Rng;
use crate::pinned::Pinned;
use crate::sampling::{self, Generated, Sampling};

/// An architecture's placed engines, as a session drives them.
pub trait Engines {
    /// One pass over `tokens` (at most a batch) continuing the held sequence: the logits after
    /// the last.
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String>;
    /// Forget the sequence (warm experts stay).
    fn reset(&mut self);
    /// The expert stores' (VRAM hits, host-tier hits, drive reads) so far.
    fn counts(&self) -> (u64, u64, u64);
    /// Routes the host CPU ran so far (experts computed where the host tier holds them).
    fn cpu_routes(&self) -> u64 {
        0
    }
    /// The bytes the expert stores read from the drive so far (every device's share).
    fn drive_bytes(&self) -> u64 {
        0
    }
    /// The expert stores' counters so far (every device's), when the engines stream experts.
    fn store_counts(&self) -> Option<crate::experts::Counts> {
        None
    }
    /// How often the engines' own drafter was asked for a draft since the last call, and how many
    /// tokens each verified (index 0: none, the draft thrown away), when there is one.
    fn take_draft_stats(&mut self) -> Option<(usize, Vec<usize>)> {
        None
    }
    /// Seconds per phase of the passes since the last call, when the engines profile them
    /// (DS41_PROFILE), else nothing; the count restarts.
    fn take_profile(&mut self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
    /// The experts the stores hold, least recently used first: in VRAM, and only in the host tier.
    fn contents(&self) -> Contents;
    /// Start warming the stores up with these experts, least recently used first: the reads in
    /// flight, and how far into `vram` and `host` it went (the rest waits for room).
    fn warm(&mut self, vram: &[(usize, usize)], host: &[(usize, usize)]) -> Result<(usize, usize, usize), String>;
    /// Of a warm-up's lists (least recently used first), what the stores' slots hold, each
    /// expert's layer to whichever store holds that layer now: the most recent of each.
    fn fit_warm(&self, vram: Vec<(usize, usize)>, host: Vec<(usize, usize)>) -> Contents {
        (vram, host)
    }
    /// Take the landed reads, without waiting; the reads still in flight.
    fn poll(&mut self) -> Result<usize, String>;
    /// The devices that each read part of an expert.
    fn devices_per_expert(&self) -> usize;
    fn gpus(&self) -> Vec<&Backend>;
    /// Wait for the stores' reads and copies (before the process ends).
    fn settle(&mut self) -> Result<(), String>;
    /// After the first passes (the runtime has taken what it keeps beside the counted buffers): lend
    /// the expert stores the VRAM the cards still have spare. Each engine's expert slots then; none:
    /// nothing is lent.
    fn lend_spare(&mut self) -> Result<Vec<usize>, String> {
        Ok(Vec::new())
    }
    /// Between requests: a few of the stores' write-backs that make room for the next prefill's
    /// experts; whether the room is made.
    fn make_room(&mut self) -> Result<bool, String> {
        Ok(true)
    }
    /// Start reading what passes over `tokens` (continuing the held sequence) will read from the
    /// drive (DeepSeek-V4.1's Engram rows), behind the pass before them; a pass reads them itself
    /// otherwise.
    fn prefetch_ahead(&mut self, _tokens: &[u32]) -> Result<(), String> {
        Ok(())
    }
    /// Whether the engines read images (DeepSeek-V4.1 beside its vision encoder).
    fn vision(&self) -> bool {
        false
    }
    /// Each image's span rows ([`crate::vision::span`]), the model's width a position.
    fn encode_images(&mut self, _images: &[crate::vision::Patches]) -> Result<Vec<Vec<f32>>, String> {
        Err("this model reads no images".into())
    }
    /// The rows the next passes embed at image positions: each span's first position and its rows.
    fn set_images(&mut self, _spans: Vec<(usize, std::sync::Arc<Vec<f32>>)>) {}
    /// Copies of the sequence state the engines keep ([`Engines::checkpoint`]); none: a session
    /// continues only a sequence the prompt extends.
    fn checkpoints(&self) -> usize {
        0
    }
    /// Save the sequence state (its position too) to checkpoint `slot`, or restore it from there.
    /// A restore keeps the K/V below the checkpoint's position, which must be its sequence's,
    /// unless the checkpoints are whole ([`Engines::whole_checkpoints`]).
    fn checkpoint(&mut self, _slot: usize, _save: bool) -> Result<(), String> {
        Err("these engines keep no checkpoints".into())
    }
    /// The checkpoint slot a budget gives up last (the conversation root's), if any.
    fn keep_checkpoint(&mut self, _slot: Option<usize>) {}
    /// Whether a checkpoint holds the whole sequence state, positional too: restored after any
    /// other sequence (a conversation come back to after another), not only below a common prefix.
    fn whole_checkpoints(&self) -> bool {
        false
    }
    /// Whether checkpoint `slot` still holds its state (whole checkpoints may give theirs up to
    /// a budget).
    fn holds(&self, _slot: usize) -> bool {
        true
    }
    /// The most tokens [`Engines::verify`] takes in a pass (1: the engines verify no drafts).
    fn verify_rows(&self) -> usize {
        1
    }
    /// Propose tokens after `anchor` (chosen but not fed yet), without advancing target state.
    /// None uses prompt lookup; Some(empty) declines this round without that fallback.
    /// Candidates are capped at `max` and the request's remaining output. After verification,
    /// [`Engines::rewind`] must also restore any drafter state to the retained prefix.
    fn draft(&mut self, _anchor: u32, _max: usize) -> Result<Option<Vec<u32>>, String> {
        Ok(None)
    }
    /// One pass over `tokens` (at most [`Engines::verify_rows`]) continuing the held sequence: the
    /// logits after each, `tokens.len()` rows of the vocabulary. Each row is what feeding the
    /// tokens one at a time gives.
    fn verify(&mut self, _tokens: &[u32]) -> Result<Vec<f32>, String> {
        Err("these engines verify no drafts".into())
    }
    /// After a verify: keep the first `keep` of its tokens, the state as though only they had been
    /// fed.
    fn rewind(&mut self, _keep: usize) -> Result<(), String> {
        Err("these engines verify no drafts".into())
    }
}

/// Per GPU: VRAM kept beside what a placement counts (dense weights, sequence state, a pass's rows,
/// the scratch and the expert slots) for the runtime's own memory, which Straydog's budget does
/// not see: code, BLAS handles, graphs, kernels' scratch. Measured as the card's peak (sysfs) less
/// the counted bytes over cold, follow-up, 4K-document and warm requests: DeepSeek-V4.1 on two
/// R9700s 0.83-0.97 GiB (a 12.9K-token prefill included), Qwen3.6 at --vram 6 on one 0.93-1.04
/// GiB (more with a larger prefill batch). 1.5 GiB left the card 0.06 GiB under the budget at its
/// peak; this keeps ~0.2 GiB.
pub const RESERVE: u64 = 13 << 27;
pub const SCRATCH: u64 = 512 << 20;
/// Without a host tier budget, Straydog takes this share of the host memory in all, and the
/// experts' host tier what its other host memory leaves of it: the row cache, and the rest below.
/// (On Doggybox, 125 GiB, DeepSeek-V4.1's tier is then 103 GiB, Straydog's own memory 104 GiB:
/// an agent's turn reads half what it did from the drive at 0.75.)
pub const HOST_SHARE: f64 = 0.85;
/// The worker's host memory beside the tier and the row cache: the GPU runtime, the verity
/// trees' level-0 caches, read buffers and the passes' host copies.
pub const HOST_OTHER: u64 = 2 << 30;

/// The experts' host tier: as given, else what Straydog's other host memory (the row cache and
/// the rest) leaves of its share of the host memory; never more than the experts VRAM lacks.
pub fn host_tier(settings: &Settings, beyond_vram: u64) -> u64 {
    settings.host_expert_bytes.unwrap_or_else(|| {
        ((settings.host_memory_bytes as f64 * HOST_SHARE) as u64).saturating_sub(settings.row_cache_bytes + HOST_OTHER).min(beyond_vram)
    })
}

/// `parts` consecutive layer ranges covering `layers`, cut only at `cuts`, the fullest part (its
/// `load` over its device's budget, `budgets` in part order) as empty as it can be; None when
/// there are too few cuts.
pub fn split_layers(layers: usize, cuts: &[usize], load: &dyn Fn(std::ops::Range<usize>) -> u64, budgets: &[u64])
                    -> Option<Vec<std::ops::Range<usize>>> {
    let parts = budgets.len();
    if parts <= 1 {
        return Some(std::iter::once(0..layers).collect());
    }
    if cuts.len() + 1 < parts {
        return None;
    }
    // Over every choice of cuts (few layers: exhaustive DP). best[part][end]: the fullest part's
    // share of its budget and the part ends, for `part` parts ending at `end`.
    type Choice = Option<(u128, Vec<usize>)>;
    let share = |range: std::ops::Range<usize>, part: usize| ((load(range) as u128) << 64) / budgets[part].max(1) as u128;
    let mut best: Vec<Vec<Choice>> = vec![vec![None; layers + 1]; parts + 1];
    best[0][0] = Some((0, Vec::new()));
    for part in 1..=parts {
        let ends: Vec<usize> = if part == parts { vec![layers] } else { cuts.to_vec() };
        for &end in &ends {
            let starts = std::iter::once(0).chain(cuts.iter().copied()).filter(|&start| start < end);
            let mut choice: Choice = None;
            for start in starts {
                if let Some((worst, path)) = &best[part - 1][start] {
                    let candidate = (*worst).max(share(start..end, part - 1));
                    if choice.as_ref().is_none_or(|(value, _)| candidate < *value) {
                        let mut path = path.clone();
                        path.push(end);
                        choice = Some((candidate, path));
                    }
                }
            }
            best[part][end] = choice;
        }
    }
    let (_, ends) = best[parts][layers].clone()?;
    let mut start = 0;
    Some(ends.into_iter().map(|end| { let range = start..end; start = end; range }).collect())
}

/// How a served model is placed.
pub struct Settings {
    pub devices: Vec<usize>,
    /// With several GPUs: Experts splits every layer across them (experts by rows), Layer gives
    /// each a range of layers.
    pub split_mode: SplitMode,
    /// Each GPU's VRAM budget, in `devices` order.
    pub vram_bytes: Vec<u64>,
    pub context: usize,
    /// Most tokens one pass takes (a prompt is prefilled in chunks of this many).
    pub batch: usize,
    /// The experts' host tier; None plans it from `host_memory_bytes`.
    pub host_expert_bytes: Option<u64>,
    pub host_memory_bytes: u64,
    /// Host memory caching the gathered tables' rows (the Engram table), read from the drive as
    /// needed and verified as read; 0 reads them through the mapping.
    pub row_cache_bytes: u64,
    /// Prefill products as FP16 GEMMs with F32 sums (`--gemm fp16`), where an architecture takes them.
    pub half_gemm: bool,
}

/// Experts still to warm up with, as (layer, expert).
type Pending = std::collections::VecDeque<(usize, usize)>;

/// The sequence state saved where a prompt's reply began (or, the root, where its system prompt
/// ended), and the tokens before it.
struct Checkpoint {
    slot: usize,
    tokens: Vec<u32>,
    root: bool,
}

/// The fewest tokens a system prompt has to be checkpointed as a conversation's root (a shorter
/// one prefills in about a second).
const ROOT_MINIMUM: usize = 1024;

/// Warm-up steps (a millisecond apart, the worker's) with no room and no read on its way before
/// the warm-up gives up.
const WARM_PATIENCE: usize = 30_000;

/// How far before a prompt's end its reply's generation prompt may begin to be checkpointed.
const TURN_REACH: usize = 64;

/// A placed model serving requests one at a time.
pub struct Session<E: Engines> {
    pub engines: E,
    files: &'static Pinned,
    vocabulary: usize,
    /// Layers and experts per layer (a warm-up state's pairs must lie within).
    layers: usize,
    experts: usize,
    pub context: usize,
    pub batch: usize,
    /// The tokens the engines hold, in order.
    history: Vec<u32>,
    /// Tokens that end a reply (unsent).
    end: Vec<u32>,
    /// The token a reply's generation prompt begins with (`<|im_start|>`, `<｜Assistant｜>`): the
    /// state before it is checkpointed when the engines keep checkpoints.
    turn_start: Option<u32>,
    /// The token a user message begins with (`<｜User｜>`): the state before a prompt's first is
    /// its conversation's root (system prompt and tools), checkpointed for the next conversation.
    root_end: Option<u32>,
    /// Saved states, oldest first: the last ("turn") where the last prompt's reply began, the
    /// earlier ones ("prior") where earlier prompts' did, and the last root ("root").
    checkpoints: Vec<Checkpoint>,
    /// The last request's prompt tokens that agreed with what was held or checkpointed (more than
    /// it reused when it changed what came before).
    agreed: usize,
    /// The sampling the model card gives, for every mode.
    pub card: Sampling,
    /// How the model was placed (the ready frame shows it).
    pub placement: Value,
    pub state_bytes: u64,
    pub load_seconds: Vec<(String, f64)>,
    pub last_metrics: Map<String, Value>,
    /// Tests only: when set, each request's prompt logits (those its first token is chosen from).
    #[doc(hidden)]
    pub prompt_logits: Option<Vec<f32>>,
    /// An earlier run's experts still to warm up with: (VRAM, host tier), least recently used
    /// first, taken from the front.
    warming: (Pending, Pending),
    /// Warm-up steps in a row that found no room and no read on its way.
    warm_stalled: usize,
}

/// What an architecture's placement gives a session.
pub struct Placed<E: Engines> {
    pub engines: E,
    pub files: &'static Pinned,
    pub vocabulary: usize,
    pub layers: usize,
    pub experts: usize,
    pub context: usize,
    pub batch: usize,
    pub end: Vec<u32>,
    pub turn_start: Option<u32>,
    pub root_end: Option<u32>,
    pub card: Sampling,
    pub placement: Value,
    pub state_bytes: u64,
    pub load_seconds: Vec<(String, f64)>,
}

/// The sampling a GGUF's `general.sampling.*` gives (the model card's), else neutral.
pub fn card(header: &straydog_gguf::ModelHeader) -> Result<Sampling, String> {
    let number = |key: &str| header.metadata.get(key).and_then(|value| value.float());
    let card = Sampling {
        temperature: number("general.sampling.temp").unwrap_or(1.0),
        top_k: number("general.sampling.top_k").map_or(0, |value| value.max(0.0) as usize),
        top_p: number("general.sampling.top_p").unwrap_or(1.0),
        min_p: number("general.sampling.min_p").unwrap_or(0.0),
        presence_penalty: 0.0,
    };
    card.check()?;
    Ok(card)
}

/// What a reply's decoding did: tokens emitted, tokens fed in how many passes, drafts verified and
/// accepted.
#[derive(Debug, Default, PartialEq)]
struct Decoded {
    emitted: usize,
    fed: usize,
    passes: usize,
    drafted: usize,
    accepted: usize,
    engine_drafted: usize,
    engine_accepted: usize,
}

/// A prompt-lookup match of this many tokens (the lookup policy's widest) is verified without
/// asking the engines for a draft ([`Engines::draft`]).
const LONG_MATCH: usize = 8;

/// Sample from `logits` (after the held sequence, `history`) and continue it: up to `max_new`
/// tokens, each to `emit`; an end token ends the reply unsent. With engines that verify drafts,
/// each token is fed with engine drafts, else prompt-lookup drafts (drafts::Ngrams), in one pass;
/// the drafts the target distribution accepts (sampling::accept: exact for any sampling) are
/// emitted after it and kept, the rest rewound. A `constraint` chooses every token itself (no
/// drafts).
#[allow(clippy::too_many_arguments)]
fn decode<E: Engines>(engines: &mut E, history: &mut Vec<u32>, logits: Vec<f32>, vocabulary: usize, end: &[u32], max_new: usize,
                      sampling: &Sampling, seed: u64, cancelled: &dyn Fn() -> bool, emit: &mut dyn FnMut(u32),
                      mut constraint: Option<&mut Constraint>) -> Result<Decoded, String> {
    let mut rng = Rng::new(seed);
    let mut generated = Generated::new(vocabulary);
    let rows = if constraint.is_some() { 1 } else { engines.verify_rows() };
    let mut ngrams = Ngrams::new();
    if rows > 1 {
        ngrams.extend(history);
    }
    let mut policy = Policy::new(rows);
    let mut done = Decoded::default();
    let mut next = pick(&logits, sampling, &generated, &mut rng, &mut constraint)?;
    loop {
        if cancelled() || end.contains(&next) {
            break;
        }
        generated.insert(next);
        emit(next);
        done.emitted += 1;
        if done.emitted == max_new {
            break;
        }
        let (drafts, engine_draft) = if rows > 1 {
            ngrams.push(next);
            // A long match is verified as it is: those drafts are accepted nearly always (an edit
            // repeats its source), and an engine's draft costs milliseconds to be no better.
            let (proposal, matched) = ngrams.propose(rows - 1);
            let mut lookup = None;
            if matched >= LONG_MATCH {
                lookup = Some(policy.width(matched).min(proposal.len()));
            }
            match lookup.filter(|&width| width > 0) {
                Some(width) => (proposal[..width.min(max_new - done.emitted)].to_vec(), false),
                None => match engines.draft(next, rows - 1)? {
                    Some(mut proposal) => {
                        proposal.truncate((rows - 1).min(max_new - done.emitted));
                        if proposal.iter().any(|&token| token as usize >= vocabulary) {
                            return Err("Straydog engine drafted a token out of range".into());
                        }
                        (proposal, true)
                    }
                    None => {
                        let width = lookup.unwrap_or_else(|| policy.width(matched)).min(proposal.len()).min(max_new - done.emitted);
                        (proposal[..width].to_vec(), false)
                    }
                },
            }
        } else {
            (Vec::new(), false)
        };
        if drafts.is_empty() {
            let logits = engines.forward(&[next])?;
            history.push(next);
            (done.passes, done.fed) = (done.passes + 1, done.fed + 1);
            next = pick(&logits, sampling, &generated, &mut rng, &mut constraint)?;
            continue;
        }
        let fed: Vec<u32> = std::iter::once(next).chain(drafts.iter().copied()).collect();
        let rows_logits = engines.verify(&fed)?;
        if rows_logits.len() != fed.len() * vocabulary {
            return Err("Straydog engines verified the wrong number of logits".into());
        }
        (done.passes, done.fed, done.drafted) = (done.passes + 1, done.fed + fed.len(), done.drafted + drafts.len());
        if engine_draft { done.engine_drafted += drafts.len(); }
        let (accepted, token) = sampling::accept(&drafts, &rows_logits, vocabulary, sampling, &mut generated, &mut rng);
        if !engine_draft { policy.record(drafts.len(), accepted); }
        // The accepted drafts go out in order: an end token among them ends the reply unsent, the
        // last token allowed ends it sent; either is left out of the state, as without drafts.
        let (mut kept, mut stop) = (accepted, false);
        for (index, &draft) in drafts[..accepted].iter().enumerate() {
            if end.contains(&draft) {
                (kept, stop) = (index, true);
                break;
            }
            emit(draft);
            done.emitted += 1;
            if done.emitted == max_new {
                (kept, stop) = (index, true);
                break;
            }
        }
        engines.rewind(1 + kept)?;
        done.accepted += kept;
        if engine_draft { done.engine_accepted += kept; }
        history.push(next);
        history.extend_from_slice(&drafts[..kept]);
        ngrams.extend(&drafts[..kept]);
        if stop {
            break;
        }
        next = token;
    }
    Ok(done)
}

impl<E: Engines> Session<E> {
    pub fn new(placed: Placed<E>) -> Session<E> {
        Session {
            engines: placed.engines, files: placed.files, vocabulary: placed.vocabulary, layers: placed.layers, experts: placed.experts,
            context: placed.context, batch: placed.batch, history: Vec::new(), end: placed.end, turn_start: placed.turn_start,
            root_end: placed.root_end, checkpoints: Vec::new(), agreed: 0, card: placed.card,
            placement: placed.placement, state_bytes: placed.state_bytes, load_seconds: placed.load_seconds,
            last_metrics: Map::new(), prompt_logits: None, warming: Default::default(), warm_stalled: 0,
        }
    }

    pub fn vocabulary(&self) -> usize {
        self.vocabulary
    }

    /// The tokens that end a reply.
    pub fn end_tokens(&self) -> &[u32] {
        &self.end
    }

    /// Tests only: forget the held sequence and every checkpoint (the engines' copies are then
    /// unused), as a fresh start does, to run the same requests again from nothing.
    #[doc(hidden)]
    pub fn forget(&mut self) {
        self.reset();
        self.checkpoints.clear();
    }

    /// The model's metadata, authenticated.
    pub fn header(&self) -> &straydog_gguf::ModelHeader {
        &self.files.header
    }

    /// The model's files (resources beside them are read through the bundle's pins).
    pub fn files(&self) -> &'static Pinned {
        self.files
    }

    /// Forget the sequence (warm experts stay).
    pub fn reset(&mut self) {
        self.history.clear();
        self.engines.reset();
    }

    /// The GPUs as the server reports them: summed, then each.
    pub fn info(&self) -> Result<Value, String> {
        let devices = self.engines.gpus().into_iter().map(|gpu| gpu.info()).collect::<Result<Vec<_>, _>>()?;
        let names: Vec<&str> = devices.iter().map(|device| device["name"].as_str().unwrap_or("")).collect();
        let mut info = json!({"name": names.join(" + "), "device": devices.first().map_or(Value::Null, |device| device["device"].clone())});
        for key in ["free_bytes", "total_bytes", "allocated_bytes", "budget_bytes"] {
            info[key] = devices.iter().map(|device| device[key].as_u64().unwrap_or(0)).sum::<u64>().into();
        }
        let (hits, host, drive) = self.engines.counts();
        info["experts"] = json!({"vram_hits": hits, "host_hits": host, "drive_reads": drive});
        info["devices"] = devices.into();
        Ok(info)
    }

    /// Feed `tokens` (any count) through the engines in passes of at most the batch; the logits
    /// after the last. Cancelled between passes: None, with the history at what was fed.
    fn feed(&mut self, tokens: &[u32], cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize)) -> Result<Option<Vec<f32>>, String> {
        let mut logits = None;
        for (index, piece) in tokens.chunks(self.batch).enumerate() {
            if index > 0 && cancelled() {
                return Ok(None);
            }
            // (the next pass's drive reads start before this pass, to land while it runs)
            let (first, ahead) = (index * self.batch, ((index + 2) * self.batch).min(tokens.len()));
            if ahead > first + piece.len() {
                if let Err(error) = self.engines.prefetch_ahead(&tokens[first..ahead]) {
                    self.reset();
                    return Err(error);
                }
            }
            match self.engines.forward(piece) {
                Ok(output) => logits = Some(output),
                Err(error) => {
                    self.reset();
                    return Err(error);
                }
            }
            self.history.extend_from_slice(piece);
            progress(self.history.len());
        }
        Ok(logits)
    }

    /// Continue the held sequence over `tokens` (or start anew when they do not extend it) and
    /// sample up to `max_new` tokens, each passed to `emit`; an end token ends it unsent.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(&mut self, tokens: &[u32], max_new: usize, sampling: &Sampling, seed: u64, cancelled: &dyn Fn() -> bool,
                    progress: &mut dyn FnMut(usize, usize), emit: &mut dyn FnMut(u32)) -> Result<(), String> {
        self.generate_constrained(tokens, max_new, sampling, seed, cancelled, progress, emit, None)
    }

    /// [`Session::generate`] with the reply held to `constraint` ([`crate::constraint`]).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_constrained(&mut self, tokens: &[u32], max_new: usize, sampling: &Sampling, seed: u64,
                                cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize), emit: &mut dyn FnMut(u32),
                                constraint: Option<&mut Constraint>) -> Result<(), String> {
        let started = Instant::now();
        self.last_metrics = Map::new();
        if tokens.is_empty() || max_new == 0 || tokens.len() + max_new > self.context {
            return Err("Straydog prompt plus output exceeds the allocated context".into());
        }
        // (an image position's id is past the vocabulary: engines that read images take its rows)
        let vision = self.engines.vision();
        if tokens.iter().any(|&token| token as usize >= self.vocabulary && !(vision && token >= crate::vision::IMAGE_TOKEN)) {
            return Err("Straydog token is out of range".into());
        }
        sampling.check()?;
        self.files.check_unchanged()?;
        self.agreed = 0;
        let (reused, resumed_from) = self.resume(tokens)?;
        let agreed = self.agreed.max(reused);
        let resume_seconds = started.elapsed().as_secs_f64();
        let total = tokens.len();
        // (where the prefill's experts came from: the wait before the first token, broken down)
        let (hits0, host0, drive0) = self.engines.counts();
        // (a profile counts from here: the time since the last request is not this prefill's)
        drop(self.engines.take_profile());
        let drive_bytes0 = self.engines.drive_bytes();
        let store0 = self.engines.store_counts();
        // The state where this reply's generation prompt begins is saved on the way: the next
        // turn's prompt keeps what precedes it even where the template renders this reply
        // otherwise than it was generated (a past turn's thinking dropped).
        // So is the state where its system prompt ends: the next conversation keeps that.
        let mut fed = reused;
        // (the seconds the checkpoints took: copies of the whole state, made before the first token)
        let mut checkpoint_seconds = 0.0;
        if let Some(root) = self.root_mark(tokens, reused) {
            if self.feed(&tokens[reused..root], cancelled, &mut |done| progress(done, total))?.is_none() {
                return Ok(());
            }
            let saving = Instant::now();
            self.save_checkpoint(true)?;
            checkpoint_seconds += saving.elapsed().as_secs_f64();
            fed = root;
        }
        if let Some(start) = self.turn_mark(tokens, fed) {
            if self.feed(&tokens[fed..start], cancelled, &mut |done| progress(done, total))?.is_none() {
                return Ok(());
            }
            let saving = Instant::now();
            self.save_checkpoint(false)?;
            checkpoint_seconds += saving.elapsed().as_secs_f64();
            fed = start;
        }
        let Some(logits) = self.feed(&tokens[fed..], cancelled, &mut |done| progress(done, total))? else {
            return Ok(());
        };
        if let Some(kept) = self.prompt_logits.as_mut() {
            kept.clone_from(&logits);
        }
        let prefill_seconds = started.elapsed().as_secs_f64();
        let (hits, host, drive) = self.engines.counts();
        let prefill_drive_bytes = self.engines.drive_bytes() - drive_bytes0;
        // (what else the prefill's experts cost: write-backs to the host tier, prefetches it did not
        // use, and the seconds the stores waited for reads, for free slots and for pinning)
        let prefill_store = match (store0, self.engines.store_counts()) {
            (Some(before), Some(after)) => json!({
                "writebacks": after.writebacks - before.writebacks,
                "host_drops": after.host_drops - before.host_drops,
                "host_drops_short": after.host_drops_short - before.host_drops_short,
                "prefetched": after.prefetched - before.prefetched,
                "prefetch_unused": after.prefetch_unused - before.prefetch_unused,
                "wait_seconds": after.wait_seconds - before.wait_seconds,
                "slot_wait_seconds": after.slot_wait_seconds - before.slot_wait_seconds,
                "pin_seconds": after.pin_seconds - before.pin_seconds,
            }),
            _ => Value::Null,
        };
        let prefill_profile: Map<String, Value> = self.engines.take_profile().into_iter()
            .map(|(phase, seconds)| (phase.to_string(), json!(seconds))).collect();
        let cpu = self.engines.cpu_routes();
        drop(self.engines.take_draft_stats());
        let mut ttft = None;
        let decoding = Instant::now();
        let decoded = decode(&mut self.engines, &mut self.history, logits, self.vocabulary, &self.end, max_new, sampling, seed, cancelled,
                             &mut |token| {
                                 ttft.get_or_insert_with(|| started.elapsed().as_secs_f64());
                                 emit(token);
                             }, constraint);
        let decoded = match decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                self.reset();
                return Err(error);
            }
        };
        let decode_seconds = decoding.elapsed().as_secs_f64();
        let emitted = decoded.emitted;
        let steps = emitted.saturating_sub(1);
        // (with DS41_PROFILE: the decode's phases, in milliseconds a token)
        let decode_profile: Map<String, Value> = self.engines.take_profile().into_iter()
            .map(|(phase, seconds)| (phase.to_string(), json!(1e3 * seconds / steps.max(1) as f64))).collect();
        let (hits2, host2, drive2) = self.engines.counts();
        let cpu2 = self.engines.cpu_routes();
        let per = |value: u64| if steps > 0 { json!(value as f64 / steps as f64) } else { Value::Null };
        let draft_stats = self.engines.take_draft_stats();
        self.last_metrics = json!({
            "cached_tokens": reused, "resumed_from": resumed_from, "agreed_tokens": agreed,
            "prefill_tokens": total - reused, "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": (total - reused) as f64 / prefill_seconds.max(1e-9),
            "resume_seconds": resume_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            // (store events of every device's share, a prefetched expert counted again when used;
            // the drive bytes are what the wait for the first token read)
            "prefill_experts": {"vram": hits - hits0, "host": host - host0, "drive": drive - drive0},
            "prefill_drive_bytes": prefill_drive_bytes,
            "prefill_profile": prefill_profile,
            "prefill_store": prefill_store,
            "ttft_seconds": ttft, "generated_tokens": emitted, "decode_forward_tokens": decoded.fed, "decode_passes": decoded.passes,
            "draft_tokens": decoded.drafted, "accepted_drafts": decoded.accepted, "decode_seconds": decode_seconds,
            "engine_draft_tokens": decoded.engine_drafted, "accepted_engine_drafts": decoded.engine_accepted,
            "decode_tokens_per_second": if steps > 0 { json!(steps as f64 / decode_seconds) } else { Value::Null },
            "experts_per_token": {"vram": per(hits2 - hits), "host": per(host2 - host), "drive": per(drive2 - drive), "cpu": per(cpu2 - cpu)},
            "decode_profile": decode_profile,
        })
        .as_object()
        .unwrap()
        .clone();
        if let Some((runs, widths)) = draft_stats {
            self.last_metrics.insert("engine_draft_runs".into(), json!(runs));
            self.last_metrics.insert("engine_draft_widths".into(), json!(widths));
        }
        Ok(())
    }

    /// The prompt tokens already held, after restoring the state they need: the held sequence when
    /// the prompt extends it ("live"), else the longest checkpoint the prompt extends that the
    /// held sequence matches as far ("turn", the last, "prior", or "root"; its K/V rows are then
    /// the held ones) or, whole, whatever is held, else none (the engines start over). At least one
    /// prompt token is left to run.
    fn resume(&mut self, tokens: &[u32]) -> Result<(usize, &'static str), String> {
        let limit = tokens.len() - 1;
        let held = &self.history;
        if !held.is_empty() && held.len() <= limit && tokens.starts_with(held) {
            return Ok((held.len(), "live"));
        }
        let span = held.len().min(limit);
        let common = (0..span).find(|&index| held[index] != tokens[index]).unwrap_or(span);
        // (how far the prompt agrees with what is held or checkpointed: where a client changed it)
        self.agreed = self.checkpoints.iter()
            .map(|checkpoint| checkpoint.tokens.iter().zip(tokens).take_while(|(a, b)| a == b).count())
            .chain([common]).max().unwrap_or(0);
        let whole = self.engines.whole_checkpoints();
        let best = self.checkpoints.iter().enumerate()
            .filter(|(_, checkpoint)| (whole || checkpoint.tokens.len() <= common) && checkpoint.tokens.len() <= limit
                    && tokens.starts_with(&checkpoint.tokens) && self.engines.holds(checkpoint.slot))
            .max_by_key(|(_, checkpoint)| checkpoint.tokens.len());
        let Some((index, checkpoint)) = best else {
            self.reset();
            return Ok((0, "none"));
        };
        let kind = if checkpoint.root {
            "root"
        } else if self.checkpoints[index + 1..].iter().all(|later| later.root) {
            "turn"
        } else {
            "prior"
        };
        let (slot, saved) = (checkpoint.slot, checkpoint.tokens.clone());
        if let Err(error) = self.engines.checkpoint(slot, false) {
            self.reset();
            return Err(error);
        }
        self.history = saved;
        Ok((self.history.len(), kind))
    }

    /// Where the prompt's last generation prompt begins, to be checkpointed: past what is held,
    /// within the prompt's last [`TURN_REACH`] tokens.
    fn turn_mark(&self, tokens: &[u32], reused: usize) -> Option<usize> {
        let start = self.turn_start.filter(|_| self.engines.checkpoints() > 0)?;
        let lowest = tokens.len().saturating_sub(TURN_REACH).max(reused + 1);
        (lowest..tokens.len()).rev().find(|&index| tokens[index] == start)
    }

    /// Where the prompt's system prompt ends (its first user message begins), to be checkpointed
    /// as its conversation's root: past what is held, at least [`ROOT_MINIMUM`] tokens in, and only
    /// with a checkpoint left for the turns.
    fn root_mark(&self, tokens: &[u32], reused: usize) -> Option<usize> {
        let end = self.root_end.filter(|_| self.engines.checkpoints() > 1)?;
        let root = tokens.iter().position(|&token| token == end)?;
        (root >= ROOT_MINIMUM && reused < root).then_some(root)
    }

    /// Save the held state as a checkpoint, the root (the last root becoming a turn's) or the last
    /// turn's: one of the same tokens just becomes the last; else a free slot takes it, the oldest
    /// turn's giving its up when all are taken. The engines' budget gives the root's up last.
    fn save_checkpoint(&mut self, root: bool) -> Result<(), String> {
        // (checkpoints the engines gave up are gone: one of the same tokens is taken again)
        let engines = &self.engines;
        self.checkpoints.retain(|checkpoint| engines.holds(checkpoint.slot));
        if root {
            self.checkpoints.iter_mut().for_each(|checkpoint| checkpoint.root = false);
        }
        if let Some(index) = self.checkpoints.iter().position(|checkpoint| checkpoint.tokens == self.history) {
            let mut same = self.checkpoints.remove(index);
            same.root |= root;
            self.checkpoints.push(same);
        } else {
            let slots = self.engines.checkpoints();
            if self.checkpoints.len() >= slots {
                let oldest = self.checkpoints.iter().position(|checkpoint| !checkpoint.root).unwrap_or(0);
                self.checkpoints.remove(oldest);
            }
            let slot = (0..slots).find(|slot| self.checkpoints.iter().all(|checkpoint| checkpoint.slot != *slot))
                .ok_or("no free Straydog checkpoint")?;
            self.engines.checkpoint(slot, true)?;
            self.checkpoints.push(Checkpoint { slot, tokens: self.history.clone(), root });
        }
        let kept = self.checkpoints.iter().find(|checkpoint| checkpoint.root).map(|checkpoint| checkpoint.slot);
        self.engines.keep_checkpoint(kept);
        Ok(())
    }

    /// What a later start warms up with: the experts the engines hold, least recently used first,
    /// in VRAM (`vram`) and only in the host tier (`host`), as [layer, expert] pairs.
    pub fn warm_state(&self) -> Value {
        let (vram, host) = self.engines.contents();
        json!({"vram": vram, "host": host})
    }

    /// Warm up with an earlier run's [`warm_state`](Self::warm_state), step by step
    /// ([`warm_step`](Self::warm_step)) while no request runs. Pairs outside the model are dropped,
    /// and those the slots as placed do not hold. The experts to bring into VRAM and the host tier.
    pub fn warm(&mut self, state: &Value) -> Result<(usize, usize), String> {
        let (layers, experts) = (self.layers, self.experts);
        let pairs = |key: &str| -> Result<Pending, String> {
            let list = state.get(key).and_then(Value::as_array).ok_or("a Straydog warm-up state without its lists")?;
            Ok(list.iter()
                .filter_map(|pair| Some((pair.get(0)?.as_u64()? as usize, pair.get(1)?.as_u64()? as usize)))
                .filter(|&(layer, expert)| layer < layers && expert < experts)
                .collect())
        };
        let (vram, host) = self.engines.fit_warm(pairs("vram")?.into(), pairs("host")?.into());
        self.warming = (vram.into(), host.into());
        Ok((self.warming.0.len(), self.warming.1.len()))
    }

    /// Take the warm-up's reads that landed and, while fewer than `in_flight` are on their way
    /// (each device's read of an expert counts), start up to `count` more (VRAM's first, the
    /// experts a request is likeliest to use, then the host tier's; each least recently used
    /// first, so the most recent end up the most recent), without waiting: a request that comes
    /// meanwhile finds the drive and the bus nearly free. Whether the warm-up is over.
    pub fn warm_step(&mut self, count: usize, in_flight: usize) -> Result<bool, String> {
        let reading = self.engines.poll()?;
        let left = self.warming.0.len() + self.warming.1.len();
        if left == 0 || reading >= in_flight {
            return Ok(left == 0 && reading == 0);
        }
        let count = count.min((in_flight - reading).div_ceil(self.engines.devices_per_expert()));
        let vram: Vec<(usize, usize)> = self.warming.0.iter().take(count).copied().collect();
        let host: Vec<(usize, usize)> = self.warming.1.iter().take(count - vram.len()).copied().collect();
        // (those the stores have no room for yet stay first in line)
        let (reads, placed, hosted) = self.engines.warm(&vram, &host)?;
        self.warming.0.drain(..placed);
        self.warming.1.drain(..hosted);
        // No room for a long while (the stores full of what requests brought in) and nothing on
        // its way: done. (Room comes as the host tier is pinned, and as reads land.)
        self.warm_stalled = if placed + hosted == 0 && reads == 0 { self.warm_stalled + 1 } else { 0 };
        if self.warm_stalled > WARM_PATIENCE {
            self.warming = Default::default();
        }
        Ok(false)
    }

    /// Wait for the expert stores' reads and copies (before the process ends).
    pub fn settle(&mut self) -> Result<(), String> {
        self.engines.settle()
    }

    /// Between requests: a step of copying the checkpoints taken, then of making the stores' room
    /// for the next prefill (without waiting); whether that is done.
    pub fn make_room(&mut self) -> Result<bool, String> {
        self.engines.make_room()
    }

    /// After the first passes: lend the expert stores the VRAM the cards still have spare
    /// ([`Engines::lend_spare`]); the placement then counts the slots they have now.
    pub fn lend_spare(&mut self) -> Result<(), String> {
        let slots = self.engines.lend_spare()?;
        if !slots.is_empty() {
            self.placement["expert_slots"] = json!(slots);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy model over a small vocabulary: after a sequence it favors the token that followed the
    /// last token's previous occurrence (else a fixed successor), with smaller logits for the rest
    /// that depend on the whole sequence. It verifies up to `rows` tokens a pass.
    #[derive(Default)]
    struct Toy {
        vocabulary: usize,
        rows: usize,
        held: Vec<u32>,
        verified_from: Option<usize>,
        proposals: std::collections::VecDeque<Result<Option<Vec<u32>>, String>>,
        draft_calls: Vec<(u32, usize)>,
        verified: Vec<Vec<u32>>,
        rewinds: Vec<usize>,
    }

    impl Toy {
        fn logits(&self) -> Vec<f32> {
            let seq = &self.held;
            let last = *seq.last().unwrap();
            let favored = seq[..seq.len() - 1].iter().rposition(|&token| token == last)
                .map_or((last * 3 + 1) % self.vocabulary as u32, |at| seq[at + 1]);
            let mut hash = seq.iter().fold(0x9e37_79b9_7f4a_7c15u64, |hash, &token| (hash ^ u64::from(token)).wrapping_mul(0x100_0000_01b3));
            (0..self.vocabulary as u32).map(|token| {
                hash = hash.rotate_left(17).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                (hash >> 40) as f32 / (1u64 << 24) as f32 + if token == favored { 2.0 } else { 0.0 }
            }).collect()
        }
    }

    impl Engines for Toy {
        fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
            self.held.extend_from_slice(tokens);
            self.verified_from = None;
            Ok(self.logits())
        }
        fn reset(&mut self) {
            self.held.clear();
        }
        fn counts(&self) -> (u64, u64, u64) {
            (0, 0, 0)
        }
        fn contents(&self) -> Contents {
            (Vec::new(), Vec::new())
        }
        fn warm(&mut self, vram: &[(usize, usize)], host: &[(usize, usize)]) -> Result<(usize, usize, usize), String> {
            Ok((0, vram.len(), host.len()))
        }
        fn poll(&mut self) -> Result<usize, String> {
            Ok(0)
        }
        fn devices_per_expert(&self) -> usize {
            1
        }
        fn gpus(&self) -> Vec<&Backend> {
            Vec::new()
        }
        fn settle(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn verify_rows(&self) -> usize {
            self.rows
        }
        fn draft(&mut self, anchor: u32, max: usize) -> Result<Option<Vec<u32>>, String> {
            self.draft_calls.push((anchor, max));
            self.proposals.pop_front().unwrap_or(Ok(None))
        }
        fn verify(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
            assert!(tokens.len() <= self.rows);
            self.verified.push(tokens.to_vec());
            self.verified_from = Some(self.held.len());
            let mut rows = Vec::new();
            for &token in tokens {
                self.held.push(token);
                rows.extend(self.logits());
            }
            Ok(rows)
        }
        fn rewind(&mut self, keep: usize) -> Result<(), String> {
            self.rewinds.push(keep);
            let from = self.verified_from.take().ok_or("no verify to rewind")?;
            self.held.truncate(from + keep);
            Ok(())
        }
    }

    fn run(rows: usize, prompt: &[u32], end: &[u32], max_new: usize, sampling: &Sampling, seed: u64) -> (Vec<u32>, Decoded) {
        let mut toy = Toy { vocabulary: 8, rows, ..Toy::default() };
        let mut history = prompt.to_vec();
        let logits = toy.forward(prompt).unwrap();
        let mut out = Vec::new();
        let decoded = decode(&mut toy, &mut history, logits, 8, end, max_new, sampling, seed, &|| false, &mut |token| out.push(token), None).unwrap();
        // the engines hold what the session thinks they hold
        assert_eq!(toy.held, history);
        assert_eq!(history.len(), prompt.len() + decoded.emitted - usize::from(decoded.emitted == max_new), "{decoded:?}");
        (out, decoded)
    }

    const PROMPT: [u32; 9] = [1, 2, 3, 4, 5, 1, 2, 3, 6];

    /// Every token a constrained reply takes keeps it a JSON object's prefix (whole once it
    /// ends), with no drafts though the engines verify them.
    #[test]
    fn a_constrained_reply_holds_to_its_json_and_takes_no_drafts() {
        use std::sync::Arc;

        use crate::constraint::{Alternative, JsonMatcher, Schema, TokenBytes};
        let pieces = ["<e>", "{", "}", "x", " ", "\"", "a", ":"];
        let tokens = Arc::new(TokenBytes::new(pieces.iter().map(|piece| piece.as_bytes().to_vec()).collect(), vec![false; 8]));
        for seed in 1..20 {
            let mut constraint = Constraint::new(Arc::clone(&tokens), &[0], false, vec![Alternative::json(Schema::any_object())]);
            let mut toy = Toy { vocabulary: 8, rows: 3, ..Toy::default() };
            let mut history = PROMPT.to_vec();
            let logits = toy.forward(&PROMPT).unwrap();
            let mut out = Vec::new();
            let sampling = Sampling { temperature: 1.0, ..Sampling::recommended(false) };
            let done = decode(&mut toy, &mut history, logits, 8, &[0], 120, &sampling, seed, &|| false,
                              &mut |token| out.push(token), Some(&mut constraint)).unwrap();
            assert_eq!(done.drafted, 0);
            let text: String = out.iter().map(|&token| pieces[token as usize]).collect();
            let mut json = JsonMatcher::new(Arc::new(Schema::any_object()));
            assert!(json.feed_all(text.as_bytes()), "seed {seed}: {text}");
            assert!(done.emitted == 120 || json.complete(), "seed {seed}: {text}");
        }
    }

    fn scripted(rows: usize, prompt: &[u32], end: &[u32], max_new: usize,
                proposals: Vec<Result<Option<Vec<u32>>, String>>) -> (Toy, Vec<u32>, Decoded) {
        let mut toy = Toy { vocabulary: 8, rows, proposals: proposals.into(), ..Toy::default() };
        let mut history = prompt.to_vec();
        let logits = toy.forward(prompt).unwrap();
        let mut out = Vec::new();
        let done = decode(&mut toy, &mut history, logits, 8, end, max_new, &Sampling::GREEDY, 1,
            &|| false, &mut |token| out.push(token), None).unwrap();
        assert_eq!(toy.held, history);
        assert_eq!(history.len(), prompt.len() + done.emitted - usize::from(done.emitted == max_new));
        (toy, out, done)
    }

    #[test]
    fn engine_drafts_use_existing_acceptance_rewind_and_both_width_limits() {
        let (plain, _) = run(1, &[1], &[], 8, &Sampling::GREEDY, 1);
        assert_eq!(plain, [4, 5, 0, 1, 4, 5, 0, 1]);
        let wrong = (plain[2] + 1) % 8;
        let proposals = vec![
            Ok(Some(vec![plain[1], wrong, 7, u32::MAX])), // Too wide: cap at rows-1.
            Ok(Some(plain[3..6].to_vec())),
            Ok(Some(vec![plain[7], 99, 99])), // Only one output remains.
        ];
        let (toy, out, done) = scripted(4, &[1], &[], 8, proposals);
        assert_eq!(out, plain);
        assert_eq!(toy.draft_calls, [(plain[0], 3), (plain[2], 3), (plain[6], 3)]);
        assert_eq!(toy.verified, [vec![plain[0], plain[1], wrong, 7], plain[2..6].to_vec(), plain[6..8].to_vec()]);
        assert_eq!(toy.rewinds, [2, 4, 1]); // Rejection, all accepted, final emitted token stays unfed.
        assert_eq!(done, Decoded { emitted: 8, fed: 10, passes: 3, drafted: 7, accepted: 4,
            engine_drafted: 7, engine_accepted: 4 });
    }

    #[test]
    fn none_keeps_prompt_lookup_and_empty_explicitly_declines_it() {
        let (plain, _) = run(1, &PROMPT, &[], 80, &Sampling::GREEDY, 1);
        let (_, lookup_done) = run(4, &PROMPT, &[], 80, &Sampling::GREEDY, 1);
        assert!(lookup_done.drafted > 0 && lookup_done.accepted > 0);
        let (fallback, out, done) = scripted(4, &PROMPT, &[], 80, vec![Ok(None)]);
        assert_eq!(out, plain);
        assert_eq!(done, lookup_done);
        assert!(!fallback.verified.is_empty() && !fallback.draft_calls.is_empty());
        assert_eq!((done.engine_drafted, done.engine_accepted), (0, 0));
        // (eight tokens: the toy's reply repeats two tokens and its matches stay below a long one)
        let (plain, plain_done) = run(1, &PROMPT, &[], 8, &Sampling::GREEDY, 1);
        let (empty, out, done) = scripted(4, &PROMPT, &[], 8, vec![Ok(Some(Vec::new())); 8]);
        assert_eq!(out, plain);
        assert_eq!(done, plain_done);
        assert_eq!(empty.draft_calls.len(), 7);
        assert!(empty.verified.is_empty() && empty.rewinds.is_empty());
    }

    /// A lookup match of [`LONG_MATCH`] tokens is verified without asking the engine for a draft,
    /// whatever it would have said; shorter matches leave the draft to the engine.
    #[test]
    fn a_long_lookup_match_is_verified_without_asking_the_engine() {
        let (plain, _) = run(1, &PROMPT, &[], 80, &Sampling::GREEDY, 1);
        // (the toy repeats 3, 6: the match reaches 8 tokens with the eighth reply token)
        let (toy, out, done) = scripted(4, &PROMPT, &[], 80, vec![Ok(Some(Vec::new())); 80]);
        assert_eq!(out, plain);
        assert_eq!(toy.draft_calls.len(), 7);
        assert!(!toy.verified.is_empty() && done.drafted > 0 && done.accepted > 0);
        assert_eq!((done.engine_drafted, done.engine_accepted), (0, 0));
        // an engine that has drafts for every round is heard only while the matches are short
        let (toy, out, done) = scripted(4, &PROMPT, &[], 80, vec![Ok(Some(vec![3])); 80]);
        assert_eq!(out, plain);
        assert!(toy.draft_calls.len() < done.passes && done.engine_drafted >= 1 && done.drafted > done.engine_drafted);
    }

    #[test]
    fn an_engine_can_return_to_prompt_lookup_on_the_next_round() {
        let (plain, _) = run(1, &PROMPT, &[], 80, &Sampling::GREEDY, 1);
        let (toy, out, done) = scripted(4, &PROMPT, &[], 80, vec![Ok(Some(vec![plain[1]])), Ok(None)]);
        assert_eq!(out, plain);
        assert_eq!(toy.verified[0], plain[..2]);
        assert_eq!((done.engine_drafted, done.engine_accepted), (1, 1));
        assert!(done.drafted > done.engine_drafted && done.accepted > done.engine_accepted);
    }

    #[test]
    fn engine_drafts_stop_at_an_end_token_and_are_not_asked_without_verify_rows() {
        let (toy, out, done) = scripted(4, &[1], &[0], 8, vec![Ok(Some(vec![5, 0, 1]))]);
        assert_eq!(out, [4, 5]);
        assert_eq!(toy.rewinds, [2]);
        assert_eq!((done.engine_drafted, done.engine_accepted), (3, 1));
        let (plain, plain_done) = run(1, &PROMPT, &[], 20, &Sampling::GREEDY, 1);
        let (toy, out, done) = scripted(1, &PROMPT, &[], 20, vec![Err("must not ask".into())]);
        assert_eq!(out, plain);
        assert_eq!(done, plain_done);
        assert!(toy.draft_calls.is_empty() && toy.verified.is_empty());
        assert_eq!(toy.proposals.len(), 1);
    }

    #[test]
    fn engine_draft_failure_or_invalid_token_does_not_enter_verification() {
        for (proposal, expected) in [(Err("draft failed".into()), "draft failed"),
            (Ok(Some(vec![8])), "Straydog engine drafted a token out of range")] {
            let mut toy = Toy { vocabulary: 8, rows: 4, proposals: [proposal].into(), ..Toy::default() };
            let mut history = vec![1];
            let logits = toy.forward(&history).unwrap();
            let error = decode(&mut toy, &mut history, logits, 8, &[], 2, &Sampling::GREEDY, 1,
                &|| false, &mut |_| {}, None).unwrap_err();
            assert_eq!(error, expected);
            assert_eq!(toy.held, [1]);
            assert_eq!(history, [1]);
            assert!(toy.verified.is_empty() && toy.rewinds.is_empty());
        }
    }

    #[test]
    fn greedy_drafts_change_nothing_but_the_passes() {
        let greedy = Sampling { temperature: 0.0, ..Sampling::recommended(false) };
        let (plain, plain_done) = run(1, &PROMPT, &[], 300, &greedy, 1);
        assert_eq!((plain_done.drafted, plain_done.passes), (0, 299));
        for rows in [2, 3, 4] {
            let (drafted, done) = run(rows, &PROMPT, &[], 300, &greedy, 1);
            assert_eq!(drafted, plain, "{rows} rows");
            assert!(done.accepted > 0 && done.passes < plain_done.passes, "{rows} rows: {done:?}");
        }
    }

    #[test]
    fn an_end_token_or_the_limit_ends_a_reply_as_without_drafts() {
        let greedy = Sampling { temperature: 0.0, ..Sampling::recommended(false) };
        for end in [vec![5], vec![6], vec![0]] {
            for max_new in [1, 2, 7, 40] {
                let (plain, _) = run(1, &PROMPT, &end, max_new, &greedy, 3);
                let (drafted, _) = run(4, &PROMPT, &end, max_new, &greedy, 3);
                assert_eq!(drafted, plain, "end {end:?}, at most {max_new}");
            }
        }
    }

    #[test]
    fn sampled_drafts_keep_the_target_distribution() {
        let sampling = Sampling { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, presence_penalty: 0.0 };
        let trials = 6000;
        let mut counts = [[[0usize; 8]; 6]; 2];
        let mut accepted = 0;
        for (side, rows) in [1, 4].into_iter().enumerate() {
            for seed in 0..trials {
                let (out, done) = run(rows, &PROMPT, &[], 6, &sampling, seed as u64 * 7919 + 13);
                accepted += done.accepted;
                for (position, &token) in out.iter().enumerate() {
                    counts[side][position][token as usize] += 1;
                }
            }
        }
        // (the favored token's probability is about a half, so about every other draft is taken)
        assert!(accepted > trials / 10, "drafts were accepted ({accepted})");
        // every position's token frequencies agree within sampling noise
        for (position, (plain, drafted)) in counts[0].iter().zip(&counts[1]).enumerate() {
            for (token, (&a, &b)) in plain.iter().zip(drafted).enumerate() {
                let (a, b) = (a as f64 / trials as f64, b as f64 / trials as f64);
                assert!((a - b).abs() < 0.03, "position {position}, token {token}: {a:.3} vs {b:.3}");
            }
        }
    }
}
