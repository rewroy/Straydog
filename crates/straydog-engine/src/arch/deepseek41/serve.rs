//! DeepSeek-V4.1 served (docs/DESIGN.md §10): the authenticated shards ([`Pinned`]), placed on the
//! GPUs either with every layer split across them (one engine, experts divided by rows) or with
//! the layers split where no layer reads another part's compressed rows (one engine per GPU);
//! each GPU's VRAM left after its dense weights and state given to expert slots and the host
//! memory planned for the experts' host tier, then prompts continued from the sequence the engines
//! hold or from a checkpoint (the whole sequence state, in host memory, where a reply's generation
//! prompt began: a conversation switched away from comes back from there), sampling and the
//! request's metrics. The engines are driven only through their public passes.

use std::path::Path;
use std::time::Instant;

use serde_json::json;

use super::drafter::{self, Drafter, Policy, Rings, MAX_SLOTS};
use super::dspark::Dspark;
use super::engine::{divided, forward_split, pass_row_bytes, verify_split, Engine, Settings as EngineSettings, Snapshot, ATTENTION_ROWS, EXPANSIONS,
                    OWN_ROWS, TAP_LAYERS, TAP_ROWS};
use super::Hparams;
use crate::arch::served::{self, Draw, Placed, Proposal, Session, Settings};
use crate::bundle;
use crate::experts::Contents;
use crate::gpu::Backend;
use crate::group::SplitMode;
use crate::pinned::Pinned;

use served::{RESERVE, SCRATCH};


/// DeepSeek-V4.1 served: its placed engines in a session.
pub type Model = Session<Engines>;

/// Checkpoints kept, each the whole sequence state where a reply's generation prompt began (a
/// conversation switched away from comes back from there), one where the last long system prompt
/// ended (a new conversation with it starts from there).
pub const CHECKPOINTS: usize = 4;
/// Host memory the checkpoints take at most, together and at any moment (a 20K-token
/// conversation's ~140 MB): the oldest give theirs up before a new one is taken, and one alone
/// over it is not taken. Enough for one at the full 262144-token context (~1.7 GiB): at 1 GiB a
/// conversation past ~160K tokens kept none, and each new turn there (the template drops the last
/// reply's reasoning) read the conversation again from an older checkpoint, 43K tokens in 48 s.
/// The expert host tier gives up what this takes.
pub const CHECKPOINT_BYTES: u64 = 2 << 30;
/// VRAM left free on each card, as the runtime reports it, when the spare is lent to the expert
/// stores ([`Engines::lend_spare`], after the first passes). The least margin an automatic budget
/// leaves is 512 MiB; the driver counts 204 MiB of a card's memory used that the runtime's free
/// does not show (both R9700s, 31.9 GiB each); and everything a session can first use (a 144K-token
/// prompt, an image, a JSON schema, a required tool call, thinking, sampling, every verify width)
/// took a further 25 MiB of the card's used memory (46 MiB of the runtime's free) after the first
/// passes. 768 MiB keeps 564 MiB of the card free past all of that; the budget caps what is lent.
const SPARE_KEEP: u64 = 768 << 20;
/// Write-backs a device queues a step of making room between requests ([`Engines::make_room`]):
/// ~20 MB, about what its downloads move in the worker's 1 ms between steps.
const ROOM_STEP: usize = 4;
/// Floats of the checkpoints' compressed rows and index keys a step between requests copies
/// ([`Engines::make_room`]): 32 MB, ~6 ms (~0.15 s when the host has to find the pages).
const CHECKPOINT_STEP: usize = 8 << 20;

/// The engines of consecutive layer ranges (one, or one per device).
pub struct Engines {
    engines: Vec<Engine<'static, Pinned>>,
    /// The devices of the first engine (each reads its rows of an expert).
    devices_per_expert: usize,
    /// The most tokens a verify pass takes (DS41_DRAFTS=0: 1, no drafts).
    verify_rows: usize,
    /// Per checkpoint slot: each engine's snapshot, if held.
    saved: Vec<Option<Vec<Snapshot>>>,
    /// The slots holding one, oldest first.
    order: Vec<usize>,
    /// The slot given up last ([`served::Engines::keep_checkpoint`]).
    kept: Option<usize>,
    /// Host bytes they may take ([`CHECKPOINT_BYTES`]).
    budget: u64,
    /// Floats a step between requests copies of their pending rows ([`CHECKPOINT_STEP`]).
    step: usize,
    /// The model's files (the drafter reads the token embeddings from them).
    files: &'static Pinned,
    /// DSpark, when the model's drafter is pinned and there.
    drafting: Option<Drafting>,
}

/// DSpark beside the engine that runs the layers it reads ([`TAP_LAYERS`]): the drafter, the policy
/// that picks how many of a draft's tokens a round verifies, and what keeps both true to the
/// sequence the engines hold. The drafter's rings hold the target's features of the last
/// [`drafter::RING`] positions: every pass puts its rows in (what its tap kept) and a rewind only
/// the rows it keeps, a reset forgets them and a checkpoint keeps them with the sequence state, so
/// a draft never reads the features of a sequence that is not the held one.
struct Drafting {
    drafter: Drafter,
    policy: Policy,
    /// The engine it sits beside.
    engine: usize,
    /// The routed experts it keeps in each stage.
    resident: [usize; 3],
    /// Whether it drafts and keeps its features (off only in tests: [`Engines::draft_with_dspark`]).
    on: bool,
    /// Whether the pass to come is a decode pass (a draft was asked for before it): the policy learns
    /// its time.
    deciding: bool,
    /// Whether the verify pass to come checks DSpark's draft: the policy learns how much of it was kept.
    drafted: bool,
    /// Tests only: verify this many of every draft, whatever the policy (which times the passes) would.
    fixed_width: Option<usize>,
    /// Per checkpoint slot, the rings of the sequence it holds.
    rings: Vec<Option<Rings>>,
    /// The drafter's file, kept mapped.
    _files: Pinned,
}

impl Drafting {
    /// A new sequence: no history, no pause.
    fn forget(&mut self) {
        self.drafter.forget_from(0);
        self.policy.reset();
        (self.deciding, self.drafted) = (false, false);
    }
}

impl served::Engines for Engines {
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let started = Instant::now();
        let logits = forward_split(&mut self.engines, tokens)?;
        self.passed(tokens.len(), started);
        self.inject(tokens.len())?;
        Ok(logits)
    }

    fn reset(&mut self) {
        // (a checkpoint whose copy fails is given up)
        let _ = self.finish_checkpoints(usize::MAX);
        self.engines.iter_mut().for_each(Engine::reset);
        if let Some(drafting) = self.drafting.as_mut() {
            drafting.forget();
        }
    }

    fn counts(&self) -> (u64, u64, u64) {
        self.engines.iter().map(Engine::cache_counts)
            .fold((0, 0, 0), |sum, counts| (sum.0 + counts.hits, sum.1 + counts.host_hits, sum.2 + counts.drive_reads))
    }

    fn drive_bytes(&self) -> u64 {
        self.engines.iter().map(|engine| engine.cache_counts().drive_bytes).sum()
    }

    fn store_counts(&self) -> Option<crate::experts::Counts> {
        self.engines.iter().map(Engine::cache_counts).reduce(|sum, counts| sum + counts)
    }

    fn take_draft_stats(&mut self) -> Option<(usize, Vec<usize>)> {
        let stats = self.drafting.as_mut()?.policy.take_stats();
        Some((stats.runs, stats.widths.to_vec()))
    }

    fn take_profile(&mut self) -> Vec<(&'static str, f64)> {
        // (each engine is a layer group; their phases add up over the pass)
        let mut phases: std::collections::BTreeMap<&'static str, f64> = std::collections::BTreeMap::new();
        for engine in &mut self.engines {
            for (phase, seconds) in engine.take_profile() {
                *phases.entry(phase).or_default() += seconds;
            }
        }
        phases.into_iter().collect()
    }

    fn contents(&self) -> Contents {
        let (mut vram, mut host) = (Vec::new(), Vec::new());
        for engine in &self.engines {
            let (in_vram, in_host) = engine.expert_contents();
            vram.extend(in_vram);
            host.extend(in_host);
        }
        (vram, host)
    }

    fn warm(&mut self, vram: &[(usize, usize)], host: &[(usize, usize)]) -> Result<(usize, usize, usize), String> {
        let (mut reading, mut placed, mut hosted) = (0, vram.len(), host.len());
        for engine in &mut self.engines {
            let (read, engine_placed, engine_hosted) = engine.warm_experts(vram, host)?;
            (reading, placed, hosted) = (reading + read, placed.min(engine_placed), hosted.min(engine_hosted));
        }
        Ok((reading, placed, hosted))
    }

    fn fit_warm(&self, vram: Vec<(usize, usize)>, host: Vec<(usize, usize)>) -> crate::experts::Contents {
        // each engine its layers' (the stores' layers as placed now, whatever they were)
        let (mut kept_vram, mut kept_host) = (std::collections::HashSet::new(), std::collections::HashSet::new());
        for engine in &self.engines {
            let (vram, host) = engine.fit_warm(&vram, &host);
            kept_vram.extend(vram);
            kept_host.extend(host);
        }
        let host_out: Vec<(usize, usize)> = host.iter().chain(vram.iter().filter(|&key| !kept_vram.contains(key)))
            .filter(|&key| kept_host.contains(key)).copied().collect();
        (vram.into_iter().filter(|key| kept_vram.contains(key)).collect(), host_out)
    }

    fn poll(&mut self) -> Result<usize, String> {
        let mut reading = 0;
        for engine in &mut self.engines {
            reading += engine.poll_experts()?;
        }
        Ok(reading)
    }

    fn devices_per_expert(&self) -> usize {
        self.devices_per_expert
    }

    fn vision(&self) -> bool {
        self.engines[0].vision()
    }

    fn encode_images(&mut self, images: &[crate::vision::Patches]) -> Result<Vec<Vec<f32>>, String> {
        self.engines[0].encode_images(images)
    }

    fn set_images(&mut self, spans: Vec<(usize, std::sync::Arc<Vec<f32>>)>) {
        // (the first engine embeds the tokens; the others see image positions by their ids)
        self.engines[0].set_images(spans);
    }

    fn gpus(&self) -> Vec<&Backend> {
        self.engines.iter().flat_map(Engine::gpus).collect()
    }

    fn settle(&mut self) -> Result<(), String> {
        self.engines.iter_mut().try_for_each(Engine::settle)
    }

    /// Each engine's devices lend their stores what their cards have free beyond [`SPARE_KEEP`]
    /// (DS41_LEND_SPARE=0: nothing); a failure to is reported, and the placement stands.
    fn lend_spare(&mut self) -> Result<Vec<usize>, String> {
        if off("DS41_LEND_SPARE") {
            return Ok(Vec::new());
        }
        for engine in &mut self.engines {
            let before = engine.expert_capacity().0;
            match engine.lend_spare(SPARE_KEEP) {
                Ok((bytes, slots)) => eprintln!("Straydog spare VRAM: {:.0} MiB a GPU lent as {slots} expert slots ({before} before)",
                                                bytes as f64 / (1u64 << 20) as f64),
                Err(error) => eprintln!("Straydog could not lend its spare VRAM: {error}"),
            }
        }
        Ok(self.engines.iter().map(|engine| engine.expert_capacity().0).collect())
    }

    /// The checkpoints' copies a step at a time, then each store's room.
    fn make_room(&mut self) -> Result<bool, String> {
        if !self.finish_checkpoints(self.step)? {
            return Ok(false);
        }
        let mut done = true;
        for engine in &mut self.engines {
            done &= engine.make_room(ROOM_STEP)?;
        }
        Ok(done)
    }

    fn prefetch_ahead(&mut self, tokens: &[u32]) -> Result<(), String> {
        self.engines.iter().try_for_each(|engine| engine.prefetch_ahead(tokens))
    }

    fn verify_rows(&self) -> usize {
        self.verify_rows
    }

    /// DSpark's draft of up to `max` tokens after `anchor`, as many as its policy finds worth a
    /// verify pass (by the drafter's confidences and what the passes of this run cost); None when
    /// there is no drafter, it is paused, finds none worth it, or lacks the history (the lookup
    /// of the sequence's own text then drafts).
    fn draft(&mut self, anchor: u32, max: usize, draw: Option<Draw>) -> Result<Option<Proposal>, String> {
        let Some(drafting) = self.drafting.as_mut().filter(|drafting| drafting.on) else { return Ok(None) };
        drafting.deciding = true;
        let engine = &mut self.engines[drafting.engine];
        let position = engine.position();
        let slots = max.min(MAX_SLOTS - 1);
        if slots == 0 || (drafting.fixed_width.is_none() && !drafting.policy.drafting()) || !drafting.drafter.has_history(position) {
            return Ok(None);
        }
        let started = Instant::now();
        let draft = drafting.drafter.draft_drawn(&mut engine.beside()?, self.files, anchor, position, slots, draw)?;
        drafting.policy.drafted(1e3 * started.elapsed().as_secs_f64());
        let width = match drafting.fixed_width {
            Some(width) => width.min(slots),
            None => drafting.policy.width(&draft.confidences, slots),
        }
        .min(draft.tokens.len());
        drafting.drafted = width > 0;
        if width == 0 {
            return Ok(None);
        }
        // (the logits the verified tokens were drawn from, read back for them alone)
        let drawn = if draw.is_some() { drafting.drafter.drawn_rows(&mut engine.beside()?, width)? } else { Vec::new() };
        Ok(Some(Proposal { tokens: draft.tokens[..width].to_vec(), drawn }))
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let started = Instant::now();
        let logits = verify_split(&mut self.engines, tokens)?;
        self.passed(tokens.len(), started);
        Ok(logits)
    }

    /// Take back a verify pass but its first `keep` rows; DSpark's history gets those rows.
    fn rewind(&mut self, keep: usize) -> Result<(), String> {
        self.engines.iter_mut().try_for_each(|engine| engine.rewind(keep))?;
        if let Some(drafting) = self.drafting.as_mut().filter(|drafting| drafting.on) {
            if std::mem::take(&mut drafting.drafted) {
                drafting.policy.verified(keep.saturating_sub(1));
            }
        }
        self.inject(keep)
    }

    fn checkpoints(&self) -> usize {
        CHECKPOINTS
    }

    /// A snapshot of every engine's state into host memory, or every engine's back from it.
    fn checkpoint(&mut self, slot: usize, save: bool) -> Result<(), String> {
        if slot >= CHECKPOINTS {
            return Err("no such Straydog checkpoint".into());
        }
        if !save {
            // (the restore takes the devices' buffers from under the copies still to be made)
            self.finish_checkpoints(usize::MAX)?;
            let snapshots = self.saved[slot].as_ref().ok_or("the Straydog checkpoint was given up")?;
            self.engines.iter_mut().zip(snapshots).try_for_each(|(engine, snapshot)| engine.restore(snapshot))?;
            // (the history of the sequence restored, else none: it refills as the sequence goes on)
            if let Some(drafting) = self.drafting.as_mut() {
                drafting.forget();
                if let (true, Some(rings)) = (drafting.on, drafting.rings[slot].as_ref()) {
                    drafting.drafter.restore_rings(&mut self.engines[drafting.engine].beside()?, rings)?;
                }
            }
            return Ok(());
        }
        // (the slot's snapshots, or one given up for room, are taken again in place: Engine::snapshot_into)
        let mut reuse = self.saved[slot].take();
        self.order.retain(|&held| held != slot);
        // Room first, the copy after: the budget bounds the host memory at its peak too. The kept
        // slot (the conversation root's) gives its up last.
        let required: u64 = self.engines.iter().map(Engine::snapshot_bytes).sum();
        let kept = self.kept.filter(|&kept| kept != slot);
        let held: Vec<(usize, u64)> = self.order.iter().filter(|&&held| Some(held) != kept).chain(kept.iter().filter(|kept| self.order.contains(kept)))
            .map(|&held| (held, self.saved[held].iter().flatten().map(Snapshot::bytes).sum())).collect();
        let Some(given_up) = evictions(&held, required, self.budget) else { return Ok(()) };
        for oldest in given_up {
            let freed = self.saved[oldest].take();
            reuse = reuse.or(freed);
            self.order.retain(|&held| held != oldest);
        }
        let mut snapshots = reuse.unwrap_or_default();
        snapshots.resize_with(self.engines.len(), Snapshot::default);
        for (engine, snapshot) in self.engines.iter_mut().zip(&mut snapshots) {
            engine.snapshot_into(snapshot)?;
        }
        if let Some(drafting) = self.drafting.as_mut() {
            drafting.rings[slot] = None;
            if drafting.on {
                drafting.rings[slot] = Some(drafting.drafter.save_rings(&mut self.engines[drafting.engine].beside()?)?);
            }
        }
        self.saved[slot] = Some(snapshots);
        self.order.push(slot);
        Ok(())
    }

    fn keep_checkpoint(&mut self, slot: Option<usize>) {
        self.kept = slot;
    }

    fn whole_checkpoints(&self) -> bool {
        true
    }

    fn holds(&self, slot: usize) -> bool {
        self.saved.get(slot).is_some_and(Option::is_some)
    }
}

/// The checkpoints (slots with their bytes, oldest first) that give theirs up so one of `required`
/// bytes fits within `cap` with the rest; None when it alone does not (it is not kept).
fn evictions(held: &[(usize, u64)], required: u64, cap: u64) -> Option<Vec<usize>> {
    if required > cap {
        return None;
    }
    let mut total = required + held.iter().map(|(_, bytes)| bytes).sum::<u64>();
    let mut given_up = Vec::new();
    for &(slot, bytes) in held {
        if total <= cap {
            break;
        }
        total -= bytes;
        given_up.push(slot);
    }
    Some(given_up)
}

impl Engines {
    /// A pass of `rows` rows that began at `started`: if it follows a draft (a decode pass), the
    /// width policy learns its time.
    fn passed(&mut self, rows: usize, started: Instant) {
        if let Some(drafting) = self.drafting.as_mut().filter(|drafting| drafting.on) {
            if std::mem::take(&mut drafting.deciding) {
                drafting.policy.passed(rows, 1e3 * started.elapsed().as_secs_f64());
            }
        }
    }

    /// DSpark's history after a pass (or a rewind of one): the first `kept` rows of what the last
    /// pass's tap kept, at the positions up to the engine's.
    fn inject(&mut self, kept: usize) -> Result<(), String> {
        let Some(drafting) = self.drafting.as_mut().filter(|drafting| drafting.on) else { return Ok(()) };
        let engine = &mut self.engines[drafting.engine];
        let end = engine.position();
        let kept = kept.min(TAP_ROWS).min(end);
        let positions: Vec<i32> = (end - kept..end).map(|at| at as i32).collect();
        drafting.drafter.inject(&mut engine.beside()?, &positions)
    }

    /// Tests only: draft with DSpark (when it is loaded), or not: the same engines then decode
    /// without drafts of its. Whether it is loaded.
    #[doc(hidden)]
    pub fn draft_with_dspark(&mut self, on: bool) -> bool {
        let Some(drafting) = self.drafting.as_mut() else { return false };
        // (its history is kept only while it is on: turned on, it has none)
        if drafting.on != on {
            drafting.forget();
            drafting.on = on;
        }
        true
    }

    /// Tests only: verify `width` tokens of every DSpark draft (None: as its policy, which times the passes, picks). A sampled
    /// reply is then the same one twice for a seed; with the policy's widths it is not, as they follow the passes' timing.
    #[doc(hidden)]
    pub fn draft_width(&mut self, width: Option<usize>) {
        if let Some(drafting) = self.drafting.as_mut() {
            drafting.fixed_width = width;
        }
    }

    /// Tests only: whether a draft at the position the engines hold would find DSpark's history in its
    /// rings (None: no drafter).
    #[doc(hidden)]
    pub fn dspark_history(&self) -> Option<bool> {
        let drafting = self.drafting.as_ref()?;
        Some(drafting.drafter.has_history(self.engines[drafting.engine].position()))
    }

    /// Tests only: DSpark's draft of `slots` tokens after `anchor` at the held position and its
    /// confidences, whatever its policy (None: no drafter).
    #[doc(hidden)]
    pub fn dspark_probe(&mut self, anchor: u32, slots: usize) -> Result<Option<drafter::Draft>, String> {
        let Some(drafting) = self.drafting.as_mut() else { return Ok(None) };
        let engine = &mut self.engines[drafting.engine];
        let position = engine.position();
        Ok(Some(drafting.drafter.draft(&mut engine.beside()?, self.files, anchor, position, slots)?))
    }

    /// Copy up to `most` floats of the checkpoints' pending rows ([`Engine::snapshot_into`]);
    /// whether none is left. One whose copy fails is given up (the error returned).
    fn finish_checkpoints(&mut self, most: usize) -> Result<bool, String> {
        let mut left = most;
        for slot in 0..CHECKPOINTS {
            let Some(snapshots) = self.saved[slot].as_mut() else { continue };
            let copied = self.engines.iter_mut().zip(snapshots.iter_mut())
                .try_fold(0, |copied, (engine, snapshot)| Ok::<_, String>(copied + engine.finish_snapshot(snapshot, left - copied)?));
            match copied {
                Ok(copied) => left -= copied,
                Err(error) => {
                    self.saved[slot] = None;
                    self.order.retain(|&held| held != slot);
                    return Err(error);
                }
            }
            if left == 0 {
                break;
            }
        }
        Ok(!self.saved.iter().flatten().flatten().any(Snapshot::pending))
    }

    /// Tests only: copy at most `floats` of the checkpoints' pending rows a step between requests.
    #[doc(hidden)]
    pub fn limit_checkpoint_steps(&mut self, floats: usize) {
        self.step = floats;
    }

    /// Tests only: let the checkpoints take at most `bytes` (a budget too low for one).
    #[doc(hidden)]
    pub fn limit_checkpoints(&mut self, bytes: u64) {
        self.budget = bytes;
    }

    /// Diagnostics: every engine's sequence state ([`Engine::state_on`]) on each of its devices,
    /// engine after engine.
    #[doc(hidden)]
    pub fn state(&mut self) -> Result<Vec<(String, Vec<f32>)>, String> {
        let mut state = Vec::new();
        for (index, engine) in self.engines.iter_mut().enumerate() {
            for device in 0..engine.gpus().count() {
                state.extend(engine.state_on(device)?.into_iter().map(|(name, values)| (format!("engine {index} device {device} {name}"), values)));
            }
        }
        Ok(state)
    }
}

/// What placement settles about DSpark: the engine it sits beside and the experts it keeps.
struct DsparkPlan {
    engine: usize,
    resident: [usize; 3],
}

/// Whether an environment variable turns something off (set to 0).
fn off(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|value| value == "0")
}

/// DSpark's file, opened and authenticated, when the model's drafter is pinned, lies in its directory
/// ([`bundle::dspark_in`]) and can sit beside an engine: DS41_DSPARK=0 leaves it out, and so does
/// DS41_DRAFTS=0 (no verify passes to draft for). Its layout is checked against the model's.
fn open_dspark(root: &Path, quantization: &str, data_dir: &Path, target: &Pinned, groups: &[(Vec<usize>, std::ops::Range<usize>)])
               -> Result<Option<(Pinned, DsparkPlan)>, String> {
    if off("DS41_DSPARK") || off("DS41_DRAFTS") {
        return Ok(None);
    }
    let Some((source, path)) = bundle::dspark_in(root, quantization)? else { return Ok(None) };
    let (first, last) = (TAP_LAYERS[0], TAP_LAYERS[TAP_LAYERS.len() - 1]);
    let Some(engine) = groups.iter().position(|(_, range)| range.start <= first && range.end > last) else {
        eprintln!("Straydog DSpark: layers {first}-{last} are split between engines: no drafter");
        return Ok(None);
    };
    if !drafter::splits(groups[engine].0.len()) {
        eprintln!("Straydog DSpark: it does not split among {} GPUs: no drafter", groups[engine].0.len());
        return Ok(None);
    }
    let directory = path.parent().ok_or("the DSpark drafter has no directory")?;
    let files = Pinned::open(directory, data_dir, &source.pins()).map_err(|error| format!("DSpark drafter: {error}"))?;
    Dspark::from_headers(&files.header, &target.header).map_err(|error| format!("DSpark drafter: {error}"))?;
    let resident = drafter::resident_counts()?;
    Ok(Some((files, DsparkPlan { engine, resident })))
}

/// DSpark onto its engine's devices (the engine's feature tap on, before its first pass), its
/// VRAM held to what placement counted ([`drafter::footprint`]).
fn load_dspark(draft: &Pinned, plan: &DsparkPlan, engine: &mut Engine<'static, Pinned>, target: &Pinned, vocabulary: usize)
               -> Result<Drafter, String> {
    let allocated = |engine: &Engine<'static, Pinned>| -> Result<Vec<u64>, String> {
        engine.gpus().map(|gpu| Ok(gpu.info()?["allocated_bytes"].as_u64().unwrap_or(0))).collect()
    };
    let started = Instant::now();
    let before = allocated(engine)?;
    engine.enable_tap()?;
    let drafter = Drafter::load(&mut engine.beside()?, draft, target, vocabulary)?;
    let used: Vec<u64> = allocated(engine)?.iter().zip(&before).map(|(after, before)| after - before).collect();
    let planned: Vec<u64> = (0..used.len()).map(|index| drafter::footprint(index, used.len(), vocabulary, plan.resident)).collect();
    let mib = |bytes: u64| bytes as f64 / (1u64 << 20) as f64;
    eprintln!("Straydog DSpark: {:?} experts a stage, {:?} MiB a GPU (planned {:?}), loaded in {:.1} s",
              plan.resident, used.iter().map(|&bytes| mib(bytes).round()).collect::<Vec<_>>(),
              planned.iter().map(|&bytes| mib(bytes).round()).collect::<Vec<_>>(), started.elapsed().as_secs_f64());
    // (what it took within a few MiB of what was counted, alignment aside; more is the runtime's reserve's)
    if used.iter().zip(&planned).any(|(used, planned)| *used > planned + (8 << 20)) {
        eprintln!("Straydog DSpark took more VRAM than placement counted: the runtime's reserve holds the difference");
    }
    Ok(drafter)
}

/// Layer ranges, one per device, splitting where no layer reads an earlier part's compressed rows
/// and balancing the dense bytes (the head with the last part) in proportion to the budgets.
fn split(hp: &Hparams, dense: &[u64], head: u64, budgets: &[u64]) -> Result<Vec<std::ops::Range<usize>>, String> {
    let layers = hp.layers.len();
    let valid = |at: usize| {
        hp.layers[at..].iter().all(|plan| plan.compressed.is_none_or(|c| c.kv_source >= at && c.index_source >= at))
    };
    let cuts: Vec<usize> = (1..layers).filter(|&at| valid(at)).collect();
    let load = |range: std::ops::Range<usize>| dense[range.clone()].iter().sum::<u64>() + if range.end == layers { head } else { 0 };
    served::split_layers(layers, &cuts, &load, budgets).ok_or_else(|| format!("DeepSeek-V4.1 cannot be split across {} GPUs", budgets.len()))
}

fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Open and authenticate the pinned bundle in `root` (trees from `data_dir`) and place it.
/// `progress` gets (stage, done, total).
pub fn place(root: &Path, quantization: &str, data_dir: &Path, settings: &Settings,
             progress: &(dyn Fn(&str, u64, u64) + Sync)) -> Result<Model, String> {
    let started = Instant::now();
    let mut pins = bundle::on_demand(quantization)?;
    // The vision encoder, when its file is there (the model reads no images without it).
    if let Some(vision) = bundle::vision_source(quantization)?.filter(|vision| root.join(&vision.asset.name).is_file()) {
        pins.sidecars.push(bundle::Sidecar { asset: vision.asset, verity: vision.verity, architecture: bundle::VISION_ARCHITECTURE.into() });
    }
    let mut files = Pinned::open(root, data_dir, &pins)?;
    files.cache_tables(settings.row_cache_bytes)?;
    let open_seconds = started.elapsed().as_secs_f64();
    let hp = Hparams::from_model(&files.header, &files)?;
    let layers = hp.layers.len();
    // Per layer: the routed experts' bytes, and the resident bytes kept whole on every device
    // and divided among an engine's devices; the same for the head.
    let (mut expert_bytes, mut whole, mut parted) = (vec![0u64; layers], vec![0u64; layers], vec![0u64; layers]);
    let (mut head_whole, mut head_parted) = (0u64, 0u64);
    for (name, bytes) in files.tensor_bytes() {
        match layer_of(name) {
            // (Engram's tables are read a row at a time, never resident; the image routing biases are read with the first image)
            _ if crate::pinned::gathered(name) || name.ends_with(".exp_probs_b_vl.bias") => {}
            Some(layer) if layer < layers && name.ends_with("_exps.weight") => expert_bytes[layer] += bytes,
            Some(layer) if layer < layers && divided(name) => parted[layer] += bytes,
            Some(layer) if layer < layers => whole[layer] += bytes,
            _ if name.starts_with("output") && divided(name) => head_parted += bytes,
            _ if name.starts_with("output") => head_whole += bytes,
            _ => {}
        }
    }
    // The engines: their devices and layers.
    let groups: Vec<(Vec<usize>, std::ops::Range<usize>)> = match settings.split_mode {
        SplitMode::Experts => vec![(settings.devices.clone(), 0..layers)],
        SplitMode::Layer => {
            let dense: Vec<u64> = whole.iter().zip(&parted).map(|(a, b)| a + b).collect();
            let ranges = split(&hp, &dense, head_whole + head_parted, &settings.vram_bytes)?;
            settings.devices.iter().zip(ranges).map(|(&device, range)| (vec![device], range)).collect()
        }
    };
    // DSpark, when its file is there: opened (and its layout checked against the model's), and
    // the VRAM it takes counted with the engine it sits beside.
    let (dspark_files, dspark_plan) = match open_dspark(root, quantization, data_dir, &files, &groups)? {
        Some((files, plan)) => (Some(files), Some(plan)),
        None => (None, None),
    };
    let dspark_bytes = |group: usize| -> u64 {
        dspark_plan.as_ref().filter(|plan| plan.engine == group)
            .map_or(0, |plan| drafter::footprint(0, groups[group].0.len(), hp.vocabulary, plan.resident))
    };
    // The largest of the engine's layers' experts, each device holding its rows of it (whole
    // 256-byte units, as the store lays them out): the store takes VRAM as so many of these.
    let slot_bytes = |(devices, range): &(Vec<usize>, std::ops::Range<usize>)| {
        let largest = range.clone().map(|layer| expert_bytes[layer] / hp.experts as u64).max().unwrap_or(1).max(1);
        (largest / devices.len() as u64).next_multiple_of(256) * devices.len() as u64
    };
    // The sequence state of an engine's layers, on each of its devices: window rings (F32), each
    // kv source's compressed rows and keys (BF16), and a batch's new ones in F32 before rounding.
    let state = |range: &std::ops::Range<usize>| -> u64 {
        let ring = (hp.window + settings.batch.min(ATTENTION_ROWS)) * hp.head_dim * 4 * range.len();
        let sources: Vec<usize> = hp.layers.iter().enumerate().filter(|(layer, _)| range.contains(layer))
            .filter_map(|(layer, plan)| plan.compressed.filter(|c| c.kv_source == layer))
            .map(|c| (settings.context / c.ratio + 1) * (hp.head_dim + hp.index_dim) * 2).collect();
        let staging = if sources.is_empty() { 0 } else { (settings.batch.min(ATTENTION_ROWS) + 2) * (hp.head_dim + hp.index_dim) * 4 };
        (ring + sources.iter().sum::<usize>() + staging) as u64
    };
    let total_experts: u64 = expert_bytes.iter().sum();
    // Per engine: the experts its devices hold in VRAM together (each device its rows of every
    // one), from what each device has left after its resident share, state and work buffers.
    // An engine's budget: the least of its devices' (its devices hold equal shares).
    let budget = |devices: &[usize]| devices.iter()
        .map(|device| settings.vram_bytes[settings.devices.iter().position(|other| other == device).unwrap()]).min().unwrap_or(0);
    let mut slots = Vec::new();
    for (index, group) in groups.iter().enumerate() {
        let (devices, range) = group;
        let vram = budget(devices);
        let parts = devices.len() as u64;
        let head = if range.end == layers { head_whole + head_parted.div_ceil(parts) } else { 0 };
        let resident: u64 = range.clone().map(|layer| whole[layer] + parted[layer].div_ceil(parts)).sum::<u64>() + head;
        // a pass of up to the batch's rows, layer by layer: its buffers of every row (in the arena
        // lent to the expert store between long passes), and a few rows' own
        let rows = (settings.batch + OWN_ROWS.min(settings.batch)) as u64 * pass_row_bytes(&hp, devices.len(), range.clone());
        let expansions = if settings.half_gemm { EXPANSIONS } else { 0 };
        let fixed = resident + state(range) + rows + SCRATCH + expansions + RESERVE + dspark_bytes(index);
        let free = vram.checked_sub(fixed)
            .ok_or_else(|| format!("DeepSeek-V4.1 layers {range:?} need more than {:.1} GiB of VRAM on GPU {devices:?}", vram as f64 / (1u64 << 30) as f64))?;
        // Each device's share of an expert is 1/parts of a slot: free / share experts.
        let held = (free / (slot_bytes(group) / parts)) as usize;
        slots.push(held.min(range.len() * hp.experts) / devices.len() * devices.len());
    }
    let in_vram: u64 = groups.iter().zip(&slots).map(|(group, &slots)| slots as u64 * slot_bytes(group)).sum();
    // (planned, the tier leaves the checkpoints their host memory; it may hold more than the
    // experts VRAM lacks: a long pass takes the slots lent to the store back and streams its
    // experts through VRAM while the tier keeps theirs, and a tier of just the rest dropped ~1300
    // experts an agent's turn, to read them from the drive the next)
    let host = match settings.host_expert_bytes {
        Some(_) => served::host_tier(settings, total_experts.saturating_sub(in_vram)),
        None => served::host_tier(settings, u64::MAX).saturating_sub(CHECKPOINT_BYTES).min(total_experts),
    };
    // (the vision encoder's tensors are read with the first image, not at the start)
    let whole_bytes = files.whole_bytes(|name| !["vision.", "aligner.", "image_", "layers."].iter().any(|prefix| name.starts_with(prefix))
                                               && !name.ends_with(".exp_probs_b_vl.bias"))
        + dspark_files.as_ref().map_or(0, |files| files.whole_bytes(|_| true));
    let files: &'static Pinned = Box::leak(Box::new(files));
    let placed = Instant::now();
    let state_bytes = groups.iter().map(|(devices, range)| state(range) * devices.len() as u64).sum();
    // The page-locked host tier is shared out by the engines' expert bytes, then among an
    // engine's devices (each keeps its rows).
    let group_experts: Vec<u64> = groups.iter().map(|(_, range)| expert_bytes[range.clone()].iter().sum()).collect();
    let loading = std::sync::atomic::AtomicBool::new(true);
    eprintln!("Straydog DeepSeek prompt-pass products: {}; passes below 32 rows F32, up to four rows GEMV",
              if settings.half_gemm { "FP16 inputs/weights, F32 sums/output" } else { "FP32" });
    let (engines, drafter) = std::thread::scope(|scope| {
        // The dense weights verify as the engines take them: their bytes are the progress.
        scope.spawn(|| {
            while loading.load(std::sync::atomic::Ordering::Relaxed) {
                progress("load", files.verified_bytes() + dspark_files.as_ref().map_or(0, Pinned::verified_bytes), whole_bytes);
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
        });
        let mut engines = Vec::new();
        let result = (|| {
            for ((devices, range), (&slots, &experts)) in groups.iter().zip(slots.iter().zip(&group_experts)) {
                let share = (host as u128 * experts as u128 / total_experts.max(1) as u128 / devices.len() as u128) as u64;
                if host > 0 && share == 0 {
                    return Err("expert host budget is too small for every layer group".to_string());
                }
                let engine_settings = EngineSettings {
                    context: settings.context,
                    rows: settings.batch,
                    batch: settings.batch.min(ATTENTION_ROWS),
                    expert_slots: slots / devices.len(),
                    host_expert_bytes: share,
                    scratch_bytes: SCRATCH,
                    half_gemm: settings.half_gemm,
                    layers: range.clone(),
                };
                engines.push(Engine::new(hp.clone(), files, devices, budget(devices), engine_settings)?);
            }
            // DSpark beside its engine (its tensors verify as it takes them)
            let drafter = match (&dspark_files, &dspark_plan) {
                (Some(draft), Some(plan)) => Some(load_dspark(draft, plan, &mut engines[plan.engine], files, hp.vocabulary)?),
                _ => None,
            };
            Ok::<_, String>(drafter)
        })();
        loading.store(false, std::sync::atomic::Ordering::Relaxed);
        result.map(|drafter| (engines, drafter))
    })?;
    let drafting = match (dspark_files, dspark_plan, drafter) {
        (Some(files), Some(plan), Some(drafter)) => Some(Drafting {
            drafter, policy: Policy::default(), engine: plan.engine, resident: plan.resident, on: true, deciding: false, drafted: false, fixed_width: None, rings: (0..CHECKPOINTS).map(|_| None).collect(),
            _files: files,
        }),
        _ => None,
    };
    let place_seconds = placed.elapsed().as_secs_f64();
    let (planned, staging, pinned) = engines.iter().map(Engine::expert_host_memory)
        .fold((0, 0, 0), |sum, value| (sum.0 + value.0, sum.1 + value.1, sum.2 + value.2));
    if host > 0 && planned > host {
        return Err(format!("expert host pin plan {planned} exceeds budget {host}"));
    }
    let gib = (1u64 << 30) as f64;
    eprintln!("host tier: budget {:.3} GiB; planned limit {:.3} GiB (minimum staging {:.3} GiB included); pinned at placement {:.3} GiB",
              host as f64 / gib, planned as f64 / gib, staging as f64 / gib, pinned as f64 / gib);
    // (place's steps, each engine's: the wait before a server starts)
    let startup: Vec<(String, f64)> = engines.iter().flat_map(|engine| engine.startup.iter().cloned()).collect();
    let card = served::card(&files.header)?;
    let end = files.header.metadata.get("tokenizer.ggml.eos_token_id").and_then(|value| value.integer())
        .and_then(|id| u32::try_from(id).ok()).ok_or("GGUF tokenizer.ggml.eos_token_id is missing")?;
    let placement = json!({
        "split_mode": match settings.split_mode { SplitMode::Experts => "experts", SplitMode::Layer => "layer" },
        "layers": groups.iter().map(|(_, range)| [range.start, range.end]).collect::<Vec<_>>(),
        "devices": groups.iter().map(|(devices, _)| devices.clone()).collect::<Vec<_>>(),
        "expert_slots": engines.iter().map(|engine| engine.expert_capacity().0).collect::<Vec<_>>(),
        "host_expert_slots": engines.iter().map(|engine| engine.expert_capacity().1).collect::<Vec<_>>(),
        "expert_vram_bytes": groups.iter().zip(&slots).map(|(group, &slots)| slots as u64 * slot_bytes(group)).collect::<Vec<_>>(),
        "host_expert_bytes": host,
        "expert_bytes": total_experts,
        "prefill_batch_size": settings.batch,
        "dspark": drafting.as_ref().map(|drafting| json!({"engine": drafting.engine, "resident": drafting.resident})),
    });
    // (where a reply's generation prompt begins, and a user message: the states before them, and
    // before a prompt's first user message, are checkpointed)
    let special = |text: &str| files.header.metadata.get("tokenizer.ggml.tokens").and_then(|tokens| tokens.array())
        .and_then(|tokens| tokens.iter().position(|token| token.text() == Some(text)))
        .map(|id| id as u32);
    let (turn_start, root_end) = (special("<｜Assistant｜>"), special("<｜User｜>"));
    let drafts = std::env::var_os("DS41_DRAFTS").is_none_or(|value| value != "0");
    let verify_rows = if drafts { engines.iter().map(Engine::verify_rows).min().unwrap_or(1) } else { 1 };
    Ok(Session::new(Placed {
        engines: Engines { engines, devices_per_expert: groups[0].0.len(), verify_rows,
                           saved: (0..CHECKPOINTS).map(|_| None).collect(), order: Vec::new(), kept: None,
                           budget: CHECKPOINT_BYTES, step: CHECKPOINT_STEP, files, drafting },
        files, vocabulary: hp.vocabulary, layers, experts: hp.experts, context: settings.context, batch: settings.batch,
        end: vec![end], turn_start, root_end, card, placement, state_bytes,
        load_seconds: [("open".to_string(), open_seconds), ("place".to_string(), place_seconds)].into_iter().chain(startup).collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::super::engine::snapshot_bytes;
    use super::*;

    /// A checkpoint at the full 262144-token context (DeepSeek-V4.1's 40 layers, sources 2, 8, 14
    /// at ratio 2 and 20 at ratio 1) fits the budget: the oldest give theirs up for it, no more.
    /// One over the budget alone is not taken, and none given up for it.
    #[test]
    fn a_full_context_checkpoint_fits_and_one_over_the_budget_is_not_taken() {
        let position = 262_144;
        let groups = [position / 2, position / 2, position / 2, position];
        let full = snapshot_bytes(position, 128, 40, 512, 128, &groups, position);
        // (the compressed rows and index keys BF16: 0.85 GB)
        assert!(full > 768 << 20 && full < 1 << 30, "{full}");
        let held = [(0, 700 << 20), (1, 300 << 20), (2, 300 << 20)];
        assert_eq!(evictions(&held, full, CHECKPOINT_BYTES), Some(vec![0]));
        let small = snapshot_bytes(4096, 128, 40, 512, 128, &[2048, 2048, 2048, 4096], 4096);
        assert!(small < 64 << 20, "{small}");
        assert_eq!(evictions(&held, small, CHECKPOINT_BYTES), Some(vec![]));
        assert_eq!(evictions(&held, 1400 << 20, CHECKPOINT_BYTES), Some(vec![0]));
        assert_eq!(evictions(&held, CHECKPOINT_BYTES, CHECKPOINT_BYTES), Some(vec![0, 1, 2]));
        assert_eq!(evictions(&held, CHECKPOINT_BYTES + 1, CHECKPOINT_BYTES), None);
    }
}
