//! A partially resident model's assist level, chosen by measuring decode.
//!
//! The assist is the GPU taking a few of a layer's absent routes, which the CPU would otherwise run, from page-locked host banks over
//! PCIe. How many are worth taking depends on the machine: in the part of a layer where the two overlap, the GPU's side (its slot
//! experts and the assisted ones) or the CPU's is the longer, by a margin that its PCIe link, its CPU and its VRAM decide. An RTX 5070 Ti
//! on PCIe 5.0 x16 and an R9700 each gained from the most they were tried at; a PC with an RTX 4060 on PCIe 4.0 x8 may gain from fewer, or
//! from none. Nothing says which from the machine's specification, so the engine times decode at each level and keeps the fastest.
//!
//! HIP measures windows of one-row decode passes, holding drafts out while it does: [`HELD_WINDOW`] at the level in force, then
//! [`OTHER_WINDOW`] at each other level. The time is the wall time of the pass's layers (the "step: layers" of `decode_profile`), with
//! the slowest eighth trimmed. A candidate replaces the held level when it saves at least [`GAIN`] and exceeds [`CONFIDENCE`]
//! standard errors of the difference. With three levels this uses [`ROUND_PASSES`] passes per [`INTERVAL`] outside-round tokens.
//!
//! CUDA keeps the draft and copy width in force and measures whole passes until they produce 96 useful output tokens at the held
//! level and 48 at each other level. It compares total layer seconds per useful output token, without trimming. Each pass is one
//! independent observation for the ratio's standard error. If the same held level loses by at least 3%, beyond the same confidence
//! threshold, to any candidate in two consecutive completed rounds, the smallest qualified level of the second round replaces it.
//! The qualified candidate need not be the same in both rounds. Rejected drafts and END clipping affect the useful token count; an END-only pass
//! supplies no sample. Non-speculative decode contributes only after the token picked from its logits is successfully emitted.
//!
//! Both policies skip a warm-up pass of each window and retain unfinished windows across requests. CUDA excludes prefill and
//! credits only emitted decode output. The held level goes first and has the longer window. A changed mapped layer or a failed
//! CUDA level setter invalidates the measurements. If its first valid completed round at held level 0 finds a qualified candidate, CUDA waits only
//! 256 outside-round useful tokens for confirmation, once the held level is actually restored. `STRAYDOG_ASSIST_CONFIRM_INTERVAL`
//! sets that one wait to 1..=2048 tokens (0 disables the shortening); every later wait is 2048 tokens.
//!
//! Plain logic: the model feeds it passes and tokens and does what it returns (`Model::assist_apply`).

/// Decode tokens between two rounds, counted outside the rounds.
pub const INTERVAL: u64 = 2048;
/// Passes timed at the level in force, and at each other level.
pub const HELD_WINDOW: usize = 32;
pub const OTHER_WINDOW: usize = 16;
/// The part of the level in force's time that another level must save to replace it.
pub const GAIN: f64 = 0.02;
/// CUDA needs this saving in two consecutive completed rounds before replacing the held level.
const CUDA_GAIN: f64 = 0.03;
/// Useful output tokens measured at each CUDA level, keeping whole passes.
const CUDA_HELD_TOKENS: u64 = 96;
const CUDA_OTHER_TOKENS: u64 = 48;
/// The one confirmation wait after the first valid CUDA round loses.
const CUDA_CONFIRM_TOKENS: u64 = 256;

fn confirmation_tokens_named(value: Option<&str>) -> u64 {
    match value.and_then(|value| value.trim().parse::<u64>().ok()) {
        Some(0) => INTERVAL,
        Some(tokens @ 1..=INTERVAL) => tokens,
        _ => CUDA_CONFIRM_TOKENS,
    }
}
/// How many standard errors of the difference of two windows' means that saving must exceed.
pub const CONFIDENCE: f64 = 2.0;

/// Passes of one row a round with `levels` levels spends: each window, and the pass after each change of level that is not timed (the
/// layers' graphs are captured again in it) and the one at the start of a round.
pub const fn round_passes(levels: usize) -> usize {
    HELD_WINDOW + 1 + (levels - 1) * (OTHER_WINDOW + 1)
}

/// What a round spends with two levels besides the one in force.
pub const ROUND_PASSES: usize = round_passes(3);

/// The levels worth timing under `cap`, the most a layer's GPU takes: every one up to 2, as the CUDA build has them, and from there
/// halves of the cap (an R9700's 0, 2 and 4: its rate rose with each level from 1 to 4, and all five in windows would cost 4.7% of the
/// tokens).
pub fn levels(cap: u32) -> Vec<u32> {
    let mut levels: Vec<u32> = (0..=cap).step_by((cap / 2).max(1) as usize).collect();
    if levels.last() != Some(&cap) {
        levels.push(cap);
    }
    levels
}

/// A level's measured layer time and standard error: HIP's trimmed seconds per one-row pass,
/// or CUDA's total seconds per useful output token with a pass-cluster ratio error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timing {
    pub level: u32,
    pub seconds: f64,
    pub error: f64,
}

/// What a finished round found.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// The first round of the model's life.
    pub first: bool,
    /// The level in force when the round began, and the one in force after it.
    pub from: u32,
    pub chosen: u32,
    /// Each level timed, by level.
    pub measured: Vec<Timing>,
}

impl Outcome {
    /// Whether the log says so: the first round, and a round that changed the level.
    pub fn worth_saying(&self) -> bool {
        self.first || self.chosen != self.from
    }

    /// The log line: each level's milliseconds a pass of the layers with their standard error (to see whether the levels were told apart),
    /// and the level kept.
    pub fn line(&self) -> String {
        let mut text = String::from("Straydog assist:");
        for (at, timing) in self.measured.iter().enumerate() {
            let (level, seconds, error) = (timing.level, timing.seconds * 1e3, timing.error * 1e3);
            if at == 0 {
                text.push_str(&format!(" {level} a layer {seconds:.1}±{error:.1} ms a pass"));
            } else {
                text.push_str(&format!(", {level}: {seconds:.1}±{error:.1}"));
            }
        }
        if self.chosen == self.from {
            text.push_str(&format!("; keeping {}", self.chosen));
        } else {
            text.push_str(&format!("; now {} (was {})", self.chosen, self.from));
        }
        text
    }

    /// CUDA's timings use the useful output token as their unit; HIP keeps line().
    pub fn line_per_token(&self) -> String {
        self.line().replacen(" ms a pass", " ms a token", 1)
    }
}

struct Round {
    /// The levels to time: the one in force first.
    order: Vec<u32>,
    /// The window under way, and the passes of it still to be passed over.
    at: usize,
    skip: usize,
    samples: Vec<f64>,
    token_samples: Vec<(f64, u64)>,
    useful_tokens: u64,
    measured: Vec<Timing>,
    held: u32,
    /// Once every window is timed: the level chosen, until it is in force.
    target: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitialConfirmation {
    Eligible,
    AwaitingHeld(u32),
    Consumed,
}

pub struct Tuner {
    levels: Vec<u32>,
    level: u32,
    /// Decode tokens since the last round, outside the rounds.
    since: u64,
    round: Option<Round>,
    rounds: u32,
    outcome: Option<Outcome>,
    cuda: bool,
    /// The held level that lost its first completed round, awaiting a second loss.
    pending: Option<u32>,
    /// An invalidated round's held level, until the model actually restores it.
    restore: Option<u32>,
    /// CUDA diagnostics are selected once; HIP never enables them.
    profile: bool,
    /// Independent of completed-round numbering: invalid votes do not use this opportunity.
    initial_confirmation: InitialConfirmation,
    confirmation_tokens: u64,
    /// The actual number of outside-round tokens to wait; since is never offset.
    next_interval: u64,
}

/// The mean of `samples` without the slowest eighth, and its standard error: a stall of the machine in a pass (a page fault's wait,
/// another process's thread on a core) says nothing of its level, and one pass in 16 at ten times the rest moves the plain mean by more
/// than [`GAIN`]. The error is that of the passes kept.
pub(crate) fn trimmed_mean(samples: &[f64]) -> (f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let kept = &sorted[..sorted.len() - sorted.len() / 8];
    let n = kept.len() as f64;
    let mean = kept.iter().sum::<f64>() / n;
    let variance = kept.iter().map(|seconds| (seconds - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (variance / n).sqrt())
}

/// CUDA's total layer seconds per useful output token. Passes, not individual
/// tokens, are the independent observations of the ratio's standard error.
fn token_mean(samples: &[(f64, u64)]) -> (f64, f64) {
    let tokens = samples.iter().map(|&(_, tokens)| tokens as f64).sum::<f64>();
    if tokens == 0.0 { return (0.0, f64::INFINITY); }
    let mean = samples.iter().map(|&(seconds, _)| seconds).sum::<f64>() / tokens;
    if samples.len() < 2 { return (mean, f64::INFINITY); }
    let residual = samples.iter().map(|&(seconds, count)| (seconds - mean * count as f64).powi(2)).sum::<f64>();
    let passes = samples.len() as f64;
    (mean, (passes / (passes - 1.0) * residual).sqrt() / tokens)
}

impl Tuner {
    /// A tuner of `levels` (the levels that can be set) with `level` in force. It is due at the first request.
    pub fn new(mut levels: Vec<u32>, level: u32) -> Tuner {
        levels.push(level);
        levels.sort_unstable();
        levels.dedup();
        Tuner { levels, level, since: INTERVAL + 1, round: None, rounds: 0, outcome: None,
                cuda: false, pending: None, restore: None, profile: false,
                initial_confirmation: InitialConfirmation::Consumed, confirmation_tokens: INTERVAL, next_interval: INTERVAL }
    }

    /// CUDA's tuner: the same held level must lose by at least 3%, beyond the same
    /// standard-error threshold, in two complete rounds. The second round's smallest
    /// qualified level wins, irrespective of the candidates' speed ranking.
    pub fn new_cuda(levels: Vec<u32>, level: u32) -> Tuner {
        let confirmation_tokens = confirmation_tokens_named(std::env::var("STRAYDOG_ASSIST_CONFIRM_INTERVAL").ok().as_deref());
        Self::new_cuda_with_confirm_tokens(levels, level, confirmation_tokens)
    }

    fn new_cuda_with_confirm_tokens(levels: Vec<u32>, level: u32, confirmation_tokens: u64) -> Tuner {
        let profile = std::env::var("DS41_PROFILE").is_ok_and(|value| !value.is_empty() && value != "0");
        let tuner = Tuner { cuda: true, profile, initial_confirmation: InitialConfirmation::Eligible, confirmation_tokens,
                            ..Self::new(levels, level) };
        tuner.profile_state("new", None);
        tuner
    }

    pub fn profiling(&self) -> bool { self.profile }

    /// Raw counters and round state, without consuming an outcome or rounding routes/token.
    pub fn profile_state(&self, event: &str, routes: Option<[u64; 3]>) {
        if !self.profile { return; }
        if let Some(round) = &self.round {
            eprintln!("Straydog assist profile: event={event} completed={} level={} held={} order={:?} window={} skip={} samples={} useful_tokens={} unit=seconds_per_token measured={:?} loss_streak_held={:?} restore={:?} routes_assisted_gpu_cpu={routes:?}",
                      self.rounds, self.level, round.held, round.order, round.at, round.skip, round.token_samples.len(), round.useful_tokens, round.measured, self.pending, self.restore);
        } else {
            eprintln!("Straydog assist profile: event={event} completed={} level={} levels={:?} since={} next_interval={} confirmation_tokens={} initial_confirmation={:?} loss_streak_held={:?} restore={:?} unit=seconds_per_token outcome={:?} routes_assisted_gpu_cpu={routes:?}",
                      self.rounds, self.level, self.levels, self.since, self.next_interval, self.confirmation_tokens, self.initial_confirmation, self.pending, self.restore, self.outcome);
        }
    }

    /// The level in force.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Whether there is more than one level to choose from.
    pub fn active(&self) -> bool {
        self.levels.len() >= 2
    }

    /// Whether windows are being measured, at the ordinary width on CUDA.
    pub fn probing(&self) -> bool {
        self.round.as_ref().is_some_and(|round| round.target.is_none())
    }

    /// HIP keeps its one-row measurement passes; CUDA measures the ordinary width.
    pub fn force_single_row(&self) -> bool { !self.cuda && self.probing() }

    fn begin(&mut self) {
        // A shortened wait is used only by this next round, never by a later one.
        self.next_interval = INTERVAL;
        let mut order = vec![self.level];
        order.extend(self.levels.iter().copied().filter(|level| *level != self.level));
        self.round = Some(Round { order, at: 0, skip: 1, samples: Vec::new(), token_samples: Vec::new(), useful_tokens: 0,
                                  measured: Vec::new(), held: self.level, target: None });
    }

    /// `count` outside-round decode tokens were generated: begin at the selected wait,
    /// normally [`INTERVAL`], with one shorter CUDA confirmation after its initial loss.
    pub fn tokens(&mut self, count: u64) {
        if self.round.is_some() || self.restore.is_some() || !self.active() {
            return;
        }
        self.since = self.since.saturating_add(count);
        if self.since >= self.next_interval {
            self.begin();
        }
    }

    /// A request begins decoding: begin if more than the selected wait has gone since
    /// the last round (also retaining the existing due-next-request rule after restart).
    pub fn request(&mut self) {
        if self.round.is_none() && self.restore.is_none() && self.active() && self.since > self.next_interval {
            self.begin();
        }
    }

    /// The level chosen from `measured`: the fastest of those that are [`GAIN`] faster than `held` and by more than [`CONFIDENCE`] standard
    /// errors of the difference (the two windows' errors added as squares), else `held`.
    fn choose(measured: &[Timing], held: u32) -> u32 {
        let Some(&kept) = measured.iter().find(|timing| timing.level == held) else {
            return measured.iter().min_by(|a, b| a.seconds.total_cmp(&b.seconds)).map_or(held, |best| best.level);
        };
        measured.iter().filter(|timing| {
            let saved = kept.seconds - timing.seconds;
            timing.level != held && timing.seconds <= kept.seconds * (1.0 - GAIN)
                && saved > CONFIDENCE * kept.error.hypot(timing.error)
        }).min_by(|a, b| a.seconds.total_cmp(&b.seconds)).map_or(held, |best| best.level)
    }

    fn valid_cuda_timings(measured: &[Timing], held: u32) -> bool {
        measured.iter().any(|timing| timing.level == held) && !measured.iter().enumerate().any(|(index, timing)| {
            !timing.seconds.is_finite() || timing.seconds <= 0.0 || !timing.error.is_finite() || timing.error < 0.0
                || measured[..index].iter().any(|other| other.level == timing.level)
        })
    }

    fn qualified_cuda(measured: &[Timing], held: u32) -> Vec<u32> {
        if !Self::valid_cuda_timings(measured, held) { return Vec::new(); }
        let Some(&kept) = measured.iter().find(|timing| timing.level == held) else { return Vec::new() };
        let mut qualified: Vec<u32> = measured.iter().filter(|timing| {
            let saved = kept.seconds - timing.seconds;
            timing.level != held && timing.seconds <= kept.seconds * (1.0 - CUDA_GAIN)
                && saved > CONFIDENCE * kept.error.hypot(timing.error)
        }).map(|timing| timing.level).collect();
        qualified.sort_unstable();
        qualified
    }

    /// The next window, or the end of the round. The level to put in force, if it is not.
    fn advance(&mut self) -> Option<u32> {
        let round = self.round.as_mut()?;
        if self.cuda {
            if let Some(level) = round.target {
                return (level != self.level).then_some(level);
            }
        }
        if let Some(&level) = round.order.get(round.at) {
            round.skip = 1;
            return (level != self.level).then_some(level);
        }
        let chosen = if self.cuda {
            let valid = Self::valid_cuda_timings(&round.measured, round.held)
                && round.at == round.order.len() && round.measured.len() == round.order.len()
                && round.order.iter().all(|level| round.measured.iter().any(|timing| timing.level == *level));
            let qualified = if valid { Self::qualified_cuda(&round.measured, round.held) } else { Vec::new() };
            let before = self.pending;
            let chosen = if qualified.is_empty() {
                self.pending = None;
                round.held
            } else if self.pending == Some(round.held) {
                self.pending = None;
                qualified[0]
            } else {
                self.pending = Some(round.held);
                round.held
            };
            if self.initial_confirmation == InitialConfirmation::Eligible && valid {
                self.initial_confirmation = if round.held == 0 && !qualified.is_empty() && chosen == round.held {
                    InitialConfirmation::AwaitingHeld(round.held)
                } else { InitialConfirmation::Consumed };
            }
            if self.profile {
                let fastest = round.measured.iter().min_by(|a, b| a.seconds.total_cmp(&b.seconds)).map(|value| value.level);
                eprintln!("Straydog assist profile: decision round={} held={} fastest={fastest:?} qualified={qualified:?} chosen={} loss_streak_before={before:?} loss_streak_after={:?} unit=seconds_per_token measured={:?}",
                          self.rounds + 1, round.held, chosen, self.pending, round.measured);
            }
            chosen
        } else { Self::choose(&round.measured, round.held) };
        if chosen == self.level {
            self.conclude();
            return None;
        }
        round.target = Some(chosen);
        Some(chosen)
    }

    fn conclude(&mut self) {
        let Some(round) = self.round.take() else { return };
        self.rounds += 1;
        self.since = 0;
        if let InitialConfirmation::AwaitingHeld(held) = self.initial_confirmation {
            // conclude follows the actual held setter acknowledgement, never its request.
            self.initial_confirmation = InitialConfirmation::Consumed;
            if round.held == held && self.level == held {
                self.next_interval = self.confirmation_tokens;
            }
        }
        let mut measured = round.measured;
        measured.sort_by_key(|timing| timing.level);
        self.outcome = Some(Outcome { first: self.rounds == 1, from: round.held, chosen: self.level, measured });
        self.profile_state("complete", None);
    }

    /// A decode pass of one row took `seconds` over its layers. The level to put in force, if the pass ended a window.
    pub fn pass(&mut self, seconds: f64) -> Option<u32> {
        if self.cuda { return self.pass_tokens(seconds, 1); }
        let round = self.round.as_mut().filter(|round| round.target.is_none())?;
        if round.skip > 0 {
            round.skip -= 1;
            return None;
        }
        round.samples.push(seconds);
        if round.samples.len() < if round.at == 0 { HELD_WINDOW } else { OTHER_WINDOW } {
            return None;
        }
        let (seconds, error) = trimmed_mean(&round.samples);
        round.measured.push(Timing { level: round.order[round.at], seconds, error });
        round.samples.clear();
        round.at += 1;
        self.profile_state("window_complete", None);
        self.advance()
    }

    /// A successfully committed CUDA decode pass, after acceptance and END clipping.
    /// Zero useful tokens (an END-only pass) supply no sample, token credit or vote.
    pub fn pass_tokens(&mut self, seconds: f64, useful_tokens: u64) -> Option<u32> {
        if !self.cuda || useful_tokens == 0 || !seconds.is_finite() || seconds <= 0.0 { return None; }
        let round = self.round.as_mut().filter(|round| round.target.is_none())?;
        if round.skip > 0 {
            round.skip -= 1;
            return None;
        }
        round.token_samples.push((seconds, useful_tokens));
        round.useful_tokens = round.useful_tokens.saturating_add(useful_tokens);
        let window = if round.at == 0 { CUDA_HELD_TOKENS } else { CUDA_OTHER_TOKENS };
        if round.useful_tokens < window { return None; }
        let (seconds, error) = token_mean(&round.token_samples);
        round.measured.push(Timing { level: round.order[round.at], seconds, error });
        round.token_samples.clear();
        round.useful_tokens = 0;
        round.at += 1;
        self.profile_state("window_complete", None);
        self.advance()
    }

    /// The layers the GPU assists have changed (one more was page-locked): what a round measured was of the layers before, so one under way is
    /// dropped and a round is due at the next request. CUDA restores the held level if a
    /// probe was in force; the caller must set it on the GPU before acknowledging with set().
    pub fn restart(&mut self) -> Option<u32> {
        self.profile_state("restart_discard", None);
        if self.cuda {
            self.pending = None;
            if self.initial_confirmation != InitialConfirmation::Eligible {
                self.initial_confirmation = InitialConfirmation::Consumed;
            }
            self.next_interval = INTERVAL;
            self.restore = self.restore.or_else(|| self.round.as_ref().map(|round| round.held).filter(|held| *held != self.level));
        }
        self.round = None;
        self.since = INTERVAL + 1;
        self.restore
    }

    /// The level asked for is in force.
    pub fn set(&mut self, level: u32) {
        if self.profile {
            eprintln!("Straydog assist profile: setter_succeeded from={} to={level}", self.level);
        }
        if self.cuda && self.round.is_none() && level != self.level && self.restore != Some(level) {
            self.pending = None;
            self.initial_confirmation = InitialConfirmation::Consumed;
            self.next_interval = INTERVAL;
        }
        self.level = level;
        if self.restore == Some(level) {
            self.restore = None;
        }
        if self.round.as_ref().is_some_and(|round| round.target == Some(level)) {
            self.conclude();
        }
    }

    /// This level is the CUDA held level being restored, whose failure must be returned to
    /// the caller instead of trying another unconfirmed candidate.
    pub fn restoring(&self, level: u32) -> bool {
        self.cuda && (self.restore == Some(level)
            || self.round.as_ref().is_some_and(|round| round.target == Some(level) && round.held == level))
    }

    /// The level asked for cannot be set: it is not tried again. The level to put in force instead, if any.
    pub fn cannot_set(&mut self, level: u32) -> Option<u32> {
        if self.profile { eprintln!("Straydog assist profile: setter_failed requested={level}"); }
        self.profile_state("cannot_set_discard", None);
        if self.cuda {
            self.pending = None;
            self.initial_confirmation = InitialConfirmation::Consumed;
            self.next_interval = INTERVAL;
            let held = self.restore.or_else(|| self.round.as_ref().map(|round| round.held)).unwrap_or(self.level);
            self.levels.retain(|other| *other != level || *other == held);
            self.round = None;
            self.since = 0;
            self.outcome = None;
            // A failed multi-device setter may have changed some devices even when level()
            // still equals held. Always request an actual restore, then wait for its set().
            self.restore = Some(held);
            return Some(held);
        }
        self.levels.retain(|other| *other != level);
        let round = self.round.as_mut()?;
        round.measured.retain(|timing| timing.level != level);
        if round.target.is_some() {
            // (the round was over and its choice cannot be had: the best of the rest)
            let chosen = Self::choose(&round.measured, round.held);
            if chosen == self.level {
                self.conclude();
                return None;
            }
            round.target = Some(chosen);
            return Some(chosen);
        }
        if let Some(index) = round.order.iter().position(|other| *other == level) {
            round.order.remove(index);
            if index < round.at {
                round.at -= 1;
            }
        }
        round.samples.clear();
        self.advance()
    }

    /// What the last round found, once.
    pub fn outcome(&mut self) -> Option<Outcome> {
        self.outcome.take()
    }
}

#[cfg(test)]
mod cuda_tests {
    use super::*;

    fn choose_cuda(measured: &[Timing], held: u32) -> u32 {
        Tuner::qualified_cuda(measured, held).first().copied().unwrap_or(held)
    }

    fn finish(tuner: &mut Tuner, times: &dyn Fn(u32, usize) -> f64) -> (Outcome, Vec<u32>) {
        tuner.tokens(INTERVAL);
        tuner.request();
        assert!(tuner.probing());
        let (mut commands, mut nth, mut passes) = (Vec::new(), 0, 0);
        while tuner.round.is_some() {
            passes += 1;
            assert!(passes <= (CUDA_HELD_TOKENS + 2 * CUDA_OTHER_TOKENS + 3) as usize);
            if let Some(level) = tuner.pass(times(tuner.level(), nth)) {
                commands.push(level);
                tuner.set(level);
                nth = 0;
            } else { nth += 1; }
        }
        (tuner.outcome().unwrap(), commands)
    }

    fn times(level: u32, _: usize) -> f64 { if level == 2 { 0.080 } else { 0.100 } }

    #[test]
    fn cuda_request_boundaries_keep_partial_windows_and_the_first_win() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        tuner.request();
        for _ in 0..11 { assert_eq!(tuner.pass(0.100), None); }
        let window = tuner.round.as_ref().unwrap();
        let progress = (window.at, window.skip, (window.token_samples.clone(), window.useful_tokens));
        tuner.request();
        let window = tuner.round.as_ref().unwrap();
        assert_eq!((window.at, window.skip, (window.token_samples.clone(), window.useful_tokens)), progress);
        assert_eq!(finish(&mut tuner, &times).0.chosen, 0);
        assert_eq!(tuner.pending, Some(0));
        tuner.request();
        assert!(!tuner.probing());
        assert_eq!(tuner.pending, Some(0));
        tuner.tokens(INTERVAL);
        for _ in 0..3 { assert_eq!(tuner.pass(0.100), None); }
        tuner.request();
        assert_eq!(tuner.pending, Some(0));
        assert_eq!(finish(&mut tuner, &times).0.chosen, 2);
    }

    #[test]
    fn cuda_needs_two_complete_wins_and_restores_the_first_winner_from_the_last_probe() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        let (first, commands) = finish(&mut tuner, &times);
        assert_eq!(commands, [1, 2, 0], "even when the winning candidate was the last probe, return to the actual held level");
        assert_eq!((first.from, first.chosen), (0, 0));
        assert!(first.first && first.worth_saying());
        assert_eq!(tuner.level(), 0);
        assert_eq!(tuner.pending, Some(0));
        let (second, commands) = finish(&mut tuner, &times);
        assert_eq!(commands, [1, 2]);
        assert_eq!((second.from, second.chosen), (0, 2));
        assert!(!second.first && second.worth_saying());
        assert_eq!(tuner.level(), 2);
        assert_eq!(tuner.pending, None);
    }

    #[test]
    fn cuda_requires_at_least_three_percent_and_strictly_more_than_two_standard_errors() {
        let held = Timing { level: 0, seconds: 100.0, error: 0.0 };
        let candidate = |seconds, error| Timing { level: 1, seconds, error };
        assert_eq!(choose_cuda(&[held, candidate(97.00000000000001, 0.0)], 0), 0);
        assert_eq!(choose_cuda(&[held, candidate(97.0, 0.0)], 0), 1);
        assert_eq!(choose_cuda(&[held, candidate(96.99999999999999, 0.0)], 0), 1);
        assert_eq!(choose_cuda(&[held, candidate(97.0, 1.5)], 0), 0, "equality with CONFIDENCE is insufficient");
        assert_eq!(choose_cuda(&[held, candidate(97.0, 1.49)], 0), 1);
        assert_eq!(choose_cuda(&[candidate(1.0, 0.0)], 0), 0, "a missing held window cannot confirm a gain");
        // The old constructor still has its 2% single-round rule.
        assert_eq!(Tuner::choose(&[held, candidate(97.9, 0.0)], 0), 1);
        assert_eq!(choose_cuda(&[held, candidate(97.9, 0.0)], 0), 0);
    }

    #[test]
    fn cuda_stays_at_zero_through_two_percent_noise_and_small_gains() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        for _ in 0..12 {
            let (outcome, _) = finish(&mut tuner, &|level, nth| {
                let base = if level == 0 { 0.100 } else { 0.098 };
                base * (1.0 + [0.02, -0.02, 0.01, -0.01][nth % 4])
            });
            assert_eq!((outcome.from, outcome.chosen), (0, 0));
            assert_eq!(tuner.pending, None);
        }
    }

    #[test]
    fn cuda_a_changed_qualified_candidate_still_confirms_the_same_held_loss() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        let one = |level, _| if level == 1 { 0.080 } else { 0.100 };
        assert_eq!(finish(&mut tuner, &one).0.chosen, 0);
        assert_eq!(tuner.pending, Some(0));
        assert_eq!(finish(&mut tuner, &times).0.chosen, 2);
        assert_eq!(tuner.pending, None);
    }

    #[test]
    fn cuda_a_round_without_a_winner_breaks_the_streak() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        finish(&mut tuner, &times);
        assert_eq!(finish(&mut tuner, &|_, _| 0.100).0.chosen, 0);
        assert_eq!(tuner.pending, None);
        assert_eq!(finish(&mut tuner, &times).0.chosen, 0);
        assert_eq!(finish(&mut tuner, &times).0.chosen, 2);
    }

    #[test]
    fn cuda_restart_during_a_probe_waits_for_the_actual_gpu_restore_and_breaks_the_streak() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        finish(&mut tuner, &times);
        tuner.tokens(INTERVAL);
        let next = (0..=CUDA_HELD_TOKENS).find_map(|_| tuner.pass(0.100));
        assert_eq!(next, Some(1));
        tuner.set(1);
        assert_eq!(tuner.restart(), Some(0));
        assert_eq!(tuner.level(), 1, "the model has not restored the GPU yet");
        assert_eq!(tuner.pending, None);
        assert!(tuner.restoring(0));
        tuner.tokens(10 * INTERVAL);
        tuner.request();
        assert!(!tuner.probing(), "do not begin from the temporary probe while its restore is pending");
        assert_eq!(tuner.restart(), Some(0), "a failed restore remains due");
        assert_eq!(tuner.level(), 1);
        tuner.set(0);
        assert!(!tuner.restoring(0));
        assert_eq!(finish(&mut tuner, &times).0.chosen, 0);
        assert_eq!(finish(&mut tuner, &times).0.chosen, 2);
    }

    #[test]
    fn cuda_cannot_set_a_probe_invalidates_the_round_instead_of_adopting_a_runner_up() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        finish(&mut tuner, &times);
        tuner.tokens(INTERVAL);
        let mut failed = false;
        for _ in 0..(CUDA_HELD_TOKENS + 2 * CUDA_OTHER_TOKENS + 3) as usize {
            if let Some(level) = tuner.pass(if tuner.level() == 1 { 0.070 } else { 0.100 }) {
                if level == 2 {
                    assert_eq!(tuner.cannot_set(level), Some(0));
                    failed = true;
                    break;
                }
                tuner.set(level);
            }
        }
        assert!(failed);
        assert_eq!(tuner.level(), 1, "the actual restore has not succeeded");
        assert!(tuner.round.is_none() && tuner.pending.is_none() && tuner.outcome().is_none());
        assert!(tuner.restoring(0));
        tuner.set(0);
        assert_eq!(tuner.levels, [0, 1]);
        let one = |level, _| if level == 1 { 0.070 } else { 0.100 };
        assert_eq!(finish(&mut tuner, &one).0.chosen, 0);
        assert_eq!(finish(&mut tuner, &one).0.chosen, 1);
    }

    #[test]
    fn cuda_cannot_set_a_confirmed_target_still_restores_held_without_a_runner_up() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        let one = |level, _| match level { 1 => 0.060, 2 => 0.080, _ => 0.100 };
        finish(&mut tuner, &one);
        tuner.tokens(INTERVAL);
        let mut target = None;
        for _ in 0..(CUDA_HELD_TOKENS + 2 * CUDA_OTHER_TOKENS + 3) as usize {
            if let Some(level) = tuner.pass(one(tuner.level(), 0)) {
                if tuner.round.as_ref().is_some_and(|round| round.target == Some(1)) { target = Some(level); break; }
                tuner.set(level);
            }
        }
        assert_eq!(target, Some(1));
        assert_eq!(tuner.level(), 2);
        assert_eq!(tuner.cannot_set(1), Some(0));
        assert!(tuner.restoring(0) && tuner.pending.is_none() && tuner.round.is_none());
        assert_eq!(tuner.level(), 2);
        tuner.set(0);
        assert_eq!(tuner.level(), 0);
        assert_eq!(tuner.levels, [0, 2]);
    }

    #[test]
    fn cuda_a_partial_set_failure_requests_held_even_if_the_recorded_level_is_already_held() {
        let mut tuner = Tuner::new_cuda(vec![0, 1], 0);
        tuner.request();
        assert_eq!((0..=CUDA_HELD_TOKENS).find_map(|_| tuner.pass(0.100)), Some(1));
        // The multi-GPU setter failed before tuner.set(1); some GPUs may still have changed.
        assert_eq!(tuner.level(), 0);
        assert_eq!(tuner.cannot_set(1), Some(0));
        assert!(tuner.restoring(0));
        tuner.request();
        assert!(!tuner.probing());
        tuner.set(0);
        assert!(!tuner.restoring(0));
    }

    #[test]
    fn cuda_the_same_round_cannot_vote_twice_while_its_restore_is_waiting() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        tuner.request();
        let mut restore = None;
        for _ in 0..(CUDA_HELD_TOKENS + 2 * CUDA_OTHER_TOKENS + 3) as usize {
            if let Some(level) = tuner.pass(times(tuner.level(), 0)) {
                if tuner.restoring(level) { restore = Some(level); break; }
                tuner.set(level);
            }
        }
        assert_eq!(restore, Some(0));
        for _ in 0..5 {
            assert_eq!(tuner.pass(0.080), None);
            assert_eq!(tuner.advance(), Some(0));
            assert_eq!(tuner.pending, Some(0));
            assert_eq!(tuner.rounds, 0);
        }
        tuner.set(0);
        assert_eq!(tuner.rounds, 1);
        assert_eq!(tuner.level(), 0);
    }
}

#[cfg(test)]
mod token_tests {
    use super::*;

    #[test]
    fn token_rates_weight_by_useful_tokens_and_keep_passes_as_the_error_units() {
        let (mean, error) = token_mean(&[(1.0, 1), (2.0, 3)]);
        assert_eq!(mean, 0.75);
        assert_eq!(error, 0.125);
        // Eight tokens are two independent passes, not eight independent observations.
        assert_eq!(token_mean(&[(4.0, 4), (8.0, 4)]), (1.5, 0.5));
        assert!(token_mean(&[(4.0, 4)]).1.is_infinite());
        assert!(token_mean(&[]).1.is_infinite());
        // CUDA retains a stall rather than trimming it out of the objective.
        let samples: Vec<_> = (0..16).map(|i| (if i == 15 { 17.0 } else { 1.0 }, 1)).collect();
        assert_eq!(token_mean(&samples).0, 2.0);
    }

    #[test]
    fn cuda_keeps_normal_width_while_hip_keeps_single_row_sampling() {
        let (mut cuda, mut hip) = (Tuner::new_cuda(vec![0, 1], 0), Tuner::new(vec![0, 1], 1));
        cuda.request(); hip.request();
        assert!(cuda.probing() && !cuda.force_single_row());
        assert!(hip.probing() && hip.force_single_row());
        assert_eq!(hip.pass_tokens(0.01, 4), None);
        assert!(hip.round.as_ref().unwrap().token_samples.is_empty());
    }

    #[test]
    fn zero_output_or_invalid_time_changes_neither_warmup_nor_token_progress() {
        let mut tuner = Tuner::new_cuda(vec![0, 1], 0);
        tuner.request();
        for (seconds, tokens) in [(1.0, 0), (f64::NAN, 4), (f64::INFINITY, 4), (-1.0, 4), (0.0, 4)] {
            assert_eq!(tuner.pass_tokens(seconds, tokens), None);
        }
        let round = tuner.round.as_ref().unwrap();
        assert_eq!((round.skip, round.useful_tokens, round.token_samples.len()), (1, 0, 0));
        tuner.pass_tokens(0.04, 4); // The first useful pass warms the graph, but supplies no measurement.
        tuner.pass_tokens(0.01, 1);
        for _ in 0..100 { assert_eq!(tuner.pass_tokens(1.0, 0), None); }
        let round = tuner.round.as_ref().unwrap();
        assert_eq!((round.skip, round.useful_tokens, round.token_samples.len()), (0, 1, 1));
        assert_eq!(tuner.pending, None);
    }

    #[test]
    fn token_windows_skip_one_whole_pass_and_keep_the_whole_boundary_pass() {
        let mut tuner = Tuner::new_cuda(vec![0, 1], 0);
        tuner.request();
        tuner.pass_tokens(99.0, 4);
        for _ in 0..23 { assert_eq!(tuner.pass_tokens(0.04, 4), None); }
        assert_eq!(tuner.round.as_ref().unwrap().useful_tokens, 92);
        assert_eq!(tuner.pass_tokens(0.04, 4), Some(1));
        tuner.set(1);
        tuner.pass_tokens(99.0, 3);
        for _ in 0..15 { assert_eq!(tuner.pass_tokens(0.03, 3), None); }
        assert_eq!(tuner.round.as_ref().unwrap().useful_tokens, 45);
        assert_eq!(tuner.pass_tokens(0.08, 4), Some(0));
        let measured = &tuner.round.as_ref().unwrap().measured;
        assert!((measured[0].seconds - 0.01).abs() < 1e-15);
        assert!((measured[1].seconds - 0.53 / 49.0).abs() < 1e-15, "all four useful tokens of the final pass count");
        tuner.set(0);
        assert_eq!(tuner.since, 0, "the last measured pass is not an outside-round interval tick");
    }

    fn finish_mixed(tuner: &mut Tuner) -> Outcome {
        tuner.tokens(INTERVAL);
        tuner.request();
        let (mut nth, mut passes) = (0, 0);
        while tuner.round.is_some() {
            passes += 1;
            assert!(passes < 1000);
            let tokens = [1, 4, 2, 3][nth % 4];
            // One-row time is identical. Only the ordinary multirow passes benefit.
            let seconds = if tuner.level() == 0 { [0.020, 0.052, 0.035, 0.045][nth % 4] }
                          else { [0.020, 0.028, 0.025, 0.026][nth % 4] };
            if let Some(level) = tuner.pass_tokens(seconds, tokens) { tuner.set(level); nth = 0; }
            else { nth += 1; }
        }
        tuner.outcome().unwrap()
    }

    #[test]
    fn two_mixed_width_rounds_can_confirm_a_gain_that_has_no_one_row_gain() {
        let mut tuner = Tuner::new_cuda(vec![0, 1], 0);
        assert_eq!(finish_mixed(&mut tuner).chosen, 0);
        assert_eq!(tuner.pending, Some(0));
        let outcome = finish_mixed(&mut tuner);
        assert_eq!(outcome.chosen, 1, "actual multirow throughput is the objective");
        assert_eq!(tuner.pending, None);
        assert!(outcome.line_per_token().contains("ms a token"));
        assert!(outcome.line().contains("ms a pass"), "the HIP formatter remains available unchanged");
    }

    #[test]
    fn token_windows_and_confirmations_survive_request_boundaries() {
        let mut tuner = Tuner::new_cuda_with_confirm_tokens(vec![0, 1], 0, 256);
        tuner.request();
        tuner.pass_tokens(0.04, 4);
        for _ in 0..12 { tuner.pass_tokens(0.04, 4); }
        let before = tuner.round.as_ref().unwrap().token_samples.clone();
        tuner.request();
        assert_eq!(tuner.round.as_ref().unwrap().useful_tokens, 48);
        assert_eq!(tuner.round.as_ref().unwrap().token_samples, before);
        // Complete both windows with constant rates; request changes never supply a vote.
        while tuner.round.is_some() {
            let seconds = if tuner.level() == 0 { 0.04 } else { 0.024 };
            if let Some(level) = tuner.pass_tokens(seconds, 4) { tuner.set(level); }
            tuner.request();
        }
        assert_eq!(tuner.pending, Some(0));
        assert_eq!(tuner.level(), 0);
        tuner.outcome();
        tuner.request();
        assert!(!tuner.probing());
        tuner.tokens(255);
        tuner.request();
        assert!(!tuner.probing());
        tuner.tokens(1);
        for _ in 0..5 { tuner.pass_tokens(0.04, 4); }
        tuner.request();
        assert_eq!(tuner.pending, Some(0));
        while tuner.round.is_some() {
            let seconds = if tuner.level() == 0 { 0.04 } else { 0.024 };
            if let Some(level) = tuner.pass_tokens(seconds, 4) { tuner.set(level); }
        }
        assert_eq!(tuner.level(), 1);
    }

    #[test]
    fn acceptance_and_end_clipping_credit_outputs_not_attempted_rows() {
        let mut tuner = Tuner::new_cuda(vec![0, 1], 0);
        tuner.request();
        tuner.pass_tokens(0.04, 4); // warm-up
        // A four-row verification rejects drafts and emits just its bonus token.
        tuner.pass_tokens(0.04, 1);
        assert_eq!(tuner.round.as_ref().unwrap().useful_tokens, 1);
        // END as the first emitted token leaves emitted.len()==0, although kept>=1.
        tuner.pass_tokens(0.04, 0);
        assert_eq!(tuner.round.as_ref().unwrap().useful_tokens, 1);
        assert_eq!(tuner.round.as_ref().unwrap().token_samples, [(0.04, 1)]);
    }
}

#[cfg(test)]
mod loss_streak_tests {
    use super::*;

    fn timing(level: u32, seconds: f64, error: f64) -> Timing { Timing { level, seconds, error } }

    /// Inject completed windows with recorded means/errors, preserving the actual
    /// round decision, last-probe level and GPU-setter acknowledgement protocol.
    fn completed(tuner: &mut Tuner, measured: Vec<Timing>) -> (Outcome, Option<u32>) {
        tuner.tokens(INTERVAL);
        tuner.request();
        let last = *tuner.round.as_ref().unwrap().order.last().unwrap();
        tuner.set(last);
        let round = tuner.round.as_mut().unwrap();
        round.at = round.order.len();
        round.measured = measured;
        let next = tuner.advance();
        if let Some(level) = next { tuner.set(level); }
        (tuner.outcome().unwrap(), next)
    }

    #[test]
    fn recorded_alternating_fastest_rounds_confirm_held_loss_and_choose_four() {
        // These are three completed windows' exact recorded means/errors. They
        // test the decision policy; they are not measurements of this prototype.
        let rounds = [
            vec![timing(0, 0.014361747535714285, 0.0005829069859352363),
                 timing(4, 0.010575186285714286, 8.745716388204236e-5),
                 timing(8, 0.010661454285714285, 7.272802153087838e-5)],
            vec![timing(0, 0.01112871267857143, 0.0001481242377349802),
                 timing(4, 0.010361437428571428, 3.081418711017045e-5),
                 timing(8, 0.010344267142857144, 2.9328321803158852e-5)],
            vec![timing(0, 0.010958251285714284, 0.0001228208746137582),
                 timing(4, 0.010432022928571427, 5.064796576374122e-5),
                 timing(8, 0.010473963142857146, 5.375316553406413e-5)],
        ];
        let fastest: Vec<_> = rounds.iter().map(|round| round.iter().min_by(|a,b| a.seconds.total_cmp(&b.seconds)).unwrap().level).collect();
        assert_eq!(fastest, [4, 8, 4]);
        for round in &rounds { assert_eq!(Tuner::qualified_cuda(round, 0), [4, 8]); }
        let mut tuner = Tuner::new_cuda(vec![0, 4, 8], 0);
        let (first, restore) = completed(&mut tuner, rounds[0].clone());
        assert_eq!((first.chosen, restore), (0, Some(0)));
        assert_eq!(tuner.pending, Some(0), "a successful first held restore preserves its loss streak");
        assert_eq!(completed(&mut tuner, rounds[1].clone()).0.chosen, 4);
        assert_eq!(tuner.pending, None);
        assert_eq!(completed(&mut tuner, rounds[2].clone()).0.chosen, 4);
    }

    #[test]
    fn the_second_round_can_qualify_only_eight() {
        let mut tuner = Tuner::new_cuda(vec![0, 4, 8], 0);
        completed(&mut tuner, vec![timing(0, 1.0, 0.0), timing(4, 0.8, 0.0), timing(8, 0.99, 0.0)]);
        assert_eq!(tuner.pending, Some(0));
        assert_eq!(completed(&mut tuner, vec![timing(0, 1.0, 0.0), timing(4, 0.99, 0.0), timing(8, 0.8, 0.0)]).0.chosen, 8);
    }

    #[test]
    fn the_smallest_qualified_level_wins_even_with_a_large_speed_gap() {
        let mut tuner = Tuner::new_cuda(vec![0, 4, 8], 0);
        let measured = vec![timing(0, 1.0, 0.0), timing(8, 0.1, 0.0), timing(4, 0.9, 0.0)];
        assert_eq!(Tuner::qualified_cuda(&measured, 0), [4, 8]);
        assert_eq!(completed(&mut tuner, measured.clone()).0.chosen, 0);
        assert_eq!(completed(&mut tuner, measured).0.chosen, 4);
    }

    #[test]
    fn no_significant_gain_missing_held_or_invalid_vote_breaks_the_loss_streak() {
        let winner = vec![timing(0, 1.0, 0.0), timing(4, 0.8, 0.0), timing(8, 0.9, 0.0)];
        for invalid in [
            vec![timing(0, 1.0, 0.0), timing(4, 0.9, 0.06), timing(8, 0.99, 0.0)],
            vec![timing(4, 0.8, 0.0), timing(8, 0.9, 0.0)],
            vec![timing(0, 1.0, 0.0), timing(4, 0.8, f64::INFINITY), timing(8, 0.9, 0.0)],
            vec![timing(0, 1.0, 0.0), timing(4, f64::NAN, 0.0), timing(8, 0.9, 0.0)],
            vec![timing(0, 1.0, 0.0), timing(4, 0.8, -0.01), timing(8, 0.9, 0.0)],
        ] {
            let mut tuner = Tuner::new_cuda(vec![0, 4, 8], 0);
            completed(&mut tuner, winner.clone());
            assert_eq!(completed(&mut tuner, invalid).0.chosen, 0);
            assert_eq!(tuner.pending, None);
            assert_eq!(completed(&mut tuner, winner.clone()).0.chosen, 0);
        }
    }

    #[test]
    fn an_external_held_change_or_explicit_restore_breaks_the_loss_streak() {
        let mut tuner = Tuner::new_cuda(vec![0, 4, 8], 0);
        completed(&mut tuner, vec![timing(0, 1.0, 0.0), timing(4, 0.8, 0.0), timing(8, 0.9, 0.0)]);
        tuner.set(4);
        assert_eq!(tuner.pending, None);
        assert_eq!(completed(&mut tuner, vec![timing(0, 1.0, 0.0), timing(4, 0.9, 0.0), timing(8, 0.7, 0.0)]).0.chosen, 4);
        assert_eq!(tuner.pending, Some(4));
        assert_eq!(tuner.cannot_set(8), Some(4));
        assert_eq!(tuner.pending, None);
        tuner.set(4);
        assert_eq!(tuner.pending, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model's part of a round: each pass takes what `times` says of the level in force, and the level asked for is set (a pass
    /// right after a change takes `fresh` more: the graphs are captured again). Returns the passes the round took and the levels set,
    /// in order.
    fn run(tuner: &mut Tuner, times: &dyn Fn(u32) -> f64, fresh: f64) -> (usize, Vec<u32>) {
        run_with(tuner, &|level, _| times(level), fresh)
    }

    /// [`run`] with the passes' times varying: `times` gets the level and the number of the pass at it (0: the one not timed).
    fn run_with(tuner: &mut Tuner, times: &dyn Fn(u32, usize) -> f64, fresh: f64) -> (usize, Vec<u32>) {
        let (mut passes, mut set, mut extra, mut nth) = (0, Vec::new(), fresh, 0);
        while tuner.probing() {
            passes += 1;
            assert!(passes < 1000, "the round never ends");
            let seconds = times(tuner.level(), nth) + std::mem::take(&mut extra);
            nth += 1;
            if let Some(level) = tuner.pass(seconds) {
                tuner.set(level);
                set.push(level);
                extra = fresh;
                nth = 0;
            }
        }
        (passes, set)
    }

    #[test]
    fn the_levels_of_each_build() {
        assert_eq!(levels(2), [0, 1, 2]);
        assert_eq!(levels(4), [0, 2, 4]);
        assert_eq!(levels(1), [0, 1]);
        assert_eq!(levels(5), [0, 2, 4, 5]);
        assert_eq!(levels(0), [0]);
    }

    #[test]
    fn a_round_times_each_level_in_its_window() {
        let mut tuner = Tuner::new(vec![0, 1, 2], 2);
        assert!(tuner.active() && !tuner.probing());
        tuner.request();
        assert!(tuner.probing());
        // the level in force first, then the others up: 0 is the fastest by 2 of 62 ms, 1 by one
        let (passes, set) = run(&mut tuner, &|level| 0.060 + 0.001 * f64::from(level), 0.0);
        assert_eq!(passes, round_passes(3));
        // (the last window was 1's: 0 is asked for once all are timed)
        assert_eq!(set, [0, 1, 0]);
        let outcome = tuner.outcome().unwrap();
        assert_eq!((outcome.first, outcome.from, outcome.chosen), (true, 2, 0));
        assert_eq!(outcome.measured.iter().map(|timing| timing.level).collect::<Vec<_>>(), [0, 1, 2]);
        assert!(!tuner.probing() && tuner.level() == 0);
        assert!(tuner.outcome().is_none());
    }

    #[test]
    fn the_level_in_force_stays_below_the_gain() {
        // 1.9% faster at 0 and at 1: not enough to leave 2
        let mut tuner = Tuner::new(vec![0, 1, 2], 2);
        tuner.request();
        let (_, set) = run(&mut tuner, &|level| if level == 2 { 0.1000 } else { 0.1000 * (1.0 - 0.019) }, 0.0);
        // (each window's level was set, and the one in force comes back at the end)
        assert_eq!(set, [0, 1, 2]);
        assert_eq!(tuner.level(), 2);
        let outcome = tuner.outcome().unwrap();
        assert_eq!((outcome.from, outcome.chosen), (2, 2));
        assert!(outcome.first && outcome.worth_saying());
    }

    #[test]
    fn a_level_faster_by_the_gain_replaces_it_and_the_fastest_wins() {
        let mut tuner = Tuner::new(vec![0, 1, 2], 2);
        tuner.request();
        run(&mut tuner, &|level| match level {
            2 => 0.1000,
            1 => 0.1000 * (1.0 - 0.021),
            _ => 0.1000 * (1.0 - 0.10),
        }, 0.0);
        assert_eq!(tuner.level(), 0);
        let outcome = tuner.outcome().unwrap();
        assert_eq!((outcome.from, outcome.chosen), (2, 0));
        assert!(outcome.worth_saying());
        // only 1 is faster by the gain
        let mut tuner = Tuner::new(vec![0, 1, 2], 2);
        tuner.request();
        run(&mut tuner, &|level| if level == 1 { 0.1000 * (1.0 - 0.021) } else { 0.1000 }, 0.0);
        assert_eq!(tuner.level(), 1);
    }

    #[test]
    fn a_round_comes_every_interval_of_decode_tokens() {
        let mut tuner = Tuner::new(vec![0, 1, 2], 1);
        tuner.request();
        run(&mut tuner, &|_| 0.05, 0.0);
        tuner.outcome();
        // tokens in a round are not counted, and after it 2,047 are not enough: the 2,048th is
        tuner.request();
        assert!(!tuner.probing());
        tuner.tokens(1000);
        tuner.tokens(1047);
        assert!(!tuner.probing());
        tuner.tokens(1);
        assert!(tuner.probing());
        tuner.tokens(5 * INTERVAL);
        run(&mut tuner, &|_| 0.05, 0.0);
        let outcome = tuner.outcome().unwrap();
        // (the second round, which changed nothing: not worth a line)
        assert!(!outcome.first && !outcome.worth_saying());
        tuner.tokens(INTERVAL - 1);
        assert!(!tuner.probing());
    }

    #[test]
    fn a_request_after_more_than_the_interval_begins_a_round() {
        let mut tuner = Tuner::new(vec![0, 1], 1);
        tuner.since = INTERVAL;
        tuner.request();
        assert!(!tuner.probing());
        tuner.since = INTERVAL + 1;
        tuner.request();
        assert!(tuner.probing());
        // the first request of a model's life is one
        let mut fresh = Tuner::new(vec![0, 1], 1);
        fresh.request();
        assert!(fresh.probing());
    }

    #[test]
    fn a_layer_locked_drops_the_round_under_way_and_makes_another_due() {
        let mut tuner = Tuner::new(vec![0, 1, 2], 2);
        tuner.request();
        assert!(tuner.probing());
        // (a window of the round in force, then a layer is locked)
        for _ in 0..HELD_WINDOW {
            tuner.pass(0.050);
        }
        tuner.restart();
        assert!(!tuner.probing());
        assert_eq!(tuner.level(), 2, "the level in force stays");
        // the next request begins a round at once, however few tokens have gone
        tuner.request();
        assert!(tuner.probing());
        // and a model with none under way is due as well
        let mut idle = Tuner::new(vec![0, 1], 1);
        idle.request();
        idle.round = None;
        idle.since = 0;
        idle.restart();
        idle.request();
        assert!(idle.probing());
    }

    #[test]
    fn a_stall_in_a_window_does_not_move_it() {
        // 5 s among 50 ms passes: the trimmed mean reads 50 ms, as does the one of a window with three
        let mut with = vec![0.050; HELD_WINDOW];
        with[7] = 5.0;
        assert!((trimmed_mean(&with).0 - 0.050).abs() < 1e-12);
        with[8] = 3.0;
        with[30] = 2.0;
        assert!((trimmed_mean(&with).0 - 0.050).abs() < 1e-12);
        let mut other = vec![0.050; OTHER_WINDOW];
        other[3] = 5.0;
        assert!((trimmed_mean(&other).0 - 0.050).abs() < 1e-12);
        // a stall in the level in force's window does not make 0 look 2% faster
        let mut tuner = Tuner::new(vec![0, 1], 1);
        tuner.request();
        let mut asked = 0;
        for pass in 0..round_passes(2) {
            if let Some(level) = tuner.pass(if pass == 8 { 5.0 } else { 0.050 }) {
                tuner.set(level);
                asked += 1;
            }
        }
        // (0 for its window, and 1 back: nothing changed)
        assert_eq!(asked, 2);
        assert_eq!(tuner.level(), 1);
    }

    #[test]
    fn the_first_pass_after_a_change_is_not_timed() {
        let mut tuner = Tuner::new(vec![0, 1, 2], 1);
        tuner.request();
        run(&mut tuner, &|_| 0.1, 0.8);
        for timing in tuner.outcome().unwrap().measured {
            assert!((timing.seconds - 0.1).abs() < 1e-12, "{timing:?}");
        }
    }

    #[test]
    fn a_level_that_cannot_be_set_is_left_out() {
        // 1 cannot be set: the round goes on, 0 is the best of what is left
        let mut tuner = Tuner::new(vec![0, 1, 2], 2);
        tuner.request();
        let mut asked = Vec::new();
        while tuner.probing() {
            let mut next = tuner.pass(if tuner.level() == 0 { 0.050 } else { 0.1 });
            while let Some(level) = next.take() {
                asked.push(level);
                if level == 1 {
                    next = tuner.cannot_set(level);
                } else {
                    tuner.set(level);
                }
            }
        }
        assert_eq!(asked, [0, 1]);
        assert_eq!(tuner.level(), 0);
        let outcome = tuner.outcome().unwrap();
        assert_eq!(outcome.measured.iter().map(|timing| timing.level).collect::<Vec<_>>(), [0, 2]);
        // not asked for again in the rounds after it
        tuner.tokens(INTERVAL);
        assert!(tuner.probing());
        let (_, set) = run(&mut tuner, &|_| 0.05, 0.0);
        assert_eq!(set, [2, 0]);
    }

    #[test]
    fn a_choice_that_cannot_be_set_gives_way_to_the_best_of_the_rest() {
        // 0 is the fastest, then 1; 2 is the level in force, and the last window is 1's
        let times = |level: u32| match level {
            2 => 0.100,
            1 => 0.080,
            _ => 0.060,
        };
        let mut tuner = Tuner::new(vec![0, 1, 2], 2);
        tuner.request();
        let mut asked = Vec::new();
        while tuner.probing() {
            if let Some(level) = tuner.pass(times(tuner.level())) {
                asked.push(level);
                if level == 0 && tuner.level() == 1 {
                    // (0 cannot be had, once the round is over: 1 is the best left and is in force)
                    assert_eq!(tuner.cannot_set(0), None);
                } else {
                    tuner.set(level);
                }
            }
        }
        assert_eq!(asked, [0, 1, 0]);
        assert_eq!(tuner.level(), 1);
        let outcome = tuner.outcome().unwrap();
        assert_eq!((outcome.from, outcome.chosen), (2, 1));
        assert_eq!(outcome.measured.iter().map(|timing| timing.level).collect::<Vec<_>>(), [1, 2]);
    }

    #[test]
    fn one_level_is_never_probed() {
        let mut tuner = Tuner::new(vec![0], 0);
        assert!(!tuner.active());
        tuner.request();
        tuner.tokens(10 * INTERVAL);
        assert!(!tuner.probing());
        assert_eq!(tuner.pass(0.05), None);
        // all but the one in force lost to refusals: the same from then on
        let mut tuner = Tuner::new(vec![0, 1], 1);
        tuner.request();
        let asked = (0..round_passes(2)).find_map(|_| tuner.pass(0.05));
        assert_eq!(asked, Some(0));
        assert_eq!(tuner.cannot_set(0), None);
        assert!(!tuner.active() && !tuner.probing());
        assert_eq!(tuner.level(), 1);
        tuner.tokens(10 * INTERVAL);
        tuner.request();
        assert!(!tuner.probing());
    }

    #[test]
    fn the_error_of_a_trimmed_mean() {
        // 1 to 8: the slowest eighth (the 8) goes, leaving 1 to 7 with a mean of 4, a sample variance of 28/6 and an error of its root over 7
        let (mean, error) = trimmed_mean(&[3.0, 1.0, 2.0, 8.0, 5.0, 4.0, 6.0, 7.0]);
        assert!((mean - 4.0).abs() < 1e-12);
        assert!((error - (28.0f64 / 6.0 / 7.0).sqrt()).abs() < 1e-12, "{error}");
        assert_eq!(trimmed_mean(&[0.05; 16]).1, 0.0);
    }

    /// A window's passes: `seconds` with the swing of `noise` (a part of it) up and down in turn, and a little else between so that no two
    /// windows are alike.
    fn noisy(seconds: f64, noise: f64, nth: usize) -> f64 {
        seconds * (1.0 + noise * [1.0, -1.0, 0.5, -0.5][nth % 4])
    }

    /// A round from the level in force, 2, at `held` seconds, against 0 at `faster` times that, with passes swinging by `noise`; the level
    /// in force afterwards.
    fn after_a_round(faster: f64, noise: f64) -> (u32, Outcome) {
        let mut tuner = Tuner::new(vec![0, 2], 2);
        tuner.request();
        run_with(&mut tuner, &|level, nth| noisy(if level == 2 { 0.100 } else { 0.100 * faster }, noise, nth), 0.0);
        (tuner.level(), tuner.outcome().unwrap())
    }

    #[test]
    fn a_gain_within_the_noise_leaves_the_level_in_force() {
        // 3% faster by the means, with passes swinging by 12%: the windows cannot tell the two apart
        let (level, outcome) = after_a_round(0.97, 0.12);
        assert_eq!(level, 2, "{outcome:?}");
        assert_eq!(outcome.chosen, 2);
        let (held, other) = (outcome.measured[1], outcome.measured[0]);
        assert!(other.seconds < held.seconds * (1.0 - GAIN), "the gain is there in the means: {outcome:?}");
        assert!(held.seconds - other.seconds < CONFIDENCE * held.error.hypot(other.error), "and within the noise: {outcome:?}");
        // the same 3% from clean passes is a gain
        let (level, outcome) = after_a_round(0.97, 0.002);
        assert_eq!(level, 0, "{outcome:?}");
        assert!(outcome.worth_saying());
    }

    #[test]
    fn a_gain_beyond_the_noise_replaces_the_level_in_force() {
        // 20% faster against swings of 12%: far beyond twice the error of a difference
        let (level, outcome) = after_a_round(0.80, 0.12);
        assert_eq!(level, 0, "{outcome:?}");
        // and a level slower by the same is left
        let (level, _) = after_a_round(1.20, 0.12);
        assert_eq!(level, 2);
    }

    #[test]
    fn the_noisier_window_needs_the_bigger_gain() {
        // the same 4% gain: told apart in passes swinging by 1%, not in passes swinging by 15%
        assert_eq!(after_a_round(0.96, 0.01).0, 0);
        assert_eq!(after_a_round(0.96, 0.15).0, 2);
    }

    #[test]
    fn a_round_costs_a_few_percent_of_the_tokens() {
        // two levels beside the one in force
        assert_eq!(ROUND_PASSES, 67);
        assert!((ROUND_PASSES as f64) / (INTERVAL as f64) < 0.035);
        // (the passes at the levels not in force: 34 of them, 1.7%)
        assert_eq!(2 * (OTHER_WINDOW + 1), 34);
    }

    #[test]
    fn the_line_names_each_level_and_the_one_kept() {
        let timing = |level, seconds, error| Timing { level, seconds, error };
        let kept = Outcome { first: true, from: 1, chosen: 1, measured: vec![timing(0, 0.0612, 0.0015), timing(1, 0.0580, 0.0009), timing(2, 0.0635, 0.0020)] };
        assert_eq!(kept.line(), "Straydog assist: 0 a layer 61.2±1.5 ms a pass, 1: 58.0±0.9, 2: 63.5±2.0; keeping 1");
        let moved = Outcome { first: false, from: 2, chosen: 0, measured: vec![timing(0, 0.05, 0.0004), timing(2, 0.06, 0.0)] };
        assert_eq!(moved.line(), "Straydog assist: 0 a layer 50.0±0.4 ms a pass, 2: 60.0±0.0; now 0 (was 2)");
        assert!(moved.worth_saying() && kept.worth_saying());
        assert!(!Outcome { first: false, ..kept }.worth_saying());
    }
}

#[cfg(test)]
mod initial_confirmation_tests {
    use super::*;

    fn tuner(interval: u64) -> Tuner {
        Tuner::new_cuda_with_confirm_tokens(vec![0, 4, 8], 0, interval)
    }

    fn sample(tuner: &mut Tuner, wins: bool) -> Option<u32> {
        let held = tuner.round.as_ref().unwrap().held;
        let rate = if !wins || tuner.level() == held { 0.010 } else { 0.007 };
        tuner.pass_tokens(rate * 4.0, 4)
    }

    fn finish(tuner: &mut Tuner, wins: bool) -> Outcome {
        assert!(tuner.probing());
        for _ in 0..100 {
            if let Some(level) = sample(tuner, wins) { tuner.set(level); }
            if tuner.round.is_none() { return tuner.outcome().unwrap(); }
        }
        panic!("round did not finish");
    }

    fn waiting_for_first_restore(tuner: &mut Tuner) {
        tuner.request();
        for _ in 0..100 {
            if let Some(level) = sample(tuner, true) {
                if tuner.restoring(level) {
                    assert_eq!(level, 0);
                    return;
                }
                tuner.set(level);
            }
        }
        panic!("held restoration was not requested");
    }

    #[test]
    fn confirmation_setting_defaults_bounds_and_disabled_baseline() {
        for (value, expected) in [(None, 256), (Some("256"), 256), (Some("512"), 512),
            (Some("0"), 2048), (Some("2048"), 2048), (Some("1"), 1), (Some(" 512 "), 512),
            (Some("2049"), 256), (Some("-1"), 256), (Some(""), 256), (Some("junk"), 256),
            (Some("18446744073709551616"), 256)] {
            assert_eq!(confirmation_tokens_named(value), expected, "{value:?}");
        }
    }

    #[test]
    fn only_the_first_confirmation_wait_uses_256_512_or_disabled_2048() {
        for (setting, interval) in [("256", 256), ("512", 512), ("0", 2048), ("2048", 2048)] {
            let mut tuner = tuner(confirmation_tokens_named(Some(setting)));
            tuner.request();
            assert_eq!(finish(&mut tuner, true).chosen, 0);
            assert_eq!((tuner.since, tuner.next_interval, tuner.pending), (0, interval, Some(0)));
            tuner.tokens(interval - 1);
            tuner.request();
            assert!(!tuner.probing(), "one token before {interval}");
            assert_eq!(tuner.since, interval - 1);
            tuner.tokens(1);
            assert!(tuner.probing());
            assert_eq!(tuner.next_interval, 2048, "the shortened wait was consumed at begin");
            assert_eq!(finish(&mut tuner, true).chosen, 4);
            assert_eq!((tuner.since, tuner.next_interval), (0, 2048));
            tuner.tokens(2047);
            tuner.request();
            assert!(!tuner.probing());
            tuner.tokens(1);
            assert!(tuner.probing(), "the next steady round keeps the 2048-token rule");
        }
    }

    #[test]
    fn a_first_valid_round_without_a_gain_never_offers_a_later_short_wait() {
        let mut tuner = tuner(256);
        tuner.request();
        assert_eq!(finish(&mut tuner, false).chosen, 0);
        assert_eq!(tuner.initial_confirmation, InitialConfirmation::Consumed);
        for _ in 0..2 {
            tuner.tokens(2048);
            finish(&mut tuner, true);
            assert_eq!(tuner.next_interval, 2048);
        }
    }

    #[test]
    fn an_initial_nonzero_held_loss_consumes_eligibility_without_shortening() {
        let mut tuner = Tuner::new_cuda_with_confirm_tokens(vec![0, 4, 8], 4, 256);
        tuner.request();
        assert_eq!(finish(&mut tuner, true).chosen, 4);
        assert_eq!(tuner.pending, Some(4));
        assert_eq!(tuner.initial_confirmation, InitialConfirmation::Consumed);
        assert_eq!(tuner.next_interval, 2048);
        tuner.tokens(255);
        assert!(!tuner.probing());
        tuner.tokens(1792);
        assert!(!tuner.probing());
        tuner.tokens(1);
        assert!(tuner.probing());
    }

    #[test]
    fn held_ack_arms_once_and_the_final_sample_never_ticks_the_outside_interval() {
        let mut tuner = tuner(256);
        waiting_for_first_restore(&mut tuner);
        assert_eq!(tuner.initial_confirmation, InitialConfirmation::AwaitingHeld(0));
        assert_eq!(tuner.next_interval, 2048);
        let since = tuner.since;
        for _ in 0..3 {
            assert_eq!(tuner.advance(), Some(0));
            assert_eq!(tuner.pass_tokens(0.01, 4), None);
            tuner.tokens(1000);
            assert_eq!(tuner.next_interval, 2048);
            assert_eq!(tuner.since, since);
        }
        tuner.set(0);
        assert_eq!((tuner.rounds, tuner.since, tuner.next_interval), (1, 0, 256));
        assert_eq!(tuner.initial_confirmation, InitialConfirmation::Consumed);
        tuner.tokens(255);
        assert!(!tuner.probing(), "the final measured pass did not supply outside credit");
        tuner.tokens(1);
        assert!(tuner.probing());
    }

    fn complete_injected(tuner: &mut Tuner, measured: Vec<Timing>) {
        tuner.request();
        let round = tuner.round.as_mut().unwrap();
        round.at = round.order.len() - 1;
        let last = round.order[round.at];
        tuner.set(last);
        let round = tuner.round.as_mut().unwrap();
        round.at = round.order.len();
        round.measured = measured;
        if let Some(level) = tuner.advance() { tuner.set(level); }
        assert!(tuner.round.is_none());
        tuner.outcome();
    }

    #[test]
    fn invalid_or_incomplete_votes_do_not_consume_the_first_valid_opportunity() {
        let timing = |level, seconds, error| Timing { level, seconds, error };
        let invalid = [
            vec![timing(4, 0.007, 0.0), timing(8, 0.007, 0.0)],
            vec![timing(0, 0.010, 0.0), timing(4, 0.007, 0.0)],
            vec![timing(0, 0.010, 0.0), timing(4, 0.007, 0.0), timing(4, 0.007, 0.0)],
            vec![timing(0, 0.010, 0.0), timing(4, f64::NAN, 0.0), timing(8, 0.007, 0.0)],
            vec![timing(0, 0.010, f64::INFINITY), timing(4, 0.007, 0.0), timing(8, 0.007, 0.0)],
            vec![timing(0, 0.010, 0.0), timing(4, 0.007, -1.0), timing(8, 0.007, 0.0)],
        ];
        for vote in invalid {
            let mut tuner = tuner(256);
            complete_injected(&mut tuner, vote);
            assert_eq!(tuner.initial_confirmation, InitialConfirmation::Eligible);
            assert_eq!(tuner.pending, None, "an incomplete vote must not start a loss streak");
            assert_eq!(tuner.next_interval, 2048);
            tuner.tokens(2048);
            finish(&mut tuner, true);
            assert_eq!((tuner.rounds, tuner.next_interval), (2, 256), "validity is independent of round numbering");
        }
    }

    #[test]
    fn restart_before_the_first_valid_round_preserves_eligibility_through_real_restore() {
        let mut tuner = tuner(256);
        tuner.request();
        loop {
            if let Some(level) = sample(&mut tuner, true) {
                tuner.set(level);
                assert_eq!(level, 4);
                break;
            }
        }
        assert_eq!(tuner.restart(), Some(0));
        assert_eq!(tuner.initial_confirmation, InitialConfirmation::Eligible);
        tuner.request();
        assert!(!tuner.probing(), "GPU restoration is still pending");
        tuner.set(0);
        assert_eq!(tuner.initial_confirmation, InitialConfirmation::Eligible);
        tuner.request();
        assert!(tuner.probing(), "restart remains due at the next request");
        finish(&mut tuner, true);
        assert_eq!(tuner.next_interval, 256);
    }

    #[test]
    fn restart_after_the_first_valid_vote_cancels_both_pending_and_armed_short_waits() {
        for ack in [false, true] {
            let mut tuner = tuner(256);
            waiting_for_first_restore(&mut tuner);
            if ack { tuner.set(0); tuner.outcome(); }
            let restore = tuner.restart();
            assert_eq!(restore, (!ack).then_some(0));
            assert_eq!(tuner.initial_confirmation, InitialConfirmation::Consumed);
            assert_eq!(tuner.next_interval, 2048);
            if let Some(level) = restore { tuner.set(level); }
            tuner.request();
            finish(&mut tuner, true);
            assert_eq!(tuner.next_interval, 2048, "the opportunity is never rearmed after restart");
        }
    }

    #[test]
    fn a_failed_setter_or_failed_held_restore_permanently_cancels_shortening() {
        for stage in 0..3 {
            let mut tuner = tuner(256);
            if stage > 0 { waiting_for_first_restore(&mut tuner); }
            if stage == 2 { tuner.set(0); tuner.outcome(); }
            assert_eq!(tuner.cannot_set(if stage == 1 { 0 } else { 4 }), Some(0));
            assert_eq!(tuner.initial_confirmation, InitialConfirmation::Consumed);
            assert_eq!(tuner.next_interval, 2048);
            tuner.set(0);
            tuner.tokens(2048);
            finish(&mut tuner, true);
            assert_eq!(tuner.next_interval, 2048);
        }
    }

    #[test]
    fn an_actual_external_level_change_cancels_an_armed_wait() {
        let mut tuner = tuner(256);
        tuner.request();
        finish(&mut tuner, true);
        assert_eq!(tuner.next_interval, 256);
        tuner.set(4);
        assert_eq!(tuner.pending, None);
        assert_eq!(tuner.initial_confirmation, InitialConfirmation::Consumed);
        assert_eq!(tuner.next_interval, 2048);
        tuner.tokens(2048);
        finish(&mut tuner, true);
        assert_eq!(tuner.next_interval, 2048);
    }
}
