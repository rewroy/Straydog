//! A partially resident model's assist level, chosen by measuring decode.
//!
//! The assist is the GPU taking a few of a layer's absent routes, which the CPU would otherwise run, from page-locked host banks over
//! PCIe. How many are worth taking depends on the machine: in the part of a layer where the two overlap, the GPU's side (its slot
//! experts and the assisted ones) or the CPU's is the longer, by a margin that its PCIe link, its CPU and its VRAM decide. An RTX 5070 Ti
//! on PCIe 5.0 x16 and an R9700 each gained from the most they were tried at; a PC with an RTX 4060 on PCIe 4.0 x8 may gain from fewer, or
//! from none. Nothing says which from the machine's specification, so the engine times decode at each level and keeps the fastest.
//!
//! A round: a window of decode passes of one row at the level in force, then one at each other level, each timed as the wall time of the
//! pass's layers (the "step: layers" of `decode_profile`). The level in force goes first and has the longer window: it is the level the
//! model would run at anyway, while a window at another level costs the difference of the two for each of its passes. A round spends
//! [`ROUND_PASSES`] passes of every [`INTERVAL`] decode tokens (3% with two levels besides the one in force), most of them at the
//! level in force. A level replaces it only if it is [`GAIN`] faster and the difference is more than [`CONFIDENCE`] standard errors of
//! it: passes vary by a quarter from one to the next, and a window of 16 cannot tell 2% from nothing, so a level that was not told apart
//! from the one in force by its windows does not replace it (the engine never stays at a level on noise). Passes of other rows, and
//! prefill, run at the level in force and are never timed; while a round times, a speculating model leaves its drafts out so that every
//! pass is one of one row.
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

/// A level's window: the mean seconds of a pass's layers (without the slowest eighth) and its standard error.
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
}

struct Round {
    /// The levels to time: the one in force first.
    order: Vec<u32>,
    /// The window under way, and the passes of it still to be passed over.
    at: usize,
    skip: usize,
    samples: Vec<f64>,
    measured: Vec<Timing>,
    held: u32,
    /// Once every window is timed: the level chosen, until it is in force.
    target: Option<u32>,
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
    /// The first winning round's stable level and candidate, awaiting a second round.
    pending: Option<(u32, u32)>,
    /// An invalidated round's held level, until the model actually restores it.
    restore: Option<u32>,
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

impl Tuner {
    /// A tuner of `levels` (the levels that can be set) with `level` in force. It is due at the first request.
    pub fn new(mut levels: Vec<u32>, level: u32) -> Tuner {
        levels.push(level);
        levels.sort_unstable();
        levels.dedup();
        Tuner { levels, level, since: INTERVAL + 1, round: None, rounds: 0, outcome: None,
                cuda: false, pending: None, restore: None }
    }

    /// CUDA's tuner: a candidate must save at least 3%, beyond the same standard-error
    /// threshold, in two consecutive complete rounds against the same held level.
    pub fn new_cuda(levels: Vec<u32>, level: u32) -> Tuner {
        Tuner { cuda: true, ..Self::new(levels, level) }
    }

    /// The level in force.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Whether there is more than one level to choose from.
    pub fn active(&self) -> bool {
        self.levels.len() >= 2
    }

    /// Whether windows are being timed: the passes that matter are those of one row, which the model keeps to while it is so.
    pub fn probing(&self) -> bool {
        self.round.as_ref().is_some_and(|round| round.target.is_none())
    }

    fn begin(&mut self) {
        let mut order = vec![self.level];
        order.extend(self.levels.iter().copied().filter(|level| *level != self.level));
        self.round = Some(Round { order, at: 0, skip: 1, samples: Vec::new(), measured: Vec::new(), held: self.level, target: None });
    }

    /// `count` decode tokens were generated: a round begins every [`INTERVAL`] of them.
    pub fn tokens(&mut self, count: u64) {
        if self.round.is_some() || self.restore.is_some() || !self.active() {
            return;
        }
        self.since = self.since.saturating_add(count);
        if self.since >= INTERVAL {
            self.begin();
        }
    }

    /// A request begins decoding: a round begins if more than [`INTERVAL`] tokens have gone since the last.
    pub fn request(&mut self) {
        if self.round.is_none() && self.restore.is_none() && self.active() && self.since > INTERVAL {
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

    fn choose_cuda(measured: &[Timing], held: u32) -> u32 {
        let Some(&kept) = measured.iter().find(|timing| timing.level == held) else { return held };
        measured.iter().filter(|timing| {
            let saved = kept.seconds - timing.seconds;
            timing.level != held && timing.seconds <= kept.seconds * (1.0 - CUDA_GAIN)
                && saved > CONFIDENCE * kept.error.hypot(timing.error)
        }).min_by(|a, b| a.seconds.total_cmp(&b.seconds)).map_or(held, |best| best.level)
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
            let candidate = Self::choose_cuda(&round.measured, round.held);
            if candidate == round.held {
                self.pending = None;
                round.held
            } else if self.pending == Some((round.held, candidate)) {
                self.pending = None;
                candidate
            } else {
                self.pending = Some((round.held, candidate));
                round.held
            }
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
        let mut measured = round.measured;
        measured.sort_by_key(|timing| timing.level);
        self.outcome = Some(Outcome { first: self.rounds == 1, from: round.held, chosen: self.level, measured });
    }

    /// A decode pass of one row took `seconds` over its layers. The level to put in force, if the pass ended a window.
    pub fn pass(&mut self, seconds: f64) -> Option<u32> {
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
        self.advance()
    }

    /// The layers the GPU assists have changed (one more was page-locked): what a round measured was of the layers before, so one under way is
    /// dropped and a round is due at the next request. CUDA restores the held level if a
    /// probe was in force; the caller must set it on the GPU before acknowledging with set().
    pub fn restart(&mut self) -> Option<u32> {
        if self.cuda {
            self.pending = None;
            self.restore = self.restore.or_else(|| self.round.as_ref().map(|round| round.held).filter(|held| *held != self.level));
        }
        self.round = None;
        self.since = INTERVAL + 1;
        self.restore
    }

    /// The level asked for is in force.
    pub fn set(&mut self, level: u32) {
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
        if self.cuda {
            self.pending = None;
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

    fn finish(tuner: &mut Tuner, times: &dyn Fn(u32, usize) -> f64) -> (Outcome, Vec<u32>) {
        tuner.tokens(INTERVAL);
        tuner.request();
        assert!(tuner.probing());
        let (mut commands, mut nth, mut passes) = (Vec::new(), 0, 0);
        while tuner.round.is_some() {
            passes += 1;
            assert!(passes <= round_passes(3));
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
    fn cuda_needs_two_complete_wins_and_restores_the_first_winner_from_the_last_probe() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        let (first, commands) = finish(&mut tuner, &times);
        assert_eq!(commands, [1, 2, 0], "even when the winning candidate was the last probe, return to the actual held level");
        assert_eq!((first.from, first.chosen), (0, 0));
        assert!(first.first && first.worth_saying());
        assert_eq!(tuner.level(), 0);
        assert_eq!(tuner.pending, Some((0, 2)));
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
        assert_eq!(Tuner::choose_cuda(&[held, candidate(97.00000000000001, 0.0)], 0), 0);
        assert_eq!(Tuner::choose_cuda(&[held, candidate(97.0, 0.0)], 0), 1);
        assert_eq!(Tuner::choose_cuda(&[held, candidate(96.99999999999999, 0.0)], 0), 1);
        assert_eq!(Tuner::choose_cuda(&[held, candidate(97.0, 1.5)], 0), 0, "equality with CONFIDENCE is insufficient");
        assert_eq!(Tuner::choose_cuda(&[held, candidate(97.0, 1.49)], 0), 1);
        assert_eq!(Tuner::choose_cuda(&[candidate(1.0, 0.0)], 0), 0, "a missing held window cannot confirm a gain");
        // The old constructor still has its 2% single-round rule.
        assert_eq!(Tuner::choose(&[held, candidate(97.9, 0.0)], 0), 1);
        assert_eq!(Tuner::choose_cuda(&[held, candidate(97.9, 0.0)], 0), 0);
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
    fn cuda_a_changed_candidate_starts_a_new_streak() {
        let mut tuner = Tuner::new_cuda(vec![0, 1, 2], 0);
        let one = |level, _| if level == 1 { 0.080 } else { 0.100 };
        assert_eq!(finish(&mut tuner, &one).0.chosen, 0);
        assert_eq!(tuner.pending, Some((0, 1)));
        assert_eq!(finish(&mut tuner, &times).0.chosen, 0);
        assert_eq!(tuner.pending, Some((0, 2)));
        assert_eq!(finish(&mut tuner, &times).0.chosen, 2);
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
        let next = (0..=HELD_WINDOW).find_map(|_| tuner.pass(0.100));
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
        for _ in 0..round_passes(3) {
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
        for _ in 0..round_passes(3) {
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
        assert_eq!((0..=HELD_WINDOW).find_map(|_| tuner.pass(0.100)), Some(1));
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
        for _ in 0..round_passes(3) {
            if let Some(level) = tuner.pass(times(tuner.level(), 0)) {
                if tuner.restoring(level) { restore = Some(level); break; }
                tuner.set(level);
            }
        }
        assert_eq!(restore, Some(0));
        for _ in 0..5 {
            assert_eq!(tuner.pass(0.080), None);
            assert_eq!(tuner.advance(), Some(0));
            assert_eq!(tuner.pending, Some((0, 2)));
            assert_eq!(tuner.rounds, 0);
        }
        tuner.set(0);
        assert_eq!(tuner.rounds, 1);
        assert_eq!(tuner.level(), 0);
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
