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
//! level in force. A level replaces it only if it is [`GAIN`] faster. Passes of other rows, and prefill, run at the level in force and
//! are never timed; while a round times, a speculating model leaves its drafts out so that every pass is one of one row.
//!
//! Plain logic: the model feeds it passes and tokens and does what it returns (`Model::assist_apply`).

/// Decode tokens between two rounds, counted outside the rounds.
pub const INTERVAL: u64 = 2048;
/// Passes timed at the level in force, and at each other level.
pub const HELD_WINDOW: usize = 32;
pub const OTHER_WINDOW: usize = 16;
/// The part of the level in force's time that another level must save to replace it.
pub const GAIN: f64 = 0.02;

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

/// What a finished round found.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// The first round of the model's life.
    pub first: bool,
    /// The level in force when the round began, and the one in force after it.
    pub from: u32,
    pub chosen: u32,
    /// Each level timed, with the seconds of a pass's layers there, by level.
    pub measured: Vec<(u32, f64)>,
}

impl Outcome {
    /// Whether the log says so: the first round, and a round that changed the level.
    pub fn worth_saying(&self) -> bool {
        self.first || self.chosen != self.from
    }

    /// The log line: each level's milliseconds a pass of the layers, and the level kept.
    pub fn line(&self) -> String {
        let mut text = String::from("Straydog assist:");
        for (at, (level, seconds)) in self.measured.iter().enumerate() {
            if at == 0 {
                text.push_str(&format!(" {level} a layer {:.1} ms a pass", seconds * 1e3));
            } else {
                text.push_str(&format!(", {level}: {:.1}", seconds * 1e3));
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
    measured: Vec<(u32, f64)>,
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
}

/// The mean of `samples` without the slowest eighth: a stall of the machine in a pass (a page fault's wait, another process's thread
/// on a core) says nothing of its level, and one pass in 16 at ten times the rest moves the plain mean by more than [`GAIN`].
fn trimmed_mean(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let kept = &sorted[..sorted.len() - sorted.len() / 8];
    kept.iter().sum::<f64>() / kept.len() as f64
}

impl Tuner {
    /// A tuner of `levels` (the levels that can be set) with `level` in force. It is due at the first request.
    pub fn new(mut levels: Vec<u32>, level: u32) -> Tuner {
        levels.push(level);
        levels.sort_unstable();
        levels.dedup();
        Tuner { levels, level, since: INTERVAL + 1, round: None, rounds: 0, outcome: None }
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
        if self.round.is_some() || !self.active() {
            return;
        }
        self.since = self.since.saturating_add(count);
        if self.since >= INTERVAL {
            self.begin();
        }
    }

    /// A request begins decoding: a round begins if more than [`INTERVAL`] tokens have gone since the last.
    pub fn request(&mut self) {
        if self.round.is_none() && self.active() && self.since > INTERVAL {
            self.begin();
        }
    }

    /// The level chosen from `measured`: the fastest, if it is [`GAIN`] faster than `held`'s, else `held`.
    fn choose(measured: &[(u32, f64)], held: u32) -> u32 {
        let Some(&(_, held_time)) = measured.iter().find(|(level, _)| *level == held) else {
            return measured.iter().min_by(|a, b| a.1.total_cmp(&b.1)).map_or(held, |best| best.0);
        };
        measured.iter().filter(|(level, seconds)| *level != held && *seconds <= held_time * (1.0 - GAIN))
            .min_by(|a, b| a.1.total_cmp(&b.1)).map_or(held, |best| best.0)
    }

    /// The next window, or the end of the round. The level to put in force, if it is not.
    fn advance(&mut self) -> Option<u32> {
        let round = self.round.as_mut()?;
        if let Some(&level) = round.order.get(round.at) {
            round.skip = 1;
            return (level != self.level).then_some(level);
        }
        let chosen = Self::choose(&round.measured, round.held);
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
        measured.sort_by_key(|(level, _)| *level);
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
        let measured = (round.order[round.at], trimmed_mean(&round.samples));
        round.measured.push(measured);
        round.samples.clear();
        round.at += 1;
        self.advance()
    }

    /// The level asked for is in force.
    pub fn set(&mut self, level: u32) {
        self.level = level;
        if self.round.as_ref().is_some_and(|round| round.target == Some(level)) {
            self.conclude();
        }
    }

    /// The level asked for cannot be set: it is not tried again. The level to put in force instead, if any.
    pub fn cannot_set(&mut self, level: u32) -> Option<u32> {
        self.levels.retain(|other| *other != level);
        let round = self.round.as_mut()?;
        round.measured.retain(|(other, _)| *other != level);
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
mod tests {
    use super::*;

    /// A model's part of a round: each pass takes what `times` says of the level in force, and the level asked for is set (a pass
    /// right after a change takes `fresh` more: the graphs are captured again). Returns the passes the round took and the levels set,
    /// in order.
    fn run(tuner: &mut Tuner, times: &dyn Fn(u32) -> f64, fresh: f64) -> (usize, Vec<u32>) {
        let (mut passes, mut set, mut extra) = (0, Vec::new(), fresh);
        while tuner.probing() {
            passes += 1;
            assert!(passes < 1000, "the round never ends");
            let seconds = times(tuner.level()) + std::mem::take(&mut extra);
            if let Some(level) = tuner.pass(seconds) {
                tuner.set(level);
                set.push(level);
                extra = fresh;
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
        assert_eq!(outcome.measured.iter().map(|(level, _)| *level).collect::<Vec<_>>(), [0, 1, 2]);
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
    fn a_stall_in_a_window_does_not_move_it() {
        // 5 s among 50 ms passes: the trimmed mean reads 50 ms, as does the one of a window with three
        let mut with = vec![0.050; HELD_WINDOW];
        with[7] = 5.0;
        assert!((trimmed_mean(&with) - 0.050).abs() < 1e-12);
        with[8] = 3.0;
        with[30] = 2.0;
        assert!((trimmed_mean(&with) - 0.050).abs() < 1e-12);
        let mut other = vec![0.050; OTHER_WINDOW];
        other[3] = 5.0;
        assert!((trimmed_mean(&other) - 0.050).abs() < 1e-12);
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
        for (_, seconds) in tuner.outcome().unwrap().measured {
            assert!((seconds - 0.1).abs() < 1e-12, "{seconds}");
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
        assert_eq!(outcome.measured.iter().map(|(level, _)| *level).collect::<Vec<_>>(), [0, 2]);
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
        assert_eq!(outcome.measured.iter().map(|(level, _)| *level).collect::<Vec<_>>(), [1, 2]);
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
    fn a_round_costs_a_few_percent_of_the_tokens() {
        // two levels beside the one in force
        assert_eq!(ROUND_PASSES, 67);
        assert!((ROUND_PASSES as f64) / (INTERVAL as f64) < 0.035);
        // (the passes at the levels not in force: 34 of them, 1.7%)
        assert_eq!(2 * (OTHER_WINDOW + 1), 34);
    }

    #[test]
    fn the_line_names_each_level_and_the_one_kept() {
        let kept = Outcome { first: true, from: 1, chosen: 1, measured: vec![(0, 0.0612), (1, 0.0580), (2, 0.0635)] };
        assert_eq!(kept.line(), "Straydog assist: 0 a layer 61.2 ms a pass, 1: 58.0, 2: 63.5; keeping 1");
        let moved = Outcome { first: false, from: 2, chosen: 0, measured: vec![(0, 0.05), (2, 0.06)] };
        assert_eq!(moved.line(), "Straydog assist: 0 a layer 50.0 ms a pass, 2: 60.0; now 0 (was 2)");
        assert!(moved.worth_saying() && kept.worth_saying());
        assert!(!Outcome { first: false, ..kept }.worth_saying());
    }
}
