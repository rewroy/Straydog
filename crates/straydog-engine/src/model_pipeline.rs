//! Pipelined verification passes (STRAYDOG_PIPELINE=1): a partially resident model on two NVIDIA GPUs, greedy requests.
//!
//! A pass today runs its layers on the first GPU, then on the second, and the first waits. Here the first GPU starts the next pass's
//! layers (the "front": layer 0, the PLE layer and the first device's span) while the second finishes this one's (the "back": its
//! span and the head). The next pass's tokens are a guess: this pass's drafts all accepted, and its bonus token as the drafter
//! predicts it (the drafter's chain, fed this pass's drafts). Where the guess is right the next pass has been under way for a whole
//! pass; where it is wrong the first device is told to stop, its states are put back to what this pass kept (its snapshots, and the
//! pass behind it kept its own: two windows, `sd_set_window`), and the ordinary order goes on with the pass's real tokens. A pass's
//! tokens are verified with the one-token arithmetic whatever its rows were, so the tokens are the serial loop's.
//!
//! Two threads: this one drives the second GPU (its span, the head, the drafter, the acceptance) and a worker drives the first GPU.
//! The worker owns the first device while the loop runs (the loop touches only the second device then, and the first once the worker
//! is idle); each device has a CPU engine of its own.

use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

use super::*;
use crate::gpu::{Backend, SpanAbort, SPAN_ABORTED};

/// STRAYDOG_PIPELINE: unset or 1, the pipeline where it applies (a partially resident model on two NVIDIA GPUs: the first runs the next pass's
/// layers while the second has this one's; the assist level held at [`PIPELINE_ASSIST`]); 0, off (the ordinary loop, as on one GPU and on AMD
/// GPUs); `freeze`, the ordinary loop with the slots and the assist level held (the other side of a comparison of the two). On two RTX 3090
/// (IQ3_S, context 262144) the pipeline decodes 146 tok/s where the ordinary loop has 122, a new conversation's code 123-143 against
/// 87-113, reasoning 117-129 against 84-101. STRAYDOG_PIPELINE_DRAFTS: the drafts of a pipelined pass (the width in force
/// by default). STRAYDOG_PIPELINE_FORCE_MISS=N: the guess of the next pass's first token made wrong in every Nth pass (a test). STRAYDOG_PIPELINE_CPU_THREADS: the
/// threads of each device's CPU engine. STRAYDOG_PIPELINE_GATE: the hit rate of the last guesses under which no pass is guessed ahead for a while, whatever the passes cost (the gate chooses by cost, see Gate).
#[derive(Clone, Copy, Debug)]
pub(super) struct PipelineConfig {
    mode: Mode,
    drafts: Option<usize>,
    force_miss: usize,
    cpu_threads: Option<usize>,
    gate: f32,
    assist: Option<u32>,
    /// STRAYDOG_PIPELINE_LONE=0: a pass after a wrong guess waits for its drafts as before (the lone row is sent at once by default).
    lone: bool,
    /// STRAYDOG_PIPELINE_DRAWN_GUESSES=1 (to measure): a prediction of the chain is what the target's keyed draw would give on the drafter's
    /// logits (sampling::guess_keyed), where it is the drafter's most likely token by default. With the model card's sampling for answers
    /// (temperature 0.7, top_p 0.8) the target's cut distribution is mostly one candidate and the two guessed alike (80.5% against
    /// 80.3% to 82.1% right); a reasoning request's sampling (temperature 1.0, top_p 0.95) draws past the most likely token far more often.
    keyed_guesses: bool,
    /// STRAYDOG_PIPELINE_SWAPS=held: the expert slots are held as they are while passes are run ahead (to measure; by default they swap as
    /// in the ordinary loop: a new conversation's code was 128-143 tok/s with the swaps and 110-119 held, the steady state 146 and 143).
    free_slots: bool,
    /// STRAYDOG_PIPELINE_RECOVER=0: a pass given up behind one whose only miss was its bonus is put back by the forced replay, not by the snapshots.
    recover: bool,
    /// STRAYDOG_PIPELINE_MTP_BATCH=1: the drafts of the pass behind join the real pairs' catch-up in one batch of the drafter, with the hidden
    /// states the drafter made them with (by default they are fed one step at a time after it: the batch saves a call, 0.25 ms of the chain
    /// on 2 x RTX 4090, but the guess of the bonus token is right in 0.71-0.73 of the full windows against 0.80-0.84).
    batch: bool,
    /// STRAYDOG_PIPELINE_HANDOVER=0: where the gate settles in the ordinary order the pipeline's loop runs it itself (as before), instead of
    /// handing the request to the ordinary loop for a stretch (its chained passes, copy drafts, assist tuner and swaps) and taking it back.
    handover: bool,
    /// STRAYDOG_PIPELINE_HANDOVER=hold (a test's): the stretches of the ordinary loop keep what the pipeline holds (the slots, the assist level, the
    /// head on the second GPU alone), so that a request that is handed over is the same arithmetic as one that never leaves the ordinary loop (`freeze`).
    hand_held: bool,
    /// STRAYDOG_PIPELINE_HANDOVER=start (a test's, holding what the pipeline holds): the request is handed to the ordinary loop at the first drain
    /// and kept there (it must be the one that never left it); =zero: handed over and taken back at every drain with no pass between (it must be the
    /// pipelined one).
    hand_test: u8,
    /// STRAYDOG_PIPELINE_NO_COPY=1 (a test's): the ordinary loop copies no drafts from the sequence and makes no lone row to measure what one costs
    /// (the pipelined passes do neither), so that two runs make the same passes.
    no_copy: bool,
    /// STRAYDOG_PIPELINE_PASS_LOG=1 (a test's): each verified pass's position, tokens, accepted drafts and next token on a line, in both loops.
    pass_log: bool,
    /// STRAYDOG_PIPELINE_HEAD_SHARE=1 lets the drafter's head borrow an idle first device.
    head_share: bool,
}

fn head_share_named(value: Option<&str>) -> bool {
    value == Some("1")
}

fn head_share_eligible(enabled: bool, idle: bool, lower_head: bool, extras: u64, read_rows: u64) -> bool {
    enabled && idle && lower_head && extras == 0 && read_rows >= 2
}

/// The distinct missed experts of a layer's pass that a pipelined request stages to the GPU (the most a CUDA device takes). New
/// conversations on two RTX 3090: code 126-134 tok/s at 8 and 110-119 at 4, reasoning 88-94 and 80-82; the steady state 143 and 138.
const PIPELINE_ASSIST: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Off,
    Frozen,
    On,
}

impl PipelineConfig {
    pub(super) fn from_env() -> PipelineConfig {
        let number = |name: &str| std::env::var(name).ok().and_then(|value| value.parse::<usize>().ok());
        let mode = Self::mode_named(std::env::var("STRAYDOG_PIPELINE").ok().as_deref());
        PipelineConfig { mode, drafts: number("STRAYDOG_PIPELINE_DRAFTS"), force_miss: number("STRAYDOG_PIPELINE_FORCE_MISS").unwrap_or(0),
                         cpu_threads: number("STRAYDOG_PIPELINE_CPU_THREADS").filter(|threads| (1..=256).contains(threads)),
                         gate: std::env::var("STRAYDOG_PIPELINE_GATE").ok().and_then(|value| value.parse::<f32>().ok()).unwrap_or(0.3),
                         assist: std::env::var("STRAYDOG_PIPELINE_ASSIST").ok().and_then(|value| value.parse::<u32>().ok()),
                         lone: std::env::var("STRAYDOG_PIPELINE_LONE").map_or(true, |value| value != "0"),
                         free_slots: std::env::var("STRAYDOG_PIPELINE_SWAPS").map_or(true, |value| value != "held"),
                         recover: std::env::var("STRAYDOG_PIPELINE_RECOVER").map_or(true, |value| value.trim() != "0"),
                         batch: std::env::var("STRAYDOG_PIPELINE_MTP_BATCH").is_ok_and(|value| value == "1"),
                         handover: std::env::var("STRAYDOG_PIPELINE_HANDOVER").map_or(true, |value| value != "0"),
                         hand_held: std::env::var("STRAYDOG_PIPELINE_HANDOVER").is_ok_and(|value| ["hold", "start", "zero"].contains(&value.as_str())),
                         hand_test: match std::env::var("STRAYDOG_PIPELINE_HANDOVER").as_deref() { Ok("start") => 1, Ok("zero") => 2, _ => 0 },
                         no_copy: std::env::var("STRAYDOG_PIPELINE_NO_COPY").is_ok_and(|value| value == "1"),
                         pass_log: std::env::var("STRAYDOG_PIPELINE_PASS_LOG").is_ok_and(|value| value == "1"),
                         head_share: head_share_named(std::env::var("STRAYDOG_PIPELINE_HEAD_SHARE").ok().as_deref()),
                         keyed_guesses: std::env::var("STRAYDOG_PIPELINE_DRAWN_GUESSES").is_ok_and(|value| value == "1") }
    }

    /// STRAYDOG_PIPELINE's mode: the pipeline unless it is named off (`for_devices` leaves it off where it does not apply).
    fn mode_named(value: Option<&str>) -> Mode {
        match value.map(str::trim) {
            None | Some("1") => Mode::On,
            Some("freeze") => Mode::Frozen,
            Some(_) => Mode::Off,
        }
    }

    /// The expert slots and the assist level held as they are.
    pub(super) fn frozen_slots(&self) -> bool {
        self.mode != Mode::Off
    }

    pub(super) fn no_copy(&self) -> bool {
        self.no_copy
    }

    pub(super) fn pass_log(&self) -> bool {
        self.pass_log
    }

    /// The assist level the tuner is not there to choose (the pipeline and `freeze` hold it): STRAYDOG_PIPELINE_ASSIST, else the
    /// pipeline's [`PIPELINE_ASSIST`] (`freeze` then leaves it to STRAYDOG_ASSIST or the tuner's start).
    pub(super) fn assist(&self) -> Option<u32> {
        self.assist.or((self.mode == Mode::On).then_some(PIPELINE_ASSIST))
    }

    /// STRAYDOG_PIPELINE_DRAFTS: the drafts of every pass, held.
    pub(super) fn drafts(&self) -> Option<usize> {
        self.drafts
    }

    /// The threads of each device's CPU engine: STRAYDOG_PIPELINE_CPU_THREADS, else half of what one engine would take.
    pub(super) fn engine_threads(&self, one_engine: u32) -> u32 {
        self.cpu_threads.map_or_else(|| one_engine.div_ceil(2).max(1), |threads| threads as u32)
    }

    /// Whether the first device keeps a second window of DeltaNet snapshots: the pipeline's, and the other side of its comparison's
    /// (`freeze`), which holds the same VRAM, so that its expert slots are the same.
    pub(super) fn windows(&self, devices: usize) -> bool {
        self.mode != Mode::Off && devices == 2 && !cfg!(feature = "hip")
    }

    /// Whether the passes overlap on this many devices (two NVIDIA GPUs).
    pub(super) fn enabled(&self, devices: usize) -> bool {
        self.mode == Mode::On && devices == 2 && !cfg!(feature = "hip")
    }

    /// The configuration for a group of `devices`: where the pipeline does not apply (one GPU, AMD GPUs) STRAYDOG_PIPELINE=1 sets up nothing, as
    /// if it were unset (no held slots, no second window, no second CPU engine, nothing in the ordinary loop changed); `freeze`, the other side
    /// of a comparison, is kept as it was asked for.
    pub(super) fn for_devices(mut self, devices: usize) -> PipelineConfig {
        if self.mode == Mode::On && !self.enabled(devices) {
            self.mode = Mode::Off;
        }
        self
    }
}

/// A pass whose front is under way or done on the first device.
struct Pass {
    /// The token the pass begins with and its drafts.
    rows: Vec<u32>,
    /// Where its first row sits in the sequence.
    position: usize,
    /// Its window on the first device (0 or 1: passes in flight at once differ).
    parity: usize,
    /// What the width chooser planned for it and how many drafts it could offer (the chooser's `record` at its verdict).
    plan: Option<(width::Plan, usize)>,
    /// The drafter's own hidden state at each of its drafts (the output of the step that made it), one WIDTH row a draft: what the
    /// pass's drafts are fed with in the next batch (see `chain_open`).
    stash: Vec<f32>,
}

/// A request where the pipelined loop and the ordinary one hand it to each other, at a pass's end with nothing in flight: the token chosen (not yet
/// in the history), the pairs the drafter has not caught up on, the tokens produced and the set the presence penalty reads.
pub(super) struct Hand {
    pub(super) generated: Generated,
    pub(super) token: u32,
    pub(super) produced: usize,
    pub(super) mtp_hidden: Vec<f32>,
    pub(super) mtp_tokens: Vec<u32>,
    pub(super) mtp_position: u64,
}

/// A pass to verify now, and the pass behind it whose front is under way for the case that this one's guess holds.
struct Current {
    pass: Pass,
    spec: Option<Pass>,
    /// The guess of the token after this pass's last row, when one was made though no pass was sent behind it (the gate was shut): its
    /// outcome still feeds the gate.
    guess: Option<u32>,
    /// The pass is the one row sent at once after a wrong guess.
    lone: bool,
}

/// What each pass of a request was, for the spec's quantities: F (run ahead beside the pass before it, a full window), R (the one row sent after a
/// wrong guess), O (the ordinary order's: nothing run ahead behind it) and S (another begun after a wrong guess, or the first), as [passes, guesses
/// made, guesses right, seconds, tokens], and which role followed which.
#[derive(Default)]
struct Roles {
    stats: [[f64; 5]; 4],
    transitions: [[u64; 4]; 4],
    last: Option<usize>,
}

impl Roles {
    const NAMES: [&'static str; 4] = ["F", "R", "O", "S"];

    fn add(&mut self, role: usize, guess: Option<bool>, seconds: f64, tokens: usize) {
        for (slot, amount) in self.stats[role].iter_mut().zip([1.0, f64::from(guess.is_some()), f64::from(guess == Some(true)), seconds, tokens as f64]) {
            *slot += amount;
        }
        if let Some(last) = self.last {
            self.transitions[last][role] += 1;
        }
        self.last = Some(role);
    }

    fn line(&self) -> String {
        let mut text = String::new();
        for (name, [passes, made, right, seconds, tokens]) in Self::NAMES.iter().zip(&self.stats) {
            if *passes > 0.0 {
                text += &format!("; role {name}: {passes:.0} passes, guesses {right:.0} right of {made:.0} made, {:.0} ms, {tokens:.0} tokens", 1e3 * seconds);
            }
        }
        let moves: Vec<String> = (0..4).flat_map(|from| (0..4).map(move |to| (from, to))).filter(|(from, to)| self.transitions[*from][*to] > 0)
            .map(|(from, to)| format!("{}{} {}", Self::NAMES[from], Self::NAMES[to], self.transitions[from][to])).collect();
        text + "; role changes: " + &moves.join(", ")
    }

    /// The request's metrics (for the reply's timings): each role's counts and each change's count.
    fn metrics(&self) -> Vec<(String, f64)> {
        let mut metrics = Vec::new();
        for (name, stats) in Self::NAMES.iter().zip(&self.stats) {
            for (field, value) in ["passes", "guesses", "right", "ms", "tokens"].iter().zip(stats) {
                metrics.push((format!("pipeline_role_{name}_{field}"), if *field == "ms" { 1e3 * value } else { *value }));
            }
        }
        for from in 0..4 {
            for to in 0..4 {
                metrics.push((format!("pipeline_change_{}{}", Self::NAMES[from], Self::NAMES[to]), self.transitions[from][to] as f64));
            }
        }
        metrics
    }
}

/// Whether passes are run ahead of their verdict, decided per request by what the passes cost: the pipeline's passes against the ordinary
/// order's (here: nothing sent behind a pass), each as a decaying mean of milliseconds a committed token over the request's last passes. It
/// begins speculating; after a stretch it runs a short probe in the other mode and keeps the cheaper, and probes the other again after a longer
/// stretch (sparsely where speculating wins: a probe in the ordinary order costs the saving it measures). A hit rate under the floor shuts it at
/// once (a miss costs the first device's stopped pass, its restore and a pass started in the ordinary order).
struct Gate {
    held: std::collections::VecDeque<bool>,
    floor: f32,
    speculating: bool,
    probing: bool,
    /// Passes done in this mode, and the first of them left out of the means (the mode's start is not its steady state).
    in_mode: usize,
    skip: usize,
    /// [plain, speculating]: decayed seconds and tokens.
    cost: [[f64; 2]; 2],
    /// Passes by mode, for the share of them that speculated.
    passes: [u64; 2],
    /// How many times longer than its base the settled mode's run is: it doubles (to 8) each time the probe of the other mode found it dearer by
    /// a tenth or more (a probe costs the saving it measures, which is largest where the mode wins by most), and is 1 again where the two were close.
    backoff: usize,
}

impl Gate {
    const WINDOW: usize = 16;
    const FIRST_RUN: usize = 40;
    const PROBE: usize = 12;
    const SPECULATING_RUN: usize = 160;
    const PLAIN_RUN: usize = 64;
    /// The tokens at the start of a stretch of the ordinary loop that are not timed.
    const STRETCH_WARM_UP: usize = 4;

    fn new(floor: f32) -> Gate {
        // (a floor above 1: never speculate, the ordinary order through the pipeline's loop, a test's)
        Gate { held: std::collections::VecDeque::with_capacity(Self::WINDOW), floor, speculating: floor <= 1.0, probing: false, in_mode: 0, skip: 0,
               cost: [[0.0; 2]; 2], passes: [0; 2], backoff: 1 }
    }

    /// Guess ahead now.
    fn open(&self) -> bool {
        self.speculating
    }

    /// Settled in the ordinary order: nothing the pipeline holds (the expert slots' swaps) need be held.
    fn settled_plain(&self) -> bool {
        !self.speculating && !self.probing
    }

    fn switch(&mut self, speculating: bool, probing: bool) {
        (self.speculating, self.probing, self.in_mode, self.skip) = (speculating, probing, 0, 2);
    }

    /// A guess was right or wrong: a hit rate under the floor, over enough guesses, shuts the gate for the ordinary run.
    fn record(&mut self, right: bool) {
        if self.held.len() == Self::WINDOW {
            self.held.pop_front();
        }
        self.held.push_back(right);
        let rate = self.held.iter().filter(|held| **held).count() as f32 / self.held.len() as f32;
        if self.speculating && self.held.len() >= 8 && rate < self.floor {
            self.held.clear();
            self.switch(false, false);
        }
    }

    /// A pass took `seconds` for `tokens` committed, in the pipeline's mode (`ran_ahead`) or the ordinary one's.
    fn pass_done(&mut self, seconds: f64, tokens: usize, ran_ahead: bool) {
        let mode = usize::from(ran_ahead);
        self.passes[mode] += 1;
        // (a floor of 0: always run ahead; above 1: never; both for a test, with no probes)
        if self.floor > 1.0 || self.floor <= 0.0 {
            return;
        }
        if mode != usize::from(self.speculating) {
            // (a pass of the mode just left, finishing)
            return;
        }
        self.in_mode += 1;
        if self.skip > 0 {
            self.skip -= 1;
            return;
        }
        for (slot, amount) in self.cost[mode].iter_mut().zip([seconds, tokens as f64]) {
            *slot = *slot * 0.9 + amount;
        }
        let run = match (self.speculating, self.probing) {
            (_, true) => Self::PROBE,
            (true, false) if self.cost[0][1] == 0.0 => Self::FIRST_RUN,
            (true, false) => Self::SPECULATING_RUN * self.backoff,
            (false, false) => Self::PLAIN_RUN * self.backoff,
        };
        if self.in_mode < run {
            return;
        }
        self.run_over();
    }

    /// Whether the ordinary order's stretches are the ordinary loop's: not where the gate is held open or shut for a test (a floor of 0 or above 1).
    fn hands_over(&self) -> bool {
        self.floor > 0.0 && self.floor <= 1.0
    }

    /// The tokens of the next stretch in the ordinary order, as many as the run or the probe of the pass counts below would commit.
    fn stretch_tokens(&self) -> usize {
        2 * if self.probing { Self::PROBE } else { Self::PLAIN_RUN * self.backoff }
    }

    /// A stretch of the ordinary loop committed `tokens` in `seconds`: it is the mode's whole run (or probe).
    fn stretch_done(&mut self, seconds: f64, tokens: usize) {
        self.passes[0] += (tokens / 2).max(1) as u64;
        if !self.hands_over() || self.speculating {
            return;
        }
        for (slot, amount) in self.cost[0].iter_mut().zip([seconds, tokens as f64]) {
            *slot = *slot * 0.9 + amount;
        }
        self.run_over();
    }

    /// Whether the mode being probed against is the speculating one (the probe is of the ordinary order).
    fn probing_from_speculating(&self) -> bool {
        // (a probe of the ordinary order has `speculating` false, one of the pipeline true: the mode probed against is the other)
        !self.speculating
    }

    /// The mode's run is over: a probe of the other mode, or after the probe the cheaper of the two.
    fn run_over(&mut self) {
        let per_token = |mode: usize| (self.cost[mode][1] > 0.0).then(|| self.cost[mode][0] / self.cost[mode][1]);
        if self.probing {
            // The probe of the other mode is over: the cheaper one, the speculating one by a hair when they are as dear.
            let (speculating, wide) = match (per_token(0), per_token(1)) {
                (Some(plain), Some(ahead)) => (ahead <= plain * 1.01, ahead < plain * 0.9 || plain < ahead * 0.9),
                _ => (true, false),
            };
            // (the mode that stays was cheaper by a margin: its next run is longer; one that was nearly as dear as the other's is probed as often as before)
            let stays = speculating == self.probing_from_speculating();
            self.backoff = if wide && stays { (self.backoff * 2).min(8) } else { 1 };
            self.switch(speculating, false);
        } else {
            // The stretch is over: a short probe of the other mode.
            let speculating = !self.speculating;
            self.switch(speculating, true);
        }
    }
}

/// What the passes of a request cost, by what became of the pass and how it began (DS41_PROFILE adds the devices' times, see `Devices`). The rows of
/// the table: a pass whose guess behind it held and was continued ("hit"), whose guess was wrong with every draft kept ("bonus miss"), whose draft
/// was rejected ("draft miss"), and a pass with nothing sent behind it ("ordinary"); its columns: continued (its front ran beside the pass before it)
/// or started (after a wrong guess, or from the start). A cell sums, per pass: 1, the period (from the end of the previous pass's work to the end of
/// this one's: its verdict, its tokens out, the chain and the send of the next pass, its commit and, when the pass behind it was given up, the first
/// device's stop and restore), its tokens committed, its rows, the restore alone (stop and restore of the first device) and the chain alone (the
/// drafter's steps and the send of the pass behind it, and for a started pass the drafts of its own).
#[derive(Default)]
struct Costs([[f64; 6]; 10]);

impl Costs {
    const NAMES: [&'static str; 5] = ["hit", "bonus miss", "draft miss", "ordinary", "ended with the guess right"];

    fn add(&mut self, outcome: usize, started: bool, period: f64, tokens: usize, rows: usize, restore: f64, chain: f64) {
        for (slot, amount) in self.0[outcome * 2 + usize::from(started)].iter_mut().zip([1.0, period, tokens as f64, rows as f64, restore, chain]) {
            *slot += amount;
        }
    }

    fn lines(&self) -> String {
        let mut text = String::new();
        for outcome in 0..5 {
            for started in 0..2 {
                let [passes, period, tokens, rows, restore, chain] = self.0[outcome * 2 + started];
                if passes == 0.0 {
                    continue;
                }
                text += &format!("; {} {}: {passes:.0} passes of {:.1} rows, {:.1} ms a pass, {:.1} ms a token, restore {:.2} ms, chain {:.2} ms",
                                 Self::NAMES[outcome], if started == 1 { "started" } else { "continued" }, rows / passes, 1e3 * period / passes,
                                 1e3 * period / tokens.max(1.0), 1e3 * restore / passes, 1e3 * chain / passes);
            }
        }
        text
    }
}

/// With DS41_PROFILE, a pass's time on each device, summed over the request's passes, in seconds. A: the first device's front (the worker's wall time
/// in its job, and in it the GPU's layer work, its resident experts, the host's wait for the GPU's exports and the CPU's experts, from the device's
/// counters). B: the second device's span (the same four) and its head (wall). C: the drafter's chain (wall, with the sends). The time the loop waited
/// for the front, and the passes counted.
#[derive(Default)]
struct Devices {
    passes: f64,
    a: [f64; 5],
    b: [f64; 5],
    head: f64,
    chain: f64,
    front_wait: f64,
    /// The verdict's work (acceptance), the commit on the second device, the rows' making beside the devices, and the hand-off of the passes
    /// sent to an idle first device (sum, count).
    accept: f64,
    commit: f64,
    prepared: f64,
    handoff: (f64, f64),
    /// The passes' periods summed.
    period: f64,
}

impl Devices {
    fn line(&self) -> String {
        let n = self.passes.max(1.0);
        let ms = |seconds: f64| 1e3 * seconds / n;
        format!("a pass, ms: A first device (job wall {:.2}: GPU layers {:.2} + experts {:.2}, host wait {:.2}, CPU {:.2}), B second device (span wall {:.2}: GPU \
                 layers {:.2} + experts {:.2}, host wait {:.2}, CPU {:.2}; head wall {:.2}), C chain wall {:.2}, the loop waited for the front {:.2} ({} passes)",
                ms(self.a[0]), ms(self.a[1]), ms(self.a[2]), ms(self.a[3]), ms(self.a[4]), ms(self.b[0]), ms(self.b[1]), ms(self.b[2]), ms(self.b[3]),
                ms(self.b[4]), ms(self.head), ms(self.chain), ms(self.front_wait), self.passes)
            + &format!("; also a pass: accept {:.2}, commit {:.2}, rows made beside the devices {:.2}, hand-off to an idle first device {:.3} ({:.0} of them); the period {:.2}, of which the loop's wait for the front, the second device's span, \
                       head, acceptance, chain and commit make {:.2} (the rest: tokens out, the history, the bookkeeping)",
                       ms(self.accept), ms(self.commit), ms(self.prepared), 1e3 * self.handoff.0 / self.handoff.1.max(1.0), self.handoff.1, ms(self.period),
                       ms(self.front_wait + self.b[0] + self.head + self.accept + self.chain + self.commit))
    }
}

/// The drafter's calls (and the stash's read-back) of `passes` passes, each kind as calls a pass and milliseconds a pass (and a call).
fn drafter_line(stats: &[[f64; 2]; 4], passes: f64) -> String {
    ["catch-up batch", "forced step", "draft step", "stash read-back"].iter().zip(stats)
        .map(|(name, [calls, seconds])| format!("{name} {:.2} calls {:.2} ms ({:.2} each)", calls / passes.max(1.0), 1e3 * seconds / passes.max(1.0), 1e3 * seconds / calls.max(1.0)))
        .collect::<Vec<_>>().join(", ")
}

/// A device's timers (wall; GPU layers; GPU experts; host wait; CPU), as differences.
fn timers(device: &Backend) -> [f64; 4] {
    device.hybrid_counts().map_or([0.0; 4], |counts| [counts[9], counts[10], counts[4], counts[5]].map(|value| value as f64 / 1e9))
}

/// A pass asked for: its tokens, the tokens before them (the PLE rows are read after the last two), where it sits, its window. The loop
/// sends it and goes on; the rows it needs are made beside the devices (see `prepare`).
struct FrontRequest {
    rows: Vec<u32>,
    before: Vec<u32>,
    position: usize,
    parity: usize,
    /// Sent behind a pass that is not decided yet: what its rows observe is noted before it runs, to be put back if it is given up.
    speculative: bool,
    /// DS41_PROFILE: the job's time on the device is taken.
    profile: bool,
    /// When it was sent, and whether the first device was idle then (the hand-off is timed for those).
    sent: Instant,
    idle: bool,
}

/// What the first device runs for a pass: its rows (the tiled token embeddings), its PLE rows, where it sits, its window.
struct FrontJob {
    hidden: Vec<f32>,
    embeddings: Vec<f32>,
    /// The seconds the rows took to make.
    prepared: f64,
    sent: Instant,
    idle: bool,
    position: usize,
    parity: usize,
    /// Sent behind a pass that is not decided yet: what its rows observe is noted before it runs, to be put back if it is given up.
    speculative: bool,
    /// DS41_PROFILE: the job's time on the device is taken.
    profile: bool,
}

/// The first device put back to the first `kept` rows of the pass of window `parity` (`force`: with a pass behind it given up).
struct RestoreJob {
    parity: usize,
    kept: usize,
    force: bool,
    recover: Option<Recovery>,
    first: Vec<f32>,
    embeddings: Vec<f32>,
}

/// Identity of the speculative front whose input prefix is now wholly committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Recovery {
    parity: usize,
    position: usize,
    rows: usize,
}

#[derive(Default)]
struct FrontTrace {
    front: Option<Recovery>,
    epoch: u64,
    speculative: bool,
    ple_saved: bool,
}

impl FrontTrace {
    fn matches(&self, front: Recovery) -> bool {
        self.speculative && self.epoch != 0 && self.front == Some(front)
    }
}

fn bonus_missed(accepted: usize, offered: usize, kept: usize, rows: usize, ended: bool, cancelled: bool, following: u32, guessed: u32) -> bool {
    accepted == offered && kept == rows && !ended && !cancelled && following != guessed
}

enum Command {
    /// To the relay: a pass asked for, which becomes a `Front` with its rows made.
    Prepare(FrontRequest),
    Front(FrontJob),
    /// A pass whose rows could not be made.
    Failed(String),
    Restore(RestoreJob),
    /// The first device, handed back to the loop for its own use at a drain (nothing in flight).
    Yield,
    /// The first device, lent to the worker again.
    Resume(Box<Backend>),
    Stop,
}

enum Reply {
    Front(Result<FrontOut>),
    Restored(Result<Option<u64>>),
    Yielded(Result<Box<Backend>>),
}

/// The first device's rows: after layer 0, and after its span.
struct FrontOut {
    first: Vec<f32>,
    hidden: Vec<f32>,
    /// The pass's PLE rows, which a partial acceptance replays.
    embeddings: Vec<f32>,
    /// The seconds its rows took to make (beside the devices).
    prepared: f64,
    /// For a pass sent to an idle first device: the seconds from the send to the job's start, less the rows' making (the hand-off).
    handoff: Option<f64>,
    /// The job's [wall, GPU layers, GPU experts, host wait, CPU] in seconds (DS41_PROFILE).
    time: Option<[f64; 5]>,
}

/// The PLE state before the pass of a window, which a replay restores.
fn snapshot_name(parity: usize) -> String {
    format!("@ple-snapshot{parity}:blk.1.")
}

/// The rows of a pass: the token embeddings tiled over the residual streams and the PLE rows (after the last two tokens before them), as the
/// ordinary loop reads them; the reads are started before the first is waited for.
fn prepare(weights: &Weights, request: &FrontRequest) -> Result<FrontJob> {
    let started = Instant::now();
    let id = |token: &u32| if *token >= IMAGE_TOKEN { i64::from(IMAGE_PAD) } else { i64::from(*token) };
    let history: Vec<i64> = request.before[request.before.len().saturating_sub(2)..].iter().map(id).collect();
    let tokens: Vec<i64> = request.rows.iter().map(id).collect();
    let indices = ngram_rows(&history, &tokens, &weights.integer_arrays)?;
    weights.prefetch_rows(PLE_TABLE, &indices)?;
    let ids: Vec<i64> = request.rows.iter().map(|token| *token as i64).collect();
    let hidden = tile(&weights.rows("token_embd.weight", &ids)?);
    let embeddings = weights.rows(PLE_TABLE, &indices)?;
    Ok(FrontJob { hidden, embeddings, prepared: started.elapsed().as_secs_f64(), position: request.position, parity: request.parity,
                  speculative: request.speculative, profile: request.profile, sent: request.sent, idle: request.idle })
}

/// Makes the rows of the passes the loop asks for, in order, and hands everything else on to the first device's thread as it comes: the
/// rows of a pass sent behind another are made while the first device is on that one.
/// A failed forwarding hop still owns an unsent Resume's actual Backend.
fn forward_front_command(worker: &Sender<Command>, command: Command) -> std::result::Result<(), (String, Option<Backend>)> {
    worker.send(command).map_err(|error| {
        let returned = match error.0 { Command::Resume(device) => Some(*device), _ => None };
        ("the first device's thread ended before receiving a command".to_string(), returned)
    })
}

fn relay(weights: std::sync::Arc<Weights>, commands: Receiver<Command>, worker: Sender<Command>) -> (Result<()>, Option<Backend>) {
    // (a counted reference: the reads are through `&Weights`, which locks what it caches, and the model's own reads are shared ones too)
    for command in commands {
        let command = match command {
            Command::Prepare(request) => match prepare(&weights, &request) {
                Ok(job) => Command::Front(job),
                Err(error) => Command::Failed(error),
            },
            other => other,
        };
        let stop = matches!(command, Command::Stop);
        if let Err((error, returned)) = forward_front_command(&worker, command) { return (Err(error), returned); }
        if stop { return (Ok(()), None); }
    }
    (Ok(()), None)
}

fn run_front(device: &mut Backend, ple: &[Operand], split: usize, job: FrontJob, trace: &mut FrontTrace) -> Result<FrontOut> {
    let started = Instant::now();
    let handoff = job.idle.then(|| (started.duration_since(job.sent).as_secs_f64() - job.prepared).max(0.0));
    let before = job.profile.then(|| timers(device));
    *trace = FrontTrace::default();
    let rows = job.hidden.len() / WIDTH;
    trace.epoch = device.begin_verify_window(job.parity, job.position, rows)?;
    trace.front = Some(Recovery { parity: job.parity, position: job.position, rows });
    trace.speculative = job.speculative;
    if job.speculative {
        device.usage_mark()?;
    }
    let first = device.verify_span(0, 1, &job.hidden, job.position as u64)?;
    device.copy_state(&snapshot_name(job.parity), "@ple:blk.1.")?;
    trace.ple_saved = true;
    let hidden = device.ple_rows("blk.1.", &first, &job.embeddings, ple)?;
    let hidden = device.verify_span(1, split, &hidden, job.position as u64)?;
    let wall = started.elapsed().as_secs_f64();
    let time = before.map(|before| {
        let after = timers(device);
        [wall, after[0] - before[0], after[1] - before[1], after[2] - before[2], after[3] - before[3]]
    });
    Ok(FrontOut { first, hidden, embeddings: job.embeddings, prepared: job.prepared, handoff, time })
}

fn run_restore(device: &mut Backend, ple: &[Operand], split: usize, job: &RestoreJob, trace: &mut FrontTrace) -> Result<Option<u64>> {
    if let Some(front) = job.recover.filter(|front| trace.matches(*front)) {
        device.set_window(front.parity)?;
        if let Some(touched) = device.restore_verify_window(split, trace.epoch, front.position, front.rows)? {
            if trace.ple_saved {
                device.copy_state("@ple:blk.1.", &snapshot_name(front.parity))?;
            }
            device.set_window(job.parity)?;
            for (begin, end) in [(0, 1), (1, split)] {
                device.verify_commit(begin, end, job.kept as u64)?;
            }
            device.usage_restore()?;
            *trace = FrontTrace::default();
            return Ok(Some(touched));
        }
    }
    device.set_window(job.parity)?;
    for (begin, end) in [(0, 1), (1, split)] {
        if job.force {
            device.verify_commit_forced(begin, end, job.kept as u64)?;
        } else {
            device.verify_commit(begin, end, job.kept as u64)?;
        }
    }
    if job.force {
        device.usage_restore()?;
    }
    if job.force || job.kept * WIDTH < job.first.len() {
        device.copy_state("@ple:blk.1.", &snapshot_name(job.parity))?;
        device.ple_rows("blk.1.", &job.first[..job.kept * WIDTH], &job.embeddings[..job.kept * HIDDEN], ple)?;
    }
    *trace = FrontTrace::default();
    Ok(None)
}

/// The first device's thread: it owns the device (the Backend, moved in by value) while the loop runs, lends it back when the loop asks at a
/// drain (`Command::Yield`, answered with the Backend itself) and takes it again (`Command::Resume`), and returns it when the loop is over.
fn front_worker(device: Backend, ple: Vec<Operand<'static>>, split: usize, commands: Receiver<Command>, replies: Sender<Reply>) -> Option<Backend> {
    let mut device = Some(device);
    let mut trace = FrontTrace::default();
    const LENT: &str = "the first device is lent to the loop";
    for command in commands {
        let reply = match command {
            Command::Front(job) => Reply::Front(match device.as_mut() {
                Some(device) => run_front(device, &ple, split, job, &mut trace),
                None => Err(LENT.to_string()),
            }),
            Command::Failed(error) => Reply::Front(Err(error)),
            Command::Restore(job) => Reply::Restored(match device.as_mut() {
                Some(device) => run_restore(device, &ple, split, &job, &mut trace),
                None => Err(LENT.to_string()),
            }),
            Command::Prepare(_) => Reply::Front(Err("a pass reached the first device unprepared".into())),
            Command::Yield => Reply::Yielded(device.take().map(Box::new).ok_or_else(|| LENT.to_string())),
            Command::Resume(back) => {
                device = Some(*back);
                continue;
            }
            Command::Stop => return device,
        };
        if replies.send(reply).is_err() {
            return device;
        }
    }
    device
}

/// Completed fronts retain their FIFO order when an idle check receives them before their verdict.
struct FrontReplies {
    replies: Receiver<Reply>,
    cached: VecDeque<Result<FrontOut>>,
    /// Physical fronts not yet received from the channel; cached fronts are no longer in flight.
    in_flight: usize,
    protocol_error: Option<String>,
    /// Unexpected control replies stay owned here, including any yielded Backend.
    controls: Vec<Reply>,
}

impl FrontReplies {
    fn new(replies: Receiver<Reply>) -> Self {
        Self { replies, cached: Default::default(), in_flight: 0, protocol_error: None, controls: Vec::new() }
    }

    fn failed(&mut self, error: &str) -> String {
        self.protocol_error.get_or_insert_with(|| error.to_string()).clone()
    }

    fn receive(&mut self, reply: Reply) -> Result<()> {
        match reply {
            Reply::Front(out) => {
                self.cached.push_back(out);
                let Some(left) = self.in_flight.checked_sub(1) else {
                    return Err(self.failed("the first device sent an unsolicited front"));
                };
                self.in_flight = left;
                Ok(())
            }
            control => {
                self.controls.push(control);
                Err(self.failed("the first device's thread answered out of order"))
            }
        }
    }

    fn completed(&self) -> Result<()> {
        let mut result = Ok(());
        for error in self.cached.iter().filter_map(|out| out.as_ref().err()) {
            result = finish_device_work(result, Err(error.clone()));
        }
        finish_device_work(result, self.protocol_error.clone().map_or(Ok(()), Err))
    }

    /// Poll only: all submitted fronts must physically finish, and every cached result must be sound.
    fn idle(&mut self) -> Result<bool> {
        loop {
            match self.replies.try_recv() {
                Ok(reply) => self.receive(reply)?,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Err(self.failed("the first device's thread ended")),
            }
        }
        self.completed()?;
        Ok(self.in_flight == 0)
    }

    /// Outer errors describe a broken protocol; the inner result belongs to the oldest front.
    fn next(&mut self) -> Result<Result<FrontOut>> {
        if self.cached.is_empty() {
            if self.in_flight == 0 {
                return Err(self.failed("the first device has no pending front"));
            }
            let reply = match self.replies.recv() {
                Ok(reply) => reply,
                Err(_) => {
                    self.in_flight = 0;
                    return Err(self.failed("the first device's thread ended"));
                }
            };
            self.receive(reply)?;
        }
        self.cached.pop_front().ok_or_else(|| self.failed("the first device's front is missing"))
    }

    fn empty(&self) -> bool {
        self.in_flight == 0 && self.cached.is_empty()
    }

    /// Called after joining the worker: preserve a Backend delivered with an unexpected Yield,
    /// and report every late front/control error before the model can be reused.
    fn after_shutdown(&mut self) -> (Result<()>, Option<Backend>) {
        while let Ok(reply) = self.replies.try_recv() { let _ = self.receive(reply); }
        let drained = drain_fronts(self);
        let mut returned = None;
        for reply in self.controls.drain(..) {
            if let Reply::Yielded(Ok(device)) = reply {
                if returned.is_none() { returned = Some(*device); }
            }
        }
        (drained, returned)
    }
}

/// Receive cached and running fronts before shutdown. Normal span aborts are expected; a
/// later device fault replaces an earlier recovered wait, including wholly cached failures.
fn drain_fronts(fronts: &mut FrontReplies) -> Result<()> {
    let mut result = fronts.protocol_error.clone().map_or(Ok(()), Err);
    while !fronts.empty() {
        let front = match fronts.next() {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(error)) if error.contains(SPAN_ABORTED) => Ok(()),
            Ok(Err(error)) | Err(error) => Err(error),
        };
        result = finish_device_work(result, front);
    }
    result
}

/// The loop's side of the worker. A synchronous loan admits no new front, restore or abort.
struct Link {
    commands: Sender<Command>,
    fronts: FrontReplies,
    abort: SpanAbort,
    aborting: bool,
    restoring: bool,
    loaned: bool,
}

impl Link {
    fn submit(&mut self, request: FrontRequest) -> Result<()> {
        if self.loaned || self.aborting || self.restoring {
            return Err("the first device is unavailable for a new front".into());
        }
        self.fronts.completed()?;
        let count = self.fronts.in_flight.checked_add(1).ok_or("the first device has too many pending fronts")?;
        self.commands.send(Command::Prepare(request)).map_err(|_| "the first device's thread ended".to_string())?;
        self.fronts.in_flight = count;
        Ok(())
    }

    /// The oldest semantic front: cached results precede channel results and are consumed only once.
    fn front(&mut self) -> Result<FrontOut> {
        self.fronts.next()?
    }

    fn idle(&mut self) -> Result<bool> {
        if self.loaned || self.aborting || self.restoring { return Ok(false); }
        self.fronts.idle()
    }

    fn cancel(&mut self) -> Result<()> {
        if self.loaned || self.restoring { return Err("the first device cannot abort during a loan or restore".into()); }
        self.aborting = true;
        self.abort.raise(true);
        Ok(())
    }

    /// Give up semantic fronts too: an idle poll may have cached a failed front already.
    fn give_up(&mut self) -> Result<()> {
        if self.fronts.in_flight > 0 { self.cancel()?; }
        let drained = drain_fronts(&mut self.fronts);
        self.abort.raise(false);
        self.aborting = false;
        drained
    }

    /// The actual device comes back even when its completed fronts still await their verdict.
    fn recall(&mut self) -> Result<Backend> {
        if self.loaned || self.aborting || self.restoring || self.fronts.in_flight != 0 {
            return Err("the first device was asked for while unavailable".into());
        }
        self.fronts.completed()?;
        self.commands.send(Command::Yield).map_err(|_| "the first device's thread ended".to_string())?;
        self.loaned = true;
        match self.fronts.replies.recv() {
            Ok(Reply::Yielded(device)) => device.map(|device| *device),
            Ok(other) => {
                let _ = self.fronts.receive(other);
                Err(self.fronts.failed("the first device's thread answered out of order"))
            }
            Err(_) => Err(self.fronts.failed("the first device's thread ended")),
        }
    }

    fn lend(&mut self, device: Backend) -> Result<()> {
        match self.commands.send(Command::Resume(Box::new(device))) {
            Ok(()) => { self.loaned = false; Ok(()) }
            Err(error) => {
                // Keep the actual Backend for the model's final return if the receiver ended.
                if let Command::Resume(device) = error.0 { self.fronts.controls.push(Reply::Yielded(Ok(device))); }
                Err(self.fronts.failed("the first device's thread ended"))
            }
        }
    }

    fn restore(&mut self, job: RestoreJob) -> Result<Option<u64>> {
        if self.loaned || self.aborting || self.restoring || !self.fronts.empty() {
            return Err("the first device is unavailable for a restore".into());
        }
        self.fronts.completed()?;
        self.commands.send(Command::Restore(job)).map_err(|_| "the first device's thread ended".to_string())?;
        self.restoring = true;
        let result = match self.fronts.replies.recv() {
            Ok(Reply::Restored(result)) => result,
            Ok(other) => {
                let _ = self.fronts.receive(other);
                Err(self.fronts.failed("the first device's thread answered out of order"))
            }
            Err(_) => Err(self.fronts.failed("the first device's thread ended")),
        };
        self.restoring = false;
        result
    }

    fn shutdown(&mut self) -> Result<()> {
        let drained = self.give_up();
        self.abort.raise(true);
        let stopped = self.commands.send(Command::Stop).map_err(|_| "the first device's thread ended".to_string());
        finish_device_work(drained, stopped)
    }
}

/// The operands the loop uses, made once: the loop cannot borrow the model for them while the worker is out.
struct Tools {
    head: Vec<Operand<'static>>,
    draft_with_head: Vec<Operand<'static>>,
    draft_without_head: Vec<Operand<'static>>,
    /// Pinned while the first Backend still belongs to the model; read only during an exclusive loan.
    lower_head: Option<(u64, *mut std::ffi::c_void)>,
    /// DS41_PROFILE: the drafter's calls by kind (the catch-up's batch, a forced token's step, a draft's step, the drafter's own hidden state read
    /// back for the pass behind) as [calls, seconds], and the time of the sends (the request made and handed to the relay).
    stats: std::cell::RefCell<[[[f64; 2]; 4]; 2]>,
    /// Whether the chain being made is that of a pass begun after a wrong guess (else that of a pass that follows one): which half of `stats`.
    started: std::cell::Cell<bool>,
    /// What the chain's predictions are drawn with (sampling::guess_keyed), set where a chain begins: None where they are the drafter's most
    /// likely tokens (the request's choice is the greatest logit, or STRAYDOG_PIPELINE_DRAWN_GUESSES=0).
    drawing: std::cell::RefCell<Option<Drawing>>,
}

/// The request's sampling, the tokens generated so far with the ones the chain has been fed since (each joins when it is fed: the target's
/// penalty at a row counts the tokens before it), and the seed of the keyed draws.
struct Drawing {
    sampling: Sampling,
    generated: Generated,
    seed: u64,
}

/// The drafter's chain from the second device alone: the catch-up of the pairs of a pass, the tokens fed in turn, then each
/// prediction fed to the next step. Every prediction comes with its probability under the drafter.
struct Chain {
    first: Option<(u32, f32)>,
    last: u32,
    previous: u64,
    position: u64,
}

/// STRAYDOG_PIPELINE_DRAWS=0 (development): a request whose draw reads a row's greatest logits alone (a temperature, a small top_k, no
/// presence penalty: the model card's sampling for reasoning) keeps the ordinary loop and its drawn drafts, as before the pipeline took it.
fn pipeline_draws() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("STRAYDOG_PIPELINE_DRAWS").map_or(true, |value| value.trim() != "0"))
}

impl Model {
    /// `work` with the first device in the group: it has been the worker's, by value, while the loop runs (a vacant backend stands in its place),
    /// so the loop asks for it back at a drain, uses it, and lends it to the worker again, whatever `work` returned.
    fn on_first_device<T>(&mut self, link: &mut Link, work: impl FnOnce(&mut Model) -> Result<T>) -> Result<T> {
        let device = link.recall()?;
        let ordinal = device.ordinal;
        drop(std::mem::replace(&mut self.group.backends[0], device));
        let result = work(self);
        let device = std::mem::replace(&mut self.group.backends[0], Backend::vacant(ordinal));
        finish_device_work(result, link.lend(device))
    }

    /// Whether what the pipeline holds is held now (see `pipeline_live`).
    pub(super) fn frozen_now(&self) -> bool {
        self.pipeline.frozen_slots() && self.pipeline_live
    }

    /// A request begins: it holds what the pipeline holds if it is pipelined (or the mode is `freeze`), else the ordinary loop has the slots' swaps
    /// and the assist tuner as without STRAYDOG_PIPELINE.
    pub(super) fn pipeline_live_for(&mut self, pipelined: bool) -> Result<()> {
        let live = pipelined || self.pipeline.mode == Mode::Frozen;
        if live != self.pipeline_live {
            self.pipeline_live = live;
            if self.partial && self.pipeline.windows(self.group.backends.len()) {
                self.group.hold_slots(live && !self.pipeline.free_slots)?;
            }
        }
        Ok(())
    }

    /// Whether this request's passes are pipelined: greedy or sampled (in its pipelined passes its drafts are point proposals and its tokens
    /// the target's draws keyed by position, from the whole rows' logits or from the rows' greatest where the draw reads only those; a
    /// stretch handed over to the ordinary loop draws as that loop does), nothing constraining it, no image, the layers split between two
    /// devices with a CPU engine for each.
    pub(super) fn pipeline_applies(&self, sampling: &Sampling, constraint: Option<&Constraint>) -> bool {
        self.pipeline.enabled(self.group.backends.len())
            && self.partial
            && self.speculative > 0
            && self.cpu_second.is_some()
            && constraint.is_none()
            && (sampling.top_candidates().is_none() || pipeline_draws())
            && self.forced_rows.is_none()
            && self.images.is_empty()
            && self.rope_delta == 0
            && !self.rope_table
            && !self.group.parts()
            && self.group.split().is_some_and(|split| split >= 2)
    }

    /// The token ids a pass's PLE rows read after: the last two of `history` then `after`.
    fn tail_with(&self, after: &[u32]) -> Vec<u32> {
        let mut tail: Vec<u32> = self.history[self.history.len().saturating_sub(2)..].to_vec();
        tail.extend_from_slice(after);
        tail
    }

    /// The first device's pass for `rows` at `position`, the two tokens before them being the last two of `before`: asked for here and
    /// made beside the devices.
    fn front_request(&self, rows: &[u32], position: usize, before: &[u32], parity: usize, speculative: bool, idle: bool) -> FrontRequest {
        let profile = self.profile.as_ref().is_some_and(|profile| profile.every);
        FrontRequest { rows: rows.to_vec(), before: before.to_vec(), position, parity, speculative, profile, sent: Instant::now(), idle }
    }

    /// The opt-in head may borrow the first device only after all its physical fronts finish.
    /// Every error after queueing drains the drafter before the pipeline begins its shutdown.
    fn chain_step(&mut self, link: &mut Link, tools: &Tools, embeddings: &[f32], hidden: Option<&[f32]>, previous: u64, position: u64,
                  head: bool, kind: usize) -> Result<Option<crate::gpu::Draft>> {
        let operands = if head { &tools.draft_with_head } else { &tools.draft_without_head };
        let last = self.group.backends.len() - 1;
        let started = Instant::now();
        let step = if self.pipeline.head_share && head {
            let queued = self.group.backends[last].draft_queue(&tools.draft_without_head, 48, position, embeddings, hidden, previous);
            let result = queued.and_then(|()| {
                let polling = Instant::now();
                let idle = link.idle()?;
                self.metric_add("pipeline_head_share_poll_seconds", polling.elapsed().as_secs_f64());
                let (vocabulary_rows, extras, read_rows) = self.group.backends[last].draft_vocabulary_info(operands[7].rows, 0)?;
                self.last_metrics.entry("pipeline_head_vocabulary_rows").or_insert_with(|| json!(vocabulary_rows));
                self.last_metrics.entry("pipeline_head_extra_count").or_insert_with(|| json!(extras));
                self.last_metrics.insert("pipeline_head_read_rows".into(), json!(read_rows));
                let eligible = head_share_eligible(true, idle, tools.lower_head.is_some(), extras, read_rows);
                self.metric_count("pipeline_head_share_ready", u64::from(eligible));
                self.metric_count("pipeline_head_share_busy", u64::from(!idle));
                self.metric_count("pipeline_head_share_vocab_fallback", u64::from(extras != 0 || read_rows < 2 || tools.lower_head.is_none()));
                let split = if eligible { tools.lower_head } else { None };
                let rows = hidden.map_or(1, |hidden| hidden.len() / WIDTH) as u64;
                let head_started = Instant::now();
                let result = if split.is_some() {
                    // The native split head mutates the peer's scratch and transfer counters:
                    // its actual Backend is back in the model until this synchronous call ends.
                    self.on_first_device(link, |model| model.group.backends[last].draft_head(&operands[4..], 48, rows - 1, split, 0))
                } else {
                    self.group.backends[last].draft_head(&operands[4..], 48, rows - 1, None, 0)
                };
                self.metric_count("pipeline_head_share_used", u64::from(split.is_some() && result.is_ok()));
                self.metric_add(if split.is_some() { "pipeline_head_split_seconds" } else { "pipeline_head_single_seconds" },
                                head_started.elapsed().as_secs_f64());
                self.metric_count(if split.is_some() { "pipeline_head_split_count" } else { "pipeline_head_single_count" }, 1);
                result.map(Some)
            });
            if result.is_err() {
                self.group.backends[last].draft_drain()
                    .map_err(|error| format!("could not drain queued drafter: {error}; pipeline state is unusable"))
                    .and(result)
            } else { result }
        } else {
            // Default-off requests retain the original step, without a queue, idle poll or split-head pin.
            // (the drafter's greatest logits with the step, where the prediction is drawn from them)
            let top = if head && tools.drawing.borrow().is_some() { sampling::GUESS_TOP } else { 0 };
            self.group.backends[last].draft_step(operands, 48, position, embeddings, hidden, previous, None, top)
        };
        let mut stats = tools.stats.borrow_mut();
        let slot = &mut stats[usize::from(tools.started.get())][kind];
        slot[0] += 1.0;
        slot[1] += started.elapsed().as_secs_f64();
        step
    }

    /// A step's prediction: the token after the one at `position` (the last the chain was fed), and the probability of the drafter's most
    /// likely token. Drawn as the target will draw at that row where the chain draws (`Tools::drawing`), else the most likely token.
    fn predicted(tools: &Tools, draft: Option<crate::gpu::Draft>, position: u64) -> Result<(u32, f32)> {
        let draft = draft.ok_or("Straydog drafter head returned no token")?;
        if let Some(drawing) = tools.drawing.borrow().as_ref() {
            if let Some(guess) = sampling::guess_keyed(&draft.top, &drawing.sampling, &drawing.generated, &mut Rng::keyed(drawing.seed, position)) {
                return Ok(guess);
            }
        }
        Ok((u32::try_from(draft.token).map_err(|_| "Straydog drafter head returned no token")?, draft.confidence))
    }

    /// The chain begins: its predictions are drawn as the target's keyed draws where the request's choice is not the greatest logit.
    fn chain_drawing(&self, tools: &Tools, sampling: &Sampling, generated: &Generated) {
        let drawing = (self.pipeline.keyed_guesses && sampling::guesses_keyed(sampling))
            .then(|| Drawing { sampling: sampling.clone(), generated: generated.clone(), seed: self.sample_seed });
        tools.drawing.replace(drawing);
    }

    /// A token the chain is fed joins the tokens its later predictions' penalty counts.
    fn chain_fed(tools: &Tools, tokens: &[u32]) {
        if let Some(drawing) = tools.drawing.borrow_mut().as_mut() {
            for token in tokens {
                drawing.generated.insert(*token);
            }
        }
    }

    /// The noted tokens' rows into the draft's vocabulary, on the second device alone: the first is the worker's while the loop runs, so the
    /// group's own update (which asks every device) is not used.
    fn chain_vocabulary(&mut self) -> Result<()> {
        let Some((pending, reset)) = self.draft_vocabulary_pending() else { return Ok(()) };
        let last = self.group.backends.len() - 1;
        self.with_operands(|model| Ok(vec![model.projection("output.weight")?]), |group, operands| {
            group.backends[last].draft_vocabulary(&operands[0], crate::model::DRAFT_ROWS, &pending, reset)
        })
    }

    /// The catch-up of `tokens` (their hidden rows `hidden`, the first pair at `position`), then `forced` fed in turn: the chain's first
    /// prediction is the token after them all.
    fn chain_open(&mut self, link: &mut Link, tools: &Tools, hidden: &[f32], tokens: &[u32], position: u64, forced: &[u32]) -> Result<Chain> {
        // (the rows of the tokens noted since the last chain go into the draft's vocabulary first, as the ordinary draft does: it lives on the
        // second device, which this loop drives)
        self.chain_vocabulary()?;
        let mut clock = Instant::now();
        let embedded = self.token_embeddings(tokens)?;
        let step = self.chain_step(link, tools, &embedded, Some(hidden), 0, position, forced.is_empty(), 0)?;
        self.phase("draft: catch-up and first draft", &mut clock);
        let mut chain = Chain { first: None, last: *tokens.last().unwrap_or(&0), previous: tokens.len() as u64, position: position + tokens.len() as u64 };
        if forced.is_empty() {
            chain.first = Some(Self::predicted(tools, step, chain.position)?);
            return Ok(chain);
        }
        for (index, token) in forced.iter().enumerate() {
            let embedded = self.token_embeddings(&[*token])?;
            Self::chain_fed(tools, &[*token]);
            let step = self.chain_step(link, tools, &embedded, None, chain.previous, chain.position, index + 1 == forced.len(), 1)?;
            chain.last = *token;
            (chain.previous, chain.position) = (1, chain.position + 1);
            if index + 1 == forced.len() {
                chain.first = Some(Self::predicted(tools, step, chain.position)?);
            }
        }
        self.phase("draft: forced steps", &mut clock);
        Ok(chain)
    }

    /// As `chain_open`, with the forced tokens joining the catch-up's batch: each is fed with `stash`'s row for it (the hidden state the
    /// drafter made it with, when it drafted it) and not the output of the step before, so that the whole is one step of the drafter (more, past
    /// `PARTIAL_ROWS` rows) in place of the catch-up and one step for each forced token. The rows of the real pairs are as before; the forced
    /// tokens' hidden states are the drafter's own from before this pass's verdict, as they were in the steps (those it chained from the
    /// catch-up's output are the ones that change). The drafts are only proposals: what is kept is what the target says.
    fn chain_open_stashed(&mut self, link: &mut Link, tools: &Tools, hidden: &[f32], tokens: &[u32], position: u64, forced: &[u32], stash: &[f32]) -> Result<Chain> {
        self.chain_vocabulary()?;
        let mut clock = Instant::now();
        let rows: Vec<f32> = hidden.iter().chain(stash).copied().collect();
        let all: Vec<u32> = tokens.iter().chain(forced).copied().collect();
        Self::chain_fed(tools, forced);
        let mut chain = Chain { first: None, last: 0, previous: 0, position };
        let mut at = 0;
        while at < all.len() {
            let end = (at + PARTIAL_ROWS).min(all.len());
            let last = end == all.len();
            let embedded = self.token_embeddings(&all[at..end])?;
            let step = self.chain_step(link, tools, &embedded, Some(&rows[at * WIDTH..end * WIDTH]), 0, position + at as u64, last, 0)?;
            chain.last = all[end - 1];
            (chain.previous, chain.position) = ((end - at) as u64, position + end as u64);
            if last {
                chain.first = Some(Self::predicted(tools, step, chain.position)?);
            }
            at = end;
        }
        self.phase("draft: catch-up, drafts fed and first draft", &mut clock);
        Ok(chain)
    }

    /// The chain's next prediction.
    fn chain_next(&mut self, link: &mut Link, tools: &Tools, chain: &mut Chain) -> Result<(u32, f32)> {
        if let Some(prediction) = chain.first.take() {
            chain.last = prediction.0;
            return Ok(prediction);
        }
        let mut clock = Instant::now();
        let embedded = self.token_embeddings(&[chain.last])?;
        Self::chain_fed(tools, &[chain.last]);
        let step = self.chain_step(link, tools, &embedded, None, chain.previous, chain.position, true, 2)?;
        self.phase("draft: next draft", &mut clock);
        (chain.previous, chain.position) = (1, chain.position + 1);
        let prediction = Self::predicted(tools, step, chain.position)?;
        chain.last = prediction.0;
        Ok(prediction)
    }

    /// The drafts the chain goes on with while the product of their probabilities stays at the cut (the first draft's own, `at` 1), at
    /// most `limit`.
    /// With `stashing`, the drafter's own hidden state at each draft comes with them (a WIDTH row a draft), for the pass they go in to
    /// be fed with (see `chain_open_stashed`): taken from the second device's workspace as each step ends.
    fn chain_drafts(&mut self, link: &mut Link, tools: &Tools, chain: &mut Chain, limit: usize, stashing: bool) -> Result<(Vec<u32>, Vec<f32>)> {
        let mut drafts = Vec::new();
        let mut stash = Vec::new();
        let mut product = 1.0f32;
        while drafts.len() < limit {
            let stepped = chain.first.is_none();
            let (token, probability) = self.chain_next(link, tools, chain)?;
            product *= probability;
            if product < self.draft_cut(false, drafts.len() + 1) {
                break;
            }
            drafts.push(token);
            if stashing && stepped {
                let last = self.group.backends.len() - 1;
                let started = Instant::now();
                stash.extend(self.group.backends[last].draft_rows(48, 0, 1)?);
                let mut stats = tools.stats.borrow_mut();
                let slot = &mut stats[usize::from(tools.started.get())][3];
                slot[0] += 1.0;
                slot[1] += started.elapsed().as_secs_f64();
            }
        }
        Ok((drafts, stash))
    }

    /// The drafts of the pass about to be made: the width chooser's plan (held by STRAYDOG_PIPELINE_DRAFTS, which pins it), cut to what a pass takes.
    fn pipelined_width(&mut self, speculating: bool) -> (usize, Option<width::Plan>) {
        let most = self.speculative.min(PARTIAL_ROWS - 1);
        // (passes run ahead keep to one draft, whose cost was measured: each further draft is two steps of the chain on the second device's
        // critical path; the ordinary order's passes take what the chooser plans from the passes of that order alone)
        if speculating {
            // (STRAYDOG_PIPELINE_DRAFTS names another width for them, to measure)
            return (self.pipeline.drafts.unwrap_or(1).min(most), None);
        }
        match self.width.as_mut() {
            Some(chooser) => {
                let plan = chooser.plan();
                (plan.drafts.min(most), Some(plan))
            }
            None => (self.drafts.min(most), None),
        }
    }

    /// STRAYDOG_STATE_DIGEST=1 (a test's): after a request, the digest of the sequence state it left, the drafter's K/V rows (layer 48's, the
    /// row before the last excluded: its token is not in the sequence yet) apart from the rest; `lines`: each part's digest of the drafter's on a line of its own, `all`: of every part (to tell where two runs differ). The next request, or a parked conversation's, drafts from the rows the drafter holds there.
    pub(super) fn report_state_digest(&mut self) {
        use sha2::Digest;
        let Some(mode) = std::env::var("STRAYDOG_STATE_DIGEST").ok().filter(|mode| !mode.is_empty() && mode != "0") else { return };
        match self.state_digest() {
            Ok(lines) => {
                let digest = |wanted: bool| {
                    let mut hash = sha2::Sha256::new();
                    let mut parts = 0;
                    for line in lines.iter().filter(|line| line.contains(":blk.48.") == wanted) {
                        hash.update(line.as_bytes());
                        hash.update(b"\n");
                        parts += 1;
                    }
                    (format!("{:x}", hash.finalize()), parts)
                };
                let (drafter, drafter_parts) = digest(true);
                let (rest, rest_parts) = digest(false);
                eprintln!("Straydog state digest after the request: history {}, drafter layer 48 {} ({drafter_parts} parts), the rest {} ({rest_parts} parts)",
                          self.history.len(), &drafter[..16], &rest[..16]);
                if mode == "lines" || mode == "all" {
                    for line in lines.iter().filter(|line| mode == "all" || line.contains(":blk.48.")) {
                        eprintln!("Straydog state digest line: {line}");
                    }
                }
            }
            Err(error) => eprintln!("Straydog state digest failed: {error}"),
        }
    }

    /// The guess of the bonus token, made wrong when the test asks (every Nth guess).
    fn guess(&self, token: u32, made: &mut usize) -> u32 {
        *made += 1;
        if self.pipeline.force_miss > 0 && *made % self.pipeline.force_miss == 0 {
            (token + 1) % VOCABULARY as u32
        } else {
            token
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_pipelined(&mut self, logits: &[f32], max_new: usize, sampling: &Sampling, rng: &mut Rng,
                                     cancelled: &dyn Fn() -> bool, started: Instant, emit: &mut dyn FnMut(u32)) -> Result<()> {
        let mut generated = Generated::new(VOCABULARY);
        let mut token = constraint::pick(logits, sampling, &generated, rng, &mut None)?;
        generated.insert(token);
        if END_TOKENS.contains(&token) {
            return Ok(());
        }
        self.last_metrics.insert("ttft_seconds".into(), json!(started.elapsed().as_secs_f64()));
        for (key, value) in [("generated_tokens", 1), ("speculative_passes", 0), ("drafted_tokens", 0), ("accepted_tokens", 0),
                             ("pipeline_passes_continued", 0), ("pipeline_passes_given_up", 0), ("pipeline_bonus_snapshot_restores", 0), ("pipeline_bonus_recovered_layers", 0), ("pipeline_snapshot_restore_fallbacks", 0), ("pipeline_replay_restores", 0), ("pipeline_guesses_made", 0), ("pipeline_guesses_right", 0)] {
            self.last_metrics.insert(key.into(), json!(value));
        }
        emit(token);
        let mtp_hidden = self.final_hidden.clone().ok_or("Straydog has no hidden state to draft from")?;
        let split = self.group.split().ok_or("Straydog pipelined passes need two devices")?;
        // SAFETY of the borrows: operands point into the weights, which outlive the group (as in `with_operands`).
        let ple: Vec<Operand<'static>> = unsafe { std::mem::transmute(self.ple_operands()?) };
        let mut tools = unsafe {
            Tools { head: std::mem::transmute(self.head_operands()?), draft_with_head: std::mem::transmute(self.draft_operands(true)?),
                    draft_without_head: std::mem::transmute(self.draft_operands(false)?), lower_head: None,
                    stats: Default::default(), started: Default::default(), drawing: Default::default() }
        };
        if self.pipeline.head_share {
            tools.lower_head = self.group.pin_head(&tools.draft_with_head[7])?;
        }
        // Partial models also carry a default profile with every=false. Only an
        // explicit profile enables the native head's timing events.
        let last = self.group.backends.len() - 1;
        let head_timing = if self.pipeline.head_share && self.profile.as_ref().is_some_and(|profile| profile.every) {
            let before = self.group.backends[last].head_timing()?;
            self.group.backends[last].head_timing_enable(true)?;
            Some(before)
        } else { None };
        let abort = self.group.backends[0].span_abort();
        abort.raise(false);
        // (the first device goes to the worker by value, a vacant backend in its place: nothing of the group reaches it from here until it is back)
        let ordinal = self.group.backends[0].ordinal;
        let device = std::mem::replace(&mut self.group.backends[0], Backend::vacant(ordinal));
        let weights = self.weights.clone();
        let (outcome, returned) = std::thread::scope(|scope| {
            let (command_tx, command_rx) = channel();
            let (job_tx, job_rx) = channel();
            let (reply_tx, reply_rx) = channel();
            let worker = scope.spawn(move || front_worker(device, ple, split, job_rx, reply_tx));
            let relaying = scope.spawn(move || relay(weights, command_rx, job_tx));
            let mut link = Link { commands: command_tx, fronts: FrontReplies::new(reply_rx), abort: abort.clone(),
                                  aborting: false, restoring: false, loaned: false };
            let outcome = self.pipelined_passes(&mut link, &tools, split, &mut generated, &mut token, mtp_hidden, max_new, sampling, rng,
                                                cancelled, emit);
            let stopped = link.shutdown();
            let (relayed, relay_backend) = relaying.join()
                .unwrap_or_else(|_| (Err("the first device's relay thread panicked".into()), None));
            let joined = worker.join();
            let (late, retained) = link.fronts.after_shutdown();
            let outcome = finish_device_work(finish_device_work(finish_device_work(outcome, stopped), relayed), late);
            let (outcome, returned) = match joined {
                Ok(device) => (outcome, device.or(relay_backend).or(retained)),
                Err(_) => (finish_device_work(outcome, Err("the first device's thread panicked".into())), relay_backend.or(retained)),
            };
            abort.raise(false);
            (outcome, returned)
        });
        // The first device is back, from the worker's thread or a retained Yield reply.
        match returned {
            Some(device) => drop(std::mem::replace(&mut self.group.backends[0], device)),
            None => {
                let failed = finish_device_work(outcome, Err("the first device's thread ended without the device".into()));
                let timing = self.pipeline_head_timing(head_timing);
                return finish_device_work(failed, timing);
            }
        }
        let closed = self.group.backends[0].end_verify_window()
            .map_err(|error| format!("could not close the first device's verification window: {error}; pipeline state is unusable"));
        let timing = self.pipeline_head_timing(head_timing);
        finish_device_work(finish_device_work(outcome, closed), timing)
    }

    fn pipeline_head_timing(&mut self, before: Option<([u64; 2], [f64; 6])>) -> Result<()> {
        let Some((calls, milliseconds)) = before else { return Ok(()) };
        let last = self.group.backends.len() - 1;
        let after = self.group.backends[last].head_timing();
        let disabled = self.group.backends[last].head_timing_enable(false);
        let (after_calls, after_ms) = finish_device_work(after, disabled)?;
        for (index, key) in ["pipeline_head_native_single_count", "pipeline_head_native_split_count"].iter().enumerate() {
            self.metric_count(key, after_calls[index].saturating_sub(calls[index]));
        }
        for (index, key) in ["pipeline_head_single_projection_seconds", "pipeline_head_single_device_seconds", "pipeline_head_split_local_projection_seconds",
                             "pipeline_head_split_peer_projection_seconds", "pipeline_head_split_device_seconds", "pipeline_head_split_residual_seconds"].iter().enumerate() {
            self.metric_add(key, (after_ms[index] - milliseconds[index]) / 1e3);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn pipelined_passes(&mut self, link: &mut Link, tools: &Tools, split: usize, generated: &mut Generated, token: &mut u32,
                        mut mtp_hidden: Vec<f32>, max_new: usize, sampling: &Sampling, rng: &mut Rng, cancelled: &dyn Fn() -> bool,
                        emit: &mut dyn FnMut(u32)) -> Result<()> {
        let mut produced = 1usize;
        // The MTP layer pairs each hidden state with the token after it, at the hidden's position.
        let mut mtp_tokens = vec![*token];
        let mut mtp_position = self.history.len() as u64 - 1;
        if let Some(chooser) = self.width.as_mut() {
            chooser.begin_request();
        }
        let (mut parity, mut guesses) = (0usize, 0usize);
        let mut current: Option<Current> = None;
        let mut clock_period = Instant::now();
        let mut gate = Gate::new(self.pipeline.gate);
        let mut costs = Costs::default();
        let mut roles = Roles::default();
        let profiling = self.profile.as_ref().is_some_and(|profile| profile.every);
        // (by what the pass was: one that follows a pass, one of a single row after a wrong guess, another begun after one)
        let mut devices: [Devices; 3] = Default::default();
        // (the expert slots' swaps are held while passes are run ahead; settled in the ordinary order they go on as in the ordinary loop)
        let mut slots_held = !self.pipeline.free_slots;
        if self.pipeline.free_slots {
            self.on_first_device(link, |model| model.group.hold_slots(false))?;
        }
        // (the last verdict gave up the pass behind it: the next pass is sent at once with its known row)
        let mut after_miss = false;
        while produced < max_new {
            // (a request cancelled while a pass is in flight is noticed at its verdict)
            if current.is_none() && cancelled() {
                break;
            }
            let test_start = self.pipeline.hand_test == 1 && produced == 1;
            let test_zero = self.pipeline.hand_test == 2;
            if current.is_none() && self.pipeline.handover && (test_start || test_zero || (gate.hands_over() && !gate.open())) {
                // Nothing is in flight and both devices are idle: the gate wants the ordinary order, so the ordinary loop takes the request for a
                // stretch (its chained passes, copy drafts, assist tuner and swaps) and hands it back at a pass's end, for the probe of the pipeline.
                let hand = Hand { generated: std::mem::replace(generated, Generated::new(VOCABULARY)), token: *token, produced,
                                  mtp_hidden: std::mem::take(&mut mtp_hidden), mtp_tokens: std::mem::take(&mut mtp_tokens), mtp_position };
                let stop = if test_start { usize::MAX } else if test_zero { produced } else { (produced + gate.stretch_tokens()).min(max_new) };
                let held = self.pipeline.hand_held;
                // (the first device comes back from its worker for the stretch, which runs the whole group, and goes to it again after)
                let (handed, before, stretching) = self.on_first_device(link, |model| {
                    model.group.backends[0].end_verify_window()?;
                    if !held {
                        model.pipeline_live_for(false)?;
                    }
                    // (the first passes of a stretch are left out of what the gate is told it costs: the ordinary loop starts from a pipelined state, as
                    // the pipeline's mode does from the ordinary one's: `Gate::skip`)
                    let warm = if test_start || test_zero { None } else { Some((produced + Gate::STRETCH_WARM_UP).min(stop)).filter(|warm| *warm < stop) };
                    let mut hand = Some(hand);
                    let mut handed = Ok(None);
                    if let Some(warm) = warm {
                        handed = model.generate_speculative(&[], max_new, sampling, rng, cancelled, Instant::now(), emit, None, hand.take(), Some(warm));
                        if let Ok(Some(back)) = handed {
                            hand = Some(back);
                            handed = Ok(None);
                        }
                    }
                    let (before, stretching) = (hand.as_ref().map_or(produced, |hand| hand.produced), Instant::now());
                    if hand.is_some() {
                        handed = model.generate_speculative(&[], max_new, sampling, rng, cancelled, stretching, emit, None, hand.take(), Some(stop));
                    }
                    if !held {
                        model.pipeline_live_for(true)?;
                    }
                    Ok((handed, before, stretching))
                })?;
                slots_held = !self.pipeline.free_slots;
                match handed? {
                    Some(back) => {
                        *generated = back.generated;
                        *token = back.token;
                        (produced, mtp_hidden, mtp_tokens, mtp_position) = (back.produced, back.mtp_hidden, back.mtp_tokens, back.mtp_position);
                        gate.stretch_done(stretching.elapsed().as_secs_f64(), produced - before);
                        self.metric_count("pipeline_stretch_tokens", (produced - before) as u64);
                        self.metric_count("pipeline_stretches", 1);
                        clock_period = Instant::now();
                        if !test_zero {
                            continue;
                        }
                    }
                    None => {
                        // (the ordinary loop ended the request, and caught the drafter up on what it left)
                        mtp_tokens.clear();
                        break;
                    }
                }
            }
            let continued = current.is_some();
            let (mut chain_time, mut restore_time) = (0.0f64, 0.0f64);
            let Current { pass, spec, guess, lone } = match current.take() {
                Some(current) => current,
                None => {
                    // (nothing is in flight: the first device is idle)
                    if !self.pipeline.free_slots && slots_held == gate.settled_plain() {
                        slots_held = !slots_held;
                        self.on_first_device(link, |model| model.group.hold_slots(slots_held))?;
                    }
                    let started = Instant::now();
                    tools.started.set(true);
                    let current = self.pipelined_start(link, tools, token, &mtp_hidden, &mtp_tokens, mtp_position, produced, max_new, sampling, rng, generated,
                                                       &gate, &mut parity, &mut guesses, after_miss)?;
                    after_miss = false;
                    chain_time += started.elapsed().as_secs_f64();
                    current
                }
            };
            let rows = pass.rows.len();
            let drafts = &pass.rows[1..];
            // The back: the first device's rows through the second device's span and the head.
            let mut clock = Instant::now();
            let front = link.front()?;
            let front_wait = clock.elapsed().as_secs_f64();
            let front_time = front.time;
            let (prepared, handoff) = (front.prepared, front.handoff);
            if let Some(profile) = self.profile.as_mut() {
                profile.add("front job: embeddings and PLE rows (beside the devices)", front.prepared);
            }
            self.phase("verify: layers (first device)", &mut clock);
            let last = self.group.backends.len() - 1;
            let before = profiling.then(|| timers(&self.group.backends[last]));
            // (the rows stay in the second device's workspace: the head reads them there and hands them back with its answer, one wait)
            self.group.backends[last].verify_span_kept(split, 48, &front.hidden, pass.position as u64)?;
            let span_wall = clock.elapsed().as_secs_f64();
            let span_timers = before.map(|before| {
                let after = timers(&self.group.backends[last]);
                [after[0] - before[0], after[1] - before[1], after[2] - before[2], after[3] - before[3]]
            });
            self.phase("verify: layers (second device)", &mut clock);
            // (a draw that reads a row's greatest logits alone has them from the device, penalized or not: a temperature with a small top_k
            // and no penalty read the whole rows here, 1 MB a row to the host and a softmax over the vocabulary)
            let top = if sampling.presence_penalty != 0.0 || sampling.top_candidates().is_some() { sampling.target_read_top().unwrap_or(0) } else { 0 };
            let ((targets, mut scores, _), hidden) = self.group.backends[last].output_head_rows(&tools.head, crate::gpu::Rows::Layer(47), rows, sampling.needs_logits(),
                                                                                               top, false, None)?;
            let mut head_wall = clock.elapsed().as_secs_f64();
            self.phase("verify: head", &mut clock);
            let accepting = Instant::now();
            let certified = if let Some(Scores::Top(top)) = scores.as_ref() {
                self.try_penalty_top(drafts, &[], top, sampling, generated, rng, pass.position as u64, true)
            } else { None };
            let mut fallback_head_wall = 0.0;
            if matches!(scores.as_ref(), Some(Scores::Top(_))) && certified.is_none() {
                self.phase("accept", &mut clock);
                // Layer 47's retained rows are still live: do not queue a draft or replay a model layer before this head-only read.
                let rereading = Instant::now();
                let ((_, full, _), _) = self.group.backends[last].output_head_rows(&tools.head, crate::gpu::Rows::Layer(47), rows,
                                                                                true, 0, false, None)?;
                scores = match full {
                    Some(full @ Scores::Logits(_)) => Some(full),
                    _ => return Err("Straydog pipeline head returned no full logits".into()),
                };
                fallback_head_wall = rereading.elapsed().as_secs_f64();
                head_wall += fallback_head_wall;
                self.phase("verify: head", &mut clock);
            }
            let (accepted, following) = match certified {
                Some(accepted) => accepted,
                None => match scores {
                    Some(Scores::Logits(logits)) => {
                        let (seed, position) = (self.sample_seed, pass.position as u64);
                        sampling::accept_keyed(drafts, &logits, VOCABULARY, sampling, generated, &|index| Rng::keyed(seed, position + index as u64))
                    }
                    _ => Self::accept(drafts, &[], &targets, None, sampling, generated, rng),
                },
            };
            let accept_wall = (accepting.elapsed().as_secs_f64() - fallback_head_wall).max(0.0);
            self.phase("accept", &mut clock);
            if self.pipeline.pass_log() {
                eprintln!("Straydog pass: position {} rows {:?} accepted {accepted} following {following}", pass.position, pass.rows);
            }
            generated.insert(following);
            let mut emitted: Vec<u32> = drafts[..accepted].iter().copied().chain([following]).collect();
            let mut kept = accepted + 1;
            if let Some(index) = emitted.iter().position(|value| END_TOKENS.contains(value)) {
                kept = kept.min(index + 1);
                emitted.truncate(index);
            }
            let ended = emitted.len() < accepted + 1;
            // The guess holds: every draft kept, the bonus token the next pass begins with, nothing ending the reply or the request.
            let speculated = spec.is_some();
            let guessed = spec.as_ref().map(|spec| spec.rows[0]).or(guess);
            let would_hold = guessed.is_some_and(|guess| accepted == drafts.len() && following == guess) && !ended;
            let guess_right = guessed.map(|guessed| accepted == drafts.len() && following == guessed);
            if let Some(guessed) = guessed {
                // (a guess made with nothing sent behind it, the gate being shut, counts too: it is how the gate opens again)
                let right = accepted == drafts.len() && following == guessed;
                gate.record(right);
                self.metric_count("pipeline_guesses_made", 1);
                self.metric_count("pipeline_guesses_right", u64::from(right));
            }
            let holds = spec.is_some() && would_hold && produced + emitted.len() < max_new && !cancelled();
            if spec.is_some() {
                self.metric_count(if holds { "pipeline_passes_continued" } else { "pipeline_passes_given_up" }, 1);
                if !holds {
                    // (the first device stops the pass it is on at its next layer while the rest of this one is settled)
                    link.cancel()?;
                }
            }
            // The tokens go out, and the history and the first device's states follow.
            for value in &emitted {
                produced += 1;
                self.metric_count("generated_tokens", 1);
                self.metric_count("decode_forward_tokens", 1);
                self.update_rate();
                emit(*value);
            }
            self.final_hidden = Some(hidden[(kept - 1) * WIDTH..kept * WIDTH].to_vec());
            let mut clock = Instant::now();
            let mut following_pass = None;
            let commit_wall;
            if holds {
                // The guess holds: the next pass is this pass's (its front under way), and the one behind it is made from what this pass
                // kept, its rows' hidden states and tokens, with the next pass's drafts fed in turn.
                let spec = spec.unwrap();
                let chaining = Instant::now();
                tools.started.set(false);
                let next = self.pipelined_follow(link, tools, &gate, &pass, &spec, &hidden, &emitted, produced, max_new, &mut parity,
                                                 &mut guesses, sampling, generated)?;
                chain_time += chaining.elapsed().as_secs_f64();
                following_pass = Some(Current { pass: spec, spec: next, guess: None, lone: false });
                let committing = Instant::now();
                self.group.verify_commit(split, 48, rows as u64)?;
                commit_wall = committing.elapsed().as_secs_f64();
            } else {
                let committing = Instant::now();
                self.group.verify_commit(split, 48, kept as u64)?;
                commit_wall = committing.elapsed().as_secs_f64();
                // (the first device, if a pass was given to it behind this one, or this one's rows were not all kept)
                let given_up = spec.is_some();
                after_miss = given_up;
                let restoring = Instant::now();
                if given_up {
                    link.give_up()?;
                }
                if given_up || kept < rows {
                    let recover = spec.as_ref().filter(|next| {
                        self.pipeline.recover && next.position == pass.position + rows &&
                            bonus_missed(accepted, drafts.len(), kept, rows, ended, cancelled(), following, next.rows[0])
                    }).map(|next| Recovery { parity: next.parity, position: next.position, rows: next.rows.len() });
                    if let Some(touched) = link.restore(RestoreJob { parity: pass.parity, kept, force: given_up, recover, first: front.first,
                                                 embeddings: front.embeddings })? {
                        self.metric_count("pipeline_bonus_snapshot_restores", 1);
                        self.metric_count("pipeline_bonus_recovered_layers", u64::from(touched.count_ones()));
                    } else {
                        self.metric_count("pipeline_replay_restores", 1);
                        if recover.is_some() {
                            self.metric_count("pipeline_snapshot_restore_fallbacks", 1);
                        }
                    }
                }
                restore_time = restoring.elapsed().as_secs_f64();
            }
            self.phase("commit: DeltaNet replay", &mut clock);
            self.history.extend_from_slice(&pass.rows[..kept]);
            self.draft_note(&pass.rows[..kept]);
            let seconds = clock_period.elapsed().as_secs_f64();
            clock_period = Instant::now();
            // (what became of the pass: its guess held, was wrong with every draft kept, or a draft was rejected; or none was made)
            let outcome = if !speculated { 3 } else if holds { 0 } else if would_hold { 4 } else if accepted == drafts.len() { 1 } else { 2 };
            gate.pass_done(seconds, emitted.len(), continued || speculated);
            costs.add(outcome, !continued, seconds, emitted.len(), rows, restore_time, chain_time);
            roles.add(if continued { 0 } else if lone { 1 } else if !speculated && guessed.is_none() { 2 } else { 3 }, guess_right, seconds, emitted.len());
            if let (true, Some(a), Some(b)) = (profiling, front_time, span_timers) {
                let devices = &mut devices[if continued { 0 } else if rows == 1 { 1 } else { 2 }];
                devices.passes += 1.0;
                for (slot, amount) in devices.a.iter_mut().zip(a) {
                    *slot += amount;
                }
                for (slot, amount) in devices.b.iter_mut().zip([span_wall, b[0], b[1], b[2], b[3]]) {
                    *slot += amount;
                }
                devices.head += head_wall;
                devices.chain += chain_time;
                devices.front_wait += front_wait;
                devices.accept += accept_wall;
                devices.commit += commit_wall;
                devices.prepared += prepared;
                devices.period += seconds;
                if let Some(handoff) = handoff {
                    devices.handoff.0 += handoff;
                    devices.handoff.1 += 1.0;
                }
            }
            // What the pass cost and kept goes to the windows the drafts are chosen by (its period here: the cost of a pass as the pipeline runs it).
            if let (Some(chooser), Some((plan, offered)), false) = (self.width.as_mut(), pass.plan, continued || speculated) {
                if let Some(outcome) = chooser.record(plan, offered, rows, accepted, seconds) {
                    if outcome.worth_saying() {
                        eprintln!("{}", outcome.line());
                    }
                    if outcome.chosen != self.drafts {
                        self.drafts = outcome.chosen;
                        self.copy_drafts = outcome.chosen;
                    }
                }
            }
            self.metric_count("speculative_passes", 1);
            self.metric_count("drafted_tokens", drafts.len() as u64);
            self.metric_count("accepted_tokens", accepted as u64);
            self.metric_add("decode_seconds", seconds);
            if let Some(mean) = self.pass_seconds.get_mut(rows.wrapping_sub(1)) {
                *mean = if *mean > 0.0 { *mean + (seconds.min(3.0 * *mean) - *mean) / 16.0 } else { seconds };
            }
            if let Some(count) = self.pass_counts.get_mut(rows.wrapping_sub(1)) {
                *count += 1;
            }
            if let Some(profile) = self.profile.as_mut() {
                profile.end_pass(rows, seconds);
            }
            *token = following;
            mtp_hidden = hidden[..emitted.len().min(kept) * WIDTH].to_vec();
            mtp_tokens = emitted;
            mtp_position = pass.position as u64;
            if ended {
                break;
            }
            current = following_pass;
        }
        // The pairs no pass caught up on, now: else the MTP layer's rows from the last pass up to the sequence's last would keep what was
        // there before, which every later draft attends to. (The first device is idle: nothing is in flight here.)
        link.give_up()?;
        if !slots_held && !self.pipeline.free_slots {
            self.on_first_device(link, |model| model.group.hold_slots(true))?;
        }
        let count = |key: &str| self.last_metrics.get(key).and_then(Value::as_u64).unwrap_or(0);
        eprintln!("Straydog pipelined verification: passes in the pipeline's mode {} of {}; {} passes, guesses right {} of {} made (the gate's rate, the shut \
                   gate's included), passes run ahead {} of which {} continued and {} given up, {} drafts offered, {} kept{}",
                  gate.passes[1], gate.passes[0] + gate.passes[1],
                  count("speculative_passes"), count("pipeline_guesses_right"), count("pipeline_guesses_made"),
                  count("pipeline_passes_continued") + count("pipeline_passes_given_up"), count("pipeline_passes_continued"),
                  count("pipeline_passes_given_up"), count("drafted_tokens"), count("accepted_tokens"), costs.lines() + &roles.line());
        for (key, value) in roles.metrics() {
            self.last_metrics.insert(key, json!(value));
        }
        if profiling {
            let stats = tools.stats.borrow();
            for (name, devices) in ["passes that follow a pass", "repair rows (one row after a wrong guess)", "other passes begun after a wrong guess or the start"].iter().zip(&devices) {
                if devices.passes > 0.0 {
                    eprintln!("Straydog pipelined devices, {name}: {}", devices.line());
                }
            }
            let started = (devices[1].passes + devices[2].passes).max(1.0);
            eprintln!("Straydog pipelined drafter, a pass that follows a pass: {}; a pass begun after a wrong guess or the start: {}",
                      drafter_line(&stats[0], devices[0].passes), drafter_line(&stats[1], started));
        }
        if !mtp_tokens.is_empty() {
            self.on_first_device(link, |model| model.draft(&mtp_hidden, &mtp_tokens, mtp_position, 0, sampling, rng, None).map(drop))?;
        }
        Ok(())
    }

    /// The first pass after the ordinary order: the drafts of the sequence's next token from the chain, the pass sent to the first device
    /// and, when it can be, the pass behind it, from the same chain (the drafter's next prediction is the guess of this pass's bonus
    /// token), sent after it. Nothing is in flight.
    #[allow(clippy::too_many_arguments)]
    fn pipelined_start(&mut self, link: &mut Link, tools: &Tools, token: &u32, hidden: &[f32], tokens: &[u32], position: u64, produced: usize,
                       max_new: usize, sampling: &Sampling, rng: &mut Rng, generated: &Generated, gate: &Gate, parity: &mut usize,
                       guesses: &mut usize, lone: bool) -> Result<Current> {
        self.chain_drawing(tools, sampling, generated);
        let room = self.context.saturating_sub(self.history.len() + 1);
        // After a wrong guess the pass goes to the first device at once, with the one row that is known (the token the verdict gave) and no
        // drafts: the drafter's steps then run on the second device while the first device works, and the pass behind it is made from their
        // result (the catch-up's prediction is the guess of this pass's bonus token). A miss then costs a lone row, not the chain before a send.
        let lone = lone && self.pipeline.lone && gate.open() && produced + 1 < max_new;
        let (width, plan) = self.pipelined_width(gate.open());
        let limit = if lone { 0 } else { width.min(max_new - produced - 1).min(room) };
        let plan = plan.map(|plan| (plan, limit));
        // The K/V grows for what the ordinary loop would run now (this pass's rows and the drafter's steps for them) and no more: a pass behind this
        // one is sent only where the K/V already holds it (below), else the loop goes on one pass at a time and the next start grows it.
        // (both grow what the group holds, the first device's too: it comes back from its worker for them, nothing being in flight)
        self.on_first_device(link, |model| {
            model.ensure_kv(model.history.len() + 1 + limit)?;
            model.protect(model.history.len(), (model.history.len() + 1 + limit + 2 * PARTIAL_ROWS).min(model.context))
        })?;
        let early_sent = lone;
        let mut early = if lone { Some(self.pipelined_send(link, vec![*token], self.history.len(), None, parity, None)?) } else { None };
        // (STRAYDOG_PIPELINE_DRAFTS=0, a test's: one-row passes run ahead all the same, the same shapes as the ordinary loop's)
        if limit == 0 && !lone && !(self.pipeline.drafts() == Some(0) && gate.open()) {
            // (the catch-up alone: a pass of one row)
            self.draft(hidden, tokens, position, 0, sampling, rng, None)?;
            let pass = self.pipelined_send(link, vec![*token], self.history.len(), None, parity, plan)?;
            return Ok(Current { pass, spec: None, guess: None, lone: false });
        }
        let mut chain = self.chain_open(link, tools, hidden, tokens, position, &[])?;
        let mut drafts = Vec::new();
        let mut product = 1.0f32;
        // The drafts while their product stays at the cut. A prediction the cut refuses is the guess of the bonus token.
        let mut guess = None;
        while drafts.len() < limit {
            let (next, probability) = self.chain_next(link, tools, &mut chain)?;
            product *= probability;
            if product < self.draft_cut(false, drafts.len() + 1) {
                guess = Some(next);
                break;
            }
            drafts.push(next);
        }
        // (the pass goes to the first device now, as soon as its own drafts are there, and the drafter goes on while it runs: the guess of
        // the bonus token is the step after the last draft, which the pass does not wait for -- a drafting step on the second device, 1 ms
        // of two RTX 3090, that the first device then spends on the pass)
        let pass = match early.take() {
            Some(pass) => pass,
            None => {
                let rows: Vec<u32> = std::iter::once(*token).chain(drafts).collect();
                self.pipelined_send(link, rows, self.history.len(), None, parity, plan)?
            }
        };
        // (none is asked for in the ordinary order: the steps are the ordinary loop's)
        if guess.is_none() && gate.open() {
            guess = Some(self.chain_next(link, tools, &mut chain)?.0);
        }
        let after = produced + pass.rows.len();
        let next_room = self.context.saturating_sub(self.history.len() + pass.rows.len() + 1);
        let most = self.speculative.min(PARTIAL_ROWS - 1);
        let guess = guess.map(|guess| self.guess(guess, guesses));
        // (the pass behind it only where the K/V already holds its widest form)
        let room_behind = after < max_new && self.history.len() + pass.rows.len() + 1 + most.min(next_room) + 8 <= self.kv_capacity;
        let spec = if let (Some(guess), true) = (guess, gate.open() && room_behind) {
            let (next_width, next_plan) = self.pipelined_width(true);
            let next_limit = next_width.min(max_new.saturating_sub(after + 1)).min(next_room);
            let (drafts_after, stash) = self.chain_drafts(link, tools, &mut chain, next_limit, self.pipeline.batch)?;
            let rows_after: Vec<u32> = std::iter::once(guess).chain(drafts_after).collect();
            let position = self.history.len() + pass.rows.len();
            let mut behind = self.pipelined_send(link, rows_after, position, Some(&pass.rows), parity, next_plan.map(|plan| (plan, next_limit)))?;
            behind.stash = stash;
            Some(behind)
        } else {
            None
        };
        Ok(Current { pass, spec, guess, lone: early_sent })
    }

    /// The pass behind `spec`, which `pass` has just proved right: the drafter catches up on `pass`'s kept pairs (their real hidden
    /// states), takes `spec`'s drafts in turn and goes on from its bonus guess; the pass goes to the first device behind `spec`'s.
    #[allow(clippy::too_many_arguments)]
    fn pipelined_follow(&mut self, link: &mut Link, tools: &Tools, gate: &Gate, pass: &Pass, spec: &Pass, hidden: &[f32], emitted: &[u32],
                        produced: usize, max_new: usize, parity: &mut usize, guesses: &mut usize, sampling: &Sampling, generated: &Generated)
                        -> Result<Option<Pass>> {
        self.chain_drawing(tools, sampling, generated);
        // `produced` counts what this pass emitted: the next pass emits its own rows at most, and a pass behind it is wanted if more is.
        let history = self.history.len() + pass.rows.len();
        let after = produced + spec.rows.len();
        let room = self.context.saturating_sub(history + spec.rows.len() + 1);
        let most = self.speculative.min(PARTIAL_ROWS - 1);
        if !gate.open() || after >= max_new || history + spec.rows.len() + 1 + most.min(room) + 8 > self.kv_capacity {
            // (nothing is sent behind `spec`, but the real pairs of `pass` still go to the drafter, as they do where a pass follows: the
            // catch-up after `spec`'s verdict takes `spec`'s pairs alone, and the drafter's K/V at `pass`'s positions would keep the
            // rows it predicted them with)
            let embedded = self.token_embeddings(emitted)?;
            self.chain_step(link, tools, &embedded, Some(&hidden[..emitted.len() * WIDTH]), 0, pass.position as u64, false, 0)?;
            return Ok(None);
        }
        let (width, plan) = self.pipelined_width(true);
        let limit = width.min(max_new.saturating_sub(after + 1)).min(room);
        let (real, forced) = (&hidden[..emitted.len() * WIDTH], &spec.rows[1..]);
        let mut chain = if self.pipeline.batch && !forced.is_empty() && spec.stash.len() == forced.len() * WIDTH {
            self.chain_open_stashed(link, tools, real, emitted, pass.position as u64, forced, &spec.stash)?
        } else {
            self.chain_open(link, tools, real, emitted, pass.position as u64, forced)?
        };
        let (guess, _) = self.chain_next(link, tools, &mut chain)?;
        let (drafts, stash) = self.chain_drafts(link, tools, &mut chain, limit, self.pipeline.batch)?;
        let rows: Vec<u32> = std::iter::once(self.guess(guess, guesses)).chain(drafts).collect();
        let position = history + spec.rows.len();
        // (the pass before `spec` is not in the history yet: the tokens before the new rows are `pass`'s and `spec`'s)
        let mut before = pass.rows.clone();
        before.extend_from_slice(&spec.rows);
        let mut behind = self.pipelined_send(link, rows, position, Some(&before), parity, plan.map(|plan| (plan, limit)))?;
        behind.stash = stash;
        Ok(Some(behind))
    }

    /// `rows` at `position` to the first device. `before`: the tokens in flight ahead of them, past the history.
    fn pipelined_send(&mut self, link: &mut Link, rows: Vec<u32>, position: usize, before: Option<&[u32]>, parity: &mut usize,
                      plan: Option<(width::Plan, usize)>) -> Result<Pass> {
        let tail = self.tail_with(before.unwrap_or(&[]));
        let job = self.front_request(&rows, position, &tail, *parity, before.is_some(), link.fronts.in_flight == 0);
        let pass = Pass { rows, position, parity: *parity, plan, stash: Vec::new() };
        *parity ^= 1;
        link.submit(job)?;
        Ok(pass)
    }
}

#[cfg(test)]
mod shutdown_tests {
    use super::*;
    use crate::gpu::HYBRID_WAIT_RECOVERED;

    fn fronts(errors: &[&str]) -> (Result<()>, usize) {
        let (send, receive) = channel();
        for error in errors { assert!(send.send(Reply::Front(Err((*error).into()))).is_ok()); }
        drop(send);
        let mut fronts = FrontReplies::new(receive);
        fronts.in_flight = errors.len();
        let result = drain_fronts(&mut fronts);
        (result, fronts.in_flight)
    }

    #[test]
    fn shutdown_receives_a_later_device_fault_after_a_recovered_wait() {
        let fault = "an illegal memory access was encountered";
        for errors in [[HYBRID_WAIT_RECOVERED, fault], [fault, HYBRID_WAIT_RECOVERED]] {
            assert_eq!(fronts(&errors), (Err(fault.into()), 0));
        }
    }

    #[test]
    fn normal_span_aborts_are_drained_without_becoming_failures() {
        assert_eq!(fronts(&[SPAN_ABORTED, SPAN_ABORTED]), (Ok(()), 0));
        assert_eq!(fronts(&[HYBRID_WAIT_RECOVERED, SPAN_ABORTED]), (Err(HYBRID_WAIT_RECOVERED.into()), 0));
    }
}

#[cfg(test)]
mod front_reply_tests {
    use super::*;
    use crate::gpu::HYBRID_WAIT_RECOVERED;

    fn completed(marker: f32) -> Reply {
        Reply::Front(Ok(FrontOut { first: Vec::new(), hidden: vec![marker], embeddings: Vec::new(), prepared: 0.0, handoff: None, time: None }))
    }

    #[test]
    fn idle_poll_preserves_fifo_fronts_without_double_decrementing() {
        let (send, receive) = channel();
        let mut fronts = FrontReplies::new(receive);
        fronts.in_flight = 2;
        assert!(send.send(completed(1.0)).is_ok());
        assert!(!fronts.idle().unwrap());
        assert_eq!((fronts.in_flight, fronts.cached.len()), (1, 1));
        assert!(send.send(completed(2.0)).is_ok());
        assert!(fronts.idle().unwrap());
        assert_eq!((fronts.in_flight, fronts.cached.len()), (0, 2));
        assert_eq!(fronts.next().unwrap().unwrap().hidden, [1.0]);
        assert_eq!(fronts.next().unwrap().unwrap().hidden, [2.0]);
        assert!(fronts.empty());
        assert!(fronts.next().is_err());
        assert_eq!(fronts.in_flight, 0);
    }

    #[test]
    fn cached_errors_are_drained_and_a_later_fault_stays_fatal() {
        let fault = "an illegal memory access was encountered";
        for cache_all in [false, true] {
            let (send, receive) = channel();
            let mut fronts = FrontReplies::new(receive);
            fronts.in_flight = 2;
            assert!(send.send(Reply::Front(Err(HYBRID_WAIT_RECOVERED.into()))).is_ok());
            if cache_all { assert!(send.send(Reply::Front(Err(fault.into()))).is_ok()); }
            assert!(fronts.idle().is_err());
            if !cache_all { assert!(send.send(Reply::Front(Err(fault.into()))).is_ok()); }
            assert_eq!(drain_fronts(&mut fronts), Err(fault.into()));
            assert!(fronts.empty());
        }
    }

    #[test]
    fn unexpected_controls_stay_fatal_and_do_not_consume_a_front() {
        for control in [Reply::Restored(Ok(None)), Reply::Yielded(Err("unexpected yield".into()))] {
            let (send, receive) = channel();
            let mut fronts = FrontReplies::new(receive);
            fronts.in_flight = 1;
            assert!(send.send(control).is_ok());
            assert!(send.send(Reply::Front(Err(HYBRID_WAIT_RECOVERED.into()))).is_ok());
            let error = fronts.idle().unwrap_err();
            assert!(error.contains("out of order"));
            assert_eq!((fronts.in_flight, fronts.controls.len()), (1, 1));
            assert_eq!(drain_fronts(&mut fronts), Err(error));
            assert!(fronts.empty());
        }
    }

    #[test]
    fn an_unexpected_yield_keeps_the_backend_until_shutdown_has_joined() {
        let (send, receive) = channel();
        let mut fronts = FrontReplies::new(receive);
        // Vacant backends perform no GPU calls, including in Drop.
        assert!(send.send(Reply::Yielded(Ok(Box::new(Backend::vacant(7))))).is_ok());
        assert!(fronts.idle().is_err());
        drop(send);
        let (result, returned) = fronts.after_shutdown();
        assert!(result.unwrap_err().contains("out of order"));
        assert_eq!(returned.unwrap().ordinal, 7);
    }

    #[test]
    fn a_failed_relay_hop_returns_the_unsent_backend_and_a_fatal_error() {
        let (send, receive) = channel();
        drop(receive);
        let forwarded = forward_front_command(&send, Command::Resume(Box::new(Backend::vacant(11))));
        match forwarded {
            Err((error, Some(device))) => {
                assert!(error.contains("thread ended") && !error.ends_with(HYBRID_WAIT_RECOVERED));
                assert_eq!(device.ordinal, 11);
            }
            _ => panic!("a failed Resume must return the actual Backend and an error"),
        }
    }

    #[test]
    fn an_unsolicited_front_does_not_underflow_and_its_error_survives() {
        let (_send, receive) = channel();
        let mut fronts = FrontReplies::new(receive);
        assert!(fronts.receive(Reply::Front(Err(HYBRID_WAIT_RECOVERED.into()))).is_err());
        assert_eq!(fronts.in_flight, 0);
        assert_eq!(fronts.cached.len(), 1);
        let error = drain_fronts(&mut fronts).unwrap_err();
        assert!(error.contains("unsolicited front") && !error.ends_with(HYBRID_WAIT_RECOVERED));
        assert!(fronts.empty());
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    fn on() -> PipelineConfig {
        PipelineConfig { mode: Mode::On, drafts: None, force_miss: 0, cpu_threads: None, gate: 0.3, assist: None, lone: true, free_slots: false,
                         recover: true, batch: false, handover: true, hand_held: false, hand_test: 0, no_copy: false, pass_log: false, head_share: false, keyed_guesses: true }
    }

    #[test]
    fn sharing_is_opt_in_and_requires_an_idle_complete_vocabulary() {
        for value in [None, Some("0"), Some("true"), Some(" 1 ")] { assert!(!head_share_named(value)); }
        assert!(head_share_named(Some("1")));
        assert!(!on().head_share);
        assert!(head_share_eligible(true, true, true, 0, 2));
        for (enabled, idle, lower, extras, rows) in [
            (false, true, true, 0, 2), (true, false, true, 0, 2), (true, true, false, 0, 2),
            (true, true, true, 1, 2), (true, true, true, 0, 1),
        ] {
            assert!(!head_share_eligible(enabled, idle, lower, extras, rows));
        }
    }

    #[test]
    fn the_pipeline_asked_for_where_it_does_not_apply_sets_up_nothing() {
        for devices in [1, 2, 3] {
            let config = on().for_devices(devices);
            if on().enabled(devices) {
                assert!(config.frozen_slots() && config.windows(devices));
            } else {
                // (one GPU, or AMD's: the expert slots keep swapping, no window, the ordinary loop's way)
                assert!(!config.frozen_slots() && !config.windows(devices) && !config.enabled(devices), "{devices} devices");
            }
        }
        // (and the two-GPU NVIDIA case is the one that applies)
        assert_eq!(on().enabled(2), !cfg!(feature = "hip"));
    }

    #[test]
    fn the_pipeline_is_the_default_and_is_named_off_or_frozen() {
        assert!(PipelineConfig::mode_named(None) == Mode::On && PipelineConfig::mode_named(Some("1")) == Mode::On);
        assert!(PipelineConfig::mode_named(Some("0")) == Mode::Off && PipelineConfig::mode_named(Some("off")) == Mode::Off);
        assert!(PipelineConfig::mode_named(Some("freeze")) == Mode::Frozen);
        // (unnamed, the pipeline holds its own assist level; `freeze` and the ordinary loop leave it to the tuner)
        assert_eq!(on().assist(), Some(PIPELINE_ASSIST));
        assert_eq!(PipelineConfig { assist: Some(4), ..on() }.assist(), Some(4));
        assert_eq!(PipelineConfig { mode: Mode::Frozen, ..on() }.assist(), None);
        assert_eq!(on().for_devices(1).assist(), None);
    }

    #[test]
    fn freeze_is_kept_as_asked_for() {
        let freeze = PipelineConfig { mode: Mode::Frozen, ..on() };
        for devices in [1, 2] {
            assert!(freeze.for_devices(devices).frozen_slots());
        }
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    /// A gate that has run its first speculating stretch at `ahead` seconds for each of two tokens.
    fn after_first_run(ahead: f64) -> Gate {
        let mut gate = Gate::new(0.3);
        for _ in 0..(Gate::FIRST_RUN + 2) {
            gate.pass_done(ahead, 2, true);
        }
        gate
    }

    #[test]
    fn a_plain_probe_is_a_stretch_of_the_ordinary_loop_and_the_cheaper_mode_wins() {
        // speculating costs 10 ms for two tokens; the ordinary stretch 12 ms for two: speculating stays
        let mut gate = after_first_run(0.010);
        assert!(gate.probing && !gate.open());
        assert_eq!(gate.stretch_tokens(), 2 * Gate::PROBE);
        gate.stretch_done(0.012 * Gate::PROBE as f64, 2 * Gate::PROBE);
        assert!(gate.open() && !gate.probing);
        // speculating costs 12 ms for two tokens; the ordinary stretch 10 ms: the gate settles in the ordinary order
        let mut gate = after_first_run(0.012);
        gate.stretch_done(0.010 * Gate::PROBE as f64, 2 * Gate::PROBE);
        assert!(gate.settled_plain());
        assert_eq!(gate.stretch_tokens(), 2 * Gate::PLAIN_RUN);
        // the stretch ends: a probe of the pipeline
        gate.stretch_done(0.010 * Gate::PLAIN_RUN as f64, 2 * Gate::PLAIN_RUN);
        assert!(gate.open() && gate.probing);
    }

    #[test]
    fn a_mode_that_wins_by_a_margin_is_probed_less_often() {
        let mut gate = after_first_run(0.010);
        // probe of the ordinary order: 14 ms for two tokens against the pipeline's 10: speculating stays, the next run is twice as long
        gate.stretch_done(0.014 * Gate::PROBE as f64, 2 * Gate::PROBE);
        assert!(gate.open() && !gate.probing);
        assert_eq!(gate.backoff, 2);
        for _ in 0..(Gate::SPECULATING_RUN * 2 - 1) {
            gate.pass_done(0.010, 2, true);
        }
        assert!(gate.open() && !gate.probing, "the run is not over before its doubled length");
        gate.pass_done(0.010, 2, true);
        assert!(!gate.open() && gate.probing);
        // a probe that finds the two nearly as dear leaves the base length
        let mut gate = after_first_run(0.010);
        gate.stretch_done(0.0102 * Gate::PROBE as f64, 2 * Gate::PROBE);
        assert!(gate.open() && gate.backoff == 1);
    }

    #[test]
    fn a_gate_held_open_or_shut_for_a_test_does_not_hand_over() {
        assert!(!Gate::new(0.0).hands_over());
        assert!(!Gate::new(2.0).hands_over());
        assert!(Gate::new(0.3).hands_over());
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;

    #[test]
    fn every_draft_accepted_with_only_a_wrong_bonus_recovers() {
        assert!(bonus_missed(2, 2, 3, 3, false, false, 42, 43));
        assert!(!bonus_missed(1, 2, 2, 3, false, false, 42, 43));
        assert!(!bonus_missed(2, 2, 3, 3, false, false, 42, 42));
        assert!(!bonus_missed(2, 2, 2, 3, true, false, 42, 43));
        assert!(!bonus_missed(2, 2, 3, 3, false, true, 42, 43));
    }

    #[test]
    fn an_abort_before_ple_has_a_front_identity_without_a_ple_snapshot() {
        let front = Recovery { parity: 1, position: 128, rows: 2 };
        let trace = FrontTrace { front: Some(front), epoch: 7, speculative: true, ple_saved: false };
        assert!(trace.matches(front));
        assert!(!trace.ple_saved);
    }

    #[test]
    fn stale_or_unstarted_windows_do_not_prove_snapshot_recovery() {
        let front = Recovery { parity: 1, position: 128, rows: 2 };
        assert!(!FrontTrace::default().matches(front));
        let trace = FrontTrace { front: Some(front), epoch: 7, speculative: true, ple_saved: true };
        for stale in [Recovery { parity: 0, ..front }, Recovery { position: 126, ..front }, Recovery { rows: 1, ..front }] {
            assert!(!trace.matches(stale));
        }
        assert!(!FrontTrace { epoch: 0, ..trace }.matches(front));
    }
}
