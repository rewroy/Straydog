//! How many drafts a partially resident model verifies in a pass, chosen by what passes cost this machine.
//!
//! A pass with a draft reads the layers' dense weights once whatever its rows, so on a machine where the GPU's part of a layer is the
//! longer (an RTX 4060 on a PC with a fast CPU) a second and third row cost little, and drafts pay well; on one where the CPU's experts
//! are the limit (an RTX 5070 Ti at 5.5 GiB) each row doubles the CPU's routes and one draft gains next to nothing. Nothing in the
//! machine's specification says which, so the engine times its passes and keeps what they say.
//!
//! What is kept, from the passes it runs, each as a window of the latest:
//! - the seconds of a pass by its rows (1 to the most), and those of a pass that tried no draft (it still takes the drafter's catch-up, but
//!   not its head and first draft: it costs less than a lone row after a draft that was cut);
//! - by draft position, the share of the passes that could offer a draft in which the chain of drafts reached it (the drawn-draft cut ends
//!   a chain where the drafter is unsure), and the share of the passes that offered it in which it was kept.
//!
//! The chain a pass drafts is the same whatever the width: a width only stops it sooner. So what a pass of k drafts does follows from what the
//! windows hold, whichever widths made them: with p(j) the share of passes whose chain reaches position j (p(0) = 1, none past k) and a(j) the
//! share of drafts offered at j that are kept, a pass has 1 + j rows with probability p(j) - p(j+1), and a token takes
//! `sum of those rows' costs weighted by that / (1 + the sum of p(j) * a(j))` seconds, with an error from the windows' errors. With no cut
//! (every chain reaching k) this is the cost of k+1 rows over 1 plus the shares of the first k drafts kept.
//!
//! The passes at the width in force feed the windows of the rows and positions it reaches. The others are fed by probes, now and then: a
//! stretch of passes at each width not in force in turn (a pass of k+1 rows is only seen at a width of k or more, and one that tried no
//! draft only at none), a few percent of all passes ([`PROBE_SHARE`]), the widths never probed first and none before the
//! [`FIRST_PROBE_AFTER`]th pass (a machine's first passes are slow while its memory fills). A probe is a stretch of passes at the width
//! asked for with the cut in force as in any: what it measures is what that width does, and every stretch feeds the shares that the widths
//! up to it have in common. A width replaces the one in force only if it is [`MARGIN`] faster and the difference is more than
//! [`CONFIDENCE`] standard errors of it, as the assist level's tuner has it (`crate::assist`): passes vary by a quarter from one to the next,
//! and a width not told apart from the one in force by its windows never replaces it.
//!
//! Plain logic: the model asks for a plan before each pass and feeds back what the pass did (`Model::generate_speculative`).

use std::collections::VecDeque;

use crate::assist;

/// The part of the width in force's seconds a token that another must save to replace it.
pub const MARGIN: f64 = 0.03;
/// How many standard errors of the difference of two widths' seconds a token that saving must exceed. The assist level's tuner asks for two
/// ([`assist::CONFIDENCE`]); here there are up to three widths to be told apart from the one in force, at every probe for as long as the
/// model runs: at two, a run of equal widths moved on noise in one decision of twenty.
pub const CONFIDENCE: f64 = 2.5;
/// Passes a cost window holds, by rows; and the outcomes a share's window holds, by position.
pub const COST_WINDOW: usize = 64;
pub const SHARE_WINDOW: usize = 128;
/// The fewest passes (or offers) a window must hold to be believed.
pub const LEAST_SAMPLES: usize = 8;
/// Passes of a probe, and the share of all passes the probes may take.
pub const PROBE_PASSES: usize = 24;
pub const PROBE_SHARE: f64 = 0.03;
/// The passes after which the first probe may begin: a request that ends sooner is run as the width in force asks, and what a machine's
/// first passes cost (its memory still filling with the model's experts) is not what it costs.
pub const FIRST_PROBE_AFTER: u64 = 256;
/// A row that fewer passes than this share have is costed from its neighbour where its window is empty. Where it holds only a few passes
/// ([`FEW_SAMPLES`] or more) they cost it, with at least [`FEW_DOUBT`] of the cost as the error.
const MINOR_ROW: f64 = 0.05;
const FEW_SAMPLES: usize = 3;
const FEW_DOUBT: f64 = 0.15;

/// Normal passes between the end of a probe and the start of the next, so that probes (the pass at the start of a stretch, which is not
/// timed, included) take [`PROBE_SHARE`] of the passes.
pub fn probe_gap() -> u64 {
    ((PROBE_PASSES + 1) as f64 * (1.0 - PROBE_SHARE) / PROBE_SHARE).ceil() as u64
}

/// The latest `capacity` values of something measured.
#[derive(Debug, Clone)]
struct Window {
    values: VecDeque<f64>,
    capacity: usize,
}

impl Window {
    fn new(capacity: usize) -> Window {
        Window { values: VecDeque::with_capacity(capacity), capacity }
    }

    fn push(&mut self, value: f64) {
        if self.values.len() == self.capacity {
            self.values.pop_front();
        }
        self.values.push_back(value);
    }

    fn clear(&mut self) {
        self.values.clear();
    }

    /// Seconds: the mean without the slowest eighth, and its standard error (a stall says nothing of a pass's rows: `assist::trimmed_mean`),
    /// if the window holds `least` values.
    fn seconds_from(&self, least: usize) -> Option<(f64, f64)> {
        if self.values.len() < least {
            return None;
        }
        let values: Vec<f64> = self.values.iter().copied().collect();
        Some(assist::trimmed_mean(&values))
    }

    fn seconds(&self) -> Option<(f64, f64)> {
        self.seconds_from(LEAST_SAMPLES)
    }

    /// A share of ones among values of 0 and 1, and its standard error: from a share one count shy of all or none, so that a window of
    /// all kept does not claim to be certain.
    fn share(&self) -> Option<(f64, f64)> {
        let n = self.values.len();
        if n < LEAST_SAMPLES {
            return None;
        }
        let ones: f64 = self.values.iter().sum();
        let spread = (ones + 1.0) / (n as f64 + 2.0);
        Some((ones / n as f64, (spread * (1.0 - spread) / (n as f64 + 2.0)).sqrt()))
    }
}

/// What the model is to do in the next pass: offer up to `drafts` drafts (a `probe` is a stretch of passes at a width not in force).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub drafts: usize,
    pub probe: bool,
}

/// What a width is expected to cost: seconds a token, and the standard error of that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    pub drafts: usize,
    pub seconds: f64,
    pub error: f64,
}

/// What the windows said when a probe ended.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// The first time every width a probe is of use at had been probed.
    pub first: bool,
    /// The width in force before, and after.
    pub from: usize,
    pub chosen: usize,
    /// Each width the windows could tell the seconds a token of, by width.
    pub estimates: Vec<Estimate>,
    /// The share of the drafts offered at each position (from the first) that were kept.
    pub kept: Vec<f64>,
    /// The seconds of a pass by its rows (from one) where the window holds enough.
    pub costs: Vec<Option<f64>>,
}

impl Outcome {
    /// Whether the log says so: the first table, and one that changed the width.
    pub fn worth_saying(&self) -> bool {
        self.first || self.chosen != self.from
    }

    /// The log line: milliseconds a token at each width, the share of drafts kept at each position and the seconds of a pass by its rows,
    /// and the width kept.
    pub fn line(&self) -> String {
        let mut text = String::from("Straydog drafts:");
        for (at, estimate) in self.estimates.iter().enumerate() {
            let (drafts, ms) = (estimate.drafts, estimate.seconds * 1e3);
            if at == 0 {
                text.push_str(&format!(" {drafts} {ms:.1} ms/token"));
            } else {
                text.push_str(&format!(", {drafts} {ms:.1}"));
            }
        }
        let mut details = Vec::new();
        if !self.kept.is_empty() {
            let shares: Vec<String> = self.kept.iter().map(|share| format!("{share:.2}")).collect();
            details.push(format!("accepted {}", shares.join(", ")));
        }
        let costs: Vec<String> = self.costs.iter().enumerate().filter_map(|(at, seconds)| {
            seconds.map(|seconds| format!("{} {:.1}", if at == 0 { "1 row".to_string() } else { format!("{} rows", at + 1) }, seconds * 1e3))
        }).collect();
        if !costs.is_empty() {
            details.push(format!("a pass of {} ms", costs.join(", ")));
        }
        if !details.is_empty() {
            text.push_str(&format!(" ({})", details.join("; ")));
        }
        if self.chosen == self.from {
            text.push_str(&format!("; keeping {}", self.chosen));
        } else {
            text.push_str(&format!("; now {} (was {})", self.chosen, self.from));
        }
        text
    }
}

/// A stretch of probe passes under way.
#[derive(Debug, Clone, Copy)]
struct Probe {
    width: usize,
    /// Passes still to be passed over (the first of a stretch: the layers' graphs for its rows are captured in it).
    skip: usize,
    /// Passes timed so far.
    taken: usize,
}

/// The width in force and the windows that choose it.
pub struct Chooser {
    most: usize,
    drafts: usize,
    /// A test holds the width: nothing is probed or chosen.
    pinned: bool,
    /// Seconds of a pass that tried no draft.
    bare: Window,
    /// Seconds of a pass that tried a draft, by its rows (index: rows - 1).
    cost: Vec<Window>,
    /// By draft position (index: position - 1), of the passes that could offer it: 1 where the chain reached it, else 0.
    reach: Vec<Window>,
    /// By draft position, of the passes that offered it: 1 where it was kept, else 0.
    kept: Vec<Window>,
    /// Passes seen, of them in probes, and when the last probe ended.
    passes: u64,
    probe_passes: u64,
    last_probe: u64,
    probe: Option<Probe>,
    /// Widths a probe has been at, and the one the last was.
    probed: Vec<bool>,
    cursor: usize,
    /// Passes still to be passed over at the start of a request.
    request_skip: usize,
    /// The first table has been given.
    decided: bool,
}

/// The seconds a token takes with k drafts, as the windows' values say: `cost` of a pass of 1 to k+1 rows, `reach` the share of passes whose
/// chain reaches positions 1 to k, `kept` the share of drafts kept at those positions.
fn rate(cost: &[f64], reach: &[f64], kept: &[f64]) -> f64 {
    let k = reach.len();
    let (mut seconds, mut tokens, mut before) = (0.0, 1.0, 1.0f64);
    for rows in 0..=k {
        // (a share that rises with the position is a window's chance: the chain reaches no further than the position before it did)
        let next = if rows < k { reach[rows].min(before) } else { 0.0 };
        seconds += (before - next) * cost[rows];
        if rows < k {
            tokens += next * kept[rows];
        }
        before = next;
    }
    seconds / tokens
}

/// The width among `estimates` that replaces `held`: the fastest of those [`MARGIN`] faster than it and by more than
/// [`CONFIDENCE`] standard errors of the difference (the two errors added as squares), else `held`; `held` itself if it has no estimate.
fn choose(estimates: &[Estimate], held: usize) -> usize {
    let Some(&kept) = estimates.iter().find(|estimate| estimate.drafts == held) else { return held };
    estimates.iter().filter(|estimate| {
        estimate.drafts != held && estimate.seconds <= kept.seconds * (1.0 - MARGIN)
            && kept.seconds - estimate.seconds > CONFIDENCE * kept.error.hypot(estimate.error)
    }).min_by(|a, b| a.seconds.total_cmp(&b.seconds)).map_or(held, |best| best.drafts)
}

impl Chooser {
    /// A chooser of 0 to `most` drafts, `start` in force.
    pub fn new(most: usize, start: usize) -> Chooser {
        let start = start.min(most);
        Chooser {
            most,
            drafts: start,
            pinned: false,
            bare: Window::new(COST_WINDOW),
            cost: vec![Window::new(COST_WINDOW); most + 1],
            reach: vec![Window::new(SHARE_WINDOW); most],
            kept: vec![Window::new(SHARE_WINDOW); most],
            passes: 0,
            probe_passes: 0,
            last_probe: 0,
            probe: None,
            probed: vec![false; most + 1],
            cursor: 0,
            request_skip: 0,
            decided: false,
        }
    }

    /// The width in force.
    pub fn drafts(&self) -> usize {
        self.drafts
    }

    /// Tests only: hold the width at `drafts` (None: choose again), with nothing probed meanwhile.
    pub fn pin(&mut self, drafts: Option<usize>) {
        self.probe = None;
        match drafts {
            Some(drafts) => {
                self.drafts = drafts.min(self.most);
                self.pinned = true;
            }
            None => self.pinned = false,
        }
    }

    /// Passes seen, and of them those of probes.
    pub fn passes(&self) -> (u64, u64) {
        (self.passes, self.probe_passes)
    }

    /// A request begins to decode: its first pass says nothing of the rows' cost (the experts its prompt left are read again).
    pub fn begin_request(&mut self) {
        self.request_skip = 1;
    }

    /// The seconds of a pass of `rows` rows, if its window holds enough.
    fn pass_seconds(&self, rows: usize) -> Option<(f64, f64)> {
        self.cost.get(rows - 1)?.seconds()
    }

    /// The seconds a token takes with `drafts` drafts, from the windows, if they hold enough: every position up to it must have been
    /// offered enough, and the cost of every row that a pass of it has often enough (a row it has seldom is costed from its neighbour's).
    pub fn estimate(&self, drafts: usize) -> Option<Estimate> {
        if drafts > self.most {
            return None;
        }
        if drafts == 0 {
            let (seconds, error) = self.bare.seconds()?;
            return Some(Estimate { drafts, seconds, error });
        }
        let mut reach: Vec<(f64, f64)> = self.reach[..drafts].iter().map(Window::share).collect::<Option<_>>()?;
        // (chains never reach further than they did the position before)
        for at in 1..drafts {
            reach[at].0 = reach[at].0.min(reach[at - 1].0);
        }
        let mut kept: Vec<(f64, f64)> = Vec::with_capacity(drafts);
        for at in 0..drafts {
            // (a position offered too seldom to tell is taken as kept as the one before it was, with the doubt that is)
            kept.push(match (self.kept[at].share(), kept.last()) {
                (Some(share), _) => share,
                (None, Some(&(share, _))) => (share, 0.25),
                (None, None) => return None,
            });
        }
        // (the share of passes of `rows` rows: reaching the position before it, and not its own)
        let before = |position: usize| match position {
            0 => 1.0,
            position if position > drafts => 0.0,
            position => reach[position - 1].0,
        };
        let mut cost: Vec<(f64, f64)> = Vec::with_capacity(drafts + 1);
        for rows in 1..=drafts + 1 {
            match (self.pass_seconds(rows), self.cost[rows - 1].seconds_from(FEW_SAMPLES)) {
                (Some(measured), _) => cost.push(measured),
                (None, Some((mean, error))) => cost.push((mean, error.max(FEW_DOUBT * mean))),
                (None, None) if before(rows - 1) - before(rows) < MINOR_ROW => {
                    let near = (rows + 1..=drafts + 1).chain((1..rows).rev()).find_map(|other| self.pass_seconds(other))?;
                    cost.push((near.0, 0.2 * near.0));
                }
                (None, None) => return None,
            }
        }
        let values = |parts: &[(f64, f64)]| parts.iter().map(|part| part.0).collect::<Vec<f64>>();
        let (cost_values, reach_values, kept_values) = (values(&cost), values(&reach), values(&kept));
        let seconds = rate(&cost_values, &reach_values, &kept_values);
        // (each window's own error moves the rate by what it does alone, the way that moves it more (a share of all or none can only move
        // one way, and not that way); the windows are of other passes: the squares add)
        let moved = |values: &[f64], at: usize, by: f64| {
            let mut moved = values.to_vec();
            moved[at] = (moved[at] + by).max(0.0);
            moved
        };
        let share = |values: &[f64], at: usize, by: f64| {
            let mut moved = moved(values, at, by);
            moved[at] = moved[at].min(1.0);
            moved
        };
        let mut variance = 0.0;
        for (at, part) in cost.iter().enumerate() {
            let shift = [part.1, -part.1].map(|by| rate(&moved(&cost_values, at, by), &reach_values, &kept_values) - seconds);
            variance += shift[0].abs().max(shift[1].abs()).powi(2);
        }
        for (at, part) in reach.iter().enumerate() {
            let shift = [part.1, -part.1].map(|by| rate(&cost_values, &share(&reach_values, at, by), &kept_values) - seconds);
            variance += shift[0].abs().max(shift[1].abs()).powi(2);
        }
        for (at, part) in kept.iter().enumerate() {
            let shift = [part.1, -part.1].map(|by| rate(&cost_values, &reach_values, &share(&kept_values, at, by)) - seconds);
            variance += shift[0].abs().max(shift[1].abs()).powi(2);
        }
        Some(Estimate { drafts, seconds, error: variance.sqrt() })
    }

    /// The widths a probe is of use at: every one but the width in force (a pass of k+1 rows is seen at a width of k or more, and only a
    /// chain that is cut makes a shorter one at a wider width; a pass that tried no draft is only seen at none).
    fn wanted(&self) -> Vec<usize> {
        (0..=self.most).filter(|&width| width != self.drafts).collect()
    }

    /// The width a probe is due at, if one is: a width never probed (once the first passes are past) at once, then those wanted in turn,
    /// after [`probe_gap`] passes.
    fn due(&self) -> Option<usize> {
        let wanted = self.wanted();
        if let Some(&width) = wanted.iter().find(|&&width| !self.probed[width]) {
            return Some(width);
        }
        if self.passes.saturating_sub(self.last_probe) < probe_gap() {
            return None;
        }
        wanted.iter().copied().find(|&width| width > self.cursor).or_else(|| wanted.first().copied())
    }

    /// What the next pass is to do. A probe, once begun, goes on until its stretch is timed (`record`).
    pub fn plan(&mut self) -> Plan {
        if self.pinned || self.most == 0 {
            return Plan { drafts: self.drafts, probe: false };
        }
        if let Some(probe) = &self.probe {
            return Plan { drafts: probe.width, probe: true };
        }
        if self.passes >= FIRST_PROBE_AFTER {
            if let Some(width) = self.due() {
                // (fresh windows where this probe is the one to feed them, the cost of its own rows and, past the width in force, its own
                // position's shares: what was timed there before may be of another time of the machine)
                if width == 0 {
                    self.bare.clear();
                } else {
                    self.cost[width].clear();
                    if width > self.drafts {
                        self.reach[width - 1].clear();
                        self.kept[width - 1].clear();
                    }
                }
                self.probe = Some(Probe { width, skip: 1, taken: 0 });
                return Plan { drafts: width, probe: true };
            }
        }
        Plan { drafts: self.drafts, probe: false }
    }

    /// A pass of `plan` offered up to `limit` drafts (the plan's, or fewer where the request or the context left no room), had `rows` rows,
    /// `accepted` of its drafts were kept and it took `seconds`. What a probe's end found, when this pass ended one.
    pub fn record(&mut self, plan: Plan, limit: usize, rows: usize, accepted: usize, seconds: f64) -> Option<Outcome> {
        let rows = rows.max(1);
        self.passes += 1;
        let probing = plan.probe && self.probe.is_some();
        if plan.probe {
            self.probe_passes += 1;
        }
        let mut cold = std::mem::take(&mut self.request_skip) > 0;
        if let Some(probe) = self.probe.as_mut().filter(|_| probing) {
            if probe.skip > 0 {
                probe.skip -= 1;
                cold = true;
            }
        }
        if !cold {
            if limit == 0 {
                self.bare.push(seconds);
            } else {
                if let Some(window) = self.cost.get_mut(rows - 1) {
                    window.push(seconds);
                }
                for position in 1..=limit.min(self.most) {
                    self.reach[position - 1].push(f64::from(rows > position));
                }
                for position in 1..rows {
                    if let Some(window) = self.kept.get_mut(position - 1) {
                        window.push(f64::from(accepted >= position));
                    }
                }
            }
            if let Some(probe) = self.probe.as_mut().filter(|_| probing) {
                probe.taken += 1;
            }
        }
        let probe = self.probe.filter(|probe| probing && probe.taken >= PROBE_PASSES)?;
        self.probe = None;
        self.probed[probe.width] = true;
        self.cursor = probe.width;
        self.last_probe = self.passes;
        self.decide()
    }

    /// The width in force after what the windows hold, once every width a probe is of use at has been probed.
    fn decide(&mut self) -> Option<Outcome> {
        if self.pinned || self.wanted().iter().any(|&width| !self.probed[width]) {
            return None;
        }
        let estimates: Vec<Estimate> = (0..=self.most).filter_map(|width| self.estimate(width)).collect();
        let from = self.drafts;
        self.drafts = choose(&estimates, from);
        let first = !std::mem::replace(&mut self.decided, true);
        let kept = self.kept.iter().map_while(|window| window.share().map(|(share, _)| share)).collect();
        let costs = (1..=self.most + 1).map(|rows| self.pass_seconds(rows).map(|(seconds, _)| seconds)).collect();
        Some(Outcome { first, from, chosen: self.drafts, estimates, kept, costs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A machine: what a pass costs by its rows, what one that tried no draft costs, how likely the chain of drafts is to go on to the next
    /// position (the cut ends it where the drafter is unsure) and how likely each draft is kept given the ones before it; the passes' times
    /// vary by `noise` (a share of the cost, either way, drawn from a seeded generator).
    #[derive(Clone)]
    struct Machine {
        cost: [f64; 4],
        bare: f64,
        goes_on: [f64; 3],
        keeps: [f64; 3],
        noise: f64,
        state: u64,
    }

    impl Machine {
        fn new(bare: f64, cost: [f64; 4], keeps: [f64; 3]) -> Machine {
            Machine { cost, bare, goes_on: [1.0; 3], keeps, noise: 0.0, state: 0x9E37_79B9_7F4A_7C15 }
        }

        fn cut(mut self, goes_on: [f64; 3]) -> Machine {
            self.goes_on = goes_on;
            self
        }

        fn noisy(mut self, noise: f64, seed: u64) -> Machine {
            self.noise = noise;
            self.state ^= seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
            self
        }

        /// A number in [0, 1) (xorshift64*).
        fn next(&mut self) -> f64 {
            self.state ^= self.state >> 12;
            self.state ^= self.state << 25;
            self.state ^= self.state >> 27;
            (self.state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }

        /// A pass of `plan`: the drafts it could offer, its rows, the drafts kept and its seconds.
        fn pass(&mut self, plan: Plan) -> (usize, usize, usize, f64) {
            let limit = plan.drafts;
            let mut offered = 0;
            while offered < limit && self.next() < self.goes_on[offered] {
                offered += 1;
            }
            let mut accepted = 0;
            while accepted < offered && self.next() < self.keeps[accepted] {
                accepted += 1;
            }
            let base = if limit == 0 { self.bare } else { self.cost[offered] };
            (limit, offered + 1, accepted, base * (1.0 + self.noise * (2.0 * self.next() - 1.0)) * 1e-3)
        }

        /// The seconds a token takes at each width, as the machine is.
        fn truth(&self) -> [f64; 4] {
            let (mut reach, mut kept) = ([0.0; 3], [0.0; 3]);
            let (mut goes, mut keeps) = (1.0, 1.0);
            for at in 0..3 {
                goes *= self.goes_on[at];
                keeps *= self.keeps[at];
                reach[at] = goes;
                kept[at] = keeps;
            }
            let at = |width: usize| rate(&self.cost[..=width], &reach[..width], &kept[..width]);
            [self.bare, at(1), at(2), at(3)]
        }
    }

    /// Every change of width was to a width that the machine takes no more seconds a token at (within 3%, what its windows cannot tell).
    fn no_change_loses(run: &Run, machine: &Machine) {
        let truth = machine.truth();
        for (from, to) in &run.changes {
            assert!(truth[*to] <= truth[*from] * 1.03, "{from} to {to} loses: {:?}: {:?}", truth, run.lines);
        }
    }

    /// What a run of the chooser on a machine did.
    struct Run {
        /// The width in force at the end, and the changes of it (from, to).
        drafts: usize,
        changes: Vec<(usize, usize)>,
        lines: Vec<String>,
        passes: u64,
        probes: u64,
        /// The machine's own seconds a token at the width in force over the whole run, against the best width's.
        loss: f64,
    }

    fn run(chooser: &mut Chooser, machine: &mut Machine, passes: usize) -> Run {
        let truth = machine.truth();
        let best = truth.iter().copied().fold(f64::INFINITY, f64::min);
        let (mut changes, mut lines, mut spent) = (Vec::new(), Vec::new(), 0.0);
        chooser.begin_request();
        for _ in 0..passes {
            let plan = chooser.plan();
            let (limit, rows, accepted, seconds) = machine.pass(plan);
            spent += truth[chooser.drafts()];
            if let Some(outcome) = chooser.record(plan, limit, rows, accepted, seconds) {
                if outcome.chosen != outcome.from {
                    changes.push((outcome.from, outcome.chosen));
                }
                if outcome.worth_saying() {
                    lines.push(outcome.line());
                }
            }
        }
        let (seen, probes) = chooser.passes();
        Run { drafts: chooser.drafts(), changes, lines, passes: seen, probes, loss: spent / passes as f64 / best - 1.0 }
    }

    /// A CPU-bound machine: each row adds the CPU's routes again, so a pass of two rows costs nearly twice one, and one draft that is kept
    /// nearly always gains a sixth over none; a second draft gains nothing more.
    fn cpu_bound() -> Machine {
        Machine::new(30.0, [31.5, 50.0, 69.0, 88.0], [0.93, 0.75, 0.7])
    }

    /// A machine whose GPU part of a layer is the longer on a pass of one row (the CPU's routes hide under it for a second row, and add to it
    /// from a third): a draft gains more than a quarter, a second more than 15% on that, and a third loses.
    fn gpu_bound_two() -> Machine {
        Machine::new(58.0, [59.0, 78.0, 90.0, 125.0], [0.9, 0.78, 0.7])
    }

    /// A machine where a row costs almost nothing: every width up to three pays.
    fn rows_free() -> Machine {
        Machine::new(40.0, [41.0, 42.0, 44.0, 46.0], [0.9, 0.8, 0.7])
    }

    /// A machine where drafts are wasted: a row costs as much as the token it adds, and drafts are kept half the time.
    fn drafts_wasted() -> Machine {
        Machine::new(30.0, [31.0, 62.0, 93.0, 124.0], [0.5, 0.5, 0.5])
    }

    #[test]
    fn the_machines_have_the_winners_they_are_named_for() {
        let best = |machine: &Machine| {
            let truth = machine.truth();
            (0..4).min_by(|a, b| truth[*a].total_cmp(&truth[*b])).unwrap()
        };
        assert_eq!([best(&cpu_bound()), best(&gpu_bound_two()), best(&rows_free()), best(&drafts_wasted())], [1, 2, 3, 0]);
        // (with the cut ending chains early the same: the passes are shorter, and the best width is what it was)
        let cut = [0.9, 0.85, 0.8];
        assert_eq!([best(&cpu_bound().cut(cut)), best(&gpu_bound_two().cut(cut)), best(&rows_free().cut(cut))], [1, 2, 3]);
    }

    /// The windows say what the machine is: with each width run long enough to fill the windows (a window holds a few dozen passes, and the
    /// share of one 128), the estimate of every width is within 5% of what a token takes at it there, whether the chains are cut or not and
    /// with the passes' times varying by a tenth (the mean without the slowest eighth, as the windows take it, is 1.2% under the mean of
    /// such passes).
    #[test]
    fn the_estimates_are_what_the_machine_does_with_or_without_a_cut() {
        for (name, machine) in [("cpu bound", cpu_bound()), ("gpu bound", gpu_bound_two()), ("rows free", rows_free())] {
            for cut in [[1.0; 3], [0.9, 0.8, 0.7], [0.7, 0.6, 0.5]] {
                let mut machine = machine.clone().cut(cut).noisy(0.1, 7);
                let mut chooser = Chooser::new(3, 1);
                for width in [0, 3, 2, 1] {
                    chooser.pin(Some(width));
                    run(&mut chooser, &mut machine, 700);
                }
                let truth = machine.truth();
                for width in 0..4 {
                    let estimate = chooser.estimate(width).unwrap_or_else(|| panic!("{name} {cut:?}: no estimate at {width}"));
                    assert!((estimate.seconds * 1e3 / truth[width] - 1.0).abs() < 0.05,
                            "{name} {cut:?}: width {width} {:.2} ms a token, the machine's {:.2}", estimate.seconds * 1e3, truth[width]);
                }
            }
        }
    }

    /// From the width the CUDA build starts at, and from none, a CPU-bound machine ends at one draft. (From two or three, where it is 1.4% and
    /// 8% slower, the windows cannot tell it from where it is, and it stays: what they cannot tell apart does not move it.)
    #[test]
    fn one_draft_wins_on_a_cpu_bound_machine() {
        for start in 0..=1 {
            for cut in [[1.0; 3], [0.9, 0.8, 0.7]] {
                let mut chooser = Chooser::new(3, start);
                let mut machine = cpu_bound().cut(cut);
                let run = run(&mut chooser, &mut machine, 4000);
                assert_eq!(run.drafts, 1, "from {start}, cut {cut:?}: {:?}", run.lines);
                no_change_loses(&run, &machine);
            }
        }
    }

    #[test]
    fn two_drafts_win_where_the_gpu_is_the_limit_but_not_free() {
        for start in 0..=3 {
            for cut in [[1.0; 3], [0.9, 0.8, 0.7]] {
                let mut chooser = Chooser::new(3, start);
                let run = run(&mut chooser, &mut gpu_bound_two().cut(cut), 4000);
                assert_eq!(run.drafts, 2, "from {start}, cut {cut:?}: {:?}", run.lines);
            }
        }
    }

    /// (With the cut ending chains early the third draft's gain over the second is 6%, which the windows cannot tell from nothing: from two
    /// it stays.)
    #[test]
    fn three_drafts_win_where_a_rows_cost_rises_little() {
        for start in 0..=3 {
            for cut in [[1.0; 3], [0.9, 0.8, 0.7]] {
                let mut chooser = Chooser::new(3, start);
                let run = run(&mut chooser, &mut rows_free().cut(cut), 4000);
                let wanted: &[usize] = if cut[0] < 1.0 && start == 2 { &[2, 3] } else { &[3] };
                assert!(wanted.contains(&run.drafts), "from {start}, cut {cut:?}: {:?}", run.lines);
            }
        }
    }

    #[test]
    fn a_choice_is_never_beyond_the_most_the_model_has() {
        let mut chooser = Chooser::new(2, 1);
        let run = run(&mut chooser, &mut rows_free(), 4000);
        assert_eq!(run.drafts, 2);
        assert_eq!(Chooser::new(3, 9).drafts(), 3);
    }

    /// Where drafts are wasted none is the choice, from one.
    #[test]
    fn no_drafts_win_where_a_row_costs_more_than_it_earns() {
        let mut chooser = Chooser::new(3, 1);
        let run = run(&mut chooser, &mut drafts_wasted(), 4000);
        assert_eq!(run.drafts, 0, "{:?}", run.lines);
    }

    /// The width in force is left only for one that is [`MARGIN`] faster: with the alternative 2% faster it stays, however sure the windows are
    /// (acceptance that is kept every time, and costs that do not vary).
    #[test]
    fn the_margin_keeps_a_width_that_is_nearly_as_good() {
        // widths 1 and 2 on a machine whose second draft is worth 2%
        let one = 50.0 / 2.0;
        let machine = Machine::new(60.0, [31.5, 50.0, one * 0.98 * 3.0, 200.0], [1.0, 1.0, 0.0]);
        let truth = machine.truth();
        assert!((truth[2] / truth[1] - 0.98).abs() < 1e-9, "{truth:?}");
        let mut chooser = Chooser::new(3, 1);
        let run = run(&mut chooser, &mut { machine }, 4000);
        assert_eq!(run.drafts, 1, "{:?}", run.lines);
        // (and at 4% the same machine moves, as far as the estimates know: the choice, which the windows' error then decides)
        let estimate = |drafts, ms: f64| Estimate { drafts, seconds: ms * 1e-3, error: 0.0 };
        assert_eq!(choose(&[estimate(1, 25.0), estimate(2, 25.0 * 0.96)], 1), 2);
        assert_eq!(choose(&[estimate(1, 25.0), estimate(2, 25.0 * 0.98)], 1), 1);
    }

    /// Passes that vary by a quarter either way do not move the width between widths that cost the same: what the windows cannot tell apart
    /// stays as it was.
    #[test]
    fn noise_does_not_move_the_width() {
        let mut moved = 0;
        for seed in 1..=12 {
            let mut chooser = Chooser::new(3, 1);
            // widths 0 to 3 within 2% of each other
            let mut machine = Machine::new(25.6, [26.0, 49.0, 67.8, 79.4], [0.93, 0.75, 0.7]).noisy(0.25, seed);
            let truth = machine.truth();
            let (low, high) = (truth.iter().copied().fold(f64::INFINITY, f64::min), truth.iter().copied().fold(0.0, f64::max));
            assert!(high / low < 1.02, "{truth:?}");
            let run = run(&mut chooser, &mut machine, 8000);
            moved += run.changes.len();
            assert!(run.loss < 0.04, "seed {seed}: a loss of {:.3}", run.loss);
        }
        // (a width must look more than 2.5 of its errors better: about 2% of the decisions, of which a run of this length has a dozen, on noise)
        assert!(moved <= 3, "the width moved {moved} times in 12 runs of equal widths");
    }

    /// A width that is far better is found through the same noise.
    #[test]
    fn a_clear_gain_is_found_through_noise() {
        for seed in 1..=8 {
            let mut chooser = Chooser::new(3, 1);
            let run = run(&mut chooser, &mut rows_free().noisy(0.25, seed), 8000);
            assert_eq!(run.drafts, 3, "seed {seed}: {:?}", run.lines);
        }
    }

    /// Probing takes a few percent of the passes, however many there are, and none before [`FIRST_PROBE_AFTER`].
    #[test]
    fn probing_stays_under_its_share() {
        let mut chooser = Chooser::new(3, 1);
        let mut machine = cpu_bound();
        chooser.begin_request();
        let mut first = None;
        for pass in 0..40_000u64 {
            let plan = chooser.plan();
            if plan.probe && first.is_none() {
                first = Some(pass);
            }
            let (limit, rows, accepted, seconds) = machine.pass(plan);
            chooser.record(plan, limit, rows, accepted, seconds);
        }
        assert!(first.unwrap() >= FIRST_PROBE_AFTER, "the first probe at pass {first:?}");
        let (passes, probes) = chooser.passes();
        // (the first round probes the three widths there are to probe at once; after it the share is the one asked for)
        let first_round = 3 * (PROBE_PASSES as u64 + 1);
        assert!(probes as f64 <= PROBE_SHARE * passes as f64 + first_round as f64, "{probes} probe passes of {passes}");
        assert!(probes as f64 >= 0.5 * PROBE_SHARE * passes as f64, "{probes} probe passes of {passes}: the widths are no longer timed");
    }

    /// A request that ends before the first probe is due runs at the width in force, as every request before the chooser did.
    #[test]
    fn a_short_request_is_not_probed() {
        let mut chooser = Chooser::new(3, 1);
        let mut machine = cpu_bound();
        let run = run(&mut chooser, &mut machine, FIRST_PROBE_AFTER as usize);
        assert_eq!(run.probes, 0);
        assert_eq!(run.passes, FIRST_PROBE_AFTER);
    }

    /// The probes are stretches at one width, each width not in force in turn; the width in force is not probed.
    #[test]
    fn probes_visit_the_widths_not_in_force() {
        // (a machine on which the width in force is the best, so that it stays)
        for (start, wanted, mut machine) in [(1usize, vec![0, 2, 3], cpu_bound()), (2, vec![0, 1, 3], gpu_bound_two()), (0, vec![1, 2, 3], drafts_wasted()),
                                              (3, vec![0, 1, 2], rows_free())] {
            let mut chooser = Chooser::new(3, start);
            chooser.begin_request();
            let (mut visited, mut before) = (Vec::new(), false);
            for _ in 0..8000 {
                let plan = chooser.plan();
                if plan.probe && !before {
                    visited.push(plan.drafts);
                }
                before = plan.probe;
                assert!(plan.drafts <= 3);
                let (limit, rows, accepted, seconds) = machine.pass(plan);
                chooser.record(plan, limit, rows, accepted, seconds);
            }
            assert_eq!(chooser.drafts(), start);
            assert!(visited.iter().all(|width| wanted.contains(width)), "from {start}: {visited:?}");
            assert!(wanted.iter().all(|width| visited.contains(width)), "from {start}: {visited:?}");
            assert!(visited.len() >= 2 * wanted.len(), "from {start}: the widths go on being timed: {visited:?}");
        }
    }

    /// The first table is said once, with every width probed; a later one only when it changes the width.
    #[test]
    fn the_log_line_names_each_width_and_what_was_kept() {
        let mut chooser = Chooser::new(3, 1);
        let run = run(&mut chooser, &mut gpu_bound_two(), 4000);
        assert_eq!(run.lines.len(), 1, "{:?}", run.lines);
        let first = &run.lines[0];
        assert!(first.starts_with("Straydog drafts: 0 58.0 ms/token, 1 "), "{first}");
        assert!(first.contains(" (accepted 0.") && first.contains("; a pass of 2 rows 78.0, 3 rows 90.0, 4 rows 125.0 ms)")
                && first.ends_with("; now 2 (was 1)"), "{first}");
        let outcome = Outcome { first: false, from: 1, chosen: 1, estimates: vec![Estimate { drafts: 0, seconds: 0.0182, error: 0.0 },
                                                                               Estimate { drafts: 1, seconds: 0.0169, error: 0.0 },
                                                                               Estimate { drafts: 2, seconds: 0.0174, error: 0.0 }],
                                kept: vec![0.93, 0.71], costs: Vec::new() };
        assert_eq!(outcome.line(), "Straydog drafts: 0 18.2 ms/token, 1 16.9, 2 17.4 (accepted 0.93, 0.71); keeping 1");
        assert!(!outcome.worth_saying());
        let with_costs = Outcome { costs: vec![Some(0.0215), Some(0.0325), None, None], ..outcome };
        assert_eq!(with_costs.line(),
                   "Straydog drafts: 0 18.2 ms/token, 1 16.9, 2 17.4 (accepted 0.93, 0.71; a pass of 1 row 21.5, 2 rows 32.5 ms); keeping 1");
    }

    /// A width that nothing is known of is not chosen, and a width in force that nothing is known of is not left.
    #[test]
    fn nothing_is_chosen_blind() {
        let estimate = |drafts, ms: f64| Estimate { drafts, seconds: ms * 1e-3, error: 0.0 };
        assert_eq!(choose(&[estimate(0, 30.0), estimate(2, 10.0)], 1), 1);
        assert_eq!(choose(&[estimate(1, 30.0), estimate(2, 10.0)], 1), 2);
        assert_eq!(choose(&[estimate(1, 30.0)], 1), 1);
        // the fastest of those that clear the margin, and the one in force at a tie
        assert_eq!(choose(&[estimate(0, 20.0), estimate(1, 30.0), estimate(2, 15.0), estimate(3, 15.0)], 1), 2);
        assert_eq!(choose(&[estimate(0, 29.5), estimate(1, 30.0)], 1), 1);
    }

    /// The error of a width's estimate counts: a gain that is large but within 2.5 times its error is not taken.
    #[test]
    fn a_gain_within_the_error_is_not_taken() {
        let held = Estimate { drafts: 1, seconds: 0.030, error: 0.001 };
        let near = Estimate { drafts: 2, seconds: 0.024, error: 0.004 };
        let clear = Estimate { drafts: 2, seconds: 0.024, error: 0.0005 };
        assert_eq!(choose(&[held, near], 1), 1);
        assert_eq!(choose(&[held, clear], 1), 2);
    }

    /// The seconds a token takes by the windows' values: with every chain reaching the width it is the cost of k+1 rows over one plus the
    /// shares kept; a chain that ends sooner makes a pass shorter and keeps less.
    #[test]
    fn a_token_takes_what_the_chains_that_are_cut_make_of_it() {
        // two drafts, never cut: 90 ms a pass of three rows, 1 + 0.9 + 0.72 tokens
        assert!((rate(&[59.0, 78.0, 90.0], &[1.0, 1.0], &[0.9, 0.72]) - 90.0 / 2.62).abs() < 1e-9);
        // half the chains stop at the first position: half the passes are a lone row of 59 ms and half have two rows of 78, 1 + 0.5 * 0.9 tokens
        assert!((rate(&[59.0, 78.0], &[0.5], &[0.9]) - (0.5 * 59.0 + 0.5 * 78.0) / 1.45).abs() < 1e-9);
        // none offered at all: a lone row
        assert!((rate(&[59.0, 78.0], &[0.0], &[0.9]) - 59.0).abs() < 1e-9);
        // a share that rises with the position is the chain's chance: it is taken as no more than the one before
        assert_eq!(rate(&[10.0, 20.0, 30.0], &[0.5, 0.9], &[1.0, 1.0]), rate(&[10.0, 20.0, 30.0], &[0.5, 0.5], &[1.0, 1.0]));
    }

    /// Where a cut ends nearly every chain at the first draft, the rows past it are seldom seen: a row that a tenth of the passes have is
    /// costed from the few that were (three or more), and one that none have from its neighbour, so that the width is told all the same;
    /// a row a tenth have and two were seen is not told.
    #[test]
    fn a_row_few_passes_have_is_costed_from_them() {
        let filled = |wide: usize| {
            let mut chooser = Chooser::new(3, 1);
            for _ in 0..20 {
                chooser.bare.push(0.020);
                chooser.cost[0].push(0.021);
                chooser.cost[1].push(0.030);
                chooser.reach[0].push(1.0);
                chooser.reach[2].push(0.0);
                chooser.kept[0].push(1.0);
            }
            for at in 0..20 {
                chooser.reach[1].push(f64::from(at < 2));
                chooser.kept[1].push(1.0);
            }
            for _ in 0..wide {
                chooser.cost[2].push(0.045);
            }
            chooser
        };
        // (one draft: two rows, the passes of a row of three hardly come into it)
        assert!(filled(0).estimate(1).is_some());
        let told = filled(4).estimate(2).expect("a row seen four times costs its width");
        // (a tenth of the passes cost 45 ms, nine tenths 30: over 1 + 1.0 + 0.1 tokens, with a 15% doubt on the 45)
        assert!((told.seconds - (0.9 * 0.030 + 0.1 * 0.045) / 2.1).abs() < 1e-9, "{}", told.seconds);
        assert!(told.error > 0.0);
        assert!(filled(2).estimate(2).is_none(), "two passes do not tell a row that a tenth of the passes have");
        // (past the chains' reach nothing weighs: the width is told from the row before it)
        assert!(filled(4).estimate(3).is_some());
    }

    /// Pinned, the width stays and nothing is probed.
    #[test]
    fn a_pinned_width_stays() {
        let mut chooser = Chooser::new(3, 1);
        chooser.pin(Some(3));
        let run = run(&mut chooser, &mut cpu_bound(), 4000);
        assert_eq!((run.drafts, run.probes, run.changes.len()), (3, 0, 0));
        chooser.pin(None);
        let run = self::run(&mut chooser, &mut cpu_bound(), 4000);
        assert_eq!(run.drafts, 1);
    }

    /// The windows of a width that was probed long ago are replaced by fresh ones when it is probed again: a machine whose costs changed is
    /// read as it is now.
    #[test]
    fn a_probe_replaces_what_was_timed_before() {
        let mut chooser = Chooser::new(3, 1);
        assert_eq!(run(&mut chooser, &mut cpu_bound(), 4000).drafts, 1);
        // the machine now costs a fifth of what it did, with rows that cost almost nothing: the widest width pays
        let mut cheaper = Machine::new(8.5, [8.5, 8.8, 9.0, 9.2], [0.93, 0.75, 0.7]);
        let run = run(&mut chooser, &mut cheaper, 8000);
        assert_eq!(run.drafts, 3, "{:?}", run.lines);
    }
}
