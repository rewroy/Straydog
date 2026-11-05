//! A start's warm-up of the expert stores with the experts an earlier run held (docs/DESIGN.md
//! §10.1), and the fill of the room that leaves. The state a run saves lists what its stores held,
//! least recently used first, and how often its passes used each expert; a later start reads the
//! listed experts back most recently used first (a start cut short by a request has the experts
//! likeliest needed), each in the place in the stores' LRU order it had, then offers the stores'
//! free slots the experts the passes used most. All of it runs between requests, behind the reads
//! of every pass, and never evicts.

use serde_json::{json, Value};

use super::served::Engines;

/// Experts still to warm up with, as (layer, expert), least recently used first (the warm-up takes
/// the most recent first, from the back).
type Pending = std::collections::VecDeque<(usize, usize)>;

/// Warm-up steps (a millisecond apart, the worker's) with no room and no read on its way before
/// the warm-up gives up.
const WARM_PATIENCE: usize = 30_000;
/// The same for a window of the fill, which only guesses: its stores' room (the host tier's pinned
/// arenas) is there or it is not, and the next window may fit.
const FILL_PATIENCE: usize = 5_000;

/// The recency rank ([`Engines::rank_warm`]) of the first of an earlier run's listed experts, each
/// other one more by its place in its list. The fill's ranks are below these, the likeliest the
/// highest: a guess is older than what a run used.
const RANK_LISTS: u64 = 1 << 16;
const RANK_FILL: u64 = 1 << 15;

/// How far a warm-up has got ([`Warm::step`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Warmed {
    /// Reads of the earlier run's experts, or of the fill's, remain.
    Going,
    /// The earlier run's experts are in the stores; the free slots fill next.
    Listed,
    /// Over: the free slots are filled as far as they go.
    Done,
}

/// The warm-up of one run: the experts still to read, and the fill that follows.
pub struct Warm {
    /// Layers, and experts a layer (a state's pairs must lie within).
    layers: usize,
    experts: usize,
    /// An earlier run's experts still to warm up with: (VRAM, host tier).
    pending: (Pending, Pending),
    /// Warm-up steps in a row that found no room and no read on its way.
    stalled: usize,
    /// The recency rank of the pending lists' first expert (see [`RANK_LISTS`]).
    base: u64,
    /// Whether the earlier run's experts have all landed (the fill goes after them).
    listed: bool,
    /// How many passes the earlier run used each expert, layer-major (empty without a state).
    prior: Vec<u32>,
    /// The experts the free slots are offered, likeliest first, and how many were weighed so far;
    /// none when no fill is to come.
    fill: Option<(Vec<(usize, usize)>, usize)>,
    /// What the fill placed so far: in VRAM, in the host tier, and the drive's bytes at its start.
    filled: (usize, usize, u64),
}

impl Warm {
    pub fn new(layers: usize, experts: usize) -> Warm {
        Warm {
            layers, experts, pending: Default::default(), stalled: 0, base: RANK_LISTS, listed: false, prior: Vec::new(), fill: None,
            filled: (0, 0, 0),
        }
    }

    /// What a later start warms up with: the experts the engines hold, least recently used first,
    /// in VRAM (`vram`) and only in the host tier (`host`), as [layer, expert] pairs; and how often
    /// the passes used each expert (`counts`, as [layer, expert, passes]: the earlier run's, aged by
    /// what this run adds to them, plus its own). The experts of an earlier
    /// state this run has not yet read are kept, older than all the engines hold (they are the less
    /// recent end of that state, the warm-up reading the more recent first): a run that ends
    /// before its warm-up does forgets nothing its state listed but what its own requests evicted.
    pub fn state<E: Engines>(&self, engines: &E) -> Value {
        let (held_vram, held_host) = engines.contents();
        let held: std::collections::HashSet<(usize, usize)> = held_vram.iter().chain(&held_host).copied().collect();
        let older = |pending: &Pending| -> Vec<(usize, usize)> { pending.iter().filter(|key| !held.contains(key)).copied().collect() };
        let (mut vram, mut host) = (older(&self.pending.0), older(&self.pending.1));
        vram.extend(held_vram);
        host.extend(held_host);
        json!({"vram": vram, "host": host, "counts": self.counts_to_save(engines)})
    }

    /// The route counts a state keeps, as [layer, expert, passes] for each expert counted.
    fn counts_to_save<E: Engines>(&self, engines: &E) -> Vec<[u64; 3]> {
        let (layers, experts) = (self.layers, self.experts);
        let uses = engines.uses();
        let (had, ran) = (self.prior.iter().map(|&count| u64::from(count)).sum::<u64>(), uses.iter().map(|use_| u64::from(use_.2)).sum::<u64>());
        let keep = kept_share(had, ran);
        let mut counts = vec![0u32; layers * experts];
        for (count, &before) in counts.iter_mut().zip(&self.prior) {
            *count = (f64::from(before) * keep).round() as u32;
        }
        for (layer, expert, passes) in uses {
            if layer < layers && expert < experts {
                let count = &mut counts[layer * experts + expert];
                *count = count.saturating_add(passes);
            }
        }
        counts.iter().enumerate().filter(|(_, &count)| count > 0)
            .map(|(index, &count)| [(index / experts) as u64, (index % experts) as u64, u64::from(count)]).collect()
    }

    /// Warm up with an earlier run's [`state`](Self::state), step by step ([`step`](Self::step))
    /// while no request runs. Pairs outside the model are dropped, and those the slots as placed
    /// do not hold. The experts to bring into VRAM and the host tier. (A state without `counts`, as
    /// earlier versions saved, warms up the same, and its fill goes in a fixed order.)
    pub fn start<E: Engines>(&mut self, engines: &mut E, state: &Value) -> Result<(usize, usize), String> {
        let (layers, experts) = (self.layers, self.experts);
        let pairs = |key: &str| -> Result<Pending, String> {
            let list = state.get(key).and_then(Value::as_array).ok_or("a Straydog warm-up state without its lists")?;
            Ok(list.iter()
                .filter_map(|pair| Some((pair.get(0)?.as_u64()? as usize, pair.get(1)?.as_u64()? as usize)))
                .filter(|&(layer, expert)| layer < layers && expert < experts)
                .collect())
        };
        let (vram, host) = engines.fit_warm(pairs("vram")?.into(), pairs("host")?.into());
        self.prior = vec![0; layers * experts];
        for count in state.get("counts").and_then(Value::as_array).into_iter().flatten() {
            let (Some(layer), Some(expert), Some(passes)) = (count.get(0).and_then(Value::as_u64), count.get(1).and_then(Value::as_u64),
                                                              count.get(2).and_then(Value::as_u64)) else { continue };
            if (layer as usize) < layers && (expert as usize) < experts {
                self.prior[layer as usize * experts + expert as usize] = passes.min(u64::from(u32::MAX)) as u32;
            }
        }
        // (the passes of the start-up's own prompt say nothing of a later run's)
        engines.clear_uses();
        self.pending = (vram.into(), host.into());
        self.base = RANK_LISTS;
        self.stalled = 0;
        self.listed = false;
        self.fill = Some((fill_order(&self.prior, layers, experts), 0));
        self.filled = (0, 0, 0);
        Ok((self.pending.0.len(), self.pending.1.len()))
    }

    /// What the fill placed so far: experts into VRAM and into the host tier, and the bytes the
    /// drive was read for since it began.
    pub fn filled<E: Engines>(&self, engines: &E) -> (usize, usize, u64) {
        (self.filled.0, self.filled.1, engines.drive_bytes().saturating_sub(self.filled.2))
    }

    /// Take the warm-up's reads that landed and, while fewer than `in_flight` are on their way
    /// (each device's read of an expert counts), start up to `count` more (VRAM's first, the
    /// experts a request is likeliest to use, then the host tier's; each most recently used
    /// first, and each in its earlier place in the LRU order), without waiting: a request that
    /// comes meanwhile finds the drive and the bus nearly free. When the earlier run's experts
    /// have all landed, the stores' free slots are offered the rest, `count` experts at a time,
    /// the ones the earlier run's passes used most first ([`Engines::fill_plan`]: never an
    /// eviction), and filled the same way.
    pub fn step<E: Engines>(&mut self, engines: &mut E, count: usize, in_flight: usize) -> Result<Warmed, String> {
        let reading = engines.poll()?;
        if self.pending.0.is_empty() && self.pending.1.is_empty() {
            if !self.listed {
                if reading > 0 {
                    return Ok(Warmed::Going);
                }
                self.listed = true;
                self.filled.2 = engines.drive_bytes();
                return Ok(Warmed::Listed);
            }
            if !self.plan_fill(engines, count) {
                return Ok(if reading == 0 { Warmed::Done } else { Warmed::Going });
            }
        }
        if reading >= in_flight || (self.pending.0.is_empty() && self.pending.1.is_empty()) {
            return Ok(Warmed::Going);
        }
        let count = count.min((in_flight - reading).div_ceil(engines.devices_per_expert()));
        let vram: Vec<(usize, usize)> = self.pending.0.iter().rev().take(count).copied().collect();
        let host: Vec<(usize, usize)> = self.pending.1.iter().rev().take(count - vram.len()).copied().collect();
        // (each in its place in its list, which taking from the back leaves alone)
        let (base, vram_len, host_len) = (self.base, self.pending.0.len(), self.pending.1.len());
        let ranks: Vec<((usize, usize), u64)> = vram.iter().enumerate().map(|(index, &key)| (key, base + (vram_len - 1 - index) as u64))
            .chain(host.iter().enumerate().map(|(index, &key)| (key, base + (host_len - 1 - index) as u64))).collect();
        engines.rank_warm(&ranks);
        // (those the stores have no room for yet stay first in line)
        let (reads, placed, hosted) = engines.warm(&vram, &host)?;
        self.pending.0.truncate(vram_len - placed);
        self.pending.1.truncate(host_len - hosted);
        if self.listed {
            self.filled.0 += placed;
            self.filled.1 += hosted;
        }
        // No room for a long while (the stores full of what requests brought in) and nothing on
        // its way: done. (Room comes as the host tier is pinned, and as reads land.)
        self.stalled = if placed + hosted == 0 && reads == 0 { self.stalled + 1 } else { 0 };
        if self.stalled > if self.listed { FILL_PATIENCE } else { WARM_PATIENCE } {
            self.pending = Default::default();
        }
        Ok(Warmed::Going)
    }

    /// Offer the next `count` experts of the fill to the stores' free slots: what they take becomes
    /// the experts to warm up with, each in its place by how likely it is. Whether any of the fill
    /// is left to offer.
    fn plan_fill<E: Engines>(&mut self, engines: &E, count: usize) -> bool {
        let Some((ranked, cursor)) = self.fill.as_mut() else { return false };
        if *cursor >= ranked.len() {
            self.fill = None;
            return false;
        }
        let end = (*cursor + count.max(1)).min(ranked.len());
        let (mut vram, mut host) = engines.fill_plan(&ranked[*cursor..end]);
        // (the likeliest last in the lists, as the warm-up takes the most recent first)
        vram.reverse();
        host.reverse();
        self.pending = (vram.into(), host.into());
        self.base = RANK_FILL.saturating_sub(end as u64);
        self.stalled = 0;
        *cursor = end;
        true
    }
}

/// The share an earlier state's counts keep when a run adds `ran` uses to the `had` they hold: all of them when it added nothing, hardly less for a few
/// (a restart's one reply), half when it added as many as they held or more (the most, so that a long run does not erase them). A partially resident
/// model's route counts ([`crate::routes`]) age by the same rule.
pub fn kept_share(had: u64, ran: u64) -> f64 {
    if ran == 0 { 1.0 } else { (had as f64 / (had + ran) as f64).max(0.5) }
}

/// The experts in the order a fill weighs them: by how often the earlier run's passes used each
/// (`prior`, layer-major), the most used first; the same count (none counted: all) by expert, then
/// layer, so that each layer gets as many.
fn fill_order(prior: &[u32], layers: usize, experts: usize) -> Vec<(usize, usize)> {
    let mut order: Vec<(usize, usize)> = (0..experts).flat_map(|expert| (0..layers).map(move |layer| (layer, expert))).collect();
    order.sort_by_key(|&(layer, expert)| std::cmp::Reverse(prior.get(layer * experts + expert).copied().unwrap_or(0)));
    order
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::gpu::Backend;

    type Key = (usize, usize);

    /// Stores that behave as the experts' stores do for a warm-up: a list of experts placed in its
    /// order into room that is free (never evicting; with none, it stops there), each in the LRU
    /// place its rank gives (else just used), and `contents` least recently used first.
    #[derive(Default)]
    struct Stores {
        vram_slots: usize,
        host_slots: usize,
        vram: Vec<(u64, Key)>,
        host: Vec<(u64, Key)>,
        ranks: HashMap<Key, u64>,
        clock: u64,
        uses: Vec<(usize, usize, u32)>,
        placed_in_order: Vec<Key>,
    }

    impl Stores {
        fn new(vram_slots: usize, host_slots: usize) -> Stores {
            Stores { vram_slots, host_slots, clock: 1 << 20, ..Stores::default() }
        }
        fn held(&self, key: &Key) -> bool {
            self.vram.iter().chain(&self.host).any(|(_, other)| other == key)
        }
        fn used(&mut self, key: &Key) -> u64 {
            self.ranks.remove(key).unwrap_or_else(|| {
                self.clock += 1;
                self.clock
            })
        }
        /// A request's use of an expert: just used, in VRAM if there is room, else the host tier.
        fn touch(&mut self, key: Key) {
            self.vram.retain(|(_, other)| *other != key);
            self.host.retain(|(_, other)| *other != key);
            self.clock += 1;
            self.vram.push((self.clock, key));
        }
        fn evict(&mut self, key: Key) {
            self.vram.retain(|(_, other)| *other != key);
            self.host.retain(|(_, other)| *other != key);
        }
        fn keys(list: &[(u64, Key)]) -> Vec<Key> {
            let mut sorted = list.to_vec();
            sorted.sort_unstable();
            sorted.into_iter().map(|(_, key)| key).collect()
        }
    }

    impl Engines for Stores {
        fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>, String> {
            Err("no model".into())
        }
        fn reset(&mut self) {}
        fn counts(&self) -> (u64, u64, u64) {
            (0, 0, 0)
        }
        fn contents(&self) -> crate::experts::Contents {
            (Stores::keys(&self.vram), Stores::keys(&self.host))
        }
        fn warm(&mut self, vram: &[Key], host: &[Key]) -> Result<(usize, usize, usize), String> {
            let mut hosted = 0;
            for key in host {
                if !self.held(key) {
                    if self.host.len() >= self.host_slots {
                        break;
                    }
                    let used = self.used(key);
                    self.host.push((used, *key));
                    self.placed_in_order.push(*key);
                }
                hosted += 1;
            }
            let mut placed = 0;
            for key in vram {
                if !self.held(key) {
                    let used = self.used(key);
                    if self.vram.len() < self.vram_slots {
                        self.vram.push((used, *key));
                    } else if self.host.len() < self.host_slots {
                        self.host.push((used, *key));
                    } else {
                        break;
                    }
                    self.placed_in_order.push(*key);
                }
                placed += 1;
            }
            Ok((0, placed, hosted))
        }
        fn rank_warm(&mut self, ranks: &[(Key, u64)]) {
            self.ranks.extend(ranks.iter().copied());
        }
        fn uses(&self) -> Vec<(usize, usize, u32)> {
            self.uses.clone()
        }
        fn clear_uses(&mut self) {
            self.uses.clear();
        }
        fn fill_plan(&self, window: &[Key]) -> crate::experts::Contents {
            let (mut vram_room, mut host_room) = (self.vram_slots - self.vram.len(), self.host_slots - self.host.len());
            let (mut vram, mut host) = (Vec::new(), Vec::new());
            for key in window.iter().filter(|key| !self.held(key)) {
                if vram_room > 0 {
                    vram_room -= 1;
                    vram.push(*key);
                } else if host_room > 0 {
                    host_room -= 1;
                    host.push(*key);
                }
            }
            (vram, host)
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
    }

    fn keys(range: std::ops::Range<usize>, layer: usize) -> Vec<Key> {
        range.map(|expert| (layer, expert)).collect()
    }

    fn run(warm: &mut Warm, stores: &mut Stores, count: usize) -> Vec<Warmed> {
        let mut events = Vec::new();
        for _ in 0..10_000 {
            let step = warm.step(stores, count, 1000).unwrap();
            if step != Warmed::Going {
                events.push(step);
            }
            if step == Warmed::Done {
                break;
            }
        }
        events
    }

    /// Decides: reading the most recent first, with a state saved before the warm-up is over, loses
    /// nothing the old state listed (but what a request evicted), keeps its order, and lists an
    /// expert a request touched meanwhile once, as the newest.
    #[test]
    fn a_save_during_an_unfinished_warm_up_lists_everything_the_old_state_did_in_recency_order() {
        let (vram, host) = (keys(0..10, 0), keys(0..10, 1));
        let old = json!({"vram": vram, "host": host});
        let mut stores = Stores::new(10, 10);
        let mut warm = Warm::new(2, 16);
        assert_eq!(warm.start(&mut stores, &old).unwrap(), (10, 10));
        // two steps of three: the most recent VRAM experts (9, 8, 7, then 6, 5, 4) are in
        for _ in 0..2 {
            assert_eq!(warm.step(&mut stores, 3, 1000).unwrap(), Warmed::Going);
        }
        assert_eq!(Stores::keys(&stores.vram), keys(4..10, 0), "the most recent first, in the old order");
        assert!(stores.host.is_empty());
        let saved = warm.state(&stores);
        let list = |key: &str| -> Vec<Key> { saved[key].as_array().unwrap().iter().map(|pair| (pair[0].as_u64().unwrap() as usize, pair[1].as_u64().unwrap() as usize)).collect() };
        assert_eq!(list("vram"), vram, "all ten, as listed");
        assert_eq!(list("host"), host);
        // a request touches host-listed (1, 3) and evicts (0, 5)
        stores.touch((1, 3));
        stores.evict((0, 5));
        let saved = warm.state(&stores);
        let list = |key: &str| -> Vec<Key> { saved[key].as_array().unwrap().iter().map(|pair| (pair[0].as_u64().unwrap() as usize, pair[1].as_u64().unwrap() as usize)).collect() };
        let mut expected_vram: Vec<Key> = vram.iter().copied().filter(|key| *key != (0, 5)).collect();
        expected_vram.push((1, 3));
        assert_eq!(list("vram"), expected_vram, "(1, 3) once, newest; (0, 5) gone");
        assert_eq!(list("host"), host.iter().copied().filter(|key| *key != (1, 3)).collect::<Vec<_>>());
        // and finishing the warm-up leaves the old order whole (the host tier too)
        let mut stores = Stores::new(10, 10);
        let mut warm = Warm::new(2, 16);
        warm.start(&mut stores, &old).unwrap();
        run(&mut warm, &mut stores, 3);
        assert_eq!(Stores::keys(&stores.vram), vram);
        assert_eq!(Stores::keys(&stores.host), host);
        let at = |key: Key| stores.placed_in_order.iter().position(|placed| *placed == key).unwrap();
        assert!(at((0, 9)) == 0 && at((1, 9)) < at((1, 8)), "the most recent first in each list");
    }

    /// Decides: the fill offers only the room the stores have, never an expert they hold, the ones
    /// the earlier run's passes used most first, and ends once the stores are full.
    #[test]
    fn the_fill_never_evicts_and_stops_at_capacity() {
        let old = json!({"vram": [[0, 0], [0, 1]], "host": [[0, 2]],
                         "counts": [[1, 5, 9], [0, 7, 7], [1, 3, 3], [0, 0, 100]]});
        let mut stores = Stores::new(6, 5);
        let mut warm = Warm::new(2, 8);
        warm.start(&mut stores, &old).unwrap();
        let events = run(&mut warm, &mut stores, 4);
        assert_eq!(events, vec![Warmed::Listed, Warmed::Done]);
        let (vram, host) = (Stores::keys(&stores.vram), Stores::keys(&stores.host));
        assert_eq!((vram.len(), host.len()), (6, 5), "full, no more");
        for key in [(0, 0), (0, 1)] {
            assert!(vram.contains(&key), "{key:?} was not evicted");
        }
        assert!(host.contains(&(0, 2)));
        // the free VRAM slots (4) go to the most used not held, then the first of the rest by expert and
        // layer; the host tier's go to the next
        let sorted = |mut keys: Vec<Key>| -> Vec<Key> { keys.sort_unstable(); keys };
        assert_eq!(sorted(vram), vec![(0, 0), (0, 1), (0, 7), (1, 0), (1, 3), (1, 5)]);
        assert_eq!(sorted(host), vec![(0, 2), (0, 3), (0, 4), (1, 1), (1, 2)]);
        assert_eq!(warm.filled(&stores), (4, 4, 0));
        // the fill runs on only while there is room: a second pass over a full store places nothing
        let before = stores.placed_in_order.len();
        let mut again = Warm::new(2, 8);
        again.start(&mut stores, &old).unwrap();
        run(&mut again, &mut stores, 4);
        assert_eq!(stores.placed_in_order.len(), before);
    }

    /// Decides: a state as earlier versions saved it (no counts) still warms up, its fill goes by
    /// expert and layer, and the counts a state keeps age by what the run adds to them: a restart's
    /// few passes hardly, a run that adds as many as they hold by half.
    #[test]
    fn an_old_state_loads_and_the_counts_it_keeps_age_by_what_a_run_adds() {
        let old = json!({"vram": [[0, 1]], "host": []});
        let mut stores = Stores::new(3, 0);
        let mut warm = Warm::new(2, 4);
        assert_eq!(warm.start(&mut stores, &old).unwrap(), (1, 0));
        run(&mut warm, &mut stores, 8);
        // no counts: the fixed order, expert then layer: (0, 0) and (1, 0) beside the (0, 1) listed
        let mut held = Stores::keys(&stores.vram);
        held.sort_unstable();
        assert_eq!(held, vec![(0, 0), (0, 1), (1, 0)]);
        assert_eq!(warm.state(&stores)["counts"], json!([]), "nothing counted, nothing kept");
        let order = fill_order(&[0, 0, 0, 5, 0, 0, 0, 0], 2, 4);
        assert_eq!(order[0], (0, 3), "the most used first");
        assert_eq!(&order[1..4], &[(0, 0), (1, 0), (0, 1)], "the rest by expert, then layer");
        // a run that served nothing keeps the counts it was given; a few passes age them by 9 of 12
        // (they held 9, the run adds 3), and a run that adds as many as they held (or more) by half
        let given = json!({"vram": [], "host": [], "counts": [[0, 3, 5], [1, 2, 4]]});
        let mut stores = Stores::new(3, 0);
        let mut warm = Warm::new(2, 4);
        warm.start(&mut stores, &given).unwrap();
        assert_eq!(warm.state(&stores)["counts"], json!([[0, 3, 5], [1, 2, 4]]));
        stores.uses = vec![(0, 3, 2), (1, 0, 1)];
        assert_eq!(warm.state(&stores)["counts"], json!([[0, 3, 6], [1, 0, 1], [1, 2, 3]]));
        stores.uses = vec![(0, 3, 20), (1, 0, 1)];
        assert_eq!(warm.state(&stores)["counts"], json!([[0, 3, 23], [1, 0, 1], [1, 2, 2]]));
    }
}
