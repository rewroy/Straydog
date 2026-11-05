//! The warm start of a partially resident model (Qwen3.8; docs/DESIGN.md §10.1): how often decode and
//! verification rows were routed to each expert of each layer, on the GPU (the expert held a slot) and on
//! the host (the CPU computed it from the mapped banks, or the GPU read it over the bus), kept beside the
//! bundle like the expert stores' warm-up state of a model read on demand; and what a start does with
//! them: the likeliest experts' scores are set before the first request, so that the expert slots fill
//! with them rather than with what the start-up's own passes left, and the OS is asked for the bank pages of the
//! next likeliest ones that no page lock already holds.
//!
//! A prompt's routes are not counted: a prefill takes nearly every expert whatever the prompt says.

use serde_json::{json, Value};

/// What a state of this kind says of itself, and its format: a start reads no state of another (the expert stores'
/// state of a model read on demand has lists of experts, no counts by side).
pub const KIND: &str = "routes";
pub const VERSION: u64 = 1;

/// The routed experts a token takes in a layer (Qwen3.8's top 10).
pub const ROUTES: f64 = 10.0;

/// Bytes of expert reads a single request to the OS asks for: the ranked experts are asked for in
/// requests of this size, the likeliest first, each one of few large reads of the drive rather than a page's fault.
pub const CHUNK_BYTES: u64 = 512 << 20;

/// The counts of the routes by layer and expert (layer-major), and the side of each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Routes {
    layers: usize,
    experts: usize,
    gpu: Vec<u32>,
    host: Vec<u32>,
}

impl Routes {
    pub fn new(layers: usize, experts: usize) -> Routes {
        Routes { layers, experts, gpu: vec![0; layers * experts], host: vec![0; layers * experts] }
    }

    /// Set a layer's counts (the device's, then the host's; a layer outside the model, or counts of another size, are left out).
    pub fn set_layer(&mut self, layer: usize, gpu: &[u32], host: &[u32]) {
        if layer < self.layers && gpu.len() == self.experts && host.len() == self.experts {
            let at = layer * self.experts;
            self.gpu[at..at + self.experts].copy_from_slice(gpu);
            self.host[at..at + self.experts].copy_from_slice(host);
        }
    }

    /// The routes counted on the GPU and on the host.
    pub fn sides(&self) -> (u64, u64) {
        (self.gpu.iter().map(|&count| u64::from(count)).sum(), self.host.iter().map(|&count| u64::from(count)).sum())
    }

    /// The routes counted in all.
    pub fn total(&self) -> u64 {
        let (gpu, host) = self.sides();
        gpu + host
    }

    /// The routes to an expert, on both sides.
    pub fn used(&self, layer: usize, expert: usize) -> u64 {
        let at = layer * self.experts + expert;
        u64::from(self.gpu[at]) + u64::from(self.host[at])
    }

    /// The routes of a layer that went to the host: to an expert the CPU computed, or one the GPU read over the bus (none for a layer outside the
    /// model).
    pub fn host_routes(&self, layer: usize) -> u64 {
        if layer >= self.layers {
            return 0;
        }
        self.host[layer * self.experts..(layer + 1) * self.experts].iter().map(|&count| u64::from(count)).sum()
    }

    fn layer_total(&self, layer: usize) -> u64 {
        (0..self.experts).map(|expert| self.used(layer, expert)).sum()
    }

    /// A layer's experts' routes a token, as the counts say: how many of the layer's [`ROUTES`] each one took on average (all zero without counts).
    pub fn rates(&self, layer: usize) -> Vec<f64> {
        let total = self.layer_total(layer);
        (0..self.experts).map(|expert| if total == 0 { 0.0 } else { ROUTES * self.used(layer, expert) as f64 / total as f64 }).collect()
    }

    /// The experts as (layer, expert) that took any route, the most used first (by their rate in their layer, which the layers compare by; ties by
    /// layer, then expert).
    pub fn ranked(&self) -> Vec<(usize, usize)> {
        let layer_totals: Vec<u64> = (0..self.layers).map(|layer| self.layer_total(layer)).collect();
        let mut ranked: Vec<(usize, usize, f64)> = (0..self.layers)
            .flat_map(|layer| (0..self.experts).map(move |expert| (layer, expert)))
            .filter(|&(layer, expert)| self.used(layer, expert) > 0)
            .map(|(layer, expert)| (layer, expert, self.used(layer, expert) as f64 / layer_totals[layer] as f64))
            .collect();
        ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        ranked.into_iter().map(|(layer, expert, _)| (layer, expert)).collect()
    }

    /// What a later start keeps: these counts (an earlier run's) aged by what `run` adds to them, plus `run`'s own, by the rule the expert stores'
    /// state follows ([`crate::arch::warm::kept_share`]: not at all when the run counted nothing, hardly for a restart's one reply, by half when it
    /// counted as many routes as they held).
    pub fn aged_with(&self, run: &Routes) -> Routes {
        let keep = crate::arch::warm::kept_share(self.total(), run.total());
        let age = |before: &[u32], added: &[u32]| -> Vec<u32> {
            before.iter().zip(added).map(|(&before, &added)| ((f64::from(before) * keep).round() as u32).saturating_add(added)).collect()
        };
        Routes { layers: self.layers, experts: self.experts, gpu: age(&self.gpu, &run.gpu), host: age(&self.host, &run.host) }
    }

    /// The state to keep: every expert counted, as [layer, expert, GPU routes, host routes].
    pub fn to_state(&self) -> Value {
        let counts: Vec<Value> = (0..self.layers)
            .flat_map(|layer| (0..self.experts).map(move |expert| (layer, expert)))
            .filter(|&(layer, expert)| self.used(layer, expert) > 0)
            .map(|(layer, expert)| {
                let at = layer * self.experts + expert;
                json!([layer, expert, self.gpu[at], self.host[at]])
            })
            .collect();
        json!({"kind": KIND, "version": VERSION, "layers": self.layers, "experts": self.experts, "counts": counts})
    }

    /// The counts of a state, if it is one of this kind and format, for a model of `layers` layers of `experts` experts.
    pub fn from_state(state: &Value, layers: usize, experts: usize) -> Result<Routes, String> {
        if state.get("kind").and_then(Value::as_str) != Some(KIND) {
            return Err("a Straydog warm-up state of another kind".into());
        }
        if state.get("version").and_then(Value::as_u64) != Some(VERSION) {
            return Err("a Straydog warm-up state of another format".into());
        }
        if state.get("layers").and_then(Value::as_u64) != Some(layers as u64) || state.get("experts").and_then(Value::as_u64) != Some(experts as u64) {
            return Err("a Straydog warm-up state of another model".into());
        }
        let mut routes = Routes::new(layers, experts);
        for count in state.get("counts").and_then(Value::as_array).ok_or("a Straydog warm-up state without its counts")? {
            let field = |index: usize| count.get(index).and_then(Value::as_u64);
            let (Some(layer), Some(expert), Some(gpu), Some(host)) = (field(0), field(1), field(2), field(3)) else { continue };
            if (layer as usize) < layers && (expert as usize) < experts {
                let at = layer as usize * experts + expert as usize;
                routes.gpu[at] = gpu.min(u64::from(u32::MAX)) as u32;
                routes.host[at] = host.min(u64::from(u32::MAX)) as u32;
            }
        }
        Ok(routes)
    }
}

/// A layer's three expert banks as mapped: where each starts (an address, as a number: the ranges cross to the thread that asks the OS) and
/// the bytes an expert takes of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Banks {
    pub start: [usize; 3],
    pub bytes: [usize; 3],
}

/// What a start asks the OS for: requests of byte ranges ([`CHUNK_BYTES`] of experts each, the likeliest experts first, a request's ranges in
/// address order with those that touch joined), and what they come to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Hints {
    pub requests: Vec<Request>,
    pub experts: usize,
    pub bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Request {
    pub ranges: Vec<(usize, usize)>,
    /// The experts' bytes in the ranges.
    pub bytes: u64,
}

/// The experts to ask the OS for the pages of: the `ranked` ones after the first `skip` (the slots take those), of the layers whose banks `banks`
/// says are not page-locked already (None: they are, or are not the model's), as far as `budget` bytes go.
pub fn plan_hints(ranked: &[(usize, usize)], skip: usize, budget: u64, banks: &dyn Fn(usize) -> Option<Banks>) -> Hints {
    let mut hints = Hints::default();
    let (mut ranges, mut chunk) = (Vec::new(), 0u64);
    for &(layer, expert) in ranked.iter().skip(skip) {
        let Some(banks) = banks(layer) else { continue };
        let size: u64 = banks.bytes.iter().map(|&bytes| bytes as u64).sum();
        if hints.bytes + size > budget {
            break;
        }
        for role in 0..3 {
            ranges.push((banks.start[role] + expert * banks.bytes[role], banks.bytes[role]));
        }
        hints.bytes += size;
        hints.experts += 1;
        chunk += size;
        if chunk >= CHUNK_BYTES {
            hints.requests.push(Request { ranges: join(std::mem::take(&mut ranges)), bytes: std::mem::take(&mut chunk) });
        }
    }
    if !ranges.is_empty() {
        hints.requests.push(Request { ranges: join(ranges), bytes: chunk });
    }
    hints
}

/// Byte ranges in address order, those that touch or overlap joined.
fn join(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut joined: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (start, length) in ranges {
        match joined.last_mut() {
            Some(last) if start <= last.0 + last.1 => last.1 = last.1.max(start + length - last.0),
            _ => joined.push((start, length)),
        }
    }
    joined
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counted(layers: usize, experts: usize, counts: &[(usize, usize, u32, u32)]) -> Routes {
        let mut routes = Routes::new(layers, experts);
        for &(layer, expert, gpu, host) in counts {
            routes.gpu[layer * experts + expert] = gpu;
            routes.host[layer * experts + expert] = host;
        }
        routes
    }

    /// Decides: a state keeps every counted expert with its two sides and nothing else, reads back as it was written, and is refused by a model of another
    /// geometry, a state of another kind or format, and a model read on demand's state (its lists, no counts).
    #[test]
    fn a_state_reads_back_as_written_and_only_for_its_own_model() {
        let routes = counted(3, 8, &[(0, 1, 5, 0), (2, 7, 0, 9), (1, 3, 4, 4)]);
        let state = routes.to_state();
        assert_eq!(state["counts"], json!([[0, 1, 5, 0], [1, 3, 4, 4], [2, 7, 0, 9]]), "by layer, then expert, the counted only");
        assert_eq!(Routes::from_state(&state, 3, 8).unwrap(), routes);
        let reread: Value = serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(Routes::from_state(&reread, 3, 8).unwrap(), routes, "through the file");
        assert!(Routes::from_state(&state, 3, 16).unwrap_err().contains("another model"));
        assert!(Routes::from_state(&state, 4, 8).unwrap_err().contains("another model"));
        let mut newer = state.clone();
        newer["version"] = json!(VERSION + 1);
        assert!(Routes::from_state(&newer, 3, 8).unwrap_err().contains("another format"));
        let stores = json!({"vram": [[0, 1]], "host": [], "counts": [[0, 1, 5]]});
        assert!(Routes::from_state(&stores, 3, 8).unwrap_err().contains("another kind"));
        // (pairs outside the model, and short ones, are dropped; counts past u32 are held at it)
        let odd = json!({"kind": KIND, "version": VERSION, "layers": 3, "experts": 8,
                         "counts": [[3, 0, 1, 1], [0, 8, 1, 1], [0, 0, 1], [1, 1, 5000000000u64, 2]]});
        let read = Routes::from_state(&odd, 3, 8).unwrap();
        assert_eq!(read, counted(3, 8, &[(1, 1, u32::MAX, 2)]));
        assert!(Routes::from_state(&json!({"kind": KIND, "version": VERSION, "layers": 3, "experts": 8}), 3, 8).is_err());
    }

    /// Decides: the earlier counts age as the expert stores' do: whole when the run counted nothing, hardly after a restart's few routes,
    /// by half once it counted as many as they held, and the run's own are added.
    #[test]
    fn earlier_counts_age_by_what_a_run_adds() {
        let earlier = counted(2, 4, &[(0, 0, 60, 40), (1, 2, 0, 100)]);
        let none = Routes::new(2, 4);
        assert_eq!(earlier.aged_with(&none), earlier, "a run that counted nothing keeps them whole");
        let few = counted(2, 4, &[(0, 0, 2, 1)]);
        // (200 held, 3 added: 200/203 of each)
        let aged = earlier.aged_with(&few);
        assert_eq!((aged.gpu[0], aged.host[0]), (59 + 2, 39 + 1), "60 and 40 by 200/203, the run's added");
        assert_eq!(aged.used(1, 2), 99);
        let long = counted(2, 4, &[(1, 3, 150, 250)]);
        let aged = earlier.aged_with(&long);
        assert_eq!((aged.gpu[0], aged.host[0]), (30, 20), "by half: 200 held, 400 added");
        assert_eq!(aged.used(1, 3), 400);
        assert_eq!(aged.used(1, 2), 50);
        assert_eq!(Routes::new(2, 4).aged_with(&few), few, "nothing earlier");
    }

    /// Decides: the rates give each layer's routes a token as the counts share them (ten in all), the ranking puts the experts by that share across
    /// layers, and leaves out those never routed to.
    #[test]
    fn the_ranking_follows_each_experts_share_of_its_layer() {
        // layer 0 saw 100 routes (expert 2: 60, expert 0: 40), layer 1 saw 10 (expert 1: 9, expert 3: 1), layer 2 none
        let routes = counted(3, 4, &[(0, 0, 10, 30), (0, 2, 60, 0), (1, 1, 0, 9), (1, 3, 1, 0)]);
        let rates = routes.rates(0);
        assert_eq!(rates.iter().sum::<f64>(), ROUTES);
        assert!((rates[2] - 6.0).abs() < 1e-12 && (rates[0] - 4.0).abs() < 1e-12);
        assert_eq!(routes.rates(2), vec![0.0; 4], "a layer without counts has no rates");
        // (a layer's few routes weigh as much as another's many: 9 of 10 is the likeliest)
        assert_eq!(routes.ranked(), vec![(1, 1), (0, 2), (0, 0), (1, 3)]);
        assert_eq!(routes.sides(), (71, 39));
        assert_eq!(routes.total(), 110);
    }

    /// Decides: a layer's host routes are those of its experts that went to the host, and a layer outside the model has none.
    #[test]
    fn a_layers_host_routes_are_those_that_went_to_the_host() {
        let routes = counted(3, 4, &[(0, 0, 10, 30), (0, 2, 60, 0), (1, 1, 0, 9), (1, 3, 1, 0)]);
        assert_eq!((routes.host_routes(0), routes.host_routes(1), routes.host_routes(2), routes.host_routes(3)), (30, 9, 0, 0));
    }

    fn mapped(layer: usize) -> Option<Banks> {
        // layers 0 and 1 are not locked, the rest are; each expert takes 1000, 1000 and 500 bytes of its banks
        let base = 1_000_000_000 * (layer + 1);
        (layer < 2).then(|| Banks { start: [base, base + 300_000_000, base + 600_000_000], bytes: [1000, 1000, 500] })
    }

    /// Decides: the hints skip the experts the slots take and the layers that are locked, ask for the likeliest first, stop within the budget, and
    /// join the experts that lie side by side into one range.
    #[test]
    fn the_hints_take_the_next_likeliest_unlocked_experts_within_the_budget() {
        let ranked = [(5, 0), (0, 4), (0, 5), (3, 1), (1, 9), (0, 6), (0, 0), (1, 2)];
        // the first 2 (a locked layer's, and (0, 4)) go to the slots; (0, 5), (3, 1: locked), (1, 9), (0, 6), (0, 0), (1, 2) follow
        let hints = plan_hints(&ranked, 2, u64::MAX, &mapped);
        assert_eq!((hints.experts, hints.bytes), (5, 5 * 2500));
        assert_eq!(hints.requests.len(), 1);
        let ranges = &hints.requests[0].ranges;
        // layer 0's three banks hold experts 5 and 6 side by side (one range) and 0 (another); layer 1's, 9 and 2 apart
        assert_eq!(ranges.len(), 12);
        assert_eq!(ranges.iter().filter(|range| range.0 < 2_000_000_000).count(), 6);
        assert!(ranges.contains(&(1_000_000_000 + 5 * 1000, 2000)), "experts 5 and 6 of layer 0's gate bank, joined");
        assert!(ranges.contains(&(1_000_000_000 + 600_000_000 + 5 * 500, 1000)), "and of its down bank");
        assert!(ranges.contains(&(2_000_000_000 + 9 * 1000, 1000)) && ranges.contains(&(2_000_000_000 + 2 * 1000, 1000)));
        assert!(ranges.windows(2).all(|pair| pair[0].0 + pair[0].1 < pair[1].0), "in address order, none touching");
        // a budget that holds three of them: the likeliest three
        let hints = plan_hints(&ranked, 2, 3 * 2500, &mapped);
        assert_eq!((hints.experts, hints.bytes), (3, 3 * 2500));
        assert!(hints.requests[0].ranges.contains(&(2_000_000_000 + 9 * 1000, 1000)), "(1, 9) is in");
        assert!(!hints.requests[0].ranges.contains(&(1_000_000_000, 1000)), "(0, 0) is not");
        assert_eq!(plan_hints(&ranked, 2, 2499, &mapped), Hints::default(), "a budget under one expert asks for none");
        assert_eq!(plan_hints(&ranked, 100, u64::MAX, &mapped), Hints::default(), "all of them in the slots");
        assert_eq!(plan_hints(&[], 0, u64::MAX, &mapped), Hints::default());
    }

    /// Decides: the experts are asked for in requests of CHUNK_BYTES each, in the order of their ranks, every request in address order.
    #[test]
    fn the_hints_go_in_large_requests_the_likeliest_first() {
        let big = |layer: usize| (layer == 0).then(|| Banks { start: [1 << 40, 2 << 40, 3 << 40], bytes: [1 << 20; 3] });
        // 3 MiB an expert: 512 MiB is 171 experts and a bit; 400 experts make three requests
        let ranked: Vec<(usize, usize)> = (0..400).rev().map(|expert| (0, expert)).collect();
        let hints = plan_hints(&ranked, 0, u64::MAX, &big);
        assert_eq!(hints.experts, 400);
        let sizes: Vec<u64> = hints.requests.iter().map(|request| request.bytes).collect();
        assert_eq!(sizes.len(), 3);
        assert!(sizes[0] >= CHUNK_BYTES && sizes[1] >= CHUNK_BYTES && sizes[2] < CHUNK_BYTES);
        assert_eq!(sizes.iter().sum::<u64>(), hints.bytes);
        // the first request holds the likeliest (the highest numbered, here) experts, the last the least
        let first_gate = hints.requests[0].ranges.iter().find(|range| range.0 >> 40 == 1).unwrap();
        assert_eq!(first_gate.0 + first_gate.1, (1 << 40) + 400 * (1 << 20), "ending at the top expert");
        let last_gate = hints.requests[2].ranges.iter().find(|range| range.0 >> 40 == 1).unwrap();
        assert_eq!(last_gate.0, 1 << 40, "beginning at expert 0");
        // (each request's experts lie side by side, so each is a range a bank)
        assert!(hints.requests.iter().all(|request| request.ranges.len() == 3));
    }

    #[test]
    fn ranges_that_touch_or_overlap_are_one() {
        assert_eq!(join(vec![(100, 10), (110, 10), (130, 5), (90, 15), (500, 1)]), vec![(90, 30), (130, 5), (500, 1)]);
        assert_eq!(join(Vec::new()), Vec::new());
    }
}
