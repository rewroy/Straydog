//! A partially resident decode step, pipelined (docs/DESIGN.md §10). The step is queued whole (a
//! recorded graph): VRAM experts run by a device table of slot addresses (zero for an expert
//! outside VRAM: its routes run nothing there), and each layer exports its routes, their weights
//! and its MoE input to a host exchange, then waits on the device for the host's sum of the other
//! routes and adds it. The host serves the layers in order: it waits for a layer's export, runs the
//! routes the table does not take on the CPU from the host tier (an expert neither tier holds is
//! read from the drive into the host tier first, while the CPU runs the others), and publishes
//! their sum. The shared expert and the VRAM experts run on the GPU meanwhile, and no host launch
//! or read-back sits between the layers. A layer also exports the next layer's router applied to
//! its own MoE input: the experts that guess names and neither tier holds start reading from the
//! drive a layer early (the guess only moves reads: every route still runs its own expert).
//!
//! The device reads the table during a step, so it changes only between steps. Experts the CPU ran
//! often (a decaying count of their routes) move into VRAM: into a free slot, or one a clearly
//! weaker expert leaves (its entry cleared before the step, its slot written only once the device
//! has passed that point); each is uploaded once the host has served its layer in the step, and
//! enters the table at the next. A pass that takes slots otherwise (a prompt's) has the table built
//! from the store again.

use std::collections::HashMap;

use super::{name, Engine, Tensors};
use crate::exchange::{Exchange, HEADER};
use crate::experts::HostSource;
use crate::gpu::Result;
use crate::ops::Buffer;
use crate::storage;
use crate::transfer::{self, Mark, Stream};

/// The graph a partially resident decode step records.
const HYBRID_GRAPH: u64 = 2;
/// Experts moved into VRAM at most per step.
const SWAPS: usize = 4;
/// A route's count decays by this per step (about the last 20 steps count).
const DECAY: f32 = 0.95;

fn round(bytes: u64) -> u64 {
    bytes.next_multiple_of(256)
}

/// An engine's partially resident decode steps.
pub(super) struct Hybrid {
    exchange: Exchange,
    /// Per layer of the engine, its experts' slot addresses (zero: not in VRAM).
    table: Buffer,
    /// The table's entries as the step queued last reads them (layer-major from the engine's
    /// first layer).
    entries: Vec<u64>,
    /// Per layer: its slots' bytes, its experts' formats (gate, up, down) and their offsets in a slot.
    slot_bytes: Vec<u64>,
    kinds: Vec<[u32; 3]>,
    offsets: Vec<[u64; 3]>,
    /// Layers whose entries changed since the table was written.
    dirty: Vec<bool>,
    /// The store's VRAM generation the entries follow (None: build them from the store).
    built: Option<u64>,
    /// Per layer: experts a slot was freed for, promoted once the host has served the layer.
    planned: Vec<Vec<usize>>,
    /// Promotions uploading: (layer index, expert, slot address, the upload's mark).
    landing: Vec<(usize, usize, u64, Mark)>,
    /// Each (layer index, expert)'s decaying count of routes.
    scores: Vec<f32>,
    /// Experts the CPU ran in the last step (the candidates for VRAM).
    recent: Vec<(usize, usize)>,
    /// Bytes of a layer's export (routes, weights, input) and of its result.
    export_stride: u64,
    result_stride: u64,
    /// Whether the step's graph is recorded; whether ops are being queued for a step.
    graph: bool,
    queuing: bool,
    /// The step queued and not yet served: the exports before it, and a mark after it.
    pending: Option<(u32, Mark)>,
    /// The next layer's routes as guessed from a layer's MoE input: router logits, then the top
    /// experts and their weights.
    guess: Buffer,
    /// A page-locked copy of one expert, for one only VRAM holds (rare: the table not yet showing it).
    scratch: Option<storage::Buffer>,
    /// Routes the GPU ran by table.
    gpu_routes: u64,
    /// Seconds the host spent: waiting for exports, running CPU experts, waiting for drive reads,
    /// between steps on the table, queuing the step; and the steps.
    timing: [f64; 5],
    steps: u64,
    /// Routes that waited for a drive read, and reads started on a guess.
    waits: u64,
    guesses: u64,
}

impl Drop for Hybrid {
    fn drop(&mut self) {
        if let Some(scratch) = &mut self.scratch {
            // SAFETY: pinned when made; every copy into it was waited for.
            let _ = unsafe { transfer::unpin_host(scratch.as_mut_slice().as_mut_ptr()) };
        }
    }
}

impl<T: Tensors> Engine<'_, T> {
    /// Whether a pass of `n` tokens runs as a pipelined partially resident decode step (made ready
    /// at the first: the exchange, the table).
    pub(super) fn hybrid_step(&mut self, n: usize) -> Result<bool> {
        if n != 1 || self.resident.is_some() || self.cpu.is_none() || self.compare.is_some() || self.hybrid_off {
            return Ok(false);
        }
        if self.hybrid.is_some() {
            return Ok(true);
        }
        if !self.gpu.exchange_supported()? || std::env::var_os("Q35_HYBRID_SERIAL").is_some() {
            self.hybrid_off = true;
            return Ok(false);
        }
        let hp = self.hp.clone();
        let (layers, experts, k, w) = (self.settings.layers.len(), hp.experts, hp.experts_used, hp.width);
        let (export_stride, result_stride) = (round(((3 * k + w) * 4) as u64), round((w * 4) as u64));
        let exchange = self.gpu.exchange(HEADER + layers as u64 * (export_stride + result_stride))?;
        let table = self.gpu.buffer((layers * experts * 8) as u64)?;
        let guess = self.gpu.buffer(((experts + 2 * k) * 4) as u64)?;
        let slot_bytes = self.settings.layers.clone()
            .map(|layer| self.experts.slot_size(layer).ok_or_else(|| format!("layer {layer} has no expert slots")))
            .collect::<Result<Vec<u64>>>()?;
        let (kinds, offsets) = self.settings.layers.clone().map(|layer| self.expert_layout(layer)).collect::<Result<Vec<_>>>()?
            .into_iter().unzip();
        self.hybrid = Some(Hybrid {
            exchange, table, entries: vec![0; layers * experts], slot_bytes, kinds, offsets, dirty: vec![true; layers], built: None,
            planned: vec![Vec::new(); layers], landing: Vec::new(), scores: vec![0.0; layers * experts], recent: Vec::new(),
            export_stride, result_stride, graph: false, queuing: false, pending: None, guess, scratch: None, gpu_routes: 0,
            timing: [0.0; 5], steps: 0, waits: 0, guesses: 0,
        });
        Ok(true)
    }

    /// Whether ops are being queued for a pipelined decode step (the MoE then exchanges with the host).
    pub(super) fn hybrid_queuing(&self) -> bool {
        self.hybrid.as_ref().is_some_and(|hybrid| hybrid.queuing)
    }

    /// Routes the GPU ran by table.
    pub(super) fn hybrid_routes(&self) -> u64 {
        self.hybrid.as_ref().map_or(0, |hybrid| hybrid.gpu_routes)
    }

    /// Pipelined decode steps so far, and the host's seconds in them: waiting for the layers'
    /// exports, running CPU experts, waiting for drive reads, keeping the table between steps,
    /// queuing the steps; then the routes that waited for a drive read and the reads started on a
    /// guess.
    pub fn hybrid_timing(&self) -> (u64, [f64; 5], [u64; 2]) {
        self.hybrid.as_ref().map_or((0, [0.0; 5], [0; 2]), |hybrid| (hybrid.steps, hybrid.timing, [hybrid.waits, hybrid.guesses]))
    }

    fn hybrid_mut(&mut self) -> Result<&mut Hybrid> {
        self.hybrid.as_mut().ok_or_else(|| "no partially resident decode state".into())
    }

    /// Queue a decode step (its positions and embedding written): the table brought up to date,
    /// then the step's graph (recorded its first time).
    pub(super) fn hybrid_queue(&mut self) -> Result<()> {
        let started = std::time::Instant::now();
        self.hybrid_table()?;
        let hybrid = self.hybrid_mut()?;
        hybrid.timing[3] += started.elapsed().as_secs_f64();
        hybrid.steps += 1;
        let started = std::time::Instant::now();
        let direct = std::env::var_os("Q35_HYBRID_GRAPH").is_some_and(|value| value == "0");
        let hybrid = self.hybrid_mut()?;
        let before = hybrid.exchange.exported();
        let recorded = hybrid.graph;
        if direct || !recorded {
            if !direct {
                self.gpu.graph_begin()?;
            }
            self.hybrid_mut()?.queuing = true;
            let queued = self.layers_and_head(1);
            self.hybrid_mut()?.queuing = false;
            if !direct {
                self.gpu.graph_end(HYBRID_GRAPH, queued.is_ok())?;
            }
            queued?;
            self.hybrid_mut()?.graph = !direct;
        }
        if !direct && !self.gpu.graph_run(HYBRID_GRAPH)? {
            return Err("the qwen35moe decode graph went missing".into());
        }
        let end = self.gpu.mark(Stream::Compute)?;
        let hybrid = self.hybrid_mut()?;
        hybrid.pending = Some((before, end));
        hybrid.timing[4] += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// The MoE's export of a queued step's layer: its routes, their weights, its input, and the
    /// next layer's routes guessed from it (when this engine runs that layer).
    pub(super) fn hybrid_export(&mut self, layer: usize) -> Result<()> {
        let (k, w, experts, work) = (self.hp.experts_used, self.hp.width, self.hp.experts, self.work);
        let index = layer - self.settings.layers.start;
        let hybrid = self.hybrid.as_ref().ok_or("no partially resident decode state")?;
        let (exchange, offset, guess) = (hybrid.exchange, HEADER + index as u64 * hybrid.export_stride, hybrid.guess);
        let mut parts = vec![(work.ids.base().cast_const(), k as u64), (work.weights.base().cast_const(), k as u64),
                             (work.x.base().cast_const(), w as u64)];
        if layer + 1 < self.settings.layers.end {
            self.linear(&name(layer + 1, "ffn_gate_inp"), work.x.base(), 1, guess.base())?;
            self.gpu.route(guess.base(), None, 1, experts as u64, k as u64, 0, true, 1.0, guess.at(experts), guess.at(experts + k))?;
            parts.push((guess.at(experts).cast_const(), k as u64));
        }
        self.gpu.export(&parts, &exchange, offset)
    }

    /// The routed experts of a queued step's layer: those in VRAM by table into `out`, then (once
    /// the host publishes it) the host's sum of the others added.
    pub(super) fn hybrid_experts(&mut self, layer: usize, kinds: [u32; 3], offsets: [u64; 3]) -> Result<()> {
        let hp = self.hp.clone();
        let (w, k, ff, experts, work) = (hp.width, hp.experts_used, hp.expert_ff, hp.experts, self.work);
        let index = layer - self.settings.layers.start;
        let hybrid = self.hybrid.as_ref().ok_or("no partially resident decode state")?;
        let table = hybrid.table.slice((index * experts * 8) as u64, (experts * 8) as u64);
        let (slot_bytes, exchange) = (hybrid.slot_bytes[index], hybrid.exchange);
        let result = HEADER + self.settings.layers.len() as u64 * hybrid.export_stride + index as u64 * hybrid.result_stride;
        self.gpu.experts_table(&table, experts as u64, work.ids.base(), slot_bytes, offsets, kinds, w as u64, ff as u64, 1, k as u64,
                               work.x.base(), Some(work.weights.base()), 0.0, true, work.hidden.base(), work.out.base())?;
        self.gpu.await_add(&exchange, result, None, work.out.base(), w as u64)
    }

    /// Between steps: the table's entries from the store again when a pass changed its slots, else
    /// the last step's promotions entered; the next promotions planned (their slots' experts out);
    /// the changed rows written.
    fn hybrid_table(&mut self) -> Result<()> {
        let (first, experts) = (self.settings.layers.start, self.hp.experts);
        // Fills that landed since (a prompt's prefetches, a warm start's) upload now.
        self.experts.poll(&mut self.gpu)?;
        let generation = self.experts.vram_generation();
        let hybrid = self.hybrid.as_mut().ok_or("no partially resident decode state")?;
        if hybrid.built != Some(generation) {
            hybrid.landing.clear();
            hybrid.planned.iter_mut().for_each(Vec::clear);
            let old = std::mem::replace(&mut hybrid.entries, vec![0; hybrid.scores.len()]);
            let mut complete = true;
            for (layer, expert) in self.experts.vram_keys() {
                if !self.settings.layers.contains(&layer) {
                    continue;
                }
                let base = self.experts.vram_ready(&mut self.gpu, layer, expert)?;
                complete &= base.is_some();
                hybrid.entries[(layer - first) * experts + expert] = base.unwrap_or(0);
            }
            for (index, dirty) in hybrid.dirty.iter_mut().enumerate() {
                *dirty |= old[index * experts..(index + 1) * experts] != hybrid.entries[index * experts..(index + 1) * experts];
            }
            hybrid.built = complete.then_some(generation);
        } else {
            for (index, expert, base, mark) in std::mem::take(&mut hybrid.landing) {
                self.gpu.wait_mark(Stream::Compute, mark)?;
                hybrid.entries[index * experts + expert] = base;
                hybrid.dirty[index] = true;
            }
        }
        let victims = self.plan_promotions()?;
        let hybrid = self.hybrid.as_mut().ok_or("no partially resident decode state")?;
        for &(index, expert) in &victims {
            hybrid.entries[index * experts + expert] = 0;
            hybrid.dirty[index] = true;
        }
        for index in 0..hybrid.dirty.len() {
            if std::mem::take(&mut hybrid.dirty[index]) {
                let row = hybrid.table.slice((index * experts * 8) as u64, (experts * 8) as u64);
                self.gpu.expert_table(&row, &hybrid.entries[index * experts..(index + 1) * experts], hybrid.slot_bytes[index])?;
            }
        }
        if !victims.is_empty() {
            // Their slots are written only once the device has passed the rows that no longer
            // name them (and the steps before, which did).
            let after = self.gpu.mark(Stream::Compute)?;
            for &(index, expert) in &victims {
                let before = self.experts.vram_generation();
                let demoted = self.experts.demote(&mut self.gpu, first + index, expert, after)?;
                let now = self.experts.vram_generation();
                let hybrid = self.hybrid_mut()?;
                hybrid.follow(before, now);
                if !demoted {
                    // (still in VRAM, out of the table: build it from the store again)
                    hybrid.built = None;
                }
            }
        }
        let hybrid = self.hybrid_mut()?;
        hybrid.scores.iter_mut().for_each(|score| *score *= DECAY);
        Ok(())
    }

    /// The next step's promotions: the experts the CPU ran in the last step, strongest first, each
    /// given a free slot of its size, else the slot of a clearly weaker VRAM expert (returned: to
    /// leave the table and VRAM before the step).
    fn plan_promotions(&mut self) -> Result<Vec<(usize, usize)>> {
        let (first, experts) = (self.settings.layers.start, self.hp.experts);
        let hybrid = self.hybrid.as_mut().ok_or("no partially resident decode state")?;
        let mut candidates: Vec<(f32, usize, usize)> =
            hybrid.recent.drain(..).map(|(index, expert)| (hybrid.scores[index * experts + expert], index, expert)).collect();
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        candidates.dedup_by(|a, b| a.1 == b.1 && a.2 == b.2);
        candidates.truncate(SWAPS);
        // The weakest experts the table runs, weakest first.
        let mut residents: Vec<(f32, usize, usize)> = self.experts.vram_keys().into_iter()
            .filter(|&(layer, _)| self.settings.layers.contains(&layer))
            .map(|(layer, expert)| (layer - first, expert))
            .filter(|&(index, expert)| hybrid.entries[index * experts + expert] != 0)
            .map(|(index, expert)| (hybrid.scores[index * experts + expert], index, expert))
            .collect();
        let weakest = (SWAPS * 4).min(residents.len());
        if weakest > 0 && weakest < residents.len() {
            residents.select_nth_unstable_by(weakest - 1, |a, b| a.0.total_cmp(&b.0));
            residents.truncate(weakest);
        }
        residents.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut taken: HashMap<u64, usize> = HashMap::new();
        let mut victims = Vec::new();
        let mut used = vec![false; residents.len()];
        for (score, index, expert) in candidates {
            let layer = first + index;
            let class = hybrid.slot_bytes[index];
            let reserved = taken.entry(class).or_default();
            if self.experts.free_slots(layer) > *reserved {
                *reserved += 1;
                hybrid.planned[index].push(expert);
                continue;
            }
            // Hysteresis: only a clearly weaker expert gives its slot up.
            let found = residents.iter().enumerate().find(|(at, resident)| {
                !used[*at] && self.experts.same_class(first + resident.1, layer) && score > resident.0 * 1.25 + 0.05
            }).map(|(at, _)| at);
            if let Some(at) = found {
                used[at] = true;
                *reserved += 1;
                victims.push((residents[at].1, residents[at].2));
                hybrid.planned[index].push(expert);
            }
        }
        Ok(victims)
    }

    /// Serve the queued step's layers on the host: each layer's routes outside the table run on the
    /// CPU and their sum published. On a failure the device's waits are cancelled and the step
    /// drained before the error returns.
    pub(super) fn hybrid_serve(&mut self) -> Result<()> {
        let Some(hybrid) = self.hybrid.as_mut() else { return Ok(()) };
        let Some((before, end)) = hybrid.pending.take() else { return Ok(()) };
        let mut served = Ok(());
        for index in 0..self.settings.layers.len() {
            served = self.serve_layer(index, before.wrapping_add(index as u32 + 1), end);
            if served.is_err() {
                break;
            }
        }
        if served.is_err() {
            self.hybrid_mut()?.pending = Some((before, end));
            self.hybrid_abandon();
        }
        served
    }

    /// Cancel the queued step's waits for the host (its output void) and drain the device.
    pub(super) fn hybrid_abandon(&mut self) {
        let Some(hybrid) = self.hybrid.as_mut() else { return };
        let Some((_, end)) = hybrid.pending.take() else { return };
        let exchange = hybrid.exchange;
        exchange.cancel(true);
        // (a device that fails here fails the next pass, which reports it)
        let _ = self.gpu.mark_done(end, true);
        exchange.restart();
    }

    /// After the step's logits were read (the device idle): whether a wait of the step gave up.
    pub(super) fn hybrid_check(&mut self) -> Result<()> {
        if let Some(hybrid) = &self.hybrid {
            if hybrid.exchange.failed() {
                hybrid.exchange.restart();
                return Err("the GPU gave up waiting for the CPU's experts".into());
            }
        }
        Ok(())
    }

    fn serve_layer(&mut self, index: usize, count: u32, end: Mark) -> Result<()> {
        let hp = self.hp.clone();
        let (w, k, experts) = (hp.width, hp.experts_used, hp.experts);
        let layer = self.settings.layers.start + index;
        let layers = self.settings.layers.len() as u64;
        let hybrid = self.hybrid.as_ref().ok_or("no partially resident decode state")?;
        let exchange = hybrid.exchange;
        let export = HEADER + index as u64 * hybrid.export_stride;
        let result = HEADER + layers * hybrid.export_stride + index as u64 * hybrid.result_stride;
        let started = std::time::Instant::now();
        exchange.wait_export(&mut self.gpu, count, end)?;
        let waited = std::time::Instant::now();
        let guessed = layer + 1 < self.settings.layers.end;
        let mut packed = vec![0f32; if guessed { 3 * k + w } else { 2 * k + w }];
        exchange.read(export, &mut packed)?;
        let ids: Vec<usize> = packed[..k].iter().map(|id| id.to_bits() as usize).collect();
        let next: Vec<usize> = packed[2 * k + w..].iter().map(|id| id.to_bits() as usize).collect();
        if ids.iter().chain(&next).any(|&id| id >= experts) {
            return Err("a qwen35moe route outside the experts".into());
        }
        // The routes the table takes ran on the GPU; the host starts the drive reads of those
        // neither tier holds before it runs the others.
        let mut cpu = Vec::new();
        let mut gpu_routes = 0;
        for (route, &id) in ids.iter().enumerate() {
            let hybrid = self.hybrid.as_mut().ok_or("no partially resident decode state")?;
            hybrid.scores[index * experts + id] += 1.0;
            if hybrid.entries[index * experts + id] != 0 {
                gpu_routes += 1;
                self.experts.touch_vram(layer, id);
                continue;
            }
            hybrid.recent.push((index, id));
            if matches!(self.experts.host_source(&mut self.gpu, layer, id)?, HostSource::Absent) {
                self.experts.fetch_host(&mut self.gpu, layer, id, true)?;
            }
            cpu.push(route);
        }
        // The next layer's guessed experts that neither tier holds, read behind this layer's.
        for &id in &next {
            let hybrid = self.hybrid.as_ref().ok_or("no partially resident decode state")?;
            if hybrid.entries[(index + 1) * experts + id] == 0
                && matches!(self.experts.host_source(&mut self.gpu, layer + 1, id)?, HostSource::Absent)
            {
                self.experts.fetch_host(&mut self.gpu, layer + 1, id, false)?;
                self.hybrid_mut()?.guesses += 1;
            }
        }
        self.hybrid_mut()?.gpu_routes += gpu_routes;
        let mut sum = vec![0f32; w];
        let mut waiting = Vec::new();
        let mut ready = Vec::new();
        for &route in &cpu {
            match self.experts.host_source(&mut self.gpu, layer, ids[route])? {
                HostSource::Ready(roles) => ready.push((route, roles)),
                HostSource::Reading => waiting.push(route),
                HostSource::Vram => self.cpu_from_vram(layer, ids[route], packed[k + route], &packed[2 * k..2 * k + w], &mut sum)?,
                HostSource::Absent => return Err("an expert's drive read failed".into()),
            }
        }
        self.cpu_experts(layer, &ready, &packed, &mut sum)?;
        self.hybrid_mut()?.waits += waiting.len() as u64;
        let mut read_wait = 0.0;
        while !waiting.is_empty() {
            let reading = std::time::Instant::now();
            self.experts.wait_reads(&mut self.gpu)?;
            read_wait += reading.elapsed().as_secs_f64();
            let mut landed = Vec::new();
            let mut still = Vec::new();
            for route in waiting {
                match self.experts.host_source(&mut self.gpu, layer, ids[route])? {
                    HostSource::Ready(roles) => landed.push((route, roles)),
                    HostSource::Reading => still.push(route),
                    HostSource::Vram => self.cpu_from_vram(layer, ids[route], packed[k + route], &packed[2 * k..2 * k + w], &mut sum)?,
                    HostSource::Absent => return Err("an expert's drive read failed".into()),
                }
            }
            self.cpu_experts(layer, &landed, &packed, &mut sum)?;
            waiting = still;
        }
        self.cpu_routes += cpu.len() as u64;
        exchange.publish(result, &sum)?;
        let hybrid = self.hybrid_mut()?;
        hybrid.timing[0] += (waited - started).as_secs_f64();
        hybrid.timing[1] += waited.elapsed().as_secs_f64() - read_wait;
        hybrid.timing[2] += read_wait;
        // The device runs on; the experts planned for VRAM upload now (the host is done with this
        // layer's copies in this step).
        let planned = std::mem::take(&mut self.hybrid_mut()?.planned[index]);
        for expert in planned {
            let before = self.experts.vram_generation();
            let promoted = self.experts.promote(&mut self.gpu, layer, expert)?;
            let now = self.experts.vram_generation();
            let hybrid = self.hybrid_mut()?;
            hybrid.follow(before, now);
            if let Some((base, mark)) = promoted {
                hybrid.landing.push((index, expert, base, mark));
            }
        }
        Ok(())
    }

    /// sum += the CPU's run of `routes` (each route with its expert's roles in host memory).
    fn cpu_experts(&mut self, layer: usize, routes: &[(usize, [*const u8; 3])], packed: &[f32], sum: &mut [f32]) -> Result<()> {
        if routes.is_empty() {
            return Ok(());
        }
        let (w, k, ff) = (self.hp.width, self.hp.experts_used, self.hp.expert_ff);
        let index = layer - self.settings.layers.start;
        let kinds = self.hybrid.as_ref().ok_or("no partially resident decode state")?.kinds[index];
        let pick = |role: usize| -> Vec<*const u8> { routes.iter().map(|(_, roles)| roles[role]).collect() };
        let weights: Vec<f32> = routes.iter().map(|&(route, _)| packed[k + route]).collect();
        let mut out = vec![0f32; w];
        let engine = self.cpu.clone().ok_or("no CPU expert engine")?;
        // SAFETY: each expert's matrices lie in its host slot, which nothing fills, reads into or
        // evicts before this call returns (this thread drives the store).
        unsafe { engine.run(kinds, &pick(0), &pick(1), &pick(2), &weights, w, ff, &packed[2 * k..2 * k + w], &mut out)? };
        for (total, value) in sum.iter_mut().zip(&out) {
            *total += value;
        }
        Ok(())
    }

    /// sum += weight times `layer`'s `expert` on the CPU, from a copy out of its VRAM slot (one the
    /// table does not show yet).
    fn cpu_from_vram(&mut self, layer: usize, expert: usize, weight: f32, x: &[f32], sum: &mut [f32]) -> Result<()> {
        let slot = self.experts.slot_size(layer).ok_or("a layer without expert slots")? as usize;
        let hybrid = self.hybrid_mut()?;
        if hybrid.scratch.as_ref().is_none_or(|scratch| scratch.capacity() < slot) {
            let mut scratch = storage::Buffer::new(slot);
            // SAFETY: the buffer lives in the hybrid state until its drop, which unpins it first.
            unsafe { transfer::pin_host(scratch.as_mut_slice().as_mut_ptr(), scratch.capacity())? };
            if let Some(mut old) = hybrid.scratch.replace(scratch) {
                // SAFETY: pinned when made; its copies were waited for.
                unsafe { transfer::unpin_host(old.as_mut_slice().as_mut_ptr())? };
            }
        }
        let target = hybrid.scratch.as_mut().map(|scratch| scratch.as_mut_slice().as_mut_ptr()).ok_or("no scratch")?;
        // SAFETY: page-locked, of the slot's bytes.
        if !unsafe { self.experts.copy_out(&mut self.gpu, layer, expert, target)? } {
            return Err("an expert left VRAM before the host could copy it".into());
        }
        let index = layer - self.settings.layers.start;
        let (kinds, offsets) = {
            let hybrid = self.hybrid.as_ref().ok_or("no partially resident decode state")?;
            (hybrid.kinds[index], hybrid.offsets[index])
        };
        // SAFETY: the roles lie back to back in the copy, at the slot's offsets.
        let roles = offsets.map(|offset| unsafe { target.add(offset as usize).cast_const() });
        let (w, ff) = (self.hp.width, self.hp.expert_ff);
        let mut out = vec![0f32; w];
        let engine = self.cpu.clone().ok_or("no CPU expert engine")?;
        // SAFETY: the copy is this thread's until the next copy_out.
        unsafe { engine.run(kinds, &[roles[0]], &[roles[1]], &[roles[2]], &[weight], w, ff, x, &mut out)? };
        for (total, value) in sum.iter_mut().zip(&out) {
            *total += value;
        }
        Ok(())
    }
}

impl Hybrid {
    /// After the engine's own change to the store: the entries follow it when they followed the
    /// store before it.
    fn follow(&mut self, before: u64, after: u64) {
        if self.built == Some(before) {
            self.built = Some(after);
        }
    }
}
