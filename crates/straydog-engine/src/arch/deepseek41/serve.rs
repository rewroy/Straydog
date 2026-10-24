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

use super::engine::{divided, forward_split, pass_row_bytes, verify_split, Engine, Settings as EngineSettings, Snapshot, ATTENTION_ROWS,
                    OWN_ROWS};
use super::Hparams;
use crate::arch::served::{self, Placed, Session, Settings};
use crate::bundle;
use crate::experts::Contents;
use crate::gpu::Backend;
use crate::group::SplitMode;
use crate::pinned::Pinned;

use served::{RESERVE, SCRATCH};


/// DeepSeek-V4.1 served: its placed engines in a session.
pub type Model = Session<Engines>;

/// Checkpoints kept, each the whole sequence state where a reply's generation prompt began (a
/// conversation switched away from comes back from there).
pub const CHECKPOINTS: usize = 4;
/// Host memory the checkpoints take at most, together and at any moment (a 20K-token
/// conversation's ~140 MB): the oldest give theirs up before a new one is taken, and one alone
/// over it (past ~160K tokens) is not taken.
pub const CHECKPOINT_BYTES: u64 = 1 << 30;

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
    /// Host bytes they may take ([`CHECKPOINT_BYTES`]).
    budget: u64,
}

impl served::Engines for Engines {
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        forward_split(&mut self.engines, tokens)
    }

    fn reset(&mut self) {
        self.engines.iter_mut().for_each(Engine::reset);
    }

    fn counts(&self) -> (u64, u64, u64) {
        self.engines.iter().map(Engine::cache_counts)
            .fold((0, 0, 0), |sum, counts| (sum.0 + counts.hits, sum.1 + counts.host_hits, sum.2 + counts.drive_reads))
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

    fn warm(&mut self, vram: &[(usize, usize)], host: &[(usize, usize)]) -> Result<usize, String> {
        let mut reading = 0;
        for engine in &mut self.engines {
            reading += engine.warm_experts(vram, host)?;
        }
        Ok(reading)
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

    fn gpus(&self) -> Vec<&Backend> {
        self.engines.iter().flat_map(Engine::gpus).collect()
    }

    fn settle(&mut self) -> Result<(), String> {
        self.engines.iter_mut().try_for_each(Engine::settle)
    }

    fn verify_rows(&self) -> usize {
        self.verify_rows
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        verify_split(&mut self.engines, tokens)
    }

    fn rewind(&mut self, keep: usize) -> Result<(), String> {
        self.engines.iter_mut().try_for_each(|engine| engine.rewind(keep))
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
            let snapshots = self.saved[slot].as_ref().ok_or("the Straydog checkpoint was given up")?;
            return self.engines.iter_mut().zip(snapshots).try_for_each(|(engine, snapshot)| engine.restore(snapshot));
        }
        self.saved[slot] = None;
        self.order.retain(|&held| held != slot);
        // Room first, the copy after: the budget bounds the host memory at its peak too.
        let required: u64 = self.engines.iter().map(Engine::snapshot_bytes).sum();
        let held: Vec<(usize, u64)> = self.order.iter()
            .map(|&held| (held, self.saved[held].iter().flatten().map(Snapshot::bytes).sum())).collect();
        let Some(given_up) = evictions(&held, required, self.budget) else { return Ok(()) };
        for oldest in given_up {
            self.saved[oldest] = None;
            self.order.retain(|&held| held != oldest);
        }
        self.saved[slot] = Some(self.engines.iter_mut().map(Engine::snapshot).collect::<Result<Vec<_>, _>>()?);
        self.order.push(slot);
        Ok(())
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
    let pins = bundle::on_demand(quantization)?;
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
            // (Engram's tables are read a row at a time, never resident)
            _ if crate::pinned::gathered(name) => {}
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
    // The largest of the engine's layers' experts, each device holding its rows of it (whole
    // 256-byte units, as the store lays them out): the store takes VRAM as so many of these.
    let slot_bytes = |(devices, range): &(Vec<usize>, std::ops::Range<usize>)| {
        let largest = range.clone().map(|layer| expert_bytes[layer] / hp.experts as u64).max().unwrap_or(1).max(1);
        (largest / devices.len() as u64).next_multiple_of(256) * devices.len() as u64
    };
    // The sequence state of an engine's layers, on each of its devices: window rings, and each
    // kv source's compressed rows and keys.
    let state = |range: &std::ops::Range<usize>| -> u64 {
        let ring = (hp.window + settings.batch.min(ATTENTION_ROWS)) * hp.head_dim * 4 * range.len();
        let sources: usize = hp.layers.iter().enumerate().filter(|(layer, _)| range.contains(layer))
            .filter_map(|(layer, plan)| plan.compressed.filter(|c| c.kv_source == layer))
            .map(|c| (settings.context / c.ratio + 1) * (hp.head_dim + hp.index_dim) * 4).sum();
        (ring + sources) as u64
    };
    let total_experts: u64 = expert_bytes.iter().sum();
    // Per engine: the experts its devices hold in VRAM together (each device its rows of every
    // one), from what each device has left after its resident share, state and work buffers.
    // An engine's budget: the least of its devices' (its devices hold equal shares).
    let budget = |devices: &[usize]| devices.iter()
        .map(|device| settings.vram_bytes[settings.devices.iter().position(|other| other == device).unwrap()]).min().unwrap_or(0);
    let mut slots = Vec::new();
    for group in &groups {
        let (devices, range) = group;
        let vram = budget(devices);
        let parts = devices.len() as u64;
        let head = if range.end == layers { head_whole + head_parted.div_ceil(parts) } else { 0 };
        let resident: u64 = range.clone().map(|layer| whole[layer] + parted[layer].div_ceil(parts)).sum::<u64>() + head;
        // a pass of up to the batch's rows, layer by layer: its buffers of every row (in the arena
        // lent to the expert store between long passes), and a few rows' own
        let rows = (settings.batch + OWN_ROWS.min(settings.batch)) as u64 * pass_row_bytes(&hp, devices.len(), range.clone());
        let fixed = resident + state(range) + rows + SCRATCH + RESERVE;
        let free = vram.checked_sub(fixed)
            .ok_or_else(|| format!("DeepSeek-V4.1 layers {range:?} need more than {:.1} GiB of VRAM on GPU {devices:?}", vram as f64 / (1u64 << 30) as f64))?;
        // Each device's share of an expert is 1/parts of a slot: free / share experts.
        let held = (free / (slot_bytes(group) / parts)) as usize;
        slots.push(held.min(range.len() * hp.experts) / devices.len() * devices.len());
    }
    let in_vram: u64 = groups.iter().zip(&slots).map(|(group, &slots)| slots as u64 * slot_bytes(group)).sum();
    // (planned, the tier leaves the checkpoints their host memory)
    let host = match settings.host_expert_bytes {
        Some(_) => served::host_tier(settings, total_experts.saturating_sub(in_vram)),
        None => served::host_tier(settings, u64::MAX).saturating_sub(CHECKPOINT_BYTES).min(total_experts.saturating_sub(in_vram)),
    };
    let whole_bytes = files.whole_bytes(|_| true);
    let files: &'static Pinned = Box::leak(Box::new(files));
    let placed = Instant::now();
    let state_bytes = groups.iter().map(|(devices, range)| state(range) * devices.len() as u64).sum();
    // The page-locked host tier is shared out by the engines' expert bytes, then among an
    // engine's devices (each keeps its rows).
    let group_experts: Vec<u64> = groups.iter().map(|(_, range)| expert_bytes[range.clone()].iter().sum()).collect();
    let loading = std::sync::atomic::AtomicBool::new(true);
    eprintln!("Straydog DeepSeek FFN GEMM: {}; remaining ops F32, up to four rows unchanged GEMV",
              if settings.half_gemm { "FP16 inputs/weights, F32 sums/output" } else { "FP32" });
    let engines = std::thread::scope(|scope| {
        // The dense weights verify as the engines take them: their bytes are the progress.
        scope.spawn(|| {
            while loading.load(std::sync::atomic::Ordering::Relaxed) {
                progress("load", files.verified_bytes(), whole_bytes);
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
                    half_ffn: settings.half_gemm,
                    layers: range.clone(),
                };
                engines.push(Engine::new(hp.clone(), files, devices, budget(devices), engine_settings)?);
            }
            Ok::<_, String>(())
        })();
        loading.store(false, std::sync::atomic::Ordering::Relaxed);
        result.map(|()| engines)
    })?;
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
    });
    // (where a reply's generation prompt begins: its state is checkpointed there)
    let turn_start = files.header.metadata.get("tokenizer.ggml.tokens").and_then(|tokens| tokens.array())
        .and_then(|tokens| tokens.iter().position(|token| token.text() == Some("<｜Assistant｜>")))
        .map(|id| id as u32);
    let drafts = std::env::var_os("DS41_DRAFTS").is_none_or(|value| value != "0");
    let verify_rows = if drafts { engines.iter().map(Engine::verify_rows).min().unwrap_or(1) } else { 1 };
    Ok(Session::new(Placed {
        engines: Engines { engines, devices_per_expert: groups[0].0.len(), verify_rows,
                           saved: (0..CHECKPOINTS).map(|_| None).collect(), order: Vec::new(), budget: CHECKPOINT_BYTES },
        files, vocabulary: hp.vocabulary, layers, experts: hp.experts, context: settings.context, batch: settings.batch,
        end: vec![end], turn_start, card, placement, state_bytes,
        load_seconds: [("open".to_string(), open_seconds), ("place".to_string(), place_seconds)].into_iter().chain(startup).collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::super::engine::snapshot_bytes;
    use super::*;

    /// A checkpoint near a 262144-token context (DeepSeek-V4.1's 40 layers, sources 2, 8, 14 at
    /// ratio 2 and 20 at ratio 1) is over the budget: not taken, and none given up for it. Under
    /// the budget the oldest give theirs up until it fits, no more.
    #[test]
    fn a_checkpoint_over_the_budget_is_not_taken() {
        let position = 262_000;
        let groups = [position / 2, position / 2, position / 2, position];
        let huge = snapshot_bytes(position, 128, 40, 512, 128, &groups, position);
        assert!(huge > CHECKPOINT_BYTES && huge < 2 * CHECKPOINT_BYTES, "{huge}");
        let held = [(0, 300 << 20), (1, 300 << 20), (2, 300 << 20)];
        assert_eq!(evictions(&held, huge, CHECKPOINT_BYTES), None);
        let small = snapshot_bytes(4096, 128, 40, 512, 128, &[2048, 2048, 2048, 4096], 4096);
        assert!(small < 64 << 20, "{small}");
        assert_eq!(evictions(&held, small, CHECKPOINT_BYTES), Some(vec![]));
        assert_eq!(evictions(&held, 400 << 20, CHECKPOINT_BYTES), Some(vec![0]));
        assert_eq!(evictions(&held, CHECKPOINT_BYTES, CHECKPOINT_BYTES), Some(vec![0, 1, 2]));
    }
}
