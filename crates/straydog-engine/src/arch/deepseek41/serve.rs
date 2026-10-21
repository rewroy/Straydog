//! DeepSeek-V4.1 served (docs/DESIGN.md §10): the authenticated shards ([`Pinned`]), placed on the
//! GPUs either with every layer split across them (one engine, experts divided by rows) or with
//! the layers split where no layer reads another part's compressed rows (one engine per GPU);
//! each GPU's VRAM left after its dense weights and state given to expert slots and the host
//! memory planned for the experts' host tier, then prompts continued from the sequence the engines
//! hold, sampling and the request's metrics. The engines are driven only through their public
//! passes.

use std::path::Path;
use std::time::Instant;

use serde_json::json;

use super::engine::{divided, forward_split, pass_row_bytes, Engine, Settings as EngineSettings, ATTENTION_ROWS};
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

/// The engines of consecutive layer ranges (one, or one per device).
pub struct Engines {
    engines: Vec<Engine<'static, Pinned>>,
    /// The devices of the first engine (each reads its rows of an expert).
    devices_per_expert: usize,
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
    let hp = Hparams::from_header(&files.header)?;
    let layers = hp.layers.len();
    // Per layer: the routed experts' bytes, and the resident bytes kept whole on every device
    // and divided among an engine's devices; the same for the head.
    let (mut expert_bytes, mut whole, mut parted) = (vec![0u64; layers], vec![0u64; layers], vec![0u64; layers]);
    let (mut head_whole, mut head_parted) = (0u64, 0u64);
    for (name, bytes) in files.tensor_bytes() {
        match layer_of(name) {
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
        // a pass of up to the batch's rows, layer by layer: its buffers of every row
        let rows = settings.batch as u64 * pass_row_bytes(&hp, devices.len(), range.clone());
        let fixed = resident + state(range) + rows + SCRATCH + RESERVE;
        let free = vram.checked_sub(fixed)
            .ok_or_else(|| format!("DeepSeek-V4.1 layers {range:?} need more than {:.1} GiB of VRAM on GPU {devices:?}", vram as f64 / (1u64 << 30) as f64))?;
        // Each device's share of an expert is 1/parts of a slot: free / share experts.
        let held = (free / (slot_bytes(group) / parts)) as usize;
        slots.push(held.min(range.len() * hp.experts) / devices.len() * devices.len());
    }
    let in_vram: u64 = groups.iter().zip(&slots).map(|(group, &slots)| slots as u64 * slot_bytes(group)).sum();
    let host = served::host_tier(settings, total_experts.saturating_sub(in_vram));
    let whole_bytes = files.whole_bytes(|_| true);
    let files: &'static Pinned = Box::leak(Box::new(files));
    let placed = Instant::now();
    let state_bytes = groups.iter().map(|(devices, range)| state(range) * devices.len() as u64).sum();
    // The page-locked host tier is shared out by the engines' expert bytes, then among an
    // engine's devices (each keeps its rows).
    let group_experts: Vec<u64> = groups.iter().map(|(_, range)| expert_bytes[range.clone()].iter().sum()).collect();
    let loading = std::sync::atomic::AtomicBool::new(true);
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
                let share = host as f64 * experts as f64 / total_experts.max(1) as f64;
                let engine_settings = EngineSettings {
                    context: settings.context,
                    rows: settings.batch,
                    batch: settings.batch.min(ATTENTION_ROWS),
                    expert_slots: slots / devices.len(),
                    host_expert_bytes: (share / devices.len() as f64) as u64,
                    scratch_bytes: SCRATCH,
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
    Ok(Session::new(Placed {
        engines: Engines { engines, devices_per_expert: groups[0].0.len() },
        files, vocabulary: hp.vocabulary, layers, experts: hp.experts, context: settings.context, batch: settings.batch,
        end: vec![end], turn_start: None, card, placement, state_bytes,
        load_seconds: vec![("open", open_seconds), ("place", place_seconds)],
    }))
}
