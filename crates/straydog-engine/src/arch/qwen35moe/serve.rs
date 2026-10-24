//! Qwen3.5 MoE served (docs/DESIGN.md §10): the authenticated GGUF ([`Pinned`]) placed on the
//! GPUs. Two GPUs that can each hold half of every layer with all of its experts (split mode
//! experts) run as tensor parallel: an engine per GPU over every layer, each with its part of the
//! heads, the shared expert, the routed experts and the vocabulary ([`super::part`]), the two
//! adding their partial sums on the devices. Otherwise one engine per GPU over a range of layers
//! balancing the dense bytes; each GPU's VRAM left after its dense weights, the sequence state
//! (DeltaNet states, K/V) and the work buffers given to its layers' experts: all of them for good
//! when they fit (resident: taken by table, a decode step one recorded graph), else expert slots,
//! with the host memory planned for the experts' host tier behind them (a decode step's routes to
//! experts in the host tier then run on the CPU). Requests then run in a
//! [`Session`]; a reply ends at any of the official generation config's end tokens. The DeltaNets'
//! state is checkpointed where each reply's generation prompt begins ([`CHECKPOINTS`] copies), so
//! a next turn resumes there when the template renders the reply otherwise than it ran.

use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use serde_json::json;

use super::engine::{self, forward_parts, forward_split, link, reset_parts, Engine, Settings as EngineSettings};
use super::part::Part;
use super::{Hparams, Mixer};
use crate::arch::served::{self, Placed, Session, Settings, RESERVE, SCRATCH};
use crate::bundle;
use crate::experts::Contents;
use crate::group::SplitMode;
use crate::gpu::{Backend, CpuExperts};
use crate::pinned::Pinned;

/// Qwen3.5 MoE served: its placed engines in a session.
pub type Model = Session<Engines>;

/// Copies of the DeltaNets' state each engine keeps: the last turn's and the one before.
pub const CHECKPOINTS: usize = 2;
/// The CPU expert engine's threads.
const CPU_THREADS: u32 = 6;

/// The engines, one per device: of consecutive layer ranges, or (tensor parallel) the parts of
/// every layer.
pub struct Engines {
    engines: Vec<Engine<'static, Pinned>>,
    parts: bool,
}

impl served::Engines for Engines {
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if self.parts { forward_parts(&mut self.engines, tokens) } else { forward_split(&mut self.engines, tokens) }
    }

    fn reset(&mut self) {
        // A device that cannot be written to fails the next pass, which reports it.
        if self.parts {
            let _ = reset_parts(&mut self.engines);
            return;
        }
        for engine in &mut self.engines {
            let _ = engine.reset();
        }
    }

    fn counts(&self) -> (u64, u64, u64) {
        self.engines.iter().map(Engine::cache_counts)
            .fold((0, 0, 0), |sum, counts| (sum.0 + counts.hits, sum.1 + counts.host_hits, sum.2 + counts.drive_reads))
    }

    fn cpu_routes(&self) -> u64 {
        self.engines.iter().map(Engine::cpu_routes).sum()
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
        1
    }

    fn gpus(&self) -> Vec<&Backend> {
        self.engines.iter().map(Engine::gpu).collect()
    }

    fn settle(&mut self) -> Result<(), String> {
        self.engines.iter_mut().try_for_each(Engine::settle)
    }

    fn checkpoints(&self) -> usize {
        CHECKPOINTS
    }

    fn checkpoint(&mut self, slot: usize, save: bool) -> Result<(), String> {
        self.engines.iter_mut().try_for_each(|engine| engine.checkpoint(slot, save))
    }
}

/// A layer's sequence state: a DeltaNet's convolution history and recurrent state (with its
/// checkpoints' copies), an attention layer's K/V over the context.
fn layer_state_bytes(hp: &Hparams, context: usize, layer: usize) -> u64 {
    let bytes = match hp.mixers[layer] {
        Mixer::DeltaNet => ((hp.conv_kernel - 1) * hp.conv_channels() + hp.v_heads * hp.state_dim * hp.state_dim) * 4 * (1 + CHECKPOINTS),
        Mixer::Attention => context * hp.kv_heads * hp.head_dim * 2 * 4,
    };
    bytes as u64
}

fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// The tokens that end a reply: the official generation config's (`eos_token_id`, one or a list),
/// else the GGUF's end-of-sequence token.
fn end_tokens(root: &Path, pins: &bundle::OnDemand, files: &Pinned) -> Result<Vec<u32>, String> {
    if let Some(asset) = pins.resources.iter().find(|asset| asset.name == "generation_config.json") {
        let config: serde_json::Value = serde_json::from_slice(&crate::pinned::resource(root, asset)?).map_err(|exc| exc.to_string())?;
        let ids: Vec<u32> = match &config["eos_token_id"] {
            serde_json::Value::Array(items) => items.iter().filter_map(|id| id.as_u64()).map(|id| id as u32).collect(),
            other => other.as_u64().map(|id| id as u32).into_iter().collect(),
        };
        if !ids.is_empty() {
            return Ok(ids);
        }
    }
    let eos = files.header.metadata.get("tokenizer.ggml.eos_token_id").and_then(|value| value.integer())
        .and_then(|id| u32::try_from(id).ok()).ok_or("GGUF tokenizer.ggml.eos_token_id is missing")?;
    Ok(vec![eos])
}

/// Open and authenticate the pinned bundle in `root` (trees from `data_dir`) and place it, one
/// engine per device. `progress` gets (stage, done, total).
pub fn place(root: &Path, quantization: &str, data_dir: &Path, settings: &Settings,
             progress: &(dyn Fn(&str, u64, u64) + Sync)) -> Result<Model, String> {
    let started = Instant::now();
    let pins = bundle::on_demand(quantization)?;
    let files = Pinned::open(root, data_dir, &pins)?;
    let open_seconds = started.elapsed().as_secs_f64();
    let hp = Hparams::from_header(&files.header)?;
    let layers = hp.layers();
    // Per layer: the routed experts' bytes and the resident ones; the head's (MTP blocks after the
    // main layers stay on the drive).
    let (mut expert_bytes, mut dense, mut head) = (vec![0u64; layers], vec![0u64; layers], 0u64);
    for (name, bytes) in files.tensor_bytes() {
        match layer_of(name) {
            Some(layer) if layer < layers && name.ends_with("_exps.weight") => expert_bytes[layer] += bytes,
            Some(layer) if layer < layers => dense[layer] += bytes,
            _ if name.starts_with("output") => head += bytes,
            _ => {}
        }
    }
    let cuts: Vec<usize> = (1..layers).collect();
    let load = |range: std::ops::Range<usize>| dense[range.clone()].iter().sum::<u64>() + if range.end == layers { head } else { 0 };
    // Tensor parallel: half of every layer and of its experts on each of two GPUs, when it fits.
    let mut tensor_batch = settings.batch;
    let tensor = settings.split_mode == SplitMode::Experts && settings.devices.len() == 2 && Part::fits(&hp, 2).is_ok() && 'tensor: {
        let first = Part { index: 0, count: 2 };
        // A part's dense bytes: half of each tensor it slices (heads, rows and vocabulary split
        // evenly), the others whole.
        let dense_part: u64 = files.tensor_bytes().filter(|(name, _)| !name.ends_with("_exps.weight")).filter_map(|(name, bytes)| {
            let layer = layer_of(name);
            match layer {
                Some(layer) if layer >= layers => None,
                None if !name.starts_with("output") => None,
                _ => {
                    let role = name.split('.').nth(if layer.is_some() { 2 } else { 0 }).unwrap_or("");
                    Some(if first.slice(&hp, layer, role).is_some() { bytes / 2 } else { bytes })
                }
            }
        }).sum();
        let local = first.hparams(&hp);
        let state_part: u64 = (0..layers).map(|layer| layer_state_bytes(&local, settings.context, layer)).sum();
        // (experts whose ffn rows do not split into whole blocks of their format stay split by layers)
        let Ok(experts_part) = Engine::<Pinned>::resident_bytes(&hp, &files, 0..layers, first) else { break 'tensor false };
        // A pass's rows as below, each a part's work buffers and its linked sums' two inbox slots.
        let row_bytes = engine::batch_row_bytes(&local) + 2 * hp.width as u64 * 4;
        // (the parts hold equal shares: the smaller budget bounds both)
        let vram = settings.vram_bytes.iter().copied().min().unwrap_or(0);
        let fixed = dense_part + state_part + experts_part + SCRATCH + RESERVE;
        let room = vram.saturating_sub(fixed);
        let fits = ((vram / 8).min(room / 4) / row_bytes) as usize / 64 * 64;
        tensor_batch = settings.batch.min(fits.max(64));
        fixed + tensor_batch as u64 * row_bytes <= vram
    };
    let parts = if tensor {
        vec![0..layers; 2]
    } else {
        served::split_layers(layers, &cuts, &load, &settings.vram_bytes)
            .ok_or_else(|| format!("the model cannot be split across {} GPUs", settings.devices.len()))?
    };
    // The largest of a part's layers' experts: the store takes VRAM as so many of these.
    let slot_bytes = |range: &std::ops::Range<usize>| {
        range.clone().map(|layer| expert_bytes[layer] / hp.experts as u64).max().unwrap_or(1).max(1).next_multiple_of(256)
    };
    // The sequence state of a part's layers.
    let state = |range: &std::ops::Range<usize>| -> u64 { range.clone().map(|layer| layer_state_bytes(&hp, settings.context, layer)).sum() };
    // A prefill pass reads each routed expert its rows need once: the longer the pass, the fewer
    // reads (with 64 rows, a 4K-token prompt read nearly every expert 64 times). Its rows' work
    // buffers take at most an eighth of each GPU's budget and a quarter of what it has left after
    // its fixed bytes; the expert slots keep the rest.
    let row_bytes = engine::batch_row_bytes(&hp);
    let base = |range: &std::ops::Range<usize>| load(range.clone()) + state(range) + SCRATCH + RESERVE;
    let fits = parts.iter().zip(&settings.vram_bytes).map(|(range, &vram)| {
        ((vram / 8).min(vram.saturating_sub(base(range)) / 4) / row_bytes) as usize / 64 * 64
    }).min().unwrap_or(0);
    let batch = if tensor { tensor_batch } else { settings.batch.min(fits.max(64)) };
    let total_experts: u64 = expert_bytes.iter().sum();
    let (mut slots, mut resident) = (Vec::new(), Vec::new());
    for (range, (&vram, device)) in parts.iter().zip(settings.vram_bytes.iter().zip(&settings.devices)).filter(|_| !tensor) {
        let fixed = base(range) + batch as u64 * row_bytes;
        let free = vram.checked_sub(fixed).ok_or_else(|| {
            format!("layers {range:?} need more than {:.1} GiB of VRAM on GPU {device}", vram as f64 / (1u64 << 30) as f64)
        })?;
        // A GPU with room for every expert of its layers keeps them all.
        resident.push(Engine::<Pinned>::resident_bytes(&hp, &files, range.clone(), Part::WHOLE)? <= free);
        slots.push(((free / slot_bytes(range)) as usize).min(range.len() * hp.experts));
    }
    if tensor {
        slots = vec![0; 2];
        resident = vec![true; 2];
    }
    // (tensor parallel: each part holds half of every layer's experts)
    let part_experts: Vec<u64> = parts.iter().map(|range| {
        let bytes: u64 = expert_bytes[range.clone()].iter().sum();
        if tensor { bytes / 2 } else { bytes }
    }).collect();
    let in_vram: u64 = parts.iter().enumerate()
        .map(|(part, range)| if resident[part] { part_experts[part] } else { slots[part] as u64 * slot_bytes(range) }).sum();
    let host = served::host_tier(settings, total_experts.saturating_sub(in_vram));
    // The page-locked host tier is shared out among the parts not all resident by the expert bytes
    // each one's VRAM leaves: a part given less reads some of its experts from the drive again
    // whenever a pass (a long prompt's) goes through them all, while another has room to spare.
    let beyond: Vec<u64> = parts.iter().enumerate().map(|(part, range)| {
        if resident[part] { 0 } else { part_experts[part].saturating_sub(slots[part] as u64 * slot_bytes(range)) }
    }).collect();
    let tiered: u64 = beyond.iter().sum();
    let end = end_tokens(root, &pins, &files)?;
    let turn_start = files.header.metadata.get("tokenizer.ggml.tokens").and_then(|tokens| tokens.array())
        .and_then(|tokens| tokens.iter().position(|token| token.text() == Some("<|im_start|>")))
        .map(|id| id as u32);
    let card = served::card(&files.header)?;
    let whole_bytes = files.whole_bytes(|name| layer_of(name).is_none_or(|layer| layer < layers));
    let files: &'static Pinned = Box::leak(Box::new(files));
    let placed = Instant::now();
    let state_bytes = if tensor {
        let local = Part { index: 0, count: 2 }.hparams(&hp);
        2 * (0..layers).map(|layer| layer_state_bytes(&local, settings.context, layer)).sum::<u64>()
    } else {
        parts.iter().map(state).sum()
    };
    let loading = std::sync::atomic::AtomicBool::new(true);
    let mut engines = std::thread::scope(|scope| {
        // The dense weights verify as the engines take them: their bytes are the progress.
        scope.spawn(|| {
            while loading.load(std::sync::atomic::Ordering::Relaxed) {
                progress("load", files.verified_bytes(), whole_bytes);
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
        });
        let mut engines = Vec::new();
        let result = (|| {
            for (part, (&device, range)) in settings.devices.iter().zip(&parts).enumerate() {
                let engine_settings = EngineSettings {
                    context: settings.context,
                    batch,
                    expert_slots: slots[part],
                    host_expert_bytes: if resident[part] { 0 } else { (host as f64 * beyond[part] as f64 / tiered.max(1) as f64) as u64 },
                    scratch_bytes: SCRATCH,
                    layers: range.clone(),
                    checkpoints: CHECKPOINTS,
                    resident: resident[part],
                    part: if tensor { Part { index: part, count: 2 } } else { Part::WHOLE },
                    half_gemm: settings.half_gemm,
                };
                engines.push(Engine::new(hp.clone(), files, device, settings.vram_bytes[part], engine_settings)?);
            }
            if tensor {
                link(&mut engines, batch)?;
            }
            Ok::<_, String>(())
        })();
        loading.store(false, std::sync::atomic::Ordering::Relaxed);
        result.map(|()| engines)
    })?;
    // A decode step's routes to experts the host tier holds run on the CPU, shared by the GPUs
    // whose experts are not all resident.
    if resident.iter().any(|&resident| !resident) {
        let cpu = Rc::new(CpuExperts::new(CPU_THREADS)?);
        for (engine, _) in engines.iter_mut().zip(&resident).filter(|(_, &resident)| !resident) {
            engine.set_cpu(cpu.clone());
        }
    }
    let place_seconds = placed.elapsed().as_secs_f64();
    // Tensor parallel reads as one range of layers over both GPUs.
    let (layer_ranges, devices, expert_slots): (Vec<[usize; 2]>, Vec<Vec<usize>>, Vec<usize>) = if tensor {
        (vec![[0, layers]], vec![settings.devices.clone()], vec![engines.iter().map(|engine| engine.expert_capacity().0).sum()])
    } else {
        (parts.iter().map(|range| [range.start, range.end]).collect(), settings.devices.iter().take(parts.len()).map(|&device| vec![device]).collect(),
         engines.iter().map(|engine| engine.expert_capacity().0).collect())
    };
    let placement = json!({
        "split_mode": if tensor { "tensor" } else { "layer" },
        "layers": layer_ranges,
        "devices": devices,
        "expert_slots": expert_slots,
        "host_expert_slots": engines.iter().map(|engine| engine.expert_capacity().1).collect::<Vec<_>>(),
        "expert_vram_bytes": parts.iter().enumerate()
            .map(|(part, range)| if resident[part] { part_experts[part] } else { slots[part] as u64 * slot_bytes(range) }).collect::<Vec<_>>(),
        "host_expert_bytes": host,
        "expert_bytes": total_experts,
        "resident_experts": resident,
        "cpu_experts": resident.iter().any(|&resident| !resident),
        "prefill_batch_size": batch,
    });
    let mut engines = Engines { engines, parts: tensor };
    // The first FP16 GEMMs load the BLAS library's kernels (~0.7 s): a pass of a few rows takes
    // that at placement rather than in the first request; then it is forgotten.
    if settings.half_gemm && batch >= 32 {
        served::Engines::forward(&mut engines, &[0; 32])?;
        served::Engines::reset(&mut engines);
    }
    Ok(Session::new(Placed {
        engines,
        files, vocabulary: hp.vocabulary, layers, experts: hp.experts, context: settings.context, batch, end, turn_start,
        card,
        placement, state_bytes,
        load_seconds: vec![("open".to_string(), open_seconds), ("place".to_string(), place_seconds)],
    }))
}
