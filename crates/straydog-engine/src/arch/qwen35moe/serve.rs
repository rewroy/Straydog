//! Qwen3.5 MoE served (docs/DESIGN.md §10): the authenticated GGUF ([`Pinned`]) placed on the
//! GPUs, one engine per GPU over a range of layers balancing the dense bytes; each GPU's VRAM left
//! after its dense weights, the sequence state (DeltaNet states, K/V) and the work buffers given to
//! its layers' experts: all of them for good when they fit (resident: taken by table, a decode step
//! one recorded graph), else expert slots, with the host memory planned for the experts' host tier
//! behind them. Requests then run in a
//! [`Session`]; a reply ends at any of the official generation config's end tokens. The DeltaNets'
//! state is checkpointed where each reply's generation prompt begins ([`CHECKPOINTS`] copies), so
//! a next turn resumes there when the template renders the reply otherwise than it ran.

use std::path::Path;
use std::time::Instant;

use serde_json::json;

use super::engine::{forward_split, Engine, Settings as EngineSettings};
use super::{Hparams, Mixer};
use crate::arch::served::{self, Placed, Session, Settings, RESERVE, SCRATCH};
use crate::bundle;
use crate::experts::Contents;
use crate::gpu::Backend;
use crate::pinned::Pinned;

/// Qwen3.5 MoE served: its placed engines in a session.
pub type Model = Session<Engines>;

/// Copies of the DeltaNets' state each engine keeps: the last turn's and the one before.
pub const CHECKPOINTS: usize = 2;

/// The engines of consecutive layer ranges, one per device.
pub struct Engines {
    engines: Vec<Engine<'static, Pinned>>,
}

impl served::Engines for Engines {
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        forward_split(&mut self.engines, tokens)
    }

    fn reset(&mut self) {
        for engine in &mut self.engines {
            // A device that cannot be written to fails the next pass, which reports it.
            let _ = engine.reset();
        }
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
    let parts = served::split_layers(layers, &cuts, &load, settings.devices.len())
        .ok_or_else(|| format!("the model cannot be split across {} GPUs", settings.devices.len()))?;
    // The largest of a part's layers' experts: the store takes VRAM as so many of these.
    let slot_bytes = |range: &std::ops::Range<usize>| {
        range.clone().map(|layer| expert_bytes[layer] / hp.experts as u64).max().unwrap_or(1).max(1).next_multiple_of(256)
    };
    // The sequence state of a part's layers: each DeltaNet's convolution history and recurrent
    // state (with its checkpoints' copies), each attention layer's K/V over the context.
    let state = |range: &std::ops::Range<usize>| -> u64 {
        range.clone().map(|layer| {
            let bytes = match hp.mixers[layer] {
                Mixer::DeltaNet => {
                    ((hp.conv_kernel - 1) * hp.conv_channels() + hp.v_heads * hp.state_dim * hp.state_dim) * 4 * (1 + CHECKPOINTS)
                }
                Mixer::Attention => settings.context * hp.kv_heads * hp.head_dim * 2 * 4,
            };
            bytes as u64
        }).sum()
    };
    let total_experts: u64 = expert_bytes.iter().sum();
    let (mut slots, mut resident) = (Vec::new(), Vec::new());
    for range in &parts {
        let fixed = load(range.clone()) + state(range) + SCRATCH + RESERVE;
        let free = settings.vram_bytes.checked_sub(fixed).ok_or_else(|| {
            format!("layers {range:?} need more than {:.1} GiB of VRAM per GPU", settings.vram_bytes as f64 / (1u64 << 30) as f64)
        })?;
        // A GPU with room for every expert of its layers keeps them all.
        resident.push(Engine::<Pinned>::resident_bytes(&hp, &files, range.clone())? <= free);
        slots.push(((free / slot_bytes(range)) as usize).min(range.len() * hp.experts));
    }
    let part_experts: Vec<u64> = parts.iter().map(|range| expert_bytes[range.clone()].iter().sum()).collect();
    let in_vram: u64 = parts.iter().enumerate()
        .map(|(part, range)| if resident[part] { part_experts[part] } else { slots[part] as u64 * slot_bytes(range) }).sum();
    let host = served::host_tier(settings, total_experts.saturating_sub(in_vram));
    // The page-locked host tier is shared out by the other parts' expert bytes.
    let tiered: u64 = part_experts.iter().zip(&resident).filter(|(_, &resident)| !resident).map(|(bytes, _)| bytes).sum();
    let end = end_tokens(root, &pins, &files)?;
    let turn_start = files.header.metadata.get("tokenizer.ggml.tokens").and_then(|tokens| tokens.array())
        .and_then(|tokens| tokens.iter().position(|token| token.text() == Some("<|im_start|>")))
        .map(|id| id as u32);
    let card = served::card(&files.header)?;
    let whole_bytes = files.whole_bytes(|name| layer_of(name).is_none_or(|layer| layer < layers));
    let files: &'static Pinned = Box::leak(Box::new(files));
    let placed = Instant::now();
    let state_bytes = parts.iter().map(state).sum();
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
            for (part, (&device, range)) in settings.devices.iter().zip(&parts).enumerate() {
                let engine_settings = EngineSettings {
                    context: settings.context,
                    batch: settings.batch,
                    expert_slots: slots[part],
                    host_expert_bytes: if resident[part] { 0 } else { (host as f64 * part_experts[part] as f64 / tiered.max(1) as f64) as u64 },
                    scratch_bytes: SCRATCH,
                    layers: range.clone(),
                    checkpoints: CHECKPOINTS,
                    resident: resident[part],
                };
                engines.push(Engine::new(hp.clone(), files, device, settings.vram_bytes, engine_settings)?);
            }
            Ok::<_, String>(())
        })();
        loading.store(false, std::sync::atomic::Ordering::Relaxed);
        result.map(|()| engines)
    })?;
    let place_seconds = placed.elapsed().as_secs_f64();
    let placement = json!({
        "split_mode": "layer",
        "layers": parts.iter().map(|range| [range.start, range.end]).collect::<Vec<_>>(),
        "devices": settings.devices.iter().take(parts.len()).map(|&device| vec![device]).collect::<Vec<_>>(),
        "expert_slots": engines.iter().map(|engine| engine.expert_capacity().0).collect::<Vec<_>>(),
        "host_expert_slots": engines.iter().map(|engine| engine.expert_capacity().1).collect::<Vec<_>>(),
        "expert_vram_bytes": parts.iter().enumerate()
            .map(|(part, range)| if resident[part] { part_experts[part] } else { slots[part] as u64 * slot_bytes(range) }).collect::<Vec<_>>(),
        "host_expert_bytes": host,
        "expert_bytes": total_experts,
        "resident_experts": resident,
        "prefill_batch_size": settings.batch,
    });
    Ok(Session::new(Placed {
        engines: Engines { engines },
        files, vocabulary: hp.vocabulary, layers, experts: hp.experts, context: settings.context, batch: settings.batch, end, turn_start,
        card,
        placement, state_bytes,
        load_seconds: vec![("open", open_seconds), ("place", place_seconds)],
    }))
}
