//! DeepSeek-V4.1 served (docs/DESIGN.md §10): the authenticated shards ([`Pinned`]), placed on the
//! GPUs either with every layer split across them (one engine, experts divided by rows) or with
//! the layers split where no layer reads another part's compressed rows (one engine per GPU);
//! each GPU's VRAM left after its dense weights and state given to expert slots and the host
//! memory planned for the experts' host tier, then prompts continued from the sequence the engines
//! hold, sampling and the request's metrics. The engines are driven only through their public
//! passes.

use std::path::Path;
use std::time::Instant;

use serde_json::{json, Map, Value};

use super::engine::{divided, forward_split, Engine, Settings as EngineSettings};
use super::Hparams;
use crate::bundle;
use crate::group::SplitMode;
use crate::model::Rng;
use crate::pinned::Pinned;
use crate::sampling::{self, Generated, Sampling};

/// Per GPU: VRAM kept for the runtime's own buffers, the passes' work buffers and their scratch
/// (measured with a 64-row batch: 1.3-1.5 GiB), beside the dense weights, the sequence state and
/// the expert slots.
const RESERVE: u64 = 2 << 30;
const SCRATCH: u64 = 512 << 20;
/// Without a host tier budget, the experts get this share of the host memory, less some for the
/// rest of the system (the page cache holds the gathered table's rows).
const HOST_SHARE: f64 = 0.5;
const HOST_KEEP: u64 = 4 << 30;

pub struct Settings {
    pub devices: Vec<usize>,
    /// With several GPUs: Experts splits every layer across them (experts by rows), Layer gives
    /// each a range of layers.
    pub split_mode: SplitMode,
    /// VRAM budget per GPU.
    pub vram_bytes: u64,
    pub context: usize,
    /// Most tokens one pass takes (a prompt is prefilled in chunks of this many).
    pub batch: usize,
    /// The experts' host tier; None plans it from `host_memory_bytes`.
    pub host_expert_bytes: Option<u64>,
    pub host_memory_bytes: u64,
    /// Host memory caching the gathered tables' rows (the Engram table), read from the drive as
    /// needed and verified as read; 0 reads them through the mapping.
    pub row_cache_bytes: u64,
}

pub struct Model {
    files: &'static Pinned,
    engines: Vec<Engine<'static, Pinned>>,
    hp: Hparams,
    pub context: usize,
    pub batch: usize,
    /// The tokens the engines hold, in order.
    history: Vec<u32>,
    end: u32,
    /// The sampling the model card gives (GGUF `general.sampling.*`), for every mode.
    pub card: Sampling,
    /// How the model was placed (the ready frame shows it).
    pub placement: Value,
    pub state_bytes: u64,
    pub load_seconds: Vec<(&'static str, f64)>,
    pub last_metrics: Map<String, Value>,
}

/// Layer ranges, one per device, splitting where no layer reads an earlier part's compressed rows
/// and balancing the dense bytes (the head with the last part).
fn split(hp: &Hparams, dense: &[u64], head: u64, parts: usize) -> Result<Vec<std::ops::Range<usize>>, String> {
    let layers = hp.layers.len();
    let valid = |at: usize| {
        hp.layers[at..].iter().all(|plan| plan.compressed.is_none_or(|c| c.kv_source >= at && c.index_source >= at))
    };
    let cuts: Vec<usize> = (1..layers).filter(|&at| valid(at)).collect();
    if parts == 1 {
        return Ok(std::iter::once(0..layers).collect());
    }
    if cuts.len() + 1 < parts {
        return Err(format!("DeepSeek-V4.1 cannot be split across {parts} GPUs"));
    }
    let load = |range: std::ops::Range<usize>| dense[range.clone()].iter().sum::<u64>() + if range.end == layers { head } else { 0 };
    // The fewest bytes on the fullest part, over every choice of cuts (few layers: exhaustive DP).
    // best[part][end]: the fullest part's bytes and the part ends, for `part` parts ending at `end`.
    type Choice = Option<(u64, Vec<usize>)>;
    let mut best: Vec<Vec<Choice>> = vec![vec![None; layers + 1]; parts + 1];
    best[0][0] = Some((0, Vec::new()));
    for part in 1..=parts {
        let ends: Vec<usize> = if part == parts { vec![layers] } else { cuts.clone() };
        for &end in &ends {
            let starts = std::iter::once(0).chain(cuts.iter().copied()).filter(|&start| start < end);
            let mut choice: Option<(u64, Vec<usize>)> = None;
            for start in starts {
                if let Some((worst, path)) = &best[part - 1][start] {
                    let candidate = (*worst).max(load(start..end));
                    if choice.as_ref().is_none_or(|(value, _)| candidate < *value) {
                        let mut path = path.clone();
                        path.push(end);
                        choice = Some((candidate, path));
                    }
                }
            }
            best[part][end] = choice;
        }
    }
    let (_, ends) = best[parts][layers].clone().ok_or("no DeepSeek-V4.1 layer split")?;
    let mut start = 0;
    Ok(ends.into_iter().map(|end| { let range = start..end; start = end; range }).collect())
}

fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

impl Model {
    /// Open and authenticate the pinned bundle in `root` (trees from `data_dir`) and place it.
    /// `progress` gets (stage, done, total).
    pub fn new(root: &Path, quantization: &str, data_dir: &Path, settings: &Settings,
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
                let ranges = split(&hp, &dense, head_whole + head_parted, settings.devices.len())?;
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
            let ring = (hp.window + settings.batch) * hp.head_dim * 4 * range.len();
            let sources: usize = hp.layers.iter().enumerate().filter(|(layer, _)| range.contains(layer))
                .filter_map(|(layer, plan)| plan.compressed.filter(|c| c.kv_source == layer))
                .map(|c| (settings.context / c.ratio + 1) * (hp.head_dim + hp.index_dim) * 4).sum();
            (ring + sources) as u64
        };
        let total_experts: u64 = expert_bytes.iter().sum();
        // Per engine: the experts its devices hold in VRAM together (each device its rows of every
        // one), from what each device has left after its resident share, state and work buffers.
        let mut slots = Vec::new();
        for group in &groups {
            let (devices, range) = group;
            let parts = devices.len() as u64;
            let head = if range.end == layers { head_whole + head_parted.div_ceil(parts) } else { 0 };
            let resident: u64 = range.clone().map(|layer| whole[layer] + parted[layer].div_ceil(parts)).sum::<u64>() + head;
            let fixed = resident + state(range) + SCRATCH + RESERVE;
            let free = settings.vram_bytes.checked_sub(fixed)
                .ok_or_else(|| format!("DeepSeek-V4.1 layers {range:?} need more than {:.1} GiB of VRAM per GPU", settings.vram_bytes as f64 / (1u64 << 30) as f64))?;
            // Each device's share of an expert is 1/parts of a slot: free / share experts.
            let held = (free / (slot_bytes(group) / parts)) as usize;
            slots.push(held.min(range.len() * hp.experts) / devices.len() * devices.len());
        }
        let in_vram: u64 = groups.iter().zip(&slots).map(|(group, &slots)| slots as u64 * slot_bytes(group)).sum();
        let host = settings.host_expert_bytes.unwrap_or_else(|| {
            ((settings.host_memory_bytes as f64 * HOST_SHARE) as u64).saturating_sub(HOST_KEEP).min(total_experts.saturating_sub(in_vram))
        });
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
                        batch: settings.batch,
                        expert_slots: slots / devices.len(),
                        host_expert_bytes: (share / devices.len() as f64) as u64,
                        scratch_bytes: SCRATCH,
                        layers: range.clone(),
                    };
                    engines.push(Engine::new(hp.clone(), files, devices, settings.vram_bytes, engine_settings)?);
                }
                Ok::<_, String>(())
            })();
            loading.store(false, std::sync::atomic::Ordering::Relaxed);
            result.map(|()| engines)
        })?;
        let place_seconds = placed.elapsed().as_secs_f64();
        let number = |key: &str| files.header.metadata.get(key).and_then(|value| value.float());
        let card = Sampling {
            temperature: number("general.sampling.temp").unwrap_or(1.0),
            top_k: number("general.sampling.top_k").map_or(0, |value| value.max(0.0) as usize),
            top_p: number("general.sampling.top_p").unwrap_or(1.0),
            min_p: number("general.sampling.min_p").unwrap_or(0.0),
            presence_penalty: 0.0,
        };
        card.check()?;
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
        Ok(Model {
            files, engines, hp, context: settings.context, batch: settings.batch, history: Vec::new(), end, card, placement,
            state_bytes, load_seconds: vec![("open", open_seconds), ("place", place_seconds)], last_metrics: Map::new(),
        })
    }

    pub fn vocabulary(&self) -> usize {
        self.hp.vocabulary
    }

    /// The model's metadata (tokenizer, template), authenticated.
    pub fn header(&self) -> &straydog_gguf::ModelHeader {
        &self.files.header
    }

    /// Forget the sequence (warm experts stay).
    pub fn reset(&mut self) {
        self.history.clear();
        self.engines.iter_mut().for_each(Engine::reset);
    }

    /// The GPUs as the server reports them: summed, then each.
    pub fn info(&self) -> Result<Value, String> {
        let devices = self.engines.iter().flat_map(Engine::gpus).map(|gpu| gpu.info()).collect::<Result<Vec<_>, _>>()?;
        let names: Vec<&str> = devices.iter().map(|device| device["name"].as_str().unwrap_or("")).collect();
        let mut info = json!({"name": names.join(" + "), "device": devices[0]["device"]});
        for key in ["free_bytes", "total_bytes", "allocated_bytes", "budget_bytes"] {
            info[key] = devices.iter().map(|device| device[key].as_u64().unwrap_or(0)).sum::<u64>().into();
        }
        let counts = self.engines.iter().map(Engine::cache_counts)
            .fold((0, 0, 0), |sum, counts| (sum.0 + counts.hits, sum.1 + counts.host_hits, sum.2 + counts.drive_reads));
        info["experts"] = json!({"vram_hits": counts.0, "host_hits": counts.1, "drive_reads": counts.2});
        info["devices"] = devices.into();
        Ok(info)
    }

    /// Feed `tokens` (any count) through the engines in passes of at most the batch; the logits
    /// after the last. Cancelled between passes: None, with the history at what was fed.
    fn feed(&mut self, tokens: &[u32], cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize)) -> Result<Option<Vec<f32>>, String> {
        let mut logits = None;
        for (index, piece) in tokens.chunks(self.batch).enumerate() {
            if index > 0 && cancelled() {
                return Ok(None);
            }
            match forward_split(&mut self.engines, piece) {
                Ok(output) => logits = Some(output),
                Err(error) => {
                    self.reset();
                    return Err(error);
                }
            }
            self.history.extend_from_slice(piece);
            progress(self.history.len());
        }
        Ok(logits)
    }

    /// Continue the held sequence over `tokens` (or start anew when they do not extend it) and
    /// sample up to `max_new` tokens, each passed to `emit`; the end token ends it unsent.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(&mut self, tokens: &[u32], max_new: usize, sampling: &Sampling, seed: u64, cancelled: &dyn Fn() -> bool,
                    progress: &mut dyn FnMut(usize, usize), emit: &mut dyn FnMut(u32)) -> Result<(), String> {
        let started = Instant::now();
        self.last_metrics = Map::new();
        if tokens.is_empty() || max_new == 0 || tokens.len() + max_new > self.context {
            return Err("Straydog prompt plus output exceeds the allocated context".into());
        }
        if tokens.iter().any(|&token| token as usize >= self.hp.vocabulary) {
            return Err("Straydog token is out of range".into());
        }
        sampling.check()?;
        self.files.check_unchanged()?;
        // A prompt that extends the held sequence continues it; anything else starts over.
        let reused = if self.history.len() < tokens.len() && tokens.starts_with(&self.history) { self.history.len() } else { 0 };
        if reused == 0 {
            self.reset();
        }
        let counts = |engines: &[Engine<'_, Pinned>]| engines.iter().map(Engine::cache_counts)
            .fold((0u64, 0u64, 0u64), |sum, counts| (sum.0 + counts.hits, sum.1 + counts.host_hits, sum.2 + counts.drive_reads));
        let total = tokens.len();
        let Some(mut logits) = self.feed(&tokens[reused..], cancelled, &mut |done| progress(done, total))? else {
            return Ok(());
        };
        let prefill_seconds = started.elapsed().as_secs_f64();
        let (hits, host, drive) = counts(&self.engines);
        let mut rng = Rng::new(seed);
        let mut generated = Generated::new(self.hp.vocabulary);
        let (mut emitted, mut ttft) = (0usize, None);
        let decoding = Instant::now();
        loop {
            if cancelled() {
                break;
            }
            let token = sampling::choose(&logits, sampling, &generated, &mut rng);
            if token == self.end {
                break;
            }
            generated.insert(token);
            ttft.get_or_insert_with(|| started.elapsed().as_secs_f64());
            emit(token);
            emitted += 1;
            if emitted == max_new {
                break;
            }
            match self.feed(&[token], &|| false, &mut |_| {})? {
                Some(next) => logits = next,
                None => break,
            }
        }
        let decode_seconds = decoding.elapsed().as_secs_f64();
        let steps = emitted.saturating_sub(1);
        let (hits2, host2, drive2) = counts(&self.engines);
        let per = |value: u64| if steps > 0 { json!(value as f64 / steps as f64) } else { Value::Null };
        self.last_metrics = json!({
            "cached_tokens": reused, "resumed_from": if reused > 0 { "live" } else { "none" },
            "prefill_tokens": total - reused, "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": (total - reused) as f64 / prefill_seconds.max(1e-9),
            "ttft_seconds": ttft, "generated_tokens": emitted, "decode_forward_tokens": steps, "decode_seconds": decode_seconds,
            "decode_tokens_per_second": if steps > 0 { json!(steps as f64 / decode_seconds) } else { Value::Null },
            "experts_per_token": {"vram": per(hits2 - hits), "host": per(host2 - host), "drive": per(drive2 - drive)},
        })
        .as_object()
        .unwrap()
        .clone();
        Ok(())
    }

    /// Wait for the expert store's reads and copies (before the process ends).
    pub fn settle(&mut self) -> Result<(), String> {
        self.engines.iter_mut().try_for_each(Engine::settle)
    }
}
