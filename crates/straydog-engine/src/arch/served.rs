//! What a served model of any architecture does with its placed engines (docs/DESIGN.md §10.1,
//! layer 3): continue the sequence they hold when a prompt extends it (or, with engines that keep
//! checkpoints, from where an earlier prompt's reply began), sample, report a request's metrics,
//! and warm the expert stores up with an earlier run's contents. An architecture places its
//! engines (its `serve` module) and drives them through [`Engines`].

use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::experts::Contents;
use crate::gpu::Backend;
use crate::group::SplitMode;
use crate::model::Rng;
use crate::pinned::Pinned;
use crate::sampling::{self, Generated, Sampling};

/// An architecture's placed engines, as a session drives them.
pub trait Engines {
    /// One pass over `tokens` (at most a batch) continuing the held sequence: the logits after
    /// the last.
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String>;
    /// Forget the sequence (warm experts stay).
    fn reset(&mut self);
    /// The expert stores' (VRAM hits, host-tier hits, drive reads) so far.
    fn counts(&self) -> (u64, u64, u64);
    /// The experts the stores hold, least recently used first: in VRAM, and only in the host tier.
    fn contents(&self) -> Contents;
    /// Start warming the stores up with these experts; the reads in flight.
    fn warm(&mut self, vram: &[(usize, usize)], host: &[(usize, usize)]) -> Result<usize, String>;
    /// Take the landed reads, without waiting; the reads still in flight.
    fn poll(&mut self) -> Result<usize, String>;
    /// The devices that each read part of an expert.
    fn devices_per_expert(&self) -> usize;
    fn gpus(&self) -> Vec<&Backend>;
    /// Wait for the stores' reads and copies (before the process ends).
    fn settle(&mut self) -> Result<(), String>;
    /// Copies of the sequence state the engines keep ([`Engines::checkpoint`]); none: a session
    /// continues only a sequence the prompt extends.
    fn checkpoints(&self) -> usize {
        0
    }
    /// Save the sequence state (its position too) to checkpoint `slot`, or restore it from there.
    /// A restore keeps the K/V below the checkpoint's position, which must be its sequence's.
    fn checkpoint(&mut self, _slot: usize, _save: bool) -> Result<(), String> {
        Err("these engines keep no checkpoints".into())
    }
}

/// Per GPU: VRAM kept for the runtime's own buffers, the passes' work buffers and their scratch
/// (measured with a 64-row batch: 1.3-1.5 GiB), beside the dense weights, the sequence state and
/// the expert slots.
pub const RESERVE: u64 = 2 << 30;
pub const SCRATCH: u64 = 512 << 20;
/// Without a host tier budget, Straydog takes this share of the host memory in all, and the
/// experts' host tier what its other host memory leaves of it: the row cache, and the rest below.
pub const HOST_SHARE: f64 = 0.75;
/// The worker's host memory beside the tier and the row cache: the GPU runtime, the verity
/// trees' level-0 caches, read buffers and the passes' host copies.
pub const HOST_OTHER: u64 = 2 << 30;

/// The experts' host tier: as given, else what Straydog's other host memory (the row cache and
/// the rest) leaves of its share of the host memory; never more than the experts VRAM lacks.
pub fn host_tier(settings: &Settings, beyond_vram: u64) -> u64 {
    settings.host_expert_bytes.unwrap_or_else(|| {
        ((settings.host_memory_bytes as f64 * HOST_SHARE) as u64).saturating_sub(settings.row_cache_bytes + HOST_OTHER).min(beyond_vram)
    })
}

/// `parts` consecutive layer ranges covering `layers`, cut only at `cuts`, with the fewest bytes
/// on the fullest part (`load` of a range); None when there are too few cuts.
pub fn split_layers(layers: usize, cuts: &[usize], load: &dyn Fn(std::ops::Range<usize>) -> u64, parts: usize)
                    -> Option<Vec<std::ops::Range<usize>>> {
    if parts <= 1 {
        return Some(std::iter::once(0..layers).collect());
    }
    if cuts.len() + 1 < parts {
        return None;
    }
    // Over every choice of cuts (few layers: exhaustive DP). best[part][end]: the fullest part's
    // bytes and the part ends, for `part` parts ending at `end`.
    type Choice = Option<(u64, Vec<usize>)>;
    let mut best: Vec<Vec<Choice>> = vec![vec![None; layers + 1]; parts + 1];
    best[0][0] = Some((0, Vec::new()));
    for part in 1..=parts {
        let ends: Vec<usize> = if part == parts { vec![layers] } else { cuts.to_vec() };
        for &end in &ends {
            let starts = std::iter::once(0).chain(cuts.iter().copied()).filter(|&start| start < end);
            let mut choice: Choice = None;
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
    let (_, ends) = best[parts][layers].clone()?;
    let mut start = 0;
    Some(ends.into_iter().map(|end| { let range = start..end; start = end; range }).collect())
}

/// How a served model is placed.
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

/// Experts still to warm up with, as (layer, expert).
type Pending = std::collections::VecDeque<(usize, usize)>;

/// The sequence state saved where a prompt's reply began, and the tokens before it.
struct Checkpoint {
    slot: usize,
    tokens: Vec<u32>,
}

/// How far before a prompt's end its reply's generation prompt may begin to be checkpointed.
const TURN_REACH: usize = 64;

/// A placed model serving requests one at a time.
pub struct Session<E: Engines> {
    pub engines: E,
    files: &'static Pinned,
    vocabulary: usize,
    /// Layers and experts per layer (a warm-up state's pairs must lie within).
    layers: usize,
    experts: usize,
    pub context: usize,
    pub batch: usize,
    /// The tokens the engines hold, in order.
    history: Vec<u32>,
    /// Tokens that end a reply (unsent).
    end: Vec<u32>,
    /// The token a reply's generation prompt begins with (`<|im_start|>`): the state before it is
    /// checkpointed when the engines keep checkpoints.
    turn_start: Option<u32>,
    /// Saved states, oldest first: the last ("turn") where the last prompt's reply began, the
    /// earlier ones ("prior") where earlier prompts' did.
    checkpoints: Vec<Checkpoint>,
    /// The sampling the model card gives, for every mode.
    pub card: Sampling,
    /// How the model was placed (the ready frame shows it).
    pub placement: Value,
    pub state_bytes: u64,
    pub load_seconds: Vec<(&'static str, f64)>,
    pub last_metrics: Map<String, Value>,
    /// An earlier run's experts still to warm up with: (VRAM, host tier), least recently used
    /// first, taken from the front.
    warming: (Pending, Pending),
}

/// What an architecture's placement gives a session.
pub struct Placed<E: Engines> {
    pub engines: E,
    pub files: &'static Pinned,
    pub vocabulary: usize,
    pub layers: usize,
    pub experts: usize,
    pub context: usize,
    pub batch: usize,
    pub end: Vec<u32>,
    pub turn_start: Option<u32>,
    pub card: Sampling,
    pub placement: Value,
    pub state_bytes: u64,
    pub load_seconds: Vec<(&'static str, f64)>,
}

/// The sampling a GGUF's `general.sampling.*` gives (the model card's), else neutral.
pub fn card(header: &straydog_gguf::ModelHeader) -> Result<Sampling, String> {
    let number = |key: &str| header.metadata.get(key).and_then(|value| value.float());
    let card = Sampling {
        temperature: number("general.sampling.temp").unwrap_or(1.0),
        top_k: number("general.sampling.top_k").map_or(0, |value| value.max(0.0) as usize),
        top_p: number("general.sampling.top_p").unwrap_or(1.0),
        min_p: number("general.sampling.min_p").unwrap_or(0.0),
        presence_penalty: 0.0,
    };
    card.check()?;
    Ok(card)
}

impl<E: Engines> Session<E> {
    pub fn new(placed: Placed<E>) -> Session<E> {
        Session {
            engines: placed.engines, files: placed.files, vocabulary: placed.vocabulary, layers: placed.layers, experts: placed.experts,
            context: placed.context, batch: placed.batch, history: Vec::new(), end: placed.end, turn_start: placed.turn_start,
            checkpoints: Vec::new(), card: placed.card,
            placement: placed.placement, state_bytes: placed.state_bytes, load_seconds: placed.load_seconds,
            last_metrics: Map::new(), warming: Default::default(),
        }
    }

    pub fn vocabulary(&self) -> usize {
        self.vocabulary
    }

    /// The model's metadata, authenticated.
    pub fn header(&self) -> &straydog_gguf::ModelHeader {
        &self.files.header
    }

    /// The model's files (resources beside them are read through the bundle's pins).
    pub fn files(&self) -> &'static Pinned {
        self.files
    }

    /// Forget the sequence (warm experts stay).
    pub fn reset(&mut self) {
        self.history.clear();
        self.engines.reset();
    }

    /// The GPUs as the server reports them: summed, then each.
    pub fn info(&self) -> Result<Value, String> {
        let devices = self.engines.gpus().into_iter().map(|gpu| gpu.info()).collect::<Result<Vec<_>, _>>()?;
        let names: Vec<&str> = devices.iter().map(|device| device["name"].as_str().unwrap_or("")).collect();
        let mut info = json!({"name": names.join(" + "), "device": devices.first().map_or(Value::Null, |device| device["device"].clone())});
        for key in ["free_bytes", "total_bytes", "allocated_bytes", "budget_bytes"] {
            info[key] = devices.iter().map(|device| device[key].as_u64().unwrap_or(0)).sum::<u64>().into();
        }
        let (hits, host, drive) = self.engines.counts();
        info["experts"] = json!({"vram_hits": hits, "host_hits": host, "drive_reads": drive});
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
            match self.engines.forward(piece) {
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
    /// sample up to `max_new` tokens, each passed to `emit`; an end token ends it unsent.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(&mut self, tokens: &[u32], max_new: usize, sampling: &Sampling, seed: u64, cancelled: &dyn Fn() -> bool,
                    progress: &mut dyn FnMut(usize, usize), emit: &mut dyn FnMut(u32)) -> Result<(), String> {
        let started = Instant::now();
        self.last_metrics = Map::new();
        if tokens.is_empty() || max_new == 0 || tokens.len() + max_new > self.context {
            return Err("Straydog prompt plus output exceeds the allocated context".into());
        }
        if tokens.iter().any(|&token| token as usize >= self.vocabulary) {
            return Err("Straydog token is out of range".into());
        }
        sampling.check()?;
        self.files.check_unchanged()?;
        let (reused, resumed_from) = self.resume(tokens)?;
        let total = tokens.len();
        // The state where this reply's generation prompt begins is saved on the way: the next
        // turn's prompt keeps what precedes it even where the template renders this reply
        // otherwise than it was generated (a past turn's thinking dropped).
        let mut fed = reused;
        if let Some(start) = self.turn_mark(tokens, reused) {
            if self.feed(&tokens[reused..start], cancelled, &mut |done| progress(done, total))?.is_none() {
                return Ok(());
            }
            self.save_turn()?;
            fed = start;
        }
        let Some(mut logits) = self.feed(&tokens[fed..], cancelled, &mut |done| progress(done, total))? else {
            return Ok(());
        };
        let prefill_seconds = started.elapsed().as_secs_f64();
        let (hits, host, drive) = self.engines.counts();
        let mut rng = Rng::new(seed);
        let mut generated = Generated::new(self.vocabulary);
        let (mut emitted, mut ttft) = (0usize, None);
        let decoding = Instant::now();
        loop {
            if cancelled() {
                break;
            }
            let token = sampling::choose(&logits, sampling, &generated, &mut rng);
            if self.end.contains(&token) {
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
        let (hits2, host2, drive2) = self.engines.counts();
        let per = |value: u64| if steps > 0 { json!(value as f64 / steps as f64) } else { Value::Null };
        self.last_metrics = json!({
            "cached_tokens": reused, "resumed_from": resumed_from,
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

    /// The prompt tokens already held, after restoring the state they need: the held sequence when
    /// the prompt extends it ("live"), else the longest checkpoint the prompt extends that the
    /// held sequence matches as far ("turn", the last, or "prior"; its K/V rows are then the held
    /// ones), else none (the engines start over). At least one prompt token is left to run.
    fn resume(&mut self, tokens: &[u32]) -> Result<(usize, &'static str), String> {
        let limit = tokens.len() - 1;
        let held = &self.history;
        if !held.is_empty() && held.len() <= limit && tokens.starts_with(held) {
            return Ok((held.len(), "live"));
        }
        let span = held.len().min(limit);
        let common = (0..span).find(|&index| held[index] != tokens[index]).unwrap_or(span);
        let best = self.checkpoints.iter().enumerate()
            .filter(|(_, checkpoint)| checkpoint.tokens.len() <= common && tokens.starts_with(&checkpoint.tokens))
            .max_by_key(|(_, checkpoint)| checkpoint.tokens.len());
        let Some((index, checkpoint)) = best else {
            self.reset();
            return Ok((0, "none"));
        };
        let kind = if index + 1 == self.checkpoints.len() { "turn" } else { "prior" };
        let (slot, saved) = (checkpoint.slot, checkpoint.tokens.clone());
        if let Err(error) = self.engines.checkpoint(slot, false) {
            self.reset();
            return Err(error);
        }
        self.history = saved;
        Ok((self.history.len(), kind))
    }

    /// Where the prompt's last generation prompt begins, to be checkpointed: past what is held,
    /// within the prompt's last [`TURN_REACH`] tokens.
    fn turn_mark(&self, tokens: &[u32], reused: usize) -> Option<usize> {
        let start = self.turn_start.filter(|_| self.engines.checkpoints() > 0)?;
        let lowest = tokens.len().saturating_sub(TURN_REACH).max(reused + 1);
        (lowest..tokens.len()).rev().find(|&index| tokens[index] == start)
    }

    /// Save the held state as the last turn's checkpoint: one of the same tokens just becomes the
    /// last; else a free slot takes it, the oldest checkpoint giving its up when all are taken.
    fn save_turn(&mut self) -> Result<(), String> {
        if let Some(index) = self.checkpoints.iter().position(|checkpoint| checkpoint.tokens == self.history) {
            let same = self.checkpoints.remove(index);
            self.checkpoints.push(same);
            return Ok(());
        }
        let slots = self.engines.checkpoints();
        if self.checkpoints.len() >= slots {
            self.checkpoints.remove(0);
        }
        let slot = (0..slots).find(|slot| self.checkpoints.iter().all(|checkpoint| checkpoint.slot != *slot))
            .ok_or("no free Straydog checkpoint")?;
        self.engines.checkpoint(slot, true)?;
        self.checkpoints.push(Checkpoint { slot, tokens: self.history.clone() });
        Ok(())
    }

    /// What a later start warms up with: the experts the engines hold, least recently used first,
    /// in VRAM (`vram`) and only in the host tier (`host`), as [layer, expert] pairs.
    pub fn warm_state(&self) -> Value {
        let (vram, host) = self.engines.contents();
        json!({"vram": vram, "host": host})
    }

    /// Warm up with an earlier run's [`warm_state`](Self::warm_state), step by step
    /// ([`warm_step`](Self::warm_step)) while no request runs. Pairs outside the model are dropped.
    pub fn warm(&mut self, state: &Value) -> Result<usize, String> {
        let (layers, experts) = (self.layers, self.experts);
        let pairs = |key: &str| -> Result<Pending, String> {
            let list = state.get(key).and_then(Value::as_array).ok_or("a Straydog warm-up state without its lists")?;
            Ok(list.iter()
                .filter_map(|pair| Some((pair.get(0)?.as_u64()? as usize, pair.get(1)?.as_u64()? as usize)))
                .filter(|&(layer, expert)| layer < layers && expert < experts)
                .collect())
        };
        self.warming = (pairs("vram")?, pairs("host")?);
        Ok(self.warming.0.len() + self.warming.1.len())
    }

    /// Take the warm-up's reads that landed and, while fewer than `in_flight` are on their way
    /// (each device's read of an expert counts), start up to `count` more (VRAM's first, the
    /// experts a request is likeliest to use, then the host tier's; each least recently used
    /// first, so the most recent end up the most recent), without waiting: a request that comes
    /// meanwhile finds the drive and the bus nearly free. Whether the warm-up is over.
    pub fn warm_step(&mut self, count: usize, in_flight: usize) -> Result<bool, String> {
        let reading = self.engines.poll()?;
        let left = self.warming.0.len() + self.warming.1.len();
        if left == 0 || reading >= in_flight {
            return Ok(left == 0 && reading == 0);
        }
        let take = |list: &mut Pending, count: usize| -> Vec<(usize, usize)> {
            let count = count.min(list.len());
            list.drain(..count).collect()
        };
        let count = count.min((in_flight - reading).div_ceil(self.engines.devices_per_expert()));
        let vram = take(&mut self.warming.0, count);
        let host = take(&mut self.warming.1, count - vram.len());
        self.engines.warm(&vram, &host)?;
        Ok(false)
    }

    /// Wait for the expert stores' reads and copies (before the process ends).
    pub fn settle(&mut self) -> Result<(), String> {
        self.engines.settle()
    }
}
