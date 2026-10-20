//! DeepSeek-V4.1 on one GPU or across several, composed from the generic ops: dense weights
//! resident, routed experts in the expert store's tiers (VRAM slots, a host tier, the files;
//! src/experts.rs), and the sequence state a forward pass continues (window rings, compressed
//! rows and index keys, compression tails, the Engram history).
//! Checked against the CPU reference by examples/ds41_gpu and examples/ds41_chat.
//!
//! Across several devices every layer is split (tensor parallel): a device runs its share of the
//! attention heads (its rows of q_b, its sinks, its groups of wo_a and their columns of wo_b), its
//! rows of the head, and its rows of every expert, shared and routed: of gate and up its share of
//! the expert's width, of down its share of the output. Each device's store holds its rows of the
//! experts, so a missing expert comes over every device's bus at once (each a share of it) and the
//! devices' drive reads run side by side; every route runs on every device, so the work is even.
//! The rest is replicated, computed alike on every device. Every device sums the devices' parts
//! in device order after the attention, of the experts' activations (each device's share of the
//! width) and after the experts, so all hold the same values.
//!
//! A decode row's next-layer experts are predicted and prefetched from the host tier; for
//! measurements, DS41_NO_PREDICT turns that off and DS41_PREDICT_DRIVE also reads predicted
//! experts from the drive (so far no faster: wrong guesses spend the drive's time).

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::Arc;

use super::{Gating, Hparams, Rope};
use crate::arch::Tensors;
use crate::experts::{Counts, DeviceExperts, Layout, Reader, Role};
use crate::formats;
use crate::gpu::{Backend, Result};
use crate::ops::{self, Buffer, Linear};

type P = *const c_void;
type M = *mut c_void;

pub struct Settings {
    /// Longest sequence (positions) the state holds.
    pub context: usize,
    /// Most tokens one forward pass takes.
    pub batch: usize,
    /// Experts the VRAM cache holds, per device (with experts split across n devices, each holds
    /// n times as many shares of experts in the same memory).
    pub expert_slots: usize,
    /// Bytes of host tier (page-locked memory) for experts VRAM evicts, per device.
    pub host_expert_bytes: u64,
    pub scratch_bytes: u64,
    /// The layers this engine runs: the first embeds the tokens, the last has the head, and an
    /// engine between takes the stream the one before hands over (see [`forward_split`]).
    pub layers: std::ops::Range<usize>,
}

#[derive(Clone)]
struct Dense {
    buffer: Buffer,
    kind: u32,
    dims: Vec<usize>,
}

/// An expert (or a device's rows of it) laid out as a slot: gate, up and down at byte offsets of
/// one buffer.
#[derive(Clone, Copy)]
struct Slot {
    buffer: Buffer,
    offsets: [u64; 3],
    kinds: [u32; 3],
}

/// A device's rows of every expert: gate and up rows [ffn.0, ffn.0 + ffn.1), down rows
/// [hidden.0, hidden.0 + hidden.1).
#[derive(Clone, Copy)]
struct Rows {
    ffn: (usize, usize),
    hidden: (usize, usize),
}

impl Rows {
    fn of(part: usize, parts: usize, ffn: usize, hidden: usize) -> Rows {
        Rows { ffn: (part * ffn / parts, ffn / parts), hidden: (part * hidden / parts, hidden / parts) }
    }

    fn split(&self, stages: i32) -> ops::ExpertSplit {
        ops::ExpertSplit { ffn_first: self.ffn.0 as u64, ffn_rows: self.ffn.1 as u64, hidden_first: self.hidden.0 as u64,
                           hidden_rows: self.hidden.1 as u64, stages }
    }
}

/// A kv source layer's compressed rows so far (one per group of `ratio` positions), and the rows
/// of its current incomplete group (`tail`).
struct Source {
    ratio: usize,
    groups: usize,
    tail: usize,
}

/// A device's copy of a kv source layer's compressed rows and index keys, and of the rows of its
/// incomplete group.
struct SourceRows {
    rows: Buffer,
    keys: Buffer,
    tail_kv: Buffer,
    tail_gate: Buffer,
}

/// Per-pass work buffers, sized for `batch` tokens.
struct Work {
    stream: Buffer,
    stream2: Buffer,
    pre_a: Buffer,
    post_a: Buffer,
    comb_a: Buffer,
    pre_f: [Buffer; 2],
    post_f: Buffer,
    comb_f: Buffer,
    identity: Buffer,
    x: Buffer,
    qr: Buffer,
    q: Buffer,
    kv: Buffer,
    attended: Buffer,
    low: Buffer,
    out: Buffer,
    /// With several devices: this device's parts of the attention's and the experts' outputs.
    parts: [Buffer; 2],
    /// With several devices: this device's share of the experts' activations (per route a row of
    /// the expert width, its share filled; two, used by waves in turn), and their sum.
    products: [Buffer; 2],
    product: Buffer,
    positions: Buffer,
    ring_rows: Buffer,
    visible: [Buffer; 3],
    pool_kv: Buffer,
    pool_gate: Buffer,
    group_positions: Buffer,
    index_q: Buffer,
    index_w: Buffer,
    scores: Buffer,
    chosen: BTreeMap<usize, Buffer>,
    router: Buffer,
    ids: Buffer,
    weights: Buffer,
    all_rows: Buffer,
    expert_rows: Buffer,
    expert_weights: Buffer,
    xe: Buffer,
    gate: Buffer,
    up: Buffer,
    hidden: Buffer,
    y: Buffer,
    engram_rows: Buffer,
    engram_kv: Buffer,
    head: Buffer,
    /// hc_step's partial sums and counters.
    hc_workspace: Buffer,
}

/// The rows routed to an expert, with their weights.
type Users = (Vec<i32>, Vec<f32>);
/// An expert and the rows routed to it.
type Routed = (usize, Users);

/// Experts acquired at a time by a pass of many rows (each wave's reads overlap the one before's
/// compute), fewer when a size of experts has few slots (DeviceExperts::wave_limit).
const WAVE: usize = 32;
/// Experts of the next layer prefetched from its router applied to this layer's input (decode):
/// the 8 best hold 79% of the 6 it takes.
const PREDICTED: usize = 8;
/// Reader threads per device: reads in flight, enough for an expert's pieces at once.
const READERS: usize = 32;
/// Passes of up to this many rows run the routed experts over their slots (decode); longer ones
/// expert by expert.
const SLOT_ROWS: usize = 4;

/// One device's share of the engine: its weights (its parts of the split ones), its copy of the
/// sequence state and work buffers, and its expert store.
struct Shard {
    /// This device's rows of the experts.
    rows: Rows,
    gpu: Backend,
    dense: HashMap<String, Dense>,
    /// Per layer: this device's rows of the shared expert, as a slot.
    shared: HashMap<usize, Slot>,
    scratch: Buffer,
    rings: BTreeMap<usize, Buffer>,
    sources: BTreeMap<usize, SourceRows>,
    plain: Buffer,
    compressed_rope: Buffer,
    work: Work,
    experts: DeviceExperts,
}

pub struct Engine<'a, T: Tensors> {
    /// Shared, so a pass takes the parameters without copying the Engram tables' metadata.
    hp: std::rc::Rc<Hparams>,
    tensors: &'a T,
    settings: Settings,
    /// One per device; a layer's work is split across them.
    shards: Vec<Shard>,
    position: usize,
    history: Vec<u32>,
    ring: usize,
    sources: BTreeMap<usize, Source>,
    /// Next-layer experts predicted by this pass, prefetched after the layer's own.
    predicted: Vec<usize>,
    /// With DS41_PROFILE set: seconds per phase, the GPUs synchronized at each phase's end.
    profile: Option<(std::time::Instant, BTreeMap<&'static str, f64>)>,
}

fn frequencies(dims: usize, rope: &Rope) -> Vec<f32> {
    let mut freqs: Vec<f64> = (0..dims / 2).map(|i| 1.0 / rope.base.powf((2 * i) as f64 / dims as f64)).collect();
    if let Some((factor, original, beta_fast, beta_slow)) = rope.yarn {
        let corrected =
            |rotations: f64| dims as f64 * (original / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * rope.base.ln());
        let low = corrected(beta_fast).floor().max(0.0);
        let high = corrected(beta_slow).ceil().min(dims as f64 - 1.0);
        for (i, freq) in freqs.iter_mut().enumerate() {
            let smooth = 1.0 - ((i as f64 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    freqs.into_iter().map(|f| f as f32).collect()
}

fn row_bytes(kind: u32, cols: usize) -> Result<usize> {
    formats::matrix_bytes(kind, 1, cols).ok_or_else(|| format!("unsupported tensor type {kind}"))
}

fn name(layer: usize, tensor: &str) -> String {
    format!("blk.{layer}.{tensor}.weight")
}

/// Columns [first, first + count) of every row of a `rows` x `cols` matrix, whole blocks only.
fn columns(kind: u32, rows: usize, cols: usize, data: &[u8], first: usize, count: usize) -> Result<Vec<u8>> {
    let whole = row_bytes(kind, cols)?;
    let (skip, take) = (formats::matrix_bytes(kind, 1, first), formats::matrix_bytes(kind, 1, count));
    let (Some(skip), Some(take)) = (skip, take) else {
        return Err(format!("columns {first}..{} of a type {kind} matrix split a block", first + count));
    };
    Ok((0..rows).flat_map(|row| &data[row * whole + skip..row * whole + skip + take]).copied().collect())
}

impl Shard {
    fn new(hp: &Hparams, settings: &Settings, device: usize, budget: u64, layout: Arc<Layout>, rows: Rows, parts: usize)
           -> Result<Shard> {
        let range = settings.layers.clone();
        let mut gpu = Backend::new(device, budget)?;
        let scratch = gpu.buffer(settings.scratch_bytes)?;
        let (b, hc, w) = (settings.batch, hp.hc, hp.width);
        let ring = hp.window + b;
        let floats = |gpu: &mut Backend, count: usize| gpu.buffer((count.max(1) * 4) as u64);
        let rings = range.clone().map(|layer| Ok((layer, floats(&mut gpu, ring * hp.head_dim)?))).collect::<Result<BTreeMap<_, _>>>()?;
        let mut sources = BTreeMap::new();
        for (layer, plan) in hp.layers.iter().enumerate().filter(|(layer, _)| range.contains(layer)) {
            if let Some(c) = plan.compressed.filter(|c| c.kv_source == layer) {
                let groups = settings.context / c.ratio + 1;
                sources.insert(layer, SourceRows {
                    rows: floats(&mut gpu, groups * hp.head_dim)?,
                    keys: floats(&mut gpu, groups * hp.index_dim)?,
                    tail_kv: floats(&mut gpu, hp.head_dim)?,
                    tail_gate: floats(&mut gpu, hp.head_dim)?,
                });
            }
        }
        let max_groups = settings.context + 1;
        let identity: Vec<f32> = (0..b).flat_map(|_| (0..hc).map(|j| f32::from(u8::from(j == 0)))).collect();
        let all_rows: Vec<i32> = (0..b as i32).collect();
        let mut chosen = BTreeMap::new();
        for (layer, plan) in hp.layers.iter().enumerate().filter(|(layer, _)| range.contains(layer)) {
            if plan.compressed.is_some_and(|c| c.index_source == layer) {
                chosen.insert(layer, floats(&mut gpu, b * hp.index_top_k)?);
            }
        }
        let work = Work {
            stream: floats(&mut gpu, b * hc * w)?,
            stream2: floats(&mut gpu, b * hc * w)?,
            pre_a: floats(&mut gpu, b * hc)?,
            post_a: floats(&mut gpu, b * hc)?,
            comb_a: floats(&mut gpu, b * hc * hc)?,
            pre_f: [floats(&mut gpu, b * hc)?, floats(&mut gpu, b * hc)?],
            post_f: floats(&mut gpu, b * hc)?,
            comb_f: floats(&mut gpu, b * hc * hc)?,
            identity: floats(&mut gpu, b * hc)?,
            x: floats(&mut gpu, b * w)?,
            qr: floats(&mut gpu, b * hp.q_rank)?,
            q: floats(&mut gpu, b * hp.heads * hp.head_dim)?,
            kv: floats(&mut gpu, b * hp.head_dim)?,
            attended: floats(&mut gpu, b * hp.heads * hp.head_dim)?,
            low: floats(&mut gpu, b * hp.out_groups * hp.out_rank)?,
            out: floats(&mut gpu, b * w)?,
            parts: [floats(&mut gpu, b * w)?, floats(&mut gpu, b * w)?],
            products: [floats(&mut gpu, b * (hp.experts_used + 1) * hp.expert_ff)?, floats(&mut gpu, b * (hp.experts_used + 1) * hp.expert_ff)?],
            product: floats(&mut gpu, b * (hp.experts_used + 1) * hp.expert_ff)?,
            positions: floats(&mut gpu, b)?,
            ring_rows: floats(&mut gpu, b)?,
            visible: [floats(&mut gpu, b)?, floats(&mut gpu, b)?, floats(&mut gpu, b)?],
            pool_kv: floats(&mut gpu, (b + 1) * hp.head_dim)?,
            pool_gate: floats(&mut gpu, (b + 1) * hp.head_dim)?,
            group_positions: floats(&mut gpu, b + 1)?,
            index_q: floats(&mut gpu, b * hp.index_heads * hp.index_dim)?,
            index_w: floats(&mut gpu, b * hp.index_heads)?,
            scores: floats(&mut gpu, b * max_groups)?,
            chosen,
            router: floats(&mut gpu, (b + 1) * hp.experts)?,
            ids: floats(&mut gpu, b * hp.experts_used + PREDICTED)?,
            weights: floats(&mut gpu, b * hp.experts_used)?,
            all_rows: floats(&mut gpu, b)?,
            expert_rows: floats(&mut gpu, b)?,
            expert_weights: floats(&mut gpu, b.max(PREDICTED))?,
            xe: floats(&mut gpu, b * w)?,
            gate: floats(&mut gpu, b * hp.expert_ff)?,
            up: floats(&mut gpu, b * hp.expert_ff)?,
            hidden: floats(&mut gpu, b * hp.experts_used.max(1) * hp.expert_ff)?,
            y: floats(&mut gpu, b * w)?,
            engram_rows: floats(&mut gpu, b * hp.engram_columns() * hp.engram.as_ref().map_or(1, |e| e.head_dim))?,
            engram_kv: floats(&mut gpu, b * (hc + 1) * w)?,
            head: floats(&mut gpu, if range.end == hp.layers.len() { hp.vocabulary } else { 1 })?,
            hc_workspace: floats(&mut gpu, b * w.div_ceil(64) * ((2 + hc) * hc + 2) + b)?,
        };
        gpu.write_buffer(&work.identity, 0, &identity)?;
        gpu.write_buffer(&work.all_rows, 0, &all_rows)?;
        let plain = frequencies(hp.rope_dims, &hp.rope);
        let compressed = frequencies(hp.rope_dims, &hp.compressed_rope);
        let plain_buffer = floats(&mut gpu, plain.len())?;
        gpu.write_buffer(&plain_buffer, 0, &plain)?;
        let compressed_buffer = floats(&mut gpu, compressed.len())?;
        gpu.write_buffer(&compressed_buffer, 0, &compressed)?;
        let experts = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(READERS)), settings.expert_slots * parts,
                                         settings.host_expert_bytes)?;
        Ok(Shard {
            rows,
            gpu, dense: HashMap::new(), shared: HashMap::new(), scratch, rings, sources, plain: plain_buffer,
            compressed_rope: compressed_buffer, work, experts,
        })
    }

    fn dense(&self, name: &str) -> Result<Dense> {
        self.dense.get(name).cloned().ok_or_else(|| format!("tensor {name} is not resident"))
    }

    fn upload(&mut self, name: String, kind: u32, dims: Vec<usize>, data: &[u8]) -> Result<()> {
        let buffer = self.gpu.upload_bytes(data)?;
        self.dense.insert(name, Dense { buffer, kind, dims });
        Ok(())
    }

    /// out = W . input over `steps` rows (a whole resident 2D tensor).
    fn linear(&mut self, name: &str, input: P, steps: usize, out: M) -> Result<()> {
        let weight = self.dense(name)?;
        let scratch = self.scratch;
        self.gpu.linear(&weight.buffer, weight.kind, weight.dims[1] as u64, weight.dims[0] as u64, 0, input, 0, steps as u64, out, 0,
                        &scratch)
    }

    fn norm(&mut self, name: &str, x: M, rows: usize, width: usize, eps: f32) -> Result<()> {
        let weight = self.dense(name)?;
        self.gpu.rms_norm(x, &weight.buffer, rows as u64, width as u64, eps, x)
    }
}

impl<'a, T: Tensors> Engine<'a, T> {
    /// An engine over `devices`: one, or several splitting every layer (tensor parallel).
    pub fn new(hp: Hparams, tensors: &'a T, devices: &[usize], budget: u64, settings: Settings) -> Result<Engine<'a, T>> {
        let range = settings.layers.clone();
        if range.is_empty() || range.end > hp.layers.len() {
            return Err("DeepSeek-V4.1 engine layers outside the model".into());
        }
        for plan in &hp.layers[range.clone()] {
            if let Some(c) = plan.compressed.filter(|c| c.kv_source < range.start || c.index_source < range.start) {
                return Err(format!("DeepSeek-V4.1 layers {range:?} read layer {}'s compressed rows", c.kv_source.min(c.index_source)));
            }
        }
        let parts = devices.len();
        if parts == 0 || !hp.heads.is_multiple_of(hp.out_groups) || !hp.out_groups.is_multiple_of(parts)
            || !hp.vocabulary.is_multiple_of(parts) || !hp.expert_ff.is_multiple_of(parts) || !hp.width.is_multiple_of(parts)
        {
            return Err(format!("DeepSeek-V4.1 does not split over {parts} devices"));
        }
        // Each device's rows of each layer's experts as three roles of the files (a slot holds them
        // back to back).
        let files = tensors.files()?;
        let mut shards = Vec::with_capacity(parts);
        for (part, &device) in devices.iter().enumerate() {
            let rows = Rows::of(part, parts, hp.expert_ff, hp.width);
            let mut roles = vec![None; hp.layers.len()];
            for layer in range.clone() {
                let mut layer_roles = Vec::new();
                for (role, (first, count)) in [("ffn_gate_exps", rows.ffn), ("ffn_up_exps", rows.ffn), ("ffn_down_exps", rows.hidden)] {
                    let tensor_name = name(layer, role);
                    let tensor = tensors.tensor(&tensor_name)?;
                    let (file, base) = tensors.location(&tensor_name).ok_or("the expert store needs the model's files")?;
                    let bytes = row_bytes(tensor.kind, tensor.dims[0])?;
                    let stride = (bytes * tensor.dims[1]) as u64;
                    layer_roles.push(Role { file, base: base + (first * bytes) as u64, stride, bytes: (count * bytes) as u64 });
                }
                roles[layer] = Some(layer_roles);
            }
            let layout = Arc::new(Layout::new(&files, roles, tensors.verifier())?);
            shards.push(Shard::new(&hp, &settings, device, budget, layout, rows, parts)?);
        }
        let mut sources = BTreeMap::new();
        for (layer, plan) in hp.layers.iter().enumerate().filter(|(layer, _)| range.contains(layer)) {
            if let Some(c) = plan.compressed.filter(|c| c.kv_source == layer) {
                sources.insert(layer, Source { ratio: c.ratio, groups: 0, tail: 0 });
            }
        }
        let mut engine = Engine {
            ring: hp.window + settings.batch,
            hp: std::rc::Rc::new(hp),
            tensors,
            settings,
            shards,
            position: 0,
            history: Vec::new(),
            sources,
            predicted: Vec::new(),
            profile: std::env::var_os("DS41_PROFILE").map(|_| (std::time::Instant::now(), BTreeMap::new())),
        };
        engine.upload_dense()?;
        Ok(engine)
    }

    /// Every tensor of this engine's layers but the routed experts and the Engram tables, resident
    /// (with the head on the last engine): each device's parts of the split ones (see the module
    /// documentation), the rest whole; each shared expert's part as a slot.
    fn upload_dense(&mut self) -> Result<()> {
        let hp = self.hp.clone();
        let range = self.settings.layers.clone();
        let parts = self.shards.len();
        let mut names = Vec::new();
        if range.end == hp.layers.len() {
            names.extend(["output.weight".to_string(), "output_norm.weight".to_string()]);
        }
        for layer in range.clone() {
            for tensor in ["attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv", "attn_kv_a_norm", "attn_sinks",
                           "attn_output_a", "attn_output_b", "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn",
                           "hc_ffn_base", "hc_ffn_scale", "ffn_norm", "ffn_gate_inp", "attn_compressor_kv", "attn_compressor_gate",
                           "attn_compressor_norm", "indexer.attn_k", "indexer.k_norm", "indexer.attn_q_b", "indexer.proj",
                           "engram_kv", "engram_gate"] {
                names.push(name(layer, tensor));
            }
            names.push(format!("blk.{layer}.exp_probs_b.bias"));
        }
        for name in names {
            let Ok(tensor) = self.tensors.tensor(&name) else { continue };
            if parts == 1 || !divided(&name) {
                for shard in &mut self.shards {
                    shard.upload(name.clone(), tensor.kind, tensor.dims.clone(), tensor.data)?;
                }
                continue;
            }
            let (cols, rows) = (tensor.dims[0], tensor.dims.get(1).copied().unwrap_or(1));
            for (part, shard) in self.shards.iter_mut().enumerate() {
                if name.ends_with(".attn_output_b.weight") {
                    // its groups' columns: a part of every output
                    let count = cols / parts;
                    let data = columns(tensor.kind, rows, cols, tensor.data, part * count, count)?;
                    shard.upload(name.clone(), tensor.kind, vec![count, rows], &data)?;
                } else if name.ends_with(".attn_sinks.weight") {
                    let count = cols / parts;
                    let data = &tensor.data[part * count * 4..(part + 1) * count * 4];
                    shard.upload(name.clone(), tensor.kind, vec![count], data)?;
                } else {
                    // its rows: of q_b its heads, of wo_a its groups, of the head its tokens
                    let (count, bytes) = (rows / parts, row_bytes(tensor.kind, cols)?);
                    let data = &tensor.data[part * count * bytes..(part + 1) * count * bytes];
                    shard.upload(name.clone(), tensor.kind, vec![cols, count], data)?;
                }
            }
        }
        for layer in range {
            let roles = ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"].map(|role| self.tensors.tensor(&name(layer, role)));
            let [gate, up, down] = roles;
            let (gate, up, down) = (gate?, up?, down?);
            for shard in &mut self.shards {
                let rows = |tensor: &crate::arch::Tensor<'_>, (first, count): (usize, usize)| -> Result<Vec<u8>> {
                    let bytes = row_bytes(tensor.kind, tensor.dims[0])?;
                    Ok(tensor.data[first * bytes..(first + count) * bytes].to_vec())
                };
                let parts = [rows(&gate, shard.rows.ffn)?, rows(&up, shard.rows.ffn)?, rows(&down, shard.rows.hidden)?];
                let offsets = [0, parts[0].len() as u64, (parts[0].len() + parts[1].len()) as u64];
                let buffer = shard.gpu.upload_bytes(&parts.concat())?;
                shard.shared.insert(layer, Slot { buffer, offsets, kinds: [gate.kind, up.kind, down.kind] });
            }
        }
        Ok(())
    }

    /// Attribute the time since the last mark to `phase` (profiling only: waits for the GPUs).
    fn mark(&mut self, phase: &'static str) -> Result<()> {
        if self.profile.is_none() {
            return Ok(());
        }
        for shard in &mut self.shards {
            let mut one = [0f32; 1];
            let x = shard.work.x;
            shard.gpu.read_buffer(&x, 0, &mut one)?;
        }
        let (since, phases) = self.profile.as_mut().unwrap();
        *phases.entry(phase).or_default() += since.elapsed().as_secs_f64();
        *since = std::time::Instant::now();
        Ok(())
    }

    /// Seconds per phase so far (DS41_PROFILE), and restart the count.
    pub fn take_profile(&mut self) -> BTreeMap<&'static str, f64> {
        match self.profile.as_mut() {
            Some((since, phases)) => {
                *since = std::time::Instant::now();
                std::mem::take(phases)
            }
            None => BTreeMap::new(),
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// The devices this engine runs on, one per shard (memory and identity for reports).
    pub fn gpus(&self) -> impl Iterator<Item = &Backend> {
        self.shards.iter().map(|shard| &shard.gpu)
    }

    /// How many experts the devices hold in VRAM and can keep in the host tier (each device its
    /// rows of the same experts).
    pub fn expert_capacity(&self) -> (usize, usize) {
        self.shards.first().map_or((0, 0), |shard| shard.experts.capacity())
    }

    /// The expert stores' counters, summed over the devices.
    pub fn cache_counts(&self) -> Counts {
        self.shards.iter().map(|shard| shard.experts.counts()).fold(Counts::default(), |sum, counts| sum + counts)
    }

    /// Wait for the expert stores' reads and copies (before dropping the engine).
    pub fn settle(&mut self) -> Result<()> {
        for shard in &mut self.shards {
            shard.experts.settle(&mut shard.gpu)?;
        }
        Ok(())
    }

    /// Forget the sequence (the caches' contents stay, as warm experts do).
    pub fn reset(&mut self) {
        self.position = 0;
        self.history.clear();
        for source in self.sources.values_mut() {
            source.groups = 0;
            source.tail = 0;
        }
    }

    /// With several devices: every device's `out` = the sum of the devices' parts `part`.
    fn sum_parts(&mut self, part: usize, rows: usize) -> Result<()> {
        if self.shards.len() == 1 {
            return Ok(());
        }
        let devices: Vec<&Backend> = self.shards.iter().map(|shard| &shard.gpu).collect();
        let outs: Vec<M> = self.shards.iter().map(|shard| shard.work.out.base()).collect();
        let parts: Vec<P> = self.shards.iter().map(|shard| shard.work.parts[part].base() as P).collect();
        ops::sum_devices(&devices, &outs, &parts, (rows * self.hp.width) as u64)
    }

    /// Where a device's share of a sublayer's output goes: `out` itself on one device, else its part.
    fn target(&self, shard: &Shard, part: usize) -> M {
        if self.shards.len() == 1 { shard.work.out.base() } else { shard.work.parts[part].base() }
    }

    /// Advance the sequence over `tokens` (at most the batch); the logits after the last one. The
    /// engine runs every layer.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.pass(tokens)?.ok_or_else(|| "this DeepSeek-V4.1 engine runs only some of the layers".into())
    }

    /// Run this engine's layers over `tokens`: the first engine embeds them, a later one continues
    /// the stream handed over to it ([`Engine::hand_over`]); the last returns the logits after the
    /// last token.
    pub fn pass(&mut self, tokens: &[u32]) -> Result<Option<Vec<f32>>> {
        let n = tokens.len();
        if n == 0 || n > self.settings.batch || self.position + n > self.settings.context {
            return Err("DeepSeek-V4.1 forward exceeds the batch or the context".into());
        }
        let hp = self.hp.clone();
        let (hc, w) = (hp.hc, hp.width);
        let range = self.settings.layers.clone();
        if let Some((since, _)) = self.profile.as_mut() {
            // another engine's pass ran since this one's last mark
            *since = std::time::Instant::now();
        }
        let start = self.position;
        let positions: Vec<i32> = (start..start + n).map(|p| p as i32).collect();
        let ring_rows: Vec<i32> = (start..start + n).map(|p| (p % self.ring) as i32).collect();
        let visible: Vec<Vec<i32>> = (0..3).map(|ratio| (start..start + n).map(|p| ((p + 1) / ratio.max(1)) as i32).collect()).collect();
        // embeddings repeated into the hc copies
        let mut stream = Vec::new();
        if range.start == 0 {
            let embed = self.tensors.tensor("token_embd.weight")?;
            let bytes = row_bytes(embed.kind, embed.dims[0])?;
            stream.reserve(n * hc * w);
            let mut row = Vec::new();
            for &token in tokens {
                row.clear();
                formats::decode_into(embed.kind, &embed.data[token as usize * bytes..(token as usize + 1) * bytes], &mut row)?;
                for _ in 0..hc {
                    stream.extend_from_slice(&row);
                }
            }
        }
        for shard in &mut self.shards {
            let work = &shard.work;
            shard.gpu.write_buffer(&work.positions, 0, &positions)?;
            shard.gpu.write_buffer(&work.ring_rows, 0, &ring_rows)?;
            for ratio in [1usize, 2] {
                shard.gpu.write_buffer(&work.visible[ratio], 0, &visible[ratio])?;
            }
            if !stream.is_empty() {
                shard.gpu.write_buffer(&work.stream, 0, &stream)?;
            }
        }
        if hp.layers[range.clone()].iter().any(|plan| plan.engram.is_some()) {
            self.prefetch_engram(tokens)?;
        }
        self.mark("embedding")?;
        // A handed-over stream comes expanded, with the collapse coefficients of the layer before.
        let mut pre_previous = range.start.checked_sub(1).map(|layer| layer % 2);
        for layer in range.clone() {
            self.layer(layer, tokens, n, pre_previous, layer > range.start)?;
            pre_previous = Some(layer % 2);
        }
        let last_layer = range.end - 1;
        let logits = if range.end < hp.layers.len() {
            for shard in &mut self.shards {
                let work = &shard.work;
                shard.gpu.hc_expand(work.out.base(), work.stream2.base(), work.post_f.base(), work.comb_f.base(), n as u64, hc as u64,
                                    w as u64, work.stream.base())?;
            }
            None
        } else {
            // The last row's expansion of the last FFN output, collapsed and normed for the head;
            // each device has its rows of the head.
            let (last, part_rows) = (n - 1, hp.vocabulary / self.shards.len());
            let mut logits = vec![0f32; hp.vocabulary];
            for shard in &mut self.shards {
                let work = shard.work.clone_handles();
                let pre = work.pre_f[last_layer % 2];
                let norm = shard.dense("output_norm.weight")?;
                let expand = (work.out.at(last * w) as P, work.stream2.at(last * hc * w) as P, work.post_f.at(last * hc) as P,
                              work.comb_f.at(last * hc * hc) as P);
                shard.gpu.hc_step(Some(expand), work.stream.at(last * hc * w), None, 0, 0.0, hp.norm_eps, pre.at(last * hc),
                                  norm.buffer.base(), 1, hc as u64, w as u64, &work.hc_workspace, work.x.base())?;
                shard.linear("output.weight", work.x.base(), 1, work.head.base())?;
            }
            for (part, shard) in self.shards.iter_mut().enumerate() {
                let head = shard.work.head;
                shard.gpu.read_buffer(&head, 0, &mut logits[part * part_rows..(part + 1) * part_rows])?;
            }
            self.mark("head")?;
            Some(logits)
        };
        self.history.extend_from_slice(tokens);
        self.position += n;
        Ok(logits)
    }

    /// Hand the stream of the pass that just ran over `rows` tokens to the engine of the next layers.
    pub fn hand_over(&mut self, next: &mut Engine<'a, T>, rows: usize) -> Result<()> {
        let (hc, w) = (self.hp.hc, self.hp.width);
        let pre = (self.settings.layers.end - 1) % 2;
        if next.settings.layers.start != self.settings.layers.end {
            return Err("DeepSeek-V4.1 engines hand over between adjacent layers only".into());
        }
        let from = &mut self.shards[0];
        let (stream, coefficients) = (from.work.stream, from.work.pre_f[pre]);
        for to in &mut next.shards {
            from.gpu.copy_to(stream.base(), &mut to.gpu, to.work.stream.base(), (rows * hc * w * 4) as u64)?;
            from.gpu.copy_to(coefficients.base(), &mut to.gpu, to.work.pre_f[pre].base(), (rows * hc * 4) as u64)?;
        }
        Ok(())
    }

    /// One layer; with `expand`, the previous layer's FFN output (in `out`) is expanded into `stream`
    /// on the way in.
    fn layer(&mut self, layer: usize, tokens: &[u32], n: usize, pre_previous: Option<usize>, expand: bool) -> Result<()> {
        let hp = self.hp.clone();
        let (hc, w, rows) = (hp.hc as u64, hp.width as u64, n as u64);
        let plan = &hp.layers[layer];
        let mut expand = expand;
        if let Some(module) = plan.engram {
            if std::mem::take(&mut expand) {
                for shard in &mut self.shards {
                    let work = &shard.work;
                    shard.gpu.hc_expand(work.out.base(), work.stream2.base(), work.post_f.base(), work.comb_f.base(), rows, hc, w,
                                        work.stream.base())?;
                }
            }
            self.engram(layer, module, tokens, n)?;
            self.mark("engram")?;
        }
        // Decode repeats a layer's launches every token: a device queues each run of them that
        // takes no host data at once, from a graph recorded the first time (not while profiling).
        let record = n == 1 && self.profile.is_none() && std::env::var_os("DS41_NO_GRAPHS").is_none();
        let attention = self.attention_plan(layer, n);
        // a layer that compresses or indexes changes its launches from token to token
        let fixed = plan.compressed.is_none_or(|c| c.kv_source != layer && c.index_source != layer);
        let key = |segment: u64| (layer as u64) << 8 | segment << 4 | u64::from(expand) << 1 | u64::from(attention.existing > 0);
        for index in 0..self.shards.len() {
            let target = self.target(&self.shards[index], 0);
            let shard = &mut self.shards[index];
            if record && fixed {
                recorded(shard, key(0), |shard| {
                    hc_attention(shard, &hp, layer, n, expand, pre_previous)?;
                    attention_ops(shard, &hp, &attention, target)
                })?;
            } else {
                hc_attention(shard, &hp, layer, n, expand, pre_previous)?;
            }
        }
        if !(record && fixed) {
            self.mark("hyper-connections")?;
            for index in 0..self.shards.len() {
                let target = self.target(&self.shards[index], 0);
                attention_ops(&mut self.shards[index], &hp, &attention, target)?;
            }
        }
        if let Some((ratio, tail, _)) = attention.appended {
            let source = self.sources.get_mut(&layer).unwrap();
            source.groups += (tail + n) / ratio;
            source.tail = (tail + n) % ratio;
        }
        self.sum_parts(0, n)?;
        self.mark("attention")?;
        let predict = n == 1 && self.settings.layers.contains(&(layer + 1)) && std::env::var_os("DS41_NO_PREDICT").is_none();
        for (index, shard) in self.shards.iter_mut().enumerate() {
            // the next layer's experts are predicted on the first device alone
            let predict = predict && index == 0;
            if record {
                recorded(shard, key(1) | u64::from(predict) << 2, |shard| {
                    hc_ffn(shard, &hp, layer, n)?;
                    router_ops(shard, &hp, layer, n, predict)
                })?;
            } else {
                hc_ffn(shard, &hp, layer, n)?;
            }
        }
        if !record {
            self.mark("hyper-connections")?;
            for (index, shard) in self.shards.iter_mut().enumerate() {
                router_ops(shard, &hp, layer, n, predict && index == 0)?;
            }
        }
        self.mark("router")?;
        self.moe(layer, n, predict)?;
        self.sum_parts(1, n)
    }

    /// What every device's attention of `layer` needs to know of the sequence.
    fn attention_plan(&self, layer: usize, n: usize) -> AttentionPlan {
        let compressed = self.hp.layers[layer].compressed;
        AttentionPlan {
            layer,
            n,
            parts: self.shards.len(),
            compressed,
            appended: compressed.filter(|c| c.kv_source == layer).map(|_| {
                let source = &self.sources[&layer];
                (source.ratio, source.tail, source.groups)
            }),
            existing: compressed.map_or(0, |c| self.sources[&c.kv_source].groups),
            ring_rows: self.ring as u64,
        }
    }

    /// Start reading every Engram row these tokens will gather, while the layers before
    /// the modules run.
    fn prefetch_engram(&self, tokens: &[u32]) -> Result<()> {
        let Some(engram) = self.hp.engram.as_ref() else { return Ok(()) };
        let mut history = self.history[self.history.len().saturating_sub(engram.ngram - 1)..].to_vec();
        let mut rows = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let before = history[history.len().saturating_sub(engram.ngram - 1)..].to_vec();
            for module in 0..engram.multipliers.len() / engram.ngram {
                engram.rows(module, token, &before, self.position + i, &mut rows);
            }
            history.push(token);
        }
        self.tensors.prefetch_rows("per_layer_token_embd.weight", &rows)
    }

    fn engram(&mut self, layer: usize, module: usize, tokens: &[u32], n: usize) -> Result<()> {
        let hp = self.hp.clone();
        let engram = hp.engram.as_ref().ok_or("no Engram metadata")?;
        let table = self.tensors.tensor("per_layer_token_embd.weight")?;
        let (kind, width) = (table.kind, row_bytes(table.kind, table.dims[0])?);
        let mut history = self.history[self.history.len().saturating_sub(engram.ngram - 1)..].to_vec();
        let mut rows = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let before = &history[history.len().saturating_sub(engram.ngram - 1)..];
            engram.rows(module, token, before, self.position + i, &mut rows);
            history.push(token);
        }
        let bytes = self.tensors.rows("per_layer_token_embd.weight", &rows)?;
        let mut gathered = Vec::with_capacity(n * hp.engram_columns() * engram.head_dim);
        for row in bytes.chunks_exact(width) {
            formats::decode_into(kind, row, &mut gathered)?;
        }
        for shard in &mut self.shards {
            let work = shard.work.clone_handles();
            shard.gpu.write_buffer(&work.engram_rows, 0, &gathered)?;
            shard.linear(&name(layer, "engram_kv"), work.engram_rows.base(), n, work.engram_kv.base())?;
            let gate = shard.dense(&name(layer, "engram_gate"))?;
            shard.gpu.engram_gate(work.stream.base(), work.engram_kv.base(), gate.buffer.base(), n as u64, hp.hc as u64,
                                  hp.width as u64, hp.norm_eps)?;
        }
        Ok(())
    }

    /// The experts of `layer`, after the router (see [`router_ops`]); with `predict`, the first
    /// device also predicted the next layer's.
    fn moe(&mut self, layer: usize, n: usize, predict: bool) -> Result<()> {
        let hp = self.hp.clone();
        let (w, k) = (hp.width, hp.experts_used);
        // The routes (alike on every device) travel to the host while the shared expert runs.
        let read = n * k + if predict { PREDICTED } else { 0 };
        let ids = self.shards[0].work.ids;
        self.shards[0].gpu.read_start(&ids, 0, (read * 4) as u64)?;
        if self.shards.len() > 1 {
            return self.experts_split(layer, n, read);
        }
        // One device: the shared expert writes `out`, the routed ones add to it.
        let ff = hp.expert_ff;
        let shard = &mut self.shards[0];
        let work = shard.work.clone_handles();
        let shared = shard.shared[&layer];
        if n <= SLOT_ROWS {
            shard.gpu.experts(Some(&vec![shared.buffer.base() as u64; n]), shared.buffer.bytes, shared.offsets, shared.kinds, w as u64,
                              ff as u64, n as u64, 1, work.x.base(), 0, None, hp.layers[layer].clamp_shared, false, work.hidden.base(),
                              work.out.base(), 0, None)?;
        } else {
            let scratch = shard.scratch;
            let role = |r: usize, bytes: u64| shared.buffer.slice(shared.offsets[r], bytes);
            let (gate, up) = (role(0, shared.offsets[1]), role(1, shared.offsets[2] - shared.offsets[1]));
            let down = role(2, shared.buffer.bytes - shared.offsets[2]);
            shard.gpu.linear(&gate, shared.kinds[0], ff as u64, w as u64, 0, work.x.base(), 0, n as u64, work.gate.base(), 0, &scratch)?;
            shard.gpu.linear(&up, shared.kinds[1], ff as u64, w as u64, 0, work.x.base(), 0, n as u64, work.up.base(), 0, &scratch)?;
            shard.gpu.swiglu(work.gate.base(), work.up.base(), n as u64, ff as u64, hp.layers[layer].clamp_shared, None,
                             work.hidden.base())?;
            shard.gpu.linear(&down, shared.kinds[2], w as u64, ff as u64, 0, work.hidden.base(), 0, n as u64, work.out.base(), 0,
                             &scratch)?;
        }
        self.mark("shared expert")?;
        let mut routes = vec![0i32; read];
        self.shards[0].gpu.read_finish(&mut routes)?;
        self.predicted = routes.split_off(n * k).into_iter().map(|expert| expert as usize).collect();
        if n <= SLOT_ROWS {
            return self.routed(layer, n, &routes);
        }
        let shard = &mut self.shards[0];
        let mut weights = vec![0f32; n * k];
        shard.gpu.read_buffer(&work.weights, 0, &mut weights)?;
        let mut by_expert: BTreeMap<usize, Users> = BTreeMap::new();
        for (slot, &expert) in routes.iter().enumerate() {
            let entry = by_expert.entry(expert as usize).or_default();
            entry.0.push((slot / k) as i32);
            entry.1.push(weights[slot]);
        }
        let gate = self.tensors.tensor(&name(layer, "ffn_gate_exps"))?;
        let up = self.tensors.tensor(&name(layer, "ffn_up_exps"))?;
        let down = self.tensors.tensor(&name(layer, "ffn_down_exps"))?;
        let (gate_bytes, down_bytes) = (ff * row_bytes(gate.kind, w)?, w * row_bytes(down.kind, ff)?);
        let (gate_kind, up_kind, down_kind) = (gate.kind, up.kind, down.kind);
        let scratch = shard.scratch;
        // In waves: the next wave's experts load while this one's compute.
        let by_expert: Vec<Routed> = by_expert.into_iter().collect();
        let waves: Vec<&[Routed]> = by_expert.chunks(WAVE.min(shard.experts.wave_limit())).collect();
        for (index, wave) in waves.iter().enumerate() {
            let experts: Vec<usize> = wave.iter().map(|(expert, _)| *expert).collect();
            let slots = shard.experts.acquire(&mut shard.gpu, layer, &experts)?;
            if let Some(next) = waves.get(index + 1) {
                let experts: Vec<usize> = next.iter().map(|(expert, _)| *expert).collect();
                shard.experts.prefetch(&mut shard.gpu, layer, &experts, true)?;
            }
            for ((_, (rows, user_weights)), slot) in wave.iter().zip(slots) {
                let count = rows.len();
                shard.gpu.write_buffer(&work.expert_rows, 0, rows)?;
                shard.gpu.write_buffer(&work.expert_weights, 0, user_weights)?;
                shard.gpu.gather(work.x.base(), n as u64, work.expert_rows.base(), count as u64, w as u64, work.xe.base())?;
                let gate_weights = slot.slice(0, gate_bytes as u64);
                let up_weights = slot.slice(gate_bytes as u64, gate_bytes as u64);
                let down_weights = slot.slice(2 * gate_bytes as u64, down_bytes as u64);
                shard.gpu.linear(&gate_weights, gate_kind, ff as u64, w as u64, 0, work.xe.base(), 0, count as u64, work.gate.base(), 0,
                                 &scratch)?;
                shard.gpu.linear(&up_weights, up_kind, ff as u64, w as u64, 0, work.xe.base(), 0, count as u64, work.up.base(), 0,
                                 &scratch)?;
                shard.gpu.swiglu(work.gate.base(), work.up.base(), count as u64, ff as u64, hp.layers[layer].clamp_routed,
                                 Some(work.expert_weights.base()), work.hidden.base())?;
                shard.gpu.linear(&down_weights, down_kind, w as u64, ff as u64, 0, work.hidden.base(), 0, count as u64, work.y.base(), 0,
                                 &scratch)?;
                shard.gpu.scatter_add(work.y.base(), work.expert_rows.base(), count as u64, w as u64, work.out.base(), n as u64)?;
            }
            shard.experts.release(&mut shard.gpu)?;
        }
        self.mark("routed experts")
    }

    /// The routed experts of a few rows on one device (decode): every route's expert made
    /// resident, then both stages over the slots in one launch each, added to `out`.
    fn routed(&mut self, layer: usize, n: usize, routes: &[i32]) -> Result<()> {
        let (w, ff) = (self.hp.width, self.hp.expert_ff);
        let (gate_bytes, kinds) = self.expert_kinds(layer, self.shards[0].rows)?;
        let shard = &mut self.shards[0];
        let work = shard.work.clone_handles();
        let experts: Vec<usize> = routes.iter().map(|&expert| expert as usize).collect::<std::collections::BTreeSet<_>>()
            .into_iter().collect();
        let slots = shard.experts.acquire(&mut shard.gpu, layer, &experts)?;
        // The next layer's predicted experts start their reads and copies behind this layer's.
        let predicted = std::mem::take(&mut self.predicted);
        if !predicted.is_empty() {
            let drive = std::env::var_os("DS41_PREDICT_DRIVE").is_some();
            shard.experts.prefetch(&mut shard.gpu, layer + 1, &predicted, drive)?;
        }
        self.mark("expert fills")?;
        let shard = &mut self.shards[0];
        let bases: Vec<u64> = routes.iter()
            .map(|expert| slots[experts.binary_search(&(*expert as usize)).unwrap()].base() as u64).collect();
        let slot_bytes = slots.first().map_or(0, |slot| slot.bytes);
        let offsets = [0, gate_bytes, 2 * gate_bytes];
        shard.gpu.experts(Some(&bases), slot_bytes, offsets, kinds, w as u64, ff as u64, n as u64, self.hp.experts_used as u64, work.x.base(),
                          0, Some(work.weights.base()), self.hp.layers[layer].clamp_routed, true, work.hidden.base(), work.out.base(), 0,
                          None)?;
        shard.experts.release(&mut shard.gpu)?;
        self.mark("routed experts")
    }

    /// A layer's routed expert formats, and the bytes of a device's rows of an expert's gate (or
    /// up) in a slot.
    fn expert_kinds(&self, layer: usize, rows: Rows) -> Result<(u64, [u32; 3])> {
        let gate = self.tensors.tensor(&name(layer, "ffn_gate_exps"))?;
        let up = self.tensors.tensor(&name(layer, "ffn_up_exps"))?;
        let down = self.tensors.tensor(&name(layer, "ffn_down_exps"))?;
        Ok(((rows.ffn.1 * row_bytes(gate.kind, self.hp.width)?) as u64, [gate.kind, up.kind, down.kind]))
    }

    /// The experts on several devices, each with its rows of every expert: on every device, the
    /// shared expert's and every route's gate and up (its share of the width), the activations
    /// summed across devices, then its rows of down into its part of the output. The routes'
    /// experts come in waves (one for decode): a wave's fills start on every device before any
    /// waits, and the next wave's while this one computes.
    fn experts_split(&mut self, layer: usize, n: usize, read: usize) -> Result<()> {
        let hp = self.hp.clone();
        let (w, ff, k) = (hp.width as u64, hp.expert_ff as u64, hp.experts_used);
        let routed_rows = n * k;
        for shard in &mut self.shards {
            let work = shard.work.clone_handles();
            let shared = shard.shared[&layer];
            shard.gpu.experts(Some(&vec![shared.buffer.base() as u64; n]), shared.buffer.bytes, shared.offsets, shared.kinds, w, ff, n as u64, 1,
                              work.x.base(), 0, None, hp.layers[layer].clamp_shared, false, work.products[0].at(routed_rows * ff as usize),
                              work.parts[1].base(), 0, Some(shard.rows.split(1)))?;
        }
        self.mark("shared expert")?;
        let mut routes = vec![0i32; read];
        self.shards[0].gpu.read_finish(&mut routes)?;
        let predicted: Vec<usize> = routes.split_off(routed_rows).into_iter().map(|expert| expert as usize).collect();
        let experts: Vec<usize> = routes.iter().map(|&expert| expert as usize).collect::<std::collections::BTreeSet<_>>()
            .into_iter().collect();
        let wave = self.shards.iter().map(|shard| shard.experts.wave_limit()).fold(WAVE, usize::min);
        let waves: Vec<&[usize]> = experts.chunks(wave).collect();
        if let Some(first) = waves.first() {
            for shard in &mut self.shards {
                shard.experts.request(&mut shard.gpu, layer, first)?;
            }
        }
        let drive = std::env::var_os("DS41_PREDICT_DRIVE").is_some();
        // every device holds as many rows of an expert
        let (gate_bytes, kinds) = self.expert_kinds(layer, self.shards[0].rows)?;
        for (index, wave) in waves.iter().enumerate() {
            let buffer = index % 2;
            let mut acquired = Vec::with_capacity(self.shards.len());
            for shard in &mut self.shards {
                acquired.push(shard.experts.acquire(&mut shard.gpu, layer, wave)?);
                if let Some(next) = waves.get(index + 1) {
                    shard.experts.prefetch(&mut shard.gpu, layer, next, true)?;
                } else if !predicted.is_empty() {
                    // the next layer's predicted experts start their reads and copies behind this layer's
                    shard.experts.prefetch(&mut shard.gpu, layer + 1, &predicted, drive)?;
                }
            }
            self.mark("expert fills")?;
            // A route of another wave has no slot (a zero base: skipped).
            let bases = |slots: &[Buffer]| -> Vec<u64> {
                routes.iter().map(|&expert| wave.binary_search(&(expert as usize)).map_or(0, |at| slots[at].base() as u64)).collect()
            };
            for (shard, slots) in self.shards.iter_mut().zip(&acquired) {
                let work = shard.work.clone_handles();
                shard.gpu.experts(Some(&bases(slots)), slots[0].bytes, [0, gate_bytes, 2 * gate_bytes], kinds, w, ff, n as u64, k as u64,
                                  work.x.base(), 0, Some(work.weights.base()), hp.layers[layer].clamp_routed, true,
                                  work.products[buffer].base(), work.parts[1].base(), 0, Some(shard.rows.split(1)))?;
            }
            // Every device's share of the activations (of the shared expert's too, in the first wave).
            let rows = routed_rows + if index == 0 { n } else { 0 };
            let devices: Vec<&Backend> = self.shards.iter().map(|shard| &shard.gpu).collect();
            let outs: Vec<M> = self.shards.iter().map(|shard| shard.work.product.base()).collect();
            let parts: Vec<P> = self.shards.iter().map(|shard| shard.work.products[buffer].base() as P).collect();
            ops::sum_devices(&devices, &outs, &parts, rows as u64 * ff)?;
            for (shard, slots) in self.shards.iter_mut().zip(&acquired) {
                let work = shard.work.clone_handles();
                if index == 0 {
                    let shared = shard.shared[&layer];
                    shard.gpu.experts(Some(&vec![shared.buffer.base() as u64; n]), shared.buffer.bytes, shared.offsets, shared.kinds, w, ff,
                                      n as u64, 1, work.x.base(), 0, None, hp.layers[layer].clamp_shared, false,
                                      work.product.at(routed_rows * ff as usize), work.parts[1].base(), 0, Some(shard.rows.split(2)))?;
                }
                shard.gpu.experts(None, slots[0].bytes, [0, gate_bytes, 2 * gate_bytes], kinds, w, ff, n as u64, k as u64,
                                  work.x.base(), 0, None, hp.layers[layer].clamp_routed, true, work.product.base(), work.parts[1].base(),
                                  0, Some(shard.rows.split(2)))?;
                shard.experts.release(&mut shard.gpu)?;
            }
        }
        self.mark("routed experts")
    }
}

/// What a device's attention of a layer needs to know of the sequence: the compressed rows it
/// appends (kv source layers: ratio, rows waiting, first group) and those there were before.
struct AttentionPlan {
    layer: usize,
    n: usize,
    /// devices the heads are split across
    parts: usize,
    compressed: Option<super::Compressed>,
    appended: Option<(usize, usize, usize)>,
    existing: usize,
    ring_rows: u64,
}

/// Queue `ops` on a device from the graph recorded under `key`, recording it the first time.
fn recorded(shard: &mut Shard, key: u64, ops: impl FnOnce(&mut Shard) -> Result<()>) -> Result<()> {
    if shard.gpu.graph_run(key)? {
        return Ok(());
    }
    shard.gpu.graph_begin()?;
    let result = ops(shard);
    shard.gpu.graph_end(key, result.is_ok())?;
    result?;
    if shard.gpu.graph_run(key)? { Ok(()) } else { Err("a recorded graph went missing".into()) }
}

/// The hyper-connection weights of a sublayer and its input norm.
fn sublayer(shard: &Shard, layer: usize, side: &str, norm: &str) -> Result<(Dense, Dense, Dense, Dense)> {
    Ok((shard.dense(&name(layer, &format!("hc_{side}_fn")))?, shard.dense(&name(layer, &format!("hc_{side}_base")))?,
        shard.dense(&name(layer, &format!("hc_{side}_scale")))?, shard.dense(&name(layer, norm))?))
}

/// The hyper-connection step into a layer's attention; with `expand`, the previous layer's FFN
/// output (in `out`) is expanded into `stream` on the way in.
fn hc_attention(shard: &mut Shard, hp: &Hparams, layer: usize, n: usize, expand: bool, pre_previous: Option<usize>) -> Result<()> {
    let work = shard.work.clone_handles();
    let (fn_, base, scale, norm) = sublayer(shard, layer, "attn", "attn_norm")?;
    let pre = pre_previous.map_or(work.identity, |slot| work.pre_f[slot]);
    let expansion = expand.then(|| (work.out.base() as P, work.stream2.base() as P, work.post_f.base() as P, work.comb_f.base() as P));
    shard.gpu.hc_step(expansion, work.stream.base(), Some((fn_.buffer.base(), base.buffer.base(), scale.buffer.base(), work.pre_a.base(),
                      work.post_a.base(), work.comb_a.base())), hp.sinkhorn_iterations as u64, hp.hc_eps, hp.norm_eps, pre.base(),
                      norm.buffer.base(), n as u64, hp.hc as u64, hp.width as u64, &work.hc_workspace, work.x.base())
}

/// The hyper-connection step from a layer's attention (in `out`) into its FFN.
fn hc_ffn(shard: &mut Shard, hp: &Hparams, layer: usize, n: usize) -> Result<()> {
    let work = shard.work.clone_handles();
    let (fn_, base, scale, norm) = sublayer(shard, layer, "ffn", "ffn_norm")?;
    let expansion = (work.out.base() as P, work.stream.base() as P, work.post_a.base() as P, work.comb_a.base() as P);
    shard.gpu.hc_step(Some(expansion), work.stream2.base(), Some((fn_.buffer.base(), base.buffer.base(), scale.buffer.base(),
                      work.pre_f[layer % 2].base(), work.post_f.base(), work.comb_f.base())), hp.sinkhorn_iterations as u64, hp.hc_eps,
                      hp.norm_eps, work.pre_a.base(), norm.buffer.base(), n as u64, hp.hc as u64, hp.width as u64, &work.hc_workspace,
                      work.x.base())
}

/// A device's attention: q_a, kv, the compressed rows and the indexer (as every device), and its
/// heads of q, the attention, wo_a and wo_b into `target`.
fn attention_ops(shard: &mut Shard, hp: &Hparams, plan: &AttentionPlan, target: M) -> Result<()> {
    let (layer, n) = (plan.layer, plan.n);
    let (heads, dim, rope) = (hp.heads / plan.parts, hp.head_dim, hp.rope_dims as u64);
    let groups = hp.out_groups / plan.parts;
    let work = shard.work.clone_handles();
    let freqs = if plan.compressed.is_some() { shard.compressed_rope } else { shard.plain };
    let scratch = shard.scratch;
    let whole = |weight: &Dense, input: M, output: M| Linear {
        weights: weight.buffer, kind: weight.kind, rows: weight.dims[1] as u64, cols: weight.dims[0] as u64, first_row: 0,
        input: input as P, output,
    };
    let (q_a, kv) = (shard.dense(&name(layer, "attn_q_a"))?, shard.dense(&name(layer, "attn_kv"))?);
    shard.gpu.linears(&[whole(&q_a, work.x.base(), work.qr.base()), whole(&kv, work.x.base(), work.kv.base())], n as u64, 0, 0,
                      &scratch)?;
    shard.norm(&name(layer, "attn_q_a_norm"), work.qr.base(), n, hp.q_rank, hp.norm_eps)?;
    // q (this device's heads) stays unrotated: the attention rotates it as it reads it
    shard.linear(&name(layer, "attn_q_b"), work.qr.base(), n, work.q.base())?;
    let ring = shard.rings[&layer];
    let kv_norm = shard.dense(&name(layer, "attn_kv_a_norm"))?;
    shard.gpu.norm_rope(work.kv.base(), &kv_norm.buffer, n as u64, dim as u64, hp.norm_eps, Some((rope, &work.positions, &freqs)),
                        Some(work.ring_rows.base()), ring.base(), plan.ring_rows)?;
    let mut compressed = None;
    if let Some(c) = plan.compressed {
        let mut groups_now = plan.existing;
        if let Some((ratio, tail, first_group)) = plan.appended {
            groups_now += compress(shard, hp, layer, n, freqs, ratio, tail, first_group)?.0;
        }
        if c.index_source == layer && groups_now > 0 {
            let keys = shard.sources[&c.kv_source].keys;
            shard.linear(&name(layer, "indexer.attn_q_b"), work.qr.base(), n, work.index_q.base())?;
            shard.gpu.rope(work.index_q.base(), n as u64, hp.index_heads as u64, hp.index_dim as u64, rope, &work.positions, &freqs,
                           false)?;
            shard.linear(&name(layer, "indexer.proj"), work.x.base(), n, work.index_w.base())?;
            let scale = 1.0 / ((hp.index_dim as f32).sqrt() * (hp.index_heads as f32).sqrt());
            shard.gpu.index_scores(work.index_q.base(), keys.base(), work.index_w.base(), work.visible[c.ratio].base(), n as u64,
                                   groups_now as u64, hp.index_heads as u64, hp.index_dim as u64, scale, work.scores.base())?;
            let chosen = work.chosen[&layer];
            shard.gpu.top_k(work.scores.base(), n as u64, groups_now as u64, hp.index_top_k as u64, chosen.base())?;
        }
        if groups_now > 0 {
            compressed = Some((shard.sources[&c.kv_source].rows, groups_now, work.chosen[&c.index_source]));
        }
    }
    let sinks = shard.dense(&name(layer, "attn_sinks"))?;
    shard.gpu.sparse_attention(work.q.base(), ring.base(), plan.ring_rows, work.positions.base(), hp.window as u64,
                               compressed.map(|(rows, groups, _)| (rows.base() as P, groups as u64)),
                               compressed.map(|(_, _, chosen)| (chosen.base() as P, hp.index_top_k as u64)), sinks.buffer.base(),
                               n as u64, heads as u64, dim as u64, 1.0 / (dim as f32).sqrt(), Some((rope, &freqs)), Some(&scratch),
                               work.attended.base())?;
    // wo_a is block-diagonal: group g projects its heads' slice alone.
    let wo_a = shard.dense(&name(layer, "attn_output_a"))?;
    let group_width = hp.heads * dim / hp.out_groups;
    let projections: Vec<Linear> = (0..groups).map(|g| Linear {
        weights: wo_a.buffer, kind: wo_a.kind, rows: hp.out_rank as u64, cols: wo_a.dims[0] as u64, first_row: (g * hp.out_rank) as u64,
        input: work.attended.at(g * group_width) as P, output: work.low.at(g * hp.out_rank),
    }).collect();
    shard.gpu.linears(&projections, n as u64, (heads * dim) as u64, (groups * hp.out_rank) as u64, &scratch)?;
    shard.linear(&name(layer, "attn_output_b"), work.low.base(), n, target)
}

/// A layer's router on a device (every device routes alike); with `predict`, also the next
/// layer's router on this layer's input, its best PREDICTED experts after the routes in `ids`.
fn router_ops(shard: &mut Shard, hp: &Hparams, layer: usize, n: usize, predict: bool) -> Result<()> {
    let k = hp.experts_used;
    let gating = match hp.gating { Gating::Softmax => 0, Gating::Sigmoid => 1, Gating::SqrtSoftplus => 2 };
    let work = shard.work.clone_handles();
    shard.linear(&name(layer, "ffn_gate_inp"), work.x.base(), n, work.router.base())?;
    let bias = shard.dense(&format!("blk.{layer}.exp_probs_b.bias"))?;
    shard.gpu.route(work.router.base(), Some(bias.buffer.base()), n as u64, hp.experts as u64, k as u64, gating, hp.route_normalize,
                    hp.route_scale, work.ids.base(), work.weights.base())?;
    if predict {
        let next_bias = shard.dense(&format!("blk.{}.exp_probs_b.bias", layer + 1))?;
        shard.linear(&name(layer + 1, "ffn_gate_inp"), work.x.base(), 1, work.router.at(hp.experts))?;
        shard.gpu.route(work.router.at(hp.experts), Some(next_bias.buffer.base()), 1, hp.experts as u64, PREDICTED as u64, gating,
                        hp.route_normalize, hp.route_scale, work.ids.at(k), work.expert_weights.base())?;
    }
    Ok(())
}

/// Append this pass's complete groups of a kv source layer to a device's compressed rows and index
/// keys (from `first_group`, with `tail` rows of an incomplete group waiting); the rows of a still
/// incomplete group wait in its tail. The groups appended and the rows left over.
#[allow(clippy::too_many_arguments)]
fn compress(shard: &mut Shard, hp: &Hparams, layer: usize, n: usize, freqs: Buffer, ratio: usize, tail: usize, first_group: usize)
            -> Result<(usize, usize)> {
    let dim = hp.head_dim;
    let work = shard.work.clone_handles();
    // rows 1..=n take this pass; row 0 the waiting one, when there is one
    shard.linear(&name(layer, "attn_compressor_kv"), work.x.base(), n, work.pool_kv.at(dim))?;
    if ratio > 1 {
        shard.linear(&name(layer, "attn_compressor_gate"), work.x.base(), n, work.pool_gate.at(dim))?;
    }
    let (rows, keys, tail_kv, tail_gate) = {
        let source = &shard.sources[&layer];
        (source.rows, source.keys, source.tail_kv, source.tail_gate)
    };
    if tail > 0 {
        shard.gpu.gather(tail_kv.base(), 1, work.all_rows.base(), 1, dim as u64, work.pool_kv.base())?;
        shard.gpu.gather(tail_gate.base(), 1, work.all_rows.base(), 1, dim as u64, work.pool_gate.base())?;
    }
    let first_row = 1 - tail;
    let groups = (tail + n) / ratio;
    let left = (tail + n) % ratio;
    if groups > 0 {
        let target = rows.at(first_group * dim);
        shard.gpu.compress(work.pool_kv.at(first_row * dim), (ratio > 1).then(|| work.pool_gate.at(first_row * dim) as P),
                           groups as u64, ratio as u64, dim as u64, target)?;
        shard.norm(&name(layer, "attn_compressor_norm"), target, groups, dim, hp.norm_eps)?;
        let key_target = keys.at(first_group * hp.index_dim);
        let weight = shard.dense(&name(layer, "indexer.attn_k"))?;
        let scratch = shard.scratch;
        shard.gpu.linear(&weight.buffer, weight.kind, weight.dims[1] as u64, weight.dims[0] as u64, 0, target, 0, groups as u64,
                         key_target, 0, &scratch)?;
        let starts: Vec<i32> = (first_group..first_group + groups).map(|g| (g * ratio) as i32).collect();
        shard.gpu.write_buffer(&work.group_positions, 0, &starts)?;
        let key_norm = shard.dense(&name(layer, "indexer.k_norm"))?;
        shard.gpu.norm_rope(key_target, &key_norm.buffer, groups as u64, hp.index_dim as u64, hp.norm_eps,
                            Some((hp.rope_dims as u64, &work.group_positions, &freqs)), None, key_target, groups as u64)?;
        shard.gpu.rope(target, groups as u64, 1, dim as u64, hp.rope_dims as u64, &work.group_positions, &freqs, false)?;
    }
    if left > 0 {
        let last = first_row + groups * ratio;
        shard.gpu.gather(work.pool_kv.at(last * dim), 1, work.all_rows.base(), 1, dim as u64, tail_kv.base())?;
        shard.gpu.gather(work.pool_gate.at(last * dim), 1, work.all_rows.base(), 1, dim as u64, tail_gate.base())?;
    }
    Ok((groups, left))
}

/// Whether an engine over several devices divides this resident tensor among them (each holds its
/// part: of the head and the attention's q_b, sinks and output its heads' or groups', of a shared
/// expert its rows); every other resident tensor is whole on each device.
pub fn divided(name: &str) -> bool {
    ["attn_q_b", "attn_sinks", "attn_output_a", "attn_output_b", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"].iter()
        .any(|role| name.ends_with(&format!(".{role}.weight"))) || name == "output.weight"
}

/// One pass over engines that run consecutive layers (on their own devices): each hands the stream
/// to the next; the logits after the last token.
pub fn forward_split<T: Tensors>(engines: &mut [Engine<'_, T>], tokens: &[u32]) -> Result<Vec<f32>> {
    for i in 0..engines.len().saturating_sub(1) {
        engines[i].pass(tokens)?;
        let (done, rest) = engines.split_at_mut(i + 1);
        done[i].hand_over(&mut rest[0], tokens.len())?;
    }
    engines.last_mut().ok_or("no DeepSeek-V4.1 engines")?.forward(tokens)
}

impl Work {
    /// The buffers' handles (copies of device addresses) to use while `self` is borrowed.
    fn clone_handles(&self) -> Work {
        Work {
            stream: self.stream, stream2: self.stream2, pre_a: self.pre_a, post_a: self.post_a, comb_a: self.comb_a,
            pre_f: self.pre_f, post_f: self.post_f, comb_f: self.comb_f, identity: self.identity, x: self.x, qr: self.qr, q: self.q,
            kv: self.kv, attended: self.attended, low: self.low, out: self.out, parts: self.parts, products: self.products,
            product: self.product, positions: self.positions,
            ring_rows: self.ring_rows, visible: self.visible, pool_kv: self.pool_kv, pool_gate: self.pool_gate,
            group_positions: self.group_positions, index_q: self.index_q, index_w: self.index_w, scores: self.scores,
            chosen: self.chosen.clone(), router: self.router, ids: self.ids, weights: self.weights, all_rows: self.all_rows,
            expert_rows: self.expert_rows, expert_weights: self.expert_weights, xe: self.xe, gate: self.gate, up: self.up,
            hidden: self.hidden, y: self.y, engram_rows: self.engram_rows, engram_kv: self.engram_kv, head: self.head,
            hc_workspace: self.hc_workspace,
        }
    }
}
