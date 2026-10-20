//! DeepSeek-V4.1 on one GPU, composed from the generic ops: dense weights resident, routed
//! experts in the expert store's tiers (VRAM slots, a host tier, the files; src/experts.rs),
//! and the sequence state a forward pass continues (window rings, compressed rows and index
//! keys, compression tails, the Engram history).
//! Checked against the CPU reference by examples/ds41_gpu and examples/ds41_chat.
//!
//! A decode row's next-layer experts are predicted and prefetched from the host tier; for
//! measurements, DS41_NO_PREDICT turns that off and DS41_PREDICT_DRIVE also reads predicted
//! experts from the drive (so far no faster: wrong guesses spend the drive's time).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::{Gating, Hparams, Rope};
use crate::arch::Tensors;
use crate::experts::{DeviceExperts, Layout, Reader, Role};
use crate::formats;
use crate::gpu::{Backend, Result};
use crate::ops::{Buffer, Linear};

pub struct Settings {
    /// Longest sequence (positions) the state holds.
    pub context: usize,
    /// Most tokens one forward pass takes.
    pub batch: usize,
    /// Experts the VRAM cache holds.
    pub expert_slots: usize,
    /// Bytes of host tier (page-locked memory) for experts VRAM evicts.
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

/// An expert laid out as a slot: gate, up and down at byte offsets of one buffer.
#[derive(Clone, Copy)]
struct Slot {
    buffer: Buffer,
    offsets: [u64; 3],
    kinds: [u32; 3],
}

/// A kv source layer's compressed rows and index keys (one per group of `ratio` positions), and
/// the rows of its current incomplete group.
struct Source {
    ratio: usize,
    rows: Buffer,
    keys: Buffer,
    groups: usize,
    tail_kv: Buffer,
    tail_gate: Buffer,
    tail: usize,
}

/// Per-pass work buffers, sized for `batch` tokens.
struct Work {
    stream: Buffer,
    stream2: Buffer,
    mixes: Buffer,
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
    shared: Buffer,
    engram_rows: Buffer,
    engram_kv: Buffer,
    head: Buffer,
    /// hc_step's partial sums and counters.
    hc_workspace: Buffer,
}

/// An expert and the rows routed to it, with their weights.
type Routed = (usize, (Vec<i32>, Vec<f32>));

/// Experts acquired at a time by a pass of many rows (each wave's reads overlap the one before's
/// compute).
const WAVE: usize = 32;
/// Experts of the next layer prefetched from its router applied to this layer's input (decode):
/// the 8 best hold 79% of the 6 it takes.
const PREDICTED: usize = 8;
/// Reader threads per engine: reads in flight, enough for an expert's pieces at once.
const READERS: usize = 32;

pub struct Engine<'a, T: Tensors> {
    /// Shared, so a pass takes the parameters without copying the Engram tables' metadata.
    hp: std::rc::Rc<Hparams>,
    tensors: &'a T,
    gpu: Backend,
    dense: HashMap<String, Dense>,
    /// Per layer this engine runs: the shared expert, as a slot.
    shared: HashMap<usize, Slot>,
    scratch: Buffer,
    settings: Settings,
    position: usize,
    history: Vec<u32>,
    rings: BTreeMap<usize, Buffer>,
    ring: usize,
    sources: BTreeMap<usize, Source>,
    plain: Buffer,
    compressed_rope: Buffer,
    work: Work,
    experts: DeviceExperts,
    /// Next-layer experts predicted by this pass, prefetched after the layer's own.
    predicted: Vec<usize>,
    /// With DS41_PROFILE set: seconds per phase, the GPU synchronized at each phase's end.
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

/// Passes of up to this many rows run the routed experts over their slots (decode); longer ones
/// expert by expert.
const SLOT_ROWS: usize = 4;

fn name(layer: usize, tensor: &str) -> String {
    format!("blk.{layer}.{tensor}.weight")
}

impl<'a, T: Tensors> Engine<'a, T> {
    pub fn new(hp: Hparams, tensors: &'a T, device: usize, budget: u64, settings: Settings) -> Result<Engine<'a, T>> {
        let range = settings.layers.clone();
        if range.is_empty() || range.end > hp.layers.len() {
            return Err("DeepSeek-V4.1 engine layers outside the model".into());
        }
        for plan in &hp.layers[range.clone()] {
            if let Some(c) = plan.compressed.filter(|c| c.kv_source < range.start || c.index_source < range.start) {
                return Err(format!("DeepSeek-V4.1 layers {range:?} read layer {}'s compressed rows", c.kv_source.min(c.index_source)));
            }
        }
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
                sources.insert(layer, Source {
                    ratio: c.ratio,
                    rows: floats(&mut gpu, groups * hp.head_dim)?,
                    keys: floats(&mut gpu, groups * hp.index_dim)?,
                    groups: 0,
                    tail_kv: floats(&mut gpu, hp.head_dim)?,
                    tail_gate: floats(&mut gpu, hp.head_dim)?,
                    tail: 0,
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
            mixes: floats(&mut gpu, b * (2 + hc) * hc)?,
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
            router: floats(&mut gpu, b * hp.experts)?,
            ids: floats(&mut gpu, b * hp.experts_used)?,
            weights: floats(&mut gpu, b * hp.experts_used)?,
            all_rows: floats(&mut gpu, b)?,
            expert_rows: floats(&mut gpu, b)?,
            expert_weights: floats(&mut gpu, b)?,
            xe: floats(&mut gpu, b * w)?,
            gate: floats(&mut gpu, b * hp.expert_ff)?,
            up: floats(&mut gpu, b * hp.expert_ff)?,
            hidden: floats(&mut gpu, b * hp.experts_used.max(1) * hp.expert_ff)?,
            y: floats(&mut gpu, b * w)?,
            shared: floats(&mut gpu, b * w)?,
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
        // Each layer's experts as three roles of the files (a slot holds them back to back).
        let mut roles = vec![None; hp.layers.len()];
        for layer in range.clone() {
            let mut layer_roles = Vec::new();
            for role in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                let tensor_name = name(layer, role);
                let tensor = tensors.tensor(&tensor_name)?;
                let (file, base) = tensors.location(&tensor_name).ok_or("the expert store needs the model's files")?;
                let stride = (row_bytes(tensor.kind, tensor.dims[0])? * tensor.dims[1]) as u64;
                layer_roles.push(Role { file, base, stride });
            }
            roles[layer] = Some(layer_roles);
        }
        let layout = Arc::new(Layout::new(&tensors.files()?, roles, tensors.verifier())?);
        let experts = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(READERS)), settings.expert_slots,
                                         settings.host_expert_bytes)?;
        let mut engine = Engine {
            hp: std::rc::Rc::new(hp),
            tensors,
            gpu,
            dense: HashMap::new(),
            shared: HashMap::new(),
            scratch,
            settings,
            position: 0,
            history: Vec::new(),
            rings,
            ring,
            sources,
            plain: plain_buffer,
            compressed_rope: compressed_buffer,
            work,
            profile: std::env::var_os("DS41_PROFILE").map(|_| (std::time::Instant::now(), BTreeMap::new())),
            experts,
            predicted: Vec::new(),
        };
        engine.upload_dense()?;
        Ok(engine)
    }

    /// Every tensor of this engine's layers but the routed experts and the Engram tables, resident
    /// (with the head on the last engine); each shared expert as a slot.
    fn upload_dense(&mut self) -> Result<()> {
        let range = self.settings.layers.clone();
        let mut names = Vec::new();
        if range.end == self.hp.layers.len() {
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
            let buffer = self.gpu.upload_bytes(tensor.data)?;
            self.dense.insert(name, Dense { buffer, kind: tensor.kind, dims: tensor.dims });
        }
        for layer in range {
            let roles = ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"].map(|role| self.tensors.tensor(&name(layer, role)));
            let [gate, up, down] = roles;
            let (gate, up, down) = (gate?, up?, down?);
            let offsets = [0, gate.data.len() as u64, (gate.data.len() + up.data.len()) as u64];
            let bytes = [gate.data, up.data, down.data].concat();
            let buffer = self.gpu.upload_bytes(&bytes)?;
            self.shared.insert(layer, Slot { buffer, offsets, kinds: [gate.kind, up.kind, down.kind] });
        }
        Ok(())
    }

    /// Attribute the time since the last mark to `phase` (profiling only: waits for the GPU).
    fn mark(&mut self, phase: &'static str) -> Result<()> {
        if self.profile.is_none() {
            return Ok(());
        }
        let mut one = [0f32; 1];
        let x = self.work.x;
        self.gpu.read_buffer(&x, 0, &mut one)?;
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

    /// The expert store's counters.
    pub fn cache_counts(&self) -> crate::experts::Counts {
        self.experts.counts()
    }

    /// Wait for the expert store's reads and copies (before dropping the engine).
    pub fn settle(&mut self) -> Result<()> {
        self.experts.settle(&mut self.gpu)
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

    fn dense(&self, name: &str) -> Result<Dense> {
        self.dense.get(name).cloned().ok_or_else(|| format!("tensor {name} is not resident"))
    }

    /// out = W . input over `steps` rows (a whole resident 2D tensor).
    fn linear(&mut self, name: &str, input: *const std::ffi::c_void, steps: usize, out: *mut std::ffi::c_void) -> Result<()> {
        let weight = self.dense(name)?;
        let scratch = self.scratch;
        self.gpu.linear(&weight.buffer, weight.kind, weight.dims[1] as u64, weight.dims[0] as u64, 0, input, 0, steps as u64, out, 0,
                        &scratch)
    }

    fn norm(&mut self, name: &str, x: *mut std::ffi::c_void, rows: usize, width: usize) -> Result<()> {
        let weight = self.dense(name)?;
        self.gpu.rms_norm(x, &weight.buffer, rows as u64, width as u64, self.hp.norm_eps, x)
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
        let (hc, w) = (self.hp.hc, self.hp.width);
        let range = self.settings.layers.clone();
        if let Some((since, _)) = self.profile.as_mut() {
            // another engine's pass ran since this one's last mark
            *since = std::time::Instant::now();
        }
        let start = self.position;
        let positions: Vec<i32> = (start..start + n).map(|p| p as i32).collect();
        let ring_rows: Vec<i32> = (start..start + n).map(|p| (p % self.ring) as i32).collect();
        self.gpu.write_buffer(&self.work.positions, 0, &positions)?;
        self.gpu.write_buffer(&self.work.ring_rows, 0, &ring_rows)?;
        for ratio in [1usize, 2] {
            let visible: Vec<i32> = (start..start + n).map(|p| ((p + 1) / ratio) as i32).collect();
            self.gpu.write_buffer(&self.work.visible[ratio], 0, &visible)?;
        }
        if range.start == 0 {
            // embeddings repeated into the hc copies
            let embed = self.tensors.tensor("token_embd.weight")?;
            let bytes = row_bytes(embed.kind, embed.dims[0])?;
            let mut stream = Vec::with_capacity(n * hc * w);
            let mut row = Vec::new();
            for &token in tokens {
                row.clear();
                formats::decode_into(embed.kind, &embed.data[token as usize * bytes..(token as usize + 1) * bytes], &mut row)?;
                for _ in 0..hc {
                    stream.extend_from_slice(&row);
                }
            }
            self.gpu.write_buffer(&self.work.stream, 0, &stream)?;
        }
        if self.hp.layers[range.clone()].iter().any(|plan| plan.engram.is_some()) {
            self.prefetch_engram(tokens)?;
        }
        self.mark("embedding")?;
        // A handed-over stream comes expanded, with the collapse coefficients of the layer before.
        let mut pre_previous = range.start.checked_sub(1).map(|layer| layer % 2);
        for layer in range.clone() {
            self.layer(layer, tokens, n, pre_previous, layer > range.start)?;
            pre_previous = Some(layer % 2);
        }
        let work = self.work.clone_handles();
        let last_layer = range.end - 1;
        let logits = if range.end < self.hp.layers.len() {
            self.gpu.hc_expand(work.out.base(), work.stream2.base(), work.post_f.base(), work.comb_f.base(), n as u64, hc as u64,
                               w as u64, work.stream.base())?;
            None
        } else {
            // The last row's expansion of the last FFN output, collapsed and normed for the head.
            let (last, pre) = (n - 1, work.pre_f[last_layer % 2]);
            let norm = self.dense("output_norm.weight")?;
            let expand = (work.out.at(last * w) as *const _, work.stream2.at(last * hc * w) as *const _,
                          work.post_f.at(last * hc) as *const _, work.comb_f.at(last * hc * hc) as *const _);
            self.gpu.hc_step(Some(expand), work.stream.at(last * hc * w), None, 0, 0.0, self.hp.norm_eps, pre.at(last * hc),
                             norm.buffer.base(), 1, hc as u64, w as u64, &work.hc_workspace, work.x.base())?;
            self.linear("output.weight", work.x.base(), 1, work.head.base())?;
            let mut logits = vec![0f32; self.hp.vocabulary];
            self.gpu.read_buffer(&work.head, 0, &mut logits)?;
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
        let (stream, coefficients) = (self.work.stream, self.work.pre_f[pre]);
        self.gpu.copy_to(stream.base(), &mut next.gpu, next.work.stream.base(), (rows * hc * w * 4) as u64)?;
        self.gpu.copy_to(coefficients.base(), &mut next.gpu, next.work.pre_f[pre].base(), (rows * hc * 4) as u64)
    }

    /// One layer; with `expand`, the previous layer's FFN output (in `out`) is expanded into `stream`
    /// on the way in.
    fn layer(&mut self, layer: usize, tokens: &[u32], n: usize, pre_previous: Option<usize>, expand: bool) -> Result<()> {
        let hp = self.hp.clone();
        let (hc, w, rows) = (hp.hc as u64, hp.width as u64, n as u64);
        let plan = &hp.layers[layer];
        let work = self.work.clone_handles();
        let mut expand = expand.then(|| (work.out.base() as *const _, work.stream2.base() as *const _,
                                         work.post_f.base() as *const _, work.comb_f.base() as *const _));
        if let Some(module) = plan.engram {
            if expand.take().is_some() {
                self.gpu.hc_expand(work.out.base(), work.stream2.base(), work.post_f.base(), work.comb_f.base(), rows, hc, w,
                                   work.stream.base())?;
            }
            self.engram(layer, module, tokens, n)?;
            self.mark("engram")?;
        }
        let iterations = hp.sinkhorn_iterations as u64;
        let sublayer = |engine: &Self, side: &str, norm: &str| -> Result<(Dense, Dense, Dense, Dense)> {
            Ok((engine.dense(&name(layer, &format!("hc_{side}_fn")))?, engine.dense(&name(layer, &format!("hc_{side}_base")))?,
                engine.dense(&name(layer, &format!("hc_{side}_scale")))?, engine.dense(&name(layer, norm))?))
        };
        let (fn_, base, scale, norm) = sublayer(self, "attn", "attn_norm")?;
        let pre = pre_previous.map_or(work.identity, |slot| work.pre_f[slot]);
        self.gpu.hc_step(expand, work.stream.base(), Some((fn_.buffer.base(), base.buffer.base(), scale.buffer.base(), work.pre_a.base(),
                         work.post_a.base(), work.comb_a.base())), iterations, hp.hc_eps, hp.norm_eps, pre.base(), norm.buffer.base(),
                         rows, hc, w, &work.hc_workspace, work.x.base())?;
        self.mark("hyper-connections")?;
        self.attention(layer, n)?;
        self.mark("attention")?;
        let (fn_, base, scale, norm) = sublayer(self, "ffn", "ffn_norm")?;
        let expand = (work.out.base() as *const _, work.stream.base() as *const _, work.post_a.base() as *const _,
                      work.comb_a.base() as *const _);
        self.gpu.hc_step(Some(expand), work.stream2.base(), Some((fn_.buffer.base(), base.buffer.base(), scale.buffer.base(),
                         work.pre_f[layer % 2].base(), work.post_f.base(), work.comb_f.base())), iterations, hp.hc_eps, hp.norm_eps,
                         work.pre_a.base(), norm.buffer.base(), rows, hc, w, &work.hc_workspace, work.x.base())?;
        self.mark("hyper-connections")?;
        self.moe(layer, n)
    }

    /// Start reading every Engram row these tokens will gather, while the layers before
    /// the modules run.
    fn prefetch_engram(&self, tokens: &[u32]) -> Result<()> {
        let Some(engram) = self.hp.engram.as_ref() else { return Ok(()) };
        let table = self.tensors.tensor("per_layer_token_embd.weight")?;
        let bytes = row_bytes(table.kind, table.dims[0])?;
        let mut history = self.history[self.history.len().saturating_sub(engram.ngram - 1)..].to_vec();
        let (mut rows, mut ranges) = (Vec::new(), Vec::new());
        for (i, &token) in tokens.iter().enumerate() {
            let before = history[history.len().saturating_sub(engram.ngram - 1)..].to_vec();
            for module in 0..engram.multipliers.len() / engram.ngram {
                rows.clear();
                engram.rows(module, token, &before, self.position + i, &mut rows);
                ranges.extend(rows.iter().map(|&row| &table.data[row as usize * bytes..(row as usize + 1) * bytes]));
            }
            history.push(token);
        }
        self.tensors.prefetch(&ranges);
        Ok(())
    }

    fn engram(&mut self, layer: usize, module: usize, tokens: &[u32], n: usize) -> Result<()> {
        let engram = self.hp.engram.clone().ok_or("no Engram metadata")?;
        let table = self.tensors.tensor("per_layer_token_embd.weight")?;
        let bytes = row_bytes(table.kind, table.dims[0])?;
        let mut history = self.history[self.history.len().saturating_sub(engram.ngram - 1)..].to_vec();
        let (mut rows, mut gathered) = (Vec::new(), Vec::with_capacity(n * self.hp.engram_columns() * engram.head_dim));
        for (i, &token) in tokens.iter().enumerate() {
            rows.clear();
            let before = &history[history.len().saturating_sub(engram.ngram - 1)..];
            engram.rows(module, token, before, self.position + i, &mut rows);
            for &row in &rows {
                formats::decode_into(table.kind, &table.data[row as usize * bytes..(row as usize + 1) * bytes], &mut gathered)?;
            }
            history.push(token);
        }
        let work = self.work.clone_handles();
        self.gpu.write_buffer(&work.engram_rows, 0, &gathered)?;
        self.linear(&name(layer, "engram_kv"), work.engram_rows.base(), n, work.engram_kv.base())?;
        let gate = self.dense(&name(layer, "engram_gate"))?;
        self.gpu.engram_gate(work.stream.base(), work.engram_kv.base(), gate.buffer.base(), n as u64, self.hp.hc as u64,
                             self.hp.width as u64, self.hp.norm_eps)
    }

    fn attention(&mut self, layer: usize, n: usize) -> Result<()> {
        let hp = self.hp.clone();
        let (heads, dim, rope) = (hp.heads, hp.head_dim, hp.rope_dims as u64);
        let work = self.work.clone_handles();
        let plan = hp.layers[layer].compressed;
        let freqs = if plan.is_some() { self.compressed_rope } else { self.plain };
        let scratch = self.scratch;
        let whole = |weight: &Dense, input: *mut std::ffi::c_void, output: *mut std::ffi::c_void| Linear {
            weights: weight.buffer, kind: weight.kind, rows: weight.dims[1] as u64, cols: weight.dims[0] as u64, first_row: 0,
            input: input as *const _, output,
        };
        let (q_a, kv) = (self.dense(&name(layer, "attn_q_a"))?, self.dense(&name(layer, "attn_kv"))?);
        self.gpu.linears(&[whole(&q_a, work.x.base(), work.qr.base()), whole(&kv, work.x.base(), work.kv.base())], n as u64, 0, 0,
                         &scratch)?;
        self.norm(&name(layer, "attn_q_a_norm"), work.qr.base(), n, hp.q_rank)?;
        // q stays unrotated: the attention rotates it as it reads it
        self.linear(&name(layer, "attn_q_b"), work.qr.base(), n, work.q.base())?;
        let ring = self.rings[&layer];
        let kv_norm = self.dense(&name(layer, "attn_kv_a_norm"))?;
        self.gpu.norm_rope(work.kv.base(), &kv_norm.buffer, n as u64, dim as u64, hp.norm_eps, Some((rope, &work.positions, &freqs)),
                           Some(work.ring_rows.base()), ring.base(), self.ring as u64)?;
        let mut compressed = None;
        if let Some(c) = plan {
            if c.kv_source == layer {
                self.compress(layer, n, freqs)?;
            }
            let groups = self.sources[&c.kv_source].groups;
            if c.index_source == layer && groups > 0 {
                let keys = self.sources[&c.kv_source].keys;
                self.linear(&name(layer, "indexer.attn_q_b"), work.qr.base(), n, work.index_q.base())?;
                self.gpu.rope(work.index_q.base(), n as u64, hp.index_heads as u64, hp.index_dim as u64, rope, &work.positions, &freqs,
                              false)?;
                self.linear(&name(layer, "indexer.proj"), work.x.base(), n, work.index_w.base())?;
                let scale = 1.0 / ((hp.index_dim as f32).sqrt() * (hp.index_heads as f32).sqrt());
                self.gpu.index_scores(work.index_q.base(), keys.base(), work.index_w.base(), work.visible[c.ratio].base(), n as u64,
                                      groups as u64, hp.index_heads as u64, hp.index_dim as u64, scale, work.scores.base())?;
                let chosen = work.chosen[&layer];
                self.gpu.top_k(work.scores.base(), n as u64, groups as u64, hp.index_top_k as u64, chosen.base())?;
            }
            if groups > 0 {
                compressed = Some((self.sources[&c.kv_source].rows, groups, work.chosen[&c.index_source]));
            }
        }
        let sinks = self.dense(&name(layer, "attn_sinks"))?;
        self.gpu.sparse_attention(work.q.base(), ring.base(), self.ring as u64, work.positions.base(), hp.window as u64,
                                  compressed.map(|(rows, groups, _)| (rows.base() as *const _, groups as u64)),
                                  compressed.map(|(_, _, chosen)| (chosen.base() as *const _, hp.index_top_k as u64)),
                                  sinks.buffer.base(), n as u64, heads as u64, dim as u64, 1.0 / (dim as f32).sqrt(),
                                  Some((rope, &freqs)), Some(&self.scratch), work.attended.base())?;
        // wo_a is block-diagonal: group g projects its heads' slice alone.
        let wo_a = self.dense(&name(layer, "attn_output_a"))?;
        let group_width = heads * dim / hp.out_groups;
        let groups: Vec<Linear> = (0..hp.out_groups).map(|g| Linear {
            weights: wo_a.buffer, kind: wo_a.kind, rows: hp.out_rank as u64, cols: wo_a.dims[0] as u64, first_row: (g * hp.out_rank) as u64,
            input: work.attended.at(g * group_width) as *const _, output: work.low.at(g * hp.out_rank),
        }).collect();
        self.gpu.linears(&groups, n as u64, (heads * dim) as u64, (hp.out_groups * hp.out_rank) as u64, &scratch)?;
        self.linear(&name(layer, "attn_output_b"), work.low.base(), n, work.out.base())
    }

    /// Append this pass's complete groups to the source's compressed rows and index keys; the
    /// rows of a still incomplete group wait in the tail.
    fn compress(&mut self, layer: usize, n: usize, freqs: Buffer) -> Result<()> {
        let hp = self.hp.clone();
        let dim = hp.head_dim;
        let work = self.work.clone_handles();
        let (ratio, tail, first_group) = {
            let source = &self.sources[&layer];
            (source.ratio, source.tail, source.groups)
        };
        // rows 1..=n take this pass; row 0 the waiting one, when there is one
        self.linear(&name(layer, "attn_compressor_kv"), work.x.base(), n, work.pool_kv.at(dim))?;
        if ratio > 1 {
            self.linear(&name(layer, "attn_compressor_gate"), work.x.base(), n, work.pool_gate.at(dim))?;
        }
        if tail > 0 {
            let source = &self.sources[&layer];
            let (tail_kv, tail_gate) = (source.tail_kv, source.tail_gate);
            self.gpu.gather(tail_kv.base(), 1, work.all_rows.base(), 1, dim as u64, work.pool_kv.base())?;
            self.gpu.gather(tail_gate.base(), 1, work.all_rows.base(), 1, dim as u64, work.pool_gate.base())?;
        }
        let first_row = 1 - tail;
        let groups = (tail + n) / ratio;
        let left = (tail + n) % ratio;
        if groups > 0 {
            let source = &self.sources[&layer];
            let (rows, keys) = (source.rows, source.keys);
            let target = rows.at(first_group * dim);
            self.gpu.compress(work.pool_kv.at(first_row * dim), (ratio > 1).then(|| work.pool_gate.at(first_row * dim) as *const _),
                              groups as u64, ratio as u64, dim as u64, target)?;
            self.norm(&name(layer, "attn_compressor_norm"), target, groups, dim)?;
            let key_target = keys.at(first_group * hp.index_dim);
            let weight = self.dense(&name(layer, "indexer.attn_k"))?;
            let scratch = self.scratch;
            self.gpu.linear(&weight.buffer, weight.kind, weight.dims[1] as u64, weight.dims[0] as u64, 0, target, 0, groups as u64,
                            key_target, 0, &scratch)?;
            let starts: Vec<i32> = (first_group..first_group + groups).map(|g| (g * ratio) as i32).collect();
            self.gpu.write_buffer(&work.group_positions, 0, &starts)?;
            let key_norm = self.dense(&name(layer, "indexer.k_norm"))?;
            self.gpu.norm_rope(key_target, &key_norm.buffer, groups as u64, hp.index_dim as u64, hp.norm_eps,
                               Some((hp.rope_dims as u64, &work.group_positions, &freqs)), None, key_target, groups as u64)?;
            self.gpu.rope(target, groups as u64, 1, dim as u64, hp.rope_dims as u64, &work.group_positions, &freqs, false)?;
        }
        if left > 0 {
            let last = first_row + groups * ratio;
            let source = &self.sources[&layer];
            let (tail_kv, tail_gate) = (source.tail_kv, source.tail_gate);
            self.gpu.gather(work.pool_kv.at(last * dim), 1, work.all_rows.base(), 1, dim as u64, tail_kv.base())?;
            self.gpu.gather(work.pool_gate.at(last * dim), 1, work.all_rows.base(), 1, dim as u64, tail_gate.base())?;
        }
        let source = self.sources.get_mut(&layer).unwrap();
        source.groups += groups;
        source.tail = left;
        Ok(())
    }

    fn moe(&mut self, layer: usize, n: usize) -> Result<()> {
        let hp = self.hp.clone();
        let (w, ff, k) = (hp.width, hp.expert_ff, hp.experts_used);
        let work = self.work.clone_handles();
        self.linear(&name(layer, "ffn_gate_inp"), work.x.base(), n, work.router.base())?;
        let bias = self.dense(&format!("blk.{layer}.exp_probs_b.bias"))?;
        let gating = match hp.gating { Gating::Softmax => 0, Gating::Sigmoid => 1, Gating::SqrtSoftplus => 2 };
        self.gpu.route(work.router.base(), Some(bias.buffer.base()), n as u64, hp.experts as u64, k as u64, gating,
                       hp.route_normalize, hp.route_scale, work.ids.base(), work.weights.base())?;
        // One decode row: the next layer's router applied to this layer's input names the experts
        // to prefetch (their reads and copies overlap this layer's experts and the next attention).
        let predict = n == 1 && self.settings.layers.contains(&(layer + 1)) && std::env::var_os("DS41_NO_PREDICT").is_none();
        if predict {
            let next_bias = self.dense(&format!("blk.{}.exp_probs_b.bias", layer + 1))?;
            self.linear(&name(layer + 1, "ffn_gate_inp"), work.x.base(), 1, work.router.at(hp.experts))?;
            self.gpu.route(work.router.at(hp.experts), Some(next_bias.buffer.base()), 1, hp.experts as u64, PREDICTED as u64,
                           gating, hp.route_normalize, hp.route_scale, work.ids.at(k), work.expert_weights.base())?;
        }
        self.mark("router")?;
        // The routes travel to the host while the shared expert writes `out`; the routed ones add to it.
        let read = n * k + if predict { PREDICTED } else { 0 };
        self.gpu.read_start(&work.ids, 0, (read * 4) as u64)?;
        let shared = self.shared[&layer];
        if n <= SLOT_ROWS {
            self.gpu.experts(&vec![shared.buffer.base() as u64; n], shared.buffer.bytes, shared.offsets, shared.kinds, w as u64,
                             ff as u64, n as u64, 1, work.x.base(), 0, None, hp.layers[layer].clamp_shared, false,
                             work.hidden.base(), work.out.base(), 0)?;
        } else {
            let scratch = self.scratch;
            let role = |r: usize, bytes: u64| shared.buffer.slice(shared.offsets[r], bytes);
            let (gate, up) = (role(0, shared.offsets[1]), role(1, shared.offsets[2] - shared.offsets[1]));
            let down = role(2, shared.buffer.bytes - shared.offsets[2]);
            self.gpu.linear(&gate, shared.kinds[0], ff as u64, w as u64, 0, work.x.base(), 0, n as u64, work.gate.base(), 0, &scratch)?;
            self.gpu.linear(&up, shared.kinds[1], ff as u64, w as u64, 0, work.x.base(), 0, n as u64, work.up.base(), 0, &scratch)?;
            self.gpu.swiglu(work.gate.base(), work.up.base(), n as u64, ff as u64, hp.layers[layer].clamp_shared, None,
                            work.hidden.base())?;
            self.gpu.linear(&down, shared.kinds[2], w as u64, ff as u64, 0, work.hidden.base(), 0, n as u64, work.out.base(), 0,
                            &scratch)?;
        }
        self.mark("shared expert")?;
        let mut routes = vec![0i32; read];
        self.gpu.read_finish(&mut routes)?;
        self.predicted = routes.split_off(n * k).into_iter().map(|expert| expert as usize).collect();
        if n <= SLOT_ROWS {
            return self.routed(layer, n, &routes);
        }
        let mut weights = vec![0f32; n * k];
        self.gpu.read_buffer(&work.weights, 0, &mut weights)?;
        let mut by_expert: BTreeMap<usize, (Vec<i32>, Vec<f32>)> = BTreeMap::new();
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
        let scratch = self.scratch;
        // In waves: the next wave's experts load while this one's compute.
        let by_expert: Vec<Routed> = by_expert.into_iter().collect();
        let waves: Vec<&[Routed]> = by_expert.chunks(WAVE).collect();
        for (index, wave) in waves.iter().enumerate() {
            let experts: Vec<usize> = wave.iter().map(|(expert, _)| *expert).collect();
            let slots = self.experts.acquire(&mut self.gpu, layer, &experts)?;
            if let Some(next) = waves.get(index + 1) {
                let experts: Vec<usize> = next.iter().map(|(expert, _)| *expert).collect();
                self.experts.prefetch(&mut self.gpu, layer, &experts, true)?;
            }
            self.mark("expert fills")?;
            for ((_, (rows, user_weights)), slot) in wave.iter().zip(slots) {
                let count = rows.len();
                self.gpu.write_buffer(&work.expert_rows, 0, rows)?;
                self.gpu.write_buffer(&work.expert_weights, 0, user_weights)?;
                self.gpu.gather(work.x.base(), n as u64, work.expert_rows.base(), count as u64, w as u64, work.xe.base())?;
                let gate_weights = slot.slice(0, gate_bytes as u64);
                let up_weights = slot.slice(gate_bytes as u64, gate_bytes as u64);
                let down_weights = slot.slice(2 * gate_bytes as u64, down_bytes as u64);
                self.gpu.linear(&gate_weights, gate_kind, ff as u64, w as u64, 0, work.xe.base(), 0, count as u64, work.gate.base(), 0,
                                &scratch)?;
                self.gpu.linear(&up_weights, up_kind, ff as u64, w as u64, 0, work.xe.base(), 0, count as u64, work.up.base(), 0,
                                &scratch)?;
                self.gpu.swiglu(work.gate.base(), work.up.base(), count as u64, ff as u64, hp.layers[layer].clamp_routed,
                                Some(work.expert_weights.base()), work.hidden.base())?;
                self.gpu.linear(&down_weights, down_kind, w as u64, ff as u64, 0, work.hidden.base(), 0, count as u64, work.y.base(),
                                0, &scratch)?;
                self.gpu.scatter_add(work.y.base(), work.expert_rows.base(), count as u64, w as u64, work.out.base(), n as u64)?;
            }
            self.experts.release(&mut self.gpu)?;
        }
        self.mark("routed experts")
    }

    /// The routed experts of a few rows (decode): every route's expert made resident, then both
    /// stages over the slots in one launch each, added to `out`.
    fn routed(&mut self, layer: usize, n: usize, routes: &[i32]) -> Result<()> {
        let (w, ff) = (self.hp.width, self.hp.expert_ff);
        let work = self.work.clone_handles();
        let gate = self.tensors.tensor(&name(layer, "ffn_gate_exps"))?;
        let up = self.tensors.tensor(&name(layer, "ffn_up_exps"))?;
        let down = self.tensors.tensor(&name(layer, "ffn_down_exps"))?;
        let gate_bytes = ff * row_bytes(gate.kind, w)?;
        let kinds = [gate.kind, up.kind, down.kind];
        let experts: Vec<usize> = routes.iter().map(|&expert| expert as usize).collect::<std::collections::BTreeSet<_>>()
            .into_iter().collect();
        let slots = self.experts.acquire(&mut self.gpu, layer, &experts)?;
        // The next layer's predicted experts start their reads and copies behind this layer's.
        let predicted = std::mem::take(&mut self.predicted);
        if !predicted.is_empty() {
            let drive = std::env::var_os("DS41_PREDICT_DRIVE").is_some();
            self.experts.prefetch(&mut self.gpu, layer + 1, &predicted, drive)?;
        }
        self.mark("expert fills")?;
        let bases: Vec<u64> = routes.iter()
            .map(|expert| slots[experts.binary_search(&(*expert as usize)).unwrap()].base() as u64).collect();
        let slot_bytes = slots.first().map_or(0, |slot| slot.bytes);
        let offsets = [0, gate_bytes as u64, 2 * gate_bytes as u64];
        self.gpu.experts(&bases, slot_bytes, offsets, kinds, w as u64, ff as u64, n as u64, self.hp.experts_used as u64,
                         work.x.base(), 0, Some(work.weights.base()), self.hp.layers[layer].clamp_routed, true, work.hidden.base(),
                         work.out.base(), 0)?;
        self.experts.release(&mut self.gpu)?;
        self.mark("routed experts")
    }
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
            stream: self.stream, stream2: self.stream2, mixes: self.mixes, pre_a: self.pre_a, post_a: self.post_a, comb_a: self.comb_a,
            pre_f: self.pre_f, post_f: self.post_f, comb_f: self.comb_f, identity: self.identity, x: self.x, qr: self.qr, q: self.q,
            kv: self.kv, attended: self.attended, low: self.low, out: self.out, positions: self.positions, ring_rows: self.ring_rows,
            visible: self.visible, pool_kv: self.pool_kv, pool_gate: self.pool_gate, group_positions: self.group_positions,
            index_q: self.index_q, index_w: self.index_w, scores: self.scores, chosen: self.chosen.clone(), router: self.router,
            ids: self.ids, weights: self.weights, all_rows: self.all_rows, expert_rows: self.expert_rows,
            expert_weights: self.expert_weights, xe: self.xe, gate: self.gate, up: self.up, hidden: self.hidden, y: self.y,
            shared: self.shared, engram_rows: self.engram_rows, engram_kv: self.engram_kv, head: self.head,
            hc_workspace: self.hc_workspace,
        }
    }
}
