//! DeepSeek-V4.1 on one GPU, composed from the generic ops: dense weights resident, routed
//! experts in a VRAM cache refilled by use, and the sequence state a forward pass continues
//! (window rings, compressed rows and index keys, compression tails, the Engram history).
//! Checked against the CPU reference by examples/ds41_gpu and examples/ds41_chat.

use std::collections::{BTreeMap, HashMap};

use super::{Gating, Hparams, Rope};
use crate::arch::Tensors;
use crate::formats;
use crate::gpu::{Backend, Result};
use crate::ops::Buffer;

pub struct Settings {
    /// Longest sequence (positions) the state holds.
    pub context: usize,
    /// Most tokens one forward pass takes.
    pub batch: usize,
    /// Experts the VRAM cache holds.
    pub expert_slots: usize,
    pub scratch_bytes: u64,
}

#[derive(Clone)]
struct Dense {
    buffer: Buffer,
    kind: u32,
    dims: Vec<usize>,
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
}

/// Routed experts in VRAM slots ([gate | up | down] bytes each), evicting the least recently used.
struct Experts {
    slots: Vec<Buffer>,
    owner: Vec<Option<(usize, usize)>>,
    used: Vec<u64>,
    map: HashMap<(usize, usize), usize>,
    clock: u64,
    hits: u64,
    misses: u64,
    /// Seconds spent copying missed experts in.
    fill_seconds: f64,
}

pub struct Engine<'a, T: Tensors> {
    hp: Hparams,
    tensors: &'a T,
    gpu: Backend,
    dense: HashMap<String, Dense>,
    scratch: Buffer,
    settings: Settings,
    position: usize,
    history: Vec<u32>,
    rings: Vec<Buffer>,
    ring: usize,
    sources: BTreeMap<usize, Source>,
    plain: Buffer,
    compressed_rope: Buffer,
    work: Work,
    experts: Experts,
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

impl<'a, T: Tensors> Engine<'a, T> {
    pub fn new(hp: Hparams, tensors: &'a T, device: usize, budget: u64, settings: Settings) -> Result<Engine<'a, T>> {
        let mut gpu = Backend::new(device, budget)?;
        let scratch = gpu.buffer(settings.scratch_bytes)?;
        let (b, hc, w) = (settings.batch, hp.hc, hp.width);
        let ring = hp.window + b;
        let floats = |gpu: &mut Backend, count: usize| gpu.buffer((count.max(1) * 4) as u64);
        let rings = (0..hp.layers.len()).map(|_| floats(&mut gpu, ring * hp.head_dim)).collect::<Result<Vec<_>>>()?;
        let mut sources = BTreeMap::new();
        for (layer, plan) in hp.layers.iter().enumerate() {
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
        for (layer, plan) in hp.layers.iter().enumerate() {
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
            hidden: floats(&mut gpu, b * hp.expert_ff)?,
            y: floats(&mut gpu, b * w)?,
            shared: floats(&mut gpu, b * w)?,
            engram_rows: floats(&mut gpu, b * hp.engram_columns() * hp.engram.as_ref().map_or(1, |e| e.head_dim))?,
            engram_kv: floats(&mut gpu, b * (hc + 1) * w)?,
            head: floats(&mut gpu, hp.vocabulary)?,
        };
        gpu.write_buffer(&work.identity, 0, &identity)?;
        gpu.write_buffer(&work.all_rows, 0, &all_rows)?;
        let plain = frequencies(hp.rope_dims, &hp.rope);
        let compressed = frequencies(hp.rope_dims, &hp.compressed_rope);
        let plain_buffer = floats(&mut gpu, plain.len())?;
        gpu.write_buffer(&plain_buffer, 0, &plain)?;
        let compressed_buffer = floats(&mut gpu, compressed.len())?;
        gpu.write_buffer(&compressed_buffer, 0, &compressed)?;
        // One slot fits the largest layer's expert.
        let mut slot_bytes = 0;
        for layer in 0..hp.layers.len() {
            let mut bytes = 0;
            for role in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                let tensor = tensors.tensor(&name(layer, role))?;
                bytes += row_bytes(tensor.kind, tensor.dims[0])? * tensor.dims[1];
            }
            slot_bytes = slot_bytes.max(bytes);
        }
        let slots = (0..settings.expert_slots).map(|_| gpu.buffer(slot_bytes as u64)).collect::<Result<Vec<_>>>()?;
        let count = slots.len();
        let mut engine = Engine {
            hp,
            tensors,
            gpu,
            dense: HashMap::new(),
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
            experts: Experts { slots, owner: vec![None; count], used: vec![0; count], map: HashMap::new(), clock: 0, hits: 0, misses: 0, fill_seconds: 0.0 },
        };
        engine.upload_dense()?;
        Ok(engine)
    }

    /// Every tensor but the routed experts and the Engram tables, resident.
    fn upload_dense(&mut self) -> Result<()> {
        let mut names = vec!["output.weight".to_string(), "output_norm.weight".to_string()];
        for layer in 0..self.hp.layers.len() {
            for tensor in ["attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv", "attn_kv_a_norm", "attn_sinks",
                           "attn_output_a", "attn_output_b", "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn",
                           "hc_ffn_base", "hc_ffn_scale", "ffn_norm", "ffn_gate_inp", "ffn_gate_shexp", "ffn_up_shexp",
                           "ffn_down_shexp", "attn_compressor_kv", "attn_compressor_gate", "attn_compressor_norm",
                           "indexer.attn_k", "indexer.k_norm", "indexer.attn_q_b", "indexer.proj", "engram_kv", "engram_gate"] {
                names.push(name(layer, tensor));
            }
            names.push(format!("blk.{layer}.exp_probs_b.bias"));
        }
        for name in names {
            let Ok(tensor) = self.tensors.tensor(&name) else { continue };
            let buffer = self.gpu.upload_bytes(tensor.data)?;
            self.dense.insert(name, Dense { buffer, kind: tensor.kind, dims: tensor.dims });
        }
        Ok(())
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// The expert cache's hits, misses and seconds spent filling missed slots.
    pub fn cache_counts(&self) -> (u64, u64, f64) {
        (self.experts.hits, self.experts.misses, self.experts.fill_seconds)
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

    /// Advance the sequence over `tokens` (at most the batch); the logits after the last one.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let n = tokens.len();
        if n == 0 || n > self.settings.batch || self.position + n > self.settings.context {
            return Err("DeepSeek-V4.1 forward exceeds the batch or the context".into());
        }
        let (hc, w) = (self.hp.hc, self.hp.width);
        let start = self.position;
        let positions: Vec<i32> = (start..start + n).map(|p| p as i32).collect();
        let ring_rows: Vec<i32> = (start..start + n).map(|p| (p % self.ring) as i32).collect();
        self.gpu.write_buffer(&self.work.positions, 0, &positions)?;
        self.gpu.write_buffer(&self.work.ring_rows, 0, &ring_rows)?;
        for ratio in [1usize, 2] {
            let visible: Vec<i32> = (start..start + n).map(|p| ((p + 1) / ratio) as i32).collect();
            self.gpu.write_buffer(&self.work.visible[ratio], 0, &visible)?;
        }
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
        let mut pre_previous: Option<usize> = None;
        for layer in 0..self.hp.layers.len() {
            self.layer(layer, tokens, n, pre_previous)?;
            pre_previous = Some(layer % 2);
        }
        let work = &self.work;
        let pre = work.pre_f[(self.hp.layers.len() - 1) % 2];
        self.gpu.hc_collapse(work.stream.at((n - 1) * hc * w), pre.at((n - 1) * hc), 1, hc as u64, w as u64, work.x.base())?;
        let x = self.work.x.base();
        self.norm("output_norm.weight", x, 1, w)?;
        let head = self.work.head.base();
        self.linear("output.weight", x, 1, head)?;
        let mut logits = vec![0f32; self.hp.vocabulary];
        let buffer = self.work.head;
        self.gpu.read_buffer(&buffer, 0, &mut logits)?;
        self.history.extend_from_slice(tokens);
        self.position += n;
        Ok(logits)
    }

    fn layer(&mut self, layer: usize, tokens: &[u32], n: usize, pre_previous: Option<usize>) -> Result<()> {
        let hp = self.hp.clone();
        let (hc, w) = (hp.hc, hp.width);
        let plan = hp.layers[layer].clone();
        let work = self.work.clone_handles();
        if let Some(module) = plan.engram {
            self.engram(layer, module, tokens, n)?;
        }
        self.linear(&name(layer, "hc_attn_fn"), work.stream.base(), n, work.mixes.base())?;
        let (base, scale) = (self.dense(&name(layer, "hc_attn_base"))?, self.dense(&name(layer, "hc_attn_scale"))?);
        self.gpu.hc_coefficients(work.mixes.base(), work.stream.base(), n as u64, hc as u64, w as u64, base.buffer.base(),
                                 scale.buffer.base(), hp.sinkhorn_iterations as u64, hp.hc_eps, hp.norm_eps, work.pre_a.base(),
                                 work.post_a.base(), work.comb_a.base())?;
        let pre = pre_previous.map_or(work.identity, |slot| work.pre_f[slot]);
        self.gpu.hc_collapse(work.stream.base(), pre.base(), n as u64, hc as u64, w as u64, work.x.base())?;
        self.norm(&name(layer, "attn_norm"), work.x.base(), n, w)?;
        self.attention(layer, n)?;
        self.gpu.hc_expand(work.out.base(), work.stream.base(), work.post_a.base(), work.comb_a.base(), n as u64, hc as u64,
                           w as u64, work.stream2.base())?;
        self.linear(&name(layer, "hc_ffn_fn"), work.stream2.base(), n, work.mixes.base())?;
        let (base, scale) = (self.dense(&name(layer, "hc_ffn_base"))?, self.dense(&name(layer, "hc_ffn_scale"))?);
        let pre_f = work.pre_f[layer % 2];
        self.gpu.hc_coefficients(work.mixes.base(), work.stream2.base(), n as u64, hc as u64, w as u64, base.buffer.base(),
                                 scale.buffer.base(), hp.sinkhorn_iterations as u64, hp.hc_eps, hp.norm_eps, pre_f.base(),
                                 work.post_f.base(), work.comb_f.base())?;
        self.gpu.hc_collapse(work.stream2.base(), work.pre_a.base(), n as u64, hc as u64, w as u64, work.x.base())?;
        self.norm(&name(layer, "ffn_norm"), work.x.base(), n, w)?;
        self.moe(layer, n)?;
        self.gpu.hc_expand(work.out.base(), work.stream2.base(), work.post_f.base(), work.comb_f.base(), n as u64, hc as u64,
                           w as u64, work.stream.base())
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
        self.linear(&name(layer, "attn_q_a"), work.x.base(), n, work.qr.base())?;
        self.norm(&name(layer, "attn_q_a_norm"), work.qr.base(), n, hp.q_rank)?;
        self.linear(&name(layer, "attn_q_b"), work.qr.base(), n, work.q.base())?;
        self.gpu.rope(work.q.base(), n as u64, heads as u64, dim as u64, rope, &work.positions, &freqs, false)?;
        self.linear(&name(layer, "attn_kv"), work.x.base(), n, work.kv.base())?;
        self.norm(&name(layer, "attn_kv_a_norm"), work.kv.base(), n, dim)?;
        self.gpu.rope(work.kv.base(), n as u64, 1, dim as u64, rope, &work.positions, &freqs, false)?;
        let ring = self.rings[layer];
        self.gpu.scatter(work.kv.base(), work.ring_rows.base(), n as u64, dim as u64, ring.base(), self.ring as u64)?;
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
                                  work.attended.base())?;
        self.gpu.rope(work.attended.base(), n as u64, heads as u64, dim as u64, rope, &work.positions, &freqs, true)?;
        let wo_a = self.dense(&name(layer, "attn_output_a"))?;
        let group_width = heads * dim / hp.out_groups;
        let scratch = self.scratch;
        for g in 0..hp.out_groups {
            self.gpu.linear(&wo_a.buffer, wo_a.kind, hp.out_rank as u64, wo_a.dims[0] as u64, (g * hp.out_rank) as u64,
                            work.attended.at(g * group_width), (heads * dim) as u64, n as u64, work.low.at(g * hp.out_rank),
                            (hp.out_groups * hp.out_rank) as u64, &scratch)?;
        }
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
            self.norm(&name(layer, "indexer.k_norm"), key_target, groups, hp.index_dim)?;
            let starts: Vec<i32> = (first_group..first_group + groups).map(|g| (g * ratio) as i32).collect();
            self.gpu.write_buffer(&work.group_positions, 0, &starts)?;
            self.gpu.rope(key_target, groups as u64, 1, hp.index_dim as u64, hp.rope_dims as u64, &work.group_positions, &freqs, false)?;
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
        let (w, ff) = (hp.width, hp.expert_ff);
        let work = self.work.clone_handles();
        self.linear(&name(layer, "ffn_gate_inp"), work.x.base(), n, work.router.base())?;
        let bias = self.dense(&format!("blk.{layer}.exp_probs_b.bias"))?;
        let gating = match hp.gating { Gating::Softmax => 0, Gating::Sigmoid => 1, Gating::SqrtSoftplus => 2 };
        self.gpu.route(work.router.base(), Some(bias.buffer.base()), n as u64, hp.experts as u64, hp.experts_used as u64, gating,
                       hp.route_normalize, hp.route_scale, work.ids.base(), work.weights.base())?;
        let mut routes = vec![0i32; n * hp.experts_used];
        self.gpu.read_buffer(&work.ids, 0, &mut routes)?;
        let mut weights = vec![0f32; n * hp.experts_used];
        self.gpu.read_buffer(&work.weights, 0, &mut weights)?;
        let mut by_expert: BTreeMap<usize, (Vec<i32>, Vec<f32>)> = BTreeMap::new();
        for (slot, &expert) in routes.iter().enumerate() {
            let entry = by_expert.entry(expert as usize).or_default();
            entry.0.push((slot / hp.experts_used) as i32);
            entry.1.push(weights[slot]);
        }
        // the shared expert writes `out`, the routed ones add to it
        self.linear(&name(layer, "ffn_gate_shexp"), work.x.base(), n, work.gate.base())?;
        self.linear(&name(layer, "ffn_up_shexp"), work.x.base(), n, work.up.base())?;
        self.gpu.swiglu(work.gate.base(), work.up.base(), n as u64, ff as u64, hp.layers[layer].clamp_shared, None, work.hidden.base())?;
        self.linear(&name(layer, "ffn_down_shexp"), work.hidden.base(), n, work.out.base())?;
        let gate = self.tensors.tensor(&name(layer, "ffn_gate_exps"))?;
        let up = self.tensors.tensor(&name(layer, "ffn_up_exps"))?;
        let down = self.tensors.tensor(&name(layer, "ffn_down_exps"))?;
        let (gate_bytes, down_bytes) = (ff * row_bytes(gate.kind, w)?, w * row_bytes(down.kind, ff)?);
        let (gate_kind, up_kind, down_kind) = (gate.kind, up.kind, down.kind);
        let scratch = self.scratch;
        for (expert, (rows, user_weights)) in by_expert {
            let count = rows.len();
            let slot = self.expert(layer, expert, gate_bytes, down_bytes)?;
            self.gpu.write_buffer(&work.expert_rows, 0, &rows)?;
            self.gpu.write_buffer(&work.expert_weights, 0, &user_weights)?;
            self.gpu.gather(work.x.base(), n as u64, work.expert_rows.base(), count as u64, w as u64, work.xe.base())?;
            let gate_weights = slot.slice(0, gate_bytes as u64);
            let up_weights = slot.slice(gate_bytes as u64, gate_bytes as u64);
            let down_weights = slot.slice(2 * gate_bytes as u64, down_bytes as u64);
            self.gpu.linear(&gate_weights, gate_kind, ff as u64, w as u64, 0, work.xe.base(), 0, count as u64, work.gate.base(), 0,
                            &scratch)?;
            self.gpu.linear(&up_weights, up_kind, ff as u64, w as u64, 0, work.xe.base(), 0, count as u64, work.up.base(), 0, &scratch)?;
            self.gpu.swiglu(work.gate.base(), work.up.base(), count as u64, ff as u64, hp.layers[layer].clamp_routed,
                            Some(work.expert_weights.base()), work.hidden.base())?;
            self.gpu.linear(&down_weights, down_kind, w as u64, ff as u64, 0, work.hidden.base(), 0, count as u64, work.y.base(), 0,
                            &scratch)?;
            self.gpu.scatter_add(work.y.base(), work.expert_rows.base(), count as u64, w as u64, work.out.base(), n as u64)?;
        }
        Ok(())
    }

    /// The slot holding (layer, expert), copied in on a miss over the least recently used one.
    fn expert(&mut self, layer: usize, expert: usize, gate_bytes: usize, down_bytes: usize) -> Result<Buffer> {
        self.experts.clock += 1;
        let clock = self.experts.clock;
        if let Some(&slot) = self.experts.map.get(&(layer, expert)) {
            self.experts.hits += 1;
            self.experts.used[slot] = clock;
            return Ok(self.experts.slots[slot]);
        }
        self.experts.misses += 1;
        let slot = (0..self.experts.slots.len()).min_by_key(|&slot| self.experts.used[slot]).ok_or("no expert slots")?;
        if let Some(previous) = self.experts.owner[slot].take() {
            self.experts.map.remove(&previous);
        }
        let buffer = self.experts.slots[slot];
        let started = std::time::Instant::now();
        for (index, (role, bytes)) in [("ffn_gate_exps", gate_bytes), ("ffn_up_exps", gate_bytes), ("ffn_down_exps", down_bytes)]
            .into_iter().enumerate()
        {
            let tensor = self.tensors.tensor(&name(layer, role))?;
            self.gpu.write_buffer(&buffer, (index.min(2) * gate_bytes) as u64, &tensor.data[expert * bytes..(expert + 1) * bytes])?;
        }
        self.experts.fill_seconds += started.elapsed().as_secs_f64();
        self.experts.owner[slot] = Some((layer, expert));
        self.experts.map.insert((layer, expert), slot);
        self.experts.used[slot] = clock;
        Ok(buffer)
    }
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
        }
    }
}
