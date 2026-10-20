//! A CPU F32 reference of DeepSeek-V4.1 (`deepseek41`) for validating the GPU path: the
//! reference implementation's forward pass, written plainly, over the GGUF's own weights.
//! Development only: the shards are read without authentication.
//!
//! ds41_reference SHARD_DIR TOKENS [--dump DIR] [--layers N] [--logits FILE] [--routes FILE] [--save DIR]
//!
//! TOKENS: a probe `.tokens` file (int32 prompt length, kept, generated, then the ids). With
//! --dump, every layer's tensors are compared against the reference run's dumps (int64 extents
//! then F32 values, named like `ffn_out-7.f32`); --logits compares the final logits of the
//! prompt's kept positions (F32 rows); --save writes every compared tensor of this run in the
//! dump format (the oracle for the GPU path).

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use straydog_engine::arch::deepseek41::{Gating, Hparams, Rope};
use straydog_engine::formats;
use straydog_gguf::{ModelHeader, TensorInfo};

struct Tensor<'a> {
    kind: u32,
    dims: Vec<usize>,
    data: &'a [u8],
}

impl Tensor<'_> {
    fn row_bytes(&self) -> usize {
        formats::matrix_bytes(self.kind, 1, self.dims[0]).expect("a known format")
    }
}

/// Every shard mapped, tensors by name.
struct Files {
    maps: Vec<Mmap>,
    tensors: HashMap<String, (usize, TensorInfo)>,
    header: ModelHeader,
}

impl Files {
    fn open(dir: &Path) -> Result<Files, String> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|exc| exc.to_string())?
            .flatten().map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "gguf")).collect();
        shards.sort();
        let (mut maps, mut tensors, mut first) = (Vec::new(), HashMap::new(), None);
        for (index, path) in shards.iter().enumerate() {
            let file = File::open(path).map_err(|exc| format!("{}: {exc}", path.display()))?;
            // SAFETY: a read-only mapping of a file this process never writes.
            let map = unsafe { Mmap::map(&file) }.map_err(|exc| exc.to_string())?;
            let header = straydog_gguf::inspect(&mut std::io::Cursor::new(&map[..])).map_err(|exc| exc.to_string())?;
            for tensor in &header.tensors {
                tensors.insert(tensor.name.clone(), (index, tensor.clone()));
            }
            if header.shard_index == 0 {
                first = Some(header);
            }
            maps.push(map);
        }
        Ok(Files { maps, tensors, header: first.ok_or("no first shard")? })
    }

    fn tensor(&self, name: &str) -> Tensor<'_> {
        let (shard, info) = self.tensors.get(name).unwrap_or_else(|| panic!("tensor {name} is missing"));
        let start = info.offset as usize;
        Tensor {
            kind: info.quant_type,
            dims: info.dimensions.iter().map(|&dim| dim as usize).collect(),
            data: &self.maps[*shard][start..start + info.bytes as usize],
        }
    }

    fn vector(&self, name: &str) -> Vec<f32> {
        let tensor = self.tensor(name);
        decode(tensor.kind, tensor.data)
    }
}

fn decode(kind: u32, data: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    formats::decode_into(kind, data, &mut out).unwrap_or_else(|exc| panic!("decode type {kind}: {exc}"));
    out
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(8, |count| count.get())
}

/// `out[t][r] = W[r] . x[t]` for `rows` rows of a matrix starting at `first`; `x` holds
/// `tokens` rows of the matrix's width. Rows are decoded in chunks, in parallel.
fn linear_rows(matrix: &Tensor, first: usize, rows: usize, x: &[f32], tokens: usize) -> Vec<f32> {
    let width = matrix.dims[0];
    assert_eq!(x.len(), tokens * width);
    let row_bytes = matrix.row_bytes();
    let mut out = vec![0f32; tokens * rows];
    let chunk = rows.div_ceil(threads() * 4).max(1);
    let pieces: Vec<(usize, Vec<f32>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..rows).step_by(chunk).map(|start| {
            scope.spawn(move || {
                let count = chunk.min(rows - start);
                let bytes = &matrix.data[(first + start) * row_bytes..(first + start + count) * row_bytes];
                let weights = decode(matrix.kind, bytes);
                let mut part = vec![0f32; tokens * count];
                for t in 0..tokens {
                    let input = &x[t * width..(t + 1) * width];
                    for r in 0..count {
                        let row = &weights[r * width..(r + 1) * width];
                        part[t * count + r] = dot(row, input);
                    }
                }
                (start, part)
            })
        }).collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    });
    for (start, part) in pieces {
        let count = part.len() / tokens;
        for t in 0..tokens {
            out[t * rows + start..t * rows + start + count].copy_from_slice(&part[t * count..(t + 1) * count]);
        }
    }
    out
}

fn linear(matrix: &Tensor, x: &[f32], tokens: usize) -> Vec<f32> {
    linear_rows(matrix, 0, matrix.dims[1], x, tokens)
}

/// Dot product accumulated in F64, so the reference's sums do not depend on the order.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum::<f64>() as f32
}

fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    for row in x.chunks_mut(weight.len()) {
        let mean = row.iter().map(|v| *v as f64 * *v as f64).sum::<f64>() / row.len() as f64;
        let scale = 1.0 / (mean + eps as f64).sqrt();
        for (v, w) in row.iter_mut().zip(weight) {
            *v = (*v as f64 * scale) as f32 * w;
        }
    }
}

/// Rotary frequencies per pair, with YaRN's ramp between the bands that keep and divide them.
fn frequencies(dims: usize, rope: &Rope) -> Vec<f64> {
    let mut freqs: Vec<f64> = (0..dims / 2).map(|i| 1.0 / rope.base.powf((2 * i) as f64 / dims as f64)).collect();
    if let Some((factor, original, beta_fast, beta_slow)) = rope.yarn {
        let corrected = |rotations: f64| {
            dims as f64 * (original / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * rope.base.ln())
        };
        let low = corrected(beta_fast).floor().max(0.0);
        let high = corrected(beta_slow).ceil().min(dims as f64 - 1.0);
        for (i, freq) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f64 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    freqs
}

/// Rotate the last `freqs.len() * 2` values of `x` (adjacent pairs) by `position`.
fn rotate(x: &mut [f32], freqs: &[f64], position: usize, inverse: bool) {
    let start = x.len() - 2 * freqs.len();
    for (i, freq) in freqs.iter().enumerate() {
        let angle = position as f64 * freq * if inverse { -1.0 } else { 1.0 };
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (x[start + 2 * i] as f64, x[start + 2 * i + 1] as f64);
        x[start + 2 * i] = (a * cos - b * sin) as f32;
        x[start + 2 * i + 1] = (a * sin + b * cos) as f32;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

struct Model<'a> {
    files: &'a Files,
    hp: Hparams,
    /// The reference run's experts per layer, [token][slot], for comparison.
    routes: HashMap<usize, Vec<Vec<usize>>>,
}

/// What attention layers hand down: each kv source's compressed rows and index keys, and each
/// index source's selection per query (positions into the compressed rows).
#[derive(Default)]
struct Shared {
    rows: HashMap<usize, Vec<Vec<f32>>>,
    index_keys: HashMap<usize, Vec<Vec<f32>>>,
    selections: HashMap<usize, Vec<Vec<usize>>>,
    candidates: Option<Vec<Vec<bool>>>,
}

impl Model<'_> {
    fn name(layer: usize, tensor: &str) -> String {
        format!("blk.{layer}.{tensor}.weight")
    }

    /// The pre, post and combination coefficients of one sublayer from the hc streams.
    fn mixes(&self, layer: usize, which: &str, stream: &[f32], tokens: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (hc, width) = (self.hp.hc, self.hp.width);
        let function = self.files.tensor(&Model::name(layer, &format!("hc_{which}_fn")));
        let base = self.files.vector(&Model::name(layer, &format!("hc_{which}_base")));
        let scale = self.files.vector(&Model::name(layer, &format!("hc_{which}_scale")));
        let mixes = linear(&function, stream, tokens);
        let mix = (2 + hc) * hc;
        let eps = self.hp.hc_eps;
        let (mut pre, mut post, mut comb) = (vec![0f32; tokens * hc], vec![0f32; tokens * hc], vec![0f32; tokens * hc * hc]);
        for t in 0..tokens {
            let flat = &stream[t * hc * width..(t + 1) * hc * width];
            let mean = flat.iter().map(|v| *v as f64 * *v as f64).sum::<f64>() / flat.len() as f64;
            let rsqrt = (1.0 / (mean + self.hp.norm_eps as f64).sqrt()) as f32;
            let m: Vec<f32> = mixes[t * mix..(t + 1) * mix].iter().map(|v| v * rsqrt).collect();
            let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
            for j in 0..hc {
                pre[t * hc + j] = sigmoid(m[j] * scale[0] + base[j]) + eps;
                post[t * hc + j] = 2.0 * sigmoid(m[j + hc] * scale[1] + base[j + hc]);
            }
            let mut c = vec![0f32; hc * hc];
            for j in 0..hc {
                for k in 0..hc {
                    c[j * hc + k] = m[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
                }
            }
            for j in 0..hc {
                let row = &mut c[j * hc..(j + 1) * hc];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
                for v in row.iter_mut() {
                    *v = (*v - max).exp() / sum + eps;
                }
            }
            let normalize_columns = |c: &mut [f32]| {
                for k in 0..hc {
                    let sum: f32 = (0..hc).map(|j| c[j * hc + k]).sum();
                    for j in 0..hc {
                        c[j * hc + k] /= sum + eps;
                    }
                }
            };
            normalize_columns(&mut c);
            for _ in 1..self.hp.sinkhorn_iterations {
                for j in 0..hc {
                    let sum: f32 = c[j * hc..(j + 1) * hc].iter().sum();
                    for k in 0..hc {
                        c[j * hc + k] /= sum + eps;
                    }
                }
                normalize_columns(&mut c);
            }
            comb[t * hc * hc..(t + 1) * hc * hc].copy_from_slice(&c);
        }
        (pre, post, comb)
    }

    /// `y[j] = post[j] x + sum_i comb[i][j] residual[i]`.
    fn post(&self, x: &[f32], residual: &[f32], post: &[f32], comb: &[f32], tokens: usize) -> Vec<f32> {
        let (hc, width) = (self.hp.hc, self.hp.width);
        let mut out = vec![0f32; tokens * hc * width];
        for t in 0..tokens {
            for j in 0..hc {
                let target = &mut out[(t * hc + j) * width..(t * hc + j + 1) * width];
                for (d, value) in target.iter_mut().enumerate() {
                    let mut sum = post[t * hc + j] as f64 * x[t * width + d] as f64;
                    for i in 0..hc {
                        sum += comb[t * hc * hc + i * hc + j] as f64 * residual[(t * hc + i) * width + d] as f64;
                    }
                    *value = sum as f32;
                }
            }
        }
        out
    }

    /// The stream collapsed by `pre` into one row per token.
    fn collapse(&self, stream: &[f32], pre: &[f32], tokens: usize) -> Vec<f32> {
        let (hc, width) = (self.hp.hc, self.hp.width);
        let mut out = vec![0f32; tokens * width];
        for t in 0..tokens {
            for d in 0..width {
                out[t * width + d] = (0..hc).map(|j| pre[t * hc + j] as f64 * stream[(t * hc + j) * width + d] as f64).sum::<f64>() as f32;
            }
        }
        out
    }

    fn engram(&self, layer: usize, module: usize, stream: &mut [f32], ids: &[u32], tokens: usize) {
        let (hc, width) = (self.hp.hc, self.hp.width);
        let engram = self.hp.engram.as_ref().unwrap();
        let table = self.files.tensor("per_layer_token_embd.weight");
        let row_bytes = table.row_bytes();
        let mut rows = Vec::new();
        let mut gathered = Vec::with_capacity(tokens * self.hp.engram_columns() * engram.head_dim);
        for t in 0..tokens {
            rows.clear();
            engram.rows(module, ids[t], &ids[t.saturating_sub(engram.ngram - 1)..t], t, &mut rows);
            for &row in &rows {
                gathered.extend(decode(table.kind, &table.data[row as usize * row_bytes..(row as usize + 1) * row_bytes]));
            }
        }
        let kv = linear(&self.files.tensor(&Model::name(layer, "engram_kv")), &gathered, tokens);
        let gate = self.files.vector(&Model::name(layer, "engram_gate"));
        let eps = self.hp.norm_eps as f64;
        for t in 0..tokens {
            let row = &kv[t * (hc + 1) * width..(t + 1) * (hc + 1) * width];
            let value = &row[hc * width..];
            for j in 0..hc {
                let key = &row[j * width..(j + 1) * width];
                let h = &stream[(t * hc + j) * width..(t * hc + j + 1) * width];
                let rms = |v: &[f32]| 1.0 / (v.iter().map(|x| *x as f64 * *x as f64).sum::<f64>() / width as f64 + eps).sqrt();
                let rstd = rms(h) * rms(key);
                let product: f64 = (0..width).map(|d| h[d] as f64 * gate[j * width + d] as f64 * key[d] as f64).sum();
                let dot = product * rstd / (width as f64).sqrt();
                let signed = dot.abs().max(1e-6).sqrt().copysign(dot);
                let g = 1.0 / (1.0 + (-signed).exp());
                let h = &mut stream[(t * hc + j) * width..(t * hc + j + 1) * width];
                for d in 0..width {
                    h[d] = (h[d] as f64 + g * value[d] as f64) as f32;
                }
            }
        }
    }

    fn attention(&self, layer: usize, x: &[f32], tokens: usize, shared: &mut Shared, dump: Option<&Path>) -> Vec<f32> {
        let hp = &self.hp;
        let (heads, head_dim, rope_dims) = (hp.heads, hp.head_dim, hp.rope_dims);
        let plan = hp.layers[layer].compressed;
        let freqs = frequencies(rope_dims, if plan.is_some() { &hp.compressed_rope } else { &hp.rope });
        let eps = hp.norm_eps;
        let mut qr = linear(&self.files.tensor(&Model::name(layer, "attn_q_a")), x, tokens);
        rms_norm(&mut qr, &self.files.vector(&Model::name(layer, "attn_q_a_norm")), eps);
        compare(dump, &format!("qr_norm-{layer}"), &qr);
        let mut q = linear(&self.files.tensor(&Model::name(layer, "attn_q_b")), &qr, tokens);
        for t in 0..tokens {
            for h in 0..heads {
                rotate(&mut q[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim], &freqs, t, false);
            }
        }
        let mut kv = linear(&self.files.tensor(&Model::name(layer, "attn_kv")), x, tokens);
        rms_norm(&mut kv, &self.files.vector(&Model::name(layer, "attn_kv_a_norm")), eps);
        for t in 0..tokens {
            rotate(&mut kv[t * head_dim..(t + 1) * head_dim], &freqs, t, false);
        }
        compare(dump, &format!("q-{layer}"), &q);
        compare(dump, &format!("kv-{layer}"), &kv);
        if let Some(plan) = plan {
            if plan.kv_source == layer {
                self.compress(layer, plan.ratio, x, tokens, &freqs, shared);
            }
            if plan.index_source == layer {
                self.index(layer, plan.ratio, plan.kv_source, x, &qr, tokens, &freqs, shared);
            }
        }
        let sinks = self.files.vector(&Model::name(layer, "attn_sinks"));
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = vec![0f32; tokens * heads * head_dim];
        for t in 0..tokens {
            // keys: the window's rows, then the selected compressed rows
            let mut keys: Vec<&[f32]> = (t.saturating_sub(hp.window - 1)..=t).map(|s| &kv[s * head_dim..(s + 1) * head_dim]).collect();
            if let Some(plan) = plan {
                let rows = &shared.rows[&plan.kv_source];
                for &position in &shared.selections[&plan.index_source][t] {
                    keys.push(&rows[position]);
                }
            }
            for h in 0..heads {
                let query = &q[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
                let scores: Vec<f64> = keys.iter().map(|k| dot(query, k) as f64 * scale).collect();
                let max = scores.iter().cloned().fold(sinks[h] as f64, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total = weights.iter().sum::<f64>() + (sinks[h] as f64 - max).exp();
                let target = &mut out[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
                for (k, w) in keys.iter().zip(&weights) {
                    for d in 0..head_dim {
                        target[d] += (w / total * k[d] as f64) as f32;
                    }
                }
                rotate(target, &freqs, t, true);
            }
        }
        // grouped low-rank output: each group of heads through its own slice of wo_a
        let (groups, rank) = (hp.out_groups, hp.out_rank);
        let wo_a = self.files.tensor(&Model::name(layer, "attn_output_a"));
        let group_width = heads * head_dim / groups;
        let mut low = vec![0f32; tokens * groups * rank];
        for g in 0..groups {
            let inputs: Vec<f32> = (0..tokens).flat_map(|t| out[t * heads * head_dim + g * group_width..t * heads * head_dim + (g + 1) * group_width].to_vec()).collect();
            let matrix = Tensor { kind: wo_a.kind, dims: vec![group_width, rank], data: wo_a.data };
            let part = linear_rows(&matrix, g * rank, rank, &inputs, tokens);
            for t in 0..tokens {
                low[t * groups * rank + g * rank..t * groups * rank + (g + 1) * rank].copy_from_slice(&part[t * rank..(t + 1) * rank]);
            }
        }
        linear(&self.files.tensor(&Model::name(layer, "attn_output_b")), &low, tokens)
    }

    /// Pool `ratio` consecutive tokens into compressed rows (with their index keys) for the
    /// layers that read this source; a row stands for its group's first position.
    fn compress(&self, layer: usize, ratio: usize, x: &[f32], tokens: usize, freqs: &[f64], shared: &mut Shared) {
        let hp = &self.hp;
        let head_dim = hp.head_dim;
        let kv = linear(&self.files.tensor(&Model::name(layer, "attn_compressor_kv")), x, tokens);
        let gate = if ratio > 1 { Some(linear(&self.files.tensor(&Model::name(layer, "attn_compressor_gate")), x, tokens)) } else { None };
        let norm = self.files.vector(&Model::name(layer, "attn_compressor_norm"));
        let mut latents = Vec::new();
        for group in 0..tokens / ratio {
            let mut latent = vec![0f32; head_dim];
            for (d, value) in latent.iter_mut().enumerate() {
                match &gate {
                    None => *value = kv[group * head_dim + d],
                    Some(gate) => {
                        let scores: Vec<f64> = (0..ratio).map(|r| gate[(group * ratio + r) * head_dim + d] as f64).collect();
                        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let sum: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                        *value = (0..ratio).map(|r| (scores[r] - max).exp() / sum * kv[(group * ratio + r) * head_dim + d] as f64).sum::<f64>() as f32;
                    }
                }
            }
            rms_norm(&mut latent, &norm, hp.norm_eps);
            latents.push(latent);
        }
        let flat: Vec<f32> = latents.concat();
        let groups = latents.len();
        let mut keys = if groups > 0 { linear(&self.files.tensor(&Model::name(layer, "indexer.attn_k")), &flat, groups) } else { Vec::new() };
        if groups > 0 {
            rms_norm(&mut keys, &self.files.vector(&Model::name(layer, "indexer.k_norm")), hp.norm_eps);
        }
        let index_keys: Vec<Vec<f32>> = (0..groups).map(|g| {
            let mut key = keys[g * hp.index_dim..(g + 1) * hp.index_dim].to_vec();
            rotate(&mut key, freqs, g * ratio, false);
            key
        }).collect();
        for (g, latent) in latents.iter_mut().enumerate() {
            rotate(latent, freqs, g * ratio, false);
        }
        shared.rows.insert(layer, latents);
        shared.index_keys.insert(layer, index_keys);
    }

    #[allow(clippy::too_many_arguments)]
    fn index(&self, layer: usize, ratio: usize, source: usize, x: &[f32], qr: &[f32], tokens: usize, freqs: &[f64], shared: &mut Shared) {
        let hp = &self.hp;
        let (heads, dim) = (hp.index_heads, hp.index_dim);
        let mut queries = linear(&self.files.tensor(&Model::name(layer, "indexer.attn_q_b")), qr, tokens);
        for t in 0..tokens {
            for h in 0..heads {
                rotate(&mut queries[(t * heads + h) * dim..(t * heads + h + 1) * dim], freqs, t, false);
            }
        }
        let weights = linear(&self.files.tensor(&Model::name(layer, "indexer.proj")), x, tokens);
        let scale = 1.0 / (dim as f64).sqrt() / (heads as f64).sqrt();
        let keys = &shared.index_keys[&source];
        let candidate_source = hp.candidates.map(|(source, _, _)| source);
        let mut selections = Vec::with_capacity(tokens);
        let mut candidates = Vec::new();
        for t in 0..tokens {
            let visible = (t + 1) / ratio;
            let mut scores: Vec<f64> = (0..visible).map(|s| {
                (0..heads).map(|h| {
                    let q = &queries[(t * heads + h) * dim..(t * heads + h + 1) * dim];
                    (dot(q, &keys[s]) as f64).max(0.0) * weights[t * heads + h] as f64 * scale
                }).sum()
            }).collect();
            if candidate_source == Some(layer) {
                let (_, blocks, block) = hp.candidates.unwrap();
                let count = visible.div_ceil(block);
                let mut best: Vec<(f64, usize)> = (0..count).map(|b| {
                    let score = scores[b * block..((b + 1) * block).min(visible)].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    (if visible > 0 && b == (visible - 1) / block { f64::INFINITY } else { score }, b)
                }).collect();
                best.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                let mut keep = vec![false; visible];
                for &(score, b) in best.iter().take(blocks) {
                    if score > f64::NEG_INFINITY {
                        keep[b * block..((b + 1) * block).min(visible)].fill(true);
                    }
                }
                candidates.push(keep);
            } else if candidate_source.is_some_and(|source| source < layer) {
                let keep = &shared.candidates.as_ref().unwrap()[t];
                for (s, score) in scores.iter_mut().enumerate() {
                    if !keep[s] {
                        *score = f64::NEG_INFINITY;
                    }
                }
            }
            let mut order: Vec<usize> = (0..visible).collect();
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            let mut selected: Vec<usize> = order.into_iter().take(hp.index_top_k).filter(|&s| scores[s] > f64::NEG_INFINITY).collect();
            selected.sort_unstable();
            selections.push(selected);
        }
        if candidate_source == Some(layer) {
            shared.candidates = Some(candidates);
        }
        shared.selections.insert(layer, selections);
    }

    fn moe(&self, layer: usize, x: &[f32], tokens: usize) -> (Vec<f32>, Vec<f32>) {
        let hp = &self.hp;
        let (width, ff) = (hp.width, hp.expert_ff);
        let plan = &hp.layers[layer];
        let logits = linear(&self.files.tensor(&Model::name(layer, "ffn_gate_inp")), x, tokens);
        let bias = self.files.vector(&format!("blk.{layer}.exp_probs_b.bias"));
        let mut routes: Vec<Vec<(usize, f32)>> = Vec::with_capacity(tokens);
        for t in 0..tokens {
            let scores: Vec<f32> = logits[t * hp.experts..(t + 1) * hp.experts].iter().map(|&l| match hp.gating {
                Gating::Softmax => l,
                Gating::Sigmoid => 1.0 / (1.0 + (-l).exp()),
                Gating::SqrtSoftplus => (if l > 20.0 { l } else { l.exp().ln_1p() }).sqrt(),
            }).collect();
            let scores = if hp.gating == Gating::Softmax {
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                scores.iter().map(|s| (s - max).exp() / sum).collect()
            } else {
                scores
            };
            let mut order: Vec<usize> = (0..hp.experts).collect();
            order.sort_by(|&a, &b| (scores[b] + bias[b]).total_cmp(&(scores[a] + bias[a])).then(a.cmp(&b)));
            let chosen: Vec<usize> = order[..hp.experts_used].to_vec();
            let mut weights: Vec<f32> = chosen.iter().map(|&e| scores[e]).collect();
            if hp.route_normalize && hp.experts_used > 1 {
                let sum: f32 = weights.iter().sum::<f32>() + 1e-20;
                weights.iter_mut().for_each(|w| *w /= sum);
            }
            routes.push(chosen.into_iter().zip(weights.into_iter().map(|w| w * hp.route_scale)).collect());
        }
        if let Some(theirs) = self.routes.get(&layer) {
            let (mut same, mut shared) = (0, 0);
            for (ours, theirs) in routes.iter().zip(theirs) {
                let mut a: Vec<usize> = ours.iter().map(|r| r.0).collect();
                let mut b = theirs.clone();
                a.sort_unstable();
                b.sort_unstable();
                same += usize::from(a == b);
                shared += a.iter().filter(|e| b.contains(e)).count();
            }
            println!("  routes-{layer:<11} {same}/{tokens} tokens identical, {:.1}% of slots shared", 100.0 * shared as f64 / (tokens * hp.experts_used) as f64);
        }
        let gate = self.files.tensor(&Model::name(layer, "ffn_gate_exps"));
        let up = self.files.tensor(&Model::name(layer, "ffn_up_exps"));
        let down = self.files.tensor(&Model::name(layer, "ffn_down_exps"));
        let mut routed = vec![0f32; tokens * width];
        let mut by_expert: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (t, route) in routes.iter().enumerate() {
            for &(e, w) in route {
                by_expert.entry(e).or_default().push((t, w));
            }
        }
        let clamp = plan.clamp_routed;
        let mut experts: Vec<_> = by_expert.into_iter().collect();
        experts.sort_by_key(|(e, _)| *e);
        for (e, users) in experts {
            let inputs: Vec<f32> = users.iter().flat_map(|&(t, _)| x[t * width..(t + 1) * width].to_vec()).collect();
            let n = users.len();
            let g = linear_rows(&gate, e * ff, ff, &inputs, n);
            let u = linear_rows(&up, e * ff, ff, &inputs, n);
            let hidden: Vec<f32> = (0..n * ff).map(|i| {
                let (mut gv, mut uv) = (g[i], u[i]);
                if clamp > 0.0 {
                    uv = uv.clamp(-clamp, clamp);
                    gv = gv.min(clamp);
                }
                silu(gv) * uv * users[i / ff].1
            }).collect();
            let y = linear_rows(&down, e * width, width, &hidden, n);
            for (k, &(t, _)) in users.iter().enumerate() {
                for d in 0..width {
                    routed[t * width + d] += y[k * width + d];
                }
            }
        }
        let g = linear(&self.files.tensor(&Model::name(layer, "ffn_gate_shexp")), x, tokens);
        let u = linear(&self.files.tensor(&Model::name(layer, "ffn_up_shexp")), x, tokens);
        let clamp = plan.clamp_shared;
        let hidden: Vec<f32> = g.iter().zip(&u).map(|(&gv, &uv)| {
            let (gv, uv) = if clamp > 0.0 { (gv.min(clamp), uv.clamp(-clamp, clamp)) } else { (gv, uv) };
            silu(gv) * uv
        }).collect();
        let shared = linear(&self.files.tensor(&Model::name(layer, "ffn_down_shexp")), &hidden, tokens);
        (routed, shared)
    }
}

/// Where `--save` writes this run's tensors, in the dump format, for checking other paths.
static SAVE: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();

/// Compare a computed [tokens][width] tensor with a dump of the reference run.
fn compare(dump: Option<&Path>, name: &str, values: &[f32]) {
    if let Some(Some(dir)) = SAVE.get() {
        let mut bytes = Vec::with_capacity(32 + values.len() * 4);
        for extent in [values.len() as i64, 1, 1, 1] {
            bytes.extend_from_slice(&extent.to_le_bytes());
        }
        bytes.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        std::fs::write(dir.join(format!("{name}.f32")), bytes).unwrap();
    }
    let Some(dir) = dump else { return };
    let Ok(bytes) = std::fs::read(dir.join(format!("{name}.f32"))) else {
        println!("  {name:<18} (no dump)");
        return;
    };
    let reference: Vec<f32> = bytes[32..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    if reference.len() != values.len() {
        println!("  {name:<18} size {} vs dump {}", values.len(), reference.len());
        return;
    }
    let (mut diff, mut norm, mut max) = (0f64, 0f64, 0f64);
    for (a, b) in values.iter().zip(&reference) {
        diff += (*a as f64 - *b as f64).powi(2);
        norm += (*b as f64).powi(2);
        max = max.max((*a as f64 - *b as f64).abs());
    }
    println!("  {name:<18} rel {:.2e}  max {:.2e}", (diff / norm.max(1e-30)).sqrt(), max);
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut option = |name: &str| args.iter().position(|arg| arg == name).map(|at| {
        let value = args[at + 1].clone();
        args.drain(at..at + 2);
        value
    });
    let dump = option("--dump").map(PathBuf::from);
    let logits_file = option("--logits").map(PathBuf::from);
    let last_layer: Option<usize> = option("--layers").map(|n| n.parse().unwrap());
    let option_routes = option("--routes").map(PathBuf::from);
    let save = option("--save").map(PathBuf::from);
    if let Some(dir) = &save {
        std::fs::create_dir_all(dir).map_err(|exc| exc.to_string())?;
    }
    SAVE.set(save).unwrap();
    let [dir, tokens_file] = &args[..] else {
        return Err("usage: ds41_reference SHARD_DIR TOKENS [--dump DIR] [--layers N] [--logits FILE]".into());
    };
    let files = Files::open(Path::new(dir))?;
    let hp = Hparams::from_header(&files.header)?;
    let raw = std::fs::read(tokens_file).map_err(|exc| exc.to_string())?;
    let ints: Vec<i32> = raw.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let (prompt, kept) = (ints[0] as usize, ints[1] as usize);
    let ids: Vec<u32> = ints[3..3 + prompt].iter().map(|&id| id as u32).collect();
    let tokens = ids.len();
    println!("{} layers, width {}, {} tokens", hp.layers.len(), hp.width, tokens);
    let mut routes = HashMap::new();
    if let Some(path) = option_routes {
        let bytes = std::fs::read(&path).map_err(|exc| exc.to_string())?;
        let ints: Vec<i32> = bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut at = 0;
        while at + 3 <= ints.len() {
            let (layer, used, count) = (ints[at] as usize, ints[at + 1] as usize, ints[at + 2] as usize);
            at += 3;
            if count == tokens && !routes.contains_key(&layer) {
                routes.insert(layer, (0..count).map(|t| ints[at + t * used..at + (t + 1) * used].iter().map(|&e| e as usize).collect()).collect());
            }
            at += used * count;
        }
    }
    let model = Model { files: &files, hp, routes };
    let (hc, width) = (model.hp.hc, model.hp.width);
    let embed = files.tensor("token_embd.weight");
    let row_bytes = embed.row_bytes();
    let mut stream = Vec::with_capacity(tokens * hc * width);
    for &id in &ids {
        let row = decode(embed.kind, &embed.data[id as usize * row_bytes..(id as usize + 1) * row_bytes]);
        for _ in 0..hc {
            stream.extend_from_slice(&row);
        }
    }
    let mut pre_previous: Option<Vec<f32>> = None;
    let mut shared = Shared::default();
    let started = std::time::Instant::now();
    let layers = last_layer.unwrap_or(model.hp.layers.len()).min(model.hp.layers.len());
    for layer in 0..layers {
        let dump = dump.as_deref();
        if let Some(module) = model.hp.layers[layer].engram {
            model.engram(layer, module, &mut stream, &ids, tokens);
        }
        let residual = stream.clone();
        let (attn_pre, post, comb) = model.mixes(layer, "attn", &stream, tokens);
        let mut x = match &pre_previous {
            Some(pre) => model.collapse(&stream, pre, tokens),
            None => (0..tokens).flat_map(|t| stream[t * hc * width..t * hc * width + width].to_vec()).collect(),
        };
        compare(dump, &format!("hc_attn_pre-{layer}"), &x);
        rms_norm(&mut x, &files.vector(&Model::name(layer, "attn_norm")), model.hp.norm_eps);
        compare(dump, &format!("attn_norm-{layer}"), &x);
        let attended = model.attention(layer, &x, tokens, &mut shared, dump);
        compare(dump, &format!("attn_out-{layer}"), &attended);
        stream = model.post(&attended, &residual, &post, &comb, tokens);
        let residual = stream.clone();
        let (ffn_pre, post, comb) = model.mixes(layer, "ffn", &stream, tokens);
        let mut x = model.collapse(&stream, &attn_pre, tokens);
        compare(dump, &format!("hc_ffn_pre-{layer}"), &x);
        rms_norm(&mut x, &files.vector(&Model::name(layer, "ffn_norm")), model.hp.norm_eps);
        compare(dump, &format!("ffn_norm-{layer}"), &x);
        let (routed, shared_out) = model.moe(layer, &x, tokens);
        compare(dump, &format!("ffn_moe_out-{layer}"), &routed);
        compare(dump, &format!("ffn_shexp-{layer}"), &shared_out);
        let out: Vec<f32> = routed.iter().zip(&shared_out).map(|(a, b)| a + b).collect();
        compare(dump, &format!("ffn_out-{layer}"), &out);
        stream = model.post(&out, &residual, &post, &comb, tokens);
        pre_previous = Some(ffn_pre);
        println!("layer {layer} done at {:.1} s", started.elapsed().as_secs_f64());
    }
    if layers < model.hp.layers.len() {
        return Ok(());
    }
    let mut x = model.collapse(&stream, pre_previous.as_ref().unwrap(), tokens);
    rms_norm(&mut x, &files.vector("output_norm.weight"), model.hp.norm_eps);
    let last = &x[(tokens - kept) * width..];
    let logits = linear(&files.tensor("output.weight"), last, kept);
    compare(dump.as_deref(), "result_output", &logits);
    if let Some(path) = logits_file {
        let bytes = std::fs::read(&path).map_err(|exc| exc.to_string())?;
        let reference: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let vocabulary = model.hp.vocabulary;
        for k in 0..kept {
            let ours = &logits[k * vocabulary..(k + 1) * vocabulary];
            let theirs = &reference[k * vocabulary..(k + 1) * vocabulary];
            let softmax = |v: &[f32]| {
                let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum: f64 = v.iter().map(|x| (*x as f64 - max).exp()).sum();
                v.iter().map(|x| (*x as f64 - max).exp() / sum).collect::<Vec<f64>>()
            };
            let (p, q) = (softmax(theirs), softmax(ours));
            let kl: f64 = p.iter().zip(&q).filter(|(a, _)| **a > 0.0).map(|(a, b)| a * (a / b.max(1e-300)).ln()).sum();
            let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            println!("position {}: KL(ref||ours) {kl:.3e}, top-1 {} vs {}", tokens - kept + k, argmax(theirs), argmax(ours));
        }
    }
    Ok(())
}
