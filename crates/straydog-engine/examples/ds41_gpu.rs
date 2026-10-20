//! DeepSeek-V4.1's forward pass on one GPU, composed from the generic ops (docs/DESIGN.md §10),
//! checked layer by layer against the CPU reference's tensors (`ds41_reference --save DIR`).
//! Development only: shards are read without authentication, dense weights are uploaded as
//! they are first used, and each layer's experts are copied in for the prompt's routes.
//!
//! ds41_gpu SHARD_DIR TOKENS ORACLE_DIR [--layers N]

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use straydog_engine::arch::deepseek41::{Gating, Hparams, Rope};
use straydog_engine::formats;
use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;
use straydog_gguf::{ModelHeader, TensorInfo};

/// GGUF types the GPU's linear op decodes (the rest run on the host here).
const GPU_KINDS: [u32; 19] = [0, 1, 2, 6, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 30, 42];

struct Files {
    maps: Vec<Mmap>,
    tensors: HashMap<String, (usize, TensorInfo)>,
    header: ModelHeader,
}

impl Files {
    fn open(dir: &Path) -> Result<Files, String> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|exc| exc.to_string())?
            .flatten().map(|entry| entry.path()).filter(|path| path.extension().is_some_and(|ext| ext == "gguf")).collect();
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

    /// (type, extents, bytes)
    fn tensor(&self, name: &str) -> (u32, Vec<usize>, &[u8]) {
        let (shard, info) = self.tensors.get(name).unwrap_or_else(|| panic!("tensor {name} is missing"));
        let start = info.offset as usize;
        (info.quant_type, info.dimensions.iter().map(|&d| d as usize).collect(), &self.maps[*shard][start..start + info.bytes as usize])
    }
}

fn decode(kind: u32, data: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    formats::decode_into(kind, data, &mut out).unwrap();
    out
}

fn row_bytes(kind: u32, cols: usize) -> usize {
    formats::matrix_bytes(kind, 1, cols).unwrap()
}

/// Rotary frequencies per pair (YaRN between the bands it keeps and divides), as the reference.
fn frequencies(dims: usize, rope: &Rope) -> Vec<f32> {
    let mut freqs: Vec<f64> = (0..dims / 2).map(|i| 1.0 / rope.base.powf((2 * i) as f64 / dims as f64)).collect();
    if let Some((factor, original, beta_fast, beta_slow)) = rope.yarn {
        let corrected = |rotations: f64| dims as f64 * (original / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * rope.base.ln());
        let low = corrected(beta_fast).floor().max(0.0);
        let high = corrected(beta_slow).ceil().min(dims as f64 - 1.0);
        for (i, freq) in freqs.iter_mut().enumerate() {
            let smooth = 1.0 - ((i as f64 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    freqs.into_iter().map(|f| f as f32).collect()
}

struct Gpu<'a> {
    backend: Backend,
    files: &'a Files,
    weights: HashMap<String, (Buffer, u32, Vec<usize>)>,
    scratch: Buffer,
    oracle: PathBuf,
}

type P = *const c_void;

impl Gpu<'_> {
    fn weight(&mut self, name: &str) -> (Buffer, u32, Vec<usize>) {
        if let Some(found) = self.weights.get(name) {
            return found.clone();
        }
        let (kind, dims, bytes) = self.files.tensor(name);
        let buffer = self.backend.upload_bytes(bytes).unwrap();
        self.weights.insert(name.to_string(), (buffer, kind, dims.clone()));
        (buffer, kind, dims)
    }

    fn buffer(&mut self, floats: usize) -> Buffer {
        self.backend.buffer((floats.max(1) * 4) as u64).unwrap()
    }

    fn values<T: Copy>(&mut self, values: &[T]) -> Buffer {
        let buffer = self.backend.buffer((std::mem::size_of_val(values).max(4)) as u64).unwrap();
        self.backend.write_buffer(&buffer, 0, values).unwrap();
        buffer
    }

    fn read(&mut self, buffer: Buffer, offset: usize, count: usize) -> Vec<f32> {
        let mut out = vec![0f32; count];
        self.backend.read_buffer(&buffer, offset, &mut out).unwrap();
        out
    }

    /// `out = W . input` for a whole 2D tensor (rows = extent 1).
    fn linear(&mut self, name: &str, input: P, steps: usize, out: *mut c_void) {
        let (weights, kind, dims) = self.weight(name);
        let scratch = self.scratch;
        self.backend.linear(&weights, kind, dims[1] as u64, dims[0] as u64, 0, input, 0, steps as u64, out, 0, &scratch).unwrap();
    }

    fn norm(&mut self, name: &str, x: *mut c_void, rows: usize, width: usize, eps: f32) {
        let (weights, _, _) = self.weight(name);
        self.backend.rms_norm(x, &weights, rows as u64, width as u64, eps, x).unwrap();
    }

    fn compare(&mut self, name: &str, buffer: Buffer, count: usize) {
        let ours = self.read(buffer, 0, count);
        let Ok(bytes) = std::fs::read(self.oracle.join(format!("{name}.f32"))) else {
            println!("  {name:<18} (no oracle)");
            return;
        };
        let oracle: Vec<f32> = bytes[32..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        if oracle.len() != ours.len() {
            println!("  {name:<18} size {} vs oracle {}", ours.len(), oracle.len());
            return;
        }
        let (mut diff, mut norm, mut max) = (0f64, 0f64, 0f64);
        for (a, b) in ours.iter().zip(&oracle) {
            diff += (*a as f64 - *b as f64).powi(2);
            norm += (*b as f64).powi(2);
            max = max.max((*a as f64 - *b as f64).abs());
        }
        println!("  {name:<18} rel {:.2e}  max {:.2e}", (diff / norm.max(1e-30)).sqrt(), max);
    }
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let layers_limit = args.iter().position(|arg| arg == "--layers").map(|at| {
        let value: usize = args[at + 1].parse().unwrap();
        args.drain(at..at + 2);
        value
    });
    let [dir, tokens_file, oracle] = &args[..] else {
        return Err("usage: ds41_gpu SHARD_DIR TOKENS ORACLE_DIR [--layers N]".into());
    };
    let files = Files::open(Path::new(dir))?;
    let hp = Hparams::from_header(&files.header)?;
    let raw = std::fs::read(tokens_file).map_err(|exc| exc.to_string())?;
    let ints: Vec<i32> = raw.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let (count, kept) = (ints[0] as usize, ints[1] as usize);
    let ids: Vec<u32> = ints[3..3 + count].iter().map(|&id| id as u32).collect();
    let t = ids.len();
    let (hc, w, heads, dim, rope) = (hp.hc, hp.width, hp.heads, hp.head_dim, hp.rope_dims);
    let eps = hp.norm_eps;
    let mut backend = Backend::new(0, 30 << 30)?;
    let scratch = backend.buffer(512 << 20)?;
    let mut gpu = Gpu { backend, files: &files, weights: HashMap::new(), scratch, oracle: PathBuf::from(oracle) };

    // Embeddings repeated into the hc copies.
    let (kind, dims, embed) = files.tensor("token_embd.weight");
    let mut stream_host = Vec::with_capacity(t * hc * w);
    for &id in &ids {
        let row = decode(kind, &embed[id as usize * row_bytes(kind, dims[0])..(id as usize + 1) * row_bytes(kind, dims[0])]);
        for _ in 0..hc {
            stream_host.extend_from_slice(&row);
        }
    }
    let stream = gpu.values(&stream_host);
    let stream2 = gpu.buffer(t * hc * w);
    let positions: Vec<i32> = (0..t as i32).collect();
    let positions = gpu.values(&positions);
    let plain = frequencies(rope, &hp.rope);
    let plain = gpu.values(&plain);
    let compressed_freqs = frequencies(rope, &hp.compressed_rope);
    let compressed_freqs = gpu.values(&compressed_freqs);
    let identity: Vec<f32> = (0..t).flat_map(|_| (0..hc).map(|j| if j == 0 { 1.0 } else { 0.0 })).collect();
    let identity = gpu.values(&identity);
    let all_rows: Vec<i32> = (0..t as i32).collect();
    let all_rows = gpu.values(&all_rows);
    let mix = (2 + hc) * hc;
    let (mixes, pre_a, post_a, comb_a) = (gpu.buffer(t * mix), gpu.buffer(t * hc), gpu.buffer(t * hc), gpu.buffer(t * hc * hc));
    let (pre_f, post_f, comb_f) = (gpu.buffer(t * hc), gpu.buffer(t * hc), gpu.buffer(t * hc * hc));
    let mut pre_previous: Option<Buffer> = None;
    let x = gpu.buffer(t * w);
    let qr = gpu.buffer(t * hp.q_rank);
    let q = gpu.buffer(t * heads * dim);
    let kv = gpu.buffer(t * dim);
    let attended = gpu.buffer(t * heads * dim);
    let low = gpu.buffer(t * hp.out_groups * hp.out_rank);
    let out = gpu.buffer(t * w);
    let index_q = gpu.buffer(t * hp.index_heads * hp.index_dim);
    let index_w = gpu.buffer(t * hp.index_heads);
    let mut sources: BTreeMap<usize, (Buffer, Buffer, usize)> = BTreeMap::new();   // kv source -> (rows, index keys, groups)
    let mut selections: BTreeMap<usize, (Buffer, usize)> = BTreeMap::new();         // index source -> (indices, k)
    let started = std::time::Instant::now();
    let layers = layers_limit.unwrap_or(hp.layers.len()).min(hp.layers.len());
    for layer in 0..layers {
        let name = |tensor: &str| format!("blk.{layer}.{tensor}.weight");
        let plan = hp.layers[layer].clone();
        // Engram: hashed table rows (gathered on the host), projected, gated into each copy.
        if let Some(module) = plan.engram {
            let engram = hp.engram.as_ref().unwrap();
            let (kind, dims, table) = files.tensor("per_layer_token_embd.weight");
            let bytes = row_bytes(kind, dims[0]);
            let (mut rows, mut gathered) = (Vec::new(), Vec::new());
            for i in 0..t {
                rows.clear();
                engram.rows(module, ids[i], &ids[i.saturating_sub(engram.ngram - 1)..i], i, &mut rows);
                for &row in &rows {
                    gathered.extend(decode(kind, &table[row as usize * bytes..(row as usize + 1) * bytes]));
                }
            }
            let gathered = gpu.values(&gathered);
            let engram_kv = gpu.buffer(t * (hc + 1) * w);
            gpu.linear(&name("engram_kv"), gathered.base(), t, engram_kv.base());
            let (gate, _, _) = gpu.weight(&name("engram_gate"));
            gpu.backend.engram_gate(stream.base(), engram_kv.base(), gate.base(), t as u64, hc as u64, w as u64, eps).unwrap();
        }
        // Attention sublayer.
        gpu.linear(&name("hc_attn_fn"), stream.base(), t, mixes.base());
        let (base, _, _) = gpu.weight(&name("hc_attn_base"));
        let (scale, _, _) = gpu.weight(&name("hc_attn_scale"));
        gpu.backend.hc_coefficients(mixes.base(), stream.base(), t as u64, hc as u64, w as u64, base.base(), scale.base(),
                                    hp.sinkhorn_iterations as u64, hp.hc_eps, eps, pre_a.base(), post_a.base(), comb_a.base()).unwrap();
        let pre = pre_previous.unwrap_or(identity);
        gpu.backend.hc_collapse(stream.base(), pre.base(), t as u64, hc as u64, w as u64, x.base()).unwrap();
        gpu.compare(&format!("hc_attn_pre-{layer}"), x, t * w);
        gpu.norm(&name("attn_norm"), x.base(), t, w, eps);
        gpu.compare(&format!("attn_norm-{layer}"), x, t * w);
        let freqs = if plan.compressed.is_some() { compressed_freqs } else { plain };
        gpu.linear(&name("attn_q_a"), x.base(), t, qr.base());
        gpu.norm(&name("attn_q_a_norm"), qr.base(), t, hp.q_rank, eps);
        gpu.compare(&format!("qr_norm-{layer}"), qr, t * hp.q_rank);
        gpu.linear(&name("attn_q_b"), qr.base(), t, q.base());
        gpu.backend.rope(q.base(), t as u64, heads as u64, dim as u64, rope as u64, &positions, &freqs, false).unwrap();
        gpu.linear(&name("attn_kv"), x.base(), t, kv.base());
        gpu.norm(&name("attn_kv_a_norm"), kv.base(), t, dim, eps);
        gpu.backend.rope(kv.base(), t as u64, 1, dim as u64, rope as u64, &positions, &freqs, false).unwrap();
        gpu.compare(&format!("q-{layer}"), q, t * heads * dim);
        gpu.compare(&format!("kv-{layer}"), kv, t * dim);
        let mut compressed = None;
        if let Some(c) = plan.compressed {
            if c.kv_source == layer {
                let groups = t / c.ratio;
                if groups > 0 {
                    let pooled_kv = gpu.buffer(t * dim);
                    gpu.linear(&name("attn_compressor_kv"), x.base(), t, pooled_kv.base());
                    let gate = (c.ratio > 1).then(|| {
                        let gate = gpu.buffer(t * dim);
                        gpu.linear(&name("attn_compressor_gate"), x.base(), t, gate.base());
                        gate
                    });
                    let latent = gpu.buffer(groups * dim);
                    gpu.backend.compress(pooled_kv.base(), gate.map(|g| g.base() as P), groups as u64, c.ratio as u64, dim as u64,
                                         latent.base()).unwrap();
                    gpu.norm(&name("attn_compressor_norm"), latent.base(), groups, dim, eps);
                    let keys = gpu.buffer(groups * hp.index_dim);
                    gpu.linear(&name("indexer.attn_k"), latent.base(), groups, keys.base());
                    gpu.norm(&name("indexer.k_norm"), keys.base(), groups, hp.index_dim, eps);
                    let starts: Vec<i32> = (0..groups as i32).map(|g| g * c.ratio as i32).collect();
                    let starts = gpu.values(&starts);
                    gpu.backend.rope(keys.base(), groups as u64, 1, hp.index_dim as u64, rope as u64, &starts, &freqs, false).unwrap();
                    gpu.backend.rope(latent.base(), groups as u64, 1, dim as u64, rope as u64, &starts, &freqs, false).unwrap();
                    sources.insert(layer, (latent, keys, groups));
                }
            }
            if c.index_source == layer {
                if let Some(&(_, keys, groups)) = sources.get(&c.kv_source) {
                    gpu.linear(&name("indexer.attn_q_b"), qr.base(), t, index_q.base());
                    gpu.backend.rope(index_q.base(), t as u64, hp.index_heads as u64, hp.index_dim as u64, rope as u64, &positions,
                                     &freqs, false).unwrap();
                    gpu.linear(&name("indexer.proj"), x.base(), t, index_w.base());
                    let visible: Vec<i32> = (0..t as i32).map(|i| (i + 1) / c.ratio as i32).collect();
                    let visible = gpu.values(&visible);
                    let scores = gpu.buffer(t * groups);
                    let scale = 1.0 / ((hp.index_dim as f32).sqrt() * (hp.index_heads as f32).sqrt());
                    gpu.backend.index_scores(index_q.base(), keys.base(), index_w.base(), visible.base(), t as u64, groups as u64,
                                             hp.index_heads as u64, hp.index_dim as u64, scale, scores.base()).unwrap();
                    let k = hp.index_top_k.min(groups);
                    let chosen = gpu.buffer(t * k);
                    gpu.backend.top_k(scores.base(), t as u64, groups as u64, k as u64, chosen.base()).unwrap();
                    selections.insert(layer, (chosen, k));
                }
            }
            if let (Some(&(rows, _, groups)), Some(&(chosen, k))) = (sources.get(&c.kv_source), selections.get(&c.index_source)) {
                compressed = Some((rows, groups, chosen, k));
            }
        }
        let (sinks, _, _) = gpu.weight(&name("attn_sinks"));
        let scale = 1.0 / (dim as f32).sqrt();
        gpu.backend.sparse_attention(q.base(), kv.base(), t as u64, positions.base(), hp.window as u64,
                                     compressed.map(|(rows, groups, _, _)| (rows.base() as P, groups as u64)),
                                     compressed.map(|(_, _, chosen, k)| (chosen.base() as P, k as u64)), sinks.base(), t as u64,
                                     heads as u64, dim as u64, scale, attended.base()).unwrap();
        gpu.backend.rope(attended.base(), t as u64, heads as u64, dim as u64, rope as u64, &positions, &freqs, true).unwrap();
        let (wo_a, kind_a, dims_a) = gpu.weight(&name("attn_output_a"));
        let group_width = heads * dim / hp.out_groups;
        let scratch = gpu.scratch;
        for g in 0..hp.out_groups {
            gpu.backend.linear(&wo_a, kind_a, hp.out_rank as u64, dims_a[0] as u64, (g * hp.out_rank) as u64,
                               attended.at(g * group_width), (heads * dim) as u64, t as u64, low.at(g * hp.out_rank),
                               (hp.out_groups * hp.out_rank) as u64, &scratch).unwrap();
        }
        gpu.linear(&name("attn_output_b"), low.base(), t, out.base());
        gpu.compare(&format!("attn_out-{layer}"), out, t * w);
        gpu.backend.hc_expand(out.base(), stream.base(), post_a.base(), comb_a.base(), t as u64, hc as u64, w as u64, stream2.base()).unwrap();
        // Feed-forward sublayer.
        gpu.linear(&name("hc_ffn_fn"), stream2.base(), t, mixes.base());
        let (base, _, _) = gpu.weight(&name("hc_ffn_base"));
        let (scale, _, _) = gpu.weight(&name("hc_ffn_scale"));
        gpu.backend.hc_coefficients(mixes.base(), stream2.base(), t as u64, hc as u64, w as u64, base.base(), scale.base(),
                                    hp.sinkhorn_iterations as u64, hp.hc_eps, eps, pre_f.base(), post_f.base(), comb_f.base()).unwrap();
        gpu.backend.hc_collapse(stream2.base(), pre_a.base(), t as u64, hc as u64, w as u64, x.base()).unwrap();
        gpu.compare(&format!("hc_ffn_pre-{layer}"), x, t * w);
        gpu.norm(&name("ffn_norm"), x.base(), t, w, eps);
        gpu.compare(&format!("ffn_norm-{layer}"), x, t * w);
        moe(&mut gpu, &hp, layer, x, t, out, all_rows);
        gpu.compare(&format!("ffn_out-{layer}"), out, t * w);
        gpu.backend.hc_expand(out.base(), stream2.base(), post_f.base(), comb_f.base(), t as u64, hc as u64, w as u64, stream.base()).unwrap();
        // The next attention collapses with this FFN's pre coefficients (read before the next
        // FFN overwrites them).
        pre_previous = Some(pre_f);
        println!("layer {layer} done at {:.1} s", started.elapsed().as_secs_f64());
    }
    if layers == hp.layers.len() {
        gpu.backend.hc_collapse(stream.base(), pre_previous.unwrap().base(), t as u64, hc as u64, w as u64, x.base()).unwrap();
        gpu.norm("output_norm.weight", x.base(), t, w, eps);
        let logits = gpu.buffer(kept * hp.vocabulary);
        gpu.linear("output.weight", x.at((t - kept) * w), kept, logits.base());
        gpu.compare("result_output", logits, kept * hp.vocabulary);
    }
    Ok(())
}

/// Routed experts (copied in per expert; host-only formats computed on the host) plus the shared
/// expert, summed into `out`.
fn moe(gpu: &mut Gpu, hp: &Hparams, layer: usize, x: Buffer, t: usize, out: Buffer, all_rows: Buffer) {
    let name = |tensor: &str| format!("blk.{layer}.{tensor}.weight");
    let (w, ff) = (hp.width, hp.expert_ff);
    let logits = gpu.buffer(t * hp.experts);
    gpu.linear(&name("ffn_gate_inp"), x.base(), t, logits.base());
    let (bias, _, _) = gpu.weight(&format!("blk.{layer}.exp_probs_b.bias"));
    let ids = gpu.buffer(t * hp.experts_used);
    let weights = gpu.buffer(t * hp.experts_used);
    let gating = match hp.gating { Gating::Softmax => 0, Gating::Sigmoid => 1, Gating::SqrtSoftplus => 2 };
    gpu.backend.route(logits.base(), Some(bias.base()), t as u64, hp.experts as u64, hp.experts_used as u64, gating,
                      hp.route_normalize, hp.route_scale, ids.base(), weights.base()).unwrap();
    let mut routes = vec![0i32; t * hp.experts_used];
    gpu.backend.read_buffer(&ids, 0, &mut routes).unwrap();
    let route_weights = gpu.read(weights, 0, t * hp.experts_used);
    let mut by_expert: BTreeMap<usize, Vec<(i32, f32)>> = BTreeMap::new();
    for (slot, &expert) in routes.iter().enumerate() {
        by_expert.entry(expert as usize).or_default().push(((slot / hp.experts_used) as i32, route_weights[slot]));
    }
    // routed sum into `out`, then the shared expert added
    gpu.backend.write_buffer(&out, 0, &vec![0f32; t * w]).unwrap();
    let (gate_kind, _, gate_all) = gpu.files.tensor(&name("ffn_gate_exps"));
    let (up_kind, _, up_all) = gpu.files.tensor(&name("ffn_up_exps"));
    let (down_kind, _, down_all) = gpu.files.tensor(&name("ffn_down_exps"));
    let (gate_bytes, down_bytes) = (ff * row_bytes(gate_kind, w), w * row_bytes(down_kind, ff));
    let on_gpu = GPU_KINDS.contains(&gate_kind) && GPU_KINDS.contains(&up_kind) && GPU_KINDS.contains(&down_kind);
    let x_host = if on_gpu { Vec::new() } else { gpu.read(x, 0, t * w) };
    for (expert, users) in by_expert {
        let n = users.len();
        let rows: Vec<i32> = users.iter().map(|u| u.0).collect();
        let user_weights: Vec<f32> = users.iter().map(|u| u.1).collect();
        let rows_buffer = gpu.values(&rows);
        let y = if on_gpu {
            let slice = |all: &[u8], bytes: usize| all[expert * bytes..(expert + 1) * bytes].to_vec();
            let gate = gpu.backend.upload_bytes(&slice(gate_all, gate_bytes)).unwrap();
            let up = gpu.backend.upload_bytes(&slice(up_all, gate_bytes)).unwrap();
            let down = gpu.backend.upload_bytes(&slice(down_all, down_bytes)).unwrap();
            let xe = gpu.buffer(n * w);
            gpu.backend.gather(x.base(), t as u64, rows_buffer.base(), n as u64, w as u64, xe.base()).unwrap();
            let (g, u, h, y) = (gpu.buffer(n * ff), gpu.buffer(n * ff), gpu.buffer(n * ff), gpu.buffer(n * w));
            let scratch = gpu.scratch;
            gpu.backend.linear(&gate, gate_kind, ff as u64, w as u64, 0, xe.base(), 0, n as u64, g.base(), 0, &scratch).unwrap();
            gpu.backend.linear(&up, up_kind, ff as u64, w as u64, 0, xe.base(), 0, n as u64, u.base(), 0, &scratch).unwrap();
            let rw = gpu.values(&user_weights);
            gpu.backend.swiglu(g.base(), u.base(), n as u64, ff as u64, hp.layers[layer].clamp_routed, Some(rw.base()), h.base()).unwrap();
            gpu.backend.linear(&down, down_kind, w as u64, ff as u64, 0, h.base(), 0, n as u64, y.base(), 0, &scratch).unwrap();
            for buffer in [gate, up, down, xe, g, u, h, rw] {
                release(gpu, buffer);
            }
            y
        } else {
            // host: exact decode, F64 dots, as the reference
            let gate = decode(gate_kind, &gate_all[expert * gate_bytes..(expert + 1) * gate_bytes]);
            let up = decode(up_kind, &up_all[expert * gate_bytes..(expert + 1) * gate_bytes]);
            let down = decode(down_kind, &down_all[expert * down_bytes..(expert + 1) * down_bytes]);
            let clamp = hp.layers[layer].clamp_routed;
            let mut y = vec![0f32; n * w];
            for (k, &(row, weight)) in users.iter().enumerate() {
                let input = &x_host[row as usize * w..(row as usize + 1) * w];
                let hidden: Vec<f32> = (0..ff).map(|r| {
                    let dot = |m: &[f32]| m[r * w..(r + 1) * w].iter().zip(input).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32;
                    let (mut g, mut u) = (dot(&gate), dot(&up));
                    if clamp > 0.0 {
                        g = g.min(clamp);
                        u = u.clamp(-clamp, clamp);
                    }
                    g / (1.0 + (-g).exp()) * u * weight
                }).collect();
                for d in 0..w {
                    y[k * w + d] = down[d * ff..(d + 1) * ff].iter().zip(&hidden).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32;
                }
            }
            gpu.values(&y)
        };
        gpu.backend.scatter_add(y.base(), rows_buffer.base(), n as u64, w as u64, out.base(), t as u64).unwrap();
        release(gpu, y);
        release(gpu, rows_buffer);
    }
    gpu.compare(&format!("ffn_moe_out-{layer}"), out, t * w);
    let (g, u, h, shared) = (gpu.buffer(t * ff), gpu.buffer(t * ff), gpu.buffer(t * ff), gpu.buffer(t * w));
    gpu.linear(&name("ffn_gate_shexp"), x.base(), t, g.base());
    gpu.linear(&name("ffn_up_shexp"), x.base(), t, u.base());
    gpu.backend.swiglu(g.base(), u.base(), t as u64, ff as u64, hp.layers[layer].clamp_shared, None, h.base()).unwrap();
    gpu.linear(&name("ffn_down_shexp"), h.base(), t, shared.base());
    gpu.compare(&format!("ffn_shexp-{layer}"), shared, t * w);
    gpu.backend.scatter_add(shared.base(), all_rows.base(), t as u64, w as u64, out.base(), t as u64).unwrap();
    for buffer in [logits, ids, weights, g, u, h, shared] {
        release(gpu, buffer);
    }
}

fn release(gpu: &mut Gpu, buffer: Buffer) {
    gpu.backend.release_buffer(buffer).unwrap();
}
