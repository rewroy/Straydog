//! A CPU F32 reference of Qwen3.5 MoE (`qwen35moe`) for validating the GPU path: the reference
//! implementation's forward pass, written plainly, over the GGUF's own weights. Development only:
//! the file is read without authentication.
//!
//! q35_reference MODEL.gguf DUMP_DIR [--layers N] [--routes] [--save DIR] [--generate N]
//!
//! DUMP_DIR: the reference run's dumps (q35dump: four int64 extents then F32 values, named like
//! `ffn_out-7.f32`) and its `tokens.txt` (the prompt's ids, then its greedy continuation). Every
//! layer's tensors are compared with the dumps; with --routes the MoE takes the reference's
//! expert choices (so that a near tie does not send the comparison down another path);
//! --generate N decodes N tokens greedily (recomputing the sequence) and compares them with the
//! reference's; --save writes every compared tensor of this run in the dump format.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use straydog_engine::arch::qwen35moe::{Hparams, Mixer};
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

struct Files {
    map: Mmap,
    tensors: HashMap<String, TensorInfo>,
    header: ModelHeader,
}

impl Files {
    fn open(path: &Path) -> Result<Files, String> {
        let file = File::open(path).map_err(|exc| format!("{}: {exc}", path.display()))?;
        // SAFETY: a read-only mapping of a file this process never writes.
        let map = unsafe { Mmap::map(&file) }.map_err(|exc| exc.to_string())?;
        let header = straydog_gguf::inspect(&mut std::io::Cursor::new(&map[..])).map_err(|exc| exc.to_string())?;
        let tensors = header.tensors.iter().map(|tensor| (tensor.name.clone(), tensor.clone())).collect();
        Ok(Files { map, tensors, header })
    }

    fn tensor(&self, name: &str) -> Tensor<'_> {
        let info = self.tensors.get(name).unwrap_or_else(|| panic!("tensor {name} is missing"));
        let start = info.offset as usize;
        Tensor {
            kind: info.quant_type,
            dims: info.dimensions.iter().map(|&dim| dim as usize).collect(),
            data: &self.map[start..start + info.bytes as usize],
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

/// `out[t][r] = W[first + r] . x[t]` for `rows` rows; `x` holds `tokens` rows of the matrix's
/// width. Rows are decoded in chunks, in parallel.
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
                        part[t * count + r] = dot(&weights[r * width..(r + 1) * width], input);
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

/// RMS norm of each `weight.len()`-wide row, scaled by the weight.
fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    for row in x.chunks_mut(weight.len()) {
        let mean = row.iter().map(|v| *v as f64 * *v as f64).sum::<f64>() / row.len() as f64;
        let scale = 1.0 / (mean + eps as f64).sqrt();
        for (v, w) in row.iter_mut().zip(weight) {
            *v = (*v as f64 * scale) as f32 * w;
        }
    }
}

/// `x / sqrt(sum(x^2) + eps)` of each `width`-wide row (the reference's L2 norm).
fn l2_norm(x: &mut [f32], width: usize, eps: f32) {
    for row in x.chunks_mut(width) {
        let sum = row.iter().map(|v| *v as f64 * *v as f64).sum::<f64>();
        let scale = 1.0 / (sum + eps as f64).sqrt();
        row.iter_mut().for_each(|v| *v = (*v as f64 * scale) as f32);
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Rotate pairs `(i, i + n/2)` of the first `2 * freqs.len()` values of `x` by `position`.
fn rotate(x: &mut [f32], freqs: &[f64], position: usize) {
    let half = freqs.len();
    for (i, freq) in freqs.iter().enumerate() {
        let (sin, cos) = (position as f64 * freq).sin_cos();
        let (a, b) = (x[i] as f64, x[i + half] as f64);
        x[i] = (a * cos - b * sin) as f32;
        x[i + half] = (a * sin + b * cos) as f32;
    }
}

struct Model<'a> {
    files: &'a Files,
    hp: Hparams,
    /// The reference run's experts per layer, [token][slot], when routing is forced to them.
    routes: HashMap<usize, Vec<Vec<usize>>>,
}

impl Model<'_> {
    fn name(layer: usize, tensor: &str) -> String {
        format!("blk.{layer}.{tensor}.weight")
    }

    /// Gated DeltaNet over `x` (normalized input, `tokens` rows) from a zero state.
    fn delta_net(&self, layer: usize, x: &[f32], tokens: usize, dump: Option<&Path>) -> Vec<f32> {
        let hp = &self.hp;
        let (dim, k_heads, v_heads) = (hp.state_dim, hp.k_heads, hp.v_heads);
        let channels = hp.conv_channels();
        let qkv = linear(&self.files.tensor(&Model::name(layer, "attn_qkv")), x, tokens);
        let z = linear(&self.files.tensor(&Model::name(layer, "attn_gate")), x, tokens);
        let beta: Vec<f32> = linear(&self.files.tensor(&Model::name(layer, "ssm_beta")), x, tokens).into_iter().map(sigmoid).collect();
        let alpha = linear(&self.files.tensor(&Model::name(layer, "ssm_alpha")), x, tokens);
        let dt = self.files.vector(&format!("blk.{layer}.ssm_dt.bias"));
        let a = self.files.vector(&format!("blk.{layer}.ssm_a"));
        // Per token and value head: the state's decay exp(a * softplus(alpha + dt)).
        let decay: Vec<f32> = (0..tokens * v_heads).map(|i| (a[i % v_heads] * softplus(alpha[i] + dt[i % v_heads])).exp()).collect();
        // Causal depthwise convolution over time (zeros before the sequence), then SiLU.
        let kernel = self.files.vector(&format!("blk.{layer}.ssm_conv1d.weight"));
        let width = hp.conv_kernel;
        let mut conv = vec![0f32; tokens * channels];
        for t in 0..tokens {
            for c in 0..channels {
                let mut sum = 0f64;
                for j in 0..width {
                    // kernel tap j reads input t - (width - 1) + j
                    let source = t as isize - (width as isize - 1) + j as isize;
                    if source >= 0 {
                        sum += qkv[source as usize * channels + c] as f64 * kernel[c * width + j] as f64;
                    }
                }
                conv[t * channels + c] = silu(sum as f32);
            }
        }
        compare(dump, &format!("conv_output_silu-{layer}"), &conv);
        let key_width = k_heads * dim;
        let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..tokens {
            let row = &conv[t * channels..(t + 1) * channels];
            q.extend_from_slice(&row[..key_width]);
            k.extend_from_slice(&row[key_width..2 * key_width]);
            v.extend_from_slice(&row[2 * key_width..]);
        }
        l2_norm(&mut q, dim, hp.norm_eps);
        l2_norm(&mut k, dim, hp.norm_eps);
        // The delta rule per value head h, reading key head h % k_heads: S[i][j] over key i and
        // value j; S *= decay; d = (v - S^T k) * beta; S += k d^T; out = S^T q / sqrt(dim).
        let scale = 1.0 / (dim as f64).sqrt();
        let mut out = vec![0f32; tokens * v_heads * dim];
        let mut states = vec![vec![0f64; dim * dim]; v_heads];
        for t in 0..tokens {
            for (h, state) in states.iter_mut().enumerate() {
                let kh = h % k_heads;
                let qv = &q[t * key_width + kh * dim..t * key_width + (kh + 1) * dim];
                let kv = &k[t * key_width + kh * dim..t * key_width + (kh + 1) * dim];
                let vv = &v[t * v_heads * dim + h * dim..t * v_heads * dim + (h + 1) * dim];
                let g = decay[t * v_heads + h] as f64;
                state.iter_mut().for_each(|s| *s *= g);
                let b = beta[t * v_heads + h] as f64;
                let delta: Vec<f64> = (0..dim).map(|j| {
                    let sk: f64 = (0..dim).map(|i| state[i * dim + j] * kv[i] as f64).sum();
                    (vv[j] as f64 - sk) * b
                }).collect();
                for i in 0..dim {
                    for j in 0..dim {
                        state[i * dim + j] += kv[i] as f64 * delta[j];
                    }
                }
                for j in 0..dim {
                    let o: f64 = (0..dim).map(|i| state[i * dim + j] * qv[i] as f64).sum();
                    out[t * v_heads * dim + h * dim + j] = (o * scale) as f32;
                }
            }
        }
        compare(dump, &format!("attn_output-{layer}"), &out);
        // RMS norm per value head, gated by SiLU of z.
        rms_norm(&mut out, &self.files.vector(&Model::name(layer, "ssm_norm")), hp.norm_eps);
        for (o, zv) in out.iter_mut().zip(&z) {
            *o *= silu(*zv);
        }
        linear(&self.files.tensor(&Model::name(layer, "ssm_out")), &out, tokens)
    }

    /// Gated full attention over `x` (normalized input, `tokens` rows at positions 0..).
    fn attention(&self, layer: usize, x: &[f32], tokens: usize, dump: Option<&Path>) -> Vec<f32> {
        let hp = &self.hp;
        let (heads, kv_heads, dim) = (hp.heads, hp.kv_heads, hp.head_dim);
        let full = linear(&self.files.tensor(&Model::name(layer, "attn_q")), x, tokens);
        let mut keys = linear(&self.files.tensor(&Model::name(layer, "attn_k")), x, tokens);
        let values = linear(&self.files.tensor(&Model::name(layer, "attn_v")), x, tokens);
        // Each head's projection holds its query, then its gate.
        let (mut queries, mut gates) = (Vec::with_capacity(tokens * heads * dim), Vec::with_capacity(tokens * heads * dim));
        for t in 0..tokens {
            for h in 0..heads {
                let at = t * heads * 2 * dim + h * 2 * dim;
                queries.extend_from_slice(&full[at..at + dim]);
                gates.extend_from_slice(&full[at + dim..at + 2 * dim]);
            }
        }
        rms_norm(&mut queries, &self.files.vector(&Model::name(layer, "attn_q_norm")), hp.norm_eps);
        rms_norm(&mut keys, &self.files.vector(&Model::name(layer, "attn_k_norm")), hp.norm_eps);
        let freqs = hp.rope_frequencies();
        for t in 0..tokens {
            for h in 0..heads {
                rotate(&mut queries[(t * heads + h) * dim..(t * heads + h + 1) * dim], &freqs, t);
            }
            for h in 0..kv_heads {
                rotate(&mut keys[(t * kv_heads + h) * dim..(t * kv_heads + h + 1) * dim], &freqs, t);
            }
        }
        compare(dump, &format!("Qcur-{layer}"), &queries);
        compare(dump, &format!("Kcur-{layer}"), &keys);
        let scale = 1.0 / (dim as f64).sqrt();
        let group = heads / kv_heads;
        let mut out = vec![0f32; tokens * heads * dim];
        for t in 0..tokens {
            for h in 0..heads {
                let kh = h / group;
                let q = &queries[(t * heads + h) * dim..(t * heads + h + 1) * dim];
                let scores: Vec<f64> = (0..=t).map(|s| {
                    let key = &keys[(s * kv_heads + kh) * dim..(s * kv_heads + kh + 1) * dim];
                    q.iter().zip(key).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * scale
                }).collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for d in 0..dim {
                    let value: f64 = (0..=t).map(|s| weights[s] * values[(s * kv_heads + kh) * dim + d] as f64).sum();
                    out[(t * heads + h) * dim + d] = (value / sum) as f32;
                }
            }
        }
        compare(dump, &format!("attn_pregate-{layer}"), &out);
        for (o, g) in out.iter_mut().zip(&gates) {
            *o *= sigmoid(*g);
        }
        linear(&self.files.tensor(&Model::name(layer, "attn_output")), &out, tokens)
    }

    /// The MoE: softmax routing, the top experts' weights normalized, SwiGLU experts, plus the
    /// shared expert gated by a sigmoid. Returns (routed, gated shared).
    fn moe(&self, layer: usize, x: &[f32], tokens: usize, dump: Option<&Path>) -> (Vec<f32>, Vec<f32>) {
        let hp = &self.hp;
        let (width, ff, used) = (hp.width, hp.expert_ff, hp.experts_used);
        let logits = linear(&self.files.tensor(&Model::name(layer, "ffn_gate_inp")), x, tokens);
        let mut users: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        let mut mismatches = 0;
        for t in 0..tokens {
            let row = &logits[t * hp.experts..(t + 1) * hp.experts];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let exps: Vec<f64> = row.iter().map(|v| (*v as f64 - max).exp()).collect();
            let total: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|e| e / total).collect();
            let mut order: Vec<usize> = (0..hp.experts).collect();
            order.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]));
            let mut chosen: Vec<usize> = order[..used].to_vec();
            if let Some(forced) = self.routes.get(&layer) {
                let mut ours = chosen.clone();
                let mut theirs = forced[t].clone();
                ours.sort_unstable();
                theirs.sort_unstable();
                mismatches += usize::from(ours != theirs);
                chosen = forced[t].clone();
            }
            let sum: f64 = chosen.iter().map(|&e| probs[e]).sum::<f64>().max(6.103515625e-5);
            for &e in &chosen {
                users.entry(e).or_default().push((t, (probs[e] / sum) as f32));
            }
        }
        if mismatches > 0 {
            println!("  layer {layer}: {mismatches} of {tokens} tokens route differently than the reference");
        }
        let (gate, up, down) = (self.files.tensor(&Model::name(layer, "ffn_gate_exps")), self.files.tensor(&Model::name(layer, "ffn_up_exps")),
                                self.files.tensor(&Model::name(layer, "ffn_down_exps")));
        let mut routed = vec![0f32; tokens * width];
        for (e, users) in users {
            let n = users.len();
            let inputs: Vec<f32> = users.iter().flat_map(|&(t, _)| x[t * width..(t + 1) * width].iter().copied()).collect();
            let g = linear_rows(&gate, e * ff, ff, &inputs, n);
            let u = linear_rows(&up, e * ff, ff, &inputs, n);
            let hidden: Vec<f32> = (0..n * ff).map(|i| silu(g[i]) * u[i]).collect();
            let y = linear_rows(&down, e * width, width, &hidden, n);
            for (k, &(t, weight)) in users.iter().enumerate() {
                for d in 0..width {
                    routed[t * width + d] += weight * y[k * width + d];
                }
            }
        }
        compare(dump, &format!("ffn_moe_out-{layer}"), &routed);
        let g = linear(&self.files.tensor(&Model::name(layer, "ffn_gate_shexp")), x, tokens);
        let u = linear(&self.files.tensor(&Model::name(layer, "ffn_up_shexp")), x, tokens);
        let hidden: Vec<f32> = g.iter().zip(&u).map(|(&gv, &uv)| silu(gv) * uv).collect();
        let mut shared = linear(&self.files.tensor(&Model::name(layer, "ffn_down_shexp")), &hidden, tokens);
        let gate_vector = self.files.vector(&Model::name(layer, "ffn_gate_inp_shexp"));
        for t in 0..tokens {
            let gate = sigmoid(dot(&gate_vector, &x[t * width..(t + 1) * width]));
            shared[t * width..(t + 1) * width].iter_mut().for_each(|v| *v *= gate);
        }
        compare(dump, &format!("ffn_shexp_gated-{layer}"), &shared);
        (routed, shared)
    }

    /// The final hidden rows (normalized) of `ids`, comparing each layer with the dumps.
    fn forward(&self, ids: &[u32], layers: usize, dump: Option<&Path>) -> Vec<f32> {
        let hp = &self.hp;
        let tokens = ids.len();
        let embed = self.files.tensor("token_embd.weight");
        let row_bytes = embed.row_bytes();
        let mut stream: Vec<f32> = ids.iter()
            .flat_map(|&id| decode(embed.kind, &embed.data[id as usize * row_bytes..(id as usize + 1) * row_bytes]))
            .collect();
        let started = std::time::Instant::now();
        for layer in 0..layers {
            let mut x = stream.clone();
            rms_norm(&mut x, &self.files.vector(&Model::name(layer, "attn_norm")), hp.norm_eps);
            compare(dump, &format!("attn_norm-{layer}"), &x);
            let mixed = match hp.mixers[layer] {
                Mixer::DeltaNet => {
                    let out = self.delta_net(layer, &x, tokens, dump);
                    compare(dump, &format!("linear_attn_out-{layer}"), &out);
                    out
                }
                Mixer::Attention => {
                    let out = self.attention(layer, &x, tokens, dump);
                    compare(dump, &format!("attn_output-{layer}"), &out);
                    out
                }
            };
            stream.iter_mut().zip(&mixed).for_each(|(s, m)| *s += m);
            compare(dump, &format!("attn_residual-{layer}"), &stream);
            let mut x = stream.clone();
            rms_norm(&mut x, &self.files.vector(&Model::name(layer, "post_attention_norm")), hp.norm_eps);
            compare(dump, &format!("attn_post_norm-{layer}"), &x);
            let (routed, shared) = self.moe(layer, &x, tokens, dump);
            let out: Vec<f32> = routed.iter().zip(&shared).map(|(a, b)| a + b).collect();
            compare(dump, &format!("ffn_out-{layer}"), &out);
            stream.iter_mut().zip(&out).for_each(|(s, o)| *s += o);
            compare(dump, &format!("l_out-{layer}"), &stream);
            if dump.is_some() {
                println!("layer {layer} ({:?}) done at {:.1} s", hp.mixers[layer], started.elapsed().as_secs_f64());
            }
        }
        rms_norm(&mut stream, &self.files.vector("output_norm.weight"), hp.norm_eps);
        stream
    }
}

/// Where `--save` writes this run's tensors, in the dump format, for checking other paths.
static SAVE: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();

/// A dump of the reference run: its extents and values.
fn read_dump(dir: &Path, name: &str) -> Option<([i64; 4], Vec<f32>)> {
    let bytes = std::fs::read(dir.join(format!("{name}.f32"))).ok()?;
    let extents: Vec<i64> = bytes[..32].chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect();
    let values = bytes[32..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    Some(([extents[0], extents[1], extents[2], extents[3]], values))
}

/// Compare a computed tensor (row-major, the reference's extents) with a dump of the reference run.
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
    let Some((_, reference)) = read_dump(dir, name) else {
        println!("  {name:<22} (no dump)");
        return;
    };
    if reference.len() != values.len() {
        println!("  {name:<22} size {} vs dump {}", values.len(), reference.len());
        return;
    }
    let (mut diff, mut norm, mut max) = (0f64, 0f64, 0f64);
    for (a, b) in values.iter().zip(&reference) {
        diff += (*a as f64 - *b as f64).powi(2);
        norm += (*b as f64).powi(2);
        max = max.max((*a as f64 - *b as f64).abs());
    }
    println!("  {name:<22} rel {:.2e}  max {:.2e}", (diff / norm.max(1e-30)).sqrt(), max);
}

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut option = |name: &str| args.iter().position(|arg| arg == name).map(|at| {
        let value = args[at + 1].clone();
        args.drain(at..at + 2);
        value
    });
    let last_layer: Option<usize> = option("--layers").map(|n| n.parse().unwrap());
    let save = option("--save").map(PathBuf::from);
    let generate: usize = option("--generate").map_or(0, |n| n.parse().unwrap());
    let forced = args.iter().position(|arg| arg == "--routes").map(|at| args.remove(at)).is_some();
    if let Some(dir) = &save {
        std::fs::create_dir_all(dir).map_err(|exc| exc.to_string())?;
    }
    SAVE.set(save).unwrap();
    let [model_path, dump_dir] = &args[..] else {
        return Err("usage: q35_reference MODEL.gguf DUMP_DIR [--layers N] [--routes] [--save DIR] [--generate N]".into());
    };
    let dump = Path::new(dump_dir);
    let files = Files::open(Path::new(model_path))?;
    let hp = Hparams::from_header(&files.header)?;
    let text = std::fs::read_to_string(dump.join("tokens.txt")).map_err(|exc| exc.to_string())?;
    let mut lines = text.lines();
    let parse = |line: Option<&str>| -> Vec<u32> { line.unwrap_or("").split_whitespace().map(|id| id.parse().unwrap()).collect() };
    let (ids, continuation) = (parse(lines.next()), parse(lines.next()));
    let tokens = ids.len();
    println!("{} layers, width {}, {} tokens", hp.layers(), hp.width, tokens);
    let mut routes = HashMap::new();
    if forced {
        for layer in 0..hp.layers() {
            if let Some((extents, values)) = read_dump(dump, &format!("ffn_moe_topk-{layer}")) {
                let used = extents[0] as usize;
                routes.insert(layer, values.chunks_exact(used).map(|row| row.iter().map(|&e| e as usize).collect()).collect());
            }
        }
    }
    let model = Model { files: &files, hp, routes };
    let layers = last_layer.unwrap_or(model.hp.layers()).min(model.hp.layers());
    let hidden = model.forward(&ids, layers, Some(dump));
    if layers < model.hp.layers() {
        return Ok(());
    }
    compare(Some(dump), "result_norm", &hidden);
    let width = model.hp.width;
    let logits = linear(&files.tensor("output.weight"), &hidden, tokens);
    compare(Some(dump), "result_output", &logits);
    if let Some((_, reference)) = read_dump(dump, "result_output") {
        let vocabulary = model.hp.vocabulary;
        for t in 0..tokens.min(reference.len() / vocabulary) {
            let ours = &logits[t * vocabulary..(t + 1) * vocabulary];
            let theirs = &reference[t * vocabulary..(t + 1) * vocabulary];
            let softmax = |v: &[f32]| {
                let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum: f64 = v.iter().map(|x| (*x as f64 - max).exp()).sum();
                v.iter().map(|x| (*x as f64 - max).exp() / sum).collect::<Vec<f64>>()
            };
            let (p, q) = (softmax(theirs), softmax(ours));
            let kl: f64 = p.iter().zip(&q).filter(|(a, _)| **a > 0.0).map(|(a, b)| a * (a / b.max(1e-300)).ln()).sum();
            println!("position {t}: KL(ref||ours) {kl:.3e}, top-1 {} vs {}", argmax(theirs), argmax(ours));
        }
    }
    // The last position's leading logits, to tell a near tie from a real disagreement.
    let vocabulary = model.hp.vocabulary;
    let last = &logits[(tokens - 1) * vocabulary..tokens * vocabulary];
    let mut order: Vec<usize> = (0..vocabulary).collect();
    order.sort_by(|&a, &b| last[b].total_cmp(&last[a]));
    let leading: Vec<String> = order[..5].iter().map(|&t| format!("{t}:{:.3}", last[t])).collect();
    println!("last position's leading logits: {}", leading.join("  "));
    if generate > 0 {
        // Greedy continuation, the whole sequence recomputed per token (no state carried over).
        let mut sequence = ids.clone();
        let mut next = argmax(&logits[(tokens - 1) * model.hp.vocabulary..]) as u32;
        let mut ours = Vec::new();
        for _ in 0..generate {
            ours.push(next);
            sequence.push(next);
            let model = Model { files: &files, hp: model.hp.clone(), routes: HashMap::new() };
            let hidden = model.forward(&sequence, model.hp.layers(), None);
            let last = &hidden[(sequence.len() - 1) * width..];
            next = argmax(&linear(&files.tensor("output.weight"), last, 1)) as u32;
        }
        println!("greedy:    {ours:?}");
        println!("reference: {continuation:?}");
        let same = ours.iter().zip(&continuation).take_while(|(a, b)| a == b).count();
        println!("{same} of {} tokens match", ours.len().min(continuation.len()));
    }
    Ok(())
}
