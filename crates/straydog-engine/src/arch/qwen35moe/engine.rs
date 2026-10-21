//! Qwen3.5 MoE on the GPU, composed from the generic ops: dense weights resident, routed experts in
//! the expert store's tiers (VRAM slots, a host tier, the files; src/experts.rs), and the sequence
//! state a pass continues (each DeltaNet's convolution history and recurrent state, each attention
//! layer's K/V). One engine runs a range of layers on one device; engines of consecutive ranges
//! hand the residual stream over ([`forward_split`]). Checked against the CPU reference by
//! examples/q35_chat (with Q35_COMPARE naming a directory of the reference's tensors, each layer's
//! are compared as the pass goes).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::c_void;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use super::{Hparams, Mixer};
use crate::arch::Tensors;
use crate::experts::{Contents, Counts, DeviceExperts, Layout, Reader, Role};
use crate::formats;
use crate::gpu::{Backend, Result};
use crate::ops::Buffer;

type P = *const c_void;
type M = *mut c_void;

/// Expert reads in flight (drive threads of the store's reader).
const READERS: usize = 32;
/// Rows whose routes run in one launch over the experts' slots (decode, short batches).
const SLOT_ROWS: usize = 4;
/// Experts a prefill wave holds at once.
const WAVE: usize = 32;

pub struct Settings {
    /// Longest sequence (positions) the state holds.
    pub context: usize,
    /// Most tokens one pass takes.
    pub batch: usize,
    /// Experts the VRAM cache holds (worth so many of the largest).
    pub expert_slots: usize,
    /// Bytes of host tier (page-locked memory) for experts VRAM evicts.
    pub host_expert_bytes: u64,
    pub scratch_bytes: u64,
    /// The layers this engine runs: the first embeds the tokens, the last has the head.
    pub layers: std::ops::Range<usize>,
    /// Copies of the DeltaNets' state a session can save and restore ([`Engine::checkpoint`]).
    pub checkpoints: usize,
}

#[derive(Clone)]
struct Dense {
    buffer: Buffer,
    kind: u32,
    dims: Vec<usize>,
}

/// A layer's sequence state.
enum State {
    DeltaNet { history: Buffer, recurrent: Buffer },
    Attention { keys: Buffer, values: Buffer },
}

/// A copy of the DeltaNets' state (each one's history, then its recurrent state, in layer order)
/// and the position it was taken at.
struct Saved {
    buffers: Vec<Buffer>,
    position: usize,
}

/// Work buffers sized for a batch.
#[derive(Clone, Copy)]
struct Work {
    stream: Buffer,
    x: Buffer,
    mixed: Buffer,
    out: Buffer,
    qkv: Buffer,
    z: Buffer,
    beta: Buffer,
    alpha: Buffer,
    core: Buffer,
    normed: Buffer,
    q_full: Buffer,
    q: Buffer,
    gate: Buffer,
    k: Buffer,
    v: Buffer,
    attended: Buffer,
    router: Buffer,
    ids: Buffer,
    weights: Buffer,
    shared_gate: Buffer,
    xe: Buffer,
    ffn_gate: Buffer,
    ffn_up: Buffer,
    hidden: Buffer,
    y: Buffer,
    expert_rows: Buffer,
    expert_weights: Buffer,
    positions: Buffer,
    head: Buffer,
}

pub struct Engine<'a, T: Tensors> {
    hp: Rc<Hparams>,
    tensors: &'a T,
    settings: Settings,
    gpu: Backend,
    dense: HashMap<String, Dense>,
    scratch: Buffer,
    work: Work,
    states: BTreeMap<usize, State>,
    saved: Vec<Saved>,
    experts: DeviceExperts,
    frequencies: Buffer,
    position: usize,
    /// Q35_COMPARE: the reference's tensors to compare each layer's with (development).
    compare: Option<PathBuf>,
}

fn row_bytes(kind: u32, cols: usize) -> Result<usize> {
    formats::matrix_bytes(kind, 1, cols).ok_or_else(|| format!("unsupported tensor type {kind}"))
}

fn name(layer: usize, tensor: &str) -> String {
    format!("blk.{layer}.{tensor}.weight")
}

impl<'a, T: Tensors> Engine<'a, T> {
    pub fn new(hp: Hparams, tensors: &'a T, device: usize, budget: u64, settings: Settings) -> Result<Engine<'a, T>> {
        let range = settings.layers.clone();
        if range.is_empty() || range.end > hp.layers() {
            return Err("qwen35moe engine layers outside the model".into());
        }
        let mut gpu = Backend::new(device, budget)?;
        let scratch = gpu.buffer(settings.scratch_bytes)?;
        let b = settings.batch;
        let floats = |gpu: &mut Backend, count: usize| gpu.buffer((count.max(1) * 4) as u64);
        let (w, k) = (hp.width, hp.experts_used);
        let v_width = hp.v_heads * hp.state_dim;
        let attention_width = hp.heads * hp.head_dim;
        let kv_width = hp.kv_heads * hp.head_dim;
        let ff = hp.expert_ff.max(hp.shared_ff);
        let work = Work {
            stream: floats(&mut gpu, b * w)?,
            x: floats(&mut gpu, b * w)?,
            mixed: floats(&mut gpu, b * w)?,
            out: floats(&mut gpu, b * w)?,
            qkv: floats(&mut gpu, b * hp.conv_channels())?,
            z: floats(&mut gpu, b * v_width)?,
            beta: floats(&mut gpu, b * hp.v_heads)?,
            alpha: floats(&mut gpu, b * hp.v_heads)?,
            core: floats(&mut gpu, b * v_width)?,
            normed: floats(&mut gpu, b * v_width)?,
            q_full: floats(&mut gpu, b * 2 * attention_width)?,
            q: floats(&mut gpu, b * attention_width)?,
            gate: floats(&mut gpu, b * attention_width)?,
            k: floats(&mut gpu, b * kv_width)?,
            v: floats(&mut gpu, b * kv_width)?,
            attended: floats(&mut gpu, b * attention_width)?,
            router: floats(&mut gpu, b * hp.experts)?,
            ids: floats(&mut gpu, b * k)?,
            weights: floats(&mut gpu, b * k)?,
            shared_gate: floats(&mut gpu, b)?,
            xe: floats(&mut gpu, b * w)?,
            ffn_gate: floats(&mut gpu, b * k.max(1) * ff)?,
            ffn_up: floats(&mut gpu, b * ff)?,
            hidden: floats(&mut gpu, b * k.max(1) * ff)?,
            y: floats(&mut gpu, b * w)?,
            expert_rows: floats(&mut gpu, b)?,
            expert_weights: floats(&mut gpu, b)?,
            positions: floats(&mut gpu, b)?,
            head: floats(&mut gpu, if range.end == hp.layers() { hp.vocabulary } else { 1 })?,
        };
        let mut states = BTreeMap::new();
        for layer in range.clone() {
            let state = match hp.mixers[layer] {
                Mixer::DeltaNet => State::DeltaNet {
                    history: floats(&mut gpu, (hp.conv_kernel - 1) * hp.conv_channels())?,
                    recurrent: floats(&mut gpu, hp.v_heads * hp.state_dim * hp.state_dim)?,
                },
                Mixer::Attention => State::Attention {
                    keys: floats(&mut gpu, settings.context * kv_width)?,
                    values: floats(&mut gpu, settings.context * kv_width)?,
                },
            };
            states.insert(layer, state);
        }
        let mut saved = Vec::new();
        for _ in 0..settings.checkpoints {
            let mut buffers = Vec::new();
            for state in states.values() {
                if let State::DeltaNet { history, recurrent } = state {
                    buffers.push(gpu.buffer(history.bytes)?);
                    buffers.push(gpu.buffer(recurrent.bytes)?);
                }
            }
            saved.push(Saved { buffers, position: 0 });
        }
        let frequencies: Vec<f32> = hp.rope_frequencies().into_iter().map(|f| f as f32).collect();
        let frequency_buffer = floats(&mut gpu, frequencies.len())?;
        gpu.write_buffer(&frequency_buffer, 0, &frequencies)?;
        // Each layer's routed experts as three roles of the file (a slot holds them back to back).
        let files = tensors.files()?;
        let mut roles = vec![None; hp.layers()];
        for layer in range.clone() {
            let mut layer_roles = Vec::new();
            for role in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                let tensor_name = name(layer, role);
                let tensor = tensors.tensor(&tensor_name)?;
                let (file, base) = tensors.location(&tensor_name).ok_or("the expert store needs the model's files")?;
                let bytes = (row_bytes(tensor.kind, tensor.dims[0])? * tensor.dims[1]) as u64;
                layer_roles.push(Role { file, base, stride: bytes, bytes });
            }
            roles[layer] = Some(layer_roles);
        }
        let layout = Arc::new(Layout::new(&files, roles, tensors.verifier())?);
        let experts = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(READERS)), settings.expert_slots,
                                         settings.host_expert_bytes)?;
        let mut engine = Engine {
            hp: Rc::new(hp),
            tensors,
            settings,
            gpu,
            dense: HashMap::new(),
            scratch,
            work,
            states,
            saved,
            experts,
            frequencies: frequency_buffer,
            position: 0,
            compare: std::env::var_os("Q35_COMPARE").map(PathBuf::from),
        };
        engine.upload_dense()?;
        Ok(engine)
    }

    /// Every tensor of this engine's layers but the routed experts, resident (with the head and the
    /// final norm on the last engine); the token embeddings stay on the host.
    fn upload_dense(&mut self) -> Result<()> {
        let range = self.settings.layers.clone();
        let mut names = Vec::new();
        if range.end == self.hp.layers() {
            names.extend(["output.weight".to_string(), "output_norm.weight".to_string()]);
        }
        for layer in range {
            let tensors: &[&str] = match self.hp.mixers[layer] {
                Mixer::DeltaNet => &["attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_dt.bias", "ssm_a",
                                     "ssm_conv1d.weight", "ssm_norm.weight", "ssm_out.weight"],
                Mixer::Attention => &["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                                      "attn_output.weight"],
            };
            for tensor in tensors.iter().chain(&["attn_norm.weight", "post_attention_norm.weight", "ffn_gate_inp.weight",
                                                 "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                                 "ffn_down_shexp.weight"]) {
                names.push(format!("blk.{layer}.{tensor}"));
            }
        }
        for name in names {
            let tensor = self.tensors.tensor(&name)?;
            let buffer = self.gpu.upload_bytes(tensor.data)?;
            self.dense.insert(name, Dense { buffer, kind: tensor.kind, dims: tensor.dims.clone() });
        }
        Ok(())
    }

    fn dense(&self, name: &str) -> Result<Dense> {
        self.dense.get(name).cloned().ok_or_else(|| format!("tensor {name} is not resident"))
    }

    /// out = W . input over `steps` rows (a whole resident matrix; a vector is one row).
    fn linear(&mut self, name: &str, input: P, steps: usize, out: M) -> Result<()> {
        let weight = self.dense(name)?;
        let (cols, rows) = (weight.dims[0] as u64, weight.dims.get(1).copied().unwrap_or(1) as u64);
        let scratch = self.scratch;
        self.gpu.linear(&weight.buffer, weight.kind, rows, cols, 0, input, 0, steps as u64, out, 0, &scratch)
    }

    fn norm(&mut self, name: &str, input: P, rows: usize, width: usize, out: M) -> Result<()> {
        let weight = self.dense(name)?;
        let eps = self.hp.norm_eps;
        self.gpu.rms_norm(input, &weight.buffer, rows as u64, width as u64, eps, out)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// The device this engine runs on.
    pub fn gpu(&self) -> &Backend {
        &self.gpu
    }

    /// How many experts the device holds in VRAM and can keep in the host tier.
    pub fn expert_capacity(&self) -> (usize, usize) {
        self.experts.capacity()
    }

    /// The experts held, least recently used first: in VRAM, and only in the host tier.
    pub fn expert_contents(&self) -> Contents {
        self.experts.contents()
    }

    /// Warm the expert store up with an earlier run's contents of this engine's layers.
    pub fn warm_experts(&mut self, vram: &[(usize, usize)], host: &[(usize, usize)]) -> Result<usize> {
        let layers = self.settings.layers.clone();
        let experts = self.hp.experts;
        let mine = |keys: &[(usize, usize)]| -> Vec<(usize, usize)> {
            keys.iter().copied().filter(|&(layer, expert)| layers.contains(&layer) && expert < experts).collect()
        };
        let (vram, host) = (mine(vram), mine(host));
        self.experts.warm(&mut self.gpu, &vram, &host)
    }

    /// Take the expert reads that have landed, without waiting; those still in flight.
    pub fn poll_experts(&mut self) -> Result<usize> {
        self.experts.poll(&mut self.gpu)
    }

    pub fn cache_counts(&self) -> Counts {
        self.experts.counts()
    }

    /// Wait for the expert store's reads and copies (before dropping the engine).
    pub fn settle(&mut self) -> Result<()> {
        self.experts.settle(&mut self.gpu)
    }

    /// Forget the sequence: the DeltaNets' states start from zero again (K/V rows past the
    /// position are never read).
    pub fn reset(&mut self) -> Result<()> {
        for state in self.states.values() {
            if let State::DeltaNet { history, recurrent } = state {
                for buffer in [history, recurrent] {
                    self.gpu.write_buffer(buffer, 0, &vec![0f32; (buffer.bytes / 4) as usize])?;
                }
            }
        }
        self.position = 0;
        Ok(())
    }

    /// Save the sequence state to checkpoint `slot` (each DeltaNet's convolution history and
    /// recurrent state, and the position), or restore it from there. K/V is not copied: a restore
    /// keeps the rows below the checkpoint's position, which the caller must know to be the
    /// checkpoint's sequence's (the live sequence matching it that far).
    pub fn checkpoint(&mut self, slot: usize, save: bool) -> Result<()> {
        let saved = self.saved.get_mut(slot).ok_or("no such qwen35moe checkpoint")?;
        let live = self.states.values().flat_map(|state| match state {
            State::DeltaNet { history, recurrent } => vec![*history, *recurrent],
            State::Attention { .. } => Vec::new(),
        });
        for (live, copy) in live.zip(&saved.buffers) {
            let (from, to) = if save { (live, *copy) } else { (*copy, live) };
            self.gpu.copy(from.base(), to.base(), from.bytes)?;
        }
        if save {
            saved.position = self.position;
        } else {
            self.position = saved.position;
        }
        Ok(())
    }

    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.pass(tokens)?.ok_or_else(|| "this qwen35moe engine runs only some of the layers".into())
    }

    /// Run this engine's layers over `tokens`: the first engine embeds them, a later one continues
    /// the stream handed over to it ([`Engine::hand_over`]); the last returns the logits after the
    /// last token.
    pub fn pass(&mut self, tokens: &[u32]) -> Result<Option<Vec<f32>>> {
        let n = tokens.len();
        if n == 0 || n > self.settings.batch || self.position + n > self.settings.context {
            return Err("qwen35moe forward exceeds the batch or the context".into());
        }
        let hp = self.hp.clone();
        let w = hp.width;
        let range = self.settings.layers.clone();
        let positions: Vec<i32> = (self.position..self.position + n).map(|p| p as i32).collect();
        let work = self.work;
        self.gpu.write_buffer(&work.positions, 0, &positions)?;
        if range.start == 0 {
            let embed = self.tensors.tensor("token_embd.weight")?;
            let bytes = row_bytes(embed.kind, embed.dims[0])?;
            let mut rows = Vec::with_capacity(n * w);
            for &token in tokens {
                if token as usize >= hp.vocabulary {
                    return Err("qwen35moe token is out of range".into());
                }
                formats::decode_into(embed.kind, &embed.data[token as usize * bytes..(token as usize + 1) * bytes], &mut rows)?;
            }
            self.gpu.write_buffer(&work.stream, 0, &rows)?;
        }
        for layer in range.clone() {
            self.layer(layer, n)?;
        }
        let logits = if range.end == hp.layers() {
            let last = (n - 1) * w;
            self.norm("output_norm.weight", work.stream.at(last), 1, w, work.x.base())?;
            self.linear("output.weight", work.x.base(), 1, work.head.base())?;
            let mut logits = vec![0f32; hp.vocabulary];
            self.gpu.read_buffer(&work.head, 0, &mut logits)?;
            Some(logits)
        } else {
            None
        };
        self.position += n;
        Ok(logits)
    }

    /// Hand the stream of the pass that just ran over `rows` tokens to the engine of the next layers.
    pub fn hand_over(&mut self, next: &mut Engine<'a, T>, rows: usize) -> Result<()> {
        if next.settings.layers.start != self.settings.layers.end {
            return Err("qwen35moe engines hand over between adjacent layers only".into());
        }
        let bytes = (rows * self.hp.width * 4) as u64;
        let (source, target) = (self.work.stream.base(), next.work.stream.base());
        self.gpu.copy_to(source, &mut next.gpu, target, bytes)
    }

    /// Compare a buffer's `rows` rows of `row` values (this pass's positions) with the
    /// reference's tensor over the same positions (Q35_COMPARE).
    fn check(&mut self, label: &str, layer: usize, buffer: Buffer, rows: usize, row: usize) -> Result<()> {
        let Some(dir) = self.compare.clone() else { return Ok(()) };
        let Ok(bytes) = std::fs::read(dir.join(format!("{label}-{layer}.f32"))) else { return Ok(()) };
        let reference: Vec<f32> = bytes[32..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut values = vec![0f32; rows * row];
        self.gpu.read_buffer(&buffer, 0, &mut values)?;
        let Some(reference) = reference.get(self.position * row..(self.position + rows) * row) else {
            eprintln!("  {label}-{layer}: positions {}..{} are past the reference's {}", self.position, self.position + rows,
                      reference.len() / row);
            return Ok(());
        };
        let (mut diff, mut norm) = (0f64, 0f64);
        for (a, b) in values.iter().zip(reference) {
            diff += (*a as f64 - *b as f64).powi(2);
            norm += (*b as f64).powi(2);
        }
        eprintln!("  {label}-{layer} @{}: rel {:.2e}", self.position, (diff / norm.max(1e-30)).sqrt());
        Ok(())
    }

    fn layer(&mut self, layer: usize, n: usize) -> Result<()> {
        let hp = self.hp.clone();
        let w = hp.width;
        let work = self.work;
        self.norm(&name(layer, "attn_norm"), work.stream.base(), n, w, work.x.base())?;
        self.check("attn_norm", layer, work.x, n, w)?;
        match hp.mixers[layer] {
            Mixer::DeltaNet => self.delta_net(layer, n)?,
            Mixer::Attention => self.attention(layer, n)?,
        }
        self.gpu.add(work.mixed.base(), (n * w) as u64, work.stream.base())?;
        self.check("attn_residual", layer, work.stream, n, w)?;
        self.norm(&name(layer, "post_attention_norm"), work.stream.base(), n, w, work.x.base())?;
        self.moe(layer, n)?;
        self.check("ffn_out", layer, work.out, n, w)?;
        self.gpu.add(work.out.base(), (n * w) as u64, work.stream.base())?;
        self.check("l_out", layer, work.stream, n, w)
    }

    /// The DeltaNet of `x` into `mixed`, its state carried.
    fn delta_net(&mut self, layer: usize, n: usize) -> Result<()> {
        let hp = self.hp.clone();
        let work = self.work;
        let (dim, v_heads) = (hp.state_dim, hp.v_heads);
        let channels = hp.conv_channels();
        self.linear(&name(layer, "attn_qkv"), work.x.base(), n, work.qkv.base())?;
        self.linear(&name(layer, "attn_gate"), work.x.base(), n, work.z.base())?;
        self.linear(&name(layer, "ssm_beta"), work.x.base(), n, work.beta.base())?;
        self.linear(&name(layer, "ssm_alpha"), work.x.base(), n, work.alpha.base())?;
        let (history, recurrent) = match &self.states[&layer] {
            State::DeltaNet { history, recurrent } => (*history, *recurrent),
            State::Attention { .. } => return Err("a DeltaNet layer without its state".into()),
        };
        let taps = self.dense(&name(layer, "ssm_conv1d"))?;
        self.gpu.conv_silu(work.qkv.base(), history.base(), taps.buffer.base(), channels as u64, hp.conv_kernel as u64, n as u64)?;
        self.check("conv_output_silu", layer, work.qkv, n, channels)?;
        let dt = self.dense(&format!("blk.{layer}.ssm_dt.bias"))?;
        let a = self.dense(&format!("blk.{layer}.ssm_a"))?;
        self.gpu.delta_rule(work.qkv.base(), work.beta.base(), work.alpha.base(), dt.buffer.base(), a.buffer.base(), recurrent.base(),
                            hp.k_heads as u64, v_heads as u64, dim as u64, n as u64, hp.norm_eps, work.core.base())?;
        self.check("attn_output", layer, work.core, n, v_heads * dim)?;
        let norm = self.dense(&name(layer, "ssm_norm"))?;
        self.gpu.gated_norm(work.core.base(), work.z.base(), &norm.buffer, (n * v_heads) as u64, dim as u64, hp.norm_eps,
                            work.normed.base())?;
        self.linear(&name(layer, "ssm_out"), work.normed.base(), n, work.mixed.base())?;
        self.check("linear_attn_out", layer, work.mixed, n, hp.width)
    }

    /// Gated full attention of `x` into `mixed`, the layer's K/V appended.
    fn attention(&mut self, layer: usize, n: usize) -> Result<()> {
        let hp = self.hp.clone();
        let work = self.work;
        let (heads, kv_heads, dim) = (hp.heads, hp.kv_heads, hp.head_dim);
        self.linear(&name(layer, "attn_q"), work.x.base(), n, work.q_full.base())?;
        self.linear(&name(layer, "attn_k"), work.x.base(), n, work.k.base())?;
        self.linear(&name(layer, "attn_v"), work.x.base(), n, work.v.base())?;
        // Each head's projection holds its query, then its gate.
        self.gpu.split(work.q_full.base(), (n * heads) as u64, dim as u64, work.q.base(), work.gate.base())?;
        self.norm(&name(layer, "attn_q_norm"), work.q.base(), n * heads, dim, work.q.base())?;
        self.norm(&name(layer, "attn_k_norm"), work.k.base(), n * kv_heads, dim, work.k.base())?;
        let frequencies = self.frequencies;
        self.gpu.rope_halves(work.q.base(), n as u64, heads as u64, dim as u64, hp.rope_dims as u64, &work.positions, &frequencies)?;
        self.gpu.rope_halves(work.k.base(), n as u64, kv_heads as u64, dim as u64, hp.rope_dims as u64, &work.positions, &frequencies)?;
        self.check("Qcur", layer, work.q, n, heads * dim)?;
        self.check("Kcur", layer, work.k, n, kv_heads * dim)?;
        let (keys, values) = match &self.states[&layer] {
            State::Attention { keys, values } => (*keys, *values),
            State::DeltaNet { .. } => return Err("an attention layer without its K/V".into()),
        };
        let (kv_width, context) = ((kv_heads * dim) as u64, self.settings.context as u64);
        self.gpu.scatter(work.k.base(), work.positions.base(), n as u64, kv_width, keys.base(), context)?;
        self.gpu.scatter(work.v.base(), work.positions.base(), n as u64, kv_width, values.base(), context)?;
        let scale = 1.0 / (dim as f32).sqrt();
        self.gpu.attention(work.q.base(), keys.base(), values.base(), context, work.positions.base(), n as u64, heads as u64,
                           kv_heads as u64, dim as u64, scale, Some(work.gate.base()), work.attended.base())?;
        self.linear(&name(layer, "attn_output"), work.attended.base(), n, work.mixed.base())?;
        self.check("attn_output", layer, work.mixed, n, hp.width)
    }

    /// The MoE of `x` into `out`: the sigmoid-gated shared expert, then the routed ones added.
    fn moe(&mut self, layer: usize, n: usize) -> Result<()> {
        let hp = self.hp.clone();
        let work = self.work;
        let (w, k, ff, shared_ff) = (hp.width, hp.experts_used, hp.expert_ff, hp.shared_ff);
        self.linear(&name(layer, "ffn_gate_inp"), work.x.base(), n, work.router.base())?;
        self.gpu.route(work.router.base(), None, n as u64, hp.experts as u64, k as u64, 0, true, 1.0, work.ids.base(),
                       work.weights.base())?;
        // The routes travel to the host while the shared expert runs.
        self.gpu.read_start(&work.ids, 0, (n * k * 4) as u64)?;
        self.linear(&name(layer, "ffn_gate_shexp"), work.x.base(), n, work.ffn_gate.base())?;
        self.linear(&name(layer, "ffn_up_shexp"), work.x.base(), n, work.ffn_up.base())?;
        self.gpu.swiglu(work.ffn_gate.base(), work.ffn_up.base(), n as u64, shared_ff as u64, 0.0, None, work.hidden.base())?;
        self.linear(&name(layer, "ffn_down_shexp"), work.hidden.base(), n, work.out.base())?;
        self.linear(&name(layer, "ffn_gate_inp_shexp"), work.x.base(), n, work.shared_gate.base())?;
        self.gpu.scale_rows(work.out.base(), work.shared_gate.base(), n as u64, w as u64)?;
        self.check("ffn_shexp_gated", layer, work.out, n, w)?;
        let mut routes = vec![0i32; n * k];
        self.gpu.read_finish(&mut routes)?;
        let gate = self.tensors.tensor(&name(layer, "ffn_gate_exps"))?;
        let up = self.tensors.tensor(&name(layer, "ffn_up_exps"))?;
        let down = self.tensors.tensor(&name(layer, "ffn_down_exps"))?;
        let kinds = [gate.kind, up.kind, down.kind];
        let (gate_bytes, up_bytes, down_bytes) =
            ((ff * row_bytes(gate.kind, w)?) as u64, (ff * row_bytes(up.kind, w)?) as u64, (w * row_bytes(down.kind, ff)?) as u64);
        let offsets = [0, gate_bytes, gate_bytes + up_bytes];
        let distinct: Vec<usize> = routes.iter().map(|&expert| expert as usize).collect::<BTreeSet<_>>().into_iter().collect();
        // A few rows run over their experts' slots in one launch, when the store's smallest size
        // class can hold them all at once (a store of few slots takes them in waves instead).
        if n <= SLOT_ROWS && distinct.len() <= self.experts.wave_limit() {
            let experts = distinct;
            let slots = self.experts.acquire(&mut self.gpu, layer, &experts)?;
            let bases: Vec<u64> = routes.iter()
                .map(|expert| slots[experts.binary_search(&(*expert as usize)).unwrap()].base() as u64).collect();
            let slot_bytes = slots.first().map_or(0, |slot| slot.bytes);
            self.gpu.experts(Some(&bases), slot_bytes, offsets, kinds, w as u64, ff as u64, n as u64, k as u64, work.x.base(), 0,
                             Some(work.weights.base()), 0.0, true, work.hidden.base(), work.out.base(), 0, None)?;
            return self.experts.release(&mut self.gpu);
        }
        // A batch: each expert's rows gathered and run through its matrices, in waves (the next
        // wave's experts load while this one's compute).
        let mut weights = vec![0f32; n * k];
        self.gpu.read_buffer(&work.weights, 0, &mut weights)?;
        type Users = (Vec<i32>, Vec<f32>);
        let mut by_expert: BTreeMap<usize, Users> = BTreeMap::new();
        for (slot, &expert) in routes.iter().enumerate() {
            let entry = by_expert.entry(expert as usize).or_default();
            entry.0.push((slot / k) as i32);
            entry.1.push(weights[slot]);
        }
        let by_expert: Vec<(usize, Users)> = by_expert.into_iter().collect();
        let waves: Vec<&[(usize, Users)]> = by_expert.chunks(WAVE.min(self.experts.wave_limit())).collect();
        let scratch = self.scratch;
        for (index, wave) in waves.iter().enumerate() {
            let experts: Vec<usize> = wave.iter().map(|(expert, _)| *expert).collect();
            let slots = self.experts.acquire(&mut self.gpu, layer, &experts)?;
            if let Some(next) = waves.get(index + 1) {
                let experts: Vec<usize> = next.iter().map(|(expert, _)| *expert).collect();
                self.experts.prefetch(&mut self.gpu, layer, &experts, true)?;
            }
            for ((_, (rows, user_weights)), slot) in wave.iter().zip(slots) {
                let count = rows.len() as u64;
                self.gpu.write_buffer(&work.expert_rows, 0, rows)?;
                self.gpu.write_buffer(&work.expert_weights, 0, user_weights)?;
                self.gpu.gather(work.x.base(), n as u64, work.expert_rows.base(), count, w as u64, work.xe.base())?;
                self.gpu.linear(&slot.slice(0, gate_bytes), kinds[0], ff as u64, w as u64, 0, work.xe.base(), 0, count, work.ffn_gate.base(),
                                0, &scratch)?;
                self.gpu.linear(&slot.slice(gate_bytes, up_bytes), kinds[1], ff as u64, w as u64, 0, work.xe.base(), 0, count,
                                work.ffn_up.base(), 0, &scratch)?;
                self.gpu.swiglu(work.ffn_gate.base(), work.ffn_up.base(), count, ff as u64, 0.0, Some(work.expert_weights.base()),
                                work.hidden.base())?;
                self.gpu.linear(&slot.slice(gate_bytes + up_bytes, down_bytes), kinds[2], w as u64, ff as u64, 0, work.hidden.base(), 0,
                                count, work.y.base(), 0, &scratch)?;
                self.gpu.scatter_add(work.y.base(), work.expert_rows.base(), count, w as u64, work.out.base(), n as u64)?;
            }
            self.experts.release(&mut self.gpu)?;
        }
        Ok(())
    }
}

/// One pass over engines that run consecutive layers (on their own devices): each hands the stream
/// to the next; the logits after the last token.
pub fn forward_split<T: Tensors>(engines: &mut [Engine<'_, T>], tokens: &[u32]) -> Result<Vec<f32>> {
    let count = engines.len();
    for index in 0..count {
        let (head, tail) = engines.split_at_mut(index + 1);
        let engine = &mut head[index];
        let logits = engine.pass(tokens)?;
        match tail.first_mut() {
            Some(next) => engine.hand_over(next, tokens.len())?,
            None => return logits.ok_or_else(|| "the last qwen35moe engine has no head".into()),
        }
    }
    Err("no qwen35moe engines".into())
}
