//! The Qwen3.8-Flash-Next decoder on resident GPU layers, with MTP speculative decoding and
//! conversation checkpoints.
//!
//! Every layer runs in native spans on the GPUs; the host gathers embedding rows, takes the
//! first maximum of the logits and samples. Speculative decoding drafts with the model's own
//! MTP layer and verifies up to four tokens per pass with the one-token arithmetic, so greedy
//! output is identical to token-by-token decode.
//!
//! A prompt resumes from the longest prefix it shares with what the model holds: the live
//! sequence when the prompt extends it, else the deepest conversation checkpoint it matches
//! (the end of the system prompt, and where the last user-initiated and the last tool-result
//! request began generating).

use std::time::Instant;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::gpu::{expert_bytes, CpuExperts, Operand, Result, HIDDEN, WIDTH};
use crate::group::{row_count, Group};
use crate::sampling::{self, Generated, Sampling};
use crate::weights::{ngram_rows, Weights, CONTEXT_LIMIT};

pub const MAX_DRAFTS: usize = 3;
const END_TOKENS: [u32; 2] = [248044, 248046];
const IM_START: u32 = 248045;
const IM_END: u32 = 248046;
/// Verifying (and drafting) one more row, relative to a one-token pass.
const ROW_COST: f64 = 0.45;
/// Conversation checkpoints, each a page-locked host copy of the recurrent state (~118 MB):
/// root (the end of the system prompt), turn (where the reply to the last user message
/// began), prior (the same for the message before) and step (the reply to the last tool result).
const CHECKPOINTS: [&str; 4] = ["root", "turn", "prior", "step"];
/// Only a system prompt this long is worth a copy.
const ROOT_MINIMUM: usize = 2048;
const VOCABULARY: usize = 248320;

/// Drafts that maximize expected tokens per pass cost for a per-draft acceptance rate.
pub fn draft_length(acceptance: f64, limit: usize) -> usize {
    let (mut best, mut chosen) = (1.0, 0);
    for drafts in 1..=limit {
        let expected: f64 = (0..=drafts).map(|index| acceptance.powi(index as i32)).sum();
        let value = expected / (1.0 + ROW_COST * drafts as f64);
        if value > best {
            (best, chosen) = (value, drafts);
        }
    }
    chosen
}

/// Host bytes of one sequence's state before the GPU takes its attention and recurrence:
/// QSA K/V and pooled indexer keys, DeltaNet state and convolution, PLE history and the
/// partial indexer pooling buffers.
pub fn context_state_bytes(context: u64) -> u64 {
    let state = 12 * (context * 2 * 2 * 256 + (context / 4 + 1) * 128) * 4;
    state + 36 * (48 * 128 * 128 + 4 * 10240) * 4 + 10 * 10240 * 4 + 12 * 4 * 128 * 4
}

/// Expert bank bytes to page-lock given the host memory (0 if unknown): all of them with
/// memory twice their size; with at least 1.25 times, half the memory less the page-locked
/// sequence state; else none (the unlocked layers and the PLE table would thrash).
pub fn lock_budget(memory: u64, banks: u64, pinned_state: u64) -> u64 {
    if memory == 0 || 4 * memory < 5 * banks {
        return 0;
    }
    if memory >= 2 * banks {
        return banks;
    }
    (memory / 2).saturating_sub(pinned_state)
}

struct Checkpoint {
    slot: usize,
    tokens: Vec<u32>,
    final_hidden: Option<Vec<f32>>,
}

pub struct Settings {
    pub context: usize,
    pub state_budget_bytes: u64,
    pub prefill_batch_size: usize,
    pub speculative: usize,
    /// CPU expert threads for partial residency (0: one per physical core).
    pub cpu_threads: u32,
}

/// A small deterministic generator for sampling (xoshiro256**, seeded by SplitMix64).
pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Rng([next(), next(), next(), next()])
    }

    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    pub fn random(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// An index drawn with the given (not necessarily normalized) weights.
    pub fn choice(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut target = self.random() * total;
        for (index, weight) in weights.iter().enumerate() {
            if target < *weight {
                return index;
            }
            target -= weight;
        }
        weights.iter().rposition(|weight| *weight > 0.0).unwrap_or(0)
    }
}

/// Softmax of `logits / temperature` in double precision.
pub fn softmax(logits: &[f32], temperature: f64) -> Vec<f64> {
    let peak = logits.iter().fold(f32::NEG_INFINITY, |peak, value| peak.max(*value)) as f64;
    let weights: Vec<f64> = logits.iter().map(|value| ((*value as f64 - peak) / temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|weight| weight / total).collect()
}

/// The first maximum.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as u32
}

/// Each embedding row repeated over the four residual streams.
fn tile(embeddings: &[f32]) -> Vec<f32> {
    let mut hidden = Vec::with_capacity(embeddings.len() * 4);
    for row in embeddings.chunks_exact(HIDDEN) {
        for _ in 0..4 {
            hidden.extend_from_slice(row);
        }
    }
    hidden
}

fn last_row(values: &[f32]) -> &[f32] {
    &values[values.len() - WIDTH..]
}

pub struct Model {
    // Field order is drop order: the devices go before the CPU engine they use, and both
    // before the mappings their native plans point into.
    group: Group,
    cpu: Option<CpuExperts>,
    pub weights: Weights,
    pub context: usize,
    pub prefill_batch_size: usize,
    pub state_bytes: u64,
    pub speculative: usize,
    pub resident: bool,
    pub partial: bool,
    pub host_kv: bool,
    pub history: Vec<u32>,
    final_hidden: Option<Vec<f32>>,
    checkpoints: IndexMap<&'static str, Checkpoint>,
    pub last_metrics: Map<String, Value>,
    /// When set, the SHA-256 of every logits vector the model returns (a determinism check).
    pub logits_digests: Option<Vec<String>>,
    /// When set, every logits vector the model returns (validation tooling).
    pub logits_capture: Option<Vec<Vec<f32>>>,
}

impl Model {
    pub fn new(weights: Weights, mut group: Group, settings: &Settings) -> Result<Model> {
        let context = settings.context;
        if !(1..=CONTEXT_LIMIT).contains(&context) {
            return Err("Straydog context must be within the native model limit".into());
        }
        if !(1..=4096).contains(&settings.prefill_batch_size) {
            return Err("Straydog prefill batch must be between 1 and 4096".into());
        }
        if settings.speculative > MAX_DRAFTS {
            return Err(format!("Straydog speculative drafts must be between 0 and {MAX_DRAFTS}"));
        }
        let (layer_bytes, tail) = weights.placement_bytes(group.expert_parallel()?);
        group.balance_layers(&layer_bytes, tail)?;
        let drafting = settings.speculative > 0 && weights.drafter;
        validate_geometry(&weights)?;
        if settings.prefill_batch_size > group.logical_capacity() {
            return Err("Straydog prefill exceeds this GPU workspace capacity".into());
        }
        let context64 = context as u64;
        let mut state_bytes = context_state_bytes(context64);
        let attention_capable = group.batch_capacity() == 256;
        let kv_bytes = 12 * context64 * 1024 * 4;
        let fits = attention_capable && kv_bytes <= group.memory()?.1 / 4;
        let host_capable = attention_capable && group.host_state_supported()?;
        let recurrent_bytes = 36 * 48 * 128 * 128 * 4;
        // K/V beyond a quarter of the VRAM budget stays in mapped host memory, which the
        // GPU kernels read over the bus.
        let (mut host, mut batch) = (attention_capable && !fits && host_capable, settings.prefill_batch_size);
        if host_capable && state_bytes - recurrent_bytes <= settings.state_budget_bytes {
            (host, batch) = resident_placement(&weights, &group, context64, settings.prefill_batch_size, drafting, fits,
                                               (host, batch))?;
        }
        let gpu_attention = fits || host;
        if !gpu_attention {
            return Err("the GPU budget cannot hold the attention state; raise --vram".into());
        }
        if !host {
            state_bytes -= kv_bytes;
        }
        state_bytes -= recurrent_bytes;
        if state_bytes > settings.state_budget_bytes {
            return Err(format!("Straydog context state needs {state_bytes} bytes, over its RAM budget"));
        }
        group.reserve_model_state(context64, batch as u64, drafting, host)?;
        let checkpoint_bytes = group.reserve_checkpoints(CHECKPOINTS.len())?;
        let mut model = Model {
            group,
            cpu: None,
            weights,
            context,
            prefill_batch_size: batch,
            state_bytes,
            speculative: 0,
            resident: false,
            partial: false,
            host_kv: host,
            history: Vec::new(),
            final_hidden: None,
            checkpoints: IndexMap::new(),
            last_metrics: Map::new(),
            logits_digests: None,
            logits_capture: None,
        };
        model.resident = model.prepare_resident_moe()?;
        if !model.resident {
            let pinned = checkpoint_bytes + if host { kv_bytes } else { 0 };
            model.partial = model.group.batch_capacity() == 256 && model.prepare_hybrid_moe(settings.cpu_threads, pinned)?;
        }
        if !model.resident && !model.partial {
            return Err("the GPU budget cannot hold the model's dense weights; raise --vram".into());
        }
        model.prepare_decoder(context64)?;
        if model.partial {
            model.group.create_expert_slots()?;
        }
        // MTP drafting needs the fully resident decoder (exact multi-token verification).
        model.speculative = if drafting && model.resident && model.weights.drafter { settings.speculative } else { 0 };
        Ok(model)
    }

    pub fn info(&self) -> Result<Value> {
        self.group.info()
    }

    // ---- operands ----

    fn projection(&self, name: &str) -> Result<Operand<'_>> {
        let tensor = self.weights.tensor(name)?;
        let [cols, rows] = tensor.dims[..] else { return Err(format!("Straydog matrix rank mismatch: {name}")) };
        if crate::formats::block(tensor.kind).is_none() {
            return Err(format!("unsupported Straydog weight format {}: {name}", tensor.kind));
        }
        let (data, kind) = self.weights.raw(name, 0, None)?;
        Ok(Operand::bytes(format!("{name}:None"), data, kind, rows, cols))
    }

    fn vector(&self, name: &str, key: Option<&str>) -> Result<Operand<'_>> {
        let key = key.map_or_else(|| format!("@vector:{name}"), str::to_string);
        Ok(Operand::floats(key, self.weights.vector(name)?))
    }

    /// A HyperConnection's norm (keyed by its prefix) and its down, up and inject projections.
    fn hc_operands(&self, prefix: &str) -> Result<Vec<Operand<'_>>> {
        let mut operands = vec![self.vector(&format!("{prefix}_norm.weight"), Some(&format!("@norm:{prefix}")))?];
        for suffix in ["_down.weight", "_up.weight", "_inject.weight"] {
            operands.push(self.projection(&format!("{prefix}{suffix}"))?);
        }
        Ok(operands)
    }

    fn ple_operands(&self) -> Result<Vec<Operand<'_>>> {
        let mut operands = vec![self.projection("blk.1.ple_key.weight")?, self.projection("blk.1.ple_value.weight")?];
        for name in ["ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight"] {
            operands.push(self.vector(&format!("blk.1.{name}"), None)?);
        }
        Ok(operands)
    }

    fn head_operands(&self) -> Result<Vec<Operand<'_>>> {
        Ok(vec![self.vector("output_hc_norm.weight", None)?, self.projection("output_hc_down.weight")?,
                self.projection("output_hc_up.weight")?, self.projection("output.weight")?])
    }

    fn resident_requirements(&self, drafter: bool) -> Vec<(String, u64)> {
        self.weights.resident_requirements(drafter && self.weights.drafter)
    }

    // ---- placement ----

    /// Full residency: pin every layer's banks, router and shared expert, once the whole
    /// weight footprint fits every GPU.
    fn prepare_resident_moe(&mut self) -> Result<bool> {
        if self.weights.drafter && !self.group.can_pin_model(&self.resident_requirements(true), None)? {
            self.weights.drop_drafter();
        }
        if !self.group.can_pin_model(&self.resident_requirements(self.weights.drafter), None)? {
            return Ok(false);
        }
        for layer in self.weights.layers() {
            let prefix = format!("blk.{layer}.");
            let mut operands = Vec::new();
            for role in ["gate", "up", "down"] {
                let name = format!("{prefix}ffn_{role}_exps.weight");
                let tensor = self.weights.tensor(&name)?;
                let (cols, rows, experts) = (tensor.dims[0], tensor.dims[1], tensor.dims[2]);
                let (data, kind) = self.weights.raw(&name, 0, None)?;
                operands.push(Operand::bytes(format!("{name}:bank"), data, kind, rows * experts, cols));
            }
            for role in ["gate", "up", "down"] {
                operands.push(self.projection(&format!("{prefix}ffn_{role}_shexp.weight"))?);
            }
            operands.push(self.projection(&format!("{prefix}ffn_gate_inp.weight"))?);
            let name = format!("{prefix}ffn_gate_inp_shexp.weight");
            operands.push(self.vector(&name, Some(&format!("{name}:vector")))?);
            // SAFETY of the borrow: operands point into the weights, which outlive the group.
            let operands: Vec<Operand<'static>> = unsafe { std::mem::transmute(operands) };
            self.group.register_resident_moe(&prefix, &operands, 512)?;
        }
        Ok(true)
    }

    /// Partial residency: pin every dense operand and bind the host expert banks for the CPU
    /// engine and prefill streaming; page-lock whole layers within the lock budget.
    fn prepare_hybrid_moe(&mut self, cpu_threads: u32, pinned_state: u64) -> Result<bool> {
        if !self.group.hybrid_supported()? {
            return Ok(false);
        }
        self.weights.drop_drafter();
        let kinds = (0..48).map(|layer| self.weights.expert_kinds(layer)).collect::<Result<Vec<_>>>()?;
        let staging = 64 * kinds.iter().map(|kinds| expert_bytes(*kinds)).max().unwrap() + (256 << 20);
        let mut requirements = self.weights.dense_requirements();
        requirements.push(("@hybrid:staging".into(), staging));
        if !self.group.can_pin_model(&requirements, None)? {
            return Ok(false);
        }
        let cpu = CpuExperts::new(cpu_threads)?;
        let mut banks_total = 0u64;
        let mut layer_banks = Vec::new();
        for (layer, kinds) in kinds.iter().enumerate() {
            let prefix = format!("blk.{layer}.");
            let mut banks: Vec<&[u8]> = Vec::new();
            for role in ["gate", "up", "down"] {
                banks.push(self.weights.raw(&format!("{prefix}ffn_{role}_exps.weight"), 0, None)?.0);
            }
            banks_total += banks.iter().map(|bank| bank.len() as u64).sum::<u64>();
            let mut operands = Vec::new();
            for role in ["gate", "up", "down"] {
                operands.push(self.projection(&format!("{prefix}ffn_{role}_shexp.weight"))?);
            }
            operands.push(self.projection(&format!("{prefix}ffn_gate_inp.weight"))?);
            let name = format!("{prefix}ffn_gate_inp_shexp.weight");
            operands.push(self.vector(&name, Some(&format!("{name}:vector")))?);
            // SAFETY of the borrows: banks and operands point into the weights, which outlive the group.
            let (operands, banks): (Vec<Operand<'static>>, [&'static [u8]; 3]) = unsafe {
                (std::mem::transmute::<Vec<Operand<'_>>, Vec<Operand<'static>>>(operands),
                 std::mem::transmute::<[&[u8]; 3], [&'static [u8]; 3]>([banks[0], banks[1], banks[2]]))
            };
            self.group.register_hybrid_moe(&prefix, &operands, banks, 512, *kinds)?;
            layer_banks.push(banks);
        }
        // The head and its mixing are not decoder-layer operands; keep them resident too.
        let head: Vec<Operand<'static>> = unsafe {
            std::mem::transmute(vec![self.projection("output.weight")?, self.projection("output_hc_down.weight")?,
                                     self.projection("output_hc_up.weight")?])
        };
        self.group.pin(&head)?;
        let budget = lock_budget(self.weights.host_memory_bytes, banks_total, pinned_state);
        // Whole layers from the last: authentication read those most recently, so they are the
        // likeliest to be cached; the rest of the locked range is read ahead meanwhile.
        let (mut layers, mut locked) = (Vec::new(), 0u64);
        for layer in (0..48).rev() {
            let size: u64 = layer_banks[layer].iter().map(|bank| bank.len() as u64).sum();
            if locked + size > budget {
                break;
            }
            layers.push(layer);
            locked += size;
        }
        let mut registered = 0;
        if !layers.is_empty() {
            let weights = &self.weights;
            let ranges: Vec<(String, (u64, u64))> = layers
                .iter()
                .flat_map(|layer| ["gate", "up", "down"].map(|role| format!("blk.{layer}.ffn_{role}_exps.weight")))
                .map(|name| {
                    let tensor = weights.tensor(&name).unwrap();
                    (name, (tensor.offset, tensor.bytes))
                })
                .collect();
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    for (name, range) in &ranges {
                        if let Ok(file) = weights.file_of(name) {
                            crate::digest::will_need(file, std::slice::from_ref(range));
                        }
                    }
                });
                for layer in &layers {
                    if !layer_banks[*layer].iter().all(|bank| self.group.register_host(bank)) {
                        break;
                    }
                    registered += 1;
                }
            });
        }
        if registered > 0 {
            // The GPU reads a few absent experts per layer over PCIe while the CPU computes the
            // rest; both read host DRAM, so one route per layer.
            self.group.enable_assist(1)?;
        }
        self.group.attach_cpu(&cpu)?;
        self.cpu = Some(cpu);
        Ok(true)
    }

    /// Pin each complete decoder layer after the residency preflight.
    fn prepare_decoder(&mut self, context: u64) -> Result<()> {
        for layer in self.weights.layers() {
            let prefix = format!("blk.{layer}.");
            let mut operands = self.hc_operands(&format!("{prefix}hc_attn"))?;
            let (names, norms): (&[&str], &[&str]) = if layer % 4 != 3 && layer < 48 {
                (&["attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_out.weight"],
                 &["ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight"])
            } else {
                (&["attn_q.weight", "attn_k.weight", "attn_v.weight", "indexer.q_proj.weight", "indexer.k_proj.weight",
                   "attn_output.weight"],
                 &["attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"])
            };
            for name in names {
                operands.push(self.projection(&format!("{prefix}{name}"))?);
            }
            for name in norms {
                operands.push(self.vector(&format!("{prefix}{name}"), None)?);
            }
            operands.extend(self.hc_operands(&format!("{prefix}hc_ffn"))?);
            // SAFETY of the borrow: operands point into the weights, which outlive the group.
            let operands: Vec<Operand<'static>> = unsafe { std::mem::transmute(operands) };
            self.group.register_decoder_layer(layer, &operands, context)?;
        }
        Ok(())
    }

    // ---- state ----

    pub fn reset(&mut self) -> Result<()> {
        self.history.clear();
        self.final_hidden = None;
        self.group.reset_state()
    }

    fn save_checkpoint(&mut self, kind: &'static str) -> Result<()> {
        if kind == "turn" {
            if let Some(turn) = self.checkpoints.shift_remove("turn") {
                self.checkpoints.shift_remove("prior");
                self.checkpoints.insert("prior", turn);
            }
        }
        let used: Vec<usize> =
            self.checkpoints.iter().filter(|(name, _)| **name != kind).map(|(_, checkpoint)| checkpoint.slot).collect();
        let slot = (0..CHECKPOINTS.len()).find(|slot| !used.contains(slot)).unwrap();
        self.group.checkpoint(slot, true)?;
        self.checkpoints.insert(kind, Checkpoint {
            slot,
            tokens: self.history.clone(),
            final_hidden: self.final_hidden.clone(),
        });
        Ok(())
    }

    /// Prompt tokens already held, after restoring the state they need; at least one token is
    /// left to run. K/V below the restored position comes from the live sequence, which
    /// matches the prompt there; the checkpoint restores the recurrent state.
    fn resume(&mut self, tokens: &[u32]) -> Result<(usize, &'static str)> {
        let limit = tokens.len() - 1;
        let held = &self.history;
        if held.len() <= limit && held[..] == tokens[..held.len()] {
            return Ok((held.len(), "live"));
        }
        let span = held.len().min(limit);
        let common = (0..span).find(|index| held[*index] != tokens[*index]).unwrap_or(span);
        let mut best: Option<&'static str> = None;
        for (kind, checkpoint) in &self.checkpoints {
            let position = checkpoint.tokens.len();
            if position <= common
                && best.is_none_or(|best| position > self.checkpoints[best].tokens.len())
                && checkpoint.tokens[..] == tokens[..position]
            {
                best = Some(kind);
            }
        }
        let Some(kind) = best else {
            self.reset()?;
            return Ok((0, "none"));
        };
        let checkpoint = &self.checkpoints[kind];
        let (slot, saved, hidden) = (checkpoint.slot, checkpoint.tokens.clone(), checkpoint.final_hidden.clone());
        self.group.checkpoint(slot, false)?;
        self.final_hidden = hidden;
        self.history = saved;
        Ok((self.history.len(), kind))
    }

    /// Positions (multiples of four) this prefill checkpoints at, with their kinds. A request
    /// no later one continues (`keep` false) saves only the system-prompt root.
    fn checkpoint_plan(&self, tokens: &[u32], reused: usize, turn: bool, keep: bool) -> Vec<(usize, Vec<&'static str>)> {
        let mut plan: Vec<(usize, Vec<&'static str>)> = Vec::new();
        let mut add = |position: usize, kind: &'static str| match plan.iter_mut().find(|(at, _)| *at == position) {
            Some((_, kinds)) => kinds.push(kind),
            None => plan.push((position, vec![kind])),
        };
        let root_held = self.checkpoints.get("root").is_some_and(|root| {
            root.tokens.len() <= tokens.len() && root.tokens[..] == tokens[..root.tokens.len()]
        });
        if !root_held {
            if let Some(end) = tokens.iter().position(|token| *token == IM_END) {
                let root = (end + 1) / 4 * 4;
                if root >= ROOT_MINIMUM && reused < root && root < tokens.len() {
                    add(root, "root");
                }
            }
        }
        if keep {
            // The reply starts at the prompt's last <|im_start|> (the generation prompt).
            let lowest = tokens.len().saturating_sub(63);
            if let Some(index) = (lowest..tokens.len()).rev().find(|index| tokens[*index] == IM_START) {
                let start = index / 4 * 4;
                if reused < start {
                    add(start, if turn { "turn" } else { "step" });
                }
            }
        }
        plan.sort_by_key(|(position, _)| *position);
        plan
    }

    // ---- forward ----

    fn ple_embeddings(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let history: Vec<i64> = self.history[self.history.len().saturating_sub(2)..].iter().map(|t| *t as i64).collect();
        let tokens: Vec<i64> = tokens.iter().map(|token| *token as i64).collect();
        let indices = ngram_rows(&history, &tokens, &self.weights.integer_arrays)?;
        self.weights.rows("per_layer_token_embd.weight", &indices)
    }

    fn token_embeddings(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let ids: Vec<i64> = tokens.iter().map(|token| *token as i64).collect();
        self.weights.rows("token_embd.weight", &ids)
    }

    fn with_operands<T>(&mut self, build: impl FnOnce(&Model) -> Result<Vec<Operand<'_>>>,
                        run: impl FnOnce(&mut Group, &[Operand]) -> Result<T>) -> Result<T> {
        let operands = build(self)?;
        // SAFETY of the borrow: operands point into the weights, which the group never frees.
        let operands: Vec<Operand<'static>> = unsafe { std::mem::transmute(operands) };
        run(&mut self.group, &operands)
    }

    /// Advance the model over `tokens`; the logits after the last one when asked (None on cancellation).
    fn forward(&mut self, tokens: &[u32], logits: bool, cancelled: &dyn Fn() -> bool) -> Result<Option<Option<Vec<f32>>>> {
        if self.history.len() + tokens.len() > self.context || tokens.iter().any(|token| *token as usize >= VOCABULARY) {
            return Err("Straydog token or context is out of range".into());
        }
        let embeddings = self.token_embeddings(tokens)?;
        let previous = self.final_hidden.clone();
        let position = self.history.len() as u64;
        let hidden = if tokens.len() > 1 {
            // With another device to keep them on, the prompt's rows never leave the GPUs.
            let stash = self.speculative > 0 && self.group.mtp_stash();
            let Some(rows) = self.resident_prefill(&embeddings, tokens, cancelled, stash)? else { return Ok(None) };
            if self.speculative > 0 {
                if stash {
                    self.mtp_prompt_resident(tokens, previous)?;
                } else {
                    self.mtp_prompt(tokens, &rows, previous)?;
                }
            }
            rows
        } else {
            let mut hidden = self.group.decode_span(0, 1, &tile(&embeddings), position)?;
            if cancelled() {
                self.reset()?;
                return Ok(None);
            }
            let embeddings = self.ple_embeddings(tokens)?;
            hidden = self.with_operands(Model::ple_operands, |group, operands| {
                group.ple("blk.1.", &hidden, &embeddings, operands)
            })?;
            for (begin, end) in self.group.decoder_spans(1) {
                if cancelled() {
                    self.reset()?;
                    return Ok(None);
                }
                hidden = self.group.decode_span(begin, end, &hidden, position)?;
            }
            hidden
        };
        self.history.extend_from_slice(tokens);
        self.final_hidden = Some(last_row(&hidden).to_vec());
        if !logits {
            return Ok(Some(None));
        }
        let row = last_row(&hidden).to_vec();
        let mixed = self.with_operands(
            |model| Ok(vec![model.vector("output_hc_norm.weight", None)?, model.projection("output_hc_down.weight")?,
                            model.projection("output_hc_up.weight")?]),
            |group, operands| group.hyperconnection("output_hc", &row, operands[0].clone(), &operands[1..]),
        )?;
        let output = self.with_operands(|model| Ok(vec![model.projection("output.weight")?]), |group, operands| {
            group.project_many(&[(operands[0].clone(), &mixed[..])])
        })?;
        let logits = output.into_iter().next().unwrap();
        if let Some(digests) = self.logits_digests.as_mut() {
            let bytes: Vec<u8> = logits.iter().flat_map(|value| value.to_le_bytes()).collect();
            digests.push(format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&bytes)));
        }
        if let Some(capture) = self.logits_capture.as_mut() {
            capture.push(logits.clone());
        }
        Ok(Some(Some(logits)))
    }

    /// The last processed token's four-stream hidden state.
    pub fn final_hidden(&self) -> Option<&[f32]> {
        self.final_hidden.as_deref()
    }

    /// Keep prompt rows on each GPU between spans; control returns after at most four layers
    /// for cancellation, and rows cross the host only at device boundaries.
    /// `stash`: the MTP layer reads every final row from the first layer's workspace, so only
    /// the last comes back; otherwise speculation takes them all.
    fn resident_prefill(&mut self, embeddings: &[f32], tokens: &[u32], cancelled: &dyn Fn() -> bool, stash: bool)
                        -> Result<Option<Vec<f32>>> {
        let (steps, position) = (tokens.len() as u64, self.history.len() as u64);
        let boundaries = self.group.decoder_spans(1);
        let mut spans = vec![(0, 1)];
        for (begin, end) in &boundaries {
            spans.extend((*begin..*end).step_by(4).map(|first| (first, (first + 4).min(*end))));
        }
        // The prompt's embeddings start the first span; later spans continue on the device, and
        // rows move GPU to GPU where the next span is on the other device.
        self.group.prefill_embed(0, embeddings)?;
        let mut rows_out = Vec::new();
        let mut previous: Option<usize> = None;
        for (index, (first, last)) in spans.into_iter().enumerate() {
            if let Some(previous) = previous {
                self.group.prefill_handoff(previous, first, steps)?;
            }
            previous = Some(last - 1);
            if cancelled() {
                self.reset()?;
                return Ok(None);
            }
            if index == 1 {
                let embeddings = self.ple_embeddings(tokens)?;
                self.with_operands(Model::ple_operands, |group, operands| {
                    group.prefill_ple("blk.1.", &embeddings, operands)
                })?;
            }
            // Speculation keeps every final row: the MTP layer reads the prompt's hidden states.
            let rows = if last == 48 { if self.speculative > 0 && !stash { steps } else { 1 } } else { 0 };
            let output = self.group.prefill_layers(first, last, None, position, steps, rows)?;
            if rows > 0 {
                rows_out = output;
            }
        }
        if stash && !self.group.prefill_handoff(47, 0, steps)? {
            return Err("Straydog could not keep the prompt rows for the MTP layer".into());
        }
        Ok(Some(rows_out))
    }

    /// Advance the MTP layer's K/V over prologue rows starting at `position`.
    fn mtp_layer(&mut self, rows: &[f32], position: u64) -> Result<()> {
        let steps = row_count(rows) as u64;
        if steps == 1 {
            self.group.verify_span(48, 49, rows, position)?;
        } else {
            self.group.prefill_layers(48, 49, Some(rows), position, steps, 0)?;
        }
        Ok(())
    }

    fn mtp_operands(&self) -> Result<Vec<Operand<'_>>> {
        Ok(vec![self.vector("mtp.embed_norm.weight", None)?, self.projection("mtp.fc_embedding.weight")?,
                self.vector("mtp.hidden_norm.weight", None)?, self.projection("mtp.fc_hidden.weight")?])
    }

    fn mtp_prologue(&mut self, embeddings: &[f32], hidden: &[f32]) -> Result<Vec<f32>> {
        self.with_operands(Model::mtp_operands, |group, operands| group.mtp_prologue(operands, embeddings, hidden))
    }

    /// Draft tokens from MTP rows through its own mixer and the shared vocabulary head.
    fn mtp_head(&mut self, values: &[f32]) -> Result<Vec<i32>> {
        let (tokens, _) = self.with_operands(
            |model| Ok(vec![model.vector("mtp.hc_out_norm.weight", None)?, model.projection("mtp.hc_out_down.weight")?,
                            model.projection("mtp.hc_out_up.weight")?, model.projection("output.weight")?]),
            |group, operands| group.output_head(operands, values, false),
        )?;
        Ok(tokens)
    }

    /// [`Model::mtp_prompt`] with the prompt's rows kept on the GPUs (see [`Group::mtp_stash`]):
    /// the same pairs, in the same batches of 512.
    fn mtp_prompt_resident(&mut self, tokens: &[u32], previous: Option<Vec<f32>>) -> Result<()> {
        let start = self.history.len();
        let previous = previous.filter(|_| start > 0);
        let (following, first) = if previous.is_some() { (tokens, start - 1) } else { (&tokens[1..], start) };
        for offset in (0..following.len()).step_by(512) {
            let end = (offset + 512).min(following.len());
            let embedded = self.token_embeddings(&following[offset..end])?;
            // Pair j's hidden row is the prompt's row j, or row j - 1 after the previous row.
            let (row, earlier) = match previous.as_deref() {
                Some(row) if offset == 0 => (0, Some(row)),
                Some(_) => (offset - 1, None),
                None => (offset, None),
            };
            self.with_operands(Model::mtp_operands, |group, operands| {
                group.mtp_prompt_rows(operands, row, end - offset, earlier, &embedded, (first + offset) as u64)
            })?;
        }
        Ok(())
    }

    /// Fill the MTP layer's K/V for a prompt batch: pair (hidden_t, token_t+1) sits at position t.
    fn mtp_prompt(&mut self, tokens: &[u32], rows: &[f32], previous: Option<Vec<f32>>) -> Result<()> {
        let start = self.history.len();
        let mut hidden: Vec<&[f32]> = rows.chunks_exact(WIDTH).take(row_count(rows) - 1).collect();
        let mut following: Vec<u32> = tokens[1..].to_vec();
        let mut first = start;
        if start > 0 {
            if let Some(previous) = previous.as_deref() {
                hidden.insert(0, previous);
                following.insert(0, tokens[0]);
                first = start - 1;
            }
        }
        for offset in (0..hidden.len()).step_by(512) {
            let end = (offset + 512).min(hidden.len());
            let chunk: Vec<f32> = hidden[offset..end].concat();
            let embedded = self.token_embeddings(&following[offset..end])?;
            let prologue = self.mtp_prologue(&embedded, &chunk)?;
            self.mtp_layer(&prologue, (first + offset) as u64)?;
        }
        Ok(())
    }

    /// Catch the MTP layer up on the accepted pairs, then chain `limit` greedy drafts.
    fn draft(&mut self, hidden: &[f32], tokens: &[u32], position: u64, limit: usize) -> Result<Vec<u32>> {
        let embedded = self.token_embeddings(tokens)?;
        let rows = self.mtp_prologue(&embedded, hidden)?;
        let output = self.group.verify_span(48, 49, &rows, position)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut drafts = vec![self.mtp_head(last_row(&output))?[0] as u32];
        let (mut multi, mut position) = (last_row(&output).to_vec(), position + tokens.len() as u64);
        while drafts.len() < limit {
            let embedded = self.token_embeddings(&drafts[drafts.len() - 1..])?;
            let rows = self.mtp_prologue(&embedded, &multi)?;
            let output = self.group.verify_span(48, 49, &rows, position)?;
            drafts.push(self.mtp_head(&output)?[0] as u32);
            (multi, position) = (output, position + 1);
        }
        Ok(drafts)
    }

    /// The next token and its drafts through every layer with the one-token arithmetic.
    #[allow(clippy::type_complexity)]
    fn verify(&mut self, tokens: &[u32], sampling: bool)
              -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<i32>, Option<Vec<f32>>)> {
        let position = self.history.len() as u64;
        let hidden = tile(&self.token_embeddings(tokens)?);
        let first = self.group.verify_span(0, 1, &hidden, position)?;
        let embeddings = self.ple_embeddings(tokens)?;
        self.group.copy_state("@ple-snapshot:blk.1.", "@ple:blk.1.", "blk.1.")?;
        let mut hidden = self.with_operands(Model::ple_operands, |group, operands| {
            group.ple_rows("blk.1.", &first, &embeddings, operands)
        })?;
        for (begin, end) in self.group.decoder_spans(1) {
            hidden = self.group.verify_span(begin, end, &hidden, position)?;
        }
        let (targets, scores) =
            self.with_operands(Model::head_operands, |group, operands| group.output_head(operands, &hidden, sampling))?;
        Ok((first, embeddings, hidden, targets, scores))
    }

    /// Keep the first `kept` verified tokens; replay DeltaNet and PLE state for the rest.
    fn commit(&mut self, tokens: &[u32], kept: usize, first: &[f32], embeddings: &[f32]) -> Result<()> {
        for (begin, end) in std::iter::once((0, 1)).chain(self.group.decoder_spans(1)) {
            self.group.verify_commit(begin, end, kept as u64)?;
        }
        if kept < tokens.len() {
            self.group.copy_state("@ple:blk.1.", "@ple-snapshot:blk.1.", "blk.1.")?;
            let (first, embeddings) = (&first[..kept * WIDTH], &embeddings[..kept * HIDDEN]);
            self.with_operands(Model::ple_operands, |group, operands| {
                group.ple_rows("blk.1.", first, embeddings, operands)
            })?;
        }
        self.history.extend_from_slice(&tokens[..kept]);
        Ok(())
    }

    /// Accepted draft count and the next token: a match with the GPU's argmax when greedy,
    /// else decided on the rows' logits (see [`sampling::accept`]).
    fn accept(drafts: &[u32], targets: &[i32], scores: Option<&[f32]>, sampling: &Sampling, generated: &mut Generated,
              rng: &mut Rng) -> (usize, u32) {
        let Some(scores) = scores.filter(|_| sampling.needs_logits()) else {
            let accepted = drafts.iter().zip(targets).take_while(|(draft, target)| **draft == **target as u32).count();
            for draft in &drafts[..accepted] {
                generated.insert(*draft);
            }
            return (accepted, targets[accepted] as u32);
        };
        sampling::accept(drafts, scores, VOCABULARY, sampling, generated, rng)
    }

    fn metric_add(&mut self, key: &str, amount: f64) {
        let value = self.last_metrics.get(key).and_then(Value::as_f64).unwrap_or(0.0) + amount;
        self.last_metrics.insert(key.into(), json!(value));
    }

    fn metric_count(&mut self, key: &str, amount: u64) {
        let value = self.last_metrics.get(key).and_then(Value::as_u64).unwrap_or(0) + amount;
        self.last_metrics.insert(key.into(), json!(value));
    }

    fn update_rate(&mut self) {
        let forwards = self.last_metrics["decode_forward_tokens"].as_u64().unwrap_or(0) as f64;
        let seconds = self.last_metrics["decode_seconds"].as_f64().unwrap_or(0.0);
        self.last_metrics.insert("decode_tokens_per_second".into(), json!(forwards / seconds));
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_speculative(&mut self, logits: &[f32], max_new: usize, sampling: &Sampling, rng: &mut Rng,
                            cancelled: &dyn Fn() -> bool, started: Instant, emit: &mut dyn FnMut(u32))
                            -> Result<()> {
        let mut generated = Generated::new(VOCABULARY);
        let mut token = sampling::choose(logits, sampling, &generated, rng);
        generated.insert(token);
        if END_TOKENS.contains(&token) {
            return Ok(());
        }
        self.last_metrics.insert("ttft_seconds".into(), json!(started.elapsed().as_secs_f64()));
        for (key, value) in [("generated_tokens", 1), ("speculative_passes", 0), ("drafted_tokens", 0), ("accepted_tokens", 0)] {
            self.last_metrics.insert(key.into(), json!(value));
        }
        emit(token);
        let (mut produced, mut acceptance, mut passes) = (1usize, 0.7f64, 0usize);
        // The MTP layer pairs each hidden state with the token after it, at the hidden's position.
        let mut mtp_hidden = self.final_hidden.clone().ok_or("Straydog has no hidden state to draft from")?;
        let mut mtp_tokens = vec![token];
        let mut mtp_position = self.history.len() as u64 - 1;
        while produced < max_new {
            if cancelled() {
                return Ok(());
            }
            let pass_started = Instant::now();
            let limit = self.speculative.min(max_new - produced - 1).min(self.context.saturating_sub(self.history.len() + 1));
            let mut drafting = draft_length(acceptance, limit);
            if drafting == 0 && limit > 0 && passes % 8 == 7 {
                drafting = 1;   // probe: the acceptance estimate must keep tracking the text
            }
            passes += 1;
            let drafts = self.draft(&mtp_hidden, &mtp_tokens, mtp_position, drafting)?;
            let rows: Vec<u32> = std::iter::once(token).chain(drafts.iter().copied()).collect();
            if self.history.len() + rows.len() > self.context {
                return Err("Straydog prompt plus output exceeds the allocated context".into());
            }
            let position = self.history.len() as u64;
            let (first, embeddings, hidden, targets, scores) = self.verify(&rows, sampling.needs_logits())?;
            let (accepted, following) = Self::accept(&drafts, &targets, scores.as_deref(), sampling, &mut generated, rng);
            generated.insert(following);
            if !drafts.is_empty() {
                let attempted = accepted + usize::from(accepted < drafts.len());
                acceptance = 0.9 * acceptance + 0.1 * accepted as f64 / attempted as f64;
            }
            let mut emitted: Vec<u32> = drafts[..accepted].iter().copied().chain([following]).collect();
            let mut kept = accepted + 1;
            if let Some(index) = emitted.iter().position(|value| END_TOKENS.contains(value)) {
                kept = kept.min(index + 1);
                emitted.truncate(index);
            }
            self.commit(&rows, kept, &first, &embeddings)?;
            self.final_hidden = Some(hidden[(kept - 1) * WIDTH..kept * WIDTH].to_vec());
            self.metric_count("speculative_passes", 1);
            self.metric_count("drafted_tokens", drafts.len() as u64);
            self.metric_count("accepted_tokens", accepted as u64);
            self.metric_add("decode_seconds", pass_started.elapsed().as_secs_f64());
            for value in &emitted {
                produced += 1;
                self.metric_count("generated_tokens", 1);
                self.metric_count("decode_forward_tokens", 1);
                self.update_rate();
                emit(*value);
            }
            if emitted.len() < accepted + 1 {
                return Ok(());
            }
            token = following;
            mtp_hidden = hidden[..(accepted + 1) * WIDTH].to_vec();
            mtp_tokens = emitted;
            mtp_position = position;
        }
        Ok(())
    }

    /// Generate without truncation or hidden retry, resuming from the longest prefix already
    /// held. `turn`: the prompt ends with a user message rather than a tool result;
    /// `checkpoint` false marks a one-off request whose reply no later request continues.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(&mut self, tokens: &[u32], max_new: usize, sampling: &Sampling, seed: u64,
                    cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize), turn: bool, checkpoint: bool,
                    emit: &mut dyn FnMut(u32)) -> Result<()> {
        let started = Instant::now();
        self.last_metrics = Map::new();
        if tokens.is_empty() || max_new == 0 || max_new > self.context || tokens.len() + max_new > self.context {
            return Err("Straydog prompt plus output exceeds the allocated context".into());
        }
        sampling.check()?;
        self.weights.check_unchanged()?;
        let (reused, resumed) = self.resume(tokens)?;
        self.last_metrics = json!({
            "cached_tokens": reused, "resumed_from": resumed, "prefill_tokens": tokens.len() - reused,
            "decode_forward_tokens": 0, "decode_seconds": 0.0, "decode_tokens_per_second": null,
            "generated_tokens": 0, "ttft_seconds": null,
        })
        .as_object()
        .unwrap()
        .clone();
        let mut output = None;
        let prefill_started = Instant::now();
        let plan = self.checkpoint_plan(tokens, reused, turn, checkpoint);
        let mut position = reused;
        let stops: Vec<usize> = plan.iter().map(|(at, _)| *at).chain([tokens.len()]).collect();
        for stop in stops {
            while position < stop {
                if cancelled() {
                    return Ok(());
                }
                let end = (position + self.prefill_batch_size).min(stop);
                // A one-token batch runs whole, as decode does; a longer one checks between spans.
                let check: &dyn Fn() -> bool = if self.prefill_batch_size == 1 { &|| false } else { cancelled };
                let Some(logits) = self.forward(&tokens[position..end], end == tokens.len(), check)? else {
                    return Ok(());
                };
                output = logits;
                if cancelled() {
                    return Ok(());
                }
                position = end;
                progress(end, tokens.len());
            }
            if let Some((_, kinds)) = plan.iter().find(|(at, _)| *at == stop) {
                for kind in kinds.clone() {
                    self.save_checkpoint(kind)?;
                }
            }
        }
        let elapsed = prefill_started.elapsed().as_secs_f64();
        self.last_metrics.insert("prefill_seconds".into(), json!(elapsed));
        self.last_metrics.insert("prefill_tokens_per_second".into(), json!((tokens.len() - reused) as f64 / elapsed));
        if self.partial {
            // Re-pick cached experts from this prompt's routing before decoding.
            let rebalance = Instant::now();
            self.group.rebalance(0)?;
            self.last_metrics.insert("rebalance_seconds".into(), json!(rebalance.elapsed().as_secs_f64()));
        }
        let mut rng = Rng::new(seed);
        let Some(mut logits) = output else { return Ok(()) };
        if self.speculative > 0 {
            return self.generate_speculative(&logits, max_new, sampling, &mut rng, cancelled, started, emit);
        }
        let mut generated = Generated::new(VOCABULARY);
        for index in 0..max_new {
            if cancelled() {
                return Ok(());
            }
            let token = sampling::choose(&logits, sampling, &generated, &mut rng);
            generated.insert(token);
            if END_TOKENS.contains(&token) {
                return Ok(());
            }
            if self.last_metrics["ttft_seconds"].is_null() {
                self.last_metrics.insert("ttft_seconds".into(), json!(started.elapsed().as_secs_f64()));
            }
            self.metric_count("generated_tokens", 1);
            emit(token);
            if index + 1 < max_new {
                let decode_started = Instant::now();
                let Some(next) = self.forward(&[token], true, &|| false)? else { return Ok(()) };
                logits = next.unwrap();
                self.metric_add("decode_seconds", decode_started.elapsed().as_secs_f64());
                self.metric_count("decode_forward_tokens", 1);
                self.update_rate();
            }
        }
        Ok(())
    }
}

/// (K/V in host memory, prefill batch) that lets every weight live on the GPUs, else
/// `default`. Full residency (and with it MTP) is worth more than K/V in VRAM; a placement
/// keeping the drafter wins over one that drops it; the configured batch and VRAM K/V first.
fn resident_placement(weights: &Weights, group: &Group, context: u64, prefill: usize, drafting: bool, vram_kv: bool,
                      default: (bool, usize)) -> Result<(bool, usize)> {
    let mut batches = vec![prefill];
    while *batches.last().unwrap() > 1024 {
        batches.push((batches.last().unwrap() / 2).max(1024));
    }
    let mut candidates: Vec<(bool, usize)> = if vram_kv { vec![(false, prefill)] } else { Vec::new() };
    candidates.extend(batches.iter().map(|batch| (true, *batch)));
    let keeps: &[bool] = if drafting { &[true, false] } else { &[false] };
    for keep in keeps {
        for (host, batch) in &candidates {
            let planned = group.planned_state_bytes(context, *batch as u64, drafting, *host)?;
            let requirements = weights.resident_requirements(*keep && weights.drafter);
            if group.can_pin_model(&requirements, Some(&planned))? {
                return Ok((*host, *batch));
            }
        }
    }
    Ok(default)
}

fn validate_geometry(weights: &Weights) -> Result<()> {
    let mut expected: Vec<(String, Vec<u64>)> = vec![
        ("token_embd.weight".into(), vec![2560, 248320]),
        ("output.weight".into(), vec![2560, 248320]),
        ("per_layer_token_embd.weight".into(), vec![160, 320001536]),
        ("output_hc_norm.weight".into(), vec![10240]),
        ("output_hc_down.weight".into(), vec![10240, 320]),
        ("output_hc_up.weight".into(), vec![320, 10240]),
    ];
    for layer in 0..48 {
        let prefix = format!("blk.{layer}.");
        for operation in ["hc_attn", "hc_ffn"] {
            expected.push((format!("{prefix}{operation}_norm.weight"), vec![10240]));
            expected.push((format!("{prefix}{operation}_down.weight"), vec![10240, 320]));
            expected.push((format!("{prefix}{operation}_up.weight"), vec![320, 10240]));
            expected.push((format!("{prefix}{operation}_inject.weight"), vec![10240, 4]));
        }
        expected.push((format!("{prefix}ffn_gate_inp.weight"), vec![2560, 512]));
        expected.push((format!("{prefix}ffn_gate_inp_shexp.weight"), vec![2560]));
        expected.push((format!("{prefix}ffn_gate_exps.weight"), vec![2560, 640, 512]));
        expected.push((format!("{prefix}ffn_up_exps.weight"), vec![2560, 640, 512]));
        expected.push((format!("{prefix}ffn_down_exps.weight"), vec![640, 2560, 512]));
        let shapes: &[(&str, [u64; 2])] = if layer % 4 != 3 {
            &[("attn_qkv.weight", [2560, 10240]), ("attn_gate.weight", [2560, 6144]), ("ssm_conv1d.weight", [4, 10240]),
              ("ssm_out.weight", [6144, 2560])]
        } else {
            &[("attn_q.weight", [2560, 12288]), ("attn_k.weight", [2560, 512]), ("attn_v.weight", [2560, 512]),
              ("attn_output.weight", [6144, 2560])]
        };
        for (name, shape) in shapes {
            expected.push((format!("{prefix}{name}"), shape.to_vec()));
        }
    }
    for (name, shape) in expected {
        if weights.shape(&name)? != shape.as_slice() {
            return Err(format!("Straydog model geometry mismatch: {name}"));
        }
    }
    if weights.integer_arrays.get("qwen4exp.ple.layers") != Some(&vec![1]) {
        return Err("Straydog model PLE placement mismatch".into());
    }
    Ok(())
}
