//! Qwen3.5 MoE (`qwen35moe`, Ornith-1.5): a single residual stream through blocks that are
//! either a Gated DeltaNet (linear attention with a recurrent state) or gated full attention
//! (Q/K normalized, partial rotary positions, causal GQA, a sigmoid output gate), each followed
//! by a softmax-routed MoE with a sigmoid-gated shared expert. The definitions follow the
//! reference implementation (llama.cpp `qwen35moe`, the same GGUF); every size comes from the
//! file's metadata.

pub mod engine;

use super::Keys;
use straydog_gguf::{Metadata, ModelHeader};

/// What a block's token mixer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// Gated DeltaNet: a convolved QKV projection, an L2-normalized delta rule over a per-head
    /// state, and an RMS norm gated by SiLU of a second projection.
    DeltaNet,
    /// Gated full attention.
    Attention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hparams {
    pub vocabulary: usize,
    pub width: usize,
    pub norm_eps: f32,
    /// The main blocks' mixers (the MTP blocks after them are not part of the forward pass).
    pub mixers: Vec<Mixer>,
    /// Blocks after the main ones that only a drafter uses.
    pub nextn: usize,
    pub context: usize,
    // Full attention.
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Leading dimensions of each head that rotate (pairs `i` and `i + rope_dims / 2`).
    pub rope_dims: usize,
    pub rope_base: f64,
    // Gated DeltaNet.
    pub conv_kernel: usize,
    /// Key and value dimension per head (the state is `state_dim` x `state_dim` per value head).
    pub state_dim: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    // Mixture of experts.
    pub experts: usize,
    pub experts_used: usize,
    pub expert_ff: usize,
    pub shared_ff: usize,
}

impl Hparams {
    pub fn from_header(header: &ModelHeader) -> Result<Hparams, String> {
        if header.architecture != "qwen35moe" {
            return Err(format!("expected a qwen35moe GGUF, found {}", header.architecture));
        }
        let keys = Keys::new(header);
        let blocks = keys.count("block_count")?;
        let nextn = if keys.has("nextn_predict_layers") { keys.count("nextn_predict_layers")? } else { 0 };
        let layers = blocks.checked_sub(nextn).filter(|&layers| layers > 0).ok_or("qwen35moe has no main blocks")?;
        let vocabulary = keys
            .raw("tokenizer.ggml.tokens")
            .and_then(|tokens| tokens.array())
            .map(|tokens| tokens.len())
            .ok_or("GGUF tokenizer tokens are missing")?;
        // Recurrent blocks as the file lists them, else every `full_attention_interval`-th block
        // attends and the others are DeltaNets.
        let recurrent: Vec<bool> = match keys.raw(&format!("{}.attention.recurrent_layers", header.architecture)) {
            Some(Metadata::Array(items)) => items.iter().map(|item| matches!(item, Metadata::Bool(true))).collect(),
            _ => {
                let interval = if keys.has("full_attention_interval") { keys.count("full_attention_interval")? } else { 4 };
                (0..blocks).map(|layer| (layer + 1) % interval.max(1) != 0).collect()
            }
        };
        if recurrent.len() < layers {
            return Err("qwen35moe recurrent layers do not cover every block".into());
        }
        let mixers = recurrent[..layers].iter().map(|&recurrent| if recurrent { Mixer::DeltaNet } else { Mixer::Attention }).collect();
        let heads = keys.count("attention.head_count")?;
        let kv_heads = keys.count("attention.head_count_kv")?;
        let head_dim = keys.count("attention.key_length")?;
        if keys.count("attention.value_length")? != head_dim {
            return Err("qwen35moe attention keys and values differ in size".into());
        }
        let rope_dims = keys.count("rope.dimension_count")?;
        let state_dim = keys.count("ssm.state_size")?;
        let k_heads = keys.count("ssm.group_count")?;
        let v_heads = keys.count("ssm.time_step_rank")?;
        if keys.count("ssm.inner_size")? != v_heads * state_dim {
            return Err("qwen35moe DeltaNet value width is not its heads times the state size".into());
        }
        let hp = Hparams {
            vocabulary,
            width: keys.count("embedding_length")?,
            norm_eps: keys.float("attention.layer_norm_rms_epsilon")? as f32,
            mixers,
            nextn,
            context: keys.count("context_length")?,
            heads,
            kv_heads,
            head_dim,
            rope_dims,
            rope_base: keys.float("rope.freq_base")?,
            conv_kernel: keys.count("ssm.conv_kernel")?,
            state_dim,
            k_heads,
            v_heads,
            experts: keys.count("expert_count")?,
            experts_used: keys.count("expert_used_count")?,
            expert_ff: keys.count("expert_feed_forward_length")?,
            shared_ff: keys.count("expert_shared_feed_forward_length")?,
        };
        let valid = hp.heads > 0 && hp.kv_heads > 0 && hp.heads.is_multiple_of(hp.kv_heads) && hp.rope_dims <= hp.head_dim
            && hp.rope_dims.is_multiple_of(2) && hp.k_heads > 0 && hp.v_heads.is_multiple_of(hp.k_heads) && hp.conv_kernel > 0
            && hp.experts_used > 0 && hp.experts_used <= hp.experts;
        if !valid {
            return Err("qwen35moe hyperparameters are inconsistent".into());
        }
        Ok(hp)
    }

    /// The main blocks.
    pub fn layers(&self) -> usize {
        self.mixers.len()
    }

    /// Width of a DeltaNet's convolved QKV: queries and keys over the key heads, values over
    /// the value heads.
    pub fn conv_channels(&self) -> usize {
        2 * self.k_heads * self.state_dim + self.v_heads * self.state_dim
    }

    /// Rotary frequency of each rotated pair `i` (text positions: every IMRoPE section reads the
    /// same position).
    pub fn rope_frequencies(&self) -> Vec<f64> {
        (0..self.rope_dims / 2).map(|i| self.rope_base.powf(-2.0 * i as f64 / self.rope_dims as f64)).collect()
    }
}
