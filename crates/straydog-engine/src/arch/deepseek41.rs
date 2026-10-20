//! DeepSeek-V4.1 (`deepseek41`): a causal encoder-decoder of hyper-connected MoE blocks with
//! latent attention over a sliding window plus compressed, index-selected rows (CSA2), and
//! n-gram Engram tables. The plan below follows the reference implementation's definitions.

use super::Keys;
use straydog_gguf::ModelHeader;

/// How a layer's attention obtains the compressed rows it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Compressed {
    /// Tokens pooled into one compressed row (1 or 2).
    pub ratio: usize,
    /// The layer whose compressed rows and index keys this layer reads (itself: "Full").
    pub kv_source: usize,
    /// The layer whose top-k selection this layer uses (itself: it scores; else "Reuse").
    pub index_source: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// None: sliding window only.
    pub compressed: Option<Compressed>,
    /// The Engram module (index into the tables' heads) applied before this layer, if any.
    pub engram: Option<usize>,
    /// SwiGLU clamps of the routed and shared experts (0: none).
    pub clamp_routed: f32,
    pub clamp_shared: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gating {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rope {
    pub base: f64,
    /// YaRN: (factor, original context, beta fast, beta slow); None keeps the base frequencies.
    pub yarn: Option<(f64, f64, f64, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Engram {
    pub ngram: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// Per module: `ngram` multipliers; per module and column: bucket count and row offset.
    pub multipliers: Vec<u64>,
    pub primes: Vec<u64>,
    pub offsets: Vec<u64>,
    /// Token id to the tokenizer-compressed id the hash uses; the padding token's entry fills
    /// shifts before the sequence start.
    pub token_map: Vec<i64>,
    pub pad: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hparams {
    pub vocabulary: usize,
    pub width: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub q_rank: usize,
    pub out_groups: usize,
    pub out_rank: usize,
    pub window: usize,
    pub norm_eps: f32,
    pub experts: usize,
    pub experts_used: usize,
    pub expert_ff: usize,
    pub shared_experts: usize,
    pub gating: Gating,
    pub route_scale: f32,
    pub route_normalize: bool,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_top_k: usize,
    /// The layer whose indexer picks candidate blocks for later indexers: (layer, blocks, block size).
    pub candidates: Option<(usize, usize, usize)>,
    pub hc: usize,
    pub sinkhorn_iterations: usize,
    pub hc_eps: f32,
    /// Window-only layers rotate at `rope`; compressed ones at `compressed_rope`.
    pub rope: Rope,
    pub compressed_rope: Rope,
    pub engram: Option<Engram>,
    pub layers: Vec<Layer>,
}

impl Hparams {
    pub fn from_header(header: &ModelHeader) -> Result<Hparams, String> {
        if header.architecture != "deepseek41" {
            return Err(format!("expected a deepseek41 GGUF, found {}", header.architecture));
        }
        let keys = Keys::new(header);
        let layers = keys.count("block_count")?;
        let vocabulary = keys
            .raw("tokenizer.ggml.tokens")
            .and_then(|tokens| tokens.array())
            .map(|tokens| tokens.len())
            .ok_or("GGUF tokenizer tokens are missing")?;
        let ratios = keys.integers("attention.compress_ratios")?;
        let kv_sources = keys.integers("attention.kv_source_layers")?;
        let index_sources = keys.integers("attention.indexer.source_layers")?;
        if ratios.len() < layers {
            return Err("deepseek41 compress ratios do not cover every layer".into());
        }
        let clamp_routed = keys.per_layer("swiglu_clamp_exp", layers)?;
        let clamp_shared = if keys.has("swiglu_clamp_shexp") { keys.per_layer("swiglu_clamp_shexp", layers)? } else { clamp_routed.clone() };
        let engram_layers = if keys.has("ple.layers") { keys.integers("ple.layers")? } else { Vec::new() };
        // Every compressed layer reads the rows and the selection of the latest source at or
        // before it, which must compress at the same ratio.
        let (mut kv_last, mut index_last) = (None, None);
        let mut plan = Vec::with_capacity(layers);
        for layer in 0..layers {
            let ratio = usize::try_from(ratios[layer]).map_err(|_| "negative compress ratio")?;
            let is_kv = kv_sources.contains(&(layer as i64));
            let is_index = index_sources.contains(&(layer as i64));
            if (is_kv || is_index) && ratio == 0 {
                return Err(format!("deepseek41 source layer {layer} does not compress"));
            }
            kv_last = if is_kv { Some(layer) } else { kv_last };
            index_last = if is_index { Some(layer) } else { index_last };
            let compressed = if ratio == 0 {
                None
            } else {
                let (Some(kv_source), Some(index_source)) = (kv_last, index_last) else {
                    return Err(format!("deepseek41 layer {layer} compresses without a source"));
                };
                if ratio > 2 || index_source < kv_source || ratios[kv_source] != ratio as i64 || ratios[index_source] != ratio as i64 {
                    return Err(format!("deepseek41 layer {layer}: ratio {ratio} without a matching source"));
                }
                Some(Compressed { ratio, kv_source, index_source })
            };
            plan.push(Layer {
                compressed,
                engram: engram_layers.iter().position(|&at| at == layer as i64),
                clamp_routed: clamp_routed[layer] as f32,
                clamp_shared: clamp_shared[layer] as f32,
            });
        }
        let candidates = if keys.has("attention.indexer.candidate_source_layer") {
            let layer = keys.count("attention.indexer.candidate_source_layer")?;
            let blocks = keys.count("attention.indexer.candidate_top_k_blocks")?;
            let block = keys.count("attention.indexer.candidate_block_size")?;
            if layer >= layers || plan[layer].compressed.map(|c| c.index_source) != Some(layer) || blocks == 0 || block == 0 {
                return Err("deepseek41 candidate source is not an index source".into());
            }
            Some((layer, blocks, block))
        } else {
            None
        };
        let gating = match keys.integer("expert_gating_func")? {
            1 => Gating::Softmax,
            2 => Gating::Sigmoid,
            4 => Gating::SqrtSoftplus,
            other => return Err(format!("unsupported expert gating function {other}")),
        };
        let yarn = if keys.has("rope.scaling.factor") {
            Some((
                keys.float("rope.scaling.factor")?,
                keys.float("rope.scaling.original_context_length")?,
                keys.float("rope.scaling.yarn_beta_fast")?,
                keys.float("rope.scaling.yarn_beta_slow")?,
            ))
        } else {
            None
        };
        let engram = if engram_layers.is_empty() {
            None
        } else {
            let ngram = keys.count("ple.ngram_size")?;
            let heads = keys.count("ple.heads_per_ngram")?;
            let columns = (ngram - 1) * heads;
            let modules = engram_layers.len();
            let token_map = keys.integers("engram.token_map")?;
            let pad_token = keys.count("engram.pad_token_id")?;
            let engram = Engram {
                ngram,
                heads,
                head_dim: keys.count("embedding_length_per_layer_input")?,
                multipliers: keys.words("ple.layer_multipliers")?,
                primes: keys.words("ple.head_vocab_sizes")?,
                offsets: keys.words("ple.head_offsets")?,
                pad: *token_map.get(pad_token).ok_or("Engram padding token outside the token map")?,
                token_map,
            };
            if ngram < 2 || heads == 0 || engram.multipliers.len() != modules * ngram || engram.primes.len() != modules * columns
                || engram.offsets.len() != modules * columns || engram.primes.iter().any(|&prime| prime < 2)
            {
                return Err("deepseek41 Engram metadata is inconsistent".into());
            }
            Some(engram)
        };
        Ok(Hparams {
            vocabulary,
            width: keys.count("embedding_length")?,
            heads: keys.count("attention.head_count")?,
            head_dim: keys.count("attention.key_length")?,
            rope_dims: keys.count("rope.dimension_count")?,
            q_rank: keys.count("attention.q_lora_rank")?,
            out_groups: keys.count("attention.output_group_count")?,
            out_rank: keys.count("attention.output_lora_rank")?,
            window: keys.count("attention.sliding_window")?,
            norm_eps: keys.float("attention.layer_norm_rms_epsilon")? as f32,
            experts: keys.count("expert_count")?,
            experts_used: keys.count("expert_used_count")?,
            expert_ff: keys.count("expert_feed_forward_length")?,
            shared_experts: keys.count("expert_shared_count")?,
            gating,
            route_scale: keys.float("expert_weights_scale")? as f32,
            route_normalize: keys.flag("expert_weights_norm")?,
            index_heads: keys.count("attention.indexer.head_count")?,
            index_dim: keys.count("attention.indexer.key_length")?,
            index_top_k: keys.count("attention.indexer.top_k")?,
            candidates,
            hc: keys.count("hyper_connection.count")?,
            sinkhorn_iterations: keys.count("hyper_connection.sinkhorn_iterations")?,
            hc_eps: keys.float("hyper_connection.epsilon")? as f32,
            rope: Rope { base: keys.float("rope.freq_base")?, yarn: None },
            compressed_rope: Rope { base: keys.float("attention.compress_rope_freq_base")?, yarn },
            engram,
            layers: plan,
        })
    }

    /// Engram columns per module: every n-gram order above one, times the hash heads.
    pub fn engram_columns(&self) -> usize {
        self.engram.as_ref().map_or(0, |engram| (engram.ngram - 1) * engram.heads)
    }
}

impl Engram {
    /// Table rows for `token` at `position`, given the up to `ngram - 1` tokens before it
    /// (`history`, oldest first). A shift that reaches before the sequence start, or past a
    /// token outside the map, and every longer one, takes the padding id.
    pub fn rows(&self, module: usize, token: u32, history: &[u32], position: usize, out: &mut Vec<u32>) {
        let columns = (self.ngram - 1) * self.heads;
        let multipliers = &self.multipliers[module * self.ngram..(module + 1) * self.ngram];
        let primes = &self.primes[module * columns..(module + 1) * columns];
        let offsets = &self.offsets[module * columns..(module + 1) * columns];
        let compressed = |token: u32| self.token_map.get(token as usize).copied();
        let mut ids = vec![self.pad as u64; self.ngram];
        let mut blocked = false;
        for (shift, id) in ids.iter_mut().enumerate() {
            let source = if shift == 0 {
                Some(token)
            } else if position >= shift && history.len() >= shift {
                Some(history[history.len() - shift])
            } else {
                None
            };
            match source.and_then(compressed) {
                Some(value) if !blocked => *id = value as u64,
                _ => blocked = true,
            }
        }
        let mut rolling = ids[0].wrapping_mul(multipliers[0]);
        for order in 1..self.ngram {
            rolling ^= ids[order].wrapping_mul(multipliers[order]);
            for head in 0..self.heads {
                let column = (order - 1) * self.heads + head;
                out.push((rolling % primes[column] + offsets[column]) as u32);
            }
        }
    }
}
