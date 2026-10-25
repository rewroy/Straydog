//! DeepSeek-V4.1 (`deepseek41`): a causal encoder-decoder of hyper-connected MoE blocks with
//! latent attention over a sliding window plus compressed, index-selected rows (CSA2), and
//! n-gram Engram tables. The plan below follows the reference implementation's definitions.

pub mod engine;
pub mod serve;

/// DeepSeek-V4.1's chat template: Unsloth's port of the model's encoding.py (the UD GGUFs'),
/// which renders the reference encoder's prompts (tests/golden/deepseek41_chat.json). Some GGUFs
/// carry another: ISTA's GSQ-RCO writes the DSML tags without the leading space the model writes
/// them with ("<｜DSML｜tool_calls>" for "<｜DSML｜ calls>"), so a reply that called a tool never
/// read back as the model wrote it.
pub const CHAT_TEMPLATE: &str = include_str!("chat_template.jinja");

use super::{Keys, Tensors};
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
    /// Per module, the table its rows (`offsets` + bucket) are in: one table for every module
    /// (offsets running on across them) or one of its own each (offsets from 0).
    pub tables: Vec<String>,
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

/// A metadata key under its current name or the name older GGUFs gave it.
fn key<'k>(keys: &Keys<'_>, current: &'k str, older: &'k str) -> &'k str {
    if keys.has(current) { current } else { older }
}

/// An integer table of the model's (I32 or I64, `dims`), exactly: none if the model has no such
/// tensor.
fn integers(tensors: Option<&dyn Tensors>, name: &str, dims: &[usize]) -> Result<Option<Vec<i64>>, String> {
    let Some(tensor) = tensors.and_then(|tensors| tensors.tensor(name).ok()) else { return Ok(None) };
    if tensor.dims != dims {
        return Err(format!("deepseek41 {name} is {:?}, not {dims:?}", tensor.dims));
    }
    let values: Vec<i64> = match tensor.kind {
        26 => tensor.data.chunks_exact(4).map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()) as i64).collect(),
        27 => tensor.data.chunks_exact(8).map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap())).collect(),
        kind => return Err(format!("deepseek41 {name} is of type {kind}, not an integer table")),
    };
    if values.len() != dims.iter().product::<usize>() {
        return Err(format!("deepseek41 {name} is short"));
    }
    Ok(Some(values))
}

impl Hparams {
    /// From the metadata alone: a GGUF that keeps Engram's hash parameters there.
    pub fn from_header(header: &ModelHeader) -> Result<Hparams, String> {
        Self::from_parts(header, None)
    }

    /// From the metadata and the tensors: also a GGUF that keeps Engram's hash parameters in
    /// integer tensors (engram_token_map, _multipliers, _primes, _offsets).
    pub fn from_model(header: &ModelHeader, tensors: &impl Tensors) -> Result<Hparams, String> {
        Self::from_parts(header, Some(tensors as &dyn Tensors))
    }

    fn from_parts(header: &ModelHeader, tensors: Option<&dyn Tensors>) -> Result<Hparams, String> {
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
        let kv_sources = keys.integers(key(&keys, "kv_source_layer_ids", "attention.kv_source_layers"))?;
        let index_sources = keys.integers(key(&keys, "index_source_layer_ids", "attention.indexer.source_layers"))?;
        if ratios.len() < layers {
            return Err("deepseek41 compress ratios do not cover every layer".into());
        }
        let clamp_routed = keys.per_layer("swiglu_clamp_exp", layers)?;
        let clamp_shared = if keys.has("swiglu_clamp_shexp") { keys.per_layer("swiglu_clamp_shexp", layers)? } else { clamp_routed.clone() };
        let engram_key = key(&keys, "engram.layer_ids", "ple.layers");
        let engram_layers = if keys.has(engram_key) { keys.integers(engram_key)? } else { Vec::new() };
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
        let candidate = key(&keys, "attention.candidate_source_layer_id", "attention.indexer.candidate_source_layer");
        let candidates = if keys.has(candidate) {
            let layer = keys.count(candidate)?;
            let blocks = keys.count(key(&keys, "attention.candidate_topk_blocks", "attention.indexer.candidate_top_k_blocks"))?;
            let block = keys.count(key(&keys, "attention.candidate_block_size", "attention.indexer.candidate_block_size"))?;
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
            let ngram = keys.count(key(&keys, "engram_max_ngram_size", "ple.ngram_size"))?;
            let heads = keys.count(key(&keys, "engram_n_heads", "ple.heads_per_ngram"))?;
            let columns = ngram.saturating_sub(1) * heads;
            let modules = engram_layers.len();
            // the hash parameters: in the metadata, or in integer tensors (a table each, row a module)
            let words = |metadata: &str, tensor: &str, per_module: usize| -> Result<Vec<u64>, String> {
                if keys.has(metadata) {
                    return keys.words(metadata);
                }
                let values = integers(tensors, tensor, &[per_module, modules])?
                    .ok_or_else(|| format!("deepseek41 Engram {metadata} is missing (in the metadata or as {tensor})"))?;
                // (stored as I64; the multipliers are bits of a u64)
                Ok(values.into_iter().map(|value| value as u64).collect())
            };
            let token_map = if keys.has("engram.token_map") {
                keys.integers("engram.token_map")?
            } else {
                integers(tensors, "engram_token_map.weight", &[vocabulary])?.ok_or("deepseek41 Engram token map is missing")?
            };
            let pad_token = keys.count(key(&keys, "engram_pad_token_id", "engram.pad_token_id"))?;
            // a table of its own for each module (offsets from 0) where the model has them
            let own: Vec<String> = engram_layers.iter().map(|layer| format!("blk.{layer}.engram_embd.weight")).collect();
            let separate = tensors.is_some_and(|tensors| own.iter().all(|name| tensors.tensor(name).is_ok()));
            let engram = Engram {
                ngram,
                heads,
                head_dim: keys.count(key(&keys, "engram_head_dim", "embedding_length_per_layer_input"))?,
                multipliers: words("ple.layer_multipliers", "engram_multipliers.weight", ngram)?,
                primes: words("ple.head_vocab_sizes", "engram_primes.weight", columns)?,
                offsets: words("ple.head_offsets", "engram_offsets.weight", columns)?,
                pad: *token_map.get(pad_token).ok_or("Engram padding token outside the token map")?,
                token_map,
                tables: if separate { own } else { vec!["per_layer_token_embd.weight".to_string(); modules] },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Tensor;
    use std::collections::BTreeMap;
    use straydog_gguf::Metadata;

    /// A model's tensors by name: kind, dims, bytes.
    struct Fake(BTreeMap<&'static str, (u32, Vec<usize>, Vec<u8>)>);

    impl Tensors for Fake {
        fn tensor(&self, name: &str) -> Result<Tensor<'_>, String> {
            let (kind, dims, data) = self.0.get(name).ok_or_else(|| format!("tensor {name} is missing"))?;
            Ok(Tensor { kind: *kind, dims: dims.clone(), data })
        }

        fn verifier(&self) -> Option<crate::experts::Verifier> {
            None
        }
    }

    /// A three-layer DeepSeek-V4.1 header under the current key names, Engram before layers 0
    /// and 2 (tri-grams, two heads), five tokens.
    fn header() -> ModelHeader {
        let mut metadata = BTreeMap::new();
        let mut put = |key: &str, value: Metadata| { metadata.insert(format!("deepseek41.{key}"), value); };
        let numbers = |values: &[u64]| Metadata::Array(values.iter().map(|&value| Metadata::Unsigned(value)).collect());
        for (key, value) in [("block_count", 3), ("embedding_length", 16), ("attention.head_count", 4), ("attention.key_length", 8),
                             ("rope.dimension_count", 4), ("attention.q_lora_rank", 8), ("attention.output_group_count", 2),
                             ("attention.output_lora_rank", 8), ("attention.sliding_window", 8), ("expert_count", 4),
                             ("expert_used_count", 2), ("expert_feed_forward_length", 8), ("expert_shared_count", 1),
                             ("expert_gating_func", 4), ("attention.indexer.head_count", 2), ("attention.indexer.key_length", 8),
                             ("attention.indexer.top_k", 4), ("hyper_connection.count", 2), ("hyper_connection.sinkhorn_iterations", 3),
                             ("engram_max_ngram_size", 3), ("engram_n_heads", 2), ("engram_head_dim", 4), ("engram_pad_token_id", 1)] {
            put(key, Metadata::Unsigned(value));
        }
        for (key, value) in [("attention.layer_norm_rms_epsilon", 1e-6), ("expert_weights_scale", 1.5), ("hyper_connection.epsilon", 1e-6),
                             ("rope.freq_base", 10000.0), ("attention.compress_rope_freq_base", 160000.0), ("swiglu_clamp_exp", 10.0)] {
            put(key, Metadata::Float(value));
        }
        put("expert_weights_norm", Metadata::Bool(true));
        put("attention.compress_ratios", numbers(&[0, 1, 1]));
        put("kv_source_layer_ids", numbers(&[1]));
        put("index_source_layer_ids", numbers(&[1]));
        put("engram.layer_ids", numbers(&[0, 2]));
        metadata.insert("tokenizer.ggml.tokens".into(), Metadata::Array((0..5).map(|id| Metadata::Text(format!("t{id}"))).collect()));
        ModelHeader { architecture: "deepseek41".into(), declared_context: None, alignment: 32, data_offset: 0, file_bytes: 0,
                      shard_index: 0, shard_count: 1, tensors: Vec::new(), integer_arrays: BTreeMap::new(), metadata }
    }

    fn i64s(values: &[i64]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    /// The renamed keys and the Engram's hash parameters from integer tables (exact, a multiplier
    /// above i64 kept as its bits), each module reading a table of its own.
    #[test]
    fn current_keys_and_integer_tables() {
        let multipliers: Vec<i64> = vec![3, 5, -7, 11, 13, 17];
        let tensors = Fake(BTreeMap::from([
            ("engram_token_map.weight", (26, vec![5], [4i32, 3, 2, 1, -1].iter().flat_map(|v| v.to_le_bytes()).collect())),
            ("engram_multipliers.weight", (27, vec![3, 2], i64s(&multipliers))),
            ("engram_primes.weight", (27, vec![4, 2], i64s(&[101, 103, 107, 109, 113, 127, 131, 137]))),
            ("engram_offsets.weight", (27, vec![4, 2], i64s(&[0, 101, 204, 311, 0, 113, 240, 371]))),
            ("blk.0.engram_embd.weight", (20, vec![32, 420], Vec::new())),
            ("blk.2.engram_embd.weight", (20, vec![32, 508], Vec::new())),
        ]));
        let hp = Hparams::from_model(&header(), &tensors).unwrap();
        let engram = hp.engram.as_ref().unwrap();
        assert_eq!((engram.ngram, engram.heads, engram.head_dim, engram.pad), (3, 2, 4, 3));
        assert_eq!(engram.token_map, [4, 3, 2, 1, -1]);
        assert_eq!(engram.multipliers[2], (-7i64) as u64);
        assert_eq!(engram.offsets[4..], [0, 113, 240, 371]);
        assert_eq!(engram.tables, ["blk.0.engram_embd.weight", "blk.2.engram_embd.weight"]);
        assert_eq!(hp.layers.iter().map(|layer| layer.engram).collect::<Vec<_>>(), [Some(0), None, Some(1)]);
        assert_eq!(hp.layers[2].compressed, Some(Compressed { ratio: 1, kv_source: 1, index_source: 1 }));
        // without the tables in the metadata or the tensors, the Engram cannot be set up
        assert!(Hparams::from_header(&header()).unwrap_err().contains("Engram"));
        let mut short = tensors;
        short.0.insert("engram_primes.weight", (27, vec![4, 1], i64s(&[101, 103, 107, 109])));
        assert!(Hparams::from_model(&header(), &short).is_err());
    }
}
