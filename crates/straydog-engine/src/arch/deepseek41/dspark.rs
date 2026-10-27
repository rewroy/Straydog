//! The supported DSpark GGUF layout, before any weights are mapped or placed.
//! Read headers with `straydog_gguf::inspect`; this borrows their tensor descriptors only.

use std::collections::BTreeMap;

use straydog_gguf::{Metadata, ModelHeader, TensorInfo};

use crate::arch::Keys;

/// Validated DSpark descriptors. Payload authentication and placement are not performed here.
#[derive(Debug)]
pub struct Dspark<'a> {
    pub header: &'a ModelHeader,
    pub tensors: BTreeMap<&'a str, &'a TensorInfo>,
}

impl<'a> Dspark<'a> {
    /// Validate the three-stage DSpark against the target's declared DSpark configuration.
    pub fn from_headers(header: &'a ModelHeader, target: &ModelHeader) -> Result<Self, String> {
        if header.architecture != "dflash" || target.architecture != "deepseek41" {
            return Err(format!("DSpark expects dflash with a deepseek41 target, found {} with {}", header.architecture, target.architecture));
        }
        if header.shard_count != 1 || header.shard_index != 0 {
            return Err("DSpark expects one GGUF shard (index 0)".into());
        }
        let keys = Keys::new(header);
        for (key, expected) in [
            ("block_count", 3), ("embedding_length", 5120),
            ("attention.head_count", 64), ("attention.head_count_kv", 1),
            ("attention.key_length", 512), ("attention.value_length", 512),
            ("rope.dimension_count", 64), ("attention.q_lora_rank", 1280),
            ("attention.sliding_window", 128), ("feed_forward_length", 2304),
            ("expert_feed_forward_length", 2304), ("expert_count", 128),
            ("expert_used_count", 3), ("expert_shared_count", 1),
            ("expert_gating_func", 4), ("hyper_connection.count", 4),
            ("hyper_connection.sinkhorn_iterations", 20),
            ("attention.output_group_count", 8), ("attention.output_lora_rank", 1024),
            ("block_size", 5),
        ] {
            let actual = keys.integer(key)?;
            if actual != expected {
                return Err(format!("DSpark dflash.{key}: expected {expected}, found {actual}"));
            }
        }
        for (key, expected) in [
            ("rope.freq_base", 10000f32), ("attention.layer_norm_rms_epsilon", 1e-20),
            ("expert_weights_scale", 1.5), ("hyper_connection.epsilon", 1e-6),
        ] {
            let actual = keys.float(key)?;
            if actual as f32 != expected {
                return Err(format!("DSpark dflash.{key}: expected {expected}, found {actual}"));
            }
        }
        for key in ["expert_weights_norm", "hyper_connection.single_pass", "sample_from_anchor", "has_confidence_head"] {
            if !keys.flag(key)? {
                return Err(format!("DSpark dflash.{key}: expected true, found false"));
            }
        }
        for (key, expected) in [("attention.compress_ratios", [0, 0, 0]), ("target_layers", [37, 38, 39])] {
            let actual = keys.integers(key)?;
            if actual != expected {
                return Err(format!("DSpark dflash.{key}: expected {expected:?}, found {actual:?}"));
            }
        }
        for key in ["swiglu_clamp_exp", "swiglu_clamp_shexp"] {
            let actual = keys.floats(key)?;
            if actual != [10.0; 3] {
                return Err(format!("DSpark dflash.{key}: expected [10, 10, 10], found {actual:?}"));
            }
        }
        let target_keys = Keys::new(target);
        // (Unsloth's GGUFs declare the target's DSpark in full; GSQ-RCO's only its target layers)
        let declared = target.metadata.contains_key("deepseek41.dspark.block_size");
        let keys: &[(&str, i64)] = if declared {
            &[("embedding_length", 5120), ("dspark.block_size", 5), ("dspark.noise_token_id", 128799), ("dspark.markov_rank", 256),
              ("dspark.expert_count", 128), ("dspark.expert_used_count", 3)]
        } else {
            &[("embedding_length", 5120)]
        };
        for &(key, expected) in keys {
            let actual = target_keys.integer(key)?;
            if actual != expected {
                return Err(format!("DSpark target deepseek41.{key}: expected {expected}, found {actual}"));
            }
        }
        let layers = target_keys.integers(if declared { "dspark.target_layers" } else { "dspark_target_layer_ids" })?;
        if layers != [37, 38, 39] || target_keys.count("block_count")? <= 39 {
            return Err(format!("DSpark target layers must be [37, 38, 39] within the target: found {layers:?}"));
        }
        for (label, source) in [("draft", header), ("target", target)] {
            let vocabulary = source.metadata.get("tokenizer.ggml.tokens").and_then(Metadata::array).map(<[_]>::len);
            if vocabulary != Some(129280) {
                return Err(format!("DSpark {label} vocabulary: expected 129280 tokens, found {vocabulary:?}"));
            }
        }
        let noise = header.metadata.get("tokenizer.ggml.mask_token_id").and_then(Metadata::integer);
        if noise != Some(128799) {
            return Err(format!("DSpark tokenizer.ggml.mask_token_id: expected 128799, found {noise:?}"));
        }
        let tensors: BTreeMap<_, _> = header.tensors.iter().map(|tensor| (tensor.name.as_str(), tensor)).collect();
        if tensors.len() != header.tensors.len() {
            return Err("DSpark contains duplicate tensor names".into());
        }
        if tensors.len() != 78 {
            return Err(format!("DSpark expects exactly 78 tensors, found {}", tensors.len()));
        }
        let check = |name: &str, kind: u32, dimensions: &[u64]| -> Result<(), String> {
            let tensor = tensors.get(name).ok_or_else(|| format!("DSpark tensor missing: {name}"))?;
            if tensor.quant_type != kind || tensor.dimensions != dimensions {
                return Err(format!("DSpark tensor {name}: expected type {kind} shape {dimensions:?}, found type {} shape {:?}",
                                   tensor.quant_type, tensor.dimensions));
            }
            Ok(())
        };
        for stage in 0..3 {
            for &(suffix, kind, dimensions) in STAGE_TENSORS {
                check(&format!("blk.{stage}.{suffix}"), kind, dimensions)?;
            }
        }
        for &(name, kind, dimensions) in GLOBAL_TENSORS {
            check(name, kind, dimensions)?;
        }
        Ok(Self { header, tensors })
    }
}

const STAGE_TENSORS: &[(&str, u32, &[u64])] = &[
    ("attn_q_a.weight", 8, &[5120, 1280]), ("attn_q_b.weight", 8, &[1280, 32768]),
    ("attn_kv.weight", 8, &[5120, 512]),
    ("attn_output_a.weight", 8, &[4096, 8192]), ("attn_output_b.weight", 8, &[8192, 5120]),
    ("attn_norm.weight", 0, &[5120]), ("ffn_norm.weight", 0, &[5120]),
    ("attn_q_a_norm.weight", 0, &[1280]), ("attn_kv_a_norm.weight", 0, &[512]),
    ("attn_sinks.weight", 0, &[64]),
    ("hc_attn_fn.weight", 0, &[20480, 24]), ("hc_attn_base.weight", 0, &[24]), ("hc_attn_scale.weight", 0, &[3]),
    ("hc_ffn_fn.weight", 0, &[20480, 24]), ("hc_ffn_base.weight", 0, &[24]), ("hc_ffn_scale.weight", 0, &[3]),
    ("ffn_gate_inp.weight", 0, &[5120, 128]), ("exp_probs_b.bias", 0, &[128]),
    ("ffn_gate_exps.weight", 39, &[5120, 2304, 128]), ("ffn_up_exps.weight", 39, &[5120, 2304, 128]),
    ("ffn_down_exps.weight", 39, &[2304, 5120, 128]),
    ("ffn_gate_shexp.weight", 8, &[5120, 2304]), ("ffn_up_shexp.weight", 8, &[5120, 2304]),
    ("ffn_down_shexp.weight", 8, &[2304, 5120]),
];
const GLOBAL_TENSORS: &[(&str, u32, &[u64])] = &[
    ("fc.weight", 8, &[15360, 5120]), ("enc.output_norm.weight", 0, &[5120]), ("output_norm.weight", 0, &[5120]),
    ("markov_w1.weight", 30, &[256, 129280]), ("markov_w2.weight", 30, &[256, 129280]),
    ("conf_proj.weight", 0, &[5376, 1]),
];
