//! One GPU's share of every layer when several split the layers' work (tensor parallel): which
//! heads, shared-expert rows, routed-expert rows and vocabulary rows it holds, the hyperparameters
//! of its share, and its slices of the dense tensors. Each GPU keeps the whole residual stream;
//! the out-projections, the shared expert's down and the routed experts' downs give partial sums
//! the GPUs add together (gpu/device_link.hpp).
//!
//! The split follows the heads' grouping so that every part is a smaller model of the same shape:
//! - Attention: heads and KV heads in contiguous blocks (a head reads KV head h / (heads /
//!   kv_heads), so each part keeps its groups whole).
//! - Gated DeltaNet: key heads in contiguous blocks, and with them the value heads that read them
//!   (value head h reads key head h % k_heads, so a part's value heads are those congruent to its
//!   key heads; in the part, local value head j reads local key head j % its key heads, as the
//!   kernel computes it).
//! - The shared expert: its ffn rows (gate and up) and the matching columns of its down.
//! - Routed experts: alike, a block of every expert's ffn rows, so that every part runs every route
//!   of a step (with whole experts a part, expert parallel, the part with fewer of a step's routes
//!   waited for the other: their MoE work differed by ~19 us a layer on Qwen3.6).
//! - The head: a contiguous block of vocabulary rows; the host joins the logits.

use std::ops::Range;

use super::{Hparams, Mixer};
use crate::arch::Tensor;
use crate::formats;

/// This GPU's part of `count` (1: the whole layers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Part {
    pub index: usize,
    pub count: usize,
}

impl Part {
    pub const WHOLE: Part = Part { index: 0, count: 1 };

    pub fn is_whole(&self) -> bool {
        self.count == 1
    }

    /// This part's block of `total` items (contiguous; `total` must divide evenly).
    pub fn block(&self, total: usize) -> Range<usize> {
        let size = total / self.count;
        self.index * size..(self.index + 1) * size
    }

    /// The routed experts this part runs (each one's block of ffn rows): all of them.
    pub fn experts(&self, hp: &Hparams) -> Range<usize> {
        0..hp.experts
    }

    /// The vocabulary rows of the head this part computes.
    pub fn vocabulary(&self, hp: &Hparams) -> Range<usize> {
        let size = hp.vocabulary.div_ceil(self.count);
        (self.index * size).min(hp.vocabulary)..((self.index + 1) * size).min(hp.vocabulary)
    }

    /// Whether the model splits into `count` such parts.
    pub fn fits(hp: &Hparams, count: usize) -> Result<(), String> {
        let even = |total: usize, what: &str| -> Result<(), String> {
            if total.is_multiple_of(count) { Ok(()) } else { Err(format!("qwen35moe {what} ({total}) do not split across {count} GPUs")) }
        };
        even(hp.heads, "attention heads")?;
        even(hp.kv_heads, "KV heads")?;
        even(hp.k_heads, "DeltaNet key heads")?;
        even(hp.expert_ff, "routed expert rows")?;
        even(hp.shared_ff, "shared expert rows")?;
        if !(hp.shared_ff / count).is_multiple_of(256) {
            return Err("qwen35moe shared expert rows split into partial blocks".into());
        }
        Ok(())
    }

    /// The DeltaNet's key heads of this part, and its value heads in order (global indices).
    pub fn delta_heads(&self, hp: &Hparams) -> (Range<usize>, Vec<usize>) {
        let keys = self.block(hp.k_heads);
        let values = (0..hp.v_heads).filter(|head| keys.contains(&(head % hp.k_heads))).collect();
        (keys, values)
    }

    /// The hyperparameters of this part: a model of the same shape with its heads, expert rows
    /// (routed and shared) and head rows.
    pub fn hparams(&self, hp: &Hparams) -> Hparams {
        let mut local = hp.clone();
        local.heads = hp.heads / self.count;
        local.kv_heads = hp.kv_heads / self.count;
        local.k_heads = hp.k_heads / self.count;
        local.v_heads = hp.v_heads / self.count;
        local.expert_ff = hp.expert_ff / self.count;
        local.shared_ff = hp.shared_ff / self.count;
        local.vocabulary = self.vocabulary(hp).len();
        local
    }

    /// This part's slice of a dense tensor of `layer` (None: the head or final norm), or None
    /// when the part holds the whole tensor.
    pub fn slice(&self, hp: &Hparams, layer: Option<usize>, role: &str) -> Option<Slice> {
        if self.is_whole() {
            return None;
        }
        let d = hp.state_dim;
        let rows = |heads: &[usize], width: usize, offset: usize| -> Vec<Range<usize>> {
            merge(heads.iter().map(|&head| offset + head * width..offset + (head + 1) * width).collect())
        };
        let Some(layer) = layer else {
            return (role == "output").then(|| Slice::Rows(vec![self.vocabulary(hp)]));
        };
        match (hp.mixers[layer], role) {
            (Mixer::DeltaNet, "attn_qkv" | "ssm_conv1d") => {
                let (keys, values) = self.delta_heads(hp);
                let keys: Vec<usize> = keys.collect();
                let mut ranges = rows(&keys, d, 0);
                ranges.extend(rows(&keys, d, hp.k_heads * d));
                ranges.extend(rows(&values, d, 2 * hp.k_heads * d));
                Some(Slice::Rows(merge(ranges)))
            }
            (Mixer::DeltaNet, "attn_gate") => Some(Slice::Rows(rows(&self.delta_heads(hp).1, d, 0))),
            (Mixer::DeltaNet, "ssm_beta" | "ssm_alpha") => Some(Slice::Rows(rows(&self.delta_heads(hp).1, 1, 0))),
            (Mixer::DeltaNet, "ssm_dt" | "ssm_a") => Some(Slice::Columns(rows(&self.delta_heads(hp).1, 1, 0))),
            (Mixer::DeltaNet, "ssm_out") => Some(Slice::Columns(rows(&self.delta_heads(hp).1, d, 0))),
            (Mixer::Attention, "attn_q") => Some(Slice::Rows(vec![scale(self.block(hp.heads), 2 * hp.head_dim)])),
            (Mixer::Attention, "attn_k" | "attn_v") => Some(Slice::Rows(vec![scale(self.block(hp.kv_heads), hp.head_dim)])),
            (Mixer::Attention, "attn_output") => Some(Slice::Columns(vec![scale(self.block(hp.heads), hp.head_dim)])),
            (_, "ffn_gate_shexp" | "ffn_up_shexp") => Some(Slice::Rows(vec![self.block(hp.shared_ff)])),
            (_, "ffn_down_shexp") => Some(Slice::Columns(vec![self.block(hp.shared_ff)])),
            _ => None,
        }
    }
}

fn scale(range: Range<usize>, by: usize) -> Range<usize> {
    range.start * by..range.end * by
}

/// Adjacent ranges joined.
fn merge(ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in ranges {
        match merged.last_mut() {
            Some(last) if last.end == range.start => last.end = range.end,
            _ => merged.push(range),
        }
    }
    merged
}

/// A part's rows (output rows: GGUF's second dimension) or columns (its first) of a matrix, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slice {
    Rows(Vec<Range<usize>>),
    Columns(Vec<Range<usize>>),
}

impl Slice {
    /// The slice's bytes and dimensions (GGUF order: columns, then rows).
    pub fn take(&self, tensor: &Tensor<'_>) -> Result<(Vec<u8>, Vec<usize>), String> {
        let cols = tensor.dims[0];
        let rows = tensor.dims.get(1).copied().unwrap_or(1);
        let row_bytes = formats::matrix_bytes(tensor.kind, 1, cols).ok_or_else(|| format!("unsupported tensor type {}", tensor.kind))?;
        match self {
            Slice::Rows(ranges) => {
                let mut out = Vec::new();
                for range in ranges {
                    if range.end > rows {
                        return Err("a part's rows past its tensor".into());
                    }
                    out.extend_from_slice(&tensor.data[range.start * row_bytes..range.end * row_bytes]);
                }
                let taken = ranges.iter().map(|range| range.len()).sum();
                Ok((out, if tensor.dims.len() > 1 { vec![cols, taken] } else { vec![cols] }))
            }
            Slice::Columns(ranges) => {
                let spans = ranges.iter().map(|range| {
                    let (skip, take) = (formats::matrix_bytes(tensor.kind, 1, range.start), formats::matrix_bytes(tensor.kind, 1, range.len()));
                    match (skip, take) {
                        (Some(skip), Some(take)) if range.end <= cols => Ok((skip, take)),
                        _ => Err(format!("columns {range:?} of a type {} matrix split a block", tensor.kind)),
                    }
                }).collect::<Result<Vec<_>, String>>()?;
                let mut out = Vec::new();
                for row in tensor.data.chunks_exact(row_bytes).take(rows) {
                    for &(skip, take) in &spans {
                        out.extend_from_slice(&row[skip..skip + take]);
                    }
                }
                let taken = ranges.iter().map(|range| range.len()).sum();
                let mut dims = tensor.dims.clone();
                dims[0] = taken;
                Ok((out, dims))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    fn qwen36() -> Hparams {
        Hparams {
            vocabulary: 248320, width: 2048, norm_eps: 1e-6,
            mixers: (0..40).map(|layer| if (layer + 1) % 4 == 0 { Mixer::Attention } else { Mixer::DeltaNet }).collect(),
            nextn: 0, context: 262144, heads: 16, kv_heads: 2, head_dim: 256, rope_dims: 64, rope_base: 1e7, conv_kernel: 4,
            state_dim: 128, k_heads: 16, v_heads: 32, experts: 256, experts_used: 8, expert_ff: 512, shared_ff: 512,
        }
    }

    #[test]
    fn a_part_holds_its_key_heads_and_the_value_heads_that_read_them() {
        let hp = qwen36();
        Part::fits(&hp, 2).unwrap();
        let second = Part { index: 1, count: 2 };
        let (keys, values) = second.delta_heads(&hp);
        assert_eq!(keys, 8..16);
        assert_eq!(values, (8..16).chain(24..32).collect::<Vec<_>>());
        // local value head j reads local key head j % 8: global value 24 + i reads key 8 + i
        for (local, &global) in values.iter().enumerate() {
            assert_eq!(global % hp.k_heads - keys.start, local % (hp.k_heads / 2));
        }
        let d = 128;
        assert_eq!(second.slice(&hp, Some(0), "attn_qkv"),
                   Some(Slice::Rows(vec![8 * d..16 * d, 24 * d..32 * d, 32 * d + 8 * d..32 * d + 16 * d, 32 * d + 24 * d..32 * d + 32 * d])));
        assert_eq!(second.slice(&hp, Some(0), "ssm_out"), Some(Slice::Columns(vec![8 * d..16 * d, 24 * d..32 * d])));
        assert_eq!(second.slice(&hp, Some(3), "attn_q"), Some(Slice::Rows(vec![8 * 512..16 * 512])));
        assert_eq!(second.slice(&hp, Some(3), "attn_k"), Some(Slice::Rows(vec![256..512])));
        assert_eq!(second.slice(&hp, Some(3), "attn_q_norm"), None);
        assert_eq!(second.slice(&hp, None, "output"), Some(Slice::Rows(vec![124160..248320])));
        let local = second.hparams(&hp);
        assert_eq!((local.heads, local.kv_heads, local.k_heads, local.v_heads, local.expert_ff, local.shared_ff, local.vocabulary),
                   (8, 1, 8, 16, 256, 256, 124160));
        assert_eq!(second.experts(&hp), 0..256);
        assert_eq!(local.conv_channels(), 2 * 8 * d + 16 * d);
    }

    #[test]
    fn slices_take_whole_rows_and_whole_blocks_of_columns() {
        // F32 4 x 3 (columns, rows): rows 1..3, columns 1..3
        let data: Vec<u8> = (0..12).flat_map(|value: u32| (value as f32).to_le_bytes()).collect();
        let tensor = Tensor { kind: 0, dims: vec![4, 3], data: &data };
        let floats = |bytes: Vec<u8>| bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        let (rows, dims) = Slice::Rows(vec![1..3]).take(&tensor).unwrap();
        assert_eq!((floats(rows), dims), (vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], vec![4, 2]));
        let (columns, dims) = Slice::Columns(vec![1..2, 3..4]).take(&tensor).unwrap();
        assert_eq!((floats(columns), dims), (vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0], vec![2, 3]));
        // Q8_0 (34-byte blocks of 32): columns must take whole blocks
        let q8 = vec![0u8; 34 * 2];
        let tensor = Tensor { kind: 8, dims: vec![64, 1], data: &q8 };
        assert!(Slice::Columns(vec![0..16]).take(&tensor).is_err());
        assert_eq!(Slice::Columns(vec![32..64]).take(&tensor).unwrap().1, vec![32, 1]);
    }
}
