//! Images in a DeepSeek-V4.1 pass (the reference's `merge_image_embeddings`): an image's span of
//! positions takes the vision encoder's rows (and the learned START, NEWLINE and END rows) in place
//! of token embeddings, routes its experts with each layer's image bias, and gathers no Engram
//! rows (its gate shut, its n-grams cut: an image position is outside Engram's token map).
//!
//! The encoder runs on the first device, in the bytes of its arena (lent to the expert store
//! between long passes, taken back as a long pass would): its weights uploaded from the file
//! beside the bundle for each request's images, then its activations.

use std::sync::Arc;

use straydog_gpu::ffi;

use super::{Engine, Gating, Result};
use crate::gpu::Backend;
use crate::ops::Buffer;
use crate::arch::Tensors;
pub use crate::vision::IMAGE_TOKEN;
use crate::vision::{bf16, span, Patches, Slot, VisionConfig};


/// The runs of image positions among a pass's tokens: (first row, end).
pub(super) fn runs(tokens: &[u32]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (row, &token) in tokens.iter().enumerate() {
        if token < IMAGE_TOKEN {
            continue;
        }
        match runs.last_mut() {
            Some((_, end)) if *end == row => *end += 1,
            _ => runs.push((row, row + 1)),
        }
    }
    runs
}

const PATCH: &str = "vision.patch_embed.proj";

/// An encoder weight in the device's copy: a product's (FP16 there) or a vector (F32).
struct Weight {
    name: String,
    matrix: bool,
    offset: u64,
    bytes: u64,
}

fn block(layer: usize, part: &str) -> String {
    format!("vision.blocks.{layer}.{part}")
}

impl<T: Tensors> Engine<'_, T> {
    /// Whether the model reads images: its vision encoder's file is beside the bundle.
    pub fn vision(&self) -> bool {
        self.tensors.tensor(&format!("{PATCH}.weight")).is_ok()
    }

    /// The rows the next passes embed at image positions: each span's first position and its
    /// rows ([`Engine::encode_images`]), replacing those set before.
    pub fn set_images(&mut self, spans: Vec<(usize, Arc<Vec<f32>>)>) {
        self.images = spans;
    }

    /// The row set for an image position.
    pub(super) fn image_row(&self, position: usize) -> Result<&[f32]> {
        let width = self.hp.width;
        self.images.iter()
            .find(|(first, rows)| position >= *first && position < first + rows.len() / width)
            .map(|(first, rows)| &rows[(position - first) * width..(position - first + 1) * width])
            .ok_or_else(|| format!("no image rows for position {position}"))
    }

    /// Every device's copy of the routers' image biases (once).
    pub(super) fn upload_image_bias(&mut self) -> Result<()> {
        if self.shards.iter().all(|shard| shard.image_bias.is_some()) {
            return Ok(());
        }
        let experts = self.hp.experts;
        let mut values = Vec::with_capacity(self.hp.layers.len() * experts);
        for layer in 0..self.hp.layers.len() {
            // (in the model's own GGUF where its layout has them, else in the vision encoder's file)
            let name = format!("blk.{layer}.exp_probs_b_vl.bias");
            let tensor = self.tensors.tensor(&name).or_else(|_| self.tensors.tensor(&format!("layers.{layer}.ffn.gate.bias_vl")))
                .map_err(|_| "this model has no image routing (its vision encoder file is missing)")?;
            if tensor.kind != 0 || tensor.dims != [experts] {
                return Err(format!("{name} is not {experts} F32 values"));
            }
            values.extend(tensor.data.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())));
        }
        for shard in &mut self.shards {
            if shard.image_bias.is_none() {
                let buffer = shard.gpu.buffer((values.len() * 4) as u64)?;
                shard.gpu.write_buffer(&buffer, 0, &values)?;
                shard.image_bias = Some(buffer);
            }
        }
        Ok(())
    }

    /// Route the image rows of the batch `attending..end` of `layer` again, with the layer's image
    /// bias (the reference's `bias_vl`): on every device, over the router's logits there.
    pub(super) fn route_images(&mut self, layer: usize, attending: usize, end: usize) -> Result<()> {
        let hp = &self.hp;
        let (experts, k) = (hp.experts, hp.experts_used);
        let gating = match hp.gating { Gating::Softmax => 0, Gating::Sigmoid => 1, Gating::SqrtSoftplus => 2 };
        for &(first, last) in &self.image_rows {
            let (from, to) = (first.max(attending), last.min(end));
            if from >= to {
                continue;
            }
            for shard in &mut self.shards {
                let work = shard.work.view(attending);
                let bias = shard.image_bias.ok_or("the image biases are not on the device")?;
                let row = from - attending;
                shard.gpu.route(work.router.at(row * experts), Some(bias.at(layer * experts)), (to - from) as u64, experts as u64,
                                k as u64, gating, hp.route_normalize, hp.route_scale, work.ids.at(row * k), work.weights.at(row * k))?;
            }
        }
        Ok(())
    }

    /// Each image's span rows ([`encode_image_rows`]) on the first device, in its arena.
    pub fn encode_images(&mut self, images: &[Patches]) -> Result<Vec<Vec<f32>>> {
        let width = self.hp.width;
        let shard = &mut self.shards[0];
        let arena = shard.arena.ok_or("this placement has no arena for the vision encoder")?;
        let (experts, reclaimed, region) = (&mut shard.experts, &mut shard.reclaimed, shard.arena_region);
        // (the arena taken back from the expert store as far as the encoder reaches, as a long pass takes it)
        encode_image_rows(&mut shard.gpu, arena, self.tensors, width, images, |gpu, total| {
            if let Some(region) = region.filter(|_| total > *reclaimed) {
                experts.reclaim_region(gpu, region, total)?;
                *reclaimed = total;
            }
            Ok(())
        })
    }
}

/// Each image's span rows ([`crate::vision::span`]): START, the encoder's rows with a NEWLINE after
/// each row of the grid, END; `width` (the model's) values a position, rounded to BF16 as the
/// reference's embeddings are. The encoder runs in `arena`, whose bytes `reserve` makes free for
/// as far as it reaches (given before anything is written there).
pub fn encode_image_rows<T: Tensors + ?Sized>(gpu: &mut Backend, arena: Buffer, tensors: &T, width: usize, images: &[Patches],
                                              reserve: impl FnOnce(&mut Backend, u64) -> Result<()>) -> Result<Vec<Vec<f32>>> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    let config = VisionConfig::DEEPSEEK41;
    let patch = tensors.tensor(&format!("{PATCH}.weight"))?;
    let [values, vision_width] = patch.dims[..] else { return Err("an unexpected vision patch embedding".into()) };
    let layers = (0..).take_while(|&layer| tensors.tensor(&block(layer, "norm1.weight")).is_ok()).count();
    // (the reference's vision_n_heads: the file's tensors do not tell it)
    let heads = 16;
    let ffn = tensors.tensor(&block(0, "mlp.w2.weight"))?.dims[0];
    if values != 3 * config.patch * config.patch || layers == 0 || vision_width % heads != 0 {
        return Err("an unexpected vision encoder shape".into());
    }
    // The weights' places: the products first (one conversion to FP16 covers them), then the vectors.
    let mut names: Vec<(String, bool)> = vec![(format!("{PATCH}.weight"), true)];
    for layer in 0..layers {
        for part in ["attn.wqkv.weight", "attn.wo.weight", "mlp.w1.weight", "mlp.w2.weight"] {
            names.push((block(layer, part), true));
        }
    }
    names.extend([("aligner.w1.weight".to_string(), true), ("aligner.w2.weight".to_string(), true),
                  (format!("{PATCH}.bias"), false), ("vision.norm.weight".to_string(), false),
                  ("aligner.w1.bias".to_string(), false), ("aligner.w2.bias".to_string(), false)]);
    for layer in 0..layers {
        for part in ["norm1.weight", "attn.wqkv.bias", "attn.wo.bias", "norm2.weight"] {
            names.push((block(layer, part), false));
        }
    }
    let mut weights = Vec::with_capacity(names.len());
    let mut at = 0u64;
    for (name, matrix) in names {
        let tensor = tensors.tensor(&name)?;
        if tensor.kind != 30 {
            return Err(format!("{name} is not BF16"));
        }
        let count = tensor.dims.iter().product::<usize>() as u64;
        let bytes = if matrix { count * 2 } else { count * 4 };
        weights.push(Weight { name, matrix, offset: at, bytes });
        at = (at + bytes).next_multiple_of(256);
    }
    let matrices = weights.iter().filter(|weight| weight.matrix).map(|weight| weight.offset + weight.bytes).max().unwrap_or(0);
    let weight_bytes = at;
    let pointers = |base: *mut std::ffi::c_void, name: &str| -> *const std::ffi::c_void {
        let weight = weights.iter().find(|weight| weight.name == name).expect("every encoder weight has a place");
        base.wrapping_byte_add(weight.offset as usize).cast_const()
    };
    let base = arena.base();
    let blocks: Vec<ffi::sd_vision_layer> = (0..layers).map(|layer| ffi::sd_vision_layer {
        norm1: pointers(base, &block(layer, "norm1.weight")),
        qkv: pointers(base, &block(layer, "attn.wqkv.weight")),
        qkv_bias: pointers(base, &block(layer, "attn.wqkv.bias")),
        out: pointers(base, &block(layer, "attn.wo.weight")),
        out_bias: pointers(base, &block(layer, "attn.wo.bias")),
        norm2: pointers(base, &block(layer, "norm2.weight")),
        gate_up: pointers(base, &block(layer, "mlp.w1.weight")),
        down: pointers(base, &block(layer, "mlp.w2.weight")),
    }).collect();
    let align2 = tensors.tensor("aligner.w2.weight")?;
    let model = ffi::sd_vision {
        layers: layers as u64, width: vision_width as u64, heads: heads as u64, ffn: ffn as u64, patch_values: values as u64,
        out_width: align2.dims[1] as u64, downsample: config.downsample as u64, eps: 1e-6, rope_theta: 10_000.0,
        patch: pointers(base, &format!("{PATCH}.weight")), patch_bias: pointers(base, &format!("{PATCH}.bias")),
        norm: pointers(base, "vision.norm.weight"),
        align1: pointers(base, "aligner.w1.weight"), align1_bias: pointers(base, "aligner.w1.bias"),
        align2: pointers(base, "aligner.w2.weight"), align2_bias: pointers(base, "aligner.w2.bias"),
        layer: blocks.as_ptr(),
    };
    if model.out_width as usize != width {
        return Err("the vision aligner's width is not the model's".into());
    }
    let mut scratch = 0u64;
    for image in images {
        let mut bytes = 0u64;
        // SAFETY: the model describes `layers` blocks; nothing is read through its pointers.
        straydog_gpu::check(unsafe { ffi::sd_vision_scratch_bytes(&model, image.rows as u64, image.columns as u64, &mut bytes) })
            .map_err(|error| error.to_string())?;
        scratch = scratch.max(bytes);
    }
    let most_patches = images.iter().map(|image| image.values.len()).max().unwrap_or(0) as u64;
    let most_rows = images.iter().map(|image| image.plan.rows * image.plan.columns).max().unwrap_or(0) as u64;
    let patches_at = weight_bytes;
    let out_at = (patches_at + most_patches * 2).next_multiple_of(256);
    let scratch_at = (out_at + most_rows * width as u64 * 4).next_multiple_of(256);
    let total = scratch_at + scratch;
    if total > arena.bytes {
        return Err(format!("the vision encoder needs {} MB of the arena's {} MB", total >> 20, arena.bytes >> 20));
    }
    reserve(gpu, total)?;
    for weight in &weights {
        let tensor = tensors.tensor(&weight.name)?;
        if weight.matrix {
            gpu.write_buffer(&arena, weight.offset, tensor.data)?;
        } else {
            let values: Vec<f32> = tensor.data.chunks_exact(2)
                .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)).collect();
            gpu.write_buffer(&arena, weight.offset, &values)?;
        }
    }
    // SAFETY: the products' BF16 values lie in the arena's first `matrices` bytes.
    straydog_gpu::check(unsafe { ffi::sd_vision_half(gpu.handle(), base, matrices / 2) }).map_err(|error| error.to_string())?;
    let start = tensors.tensor("image_start")?;
    let newline = tensors.tensor("image_newline")?;
    let end = tensors.tensor("image_end")?;
    let learned = |tensor: &crate::arch::Tensor<'_>| -> Result<Vec<f32>> {
        if tensor.kind != 30 || tensor.dims != [width] {
            return Err("an unexpected image span embedding".into());
        }
        Ok(tensor.data.chunks_exact(2).map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)).collect())
    };
    let (start, newline, end) = (learned(&start)?, learned(&newline)?, learned(&end)?);
    let mut spans = Vec::with_capacity(images.len());
    for image in images {
        let halves: Vec<u16> = image.values.iter().map(|&bits| half(f32::from_bits(u32::from(bits) << 16))).collect();
        gpu.write_buffer(&arena, patches_at, &halves)?;
        // SAFETY: patches, scratch and rows lie in the arena (checked against its size above), the
        // weights' addresses too; the library checks each range against its allocations.
        straydog_gpu::check(unsafe {
            ffi::sd_vision_encode(gpu.handle(), &model, base.wrapping_byte_add(patches_at as usize).cast_const(), image.rows as u64,
                                  image.columns as u64, base.wrapping_byte_add(scratch_at as usize), scratch,
                                  base.wrapping_byte_add(out_at as usize).cast())
        }).map_err(|error| error.to_string())?;
        let count = image.plan.rows * image.plan.columns;
        let mut rows = vec![0f32; count * width];
        gpu.read_buffer(&arena, (out_at / 4) as usize, &mut rows)?;
        let mut values = Vec::with_capacity(image.plan.tokens() * width);
        for slot in span(image.plan.rows, image.plan.columns) {
            match slot {
                Slot::Start => values.extend_from_slice(&start),
                Slot::NewLine => values.extend_from_slice(&newline),
                Slot::End => values.extend_from_slice(&end),
                Slot::Image(index) => values.extend(rows[index * width..(index + 1) * width].iter()
                    .map(|&value| f32::from_bits(u32::from(bf16(value)) << 16))),
            }
        }
        spans.push(values);
    }
    Ok(spans)
}

/// F32 to FP16 bits, rounded to nearest even (normal values and zero; the patches' values are
/// BF16 in [-1, 1], which FP16 holds exactly).
fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if value == 0.0 || exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    let mantissa = bits & 0x7f_ffff;
    let rounded = (mantissa + 0xfff + ((mantissa >> 13) & 1)) >> 13;
    sign | ((((exponent as u32) << 10) + rounded) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_runs_are_found() {
        let i = IMAGE_TOKEN;
        assert_eq!(runs(&[1, 2, 3]), []);
        assert_eq!(runs(&[1, i, i + 1, 2, i + 9]), [(1, 3), (4, 5)]);
        assert_eq!(runs(&[i, i]), [(0, 2)]);
    }

    #[test]
    fn half_holds_patch_values() {
        assert_eq!(half(1.0), 0x3c00);
        assert_eq!(half(-1.0), 0xbc00);
        assert_eq!(half(0.0), 0);
        assert_eq!(half(0.5), 0x3800);
        // 2/255 - 1 as BF16: -0.9921875
        assert_eq!(f32::from_bits(u32::from(bf16(2.0 / 255.0 - 1.0)) << 16), -0.9921875);
        assert_eq!(half(-0.9921875), 0xbbf0);
    }
}
