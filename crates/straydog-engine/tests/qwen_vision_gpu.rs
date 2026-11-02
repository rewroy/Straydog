//! Qwen3.8's rope positions on real GPUs: the table a batch of a sequence with images reads, and the delta its
//! decode rows read, against each other and against the text path.
//!
//! cargo test -p straydog-engine --features hip --test qwen_vision_gpu -- --test-threads=1, with STRAYDOG_MODEL (a bundle
//! directory), STRAYDOG_QUANT (its quantization) and optionally STRAYDOG_VRAM_GIB; under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::qwen_vision::{ImageSpan, ROW, VISION_END, VISION_START};
use straydog_engine::sampling::Sampling;
use straydog_engine::vision::IMAGE_TOKEN;
use straydog_engine::weights::Weights;

fn open() -> Option<Model> {
    let root = common::model_env("STRAYDOG_MODEL", "the model bundle directory")?;
    let quant = common::model_env("STRAYDOG_QUANT", "the bundle's quantization")?;
    let gib: f64 = std::env::var("STRAYDOG_VRAM_GIB").ok().and_then(|value| value.parse().ok()).unwrap_or(8.5);
    let mut group = Group::new(&[0], &[(gib * (1u64 << 30) as f64) as u64], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let weights = Weights::open(&PathBuf::from(root), quant.to_str().unwrap(), None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let settings = Settings { context: 32768, state_budget_bytes: 14 << 30, prefill_batch_size: 1024, speculative: 0, cpu_threads: 0, row_cache_bytes: None };
    Some(Model::new(weights, group, &settings, &|_, _, _| {}).unwrap())
}

/// A greedy reply from nothing held, with its logits.
fn reply(model: &mut Model, prompt: &[u32], count: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    model.reset().unwrap();
    model.logits_capture = Some(Vec::new());
    let mut tokens = Vec::new();
    model.generate(prompt, count, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |token| tokens.push(token)).unwrap();
    (tokens, model.logits_capture.take().unwrap())
}

/// Token ids of ordinary text.
fn text(count: usize, seed: u32) -> Vec<u32> {
    (0..count as u32).map(|index| 1000 + (index * 37 + seed) % 40_000).collect()
}

fn close(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

#[test]
fn rope_positions_agree_on_every_path() {
    let Some(mut model) = open() else { return };
    let prompt = text(300, 5);
    // (CUDA's first reply after a start is not deterministic: from the second)
    reply(&mut model, &prompt, 4);

    // Text through a table of its own positions: the same kernels' arithmetic, bit for bit (the empty span turns the table on).
    let (plain, plain_logits) = reply(&mut model, &prompt, 4);
    model.set_images(vec![ImageSpan { start: 0, grid: (0, 0), rows: Arc::new(Vec::new()) }]);
    let (table, table_logits) = reply(&mut model, &prompt, 4);
    assert_eq!(table, plain);
    assert_eq!(table_logits[0], plain_logits[0], "the prompt's last row through the table");
    model.set_images(Vec::new());

    // A sequence with an image: the reply's decode rows (the delta) are what the same tokens in a batch (the table) are.
    let grid = (8, 8);
    let rows: Vec<f32> = (0..64 * ROW).map(|index| ((index * 2_654_435_761usize) % 1000) as f32 / 10_000.0 - 0.05).collect();
    let mut prompt = text(40, 9);
    prompt.push(VISION_START);
    let start = prompt.len();
    prompt.extend((0..64u32).map(|offset| IMAGE_TOKEN | (12345 + offset)));
    prompt.push(VISION_END);
    prompt.extend(text(40, 11));
    let span = ImageSpan { start, grid, rows: Arc::new(rows) };
    model.set_images(vec![span.clone()]);
    let (tokens, logits) = reply(&mut model, &prompt, 12);
    assert_eq!(tokens.len(), 12);
    for kept in [1, 4, 7, 11] {
        let longer: Vec<u32> = prompt.iter().chain(&tokens[..kept]).copied().collect();
        let (next, next_logits) = reply(&mut model, &longer, 1);
        assert_eq!(next[0], tokens[kept], "after {kept} tokens read in a batch");
        assert!(close(&next_logits[0], &logits[kept]) < 0.05, "after {kept}: logits apart by {}", close(&next_logits[0], &logits[kept]));
    }
    // And the same text past the image is not what it is without its positions' delta: the rope position does something here.
    let without_delta: Vec<f32> = {
        let text_only: Vec<u32> = prompt.iter().map(|token| if *token >= IMAGE_TOKEN { 2000 } else { *token }).collect();
        model.set_images(Vec::new());
        reply(&mut model, &text_only, 1).1.remove(0)
    };
    assert!(close(&without_delta, &logits[0]) > 0.0);
}
