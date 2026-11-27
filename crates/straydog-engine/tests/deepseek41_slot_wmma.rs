//! DeepSeek-V4.1's routed experts of an FP16 prompt pass through the grouped WMMA kernels (RDNA4,
//! sd_op_slot_moe_*, decoding each slot's quantized weights straight into WMMA fragments) are as
//! accurate as the expanded GEMM per expert (DS41_SLOT_WMMA=0), both measured against F32
//! (--gemm fp32): over a prompt of several waves of multi-row experts on two devices, the first
//! layers' sequence state (where FP16's differences have not compounded yet) is as near F32's,
//! the next-token distribution about as near, and the next token F32's or one it all but ties with. A lost or wrong
//! expert contribution moves the first layers' state some 60 times further than FP16 does (a
//! scatter whose adds to one row raced did).
//!
//! A layer's distance is compared by each path's worst layer, not layer by layer: since the compressed rows and
//! index keys are BF16 (85b1882) a layer's state moves in steps. One rounding that goes the other way, or one
//! index pick that flips, in one path adds ~1.3e-3 to a layer's distance and none to the other's, and which of
//! the two paths it lands on, and in which layer, is chance: expanded 2.5e-3 at layer 2 and grouped 1.1e-3, grouped
//! 2.5e-3 at layer 3 and expanded 1.3e-3 (before BF16: layer 3 expanded 1.2e-3, grouped 0.8e-3).
//!
//! So the bounds hold for three prompts together, each path's worst layers and KLs summed: one prompt's steps decide too much. The
//! prompts are fixed (tests/data/ds41_slot_wmma: 6,000 characters of docs/DESIGN.md as it was at three places on 2026-11-27); they were
//! its first 6,000 characters as the file stood, and a change of the document at character 2,701 made the test fail on the same code.
//! Over six such prompts of that file the worst layers' ratio (grouped to expanded) was 1.91, 1.00, 1.04, 1.67, 0.91 and 1.00, and the
//! KLs' 0.53, 1.09, 1.39, 2.60, 1.15 and 1.23; these three hold the two that failed alone (1.91 and 1.67): together 1.38 and 1.44.
//!
//! About 14 minutes (nine placements of the model). cargo test -p straydog-engine --features hip --test deepseek41_slot_wmma, with STRAYDOG_DS41 (the
//! GSQ-RCO IQ2_XS bundle directory (STRAYDOG_DS41_QUANT=GSQ-RCO-IQ2_XXS for the IQ2_XXS one), its verity trees in .straydog there); under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use straydog_engine::arch::deepseek41::serve::place;
use straydog_engine::arch::served::Settings;
use straydog_engine::group::SplitMode;
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;

/// ||a - b|| / ||a||.
fn relative(a: &[f32], b: &[f32]) -> f64 {
    let (mut difference, mut norm) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        difference += ((x - y) as f64).powi(2);
        norm += (x as f64).powi(2);
    }
    (difference / norm.max(f64::MIN_POSITIVE)).sqrt()
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&value| (value as f64 - top).exp()).sum();
    logits.iter().map(|&value| value as f64 - top - sum.ln()).collect()
}

/// KL(p || q) of two rows of logits.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (p, q) = (log_softmax(p), log_softmax(q));
    p.iter().zip(&q).map(|(&a, &b)| a.exp() * (a - b)).sum()
}

fn best(logits: &[f32]) -> usize {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

#[test]
fn an_fp16_pass_through_the_grouped_kernels_is_as_accurate_as_the_expanded_gemms() {
    let Some(root) = common::model_env("STRAYDOG_DS41", "the DeepSeek-V4.1 GSQ-RCO IQ2_XS bundle directory") else { return };
    let root = PathBuf::from(root);
    let documents = [include_str!("data/ds41_slot_wmma/a.txt"), include_str!("data/ds41_slot_wmma/b.txt"), include_str!("data/ds41_slot_wmma/c.txt")];
    // A run's prompt logits and its first layers' window rings (device 0's), in F32 or FP16, through
    // the grouped kernels or the expanded GEMMs.
    let run = |document: &str, half: bool, grouped: bool| -> (Vec<f32>, Vec<Vec<f32>>) {
        // (read when the engines are made, on this thread alone)
        std::env::set_var("DS41_SLOT_WMMA", if grouped { "1" } else { "0" });
        let settings = Settings {
            devices: vec![0, 1], split_mode: SplitMode::Experts, vram_bytes: vec![28 << 30; 2], context: 16384, batch: 2048,
            // (a small host tier: the test places the model three times in one process)
            host_expert_bytes: Some(12 << 30), host_memory_bytes: straydog_engine::host::host_memory_bytes(),
            row_cache_bytes: straydog_engine::rows::DEFAULT_CACHE_BYTES, half_gemm: half,
        };
        let mut model = place(&root, common::deepseek41_bundle(), &root.join(".straydog"), &settings, &|_, _, _| {}).unwrap();
        model.prompt_logits = Some(Vec::new());
        let prompt = Text::from_gguf(model.header()).unwrap().encode(&format!("{document}\n\nWhat is this project?")).unwrap();
        assert!(prompt.len() > 1000, "a prompt of several waves");
        model.generate(&prompt, 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, &mut |_| {}).unwrap();
        let state = model.engines.state().unwrap();
        let rings = (1..=4).map(|layer| {
            let name = format!("engine 0 device 0 layer {layer} window");
            state.iter().find(|(part, _)| *part == name).unwrap_or_else(|| panic!("no {name}")).1.clone()
        }).collect();
        (model.prompt_logits.clone().unwrap(), rings)
    };
    let (mut worst_expanded, mut worst_grouped, mut from_expanded, mut from_grouped) = (0f64, 0f64, 0f64, 0f64);
    for (prompt, document) in documents.iter().enumerate() {
        let (exact, exact_rings) = run(document, false, false);
        let (expanded, expanded_rings) = run(document, true, false);
        let (grouped, grouped_rings) = run(document, true, true);
        let differ = expanded.iter().zip(&grouped).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert!(differ > 0, "prompt {prompt}: the grouped kernels did not run (the logits are bit for bit the expanded GEMMs')");
        let (mut worst, mut worst_of_grouped) = (0f64, 0f64);
        for (layer, ((exact, expanded), grouped)) in exact_rings.iter().zip(&expanded_rings).zip(&grouped_rings).enumerate() {
            let (expanded, grouped) = (relative(exact, expanded), relative(exact, grouped));
            eprintln!("prompt {prompt} layer {} window: expanded {expanded:.3e}, grouped {grouped:.3e} from F32", layer + 1);
            (worst, worst_of_grouped) = (worst.max(expanded), worst_of_grouped.max(grouped));
        }
        let (kl_expanded, kl_grouped) = (kl(&exact, &expanded), kl(&exact, &grouped));
        eprintln!("prompt {prompt}: worst layer expanded {worst:.3e}, grouped {worst_of_grouped:.3e}; KL from F32 expanded {kl_expanded:.3e}, \
                   grouped {kl_grouped:.3e}");
        (worst_expanded, worst_grouped, from_expanded, from_grouped) =
            (worst_expanded + worst, worst_grouped + worst_of_grouped, from_expanded + kl_expanded, from_grouped + kl_grouped);
        // The next token: F32's, or one F32 itself all but ties with it (a near tie two FP16 paths
        // may break either way: on these prompts two tokens may lie within FP16's differences).
        let (picked, ideal) = (best(&grouped), best(&exact));
        eprintln!("prompt {prompt} next token: F32 {ideal}, expanded {}, grouped {picked} ({:.3} below F32's best under F32)", best(&expanded),
                  exact[ideal] - exact[picked]);
        assert!(exact[picked] >= exact[ideal] - 0.25, "prompt {prompt}, the next token: grouped picks {picked}, {:.3} below F32's {ideal}",
                exact[ideal] - exact[picked]);
    }
    assert!(worst_grouped <= 1.5 * worst_expanded + 1e-5, "the grouped kernels' worst layers sum to {worst_grouped:.3e} from F32, the \
                                                           expanded GEMMs' {worst_expanded:.3e}");
    eprintln!("KL from F32 over the prompts: expanded {from_expanded:.3e}, grouped {from_grouped:.3e}");
    assert!(from_grouped <= 2.0 * from_expanded + 1e-3, "KL from F32: grouped {from_grouped:.3e}, expanded {from_expanded:.3e}");
}
