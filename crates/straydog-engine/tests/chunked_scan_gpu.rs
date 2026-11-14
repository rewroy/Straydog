//! A read whose DeltaNet layers scan in chunks of tokens (gpu/gated_delta_chunk.hpp) against the same read scanned token by token, on a real
//! prompt of a few thousand tokens (the rule engages it from model::CHUNKED_SCAN_READ new tokens; the test forces it). The sums run in
//! another order (some products on the matrix cores in bf16 pieces), and the model amplifies any such change: a router or a block selection
//! that flips on a near tie moves the first logits by tenths, so the yardstick is what another batch width does to the token-by-token scan,
//! an order of F32 sums the engine already has. A read the rule leaves to the token-by-token scan is that scan's, bit for bit.
//! A CUDA build's batches are the same bits at every width, so there the floor of 1e-3 is the rule, and it is held under the F32 attention
//! kernel (STRAYDOG_QSA_MMA=0), where the scans' difference is the scans' own. Under the tensor-core kernel, the default, the same
//! difference may be four times that: the kernel rounds its operands to FP16, and two reads a few F32 bits apart leave it FP16 steps
//! apart (8.6e-4 under the F32 kernel and 1.7e-3 under the tensor cores on an RTX 5070 Ti, where the two kernels themselves are 2.8e-5
//! apart under the token-by-token scan).
//!
//! cargo test --release -p straydog-engine --features hip --test chunked_scan_gpu, under the GPU lock, with STRAYDOG_MODEL (the Qwen3.8
//! bundle directory), STRAYDOG_DRAFTER, and optionally STRAYDOG_CHUNKED_QUANT (UD-IQ3_XXS), STRAYDOG_CHUNKED_DEVICES (0,1) and
//! STRAYDOG_CHUNKED_VRAM (GiB a device, 30).
#![cfg(any(feature = "hip", feature = "cuda"))]
mod common;

use std::path::Path;
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings, CHUNKED_SCAN_READ};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

fn open(root: &Path, drafter: &Path) -> (Model, Text) {
    let devices: Vec<usize> = std::env::var("STRAYDOG_CHUNKED_DEVICES").unwrap_or_else(|_| "0,1".into()).split(',').map(|d| d.parse().unwrap()).collect();
    let gib: f64 = std::env::var("STRAYDOG_CHUNKED_VRAM").map_or(30.0, |v| v.parse().unwrap());
    let budget = (gib * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&devices, &vec![budget; devices.len()], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let quant = std::env::var("STRAYDOG_CHUNKED_QUANT").unwrap_or_else(|_| "UD-IQ3_XXS".into());
    let id = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, &quant).unwrap();
    let weights = Weights::open(root, id, Some(drafter), straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let text = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap();
    let settings = Settings { context: 32768, state_budget_bytes: 14 << 30, prefill_batch_size: 2048, speculative: 3, cpu_threads: 0, row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    model.prompt_share(false).unwrap();
    (model, text)
}

/// The first logits of a read, and the model's state digests after it.
fn read(model: &mut Model, tokens: &[u32]) -> (Vec<f32>, Vec<String>) {
    model.reset().unwrap();
    model.logits_capture = Some(Vec::new());
    model.generate(tokens, 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |_| {}).unwrap();
    let logits = model.logits_capture.take().unwrap().remove(0);
    (logits, model.state_digest().unwrap())
}

/// Sets or removes a variable where the engine's C++ reads it. On Windows std::env::set_var changes the process's environment block, which
/// the C runtime's getenv (its own copy, made at the start) does not see; the runtime's _putenv_s changes both.
fn set_engine_variable(name: &str, value: Option<&str>) {
    #[cfg(windows)]
    {
        extern "C" { fn _putenv_s(name: *const std::ffi::c_char, value: *const std::ffi::c_char) -> i32; }
        let name = std::ffi::CString::new(name).unwrap();
        let value = std::ffi::CString::new(value.unwrap_or("")).unwrap();   // (an empty value removes the variable)
        assert_eq!(unsafe { _putenv_s(name.as_ptr(), value.as_ptr()) }, 0);
    }
    #[cfg(not(windows))]
    match value { Some(value) => std::env::set_var(name, value), None => std::env::remove_var(name) }
}

fn kl(a: &[f32], b: &[f32]) -> f64 {
    let log_softmax = |x: &[f32]| {
        let top = x.iter().copied().map(f64::from).fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = x.iter().map(|v| (f64::from(*v) - top).exp()).sum();
        x.iter().map(|v| f64::from(*v) - top - sum.ln()).collect::<Vec<_>>()
    };
    let (p, q) = (log_softmax(a), log_softmax(b));
    p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
}

#[test]
fn the_chunked_scan_keeps_a_read_to_f32_rounding_and_a_short_read_to_its_bits() {
    let (Some(root), Some(drafter)) = (common::model_env("STRAYDOG_MODEL", "the Qwen model bundle directory"),
                                     common::model_env("STRAYDOG_DRAFTER", "the MTP bundle directory")) else { return };
    let (mut model, text) = open(Path::new(&root), Path::new(&drafter));
    let line = "A printed source lets readers compare the same words, check the evidence, and correct errors in the next edition. ";
    let mut prompt = text.encode(&line.repeat(400)).unwrap();
    prompt.truncate(3001);
    assert!(prompt.len() == 3001 && prompt.len() < CHUNKED_SCAN_READ, "a read the rule leaves alone");
    // (the default rule: the token-by-token scan)
    let by_default = read(&mut model, &prompt);
    model.chunked_scan_from(usize::MAX);
    let by_token = read(&mut model, &prompt);
    model.prefill_batch_size = 1024;
    let by_width = read(&mut model, &prompt);
    model.prefill_batch_size = 2048;
    model.chunked_scan_from(0);
    let in_chunks = read(&mut model, &prompt);
    assert!(by_default.0.iter().zip(&by_token.0).all(|(x, y)| x.to_bits() == y.to_bits()) && by_default.1 == by_token.1,
            "a read shorter than the rule changed");
    assert!(in_chunks.0.iter().all(|x| x.is_finite()));
    let divergence = kl(&by_token.0, &in_chunks.0);
    let width = kl(&by_token.0, &by_width.0);
    let top = |x: &[f32]| x.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let worst = by_token.0.iter().zip(&in_chunks.0).map(|(x, y)| f64::from((x - y).abs())).fold(0.0, f64::max);
    eprintln!("chunked against token-by-token scan, {} tokens: KL {divergence:e}, largest logit difference {worst:e}, same top token {}; \
               the token-by-token scan at batch 1024 against 2048: KL {width:e}", prompt.len(), top(&by_token.0) == top(&in_chunks.0));
    assert!(by_token.0 != in_chunks.0 || cfg!(not(feature = "hip")), "the chunked scan did not run");
    assert!(top(&by_token.0) == top(&in_chunks.0), "the chunked scan changed the first token");
    let allowed = (30.0 * width).max(1e-3);
    // (CUDA, the switch not named: both scans once more under the F32 attention kernel, in a model of their own: the switch is read once
    // a device)
    if cfg!(feature = "cuda") && std::env::var_os("STRAYDOG_QSA_MMA").is_none() {
        drop(model);
        set_engine_variable("STRAYDOG_QSA_MMA", Some("0"));
        let (mut other, _) = open(Path::new(&root), Path::new(&drafter));
        other.chunked_scan_from(usize::MAX);
        let token_f32 = read(&mut other, &prompt);
        other.chunked_scan_from(0);
        let chunks_f32 = read(&mut other, &prompt);
        drop(other);
        set_engine_variable("STRAYDOG_QSA_MMA", None);
        let (under_f32, form) = (kl(&token_f32.0, &chunks_f32.0), kl(&by_token.0, &token_f32.0));
        eprintln!("under the F32 attention kernel: chunked against token-by-token KL {under_f32:e}, same top token {}; the token-by-token scan \
                   under the two attention kernels: KL {form:e}", top(&token_f32.0) == top(&chunks_f32.0));
        assert!(top(&token_f32.0) == top(&chunks_f32.0), "the chunked scan changed the first token under the F32 attention kernel");
        assert!(under_f32 <= allowed, "under the F32 attention kernel: KL {under_f32:e} against {width:e} for another batch width");
        assert!(divergence <= allowed.max(4.0 * under_f32), "KL {divergence:e} against {under_f32:e} under the F32 attention kernel");
    } else {
        assert!(divergence <= allowed, "KL {divergence:e} against {width:e} for another batch width");
    }
}
