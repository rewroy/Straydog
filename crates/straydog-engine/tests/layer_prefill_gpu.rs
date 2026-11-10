//! Layer-ordered prefill: the same logits and causal state as the batch order, including checkpoints, MTP and cancellation.
//! Run under the GPU lock with STRAYDOG_LAYER_MODEL, STRAYDOG_DRAFTER and STRAYDOG_REQUIRE_MODEL=1.
#![cfg(any(feature = "hip", feature = "cuda"))]
mod common;

use std::cell::Cell;
use std::path::Path;
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

struct Observed { logits: Vec<f32>, token: Option<u32>, state: Vec<String>, batch: u64, order: serde_json::Value }

fn open(root: &Path, drafter: &Path, layer: bool) -> (Model, Text) {
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let quant = std::env::var("STRAYDOG_LAYER_QUANT").unwrap_or_else(|_| "UD-IQ2_XS".into());
    let id = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, &quant).unwrap();
    let weights = Weights::open(root, id, Some(drafter), straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let text = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap();
    let settings = Settings { context: 262144, state_budget_bytes: 14 << 30, prefill_batch_size: 256,
                              speculative: 3, cpu_threads: 0, row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial && model.host_kv && model.speculative > 0);
    // Keep both arithmetic paths on the GPU during the prompt; decode's ordinary cache still gets restored.
    model.prompt_share(false).unwrap();
    model.layer_prefill(layer);
    (model, text)
}

fn observed(model: &mut Model, tokens: &[u32], keep: bool) -> Observed {
    model.logits_capture = Some(Vec::new());
    let mut emitted = Vec::new();
    model.generate(tokens, 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, keep,
                   &mut |t| emitted.push(t)).unwrap();
    let logits = model.logits_capture.take().unwrap().remove(0);
    Observed { logits, token: emitted.first().copied(), state: model.state_digest().unwrap(),
               batch: model.last_metrics["prefill_batch_rows"].as_u64().unwrap_or(0), order: model.last_metrics["prefill_order"].clone() }
}

fn compare(name: &str, a: &Observed, b: &Observed) -> bool {
    assert_eq!(a.logits.len(), b.logits.len());
    let max_a = a.logits.iter().copied().map(f64::from).fold(f64::NEG_INFINITY, f64::max);
    let max_b = b.logits.iter().copied().map(f64::from).fold(f64::NEG_INFINITY, f64::max);
    let za: f64 = a.logits.iter().map(|x| (f64::from(*x) - max_a).exp()).sum();
    let zb: f64 = b.logits.iter().map(|x| (f64::from(*x) - max_b).exp()).sum();
    let mut max_abs = 0f64;
    let (mut diff, mut norm, mut kl) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.logits.iter().zip(&b.logits) {
        assert!(x.is_finite() && y.is_finite());
        let (x, y) = (f64::from(x), f64::from(y));
        max_abs = max_abs.max((x - y).abs()); diff += (x - y).powi(2); norm += x * x;
        let logp = x - max_a - za.ln();
        let logq = y - max_b - zb.ln();
        kl += logp.exp() * (logp - logq);
    }
    let bitwise = a.logits.iter().zip(&b.logits).all(|(x, y)| x.to_bits() == y.to_bits());
    eprintln!("{}", serde_json::json!({"case":name,"bitwise":bitwise,"max_abs":max_abs,
        "relative_l2":(diff/norm.max(1e-30)).sqrt(),"kl":kl,"same_token":a.token==b.token,"same_state":a.state==b.state,
        "batch_a":a.batch,"batch_b":b.batch,"order_a":a.order,"order_b":b.order,
        "state_mismatches":a.state.iter().zip(&b.state).filter(|(x,y)| x!=y).take(16).collect::<Vec<_>>()}));
    a.token == b.token && bitwise && a.state == b.state
}

#[test]
fn layer_order_preserves_logits_checkpoints_and_cancelled_prefixes() {
    let (Some(root), Some(drafter)) = (common::model_env("STRAYDOG_LAYER_MODEL", "the Qwen model bundle directory"),
                                     common::model_env("STRAYDOG_DRAFTER", "the MTP bundle directory")) else { return };
    let mut runs = Vec::new();
    for layer in [false, true] {
        let (mut model, text) = open(Path::new(&root), Path::new(&drafter), layer);
        let line = "A printed source lets readers compare the same words, check the evidence, and correct errors in the next edition. ";
        let body = line.repeat(140);
        let mut plain = text.encode(&body).unwrap();
        plain.truncate(1025); // not a multiple of either temporary batch size
        assert_eq!(plain.len(), 1025);
        let mut cases = vec![observed(&mut model, &plain, false)];
        if layer { assert_eq!(model.last_metrics["prefill_order"], "layer", "the candidate fell back"); }
        model.reset().unwrap();
        let system = format!("<|im_start|>system\n{}<|im_end|>\n", line.repeat(170));
        assert!(text.encode(&system).unwrap().len() >= 2048);
        let first = format!("{system}<|im_start|>user\n{} What is the capital of France?<|im_end|>\n<|im_start|>assistant\n", line.repeat(40));
        let follow = format!("{first}Paris.<|im_end|>\n<|im_start|>user\nAnd of Japan?<|im_end|>\n<|im_start|>assistant\n");
        cases.push(observed(&mut model, &text.encode(&first).unwrap(), true));
        cases.push(observed(&mut model, &text.encode(&follow).unwrap(), true));
        assert!(model.last_metrics["cached_tokens"].as_u64().unwrap() > 2048, "the follow-up did not restore a checkpoint");
        model.reset().unwrap();
        let cancelled = Cell::new(false);
        model.generate(&plain, 1, &Sampling::GREEDY, 0, &|| cancelled.get(),
                       &mut |_, _| cancelled.set(true), true, false, &mut |_| {}).unwrap();
        assert!(cancelled.get() && !model.history.is_empty());
        // resume deliberately leaves at least one token to run: continue the cancelled prefix, rather than repeat it whole.
        let continued = [plain.clone(), plain[..512].to_vec()].concat();
        cases.push(observed(&mut model, &continued, false));
        assert_eq!(model.last_metrics["cached_tokens"].as_u64().unwrap(), plain.len() as u64);
        runs.push(cases);
    }
    if let Some(path) = std::env::var_os("STRAYDOG_LAYER_REPORT") {
        let report: Vec<Vec<_>> = runs.iter().map(|cases| cases.iter().map(|case| serde_json::json!({
            "logits":case.logits,"token":case.token,"state":case.state,"batch":case.batch,"order":case.order,
        })).collect()).collect();
        serde_json::to_writer(std::fs::File::create(path).unwrap(), &report).unwrap();
    }
    let mut exact = true;
    for (i, name) in ["odd-sized prompt", "root and turn checkpoint", "checkpoint continuation", "cancelled prefix continuation"].iter().enumerate() {
        exact &= compare(name, &runs[0][i], &runs[1][i]);
    }
    assert!(exact, "inspect all reported logit and state differences before accepting");
}
