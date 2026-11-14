//! A partially resident model's K/V in VRAM starting small and growing under a live sequence (`Model::ensure_kv`, `sd_kv_grow` on a device K/V): what it
//! held stays, and what it takes comes from the expert slots for good.
//!
//! cargo test --release -p straydog-engine --features hip --test kv_growth_vram_gpu, with STRAYDOG_MODEL (a Qwen3.8 bundle with experts beyond one GPU's
//! budget), STRAYDOG_QUANT (its quantization; the golden's where unset), STRAYDOG_VRAM_GIB (the budget; 24, which holds a 262144 context's K/V of twelve
//! layers in VRAM, where unset; 26 with STRAYDOG_DRAFTER, thirteen layers) and, for the second test, STRAYDOG_DRAFTER; one test at a time. KV_GROWTH_OUT
//! writes each request's logits and state digests: with STRAYDOG_KV_START=0 (the whole context from the start, no growth) the file is the same.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings, STARTING_KV};
use straydog_engine::weights::Weights;

fn quantization() -> String {
    std::env::var("STRAYDOG_QUANT").unwrap_or_else(|_| {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/model.json");
        let golden: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        golden["quantization"].as_str().unwrap().to_string()
    })
}

fn open(root: &std::ffi::OsStr, drafter: Option<&std::ffi::OsStr>) -> Model {
    let gib: f64 = std::env::var("STRAYDOG_VRAM_GIB").ok().and_then(|gib| gib.parse().ok()).unwrap_or(if drafter.is_some() { 26.0 } else { 24.0 });
    let budget = (gib * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let id = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, &quantization()).unwrap();
    let weights = Weights::open(&PathBuf::from(root), id, drafter.map(PathBuf::from).as_deref(), straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let settings = Settings { context: 262144, state_budget_bytes: 14 << 30, prefill_batch_size: 4096, speculative: if drafter.is_some() { 3 } else { 0 },
                              cpu_threads: 0, row_cache_bytes: None };
    Model::new(weights, group, &settings, &|_, _, _| {}).unwrap()
}

struct Seen { first: u32, logits: Vec<String>, state: Vec<String>, resumed: Json }

fn go(model: &mut Model, prompt: &[u32], max_new: usize) -> Seen {
    model.logits_digests = Some(Vec::new());
    let mut tokens = Vec::new();
    model.generate(prompt, max_new, &straydog_engine::sampling::Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true,
                   &mut |token| tokens.push(token)).unwrap();
    Seen { first: tokens[0], logits: model.logits_digests.take().unwrap(), state: model.state_digest().unwrap(),
           resumed: model.last_metrics.get("resumed_from").cloned().unwrap_or(Json::Null) }
}

/// The positional parts both digests have, which must be equal.
fn kept(before: &[String], after: &[String]) -> usize {
    let label = |line: &str| line.rsplit_once(' ').unwrap().0.to_string();
    let positional = |line: &&String| [" rows ", " blocks ", " row ", " block "].iter().any(|part| line.contains(part));
    let later: std::collections::HashMap<String, String> = after.iter().map(|line| (label(line), line.clone())).collect();
    let mut same = 0;
    for line in before.iter().filter(positional) {
        // (the last rows and blocks one by one are only the later digest's where it ended)
        let Some(after) = later.get(&label(line)) else { continue };
        assert_eq!(after, line, "changed by the growth");
        same += 1;
    }
    same
}

fn text(length: usize) -> Vec<u32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/model.json");
    let golden: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let requests = golden["placements"]["two-gpu-experts-mtp"]["requests"].as_array().unwrap();
    let mut text: Vec<u32> = Vec::new();
    while text.len() < length {
        for request in requests {
            text.extend(serde_json::from_value::<Vec<u32>>(request["prompt"].clone()).unwrap());
        }
        text.push(text.len() as u32 % 200000);
    }
    text
}

fn record(out: &mut String, name: &str, seen: &Seen) {
    let state = Sha256::digest(seen.state.join("\n").as_bytes());
    let line = format!("{name} first {} logits {:?} state {state:x} resumed {}\n", seen.first, seen.logits, seen.resumed);
    eprint!("{line}");
    out.push_str(&line);
}

/// A K/V in VRAM that starts at 32768 positions and doubles: the slots at the start are more than at the whole context's, they give up the VRAM of each
/// growth for good (fewer after it than before), the rows and indexer blocks the sequence held are the same bytes after it, and a departure and a longer
/// prompt (with a parked sequence, which a growth drops) go on from there.
#[test]
fn a_vram_kv_grows_and_the_slots_pay_for_it() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the model bundle directory") else { return };
    let text = text(140_100);
    let mut out = String::new();
    let mut model = open(&root, None);
    assert!(model.partial && !model.host_kv, "partial {} host K/V {}", model.partial, model.host_kv);
    let whole = std::env::var("STRAYDOG_KV_START").is_ok_and(|start| start == "0");
    let start = if whole { 262144 } else { STARTING_KV };
    assert_eq!(model.kv_positions(), start, "the K/V starts at its first capacity");
    let slots_at_start = model.slot_counts().0;
    assert!(slots_at_start > 0);
    // Below the first capacity, then past it in a prefill.
    let before = go(&mut model, &text[..32000], 1);
    record(&mut out, "below", &before);
    assert_eq!(model.kv_positions(), start);
    let grown = go(&mut model, &text[..40000], 1);
    record(&mut out, "grown", &grown);
    eprintln!("{} parts kept past 32768", kept(&before.state, &grown.state));
    if !whole {
        assert_eq!(model.kv_positions(), 65536);
        let slots = model.slot_counts().0;
        assert!(slots < slots_at_start, "the growth took VRAM from the slots: {slots_at_start} -> {slots}");
    }
    // A departure resuming below the growth (a checkpoint's recurrent state), parking the sequence.
    let mut other = text[..32000].to_vec();
    other.extend_from_slice(&text[50000..50100]);
    record(&mut out, "departed", &go(&mut model, &other, 1));
    // Past the next capacity with that sequence parked: it goes.
    let far = go(&mut model, &text[..70000], 1);
    record(&mut out, "far", &far);
    // And to the whole context.
    let whole_context = go(&mut model, &text[..140_000], 1);
    record(&mut out, "whole", &whole_context);
    assert_eq!(model.kv_positions(), 262144);
    let slots_after = model.slot_counts().0;
    if !whole {
        assert!(slots_after < slots_at_start, "{slots_at_start} -> {slots_after}");
    }
    drop(model);
    // Past the first capacity one token at a time, in decode.
    let mut model = open(&root, None);
    let before = go(&mut model, &text[..32760], 1);
    let decoded = go(&mut model, &text[..32760], 32);
    eprintln!("{} parts kept past 32768 in decode (from {:?})", kept(&before.state, &decoded.state), decoded.resumed);
    if let Some(file) = std::env::var_os("KV_GROWTH_OUT") {
        std::fs::write(file, out).unwrap();
    }
}

/// With the drafter, whose K/V is one more layer's (thirteen layers: 26 GiB hold the K/V of 262144 in VRAM): past the first capacity, the MTP layer's
/// K/V and indexer grown with the others' (its rows are in the state digest, which the file KV_GROWTH_OUT keeps).
#[test]
fn a_vram_kv_grows_with_a_drafter() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the model bundle directory") else { return };
    let Some(drafter) = common::model_env("STRAYDOG_DRAFTER", "the drafter directory") else { return };
    let text = text(70_100);
    let mut model = open(&root, Some(&drafter));
    assert!(model.partial && !model.host_kv, "partial {} host K/V {}", model.partial, model.host_kv);
    let mut out = String::new();
    for (name, end) in [("below", 32000), ("grown", 40000)] {
        let seen = go(&mut model, &text[..end], 1);
        record(&mut out, name, &seen);
    }
    if let Some(file) = std::env::var_os("KV_GROWTH_OUT") {
        std::fs::write(file, out).unwrap();
    }
}
