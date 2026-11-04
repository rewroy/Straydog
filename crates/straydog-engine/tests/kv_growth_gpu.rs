//! A partially resident model's host K/V growing under a live sequence (`Model::ensure_kv`, `sd_kv_grow`): what it held stays.
//!
//! cargo test --release -p straydog-engine --features hip --test kv_growth_gpu, with STRAYDOG_MODEL and STRAYDOG_DRAFTER, one test
//! at a time. KV_GROWTH_OUT writes each request's logits and state digests: with `STARTING_KV` at the whole context (no growth)
//! the file is the same, and the same with the drafter dropped.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::weights::Weights;

fn open(root: &std::ffi::OsStr, quantization: &str) -> Model { open_with(root, None, quantization) }

fn open_with(root: &std::ffi::OsStr, drafter: Option<&std::ffi::OsStr>, quantization: &str) -> Model {
    let budget = 24u64 << 30;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let weights = Weights::open(&PathBuf::from(root), straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, quantization).unwrap(), drafter.map(PathBuf::from).as_deref(), straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let settings = Settings { context: 262144, state_budget_bytes: 14 << 30, prefill_batch_size: 4096, speculative: if drafter.is_some() { 3 } else { 0 }, cpu_threads: 0, row_cache_bytes: None };
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

/// The positional parts both digests have, which must be equal: `below` was the whole of the earlier sequence.
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

#[test]
fn growth_keeps_the_sequence() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the model bundle directory") else { return };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/model.json");
    let golden: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let quantization = golden["quantization"].as_str().unwrap();
    let requests = golden["placements"]["two-gpu-experts-mtp"]["requests"].as_array().unwrap();
    let mut text: Vec<u32> = Vec::new();
    while text.len() < 70100 {
        for request in requests {
            text.extend(serde_json::from_value::<Vec<u32>>(request["prompt"].clone()).unwrap());
        }
        text.push(text.len() as u32 % 200000);
    }
    let mut out = String::new();
    let mut record = |name: &str, seen: &Seen| {
        let state = Sha256::digest(seen.state.join("\n").as_bytes());
        let line = format!("{name} first {} logits {:?} state {state:x} resumed {}\n", seen.first, seen.logits, seen.resumed);
        eprint!("{line}");
        out.push_str(&line);
    };
    let mut model = open(&root, quantization);
    assert!(model.partial && model.host_kv, "partial {} host K/V {}", model.partial, model.host_kv);
    // Below the first capacity, then past it in a prefill.
    let before = go(&mut model, &text[..32000], 1);
    record("below", &before);
    let grown = go(&mut model, &text[..40000], 1);
    record("grown", &grown);
    eprintln!("{} parts kept past 32768", kept(&before.state, &grown.state));
    // A departure resuming below the growth (a checkpoint's recurrent state), parking the sequence.
    let mut other = text[..32000].to_vec();
    other.extend_from_slice(&text[50000..50100]);
    record("departed", &go(&mut model, &other, 1));
    // Past the next capacity with that sequence parked: it goes.
    let far = go(&mut model, &text[..70000], 1);
    record("far", &far);
    drop(model);
    // Past the first capacity one token at a time, in decode.
    let mut model = open(&root, quantization);
    let before = go(&mut model, &text[..32760], 1);
    let decoded = go(&mut model, &text[..32760], 32);
    eprintln!("{} parts kept past 32768 in decode (from {:?})", kept(&before.state, &decoded.state), decoded.resumed);
    if let Some(file) = std::env::var_os("KV_GROWTH_OUT") {
        std::fs::write(file, out).unwrap();
    }
}

/// With the drafter given, which a partially resident model keeps where it fits, past the first capacity.
#[test]
fn growth_with_a_drafter() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the model bundle directory") else { return };
    let Some(drafter) = common::model_env("STRAYDOG_DRAFTER", "the drafter directory") else { return };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/model.json");
    let golden: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let requests = golden["placements"]["two-gpu-experts-mtp"]["requests"].as_array().unwrap();
    let mut text: Vec<u32> = Vec::new();
    while text.len() < 70100 {
        for request in requests {
            text.extend(serde_json::from_value::<Vec<u32>>(request["prompt"].clone()).unwrap());
        }
        text.push(text.len() as u32 % 200000);
    }
    let mut model = open_with(&root, Some(&drafter), golden["quantization"].as_str().unwrap());
    assert!(model.partial && model.host_kv, "partial {} host K/V {}", model.partial, model.host_kv);
    let mut out = String::new();
    for (name, end) in [("below", 32000), ("grown", 40000)] {
        let seen = go(&mut model, &text[..end], 1);
        let state = Sha256::digest(seen.state.join("\n").as_bytes());
        out.push_str(&format!("{name} first {} logits {:?} state {state:x} resumed {}\n", seen.first, seen.logits, seen.resumed));
    }
    eprint!("{out}");
    if let Some(file) = std::env::var_os("KV_GROWTH_OUT") {
        std::fs::write(file, out).unwrap();
    }
}
