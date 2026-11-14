//! A partially resident model's K/V in VRAM starting small and growing under a live sequence (`Model::ensure_kv`, `sd_kv_grow` on a device K/V): what it
//! held stays, and what it takes comes from the expert slots for good.
//!
//! cargo test --release -p straydog-engine --features hip --test kv_growth_vram_gpu, with STRAYDOG_MODEL (a Qwen3.8 bundle with experts beyond one GPU's
//! budget), STRAYDOG_QUANT (its quantization; the golden's where unset), STRAYDOG_VRAM_GIB (the budget; 24, which holds a 262144 context's K/V of twelve
//! layers in VRAM, where unset; 26 with STRAYDOG_DRAFTER, thirteen layers) and, for the second test, STRAYDOG_DRAFTER; one test at a time. KV_GROWTH_OUT
//! writes each request's logits and state digests: with STRAYDOG_KV_START=0 (the whole context from the start, no growth) the file is the same.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::collections::{BTreeMap, BTreeSet};
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

struct Seen { first: u32, logits: Vec<String>, state: Vec<String>, resumed: Json, cached: u64, generated: usize }

fn go(model: &mut Model, prompt: &[u32], max_new: usize) -> Seen {
    model.logits_digests = Some(Vec::new());
    let mut tokens = Vec::new();
    model.generate(prompt, max_new, &straydog_engine::sampling::Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true,
                   &mut |token| tokens.push(token)).unwrap();
    Seen { first: tokens[0], logits: model.logits_digests.take().unwrap(), state: model.state_digest().unwrap(),
           resumed: model.last_metrics.get("resumed_from").cloned().unwrap_or(Json::Null),
           cached: model.last_metrics.get("cached_tokens").and_then(Json::as_u64).expect("numeric cached_tokens after generate"),
           generated: tokens.len() }
}

struct Positional {
    key: String, hash: String, start: u64, end: u64, full_run: bool, indexer: bool, mtp: bool,
}

fn position_number(text: &str) -> u64 {
    assert!(!text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()), "invalid position: {text}");
    let value = text.parse::<u64>().expect("position overflow");
    assert_eq!(value.to_string(), text, "noncanonical position");
    value
}

fn positional(state: &[String]) -> BTreeMap<String, Positional> {
    let mut seen = BTreeSet::new();
    let mut parts = BTreeMap::new();
    for line in state {
        let (label, hash) = line.rsplit_once(' ').expect("digest label and hash");
        assert!(hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()), "invalid digest: {line}");
        assert!(seen.insert(label), "duplicate digest label: {label}");
        let fields: Vec<_> = label.split(' ').collect();
        assert!(fields.len() >= 3 && fields[0] == "GPU" && fields.iter().all(|field| !field.is_empty()), "unknown digest label: {label}");
        position_number(fields[1]);
        let indexer = fields[2].starts_with("@indexer:");
        if !indexer && !fields[2].starts_with("@qsa:") {
            assert_eq!(fields.len(), 3, "unknown positional state: {label}");
            continue;
        }
        assert_eq!(fields.len(), 5, "unknown positional label: {label}");
        if indexer && fields[3] == "history" {
            position_number(fields[4]);
            continue;
        }
        let (run, unit, width) = match (indexer, fields[3]) {
            (false, "rows") => (true, 1u64, 256u64),
            (false, "row") => (false, 1, 256),
            (true, "blocks") => (true, 4, 64),
            (true, "block") => (false, 4, 64),
            _ => panic!("unknown positional units: {label}"),
        };
        let (low, high) = if run {
            let (low, high) = fields[4].split_once("..").expect("positional range");
            let (low, high) = (position_number(low), position_number(high));
            assert!(low < high && high - low <= width && low % width == 0, "invalid positional run: {label}");
            (low, high)
        } else {
            let low = position_number(fields[4]);
            (low, low.checked_add(1).expect("position end overflow"))
        };
        let start = low.checked_mul(unit).expect("position start overflow");
        let end = high.checked_mul(unit).expect("position end overflow");
        parts.insert(label.to_owned(), Positional { key: fields[..3].join(" "), hash: hash.to_owned(), start, end,
            full_run: run && end - start == 256, indexer, mtp: fields[2].ends_with(":blk.48.") });
    }
    parts
}

fn positional_keys(parts: &BTreeMap<String, Positional>) -> BTreeMap<String, (bool, bool, u64)> {
    let mut keys = BTreeMap::new();
    for part in parts.values() {
        let entry = keys.entry(part.key.clone()).or_insert((part.indexer, part.mtp, 0));
        entry.2 = entry.2.max(part.end);
    }
    assert!(!keys.is_empty(), "no positional state to compare");
    for (key, (indexer, _, _)) in &keys {
        let paired = if *indexer { key.replacen("@indexer:", "@qsa:", 1) } else { key.replacen("@qsa:", "@indexer:", 1) };
        assert!(keys.contains_key(&paired), "missing qsa/indexer partner of {key}");
    }
    keys
}

fn kept_by_growth(before: &[String], after: &[String]) -> usize {
    let before = positional(before);
    let after = positional(after);
    for key in positional_keys(&before).keys() {
        assert!(before.values().any(|part| &part.key == key && part.full_run), "no complete run for {key}");
    }
    assert_eq!(before.len(), after.len(), "allocation-only growth changed the positional label set");
    for (label, part) in &before {
        let later = after.get(label).unwrap_or_else(|| panic!("growth dropped {label}"));
        assert_eq!(part.hash, later.hash, "changed by allocation-only growth: {label}");
    }
    before.len()
}

fn kept_prefix(before: &[String], after: &[String], cached: u64, held: u64) -> usize {
    assert!(cached > 0 && cached <= held, "no valid untouched prefix: cached {cached}, held {held}");
    let before = positional(before);
    let after = positional(after);
    let keys = positional_keys(&before);
    let mut required = BTreeMap::new();
    for (key, (indexer, mtp, valid_end)) in &keys {
        let boundary = cached.saturating_sub(u64::from(*mtp));
        assert!(*valid_end <= held.saturating_sub(u64::from(*mtp)), "digest extends past held positions: {key}");
        let runs = boundary.min(*valid_end) / 256;
        assert!(runs > 0, "no complete untouched prefix run for {key}, cached {cached}");
        let (tag, unit) = if *indexer { ("blocks", 4) } else { ("rows", 1) };
        for run in 0..runs {
            let low = run * 256 / unit;
            let high = (run + 1) * 256 / unit;
            let label = format!("{key} {tag} {low}..{high}");
            assert!(before.get(&label).is_some_and(|part| part.full_run), "before digest lacks required prefix run: {label}");
            assert!(after.get(&label).is_some_and(|part| part.full_run), "after digest lost required prefix run: {label}");
        }
        required.insert(key.clone(), (runs, 0u64));
    }
    let mut same = 0;
    for (label, part) in &before {
        let boundary = cached.saturating_sub(u64::from(part.mtp));
        if part.start >= boundary || part.end > boundary { continue; }
        let Some(later) = after.get(label) else { continue };
        assert_eq!(part.hash, later.hash, "untouched prefix changed at cached {cached}: {label}");
        if part.full_run { required.get_mut(&part.key).unwrap().1 += 1; }
        same += 1;
    }
    for (key, (expected, compared)) in required {
        assert_eq!(compared, expected, "incomplete prefix coverage for {key}");
    }
    assert!(same > 0, "no untouched positional parts compared");
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
    // Below the first capacity, then an allocation-only growth before another request.
    let before = go(&mut model, &text[..32000], 1);
    record(&mut out, "below", &before);
    assert_eq!(model.kv_positions(), start);
    model.ensure_kv_for_test(STARTING_KV + 1).unwrap();
    let allocated = model.state_digest().unwrap();
    eprintln!("{} positional parts preserved by allocation-only growth ({start} -> {})",
              kept_by_growth(&before.state, &allocated), model.kv_positions());
    if !whole {
        assert_eq!(model.kv_positions(), 65536);
        assert!(model.slot_counts().0 < slots_at_start, "allocation-only growth must take slots");
    } else {
        assert_eq!(model.kv_positions(), start);
    }
    // A no-growth re-entry changed 2760 parts, first at @qsa:blk.3. rows 4864..5120, because checkpoint resumption recomputed the suffix.
    // Check the allocation copy exactly, then only whole records below numeric cached_tokens; indexer blocks cover four tokens each.
    let grown = go(&mut model, &text[..40000], 1);
    record(&mut out, "grown", &grown);
    eprintln!("{} untouched prefix parts kept at cached {}", kept_prefix(&allocated, &grown.state, grown.cached, 32000), grown.cached);
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
    if !whole { assert_eq!(model.kv_positions(), 131072); }
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
    assert_eq!(model.kv_positions(), start);
    let decoded = go(&mut model, &text[..32760], 32);
    eprintln!("{} untouched prefix parts kept in decode at cached {} (from {:?})",
              kept_prefix(&before.state, &decoded.state, decoded.cached, 32760), decoded.cached, decoded.resumed);
    if !whole {
        assert!(32760usize.checked_add(decoded.generated.saturating_sub(1)).unwrap() > 32768,
                "decode ended before crossing the first capacity: {} emitted tokens", decoded.generated);
        assert_eq!(model.kv_positions(), 65536, "decode must cross the first capacity");
    }
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
