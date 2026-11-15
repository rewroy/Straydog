//! Resident expert payloads survive repeated growth of a live sequence's host K/V.
//!
//! Run one process under the GPU lock with STRAYDOG_KV_CACHE_RETAIN=1 and STRAYDOG_MODEL naming the UD-IQ1_S bundle:
//! cargo test --release --features hip -p straydog-engine --test kv_cache_retain_gpu -- --nocapture
//! STRAYDOG_QUANT may select another compatible bundle. KV_CACHE_RETAIN_BASELINE=1 runs the same fixture with retention disabled for comparison.
//! KV_CACHE_RETAIN_OUT writes growth timings, counters and resident snapshots as JSON, and each request's first logits in adjacent .f32 files.
//! Environment switches are set by the caller; the test never changes process-wide settings.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings, STARTING_KV};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

const STAT_NAMES: [&str; 16] = [
    "attempts", "successes", "fallbacks", "inplace", "d2d_experts", "d2d_bytes", "new_h2d_experts", "new_h2d_bytes",
    "temp_evicted_experts", "temp_reload_experts", "temp_reload_bytes", "cache_freed_bytes", "pack_launches",
    "chunks_allocated", "chunks_freed", "temp_evicted_bytes",
];

// Each row: layer, expert, group, slot, address, payload bytes, layout mask, payload hash.
type Residency = Vec<(usize, Vec<[u64; 8]>)>;
type Stats = Vec<(usize, [u64; 16])>;
type Key = (usize, u64, u64);

fn open(root: &std::ffi::OsStr, quantization: &str) -> (Model, Vec<u32>) {
    // The give-back regression uses the same budget and context to place K/V on the host while keeping a nonempty expert cache.
    let budget = 11u64 << 29;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, quantization).unwrap();
    let weights = Weights::open(&PathBuf::from(root), bundle, None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let prompt = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap()
        .encode("The history of the printing press begins in the fifteenth century, when").unwrap();
    let settings = Settings { context: 131072, state_budget_bytes: 6 << 30, prefill_batch_size: 256, speculative: 0,
                              cpu_threads: 0, row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial && model.host_kv, "partial {} host K/V {}: this test needs the host K/V path", model.partial, model.host_kv);
    model.prompt_share(false).unwrap();
    assert_eq!(model.locked_layers(), 0, "bank locking must not alter the cache while measuring a growth");
    assert_eq!(model.kv_positions(), STARTING_KV, "run with the default initial K/V capacity");
    (model, prompt)
}

fn memory_snapshot(model: &Model) -> Json {
    let raw: BTreeMap<_, _> = model.vram_budgets().unwrap().into_iter()
        .map(|(ordinal, used, budget)| (ordinal, (used, budget))).collect();
    let devices: Vec<_> = model.memory_accounting_for_test().unwrap().into_iter().map(|(ordinal, accounting)| {
        assert_eq!((accounting.requested_used, accounting.budget), raw[&ordinal], "raw telemetry changed meaning");
        let effective = accounting.effective_used().unwrap();
        assert_eq!(accounting.reserve, accounting.pool_reserve + accounting.kv_reserve + accounting.partial_reserve);
        assert!(accounting.peak_reserve >= accounting.reserve);
        if accounting.flags & 1 != 0 {
            assert_eq!(accounting.reserve, accounting.live_allocations.checked_mul(2 << 20).unwrap());
            assert_eq!(accounting.live_allocations, accounting.pool_allocations + accounting.kv_allocations + accounting.partial_allocations);
            assert!(effective <= accounting.budget, "GPU {ordinal} exceeds its effective budget: {accounting:?}");
            assert!([64u64 << 20, 128 << 20, 256 << 20].contains(&accounting.chunk_bytes));
        } else {
            assert_eq!(accounting.reserve, 0, "the ordinary path must not charge retention reserve");
        }
        let (name, free, total) = straydog_gpu::device_info(ordinal).unwrap();
        json!({ "ordinal": ordinal, "name": name, "device_info_free_bytes": free, "device_info_total_bytes": total,
                "effective_used_bytes": effective, "accounting": accounting.record() })
    }).collect();
    json!(devices)
}

struct Snapshot {
    capacity: usize,
    slots: (u64, u64),
    resident: Residency,
    stats: Stats,
    memory: Json,
}

impl Snapshot {
    fn read(model: &mut Model) -> Self {
        // The payload snapshot settles pending slot publication. Count slots after it, so rows cover every published owner.
        let resident = model.expert_cache_residency_for_test().unwrap();
        let slots = model.slot_counts();
        let stats = model.kv_cache_retain_stats_for_test().unwrap();
        let memory = memory_snapshot(model);
        let snapshot = Self { capacity: model.kv_positions(), slots, resident, stats, memory };
        assert_eq!(snapshot.resident.iter().map(|(ordinal, _)| *ordinal).collect::<Vec<_>>(), vec![0]);
        assert_eq!(snapshot.stats.iter().map(|(ordinal, _)| *ordinal).collect::<Vec<_>>(), vec![0]);
        let rows = snapshot.canonical();
        assert_eq!(rows.len() as u64, slots.1, "the snapshot must cover every settled resident expert");
        assert!(slots.1 <= slots.0 && slots.0 > 0, "invalid or empty expert cache: {slots:?}");
        snapshot
    }

    fn canonical(&self) -> BTreeMap<Key, [u64; 8]> {
        let mut rows = BTreeMap::new();
        let mut occupied = BTreeSet::new();
        for &(ordinal, ref resident) in &self.resident {
            for row in resident {
                let key = (ordinal, row[0], row[1]);
                assert!(row[4] > 0 && row[5] > 0, "a published resident must have an address and payload: {key:?} {row:?}");
                assert!(row[3] < self.slots.0, "resident slot exceeds the cache: {key:?} {row:?}");
                assert!(occupied.insert((ordinal, row[3])), "two experts own the same slot: {row:?}");
                assert!(rows.insert(key, *row).is_none(), "duplicate resident expert: {key:?}");
            }
        }
        rows
    }

    fn record(&self) -> Json {
        json!({ "kv_capacity": self.capacity, "slot_capacity": self.slots.0, "slots_held": self.slots.1,
                "resident_rows": &self.resident, "stats": &self.stats, "memory": &self.memory })
    }
}

fn stats_delta(before: &Snapshot, after: &Snapshot, retained: bool) -> Json {
    let earlier: BTreeMap<_, _> = before.stats.iter().copied().collect();
    assert_eq!(earlier.len(), before.stats.len());
    let later: BTreeMap<_, _> = after.stats.iter().copied().collect();
    assert_eq!(earlier.keys().collect::<Vec<_>>(), later.keys().collect::<Vec<_>>());
    let mut devices = Vec::new();
    for (ordinal, stats) in later {
        let mut delta = [0u64; 16];
        for index in 0..delta.len() {
            delta[index] = stats[index].checked_sub(earlier[&ordinal][index])
                .unwrap_or_else(|| panic!("retention counter went backwards: GPU {ordinal} {}", STAT_NAMES[index]));
        }
        if retained {
            assert!(delta[0] > 0 && delta[1] > 0, "no successful retention transaction on GPU {ordinal}: {delta:?}");
            assert!(delta[3] > 0 || delta[4] > 0, "no expert retained in place or by D2D on GPU {ordinal}: {delta:?}");
        } else {
            assert_eq!(delta, [0; 16], "the baseline must use the ordinary growth path");
        }
        let counters: serde_json::Map<String, Json> = STAT_NAMES.iter().zip(delta)
            .map(|(name, value)| ((*name).to_owned(), json!(value))).collect();
        devices.push(json!({ "ordinal": ordinal, "counters": counters }));
    }
    json!(devices)
}

fn first_logits(model: &mut Model, label: &str) -> String {
    let captured = model.logits_capture.take().expect("first-logits capture enabled");
    let first = captured.first().expect("the request produced logits");
    assert!(!first.is_empty() && first.iter().all(|value| value.is_finite()));
    let bytes: Vec<u8> = first.iter().flat_map(|value| value.to_le_bytes()).collect();
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if let Some(path) = std::env::var_os("KV_CACHE_RETAIN_OUT") {
        let mut path = PathBuf::from(path);
        path.set_extension(format!("{label}.f32"));
        std::fs::write(path, bytes).unwrap();
    }
    eprintln!("K/V cache first logits {label}: {digest}");
    digest
}

fn kept_by_growth(before: &Snapshot, after: &Snapshot) -> Json {
    let earlier = before.canonical();
    let later = after.canonical();
    let (mut common, mut unmoved, mut moved) = (0usize, 0usize, 0usize);
    for (key, row) in &earlier {
        let Some(next) = later.get(key) else { continue };
        assert_eq!(row[5], next[5], "resident payload length changed: {key:?}");
        assert_eq!(row[6], next[6], "resident layout changed: {key:?}");
        assert_eq!(row[7], next[7], "resident payload bytes changed: {key:?}");
        common += 1;
        if row[4] == next[4] { unmoved += 1; } else { moved += 1; }
    }
    assert!(common > 0, "growth left no common resident expert to check");
    // A common expert can also have been temporarily evicted and reloaded. Do not equate this set intersection with inplace/D2D counters.
    json!({ "common": common, "common_address_unchanged": unmoved, "common_address_changed": moved,
            "removed": earlier.len() - common, "added": later.len() - common })
}

/// Repeated allocation-only growth keeps the bytes and layout of every expert common to its settled snapshots.
#[test]
fn repeated_host_kv_growth_keeps_resident_expert_payloads() {
    let retained = std::env::var("STRAYDOG_KV_CACHE_RETAIN").is_ok_and(|value| value == "1");
    let baseline = std::env::var("KV_CACHE_RETAIN_BASELINE").is_ok_and(|value| value == "1");
    assert!(!(retained && baseline), "a baseline comparison must disable retention");
    if !retained && !baseline {
        eprintln!("skipped: set STRAYDOG_KV_CACHE_RETAIN=1 to check expert-cache retention during K/V growth");
        return;
    }
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let quantization = std::env::var("STRAYDOG_QUANT").unwrap_or_else(|_| "UD-IQ1_S".to_string());
    let (mut model, prompt) = open(&root, &quantization);
    let startup_memory = memory_snapshot(&model);
    let startup_slots = model.slot_counts();
    eprintln!("K/V cache startup memory: {startup_memory}");
    eprintln!("K/V cache startup slots: capacity {} held {}", startup_slots.0, startup_slots.1);
    assert_eq!(STARTING_KV, 32768, "this regression exercises 32K -> 64K -> 128K");
    model.logits_digests = Some(Vec::new());
    model.logits_capture = Some(Vec::new());
    let mut tokens = Vec::new();
    model.generate(&prompt, 13, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false,
                   &mut |token| tokens.push(token)).unwrap();
    assert!(!tokens.is_empty(), "the warm request emitted no tokens");
    let warm_logits = model.logits_digests.take().unwrap();
    assert!(!warm_logits.is_empty(), "the warm request captured no logits");
    let warm_first = first_logits(&mut model, "warm");
    let warm_metrics = model.last_metrics.clone();
    let mut before = Snapshot::read(&mut model);
    assert_eq!(before.capacity, 32768);
    assert!(before.slots.1 > 0, "the warm request left no resident expert to retain");
    let initial = before.record();
    let mut growths = Vec::new();
    for (needed, expected) in [(32769, 65536), (65537, 131072)] {
        let began = Instant::now();
        model.ensure_kv_for_test(needed).expect("allocation-only growth succeeds");
        let seconds = began.elapsed().as_secs_f64();
        assert_eq!(model.kv_positions(), expected, "the test needs memory for the doubled K/V capacity");
        let after = Snapshot::read(&mut model);
        let kept = kept_by_growth(&before, &after);
        let stats = stats_delta(&before, &after, retained);
        let growth_metrics = model.last_metrics.clone();
        // The next request also exercises the published pointers and the larger K/V pitches.
        model.reset().unwrap();
        model.logits_capture = Some(Vec::new());
        model.generate(&prompt, 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |_| {}).unwrap();
        let logits = first_logits(&mut model, &format!("grown-{expected}"));
        let record = json!({ "from": before.capacity, "requested": needed, "to": after.capacity, "seconds": seconds,
                             "retained": kept, "stats_delta": stats, "after": after.record(), "first_logits_sha256": logits,
                             "growth_metrics": growth_metrics });
        eprintln!("K/V cache retention: {} -> {} positions, {seconds:.6} s, {} common resident experts ({} same address, {} moved)",
                  before.capacity, after.capacity, record["retained"]["common"], record["retained"]["common_address_unchanged"],
                  record["retained"]["common_address_changed"]);
        eprintln!("K/V cache retention counters: {}", record["stats_delta"]);
        growths.push(record);
        before = Snapshot::read(&mut model);
    }
    assert_eq!(model.locked_layers(), 0, "the test must not page-lock banks between snapshots");
    if let Some(path) = std::env::var_os("KV_CACHE_RETAIN_OUT") {
        let report = json!({ "version": 1, "retain_enabled": retained, "quantization": quantization, "devices": [0], "vram_budget_bytes": 11u64 << 29,
                             "context": 131072, "state_budget_bytes": 6u64 << 30, "batch": 256, "host_kv": model.host_kv,
                             "warm_tokens": tokens, "warm_logits_digests": warm_logits, "warm_first_logits_sha256": warm_first,
                             "warm_metrics": warm_metrics, "prompt_tokens": prompt, "startup_memory": startup_memory,
                             "startup_slot_capacity": startup_slots.0, "startup_slots_held": startup_slots.1,
                             "resident_row_fields": ["layer", "expert", "group", "slot", "address", "payloadbytes", "layoutmask", "hash64"],
                             "stats_fields": STAT_NAMES, "initial": initial, "growths": growths });
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}
