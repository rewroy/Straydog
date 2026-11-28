//! A partially resident model whose expert pool is made smaller between two requests of a conversation (the served worker's Windows
//! budget falling: `adjust_windows_vram`, then `Model::resize_vram`) serves the second as it does with the pool it had: on an RTX 4060
//! (UD-IQ1_S, 5.5 GiB) five of seven second messages that failed with an illegal memory access came right after such a shrink, and no
//! test made one.
//!
//! cargo test --release --features hip -p straydog-engine --test shrink_gpu -- --nocapture, with STRAYDOG_MODEL naming the UD-IQ1_S
//! bundle; under the GPU lock. One model serves two conversations, with and without a shrink before the second request, its host
//! memory as SHRINK_MODE says: `unlocked` (no bank locked), `locked` (every layer the start planned locked, as the worker locks them
//! behind "ready") or `tier` (the host tier the worker copies in place of whole layers where the lock budget holds fewer than all: the
//! caller sets STRAYDOG_HOST_TIER=1 and a STRAYDOG_LOCK_BUDGET of fewer GiB than the banks; a first conversation gives the routes it
//! ranks experts by). STRAYDOG_PREGATHER=1 adds the gather ahead on Linux, as Windows has it. On a CUDA build the locked layers' and the
//! tier's streamed chunks go over through CopyBatch (cudaMemcpyBatchAsync), one call a copy elsewhere.
//! A conversation is a long prompt (the three fixed texts of tests/data/ds41_slot_wmma, about 4,500 tokens, read layer by layer) and a
//! continuation of a few hundred characters (its new rows stream the experts the slots lack, chunk by chunk, where the RTX 4060 failed).
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

/// The tokens a request takes.
const TOKENS: usize = 9;
/// What the shrink takes from the budget: a little more than the 4060's 0.25 GiB (5.44 -> 5.19).
const SHRINK: u64 = 384 << 20;
/// The largest relative difference of two runs' logits allowed where their tokens agree (tests/giveback_gpu.rs: which experts the slots
/// hold depends on timing, and that moved a pass's logits by up to 0.163 on an R9700; a bank read wrongly moves them past 1).
const NOISE_FLOOR: f64 = 0.30;

struct Run {
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

fn open(root: &std::ffi::OsStr) -> (Model, Text) {
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
    let weights = Weights::open(&PathBuf::from(root), bundle, None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let text = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap();
    let settings = Settings { context: 131072, state_budget_bytes: 6 << 30, prefill_batch_size: 256, speculative: 0, cpu_threads: 0, row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial, "the model must be partially resident at this budget");
    model.prompt_share(false).unwrap();
    (model, text)
}

fn request(model: &mut Model, tokens: &[u32]) -> Run {
    model.logits_capture = Some(Vec::new());
    let mut emitted = Vec::new();
    model.generate(tokens, TOKENS, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |token| emitted.push(token)).unwrap();
    let logits = model.logits_capture.take().unwrap();
    assert!(!logits.is_empty() && logits.iter().all(|row| row.iter().all(|value| value.is_finite())), "every row's logits are finite");
    Run { tokens: emitted, logits }
}

fn relative_difference(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        diff += f64::from(x - y).powi(2);
        norm += f64::from(*y).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}

/// The rows of two runs up to where their tokens part (the first row always): how many, and the worst relative difference.
fn compare(one: &Run, other: &Run) -> (usize, f64) {
    let mut worst = relative_difference(&one.logits[0], &other.logits[0]);
    let mut compared = 1;
    for row in 1..one.logits.len().min(other.logits.len()) {
        if one.tokens.get(..row) != other.tokens.get(..row) {
            break;
        }
        worst = worst.max(relative_difference(&one.logits[row], &other.logits[row]));
        compared += 1;
    }
    (compared, worst)
}

/// A conversation from the start: the prompt, then its continuation, with the pool made smaller between them where `shrink`, and the budget
/// given back after. The continuation's run.
fn conversation(model: &mut Model, prompt: &[u32], more: &[u32], shrink: bool, label: &str) -> Run {
    model.reset().unwrap();
    let first = request(model, prompt);
    let mut continued = model.history.clone();
    continued.push(*first.tokens.last().unwrap());
    continued.extend_from_slice(more);
    let (_, _, budget) = model.vram_budgets().unwrap()[0];
    if shrink {
        let (before, after) = model.resize_vram(0, budget - SHRINK).unwrap();
        eprintln!("shrink {label}: budget {:.2} -> {:.2} GiB, expert slots {before} -> {after}", budget as f64 / (1u64 << 30) as f64,
                  model.vram_budgets().unwrap()[0].2 as f64 / (1u64 << 30) as f64);
        assert!(after < before && after > 0, "the shrink must leave a smaller pool that is not empty: {before} -> {after}");
    }
    let second = request(model, &continued);
    let reused = model.last_metrics["cached_tokens"].as_u64().unwrap_or(0);
    eprintln!("shrink {label}: continuation of {} tokens, {reused} reused, tokens {:?}, locked layers {}", continued.len(), second.tokens,
              model.locked_layers());
    if shrink {
        let (before, after) = model.resize_vram(0, budget).unwrap();
        eprintln!("shrink {label}: budget given back, expert slots {before} -> {after}");
    }
    second
}

/// The continuation after a shrink has the logits of one without, the host memory as SHRINK_MODE says.
#[test]
fn a_pool_made_smaller_between_requests_serves_the_continuation() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let (mut model, text) = open(&root);
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ds41_slot_wmma");
    let long: String = ["a.txt", "b.txt", "c.txt"].iter().map(|name| std::fs::read_to_string(data.join(name)).unwrap()).collect::<Vec<_>>().join("\n\n");
    let prompt = text.encode(&format!("Read these notes and say what they are about.\n\n{long}")).unwrap();
    let more = text.encode("\n\nThank you. Now, in two sentences, which of the notes above would a new reader find hardest to follow, and why? \
                            Answer plainly, without lists, and name the note by its first words.").unwrap();
    eprintln!("shrink: prompt {} tokens, continuation {} more", prompt.len(), more.len());
    let mode = std::env::var("SHRINK_MODE").unwrap_or_else(|_| "unlocked".into());
    match mode.as_str() {
        "unlocked" => {}
        "locked" => {
            model.lock_banks_now().unwrap();
            eprintln!("shrink: {} of 48 layers locked", model.locked_layers());
            assert!(model.locked_layers() > 0, "the start planned no lock: nothing to compare with locks");
        }
        "tier" => {
            // (a conversation first, for the routes the tier ranks experts by: its decode's host routes)
            let _ = conversation(&mut model, &prompt, &more, false, "tier, routes");
            let routes = model.route_counts().unwrap();
            let mut job = model.bank_locks(Some(&routes)).expect("the start planned locks for the tier to take the place of");
            let mut taken = 0;
            // SAFETY: the plan is of this model's own weights, which it keeps mapped.
            while let Some(next) = unsafe { job.lock_next() } {
                model.banks_locked(next.map_err(|stop| stop.message()).unwrap()).unwrap();
                taken += 1;
            }
            model.banks_lock_ended(None);
            eprintln!("shrink: the tier's turns {taken}, {} of 48 layers locked", model.locked_layers());
            assert!(taken > 0, "no tier was made: set STRAYDOG_HOST_TIER=1 and a STRAYDOG_LOCK_BUDGET below the banks");
        }
        other => panic!("SHRINK_MODE {other}: unlocked, locked or tier"),
    }
    let kept = conversation(&mut model, &prompt, &more, false, &format!("{mode}, kept"));
    let shrunk = conversation(&mut model, &prompt, &more, true, &format!("{mode}, shrunk"));
    let (compared, worst) = compare(&kept, &shrunk);
    eprintln!("shrink {mode}: {compared} rows compared, worst relative difference {worst:.4}");
    assert!(worst <= NOISE_FLOOR, "{mode}: the continuation after a shrink differs by {worst:.4} over {compared} rows");
}
