//! A partially resident model's banks page-locked behind "ready", a layer at a time off the model's thread ([`Model::lock_job_for_test`],
//! `BankLocks::lock_next`, then `Model::banks_locked`: what the worker does, `crates/straydog/src/behind.rs`).
//!
//! cargo test --release --features cuda -p straydog-engine --test behind_gpu, with STRAYDOG_MODEL naming the UD-IQ1_S bundle; under the GPU lock.
//! The model's own start locks no bank (it only plans them), so the test locks the last few layers itself, through the same path.
//!
//! The model is opened again for each run: a run's state (the slots' contents and the routing scores they follow) is what its passes leave,
//! and the comparisons are between runs that began alike.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

/// The tests load the model on the same GPU: one at a time.
static GPUS: Mutex<()> = Mutex::new(());

/// The layers whose banks the test locks.
const LOCKED: [usize; 4] = [44, 45, 46, 47];
/// The decode passes a run compares.
const PASSES: usize = 12;
/// The least relative difference of logits that two runs of one configuration are allowed. Which experts the slots hold at a moment depends on
/// timing (the swaps that fill them run beside the passes), and where a route goes (a slot, the GPU's assist, the CPU's integer products) moves a
/// pass's logits by that much: two runs of the same start differed by 0, 8e-3, 1.9e-2, 3.2e-2 and 6.1e-2 here. A bank read wrongly, or a layer
/// mapped before it was checked, moves them past 1 and parts the tokens at once, which `compared` catches.
const NOISE_FLOOR: f64 = 0.15;

/// The model with no bank locked, and the prompt it continues.
fn open(root: &std::ffi::OsStr) -> (Model, Vec<u32>) {
    // (the budget of a 6 GiB card less what the display holds: most of the experts stay on the host)
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
    let weights = Weights::open(&PathBuf::from(root), bundle, None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let prompt = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap()
        .encode("The history of the printing press begins in the fifteenth century, when").unwrap();
    let settings = Settings { context: 8192, state_budget_bytes: 4 << 30, prefill_batch_size: 256, speculative: 0, cpu_threads: 0, row_cache_bytes: None };
    let model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial, "the model must be partially resident at this budget");
    assert_eq!(model.locked_layers(), 0, "the start only plans the locks");
    (model, prompt)
}

/// What a run of `PASSES` decode passes left: its tokens and each pass's logits.
struct Run {
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

/// A request from the start of the prompt: the model is reset, and the passes' logits kept.
fn decode(model: &mut Model, prompt: &[u32]) -> Run {
    model.reset().unwrap();
    model.logits_capture = Some(Vec::new());
    let mut tokens = Vec::new();
    model.generate(prompt, PASSES + 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |token| tokens.push(token)).unwrap();
    Run { tokens, logits: model.logits_capture.take().unwrap() }
}

/// The assist level in the request line of the last request: None while no layer is page-locked.
fn assist_level(model: &Model) -> Option<u64> {
    model.last_metrics["partial_detail"]["assist"].as_u64()
}

fn relative_difference(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        diff += f64::from(x - y).powi(2);
        norm += f64::from(*y).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}

/// The passes of two runs up to where their tokens part: the worst relative difference of their logits, and how many were compared.
fn compare(one: &Run, other: &Run) -> (usize, f64) {
    let (mut compared, mut worst) = (0, 0f64);
    for pass in 0..PASSES {
        if one.tokens.get(..=pass) != other.tokens.get(..=pass) {
            break;
        }
        worst = worst.max(relative_difference(&one.logits[pass + 1], &other.logits[pass + 1]));
        compared += 1;
    }
    (compared, worst)
}

/// The assist turns on with the first layer locked and takes each layer as it comes: the plans the GPU can read grow by one with each, the
/// level is in the request line from the first, and a request between two layers runs.
#[test]
fn the_assist_turns_on_as_layers_come() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (mut model, prompt) = open(&root);
    decode(&mut model, &prompt);
    assert_eq!(assist_level(&model), None, "no layer is locked: nothing is assisted or tuned");
    let mut job = model.lock_job_for_test(&LOCKED).unwrap();
    assert_eq!(job.remaining(), LOCKED.len());
    for (index, layer) in LOCKED.iter().enumerate() {
        assert_eq!(job.next_layer(), Some(*layer));
        // SAFETY: the model is alive and holds its weights mapped.
        let locked = unsafe { job.lock_next() }.unwrap().unwrap();
        assert_eq!(locked.layer, *layer);
        assert!(locked.bytes > 0);
        let mapped = model.banks_locked(locked).unwrap();
        assert_eq!(mapped, index as u64 + 1, "the plans the GPU can read after {} layers", index + 1);
        assert_eq!(model.locked_layers(), index + 1);
        // (a request between layers: its passes capture the layers' graphs with the banks mapped so far)
        let run = decode(&mut model, &prompt);
        assert_eq!(run.tokens.len(), PASSES + 1);
        assert!(assist_level(&model).is_some(), "the level is in force from the first layer locked");
    }
    assert!(unsafe { job.lock_next() }.is_none(), "the plan is done");
    model.banks_lock_ended(None);
    let info = model.info().unwrap();
    eprintln!("assisted routes over the runs: {}", info["assisted_experts"]);
    assert!(info["assisted_experts"].as_u64().unwrap() > 0, "the layers locked assisted nothing in the requests after them");
}

/// A decode while layers are being locked, which the worker does outside the model's lock, has the results of a decode with none locked, and one
/// after they are registered those of a model that locked them first: close, and as close as two runs of the same are to each other. Not equal
/// bit for bit in either case: the assist decides which device computes an expert, and which experts the slots hold at a moment depends on
/// timing (docs/DESIGN.md, partial residency).
#[test]
fn a_decode_during_the_locking_matches_one_after() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // the references: the locks taken before the first pass (twice, for the noise between two runs), and none
    let after = |root: &std::ffi::OsStr| {
        let (mut model, prompt) = open(root);
        model.lock_layers_for_test(&LOCKED).unwrap();
        decode(&mut model, &prompt)
    };
    let (after_one, after_two) = (after(&root), after(&root));
    let none = {
        let (mut model, prompt) = open(&root);
        decode(&mut model, &prompt)
    };
    let (compared, noise) = compare(&after_one, &after_two);
    let tolerance = (4.0 * noise).max(NOISE_FLOOR);
    eprintln!("locked first, twice: {compared} passes compared, the worst relative difference {noise:.2e}; the tolerance {tolerance:.2e}");
    assert!(compared >= 6, "too few passes before the tokens parted ({compared})");

    let (model, prompt) = open(&root);
    let model = Arc::new(Mutex::new(model));
    let job = model.lock().unwrap().lock_job_for_test(&LOCKED).unwrap();
    // A request holds the model for all of its passes, as the worker's does. The locker locks and checks the layers meanwhile, holding nothing
    // of the model, and waits to register each until the request ends.
    let mut request = model.lock().unwrap();
    let locked_meanwhile = Arc::new(AtomicUsize::new(0));
    let locker = {
        let (model, locked_meanwhile) = (model.clone(), locked_meanwhile.clone());
        std::thread::spawn(move || {
            let mut job = job;
            // SAFETY: the thread holds the model's Arc, and the model its weights mapped.
            while let Some(next) = unsafe { job.lock_next() } {
                let locked = next.unwrap();
                locked_meanwhile.fetch_add(1, Ordering::SeqCst);
                model.lock().unwrap().banks_locked(locked).unwrap();
            }
        })
    };
    // the first request: its passes are compared; more follow while the locker is still at its first layer, so that it is at work in the passes
    let during = decode(&mut request, &prompt);
    let mut requests = 1;
    let began = std::time::Instant::now();
    while locked_meanwhile.load(Ordering::SeqCst) == 0 && began.elapsed().as_secs() < 120 {
        decode(&mut request, &prompt);
        requests += 1;
    }
    assert_eq!(request.locked_layers(), 0, "no layer is registered while a request holds the model");
    assert_eq!(assist_level(&request), None, "and none is assisted");
    eprintln!("{requests} request(s) ran while the locker took its first layer ({} locked by the end of them)", locked_meanwhile.load(Ordering::SeqCst));
    drop(request);
    locker.join().unwrap();
    let mut model = model.lock().unwrap();
    assert_eq!(model.locked_layers(), LOCKED.len());
    let (compared, worst) = compare(&during, &none);
    eprintln!("a decode during the locking, against none locked: {compared} passes compared, the worst relative difference {worst:.2e}");
    assert!(compared >= 6, "too few passes before the tokens parted ({compared})");
    assert!(worst < tolerance, "{worst} against {tolerance}");
    let later = decode(&mut model, &prompt);
    assert!(assist_level(&model).is_some(), "the assist is on once the layers are registered");
    let (compared, worst) = compare(&later, &after_one);
    eprintln!("a decode after, against a model that locked first: {compared} passes compared, the worst relative difference {worst:.2e}");
    assert!(compared >= 6, "too few passes before the tokens parted ({compared})");
    assert!(worst < tolerance, "{worst} against {tolerance}");
}
