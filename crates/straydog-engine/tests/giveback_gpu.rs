//! A partially resident model's page-locked banks give way to the K/V of a conversation (`Model::give_back`, called by `Model::ensure_kv` when the
//! growth of the host K/V is refused): the newest locked layers are taken back until the growth has its memory, the assist stops reading them in place,
//! the locking behind "ready" ends, and with nothing left to give back the request fails with `KV_NO_MEMORY` and the engine serves the next.
//!
//! cargo test --release --features cuda -p straydog-engine --test giveback_gpu, with STRAYDOG_MODEL naming the UD-IQ1_S bundle; under the GPU lock.
//! The model's own start locks no bank (it only plans them), so the test locks the last few layers itself, as `tests/assist_gpu.rs` does. The system
//! has memory enough: a refusal is made by the test hook of the GPU library (`refuse_host_allocations`), which answers the next host allocations of a
//! request as a machine short of memory does.
//!
//! The context is 131072, so that even Q8 K/V is in host memory at this budget and starts at 32768 positions. The growth the test asks for is
//! to 65536. A decode after a growth is compared with one of a model that never locked a bank, within the tolerance `behind_gpu.rs` uses:
//! every run is held to the same thirteen tokens by a constraint, which chooses the tokens and leaves the logits as they are, so that all
//! twelve decode rows are compared whichever way a near tie of two tokens falls.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use straydog_engine::constraint::{Alternative, Constraint, TokenBytes};

use straydog_engine::gpu::refuse_host_allocations;
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings, END_TOKENS, KV_NO_MEMORY, STARTING_KV};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

/// The tests load the model on the same GPU, and the hook of the library is the process's: one at a time.
static GPUS: Mutex<()> = Mutex::new(());

/// The layers whose banks the test locks, in the order they are locked: 47 is the newest.
const LOCKED: [usize; 4] = [44, 45, 46, 47];
/// The decode passes a run compares.
const PASSES: usize = 12;
/// The least relative difference of logits that two runs of one configuration are allowed (which experts the slots hold at a moment depends
/// on timing, and where a route goes moves a pass's logits by that much). Two runs of one start differed by 0, by 0.05 to 0.08 or by 0.12
/// to 0.16 on an R9700 (eighteen differences measured, the largest 0.163; of ten of the runs, six had the logits of some earlier run bit
/// for bit and four for the first eight rows: a route in a slot a pass sooner or later). One pair is measured for the tolerance and can be
/// a pair that agreed, and the floor is then all there is: 0.15 failed a run whose next request differed by 0.160 from a reference that
/// had differed by 0 from its twin. A bank read wrongly moves the logits past 1 and parts the tokens at once (`behind_gpu.rs`).
/// The runs take the same thirteen tokens ([`fixed_input`]): left to their own greedy tokens, two of them parted at the fourth row where
/// one expert of layer 3 was in a slot (F32 inputs) in one and on the CPU (INT8 inputs) in the other, the rows 0.04 to 0.05 apart, and
/// too few rows were left to compare.
const NOISE_FLOOR: f64 = 0.30;
/// The positions the growth is asked for: past the first capacity, within the context.
const GROWN: usize = 40000;

/// The model with no bank locked, its K/V on the host at its first capacity, and the prompt it continues.
fn open(root: &std::ffi::OsStr) -> (Model, Vec<u32>) {
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
    let weights = Weights::open(&PathBuf::from(root), bundle, None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let prompt = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap()
        .encode("The history of the printing press begins in the fifteenth century, when").unwrap();
    let settings = Settings { context: 131072, state_budget_bytes: 6 << 30, prefill_batch_size: 256, speculative: 0, cpu_threads: 0, row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial && model.host_kv, "partial {} host K/V {}: the K/V must be on the host at this budget", model.partial, model.host_kv);
    model.prompt_share(false).unwrap();
    assert_eq!(model.locked_layers(), 0, "the start only plans the locks");
    assert_eq!(model.kv_positions(), STARTING_KV, "the K/V starts at its first capacity");
    (model, prompt)
}

struct Run {
    prompt: Vec<u32>,
    inputs: Vec<u32>,
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

/// A request from the start of the prompt, greedy: the model is reset, and the passes' logits kept. It chooses the tokens the other runs take.
fn decode(model: &mut Model, prompt: &[u32]) -> Run {
    model.reset().unwrap();
    model.logits_capture = Some(Vec::new());
    let mut tokens = Vec::new();
    model.generate(prompt, PASSES + 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |token| tokens.push(token)).unwrap();
    Run { prompt: prompt.to_vec(), inputs: model.history.clone(), tokens, logits: model.logits_capture.take().unwrap() }
}

/// A constraint that admits `forced` and nothing else: each token's bytes are its id's four, so the sequence is one of ids, whatever
/// text the tokenizer gives them. The logits captured are those before the constraint.
fn fixed_constraint(tokens: Arc<TokenBytes>, forced: &[u32]) -> Constraint {
    let rest = forced.iter().flat_map(|token| token.to_be_bytes()).collect();
    Constraint::new(tokens, &[], false, vec![Alternative::Exact { rest }])
}

fn fixed_vocabulary(vocabulary: usize) -> Arc<TokenBytes> {
    assert!(vocabulary <= u32::MAX as usize);
    Arc::new(TokenBytes::new((0..vocabulary as u32).map(|token| token.to_be_bytes().to_vec()).collect(), vec![false; vocabulary]))
}

fn check_fixed_constraint(tokens: Arc<TokenBytes>, forced: &[u32]) {
    let mut constraint = fixed_constraint(tokens.clone(), forced);
    for &expected in forced {
        let allowed: Vec<_> = (0..tokens.len() as u32).filter(|&token| constraint.allows(token)).collect();
        assert_eq!(allowed, vec![expected], "the fixed-input constraint must admit exactly the next token id");
        constraint.advance(expected);
    }
    assert!((0..tokens.len() as u32).all(|token| !constraint.allows(token)), "no token is allowed after the exact sequence");
}

struct FixedInput {
    tokens: Arc<TokenBytes>,
    forced: Vec<u32>,
    prompt: Vec<u32>,
    vocabulary: usize,
}

fn fixed_input(root: &std::ffi::OsStr, grown: bool) -> FixedInput {
    // (a model of its own chooses the tokens and is gone before the runs compared open theirs)
    let seed = {
        let (mut model, prompt) = open(root);
        if grown { model.ensure_kv_for_test(GROWN).unwrap(); }
        decode(&mut model, &prompt)
    };
    assert_eq!(seed.tokens.len(), PASSES + 1, "the seed must provide all thirteen non-END token ids");
    assert_eq!(seed.logits.len(), PASSES + 1);
    let vocabulary = seed.logits[0].len();
    assert_eq!(vocabulary, 248320, "the complete model vocabulary is required");
    assert!(seed.logits.iter().all(|row| row.len() == vocabulary && row.iter().all(|value| value.is_finite())));
    assert!(seed.tokens.iter().all(|token| (*token as usize) < vocabulary && !END_TOKENS.contains(token)));
    let tokens = fixed_vocabulary(vocabulary);
    check_fixed_constraint(tokens.clone(), &seed.tokens);
    FixedInput { tokens, forced: seed.tokens, prompt: seed.prompt, vocabulary }
}

fn decode_fixed(model: &mut Model, prompt: &[u32], fixed: &FixedInput) -> Run {
    assert_eq!(prompt, fixed.prompt, "every run must receive the same prompt");
    model.reset().unwrap();
    model.logits_capture = Some(Vec::new());
    let mut constraint = fixed_constraint(fixed.tokens.clone(), &fixed.forced);
    let mut tokens = Vec::new();
    model.generate_constrained(prompt, PASSES + 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false,
                               &mut |token| tokens.push(token), Some(&mut constraint)).unwrap();
    assert_eq!(tokens, fixed.forced, "all thirteen emitted ids must match the fixed input sequence");
    let mut inputs = prompt.to_vec();
    inputs.extend_from_slice(&fixed.forced[..PASSES]); // The final emitted token is not forwarded.
    assert_eq!(model.history, inputs, "the model must actually process the same prompt plus twelve input ids");
    let logits = model.logits_capture.take().unwrap();
    assert_eq!(logits.len(), PASSES + 1, "one prefill row and all twelve ordinary decode rows are required");
    assert!(logits.iter().all(|row| row.len() == fixed.vocabulary && row.iter().all(|value| value.is_finite())));
    assert_eq!(model.last_metrics["decode_forward_tokens"].as_u64(), Some(PASSES as u64));
    Run { prompt: prompt.to_vec(), inputs, tokens, logits }
}

fn relative_difference(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        diff += f64::from(x - y).powi(2);
        norm += f64::from(*y).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}

/// The passes of two runs up to where their tokens part: how many were compared, and the worst relative difference of their logits.
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

/// [`compare`] of two runs over the same tokens: all twelve rows, the prompt and the tokens fed the same.
fn compare_fixed(one: &Run, other: &Run) -> (usize, f64) {
    assert_eq!(one.prompt, other.prompt);
    assert_eq!(one.inputs, other.inputs, "the processed input prefixes must be identical");
    assert_eq!(one.tokens, other.tokens, "the constrained runs must emit the same fixed ids");
    for run in [one, other] {
        assert_eq!(run.tokens.len(), PASSES + 1);
        assert_eq!(run.logits.len(), PASSES + 1);
        assert!(run.logits.iter().all(|row| row.len() == 248320 && row.iter().all(|value| value.is_finite())),
                "every compared row must contain the full finite vocabulary");
    }
    let (compared, worst) = compare(one, other);
    assert_eq!(compared, PASSES, "all twelve fixed-input comparisons are mandatory");
    (compared, worst)
}

/// The assist level in the request line of the last request: None while no layer is page-locked.
fn assist_level(model: &Model) -> Option<u64> {
    model.last_metrics["partial_detail"]["assist"].as_u64()
}

/// A K/V growth that is refused twice while four layers are locked takes back the two newest, one a refusal, and then succeeds; the decode after it
/// has the logits of a model that never locked a bank, the layers left locked are still assisted, and the GPU reads no bank of a layer given back.
#[test]
fn a_refused_growth_takes_back_the_newest_layers_and_then_succeeds() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let fixed = fixed_input(&root, true);
    // the reference: a model that never locked a bank, its K/V grown without a refusal (twice, for the noise between two runs)
    let reference = |root: &std::ffi::OsStr| {
        let (mut model, prompt) = open(root);
        model.ensure_kv_for_test(GROWN).unwrap();
        assert!(model.kv_positions() >= GROWN, "the K/V grew to {}", model.kv_positions());
        assert!(model.gave_back.is_empty());
        decode_fixed(&mut model, &prompt, &fixed)
    };
    let (none_one, none_two) = (reference(&root), reference(&root));
    let (compared, noise) = compare_fixed(&none_one, &none_two);
    let tolerance = (4.0 * noise).max(NOISE_FLOOR);
    eprintln!("never locked, twice: {compared} passes compared, the worst relative difference {noise:.2e}; the tolerance {tolerance:.2e}");
    assert!(compared >= 6, "too few passes before the tokens parted ({compared})");

    let (mut model, prompt) = open(&root);
    let mapped = model.lock_layers_for_test(&LOCKED).unwrap();
    assert!(mapped >= LOCKED.len() as u64, "{mapped} plans mapped of {}", LOCKED.len());
    assert_eq!(model.locked_layers(), LOCKED.len());
    decode_fixed(&mut model, &prompt, &fixed);
    assert!(assist_level(&model).is_some(), "the layers locked are assisted");

    // two refusals: the first layer's allocation of the growth, and the one asked again after the newest layer was given back
    straydog_engine::gpu::refuse_host_allocations(2);
    model.ensure_kv_for_test(GROWN).expect("the growth succeeds once layers are given back");
    refuse_host_allocations(0);
    assert!(model.kv_positions() >= GROWN, "the K/V grew to {}", model.kv_positions());
    assert_eq!(model.gave_back.len(), 2, "one give-back for each refusal: {:?}", model.gave_back);
    assert_eq!(model.gave_back[0].layers, vec![47], "the newest first");
    assert_eq!(model.gave_back[1].layers, vec![46]);
    assert_eq!(model.locked_layers(), 2, "the older layers stay locked");
    for gave in &model.gave_back {
        eprintln!("{}", straydog_engine::banks::give_back_line(gave));
        assert!(gave.bytes > 0 && gave.wanted > 0 && gave.asked_by.contains("K/V growth"));
    }

    // the layers left are assisted; the ones given back are read as any layer that is not locked: the decode is that of a model that never locked
    let run = decode_fixed(&mut model, &prompt, &fixed);
    assert!(assist_level(&model).is_some(), "the two layers left locked are still assisted");
    let (compared, worst) = compare_fixed(&run, &none_one);
    eprintln!("a decode after the give-back, against a model that never locked: {compared} passes compared, the worst relative difference {worst:.2e}");
    assert!(compared >= 6, "too few passes before the tokens parted ({compared})");
    assert!(worst < tolerance, "{worst} against {tolerance}");
}

/// With no layer left to give back the growth fails with `KV_NO_MEMORY`, every layer having been taken back first; the engine then serves the next
/// request (a shorter one, which needs no growth), and a growth that is not refused after it succeeds.
#[test]
fn with_nothing_left_to_give_back_the_request_fails_and_the_engine_serves_the_next() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let fixed = fixed_input(&root, false);
    let (mut never, prompt) = open(&root);
    let reference = decode_fixed(&mut never, &prompt, &fixed);
    drop(never);
    // (the same start once more, for what two runs of one configuration differ by)
    let (mut again, _) = open(&root);
    let (compared, noise) = compare_fixed(&decode_fixed(&mut again, &prompt, &fixed), &reference);
    drop(again);
    let tolerance = (4.0 * noise).max(NOISE_FLOOR);
    eprintln!("never locked, twice: {compared} passes compared, the worst relative difference {noise:.2e}; the tolerance {tolerance:.2e}");
    assert!(compared >= 6, "too few passes before the tokens parted ({compared})");

    let (mut model, prompt) = open(&root);
    model.lock_layers_for_test(&LOCKED).unwrap();
    assert_eq!(model.locked_layers(), LOCKED.len());
    // a request of 33000 tokens needs a growth at its start; every host allocation is refused
    let long: Vec<u32> = prompt.iter().copied().cycle().take(STARTING_KV + 300).collect();
    refuse_host_allocations(1000);
    let error = model.generate(&long, 4, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |_| {}).unwrap_err();
    refuse_host_allocations(0);
    assert!(error.ends_with(KV_NO_MEMORY), "{error}");
    assert_eq!(model.locked_layers(), 0, "every layer was given back before the request failed");
    let layers: Vec<usize> = model.gave_back.iter().flat_map(|gave| gave.layers.iter().copied()).collect();
    assert_eq!(layers, vec![47, 46, 45, 44], "newest first, one for each refusal");
    assert_eq!(model.kv_positions(), STARTING_KV, "the K/V is as it was");
    // the next request
    let run = decode_fixed(&mut model, &prompt, &fixed);
    assert_eq!(assist_level(&model), None, "no layer is locked: nothing is assisted or tuned");
    let (compared, worst) = compare_fixed(&run, &reference);
    eprintln!("the next request, against a model that never locked: {compared} passes compared, the worst relative difference {worst:.2e}");
    assert!(compared >= 6 && worst < tolerance, "{compared} passes, {worst} against {tolerance}");
    // and a growth that is not refused
    model.ensure_kv_for_test(GROWN).unwrap();
    assert!(model.kv_positions() >= GROWN, "the K/V grew to {}", model.kv_positions());
    let run = decode_fixed(&mut model, &prompt, &fixed);
    let (compared, worst) = compare_fixed(&run, &reference);
    assert!(compared >= 6 && worst < tolerance, "{compared} passes, {worst} against {tolerance}");
}

/// A give-back ends the locking behind "ready": its plan has no layer left, and a layer its thread locked meanwhile is let go when the model is handed it,
/// not taken.
#[test]
fn a_give_back_ends_the_locking_and_a_layer_locked_meanwhile_is_let_go() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (mut model, prompt) = open(&root);
    let mut job = model.lock_job_for_test(&LOCKED).unwrap();
    assert_eq!(job.remaining(), LOCKED.len());
    // SAFETY (all of them): the model is alive and holds its weights mapped.
    let first = unsafe { job.lock_next() }.unwrap().unwrap();
    model.banks_locked(first).unwrap();
    // the second layer is locked and checked, and a request comes before the model is handed it
    let second = unsafe { job.lock_next() }.unwrap().unwrap();
    assert_eq!(second.layer, 45);
    refuse_host_allocations(1);
    model.ensure_kv_for_test(GROWN).unwrap();
    refuse_host_allocations(0);
    assert_eq!(model.gave_back.iter().map(|gave| gave.layers.clone()).collect::<Vec<_>>(), vec![vec![44]]);
    assert_eq!(model.locked_layers(), 0);
    assert_eq!((job.remaining(), job.next_layer(), job.halted()), (0, None, true), "the plan has nothing left");
    assert!(unsafe { job.lock_next_behind() }.is_none());
    // the layer locked meanwhile is let go, with the reason the worker's line gives
    let error = model.banks_locked(second).unwrap_err();
    assert!(error.contains("given back"), "{error}");
    assert_eq!(model.locked_layers(), 0, "and it is not counted");
    // the model goes on: nothing is assisted, and the decode runs
    decode(&mut model, &prompt);
    assert_eq!(assist_level(&model), None);
    assert!(model.kv_positions() >= GROWN, "the K/V grew to {}", model.kv_positions());
}

/// A text of about `tokens` tokens, the same for the same `seed`: sentences of a sort that continue each other, so that two texts begin alike only when
/// they are given the same beginning.
fn text_of(text: &Text, seed: &str, tokens: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let mut n = 0;
    while out.len() < tokens {
        out.extend(text.encode(&format!("{seed} number {n}: the harbour clerk wrote down every ship that came in before the tide turned. ")).unwrap());
        n += 1;
    }
    out.truncate(tokens);
    out
}

/// What a parked sequence's copies need of host memory is refused: the request that overwrites the parked sequence's positions still runs. Layers locked
/// are given back for the copy first, newest first, all of them where it stays refused; the parked sequence is then dropped, and the engine goes on.
#[test]
fn a_copy_the_system_refuses_gives_back_the_locks_and_then_is_gone_without() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (mut model, _) = open(&root);
    let tokenizer = {
        let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
        let weights = Weights::open(&PathBuf::from(&root), bundle, None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
        Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap()
    };
    model.lock_layers_for_test(&LOCKED).unwrap();
    // a first request of 2000 tokens, then one that begins alike for 100 and goes on otherwise: the first sequence is parked, and its positions are copied
    // out as the second overwrites them
    let first = text_of(&tokenizer, "first", 2000);
    let mut second = first[..100].to_vec();
    second.extend(text_of(&tokenizer, "second", 2000));
    let go = |model: &mut Model, prompt: &[u32]| model.generate(prompt, 4, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |_| {});
    go(&mut model, &first).unwrap();
    straydog_engine::gpu::refuse_host_copies(1000);
    go(&mut model, &second).expect("a request survives the refusal of the memory to keep the sequence it overwrites");
    straydog_engine::gpu::refuse_host_copies(0);
    assert_eq!(model.locked_layers(), 0, "every locked layer was given back for the copy before it was given up");
    let layers: Vec<usize> = model.gave_back.iter().flat_map(|gave| gave.layers.iter().copied()).collect();
    assert_eq!(layers, vec![47, 46, 45, 44], "newest first");
    assert!(model.gave_back.iter().all(|gave| gave.asked_by.contains("copy")), "{:?}", model.gave_back);
    // and the next request runs
    go(&mut model, &first).expect("the engine serves the next request");
}
