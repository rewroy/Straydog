//! A partially resident layer's assist level changed between decode passes (`Model::assist_script`, which stands in for the tuner's rounds,
//! `crate::assist`): every pass runs at the level asked for and the results stay those of the model at that level.
//!
//! cargo test --release --features cuda -p straydog-engine --test assist_gpu, with STRAYDOG_MODEL naming the UD-IQ1_S bundle; under the GPU lock.
//! No bank is page-locked by the model's own start on a machine whose commit is short (the real start locks as many layers as the commit
//! allows), so the test locks the last few layers' banks itself (`Model::lock_layers_for_test`): the assist is then taken by those layers.
//!
//! The model is opened again for each run: a run's state (the slots' contents and the routing scores they follow) is what its passes
//! leave, and the comparisons are between runs that began alike.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use serde_json::Value as Json;
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

/// The tests load the model on the same GPU: one at a time.
static GPUS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The layers whose banks the test locks.
const LOCKED: [usize; 4] = [44, 45, 46, 47];

/// The model, with the banks of the last layers locked, the prompt it continues, and the plans whose banks the GPU can read (the layers
/// locked, here or by the start).
fn open(root: &std::ffi::OsStr) -> (Model, Vec<u32>, u64) {
    // (the budget of a 6 GiB card less what the display holds: most of the experts stay on the host)
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
    let weights = Weights::open(&PathBuf::from(root), bundle, None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    // (a continuation of plain text goes on for as long as the test asks)
    let prompt = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap()
        .encode("The history of the printing press begins in the fifteenth century, when").unwrap();
    let settings = Settings { context: 8192, state_budget_bytes: 4 << 30, prefill_batch_size: 256, speculative: 0, cpu_threads: 0, row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial, "the model must be partially resident at this budget");
    // (the prompt's experts all stream: the runs compared begin alike; prompt_cpu_gpu checks the CPU's share of a prompt)
    model.prompt_share(false).unwrap();
    let mapped = model.lock_layers_for_test(&LOCKED).unwrap();
    assert!(mapped >= LOCKED.len() as u64, "{mapped} plans mapped of {}", LOCKED.len());
    (model, prompt, mapped)
}

/// What a run of `passes` decode passes left: the tokens, each pass's logits, and the routes the GPU assisted in each pass.
struct Run {
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
    /// (the level a pass ran at, its routes assisted, on the GPU and on the CPU, and the seconds of its layers)
    trace: Vec<(u32, [u64; 3], f64)>,
    detail: Json,
    /// The plans whose banks the GPU can read.
    mapped: u64,
}

/// A fresh model's run: `script` gives the level of each decode pass (the last goes on).
fn run(root: &std::ffi::OsStr, script: &[u32], passes: usize) -> Run {
    let (mut model, prompt, mapped) = open(root);
    model.logits_capture = Some(Vec::new());
    model.assist_script(script).unwrap();
    let mut tokens = Vec::new();
    // (a token a pass: the first comes of the prompt's logits)
    model.generate(&prompt, passes + 1, &straydog_engine::sampling::Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false,
                   &mut |token| tokens.push(token)).unwrap();
    let logits = model.logits_capture.take().unwrap();
    Run { tokens, logits, trace: model.assist_trace(), detail: model.last_metrics["partial_detail"].clone(), mapped }
}

fn relative_difference(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        diff += f64::from(x - y).powi(2);
        norm += f64::from(*y).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}

/// The level asked for is the level every pass runs at: the routes a pass assists are none at 0 and some at more, from the first pass after
/// a change on (a graph captured at another level, replayed, would assist as that level did), and their count rises with the level.
#[test]
fn every_pass_runs_at_its_level() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // blocks of four passes: 2, 0, 1, 2, 0, 2
    let script: Vec<u32> = [2, 0, 1, 2, 0, 2].iter().flat_map(|level| [*level; 4]).collect();
    let run = run(&root, &script, script.len());
    assert_eq!(run.tokens.len(), script.len() + 1, "the model went on for the passes asked");
    assert_eq!(run.trace.len(), script.len(), "a pass is a trace entry");
    assert_eq!(run.trace.iter().map(|(level, _, _)| *level).collect::<Vec<_>>(), script);
    let counts: Vec<(u32, u64)> = run.trace.iter().map(|(level, routed, _)| (*level, routed[0])).collect();
    eprintln!("(level, assisted, GPU, CPU, ms of the layers) by pass: {:?}", run.trace.iter().map(|(level, routed, seconds)| (level, routed, (seconds * 1e3 * 10.0).round() / 10.0)).collect::<Vec<_>>());
    // (the first pass after a change captures the layers' graphs again: what that costs a pass, for the log of a change)
    let changed: Vec<f64> = run.trace.windows(2).filter(|pair| pair[0].0 != pair[1].0).map(|pair| pair[1].2).collect();
    let steady: Vec<f64> = run.trace.windows(2).filter(|pair| pair[0].0 == pair[1].0).map(|pair| pair[1].2).collect();
    eprintln!("a pass after a change of level: {:.1} ms on average over {}, any other: {:.1} ms over {}",
              1e3 * changed.iter().sum::<f64>() / changed.len() as f64, changed.len(), 1e3 * steady.iter().sum::<f64>() / steady.len() as f64, steady.len());
    // every pass's routes (10 in each hybrid layer) went somewhere: whatever the GPU did not take, the CPU ran
    let routes = run.trace[0].1.iter().sum::<u64>();
    assert!((470..=480).contains(&routes), "{routes} routes in the first pass");
    for (level, routed, _) in &run.trace {
        assert_eq!(routed.iter().sum::<u64>(), routes, "level {level}: {routed:?}");
    }
    for block in counts.chunks(4) {
        let level = block[0].0;
        let total: u64 = block.iter().map(|(_, count)| *count).sum();
        if level == 0 {
            // (the pass the level changed at too: its graphs were dropped and captured at 0)
            assert_eq!(total, 0, "a block at level 0 assisted {block:?}");
        } else {
            assert!(total > 0, "a block at level {level} assisted nothing: {block:?}");
            // (a layer takes at most `level` routes, over the layers whose banks are mapped: the four the test locks, or every layer
            // where the start has page-locked them all)
            assert!(block.iter().all(|(_, count)| *count <= level as u64 * run.mapped), "{block:?}");
        }
    }
    let by_level = |wanted: u32| counts.iter().filter(|(level, _)| *level == wanted).map(|(_, count)| *count).sum::<u64>() as f64
        / counts.iter().filter(|(level, _)| *level == wanted).count() as f64;
    eprintln!("assisted a pass: 0 {:.2}, 1 {:.2}, 2 {:.2}", by_level(0), by_level(1), by_level(2));
    assert!(by_level(2) > by_level(1), "level 2 must assist more than level 1");
    // the level in force at the end is the request line's
    assert_eq!(run.detail["assist"], 2, "{}", run.detail);
}

/// A model that changes level between passes gives each pass the results of the model at that level: close to the pass of a model that
/// ran at that level throughout, and as close as two runs of the same levels are to each other. Not equal bit for bit in either case: the
/// level decides which device computes an expert, and which experts the slots hold at a moment depends on timing (docs/DESIGN.md, partial
/// residency).
#[test]
fn passes_after_a_change_have_the_results_of_their_level() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let script: Vec<u32> = [2, 0, 1, 2, 0, 1].iter().flat_map(|level| [*level; 3]).collect();
    let passes = script.len();
    // What two runs that never change level differ by. Which experts the slots hold at a moment, and so which device computes an expert,
    // depends on timing; how far that moves the logits depends on how many routes the GPU takes from the host: a few layers' where the test
    // locks four, every layer's where the start page-locks all of them. The tolerance follows it, and is never below 1e-4.
    let held = run(&root, &[2], passes);
    let held_again = run(&root, &[2], passes);
    let (compared, noise) = compare(&held, &held_again, &vec![2; passes], &[2]);
    let tolerance = (4.0 * noise).max(1e-4);
    eprintln!("level 2 twice: {compared} passes compared, the worst relative difference {noise:.2e}; the tolerance {tolerance:.2e}");
    assert!(compared >= 6, "too few passes before the tokens parted ({compared})");
    let switched = run(&root, &script, passes);
    // the same levels again, in a fresh model
    let again = run(&root, &script, passes);
    let (compared, worst) = compare(&switched, &again, &script, &[0, 1, 2]);
    eprintln!("the same levels again: {compared} passes compared, the worst relative difference {worst:.2e}");
    assert!(compared >= 6, "too few passes before the tokens parted ({compared})");
    assert!(worst < tolerance, "{worst} against {tolerance}");
    // the references: the level of each pass throughout
    for level in 0..=2u32 {
        let fixed = if level == 2 { None } else { Some(run(&root, &[level], passes)) };
        let (compared, worst) = compare(&switched, fixed.as_ref().unwrap_or(&held), &script, &[level]);
        eprintln!("level {level}: {compared} passes compared with a run at that level throughout, the worst relative difference {worst:.2e}");
        assert!(compared >= 3, "level {level}: too few passes before the tokens parted ({compared})");
        assert!(worst < tolerance, "level {level}: {worst} against {tolerance}");
    }
}

/// The passes of `script` whose level is among `levels`, up to where the two runs' tokens part: the worst relative difference of their
/// logits, and how many were compared.
fn compare(one: &Run, other: &Run, script: &[u32], levels: &[u32]) -> (usize, f64) {
    let (mut compared, mut worst) = (0, 0f64);
    for pass in 0..script.len() {
        // (the logits of pass i come after the prompt's, and follow the tokens up to the pass's own)
        if one.tokens.get(..=pass) != other.tokens.get(..=pass) {
            break;
        }
        if levels.contains(&script[pass]) {
            worst = worst.max(relative_difference(&one.logits[pass + 1], &other.logits[pass + 1]));
            compared += 1;
        }
    }
    (compared, worst)
}
