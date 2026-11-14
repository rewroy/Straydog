//! A partially resident model's drafts a pass (`crate::width`): a greedy decode gives the same tokens with any number of them, and the number
//! in force follows what the passes cost when the costs are scripted (`Model::script_pass_seconds` stands in for the clock, as
//! `Model::assist_script` does for the assist tuner's rounds).
//!
//! cargo test --release --features cuda -p straydog-engine --test drafts_gpu, with STRAYDOG_MODEL naming the UD-IQ1_S bundle and
//! STRAYDOG_DRAFTER its MTP drafter's directory; under the GPU lock. The model is placed as a 6 GiB card places it with the K/V in host
//! memory: partially resident, 5.5 GiB.
//!
//! Partial residency's runs differ from one another (which experts the slots hold at a moment decides which device computes one, and the
//! sums are in another order: docs/DESIGN.md, partial residency), so tokens are compared as far as the runs agree, which must be far
//! (`AGREE`).
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

/// The tests load the model on the same GPU: one at a time.
static GPUS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The model with its drafter (up to three drafts), placed in 5.5 GiB with the K/V in host memory, and the prompt it continues.
fn open(root: &std::ffi::OsStr, drafter: &std::ffi::OsStr) -> (Model, Vec<u32>) {
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
    let drafter = PathBuf::from(drafter);
    let weights = Weights::open(&PathBuf::from(root), bundle, Some(&drafter), straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let prompt = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap()
        .encode("<|im_start|>user\nExplain how a hash map gives average constant time lookups, and when that breaks down.<|im_end|>\n<|im_start|>assistant\n")
        .unwrap();
    // (128K of context: Q8 K/V stays in host memory at this budget and starts at 32K positions)
    let settings = Settings { context: 131072, state_budget_bytes: 8 << 30, prefill_batch_size: 256, speculative: 3, cpu_threads: 0, row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial, "the model must be partially resident at this budget");
    assert!(model.host_kv, "its K/V must be in host memory");
    assert_eq!(model.speculative, 3, "the most drafts a pass takes is the settings'");
    // (the prompt's experts all stream: the runs compared begin alike)
    model.prompt_share(false).unwrap();
    (model, prompt)
}

struct Run {
    tokens: Vec<u32>,
    drafted: u64,
    accepted: u64,
    passes: u64,
}

/// A request from the start of the prompt, `new` tokens, greedy: the tokens, and the drafts of the request.
fn decode(model: &mut Model, prompt: &[u32], new: usize) -> Run {
    model.reset().unwrap();
    let mut tokens = Vec::new();
    model.generate(prompt, new, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |token| tokens.push(token)).unwrap();
    let count = |key: &str| model.last_metrics.get(key).and_then(|value| value.as_u64()).unwrap_or(0);
    Run { tokens, drafted: count("drafted_tokens"), accepted: count("accepted_tokens"), passes: count("speculative_passes") }
}

/// How many leading tokens two runs share.
fn shared(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The leading tokens a reply must share with a reply of none: two runs that part do so at a near tie, which timing alone can break (two runs
/// of one width part at 39 tokens in one pair of them, at 118 in another), and a draft row gone wrong parts within a few tokens.
const AGREE: usize = 24;

/// Every number of drafts a pass takes (0 to 3) decodes the same tokens: the drafts a pass keeps are the tokens the target would have chosen.
#[test]
fn any_number_of_drafts_decodes_the_same_tokens() {
    let (Some(root), Some(drafter)) = (common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory"),
                                       common::model_env("STRAYDOG_DRAFTER", "the MTP drafter's directory")) else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (mut model, prompt) = open(&root, &drafter);
    const NEW: usize = 160;
    let mut runs = Vec::new();
    // (each width twice: what two runs of one width differ by is what timing alone does)
    for width in [0usize, 0, 1, 1, 2, 2, 3, 3] {
        model.pin_drafts(Some(width));
        let run = decode(&mut model, &prompt, NEW);
        eprintln!("{width} drafts: {} tokens in {} passes, {} drafted, {} accepted", run.tokens.len(), run.passes, run.drafted, run.accepted);
        assert_eq!(model.drafts(), width);
        assert_eq!(model.last_metrics["partial_detail"]["drafts"], width, "the request's detail says the drafts in force");
        if width == 0 {
            assert_eq!(run.drafted, 0, "no draft is offered at 0");
        } else {
            assert!(run.drafted > 0 && run.accepted > 0, "{width} drafts: {} drafted, {} accepted", run.drafted, run.accepted);
            // (no pass offers more than the width: the passes' drafts, each pass at most `width`)
            assert!(run.drafted <= width as u64 * run.passes, "{width} drafts: {} drafted in {} passes", run.drafted, run.passes);
        }
        assert_eq!(run.tokens.len(), NEW, "{width} drafts: the reply's length");
        runs.push((width, run));
    }
    for (width, run) in &runs {
        let agree: Vec<String> = runs.iter().map(|(_, other)| shared(&run.tokens, &other.tokens).to_string()).collect();
        eprintln!("{width} drafts agrees with 0 0 1 1 2 2 3 3 for {}", agree.join(" "));
        // (as far as it follows one of the two replies of none: the replies of none are the reference, and they differ)
        let agrees = shared(&run.tokens, &runs[0].1.tokens).max(shared(&run.tokens, &runs[1].1.tokens));
        assert!(agrees >= AGREE, "{width} drafts: the tokens part from both replies of none by {agrees}");
    }
}

/// Tokens a request of the scripted tests asks for: the chooser probes after its 256th pass and takes the windows of two probes of 25 passes
/// before it decides, a few hundred passes at one to three tokens each.
const SCRIPTED: usize = 1200;

/// The model's width follows the passes' scripted costs: a row that costs nearly nothing makes three drafts the choice from one, a row that costs what
/// a pass of one does makes it none; the model says so in its request's detail, and what it decoded is still as it would have been.
fn follow(root: &std::ffi::OsStr, drafter: &std::ffi::OsStr, cost: fn(usize) -> f64, wanted: usize) {
    let (mut model, prompt) = open(root, drafter);
    assert!((1..=2).contains(&model.drafts()), "a model starts with one draft (two on HIP where the slots hold enough), not {}", model.drafts());
    model.script_pass_seconds(Some(Box::new(cost)));
    let run = decode(&mut model, &prompt, SCRIPTED);
    eprintln!("scripted costs: {} tokens in {} passes, {} drafted, {} accepted; drafts now {}", run.tokens.len(), run.passes, run.drafted, run.accepted, model.drafts());
    assert_eq!(model.drafts(), wanted, "the drafts in force");
    assert_eq!(model.last_metrics["partial_detail"]["drafts"], wanted);
    // (the tokens: as far as the plain decode's go, which the run before was)
    model.script_pass_seconds(None);
    model.pin_drafts(Some(0));
    let plain = decode(&mut model, &prompt, SCRIPTED);
    assert!(shared(&plain.tokens, &run.tokens) >= AGREE, "the tokens part from a decode without drafts at {}", shared(&plain.tokens, &run.tokens));
}

#[test]
fn rows_that_cost_little_give_three_drafts() {
    let (Some(root), Some(drafter)) = (common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory"),
                                       common::model_env("STRAYDOG_DRAFTER", "the MTP drafter's directory")) else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // (a pass of 40 ms and one more millisecond a row: a GPU-bound machine)
    follow(&root, &drafter, |rows| 0.040 + 0.001 * rows as f64, 3);
}

#[test]
fn rows_that_cost_a_pass_each_give_none() {
    let (Some(root), Some(drafter)) = (common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory"),
                                       common::model_env("STRAYDOG_DRAFTER", "the MTP drafter's directory")) else { return };
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // (each row doubles, triples, quadruples the pass: the CPU-bound machine at its worst)
    follow(&root, &drafter, |rows| 0.020 * rows as f64, 0);
}
