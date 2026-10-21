//! DeepSeek-V4.1's passes of a few tokens taken back on real GPUs ([`Engine::rewind`]): after a
//! pass of up to four tokens taken back to its first few, the sequence state and the logits of
//! every later pass are those of a sequence fed only the tokens kept, one at a time, bit for bit.
//! The prompt's length is odd, so a ratio-2 group of compressed rows straddles the pass's start
//! (a rewind leaves either the row waiting before the pass or one of the pass's waiting), and
//! every pass runs through the Engram layers, whose n-grams read the tokens kept.
//!
//! cargo test -p straydog-engine --features hip --test deepseek41_verify, with STRAYDOG_DEEPSEEK
//! (the UD-IQ2_XXS directory); under the GPU lock. On one GPU and split across two.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;
#[path = "../examples/common/files.rs"]
mod files;

use std::path::Path;

use files::Files;
use straydog_engine::arch::deepseek41::engine::{Engine, Settings, VERIFY_ROWS};
use straydog_engine::arch::deepseek41::Hparams;
use straydog_engine::text::Text;

/// The tests load the model on the same GPUs: one at a time.
static GPUS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The prompt's tokens: odd, so a ratio-2 kv source has a row waiting when the drafts start.
const PROMPT: usize = 29;

type State = Vec<(String, Vec<f32>)>;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn same_state(ours: &State, expected: &State, what: &str) {
    assert_eq!(ours.len(), expected.len(), "{what}: state parts");
    for ((name, values), (expected_name, expected_values)) in ours.iter().zip(expected) {
        assert_eq!(name, expected_name, "{what}");
        assert!(bits(values) == bits(expected_values), "{what}: {name} differs");
    }
}

fn same_logits(ours: &[f32], expected: &[f32], what: &str) {
    assert_eq!(ours.len(), expected.len(), "{what}: logits");
    let differ = ours.iter().zip(expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(differ, 0, "{what}: {differ} logits differ");
}

/// The best token of `logits` but `not`.
fn best(logits: &[f32], not: Option<u32>) -> u32 {
    logits.iter().enumerate().filter(|&(token, _)| Some(token as u32) != not).max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}

fn take_back(devices: &[usize]) {
    let Some(model) = common::model_env("STRAYDOG_DEEPSEEK", "the DeepSeek-V4.1-Flash UD-IQ2_XXS directory") else { return };
    let _gpus = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let files = Files::open(Path::new(&model)).unwrap();
    let hp = Hparams::from_header(&files.header).unwrap();
    assert!(hp.layers.iter().any(|plan| plan.engram.is_some()), "no Engram layer");
    assert!(hp.layers.iter().enumerate().any(|(layer, plan)| plan.compressed.is_some_and(|c| c.kv_source == layer && c.ratio == 2)),
            "no ratio-2 kv source");
    let text = Text::from_gguf(&files.header).unwrap();
    let mut prompt = text.encode("The expert stores of a mixture of experts keep the hottest experts in VRAM, the next in page-locked \
                                  host memory and the rest on the drive; a decode step reads").unwrap();
    assert!(prompt.len() >= PROMPT);
    prompt.truncate(PROMPT);
    // (slots enough for a pass's experts in one wave, as a served placement has)
    let settings = Settings {
        context: 1024, rows: 64, batch: 64, expert_slots: 1024, host_expert_bytes: (4 << 30) / devices.len() as u64,
        scratch_bytes: 512 << 20, layers: 0..hp.layers.len(),
    };
    let mut engine = Engine::new(hp.clone(), &files, devices, 31 << 30, settings).unwrap();
    assert!(engine.rewind(0).is_err(), "a rewind before any pass");

    // The drafts (greedy), fed one at a time: the logits after each and the states before each.
    let after_prompt = engine.forward(&prompt).unwrap();
    let mut states = vec![engine.state().unwrap()];
    let (mut drafts, mut logits): (Vec<u32>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
    for at in 0..VERIFY_ROWS {
        let token = best(logits.last().unwrap_or(&after_prompt), None);
        drafts.push(token);
        logits.push(engine.forward(&[token]).unwrap());
        states.push(engine.state().unwrap());
        assert_eq!(engine.position(), PROMPT + at + 1);
    }
    // Per number kept: the token fed instead of the next draft (the best but that draft; after all
    // four, the best), and the logits and state after it.
    let mut instead = Vec::new();
    for keep in 0..=VERIFY_ROWS {
        let before = if keep == 0 { &after_prompt } else { &logits[keep - 1] };
        let token = best(before, drafts.get(keep).copied());
        engine.reset();
        engine.forward(&prompt).unwrap();
        for &draft in &drafts[..keep] {
            engine.forward(&[draft]).unwrap();
        }
        instead.push((token, engine.forward(&[token]).unwrap(), engine.state().unwrap()));
    }

    // One pass over the four drafts, taken back to each number kept, then the token instead.
    for keep in 0..=VERIFY_ROWS {
        engine.reset();
        engine.forward(&prompt).unwrap();
        let last = engine.forward(&drafts).unwrap();
        same_logits(&last, &logits[VERIFY_ROWS - 1], "a pass of the drafts");
        engine.rewind(keep).unwrap();
        assert_eq!(engine.position(), PROMPT + keep);
        same_state(&engine.state().unwrap(), &states[keep], &format!("kept {keep}"));
        assert!(engine.rewind(keep).is_err(), "a second rewind of a pass");
        let (token, expected, state) = &instead[keep];
        same_logits(&engine.forward(&[*token]).unwrap(), expected, &format!("kept {keep}, then another token"));
        same_state(&engine.state().unwrap(), state, &format!("kept {keep}, then another token"));
    }

    // Passes from both parities of the waiting row: one taken back to its first token (then none
    // waits), the next three from there taken back to their first (one waits again: the pass's
    // first row), and a pass of one token taken back whole.
    engine.reset();
    engine.forward(&prompt).unwrap();
    engine.forward(&drafts).unwrap();
    engine.rewind(1).unwrap();
    same_state(&engine.state().unwrap(), &states[1], "kept 1 of 4");
    engine.forward(&drafts[1..]).unwrap();
    engine.rewind(1).unwrap();
    same_state(&engine.state().unwrap(), &states[2], "then 1 of 3");
    engine.forward(&drafts[2..3]).unwrap();
    assert!(engine.rewind(2).is_err(), "keeping more than a pass took");
    engine.rewind(0).unwrap();
    same_state(&engine.state().unwrap(), &states[2], "then none of 1");
    let (token, expected, state) = &instead[2];
    same_logits(&engine.forward(&[*token]).unwrap(), expected, "then another token");
    same_state(&engine.state().unwrap(), state, "then another token");
    engine.settle().unwrap();
}

#[test]
fn a_pass_taken_back_on_one_gpu_leaves_the_tokens_kept() {
    take_back(&[0]);
}

#[test]
fn a_pass_taken_back_across_two_gpus_leaves_the_tokens_kept() {
    take_back(&[0, 1]);
}
