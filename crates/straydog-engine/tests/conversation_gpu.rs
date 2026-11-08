//! A conversation's state handed to a fresh model (`Model::export_conversation`, `Model::import_conversation`): the model takes the next request of the
//! conversation from the place the last reply began, and its first token's logits are those of the model that never stopped.
//!
//! cargo test --release --features cuda -p straydog-engine --test conversation_gpu -- --nocapture --test-threads=1, with STRAYDOG_MODEL naming the
//! UD-IQ1_S bundle (and STRAYDOG_DRAFTER its MTP drafter's directory, for the run that carries the MTP layer's state too); under the GPU lock. The model
//! is placed as a 6 GiB card places it with the K/V in host memory: partially resident, as `straydog setup` leaves a small card.
//! STRAYDOG_CONVERSATION_TOKENS sets the conversation's length in tokens (default 6000; 36000 is past the 32768 positions the K/V starts with, so the
//! import grows it), and STRAYDOG_PROMPT_CPU=0 makes a prompt batch stream every expert beyond the slots, with which the two models compared agree to the bit.
//!
//! The comparison: model A reads the conversation, answers, and is asked its next request (the one that goes on from the conversation): it resumes
//! from the checkpoint its last reply began at. Model B is a fresh one handed the state A exported before that request: it must take the same request
//! from the same place, and its logits must be A's. Partial residency's runs differ from one another (which experts the slots hold at a moment depends on
//! timing: docs/DESIGN.md, partial residency), so the tolerance, where they differ, is four times what two fresh models that read the whole request
//! differ by (C and D), and never below 1e-4. (A model that read the whole request is not the comparison: a request that goes on from a checkpoint reads
//! its tokens in other batches than one that begins at the start, which moves the logits of this quantization by a few percent, restart or none.)
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::prefix;
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

/// The tests load a model on the same GPU: one at a time.
static GPUS: std::sync::Mutex<()> = std::sync::Mutex::new(());

const IM_START: u32 = 248045;

/// A model of the bundle, placed in 5.5 GiB with the K/V in host memory, and its tokenizer.
fn open(root: &std::ffi::OsStr, drafter: Option<&std::ffi::OsStr>) -> (Model, Text) {
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
    let drafter = drafter.map(PathBuf::from);
    let weights = Weights::open(&PathBuf::from(root), bundle, drafter.as_deref(), straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let text = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap();
    // (64K of context: the K/V of that is over a quarter of the budget, so it stays in host memory, and it starts at 32K positions)
    let settings = Settings { context: 65536, state_budget_bytes: 8 << 30, prefill_batch_size: 256, speculative: if drafter.is_some() { 3 } else { 0 },
                              cpu_threads: 0, row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial, "the model must be partially resident at this budget");
    assert!(model.host_kv, "its K/V must be in host memory");
    // (the prompt's experts all stream: the fresh models compared begin alike)
    model.prompt_share(false).unwrap();
    (model, text)
}

/// The conversation, `tokens` long at least, whose last message is the user's: a system block, turns of a user and an assistant, and a last user message
/// with the generation prompt after it. And the request that goes on from it: the same up to the generation prompt, then a reply as the template
/// writes it back (not the tokens the model wrote), a new user message and its generation prompt.
fn conversation(text: &Text, tokens: usize) -> (Vec<u32>, Vec<u32>) {
    let mut chat = String::from("<|im_start|>system\nYou answer questions about rules and sources, briefly.<|im_end|>\n");
    let mut line = 0;
    loop {
        chat.push_str(&format!("<|im_start|>user\nTurn {line}: what do the rules say about topic {} when {} facts are asked, citing source {}?<|im_end|>\n\
                                <|im_start|>assistant\nThe rules for topic {} give {} facts, from source {}.<|im_end|>\n",
                               line * 7 % 13, line % 9 + 3, line * 5 % 17, line * 7 % 13, line % 9 + 3, line * 5 % 17));
        line += 1;
        if line % 50 == 0 && text.encode(&chat).unwrap().len() >= tokens {
            break;
        }
    }
    let first = format!("{chat}<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n");
    let second = format!("{chat}<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nParis.<|im_end|>\n\
                          <|im_start|>user\nAnd of Japan?<|im_end|>\n<|im_start|>assistant\n");
    (text.encode(&first).unwrap(), text.encode(&second).unwrap())
}

/// Where the last reply of `prompt` began: its last `<|im_start|>`, rounded down to four.
fn reply_place(prompt: &[u32]) -> usize {
    prompt.iter().rposition(|token| *token == IM_START).unwrap() / 4 * 4
}

/// The logits of the first token after `prompt`, read by `model`.
fn first_logits(model: &mut Model, prompt: &[u32]) -> Vec<f32> {
    model.logits_capture = Some(Vec::new());
    model.generate(prompt, 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true, &mut |_| {}).unwrap();
    model.logits_capture.take().unwrap().remove(0)
}

fn relative_difference(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        diff += f64::from(x - y).powi(2);
        norm += f64::from(*y).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}

/// Every way a state can differ from the layout of the model it is offered to, which must refuse it with nothing changed.
fn refusals(model: &mut Model, state: &[u8], positions: usize) {
    let tokens = prefix::parse(state).unwrap().tokens;
    let mut cases: Vec<(&str, Vec<u8>)> = vec![("a byte short", state[..state.len() - 1].to_vec()), ("a byte over", [state, &[0u8]].concat()), ("nothing", Vec::new())];
    let mut other = state.to_vec();
    other[8] ^= 1;
    cases.push(("another version", other));
    let mut other = state.to_vec();
    other[12..16].copy_from_slice(&(tokens as u32 + 4).to_le_bytes());
    cases.push(("more tokens than its sections hold", other));
    let mut other = state.to_vec();
    other[12..16].copy_from_slice(&(tokens as u32 + 1).to_le_bytes());
    cases.push(("a count of tokens that is not a multiple of four", other));
    for (what, bytes) in cases {
        assert!(model.import_conversation(&bytes, positions).is_err(), "{what}: the model took it");
        assert!(model.parked_coverage().is_none(), "{what}: the model changed");
    }
}

fn kept_conversation(drafter: Option<&std::ffi::OsStr>, root_dir: &std::ffi::OsStr) {
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let length: usize = std::env::var("STRAYDOG_CONVERSATION_TOKENS").ok().and_then(|count| count.parse().ok()).unwrap_or(6000);
    // A: reads the conversation, answers, and has its state to give
    let (mut a, text) = open(root_dir, drafter);
    let (first, second) = conversation(&text, length);
    let place = reply_place(&first);
    assert!(place >= 1024 && second[..place] == first[..place], "the next request goes on from the place the reply began");
    assert!(a.export_conversation().unwrap().is_none() && a.conversation_size().is_none(), "a model that has read nothing keeps nothing");
    eprintln!("the conversation: {} tokens, its last reply began at {place}; the next request is {} tokens", first.len(), second.len());
    let started = std::time::Instant::now();
    a.generate(&first, 8, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true, &mut |_| {}).unwrap();
    eprintln!("A read the conversation in {:.1} s", started.elapsed().as_secs_f64());
    let (size_tokens, size) = a.conversation_size().expect("a conversation with a place its reply began at");
    assert_eq!(size_tokens, place);
    let started = std::time::Instant::now();
    let kept = a.export_conversation().unwrap().expect("a state to give");
    eprintln!("exported {} tokens, {} MB, in {:.2} s", kept.tokens, kept.bytes.len() >> 20, started.elapsed().as_secs_f64());
    assert_eq!((kept.tokens, kept.digest.as_str()), (place, prefix::tokens_digest(&first[..place]).as_str()));
    assert!(kept.bytes.len() as u64 <= size, "{} bytes where {size} were said", kept.bytes.len());
    // (the state is the checkpoint's, not the sequence's: asking again gives the same bytes, the reply the sequence went on with being no part of it)
    let again = a.export_conversation().unwrap().unwrap();
    assert!(again.bytes == kept.bytes, "the state changed between two exports");
    // A's next request: from the checkpoint its last reply began at
    let continued = first_logits(&mut a, &second);
    assert_eq!(a.last_metrics["cached_tokens"], place, "{:?}", a.last_metrics);
    drop(a);

    // B: a fresh model that is handed the state, and takes the same request
    let (mut b, _) = open(root_dir, drafter);
    refusals(&mut b, &kept.bytes, second.len() + 1);
    assert_eq!(b.held_prefix(&second), 0);
    let started = std::time::Instant::now();
    assert_eq!(b.import_conversation(&kept.bytes, second.len() + 1).unwrap(), place);
    eprintln!("imported in {:.2} s (the K/V holds {} positions)", started.elapsed().as_secs_f64(), b.kv_positions());
    assert_eq!(b.parked_coverage(), Some((Some((0, place)), place)));
    assert_eq!(b.held_prefix(&second), place);
    let started = std::time::Instant::now();
    let handed = first_logits(&mut b, &second);
    eprintln!("B's request took {:.2} s", started.elapsed().as_secs_f64());
    assert_eq!(b.last_metrics["cached_tokens"], place, "{:?}", b.last_metrics);
    assert_eq!(b.last_metrics["resumed_from"], "kept");
    assert_eq!(b.last_metrics["prefill_tokens"], second.len() - place);
    assert!(b.parked_coverage().is_none(), "the state is the held sequence now");
    // (and the place its reply began is a checkpoint again, as after any request that made one: the next start can keep this conversation too)
    let again = b.export_conversation().unwrap().expect("the conversation is kept again after a restore");
    assert_eq!(again.tokens, reply_place(&second), "the new request's own checkpoint is the deepest");
    drop(b);

    // The same bits as the model that never stopped: nothing to calibrate. Otherwise (a partially resident model's slots differ between runs) the tolerance
    // is four times what two fresh models that read the whole request differ by (C and D), and never below 1e-4.
    let same_bits = handed.len() == continued.len() && handed.iter().zip(&continued).all(|(x, y)| x.to_bits() == y.to_bits());
    let worst = relative_difference(&handed, &continued);
    if same_bits {
        eprintln!("the logits of the first token after the restore are the same bits as the model that never stopped");
    } else {
        let (mut c, _) = open(root_dir, drafter);
        let whole = first_logits(&mut c, &second);
        assert_eq!(c.last_metrics["cached_tokens"], 0);
        drop(c);
        let (mut d, _) = open(root_dir, drafter);
        let whole_again = first_logits(&mut d, &second);
        drop(d);
        let noise = relative_difference(&whole, &whole_again);
        let tolerance = (4.0 * noise).max(1e-4);
        eprintln!("two fresh models differ by {noise:.2e}; the one handed the state differs from the one that never stopped by {worst:.2e}; the tolerance {tolerance:.2e}");
        assert!(worst <= tolerance, "{worst} against {tolerance}");
    }
    assert!(!std::env::var("STRAYDOG_PROMPT_CPU").is_ok_and(|value| value == "0") || same_bits,
            "with every expert beyond the slots streamed, the two agree to the bit: they differ by {worst:.2e}");
}

#[test]
fn a_fresh_model_takes_a_conversations_state() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    kept_conversation(None, &root);
}

#[test]
fn a_fresh_model_takes_the_conversation_with_the_mtp_layers_too() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let Some(drafter) = common::model_env("STRAYDOG_DRAFTER", "the MTP drafter's directory") else { return };
    kept_conversation(Some(&drafter), &root);
}
