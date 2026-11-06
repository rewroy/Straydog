//! A system block's state handed to a fresh model (`Model::export_prefix`, `Model::import_prefix`): the model takes the next request from the
//! end of the block and its first token's logits are those of a model that read everything.
//!
//! cargo test --release --features cuda -p straydog-engine --test prefix_gpu, with STRAYDOG_MODEL naming the UD-IQ1_S bundle (and
//! STRAYDOG_DRAFTER its MTP drafter's directory, for the run that carries the MTP layer's state too); under the GPU lock. The model is placed as a
//! 6 GiB card places it with the K/V in host memory: partially resident, as `straydog setup` leaves a small card.
//!
//! The comparison: the fresh model that was handed the state (B) against one that read the whole prompt (C). Partial residency's runs differ
//! from one another (which experts the slots hold at a moment depends on timing: docs/DESIGN.md, partial residency), so the tolerance is four
//! times what two fresh models that read the whole prompt (C and D) differ by, and never below 1e-4.
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

const IM_END: u32 = 248046;

/// A model of the bundle, placed in 5.5 GiB with the K/V in host memory, and the prompt: a system block of over 2100 tokens, then a short user turn.
fn open(root: &std::ffi::OsStr, drafter: Option<&std::ffi::OsStr>) -> (Model, Vec<u32>) {
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
    let drafter = drafter.map(PathBuf::from);
    let weights = Weights::open(&PathBuf::from(root), bundle, drafter.as_deref(), straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let text = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap();
    let mut system = String::new();
    let mut line = 0;
    let prompt = loop {
        system.push_str(&format!("Rule {line}: when asked about topic {} give {} short facts and name the source number {}.\n", line * 7 % 13, line % 9 + 3, line * 5 % 17));
        line += 1;
        let prompt = text.encode(&format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n")).unwrap();
        if prompt.iter().position(|token| *token == IM_END).is_some_and(|end| end >= 2100) {
            break prompt;
        }
    };
    // (64K of context: the K/V of that is over a quarter of the budget, so it stays in host memory, and it starts at 32K positions)
    let settings = Settings { context: 65536, state_budget_bytes: 8 << 30, prefill_batch_size: 256, speculative: if drafter.is_some() { 3 } else { 0 },
                              cpu_threads: 0, row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial, "the model must be partially resident at this budget");
    assert!(model.host_kv, "its K/V must be in host memory");
    // (the prompt's experts all stream: the fresh models compared begin alike; prompt_cpu_gpu checks the CPU's share of a prompt)
    model.prompt_share(false).unwrap();
    (model, prompt)
}

/// The root a prompt's first block makes: the end of its first `<|im_end|>`, rounded down to four.
fn root_of(prompt: &[u32]) -> usize {
    (prompt.iter().position(|token| *token == IM_END).unwrap() + 1) / 4 * 4
}

/// The logits of the first token after `prompt`, read by `model`, and what it said of the request.
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

/// Every way a state can differ from the layout of the model it is offered to, which must refuse it with nothing changed: a byte more or less,
/// another version, a count of tokens that does not fit the sections, a section of another size (the total kept), a section of another kind.
fn refusals(model: &mut Model, state: &[u8]) {
    let (tokens, sections) = {
        let parsed = prefix::parse(state).unwrap();
        (parsed.tokens, parsed.sections.len())
    };
    let section = |index: usize| 24 + 16 * index;
    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
    cases.push(("a byte short", state[..state.len() - 1].to_vec()));
    cases.push(("a byte over", [state, &[0u8]].concat()));
    cases.push(("nothing", Vec::new()));
    let mut other = state.to_vec();
    other[8] ^= 1;
    cases.push(("another version", other));
    let mut other = state.to_vec();
    other[12..16].copy_from_slice(&(tokens as u32 + 4).to_le_bytes());
    cases.push(("more tokens than its sections hold", other));
    let mut other = state.to_vec();
    other[12..16].copy_from_slice(&(tokens as u32 + 1).to_le_bytes());
    cases.push(("a count of tokens that is not a multiple of four", other));
    // (four bytes of one section moved to another: the total is the same, the sizes are not the layout's)
    let (low, high) = (section(sections - 2) + 8, section(sections - 1) + 8);
    let mut other = state.to_vec();
    let (a, b) = (u64::from_le_bytes(other[low..low + 8].try_into().unwrap()), u64::from_le_bytes(other[high..high + 8].try_into().unwrap()));
    other[low..low + 8].copy_from_slice(&(a + 4).to_le_bytes());
    other[high..high + 8].copy_from_slice(&(b - 4).to_le_bytes());
    cases.push(("a section of another size", other));
    let mut other = state.to_vec();
    other[section(3)] ^= 1;
    cases.push(("a section of another kind", other));
    let mut other = state.to_vec();
    other[section(1) + 8..section(1) + 16].copy_from_slice(&0u64.to_le_bytes());
    cases.push(("a hidden section emptied, the total no longer adding up", other));
    for (what, bytes) in cases {
        let refused = model.import_prefix(&bytes);
        assert!(refused.is_err(), "{what}: the model took it");
        assert!(model.parked_coverage().is_none(), "{what}: the model changed");
    }
}

/// Model A reads a prompt and exports its root; a fresh model B takes it and reads the same prompt: it reuses the root's tokens, reads only what
/// follows, and its first token's logits are within the tolerance of a fresh model's that read everything.
fn handed_over(drafter: Option<&std::ffi::OsStr>, root_dir: &std::ffi::OsStr) {
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // A: reads the prompt, and has the state of its root to give
    let (mut a, prompt) = open(root_dir, drafter);
    let root = root_of(&prompt);
    assert!(root >= 2048 && root < prompt.len(), "{root} of {} tokens", prompt.len());
    assert!(a.export_prefix().unwrap().is_none(), "a model that has read nothing has no state to give");
    first_logits(&mut a, &prompt);
    assert_eq!(a.last_metrics["root_checkpoint"], root, "the request says the root it made");
    assert_eq!(a.last_metrics["root_digest"], prefix::tokens_digest(&prompt[..root]));
    let started = std::time::Instant::now();
    let exported = a.export_prefix().unwrap().expect("a root the held sequence still begins with");
    eprintln!("exported {} tokens, {} MB, in {:.2} s (layout {})", exported.tokens, exported.bytes.len() >> 20, started.elapsed().as_secs_f64(), a.prefix_layout());
    assert_eq!(exported.tokens, root);
    assert_eq!(exported.digest, prefix::tokens_digest(&prompt[..root]));
    let layout = a.prefix_layout();
    // a request with other tokens after the root leaves the root as it was: the state is the root's, not the sequence's
    let mut other = prompt[..root].to_vec();
    other.extend(text_tokens(&prompt));
    first_logits(&mut a, &other);
    let again = a.export_prefix().unwrap().expect("the root is still held");
    assert_eq!(again.bytes.len(), exported.bytes.len());
    assert!(again.bytes[..] == exported.bytes[..], "the state of the root changed with the sequence that went on from it");
    drop(a);

    // B: a fresh model that is handed the state
    let (mut b, _) = open(root_dir, drafter);
    assert_eq!(b.prefix_layout(), layout, "two models of one placement have one layout");
    refusals(&mut b, &exported.bytes);
    let started = std::time::Instant::now();
    assert_eq!(b.import_prefix(&exported.bytes).unwrap(), root);
    eprintln!("imported in {:.2} s", started.elapsed().as_secs_f64());
    assert_eq!(b.parked_coverage(), Some((Some((0, root)), root)));
    let handed = first_logits(&mut b, &prompt);
    assert_eq!(b.last_metrics["cached_tokens"], root, "{:?}", b.last_metrics);
    assert_eq!(b.last_metrics["resumed_from"], "prefix");
    assert_eq!(b.last_metrics["prefill_tokens"], prompt.len() - root);
    assert!(b.parked_coverage().is_none(), "the state is the held sequence now");
    // (and its root is a checkpoint again, as it is after any request that made one)
    let kept = b.export_prefix().unwrap().expect("the root is saved again after a restore");
    assert_eq!((kept.tokens, kept.digest.as_str()), (root, exported.digest.as_str()));
    drop(b);

    // C and D: fresh models that read everything
    let (mut c, _) = open(root_dir, drafter);
    let whole = first_logits(&mut c, &prompt);
    assert_eq!(c.last_metrics["cached_tokens"], 0);
    drop(c);
    let (mut d, _) = open(root_dir, drafter);
    let whole_again = first_logits(&mut d, &prompt);
    drop(d);
    let noise = relative_difference(&whole, &whole_again);
    let tolerance = (4.0 * noise).max(1e-4);
    let worst = relative_difference(&handed, &whole);
    eprintln!("two fresh models differ by {noise:.2e}; the one handed the state differs from a fresh one by {worst:.2e}; the tolerance {tolerance:.2e}");
    assert!(worst <= tolerance, "{worst} against {tolerance}");
}

/// The tokens of another short user turn, after the system block.
fn text_tokens(prompt: &[u32]) -> Vec<u32> {
    // (the tail of the prompt, ended by its generation prompt, with its last user token changed: a different turn)
    let mut tail = prompt[root_of(prompt)..].to_vec();
    let at = tail.len().saturating_sub(8);
    tail[at] = tail[at].wrapping_add(1) % 1000 + 1000;
    tail
}

#[test]
fn a_fresh_model_takes_a_system_blocks_state() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    handed_over(None, &root);
}

#[test]
fn a_fresh_model_takes_the_state_with_the_mtp_layers_too() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let Some(drafter) = common::model_env("STRAYDOG_DRAFTER", "the MTP drafter's directory") else { return };
    handed_over(Some(&drafter), &root);
}
