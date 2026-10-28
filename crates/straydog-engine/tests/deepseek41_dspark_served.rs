//! DeepSeek-V4.1 served with its DSpark drafter: the tokens a greedy reply takes are those without it,
//! and the drafter's history of the target's features follows the sequence the engines hold through
//! verify passes and rewinds, resets and checkpoints, so that a draft never reads another sequence's.
//!
//! cargo test -p straydog-engine --features hip --test deepseek41_dspark_served, with STRAYDOG_DS41 (the
//! GSQ-RCO IQ2_XS bundle directory (STRAYDOG_DS41_QUANT=GSQ-RCO-IQ2_XXS for the IQ2_XXS one), its verity trees in .straydog there, DSpark's file in its dspark
//! directory or beside it); under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use sha2::{Digest, Sha256};
use straydog_engine::arch::deepseek41::serve::{place, Model};
use straydog_engine::arch::served::{Engines as _, Settings};
use straydog_engine::chat::ChatTemplate;
use straydog_engine::group::SplitMode;
use straydog_engine::pyjson;
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;

/// What a greedy request left: its tokens and the digest of its prompt's logits.
#[derive(Debug, PartialEq)]
struct Outcome {
    tokens: Vec<u32>,
    logits: String,
}

fn digest(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    format!("{:x}", Sha256::digest(&bytes))
}

/// A request with DSpark on or off: the model starts from nothing.
fn go(model: &mut Model, dspark: bool, prompt: &[u32], max_new: usize) -> (Outcome, u64, u64) {
    model.forget();
    go_on(model, dspark, prompt, max_new)
}

/// A request that continues what the model holds.
fn go_on(model: &mut Model, dspark: bool, prompt: &[u32], max_new: usize) -> (Outcome, u64, u64) {
    assert!(model.engines.draft_with_dspark(dspark), "the DSpark drafter is not loaded: its file is not beside the model");
    let mut tokens = Vec::new();
    model.generate(prompt, max_new, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, &mut |token| tokens.push(token)).unwrap();
    let count = |key: &str| model.last_metrics[key].as_u64().unwrap();
    let drafts = (count("engine_draft_tokens"), count("accepted_engine_drafts"));
    (Outcome { tokens, logits: digest(model.prompt_logits.as_deref().unwrap()) }, drafts.0, drafts.1)
}

#[test]
fn greedy_tokens_are_those_without_the_drafter_and_its_history_follows_the_sequence() {
    let Some(root) = common::model_env("STRAYDOG_DS41", "the DeepSeek-V4.1 GSQ-RCO IQ2_XS bundle directory") else { return };
    let root = PathBuf::from(root);
    let settings = Settings {
        devices: vec![0, 1], split_mode: SplitMode::Experts, vram_bytes: vec![28 << 30; 2], context: 16384, batch: 2048,
        host_expert_bytes: None, host_memory_bytes: straydog_engine::host::host_memory_bytes(),
        row_cache_bytes: straydog_engine::rows::DEFAULT_CACHE_BYTES, half_gemm: true,
    };
    let mut model = place(&root, &common::deepseek41_quantization(), &root.join(".straydog"), &settings, &|_, _, _| {}).unwrap();
    // (as the worker does after its first passes: the VRAM the cards have spare goes to the expert stores)
    model.generate(&[0, 1000, 2000, 3000], 2, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, &mut |_| {}).unwrap();
    model.forget();
    let slots = model.placement["expert_slots"].clone();
    model.lend_spare().unwrap();
    assert!(model.placement["expert_slots"][0].as_u64() > slots[0].as_u64(), "the spare VRAM made slots: {slots} before, {} after", model.placement["expert_slots"]);
    model.prompt_logits = Some(Vec::new());
    assert!(model.engines.draft_with_dspark(true), "the DSpark drafter is not loaded: its file is not beside the model");
    assert_eq!(model.placement["dspark"]["resident"], serde_json::json!([87, 50, 29]));
    let text = Text::from_gguf(model.header()).unwrap();
    let template = ChatTemplate::new(straydog_engine::arch::deepseek41::CHAT_TEMPLATE).unwrap();
    let render = |content: &str| -> Vec<u32> {
        let messages = format!("[{{\"role\": \"user\", \"content\": {}}}]", serde_json::to_string(content).unwrap());
        text.encode(&template.render(&pyjson::parse(&messages).unwrap(), &pyjson::parse("[]").unwrap(), false, "low").unwrap()).unwrap()
    };
    let document: String = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/DESIGN.md")).unwrap()
        .chars().take(12000).collect();

    // Prose, a reply that repeats its source (the lookup drafts too), and a long document read in two
    // prefill passes: the same tokens and prompt logits with the drafter as without it, and the drafter
    // drafted and had drafts accepted.
    let prompts = [
        render("Write four lines about the sea in winter."),
        render("Repeat this line three times, exactly, one under another, and nothing else: \
                the quick brown fox jumps over the lazy dog while the rain falls on the tin roof."),
        render(&format!("{document}\n\nName this project's main parts in one sentence.")),
    ];
    let (mut drafted, mut accepted) = (0, 0);
    for (index, prompt) in prompts.iter().enumerate() {
        let (without, none, _) = go(&mut model, false, prompt, 96);
        assert_eq!(none, 0, "prompt {index}: drafts with the drafter off");
        let (with, made, kept) = go(&mut model, true, prompt, 96);
        assert_eq!(with.tokens, without.tokens, "prompt {index}: tokens");
        assert_eq!(with.logits, without.logits, "prompt {index}: prompt logits");
        eprintln!("prompt {index} ({} tokens): DSpark drafted {made}, {kept} accepted; {} passes for {} tokens", prompt.len(),
                  model.last_metrics["decode_passes"], model.last_metrics["generated_tokens"]);
        (drafted, accepted) = (drafted + made, accepted + kept);
    }
    assert!(drafted > 0 && accepted > 0, "DSpark drafted {drafted} and had {accepted} accepted");

    // A request that continues the held sequence (live): the drafter's history carries over from
    // the request before. A reply cut at 24 tokens and continued to 72 leaves the tokens of one
    // reply of 72 without the drafter.
    let essay = render("Write an essay of at least four hundred words about the sea in winter.");
    let (whole, _, _) = go(&mut model, false, &essay, 72);
    assert_eq!(whole.tokens.len(), 72);
    let (first, _, _) = go(&mut model, true, &essay, 24);
    assert_eq!(first.tokens, whole.tokens[..24]);
    let mut next = essay.clone();
    next.extend_from_slice(&first.tokens);
    let (live, made, kept) = go_on(&mut model, true, &next, 48);
    assert_eq!(model.last_metrics["resumed_from"], "live");
    assert_eq!(live.tokens, whole.tokens[24..], "a live continuation");
    eprintln!("live continuation: {made} drafted, {kept} accepted");

    // The history through the engines' own passes, with the trait the session drives.
    let tokens = text.encode(&document).unwrap();
    assert!(tokens.len() > 420, "{}", tokens.len());
    let rows = model.engines.verify_rows();
    assert!(rows >= 4, "{rows}");
    let anchor = tokens[300];

    // a checkpoint keeps the history of its sequence: restored after another sequence, a draft at
    // its position is the draft it was (a draft that read the other sequence's features, or none,
    // would not be)
    model.forget();
    assert!(model.engines.draft_with_dspark(true));
    let engines = &mut model.engines;
    engines.forward(&tokens[..200]).unwrap();
    assert_eq!(engines.dspark_history(), Some(true));
    let before = engines.dspark_probe(anchor, 3).unwrap().unwrap();
    assert_eq!(engines.dspark_probe(anchor, 3).unwrap().unwrap(), before, "a draft is repeatable");
    assert_eq!(before.tokens.len(), before.confidences.len());
    engines.checkpoint(0, true).unwrap();
    for &token in &tokens[200..205] {
        engines.forward(&[token]).unwrap();
    }
    assert_eq!(engines.dspark_history(), Some(true));
    let later = engines.dspark_probe(anchor, 3).unwrap().unwrap();
    engines.reset();
    // (a sequence of 40 tokens: its history is its 40 positions, of the same rows as the other's)
    engines.forward(&tokens[300..340]).unwrap();
    assert_eq!(engines.dspark_history(), Some(true));
    let other = engines.dspark_probe(anchor, 3).unwrap().unwrap();
    engines.checkpoint(0, false).unwrap();
    assert_eq!(engines.dspark_history(), Some(true));
    assert_eq!(engines.dspark_probe(anchor, 3).unwrap().unwrap(), before, "a draft after a restore is the draft before the checkpoint");
    assert_ne!(other, before, "(the other sequence's draft differs)");
    assert_ne!(later, before, "(the sequence a few tokens on drafts differently)");
    // a reset forgets: a sequence that is not yet 128 positions long has its own positions alone
    engines.reset();
    engines.forward(&tokens[..20]).unwrap();
    assert_eq!(engines.dspark_history(), Some(true));
    engines.reset();
    assert_eq!(engines.dspark_history(), Some(true), "(a draft at position 0 reads nothing)");

    // The first rows of a sequence (and a few more) are attention sinks, their features past FP16's 65504, and a
    // prompt of 32 rows or more is read with FP16 products: the drafter's injection of such a prompt's features must
    // not cast them (infinite rows made every draft wrong for as long as one was in the window). The draft after a
    // prompt read in one pass is the draft after the same prompt read in passes of fewer rows.
    let sinks = render("Write a complete, well-commented Rust implementation of an LRU cache (a HashMap plus a doubly linked list \
                        kept in a Vec by index), with unit tests. Output only code.");
    assert!(sinks.len() > 32, "{}", sinks.len());
    let anchor = sinks[sinks.len() - 1];
    engines.reset();
    engines.forward(&sinks).unwrap();
    let whole = engines.dspark_probe(anchor, 3).unwrap().unwrap();
    engines.reset();
    for piece in sinks.chunks(16) {
        engines.forward(piece).unwrap();
    }
    let pieces = engines.dspark_probe(anchor, 3).unwrap().unwrap();
    eprintln!("a prompt of {} tokens, read whole: {:?} {:?}; in passes of 16: {:?} {:?}", sinks.len(), whole.tokens, whole.confidences,
              pieces.tokens, pieces.confidences);
    let agree = whole.tokens.iter().zip(&pieces.tokens).filter(|(a, b)| a == b).count();
    assert!(whole.tokens.first() == pieces.tokens.first() && agree >= 2, "{whole:?} against {pieces:?}");
    let anchor = tokens[300];

    // a verify pass taken back to part of it: the history has the rows kept, as when they were fed
    // one at a time, and not the rows taken back
    let probe_after = |engines: &mut straydog_engine::arch::deepseek41::serve::Engines, feed: &dyn Fn(&mut straydog_engine::arch::deepseek41::serve::Engines)| {
        engines.reset();
        engines.forward(&tokens[..200]).unwrap();
        feed(engines);
        assert_eq!(engines.dspark_history(), Some(true));
        engines.dspark_probe(anchor, 3).unwrap().unwrap()
    };
    let one_by_one = probe_after(engines, &|engines| {
        for &token in &tokens[200..202] {
            engines.forward(&[token]).unwrap();
        }
    });
    let taken_back = probe_after(engines, &|engines| {
        engines.verify(&tokens[200..204]).unwrap();
        engines.rewind(2).unwrap();
    });
    assert_eq!(taken_back, one_by_one, "a verify pass of four taken back to two");
    let wrong = tokens[330];
    let one_by_one = probe_after(engines, &|engines| {
        for &token in [tokens[200], tokens[201], wrong].iter() {
            engines.forward(&[token]).unwrap();
        }
    });
    let continued = probe_after(engines, &|engines| {
        engines.verify(&tokens[200..204]).unwrap();
        engines.rewind(2).unwrap();
        engines.forward(&[wrong]).unwrap();
    });
    assert_eq!(continued, one_by_one, "tokens fed after a rewind take the place of those taken back");
    let all_kept = probe_after(engines, &|engines| {
        engines.verify(&tokens[200..203]).unwrap();
        engines.rewind(3).unwrap();
    });
    let singles = probe_after(engines, &|engines| {
        for &token in &tokens[200..203] {
            engines.forward(&[token]).unwrap();
        }
    });
    assert_eq!(all_kept, singles, "a verify pass of three kept whole");
}
