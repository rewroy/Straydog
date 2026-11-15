//! DeepSeek-V4.1 served on real GPUs: a conversation switched away from comes back from its
//! checkpoint as if nothing had come between, and a reply goes on past a void pass.
//!
//! cargo test -p straydog-engine --features hip --test deepseek41_served, with STRAYDOG_DS41 (the
//! GSQ-RCO IQ2_XS bundle directory (STRAYDOG_DS41_QUANT=GSQ-RCO-IQ2_XXS for the IQ2_XXS one), its verity trees in .straydog there); under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use sha2::{Digest, Sha256};
use straydog_engine::arch::deepseek41::serve::{place, Model, CHECKPOINT_BYTES};
use straydog_engine::arch::served::{Engines as _, Settings};
use straydog_engine::group::SplitMode;
use straydog_engine::chat::ChatTemplate;
use straydog_engine::pyjson;
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;

/// What a request left: its greedy tokens, where it resumed, its prompt logits' digest, and the
/// digests of the sequence state then (every engine's window rings, compressed rows, index keys
/// and tails, on each of its devices).
#[derive(Debug, PartialEq)]
struct Outcome {
    tokens: Vec<u32>,
    cached: Option<u64>,
    logits: String,
    state: Vec<String>,
}

fn digest(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    format!("{:x}", Sha256::digest(&bytes))
}

fn go(model: &mut Model, prompt: &[u32], max_new: usize) -> Outcome {
    let mut tokens = Vec::new();
    model.generate(prompt, max_new, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, &mut |token| tokens.push(token)).unwrap();
    let state = model.engines.state().unwrap().into_iter().map(|(name, values)| format!("{name} {}", digest(&values))).collect();
    let logits = digest(model.prompt_logits.as_deref().unwrap());
    Outcome { tokens, cached: model.last_metrics["cached_tokens"].as_u64(), logits, state }
}

/// A chat's turns (thinking on, the replies sent back without their reasoning), then another
/// chat, then the first one's next turn: that turn resumes from the first chat's checkpoint (where
/// its last reply's generation prompt began, the whole state in host memory) and leaves the same
/// tokens and state as when it follows its chat directly. The checkpoints' rows are copied when a
/// restore needs them, as far as the other chat's passes write over them, or (the switched run's
/// first) in small steps between requests.
#[test]
fn a_switched_conversation_comes_back_from_its_checkpoint() {
    let Some(root) = common::model_env("STRAYDOG_DS41", "the DeepSeek-V4.1 GSQ-RCO IQ2_XS bundle directory") else { return };
    let root = PathBuf::from(root);
    let settings = Settings {
        devices: vec![0, 1], split_mode: SplitMode::Experts, vram_bytes: vec![28 << 30; 2], context: 16384, batch: 2048,
        host_expert_bytes: None, host_memory_bytes: straydog_engine::host::host_memory_bytes(),
        row_cache_bytes: straydog_engine::rows::DEFAULT_CACHE_BYTES, half_gemm: false,
    };
    let mut model = place(&root, common::deepseek41_bundle(), &root.join(".straydog"), &settings, &|_, _, _| {}).unwrap();
    // (as the worker does after its first passes: the VRAM the cards have spare goes to the expert stores)
    model.generate(&[0, 1000, 2000, 3000], 2, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, &mut |_| {}).unwrap();
    model.forget();
    let slots = model.placement["expert_slots"].clone();
    model.lend_spare().unwrap();
    assert!(model.placement["expert_slots"][0].as_u64() > slots[0].as_u64(), "the spare VRAM made slots: {slots} before, {} after", model.placement["expert_slots"]);
    model.prompt_logits = Some(Vec::new());
    let text = Text::from_gguf(model.header()).unwrap();
    let template = ChatTemplate::new(straydog_engine::arch::deepseek41::CHAT_TEMPLATE).unwrap();
    // Chats with thinking on, sent back as clients send them: each reply's content alone, without
    // its reasoning, so a turn never extends the held sequence (its reply's thinking) and resumes
    // from the checkpoint where that reply's generation prompt began.
    let render = |messages: &str| -> Vec<u32> {
        let messages = pyjson::parse(messages).unwrap();
        text.encode(&template.render(&messages, &pyjson::parse("[]").unwrap(), true, "low").unwrap()).unwrap()
    };
    let document: String = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/DESIGN.md")).unwrap()
        .chars().take(12000).collect();
    let user = |content: &str| format!("{{\"role\": \"user\", \"content\": {}}}", serde_json::to_string(content).unwrap());
    let reply = |content: &str| format!("{{\"role\": \"assistant\", \"content\": {}}}", serde_json::to_string(content).unwrap());
    let system = r#"{"role": "system", "content": "You are a careful technical reader."}"#;
    let turns = [user(&format!("{document}\n\nWhat is this project?")), reply("A GPU inference server."), user("Name its main parts."),
                 reply("A server, a worker and engines."), user("How does it keep a conversation?")];
    let first = render(&format!("[{system}, {}]", turns[0]));
    let second = render(&format!("[{system}, {}]", turns[..3].join(", ")));
    let third = render(&format!("[{system}, {}]", turns.join(", ")));
    let other = render(&format!("[{}]", user("Write four lines about the sea in winter.")));

    let mut runs = Vec::new();
    for switched in [false, true] {
        model.forget();
        go(&mut model, &first, 8);
        if switched {
            model.engines.limit_checkpoint_steps(1 << 20);
            let mut steps = 1;
            while !model.make_room().unwrap() {
                steps += 1;
            }
            assert!(steps > 2, "the first turn's checkpoints copied in steps ({steps})");
        }
        let next = go(&mut model, &second, 8);
        assert!(next.cached.is_some_and(|cached| cached as usize > first.len() / 2), "the second turn from the first's checkpoint");
        if switched {
            let outcome = go(&mut model, &other, 16);
            assert_eq!(outcome.cached, Some(0), "the other conversation from nothing");
        }
        runs.push(go(&mut model, &third, 32));
        eprintln!("switched {switched}: resumed {:?} at {:?}, {} tokens", model.last_metrics["resumed_from"], runs.last().unwrap().cached,
                  runs.last().unwrap().tokens.len());
    }
    let (direct, back) = (&runs[0], &runs[1]);
    assert!(back.cached.is_some_and(|cached| cached as usize > first.len()), "the third turn from the second's checkpoint");
    assert_eq!(back.cached, direct.cached, "resumed at");
    assert_eq!(back.logits, direct.logits, "prompt logits");
    assert_eq!(back.tokens, direct.tokens, "tokens");
    let differing: Vec<_> = back.state.iter().zip(&direct.state).filter(|(a, b)| a != b).collect();
    for (a, b) in differing.iter().take(12) {
        eprintln!("{a} (direct {})", b.rsplit(' ').next().unwrap());
    }
    assert!(differing.is_empty() && back.state.len() == direct.state.len(), "state ({} parts differ)", differing.len());

    // A turn whose checkpoint the budget did not let be taken, regenerated once there is room,
    // takes it then: the turn after it resumes from there.
    let fourth = render(&format!("[{system}, {}, {}, {}]", turns.join(", "), reply("In checkpoints."), user("Where are they kept?")));
    let fifth = render(&format!("[{system}, {}, {}, {}, {}, {}]", turns.join(", "), reply("In checkpoints."), user("Where are they kept?"),
                                reply("In host memory."), user("How many?")));
    model.engines.limit_checkpoints(0);
    go(&mut model, &fourth, 8);
    model.engines.limit_checkpoints(CHECKPOINT_BYTES);
    let again = go(&mut model, &fourth, 8);
    assert!(again.cached.is_some_and(|cached| (cached as usize) < third.len()), "regenerated from the third turn's checkpoint");
    let after = go(&mut model, &fifth, 8);
    assert!(after.cached.is_some_and(|cached| cached as usize > third.len()), "the fifth turn from the regenerated fourth's checkpoint");

    // A long system prompt is its conversations' root: a new conversation with it starts from
    // there and leaves what starting over leaves.
    let long_system = format!("{{\"role\": \"system\", \"content\": {}}}", serde_json::to_string(&document).unwrap());
    let opening = render(&format!("[{long_system}, {}]", user("What is this project?")));
    let another = render(&format!("[{long_system}, {}]", user("Name its main parts.")));
    model.forget();
    go(&mut model, &opening, 8);
    let rooted = go(&mut model, &another, 16);
    assert_eq!(model.last_metrics["resumed_from"], "root");
    assert!(rooted.cached.is_some_and(|cached| cached as usize > first.len() / 2 && (cached as usize) < another.len() - 4),
            "the new conversation from the root: {:?}", rooted.cached);
    model.forget();
    let direct = go(&mut model, &another, 16);
    assert_eq!(direct.cached, Some(0));
    assert_eq!(rooted.logits, direct.logits, "prompt logits from the root");
    assert_eq!(rooted.tokens, direct.tokens, "tokens from the root");
    assert_eq!(rooted.state, direct.state, "state from the root");

    // A pass made void mid-reply (as a part link's wait that gave up makes it) voids that pass, not
    // the reply: the sequence starts over from the checkpoints with the tokens already sent, and the
    // reply goes on to its length. Its tokens before the void are the uninterrupted run's.
    model.forget();
    let clean = go(&mut model, &other, 24);
    model.forget();
    model.engines.void_pass_in(5);
    let retried = go(&mut model, &other, 24);
    assert_eq!(model.last_metrics.get("void_pass_retried"), Some(&serde_json::json!(true)), "the void pass was taken again");
    assert_eq!(retried.tokens.len(), clean.tokens.len(), "the reply's length");
    assert_eq!(retried.tokens[..5], clean.tokens[..5], "the tokens sent before the void");
    eprintln!("after a void pass: {} of {} tokens as the uninterrupted run's", retried.tokens.iter().zip(&clean.tokens).take_while(|(a, b)| a == b).count(),
              clean.tokens.len());

    // A long reply is marked as it grows. A client that keeps it only in part (a file it cut at a line's end to have the
    // model go on, a reply its person stopped) sends a prompt that leaves the held sequence before its end: the request goes
    // on from the last mark the part kept reaches, where it read the reply again from the checkpoint before it.
    model.forget();
    let long = render(&format!("[{}]", user("Write a long and detailed essay about the sea in winter, of at least two thousand words.")));
    let written = go(&mut model, &long, 1300);
    let marks = model.last_metrics["marks"].as_u64().unwrap();
    let marked: Vec<usize> = model.last_metrics["marked_tokens"].as_array().unwrap().iter().map(|at| at.as_u64().unwrap() as usize).collect();
    eprintln!("a reply of {} tokens: {marks} marks, {:.1} ms each, kept at {marked:?} of {}", written.tokens.len(),
              1e3 * model.last_metrics["mark_seconds"].as_f64().unwrap() / marks.max(1) as f64, long.len() + written.tokens.len());
    assert!(marks >= 2 && marked.len() == 2 && marked[0] > long.len(), "the reply was long enough to be marked twice: {} tokens", written.tokens.len());
    let tail = text.encode("\n\nGo on from there.").unwrap();
    // (the part kept ends 40 tokens past a mark: past the last, and between the two, where the last is of no use)
    let part = |mark: usize| -> Vec<u32> { long.iter().chain(&written.tokens[..mark - long.len() + 40]).chain(&tail).copied().collect() };
    assert!(marked[1] - long.len() + 48 < written.tokens.len() && marked[0] + 40 < marked[1], "the parts kept end before the reply and the last mark");
    let next = part(marked[1]);
    let stepped = go(&mut model, &next, 8);
    let stepped_logits = model.prompt_logits.clone().unwrap();
    assert_eq!(model.last_metrics["resumed_from"], "step");
    assert_eq!(stepped.cached, Some(marked[1] as u64), "from the last mark");
    model.forget();
    let again = go(&mut model, &long, 1300);
    assert_eq!(again.tokens, written.tokens, "the reply written again");
    let earlier = go(&mut model, &part(marked[0]), 8);
    assert_eq!(model.last_metrics["resumed_from"], "step");
    assert_eq!(earlier.cached, Some(marked[0] as u64), "from the mark before the last");
    // Without the marks the same request starts where the reply's generation prompt began; its first token is the same.
    model.forget();
    go(&mut model, &long, 1300);
    model.forget_marks();
    let turned = go(&mut model, &next, 8);
    let turned_logits = model.prompt_logits.clone().unwrap();
    assert_eq!(model.last_metrics["resumed_from"], "turn");
    assert!(turned.cached.is_some_and(|cached| (cached as usize) < long.len()), "from before the reply: {:?}", turned.cached);
    let top = |logits: &[f32]| logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(token, _)| token);
    let widest = stepped_logits.iter().zip(&turned_logits).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    eprintln!("from the mark and from the turn: the largest logit difference {widest}, {} of {} tokens the same",
              stepped.tokens.iter().zip(&turned.tokens).take_while(|(a, b)| a == b).count(), turned.tokens.len());
    assert_eq!(top(&stepped_logits), top(&turned_logits), "the first token");

    // Back at a mark the engines are where they were when it was taken: the same pass then gives the same logits, bit for
    // bit, after a pass of one token and after batches that wrote over the whole window and began new groups; at the last
    // mark and at the one before it, which leaves the last behind.
    model.forget();
    go(&mut model, &other, 8);
    let at = model.held().len();
    let probe = &written.tokens[..3];
    let batch = &written.tokens[3..303];
    assert!(model.engines.mark(None).unwrap());
    let one = model.engines.forward(&probe[..1]).unwrap();
    model.engines.forward(batch).unwrap();
    assert!(model.engines.mark(None).unwrap());
    let far = model.engines.forward(&probe[..1]).unwrap();
    model.engines.forward(batch).unwrap();
    assert!(model.engines.mark(Some(at + 301)).unwrap());
    assert!(far == model.engines.forward(&probe[..1]).unwrap(), "a pass of one token from the last mark");
    assert!(model.engines.mark(Some(at)).unwrap());
    assert!(one == model.engines.forward(&probe[..1]).unwrap(), "a pass of one token from the mark before it");
    assert!(!model.engines.mark(Some(at + 301)).unwrap(), "the mark the sequence went back past");
    assert!(model.engines.mark(Some(at)).unwrap());
    let rows = model.engines.verify(probe).unwrap();
    model.engines.rewind(2).unwrap();
    model.engines.forward(batch).unwrap();
    assert!(model.engines.mark(Some(at)).unwrap());
    assert!(rows == model.engines.verify(probe).unwrap(), "a verify pass from the mark");
    model.engines.rewind(1).unwrap();
    model.forget();
    // (another sequence has left the mark)
    go(&mut model, &first, 8);
    assert!(!model.engines.mark(Some(at)).unwrap());

    // A client that writes its folder and the date into its system prompt sends a system prompt that ends otherwise at every new
    // conversation: none extends another's root. The second such prompt is checkpointed where it parts from the first (its
    // stem), and the third starts from there, with the first token of a model that reads it whole.
    let ending = |folder: &str| -> Vec<u32> {
        let system = format!("{{\"role\": \"system\", \"content\": {}}}",
                             serde_json::to_string(&format!("{document}\n\nWorking folder: /work/{folder}\nToday is a day in November.")).unwrap());
        render(&format!("[{system}, {}]", user("What is this project?")))
    };
    let (one, two, three) = (ending("alpha-run"), ending("bravo-run"), ending("charlie-run"));
    let shared = one.iter().zip(&two).take_while(|(a, b)| a == b).count();
    assert!(shared > 2000 && shared < one.len() - 8, "the system prompts part near their end: {shared} of {}", one.len());
    model.forget();
    go(&mut model, &one, 8);
    let second = go(&mut model, &two, 8);
    assert_eq!(second.cached, Some(0), "the second ending extends nothing yet");
    let from_stem = go(&mut model, &three, 8);
    let from_stem_logits = model.prompt_logits.clone().unwrap();
    assert_eq!(model.last_metrics["resumed_from"], "root");
    assert!(from_stem.cached.is_some_and(|cached| (cached as usize) < shared && (cached as usize) + 64 > shared), "from the stem: {:?} of {shared} shared", from_stem.cached);
    model.forget();
    let whole = go(&mut model, &three, 8);
    let whole_logits = model.prompt_logits.clone().unwrap();
    assert_eq!(whole.cached, Some(0));
    let widest = from_stem_logits.iter().zip(&whole_logits).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    eprintln!("from the stem ({:?} of {} tokens) and read whole: the largest logit difference {widest}, {} of {} tokens the same", from_stem.cached, three.len(),
              from_stem.tokens.iter().zip(&whole.tokens).take_while(|(a, b)| a == b).count(), whole.tokens.len());
    assert_eq!(top(&from_stem_logits), top(&whole_logits), "the first token from the stem");

    // The ids of a reply are kept where its text comes back (Text::reach): all of them spell the text they decode to,
    // whether or not the tokenizer splits that text as the model wrote it.
    let ids: Vec<u32> = long.iter().chain(&written.tokens).copied().collect();
    let spelled = text.decode(&ids).unwrap();
    assert_eq!(text.reach(&ids, &spelled), (ids.len(), spelled.len()));
    let encoded = text.encode(&spelled).unwrap();
    eprintln!("the reply's text encoded again: the same ids for {} of {}", encoded.iter().zip(&ids).take_while(|(a, b)| a == b).count(), ids.len());
}
