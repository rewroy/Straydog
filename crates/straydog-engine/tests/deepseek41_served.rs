//! DeepSeek-V4.1 served on real GPUs: a conversation switched away from comes back from its
//! checkpoint as if nothing had come between.
//!
//! cargo test -p straydog-engine --features hip --test deepseek41_served, with STRAYDOG_DS41 (the
//! GSQ-RCO IQ2_XS bundle directory, its verity trees in .straydog there); under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use sha2::{Digest, Sha256};
use straydog_engine::arch::deepseek41::serve::{place, Model, CHECKPOINT_BYTES};
use straydog_engine::arch::served::Settings;
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
/// restore or a reset needs them, or (the switched run's first) in small steps between requests.
#[test]
fn a_switched_conversation_comes_back_from_its_checkpoint() {
    let Some(root) = common::model_env("STRAYDOG_DS41", "the DeepSeek-V4.1 GSQ-RCO IQ2_XS bundle directory") else { return };
    let root = PathBuf::from(root);
    let settings = Settings {
        devices: vec![0, 1], split_mode: SplitMode::Experts, vram_bytes: vec![28 << 30; 2], context: 16384, batch: 2048,
        host_expert_bytes: None, host_memory_bytes: straydog_engine::host::host_memory_bytes(),
        row_cache_bytes: straydog_engine::rows::DEFAULT_CACHE_BYTES, half_gemm: false,
    };
    let mut model = place(&root, "GSQ-RCO-IQ2_XS", &root.join(".straydog"), &settings, &|_, _, _| {}).unwrap();
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
}
