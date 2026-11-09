//! Pinned generations reproduced bit for bit on real GPUs: for each placement, every request's
//! greedy tokens, metrics, logits and final hidden state.
//!
//! cargo test -p straydog-engine --features hip --test model_gpu, with STRAYDOG_MODEL and
//! STRAYDOG_DRAFTER (STRAYDOG_PLACEMENT runs one placement); under the GPU lock. A deliberate
//! numerical change, once validated against an F32 reference (examples/prefill --dump and
//! --gemm f32), re-pins the generations with STRAYDOG_GOLDEN_UPDATE=1.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::weights::Weights;

fn digest(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    format!("{:x}", Sha256::digest(&bytes))
}

/// The tests load the model on the same GPUs: one at a time.
static GPUS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The model as a golden placement describes it.
fn open(model_root: &std::ffi::OsStr, drafter: &std::ffi::OsStr, quantization: &str, placement: &Json) -> Model {
    let devices: Vec<usize> = serde_json::from_value(placement["devices"].clone()).unwrap();
    let budget = (placement["vram_gib"].as_f64().unwrap() * (1u64 << 30) as f64) as u64;
    let split = if placement["split_mode"] == "layer" { SplitMode::Layer } else { SplitMode::Experts };
    let speculative = placement["speculative"].as_u64().unwrap() as usize;
    let mut group = Group::new(&devices, &vec![budget; devices.len()], split).unwrap();
    group.configure_gemm(2, true).unwrap();
    let mut weights = Weights::open(&PathBuf::from(model_root), straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, quantization).unwrap(),
                                    (speculative > 0).then(|| PathBuf::from(drafter)).as_deref(),
                                    straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    // With STRAYDOG_DATA_DIR (the shards' verity trees there), what is read again after
    // loading is checked block by block: the generations must not change.
    if let Some(data_dir) = std::env::var_os("STRAYDOG_DATA_DIR") {
        weights.open_trees(std::path::Path::new(&data_dir)).unwrap();
    }
    let settings = Settings {
        context: placement["context"].as_u64().unwrap() as usize,
        // (--state-ram's default at 262144 tokens with the drafter)
        state_budget_bytes: 14 << 30,
        prefill_batch_size: 4096,
        speculative,
        cpu_threads: 0,
        row_cache_bytes: None,
    };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    // (a partially resident prompt's experts all stream, so its logits are the same from run to run: which go to the CPU depends on
    // timing, and prompt_cpu_gpu checks that share)
    model.prompt_share(false).unwrap();
    model
}

/// A request's greedy tokens and logits digests.
fn run(model: &mut Model, request: &Json) -> (Vec<u32>, Vec<String>) {
    let prompt: Vec<u32> = serde_json::from_value(request["prompt"].clone()).unwrap();
    model.logits_digests = Some(Vec::new());
    let mut tokens = Vec::new();
    model.generate(&prompt, request["max_new"].as_u64().unwrap() as usize, &straydog_engine::sampling::Sampling::GREEDY, 0, &|| false,
                   &mut |_, _| {}, request["turn"].as_bool().unwrap(), request["checkpoint"].as_bool().unwrap(),
                   &mut |token| tokens.push(token)).unwrap();
    (tokens, model.logits_digests.take().unwrap())
}

/// A conversation another prompt departs from is parked and comes back: the root request, then
/// the tools one (another conversation: the root's is parked), then the live one, which continues
/// the root's. It resumes on the restored sequence (or its checkpoint) as it does straight after
/// the root request, with the same logits (pinned), with the K/V in VRAM and in host memory.
#[test]
fn a_parked_conversation_comes_back() {
    let (Some(model_root), Some(drafter)) = (common::model_env("STRAYDOG_MODEL", "the model bundle directory"),
                                             common::model_env("STRAYDOG_DRAFTER", "the drafter directory"))
    else {
        return;
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/model.json");
    let golden: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let quantization = golden["quantization"].as_str().unwrap();
    let _gpus = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    for name in ["two-gpu-experts-mtp", "two-gpu-host-kv-mtp"] {
        let expected = &golden["placements"][name];
        let request = |label: &str| expected["requests"].as_array().unwrap().iter().find(|request| request["name"] == label).unwrap();
        let mut model = open(&model_root, &drafter, quantization, &expected["placement"]);
        run(&mut model, request("root"));
        run(&mut model, request("tools"));
        assert_eq!(model.last_metrics["parked"], "saved", "{name}: the root conversation parked");
        let saved = model.last_metrics["park_seconds"].as_f64().unwrap();
        let (tokens, logits) = run(&mut model, request("live"));
        assert_eq!(model.last_metrics["parked"], "restored", "{name}: the root conversation back");
        // The same position (the tools request's own checkpoint may have renamed the root's turn
        // checkpoint "prior").
        assert_eq!(model.last_metrics["cached_tokens"], request("live")["metrics"]["cached_tokens"], "{name}: cached tokens");
        assert_ne!(model.last_metrics["resumed_from"], "none", "{name}");
        let pinned: Vec<u32> = serde_json::from_value(request("live")["tokens"].clone()).unwrap();
        let digests: Vec<String> = serde_json::from_value(request("live")["logits"].clone()).unwrap();
        assert_eq!(tokens, pinned, "{name}: tokens");
        assert_eq!(logits, digests, "{name}: logits");
        assert_eq!(digest(model.final_hidden().unwrap()), request("live")["final_hidden"].as_str().unwrap(), "{name}: final hidden");
        eprintln!("{name}: parked in {saved:.3} s, restored in {:.3} s", model.last_metrics["park_seconds"].as_f64().unwrap());
    }
    bounded_parking(&model_root, &drafter, quantization, &golden);
}

/// Speculative decoding leaves the target as one-token decoding does (T173). Past 512 pooled
/// blocks (2048 positions) a token attends to the 512 blocks its indexer picks by their pooled
/// keys, so a block pooled from a rejected draft's input changes what later tokens see: a 20K
/// conversation decoded greedily with the MTP drafts and without them leaves the same tokens,
/// final hidden state and target state (every layer's K/V, pooled blocks and history and the
/// recurrent state; the MTP layer runs only with drafts), and the next turn the same logits.
#[test]
fn speculation_leaves_the_target_as_decoding() {
    let (Some(model_root), Some(drafter)) = (common::model_env("STRAYDOG_MODEL", "the model bundle directory"),
                                             common::model_env("STRAYDOG_DRAFTER", "the drafter directory"))
    else {
        return;
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/model.json");
    let golden: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let quantization = golden["quantization"].as_str().unwrap();
    let _gpus = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let expected = &golden["placements"]["two-gpu-experts-mtp"];
    let prompt = |label: &str| -> Vec<u32> {
        let request = expected["requests"].as_array().unwrap().iter().find(|request| request["name"] == label).unwrap();
        serde_json::from_value(request["prompt"].clone()).unwrap()
    };
    // (then a request for a long story of its own: drafts it often rejects)
    let text = straydog_engine::text::Text::new(&std::fs::read(PathBuf::from(&model_root).join("tokenizer.json")).unwrap()).unwrap();
    let story = text.encode("Write a long, original short story about a lighthouse keeper who receives a strange radio \
                             signal during a winter storm. Use vivid detail and dialogue, at least 800 words.").unwrap();
    let earlier: Vec<u32> = ["root", "tools", "turn", "new-chat"].iter().flat_map(|label| prompt(label)).collect();
    let conversation = turn(&earlier, &[], &story);
    let mut runs = Vec::new();
    for drafting in [true, false] {
        let mut model = open(&model_root, &drafter, quantization, &expected["placement"]);
        if !drafting {
            model.speculative = 0;
        }
        let first = go(&mut model, &conversation, 512, None);
        let next = turn(&conversation, &first.tokens, &prompt("root")[4000..4030]);
        let second = go(&mut model, &next, 1, None);
        eprintln!("{} drafts: {} prompt tokens, {} generated, drafts {:?}/{:?}", if drafting { "with" } else { "without" },
                  conversation.len(), first.tokens.len(), first.metrics.get("drafted_tokens"), first.metrics.get("accepted_tokens"));
        runs.push((first, second));
    }
    let ((with, with_next), (without, without_next)) = (&runs[0], &runs[1]);
    // (drafts rejected, so verified rows past the kept ones)
    assert!(with.tokens.len() >= 256 && with.metrics["drafted_tokens"].as_u64() > with.metrics["accepted_tokens"].as_u64(),
            "a long reply with rejected drafts");
    let target = |outcome: &Outcome| -> Vec<String> { outcome.state.iter().filter(|line| !line.contains(":blk.48.")).cloned().collect() };
    let (with_state, without_state) = (target(with), target(without));
    let differing: Vec<_> = with_state.iter().zip(&without_state).filter(|(a, b)| a != b).collect();
    for (a, b) in differing.iter().take(24) {
        eprintln!("with drafts: {a} (without {})", b.rsplit(' ').next().unwrap());
    }
    let parted = with.tokens.iter().zip(&without.tokens).position(|(a, b)| a != b);
    eprintln!("{} state parts differ of {}; tokens part at {parted:?}", differing.len(), with_state.len());
    assert_eq!(with.tokens, without.tokens, "tokens");
    assert_eq!(with.hidden, without.hidden, "final hidden");
    assert!(differing.is_empty() && with_state.len() == without_state.len(), "target state ({} parts differ)", differing.len());
    assert_eq!((&with_next.tokens, &with_next.logits), (&without_next.tokens, &without_next.logits), "the next turn");
}

/// Passes of up to eight rows, drafts rejected at every place, on the resident placements (each its own path through the layers:
/// layers in parts, whole layers with their experts split, host K/V behind its stage): the tokens, final hidden state and target
/// state of plain one-row decode, and the next turn the same logits. A row taken back must leave what the kept rows read (K/V
/// mirrors' margin, the indexers' history ring, DeltaNet and PLE replay).
#[test]
fn wide_passes_leave_the_target_as_decoding() {
    let (Some(model_root), Some(drafter)) = (common::model_env("STRAYDOG_MODEL", "the model bundle directory"),
                                             common::model_env("STRAYDOG_DRAFTER", "the drafter directory"))
    else {
        return;
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/model.json");
    let golden: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let quantization = golden["quantization"].as_str().unwrap();
    let _gpus = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let only = std::env::var("STRAYDOG_PLACEMENT").ok();
    for name in ["two-gpu-experts-mtp", "two-gpu-layers-mtp", "two-gpu-host-kv-mtp"] {
        if only.as_ref().is_some_and(|only| only != name) {
            continue;
        }
        let expected = &golden["placements"][name];
        let prompt = |label: &str| -> Vec<u32> {
            let request = expected["requests"].as_array().unwrap().iter().find(|request| request["name"] == label).unwrap();
            serde_json::from_value(request["prompt"].clone()).unwrap()
        };
        // (a request for a long story, so that the passes go past every width and rows are taken back at every place)
        let text = straydog_engine::text::Text::new(&std::fs::read(PathBuf::from(&model_root).join("tokenizer.json")).unwrap()).unwrap();
        let story = text.encode("Write a long, original short story about a lighthouse keeper who receives a strange radio \
                                 signal during a winter storm. Use vivid detail and dialogue, at least 800 words.").unwrap();
        let conversation = turn(&prompt("root"), &[], &story);
        let mut plain = open(&model_root, &drafter, quantization, &expected["placement"]);
        plain.speculative = 0;
        let without = go(&mut plain, &conversation, 192, None);
        assert_eq!(without.tokens.len(), 192, "{name}: a long reply");
        let next = turn(&conversation, &without.tokens, &prompt("root")[4030..4060]);
        let without_next = go(&mut plain, &next, 1, None);
        drop(plain);
        for reject in [0, 2] {
            let mut model = open(&model_root, &drafter, quantization, &expected["placement"]);
            model.forced_rows = Some(straydog_engine::model::ForcedRows {
                tokens: without.tokens.clone(), widths: vec![8, 5, 7, 6, 8, 2, 3, 4, 1], reject, ..Default::default()
            });
            let wide = go(&mut model, &conversation, 192, None);
            let forced = model.forced_rows.take().unwrap();
            let widest = forced.passes.iter().map(|(rows, _)| *rows).max().unwrap();
            eprintln!("{name}: {} passes, rejected {}", forced.passes.len(), forced.rejected);
            assert!(widest == 8 && forced.passes.len() >= 30, "{name}: {} passes up to {widest} rows", forced.passes.len());
            assert!(if reject > 0 { forced.rejected > 0 } else { forced.rejected == 0 }, "{name}: rejected {}", forced.rejected);
            assert_eq!(wide.tokens, without.tokens, "{name}: tokens");
            assert_eq!(wide.hidden, without.hidden, "{name}: final hidden");
            let target = |outcome: &Outcome| -> Vec<String> { outcome.state.iter().filter(|line| !line.contains(":blk.48.")).cloned().collect() };
            let (wide_state, without_state) = (target(&wide), target(&without));
            for (a, b) in wide_state.iter().zip(&without_state).filter(|(a, b)| a != b).take(24) {
                eprintln!("{name}: wide {a} (one row {})", b.rsplit(' ').next().unwrap());
            }
            assert!(wide_state == without_state, "{name}: target state");
            let wide_next = go(&mut model, &next, 1, None);
            assert_eq!((&wide_next.tokens, &wide_next.logits), (&without_next.tokens, &without_next.logits), "{name}: the next turn");
        }
    }
}

#[test]
fn generations_equal_the_reference() {
    let (Some(model_root), Some(drafter)) = (common::model_env("STRAYDOG_MODEL", "the model bundle directory"),
                                             common::model_env("STRAYDOG_DRAFTER", "the drafter directory"))
    else {
        return;
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/model.json");
    let mut golden: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let update = std::env::var_os("STRAYDOG_GOLDEN_UPDATE").is_some();
    let _gpus = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let only = std::env::var("STRAYDOG_PLACEMENT").ok();
    let quantization = golden["quantization"].as_str().unwrap().to_string();
    for (name, expected) in golden["placements"].as_object_mut().unwrap() {
        if only.as_ref().is_some_and(|only| only != name) {
            continue;
        }
        eprintln!("placement {name}");
        let mut model = open(&model_root, &drafter, &quantization, &expected["placement"]);
        if update {
            expected["resident"] = model.resident.into();
            expected["partial"] = model.partial.into();
            expected["host_kv"] = model.host_kv.into();
            expected["prefill_batch_size"] = model.prefill_batch_size.into();
            expected["speculative"] = model.speculative.into();
            expected["state_bytes"] = model.state_bytes.into();
        }
        assert_eq!(model.resident, expected["resident"].as_bool().unwrap(), "{name}: resident");
        assert_eq!(model.partial, expected["partial"].as_bool().unwrap(), "{name}: partial");
        assert_eq!(model.host_kv, expected["host_kv"].as_bool().unwrap(), "{name}: host K/V");
        assert_eq!(model.prefill_batch_size as u64, expected["prefill_batch_size"].as_u64().unwrap(), "{name}: batch");
        assert_eq!(model.speculative as u64, expected["speculative"].as_u64().unwrap(), "{name}: speculative");
        assert_eq!(model.state_bytes, expected["state_bytes"].as_u64().unwrap(), "{name}: state bytes");
        for request in expected["requests"].as_array_mut().unwrap() {
            let label = format!("{name}/{}", request["name"].as_str().unwrap());
            // Partial residency swaps expert slots asynchronously during decode, so whether an
            // expert runs on the GPU or the CPU (the same F32 weights, summed in another order)
            // depends on timing, and the reference does not reproduce its own decode bit for bit.
            // There only a request whose state comes from prefill alone (a new sequence, or the
            // system-prompt checkpoint) is compared: its prompt logits and the token they give
            // (the tokens after it come from decode).
            let resumed = request["metrics"]["resumed_from"].as_str().unwrap();
            let fresh = request["metrics"]["cached_tokens"].as_u64() == Some(0);
            let comparable = !model.partial || fresh || resumed == "root";
            let (tokens, logits) = run(&mut model, request);
            if update {
                request["tokens"] = serde_json::to_value(&tokens).unwrap();
                for (key, value) in request["metrics"].as_object_mut().unwrap() {
                    *value = model.last_metrics.get(key).cloned().unwrap_or(Json::Null);
                }
                request["logits"] = serde_json::to_value(&logits).unwrap();
                if let Some(hidden) = model.final_hidden() {
                    request["final_hidden"] = digest(hidden).into();
                }
                eprintln!("  {label}: pinned ({} tokens)", tokens.len());
                continue;
            }
            if !comparable {
                eprintln!("  {label}: run (state from decode under partial residency)");
                continue;
            }
            let expected_tokens: Vec<u32> = serde_json::from_value(request["tokens"].clone()).unwrap();
            if model.partial {
                assert_eq!(tokens.first(), expected_tokens.first(), "{label}: first token");
            } else {
                assert_eq!(tokens, expected_tokens, "{label}: tokens");
            }
            for (key, value) in request["metrics"].as_object().unwrap() {
                let actual = model.last_metrics.get(key).cloned().unwrap_or(Json::Null);
                assert_eq!(actual.as_f64().or(actual.as_str().map(|_| 0.0)), value.as_f64().or(value.as_str().map(|_| 0.0)),
                           "{label}: {key}");
                if let Some(text) = value.as_str() {
                    assert_eq!(actual.as_str(), Some(text), "{label}: {key}");
                }
            }
            let digests: Vec<String> = serde_json::from_value(request["logits"].clone()).unwrap();
            let actual = logits;
            if model.partial {
                assert_eq!(actual[0], digests[0], "{label}: prompt logits");
            } else {
                assert_eq!(actual, digests, "{label}: logits");
                assert_eq!(digest(model.final_hidden().unwrap()), request["final_hidden"].as_str().unwrap(),
                           "{label}: final hidden");
            }
            eprintln!("  {label}: ok ({} tokens)", tokens.len());
        }
    }
    if update {
        // The file's layout: one space per level (to_string_pretty indents by two).
        let text: String = serde_json::to_string_pretty(&golden).unwrap().lines()
            .map(|line| {
                let depth = line.len() - line.trim_start_matches(' ').len();
                format!("{}{}\n", " ".repeat(depth / 2), &line[depth..])
            })
            .collect();
        std::fs::write(&path, text).unwrap();
    }
}

/// What a request left: its greedy tokens, its logits' digests, the final hidden state's digest,
/// the digest of the sequence state then valid (K/V and pooled rows below its end, the indexers'
/// history, the recurrent state) and its metrics.
struct Outcome {
    tokens: Vec<u32>,
    logits: Vec<String>,
    hidden: String,
    state: Vec<String>,
    metrics: serde_json::Map<String, Json>,
}

/// `prompt` with greedy decoding of `max_new`, cancelled at the `cancel`th check if given.
fn go(model: &mut Model, prompt: &[u32], max_new: usize, cancel: Option<usize>) -> Outcome {
    model.logits_digests = Some(Vec::new());
    let checks = std::cell::Cell::new(0usize);
    let cancelled = || {
        checks.set(checks.get() + 1);
        cancel.is_some_and(|at| checks.get() >= at)
    };
    let mut tokens = Vec::new();
    model.generate(prompt, max_new, &straydog_engine::sampling::Sampling::GREEDY, 0, &cancelled, &mut |_, _| {}, true, true,
                   &mut |token| tokens.push(token)).unwrap();
    Outcome { tokens, logits: model.logits_digests.take().unwrap(), hidden: model.final_hidden().map(digest).unwrap_or_default(),
              state: model.state_digest().unwrap(), metrics: model.last_metrics.clone() }
}

/// The length of the prefix `a` and `b` share.
fn shared(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The chat template's next turn: the conversation so far (`prompt` and its reply `reply`), a
/// user message of `content`, and the assistant's header.
fn turn(prompt: &[u32], reply: &[u32], content: &[u32]) -> Vec<u32> {
    let (start, end, user, assistant, newline) = (248045, 248046, 846, 74455, 198);
    let mut next = prompt.to_vec();
    if !reply.is_empty() {
        next.extend_from_slice(reply);
        next.extend([end, newline]);
    }
    next.extend([start, user, newline]);
    next.extend_from_slice(content);
    next.extend([end, newline, start, assistant, newline, 248068, 271, 248069, 271]);
    next
}

/// The conversations' prompts (the later turns built on the reference's replies).
struct Prompts {
    root: Vec<u32>, short: Vec<u32>, live: Vec<u32>, tools: Vec<u32>, tools_more: Vec<u32>, step: Vec<u32>,
    step_last: Vec<u32>, other: Vec<u32>, other_next: Vec<u32>, other_last: Vec<u32>, other_final: Vec<u32>, past: Vec<u32>,
    below: Vec<u32>,
}

/// Parking a conversation copies only what later prompts overwrite (T164's first policy), and a
/// restored conversation goes on exactly as if nothing had come between. The requests run
/// interleaved, conversations parked and restored, twice: parking each departing conversation
/// whole at once (the reference for every bit) and bounded; every request leaves the same
/// tokens, logits, final hidden state, resume position, drafts and sequence state (the MTP
/// layer's too) in both. Each conversation's requests also run back to back (none coming back
/// once left), and the interleaved ones leave all of it the same as there too, whatever ran
/// before them. Among
/// them: a short prompt that copies a few positions and borrows the rest of a 4.9K conversation
/// (1); its continuation past what was copied (2); a prompt that resumes from a checkpoint below
/// where it departs (R < live) and prompt lengths at different residues of four (3); drafts at
/// max_new 1, 2 and 3 and prompts on the stashed and the kept MTP paths, one a token past a
/// prefill batch (4); a prompt cancelled during its prefill, which keeps its whole batches (5); two long conversations trading
/// places, both slots, and a budget too low for a swap or a copy (6); with the K/V in VRAM and
/// in host memory behind its stage (7). Run by a_parked_conversation_comes_back, which holds the
/// GPUs.
fn bounded_parking(model_root: &std::ffi::OsStr, drafter: &std::ffi::OsStr, quantization: &str, golden: &Json) {
    for name in ["two-gpu-experts-mtp", "two-gpu-host-kv-mtp"] {
        let expected = &golden["placements"][name];
        let prompt = |label: &str| -> Vec<u32> {
            let request = expected["requests"].as_array().unwrap().iter().find(|request| request["name"] == label).unwrap();
            serde_json::from_value(request["prompt"].clone()).unwrap()
        };
        let (root, tools, short) = (prompt("root"), prompt("tools"), prompt("short"));
        // A second conversation whose system prompt ends at 2051 (its root checkpoint at 2052).
        let mut other = root[..2051].to_vec();
        other.extend([248046, 198]);
        let other = turn(&other, &[], &root[100..3100]);

        // Each conversation's requests back to back.
        let mut model = open(model_root, drafter, quantization, &expected["placement"]);
        let mut apart = std::collections::HashMap::new();
        let root_out = go(&mut model, &root, 24, None);
        let live = turn(&root, &root_out.tokens, &root[4000..4030]);
        apart.insert("live", go(&mut model, &live, 24, None));
        let other_out = go(&mut model, &other, 24, None);
        let other_next = turn(&other, &other_out.tokens, &root[200..260]);
        let other_next_out = go(&mut model, &other_next, 16, None);
        let other_last = turn(&other_next, &other_next_out.tokens, &root[300..320]);
        apart.insert("other", other_out);
        apart.insert("other-next", other_next_out);
        let other_last_out = go(&mut model, &other_last, 16, None);
        let other_final = turn(&other_last, &other_last_out.tokens, &root[700..720]);
        apart.insert("other-last", other_last_out);
        apart.insert("other-final", go(&mut model, &other_final, 16, None));
        // (from the other conversation's root checkpoint, thousands of positions below its end)
        let mut below = other[..2500].to_vec();
        below.extend_from_slice(&short);
        apart.insert("below", go(&mut model, &below, 3, None));
        let tools_out = go(&mut model, &tools, 40, None);
        // (its reply decoded further, from the one token after it)
        let tools_more = [&tools[..], &tools_out.tokens[..]].concat();
        apart.insert("tools-more", go(&mut model, &tools_more, 8, None));
        let step = turn(&tools, &tools_out.tokens, &root[400..430]);
        let step_out = go(&mut model, &step, 24, None);
        let step_last = turn(&step, &step_out.tokens, &root[500..520]);
        apart.insert("tools", tools_out);
        apart.insert("step", step_out);
        apart.insert("step-last", go(&mut model, &step_last, 16, None));
        // A prompt one token past a prefill batch (4096): its last token runs alone after the
        // pipelined batch, its MTP pair from the row that batch kept (no checkpoint between).
        let past: Vec<u32> = root.iter().copied().filter(|token| ![248045, 248046].contains(token)).take(4097).collect();
        apart.insert("pipeline-end", go(&mut model, &past, 4, None));
        apart.insert("root", root_out);
        drop(model);

        let prompts =
            Prompts { root, short, live, tools, tools_more, step, step_last, other, other_next, other_last, other_final, past, below };
        let mut model = open(model_root, drafter, quantization, &expected["placement"]);
        model.park_whole();
        let whole = interleaved(&mut model, name, &prompts, true);
        drop(model);
        let mut model = open(model_root, drafter, quantization, &expected["placement"]);
        let bounded = interleaved(&mut model, name, &prompts, false);
        drop(model);

        for ((label, outcome), (_, wanted)) in bounded.iter().zip(&whole) {
            compare(name, label, outcome, wanted);
            if let Some(apart) = apart.get(label) {
                compare(name, label, outcome, apart);
            }
        }
        eprintln!("{name}: every request as with whole copies, bit for bit");
    }
}

/// `outcome` as `wanted`: resume position, tokens, logits, final hidden state, drafts and state.
fn compare(name: &str, label: &str, outcome: &Outcome, wanted: &Outcome) {
    assert_eq!(outcome.metrics["cached_tokens"], wanted.metrics["cached_tokens"], "{name} {label}: resumed at");
    assert_eq!(outcome.tokens, wanted.tokens, "{name} {label}: tokens");
    assert_eq!(outcome.logits, wanted.logits, "{name} {label}: logits");
    assert_eq!(outcome.hidden, wanted.hidden, "{name} {label}: final hidden");
    let differing: Vec<_> = outcome.state.iter().zip(&wanted.state).filter(|(a, b)| a != b).collect();
    for (actual, expected) in &differing {
        eprintln!("{name} {label}: {actual} (reference {})", expected.rsplit(' ').next().unwrap());
    }
    let drafts = ["speculative_passes", "drafted_tokens", "accepted_tokens"].map(|key| (outcome.metrics.get(key), wanted.metrics.get(key)));
    if drafts.iter().any(|(a, b)| a != b) {
        eprintln!("{name} {label}: drafts {drafts:?} (this run, the reference)");
    }
    assert!(differing.is_empty() && outcome.state.len() == wanted.state.len(), "{name} {label}: state ({} parts differ)", differing.len());
    assert!(drafts.iter().all(|(a, b)| a == b), "{name} {label}: the MTP layer's drafts");
}

/// The interleaved requests and what each left, the parking's own checks along them (those of
/// bounded copies unless `whole`).
fn interleaved(model: &mut Model, name: &str, p: &Prompts, whole: bool) -> Vec<(&'static str, Outcome)> {
    let mut outcomes = Vec::new();
    fn record<'a>(outcomes: &'a mut Vec<(&'static str, Outcome)>, label: &'static str, outcome: Outcome) -> &'a Outcome {
        outcomes.push((label, outcome));
        &outcomes.last().unwrap().1
    }
    let root = &p.root;
    record(&mut outcomes, "root", go(model, root, 24, None));
    // (1) a short prompt: the conversation parked, a few positions copied, the rest borrowed
    let short_out = go(model, &p.short, 1, None);
    assert_eq!(short_out.metrics["parked"], "saved", "{name}: parked");
    let (covered, end) = model.parked_coverage().unwrap();
    let (low, high) = covered.unwrap();
    if !whole {
        assert!(low == 0 && high < 64 && end > 4900, "{name}: only the short prompt's positions copied ({low}..{high} of {end})");
        assert_eq!(short_out.metrics["park_borrowed_positions"], (end - high) as u64, "{name}: borrowed");
    }
    // (2) its continuation, past what was copied (max_new 2): the coverage grows first
    let short_next = turn(&p.short, &short_out.tokens, &root[600..640]);
    let next_out = go(model, &short_next, 2, None);
    let (grown, _) = model.parked_coverage().unwrap();
    if !whole {
        assert!(grown.unwrap().1 > high && next_out.metrics["park_copied_positions"].as_u64().unwrap() > 0, "{name}: grown");
    }
    let outcome = record(&mut outcomes, "short", short_out);
    let short_cached = outcome.metrics["cached_tokens"].clone();
    let short_tokens = outcome.tokens.clone();
    let next = record(&mut outcomes, "short-next", next_out);
    let next_digests = (next.tokens.clone(), next.logits.clone(), next.hidden.clone(), next.state.clone());
    let restored = record(&mut outcomes, "live", go(model, &p.live, 24, None));
    assert_eq!(restored.metrics["parked"], "restored", "{name}: back");
    // (5) a long prompt cancelled during its prefill (the root conversation parked again)
    let cancelled = record(&mut outcomes, "cancelled", go(model, &p.tools, 40, Some(3)));
    assert_eq!(cancelled.metrics["parked"], "saved", "{name}: parked before the cancelled prompt");
    // (what its whole batches read stays held: the next request that shares it goes on from there)
    assert!(model.held_prefix(&p.tools) >= p.tools.len().min(model.prefill_batch_size), "{name}: the cancelled prompt's batches kept");
    record(&mut outcomes, "other", go(model, &p.other, 24, None));
    // (3) resuming from the other conversation's root checkpoint (2052) below where it departs
    // (2500), max_new 3: the copy starts under the checkpoint (and the MTP layer's pair before
    // completes a pooled block from the history the checkpoint kept, not the ring's later inputs)
    let below_out = record(&mut outcomes, "below", go(model, &p.below, 3, None));
    assert_eq!((below_out.metrics["parked"].as_str(), below_out.metrics["cached_tokens"].as_u64()), (Some("saved"), Some(2052)),
               "{name}: from the root checkpoint");
    let (covered, _) = model.parked_coverage().unwrap();
    if !whole {
        assert_eq!(covered.unwrap().0, 2048, "{name}: copied from the MTP pair under the checkpoint");
    }
    record(&mut outcomes, "other-next", go(model, &p.other_next, 16, None));
    // (6) two long conversations trading places: the other parked as the tools one starts, each
    // back in turn, the live one parked whole in the other slot
    record(&mut outcomes, "tools", go(model, &p.tools, 40, None));
    record(&mut outcomes, "tools-more", go(model, &p.tools_more, 8, None));
    record(&mut outcomes, "step", go(model, &p.step, 24, None));
    let last = record(&mut outcomes, "other-last", go(model, &p.other_last, 16, None));
    assert_eq!(last.metrics["parked"], "restored", "{name}: the other back");
    let (covered, end) = model.parked_coverage().unwrap();
    assert_eq!(covered.unwrap().1, end, "{name}: the tools conversation parked whole");
    let back = record(&mut outcomes, "step-last", go(model, &p.step_last, 16, None));
    assert_eq!(back.metrics["parked"], "restored", "{name}: the tools conversation back");
    // (6, low headroom) the other conversation back once more with room for the live one's own
    // part only, not down to the lowest copy: the live one is not parked
    let (covered, _) = model.parked_coverage().unwrap();
    let (low, common) = (covered.unwrap().0, shared(&model.history, &p.other_final));
    assert!(low < common && model.history.len() - common >= 1024, "{name}: a copy below the shared {common} ({low})");
    model.limit_parking(model.history.len() - common);
    let last = record(&mut outcomes, "other-final", go(model, &p.other_final, 16, None));
    assert_eq!(last.metrics["parked"], "restored", "{name}: the other back");
    assert!(model.parked_coverage().is_none(), "{name}: the live one parked from {common}");
    // (6, low headroom) a prompt whose copies would outgrow the budget: the parked conversation
    // is dropped (bounded), and the prompt runs as it did
    model.limit_parking(1 << 20);
    let again = record(&mut outcomes, "short-again", go(model, &p.short, 1, None));
    assert_eq!((again.metrics["parked"].as_str(), &again.metrics["cached_tokens"], &again.tokens),
               (Some("saved"), &short_cached, &short_tokens), "{name}: parked again");
    let (covered, _) = model.parked_coverage().unwrap();
    let (low, high) = covered.unwrap();
    model.limit_parking(high - low);
    let dropped = record(&mut outcomes, "short-next-again", go(model, &short_next, 2, None));
    assert_eq!((&dropped.tokens, &dropped.logits, &dropped.hidden, &dropped.state),
               (&next_digests.0, &next_digests.1, &next_digests.2, &next_digests.3), "{name}: the prompt unchanged");
    if !whole {
        assert!(model.parked_coverage().is_none(), "{name}: dropped");
    }
    // (4) the prompt one past a batch: its MTP pair at 4095 written whatever row 4095 held before
    record(&mut outcomes, "pipeline-end", go(model, &p.past, 4, None));
    outcomes
}
