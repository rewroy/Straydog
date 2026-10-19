//! The reference's generations reproduced bit for bit on real GPUs (design phase R3): for
//! each placement, every request's greedy tokens, metrics, logits and final hidden state.
//!
//! cargo test -p straydog-engine --features hip --test model_gpu, with STRAYDOG_MODEL and
//! STRAYDOG_DRAFTER (STRAYDOG_PLACEMENT runs one placement); under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

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

#[test]
fn generations_equal_the_reference() {
    let (Some(model_root), Some(drafter)) = (std::env::var_os("STRAYDOG_MODEL"), std::env::var_os("STRAYDOG_DRAFTER"))
    else {
        eprintln!("skipped: set STRAYDOG_MODEL and STRAYDOG_DRAFTER");
        return;
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/model.json");
    let golden: Json = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let only = std::env::var("STRAYDOG_PLACEMENT").ok();
    for (name, expected) in golden["placements"].as_object().unwrap() {
        if only.as_ref().is_some_and(|only| only != name) {
            continue;
        }
        let placement = &expected["placement"];
        eprintln!("placement {name}");
        let devices: Vec<usize> = serde_json::from_value(placement["devices"].clone()).unwrap();
        let budget = (placement["vram_gib"].as_f64().unwrap() * (1u64 << 30) as f64) as u64;
        let split = if placement["split_mode"] == "layer" { SplitMode::Layer } else { SplitMode::Experts };
        let speculative = placement["speculative"].as_u64().unwrap() as usize;
        let group = Group::new(&devices, budget, split).unwrap();
        group.configure_gemm(2, true).unwrap();
        let weights = Weights::open(&PathBuf::from(&model_root), golden["quantization"].as_str().unwrap(),
                                    (speculative > 0).then(|| PathBuf::from(&drafter)).as_deref(),
                                    straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
        let settings = Settings {
            context: placement["context"].as_u64().unwrap() as usize,
            state_budget_bytes: 13 << 30,
            prefill_batch_size: 4096,
            speculative,
            cpu_threads: 0,
        };
        let mut model = Model::new(weights, group, &settings).unwrap();
        assert_eq!(model.resident, expected["resident"].as_bool().unwrap(), "{name}: resident");
        assert_eq!(model.partial, expected["partial"].as_bool().unwrap(), "{name}: partial");
        assert_eq!(model.host_kv, expected["host_kv"].as_bool().unwrap(), "{name}: host K/V");
        assert_eq!(model.prefill_batch_size as u64, expected["prefill_batch_size"].as_u64().unwrap(), "{name}: batch");
        assert_eq!(model.speculative as u64, expected["speculative"].as_u64().unwrap(), "{name}: speculative");
        assert_eq!(model.state_bytes, expected["state_bytes"].as_u64().unwrap(), "{name}: state bytes");
        for request in expected["requests"].as_array().unwrap() {
            let label = format!("{name}/{}", request["name"].as_str().unwrap());
            let prompt: Vec<u32> = serde_json::from_value(request["prompt"].clone()).unwrap();
            // Partial residency swaps expert slots asynchronously during decode, so whether an
            // expert runs on the GPU or the CPU (the same F32 weights, summed in another order)
            // depends on timing, and the reference does not reproduce its own decode bit for bit.
            // There only a request whose state comes from prefill alone (a new sequence, or the
            // system-prompt checkpoint) is compared: its prompt logits and its tokens.
            let resumed = request["metrics"]["resumed_from"].as_str().unwrap();
            let fresh = request["metrics"]["cached_tokens"].as_u64() == Some(0);
            let comparable = !model.partial || fresh || resumed == "root";
            model.logits_digests = Some(Vec::new());
            let mut tokens = Vec::new();
            model.generate(&prompt, request["max_new"].as_u64().unwrap() as usize, 0.0, 0, &|| false, &mut |_, _| {},
                           request["turn"].as_bool().unwrap(), request["checkpoint"].as_bool().unwrap(),
                           &mut |token| tokens.push(token)).unwrap();
            if !comparable {
                eprintln!("  {label}: run (state from decode under partial residency)");
                continue;
            }
            let expected_tokens: Vec<u32> = serde_json::from_value(request["tokens"].clone()).unwrap();
            assert_eq!(tokens, expected_tokens, "{label}: tokens");
            for (key, value) in request["metrics"].as_object().unwrap() {
                let actual = model.last_metrics.get(key).cloned().unwrap_or(Json::Null);
                assert_eq!(actual.as_f64().or(actual.as_str().map(|_| 0.0)), value.as_f64().or(value.as_str().map(|_| 0.0)),
                           "{label}: {key}");
                if let Some(text) = value.as_str() {
                    assert_eq!(actual.as_str(), Some(text), "{label}: {key}");
                }
            }
            let digests: Vec<String> = serde_json::from_value(request["logits"].clone()).unwrap();
            let actual = model.logits_digests.take().unwrap();
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
}
