//! Authenticated weights equal to the reference's (design phase R3): block decoding, the
//! tensor table, vectors, embedding rows and PLE n-gram rows.
//!
//! The model cases need the bundle the golden file was made from: set STRAYDOG_MODEL.

use std::path::PathBuf;

use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use straydog_engine::formats::{self, IQ_GRIDS};
use straydog_engine::weights::{ngram_rows, Weights};

fn golden() -> Json {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/weights.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn digest(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    format!("{:x}", Sha256::digest(&bytes))
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len()).step_by(2).map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap()).collect()
}

#[test]
fn iq_grids_and_block_decoding_equal_the_reference() {
    let golden = golden();
    assert_eq!(format!("{:x}", Sha256::digest(IQ_GRIDS)), golden["iq_grids_sha256"].as_str().unwrap());
    for case in golden["blocks"].as_array().unwrap() {
        let kind = case["kind"].as_u64().unwrap() as u32;
        let mut decoded = Vec::new();
        formats::decode_into(kind, &hex(case["data"].as_str().unwrap()), &mut decoded).unwrap();
        let bytes: Vec<u8> = decoded.iter().flat_map(|value| value.to_le_bytes()).collect();
        assert_eq!(bytes, hex(case["f32"].as_str().unwrap()), "kind {kind}");
    }
}

#[test]
fn model_tensors_vectors_and_rows_equal_the_reference() {
    let golden = golden();
    let Some(root) = std::env::var_os("STRAYDOG_MODEL") else {
        eprintln!("skipped: set STRAYDOG_MODEL to the bundle of tests/golden/weights.json");
        return;
    };
    let model = &golden["model"];
    let weights = Weights::open(&PathBuf::from(root), model["quantization"].as_str().unwrap(), None, 0,
                                &|_, _, _| {}).unwrap();
    let mut names: Vec<&String> = weights.names().iter().collect();
    names.sort();
    let table: Vec<Json> = names.iter().map(|name| {
        let tensor = weights.tensor(name).unwrap();
        serde_json::json!([name, tensor.file, tensor.dims, tensor.kind, tensor.offset, tensor.bytes])
    }).collect();
    assert_eq!(Json::Array(table), model["tensors"]);
    assert_eq!(serde_json::to_value(&weights.integer_arrays).unwrap(), model["integer_arrays"]);
    for (name, expected) in model["vectors"].as_object().unwrap() {
        assert_eq!(digest(&weights.vector(name).unwrap().values), expected.as_str().unwrap(), "{name}");
    }
    let rows = model["token_rows"].as_object().unwrap();
    let mut ids = Vec::new();
    for (key, expected) in rows {
        if let Ok(id) = key.parse::<i64>() {
            ids.push(id);
            assert_eq!(digest(&weights.rows("token_embd.weight", &[id]).unwrap()), expected.as_str().unwrap());
        }
    }
    assert_eq!(digest(&weights.rows("token_embd.weight", &ids).unwrap()), rows["batch"].as_str().unwrap());
    // Cached rows are the decoded rows.
    assert_eq!(digest(&weights.rows("token_embd.weight", &ids[..4]).unwrap()),
               digest(&weights.rows("token_embd.weight", &ids[..4]).unwrap()));
    for case in model["ple"].as_array().unwrap() {
        let history: Vec<i64> = serde_json::from_value(case["history"].clone()).unwrap();
        let tokens: Vec<i64> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let indices = ngram_rows(&history, &tokens, &weights.integer_arrays).unwrap();
        let expected: Vec<Vec<i64>> = serde_json::from_value(case["indices"].clone()).unwrap();
        assert_eq!(indices, expected.concat());
        let embedded = weights.rows("per_layer_token_embd.weight", &indices).unwrap();
        assert_eq!(digest(&embedded), case["embeddings"].as_str().unwrap());
    }
    weights.check_unchanged().unwrap();
}
