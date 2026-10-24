//! Header validation is CPU-only; the real-file case is opt-in, without mapping tensor payloads.
mod common;

use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;
use serde_json::Value;
use straydog_engine::arch::deepseek41::dspark::Dspark;
use straydog_gguf::{Metadata, ModelHeader, TensorInfo};

fn metadata(value: &Value) -> Metadata {
    match value {
        Value::Bool(value) => Metadata::Bool(*value),
        Value::Number(value) => value.as_i64().map_or_else(|| Metadata::Float(value.as_f64().unwrap()), Metadata::Signed),
        Value::String(value) => Metadata::Text(value.clone()),
        Value::Array(values) => Metadata::Array(values.iter().map(metadata).collect()),
        _ => panic!("unexpected fixture metadata"),
    }
}

// The names, shapes and types are an independent snapshot of the real GGUF header,
// not constructed from the loader's expected-layout table. Token strings are omitted:
// this validator checks the declared vocabulary size, not tokenizer identity.
fn headers() -> (ModelHeader, ModelHeader) {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/dspark-header.json")).unwrap();
    let make = |key: &str, architecture: &str| {
        let mut values: BTreeMap<_, _> = fixture[key].as_object().unwrap().iter()
            .map(|(key, value)| (key.clone(), metadata(value))).collect();
        values.insert("tokenizer.ggml.tokens".into(), Metadata::Array(vec![Metadata::Text(String::new()); fixture["vocabulary"].as_u64().unwrap() as usize]));
        ModelHeader { architecture: architecture.into(), declared_context: None, alignment: 32,
            data_offset: fixture["data_offset"].as_u64().unwrap(), file_bytes: fixture["file_bytes"].as_u64().unwrap(),
            shard_index: 0, shard_count: 1, tensors: Vec::new(), integer_arrays: BTreeMap::new(), metadata: values }
    };
    let mut draft = make("draft_metadata", "dflash");
    draft.tensors = fixture["tensors"].as_array().unwrap().iter().map(|value| {
        let dimensions: Vec<u64> = value["dims"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let kind = value["kind"].as_u64().unwrap() as u32;
        let (elements, block_bytes) = straydog_engine::formats::block(kind).unwrap();
        let bytes = dimensions.iter().product::<u64>() / elements as u64 * block_bytes as u64;
        TensorInfo { name: value["name"].as_str().unwrap().into(), dimensions, quant_type: kind,
            offset: draft.data_offset + value["offset"].as_u64().unwrap(), bytes }
    }).collect();
    (draft, make("target_metadata", "deepseek41"))
}

#[test]
fn captured_header_loads_borrowed_tensor_descriptors() {
    let (draft, target) = headers();
    let loaded = Dspark::from_headers(&draft, &target).unwrap();
    assert_eq!(loaded.tensors.len(), 78);
    assert!(std::ptr::eq(loaded.header, &draft));
    for tensor in &draft.tensors {
        assert!(std::ptr::eq(loaded.tensors[tensor.name.as_str()], tensor));
    }
}

#[test]
fn every_tensor_rejects_a_wrong_type_or_extent() {
    let (mut draft, target) = headers();
    for index in 0..draft.tensors.len() {
        let kind = draft.tensors[index].quant_type;
        draft.tensors[index].quant_type = if kind == 0 { 8 } else { 0 };
        let error = Dspark::from_headers(&draft, &target).unwrap_err();
        assert!(error.contains(&draft.tensors[index].name), "{error}");
        draft.tensors[index].quant_type = kind;
        draft.tensors[index].dimensions[0] += 1;
        let error = Dspark::from_headers(&draft, &target).unwrap_err();
        assert!(error.contains(&draft.tensors[index].name), "{error}");
        draft.tensors[index].dimensions[0] -= 1;
    }
}

#[test]
fn tensor_names_are_an_exact_set() {
    let (mut draft, target) = headers();
    let name = draft.tensors[0].name.clone();
    draft.tensors[0].name = "unknown.weight".into();
    assert!(Dspark::from_headers(&draft, &target).unwrap_err().contains(&name));
    draft.tensors[0].name = draft.tensors[1].name.clone();
    assert!(Dspark::from_headers(&draft, &target).unwrap_err().contains("duplicate"));
    draft.tensors[0].name = name;
    let tensor = draft.tensors.pop().unwrap();
    assert!(Dspark::from_headers(&draft, &target).unwrap_err().contains("78"));
    draft.tensors.push(tensor.clone());
    let mut extra = tensor;
    extra.name = "unexpected.weight".into();
    draft.tensors.push(extra);
    assert!(Dspark::from_headers(&draft, &target).unwrap_err().contains("78"));
}

#[test]
fn incompatible_target_and_draft_metadata_are_rejected() {
    let (mut draft, mut target) = headers();
    for key in ["dspark.block_size", "dspark.noise_token_id", "dspark.markov_rank", "dspark.expert_count", "dspark.expert_used_count"] {
        let key = format!("deepseek41.{key}");
        let original = target.metadata[&key].clone();
        target.metadata.insert(key.clone(), Metadata::Signed(original.integer().unwrap() + 1));
        assert!(Dspark::from_headers(&draft, &target).unwrap_err().contains(&key));
        target.metadata.insert(key, original);
    }
    target.metadata.insert("deepseek41.dspark.target_layers".into(), Metadata::Array(vec![Metadata::Signed(36); 3]));
    assert!(Dspark::from_headers(&draft, &target).unwrap_err().contains("target layers"));
    let (_, target) = headers();
    for key in ["sample_from_anchor", "has_confidence_head", "hyper_connection.single_pass"] {
        let key = format!("dflash.{key}");
        draft.metadata.insert(key.clone(), Metadata::Bool(false));
        assert!(Dspark::from_headers(&draft, &target).unwrap_err().contains(&key));
        draft.metadata.insert(key, Metadata::Bool(true));
    }
    draft.metadata.remove("dflash.attention.key_length");
    assert!(Dspark::from_headers(&draft, &target).unwrap_err().contains("dflash.attention.key_length"));
}

#[test]
fn real_dspark_file_matches_the_layout_when_requested() {
    let Some(path) = common::model_env("STRAYDOG_DSPARK", "the DSpark Q8_0 GGUF file") else { return };
    let draft = straydog_gguf::inspect(&mut File::open(path).unwrap()).unwrap();
    let (_, captured_target) = headers();
    let target = if let Some(path) = std::env::var_os("STRAYDOG_DEEPSEEK") {
        let mut path = PathBuf::from(path);
        if path.is_dir() {
            let mut paths: Vec<_> = std::fs::read_dir(&path).unwrap().map(|entry| entry.unwrap().path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "gguf")).collect();
            paths.sort();
            path = paths.into_iter().next().expect("no target GGUF shard");
        }
        straydog_gguf::inspect(&mut File::open(path).unwrap()).unwrap()
    } else {
        eprintln!("target metadata: captured fixture (set STRAYDOG_DEEPSEEK for the real target header)");
        captured_target
    };
    assert_eq!(Dspark::from_headers(&draft, &target).unwrap().tensors.len(), 78);
}
