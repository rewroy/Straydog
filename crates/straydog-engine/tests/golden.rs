//! The reference's renders, token ids and decodes, reproduced byte for byte (design phase R2).
//!
//! The golden file comes from `scripts/golden.py`. Chat cases need the pinned tokenizer and
//! template: set STRAYDOG_MODEL to a model bundle directory.

use std::path::PathBuf;

use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use straydog_engine::chat::ChatTemplate;
use straydog_engine::pyjson;
use straydog_engine::text::Text;

fn golden() -> Json {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/chat.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The pinned resources the golden file was made from, or None (the test is skipped).
fn resources(golden: &Json) -> Option<(ChatTemplate, Text)> {
    let Some(model) = std::env::var_os("STRAYDOG_MODEL") else {
        eprintln!("skipped: set STRAYDOG_MODEL to a model bundle directory");
        return None;
    };
    let read = |name: &str| {
        let data = std::fs::read(PathBuf::from(&model).join(name)).unwrap();
        let pin = golden["resources"][name].as_str().unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&data)), pin, "{name} is not the pinned file");
        data
    };
    let template = ChatTemplate::new(std::str::from_utf8(&read("chat_template.jinja")).unwrap()).unwrap();
    Some((template, Text::new(&read("tokenizer.json")).unwrap()))
}

#[test]
fn renders_and_token_ids_equal_the_reference() {
    let golden = golden();
    let Some((template, text)) = resources(&golden) else { return };
    for case in golden["chats"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let body = pyjson::parse(case["request"].as_str().unwrap()).unwrap();
        let thinking = body.get("thinking").is_some_and(pyjson::Value::truthy);
        let effort = body.get("reasoning_effort").and_then(pyjson::Value::as_str).unwrap_or("medium");
        let tools = body.get("tools").cloned().unwrap_or(pyjson::Value::Null);
        let result = template.render(body.get("messages").unwrap(), &tools, thinking, effort);
        match case.get("error") {
            Some(expected) => assert_eq!(result.unwrap_err().0, expected.as_str().unwrap(), "{name}"),
            None => {
                let rendered = result.unwrap_or_else(|exc| panic!("{name}: {exc}"));
                assert_eq!(rendered, case["render"].as_str().unwrap(), "{name}: render");
                let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
                assert_eq!(text.encode(&rendered).unwrap(), ids, "{name}: ids");
            }
        }
    }
}

#[test]
fn decoding_and_streaming_equal_the_reference() {
    let golden = golden();
    let Some((_, text)) = resources(&golden) else { return };
    for case in golden["decodes"].as_array().unwrap() {
        let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        let expected = case["text"].as_str().unwrap();
        assert_eq!(text.decode(&ids).unwrap(), expected);
        // Every cut point, as a generation could end there.
        for end in 0..=ids.len() {
            let mut stream = text.stream();
            let mut joined = String::new();
            for &id in &ids[..end] {
                joined += &stream.push(id).unwrap();
                assert!(!joined.ends_with('\u{fffd}') || expected.contains('\u{fffd}'));
            }
            joined += &stream.finish().unwrap();
            assert_eq!(joined, text.decode(&ids[..end]).unwrap());
        }
    }
}

#[test]
fn json_round_trips_equal_python() {
    for case in golden()["json"].as_array().unwrap() {
        let source = case["text"].as_str().unwrap();
        match case.get("dumps") {
            Some(expected) => assert_eq!(pyjson::dumps(&pyjson::parse(source).unwrap()), expected.as_str().unwrap(),
                                         "{source}"),
            None => assert!(pyjson::parse(source).is_err(), "{source} should be refused"),
        }
    }
}

#[test]
fn float_reprs_equal_python() {
    for pair in golden()["floats"].as_array().unwrap() {
        let number = f64::from_bits(pair[0].as_u64().unwrap());
        assert_eq!(pyjson::float_repr(number), pair[1].as_str().unwrap(), "{number:?}");
    }
}
