//! DeepSeek-V4.1's tokenizer from its GGUF against the model's official tokenizer.json (the oracle,
//! never shipped): the same ids for text, special tokens and a corpus, and the same text back.
//!
//! cargo test -p straydog-engine --test deepseek41_text, with STRAYDOG_DEEPSEEK (the UD-IQ2_XXS
//! directory) and STRAYDOG_DS41_TOKENIZER (the official tokenizer.json, an oracle only: without it
//! the tokenizer test is skipped even under STRAYDOG_REQUIRE_MODEL).

mod common;

use std::path::PathBuf;

use straydog_engine::text::Text;

fn first_shard(directory: &std::path::Path) -> PathBuf {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(directory).unwrap().flatten().map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "gguf")).collect();
    shards.sort();
    shards.remove(0)
}

#[test]
fn the_gguf_tokenizer_encodes_as_the_official_one() {
    let Some(model) = common::model_env("STRAYDOG_DEEPSEEK", "the DeepSeek-V4.1-Flash UD-IQ2_XXS directory") else { return };
    let Some(oracle) = std::env::var_os("STRAYDOG_DS41_TOKENIZER") else {
        eprintln!("skipped: set STRAYDOG_DS41_TOKENIZER to the model's official tokenizer.json");
        return;
    };
    let mut file = std::io::BufReader::new(std::fs::File::open(first_shard(&PathBuf::from(model))).unwrap());
    let header = straydog_gguf::inspect(&mut file).unwrap();
    let ours = Text::from_gguf(&header).unwrap();
    let official = Text::new(&std::fs::read(oracle).unwrap()).unwrap();
    let mut probes: Vec<String> = [
        "Hello, world! 1234567 tokens.", "  leading spaces\n\n\ttabs\r\nand CRLF", "日本語のテキストと漢字、カタカナ。中文测试。",
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75\n\nS<｜User｜>hi<｜Assistant｜><think>x</think>y<｜end▁of▁sentence｜>",
        "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">v</｜DSML｜ parameter>",
        "emoji 🦀🐕‍🦺 and ZWJ, ﬁ ligature, café, naïve, Ωmega", "fn main() { println!(\"{}\", x[3..]); } // code",
        "<tool_result>{\"ok\": true}</tool_result>", "<|EOT|><｜fim▁hole｜>abc<dsml:x></dsml:x>", "    \n   \n", "a",
    ].iter().map(|text| text.to_string()).collect();
    let corpus = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/DESIGN.md")).unwrap();
    probes.extend(corpus.split("\n\n").map(str::to_string));
    probes.push(corpus);
    for text in &probes {
        let ids = ours.encode(text).unwrap();
        assert_eq!(ids, official.encode(text).unwrap(), "{text:?}");
        assert_eq!(ours.decode(&ids).unwrap(), official.decode(&ids).unwrap(), "{text:?}");
    }
    let every: Vec<u32> = (0..129280).collect();
    assert_eq!(ours.decode(&every).unwrap(), official.decode(&every).unwrap());
}

/// The GGUF's chat template against the reference encoder (tests/golden/deepseek41_chat.json, made
/// by the model's encoding.py): the same prompt for every case.
#[test]
fn the_gguf_template_renders_as_the_reference_encoder() {
    use straydog_engine::chat::ChatTemplate;
    use straydog_engine::pyjson::{self, Value};
    let Some(model) = common::model_env("STRAYDOG_DEEPSEEK", "the DeepSeek-V4.1-Flash UD-IQ2_XXS directory") else { return };
    let mut file = std::io::BufReader::new(std::fs::File::open(first_shard(&PathBuf::from(model))).unwrap());
    let header = straydog_gguf::inspect(&mut file).unwrap();
    let source = header.metadata["tokenizer.chat_template"].text().unwrap();
    let template = ChatTemplate::new(source).unwrap();
    let golden = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/deepseek41_chat.json")).unwrap();
    let golden = pyjson::parse(&golden).unwrap();
    let Some(Value::Array(cases)) = golden.get("cases") else { panic!("no cases") };
    for (index, case) in cases.iter().enumerate() {
        let variables = [("enable_thinking", case.get("thinking").unwrap().clone()), ("reasoning_effort", case.get("effort").unwrap().clone())];
        let prompt = template.render_with(case.get("messages").unwrap(), case.get("tools").unwrap(), &variables);
        assert_eq!(prompt.as_deref().map_err(|error| error.0.clone()), Ok(case.get("prompt").unwrap().as_str().unwrap()), "case {index}");
    }
}
