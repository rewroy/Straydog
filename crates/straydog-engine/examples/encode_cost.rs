//! Where a long agent conversation's request spends its time before the engine runs: the messages' defusing, the chat template's
//! rendering and the encoding, for DeepSeek-V4.1's tokenizer and template over a conversation of tool calls built from this repository's
//! files, at a few lengths, each step timed twice (the second as the next request of the same conversation finds the memo); then, for a
//! body with 26 screenshots, parsing it, writing it to the worker, cloning it and the images' digests.
//!
//! cargo run --release -p straydog-engine --example encode_cost -- FIRST_SHARD.gguf

use std::time::Instant;

use straydog_engine::chat::{defuse_messages, ChatTemplate};
use straydog_engine::pyjson::{self, Value};
use straydog_engine::text::Text;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [shard] = &args[..] else { return Err("usage: encode_cost FIRST_SHARD.gguf".into()) };
    // (the header only: the shard is tens of GB)
    let mut file = std::io::BufReader::new(std::fs::File::open(shard).map_err(|exc| exc.to_string())?);
    let header = straydog_gguf::inspect(&mut file).map_err(|exc| exc.to_string())?;
    let text = Text::from_gguf(&header).map_err(|exc| exc.0)?;
    let template = ChatTemplate::new(straydog_engine::arch::deepseek41::CHAT_TEMPLATE).map_err(|exc| exc.0)?;
    // (tool outputs: this repository's sources in pieces of a few sizes; reasoning: its design notes)
    let mut sources = String::new();
    for entry in walk(std::path::Path::new("crates")) {
        if entry.extension().is_some_and(|ext| ext == "rs") {
            sources.push_str(&std::fs::read_to_string(&entry).unwrap_or_default());
        }
    }
    let prose = std::fs::read_to_string("docs/DESIGN.md").map_err(|exc| exc.to_string())?;
    let sizes = [600, 1500, 3000, 6000, 12000, 30000];
    let take = |source: &str, at: &mut usize, size: usize| -> String {
        let mut start = *at % source.len().saturating_sub(size + 8).max(1);
        while !source.is_char_boundary(start) { start += 1; }
        let mut end = (start + size).min(source.len());
        while !source.is_char_boundary(end) { end -= 1; }
        *at += size * 7 + 13;
        source[start..end].to_string()
    };
    let system = format!("You are an agent.\n{}", take(&prose, &mut 0, 12000));
    let mut messages = vec![
        format!(r#"{{"role": "system", "content": {}}}"#, quote(&system)),
        format!(r#"{{"role": "user", "content": {}}}"#, quote("Please look through the engine and tell me where the prompt's time goes.")),
    ];
    let tools = pyjson::parse(r#"[{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]"#)
        .map_err(|exc| format!("{exc:?}"))?;
    let (mut code_at, mut prose_at) = (0usize, 0usize);
    let mut round = 0usize;
    let targets = [50_000usize, 100_000, 150_000, 190_000];
    let mut target = 0;
    println!("tokens   messages  defuse ms  render ms  encode ms (first, again)");
    while target < targets.len() {
        let reasoning = take(&prose, &mut prose_at, 600 + (round * 379) % 2400);
        let id = format!("call_{round}");
        messages.push(format!(
            r#"{{"role": "assistant", "content": {}, "reasoning_content": {}, "tool_calls": [{{"id": "{id}", "type": "function", "function": {{"name": "read_file", "arguments": {}}}}}]}}"#,
            quote(&take(&prose, &mut prose_at, 120 + (round * 53) % 300)), quote(&reasoning),
            quote(&format!(r#"{{"path": "crates/straydog-engine/src/file_{round}.rs"}}"#))));
        messages.push(format!(r#"{{"role": "tool", "tool_call_id": "{id}", "content": {}}}"#,
                              quote(&take(&sources, &mut code_at, sizes[(round * 7 + round / 3) % sizes.len()]))));
        round += 1;
        if round % 4 != 0 {
            continue;
        }
        let body = pyjson::parse(&format!("[{}]", messages.join(","))).map_err(|exc| format!("{exc:?}"))?;
        let variables = [("enable_thinking", Value::Bool(true)), ("reasoning_effort", Value::Int(75))];
        let started = Instant::now();
        let defused = defuse_messages(&body, &|own| text.defuse(own).into_owned());
        let defuse = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let prompt = template.render_with(&defused, &tools, &variables).map_err(|exc| exc.0)?;
        let render = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let ids = text.encode(&prompt).map_err(|exc| exc.0)?;
        let first = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let again = text.encode(&prompt).map_err(|exc| exc.0)?;
        let second = started.elapsed().as_secs_f64() * 1e3;
        assert_eq!(ids, again);
        if ids.len() < targets[target] {
            continue;
        }
        target += 1;
        println!("{:>7}  {:>8}  {defuse:>9.1}  {render:>9.1}  {first:>9.1} {second:>9.1}", ids.len(), messages.len());
    }
    // A request's bytes beside the text: 26 screenshots of a few sizes as base64 data URLs, as an agent's page checks send them with every
    // request. What each step costs: parsing the body, writing it, cloning it, the images' digests.
    for size in [150_000usize, 400_000, 1_000_000] {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let images: Vec<Vec<u8>> = (0..26).map(|_| (0..size).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed as u8 }).collect()).collect();
        let parts: Vec<String> = images.iter().map(|bytes| format!(r#"{{"type": "image_url", "image_url": {{"url": "data:image/png;base64,{}"}}}}"#, encode64(bytes))).collect();
        let body = format!(r#"{{"messages": [{}, {{"role": "user", "content": [{}]}}]}}"#, messages.join(","), parts.join(","));
        let megabytes = body.len() as f64 / 1e6;
        let started = Instant::now();
        let parsed = pyjson::parse(&body).map_err(|exc| format!("{exc:?}"))?;
        let parse = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let written = pyjson::dumps(&parsed);
        let dumps = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let cloned = parsed.clone();
        let clone = started.elapsed().as_secs_f64() * 1e3;
        use sha2::Digest;
        let started = Instant::now();
        let digests: Vec<[u8; 32]> = images.iter().map(|bytes| sha2::Sha256::digest(bytes).into()).collect();
        let digest = started.elapsed().as_secs_f64() * 1e3;
        assert!(digests.len() == images.len() && !written.is_empty() && matches!(cloned, Value::Object(_)));
        println!("images 26 x {} KB, body {megabytes:.1} MB: parse {parse:.1} ms, dumps {dumps:.1}, clone {clone:.1}, digests {digest:.1}", size / 1000);
    }
    Ok(())
}

fn encode64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8) | u32::from(*chunk.get(2).unwrap_or(&0));
        for index in 0..4 {
            out.push(if index <= chunk.len() { ALPHABET[(n >> (18 - 6 * index) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

fn quote(text: &str) -> String {
    let mut out = String::new();
    pyjson::write_string(&mut out, text);
    out
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() { found.extend(walk(&path)); } else { found.push(path); }
    }
    found.sort();
    found
}
