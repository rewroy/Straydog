//! Alternating A/B of two Straydog builds, a baseline and a candidate, both driven through the
//! same runtime and worker protocol, under the GPU lock:
//!
//! ~/benchlock.sh ab -- cargo run --release --features hip --example ab -- --baseline BIN \
//!   --model DIR [--drafter DIR] --devices 0,1 --vram 30 --ctx 262144 --rounds 2 --output ab.json
//!
//! Each round starts each build in turn (the order alternates), runs the single-request
//! cases of bench/cases.json twice (the second counts) and a multi-turn agent session over
//! this repository at a fixed commit, then stops it. Timings are the engines' own.

use std::path::{Path, PathBuf};
use std::process::Command as Process;
use std::time::Duration;

use clap::Parser;
use futures::StreamExt;
use serde_json::{json, Value};
use straydog::config::{self, Options};
use straydog::runtime::{self, Runtime};
use straydog_api::{Engine, EngineState};
use straydog_engine::pyjson;
use straydog_engine::reply::{ReasoningSplitter, ToolCallSplitter};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    drafter: Option<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "0,1")]
    devices: Vec<usize>,
    #[arg(long, default_value_t = 30.0)]
    vram: f64,
    #[arg(long, default_value_t = 262144)]
    ctx: u32,
    #[arg(long, default_value_t = 2)]
    rounds: usize,
    #[arg(long)]
    output: PathBuf,
    /// The repository root (bench/cases.json, and the session's files)
    #[arg(long, default_value = ".")]
    repo: PathBuf,
    /// The executable measured against
    #[arg(long)]
    baseline: PathBuf,
    /// The executable measured
    #[arg(long, default_value = "target/release/straydog")]
    candidate: PathBuf,
    /// Skip the multi-turn session
    #[arg(long)]
    no_session: bool,
    /// Only these cases (names in bench/cases.json)
    #[arg(long, value_delimiter = ',')]
    cases: Option<Vec<String>>,
    /// Measured runs of each case (after one warm-up run)
    #[arg(long, default_value_t = 1)]
    repeat: usize,
}

const SESSION_COMMIT: &str = "9ee90c6";
const SESSION_TURNS: [&str; 3] = [
    "Read crates/straydog-engine/src/pyjson.rs and explain how float_repr chooses between positional and exponent \
     notation.",
    "Where does crates/straydog/src/runtime.rs cancel a generation whose client went away? Find it and summarize the \
     steps.",
    "Which file declares the GPU library's C ABI, and how many functions does it declare? Check by reading it.",
];

async fn run(engine: &Runtime, body: &pyjson::Value) -> (String, Value) {
    let mut generation = engine.generate(body.clone()).await.unwrap();
    let mut text = String::new();
    while let Some(event) = generation.events.next().await {
        match event["event"].as_str().unwrap() {
            "delta" => text += event["content"].as_str().unwrap(),
            "done" => return (text, event),
            "error" => panic!("{event}"),
            _ => {}
        }
    }
    panic!("the generation ended without a terminal event");
}

fn git(repo: &Path, args: &[&str]) -> String {
    let output = Process::new("git").arg("-C").arg(repo).args(args).output().unwrap();
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Read-only answers from the repository at SESSION_COMMIT: the same calls get the same results.
fn tool(repo: &Path, name: &str, arguments: &Value) -> String {
    let text = |key: &str| arguments[key].as_str().unwrap_or("").to_string();
    let result = match name {
        "read_file" => git(repo, &["show", &format!("{SESSION_COMMIT}:{}", text("path"))]),
        "list_directory" => git(repo, &["ls-tree", "--name-only", SESSION_COMMIT, &format!("{}/", text("path").trim_end_matches('/'))]),
        "search" => git(repo, &["grep", "-n", "-e", &text("pattern"), SESSION_COMMIT]).lines().take(60)
            .map(|line| line.trim_start_matches(&format!("{SESSION_COMMIT}:")).chars().take(240).collect::<String>())
            .collect::<Vec<_>>().join("\n"),
        _ => format!("error: {name} is not available"),
    };
    if result.is_empty() { "error: not found".into() } else { result.chars().take(24000).collect() }
}

fn session_tools() -> Value {
    let function = |name: &str, description: &str, parameter: &str| json!({"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {parameter: {"type": "string"}}, "required": [parameter]}}});
    json!([function("read_file", "Read a file of the repository", "path"),
           function("list_directory", "List a directory of the repository", "path"),
           function("search", "Search the repository for a regular expression", "pattern")])
}

async fn session(engine: &Runtime, repo: &Path, record: &mut Vec<Value>) {
    let system = "You are a careful coding assistant working in the Straydog repository, a Rust inference engine. \
                  Use the tools to read code before answering; answer concisely.";
    let mut messages = vec![json!({"role": "system", "content": system})];
    for (turn, question) in SESSION_TURNS.iter().enumerate() {
        messages.push(json!({"role": "user", "content": question}));
        for iteration in 0..6 {
            let body = json!({"messages": messages, "tools": session_tools(), "max_tokens": 2048, "thinking": true,
                              "reasoning_effort": "medium"});
            let body = pyjson::parse(&serde_json::to_string(&body).unwrap()).unwrap();
            let (raw, done) = run(engine, &body).await;
            let tools = pyjson::parse(&session_tools().to_string()).unwrap();
            let (mut reasoning, mut calls) = (ReasoningSplitter::new(true), ToolCallSplitter::new(Some(&tools)));
            let (content, thought) = reasoning.feed(&raw);
            let mut answer = calls.feed(&content);
            let thought = thought + &reasoning.finish();
            let (late, calls) = calls.finish();
            answer += &late;
            record.push(json!({"turn": turn, "iteration": iteration, "usage": done["usage"], "timings": done["timings"]}));
            let mut message = json!({"role": "assistant", "content": answer, "reasoning_content": thought});
            if calls.is_empty() {
                messages.push(message);
                break;
            }
            message["tool_calls"] = calls.iter().map(|call| json!({"id": call.id, "type": "function",
                "function": {"name": call.name, "arguments": call.arguments}})).collect();
            messages.push(message);
            for call in &calls {
                let arguments: Value = serde_json::from_str(&call.arguments).unwrap_or(Value::Null);
                messages.push(json!({"role": "tool", "tool_call_id": call.id, "content": tool(repo, &call.name, &arguments)}));
            }
        }
    }
}

fn main() {
    let args = Args::parse();
    let options = Options {
        model: Some(args.model.clone()),
        model_draft: args.drafter.clone(),
        gpu: Some(args.devices.clone()),
        vram: Some(args.vram),
        ctx_size: Some(args.ctx),
        ..Options::default()
    };
    let settings = config::resolve(options).unwrap();
    let cases: Vec<Value> =
        serde_json::from_str::<Vec<Value>>(&std::fs::read_to_string(args.repo.join("bench/cases.json")).unwrap())
            .unwrap()
            .into_iter()
            .filter(|case| args.cases.as_ref().is_none_or(|names| names.iter().any(|name| case["name"] == name.as_str())))
            .collect();
    let repo = args.repo.canonicalize().unwrap();
    let mut report = json!({"settings": format!("{settings:?}"), "rounds": []});
    let tokio = tokio::runtime::Runtime::new().unwrap();
    for round in 0..args.rounds {
        let order = if round % 2 == 0 { ["baseline", "candidate"] } else { ["candidate", "baseline"] };
        for engine_name in order {
            let executable = if engine_name == "baseline" { &args.baseline } else { &args.candidate };
            let command = (executable.canonicalize().unwrap(), runtime::worker_args(&settings));
            let result = tokio.block_on(async {
                let started = std::time::Instant::now();
                let engine = Runtime::start(&settings, command);
                while matches!(engine.state(), EngineState::Loading { .. }) {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                let EngineState::Ready { .. } = engine.state() else { panic!("{engine_name}: {:?}", engine.state()) };
                let startup = started.elapsed().as_secs_f64();
                let mut results = Vec::new();
                for case in &cases {
                    // Each run gets its own first line, so no prefix is ever reused: every prefill
                    // starts from nothing. The first run warms the GPU; the second counts.
                    let body = |run: usize| {
                        let mut body = case.clone();
                        body.as_object_mut().unwrap().remove("name");
                        let content = body["messages"][0]["content"].as_str().unwrap().to_string();
                        body["messages"][0]["content"] = json!(format!("Run {run}.\n{content}"));
                        pyjson::parse(&body.to_string()).unwrap()
                    };
                    run(&engine, &body(0)).await;
                    for repeat in 1..=args.repeat {
                        let (text, done) = run(&engine, &body(repeat)).await;
                        results.push(json!({"name": case["name"], "content": text, "usage": done["usage"],
                                            "timings": done["timings"]}));
                    }
                }
                let mut turns = Vec::new();
                if !args.no_session {
                    session(&engine, &repo, &mut turns).await;
                }
                engine.close().await;
                json!({"engine": engine_name, "round": round, "startup_seconds": startup, "cases": results,
                       "session": turns})
            });
            eprintln!("{}", summary(&result));
            report["rounds"].as_array_mut().unwrap().push(result);
            std::fs::write(&args.output, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        }
    }
}

/// One line per case and the session's figures.
fn summary(result: &Value) -> String {
    let mut lines = vec![format!("{} round {}: startup {:.1} s", result["engine"], result["round"],
                                 result["startup_seconds"].as_f64().unwrap_or(0.0))];
    for case in result["cases"].as_array().unwrap() {
        let timings = &case["timings"];
        lines.push(format!("  {:<9} prefill {:>7.1} tok/s  ttft {:>6.3} s  decode {:>6.1} tok/s",
                           case["name"].as_str().unwrap(), timings["prefill_tokens_per_second"].as_f64().unwrap_or(0.0),
                           timings["ttft_seconds"].as_f64().unwrap_or(0.0),
                           timings["decode_tokens_per_second"].as_f64().unwrap_or(0.0)));
    }
    let turns = result["session"].as_array().unwrap();
    if !turns.is_empty() {
        let (mut follow, mut short, mut long) = (Vec::new(), Vec::new(), Vec::new());
        for turn in turns.iter().skip(1) {
            follow.push(turn["timings"]["ttft_seconds"].as_f64().unwrap_or(0.0));
            let rate = turn["timings"]["decode_tokens_per_second"].as_f64();
            let tokens = turn["usage"]["completion_tokens"].as_u64().unwrap_or(0);
            if let Some(rate) = rate {
                if tokens < 128 { short.push(rate) } else { long.push(rate) }
            }
        }
        let mean = |values: &[f64]| if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 };
        lines.push(format!("  session: {} requests, first ttft {:.2} s, follow-up ttft {:.2} s, decode short {:.1} / long {:.1} tok/s",
                           turns.len(), turns[0]["timings"]["ttft_seconds"].as_f64().unwrap_or(0.0), mean(&follow),
                           mean(&short), mean(&long)));
    }
    lines.join("\n")
}
