//! The worker behind the runtime on real GPUs: startup, a streamed generation, the
//! conversation cache across requests, interruption, a refused request and a cancelled one.
//!
//! STRAYDOG_MODEL (and optionally STRAYDOG_DRAFTER) under the GPU lock:
//! cargo test -p straydog --features hip --test runtime_gpu. This test binary also serves as
//! the worker (run with `worker ...`), started unisolated: isolation is tested on its own.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use futures::StreamExt;
use straydog::config::{self, Options};
use straydog::runtime::{self, Runtime};
use straydog_api::{Engine, EngineState};
use straydog_engine::pyjson;

#[global_allocator]
static ALLOCATOR: straydog_engine::alloc::HugePages = straydog_engine::alloc::HugePages;

#[derive(Parser)]
struct Worker {
    #[command(flatten)]
    args: straydog::worker::WorkerArgs,
}

fn request(text: &str) -> pyjson::Value {
    pyjson::parse(text).unwrap()
}

async fn collect(engine: &Runtime, body: &str) -> (String, serde_json::Value) {
    let mut generation = engine.generate(request(body)).await.unwrap();
    let mut text = String::new();
    while let Some(event) = generation.events.next().await {
        match event["event"].as_str().unwrap() {
            "delta" => text += event["content"].as_str().unwrap(),
            "done" | "error" => return (text, event),
            _ => {}
        }
    }
    panic!("the generation ended without a terminal event");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("worker") {
        let worker = Worker::parse_from(args.into_iter().skip(1));
        if let Err(message) = straydog::worker::main(worker.args) {
            eprintln!("{message}");
            std::process::exit(1);
        }
        return;
    }
    let Some(model) = common::model_env("STRAYDOG_MODEL", "a model bundle directory") else { return };
    let options = Options {
        model: Some(PathBuf::from(model)),
        model_draft: std::env::var_os("STRAYDOG_DRAFTER").map(PathBuf::from),
        gpu: Some(vec![0, 1]),
        vram: Some(30.0),
        ctx_size: Some(32768),
        ..Options::default()
    };
    let settings = config::resolve(options).unwrap();
    let command = (std::env::current_exe().unwrap(), runtime::worker_args(&settings));
    let tokio = tokio::runtime::Runtime::new().unwrap();
    tokio.block_on(async {
        let engine = Runtime::start(&settings, command);
        while matches!(engine.state(), EngineState::Loading { .. }) {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let EngineState::Ready { context, .. } = engine.state() else { panic!("{:?}", engine.state()) };
        assert_eq!(context, 32768);
        let first = r#"{"messages": [{"role": "user", "content": "2+3はいくつ？数字だけ答えて。"}], "max_tokens": 16,
                        "thinking": false}"#;
        let (answer, done) = collect(&engine, first).await;
        assert!(answer.contains('5'), "{answer}");
        assert_eq!(done["finish_reason"], "stop");
        println!("first: {answer:?} {}", done["timings"]["decode_tokens_per_second"]);
        // The follow-up resumes from the live sequence.
        let second = format!(r#"{{"messages": [{{"role": "user", "content": "2+3はいくつ？数字だけ答えて。"}},
            {{"role": "assistant", "content": {}}}, {{"role": "user", "content": "では 4+4 は？"}}],
            "max_tokens": 16, "thinking": false}}"#, serde_json::to_string(&answer).unwrap());
        let (answer, done) = collect(&engine, &second).await;
        assert!(answer.contains('8'), "{answer}");
        assert!(done["usage"]["cached_tokens"].as_u64().unwrap() > 0, "{done}");
        // A context overflow is refused without harming the worker.
        let (_, refused) = collect(&engine, r#"{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 40000}"#).await;
        assert_eq!((refused["event"].as_str(), refused["fatal"].as_bool()), (Some("error"), Some(false)), "{refused}");
        // An interrupted generation still ends with done.
        let mut generation = engine.generate(request(
            r#"{"messages": [{"role": "user", "content": "Count from 1 to 200, one number per line."}],
                "max_tokens": 600, "thinking": false}"#)).await.unwrap();
        let mut deltas = 0;
        let terminal = loop {
            let event = generation.events.next().await.unwrap();
            if event["event"] == "delta" {
                deltas += 1;
                if deltas == 5 {
                    (generation.interrupt)();
                }
            }
            if event["event"] == "done" || event["event"] == "error" {
                break event;
            }
        };
        assert_eq!(terminal["finish_reason"], "cancelled", "{terminal}");
        // A dropped generation is cancelled and the engine serves the next one.
        let mut generation = engine.generate(request(
            r#"{"messages": [{"role": "user", "content": "Write a long story."}], "max_tokens": 600}"#)).await.unwrap();
        generation.events.next().await;
        drop(generation);
        let (answer, _) = collect(&engine, first).await;
        assert!(answer.contains('5'), "{answer}");
        assert!(matches!(engine.state(), EngineState::Ready { .. }));
        println!("runtime_gpu: ok");
    });
}
