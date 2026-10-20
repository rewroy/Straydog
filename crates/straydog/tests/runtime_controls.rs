//! The runtime's control path against a stand-in worker (this test binary, run as
//! `stand-in MARKER`): an interrupt from a client that reads none of its events still reaches
//! the worker at once, and the client then gets the generation's end. No GPU or model.
//!
//! cargo test -p straydog --features hip --test runtime_controls

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use straydog::config::{self, Options};
use straydog::runtime::Runtime;
use straydog_api::{Engine, EngineState};
use straydog_engine::pyjson;

const CONTEXT: u32 = 4096;
const QUANT: &str = "IQ3_S";

/// The worker's side of the protocol: ready, then for a chat a delta every millisecond (up to
/// 20000) until a cancel arrives, which it records in MARKER before ending the generation.
fn stand_in(marker: &Path) {
    let stdout = Arc::new(std::sync::Mutex::new(std::io::stdout()));
    let emit = |frame: serde_json::Value| {
        let mut out = stdout.lock().unwrap();
        writeln!(out, "{frame}").unwrap();
        out.flush().unwrap();
    };
    emit(serde_json::json!({"event": "ready", "protocol": 1, "engine": "Straydog", "context": CONTEXT,
                            "model": straydog_engine::bundle::MODEL, "quantization": QUANT}));
    let cancelled = Arc::new(AtomicBool::new(false));
    let (chats, requests) = std::sync::mpsc::channel::<String>();
    {
        let (cancelled, marker) = (cancelled.clone(), marker.to_path_buf());
        std::thread::spawn(move || {
            for line in std::io::stdin().lock().lines() {
                let frame: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
                match frame["command"].as_str() {
                    Some("chat") => chats.send(frame["id"].as_str().unwrap().to_string()).unwrap(),
                    Some("cancel") => {
                        std::fs::write(&marker, "cancelled").unwrap();
                        cancelled.store(true, Ordering::SeqCst);
                    }
                    _ => {}
                }
            }
        });
    }
    for id in requests {
        cancelled.store(false, Ordering::SeqCst);
        for _ in 0..20000 {
            if cancelled.load(Ordering::SeqCst) {
                break;
            }
            emit(serde_json::json!({"id": id, "event": "delta", "content": "x"}));
            std::thread::sleep(Duration::from_millis(1));
        }
        emit(serde_json::json!({"id": id, "event": "done", "finish_reason": "stop",
                                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("stand-in") {
        stand_in(Path::new(&args[2]));
        return;
    }
    let scratch = std::env::temp_dir().join(format!("straydog-runtime-controls-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let marker = scratch.join("cancel");
    let options = Options { model: Some(scratch.clone()), quant: Some(QUANT.into()), ctx_size: Some(CONTEXT), ..Options::default() };
    let settings = config::resolve(options).unwrap();
    let command = (std::env::current_exe().unwrap(), vec!["stand-in".into(), PathBuf::from(&marker).into_os_string()]);
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let engine = Runtime::start(&settings, command);
        let started = Instant::now();
        while !matches!(engine.state(), EngineState::Ready { .. }) {
            assert!(started.elapsed() < Duration::from_secs(10), "the stand-in worker did not start: {:?}", engine.state());
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let body = pyjson::parse(r#"{"messages": [{"role": "user", "content": "hi"}]}"#).unwrap();
        let mut generation = engine.generate(body).await.unwrap();
        // Read nothing until the events channel is full and the runtime waits to send.
        tokio::time::sleep(Duration::from_millis(500)).await;
        (generation.interrupt)();
        let started = Instant::now();
        while !marker.exists() {
            assert!(started.elapsed() < Duration::from_secs(5),
                    "an interrupt from a client that reads nothing did not reach the worker");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let mut deltas = 0;
        let mut ended = false;
        while let Some(event) = tokio::time::timeout(Duration::from_secs(10), generation.events.next()).await.unwrap() {
            match event["event"].as_str() {
                Some("delta") => deltas += 1,
                Some("done") => {
                    ended = true;
                    break;
                }
                other => panic!("unexpected event {other:?}"),
            }
        }
        assert!(ended, "the interrupted generation did not end");
        assert!(deltas < 5000, "{deltas} deltas: the worker was not stopped early");
        println!("runtime_controls: interrupt reached the stand-in in {:.0} ms with the client not reading; {deltas} deltas",
                 started.elapsed().as_secs_f64() * 1e3);
        engine.close().await;
    });
    std::fs::remove_dir_all(&scratch).unwrap();
}
