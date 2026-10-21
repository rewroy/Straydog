//! The runtime's control path against stand-in workers (this test binary, run as
//! `stand-in MARKER` and so on): an interrupt from a client that reads none of its events
//! still reaches the worker at once, and the client then gets the generation's end; a worker
//! that fails to start fails the runtime, which says so (the server then exits); closing the
//! runtime returns only once every process of the worker's tree has exited, the one outside
//! its process group too (as a sandbox's is). No GPU or model.
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

fn ready() {
    println!("{}", serde_json::json!({"event": "ready", "protocol": 1, "engine": "Straydog", "context": CONTEXT,
                                      "model": straydog_engine::bundle::MODEL, "quantization": QUANT}));
}

/// A worker whose process (as a sandbox's) leaves the tree's process group and outlives the
/// worker by a second; its id goes to MARKER.
#[allow(clippy::zombie_processes)] // the process outlives this one on purpose (its new parent reaps it)
fn lingering(marker: &Path) {
    use std::os::unix::process::CommandExt;
    let child = std::process::Command::new(std::env::current_exe().unwrap()).arg("grandchild").process_group(0)
        .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).spawn().unwrap();
    std::fs::write(marker, child.id().to_string()).unwrap();
    ready();
    // Until the runtime asks it to shut down (or closes its input).
    for line in std::io::stdin().lock().lines() {
        if line.unwrap().contains("shutdown") {
            break;
        }
    }
}

fn grandchild() {
    let parent = std::os::unix::process::parent_id();
    while std::os::unix::process::parent_id() == parent {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_secs(1));
}

/// Whether `pid` runs (a zombie has exited).
fn running(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .is_ok_and(|stat| stat.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().next()) != Some("Z"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("stand-in") => return stand_in(Path::new(&args[2])),
        // As a worker that finds its GPUs' memory held fails: its reason on stderr, status 1.
        Some("fails") => {
            eprintln!("Straydog VRAM budget is unavailable");
            std::process::exit(1);
        }
        Some("lingers") => return lingering(Path::new(&args[2])),
        Some("grandchild") => return grandchild(),
        _ => {}
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

        // A worker that fails to start: the runtime fails with its reason, at once.
        let engine = Runtime::start(&settings, (std::env::current_exe().unwrap(), vec!["fails".into()]));
        let message = tokio::time::timeout(Duration::from_secs(10), engine.failed()).await
            .expect("a worker that failed to start did not fail the runtime");
        assert!(message.contains("VRAM budget is unavailable"), "{message}");
        assert!(matches!(engine.state(), EngineState::Failed(_)));
        engine.close().await;

        // Closing waits for the worker's processes, the one outside its process group too.
        let pid_file = scratch.join("grandchild");
        let engine = Runtime::start(&settings, (std::env::current_exe().unwrap(),
                                                vec!["lingers".into(), PathBuf::from(&pid_file).into_os_string()]));
        let started = Instant::now();
        while !matches!(engine.state(), EngineState::Ready { .. }) {
            assert!(started.elapsed() < Duration::from_secs(10), "the lingering worker did not start: {:?}", engine.state());
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let grandchild: u32 = std::fs::read_to_string(&pid_file).unwrap().parse().unwrap();
        let closing = Instant::now();
        engine.close().await;
        assert!(!running(grandchild), "the runtime closed while its worker's sandboxed process still ran");
        println!("runtime_controls: a failed start fails the runtime; closing waited {:.1} s for the worker's last process",
                 closing.elapsed().as_secs_f64());
    });
    std::fs::remove_dir_all(&scratch).unwrap();
}
