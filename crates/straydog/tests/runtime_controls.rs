//! The runtime's control path against stand-in workers (this test binary, run as
//! `stand-in MARKER` and so on): an interrupt from a client that reads none of its events
//! still reaches the worker at once, and the client then gets the generation's end; a worker
//! that fails to start fails the runtime, which says so (the server then exits), and when the OS
//! refused it, says why after the worker's own reason; closing the
//! runtime returns only once every process of the worker's tree has exited, the one outside
//! its process group too (as a sandbox's is); a request too large for the worker is refused alone;
//! a client that leaves during a long prefill pass costs nothing but the pass, and only a worker
//! that does not end a cancelled generation in time fails the runtime.
//! No GPU or model.
//!
//! cargo test -p straydog --features hip --test runtime_controls

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use straydog::config::{self, Options};
use straydog::runtime::{Refusal, Runtime};
use straydog_api::{Engine, EngineState};
use straydog_engine::pyjson;

const CONTEXT: u32 = 4096;
const QUANT: &str = "IQ3_S";
/// Longer than the 10 s a worker once had to end a cancelled generation, which a real pass exceeds.
const PASS: Duration = Duration::from_secs(11);
/// What a worker the OS refused says last (as Windows words it, ending in Rust's rendering of the code), and
/// the warning its launcher printed before it started.
const DENIED: &str = r"\\?\D:\Models\IQ2_XS: Access is denied. (os error 5)";
const WARNING: &str = r"Straydog warning: D:\ is formatted exFAT, which keeps no file permissions, so Straydog's contained worker may not read D:\Models\IQ2_XS there; put it on an NTFS drive";

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

/// A worker in a prefill pass that takes `PASS`: its first chat answers only once the pass has
/// ended, whatever arrives meanwhile, and the next at once.
fn in_a_pass() {
    ready();
    let mut pass = Some(PASS);
    for line in std::io::stdin().lock().lines() {
        let frame: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        match frame["command"].as_str() {
            Some("chat") => {
                std::thread::sleep(pass.take().unwrap_or_default());
                println!("{}", serde_json::json!({"id": frame["id"], "event": "done", "finish_reason": "stop",
                                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}}));
            }
            Some("shutdown") => break,
            _ => {}
        }
    }
}

/// A worker that never answers a chat.
fn hung() {
    ready();
    for line in std::io::stdin().lock().lines() {
        if line.unwrap().contains("shutdown") {
            break;
        }
    }
}

/// A worker whose process (as a sandbox's) leaves the tree's process group and outlives the
/// worker by a second; its id goes to MARKER. Linux process groups have no Windows counterpart
/// here; the Job Object path is covered by the isolation tests.
#[cfg(unix)]
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

#[cfg(unix)]
fn grandchild() {
    let parent = std::os::unix::process::parent_id();
    while std::os::unix::process::parent_id() == parent {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_secs(1));
}

/// Whether `pid` runs (a zombie has exited).
#[cfg(unix)]
fn running(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .is_ok_and(|stat| stat.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().next()) != Some("Z"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("stand-in") => return stand_in(Path::new(&args[2])),
        Some("in-a-pass") => return in_a_pass(),
        Some("hung") => return hung(),
        // As a worker that finds its GPUs' memory held fails: its reason on stderr, status 1.
        Some("fails") => {
            eprintln!("Straydog VRAM budget is unavailable");
            std::process::exit(1);
        }
        // As a worker that panics while it starts: Rust's own panic output, then status 101.
        Some("panics") => {
            let held: Option<u32> = None;
            held.expect("Straydog VRAM budget is held");
            return;
        }
        // As a worker the OS refused a file: its launcher's warning first, then its reason.
        Some("denied") => {
            eprintln!("{WARNING}");
            eprintln!("{DENIED}");
            std::process::exit(1);
        }
        #[cfg(unix)]
        Some("lingers") => return lingering(Path::new(&args[2])),
        #[cfg(unix)]
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

        // A request past the frame limit is refused alone, the worker serving on; one past the old
        // 16 MiB limit (a conversation's tenth screenshot, which stopped the server) reaches it.
        for (bytes, refused) in [(straydog::worker::FRAME_BYTES + 1, true), (20 << 20, false)] {
            let body = pyjson::parse(&format!(r#"{{"messages": [{{"role": "user", "content": "{}"}}]}}"#, "x".repeat(bytes))).unwrap();
            let mut generation = engine.generate(body).await.unwrap();
            let first = tokio::time::timeout(Duration::from_secs(30), generation.events.next()).await.unwrap().unwrap();
            if refused {
                assert_eq!((first["event"].as_str(), first["fatal"].as_bool()), (Some("error"), Some(false)), "{first}");
                assert!(matches!(engine.state(), EngineState::Ready { .. }), "{:?}", engine.state());
                continue;
            }
            assert_eq!(first["event"].as_str(), Some("delta"), "{first}");
            (generation.interrupt)();
            while let Some(event) = tokio::time::timeout(Duration::from_secs(10), generation.events.next()).await.unwrap() {
                if event["event"].as_str() == Some("done") {
                    break;
                }
            }
        }
        println!("runtime_controls: a request past the frame limit refused alone; a 20 MiB one served");
        engine.close().await;

        // A client leaves during a long pass: the pass ends, and the worker serves the next request.
        let wait_ready = |engine: Runtime| async move {
            let started = Instant::now();
            while !matches!(engine.state(), EngineState::Ready { .. }) {
                assert!(started.elapsed() < Duration::from_secs(10), "the worker did not start: {:?}", engine.state());
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            engine
        };
        let hi = || pyjson::parse(r#"{"messages": [{"role": "user", "content": "hi"}]}"#).unwrap();
        let engine = wait_ready(Runtime::start(&settings, (std::env::current_exe().unwrap(), vec!["in-a-pass".into()]))).await;
        let left = engine.generate(hi()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(left);
        let started = Instant::now();
        let mut next = engine.generate(hi()).await.unwrap();
        let event = tokio::time::timeout(PASS * 2, next.events.next()).await.unwrap().unwrap();
        assert_eq!(event["event"].as_str(), Some("done"), "{event}");
        assert!(started.elapsed() > PASS - Duration::from_secs(1), "the left generation's pass did not run: {:?}", started.elapsed());
        assert!(matches!(engine.state(), EngineState::Ready { .. }), "{:?}", engine.state());
        println!("runtime_controls: a client that left during a pass of {PASS:?} cost the pass alone, {:.1} s", started.elapsed().as_secs_f64());
        engine.close().await;

        // A worker that does not end a cancelled generation in time fails the runtime.
        let patience = Duration::from_secs(1);
        let engine = wait_ready(Runtime::start_with_cancel_patience(&settings, (std::env::current_exe().unwrap(), vec!["hung".into()]), patience)).await;
        let left = engine.generate(hi()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(left);
        let message = tokio::time::timeout(Duration::from_secs(10), engine.failed()).await
            .expect("a hung worker did not fail the runtime");
        assert!(message.contains("did not acknowledge cancellation"), "{message}");
        assert!(matches!(engine.state(), EngineState::Failed(_)));
        engine.close().await;

        // A worker that fails to start: the runtime fails with its reason, at once.
        let engine = Runtime::start(&settings, (std::env::current_exe().unwrap(), vec!["fails".into()]));
        let message = tokio::time::timeout(Duration::from_secs(10), engine.failed()).await
            .expect("a worker that failed to start did not fail the runtime");
        assert!(message.contains("VRAM budget is unavailable"), "{message}");
        assert!(matches!(engine.state(), EngineState::Failed(_)));
        engine.close().await;
        // (one that panicked: by its panic and message, never Rust's note on backtraces)
        let engine = Runtime::start(&settings, (std::env::current_exe().unwrap(), vec!["panics".into()]));
        let message = tokio::time::timeout(Duration::from_secs(10), engine.failed()).await
            .expect("a worker that panicked did not fail the runtime");
        assert!(message.contains("panicked at") && message.contains("Straydog VRAM budget is held")
                && !message.contains("RUST_BACKTRACE"), "{message}");
        engine.close().await;
        println!("runtime_controls: a worker that panicked at start is reported by its panic: {message}");

        // A worker the OS refused: the runtime looks for why and says it after the worker's own reason
        // (the launcher's warning is not that reason); for any other failure it does not look.
        let program = std::env::current_exe().unwrap();
        let why = r"D:\ is formatted exFAT, which keeps no file permissions, so Straydog's contained worker may not read D:\Models\IQ2_XS\a.gguf there; put it on an NTFS drive";
        let looked = Arc::new(AtomicUsize::new(0));
        let refusal = |answer: Option<&'static str>| -> Refusal {
            let looked = looked.clone();
            Arc::new(move || {
                looked.fetch_add(1, Ordering::SeqCst);
                answer.map(str::to_string)
            })
        };
        let failure = |engine: Runtime| async move {
            let message = tokio::time::timeout(Duration::from_secs(10), engine.failed()).await
                .expect("a worker that failed to start did not fail the runtime");
            assert!(matches!(engine.state(), EngineState::Failed(_)));
            engine.close().await;
            message
        };
        let plain = format!("Straydog failed to start ({DENIED}); see the server log and restart the server");
        let told = failure(Runtime::start_with_refusal(&settings, (program.clone(), vec!["denied".into()]), refusal(Some(why)))).await;
        assert_eq!(told, format!("Straydog failed to start ({DENIED}: {why}); see the server log and restart the server"));
        assert_eq!(looked.load(Ordering::SeqCst), 1);
        // (nothing to tell: the reason alone)
        let untold = failure(Runtime::start_with_refusal(&settings, (program.clone(), vec!["denied".into()]), refusal(None))).await;
        assert_eq!((untold, looked.load(Ordering::SeqCst)), (plain.clone(), 2));
        // (a worker that failed some other way is not looked into)
        let other = failure(Runtime::start_with_refusal(&settings, (program.clone(), vec!["fails".into()]), refusal(Some(why)))).await;
        assert!(other.contains("VRAM budget is unavailable") && !other.contains(why) && !other.contains(": D:"), "{other}");
        assert_eq!(looked.load(Ordering::SeqCst), 2);
        // (the runtime's own search, over files it has no verity trees for: it cannot tell)
        let searched = failure(Runtime::start(&settings, (program, vec!["denied".into()]))).await;
        assert_eq!(searched, plain);
        println!("runtime_controls: a worker the OS refused is told why after its own reason, and no other");

        // Closing waits for the worker's processes, the one outside its process group too.
        // Windows has no process-group escape of this shape; Job Object cleanup is tested elsewhere.
        #[cfg(unix)]
        {
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
        }
        #[cfg(not(unix))]
        println!("runtime_controls: a failed start fails the runtime");
    });
    std::fs::remove_dir_all(&scratch).unwrap();
}
