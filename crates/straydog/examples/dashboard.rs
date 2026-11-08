//! The dashboard and the API over a simulated engine, for working on the page without a GPU
//! or a model: `cargo run -p straydog --features hip --example dashboard [-- PORT]`, then
//! open http://127.0.0.1:18090/dashboard. The worker is this program again; it speaks the
//! worker protocol with made-up text and timings (no inference happens). It plays Qwen3.8 with
//! its MTP drafter; with `DASHBOARD_MODEL=deepseek` it plays DeepSeek-V4.1 (DSpark and prompt lookup).

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use straydog::config::{Effort, Gemm, Settings, SplitMode, Vram};
use straydog::dashboard::{self, Dashboard};
use straydog::monitor::{LogBuffer, LogSink};
use straydog::runtime::Runtime;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

const GIB: u64 = 1 << 30;

/// Whether to play DeepSeek-V4.1 rather than Qwen3.8: asked for by the variable, and told to the
/// worker (which the server starts with an environment of its own) as its argument.
fn deepseek() -> bool {
    std::env::var("DASHBOARD_MODEL").is_ok_and(|model| model == "deepseek") || std::env::args().any(|arg| arg == "deepseek")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("simulate-worker") {
        simulate();
        return;
    }
    let port: u16 = args.first().and_then(|port| port.parse().ok()).unwrap_or(18090);
    let logs = Arc::new(LogBuffer::default());
    let sink = logs.clone();
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new("info"))
        .with(fmt::layer().with_target(false).with_writer(std::io::stderr))
        .with(fmt::layer().with_target(false).with_ansi(false).with_writer(move || LogSink(sink.clone())))
        .init();
    let (quant, model_draft, draft_max) =
        if deepseek() { ("IQ2_XXS", None, 0) } else { ("IQ3_S", Some(PathBuf::from("/simulated/Qwen3.8-Flash-Next-MTP")), 3) };
    let settings = Settings {
        model: PathBuf::from("/simulated").join(quant),
        bundle: straydog_engine::bundle::Id::find(Some(if deepseek() { straydog_engine::bundle::DEEPSEEK } else { straydog_engine::bundle::MODEL }), None, quant).unwrap(),
        engram: None,
        expert_floor: None,
        model_draft,
        model_vision: None,
        draft_max,
        ctx_size: 262_144,
        batch_size: Some(4096),
        gpu: vec![0, 1],
        split_mode: SplitMode::Experts,
        vram: Vram::Gib(vec![30.0]),
        state_ram: 0.0,
        row_cache: None,
        gemm: Gemm::Fp16,
        gemm_min_batch: 2,
        host: "127.0.0.1".into(),
        port,
        api_key: None,
        api_key_file: None,
        alias: format!("simulated-{quant}"),
        temp: None,
        top_k: None,
        top_p: None,
        min_p: None,
        presence_penalty: None,
        seed: None,
        reasoning_effort: Effort::High,
        startup_timeout: 60.0,
        log_file: None,
        verbose: false,
        data_dir: PathBuf::from("/nonexistent/.straydog"),
        expert_ram: None,
        prefix_cache: false,
        keep_conversation: false,
        stop_on_stdin: false,
    };
    let mut worker_args = vec!["simulate-worker".into()];
    if deepseek() {
        worker_args.push("deepseek".into());
    }
    let command = (std::env::current_exe().unwrap(), worker_args);
    tokio::runtime::Runtime::new().unwrap().block_on(async move {
        let engine = Arc::new(Runtime::start(&settings, command));
        let api = straydog_api::ApiConfig {
            model_id: settings.alias.clone(),
            context: u64::from(settings.ctx_size),
            quantization: settings.bundle.quant.to_string(),
            api_key: None,
            sampling: straydog_api::SamplingDefaults::default(),
            seed: None,
            reasoning_effort: straydog_api::ReasoningEffort::High,
            created: 0,
            markup: straydog_api::Markup::Qwen,
        };
        let monitor = Arc::new(Dashboard::new(engine.clone(), logs, &settings));
        let router = straydog_api::router(engine.clone(), api).merge(dashboard::router(monitor));
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.unwrap();
        tracing::info!("Simulated Straydog: http://127.0.0.1:{port}/dashboard");
        axum::serve(listener, router).await.unwrap();
    });
}

/// The simulated worker: JSON lines in, JSON lines out.
fn simulate() {
    let output = Arc::new(Mutex::new(std::io::stdout()));
    let send = {
        let output = output.clone();
        move |value: Value| {
            let mut output = output.lock().unwrap();
            writeln!(output, "{value}").unwrap();
            output.flush().unwrap();
        }
    };
    for (stage, total, steps) in [("authenticate", 83_600_000_000u64, 12), ("drafter", 889_021_728, 3), ("prepare", 0, 1)] {
        for step in 0..=steps {
            send(json!({"event": "loading", "stage": stage, "done": total * step / steps, "total": total}));
            std::thread::sleep(Duration::from_millis(250));
        }
    }
    let gpu = gpu_info();
    if deepseek() {
        send(json!({
            "event": "ready", "protocol": 1, "engine": "Straydog", "model": "DeepSeek-V4.1-Flash-1017", "provider": "ISTA-DASLab", "quantization": "IQ2_XXS",
            "context": 262_144, "state_bytes": 3 * GIB, "speculative_drafts": 0,
            "speculation": [{"kind": "dspark", "resident": [87, 50, 29]}, {"kind": "lookup"}],
            "placement": {"expert_slots": [2916, 2916], "layers": [[0, 40]], "devices": [[0, 1]], "host_expert_bytes": 100 * GIB,
                          "prefill_batch_size": 4096, "dspark": {"engine": 0, "resident": [87, 50, 29]}},
            "gpu": gpu,
        }));
    } else {
        send(json!({
            "event": "ready", "protocol": 1, "engine": "Straydog", "model": "Qwen3.8-Flash-Next", "provider": "ISTA-DASLab", "quantization": "IQ3_S",
            "context": 262_144, "state_bytes": 7 * GIB, "speculative_drafts": 3,
            "speculation": [{"kind": "mtp", "drafts": 3}],
            "placement": {"resident": true, "partial": false, "host_kv": true, "prefill_batch_size": 4096},
            "gpu": gpu,
        }));
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let mut generation: Option<std::thread::JoinHandle<()>> = None;
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let Ok(frame) = serde_json::from_str::<Value>(&line) else { continue };
        match frame["command"].as_str() {
            Some("chat") => {
                if let Some(previous) = generation.take() {
                    previous.join().unwrap();
                }
                cancel.store(false, Ordering::SeqCst);
                let (send, cancel) = (send.clone(), cancel.clone());
                generation = Some(std::thread::spawn(move || answer(&frame, &send, &cancel)));
            }
            Some("cancel") => cancel.store(true, Ordering::SeqCst),
            Some("shutdown") => break,
            _ => {}
        }
    }
    if let Some(generation) = generation {
        generation.join().unwrap();
    }
}

/// The host's AMD GPUs, so that the simulated devices meet real sensors on the dashboard.
fn pci_addresses() -> Vec<String> {
    let mut addresses: Vec<String> = std::fs::read_dir("/sys/class/drm").into_iter().flatten().flatten()
        .filter(|card| card.file_name().to_string_lossy().starts_with("card") && !card.file_name().to_string_lossy().contains('-'))
        .filter_map(|card| std::fs::canonicalize(card.path().join("device")).ok())
        .filter(|device| std::fs::read_to_string(device.join("vendor")).is_ok_and(|vendor| vendor.trim() == "0x1002"))
        .filter_map(|device| Some(device.file_name()?.to_string_lossy().to_lowercase()))
        .collect();
    addresses.sort();
    addresses
}

fn gpu_info() -> Value {
    let addresses = pci_addresses();
    let device = |ordinal: u64| json!({"name": "Simulated GPU", "device": ordinal,
                                       "pci": addresses.get(ordinal as usize).cloned().unwrap_or_else(|| format!("0000:{:02x}:00.0", ordinal + 1)),
                                       "free_bytes": 3 * GIB, "total_bytes": 32 * GIB, "allocated_bytes": 28 * GIB, "budget_bytes": 30 * GIB});
    json!({"name": "Simulated GPU + Simulated GPU", "device": 0, "free_bytes": 6 * GIB, "total_bytes": 64 * GIB,
           "allocated_bytes": 56 * GIB, "budget_bytes": 60 * GIB, "devices": [device(0), device(1)]})
}

const REASONING: &str = "The user asked something; this simulated engine thinks out loud for a moment, \
                         weighing nothing in particular, before it answers.";
const ANSWER: &str = "This answer comes from the simulated engine of the dashboard example. No model ran: the \
                      text, the token counts and the timings are made up so that the page can be developed \
                      without a GPU.\n\n```rust\nfn main() {\n    println!(\"Straydog\");\n}\n```\n\nEach word \
                      arrives as one token at a steady pace, with a short pause for prefill first.";

fn answer(frame: &Value, send: &(impl Fn(Value) + ?Sized), cancel: &AtomicBool) {
    let id = frame["id"].clone();
    let body = &frame["body"];
    let started = Instant::now();
    let prompt = body["messages"].to_string().len() as u64 / 4 + 16;
    // The first turn is computed in full; later turns reuse the conversation so far.
    let cached = if body["messages"].as_array().is_some_and(|messages| messages.len() > 1) { prompt * 4 / 5 } else { 0 };
    let fresh = prompt - cached;
    for step in 1..=4 {
        send(json!({"id": id, "event": "progress", "done": cached + fresh * step / 4, "total": prompt}));
        std::thread::sleep(Duration::from_millis(fresh / 8 + 20));
    }
    let prefill_seconds = started.elapsed().as_secs_f64();
    let ttft = started.elapsed().as_secs_f64();
    let thinking = body["thinking"] == true;
    let text = if thinking { format!("{REASONING}\n</think>\n\n{ANSWER}") } else { ANSWER.to_string() };
    // A long request gets the answer again until its limit (at most 400 tokens).
    let limit = body["max_tokens"].as_u64().unwrap_or(u64::MAX);
    let repeats = if limit == u64::MAX { 1 } else { (limit.min(400) / 60).max(1) as usize };
    let text = vec![text; repeats].join("\n\n");
    let decode_started = Instant::now();
    let mut tokens = 0u64;
    let mut finish = "stop";
    for piece in text.split_inclusive(' ') {
        if cancel.load(Ordering::SeqCst) {
            finish = "cancelled";
            break;
        }
        if tokens == limit {
            finish = "length";
            break;
        }
        tokens += 1;
        send(json!({"id": id, "event": "delta", "content": piece, "tokens": tokens}));
        std::thread::sleep(Duration::from_millis(9 + tokens % 5));
    }
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let drafted = tokens * 3 / 4;
    let mut timings = json!({"cached_tokens": cached, "resumed_from": if cached > 0 { "live" } else { "none" },
                             "prefill_tokens": fresh, "prefill_seconds": prefill_seconds,
                             "prefill_tokens_per_second": fresh as f64 / prefill_seconds, "ttft_seconds": ttft,
                             "decode_seconds": decode_seconds, "decode_tokens_per_second": tokens as f64 / decode_seconds,
                             "generated_tokens": tokens, "request_seconds": started.elapsed().as_secs_f64()});
    if deepseek() {
        // (DSpark's drafts are most of them, prompt lookup's the rest)
        let dspark = drafted * 2 / 3;
        timings["draft_tokens"] = drafted.into();
        timings["accepted_drafts"] = (dspark * 3 / 4 + (drafted - dspark) * 19 / 20).into();
        timings["engine_draft_tokens"] = dspark.into();
        timings["accepted_engine_drafts"] = (dspark * 3 / 4).into();
    } else {
        timings["drafted_tokens"] = drafted.into();
        timings["accepted_tokens"] = (drafted * 7 / 10).into();
    }
    send(json!({
        "id": id, "event": "done", "finish_reason": finish,
        "usage": {"prompt_tokens": prompt, "completion_tokens": tokens, "total_tokens": prompt + tokens, "cached_tokens": cached},
        "gpu": gpu_info(),
        "timings": timings,
    }));
}
