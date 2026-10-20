//! `straydog worker`: the isolated engine process behind the server's private pipes.
//!
//! It opens no network listener and never executes tools. Requests arrive as JSON lines on
//! stdin; events leave as JSON lines on stdout. One generation runs at a time on its own
//! thread, so a cancellation is read while it runs. Every model byte is authenticated first.

use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::Args;
use serde_json::{json, Value};
use straydog_engine::chat::ChatTemplate;
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::pyjson;
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

/// The largest frame either way.
pub const FRAME_BYTES: usize = 16 << 20;

/// What the runtime passes to its worker (not a public command line).
#[derive(Debug, Clone, Args)]
pub struct WorkerArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub quant: String,
    #[arg(long)]
    pub drafter: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub speculative: usize,
    #[arg(long, value_delimiter = ',', default_value = "0")]
    pub devices: Vec<usize>,
    #[arg(long, default_value = "experts")]
    pub split_mode: String,
    #[arg(long)]
    pub context: usize,
    #[arg(long)]
    pub vram_bytes: u64,
    #[arg(long)]
    pub state_ram_bytes: u64,
    #[arg(long)]
    pub prefill_batch_size: usize,
    /// Bytes of the PLE row cache (0: rows through the page cache).
    #[arg(long, default_value_t = straydog_engine::rows::DEFAULT_CACHE_BYTES)]
    pub row_cache_bytes: u64,
    #[arg(long, default_value = "fp16")]
    pub gemm: String,
    #[arg(long, default_value_t = 2)]
    pub gemm_min_batch: u64,
    /// Host memory available to the worker (its cgroup limits are invisible inside isolation).
    #[arg(long, default_value_t = 0)]
    pub host_memory_bytes: u64,
}

/// Frames go to stdout whole, one line each, from whichever thread sends them.
#[derive(Clone)]
struct Output(Arc<Mutex<std::io::Stdout>>);

impl Output {
    fn send(&self, frame: &Value) {
        let mut line = serde_json::to_vec(frame).expect("frames serialize");
        if line.len() > FRAME_BYTES {
            line = serde_json::to_vec(&json!({"id": frame["id"], "event": "error", "fatal": false,
                                              "message": "Straydog response exceeds the IPC frame limit"}))
                .unwrap();
        }
        line.push(b'\n');
        let mut out = self.0.lock().unwrap();
        // A closed pipe means the server is gone; the next read of stdin ends the worker.
        let _ = out.write_all(&line).and_then(|()| out.flush());
    }
}

/// Startup progress before `ready`: each stage change at once, then a few frames per second.
struct Loading {
    output: Output,
    last: Mutex<(String, Instant)>,
}

impl Loading {
    fn report(&self, stage: &str, done: u64, total: u64) {
        let mut last = self.last.lock().unwrap();
        if last.0 == stage && last.1.elapsed().as_secs_f64() < 0.25 && done < total {
            return;
        }
        *last = (stage.to_string(), Instant::now());
        self.output.send(&json!({"event": "loading", "stage": stage, "done": done, "total": total}));
    }
}

struct Engine {
    model: Model,
    template: ChatTemplate,
}

/// Run two prompts and a few tokens once, so that no request pays for loading the GPU code it
/// runs (1.5 s more for the first 4,096 tokens on two R9700s); the model then starts empty.
fn warm_up(model: &mut Model) -> Result<(), String> {
    let longest = model.prefill_batch_size.min(model.context.saturating_sub(4));
    for length in [longest, longest.min(300)] {
        if length >= 2 {
            let prompt: Vec<u32> = (0..length as u32).map(|index| 1000 + index % 50_000).collect();
            model.generate(&prompt, 2, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |_| {})?;
        }
    }
    model.reset()
}

pub fn main(args: WorkerArgs) -> Result<(), String> {
    let output = Output(Arc::new(Mutex::new(std::io::stdout())));
    let loading = Loading { output: output.clone(), last: Mutex::new((String::new(), Instant::now())) };
    let split = if args.split_mode == "layer" { SplitMode::Layer } else { SplitMode::Experts };
    let drafter = args.drafter.as_deref().filter(|_| args.speculative > 0);
    let report = |stage: &str, done, total| loading.report(stage, done, total);
    let opened = Instant::now();
    // The model files open while the GPUs initialize.
    let (group, weights) = std::thread::scope(|scope| {
        let weights = scope.spawn(|| Weights::open(&args.model, &args.quant, drafter, args.host_memory_bytes, &report));
        let group = Group::new(&args.devices, args.vram_bytes, split)
            .and_then(|mut group| group.configure_gemm(args.gemm_min_batch, args.gemm == "fp16").map(|()| group));
        (group, weights.join().unwrap())
    });
    let (group, weights) = (group?, weights?);
    let open_seconds = opened.elapsed().as_secs_f64();
    loading.report("prepare", 0, 0);
    let resource = |name: &str| weights.resources.get(name).cloned().ok_or(format!("missing resource {name}"));
    let tokenizer = resource("tokenizer.json")?;
    let template = String::from_utf8(resource("chat_template.jinja")?).map_err(|exc| exc.to_string())?;
    let template = ChatTemplate::new(&template).map_err(|exc| exc.0)?;
    let settings = Settings {
        context: args.context,
        state_budget_bytes: args.state_ram_bytes,
        prefill_batch_size: args.prefill_batch_size,
        speculative: args.speculative,
        cpu_threads: 0,
        row_cache_bytes: args.row_cache_bytes,
    };
    // The tokenizer builds while the model loads.
    let (text, model) = std::thread::scope(|scope| {
        let text = scope.spawn(move || Text::new(&tokenizer).map_err(|exc| exc.to_string()));
        let model = Model::new(weights, group, &settings, &report);
        (text.join().unwrap(), model)
    });
    let (text, mut model) = (Arc::new(text?), model?);
    loading.report("warm", 0, 0);
    let warming = Instant::now();
    warm_up(&mut model)?;
    let mut seconds = serde_json::Map::new();
    seconds.insert("open".into(), open_seconds.into());
    for (step, value) in &model.load_seconds {
        seconds.insert(step.to_string(), (*value).into());
    }
    seconds.insert("warm".into(), warming.elapsed().as_secs_f64().into());
    output.send(&json!({
        "event": "ready", "protocol": 1, "engine": "Straydog", "model": straydog_engine::bundle::MODEL,
        "quantization": model.weights.quantization, "context": model.context, "state_bytes": model.state_bytes,
        "speculative_drafts": model.speculative,
        "placement": {"resident": model.resident, "partial": model.partial, "host_kv": model.host_kv,
                      "prefill_batch_size": model.prefill_batch_size},
        "load_seconds": seconds,
        "gpu": model.info()?,
    }));
    serve(Arc::new(Mutex::new(Engine { model, template })), text, output)
}

/// Read bounded frames until stdin closes or `shutdown`; reject concurrent generation.
fn serve(engine: Arc<Mutex<Engine>>, text: Arc<Text>, output: Output) -> Result<(), String> {
    let active: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let cancel = Arc::new(AtomicBool::new(false));
    let mut thread: Option<std::thread::JoinHandle<()>> = None;
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut line = Vec::new();
    let result = loop {
        line.clear();
        let read = (&mut reader).take(FRAME_BYTES as u64 + 1).read_until(b'\n', &mut line);
        match read {
            Ok(0) => break Ok(()),
            Ok(_) if line.len() > FRAME_BYTES || !line.ends_with(b"\n") => break Err("invalid Straydog IPC frame".into()),
            Ok(_) => {}
            Err(exc) => break Err(exc.to_string()),
        }
        let request = std::str::from_utf8(&line).ok().and_then(|text| pyjson::parse(text).ok());
        let Some(request @ pyjson::Value::Object(_)) = request else { break Err("invalid Straydog request".into()) };
        let Some(id) = request.get("id").and_then(pyjson::Value::as_str).map(str::to_string) else {
            break Err("invalid Straydog request".into());
        };
        if id.len() > 64 {
            break Err("invalid Straydog request id".into());
        }
        match request.get("command").and_then(pyjson::Value::as_str) {
            Some("shutdown") => break Ok(()),
            Some("cancel") => {
                if active.lock().unwrap().as_deref() == Some(id.as_str()) {
                    cancel.store(true, Ordering::SeqCst);
                }
            }
            Some("tokenize") => {
                let Some(content) = request.get("text").and_then(pyjson::Value::as_str) else {
                    break Err("Straydog tokenize needs text".into());
                };
                match text.encode(content) {
                    Ok(ids) => output.send(&json!({"id": id, "event": "count", "tokens": ids.len()})),
                    Err(exc) => output.send(&json!({"id": id, "event": "error", "message": exc.to_string(),
                                                    "fatal": false})),
                }
            }
            Some("chat") if matches!(request.get("body"), Some(pyjson::Value::Object(_))) => {
                let mut current = active.lock().unwrap();
                if current.is_some() {
                    output.send(&json!({"id": id, "event": "error", "message": "Straydog is busy"}));
                    continue;
                }
                // Join a finished predecessor before reusing the cancellation flag.
                if let Some(previous) = thread.take() {
                    let _ = previous.join();
                }
                *current = Some(id.clone());
                cancel.store(false, Ordering::SeqCst);
                let (engine, text, output, active, cancel) =
                    (engine.clone(), text.clone(), output.clone(), active.clone(), cancel.clone());
                let body = request.get("body").unwrap().clone();
                thread = Some(std::thread::Builder::new().name("straydog-generation".into()).spawn(move || {
                    let terminal = generate(&engine, &text, &output, &id, &body, &cancel);
                    *active.lock().unwrap() = None;
                    output.send(&terminal);
                }).map_err(|exc| exc.to_string())?);
            }
            _ => break Err("unsupported Straydog command".into()),
        }
    };
    cancel.store(true, Ordering::SeqCst);
    if let Some(thread) = thread {
        let _ = thread.join();
    }
    result
}

/// One request's failure: a request the engine refuses, or an engine failure (fatal).
struct Failure {
    message: String,
    fatal: bool,
}

fn refused(message: impl Into<String>) -> Failure {
    Failure { message: message.into(), fatal: false }
}

/// Run one chat request; returns its terminal frame (`done` or `error`).
fn generate(engine: &Mutex<Engine>, text: &Text, output: &Output, id: &str, body: &pyjson::Value,
            cancel: &AtomicBool) -> Value {
    let started = Instant::now();
    let mut engine = engine.lock().unwrap();
    match run(&mut engine, text, output, id, body, cancel, started) {
        Ok(done) => done,
        Err(failure) => {
            let mut fatal = failure.fatal;
            // A failed device (an illegal memory access) cannot be reset; the request still ends.
            if let Err(exc) = engine.model.reset() {
                tracing::error!("Straydog could not reset after a failed generation: {exc}");
                fatal = true;
            }
            json!({"id": id, "event": "error", "message": failure.message.chars().take(500).collect::<String>(),
                   "fatal": fatal})
        }
    }
}

fn run(engine: &mut Engine, text: &Text, output: &Output, id: &str, body: &pyjson::Value, cancel: &AtomicBool,
       started: Instant) -> Result<Value, Failure> {
    use pyjson::Value as Py;
    let messages = body.get("messages").ok_or_else(|| refused("'messages'"))?;
    let tools = body.get("tools").cloned().unwrap_or(Py::Null);
    let thinking = body.get("thinking").is_some_and(Py::truthy);
    let effort = match body.get("reasoning_effort") {
        None => "medium",
        Some(Py::Str(effort)) => effort.as_str(),
        Some(_) => return Err(refused("invalid Straydog reasoning effort")),
    };
    let prompt = engine.template.render(messages, &tools, thinking, effort).map_err(|exc| refused(exc.0))?;
    let ids = text.encode(&prompt).map_err(|exc| Failure { message: exc.to_string(), fatal: true })?;
    let encode_seconds = started.elapsed().as_secs_f64();
    let context = engine.model.context;
    let max_new = match body.get("max_tokens") {
        None => 512,
        Some(Py::Null) => context as i128 - ids.len() as i128,
        Some(Py::Int(value)) => *value,
        Some(_) => return Err(refused("Straydog max_tokens must be an integer")),
    };
    let checkpoint = match body.get("checkpoint") {
        None => true,
        Some(Py::Bool(value)) => *value,
        Some(_) => return Err(refused("Straydog checkpoint must be a boolean")),
    };
    let temperature = match body.get("temperature") {
        None => 0.0,
        Some(Py::Int(value)) => *value as f64,
        Some(Py::Float(value)) => *value,
        Some(Py::Bool(value)) => f64::from(u8::from(*value)),
        Some(_) => return Err(refused("invalid Straydog temperature")),
    };
    // Absent, each control is off: the server's API fills in the model card's settings.
    let number = |key: &str, neutral: f64| match body.get(key) {
        None | Some(Py::Null) => Ok(neutral),
        Some(Py::Int(value)) => Ok(*value as f64),
        Some(Py::Float(value)) => Ok(*value),
        Some(_) => Err(refused(format!("invalid Straydog {key}"))),
    };
    let top_k = match body.get("top_k") {
        None | Some(Py::Null) => 0,
        Some(Py::Int(value)) if (0..=1_000_000).contains(value) => *value as usize,
        Some(_) => return Err(refused("invalid Straydog top_k")),
    };
    let sampling = Sampling {
        temperature,
        top_k,
        top_p: number("top_p", 1.0)?,
        min_p: number("min_p", 0.0)?,
        presence_penalty: number("presence_penalty", 0.0)?,
    };
    let seed = match body.get("seed") {
        None => 0,
        Some(Py::Int(value)) if *value >= 0 => *value as u64,
        Some(_) => return Err(refused("invalid Straydog seed")),
    };
    if ids.is_empty() || max_new < 1 || max_new > context as i128 || ids.len() as i128 + max_new > context as i128 {
        return Err(refused("Straydog prompt plus output exceeds the allocated context"));
    }
    sampling.check().map_err(refused)?;
    // A prompt ending in a tool result continues the same reply; one ending in a user message starts one.
    let turn = match messages {
        Py::Array(items) => items.last().and_then(|last| last.get("role")).and_then(Py::as_str) != Some("tool"),
        _ => true,
    };
    let max_new = max_new as usize;
    let mut stream = text.stream();
    let mut generated = 0usize;
    let mut first_text: Option<f64> = None;
    let mut decode_error: Option<String> = None;
    let cancelled = || cancel.load(Ordering::SeqCst);
    let result = engine.model.generate(
        &ids, max_new, &sampling, seed, &cancelled,
        &mut |done, total| output.send(&json!({"id": id, "event": "progress", "done": done, "total": total})),
        turn, checkpoint,
        &mut |token| {
            generated += 1;
            match stream.push(token) {
                Ok(delta) if !delta.is_empty() => {
                    first_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
                    output.send(&json!({"id": id, "event": "delta", "content": delta, "tokens": generated}));
                }
                Ok(_) => {}
                Err(exc) => {
                    decode_error.get_or_insert(exc.to_string());
                }
            }
        },
    );
    result.map_err(|message| Failure { message, fatal: true })?;
    if let Some(message) = decode_error {
        return Err(Failure { message, fatal: true });
    }
    let rest = stream.finish().map_err(|exc| Failure { message: exc.to_string(), fatal: true })?;
    if !rest.is_empty() {
        output.send(&json!({"id": id, "event": "delta", "content": rest, "tokens": generated}));
    }
    let finish_reason = if cancelled() {
        "cancelled"
    } else if generated == max_new {
        "length"
    } else {
        "stop"
    };
    let mut timings = engine.model.last_metrics.clone();
    timings.insert("encode_seconds".into(), json!(encode_seconds));
    timings.insert("first_text_seconds".into(), json!(first_text));
    timings.insert("request_seconds".into(), json!(started.elapsed().as_secs_f64()));
    let cached = timings.get("cached_tokens").and_then(Value::as_u64).unwrap_or(0);
    Ok(json!({
        "id": id, "event": "done", "finish_reason": finish_reason,
        "usage": {"prompt_tokens": ids.len(), "completion_tokens": generated, "total_tokens": ids.len() + generated,
                  "cached_tokens": cached},
        "gpu": engine.model.info().map_err(|message| Failure { message, fatal: true })?,
        "timings": timings,
    }))
}
