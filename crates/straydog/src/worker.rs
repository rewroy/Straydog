//! `straydog worker`: the isolated engine process behind the server's private pipes.
//!
//! It opens no network listener and never executes tools. Requests arrive as JSON lines on
//! stdin; events leave as JSON lines on stdout. One generation runs at a time on its own
//! thread, so a cancellation is read while it runs. Every model byte is authenticated first.

use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Args;
use serde_json::{json, Value};
use straydog_engine::chat::{self, ChatTemplate};
use straydog_engine::constraint::Constraint;
use straydog_engine::reply::THINK_CLOSE;
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings, END_TOKENS};
use straydog_engine::pyjson;
use straydog_engine::sampling::Sampling;
use straydog_engine::text::{Text, TextError};
use straydog_engine::weights::Weights;

/// The largest frame either way: any request body the API takes ([`straydog_api::BODY_LIMIT`])
/// with room for the frame around it. At 16 MiB a conversation's tenth screenshot (a 64 MiB body
/// was taken) failed the runtime, and the server stopped serving.
pub const FRAME_BYTES: usize = straydog_api::BODY_LIMIT + (16 << 20);

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
    /// Each GPU's VRAM budget in bytes (comma separated, one per device), or auto.
    #[arg(long)]
    pub vram_bytes: String,
    #[arg(long)]
    pub state_ram_bytes: u64,
    /// Tokens a prefill pass takes, or auto (from the budgets).
    #[arg(long)]
    pub prefill_batch_size: String,
    /// Bytes of the row cache of the PLE and Engram tables (0: rows through the page cache) [default:
    /// the default budget, and for Qwen3.8's PLE table as much as the host memory leaves].
    #[arg(long)]
    pub row_cache_bytes: Option<u64>,
    #[arg(long, default_value = "fp16")]
    pub gemm: String,
    #[arg(long, default_value_t = 2)]
    pub gemm_min_batch: u64,
    /// Host memory available to the worker (its cgroup limits are invisible inside isolation).
    #[arg(long, default_value_t = 0)]
    pub host_memory_bytes: u64,
    /// Straydog's data directory: the shards' verity trees (read only).
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    /// The routed experts' host tier of a bundle read on demand [default: planned from host memory].
    #[arg(long)]
    pub host_expert_bytes: Option<u64>,
}

/// Frames go to stdout whole, one line each, from whichever thread sends them.
#[derive(Clone)]
pub(crate) struct Output(Arc<Mutex<std::io::Stdout>>);

impl Output {
    pub(crate) fn send(&self, frame: &Value) {
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

/// A loaded model as the request loop drives it: one request at a time, and a reset after a
/// failed one.
pub(crate) trait Served: Send {
    fn run(&mut self, text: &Arc<Text>, output: &Output, id: &str, body: &pyjson::Value, cancel: &Arc<AtomicBool>,
           started: Instant) -> Result<Value, Failure>;
    fn reset(&mut self) -> Result<(), String>;

    /// What a later start warms up with (the expert store's contents), or null.
    fn warm_state(&mut self) -> Result<Value, String> {
        Ok(Value::Null)
    }

    /// Warm up with an earlier run's state, in the background, never ahead of a request.
    fn warm(&mut self, _state: &Value) -> Result<(), String> {
        Ok(())
    }
}

impl Served for Engine {
    fn run(&mut self, text: &Arc<Text>, output: &Output, id: &str, body: &pyjson::Value, cancel: &Arc<AtomicBool>,
           started: Instant) -> Result<Value, Failure> {
        run(self, text, output, id, body, cancel, started)
    }

    fn reset(&mut self) -> Result<(), String> {
        self.model.reset()
    }
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

/// How long a GPU's memory held by another process is waited for.
const VRAM_PATIENCE: Duration = Duration::from_secs(60);

/// What the worker settles before placing the model: each GPU's VRAM budget, the prefill batch
/// and the split.
pub(crate) struct Plan {
    pub budgets: Vec<u64>,
    pub batch: usize,
    pub split: SplitMode,
}

/// The margin `auto` leaves free on a GPU: 512 MiB or 3% of it, whichever is more; on Windows, where the
/// desktop and other programs share the GPU, 2 GiB at least.
fn margin(total: u64) -> u64 {
    #[cfg(windows)]
    {
        return (2u64 << 30).max(total / 100 * 3);
    }
    #[cfg(not(windows))]
    (512 << 20).max(total / 100 * 3)
}

/// How long `auto` waits for a GPU's free memory to stop growing (a previous worker releasing it).
const SETTLE: Duration = Duration::from_secs(5);

/// Each GPU's budget. Given ones wait until the GPU has them free: memory a previous worker still
/// holds is released after it exits, while memory another process keeps fails the start after
/// [`VRAM_PATIENCE`] (a GPU smaller than its budget is left to fail as it does). `auto` waits
/// until nine tenths of each GPU are free, or until its free memory has stopped growing for
/// [`SETTLE`] with at most a quarter of it held, and at most [`VRAM_PATIENCE`] in all; then it
/// takes what is free less a margin, in whole 256 MiB. A display's or another program's share is
/// left to it, but a job holding more of the GPU (a test or benchmark about to end) is waited
/// for, not placed around in the memory it leaves.
fn budgets(devices: &[usize], given: &str, report: &dyn Fn(&str, u64, u64)) -> Result<Vec<u64>, String> {
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    let wanted: Option<Vec<u64>> = if given == "auto" {
        None
    } else {
        let bytes = given.split(',').map(|part| part.parse::<u64>().map_err(|_| format!("invalid Straydog VRAM budget {given:?}")))
            .collect::<Result<Vec<_>, _>>()?;
        if bytes.len() != devices.len() {
            return Err("one Straydog VRAM budget per GPU".into());
        }
        Some(bytes)
    };
    let started = Instant::now();
    let mut chosen = Vec::new();
    for (index, &device) in devices.iter().enumerate() {
        // (the most free memory seen and when it last grew by more than 64 MiB)
        let mut grown = (0u64, Instant::now());
        loop {
            let (_, free, total) = straydog_gpu::device_info(device).map_err(|exc| exc.to_string())?;
            let need = match &wanted {
                Some(bytes) if free >= bytes[index] || total < bytes[index] => {
                    chosen.push(bytes[index]);
                    break;
                }
                Some(bytes) if started.elapsed() >= VRAM_PATIENCE => {
                    return Err(format!("GPU {device} has {:.1} GiB free for a {:.1} GiB budget after {} s: another process holds its memory",
                                       gib(free), gib(bytes[index]), VRAM_PATIENCE.as_secs()));
                }
                Some(bytes) => bytes[index],
                None => {
                    if free > grown.0 + (64 << 20) {
                        grown = (free, Instant::now());
                    }
                    // (a share of up to a quarter that another program keeps is left to it)
                    let settled = grown.1.elapsed() >= SETTLE && free >= total / 4 * 3;
                    if free >= total / 10 * 9 || settled || started.elapsed() >= VRAM_PATIENCE {
                        chosen.push(free.saturating_sub(margin(total)) / (256 << 20) * (256 << 20));
                        break;
                    }
                    total / 10 * 9
                }
            };
            report("vram", free, need);
            std::thread::sleep(Duration::from_millis(250));
        }
    }
    if chosen.iter().any(|&budget| budget < 1 << 30) {
        return Err(format!("too little free GPU memory for Straydog: {} GiB",
                           chosen.iter().map(|&bytes| format!("{:.1}", gib(bytes))).collect::<Vec<_>>().join(", ")));
    }
    Ok(chosen)
}

/// The budgets, batch and split for `args`. Budgets more than a tenth apart do not split every
/// layer across the GPUs (each would take the smaller share): the layers are split instead, in
/// proportion to the budgets.
pub(crate) fn plan(args: &WorkerArgs, report: &dyn Fn(&str, u64, u64)) -> Result<Plan, String> {
    let budgets = budgets(&args.devices, &args.vram_bytes, report)?;
    let batch = match args.prefill_batch_size.as_str() {
        // (as the configuration derives it from given budgets: 2048 for the resident models, see config::resolve)
        "auto" if straydog_engine::bundle::is_on_demand(&args.quant) => if budgets.iter().sum::<u64>() >= 48 << 30 { 4096 } else { 1024 },
        "auto" => if budgets.iter().sum::<u64>() >= 48 << 30 { 2048 } else { 1024 },
        given => given.parse().map_err(|_| format!("invalid Straydog prefill batch {given:?}"))?,
    };
    let (least, most) = (*budgets.iter().min().unwrap(), *budgets.iter().max().unwrap());
    let split = if args.split_mode == "layer" || most - least > most / 10 { SplitMode::Layer } else { SplitMode::Experts };
    Ok(Plan { budgets, batch, split })
}

pub fn main(args: WorkerArgs) -> Result<(), String> {
    let output = Output(Arc::new(Mutex::new(std::io::stdout())));
    let loading = Loading { output: output.clone(), last: Mutex::new((String::new(), Instant::now())) };
    let plan = plan(&args, &|stage, done, total| loading.report(stage, done, total))?;
    if straydog_engine::bundle::is_on_demand(&args.quant) {
        return crate::worker_on_demand::main(&args, &plan, output, &|stage, done, total| loading.report(stage, done, total));
    }
    let split = plan.split;
    let drafter = args.drafter.as_deref().filter(|_| args.speculative > 0);
    let report = |stage: &str, done, total| loading.report(stage, done, total);
    let opened = Instant::now();
    // The model files open while the GPUs initialize.
    let (group, weights) = std::thread::scope(|scope| {
        let weights = scope.spawn(|| Weights::open(&args.model, &args.quant, drafter, args.host_memory_bytes, &report));
        let group = Group::new(&args.devices, &plan.budgets, split)
            .and_then(|mut group| group.configure_gemm(args.gemm_min_batch, args.gemm == "fp16").map(|()| group));
        (group, weights.join().unwrap())
    });
    let (group, mut weights) = (group?, weights?);
    // With the shards' verity trees, what is read again after loading is checked block by block.
    if let Some(data_dir) = &args.data_dir {
        weights.open_trees(data_dir)?;
    }
    let open_seconds = opened.elapsed().as_secs_f64();
    loading.report("prepare", 0, 0);
    let resource = |name: &str| weights.resources.get(name).cloned().ok_or(format!("missing resource {name}"));
    let tokenizer = resource("tokenizer.json")?;
    let template = String::from_utf8(resource("chat_template.jinja")?).map_err(|exc| exc.to_string())?;
    let template = ChatTemplate::new(&template).map_err(|exc| exc.0)?;
    let settings = Settings {
        context: args.context,
        state_budget_bytes: args.state_ram_bytes,
        prefill_batch_size: plan.batch,
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
        "speculation": if model.speculative > 0 { json!([{"kind": "mtp", "drafts": model.speculative}]) } else { json!([]) },
        "placement": {"resident": model.resident, "partial": model.partial, "host_kv": model.host_kv,
                      "prefill_batch_size": model.prefill_batch_size, "vram_budgets": plan.budgets,
                      "split_mode": if split == SplitMode::Layer { "layer" } else { "experts" }},
        "load_seconds": seconds,
        "gpu": model.info()?,
    }));
    serve(Arc::new(Mutex::new(Engine { model, template })), text, output)
}

/// Read bounded frames until stdin closes or `shutdown`; reject concurrent generation.
pub(crate) fn serve<S: Served + 'static>(engine: Arc<Mutex<S>>, text: Arc<Text>, output: Output) -> Result<(), String> {
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
            Some("warm_state") => {
                if active.lock().unwrap().is_some() {
                    output.send(&json!({"id": id, "event": "error", "message": "Straydog is busy", "fatal": false}));
                    continue;
                }
                match engine.lock().unwrap().warm_state() {
                    Ok(state) => output.send(&json!({"id": id, "event": "warm_state", "state": state})),
                    Err(message) => output.send(&json!({"id": id, "event": "error", "message": message, "fatal": false})),
                }
            }
            Some("warm") => {
                let state = request.get("state").and_then(|state| serde_json::from_str(&pyjson::dumps(state)).ok())
                    .unwrap_or(Value::Null);
                if let Err(message) = engine.lock().unwrap().warm(&state) {
                    eprintln!("Straydog could not warm up: {message}");
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

/// One request's failure. Request errors need a successful reset; engine/device
/// failures stay fatal even where resetting their host state succeeds.
pub(crate) struct Failure {
    pub message: String,
    pub fatal: bool,
}

pub(crate) fn refused(message: impl Into<String>) -> Failure {
    Failure { message: message.into(), fatal: false }
}

pub(crate) fn engine_failed(message: impl Into<String>, site: &str) -> Failure {
    let message = message.into();
    // A void pass leaves the devices sound: the request fails and the engine resets, so the next
    // request restores from the checkpoints instead of waiting for a server restart. So does a
    // K/V that found no memory to grow into: only sequences that long are refused.
    // Nor does a prompt found too long before anything ran.
    let fatal = !message.ends_with(straydog_engine::arch::deepseek41::engine::VOID_PASS)
        && !message.ends_with(straydog_engine::model::KV_NO_MEMORY)
        && !message.ends_with("exceeds the allocated context");
    Failure { message: format!("{site}: {message}"), fatal }
}

/// Telemetry must not turn a completed generation into a failed request.
pub(crate) fn gpu_info_or_null(info: Result<Value, String>, site: &str) -> Value {
    info.unwrap_or_else(|message| {
        eprintln!("WARN Straydog GPU info unavailable at {site}: {message}");
        Value::Null
    })
}

// Tokenization and detokenization use host state owned by this request. Neither can
// invalidate the device; generate still resets the model before accepting another request.
impl From<TextError> for Failure {
    fn from(error: TextError) -> Self {
        refused(error.to_string())
    }
}

/// Run one chat request; returns its terminal frame (`done` or `error`).
fn generate<S: Served>(engine: &Mutex<S>, text: &Arc<Text>, output: &Output, id: &str, body: &pyjson::Value,
                       cancel: &Arc<AtomicBool>) -> Value {
    let started = Instant::now();
    let mut engine = engine.lock().unwrap();
    match engine.run(text, output, id, body, cancel, started) {
        Ok(done) => done,
        Err(failure) => {
            let mut fatal = failure.fatal;
            // A failed device (an illegal memory access) cannot be reset; the request still ends.
            if let Err(exc) = engine.reset() {
                eprintln!("Straydog could not reset after a failed generation: {exc}");
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
    // (a message's own text never writes the template's tokens: see Text::defuse)
    let messages = &chat::defuse_messages(messages, &|own| text.defuse(own).into_owned());
    let prompt = engine.template.render(messages, &tools, thinking, effort).map_err(|exc| refused(exc.0))?;
    let ids = text.encode(&prompt).map_err(Failure::from)?;
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
    let mut decode_error: Option<Failure> = None;
    let cancelled = || cancel.load(Ordering::SeqCst);
    let mut constraint = Constraint::from_request(body.get("constraint").unwrap_or(&Py::Null), text.token_bytes(), &END_TOKENS, thinking)
        .map_err(refused)?;
    // A reply that reasons ends it with the template's `</think>`, its one added token: the deltas go
    // without it and a reasoning_end event marks the place. The same text otherwise written (the
    // reasoning quoting a tag) stays the reasoning's.
    let mut think_close = if thinking { text.added_token(THINK_CLOSE) } else { None };
    // The held tokens can be more than the tokenizer's for the same text, and with no `max_tokens` the reply's room was all that `ids` left: a
    // client that sends none, as Dogmind App does, ended the worker on "exceeds the allocated context" on the first such reply.
    let agreed = engine.model.agree_with_held(&ids, &text.token_bytes());
    let agreed = if agreed.len() < context { agreed } else { ids.clone() };
    let max_new = max_new.min(context - agreed.len());
    let result = engine.model.generate_constrained(
        &agreed, max_new, &sampling, seed, &cancelled,
        &mut |done, total| output.send(&json!({"id": id, "event": "progress", "done": done, "total": total})),
        turn, checkpoint,
        &mut |token| {
            generated += 1;
            match stream.push(token) {
                Ok(delta) => {
                    let ended = think_close == Some(token);
                    let delta = if ended { delta.strip_suffix(THINK_CLOSE).map(str::to_string).unwrap_or(delta) } else { delta };
                    if !delta.is_empty() {
                        first_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
                        output.send(&json!({"id": id, "event": "delta", "content": delta, "tokens": generated}));
                    }
                    if ended {
                        think_close = None;
                        output.send(&json!({"id": id, "event": "reasoning_end", "tokens": generated}));
                    }
                }
                Err(exc) => {
                    decode_error.get_or_insert_with(|| exc.into());
                }
            }
        },
        constraint.as_mut(),
    );
    result.map_err(|message| engine_failed(message, concat!(file!(), ":", line!(), " model.generate")))?;
    if let Some(failure) = decode_error {
        return Err(failure);
    }
    let rest = stream.finish().map_err(Failure::from)?;
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
        "gpu": gpu_info_or_null(engine.model.info(), concat!(file!(), ":", line!(), " model.info")),
        "timings": timings,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // No GPU or model: an actual tokenizer error goes through the production request
    // completion/reset path, followed by another request on the same served instance.
    struct TokenizerOnly {
        resets: usize,
        reset_fails: bool,
    }

    impl Served for TokenizerOnly {
        fn run(&mut self, text: &Arc<Text>, _output: &Output, id: &str, body: &pyjson::Value,
               _cancel: &Arc<AtomicBool>, _started: Instant) -> Result<Value, Failure> {
            let prompt = body.as_str().unwrap();
            if prompt == "device-fault" {
                return Err(engine_failed("an illegal memory access was encountered", concat!(file!(), ":", line!(), " device fault")));
            }
            if prompt == "gpu-info-error" {
                return Ok(json!({"id": id, "event": "done", "gpu": gpu_info_or_null(Err("injected info error".into()), "test")}));
            }
            let tokens = text.encode(prompt).map_err(Failure::from)?;
            Ok(json!({"id": id, "event": "done", "tokens": tokens}))
        }

        fn reset(&mut self) -> Result<(), String> {
            self.resets += 1;
            if self.reset_fails { Err("device reset failed".into()) } else { Ok(()) }
        }
    }

    fn tokenizer() -> Arc<Text> {
        // Unknown text fails because the declared unknown token is absent. Known text
        // still encodes, so this exercises a failure and then real successful encoding.
        Arc::new(Text::new(br#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"ok":0},"unk_token":"[UNK]"}}"#).unwrap())
    }

    fn request(engine: &Mutex<TokenizerOnly>, text: &Arc<Text>, prompt: &str, id: &str) -> Value {
        let output = Output(Arc::new(Mutex::new(std::io::stdout())));
        generate(engine, text, &output, id, &pyjson::Value::Str(prompt.into()), &Arc::new(AtomicBool::new(false)))
    }

    #[test]
    fn tokenizer_failure_resets_and_next_request_succeeds() {
        let engine = Mutex::new(TokenizerOnly { resets: 0, reset_fails: false });
        let text = tokenizer();
        let failed = request(&engine, &text, "unknown", "first");
        assert_eq!(failed["event"], "error");
        assert_eq!(failed["fatal"], false);
        assert!(failed["message"].as_str().unwrap().starts_with("Straydog tokenizer:"));
        assert_eq!(engine.lock().unwrap().resets, 1);
        let next = request(&engine, &text, "ok", "second");
        assert_eq!(next["event"], "done");
        assert_eq!(next["id"], "second");
        assert_eq!(next["tokens"], json!([0]));
    }

    #[test]
    fn reset_failure_makes_a_request_error_fatal() {
        let engine = Mutex::new(TokenizerOnly { resets: 0, reset_fails: true });
        assert_eq!(request(&engine, &tokenizer(), "unknown", "reset-fails")["fatal"], true);
        assert_eq!(engine.lock().unwrap().resets, 1);
    }

    #[test]
    fn device_failure_stays_fatal_after_a_successful_host_reset() {
        let engine = Mutex::new(TokenizerOnly { resets: 0, reset_fails: false });
        let failed = request(&engine, &tokenizer(), "device-fault", "device-fails");
        assert_eq!(failed["fatal"], true);
        assert!(failed["message"].as_str().unwrap().contains("device fault: an illegal memory access"));
        assert_eq!(engine.lock().unwrap().resets, 1);
    }

    #[test]
    fn a_void_pass_is_not_fatal_but_other_engine_failures_are() {
        let void = straydog_engine::arch::deepseek41::engine::VOID_PASS;
        let failed = engine_failed(void, "site");
        assert!(!failed.fatal);
        assert_eq!(failed.message, format!("site: {void}"));
        assert!(engine_failed("an illegal memory access was encountered", "site").fatal);
        // (a prompt found too long before anything ran leaves the engine as it was)
        assert!(!engine_failed("Straydog prompt plus output exceeds the allocated context", "site").fatal);
    }

    #[test]
    fn gpu_info_failure_preserves_done_and_the_next_request() {
        let engine = Mutex::new(TokenizerOnly { resets: 0, reset_fails: false });
        let text = tokenizer();
        let done = request(&engine, &text, "gpu-info-error", "no-info");
        assert_eq!(done["event"], "done");
        assert!(done.get("gpu").unwrap().is_null());
        assert_eq!(request(&engine, &text, "ok", "next")["event"], "done");
        assert_eq!(engine.lock().unwrap().resets, 0);
        let info = json!({"free_bytes": 0});
        assert_eq!(gpu_info_or_null(Ok(info.clone()), "test"), info);
    }
}
