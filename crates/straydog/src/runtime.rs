//! The server's ownership of the single isolated worker: startup, a bounded queue of
//! generations run one at a time, interruption and cancellation, and failure reports.

use std::collections::{HashSet, VecDeque};
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use futures::future::BoxFuture;
use futures::FutureExt;
use serde_json::{json, Value};
use straydog_api::{Engine, EngineState, Generation, SubmitError};
use straydog_engine::pyjson;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::{mpsc, watch};

use crate::config::{Gemm, Settings, SplitMode};
use crate::worker::FRAME_BYTES;

/// Generations accepted at once: the running one and three waiting.
const QUEUE: usize = 4;
const STDERR_TAIL: usize = 8192;
/// Finished generations the dashboard keeps.
const RECENT: usize = 120;
/// A client that takes none of its generation's events for this long has gone: the generation
/// is cancelled (its events channel holds 64, so a reader that keeps up never waits this long).
const SEND_PATIENCE: Duration = Duration::from_secs(30);

/// The program and arguments that start the worker (isolated, in production).
pub type Command = (PathBuf, Vec<OsString>);

/// What runs before the worker starts and yields its command (building missing verity trees,
/// then the isolation command over them), reporting (stage, done, total).
pub type Prepare = Box<dyn FnOnce(&(dyn Fn(&str, u64, u64) + Sync)) -> Result<Command, String> + Send>;

/// One startup progress frame or the state the server reports.
#[derive(Debug, Clone)]
enum Phase {
    Idle,
    Loading(Value),
    Ready(Value),
    Failed(String),
}

enum Control {
    /// End this generation early; its done event still follows.
    Interrupt(String),
    /// The client is gone: end it and drop its events.
    Cancel(String),
}

struct Job {
    id: String,
    body: pyjson::Value,
    events: mpsc::Sender<Value>,
}

struct Shared {
    phase: Mutex<Phase>,
    /// The failure once the worker has failed (to start or later).
    failure: watch::Sender<Option<String>>,
    jobs: AtomicUsize,
    cancelled: Mutex<HashSet<String>>,
    stderr_tail: Mutex<Vec<u8>>,
    activity: Mutex<Activity>,
}

/// What the dashboard shows of the generations: the one running and the latest finished.
#[derive(Default)]
struct Activity {
    live: Option<Live>,
    recent: VecDeque<Value>,
    requests: u64,
    failed: u64,
    prompt_tokens: u64,
    cached_tokens: u64,
    completion_tokens: u64,
}

/// The generation running now.
struct Live {
    started: Instant,
    /// Prompt tokens computed and to compute, from the worker's progress events.
    prefill: (u64, u64),
    /// Tokens generated so far, and when the first came.
    tokens: u64,
    first_token: Option<Instant>,
}

impl Activity {
    fn record(&mut self, event: &Value) {
        let (usage, timings) = (&event["usage"], &event["timings"]);
        self.requests += 1;
        self.prompt_tokens += usage["prompt_tokens"].as_u64().unwrap_or(0);
        self.cached_tokens += usage["cached_tokens"].as_u64().unwrap_or(0);
        self.completion_tokens += usage["completion_tokens"].as_u64().unwrap_or(0);
        let time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0.0, |time| time.as_secs_f64());
        let mut entry = json!({"time": time, "finish_reason": event["finish_reason"], "prompt_tokens": usage["prompt_tokens"],
                               "cached_tokens": usage["cached_tokens"], "completion_tokens": usage["completion_tokens"]});
        for key in ["prefill_tokens_per_second", "decode_tokens_per_second", "ttft_seconds", "request_seconds",
                    "drafted_tokens", "accepted_tokens", "resumed_from"] {
            entry[key] = timings[key].clone();
        }
        if self.recent.len() == RECENT {
            self.recent.pop_front();
        }
        self.recent.push_back(entry);
    }

    fn snapshot_live(&self) -> Value {
        self.live.as_ref().map_or(Value::Null, |live| {
            let decoding = live.first_token.map_or(0.0, |first| first.elapsed().as_secs_f64());
            json!({"seconds": live.started.elapsed().as_secs_f64(), "prefill_done": live.prefill.0,
                   "prefill_total": live.prefill.1, "tokens": live.tokens, "decode_seconds": decoding})
        })
    }

    fn snapshot(&self) -> Value {
        json!({"live": self.snapshot_live(), "recent": self.recent, "requests": self.requests, "failed": self.failed,
               "prompt_tokens": self.prompt_tokens, "cached_tokens": self.cached_tokens,
               "completion_tokens": self.completion_tokens})
    }
}

pub struct Runtime {
    shared: Arc<Shared>,
    queue: mpsc::UnboundedSender<Job>,
    control: mpsc::UnboundedSender<Control>,
    driver: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// The `straydog worker` arguments for these settings.
pub fn worker_args(settings: &Settings) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec!["worker".into()];
    let mut push = |name: &str, value: String| {
        args.push(format!("--{name}").into());
        args.push(value.into());
    };
    push("model", settings.model.display().to_string());
    push("quant", settings.quant.clone());
    if let (Some(drafter), true) = (&settings.model_draft, settings.draft_max > 0) {
        push("drafter", drafter.display().to_string());
        push("speculative", settings.draft_max.to_string());
    }
    push("devices", settings.gpu.iter().map(usize::to_string).collect::<Vec<_>>().join(","));
    push("split-mode", if settings.split_mode == SplitMode::Layer { "layer" } else { "experts" }.into());
    push("context", settings.ctx_size.to_string());
    push("vram-bytes", settings.vram.bytes(settings.gpu.len())
        .map_or_else(|| "auto".into(), |bytes| bytes.iter().map(u64::to_string).collect::<Vec<_>>().join(",")));
    push("state-ram-bytes", ((settings.state_ram * (1u64 << 30) as f64) as u64).to_string());
    push("row-cache-bytes", (u64::from(settings.row_cache) << 20).to_string());
    push("prefill-batch-size", settings.batch_size.map_or_else(|| "auto".into(), |batch| batch.to_string()));
    push("gemm", if settings.gemm == Gemm::Fp16 { "fp16" } else { "fp32" }.into());
    push("gemm-min-batch", settings.gemm_min_batch.to_string());
    // Isolation hides the worker's cgroup limits from it; host-memory policies (page-locking
    // expert banks) must see the memory this process tree may actually use.
    push("host-memory-bytes", straydog_engine::host::host_memory_bytes().to_string());
    push("data-dir", settings.data_dir.display().to_string());
    if let Some(gib) = settings.expert_ram {
        push("host-expert-bytes", ((gib * (1u64 << 30) as f64) as u64).to_string());
    }
    args
}

/// The verity trees the worker reads: the pinned shards' in the data directory.
fn trees(settings: &Settings) -> Result<Vec<PathBuf>, String> {
    use straydog_engine::bundle;
    Ok(bundle::verity_pins(&settings.quant)?.iter().map(|(asset, _)| bundle::tree_path(&settings.data_dir, asset)).collect())
}

/// Whether verity trees are still to be built before the worker can start.
pub fn missing_trees(settings: &Settings) -> Result<bool, String> {
    Ok(trees(settings)?.iter().any(|tree| !tree.is_file()))
}

/// The worker's isolated command (its trees must exist).
pub fn worker_command(settings: &Settings) -> Result<Command, String> {
    straydog_isolation::worker_command(&worker_args(settings), &exposure(settings)?)
}

/// What the worker reads (never writes): the pinned files and their directories, never the API
/// key file, and the shards' verity trees in the data directory.
pub fn exposure(settings: &Settings) -> Result<straydog_isolation::Exposure, String> {
    use straydog_engine::bundle;
    let trees = trees(settings)?;
    if let Some(tree) = trees.iter().find(|tree| !tree.is_file()) {
        return Err(format!("no verity tree {}: it is built at the server's start, or by `straydog verify`", tree.display()));
    }
    let tree_directory = Some(settings.data_dir.join("verity")).filter(|directory| directory.is_dir());
    if bundle::is_on_demand(&settings.quant) {
        let pins = bundle::on_demand(&settings.quant)?;
        let mut files: Vec<PathBuf> =
            pins.shards.iter().chain(&pins.resources).map(|asset| settings.model.join(&asset.name)).collect();
        files.extend(trees);
        return Ok(straydog_isolation::Exposure {
            executable: std::env::current_exe().map_err(|exc| exc.to_string())?,
            files,
            directories: [settings.model.clone()].into_iter().chain(tree_directory).collect(),
            working_directory: settings.model.clone(),
            secrets: settings.api_key_file.iter().cloned().collect(),
        });
    }
    let pins = straydog_engine::bundle::bundle(&settings.quant)?;
    let mut files: Vec<PathBuf> =
        pins.shards.iter().chain(&pins.resources).map(|asset| settings.model.join(&asset.name)).collect();
    let mut directories = vec![settings.model.clone()];
    if let (Some(drafter), true) = (&settings.model_draft, settings.draft_max > 0) {
        files.push(drafter.join(&pins.drafter.name));
        directories.push(drafter.clone());
    }
    files.extend(trees);
    directories.extend(tree_directory);
    Ok(straydog_isolation::Exposure {
        executable: std::env::current_exe().map_err(|exc| exc.to_string())?,
        files,
        directories,
        working_directory: settings.model.clone(),
        secrets: settings.api_key_file.iter().cloned().collect(),
    })
}

/// The minimal environment the worker starts with: no proxy, token, secret or key variables.
fn worker_environment() -> Vec<(OsString, OsString)> {
    // On Windows the launcher passes on what the worker needs (and gives it its own TEMP).
    #[cfg(windows)]
    let keep: &[&str] = &straydog_isolation::WORKER_VARIABLES;
    #[cfg(not(windows))]
    let keep: &[&str] = &["PATH", "HOME", "USER", "LANG", "TZ", "TERM"];
    std::env::vars_os()
        .filter(|(name, _)| keep.iter().any(|kept| name.to_string_lossy().eq_ignore_ascii_case(kept)))
        .collect()
}

/// How the model was placed: what decides its speed.
pub fn ready_summary(ready: &Value) -> String {
    let placement = &ready["placement"];
    if let Some(slots) = placement["expert_slots"].as_array() {
        // Engines over the GPUs (each over one or several, splitting its layers), routed experts
        // in VRAM slots and a host tier.
        let gib = |value: &Value| value.as_u64().unwrap_or(0) as f64 / (1u64 << 30) as f64;
        let devices = placement["devices"].as_array().cloned().unwrap_or_default();
        let layers: Vec<String> = placement["layers"].as_array().into_iter().flatten().enumerate().map(|(index, range)| {
            let on: Vec<String> = devices.get(index).and_then(Value::as_array).into_iter().flatten().map(Value::to_string).collect();
            let gpus = if on.len() > 1 { format!(" on GPUs {}", on.join("+")) } else if on.len() == 1 { format!(" on GPU {}", on[0]) } else { String::new() };
            format!("{}-{}{gpus}", range[0], range[1].as_u64().unwrap_or(1).saturating_sub(1))
        }).collect();
        let slots: Vec<String> = slots.iter().map(|slots| slots.to_string()).collect();
        return format!(
            "Straydog ready: {} {}, context {}, layers {} with {} expert slots, {:.1} GiB host tier, prefill batch {}; {} using {:.1} of {:.1} GiB",
            ready["model"].as_str().unwrap_or("-"), ready["quantization"].as_str().unwrap_or("-"), ready["context"],
            layers.join(" | "), slots.join(" | "), gib(&placement["host_expert_bytes"]), placement["prefill_batch_size"],
            ready["gpu"]["name"].as_str().unwrap_or("GPU"), gib(&ready["gpu"]["allocated_bytes"]), gib(&ready["gpu"]["budget_bytes"]),
        );
    }
    let residency = if placement["resident"] == true {
        "fully resident"
    } else if placement["partial"] == true {
        "partially resident"
    } else {
        "bounded expert cache"
    };
    let drafts = ready["speculative_drafts"].as_u64().unwrap_or(0);
    let gib = |value: &Value| value.as_u64().unwrap_or(0) as f64 / (1u64 << 30) as f64;
    let gpu = &ready["gpu"];
    let names: Vec<&str> = gpu["name"].as_str().unwrap_or("GPU").split(" + ").collect();
    let devices = if names.len() > 1 && names.iter().all(|name| *name == names[0]) {
        format!("{}× {}", names.len(), names[0])
    } else {
        names.join(" + ")
    };
    format!(
        "Straydog ready: {}, context {}, {residency}, {}, K/V in {}, prefill batch {}; {devices} using {:.1} of {:.1} GiB",
        ready["quantization"].as_str().unwrap_or("-"),
        ready["context"],
        if drafts > 0 { format!("MTP {drafts}") } else { "no speculation".into() },
        if placement["host_kv"] == true { "host memory" } else { "VRAM" },
        placement["prefill_batch_size"],
        gib(&gpu["allocated_bytes"]),
        gib(&gpu["budget_bytes"]),
    )
}

/// A loading stage as the log names it.
fn stage_name(stage: &str, total: u64) -> String {
    let size = if total > 0 { format!(" ({:.1} GB)", total as f64 / 1e9) } else { String::new() };
    match stage {
        "authenticate" => format!("authenticating the model files{size}"),
        "drafter" => format!("authenticating the drafter{size}"),
        "prepare" => "placing the model on the GPUs".into(),
        "load" => format!("reading and authenticating the model into place{size}"),
        "warm" => "warming up the GPU code".into(),
        "vram" if total > 0 => format!("waiting for another process to free GPU memory ({:.1} GiB needed on a GPU)",
                                       total as f64 / (1u64 << 30) as f64),
        "vram" => "waiting for another process to free GPU memory".into(),
        other => format!("{other}{size}"),
    }
}

/// One line per finished generation: what was read, what was reused, how fast.
pub fn request_summary(event: &Value) -> String {
    let (usage, timings) = (&event["usage"], &event["timings"]);
    let number = |value: &Value, digits: usize| value.as_f64().map_or("-".into(), |value| format!("{value:.digits$}"));
    let count = |value: &Value| value.as_u64().map_or("-".into(), |value| value.to_string());
    format!(
        "Straydog request: prompt {} tokens (reused {} from {}), prefill {} tok/s, first token {} s, {} tokens at {} tok/s",
        count(&usage["prompt_tokens"]),
        count(&usage["cached_tokens"]),
        timings["resumed_from"].as_str().unwrap_or("-"),
        number(&timings["prefill_tokens_per_second"], 0),
        number(&timings["ttft_seconds"], 2),
        count(&usage["completion_tokens"]),
        number(&timings["decode_tokens_per_second"], 1),
    )
}

/// How long a stopped worker's processes and GPU memory are waited for.
///
/// Only the Linux wait reads it. Elsewhere there is no driver path to wait on.
#[cfg(target_os = "linux")]
const RELEASE_PATIENCE: Duration = Duration::from_secs(60);

/// The processes under `root`, it included, from /proc's parent links (Linux; elsewhere none).
fn descendants(root: u32) -> Vec<u32> {
    #[cfg(target_os = "linux")]
    {
        let mut children: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for entry in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
            let Some(pid) = entry.file_name().to_str().and_then(|name| name.parse::<u32>().ok()) else { continue };
            // The parent is the second field after the command name's closing parenthesis.
            let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else { continue };
            if let Some(parent) = stat.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().nth(1)?.parse().ok()) {
                children.entry(parent).or_default().push(pid);
            }
        }
        let (mut found, mut todo) = (Vec::new(), vec![root]);
        while let Some(pid) = todo.pop() {
            found.push(pid);
            todo.extend(children.remove(&pid).unwrap_or_default());
        }
        found
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = root;
        Vec::new()
    }
}

/// Whether `pid` still runs or still holds GPU memory: a process that has not exited (a zombie
/// has), or one whose memory the amdgpu driver is still freeing, which it does after the
/// process is gone (its entry under /sys/class/kfd/kfd/proc goes then).
#[cfg(target_os = "linux")]
fn holding(pid: u32) -> bool {
    let running = std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .is_ok_and(|stat| stat.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().next()) != Some("Z"));
    running || std::path::Path::new(&format!("/sys/class/kfd/kfd/proc/{pid}")).exists()
}

/// Wait until the worker's processes have exited and the GPU driver has let go of their
/// memory, so that a server started next finds the GPUs free (up to [`RELEASE_PATIENCE`]).
async fn released(holders: &[u32]) {
    #[cfg(target_os = "linux")]
    {
        let started = Instant::now();
        let mut said = false;
        loop {
            let held = holders.iter().filter(|&&pid| holding(pid)).count();
            if held == 0 {
                if said {
                    tracing::info!("Straydog's worker released the GPUs after {:.1} s", started.elapsed().as_secs_f64());
                }
                return;
            }
            if started.elapsed() >= RELEASE_PATIENCE {
                tracing::warn!("Straydog's worker has not released the GPUs after {} s", RELEASE_PATIENCE.as_secs());
                return;
            }
            if !said && started.elapsed() >= Duration::from_millis(500) {
                tracing::info!("Straydog waiting for its worker to exit and release the GPUs");
                said = true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = holders;
}

/// The worker's pipes, and the process tree it heads (the sandbox and the worker in it).
struct Process {
    child: Child,
    tree: straydog_isolation::ProcessTree,
    stdin: ChildStdin,
    stdout: Frames<BufReader<ChildStdout>>,
    /// The tree's processes as last seen (the sandbox's leave the child's process group).
    holders: Vec<u32>,
}

/// Newline-terminated JSON frames from the worker. A read may be dropped midway (a control
/// message or a timeout wins the select it waits in): the bytes of the frame read so far stay
/// here, and the next read continues the same frame.
struct Frames<R> {
    reader: R,
    partial: Vec<u8>,
}

impl<R: AsyncBufRead + Unpin> Frames<R> {
    fn new(reader: R) -> Frames<R> {
        Frames { reader, partial: Vec::new() }
    }

    async fn read(&mut self) -> Result<Value, String> {
        // read_until appends what it consumes to `partial` before it can be dropped.
        let room = (FRAME_BYTES + 1).saturating_sub(self.partial.len()) as u64;
        let read = (&mut self.reader).take(room).read_until(b'\n', &mut self.partial).await;
        let line = std::mem::take(&mut self.partial);
        match read {
            Ok(0) if line.is_empty() => Err("Straydog worker ended".into()),
            Ok(_) if line.len() > FRAME_BYTES || !line.ends_with(b"\n") => Err("Straydog worker sent an invalid frame".into()),
            Ok(_) => serde_json::from_slice::<Value>(&line)
                .ok()
                .filter(Value::is_object)
                .ok_or_else(|| "invalid Straydog response".into()),
            Err(exc) => Err(exc.to_string()),
        }
    }
}

impl Process {
    /// Add the tree's running processes to [`Process::holders`].
    fn note_holders(&mut self) {
        if let Some(root) = self.child.id() {
            for pid in descendants(root) {
                if !self.holders.contains(&pid) {
                    self.holders.push(pid);
                }
            }
        }
    }

    /// End the whole tree: ask, then after `grace` force, and reap the child.
    async fn end(&mut self, grace: Duration) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.tree.terminate();
            if tokio::time::timeout(grace, self.child.wait()).await.is_err() {
                let _ = self.tree.kill();
                let _ = self.child.wait().await;
            }
        }
    }
}

impl Process {
    async fn send(&mut self, frame: &pyjson::Value) -> Result<(), String> {
        let mut line = pyjson::dumps(frame).into_bytes();
        if line.len() >= FRAME_BYTES {
            return Err("Straydog request exceeds its IPC frame limit".into());
        }
        line.push(b'\n');
        self.stdin.write_all(&line).await.map_err(|exc| exc.to_string())?;
        self.stdin.flush().await.map_err(|exc| exc.to_string())
    }

    async fn read(&mut self) -> Result<Value, String> {
        self.stdout.read().await
    }
}

fn frame(entries: Vec<(&str, pyjson::Value)>) -> pyjson::Value {
    pyjson::Value::Object(entries.into_iter().map(|(key, value)| (key.to_string(), value)).collect())
}

fn new_id() -> String {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    format!("{:x}-{:x}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed))
}

impl Runtime {
    /// Start the worker from `command` and drive it; generations wait until it is ready.
    pub fn start(settings: &Settings, command: Command) -> Runtime {
        Runtime::start_prepared(settings, Box::new(move |_| Ok(command)))
    }

    /// [`Runtime::start`] once `prepare` has run (shown as loading) and given the command.
    pub fn start_prepared(settings: &Settings, prepare: Prepare) -> Runtime {
        let shared = Arc::new(Shared {
            phase: Mutex::new(Phase::Idle),
            failure: watch::Sender::new(None),
            jobs: AtomicUsize::new(0),
            cancelled: Mutex::new(HashSet::new()),
            stderr_tail: Mutex::new(Vec::new()),
            activity: Mutex::new(Activity::default()),
        });
        let (queue, jobs) = mpsc::unbounded_channel();
        let (control, controls) = mpsc::unbounded_channel();
        let driver = Driver {
            shared: shared.clone(),
            timeout: Duration::from_secs_f64(settings.startup_timeout),
            expected: (settings.ctx_size as u64, settings.quant.clone()),
            verbose: settings.verbose,
            warm_directory: Some(settings.data_dir.join("warm")),
        };
        let driver = tokio::spawn(driver.run(prepare, jobs, controls));
        Runtime { shared, queue, control, driver: Mutex::new(Some(driver)) }
    }

    /// The engine as the dashboard shows it: its state (with the ready frame, the latest GPU
    /// telemetry and the load time), the queue and the generations.
    pub fn snapshot(&self) -> Value {
        let mut snapshot = match &*self.shared.phase.lock().unwrap() {
            Phase::Idle => json!({"state": "loading", "progress": {}}),
            Phase::Loading(progress) => json!({"state": "loading", "progress": progress}),
            Phase::Ready(ready) => json!({"state": "ready", "ready": ready}),
            Phase::Failed(message) => json!({"state": "failed", "error": message}),
        };
        let activity = self.shared.activity.lock().unwrap();
        let running = usize::from(activity.live.is_some());
        snapshot["queued"] = self.shared.jobs.load(Ordering::SeqCst).saturating_sub(running).into();
        snapshot["activity"] = activity.snapshot();
        snapshot
    }

    /// The running generation as [`Runtime::snapshot`] shows it, or null.
    pub fn live(&self) -> Value {
        self.shared.activity.lock().unwrap().snapshot_live()
    }

    /// The failure, once the worker has failed to start or later failed: the runtime serves
    /// nothing more (the server then ends).
    pub fn failed(&self) -> impl std::future::Future<Output = String> + Send + 'static {
        let mut failure = self.shared.failure.subscribe();
        async move {
            let message = failure.wait_for(Option::is_some).await.map(|message| message.clone().unwrap_or_default());
            match message {
                Ok(message) => message,
                Err(_) => std::future::pending().await,
            }
        }
    }

    /// Finish the queued generations, stop the worker and wait until it has exited and the
    /// GPU driver has let go of its memory.
    pub async fn close(self) {
        let Runtime { queue, driver, .. } = self;
        drop(queue);
        let driver = driver.lock().unwrap().take();
        if let Some(driver) = driver {
            let _ = driver.await;
        }
    }
}

struct Driver {
    shared: Arc<Shared>,
    timeout: Duration,
    expected: (u64, String),
    verbose: bool,
    /// Where the expert store's contents are kept between runs (`<data-dir>/warm`), if anywhere.
    warm_directory: Option<PathBuf>,
}

/// How often a changed warm-up state is saved while the server runs.
const WARM_SAVE: Duration = Duration::from_secs(120);
/// The largest warm-up state read back.
const WARM_BYTES: u64 = 8 << 20;

impl Driver {
    fn set(&self, phase: Phase) {
        *self.shared.phase.lock().unwrap() = phase;
    }

    fn fail(&self, message: String) {
        tracing::error!("{message}");
        self.set(Phase::Failed(message.clone()));
        self.shared.failure.send_replace(Some(message));
    }

    async fn run(self, prepare: Prepare, mut jobs: mpsc::UnboundedReceiver<Job>,
                 mut controls: mpsc::UnboundedReceiver<Control>) {
        let launched = match self.prepare(prepare).await {
            Ok(command) => self.launch(command).await,
            Err(message) => Err((format!("Straydog failed to start ({message}); see the server log and restart the server"),
                                 Vec::new())),
        };
        let holders = match launched {
            Ok(mut process) => {
                self.restore_warm_state(&mut process).await;
                let changed = self.serve(&mut process, &mut jobs, &mut controls).await;
                if changed {
                    self.save_warm_state(&mut process).await;
                }
                process.note_holders();
                let shutdown = frame(vec![("id", pyjson::Value::Str("shutdown".into())),
                                          ("command", pyjson::Value::Str("shutdown".into()))]);
                let _ = process.send(&shutdown).await;
                if tokio::time::timeout(Duration::from_secs(10), process.child.wait()).await.is_err() {
                    process.end(Duration::from_secs(5)).await;
                }
                process.holders
            }
            Err((message, holders)) => {
                self.fail(message);
                holders
            }
        };
        // The worker is gone: every queued generation fails instead of waiting forever.
        jobs.close();
        while let Some(job) = jobs.recv().await {
            let _ = job.events.send(json!({"event": "error", "message": self.failure(), "fatal": true})).await;
            self.shared.jobs.fetch_sub(1, Ordering::SeqCst);
        }
        released(&holders).await;
    }

    fn failure(&self) -> String {
        match &*self.shared.phase.lock().unwrap() {
            Phase::Failed(message) => message.clone(),
            _ => "Straydog runtime is closed".into(),
        }
    }

    /// Run queued generations until the runtime closes or the worker fails.
    /// Serve generations until the runtime closes or the worker fails; whether the warm-up state
    /// changed since it was last saved (it is saved meanwhile every [`WARM_SAVE`] it changed).
    async fn serve(&self, process: &mut Process, jobs: &mut mpsc::UnboundedReceiver<Job>,
                   controls: &mut mpsc::UnboundedReceiver<Control>) -> bool {
        let mut changed = false;
        let mut save = tokio::time::interval(WARM_SAVE);
        save.tick().await;
        loop {
            let job = tokio::select! {
                job = jobs.recv() => match job { Some(job) => job, None => return changed },
                status = process.child.wait() => {
                    let status = status.map_or_else(|exc| exc.to_string(), |status| status.to_string());
                    self.fail(format!("Straydog's worker exited ({status}); restart the server"));
                    return false;
                }
                _ = save.tick(), if changed => {
                    self.save_warm_state(process).await;
                    changed = false;
                    continue;
                }
            };
            changed = true;
            let skipped = self.shared.cancelled.lock().unwrap().remove(&job.id);
            let result = if skipped { Ok(()) } else { self.generation(process, &job, controls).await };
            self.shared.activity.lock().unwrap().live = None;
            if let Err(message) = result {
                let message = format!("{message}; restart the server");
                let _ = job.events.send(json!({"event": "error", "message": message, "fatal": true})).await;
                self.shared.jobs.fetch_sub(1, Ordering::SeqCst);
                self.fail(message);
                process.end(Duration::from_secs(5)).await;
                return false;
            }
            self.shared.jobs.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// The file an engine placed as `ready` says keeps its warm-up state: one per model and
    /// placement (a state is a list of the experts a store held, only meaningful for the same).
    fn warm_path(&self, ready: &Value) -> Option<PathBuf> {
        use sha2::Digest;
        let directory = self.warm_directory.as_ref()?;
        let quant = &self.expected.1;
        let mut key = sha2::Sha256::new();
        key.update(b"straydog warm-up state 1\n");
        for shard in straydog_engine::bundle::shards(quant).ok()? {
            key.update(shard.sha256.as_bytes());
        }
        key.update(serde_json::to_vec(&ready["placement"]).ok()?);
        let key: String = key.finalize().iter().take(12).map(|byte| format!("{byte:02x}")).collect();
        Some(directory.join(format!("{quant}-{key}.json")))
    }

    fn ready_frame(&self) -> Option<Value> {
        match &*self.shared.phase.lock().unwrap() {
            Phase::Ready(ready) => Some(ready.clone()),
            _ => None,
        }
    }

    /// Hand the worker the state an earlier run saved for this model and placement: it warms up
    /// with it in the background, never ahead of a request.
    async fn restore_warm_state(&self, process: &mut Process) {
        let Some(path) = self.ready_frame().and_then(|ready| self.warm_path(&ready)) else { return };
        let state = std::fs::File::open(&path).ok().and_then(|file| {
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut std::io::Read::take(file, WARM_BYTES + 1), &mut bytes).ok()?;
            (bytes.len() as u64 <= WARM_BYTES).then(|| std::str::from_utf8(&bytes).ok().and_then(|text| pyjson::parse(text).ok()))?
        });
        let Some(state) = state.and_then(|saved| saved.get("state").cloned()) else { return };
        tracing::info!("Straydog warming up with {}", path.display());
        let warm = frame(vec![("id", pyjson::Value::Str("warm".into())), ("command", pyjson::Value::Str("warm".into())),
                              ("state", state)]);
        if let Err(message) = process.send(&warm).await {
            tracing::warn!("Straydog could not hand its worker the warm-up state: {message}");
        }
    }

    /// Ask the worker for its warm-up state and keep it in the data directory (written whole,
    /// then renamed), for the next start with this model and placement.
    async fn save_warm_state(&self, process: &mut Process) {
        let Some(path) = self.ready_frame().and_then(|ready| self.warm_path(&ready)) else { return };
        let request = frame(vec![("id", pyjson::Value::Str("warm-state".into())),
                                 ("command", pyjson::Value::Str("warm_state".into()))]);
        let reply = async {
            process.send(&request).await?;
            tokio::time::timeout(Duration::from_secs(30), process.read()).await.map_err(|_| "no reply".to_string())?
        }
        .await;
        let state = match reply {
            Ok(reply) if reply["id"] == "warm-state" && reply["event"] == "warm_state" => reply["state"].clone(),
            Ok(reply) => {
                tracing::warn!("Straydog's worker gave no warm-up state ({})", reply["message"].as_str().unwrap_or("unexpected reply"));
                return;
            }
            Err(message) => {
                tracing::warn!("Straydog's worker gave no warm-up state ({message})");
                return;
            }
        };
        if state.is_null() {
            return;
        }
        let saved = json!({"state": state, "saved": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |time| time.as_secs())});
        let partial = path.with_extension("json.part");
        let written = std::fs::create_dir_all(path.parent().unwrap())
            .and_then(|()| std::fs::write(&partial, serde_json::to_vec(&saved).unwrap_or_default()))
            .and_then(|()| std::fs::rename(&partial, &path));
        match written {
            Ok(()) => tracing::debug!("Straydog saved its warm-up state to {}", path.display()),
            Err(error) => tracing::warn!("Straydog could not save its warm-up state to {}: {error}", path.display()),
        }
    }

    /// Run the preparation on a blocking thread, its progress shown as loading.
    async fn prepare(&self, prepare: Prepare) -> Result<Command, String> {
        let shared = self.shared.clone();
        let logged = Mutex::new((String::new(), 0u64));
        tokio::task::spawn_blocking(move || {
            prepare(&|stage: &str, done: u64, total: u64| {
                let mut logged = logged.lock().unwrap();
                let quarter = (done.min(total) * 4).checked_div(total).unwrap_or(0);
                if stage != logged.0 {
                    tracing::info!("Straydog preparing: {stage}");
                    *logged = (stage.to_string(), 0);
                } else if quarter > logged.1 {
                    tracing::info!("Straydog preparing: {stage} {}%", quarter * 25);
                    logged.1 = quarter;
                }
                *shared.phase.lock().unwrap() = Phase::Loading(json!({"stage": stage, "done": done, "total": total}));
            })
        })
        .await
        .map_err(|exc| exc.to_string())?
    }

    /// Start the worker and wait until it is ready; on failure the message and the worker's
    /// processes (gone, or going, by then).
    async fn launch(&self, (program, args): Command) -> Result<Process, (String, Vec<u32>)> {
        let mut command = std::process::Command::new(&program);
        command
            .args(&args)
            .env_clear()
            .envs(worker_environment())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // Its own session (Linux) or console group (Windows), ended with the server. Spawned
        // here, on a runtime thread that lives as long as the worker.
        straydog_isolation::prepare(&mut command);
        let mut command = tokio::process::Command::from(command);
        command.kill_on_drop(true);
        let mut child = command.spawn()
            .map_err(|exc| (format!("Straydog failed to start ({}: {exc})", program.display()), Vec::new()))?;
        #[cfg(unix)]
        let tree = straydog_isolation::ProcessTree::from_pid(child.id().unwrap_or(0));
        #[cfg(windows)]
        let tree = straydog_isolation::ProcessTree::from_handle(child.raw_handle().unwrap_or(std::ptr::null_mut()));
        let tree = tree.map_err(|exc| (format!("Straydog failed to start ({exc})"), Vec::new()))?;
        let stderr = child.stderr.take().unwrap();
        let (shared, verbose) = (self.shared.clone(), self.verbose);
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).split(b'\n');
            while let Ok(Some(line)) = lines.next_segment().await {
                // Kept as bytes: a localized OS error on Windows is in the system code page.
                let mut tail = shared.stderr_tail.lock().unwrap();
                tail.extend_from_slice(&line);
                tail.push(b'\n');
                let excess = tail.len().saturating_sub(STDERR_TAIL);
                tail.drain(..excess);
                if verbose {
                    tracing::info!("worker: {}", String::from_utf8_lossy(&line));
                } else {
                    tracing::debug!("worker: {}", String::from_utf8_lossy(&line));
                }
            }
        });
        let mut process = Process {
            stdin: child.stdin.take().unwrap(),
            stdout: Frames::new(BufReader::new(child.stdout.take().unwrap())),
            child,
            tree,
            holders: Vec::new(),
        };
        self.set(Phase::Loading(json!({})));
        let started = Instant::now();
        let deadline = started + self.timeout;
        // The stage last logged, and the quarter of it reached.
        let mut logged = (String::new(), 0u64);
        let result = async {
            loop {
                let remaining = deadline.saturating_duration_since(Instant::now());
                let frame = tokio::time::timeout(remaining, process.read())
                    .await
                    .map_err(|_| "Straydog did not load within its startup timeout".to_string())??;
                // The worker (in its sandbox) is running: its processes are the ones to see gone.
                if process.holders.is_empty() {
                    process.note_holders();
                }
                if frame["event"] == "loading" {
                    let stage = frame["stage"].as_str().unwrap_or("").chars().take(32).collect::<String>();
                    let (done, total) = (frame["done"].as_u64().unwrap_or(0), frame["total"].as_u64().unwrap_or(0));
                    let quarter = (done.min(total) * 4).checked_div(total).unwrap_or(0);
                    if stage != logged.0 {
                        tracing::info!("Straydog loading: {}", stage_name(&stage, total));
                        logged = (stage.clone(), 0);
                    } else if quarter > logged.1 {
                        tracing::info!("Straydog loading: {} {}%", stage_name(&stage, 0), quarter * 25);
                        logged.1 = quarter;
                    }
                    self.set(Phase::Loading(json!({"stage": stage, "done": done, "total": total})));
                    continue;
                }
                let (context, quant) = &self.expected;
                if frame["event"] != "ready" || frame["protocol"] != 1 || frame["engine"] != "Straydog"
                    || frame["context"].as_u64() != Some(*context)
                    || frame["model"].as_str() != straydog_engine::bundle::model_of(quant).ok().as_deref()
                    || frame["quantization"].as_str() != Some(quant)
                {
                    return Err("Straydog startup contract mismatch".to_string());
                }
                return Ok(frame);
            }
        }
        .await;
        match result {
            Ok(mut ready) => {
                tracing::info!("{} (loaded in {:.0} s)", ready_summary(&ready), started.elapsed().as_secs_f64());
                if let Some(steps) = ready["load_seconds"].as_object() {
                    let steps: Vec<String> = steps.iter()
                        .map(|(step, seconds)| format!("{step} {:.2} s", seconds.as_f64().unwrap_or(0.0)))
                        .collect();
                    tracing::info!("Straydog load steps: {}", steps.join(", "));
                }
                ready["loaded_seconds"] = started.elapsed().as_secs_f64().into();
                self.set(Phase::Ready(ready));
                Ok(process)
            }
            Err(message) => {
                process.note_holders();
                process.end(Duration::from_secs(2)).await;
                // Let the last stderr lines arrive: the worker's last line names the cause.
                tokio::time::sleep(Duration::from_millis(200)).await;
                let tail = String::from_utf8_lossy(&self.shared.stderr_tail.lock().unwrap()).to_string();
                let reason = tail.lines().rev().map(str::trim).find(|line| !line.is_empty()).unwrap_or("").to_string();
                let reason = if reason.is_empty() { message } else { reason.chars().rev().take(500).collect::<Vec<_>>()
                    .into_iter().rev().collect() };
                Err((format!("Straydog failed to start ({reason}); see the server log and restart the server"), process.holders))
            }
        }
    }

    /// Follow a generation's events for the dashboard.
    fn observe(&self, kind: &str, event: &Value) {
        let mut activity = self.shared.activity.lock().unwrap();
        match kind {
            "done" => activity.record(event),
            "error" => activity.failed += 1,
            _ => {}
        }
        let Some(live) = activity.live.as_mut() else { return };
        match kind {
            "progress" => live.prefill = (event["done"].as_u64().unwrap_or(0), event["total"].as_u64().unwrap_or(0)),
            "delta" => {
                live.first_token.get_or_insert_with(Instant::now);
                live.tokens = event["tokens"].as_u64().unwrap_or(live.tokens);
            }
            _ => {}
        }
    }

    /// Run one generation, forwarding its events; Err only when the worker is unusable.
    async fn generation(&self, process: &mut Process, job: &Job, controls: &mut mpsc::UnboundedReceiver<Control>)
                        -> Result<(), String> {
        process.send(&frame(vec![("id", pyjson::Value::Str(job.id.clone())), ("command", pyjson::Value::Str("chat".into())),
                                 ("body", job.body.clone())])).await?;
        let cancel_frame = frame(vec![("id", pyjson::Value::Str(job.id.clone())),
                                      ("command", pyjson::Value::Str("cancel".into()))]);
        let mut forwarding = true;
        let mut cancel_deadline: Option<Instant> = None;
        self.shared.activity.lock().unwrap().live =
            Some(Live { started: Instant::now(), prefill: (0, 0), tokens: 0, first_token: None });
        loop {
            let wait = cancel_deadline.map_or(Duration::from_secs(86400), |deadline| {
                deadline.saturating_duration_since(Instant::now())
            });
            tokio::select! {
                read = tokio::time::timeout(wait, process.read()) => {
                    let event = read.map_err(|_| "Straydog did not acknowledge cancellation".to_string())??;
                    if event["id"].as_str() != Some(job.id.as_str()) {
                        return Err("Straydog generation request id mismatch".into());
                    }
                    if let Some(gpu) = event.get("gpu").filter(|gpu| gpu.is_object()) {
                        if let Phase::Ready(ready) = &mut *self.shared.phase.lock().unwrap() {
                            ready["gpu"] = gpu.clone();
                        }
                    }
                    let kind = event["event"].as_str().unwrap_or("").to_string();
                    if kind == "done" {
                        tracing::info!("{}", request_summary(&event));
                    }
                    self.observe(&kind, &event);
                    let fatal = kind == "error" && event["fatal"] == true;
                    if forwarding {
                        // Wait for room in the client's channel while still taking control
                        // messages: a client that reads slowly must not hold up an interrupt, a
                        // cancel or the queue.
                        let patience = tokio::time::sleep(SEND_PATIENCE);
                        tokio::pin!(patience);
                        let mut pending = Some(event);
                        while let Some(event) = pending.take() {
                            tokio::select! {
                                biased;
                                permit = job.events.reserve() => match permit {
                                    Ok(permit) => permit.send(event),
                                    // The client went away: end the generation, keep reading until it ends.
                                    Err(_) => stop_forwarding(process, &cancel_frame, &mut forwarding, &mut cancel_deadline).await?,
                                },
                                Some(control) = controls.recv() => {
                                    self.control(process, job, control, &cancel_frame, &mut forwarding, &mut cancel_deadline).await?;
                                    if forwarding {
                                        pending = Some(event);
                                    }
                                }
                                () = &mut patience => {
                                    tracing::warn!("Straydog client took no events for {} s: cancelling its generation",
                                                   SEND_PATIENCE.as_secs());
                                    stop_forwarding(process, &cancel_frame, &mut forwarding, &mut cancel_deadline).await?;
                                }
                            }
                        }
                    }
                    if kind == "done" || kind == "error" {
                        if fatal {
                            return Err("Straydog reported a fatal inference error".into());
                        }
                        return Ok(());
                    }
                }
                Some(control) = controls.recv() => {
                    self.control(process, job, control, &cancel_frame, &mut forwarding, &mut cancel_deadline).await?;
                }
            }
        }
    }

    /// A control message during `job`'s generation.
    async fn control(&self, process: &mut Process, job: &Job, control: Control, cancel_frame: &pyjson::Value,
                     forwarding: &mut bool, cancel_deadline: &mut Option<Instant>) -> Result<(), String> {
        match control {
            Control::Interrupt(id) if id == job.id => process.send(cancel_frame).await?,
            Control::Cancel(id) if id == job.id => stop_forwarding(process, cancel_frame, forwarding, cancel_deadline).await?,
            // A queued generation's client left: skip it when its turn comes.
            Control::Cancel(id) => {
                self.shared.cancelled.lock().unwrap().insert(id);
            }
            Control::Interrupt(_) => {}
        }
        Ok(())
    }
}

/// Stop sending a generation's events and end it; the worker has 10 s to acknowledge.
async fn stop_forwarding(process: &mut Process, cancel_frame: &pyjson::Value, forwarding: &mut bool,
                         cancel_deadline: &mut Option<Instant>) -> Result<(), String> {
    if *forwarding {
        *forwarding = false;
        process.send(cancel_frame).await?;
    }
    cancel_deadline.get_or_insert(Instant::now() + Duration::from_secs(10));
    Ok(())
}

/// Cancels a generation whose events stream is dropped before it ended.
struct CancelOnDrop {
    id: String,
    control: mpsc::UnboundedSender<Control>,
    ended: bool,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !self.ended {
            let _ = self.control.send(Control::Cancel(self.id.clone()));
        }
    }
}

impl Engine for Runtime {
    fn state(&self) -> EngineState {
        match &*self.shared.phase.lock().unwrap() {
            Phase::Idle => EngineState::Loading { progress: json!({}) },
            Phase::Loading(progress) => EngineState::Loading { progress: progress.clone() },
            Phase::Ready(ready) => EngineState::Ready {
                context: ready["context"].as_u64().unwrap_or(0),
                speculative: ready["speculative_drafts"].as_u64().unwrap_or(0),
                processing: self.shared.jobs.load(Ordering::SeqCst) > 0,
                placement: json!({"placement": ready["placement"], "speculative_drafts": ready["speculative_drafts"],
                                  "gpu": ready["gpu"], "sampling": ready["sampling"]}),
            },
            Phase::Failed(message) => EngineState::Failed(message.clone()),
        }
    }

    fn generate(&self, request: pyjson::Value) -> BoxFuture<'static, Result<Generation, SubmitError>> {
        let state = self.state();
        let shared = self.shared.clone();
        let (queue, control) = (self.queue.clone(), self.control.clone());
        async move {
            match state {
                EngineState::Ready { .. } => {}
                EngineState::Loading { .. } => return Err(SubmitError::Unavailable("Loading model".into())),
                EngineState::Failed(message) => return Err(SubmitError::Unavailable(message)),
            }
            if shared.jobs.fetch_add(1, Ordering::SeqCst) >= QUEUE {
                shared.jobs.fetch_sub(1, Ordering::SeqCst);
                return Err(SubmitError::Busy("Straydog generation queue is full".into()));
            }
            let id = new_id();
            let (events, receiver) = mpsc::channel(64);
            if queue.send(Job { id: id.clone(), body: request, events }).is_err() {
                shared.jobs.fetch_sub(1, Ordering::SeqCst);
                return Err(SubmitError::Unavailable("Straydog runtime is closed".into()));
            }
            let guard = CancelOnDrop { id: id.clone(), control: control.clone(), ended: false };
            let events = Box::pin(futures::stream::unfold((receiver, guard), |(mut receiver, mut guard)| async move {
                let event = receiver.recv().await?;
                // After its last event there is nothing left to cancel.
                guard.ended |= matches!(event["event"].as_str(), Some("done" | "error"));
                Some((event, (receiver, guard)))
            }));
            let interrupt = move || {
                let _ = control.send(Control::Interrupt(id.clone()));
            };
            Ok(Generation { events, interrupt: Box::new(interrupt) })
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame split across writes survives a read dropped between its halves (as when a
    /// control message wins the select), and frames after it stay in step.
    #[tokio::test]
    async fn a_dropped_read_keeps_the_partial_frame() {
        let (mut worker, server) = tokio::io::duplex(64);
        let mut frames = Frames::new(BufReader::new(server));
        worker.write_all(br#"{"id":"a","event":"tok"#).await.unwrap();
        tokio::select! {
            biased;
            _ = frames.read() => panic!("half a frame read as a whole"),
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
        worker.write_all(b"en\",\"text\":\"x\"}\n{\"id\":\"a\",\"event\":\"done\"}\n").await.unwrap();
        let first = frames.read().await.unwrap();
        assert_eq!(first, json!({"id": "a", "event": "token", "text": "x"}));
        assert_eq!(frames.read().await.unwrap()["event"], "done");
        drop(worker);
        assert_eq!(frames.read().await.unwrap_err(), "Straydog worker ended");
    }

    /// A frame cut off by the worker's end is refused, not taken as the end alone.
    #[tokio::test]
    async fn a_truncated_frame_is_invalid() {
        let (mut worker, server) = tokio::io::duplex(64);
        let mut frames = Frames::new(BufReader::new(server));
        worker.write_all(br#"{"id":"a""#).await.unwrap();
        drop(worker);
        assert_eq!(frames.read().await.unwrap_err(), "Straydog worker sent an invalid frame");
    }
}
