//! The server's ownership of the single isolated worker: startup, a bounded queue of
//! generations run one at a time, interruption and cancellation, and failure reports.

use std::collections::HashSet;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use futures::FutureExt;
use serde_json::{json, Value};
use straydog_api::{Engine, EngineState, Generation, SubmitError};
use straydog_engine::pyjson;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::mpsc;

use crate::config::{Gemm, Settings, SplitMode};
use crate::worker::FRAME_BYTES;

/// Generations accepted at once: the running one and three waiting.
const QUEUE: usize = 4;
const STDERR_TAIL: usize = 8192;

/// The program and arguments that start the worker (isolated, in production).
pub type Command = (PathBuf, Vec<OsString>);

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
    jobs: AtomicUsize,
    cancelled: Mutex<HashSet<String>>,
    stderr_tail: Mutex<Vec<u8>>,
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
    push("devices", settings.device.iter().map(usize::to_string).collect::<Vec<_>>().join(","));
    push("split-mode", if settings.split_mode == SplitMode::Layer { "layer" } else { "experts" }.into());
    push("context", settings.ctx_size.to_string());
    push("vram-bytes", ((settings.vram * (1u64 << 30) as f64) as u64).to_string());
    push("state-ram-bytes", ((settings.state_ram * (1u64 << 30) as f64) as u64).to_string());
    push("prefill-batch-size", settings.batch_size.to_string());
    push("gemm", if settings.gemm == Gemm::Fp16 { "fp16" } else { "fp32" }.into());
    push("gemm-min-batch", settings.gemm_min_batch.to_string());
    // Isolation hides the worker's cgroup limits from it; host-memory policies (page-locking
    // expert banks) must see the memory this process tree may actually use.
    push("host-memory-bytes", straydog_engine::host::host_memory_bytes().to_string());
    args
}

/// What the worker reads: the pinned files and their directories, never the API key file.
pub fn exposure(settings: &Settings) -> Result<straydog_isolation::Exposure, String> {
    let pins = straydog_engine::bundle::bundle(&settings.quant)?;
    let mut files: Vec<PathBuf> =
        pins.shards.iter().chain(&pins.resources).map(|asset| settings.model.join(&asset.name)).collect();
    let mut directories = vec![settings.model.clone()];
    if let (Some(drafter), true) = (&settings.model_draft, settings.draft_max > 0) {
        files.push(drafter.join(&pins.drafter.name));
        directories.push(drafter.clone());
    }
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
    let keep: &[&str] = if cfg!(windows) {
        &["PATH", "PATHEXT", "SystemRoot", "SystemDrive", "windir", "TEMP", "TMP", "LOCALAPPDATA", "NUMBER_OF_PROCESSORS",
          "PROCESSOR_ARCHITECTURE", "CUDA_PATH"]
    } else {
        &["PATH", "HOME", "USER", "LANG", "TZ", "TERM"]
    };
    std::env::vars_os()
        .filter(|(name, _)| keep.iter().any(|kept| name.to_string_lossy().eq_ignore_ascii_case(kept)))
        .collect()
}

/// How the model was placed: what decides its speed.
pub fn ready_summary(ready: &Value) -> String {
    let placement = &ready["placement"];
    let residency = if placement["resident"] == true {
        "fully resident"
    } else if placement["partial"] == true {
        "partially resident"
    } else {
        "bounded expert cache"
    };
    let drafts = ready["speculative_drafts"].as_u64().unwrap_or(0);
    format!(
        "Straydog ready: {}, context {}, {residency}, {}, K/V in {}, prefill batch {}",
        ready["quantization"].as_str().unwrap_or("-"),
        ready["context"],
        if drafts > 0 { format!("MTP {drafts}") } else { "no speculation".into() },
        if placement["host_kv"] == true { "host memory" } else { "VRAM" },
        placement["prefill_batch_size"],
    )
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

/// The worker's pipes.
struct Process {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
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
        let mut line = Vec::new();
        let read = (&mut self.stdout).take(FRAME_BYTES as u64 + 1).read_until(b'\n', &mut line).await;
        match read {
            Ok(0) => Err("Straydog worker ended".into()),
            Ok(_) if line.len() > FRAME_BYTES || !line.ends_with(b"\n") => Err("Straydog worker sent an invalid frame".into()),
            Ok(_) => serde_json::from_slice::<Value>(&line)
                .ok()
                .filter(Value::is_object)
                .ok_or_else(|| "invalid Straydog response".into()),
            Err(exc) => Err(exc.to_string()),
        }
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
        let shared = Arc::new(Shared {
            phase: Mutex::new(Phase::Idle),
            jobs: AtomicUsize::new(0),
            cancelled: Mutex::new(HashSet::new()),
            stderr_tail: Mutex::new(Vec::new()),
        });
        let (queue, jobs) = mpsc::unbounded_channel();
        let (control, controls) = mpsc::unbounded_channel();
        let driver = Driver {
            shared: shared.clone(),
            timeout: Duration::from_secs_f64(settings.startup_timeout),
            expected: (settings.ctx_size as u64, settings.quant.clone()),
            verbose: settings.verbose,
        };
        let driver = tokio::spawn(driver.run(command, jobs, controls));
        Runtime { shared, queue, control, driver: Mutex::new(Some(driver)) }
    }

    /// Finish the queued generations, stop the worker and wait until it has exited (its
    /// GPU memory is free again).
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
}

impl Driver {
    fn set(&self, phase: Phase) {
        *self.shared.phase.lock().unwrap() = phase;
    }

    fn fail(&self, message: String) {
        tracing::error!("{message}");
        self.set(Phase::Failed(message));
    }

    async fn run(self, command: Command, mut jobs: mpsc::UnboundedReceiver<Job>,
                 mut controls: mpsc::UnboundedReceiver<Control>) {
        match self.launch(command).await {
            Ok(mut process) => {
                self.serve(&mut process, &mut jobs, &mut controls).await;
                let shutdown = frame(vec![("id", pyjson::Value::Str("shutdown".into())),
                                          ("command", pyjson::Value::Str("shutdown".into()))]);
                let _ = process.send(&shutdown).await;
                if tokio::time::timeout(Duration::from_secs(10), process.child.wait()).await.is_err() {
                    // kill() also waits for the exit.
                    let _ = process.child.kill().await;
                }
            }
            Err(message) => self.fail(message),
        }
        // The worker is gone: every queued generation fails instead of waiting forever.
        jobs.close();
        while let Some(job) = jobs.recv().await {
            let _ = job.events.send(json!({"event": "error", "message": self.failure(), "fatal": true})).await;
            self.shared.jobs.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn failure(&self) -> String {
        match &*self.shared.phase.lock().unwrap() {
            Phase::Failed(message) => message.clone(),
            _ => "Straydog runtime is closed".into(),
        }
    }

    /// Run queued generations until the runtime closes or the worker fails.
    async fn serve(&self, process: &mut Process, jobs: &mut mpsc::UnboundedReceiver<Job>,
                   controls: &mut mpsc::UnboundedReceiver<Control>) {
        loop {
            let job = tokio::select! {
                job = jobs.recv() => match job { Some(job) => job, None => return },
                status = process.child.wait() => {
                    let status = status.map_or_else(|exc| exc.to_string(), |status| status.to_string());
                    self.fail(format!("Straydog's worker exited ({status}); restart the server"));
                    return;
                }
            };
            let skipped = self.shared.cancelled.lock().unwrap().remove(&job.id);
            let result = if skipped { Ok(()) } else { self.generation(process, &job, controls).await };
            if let Err(message) = result {
                let message = format!("{message}; restart the server");
                let _ = job.events.send(json!({"event": "error", "message": message, "fatal": true})).await;
                self.shared.jobs.fetch_sub(1, Ordering::SeqCst);
                self.fail(message);
                let _ = process.child.kill().await;
                return;
            }
            self.shared.jobs.fetch_sub(1, Ordering::SeqCst);
        }
    }

    async fn launch(&self, (program, args): Command) -> Result<Process, String> {
        let mut command = tokio::process::Command::new(&program);
        command
            .args(&args)
            .env_clear()
            .envs(worker_environment())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = command.spawn().map_err(|exc| format!("Straydog failed to start ({}: {exc})", program.display()))?;
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
            stdout: BufReader::new(child.stdout.take().unwrap()),
            child,
        };
        self.set(Phase::Loading(json!({})));
        let deadline = Instant::now() + self.timeout;
        let result = async {
            loop {
                let remaining = deadline.saturating_duration_since(Instant::now());
                let frame = tokio::time::timeout(remaining, process.read())
                    .await
                    .map_err(|_| "Straydog did not load within its startup timeout".to_string())??;
                if frame["event"] == "loading" {
                    let stage = frame["stage"].as_str().unwrap_or("").chars().take(32).collect::<String>();
                    tracing::debug!("Straydog loading: {stage}");
                    self.set(Phase::Loading(json!({"stage": stage, "done": frame["done"].as_u64().unwrap_or(0),
                                                   "total": frame["total"].as_u64().unwrap_or(0)})));
                    continue;
                }
                let (context, quant) = &self.expected;
                if frame["event"] != "ready" || frame["protocol"] != 1 || frame["engine"] != "Straydog"
                    || frame["context"].as_u64() != Some(*context) || frame["model"] != straydog_engine::bundle::MODEL
                    || frame["quantization"].as_str() != Some(quant)
                {
                    return Err("Straydog startup contract mismatch".to_string());
                }
                return Ok(frame);
            }
        }
        .await;
        match result {
            Ok(ready) => {
                tracing::info!("{}", ready_summary(&ready));
                self.set(Phase::Ready(ready));
                Ok(process)
            }
            Err(message) => {
                let _ = process.child.kill().await;
                // Let the last stderr lines arrive: the worker's last line names the cause.
                tokio::time::sleep(Duration::from_millis(200)).await;
                let tail = String::from_utf8_lossy(&self.shared.stderr_tail.lock().unwrap()).to_string();
                let reason = tail.lines().rev().map(str::trim).find(|line| !line.is_empty()).unwrap_or("").to_string();
                let reason = if reason.is_empty() { message } else { reason.chars().rev().take(500).collect::<Vec<_>>()
                    .into_iter().rev().collect() };
                Err(format!("Straydog failed to start ({reason}); see the server log and restart the server"))
            }
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
                    let fatal = kind == "error" && event["fatal"] == true;
                    if forwarding && job.events.send(event).await.is_err() {
                        // The client went away: end the generation, keep reading until it ends.
                        forwarding = false;
                        process.send(&cancel_frame).await?;
                        cancel_deadline = Some(Instant::now() + Duration::from_secs(10));
                    }
                    if kind == "done" || kind == "error" {
                        if fatal {
                            return Err("Straydog reported a fatal inference error".into());
                        }
                        return Ok(());
                    }
                }
                Some(control) = controls.recv() => match control {
                    Control::Interrupt(id) if id == job.id => process.send(&cancel_frame).await?,
                    Control::Cancel(id) if id == job.id => {
                        forwarding = false;
                        process.send(&cancel_frame).await?;
                        cancel_deadline.get_or_insert(Instant::now() + Duration::from_secs(10));
                    }
                    // A queued generation's client left: skip it when its turn comes.
                    Control::Cancel(id) => { self.shared.cancelled.lock().unwrap().insert(id); }
                    Control::Interrupt(_) => {}
                },
            }
        }
    }
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
                placement: json!({"placement": ready["placement"], "speculative_drafts": ready["speculative_drafts"],
                                  "gpu": ready["gpu"]}),
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
