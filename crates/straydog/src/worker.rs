//! `straydog worker`: the isolated engine process behind the server's private pipes.
//!
//! It opens no network listener and never executes tools. Requests arrive as JSON lines on
//! stdin; events leave as JSON lines on stdout. One generation runs at a time on its own
//! thread, so a cancellation is read while it runs. Every model byte is authenticated first.

use std::collections::BTreeMap;
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use crate::behind;
use crate::images;

use clap::Args;
use serde_json::{json, Value};
use straydog_engine::chat::{self, ChatTemplate};
use straydog_engine::constraint::Constraint;
use straydog_engine::reply::THINK_CLOSE;
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings, WarmStart, END_TOKENS, ROUTED_EXPERTS, ROUTED_LAYERS};
use straydog_engine::pyjson;
use straydog_engine::qwen_vision::{Encoder, ImageSpan};
use straydog_engine::routes::{self, Routes};
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
    /// The bundle's three labels ([`straydog_engine::bundle::Id`]).
    #[arg(long)]
    pub model_name: String,
    #[arg(long)]
    pub provider: String,
    #[arg(long)]
    pub quant: String,
    #[arg(long)]
    pub drafter: Option<PathBuf>,
    /// Qwen3.8's vision encoder and projector.
    #[arg(long)]
    pub mmproj: Option<PathBuf>,
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
    /// Parent cgroup counters, exposed read-only for a live prefetch budget inside isolation.
    #[arg(long = "host-memory-limit-file", hide = true)]
    pub host_memory_limit_files: Vec<PathBuf>,
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

    /// A frame that carries bytes ([`crate::frames`]): its header, then `body`, written together so that no other frame comes between.
    pub(crate) fn send_with_body(&self, frame: &Value, body: &[u8]) {
        let mut out = self.0.lock().unwrap();
        let _ = crate::frames::write_with_body(&mut *out, frame, body);
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
    /// The vision encoder, or why images are refused.
    vision: Result<Encoder, String>,
    /// The rows of the images encoded last (by their bytes' digest), newest last: a conversation sends its images
    /// again with every turn.
    encoded: Vec<([u8; 32], Arc<Vec<f32>>)>,
    /// The route counts of the runs before this one, from the warm-up state it started with (a partially resident model's: [`straydog_engine::routes`]).
    earlier: Option<Routes>,
    /// The draft widths that state kept, by placement, which the next state keeps beside this placement's own ([`routes::with_drafts`]).
    earlier_drafts: BTreeMap<String, u64>,
    /// This placement's key in those ([`placement_key`]).
    placement: String,
    /// The requests this run has begun: a warm start's slot fill ends with the first ([`fill_slots`]).
    requests: u64,
    /// The handle the request loop shares, for a locking that memory freed since the start allows ([`Engine::lock_more`]).
    me: Weak<Mutex<Engine>>,
    /// The GPUs' budgets were `auto`: GPU memory freed since the start is taken too ([`Engine::more_vram`]).
    #[cfg(not(windows))]
    vram_auto: bool,
    #[cfg(windows)]
    vram_limits: Vec<Option<VramLimit>>,
    /// Original per-device ceilings for explicit budgets, independent of pressure cooldowns.
    #[cfg(windows)]
    vram_caps: Option<Vec<u64>>,
}

/// Bytes of image rows kept for the turns after ([`Engine::encoded`]): about 40 full screens (1440x900 is 1260 rows of 2560 values).
const ENCODED_BYTES: usize = 512 << 20;

/// What a model's images are refused with when it has no encoder file.
const NO_VISION_FILE: &str = "its vision encoder file is not beside it: `straydog setup` fetches it, or give --mmproj FILE";

/// A loaded model as the request loop drives it: one request at a time, and a reset after a
/// failed one.
pub(crate) trait Served: Send {
    fn run(&mut self, text: &Arc<Text>, output: &Output, id: &str, body: &pyjson::Value, cancel: &Arc<AtomicBool>,
           started: Instant) -> Result<Value, Failure>;
    fn reset(&mut self) -> Result<(), String>;
    /// After a request's last frame was sent, if the engine is still free: what the engine does between requests.
    fn after_request(&mut self) {}

    /// What a later start warms up with (the expert store's contents), or null.
    fn warm_state(&mut self) -> Result<Value, String> {
        Ok(Value::Null)
    }

    /// Warm up with an earlier run's state, in the background, never ahead of a request.
    fn warm(&mut self, _state: &Value) -> Result<(), String> {
        Ok(())
    }

    /// The state of the system block the held sequence began with, as bytes, and what names it (its tokens and their digest): none when the
    /// model keeps none.
    fn prefix_state(&mut self) -> Result<Option<(Value, Vec<u8>)>, String> {
        Ok(None)
    }

    /// The state of the conversation the held sequence is, at the place its last reply began, as bytes, and what names it: none when the model keeps
    /// none or holds no conversation worth keeping. `limit` is the most bytes the server has room for: a state over it is no state given, and the
    /// reason is the error.
    fn conversation_state(&mut self, _limit: u64) -> Result<Option<(Value, Vec<u8>)>, String> {
        Ok(None)
    }

    /// Take a system block's state an earlier run saved, in the background, never ahead of a request. `read_seconds` is what the server
    /// spent getting it, `arrived` when its frame began.
    fn prefix_in_background(_engine: &Arc<Mutex<Self>>, _state: Vec<u8>, _read_seconds: f64, _arrived: Instant)
    where
        Self: Sized + 'static,
    {
    }

    /// [`Served::warm`] for a model whose warm-up steps run on a thread of their own, each under the engine's lock (`engine` is the handle the request loop
    /// shares: a request that comes meanwhile waits for one step at most).
    fn warm_in_background(engine: &Arc<Mutex<Self>>, state: &Value) -> Result<(), String>
    where
        Self: Sized + 'static,
    {
        engine.lock().unwrap().warm(state)
    }
}

impl Served for Engine {
    fn run(&mut self, text: &Arc<Text>, output: &Output, id: &str, body: &pyjson::Value, cancel: &Arc<AtomicBool>,
           started: Instant) -> Result<Value, Failure> {
        self.requests += 1;
        self.more_vram(false, "request").map_err(|message| engine_failed(message, "adjusting the GPU memory budget"))?;
        let result = run(self, text, output, id, body, cancel, started);
        #[cfg(windows)]
        if let Err(error) = &result {
            if !error.fatal && error.message.ends_with(straydog_engine::gpu::HYBRID_WAIT_RECOVERED) {
                let lowered = self.backoff_vram().map_err(|message| engine_failed(message, "reducing the GPU memory budget after a wait timeout"))?;
                return Err(refused(if lowered {
                    "The GPU wait exceeded its limit. The memory budget was lowered and this request was stopped. You can retry with the next request."
                } else {
                    "The GPU wait exceeded its limit and this request was stopped. The GPU state was reset; you can retry with the next request."
                }));
            }
        }
        result
    }

    fn reset(&mut self) -> Result<(), String> {
        self.model.reset()
    }

    /// Memory freed since the start: GPU memory to the expert slots, host memory to more page-locked layers.
    fn after_request(&mut self) {
        if let Err(error) = self.more_vram(true, "after_request") {
            eprintln!("WARN Straydog VRAM: {error}");
        }
        self.lock_more();
    }

    /// A partially resident model's route counts: the earlier runs' aged by what this one adds, and its own; and the draft widths by placement: the
    /// earlier state's, with this placement's set to the width in force once the passes have chosen it (before, its earlier one stays). Null for a model
    /// that holds all its experts, and before any route was counted.
    fn warm_state(&mut self) -> Result<Value, String> {
        if !self.model.partial {
            return Ok(Value::Null);
        }
        let run = self.model.route_counts()?;
        let earlier = self.earlier.clone().unwrap_or_else(|| Routes::new(ROUTED_LAYERS, ROUTED_EXPERTS));
        let kept = earlier.aged_with(&run);
        if kept.total() == 0 {
            return Ok(Value::Null);
        }
        Ok(routes::with_drafts(kept.to_state(), &self.earlier_drafts, &self.placement, self.model.chosen_drafts()))
    }

    fn prefix_state(&mut self) -> Result<Option<(Value, Vec<u8>)>, String> {
        Ok(self.model.export_prefix()?.map(|prefix| (json!({"tokens": prefix.tokens, "digest": prefix.digest}), prefix.bytes)))
    }

    fn conversation_state(&mut self, limit: u64) -> Result<Option<(Value, Vec<u8>)>, String> {
        let Some((tokens, size)) = self.model.conversation_size() else { return Ok(None) };
        if size > limit {
            return Err(format!("the state of {tokens} tokens is {} MB, and there is room for {} MB", size >> 20, limit >> 20));
        }
        Ok(self.model.export_conversation()?.map(|state| (json!({"tokens": state.tokens, "digest": state.digest}), state.bytes)))
    }

    fn prefix_in_background(engine: &Arc<Mutex<Engine>>, state: Vec<u8>, read_seconds: f64, arrived: Instant) {
        let weak = Arc::downgrade(engine);
        let spawned = std::thread::Builder::new().name("straydog-prefix".into()).spawn(move || restore_prefix(&weak, &state, read_seconds, arrived));
        if let Err(error) = spawned {
            eprintln!("Straydog prefix: not restored ({error})");
        }
    }

    fn warm_in_background(engine: &Arc<Mutex<Engine>>, state: &Value) -> Result<(), String> {
        // The warm-up holds the engine for its passes, and the request loop that is asking must go on reading: the thread behind "ready" begins
        // the warm start when the warm-up is over, as it was when the warm-up came before "ready".
        if behind::hold_warm_start(state) {
            return Ok(());
        }
        begin_warm_start(engine, state)
    }
}

/// The first step of a warm start under the engine's lock, then its background work on a thread of its own ([`warm_start`]).
fn begin_warm_start(engine: &Arc<Mutex<Engine>>, state: &Value) -> Result<(), String> {
    let began = Instant::now();
    let (start, requests) = {
        let mut engine = engine.lock().unwrap();
        match engine.warm_begin(state)? {
            Some(start) => (start, engine.requests),
            None => return Ok(()),
        }
    };
    spawn_warm_start(&Arc::downgrade(engine), start, requests, began)
}

/// The background work of a warm start whose first step is done ([`Engine::warm_begin`]).
fn spawn_warm_start(engine: &Weak<Mutex<Engine>>, start: WarmStart, requests: u64, began: Instant) -> Result<(), String> {
    let weak = engine.clone();
    std::thread::Builder::new().name("straydog-warm".into()).spawn(move || warm_start(&weak, start, requests, began))
        .map(drop).map_err(|exc| exc.to_string())
}

/// Swaps one step of a warm start's slot fill may issue (about 64 MB of experts at IQ1_M, which the device's copy thread issues without the engine's
/// compute thread waiting), and the pause between steps that lets a request take the engine.
const WARM_SWAPS: u64 = 64;
const WARM_PAUSE: Duration = Duration::from_millis(2);
/// How long a step waits for the engine before it takes it to be a request's.
const WARM_PATIENCE: Duration = Duration::from_millis(100);
/// The longest the slot fill goes on: it ends when the slots hold what the scores want, well within this.
const WARM_LONGEST: Duration = Duration::from_secs(120);

impl Engine {
    /// Reconcile a partial model's budget with current headroom. Only the pause after a completed request may grow it.
    fn more_vram(&mut self, allow_grow: bool, phase: &str) -> Result<(), String> {
        if !self.model.partial {
            return Ok(());
        }
        #[cfg(windows)]
        return adjust_windows_vram(&mut self.model, &mut self.vram_limits, self.vram_caps.as_deref(), self.requests, allow_grow, phase);
        #[cfg(not(windows))]
        {
            let _ = phase;
            if !self.vram_auto || !allow_grow { return Ok(()); }
            let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
            for (index, (ordinal, used, budget)) in self.model.vram_budgets().unwrap_or_default().into_iter().enumerate() {
                let Ok(free) = self.model.vram_free(index) else { continue };
                let Ok((_, _, total)) = straydog_gpu::device_info(ordinal) else { continue };
                let wanted = auto_budget(cfg!(windows), free + used, total);
                if wanted < budget + (512 << 20) {
                    continue;
                }
                match self.model.raise_vram(index, wanted) {
                    Ok((before, after)) => eprintln!("Straydog VRAM: GPU {ordinal}'s budget {:.1} -> {:.1} GiB, memory other programs freed since the start \
                                                      ({:.1} GiB free); expert slots {before} -> {after}", gib(budget), gib(wanted), gib(free)),
                    Err(error) => eprintln!("WARN Straydog VRAM: GPU {ordinal}'s budget stays {:.1} GiB: {error}", gib(budget)),
                }
            }
            Ok(())
        }
    }

    #[cfg(windows)]
    fn backoff_vram(&mut self) -> Result<bool, String> {
        if !self.model.partial { return Ok(false); }
        let devices = self.model.vram_budgets()?;
        validate_vram_caps(self.vram_caps.as_deref(), devices.len())?;
        self.vram_limits.resize(devices.len(), None);
        let mut lowered = false;
        for (index, (ordinal, _, budget)) in devices.into_iter().enumerate() {
            let cap = self.vram_caps.as_ref().map(|caps| caps[index]);
            let wanted = budget.saturating_sub(VRAM_GROWTH).min(cap.unwrap_or(u64::MAX));
            let (before, after) = self.model.resize_vram(index, wanted)?;
            let actual = self.model.vram_budgets()?[index].2;
            if actual >= budget {
                return Err(vram_refused(ordinal, "the memory budget could not be reduced after the wait timeout"));
            }
            self.vram_limits[index] = Some(VramLimit::new(actual, self.requests));
            eprintln!("Straydog VRAM: GPU {ordinal}'s budget {budget} -> {actual} bytes after a wait timeout; expert slots {before} -> {after}");
            lowered = true;
        }
        Ok(lowered)
    }

    /// Memory freed since the start page-locks more layers ([`Model::more_bank_locks`]), on a thread of its own behind the requests as the start's
    /// locking is ([`lock_banks_behind`]): asked after each request's last frame ([`Served::after_request`]).
    fn lock_more(&mut self) {
        if !self.model.partial {
            return;
        }
        // (this run's routes with the earlier runs' aged by them, as a warm state keeps them)
        let earlier = self.earlier.clone();
        let prior = |model: &Model| model.route_counts().ok().map(|run| earlier.as_ref().map_or_else(|| run.clone(), |earlier| earlier.aged_with(&run)));
        let Some(job) = self.model.more_bank_locks(prior) else { return };
        let (weak, requests) = (self.me.clone(), self.requests);
        let spawned = std::thread::Builder::new().name("straydog-relock".into())
            .spawn(move || lock_banks_behind(&weak, Some((job, requests)), Instant::now(), "behind the requests", "after the plan"));
        if let Err(error) = spawned {
            eprintln!("WARN Straydog cannot lock more banks behind the requests: {error}");
            self.model.banks_lock_ended(None);
        }
    }

    /// The first step of a warm start with `state`, under the engine's lock ([`straydog_engine::model::Model::warm_start`]): the earlier counts it keeps
    /// to save with the next state, the scores it sets, and what it plans; and the draft width the last run on this placement chose, which the model's
    /// passes begin at ([`straydog_engine::model::Model::start_drafts_at`]), the state's other widths kept to save beside it. None for a model that holds
    /// all its experts. A state of another kind, format or model is refused and the model starts as it would without.
    fn warm_begin(&mut self, state: &Value) -> Result<Option<WarmStart>, String> {
        if !self.model.partial {
            return Ok(None);
        }
        let prior = Routes::from_state(state, ROUTED_LAYERS, ROUTED_EXPERTS)?;
        let start = self.model.warm_start(&prior)?;
        self.earlier = Some(prior);
        self.earlier_drafts = routes::drafts_in(state);
        if let Some(width) = routes::saved_drafts(state, &self.placement, self.model.speculative) {
            if self.model.start_drafts_at(width) {
                eprintln!("Straydog drafts: starting at {width}, as the last run on this placement chose");
            }
        }
        Ok(Some(start))
    }
}

/// How a warm start's slot fill ended.
#[derive(Debug, PartialEq, Eq)]
enum Filled {
    /// The slots hold what the scores want.
    Done,
    /// A request began: its own routes and its rebalance take the slots from there.
    CutShort,
    Failed(String),
    /// The engine is gone (the worker ended).
    Gone,
}

/// Step `step` under the engine's lock (it says whether more is wanted), pausing between steps so that a request takes the lock, until it has nothing
/// more to do, the engine's request count moves from `began`, `longest` passes, or the engine is gone. The lock is only held in steps, and a request
/// that has it ends the fill: it is the request's slots by then, so a lock held for `patience` (a step is milliseconds) is a request's, and the fill is over
/// without waiting for the request to end.
fn fill_slots<S>(engine: &Weak<Mutex<S>>, began: u64, requests: impl Fn(&S) -> u64, mut step: impl FnMut(&mut S) -> Result<bool, String>,
                 pause: Duration, patience: Duration, longest: Duration) -> Filled {
    let started = Instant::now();
    loop {
        let Some(alive) = engine.upgrade() else { return Filled::Gone };
        {
            let asked = Instant::now();
            let mut guard = loop {
                match alive.try_lock() {
                    Ok(guard) => break guard,
                    Err(std::sync::TryLockError::Poisoned(_)) => return Filled::Gone,
                    Err(std::sync::TryLockError::WouldBlock) if asked.elapsed() >= patience => return Filled::CutShort,
                    Err(std::sync::TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(1)),
                }
            };
            if requests(&guard) != began {
                return Filled::CutShort;
            }
            match step(&mut guard) {
                Ok(true) if started.elapsed() < longest => {}
                Ok(true) | Ok(false) => return Filled::Done,
                Err(message) => return Filled::Failed(message),
            }
        }
        drop(alive);
        std::thread::sleep(pause);
    }
}

/// A warm start's background work, after `warm_in_background`, on two threads at once, for the slots' copies and the OS's reads share the drive: one asks the
/// OS for the bank pages planned (each request a few hundred MB of the likeliest experts, the engine kept alive while the OS reads its mapping, no lock held),
/// the other fills the expert slots by the scores the first step set, a bounded number of swaps at a time. One line in the log says what it did and how long
/// it took.
fn warm_start(engine: &Weak<Mutex<Engine>>, start: WarmStart, requests: u64, began: Instant) {
    let mut fill = None;
    let (asked, refused, asking, filled) = std::thread::scope(|scope| {
        let asking = scope.spawn(|| {
            let (mut asked, mut refused) = (0u64, None);
            for request in &start.hints.requests {
                let Some(alive) = engine.upgrade() else { break };
                match straydog_engine::host::prefetch(&request.ranges) {
                    Ok(()) => asked += request.bytes,
                    Err(message) => {
                        refused = Some(message);
                        break;
                    }
                }
                drop(alive);
            }
            (asked, refused, began.elapsed())
        });
        let filled = fill_slots(engine, requests, |engine: &Engine| engine.requests, |engine: &mut Engine| {
            let step = engine.model.warm_fill(WARM_SWAPS)?;
            fill = Some(step);
            Ok(step.pending > 0)
        }, WARM_PAUSE, WARM_PATIENCE, WARM_LONGEST);
        let (asked, refused, asking) = asking.join().unwrap_or_else(|_| (0, Some("its thread failed".to_string()), began.elapsed()));
        (asked, refused, asking, filled)
    });
    let (swapped, held, slots) = fill.map_or((0, start.held, start.slots), |fill| (fill.swapped.saturating_sub(start.swapped), fill.held, fill.slots));
    let note = match (&filled, &refused) {
        (Filled::CutShort, _) => " (a request began, and its own routes took the slots from there)".to_string(),
        (Filled::Failed(message), _) => format!(" (the slot fill stopped: {message})"),
        (_, Some(message)) => format!(" (the OS stopped a read-ahead: {message})"),
        _ => String::new(),
    };
    eprintln!("Straydog warm start: {swapped} expert slots filled ({held} of {slots} held), the OS asked for {} MB of bank pages ({} experts) in {:.1} s, \
               the slots done in {:.1} s{note}", asked >> 20, start.hints.experts, asking.as_secs_f64(), began.elapsed().as_secs_f64());
}

/// How long a restore waits for the engine before it takes the engine to be a request's: a warm start's step holds it for a few milliseconds, a request
/// for seconds.
const PREFIX_PATIENCE: Duration = Duration::from_millis(500);

/// The engine, once it is free within [`PREFIX_PATIENCE`], unless a request has had it: what a restore may take, never ahead of a request.
fn engine_for_prefix(engine: &Mutex<Engine>) -> Option<std::sync::MutexGuard<'_, Engine>> {
    let mut asked = Instant::now();
    loop {
        match engine.try_lock() {
            Ok(guard) => return (guard.requests == 0).then_some(guard),
            Err(std::sync::TryLockError::Poisoned(_)) => return None,
            Err(std::sync::TryLockError::WouldBlock) => {
                // (the warm-up behind "ready" holds the engine for seconds, and is not a request: a restore waits for it, for it ends at once
                // if a request comes, and its patience counts from there)
                if behind::WARMING.load(Ordering::SeqCst) {
                    asked = Instant::now();
                } else if asked.elapsed() >= PREFIX_PATIENCE {
                    return None;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// Hand the model a system block's state an earlier run saved, if no request has the engine or has had it: a request that came first is never held up
/// by this (it holds the engine for the moment the plan takes and the moment the install takes, and decodes the few hundred megabytes without it), and
/// a model that has served one holds a sequence this state would only displace. One line in the log says what became of it.
fn restore_prefix(engine: &Weak<Mutex<Engine>>, state: &[u8], read_seconds: f64, arrived: Instant) {
    let not_restored = |why: &str| eprintln!("Straydog prefix: not restored ({why})");
    let Some(alive) = engine.upgrade() else { return };
    let parsed = match straydog_engine::prefix::parse(state) {
        Ok(parsed) => parsed,
        Err(message) => return not_restored(&message),
    };
    // (a moment under the engine's lock: what the state must match)
    let plan = match engine_for_prefix(&alive) {
        Some(engine) => engine.model.prefix_plan(parsed.tokens, parsed.hidden),
        None => return not_restored("a request began first"),
    };
    let image = match plan.and_then(|plan| plan.decode(&parsed)) {
        Ok(image) => image,
        Err(message) => return not_restored(&message),
    };
    // (and another: the sequence it becomes)
    let Some(mut engine) = engine_for_prefix(&alive) else { return not_restored("a request began first") };
    match engine.model.install_prefix(image, state) {
        Ok(tokens) => eprintln!("Straydog prefix: restored {tokens} tokens ({} MB) in {:.1} s", state.len() >> 20, read_seconds + arrived.elapsed().as_secs_f64()),
        Err(message) => not_restored(&message),
    }
}

/// The conversation an earlier run kept ([`crate::conversation`]), as this run knows it: the tokens its state is held at, which the server read from the
/// file without the state, and where the state will arrive if a request has asked for it. A static, for the request loop that reads the frames and the
/// request that waits for one are different threads, and the engine is behind the lock the request holds.
struct KeptConversation {
    tokens: Option<Vec<u32>>,
    /// Where the state arrives, with the seconds the server took to read and open it.
    waiting: Option<std::sync::mpsc::Sender<Result<(Vec<u8>, f64), String>>>,
}

static KEPT: Mutex<KeptConversation> = Mutex::new(KeptConversation { tokens: None, waiting: None });

fn kept() -> std::sync::MutexGuard<'static, KeptConversation> {
    KEPT.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// How long a request waits for the server to read the kept state and send it.
const KEPT_PATIENCE: Duration = Duration::from_secs(120);
/// The host memory a load of the kept state leaves free besides its copies: the commit the rest of the engine asks for as it goes.
const KEPT_MARGIN: u64 = 2 << 30;

/// The server named the tokens of the conversation its last run kept: this run takes the first request that begins with them from there.
fn keep_tokens(bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(format!("{} bytes are no list of tokens", bytes.len()));
    }
    let tokens = straydog_engine::prefix::tokens_of(bytes, bytes.len() / 4);
    if tokens.iter().any(|token| u64::from(*token) >= straydog_engine::gpu::VOCABULARY) {
        return Err("a token outside the vocabulary".into());
    }
    kept().tokens = Some(tokens);
    Ok(())
}

/// The kept conversation's state, if the prompt begins with the tokens it is held at and the model holds no more of it than they: asked of the
/// server, which reads it from the drive, and handed to the model, which takes the request from there. The seconds it took, or None when the prompt
/// does not begin with them, or the state could not be had (one line in the log says why, and the file is left as it is for a later request).
fn take_kept(engine: &mut Engine, output: &Output, id: &str, cancel: &AtomicBool, prompt: &[u32]) -> Option<f64> {
    let wanted = {
        let kept = kept();
        let tokens = kept.tokens.as_ref()?;
        // (the prompt begins with all of them, and a token is left to run)
        (prompt.len() > tokens.len() && prompt[..tokens.len()] == tokens[..]).then_some(tokens.len())?
    };
    let not_loaded = |why: String| eprintln!("Straydog conversation: not loaded ({why})");
    if engine.model.held_prefix(prompt) >= wanted {
        // (the model holds this much of it already: the state would add nothing)
        kept().tokens = None;
        return None;
    }
    // (the file's bytes in the server, the same in this process, and the copies of its parts: three of the state's size at once)
    let size = engine.model.state_size(wanted, straydog_engine::gpu::WIDTH);
    if let Some((_, left)) = straydog_engine::host::commit_left() {
        if left < 3 * size + KEPT_MARGIN {
            not_loaded(format!("the state of {wanted} tokens is {} MB, and the host memory left for it is {} MB", size >> 20, left >> 20));
            return None;
        }
    }
    let started = Instant::now();
    let (sender, receiver) = std::sync::mpsc::channel();
    kept().waiting = Some(sender);
    output.send(&json!({"id": id, "event": "conversation_wanted", "tokens": wanted}));
    let arrived = loop {
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(arrived) => break arrived,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) if cancel.load(Ordering::SeqCst) || started.elapsed() > KEPT_PATIENCE => {
                // (what arrives later is read and dropped by the request loop)
                kept().waiting = None;
                if !cancel.load(Ordering::SeqCst) {
                    not_loaded(format!("the server sent nothing in {} s", KEPT_PATIENCE.as_secs()));
                }
                return None;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return None,
        }
    };
    let arrived_at = started.elapsed().as_secs_f64();
    let (state, read_seconds) = match arrived {
        Ok(arrived) => arrived,
        Err(why) => {
            // (the file is gone or cannot be read: asking again would only ask the same)
            kept().tokens = None;
            not_loaded(why);
            return None;
        }
    };
    let importing = Instant::now();
    match engine.model.import_conversation(&state, prompt.len() + 1) {
        Ok(tokens) => {
            kept().tokens = None;
            let seconds = started.elapsed().as_secs_f64();
            eprintln!("Straydog conversation: loaded {tokens} tokens ({} MB) in {seconds:.1} s (the server read it in {read_seconds:.1} s, it came in {:.1} s, the import took {:.1} s)",
                      state.len() >> 20, (arrived_at - read_seconds).max(0.0), importing.elapsed().as_secs_f64());
            Some(seconds)
        }
        Err(why) => {
            kept().tokens = None;
            not_loaded(why);
            None
        }
    }
}

/// Run two prompts and a few tokens once, so that no request pays for loading the GPU code it
/// runs (1.5 s more for the first 4,096 tokens on two R9700s); the model then starts empty. It runs behind "ready" ([`behind`]), so a request can
/// come in the middle of it: `cancelled` says so, and it stops at the next pass, resets the model as it does at its end, and a request's first
/// pass then loads what it needs, as after any reset. A prompt's pass reads the experts a cold page cache has not got from the drive, which took
/// 10 s of a 256-token prompt, so the model looks at `arrived` between the layers of the pass ([`Model::set_warm_stop`]) and ends it there. Whether it
/// ran to its end.
fn warm_up(model: &mut Model, cancelled: &dyn Fn() -> bool, arrived: &'static AtomicBool) -> Result<bool, String> {
    // (set for these passes alone, and gone before the reset and on every way out: a request's own prefill is read between its batches)
    model.set_warm_stop(Some(arrived));
    let passes = warm_up_passes(model, cancelled);
    model.set_warm_stop(None);
    let complete = match passes {
        Ok(complete) => complete,
        // (a pass stopped half way: what it left is what the reset below takes back)
        Err(error) if straydog_engine::model::is_warm_stop(&error) => false,
        Err(error) => return Err(error),
    };
    model.reset()?;
    // (the routes of these passes say nothing of a later run's: the warm-up state counts from the first request)
    model.clear_route_counts()?;
    Ok(complete)
}

/// The passes of [`warm_up`]: whether they ran to their ends.
fn warm_up_passes(model: &mut Model, cancelled: &dyn Fn() -> bool) -> Result<bool, String> {
    let longest = model.prefill_batch_size.min(model.context.saturating_sub(4));
    let mut complete = true;
    for length in [longest, longest.min(300)] {
        if length >= 2 {
            if cancelled() {
                complete = false;
                break;
            }
            let prompt: Vec<u32> = (0..length as u32).map(|index| 1000 + index % 50_000).collect();
            model.generate(&prompt, 2, &Sampling::GREEDY, 0, cancelled, &mut |_, _| {}, true, false, &mut |_| {})?;
            // (a request seen in the middle of a pass: that pass may not have run to its end)
            if cancelled() {
                complete = false;
                break;
            }
        }
    }
    Ok(complete)
}

/// How long a GPU's memory held by another process is waited for.
const VRAM_PATIENCE: Duration = Duration::from_secs(60);

/// What the worker settles before placing the model: each GPU's VRAM budget, the prefill batch
/// and the split.
pub(crate) struct Plan {
    pub budgets: Vec<u64>,
    pub batch: usize,
    /// The batch was not given (`auto`): the engine's placement may narrow it.
    pub derived: bool,
    pub split: SplitMode,
}

/// The margin `auto` leaves free on a GPU: 512 MiB or 3% of it, whichever is more; on Windows, where the
/// desktop and other programs share the GPU, an eighth of the card and 256 MiB, in whole 256 MiB, up to
/// 2 GiB: 1 GiB of 6 GB, 1.25 of 8, 1.75 of 12, and 2 from a 16 GB card up as before.
///
/// What it has to hold is the engine's own memory outside its budget and what the others come to want after
/// `auto` has looked. On an RTX 5070 Ti (15.9 GiB) the first is 0.74-0.77 GiB at --vram 4.5 to 5.3; of the card, with
/// others holding 2.85 GiB, 0.2-0.8 GiB left free stalled the desktop and 1.8 GiB did not (271d191). That leaves 0.25 GiB
/// for the others at 6 GB, and more would leave no budget that starts Qwen3.8 beside a desktop (`catalog::min_vram`). No small
/// card has been tried: that their engine's own memory is no larger and the others' growth smaller is the reasoning.
pub(crate) fn margin_on(windows: bool, total: u64) -> u64 {
    let least = if windows {
        ((total.div_ceil(8).div_ceil(256 << 20) + 1) * (256 << 20)).clamp(512 << 20, 2 << 30)
    } else {
        512 << 20
    };
    least.max(total / 100 * 3)
}

/// What `auto` takes of a GPU with `free` of `total` bytes free: the free memory less the margin, in whole 256 MiB
/// (on a small card under Windows, whole 64 MiB: there a rounding is a tenth of the expert slots).
pub(crate) fn auto_budget(windows: bool, free: u64, total: u64) -> u64 {
    let step: u64 = if windows && total < 12 << 30 { 64 << 20 } else { 256 << 20 };
    free.saturating_sub(margin_on(windows, total)) / step * step
}

#[cfg(any(windows, test))]
const VRAM_GROWTH: u64 = 512 << 20;

/// A budget reduced under pressure is retried gradually, after two further requests.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VramLimit {
    bytes: u64,
    retry_after: u64,
}

#[cfg(any(windows, test))]
impl VramLimit {
    fn new(bytes: u64, requests: u64) -> Self {
        Self { bytes, retry_after: requests.saturating_add(2) }
    }
}

/// DXGI's headroom already excludes other allocations and may be negative. Leave a runtime margin
/// within that allowance; the larger desktop margin is still used when only runtime free bytes exist.
#[cfg(any(windows, test))]
fn headroom_budget(used: u64, headroom: i64, process_budget: bool, total: u64, cap: Option<u64>) -> u64 {
    let wanted = if !process_budget {
        auto_budget(true, used.saturating_add(headroom.max(0) as u64), total)
    } else {
        let available = (i128::from(used) + i128::from(headroom)).clamp(0, i128::from(total)) as u64;
        let step = if total < 12 << 30 { 64 << 20 } else { 256 << 20 };
        available.saturating_sub(margin_on(false, total)) / step * step
    };
    wanted.min(cap.unwrap_or(u64::MAX))
}

#[cfg(any(windows, test))]
fn vram_target(budget: u64, wanted: u64, requests: u64, allow_grow: bool, limit: Option<VramLimit>, cap: Option<u64>) -> u64 {
    let ceiling = cap.unwrap_or(u64::MAX);
    let (budget, wanted) = (budget.min(ceiling), wanted.min(ceiling));
    if wanted < budget { return wanted; }
    // Only reaching the original ceiling may use a smaller final growth step.
    let reaches_cap = cap.is_some_and(|cap| wanted == cap && budget < cap);
    if !allow_grow || (wanted < budget.saturating_add(VRAM_GROWTH) && !reaches_cap) { return budget; }
    if let Some(limit) = limit {
        if requests <= limit.retry_after { return budget; }
        return wanted.min(budget.saturating_add(VRAM_GROWTH)).min(limit.bytes.saturating_add(VRAM_GROWTH)).max(budget).min(ceiling);
    }
    wanted.min(budget.saturating_add(VRAM_GROWTH)).min(ceiling)
}

#[cfg(any(windows, test))]
fn validate_vram_caps(caps: Option<&[u64]>, devices: usize) -> Result<(), String> {
    if let Some(caps) = caps {
        if caps.len() != devices {
            return Err(format!("{} explicit GPU memory ceilings do not match {devices} devices", caps.len()));
        }
    }
    Ok(())
}

#[cfg(windows)]
fn vram_refused(ordinal: usize, why: &str) -> String {
    format!("GPU {ordinal}: {why}; {}", straydog_engine::model::VRAM_NO_MEMORY)
}

/// Runs between passes with the model exclusively held. Every reduction is followed by a fresh
/// measurement: releasing the pool may change Windows' process budget as well as its usage.
#[cfg(windows)]
fn adjust_windows_vram(model: &mut Model, limits: &mut Vec<Option<VramLimit>>, caps: Option<&[u64]>, requests: u64,
                       allow_grow: bool, phase: &str) -> Result<(), String> {
    let devices = model.vram_budgets()?;
    validate_vram_caps(caps, devices.len())?;
    limits.resize(devices.len(), None);
    let explicit_startup = caps.is_some() && phase == "ready";
    for (index, (ordinal, _, specified)) in devices.into_iter().enumerate() {
        let cap = caps.map(|caps| caps[index]);
        let mut startup_change = None;
        loop {
            let (_, used, budget) = model.vram_budgets()?[index];
            let (headroom, process_budget) = model.vram_headroom(index)?;
            let (_, _, total) = straydog_gpu::device_info(ordinal).map_err(|error| error.to_string())?;
            let wanted = headroom_budget(used, headroom, process_budget, total, cap);
            let target = vram_target(budget, wanted, requests, allow_grow, limits[index], cap);
            if std::env::var("STRAYDOG_VRAM_DIAGNOSTIC").as_deref() == Ok("1") {
                eprintln!("Straydog VRAM diagnostic: phase={phase} GPU={ordinal} used={used} budget={budget} headroom={headroom} \
                           process_budget={} wanted={wanted} target={target}", u8::from(process_budget));
            }
            if target == budget { break; }
            let shrinking = target < budget;
            let (before, after) = model.resize_vram(index, target)?;
            let actual = model.vram_budgets()?[index].2;
            if shrinking && actual >= budget {
                return Err(vram_refused(ordinal, &format!("observed headroom is {headroom} bytes and the memory budget cannot be reduced further")));
            }
            if actual != budget {
                if shrinking || limits[index].is_some() { limits[index] = Some(VramLimit::new(actual, requests)); }
                if explicit_startup {
                    let first = startup_change.map_or(before, |(first, _, _, _)| first);
                    startup_change = Some((first, after, actual, headroom));
                } else {
                    eprintln!("Straydog VRAM: GPU {ordinal}'s budget {:.2} -> {:.2} GiB; observed headroom {headroom} bytes; expert slots {before} -> {after}",
                              budget as f64 / (1u64 << 30) as f64, actual as f64 / (1u64 << 30) as f64);
                }
            }
            if !shrinking { break; }
        }
        if let Some((before, after, actual, headroom)) = startup_change {
            eprintln!("Straydog VRAM: GPU {ordinal}'s given {:.2} GiB is above the usable Windows budget after leaving its safety margin; using {:.2} GiB at startup; observed headroom {headroom} bytes; expert slots {before} -> {after}",
                      cap.unwrap_or(specified) as f64 / (1u64 << 30) as f64, actual as f64 / (1u64 << 30) as f64);
        }
    }
    Ok(())
}

/// How long `auto` waits for a GPU's free memory to stop growing (a previous worker releasing it).
const SETTLE: Duration = Duration::from_secs(5);
/// The same with more than a quarter of the GPU held. On Windows the desktop's programs hold that as often as not and keep it, so a start
/// waited [`VRAM_PATIENCE`] for nothing each time: `--vram auto` on an RTX 5070 Ti with 72% free came up in 78 s, of which 18 were the model's.
#[cfg(windows)]
const SETTLE_HELD: Duration = Duration::from_secs(10);
#[cfg(not(windows))]
const SETTLE_HELD: Duration = VRAM_PATIENCE;

/// Each GPU's budget. Given ones wait until the GPU has them free: memory a previous worker still
/// holds is released after it exits, while memory another process keeps fails the start after
/// [`VRAM_PATIENCE`] (a GPU smaller than its budget is left to fail as it does). `auto` waits
/// until nine tenths of each GPU are free, or until its free memory has stopped growing for
/// [`SETTLE`] with at most a quarter of it held ([`SETTLE_HELD`] with more), and at most [`VRAM_PATIENCE`] in all; then it
/// takes what is free less a margin ([`auto_budget`]). A display's or another program's share is
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
                    let settled = grown.1.elapsed() >= if free >= total / 4 * 3 { SETTLE } else { SETTLE_HELD };
                    if free >= total / 10 * 9 || settled || started.elapsed() >= VRAM_PATIENCE {
                        chosen.push(auto_budget(cfg!(windows), free, total));
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

impl WorkerArgs {
    /// The pinned bundle its labels name.
    pub fn bundle(&self) -> Result<straydog_engine::bundle::Id, String> {
        straydog_engine::bundle::Id::find(Some(&self.model_name), Some(&self.provider), &self.quant)
    }
}

/// The budgets, batch and split for `args`. Budgets more than a tenth apart do not split every
/// layer across the GPUs (each would take the smaller share): the layers are split instead, in
/// proportion to the budgets.
pub(crate) fn plan(args: &WorkerArgs, report: &dyn Fn(&str, u64, u64)) -> Result<Plan, String> {
    let budgets = budgets(&args.devices, &args.vram_bytes, report)?;
    let on_demand = args.bundle()?.is_on_demand();
    let batch = match args.prefill_batch_size.as_str() {
        // (as the configuration derives it from given budgets: 4096 for the resident models from 48 GiB, narrowed by the placement, see config::resolve)
        "auto" if on_demand => if budgets.iter().sum::<u64>() >= 48 << 30 { 4096 } else { 1024 },
        "auto" => straydog_engine::model::resident_batch(budgets.iter().sum()),
        given => given.parse().map_err(|_| format!("invalid Straydog prefill batch {given:?}"))?,
    };
    let (least, most) = (*budgets.iter().min().unwrap(), *budgets.iter().max().unwrap());
    let split = if args.split_mode == "layer" || most - least > most / 10 { SplitMode::Layer } else { SplitMode::Experts };
    Ok(Plan { budgets, batch, derived: args.prefill_batch_size == "auto", split })
}

#[cfg(all(windows, feature = "cuda"))]
fn memory_diagnostic(devices: &[usize], ready: bool) {
    if std::env::var("STRAYDOG_VRAM_DIAGNOSTIC").as_deref() != Ok("1") {
        return;
    }
    for &ordinal in devices {
        if let Err(error) = straydog_gpu::memory_diagnostic(ordinal, ready) {
            let phase = if ready { "ready" } else { "startup" };
            eprintln!("WARN Straydog VRAM diagnostic: GPU {ordinal} {phase}: {error}");
        }
    }
}

/// A VRAM budget as the key of a placement names it: its size in GiB, to hundredths, without the zeros the hundredths leave (`12G`, `11.5G`).
fn budget_label(bytes: u64) -> String {
    let gib = format!("{:.2}", bytes as f64 / (1u64 << 30) as f64);
    format!("{}G", gib.trim_end_matches('0').trim_end_matches('.'))
}

/// The key of this placement in the warm-up state's draft widths ([`routes::saved_drafts`]), so that another placement of the same model does not take this
/// one's width: each GPU used (its ordinal and name) with its VRAM budget as it was given (`vram`, as the worker takes it: `auto`, or each GPU's bytes; not
/// what `auto` took, which moves with what other programs hold), the context size, and whether a drafter is loaded: `0:NVIDIA GeForce RTX 5090@12G; ctx
/// 32768; drafter` for one GPU given 12 GiB, `0:NVIDIA GeForce RTX 5090@auto,1:NVIDIA GeForce RTX 3090@auto; ctx 32768; drafter` for two left to `auto`.
fn placement_key(gpus: &[(usize, String)], vram: &str, context: usize, drafter: bool) -> String {
    let budgets: Vec<String> = if vram == "auto" {
        vec!["auto".into(); gpus.len()]
    } else {
        vram.split(',').map(|part| part.parse::<u64>().map_or_else(|_| part.to_string(), budget_label)).collect()
    };
    let gpus: Vec<String> = gpus.iter().enumerate()
        .map(|(index, (ordinal, name))| format!("{ordinal}:{name}@{}", budgets.get(index).map_or("?", String::as_str)))
        .collect();
    format!("{}; ctx {context}; {}", gpus.join(","), if drafter { "drafter" } else { "no drafter" })
}

pub fn main(args: WorkerArgs) -> Result<(), String> {
    straydog_engine::host::set_prefetch_memory_limits(args.host_memory_limit_files.clone());
    // (Windows: a worker has no window, so the system may count it as background and lower its clocks: it asks not to be)
    if !straydog_engine::host::full_speed() {
        eprintln!("WARN Straydog could not turn off the system's power throttling of its worker");
    }
    let output = Output(Arc::new(Mutex::new(std::io::stdout())));
    let loading = Loading { output: output.clone(), last: Mutex::new((String::new(), Instant::now())) };
    #[cfg(all(windows, feature = "cuda"))]
    memory_diagnostic(&args.devices, false);
    let plan = plan(&args, &|stage, done, total| loading.report(stage, done, total))?;
    #[cfg(feature = "cuda")]
    {
        let devices = args.devices.iter().map(|&ordinal| {
            let name = straydog_gpu::device_info(ordinal).map(|(name, _, _)| name)
                .unwrap_or_else(|_| "unknown GPU".into());
            let supported = match straydog_gpu::host_register_read_only_supported(ordinal) {
                Ok(true) => "supported".into(),
                Ok(false) => "not supported".into(),
                Err(error) => format!("unknown ({error})"),
            };
            format!("GPU {ordinal} ({name}): {supported}")
        }).collect::<Vec<String>>();
        eprintln!("Straydog GPU read-only host registration: {}", devices.join(", "));
    }
    let id = args.bundle()?;
    if id.is_on_demand() {
        return crate::worker_on_demand::main(&args, &plan, output, &|stage, done, total| loading.report(stage, done, total));
    }
    // The first Qwen prefill GEMM otherwise loads the BLAS code while the ready-time warm-up holds
    // the engine. Prime each device beside the model's reading and placement, when FP16 GEMM is used.
    #[cfg(any(feature = "cuda", feature = "hip"))]
    let gemm_primes = if args.gemm == "fp16" && plan.batch as u64 >= args.gemm_min_batch {
        args.devices.iter().copied().filter_map(|ordinal| {
            let tokens = plan.batch;
            match std::thread::Builder::new().name(format!("straydog-gemm-prime-{ordinal}")).spawn(move || {
                let started = Instant::now();
                let result = straydog_gpu::prime_qwen_prefill_gemm(ordinal, tokens);
                (started.elapsed(), result)
            }) {
                Ok(handle) => Some((ordinal, handle)),
                Err(error) => {
                    eprintln!("WARN Straydog warm code: GPU {ordinal} prefill GEMM priming could not start: {error}");
                    None
                }
            }
        }).collect::<Vec<_>>()
    } else { Vec::new() };
    let split = plan.split;
    let drafter = args.drafter.as_deref().filter(|_| args.speculative > 0);
    let report = |stage: &str, done, total| loading.report(stage, done, total);
    let opened = Instant::now();
    // The model files open while the GPUs initialize, and the vision encoder's is hashed (it is read through the page cache:
    // none of it stays in this process until an image comes).
    let (group, weights, vision) = std::thread::scope(|scope| {
        let weights = scope.spawn(|| Weights::open(&args.model, id, drafter, args.host_memory_bytes, &report));
        let vision = scope.spawn(|| match &args.mmproj {
            Some(path) => bundle_vision().and_then(|pin| Encoder::open(path, &pin)),
            None => Err(NO_VISION_FILE.to_string()),
        });
        let group = Group::new(&args.devices, &plan.budgets, split)
            .and_then(|mut group| group.configure_gemm(args.gemm_min_batch, args.gemm == "fp16").map(|()| group));
        (group, weights.join().unwrap(), vision.join().unwrap())
    });
    let (group, mut weights) = (group?, weights?);
    // (the model serves text all the same)
    if let (Err(reason), Some(_)) = (&vision, &args.mmproj) {
        eprintln!("WARN Straydog reads no images: {reason}");
    }
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
        // (a derived batch is the engine's to derive and narrow; `plan.batch` is what it will derive, for warming the GEMM code)
        prefill_batch_size: if plan.derived { straydog_engine::model::DERIVED_BATCH } else { plan.batch },
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
    #[cfg(any(feature = "cuda", feature = "hip"))]
    for (ordinal, gemm_prime) in gemm_primes {
        match gemm_prime.join() {
            Ok((elapsed, Ok(()))) => eprintln!("Straydog warm code: GPU {ordinal} prefill GEMM primed in {:.1} s", elapsed.as_secs_f64()),
            Ok((_, Err(error))) => eprintln!("WARN Straydog warm code: GPU {ordinal} prefill GEMM priming failed: {error}"),
            Err(_) => eprintln!("WARN Straydog warm code: GPU {ordinal} prefill GEMM priming panicked"),
        }
    }
    let model = model.map_err(|error| auto_too_small(error, &args.vram_bytes, &plan.budgets));
    let (text, model) = (Arc::new(text?), model?);
    #[cfg(windows)]
    let (model, vram_limits, vram_caps, ready_budgets) = {
        let mut model = model;
        let caps = (args.vram_bytes != "auto").then(|| plan.budgets.clone());
        let devices = model.vram_budgets()?.len();
        validate_vram_caps(caps.as_deref(), devices)?;
        let mut limits = vec![None; devices];
        if model.partial {
            adjust_windows_vram(&mut model, &mut limits, caps.as_deref(), 0, false, "ready")?;
        }
        let budgets = if model.partial {
            model.vram_budgets()?.into_iter().map(|(_, _, budget)| budget).collect::<Vec<_>>()
        } else { plan.budgets.clone() };
        (model, limits, caps, budgets)
    };
    #[cfg(not(windows))]
    let ready_budgets = &plan.budgets;
    // The GPU code's warm-up and the page-locking of the banks come behind "ready" ([`behind`]): "plans" and "warm" are what they cost before it.
    let warming = Instant::now();
    behind::WARMING.store(true, Ordering::SeqCst);
    let mut seconds = serde_json::Map::new();
    seconds.insert("open".into(), open_seconds.into());
    for (step, value) in &model.load_seconds {
        seconds.insert(step.to_string(), (*value).into());
    }
    seconds.insert("warm".into(), warming.elapsed().as_secs_f64().into());
    let ready = json!({
        "event": "ready", "protocol": 1, "engine": "Straydog", "model": id.model, "provider": id.provider,
        "quantization": id.quant, "context": model.context, "state_bytes": model.state_bytes,
        "speculative_drafts": model.speculative,
        "speculation": if model.speculative > 0 { json!([{"kind": "mtp", "drafts": model.speculative}]) } else { json!([]) },
        // (what a system block's state a start may be handed must match: its layout, to the byte)
        "prefix": {"version": straydog_engine::prefix::VERSION, "layout": model.prefix_layout()},
        "vision": vision.is_ok(),
        // (the engram table read: its precision and file, Unsloth's Qwen3.8)
        "engram": straydog_engine::bundle::table_in(&args.model, id).ok().flatten()
            .map(|(table, path)| json!({"precision": table.precision, "file": path.file_name().map(|name| name.to_string_lossy().into_owned())})),
        "vision_refusal": vision.as_ref().err(),
        "placement": {"resident": model.resident, "partial": model.partial, "host_kv": model.host_kv,
                      "prefill_batch_size": model.prefill_batch_size, "vram_budgets": ready_budgets,
                      // (a long read's wider batches: their rows, from how many new tokens, and what they lacked when they were not made)
                      "prefill_wide_rows": model.wide_batch_size, "prefill_wide_from": straydog_engine::model::WIDE_READ_TOKENS,
                      "prefill_wide_note": model.wide_note,
                      "split_mode": if split == SplitMode::Layer { "layer" } else { "experts" },
                      "note": model.placement_note},
        "load_seconds": seconds,
        "gpu": model.info()?,
    });
    let gpus: Vec<(usize, String)> = args.devices.iter()
        .map(|&ordinal| (ordinal, straydog_gpu::device_info(ordinal).map_or_else(|_| "unknown GPU".into(), |(name, _, _)| name)))
        .collect();
    let placement = placement_key(&gpus, &args.vram_bytes, model.context, model.weights.drafter);
    let engine = Arc::new_cyclic(|me| Mutex::new(Engine { model, template, vision, encoded: Vec::new(), earlier: None, earlier_drafts: BTreeMap::new(), placement,
                                                       requests: 0, me: me.clone(),
                                                       #[cfg(not(windows))] vram_auto: args.vram_bytes == "auto",
                                                       #[cfg(windows)] vram_limits,
                                                       #[cfg(windows)] vram_caps }));
    #[cfg(all(windows, feature = "cuda"))]
    memory_diagnostic(&args.devices, true);
    output.send(&ready);
    let ready_at = Instant::now();
    behind_ready(&engine, ready_at);
    serve(engine, text, output)
}

/// The warm-up and then the locking of the banks, on a thread of their own, from "ready" on ([`behind`]). One INFO line each says how it went and
/// when it ended, in seconds after ready.
fn behind_ready(engine: &Arc<Mutex<Engine>>, ready_at: Instant) {
    let weak = Arc::downgrade(engine);
    let spawned = std::thread::Builder::new().name("straydog-behind".into()).spawn(move || {
        let began = Instant::now();
        // What is made of the engine while it is in hand: a warm start asked for meanwhile begins, so that the layers it counts the host's routes of
        // are locked first, and the locking is planned. Both are done under the lock the warm-up holds, so that no request takes the engine first
        // and the locking, which a request would otherwise keep waiting for the plan, begins as soon as the warm-up ends.
        let plan = |engine: &mut Engine, state: Option<Value>| {
            let started = state.and_then(|state| match engine.warm_begin(&state) {
                Ok(start) => start.map(|start| (start, engine.requests)),
                Err(message) => {
                    eprintln!("Straydog cannot warm up: {message}");
                    None
                }
            });
            (started, engine.model.bank_locks(engine.earlier.as_ref()).map(|job| (job, engine.requests)))
        };
        let mut planned = (None, None);
        let outcome = behind::warm_up_behind(&weak, &behind::ARRIVED, |engine: &mut Engine, cancelled| {
            let result = warm_up(&mut engine.model, cancelled, &behind::ARRIVED);
            planned = plan(engine, behind::warm_up_over());
            result
        });
        // (the engine is let go: how long the request that cut the warm-up short waited for it, from the moment it came)
        let waited = behind::waited_ms().unwrap_or(0);
        if outcome == behind::WarmedUp::Skipped {
            // (a request had the engine first, and is waited for: the plan is made when it lets go)
            let state = behind::warm_up_over();
            if let Some(alive) = weak.upgrade() {
                if let Ok(mut engine) = alive.lock() {
                    planned = plan(&mut engine, state);
                }
            }
        }
        let (seconds, after) = (began.elapsed().as_secs_f64(), ready_at.elapsed().as_secs_f64());
        match &outcome {
            behind::WarmedUp::Done => eprintln!("Straydog warm-up: done in {seconds:.1} s, {after:.1} s after ready"),
            behind::WarmedUp::CutShort => eprintln!("Straydog warm-up: cut short by a request after {seconds:.1} s, {after:.1} s after ready, {waited} ms after the request came"),
            behind::WarmedUp::Skipped => eprintln!("Straydog warm-up: not run, a request had the engine first ({after:.1} s after ready)"),
            behind::WarmedUp::Failed(message) => eprintln!("WARN Straydog warm-up failed after {seconds:.1} s: {message}"),
            behind::WarmedUp::Gone => return,
        }
        let (started, job) = planned;
        if let Some((start, requests)) = started {
            if let Err(message) = spawn_warm_start(&weak, start, requests, Instant::now()) {
                eprintln!("Straydog cannot warm up: {message}");
            }
        }
        lock_banks_behind(&weak, job, ready_at, "behind ready", "after ready");
    });
    if let Err(error) = spawned {
        behind::warm_up_over();
        eprintln!("WARN Straydog cannot warm up or lock the banks behind ready: {error}");
    }
}

/// Lock the banks of `job` layer by layer behind the requests ([`behind::lock_behind`]), with their pages read ahead of the locking: the start's plan
/// behind "ready", or the layers that memory freed since allows ([`Engine::lock_more`]). The log's line says `what` it was, and the seconds
/// since `ready_at` (`since`).
fn lock_banks_behind(weak: &Weak<Mutex<Engine>>, job: Option<(straydog_engine::banks::BankLocks, u64)>, ready_at: Instant, what: &str, since: &str) {
    // The pages of the layers to be locked are read meanwhile, ahead of the locking, so that a request that comes before the locking is done finds them
    // in memory ([`behind::read_ahead_behind`]).
    let progress = Arc::new(behind::Progress::default());
    if let Some((job, _)) = &job {
        spawn_read_ahead(weak, job, &progress, ready_at);
    }
    behind::lock_behind(weak, |engine: &Engine| engine.requests, behind::YIELDING, job,
        |job| job.remaining() > 0,
        // (the memory available now, less the layer: the plan's margin and the K/V the first 64K tokens have yet to grow stay free)
        |job| job.room_for_next(straydog_engine::host::available_bytes()).map(|short| short.message()),
        |job| {
            // SAFETY: `lock_behind` calls this with a strong reference to the engine held, which keeps the weights the plan was made from mapped.
            // (the turn works at a lower priority, on fewer cores: a request that generates meanwhile needs them)
            let next = unsafe { job.lock_next_behind() };
            // (the read-ahead follows the locking: a turn done, or the locking over)
            match &next {
                Some(Ok(_)) => {
                    progress.turns.fetch_add(1, Ordering::SeqCst);
                }
                _ => progress.over.store(true, Ordering::SeqCst),
            }
            let next = next?;
            Some(match next {
                Ok(locked) => {
                    let (layer, bytes) = (locked.layer, locked.bytes);
                    Ok((locked, layer, bytes))
                }
                Err(stop) => Err(behind::Stopped {
                    unverified: matches!(stop, straydog_engine::banks::Stop::Unverified { .. }), message: stop.message(),
                }),
            })
        },
        |engine: &mut Engine, locked| engine.model.banks_locked(locked).map(drop),
        |locked| {
            // SAFETY: as above, the engine is held while a layer that was locked is given up.
            unsafe { locked.unlock() }
        },
        |engine: &mut Engine, locking| engine.model.banks_lock_ended(locking.unverified.then(|| locking.stopped.clone()).flatten()),
        |locking| {
            let (after, gib) = (ready_at.elapsed().as_secs_f64(), locking.bytes as f64 / (1u64 << 30) as f64);
            let ran = if locking.requests == 0 { "no request ran meanwhile".to_string() } else { format!("{} request(s) ran meanwhile", locking.requests) };
            let stopped = locking.stopped.as_ref().map(|why| format!(", stopped because {why}")).unwrap_or_default();
            // (the turns wait for a request that has the engine: how long in all, and whether a wait reached the guard's time)
            let guard = if locking.guard_fired { "fired" } else { "did not fire" };
            let waited = format!("waited {:.1} s for requests, the {} s guard {guard}", locking.waited.as_secs_f64(), behind::YIELDING.at_most.as_secs());
            if locking.layers > 0 || locking.stopped.is_some() {
                eprintln!("Straydog host banks locked {what}: {} layers, {gib:.1} GiB, in {:.1} s, {after:.1} s {since}; {ran}; {waited}{stopped}",
                          locking.layers, locking.seconds);
            }
        });
    // (whatever ended it, the pages ahead of it are not wanted)
    progress.over.store(true, Ordering::SeqCst);
}

/// Read the banks the locking is planned for, ahead of it, on a thread of its own ([`behind::read_ahead_behind`]): the pieces of the plan in the order they will be
/// locked, asked of the system a few hundred MB at a time at a priority below a request's own faults and reads (Windows' background mode for the thread, the idle I/O
/// class on Linux). The locking and a request that comes first then find the pages in memory, where a cold page cache left them to be read by faulting, one
/// fault after another. One line in the log says how it went when it ends.
fn spawn_read_ahead(engine: &Weak<Mutex<Engine>>, job: &straydog_engine::banks::BankLocks, progress: &Arc<behind::Progress>, ready_at: Instant) {
    let (pieces, margin) = (job.read_ahead(), job.margin());
    if pieces.is_empty() {
        return;
    }
    let (engine, progress) = (engine.clone(), progress.clone());
    let spawned = std::thread::Builder::new().name("straydog-ahead".into()).spawn(move || {
        // (Windows: its memory priority stays normal, or the pages it reads in are the first the system takes back, the next it reads among them)
        let _lowered = straydog_engine::host::Lowered::begin_reading(straydog_engine::host::Priority::Background);
        // SAFETY (all of them): `read_ahead_behind` calls them with a strong reference to the engine held, which keeps the weights the pieces lie in mapped.
        // (Linux: `madvise` returns once the reads are queued, and a piece is in when its pages are in the page cache; Windows: the call returns when they are in)
        let done = behind::read_ahead_behind(&engine, &pieces, &progress, margin, behind::READING, |piece| unsafe { piece.resident() },
                                             |piece| cfg!(target_os = "linux") && unsafe { piece.resident() } < 0.98,
                                             straydog_engine::host::available_bytes, |piece| unsafe { piece.ask() });
        eprintln!("{}", behind::ahead_line(&done, ready_at.elapsed().as_secs_f64()));
    });
    if let Err(error) = spawned {
        eprintln!("WARN Straydog cannot read the banks ahead of the locking: {error}");
    }
}

/// A placement's refusal under `auto`, which no one chose the budget of: what it took and what to do.
fn auto_too_small(error: String, given: &str, budgets: &[u64]) -> String {
    match error.strip_suffix("; raise --vram") {
        Some(reason) if given == "auto" => {
            let taken: Vec<String> = budgets.iter().map(|&bytes| format!("{:.2}", bytes as f64 / (1u64 << 30) as f64)).collect();
            format!("{reason}: `auto` took {} GiB, what the GPU had free less a margin for the desktop; close what holds its memory, or give --vram", taken.join(", "))
        }
        _ => error,
    }
}

fn bundle_vision() -> Result<straydog_engine::bundle::Asset, String> {
    straydog_engine::bundle::qwen_vision_source().map(|source| source.asset)
}

impl Engine {
    /// The rows of the placed images (by index; none for an image the prompt does not hold): those encoded lately kept, the
    /// others encoded. None when the request was cancelled meanwhile.
    fn image_rows(&mut self, images: &[images::QwenImage], spans: &[(usize, usize)], cancel: &AtomicBool)
                  -> Result<Option<Vec<Option<Arc<Vec<f32>>>>>, Failure> {
        let encoder = self.vision.as_ref().map_err(|reason| refused(format!("this model reads no images: {reason}")))?;
        let mut rows: Vec<Option<Arc<Vec<f32>>>> = vec![None; images.len()];
        for &(index, _) in spans {
            if rows[index].is_some() {
                continue;
            }
            let digest = images[index].digest;
            if let Some(at) = self.encoded.iter().position(|(kept, _)| *kept == digest) {
                let kept = self.encoded.remove(at);
                rows[index] = Some(kept.1.clone());
                self.encoded.push(kept);
                continue;
            }
            // (a refusal: nothing of the model's state has moved, and the worker reads on)
            // (whatever an image does in there, the worker reads on)
            let encoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encoder.encode(&images[index].prepared, &|| cancel.load(Ordering::SeqCst))))
                .unwrap_or_else(|_| Err("it failed on this image".to_string()))
                .map_err(|message| refused(format!("the vision encoder: {message}")))?;
            let Some(encoded) = encoded else { return Ok(None) };
            let encoded = Arc::new(encoded);
            rows[index] = Some(encoded.clone());
            self.encoded.push((digest, encoded));
            while self.encoded.iter().map(|(_, values)| values.len() * 4).sum::<usize>() > ENCODED_BYTES && self.encoded.len() > 1 {
                self.encoded.remove(0);
            }
        }
        Ok(Some(rows))
    }
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
                behind::arrive();
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
                if let Err(message) = S::warm_in_background(&engine, &state) {
                    eprintln!("Straydog cannot warm up: {message}");
                }
            }
            Some("prefix_state") => {
                behind::arrive();
                if active.lock().unwrap().is_some() {
                    output.send(&json!({"id": id, "event": "error", "message": "Straydog is busy", "fatal": false}));
                    continue;
                }
                let state = engine.lock().unwrap().prefix_state();
                match state {
                    Ok(Some((_, bytes))) if bytes.len() as u64 > crate::frames::BODY_LIMIT => {
                        output.send(&json!({"id": id, "event": "prefix_state", "bytes": 0, "reason": "too large"}));
                    }
                    Ok(Some((mut header, bytes))) => {
                        header["id"] = id.clone().into();
                        header["event"] = "prefix_state".into();
                        output.send_with_body(&header, &bytes);
                    }
                    Ok(None) => output.send(&json!({"id": id, "event": "prefix_state", "bytes": 0})),
                    Err(message) => output.send(&json!({"id": id, "event": "error", "message": message, "fatal": false})),
                }
            }
            Some("conversation_state") => {
                behind::arrive();
                if active.lock().unwrap().is_some() {
                    output.send(&json!({"id": id, "event": "error", "message": "Straydog is busy", "fatal": false}));
                    continue;
                }
                let limit = match request.get("limit") {
                    Some(pyjson::Value::Int(limit)) if *limit >= 0 => (*limit as u64).min(crate::frames::BODY_LIMIT),
                    _ => crate::frames::BODY_LIMIT,
                };
                let state = engine.lock().unwrap().conversation_state(limit);
                match state {
                    Ok(Some((mut header, bytes))) => {
                        header["id"] = id.clone().into();
                        header["event"] = "conversation_state".into();
                        output.send_with_body(&header, &bytes);
                    }
                    Ok(None) => output.send(&json!({"id": id, "event": "conversation_state", "bytes": 0, "reason": "the model holds no conversation to keep"})),
                    Err(message) => output.send(&json!({"id": id, "event": "conversation_state", "bytes": 0, "reason": message})),
                }
            }
            Some("conversation_tokens") => {
                let count = match request.get("bytes") {
                    Some(pyjson::Value::Int(count)) if *count >= 0 => *count as u64,
                    _ => break Err("Straydog conversation frame has no byte count".into()),
                };
                let body = match crate::frames::read_body(&mut reader, count) {
                    Ok(body) => body,
                    Err(message) => break Err(message),
                };
                if let Err(message) = keep_tokens(&body) {
                    eprintln!("Straydog conversation: not kept ({message})");
                }
            }
            Some("conversation") => {
                let count = match request.get("bytes") {
                    Some(pyjson::Value::Int(count)) if *count >= 0 => *count as u64,
                    _ => break Err("Straydog conversation frame has no byte count".into()),
                };
                // (the body is read whatever it is for: the pipe stays in step)
                let body = match crate::frames::read_body(&mut reader, count) {
                    Ok(body) => body,
                    Err(message) => break Err(message),
                };
                let reason = request.get("reason").and_then(pyjson::Value::as_str).unwrap_or("the server sent no state").to_string();
                let read_seconds = match request.get("read_seconds") {
                    Some(pyjson::Value::Float(seconds)) => *seconds,
                    _ => 0.0,
                };
                if let Some(waiting) = kept().waiting.take() {
                    let _ = waiting.send(if count > 0 { Ok((body, read_seconds)) } else { Err(reason) });
                }
            }
            Some("prefix") => {
                let arrived = Instant::now();
                let count = match request.get("bytes") {
                    Some(pyjson::Value::Int(count)) if *count >= 0 => *count as u64,
                    _ => break Err("Straydog prefix frame has no byte count".into()),
                };
                // (a body that is short or over its limit leaves the pipe out of step: the worker ends)
                let state = match crate::frames::read_body(&mut reader, count) {
                    Ok(state) => state,
                    Err(message) => break Err(message),
                };
                let read_seconds = match request.get("read_seconds") {
                    Some(pyjson::Value::Float(seconds)) => *seconds,
                    _ => 0.0,
                };
                S::prefix_in_background(&engine, state, read_seconds, arrived);
            }
            Some("chat") if matches!(request.get("body"), Some(pyjson::Value::Object(_))) => {
                // (the warm-up behind "ready" stops at its next pass and lets the request have the engine)
                behind::arrive();
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
                    // (after what the runtime asks at once of a request that ended, as its prefix state, which takes the engine first;
                    // a request that comes meanwhile waits for it: it costs nothing unless memory was freed)
                    if terminal["event"] == "done" {
                        if let Ok(mut engine) = engine.lock() {
                            engine.after_request();
                        }
                    }
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
    // K/V that found no memory to grow into: only sequences that long are refused. Nor does a copy of the conversation's state that found none.
    // Nor does a prompt found too long before anything ran.
    let fatal = !message.ends_with(straydog_engine::arch::deepseek41::engine::VOID_PASS)
        && !message.ends_with(straydog_engine::model::KV_NO_MEMORY)
        && !message.ends_with(straydog_engine::model::VRAM_NO_MEMORY)
        && !message.ends_with(straydog_engine::gpu::HOST_NO_MEMORY)
        && !message.ends_with(straydog_engine::gpu::HYBRID_WAIT_RECOVERED)
        && !message.ends_with(straydog_engine::rows::ROW_READ_REFUSED)
        && !message.ends_with(straydog_engine::model::IMAGE_REFUSED)
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
            let mut message = failure.message;
            // A failed device (an illegal memory access) cannot be reset; the request still ends.
            if let Err(exc) = engine.reset() {
                eprintln!("Straydog could not reset after a failed generation: {exc}; original failure: {message}");
                fatal = true;
                message = format!("The engine could not reset after this request failed: {exc}. Restart it before retrying.");
            }
            json!({"id": id, "event": "error", "message": message.chars().take(500).collect::<String>(),
                   "fatal": fatal})
        }
    }
}

fn run(engine: &mut Engine, text: &Text, output: &Output, id: &str, body: &pyjson::Value, cancel: &AtomicBool,
       started: Instant) -> Result<Value, Failure> {
    use pyjson::Value as Py;
    let messages = body.get("messages").ok_or_else(|| refused("'messages'"))?;
    // (each image part a marker of its own until the prompt is rendered)
    let (messages, images) = images::take(messages).map_err(refused)?;
    if !images.is_empty() {
        if let Err(reason) = &engine.vision {
            return Err(refused(format!("this model reads no images: {reason}")));
        }
        // (a pass over an image's rows is a batch of two at the least)
        if engine.model.prefill_batch_size < 2 {
            return Err(refused("this model reads no images with a prefill batch of one token: raise --batch-size"));
        }
    }
    let messages = &messages;
    let tools = body.get("tools").cloned().unwrap_or(Py::Null);
    let thinking = body.get("thinking").is_some_and(Py::truthy);
    let effort = match body.get("reasoning_effort") {
        None => "medium",
        Some(Py::Str(effort)) => effort.as_str(),
        Some(_) => return Err(refused("invalid Straydog reasoning effort")),
    };
    // (a message's own text never writes the template's tokens: see Text::defuse)
    let messages = &chat::defuse_messages(messages, &|own| text.defuse(own).into_owned());
    let prompt = if images.is_empty() { engine.template.render(messages, &tools, thinking, effort) } else { engine.template.render_parts(messages, &tools, thinking, effort) }
        .map_err(|exc| refused(exc.0))?;
    let (ids, prepared, spans) = if images.is_empty() {
        (text.encode(&prompt).map_err(Failure::from)?, Vec::new(), Vec::new())
    } else {
        let pieces = images::pieces(&prompt, images.len()).map_err(refused)?;
        let prepared: Vec<images::QwenImage> = images.iter().map(|bytes| images::prepare_qwen(bytes)).collect::<Result<_, _>>().map_err(refused)?;
        let (ids, spans) = images::qwen_ids(text, &pieces, &prepared).map_err(Failure::from)?;
        (ids, prepared, spans)
    };
    let encode_seconds = started.elapsed().as_secs_f64();
    let image_started = Instant::now();
    let context = engine.model.context;
    // (a prompt too long is refused before its images are encoded)
    if ids.len() >= context {
        return Err(refused("Straydog prompt plus output exceeds the allocated context"));
    }
    let image_rows = if spans.is_empty() {
        Vec::new()
    } else {
        let Some(rows) = engine.image_rows(&prepared, &spans, cancel)? else {
            return Ok(json!({"id": id, "event": "done", "finish_reason": "cancelled",
                             "usage": {"prompt_tokens": ids.len(), "completion_tokens": 0, "total_tokens": ids.len(), "cached_tokens": 0},
                             "gpu": gpu_info_or_null(engine.model.info(), concat!(file!(), ":", line!(), " model.info")),
                             "timings": {"encode_seconds": encode_seconds, "images": spans.len()}}));
        };
        rows
    };
    let image_seconds = image_started.elapsed().as_secs_f64();
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
    let constraint = Constraint::from_request(body.get("constraint").unwrap_or(&Py::Null), text.token_bytes(), &END_TOKENS, thinking)
        .map_err(refused)?;
    let mut constraint = Constraint::suppressing(constraint, body.get("suppress").unwrap_or(&Py::Null), text.token_bytes()).map_err(refused)?;
    // A reply that reasons ends it with the template's `</think>`, its one added token: the deltas go
    // without it and a reasoning_end event marks the place. The same text otherwise written (the
    // reasoning quoting a tag) stays the reasoning's.
    let mut think_close = if thinking { text.added_token(THINK_CLOSE) } else { None };
    // The held tokens can be more than the tokenizer's for the same text, and with no `max_tokens` the reply's room was all that `ids` left: a
    // client that sends none, as some do, ended the worker on "exceeds the allocated context" on the first such reply.
    let agreed = engine.model.agree_with_held(&ids, &text.token_bytes());
    let agreed = if agreed.len() < context { agreed } else { ids.clone() };
    // (the images past such a place have moved with it)
    let (agreed, placed) = match images::spans_in(&spans, &ids, &agreed) {
        Some(placed) => (agreed, placed),
        None => (ids.clone(), spans.clone()),
    };
    if !spans.is_empty() && agreed.len() != ids.len() {
        eprintln!("Straydog: a reply read back moves this request's images by {} tokens", agreed.len() as i64 - ids.len() as i64);
    }
    engine.model.set_images(placed.iter().map(|&(index, start)| ImageSpan {
        start, grid: prepared[index].prepared.grid(), rows: image_rows[index].clone().unwrap_or_default(),
    }).collect());
    let max_new = max_new.min(context - agreed.len());
    // (a conversation an earlier run kept, which this prompt goes on from)
    let kept_seconds = take_kept(engine, output, id, cancel, &agreed);
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
    if !spans.is_empty() {
        timings.insert("images".into(), json!(spans.len()));
        timings.insert("image_seconds".into(), json!(image_seconds));
    }
    if let Some(seconds) = kept_seconds {
        timings.insert("kept_seconds".into(), json!(seconds));
    }
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

    const MIB: u64 = 1 << 20;

    #[test]
    fn autos_margin_scales_on_small_cards_and_stays_from_16_gb() {
        // Windows, as CUDA reports cards: an eighth of each and 256 MiB, in whole 256 MiB
        for (total, margin) in [(2048, 512), (4096, 768), (6144, 1024), (6078, 1024), (8188, 1280), (12288, 1792), (16303, 2048), (16376, 2048), (24564, 2048)] {
            assert_eq!(margin_on(true, total * MIB), margin * MIB, "{total} MiB");
        }
        // what f6362a9 left from 16 GB up: 2 GiB or 3%
        for total in [16303, 24564, 49140, 81920, 131072] {
            assert_eq!(margin_on(true, total * MIB), (2048 * MIB).max(total * MIB / 100 * 3), "{total} MiB");
        }
        // elsewhere, 512 MiB or 3%, whatever the card
        for total in [6144, 16303, 32624] {
            assert_eq!(margin_on(false, total * MIB), (512 * MIB).max(total * MIB / 100 * 3), "{total} MiB");
        }
        // the budget: whole 256 MiB as before from a 12 GB card, off Windows on any; finer on a small one under Windows
        assert_eq!(auto_budget(true, 13_384 * MIB, 16303 * MIB), 11_264 * MIB);
        assert_eq!(auto_budget(false, 6000 * MIB, 6144 * MIB), 5376 * MIB);
        assert_eq!(auto_budget(true, 5344 * MIB, 6144 * MIB), 4288 * MIB);
        assert_eq!(auto_budget(true, 500 * MIB, 6144 * MIB), 0);
    }

    #[test]
    fn signed_headroom_counts_over_budget_usage_and_keeps_the_runtime_margin() {
        let total = 16 << 30;
        assert_eq!(headroom_budget(13 << 30, -(2 << 30), true, total, None), 10 * (1 << 30) + 512 * MIB);
        assert_eq!(headroom_budget(1 << 30, -(2 << 30), true, total, None), 0);
        assert_eq!(headroom_budget(u64::MAX, i64::MAX, true, total, None), total - 512 * MIB);
        assert_eq!(headroom_budget(8 << 30, 3 << 30, false, total, None), auto_budget(true, 11 << 30, total));
        assert_eq!(headroom_budget(8 << 30, -1, false, total, None), auto_budget(true, 8 << 30, total));
        // Small cards keep 64 MiB granularity; larger ones keep 256 MiB and a 3% margin when greater.
        assert_eq!(headroom_budget(2000 * MIB, 0, true, 8 << 30, None), 1472 * MIB);
        assert_eq!(headroom_budget(2000 * MIB, 0, true, total, None), 1280 * MIB);
        let large = 80 << 30;
        assert_eq!(headroom_budget(70 << 30, 0, true, large, None), ((70 << 30) - margin_on(false, large)) / (256 * MIB) * (256 * MIB));
    }

    #[test]
    fn budget_reductions_ignore_cooldown_but_growth_waits_two_more_requests() {
        let budget = 8 << 30;
        let limit = VramLimit::new(budget, 0);
        assert_eq!(vram_target(budget, budget - MIB, 1, false, Some(limit), None), budget - MIB);
        assert_eq!(vram_target(budget, budget + (2 << 30), 3, false, Some(limit), None), budget);
        for requests in [0, 1, 2] {
            assert_eq!(vram_target(budget, budget + (2 << 30), requests, true, Some(limit), None), budget);
        }
        let grown = vram_target(budget, budget + (2 << 30), 3, true, Some(limit), None);
        assert_eq!(grown, budget + VRAM_GROWTH);
        // The retry limit follows the actual resulting budget, even if it was less than requested.
        let actual = grown - 64 * MIB;
        let next = VramLimit::new(actual, 3);
        assert_eq!(next.retry_after, 5);
        assert_eq!(vram_target(actual, budget + (2 << 30), 5, true, Some(next), None), actual);
        assert_eq!(vram_target(actual, budget + (2 << 30), 6, true, Some(next), None), actual + VRAM_GROWTH);
        assert_eq!(vram_target(budget, budget + VRAM_GROWTH - 1, 10, true, None, None), budget);
        assert_eq!(vram_target(budget, budget + (4 << 30), 10, true, None, None), budget + VRAM_GROWTH);
        assert_eq!(VramLimit::new(0, u64::MAX).retry_after, u64::MAX);
    }

    #[test]
    fn explicit_vram_caps_bound_signed_headroom_and_keep_the_safety_margin() {
        let total = 16 << 30;
        assert_eq!(headroom_budget(8 << 30, 4 << 30, true, total, Some(10 << 30)), 10 << 30);
        assert_eq!(headroom_budget(8 << 30, 4 << 30, true, total, Some(12 << 30)), (11 << 30) + 512 * MIB);
        // Positive raw headroom can still leave too little room after the runtime reserve.
        assert_eq!(headroom_budget(8 << 30, 256 * MIB as i64, true, total, Some(8 << 30)), (7 << 30) + 768 * MIB);
        assert_eq!(headroom_budget(13 << 30, -(2 << 30), true, total, Some(12 << 30)), (10 << 30) + 512 * MIB);
        assert_eq!(headroom_budget(13 << 30, -(2 << 30), true, total, Some(8 << 30)), 8 << 30);
        assert_eq!(headroom_budget(1 << 30, -(2 << 30), true, total, Some(8 << 30)), 0);
    }

    #[test]
    fn explicit_vram_caps_keep_cooldowns_and_allow_only_the_final_small_growth() {
        let (budget, cap) = (8 << 30, 10 << 30);
        let limit = VramLimit::new(budget, 0);
        assert_eq!(vram_target(budget, 7 << 30, 1, false, Some(limit), Some(cap)), 7 << 30);
        for requests in [0, 1, 2] {
            assert_eq!(vram_target(budget, 20 << 30, requests, true, Some(limit), Some(cap)), budget);
        }
        assert_eq!(vram_target(budget, 20 << 30, 3, false, Some(limit), Some(cap)), budget);
        assert_eq!(vram_target(budget, 20 << 30, 3, true, Some(limit), Some(cap)), budget + VRAM_GROWTH);
        assert_eq!(vram_target(12 << 30, 20 << 30, 0, false, None, Some(cap)), cap);
        let near = cap - 128 * MIB;
        let last = VramLimit::new(near, 0);
        assert_eq!(vram_target(near, cap, 2, true, Some(last), Some(cap)), near);
        assert_eq!(vram_target(near, cap, 3, false, Some(last), Some(cap)), near);
        assert_eq!(vram_target(near, cap - 1, 3, true, Some(last), Some(cap)), near);
        assert_eq!(vram_target(near, cap, 3, true, Some(last), Some(cap)), cap);
        assert_eq!(vram_target(near, cap, 3, true, Some(last), None), near);
        assert_eq!(vram_target(cap, 20 << 30, 3, true, None, Some(cap)), cap);
    }

    #[test]
    fn explicit_vram_caps_stay_per_device_and_bound_every_target_path() {
        let caps = [6 << 30, 10 << 30];
        assert!(validate_vram_caps(Some(&caps), 2).is_ok());
        assert!(validate_vram_caps(None, 2).is_ok());
        for devices in [0, 1, 3] { assert!(validate_vram_caps(Some(&caps), devices).is_err()); }
        // Runtime free-memory fallback retains the larger desktop reserve, then each GPU's own ceiling.
        let wanted = caps.map(|cap| headroom_budget(8 << 30, 3 << 30, false, 16 << 30, Some(cap)));
        assert_eq!(wanted, [6 << 30, 9 << 30]);
        assert_eq!(headroom_budget(8 << 30, -1, false, 16 << 30, Some(caps[1])), auto_budget(true, 8 << 30, 16 << 30));
        for cap in [0, caps[0], caps[1]] {
            for budget in [0, 5 << 30, 12 << 30] {
                for wanted in [0, 9 << 30, 20 << 30] {
                    for allow_grow in [false, true] {
                        for limit in [None, Some(VramLimit::new(5 << 30, 0))] {
                            assert!(vram_target(budget, wanted, 3, allow_grow, limit, Some(cap)) <= cap);
                        }
                    }
                }
            }
        }
    }

    /// A model's side of a warm start's slot fill: its request count, and the swaps it still wants.
    struct Slots {
        requests: u64,
        wanted: u64,
        steps: u64,
    }

    fn fill(engine: &Arc<Mutex<Slots>>, longest: Duration, mut after: impl FnMut(&mut Slots)) -> Filled {
        fill_slots(&Arc::downgrade(engine), 0, |slots: &Slots| slots.requests, |slots: &mut Slots| {
            slots.steps += 1;
            slots.wanted = slots.wanted.saturating_sub(1);
            after(slots);
            Ok(slots.wanted > 0)
        }, Duration::ZERO, Duration::from_millis(100), longest)
    }

    /// Decides: the fill steps until the slots want nothing more, and a model that wants none is not stepped twice.
    #[test]
    fn the_slot_fill_steps_until_nothing_is_wanted() {
        let engine = Arc::new(Mutex::new(Slots { requests: 0, wanted: 5, steps: 0 }));
        assert_eq!(fill(&engine, Duration::from_secs(30), |_| {}), Filled::Done);
        assert_eq!(engine.lock().unwrap().steps, 5);
        let engine = Arc::new(Mutex::new(Slots { requests: 0, wanted: 0, steps: 0 }));
        assert_eq!(fill(&engine, Duration::from_secs(30), |_| {}), Filled::Done);
        assert_eq!(engine.lock().unwrap().steps, 1);
    }

    /// Decides: a request that began before, or between two steps, ends the fill before the next step; a step that fails ends it with its reason; a
    /// fill that wants more for longer than it may is over; and a worker that ended (nobody holds the engine) is the end of it.
    #[test]
    fn a_request_a_failure_the_time_and_the_engine_ending_each_end_the_slot_fill() {
        let engine = Arc::new(Mutex::new(Slots { requests: 1, wanted: 5, steps: 0 }));
        assert_eq!(fill(&engine, Duration::from_secs(30), |_| {}), Filled::CutShort);
        assert_eq!(engine.lock().unwrap().steps, 0, "a request that began first leaves the slots to itself");
        let engine = Arc::new(Mutex::new(Slots { requests: 0, wanted: 50, steps: 0 }));
        assert_eq!(fill(&engine, Duration::from_secs(30), |slots| if slots.steps == 3 { slots.requests += 1 }), Filled::CutShort);
        assert_eq!(engine.lock().unwrap().steps, 3, "the steps before it stand, none after");
        let engine = Arc::new(Mutex::new(Slots { requests: 0, wanted: 50, steps: 0 }));
        let failed = fill_slots(&Arc::downgrade(&engine), 0, |slots: &Slots| slots.requests, |_: &mut Slots| Err("the device said no".to_string()),
                                Duration::ZERO, Duration::from_millis(100), Duration::from_secs(30));
        assert_eq!(failed, Filled::Failed("the device said no".into()));
        let engine = Arc::new(Mutex::new(Slots { requests: 0, wanted: u64::MAX, steps: 0 }));
        assert_eq!(fill(&engine, Duration::from_millis(20), |_| std::thread::sleep(Duration::from_millis(5))), Filled::Done);
        let steps = engine.lock().unwrap().steps;
        assert!((1..100).contains(&steps), "{steps} steps in 20 ms");
        let weak = Arc::downgrade(&engine);
        drop(engine);
        assert_eq!(fill_slots(&weak, 0, |slots: &Slots| slots.requests, |_: &mut Slots| Ok(true), Duration::ZERO, Duration::from_millis(100),
                              Duration::from_secs(30)), Filled::Gone);
    }

    /// Decides: a request that comes while the fill steps takes the engine between two of them (the fill holds it for a step at a time) and ends the fill.
    #[test]
    fn a_request_that_comes_meanwhile_gets_the_engine_between_steps() {
        let engine = Arc::new(Mutex::new(Slots { requests: 0, wanted: u64::MAX, steps: 0 }));
        let filling = {
            let weak = Arc::downgrade(&engine);
            std::thread::spawn(move || fill_slots(&weak, 0, |slots: &Slots| slots.requests, |slots: &mut Slots| {
                slots.steps += 1;
                std::thread::sleep(Duration::from_millis(2));
                Ok(true)
            }, Duration::from_millis(1), Duration::from_millis(100), Duration::from_secs(60)))
        };
        std::thread::sleep(Duration::from_millis(50));
        let waited = Instant::now();
        {
            let mut engine = engine.lock().unwrap();
            engine.requests += 1;
        }
        let waited = waited.elapsed();
        assert_eq!(filling.join().unwrap(), Filled::CutShort);
        assert!(waited < Duration::from_secs(2), "the request waited {waited:?} for the engine");
        assert!(engine.lock().unwrap().steps >= 1);
    }

    /// Decides: a request that holds the engine (for seconds, as one does) ends the fill after the patience it was given, not when the request ends.
    #[test]
    fn a_request_that_keeps_the_engine_ends_the_fill_without_waiting_for_it() {
        let engine = Arc::new(Mutex::new(Slots { requests: 0, wanted: u64::MAX, steps: 0 }));
        let held = engine.lock().unwrap();
        let waited = Instant::now();
        let filled = fill_slots(&Arc::downgrade(&engine), 0, |slots: &Slots| slots.requests, |_: &mut Slots| Ok(true), Duration::ZERO,
                                Duration::from_millis(50), Duration::from_secs(60));
        assert_eq!(filled, Filled::CutShort);
        assert!(waited.elapsed() < Duration::from_secs(2), "{:?}", waited.elapsed());
        drop(held);
    }

    /// Decides: a placement's key names each GPU with its budget as it was given, the context and the drafter, so that no other placement of the model
    /// has it: another GPU, budget, `auto` against a size, context or drafter makes another key; the same placement makes the same.
    #[test]
    fn a_placements_key_names_what_the_placement_is() {
        let gib = |gib: u64| (gib << 30).to_string();
        let one = vec![(0, "NVIDIA GeForce RTX 5090".to_string())];
        let key = placement_key(&one, &gib(12), 32768, true);
        assert_eq!(key, "0:NVIDIA GeForce RTX 5090@12G; ctx 32768; drafter");
        assert_eq!(placement_key(&one, &gib(12), 32768, true), key);
        assert_eq!(placement_key(&one, "auto", 32768, true), "0:NVIDIA GeForce RTX 5090@auto; ctx 32768; drafter");
        assert_eq!(placement_key(&one, &gib(12), 32768, false), "0:NVIDIA GeForce RTX 5090@12G; ctx 32768; no drafter");
        let others = [placement_key(&one, &gib(11), 32768, true), placement_key(&one, "auto", 32768, true), placement_key(&one, &gib(12), 65536, true),
                      placement_key(&one, &gib(12), 32768, false), placement_key(&[(1, "NVIDIA GeForce RTX 5090".into())], &gib(12), 32768, true),
                      placement_key(&[(0, "NVIDIA GeForce RTX 3090".into())], &gib(12), 32768, true)];
        assert!(others.iter().all(|other| *other != key) && (0..others.len()).all(|at| (0..at).all(|before| others[at] != others[before])), "{others:?}");
        // (two GPUs, each with its own budget; a budget between whole GiB to hundredths; a budget that is not a number, or missing for a GPU, stays as it
        // is given, or is a mark: the worker has refused those before it comes to a key)
        let two = vec![(0, "NVIDIA GeForce RTX 5090".to_string()), (1, "NVIDIA GeForce RTX 3090".to_string())];
        let given = format!("{},{}", gib(12), (11.5 * (1u64 << 30) as f64) as u64);
        assert_eq!(placement_key(&two, &given, 8192, true), "0:NVIDIA GeForce RTX 5090@12G,1:NVIDIA GeForce RTX 3090@11.5G; ctx 8192; drafter");
        assert_eq!(placement_key(&two, "auto", 8192, true), "0:NVIDIA GeForce RTX 5090@auto,1:NVIDIA GeForce RTX 3090@auto; ctx 8192; drafter");
        assert_eq!(placement_key(&two, &gib(12), 8192, true), "0:NVIDIA GeForce RTX 5090@12G,1:NVIDIA GeForce RTX 3090@?; ctx 8192; drafter");
        assert_eq!(budget_label(10 << 30), "10G");
        assert_eq!(budget_label(0), "0G");
        assert_eq!(budget_label((6.25 * (1u64 << 30) as f64) as u64), "6.25G");
        assert_eq!(placement_key(&one, "lots", 1, true), "0:NVIDIA GeForce RTX 5090@lots; ctx 1; drafter");
    }

    #[test]
    fn a_refusal_under_auto_says_what_it_took() {
        let reason = "the GPU budget cannot hold the model's dense weights";
        let refused = auto_too_small(format!("{reason}; raise --vram"), "auto", &[3 << 30]);
        assert!(refused.starts_with(reason) && refused.contains("`auto` took 3.00 GiB") && refused.contains("close what holds"), "{refused}");
        assert_eq!(auto_too_small(format!("{reason}; raise --vram"), "5368709120", &[5 << 30]), format!("{reason}; raise --vram"));
        assert_eq!(auto_too_small("other".into(), "auto", &[3 << 30]), "other");
    }

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
            if prompt == "recovered-wait" {
                return Err(refused("The GPU wait exceeded its limit. The GPU state was reset; you can retry with the next request."));
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
    fn reset_failure_replaces_a_recovered_waits_retry_message() {
        let engine = Mutex::new(TokenizerOnly { resets: 0, reset_fails: true });
        let failed = request(&engine, &tokenizer(), "recovered-wait", "reset-fails");
        assert_eq!(failed["fatal"], true);
        let message = failed["message"].as_str().unwrap();
        assert!(message.contains("device reset failed") && message.contains("Restart it before retrying"));
        assert!(!message.contains("next request"));
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
        // (a K/V that found no memory to grow into, and a copy of the conversation that found none: the devices are sound)
        assert!(!engine_failed(format!("site: {}", straydog_engine::model::KV_NO_MEMORY), "site").fatal);
        assert!(!engine_failed(format!("GPU 0 could not reduce its budget; {}", straydog_engine::model::VRAM_NO_MEMORY), "site").fatal);
        assert!(!engine_failed(format!("site: {}", straydog_engine::gpu::HOST_NO_MEMORY), "site").fatal);
        assert!(!engine_failed(format!("site: {}", straydog_engine::gpu::HYBRID_WAIT_RECOVERED), "site").fatal);
        // (a row read the working-set quota refused, after the smaller and cached retries; a bare 1453 stays fatal)
        assert!(!engine_failed(format!("row table read failed: os error 1453; {}", straydog_engine::rows::ROW_READ_REFUSED), "site").fatal);
        assert!(engine_failed("row table read failed: os error 1453", "site").fatal);
        // (what an image asked is refused, the engine reset)
        assert!(!engine_failed(format!("no memory for its positions {}", straydog_engine::model::IMAGE_REFUSED), "site").fatal);
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
