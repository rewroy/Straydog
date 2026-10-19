//! The browser dashboard at `/dashboard`: the engine's state and speed, the GPUs, the host,
//! the settings and the log, next to a chat. The page and its assets need no key; the data
//! behind them needs the API key once one is configured.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use axum::extract::{Query, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::{Effort, Gemm, Settings, SplitMode};
use crate::monitor::{Host, LogBuffer};
use crate::runtime::Runtime;

const PAGE: &str = include_str!("../dashboard/index.html");
const SCRIPT: &str = include_str!("../dashboard/app.js");
const STYLE: &str = include_str!("../dashboard/app.css");
const ICON: &str = include_str!("../dashboard/icon.svg");

/// Every asset is the server's own: no inline code, no other origin.
const POLICY: &str = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; \
                      base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

/// The most log lines one read returns.
const LOG_READ: usize = 500;
/// Samples kept for the charts: two minutes, one a second.
const HISTORY: usize = 120;
/// How long sampling goes on after a dashboard last asked for the status.
const WATCHED: Duration = Duration::from_secs(15);

pub struct Dashboard {
    runtime: Arc<Runtime>,
    logs: Arc<LogBuffer>,
    monitor: Arc<Monitor>,
    api_key: Option<String>,
    server: Value,
}

/// What the sampler keeps while a dashboard is open: the latest host snapshot and the last two
/// minutes of samples.
#[derive(Default)]
struct Monitor {
    host: Host,
    latest: Mutex<Value>,
    history: Mutex<VecDeque<Value>>,
    sampler: Mutex<Option<tokio::task::JoinHandle<()>>>,
    viewed: Mutex<Option<Instant>>,
}

impl Dashboard {
    pub fn new(runtime: Arc<Runtime>, logs: Arc<LogBuffer>, settings: &Settings) -> Dashboard {
        Dashboard { runtime, logs, monitor: Arc::default(), api_key: settings.api_key.clone(), server: describe(settings) }
    }

    /// Sample the host and the engine every second while a dashboard is open: from a status
    /// request until none came for [`WATCHED`]. Nothing is read, and nothing kept, otherwise.
    /// Between samples the sampler holds no reference to the engine.
    fn watch(&self) {
        *self.monitor.viewed.lock().unwrap() = Some(Instant::now());
        let mut sampler = self.monitor.sampler.lock().unwrap();
        if sampler.as_ref().is_some_and(|task| !task.is_finished()) {
            return;
        }
        let (monitor, runtime) = (self.monitor.clone(), Arc::downgrade(&self.runtime));
        *sampler = Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            let mut rates = Rates::default();
            loop {
                interval.tick().await;
                if monitor.viewed.lock().unwrap().is_none_or(|viewed| viewed.elapsed() > WATCHED) {
                    *monitor.latest.lock().unwrap() = Value::Null;
                    monitor.history.lock().unwrap().clear();
                    return;
                }
                let reader = monitor.clone();
                let Ok(host) = tokio::task::spawn_blocking(move || reader.host.snapshot()).await else { continue };
                let Some(live) = runtime.upgrade().map(|runtime| runtime.live()) else { return };
                let (decode, prefill) = rates.update(&live);
                let sample = sample(&host, decode, prefill);
                *monitor.latest.lock().unwrap() = host;
                let mut history = monitor.history.lock().unwrap();
                if history.len() == HISTORY {
                    history.pop_front();
                }
                history.push_back(sample);
            }
        }));
    }

    /// Stop sampling and wait until the sampler has ended (before the engine closes).
    pub async fn stop(&self) {
        let task = self.monitor.sampler.lock().unwrap().take();
        if let Some(task) = task {
            task.abort();
            let _ = task.await;
        }
    }
}

fn unix_now() -> f64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0.0, |time| time.as_secs_f64())
}

/// The running generation's speed between samples: decode in tokens per second over the
/// last second, prefill on average since the prompt began (its progress arrives per batch).
#[derive(Default)]
struct Rates {
    /// The generation's age, its token count and when that was read.
    previous: Option<(f64, u64, Instant)>,
    /// When the prefill began and how far it was then.
    prefill_start: Option<(Instant, u64)>,
}

impl Rates {
    fn update(&mut self, live: &Value) -> (f64, f64) {
        let Some(seconds) = live["seconds"].as_f64() else {
            *self = Rates::default();
            return (0.0, 0.0);
        };
        let (tokens, done) = (live["tokens"].as_u64().unwrap_or(0), live["prefill_done"].as_u64().unwrap_or(0));
        // A younger generation is a new one.
        if self.previous.is_some_and(|(age, _, _)| seconds < age) {
            *self = Rates::default();
        }
        let now = Instant::now();
        let decode = match self.previous {
            Some((_, before, at)) if tokens > before => (tokens - before) as f64 / now.duration_since(at).as_secs_f64(),
            _ => 0.0,
        };
        let prefill = if tokens == 0 {
            let (start, base) = *self.prefill_start.get_or_insert((now, done));
            let elapsed = now.duration_since(start).as_secs_f64();
            if done > base && elapsed > 0.0 { (done - base) as f64 / elapsed } else { 0.0 }
        } else {
            0.0
        };
        self.previous = Some((seconds, tokens, now));
        (decode, prefill)
    }
}

/// One point of every chart.
fn sample(host: &Value, decode: f64, prefill: f64) -> Value {
    let total = host["memory_total"].as_f64().unwrap_or(0.0).max(1.0);
    let used = total - host["memory_available"].as_f64().unwrap_or(0.0);
    let gpus: Vec<Value> = host["gpus"].as_array().into_iter().flatten().map(|gpu| {
        let hottest = gpu["temperatures"].as_array().into_iter().flatten()
            .filter_map(|reading| reading["celsius"].as_f64())
            .fold(None, |hottest: Option<f64>, celsius| Some(hottest.map_or(celsius, |hottest| hottest.max(celsius))));
        let power = match (gpu["power_watts"].as_f64(), gpu["power_cap_watts"].as_f64()) {
            (Some(watts), Some(cap)) if cap > 0.0 => Some(100.0 * watts / cap),
            _ => None,
        };
        json!({"pci": gpu["pci"], "busy": gpu["busy_percent"], "temperature": hottest, "power": power})
    }).collect();
    json!({
        "time": unix_now(),
        "cpu": host["cpu_usage"],
        "cpu_temperature": host["cpu_temperature"],
        "memory": 100.0 * used / total,
        "server_memory": 100.0 * host["server"]["memory"].as_f64().unwrap_or(0.0) / total,
        "gpus": gpus,
        "decode": decode,
        "prefill": prefill,
    })
}

/// The settings as the dashboard shows them; never the key itself.
fn describe(settings: &Settings) -> Value {
    let started = unix_now();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "backend": if cfg!(feature = "cuda") { "CUDA" } else { "HIP" },
        "started": started,
        "alias": settings.alias,
        "quantization": settings.quant,
        "model": settings.model.display().to_string(),
        "model_draft": settings.model_draft.as_ref().map(|path| path.display().to_string()),
        "draft_max": settings.draft_max,
        "context": settings.ctx_size,
        "batch_size": settings.batch_size,
        "devices": settings.device,
        "split_mode": match settings.split_mode { SplitMode::Experts => "experts", SplitMode::Layer => "layer" },
        "vram_gib": settings.vram,
        "state_ram_gib": settings.state_ram,
        "gemm": match settings.gemm { Gemm::Fp16 => "fp16", Gemm::Fp32 => "fp32" },
        "gemm_min_batch": settings.gemm_min_batch,
        "address": format!("{}:{}", settings.host, settings.port),
        "api_key": settings.api_key.is_some(),
        "temperature": settings.temp,
        "seed": settings.seed,
        "reasoning_effort": match settings.reasoning_effort {
            Effort::None => "none",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
        },
        "startup_timeout": settings.startup_timeout,
    })
}

pub fn router(dashboard: Arc<Dashboard>) -> Router {
    Router::new()
        .route("/dashboard", get(|| async { asset(PAGE, "text/html; charset=utf-8") }))
        .route("/dashboard/app.js", get(|| async { asset(SCRIPT, "text/javascript; charset=utf-8") }))
        .route("/dashboard/app.css", get(|| async { asset(STYLE, "text/css; charset=utf-8") }))
        .route("/dashboard/icon.svg", get(|| async { asset(ICON, "image/svg+xml") }))
        .route("/dashboard/status", get(status))
        .route("/dashboard/logs", get(logs))
        .with_state(dashboard)
}

fn asset(body: &'static str, kind: &'static str) -> Response {
    let mut response = body.into_response();
    let headers = response.headers_mut();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static(kind));
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    secure(headers);
    response
}

fn secure(headers: &mut HeaderMap) {
    headers.insert(HeaderName::from_static("content-security-policy"), HeaderValue::from_static(POLICY));
    headers.insert(HeaderName::from_static("x-content-type-options"), HeaderValue::from_static("nosniff"));
    headers.insert(HeaderName::from_static("referrer-policy"), HeaderValue::from_static("no-referrer"));
}

fn data(value: Value) -> Response {
    let mut response = Json(value).into_response();
    response.headers_mut().insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    secure(response.headers_mut());
    response
}

async fn status(State(dashboard): State<Arc<Dashboard>>, headers: HeaderMap, Query(query): Query<StatusQuery>) -> Response {
    if !straydog_api::authorized(dashboard.api_key.as_deref(), &headers) {
        return straydog_api::unauthorized();
    }
    dashboard.watch();
    // A sampler just started takes its first sample at once.
    for _ in 0..50 {
        if !dashboard.monitor.latest.lock().unwrap().is_null() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let host = dashboard.monitor.latest.lock().unwrap().clone();
    let samples: Vec<Value> = dashboard.monitor.history.lock().unwrap().iter()
        .filter(|sample| sample["time"].as_f64().is_some_and(|time| time > query.since))
        .cloned()
        .collect();
    data(json!({"now": unix_now(), "server": dashboard.server, "engine": dashboard.runtime.snapshot(), "host": host,
                "samples": samples}))
}

#[derive(Deserialize)]
struct StatusQuery {
    /// Samples after this time (Unix seconds) only.
    #[serde(default)]
    since: f64,
}

#[derive(Deserialize)]
struct LogQuery {
    #[serde(default)]
    after: u64,
}

async fn logs(State(dashboard): State<Arc<Dashboard>>, headers: HeaderMap, Query(query): Query<LogQuery>) -> Response {
    if !straydog_api::authorized(dashboard.api_key.as_deref(), &headers) {
        return straydog_api::unauthorized();
    }
    let (next, lines) = dashboard.logs.read(query.after, LOG_READ);
    data(json!({"next": next, "lines": lines}))
}
