//! The browser dashboard at `/dashboard`: the engine's state and speed, the GPUs, the host,
//! the settings and the log, next to a chat. The page and its assets need no key; the data
//! behind them needs the API key once one is configured.

use std::sync::Arc;

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

pub struct Dashboard {
    runtime: Arc<Runtime>,
    logs: Arc<LogBuffer>,
    host: Host,
    api_key: Option<String>,
    server: Value,
}

impl Dashboard {
    pub fn new(runtime: Arc<Runtime>, logs: Arc<LogBuffer>, settings: &Settings) -> Dashboard {
        Dashboard { runtime, logs, host: Host::default(), api_key: settings.api_key.clone(), server: describe(settings) }
    }
}

/// The settings as the dashboard shows them; never the key itself.
fn describe(settings: &Settings) -> Value {
    let started = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |time| time.as_secs_f64());
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

async fn status(State(dashboard): State<Arc<Dashboard>>, headers: HeaderMap) -> Response {
    if !straydog_api::authorized(dashboard.api_key.as_deref(), &headers) {
        return straydog_api::unauthorized();
    }
    // Reading /proc for every process takes a few milliseconds: off the async workers.
    let reader = dashboard.clone();
    let host = tokio::task::spawn_blocking(move || reader.host.snapshot()).await.unwrap_or(Value::Null);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |time| time.as_secs_f64());
    data(json!({"now": now, "server": dashboard.server, "engine": dashboard.runtime.snapshot(), "host": host}))
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
