//! The OpenAI-compatible HTTP API (docs/DESIGN.md §6) over an [`Engine`]: the server's
//! runtime, which owns the isolated worker.
//!
//! `GET /health` needs no key; `GET /v1/models`, `POST /v1/chat/completions` and
//! llama-server's `GET /props` and `GET /slots` need the Bearer key once one is configured. A chat completion request becomes a worker request
//! ([`WorkerRequest`]); the worker's raw template text comes back as reasoning, answer and
//! structured tool calls.

mod answer;
mod request;

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::header::{AUTHORIZATION, CONTENT_LENGTH};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, StreamExt};
use serde_json::json;
use straydog_engine::pyjson;
pub use straydog_engine::reply::Markup;
use straydog_engine::constraint::NOT_CP932_HAN;
use straydog_engine::sampling::Sampling;
use subtle::ConstantTimeEq;

pub use request::ReasoningEffort;

/// The engine as the API sees it.
pub trait Engine: Send + Sync + 'static {
    fn state(&self) -> EngineState;

    /// Start a generation for a worker request body (see [`WorkerRequest`]). The events end
    /// with a `done` or `error` event; dropping the stream cancels the generation.
    fn generate(&self, request: pyjson::Value) -> BoxFuture<'static, Result<Generation, SubmitError>>;
}

/// A running generation.
pub struct Generation {
    /// Worker events as JSON objects: `{"event": "delta", "content"}`, `{"event": "progress",
    /// "done", "total"}`, `{"event": "done", "finish_reason", "usage": {"prompt_tokens",
    /// "completion_tokens", "total_tokens", "cached_tokens"}, "timings"}` or `{"event":
    /// "error", "message", "fatal"}`. A non-fatal error is the request's fault (400), a fatal
    /// one the engine's (500).
    pub events: BoxStream<'static, serde_json::Value>,
    /// Ask the worker to end early (a stop string matched); its `done` event still follows.
    pub interrupt: Box<dyn Fn() + Send + Sync>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineState {
    /// `progress` is shown as is by `/health`.
    Loading { progress: serde_json::Value },
    /// `context` in tokens; `speculative` MTP drafts per step (0: none); `processing` while a
    /// generation runs or waits. `placement` is shown as is by `/v1/models`.
    Ready { context: u64, speculative: u64, processing: bool, placement: serde_json::Value },
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitError {
    /// The bounded queue is full (503).
    Busy(String),
    /// The request cannot run (400).
    Invalid(String),
    /// Loading or failed (503).
    Unavailable(String),
}

/// The worker request body, built from an OpenAI request with Python-compatible JSON
/// (`messages` and `tools` keep the client's numbers, key order and strings exactly):
/// `{"messages": [...], "tools": [...] | null, "max_tokens": n | null, "temperature": t,
/// "top_k": k, "top_p": p, "min_p": p, "presence_penalty": x, "seed": n, "thinking": bool,
/// "reasoning_effort": "low" | "medium" | "xhigh",
/// "checkpoint": bool}`, and `"suppress": "not_cp932_han"` only when asked for. `"request_id"` follows only when the
/// client sent one ([`REQUEST_ID`]): the worker reads nothing of it, and the server's log line for the request names it.
///
/// Each message keeps, in the client's order, the fields the template reads or a client
/// sends back (`role`, `content`, `name`, `tool_call_id`, `tool_calls`,
/// `reasoning_content`) that are not null; a developer message is a system one, and text
/// parts are joined into one string as the template would render them.
pub struct WorkerRequest;

/// What the API takes from the server's settings (docs/DESIGN.md §5).
#[derive(Clone)]
pub struct ApiConfig {
    /// The model id in `/v1/models` and in every answer (`--alias`).
    pub model_id: String,
    /// The configured context in tokens, shown as `n_ctx` until the engine reports its own.
    pub context: u64,
    /// The bundle's quantization, e.g. `IQ3_S`.
    pub quantization: String,
    /// The Bearer key every route but `/health` then requires (`--api-key`).
    pub api_key: Option<String>,
    /// Sampling when a request gives none (`--temp`, `--top-k`, ...).
    pub sampling: SamplingDefaults,
    /// The sampling seed when a request gives none (`--seed`); unset, a fresh one each time.
    pub seed: Option<u64>,
    /// The reasoning when a request gives none (`--reasoning-effort`).
    pub reasoning_effort: ReasoningEffort,
    /// The model's `created` time in Unix seconds: when the server started.
    pub created: u64,
    /// The model family's template: its reasoning levels and reply markup.
    pub markup: Markup,
}

/// The server's sampling settings: each one set applies to every request that gives none, and
/// each one unset follows the model card for the request's mode (thinking or not).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub presence_penalty: Option<f64>,
}

impl SamplingDefaults {
    pub fn resolve(&self, thinking: bool) -> Sampling {
        self.resolve_with(Sampling::recommended(thinking))
    }

    /// Each setting, else the given model card's.
    pub fn resolve_with(&self, card: Sampling) -> Sampling {
        Sampling {
            temperature: self.temperature.unwrap_or(card.temperature),
            top_k: self.top_k.unwrap_or(card.top_k),
            top_p: self.top_p.unwrap_or(card.top_p),
            min_p: self.min_p.unwrap_or(card.min_p),
            presence_penalty: self.presence_penalty.unwrap_or(card.presence_penalty),
        }
    }
}

/// The largest request body.
pub const BODY_LIMIT: usize = 64 * 1024 * 1024;
/// What a request sees while the model loads (llama-server's wording).
const LOADING: &str = "Loading model";

/// The routes over one engine.
pub fn router(engine: Arc<dyn Engine>, config: ApiConfig) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/props", get(props))
        .route("/slots", get(slots))
        .fallback(|| async { ApiError::new(StatusCode::NOT_FOUND, "not_found_error", "no such route") })
        .method_not_allowed_fallback(|| async {
            ApiError::new(StatusCode::METHOD_NOT_ALLOWED, "invalid_request_error", "method not allowed")
        })
        .with_state(Arc::new(Api { engine, config }))
}

struct Api {
    engine: Arc<dyn Engine>,
    config: ApiConfig,
}

impl Api {
    fn authorize(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        if authorized(self.config.api_key.as_deref(), headers) {
            Ok(())
        } else {
            Err(ApiError::new(StatusCode::UNAUTHORIZED, "authentication_error", "invalid API key"))
        }
    }

    /// The engine's state once it is ready: 503 while it loads or after it failed.
    fn ready(&self) -> Result<(u64, u64, bool), ApiError> {
        match self.engine.state() {
            EngineState::Ready { context, speculative, processing, .. } => Ok((context, speculative, processing)),
            EngineState::Loading { .. } => Err(ApiError::unavailable(LOADING)),
            EngineState::Failed(message) => Err(ApiError::unavailable(message)),
        }
    }

    /// Whether the model reads images (the worker reports its vision encoder).
    fn vision(&self) -> bool {
        matches!(self.engine.state(), EngineState::Ready { placement, .. } if placement["vision"] == true)
    }

    /// The engine's one slot as llama-server describes a slot: it answers one request at a
    /// time, with the server's defaults and no output limit but the context.
    fn slot(&self) -> Result<serde_json::Value, ApiError> {
        let (context, speculative, processing) = self.ready()?;
        // The defaults of the default reasoning (the card the worker reports, if it does); -1 is
        // llama-server's random seed.
        let sampling = match self.engine.state() {
            EngineState::Ready { placement, .. } if placement["sampling"].is_object() => {
                self.config.sampling.resolve_with(card(&placement["sampling"]))
            }
            _ => self.config.sampling.resolve(self.config.reasoning_effort != ReasoningEffort::None),
        };
        Ok(json!({
            "id": 0, "n_ctx": context, "speculative": speculative > 0, "is_processing": processing,
            "params": {"n_predict": -1, "seed": self.config.seed.map_or(-1, |seed| seed as i128),
                       "temperature": sampling.temperature, "top_k": sampling.top_k, "top_p": sampling.top_p,
                       "min_p": sampling.min_p, "presence_penalty": sampling.presence_penalty,
                       "speculative.n_max": speculative},
        }))
    }
}

/// A model card the worker reports (`{"temperature", "top_k", "top_p", "min_p", "presence_penalty"}`).
fn card(card: &serde_json::Value) -> Sampling {
    let number = |key: &str, neutral: f64| card[key].as_f64().unwrap_or(neutral);
    Sampling {
        temperature: number("temperature", 1.0),
        top_k: card["top_k"].as_u64().unwrap_or(0) as usize,
        top_p: number("top_p", 1.0),
        min_p: number("min_p", 0.0),
        presence_penalty: number("presence_penalty", 0.0),
    }
}

/// Whether `headers` carry the Bearer `key`, compared in constant time; without a key, every
/// client may call.
pub fn authorized(key: Option<&str>, headers: &HeaderMap) -> bool {
    let Some(key) = key else { return true };
    headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok()?.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .is_some_and(|(_, token)| bool::from(token.trim().as_bytes().ct_eq(key.as_bytes())))
}

/// The answer to a request without the key.
pub fn unauthorized() -> Response {
    ApiError::new(StatusCode::UNAUTHORIZED, "authentication_error", "invalid API key").into_response()
}

/// An OpenAI-style error, `{"error": {"code", "message", "type"}}`, with its HTTP status.
#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> ApiError {
        ApiError { status, kind, message: message.into() }
    }

    fn invalid(message: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    fn server(message: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
    }

    fn unavailable(message: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "unavailable_error", message)
    }

    fn object(&self) -> serde_json::Value {
        json!({"code": self.status.as_u16(), "message": self.message, "type": self.kind})
    }

    fn body(&self) -> serde_json::Value {
        json!({"error": self.object()})
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

impl From<SubmitError> for ApiError {
    fn from(error: SubmitError) -> ApiError {
        match error {
            SubmitError::Busy(message) => ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "server_busy", message),
            SubmitError::Invalid(message) => ApiError::invalid(message),
            SubmitError::Unavailable(message) => ApiError::unavailable(message),
        }
    }
}

async fn health(State(api): State<Arc<Api>>) -> Response {
    match api.engine.state() {
        EngineState::Ready { .. } => Json(json!({"status": "ok"})).into_response(),
        EngineState::Loading { progress } => {
            let error = ApiError::unavailable(LOADING).object();
            (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": "loading", "progress": progress, "error": error})))
                .into_response()
        }
        EngineState::Failed(message) => {
            let error = ApiError::server(message).object();
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"status": "error", "error": error}))).into_response()
        }
    }
}

async fn models(State(api): State<Arc<Api>>, headers: HeaderMap) -> Result<Json<serde_json::Value>, ApiError> {
    api.authorize(&headers)?;
    let config = &api.config;
    let mut meta = json!({"n_ctx": config.context, "quantization": config.quantization, "ready": false, "vision": api.vision(),
                          "suppress": [NOT_CP932_HAN]});
    if let EngineState::Ready { context, placement, .. } = api.engine.state() {
        meta["n_ctx"] = context.into();
        meta["ready"] = true.into();
        meta["placement"] = placement;
    }
    let model = json!({"id": config.model_id, "object": "model", "created": config.created, "owned_by": "straydog",
                       "meta": meta});
    Ok(Json(json!({"object": "list", "data": [model]})))
}

/// llama-server's `/props`, where its clients read the context (`n_ctx`).
async fn props(State(api): State<Arc<Api>>, headers: HeaderMap) -> Result<Json<serde_json::Value>, ApiError> {
    api.authorize(&headers)?;
    Ok(Json(json!({
        "default_generation_settings": api.slot()?, "total_slots": 1, "model_alias": api.config.model_id,
        "modalities": {"vision": api.vision(), "audio": false},
        // (what a chat completion's `suppress` may name)
        "suppress": [NOT_CP932_HAN],
        "build_info": concat!("straydog-", env!("CARGO_PKG_VERSION")), "is_sleeping": false,
    })))
}

/// llama-server's `/slots`.
async fn slots(State(api): State<Arc<Api>>, headers: HeaderMap) -> Result<Json<serde_json::Value>, ApiError> {
    api.authorize(&headers)?;
    Ok(Json(json!([api.slot()?])))
}

/// A client that disconnects drops this handler, or the stream it returned, and with it the
/// generation: hyper ends a connection whose client closed it mid-request.
async fn chat_completions(State(api): State<Arc<Api>>, headers: HeaderMap, body: Body) -> Result<Response, ApiError> {
    api.authorize(&headers)?;
    api.ready()?;
    let payload = read_payload(&headers, body).await?;
    let (mut request, shape) = request::translate(&payload, &api.config, api.vision()).map_err(|mut error| {
        // (what the worker found wrong with the model's vision encoder file, for the one who sent the image)
        if error.message == request::NO_IMAGES {
            if let EngineState::Ready { placement, .. } = api.engine.state() {
                if let Some(reason) = placement["vision_refusal"].as_str() {
                    error.message = format!("{}: {reason}", request::NO_IMAGES);
                }
            }
        }
        error
    })?;
    if let (Some(id), pyjson::Value::Object(fields)) = (request_id(&headers), &mut request) {
        fields.push(("request_id".into(), pyjson::Value::Str(id)));
    }
    let generation = api.engine.generate(request).await?;
    answer::respond(generation, shape, &api.config.model_id).await
}

/// The header a client names its request by, for matching its own records with the server's log.
pub const REQUEST_ID: &str = "x-request-id";
/// The most of it the log keeps.
const REQUEST_ID_CHARS: usize = 64;

/// The client's [`REQUEST_ID`] as the log line names it: printable ASCII but the space, and at most
/// [`REQUEST_ID_CHARS`] of it, so that no caller can break or flood the line. None when nothing of it is left.
fn request_id(headers: &HeaderMap) -> Option<String> {
    let id: String = headers.get(REQUEST_ID)?.as_bytes().iter().filter(|byte| byte.is_ascii_graphic())
        .take(REQUEST_ID_CHARS).map(|&byte| char::from(byte)).collect();
    (!id.is_empty()).then_some(id)
}

/// The request body as Python's `json.loads` reads it: the prompt is rendered from it.
async fn read_payload(headers: &HeaderMap, body: Body) -> Result<pyjson::Value, ApiError> {
    let too_large = || ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", "request body too large");
    let declared = headers.get(CONTENT_LENGTH).and_then(|value| value.to_str().ok()?.parse::<u64>().ok());
    if declared.is_some_and(|length| length > BODY_LIMIT as u64) {
        return Err(too_large());
    }
    let mut data = Vec::new();
    let mut chunks = body.into_data_stream();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(|_| ApiError::invalid("the request body could not be read"))?;
        if data.len() + chunk.len() > BODY_LIMIT {
            return Err(too_large());
        }
        data.extend_from_slice(&chunk);
    }
    let text = std::str::from_utf8(&data).map_err(|_| ApiError::invalid("the request body is not UTF-8"))?;
    let payload =
        pyjson::parse(text).map_err(|error| ApiError::invalid(format!("the request body is not valid JSON: {error}")))?;
    if !matches!(payload, pyjson::Value::Object(_)) {
        return Err(ApiError::invalid("the request body must be a JSON object"));
    }
    Ok(payload)
}
