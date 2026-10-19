//! The OpenAI-compatible HTTP API (docs/DESIGN.md §6) over an [`Engine`]: the server's
//! runtime, which owns the isolated worker.
//!
//! Interface fixed for phase R4; the router is to be written.

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use straydog_engine::pyjson;

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
    /// "error", "message", "fatal"}`.
    pub events: BoxStream<'static, serde_json::Value>,
    /// Ask the worker to end early (a stop string matched); its `done` event still follows.
    pub interrupt: Box<dyn Fn() + Send + Sync>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineState {
    Loading { progress: serde_json::Value },
    Ready { context: u64, placement: serde_json::Value },
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
/// "seed": n, "thinking": bool, "reasoning_effort": "low" | "medium" | "xhigh",
/// "checkpoint": bool}`.
pub struct WorkerRequest;
