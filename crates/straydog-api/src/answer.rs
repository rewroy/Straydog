//! The worker's raw template text as an OpenAI answer: one `chat.completion`, or
//! `chat.completion.chunk` events over SSE.
//!
//! The text is cut at the first stop string, then split into reasoning, answer and tool
//! calls by the template's own markup ([`straydog_engine::reply`]). Nothing is sent before
//! the answer's first step, so an error up to then still has its status; after it, an
//! `error` event ends the stream. Dropping the answer drops the [`Generation`], which
//! cancels it.

use std::convert::Infallible;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::http::{HeaderName, HeaderValue};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures::stream::{self, StreamExt};
use serde_json::{json, Map, Value};
use straydog_engine::reply::{ReasoningSplitter, StopMatcher, ToolCall, ToolCallSplitter};

use crate::request::Shape;
use crate::{ApiError, Generation};

/// Prefill progress becomes an SSE comment at most this often: enough to keep clients and
/// proxies from timing out.
const KEEP_ALIVE: Duration = Duration::from_secs(1);

/// The answer to a request: JSON once the generation ends, or SSE once its first step is known.
pub(crate) async fn respond(generation: Generation, shape: Shape, model: &str) -> Result<Response, ApiError> {
    let reply = Reply {
        stop: StopMatcher::new(&shape.stops),
        reasoning: ReasoningSplitter::new(shape.thinking),
        tools: ToolCallSplitter::new(shape.tools.as_ref()),
        calls: Vec::new(),
    };
    let mut completion = Completion { generation, reply };
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_secs());
    let meta = Meta { id: format!("chatcmpl-{}", random_hex(16)), created, model: model.to_string() };
    if !shape.stream {
        return Ok(Json(completion.collect(meta).await?).into_response());
    }
    let first = completion.step().await?;
    let chunks = Chunks { completion, meta, include_usage: shape.include_usage, role_sent: false, commented: None };
    let events = stream::unfold(Some((chunks, Some(first))), |state| async move {
        let (mut chunks, pending) = state?;
        let step = match pending {
            Some(step) => Ok(step),
            None => chunks.completion.step().await,
        };
        let (events, last) = chunks.events(step);
        Some((events, (!last).then_some((chunks, None))))
    })
    .flat_map(|events| stream::iter(events.into_iter().map(Ok::<_, Infallible>)));
    let headers = [(HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"))];
    Ok((headers, Sse::new(events)).into_response())
}

/// Part of the answer's message.
enum Delta {
    Reasoning(String),
    Content(String),
    Calls(Vec<ToolCall>),
}

/// Raw template text in, message deltas out.
struct Reply {
    stop: StopMatcher,
    reasoning: ReasoningSplitter,
    tools: ToolCallSplitter,
    /// The tool calls, once the text has ended.
    calls: Vec<ToolCall>,
}

impl Reply {
    fn feed(&mut self, text: &str) -> Vec<Delta> {
        let text = self.stop.feed(text);
        self.split(&text)
    }

    fn split(&mut self, text: &str) -> Vec<Delta> {
        let (content, thought) = self.reasoning.feed(text);
        let content = self.tools.feed(&content);
        let mut deltas = Vec::new();
        if !thought.is_empty() {
            deltas.push(Delta::Reasoning(thought));
        }
        if !content.is_empty() {
            deltas.push(Delta::Content(content));
        }
        deltas
    }

    /// Everything still held back once the text has ended.
    fn finish(&mut self) -> Vec<Delta> {
        let held = self.stop.finish();
        let mut deltas = self.split(&held);
        let thought = self.reasoning.finish();
        if !thought.is_empty() {
            deltas.push(Delta::Reasoning(thought));
        }
        let (late, calls) = self.tools.finish();
        if !late.is_empty() {
            deltas.push(Delta::Content(late));
        }
        if !calls.is_empty() {
            self.calls = calls.clone();
            deltas.push(Delta::Calls(calls));
        }
        deltas
    }
}

/// One step of an answer.
enum Step {
    Deltas(Vec<Delta>),
    /// Prefill progress in tokens: done, total.
    Progress(u64, u64),
    /// The last deltas, then the end.
    Done(Vec<Delta>, End),
}

struct End {
    finish_reason: &'static str,
    usage: Value,
    timings: Value,
}

/// A generation read as answer steps.
struct Completion {
    generation: Generation,
    reply: Reply,
}

impl Completion {
    /// The next step; after `Done` or an error there is none.
    async fn step(&mut self) -> Result<Step, ApiError> {
        loop {
            let Some(event) = self.generation.events.next().await else {
                return Err(ApiError::server("Straydog generation ended without a terminal event"));
            };
            match event["event"].as_str() {
                Some("delta") => {
                    let stopped = self.reply.stop.stopped;
                    let deltas = self.reply.feed(event["content"].as_str().unwrap_or_default());
                    if self.reply.stop.stopped && !stopped {
                        (self.generation.interrupt)();
                    }
                    if !deltas.is_empty() {
                        return Ok(Step::Deltas(deltas));
                    }
                }
                Some("progress") => return Ok(Step::Progress(count(&event["done"]), count(&event["total"]))),
                Some("done") => {
                    let deltas = self.reply.finish();
                    return Ok(Step::Done(deltas, self.end(&event)));
                }
                Some("error") => {
                    let message = event["message"].as_str().filter(|message| !message.is_empty());
                    let message = message.unwrap_or("Straydog failed");
                    return Err(match event["fatal"].as_bool() {
                        Some(true) => ApiError::server(message),
                        _ => ApiError::invalid(message),
                    });
                }
                _ => {}
            }
        }
    }

    fn end(&self, done: &Value) -> End {
        let finish_reason = if !self.reply.calls.is_empty() {
            "tool_calls"
        } else if done["finish_reason"] == "length" && !self.reply.stop.stopped {
            "length"
        } else {
            "stop"
        };
        let usage = &done["usage"];
        let (prompt, completion) = (count(&usage["prompt_tokens"]), count(&usage["completion_tokens"]));
        let usage = json!({
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": count(&usage["cached_tokens"])},
        });
        let timings = if done["timings"].is_object() { done["timings"].clone() } else { json!({}) };
        End { finish_reason, usage, timings }
    }

    /// The whole answer as one `chat.completion`.
    async fn collect(mut self, meta: Meta) -> Result<Value, ApiError> {
        let (mut content, mut reasoning) = (String::new(), String::new());
        loop {
            let (deltas, end) = match self.step().await? {
                Step::Deltas(deltas) => (deltas, None),
                Step::Progress(..) => continue,
                Step::Done(deltas, end) => (deltas, Some(end)),
            };
            for delta in deltas {
                match delta {
                    Delta::Reasoning(text) => reasoning += &text,
                    Delta::Content(text) => content += &text,
                    Delta::Calls(_) => {}
                }
            }
            let Some(end) = end else { continue };
            let calls = &self.reply.calls;
            // Content is null only beside tool calls, as OpenAI writes it.
            let content = if content.is_empty() && !calls.is_empty() { Value::Null } else { content.into() };
            let mut message = json!({"role": "assistant", "content": content});
            if !reasoning.is_empty() {
                message["reasoning_content"] = reasoning.into();
            }
            if !calls.is_empty() {
                message["tool_calls"] = calls.iter().map(|call| tool_call(call, false)).collect();
            }
            let mut answer = meta.object("chat.completion");
            let choice = json!({"index": 0, "message": message, "finish_reason": end.finish_reason});
            answer.insert("choices".into(), json!([choice]));
            answer.insert("usage".into(), end.usage);
            answer.insert("timings".into(), end.timings);
            return Ok(Value::Object(answer));
        }
    }
}

/// What every object of one answer shares.
struct Meta {
    id: String,
    created: u64,
    model: String,
}

impl Meta {
    fn object(&self, object: &str) -> Map<String, Value> {
        let mut fields = Map::new();
        fields.insert("id".into(), self.id.clone().into());
        fields.insert("object".into(), object.into());
        fields.insert("created".into(), self.created.into());
        fields.insert("model".into(), self.model.clone().into());
        fields
    }
}

/// An answer as SSE events.
struct Chunks {
    completion: Completion,
    meta: Meta,
    include_usage: bool,
    role_sent: bool,
    /// When progress last became a comment.
    commented: Option<Instant>,
}

impl Chunks {
    /// The events for one step, and whether they end the stream.
    fn events(&mut self, step: Result<Step, ApiError>) -> (Vec<Event>, bool) {
        match step {
            Ok(Step::Deltas(deltas)) => (deltas.iter().map(|delta| self.chunk(Some(delta), None)).collect(), false),
            Ok(Step::Progress(done, total)) => {
                if self.commented.is_some_and(|at| at.elapsed() < KEEP_ALIVE) {
                    return (Vec::new(), false);
                }
                self.commented = Some(Instant::now());
                (vec![Event::default().comment(format!("prefill {done}/{total}"))], false)
            }
            Ok(Step::Done(deltas, end)) => {
                let mut events: Vec<Event> = deltas.iter().map(|delta| self.chunk(Some(delta), None)).collect();
                events.push(self.chunk(None, Some(&end)));
                if self.include_usage {
                    let mut chunk = self.meta.object("chat.completion.chunk");
                    chunk.insert("choices".into(), json!([]));
                    chunk.insert("usage".into(), end.usage);
                    events.push(Event::default().data(Value::Object(chunk).to_string()));
                }
                events.push(Event::default().data("[DONE]"));
                (events, true)
            }
            Err(error) => (vec![Event::default().event("error").data(error.body().to_string())], true),
        }
    }

    /// A `chat.completion.chunk`; the first carries the role, the last the finish reason and
    /// the engine's timings.
    fn chunk(&mut self, delta: Option<&Delta>, end: Option<&End>) -> Event {
        let mut fields = Map::new();
        if !self.role_sent {
            self.role_sent = true;
            fields.insert("role".into(), "assistant".into());
            if delta.is_none() {
                fields.insert("content".into(), "".into());
            }
        }
        match delta {
            Some(Delta::Reasoning(text)) => fields.insert("reasoning_content".into(), text.as_str().into()),
            Some(Delta::Content(text)) => fields.insert("content".into(), text.as_str().into()),
            Some(Delta::Calls(calls)) => {
                fields.insert("tool_calls".into(), calls.iter().map(|call| tool_call(call, true)).collect())
            }
            None => None,
        };
        let mut chunk = self.meta.object("chat.completion.chunk");
        let finish_reason = end.map(|end| end.finish_reason);
        chunk.insert("choices".into(), json!([{"index": 0, "delta": fields, "finish_reason": finish_reason}]));
        // With usage requested, every chunk but the usage one says it has none (as OpenAI does).
        if self.include_usage {
            chunk.insert("usage".into(), Value::Null);
        }
        if let Some(end) = end {
            chunk.insert("timings".into(), end.timings.clone());
        }
        Event::default().data(Value::Object(chunk).to_string())
    }
}

/// A tool call as OpenAI writes it; a streamed one also carries its index.
fn tool_call(call: &ToolCall, streamed: bool) -> Value {
    let mut fields = Map::new();
    if streamed {
        fields.insert("index".into(), call.index.into());
    }
    fields.insert("id".into(), call.id.clone().into());
    fields.insert("type".into(), "function".into());
    fields.insert("function".into(), json!({"name": call.name, "arguments": call.arguments}));
    Value::Object(fields)
}

fn count(value: &Value) -> u64 {
    value.as_u64().unwrap_or(0)
}

fn random_hex(bytes: usize) -> String {
    let mut buffer = vec![0u8; bytes];
    getrandom::fill(&mut buffer).expect("the OS random source");
    buffer.iter().map(|byte| format!("{byte:02x}")).collect()
}
