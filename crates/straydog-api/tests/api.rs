//! The API over a scripted engine: request translation, answer shapes, errors, keys and
//! cancellation.

use std::collections::VecDeque;
use std::future::IntoFuture;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use straydog_api::{router, ApiConfig, Engine, EngineState, Generation, Markup, ReasoningEffort, SamplingDefaults, SubmitError};
use straydog_engine::pyjson;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Notify;
use tower::ServiceExt;

/// What one `generate` call answers.
enum Script {
    /// These events, then the end of the stream.
    Events(Vec<Value>),
    /// These events, then nothing until the stream is dropped.
    Hang(Vec<Value>),
    Refuse(SubmitError),
}

struct Mock {
    state: Mutex<EngineState>,
    scripts: Mutex<VecDeque<Script>>,
    requests: Mutex<Vec<pyjson::Value>>,
    interrupts: Arc<AtomicUsize>,
    submitted: Notify,
    /// Notified when a generation's events are dropped: it is cancelled.
    dropped: Arc<Notify>,
}

impl Mock {
    fn new(scripts: Vec<Script>) -> Arc<Mock> {
        Arc::new(Mock {
            state: Mutex::new(EngineState::Ready { context: 131072, speculative: 0, processing: false,
                                                   placement: json!({"resident": true}) }),
            scripts: Mutex::new(scripts.into()),
            requests: Mutex::new(Vec::new()),
            interrupts: Arc::new(AtomicUsize::new(0)),
            submitted: Notify::new(),
            dropped: Arc::new(Notify::new()),
        })
    }

    fn set_state(&self, state: EngineState) {
        *self.state.lock().unwrap() = state;
    }

    fn requests(&self) -> Vec<pyjson::Value> {
        self.requests.lock().unwrap().clone()
    }

    fn interrupts(&self) -> usize {
        self.interrupts.load(Ordering::SeqCst)
    }
}

struct DropGuard(Arc<Notify>);

impl Drop for DropGuard {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

impl Engine for Mock {
    fn state(&self) -> EngineState {
        self.state.lock().unwrap().clone()
    }

    fn generate(&self, request: pyjson::Value) -> BoxFuture<'static, Result<Generation, SubmitError>> {
        self.requests.lock().unwrap().push(request);
        self.submitted.notify_one();
        let script = self.scripts.lock().unwrap().pop_front().expect("a scripted generation");
        let interrupts = self.interrupts.clone();
        let guard = DropGuard(self.dropped.clone());
        Box::pin(async move {
            let (events, tail) = match script {
                Script::Events(events) => (events, stream::empty().boxed()),
                Script::Hang(events) => (events, stream::pending().boxed()),
                Script::Refuse(error) => return Err(error),
            };
            let events = stream::iter(events)
                .chain(tail)
                .map(move |event| {
                    let _ = &guard;
                    event
                })
                .boxed();
            let interrupt = Box::new(move || {
                interrupts.fetch_add(1, Ordering::SeqCst);
            });
            Ok(Generation { events, interrupt })
        })
    }
}

fn config() -> ApiConfig {
    ApiConfig {
        model_id: "Qwen3.8-Flash-Next-IQ3_S".into(),
        context: 262144,
        quantization: "IQ3_S".into(),
        api_key: None,
        sampling: SamplingDefaults::default(),
        seed: Some(0),
        reasoning_effort: ReasoningEffort::High,
        created: 1_760_000_000,
        markup: Markup::Qwen,
    }
}

fn app(mock: &Arc<Mock>) -> Router {
    router(mock.clone(), config())
}

fn delta(text: &str) -> Value {
    json!({"event": "delta", "content": text})
}

/// The worker's mark of the reasoning's end: it saw the template's `</think>` token.
fn reasoning_end() -> Value {
    json!({"event": "reasoning_end"})
}

fn progress(done: u64, total: u64) -> Value {
    json!({"event": "progress", "done": done, "total": total})
}

fn done(finish_reason: &str) -> Value {
    json!({"event": "done", "finish_reason": finish_reason,
           "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150, "cached_tokens": 100},
           "timings": {"prefill_seconds": 0.25, "decode_tokens_per_second": 97.5}})
}

fn failure(message: &str, fatal: bool) -> Value {
    json!({"event": "error", "message": message, "fatal": fatal})
}

fn read_file() -> Value {
    json!([{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object",
           "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}}}])
}

/// A chat completion request with one user message and `fields`.
fn ask(fields: Value) -> Value {
    let mut body = json!({"messages": [{"role": "user", "content": "Read main.rs"}]});
    for (key, value) in fields.as_object().unwrap() {
        body[key] = value.clone();
    }
    body
}

fn chat(body: impl Into<Body>) -> Request<Body> {
    Request::post("/v1/chat/completions").header("content-type", "application/json").body(body.into()).unwrap()
}

fn get(path: &str) -> Request<Body> {
    Request::get(path).body(Body::empty()).unwrap()
}

/// Status, content type and body of the answer.
async fn send(app: Router, request: Request<Body>) -> (StatusCode, String, String) {
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let kind = response.headers().get("content-type").map(|kind| kind.to_str().unwrap().to_string());
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, kind.unwrap_or_default(), String::from_utf8(body.to_vec()).unwrap())
}

/// A JSON answer: status and body.
async fn fetch(app: Router, request: Request<Body>) -> (StatusCode, Value) {
    let (status, kind, text) = send(app, request).await;
    assert_eq!(kind, "application/json", "{text}");
    (status, serde_json::from_str(&text).unwrap())
}

/// A non-streamed completion: status and JSON body.
async fn complete(mock: &Arc<Mock>, body: Value) -> (StatusCode, Value) {
    fetch(app(mock), chat(body.to_string())).await
}

#[derive(Debug, PartialEq)]
enum Sse {
    Chunk(Value),
    Done,
    Comment(String),
    Error(Value),
}

/// A streamed request: status and SSE events.
async fn stream(mock: &Arc<Mock>, mut body: Value) -> (StatusCode, Vec<Sse>) {
    body["stream"] = true.into();
    let (status, kind, text) = send(app(mock), chat(body.to_string())).await;
    assert_eq!(kind, "text/event-stream");
    (status, parse_sse(&text))
}

fn parse_sse(text: &str) -> Vec<Sse> {
    assert!(text.ends_with("\n\n"), "{text:?}");
    text.split_terminator("\n\n")
        .map(|block| {
            let (mut event, mut data, mut comment) = (None, None, None);
            for line in block.lines() {
                if let Some(name) = line.strip_prefix("event: ") {
                    event = Some(name);
                } else if let Some(text) = line.strip_prefix("data: ") {
                    data = Some(text);
                } else if let Some(text) = line.strip_prefix(": ") {
                    comment = Some(text);
                } else {
                    panic!("unexpected SSE line {line:?}");
                }
            }
            match (event, data, comment) {
                (Some("error"), Some(data), None) => Sse::Error(serde_json::from_str(data).unwrap()),
                (None, Some("[DONE]"), None) => Sse::Done,
                (None, Some(data), None) => Sse::Chunk(serde_json::from_str(data).unwrap()),
                (None, None, Some(comment)) => Sse::Comment(comment.to_string()),
                other => panic!("unexpected SSE event {other:?}"),
            }
        })
        .collect()
}

fn chunks(events: &[Sse]) -> Vec<&Value> {
    events.iter().filter_map(|event| if let Sse::Chunk(chunk) = event { Some(chunk) } else { None }).collect()
}

fn deltas(events: &[Sse]) -> Vec<Value> {
    let choices = chunks(events).into_iter().filter_map(|chunk| chunk["choices"].get(0));
    choices.map(|choice| choice["delta"].clone()).collect()
}

/// A tool call's id is random: check its form, then fix it for comparison.
fn fix_ids(calls: &mut Value) {
    for call in calls.as_array_mut().unwrap() {
        let id = call["id"].as_str().unwrap();
        assert!(id.starts_with("call_") && id.len() == 29, "{id}");
        call["id"] = "call".into();
    }
}

/// The status and OpenAI error object of a refusal.
fn refused(code: u16, kind: &str, message: &str) -> (StatusCode, Value) {
    (StatusCode::from_u16(code).unwrap(), json!({"error": {"code": code, "message": message, "type": kind}}))
}

fn invalid(message: &str) -> (StatusCode, Value) {
    refused(400, "invalid_request_error", message)
}

fn field<'a>(request: &'a pyjson::Value, key: &str) -> &'a pyjson::Value {
    request.get(key).unwrap_or_else(|| panic!("no {key} in the worker request"))
}

#[tokio::test]
async fn streams_reasoning_answer_and_tool_calls() {
    let mock = Mock::new(vec![Script::Events(vec![
        progress(512, 1024),
        progress(1024, 1024),
        delta("Let me check"),
        delta(" the file.\n</th"),
        delta("ink>\n\nReading it"),
        delta(" now.\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n"),
        delta("<parameter=limit>\n50\n</parameter>\n</function>\n</tool_call>"),
        done("stop"),
    ])]);
    let (status, events) = stream(&mock, ask(json!({"tools": read_file()}))).await;
    assert_eq!(status, StatusCode::OK);
    // One keep-alive comment: the second progress event came within a second.
    assert_eq!(events[0], Sse::Comment("prefill 512/1024".into()));
    assert!(!events[1..].iter().any(|event| matches!(event, Sse::Comment(_))));
    assert_eq!(events.last(), Some(&Sse::Done));

    let mut deltas = deltas(&events);
    fix_ids(&mut deltas[4]["tool_calls"]);
    let call = json!({"index": 0, "id": "call", "type": "function",
                      "function": {"name": "read_file", "arguments": r#"{"path": "src/main.rs", "limit": 50}"#}});
    assert_eq!(deltas, [
        json!({"role": "assistant", "reasoning_content": "Let me check"}),
        json!({"reasoning_content": " the file.\n"}),
        json!({"content": "Reading it"}),
        json!({"content": " now.\n"}),
        json!({"tool_calls": [call]}),
        json!({}),
    ]);
    let chunks = chunks(&events);
    let id = chunks[0]["id"].as_str().unwrap();
    assert!(id.starts_with("chatcmpl-") && id.len() == 41, "{id}");
    for (index, chunk) in chunks.iter().enumerate() {
        let last = index == chunks.len() - 1;
        assert_eq!(chunk["id"], id);
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["model"], "Qwen3.8-Flash-Next-IQ3_S");
        assert!(chunk["created"].as_u64().unwrap() > 1_760_000_000);
        assert_eq!(chunk["choices"][0]["index"], 0);
        assert_eq!(chunk["choices"][0]["finish_reason"], if last { json!("tool_calls") } else { Value::Null });
        assert!(chunk.get("usage").is_none());
        assert_eq!(chunk.get("timings").is_some(), last);
    }
    assert_eq!(chunks.last().unwrap()["timings"], json!({"prefill_seconds": 0.25, "decode_tokens_per_second": 97.5}));

    let requests = mock.requests();
    let request = &requests[0];
    assert_eq!(field(request, "thinking"), &pyjson::Value::Bool(true));
    assert_eq!(field(request, "reasoning_effort").as_str(), Some("xhigh"));
    let tools = pyjson::parse(&read_file().to_string()).unwrap();
    assert_eq!(pyjson::dumps(field(request, "tools")), pyjson::dumps(&tools));
    assert_eq!(mock.interrupts(), 0);
}

#[tokio::test]
async fn quoted_tags_stay_text() {
    let mock = Mock::new(vec![Script::Events(vec![
        delta("Templates close with `</think>` and\n"),
        delta("call with <tool_call> mid-line.\n</think>\n\n"),
        delta("Write <tool_call> like this:\n"),
        delta("<tool_call> is how a call starts."),
        done("stop"),
    ])]);
    let (status, body) = complete(&mock, ask(json!({"tools": read_file()}))).await;
    assert_eq!(status, StatusCode::OK);
    let choice = &body["choices"][0];
    assert_eq!(choice["message"], json!({
        "role": "assistant",
        "content": "Write <tool_call> like this:\n<tool_call> is how a call starts.",
        "reasoning_content": "Templates close with `</think>` and\ncall with <tool_call> mid-line.\n",
    }));
    assert_eq!(choice["finish_reason"], "stop");
}

#[tokio::test]
async fn stop_strings_end_the_answer_across_deltas() {
    let mock = Mock::new(vec![Script::Events(vec![
        delta("Hello E"),
        delta("N"),
        delta("D tail"),
        delta(" more"),
        done("cancelled"),
    ])]);
    let (status, events) = stream(&mock, ask(json!({"stop": ["</x>", "END"], "reasoning_effort": "none"}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deltas(&events), [json!({"role": "assistant", "content": "Hello "}), json!({})]);
    assert_eq!(chunks(&events).last().unwrap()["choices"][0]["finish_reason"], "stop");
    assert_eq!(mock.interrupts(), 1);

    // A held tail that never completes a stop string is answer text; the length limit still shows.
    let mock = Mock::new(vec![Script::Events(vec![delta("Hello "), delta("E"), done("length")])]);
    let (_, body) = complete(&mock, ask(json!({"stop": "END", "reasoning_effort": "none"}))).await;
    assert_eq!(body["choices"][0]["message"]["content"], "Hello E");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(mock.interrupts(), 0);

    // A stop that ends the text at the length limit is a stop.
    let mock = Mock::new(vec![Script::Events(vec![delta("Hello END"), done("length")])]);
    let (_, body) = complete(&mock, ask(json!({"stop": "END", "reasoning_effort": "none"}))).await;
    assert_eq!(body["choices"][0]["message"]["content"], "Hello ");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(mock.interrupts(), 1);
}

#[tokio::test]
async fn usage_chunk_when_requested() {
    let usage = json!({"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                       "prompt_tokens_details": {"cached_tokens": 100}});
    let mock = Mock::new(vec![Script::Events(vec![delta("Hi"), done("stop")])]);
    let body = ask(json!({"reasoning_effort": "none", "stream_options": {"include_usage": true}}));
    let (_, events) = stream(&mock, body).await;
    let [Sse::Chunk(first), Sse::Chunk(last), Sse::Chunk(totals), Sse::Done] = &events[..] else {
        panic!("{events:?}")
    };
    assert_eq!(first["choices"][0]["delta"], json!({"role": "assistant", "content": "Hi"}));
    assert_eq!((&first["usage"], &last["usage"]), (&Value::Null, &Value::Null));
    assert!(first.get("usage").is_some() && last.get("usage").is_some());
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(totals["choices"], json!([]));
    assert_eq!(totals["usage"], usage);
    assert_eq!(totals["id"], first["id"]);

    // An empty answer still says who answered.
    let mock = Mock::new(vec![Script::Events(vec![done("stop")])]);
    let (_, events) = stream(&mock, ask(json!({}))).await;
    assert_eq!(deltas(&events), [json!({"role": "assistant", "content": ""})]);
}

#[tokio::test]
async fn non_streaming_answers() {
    let events = vec![progress(1, 2), delta("Hmm.\n</think>\n\n"), delta("Hello"), done("length")];
    let mock = Mock::new(vec![Script::Events(events)]);
    let (status, body) = complete(&mock, ask(json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    let id = body["id"].as_str().unwrap();
    assert!(id.starts_with("chatcmpl-"));
    assert_eq!(body, json!({
        "id": id,
        "object": "chat.completion",
        "created": body["created"],
        "model": "Qwen3.8-Flash-Next-IQ3_S",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello", "reasoning_content": "Hmm.\n"},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                  "prompt_tokens_details": {"cached_tokens": 100}},
        "timings": {"prefill_seconds": 0.25, "decode_tokens_per_second": 97.5},
    }));

    // Tool calls alone: content is null.
    let markup = "<tool_call>\n<function=read_file>\n<parameter=path>\na.rs\n</parameter>\n</function>\n</tool_call>";
    let mock = Mock::new(vec![Script::Events(vec![delta(markup), done("stop")])]);
    let (_, body) = complete(&mock, ask(json!({"tools": read_file(), "reasoning_effort": "none"}))).await;
    let mut message = body["choices"][0]["message"].clone();
    fix_ids(&mut message["tool_calls"]);
    assert_eq!(message, json!({"role": "assistant", "content": null, "tool_calls": [
        {"id": "call", "type": "function", "function": {"name": "read_file", "arguments": r#"{"path": "a.rs"}"#}}]}));
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");

    // A close the model left out is read; the same text cut by the token limit is not a call.
    let open = "<tool_call>\n<function=read_file>\n<parameter=path>\na.rs";
    let mock = Mock::new(vec![Script::Events(vec![delta(open), done("stop")])]);
    let (_, body) = complete(&mock, ask(json!({"tools": read_file(), "reasoning_effort": "none"}))).await;
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], r#"{"path": "a.rs"}"#);
    let mock = Mock::new(vec![Script::Events(vec![delta(open), done("length")])]);
    let (_, body) = complete(&mock, ask(json!({"tools": read_file(), "reasoning_effort": "none"}))).await;
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["choices"][0]["message"]["content"], open);

    // With tool_choice none the tools leave the prompt, and markup stays text.
    let mock = Mock::new(vec![Script::Events(vec![delta(markup), done("stop")])]);
    let fields = json!({"tools": read_file(), "tool_choice": "none", "reasoning_effort": "none"});
    let (_, body) = complete(&mock, ask(fields)).await;
    assert_eq!(body["choices"][0]["message"], json!({"role": "assistant", "content": markup}));
    assert_eq!(field(&mock.requests()[0], "tools"), &pyjson::Value::Null);
}

#[tokio::test]
async fn messages_and_tools_reach_the_engine_as_python_reads_them() {
    // Expected output from Python: json.dumps(json.loads(body)[...], ensure_ascii=False) after the
    // translation (unknown and null fields dropped, developer as system, text parts joined).
    let body = r#"{"model": "anything", "temperature": 0.5E0,
     "messages": [
      {"content": "Be brief.", "role": "developer", "extra": 1},
      {"role": "user", "content": [{"type": "text", "text": "caf\u00e9 "}, {"type": "text", "text": "\ud83d\ude00 \"q\"\n\u0001"}], "name": null},
      {"role": "assistant", "content": null, "reasoning_content": "Think 1E-3", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"b\": 1e-3, \"a\": 1}"}}], "refusal": null},
      {"role": "tool", "tool_call_id": "call_1", "content": "{\"z\": 0.10}"},
      {"role": "user", "content": "again"}
     ],
     "tools": [{"type": "function", "function": {"name": "f", "description": "\u00e9\u3042\t\/", "parameters": {"type": "object", "properties": {"b": {"type": "number", "default": 1e-3, "dup": 1, "maximum": 1E+2}, "a": {"type": "integer", "minimum": -0.0, "big": 12345678901234567890123, "small": 1e400, "tiny": 5e-324, "x": 1.5e16, "dup": 2, "b": 3, "dup": [true, false, null]}}}}}]
    }"#;
    let messages = r#"[{"content": "Be brief.", "role": "system"}, {"role": "user", "content": "café 😀 \"q\"\n\u0001"}, {"role": "assistant", "reasoning_content": "Think 1E-3", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"b\": 1e-3, \"a\": 1}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "{\"z\": 0.10}"}, {"role": "user", "content": "again"}]"#;
    let tools = r#"[{"type": "function", "function": {"name": "f", "description": "éあ\t/", "parameters": {"type": "object", "properties": {"b": {"type": "number", "default": 0.001, "dup": 1, "maximum": 100.0}, "a": {"type": "integer", "minimum": -0.0, "big": 12345678901234567890123, "small": Infinity, "tiny": 5e-324, "x": 1.5e+16, "dup": [true, false, null], "b": 3}}}}}]"#;
    let mock = Mock::new(vec![Script::Events(vec![done("stop")])]);
    let (status, _, _) = send(app(&mock), chat(body)).await;
    assert_eq!(status, StatusCode::OK);
    let expected = format!(
        r#"{{"messages": {messages}, "tools": {tools}, "max_tokens": null, "temperature": 0.5, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "presence_penalty": 0.0, "seed": 0, "thinking": true, "reasoning_effort": "xhigh", "checkpoint": true, "constraint": null}}"#
    );
    assert_eq!(pyjson::dumps(&mock.requests()[0]), expected);
}

#[tokio::test]
async fn response_format_and_tool_choice_reach_the_worker_as_a_constraint() {
    let dsml_call = r#"\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\""#;
    let cases = [
        (json!({"response_format": {"type": "json_object"}}), Markup::Qwen, r#"{"alternatives": [{"json": true}]}"#.to_string()),
        (json!({"response_format": {"type": "text"}}), Markup::Qwen, "null".to_string()),
        (json!({"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "object"}}}}),
         Markup::Qwen, r#"{"alternatives": [{"json": {"type": "object"}}]}"#.to_string()),
        (json!({"response_format": {"type": "json_object"}, "tools": read_file()}), Markup::Qwen,
         r#"{"alternatives": [{"json": true}, {"prefix": "<tool_call>\n<function=", "whitespace": 4}]}"#.to_string()),
        (json!({"response_format": {"type": "json_object"}, "tools": read_file(), "tool_choice": "none"}), Markup::Qwen,
         r#"{"alternatives": [{"json": true}]}"#.to_string()),
        (json!({"tools": read_file(), "tool_choice": "required"}), Markup::Dsml,
         format!(r#"{{"alternatives": [{{"prefix": "{dsml_call}", "whitespace": 0}}]}}"#)),
        (json!({"tools": read_file(), "tool_choice": {"type": "function", "function": {"name": "read_file"}}}), Markup::Qwen,
         r#"{"alternatives": [{"prefix": "<tool_call>\n<function=read_file>\n", "whitespace": 4, "until": "</function>", "close": "\n</tool_call>"}]}"#.to_string()),
    ];
    for (fields, markup, expected) in cases {
        let mock = Mock::new(vec![Script::Events(vec![done("stop")])]);
        let (status, _, _) = send(router(mock.clone(), ApiConfig { markup, ..config() }), chat(ask(fields.clone()).to_string())).await;
        assert_eq!(status, StatusCode::OK, "{fields}");
        assert_eq!(pyjson::dumps(field(&mock.requests()[0], "constraint")), expected, "{fields}");
    }
}

/// A request that asks for a suppression hands it on, after everything else; one that does not (null is none) leaves the worker's
/// request as it was, field for field.
#[tokio::test]
async fn suppress_reaches_the_worker_only_when_given() {
    let sent = |fields: serde_json::Value| async move {
        let mock = Mock::new(vec![Script::Events(vec![done("stop")])]);
        let (status, _, _) = send(app(&mock), chat(ask(fields).to_string())).await;
        assert_eq!(status, StatusCode::OK);
        mock.requests()[0].clone()
    };
    let asking = sent(json!({"seed": 1, "suppress": "not_cp932_han"})).await;
    assert_eq!(field(&asking, "suppress"), &pyjson::Value::Str("not_cp932_han".into()));
    let (absent, null) = (sent(json!({"seed": 1})).await, sent(json!({"seed": 1, "suppress": null})).await);
    assert_eq!(pyjson::dumps(&absent), pyjson::dumps(&null));
    assert!(!pyjson::dumps(&absent).contains("suppress"));
    assert_eq!(pyjson::dumps(&asking), pyjson::dumps(&absent).strip_suffix('}').unwrap().to_string() + r#", "suppress": "not_cp932_han"}"#);
}

#[tokio::test]
async fn settings_reach_the_worker_request() {
    let cases = [
        (json!({}), ReasoningEffort::High, true, "xhigh"),
        (json!({}), ReasoningEffort::None, false, "medium"),
        (json!({"reasoning_effort": ""}), ReasoningEffort::Low, true, "low"),
        (json!({"reasoning_effort": "none"}), ReasoningEffort::High, false, "medium"),
        (json!({"reasoning_effort": "minimal"}), ReasoningEffort::High, true, "low"),
        (json!({"reasoning_effort": "low"}), ReasoningEffort::High, true, "low"),
        (json!({"reasoning_effort": "medium"}), ReasoningEffort::High, true, "medium"),
        (json!({"reasoning_effort": "high"}), ReasoningEffort::None, true, "xhigh"),
        (json!({"reasoning_effort": "xhigh"}), ReasoningEffort::None, true, "xhigh"),
        (json!({"chat_template_kwargs": {"enable_thinking": false}}), ReasoningEffort::High, false, "xhigh"),
        (json!({"reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "low"}}),
         ReasoningEffort::High, true, "low"),
    ];
    for (fields, default, thinking, level) in cases {
        let mock = Mock::new(vec![Script::Events(vec![done("stop")])]);
        let mut settings = config();
        settings.reasoning_effort = default;
        let (status, _, _) = send(router(mock.clone(), settings), chat(ask(fields.clone()).to_string())).await;
        assert_eq!(status, StatusCode::OK, "{fields}");
        let request = &mock.requests()[0];
        assert_eq!(field(request, "thinking"), &pyjson::Value::Bool(thinking), "{fields}");
        assert_eq!(field(request, "reasoning_effort").as_str(), Some(level), "{fields}");
    }

    let card = r#""top_k": 20, "top_p": 0.95, "min_p": 0.0, "presence_penalty": 0.0"#;
    let cases = [
        (json!({}), format!(r#"null, "temperature": 0.7, {card}, "seed": 42, "#), true),
        (json!({"max_tokens": 5, "temperature": 1, "seed": 7}), format!(r#"5, "temperature": 1.0, {card}, "seed": 7, "#), true),
        (json!({"max_tokens": 5, "max_completion_tokens": 9, "straydog": {"checkpoint": false}}),
         format!(r#"9, "temperature": 0.7, {card}, "seed": 42, "#), false),
        (json!({"max_completion_tokens": null, "max_tokens": 3, "straydog": {}}),
         format!(r#"3, "temperature": 0.7, {card}, "seed": 42, "#), true),
    ];
    for (fields, numbers, checkpoint) in cases {
        let mock = Mock::new(vec![Script::Events(vec![done("stop")])]);
        let mut settings = config();
        (settings.sampling.temperature, settings.seed) = (Some(0.7), Some(42));
        let (status, _, _) = send(router(mock.clone(), settings), chat(ask(fields.clone()).to_string())).await;
        assert_eq!(status, StatusCode::OK, "{fields}");
        let request = pyjson::dumps(&mock.requests()[0]);
        assert!(request.contains(&format!(r#""max_tokens": {numbers}"#)), "{request}");
        assert!(request.ends_with(&format!(r#""checkpoint": {checkpoint}, "constraint": null}}"#)), "{request}");
    }
}

#[tokio::test]
async fn sampling_follows_the_request_then_the_server_then_the_model_card_for_the_mode() {
    let sampling = |request: &pyjson::Value| -> Vec<String> {
        ["temperature", "top_k", "top_p", "min_p", "presence_penalty"].iter()
            .map(|key| pyjson::dumps(field(request, key))).collect()
    };
    let cases = [
        // The model card: thinking, and not.
        (json!({}), SamplingDefaults::default(), ["1.0", "20", "0.95", "0.0", "0.0"]),
        (json!({"reasoning_effort": "none"}), SamplingDefaults::default(), ["0.7", "20", "0.8", "0.0", "1.5"]),
        (json!({"chat_template_kwargs": {"enable_thinking": false}}), SamplingDefaults::default(),
         ["0.7", "20", "0.8", "0.0", "1.5"]),
        // The server's settings, then the request's.
        (json!({"reasoning_effort": "none"}), SamplingDefaults { top_k: Some(40), presence_penalty: Some(0.0), ..Default::default() },
         ["0.7", "40", "0.8", "0.0", "0.0"]),
        (json!({"temperature": 0, "top_k": -1, "top_p": 1, "min_p": 0.05, "presence_penalty": -0.5}),
         SamplingDefaults { top_k: Some(40), ..Default::default() }, ["0.0", "0", "1.0", "0.05", "-0.5"]),
    ];
    for (fields, defaults, expected) in cases {
        let mock = Mock::new(vec![Script::Events(vec![done("stop")])]);
        let mut settings = config();
        settings.sampling = defaults;
        let (status, _, _) = send(router(mock.clone(), settings), chat(ask(fields.clone()).to_string())).await;
        assert_eq!(status, StatusCode::OK, "{fields}");
        assert_eq!(sampling(&mock.requests()[0]), expected, "{fields}");
    }
    // Without a seed anywhere, each request gets a fresh one.
    let mock = Mock::new(vec![Script::Events(vec![done("stop")]), Script::Events(vec![done("stop")])]);
    let mut settings = config();
    settings.seed = None;
    for _ in 0..2 {
        let (status, _, _) = send(router(mock.clone(), settings.clone()), chat(ask(json!({})).to_string())).await;
        assert_eq!(status, StatusCode::OK);
    }
    let seeds: Vec<String> = mock.requests().iter().map(|request| pyjson::dumps(field(request, "seed"))).collect();
    assert_ne!(seeds[0], seeds[1]);
}

#[tokio::test]
async fn neutral_values_are_accepted() {
    let mock = Mock::new(vec![Script::Events(vec![done("stop")])]);
    let fields = json!({"n": 1, "top_p": 1.0, "top_k": 0, "min_p": 0, "typical_p": 1, "presence_penalty": 0.0,
                        "frequency_penalty": 0, "repeat_penalty": 1, "repetition_penalty": 1.0, "logit_bias": {},
                        "logprobs": false, "top_logprobs": 0, "parallel_tool_calls": true,
                        "response_format": {"type": "text"}, "modalities": ["text"], "functions": null,
                        "user": "someone", "metadata": {"a": "b"}, "store": false, "tools": [], "tool_choice": "auto",
                        "stop": null, "stream": false, "stream_options": null, "chat_template_kwargs": {}});
    let (status, body) = complete(&mock, ask(fields)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn invalid_requests_are_refused() {
    let seventeen: Vec<String> = (0..17).map(|stop| stop.to_string()).collect();
    let role = "each message needs a role: system, developer, user, assistant or tool";
    let content = "message content must be a string or a list of text parts";
    let effort = "reasoning_effort must be none, minimal, low, medium, high or xhigh";
    let kwargs = "chat_template_kwargs supports enable_thinking and reasoning_effort";
    let cases = [
        (json!({}), "messages must be a non-empty list"),
        (json!({"messages": []}), "messages must be a non-empty list"),
        (json!({"messages": "hi"}), "messages must be a non-empty list"),
        (json!({"messages": ["hi"]}), role),
        (json!({"messages": [{"content": "hi"}]}), role),
        (json!({"messages": [{"role": "robot", "content": "hi"}]}), role),
        (json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}),
         "this model reads no images"),
        (json!({"messages": [{"role": "user", "content": [{"type": "text"}]}]}), "only text and image content is supported"),
        (json!({"messages": [{"role": "user", "content": 5}]}), content),
        (ask(json!({"n": 2})), "n is not supported (only 1)"),
        (ask(json!({"top_p": 0})), "top_p must be in (0, 1]"),
        (ask(json!({"top_p": true})), "top_p must be in (0, 1]"),
        (ask(json!({"top_k": 1.5})), "top_k must be -1, 0 or a positive integer"),
        (ask(json!({"top_k": -2})), "top_k must be -1, 0 or a positive integer"),
        (ask(json!({"min_p": 1})), "min_p must be in [0, 1)"),
        (ask(json!({"typical_p": 0.9})), "typical_p is not supported (only 1)"),
        (ask(json!({"presence_penalty": 2.5})), "presence_penalty must be between -2 and 2"),
        (ask(json!({"frequency_penalty": 1})), "frequency_penalty is not supported (only 0)"),
        (ask(json!({"repeat_penalty": 1.1})), "repeat_penalty is not supported (only 1)"),
        (ask(json!({"repetition_penalty": 1.05})), "repetition_penalty is not supported (only 1)"),
        (ask(json!({"logit_bias": {"42": 5}})), "logit_bias is not supported (only {})"),
        (ask(json!({"suppress": "simplified"})), "suppress must be not_cp932_han"),
        (ask(json!({"suppress": ""})), "suppress must be not_cp932_han"),
        (ask(json!({"suppress": ["not_cp932_han"]})), "suppress must be not_cp932_han"),
        (ask(json!({"suppress": true})), "suppress must be not_cp932_han"),
        (ask(json!({"logprobs": true})), "logprobs is not supported (only false)"),
        (ask(json!({"top_logprobs": 3})), "top_logprobs is not supported (only 0)"),
        (ask(json!({"parallel_tool_calls": false})), "parallel_tool_calls is not supported (only true)"),
        (ask(json!({"response_format": {"type": "xml"}})), "response_format type must be text, json_object or json_schema"),
        (ask(json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}})),
         "response_format json_schema needs json_schema.schema"),
        (ask(json!({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "date"}}}})),
         "response_format json_schema: JSON schema type date is not supported"),
        (ask(json!({"modalities": ["text", "audio"]})), r#"modalities is not supported (only ["text"])"#),
        (ask(json!({"functions": [{"name": "f"}]})), "functions is not supported"),
        (ask(json!({"function_call": "auto"})), "function_call is not supported"),
        (ask(json!({"audio": {"voice": "alloy"}})), "audio is not supported"),
        (ask(json!({"prediction": {"type": "content", "content": "x"}})), "prediction is not supported"),
        (ask(json!({"tools": {"type": "function"}})), "tools must be a list of tool objects"),
        (ask(json!({"tools": ["read_file"]})), "tools must be a list of tool objects"),
        (ask(json!({"tool_choice": "required"})), "tool_choice required needs tools"),
        (ask(json!({"tool_choice": {"type": "function", "function": {"name": "read_file"}}})),
         "tool_choice names read_file, which is not among the tools"),
        (ask(json!({"tool_choice": "always"})),
         r#"tool_choice must be auto, none, required or {"type": "function", "function": {"name": ...}}"#),
        (ask(json!({"max_tokens": 0})), "max_tokens must be a positive integer"),
        (ask(json!({"max_tokens": 1.5})), "max_tokens must be a positive integer"),
        (ask(json!({"max_tokens": true})), "max_tokens must be a positive integer"),
        (ask(json!({"max_tokens": 5, "max_completion_tokens": -1})),
         "max_completion_tokens must be a positive integer"),
        (ask(json!({"temperature": 2.5})), "temperature must be between 0 and 2"),
        (ask(json!({"temperature": -0.1})), "temperature must be between 0 and 2"),
        (ask(json!({"temperature": "0.5"})), "temperature must be between 0 and 2"),
        (ask(json!({"temperature": false})), "temperature must be between 0 and 2"),
        (ask(json!({"seed": -1})), "seed must be a nonnegative 64-bit integer"),
        (ask(json!({"seed": 1.5})), "seed must be a nonnegative 64-bit integer"),
        (ask(json!({"reasoning_effort": "maximum"})), effort),
        (ask(json!({"reasoning_effort": 3})), effort),
        (ask(json!({"chat_template_kwargs": {"enable_thinking": "yes"}})),
         "chat_template_kwargs.enable_thinking must be a boolean"),
        (ask(json!({"chat_template_kwargs": {"reasoning_effort": "high"}})),
         "chat_template_kwargs.reasoning_effort must be low, medium or xhigh"),
        (ask(json!({"chat_template_kwargs": {"add_vision_id": true}})), kwargs),
        (ask(json!({"chat_template_kwargs": "thinking"})), kwargs),
        (ask(json!({"stop": 5})), "stop must be a string or a list of at most 16 strings"),
        (ask(json!({"stop": ["a", 1]})), "stop must be a string or a list of at most 16 strings"),
        (ask(json!({"stop": seventeen})), "stop must be a string or a list of at most 16 strings"),
        (ask(json!({"straydog": {"checkpoint": "no"}})), "straydog.checkpoint must be a boolean"),
        (ask(json!({"straydog": {"cache": false}})), "the straydog object supports checkpoint"),
        (ask(json!({"straydog": true})), "the straydog object supports checkpoint"),
        (ask(json!({"stream": "yes"})), "stream must be a boolean"),
        (ask(json!({"stream_options": true})), "stream_options must be an object"),
        (ask(json!({"stream_options": {"include_usage": 1}})), "stream_options.include_usage must be a boolean"),
    ];
    let mock = Mock::new(Vec::new());
    for (body, message) in cases {
        assert_eq!(complete(&mock, body.clone()).await, invalid(message), "{body}");
    }

    let raw: [(&[u8], &str); 5] = [
        (b"{\"messages\": ", "the request body is not valid JSON: expecting value at byte 13"),
        (b"[1]", "the request body must be a JSON object"),
        (b"{\"messages\": \"\xff\"}", "the request body is not UTF-8"),
        (br#"{"messages": [{"role": "user", "content": "\ud800"}]}"#,
         "the request body is not valid JSON: lone surrogate at byte 49"),
        (br#"{"messages": [{"role": "user", "content": "hi"}], "seed": 18446744073709551616}"#,
         "seed must be a nonnegative 64-bit integer"),
    ];
    for (body, message) in raw {
        assert_eq!(fetch(app(&mock), chat(body.to_vec())).await, invalid(message));
    }
    assert!(mock.requests().is_empty());

    // The engine's own refusals: a submit error, or an error event before any output.
    let mock = Mock::new(vec![
        Script::Refuse(SubmitError::Invalid("the prompt exceeds the context".into())),
        Script::Events(vec![failure("the prompt exceeds the context", false)]),
        Script::Events(vec![failure("No user query found in messages.", false)]),
    ]);
    assert_eq!(complete(&mock, ask(json!({}))).await, invalid("the prompt exceeds the context"));
    assert_eq!(complete(&mock, ask(json!({}))).await, invalid("the prompt exceeds the context"));
    let streamed = ask(json!({"stream": true}));
    assert_eq!(complete(&mock, streamed).await, invalid("No user query found in messages."));
}

#[tokio::test]
async fn oversized_bodies_are_refused() {
    let mock = Mock::new(Vec::new());
    let large = format!(r#"{{"messages": "{}"}}"#, "a".repeat(64 * 1024 * 1024));
    let too_large = refused(413, "invalid_request_error", "request body too large");
    assert_eq!(fetch(app(&mock), chat(large)).await, too_large);
    let mut declared = chat("{}");
    declared.headers_mut().insert("content-length", "68000000".parse().unwrap());
    assert_eq!(fetch(app(&mock), declared).await, too_large);
}

#[tokio::test]
async fn the_key_guards_everything_but_health() {
    let mock = Mock::new(vec![Script::Events(vec![done("stop")])]);
    let mut settings = config();
    settings.api_key = Some("sk-local".into());
    let app = router(mock.clone(), settings);
    let with_key = |mut request: Request<Body>, authorization: &str| {
        request.headers_mut().insert("authorization", authorization.parse().unwrap());
        request
    };
    let unauthorized = refused(401, "authentication_error", "invalid API key");
    assert_eq!(fetch(app.clone(), get("/health")).await.0, StatusCode::OK);
    for path in ["/v1/models", "/props", "/slots"] {
        assert_eq!(fetch(app.clone(), get(path)).await, unauthorized, "{path}");
    }
    for authorization in ["Bearer sk-wrong", "Bearer sk-loca", "Basic sk-local", "sk-local", "Bearer"] {
        let request = with_key(get("/v1/models"), authorization);
        assert_eq!(fetch(app.clone(), request).await, unauthorized, "{authorization}");
    }
    assert_eq!(fetch(app.clone(), with_key(get("/v1/models"), "bearer  sk-local ")).await.0, StatusCode::OK);
    assert_eq!(fetch(app.clone(), chat(ask(json!({})).to_string())).await, unauthorized);
    assert!(mock.requests().is_empty());
    let request = with_key(chat(ask(json!({})).to_string()), "Bearer sk-local");
    assert_eq!(fetch(app.clone(), request).await.0, StatusCode::OK);

    // Without a key, nothing is asked.
    assert_eq!(fetch(router(mock, config()), get("/v1/models")).await.0, StatusCode::OK);
}

#[tokio::test]
async fn health_and_models_follow_the_engine() {
    let mock = Mock::new(Vec::new());
    let progress = json!({"stage": "weights", "done": 3, "total": 10});
    mock.set_state(EngineState::Loading { progress: progress.clone() });
    let mut loading = refused(503, "unavailable_error", "Loading model");
    loading.1["status"] = "loading".into();
    loading.1["progress"] = progress;
    assert_eq!(fetch(app(&mock), get("/health")).await, loading);
    let (_, models) = fetch(app(&mock), get("/v1/models")).await;
    assert_eq!(models, json!({"object": "list", "data": [{
        "id": "Qwen3.8-Flash-Next-IQ3_S", "object": "model", "created": 1_760_000_000, "owned_by": "straydog",
        "meta": {"n_ctx": 262144, "quantization": "IQ3_S", "ready": false, "vision": false, "suppress": ["not_cp932_han"]}}]}));
    for path in ["/props", "/slots"] {
        assert_eq!(fetch(app(&mock), get(path)).await, refused(503, "unavailable_error", "Loading model"), "{path}");
    }

    mock.set_state(EngineState::Ready { context: 131072, speculative: 3, processing: true,
                                        placement: json!({"resident": true, "host_kv": false}) });
    assert_eq!(fetch(app(&mock), get("/health")).await, (StatusCode::OK, json!({"status": "ok"})));
    let (_, models) = fetch(app(&mock), get("/v1/models")).await;
    assert_eq!(models["data"][0]["meta"], json!({"n_ctx": 131072, "quantization": "IQ3_S", "ready": true, "vision": false, "suppress": ["not_cp932_han"],
                                                 "placement": {"resident": true, "host_kv": false}}));
    // llama-server's endpoints, where its clients read the context.
    let slot = json!({"id": 0, "n_ctx": 131072, "speculative": true, "is_processing": true,
                      "params": {"n_predict": -1, "seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                                 "min_p": 0.0, "presence_penalty": 0.0, "speculative.n_max": 3}});
    let (status, props) = fetch(app(&mock), get("/props")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(props, json!({"default_generation_settings": slot, "total_slots": 1,
                             "model_alias": "Qwen3.8-Flash-Next-IQ3_S", "modalities": {"vision": false, "audio": false},
                             "suppress": ["not_cp932_han"],
                             "build_info": concat!("straydog-", env!("CARGO_PKG_VERSION")), "is_sleeping": false}));
    assert_eq!(fetch(app(&mock), get("/slots")).await, (StatusCode::OK, json!([slot])));

    mock.set_state(EngineState::Failed("the GPU stopped responding".into()));
    let mut failed = refused(500, "server_error", "the GPU stopped responding");
    failed.1["status"] = "error".into();
    assert_eq!(fetch(app(&mock), get("/health")).await, failed);

    assert_eq!(fetch(app(&mock), get("/v1/nothing")).await, refused(404, "not_found_error", "no such route"));
    let wrong_method = refused(405, "invalid_request_error", "method not allowed");
    assert_eq!(fetch(app(&mock), get("/v1/chat/completions")).await, wrong_method);
}

#[tokio::test]
async fn loading_and_busy_engines_answer_503() {
    let mock = Mock::new(vec![
        Script::Refuse(SubmitError::Busy("Straydog generation queue is full".into())),
        Script::Refuse(SubmitError::Unavailable("Straydog is restarting".into())),
    ]);
    mock.set_state(EngineState::Loading { progress: json!({}) });
    assert_eq!(complete(&mock, ask(json!({}))).await, refused(503, "unavailable_error", "Loading model"));
    mock.set_state(EngineState::Failed("the worker exited".into()));
    assert_eq!(complete(&mock, ask(json!({}))).await, refused(503, "unavailable_error", "the worker exited"));
    assert!(mock.requests().is_empty());

    mock.set_state(EngineState::Ready { context: 131072, speculative: 0, processing: false, placement: json!({}) });
    let busy = refused(503, "server_busy", "Straydog generation queue is full");
    assert_eq!(complete(&mock, ask(json!({}))).await, busy);
    let restarting = refused(503, "unavailable_error", "Straydog is restarting");
    assert_eq!(complete(&mock, ask(json!({"stream": true}))).await, restarting);
}

#[tokio::test]
async fn engine_failures_before_and_after_the_first_byte() {
    let gpu_fault = refused(500, "server_error", "GPU fault");
    let unfinished = refused(500, "server_error", "Straydog generation ended without a terminal event");
    let mock = Mock::new(vec![
        Script::Events(vec![failure("GPU fault", true)]),
        Script::Events(vec![failure("GPU fault", true)]),
        Script::Events(vec![delta("Hel"), failure("GPU fault", true)]),
        Script::Events(vec![delta("Hel"), failure("GPU fault", true)]),
        Script::Events(vec![delta("Hel")]),
        Script::Events(vec![delta("Hel")]),
        Script::Events(vec![progress(1, 2), failure("GPU fault", true)]),
    ]);
    let quick = || ask(json!({"reasoning_effort": "none"}));
    // Before any output, the status says it.
    assert_eq!(complete(&mock, quick()).await, gpu_fault);
    let mut body = quick();
    body["stream"] = true.into();
    assert_eq!(complete(&mock, body).await, gpu_fault);
    // A JSON answer has sent nothing until the end.
    assert_eq!(complete(&mock, quick()).await, gpu_fault);
    // A stream that has begun ends with an error event.
    let (status, events) = stream(&mock, quick()).await;
    assert_eq!(status, StatusCode::OK);
    let [Sse::Chunk(first), Sse::Error(error)] = &events[..] else { panic!("{events:?}") };
    assert_eq!(first["choices"][0]["delta"], json!({"role": "assistant", "content": "Hel"}));
    assert_eq!(error, &gpu_fault.1);
    // A generation that ends without its terminal event failed.
    assert_eq!(complete(&mock, quick()).await, unfinished);
    let (_, events) = stream(&mock, quick()).await;
    assert_eq!(events.last(), Some(&Sse::Error(unfinished.1)));
    // Progress is output: an error after it arrives in the stream.
    let (status, events) = stream(&mock, quick()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(events, [Sse::Comment("prefill 1/2".into()), Sse::Error(gpu_fault.1)]);
}

fn timeout<F: IntoFuture>(future: F) -> tokio::time::Timeout<F::IntoFuture> {
    tokio::time::timeout(Duration::from_secs(5), future)
}

/// A raw HTTP/1.1 request on a real connection, so that the client can go away.
async fn connect(mock: &Arc<Mock>, body: Value) -> tokio::net::TcpStream {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(axum::serve(listener, app(mock)).into_future());
    let mut connection = tokio::net::TcpStream::connect(address).await.unwrap();
    let body = body.to_string();
    let head = format!("POST /v1/chat/completions HTTP/1.1\r\nhost: {address}\r\ncontent-type: application/json\r\n\
                        content-length: {}\r\n\r\n", body.len());
    connection.write_all((head + &body).as_bytes()).await.unwrap();
    connection
}

#[tokio::test]
async fn a_client_that_disconnects_cancels_its_generation() {
    // Streaming: the client reads the first chunk and leaves.
    let mock = Mock::new(vec![Script::Hang(vec![delta("Hello")])]);
    let mut connection = connect(&mock, ask(json!({"stream": true, "reasoning_effort": "none"}))).await;
    let mut received = Vec::new();
    while !String::from_utf8_lossy(&received).contains("Hello") {
        let mut buffer = [0u8; 4096];
        let read = connection.read(&mut buffer).await.unwrap();
        assert!(read > 0, "the server closed the stream");
        received.extend_from_slice(&buffer[..read]);
    }
    drop(connection);
    timeout(mock.dropped.notified()).await.expect("the generation is cancelled");

    // Non-streaming: the client leaves while the answer is collected.
    let mock = Mock::new(vec![Script::Hang(vec![delta("Hello")])]);
    let connection = connect(&mock, ask(json!({"reasoning_effort": "none"}))).await;
    timeout(mock.submitted.notified()).await.expect("the request reaches the engine");
    drop(connection);
    timeout(mock.dropped.notified()).await.expect("the generation is cancelled");
}

/// A DeepSeek-V4.1 server: its own reasoning levels, the card left to the worker (which reports
/// it), and DSML tool calls turned into structured ones.
#[tokio::test]
async fn a_dsml_model_gets_its_own_levels_card_and_tool_calls() {
    // (the reasoning quotes the tag: its end is the worker's reasoning_end, not that text)
    let text = "Reading.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read_file\">\n\
                <｜DSML｜ parameter name=\"path\" string=\"true\">main.rs</｜DSML｜ parameter>\n\
                <｜DSML｜ parameter name=\"limit\" string=\"false\">50</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let (first, rest) = text.split_at(text.find("<｜DSML｜ invoke").unwrap());
    let mock = Mock::new(vec![Script::Events(vec![delta("Need the file; after `</think>`"), delta(" the answer."), reasoning_end(),
                                                  delta(first), delta(rest), done("stop")]),
                              Script::Events(vec![delta("Hi"), done("stop")])]);
    mock.set_state(EngineState::Ready { context: 65536, speculative: 0, processing: false,
                                        placement: json!({"sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                                                                       "presence_penalty": 0.0}}) });
    let app = || router(mock.clone(), ApiConfig { markup: Markup::Dsml, ..config() });
    let (status, body) = fetch(app(), chat(ask(json!({"tools": read_file(), "reasoning_effort": "xhigh", "top_p": 0.9})).to_string())).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let message = &body["choices"][0]["message"];
    assert_eq!((message["reasoning_content"].as_str(), message["content"].as_str()),
               (Some("Need the file; after `</think>` the answer."), Some("Reading.")));
    let call = &message["tool_calls"][0]["function"];
    assert_eq!((call["name"].as_str(), call["arguments"].as_str()), (Some("read_file"), Some(r#"{"path": "main.rs", "limit": 50}"#)));
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    let request = &mock.requests()[0];
    assert_eq!(request.get("reasoning_effort"), Some(&pyjson::Value::Str("max".into())));
    // What the request leaves out stays null for the worker's card; what it gives is kept.
    assert_eq!(request.get("temperature"), Some(&pyjson::Value::Null));
    assert_eq!(request.get("top_p"), Some(&pyjson::Value::Float(0.9)));
    let (status, _) = fetch(app(), chat(ask(json!({"chat_template_kwargs": {"reasoning_effort": "xhigh"}})).to_string())).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = fetch(app(), chat(ask(json!({"chat_template_kwargs": {"reasoning_effort": "max", "enable_thinking": false}})).to_string())).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(mock.requests()[1].get("thinking"), Some(&pyjson::Value::Bool(false)));
    let (_, props) = fetch(app(), get("/props")).await;
    assert_eq!((props["default_generation_settings"]["params"]["top_k"].as_u64(), props["default_generation_settings"]["params"]["top_p"].as_f64()),
               (Some(0), Some(1.0)));
}
