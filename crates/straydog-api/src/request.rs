//! An OpenAI chat completion request as the worker's request (docs/DESIGN.md §6).
//!
//! `messages` and `tools` pass through as Python's `json.loads` read them, so the template
//! renders the client's numbers, key order and strings. Everything else is checked here, and
//! what the engine cannot honor is a 400: a sampling control is accepted only at the value
//! that turns it off, never ignored.

use std::str::FromStr;

use straydog_engine::pyjson::{self, Value};

use crate::{ApiConfig, ApiError};

/// The OpenAI `reasoning_effort` levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

const EFFORTS: &str = "reasoning_effort must be none, minimal, low, medium, high or xhigh";

impl ReasoningEffort {
    /// Whether the template thinks, and at which of its own levels: it has no "none" and
    /// calls its highest level, its default, xhigh.
    fn template(self) -> (bool, &'static str) {
        match self {
            ReasoningEffort::None => (false, "medium"),
            ReasoningEffort::Minimal | ReasoningEffort::Low => (true, "low"),
            ReasoningEffort::Medium => (true, "medium"),
            ReasoningEffort::High | ReasoningEffort::Xhigh => (true, "xhigh"),
        }
    }
}

impl FromStr for ReasoningEffort {
    type Err = String;

    fn from_str(name: &str) -> Result<ReasoningEffort, String> {
        Ok(match name {
            "none" => ReasoningEffort::None,
            "minimal" => ReasoningEffort::Minimal,
            "low" => ReasoningEffort::Low,
            "medium" => ReasoningEffort::Medium,
            "high" => ReasoningEffort::High,
            "xhigh" => ReasoningEffort::Xhigh,
            _ => return Err(EFFORTS.into()),
        })
    }
}

/// Controls the engine does not implement, each with the one value a request may give: the
/// one that turns it off.
const NEUTRAL: [(&str, &str); 15] = [
    ("n", "1"),
    ("top_p", "1"),
    ("top_k", "0"),
    ("min_p", "0"),
    ("typical_p", "1"),
    ("presence_penalty", "0"),
    ("frequency_penalty", "0"),
    ("repeat_penalty", "1"),
    ("repetition_penalty", "1"),
    ("logit_bias", "{}"),
    ("logprobs", "false"),
    ("top_logprobs", "0"),
    ("parallel_tool_calls", "true"),
    ("response_format", r#"{"type": "text"}"#),
    ("modalities", r#"["text"]"#),
];

/// Features the engine does not have: a request may only leave them out.
const UNSUPPORTED: [&str; 4] = ["functions", "function_call", "audio", "prediction"];

/// The message fields the template reads or a client sends back.
const MESSAGE_KEYS: [&str; 6] = ["role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"];

/// How the answer to a request is shaped.
pub(crate) struct Shape {
    /// The answer starts as reasoning.
    pub thinking: bool,
    /// The tools whose calls are parsed; none when tool calls stay text.
    pub tools: Option<Value>,
    pub stops: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
}

/// The worker request for an OpenAI chat completion request, and the shape of its answer.
pub(crate) fn translate(payload: &Value, config: &ApiConfig) -> Result<(Value, Shape), ApiError> {
    let messages = match given(payload, "messages") {
        Some(Value::Array(messages)) if !messages.is_empty() => messages,
        _ => return Err(ApiError::invalid("messages must be a non-empty list")),
    };
    for (key, neutral) in NEUTRAL {
        let value = pyjson::parse(neutral).expect("the neutral values are JSON");
        if given(payload, key).is_some_and(|given| !same(given, &value)) {
            return Err(ApiError::invalid(format!("{key} is not supported (only {neutral})")));
        }
    }
    if let Some(key) = UNSUPPORTED.into_iter().find(|key| given(payload, key).is_some()) {
        return Err(ApiError::invalid(format!("{key} is not supported")));
    }

    let tools = match given(payload, "tools") {
        None => None,
        Some(Value::Array(list)) if list.is_empty() => None,
        Some(tools @ Value::Array(list)) if list.iter().all(|tool| matches!(tool, Value::Object(_))) => {
            Some(tools.clone())
        }
        Some(_) => return Err(ApiError::invalid("tools must be a list of tool objects")),
    };
    // Without tools in the prompt the model cannot call one; with them it might.
    let tools = match given(payload, "tool_choice").map(Value::as_str) {
        None | Some(Some("auto")) => tools,
        Some(Some("none")) => None,
        Some(_) => return Err(ApiError::invalid("tool_choice supports auto and none")),
    };

    let limit = ["max_completion_tokens", "max_tokens"].into_iter().find_map(|key| Some((key, given(payload, key)?)));
    let max_tokens = match limit {
        None => Value::Null,
        Some((_, Value::Int(count))) if *count >= 1 => Value::Int(*count),
        Some((key, _)) => return Err(ApiError::invalid(format!("{key} must be a positive integer"))),
    };
    let temperature = match given(payload, "temperature") {
        None => Some(config.temperature),
        Some(Value::Int(number)) => Some(*number as f64),
        Some(Value::Float(number)) => Some(*number),
        Some(_) => None,
    }
    .filter(|temperature| (0.0..=2.0).contains(temperature))
    .ok_or_else(|| ApiError::invalid("temperature must be between 0 and 2"))?;
    let seed = match given(payload, "seed") {
        None => Some(config.seed),
        Some(Value::Int(seed)) => u64::try_from(*seed).ok(),
        Some(_) => None,
    }
    .ok_or_else(|| ApiError::invalid("seed must be a nonnegative 64-bit integer"))?;

    // An empty effort is no effort given, as the reference reads it.
    let effort = match given(payload, "reasoning_effort") {
        None => config.reasoning_effort,
        Some(Value::Str(name)) if name.is_empty() => config.reasoning_effort,
        Some(value) => value.as_str().unwrap_or_default().parse().map_err(ApiError::invalid)?,
    };
    let (mut thinking, mut level) = effort.template();
    let kwargs_error = || ApiError::invalid("chat_template_kwargs supports enable_thinking and reasoning_effort");
    match given(payload, "chat_template_kwargs") {
        None => {}
        Some(Value::Object(kwargs)) => {
            for (key, value) in kwargs {
                match key.as_str() {
                    "enable_thinking" => {
                        let Value::Bool(flag) = value else {
                            return Err(ApiError::invalid("chat_template_kwargs.enable_thinking must be a boolean"));
                        };
                        thinking = *flag;
                    }
                    "reasoning_effort" => {
                        let levels = ["low", "medium", "xhigh"];
                        level = levels.into_iter().find(|name| value.as_str() == Some(name)).ok_or_else(|| {
                            ApiError::invalid("chat_template_kwargs.reasoning_effort must be low, medium or xhigh")
                        })?;
                    }
                    _ => return Err(kwargs_error()),
                }
            }
        }
        Some(_) => return Err(kwargs_error()),
    }

    let stops = match given(payload, "stop") {
        None => Vec::new(),
        Some(Value::Str(stop)) => vec![stop.clone()],
        Some(Value::Array(list)) if list.len() <= 16 && list.iter().all(|stop| stop.as_str().is_some()) => {
            list.iter().filter_map(Value::as_str).map(str::to_string).collect()
        }
        Some(_) => return Err(ApiError::invalid("stop must be a string or a list of at most 16 strings")),
    };
    let mut checkpoint = true;
    match given(payload, "straydog") {
        None => {}
        Some(Value::Object(extension)) => {
            for (key, value) in extension {
                match (key.as_str(), value) {
                    ("checkpoint", Value::Bool(flag)) => checkpoint = *flag,
                    ("checkpoint", Value::Null) => {}
                    ("checkpoint", _) => return Err(ApiError::invalid("straydog.checkpoint must be a boolean")),
                    _ => return Err(ApiError::invalid("the straydog object supports checkpoint")),
                }
            }
        }
        Some(_) => return Err(ApiError::invalid("the straydog object supports checkpoint")),
    }
    let stream = match given(payload, "stream") {
        None => false,
        Some(Value::Bool(flag)) => *flag,
        Some(_) => return Err(ApiError::invalid("stream must be a boolean")),
    };
    let include_usage = match given(payload, "stream_options") {
        None => false,
        Some(options @ Value::Object(_)) => match given(options, "include_usage") {
            None => false,
            Some(Value::Bool(flag)) => *flag,
            Some(_) => return Err(ApiError::invalid("stream_options.include_usage must be a boolean")),
        },
        Some(_) => return Err(ApiError::invalid("stream_options must be an object")),
    };

    let messages = messages.iter().map(message).collect::<Result<Vec<_>, _>>()?;
    let request = Value::Object(vec![
        ("messages".into(), Value::Array(messages)),
        ("tools".into(), tools.clone().unwrap_or(Value::Null)),
        ("max_tokens".into(), max_tokens),
        ("temperature".into(), Value::Float(temperature)),
        ("seed".into(), Value::Int(seed.into())),
        ("thinking".into(), Value::Bool(thinking)),
        ("reasoning_effort".into(), Value::Str(level.into())),
        ("checkpoint".into(), Value::Bool(checkpoint)),
    ]);
    Ok((request, Shape { thinking, tools, stops, stream, include_usage }))
}

/// A field of `object`, null counting as absent.
fn given<'a>(object: &'a Value, key: &str) -> Option<&'a Value> {
    object.get(key).filter(|value| !matches!(value, Value::Null))
}

/// Python's `==` between JSON values, except that a bool is not a number.
fn same(value: &Value, other: &Value) -> bool {
    match (value, other) {
        (Value::Int(int), Value::Float(float)) | (Value::Float(float), Value::Int(int)) => *int as f64 == *float,
        _ => value == other,
    }
}

/// A message for the template: the known fields in the client's order, a developer message
/// as a system one, and the content as text.
fn message(message: &Value) -> Result<Value, ApiError> {
    let role = message.get("role").and_then(Value::as_str);
    let (Value::Object(fields), Some("system" | "developer" | "user" | "assistant" | "tool")) = (message, role) else {
        return Err(ApiError::invalid("each message needs a role: system, developer, user, assistant or tool"));
    };
    let mut kept = Vec::new();
    for (key, value) in fields {
        if !MESSAGE_KEYS.contains(&key.as_str()) || matches!(value, Value::Null) {
            continue;
        }
        let value = match key.as_str() {
            "role" if role == Some("developer") => Value::Str("system".into()),
            "content" => Value::Str(text(value)?),
            _ => value.clone(),
        };
        kept.push((key.clone(), value));
    }
    Ok(Value::Object(kept))
}

/// Message content as text: a string, or text parts concatenated, as the template renders a
/// list of them.
fn text(content: &Value) -> Result<String, ApiError> {
    match content {
        Value::Str(text) => Ok(text.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match (part.get("type").and_then(Value::as_str), part.get("text")) {
                (Some("text"), Some(Value::Str(text))) => Ok(text.as_str()),
                _ => Err(ApiError::invalid("only text content is supported")),
            })
            .collect::<Result<String, _>>(),
        _ => Err(ApiError::invalid("message content must be a string or a list of text parts")),
    }
}
