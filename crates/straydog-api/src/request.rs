//! An OpenAI chat completion request as the worker's request (docs/DESIGN.md §6).
//!
//! `messages` and `tools` pass through as Python's `json.loads` read them, so the template
//! renders the client's numbers, key order and strings. Everything else is checked here, and
//! what the engine cannot honor is a 400: a sampling control it does not implement is
//! accepted only at the value that turns it off, never ignored. Sampling a request leaves
//! out follows the server's settings, then the model card for the request's mode.

use std::str::FromStr;

use straydog_engine::pyjson::{self, Value};

use crate::{ApiConfig, ApiError, Markup};

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
    /// Whether the template thinks, and at which of its own levels. Qwen's has no "none" and calls
    /// its highest level, its default, xhigh; DeepSeek-V4.1's are budgets named low (50), medium
    /// (62), high (75, its default) and max (100).
    fn template(self, markup: Markup) -> (bool, &'static str) {
        match (markup, self) {
            (Markup::Qwen, ReasoningEffort::None) => (false, "medium"),
            (Markup::Qwen, ReasoningEffort::Minimal | ReasoningEffort::Low) => (true, "low"),
            (Markup::Qwen, ReasoningEffort::Medium) => (true, "medium"),
            (Markup::Qwen, ReasoningEffort::High | ReasoningEffort::Xhigh) => (true, "xhigh"),
            (Markup::Dsml, ReasoningEffort::None) => (false, "high"),
            (Markup::Dsml, ReasoningEffort::Minimal | ReasoningEffort::Low) => (true, "low"),
            (Markup::Dsml, ReasoningEffort::Medium) => (true, "medium"),
            (Markup::Dsml, ReasoningEffort::High) => (true, "high"),
            (Markup::Dsml, ReasoningEffort::Xhigh) => (true, "max"),
        }
    }

    /// The template's own levels a request may name in `chat_template_kwargs`.
    fn levels(markup: Markup) -> &'static [&'static str] {
        match markup {
            Markup::Qwen => &["low", "medium", "xhigh"],
            Markup::Dsml => &["low", "medium", "high", "max"],
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
const NEUTRAL: [(&str, &str); 10] = [
    ("n", "1"),
    ("typical_p", "1"),
    ("frequency_penalty", "0"),
    ("repeat_penalty", "1"),
    ("repetition_penalty", "1"),
    ("logit_bias", "{}"),
    ("logprobs", "false"),
    ("top_logprobs", "0"),
    ("parallel_tool_calls", "true"),
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
    /// How the template marks reasoning and tool calls.
    pub markup: Markup,
    /// The tools whose calls are parsed; none when tool calls stay text.
    pub tools: Option<Value>,
    pub stops: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
}

/// The worker request for an OpenAI chat completion request, and the shape of its answer.
pub(crate) fn translate(payload: &Value, config: &ApiConfig, vision: bool) -> Result<(Value, Shape), ApiError> {
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
    // Without tools in the prompt the model cannot call one; with them it might, or (required, or
    // a function named) must.
    let choice_error = || ApiError::invalid("tool_choice must be auto, none, required or {\"type\": \"function\", \"function\": {\"name\": ...}}");
    let (tools, forced) = match given(payload, "tool_choice") {
        None => (tools, None),
        Some(Value::Str(choice)) => match choice.as_str() {
            "auto" => (tools, None),
            "none" => (None, None),
            "required" if tools.is_some() => (tools, Some(None)),
            "required" => return Err(ApiError::invalid("tool_choice required needs tools")),
            _ => return Err(choice_error()),
        },
        Some(choice @ Value::Object(_)) => {
            let name = match (choice.get("type").and_then(Value::as_str), choice.get("function").and_then(|function| function.get("name"))) {
                (Some("function"), Some(Value::Str(name))) => name.clone(),
                _ => return Err(choice_error()),
            };
            let known = tools.as_ref().is_some_and(|tools| match tools {
                Value::Array(list) => list.iter().any(|tool| tool.get("function").and_then(|function| function.get("name")) == Some(&Value::Str(name.clone()))),
                _ => false,
            });
            if !known {
                return Err(ApiError::invalid(format!("tool_choice names {name}, which is not among the tools")));
            }
            (tools, Some(Some(name)))
        }
        Some(_) => return Err(choice_error()),
    };
    let json = match given(payload, "response_format") {
        None => None,
        Some(format) => match format.get("type").and_then(Value::as_str) {
            Some("text") => None,
            Some("json_object") => Some(Value::Bool(true)),
            Some("json_schema") => {
                let schema = format.get("json_schema").and_then(|spec| spec.get("schema"))
                    .ok_or_else(|| ApiError::invalid("response_format json_schema needs json_schema.schema"))?;
                straydog_engine::constraint::Schema::compile(schema)
                    .map_err(|error| ApiError::invalid(format!("response_format json_schema: {error}")))?;
                Some(schema.clone())
            }
            _ => return Err(ApiError::invalid("response_format type must be text, json_object or json_schema")),
        },
    };
    let constraint = constraint(config.markup, json, tools.is_some(), forced);

    let limit = ["max_completion_tokens", "max_tokens"].into_iter().find_map(|key| Some((key, given(payload, key)?)));
    let max_tokens = match limit {
        None => Value::Null,
        Some((_, Value::Int(count))) if *count >= 1 => Value::Int(*count),
        Some((key, _)) => return Err(ApiError::invalid(format!("{key} must be a positive integer"))),
    };
    let temperature = number(payload, "temperature", |value| (0.0..=2.0).contains(&value), "between 0 and 2")?;
    let top_p = number(payload, "top_p", |value| value > 0.0 && value <= 1.0, "in (0, 1]")?;
    let min_p = number(payload, "min_p", |value| (0.0..1.0).contains(&value), "in [0, 1)")?;
    let presence_penalty = number(payload, "presence_penalty", |value| (-2.0..=2.0).contains(&value), "between -2 and 2")?;
    // 0 and -1 (vLLM's spelling) keep every token.
    let top_k = match given(payload, "top_k") {
        None => None,
        Some(Value::Int(-1)) => Some(0),
        Some(Value::Int(count)) => Some(usize::try_from(*count).ok().filter(|count| *count <= 1_000_000)
            .ok_or_else(|| ApiError::invalid("top_k must be -1, 0 or a positive integer"))?),
        Some(_) => return Err(ApiError::invalid("top_k must be -1, 0 or a positive integer")),
    };
    let seed = match given(payload, "seed") {
        None => Some(config.seed.unwrap_or_else(|| getrandom::u64().expect("the OS random source"))),
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
    let markup = config.markup;
    let (mut thinking, mut level) = effort.template(markup);
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
                        let levels = ReasoningEffort::levels(markup);
                        level = levels.iter().copied().find(|name| value.as_str() == Some(name)).ok_or_else(|| {
                            let (last, rest) = levels.split_last().unwrap();
                            ApiError::invalid(format!("chat_template_kwargs.reasoning_effort must be {} or {last}", rest.join(", ")))
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

    let messages = messages.iter().map(|item| message(item, vision)).collect::<Result<Vec<_>, _>>()?;
    // Qwen's card is the API's; another family's worker reports its own card and fills in what is
    // left null here.
    let (temperature, top_k, top_p, min_p, presence_penalty) = match markup {
        Markup::Qwen => {
            let defaults = config.sampling.resolve(thinking);
            (Value::Float(temperature.unwrap_or(defaults.temperature)), Value::Int(top_k.unwrap_or(defaults.top_k) as i128),
             Value::Float(top_p.unwrap_or(defaults.top_p)), Value::Float(min_p.unwrap_or(defaults.min_p)),
             Value::Float(presence_penalty.unwrap_or(defaults.presence_penalty)))
        }
        Markup::Dsml => {
            let set = &config.sampling;
            let float = |value: Option<f64>| value.map_or(Value::Null, Value::Float);
            (float(temperature.or(set.temperature)), top_k.or(set.top_k).map_or(Value::Null, |k| Value::Int(k as i128)),
             float(top_p.or(set.top_p)), float(min_p.or(set.min_p)), float(presence_penalty.or(set.presence_penalty)))
        }
    };
    let request = Value::Object(vec![
        ("messages".into(), Value::Array(messages)),
        ("tools".into(), tools.clone().unwrap_or(Value::Null)),
        ("max_tokens".into(), max_tokens),
        ("temperature".into(), temperature),
        ("top_k".into(), top_k),
        ("top_p".into(), top_p),
        ("min_p".into(), min_p),
        ("presence_penalty".into(), presence_penalty),
        ("seed".into(), Value::Int(seed.into())),
        ("thinking".into(), Value::Bool(thinking)),
        ("reasoning_effort".into(), Value::Str(level.into())),
        ("checkpoint".into(), Value::Bool(checkpoint)),
        ("constraint".into(), constraint),
    ]);
    Ok((request, Shape { thinking, markup, tools, stops, stream, include_usage }))
}

/// What the worker holds the answer to ([`straydog_engine::constraint`]), as its request's
/// `constraint`: a forced tool call's opening (the markup's, with the named function's name), or
/// the JSON `response_format` asks for (`true`: any object), racing a tool call's opening when
/// the model may call one. Null: nothing.
fn constraint(markup: Markup, json: Option<Value>, tools: bool, forced: Option<Option<String>>) -> Value {
    // A named function is called once: its invoke's close, then the block's, then the end.
    let call = |name: Option<&str>| {
        let (open, whitespace, until, close) = match markup {
            Markup::Dsml => (format!("\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"{}", name.map_or(String::new(), |name| format!("{name}\">\n"))),
                             0, "</｜DSML｜ invoke>", "\n</｜DSML｜ calls>"),
            Markup::Qwen => (format!("<tool_call>\n<function={}", name.map_or(String::new(), |name| format!("{name}>\n"))), 4,
                             "</function>", "\n</tool_call>"),
        };
        let mut fields = vec![("prefix".into(), Value::Str(open)), ("whitespace".into(), Value::Int(whitespace))];
        if name.is_some() {
            fields.push(("until".into(), Value::Str(until.into())));
            fields.push(("close".into(), Value::Str(close.into())));
        }
        Value::Object(fields)
    };
    let alternatives = match (forced, json) {
        (Some(name), _) => vec![call(name.as_deref())],
        (None, Some(schema)) => {
            let mut alternatives = vec![Value::Object(vec![("json".into(), schema)])];
            if tools {
                alternatives.push(call(None));
            }
            alternatives
        }
        (None, None) => return Value::Null,
    };
    Value::Object(vec![("alternatives".into(), Value::Array(alternatives))])
}

/// A number the request gives, checked by `valid`.
fn number(payload: &Value, key: &str, valid: impl Fn(f64) -> bool, range: &str) -> Result<Option<f64>, ApiError> {
    match given(payload, key) {
        None => Ok(None),
        Some(Value::Int(number)) if valid(*number as f64) => Ok(Some(*number as f64)),
        Some(Value::Float(number)) if valid(*number) => Ok(Some(*number)),
        Some(_) => Err(ApiError::invalid(format!("{key} must be {range}"))),
    }
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

/// What an image is refused with where the model reads none.
pub(crate) const NO_IMAGES: &str = "this model reads no images";

/// A message for the template: the known fields in the client's order, a developer message
/// as a system one, and the content as text; with `vision` (a model that reads images), a user's
/// or a tool's content with images as its parts.
fn message(message: &Value, vision: bool) -> Result<Value, ApiError> {
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
            "content" if vision && matches!(role, Some("user" | "tool")) && has_images(value) => parts(value)?,
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
                (Some("image_url" | "image"), _) => Err(ApiError::invalid(NO_IMAGES)),
                _ => Err(ApiError::invalid("only text and image content is supported")),
            })
            .collect::<Result<String, _>>(),
        _ => Err(ApiError::invalid("message content must be a string or a list of text parts")),
    }
}

fn has_images(content: &Value) -> bool {
    matches!(content, Value::Array(parts) if parts.iter().any(|part| matches!(part.get("type").and_then(Value::as_str), Some("image_url" | "image"))))
}

/// Content with images, as its parts for the template (the worker takes the images out): text
/// parts, and images as `image_url` parts with a `data:` URL or `image` parts with a base64 source.
fn parts(content: &Value) -> Result<Value, ApiError> {
    let Value::Array(parts) = content else { return Err(ApiError::invalid("message content must be a list of parts")) };
    for part in parts {
        let valid = match part.get("type").and_then(Value::as_str) {
            Some("text") => matches!(part.get("text"), Some(Value::Str(_))),
            Some("image_url") => {
                let url = match part.get("image_url") {
                    Some(Value::Str(url)) => Some(url.as_str()),
                    Some(object) => object.get("url").and_then(Value::as_str),
                    None => None,
                };
                match url {
                    Some(url) if url.starts_with("data:") => true,
                    Some(_) => return Err(ApiError::invalid("an image_url must be a data: URL (the server fetches nothing)")),
                    None => false,
                }
            }
            Some("image") => part.get("source").is_some_and(|source| source.get("data").and_then(Value::as_str).is_some()),
            _ => false,
        };
        if !valid {
            return Err(ApiError::invalid("only text and image content is supported"));
        }
    }
    Ok(content.clone())
}
