//! The official chat template, rendered exactly as the reference's sandboxed Jinja does.
//!
//! The environment matches the reference's: blocks trimmed and left-stripped, no
//! auto-escaping, Python string methods, `tojson` as `json.dumps(value, ensure_ascii=False)`,
//! `trim` with Python's whitespace, and `raise_exception` failing the render with its message.

use std::sync::{Arc, Mutex};

use minijinja::value::{Value as JinjaValue, ValueKind};
use minijinja::{AutoEscape, Environment, ErrorKind};

use crate::pyjson::{self, Value};

/// The largest prompt the engine renders.
pub const MAX_PROMPT_BYTES: usize = 16 * 1024 * 1024;
const MAX_MESSAGES: usize = 4096;

/// A request the template or its validation refused; the message is the client's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatError(pub String);

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ChatError {}

fn error(message: impl Into<String>) -> ChatError {
    ChatError(message.into())
}

pub struct ChatTemplate {
    environment: Environment<'static>,
    raised: Arc<Mutex<Option<String>>>,
}

/// `messages` with each message's own text through `defuse` (the worker's
/// [`crate::text::Text::defuse`]): its content, a string or text parts, and its reasoning, which
/// templates write as they are. Roles, names and tool calls stay as they were (a template writes
/// arguments through `tojson`, which would spell the mark out).
pub fn defuse_messages(messages: &Value, defuse: &dyn Fn(&str) -> String) -> Value {
    let Value::Array(items) = messages else { return messages.clone() };
    let part = |part: &Value| match part {
        Value::Object(fields) => Value::Object(fields.iter().map(|(key, value)| match (key.as_str(), value) {
            ("text", Value::Str(text)) => (key.clone(), Value::Str(defuse(text))),
            _ => (key.clone(), value.clone()),
        }).collect()),
        _ => part.clone(),
    };
    Value::Array(items.iter().map(|message| match message {
        Value::Object(fields) => Value::Object(fields.iter().map(|(key, value)| {
            let value = match (key.as_str(), value) {
                ("content" | "reasoning_content", Value::Str(text)) => Value::Str(defuse(text)),
                ("content", Value::Array(parts)) => Value::Array(parts.iter().map(part).collect()),
                _ => value.clone(),
            };
            (key.clone(), value)
        }).collect()),
        _ => message.clone(),
    }).collect())
}

impl ChatTemplate {
    pub fn new(source: &str) -> Result<ChatTemplate, ChatError> {
        let mut environment = Environment::new();
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        environment.set_keep_trailing_newline(false);
        environment.set_auto_escape_callback(|_| AutoEscape::None);
        environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_filter("tojson", |value: JinjaValue| -> Result<String, minijinja::Error> {
            Ok(pyjson::dumps(&from_jinja(&value)?))
        });
        environment.add_filter("trim", |value: JinjaValue| -> String { python_strip(&value.to_string()).to_string() });
        let raised = Arc::new(Mutex::new(None));
        let slot = raised.clone();
        environment.add_function("raise_exception", move |message: String| -> Result<JinjaValue, minijinja::Error> {
            *slot.lock().unwrap() = Some(message.clone());
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
        });
        environment
            .add_template_owned("chat", source.to_string())
            .map_err(|exc| error(format!("invalid chat template: {exc}")))?;
        Ok(ChatTemplate { environment, raised })
    }

    /// The prompt for `messages` and `tools`, ending with the assistant's generation prompt
    /// (templates that take `enable_thinking` and a `reasoning_effort` of low, medium or xhigh).
    pub fn render(&self, messages: &Value, tools: &Value, thinking: bool, effort: &str) -> Result<String, ChatError> {
        let decoded = Self::messages(messages, false)?;
        if !matches!(effort, "low" | "medium" | "xhigh") {
            return Err(error("invalid Straydog reasoning effort"));
        }
        self.render_decoded(&decoded, tools, &[("enable_thinking", Value::Bool(thinking)), ("reasoning_effort", Value::Str(effort.into()))])
    }

    /// The prompt for `messages` and `tools` with the template's own variables (a family's
    /// thinking and effort controls), ending with the assistant's generation prompt. A message's
    /// content may also be a list of text parts (a request's with images, each image's part a text
    /// part by then: see the worker's images).
    pub fn render_with(&self, messages: &Value, tools: &Value, variables: &[(&str, Value)]) -> Result<String, ChatError> {
        self.render_decoded(&Self::messages(messages, true)?, tools, variables)
    }

    fn messages(messages: &Value, parts: bool) -> Result<Vec<Value>, ChatError> {
        let Value::Array(items) = messages else {
            return Err(error("invalid Straydog messages"));
        };
        if items.is_empty() || items.len() > MAX_MESSAGES {
            return Err(error("invalid Straydog messages"));
        }
        for message in items {
            let content = message.get("content");
            let text_parts = |list: &[Value]| list.iter().all(|part| {
                part.get("type").and_then(Value::as_str) == Some("text") && matches!(part.get("text"), Some(Value::Str(_)))
            });
            if !matches!(message, Value::Object(_))
                || !(matches!(content, None | Some(Value::Null) | Some(Value::Str(_)))
                     || parts && matches!(content, Some(Value::Array(list)) if text_parts(list)))
            {
                return Err(error("Straydog initially supports text messages only"));
            }
        }
        items.iter().map(template_message).collect()
    }

    fn render_decoded(&self, decoded: &[Value], tools: &Value, variables: &[(&str, Value)]) -> Result<String, ChatError> {
        let template = self.environment.get_template("chat").map_err(|exc| error(exc.to_string()))?;
        *self.raised.lock().unwrap() = None;
        let mut context = vec![
            ("messages".to_string(), JinjaValue::from(decoded.iter().map(to_jinja).collect::<Vec<_>>())),
            ("tools".to_string(), to_jinja(tools)),
            ("add_generation_prompt".to_string(), JinjaValue::from(true)),
        ];
        context.extend(variables.iter().map(|(name, value)| (name.to_string(), to_jinja(value))));
        let context: JinjaValue = context.into_iter().collect();
        let text = template.render(context).map_err(|exc| {
            // The template's own refusal carries its message; anything else is the engine's.
            match self.raised.lock().unwrap().take() {
                Some(message) => error(message),
                None => error(exc.detail().map(str::to_string).unwrap_or_else(|| exc.to_string())),
            }
        })?;
        if text.len() > MAX_PROMPT_BYTES {
            return Err(error("Straydog rendered prompt exceeds the IPC limit"));
        }
        Ok(text)
    }
}

/// Decode OpenAI-style JSON tool arguments into the mapping the template iterates; the
/// template reads `call.function` when present, else the call itself.
fn template_message(message: &Value) -> Result<Value, ChatError> {
    let Some(Value::Array(calls)) = message.get("tool_calls") else {
        return Ok(message.clone());
    };
    let decoded = calls
        .iter()
        .map(|call| {
            let wrapped = matches!(call.get("function"), Some(Value::Object(_)));
            let function = if wrapped { call.get("function").unwrap() } else { call };
            let Some(Value::Str(text)) = function.get("arguments") else {
                return Ok(call.clone());
            };
            let arguments = if python_strip(text).is_empty() {
                Value::Object(Vec::new())
            } else {
                pyjson::parse(text).map_err(|_| error("Straydog tool-call arguments are not valid JSON"))?
            };
            if !matches!(arguments, Value::Object(_)) {
                return Err(error("Straydog tool-call arguments must be a JSON object"));
            }
            let function = replace(function, "arguments", arguments);
            Ok(if wrapped { replace(call, "function", function) } else { function })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(replace(message, "tool_calls", Value::Array(decoded)))
}

/// A copy of `object` with `key` set, in place when present (a Python dict merge).
fn replace(object: &Value, key: &str, value: Value) -> Value {
    let Value::Object(entries) = object else { unreachable!("callers pass objects") };
    let mut entries = entries.clone();
    match entries.iter_mut().find(|(name, _)| name == key) {
        Some(entry) => entry.1 = value,
        None => entries.push((key.to_string(), value)),
    }
    Value::Object(entries)
}

/// Python's `str.strip()`: Unicode whitespace plus the separators U+001C-U+001F.
pub fn python_strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn to_jinja(value: &Value) -> JinjaValue {
    match value {
        Value::Null => JinjaValue::from(()),
        Value::Bool(flag) => JinjaValue::from(*flag),
        Value::Int(number) => JinjaValue::from(*number),
        Value::Float(number) => JinjaValue::from(*number),
        Value::Str(text) => JinjaValue::from(text.as_str()),
        Value::Array(items) => JinjaValue::from(items.iter().map(to_jinja).collect::<Vec<_>>()),
        Value::Object(entries) => entries.iter().map(|(key, item)| (key.as_str(), to_jinja(item))).collect(),
    }
}

fn from_jinja(value: &JinjaValue) -> Result<Value, minijinja::Error> {
    let unsupported = || minijinja::Error::new(ErrorKind::InvalidOperation, format!("cannot serialize {}", value.kind()));
    Ok(match value.kind() {
        ValueKind::None => Value::Null,
        ValueKind::Bool => Value::Bool(value.is_true()),
        ValueKind::Number if value.is_integer() => Value::Int(i128::try_from(value.clone())?),
        ValueKind::Number => Value::Float(f64::try_from(value.clone())?),
        ValueKind::String => Value::Str(value.as_str().unwrap_or_default().to_string()),
        ValueKind::Seq | ValueKind::Iterable => {
            Value::Array(value.try_iter()?.map(|item| from_jinja(&item)).collect::<Result<_, _>>()?)
        }
        ValueKind::Map => {
            let mut entries = Vec::new();
            for key in value.try_iter()? {
                let name = key.as_str().ok_or_else(unsupported)?.to_string();
                entries.push((name, from_jinja(&value.get_item(&key)?)?));
            }
            Value::Object(entries)
        }
        _ => return Err(unsupported()),
    })
}
