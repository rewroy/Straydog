//! A family's generated text split as an OpenAI answer: reasoning up to `</think>`, then the
//! answer, then optional tool-call markup turned into structured calls ([`Markup`]: Qwen's
//! `<tool_call>` blocks or DeepSeek-V4.1's DSML). Stop strings end the text first.

use std::sync::LazyLock;

use regex::Regex;

use crate::chat::python_strip;
use crate::pyjson::{self, Value};

/// The templates' end of reasoning.
pub const THINK_CLOSE: &str = "</think>";
const TOOL_CALL_OPEN: &str = "<tool_call>";
/// DSML: the tool calls follow the answer after a blank line (DeepSeek-V4.1's tag names start
/// with a space).
const DSML_CALLS_OPEN: &str = "\n\n<｜DSML｜ calls";
const DSML_CALLS_CLOSE: &str = "</｜DSML｜ calls>";
const DSML_INVOKE_OPEN: &str = "<｜DSML｜ invoke";
const DSML_INVOKE_CLOSE: &str = "</｜DSML｜ invoke";
const DSML_PARAMETER_OPEN: &str = "<｜DSML｜ parameter";
const DSML_PARAMETER_CLOSE: &str = "/｜DSML｜ parameter";

/// How a family's template marks the end of reasoning and writes tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Markup {
    /// Qwen: `</think>` and `<tool_call>` count only at a line start, and the answer's leading
    /// blank lines are the template's.
    Qwen,
    /// DeepSeek-V4.1: the first `</think>` ends the reasoning, the answer is verbatim, and a
    /// `<｜DSML｜ calls>` block after a blank line holds the tool calls.
    Dsml,
}

impl Markup {
    /// The markup of a GGUF architecture's chat template.
    pub fn of_architecture(architecture: &str) -> Option<Markup> {
        match architecture {
            "qwen4exp" | "qwen35moe" => Some(Markup::Qwen),
            "deepseek41" => Some(Markup::Dsml),
            _ => None,
        }
    }
}
// Python's \s: Unicode whitespace and the separators U+001C-U+001F.
static TOOL_CALL_BLOCK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>[\s\x1c-\x1f]*<function=([^>\s\x1c-\x1f]+)>(.*?)</function>[\s\x1c-\x1f]*</tool_call>")
        .unwrap()
});
static TOOL_PARAMETER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<parameter=([^>\s\x1c-\x1f]+)>(.*?)</parameter>").unwrap());
static DSML_NAME: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?s)^[\s\x1c-\x1f]*name="(.*?)">\n$"#).unwrap());
static DSML_PARAMETER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?s)^ name="(.*?)" string="(true|false)">(.*?)<$"#).unwrap());

/// Length of the tail of `text` that could be the start of a split `tag`.
fn split_tail(text: &str, tag: &str) -> usize {
    (1..tag.len().min(text.len() + 1))
        .rev()
        .find(|n| text.is_char_boundary(text.len() - n) && tag.starts_with(&text[text.len() - n..]))
        .unwrap_or(0)
}

/// Index of the first `tag` in `text` that opens a line. The template writes its control
/// tags only at a line start; a model discussing them writes them mid-line, and both decode
/// to the same token, so position is the only signal. `line_start`: the text before ended a line.
fn line_tag(text: &str, tag: &str, line_start: bool) -> Option<usize> {
    let mut from = 0;
    while let Some(found) = text[from..].find(tag) {
        let at = from + found;
        let opens_line = if at == 0 { line_start } else { text.as_bytes()[at - 1] == b'\n' };
        if opens_line {
            return Some(at);
        }
        from = at + 1;
    }
    None
}

/// Generation starts inside an open `<think>` block when thinking: only the template's
/// line-start `</think>` turns it into the answer.
pub struct ReasoningSplitter {
    reasoning: bool,
    carry: String,
    answer_lead: bool,
    line_start: bool,
    markup: Markup,
    /// Reasoning ends only at [`ReasoningSplitter::end`] (the worker's, at the template's own
    /// `</think>` token), never at the text.
    by_token: bool,
}

impl ReasoningSplitter {
    pub fn new(thinking: bool) -> ReasoningSplitter {
        ReasoningSplitter::with_markup(thinking, Markup::Qwen)
    }

    pub fn with_markup(thinking: bool, markup: Markup) -> ReasoningSplitter {
        // The prompt ends with "<think>\n" (Qwen) or "<think>" (DSML).
        ReasoningSplitter { reasoning: thinking, carry: String::new(), answer_lead: markup == Markup::Qwen, line_start: true, markup,
                            by_token: false }
    }

    /// The same, the reasoning ending only at [`ReasoningSplitter::end`]: the worker sees the
    /// template's `</think>` as its token and leaves it out of the text, where the same text is
    /// the reasoning's own (quoting the tag) and was taken for its end.
    pub fn ending_by_token(mut self) -> ReasoningSplitter {
        self.by_token = true;
        self
    }

    /// The reasoning's end ([`ReasoningSplitter::ending_by_token`]): what it held, as reasoning.
    pub fn end(&mut self) -> String {
        self.reasoning = false;
        std::mem::take(&mut self.carry)
    }

    /// (answer, reasoning) for one delta.
    pub fn feed(&mut self, text: &str) -> (String, String) {
        if !self.reasoning {
            return (self.answer(text), String::new());
        }
        if self.by_token {
            return (String::new(), text.to_string());
        }
        let mut buffer = std::mem::take(&mut self.carry) + text;
        let close = match self.markup {
            Markup::Qwen => line_tag(&buffer, THINK_CLOSE, self.line_start),
            Markup::Dsml => buffer.find(THINK_CLOSE),
        };
        if let Some(end) = close {
            self.reasoning = false;
            let answer = self.answer(&buffer[end + THINK_CLOSE.len()..]);
            buffer.truncate(end);
            return (answer, buffer);
        }
        let hold = split_tail(&buffer, THINK_CLOSE);
        if hold > 0 {
            self.carry = buffer.split_off(buffer.len() - hold);
        }
        if !buffer.is_empty() {
            self.line_start = buffer.ends_with('\n');
        }
        (String::new(), buffer)
    }

    /// A held tail when the stream ended while still reasoning.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.carry)
    }

    fn answer(&mut self, text: &str) -> String {
        // The template separates reasoning from the answer with blank lines.
        if self.answer_lead {
            let text = text.trim_start_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c));
            self.answer_lead = text.is_empty();
            return text.to_string();
        }
        text.to_string()
    }
}

/// One structured tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub index: usize,
    pub id: String,
    pub name: String,
    /// JSON, as the reference writes it.
    pub arguments: String,
    /// DSML's tool namespace (`namespace::name`), when the call names one.
    pub namespace: Option<String>,
}

/// The template's tool-call markup held back from the answer and parsed at the end. Only
/// an opener at a line start counts; markup that does not parse is released as text.
pub struct ToolCallSplitter {
    string_parameters: Vec<(String, Vec<String>)>,
    enabled: bool,
    carry: String,
    markup: Option<String>,
    line_start: bool,
    format: Markup,
    /// Malformed tool calls [`ToolCallSplitter::finish`] dropped (see `parse_dsml`, `ToolCallSplitter::repair`).
    pub dropped: usize,
    /// The text ended before the model did (a token limit, a stop string): a call left open is not a call.
    pub cut: bool,
    /// [`ToolCallSplitter::finish`] read Qwen's calls only by repairing their markup.
    pub repaired: bool,
    /// [`ToolCallSplitter::finish`] released markup that held no call it could read, as text.
    pub unread: bool,
}

fn is_string_schema(schema: Option<&Value>) -> bool {
    let kinds: Vec<&Value> = match schema.and_then(|schema| schema.get("type")) {
        Some(Value::Array(kinds)) => kinds.iter().collect(),
        Some(kind) => vec![kind],
        None => vec![&Value::Null],
    };
    kinds.iter().any(|kind| kind.as_str() == Some("string"))
        && kinds.iter().all(|kind| matches!(kind.as_str(), Some("string" | "null")))
}

impl ToolCallSplitter {
    pub fn new(tools: Option<&Value>) -> ToolCallSplitter {
        ToolCallSplitter::with_markup(tools, Markup::Qwen)
    }

    pub fn with_markup(tools: Option<&Value>, format: Markup) -> ToolCallSplitter {
        let mut string_parameters = Vec::new();
        let list = match tools {
            Some(Value::Array(list)) => list.as_slice(),
            _ => &[],
        };
        for tool in list {
            let Some(function @ Value::Object(_)) = tool.get("function") else { continue };
            let Some(name) = function.get("name").and_then(Value::as_str) else { continue };
            let mut strings = Vec::new();
            if let Some(Value::Object(properties)) = function.get("parameters").and_then(|p| p.get("properties")) {
                for (key, schema) in properties {
                    if is_string_schema(Some(schema)) {
                        strings.push(key.clone());
                    }
                }
            }
            string_parameters.push((name.to_string(), strings));
        }
        ToolCallSplitter {
            string_parameters,
            enabled: !list.is_empty(),
            carry: String::new(),
            markup: None,
            line_start: true,
            format,
            dropped: 0,
            cut: false,
            repaired: false,
            unread: false,
        }
    }

    /// The answer text safe to show for one delta.
    pub fn feed(&mut self, text: &str) -> String {
        if !self.enabled {
            return text.to_string();
        }
        if let Some(markup) = self.markup.as_mut() {
            markup.push_str(text);
            return String::new();
        }
        let mut buffer = std::mem::take(&mut self.carry) + text;
        let (open, start) = match self.format {
            Markup::Qwen => (TOOL_CALL_OPEN, line_tag(&buffer, TOOL_CALL_OPEN, self.line_start)),
            Markup::Dsml => (DSML_CALLS_OPEN, buffer.find(DSML_CALLS_OPEN)),
        };
        if let Some(start) = start {
            self.markup = Some(buffer.split_off(start));
            return buffer;
        }
        let hold = split_tail(&buffer, open);
        if hold > 0 {
            self.carry = buffer.split_off(buffer.len() - hold);
        }
        if !buffer.is_empty() {
            self.line_start = buffer.ends_with('\n');
        }
        buffer
    }

    /// (late answer text, tool calls) once the stream ends.
    pub fn finish(&mut self) -> (String, Vec<ToolCall>) {
        let carry = std::mem::take(&mut self.carry);
        let Some(markup) = self.markup.take() else { return (carry, Vec::new()) };
        let parsed = match self.format {
            Markup::Qwen => match self.parse(&markup) {
                Some(calls) => Some((calls, 0, String::new())),
                None => {
                    let repaired = self.repair(&markup);
                    self.repaired = repaired.is_some();
                    repaired
                }
            },
            Markup::Dsml => parse_dsml(&markup).map(|(calls, dropped)| (calls, dropped, String::new())),
        };
        match parsed {
            Some((calls, dropped, prose)) => {
                self.dropped = dropped;
                (carry + &prose, calls)
            }
            None => {
                self.unread = true;
                (carry + &markup, Vec::new())
            }
        }
    }

    /// Qwen's calls from markup [`ToolCallSplitter::parse`] refuses, the model's slips the reference reads as no call at all
    /// (its calls then reach the client as answer text, which ends an agent's run): prose after or between the calls (the
    /// answer's), a close left out, a block it cannot read among others (dropped), a block of JSON. (calls, dropped, prose).
    /// What could be a value holding markup is never read as the end of a call: any closing tag left outside a block, or a
    /// block open where another opens, and the whole is released as text, where a cut-short call would run as if complete.
    fn repair(&self, markup: &str) -> Option<(Vec<ToolCall>, usize, String)> {
        let (mut calls, mut dropped, mut prose) = (Vec::new(), 0, String::new());
        let mut at = 0;
        while at < markup.len() {
            let Some(open) = line_tag(&markup[at..], TOOL_CALL_OPEN, at == 0 || markup.as_bytes()[at - 1] == b'\n') else {
                prose.push_str(&markup[at..]);
                break;
            };
            prose.push_str(&markup[at..at + open]);
            let body = at + open + TOOL_CALL_OPEN.len();
            let next = line_tag(&markup[body..], TOOL_CALL_OPEN, false).map_or(markup.len(), |found| body + found);
            let block = read_block(&markup[body..next]);
            let last = next == markup.len();
            match block {
                Block::Open(..) if !last || self.cut => return None,
                Block::Open(call) => self.keep(call, &mut calls, &mut dropped),
                Block::Closed(call, end) => {
                    self.keep(call, &mut calls, &mut dropped);
                    prose.push_str(&markup[body + end..next]);
                }
            }
            at = next;
        }
        if calls.is_empty() || ["</tool_call>", "</function>", "</parameter>"].iter().any(|close| prose.contains(close)) {
            return None;
        }
        // (the blank between the answer's last line and the markup, or what the calls left)
        let prose = if python_strip(&prose).is_empty() { String::new() } else { python_strip(&prose).to_string() };
        Some((calls, dropped, prose))
    }

    /// A block's call among `calls`; `dropped` counts one it cannot read, or whose key comes twice.
    fn keep(&self, call: Option<BlockCall>, calls: &mut Vec<ToolCall>, dropped: &mut usize) {
        let Some((name, arguments)) = call else {
            *dropped += 1;
            return;
        };
        if arguments.iter().enumerate().any(|(at, (key, _))| arguments[..at].iter().any(|(earlier, _)| earlier == key)) {
            *dropped += 1;
            return;
        }
        let strings = self.string_parameters.iter().rev().find(|(tool, _)| *tool == name).map(|(_, keys)| keys.as_slice());
        let arguments = arguments.into_iter().map(|(key, value)| {
            let value = match value {
                Value::Str(text) if !strings.is_some_and(|keys| keys.contains(&key)) =>
                    pyjson::parse(&text).unwrap_or(Value::Str(text)),
                value => value,
            };
            (key, value)
        }).collect();
        calls.push(ToolCall { index: calls.len(), id: format!("call_{}", random_hex(24)), name,
                              arguments: pyjson::dumps(&Value::Object(arguments)), namespace: None });
    }

    fn parse(&self, markup: &str) -> Option<Vec<ToolCall>> {
        let blocks: Vec<_> = TOOL_CALL_BLOCK.captures_iter(markup).collect();
        if blocks.is_empty() || !python_strip(&TOOL_CALL_BLOCK.replace_all(markup, "")).is_empty() {
            return None;
        }
        let mut calls = Vec::new();
        for (index, block) in blocks.iter().enumerate() {
            let (name, body) = (&block[1], &block[2]);
            if !python_strip(&TOOL_PARAMETER.replace_all(body, "")).is_empty() {
                return None;
            }
            let strings = self.string_parameters.iter().rev().find(|(tool, _)| tool == name).map(|(_, keys)| keys.as_slice());
            let mut arguments: Vec<(String, Value)> = Vec::new();
            for parameter in TOOL_PARAMETER.captures_iter(body) {
                let key = parameter[1].to_string();
                if arguments.iter().any(|(existing, _)| *existing == key) {
                    return None;
                }
                let value = template_value(&parameter[2]);
                let parsed = if strings.is_some_and(|keys| keys.contains(&key)) {
                    Value::Str(value.to_string())
                } else {
                    // The template writes non-strings with tojson; anything else is text.
                    pyjson::parse(value).unwrap_or_else(|_| Value::Str(value.to_string()))
                };
                arguments.push((key, parsed));
            }
            calls.push(ToolCall {
                index,
                id: format!("call_{}", random_hex(24)),
                name: name.to_string(),
                arguments: pyjson::dumps(&Value::Object(arguments)),
                namespace: None,
            });
        }
        Some(calls)
    }
}

/// A `<tool_call>` block as [`ToolCallSplitter::repair`] reads it, past its opener: its call (None: one that cannot be read) and
/// whether its close was written.
enum Block {
    /// Closed; the text after its close begins at the index.
    Closed(Option<BlockCall>, usize),
    /// Its text ended first.
    Open(Option<BlockCall>),
}

/// A block's tool and its parameters in order, a key perhaps twice.
type BlockCall = (String, Vec<(String, Value)>);

/// The block in `text`, which runs up to the next opener or the end of the markup.
fn read_block(text: &str) -> Block {
    let start = text.len() - python_strip_start(text).len();
    let inner = &text[start..];
    if inner.starts_with('{') {
        return read_json_block(text, start);
    }
    let Some(rest) = inner.strip_prefix("<function=") else {
        return match text.find("</tool_call>") {
            Some(close) => Block::Closed(None, close + "</tool_call>".len()),
            None => Block::Open(None),
        };
    };
    let Some(name_end) = rest.find('>') else { return Block::Open(None) };
    let name = &rest[..name_end];
    if name.is_empty() || name.chars().any(python_space) {
        return match text.find("</tool_call>") {
            Some(close) => Block::Closed(None, close + "</tool_call>".len()),
            None => Block::Open(None),
        };
    }
    let body_start = start + "<function=".len() + name_end + 1;
    let (arguments, read, closed) = read_parameters(&text[body_start..]);
    let after = body_start + read;
    let call = arguments.map(|arguments| (name.to_string(), arguments));
    if !closed {
        return Block::Open(call);
    }
    let tail = &text[after..];
    let spaced = tail.len() - python_strip_start(tail).len();
    match tail[spaced..].strip_prefix("</tool_call>") {
        Some(_) => Block::Closed(call, after + spaced + "</tool_call>".len()),
        // (its own close left out: the call is whole, and what follows is the answer's)
        None => Block::Closed(call, after),
    }
}

/// The parameters of a function's body, up to and past `</function>`: their values (None: text no parameter holds), the bytes
/// read and whether the close was there. A value ends at the first `</parameter>` the markup goes on from; at the text's
/// end without one, the value is the rest.
fn read_parameters(body: &str) -> (Option<Vec<(String, Value)>>, usize, bool) {
    let mut arguments = Vec::new();
    let mut at = 0;
    loop {
        let spaced = at + (body[at..].len() - python_strip_start(&body[at..]).len());
        let rest = &body[spaced..];
        if rest.is_empty() {
            return (Some(arguments), body.len(), false);
        }
        if let Some(after) = rest.strip_prefix("</function>") {
            return (Some(arguments), body.len() - after.len(), true);
        }
        if rest.starts_with("</tool_call>") {
            // (the function's close left out)
            return (Some(arguments), spaced, true);
        }
        let Some(open) = rest.strip_prefix("<parameter=") else {
            return match rest.find("</function>") {
                Some(close) => (None, spaced + close + "</function>".len(), true),
                None => (None, body.len(), false),
            };
        };
        let Some(key_end) = open.find('>') else { return (None, body.len(), false) };
        let key = &open[..key_end];
        if key.is_empty() || key.chars().any(python_space) {
            return (None, body.len(), false);
        }
        let value_start = spaced + "<parameter=".len() + key_end + 1;
        let mut from = value_start;
        let end = loop {
            let Some(close) = body[from..].find("</parameter>") else { break None };
            let close = from + close;
            let past = close + "</parameter>".len();
            let next = python_strip_start(&body[past..]);
            if next.is_empty() || ["<parameter=", "</function>", "</tool_call>"].iter().any(|tag| next.starts_with(tag)) {
                break Some((close, past));
            }
            from = past;
        };
        let Some((close, past)) = end else {
            let value = template_value(&body[value_start..]);
            arguments.push((key.to_string(), Value::Str(value.to_string())));
            return (Some(arguments), body.len(), false);
        };
        arguments.push((key.to_string(), Value::Str(template_value(&body[value_start..close]).to_string())));
        at = past;
    }
}

/// A block holding `{"name": ..., "arguments": {...}}` (arguments as an object, or a string of one), as Hermes-style chat
/// templates write it.
fn read_json_block(text: &str, start: usize) -> Block {
    let close = text.find("</tool_call>");
    let json = python_strip(&text[start..close.unwrap_or(text.len())]);
    let call = pyjson::parse(json).ok().and_then(|value| {
        let name = value.get("name")?.as_str().filter(|name| !name.is_empty() && !name.contains('>') && !name.chars().any(python_space))?;
        let arguments = match value.get("arguments").or_else(|| value.get("parameters"))? {
            Value::Object(fields) => fields.clone(),
            Value::Str(encoded) => match pyjson::parse(encoded).ok()? {
                Value::Object(fields) => fields,
                _ => return None,
            },
            _ => return None,
        };
        Some((name.to_string(), arguments))
    });
    match close {
        Some(close) => Block::Closed(call, close + "</tool_call>".len()),
        None => Block::Open(call),
    }
}

fn python_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

fn python_strip_start(text: &str) -> &str {
    text.trim_start_matches(python_space)
}

/// The text from `at` to the first of `stops`: (index after the stop, text before it, the stop).
fn read_until<'a>(text: &'a str, at: usize, stops: &[&'static str]) -> (usize, &'a str, Option<&'static str>) {
    let found = stops.iter().filter_map(|stop| text[at..].find(stop).map(|position| (at + position, *stop))).min_by_key(|(position, _)| *position);
    match found {
        Some((position, stop)) => (position + stop.len(), &text[at..position], Some(stop)),
        None => (text.len(), &text[at..], None),
    }
}

/// DSML tool calls from `\n\n<｜DSML｜ calls` to the end of the text, and the invokes dropped.
/// Each invoke the reference's parser reads is a call: a string parameter's value is the text
/// itself, any other's is JSON (one that does not parse goes as a string). An invoke it would
/// refuse (the model's slip: a mismatched tag, a name used twice) is dropped and the calls around
/// it kept: the reference fails the whole reply, and its well-formed calls then reached the client
/// as answer text, which ended an agent's run. None when no invoke reads.
fn parse_dsml(markup: &str) -> Option<(Vec<ToolCall>, usize)> {
    let body = markup.strip_prefix(DSML_CALLS_OPEN)?;
    let body = body.strip_prefix(">\n")?;
    // The calls end at their close, after which only whitespace may follow (the end-of-sentence
    // token is not text); without one (an answer cut short) at the end.
    let body = match body.find(DSML_CALLS_CLOSE) {
        Some(close) if python_strip(&body[close + DSML_CALLS_CLOSE.len()..]).is_empty() => &body[..close],
        Some(_) => return None,
        None => body,
    };
    let starts: Vec<usize> = body.match_indices(DSML_INVOKE_OPEN).map(|(at, _)| at).collect();
    if starts.first() != Some(&0) {
        return None;
    }
    let (mut calls, mut dropped) = (Vec::new(), 0);
    for (index, &start) in starts.iter().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(body.len());
        match parse_invoke(&body[start + DSML_INVOKE_OPEN.len()..end]) {
            Some((namespace, name, arguments)) => calls.push(ToolCall {
                index: calls.len(),
                id: format!("call_{}", random_hex(24)),
                name,
                arguments,
                namespace,
            }),
            None => dropped += 1,
        }
    }
    (!calls.is_empty()).then_some((calls, dropped))
}

/// One invoke after its `<｜DSML｜ invoke` (up to the next invoke or the calls' close): its
/// namespace, name and JSON arguments, or None when the reference's parser would refuse it.
fn parse_invoke(invoke: &str) -> Option<(Option<String>, String, String)> {
    let (next, header, mut stop) = read_until(invoke, 0, &[DSML_PARAMETER_OPEN, DSML_INVOKE_CLOSE]);
    let mut at = next;
    let qualified = DSML_NAME.captures(header)?.get(1)?.as_str();
    let mut arguments: Vec<String> = Vec::new();
    let mut names: Vec<&str> = Vec::new();
    while stop == Some(DSML_PARAMETER_OPEN) {
        let (next, parameter, _) = read_until(invoke, at, &[DSML_PARAMETER_CLOSE]);
        at = next;
        let fields = DSML_PARAMETER.captures(parameter)?;
        let (name, string, value) = (fields.get(1)?.as_str(), &fields[2], fields.get(3)?.as_str());
        if names.contains(&name) {
            return None;
        }
        names.push(name);
        // A non-string value that is not JSON (the model's slip, such as an invalid escape) goes
        // as the string it is: the call still reaches the client, whose schema check then says
        // what is wrong, where the reference's parser would refuse the whole reply.
        let value = if string == "true" || pyjson::parse(value).is_err() {
            pyjson::dumps(&Value::Str(value.to_string()))
        } else {
            value.to_string()
        };
        arguments.push(format!("{}: {value}", pyjson::dumps(&Value::Str(name.to_string()))));
        let (next, between, after) = read_until(invoke, at, &[DSML_PARAMETER_OPEN, DSML_INVOKE_CLOSE]);
        if between != ">\n" {
            return None;
        }
        (at, stop) = (next, after);
    }
    // the invoke's close, then the line it ends
    if stop != Some(DSML_INVOKE_CLOSE) || invoke[at..] != *">\n" {
        return None;
    }
    let (namespace, name) = match qualified.split_once("::") {
        Some((namespace, name)) => (Some(namespace.to_string()), name),
        None => (None, qualified),
    };
    if name.contains("::") {
        return None;
    }
    Some((namespace, name.to_string(), format!("{{{}}}", arguments.join(", "))))
}

/// Undo the newline the template writes on each side of a value.
fn template_value(raw: &str) -> &str {
    let raw = raw.strip_prefix('\n').unwrap_or(raw);
    raw.strip_suffix('\n').unwrap_or(raw)
}

fn random_hex(digits: usize) -> String {
    let mut bytes = vec![0u8; digits.div_ceil(2)];
    getrandom::fill(&mut bytes).expect("the OS random source");
    bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>()[..digits].to_string()
}

/// Ends the text at the first stop string; a tail that could begin one across deltas is
/// held until the next delta decides it.
pub struct StopMatcher {
    stops: Vec<String>,
    carry: String,
    pub stopped: bool,
}

impl StopMatcher {
    pub fn new(stops: &[String]) -> StopMatcher {
        StopMatcher { stops: stops.iter().filter(|stop| !stop.is_empty()).cloned().collect(), carry: String::new(),
                      stopped: false }
    }

    /// The text before any stop string; after a match, nothing.
    pub fn feed(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        if self.stops.is_empty() {
            return text.to_string();
        }
        let mut buffer = std::mem::take(&mut self.carry) + text;
        if let Some(at) = self.stops.iter().filter_map(|stop| buffer.find(stop.as_str())).min() {
            self.stopped = true;
            buffer.truncate(at);
            return buffer;
        }
        let hold = self.stops.iter().map(|stop| split_tail(&buffer, stop)).max().unwrap_or(0);
        if hold > 0 {
            self.carry = buffer.split_off(buffer.len() - hold);
        }
        buffer
    }

    pub fn finish(&mut self) -> String {
        let carry = std::mem::take(&mut self.carry);
        if self.stopped { String::new() } else { carry }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_reasoning_ending_by_token_keeps_a_quoted_tag() {
        let mut splitter = ReasoningSplitter::with_markup(true, Markup::Dsml).ending_by_token();
        assert_eq!(splitter.feed("Notes: after `</think>`; tool_choice"), (String::new(), "Notes: after `</think>`; tool_choice".to_string()));
        assert_eq!(splitter.feed("</th"), (String::new(), "</th".to_string()));
        assert_eq!(splitter.end(), "");
        assert_eq!(splitter.feed("The answer."), ("The answer.".to_string(), String::new()));
        // (by text, as before: the quoted tag ends it)
        let mut by_text = ReasoningSplitter::with_markup(true, Markup::Dsml);
        assert_eq!(by_text.feed("after `</think>`; x").0, "`; x");
    }

    use super::*;

    fn split(deltas: &[&str], thinking: bool, tools: Option<&Value>) -> (String, String, Vec<ToolCall>) {
        let (mut reasoning, mut calls) = (ReasoningSplitter::new(thinking), ToolCallSplitter::new(tools));
        let (mut thoughts, mut answer) = (String::new(), String::new());
        for delta in deltas {
            let (content, thought) = reasoning.feed(delta);
            thoughts += &thought;
            answer += &calls.feed(&content);
        }
        thoughts += &reasoning.finish();
        let (late, calls) = calls.finish();
        (thoughts, answer + &late, calls)
    }

    fn read_file() -> Value {
        pyjson::parse(r#"[{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}}}]"#).unwrap()
    }

    #[test]
    fn reasoning_ends_only_at_a_line_start_close_tag() {
        let (thoughts, answer, _) = split(&["Let me think", " about it.\n</th", "ink>\n\n", "Hello", " there"], true, None);
        assert_eq!(thoughts.trim_end(), "Let me think about it.");
        assert_eq!(answer, "Hello there");
        let deltas = ["It renders `</think>` mid-line,", " and </think> again.\n", "</think>\n\nThe answer", "."];
        let (thoughts, answer, _) = split(&deltas, true, None);
        assert!(thoughts.ends_with("and </think> again.\n"));
        assert_eq!(answer, "The answer.");
        let (thoughts, answer, _) = split(&["cut off at </thi"], true, None);
        assert_eq!((thoughts.as_str(), answer.as_str()), ("cut off at </thi", ""));
    }

    #[test]
    fn tool_call_markup_becomes_structured_calls() {
        let tools = read_file();
        let deltas = ["Let me look.\n\n<tool", "_call>\n<function=read_file>\n<parameter=path>\n123\n</param",
                      "eter>\n<parameter=limit>\n50\n</parameter>\n</function>\n</tool_call>"];
        let (_, answer, calls) = split(&deltas, false, Some(&tools));
        assert_eq!(answer, "Let me look.\n\n");
        assert_eq!(calls.len(), 1);
        assert!(calls[0].id.starts_with("call_") && calls[0].id.len() == 29);
        assert_eq!(calls[0].arguments, r#"{"path": "123", "limit": 50}"#);
        let quoted = ["Qwen wraps calls in <tool_call> blocks.\n", "<tool_call>\n<function=read_file>\n</function>\n</tool_call>"];
        let (_, answer, calls) = split(&quoted, false, Some(&tools));
        assert_eq!((answer.as_str(), calls.len()), ("Qwen wraps calls in <tool_call> blocks.\n", 1));
        let bad = "Example: <tool_call><function=read_file>oops</tool_call> done";
        let (_, answer, calls) = split(&[bad], false, Some(&tools));
        assert_eq!((answer.as_str(), calls.len()), (bad, 0));
        let markup = "<tool_call>\n<function=read_file>\n</function>\n</tool_call>";
        let (_, answer, calls) = split(&[markup], false, None);
        assert_eq!((answer.as_str(), calls.len()), (markup, 0));
    }

    /// Finish a reply's markup: (answer text, calls, the splitter).
    fn repaired(markup: &str, cut: bool) -> (String, Vec<ToolCall>, ToolCallSplitter) {
        let tools = read_file();
        let mut splitter = ToolCallSplitter::new(Some(&tools));
        splitter.cut = cut;
        assert_eq!(splitter.feed(markup), "");
        let (text, calls) = splitter.finish();
        (text, calls, splitter)
    }

    const CALL: &str = "<tool_call>\n<function=read_file>\n<parameter=path>\na.rs\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn a_well_formed_call_needs_no_repair() {
        let (text, calls, splitter) = repaired(&format!("{CALL}\n{CALL}\n"), false);
        assert!(text.is_empty() && calls.len() == 2 && !splitter.repaired && !splitter.unread);
    }

    #[test]
    fn prose_after_or_between_the_calls_is_the_answers() {
        let (text, calls, splitter) = repaired(&format!("{CALL}\nI will now read it."), false);
        assert_eq!((text.as_str(), calls.len()), ("I will now read it.", 1));
        assert_eq!(calls[0].arguments, r#"{"path": "a.rs", "limit": 5}"#);
        assert!(splitter.repaired);
        let (text, calls, _) = repaired(&format!("{CALL}\nThen:\n{CALL}"), false);
        assert_eq!((text.as_str(), calls.len(), calls[1].index), ("Then:", 2, 1));
    }

    #[test]
    fn a_close_left_out_is_read_when_the_model_ended_the_reply() {
        let open = "<tool_call>\n<function=read_file>\n<parameter=path>\na.rs\n</parameter>\n";
        for markup in [format!("{open}</function>\n"), format!("{open}</tool_call>"), open.to_string(),
                       "<tool_call>\n<function=read_file>\n<parameter=path>\na.rs\n".to_string()] {
            let (text, calls, _) = repaired(&markup, false);
            assert!(text.is_empty() && calls.len() == 1 && calls[0].arguments == r#"{"path": "a.rs"}"#, "{markup:?} {text:?} {calls:?}");
        }
        let (_, calls, _) = repaired(&format!("{open}</function>\n{CALL}"), false);
        assert_eq!(calls.len(), 2);
    }

    /// A call the token limit cut short must not run as if it were whole: a write of half a file.
    #[test]
    fn a_call_cut_short_stays_text() {
        let open = "<tool_call>\n<function=read_file>\n<parameter=path>\nhalf of a";
        let (text, calls, splitter) = repaired(open, true);
        assert!(calls.is_empty() && splitter.unread);
        assert_eq!(text, open);
    }

    /// A value that holds markup is a value, or the reply is text: never a call cut short at the markup it quotes.
    #[test]
    fn a_value_holding_markup_is_not_cut_at_it() {
        let quoted = "<tool_call>\n<function=read_file>\n<parameter=path>\nthe </parameter> tag\n</parameter>\n</function>\n</tool_call>";
        let (text, calls, _) = repaired(quoted, false);
        assert!(text.is_empty() && calls[0].arguments == r#"{"path": "the </parameter> tag"}"#, "{text:?} {calls:?}");
        // a whole example in a value: the strict reading ends at its close, and what is left closes nothing it opened
        let example = "<tool_call>\n<function=read_file>\n<parameter=path>\n<tool_call>\n<function=f>\n</function>\n</tool_call>\n</parameter>\n</function>\n</tool_call>";
        let (text, calls, splitter) = repaired(example, false);
        assert!(calls.is_empty() && splitter.unread && text == example, "{text:?} {calls:?}");
        let tail = "<tool_call>\n<function=read_file>\n<parameter=path>\nx\n</parameter>\n</function>\n</tool_call>\nmore\n</parameter>";
        let (text, calls, _) = repaired(tail, false);
        assert!(calls.is_empty() && text == tail);
    }

    #[test]
    fn an_unreadable_block_among_others_is_dropped() {
        let bad = "<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>";
        let (text, calls, splitter) = repaired(&format!("{bad}\n{CALL}"), false);
        assert!(text.is_empty() && calls.len() == 1 && splitter.dropped == 1, "{text:?} {calls:?}");
        let junk = "<tool_call>\n<function=read_file>\nnot a parameter\n</function>\n</tool_call>";
        let (_, calls, splitter) = repaired(&format!("{junk}\n{CALL}"), false);
        assert!(calls.len() == 1 && calls[0].index == 0 && splitter.dropped == 1);
        let (text, calls, splitter) = repaired(junk, false);
        assert!(calls.is_empty() && splitter.unread && text == junk, "no call: the text as it was");
    }

    #[test]
    fn a_json_block_is_a_call() {
        let block = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\", \"limit\": 5}}\n</tool_call>";
        let (text, calls, _) = repaired(block, false);
        assert!(text.is_empty() && calls[0].name == "read_file" && calls[0].arguments == r#"{"path": "a.rs", "limit": 5}"#, "{calls:?}");
        let encoded = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": \"{\\\"path\\\": \\\"a.rs\\\"}\"}\n</tool_call>";
        let (_, calls, _) = repaired(encoded, false);
        assert_eq!(calls[0].arguments, r#"{"path": "a.rs"}"#);
        let (_, calls, splitter) = repaired("<tool_call>\n{\"nope\": 1}\n</tool_call>", false);
        assert!(calls.is_empty() && splitter.unread);
    }

    fn split_dsml(deltas: &[&str], thinking: bool, tools: Option<&Value>) -> (String, String, Vec<ToolCall>) {
        let (mut reasoning, mut calls) =
            (ReasoningSplitter::with_markup(thinking, Markup::Dsml), ToolCallSplitter::with_markup(tools, Markup::Dsml));
        let (mut thoughts, mut answer) = (String::new(), String::new());
        for delta in deltas {
            let (content, thought) = reasoning.feed(delta);
            thoughts += &thought;
            answer += &calls.feed(&content);
        }
        thoughts += &reasoning.finish();
        let (late, calls) = calls.finish();
        (thoughts, answer + &late, calls)
    }

    /// The reference parser's (DeepSeek-V4.1's encoding.py) results for the same completions.
    #[test]
    fn dsml_replies_split_as_the_reference_parses_them() {
        let (thoughts, answer, calls) = split_dsml(&["Simple arith", "metic.</th", "ink>2 + 2", " = 4."], true, None);
        assert_eq!((thoughts.as_str(), answer.as_str(), calls.len()), ("Simple arithmetic.", "2 + 2 = 4.", 0));
        let tools = read_file();
        let text = "Let me check.</think>I'll look it up.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read_file\">\n\
                    <｜DSML｜ parameter name=\"path\" string=\"true\">a \"b\"\nc</｜DSML｜ parameter>\n\
                    <｜DSML｜ parameter name=\"limit\" string=\"false\">50</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
                    <｜DSML｜ invoke name=\"ns::lookup\">\n<｜DSML｜ parameter name=\"q\" string=\"false\">{\"x\": [1, 2]}\
                    </｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        // Every split point: the deltas a stream could bring.
        for cut in (0..text.len()).filter(|cut| text.is_char_boundary(*cut)).step_by(7) {
            let (thoughts, answer, calls) = split_dsml(&[&text[..cut], &text[cut..]], true, Some(&tools));
            assert_eq!((thoughts.as_str(), answer.as_str()), ("Let me check.", "I'll look it up."), "cut at {cut}");
            assert_eq!(calls.len(), 2);
            assert_eq!((calls[0].name.as_str(), calls[0].namespace.as_deref()), ("read_file", None));
            assert_eq!(calls[0].arguments, r#"{"path": "a \"b\"\nc", "limit": 50}"#);
            assert_eq!((calls[1].name.as_str(), calls[1].namespace.as_deref()), ("lookup", Some("ns")));
            assert_eq!(calls[1].arguments, r#"{"q": {"x": [1, 2]}}"#);
        }
        let text = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"now\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let (_, answer, calls) = split_dsml(&[text], false, Some(&tools));
        assert_eq!((answer.as_str(), calls.len(), calls[0].arguments.as_str()), ("", 1, "{}"));
        // Without tools, or not parsing (a value that is not JSON), the markup stays text.
        let (_, answer, calls) = split_dsml(&[text], false, None);
        assert_eq!((answer.as_str(), calls.len()), (text, 0));
        // A non-string value that is not JSON goes as the string it is (the client's schema check
        // then reports it), where the reference refuses the reply.
        let bad = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"x\" string=\"false\">oops\
                   </｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let (_, answer, calls) = split_dsml(&["Answer.", bad], false, Some(&tools));
        assert_eq!((answer.as_str(), calls.len(), calls[0].arguments.as_str()), ("Answer.", 1, r#"{"x": "oops"}"#));
        // Only the first </think> ends reasoning; a later one is the answer's text.
        let (thoughts, answer, _) = split_dsml(&["a</think>b</think>c"], true, None);
        assert_eq!((thoughts.as_str(), answer.as_str()), ("a", "b</think>c"));
    }

    /// A malformed invoke (the model's slip, seen from DeepSeek-V4.1 IQ2_XS in an agent's run) is
    /// dropped and the well-formed calls around it kept.
    #[test]
    fn a_malformed_dsml_invoke_drops_alone() {
        let tools = pyjson::parse(r#"[{"type": "function", "function": {"name": "read_file", "parameters": {"properties": {"path": {"type": "string"}}}}}]"#).unwrap();
        let slipped = "Let me read it.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read_file\">\n\
                       <｜DSML｜ parameter name=\"limit\" string=\"false\">80</｜DSML｜ parameter>\n\
                       <｜DSML｜ parameter name=\"path\" string=\"true\">src/a.py</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
                       <｜DSML｜ invoke name=\"note\">placeholder</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let mut splitter = ToolCallSplitter::with_markup(Some(&tools), Markup::Dsml);
        let answer = splitter.feed(slipped);
        let (late, calls) = splitter.finish();
        assert_eq!((answer.as_str(), late.as_str(), splitter.dropped), ("Let me read it.", "", 1));
        assert_eq!((calls.len(), calls[0].name.as_str(), calls[0].arguments.as_str()),
                   (1, "read_file", r#"{"limit": 80, "path": "src/a.py"}"#));
        // A slip between two good calls keeps both, numbered in order.
        let good = "<｜DSML｜ invoke name=\"read_file\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">b</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n";
        let text = format!("\n\n<｜DSML｜ calls>\n{good}<｜DSML｜ invoke name=\"x\">\n<｜DSML｜ parameter name=\"y\" string=\"false\">1</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"y\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n{good}</｜DSML｜ calls>");
        let (_, answer, calls) = split_dsml(&[&text], false, Some(&tools));
        assert_eq!((answer.as_str(), calls.len(), calls[1].index), ("", 2, 1));
        // The slip seen next: an invalid escape in an array argument; the call still comes, its value
        // a string.
        let escape = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read_file\">\n<｜DSML｜ parameter name=\"q\" string=\"false\">\
                      [{\"p\": \"\\]\"}]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let (_, answer, calls) = split_dsml(&[escape], false, Some(&tools));
        assert_eq!((answer.as_str(), calls.len(), calls[0].arguments.as_str()), ("", 1, r#"{"q": "[{\"p\": \"\\]\"}]"}"#));
        // Text after the calls' close still makes the whole block text.
        let trailing = format!("\n\n<｜DSML｜ calls>\n{good}</｜DSML｜ calls>\nmore");
        let (_, answer, calls) = split_dsml(&[&trailing], false, Some(&tools));
        assert_eq!((answer, calls.len()), (trailing.clone(), 0));
    }

    #[test]
    fn stop_strings_end_the_text_across_deltas() {
        let mut stop = StopMatcher::new(&["END".into(), "</x>".into()]);
        assert_eq!(stop.feed("abc E"), "abc ");
        assert_eq!(stop.feed("N"), "");
        assert_eq!(stop.feed("Dmore"), "");
        assert!(stop.stopped && stop.finish().is_empty());
        let mut stop = StopMatcher::new(&["END".into()]);
        assert_eq!(stop.feed("hello E"), "hello ");
        assert_eq!(stop.finish(), "E");
    }
}
