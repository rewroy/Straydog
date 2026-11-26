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
/// The closes of an invoke and of a parameter, whole (the pieces of a call being written read up to their `>`).
const DSML_INVOKE_END: &str = "</｜DSML｜ invoke>";
const DSML_PARAMETER_END: &str = "</｜DSML｜ parameter>";

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
            "qwen4exp" => Some(Markup::Qwen),
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
/// An invoke's name line; one with no parameters is closed after a blank line, as the template writes it (its parameter lines, none, joined between two
/// line ends), and so as the model writes it.
static DSML_NAME: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?s)^[\s\x1c-\x1f]*name="(.*?)">\n\n?$"#).unwrap());
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
    /// Bytes of markup [`ToolCallSplitter::finish`] left out of the text: a call the reply was cut in, which is neither run nor
    /// shown (as text it reached the client's screen and its next request, the model's own markup half written).
    pub left_out: usize,
    /// The calls as they are being written ([`ToolCallSplitter::writing`]); none unless asked for.
    writer: Option<CallWriter>,
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
            left_out: 0,
            writer: None,
        }
    }

    /// The same, also reading the markup as it arrives for the calls being written ([`ToolCallSplitter::take_pieces`]). What
    /// [`ToolCallSplitter::feed`] and [`ToolCallSplitter::finish`] return is the same with or without it.
    pub fn writing(mut self) -> ToolCallSplitter {
        if self.enabled {
            self.writer = Some(CallWriter::new(self.format));
        }
        self
    }

    /// The pieces of the calls being written that the text fed since the last call made known (none unless
    /// [`ToolCallSplitter::writing`]). Display data: a repair, a dropped block or markup released as text at the end changes
    /// nothing in what was shown, and what was shown changes nothing in the calls [`ToolCallSplitter::finish`] returns.
    pub fn take_pieces(&mut self) -> Vec<ToolPiece> {
        self.writer.as_mut().map(|writer| std::mem::take(&mut writer.pieces)).unwrap_or_default()
    }

    /// The answer text safe to show for one delta.
    pub fn feed(&mut self, text: &str) -> String {
        if !self.enabled {
            return text.to_string();
        }
        if let Some(markup) = self.markup.as_mut() {
            markup.push_str(text);
            if let Some(writer) = self.writer.as_mut() {
                writer.feed(text, &self.string_parameters);
            }
            return String::new();
        }
        let mut buffer = std::mem::take(&mut self.carry) + text;
        let (open, start) = match self.format {
            Markup::Qwen => (TOOL_CALL_OPEN, line_tag(&buffer, TOOL_CALL_OPEN, self.line_start)),
            Markup::Dsml => (DSML_CALLS_OPEN, buffer.find(DSML_CALLS_OPEN)),
        };
        if let Some(start) = start {
            let markup = buffer.split_off(start);
            if let Some(writer) = self.writer.as_mut() {
                writer.feed(&markup, &self.string_parameters);
            }
            self.markup = Some(markup);
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
            None if self.cut => {
                self.left_out = markup.len();
                (carry, Vec::new())
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
        // (a reply cut short may stop inside a value quoting closes, with the real ones never written: nothing shows what is
        // whole, so only the strict reading makes calls of it)
        if self.cut {
            return None;
        }
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
                Block::Open(..) if !last => return None,
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

/// A piece of a tool call being written, as far as the markup read so far shows ([`ToolCallSplitter::writing`]): part of its
/// arguments JSON. A call's pieces, joined, are what the markup read so far makes of the arguments the call has once whole
/// (`pyjson::dumps` of its object: the same key order, separators and escaping): the whole of them when the markup is well
/// formed, and otherwise not guaranteed to be JSON, or to match. Display data, never the call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolPiece {
    /// The call's place among the calls the markup opened (a block dropped for a slip is counted here, not in the final calls).
    pub index: usize,
    /// The tool's name: on the call's first piece only, which carries no arguments (the name is known as soon as the
    /// `<function=NAME>` is whole, before any parameter text).
    pub name: Option<String>,
    /// The next part of the arguments JSON.
    pub arguments: String,
}

/// Where [`CallWriter`] is in the markup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    /// Qwen: past a block, or before the first, looking for the next `<tool_call>` that opens a line.
    Between,
    /// Qwen: past `<tool_call>`, before its `<function=NAME>`.
    Block,
    /// DSML: the calls' open.
    Calls,
    /// DSML: between invokes.
    Invoke,
    /// Inside a call, between its parameters.
    Body,
    /// A string parameter's value, shown as it arrives.
    Text,
    /// Any other parameter's value, shown whole once it has ended.
    Whole,
    /// Markup that is no call it can follow (a block of JSON: the rare Hermes form shows nothing before the call is whole; a slip):
    /// nothing is shown up to the block's close.
    Skip,
    /// Nothing more is shown.
    Done,
}

/// Where a parameter's value ends in the text held.
enum End {
    /// The value is `..value` and the markup goes on at `after`.
    Closed { value: usize, after: usize },
    /// No end yet: the first `sure` bytes are the value's, the rest may still be its close.
    Open { sure: usize },
}

/// Reads tool-call markup as it arrives, without reading it again: a call's name when its opener is whole, then its arguments
/// JSON, a string parameter's text as it comes (escaped) and any other parameter's value once it has ended. Only what could still
/// be a close (and, for Qwen, the one newline the template writes before it) is held, so a piece never has to be taken back.
struct CallWriter {
    format: Markup,
    part: Part,
    /// Markup received and not yet read: a tag perhaps split across deltas, or a value's held tail.
    pending: String,
    /// The text before `pending` ended a line (Qwen: a `<tool_call>` opens a block only there).
    line_start: bool,
    /// Calls opened so far.
    started: usize,
    /// Parameters begun in the call in hand.
    parameters: usize,
    /// The tool of the call in hand.
    tool: String,
    /// The call in hand's current parameter, `, "key": ` as its JSON writes it: shown with a value that is shown whole.
    key: String,
    /// Qwen: a string value's one leading newline (the template's) is still to be read away.
    lead: bool,
    /// The bytes of `pending` that hold no close of a value read whole.
    checked: usize,
    pieces: Vec<ToolPiece>,
}

/// A tag at the start of `text`: whole, the start of one still to come, or not there.
fn tag_at(text: &str, tag: &str) -> Option<bool> {
    if text.starts_with(tag) {
        Some(true)
    } else if tag.starts_with(text) {
        Some(false)
    } else {
        None
    }
}

/// A string as JSON writes it, without its quotes.
fn escaped(text: &str) -> String {
    let mut quoted = String::with_capacity(text.len() + 2);
    pyjson::write_string(&mut quoted, text);
    quoted[1..quoted.len() - 1].to_string()
}

/// A DSML parameter's tag past `<｜DSML｜ parameter`, up to its `>`: (name, whether it is a string).
fn dsml_parameter(head: &str) -> Option<(&str, bool)> {
    let (name, flag) = head.strip_prefix(" name=\"")?.split_once("\" string=\"")?;
    match flag.strip_suffix('"')? {
        "true" => Some((name, true)),
        "false" => Some((name, false)),
        _ => None,
    }
}

impl CallWriter {
    fn new(format: Markup) -> CallWriter {
        CallWriter {
            format,
            part: if format == Markup::Qwen { Part::Between } else { Part::Calls },
            pending: String::new(),
            line_start: true,
            started: 0,
            parameters: 0,
            tool: String::new(),
            key: String::new(),
            lead: false,
            checked: 0,
            pieces: Vec::new(),
        }
    }

    fn qwen(&self) -> bool {
        self.format == Markup::Qwen
    }

    fn feed(&mut self, text: &str, tools: &[(String, Vec<String>)]) {
        self.pending.push_str(text);
        while self.step(tools) {}
    }

    /// Reads what `pending` holds in the part in hand: whether that moved on, so that there may be more to read.
    fn step(&mut self, tools: &[(String, Vec<String>)]) -> bool {
        match self.part {
            Part::Done => {
                self.pending.clear();
                false
            }
            Part::Skip => self.skip(),
            Part::Between => self.between(),
            Part::Block => self.block(),
            Part::Calls => self.calls(),
            Part::Invoke => self.invoke(),
            Part::Body => self.body(tools),
            Part::Text => self.text(),
            Part::Whole => self.whole(),
        }
    }

    /// The next part of the call in hand's arguments, joined to the piece before it when that is one of the same call's
    /// arguments (the name's piece carries none).
    fn show(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let index = self.started - 1;
        match self.pieces.last_mut() {
            Some(last) if last.index == index && last.name.is_none() => last.arguments.push_str(text),
            _ => self.pieces.push(ToolPiece { index, name: None, arguments: text.to_string() }),
        }
    }

    /// A call opens: its name is shown at once, as a piece of its own.
    fn open(&mut self, name: &str) {
        self.pieces.push(ToolPiece { index: self.started, name: Some(name.to_string()), arguments: String::new() });
        self.started += 1;
        self.parameters = 0;
        self.tool = name.to_string();
        self.show("{");
    }

    /// A call ends; the markup goes on with the next block or invoke.
    fn close(&mut self) {
        self.show("}");
        (self.part, self.line_start) = (if self.qwen() { Part::Between } else { Part::Invoke }, true);
    }

    /// Drops what is read, keeping the tail of `pending` that could still be the start of `tag`.
    fn keep_tail(&mut self, tag: &str) {
        let drop = self.pending.len() - split_tail(&self.pending, tag);
        self.pending.drain(..drop);
    }

    /// Qwen: to the next `<tool_call>` that opens a line.
    fn between(&mut self) -> bool {
        if let Some(at) = line_tag(&self.pending, TOOL_CALL_OPEN, self.line_start) {
            self.pending.drain(..at + TOOL_CALL_OPEN.len());
            self.part = Part::Block;
            return true;
        }
        let drop = self.pending.len() - split_tail(&self.pending, TOOL_CALL_OPEN);
        if drop > 0 {
            self.line_start = self.pending[..drop].ends_with('\n');
            self.pending.drain(..drop);
        }
        false
    }

    /// Qwen: past `<tool_call>`, the `<function=NAME>` that opens the call.
    fn block(&mut self) -> bool {
        let rest = python_strip_start(&self.pending);
        let skipped = self.pending.len() - rest.len();
        let Some(after) = rest.strip_prefix("<function=") else {
            // (a block of JSON, or a slip: nothing is shown of it before the call is whole)
            if "<function=".starts_with(rest) {
                self.pending.drain(..skipped);
                return false;
            }
            // (a block of JSON is a call of the final answer: it keeps its place among them)
            if rest.starts_with('{') {
                self.started += 1;
            }
            self.part = Part::Skip;
            return true;
        };
        let name = match after.find('>') {
            Some(end) => Some((&after[..end], end)),
            // (a name has no whitespace, and is not thousands of characters long)
            None if after.len() > 4096 || after.chars().any(python_space) => None,
            None => {
                self.pending.drain(..skipped);
                return false;
            }
        };
        match name {
            Some((name, end)) if !name.is_empty() && !name.chars().any(python_space) => {
                let name = name.to_string();
                self.pending.drain(..skipped + "<function=".len() + end + 1);
                self.open(&name);
                self.part = Part::Body;
            }
            _ => self.part = Part::Skip,
        }
        true
    }

    /// DSML: the calls' open, `>` and the line it ends.
    fn calls(&mut self) -> bool {
        match self.pending.strip_prefix(DSML_CALLS_OPEN) {
            Some(after) if after.starts_with('>') => {
                self.pending.drain(..DSML_CALLS_OPEN.len() + 1);
                self.part = Part::Invoke;
            }
            Some(after) if after.is_empty() => return false,
            _ => self.part = Part::Done,
        }
        true
    }

    /// DSML: the next invoke, `<｜DSML｜ invoke name="NAME">`, or the calls' close.
    fn invoke(&mut self) -> bool {
        let rest = python_strip_start(&self.pending);
        let skipped = self.pending.len() - rest.len();
        let (opens, closes) = (tag_at(rest, DSML_INVOKE_OPEN), tag_at(rest, DSML_CALLS_CLOSE));
        if closes == Some(true) {
            self.part = Part::Done;
            return true;
        }
        let Some(after) = rest.strip_prefix(DSML_INVOKE_OPEN) else {
            if opens == Some(false) || closes == Some(false) {
                self.pending.drain(..skipped);
                return false;
            }
            self.part = Part::Done;
            return true;
        };
        let Some(end) = after.find('>') else {
            if after.len() > 4096 {
                self.part = Part::Done;
                return true;
            }
            self.pending.drain(..skipped);
            return false;
        };
        let name = python_strip_start(&after[..end]).strip_prefix("name=\"").and_then(|name| name.strip_suffix('"'));
        match name {
            Some(qualified) => {
                // (the namespace is the call's own field, not part of its name)
                let name = qualified.split_once("::").map_or(qualified, |(_, name)| name).to_string();
                self.pending.drain(..skipped + DSML_INVOKE_OPEN.len() + end + 1);
                self.open(&name);
                self.part = Part::Body;
            }
            None => self.part = Part::Skip,
        }
        true
    }

    /// Qwen: a function's body, DSML: an invoke's: its next parameter, or its close.
    fn body(&mut self, tools: &[(String, Vec<String>)]) -> bool {
        let rest = python_strip_start(&self.pending);
        let skipped = self.pending.len() - rest.len();
        let qwen = self.qwen();
        let (end, open) = if qwen { ("</function>", "<parameter=") } else { (DSML_INVOKE_END, DSML_PARAMETER_OPEN) };
        if rest.starts_with(end) {
            self.pending.drain(..skipped + end.len());
            self.close();
            return true;
        }
        if qwen && rest.starts_with("</tool_call>") {
            // (the function's close left out)
            self.pending.drain(..skipped);
            self.show("}");
            self.part = Part::Skip;
            return true;
        }
        let Some(after) = rest.strip_prefix(open) else {
            if end.starts_with(rest) || open.starts_with(rest) || (qwen && "</tool_call>".starts_with(rest)) {
                self.pending.drain(..skipped);
                return false;
            }
            self.part = Part::Skip;
            return true;
        };
        let Some(stop) = after.find('>') else {
            // (a Qwen key has no whitespace; a DSML tag is short)
            if (qwen && after.chars().any(python_space)) || after.len() > 4096 {
                self.part = Part::Skip;
                return true;
            }
            self.pending.drain(..skipped);
            return false;
        };
        let head = &after[..stop];
        let parameter = if qwen {
            let strings = tools.iter().rev().find(|(tool, _)| *tool == self.tool).map(|(_, keys)| keys);
            (!head.is_empty() && !head.chars().any(python_space))
                .then(|| (head, strings.is_some_and(|keys| keys.iter().any(|key| key == head))))
        } else {
            dsml_parameter(head)
        };
        let Some((key, string)) = parameter else {
            self.part = Part::Skip;
            return true;
        };
        self.key = format!("{}{}: ", if self.parameters > 0 { ", " } else { "" }, pyjson::dumps(&Value::Str(key.to_string())));
        self.pending.drain(..skipped + open.len() + stop + 1);
        self.parameters += 1;
        self.checked = 0;
        self.lead = string && qwen;
        if string {
            let key = std::mem::take(&mut self.key);
            self.show(&format!("{key}\""));
        }
        self.part = if string { Part::Text } else { Part::Whole };
        true
    }

    /// Where the value ends in `pending`, its first `from` bytes known to hold no end. Qwen's `</parameter>` ends it only where
    /// the markup goes on with a tag or the text ends (as `read_parameters` reads it); one before other text is the value's own.
    fn value_end(&self, from: usize) -> End {
        let text = &self.pending;
        if !self.qwen() {
            return match text[from..].find(DSML_PARAMETER_END) {
                Some(at) => End::Closed { value: from + at, after: from + at + DSML_PARAMETER_END.len() },
                None => End::Open { sure: text.len() - split_tail(&text[from..], DSML_PARAMETER_END) },
            };
        }
        const CLOSE: &str = "</parameter>";
        const NEXT: [&str; 3] = ["<parameter=", "</function>", "</tool_call>"];
        let mut from = from;
        loop {
            let Some(found) = text[from..].find(CLOSE) else {
                return End::Open { sure: text.len() - split_tail(&text[from..], CLOSE) };
            };
            let (close, after) = (from + found, from + found + CLOSE.len());
            let next = python_strip_start(&text[after..]);
            if NEXT.iter().any(|tag| next.starts_with(tag)) {
                return End::Closed { value: close, after };
            }
            // (the text ends, or ends inside the tag that would decide)
            if NEXT.iter().any(|tag| tag.starts_with(next)) {
                return End::Open { sure: close };
            }
            from = after;
        }
    }

    /// A string parameter's value, up to what could still be its close.
    fn text(&mut self) -> bool {
        let qwen = self.qwen();
        if self.lead {
            if self.pending.is_empty() {
                return false;
            }
            if self.pending.starts_with('\n') {
                self.pending.remove(0);
            }
            self.lead = false;
        }
        // (Qwen's template writes a newline before the close, which is not the value's: held until more follows it)
        let own = |raw: &str| if qwen { raw.strip_suffix('\n').unwrap_or(raw).len() } else { raw.len() };
        match self.value_end(0) {
            End::Closed { value, after } => {
                let shown = escaped(&self.pending[..own(&self.pending[..value])]);
                self.pending.drain(..after);
                self.show(&(shown + "\""));
                self.part = Part::Body;
                true
            }
            End::Open { sure } => {
                let shown = own(&self.pending[..sure]);
                let text = escaped(&self.pending[..shown]);
                self.pending.drain(..shown);
                self.show(&text);
                false
            }
        }
    }

    /// Any other parameter's value, whole, as the final arguments write it.
    fn whole(&mut self) -> bool {
        match self.value_end(self.checked) {
            End::Closed { value, after } => {
                let raw = &self.pending[..value];
                let value = if self.qwen() { template_value(raw) } else { raw };
                let shown = match pyjson::parse(value) {
                    // (a DSML value that is JSON stays as the model wrote it)
                    Ok(parsed) if self.qwen() => pyjson::dumps(&parsed),
                    Ok(_) => value.to_string(),
                    Err(_) => pyjson::dumps(&Value::Str(value.to_string())),
                };
                self.pending.drain(..after);
                let key = std::mem::take(&mut self.key);
                self.show(&(key + &shown));
                self.part = Part::Body;
                true
            }
            End::Open { sure } => {
                self.checked = sure;
                false
            }
        }
    }

    /// Markup no call is read from: dropped up to the block's close.
    fn skip(&mut self) -> bool {
        let close = if self.qwen() { "</tool_call>" } else { DSML_INVOKE_END };
        let Some(at) = self.pending.find(close) else {
            self.keep_tail(close);
            return false;
        };
        self.pending.drain(..at + close.len());
        (self.part, self.line_start) = (if self.qwen() { Part::Between } else { Part::Invoke }, true);
        true
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

    /// A call the token limit cut short must not run as if it were whole (a write of half a file), nor reach the answer's text as
    /// markup: it is left out.
    #[test]
    fn a_call_cut_short_is_neither_run_nor_text() {
        let open = "<tool_call>\n<function=read_file>\n<parameter=path>\nhalf of a";
        let (text, calls, splitter) = repaired(open, true);
        assert!(calls.is_empty() && !splitter.unread && text.is_empty());
        assert_eq!(splitter.left_out, open.len());
        // cut inside a value quoting closes: what precedes the quote must not run as the whole value
        for quote in ["</parameter>\n</function>\nmore of the file", "</parameter>\n</function>\n</tool_call>\nmore of the file"] {
            let quoted = format!("<tool_call>\n<function=read_file>\n<parameter=path>\nfirst half\n{quote}");
            let (text, calls, splitter) = repaired(&quoted, true);
            assert!(calls.is_empty() && text.is_empty() && splitter.left_out == quoted.len(), "{text:?} {calls:?}");
        }
        // the same markup, not cut, is the answer's text as before
        let (text, calls, splitter) = repaired(open, false);
        assert!(calls.len() == 1 || (splitter.unread && text == open), "{text:?} {calls:?}");
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
        // A call with no parameters as the template writes one, and so as the model learned it: a blank line inside the invoke. A client's
        // load_workspace_dependencies came back as answer text this way, which ended its turn.
        let blank = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"load_workspace_dependencies\">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let (_, answer, calls) = split_dsml(&[blank], false, Some(&tools));
        assert_eq!((answer.as_str(), calls.len(), calls[0].name.as_str(), calls[0].arguments.as_str()), ("", 1, "load_workspace_dependencies", "{}"));
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

    // The calls as they are written (`ToolCallSplitter::writing`).

    fn create_file() -> Value {
        pyjson::parse(r#"[{"type": "function", "function": {"name": "create_file", "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "executable": {"type": "boolean"},
                           "lines": {"type": "array"}}}}}]"#).unwrap()
    }

    /// `deltas` through a splitter that shows the calls being written: the pieces after each delta, the answer text and the
    /// calls at the end.
    fn write_calls(format: Markup, deltas: &[&str], tools: &Value) -> (Vec<Vec<ToolPiece>>, String, Vec<ToolCall>) {
        let mut splitter = ToolCallSplitter::with_markup(Some(tools), format).writing();
        let (mut answer, mut seen) = (String::new(), Vec::new());
        for delta in deltas {
            answer += &splitter.feed(delta);
            seen.push(splitter.take_pieces());
        }
        let (late, calls) = splitter.finish();
        assert!(splitter.take_pieces().is_empty());
        (seen, answer + &late, calls)
    }

    /// What the pieces show of each call: (name, arguments joined). A call's name comes first and alone.
    fn joined(pieces: &[Vec<ToolPiece>]) -> Vec<(String, String)> {
        let mut calls: Vec<(String, String)> = Vec::new();
        for piece in pieces.iter().flatten() {
            if let Some(name) = &piece.name {
                assert_eq!((piece.index, piece.arguments.as_str()), (calls.len(), ""), "a call's name comes first, with no arguments");
                calls.push((name.clone(), String::new()));
            } else {
                assert!(!piece.arguments.is_empty());
                calls.get_mut(piece.index).expect("arguments follow the call's name").1 += &piece.arguments;
            }
        }
        calls
    }

    /// Every way a stream could cut `text` into two deltas, and into single characters.
    fn cuts(text: &str) -> Vec<Vec<&str>> {
        let mut all: Vec<Vec<&str>> = (0..=text.len()).filter(|cut| text.is_char_boundary(*cut)).map(|cut| vec![&text[..cut], &text[cut..]]).collect();
        all.push(text.char_indices().map(|(at, c)| &text[at..at + c.len_utf8()]).collect());
        all
    }

    /// Whatever the stream's cuts, the pieces joined are the call's arguments, to the character, and at every delta a prefix of them.
    fn assert_written(format: Markup, markup: &str, tools: &Value, names: &[&str], answer: &str) -> Vec<ToolCall> {
        let mut last = Vec::new();
        for deltas in cuts(markup) {
            let (seen, text, calls) = write_calls(format, &deltas, tools);
            assert_eq!(text, answer, "{deltas:?}");
            assert_eq!(calls.iter().map(|call| call.name.as_str()).collect::<Vec<_>>(), names, "{deltas:?}");
            for upto in 1..=seen.len() {
                for ((_, shown), call) in joined(&seen[..upto]).iter().zip(&calls) {
                    assert!(call.arguments.starts_with(shown.as_str()), "{shown:?} is no start of {:?}", call.arguments);
                }
            }
            let shown = joined(&seen);
            assert_eq!(shown.len(), calls.len(), "{deltas:?}");
            for ((name, arguments), call) in shown.iter().zip(&calls) {
                assert_eq!((name, arguments), (&call.name, &call.arguments), "{deltas:?}");
            }
            last = calls;
        }
        last
    }

    #[test]
    fn the_pieces_of_a_well_formed_call_make_its_arguments() {
        let tools = create_file();
        let content = "fn main() {\n    println!(\"héllo \\ wörld\");\n}\n\n\ttab </parameter> not a close, nor </param\n";
        let markup = format!("I will write it.\n\n<tool_call>\n<function=create_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n\
                              <parameter=content>\n{content}\n</parameter>\n<parameter=executable>\ntrue\n</parameter>\n\
                              <parameter=lines>\n[1,  2]\n</parameter>\n</function>\n</tool_call>");
        let calls = assert_written(Markup::Qwen, &markup, &tools, &["create_file"], "I will write it.\n\n");
        let arguments = pyjson::parse(&calls[0].arguments).unwrap();
        assert_eq!(arguments.get("content").and_then(Value::as_str), Some(content));
        assert!(calls[0].arguments.ends_with(r#", "executable": true, "lines": [1, 2]}"#), "{}", calls[0].arguments);
    }

    #[test]
    fn two_calls_are_written_in_turn() {
        let tools = create_file();
        let one = "<tool_call>\n<function=create_file>\n<parameter=path>\na.rs\n</parameter>\n<parameter=content>\nfn a() {}\n</parameter>\n</function>\n</tool_call>";
        let two = "<tool_call>\n<function=create_file>\n<parameter=content>\n\n\nb\n</parameter>\n<parameter=path>\nb.rs\n</parameter>\n</function>\n</tool_call>";
        let calls = assert_written(Markup::Qwen, &format!("{one}\n{two}\n"), &tools, &["create_file", "create_file"], "");
        assert_eq!(calls[1].arguments, r#"{"content": "\n\nb", "path": "b.rs"}"#);
        // (a call with no parameters is `{}`)
        let empty = "<tool_call>\n<function=read_file>\n</function>\n</tool_call>";
        let calls = assert_written(Markup::Qwen, empty, &read_file(), &["read_file"], "");
        assert_eq!(calls[0].arguments, "{}");
    }

    /// What is held back at the end of a delta is what could still be a close, and the one newline before it.
    #[test]
    fn what_could_still_be_a_close_is_held_back() {
        let tools = create_file();
        let head = "<tool_call>\n<function=create_file>\n<parameter=path>\nsrc/a.rs\n</parameter>\n<parameter=content>";
        let args = |seen: &[Vec<ToolPiece>], delta: usize| -> String {
            seen[delta].iter().filter(|piece| piece.name.is_none()).map(|piece| piece.arguments.as_str()).collect()
        };
        let first = format!("{head}\nline one\nline two\n</para");
        let (seen, _, calls) = write_calls(Markup::Qwen, &[&first, "meter>\n</function>"], &tools);
        // the name alone, then the arguments: the template's newline after `<parameter=content>` is never sent, nor the one
        // before the close it holds, nor the start of the close
        assert_eq!(seen[0][0], ToolPiece { index: 0, name: Some("create_file".into()), arguments: String::new() });
        assert_eq!(args(&seen, 0), r#"{"path": "src/a.rs", "content": "line one\nline two"#);
        assert_eq!(args(&seen, 1), r#""}"#);
        assert_eq!(calls[0].arguments, r#"{"path": "src/a.rs", "content": "line one\nline two"}"#);
        // the value's own newlines: held until more follows, then sent
        let (seen, _, calls) = write_calls(Markup::Qwen, &[&format!("{head}\nab\n"), "\ncd\n</parameter>", "\n</function>"], &tools);
        assert_eq!(args(&seen, 0), r#"{"path": "src/a.rs", "content": "ab"#);
        assert_eq!(args(&seen, 1), r#"\n\ncd"#);
        assert_eq!(args(&seen, 2), r#""}"#);
        assert_eq!(calls[0].arguments, r#"{"path": "src/a.rs", "content": "ab\n\ncd"}"#);
        // nothing of the value before its first character
        let (seen, ..) = write_calls(Markup::Qwen, &[&format!("{head}\n")], &tools);
        assert_eq!(args(&seen, 0), r#"{"path": "src/a.rs", "content": ""#);
        // the name is shown with the opener, before any parameter
        let (seen, ..) = write_calls(Markup::Qwen, &["Text.\n<tool_call>\n<function=create_f"], &tools);
        assert!(seen[0].is_empty());
        let (seen, ..) = write_calls(Markup::Qwen, &["Text.\n<tool_call>\n<function=create_file>\n<paramet"], &tools);
        assert_eq!(seen[0], [ToolPiece { index: 0, name: Some("create_file".into()), arguments: String::new() },
                             ToolPiece { index: 0, name: None, arguments: "{".into() }]);
    }

    /// A `</parameter>` the markup does not go on from is the value's text, held until the markup shows which it is.
    #[test]
    fn a_close_the_markup_does_not_go_on_from_is_the_values_text() {
        let tools = create_file();
        let head = "<tool_call>\n<function=create_file>\n<parameter=content>\n";
        let tail = "\n</parameter>\n</function>\n</tool_call>";
        // (undecided after the close: nothing of it is sent; text follows, and it is the value's)
        let (seen, _, calls) = write_calls(Markup::Qwen, &[&format!("{head}the </parameter>"), " tag", tail], &tools);
        let shown: Vec<String> = seen.iter().map(|pieces| pieces.iter().filter(|p| p.name.is_none()).map(|p| p.arguments.as_str()).collect()).collect();
        assert_eq!(shown, [r#"{"content": "the "#, "</parameter> tag", r#""}"#]);
        assert_eq!(calls[0].arguments, r#"{"content": "the </parameter> tag"}"#);
        // (a tag that goes on as a parameter's: it was the close)
        let markup = "<tool_call>\n<function=create_file>\n<parameter=content>\nthe </parameter>\n<parameter=path>\np\n</parameter>\n</function>\n</tool_call>";
        let calls = assert_written(Markup::Qwen, markup, &tools, &["create_file"], "");
        assert_eq!(calls[0].arguments, r#"{"content": "the ", "path": "p"}"#);
        // (a close before other text, however often)
        let markup = "<tool_call>\n<function=create_file>\n<parameter=content>\nx </parameter>  y </parameter>\n</parameter>\n</function>\n</tool_call>";
        let calls = assert_written(Markup::Qwen, markup, &tools, &["create_file"], "");
        assert_eq!(calls[0].arguments, r#"{"content": "x </parameter>  y </parameter>"}"#);
    }

    #[test]
    fn a_parameter_that_is_not_a_string_is_sent_whole_once_it_ends() {
        let tools = create_file();
        let (seen, _, calls) = write_calls(Markup::Qwen, &[
            "<tool_call>\n<function=create_file>\n<parameter=path>\na\n</parameter>\n<parameter=lines>\n[1,", " 2,\n", "3]\n</param",
            "eter>\n<parameter=executable>\nfalse", "\n</parameter>\n</function>"], &tools);
        let shown: Vec<String> = seen.iter().map(|pieces| pieces.iter().filter(|p| p.name.is_none()).map(|p| p.arguments.as_str()).collect()).collect();
        assert_eq!(shown, [r#"{"path": "a""#, "", "", r#", "lines": [1, 2, 3]"#, r#", "executable": false}"#]);
        assert_eq!(calls[0].arguments, r#"{"path": "a", "lines": [1, 2, 3], "executable": false}"#);
        // a value that is no JSON goes as the string it is, as in the final arguments
        let markup = "<tool_call>\n<function=create_file>\n<parameter=lines>\nnot json\n</parameter>\n</function>\n</tool_call>";
        let calls = assert_written(Markup::Qwen, markup, &tools, &["create_file"], "");
        assert_eq!(calls[0].arguments, r#"{"lines": "not json"}"#);
        // (a tool the request does not list: every value is read as JSON where it can be)
        let markup = "<tool_call>\n<function=unknown>\n<parameter=a>\n5\n</parameter>\n<parameter=b>\nfive\n</parameter>\n</function>\n</tool_call>";
        let calls = assert_written(Markup::Qwen, markup, &tools, &["unknown"], "");
        assert_eq!(calls[0].arguments, r#"{"a": 5, "b": "five"}"#);
    }

    /// A block of JSON is a call only once it is whole: nothing is shown before. A slip shows what it showed and changes nothing.
    #[test]
    fn a_json_block_and_a_slip_show_no_call() {
        let tools = read_file();
        let block = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\", \"limit\": 5}}\n</tool_call>";
        let (seen, _, calls) = write_calls(Markup::Qwen, &cuts(block).pop().unwrap(), &tools);
        assert!(seen.iter().all(|pieces| pieces.is_empty()));
        assert_eq!(calls[0].arguments, r#"{"path": "a.rs", "limit": 5}"#);
        // (the call after it still has the place in the final calls it will have)
        let both = format!("{block}\n{CALL}");
        let (seen, _, calls) = write_calls(Markup::Qwen, &[&both], &tools);
        assert_eq!(calls.len(), 2);
        assert_eq!(seen[0][0], ToolPiece { index: 1, name: Some("read_file".into()), arguments: String::new() });
    }

    /// Whatever is shown, the splitter's answer text and calls are what they are without it.
    #[test]
    fn showing_the_calls_changes_nothing_in_the_answer() {
        let tools = read_file();
        let open = "<tool_call>\n<function=read_file>\n<parameter=path>\na.rs\n</parameter>\n";
        let slip = "<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>";
        let json = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>";
        let quoted = "<tool_call>\n<function=read_file>\n<parameter=path>\nthe </parameter> tag\n</parameter>\n</function>\n</tool_call>";
        let cases = [
            format!("Text.\n{CALL}\n{CALL}\n"), format!("{CALL}\nI will now read it."), format!("{CALL}\nThen:\n{CALL}"),
            format!("{open}</function>\n"), format!("{open}</tool_call>"), open.to_string(), format!("{slip}\n{CALL}"), slip.to_string(),
            json.to_string(), format!("{json}\n{CALL}"), quoted.to_string(), "<tool_call>\n<function=read_file>\nnot a parameter\n</function>\n</tool_call>".into(),
            "Example: <tool_call><function=read_file>oops</tool_call> done".into(), "Qwen wraps calls in <tool_call> blocks.\n".into(),
            "Text.\n<tool_call>\n<function=read_file>\n<parameter=path>\nhalf of a".to_string(),
        ];
        for text in &cases {
            for cut in [false, true] {
                for deltas in cuts(text) {
                    let run = |writing: bool| {
                        let splitter = ToolCallSplitter::new(Some(&tools));
                        let mut splitter = if writing { splitter.writing() } else { splitter };
                        splitter.cut = cut;
                        let mut answer: String = deltas.iter().map(|delta| splitter.feed(delta)).collect();
                        let (late, calls) = splitter.finish();
                        answer += &late;
                        let calls: Vec<_> = calls.into_iter().map(|call| (call.index, call.name, call.arguments)).collect();
                        (answer, calls, splitter.dropped, splitter.repaired, splitter.unread, splitter.take_pieces().len())
                    };
                    let (shown, plain) = (run(true), run(false));
                    assert_eq!(shown.0, plain.0, "{text:?} {deltas:?}");
                    assert_eq!(shown.1, plain.1, "{text:?} {deltas:?}");
                    assert_eq!((shown.2, shown.3, shown.4, plain.5), (plain.2, plain.3, plain.4, 0), "{text:?} {deltas:?}");
                }
            }
        }
    }

    /// A call is read once as it arrives: a file of 24,000 characters in 6,000 deltas, whole in its pieces.
    #[test]
    fn a_long_call_is_read_as_it_arrives() {
        let tools = create_file();
        let content: String = (0..1000).map(|line| format!("    let value_{line} = \"text\\n\"; // </p>\n")).collect();
        assert!(content.len() > 24_000);
        let markup = format!("<tool_call>\n<function=create_file>\n<parameter=path>\nbig.rs\n</parameter>\n<parameter=content>\n{content}\n</parameter>\n</function>\n</tool_call>");
        let deltas = cuts(&markup).pop().unwrap();
        let deltas: Vec<String> = deltas.chunks(4).map(|chars| chars.concat()).collect();
        let deltas: Vec<&str> = deltas.iter().map(String::as_str).collect();
        let (seen, _, calls) = write_calls(Markup::Qwen, &deltas, &tools);
        assert_eq!(joined(&seen)[0].1, calls[0].arguments);
        // (the value arrives as it is written: most deltas carry a piece)
        assert!(seen.iter().filter(|pieces| !pieces.is_empty()).count() > deltas.len() * 3 / 4);
    }

    #[test]
    fn dsml_calls_are_written_too() {
        let tools = create_file();
        let markup = "Looking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"ns::create_file\">\n\
                      <｜DSML｜ parameter name=\"path\" string=\"true\">a \"b\"\nc</｜DSML｜ parameter>\n\
                      <｜DSML｜ parameter name=\"lines\" string=\"false\">[1, 2]</｜DSML｜ parameter>\n\
                      <｜DSML｜ parameter name=\"content\" string=\"true\">fn main() {}\n\n</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
                      <｜DSML｜ invoke name=\"now\">\n\n</｜DSML｜ invoke>\n\
                      <｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"x\" string=\"false\">oops</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
                      </｜DSML｜ calls>";
        let calls = assert_written(Markup::Dsml, markup, &tools, &["create_file", "now", "f"], "Looking.");
        assert_eq!(calls[0].arguments, "{\"path\": \"a \\\"b\\\"\\nc\", \"lines\": [1, 2], \"content\": \"fn main() {}\\n\\n\"}");
        assert_eq!((calls[0].namespace.as_deref(), calls[1].arguments.as_str(), calls[2].arguments.as_str()), (Some("ns"), "{}", r#"{"x": "oops"}"#));
    }

    #[test]
    fn nothing_is_shown_without_asking() {
        let tools = read_file();
        let mut splitter = ToolCallSplitter::new(Some(&tools));
        assert_eq!(splitter.feed(&format!("Let me look.\n\n{CALL}")), "Let me look.\n\n");
        assert!(splitter.take_pieces().is_empty());
        // (and none for text that holds no tools)
        let mut splitter = ToolCallSplitter::new(None).writing();
        assert_eq!(splitter.feed(CALL), CALL);
        assert!(splitter.take_pieces().is_empty());
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
