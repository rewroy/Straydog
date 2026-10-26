//! A family's generated text split as an OpenAI answer: reasoning up to `</think>`, then the
//! answer, then optional tool-call markup turned into structured calls ([`Markup`]: Qwen's
//! `<tool_call>` blocks or DeepSeek-V4.1's DSML). Stop strings end the text first.

use std::sync::LazyLock;

use regex::Regex;

use crate::chat::python_strip;
use crate::pyjson::{self, Value};

const THINK_CLOSE: &str = "</think>";
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
}

impl ReasoningSplitter {
    pub fn new(thinking: bool) -> ReasoningSplitter {
        ReasoningSplitter::with_markup(thinking, Markup::Qwen)
    }

    pub fn with_markup(thinking: bool, markup: Markup) -> ReasoningSplitter {
        // The prompt ends with "<think>\n" (Qwen) or "<think>" (DSML).
        ReasoningSplitter { reasoning: thinking, carry: String::new(), answer_lead: markup == Markup::Qwen, line_start: true, markup }
    }

    /// (answer, reasoning) for one delta.
    pub fn feed(&mut self, text: &str) -> (String, String) {
        if !self.reasoning {
            return (self.answer(text), String::new());
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
    /// Malformed tool calls [`ToolCallSplitter::finish`] dropped (DSML's: see `parse_dsml`).
    pub dropped: usize,
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
            Markup::Qwen => self.parse(&markup).map(|calls| (calls, 0)),
            Markup::Dsml => parse_dsml(&markup),
        };
        match parsed {
            Some((calls, dropped)) => {
                self.dropped = dropped;
                (carry, calls)
            }
            None => (carry + &markup, Vec::new()),
        }
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
