//! The official template's generated text split as an OpenAI answer: reasoning up to
//! `</think>`, then the answer, then optional `<tool_call>` markup turned into structured
//! calls. Stop strings end the text first.

use std::sync::LazyLock;

use regex::Regex;

use crate::chat::python_strip;
use crate::pyjson::{self, Value};

const THINK_CLOSE: &str = "</think>";
const TOOL_CALL_OPEN: &str = "<tool_call>";
// Python's \s: Unicode whitespace and the separators U+001C-U+001F.
static TOOL_CALL_BLOCK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>[\s\x1c-\x1f]*<function=([^>\s\x1c-\x1f]+)>(.*?)</function>[\s\x1c-\x1f]*</tool_call>")
        .unwrap()
});
static TOOL_PARAMETER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<parameter=([^>\s\x1c-\x1f]+)>(.*?)</parameter>").unwrap());

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
}

impl ReasoningSplitter {
    pub fn new(thinking: bool) -> ReasoningSplitter {
        // The prompt ends with "<think>\n".
        ReasoningSplitter { reasoning: thinking, carry: String::new(), answer_lead: true, line_start: true }
    }

    /// (answer, reasoning) for one delta.
    pub fn feed(&mut self, text: &str) -> (String, String) {
        if !self.reasoning {
            return (self.answer(text), String::new());
        }
        let mut buffer = std::mem::take(&mut self.carry) + text;
        if let Some(end) = line_tag(&buffer, THINK_CLOSE, self.line_start) {
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
}

/// The template's tool-call markup held back from the answer and parsed at the end. Only
/// an opener at a line start counts; markup that does not parse is released as text.
pub struct ToolCallSplitter {
    string_parameters: Vec<(String, Vec<String>)>,
    enabled: bool,
    carry: String,
    markup: Option<String>,
    line_start: bool,
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
        if let Some(start) = line_tag(&buffer, TOOL_CALL_OPEN, self.line_start) {
            self.markup = Some(buffer.split_off(start));
            return buffer;
        }
        let hold = split_tail(&buffer, TOOL_CALL_OPEN);
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
        match self.parse(&markup) {
            Some(calls) => (carry, calls),
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
            });
        }
        Some(calls)
    }
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
