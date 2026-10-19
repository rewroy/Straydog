//! JSON exactly as the reference reads and writes it: Python's `json.loads` and
//! `json.dumps(value, ensure_ascii=False)`.
//!
//! A prompt is rendered from parsed requests, so how numbers are normalized
//! (`1e-3` becomes `0.001`), how keys are ordered and how strings are escaped all reach the
//! tokens. Differences from Python, each an error instead: integers beyond 128 bits, and
//! lone surrogates (Python parses them, but no prompt can carry one).

use std::fmt::Write;

/// A parsed value. Objects keep their keys in first-insertion order; a repeated key
/// replaces the value in place, as a Python dict does.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(entries) => entries.iter().find(|(name, _)| name == key).map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(text) => Some(text),
            _ => None,
        }
    }

    /// Python's truthiness (`bool(value)`).
    pub fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(value) => *value,
            Value::Int(value) => *value != 0,
            Value::Float(value) => *value != 0.0,
            Value::Str(text) => !text.is_empty(),
            Value::Array(items) => !items.is_empty(),
            Value::Object(entries) => !entries.is_empty(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: &'static str,
    pub position: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.message, self.position)
    }
}

impl std::error::Error for ParseError {}

/// Nesting deeper than this is rejected (Python's limit is its recursion limit).
const MAX_DEPTH: usize = 512;

pub fn parse(text: &str) -> Result<Value, ParseError> {
    let mut parser = Parser { bytes: text.as_bytes(), text, at: 0 };
    parser.whitespace();
    let value = parser.value(0)?;
    parser.whitespace();
    if parser.at != parser.bytes.len() {
        return Err(parser.error("extra data"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    text: &'a str,
    at: usize,
}

impl Parser<'_> {
    fn error(&self, message: &'static str) -> ParseError {
        ParseError { message, position: self.at }
    }

    fn whitespace(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn literal(&mut self, word: &str) -> bool {
        if self.text[self.at..].starts_with(word) {
            self.at += word.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, ParseError> {
        if depth > MAX_DEPTH {
            return Err(self.error("nesting too deep"));
        }
        match self.bytes.get(self.at) {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Value::Str),
            Some(b'-' | b'0'..=b'9') => {
                if self.literal("-Infinity") {
                    return Ok(Value::Float(f64::NEG_INFINITY));
                }
                self.number()
            }
            _ if self.literal("null") => Ok(Value::Null),
            _ if self.literal("true") => Ok(Value::Bool(true)),
            _ if self.literal("false") => Ok(Value::Bool(false)),
            _ if self.literal("NaN") => Ok(Value::Float(f64::NAN)),
            _ if self.literal("Infinity") => Ok(Value::Float(f64::INFINITY)),
            _ => Err(self.error("expecting value")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.at += 1;
        let mut entries: Vec<(String, Value)> = Vec::new();
        self.whitespace();
        if self.bytes.get(self.at) == Some(&b'}') {
            self.at += 1;
            return Ok(Value::Object(entries));
        }
        loop {
            self.whitespace();
            if self.bytes.get(self.at) != Some(&b'"') {
                return Err(self.error("expecting property name enclosed in double quotes"));
            }
            let key = self.string()?;
            self.whitespace();
            if self.bytes.get(self.at) != Some(&b':') {
                return Err(self.error("expecting ':' delimiter"));
            }
            self.at += 1;
            self.whitespace();
            let value = self.value(depth + 1)?;
            match entries.iter_mut().find(|(name, _)| *name == key) {
                Some(entry) => entry.1 = value,
                None => entries.push((key, value)),
            }
            self.whitespace();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::Object(entries));
                }
                _ => return Err(self.error("expecting ',' delimiter")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.at += 1;
        let mut items = Vec::new();
        self.whitespace();
        if self.bytes.get(self.at) == Some(&b']') {
            self.at += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.whitespace();
            items.push(self.value(depth + 1)?);
            self.whitespace();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.error("expecting ',' delimiter")),
            }
        }
    }

    /// `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?`: an int without a fraction or exponent.
    fn number(&mut self) -> Result<Value, ParseError> {
        let start = self.at;
        let digits = |parser: &mut Self| {
            let begin = parser.at;
            while matches!(parser.bytes.get(parser.at), Some(b'0'..=b'9')) {
                parser.at += 1;
            }
            parser.at - begin
        };
        if self.bytes.get(self.at) == Some(&b'-') {
            self.at += 1;
        }
        match self.bytes.get(self.at) {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => return Err(self.error("expecting value")),
        }
        let mut float = false;
        if self.bytes.get(self.at) == Some(&b'.') && matches!(self.bytes.get(self.at + 1), Some(b'0'..=b'9')) {
            self.at += 1;
            digits(self);
            float = true;
        }
        if matches!(self.bytes.get(self.at), Some(b'e' | b'E')) {
            let mark = self.at;
            self.at += 1;
            if matches!(self.bytes.get(self.at), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if digits(self) == 0 {
                self.at = mark;   // not an exponent: the number ends before it
            } else {
                float = true;
            }
        }
        let text = &self.text[start..self.at];
        if float {
            // Correctly rounded, as Python's float(); out of range is infinity, as there.
            return Ok(Value::Float(text.parse().map_err(|_| self.error("invalid number"))?));
        }
        text.parse::<i128>().map(Value::Int).map_err(|_| self.error("integer beyond 128 bits"))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.at += 1;
        let mut out = String::new();
        loop {
            let start = self.at;
            while let Some(&byte) = self.bytes.get(self.at) {
                if byte == b'"' || byte == b'\\' || byte < 0x20 {
                    break;
                }
                self.at += 1;
            }
            out.push_str(&self.text[start..self.at]);
            match self.bytes.get(self.at) {
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let escape = *self.bytes.get(self.at).ok_or_else(|| self.error("unterminated string"))?;
                    self.at += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let code = if (0xd800..0xdc00).contains(&first) && self.text[self.at..].starts_with("\\u") {
                                let mark = self.at;
                                self.at += 2;
                                let second = self.hex4()?;
                                if (0xdc00..0xe000).contains(&second) {
                                    0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
                                } else {
                                    self.at = mark;
                                    first
                                }
                            } else {
                                first
                            };
                            out.push(char::from_u32(code).ok_or_else(|| self.error("lone surrogate"))?);
                        }
                        _ => return Err(self.error("invalid \\escape")),
                    }
                }
                Some(_) => return Err(self.error("invalid control character")),
                None => return Err(self.error("unterminated string")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let digits = self.text.get(self.at..self.at + 4).ok_or_else(|| self.error("invalid \\uXXXX escape"))?;
        if !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(self.error("invalid \\uXXXX escape"));
        }
        self.at += 4;
        Ok(u32::from_str_radix(digits, 16).unwrap())
    }
}

/// `json.dumps(value, ensure_ascii=False)`.
pub fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out
}

fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(number) => write!(out, "{number}").unwrap(),
        Value::Float(number) => out.push_str(&float_repr(*number)),
        Value::Str(text) => write_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item);
            }
            out.push('}');
        }
    }
}

/// A JSON string as Python writes it without ASCII escaping: only `"`, `\` and control characters.
pub fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            control if (control as u32) < 0x20 => write!(out, "\\u{:04x}", control as u32).unwrap(),
            other => out.push(other),
        }
    }
    out.push('"');
}

/// Python's `repr(float)` as JSON writes it: the shortest digits that round-trip, positional
/// from 1e-4 up to 1e16, otherwise with an exponent of at least two digits.
pub fn float_repr(number: f64) -> String {
    if number.is_nan() {
        return "NaN".into();
    }
    if number.is_infinite() {
        return if number > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if number == 0.0 {
        return if number.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust's shortest round-trip digits: "d.ddde±x".
    let scientific = format!("{:e}", number.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let point = exponent + 1;   // the value is 0.<digits> x 10^point
    let mut out = String::from(if number < 0.0 { "-" } else { "" });
    if point > -4 && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-point) as usize));
            out.push_str(&digits);
        } else if point as usize >= digits.len() {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', point as usize - digits.len()));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        write!(out, "e{}{:02}", if exponent < 0 { '-' } else { '+' }, exponent.abs()).unwrap();
    }
    out
}
