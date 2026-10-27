//! Constrained decoding (docs/DESIGN.md §6): a reply's answer held to JSON (`response_format`)
//! or to a tool call (`tool_choice`), token by token while it is sampled.
//!
//! A [`Constraint`] knows each token id's bytes ([`TokenBytes`]). While the reply reasons it
//! only refuses an end token; after `</think>` (or from the start, without reasoning) it keeps
//! one or more [`Alternative`]s alive: a JSON value checked against a [`Schema`], or a prefix the
//! answer must start with (a tool call's opening), anything after it. A token is allowed when
//! some alternative takes all of its bytes, an end token when one is complete. Sampling draws
//! from the model's distribution and draws again without a refused token
//! ([`crate::sampling::choose_where`]), so the allowed tokens keep their relative probabilities.

use std::collections::HashMap;
use std::sync::Arc;

use crate::model::Rng;
use crate::pyjson::{self, Value};
use crate::sampling::{self, Generated, Sampling};

const THINK_CLOSE: &[u8] = b"</think>";
/// Whitespace in a row the JSON may hold: a pretty-printed line's indentation fits, a model
/// stuck emitting newlines does not.
const WHITESPACE_RUN: usize = 64;
/// Whitespace before the value starts, or after it ends: a model that wanted to write prose
/// otherwise padded the answer with lines of it (16 blank lines before a `{`).
const WHITESPACE_AROUND: usize = 4;
/// Characters a number may hold.
const NUMBER_LENGTH: usize = 40;
/// How deep a schema may nest, `$ref`s followed.
const SCHEMA_DEPTH: usize = 48;

/// The bytes each token id stands for, and whether it is a control token.
pub struct TokenBytes {
    bytes: Vec<Vec<u8>>,
    special: Vec<bool>,
    /// The template's `</think>` when it is one added token: only it ends a reasoning.
    think_close: Option<u32>,
}

impl TokenBytes {
    pub fn new(bytes: Vec<Vec<u8>>, special: Vec<bool>) -> TokenBytes {
        assert_eq!(bytes.len(), special.len());
        TokenBytes { bytes, special, think_close: None }
    }

    /// The same, with the vocabulary's `</think>` token: a reasoning ends at it, never at the same
    /// text written otherwise (the reasoning quoting the tag), as the worker's reasoning_end does.
    pub fn with_think_close(mut self, token: Option<u32>) -> TokenBytes {
        self.think_close = token;
        self
    }

    /// A token's bytes and whether it is a control token; None for an id no token has.
    fn get(&self, token: u32) -> Option<(&[u8], bool)> {
        let index = token as usize;
        let bytes = self.bytes.get(index).filter(|bytes| !bytes.is_empty())?;
        Some((bytes.as_slice(), self.special[index]))
    }
}

/// GPT-2's byte-level BPE alphabet, read back: the character each byte is written as.
pub fn byte_level_decoder() -> HashMap<char, u8> {
    let mut bytes: Vec<u32> = (u32::from(b'!')..=u32::from(b'~')).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut chars = bytes.clone();
    let mut extra = 0;
    for byte in 0..256u32 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    bytes.into_iter().zip(chars).map(|(byte, char)| (char::from_u32(char).unwrap(), byte as u8)).collect()
}

/// A JSON Schema, compiled: the subset structured outputs use (`type` and its lists, `properties`,
/// `required`, `additionalProperties`, `items`, `enum`, `const`, `anyOf`/`oneOf`, `$ref` into
/// `$defs`/`definitions`, recursion included). Keywords it does not hold to (`pattern`,
/// `minLength`, `format`, ...) are accepted and not enforced; an `anyOf` whose alternatives a
/// value's first byte does not tell apart is held to JSON alone there.
#[derive(Debug, Clone)]
pub struct Schema {
    nodes: Vec<Node>,
    root: usize,
}

#[derive(Debug, Clone)]
enum Node {
    Any,
    Object { properties: Vec<(Vec<u8>, usize)>, required: Vec<usize>, additional: Option<usize> },
    Array { items: usize },
    Str,
    Number { integer: bool },
    Bool,
    Null,
    /// `enum`/`const`: the values as JSON text.
    Literals(Vec<Vec<u8>>),
    Union(Vec<usize>),
}

/// Index of [`Node::Any`] in every schema.
const ANY: usize = 0;

/// An object's properties (names as JSON string bodies, with their nodes), the required ones'
/// indices, and the node of any other key (none: no other keys).
type ObjectShape<'a> = (&'a [(Vec<u8>, usize)], &'a [usize], Option<usize>);

impl Schema {
    /// `{"type": "json_object"}`: any JSON object.
    pub fn any_object() -> Schema {
        Schema { nodes: vec![Node::Any, Node::Object { properties: Vec::new(), required: Vec::new(), additional: Some(ANY) }], root: 1 }
    }

    pub fn compile(schema: &Value) -> Result<Schema, String> {
        let mut compiler = Compiler { root: schema, nodes: vec![Node::Any], refs: HashMap::new() };
        let root = compiler.node(schema, 0)?;
        Ok(Schema { nodes: compiler.nodes, root })
    }

    /// The node a value starting with `byte` is checked against, if one may start so.
    fn resolve(&self, node: usize, byte: u8) -> Option<usize> {
        match &self.nodes[node] {
            Node::Union(choices) => {
                let mut fits = choices.iter().filter_map(|&choice| self.resolve(choice, byte));
                let first = fits.next()?;
                Some(if fits.next().is_some() { ANY } else { first })
            }
            Node::Any => starts_value(byte).then_some(node),
            Node::Object { .. } => (byte == b'{').then_some(node),
            Node::Array { .. } => (byte == b'[').then_some(node),
            Node::Str => (byte == b'"').then_some(node),
            Node::Number { .. } => (byte == b'-' || byte.is_ascii_digit()).then_some(node),
            Node::Bool => (byte == b't' || byte == b'f').then_some(node),
            Node::Null => (byte == b'n').then_some(node),
            Node::Literals(values) => values.iter().any(|value| value.first() == Some(&byte)).then_some(node),
        }
    }

    /// An object node's properties, required ones and the schema of other keys (none: no others).
    fn object(&self, node: usize) -> ObjectShape<'_> {
        match &self.nodes[node] {
            Node::Object { properties, required, additional } => (properties, required, *additional),
            _ => (&[], &[], Some(ANY)),
        }
    }
}

fn starts_value(byte: u8) -> bool {
    matches!(byte, b'{' | b'[' | b'"' | b'-' | b't' | b'f' | b'n') || byte.is_ascii_digit()
}

struct Compiler<'a> {
    root: &'a Value,
    nodes: Vec<Node>,
    refs: HashMap<String, usize>,
}

impl Compiler<'_> {
    fn push(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn node(&mut self, schema: &Value, depth: usize) -> Result<usize, String> {
        if depth > SCHEMA_DEPTH {
            return Err("the JSON schema nests too deeply".into());
        }
        let fields = match schema {
            Value::Bool(true) => return Ok(ANY),
            Value::Object(fields) => fields,
            _ => return Err("a JSON schema must be an object".into()),
        };
        if fields.is_empty() {
            return Ok(ANY);
        }
        if let Some(reference) = schema.get("$ref") {
            let Some(reference) = reference.as_str() else { return Err("a JSON schema $ref must be a string".into()) };
            return self.reference(reference, depth);
        }
        if let Some(value) = schema.get("const") {
            return Ok(self.push(Node::Literals(vec![pyjson::dumps(value).into_bytes()])));
        }
        if let Some(values) = schema.get("enum") {
            let Value::Array(values) = values else { return Err("a JSON schema enum must be a list".into()) };
            if values.is_empty() {
                return Err("a JSON schema enum must not be empty".into());
            }
            return Ok(self.push(Node::Literals(values.iter().map(|value| pyjson::dumps(value).into_bytes()).collect())));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(choices) = schema.get(key) {
                let Value::Array(choices) = choices else { return Err(format!("a JSON schema {key} must be a list")) };
                let nodes = choices.iter().map(|choice| self.node(choice, depth + 1)).collect::<Result<Vec<_>, _>>()?;
                return Ok(self.push(Node::Union(nodes)));
            }
        }
        match schema.get("type") {
            None if schema.get("properties").is_some() => self.typed("object", schema, depth),
            None if schema.get("items").is_some() => self.typed("array", schema, depth),
            None => Ok(ANY),
            Some(Value::Str(kind)) => self.typed(kind, schema, depth),
            Some(Value::Array(kinds)) => {
                let mut nodes = Vec::new();
                for kind in kinds {
                    let kind = kind.as_str().ok_or("a JSON schema type must be a string")?;
                    nodes.push(self.typed(kind, schema, depth)?);
                }
                Ok(self.push(Node::Union(nodes)))
            }
            Some(_) => Err("a JSON schema type must be a string or a list".into()),
        }
    }

    fn typed(&mut self, kind: &str, schema: &Value, depth: usize) -> Result<usize, String> {
        Ok(match kind {
            "object" => {
                let mut properties = Vec::new();
                let mut names = Vec::new();
                if let Some(given) = schema.get("properties") {
                    let Value::Object(given) = given else { return Err("JSON schema properties must be an object".into()) };
                    for (name, property) in given {
                        let node = self.node(property, depth + 1)?;
                        properties.push((json_string_body(name), node));
                        names.push(name.as_str());
                    }
                }
                let required = match schema.get("required") {
                    None => Vec::new(),
                    Some(Value::Array(list)) => {
                        let mut required = Vec::new();
                        for name in list {
                            let index = name.as_str().and_then(|name| names.iter().position(|known| *known == name))
                                .ok_or("a JSON schema requires a property it does not define")?;
                            required.push(index);
                        }
                        required
                    }
                    Some(_) => return Err("JSON schema required must be a list".into()),
                };
                let additional = match schema.get("additionalProperties") {
                    None | Some(Value::Bool(true)) => Some(ANY),
                    Some(Value::Bool(false)) => None,
                    Some(other) => Some(self.node(other, depth + 1)?),
                };
                self.push(Node::Object { properties, required, additional })
            }
            "array" => {
                let items = match schema.get("items") {
                    None => ANY,
                    Some(items) => self.node(items, depth + 1)?,
                };
                self.push(Node::Array { items })
            }
            "string" => self.push(Node::Str),
            "number" => self.push(Node::Number { integer: false }),
            "integer" => self.push(Node::Number { integer: true }),
            "boolean" => self.push(Node::Bool),
            "null" => self.push(Node::Null),
            other => return Err(format!("JSON schema type {other} is not supported")),
        })
    }

    /// A `$ref` to `#/$defs/<name>` or `#/definitions/<name>`, compiled once (a recursive one
    /// refers to its own node).
    fn reference(&mut self, reference: &str, depth: usize) -> Result<usize, String> {
        if let Some(&node) = self.refs.get(reference) {
            return Ok(node);
        }
        let root = self.root;
        let target = ["$defs", "definitions"].into_iter().find_map(|section| {
            let name = reference.strip_prefix("#/")?.strip_prefix(section)?.strip_prefix('/')?;
            root.get(section)?.get(name)
        });
        let Some(target) = target else { return Err(format!("JSON schema $ref {reference} is not supported")) };
        let placeholder = self.push(Node::Any);
        self.refs.insert(reference.to_string(), placeholder);
        let compiled = self.node(target, depth + 1)?;
        self.nodes[placeholder] = self.nodes[compiled].clone();
        Ok(placeholder)
    }
}

/// A name as it appears between a JSON string's quotes.
fn json_string_body(name: &str) -> Vec<u8> {
    let quoted = pyjson::dumps(&Value::Str(name.to_string())).into_bytes();
    quoted[1..quoted.len() - 1].to_vec()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectState {
    KeyOrClose,
    Key,
    Colon,
    Value,
    CommaOrClose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayState {
    ValueOrClose,
    Value,
    CommaOrClose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberState {
    Sign,
    Zero,
    Int,
    Dot,
    Fraction,
    E,
    ExponentSign,
    Exponent,
}

impl NumberState {
    fn complete(self) -> bool {
        matches!(self, NumberState::Zero | NumberState::Int | NumberState::Fraction | NumberState::Exponent)
    }

    fn first(byte: u8) -> NumberState {
        match byte {
            b'-' => NumberState::Sign,
            b'0' => NumberState::Zero,
            _ => NumberState::Int,
        }
    }

    fn next(self, byte: u8, integer: bool) -> Option<NumberState> {
        Some(match (self, byte) {
            (NumberState::Sign, b'0') => NumberState::Zero,
            (NumberState::Sign | NumberState::Int, b'1'..=b'9') | (NumberState::Int, b'0') => NumberState::Int,
            (NumberState::Zero | NumberState::Int, b'.') if !integer => NumberState::Dot,
            (NumberState::Zero | NumberState::Int | NumberState::Fraction, b'e' | b'E') if !integer => NumberState::E,
            (NumberState::Dot | NumberState::Fraction, b'0'..=b'9') => NumberState::Fraction,
            (NumberState::E, b'+' | b'-') => NumberState::ExponentSign,
            (NumberState::E | NumberState::ExponentSign | NumberState::Exponent, b'0'..=b'9') => NumberState::Exponent,
            _ => return None,
        })
    }
}

/// String escapes: 0 none, 1 after a backslash, 2..=5 hex digits of `\u` left (5: four).
type Escape = u8;

#[derive(Debug, Clone)]
enum Frame {
    /// A value of the node, not started.
    Value(usize),
    Object { node: usize, state: ObjectState, seen: Vec<bool>, property: Option<usize> },
    /// An object's key: its raw bytes so far.
    Key { node: usize, buffer: Vec<u8>, escape: Escape },
    Array { items: usize, state: ArrayState },
    Str { escape: Escape },
    Number { integer: bool, state: NumberState, length: usize },
    /// Literal texts still possible, `at` bytes matched.
    Literal { choices: Vec<Vec<u8>>, at: usize },
}

enum Step {
    Taken,
    Refused,
    /// The frame's value ended before the byte, which its parent reads.
    Again,
}

/// One JSON value checked against a [`Schema`] byte by byte: whitespace around it, then nothing.
#[derive(Debug, Clone)]
pub struct JsonMatcher {
    schema: Arc<Schema>,
    stack: Vec<Frame>,
    whitespace: usize,
}

impl JsonMatcher {
    pub fn new(schema: Arc<Schema>) -> JsonMatcher {
        let root = schema.root;
        JsonMatcher { schema, stack: vec![Frame::Value(root)], whitespace: 0 }
    }

    /// The value is complete (a number or literal at the top level completes here).
    pub fn complete(&self) -> bool {
        match self.stack.as_slice() {
            [] => true,
            [Frame::Number { state, .. }] => state.complete(),
            [Frame::Literal { choices, at }] => choices.iter().any(|choice| choice.len() == *at),
            _ => false,
        }
    }

    pub fn feed_all(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&byte| self.feed(byte))
    }

    pub fn feed(&mut self, byte: u8) -> bool {
        if !is_space(byte) {
            self.whitespace = 0;
        }
        loop {
            let Some(frame) = self.stack.pop() else { return self.space(byte) };
            match self.step(frame, byte) {
                Step::Taken => return true,
                Step::Refused => return false,
                Step::Again => {}
            }
        }
    }

    fn space(&mut self, byte: u8) -> bool {
        self.whitespace += 1;
        let around = matches!(self.stack.as_slice(), [] | [Frame::Value(_)]);
        is_space(byte) && self.whitespace <= if around { WHITESPACE_AROUND } else { WHITESPACE_RUN }
    }

    fn taken(ok: bool) -> Step {
        if ok { Step::Taken } else { Step::Refused }
    }

    /// `frame` (popped) reads `byte`, pushing back what goes on.
    fn step(&mut self, frame: Frame, byte: u8) -> Step {
        match frame {
            Frame::Value(node) => {
                if is_space(byte) {
                    self.stack.push(Frame::Value(node));
                    return Self::taken(self.space(byte));
                }
                let Some(node) = self.schema.resolve(node, byte) else { return Step::Refused };
                let started = match &self.schema.nodes[node] {
                    Node::Object { properties, .. } => {
                        Frame::Object { node, state: ObjectState::KeyOrClose, seen: vec![false; properties.len()], property: None }
                    }
                    Node::Array { items } => Frame::Array { items: *items, state: ArrayState::ValueOrClose },
                    Node::Str => Frame::Str { escape: 0 },
                    Node::Number { integer } => Frame::Number { integer: *integer, state: NumberState::first(byte), length: 1 },
                    Node::Bool => literal(&[b"true", b"false"], byte),
                    Node::Null => literal(&[b"null"], byte),
                    Node::Literals(values) => {
                        Frame::Literal { choices: values.iter().filter(|value| value.first() == Some(&byte)).cloned().collect(), at: 1 }
                    }
                    Node::Any => match byte {
                        b'{' => Frame::Object { node: ANY, state: ObjectState::KeyOrClose, seen: Vec::new(), property: None },
                        b'[' => Frame::Array { items: ANY, state: ArrayState::ValueOrClose },
                        b'"' => Frame::Str { escape: 0 },
                        b't' | b'f' => literal(&[b"true", b"false"], byte),
                        b'n' => literal(&[b"null"], byte),
                        _ => Frame::Number { integer: false, state: NumberState::first(byte), length: 1 },
                    },
                    Node::Union(_) => return Step::Refused,
                };
                self.stack.push(started);
                self.finish_literal();
                Step::Taken
            }
            Frame::Object { node, state, seen, property } => {
                if is_space(byte) {
                    self.stack.push(Frame::Object { node, state, seen, property });
                    return Self::taken(self.space(byte));
                }
                let (properties, required, additional) = self.schema.object(node);
                match (state, byte) {
                    (ObjectState::KeyOrClose | ObjectState::Key, b'"') => {
                        self.stack.push(Frame::Object { node, state: ObjectState::Colon, seen, property });
                        self.stack.push(Frame::Key { node, buffer: Vec::new(), escape: 0 });
                        Step::Taken
                    }
                    (ObjectState::KeyOrClose | ObjectState::CommaOrClose, b'}') => {
                        if required.iter().any(|&index| !seen[index]) {
                            return Step::Refused;
                        }
                        self.ended();
                        Step::Taken
                    }
                    (ObjectState::Colon, b':') => {
                        let value = property.map_or(additional.unwrap_or(ANY), |index| properties[index].1);
                        self.stack.push(Frame::Object { node, state: ObjectState::Value, seen, property });
                        self.stack.push(Frame::Value(value));
                        Step::Taken
                    }
                    (ObjectState::CommaOrClose, b',') => {
                        // A closed object with every property given takes no more keys.
                        if additional.is_none() && seen.iter().all(|seen| *seen) {
                            return Step::Refused;
                        }
                        self.stack.push(Frame::Object { node, state: ObjectState::Key, seen, property: None });
                        Step::Taken
                    }
                    _ => Step::Refused,
                }
            }
            Frame::Key { node, mut buffer, mut escape } => {
                let Some(closing) = string_byte(&mut escape, byte) else { return Step::Refused };
                let schema = Arc::clone(&self.schema);
                let (properties, _, additional) = schema.object(node);
                let Some(Frame::Object { seen, property, .. }) = self.stack.last_mut() else { return Step::Refused };
                if closing {
                    let matched = properties.iter().position(|(name, _)| *name == buffer);
                    match matched {
                        Some(index) if seen[index] => return Step::Refused,
                        Some(index) => seen[index] = true,
                        None if additional.is_none() => return Step::Refused,
                        None => {}
                    }
                    *property = matched;
                    return Step::Taken;
                }
                buffer.push(byte);
                let possible = additional.is_some()
                    || properties.iter().enumerate().any(|(index, (name, _))| !seen[index] && name.starts_with(&buffer));
                self.stack.push(Frame::Key { node, buffer, escape });
                Self::taken(possible)
            }
            Frame::Array { items, state } => {
                if is_space(byte) && state != ArrayState::Value {
                    self.stack.push(Frame::Array { items, state });
                    return Self::taken(self.space(byte));
                }
                match (state, byte) {
                    (ArrayState::ValueOrClose | ArrayState::CommaOrClose, b']') => {
                        self.ended();
                        Step::Taken
                    }
                    (ArrayState::CommaOrClose, b',') => {
                        self.stack.push(Frame::Array { items, state: ArrayState::Value });
                        self.stack.push(Frame::Value(items));
                        Step::Taken
                    }
                    (ArrayState::ValueOrClose, _) => {
                        // the byte starts the first item
                        self.stack.push(Frame::Array { items, state: ArrayState::Value });
                        self.stack.push(Frame::Value(items));
                        Step::Again
                    }
                    _ => Step::Refused,
                }
            }
            Frame::Str { mut escape } => match string_byte(&mut escape, byte) {
                None => Step::Refused,
                Some(true) => {
                    self.ended();
                    Step::Taken
                }
                Some(false) => {
                    self.stack.push(Frame::Str { escape });
                    Step::Taken
                }
            },
            Frame::Number { integer, state, length } => match state.next(byte, integer) {
                Some(next) => {
                    self.stack.push(Frame::Number { integer, state: next, length: length + 1 });
                    Self::taken(length < NUMBER_LENGTH)
                }
                None if state.complete() => {
                    self.ended();
                    Step::Again
                }
                None => Step::Refused,
            },
            Frame::Literal { mut choices, at } => {
                if choices.iter().any(|choice| choice.get(at) == Some(&byte)) {
                    choices.retain(|choice| choice.get(at) == Some(&byte));
                    self.stack.push(Frame::Literal { choices, at: at + 1 });
                    self.finish_literal();
                    return Step::Taken;
                }
                if !choices.iter().any(|choice| choice.len() == at) {
                    return Step::Refused;
                }
                // A literal another extends ("1" of "1" and "12") ends before this byte.
                self.ended();
                Step::Again
            }
        }
    }

    /// A literal no other way extends, and done, has ended.
    fn finish_literal(&mut self) {
        if let Some(Frame::Literal { choices, at }) = self.stack.last() {
            if !choices.is_empty() && choices.iter().all(|choice| choice.len() == *at) {
                self.stack.pop();
                self.ended();
            }
        }
    }

    /// A value (already off the stack) has ended: its parent moves on.
    fn ended(&mut self) {
        match self.stack.last_mut() {
            Some(Frame::Object { state, .. }) => *state = ObjectState::CommaOrClose,
            Some(Frame::Array { state, .. }) => *state = ArrayState::CommaOrClose,
            _ => {}
        }
    }
}

/// A Literal frame for those of `words` that start with `byte`.
fn literal(words: &[&[u8]], byte: u8) -> Frame {
    Frame::Literal { choices: words.iter().filter(|word| word.first() == Some(&byte)).map(|word| word.to_vec()).collect(), at: 1 }
}

fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

/// A byte inside a JSON string, advancing `escape`: None when it cannot be one (a raw control
/// character, a bad escape), Some(true) for the closing quote.
fn string_byte(escape: &mut Escape, byte: u8) -> Option<bool> {
    match *escape {
        0 => match byte {
            b'"' => Some(true),
            b'\\' => {
                *escape = 1;
                Some(false)
            }
            0x00..=0x1f => None,
            _ => Some(false),
        },
        1 => {
            *escape = match byte {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => 0,
                b'u' => 5,
                _ => return None,
            };
            Some(false)
        }
        left => {
            if !byte.is_ascii_hexdigit() {
                return None;
            }
            *escape = if left == 2 { 0 } else { left - 1 };
            Some(false)
        }
    }
}

/// One way the answer may go.
#[derive(Debug, Clone)]
pub enum Alternative {
    Json(JsonMatcher),
    /// The answer starts with `rest` (after at most `whitespace` whitespace bytes), then goes on
    /// as `then`.
    Prefix { rest: Vec<u8>, whitespace: usize, then: Box<Alternative> },
    /// Anything up to and including `marker` (the bytes seen of it in `tail`), then `then`.
    Until { marker: Vec<u8>, tail: Vec<u8>, then: Box<Alternative> },
    /// Exactly `rest`, then nothing but the end.
    Exact { rest: Vec<u8> },
    Free,
}

impl Alternative {
    pub fn json(schema: Schema) -> Alternative {
        Alternative::Json(JsonMatcher::new(Arc::new(schema)))
    }

    /// The answer starts with `text`, then anything.
    pub fn prefix(text: &str, whitespace: usize) -> Alternative {
        Alternative::Prefix { rest: text.as_bytes().to_vec(), whitespace, then: Box::new(Alternative::Free) }
    }

    /// One call: the answer starts with `text`, goes on to `marker` (the call's close), then is
    /// exactly `close` and ends.
    pub fn single_call(text: &str, whitespace: usize, marker: &str, close: &str) -> Alternative {
        let closing = Alternative::Exact { rest: close.as_bytes().to_vec() };
        let until = Alternative::Until { marker: marker.as_bytes().to_vec(), tail: Vec::new(), then: Box::new(closing) };
        Alternative::Prefix { rest: text.as_bytes().to_vec(), whitespace, then: Box::new(until) }
    }

    /// Take a token's bytes; `special` (a control token) only the non-JSON ways take.
    fn take(&mut self, bytes: &[u8], special: bool) -> bool {
        if let Alternative::Json(json) = self {
            return !special && json.feed_all(bytes);
        }
        bytes.iter().all(|&byte| self.take_byte(byte))
    }

    fn take_byte(&mut self, byte: u8) -> bool {
        match self {
            Alternative::Json(json) => json.feed(byte),
            Alternative::Free => true,
            Alternative::Exact { rest } => {
                if rest.first() != Some(&byte) {
                    return false;
                }
                rest.remove(0);
                true
            }
            Alternative::Prefix { rest, whitespace, then } => {
                if rest.first() == Some(&byte) {
                    rest.remove(0);
                    // (whitespace may only come before the prefix starts)
                    *whitespace = 0;
                } else if *whitespace > 0 && is_space(byte) {
                    *whitespace -= 1;
                } else {
                    return false;
                }
                if rest.is_empty() {
                    let then = std::mem::replace(then, Box::new(Alternative::Free));
                    *self = *then;
                }
                true
            }
            Alternative::Until { marker, tail, then } => {
                tail.push(byte);
                if tail.ends_with(marker) {
                    let then = std::mem::replace(then, Box::new(Alternative::Free));
                    *self = *then;
                } else if tail.len() > marker.len() {
                    tail.remove(0);
                }
                true
            }
        }
    }

    fn can_end(&self) -> bool {
        match self {
            Alternative::Json(json) => json.complete(),
            Alternative::Free => true,
            Alternative::Exact { rest } => rest.is_empty(),
            Alternative::Prefix { .. } | Alternative::Until { .. } => false,
        }
    }
}

/// A reply held to its [`Alternative`]s after its reasoning.
#[derive(Clone)]
pub struct Constraint {
    tokens: Arc<TokenBytes>,
    end: Vec<u32>,
    /// While the reply reasons: its last bytes, to see `</think>` across tokens.
    thinking: Option<Vec<u8>>,
    alternatives: Vec<Alternative>,
}

impl Constraint {
    /// `thinking`: the reply starts as reasoning, held to nothing but going on until `</think>`.
    pub fn new(tokens: Arc<TokenBytes>, end: &[u32], thinking: bool, alternatives: Vec<Alternative>) -> Constraint {
        Constraint { tokens, end: end.to_vec(), thinking: thinking.then(Vec::new), alternatives }
    }

    /// Whether `token` may come next.
    pub fn allows(&self, token: u32) -> bool {
        self.after(token).is_some()
    }

    /// Take `token` (one [`Constraint::allows`] accepted).
    pub fn advance(&mut self, token: u32) {
        if let Some(next) = self.after(token) {
            *self = next;
        }
    }

    fn after(&self, token: u32) -> Option<Constraint> {
        let end = self.end.contains(&token);
        let (bytes, special) = self.tokens.get(token)?;
        let mut answer = bytes;
        if let Some(tail) = &self.thinking {
            if end {
                return None;
            }
            if let Some(close) = self.tokens.think_close {
                if token != close {
                    return Some(self.clone());
                }
                answer = &[];
            } else {
                let mut joined = tail.clone();
                joined.extend_from_slice(bytes);
                let Some(at) = find(&joined, THINK_CLOSE) else {
                    let keep = joined.len().saturating_sub(THINK_CLOSE.len() - 1);
                    return Some(Constraint { thinking: Some(joined[keep..].to_vec()), ..self.clone() });
                };
                answer = &bytes[at + THINK_CLOSE.len() - tail.len()..];
            }
        }
        if end {
            return self.alternatives.iter().any(Alternative::can_end).then(|| self.clone());
        }
        let alternatives: Vec<Alternative> = self.alternatives.iter().cloned()
            .filter_map(|mut alternative| alternative.take(answer, special && !answer.is_empty()).then_some(alternative))
            .collect();
        (!alternatives.is_empty()).then(|| Constraint { thinking: None, alternatives, ..self.clone() })
    }
}

impl Constraint {
    /// The constraint a worker request's `constraint` asks for (straydog-api's request.rs):
    /// `{"alternatives": [{"json": true | <schema>}, {"prefix": <text>, "whitespace": <count>}]}`;
    /// None for null.
    pub fn from_request(spec: &Value, tokens: Arc<TokenBytes>, end: &[u32], thinking: bool) -> Result<Option<Constraint>, String> {
        let list = match spec {
            Value::Null => return Ok(None),
            spec => match spec.get("alternatives") {
                Some(Value::Array(list)) if !list.is_empty() => list,
                _ => return Err("a Straydog constraint needs alternatives".into()),
            },
        };
        let mut alternatives = Vec::new();
        for alternative in list {
            if let Some(json) = alternative.get("json") {
                let schema = if matches!(json, Value::Bool(true)) { Schema::any_object() } else { Schema::compile(json)? };
                alternatives.push(Alternative::json(schema));
            } else if let Some(Value::Str(prefix)) = alternative.get("prefix") {
                let whitespace = match alternative.get("whitespace") {
                    None => 0,
                    Some(Value::Int(count)) => usize::try_from(*count).map_err(|_| "a Straydog constraint's whitespace must be a count")?,
                    Some(_) => return Err("a Straydog constraint's whitespace must be a count".into()),
                };
                let text = |key: &str| match alternative.get(key) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Str(text)) => Ok(Some(text.as_str())),
                    Some(_) => Err(format!("a Straydog constraint's {key} must be a string")),
                };
                alternatives.push(match (text("until")?, text("close")?) {
                    (Some(marker), Some(close)) => Alternative::single_call(prefix, whitespace, marker, close),
                    (None, None) => Alternative::prefix(prefix, whitespace),
                    _ => return Err("a Straydog constraint's until and close come together".into()),
                });
            } else {
                return Err("a Straydog constraint alternative is json or prefix".into());
            }
        }
        Ok(Some(Constraint::new(tokens, end, thinking, alternatives)))
    }
}

/// The next token: sampled, or chosen against `constraint` and taken by it.
pub fn pick(logits: &[f32], sampling: &Sampling, generated: &Generated, rng: &mut Rng, constraint: &mut Option<&mut Constraint>)
            -> Result<u32, String> {
    let Some(constraint) = constraint else { return Ok(sampling::choose(logits, sampling, generated, rng)) };
    let token = sampling::choose_where(logits, sampling, generated, rng, &mut |token| constraint.allows(token))
        .ok_or("Straydog's constraint left no token to choose")?;
    constraint.advance(token);
    Ok(token)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(text: &str) -> Schema {
        Schema::compile(&pyjson::parse(text).unwrap()).unwrap()
    }

    fn accepts(schema: &Schema, text: &str) -> bool {
        let mut json = JsonMatcher::new(Arc::new(schema.clone()));
        json.feed_all(text.as_bytes()) && json.complete()
    }

    /// Fed as far as it goes: a prefix some valid document continues.
    fn prefix(schema: &Schema, text: &str) -> bool {
        JsonMatcher::new(Arc::new(schema.clone())).feed_all(text.as_bytes())
    }

    #[test]
    fn any_object_takes_any_json_object_and_nothing_else() {
        let any = Schema::any_object();
        for good in [r#"{}"#, r#"{"a": 1, "b": [true, null, -1.5e3, "x\"yé"], "c": {"d": {}}}"#, " \n{ \"k\" : \"v\" } \n"] {
            assert!(accepts(&any, good), "{good}");
        }
        for bad in [r#"[]"#, r#""s""#, r#"{"a" 1}"#, r#"{"a": 01}"#, r#"{"a": tru}"#, r#"{"a": "x"#, "{\"a\": \"\n\"}", r#"{} {}"#] {
            assert!(!accepts(&any, bad), "{bad}");
        }
        assert!(prefix(&any, r#"{"a": [1, 2"#));
        assert!(!prefix(&any, "x"));
    }

    #[test]
    fn a_schema_holds_types_required_closed_keys_and_enums() {
        let s = schema(r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
            "role": {"enum": ["admin", "user"]}, "tags": {"type": "array", "items": {"type": "string"}},
            "boss": {"type": ["string", "null"]}}, "required": ["name", "age"], "additionalProperties": false}"#);
        assert!(accepts(&s, r#"{"name": "Ann", "age": 41, "role": "user", "tags": ["a", "b"], "boss": null}"#));
        assert!(accepts(&s, r#"{"age": 3, "name": "B"}"#));
        assert!(!accepts(&s, r#"{"name": "Ann"}"#), "age is required");
        assert!(!accepts(&s, r#"{"name": "Ann", "age": 4.5}"#), "an integer");
        assert!(!accepts(&s, r#"{"name": "Ann", "age": 4, "role": "root"}"#), "an enum");
        assert!(!prefix(&s, r#"{"nick"#), "no such key");
        assert!(!accepts(&s, r#"{"name": "A", "name": "B", "age": 1}"#), "a repeated key");
        assert!(!accepts(&s, r#"{"name": 1, "age": 1}"#));
        assert!(!prefix(&s, r#"{"name": "A", "age": 1, "role": "user", "tags": [], "boss": "x","#), "every key given");
    }

    #[test]
    fn refs_recurse_and_scalars_complete_at_the_top() {
        let tree = schema(r##"{"$defs": {"node": {"type": "object", "properties": {"v": {"type": "number"},
            "kids": {"type": "array", "items": {"$ref": "#/$defs/node"}}}, "required": ["v"]}}, "$ref": "#/$defs/node"}"##);
        assert!(accepts(&tree, r#"{"v": 1, "kids": [{"v": 2, "kids": []}, {"v": 3}]}"#));
        assert!(!accepts(&tree, r#"{"v": 1, "kids": [{"kids": []}]}"#));
        let number = schema(r#"{"type": "number"}"#);
        assert!(accepts(&number, "-12.5"));
        assert!(!accepts(&number, "-"));
        let literal = schema(r#"{"const": 12}"#);
        assert!(accepts(&literal, "12") && !accepts(&literal, "1") && !prefix(&literal, "13"));
    }

    #[test]
    fn whitespace_runs_are_bounded() {
        let any = Schema::any_object();
        assert!(!prefix(&any, &format!("{{{}", " ".repeat(WHITESPACE_RUN + 1))));
        assert!(prefix(&any, &format!("{{{}\"a\"", " ".repeat(WHITESPACE_RUN))));
        assert!(prefix(&any, &format!("{}{{", " ".repeat(WHITESPACE_AROUND))));
        assert!(!prefix(&any, &" ".repeat(WHITESPACE_AROUND + 1)), "padding before the value");
        assert!(!prefix(&any, &format!("{{}}{}", " ".repeat(WHITESPACE_AROUND + 1))), "padding after it");
    }

    fn tokens(pieces: &[&str]) -> Arc<TokenBytes> {
        let special = pieces.iter().map(|piece| piece.starts_with("<|")).collect();
        Arc::new(TokenBytes::new(pieces.iter().map(|piece| piece.as_bytes().to_vec()).collect(), special))
    }

    #[test]
    fn the_constraint_waits_for_the_reasoning_and_ends_only_when_complete() {
        // 0 end, 1 "</think>", 2 "{", 3 "}", 4 "hmm", 5 "\n\n", 6 "<|x|>", 7 "think>{"
        let vocabulary = tokens(&["<|end|>", "</think>", "{", "}", "hmm", "\n\n", "<|x|>", "think>{"]);
        let mut constraint = Constraint::new(vocabulary, &[0], true, vec![Alternative::json(Schema::any_object())]);
        assert!(constraint.allows(4) && constraint.allows(3) && !constraint.allows(0), "reasoning: anything but the end");
        constraint.advance(1);
        assert!(!constraint.allows(4) && !constraint.allows(0) && !constraint.allows(6));
        assert!(constraint.allows(5) && constraint.allows(2));
        constraint.advance(2);
        assert!(!constraint.allows(0));
        constraint.advance(3);
        assert!(constraint.allows(0) && !constraint.allows(2));
    }

    #[test]
    fn a_think_close_token_ends_the_reasoning_and_its_text_does_not() {
        // 0 end, 1 "</think>" (the vocabulary's one token), 2 "</", 3 "think>", 4 "{", 5 "}"
        let pieces = ["<|end|>", "</think>", "</", "think>", "{", "}"];
        let vocabulary = TokenBytes::new(pieces.iter().map(|piece| piece.as_bytes().to_vec()).collect(), vec![false; pieces.len()]);
        let mut constraint = Constraint::new(Arc::new(vocabulary.with_think_close(Some(1))), &[0], true,
                                             vec![Alternative::json(Schema::any_object())]);
        // the reasoning writes the tag's text: still reasoning, anything goes
        for token in [2, 3, 5] {
            assert!(constraint.allows(token));
            constraint.advance(token);
        }
        assert!(!constraint.allows(0));
        constraint.advance(1);
        // the answer is held to JSON
        assert!(!constraint.allows(5) && constraint.allows(4));
    }

    #[test]
    fn think_close_split_across_tokens_starts_the_answer_mid_token() {
        let vocabulary = tokens(&["<|end|>", "</", "think>{", "think>x", "}"]);
        let mut constraint = Constraint::new(vocabulary, &[0], true, vec![Alternative::json(Schema::any_object())]);
        constraint.advance(1);
        assert!(constraint.allows(2) && !constraint.allows(3));
        constraint.advance(2);
        assert!(constraint.allows(4));
    }

    #[test]
    fn a_tool_prefix_is_forced_then_free_and_races_the_json() {
        let vocabulary = tokens(&["<|end|>", "\n\n", "<tool>", "{", "get", "\"a\"", "}"]);
        let tool = || Alternative::prefix("<tool>get", 2);
        let mut forced = Constraint::new(Arc::clone(&vocabulary), &[0], false, vec![tool()]);
        assert!(!forced.allows(3) && !forced.allows(0) && forced.allows(1));
        forced.advance(1);
        forced.advance(2);
        assert!(!forced.allows(0) && forced.allows(4));
        forced.advance(4);
        assert!(forced.allows(0) && forced.allows(3), "free after the prefix");
        let mut either = Constraint::new(vocabulary, &[0], false, vec![Alternative::json(Schema::any_object()), tool()]);
        assert!(either.allows(2) && either.allows(3));
        either.advance(3);
        assert!(!either.allows(2) && either.allows(5));
    }

    #[test]
    fn a_single_call_closes_after_its_invoke_and_ends() {
        // 0 end, 1 "<call name=f>", 2 "x", 3 "</invoke>", 4 "\n</calls>", 5 "<call name=g>"
        let vocabulary = tokens(&["<|end|>", "<call name=f>", "x", "</invoke>", "\n</calls>", "<call name=g>"]);
        let one = Alternative::single_call("<call name=f>", 0, "</invoke>", "\n</calls>");
        let mut constraint = Constraint::new(vocabulary, &[0], false, vec![one]);
        assert!(constraint.allows(1) && !constraint.allows(5));
        constraint.advance(1);
        assert!(constraint.allows(2) && constraint.allows(5) && !constraint.allows(0), "free inside the call");
        constraint.advance(2);
        constraint.advance(3);
        assert!(!constraint.allows(5) && !constraint.allows(0) && constraint.allows(4), "the close, nothing else");
        constraint.advance(4);
        assert!(constraint.allows(0) && !constraint.allows(2));
    }
}
