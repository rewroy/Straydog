//! The authenticated tokenizer: prompt text to token ids, and generated ids back to text.

use straydog_gguf::ModelHeader;
use tokenizers::Tokenizer;

/// The pre-tokenizer of a GGUF's `tokenizer.ggml.pre`, as the model's own tokenizer.json has it.
fn gguf_pre_tokenizer(name: &str) -> Option<serde_json::Value> {
    let split = |pattern: &str| serde_json::json!({"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false});
    match name {
        // DeepSeek-V3's: digits in threes, runs of CJK, then words and punctuation.
        "joyai-llm" | "deepseek-v3" => Some(serde_json::json!({"type": "Sequence", "pretokenizers": [
            split(r"\p{N}{1,3}"),
            split("[一-龥぀-ゟ゠-ヿ]+"),
            split("[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"),
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        ]})),
        _ => None,
    }
}

/// Marks, in a message's own text, an added token's text the template did not write
/// ([`Text::defuse`]): [`Text::encode`] takes it as ordinary text, never as that token.
pub const DEFUSED: char = '\u{e001}';

/// Segments (the text between two added tokens) of at least this many bytes are kept by their digest once encoded ([`Text::encode`]).
const MEMO_FROM: usize = 4096;
/// The most tokens the kept segments hold, the oldest let go first: a conversation at 262,144 tokens with room to spare.
const MEMO_TOKENS: usize = 1 << 19;

/// Encoded segments by the SHA-256 of their text, and the order they came in. A conversation's next request renders its earlier messages
/// again, and their segments come back the same: only the new ones need encoding (a 250K-token conversation took 0.30 to 0.34 s to encode
/// whole for every request on a Ryzen 7 9700X). Memory only, never written anywhere.
#[derive(Default)]
struct Memo {
    segments: std::collections::HashMap<[u8; 32], std::sync::Arc<[u32]>>,
    order: std::collections::VecDeque<[u8; 32]>,
    tokens: usize,
}

pub struct Text {
    tokenizer: Tokenizer,
    /// The same without its added tokens: a defused one's text as any other text.
    plain: Tokenizer,
    /// The added tokens' texts, longest first, anywhere and at the start of a text.
    added: Option<(regex::Regex, regex::Regex)>,
    /// [`Text::token_bytes`], built on first use.
    bytes: std::sync::OnceLock<std::sync::Arc<crate::constraint::TokenBytes>>,
    memo: std::sync::Mutex<Memo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextError(pub String);

impl std::fmt::Display for TextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Straydog tokenizer: {}", self.0)
    }
}

impl std::error::Error for TextError {}

impl Text {
    /// From the bytes of `tokenizer.json`, already authenticated.
    pub fn new(tokenizer_json: &[u8]) -> Result<Text, TextError> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_json).map_err(|exc| TextError(exc.to_string()))?;
        let mut json: serde_json::Value = serde_json::from_slice(tokenizer_json).map_err(|exc| TextError(exc.to_string()))?;
        let mut contents: Vec<String> = json["added_tokens"].as_array().into_iter().flatten()
            .filter_map(|token| token["content"].as_str().filter(|content| !content.is_empty()).map(str::to_string)).collect();
        json["added_tokens"] = serde_json::Value::Array(Vec::new());
        let plain = Tokenizer::from_bytes(json.to_string().as_bytes()).map_err(|exc| TextError(exc.to_string()))?;
        contents.sort_by_key(|content| std::cmp::Reverse(content.len()));
        let alternatives = contents.iter().map(|content| regex::escape(content)).collect::<Vec<_>>().join("|");
        let added = if contents.is_empty() {
            None
        } else {
            let compile = |pattern: String| regex::Regex::new(&pattern).map_err(|exc| TextError(exc.to_string()));
            Some((compile(alternatives.clone())?, compile(format!("^(?:{alternatives})"))?))
        };
        Ok(Text { tokenizer, plain, added, bytes: std::sync::OnceLock::new(), memo: Default::default() })
    }

    /// `text`, a message's own, with each added token's text in it marked [`DEFUSED`]: only the
    /// chat template writes those tokens. A tool's output quoting `</think>` was the token itself
    /// to the model, which then wrote it in its reasoning, and the reply ended its reasoning there.
    pub fn defuse<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match &self.added {
            Some((anywhere, _)) => anywhere.replace_all(text, |found: &regex::Captures<'_>| format!("{DEFUSED}{}", &found[0])),
            None => std::borrow::Cow::Borrowed(text),
        }
    }

    /// The id of `content` when it is one added token (a template's tag), else None.
    pub fn added_token(&self, content: &str) -> Option<u32> {
        let id = self.tokenizer.token_to_id(content)?;
        self.tokenizer.get_added_tokens_decoder().get(&id).filter(|added| added.content == content).map(|_| id)
    }

    /// The tokenizer a GGUF describes (`tokenizer.ggml.*`): byte-level BPE over its tokens and
    /// merges, the pre-tokenizer `tokenizer.ggml.pre` names, and its control and user-defined
    /// tokens matched whole in the text (the former special).
    pub fn from_gguf(header: &ModelHeader) -> Result<Text, TextError> {
        let error = |message: &str| TextError(message.to_string());
        let metadata = |key: &str| header.metadata.get(key).ok_or_else(|| TextError(format!("GGUF metadata {key} is missing")));
        if metadata("tokenizer.ggml.model")?.text() != Some("gpt2") {
            return Err(error("only byte-level BPE GGUF tokenizers are supported"));
        }
        let pre = metadata("tokenizer.ggml.pre")?.text().ok_or_else(|| error("invalid GGUF tokenizer.ggml.pre"))?;
        let pre_tokenizer = gguf_pre_tokenizer(pre).ok_or_else(|| TextError(format!("unsupported GGUF pre-tokenizer {pre}")))?;
        let tokens = metadata("tokenizer.ggml.tokens")?.array().ok_or_else(|| error("invalid GGUF tokens"))?;
        let kinds = metadata("tokenizer.ggml.token_type")?.array().ok_or_else(|| error("invalid GGUF token types"))?;
        let merges = metadata("tokenizer.ggml.merges")?.array().ok_or_else(|| error("invalid GGUF merges"))?;
        if kinds.len() != tokens.len() {
            return Err(error("GGUF token types do not cover the tokens"));
        }
        // llama.cpp's token types: 1 normal, 3 control, 4 user defined, 6 byte.
        const CONTROL: i64 = 3;
        const USER_DEFINED: i64 = 4;
        let (mut vocabulary, mut added) = (serde_json::Map::new(), Vec::new());
        // Every token is in the BPE vocabulary, so an added token keeps its own id.
        for (id, (token, kind)) in tokens.iter().zip(kinds).enumerate() {
            let token = token.text().ok_or_else(|| error("invalid GGUF token"))?;
            if vocabulary.insert(token.to_string(), id.into()).is_some() {
                return Err(error("a GGUF token appears twice"));
            }
            match kind.integer() {
                Some(kind @ (CONTROL | USER_DEFINED)) => added.push(serde_json::json!({
                    "id": id, "content": token, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false,
                    "special": kind == CONTROL})),
                Some(_) => {}
                None => return Err(error("invalid GGUF token type")),
            }
        }
        let merges: Vec<&str> = merges.iter().map(|merge| merge.text().ok_or_else(|| error("invalid GGUF merge"))).collect::<Result<_, _>>()?;
        let tokenizer = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
            "normalizer": {"type": "Sequence", "normalizers": []},
            "pre_tokenizer": pre_tokenizer,
            "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
            "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                      "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "vocab": vocabulary, "merges": merges},
        });
        Text::new(tokenizer.to_string().as_bytes())
    }

    /// The ids of a rendered prompt; special-token text in it becomes those tokens, but for a
    /// [`DEFUSED`] one's, ordinary text (a mark before anything else is dropped).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TextError> {
        if !text.contains(DEFUSED) {
            return self.encode_kept(text);
        }
        let (mut ids, mut rest) = (Vec::new(), text);
        while let Some(at) = rest.find(DEFUSED) {
            ids.extend(self.encode_kept(&rest[..at])?);
            let after = &rest[at + DEFUSED.len_utf8()..];
            let length = self.added.as_ref().and_then(|(_, at_start)| at_start.find(after)).map_or(0, |found| found.end());
            ids.extend(Self::encode_with(&self.plain, &after[..length])?);
            rest = &after[length..];
        }
        ids.extend(self.encode_kept(rest)?);
        Ok(ids)
    }

    /// [`Text::encode_with`] the full tokenizer, its long segments from the [`Memo`]. The tokenizer splits a text at its added tokens before
    /// anything else (none of them strips the space around it) and encodes each piece alone, NFC included: the pieces' ids side by side are
    /// the whole text's.
    fn encode_kept(&self, text: &str) -> Result<Vec<u32>, TextError> {
        let Some((anywhere, _)) = &self.added else { return Self::encode_with(&self.tokenizer, text) };
        if text.len() < MEMO_FROM {
            return Self::encode_with(&self.tokenizer, text);
        }
        let (mut ids, mut last) = (Vec::new(), 0);
        for found in anywhere.find_iter(text) {
            self.encode_segment(&text[last..found.start()], &mut ids)?;
            ids.extend(Self::encode_with(&self.tokenizer, found.as_str())?);
            last = found.end();
        }
        self.encode_segment(&text[last..], &mut ids)?;
        Ok(ids)
    }

    fn encode_segment(&self, segment: &str, ids: &mut Vec<u32>) -> Result<(), TextError> {
        if segment.len() < MEMO_FROM {
            ids.extend(Self::encode_with(&self.tokenizer, segment)?);
            return Ok(());
        }
        use sha2::Digest;
        let key: [u8; 32] = sha2::Sha256::digest(segment.as_bytes()).into();
        if let Some(kept) = self.memo.lock().unwrap().segments.get(&key).cloned() {
            ids.extend_from_slice(&kept);
            return Ok(());
        }
        let encoded: std::sync::Arc<[u32]> = Self::encode_with(&self.tokenizer, segment)?.into();
        ids.extend_from_slice(&encoded);
        let mut memo = self.memo.lock().unwrap();
        if encoded.len() <= MEMO_TOKENS && memo.segments.insert(key, encoded.clone()).is_none() {
            memo.order.push_back(key);
            memo.tokens += encoded.len();
            while memo.tokens > MEMO_TOKENS {
                let Some(oldest) = memo.order.pop_front() else { break };
                if let Some(gone) = memo.segments.remove(&oldest) { memo.tokens -= gone.len(); }
            }
        }
        Ok(())
    }

    fn encode_with(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, TextError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let encoding = tokenizer.encode(text, false).map_err(|exc| TextError(exc.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Ids as text, special tokens included.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TextError> {
        self.tokenizer.decode(ids, false).map_err(|exc| TextError(exc.to_string()))
    }

    /// Each token id's bytes, and whether it is a control token ([`crate::constraint`]): a
    /// byte-level BPE token read back through GPT-2's alphabet, an added token as its text.
    pub fn token_bytes(&self) -> std::sync::Arc<crate::constraint::TokenBytes> {
        std::sync::Arc::clone(self.bytes.get_or_init(|| std::sync::Arc::new(self.build_token_bytes())))
    }

    fn build_token_bytes(&self) -> crate::constraint::TokenBytes {
        let vocabulary = self.tokenizer.get_vocab(true);
        let size = vocabulary.values().max().map_or(0, |id| *id as usize + 1);
        let added = self.tokenizer.get_added_tokens_decoder();
        let decoder = crate::constraint::byte_level_decoder();
        let (mut bytes, mut special) = (vec![Vec::new(); size], vec![true; size]);
        for (token, id) in vocabulary {
            let index = id as usize;
            if let Some(added) = added.get(&id) {
                bytes[index] = added.content.as_bytes().to_vec();
                special[index] = added.special;
            } else {
                bytes[index] = token.chars().map(|char| decoder.get(&char).copied()).collect::<Option<Vec<u8>>>()
                    .unwrap_or_else(|| token.as_bytes().to_vec());
                special[index] = false;
            }
        }
        crate::constraint::TokenBytes::new(bytes, special).with_think_close(self.added_token(crate::reply::THINK_CLOSE))
    }

    /// Text for a stream of generated ids, one delta at a time.
    pub fn stream(&self) -> Detokenizer<'_> {
        Detokenizer { text: self, ids: Vec::new(), prefix: String::new(), prefix_ids: 0 }
    }
}

/// Decodes a growing sequence of ids in a short window. A delta is emitted once the
/// window's text extends what was emitted and does not end inside a character (U+FFFD);
/// the deltas joined with [`Detokenizer::finish`] equal the decode of all ids at once.
pub struct Detokenizer<'a> {
    text: &'a Text,
    /// The window: the ids of the last emitted delta, then those not yet emitted.
    ids: Vec<u32>,
    /// The decode of the window's first `prefix_ids` ids, all emitted.
    prefix: String,
    prefix_ids: usize,
}

impl Detokenizer<'_> {
    pub fn push(&mut self, id: u32) -> Result<String, TextError> {
        self.ids.push(id);
        let decoded = self.text.decode(&self.ids)?;
        if decoded.len() <= self.prefix.len() || decoded.ends_with('\u{fffd}') {
            return Ok(String::new());
        }
        let delta = self.delta(&decoded)?;
        // Keep the ids just emitted as the next window's context.
        self.ids.drain(..self.prefix_ids);
        self.prefix = self.text.decode(&self.ids)?;
        self.prefix_ids = self.ids.len();
        Ok(delta)
    }

    /// The rest of the text when generation ends, incomplete characters included.
    pub fn finish(self) -> Result<String, TextError> {
        let decoded = self.text.decode(&self.ids)?;
        if decoded.len() <= self.prefix.len() {
            return Ok(String::new());
        }
        self.delta(&decoded)
    }

    fn delta(&self, decoded: &str) -> Result<String, TextError> {
        decoded
            .strip_prefix(self.prefix.as_str())
            .map(str::to_string)
            .ok_or_else(|| TextError("the decoder changed text it had already emitted".into()))
    }
}

#[cfg(test)]
mod memo_tests {
    use super::{Text, MEMO_FROM};

    /// Decides: a rendered conversation's ids are the same with the kept segments as without, the first time and when its earlier
    /// messages come back with a new one (the real Qwen3.8 tokenizer; skipped where its file is not on the machine).
    #[test]
    fn kept_segments_encode_a_conversation_as_the_tokenizer_does_whole() {
        let path = std::env::var("STRAYDOG_TOKENIZER_JSON")
            .unwrap_or_else(|_| "/server/gguf-models/Qwen3.8-Flash-Next-GGUF/UD-IQ2_XS/tokenizer.json".into());
        let Ok(bytes) = std::fs::read(&path) else { return };
        let text = Text::new(&bytes).unwrap();
        // long segments with accents in both composed and decomposed forms (NFC), code, Japanese and spaces at the seams
        let body = |n: usize| -> String {
            let line = format!("fn f{n}() {{ let café = \"cafe\\u{{301}} e\u{301}\"; }} // 日本語の行 {n}  \\n  ");
            line.repeat(MEMO_FROM / line.len() + 3)
        };
        let mut rendered = String::from("<|im_start|>system\nYou are helpful.<|im_end|>\n");
        for turn in 0..4 {
            rendered += &format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n {} <|im_end|>\n", body(turn), body(turn + 10));
            let whole = Text::encode_with(&text.tokenizer, &rendered).unwrap();
            assert_eq!(text.encode(&rendered).unwrap(), whole, "turn {turn}, first time");
            assert_eq!(text.encode(&rendered).unwrap(), whole, "turn {turn}, from the kept segments");
        }
        assert!(text.memo.lock().unwrap().segments.len() >= 4);
    }
}
