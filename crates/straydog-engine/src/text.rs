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

pub struct Text {
    tokenizer: Tokenizer,
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
        Ok(Text { tokenizer })
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

    /// The ids of a rendered prompt; special-token text in it becomes those tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TextError> {
        let encoding = self.tokenizer.encode(text, false).map_err(|exc| TextError(exc.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Ids as text, special tokens included.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TextError> {
        self.tokenizer.decode(ids, false).map_err(|exc| TextError(exc.to_string()))
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
