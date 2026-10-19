//! The authenticated tokenizer: prompt text to token ids, and generated ids back to text.

use tokenizers::Tokenizer;

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
