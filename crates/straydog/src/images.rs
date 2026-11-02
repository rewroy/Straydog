//! A request's images, for a model that reads them (DeepSeek-V4.1 and Qwen3.8 beside their vision encoders).
//!
//! Each image part of a message (`image_url` with a `data:` URL, or a base64 `image` source)
//! leaves the messages as a text part holding a marker of its own. The template renders a text
//! part where it would an image (joined to its neighbors alike), but it reorders tool results, so
//! the markers say which image each place takes. The prompt's ids then come a piece at a time: the
//! text between markers encoded, and in each marker's place its image's span of positions (each
//! with its own id: [`straydog_engine::arch::deepseek41::engine::IMAGE_TOKEN`] and up). Qwen3.8's template
//! writes `<|vision_start|><|image_pad|><|vision_end|>` where an image is: a marker's place takes those tokens,
//! the pad grown to the image's merged grid.

use sha2::{Digest, Sha256};
use straydog_engine::arch::deepseek41::engine::IMAGE_TOKEN;
use straydog_engine::pyjson::Value;
use straydog_engine::qwen_vision::{self, Prepared, VISION_END, VISION_START};
use straydog_engine::text::{Text, TextError};
use straydog_engine::vision::{self, Patches, VisionConfig};

const MARK: char = '\u{f8ff}';
const MARKER: &str = "\u{f8ff}straydog-image-";

fn marker(index: usize) -> String {
    format!("{MARKER}{index}{MARK}")
}

/// An image of the request: its encoded bytes' digest and its patches.
pub struct Image {
    pub digest: [u8; 32],
    pub patches: Patches,
}

impl Image {
    /// The ids of its span: from its digest, consecutive.
    fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        span_ids(&self.digest, self.patches.plan.tokens())
    }
}

/// The messages with each image part replaced by its marker, and the images' encoded bytes in
/// marker order. Text holding a marker's prefix is refused.
pub fn take(messages: &Value) -> Result<(Value, Vec<Vec<u8>>), String> {
    let mut images = Vec::new();
    let Value::Array(list) = messages else { return Ok((messages.clone(), images)) };
    let mut out = Vec::with_capacity(list.len());
    for message in list {
        let Value::Object(fields) = message else {
            out.push(message.clone());
            continue;
        };
        let mut kept = Vec::with_capacity(fields.len());
        for (key, value) in fields {
            let value = match (key.as_str(), value) {
                ("content", Value::Array(parts)) => Value::Array(parts.iter().map(|part| part_of(part, &mut images)).collect::<Result<_, _>>()?),
                ("content", Value::Str(text)) if text.contains(MARKER) => return Err("message text holds Straydog's image marker".into()),
                _ => value.clone(),
            };
            kept.push((key.clone(), value));
        }
        out.push(Value::Object(kept));
    }
    Ok((Value::Array(out), images))
}

fn part_of(part: &Value, images: &mut Vec<Vec<u8>>) -> Result<Value, String> {
    let text_part = |text: String| Value::Object(vec![("type".into(), Value::Str("text".into())), ("text".into(), Value::Str(text))]);
    let bytes = match part.get("type").and_then(Value::as_str) {
        Some("image_url") => {
            let url = match part.get("image_url") {
                Some(Value::Str(url)) => url.as_str(),
                Some(object) => object.get("url").and_then(Value::as_str).ok_or("an image_url part without a url")?,
                None => return Err("an image_url part without a url".into()),
            };
            data_url(url)?
        }
        Some("image") => {
            let source = part.get("source").ok_or("an image part without a source")?;
            match (source.get("type").and_then(Value::as_str), source.get("data").and_then(Value::as_str)) {
                (Some("base64"), Some(data)) => base64(data)?,
                (Some("url"), _) => data_url(source.get("url").and_then(Value::as_str).unwrap_or_default())?,
                _ => return Err("an image source must be base64 data or a data URL".into()),
            }
        }
        Some("text") if part.get("text").and_then(Value::as_str).is_some_and(|text| text.contains(MARKER)) => {
            return Err("message text holds Straydog's image marker".into());
        }
        _ => return Ok(part.clone()),
    };
    images.push(bytes);
    Ok(text_part(marker(images.len() - 1)))
}

/// A `data:` URL's bytes (base64); the server fetches no other URL.
fn data_url(url: &str) -> Result<Vec<u8>, String> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Err("an image URL must be a data: URL (Straydog fetches nothing)".into());
    };
    let (header, payload) = rest.split_once(',').ok_or("a malformed data URL")?;
    if !header.ends_with(";base64") {
        return Err("an image data URL must be base64".into());
    }
    base64(payload)
}

fn base64(text: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut bits, mut count) = (0u32, 0u32);
    for byte in text.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => return Err("invalid base64 image data".into()),
        };
        bits = (bits << 6) | u32::from(value);
        count += 6;
        if count >= 8 {
            count -= 8;
            out.push((bits >> count) as u8);
        }
        if out.len() > vision::MAX_BYTES {
            return Err(format!("an image of more than {} MB", vision::MAX_BYTES >> 20));
        }
    }
    Ok(out)
}

/// Decode an image and cut it into the encoder's patches.
pub fn prepare(bytes: &[u8]) -> Result<Image, String> {
    let rgb = vision::decode(bytes)?;
    Ok(Image { digest: Sha256::digest(bytes).into(), patches: vision::patches(&rgb, &VisionConfig::DEEPSEEK41) })
}

/// A Qwen3.8 request's image: its encoded bytes' digest and what its encoder takes.
pub struct QwenImage {
    pub digest: [u8; 32],
    pub prepared: Prepared,
}

/// Decode an image and size it for Qwen3.8's encoder.
pub fn prepare_qwen(bytes: &[u8]) -> Result<QwenImage, String> {
    let rgb = vision::decode(bytes)?;
    Ok(QwenImage { digest: Sha256::digest(bytes).into(), prepared: qwen_vision::prepare(&rgb)? })
}

/// The ids of a span of `count` positions: from its image's digest, consecutive, past the vocabulary.
fn span_ids(digest: &[u8; 32], count: usize) -> impl Iterator<Item = u32> {
    let first = u32::from_le_bytes(digest[..4].try_into().unwrap());
    (0..count as u32).map(move |offset| IMAGE_TOKEN | (first.wrapping_add(offset) & !IMAGE_TOKEN))
}

/// Qwen3.8's prompt ids, `<|vision_start|>`, an image's span and `<|vision_end|>` in each marker's place, and where each placed
/// image's span starts.
pub fn qwen_ids(text: &Text, pieces: &[Piece<'_>], images: &[QwenImage]) -> Result<(Vec<u32>, Spans), TextError> {
    let (mut ids, mut spans) = (Vec::new(), Vec::new());
    for piece in pieces {
        match *piece {
            Piece::Text(part) => ids.extend(text.encode(part)?),
            Piece::Image(index) => {
                ids.push(VISION_START);
                spans.push((index, ids.len()));
                ids.extend(span_ids(&images[index].digest, images[index].prepared.tokens()));
                ids.push(VISION_END);
            }
        }
    }
    Ok((ids, spans))
}

/// A piece of a rendered prompt.
#[derive(Debug, PartialEq, Eq)]
pub enum Piece<'a> {
    Text(&'a str),
    Image(usize),
}

/// The rendered prompt cut at its markers: each of `count` images at most once.
pub fn pieces(prompt: &str, count: usize) -> Result<Vec<Piece<'_>>, String> {
    let mut pieces = Vec::new();
    let mut seen = vec![false; count];
    let mut rest = prompt;
    while let Some(at) = rest.find(MARKER) {
        let after = &rest[at + MARKER.len()..];
        let end = after.find(MARK).ok_or("a malformed image marker in the prompt")?;
        let index: usize = after[..end].parse().map_err(|_| "a malformed image marker in the prompt")?;
        if index >= count || std::mem::replace(&mut seen[index], true) {
            return Err("an image placed twice in the prompt".into());
        }
        if at > 0 {
            pieces.push(Piece::Text(&rest[..at]));
        }
        pieces.push(Piece::Image(index));
        rest = &after[end + MARK.len_utf8()..];
    }
    if !rest.is_empty() {
        pieces.push(Piece::Text(rest));
    }
    Ok(pieces)
}

/// Each placed image's index and its span's first position.
pub type Spans = Vec<(usize, usize)>;

/// The prompt's ids, each image's span in its place, and where each placed image's span starts.
pub fn ids(text: &Text, pieces: &[Piece<'_>], images: &[Image]) -> Result<(Vec<u32>, Spans), TextError> {
    let (mut ids, mut spans) = (Vec::new(), Vec::new());
    for piece in pieces {
        match *piece {
            Piece::Text(part) => ids.extend(text.encode(part)?),
            Piece::Image(index) => {
                spans.push((index, ids.len()));
                ids.extend(images[index].ids());
            }
        }
    }
    Ok((ids, spans))
}

#[cfg(test)]
mod tests {
    use super::*;
    use straydog_engine::pyjson;

    #[test]
    fn image_parts_become_markers() {
        let messages = pyjson::parse(r#"[{"role": "user", "content": [{"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAEC"}}]},
            {"role": "tool", "tool_call_id": "a", "content": [{"type": "image", "source": {"type": "base64", "data": "/w=="}}]}]"#).unwrap();
        let (taken, images) = take(&messages).unwrap();
        assert_eq!(images, [vec![0, 1, 2], vec![255]]);
        let Value::Array(list) = &taken else { panic!("messages") };
        let Some(Value::Array(parts)) = list[0].get("content") else { panic!("content") };
        assert_eq!(parts[1].get("text").and_then(Value::as_str), Some(marker(0).as_str()));
        assert!(take(&pyjson::parse(r#"[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]"#)
            .unwrap()).unwrap_err().contains("data: URL"));
    }

    #[test]
    fn a_prompt_is_cut_at_its_markers() {
        let prompt = format!("a{}b{}", marker(1), marker(0));
        assert_eq!(pieces(&prompt, 2).unwrap(), [Piece::Text("a"), Piece::Image(1), Piece::Text("b"), Piece::Image(0)]);
        assert!(pieces(&format!("{}{}", marker(0), marker(0)), 1).is_err());
    }

    /// A marker's place takes `<|vision_start|>`, one id a merged token and `<|vision_end|>`, the same ids for the same bytes.
    #[test]
    fn a_qwen_image_is_its_span_between_the_vision_tokens() {
        let text = Text::new(br#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"a":0},"unk_token":"a"}}"#).unwrap();
        // 512x256: 16 merged columns by 8 rows
        let image = qwen_vision::prepare(&vision::Rgb { width: 512, height: 256, pixels: vec![9; 512 * 256 * 3] }).unwrap();
        assert_eq!((image.grid(), image.tokens()), ((8, 16), 128));
        let images = [QwenImage { digest: [7; 32], prepared: image }];
        let prompt = format!("a{}a", marker(0));
        let pieces = pieces(&prompt, 1).unwrap();
        let (ids, spans) = qwen_ids(&text, &pieces, &images).unwrap();
        assert_eq!(spans, [(0, 2)]);
        assert_eq!(ids.len(), 1 + 1 + 128 + 1 + 1);
        assert_eq!((ids[1], ids[130]), (VISION_START, VISION_END));
        assert!(ids[2..130].iter().all(|id| *id >= IMAGE_TOKEN));
        assert_eq!(ids[3], ids[2] + 1);
        assert_eq!(qwen_ids(&text, &pieces, &images).unwrap().0, ids);
    }

    #[test]
    fn base64_decodes() {
        assert_eq!(base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64("aGVs\nbG8").unwrap(), b"hello");
        assert!(base64("a*b").is_err());
    }
}
