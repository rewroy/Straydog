//! A prompt's first block (the system prompt with its tools) as the bytes of the state a model holds there, so that a later start of the
//! engine can be handed it and take its first request from the end of that block (`Model::export_prefix`, `Model::import_prefix`).
//!
//! The bytes are a header, then the sections it lists, one after the other, all little-endian (the state never leaves its machine's kind):
//!
//! ```text
//! 0   8   "SDPREFIX"
//! 8   4   VERSION
//! 12  4   the tokens the state is held at (a multiple of four)
//! 16  4   the floats of the last token's hidden state (0: none)
//! 20  4   the sections: S
//! 24  16*S  each section's tag (what it is, on which device) and its byte count
//! ..      the sections: the tokens (u32), the hidden state (f32), then each device's recurrent state, K/V and indexer blocks, and the MTP
//!         layer's indexer history
//! ```
//!
//! A reader checks the header against the layout its own model would give a state of that many tokens (every tag and every byte count) before it
//! reads any section, and refuses what differs by a byte.

use sha2::Digest;

/// The layout of the bytes: changed with any change to what the sections hold or how they are ordered.
pub const VERSION: u32 = 1;
const MAGIC: [u8; 8] = *b"SDPREFIX";
const FIXED: usize = 24;
/// The most sections a header lists (a model's devices hold about a dozen each).
const MOST_SECTIONS: usize = 4096;

/// What a header says, and where its sections begin.
#[derive(Debug, PartialEq, Eq)]
pub struct Parsed<'a> {
    pub tokens: usize,
    pub hidden: usize,
    /// Every section, as (a tag for what it is, its bytes): the tokens', the hidden state's, then the devices'.
    pub sections: Vec<(u64, u64)>,
    /// The sections' bytes, one after the other.
    pub payload: &'a [u8],
    /// Where they begin in the bytes that were parsed: the header's length.
    pub start: usize,
}

/// The header of a state of `tokens` tokens and a `hidden`-float hidden state, with these sections.
pub fn header(tokens: usize, hidden: usize, sections: &[(u64, u64)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FIXED + 16 * sections.len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&(tokens as u32).to_le_bytes());
    bytes.extend_from_slice(&(hidden as u32).to_le_bytes());
    bytes.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    for (tag, count) in sections {
        bytes.extend_from_slice(&tag.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    bytes
}

/// The header of `bytes`, and its sections' bytes; an error when `bytes` are not a state of this version whose sections' counts add up to
/// exactly what follows the header.
pub fn parse(bytes: &[u8]) -> Result<Parsed<'_>, String> {
    let refused = |why: &str| Err(format!("Straydog prefix state refused: {why}"));
    if bytes.len() < FIXED || bytes[..8] != MAGIC {
        return refused("it is not a prefix state");
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    if word(8) != VERSION {
        return refused(&format!("its layout is version {}, this engine's is {VERSION}", word(8)));
    }
    let (tokens, hidden, count) = (word(12) as usize, word(16) as usize, word(20) as usize);
    if count > MOST_SECTIONS {
        return refused(&format!("it lists {count} sections"));
    }
    let start = FIXED + 16 * count;
    if bytes.len() < start {
        return refused("its header is short");
    }
    let mut sections = Vec::with_capacity(count);
    let mut total = 0u64;
    for index in 0..count {
        let at = FIXED + 16 * index;
        let tag = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        let held = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap());
        let Some(sum) = total.checked_add(held) else { return refused("its sections add up to more than a machine holds") };
        total = sum;
        sections.push((tag, held));
    }
    if total != (bytes.len() - start) as u64 {
        return refused(&format!("its sections add up to {total} bytes and {} follow its header", bytes.len() - start));
    }
    Ok(Parsed { tokens, hidden, sections, payload: &bytes[start..], start })
}

/// Whether `found` (a header's sections) is exactly `expected` (what this model's layout gives a state of that many tokens): every tag and
/// every byte count.
pub fn check(found: &[(u64, u64)], expected: &[(u64, u64)]) -> Result<(), String> {
    if found.len() != expected.len() {
        return Err(format!("Straydog prefix state refused: {} sections where this layout has {}", found.len(), expected.len()));
    }
    for (index, (have, want)) in found.iter().zip(expected).enumerate() {
        if have.0 != want.0 {
            return Err(format!("Straydog prefix state refused: section {index} is of another state than this layout's"));
        }
        if have.1 != want.1 {
            return Err(format!("Straydog prefix state refused: section {index} has {} bytes where this layout has {}", have.1, want.1));
        }
    }
    Ok(())
}

/// A digest of a layout (the sections of a state of a few tokens): the same for two models that would read each other's states, and for
/// no two that would not.
pub fn layout_digest(sections: &[(u64, u64)]) -> String {
    let mut hash = sha2::Sha256::new();
    hash.update(b"straydog prefix layout\n");
    hash.update(VERSION.to_le_bytes());
    for (tag, bytes) in sections {
        hash.update(tag.to_le_bytes());
        hash.update(bytes.to_le_bytes());
    }
    hex(&hash.finalize())
}

/// A digest of tokens (their bytes, little-endian), as hex: what names a prefix without sending it.
pub fn tokens_digest(tokens: &[u32]) -> String {
    let mut hash = sha2::Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    hex(&hash.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The tokens a state's payload begins with, `count` of them.
pub fn tokens_of(payload: &[u8], count: usize) -> Vec<u32> {
    payload[..4 * count].chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect()
}

/// `values` as little-endian bytes.
pub fn token_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(tokens: usize, sections: &[(u64, u64)]) -> Vec<u8> {
        let mut bytes = header(tokens, 0, sections);
        for (tag, count) in sections {
            bytes.extend(std::iter::repeat((*tag & 0xff) as u8).take(*count as usize));
        }
        bytes
    }

    /// A state's header reads back as it was written, with its payload after it.
    #[test]
    fn a_header_reads_back_with_its_payload() {
        let sections = [(1, 8), (2, 0), (3, 12)];
        let bytes = state(4, &sections);
        let parsed = parse(&bytes).unwrap();
        assert_eq!((parsed.tokens, parsed.hidden, parsed.sections.as_slice()), (4, 0, &sections[..]));
        assert_eq!(parsed.payload.len(), 20);
        assert_eq!(&parsed.payload[..8], &[1; 8]);
    }

    /// Anything that is not exactly a state of this version is refused: another magic, another version, a header cut short, sections that add up
    /// to more or to less than follow, a count of sections no model has, counts that overflow.
    #[test]
    fn what_is_not_exactly_a_state_is_refused() {
        let good = state(8, &[(1, 8), (2, 4)]);
        assert!(parse(&good).is_ok());
        let mut other = good.clone();
        other[0] ^= 1;
        assert!(parse(&other).unwrap_err().contains("not a prefix state"));
        let mut other = good.clone();
        other[8] = 2;
        assert!(parse(&other).unwrap_err().contains("version 2"));
        assert!(parse(&good[..FIXED - 1]).is_err());
        assert!(parse(&good[..FIXED + 20]).unwrap_err().contains("header is short"));
        assert!(parse(&good[..good.len() - 1]).unwrap_err().contains("add up"));
        let mut longer = good.clone();
        longer.push(0);
        assert!(parse(&longer).unwrap_err().contains("add up"));
        let mut many = good.clone();
        many[20..24].copy_from_slice(&(MOST_SECTIONS as u32 + 1).to_le_bytes());
        assert!(parse(&many).unwrap_err().contains("sections"));
        let huge = header(4, 0, &[(1, u64::MAX), (2, 2)]);
        assert!(parse(&huge).unwrap_err().contains("more than a machine holds"));
    }

    /// A state is read by a model whose layout it matches to the byte, and by no other: a tag, a count, or a section more or less.
    #[test]
    fn a_layout_matches_to_the_byte() {
        let layout = [(10, 16), (11, 32), (12, 0)];
        assert!(check(&layout, &layout).is_ok());
        assert!(check(&layout[..2], &layout).unwrap_err().contains("2 sections where this layout has 3"));
        let mut other = layout;
        other[1].1 += 4;
        assert!(check(&other, &layout).unwrap_err().contains("section 1 has 36 bytes where this layout has 32"));
        let mut other = layout;
        other[0].0 += 1;
        assert!(check(&other, &layout).unwrap_err().contains("section 0 is of another state"));
    }

    /// The layout digest follows every tag and every count, in order; the tokens' digest follows every token.
    #[test]
    fn the_digests_follow_what_they_digest() {
        let layout = [(10, 16), (11, 32)];
        assert_eq!(layout_digest(&layout), layout_digest(&layout));
        assert_ne!(layout_digest(&layout), layout_digest(&[(11, 32), (10, 16)]));
        assert_ne!(layout_digest(&layout), layout_digest(&[(10, 16), (11, 36)]));
        assert_ne!(layout_digest(&layout), layout_digest(&[(10, 16)]));
        assert_eq!(tokens_digest(&[1, 2, 3]).len(), 64);
        assert_ne!(tokens_digest(&[1, 2, 3]), tokens_digest(&[1, 2, 4]));
        assert_ne!(tokens_digest(&[1, 2, 3]), tokens_digest(&[1, 2]));
        assert_eq!(tokens_of(&token_bytes(&[7, 248046, 0]), 3), [7, 248046, 0]);
    }
}
