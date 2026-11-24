//! A rendered prompt's tokens as the model's own tokenizer (its GGUF's) gives them, written as a TOKENS file of
//! ds41_quality's form ([the prompt's length, 1, 0, its tokens], little-endian i32): a benchmark prompt made from a
//! conversation (the chat template's text, special tokens and all).
//!
//! ds41_tokens SHARD.gguf TEXT OUT

use std::fs::File;

use memmap2::Mmap;
use straydog_engine::text::Text;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let [_, shard, text, out] = &args[..] else { return Err("ds41_tokens SHARD.gguf TEXT OUT".into()) };
    let file = File::open(shard).map_err(|exc| format!("{shard}: {exc}"))?;
    // SAFETY: a read-only mapping of a file this process never writes.
    let map = unsafe { Mmap::map(&file) }.map_err(|exc| exc.to_string())?;
    let header = straydog_gguf::inspect(&mut std::io::Cursor::new(&map[..])).map_err(|exc| exc.to_string())?;
    let tokenizer = Text::from_gguf(&header).map_err(|exc| exc.0)?;
    let prompt = std::fs::read_to_string(text).map_err(|exc| format!("{text}: {exc}"))?;
    let ids = tokenizer.encode(&prompt).map_err(|exc| exc.0)?;
    let mut bytes = Vec::with_capacity(4 * (ids.len() + 3));
    for value in [ids.len() as i32, 1, 0].into_iter().chain(ids.iter().map(|&id| id as i32)) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(out, bytes).map_err(|exc| format!("{out}: {exc}"))?;
    println!("{} tokens; the last ones: {:?}", ids.len(), tokenizer.decode(&ids[ids.len().saturating_sub(12)..]).map_err(|exc| exc.0)?);
    Ok(())
}
