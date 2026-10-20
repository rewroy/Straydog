//! DeepSeek-V4.1 through the engine (arch::deepseek41::engine): a prompt prefilled in chunks,
//! its last logits checked against the CPU reference, then greedy decoding with its speed and
//! the expert cache's hits. Development only (unauthenticated weights). With --split L, layers 0..L
//! run on GPU 0 and the rest on GPU 1, each with --slots expert slots.
//!
//! ds41_chat SHARD_DIR TOKENS ORACLE_DIR TOKENIZER_JSON [--chunk N] [--generate N] [--slots N] [--split L]

#[path = "common/files.rs"]
mod files;

use std::path::Path;

use straydog_engine::arch::deepseek41::engine::{forward_split, Engine, Settings};
use straydog_engine::arch::deepseek41::Hparams;
use straydog_engine::text::Text;

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut option = |flag: &str, default: usize| args.iter().position(|arg| arg == flag).map_or(default, |at| {
        let value = args[at + 1].parse().unwrap();
        args.drain(at..at + 2);
        value
    });
    let (chunk, generate, slots, split) = (option("--chunk", 16), option("--generate", 32), option("--slots", 1500), option("--split", 0));
    let [dir, tokens_file, oracle, tokenizer] = &args[..] else {
        return Err("usage: ds41_chat SHARD_DIR TOKENS ORACLE_DIR TOKENIZER_JSON [--chunk N] [--generate N] [--slots N] [--split L]".into());
    };
    let files = files::Files::open(Path::new(dir))?;
    let hp = Hparams::from_header(&files.header)?;
    let raw = std::fs::read(tokens_file).map_err(|exc| exc.to_string())?;
    let ints: Vec<i32> = raw.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let (count, kept) = (ints[0] as usize, ints[1] as usize);
    let prompt: Vec<u32> = ints[3..3 + count].iter().map(|&id| id as u32).collect();
    let reference: Vec<u32> = ints[3 + count..].iter().map(|&id| id as u32).collect();
    let vocabulary = hp.vocabulary;
    let started = std::time::Instant::now();
    let layers = hp.layers.len();
    let parts = if split == 0 { vec![0..layers] } else { vec![0..split, split..layers] };
    let mut engines = Vec::new();
    for (device, part) in parts.into_iter().enumerate() {
        let settings = Settings { context: 8192, batch: 64, expert_slots: slots, scratch_bytes: 512 << 20, layers: part };
        engines.push(Engine::new(hp.clone(), &files, device, 31 << 30, settings)?);
    }
    let counts = |engines: &[Engine<'_, files::Files>]| engines.iter().map(|engine| engine.cache_counts())
        .fold((0, 0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    println!("loaded in {:.1} s", started.elapsed().as_secs_f64());
    let started = std::time::Instant::now();
    let mut logits = Vec::new();
    for piece in prompt.chunks(chunk) {
        logits = forward_split(&mut engines, piece)?;
    }
    println!("prefill {} tokens in chunks of {chunk}: {:.2} s", prompt.len(), started.elapsed().as_secs_f64());
    // the oracle's last kept row is the prompt's last position
    let bytes = std::fs::read(Path::new(oracle).join("result_output.f32")).map_err(|exc| exc.to_string())?;
    let oracle: Vec<f32> = bytes[32..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let last = &oracle[(kept - 1) * vocabulary..kept * vocabulary];
    let (mut diff, mut norm) = (0f64, 0f64);
    for (a, b) in logits.iter().zip(last) {
        diff += (*a as f64 - *b as f64).powi(2);
        norm += (*b as f64).powi(2);
    }
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    println!("last logits vs the CPU reference: rel {:.2e}, argmax {} vs {}", (diff / norm).sqrt(), argmax(&logits), argmax(last));
    let text = Text::new(&std::fs::read(tokenizer).map_err(|exc| exc.to_string())?).map_err(|exc| exc.0)?;
    let mut output = vec![argmax(&logits)];
    let (hits0, misses0, fill0) = counts(&engines);
    let started = std::time::Instant::now();
    engines.iter_mut().for_each(|engine| drop(engine.take_profile()));
    while output.len() < generate {
        logits = forward_split(&mut engines, &output[output.len() - 1..])?;
        output.push(argmax(&logits));
    }
    let seconds = started.elapsed().as_secs_f64();
    let (hits, misses, fill) = counts(&engines);
    println!("decode {} tokens: {:.2} tok/s, expert cache {:.1}% hits ({} misses, {:.0}% of the time filling them)", output.len() - 1,
             (output.len() - 1) as f64 / seconds, 100.0 * (hits - hits0) as f64 / ((hits - hits0 + misses - misses0).max(1)) as f64,
             misses - misses0, 100.0 * (fill - fill0) / seconds);
    let mut profile = std::collections::BTreeMap::new();
    for engine in &mut engines {
        for (phase, seconds) in engine.take_profile() {
            *profile.entry(phase).or_insert(0.0) += seconds;
        }
    }
    if !profile.is_empty() {
        let steps = (output.len() - 1) as f64;
        println!("per token: {}", profile.iter().map(|(phase, s)| format!("{phase} {:.1} ms", 1e3 * s / steps)).collect::<Vec<_>>().join(", "));
    }
    let same = output.iter().zip(&reference).take_while(|(a, b)| a == b).count();
    println!("first {same} tokens equal llama.cpp's greedy ones");
    println!("text: {}", text.decode(&output).map_err(|exc| exc.0)?);
    Ok(())
}
