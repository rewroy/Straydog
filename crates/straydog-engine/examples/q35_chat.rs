//! Qwen3.5 MoE (`qwen35moe`) on the GPU engine, checked against the reference run: the prompt of
//! DUMP_DIR/tokens.txt fed in chunks, the logits after it compared with the reference's last row
//! (DUMP_DIR/result_output.f32), then a greedy continuation compared with the reference's.
//! Development only (unauthenticated weights). With --split L, layers 0..L run on GPU 0 and the
//! rest on GPU 1. Q35_COMPARE=DIR compares each layer's tensors with DIR's as the passes go (the
//! prompt in one chunk). With --checkpoint N, the prompt is run again with a checkpoint N tokens
//! before its end, other tokens after it, the checkpoint restored and the last N run again: the
//! logits must be the same. With --resident 1, every expert is read into VRAM first and run by
//! table (a decode step as one recorded graph); --slots is then unused.
//!
//! q35_chat MODEL_DIR DUMP_DIR [--chunk N] [--generate N] [--slots N] [--split L] [--host-gb N] [--context N] [--checkpoint N] [--resident 1]

#[path = "common/files.rs"]
mod files;

use std::path::Path;

use straydog_engine::arch::qwen35moe::engine::{forward_split, Engine, Settings};
use straydog_engine::arch::qwen35moe::Hparams;

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut option = |flag: &str, default: usize| args.iter().position(|arg| arg == flag).map_or(default, |at| {
        let value = args[at + 1].parse().unwrap();
        args.drain(at..at + 2);
        value
    });
    let (chunk, generate, slots, split) = (option("--chunk", 16), option("--generate", 16), option("--slots", 2000), option("--split", 0));
    let (host_gb, context, checkpoint) = (option("--host-gb", 0), option("--context", 4096), option("--checkpoint", 0));
    let resident = option("--resident", 0) > 0;
    let [dir, dump] = &args[..] else {
        return Err("usage: q35_chat MODEL_DIR DUMP_DIR [--chunk N] [--generate N] [--slots N] [--split L] [--host-gb N] [--context N] \
                    [--checkpoint N] [--resident 1]".into());
    };
    let files = files::Files::open(Path::new(dir))?;
    let hp = Hparams::from_header(&files.header)?;
    let text = std::fs::read_to_string(Path::new(dump).join("tokens.txt")).map_err(|exc| exc.to_string())?;
    let mut lines = text.lines();
    let parse = |line: Option<&str>| -> Vec<u32> { line.unwrap_or("").split_whitespace().map(|id| id.parse().unwrap()).collect() };
    let (prompt, continuation) = (parse(lines.next()), parse(lines.next()));
    let layers = hp.layers();
    let parts: Vec<_> = if split == 0 { std::iter::once(0..layers).collect() } else { vec![0..split, split..layers] };
    let started = std::time::Instant::now();
    let mut engines = Vec::new();
    for (device, part) in parts.into_iter().enumerate() {
        let settings = Settings { context, batch: chunk.max(1), expert_slots: slots, host_expert_bytes: (host_gb as u64) << 30,
                                  scratch_bytes: 512 << 20, layers: part, checkpoints: usize::from(checkpoint > 0), resident };
        engines.push(Engine::new(hp.clone(), &files, device, 31 << 30, settings)?);
    }
    println!("{} layers on {} GPU(s), placed in {:.1} s (resident experts: {})", layers, engines.len(), started.elapsed().as_secs_f64(),
             engines.iter().all(Engine::is_resident));
    let counts = |engines: &[Engine<'_, files::Files>]| engines.iter().map(|engine| engine.cache_counts())
        .fold((0u64, 0u64, 0u64), |a, b| (a.0 + b.hits, a.1 + b.host_hits, a.2 + b.drive_reads));
    let started = std::time::Instant::now();
    let mut logits = Vec::new();
    for piece in prompt.chunks(chunk.max(1)) {
        logits = forward_split(&mut engines, piece)?;
    }
    let prefill = started.elapsed().as_secs_f64();
    println!("prompt of {} tokens in {prefill:.2} s ({:.1} tok/s)", prompt.len(), prompt.len() as f64 / prefill);
    let bytes = std::fs::read(Path::new(dump).join("result_output.f32")).map_err(|exc| exc.to_string())?;
    let reference: Vec<f32> = bytes[32..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let vocabulary = hp.vocabulary;
    let theirs = &reference[reference.len() - vocabulary..];
    let softmax = |v: &[f32]| {
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = v.iter().map(|x| (*x as f64 - max).exp()).sum();
        v.iter().map(|x| (*x as f64 - max).exp() / sum).collect::<Vec<f64>>()
    };
    let (p, q) = (softmax(theirs), softmax(&logits));
    let kl: f64 = p.iter().zip(&q).filter(|(a, _)| **a > 0.0).map(|(a, b)| a * (a / b.max(1e-300)).ln()).sum();
    let (diff, norm) = logits.iter().zip(theirs).fold((0f64, 0f64), |(d, n), (a, b)| (d + (*a as f64 - *b as f64).powi(2), n + (*b as f64).powi(2)));
    println!("last prompt position: KL(ref||gpu) {kl:.3e}, logits rel {:.2e}, top-1 {} vs {}", (diff / norm).sqrt(), argmax(theirs),
             argmax(&logits));
    let mut order: Vec<usize> = (0..vocabulary).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let leading: Vec<String> = order[..5].iter().map(|&t| format!("{t}:{:.3}", logits[t])).collect();
    println!("last prompt position's leading logits: {}", leading.join("  "));
    let before = counts(&engines);
    let started = std::time::Instant::now();
    let mut output = vec![argmax(&logits) as u32];
    while output.len() < generate.max(1) {
        logits = forward_split(&mut engines, &output[output.len() - 1..])?;
        output.push(argmax(&logits) as u32);
    }
    let seconds = started.elapsed().as_secs_f64();
    let after = counts(&engines);
    let steps = (output.len() - 1).max(1) as f64;
    println!("decode: {} tokens at {:.1} tok/s; per token {:.1} VRAM hits, {:.1} host hits, {:.1} drive reads", output.len() - 1,
             steps / seconds, (after.0 - before.0) as f64 / steps, (after.1 - before.1) as f64 / steps, (after.2 - before.2) as f64 / steps);
    println!("greedy:    {output:?}");
    println!("reference: {continuation:?}");
    let same = output.iter().zip(&continuation).take_while(|(a, b)| a == b).count();
    println!("{same} of {} tokens match", output.len().min(continuation.len()));
    if checkpoint > 0 {
        let cut = prompt.len().checked_sub(checkpoint).filter(|&cut| cut > 0).ok_or("--checkpoint past the prompt's start")?;
        for engine in &mut engines {
            engine.reset()?;
        }
        for piece in prompt[..cut].chunks(chunk.max(1)) {
            forward_split(&mut engines, piece)?;
        }
        for engine in &mut engines {
            engine.checkpoint(0, true)?;
        }
        let first = forward_split(&mut engines, &prompt[cut..])?;
        for piece in output.chunks(chunk.max(1)) {
            forward_split(&mut engines, piece)?;
        }
        for engine in &mut engines {
            engine.checkpoint(0, false)?;
        }
        let again = forward_split(&mut engines, &prompt[cut..])?;
        let differ = first.iter().zip(&again).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let largest = first.iter().zip(&again).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("checkpoint {checkpoint} tokens before the end, restored after {} more: {differ} logits differ (largest by {largest:e}); \
                  top-1 {} vs {}", output.len(), argmax(&first), argmax(&again));
    }
    for engine in &mut engines {
        engine.settle()?;
    }
    Ok(())
}
