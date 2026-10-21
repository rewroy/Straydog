//! Qwen3.5 MoE (`qwen35moe`) on the GPU engine, checked against the reference run: the prompt of
//! DUMP_DIR/tokens.txt fed in chunks, the logits after it compared with the reference's last row
//! (DUMP_DIR/result_output.f32), then a greedy continuation compared with the reference's.
//! Development only (unauthenticated weights). With --split L, layers 0..L run on GPU 0 and the
//! rest on GPU 1. Q35_COMPARE=DIR compares each layer's tensors with DIR's as the passes go (the
//! prompt in one chunk). With --checkpoint N, the prompt is run again with a checkpoint N tokens
//! before its end, other tokens after it, the checkpoint restored and the last N run again: the
//! logits must be the same. With --resident 1, every expert is read into VRAM first and run by
//! table (a decode step as one recorded graph); --slots is then unused. With --cpu N, a decode
//! step's routes to experts in the host tier run on the CPU (N threads). With --tp 1, GPUs 0 and 1
//! each run a part of every layer (tensor parallel, experts resident).
//!
//! Against another run: --tokens FILE takes the prompt (and continuation) from FILE instead of
//! DUMP_DIR/tokens.txt (no reference logits then); --teacher 1 feeds the continuation's tokens
//! instead of the greedy ones; --record FILE writes the logits after every prefill chunk and decode
//! step, and --against FILE compares this run's with them (KL, top-1, equal logits).
//!
//! q35_chat MODEL_DIR DUMP_DIR [--chunk N] [--generate N] [--slots N] [--split L] [--host-gb N] [--context N] [--checkpoint N] [--resident 1]
//!          [--cpu N] [--tp 1] [--tokens FILE] [--teacher 1] [--record FILE] [--against FILE]

#[path = "common/files.rs"]
mod files;

use std::path::Path;

use straydog_engine::arch::qwen35moe::engine::{forward_parts, forward_split, link, Engine, Settings};
use straydog_engine::arch::qwen35moe::part::Part;
use straydog_engine::arch::qwen35moe::Hparams;

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn softmax(v: &[f32]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = v.iter().map(|x| (*x as f64 - max).exp()).sum();
    v.iter().map(|x| (*x as f64 - max).exp() / sum).collect()
}

/// KL(p||q) over the softmaxes of two logit rows.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (p, q) = (softmax(p), softmax(q));
    p.iter().zip(&q).filter(|(a, _)| **a > 0.0).map(|(a, b)| a * (a / b.max(1e-300)).ln()).sum()
}

/// Every probe's logits (after each prefill chunk and decode step), compared with a recorded run's.
struct Probes {
    rows: Vec<Vec<f32>>,
}

impl Probes {
    fn write(&self, path: &str) -> Result<(), String> {
        let bytes: Vec<u8> = self.rows.iter().flatten().flat_map(|value| value.to_le_bytes()).collect();
        std::fs::write(path, bytes).map_err(|exc| exc.to_string())
    }

    fn compare(&self, path: &str, prefill: usize) -> Result<(), String> {
        let bytes = std::fs::read(path).map_err(|exc| exc.to_string())?;
        let vocabulary = self.rows.first().map_or(1, Vec::len);
        let theirs: Vec<Vec<f32>> = bytes.chunks_exact(4 * vocabulary)
            .map(|row| row.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()).collect();
        if theirs.len() != self.rows.len() {
            return Err(format!("{path} holds {} probes, this run {}", theirs.len(), self.rows.len()));
        }
        for (label, range) in [("prefill chunks", 0..prefill), ("decode steps", prefill..self.rows.len())] {
            let (mut kls, mut top, mut equal) = (Vec::new(), 0, 0);
            for index in range.clone() {
                let (a, b) = (&theirs[index], &self.rows[index]);
                kls.push(kl(a, b));
                top += usize::from(argmax(a) == argmax(b));
                equal += usize::from(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
            }
            if kls.is_empty() {
                continue;
            }
            let mean = kls.iter().sum::<f64>() / kls.len() as f64;
            let worst = kls.iter().cloned().fold(0f64, f64::max);
            println!("against {path}: {label}: {} probes, KL(theirs||this) mean {mean:.3e} max {worst:.3e}, top-1 {top}/{}, \
                      bitwise equal {equal}/{}", kls.len(), kls.len(), kls.len());
        }
        Ok(())
    }
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
    let tp = option("--tp", 0) > 0;
    let resident = option("--resident", 0) > 0 || tp;
    let cpu_threads = option("--cpu", 0);
    let teacher = option("--teacher", 0) > 0;
    let mut text_option = |flag: &str| args.iter().position(|arg| arg == flag).map(|at| {
        let value = args[at + 1].clone();
        args.drain(at..at + 2);
        value
    });
    let (tokens_file, record, against) = (text_option("--tokens"), text_option("--record"), text_option("--against"));
    let [dir, dump] = &args[..] else {
        return Err("usage: q35_chat MODEL_DIR DUMP_DIR [--chunk N] [--generate N] [--slots N] [--split L] [--host-gb N] [--context N] \
                    [--checkpoint N] [--resident 1] [--cpu N] [--tp 1] [--tokens FILE] [--teacher 1] [--record FILE] [--against FILE]".into());
    };
    let files = files::Files::open(Path::new(dir))?;
    let hp = Hparams::from_header(&files.header)?;
    let tokens_path = tokens_file.clone().map_or_else(|| Path::new(dump).join("tokens.txt"), std::path::PathBuf::from);
    let text = std::fs::read_to_string(&tokens_path).map_err(|exc| exc.to_string())?;
    let mut lines = text.lines();
    let parse = |line: Option<&str>| -> Vec<u32> { line.unwrap_or("").split_whitespace().map(|id| id.parse().unwrap()).collect() };
    let (prompt, continuation) = (parse(lines.next()), parse(lines.next()));
    let layers = hp.layers();
    let parts: Vec<_> = if tp {
        vec![(0..layers, Part { index: 0, count: 2 }), (0..layers, Part { index: 1, count: 2 })]
    } else if split == 0 {
        vec![(0..layers, Part::WHOLE)]
    } else {
        vec![(0..split, Part::WHOLE), (split..layers, Part::WHOLE)]
    };
    let started = std::time::Instant::now();
    let mut engines = Vec::new();
    for (device, (range, part)) in parts.into_iter().enumerate() {
        let settings = Settings { context, batch: chunk.max(1), expert_slots: slots, host_expert_bytes: (host_gb as u64) << 30,
                                  scratch_bytes: 512 << 20, layers: range, checkpoints: usize::from(checkpoint > 0), resident, part };
        engines.push(Engine::new(hp.clone(), &files, device, 31 << 30, settings)?);
    }
    if tp {
        link(&mut engines, chunk.max(1))?;
    }
    println!("{} layers on {} GPU(s){}, placed in {:.1} s (resident experts: {})", layers, engines.len(),
             if tp { ", each a part of every layer" } else { "" }, started.elapsed().as_secs_f64(), engines.iter().all(Engine::is_resident));
    let forward = |engines: &mut Vec<Engine<'_, files::Files>>, tokens: &[u32]| {
        if tp { forward_parts(engines, tokens) } else { forward_split(engines, tokens) }
    };
    let mut probes = Probes { rows: Vec::new() };
    if cpu_threads > 0 {
        let cpu = std::rc::Rc::new(straydog_engine::gpu::CpuExperts::new(cpu_threads as u32)?);
        for engine in &mut engines {
            engine.set_cpu(cpu.clone());
        }
    }
    let counts = |engines: &[Engine<'_, files::Files>]| engines.iter().map(|engine| engine.cache_counts())
        .fold((0u64, 0u64, 0u64), |a, b| (a.0 + b.hits, a.1 + b.host_hits, a.2 + b.drive_reads));
    let started = std::time::Instant::now();
    let mut logits = Vec::new();
    for piece in prompt.chunks(chunk.max(1)) {
        logits = forward(&mut engines, piece)?;
        probes.rows.push(logits.clone());
    }
    let prefill = started.elapsed().as_secs_f64();
    println!("prompt of {} tokens in {prefill:.2} s ({:.1} tok/s)", prompt.len(), prompt.len() as f64 / prefill);
    let vocabulary = hp.vocabulary;
    if tokens_file.is_none() {
        let bytes = std::fs::read(Path::new(dump).join("result_output.f32")).map_err(|exc| exc.to_string())?;
        let reference: Vec<f32> = bytes[32..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let theirs = &reference[reference.len() - vocabulary..];
        let (diff, norm) = logits.iter().zip(theirs)
            .fold((0f64, 0f64), |(d, n), (a, b)| (d + (*a as f64 - *b as f64).powi(2), n + (*b as f64).powi(2)));
        println!("last prompt position: KL(ref||gpu) {:.3e}, logits rel {:.2e}, top-1 {} vs {}", kl(theirs, &logits), (diff / norm).sqrt(),
                 argmax(theirs), argmax(&logits));
    }
    let mut order: Vec<usize> = (0..vocabulary).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let leading: Vec<String> = order[..5].iter().map(|&t| format!("{t}:{:.3}", logits[t])).collect();
    println!("last prompt position's leading logits: {}", leading.join("  "));
    let before = counts(&engines);
    let started = std::time::Instant::now();
    let mut output = vec![argmax(&logits) as u32];
    // The token each step feeds: the greedy one, or the continuation's (--teacher).
    let mut fed = vec![if teacher { *continuation.first().ok_or("--teacher needs a continuation")? } else { output[0] }];
    while output.len() < generate.max(1) {
        logits = forward(&mut engines, &fed[fed.len() - 1..])?;
        probes.rows.push(logits.clone());
        output.push(argmax(&logits) as u32);
        fed.push(if teacher { *continuation.get(fed.len()).ok_or("--teacher past the continuation")? } else { output[output.len() - 1] });
    }
    let seconds = started.elapsed().as_secs_f64();
    let after = counts(&engines);
    let steps = (output.len() - 1).max(1) as f64;
    let cpu_routes: u64 = engines.iter().map(Engine::cpu_routes).sum();
    println!("decode: {} tokens at {:.1} tok/s; per token {:.1} VRAM hits, {:.1} host hits, {:.1} drive reads, {:.1} CPU routes",
             output.len() - 1, steps / seconds, (after.0 - before.0) as f64 / steps, (after.1 - before.1) as f64 / steps,
             (after.2 - before.2) as f64 / steps, cpu_routes as f64 / steps);
    println!("greedy:    {output:?}");
    println!("reference: {continuation:?}");
    let same = output.iter().zip(&continuation).take_while(|(a, b)| a == b).count();
    println!("{same} of {} tokens match", output.len().min(continuation.len()));
    let prefill_probes = prompt.len().div_ceil(chunk.max(1));
    if let Some(path) = &record {
        probes.write(path)?;
        println!("recorded {} probes to {path}", probes.rows.len());
    }
    if let Some(path) = &against {
        probes.compare(path, prefill_probes)?;
    }
    if checkpoint > 0 {
        let cut = prompt.len().checked_sub(checkpoint).filter(|&cut| cut > 0).ok_or("--checkpoint past the prompt's start")?;
        for engine in &mut engines {
            engine.reset()?;
        }
        for piece in prompt[..cut].chunks(chunk.max(1)) {
            forward(&mut engines, piece)?;
        }
        for engine in &mut engines {
            engine.checkpoint(0, true)?;
        }
        let first = forward(&mut engines, &prompt[cut..])?;
        for piece in output.chunks(chunk.max(1)) {
            forward(&mut engines, piece)?;
        }
        for engine in &mut engines {
            engine.checkpoint(0, false)?;
        }
        let again = forward(&mut engines, &prompt[cut..])?;
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
