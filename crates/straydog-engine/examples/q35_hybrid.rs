//! Partially resident Qwen3.5 MoE decode, measured and compared: a prompt (DUMP_DIR/tokens.txt, or
//! --tokens FILE) prefilled in chunks, then decode steps (greedy, or --teacher 1 feeding the
//! continuation's tokens) with the experts in VRAM slots (--slots), a host tier (--host-gb) and the
//! host CPU taking a decode step's routes outside VRAM (--cpu threads). --split L runs layers 0..L
//! on GPU 0 and the rest on GPU 1. Reports the decode rate over the steps after --settle steps,
//! routes per token by tier, and the CPU expert engine's time; --record FILE writes the logits after
//! every prefill chunk and decode step, --against FILE compares this run's with them (KL, top-1,
//! equal logits); --save FILE writes the prompt and the greedy tokens as a tokens file. Development
//! only (unauthenticated weights).
//!
//! q35_hybrid MODEL_DIR DUMP_DIR [--chunk N] [--generate N] [--settle N] [--slots N] [--host-gb N] [--cpu N] [--split L]
//!            [--context N] [--gpu N] [--tokens FILE] [--teacher 1] [--record FILE] [--against FILE] [--save FILE]

#[path = "common/files.rs"]
mod files;

use std::path::Path;

use straydog_engine::arch::qwen35moe::engine::{forward_split, Engine, Settings};
use straydog_engine::arch::qwen35moe::part::Part;
use straydog_engine::arch::qwen35moe::Hparams;
use straydog_gpu::ffi;

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

fn compare(rows: &[Vec<f32>], path: &str, prefill: usize) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|exc| exc.to_string())?;
    let vocabulary = rows.first().map_or(1, Vec::len);
    let theirs: Vec<Vec<f32>> = bytes.chunks_exact(4 * vocabulary)
        .map(|row| row.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()).collect();
    if theirs.len() != rows.len() {
        return Err(format!("{path} holds {} probes, this run {}", theirs.len(), rows.len()));
    }
    for (label, range) in [("prefill chunks", 0..prefill), ("decode steps", prefill..rows.len())] {
        let (mut kls, mut top, mut equal) = (Vec::new(), 0, 0);
        for index in range {
            let (a, b) = (&theirs[index], &rows[index]);
            kls.push(kl(a, b));
            top += usize::from(argmax(a) == argmax(b));
            equal += usize::from(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
        if kls.is_empty() {
            continue;
        }
        let mean = kls.iter().sum::<f64>() / kls.len() as f64;
        let worst = kls.iter().cloned().fold(0f64, f64::max);
        println!("against {path}: {label}: {} probes, KL(theirs||this) mean {mean:.3e} max {worst:.3e}, top-1 {top}/{}, bitwise equal \
                  {equal}/{}", kls.len(), kls.len(), kls.len());
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut option = |flag: &str, default: usize| args.iter().position(|arg| arg == flag).map_or(default, |at| {
        let value = args[at + 1].parse().unwrap();
        args.drain(at..at + 2);
        value
    });
    let (chunk, generate, settle, slots) = (option("--chunk", 64), option("--generate", 128), option("--settle", 32), option("--slots", 226));
    let (host_gb, cpu_threads, split, context) = (option("--host-gb", 22), option("--cpu", 6), option("--split", 0), option("--context", 16384));
    let (gpu, teacher) = (option("--gpu", 0), option("--teacher", 0) > 0);
    let mut text_option = |flag: &str| args.iter().position(|arg| arg == flag).map(|at| {
        let value = args[at + 1].clone();
        args.drain(at..at + 2);
        value
    });
    let (tokens_file, record, against) = (text_option("--tokens"), text_option("--record"), text_option("--against"));
    let save = text_option("--save");
    let [dir, dump] = &args[..] else { return Err("usage: q35_hybrid MODEL_DIR DUMP_DIR [options] (see the source)".into()) };
    let files = files::Files::open(Path::new(dir))?;
    let hp = Hparams::from_header(&files.header)?;
    let tokens_path = tokens_file.map_or_else(|| Path::new(dump).join("tokens.txt"), std::path::PathBuf::from);
    let text = std::fs::read_to_string(&tokens_path).map_err(|exc| exc.to_string())?;
    let mut lines = text.lines();
    let parse = |line: Option<&str>| -> Vec<u32> { line.unwrap_or("").split_whitespace().map(|id| id.parse().unwrap()).collect() };
    let (prompt, continuation) = (parse(lines.next()), parse(lines.next()));
    let layers = hp.layers();
    let parts: Vec<_> = if split == 0 { vec![0..layers] } else { vec![0..split, split..layers] };
    let started = std::time::Instant::now();
    let mut engines = Vec::new();
    let host_bytes = (host_gb as u64) << 30;
    for (index, part) in parts.iter().enumerate() {
        let share = (host_bytes as f64 * part.len() as f64 / layers as f64) as u64;
        let settings = Settings { context, batch: chunk.max(1), expert_slots: slots, host_expert_bytes: share, scratch_bytes: 512 << 20,
                                  layers: part.clone(), checkpoints: 0, resident: false, part: Part::WHOLE, half_gemm: false };
        engines.push(Engine::new(hp.clone(), &files, gpu + index, 31 << 30, settings)?);
    }
    let cpu = std::rc::Rc::new(straydog_engine::gpu::CpuExperts::new(cpu_threads as u32)?);
    for engine in &mut engines {
        engine.set_cpu(cpu.clone());
    }
    println!("{layers} layers on {} GPU(s), {slots} slots and {host_gb} GiB of host tier each part, placed in {:.1} s", engines.len(),
             started.elapsed().as_secs_f64());
    let counts = |engines: &[Engine<'_, files::Files>]| engines.iter().map(|engine| engine.cache_counts())
        .fold((0u64, 0u64, 0u64), |a, b| (a.0 + b.hits, a.1 + b.host_hits, a.2 + b.drive_reads));
    let cpu_stats = || {
        let mut values = [0u64; 5];
        // SAFETY: a live engine and five counters.
        unsafe { ffi::sd_cpu_experts_stats(cpu.handle(), values.as_mut_ptr()) };
        values
    };
    let mut rows = Vec::new();
    let started = std::time::Instant::now();
    let mut logits = Vec::new();
    for piece in prompt.chunks(chunk.max(1)) {
        logits = forward_split(&mut engines, piece)?;
        rows.push(logits.clone());
    }
    let prefill = started.elapsed().as_secs_f64();
    println!("prompt of {} tokens in {prefill:.2} s ({:.1} tok/s)", prompt.len(), prompt.len() as f64 / prefill);
    let mut output = vec![argmax(&logits) as u32];
    let mut fed = vec![if teacher { *continuation.first().ok_or("--teacher needs a continuation")? } else { output[0] }];
    let (mut before, mut cpu_before, mut started, mut routes_before) = (counts(&engines), cpu_stats(), std::time::Instant::now(), 0u64);
    let mut step_seconds = Vec::new();
    let timing = |engines: &[Engine<'_, files::Files>]| engines.iter().map(Engine::hybrid_timing)
        .fold((0u64, [0f64; 5], [0u64; 2]), |a, b| (a.0 + b.0, std::array::from_fn(|i| a.1[i] + b.1[i]), [a.2[0] + b.2[0],
                                                                                                          a.2[1] + b.2[1]]));
    let mut timing_before = timing(&engines);
    while output.len() < generate.max(1) {
        if output.len() == settle + 1 {
            before = counts(&engines);
            cpu_before = cpu_stats();
            routes_before = engines.iter().map(Engine::cpu_routes).sum();
            timing_before = timing(&engines);
            started = std::time::Instant::now();
            step_seconds.clear();
        }
        let step = std::time::Instant::now();
        logits = forward_split(&mut engines, &fed[fed.len() - 1..])?;
        step_seconds.push(step.elapsed().as_secs_f64());
        rows.push(logits.clone());
        output.push(argmax(&logits) as u32);
        let next = if teacher { *continuation.get(fed.len()).ok_or("--teacher past the continuation")? } else { output[output.len() - 1] };
        fed.push(next);
    }
    let seconds = started.elapsed().as_secs_f64();
    let (after, cpu_after) = (counts(&engines), cpu_stats());
    let steps = step_seconds.len().max(1) as f64;
    let cpu_routes: u64 = engines.iter().map(Engine::cpu_routes).sum::<u64>() - routes_before;
    step_seconds.sort_by(f64::total_cmp);
    let quantile = |q: f64| step_seconds.get(((step_seconds.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(0.0) * 1e3;
    println!("decode: {} steps after {settle} at {:.2} tok/s ({:.2} ms; p50 {:.2} p95 {:.2}); per token {:.1} VRAM hits, {:.1} host hits, \
              {:.1} drive reads, {:.1} CPU routes", step_seconds.len(), steps / seconds, seconds / steps * 1e3, quantile(0.5), quantile(0.95),
             (after.0 - before.0) as f64 / steps, (after.1 - before.1) as f64 / steps, (after.2 - before.2) as f64 / steps,
             cpu_routes as f64 / steps);
    let cpu_ms = |index: usize| (cpu_after[index] - cpu_before[index]) as f64 / 1e6 / steps;
    println!("cpu experts per token: {:.1} calls, {:.1} experts, {:.2} ms (projections {:.2}, products {:.2}, down {:.2})",
             (cpu_after[0] - cpu_before[0]) as f64 / steps, (cpu_after[1] - cpu_before[1]) as f64 / steps,
             cpu_ms(2) + cpu_ms(3) + cpu_ms(4), cpu_ms(2), cpu_ms(3), cpu_ms(4));
    let timing_after = timing(&engines);
    let host_ms = |index: usize| (timing_after.1[index] - timing_before.1[index]) * 1e3 / steps;
    let served: f64 = (0..5).map(host_ms).sum();
    println!("host per token (pipelined steps {}): waiting for exports {:.2} ms, CPU experts {:.2} ms, drive reads {:.2} ms, table \
              {:.2} ms, queuing {:.2} ms, the rest {:.2} ms", timing_after.0 - timing_before.0, host_ms(0), host_ms(1), host_ms(2),
             host_ms(3), host_ms(4), seconds / steps * 1e3 - served);
    println!("drive: per token {:.2} routes waited for a read, {:.2} reads started on a guess",
             (timing_after.2[0] - timing_before.2[0]) as f64 / steps, (timing_after.2[1] - timing_before.2[1]) as f64 / steps);
    println!("greedy:    {:?}", &output[..output.len().min(64)]);
    if !teacher {
        let same = output.iter().zip(&continuation).take_while(|(a, b)| a == b).count();
        println!("{same} of {} tokens match the reference continuation", output.len().min(continuation.len()));
    }
    if let Some(path) = &save {
        let line = |tokens: &[u32]| tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
        std::fs::write(path, format!("{}\n{}\n", line(&prompt), line(&output))).map_err(|exc| exc.to_string())?;
    }
    let prefill_probes = prompt.len().div_ceil(chunk.max(1));
    if let Some(path) = &record {
        let bytes: Vec<u8> = rows.iter().flatten().flat_map(|value| value.to_le_bytes()).collect();
        std::fs::write(path, bytes).map_err(|exc| exc.to_string())?;
        println!("recorded {} probes to {path}", rows.len());
    }
    if let Some(path) = &against {
        compare(&rows, path, prefill_probes)?;
    }
    for engine in &mut engines {
        engine.settle()?;
    }
    Ok(())
}
