//! DeepSeek-V4.1's output distribution against a reference run's logits (the llama.cpp probe:
//! scratchpad/ds41probe), by teacher forcing on the real decode path: each prompt is prefilled,
//! then the reference's own generated tokens are fed one at a time, and every step's logits are
//! compared with the reference's for the same position. Reports, per prompt and over all:
//! KL(reference || ours) mean, p99 and maximum, top-1 agreement, and the perplexity of the
//! reference's tokens under both. A hash of every logit produced tells runs apart bit for bit
//! (a nondeterminism check: the same build and inputs must give the same hash).
//!
//! ds41_quality SHARD_DIR TOKENS[,TOKENS...] [--chunk N] [--batch N] [--slots N] [--split L] [--tp N] [--host-gb N] [--row-cache MIB]
//!     [--steps N] [--show KL] [--context N]
//! --show prints every position whose KL exceeds KL, with both sides' top tokens. With
//! DS41_QUALITY_SAVE=DIR, each prompt's logits (its last position's, then one row per step) go to
//! DIR/prompt-INDEX.logits as F32, to compare two of our runs.
//! Each TOKENS file is the probe's prompt-N.tokens; its logits are read from prompt-N.logits beside it.

#[path = "common/files.rs"]
mod files;

use std::path::Path;

use straydog_engine::arch::deepseek41::engine::{forward_split, Engine, Settings};
use straydog_engine::arch::deepseek41::Hparams;

/// Log-softmax in F64.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
    let log_sum = sum.ln() + max;
    logits.iter().map(|&l| l as f64 - log_sum).collect()
}

fn argmax(logits: &[f32]) -> usize {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

/// One position: KL(reference || ours), whether the top tokens agree, and the next token's
/// negative log-likelihood under ours and under the reference.
struct Step {
    kl: f64,
    top1: bool,
    nll: (f64, f64),
    /// Each side's top token and its probability: ours, the reference's.
    tops: [(usize, f64); 2],
}

fn compare(ours: &[f32], reference: &[f32], next: Option<usize>) -> Step {
    let (p, q) = (log_softmax(reference), log_softmax(ours));
    let kl = p.iter().zip(&q).map(|(&lp, &lq)| lp.exp() * (lp - lq)).sum::<f64>().max(0.0);
    let nll = next.map_or((0.0, 0.0), |token| (-q[token], -p[token]));
    let (a, b) = (argmax(ours), argmax(reference));
    Step { kl, top1: a == b, nll, tops: [(a, q[a].exp()), (b, p[b].exp())] }
}

/// FNV-1a over the logits' bits.
fn hash(state: &mut u64, logits: &[f32]) {
    for value in logits {
        for byte in value.to_bits().to_le_bytes() {
            *state ^= byte as u64;
            *state = state.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

fn summary(label: &str, steps: &[Step]) -> String {
    let mut kls: Vec<f64> = steps.iter().map(|step| step.kl).collect();
    kls.sort_by(f64::total_cmp);
    let n = steps.len().max(1) as f64;
    let mean = kls.iter().sum::<f64>() / n;
    let p99 = kls.get(((kls.len() as f64 * 0.99).ceil() as usize).saturating_sub(1)).copied().unwrap_or(0.0);
    let top1 = steps.iter().filter(|step| step.top1).count() as f64 / n;
    let scored: Vec<&Step> = steps.iter().filter(|step| step.nll != (0.0, 0.0)).collect();
    let m = scored.len().max(1) as f64;
    let (ours, reference) = (scored.iter().map(|step| step.nll.0).sum::<f64>() / m, scored.iter().map(|step| step.nll.1).sum::<f64>() / m);
    format!("{label}: {} positions, KL mean {mean:.3e} p99 {p99:.3e} max {:.3e}, top-1 {:.1}%, PPL ours {:.4} reference {:.4}",
            steps.len(), kls.last().copied().unwrap_or(0.0), 100.0 * top1, ours.exp(), reference.exp())
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut option = |flag: &str, default: usize| args.iter().position(|arg| arg == flag).map_or(default, |at| {
        let value = args[at + 1].parse().unwrap();
        args.drain(at..at + 2);
        value
    });
    let (chunk, slots, split, tp) = (option("--chunk", 64), option("--slots", 1900), option("--split", 0), option("--tp", 1));
    let (host_gb, row_cache, max_steps) = (option("--host-gb", 0), option("--row-cache", 256) as u64, option("--steps", usize::MAX));
    // rows the attention runs at a time within a pass of up to --chunk tokens
    let batch = option("--batch", 256);
    let context = option("--context", 8192);
    let show = args.iter().position(|arg| arg == "--show").map(|at| {
        let value: f64 = args[at + 1].parse().unwrap();
        args.drain(at..at + 2);
        value
    });
    let [dir, tokens_files] = &args[..] else {
        return Err("usage: ds41_quality SHARD_DIR TOKENS[,TOKENS...] [--chunk N] [--batch N] [--slots N] [--split L] [--tp N] [--host-gb N] \
                    [--row-cache MIB] [--steps N] [--context N]".into());
    };
    let mut files = files::Files::open(Path::new(dir))?;
    if row_cache > 0 && straydog_engine::arch::Tensors::tensor(&files, "per_layer_token_embd.weight").is_ok() {
        files.cache_rows("per_layer_token_embd.weight", row_cache << 20)?;
    }
    let hp = Hparams::from_header(&files.header)?;
    let vocabulary = hp.vocabulary;
    let layers = hp.layers.len();
    if split != 0 && tp > 1 {
        return Err("--split and --tp do not combine".into());
    }
    let parts: Vec<_> = if split == 0 { std::iter::once(0..layers).collect() } else { vec![0..split, split..layers] };
    let devices = (parts.len() * tp) as u64;
    let mut engines = Vec::new();
    for (index, part) in parts.into_iter().enumerate() {
        let settings = Settings { context, rows: chunk.max(batch), batch, expert_slots: slots, host_expert_bytes: ((host_gb as u64) << 30) / devices,
                                  scratch_bytes: 512 << 20, layers: part };
        let devices: Vec<usize> = (index * tp..(index + 1) * tp).collect();
        engines.push(Engine::new(hp.clone(), &files, &devices, 31 << 30, settings)?);
    }
    let mut all = Vec::new();
    let mut run_hash = 0xcbf2_9ce4_8422_2325u64;
    for (index, tokens_file) in tokens_files.split(',').enumerate() {
        let raw = std::fs::read(tokens_file).map_err(|exc| format!("{tokens_file}: {exc}"))?;
        let ints: Vec<i32> = raw.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
        let (count, kept) = (ints[0] as usize, ints[1] as usize);
        let prompt: Vec<u32> = ints[3..3 + count].iter().map(|&id| id as u32).collect();
        let generated: Vec<u32> = ints[3 + count..].iter().map(|&id| id as u32).collect();
        let logits_path = Path::new(tokens_file).with_extension("logits");
        let bytes = std::fs::read(&logits_path).map_err(|exc| format!("{}: {exc}", logits_path.display()))?;
        let rows: Vec<&[u8]> = bytes.chunks_exact(vocabulary * 4).collect();
        let row = |at: usize| -> Vec<f32> { rows[at].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect() };
        // Rows: the prompt's last `kept` positions, then one after each generated token but an
        // end-of-generation one.
        let decode_rows = rows.len().saturating_sub(kept);
        engines.iter_mut().for_each(|engine| engine.reset());
        let mut logits = Vec::new();
        for piece in prompt.chunks(chunk) {
            logits = forward_split(&mut engines, piece)?;
        }
        let mut prompt_hash = 0xcbf2_9ce4_8422_2325u64;
        hash(&mut prompt_hash, &logits);
        let mut saved: Vec<u8> = logits.iter().flat_map(|value| value.to_le_bytes()).collect();
        let mut steps = vec![compare(&logits, &row(kept - 1), generated.first().map(|&token| token as usize))];
        for step in 0..decode_rows.min(generated.len()).min(max_steps) {
            logits = forward_split(&mut engines, &generated[step..step + 1])?;
            hash(&mut prompt_hash, &logits);
            saved.extend(logits.iter().flat_map(|value| value.to_le_bytes()));
            steps.push(compare(&logits, &row(kept + step), generated.get(step + 1).map(|&token| token as usize)));
        }
        hash(&mut run_hash, &[f32::from_bits(prompt_hash as u32), f32::from_bits((prompt_hash >> 32) as u32)]);
        println!("{} | logits hash {prompt_hash:016x}", summary(&format!("prompt {index}"), &steps));
        if let Some(dir) = std::env::var_os("DS41_QUALITY_SAVE") {
            std::fs::write(Path::new(&dir).join(format!("prompt-{index}.logits")), &saved).map_err(|exc| exc.to_string())?;
        }
        if let Some(threshold) = show {
            for (at, step) in steps.iter().enumerate().filter(|(_, step)| step.kl > threshold) {
                // position: the prompt's last (step 0), then after each generated token
                println!("  position {}: KL {:.3e}, ours {} ({:.3}), reference {} ({:.3})", count - 1 + at, step.kl, step.tops[0].0,
                         step.tops[0].1, step.tops[1].0, step.tops[1].1);
            }
        }
        all.extend(steps);
    }
    println!("{} | run hash {run_hash:016x}", summary("all", &all));
    for engine in &mut engines {
        engine.settle()?;
    }
    Ok(())
}
