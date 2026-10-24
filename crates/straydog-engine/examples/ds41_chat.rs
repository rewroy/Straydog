//! DeepSeek-V4.1 through the engine (arch::deepseek41::engine): a prompt prefilled in chunks,
//! its last logits checked against the CPU reference, then greedy decoding with its speed and
//! where its experts came from. Development only (unauthenticated weights). With --split L, layers
//! 0..L run on GPU 0 and the rest on GPU 1; with --tp N, every layer is split across GPUs 0..N
//! (each with its share of the routed experts). Each GPU has --slots expert slots and its share of
//! --host-gb GB of host tier.
//!
//! Several TOKENS files (comma separated) run one after another, as a server's requests.
//!
//! ds41_chat SHARD_DIR TOKENS[,TOKENS...] ORACLE_DIR TOKENIZER_JSON [--chunk N] [--batch N] [--generate N] [--slots N] [--split L] [--tp N]
//!     [--host-gb N] [--row-cache MIB] [--context N]

#[path = "common/files.rs"]
mod files;

use std::path::Path;

use straydog_engine::arch::deepseek41::engine::{forward_split, Engine, Settings};
use straydog_engine::arch::deepseek41::Hparams;
use straydog_engine::text::Text;

/// With DS41_STATES=DIR: the engines' sequence state (see `Engine::state`) as DIR/prompt-INDEX-STAGE:
/// .f32 its values one entry after another, .txt each entry's name, values and FNV-1a hash.
fn dump_states(engines: &mut [Engine<'_, files::Files>], index: usize, stage: &str) -> Result<(), String> {
    let Some(dir) = std::env::var_os("DS41_STATES") else { return Ok(()) };
    let (mut values, mut lines) = (Vec::new(), String::new());
    for engine in engines {
        for (name, entry) in engine.state()? {
            let hash = entry.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, value| (hash ^ value.to_bits() as u64).wrapping_mul(0x100_0000_01b3));
            lines.push_str(&format!("{name} {} {hash:016x}\n", entry.len()));
            values.extend(entry.iter().flat_map(|value| value.to_le_bytes()));
        }
    }
    let base = Path::new(&dir).join(format!("prompt-{index}-{stage}"));
    std::fs::write(base.with_extension("f32"), values).map_err(|exc| exc.to_string())?;
    std::fs::write(base.with_extension("txt"), lines).map_err(|exc| exc.to_string())
}

/// Bytes this process has had read from storage so far (Linux; 0 elsewhere).
fn read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io").ok()
        .and_then(|io| io.lines().find_map(|line| line.strip_prefix("read_bytes: ").and_then(|value| value.trim().parse().ok())))
        .unwrap_or(0)
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut option = |flag: &str, default: usize| args.iter().position(|arg| arg == flag).map_or(default, |at| {
        let value = args[at + 1].parse().unwrap();
        args.drain(at..at + 2);
        value
    });
    let (chunk, generate, slots, split) = (option("--chunk", 16), option("--generate", 32), option("--slots", 1500), option("--split", 0));
    let (host_gb, tp) = (option("--host-gb", 0), option("--tp", 1));
    // rows the attention runs at a time within a pass of up to --chunk tokens
    let batch = option("--batch", 256);
    let context = option("--context", 8192);
    // The Engram table's row cache in MiB (0: rows through the mapping and the page cache).
    let row_cache = option("--row-cache", 256) as u64;
    let [dir, tokens_files, oracle, tokenizer] = &args[..] else {
        return Err("usage: ds41_chat SHARD_DIR TOKENS[,TOKENS...] ORACLE_DIR TOKENIZER_JSON [--chunk N] [--batch N] [--generate N] [--slots N] \
                    [--split L] [--tp N] [--host-gb N] [--row-cache MIB] [--context N]".into());
    };
    let mut files = files::Files::open(Path::new(dir))?;
    let hp = Hparams::from_model(&files.header, &files)?;
    // Engram's tables (one, or one a module) read through row caches sharing the budget
    let mut tables = hp.engram.as_ref().map_or(Vec::new(), |engram| engram.tables.clone());
    tables.dedup();
    for table in &tables {
        if row_cache > 0 {
            files.cache_rows(table, (row_cache << 20) / tables.len() as u64)?;
        }
    }
    let vocabulary = hp.vocabulary;
    let started = std::time::Instant::now();
    let layers = hp.layers.len();
    if split != 0 && tp > 1 {
        return Err("--split and --tp do not combine".into());
    }
    let parts: Vec<_> = if split == 0 { std::iter::once(0..layers).collect() } else { vec![0..split, split..layers] };
    let devices = (parts.len() * tp) as u64;
    let mut engines = Vec::new();
    for (index, part) in parts.into_iter().enumerate() {
        let settings = Settings { context, rows: chunk.max(batch), batch, expert_slots: slots, host_expert_bytes: ((host_gb as u64) << 30) / devices,
                                  scratch_bytes: 512 << 20, half_ffn: false, layers: part };
        let devices: Vec<usize> = (index * tp..(index + 1) * tp).collect();
        engines.push(Engine::new(hp.clone(), &files, &devices, 31 << 30, settings)?);
    }
    // VRAM hits, host-tier hits, drive reads, seconds: waiting for the drive, for host slots,
    // page-locking, in the store
    let counts = |engines: &[Engine<'_, files::Files>]| engines.iter().map(|engine| engine.cache_counts())
        .fold((0, 0, 0, [0.0; 4]), |a, b| (a.0 + b.hits, a.1 + b.host_hits, a.2 + b.drive_reads,
              [a.3[0] + b.wait_seconds, a.3[1] + b.slot_wait_seconds, a.3[2] + b.pin_seconds, a.3[3] + b.store_seconds]));
    println!("loaded in {:.1} s", started.elapsed().as_secs_f64());
    let text = Text::new(&std::fs::read(tokenizer).map_err(|exc| exc.to_string())?).map_err(|exc| exc.0)?;
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    // Prompts one after another, as a server's requests: the experts' tiers stay warm.
    for (index, tokens_file) in tokens_files.split(',').enumerate() {
        let raw = std::fs::read(tokens_file).map_err(|exc| exc.to_string())?;
        let ints: Vec<i32> = raw.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
        let (count, kept) = (ints[0] as usize, ints[1] as usize);
        let prompt: Vec<u32> = ints[3..3 + count].iter().map(|&id| id as u32).collect();
        let reference: Vec<u32> = ints[3 + count..].iter().map(|&id| id as u32).collect();
        engines.iter_mut().for_each(|engine| engine.reset());
        let (before, read_before) = (counts(&engines), read_bytes());
        let started = std::time::Instant::now();
        let mut logits = Vec::new();
        for piece in prompt.chunks(chunk) {
            logits = forward_split(&mut engines, piece)?;
        }
        let seconds = started.elapsed().as_secs_f64();
        let (after, read) = (counts(&engines), read_bytes() - read_before);
        println!("prompt {index}: prefill {} tokens in chunks of {chunk}: {seconds:.2} s, {:.1} tok/s; experts {} in VRAM, {} from the \
                  host tier, {} from the drive; {:.2} GB read from the drive", prompt.len(), prompt.len() as f64 / seconds,
                 after.0 - before.0, after.1 - before.1, after.2 - before.2, read as f64 / 1e9);
        if index == 0 {
            // the oracle's last kept row is the first prompt's last position
            let bytes = std::fs::read(Path::new(oracle).join("result_output.f32")).map_err(|exc| exc.to_string())?;
            let oracle: Vec<f32> = bytes[32..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let last = &oracle[(kept - 1) * vocabulary..kept * vocabulary];
            let (mut diff, mut norm) = (0f64, 0f64);
            for (a, b) in logits.iter().zip(last) {
                diff += (*a as f64 - *b as f64).powi(2);
                norm += (*b as f64).powi(2);
            }
            println!("last logits vs the CPU reference: rel {:.2e}, argmax {} vs {}", (diff / norm).sqrt(), argmax(&logits), argmax(last));
        }
        dump_states(&mut engines, index, "prefill")?;
        let hash = logits.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, value| (hash ^ value.to_bits() as u64).wrapping_mul(0x100_0000_01b3));
        println!("prompt {index}: prefill logits hash {hash:016x}");
        let mut output = vec![argmax(&logits)];
        let (hits0, host0, drive0, wait0) = counts(&engines);
        let started = std::time::Instant::now();
        let mut phases = std::collections::BTreeMap::new();
        for engine in &mut engines {
            for (phase, seconds) in engine.take_profile() {
                *phases.entry(phase).or_insert(0.0) += seconds;
            }
        }
        if !phases.is_empty() {
            println!("prompt {index}: prefill seconds: {}", phases.iter().map(|(phase, s)| format!("{phase} {s:.2}")).collect::<Vec<_>>().join(", "));
        }
        while output.len() < generate {
            logits = forward_split(&mut engines, &output[output.len() - 1..])?;
            output.push(argmax(&logits));
        }
        let seconds = started.elapsed().as_secs_f64();
        dump_states(&mut engines, index, "decode")?;
        let (hits, host, drive, wait) = counts(&engines);
        let steps = (output.len() - 1).max(1) as f64;
        let ms = |i: usize| 1e3 * (wait[i] - wait0[i]) / steps;
        println!("prompt {index}: decode {} tokens: {:.2} tok/s; per token {:.1} experts in VRAM, {:.1} from the host tier, {:.1} \
                  from the drive; ms waiting for the drive {:.1}, for host slots {:.1}, page-locking {:.1}, in the store {:.1}",
                 output.len() - 1, (output.len() - 1) as f64 / seconds, (hits - hits0) as f64 / steps, (host - host0) as f64 / steps,
                 (drive - drive0) as f64 / steps, ms(0), ms(1), ms(2), ms(3));
        let mut profile = std::collections::BTreeMap::new();
        for engine in &mut engines {
            for (phase, seconds) in engine.take_profile() {
                *profile.entry(phase).or_insert(0.0) += seconds;
            }
        }
        if !profile.is_empty() {
            println!("prompt {index}: per token: {}",
                     profile.iter().map(|(phase, s)| format!("{phase} {:.1} ms", 1e3 * s / steps)).collect::<Vec<_>>().join(", "));
        }
        println!("prompt {index}: generated {output:?}");
        let same = output.iter().zip(&reference).take_while(|(a, b)| a == b).count();
        println!("prompt {index}: first {same} tokens equal llama.cpp's greedy ones");
        if index == 0 {
            println!("text: {}", text.decode(&output).map_err(|exc| exc.0)?);
        }
    }
    for engine in &mut engines {
        engine.settle()?;
    }
    Ok(())
}
