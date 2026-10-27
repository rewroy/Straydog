//! DeepSeek-V4.1's output distribution against a reference run's logits (the llama.cpp probe:
//! scratchpad/ds41probe), by teacher forcing on the real decode path: each prompt is prefilled,
//! then the reference's own generated tokens are fed one at a time, and every step's logits are
//! compared with the reference's for the same position. Reports, per prompt and over all:
//! KL(reference || ours) mean, p99 and maximum, top-1 agreement, and the perplexity of the
//! reference's tokens under both. A hash of every logit produced tells runs apart bit for bit
//! (a nondeterminism check: the same build and inputs must give the same hash).
//!
//! ds41_quality SHARD_DIR TOKENS[,TOKENS...] [--chunk N] [--batch N] [--slots N] [--split L] [--tp N] [--host-gb N] [--row-cache MIB]
//!     [--steps N] [--show KL] [--context N] [--gemm fp32|fp16] [--verify N[,N...]] [--accept-all] [--dspark DIR [--draft-slots N[,N...]] [--show-drafts]]
//!     [--generate N [--runs R] [--width W]] [--dump-routes FILE] [--draft-bench K]
//! --verify N feeds the reference's tokens N at a time in verify passes, a pseudo-random number of
//! each pass's accepted and the rest taken back (Engine::rewind): the logits hashed are those of
//! the decode path, so the hashes must not change. With --accept-all every pass's are (passes of
//! N rows throughout, for their cost); several N run the prompts once each, in turn.
//! --show prints every position whose KL exceeds KL, with both sides' top tokens. With
//! DS41_QUALITY_SAVE=DIR, each prompt's logits (its last position's, then one row per step) go to
//! DIR/prompt-INDEX.logits as F32, to compare two of our runs.
//! Each TOKENS file is the probe's prompt-N.tokens; its logits are read from prompt-N.logits beside it.
//! --dspark DIR loads DSpark (the GGUF in DIR) beside the first device, feeds it the target's
//! features of every position, and before each decode step drafts a block of each --draft-slots
//! size (default 3,4; 0 drafts nothing) after that step's token, teacher forced (one engine,
//! --verify 0): per prompt and size, the mean count of a block's tokens that match the
//! reference's continuation as a prefix, and how often that prefix reaches each slot (slot 0
//! alone, slots 0-1, ...); --show-drafts prints every block. The logits hashed are the target's,
//! so the hashes must not change.
//! --generate N instead continues each prompt greedily by N tokens of its own (no reference) and
//! times it: per prompt the tokens' hash and milliseconds a token. With --dspark each round drafts
//! ahead and verifies as many of the draft's tokens as the width policy picks
//! (drafter::Policy, by DSpark's confidences and [`VERIFY_MS`]), keeps the matching ones and the
//! target's next, and takes the rest back: the tokens must be those without the drafter. --runs R
//! goes over the prompts R times (the first warms the expert stores). --width W verifies W of each
//! draft whatever the policy picks; --draft-slots 0 takes the features but never drafts. Each pass
//! is timed by its rows, and the stores' counters taken over the decode alone. --dump-routes FILE
//! writes every draft's routes, a line a stage (read back after the draft, outside its timing).
//! --draft-bench K prefills the first prompt (128 tokens or more), pauses a second, then drafts K
//! times at its end alone and prints their times (for a kernel trace of the drafts).
//! DSpark keeps its 87, 50 and 29 most used routed experts of each stage resident (the served
//! default); DS41_DSPARK_RESIDENT=A,B,C keeps other counts, `all` all 128.

#[path = "common/files.rs"]
mod files;

use std::fmt::Write as _;
use std::path::Path;

use straydog_engine::arch::deepseek41::engine::{forward_split, verify_split, Engine, Settings};
use straydog_engine::arch::deepseek41::drafter::{Drafter, Policy, DRAFT_MS, VERIFY_MS};
use straydog_engine::arch::deepseek41::engine::TAP_ROWS;
use straydog_engine::arch::deepseek41::dspark::Dspark;
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
    assert!(ours.iter().chain(reference).all(|v| v.is_finite()), "nonfinite quality logits");
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

/// The drafter's history: the last pass's last `n` rows' target features (positions up to the
/// engine's), from each device's tap.
fn inject(engines: &mut [Engine<'_, files::Files>], drafter: &mut Option<Drafter>, n: usize) -> Result<(), String> {
    let Some(drafter) = drafter else { return Ok(()) };
    let end = engines[0].position();
    let kept = n.min(TAP_ROWS);
    let positions: Vec<i32> = (end - kept..end).map(|at| at as i32).collect();
    drafter.inject(&mut engines[0].beside()?, &positions)
}

/// --generate's settings: tokens a prompt, prefill chunk, vocabulary, whether to print every round,
/// whether the drafter drafts (else it only takes the features) and a width that overrides the
/// policy's.
struct Generate {
    count: usize,
    chunk: usize,
    vocabulary: usize,
    show: bool,
    drafts: bool,
    width: Option<usize>,
    /// Which time over the prompts this is (--runs).
    run: usize,
}

/// --dump-routes: a draft's routes, a line a stage: the run, prompt, round, position, slots, stage,
/// then each slot's experts (by commas, slots by semicolons).
fn dump_routes(text: &mut String, run: usize, prompt: usize, round: usize, position: usize, slots: usize, routes: &[Vec<i32>]) {
    for (stage, ids) in routes.iter().enumerate() {
        let groups: Vec<String> = ids.chunks(ids.len() / slots.max(1)).map(|slot| {
            slot.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",")
        }).collect();
        let _ = writeln!(text, "{run}\t{prompt}\t{round}\t{position}\t{slots}\t{stage}\t{}", groups.join(";"));
    }
}

/// --generate: each prompt continued greedily by `count` tokens, timed; with a drafter, DSpark
/// drafts each round (but in a pause) and the policy (or the width given) picks how many of its
/// tokens a verify pass checks. Timed apart: each pass by its rows (a 1-row pass also by whether a
/// verify pass came before it), the drafts and the injections; the stores' counters over the decode.
fn generate(engines: &mut [Engine<'_, files::Files>], files: &files::Files, drafter: &mut Option<Drafter>, prompts: &[Vec<u32>],
            settings: &Generate, dump: &mut Option<String>) -> Result<(), String> {
    let (count, vocabulary) = (settings.count, settings.vocabulary);
    let mut policy = Policy::new(&VERIFY_MS, DRAFT_MS);
    let counted = |engines: &[Engine<'_, files::Files>]| {
        engines.iter().map(Engine::cache_counts).fold(straydog_engine::experts::Counts::default(), |a, b| a + b)
    };
    let (mut all_hash, mut all_seconds, mut all_tokens, mut all_passes) = (0xcbf2_9ce4_8422_2325u64, 0.0, 0, 0);
    // rounds, rounds drafted, verified widths 0..=3, verified drafts accepted, seconds drafting
    let (mut all_rounds, mut all_drafted, mut all_widths, mut all_accepted, mut all_drafting) = (0, 0, [0usize; 4], 0, 0.0);
    // passes by rows (seconds, count), 1-row passes after a verify pass, seconds injecting, drafts
    // after a full history (seconds, count: they replay graphs)
    let (mut by_rows, mut after_verify, mut injecting, mut full) = ([(0.0, 0usize); 4], (0.0, 0usize), 0.0, (0.0, 0usize));
    let mut decode = straydog_engine::experts::Counts::default();
    for (index, prompt) in prompts.iter().enumerate() {
        engines.iter_mut().for_each(|engine| engine.reset());
        if let Some(drafter) = drafter.as_mut() {
            drafter.forget_from(0);
        }
        policy.reset();
        let mut logits = Vec::new();
        for piece in prompt.chunks(settings.chunk) {
            logits = forward_split(engines, piece)?;
            inject(engines, drafter, piece.len())?;
        }
        let mut emitted = vec![argmax(&logits) as u32];
        let (mut passes, mut rounds, mut drafted, mut widths, mut accepted_all, mut drafting) = (0, 0, 0, [0usize; 4], 0, 0.0);
        let mut verified = false;
        let before = counted(engines);
        let started = std::time::Instant::now();
        while emitted.len() < count {
            let next = emitted[emitted.len() - 1];
            let position = engines[0].position();
            let room = count - emitted.len();
            let mut drafts = Vec::new();
            if let Some(drafter) = drafter.as_mut() {
                // (a width given drafts every round: no pause)
                if settings.drafts && room > 1 && (settings.width.is_some() || policy.drafting()) {
                    let timed = std::time::Instant::now();
                    let draft = drafter.draft(&mut engines[0].beside()?, files, next, position, VERIFY_MS.len() - 1)?;
                    let ms = 1e3 * timed.elapsed().as_secs_f64();
                    policy.drafted(ms);
                    if position >= straydog_engine::arch::deepseek41::drafter::RING {
                        full = (full.0 + ms / 1e3, full.1 + 1);
                    }
                    let width = match settings.width {
                        Some(width) => width,
                        None => policy.width(&draft.confidences, room - 1),
                    };
                    let width = width.min(room - 1).min(draft.tokens.len());
                    drafts = draft.tokens[..width].to_vec();
                    if let Some(text) = dump.as_mut() {
                        let slots = VERIFY_MS.len() - 1;
                        dump_routes(text, settings.run, index, rounds, position, slots, &drafter.routes(&mut engines[0].beside()?, slots)?);
                    }
                    (drafted, drafting) = (drafted + 1, drafting + ms / 1e3);
                    if settings.show {
                        println!("  round at {position}: draft {:?} confidences {:?}, verifying {}", draft.tokens, draft.confidences, drafts.len());
                    }
                }
            }
            let fed: Vec<u32> = std::iter::once(next).chain(drafts.iter().copied()).collect();
            let timed = std::time::Instant::now();
            let rows = if drafts.is_empty() { forward_split(engines, &fed)? } else { verify_split(engines, &fed)? };
            let spent = timed.elapsed().as_secs_f64();
            policy.passed(fed.len(), 1e3 * spent);
            let sums = &mut by_rows[fed.len() - 1];
            *sums = (sums.0 + spent, sums.1 + 1);
            if fed.len() == 1 && verified {
                after_verify = (after_verify.0 + spent, after_verify.1 + 1);
            }
            verified = fed.len() > 1;
            let row = |at: usize| argmax(&rows[at * vocabulary..(at + 1) * vocabulary]) as u32;
            let accepted = (0..drafts.len()).take_while(|&at| row(at) == drafts[at]).count();
            if accepted + 1 < fed.len() {
                engines.iter_mut().try_for_each(|engine| engine.rewind(accepted + 1))?;
            }
            emitted.extend_from_slice(&drafts[..accepted]);
            emitted.push(row(accepted));
            let timed = std::time::Instant::now();
            inject(engines, drafter, accepted + 1)?;
            injecting += timed.elapsed().as_secs_f64();
            (passes, rounds, accepted_all) = (passes + 1, rounds + 1, accepted_all + accepted);
            widths[drafts.len()] += 1;
        }
        let seconds = started.elapsed().as_secs_f64();
        let after = counted(engines);
        decode.hits += after.hits - before.hits;
        decode.host_hits += after.host_hits - before.host_hits;
        decode.drive_reads += after.drive_reads - before.drive_reads;
        let mut hash_state = 0xcbf2_9ce4_8422_2325u64;
        hash(&mut hash_state, &emitted.iter().map(|&token| f32::from_bits(token)).collect::<Vec<_>>());
        hash(&mut all_hash, &[f32::from_bits(hash_state as u32), f32::from_bits((hash_state >> 32) as u32)]);
        // the first token comes with the prompt; the rest from the rounds
        let made = emitted.len() - 1;
        println!("generate prompt {index}: {} tokens, tokens hash {hash_state:016x}, {made} in {passes} passes, {:.2} ms a token",
                 emitted.len(), 1e3 * seconds / made.max(1) as f64);
        if drafter.is_some() {
            println!("  rounds {rounds}, drafted {drafted} ({:.0}%), widths {widths:?}, {accepted_all} drafts accepted, {:.2} ms a draft",
                     100.0 * drafted as f64 / rounds.max(1) as f64, 1e3 * drafting / drafted.max(1) as f64);
        }
        (all_seconds, all_tokens, all_passes) = (all_seconds + seconds, all_tokens + made, all_passes + passes);
        (all_rounds, all_drafted, all_accepted, all_drafting) = (all_rounds + rounds, all_drafted + drafted, all_accepted + accepted_all,
                                                                 all_drafting + drafting);
        all_widths.iter_mut().zip(widths).for_each(|(sum, width)| *sum += width);
    }
    let per_token = |count: u64| count as f64 / all_tokens.max(1) as f64;
    println!("generate all: tokens hash {all_hash:016x}, {all_tokens} tokens in {all_passes} passes, {:.2} ms a token, {:.2} ms a pass",
             1e3 * all_seconds / all_tokens.max(1) as f64, 1e3 * all_seconds / all_passes.max(1) as f64);
    println!("  a decoded token: {:.1} VRAM hits, {:.1} host-tier fills, {:.1} drive reads", per_token(decode.hits), per_token(decode.host_hits),
             per_token(decode.drive_reads));
    let mean = |(seconds, passes): (f64, usize)| 1e3 * seconds / passes.max(1) as f64;
    let rows: Vec<String> = by_rows.iter().enumerate().filter(|(_, sums)| sums.1 > 0)
        .map(|(at, &sums)| format!("{} rows {} at {:.2} ms", at + 1, sums.1, mean(sums))).collect();
    println!("  passes: {}; a 1-row pass after a verify pass {} at {:.2} ms; injecting {:.2} ms a pass", rows.join(", "), after_verify.1,
             mean(after_verify), 1e3 * injecting / all_passes.max(1) as f64);
    if drafter.is_some() {
        println!("  dspark rounds {all_rounds}, drafted {all_drafted} ({:.0}%), widths {all_widths:?}, {all_accepted} drafts accepted, \
                  {:.2} tokens a round, {:.2} ms a draft ({} after a full history at {:.2} ms)",
                 100.0 * all_drafted as f64 / all_rounds.max(1) as f64, all_tokens as f64 / all_rounds.max(1) as f64,
                 1e3 * all_drafting / all_drafted.max(1) as f64, full.1, mean(full));
    }
    Ok(())
}

/// DSpark's drafts of one size: how often their prefix matching the reference reaches each slot.
fn reach(rounds: usize, per_slot: &[usize]) -> String {
    per_slot.iter().map(|&reached| format!("{:.3}", reached as f64 / rounds.max(1) as f64)).collect::<Vec<_>>().join(", ")
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
    let (generate_count, runs, forced_width) = (option("--generate", 0), option("--runs", 1), option("--width", usize::MAX));
    let draft_bench = option("--draft-bench", 0);
    let half_gemm = match args.iter().position(|arg| arg == "--gemm") {
        Some(at) => {
            let mode = args.get(at + 1).ok_or("--gemm needs fp32 or fp16")?.clone();
            args.drain(at..at + 2);
            match mode.as_str() { "fp32" => false, "fp16" => true, _ => return Err("--gemm needs fp32 or fp16".into()) }
        }
        None => false,
    };
    eprintln!("DS prompt-pass products: {}; passes below 32 rows F32, up to four rows GEMV", if half_gemm { "FP16 inputs, F32 sums/output" } else { "FP32" });
    let dump_path = args.iter().position(|arg| arg == "--dump-routes").map(|at| {
        let value = args[at + 1].clone();
        args.drain(at..at + 2);
        value
    });
    let mut dump = dump_path.as_ref().map(|_| "# run\tprompt\tround\tposition\tslots\tstage\troutes\n".to_string());
    let widths: Vec<usize> = args.iter().position(|arg| arg == "--verify").map_or(vec![0], |at| {
        let value = args[at + 1].split(',').map(|width| width.parse().unwrap()).collect();
        args.drain(at..at + 2);
        value
    });
    let accept_all = args.iter().position(|arg| arg == "--accept-all").map(|at| args.remove(at)).is_some();
    let dspark_dir = args.iter().position(|arg| arg == "--dspark").map(|at| {
        let value = args[at + 1].clone();
        args.drain(at..at + 2);
        value
    });
    let draft_slots: Vec<usize> = args.iter().position(|arg| arg == "--draft-slots").map_or(vec![3, 4], |at| {
        let value = args[at + 1].split(',').map(|slots| slots.parse().unwrap()).collect();
        args.drain(at..at + 2);
        value
    });
    let draft_slots: Vec<usize> = draft_slots.into_iter().filter(|&slots| slots > 0 && dspark_dir.is_some()).collect();
    let show_drafts = args.iter().position(|arg| arg == "--show-drafts").map(|at| args.remove(at)).is_some();
    let show = args.iter().position(|arg| arg == "--show").map(|at| {
        let value: f64 = args[at + 1].parse().unwrap();
        args.drain(at..at + 2);
        value
    });
    let [dir, tokens_files] = &args[..] else {
        return Err("usage: ds41_quality SHARD_DIR TOKENS[,TOKENS...] [--chunk N] [--batch N] [--slots N] [--split L] [--tp N] [--host-gb N] \
                    [--row-cache MIB] [--steps N] [--context N] [--verify N[,N...]] [--accept-all] \
                    [--dspark DIR [--draft-slots N[,N...]] [--show-drafts]] [--generate N [--runs R] [--width W]] [--dump-routes FILE] [--draft-bench K]".into());
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
    let layers = hp.layers.len();
    if split != 0 && tp > 1 {
        return Err("--split and --tp do not combine".into());
    }
    let parts: Vec<_> = if split == 0 { std::iter::once(0..layers).collect() } else { vec![0..split, split..layers] };
    let devices = (parts.len() * tp) as u64;
    let mut engines = Vec::new();
    for (index, part) in parts.into_iter().enumerate() {
        let settings = Settings { context, rows: chunk.max(batch), batch, expert_slots: slots, host_expert_bytes: ((host_gb as u64) << 30) / devices,
                                  scratch_bytes: 512 << 20, half_gemm, layers: part };
        let devices: Vec<usize> = (index * tp..(index + 1) * tp).collect();
        engines.push(Engine::new(hp.clone(), &files, &devices, 31 << 30, settings)?);
    }
    let dspark_files = dspark_dir.as_ref().map(|dir| files::Files::open(Path::new(dir))).transpose()?;
    let mut drafter = match &dspark_files {
        Some(draft) => {
            if engines.len() != 1 || widths.iter().any(|&verify| verify != 0) {
                return Err("--dspark drafts on one engine (no --split) with --verify 0".into());
            }
            Dspark::from_headers(&draft.header, &files.header)?;
            engines[0].enable_tap()?;
            let loading = std::time::Instant::now();
            let drafter = Drafter::load(&mut engines[0].beside()?, draft, &files, vocabulary)?;
            println!("dspark: loaded in {:.1} s", loading.elapsed().as_secs_f64());
            Some(drafter)
        }
        None => None,
    };
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    for tokens_file in tokens_files.split(',') {
        let raw = std::fs::read(tokens_file).map_err(|exc| format!("{tokens_file}: {exc}"))?;
        let ints: Vec<i32> = raw.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
        prompts.push(ints[3..3 + ints[0] as usize].iter().map(|&id| id as u32).collect());
    }
    if draft_bench > 0 {
        // the first prompt prefilled, a second's pause (a gap in a trace), then the drafts alone
        let mut logits = Vec::new();
        for piece in prompts[0].chunks(chunk) {
            logits = forward_split(&mut engines, piece)?;
            inject(&mut engines, &mut drafter, piece.len())?;
        }
        let (next, position) = (argmax(&logits) as u32, engines[0].position());
        let drafter = drafter.as_mut().ok_or("--draft-bench drafts with --dspark")?;
        if position < straydog_engine::arch::deepseek41::drafter::RING {
            return Err("--draft-bench drafts after a full history: a prompt of 128 tokens or more".into());
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
        let mut times = Vec::with_capacity(draft_bench);
        for _ in 0..draft_bench {
            let timed = std::time::Instant::now();
            drafter.draft(&mut engines[0].beside()?, &files, next, position, VERIFY_MS.len() - 1)?;
            times.push(1e3 * timed.elapsed().as_secs_f64());
        }
        let first = times[..2.min(times.len())].to_vec();
        times.sort_by(f64::total_cmp);
        println!("draft bench: {draft_bench} drafts of {} at position {position}: median {:.3} ms, min {:.3} ms, the first two {first:.3?} ms",
                 VERIFY_MS.len() - 1, times[times.len() / 2], times[0]);
        return Ok(());
    }
    if generate_count > 0 {
        let mut settings = Generate { count: generate_count, chunk, vocabulary, show: show_drafts, drafts: !draft_slots.is_empty(),
                                      width: (forced_width < VERIFY_MS.len()).then_some(forced_width), run: 0 };
        for run in 0..runs {
            settings.run = run;
            generate(&mut engines, &files, &mut drafter, &prompts, &settings, &mut dump)?;
        }
        if let (Some(path), Some(text)) = (&dump_path, &dump) {
            std::fs::write(path, text).map_err(|exc| format!("{path}: {exc}"))?;
        }
        return Ok(());
    }
    for (run, &verify) in widths.iter().enumerate() {
        let (mut all, mut decode_tokens, mut decode_passes, mut decode_seconds) = (Vec::new(), 0, 0, 0.0);
        let mut run_hash = 0xcbf2_9ce4_8422_2325u64;
        // per block size, over the prompts: rounds, tokens matched as a prefix, rounds whose prefix
        // reaches each slot, seconds drafting
        let mut drafted_all: Vec<(usize, usize, Vec<usize>, f64)> = draft_slots.iter().map(|&slots| (0, 0, vec![0; slots], 0.0)).collect();
        // DS41_PROFILE: the decode passes' phases, and the store's counters over them
        let (mut phases, mut stores) = (std::collections::BTreeMap::<&str, f64>::new(), straydog_engine::experts::Counts::default());
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
            if let Some(drafter) = &mut drafter {
                drafter.forget_from(0);
            }
            let mut logits = Vec::new();
            for piece in prompt.chunks(chunk) {
                logits = forward_split(&mut engines, piece)?;
                inject(&mut engines, &mut drafter, piece.len())?;
            }
            // per block size: rounds, tokens matched as a prefix, rounds whose prefix reaches each slot,
            // seconds drafting
            let mut drafted: Vec<(usize, usize, Vec<usize>, f64)> = draft_slots.iter().map(|&slots| (0, 0, vec![0; slots], 0.0)).collect();
            let mut prompt_hash = 0xcbf2_9ce4_8422_2325u64;
            hash(&mut prompt_hash, &logits);
            let mut saved: Vec<u8> = logits.iter().flat_map(|value| value.to_le_bytes()).collect();
            let mut steps = vec![compare(&logits, &row(kept - 1), generated.first().map(|&token| token as usize))];
            engines.iter_mut().for_each(|engine| drop(engine.take_profile()));
            let counted = |engines: &[Engine<'_, files::Files>]| {
                engines.iter().map(Engine::cache_counts).fold(straydog_engine::experts::Counts::default(), |a, b| a + b)
            };
            let before = counted(&engines);
            let (decode_steps, decoding) = (decode_rows.min(generated.len()).min(max_steps), std::time::Instant::now());
            let (mut step, mut seed, mut passes) = (0, 0x9e37_79b9_7f4a_7c15u64 ^ index as u64, 0);
            while step < decode_steps {
                let width = verify.clamp(1, decode_steps - step);
                let rows = if verify == 0 {
                    // the drafts after this step's token (the anchor, the target's choice), before it is fed
                    if let Some(drafter) = &mut drafter {
                        let (anchor, position, truth) = (generated[step], engines[0].position(), &generated[step + 1..]);
                        for (&slots, (rounds, matched, per_slot, spent)) in draft_slots.iter().zip(&mut drafted) {
                            if truth.len() < slots {
                                continue;
                            }
                            let drafting = std::time::Instant::now();
                            let draft = drafter.draft(&mut engines[0].beside()?, &files, anchor, position, slots)?;
                            let proposal = &draft.tokens;
                            *spent += drafting.elapsed().as_secs_f64();
                            let prefix = proposal.iter().zip(truth).take_while(|(a, b)| a == b).count();
                            if let Some(text) = dump.as_mut() {
                                dump_routes(text, run, index, step, position, slots, &drafter.routes(&mut engines[0].beside()?, slots)?);
                            }
                            *rounds += 1;
                            *matched += prefix;
                            per_slot[..prefix].iter_mut().for_each(|reached| *reached += 1);
                            if show_drafts {
                                let confidences: Vec<String> = draft.confidences.iter().map(|c| format!("{c:.4}")).collect();
                                println!("  draft prompt {index} position {position} anchor {anchor} B={slots}: {proposal:?}, {prefix} matched, \
                                          confidences [{}]", confidences.join(", "));
                            }
                        }
                    }
                    let rows = forward_split(&mut engines, &generated[step..step + 1])?;
                    inject(&mut engines, &mut drafter, 1)?;
                    rows
                } else {
                    verify_split(&mut engines, &generated[step..step + width])?
                };
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let accepted = if accept_all { width } else { 1 + (seed % width as u64) as usize };
                for (at, logits) in rows.chunks_exact(vocabulary).take(accepted).enumerate() {
                    hash(&mut prompt_hash, logits);
                    saved.extend(logits.iter().flat_map(|value| value.to_le_bytes()));
                    steps.push(compare(logits, &row(kept + step + at), generated.get(step + at + 1).map(|&token| token as usize)));
                }
                if accepted < width {
                    engines.iter_mut().try_for_each(|engine| engine.rewind(accepted))?;
                }
                step += accepted;
                passes += 1;
            }
            let seconds = decoding.elapsed().as_secs_f64();
            for engine in &mut engines {
                for (phase, spent) in engine.take_profile() {
                    *phases.entry(phase).or_default() += spent;
                }
            }
            let after = counted(&engines);
            stores.hits += after.hits - before.hits;
            stores.host_hits += after.host_hits - before.host_hits;
            stores.drive_reads += after.drive_reads - before.drive_reads;
            stores.drive_bytes += after.drive_bytes - before.drive_bytes;
            stores.writebacks += after.writebacks - before.writebacks;
            stores.wait_seconds += after.wait_seconds - before.wait_seconds;
            stores.slot_wait_seconds += after.slot_wait_seconds - before.slot_wait_seconds;
            stores.store_seconds += after.store_seconds - before.store_seconds;
            (decode_tokens, decode_passes, decode_seconds) =
                (decode_tokens + decode_steps, decode_passes + passes, decode_seconds + seconds);
            hash(&mut run_hash, &[f32::from_bits(prompt_hash as u32), f32::from_bits((prompt_hash >> 32) as u32)]);
            println!("{} | logits hash {prompt_hash:016x} | {decode_steps} tokens in {passes} passes, {seconds:.2} s",
                     summary(&format!("prompt {index}"), &steps));
            for ((&slots, (rounds, matched, per_slot, spent)), total) in draft_slots.iter().zip(&drafted).zip(&mut drafted_all) {
                println!("  dspark prompt {index} B={slots}: {rounds} rounds, mean accepted {:.3}, prefix reaches slot [{}]",
                         *matched as f64 / (*rounds).max(1) as f64, reach(*rounds, per_slot));
                total.0 += rounds;
                total.1 += matched;
                total.2.iter_mut().zip(per_slot).for_each(|(sum, reached)| *sum += reached);
                total.3 += spent;
            }
            if let Some(dir) = std::env::var_os("DS41_QUALITY_SAVE") {
                let name = if widths.len() > 1 { format!("prompt-{index}-run{run}.logits") } else { format!("prompt-{index}.logits") };
                std::fs::write(Path::new(&dir).join(name), &saved).map_err(|exc| exc.to_string())?;
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
        let label = if widths.len() > 1 { format!("all, verify {verify}") } else { "all".to_string() };
        println!("{} | run hash {run_hash:016x} | {decode_tokens} tokens in {decode_passes} passes, {decode_seconds:.2} s",
                 summary(&label, &all));
        for (&slots, (rounds, matched, per_slot, spent)) in draft_slots.iter().zip(&drafted_all) {
            println!("  dspark all B={slots}: {rounds} rounds, mean accepted {:.3}, prefix reaches slot [{}], {:.2} ms a draft",
                     *matched as f64 / (*rounds).max(1) as f64, reach(*rounds, per_slot), 1e3 * spent / (*rounds).max(1) as f64);
        }
        let per_pass = |count: u64| count as f64 / decode_passes.max(1) as f64;
        println!("  a decode pass: {:.2} ms, {:.1} VRAM hits, {:.1} host-tier fills, {:.1} drive reads ({:.1} MiB), {:.1} write-backs; \
                  waits {:.2} ms for reads, {:.2} ms for slots, store {:.2} ms",
                 1e3 * decode_seconds / decode_passes.max(1) as f64, per_pass(stores.hits), per_pass(stores.host_hits),
                 per_pass(stores.drive_reads), per_pass(stores.drive_bytes) / f64::from(1 << 20), per_pass(stores.writebacks),
                 1e3 * stores.wait_seconds / decode_passes.max(1) as f64, 1e3 * stores.slot_wait_seconds / decode_passes.max(1) as f64,
                 1e3 * stores.store_seconds / decode_passes.max(1) as f64);
        if !phases.is_empty() {
            let line: Vec<String> = phases.iter().map(|(phase, spent)| format!("{phase} {:.2}", 1e3 * spent / decode_passes.max(1) as f64)).collect();
            println!("  phases, ms a pass: {}", line.join(", "));
        }
    }
    if let (Some(path), Some(text)) = (&dump_path, &dump) {
        std::fs::write(path, text).map_err(|exc| format!("{path}: {exc}"))?;
    }
    if std::env::var_os("DS41_CHECK_SLOTS").is_some() {
        println!("slot checks: {} mismatches", engines.iter().map(Engine::slot_mismatches).sum::<u64>());
    }
    if std::env::var("DS41_FP16_CHECKS").is_ok_and(|value| value == "1") {
        for (index, gpu) in engines.iter().flat_map(|engine| engine.gpus()).enumerate() {
            let counts = gpu.gemm_checks()?;
            println!("FP16 cast checks GPU{index}: {counts:?}, max_abs={}", f32::from_bits(counts[7] as u32));
            if counts[0] != 0 || counts[1] != 0 || counts[2] != 0 || counts[4] != 0 {
                return Err(format!("nonfinite/overflow FP16 operand on GPU{index}: {counts:?}"));
            }
        }
    }
    let counts = engines.iter().map(|engine| engine.cache_counts()).fold(straydog_engine::experts::Counts::default(), |a, b| a + b);
    println!("experts: {} VRAM hits, {} host-tier fills, {} drive reads, {} write-backs", counts.hits, counts.host_hits,
             counts.drive_reads, counts.writebacks);
    for engine in &mut engines {
        engine.settle()?;
    }
    Ok(())
}
