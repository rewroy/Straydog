//! Time fresh prefills of real text in-process (no sandbox), for a profiler to wrap.
//!
//! cargo run --release --example prefill --features hip -- MODEL QUANT DEVICES VRAM_GIB CONTEXT TOKENS... [--drafter DIR] [--batch N]
//!     [--dump DIR] [--generate N] [--corpus FILE] [--gemm-min-batch N] [--gemm f32] [--split layer] [--row-cache MIB]
//!     [--pause SECONDS] [--sampled] [--drafts N] [--chat] [--force-rows W,W,... --force-tokens FILE [--force-logits] [--force-reject N]] [--copy-drafts N]
//!
//! The text is this repository's design document and engine sources (or FILE), repeated; each prompt
//! starts with its own line, so no prompt reuses another's state. TOKENS 0 only loads, "+N"
//! appends N tokens to the previous prompt, and "TOKENS:G" generates G tokens for that run instead
//! of --generate's. --dump writes each prompt's first logits and its greedy generation, the material
//! for checking a numerical change against a saved run. --pause sleeps before each run, so a
//! profiler's trace of a run is apart from the one before; --drafts N drafts at most N tokens a pass
//! (0: plain decode, with the drafter loaded all the same); --sampled generates with the model
//! card's thinking sampling (temperature 1.0, top_p 0.95, top_k 20) instead of greedily; --chat puts each
//! fresh prompt in a user turn that asks for an explanation (a reply that runs to its length, where raw text
//! ends in an end token after a few). --force-rows measures passes by their rows: FILE holds the greedy tokens of the run's
//! reply (--dump's prompt-N.tokens), the passes verify them at the widths given in turn, with no drafter, and the milliseconds
//! a pass by width and the digest of the kept rows (with --force-logits, of their logits too: every width's must be the same) are printed.

use std::path::PathBuf;
use std::time::Instant;

use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let drafter = args.iter().position(|arg| arg == "--drafter").map(|at| {
        let path = PathBuf::from(&args[at + 1]);
        args.drain(at..at + 2);
        path
    });
    let batch = args.iter().position(|arg| arg == "--batch").map(|at| {
        let batch: usize = args[at + 1].parse().unwrap();
        args.drain(at..at + 2);
        batch
    });
    let force_logits = args.iter().position(|arg| arg == "--force-logits").map(|at| args.remove(at)).is_some();
    let chat = args.iter().position(|arg| arg == "--chat").map(|at| args.remove(at)).is_some();
    let sampled = args.iter().position(|arg| arg == "--sampled").map(|at| args.remove(at)).is_some();
    let mut option = |name: &str| args.iter().position(|arg| arg == name).map(|at| {
        let value = args[at + 1].clone();
        args.drain(at..at + 2);
        value
    });
    let force_rows: Option<Vec<usize>> = option("--force-rows").map(|widths| widths.split(',').map(|width| width.parse().unwrap()).collect());
    let force_tokens = option("--force-tokens");
    let copy_drafts: Option<usize> = option("--copy-drafts").map(|count| count.parse().unwrap());
    // --force-reject N: every Nth pass a draft is wrong (the digests must not change).
    let force_reject: usize = option("--force-reject").map_or(0, |count| count.parse().unwrap());
    // --dump DIR: each prompt's logits (f32, little-endian) and generated tokens land in DIR.
    let dump = option("--dump").map(PathBuf::from);
    let generate: usize = option("--generate").map_or(1, |count| count.parse().unwrap());
    let corpus_file = option("--corpus").map(PathBuf::from);
    // --ask TEXT: the chat turn's request after the prompt (default: explain the code above).
    let ask = option("--ask").unwrap_or_else(|| "\n\nExplain in detail, section by section, what the code above does.".to_string());
    let gemm_min_batch: u64 = option("--gemm-min-batch").map_or(2, |count| count.parse().unwrap());
    let pause: f64 = option("--pause").map_or(0.0, |seconds| seconds.parse().unwrap());
    let drafts: Option<usize> = option("--drafts").map(|count| count.parse().unwrap());
    // --gemm f32: batched projections in F32 throughout (the numerical reference for FP16 GEMMs).
    let half_gemm = option("--gemm").as_deref() != Some("f32");
    // --row-cache MIB: the PLE row cache (0: rows through the page cache; none: as the server sizes it).
    let row_cache = option("--row-cache").map(|mib| mib.parse::<u64>().unwrap() << 20);
    let split = match option("--split").as_deref() {
        Some("layer") => SplitMode::Layer,
        _ => SplitMode::Experts,
    };
    let [model, quant, devices, vram, context, lengths @ ..] = &args[..] else {
        return Err("usage: prefill MODEL QUANT DEVICES VRAM_GIB CONTEXT TOKENS... [--drafter DIR]".into());
    };
    let devices: Vec<usize> = devices.split(',').map(|device| device.parse().unwrap()).collect();
    let budget = (vram.parse::<f64>().unwrap() * (1u64 << 30) as f64) as u64;
    let started = Instant::now();
    let mut group = Group::new(&devices, &vec![budget; devices.len()], split)?;
    group.configure_gemm(gemm_min_batch, half_gemm)?;
    let root = PathBuf::from(model);
    let weights = Weights::open(&root, straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, quant)?, drafter.as_deref(), straydog_engine::host::host_memory_bytes(), &|_, _, _| {})?;
    let open_seconds = started.elapsed().as_secs_f64();
    let settings = Settings { context: context.parse().unwrap(), state_budget_bytes: 14 << 30, prefill_batch_size: batch.unwrap_or(4096),
                              speculative: if drafter.is_some() { drafts.unwrap_or(3).clamp(1, straydog_engine::model::MAX_DRAFTS) } else { 0 }, cpu_threads: 0,
                              row_cache_bytes: row_cache };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {})?;
    // (the worker locks the banks behind "ready"; this run is served from them locked, as the worker's is once it has)
    let locking = std::time::Instant::now();
    model.lock_banks_now()?;
    eprintln!("banks locked in {:.2} s: {} layers", locking.elapsed().as_secs_f64(), model.locked_layers());
    if drafts == Some(0) {
        model.speculative = 0;
    }
    // --copy-drafts N: the most drafts a long match's copy verifies.
    if let Some(copies) = copy_drafts {
        model.copy_drafts = copies.min(model.pass_rows() - 1);
    }
    let steps: Vec<String> =
        model.load_seconds.iter().map(|(step, seconds)| format!("{step} {seconds:.2} s")).collect();
    eprintln!("load {:.2} s: devices and open {open_seconds:.2} s, {}", started.elapsed().as_secs_f64(),
              steps.join(", "));
    let text = Text::new(&std::fs::read(root.join("tokenizer.json")).map_err(|exc| exc.to_string())?).map_err(|exc| exc.0)?;
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let corpus = match &corpus_file {
        Some(file) => std::fs::read_to_string(file).map_err(|exc| exc.to_string())?,
        None => {
            let mut corpus = std::fs::read_to_string(source.join("docs/DESIGN.md")).map_err(|exc| exc.to_string())?;
            let mut files: Vec<PathBuf> = std::fs::read_dir(source.join("crates/straydog-engine/src"))
                .map_err(|exc| exc.to_string())?.flatten().map(|entry| entry.path()).filter(|path| path.is_file()).collect();
            files.sort();
            for file in files {
                corpus += &std::fs::read_to_string(file).map_err(|exc| exc.to_string())?;
            }
            corpus
        }
    };
    let corpus = text.encode(&corpus).map_err(|exc| exc.0)?;
    eprintln!("resident {} host_kv {} batch {}", model.resident, model.host_kv, model.prefill_batch_size);
    for &device in &devices {
        if let Ok((_, free, total)) = straydog_gpu::device_info(device) {
            eprintln!("GPU {device}: {:.2} GiB free of {:.2} GiB after loading (budget {:.2} GiB)", free as f64 / (1u64 << 30) as f64,
                      total as f64 / (1u64 << 30) as f64, budget as f64 / (1u64 << 30) as f64);
        }
    }
    // "+N" appends N tokens to the previous prompt (the conversation cache resumes it).
    let mut previous: Vec<u32> = Vec::new();
    let mut row_counts = (0, 0, 0);
    for (run, length) in lengths.iter().enumerate() {
        let (length, generate) = match length.split_once(':') {
            Some((length, count)) => (length, count.parse::<usize>().unwrap()),
            None => (length.as_str(), generate),
        };
        let (append, length) = match length.strip_prefix('+') {
            Some(count) => (true, count.parse::<usize>().unwrap()),
            None => (false, length.parse::<usize>().unwrap()),
        };
        if length == 0 {
            continue;
        }
        let mut prompt = if append { previous.clone() } else { text.encode(&format!("Prompt {run}.\n")).map_err(|exc| exc.0)? };
        let target = if append { prompt.len() + length } else { length };
        let mut offset = run * 997;
        while prompt.len() < target {
            prompt.push(corpus[offset % corpus.len()]);
            offset += 1;
        }
        if chat && !append {
            // (the chat template's user turn and the assistant's header, thinking off: as model_gpu's turns)
            let ask = text.encode(&ask).map_err(|exc| exc.0)?;
            let mut turn = vec![248045, 846, 198];
            turn.extend_from_slice(&prompt);
            turn.extend_from_slice(&ask);
            turn.extend([248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]);
            prompt = turn;
        }
        let length = prompt.len();
        previous = prompt.clone();
        if pause > 0.0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(pause));
        }
        let started = Instant::now();
        model.logits_digests = Some(Vec::new());
        model.logits_capture = Some(Vec::new());
        let mut tokens = Vec::new();
        let sampling = if sampled { Sampling::recommended(true) } else { Sampling::GREEDY };
        if let (Some(widths), Some(file)) = (&force_rows, &force_tokens) {
            let replies = std::fs::read_to_string(file).map_err(|exc| exc.to_string())?;
            model.forced_rows = Some(straydog_engine::model::ForcedRows { tokens: serde_json::from_str(&replies).unwrap(), widths: widths.clone(),
                                                                          read_logits: force_logits, reject: force_reject, ..Default::default() });
        }
        model.generate(&prompt, generate, &sampling, run as u64 + 1, &|| false, &mut |_, _| {}, true, false,
                       &mut |token| tokens.push(token))?;
        let digest = model.logits_digests.take().unwrap_or_default().first().cloned().unwrap_or_default();
        let logits = model.logits_capture.take().unwrap_or_default().into_iter().next().unwrap_or_default();
        if let Some(dump) = &dump {
            std::fs::create_dir_all(dump).map_err(|exc| exc.to_string())?;
            let bytes: Vec<u8> = logits.iter().flat_map(|value| value.to_le_bytes()).collect();
            std::fs::write(dump.join(format!("prompt-{run}.logits")), bytes).map_err(|exc| exc.to_string())?;
            std::fs::write(dump.join(format!("prompt-{run}.tokens")), serde_json::to_string(&tokens).unwrap())
                .map_err(|exc| exc.to_string())?;
        }
        let seconds = model.last_metrics["prefill_seconds"].as_f64().unwrap_or(0.0);
        let reused = model.last_metrics["cached_tokens"].as_u64().unwrap_or(0);
        if generate > 1 {
            let passes = model.last_metrics.get("speculative_passes").and_then(|value| value.as_u64()).unwrap_or(0);
            let seconds = model.last_metrics["decode_seconds"].as_f64().unwrap_or(0.0);
            println!("decode {} tokens: {:.1} tok/s, {passes} passes of {:.2} ms", tokens.len(),
                     model.last_metrics["decode_tokens_per_second"].as_f64().unwrap_or(0.0),
                     1e3 * seconds / passes.max(1) as f64);
            for key in ["drafted_tokens", "accepted_tokens", "decode_pass_rows", "decode_profile", "ple_rows"] {
                if let Some(value) = model.last_metrics.get(key) {
                    println!("  {key}: {value}");
                }
            }
        }
        if let Some(forced) = model.forced_rows.take() {
            // (each width's first passes capture its graphs)
            let mut widths = forced.widths.clone();
            widths.sort_unstable();
            widths.dedup();
            for width in widths {
                let mut passes: Vec<f64> = forced.passes.iter().filter(|(rows, _)| *rows == width).map(|(_, ms)| *ms).skip(4).collect();
                passes.sort_by(f64::total_cmp);
                let mean = passes.iter().sum::<f64>() / passes.len().max(1) as f64;
                let trimmed = &passes[passes.len() / 10..passes.len() - passes.len() / 10];
                println!("rows {width}: {} passes, median {:.3} ms, trimmed mean {:.3}, mean {mean:.3}", passes.len(), passes.get(passes.len() / 2).copied().unwrap_or(0.0),
                         trimmed.iter().sum::<f64>() / trimmed.len().max(1) as f64);
            }
            println!("forced digest {:x}, logits {:x}, rejected {}, tokens {}", sha2::Digest::finalize(forced.digest), sha2::Digest::finalize(forced.logits_digest),
                     forced.rejected, tokens.len());
        }
        for &device in &devices {
            if let Ok((_, free, _)) = straydog_gpu::device_info(device) {
                println!("GPU {device}: {:.3} GiB free after the run", free as f64 / (1u64 << 30) as f64);
            }
        }
        println!("prefill {length} tokens ({reused} reused): {seconds:.3} s ({:.0} tok/s), request {:.3} s, logits {}", (length - reused as usize) as f64 / seconds,
                 started.elapsed().as_secs_f64(), &digest[..16.min(digest.len())]);
        if let Some(now) = model.weights.row_counts("per_layer_token_embd.weight") {
            let (asked, read, blocks) = (now.0 - row_counts.0, now.1 - row_counts.1, now.2 - row_counts.2);
            println!("PLE rows: {asked} used, {read} read ({:.1}% from the cache), {blocks} blocks",
                     100.0 * (1.0 - read as f64 / asked.max(1) as f64));
            row_counts = now;
        }
    }
    Ok(())
}
