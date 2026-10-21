//! Time fresh prefills of real text in-process (no sandbox), for a profiler to wrap.
//!
//! cargo run --release --example prefill --features hip -- MODEL QUANT DEVICES VRAM_GIB CONTEXT TOKENS... [--drafter DIR] [--batch N]
//!     [--dump DIR] [--generate N] [--corpus FILE] [--gemm-min-batch N] [--gemm f32] [--split layer] [--row-cache MIB]
//!
//! The text is this repository's design document and engine sources (or FILE), repeated; each prompt
//! starts with its own line, so no prompt reuses another's state. TOKENS 0 only loads, and "+N"
//! appends N tokens to the previous prompt. --dump writes each prompt's first logits and its
//! greedy generation, the material for checking a numerical change against a saved run.

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
    let mut option = |name: &str| args.iter().position(|arg| arg == name).map(|at| {
        let value = args[at + 1].clone();
        args.drain(at..at + 2);
        value
    });
    // --dump DIR: each prompt's logits (f32, little-endian) and generated tokens land in DIR.
    let dump = option("--dump").map(PathBuf::from);
    let generate: usize = option("--generate").map_or(1, |count| count.parse().unwrap());
    let corpus_file = option("--corpus").map(PathBuf::from);
    let gemm_min_batch: u64 = option("--gemm-min-batch").map_or(2, |count| count.parse().unwrap());
    // --gemm f32: batched projections in F32 throughout (the numerical reference for FP16 GEMMs).
    let half_gemm = option("--gemm").as_deref() != Some("f32");
    // --row-cache MIB: the PLE row cache (0: rows through the page cache).
    let row_cache = option("--row-cache").map_or(straydog_engine::rows::DEFAULT_CACHE_BYTES, |mib| mib.parse::<u64>().unwrap() << 20);
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
    let weights = Weights::open(&root, quant, drafter.as_deref(), straydog_engine::host::host_memory_bytes(), &|_, _, _| {})?;
    let open_seconds = started.elapsed().as_secs_f64();
    let settings = Settings { context: context.parse().unwrap(), state_budget_bytes: 13 << 30, prefill_batch_size: batch.unwrap_or(4096),
                              speculative: if drafter.is_some() { 3 } else { 0 }, cpu_threads: 0,
                              row_cache_bytes: row_cache };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {})?;
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
    // "+N" appends N tokens to the previous prompt (the conversation cache resumes it).
    let mut previous: Vec<u32> = Vec::new();
    let mut row_counts = (0, 0, 0);
    for (run, length) in lengths.iter().enumerate() {
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
        let length = prompt.len();
        previous = prompt.clone();
        let started = Instant::now();
        model.logits_digests = Some(Vec::new());
        model.logits_capture = Some(Vec::new());
        let mut tokens = Vec::new();
        model.generate(&prompt, generate, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false,
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
