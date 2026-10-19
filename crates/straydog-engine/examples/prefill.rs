//! Time fresh prefills of real text in-process (no sandbox), for a profiler to wrap.
//!
//! cargo run --release --example prefill --features hip -- MODEL QUANT DEVICES VRAM_GIB CONTEXT TOKENS... [--drafter DIR]
//!
//! The text is this repository's design document and engine sources, repeated; each prompt
//! starts with its own line, so no prompt reuses another's state. TOKENS 0 only loads.

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
    let [model, quant, devices, vram, context, lengths @ ..] = &args[..] else {
        return Err("usage: prefill MODEL QUANT DEVICES VRAM_GIB CONTEXT TOKENS... [--drafter DIR]".into());
    };
    let devices: Vec<usize> = devices.split(',').map(|device| device.parse().unwrap()).collect();
    let budget = (vram.parse::<f64>().unwrap() * (1u64 << 30) as f64) as u64;
    let group = Group::new(&devices, budget, SplitMode::Experts)?;
    group.configure_gemm(2, true)?;
    let root = PathBuf::from(model);
    let weights = Weights::open(&root, quant, drafter.as_deref(), straydog_engine::host::host_memory_bytes(), &|_, _, _| {})?;
    let settings = Settings { context: context.parse().unwrap(), state_budget_bytes: 13 << 30, prefill_batch_size: 4096,
                              speculative: if drafter.is_some() { 3 } else { 0 }, cpu_threads: 0 };
    let mut model = Model::new(weights, group, &settings)?;
    let text = Text::new(&std::fs::read(root.join("tokenizer.json")).map_err(|exc| exc.to_string())?).map_err(|exc| exc.0)?;
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut corpus = std::fs::read_to_string(source.join("docs/DESIGN.md")).map_err(|exc| exc.to_string())?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(source.join("crates/straydog-engine/src")).map_err(|exc| exc.to_string())?
        .flatten().map(|entry| entry.path()).collect();
    files.sort();
    for file in files {
        corpus += &std::fs::read_to_string(file).map_err(|exc| exc.to_string())?;
    }
    let corpus = text.encode(&corpus).map_err(|exc| exc.0)?;
    eprintln!("resident {} host_kv {} batch {}", model.resident, model.host_kv, model.prefill_batch_size);
    for (run, length) in lengths.iter().map(|length| length.parse::<usize>().unwrap()).enumerate() {
        if length == 0 {
            continue;
        }
        let mut prompt = text.encode(&format!("Prompt {run}.\n")).map_err(|exc| exc.0)?;
        while prompt.len() < length {
            prompt.extend(corpus.iter().take(length - prompt.len()));
        }
        let started = Instant::now();
        model.generate(&prompt, 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |_| {})?;
        let seconds = model.last_metrics["prefill_seconds"].as_f64().unwrap_or(0.0);
        println!("prefill {length} tokens: {seconds:.3} s ({:.0} tok/s), request {:.3} s", length as f64 / seconds,
                 started.elapsed().as_secs_f64());
    }
    Ok(())
}
