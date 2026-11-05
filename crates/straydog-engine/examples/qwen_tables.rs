//! What an engram table's precision changes in Qwen3.8's output: the bundle in MODEL is loaded with each table in turn
//! (STRAYDOG_ENGRAM), each prompt continued greedily by N tokens with every step's logits kept, and the second table's
//! logits compared with the first's while the two continuations agree (the same context): KL(first || second) mean, p99
//! and maximum, and where the continuations part.
//!
//! cargo run --release --example qwen_tables --features hip -- MODEL QUANT DEVICES VRAM_GIB FIRST SECOND [TOKENS]

use std::path::PathBuf;

use straydog_engine::bundle;
use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

const PROMPTS: [&str; 6] = [
    "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n",
    "日本の四季について、それぞれの季節の特徴を説明します。春は",
    "The history of the printing press begins in the fifteenth century, when",
    "Question: A train travels 120 km in 1.5 hours. What is its average speed?\nAnswer: Let's think step by step.",
    "| Country | Capital | Population |\n| --- | --- | --- |\n| France | Paris |",
    "fn parse_header(bytes: &[u8]) -> Result<Header, Error> {\n    let magic = ",
];

/// Each prompt's greedy tokens and every step's logits with the table of `precision`.
fn run(model_dir: &PathBuf, id: bundle::Id, devices: &[usize], budget: u64, precision: &str, steps: usize)
       -> Result<Vec<(Vec<u32>, Vec<Vec<f32>>)>, String> {
    std::env::set_var("STRAYDOG_ENGRAM", precision);
    let mut group = Group::new(devices, &vec![budget; devices.len()], SplitMode::Experts)?;
    group.configure_gemm(2, true)?;
    let weights = Weights::open(model_dir, id, None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {})?;
    let table = weights.table.as_ref().map_or_else(|| "its own".to_string(), |table| table.precision.clone());
    let text = Text::new(weights.resources.get("tokenizer.json").ok_or("no tokenizer")?).map_err(|exc| exc.0)?;
    let settings = Settings { context: 32_768, state_budget_bytes: 8 << 30, prefill_batch_size: 2048, speculative: 0, cpu_threads: 0,
                              row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {})?;
    model.lock_banks_now()?;
    eprintln!("{precision}: loaded ({table} table)");
    let mut out = Vec::new();
    for prompt in PROMPTS {
        let tokens = text.encode(prompt).map_err(|exc| exc.0)?;
        model.logits_capture = Some(Vec::new());
        let mut generated = Vec::new();
        model.generate(&tokens, steps, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, false, false, &mut |token| generated.push(token))?;
        out.push((generated, model.logits_capture.take().unwrap_or_default()));
    }
    Ok(out)
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&value| (value as f64 - max).exp()).sum();
    logits.iter().map(|&value| value as f64 - max - sum.ln()).collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, quant, devices, vram, first, second, rest @ ..] = &args[..] else {
        return Err("usage: qwen_tables MODEL QUANT DEVICES VRAM_GIB FIRST SECOND [TOKENS]".into());
    };
    let model = PathBuf::from(model);
    let id = bundle::Id::find(Some(bundle::MODEL), None, quant)?;
    let devices: Vec<usize> = devices.split(',').map(|device| device.parse().unwrap()).collect();
    let budget = (vram.parse::<f64>().unwrap() * (1u64 << 30) as f64) as u64;
    let steps: usize = rest.first().map_or(256, |tokens| tokens.parse().unwrap());
    let a = run(&model, id, &devices, budget, first, steps)?;
    let b = run(&model, id, &devices, budget, second, steps)?;
    let mut all = Vec::new();
    for (index, ((tokens_a, logits_a), (tokens_b, logits_b))) in a.iter().zip(&b).enumerate() {
        // (the steps whose context is the same: up to and including the first token the two continuations differ in)
        let parted = tokens_a.iter().zip(tokens_b).position(|(x, y)| x != y);
        let shared = parted.map_or(tokens_a.len().min(tokens_b.len()), |at| at + 1).min(logits_a.len()).min(logits_b.len());
        let mut kls = Vec::new();
        for step in 0..shared {
            let (p, q) = (log_softmax(&logits_a[step]), log_softmax(&logits_b[step]));
            kls.push(p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum::<f64>().max(0.0));
        }
        let mean = kls.iter().sum::<f64>() / kls.len().max(1) as f64;
        println!("prompt {index}: {} steps compared, continuations {}, KL mean {mean:.6}, max {:.6}", kls.len(),
                 parted.map_or("agree throughout".to_string(), |at| format!("part at token {at}")), kls.iter().copied().fold(0.0, f64::max));
        all.extend(kls);
    }
    all.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mean = all.iter().sum::<f64>() / all.len().max(1) as f64;
    let p99 = all.get(((all.len() as f64 * 0.99) as usize).min(all.len().saturating_sub(1))).copied().unwrap_or(0.0);
    println!("all: {} steps, KL({first} || {second}) mean {mean:.6}, p99 {p99:.6}, max {:.6}", all.len(), all.last().copied().unwrap_or(0.0));
    Ok(())
}
