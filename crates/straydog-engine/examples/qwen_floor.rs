//! What speed first (STRAYDOG_EXPERT_FLOOR) costs a partially resident Qwen3.8 and what it buys: the bundle in MODEL is loaded at
//! each floor in turn (the first 0: every route), each prompt continued greedily by N tokens with every step's logits kept and the
//! decode timed; each floor's logits are compared with the first's while the continuations agree: KL mean, p99 and maximum, where
//! they part, and tokens a second.
//!
//! cargo run --release --example qwen_floor --features hip -- MODEL QUANT DEVICES VRAM_GIB FLOORS [TOKENS]   (FLOORS: 0,0.03,0.05)
//! A floor written with a trailing q (0q) runs the CPU experts on 8-bit activations (STRAYDOG_CPU_INT8=1); without it, on F32.
//!
//! A floor may end in @f32 or @q8 (after a q): that run takes the GPU projections on the F32 kernels (STRAYDOG_Q8=0) or the Q8 kernels
//! (INT8 activations), whatever the environment says. `0.04@f32,0.04@q8` is the KL of the Q8 kernels against the F32 ones, at one floor.

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

/// Each prompt's greedy tokens and every step's logits at `floor`, and the decode's tokens a second.
fn run(model_dir: &PathBuf, id: bundle::Id, devices: &[usize], budget: u64, floor: &str, steps: usize)
       -> Result<(Vec<(Vec<u32>, Vec<Vec<f32>>)>, f64), String> {
    let (floor, mode) = floor.split_once('@').map_or((floor, None), |(floor, mode)| (floor, Some(mode)));
    let (floor, int8) = floor.strip_suffix('q').map_or((floor, "0"), |floor| (floor, "1"));
    std::env::set_var("STRAYDOG_EXPERT_FLOOR", floor);
    std::env::set_var("STRAYDOG_CPU_INT8", int8);
    match mode {
        Some("f32") => std::env::set_var("STRAYDOG_Q8", "0"),
        Some("q8") => std::env::remove_var("STRAYDOG_Q8"),
        Some(other) => return Err(format!("no mode {other}: @f32 or @q8")),
        None => {}
    }
    let mut group = Group::new(devices, &vec![budget; devices.len()], SplitMode::Experts)?;
    group.configure_gemm(2, true)?;
    let weights = Weights::open(model_dir, id, None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {})?;
    let text = Text::new(weights.resources.get("tokenizer.json").ok_or("no tokenizer")?).map_err(|exc| exc.0)?;
    let settings = Settings { context: 32_768, state_budget_bytes: 8 << 30, prefill_batch_size: 2048, speculative: 0, cpu_threads: 0,
                              row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {})?;
    model.lock_banks_now()?;
    eprintln!("floor {floor}: loaded");
    let mut out = Vec::new();
    let (mut seconds, mut made) = (0.0, 0usize);
    for prompt in PROMPTS {
        let tokens = text.encode(prompt).map_err(|exc| exc.0)?;
        model.logits_capture = Some(Vec::new());
        let mut generated = Vec::new();
        let started = std::time::Instant::now();
        model.generate(&tokens, steps, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, false, false, &mut |token| generated.push(token))?;
        seconds += started.elapsed().as_secs_f64();
        made += generated.len();
        out.push((generated, model.logits_capture.take().unwrap_or_default()));
    }
    Ok((out, made as f64 / seconds))
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&value| (value as f64 - max).exp()).sum();
    logits.iter().map(|&value| value as f64 - max - sum.ln()).collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, quant, devices, vram, floors, rest @ ..] = &args[..] else {
        return Err("usage: qwen_floor MODEL QUANT DEVICES VRAM_GIB FLOORS [TOKENS]".into());
    };
    let model = PathBuf::from(model);
    let id = bundle::Id::find(Some(bundle::MODEL), None, quant)?;
    let devices: Vec<usize> = devices.split(',').map(|device| device.parse().unwrap()).collect();
    let budget = (vram.parse::<f64>().unwrap() * (1u64 << 30) as f64) as u64;
    let steps: usize = rest.first().map_or(256, |tokens| tokens.parse().unwrap());
    let floors: Vec<&str> = floors.split(',').collect();
    let (a, speed) = run(&model, id, &devices, budget, floors[0], steps)?;
    println!("floor {}: {speed:.1} tok/s (the reference)", floors[0]);
    for floor in &floors[1..] {
        let (b, speed) = run(&model, id, &devices, budget, floor, steps)?;
        let mut all = Vec::new();
        let mut parted_at = Vec::new();
        for ((tokens_a, logits_a), (tokens_b, logits_b)) in a.iter().zip(&b) {
            // (the steps whose context is the same: up to and including the first token the two continuations differ in)
            let parted = tokens_a.iter().zip(tokens_b).position(|(x, y)| x != y);
            parted_at.push(parted.map_or("-".to_string(), |at| at.to_string()));
            let shared = parted.map_or(tokens_a.len().min(tokens_b.len()), |at| at + 1).min(logits_a.len()).min(logits_b.len());
            for step in 0..shared {
                let (p, q) = (log_softmax(&logits_a[step]), log_softmax(&logits_b[step]));
                all.push(p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum::<f64>().max(0.0));
            }
        }
        all.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mean = all.iter().sum::<f64>() / all.len().max(1) as f64;
        let p99 = all.get(((all.len() as f64 * 0.99) as usize).min(all.len().saturating_sub(1))).copied().unwrap_or(0.0);
        println!("floor {floor}: {speed:.1} tok/s, KL mean {mean:.5}, p99 {p99:.5}, max {:.5} over {} steps; prompts part at {}",
                 all.last().copied().unwrap_or(0.0), all.len(), parted_at.join(" "));
    }
    Ok(())
}
