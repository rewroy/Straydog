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
//! @e0, @e1 and @e2 set the routed experts' gate and up the same way (STRAYDOG_EXPERT_Q8=0, the F32 kernels; 1; 2). Unset, a CUDA
//! build takes the INT8 ones, so @e0 is F32 and not the default.
//! @mma leaves the prompt's sparse attention on tensor cores (a CUDA build's default from sm_80); a run without it sets STRAYDOG_QSA_MMA=0
//! (the F32 kernel), so `0,0@mma` is the KL of the tensor-core kernel against the F32 one.
//!
//! QWEN_FLOOR_PROMPT_FILE and QWEN_FLOOR_PROMPT_TOKENS (a comma list such as 8192,32768): the prompts are the file's first N tokens each
//! instead of the six short ones (the context grows to hold the longest and its continuation).
//!
//! QWEN_FLOOR_SAVE=<file>: after the run of the first floor, its results (each prompt's tokens, greedy continuation and every step's
//! logits) are written to <file> (temp file, then rename). QWEN_FLOOR_REFERENCE=<file>: such a file is the reference in place of the
//! first floor's own run, and every floor in this process, the first included, is compared with it; the prompts must be the same.
//! Together they compare runs across processes: run once with QWEN_FLOOR_SAVE, once more (another build, another environment) with
//! QWEN_FLOOR_REFERENCE. The modes above set their switches where the engine's C++ reads them too (through the C runtime on Windows,
//! whose getenv does not see a Rust set_var), and each floor loads the model again, so they also hold within one process. The file is little-endian: the magic "QFLR0001", u32 prompt count, u32 vocabulary, then per
//! prompt u32 prompt length and its u32 token ids, u32 step count and the u32 greedy tokens, u32 rows (the logits kept) and rows x
//! vocabulary f32 logits.

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

/// Each prompt's greedy tokens and every step's logits.
type Results = Vec<(Vec<u32>, Vec<Vec<f32>>)>;

const MAGIC: &[u8; 8] = b"QFLR0001";

/// Writes the prompts and `results` to `file` in the layout in the top comment, through a temp file renamed over it.
fn save(file: &std::path::Path, prompts: &[Vec<u32>], results: &Results) -> Result<(), String> {
    let vocabulary = results.iter().flat_map(|(_, rows)| rows.first()).map(Vec::len).next().unwrap_or(0);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(prompts.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(vocabulary as u32).to_le_bytes());
    for (prompt, (tokens, rows)) in prompts.iter().zip(results) {
        for list in [prompt, tokens] {
            bytes.extend_from_slice(&(list.len() as u32).to_le_bytes());
            for token in list { bytes.extend_from_slice(&token.to_le_bytes()); }
        }
        bytes.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        for row in rows {
            if row.len() != vocabulary { return Err(format!("a logits row of {} values, the vocabulary being {vocabulary}", row.len())); }
            for value in row { bytes.extend_from_slice(&value.to_le_bytes()); }
        }
    }
    let mut temporary = file.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    std::fs::write(&temporary, &bytes).map_err(|exc| format!("{}: {exc}", temporary.display()))?;
    std::fs::rename(&temporary, file).map_err(|exc| format!("{}: {exc}", file.display()))
}

/// Reads a file `save` wrote: the prompts and the results.
fn load(file: &std::path::Path) -> Result<(Vec<Vec<u32>>, Results), String> {
    let bytes = std::fs::read(file).map_err(|exc| format!("{}: {exc}", file.display()))?;
    let bad = || format!("{}: not a complete QWEN_FLOOR_SAVE file", file.display());
    if bytes.len() < 16 || &bytes[..8] != MAGIC { return Err(bad()); }
    let mut at = 8usize;
    let word = |at: &mut usize| -> Result<[u8; 4], String> {
        let word = bytes.get(*at..*at + 4).ok_or_else(bad)?;
        *at += 4;
        Ok([word[0], word[1], word[2], word[3]])
    };
    let count = u32::from_le_bytes(word(&mut at)?) as usize;
    let vocabulary = u32::from_le_bytes(word(&mut at)?) as usize;
    let (mut prompts, mut results) = (Vec::new(), Vec::new());
    for _ in 0..count {
        let mut lists = Vec::new();
        for _ in 0..2 {
            let length = u32::from_le_bytes(word(&mut at)?) as usize;
            let mut list = Vec::new();
            for _ in 0..length { list.push(u32::from_le_bytes(word(&mut at)?)); }
            lists.push(list);
        }
        let rows = u32::from_le_bytes(word(&mut at)?) as usize;
        let mut logits = Vec::new();
        for _ in 0..rows {
            let mut row = Vec::with_capacity(vocabulary);
            for _ in 0..vocabulary { row.push(f32::from_le_bytes(word(&mut at)?)); }
            logits.push(row);
        }
        let tokens = lists.pop().unwrap();
        prompts.push(lists.pop().unwrap());
        results.push((tokens, logits));
    }
    if at != bytes.len() { return Err(bad()); }
    Ok((prompts, results))
}

/// Sets or removes a variable where the engine reads it, its C++ included. On Windows std::env::set_var changes the process's environment
/// block, which the C runtime's getenv (its own copy, made at the start) does not see; the runtime's _putenv_s changes both.
fn set_engine_variable(name: &str, value: Option<&str>) {
    #[cfg(windows)]
    {
        extern "C" { fn _putenv_s(name: *const std::ffi::c_char, value: *const std::ffi::c_char) -> i32; }
        let name = std::ffi::CString::new(name).unwrap();
        let value = std::ffi::CString::new(value.unwrap_or("")).unwrap();   // (an empty value removes the variable)
        assert_eq!(unsafe { _putenv_s(name.as_ptr(), value.as_ptr()) }, 0);
    }
    #[cfg(not(windows))]
    match value { Some(value) => std::env::set_var(name, value), None => std::env::remove_var(name) }
}

/// Each prompt, its greedy tokens and every step's logits at `floor`, and the decode's tokens a second.
fn run(model_dir: &PathBuf, id: bundle::Id, devices: &[usize], budget: u64, floor: &str, steps: usize)
       -> Result<(Vec<Vec<u32>>, Results, f64), String> {
    let (floor, mode) = floor.split_once('@').map_or((floor, None), |(floor, mode)| (floor, Some(mode)));
    let (floor, int8) = floor.strip_suffix('q').map_or((floor, "0"), |floor| (floor, "1"));
    set_engine_variable("STRAYDOG_EXPERT_FLOOR", Some(floor));
    set_engine_variable("STRAYDOG_CPU_INT8", Some(int8));
    set_engine_variable("STRAYDOG_QSA_MMA", Some("0"));
    match mode {
        Some("f32") => set_engine_variable("STRAYDOG_Q8", Some("0")),
        Some("q8") => set_engine_variable("STRAYDOG_Q8", None),
        Some("e0") => set_engine_variable("STRAYDOG_EXPERT_Q8", Some("0")),
        Some("e1") => set_engine_variable("STRAYDOG_EXPERT_Q8", Some("1")),
        Some("e2") => set_engine_variable("STRAYDOG_EXPERT_Q8", Some("2")),
        Some("mma") => set_engine_variable("STRAYDOG_QSA_MMA", None),
        Some(other) => return Err(format!("no mode {other}: @f32, @q8, @e0, @e1, @e2 or @mma")),
        None => {}
    }
    let mut group = Group::new(devices, &vec![budget; devices.len()], SplitMode::Experts)?;
    group.configure_gemm(2, true)?;
    // (QWEN_FLOOR_DRAFTER=<MTP drafter file>: the continuations by speculative decoding, three drafts, its verification passes of several rows)
    let drafter = std::env::var_os("QWEN_FLOOR_DRAFTER").map(PathBuf::from);
    let weights = Weights::open(model_dir, id, drafter.as_deref(), straydog_engine::host::host_memory_bytes(), &|_, _, _| {})?;
    let text = Text::new(weights.resources.get("tokenizer.json").ok_or("no tokenizer")?).map_err(|exc| exc.0)?;
    let prompts: Vec<Vec<u32>> = match (std::env::var_os("QWEN_FLOOR_PROMPT_FILE"), std::env::var("QWEN_FLOOR_PROMPT_TOKENS")) {
        (Some(file), Ok(counts)) => {
            let long = text.encode(&std::fs::read_to_string(&file).map_err(|exc| format!("{}: {exc}", PathBuf::from(&file).display()))?).map_err(|exc| exc.0)?;
            counts.split(',').map(|count| {
                let count: usize = count.trim().parse().map_err(|_| format!("QWEN_FLOOR_PROMPT_TOKENS: {count}"))?;
                if count > long.len() { return Err(format!("the prompt file has {} tokens, {count} asked", long.len())); }
                Ok(long[..count].to_vec())
            }).collect::<Result<_, String>>()?
        }
        _ => PROMPTS.iter().map(|prompt| text.encode(prompt).map_err(|exc| exc.0)).collect::<Result<_, String>>()?,
    };
    let context = prompts.iter().map(|prompt| prompt.len() + steps + 64).max().unwrap_or(0).max(32_768);
    let settings = Settings { context, state_budget_bytes: 8 << 30, prefill_batch_size: 2048, speculative: if drafter.is_some() { 3 } else { 0 }, cpu_threads: 0,
                              row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {})?;
    model.lock_banks_now()?;
    eprintln!("floor {floor}: loaded");
    let mut out = Vec::new();
    let (mut seconds, mut made) = (0.0, 0usize);
    for tokens in &prompts {
        model.logits_capture = Some(Vec::new());
        let mut generated = Vec::new();
        let started = std::time::Instant::now();
        model.generate(tokens, steps, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, false, false, &mut |token| generated.push(token))?;
        seconds += started.elapsed().as_secs_f64();
        made += generated.len();
        out.push((generated, model.logits_capture.take().unwrap_or_default()));
    }
    Ok((prompts, out, made as f64 / seconds))
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
    let save_to = std::env::var_os("QWEN_FLOOR_SAVE").map(PathBuf::from);
    let reference = std::env::var_os("QWEN_FLOOR_REFERENCE").map(PathBuf::from)
        .map(|file| load(&file).map(|loaded| (file, loaded))).transpose()?;
    // (the reference: the file's results, or the first floor's own run)
    let (a, compared) = match reference {
        Some((file, (prompts, results))) => {
            println!("reference: {} ({} prompts)", file.display(), prompts.len());
            ((prompts, results), &floors[..])
        }
        None => {
            let (prompts, results, speed) = run(&model, id, &devices, budget, floors[0], steps)?;
            println!("floor {}: {speed:.1} tok/s (the reference)", floors[0]);
            ((prompts, results), &floors[1..])
        }
    };
    if let Some(file) = &save_to {
        // (the first floor's run: the reference's own results are not written back)
        if std::env::var_os("QWEN_FLOOR_REFERENCE").is_some() {
            return Err("QWEN_FLOOR_SAVE saves the first floor's run; with QWEN_FLOOR_REFERENCE there is none to save in this process".into());
        }
        save(file, &a.0, &a.1)?;
        println!("saved {} prompts to {}", a.0.len(), file.display());
    }
    let (reference_prompts, a) = a;
    for floor in compared {
        let (prompts, b, speed) = run(&model, id, &devices, budget, floor, steps)?;
        if prompts.len() != reference_prompts.len() {
            return Err(format!("the reference has {} prompts, this run {}", reference_prompts.len(), prompts.len()));
        }
        for (index, (x, y)) in reference_prompts.iter().zip(&prompts).enumerate() {
            if x != y {
                return Err(format!("prompt {index} differs from the reference's (first token {:?} against {:?}, {} tokens against {})",
                                   y.first(), x.first(), y.len(), x.len()));
            }
        }
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
