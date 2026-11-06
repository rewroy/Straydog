//! Times the PLE rows of a new prompt on a real Qwen3.8 bundle, as the engine asks for them: a prefetch as soon as the batch's token ids are
//! known, then the gather that waits for what is still on its way. Each run is a prompt of new random text (no row of it is in the row
//! cache), so it shows what a first request waits for before its first layer.
//!
//! The table opens as the worker opens it: the weights, the shards' verity trees, `Weights::cache_rows`. Without verification
//! (`--no-verify`: the trees are not opened) the time is that of the reads alone.
//!
//! usage: cargo run --release --example ple_prompt_reads -- MODEL QUANT WORDS [--runs N] [--piece TOKENS] [--no-verify]
//!   WORDS: the words of each prompt (about 1.3 tokens each); --piece: tokens a batch, the next batch's rows asked for as one starts
//!   (0, the default: the prompt in one batch, up to 4096 tokens).

use std::path::PathBuf;
use std::time::Instant;

use straydog_engine::bundle;
use straydog_engine::rows;
use straydog_engine::text::Text;
use straydog_engine::weights::{ngram_rows, Weights};

const PLE_TABLE: &str = "per_layer_token_embd.weight";
const WORDS: &str = "the engine reads a layer of weights and the router picks experts for every token while the cache keeps the keys \
    and values of earlier positions so a later token can attend to them a scheduler orders the work and a buffer holds \
    the activations between the layers when memory is short some experts stay on the host and are sent over the bus";

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, quant, words, rest @ ..] = &args[..] else {
        return Err("usage: ple_prompt_reads MODEL QUANT WORDS [--runs N] [--piece TOKENS] [--settle MS] [--no-verify]".into());
    };
    let model = PathBuf::from(model);
    let words: usize = words.parse().map_err(|_| "WORDS is a number")?;
    let option = |name: &str| rest.iter().position(|arg| arg == name).and_then(|at| rest.get(at + 1)).map(|value| value.parse::<usize>().unwrap());
    let (runs, piece) = (option("--runs").unwrap_or(4), option("--piece").unwrap_or(0));
    let verify = !rest.iter().any(|arg| arg == "--no-verify");
    let id = bundle::Id::find(Some(bundle::MODEL), None, quant)?;
    let started = Instant::now();
    let mut weights = Weights::open(&model, id, None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {})?;
    if verify {
        weights.open_trees(&bundle::default_data_dir(&model))?;
    }
    let [_, table_rows] = weights.tensor(PLE_TABLE)?.dims[..] else { return Err("not a table of rows".into()) };
    let width = weights.tensor(PLE_TABLE)?.bytes / table_rows;
    let budget = rows::auto_budget(table_rows, width, weights.host_memory_bytes, 14 << 30);
    let opened = Instant::now();
    weights.cache_rows(PLE_TABLE, budget)?;
    if let Some(wait) = option("--settle") {
        std::thread::sleep(std::time::Duration::from_millis(wait as u64));
    }
    let table = weights.row_table(PLE_TABLE).ok_or("no row table")?;
    let text = Text::new(weights.resources.get("tokenizer.json").ok_or("no tokenizer")?).map_err(|exc| exc.0)?;
    eprintln!("opened in {:.2} s: {table_rows} rows of {width} B, cache {:.1} GiB, rows verified: {verify}", started.elapsed().as_secs_f64(),
              budget as f64 / (1u64 << 30) as f64);
    // --ramp N: the first thing done is N random rows asked for at once, and how many are read at each tick says how fast the reads start.
    if let Some(count) = option("--ramp") {
        let mut seed = 0x1234_5678_9ABC_DEF1u64 ^ started.elapsed().as_nanos() as u64;
        let wanted: Vec<u32> = (0..count).map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % table_rows) as u32
        }).collect();
        let began = Instant::now();
        table.prefetch(&wanted)?;
        let mut line = String::new();
        let mut last = 0;
        for tick in [2u64, 5, 10, 20, 50, 100, 200, 400, 800, 1600] {
            while (began.elapsed().as_millis() as u64) < tick && table.counts().1 < count as u64 {
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            let done = table.counts().1;
            line += &format!(" {tick}ms:{done}");
            last = done;
            if done >= count as u64 {
                break;
            }
        }
        table.read(&wanted)?;
        println!("ramp of {count} rows:{line} (all {} at {:.1} ms)", last.max(table.counts().1), began.elapsed().as_secs_f64() * 1e3);
    }
    let as_rows = |indices: &[i64]| -> Vec<u32> { indices.iter().map(|&index| index as u32).collect() };
    let vocabulary: Vec<&str> = WORDS.split_whitespace().collect();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ std::process::id() as u64 ^ started.elapsed().as_nanos() as u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    // A short request first, so that no measured prompt is the first the table serves.
    let mut counts = (0, 0, 0);
    for run in 0..=runs {
        let count = if run == 0 { 60 } else { words };
        let body: Vec<&str> = (0..count).map(|_| vocabulary[next() as usize % vocabulary.len()]).collect();
        let prompt = format!("Document {run:02}-{:08x}. Read the notes below, then answer in one sentence: what is the main topic?\n\n{}", next() >> 32, body.join(" "));
        let tokens: Vec<i64> = text.encode(&prompt).map_err(|exc| exc.0)?.iter().map(|&token| token as i64).collect();
        let tokens = &tokens[..tokens.len().min(if piece == 0 { 4096 } else { usize::MAX })];
        let parts: Vec<&[i64]> = if piece == 0 { vec![tokens] } else { tokens.chunks(piece).collect() };
        let indices = |part: usize| -> Result<Vec<i64>, String> {
            let before = &tokens[..parts[..part].iter().map(|p| p.len()).sum()];
            ngram_rows(&before[before.len().saturating_sub(2)..], parts[part], &weights.integer_arrays)
        };
        let began = Instant::now();
        let (mut prefetch, mut wait, mut distinct) = (0.0, 0.0, 0usize);
        let mut ahead = Some(indices(0)?);
        for part in 0..parts.len() {
            let wanted = ahead.take().unwrap();
            let clock = Instant::now();
            // (the first batch's rows are asked for here; later ones were asked for as the one before them began)
            if part == 0 {
                table.prefetch(&as_rows(&wanted))?;
            }
            if part + 1 < parts.len() {
                let next_rows = indices(part + 1)?;
                table.prefetch(&as_rows(&next_rows))?;
                ahead = Some(next_rows);
            }
            prefetch += clock.elapsed().as_secs_f64();
            let clock = Instant::now();
            let bytes = table.read(&as_rows(&wanted))?;
            wait += clock.elapsed().as_secs_f64();
            assert_eq!(bytes.len(), wanted.len() * width as usize);
            let mut sorted = wanted;
            sorted.sort_unstable();
            sorted.dedup();
            distinct += sorted.len();
        }
        let now = table.counts();
        let (asked, read, blocks) = (now.0 - counts.0, now.1 - counts.1, now.2 - counts.2);
        counts = now;
        let total = began.elapsed().as_secs_f64();
        println!("at {:.0} ms: {} {} tokens, {asked} rows asked ({distinct} distinct), {read} read, {blocks} blocks: asks {:.1} ms, waits {:.1} ms, in all {:.1} ms \
                  ({:.0} blocks/s)", began.duration_since(opened).as_secs_f64() * 1e3, if run == 0 { "first:" } else { "prompt:" }, tokens.len(), prefetch * 1e3, wait * 1e3, total * 1e3,
                 blocks as f64 / total);
    }
    Ok(())
}
