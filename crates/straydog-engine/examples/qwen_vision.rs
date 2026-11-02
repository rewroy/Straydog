//! Qwen3.8-Flash-Next's vision encoder alone, on the CPU: an image's rows, from the encoder's own preprocessing, to a file
//! of F32 values (rows of 2560, little-endian) for comparison with a reference run; nothing but the projector is read.
//!
//! cargo run --release -p straydog-engine --example qwen_vision -- MMPROJ IMAGE OUT

use std::path::Path;
use std::time::Instant;

use straydog_engine::bundle;
use straydog_engine::qwen_vision::{self, Encoder, ROW};
use straydog_engine::vision;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mmproj, image, out] = &args[..] else {
        return Err("usage: qwen_vision MMPROJ IMAGE OUT".into());
    };
    let started = Instant::now();
    let mut encoder = Encoder::open(Path::new(mmproj), &bundle::qwen_vision_source()?.asset)?;
    if let Some(threads) = std::env::var("THREADS").ok().and_then(|threads| threads.parse().ok()) {
        encoder.threads = threads;
    }
    println!("opened and authenticated in {:.2} s", started.elapsed().as_secs_f64());
    let bytes = std::fs::read(image).map_err(|exc| format!("{image}: {exc}"))?;
    let prepared = qwen_vision::prepare(&vision::decode(&bytes)?)?;
    println!("patches {}x{}, tokens {} ({:?})", prepared.rows, prepared.columns, prepared.tokens(), prepared.grid());
    let started = Instant::now();
    let rows = encoder.encode(&prepared, &|| false)?.ok_or("cancelled")?;
    println!("encoded in {:.2} s", started.elapsed().as_secs_f64());
    println!("first row's first values: {:?}", &rows[..8.min(rows.len())]);
    let mean = rows.iter().map(|value| f64::from(*value)).sum::<f64>() / rows.len() as f64;
    println!("{} rows of {ROW}, mean {mean:.6}", rows.len() / ROW);
    let raw: Vec<u8> = rows.iter().flat_map(|value| value.to_le_bytes()).collect();
    std::fs::write(out, raw).map_err(|exc| format!("{out}: {exc}"))
}
