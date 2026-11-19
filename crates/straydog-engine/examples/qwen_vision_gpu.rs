//! Qwen3.8's vision encoder on the CPU and on a GPU, over the same images: how far apart their rows are (each row's largest
//! difference, against the row's own largest value) and how long each took. Development only.
//!
//! qwen_vision_gpu MMPROJ_FILE BUDGET_GIB IMAGE...

use std::time::Instant;

use straydog_engine::gpu::Backend;
use straydog_engine::qwen_vision::{self, Encoder};

fn main() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [file, budget, images @ ..] = arguments.as_slice() else {
        return Err("qwen_vision_gpu MMPROJ_FILE BUDGET_GIB IMAGE...".into());
    };
    let pin = straydog_engine::bundle::qwen_vision_source()?.asset;
    let encoder = Encoder::open(std::path::Path::new(file), &pin)?;
    let budget = (budget.parse::<f64>().map_err(|error| error.to_string())? * (1u64 << 30) as f64) as u64;
    let mut gpu = Backend::new(0, budget)?;
    let never = || false;
    for path in images {
        let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
        let rgb = straydog_engine::vision::decode_upright(&bytes)?;
        let image = qwen_vision::prepare(&rgb)?;
        let started = Instant::now();
        let cpu = encoder.encode(&image, &never)?.ok_or("cancelled")?;
        let cpu_seconds = started.elapsed().as_secs_f64();
        let mut gpu_seconds = Vec::new();
        let mut rows = Vec::new();
        // (QV_ONCE=1: one run on the GPU, for a sanitizer)
        let runs = if std::env::var("QV_ONCE").is_ok_and(|value| value == "1") { 1 } else { 2 };
        for _ in 0..runs {
            let started = Instant::now();
            rows = encoder.encode_on(&mut gpu, &image, &never)?.ok_or("cancelled")?;
            gpu_seconds.push(started.elapsed().as_secs_f64());
        }
        let width = qwen_vision::ROW;
        let (mut worst, mut worst_relative, mut at) = (0f32, 0f32, 0);
        let mut sum_relative = 0f64;
        for (index, (a, b)) in cpu.chunks_exact(width).zip(rows.chunks_exact(width)).enumerate() {
            let largest = a.iter().fold(0f32, |most, value| most.max(value.abs()));
            let difference = a.iter().zip(b).fold(0f32, |most, (x, y)| most.max((x - y).abs()));
            let relative = difference / largest.max(1e-6);
            sum_relative += f64::from(relative);
            if relative > worst_relative {
                (worst, worst_relative, at) = (difference, relative, index);
            }
        }
        let tokens = cpu.len() / width;
        let row_max: Vec<f32> = cpu.chunks_exact(width).map(|row| row.iter().fold(0f32, |m, v| m.max(v.abs()))).collect();
        let mut sorted = row_max.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let overall = cpu.iter().zip(&rows).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("  rows' largest values: least {:.4}, median {:.4}, most {:.4}; row {at}'s {:.4}; largest difference anywhere {overall:.5}",
                 sorted[0], sorted[sorted.len() / 2], sorted[sorted.len() - 1], row_max[at]);
        let cosine = {
            let (dot, a2, b2) = cpu.iter().zip(&rows).fold((0f64, 0f64, 0f64), |(d, x, y), (a, b)| (d + f64::from(*a) * f64::from(*b), x + f64::from(*a).powi(2), y + f64::from(*b).powi(2)));
            dot / (a2.sqrt() * b2.sqrt())
        };
        println!("{path}: {}x{} patches, {tokens} rows; CPU {cpu_seconds:.2} s, GPU {:?} s", image.rows, image.columns, gpu_seconds);
        println!("  largest difference in a row {worst:.5} ({:.3}% of that row's largest value, row {at}); mean of rows' {:.4}%; cosine {cosine:.7}",
                 worst_relative * 100.0, sum_relative / tokens as f64 * 100.0);
    }
    Ok(())
}
