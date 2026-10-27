//! Development: DeepSeek-V4.1's prompt-pass indexer (32 heads of 128, top 512) on one GPU: a batch of
//! 4096 rows against a context's compressed groups through the FP16 (WMMA) path, every row seeing them
//! all. Prints the wall time a launch and a digest of the scores and the chosen groups, to tell two
//! builds apart bit for bit.
//!
//! cargo run --release --features hip -p straydog-engine --example ds41_index_prefill_bench [ITERATIONS], under the GPU lock.

use sha2::{Digest, Sha256};
use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;

const HEADS: usize = 32;
const DIM: usize = 128;
const TOP: usize = 512;
const ROPE: usize = 64;
const ROWS: usize = 4096;

fn upload<T: Copy>(gpu: &mut Backend, values: &[T]) -> Buffer {
    let buffer = gpu.buffer(std::mem::size_of_val(values) as u64).unwrap();
    gpu.write_buffer(&buffer, 0, values).unwrap();
    buffer
}

fn main() -> Result<(), String> {
    let iterations: usize = std::env::args().nth(1).map_or(3, |value| value.parse().unwrap());
    let mut gpu = Backend::new(0, 30 << 30)?;
    gpu.configure_gemm(8, true)?;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut floats = |count: usize, scale: f32| -> Vec<f32> {
        (0..count).map(|_| {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        }).collect()
    };
    let q = upload(&mut gpu, &floats(ROWS * HEADS * DIM, 1.0));
    let weights = upload(&mut gpu, &floats(ROWS * HEADS, 0.2));
    let frequencies = upload(&mut gpu, &(0..ROPE / 2).map(|i| 1.0f32 / 160000f32.powf(2.0 * i as f32 / ROPE as f32)).collect::<Vec<_>>());
    let positions = upload(&mut gpu, &(0..ROWS as i32).map(|row| 70_000 + row).collect::<Vec<_>>());
    let out = gpu.buffer((ROWS * TOP * 4) as u64)?;
    let p = |buffer: &Buffer| buffer.base().cast_const();
    for groups in [4096usize, 17040, 34080, 68160] {
        let keys = upload(&mut gpu, &floats(groups * DIM, 1.0));
        let visible = upload(&mut gpu, &vec![groups as i32; ROWS]);
        let scores = gpu.buffer((ROWS * groups * 4) as u64)?;
        let scale = 1.0 / ((DIM as f32).sqrt() * (HEADS as f32).sqrt());
        let run = |gpu: &mut Backend| gpu.index_select(p(&q), p(&keys), p(&weights), p(&visible), ROWS as u64, groups as u64, HEADS as u64,
                                                           DIM as u64, scale, (ROPE as u64, &positions, &frequencies), scores.base(),
                                                           TOP as u64, out.base(), None);
        run(&mut gpu)?;
        let mut sink = [0i32; 1];
        gpu.read_buffer(&out, 0, &mut sink)?;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            run(&mut gpu)?;
        }
        gpu.read_buffer(&out, 0, &mut sink)?;
        let each = started.elapsed().as_secs_f64() / iterations as f64 * 1e3;
        let mut values = vec![0f32; ROWS * groups];
        gpu.read_buffer(&scores, 0, &mut values)?;
        let mut chosen = vec![0i32; ROWS * TOP];
        gpu.read_buffer(&out, 0, &mut chosen)?;
        let digest = |bytes: Vec<u8>| format!("{:x}", Sha256::digest(&bytes))[..16].to_string();
        println!("{groups:6} groups, {ROWS} rows: {each:8.2} ms a launch; scores {}, chosen {}",
                 digest(values.iter().flat_map(|v| v.to_le_bytes()).collect()), digest(chosen.iter().flat_map(|v| v.to_le_bytes()).collect()));
        for buffer in [keys, visible, scores] {
            gpu.release_buffer(buffer)?;
        }
    }
    Ok(())
}
