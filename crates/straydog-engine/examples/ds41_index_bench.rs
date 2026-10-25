//! Development: DeepSeek-V4.1's decode indexer (32 heads of 128, top 512) at a context's compressed
//! groups, on one GPU: the fused select (scores, then the row's top k in its last block), and the
//! scores and the top k apart. Prints the wall time a launch, after a check that the select's
//! scores are op_index_scores' bit for bit, with and without rope.
//!
//! cargo run --release --features hip -p straydog-engine --example ds41_index_bench [ITERATIONS], under the GPU lock.


use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;

const HEADS: usize = 32;
const DIM: usize = 128;
const TOP: usize = 512;
const ROPE: usize = 64;

fn upload<T: Copy>(gpu: &mut Backend, values: &[T]) -> Buffer {
    let buffer = gpu.buffer(std::mem::size_of_val(values) as u64).unwrap();
    gpu.write_buffer(&buffer, 0, values).unwrap();
    buffer
}

fn main() -> Result<(), String> {
    let iterations: usize = std::env::args().nth(1).map_or(200, |value| value.parse().unwrap());
    let mut gpu = Backend::new(0, 30 << 30)?;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut floats = |count: usize, scale: f32| -> Vec<f32> {
        (0..count).map(|_| {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        }).collect()
    };
    let q = upload(&mut gpu, &floats(HEADS * DIM, 1.0));
    let weights = upload(&mut gpu, &floats(HEADS, 0.2));
    let frequencies = upload(&mut gpu, &(0..ROPE / 2).map(|i| 1.0f32 / 160000f32.powf(2.0 * i as f32 / ROPE as f32)).collect::<Vec<_>>());
    let out = gpu.buffer((TOP * 4) as u64)?;
    let p = |buffer: &Buffer| buffer.base().cast_const();
    // (a check first: the fused select's scores against op_index_scores', with and without rope)
    for rope in [0usize, ROPE] {
        let groups = 3000usize;
        let keys = upload(&mut gpu, &floats(groups * DIM, 1.0));
        let visible = upload(&mut gpu, &[groups as i32]);
        let positions = upload(&mut gpu, &[20_000i32]);
        let (scores_a, scores_b) = (gpu.buffer((groups * 4) as u64)?, gpu.buffer((groups * 4) as u64)?);
        let q_rot = upload(&mut gpu, &floats(HEADS * DIM, 1.0));
        let q_raw = gpu.buffer((HEADS * DIM * 4) as u64)?;
        gpu.copy(p(&q_rot), q_raw.base(), (HEADS * DIM * 4) as u64)?;
        if rope > 0 {
            gpu.rope(q_rot.base(), 1, HEADS as u64, DIM as u64, rope as u64, &positions, &frequencies, false)?;
        }
        let scale = 0.01;
        gpu.index_scores(p(&q_rot), p(&keys), p(&weights), p(&visible), 1, groups as u64, HEADS as u64, DIM as u64, scale, scores_a.base())?;
        gpu.index_select(p(&q_raw), p(&keys), p(&weights), p(&visible), 1, groups as u64, HEADS as u64, DIM as u64, scale,
                         (rope as u64, &positions, &frequencies), scores_b.base(), TOP as u64, out.base())?;
        let (mut a, mut b) = (vec![0f32; groups], vec![0f32; groups]);
        gpu.read_buffer(&scores_a, 0, &mut a)?;
        gpu.read_buffer(&scores_b, 0, &mut b)?;
        let differ = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        println!("check, rope {rope}: {differ} of {groups} scores differ from op_index_scores");
    }
    for groups in [1024usize, 4096, 8192, 17040, 32768, 65536] {
        let keys = upload(&mut gpu, &floats(groups * DIM, 1.0));
        let visible = upload(&mut gpu, &[groups as i32]);
        let positions = upload(&mut gpu, &[(groups * 4) as i32]);
        let scores = gpu.buffer((groups * 4) as u64)?;
        let mut sink = [0i32; 1];
        let mut time = |gpu: &mut Backend, run: &mut dyn FnMut(&mut Backend) -> Result<(), String>| -> Result<f64, String> {
            for _ in 0..5 {
                run(gpu)?;
            }
            gpu.read_buffer(&out, 0, &mut sink)?;
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                run(gpu)?;
            }
            gpu.read_buffer(&out, 0, &mut sink)?;
            Ok(started.elapsed().as_secs_f64() / iterations as f64 * 1e6)
        };
        let scale = 1.0 / (DIM as f32).sqrt();
        let fused = time(&mut gpu, &mut |gpu| gpu.index_select(p(&q), p(&keys), p(&weights), p(&visible), 1, groups as u64, HEADS as u64,
                                                                   DIM as u64, scale, (ROPE as u64, &positions, &frequencies), scores.base(),
                                                                   TOP as u64, out.base()))?;
        let scored = time(&mut gpu, &mut |gpu| gpu.index_scores(p(&q), p(&keys), p(&weights), p(&visible), 1, groups as u64, HEADS as u64,
                                                                    DIM as u64, scale, scores.base()))?;
        let top = time(&mut gpu, &mut |gpu| gpu.top_k(p(&scores), 1, groups as u64, TOP as u64, out.base()))?;
        println!("{groups:6} groups (context {:6}): select {fused:7.1} us, scores alone {scored:7.1} us, top {TOP} alone {top:7.1} us",
                 groups * 4);
        for buffer in [keys, visible, positions, scores] {
            gpu.release_buffer(buffer)?;
        }
    }
    Ok(())
}
