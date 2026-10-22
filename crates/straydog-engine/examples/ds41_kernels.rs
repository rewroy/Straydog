//! Development: DeepSeek-V4.1's decode kernels at a GPU's two-GPU shapes, on one GPU, their weights
//! streamed from VRAM banks larger than the caches, for a kernel trace (rocprofv3 --kernel-trace):
//! each layer range's routed experts (gate and up, then down, six routes), the shared expert,
//! hc_step, and the router's route. Prints the wall time a launch as a bound.
//!
//! cargo run --release --features hip -p straydog-engine --example ds41_kernels [ITERATIONS], under the GPU lock.

use straydog_engine::gpu::Backend;
use straydog_engine::ops::{Buffer, ExpertSplit};

const WIDTH: usize = 5120;
const FFN: usize = 2304;
const ROUTES: usize = 6;
const PARTS: usize = 2;

/// A format's bytes a block and values a block.
fn block(kind: u32) -> (usize, usize) {
    match kind {
        8 => (34, 32),
        10 => (84, 256),
        16 => (66, 256),
        17 => (74, 256),
        18 => (98, 256),
        19 => (50, 256),
        22 => (82, 256),
        _ => panic!("no format {kind}"),
    }
}

fn row_bytes(kind: u32, cols: usize) -> usize {
    let (bytes, values) = block(kind);
    cols / values * bytes
}

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Random blocks of `kind` whose leading FP16 scale is 2^-4 (as every one of these formats
    /// leads with but Q2_K, whose random scales do not matter to the time).
    fn blocks(&mut self, kind: u32, count: usize) -> Vec<u8> {
        let (bytes, _) = block(kind);
        let mut out: Vec<u8> = (0..count * bytes).map(|_| self.next() as u8).collect();
        if kind != 10 {
            for chunk in out.chunks_exact_mut(bytes) {
                chunk[..2].copy_from_slice(&0x2c00u16.to_le_bytes());
            }
        }
        out
    }
    fn floats(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale).collect()
    }
}

fn upload(gpu: &mut Backend, values: &[f32]) -> Buffer {
    let buffer = gpu.buffer((values.len() * 4) as u64).unwrap();
    gpu.write_buffer(&buffer, 0, values).unwrap();
    buffer
}

fn main() -> Result<(), String> {
    let iterations: usize = std::env::args().nth(1).map_or(200, |value| value.parse().unwrap());
    let mut gpu = Backend::new(0, 30 << 30)?;
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let x = upload(&mut gpu, &random.floats(WIDTH, 1.0));
    let weights = upload(&mut gpu, &random.floats(ROUTES, 0.3));
    let product = gpu.buffer((ROUTES * FFN * 4) as u64)?;
    let out = gpu.buffer((WIDTH * 4) as u64)?;
    let (ffn_rows, hidden_rows) = (FFN / PARTS, WIDTH / PARTS);
    let split = |stages: i32| ExpertSplit { ffn_first: 0, ffn_rows: ffn_rows as u64, hidden_first: 0, hidden_rows: hidden_rows as u64, stages, send: 0 };
    // (gate and up, down): layer 0, 1-9, 10-19, 20-24, 25-34, 35-39; the shared expert
    for (label, gate, down) in [("IQ1_S / Q2_K", 19u32, 10u32), ("IQ1_S / IQ2_XS", 19, 17), ("IQ2_XXS / IQ2_S", 16, 22),
                                ("IQ2_XXS / IQ3_XXS", 16, 18), ("IQ2_S / IQ3_XXS", 22, 18), ("IQ2_S / IQ2_S", 22, 22),
                                ("Q8_0 shared", 8, 8)] {
        let gate_bytes = ffn_rows * row_bytes(gate, WIDTH);
        let down_bytes = hidden_rows * row_bytes(down, FFN);
        let slot = (2 * gate_bytes + down_bytes).next_multiple_of(256);
        let experts = ((384usize << 20) / slot).max(ROUTES);
        let mut bytes = Vec::with_capacity(experts * slot);
        for _ in 0..experts {
            let start = bytes.len();
            bytes.extend(random.blocks(gate, 2 * gate_bytes / block(gate).0));
            bytes.extend(random.blocks(down, down_bytes / block(down).0));
            bytes.resize(start + slot, 0);
        }
        let bank = gpu.upload_bytes(&bytes)?;
        drop(bytes);
        let offsets = [0, gate_bytes as u64, 2 * gate_bytes as u64];
        let started = std::time::Instant::now();
        for iteration in 0..iterations {
            let bases: Vec<u64> = (0..ROUTES).map(|route| bank.base() as u64 + (((iteration * ROUTES + route) * 7) % experts * slot) as u64).collect();
            gpu.experts(Some(&bases), slot as u64, offsets, [gate, gate, down], WIDTH as u64, FFN as u64, 1, ROUTES as u64, x.base(), 0,
                        Some(weights.base()), 10.0, false, product.base(), out.base(), 0, Some(split(1)))?;
            gpu.experts(None, slot as u64, offsets, [gate, gate, down], WIDTH as u64, FFN as u64, 1, ROUTES as u64, x.base(), 0, None,
                        10.0, false, product.base(), out.base(), 0, Some(split(2)))?;
        }
        let mut sink = [0f32; 1];
        gpu.read_buffer(&out, 0, &mut sink)?;
        let each = started.elapsed().as_secs_f64() / iterations as f64 * 1e6;
        println!("{label:20} gate+up {:.2} MB, down {:.2} MB a step (six routes): {each:.1} us a step (wall, both stages)",
                 (ROUTES * 2 * gate_bytes) as f64 / 1e6, (ROUTES * down_bytes) as f64 / 1e6);
        gpu.release_buffer(bank)?;
    }
    // hc_step at decode: one row, four streams, the coefficients' weights from a bank
    let (hc, mix) = (4usize, 24usize);
    let fn_bytes = mix * hc * WIDTH * 4;
    let count = (384usize << 20) / fn_bytes;
    let fns: Vec<Buffer> = (0..count).map(|_| upload(&mut gpu, &random.floats(mix * hc * WIDTH, 0.01))).collect();
    let stream = upload(&mut gpu, &random.floats(hc * WIDTH, 1.0));
    let stream2 = gpu.buffer((hc * WIDTH * 4) as u64)?;
    let base = upload(&mut gpu, &random.floats(mix, 0.1));
    let scale = upload(&mut gpu, &[0.5f32, 0.5, 0.5]);
    let (post, comb, pre) = (upload(&mut gpu, &random.floats(hc, 1.0)), upload(&mut gpu, &random.floats(hc * hc, 0.3)),
                             upload(&mut gpu, &random.floats(hc, 1.0)));
    let (pre_out, post_out, comb_out) = (gpu.buffer((hc * 4) as u64)?, gpu.buffer((hc * 4) as u64)?, gpu.buffer((hc * hc * 4) as u64)?);
    let norm = upload(&mut gpu, &random.floats(WIDTH, 1.0));
    let workspace = gpu.buffer(1 << 20)?;
    gpu.write_buffer(&workspace, 0, &vec![0u32; 1 << 18])?;
    let normed = gpu.buffer((WIDTH * 4) as u64)?;
    let started = std::time::Instant::now();
    for iteration in 0..iterations {
        let fn_ = &fns[iteration % count];
        gpu.hc_step(Some((out.base(), stream.base(), post.base(), comb.base())), stream2.base(),
                    Some((fn_.base(), base.base(), scale.base(), pre_out.base(), post_out.base(), comb_out.base())), 20, 1e-6, 1e-20,
                    pre.base(), norm.base(), 1, hc as u64, WIDTH as u64, &workspace, normed.base(), None)?;
    }
    let mut sink = [0f32; 1];
    gpu.read_buffer(&normed, 0, &mut sink)?;
    println!("hc_step (4 x 5120, coefficients): {:.1} us a launch (wall)", started.elapsed().as_secs_f64() / iterations as f64 * 1e6);
    // the router's route: 384 experts, square-root softplus, bias, top 6
    let logits = upload(&mut gpu, &random.floats(384, 3.0));
    let bias = upload(&mut gpu, &random.floats(384, 0.1));
    let (ids, route_weights) = (gpu.buffer(64)?, gpu.buffer(64)?);
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        gpu.route(logits.base(), Some(bias.base()), 1, 384, 6, 2, true, 1.5, ids.base(), route_weights.base())?;
    }
    gpu.read_buffer(&route_weights, 0, &mut sink)?;
    println!("route (384, sqrt softplus, bias, top 6): {:.1} us a launch (wall)", started.elapsed().as_secs_f64() / iterations as f64 * 1e6);
    Ok(())
}
