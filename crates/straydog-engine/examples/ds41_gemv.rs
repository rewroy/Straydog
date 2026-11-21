//! Development: DeepSeek-V4.1's decode projections at a GPU's two-GPU shapes, on one GPU, their
//! weights streamed from banks larger than the caches: q_b, wo_a (its four groups as one matrix) and
//! wo_b, in Q5_K (DS41_GEMV_KIND=8: Q8_0, Unsloth UD-IQ2_XXS's; 14: Q6_K, UD-IQ1_M's). Prints the wall
//! time a launch as a bound; run under rocprofv3 --kernel-trace for kernel time.
//!
//! cargo run --release --features hip -p straydog-engine --example ds41_gemv [ITERATIONS], under the GPU lock (DS41_Q8=0: the F32
//! kernels; DS41_VECTORS=N: a pass of N rows (1 to 4); DS41_GEMV_ONLY=TEXT: only the shapes whose label holds TEXT; DS41_GEMV_DUMP=PATH:
//! each shape's output to PATH.LABEL, to compare builds). "wo_a" is the engine's: four groups, each its own slice of the input.

use straydog_engine::gpu::Backend;
use straydog_engine::ops::{Buffer, Linear};


struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Random blocks of `kind` (`count` of `block` bytes): Q5_K's d and dmin 2^-4 and 2^-6, Q8_0's d 2^-4, Q6_K's d 2^-6.
    fn blocks(&mut self, kind: u32, block: usize, count: usize) -> Vec<u8> {
        let mut out: Vec<u8> = (0..count * block).map(|_| self.next() as u8).collect();
        for chunk in out.chunks_exact_mut(block) {
            match kind {
                14 => chunk[208..].copy_from_slice(&0x2400u16.to_le_bytes()),
                8 => chunk[..2].copy_from_slice(&0x2c00u16.to_le_bytes()),
                _ => {
                    chunk[..2].copy_from_slice(&0x2c00u16.to_le_bytes());
                    chunk[2..4].copy_from_slice(&0x2400u16.to_le_bytes());
                }
            }
        }
        out
    }
    fn floats(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0).collect()
    }
}

fn main() -> Result<(), String> {
    let iterations: usize = std::env::args().nth(1).map_or(300, |value| value.parse().unwrap());
    let mut gpu = Backend::new(0, 30 << 30)?;
    // (as the engine takes them: the Q8 kernels where it has them, DS41_Q8=0 the F32 ones)
    gpu.configure_decode(std::env::var("DS41_Q8").map_or(true, |value| value != "0"))?;
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    let vectors: usize = std::env::var("DS41_VECTORS").map_or(1, |value| value.parse().unwrap());
    let only = std::env::var("DS41_GEMV_ONLY").ok();
    // (the format, its values and bytes a block)
    let kind: u32 = std::env::var("DS41_GEMV_KIND").map_or(13, |value| value.parse().unwrap());
    let (per, block) = match kind {
        8 => (32usize, 34usize),
        13 => (256, 176),
        14 => (256, 210),
        _ => return Err(format!("DS41_GEMV_KIND {kind}: 8, 13 or 14")),
    };
    let input_values = random.floats(16384 * vectors);
    let input = gpu.buffer((input_values.len() * 4) as u64)?;
    gpu.write_buffer(&input, 0, &input_values)?;
    let output = gpu.buffer((16384 * 4 * vectors) as u64)?;
    let scratch = gpu.buffer(1 << 20)?;
    // (the router, F32 384 x 5120 whatever DS41_GEMV_KIND: "router" its GEMV alone, "routed" with its routes in the launch's last block,
    // sd_op_linear_route's square-root softplus, a bias and top 6, as the engine takes them)
    let (bias, ids, route_weights) = (gpu.buffer(384 * 4)?, gpu.buffer(4 * 6 * 4)?, gpu.buffer(4 * 6 * 4)?);
    gpu.write_buffer(&bias, 0, &random.floats(384))?;
    for (label, rows, cols) in [("q_b", 16384usize, 1280usize), ("wo_a", 4096, 4096), ("wo_b", 5120, 4096), ("router", 384, 5120), ("routed", 384, 5120)] {
        if only.as_ref().is_some_and(|text| !label.contains(text.as_str())) {
            continue;
        }
        let f32 = label.starts_with("rout");
        let (kind, per, block) = if f32 { (0u32, 1usize, 4usize) } else { (kind, per, block) };
        // (wo_a: the engine's four linears, group g's rows of the weights over its 4096 inputs of each row's 16384, into 4096 outputs a row)
        let run = |gpu: &mut Backend, bank: &Buffer| -> Result<(), String> {
            if label == "routed" {
                gpu.linear_route(bank, 0, rows as u64, cols as u64, input.base(), vectors as u64, output.base(), Some(bias.base()), 6, 2, true,
                                 2.5, ids.base(), route_weights.base(), &scratch)
            } else if label == "wo_a" {
                let items: Vec<Linear> = (0..4).map(|g| Linear { weights: *bank, kind, rows: 1024, cols: 4096, first_row: (g * 1024) as u64,
                                                               input: input.at(g * 4096), output: output.at(g * 1024) }).collect();
                gpu.linears(&items, vectors as u64, 16384, 4096, &scratch)
            } else {
                gpu.linear(bank, kind, rows as u64, cols as u64, 0, input.base(), cols as u64, vectors as u64, output.base(), rows as u64, &scratch)
            }
        };
        let bytes = rows * cols / per * block;
        let copies = ((384usize << 20) / bytes).max(2);
        let banks: Vec<Buffer> = (0..copies).map(|_| if f32 {
                let bank = gpu.buffer((rows * cols * 4) as u64)?;
                gpu.write_buffer(&bank, 0, &random.floats(rows * cols))?;
                Ok(bank)
            } else { gpu.upload_bytes(&random.blocks(kind, block, rows * cols / per)) })
            .collect::<Result<_, String>>()?;
        let started = std::time::Instant::now();
        for iteration in 0..iterations {
            let bank = &banks[iteration * 7 % copies];
            run(&mut gpu, bank)?;
        }
        let mut sink = [0f32; 1];
        gpu.read_buffer(&output, 0, &mut sink)?;
        let each = started.elapsed().as_secs_f64() / iterations as f64 * 1e6;
        println!("{label:8} kind {kind}, {vectors} rows, {rows} x {cols}: {:.2} MB, {each:.1} us a launch, {:.0} GB/s", bytes as f64 / 1e6, bytes as f64 / each / 1e3);
        // (DS41_GEMV_DUMP: the first bank's output, to compare kernels)
        if let Some(path) = std::env::var_os("DS41_GEMV_DUMP") {
            run(&mut gpu, &banks[0])?;
            let mut values = vec![0f32; rows * vectors];
            gpu.read_buffer(&output, 0, &mut values)?;
            let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
            let mut path = std::path::PathBuf::from(path);
            path.as_mut_os_string().push(format!(".{label}"));
            std::fs::write(&path, bytes).map_err(|error| error.to_string())?;
        }
        for bank in banks {
            gpu.release_buffer(bank)?;
        }
    }
    Ok(())
}
