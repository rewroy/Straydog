//! Development: DeepSeek-V4.1's decode Q5_K projections at a GPU's two-GPU shapes, on one GPU, their
//! weights streamed from banks larger than the caches: q_b, wo_a (its four groups as one matrix) and
//! wo_b. Prints the wall time a launch as a bound; run under rocprofv3 --kernel-trace for kernel time.
//!
//! cargo run --release --features hip -p straydog-engine --example ds41_gemv [ITERATIONS], under the GPU lock (DS41_Q8=0: the F32
//! kernels; DS41_VECTORS=N: a pass of N rows (1 to 4); DS41_GEMV_ONLY=TEXT: only the shapes whose label holds TEXT; DS41_GEMV_DUMP=PATH:
//! each shape's output to PATH.LABEL, to compare builds). "wo_a" is the engine's: four groups, each its own slice of the input.

use straydog_engine::gpu::Backend;
use straydog_engine::ops::{Buffer, Linear};

const Q5_K: u32 = 13;
const BLOCK: usize = 176;

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Random Q5_K blocks whose d and dmin are 2^-4 and 2^-6.
    fn blocks(&mut self, count: usize) -> Vec<u8> {
        let mut out: Vec<u8> = (0..count * BLOCK).map(|_| self.next() as u8).collect();
        for chunk in out.chunks_exact_mut(BLOCK) {
            chunk[..2].copy_from_slice(&0x2c00u16.to_le_bytes());
            chunk[2..4].copy_from_slice(&0x2400u16.to_le_bytes());
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
    let input_values = random.floats(16384 * vectors);
    let input = gpu.buffer((input_values.len() * 4) as u64)?;
    gpu.write_buffer(&input, 0, &input_values)?;
    let output = gpu.buffer((16384 * 4 * vectors) as u64)?;
    let scratch = gpu.buffer(1 << 20)?;
    for (label, rows, cols) in [("q_b", 16384usize, 1280usize), ("wo_a", 4096, 4096), ("wo_b", 5120, 4096)] {
        if only.as_ref().is_some_and(|text| !label.contains(text.as_str())) {
            continue;
        }
        // (wo_a: the engine's four linears, group g's rows of the weights over its 4096 inputs of each row's 16384, into 4096 outputs a row)
        let run = |gpu: &mut Backend, bank: &Buffer| -> Result<(), String> {
            if label == "wo_a" {
                let items: Vec<Linear> = (0..4).map(|g| Linear { weights: *bank, kind: Q5_K, rows: 1024, cols: 4096, first_row: (g * 1024) as u64,
                                                               input: input.at(g * 4096), output: output.at(g * 1024) }).collect();
                gpu.linears(&items, vectors as u64, 16384, 4096, &scratch)
            } else {
                gpu.linear(bank, Q5_K, rows as u64, cols as u64, 0, input.base(), cols as u64, vectors as u64, output.base(), rows as u64, &scratch)
            }
        };
        let bytes = rows * cols / 256 * BLOCK;
        let copies = ((384usize << 20) / bytes).max(2);
        let banks: Vec<Buffer> = (0..copies).map(|_| gpu.upload_bytes(&random.blocks(rows * cols / 256))).collect::<Result<_, _>>()?;
        let started = std::time::Instant::now();
        for iteration in 0..iterations {
            let bank = &banks[iteration * 7 % copies];
            run(&mut gpu, bank)?;
        }
        let mut sink = [0f32; 1];
        gpu.read_buffer(&output, 0, &mut sink)?;
        let each = started.elapsed().as_secs_f64() / iterations as f64 * 1e6;
        println!("{label:8} {vectors} rows, {rows} x {cols}: {:.2} MB, {each:.1} us a launch, {:.0} GB/s", bytes as f64 / 1e6, bytes as f64 / each / 1e3);
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
