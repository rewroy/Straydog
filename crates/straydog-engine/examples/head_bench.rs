//! The output head's GEMV (Qwen3.8: 248320 rows of 2560, Q6_K) streamed from VRAM: the time of a launch in a recorded graph and the
//! bandwidth it reaches, for one to four vectors (a decode step's, a verification's rows), and a digest of the bits of every vector's
//! rows (two builds that print the same digest formed the same logits).
//!
//! cargo run --release --features cuda -p straydog-engine --example head_bench [-- f32], under the GPU lock (f32: the F32 kernels, not the
//! Q8 ones).

use straydog_engine::gpu::Backend;
use straydog_engine::ops::Linear;

const ROWS: usize = 248_320;
const COLS: usize = 2560;
const LAUNCHES: usize = 20;
const REPLAYS: usize = 10;

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn main() -> Result<(), String> {
    let mut gpu = Backend::new(0, 2 << 30)?;
    if std::env::args().nth(1).as_deref() != Some("f32") {
        gpu.wide_q8_decode()?;
    }
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    // Q6_K blocks of random codes with a small F16 scale, so the dots stay in range
    let mut bytes: Vec<u8> = (0..ROWS * COLS / 256 * 210).map(|_| random.next() as u8).collect();
    for block in bytes.chunks_exact_mut(210) {
        block[208..210].copy_from_slice(&0x2c00u16.to_le_bytes());
    }
    let matrix = gpu.upload_bytes(&bytes)?;
    let gigabytes = bytes.len() as f64 / 1e9;
    drop(bytes);
    let x: Vec<f32> = (0..8 * COLS).map(|_| ((random.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0).collect();
    let input = gpu.buffer((x.len() * 4) as u64)?;
    gpu.write_buffer(&input, 0, &x)?;
    let output = gpu.buffer((8 * ROWS * 4) as u64)?;
    let scratch = gpu.buffer(64 << 20)?;
    for vectors in [1usize, 2, 3, 4] {
        let item = Linear { weights: matrix, kind: 14, rows: ROWS as u64, cols: COLS as u64, first_row: 0, input: input.base(),
                            output: output.base() };
        let launch = |gpu: &mut Backend| gpu.linears(&[item], vectors as u64, COLS as u64, ROWS as u64, &scratch);
        let key = 900 + vectors as u64;
        gpu.graph_begin()?;
        let mut recorded = Ok(());
        for _ in 0..LAUNCHES {
            recorded = recorded.and_then(|()| launch(&mut gpu));
        }
        gpu.graph_end(key, recorded.is_ok())?;
        recorded?;
        let mut sink = [0f32; 1];
        gpu.graph_run(key)?;
        gpu.read_buffer(&output, 0, &mut sink)?;
        let mut times = Vec::new();
        for _ in 0..REPLAYS {
            let started = std::time::Instant::now();
            gpu.graph_run(key)?;
            gpu.read_buffer(&output, 0, &mut sink)?;
            times.push(started.elapsed().as_secs_f64() / LAUNCHES as f64 * 1e6);
        }
        times.sort_by(f64::total_cmp);
        let each = times[times.len() / 2];
        let mut logits = vec![0f32; vectors * ROWS];
        gpu.read_buffer(&output, 0, &mut logits)?;
        let digest = logits.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, value| (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3));
        println!("{vectors} vectors: {each:7.1} us a launch (best {:.1}), {:.0} GB/s of the weights, digest {digest:016x}", times[0],
                 gigabytes / (each * 1e-6));
    }
    Ok(())
}
