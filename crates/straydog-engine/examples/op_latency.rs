//! Time per launch of the small ops of a Qwen3.5 MoE decode step, in recorded graphs (as a
//! decode step runs them): each op recorded 200 times in a row, the graph replayed, the time per
//! launch reported beside a bare add's. Development: where a decode step's small kernels spend.
//!
//! cargo run --release --features hip -p straydog-engine --example op_latency, under the GPU lock.

use straydog_engine::gpu::Backend;
use straydog_engine::ops::{Buffer, Linear};

const WIDTH: usize = 2048;
const EXPERTS: usize = 256;
const LAUNCHES: usize = 200;
const REPLAYS: usize = 20;

fn time(gpu: &mut Backend, key: u64, label: &str, sink: &Buffer, record: &mut dyn FnMut(&mut Backend) -> Result<(), String>)
        -> Result<f64, String> {
    gpu.graph_begin()?;
    let mut recorded = Ok(());
    for _ in 0..LAUNCHES {
        recorded = recorded.and_then(|()| record(gpu));
    }
    gpu.graph_end(key, recorded.is_ok())?;
    recorded?;
    let mut value = [0f32; 1];
    gpu.graph_run(key)?;
    gpu.read_buffer(sink, 0, &mut value)?;
    let started = std::time::Instant::now();
    for _ in 0..REPLAYS {
        gpu.graph_run(key)?;
    }
    gpu.read_buffer(sink, 0, &mut value)?;
    let each = started.elapsed().as_secs_f64() / (REPLAYS * LAUNCHES) as f64 * 1e6;
    println!("{label:44} {each:7.2} us a launch");
    Ok(each)
}

fn main() -> Result<(), String> {
    let mut gpu = Backend::new(0, 4 << 30)?;
    let floats = |gpu: &mut Backend, count: usize| gpu.buffer((count * 4) as u64);
    let (x, y, out, normed) = (floats(&mut gpu, WIDTH)?, floats(&mut gpu, WIDTH)?, floats(&mut gpu, 8 * WIDTH)?, floats(&mut gpu, WIDTH)?);
    let values: Vec<f32> = (0..WIDTH).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
    gpu.write_buffer(&x, 0, &values)?;
    gpu.write_buffer(&y, 0, &values)?;
    let weight = floats(&mut gpu, WIDTH)?;
    gpu.write_buffer(&weight, 0, &vec![1.0f32; WIDTH])?;
    let logits = floats(&mut gpu, EXPERTS)?;
    gpu.write_buffer(&logits, 0, &values[..EXPERTS])?;
    let (ids, weights) = (floats(&mut gpu, 8)?, floats(&mut gpu, 8)?);
    // F32 router (256 x 2048) and Q8_0 matrices (zero blocks: their bytes cost the same)
    let router = gpu.upload_bytes(&vec![0u8; EXPERTS * WIDTH * 4])?;
    let q8 = |gpu: &mut Backend, rows: usize, cols: usize| gpu.upload_bytes(&vec![0u8; rows * cols / 32 * 34]);
    let (qkv, down) = (q8(&mut gpu, 4096, WIDTH)?, q8(&mut gpu, WIDTH, 256)?);
    let scratch = gpu.buffer(64 << 20)?;
    let mut key = 100;
    let mut next = || { key += 1; key };
    let base = time(&mut gpu, next(), "add (2048)", &out, &mut |gpu| gpu.add(x.base(), WIDTH as u64, out.base()))?;
    time(&mut gpu, next(), "scale_rows (1 x 2048)", &out, &mut |gpu| gpu.scale_rows(out.base(), x.base(), 1, WIDTH as u64))?;
    time(&mut gpu, next(), "rms_norm (1 x 2048)", &out, &mut |gpu| gpu.rms_norm(x.base(), &weight, 1, WIDTH as u64, 1e-6, out.base()))?;
    time(&mut gpu, next(), "route (256 experts, top 8, softmax)", &ids, &mut |gpu| {
        gpu.route(logits.base(), None, 1, EXPERTS as u64, 8, 0, true, 1.0, ids.base(), weights.base())
    })?;
    time(&mut gpu, next(), "route (255 experts: the block-wide kernel)", &ids, &mut |gpu| {
        gpu.route(logits.base(), None, 1, 255, 8, 0, true, 1.0, ids.base(), weights.base())
    })?;
    time(&mut gpu, next(), "swiglu (256)", &out, &mut |gpu| gpu.swiglu(x.base(), y.base(), 1, 256, 0.0, None, out.base()))?;
    let linear = |weights: Buffer, kind: u32, rows: usize, cols: usize, input: &Buffer, output: &Buffer| Linear {
        weights, kind, rows: rows as u64, cols: cols as u64, first_row: 0, input: input.base(), output: output.base(),
    };
    time(&mut gpu, next(), "F32 GEMV router (256 x 2048)", &out, &mut |gpu| {
        gpu.linears(&[linear(router, 0, EXPERTS, WIDTH, &normed, &out)], 1, 0, 0, &scratch)
    })?;
    time(&mut gpu, next(), "Q8_0 GEMV (4096 x 2048)", &out, &mut |gpu| {
        gpu.linears(&[linear(qkv, 8, 4096, WIDTH, &normed, &out)], 1, 0, 0, &scratch)
    })?;
    time(&mut gpu, next(), "norm + Q8_0 GEMV (4096 x 2048), fused", &out, &mut |gpu| {
        gpu.norm_linears(x.base(), &weight, WIDTH as u64, 1e-6, normed.base(), &[linear(qkv, 8, 4096, WIDTH, &normed, &out)])
    })?;
    time(&mut gpu, next(), "Q8_0 GEMV (2048 x 256)", &out, &mut |gpu| {
        gpu.linears(&[linear(down, 8, WIDTH, 256, &normed, &out)], 1, 0, 0, &scratch)
    })?;
    time(&mut gpu, next(), "swiglu + Q8_0 GEMV (2048 x 256) + gate, fused", &out, &mut |gpu| {
        gpu.swiglu_linear(&down, 8, WIDTH as u64, 256, x.base(), y.base(), Some(x.base()), normed.base(), out.base())
    })?;
    // Weights streamed from VRAM (a bank larger than the caches, a matrix a launch): the GEMV with
    // its input read from global memory against the prologue kernel's (input in shared memory).
    for (kind, label, row_bytes) in [(8u32, "Q8_0", WIDTH / 32 * 34), (30, "BF16", WIDTH * 2)] {
        let rows = 4096;
        let count = (320 << 20) / (rows * row_bytes);
        let bank: Vec<Buffer> = (0..count).map(|_| gpu.upload_bytes(&vec![0u8; rows * row_bytes])).collect::<Result<_, _>>()?;
        let gb = (rows * row_bytes) as f64 / 1e9;
        let mut at = 0;
        let each = time(&mut gpu, next(), &format!("{label} GEMV 4096 x 2048, from VRAM"), &out, &mut |gpu| {
            at = (at + 1) % count;
            gpu.linears(&[linear(bank[at], kind, rows, WIDTH, &normed, &out)], 1, 0, 0, &scratch)
        })?;
        println!("    {:.0} GB/s", gb / (each * 1e-6));
        let each = time(&mut gpu, next(), &format!("{label} GEMV 4096 x 2048, from VRAM, input in shared"), &out, &mut |gpu| {
            at = (at + 1) % count;
            gpu.swiglu_linear(&bank[at], kind, rows as u64, WIDTH as u64, x.base(), y.base(), None, normed.base(), out.base())
        })?;
        println!("    {:.0} GB/s", gb / (each * 1e-6));
        time(&mut gpu, next(), &format!("rms_norm, then {label} GEMV 4096 x 2048, from VRAM"), &out, &mut |gpu| {
            at = (at + 1) % count;
            gpu.rms_norm(x.base(), &weight, 1, WIDTH as u64, 1e-6, normed.base())?;
            gpu.linears(&[linear(bank[at], kind, rows, WIDTH, &normed, &out)], 1, 0, 0, &scratch)
        })?;
        time(&mut gpu, next(), &format!("norm + {label} GEMV 4096 x 2048, from VRAM, fused"), &out, &mut |gpu| {
            at = (at + 1) % count;
            gpu.norm_linears(x.base(), &weight, WIDTH as u64, 1e-6, normed.base(), &[linear(bank[at], kind, rows, WIDTH, &normed, &out)])
        })?;
    }
    println!("(a bare add: {base:.2} us)");
    Ok(())
}
