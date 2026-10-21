//! F32 products over prompt batches (sd_op_linear's GEMM) give the same bits in every process.
//! Child processes compute them at DeepSeek-V4.1's prefill shapes and print digests of the
//! outputs; every child must print the same. hipBLASLt ranked F32 GEMM kernels differently in
//! some processes on RDNA4 (one in five to thirty), so one prompt prefilled in two runs gave
//! different logits; F32 products now run on rocBLAS's own kernel library whatever rocBLAS is
//! told to prefer (gpu/kernels.cpp, create_f32_blas). Every other child runs with
//! ROCBLAS_USE_HIPBLASLT=0, so products that went through hipBLASLt again would differ between
//! the two halves every time, not only when a process happens to rank kernels differently.
//!
//! cargo test --release -p straydog-engine --features hip --test gemm_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use std::process::Command;

use straydog_engine::gpu::Backend;

const CHILD: &str = "STRAYDOG_GEMM_CHILD";
const PROCESSES: usize = 16;
/// (rows, columns, inputs): routed experts' gate and up at a few routed rows, their down, and
/// attention projections over a 64-token batch.
const SHAPES: [(usize, usize, usize); 8] =
    [(2304, 5120, 5), (2304, 5120, 9), (2304, 5120, 16), (5120, 2304, 7), (512, 5120, 64), (1024, 4096, 64), (384, 5120, 64),
     (5120, 8192, 64)];

/// Every shape's product of generated F32 weights and inputs, as FNV-1a digests of its bits.
fn digests() -> Result<String, String> {
    let mut gpu = Backend::new(0, 2 << 30)?;
    let scratch = gpu.buffer(512 << 20)?;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut random = |count: usize| -> Vec<f32> {
        (0..count).map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }).collect()
    };
    let mut line = Vec::new();
    for (rows, cols, steps) in SHAPES {
        let bytes: Vec<u8> = random(rows * cols).iter().flat_map(|x| x.to_le_bytes()).collect();
        let weights = gpu.upload_bytes(&bytes)?;
        let input = gpu.buffer((steps * cols * 4) as u64)?;
        gpu.write_buffer(&input, 0, &random(steps * cols))?;
        let output = gpu.buffer((steps * rows * 4) as u64)?;
        gpu.linear(&weights, 0, rows as u64, cols as u64, 0, input.base(), 0, steps as u64, output.base(), 0, &scratch)?;
        let mut values = vec![0f32; steps * rows];
        gpu.read_buffer(&output, 0, &mut values)?;
        let hash = values.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, value| (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3));
        line.push(format!("{rows}x{cols}x{steps}:{hash:016x}"));
        for buffer in [weights, input, output] {
            gpu.release_buffer(buffer)?;
        }
    }
    Ok(line.join(" "))
}

#[test]
fn f32_products_are_the_same_in_every_process() {
    if std::env::var_os(CHILD).is_some() {
        println!("digests {}", digests().unwrap());
        return;
    }
    let mut seen: Vec<String> = Vec::new();
    for child in 0..PROCESSES {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "f32_products_are_the_same_in_every_process", "--nocapture"]).env(CHILD, "1");
        if child % 2 == 1 {
            command.env("ROCBLAS_USE_HIPBLASLT", "0");
        } else {
            command.env_remove("ROCBLAS_USE_HIPBLASLT");
        }
        let output = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "a child failed: {stdout}{}", String::from_utf8_lossy(&output.stderr));
        seen.push(stdout.lines().find_map(|line| line.strip_prefix("digests ")).expect("a child printed no digests").to_string());
    }
    let first = &seen[0];
    let differing = seen.iter().filter(|line| *line != first).count();
    assert_eq!(differing, 0, "{differing} of {PROCESSES} processes computed other bits:\n{}", seen.join("\n"));
}
