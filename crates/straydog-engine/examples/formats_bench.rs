//! Throughput of the formats' expert kernels at DeepSeek-V4.1's expert shapes, on its real
//! experts: the GPU's decode GEMV (GB/s of expert weights) and RDNA4 grouped WMMA GEMM (TFLOPS),
//! and the CPU expert engine (GB/s on one core and on all).
//!
//! STRAYDOG_DEEPSEEK=<UD-IQ2_XXS directory> cargo run --release --features hip --example formats_bench
//! under the GPU lock. Eight real experts of a layer fill a bank of 64 (larger than the GPU's
//! last-level cache and the CPU's), and every call takes random experts of it.

use std::ffi::c_void;
use std::fs::File;
use std::path::PathBuf;

use straydog_gguf::ModelHeader;
use straydog_gpu::{ffi, Device};

const HIDDEN: usize = 5120;
const FFN: usize = 2304;
const BANK: usize = 64;
const TOP_K: usize = 6;
/// A role's format, bytes per expert and bank bytes.
type Role = (u32, usize, Vec<u8>);

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn routes(&mut self, steps: usize) -> Vec<i32> {
        (0..steps).flat_map(|_| {
            let mut chosen: Vec<i32> = Vec::new();
            while chosen.len() < TOP_K {
                let expert = (self.next() % BANK as u64) as i32;
                if !chosen.contains(&expert) {
                    chosen.push(expert);
                }
            }
            chosen
        }).collect()
    }
    fn activations(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0).collect()
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Bank bytes of a role: eight real experts of `name`, repeated to BANK experts.
fn bank(files: &[File], headers: &[ModelHeader], name: &str) -> Role {
    let (file, tensor) = headers.iter().enumerate()
        .find_map(|(file, header)| header.tensors.iter().find(|tensor| tensor.name == name).map(|tensor| (file, tensor)))
        .unwrap();
    let count = tensor.dimensions.get(2).copied().unwrap_or(1) as usize;
    let per = tensor.bytes as usize / count;
    let mut real = vec![0u8; per * 8];
    for i in 0..8 {
        let expert = (i * 47) % count;
        straydog_engine::storage::read_at(&files[file], tensor.offset + (expert * per) as u64, &mut real[i * per..(i + 1) * per]).unwrap();
    }
    (tensor.quant_type, per, (0..BANK).flat_map(|e| real[(e % 8) * per..(e % 8 + 1) * per].to_vec()).collect())
}

fn name(kind: u32) -> &'static str {
    match kind {
        8 => "Q8_0", 10 => "Q2_K", 13 => "Q5_K", 16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS", 19 => "IQ1_S",
        22 => "IQ2_S", 39 => "MXFP4", _ => "?",
    }
}

#[allow(clippy::too_many_arguments)]
fn run(device: &Device, pointers: &[*mut c_void; 3], kinds: &[i32; 3], steps: usize, selected: &[i32], path: i32,
       input: &[f32], output: &mut [f32]) -> [f64; 2] {
    let mut timings = [0f64; 2];
    // SAFETY: banks of BANK experts on this device; buffers sized for steps * TOP_K routes.
    let status = unsafe {
        ffi::sd_expert_rows(device.handle(), pointers.as_ptr(), kinds.as_ptr(), BANK as u64, HIDDEN as u64, FFN as u64,
                            steps as u64, TOP_K as u64, selected.as_ptr(), path, input.as_ptr(), std::ptr::null(),
                            std::ptr::null_mut(), std::ptr::null_mut(), output.as_mut_ptr(), timings.as_mut_ptr())
    };
    assert_eq!(status, 0, "{}", straydog_gpu::last_error().0);
    timings
}

fn main() {
    let root = PathBuf::from(std::env::var_os("STRAYDOG_DEEPSEEK").expect("set STRAYDOG_DEEPSEEK"));
    let mut paths: Vec<PathBuf> = std::fs::read_dir(root).unwrap().map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "gguf")).collect();
    paths.sort();
    let mut files = Vec::new();
    let mut headers = Vec::new();
    for path in paths {
        let mut file = File::open(path).unwrap();
        headers.push(straydog_gguf::inspect(&mut file).unwrap());
        files.push(file);
    }
    straydog_engine::gpu::load_iq_tables().unwrap();
    let device = Device::new(0, 8 << 30).unwrap();
    // SAFETY: plain values; null on failure.
    let engines = [1u32, 6].map(|threads| (threads, unsafe { ffi::sd_cpu_experts_create(threads) }));
    let mut random = Random(0x5eed);
    // Layers covering every expert format: gate/up, down.
    let mut cases: Vec<(String, [Role; 3])> = [0, 1, 2, 10, 20, 25].iter().map(|layer| {
        (format!("layer {layer}"), ["gate", "up", "down"].map(|role| bank(&files, &headers, &format!("blk.{layer}.ffn_{role}_exps.weight"))))
    }).collect();
    // MXFP4 (no file has it): generated blocks with realistic exponents.
    let mxfp4 = |values: usize, random: &mut Random| -> Role {
        let data: Vec<u8> = (0..BANK * values / 32).flat_map(|_| {
            let mut block: Vec<u8> = (0..17).map(|_| random.next() as u8).collect();
            block[0] = 118 + (random.next() % 8) as u8;
            block
        }).collect();
        (39, data.len() / BANK, data)
    };
    cases.push(("generated".into(), [mxfp4(FFN * HIDDEN, &mut random), mxfp4(FFN * HIDDEN, &mut random), mxfp4(HIDDEN * FFN, &mut random)]));
    println!("{:<10} {:>8} {:>8} | GPU decode GB/s  | GPU prefill TFLOPS gate/up, down at rows/expert 16, 64, 256 | CPU GB/s 1 core, 6 cores",
             "case", "gate/up", "down");
    for (label, roles) in &cases {
        let kinds = [roles[0].0 as i32, roles[1].0 as i32, roles[2].0 as i32];
        let allocations: Vec<_> = roles.iter().map(|role| device.upload(&role.2).unwrap()).collect();
        let pointers = [allocations[0].pointer(), allocations[1].pointer(), allocations[2].pointer()];
        let per = [roles[0].1, roles[1].1, roles[2].1];
        // Decode: one token, six experts.
        let mut times = [Vec::new(), Vec::new()];
        let input = random.activations(HIDDEN);
        let mut output = vec![0f32; TOP_K * HIDDEN];
        for iteration in 0..120 {
            let selected = random.routes(1);
            let t = run(&device, &pointers, &kinds, 1, &selected, 0, &input, &mut output);
            if iteration >= 20 {
                times[0].push(t[0]);
                times[1].push(t[1]);
            }
        }
        let decode = [(per[0] + per[1]) * TOP_K, per[2] * TOP_K];
        let gemv = [0, 1].map(|stage| decode[stage] as f64 / (median(times[stage].clone()) * 1e-3) / 1e9);
        // Prefill: tokens such that each expert of the bank takes about 16, 64 and 256 rows.
        let mut prefill = Vec::new();
        for rows in [16, 64, 256] {
            let steps = rows * BANK / TOP_K;
            let input = random.activations(steps * HIDDEN);
            let mut output = vec![0f32; steps * TOP_K * HIDDEN];
            let mut times = [Vec::new(), Vec::new()];
            for iteration in 0..7 {
                let selected = random.routes(steps);
                let t = run(&device, &pointers, &kinds, steps, &selected, 1, &input, &mut output);
                if iteration >= 2 {
                    times[0].push(t[0]);
                    times[1].push(t[1]);
                }
            }
            let flops = [4.0, 2.0].map(|f| f * (steps * TOP_K) as f64 * (HIDDEN * FFN) as f64);
            prefill.push([0, 1].map(|stage| flops[stage] / (median(times[stage].clone()) * 1e-3) / 1e12));
        }
        // CPU: six experts' gate/up and down, weights from host memory.
        let mut cpu = Vec::new();
        for (threads, engine) in engines {
            let x = random.activations(HIDDEN);
            let mut out = vec![0f32; HIDDEN];
            let weights = [1.0f32 / 6.0; TOP_K];
            let mut before = [0u64; 5];
            let calls = 24;
            for call in 0..calls + 2 {
                if call == 2 {
                    // SAFETY: five counters.
                    unsafe { ffi::sd_cpu_experts_stats(engine, before.as_mut_ptr()) };
                }
                let selected = random.routes(1);
                let slices = |role: usize| -> Vec<*const u8> {
                    selected.iter().map(|e| roles[role].2[*e as usize * per[role]..].as_ptr()).collect()
                };
                let (gate, up, down) = (slices(0), slices(1), slices(2));
                // SAFETY: TOP_K expert slices of the declared shapes.
                let status = unsafe {
                    ffi::sd_cpu_experts_run(engine, kinds.as_ptr(), gate.as_ptr(), up.as_ptr(), down.as_ptr(), weights.as_ptr(),
                                            TOP_K as u32, HIDDEN as u64, FFN as u64, x.as_ptr(), out.as_mut_ptr())
                };
                assert_eq!(status, 0);
            }
            let mut after = [0u64; 5];
            // SAFETY: five counters.
            unsafe { ffi::sd_cpu_experts_stats(engine, after.as_mut_ptr()) };
            let gate_up = (decode[0] * calls) as f64 / ((after[2] - before[2]) as f64);
            let down = (decode[1] * calls) as f64 / ((after[4] - before[4]) as f64);
            cpu.push((threads, gate_up, down));
        }
        println!("{label:<10} {:>8} {:>8} | {:>6.0} {:>6.0}    | {} | {}", name(roles[0].0), name(roles[2].0), gemv[0], gemv[1],
                 prefill.iter().map(|p| format!("{:.1}/{:.1}", p[0], p[1])).collect::<Vec<_>>().join("  "),
                 cpu.iter().map(|(t, g, d)| format!("{t}: {g:.1}/{d:.1}")).collect::<Vec<_>>().join("  "));
    }
    for (_, engine) in engines {
        // SAFETY: created above, destroyed once.
        unsafe { ffi::sd_cpu_experts_destroy(engine) };
    }
}
