//! The CPU expert engine's throughput on a Qwen3.5 MoE GGUF's real experts: a token's top-k
//! experts of a layer (random ones of the layer each call) run from host memory, on 1 and on all
//! cores; GB/s of expert weights. Development only (unauthenticated weights).
//!
//! q35_cpu_bench MODEL.gguf LAYER [THREADS] [EXPERTS a call, default the model's top-k]

use std::fs::File;
use std::os::unix::fs::FileExt;

use straydog_gpu::ffi;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path, layer, rest @ ..] = &args[..] else { return Err("usage: q35_cpu_bench MODEL.gguf LAYER [THREADS] [EXPERTS]".into()) };
    let threads: u32 = rest.first().map_or(Ok(6), |value| value.parse()).map_err(|_| "THREADS must be a number")?;
    let per_call: Option<usize> = rest.get(1).map(|value| value.parse()).transpose().map_err(|_| "EXPERTS must be a number")?;
    let mut file = File::open(path).map_err(|exc| exc.to_string())?;
    let header = straydog_gguf::inspect(&mut file).map_err(|exc| exc.to_string())?;
    let hp = straydog_engine::arch::qwen35moe::Hparams::from_header(&header)?;
    let (hidden, ffn, experts) = (hp.width, hp.expert_ff, hp.experts);
    let top_k = per_call.unwrap_or(hp.experts_used).clamp(1, experts);
    // Each role's whole bank (every expert) of the layer, read into memory.
    let mut roles = Vec::new();
    for role in ["gate", "up", "down"] {
        let name = format!("blk.{layer}.ffn_{role}_exps.weight");
        let info = header.tensors.iter().find(|tensor| tensor.name == name).ok_or(format!("no {name}"))?;
        let mut bank = vec![0u8; info.bytes as usize];
        file.read_exact_at(&mut bank, info.offset).map_err(|exc| exc.to_string())?;
        roles.push((info.quant_type as i32, bank.len() / experts, bank));
    }
    let kinds = [roles[0].0, roles[1].0, roles[2].0];
    straydog_engine::gpu::load_iq_tables()?;
    let mut seed = 0x5eedu64;
    let mut next = move || { seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27; seed.wrapping_mul(0x2545_f491_4f6c_dd1d) };
    let x: Vec<f32> = (0..hidden).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    let bytes_per_call = (roles[0].1 + roles[1].1 + roles[2].1) * top_k;
    for count in [1, threads] {
        // SAFETY: plain value; null on failure.
        let engine = unsafe { ffi::sd_cpu_experts_create(count) };
        if engine.is_null() {
            return Err("no CPU expert engine".into());
        }
        let mut out = vec![0f32; hidden];
        let weights = vec![1.0f32 / top_k as f32; top_k];
        let calls = 40;
        let mut started = std::time::Instant::now();
        for call in 0..calls + 4 {
            if call == 4 {
                started = std::time::Instant::now();
            }
            let mut chosen: Vec<usize> = Vec::new();
            while chosen.len() < top_k {
                let expert = (next() % experts as u64) as usize;
                if !chosen.contains(&expert) {
                    chosen.push(expert);
                }
            }
            let slices = |role: usize| -> Vec<*const u8> { chosen.iter().map(|&e| roles[role].2[e * roles[role].1..].as_ptr()).collect() };
            let (gate, up, down) = (slices(0), slices(1), slices(2));
            // SAFETY: top_k expert slices of the declared shapes.
            let status = unsafe {
                ffi::sd_cpu_experts_run(engine, kinds.as_ptr(), gate.as_ptr(), up.as_ptr(), down.as_ptr(), weights.as_ptr(), top_k as u32,
                                        hidden as u64, ffn as u64, x.as_ptr(), out.as_mut_ptr())
            };
            if status != 0 {
                return Err(straydog_engine::gpu::cpu_error());
            }
        }
        let seconds = started.elapsed().as_secs_f64() / calls as f64;
        println!("layer {layer} ({:?}), {count} thread(s): {top_k} experts in {:.2} ms, {:.1} GB/s, {:.3} ms an expert", kinds,
                 seconds * 1e3, bytes_per_call as f64 / seconds / 1e9, seconds * 1e3 / top_k as f64);
        // SAFETY: created above, destroyed once.
        unsafe { ffi::sd_cpu_experts_destroy(engine) };
    }
    Ok(())
}
