//! Load a bundle on the given GPUs and print how the model was placed (a diagnostic).
//!
//! cargo run --example placement --features hip -- MODEL QUANT DEVICES VRAM_GIB CONTEXT [DRAFTER]

use std::path::PathBuf;

use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::weights::Weights;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, quant, devices, vram, context, rest @ ..] = &args[..] else {
        return Err("usage: placement MODEL QUANT DEVICES VRAM_GIB CONTEXT [DRAFTER]".into());
    };
    let devices: Vec<usize> = devices.split(',').map(|device| device.parse().unwrap()).collect();
    let budget = (vram.parse::<f64>().unwrap() * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&devices, &vec![budget; devices.len()], SplitMode::Experts)?;
    group.configure_gemm(2, true)?;
    let drafter = rest.first().map(PathBuf::from);
    let weights = Weights::open(&PathBuf::from(model), straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, quant)?, drafter.as_deref(),
                                straydog_engine::host::host_memory_bytes(), &|_, _, _| {})?;
    let settings = Settings { context: context.parse().unwrap(), state_budget_bytes: 14 << 30, prefill_batch_size: 4096,
                              speculative: if drafter.is_some() { 3 } else { 0 }, cpu_threads: 0,
                              row_cache_bytes: Some(straydog_engine::rows::DEFAULT_CACHE_BYTES) };
    let model = Model::new(weights, group, &settings, &|_, _, _| {})?;
    let info = model.info()?;
    println!("resident {} partial {} host_kv {} batch {} speculative {}", model.resident, model.partial, model.host_kv,
             model.prefill_batch_size, model.speculative);
    for key in ["allocated_bytes", "budget_bytes", "cached_experts", "gpu_experts", "cpu_experts"] {
        println!("{key} {}", info[key]);
    }
    Ok(())
}
