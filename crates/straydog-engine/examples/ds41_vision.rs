//! DeepSeek-V4.1's vision encoder alone on device 0: an image's span rows, from the encoder's
//! file beside a model (authenticated by its tree there), written as little-endian F32 for a
//! reference to compare; the encoder's time, its weights uploaded each run, on stderr.
//!
//! cargo run --release -p straydog-engine --features hip --example ds41_vision -- MODEL_DIR IMAGE OUT

use std::path::Path;
use std::time::Instant;

use straydog_engine::arch::deepseek41::engine::encode_image_rows;
use straydog_engine::bundle::{self, OnDemand};
use straydog_engine::gpu::Backend;
use straydog_engine::pinned::Pinned;
use straydog_engine::vision::{self, VisionConfig};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, image, out] = &args[..] else {
        return Err("usage: ds41_vision MODEL_DIR IMAGE OUT".into());
    };
    let root = Path::new(model);
    let vision = bundle::vision_source(bundle::Id::find(Some(bundle::DEEPSEEK), None, "IQ2_XS")?)?.ok_or("no vision encoder pinned")?;
    let pins = OnDemand {
        model: bundle::DEEPSEEK.into(), architecture: bundle::VISION_ARCHITECTURE.into(), quantization: "vision".into(),
        shards: vec![vision.asset], verity: vec![vision.verity], resources: Vec::new(), sidecars: Vec::new(),
    };
    let files = Pinned::open(root, &bundle::default_data_dir(root), &pins)?;
    let bytes = std::fs::read(image).map_err(|exc| exc.to_string())?;
    let patches = vision::patches(&vision::decode(&bytes)?, &VisionConfig::DEEPSEEK41);
    eprintln!("{} patches ({}x{}), {} positions ({}x{} rows)", patches.rows * patches.columns, patches.rows, patches.columns,
              patches.plan.tokens(), patches.plan.rows, patches.plan.columns);
    let mut gpu = Backend::new(0, 6 << 30)?;
    let arena = gpu.buffer(4 << 30)?;
    let mut rows = Vec::new();
    for run in 0..3 {
        let started = Instant::now();
        rows = encode_image_rows(&mut gpu, arena, &files, 5120, std::slice::from_ref(&patches), |_, _| Ok(()))?;
        eprintln!("run {run}: {:.3} s", started.elapsed().as_secs_f64());
    }
    let values: Vec<u8> = rows[0].iter().flat_map(|value| value.to_le_bytes()).collect();
    std::fs::write(out, values).map_err(|exc| exc.to_string())
}
