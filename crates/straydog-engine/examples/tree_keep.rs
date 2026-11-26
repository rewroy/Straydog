//! Times keeping each verity tree's leaf level whole (Verifier::keep_level0), as a partially resident model's first layer-order pass
//! does before its first bank read. Usage: tree_keep MODEL_DIR QUANT [DATA_DIR]
use std::time::Instant;
use straydog_engine::{bundle, verity};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (root, quant) = (std::path::PathBuf::from(&args[1]), args[2].as_str());
    let data = args.get(3).map(std::path::PathBuf::from).unwrap_or_else(|| bundle::default_data_dir(&root));
    let id = bundle::Id::find(Some(bundle::MODEL), None, quant)?;
    let mut total = (0u64, 0f64);
    for (path, asset, digest) in bundle::verity_files_in(&root, id)? {
        let tree = bundle::tree_path(&data, &asset);
        if !tree.is_file() || !path.is_file() { continue; }
        let opened = Instant::now();
        let verifier = verity::Verifier::open(&tree, asset.bytes, &verity::parse_digest(&digest)?, 32 << 20)?;
        let open = opened.elapsed().as_secs_f64();
        let began = Instant::now();
        let kept = verifier.keep_level0(u64::MAX)?;
        let took = began.elapsed().as_secs_f64();
        total = (total.0 + kept, total.1 + took);
        println!("{}: open {:.3} s, leaf level {:.0} MiB kept in {:.3} s ({:.2} GB/s)", asset.name, open, kept as f64 / (1u64 << 20) as f64, took,
                 kept as f64 / 1e9 / took.max(1e-9));
    }
    println!("all: {:.0} MiB in {:.3} s", total.0 as f64 / (1u64 << 20) as f64, total.1);
    Ok(())
}
