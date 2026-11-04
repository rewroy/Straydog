//! Pin fs-verity digests: check every pinned shard found under the given directories against its
//! pinned SHA-256, build its verity tree into DATA_DIR (`<DATA_DIR>/verity/<name>.verity`, as
//! the server reads them) and print the table that `src/asset_verity.json` holds.
//!
//! cargo run --release -p straydog-engine --example asset_verity -- DATA_DIR DIRECTORY... > crates/straydog-engine/src/asset_verity.json

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use straydog_engine::bundle::{self, Asset};

fn find(directories: &[PathBuf], name: &str) -> Option<PathBuf> {
    directories.iter().map(|directory| directory.join(name)).find(|path| path.is_file())
}

fn main() -> Result<(), String> {
    let args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let [data_dir, directories @ ..] = &args[..] else {
        return Err("usage: asset_verity DATA_DIR DIRECTORY...".into());
    };
    std::fs::create_dir_all(data_dir.join("verity")).map_err(|exc| exc.to_string())?;
    // Per whole-file SHA-256: the assets pinned with it (a file shared by bundles has a name in each).
    let mut assets: BTreeMap<String, (Vec<Asset>, PathBuf)> = BTreeMap::new();
    for id in bundle::bundles().filter(|id| !id.is_on_demand()) {
        for asset in bundle::bundle(id)?.shards {
            let Some(path) = find(directories, &asset.name) else {
                eprintln!("not found: {}", asset.name);
                continue;
            };
            assets.entry(asset.sha256.clone()).or_insert_with(|| (Vec::new(), path)).0.push(asset);
        }
    }
    let results: Vec<Result<(String, String), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = assets.values().map(|(names, path)| scope.spawn(move || pin(data_dir, names, path))).collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    });
    let mut files = serde_json::Map::new();
    for result in results {
        let (sha256, verity) = result?;
        files.insert(sha256, verity.into());
    }
    let table = serde_json::json!({"block_bytes": straydog_engine::verity::BLOCK, "files": files});
    println!("{}", serde_json::to_string_pretty(&table).unwrap());
    Ok(())
}

/// Check the file against its pin, build its tree under the first asset's name (linked under the
/// others') and return (SHA-256, verity digest).
fn pin(data_dir: &Path, names: &[Asset], path: &Path) -> Result<(String, String), String> {
    let asset = &names[0];
    let file = File::open(path).map_err(|exc| format!("{}: {exc}", path.display()))?;
    let started = std::time::Instant::now();
    let whole = straydog_engine::digest::file_sha256(&file, &|_| {}).map_err(|exc| exc.to_string())?;
    if whole != asset.sha256 || file.metadata().map_err(|exc| exc.to_string())?.len() != asset.bytes {
        return Err(format!("{} does not match its pin", path.display()));
    }
    let tree = bundle::tree_path(data_dir, asset);
    let digest = straydog_engine::verity::build(&file, &tree, &|_| {}).map_err(|exc| exc.to_string())?;
    for other in &names[1..] {
        let link = bundle::tree_path(data_dir, other);
        let _ = std::fs::remove_file(&link);
        std::fs::hard_link(&tree, &link).map_err(|exc| format!("{}: {exc}", link.display()))?;
    }
    let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    eprintln!("{}: {digest} ({:.0} s)", asset.name, started.elapsed().as_secs_f64());
    Ok((whole, digest))
}
