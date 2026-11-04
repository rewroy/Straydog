//! Pin chunk digests: hash every pinned shard and the drafter found under the given
//! directories, check each against its pinned SHA-256, and print the chunk table that
//! `src/asset_chunks.json` holds.
//!
//! cargo run --release -p straydog-engine --example asset_chunks -- DIRECTORY... > crates/straydog-engine/src/asset_chunks.json

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use straydog_engine::bundle::{self, Asset};
use straydog_engine::digest;

fn find(directories: &[PathBuf], name: &str) -> Option<PathBuf> {
    directories.iter().map(|directory| directory.join(name)).find(|path| path.is_file())
}

fn main() -> Result<(), String> {
    let directories: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let mut assets: BTreeMap<String, (Asset, PathBuf)> = BTreeMap::new();
    for id in bundle::bundles().filter(|id| !id.is_on_demand()) {
        let pins = bundle::bundle(id)?;
        // (and the engram tables the bundle may read instead of its last shard)
        let tables = bundle::tables(id)?.into_iter().filter(|table| !table.bundled).map(|table| table.asset);
        for asset in pins.shards.into_iter().chain([pins.drafter]).chain(tables) {
            let Some(path) = find(&directories, &asset.name) else {
                eprintln!("not found: {}", asset.name);
                continue;
            };
            assets.entry(asset.sha256.clone()).or_insert((asset, path));
        }
    }
    let results: Vec<Result<(String, Vec<String>), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = assets
            .values()
            .map(|(asset, path)| scope.spawn(move || digests(asset, path)))
            .collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    });
    let mut files = serde_json::Map::new();
    for result in results {
        let (sha256, pieces) = result?;
        files.insert(sha256, pieces.into());
    }
    let table = serde_json::json!({"chunk_bytes": digest::CHUNK_BYTES, "files": files});
    println!("{}", serde_json::to_string_pretty(&table).unwrap());
    Ok(())
}

fn digests(asset: &Asset, path: &Path) -> Result<(String, Vec<String>), String> {
    let file = File::open(path).map_err(|exc| format!("{}: {exc}", path.display()))?;
    let (whole, pieces) = digest::file_digests(&file, &|_| {}).map_err(|exc| exc.to_string())?;
    if whole != asset.sha256 || file.metadata().map_err(|exc| exc.to_string())?.len() != asset.bytes {
        return Err(format!("{} does not match its pin", path.display()));
    }
    eprintln!("{}: {} chunks", asset.name, pieces.len());
    Ok((whole, pieces))
}
