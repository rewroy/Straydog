//! Pin what each bundle takes of the machine: read the GGUF headers of the pinned shards found
//! under the given directories, sum their tensors by how the engine places them (the routed
//! experts, the gathered tables read from the drive by row, and the rest, which every token reads),
//! add the bytes of sequence state a token of context takes (from the model's hyperparameters) and
//! print the table that `src/asset_footprint.json` holds. A bundle is looked for in each
//! directory and in each directory in it.
//!
//! cargo run --release -p straydog-engine --example asset_footprint -- DIRECTORY... > crates/straydog-engine/src/asset_footprint.json

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use serde_json::{json, Map};
use straydog_engine::arch::deepseek41;
use straydog_engine::bundle;
use straydog_engine::model::context_state_bytes;
use straydog_engine::pinned::gathered;
use straydog_gguf::ModelHeader;

/// The bytes of sequence state one more token of context takes, as the placement counts them: the
/// the compressed rows and index keys of the layers that
/// hold them (DeepSeek-V4.1, BF16), Qwen3.8's state (host memory).
fn state_per_token(architecture: &str, header: &ModelHeader) -> Result<u64, String> {
    match architecture {
        "deepseek41" => {
            let hp = deepseek41::Hparams::from_header(header)?;
            Ok(hp.layers.iter().enumerate()
                .filter_map(|(layer, plan)| plan.compressed.filter(|compressed| compressed.kv_source == layer))
                .map(|compressed| 2 * (hp.head_dim + hp.index_dim) as u64 / compressed.ratio as u64)
                .sum())
        }
        _ => {
            const CONTEXT: u64 = 262_144;
            Ok((context_state_bytes(2 * CONTEXT, true) - context_state_bytes(CONTEXT, true)) / CONTEXT)
        }
    }
}

fn main() -> Result<(), String> {
    let directories: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let mut bundles = Map::new();
    // (a model's hyperparameters are one: a bundle whose metadata does not give them all, as GSQ-RCO's
    // DeepSeek-V4.1 keeps Engram's in tensors, takes those of the model's bundle that does)
    let mut states: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for id in bundle::bundles() {
        // The shards, and a bundle read on demand's sidecars: the files that hold its tensors.
        let mut names: Vec<String> = bundle::shards(id)?.into_iter().map(|asset| asset.name).collect();
        if id.is_on_demand() {
            names.extend(bundle::on_demand(id)?.sidecars.into_iter().map(|sidecar| sidecar.asset.name));
        }
        let holds = |directory: &PathBuf| directory.join(&names[0]).is_file();
        let found = directories.iter().flat_map(|directory| {
            let inside = std::fs::read_dir(directory).into_iter().flatten().flatten().map(|entry| entry.path());
            std::iter::once(directory.clone()).chain(inside)
        });
        let Some(directory) = found.into_iter().find(holds) else {
            return Err(format!("{id}: {} not found", names[0]));
        };
        let (mut dense, mut experts, mut tables, mut first) = (0u64, 0u64, 0u64, None);
        for name in &names {
            let file = File::open(directory.join(name)).map_err(|exc| format!("{name}: {exc}"))?;
            let header = straydog_gguf::inspect(&mut BufReader::new(file)).map_err(|exc| format!("{name}: {exc}"))?;
            first.get_or_insert_with(|| header.clone());
            for tensor in &header.tensors {
                let sum = if tensor.name.ends_with("_exps.weight") {
                    &mut experts
                } else if gathered(&tensor.name) {
                    &mut tables
                } else {
                    &mut dense
                };
                *sum += tensor.bytes;
            }
        }
        let model = id.model.to_string();
        let state = match state_per_token(&bundle::architecture_of(id)?, &first.ok_or("no shards")?) {
            Ok(state) => *states.entry(model).or_insert(state),
            Err(error) => *states.get(&model).ok_or(error)?,
        };
        eprintln!("{id}: dense {dense}, experts {experts}, tables {tables}, state per token {state} ({})", directory.display());
        let of_model = bundles.entry(id.model).or_insert_with(|| json!({}));
        if of_model.get(id.provider).is_none() {
            of_model[id.provider] = json!({});
        }
        of_model[id.provider][id.quant] = json!({"dense": dense, "experts": experts, "tables": tables, "state_per_token": state});
    }
    // (by model, provider and quantization, the labels that name a bundle)
    let table = json!({"schema": 3, "bundles": bundles});
    println!("{}", serde_json::to_string_pretty(&table).map_err(|exc| exc.to_string())?);
    Ok(())
}
