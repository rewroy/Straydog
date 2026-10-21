//! The pinned model bundles: every shard, resource and the drafter by size and SHA-256.

use serde_json::Value;

/// The manifest, compiled in: a binary authenticates exactly the bundles it was tested with.
pub const MANIFEST: &str = include_str!("assets.json");
pub const QUANTIZATIONS: [&str; 4] = ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"];
/// The chunk digests of pinned files (see `digest::CHUNK_BYTES`), by whole-file SHA-256:
/// taken from files that matched their pins, compiled in like the manifest.
pub const CHUNKS: &str = include_str!("asset_chunks.json");
/// The fs-verity digests of pinned files (`crate::verity`), by whole-file SHA-256: taken from
/// files that matched their pins, compiled in like the chunk digests. A file's tree, built into
/// Straydog's data directory ([`tree_path`]), lets what is read again after loading be checked
/// block by block.
pub const VERITY: &str = include_str!("asset_verity.json");
pub const MODEL: &str = "Qwen3.8-Flash-Next";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct Bundle {
    pub quantization: String,
    pub shards: Vec<Asset>,
    pub resources: Vec<Asset>,
    pub drafter: Asset,
}

fn asset(value: &Value) -> Result<Asset, String> {
    let name = value["name"].as_str().ok_or("manifest asset without a name")?;
    if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\']) {
        return Err("invalid Straydog asset name".into());
    }
    Ok(Asset {
        name: name.to_string(),
        bytes: value["bytes"].as_u64().ok_or("manifest asset without a size")?,
        sha256: value["sha256"].as_str().ok_or("manifest asset without a digest")?.to_string(),
    })
}

/// One quantization's shards with the shared resources and drafter.
pub fn bundle(quantization: &str) -> Result<Bundle, String> {
    let manifest: Value = serde_json::from_str(MANIFEST).map_err(|exc| exc.to_string())?;
    if manifest["schema"] != 1 || manifest["model"] != MODEL {
        return Err("unsupported Straydog model manifest".into());
    }
    let shards = if manifest["quantization"] == quantization {
        &manifest["assets"]
    } else if QUANTIZATIONS.contains(&quantization) {
        &manifest["bundles"][quantization]["assets"]
    } else {
        return Err(format!("unsupported Straydog model quantization: {quantization}"));
    };
    let list = |value: &Value| -> Result<Vec<Asset>, String> {
        value.as_array().ok_or("manifest list missing")?.iter().map(asset).collect()
    };
    let drafter = &manifest["drafter"];
    if drafter["kind"] != "mtp" {
        return Err("Straydog drafter is not pinned in the model manifest".into());
    }
    Ok(Bundle {
        quantization: quantization.to_string(),
        shards: list(shards)?,
        resources: list(&manifest["resources"])?,
        drafter: asset(drafter)?,
    })
}

/// The pinned chunk digests of the file whose whole SHA-256 is `sha256`, if any.
pub fn chunk_digests(sha256: &str) -> Result<Option<Vec<String>>, String> {
    let chunks: Value = serde_json::from_str(CHUNKS).map_err(|exc| exc.to_string())?;
    if chunks["chunk_bytes"].as_u64() != Some(crate::digest::CHUNK_BYTES) {
        return Err("unsupported Straydog chunk digests".into());
    }
    let Some(list) = chunks["files"].get(sha256) else { return Ok(None) };
    let list = list.as_array().ok_or("invalid Straydog chunk digests")?;
    list.iter()
        .map(|digest| digest.as_str().map(str::to_string).ok_or_else(|| "invalid Straydog chunk digests".to_string()))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// The pinned fs-verity digest (hex) of the file whose whole SHA-256 is `sha256`, if any.
pub fn verity_digest(sha256: &str) -> Result<Option<String>, String> {
    let table: Value = serde_json::from_str(VERITY).map_err(|exc| exc.to_string())?;
    if table["block_bytes"].as_u64() != Some(crate::verity::BLOCK as u64) {
        return Err("unsupported Straydog verity digests".into());
    }
    match table["files"].get(sha256) {
        None => Ok(None),
        Some(Value::String(digest)) if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) => {
            Ok(Some(digest.clone()))
        }
        Some(_) => Err("invalid Straydog verity digests".into()),
    }
}

/// A quantization's shards that have a pinned fs-verity digest, with it (hex): every shard of a
/// bundle read on demand, and of the others those pinned in [`VERITY`].
pub fn verity_pins(quantization: &str) -> Result<Vec<(Asset, String)>, String> {
    if is_on_demand(quantization) {
        let pins = on_demand(quantization)?;
        return Ok(pins.shards.into_iter().zip(pins.verity).collect());
    }
    let mut pins = Vec::new();
    for asset in bundle(quantization)?.shards {
        if let Some(digest) = verity_digest(&asset.sha256)? {
            pins.push((asset, digest));
        }
    }
    Ok(pins)
}

/// The repository the resources (tokenizer, template, config) come from, at the manifest's
/// tokenizer revision: the official model's.
pub const RESOURCE_REPOSITORY: &str = "Qwen/Qwen3.8-Flash-Next";

/// Where a pinned file comes from: a Hugging Face repository, a revision and a path in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    pub asset: Asset,
    pub repository: String,
    pub revision: String,
    pub path: String,
}

/// Where every pinned file of `quantization` comes from: its shards, then the resources.
/// Provisioning only; the engine needs the pins alone.
pub fn sources(quantization: &str) -> Result<Vec<Source>, String> {
    let pins = bundle(quantization)?;
    let manifest: Value = serde_json::from_str(MANIFEST).map_err(|exc| exc.to_string())?;
    let listed = if manifest["quantization"] == quantization {
        &manifest["assets"]
    } else {
        &manifest["bundles"][quantization]["assets"]
    };
    let text = |value: &Value, what: &str| -> Result<String, String> {
        value.as_str().map(str::to_string).ok_or_else(|| format!("manifest without {what}"))
    };
    let repository = text(&manifest["repository"], "a repository")?;
    let revision = text(&manifest["revision"], "a revision")?;
    let tokenizer_revision = text(&manifest["tokenizer_revision"], "a tokenizer revision")?;
    let mut sources = Vec::new();
    for (asset, item) in pins.shards.into_iter().zip(listed.as_array().ok_or("manifest list missing")?) {
        let path = text(&item["source_path"], "a source path")?;
        sources.push(Source { asset, repository: repository.clone(), revision: revision.clone(), path });
    }
    for asset in pins.resources {
        let path = asset.name.clone();
        sources.push(Source { asset, repository: RESOURCE_REPOSITORY.into(), revision: tokenizer_revision.clone(), path });
    }
    Ok(sources)
}

/// One official tensor the drafter is built from, pinned by size and SHA-256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTensor {
    pub name: String,
    /// The safetensors shard of the official checkpoint that holds it.
    pub shard: String,
    pub bytes: u64,
    pub sha256: String,
}

/// The drafter's pin and what it is built from: the official checkpoint's repository,
/// revision and MTP tensors (in name order).
#[derive(Debug, Clone)]
pub struct DrafterSource {
    pub asset: Asset,
    pub repository: String,
    pub revision: String,
    pub tensors: Vec<SourceTensor>,
}

pub fn drafter_source() -> Result<DrafterSource, String> {
    let manifest: Value = serde_json::from_str(MANIFEST).map_err(|exc| exc.to_string())?;
    let drafter = &manifest["drafter"];
    if manifest["schema"] != 1 || manifest["model"] != MODEL || drafter["kind"] != "mtp" {
        return Err("Straydog drafter is not pinned in the model manifest".into());
    }
    let text = |value: &Value| value.as_str().map(str::to_string).ok_or("manifest drafter source incomplete");
    let mut tensors = Vec::new();
    for (name, pin) in drafter["source_tensors"].as_object().ok_or("manifest drafter source incomplete")? {
        tensors.push(SourceTensor {
            name: name.clone(),
            shard: text(&pin["shard"])?,
            bytes: pin["bytes"].as_u64().ok_or("manifest drafter source incomplete")?,
            sha256: text(&pin["sha256"])?,
        });
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(DrafterSource {
        asset: asset(drafter)?,
        repository: text(&drafter["repository"])?,
        revision: text(&drafter["revision"])?,
        tensors,
    })
}

/// Bundles read on demand (docs/DESIGN.md §10): too large to hash whole at each start, each shard
/// is pinned by size, SHA-256 and the fs-verity digest of its Merkle tree (`crate::verity`). The
/// trees are built once from the files into Straydog's data directory ([`tree_path`]), and the
/// worker checks every block it uses against them.
pub const ON_DEMAND_MANIFEST: &str = include_str!("assets_deepseek41.json");
pub const ON_DEMAND_QUANTIZATIONS: [&str; 1] = ["UD-IQ2_XXS"];

#[derive(Debug, Clone)]
pub struct OnDemand {
    pub model: String,
    /// The GGUF `general.architecture` the shards declare.
    pub architecture: String,
    pub quantization: String,
    pub shards: Vec<Asset>,
    /// Each shard's fs-verity digest, hex.
    pub verity: Vec<String>,
}

fn on_demand_manifest() -> Result<Value, String> {
    let manifest: Value = serde_json::from_str(ON_DEMAND_MANIFEST).map_err(|exc| exc.to_string())?;
    if manifest["schema"] != 1 || !manifest["model"].is_string() || !manifest["architecture"].is_string() {
        return Err("unsupported Straydog on-demand manifest".into());
    }
    Ok(manifest)
}

/// One on-demand quantization's shards and their digests.
pub fn on_demand(quantization: &str) -> Result<OnDemand, String> {
    if !ON_DEMAND_QUANTIZATIONS.contains(&quantization) {
        return Err(format!("unsupported Straydog model quantization: {quantization}"));
    }
    let manifest = on_demand_manifest()?;
    let listed = manifest["bundles"][quantization]["assets"].as_array().ok_or("manifest list missing")?;
    let shards = listed.iter().map(asset).collect::<Result<Vec<_>, _>>()?;
    let verity = listed
        .iter()
        .map(|item| match item["verity"].as_str() {
            Some(digest) if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) => Ok(digest.to_string()),
            _ => Err("manifest asset without a verity digest".to_string()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OnDemand {
        model: manifest["model"].as_str().unwrap().to_string(),
        architecture: manifest["architecture"].as_str().unwrap().to_string(),
        quantization: quantization.to_string(),
        shards,
        verity,
    })
}

/// Where every shard of an on-demand quantization comes from (provisioning only).
pub fn on_demand_sources(quantization: &str) -> Result<Vec<Source>, String> {
    let pins = on_demand(quantization)?;
    let manifest = on_demand_manifest()?;
    let text = |value: &Value, what: &str| -> Result<String, String> {
        value.as_str().map(str::to_string).ok_or_else(|| format!("manifest without {what}"))
    };
    let repository = text(&manifest["repository"], "a repository")?;
    let revision = text(&manifest["revision"], "a revision")?;
    let listed = manifest["bundles"][quantization]["assets"].as_array().ok_or("manifest list missing")?;
    pins.shards
        .into_iter()
        .zip(listed)
        .map(|(asset, item)| {
            Ok(Source { asset, repository: repository.clone(), revision: revision.clone(), path: text(&item["source_path"], "a source path")? })
        })
        .collect()
}

/// Whether a quantization names an on-demand bundle.
pub fn is_on_demand(quantization: &str) -> bool {
    ON_DEMAND_QUANTIZATIONS.contains(&quantization)
}

/// Every pinned quantization: the resident model's, then the on-demand ones.
pub fn quantizations() -> impl Iterator<Item = &'static str> {
    QUANTIZATIONS.into_iter().chain(ON_DEMAND_QUANTIZATIONS)
}

/// The model a pinned quantization belongs to.
pub fn model_of(quantization: &str) -> Result<String, String> {
    if is_on_demand(quantization) {
        Ok(on_demand(quantization)?.model)
    } else if QUANTIZATIONS.contains(&quantization) {
        Ok(MODEL.to_string())
    } else {
        Err(format!("unsupported Straydog model quantization: {quantization}"))
    }
}

/// The GGUF architecture of a pinned quantization's model.
pub fn architecture_of(quantization: &str) -> Result<String, String> {
    if is_on_demand(quantization) {
        Ok(on_demand(quantization)?.architecture)
    } else if QUANTIZATIONS.contains(&quantization) {
        Ok("qwen4exp".into())
    } else {
        Err(format!("unsupported Straydog model quantization: {quantization}"))
    }
}

/// The shards a quantization pins (either kind of bundle).
pub fn shards(quantization: &str) -> Result<Vec<Asset>, String> {
    if is_on_demand(quantization) { Ok(on_demand(quantization)?.shards) } else { Ok(bundle(quantization)?.shards) }
}

/// Where Straydog keeps what it makes for a model (its shards' verity trees, the expert store's
/// warm-up state) unless told otherwise: this directory in the model's.
pub const DATA_DIRECTORY: &str = ".straydog";

/// The data directory of the model in `model` when none is given: [`DATA_DIRECTORY`] there.
pub fn default_data_dir(model: &std::path::Path) -> std::path::PathBuf {
    model.join(DATA_DIRECTORY)
}

/// The verity tree of a pinned shard in Straydog's data directory: built there from the file (at
/// the server's first start, or by `straydog verify`) by the server, never by its worker, and
/// checked against the pinned digest when opened.
pub fn tree_path(data_dir: &std::path::Path, shard: &Asset) -> std::path::PathBuf {
    data_dir.join("verity").join(format!("{}.verity", shard.name))
}
