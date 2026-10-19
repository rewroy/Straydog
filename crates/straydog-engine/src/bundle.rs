//! The pinned model bundles: every shard, resource and the drafter by size and SHA-256.

use serde_json::Value;

/// The manifest, compiled in: a binary authenticates exactly the bundles it was tested with.
pub const MANIFEST: &str = include_str!("../../../src/straydog/assets.json");
pub const QUANTIZATIONS: [&str; 4] = ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"];
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
