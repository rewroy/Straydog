//! The pinned model bundles: every shard, resource and the drafter by size and SHA-256.

use serde_json::Value;

/// The manifest, compiled in: a binary authenticates exactly the bundles it was tested with.
pub const MANIFEST: &str = include_str!("assets.json");
/// Qwen3.8-Flash-Next's bundles: ISTA-DASLab's (GSQ-RCO), then Unsloth's.
pub const QUANTIZATIONS: [&str; 8] = ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S", "UD-IQ1_M", "UD-IQ1_S", "UD-IQ2_XS", "UD-IQ3_XXS"];
/// The chunk digests of pinned files (see `digest::CHUNK_BYTES`), by whole-file SHA-256:
/// taken from files that matched their pins, compiled in like the manifest.
pub const CHUNKS: &str = include_str!("asset_chunks.json");
/// The fs-verity digests of pinned files (`crate::verity`), by whole-file SHA-256: taken from
/// files that matched their pins, compiled in like the chunk digests. A file's tree, built into
/// Straydog's data directory ([`tree_path`]), lets what is read again after loading be checked
/// block by block.
pub const VERITY: &str = include_str!("asset_verity.json");
pub const MODEL: &str = "Qwen3.8-Flash-Next";
/// The model of the bundles read on demand.
pub const DEEPSEEK: &str = "DeepSeek-V4.1-Flash-1017";
/// The bundles' providers: who quantized the model.
pub const ISTA: &str = "ISTA-DASLab";
pub const UNSLOTH: &str = "Unsloth";

/// Every pinned bundle: Qwen3.8's, then DeepSeek-V4.1's.
pub const BUNDLES: [Id; 12] = [
    Id { model: MODEL, provider: ISTA, quant: "Q2_0" },
    Id { model: MODEL, provider: ISTA, quant: "IQ2_XS" },
    Id { model: MODEL, provider: ISTA, quant: "IQ3_XXS" },
    Id { model: MODEL, provider: ISTA, quant: "IQ3_S" },
    Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ1_M" },
    Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ1_S" },
    Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ2_XS" },
    Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ3_XXS" },
    Id { model: DEEPSEEK, provider: UNSLOTH, quant: "UD-IQ2_XXS" },
    Id { model: DEEPSEEK, provider: ISTA, quant: "IQ2_XS" },
    Id { model: DEEPSEEK, provider: ISTA, quant: "IQ2_XXS" },
    Id { model: DEEPSEEK, provider: UNSLOTH, quant: "UD-IQ1_M" },
];

/// Names earlier releases gave a bundle (their settings files hold them): DeepSeek-V4.1's ISTA-DASLab
/// bundles by their files' GSQ-RCO names, the model without its release.
const FORMER_QUANTS: [(&str, Id); 2] = [
    ("GSQ-RCO-IQ2_XS", Id { model: DEEPSEEK, provider: ISTA, quant: "IQ2_XS" }),
    ("GSQ-RCO-IQ2_XXS", Id { model: DEEPSEEK, provider: ISTA, quant: "IQ2_XXS" }),
];
const FORMER_MODELS: [(&str, &str); 1] = [("DeepSeek-V4.1-Flash", DEEPSEEK)];

/// A model with pinned bundles: who made it and the license its weights are under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub name: &'static str,
    pub creator: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
}

pub const MODELS: [Model; 2] = [
    Model { name: MODEL, creator: "Qwen", license: "Qwen Community License 1.0",
            license_url: "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE" },
    Model { name: DEEPSEEK, creator: "DeepSeek", license: "MIT",
            license_url: "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017" },
];

/// The model `name` (as [`MODELS`] has it).
pub fn model(name: &str) -> Option<Model> {
    MODELS.into_iter().find(|model| model.name == name)
}
/// What each bundle's tensors are by how the engine places them ([`Footprint`]), summed from the
/// pinned shards' headers (examples/asset_footprint.rs).
pub const FOOTPRINTS: &str = include_str!("asset_footprint.json");

/// A pinned bundle, named by three labels: the model, the provider who quantized it and the quantization.
/// Providers name their quantizations alike (IQ2_XS of ISTA-DASLab for both models, Unsloth's UD-IQ1_M
/// for both), so no one label names a bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id {
    pub model: &'static str,
    pub provider: &'static str,
    pub quant: &'static str,
}

impl Id {
    /// The pinned bundle these labels name (model and provider in any case, a label left out matching
    /// any): an error when they name none, or more than one.
    pub fn find(model: Option<&str>, provider: Option<&str>, quant: &str) -> Result<Id, String> {
        if let Some(model) = model.filter(|model| matching(Some(model), None, None).is_empty()) {
            return Err(format!("unknown model {model}: {}", models().join(", ")));
        }
        match matching(model, provider, Some(quant))[..] {
            [id] => Ok(id),
            [] => {
                let known: Vec<String> = matching(model, provider, None).iter().map(Id::to_string).collect();
                Err(format!("unknown bundle {quant}: {}", known.join(", ")))
            }
            ref found => Err(format!("{quant} names {}: give the model and provider too",
                                     found.iter().map(Id::to_string).collect::<Vec<_>>().join(" and "))),
        }
    }

    /// Whether its shards are read on demand (DeepSeek-V4.1's) rather than loaded whole (Qwen3.8's).
    pub fn is_on_demand(self) -> bool {
        self.model == DEEPSEEK
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} / {} / {}", self.model, self.provider, self.quant)
    }
}

/// The pinned bundles these labels fit (model and provider in any case, a label left out fitting any;
/// earlier releases' names fit the bundles they named).
pub fn matching(model: Option<&str>, provider: Option<&str>, quant: Option<&str>) -> Vec<Id> {
    let model = model.map(|name| FORMER_MODELS.iter().find(|(former, _)| former.eq_ignore_ascii_case(name)).map_or(name, |(_, now)| now));
    // (a former name is its bundle's three labels)
    let former = quant.and_then(|name| FORMER_QUANTS.iter().find(|(former, _)| *former == name)).map(|(_, id)| *id);
    let fits = |given: Option<&str>, label: &str| given.is_none_or(|given| label.eq_ignore_ascii_case(given));
    BUNDLES.into_iter()
        .filter(|id| fits(model, id.model) && fits(provider, id.provider))
        .filter(|id| match former {
            Some(former) => *id == former,
            None => quant.is_none_or(|quant| id.quant == quant),
        })
        .collect()
}

/// Every pinned bundle ([`BUNDLES`]).
pub fn bundles() -> impl Iterator<Item = Id> {
    BUNDLES.into_iter()
}

/// The models with pinned bundles, in [`BUNDLES`]' order.
pub fn models() -> Vec<&'static str> {
    let mut models: Vec<&'static str> = Vec::new();
    for id in BUNDLES {
        if !models.contains(&id.model) {
            models.push(id.model);
        }
    }
    models
}

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

/// One of Qwen3.8's bundles: its shards with the shared resources and drafter.
pub fn bundle(id: Id) -> Result<Bundle, String> {
    let manifest: Value = serde_json::from_str(MANIFEST).map_err(|exc| exc.to_string())?;
    if manifest["schema"] != 1 || manifest["model"] != MODEL {
        return Err("unsupported Straydog model manifest".into());
    }
    let quantization = id.quant;
    let shards = if id.model != MODEL || !BUNDLES.contains(&id) {
        return Err(format!("{id} is not a bundle loaded whole"));
    } else if manifest["quantization"] == quantization {
        &manifest["assets"]
    } else if QUANTIZATIONS.contains(&quantization) {
        &manifest["bundles"][quantization]["assets"]
    } else {
        return Err(format!("unsupported Straydog bundle: {id}"));
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

/// What a bundle's tensors come to by how the engine places them: the `dense` weights every token
/// reads (they live in VRAM), the routed `experts` (VRAM slots over a host tier over the drive),
/// and the `tables` read by row from the drive (Qwen3.8's PLE table, DeepSeek-V4.1's Engram).
/// Headers and alignment are the little that is left of the shards' sizes. The sequence state grows
/// with the context by `state_per_token` bytes a token (K/V and the like, from the model's
/// hyperparameters).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    pub dense: u64,
    pub experts: u64,
    pub tables: u64,
    pub state_per_token: u64,
}

/// The footprint pinned for a bundle (the table's bundles by model, provider and quantization).
pub fn footprint(id: Id) -> Result<Footprint, String> {
    let table: Value = serde_json::from_str(FOOTPRINTS).map_err(|exc| exc.to_string())?;
    let entry = table["bundles"].get(id.model).and_then(|providers| providers.get(id.provider)).and_then(|bundles| bundles.get(id.quant))
        .ok_or_else(|| format!("no footprint pinned for {id}"))?;
    let bytes = |key: &str| entry[key].as_u64().ok_or_else(|| format!("footprint of {id} without {key}"));
    if table["schema"] != 3 {
        return Err("unsupported Straydog footprint table".into());
    }
    Ok(Footprint { dense: bytes("dense")?, experts: bytes("experts")?, tables: bytes("tables")?, state_per_token: bytes("state_per_token")? })
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
pub fn verity_pins(id: Id) -> Result<Vec<(Asset, String)>, String> {
    if id.is_on_demand() {
        let pins = on_demand(id)?;
        let sidecars = pins.sidecars.into_iter().map(|sidecar| (sidecar.asset, sidecar.verity));
        return Ok(pins.shards.into_iter().zip(pins.verity).chain(sidecars).collect());
    }
    let mut pins = Vec::new();
    for asset in bundle(id)?.shards {
        if let Some(digest) = verity_digest(&asset.sha256)? {
            pins.push((asset, digest));
        }
    }
    Ok(pins)
}

/// The directory beside (or inside) a Qwen3.8 bundle's that holds engram tables any of the model's bundles may read.
pub const ENGRAM_DIRECTORY: &str = "engram";

/// Table precisions from the most exact: a bundle reads the most exact table there is ([`table_in`]).
pub const TABLE_PRECISIONS: [&str; 4] = ["BF16", "Q8_0", "Q5_1", "IQ4_NL"];

/// A table a bundle may read its engram (PLE) rows from: its own last shard (`bundled`), or one of Unsloth's engram files,
/// the same rows at another precision, one tensor in a file of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub precision: String,
    pub asset: Asset,
    pub bundled: bool,
}

/// The tables a bundle may read: its own shard's first, then the engram files of the model's manifest; none for a bundle
/// whose last shard is not a table of its own.
pub fn tables(id: Id) -> Result<Vec<Table>, String> {
    if id.is_on_demand() {
        return Ok(Vec::new());
    }
    let manifest: Value = serde_json::from_str(MANIFEST).map_err(|exc| exc.to_string())?;
    let Some(precision) = manifest["bundles"][id.quant]["table"].as_str() else { return Ok(Vec::new()) };
    let shard = bundle(id)?.shards.last().cloned().ok_or("Straydog bundle without shards")?;
    let mut tables = vec![Table { precision: precision.to_string(), asset: shard, bundled: true }];
    for item in manifest["tables"]["assets"].as_array().ok_or("manifest tables missing")? {
        let precision = item["precision"].as_str().ok_or("manifest table without a precision")?;
        tables.push(Table { precision: precision.to_string(), asset: asset(item)?, bundled: false });
    }
    Ok(tables)
}

/// Where a bundle in `root` may keep an engram table: in `root`, in `root/engram`, or in an `engram` directory beside
/// `root` (the model's bundles side by side share one).
pub fn engram_directories(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    [Some(root.to_path_buf()), Some(root.join(ENGRAM_DIRECTORY)), root.parent().map(|parent| parent.join(ENGRAM_DIRECTORY))]
        .into_iter().flatten().collect()
}

/// The table a bundle in `root` reads, and where: the one `STRAYDOG_ENGRAM` names ([`table_in_with`]).
pub fn table_in(root: &std::path::Path, id: Id) -> Result<Option<(Table, std::path::PathBuf)>, String> {
    let named = std::env::var("STRAYDOG_ENGRAM").ok().filter(|precision| !precision.is_empty());
    table_in_with(root, id, named.as_deref())
}

/// The table a bundle in `root` reads, and where: the one of `precision` when one is named (an error when it is not
/// there), else its own last shard, else the most exact of the model's engram files there ([`TABLE_PRECISIONS`]: a bundle
/// set up with one in its own shard's place). A larger table is read by choice alone: on a machine short of memory its rows
/// cost the experts' banks room (Q5_1 decoded a tenth slower than the shard's IQ4_NL on 62 GiB with 30 held by other
/// programs). None for a bundle without a table of its own, or when none is there.
pub fn table_in_with(root: &std::path::Path, id: Id, precision: Option<&str>) -> Result<Option<(Table, std::path::PathBuf)>, String> {
    let tables = tables(id)?;
    if tables.is_empty() {
        return Ok(None);
    }
    let rank = |table: &Table| TABLE_PRECISIONS.iter().position(|precision| *precision == table.precision).unwrap_or(TABLE_PRECISIONS.len());
    let mut found: Vec<(Table, std::path::PathBuf)> = Vec::new();
    for table in tables {
        let directories = if table.bundled { vec![root.to_path_buf()] } else { engram_directories(root) };
        if let Some(path) = directories.into_iter().map(|directory| directory.join(&table.asset.name)).find(|path| path.is_file()) {
            found.push((table, path));
        }
    }
    if let Some(precision) = precision {
        return found.into_iter().find(|(table, _)| table.precision.eq_ignore_ascii_case(precision)).map(Some)
            .ok_or_else(|| format!("engram {precision}: no such table for {id} in {} or its engram directories", root.display()));
    }
    found.sort_by_key(|(table, _)| (!table.bundled, rank(table)));
    Ok(found.into_iter().next())
}

/// Where the pinned engram tables come from (provisioning only).
pub fn table_source(table: &Table) -> Result<Source, String> {
    let manifest: Value = serde_json::from_str(MANIFEST).map_err(|exc| exc.to_string())?;
    let tables = &manifest["tables"];
    let item = tables["assets"].as_array().into_iter().flatten().find(|item| item["name"] == table.asset.name.as_str())
        .ok_or_else(|| format!("{} is not a pinned engram table", table.asset.name))?;
    let text = |value: &Value| value.as_str().map(str::to_string).ok_or("manifest tables incomplete");
    Ok(Source { asset: table.asset.clone(), repository: text(&tables["repository"])?, revision: text(&tables["revision"])?,
                path: text(&item["source_path"])? })
}

/// [`verity_pins`] of the files a model in `root` has: with its vision encoder and its DSpark
/// drafter when pinned and there ([`vision_source`], [`dspark_in`]).
pub fn verity_pins_in(root: &std::path::Path, id: Id) -> Result<Vec<(Asset, String)>, String> {
    Ok(verity_files_in(root, id)?.into_iter().map(|(_, asset, digest)| (asset, digest)).collect())
}

/// [`verity_pins_in`] with where each file lies: the shards, sidecars and vision encoder in `root`,
/// the DSpark drafter where [`dspark_in`] finds it.
pub fn verity_files_in(root: &std::path::Path, id: Id) -> Result<Vec<(std::path::PathBuf, Asset, String)>, String> {
    let mut files: Vec<_> = verity_pins(id)?.into_iter().map(|(asset, digest)| (root.join(&asset.name), asset, digest)).collect();
    // (the table the bundle reads, when it is an engram file rather than its own last shard: in the shard's place)
    if let Some((table, path)) = table_in(root, id)?.filter(|(table, _)| !table.bundled) {
        let shard = bundle(id)?.shards.last().cloned().ok_or("Straydog bundle without shards")?;
        files.retain(|(_, asset, _)| *asset != shard);
        let digest = verity_digest(&table.asset.sha256)?.ok_or_else(|| format!("{} has no pinned verity digest", table.asset.name))?;
        files.push((path, table.asset, digest));
    }
    if let Some(vision) = vision_source(id)?.filter(|vision| root.join(&vision.asset.name).is_file()) {
        files.push((root.join(&vision.asset.name), vision.asset, vision.verity));
    }
    if let Some((dspark, path)) = dspark_in(root, id)? {
        files.push((path, dspark.asset, dspark.verity));
    }
    Ok(files)
}

/// The GGUF `general.architecture` of DeepSeek-V4.1's vision encoder file.
pub const VISION_ARCHITECTURE: &str = "deepseek41-vision";

/// The GGUF `general.architecture` of DeepSeek-V4.1's separate Engram tables file.
pub const ENGRAM_ARCHITECTURE: &str = "deepseek41-engram";

/// The pins of DeepSeek-V4.1's vision encoder: the official checkpoint's vision tower, aligner,
/// image span embeddings and image routing biases, which provisioning range-fetches (each
/// tensor's SHA-256 pinned) into one GGUF beside the bundle, itself pinned by size, SHA-256 and
/// fs-verity digest. Optional: a model without the file reads no images.
#[derive(Debug, Clone)]
pub struct VisionSource {
    pub asset: Asset,
    pub verity: String,
    pub repository: String,
    pub revision: String,
    /// In name order.
    pub tensors: Vec<SourceTensor>,
}

const VISION_MANIFEST: &str = include_str!("assets_deepseek41_vision.json");

/// The vision encoder pinned for an on-demand bundle's model, if it has one.
pub fn vision_source(id: Id) -> Result<Option<VisionSource>, String> {
    if !id.is_on_demand() {
        return Ok(None);
    }
    let pins = on_demand(id)?;
    let manifest: Value = serde_json::from_str(VISION_MANIFEST).map_err(|exc| exc.to_string())?;
    if manifest["architecture"] != pins.architecture.as_str() {
        return Ok(None);
    }
    let vision = &manifest["vision"];
    if manifest["schema"] != 1 || vision["kind"] != "vision" {
        return Err("unsupported Straydog vision manifest".into());
    }
    let incomplete = "manifest vision source incomplete";
    let text = |value: &Value| value.as_str().map(str::to_string).ok_or(incomplete);
    let mut tensors = Vec::new();
    for (name, pin) in vision["source_tensors"].as_object().ok_or(incomplete)? {
        tensors.push(SourceTensor {
            name: name.clone(),
            shard: text(&pin["shard"])?,
            bytes: pin["bytes"].as_u64().ok_or(incomplete)?,
            sha256: text(&pin["sha256"])?,
        });
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Some(VisionSource {
        asset: asset(vision)?,
        verity: verity_of(vision)?,
        repository: text(&vision["repository"])?,
        revision: text(&vision["revision"])?,
        tensors,
    }))
}

/// The GGUF `general.architecture` of DeepSeek-V4.1's DSpark drafter file.
pub const DSPARK_ARCHITECTURE: &str = "dflash";

/// The directory beside (or inside) a model's directory that holds its DSpark drafter.
pub const DSPARK_DIRECTORY: &str = "dspark";

/// The pins of DeepSeek-V4.1's DSpark drafter: Unsloth's Q8_0 GGUF of the three-stage block drafter
/// (`mtp.0..2`) of the official checkpoint, pinned by size, SHA-256 (the repository's LFS oid) and
/// fs-verity digest. A file of its own with tensor names of its own, so it opens apart from the
/// shards ([`DsparkSource::pins`]). Optional: a model without the file drafts from its own text.
#[derive(Debug, Clone)]
pub struct DsparkSource {
    pub asset: Asset,
    pub verity: String,
    pub repository: String,
    pub revision: String,
    /// The file's path in the repository.
    pub path: String,
}

const DSPARK_MANIFEST: &str = include_str!("assets_deepseek41_dspark.json");

impl DsparkSource {
    /// The drafter as a bundle of one shard, to open ([`crate::pinned::Pinned::open`]) from the
    /// directory it lies in.
    pub fn pins(&self) -> OnDemand {
        OnDemand {
            model: DEEPSEEK.into(), architecture: DSPARK_ARCHITECTURE.into(), quantization: "DSpark".into(),
            shards: vec![self.asset.clone()], verity: vec![self.verity.clone()], resources: Vec::new(), sidecars: Vec::new(),
        }
    }
}

/// The DSpark drafter pinned for an on-demand bundle, if it has one.
pub fn dspark_source(id: Id) -> Result<Option<DsparkSource>, String> {
    if !id.is_on_demand() {
        return Ok(None);
    }
    let pins = on_demand(id)?;
    let manifest: Value = serde_json::from_str(DSPARK_MANIFEST).map_err(|exc| exc.to_string())?;
    // (the model's drafter, whatever the bundle's provider and quantization: it only proposes, and every draft is verified)
    if manifest["architecture"] != pins.architecture.as_str() {
        return Ok(None);
    }
    let dspark = &manifest["dspark"];
    let incomplete = "manifest DSpark source incomplete";
    if manifest["schema"] != 1 || dspark["kind"] != "dspark" || dspark["architecture"] != DSPARK_ARCHITECTURE {
        return Err("unsupported Straydog DSpark manifest".into());
    }
    let text = |value: &Value| value.as_str().map(str::to_string).ok_or(incomplete);
    Ok(Some(DsparkSource {
        asset: asset(dspark)?,
        verity: verity_of(dspark)?,
        repository: text(&dspark["repository"])?,
        revision: text(&dspark["revision"])?,
        path: text(&dspark["source_path"])?,
    }))
}

/// Where a model in `root` keeps its DSpark drafter file: in `root/dspark`, else in a `dspark`
/// directory beside `root` (a drafter belongs to the model, not to a quantization, so the
/// directories of one model's quantizations share one).
pub fn dspark_directories(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    [Some(root.join(DSPARK_DIRECTORY)), root.parent().map(|parent| parent.join(DSPARK_DIRECTORY))].into_iter().flatten().collect()
}

/// The DSpark drafter of the model in `root` and the file's path, when one is pinned for the
/// bundle and there.
pub fn dspark_in(root: &std::path::Path, id: Id) -> Result<Option<(DsparkSource, std::path::PathBuf)>, String> {
    let Some(source) = dspark_source(id)? else { return Ok(None) };
    let found = dspark_directories(root).into_iter().map(|directory| directory.join(&source.asset.name)).find(|path| path.is_file());
    Ok(found.map(|path| (source, path)))
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

/// Where every pinned file of one of Qwen3.8's bundles comes from: its shards, then the resources.
/// Provisioning only; the engine needs the pins alone.
pub fn sources(id: Id) -> Result<Vec<Source>, String> {
    let pins = bundle(id)?;
    let quantization = id.quant;
    let manifest: Value = serde_json::from_str(MANIFEST).map_err(|exc| exc.to_string())?;
    let listed = if manifest["quantization"] == quantization {
        &manifest["assets"]
    } else {
        &manifest["bundles"][quantization]["assets"]
    };
    let text = |value: &Value, what: &str| -> Result<String, String> {
        value.as_str().map(str::to_string).ok_or_else(|| format!("manifest without {what}"))
    };
    // (a bundle another repository publishes names it and its revision: Unsloth's)
    let entry = &manifest["bundles"][quantization];
    let repository = text(if entry["repository"].is_string() { &entry["repository"] } else { &manifest["repository"] }, "a repository")?;
    let revision = text(if entry["revision"].is_string() { &entry["revision"] } else { &manifest["revision"] }, "a revision")?;
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

/// Qwen3.8-Flash-Next's vision encoder and projector, one file for every quantization: the GGUF repository's own, pinned by
/// size and SHA-256 (its LFS oid). Optional: a model without the file reads no images.
pub fn qwen_vision_source() -> Result<Source, String> {
    let manifest: Value = serde_json::from_str(MANIFEST).map_err(|exc| exc.to_string())?;
    let vision = &manifest["vision"];
    if manifest["schema"] != 1 || manifest["model"] != MODEL || vision["kind"] != "mmproj" {
        return Err("Straydog vision encoder is not pinned in the model manifest".into());
    }
    let text = |value: &Value| value.as_str().map(str::to_string).ok_or("manifest vision encoder incomplete");
    Ok(Source { asset: asset(vision)?, repository: text(&vision["repository"])?, revision: text(&vision["revision"])?, path: text(&vision["source_path"])? })
}

/// Where a Qwen3.8 model in `root` keeps its projector, when nothing names it: the pinned file, else another
/// `mmproj-*.gguf` (found to be the wrong one when it is opened, with a reason a request is told), in `root` or beside it.
pub fn qwen_vision_in(root: &std::path::Path) -> Option<std::path::PathBuf> {
    let pinned = qwen_vision_source().ok().map(|source| source.asset.name);
    for directory in [Some(root), root.parent()].into_iter().flatten() {
        if let Some(found) = pinned.as_ref().map(|name| directory.join(name)).filter(|path| path.is_file()) {
            return Some(found);
        }
        let mut others: Vec<_> = std::fs::read_dir(directory).into_iter().flatten().flatten().map(|entry| entry.path())
            .filter(|path| path.is_file() && path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("mmproj-") && name.ends_with(".gguf")))
            .collect();
        others.sort();
        if let Some(found) = others.into_iter().next() {
            return Some(found);
        }
    }
    None
}

/// Bundles read on demand (docs/DESIGN.md §10): too large to hash whole at each start, each shard
/// is pinned by size, SHA-256 and the fs-verity digest of its Merkle tree (`crate::verity`). The
/// trees are built once from the files into Straydog's data directory ([`tree_path`]), and the
/// worker checks every block it uses against them. A model whose official repository gives its
/// tokenizer and chat template has them pinned too, as resources (read whole and hashed).
/// The manifest pinning DeepSeek-V4.1's bundles of a provider: Unsloth's, ISTA-DASLab's (GSQ-RCO).
fn on_demand_manifest_text(provider: &str) -> Option<&'static str> {
    match provider {
        UNSLOTH => Some(include_str!("assets_deepseek41.json")),
        ISTA => Some(include_str!("assets_deepseek41_gsq.json")),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct OnDemand {
    pub model: String,
    /// The GGUF `general.architecture` the shards declare.
    pub architecture: String,
    pub quantization: String,
    pub shards: Vec<Asset>,
    /// Each shard's fs-verity digest, hex.
    pub verity: Vec<String>,
    /// Official files beside the shards (tokenizer, chat template, generation config), if any.
    pub resources: Vec<Asset>,
    /// GGUFs beside the shards that the model reads but that are not its shards (a separate
    /// Engram's tables), read on demand as the shards are.
    pub sidecars: Vec<Sidecar>,
}

/// A GGUF beside a bundle's shards: pinned as they are, with the architecture it declares.
#[derive(Debug, Clone)]
pub struct Sidecar {
    pub asset: Asset,
    pub verity: String,
    pub architecture: String,
}

/// A manifest asset's fs-verity digest.
fn verity_of(item: &Value) -> Result<String, String> {
    match item["verity"].as_str() {
        Some(digest) if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) => Ok(digest.to_string()),
        _ => Err("manifest asset without a verity digest".to_string()),
    }
}

/// The manifest pinning an on-demand bundle.
fn on_demand_manifest(id: Id) -> Result<Value, String> {
    let quantization = id.quant;
    let text = on_demand_manifest_text(id.provider).filter(|_| id.is_on_demand() && BUNDLES.contains(&id))
        .ok_or_else(|| format!("{id} is not a bundle read on demand"))?;
    let manifest: Value = serde_json::from_str(text).map_err(|exc| exc.to_string())?;
    if manifest["schema"] != 1 || manifest["model"] != id.model || manifest["provider"] != id.provider || !manifest["architecture"].is_string()
        || !manifest["bundles"][quantization].is_object() {
        return Err("unsupported Straydog on-demand manifest".into());
    }
    Ok(manifest)
}

/// One on-demand bundle's shards, their digests and the model's resources.
pub fn on_demand(id: Id) -> Result<OnDemand, String> {
    let manifest = on_demand_manifest(id)?;
    let quantization = id.quant;
    let listed = manifest["bundles"][quantization]["assets"].as_array().ok_or("manifest list missing")?;
    let shards = listed.iter().map(asset).collect::<Result<Vec<_>, _>>()?;
    let verity = listed.iter().map(verity_of).collect::<Result<Vec<_>, _>>()?;
    let resources = match manifest["resources"]["assets"].as_array() {
        Some(items) => items.iter().map(asset).collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    let mut sidecars = Vec::new();
    for item in manifest["bundles"][quantization]["sidecars"].as_array().into_iter().flatten() {
        let architecture = item["architecture"].as_str().ok_or("manifest sidecar without an architecture")?;
        sidecars.push(Sidecar { asset: asset(item)?, verity: verity_of(item)?, architecture: architecture.to_string() });
    }
    Ok(OnDemand {
        model: manifest["model"].as_str().unwrap().to_string(),
        architecture: manifest["architecture"].as_str().unwrap().to_string(),
        quantization: quantization.to_string(),
        shards,
        verity,
        resources,
        sidecars,
    })
}

/// Where every file of an on-demand bundle comes from: its shards, then the resources
/// (provisioning only).
pub fn on_demand_sources(id: Id) -> Result<Vec<Source>, String> {
    let pins = on_demand(id)?;
    let manifest = on_demand_manifest(id)?;
    let quantization = id.quant;
    let text = |value: &Value, what: &str| -> Result<String, String> {
        value.as_str().map(str::to_string).ok_or_else(|| format!("manifest without {what}"))
    };
    let repository = text(&manifest["repository"], "a repository")?;
    // (a bundle may come from another revision than the manifest's, and a file of it from another still:
    // the Engram file a later revision of the repository rewrote, which the bundle keeps as it was made)
    let entry = &manifest["bundles"][quantization];
    let revision = text(if entry["revision"].is_string() { &entry["revision"] } else { &manifest["revision"] }, "a revision")?;
    let listed = entry["assets"].as_array().ok_or("manifest list missing")?;
    let sidecars = entry["sidecars"].as_array().map_or(&[][..], Vec::as_slice);
    let assets = pins.shards.into_iter().chain(pins.sidecars.into_iter().map(|sidecar| sidecar.asset));
    let mut sources = assets
        .zip(listed.iter().chain(sidecars))
        .map(|(asset, item)| {
            let revision = if item["revision"].is_string() { text(&item["revision"], "a revision")? } else { revision.clone() };
            // (a file another repository holds the same tables in, as UD-IQ1_M's Engram)
            let repository = if item["repository"].is_string() { text(&item["repository"], "a repository")? } else { repository.clone() };
            Ok(Source { asset, repository, revision, path: text(&item["source_path"], "a source path")? })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if !pins.resources.is_empty() {
        let resources = &manifest["resources"];
        let (repository, revision) = (text(&resources["repository"], "a resource repository")?, text(&resources["revision"], "a resource revision")?);
        for asset in pins.resources {
            let path = asset.name.clone();
            sources.push(Source { asset, repository: repository.clone(), revision: revision.clone(), path });
        }
    }
    Ok(sources)
}

/// The GGUF architecture of a pinned bundle's model.
pub fn architecture_of(id: Id) -> Result<String, String> {
    if id.is_on_demand() {
        Ok(on_demand(id)?.architecture)
    } else if id.model == MODEL && QUANTIZATIONS.contains(&id.quant) {
        Ok("qwen4exp".into())
    } else {
        Err(format!("unsupported Straydog bundle: {id}"))
    }
}

/// Where every pinned file of a bundle comes from (either kind): provisioning only.
pub fn sources_of(id: Id) -> Result<Vec<Source>, String> {
    if id.is_on_demand() { on_demand_sources(id) } else { sources(id) }
}

/// The shards a bundle pins (either kind).
pub fn shards(id: Id) -> Result<Vec<Asset>, String> {
    if id.is_on_demand() { Ok(on_demand(id)?.shards) } else { Ok(bundle(id)?.shards) }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn qw(quant: &str) -> Id {
        Id::find(Some(MODEL), None, quant).unwrap()
    }

    fn ds(quant: &str) -> Id {
        Id::find(Some(DEEPSEEK), None, quant).unwrap()
    }

    /// Every on-demand quantization has its manifest: a model, an architecture, verity digests for
    /// every shard, and its official resources from their own repository.
    #[test]
    fn on_demand_bundles_pin_their_shards_and_resources() {
        for id in bundles().filter(|id| id.is_on_demand()) {
            let pins = on_demand(id).unwrap();
            assert_eq!(pins.verity.len(), pins.shards.len(), "{id}");
            assert!(!pins.shards.is_empty() && pins.shards.iter().all(|asset| asset.sha256.len() == 64), "{id}");
            let sources = on_demand_sources(id).unwrap();
            assert_eq!(sources.len(), pins.shards.len() + pins.resources.len() + pins.sidecars.len(), "{id}");
            assert_eq!(pins.model, DEEPSEEK);
        }
        assert!(on_demand(ds("UD-IQ2_XXS")).unwrap().resources.is_empty());
        // ISTA's GSQ-RCO DeepSeek-V4.1: four shards and the Engram tables beside them, apart from
        // Qwen3.8's IQ2_XS
        let gsq = on_demand(ds("IQ2_XS")).unwrap();
        assert_eq!((gsq.model.as_str(), gsq.architecture.as_str(), gsq.shards.len()), (DEEPSEEK, "deepseek41", 4));
        assert_eq!(gsq.sidecars.iter().map(|sidecar| sidecar.architecture.as_str()).collect::<Vec<_>>(), ["deepseek41-engram"]);
        assert_eq!(verity_pins(ds("IQ2_XS")).unwrap().len(), 5);
        // its IQ2_XXS (the repository's Oct-27 split layout): six shards at their own revision, the sixth
        // the Engram file as the IQ2_XS bundle has it (made before that layout, from the earlier
        // revision, under the sixth shard's name), and no sidecars
        let xxs = on_demand(ds("IQ2_XXS")).unwrap();
        assert_eq!((xxs.model.as_str(), xxs.architecture.as_str(), xxs.shards.len(), xxs.sidecars.len()), (DEEPSEEK, "deepseek41", 6, 0));
        assert_eq!((xxs.shards[5].sha256.as_str(), xxs.verity[5].as_str(), xxs.shards[5].bytes),
                   (gsq.sidecars[0].asset.sha256.as_str(), gsq.sidecars[0].verity.as_str(), gsq.sidecars[0].asset.bytes));
        assert_eq!(xxs.shards[5].name, "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf");
        assert_eq!(verity_pins(ds("IQ2_XXS")).unwrap().len(), 6);
        // Unsloth's UD-IQ1_M: four shards of its own, and its Engram the same file, from ISTA's repository
        let ud = on_demand(ds("UD-IQ1_M")).unwrap();
        assert_eq!((ud.shards.len(), ud.sidecars.len()), (5, 0));
        assert_eq!((ud.shards[4].sha256.as_str(), ud.verity[4].as_str()), (gsq.sidecars[0].asset.sha256.as_str(), gsq.sidecars[0].verity.as_str()));
        let sources = on_demand_sources(ds("UD-IQ1_M")).unwrap();
        assert_eq!((sources[0].repository.as_str(), sources[4].repository.as_str()),
                   ("unsloth/DeepSeek-V4.1-Flash-1017-GGUF", "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"));
        assert!(xxs.verity.iter().all(|digest| digest.bytes().any(|digit| digit != b'0')));
        let (first, engram) = (on_demand_sources(ds("IQ2_XXS")).unwrap(), on_demand_sources(ds("IQ2_XS")).unwrap());
        assert!(first[..5].iter().all(|source| source.revision == "a419d74cd28d5b9f8c5e2b9077d11a609c17bc32"));
        assert_eq!((first[5].revision.as_str(), first[5].path.as_str()),
                   ("cb02422c900df489cea5833e8a92171224059b3c", "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"));
        assert!(engram.iter().all(|source| source.revision == "cb02422c900df489cea5833e8a92171224059b3c"));
        assert!(on_demand(qw("IQ3_S")).is_err() && on_demand(qw("UD-IQ1_M")).is_err() && bundle(ds("UD-IQ1_M")).is_err());
        // a provider's manifest pins only its own bundles
        assert!(on_demand(Id { model: DEEPSEEK, provider: ISTA, quant: "UD-IQ1_M" }).is_err());
    }

    /// Unsloth's Qwen3.8 UD-IQ1_M is a resident bundle fetched from its own repository and revision, with the official
    /// resources and the drafter every Qwen3.8 bundle shares; GSQ-RCO's still come from ISTA's.
    #[test]
    fn unsloths_qwen_bundle_comes_from_its_own_repository() {
        let pins = bundle(qw("UD-IQ1_M")).unwrap();
        assert_eq!(pins.shards.len(), 3);
        assert_eq!(architecture_of(qw("UD-IQ1_M")).unwrap(), "qwen4exp");
        assert_eq!(pins.drafter, bundle(qw("IQ2_XS")).unwrap().drafter);
        let unsloth = sources(qw("UD-IQ1_M")).unwrap();
        for source in &unsloth[..3] {
            assert_eq!((source.repository.as_str(), source.revision.as_str()), ("unsloth/Qwen3.8-Flash-Next-GGUF", "d96201194f103c906994f283f504209dc8d2a10c"));
            assert_eq!(source.path, format!("UD-IQ1_M/{}", source.asset.name));
        }
        assert!(unsloth[3..].iter().all(|source| source.repository == RESOURCE_REPOSITORY));
        assert!(sources(qw("IQ2_XS")).unwrap()[..2].iter().all(|source| source.repository == "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"));
        assert!(!qw("UD-IQ1_M").is_on_demand() && ds("UD-IQ1_M").is_on_demand());
        assert_ne!(shards(qw("UD-IQ1_M")).unwrap(), shards(ds("UD-IQ1_M")).unwrap());
    }

    /// A bundle is named by its model, provider and quantization; labels that leave one out name it when only one
    /// bundle has the rest. Earlier releases' names still name theirs.
    #[test]
    fn a_bundle_is_found_by_its_labels() {
        let unsloth_qwen = Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ1_M" };
        assert_eq!(Id::find(Some("qwen3.8-flash-next"), Some("unsloth"), "UD-IQ1_M"), Ok(unsloth_qwen));
        assert_eq!(Id::find(Some("Qwen3.8-Flash-Next"), None, "UD-IQ1_M"), Ok(unsloth_qwen));
        assert_eq!(Id::find(Some("DeepSeek-V4.1-Flash-1017"), None, "UD-IQ1_M").map(|id| id.provider), Ok(UNSLOTH));
        // IQ2_XS: ISTA-DASLab's, of both models; UD-IQ1_M: Unsloth's, of both
        for (quant, provider) in [("IQ2_XS", Some("ISTA-DASLab")), ("UD-IQ1_M", None), ("UD-IQ1_M", Some("Unsloth"))] {
            let both = Id::find(None, provider, quant).unwrap_err();
            assert!(both.contains("Qwen3.8-Flash-Next") && both.contains("DeepSeek-V4.1-Flash-1017") && both.contains("give the model"), "{both}");
        }
        assert_eq!(Id::find(None, None, "UD-IQ2_XXS"), Ok(Id { model: DEEPSEEK, provider: UNSLOTH, quant: "UD-IQ2_XXS" }));
        assert!(Id::find(Some("Qwen3.8-Flash-Next"), Some("ISTA-DASLab"), "UD-IQ1_M").unwrap_err().contains("unknown bundle UD-IQ1_M"));
        assert!(Id::find(Some("Qwen3.6-35B-A3B"), None, "Q4_K_M").unwrap_err().contains("unknown model"));
        // 0.1.3's names: the GSQ-RCO files' quantizations, the model without its release
        assert_eq!(Id::find(None, None, "GSQ-RCO-IQ2_XXS"), Ok(Id { model: DEEPSEEK, provider: ISTA, quant: "IQ2_XXS" }));
        assert_eq!(Id::find(Some("DeepSeek-V4.1-Flash"), None, "UD-IQ1_M"), Id::find(Some(DEEPSEEK), None, "UD-IQ1_M"));
        assert_eq!(models(), [MODEL, DEEPSEEK]);
        assert!(models().iter().all(|name| model(name).is_some()));
        assert_eq!(unsloth_qwen.to_string(), "Qwen3.8-Flash-Next / Unsloth / UD-IQ1_M");
        // a bundle's three labels name it alone
        for (index, id) in BUNDLES.iter().enumerate() {
            assert!(!BUNDLES[..index].contains(id), "{id}");
            assert_eq!(Id::find(Some(id.model), Some(id.provider), id.quant).as_ref(), Ok(id));
        }
    }

    /// Unsloth's Qwen3.8 bundles read their own table shard unless another table is named; without the shard, the most exact
    /// of the model's engram files in the bundle's directory, its engram directory or the one beside it. Bundles without a table
    /// shard of their own have none.
    #[test]
    fn a_bundle_reads_its_own_table_or_the_one_named() {
        let id = Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ1_M" };
        let offered: Vec<(String, bool)> = tables(id).unwrap().into_iter().map(|table| (table.precision, table.bundled)).collect();
        assert_eq!(offered, [("IQ4_NL".into(), true), ("BF16".into(), false), ("Q8_0".into(), false), ("Q5_1".into(), false)]);
        assert!(tables(qw("IQ2_XS")).unwrap().is_empty() && tables(ds("UD-IQ1_M")).unwrap().is_empty());
        let family = std::env::temp_dir().join(format!("straydog-bundle-engram-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&family);
        let root = family.join("Unsloth-UD-IQ1_M");
        std::fs::create_dir_all(family.join(ENGRAM_DIRECTORY)).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        let read = |named: Option<&str>| table_in_with(&root, id, named).map(|found| found.map(|(table, path)| (table.precision, path)));
        assert_eq!(read(None), Ok(None));
        // without its shard (set up with another table), the most exact file there: the model's engram directory, or its own
        let q5 = family.join(ENGRAM_DIRECTORY).join("Qwen3.8-Flash-Next-engram-Q5_1.gguf_file");
        std::fs::write(&q5, b"").unwrap();
        assert_eq!(read(None), Ok(Some(("Q5_1".into(), q5.clone()))));
        let q8 = root.join("Qwen3.8-Flash-Next-engram-Q8_0.gguf_file");
        std::fs::write(&q8, b"").unwrap();
        assert_eq!(read(None), Ok(Some(("Q8_0".into(), q8.clone()))));
        // its own shard, there, unless another is named
        let shard = bundle(id).unwrap().shards[2].name.clone();
        std::fs::write(root.join(&shard), b"").unwrap();
        assert_eq!(read(None), Ok(Some(("IQ4_NL".into(), root.join(&shard)))));
        assert_eq!(read(Some("q5_1")), Ok(Some(("Q5_1".into(), q5))));
        assert!(read(Some("BF16")).unwrap_err().contains("engram BF16: no such table"));
        // the files whose trees the server builds: the table read, in the last shard's place
        let files = verity_files_in(&root, id).unwrap();
        assert!(files.iter().any(|(path, ..)| path.ends_with(&shard)) && !files.iter().any(|(path, ..)| *path == q8), "{files:?}");
        std::fs::remove_dir_all(family).unwrap();
    }

    /// Every bundle's footprint is the tensors of its files: what the three kinds add up to is the
    /// files' bytes but for their headers and alignment, and the kinds are those the models have.
    #[test]
    fn every_bundle_has_a_footprint_that_adds_up_to_its_files() {
        for id in bundles() {
            let footprint = footprint(id).unwrap();
            let mut files: u64 = shards(id).unwrap().iter().map(|asset| asset.bytes).sum();
            if id.is_on_demand() {
                files += on_demand(id).unwrap().sidecars.iter().map(|sidecar| sidecar.asset.bytes).sum::<u64>();
            }
            let tensors = footprint.dense + footprint.experts + footprint.tables;
            assert!(tensors <= files && files - tensors < files / 500, "{id}: {tensors} of {files}");
            assert!(footprint.dense > 0 && footprint.experts > footprint.dense, "{id}");
        }
        // the state a token of context takes, which a model's bundles share: Qwen3.8's 13 GiB at 262144 (host
        // memory), DeepSeek-V4.1's compressed rows
        assert_eq!(footprint(qw("Q2_0")).unwrap().state_per_token * 262_144 / (1 << 30), 13);
        assert!((3000..3500).contains(&footprint(ds("IQ2_XXS")).unwrap().state_per_token));
        assert_eq!(footprint(ds("IQ2_XXS")).unwrap().state_per_token, footprint(ds("UD-IQ2_XXS")).unwrap().state_per_token);
        // the tables read from the drive: Engram (DeepSeek-V4.1's own file), the PLE table (Qwen3.8's)
        let engram = on_demand(ds("IQ2_XXS")).unwrap().shards[5].bytes;
        assert!(footprint(ds("IQ2_XXS")).unwrap().tables > engram / 100 * 99 && footprint(ds("IQ2_XXS")).unwrap().tables <= engram);
        assert_eq!(footprint(qw("Q2_0")).unwrap().tables, footprint(qw("IQ3_S")).unwrap().tables);
        assert!(footprint(Id { model: MODEL, provider: ISTA, quant: "Q8_0" }).unwrap_err().contains("no footprint"));
    }

    /// DSpark's drafter is pinned for every DeepSeek-V4.1 bundle, as Unsloth's file of a repository at a revision,
    /// and found in its directory or beside it; when found, its verity tree is among the bundle's (from the
    /// directory it lies in).
    #[test]
    fn the_dspark_drafter_is_pinned_for_every_deepseek_bundle_and_found_in_its_directories() {
        let source = dspark_source(ds("IQ2_XS")).unwrap().expect("a pinned DSpark");
        assert_eq!((source.asset.sha256.len(), source.verity.len(), source.revision.len()), (64, 64, 40));
        assert_eq!((source.repository.as_str(), source.asset.name.as_str()), ("unsloth/DeepSeek-V4.1-Flash-1017-GGUF", source.path.as_str()));
        assert!(!source.asset.name.contains('/') && source.asset.bytes > 7_000_000_000);
        assert!(dspark_source(qw("IQ2_XS")).unwrap().is_none());
        for quant in ["IQ2_XXS", "UD-IQ1_M", "UD-IQ2_XXS"] {
            assert_eq!(dspark_source(ds(quant)).unwrap().unwrap().asset, source.asset, "{quant}");
        }
        let pins = source.pins();
        assert_eq!((pins.architecture.as_str(), pins.shards.len(), pins.verity.len(), pins.sidecars.len()), ("dflash", 1, 1, 0));

        let family = std::env::temp_dir().join(format!("straydog-bundle-dspark-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&family);
        let root = family.join("GSQ-RCO-IQ2_XS");
        std::fs::create_dir_all(&root).unwrap();
        assert!(dspark_in(&root, ds("IQ2_XS")).unwrap().is_none());
        assert_eq!(verity_pins_in(&root, ds("IQ2_XS")).unwrap().len(), 5);
        // beside the model's directory (the quantizations' shared one)...
        std::fs::create_dir_all(family.join("dspark")).unwrap();
        std::fs::write(family.join("dspark").join(&source.asset.name), b"").unwrap();
        let (found, path) = dspark_in(&root, ds("IQ2_XS")).unwrap().unwrap();
        assert_eq!((found.asset, path.as_path()), (source.asset.clone(), family.join("dspark").join(&source.asset.name).as_path()));
        let files = verity_files_in(&root, ds("IQ2_XS")).unwrap();
        assert_eq!(files.len(), 6);
        assert_eq!(files[5].0, path);
        assert_eq!(files[5].1.name, source.asset.name);
        assert_eq!(files[0].0, root.join(&files[0].1.name));
        // ... or inside it, which comes first
        std::fs::create_dir_all(root.join("dspark")).unwrap();
        std::fs::write(root.join("dspark").join(&source.asset.name), b"").unwrap();
        assert_eq!(dspark_in(&root, ds("IQ2_XS")).unwrap().unwrap().1, root.join("dspark").join(&source.asset.name));
        // another quantization of the model finds the same file; no other model has one
        assert_eq!(dspark_in(&root, ds("UD-IQ2_XXS")).unwrap().unwrap().1, root.join("dspark").join(&source.asset.name));
        assert!(dspark_in(&root, qw("IQ2_XS")).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&family);
    }
}
