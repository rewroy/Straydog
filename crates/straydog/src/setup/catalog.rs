//! The models `straydog setup` offers: the pinned bundles, what each takes to download and
//! what a machine needs to run it. Everything comes from the pins compiled into the executable
//! and from the constants of the placement that will run the model (`arch::served`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use straydog_engine::arch::deepseek41::serve::CHECKPOINT_BYTES;
use straydog_engine::arch::served::{HOST_OTHER, HOST_SHARE, RESERVE};
use straydog_engine::bundle::{self, Asset, Footprint};
use straydog_engine::rows::DEFAULT_CACHE_BYTES;
use straydog_engine::verity::Geometry;

use super::machine::Machine;

pub const GIB: u64 = 1 << 30;

/// The context a server starts with unless told otherwise.
pub const DEFAULT_CONTEXT: u64 = 262_144;

/// The bundles in the order they are offered and recommended from: DeepSeek-V4.1-Flash's
/// recommended build and its other two, then Qwen3.8-Flash-Next from the most exact quantization to
/// the smallest.
pub const ORDER: [&str; 7] =
    ["GSQ-RCO-IQ2_XXS", "GSQ-RCO-IQ2_XS", "UD-IQ2_XXS", "IQ3_S", "IQ3_XXS", "IQ2_XS", "Q2_0"];

/// A file a model can do without, which `setup` offers to add.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extra {
    /// DeepSeek-V4.1's DSpark drafter.
    Dspark,
    /// The vision encoder: DeepSeek-V4.1's built from the official checkpoint's tensors, Qwen3.8's the GGUF repository's projector.
    Vision,
    /// Qwen3.8's MTP drafter, built from the official checkpoint's tensors.
    Mtp,
}

impl Extra {
    /// As in `--no-dspark`, `--no-vision` and `--no-mtp`.
    pub fn name(self) -> &'static str {
        match self {
            Extra::Dspark => "dspark",
            Extra::Vision => "vision",
            Extra::Mtp => "mtp",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sidecar {
    pub kind: Extra,
    pub label: &'static str,
    pub purpose: &'static str,
    /// The file it leaves.
    pub file: String,
    /// Bytes fetched.
    pub download: u64,
    /// Bytes the file takes, and its verity tree.
    pub kept: u64,
    pub tree: u64,
    /// The official tensors it is built from, on the disk until it is built.
    pub scratch: u64,
}

/// A file of the bundle itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct File {
    pub name: String,
    pub bytes: u64,
    /// Its verity tree's bytes (0: it has none).
    pub tree: u64,
}

/// One pinned bundle.
#[derive(Debug, Clone)]
pub struct Entry {
    pub model: String,
    pub quant: String,
    pub architecture: String,
    pub files: Vec<File>,
    pub footprint: Footprint,
    pub sidecars: Vec<Sidecar>,
}

fn tree(bytes: u64) -> u64 {
    Geometry::new(bytes).tree_bytes()
}

/// Every bundle that can be set up, in [`ORDER`].
pub fn pinned() -> Result<Vec<Entry>, String> {
    ORDER.iter().map(|quant| Entry::of(quant)).collect()
}

impl Entry {
    pub fn of(quant: &str) -> Result<Entry, String> {
        let trees: HashSet<String> = bundle::verity_pins(quant)?.into_iter().map(|(asset, _)| asset.name).collect();
        let (held, resources): (Vec<Asset>, Vec<Asset>) = if bundle::is_on_demand(quant) {
            let pins = bundle::on_demand(quant)?;
            (pins.shards.into_iter().chain(pins.sidecars.into_iter().map(|sidecar| sidecar.asset)).collect(), pins.resources)
        } else {
            let pins = bundle::bundle(quant)?;
            (pins.shards, pins.resources)
        };
        let files = held
            .into_iter()
            .chain(resources)
            .map(|asset| File { tree: if trees.contains(&asset.name) { tree(asset.bytes) } else { 0 }, name: asset.name, bytes: asset.bytes })
            .collect();
        let mut sidecars = Vec::new();
        if let Some(source) = bundle::dspark_source(quant)? {
            let bytes = source.asset.bytes;
            sidecars.push(Sidecar { kind: Extra::Dspark, label: "DSpark drafter", purpose: "faster replies", file: source.asset.name,
                                    download: bytes, kept: bytes, tree: tree(bytes), scratch: 0 });
        }
        if let Some(source) = bundle::vision_source(quant)? {
            let tensors = source.tensors.iter().map(|tensor| tensor.bytes).sum();
            sidecars.push(Sidecar { kind: Extra::Vision, label: "vision encoder", purpose: "image input", file: source.asset.name.clone(),
                                    download: tensors, kept: source.asset.bytes, tree: tree(source.asset.bytes), scratch: tensors });
        }
        if !bundle::is_on_demand(quant) {
            let source = bundle::drafter_source()?;
            let tensors = source.tensors.iter().map(|tensor| tensor.bytes).sum();
            sidecars.push(Sidecar { kind: Extra::Mtp, label: "MTP drafter", purpose: "faster replies", file: source.asset.name,
                                    download: tensors, kept: source.asset.bytes, tree: 0, scratch: tensors });
            let vision = bundle::qwen_vision_source()?.asset;
            sidecars.push(Sidecar { kind: Extra::Vision, label: "vision encoder", purpose: "image input", file: vision.name,
                                    download: vision.bytes, kept: vision.bytes, tree: 0, scratch: 0 });
        }
        Ok(Entry {
            model: bundle::model_of(quant)?,
            quant: quant.to_string(),
            architecture: bundle::architecture_of(quant)?,
            files,
            footprint: bundle::footprint(quant)?,
            sidecars,
        })
    }

    pub fn sidecar(&self, kind: Extra) -> Option<&Sidecar> {
        self.sidecars.iter().find(|sidecar| sidecar.kind == kind)
    }

    /// Bytes fetched in all: the bundle's files and the sidecars `chosen`.
    pub fn download_bytes(&self, chosen: &[Extra]) -> u64 {
        self.files.iter().map(|file| file.bytes).sum::<u64>() + self.chosen(chosen).map(|sidecar| sidecar.download).sum::<u64>()
    }

    /// Bytes the disk holds at the most while the bundle is provisioned: its files and their
    /// verity trees, the sidecars `chosen` with theirs, and the official tensors the sidecar
    /// being built is made from (built one at a time, each removed once it is).
    pub fn disk_bytes(&self, chosen: &[Extra]) -> u64 {
        let files: u64 = self.files.iter().map(|file| file.bytes + file.tree).sum();
        let kept: u64 = self.chosen(chosen).map(|sidecar| sidecar.kept + sidecar.tree).sum();
        files + kept + self.chosen(chosen).map(|sidecar| sidecar.scratch).max().unwrap_or(0)
    }

    fn chosen<'a>(&'a self, chosen: &'a [Extra]) -> impl Iterator<Item = &'a Sidecar> {
        self.sidecars.iter().filter(|sidecar| chosen.contains(&sidecar.kind))
    }

    /// The sidecars worth adding unless the person says no: all of them.
    pub fn default_extras(&self) -> Vec<Extra> {
        self.sidecars.iter().map(|sidecar| sidecar.kind).collect()
    }

    pub fn name(&self) -> String {
        format!("{} {}", self.model, self.quant)
    }
}

/// Where a bundle's files lie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Place {
    /// The bundle's directory.
    pub bundle: PathBuf,
    /// Where its verity trees go.
    pub data: PathBuf,
    /// The MTP drafter's directory (Qwen3.8).
    pub drafter: PathBuf,
}

/// The bytes of `path` already on the disk: the file, else the part of it downloaded (at most `bytes`).
fn on_disk(path: &Path, bytes: u64) -> u64 {
    let size = |path: &Path| std::fs::metadata(path).map_or(0, |metadata| metadata.len());
    let mut part = path.as_os_str().to_os_string();
    part.push(".part");
    if path.is_file() { size(path).min(bytes) } else { size(Path::new(&part)).min(bytes) }
}

impl Entry {
    /// Bytes of the bundle and the sidecars `chosen` that are on the disk already (complete, or
    /// as far as they were downloaded), trees included: what a fetch resumes from.
    pub fn present_bytes(&self, place: &Place, chosen: &[Extra]) -> u64 {
        let tree_of = |name: &str, bytes: u64| on_disk(&bundle::tree_path(&place.data, &Asset { name: name.into(), bytes: 0, sha256: String::new() }), tree(bytes));
        let mut present: u64 = self.files.iter().map(|file| on_disk(&place.bundle.join(&file.name), file.bytes) + if file.tree > 0 { tree_of(&file.name, file.bytes) } else { 0 }).sum();
        for sidecar in self.chosen(chosen) {
            let places = match sidecar.kind {
                Extra::Dspark => bundle::dspark_directories(&place.bundle),
                // (Qwen3.8's serves every quantization, from the folder above theirs)
                Extra::Vision => [Some(place.bundle.clone()), place.bundle.parent().map(Path::to_path_buf)].into_iter().flatten().collect(),
                Extra::Mtp => vec![place.drafter.clone()],
            };
            present += places.iter().map(|directory| on_disk(&directory.join(&sidecar.file), sidecar.kept)).max().unwrap_or(0);
            if sidecar.tree > 0 {
                present += tree_of(&sidecar.file, sidecar.kept);
            }
        }
        present
    }
}

/// How well a machine runs a bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fit {
    /// Everything a token reads is in VRAM.
    Vram,
    /// This many bytes of experts wait in host memory.
    Ram(u64),
    /// This many bytes of experts are read from the drive as tokens need them.
    Drive(u64),
    /// No GPU was found.
    NoGpu,
    /// What every token reads, with the placement's working memory, is more than the VRAM.
    TooLittleVram { needs: u64, has: u64 },
}

impl Fit {
    /// Whether the model runs here at all.
    pub fn runs(&self) -> bool {
        !matches!(self, Fit::NoGpu | Fit::TooLittleVram { .. })
    }

    fn rank(&self) -> Option<u8> {
        match self {
            Fit::Vram => Some(0),
            Fit::Ram(_) => Some(1),
            Fit::Drive(_) => Some(2),
            Fit::NoGpu | Fit::TooLittleVram { .. } => None,
        }
    }

    /// The verdict's name, for a program.
    pub fn class(&self) -> &'static str {
        match self {
            Fit::Vram => "vram",
            Fit::Ram(_) => "ram",
            Fit::Drive(_) => "drive",
            Fit::NoGpu => "no_gpu",
            Fit::TooLittleVram { .. } => "too_little_vram",
        }
    }

    /// What it means for the person choosing.
    pub fn note(&self, entry: &Entry) -> String {
        let gib = |bytes: &u64| format!("{:.0} GiB", *bytes as f64 / GIB as f64);
        match self {
            Fit::Vram => "all of it fits in VRAM".to_string(),
            Fit::Ram(bytes) if entry.architecture == "deepseek41" => format!("{} of its experts wait in RAM and move to the GPU as tokens need them", gib(bytes)),
            Fit::Ram(bytes) => format!("{} of its experts wait in RAM and the CPU computes them: slower", gib(bytes)),
            // (a twentieth of the experts is read once and then stays in the page cache)
            Fit::Drive(bytes) if *bytes < entry.footprint.experts / 20 => format!("{} of its experts are read from the drive: a little slower", if *bytes < GIB { "under 1 GiB".to_string() } else { gib(bytes) }),
            Fit::Drive(bytes) => format!("{} of its experts are read from the drive as tokens need them: slow", gib(bytes)),
            Fit::NoGpu => "needs a GPU".to_string(),
            Fit::TooLittleVram { needs, has } => format!("needs {} of VRAM; this machine has {}", gib(needs), gib(has)),
        }
    }
}

/// Where a model keeps the sequence state, which grows with the context.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// In VRAM, a copy on every GPU (DeepSeek-V4.1: each holds the compressed rows of its part of the attention).
    EveryGpu,
    /// In host memory (Qwen3.8).
    Host,
}

/// What a placement keeps besides the weights and the state, at the default prefill batch.
struct Working {
    /// VRAM the placement counts: this much once, and this much more on every GPU (each keeps its
    /// own rows and scratch), the drafter's weights included.
    vram: u64,
    per_gpu: u64,
    state: State,
    /// VRAM a pass's rows need on every GPU that the placement does not count (the arena it lends to
    /// the expert slots between passes): the model does not start without it.
    arena: u64,
    /// Host memory beside the experts' tier: the checkpoints.
    host: u64,
}

/// Hundredths of a GiB, as the measurements below are given.
const fn gib_hundredths(hundredths: u64) -> u64 {
    hundredths * (GIB / 100)
}

fn working(architecture: &str) -> Working {
    match architecture {
        // DeepSeek-V4.1, with the DSpark drafter. What the placement counted beyond the dense weights
        // and the expert slots at 262144 tokens: 4.74 GiB on one GPU (GSQ-RCO-IQ2_XXS at --vram 16:
        // 15.2 GiB counted, 590 slots), 6.96 GiB on two (59.4 GiB, 5832 slots) and 7.14 GiB on two
        // with GSQ-RCO-IQ2_XS (5147 slots): 2.34 GiB and 2.40 GiB a GPU, of which the state, 3200
        // bytes a token, is 0.78 GiB. It started at --vram 14 and 15 and not at 13 (and 12, 10), 1.4
        // GiB more than it counted, for the rows of a prefill pass of 4096 tokens.
        "deepseek41" => Working { vram: gib_hundredths(234), per_gpu: gib_hundredths(162), state: State::EveryGpu, arena: gib_hundredths(140), host: CHECKPOINT_BYTES },
        // Qwen3.8 keeps its sequence state in host memory; the rows and buffers of its passes: 6.26
        // GiB on two GPUs with IQ3_S (57.3 GiB counted); Q2_0 started on one GPU at --vram 8
        _ => Working { vram: 0, per_gpu: gib_hundredths(290), state: State::Host, arena: 0, host: 0 },
    }
}

impl Working {
    fn vram(&self, gpus: usize, state: u64) -> u64 {
        let gpus = gpus as u64;
        self.vram + self.per_gpu * gpus + match self.state { State::EveryGpu => state * gpus, State::Host => 0 }
    }

    fn host(&self, state: u64) -> u64 {
        self.host + if self.state == State::Host { state } else { 0 }
    }
}

/// The contexts setup chooses among, the server's default first: it takes the largest at which
/// the model starts, so that a model that does not fit at the default is given less context, not left out.
pub const CONTEXTS: [u64; 5] = [DEFAULT_CONTEXT, 131_072, 65_536, 32_768, 16_384];

/// The margin an automatic VRAM budget leaves of a GPU's memory.
fn margin(total: u64) -> u64 {
    (if cfg!(windows) { 2u64 << 30 } else { 512u64 << 20 }).max(total / 100 * 3)
}

/// The VRAM the placement counts for a bundle that is not the experts, with `context` tokens: the
/// weights every token reads, the working memory, the state, and `RESERVE` on each of `gpus` GPUs.
fn counted(entry: &Entry, gpus: usize, context: u64) -> u64 {
    entry.footprint.dense + working(&entry.architecture).vram(gpus, entry.footprint.state_per_token * context) + gpus as u64 * RESERVE
}

/// The VRAM a bundle needs at the least to start with `context` tokens: what [`counted`] gives and
/// the rows a pass needs beside it (the margin of an automatic budget not counted: the caller adds
/// it for the GPUs it has).
pub fn min_vram(entry: &Entry, gpus: usize, context: u64) -> u64 {
    counted(entry, gpus, context) + gpus as u64 * working(&entry.architecture).arena
}

/// Whether `machine` runs `entry` with `context` tokens and how: the placement's own rules over the
/// footprint. The weights every token reads, the working memory, the state and `RESERVE` must fit
/// the GPUs' memory less the margin of an automatic budget (and the rows of a pass beside them); the
/// experts take what VRAM has left of that, then the host tier (85% of the RAM less the row cache,
/// 2 GiB and what the state and the checkpoints take of host memory); the rest is read from the drive.
pub fn fit(entry: &Entry, machine: &Machine, context: u64) -> Fit {
    if machine.gpus.is_empty() {
        return Fit::NoGpu;
    }
    let work = working(&entry.architecture);
    let state = entry.footprint.state_per_token * context;
    let total: u64 = machine.gpus.iter().map(|gpu| gpu.total).sum();
    let margins: u64 = machine.gpus.iter().map(|gpu| margin(gpu.total)).sum();
    let needs = min_vram(entry, machine.gpus.len(), context);
    if total < needs + margins {
        return Fit::TooLittleVram { needs: needs + margins, has: total };
    }
    // (the experts' slots take what the placement does not count: the rows' arena is lent to them)
    let room = total - margins - counted(entry, machine.gpus.len(), context);
    if entry.footprint.experts <= room {
        return Fit::Vram;
    }
    let beyond = entry.footprint.experts - room;
    let host = ((machine.ram as f64 * HOST_SHARE) as u64).saturating_sub(DEFAULT_CACHE_BYTES + HOST_OTHER + work.host(state));
    if beyond <= host { Fit::Ram(beyond) } else { Fit::Drive(beyond - host) }
}

/// The largest context in [`CONTEXTS`] at which `entry` runs on `machine`, with how; when it runs at
/// none, the verdict at the smallest (what the least it needs is).
pub fn fit_at_best_context(entry: &Entry, machine: &Machine) -> (u64, Fit) {
    for &context in &CONTEXTS {
        let verdict = fit(entry, machine, context);
        if verdict.runs() {
            return (context, verdict);
        }
    }
    let smallest = CONTEXTS[CONTEXTS.len() - 1];
    (smallest, fit(entry, machine, smallest))
}

/// The models of `entries`, in the order they first appear.
pub fn models(entries: &[Entry]) -> Vec<&str> {
    let mut models: Vec<&str> = Vec::new();
    for entry in entries {
        if !models.contains(&entry.model.as_str()) {
            models.push(&entry.model);
        }
    }
    models
}

/// The bundle to start from, by index into `entries` and their `fits`: the best class (VRAM,
/// RAM, drive) that the first model reaching VRAM or RAM has, its first bundle in that class; when
/// no model does, the bundle reading the least from the drive.
pub fn recommend(entries: &[Entry], fits: &[Fit]) -> Option<usize> {
    let rank = |index: usize| fits[index].rank();
    for model in models(entries) {
        let of_model: Vec<usize> = (0..entries.len()).filter(|&index| entries[index].model == model).collect();
        let best = of_model.iter().filter_map(|&index| rank(index)).min();
        if let Some(best @ (0 | 1)) = best {
            return of_model.into_iter().find(|&index| rank(index) == Some(best));
        }
    }
    (0..entries.len()).filter(|&index| rank(index).is_some()).min_by_key(|&index| match fits[index] { Fit::Drive(bytes) => bytes, _ => 0 })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::setup::machine::Gpu;

    /// A machine of `gpus` GPUs of `bytes` each and `ram` bytes of memory.
    pub fn machine(gpus: usize, bytes: u64, ram: u64) -> Machine {
        let gpus = (0..gpus).map(|ordinal| Gpu { ordinal, name: "GPU".into(), total: bytes, free: bytes }).collect();
        Machine { backend: "ROCm/HIP", gpus, gpu_error: None, ram }
    }

    /// The owner's: two R9700s (34208743424 bytes each) and the 134411186176 bytes of RAM the worker measures.
    pub fn doggybox() -> Machine {
        machine(2, 34_208_743_424, 134_411_186_176)
    }

    fn entry(quant: &str) -> Entry {
        Entry::of(quant).unwrap()
    }

    #[test]
    fn every_pinned_bundle_is_offered_once() {
        let mut named: Vec<&str> = ORDER.to_vec();
        named.sort_unstable();
        let mut pinned: Vec<&str> = bundle::quantizations().collect();
        pinned.sort_unstable();
        assert_eq!(named, pinned);
        let entries = pinned_entries();
        assert_eq!(models(&entries), ["DeepSeek-V4.1-Flash", "Qwen3.8-Flash-Next"]);
    }

    fn pinned_entries() -> Vec<Entry> {
        pinned().unwrap()
    }

    #[test]
    fn a_bundle_downloads_and_needs_what_its_pins_say() {
        let xxs = entry("GSQ-RCO-IQ2_XXS");
        // six shards (the Engram file the sixth) and no resources: README's 248.8 GB
        assert_eq!(xxs.files.len(), 6);
        assert_eq!(xxs.download_bytes(&[]), xxs.files.iter().map(|file| file.bytes).sum::<u64>());
        assert!((248_700_000_000..248_900_000_000).contains(&xxs.download_bytes(&[])));
        assert_eq!(xxs.default_extras(), [Extra::Dspark, Extra::Vision]);
        let dspark = xxs.sidecar(Extra::Dspark).unwrap();
        assert_eq!((dspark.download, dspark.scratch), (7_976_914_240, 0));
        let vision = xxs.sidecar(Extra::Vision).unwrap();
        // the vision encoder is built from the official tensors, which wait on the disk until it is
        assert!(vision.download > 0 && vision.scratch == vision.download && vision.kept == 970_618_336);
        assert_eq!(xxs.download_bytes(&xxs.default_extras()), xxs.download_bytes(&[]) + dspark.download + vision.download);
        // on the disk: the files, their verity trees (1/128 of them), the extras and their trees, and the tensors being built from
        let trees: u64 = xxs.files.iter().map(|file| file.tree).sum();
        assert!(trees > xxs.download_bytes(&[]) / 130 && trees < xxs.download_bytes(&[]) / 120);
        let with_dspark = xxs.disk_bytes(&[Extra::Dspark]);
        assert_eq!(with_dspark, xxs.disk_bytes(&[]) + dspark.kept + dspark.tree);
        assert_eq!(xxs.disk_bytes(&[Extra::Vision]), xxs.disk_bytes(&[]) + vision.kept + vision.tree + vision.scratch);
        // Qwen3.8 has the MTP drafter and the vision encoder, and no DSpark
        assert_eq!(entry("Q2_0").default_extras(), [Extra::Mtp, Extra::Vision]);
        assert_eq!((entry("Q2_0").sidecar(Extra::Vision).unwrap().download, entry("Q2_0").sidecar(Extra::Vision).unwrap().tree), (907_543_008, 0));
        assert_eq!(entry("UD-IQ2_XXS").default_extras(), [Extra::Vision]);
        // every bundle: its tokenizer files come with it and are small
        for quant in ORDER {
            let entry = entry(quant);
            assert!(entry.download_bytes(&[]) > 30_000_000_000, "{quant}");
            assert!(entry.footprint.dense + entry.footprint.experts + entry.footprint.tables <= entry.download_bytes(&[]), "{quant}");
        }
    }

    #[test]
    fn what_is_on_the_disk_already_is_not_needed_again() {
        let root = std::env::temp_dir().join(format!("straydog-catalog-present-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let place = Place { bundle: root.join("bundle"), data: root.join("bundle/.straydog"), drafter: root.join("mtp") };
        std::fs::create_dir_all(&place.bundle).unwrap();
        let xxs = entry("GSQ-RCO-IQ2_XXS");
        let everything = [Extra::Dspark, Extra::Vision];
        assert_eq!(xxs.present_bytes(&place, &everything), 0);
        // a part of a download counts for what arrived, a complete file for its size (a file larger than pinned for the pin)
        std::fs::write(place.bundle.join(format!("{}.part", xxs.files[1].name)), vec![0u8; 1000]).unwrap();
        assert_eq!(xxs.present_bytes(&place, &everything), 1000);
        std::fs::write(place.bundle.join(&xxs.files[0].name), vec![0u8; 5000]).unwrap();
        assert_eq!(xxs.present_bytes(&place, &everything), 6000);
        // the DSpark drafter is found where the server looks: in the bundle's dspark folder or beside it
        let dspark = xxs.sidecar(Extra::Dspark).unwrap();
        std::fs::create_dir_all(root.join("dspark")).unwrap();
        std::fs::write(root.join("dspark").join(&dspark.file), vec![0u8; 700]).unwrap();
        assert_eq!(xxs.present_bytes(&place, &everything), 6700);
        assert_eq!(xxs.present_bytes(&place, &[]), 6000);
        // a verity tree built already
        let tree = bundle::tree_path(&place.data, &Asset { name: xxs.files[0].name.clone(), bytes: 0, sha256: String::new() });
        std::fs::create_dir_all(tree.parent().unwrap()).unwrap();
        std::fs::write(&tree, vec![0u8; 300]).unwrap();
        assert_eq!(xxs.present_bytes(&place, &[]), 6300);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// The fit rule against what the placement did on the owner's machine (Homelab's logs, 2026-10-28):
    /// GSQ-RCO-IQ2_XXS held 5832 expert slots in VRAM and had a 102.2 GiB host tier, GSQ-RCO-IQ2_XS 5147
    /// slots and the same tier; the experts' bytes per slot are the average.
    #[test]
    fn the_fit_rule_agrees_with_the_placement_on_two_r9700s() {
        let machine = doggybox();
        for (quant, slots) in [("GSQ-RCO-IQ2_XXS", 5832u64), ("GSQ-RCO-IQ2_XS", 5147)] {
            let entry = entry(quant);
            let in_vram = slots * entry.footprint.experts / 15360;
            let Fit::Ram(beyond) = fit(&entry, &machine, DEFAULT_CONTEXT) else { panic!("{quant}: {:?}", fit(&entry, &machine, DEFAULT_CONTEXT)) };
            let placed = entry.footprint.experts - in_vram;
            // within 5% of the experts the placement left outside VRAM, and inside the host tier it had (102.2 GiB)
            assert!(beyond.abs_diff(placed) < placed / 20, "{quant}: rule {beyond}, placement {placed}");
            assert!(beyond < (102.2 * GIB as f64) as u64);
        }
        // UD-IQ2_XXS has 157 GB of experts: past VRAM and the tier, the drive serves the rest
        assert!(matches!(fit(&entry("UD-IQ2_XXS"), &machine, DEFAULT_CONTEXT), Fit::Drive(_)));
        // Qwen3.8 is in VRAM whole at the full context (the placement said "fully resident" and
        // "0.0 GiB host tier"), and keep it
        for quant in ["Q2_0", "IQ3_S"] {
            assert_eq!(fit_at_best_context(&entry(quant), &machine), (DEFAULT_CONTEXT, Fit::Vram), "{quant}");
        }
        let entries = pinned_entries();
        let fits: Vec<Fit> = entries.iter().map(|entry| fit(entry, &machine, DEFAULT_CONTEXT)).collect();
        assert_eq!(recommend(&entries, &fits).map(|index| entries[index].quant.as_str()), Some("GSQ-RCO-IQ2_XXS"));
    }

    #[test]
    fn a_small_machine_is_offered_what_it_can_run_and_told_what_is_slow() {
        // one 12 GB GPU and 32 GB of RAM
        let small = machine(1, 12 << 30, 32 << 30);
        let entries = pinned_entries();
        let fits: Vec<(u64, Fit)> = entries.iter().map(|entry| fit_at_best_context(entry, &small)).collect();
        for (entry, (context, fit)) in entries.iter().zip(&fits) {
            eprintln!("{:<22} {:<16} {:>6} {:?}  {}", entry.model, entry.quant, context, fit, fit.note(entry));
        }
        let of = |quant: &str| fits[entries.iter().position(|entry| entry.quant == quant).unwrap()].clone();
        // DeepSeek-V4.1's weights and working memory do not fit 12 GB whatever the context: not offered
        for quant in ["GSQ-RCO-IQ2_XXS", "GSQ-RCO-IQ2_XS", "UD-IQ2_XXS"] {
            assert!(matches!(of(quant).1, Fit::TooLittleVram { .. }), "{quant}");
        }
        // the Qwen3.8 bundles run, keeping the context they start with (the placement ran IQ3_S
        // partially resident at --vram 12, K/V in host memory), and read from the drive
        for quant in ["IQ3_S", "IQ2_XS", "Q2_0"] {
            assert!(matches!(of(quant), (DEFAULT_CONTEXT, Fit::Drive(_)) | (_, Fit::Drive(_))), "{quant}");
        }
        let recommended = recommend(&entries, &fits.iter().map(|(_, fit)| fit.clone()).collect::<Vec<_>>()).unwrap();
        assert_eq!(entries[recommended].quant, "Q2_0");
    }

    #[test]
    fn deepseeks_state_is_small_and_a_copy_on_each_gpu() {
        let ds = entry("GSQ-RCO-IQ2_XXS");
        assert!(min_vram(&ds, 1, DEFAULT_CONTEXT) - min_vram(&ds, 1, 16_384) < GIB);
        assert!(min_vram(&ds, 2, DEFAULT_CONTEXT) - min_vram(&ds, 1, DEFAULT_CONTEXT) > GIB);
    }

    #[test]
    fn nothing_runs_without_a_gpu_or_with_too_little_memory() {
        let none = machine(0, 0, 64 << 30);
        assert_eq!(fit_at_best_context(&entry("Q2_0"), &none).1, Fit::NoGpu);
        let tiny = machine(1, 4 << 30, 64 << 30);
        let (_, verdict) = fit_at_best_context(&entry("Q2_0"), &tiny);
        let Fit::TooLittleVram { needs, has } = verdict else { panic!("{verdict:?}") };
        assert!(needs > has && has == 4 << 30);
        let entries = pinned_entries();
        let fits: Vec<Fit> = entries.iter().map(|entry| fit_at_best_context(entry, &tiny).1).collect();
        assert_eq!(recommend(&entries, &fits), None);
    }

    #[test]
    fn the_recommendation_is_the_best_class_of_the_first_model_that_runs_well() {
        let entries = pinned_entries();
        // all DeepSeek on the drive, Qwen3.8 in RAM for one bundle and in VRAM for another: Qwen3.8's in VRAM
        let mut fits: Vec<Fit> = vec![Fit::Drive(5), Fit::Drive(4), Fit::Drive(3), Fit::Ram(9), Fit::Ram(8), Fit::Vram, Fit::Vram, Fit::Vram];
        assert_eq!(recommend(&entries, &fits), Some(5));
        // DeepSeek in RAM wins over Qwen3.8 in VRAM: the first model that runs well, its first bundle of the best class
        fits[1] = Fit::Ram(1);
        fits[0] = Fit::Ram(2);
        assert_eq!(recommend(&entries, &fits), Some(0));
        // DeepSeek in VRAM for a later bundle only
        fits[0] = Fit::Drive(1);
        fits[2] = Fit::Vram;
        assert_eq!(recommend(&entries, &fits), Some(2));
        // nothing runs well: the one reading least from the drive
        let slow: Vec<Fit> = vec![Fit::Drive(5), Fit::Drive(4), Fit::NoGpu, Fit::Drive(30), Fit::Drive(3), Fit::Drive(9), Fit::Drive(7), Fit::Drive(8)];
        assert_eq!(recommend(&entries, &slow), Some(4));
    }
}
