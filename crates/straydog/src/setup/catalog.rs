//! The models `straydog setup` offers: the pinned bundles, what each takes to download and
//! what a machine needs to run it. Everything comes from the pins compiled into the executable
//! and from the constants of the placement that will run the model (`arch::served`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use straydog_engine::arch::deepseek41::serve::CHECKPOINT_BYTES;
use straydog_engine::arch::served::{HOST_OTHER, HOST_SHARE, RESERVE};
use straydog_engine::bundle::{self, Asset, Footprint, Id, DEEPSEEK, ISTA, MODEL, UNSLOTH};
use straydog_engine::model::{checkpoint_bytes, partial_slot_bytes, SCARCE_ROW_CACHE, STARTING_KV};
use straydog_engine::rows::{auto_budget as row_cache_budget, DEFAULT_CACHE_BYTES};
use straydog_engine::verity::Geometry;

use super::machine::{Gpu, Machine};
use crate::worker::{auto_budget, margin_on};

pub const GIB: u64 = 1 << 30;

/// The context a server starts with unless told otherwise.
pub const DEFAULT_CONTEXT: u64 = 262_144;

/// The bundles in the order they are offered and recommended from, Unsloth's first (the owner's choice, 2026-11-05):
/// DeepSeek-V4.1-Flash-1017's UD-IQ1_M (the size of ISTA-DASLab's IQ2_XXS) and UD-IQ2_XXS, then ISTA-DASLab's two; then
/// Qwen3.8-Flash-Next's, Unsloth's and then ISTA-DASLab's, each from the most exact quantization to the smallest.
pub const ORDER: [Id; 12] = [
    Id { model: DEEPSEEK, provider: UNSLOTH, quant: "UD-IQ1_M" },
    Id { model: DEEPSEEK, provider: UNSLOTH, quant: "UD-IQ2_XXS" },
    Id { model: DEEPSEEK, provider: ISTA, quant: "IQ2_XXS" },
    Id { model: DEEPSEEK, provider: ISTA, quant: "IQ2_XS" },
    Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ3_XXS" },
    Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ2_XS" },
    Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ1_M" },
    Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ1_S" },
    Id { model: MODEL, provider: ISTA, quant: "IQ3_S" },
    Id { model: MODEL, provider: ISTA, quant: "IQ3_XXS" },
    Id { model: MODEL, provider: ISTA, quant: "IQ2_XS" },
    Id { model: MODEL, provider: ISTA, quant: "Q2_0" },
];

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
    /// Its model, provider and quantization.
    pub id: Id,
    /// The engram table it is set up with instead of its own (its precision), see [`Entry::with_table`].
    pub table: Option<String>,
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
    ORDER.into_iter().map(Entry::of).collect()
}

impl Entry {
    pub fn of(id: Id) -> Result<Entry, String> {
        let trees: HashSet<String> = bundle::verity_pins(id)?.into_iter().map(|(asset, _)| asset.name).collect();
        let (held, resources): (Vec<Asset>, Vec<Asset>) = if id.is_on_demand() {
            let pins = bundle::on_demand(id)?;
            (pins.shards.into_iter().chain(pins.sidecars.into_iter().map(|sidecar| sidecar.asset)).collect(), pins.resources)
        } else {
            let pins = bundle::bundle(id)?;
            (pins.shards, pins.resources)
        };
        let files = held
            .into_iter()
            .chain(resources)
            .map(|asset| File { tree: if trees.contains(&asset.name) { tree(asset.bytes) } else { 0 }, name: asset.name, bytes: asset.bytes })
            .collect();
        let mut sidecars = Vec::new();
        if let Some(source) = bundle::dspark_source(id)? {
            let bytes = source.asset.bytes;
            sidecars.push(Sidecar { kind: Extra::Dspark, label: "DSpark drafter", purpose: "faster replies", file: source.asset.name,
                                    download: bytes, kept: bytes, tree: tree(bytes), scratch: 0 });
        }
        if let Some(source) = bundle::vision_source(id)? {
            let tensors = source.tensors.iter().map(|tensor| tensor.bytes).sum();
            sidecars.push(Sidecar { kind: Extra::Vision, label: "vision encoder", purpose: "image input", file: source.asset.name.clone(),
                                    download: tensors, kept: source.asset.bytes, tree: tree(source.asset.bytes), scratch: tensors });
        }
        if !id.is_on_demand() {
            let source = bundle::drafter_source()?;
            let tensors = source.tensors.iter().map(|tensor| tensor.bytes).sum();
            sidecars.push(Sidecar { kind: Extra::Mtp, label: "MTP drafter", purpose: "faster replies", file: source.asset.name,
                                    download: tensors, kept: source.asset.bytes, tree: 0, scratch: tensors });
            let vision = bundle::qwen_vision_source()?.asset;
            sidecars.push(Sidecar { kind: Extra::Vision, label: "vision encoder", purpose: "image input", file: vision.name,
                                    download: vision.bytes, kept: vision.bytes, tree: 0, scratch: 0 });
        }
        Ok(Entry { id, table: None, architecture: bundle::architecture_of(id)?, files, footprint: bundle::footprint(id)?, sidecars })
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
        self.id.to_string()
    }

    /// The engram tables it may be set up with: (precision, bytes), its own first.
    pub fn tables(&self) -> Vec<(String, u64)> {
        bundle::tables(self.id).unwrap_or_default().into_iter().map(|table| (table.precision, table.asset.bytes)).collect()
    }

    /// The entry set up with the engram table of `precision` instead of its own: that file (kept in the model's `engram`
    /// directory, which its bundles share) in place of its last shard. Its own table's precision leaves it as it is.
    pub fn with_table(&self, precision: &str) -> Result<Entry, String> {
        let Some(table) = straydog_provision::chosen_table(self.id, precision)? else { return Ok(self.clone()) };
        let shard = bundle::bundle(self.id)?.shards.last().cloned().ok_or("Straydog bundle without shards")?;
        let mut entry = self.clone();
        entry.files.retain(|file| file.name != shard.name);
        entry.files.push(File { tree: tree(table.asset.bytes), name: table.asset.name, bytes: table.asset.bytes });
        entry.table = Some(table.precision);
        // (the table's tensor at its precision: what is read from the drive by row)
        entry.footprint.tables = table.asset.bytes;
        Ok(entry)
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
        // (an engram table lies in the model's engram directory, or the bundle's)
        let path = |name: &str| bundle::engram_directories(&place.bundle).into_iter().map(|directory| directory.join(name))
            .find(|path| path.is_file()).unwrap_or_else(|| place.bundle.join(name));
        let mut present: u64 = self.files.iter().map(|file| on_disk(&path(&file.name), file.bytes) + if file.tree > 0 { tree_of(&file.name, file.bytes) } else { 0 }).sum();
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

/// Bytes in GiB to a tenth: on a small card a tenth is what decides.
pub fn tenths(bytes: u64) -> String {
    let tenths = format!("{:.1}", bytes as f64 / GIB as f64);
    format!("{} GiB", tenths.strip_suffix(".0").unwrap_or(&tenths))
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
    /// `auto`'s budget would not hold what every token reads and the placement's working memory: the free
    /// VRAM that would, its margin included, and the free VRAM there is.
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
            // Qwen3.8's K/V grows from 32768 positions into the memory the banks had; what the working set cannot keep is a page's read away.
            Fit::Drive(0) => "its experts wait in RAM and the CPU computes them: slower, and more so once a long conversation's K/V takes their room".to_string(),
            Fit::Drive(bytes) if entry.architecture != "deepseek41" => format!("{} of its experts do not stay in RAM: the CPU computes them from the drive's copy, slower at first and on a change of topic", if *bytes < GIB { "under 1 GiB".to_string() } else { gib(bytes) }),
            // (a twentieth of the experts is read once and then stays in the page cache)
            Fit::Drive(bytes) if *bytes < entry.footprint.experts / 20 => format!("{} of its experts are read from the drive: a little slower", if *bytes < GIB { "under 1 GiB".to_string() } else { gib(bytes) }),
            Fit::Drive(bytes) => format!("{} of its experts are read from the drive as tokens need them: slow", gib(bytes)),
            Fit::NoGpu => "needs a GPU".to_string(),
            Fit::TooLittleVram { needs, has } => format!("needs {} of VRAM free, with what `auto` leaves for the desktop; this machine has {} for it", tenths(*needs), tenths(*has)),
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
        // Qwen3.8 with every expert resident (a partial placement counts otherwise: `partial_room`) keeps its sequence
        // state in host memory; the rows and buffers of its passes: 6.26 GiB on two GPUs with IQ3_S (57.3 GiB counted)
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

/// The VRAM the placement counts for a bundle that is not the experts, with `context` tokens: the
/// weights every token reads, the working memory, the state, and `RESERVE` on each of `gpus` GPUs.
/// (Qwen3.8: with every expert resident.)
fn counted(entry: &Entry, gpus: usize, context: u64) -> u64 {
    entry.footprint.dense + working(&entry.architecture).vram(gpus, entry.footprint.state_per_token * context) + gpus as u64 * RESERVE
}

/// A Qwen3.8 prefill batch below 48 GiB of budgets (`worker::plan`).
const PARTIAL_BATCH: u64 = 1024;

/// What a partial placement pins of a bundle's `dense` bytes (`Weights::dense_requirements`): not the HyperConnections' and the PLE
/// layer's BF16 matrices' 626442240 bytes they save going up as Q8_0, nor the token embeddings, kept on the host: 273152000 bytes
/// in IQ3_XXS, whose 3216268800 bytes pinned were read from its header; for bundles whose headers were not at hand, their
/// 635699200 weights at 2 bits at the least, which may count up to 0.11 GiB too much. Unsloth's keep their HyperConnections in
/// Q8_0 already and their token embeddings in IQ4_XS (337715200 bytes; UD-IQ3_XXS's in Q6_K, 521475200), and save 30720000
/// bytes of the PLE layer's BF16 matrices: from their headers, and the pinned bytes Weights counted for UD-IQ1_S, UD-IQ1_M and
/// UD-IQ2_XS (3802437120, 4014773760 and 4368668160).
fn pinned_dense(entry: &Entry) -> u64 {
    if entry.id.model == MODEL && entry.id.provider == UNSLOTH {
        let embeddings = if entry.id.quant == "UD-IQ3_XXS" { 521_475_200 } else { 337_715_200 };
        return entry.footprint.dense.saturating_sub(embeddings + 30_720_000);
    }
    let embeddings = if entry.id == (Id { model: MODEL, provider: ISTA, quant: "IQ3_XXS" }) { 273_152_000 } else { 635_699_200 / 4 };
    entry.footprint.dense.saturating_sub(626_442_240 + embeddings)
}

/// The largest expert of a bundle, which sizes the staging slots: a third over the average (IQ3_XXS: 2329600 bytes, 1746133 on average).
fn largest_expert(entry: &Entry) -> u64 {
    entry.footprint.experts / (48 * 512) / 3 * 4
}

/// The VRAM `budgets` leave a partially resident Qwen3.8's expert slots with `context` tokens, by the
/// placement's own arithmetic (`straydog_engine::model::partial_slot_bytes`), the pinned weights shared out
/// over the GPUs; None if it would not start, or could not grow its K/V to the context.
fn partial_room(entry: &Entry, budgets: &[u64], context: u64) -> Option<u64> {
    let dense = pinned_dense(entry) / budgets.len() as u64;
    budgets.iter().map(|&budget| partial_slot_bytes(dense, largest_expert(entry), context, PARTIAL_BATCH, budget)).sum()
}

/// The VRAM a bundle needs at the least to start with `context` tokens, as `--vram` gives it (the margin of
/// an automatic budget not counted). DeepSeek-V4.1: what [`counted`] gives and the rows a pass needs beside
/// it. Qwen3.8: what a partial placement takes, its slots few. Its K/V, in host memory, costs VRAM only for
/// the indexers, whose whole context a smaller budget cannot grow to: at 16384 tokens the least is 0.4 GiB
/// under 262144's.
pub fn min_vram(entry: &Entry, gpus: usize, context: u64) -> u64 {
    if entry.architecture == "deepseek41" {
        return counted(entry, gpus, context) + gpus as u64 * working(&entry.architecture).arena;
    }
    const STEP: u64 = 16 << 20;
    let least = (1..(1 << 16)).map(|steps| steps * STEP).find(|&budget| partial_room(entry, &vec![budget; gpus], context).is_some());
    gpus as u64 * least.unwrap_or(u64::MAX / 256)
}

/// A GPU's free memory as `auto` will find it: a share of up to a quarter that another program keeps is
/// left to it, and a job holding more is waited for (`worker::budgets`), so that it says nothing of what will be free.
fn free_for_auto(gpu: &Gpu) -> u64 {
    if gpu.free >= gpu.total / 4 * 3 { gpu.free } else { gpu.total }
}

/// Whether `machine` runs `entry` with `context` tokens and how.
pub fn fit(entry: &Entry, machine: &Machine, context: u64) -> Fit {
    fit_on(cfg!(windows), entry, machine, context)
}

/// [`fit`] by the placement's own rules over the footprint, for the budgets `auto` takes on `machine`
/// under Windows or elsewhere.
///
/// DeepSeek-V4.1: the weights every token reads, the working memory, the state and `RESERVE` must fit the
/// budgets (and the rows of a pass beside them); the experts take what VRAM has left of that, then the host
/// tier (85% of the RAM less the row cache, 2 GiB and what the state and the checkpoints take of host
/// memory); the rest is read from the drive.
///
/// Qwen3.8: every expert resident as for DeepSeek-V4.1, else a partial placement: its slots hold what the budgets
/// have left ([`partial_room`]), and the banks are mapped files the working set keeps as far as it may
/// (`host::keep_resident`, 85% of the RAM less 2 GiB and what is page-locked, in which the row cache is). Its K/V
/// starts at 32768 positions: the experts are in RAM whatever a conversation does if they fit beside the whole
/// context's K/V, else what they lack beside the starting K/V is read from the drive.
fn fit_on(windows: bool, entry: &Entry, machine: &Machine, context: u64) -> Fit {
    if machine.gpus.is_empty() {
        return Fit::NoGpu;
    }
    let work = working(&entry.architecture);
    let state = entry.footprint.state_per_token * context;
    let gpus = machine.gpus.len();
    let budgets: Vec<u64> = machine.gpus.iter().map(|gpu| auto_budget(windows, free_for_auto(gpu), gpu.total)).collect();
    let budget: u64 = budgets.iter().sum();
    let has: u64 = machine.gpus.iter().map(free_for_auto).sum();
    let needs = min_vram(entry, gpus, context) + machine.gpus.iter().map(|gpu| margin_on(windows, gpu.total)).sum::<u64>();
    let share = host_share(machine);
    // (the experts' slots take what the placement does not count: the rows' arena is lent to them)
    let whole = counted(entry, gpus, context);
    if entry.architecture == "deepseek41" {
        if budget < min_vram(entry, gpus, context) {
            return Fit::TooLittleVram { needs, has };
        }
        let room = budget - whole;
        if entry.footprint.experts <= room {
            return Fit::Vram;
        }
        let beyond = entry.footprint.experts - room;
        let host = share.saturating_sub(DEFAULT_CACHE_BYTES + HOST_OTHER + work.host(state));
        return if beyond <= host { Fit::Ram(beyond) } else { Fit::Drive(beyond - host) };
    }
    if budget >= whole && entry.footprint.experts <= budget - whole {
        return Fit::Vram;
    }
    let Some(room) = partial_room(entry, &budgets, context) else { return Fit::TooLittleVram { needs, has } };
    if entry.footprint.experts <= room {
        return Fit::Vram;
    }
    let beyond = entry.footprint.experts - room;
    let rows = row_cache(entry, machine.ram, state);
    let keep = share.saturating_sub(HOST_OTHER + checkpoint_bytes() + rows);
    if beyond <= keep.saturating_sub(state) {
        return Fit::Ram(beyond);
    }
    let starting = entry.footprint.state_per_token * context.min(STARTING_KV as u64);
    Fit::Drive(beyond.saturating_sub(keep.saturating_sub(starting)))
}

/// The host memory the server's worker may take: its share of the machine's, at most what other programs leave free now
/// less what the system keeps free beside page-locked and guaranteed memory (an RTX 5070 Ti with 62 GiB, 30 GiB of it held by
/// other programs, kept 23 GiB of its experts' banks and faulted 2-4 million pages a request: it is not a machine that keeps them in RAM).
fn host_share(machine: &Machine) -> u64 {
    let share = (machine.ram as f64 * HOST_SHARE) as u64;
    if machine.free_ram == 0 { share } else { share.min(machine.free_ram.saturating_sub(straydog_engine::host::left_free(machine.ram))) }
}

/// A partially resident Qwen3.8's row cache of the PLE table (rows of 90 bytes), as `Model::new` chooses it.
fn row_cache(entry: &Entry, ram: u64, state: u64) -> u64 {
    let budget = row_cache_budget(entry.footprint.tables / 90, 90, ram, state);
    if ram < 2 * entry.footprint.experts { budget.min(SCARCE_ROW_CACHE) } else { budget }
}

/// Whether the budgets `auto` takes here leave a Qwen3.8 drafter worth having: at --vram 5 it cost 1.4% on an
/// RTX 5070 Ti (IQ2_XS, 8 requests a side), at 6 it gained 3.4%, and below about 4.75 the placement drops it.
pub fn drafter_pays(machine: &Machine) -> bool {
    machine.gpus.iter().map(|gpu| auto_budget(cfg!(windows), free_for_auto(gpu), gpu.total)).sum::<u64>() >= (11 << 30) / 2
}

/// The largest context in [`CONTEXTS`] at which `entry` runs on `machine`, with how; when it runs at
/// none, the verdict at the smallest (what the least it needs is).
pub fn fit_at_best_context(entry: &Entry, machine: &Machine) -> (u64, Fit) {
    best_context_on(cfg!(windows), entry, machine)
}

fn best_context_on(windows: bool, entry: &Entry, machine: &Machine) -> (u64, Fit) {
    for &context in &CONTEXTS {
        let verdict = fit_on(windows, entry, machine, context);
        if verdict.runs() {
            return (context, verdict);
        }
    }
    let smallest = CONTEXTS[CONTEXTS.len() - 1];
    (smallest, fit_on(windows, entry, machine, smallest))
}

/// The models of `entries`, in the order they first appear.
pub fn models(entries: &[Entry]) -> Vec<&str> {
    let mut models: Vec<&str> = Vec::new();
    for entry in entries {
        if !models.contains(&entry.id.model) {
            models.push(entry.id.model);
        }
    }
    models
}

/// The bundle to start from, by index into `entries` and their `fits`: the best class (VRAM,
/// RAM, drive) that the first model reaching VRAM or RAM has, its first bundle in that class; when
/// no model does, of the model of the bundle reading the least from the drive, the first bundle
/// (the most exact) reading at most a twentieth of its experts more than that one. A twentieth is what
/// separates Qwen3.8's IQ2_XS from Q2_0, 4% smaller and the least exact.
pub fn recommend(entries: &[Entry], fits: &[Fit]) -> Option<usize> {
    let rank = |index: usize| fits[index].rank();
    for model in models(entries) {
        let of_model: Vec<usize> = (0..entries.len()).filter(|&index| entries[index].id.model == model).collect();
        let best = of_model.iter().filter_map(|&index| rank(index)).min();
        if let Some(best @ (0 | 1)) = best {
            return of_model.into_iter().find(|&index| rank(index) == Some(best));
        }
    }
    let read = |index: usize| match fits[index] { Fit::Drive(bytes) => bytes, _ => 0 };
    let least = (0..entries.len()).filter(|&index| rank(index).is_some()).min_by_key(|&index| read(index))?;
    (0..entries.len()).find(|&index| rank(index).is_some() && entries[index].id.model == entries[least].id.model
                                     && read(index) <= read(least) + entries[index].footprint.experts / 20)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::setup::machine::Gpu;

    /// A machine of `gpus` GPUs of `bytes` each and `ram` bytes of memory.
    pub fn machine(gpus: usize, bytes: u64, ram: u64) -> Machine {
        let gpus = (0..gpus).map(|ordinal| Gpu { ordinal, name: "GPU".into(), total: bytes, free: bytes }).collect();
        Machine { backend: "ROCm/HIP", gpus, gpu_error: None, ram, free_ram: ram }
    }

    /// The test machine: two R9700s (34208743424 bytes each) and the 134411186176 bytes of RAM the worker measures.
    pub fn two_r9700s() -> Machine {
        machine(2, 34_208_743_424, 134_411_186_176)
    }

    /// The bundle a test names by its quantization: the only one of that name (DeepSeek-V4.1's by 0.1.3's names), else
    /// Qwen3.8's.
    pub fn id(quant: &str) -> Id {
        Id::find(None, None, quant).or_else(|_| Id::find(Some(MODEL), None, quant)).unwrap()
    }

    fn entry(quant: &str) -> Entry {
        Entry::of(id(quant)).unwrap()
    }

    /// Unsloth's Qwen3.8 bundle of `quant`.
    pub fn unsloth(quant: &'static str) -> Id {
        Id { model: MODEL, provider: UNSLOTH, quant }
    }

    #[test]
    fn every_pinned_bundle_is_offered_once() {
        assert_eq!(ORDER.len(), bundle::BUNDLES.len());
        assert!(bundle::BUNDLES.iter().all(|pinned| ORDER.iter().filter(|offered| *offered == pinned).count() == 1));
        let entries = pinned_entries();
        assert_eq!(models(&entries), ["DeepSeek-V4.1-Flash-1017", "Qwen3.8-Flash-Next"]);
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
        // Unsloth's Qwen3.8 UD-IQ1_M: its three shards (68.0 GB) beside the official resources, and Qwen3.8's extras
        let unsloth = Entry::of(Id { model: MODEL, provider: UNSLOTH, quant: "UD-IQ1_M" }).unwrap();
        assert_eq!(unsloth.architecture, "qwen4exp");
        assert_eq!(unsloth.files.iter().filter(|file| file.name.ends_with(".gguf")).count(), 3);
        assert!((68_000_000_000..68_100_000_000).contains(&unsloth.download_bytes(&[])));
        assert_eq!(unsloth.default_extras(), [Extra::Mtp, Extra::Vision]);
        assert_eq!(unsloth.name(), "Qwen3.8-Flash-Next / Unsloth / UD-IQ1_M");
        // every bundle: its tokenizer files come with it and are small
        for id in ORDER {
            let entry = Entry::of(id).unwrap();
            assert!(entry.download_bytes(&[]) > 30_000_000_000, "{id}");
            assert!(entry.footprint.dense + entry.footprint.experts + entry.footprint.tables <= entry.download_bytes(&[]), "{id}");
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

    /// The fit rule against what the placement did on the test machine (its server's logs, 2026-10-28):
    /// GSQ-RCO-IQ2_XXS held 5832 expert slots in VRAM and had a 102.2 GiB host tier, GSQ-RCO-IQ2_XS 5147
    /// slots and the same tier; the experts' bytes per slot are the average.
    #[test]
    fn the_fit_rule_agrees_with_the_placement_on_two_r9700s() {
        // (it runs Linux: under Windows' margin IQ3_S would not fit whole)
        let machine = two_r9700s();
        for (quant, slots) in [("GSQ-RCO-IQ2_XXS", 5832u64), ("GSQ-RCO-IQ2_XS", 5147)] {
            let entry = entry(quant);
            let in_vram = slots * entry.footprint.experts / 15360;
            let Fit::Ram(beyond) = fit_on(false, &entry, &machine, DEFAULT_CONTEXT) else { panic!("{quant}: {:?}", fit_on(false, &entry, &machine, DEFAULT_CONTEXT)) };
            let placed = entry.footprint.experts - in_vram;
            // within 5% of the experts the placement left outside VRAM, and inside the host tier it had (102.2 GiB)
            assert!(beyond.abs_diff(placed) < placed / 20, "{quant}: rule {beyond}, placement {placed}");
            assert!(beyond < (102.2 * GIB as f64) as u64);
        }
        // UD-IQ2_XXS has 157 GB of experts: past VRAM and the tier, the drive serves the rest
        assert!(matches!(fit_on(false, &entry("UD-IQ2_XXS"), &machine, DEFAULT_CONTEXT), Fit::Drive(_)));
        // Qwen3.8 is in VRAM whole at the full context (the placement said "fully resident" and
        // "0.0 GiB host tier"), and keep it
        for quant in ["Q2_0", "IQ3_S"] {
            assert_eq!(best_context_on(false, &entry(quant), &machine), (DEFAULT_CONTEXT, Fit::Vram), "{quant}");
        }
        let entries = pinned_entries();
        let fits: Vec<Fit> = entries.iter().map(|entry| fit_on(false, entry, &machine, DEFAULT_CONTEXT)).collect();
        // (Unsloth's first: UD-IQ1_M, as large as ISTA-DASLab's IQ2_XXS, fits as it does)
        assert_eq!(recommend(&entries, &fits).map(|index| entries[index].id), Some(Id { model: DEEPSEEK, provider: UNSLOTH, quant: "UD-IQ1_M" }));
    }

    #[test]
    fn a_small_machine_is_offered_what_it_can_run_and_told_what_is_slow() {
        // one 12 GB GPU and 32 GB of RAM
        let small = machine(1, 12 << 30, 32 << 30);
        let entries = pinned_entries();
        let fits: Vec<(u64, Fit)> = entries.iter().map(|entry| fit_at_best_context(entry, &small)).collect();
        for (entry, (context, fit)) in entries.iter().zip(&fits) {
            eprintln!("{:<48} {:>6} {:?}  {}", entry.name(), context, fit, fit.note(entry));
        }
        let of = |quant: &str| fits[entries.iter().position(|entry| entry.id == id(quant)).unwrap()].clone();
        // DeepSeek-V4.1's weights and working memory do not fit 12 GB whatever the context: not offered
        for quant in ["GSQ-RCO-IQ2_XXS", "GSQ-RCO-IQ2_XS", "UD-IQ2_XXS"] {
            assert!(matches!(of(quant).1, Fit::TooLittleVram { .. }), "{quant}");
        }
        // the Qwen3.8 bundles run, keeping the context they start with (the placement ran IQ3_S
        // partially resident at --vram 12, K/V in host memory), and read from the drive
        for quant in ["IQ3_S", "IQ2_XS", "Q2_0"] {
            assert!(matches!(of(quant), (DEFAULT_CONTEXT, Fit::Drive(_))), "{quant}: {:?}", of(quant));
        }
        let recommended = recommend(&entries, &fits.iter().map(|(_, fit)| fit.clone()).collect::<Vec<_>>()).unwrap();
        // (Unsloth's first: of their bundles the one reading least from the drive, UD-IQ1_S)
        assert_eq!(entries[recommended].id, unsloth("UD-IQ1_S"));
    }

    /// `machine` with `others` bytes of each GPU held by the desktop and other programs, and a reported card of `mib`.
    fn windows_pc(mib: u64, others: u64, ram: u64) -> Machine {
        let total = mib << 20;
        let gpus = vec![Gpu { ordinal: 0, name: "GPU".into(), total, free: total - others }];
        Machine { backend: "CUDA", gpus, gpu_error: None, ram, free_ram: ram }
    }

    /// Every bundle's verdict and the recommended one's name, under Windows or elsewhere.
    fn verdicts(on_windows: bool, pc: &Machine) -> (Vec<(Id, u64, Fit)>, Option<Id>) {
        let entries = pinned_entries();
        let fits: Vec<(u64, Fit)> = entries.iter().map(|entry| best_context_on(on_windows, entry, pc)).collect();
        let classes: Vec<Fit> = fits.iter().map(|(_, fit)| fit.clone()).collect();
        let named = recommend(&entries, &classes).map(|index| entries[index].id);
        (entries.iter().zip(fits).map(|(entry, (context, fit))| (entry.id, context, fit)).collect(), named)
    }

    fn verdicts_of(verdicts: &[(Id, u64, Fit)], bundle: Id) -> Fit {
        verdicts.iter().find(|(named, ..)| *named == bundle).map(|(_, _, fit)| fit.clone()).unwrap()
    }

    fn verdict(verdicts: &[(Id, u64, Fit)], quant: &str) -> (u64, Fit) {
        verdicts.iter().find(|(named, ..)| *named == id(quant)).map(|(_, context, fit)| (*context, fit.clone())).unwrap()
    }

    const GIB_F: f64 = GIB as f64;
    /// 32 GB of RAM as Windows reports it, and a desktop with little else on the GPU.
    const RAM_32: u64 = 31_900 << 20;
    const DESKTOP: u64 = 800 << 20;

    /// What the placement takes, from its own arithmetic: a partial Qwen3.8 needs far less than the 10 GiB once claimed for IQ2_XS
    /// (dense 3.5 + working 2.9 + RESERVE 1.6 + margin 2), and it starts within what `auto` leaves on a 6 GB card.
    #[test]
    fn a_partial_qwen_needs_what_the_placement_needs() {
        let least = |quant: &str, context: u64| min_vram(&entry(quant), 1, context) as f64 / GIB_F;
        for quant in ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"] {
            eprintln!("{quant}: {:.2} GiB at 262144 tokens, {:.2} at 32768", least(quant, DEFAULT_CONTEXT), least(quant, 32_768));
        }
        // a bigger bundle takes more; a whole context takes the indexers' growth more than 32768 tokens
        assert!(least("Q2_0", DEFAULT_CONTEXT) < least("IQ2_XS", DEFAULT_CONTEXT) && least("IQ2_XS", DEFAULT_CONTEXT) < least("IQ3_XXS", DEFAULT_CONTEXT));
        assert!(least("IQ2_XS", 32_768) < least("IQ2_XS", DEFAULT_CONTEXT));
        // IQ2_XS: under what R9700s started it at (--vram 4.5, 14ed46b), and nowhere near 10 GiB
        assert!((4.0..4.5).contains(&least("IQ2_XS", DEFAULT_CONTEXT)), "{}", least("IQ2_XS", DEFAULT_CONTEXT));
        // Its state, 13 GiB at the whole context, is in host memory: only the indexers are in VRAM
        assert!(least("IQ2_XS", DEFAULT_CONTEXT) - least("IQ2_XS", 16_384) < 0.5);
        // IQ3_XXS as measured on an RTX 5070 Ti: --vram 4.15 started and 4.1 was refused; the gate is the placement's at the K/V it starts with
        let starts = |gib: f64| {
            let budget = (gib * GIB_F) as u64;
            let ours = entry("IQ3_XXS");
            partial_slot_bytes(pinned_dense(&ours), 2_329_600, 32_768, PARTIAL_BATCH, budget).is_some()
        };
        assert!(starts(4.15) && !starts(4.1));
        assert_eq!(pinned_dense(&entry("IQ3_XXS")), 3_216_268_800);
        // Unsloth's, as Weights counted them
        for (quant, pinned) in [("UD-IQ1_S", 3_802_437_120u64), ("UD-IQ1_M", 4_014_773_760), ("UD-IQ2_XS", 4_368_668_160)] {
            assert_eq!(pinned_dense(&Entry::of(unsloth(quant)).unwrap()), pinned, "{quant}");
        }
        // two GPUs each keep their workspaces, and share the weights
        let (one, two) = (min_vram(&entry("IQ2_XS"), 1, DEFAULT_CONTEXT), min_vram(&entry("IQ2_XS"), 2, DEFAULT_CONTEXT));
        assert!(two > one && two < 2 * one);
    }

    /// The machines of the goal: what is offered and what is recommended, for 6, 8, 12 and 16 GB cards, under Windows and elsewhere.
    #[test]
    fn the_cards_from_6_to_16_gb_run_qwen_and_recommend_what_they_can_hold() {
        let cards = [(6144, "6 GB"), (8188, "8 GB"), (12288, "12 GB"), (16303, "16 GB")];
        for on_windows in [true, false] {
            for (mib, name) in cards {
                let pc = windows_pc(mib, DESKTOP, RAM_32);
                let (fits, named) = verdicts(on_windows, &pc);
                eprintln!("{name} windows={on_windows}, 32 GB: recommended {named:?}");
                for (id, context, fit) in &fits {
                    eprintln!("  {:<48} {context:>6} {fit:?}  {}", id.to_string(), fit.note(&Entry::of(*id).unwrap()));
                }
                // the smaller Qwen3.8 bundles run on every card at the whole context, reading from the drive what RAM will not keep;
                // from 8 GB the larger do too (at 6 GB they are short of budget, or of the context's indexers)
                let bundles: &[&str] = if mib < 8000 { &["Q2_0", "IQ2_XS"] } else { &["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"] };
                for quant in bundles {
                    let (context, fit) = verdict(&fits, quant);
                    assert_eq!(context, DEFAULT_CONTEXT, "{name}: {quant}");
                    assert!(matches!(fit, Fit::Drive(_)), "{name}: {quant}: {fit:?}");
                }
                // and 32 GB holds none of them whole: the K/V's whole context would leave the banks 10 GiB of it. From 8 GB Unsloth's
                // first, the one reading least; at 6 GB their dense layers (Q5_K to Q8_0: 3.5 to 4.1 GiB pinned, ISTA-DASLab's IQ2_XS
                // 2.6) leave them short of VRAM, and ISTA-DASLab's IQ2_XS is the one
                assert_eq!(named, Some(if mib < 8000 { id("IQ2_XS") } else { unsloth("UD-IQ1_S") }), "{name}");
                if mib < 8000 {
                    for quant in ["UD-IQ3_XXS", "UD-IQ2_XS", "UD-IQ1_M", "UD-IQ1_S"] {
                        assert!(matches!(verdicts_of(&fits, unsloth(quant)), Fit::TooLittleVram { .. }), "{name}: {quant}");
                    }
                }
                // DeepSeek-V4.1 runs on none of these with the desktop's share gone, or when it does never well
                let (_, deepseek) = verdict(&fits, "GSQ-RCO-IQ2_XXS");
                assert!(matches!(deepseek, Fit::TooLittleVram { .. } | Fit::Drive(_)), "{name}: {deepseek:?}");
            }
        }
    }

    /// 6 GB and 32 GB, the low-spec target: IQ2_XS runs, what it misses in RAM is told as measured, and the messages say what `auto` takes.
    #[test]
    fn the_goal_machine_gets_iq2_xs_with_true_words() {
        let pc = windows_pc(6144, DESKTOP, RAM_32);
        let (fits, named) = verdicts(true, &pc);
        assert_eq!(named, Some(id("IQ2_XS")));
        let ours = entry("IQ2_XS");
        let (context, fit) = verdict(&fits, "IQ2_XS");
        assert_eq!(context, DEFAULT_CONTEXT);
        // banks of 33 GiB, of which a working set of about 22 GiB beside the page-locked 2.3 GiB stays: about a third from the drive
        let Fit::Drive(from_drive) = fit else { panic!("{fit:?}") };
        let gib = from_drive as f64 / GIB_F;
        assert!((9.0..13.0).contains(&gib), "{gib}");
        let note = fit.note(&ours);
        assert!(note.contains("do not stay in RAM") && note.contains("change of topic") && !note.contains("slow "), "{note}");
        // Q2_0 reads less, but only by about the 4% it is smaller: the most exact within a twentieth stands
        let Fit::Drive(q2) = verdict(&fits, "Q2_0").1 else { panic!() };
        assert!(q2 < from_drive && from_drive - q2 < ours.footprint.experts / 20);
        // a card whose desktop keeps it short of the least is told so, with what it would need
        let busy = windows_pc(6144, 1500 << 20, RAM_32);
        let refused = fit_on(true, &ours, &busy, DEFAULT_CONTEXT);
        let Fit::TooLittleVram { needs, has } = refused else { panic!("{refused:?}") };
        assert_eq!(has, (6144 - 1500) << 20);
        assert_eq!(needs, min_vram(&ours, 1, DEFAULT_CONTEXT) + (1 << 30));
        let note = refused.note(&ours);
        assert!(note.contains("of VRAM free") && note.contains("4.5 GiB for it"), "{note}");
        // holding more than a quarter of it is a job `auto` waits for: it says nothing of the card, as a running engine says nothing
        assert!(fit_on(true, &ours, &windows_pc(6144, 1600 << 20, RAM_32), DEFAULT_CONTEXT).runs());
        // 4 GB holds no Qwen3.8 at all
        assert!(matches!(fit_on(true, &ours, &windows_pc(4096, 0, RAM_32), DEFAULT_CONTEXT), Fit::TooLittleVram { .. }));
    }

    /// RAM's rules as the placement and the worker hold to them.
    #[test]
    fn rams_rules_follow_the_placements() {
        let (ours, big) = (entry("IQ2_XS"), entry("IQ3_S"));
        // the banks are more than half of 32, 62 and 64 GiB of memory: the row cache is 1 GiB, and it is not page-locked whole
        assert_eq!(row_cache(&ours, RAM_32, ours.footprint.state_per_token * DEFAULT_CONTEXT), SCARCE_ROW_CACHE);
        assert_eq!(row_cache(&ours, 64 << 30, ours.footprint.state_per_token * DEFAULT_CONTEXT), SCARCE_ROW_CACHE);
        // no bank is page-locked at 32 GiB (under 1.25 times the banks), at 48 half the memory is
        use straydog_engine::model::lock_budget;
        assert_eq!(lock_budget(RAM_32, ours.footprint.experts, 2 << 30), 0);
        assert_eq!(lock_budget(48 << 30, ours.footprint.experts, 2 << 30), 22 << 30);
        // 96 GiB, more than twice these banks: half of what the K/V leaves, up to the table's rows
        let wide = row_cache(&ours, 96 << 30, ours.footprint.state_per_token * DEFAULT_CONTEXT);
        assert!(wide > 30 << 30 && wide < 40 << 30, "{wide}");
        // the K/V's growth is what parts 16 GB's RAM class: the banks of IQ3_S do not keep beside it at 62 GiB, IQ3_XXS's do
        let (pc, cases) = (windows_pc(16303, DESKTOP, 62 << 30), pinned_entries());
        let (fits, named) = verdicts(true, &pc);
        // (Unsloth's first: UD-IQ2_XS, as large as ISTA-DASLab's IQ3_XXS, keeps its banks as that does)
        assert_eq!(named, Some(unsloth("UD-IQ2_XS")));
        assert!(matches!(verdicts_of(&fits, unsloth("UD-IQ2_XS")), Fit::Ram(_)));
        assert!(matches!(fit_on(true, &big, &pc, DEFAULT_CONTEXT), Fit::Drive(0)), "{:?}", fit_on(true, &big, &pc, DEFAULT_CONTEXT));
        assert!(matches!(verdict(&fits, "IQ3_XXS").1, Fit::Ram(_)));
        assert_eq!(cases.len(), fits.len());
        // no state of a Qwen3.8 is counted at the whole context for what a shorter one will take: a smaller context keeps more
        let ram_only = |context| match fit_on(true, &big, &windows_pc(16303, DESKTOP, 64 << 30), context) { Fit::Ram(_) => true, Fit::Drive(_) => false, verdict => panic!("{verdict:?}") };
        assert!(ram_only(131_072));
    }

    /// An Unsloth Qwen3.8 bundle set up with another engram table fetches that file instead of its own table shard, and counts it.
    #[test]
    fn a_bundle_set_up_with_another_engram_table_fetches_it_instead_of_its_own() {
        let own = Entry::of(unsloth("UD-IQ1_M")).unwrap();
        assert_eq!(own.tables().iter().map(|(precision, _)| precision.as_str()).collect::<Vec<_>>(), ["IQ4_NL", "BF16", "Q8_0", "Q5_1"]);
        let q8 = own.with_table("q8_0").unwrap();
        assert_eq!(q8.table.as_deref(), Some("Q8_0"));
        assert!(!q8.files.iter().any(|file| file.name.ends_with("-00003-of-00003.gguf")));
        assert!(q8.files.iter().any(|file| file.name == "Qwen3.8-Flash-Next-engram-Q8_0.gguf_file" && file.tree > 0));
        assert_eq!(q8.download_bytes(&[]), own.download_bytes(&[]) - 28_800_138_432 + 54_400_261_312);
        // its own precision changes nothing; a precision not offered, or a bundle without a table of its own, is refused
        assert_eq!(own.with_table("IQ4_NL").unwrap().files, own.files);
        assert!(own.with_table("Q6_K").unwrap_err().contains("no Q6_K engram table"));
        assert!(entry("IQ2_XS").with_table("Q8_0").unwrap_err().contains("no engram table"));
    }

    /// Memory other programs hold is not the server's: an RTX 5070 Ti with 62 GiB, 28 GiB of it free (Windows, 2026-11-05: setup
    /// said UD-IQ1_M's banks stay in RAM, the server kept 23 GiB of them and faulted millions of pages a request).
    #[test]
    fn what_other_programs_hold_is_not_counted_as_ram() {
        let ours = Entry::of(unsloth("UD-IQ1_M")).unwrap();
        let idle = windows_pc(16303, DESKTOP, 62 << 30);
        assert!(matches!(fit_on(true, &ours, &idle, DEFAULT_CONTEXT), Fit::Ram(_)), "{:?}", fit_on(true, &ours, &idle, DEFAULT_CONTEXT));
        let busy = Machine { free_ram: 28 << 30, ..idle.clone() };
        let Fit::Drive(from_drive) = fit_on(true, &ours, &busy, DEFAULT_CONTEXT) else { panic!("{:?}", fit_on(true, &ours, &busy, DEFAULT_CONTEXT)) };
        assert!(from_drive > 5 << 30, "{from_drive}");
        assert!(busy.describe().iter().any(|line| line.contains("28.0 GiB of it is free now")), "{:?}", busy.describe());
        assert!(!idle.describe().iter().any(|line| line.contains("free now")));
    }

    #[test]
    fn a_drafter_is_not_the_default_where_it_does_not_pay() {
        // (the budgets depend on the platform's margin: 6 GB under Windows leaves about 4.2 GiB, elsewhere 4.5)
        assert!(!drafter_pays(&windows_pc(6144, DESKTOP, RAM_32)));
        assert!(drafter_pays(&windows_pc(8188, DESKTOP, RAM_32)));
        assert!(drafter_pays(&two_r9700s()));
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
        // (Qwen3.8's: Unsloth's UD-IQ3_XXS, UD-IQ2_XS, UD-IQ1_M and UD-IQ1_S, ISTA-DASLab's IQ3_S, IQ3_XXS, IQ2_XS and Q2_0)
        let mut fits: Vec<Fit> = vec![Fit::Drive(5), Fit::Drive(5), Fit::Drive(4), Fit::Drive(3), Fit::Ram(9), Fit::Ram(8), Fit::Vram, Fit::Vram,
                                      Fit::Ram(9), Fit::Ram(8), Fit::Vram, Fit::Vram];
        assert_eq!(recommend(&entries, &fits), Some(6));
        // DeepSeek in RAM wins over Qwen3.8 in VRAM: the first model that runs well, its first bundle of the best class
        fits[2] = Fit::Ram(1);
        fits[0] = Fit::Ram(2);
        assert_eq!(recommend(&entries, &fits), Some(0));
        // DeepSeek in VRAM for a later bundle only
        fits[0] = Fit::Drive(1);
        fits[3] = Fit::Vram;
        assert_eq!(recommend(&entries, &fits), Some(3));
        // nothing runs well: of the model reading least from the drive, the first bundle within a twentieth of its experts of that
        let bytes = |gib: u64| gib * GIB;
        let slow: Vec<Fit> = vec![Fit::Drive(bytes(20)), Fit::Drive(bytes(20)), Fit::Drive(bytes(19)), Fit::NoGpu, Fit::Drive(bytes(30)), Fit::Drive(bytes(18)), Fit::Drive(bytes(11)), Fit::Drive(bytes(11)),
                                  Fit::Drive(bytes(30)), Fit::Drive(bytes(18)), Fit::Drive(bytes(11)), Fit::Drive(bytes(10))];
        assert_eq!(recommend(&entries, &slow), Some(6));
        // ... or the one reading least, past that
        let wide: Vec<Fit> = vec![Fit::NoGpu, Fit::NoGpu, Fit::NoGpu, Fit::NoGpu, Fit::Drive(bytes(30)), Fit::Drive(bytes(20)), Fit::Drive(bytes(16)), Fit::Drive(bytes(16)),
                                  Fit::Drive(bytes(30)), Fit::Drive(bytes(20)), Fit::Drive(bytes(16)), Fit::Drive(bytes(10))];
        assert_eq!(recommend(&entries, &wide), Some(11));
        // a model that reads least is chosen over the bundles of another, however near
        let other: Vec<Fit> = vec![Fit::Drive(bytes(11)), Fit::NoGpu, Fit::NoGpu, Fit::NoGpu, Fit::Drive(bytes(30)), Fit::Drive(bytes(20)), Fit::Drive(bytes(11)), Fit::Drive(bytes(11)),
                                  Fit::Drive(bytes(30)), Fit::Drive(bytes(20)), Fit::Drive(bytes(11)), Fit::Drive(bytes(10))];
        assert_eq!(recommend(&entries, &other), Some(6));
    }
}
