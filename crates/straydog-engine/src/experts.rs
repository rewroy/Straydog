//! Routed experts beyond VRAM (docs/DESIGN.md §10). A large mixture-of-experts model keeps its
//! experts in three tiers: slots in each device's VRAM, a bounded page-locked host tier, and the
//! model files, read directly (around the page cache, so nothing is cached twice). A decode token
//! of DeepSeek-V4.1 uses ~220 experts of ~10 MB; nine in ten are in VRAM, and the rest come from
//! the host tier at the bus's speed (0.4 ms each) or from the drive (~2 ms each, several at once).
//!
//! The two memory tiers hold different experts: the host tier takes what VRAM evicts, written back
//! over the bus's other direction, and gives up what it hands to VRAM. Measured on traced routes,
//! that halves the drive's reads against a host tier that keeps copies of what VRAM holds.
//!
//! Copies run on the device's upload and download streams beside compute ([`crate::transfer`]).
//! The compute stream waits only for the copies of the experts it uses next, so a later layer's
//! experts (predicted, or the next wave of a prompt's) arrive while this layer computes.
//! Drive reads are whole aligned 4 KiB blocks: the unit a Merkle tree over the file (fs-verity's
//! format) hashes, so a verifier checks each block between its read and its first use.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fs::File;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use crate::gpu::{Backend, Result};
use crate::ops::Buffer;
use crate::storage::{self, ALIGN};
use crate::transfer::{self, Mark, Stream};

/// One drive read: an expert's ~40 pieces go to as many readers at once, so one miss reads at
/// near the drive's rate (1 MiB pieces waited ~10% longer for a token's misses).
const CHUNK: u64 = 256 << 10;
/// Host tier memory is allocated and page-locked in arenas of this size, by a thread of its own
/// from the start (page-locking costs ~0.4 s per GiB: never in a pass's way).
const ARENA: usize = 1 << 30;

/// A tensor of a layer's experts: expert `e`'s bytes are `bytes` bytes at `base + e * stride` of
/// file `file` (its whole matrix, `bytes == stride`, or a device's rows of it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Role {
    pub file: usize,
    pub base: u64,
    pub stride: u64,
    pub bytes: u64,
}

/// Of each `row_bytes` row of a role's matrix, the `take` bytes at `skip` (whole blocks of columns):
/// a device's columns of a matrix split by them (the down matrix of experts split by their ffn
/// rows). The role reads its whole matrix; a VRAM slot holds these bytes, row after row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Columns {
    pub row_bytes: u64,
    pub skip: u64,
    pub take: u64,
}

/// Checks a drive read of `file` at `offset` (whole aligned blocks) before its first use.
pub type Verifier = Arc<dyn Fn(usize, u64, &[u8]) -> std::result::Result<(), String> + Send + Sync>;

/// The model files and where each layer's experts lie in them.
pub struct Layout {
    /// Per file: a handle, and whether it reads around the page cache (aligned reads only).
    files: Vec<(Arc<File>, bool)>,
    /// Per layer: its roles (gate, up, down...), None for a layer without routed experts.
    layers: Vec<Option<Vec<Role>>>,
    /// The roles (layer, role) whose slots keep some columns ([`keep_columns`](Self::keep_columns)).
    columns: HashMap<(usize, usize), Columns>,
    verifier: Option<Verifier>,
}

impl Layout {
    /// `files` are opened model files; direct handles are made where the platform has them.
    pub fn new(files: &[File], layers: Vec<Option<Vec<Role>>>, verifier: Option<Verifier>) -> Result<Layout> {
        let files = files.iter().map(|file| match storage::direct(file) {
            Some(direct) => Ok((Arc::new(direct), true)),
            None => file.try_clone().map(|clone| (Arc::new(clone), false)).map_err(|exc| exc.to_string()),
        }).collect::<Result<Vec<_>>>()?;
        for role in layers.iter().flatten().flatten() {
            if role.file >= files.len() {
                return Err("an expert role names a file that is not given".into());
            }
            if role.bytes == 0 || role.bytes > role.stride {
                return Err("an expert role's bytes exceed its stride".into());
            }
        }
        Ok(Layout { files, layers, columns: HashMap::new(), verifier })
    }

    /// VRAM slots keep only `columns` of `layer`'s role `role` (a resident store's: see [`Columns`]).
    pub fn keep_columns(&mut self, layer: usize, role: usize, columns: Columns) -> Result<()> {
        let found = self.roles(layer)?.get(role).copied().ok_or("no such expert role")?;
        if columns.take == 0 || columns.skip + columns.take > columns.row_bytes || found.bytes % columns.row_bytes != 0 {
            return Err("an expert role's kept columns lie outside its rows".into());
        }
        self.columns.insert((layer, role), columns);
        Ok(())
    }

    /// Role `role` of `layer`'s bytes in a VRAM slot, of `bytes` read.
    fn kept(&self, layer: usize, role: usize, bytes: u64) -> u64 {
        self.columns.get(&(layer, role)).map_or(bytes, |columns| bytes / columns.row_bytes * columns.take)
    }

    fn roles(&self, layer: usize) -> Result<&[Role]> {
        self.layers.get(layer).and_then(|roles| roles.as_deref()).ok_or_else(|| format!("layer {layer} has no routed experts"))
    }

    /// An expert's bytes in a VRAM slot: its roles one after another.
    pub fn slot_bytes(&self) -> u64 {
        self.layers.iter().enumerate()
            .filter_map(|(layer, roles)| roles.as_ref().map(|roles| roles.iter().enumerate().map(|(index, role)| self.kept(layer, index, role.bytes)).sum()))
            .max().unwrap_or(0)
    }

    /// Where each role's bytes lie in a host slot filled by drive reads.
    fn read_plan(&self, layer: usize, expert: usize) -> Result<Vec<Span>> {
        let mut at = 0u64;
        self.roles(layer)?.iter().enumerate().map(|(index, role)| {
            let start = role.base + expert as u64 * role.stride;
            let offset = start & !(ALIGN as u64 - 1);
            let len = (start + role.bytes - offset).next_multiple_of(ALIGN as u64);
            let span = Span { file: role.file, offset, len, at, data: at + (start - offset), bytes: role.bytes,
                              columns: self.columns.get(&(layer, index)).copied() };
            at += len;
            Ok(span)
        }).collect()
    }
}

/// A role's aligned read: `len` bytes from `offset` of file `file` to `at` in a host slot, where
/// the role's `bytes` start at `data` (of which a VRAM slot keeps `columns`, when given).
struct Span {
    file: usize,
    offset: u64,
    len: u64,
    at: u64,
    data: u64,
    bytes: u64,
    columns: Option<Columns>,
}

impl Span {
    /// Move the kept columns of the role's rows, read to `data` of the host slot at `base`, to the
    /// start of their rows' place there, row after row; their bytes.
    ///
    /// # Safety
    /// The host slot at `base` holds the span as read, and nothing else uses it meanwhile.
    unsafe fn keep_columns(&self, base: *mut u8) -> u64 {
        let Some(columns) = self.columns else { return self.bytes };
        let rows = self.bytes / columns.row_bytes;
        for row in 0..rows {
            // SAFETY: both ranges lie inside the span (each row's kept bytes move to before their
            // source, after the rows already moved); `copy` allows them to overlap.
            unsafe {
                let data = base.add(self.data as usize);
                std::ptr::copy(data.add((row * columns.row_bytes + columns.skip) as usize), data.add((row * columns.take) as usize),
                               columns.take as usize);
            }
        }
        rows * columns.take
    }
}

/// A pointer a reader thread writes through; the host tier keeps the slot untouched meanwhile.
struct HostPointer(*mut u8);
// SAFETY: the pointee is a host slot reserved for this read until its completion is received.
unsafe impl Send for HostPointer {}

struct ReadJob {
    file: Arc<File>,
    offset: u64,
    len: u64,
    target: HostPointer,
    verifier: Option<(Verifier, usize)>,
    done: Sender<(usize, Landed)>,
    slot: usize,
    /// Set when the read's expert is no longer wanted (a prefetch the pass did not need): a
    /// chunk not yet started is skipped.
    cancelled: Arc<AtomicBool>,
}

/// How a chunk's read ended.
enum Landed {
    Read,
    Skipped,
    Failed(String),
}

type Queue = Arc<(Mutex<(VecDeque<ReadJob>, bool)>, Condvar)>;

/// Threads that read expert chunks from the drive into host slots.
pub struct Reader {
    queue: Queue,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Reader {
    pub fn new(threads: usize) -> Reader {
        let queue: Queue = Arc::default();
        let workers = (0..threads.max(1)).map(|_| {
            let queue = queue.clone();
            std::thread::spawn(move || loop {
                let job = {
                    let mut state = queue.0.lock().unwrap();
                    loop {
                        if let Some(job) = state.0.pop_front() {
                            break job;
                        }
                        if state.1 {
                            return;
                        }
                        state = queue.1.wait(state).unwrap();
                    }
                };
                // A read that panics (a verifier's bug) fails that read; the thread lives on and the
                // pass waiting for it gets an error instead of waiting forever.
                let landed = if job.cancelled.load(Ordering::Relaxed) {
                    Landed::Skipped
                } else {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_read(&job))) {
                        Ok(result) => result.map_or_else(Landed::Failed, |()| Landed::Read),
                        Err(_) => Landed::Failed(format!("the read at {} panicked", job.offset)),
                    }
                };
                let _ = job.done.send((job.slot, landed));
            })
        }).collect();
        Reader { queue, workers }
    }

    /// Queue reads; urgent ones (a pass waits for them) go before those queued for prefetches.
    fn submit(&self, jobs: Vec<ReadJob>, urgent: bool) {
        if jobs.is_empty() {
            return;
        }
        let mut state = self.queue.0.lock().unwrap();
        if urgent {
            for job in jobs.into_iter().rev() {
                state.0.push_front(job);
            }
        } else {
            state.0.extend(jobs);
        }
        drop(state);
        self.queue.1.notify_all();
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.queue.0.lock().unwrap().1 = true;
        self.queue.1.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn run_read(job: &ReadJob) -> std::result::Result<(), String> {
    // SAFETY: the target is `len` bytes of a host slot reserved for this read (see HostPointer).
    let target = unsafe { std::slice::from_raw_parts_mut(job.target.0, job.len as usize) };
    read_fully(&job.file, job.offset, target).map_err(|exc| format!("expert read at {}: {exc}", job.offset))?;
    if let Some((verify, file)) = &job.verifier {
        verify(*file, job.offset, target)?;
    }
    Ok(())
}

/// Fill `target` from `offset`; a direct handle reads whole aligned blocks into an aligned target.
/// A read past the file's end leaves the rest zero (the tail of the last block).
fn read_fully(file: &File, offset: u64, target: &mut [u8]) -> std::io::Result<()> {
    let mut done = 0usize;
    while done < target.len() {
        #[cfg(unix)]
        let count = std::os::unix::fs::FileExt::read_at(file, &mut target[done..], offset + done as u64)?;
        #[cfg(windows)]
        let count = std::os::windows::fs::FileExt::seek_read(file, &mut target[done..], offset + done as u64)?;
        if count == 0 {
            // Past the file's end: only the last block's tail, which holds no expert bytes.
            target[done..].fill(0);
            return Ok(());
        }
        done += count;
    }
    Ok(())
}

/// Allocates and page-locks the host tier's arenas one after another, handing each over as it is
/// ready; a dropped receiver stops it, and it unpins what it could not hand over.
struct Pinner {
    arenas: Receiver<std::result::Result<storage::Buffer, String>>,
    thread: Option<std::thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl Pinner {
    fn new(count: usize, bytes: usize) -> Pinner {
        let (tx, arenas) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let thread = std::thread::spawn(move || {
            for _ in 0..count {
                if stopped.load(Ordering::Relaxed) {
                    return;
                }
                let mut arena = storage::Buffer::new(bytes);
                // SAFETY: the arena is handed over pinned; whoever holds it unpins it before freeing.
                if let Err(error) = unsafe { transfer::pin_host(arena.as_mut_slice().as_mut_ptr(), arena.capacity()) } {
                    let _ = tx.send(Err(error));
                    return;
                }
                if let Err(unsent) = tx.send(Ok(arena)) {
                    if let Ok(mut arena) = unsent.0 {
                        // SAFETY: pinned above, never used by a device.
                        let _ = unsafe { transfer::unpin_host(arena.as_mut_slice().as_mut_ptr()) };
                    }
                    return;
                }
            }
        });
        Pinner { arenas, thread: Some(thread), stop }
    }
}

type Key = (u32, u32);

/// The most VRAM slots a size class keeps free (its reserve).
const MAX_RESERVE: usize = 32;

/// A store's experts as (layer, expert), least recently used first: those in VRAM slots, and
/// those only in the host tier (see [`DeviceExperts::contents`]).
pub type Contents = (Vec<(usize, usize)>, Vec<(usize, usize)>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Content {
    /// Drive reads in flight: this many chunks still to come.
    Reading(u32),
    /// Each role's blocks as read (see [`Layout::read_plan`]).
    Read,
    /// The VRAM slot's layout, written back from the device; usable after the download mark.
    Packed(Mark),
}

struct HostSlot {
    key: Option<Key>,
    content: Content,
    /// An upload from this slot: the slot is not reused before it completes.
    busy: Option<Mark>,
    used: u64,
    failed: Option<String>,
    /// The flag of the reads in flight into this slot.
    cancel: Arc<AtomicBool>,
    /// A chunk was skipped: the slot holds no expert once its reads land.
    skipped: bool,
    /// The VRAM slot that uploads this slot's expert once its reads land.
    waiter: Option<usize>,
    class: usize,
    /// Its arena in the class, and its offset there.
    arena: usize,
    at: usize,
}

struct VramSlot {
    key: Option<Key>,
    /// The upload (and its stream) after this slot's fill, until known complete.
    ready: Option<(Stream, Mark)>,
    /// Marks the slot's bytes must outlive: its last compute use, and a download from it.
    guards: [Option<Mark>; 2],
    /// A fill waiting for the drive: the host slot whose reads it uploads once they land.
    waiting: Option<usize>,
    used: u64,
    /// Filled by a prefetch and not yet acquired.
    speculative: bool,
    /// Acquired or requested since the last release: not evicted before it.
    held: bool,
    class: usize,
}

/// The experts of one size (the layers whose experts take the same bytes): its VRAM slots and
/// host slots, each with a free list and an LRU order. A layer's share of the budgets is the same
/// for every layer (each takes the same number of routes per token), so a class's share is its
/// layers' count; a slot is just its experts' size, not the largest layer's (on DeepSeek-V4.1,
/// whose experts take 7.6 to 12.1 MB, that holds ~15% fewer drive reads per token).
struct Class {
    vram_bytes: u64,
    host_bytes: u64,
    vram: Buffer,
    first: usize,
    count: usize,
    free: Vec<usize>,
    /// Slots holding an expert, by last use.
    lru: BTreeSet<(u64, usize)>,
    /// Free slots kept ahead of need, their experts written back meanwhile.
    reserve: usize,
    host_capacity: usize,
    per_arena: usize,
    arenas: Vec<storage::Buffer>,
    pinner: Option<Pinner>,
    hosts: usize,
    /// Emptied host slots, oldest first (their uploads finish in that order).
    host_free: VecDeque<usize>,
    /// Host slots holding an expert (read or written back), by last use.
    host_lru: BTreeSet<(u64, usize)>,
}

/// Counters since creation.
#[derive(Debug, Default, Clone, Copy)]
pub struct Counts {
    pub hits: u64,
    pub host_hits: u64,
    pub drive_reads: u64,
    pub writebacks: u64,
    pub prefetched: u64,
    /// Prefetched experts their layer then did not use.
    pub prefetch_unused: u64,
    /// Failed reads no pass waited for (dropped from both tiers; read again on their next use).
    pub failed_unwaited: u64,
    /// Seconds the host waited for drive reads.
    pub wait_seconds: f64,
    /// Seconds the host waited for a host slot's copy to finish before reusing it.
    pub slot_wait_seconds: f64,
    /// Seconds spent page-locking host tier arenas as the tier grew.
    pub pin_seconds: f64,
    /// Seconds inside acquire and prefetch, all told.
    pub store_seconds: f64,
}

impl std::ops::Add for Counts {
    type Output = Counts;

    /// Two stores' counters together (the devices of one engine).
    fn add(self, other: Counts) -> Counts {
        Counts {
            hits: self.hits + other.hits,
            host_hits: self.host_hits + other.host_hits,
            drive_reads: self.drive_reads + other.drive_reads,
            writebacks: self.writebacks + other.writebacks,
            prefetched: self.prefetched + other.prefetched,
            prefetch_unused: self.prefetch_unused + other.prefetch_unused,
            failed_unwaited: self.failed_unwaited + other.failed_unwaited,
            wait_seconds: self.wait_seconds + other.wait_seconds,
            slot_wait_seconds: self.slot_wait_seconds + other.slot_wait_seconds,
            pin_seconds: self.pin_seconds + other.pin_seconds,
            store_seconds: self.store_seconds + other.store_seconds,
        }
    }
}

/// One device's expert slots, and its share of the host tier.
pub struct DeviceExperts {
    layout: Arc<Layout>,
    reader: Arc<Reader>,
    classes: Vec<Class>,
    /// Per layer: its class.
    class_of: Vec<Option<usize>>,
    slots: Vec<VramSlot>,
    map: HashMap<Key, usize>,
    host: Vec<HostSlot>,
    host_map: HashMap<Key, usize>,
    done_tx: Sender<(usize, Landed)>,
    done_rx: Receiver<(usize, Landed)>,
    clock: u64,
    /// Slots acquired since the last release, kept from eviction until then.
    in_use: Vec<usize>,
    /// Experts [`request`](Self::request)ed for the next acquire: already counted.
    requested: Vec<Key>,
    /// Per layer: slots a prefetch filled and no acquire of that layer has taken yet.
    speculative: HashMap<u32, Vec<usize>>,
    /// Host slots with reads in flight.
    reading: usize,
    counts: Counts,
    settled: bool,
}

impl DeviceExperts {
    /// VRAM worth `slots` of the largest experts on this device, and `host_budget` bytes of host
    /// tier, both shared among the layers.
    pub fn new(gpu: &mut Backend, layout: Arc<Layout>, reader: Arc<Reader>, slots: usize, host_budget: u64)
               -> Result<DeviceExperts> {
        if !layout.columns.is_empty() {
            return Err("only a resident expert store keeps some columns of a role".into());
        }
        let largest = Self::sizes(&layout.layers, &|_, _, bytes| bytes).iter().flatten().map(|size| size.0).max().unwrap_or(0);
        if slots < 16 || largest == 0 {
            return Err("an expert store needs at least 16 VRAM slots and a layer of experts".into());
        }
        let vram_budget = largest * slots as u64;
        Self::build(gpu, layout, reader, host_budget, &|share, layers, vram_bytes| ((vram_budget * share / layers) / vram_bytes).max(16) as usize)
    }

    /// A store with VRAM slots for every expert of its layers (`experts` a layer) and each size
    /// class's reserve, and no host tier: once each expert is acquired, none is ever evicted
    /// (see [`resident_bases`](Self::resident_bases)).
    pub fn resident(gpu: &mut Backend, layout: Arc<Layout>, reader: Arc<Reader>, experts: usize) -> Result<DeviceExperts> {
        if experts == 0 || Self::sizes(&layout.layers, &|_, _, bytes| bytes).iter().flatten().next().is_none() {
            return Err("an expert store needs a layer of experts".into());
        }
        Self::build(gpu, layout, reader, 0, &|share, _, _| share as usize * experts + MAX_RESERVE)
    }

    /// The VRAM bytes [`resident`](Self::resident) takes for these layers' roles, `experts` a layer.
    pub fn resident_bytes(layers: &[Option<Vec<Role>>], experts: usize) -> u64 {
        Self::resident_bytes_keeping(layers, &HashMap::new(), experts)
    }

    /// [`resident_bytes`](Self::resident_bytes) when slots keep `columns` of roles (layer, role)
    /// ([`Layout::keep_columns`]).
    pub fn resident_bytes_keeping(layers: &[Option<Vec<Role>>], columns: &HashMap<(usize, usize), Columns>, experts: usize) -> u64 {
        let sizes = Self::sizes(layers, &|layer, role, bytes| {
            columns.get(&(layer, role)).map_or(bytes, |columns| bytes / columns.row_bytes * columns.take)
        });
        let mut distinct: Vec<u64> = sizes.iter().flatten().map(|size| size.0).collect();
        distinct.sort_unstable();
        distinct.dedup();
        let layers = |bytes: u64| sizes.iter().flatten().filter(|size| size.0 == bytes).count() as u64;
        distinct.iter().map(|&bytes| bytes * (layers(bytes) * experts as u64 + MAX_RESERVE as u64)).sum()
    }

    /// Each layer's experts' bytes in a VRAM slot (roles back to back) and in a host slot (room for
    /// either layout there: each role's whole blocks as read, or the VRAM slot's).
    fn sizes(layers: &[Option<Vec<Role>>], kept: &dyn Fn(usize, usize, u64) -> u64) -> Vec<Option<(u64, u64)>> {
        layers.iter().enumerate().map(|(layer, roles)| roles.as_ref().map(|roles| {
            let vram: u64 = roles.iter().enumerate().map(|(role, found)| kept(layer, role, found.bytes)).sum();
            let read: u64 = roles.iter().map(|role| role.bytes.next_multiple_of(ALIGN as u64) + ALIGN as u64).sum();
            (vram.next_multiple_of(256), read.max(vram).next_multiple_of(ALIGN as u64))
        })).collect()
    }

    /// A class per experts' size, each with `count(its layers, all layers, its slot bytes)` VRAM
    /// slots, and its layers' share of `host_budget`.
    fn build(gpu: &mut Backend, layout: Arc<Layout>, reader: Arc<Reader>, host_budget: u64, count: &dyn Fn(u64, u64, u64) -> usize)
             -> Result<DeviceExperts> {
        let sizes = Self::sizes(&layout.layers, &|layer, role, bytes| layout.kept(layer, role, bytes));
        let mut distinct: Vec<(u64, u64)> = sizes.iter().flatten().copied().collect();
        distinct.sort_unstable();
        distinct.dedup();
        let class_of: Vec<Option<usize>> = sizes.iter()
            .map(|size| size.map(|size| distinct.iter().position(|&other| other == size).unwrap())).collect();
        let layers = class_of.iter().flatten().count() as u64;
        let mut classes = Vec::new();
        let mut all_slots = Vec::new();
        for (index, &(vram_bytes, host_bytes)) in distinct.iter().enumerate() {
            let share = class_of.iter().filter(|class| **class == Some(index)).count() as u64;
            let count = count(share, layers, vram_bytes);
            let host_capacity = ((host_budget * share / layers) / host_bytes) as usize;
            let reserve = (count / 64).clamp(4, MAX_RESERVE);
            // Small tiers (and staging alone) take one arena of just their size.
            let per_arena = ((ARENA as u64 / host_bytes) as usize).clamp(1, host_capacity.max(reserve));
            let pinner = Pinner::new(host_capacity.max(reserve).div_ceil(per_arena), per_arena * host_bytes as usize);
            let first = all_slots.len();
            all_slots.extend((0..count).map(|_| VramSlot {
                key: None, ready: None, guards: [None; 2], waiting: None, used: 0, speculative: false, held: false, class: index,
            }));
            classes.push(Class {
                vram_bytes, host_bytes, vram: gpu.buffer(vram_bytes * count as u64)?, first, count, free: (first..first + count).rev().collect(),
                lru: BTreeSet::new(), reserve, host_capacity, per_arena, arenas: Vec::new(), pinner: Some(pinner), hosts: 0,
                host_free: VecDeque::new(), host_lru: BTreeSet::new(),
            });
        }
        let (done_tx, done_rx) = channel();
        Ok(DeviceExperts {
            layout, reader, classes, class_of, slots: all_slots, map: HashMap::new(), host: Vec::new(), host_map: HashMap::new(),
            done_tx, done_rx, clock: 0, in_use: Vec::new(), requested: Vec::new(), speculative: HashMap::new(), reading: 0,
            counts: Counts::default(), settled: false,
        })
    }

    pub fn counts(&self) -> Counts {
        self.counts
    }

    /// The most experts of one layer a pass should acquire at a time: half of the smallest
    /// class's slots beyond its reserve, so the next wave's prefetch fits beside it.
    pub fn wave_limit(&self) -> usize {
        self.classes.iter().map(|class| class.count.saturating_sub(class.reserve) / 2).min().unwrap_or(0).max(1)
    }

    /// When each of `layer`'s `experts` is in a VRAM slot with no fill pending: their slots'
    /// addresses in that order and the slots' bytes (else None). For a caller that then reads them
    /// without acquiring: it must keep them from eviction (slots for every expert and their
    /// reserve, no other expert brought in), and its compute stream must already wait for their
    /// uploads (an acquire of each did).
    pub fn resident_bases(&self, layer: usize, experts: &[usize]) -> Option<(Vec<u64>, u64)> {
        let class = self.class_of.get(layer).copied().flatten()?;
        let bases = experts.iter().map(|&expert| {
            let &slot = self.map.get(&(layer as u32, expert as u32))?;
            let entry = &self.slots[slot];
            (entry.waiting.is_none() && entry.key == Some((layer as u32, expert as u32))).then(|| self.slot_buffer(slot).base() as u64)
        }).collect::<Option<Vec<u64>>>()?;
        Some((bases, self.classes[class].vram_bytes))
    }

    /// Whether `layer`'s `expert` holds a VRAM slot (filled, or its fill queued).
    pub fn in_vram(&self, layer: usize, expert: usize) -> bool {
        self.map.contains_key(&(layer as u32, expert as u32))
    }

    /// Whether the host tier holds `layer`'s `expert` (outside VRAM), read or written back.
    pub fn in_host(&self, layer: usize, expert: usize) -> bool {
        let key = (layer as u32, expert as u32);
        !self.map.contains_key(&key) && self.host_map.get(&key).is_some_and(|&host| {
            self.host[host].failed.is_none() && !matches!(self.host[host].content, Content::Reading(_))
        })
    }

    /// The host tier's copy of `layer`'s `expert` for the CPU to compute from: each role's bytes
    /// in host memory (gate, up, down), the expert touched as used (None when the tier holds no
    /// ready copy; a write-back still in flight is waited for). The bytes stay put until this
    /// store next fills, reads or evicts.
    pub fn host_roles(&mut self, gpu: &mut Backend, layer: usize, expert: usize) -> Result<Option<[*const u8; 3]>> {
        if !self.in_host(layer, expert) {
            return Ok(None);
        }
        let host = self.host_map[&(layer as u32, expert as u32)];
        let base = self.host_pointer(host).cast_const();
        let roles = match self.host[host].content {
            Content::Read => {
                let plan = self.layout.read_plan(layer, expert)?;
                // SAFETY: each role's bytes lie inside the host slot (the read plan's spans).
                [0, 1, 2].map(|role| unsafe { base.add(plan[role].data as usize) })
            }
            Content::Packed(written) => {
                gpu.mark_done(written, true)?;
                let bytes: Vec<u64> = self.layout.roles(layer)?.iter().map(|role| role.bytes).collect();
                // SAFETY: the roles lie back to back inside the host slot, as written back.
                [0, bytes[0], bytes[0] + bytes[1]].map(|at| unsafe { base.add(at as usize) })
            }
            Content::Reading(_) => return Ok(None),
        };
        self.clock += 1;
        let entry = &mut self.host[host];
        let class = &mut self.classes[entry.class];
        class.host_lru.remove(&(entry.used, host));
        entry.used = self.clock;
        class.host_lru.insert((entry.used, host));
        Ok(Some(roles))
    }

    /// VRAM and host slots.
    pub fn capacity(&self) -> (usize, usize) {
        (self.slots.len(), self.classes.iter().map(|class| class.host_capacity).sum())
    }

    /// Make these experts of `layer` resident here; the compute stream waits for their copies.
    /// Their slots (each roles one after another) stay until [`release`](Self::release).
    pub fn acquire(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize]) -> Result<Vec<Buffer>> {
        self.settled = false;
        let started = std::time::Instant::now();
        let held = self.in_use.len();
        let result = self.acquire_slots(gpu, layer, experts);
        if result.is_err() {
            for slot in self.in_use.split_off(held) {
                self.slots[slot].held = false;
            }
        }
        self.counts.store_seconds += started.elapsed().as_secs_f64();
        result
    }

    /// Start the fills of these experts of `layer` that are missing (as urgent as an acquire's),
    /// without waiting: when several devices each acquire, every device's reads start before any
    /// waits. The next [`acquire`](Self::acquire) takes them (counted here) and waits.
    pub fn request(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize]) -> Result<()> {
        self.settled = false;
        let started = std::time::Instant::now();
        let result = (|| {
            self.pump(gpu, false)?;
            // Hits first, as in acquire; all kept from eviction until the acquire's release.
            for &expert in experts {
                let key = (layer as u32, expert as u32);
                if let Some(&slot) = self.map.get(&key) {
                    self.counts.hits += 1;
                    self.hold(slot);
                    self.requested.push(key);
                }
            }
            for &expert in experts {
                let key = (layer as u32, expert as u32);
                if !self.map.contains_key(&key) {
                    let slot = self.fill(gpu, key, true)?;
                    self.hold(slot);
                    self.requested.push(key);
                }
            }
            Ok(())
        })();
        self.counts.store_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn acquire_slots(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize]) -> Result<Vec<Buffer>> {
        self.pump(gpu, false)?;
        // Every hit is held before any miss is filled: a miss never evicts a slot this same
        // request uses later. Experts a request() started were counted there; those of a later
        // acquire of this layer (a request may run ahead of an acquire of other experts) wait for it.
        let (requested, later): (Vec<Key>, Vec<Key>) = std::mem::take(&mut self.requested).into_iter()
            .filter(|key| key.0 == layer as u32)
            .partition(|key| experts.contains(&(key.1 as usize)));
        self.requested = later;
        let mut slots = vec![usize::MAX; experts.len()];
        for (index, &expert) in experts.iter().enumerate() {
            let key = (layer as u32, expert as u32);
            if let Some(&slot) = self.map.get(&key) {
                if !requested.contains(&key) {
                    self.counts.hits += 1;
                }
                self.hold(slot);
                slots[index] = slot;
            }
        }
        for (index, &expert) in experts.iter().enumerate() {
            if slots[index] == usize::MAX {
                // (a repeated expert finds the slot its first occurrence filled)
                let key = (layer as u32, expert as u32);
                let slot = match self.map.get(&key) {
                    Some(&slot) => slot,
                    None => self.fill(gpu, key, true)?,
                };
                self.hold(slot);
                slots[index] = slot;
            }
        }
        // This layer's prefetches it did not take: reads not yet started are skipped, so the
        // drive's time goes to what passes need. Those that landed keep their recency: demoting
        // them to the eviction end sent ~25% more experts back through the host tier, since a
        // later token's route often takes them.
        for slot in self.speculative.remove(&(layer as u32)).unwrap_or_default() {
            let entry = &mut self.slots[slot];
            if entry.speculative && entry.key.is_some_and(|key| key.0 == layer as u32) {
                entry.speculative = false;
                self.counts.prefetch_unused += 1;
                if let Some(host) = entry.waiting {
                    self.host[host].cancel.store(true, Ordering::Relaxed);
                }
            }
        }
        // Fills waiting for the drive upload as their reads land. A prefetch an earlier pass of this
        // layer cancelled may be taken before its read lands: the read then drops the expert, and
        // it is read again here (an urgent read, never cancelled, so this ends).
        loop {
            while slots.iter().any(|&slot| self.slots[slot].waiting.is_some()) {
                self.pump(gpu, true)?;
            }
            let mut again = false;
            for (index, &expert) in experts.iter().enumerate() {
                let key = (layer as u32, expert as u32);
                if self.slots[slots[index]].key != Some(key) {
                    let slot = match self.map.get(&key) {
                        Some(&slot) => slot,
                        None => self.fill(gpu, key, true)?,
                    };
                    self.hold(slot);
                    slots[index] = slot;
                    again = true;
                }
            }
            if !again {
                break;
            }
        }
        // Each upload stream runs in order: its newest mark among these slots covers its others.
        for stream in [Stream::Upload, Stream::LaterUpload] {
            let newest = slots.iter().filter_map(|&slot| self.slots[slot].ready.filter(|ready| ready.0 == stream)).map(|ready| ready.1).max();
            let Some(mark) = newest else { continue };
            if gpu.mark_done(mark, false)? {
                for &slot in &slots {
                    let ready = &mut self.slots[slot].ready;
                    *ready = ready.filter(|&(other, at)| other != stream || at > mark);
                }
            } else {
                gpu.wait_mark(Stream::Compute, mark)?;
            }
        }
        Ok(slots.iter().map(|&slot| self.slot_buffer(slot)).collect())
    }

    /// Start bringing these experts of `layer` in without waiting: from the host tier at once,
    /// and with `drive`, from the drive as reads land. A wrong guess costs a slot and bus time
    /// (and drive time: the scarcest), never a result. Its uploads queue apart from those a pass
    /// waits for.
    pub fn prefetch(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize], drive: bool) -> Result<()> {
        self.settled = false;
        let class = self.class_of.get(layer).copied().flatten().ok_or_else(|| format!("layer {layer} has no routed experts"))?;
        for &expert in experts {
            let key = (layer as u32, expert as u32);
            if !self.map.contains_key(&key) && (drive || self.host_map.contains_key(&key)) {
                // A guess never takes a slot a pass holds or waits for: with none left, stop.
                if self.classes[class].free.is_empty() && self.victim(class).is_err() {
                    break;
                }
                let slot = self.fill(gpu, key, false)?;
                self.slots[slot].speculative = true;
                self.speculative.entry(layer as u32).or_default().push(slot);
                self.counts.prefetched += 1;
            }
        }
        self.pump(gpu, false)
    }

    /// After the ops that read the acquired slots are queued: those slots may be evicted once
    /// the compute stream passes this point.
    pub fn release(&mut self, gpu: &mut Backend) -> Result<()> {
        if self.in_use.is_empty() {
            return Ok(());
        }
        let mark = gpu.mark(Stream::Compute)?;
        for slot in std::mem::take(&mut self.in_use) {
            self.slots[slot].guards[0] = Some(mark);
            self.slots[slot].held = false;
        }
        self.requested.clear();
        self.keep_free(gpu)
    }

    fn slot_buffer(&self, slot: usize) -> Buffer {
        let class = &self.classes[self.slots[slot].class];
        class.vram.slice((slot - class.first) as u64 * class.vram_bytes, class.vram_bytes)
    }

    /// Mark `slot` just used (its place in the LRU order).
    fn touch(&mut self, slot: usize) {
        self.clock += 1;
        let entry = &mut self.slots[slot];
        let lru = &mut self.classes[entry.class].lru;
        if entry.key.is_some() {
            lru.remove(&(entry.used, slot));
            lru.insert((self.clock, slot));
        }
        entry.used = self.clock;
    }

    /// Keep `slot` until the next release, as just used.
    fn hold(&mut self, slot: usize) {
        self.touch(slot);
        let entry = &mut self.slots[slot];
        entry.speculative = false;
        entry.held = true;
        self.in_use.push(slot);
    }

    /// Give VRAM slot `slot` (holding `key`) up: out of the map and the LRU order, onto the free list.
    fn empty_slot(&mut self, slot: usize) -> Option<Key> {
        let entry = &mut self.slots[slot];
        let key = entry.key.take()?;
        entry.speculative = false;
        let class = &mut self.classes[entry.class];
        class.lru.remove(&(entry.used, slot));
        class.free.push(slot);
        self.map.remove(&key);
        Some(key)
    }

    /// A VRAM slot for `key`, its fill queued (uploaded now, or once its drive reads land).
    fn fill(&mut self, gpu: &mut Backend, key: Key, urgent: bool) -> Result<usize> {
        let class = self.class_of.get(key.0 as usize).copied().flatten().ok_or_else(|| format!("layer {} has no routed experts", key.0))?;
        let slot = self.free_slot(gpu, class)?;
        self.map.insert(key, slot);
        self.slots[slot].key = Some(key);
        self.classes[class].lru.insert((self.slots[slot].used, slot));
        self.touch(slot);
        let stream = if urgent { Stream::Upload } else { Stream::LaterUpload };
        let filled = match self.host_map.get(&key).copied() {
            Some(host) if !matches!(self.host[host].content, Content::Reading(_)) => {
                self.counts.host_hits += 1;
                self.upload(gpu, slot, host, stream).inspect_err(|_| self.discard(host, Some(slot)))
            }
            Some(host) => {
                self.slots[slot].waiting = Some(host);
                self.host[host].waiter = Some(slot);
                Ok(())
            }
            None => {
                self.counts.drive_reads += 1;
                self.read(gpu, key, urgent).map(|host| {
                    self.slots[slot].waiting = Some(host);
                    self.host[host].waiter = Some(slot);
                })
            }
        };
        if filled.is_err() && self.slots[slot].key == Some(key) {
            self.empty_slot(slot);
        }
        filled.map(|()| slot)
    }

    /// Upload host slot `host` into VRAM slot `slot` on `stream`; the host tier gives the expert up.
    fn upload(&mut self, gpu: &mut Backend, slot: usize, host: usize, stream: Stream) -> Result<()> {
        let (layer, expert) = self.slots[slot].key.ok_or("an upload into an empty slot")?;
        if let Some(error) = &self.host[host].failed {
            return Err(error.clone());
        }
        // The slot's previous expert: its last compute use and any download from it come first.
        for guard in std::mem::take(&mut self.slots[slot].guards).into_iter().flatten() {
            gpu.wait_mark(stream, guard)?;
        }
        let base = self.host_pointer(host);
        let target = self.slot_buffer(slot);
        match self.host[host].content {
            Content::Read => {
                let mut at = 0;
                for span in self.layout.read_plan(layer as usize, expert as usize)? {
                    // SAFETY: the host slot holds this expert as read, and gives it up below (only a
                    // resident store keeps columns: no CPU reads its host slots).
                    let bytes = unsafe { span.keep_columns(base) };
                    // SAFETY: the host slot holds this expert until the mark below completes (busy).
                    unsafe { gpu.transfer(stream, &target, at, base.add(span.data as usize), bytes)? };
                    at += bytes;
                }
            }
            Content::Packed(written) => {
                gpu.wait_mark(stream, written)?;
                let bytes: u64 = self.layout.roles(layer as usize)?.iter().map(|role| role.bytes).sum();
                // SAFETY: as above.
                unsafe { gpu.transfer(stream, &target, 0, base, bytes)? };
            }
            Content::Reading(_) => return Err("an upload of an expert still being read".into()),
        }
        let mark = gpu.mark(stream)?;
        self.slots[slot].ready = Some((stream, mark));
        self.slots[slot].waiting = None;
        let entry = &mut self.host[host];
        entry.busy = Some(mark);
        entry.waiter = None;
        if let Some(key) = entry.key.take() {
            self.host_map.remove(&key);
            let class = &mut self.classes[entry.class];
            class.host_lru.remove(&(entry.used, host));
            class.host_free.push_back(host);
        }
        Ok(())
    }

    /// Start reading `key` from the drive into a host slot (a staging slot when the host tier
    /// holds nothing).
    fn read(&mut self, gpu: &mut Backend, key: Key, urgent: bool) -> Result<usize> {
        let class = self.class_of[key.0 as usize].ok_or("a read for a layer without routed experts")?;
        let host = self.free_host(gpu, class)?;
        let plan = self.layout.read_plan(key.0 as usize, key.1 as usize)?;
        let base = self.host_pointer(host);
        let cancel = Arc::new(AtomicBool::new(false));
        let mut jobs = Vec::new();
        for span in plan {
            let handle = self.layout.files[span.file].0.clone();
            let mut done = 0;
            while done < span.len {
                let piece = CHUNK.min(span.len - done);
                jobs.push(ReadJob {
                    file: handle.clone(), offset: span.offset + done, len: piece,
                    // SAFETY: inside the host slot (its class's host_bytes covers every role's whole blocks).
                    target: HostPointer(unsafe { base.add((span.at + done) as usize) }),
                    verifier: self.layout.verifier.clone().map(|verify| (verify, span.file)),
                    done: self.done_tx.clone(), slot: host, cancelled: cancel.clone(),
                });
                done += piece;
            }
        }
        let entry = &mut self.host[host];
        entry.key = Some(key);
        entry.content = Content::Reading(jobs.len() as u32);
        entry.cancel = cancel;
        entry.skipped = false;
        self.reading += 1;
        self.host_map.insert(key, host);
        self.reader.submit(jobs, urgent);
        Ok(host)
    }

    /// Take finished drive reads; upload the fills waiting for them. With `block`, wait for at
    /// least one read first.
    ///
    /// An expert whose read failed (an I/O error, or a block that did not verify) leaves both
    /// tiers: its bytes are never uploaded, and a later use reads it again. Only a pass holding
    /// the expert fails with it; a read no pass waits for (a prefetch) fails nothing, whichever
    /// call happens to take it, and is counted. Every landed read is accounted for before the
    /// first error returns.
    fn pump(&mut self, gpu: &mut Backend, block: bool) -> Result<()> {
        let mut landed = Vec::new();
        if block {
            if self.reading == 0 {
                return Err("a wait for expert reads with none in flight".into());
            }
            let started = std::time::Instant::now();
            let message = self.done_rx.recv().map_err(|_| "the expert reader stopped")?;
            self.counts.wait_seconds += started.elapsed().as_secs_f64();
            landed.push(message);
        }
        landed.extend(self.done_rx.try_iter());
        let mut first_error = None;
        for (host, result) in landed {
            let entry = &mut self.host[host];
            match result {
                Landed::Read => {}
                Landed::Skipped => entry.skipped = true,
                Landed::Failed(error) => {
                    entry.failed.get_or_insert(error);
                }
            }
            let Content::Reading(left) = entry.content else { continue };
            entry.content = if left > 1 { Content::Reading(left - 1) } else { Content::Read };
            if left > 1 {
                continue;
            }
            self.reading -= 1;
            let waiting = entry.waiter;
            if std::mem::take(&mut self.host[host].skipped) {
                // A cancelled prefetch: its slots are free again, nothing failed.
                self.host[host].failed = None;
                self.discard(host, waiting);
                continue;
            }
            if let Some(error) = self.host[host].failed.take() {
                let needed = waiting.is_some_and(|slot| self.slots[slot].held);
                self.discard(host, waiting);
                if needed {
                    first_error.get_or_insert(error);
                } else {
                    self.counts.failed_unwaited += 1;
                }
                continue;
            }
            match waiting {
                Some(slot) => {
                    let stream = if self.slots[slot].speculative { Stream::LaterUpload } else { Stream::Upload };
                    if let Err(error) = self.upload(gpu, slot, host, stream) {
                        self.discard(host, Some(slot));
                        first_error.get_or_insert(error);
                    }
                }
                None => {
                    // Read for nothing that still waits: the host tier keeps it.
                    let entry = &mut self.host[host];
                    self.clock += 1;
                    entry.used = self.clock;
                    self.classes[entry.class].host_lru.insert((entry.used, host));
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Drop an expert whose fill failed from host slot `host` and VRAM slot `slot`.
    fn discard(&mut self, host: usize, slot: Option<usize>) {
        let entry = &mut self.host[host];
        entry.waiter = None;
        if let Some(key) = entry.key.take() {
            self.host_map.remove(&key);
            let class = &mut self.classes[entry.class];
            class.host_lru.remove(&(entry.used, host));
            class.host_free.push_back(host);
        }
        if let Some(slot) = slot {
            let entry = &mut self.slots[slot];
            entry.waiting = None;
            entry.ready = None;
            self.empty_slot(slot);
        }
    }

    /// A free VRAM slot of `class`: one already free, else the least recently used one no pass
    /// holds (written back to the host tier).
    fn free_slot(&mut self, gpu: &mut Backend, class: usize) -> Result<usize> {
        if self.classes[class].free.is_empty() {
            let victim = self.victim(class)?;
            self.evict(gpu, victim)?;
        }
        self.classes[class].free.pop().ok_or_else(|| "no free expert slot after an eviction".into())
    }

    fn victim(&self, class: usize) -> Result<usize> {
        self.classes[class].lru.iter().map(|&(_, slot)| slot)
            .find(|&slot| self.slots[slot].waiting.is_none() && !self.slots[slot].held)
            .ok_or_else(|| "every expert slot of a size is in use by this pass".into())
    }

    /// Keep each class's `reserve` VRAM slots free, writing their experts back to the host tier,
    /// so a miss never waits for a write-back before its upload.
    fn keep_free(&mut self, gpu: &mut Backend) -> Result<()> {
        for class in 0..self.classes.len() {
            while self.classes[class].free.len() < self.classes[class].reserve {
                let victim = self.victim(class)?;
                self.evict(gpu, victim)?;
            }
        }
        Ok(())
    }

    /// Empty VRAM slot `slot` (onto its class's free list), writing its expert back to the host
    /// tier when that has room.
    fn evict(&mut self, gpu: &mut Backend, slot: usize) -> Result<()> {
        let Some(key) = self.empty_slot(slot) else { return Ok(()) };
        let class = self.slots[slot].class;
        if self.classes[class].host_capacity == 0 || self.host_map.contains_key(&key) {
            return Ok(());
        }
        let host = self.free_host(gpu, class)?;
        let bytes: u64 = self.layout.roles(key.0 as usize)?.iter().map(|role| role.bytes).sum();
        if let Some((_, ready)) = self.slots[slot].ready.take() {
            gpu.wait_mark(Stream::Download, ready)?;
        }
        let source = self.slot_buffer(slot);
        // SAFETY: the host slot is reserved for this expert; nothing reads it before the mark.
        unsafe { gpu.transfer(Stream::Download, &source, 0, self.host_pointer(host), bytes)? };
        let mark = gpu.mark(Stream::Download)?;
        self.slots[slot].guards[1] = Some(mark);
        self.clock += 1;
        let entry = &mut self.host[host];
        entry.key = Some(key);
        entry.content = Content::Packed(mark);
        entry.used = self.clock;
        self.classes[class].host_lru.insert((self.clock, host));
        self.host_map.insert(key, host);
        self.counts.writebacks += 1;
        Ok(())
    }

    /// A host slot of `class` to fill: an emptied one whose upload finished, a new one while the
    /// budget allows, else the least recently used expert's (after any copy still using it).
    /// With no host tier, staging slots grow as reads need.
    fn free_host(&mut self, gpu: &mut Backend, class: usize) -> Result<usize> {
        // Emptied slots' uploads finish in order: the oldest is the one to check.
        if let Some(&host) = self.classes[class].host_free.front() {
            let copies = [self.host[host].busy, match self.host[host].content { Content::Packed(mark) => Some(mark), _ => None }];
            let mut reusable = true;
            for mark in copies.into_iter().flatten() {
                reusable &= gpu.mark_done(mark, false)?;
            }
            if reusable {
                self.classes[class].host_free.pop_front();
                self.host[host].busy = None;
                return Ok(host);
            }
        }
        // A new slot while pinned room lasts (or a pinned arena is ready).
        let room = self.classes[class].hosts < self.classes[class].arenas.len() * self.classes[class].per_arena;
        if room || self.take_arena(class, false)? {
            return Ok(self.new_host_slot(class));
        }
        // The least recently used expert's slot (its copies long done), before an emptied slot
        // whose upload is still running.
        let candidate = self.classes[class].host_lru.iter().map(|&(_, host)| host).next()
            .or_else(|| self.classes[class].host_free.front().copied());
        let Some(host) = candidate else {
            // Every slot is being read into: wait for the next arena, or (all handed over) pin
            // one more for staging.
            self.take_arena(class, true)?;
            return Ok(self.new_host_slot(class));
        };
        let started = std::time::Instant::now();
        if let Some(mark) = self.host[host].busy.take() {
            gpu.mark_done(mark, true)?;
        }
        if let Content::Packed(mark) = self.host[host].content {
            gpu.mark_done(mark, true)?;
        }
        self.counts.slot_wait_seconds += started.elapsed().as_secs_f64();
        let entry = &mut self.host[host];
        match entry.key.take() {
            Some(key) => {
                self.host_map.remove(&key);
                self.classes[class].host_lru.remove(&(entry.used, host));
            }
            None => {
                self.classes[class].host_free.pop_front();
            }
        }
        Ok(host)
    }

    /// Take `class`'s next pinned arena: if ready (or, with `wait`, once it is; when its pinner
    /// has handed all over, a new one pinned here). Whether one was taken.
    fn take_arena(&mut self, class: usize, wait: bool) -> Result<bool> {
        let started = std::time::Instant::now();
        let entry = &mut self.classes[class];
        let arena = match &entry.pinner {
            Some(pinner) if wait => pinner.arenas.recv().ok(),
            Some(pinner) => match pinner.arenas.try_recv() {
                Ok(arena) => Some(arena),
                Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(false),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
            },
            None => None,
        };
        let arena = match arena {
            Some(arena) => arena?,
            None if wait => {
                let mut arena = storage::Buffer::new(entry.per_arena * entry.host_bytes as usize);
                // SAFETY: the arena lives in its class until drop, which unpins it first.
                unsafe { transfer::pin_host(arena.as_mut_slice().as_mut_ptr(), arena.capacity())? };
                arena
            }
            None => return Ok(false),
        };
        entry.arenas.push(arena);
        self.counts.pin_seconds += started.elapsed().as_secs_f64();
        Ok(true)
    }

    fn new_host_slot(&mut self, class: usize) -> usize {
        let entry = &mut self.classes[class];
        let (arena, at) = (entry.hosts / entry.per_arena, (entry.hosts % entry.per_arena) * entry.host_bytes as usize);
        entry.hosts += 1;
        self.host.push(HostSlot {
            key: None, content: Content::Read, busy: None, used: 0, failed: None, cancel: Arc::new(AtomicBool::new(false)),
            skipped: false, waiter: None, class, arena, at,
        });
        self.host.len() - 1
    }

    fn host_pointer(&mut self, host: usize) -> *mut u8 {
        let entry = &self.host[host];
        let (arena, at) = (entry.arena, entry.at);
        let arena = &mut self.classes[entry.class].arenas[arena];
        // SAFETY: the slot lies inside its arena (at + host_bytes <= the arena's size).
        unsafe { arena.as_mut_slice().as_mut_ptr().add(at) }
    }
}

impl DeviceExperts {
    /// Wait for every read and copy in flight; dropping the store frees its host memory only
    /// after this (else it leaks it, which a copy still in flight cannot then corrupt).
    pub fn settle(&mut self, gpu: &mut Backend) -> Result<()> {
        while self.reading > 0 {
            self.pump(gpu, true)?;
        }
        for stream in [Stream::Upload, Stream::LaterUpload, Stream::Download] {
            let mark = gpu.mark(stream)?;
            gpu.mark_done(mark, true)?;
        }
        self.settled = true;
        Ok(())
    }

    /// The experts held, least recently used first: those in VRAM slots, and those only in the
    /// host tier (read, or written back). A later run warms up with them ([`warm`](Self::warm)).
    pub fn contents(&self) -> Contents {
        let mut vram: Vec<(u64, Key)> = self.classes.iter().flat_map(|class| class.lru.iter())
            .filter_map(|&(used, slot)| self.slots[slot].key.map(|key| (used, key)))
            .collect();
        let mut host: Vec<(u64, Key)> = self.classes.iter().flat_map(|class| class.host_lru.iter())
            .filter_map(|&(used, slot)| self.host[slot].key.filter(|key| !self.map.contains_key(key)).map(|key| (used, key)))
            .collect();
        vram.sort_unstable();
        host.sort_unstable();
        let keys = |list: Vec<(u64, Key)>| list.into_iter().map(|(_, (layer, expert))| (layer as usize, expert as usize)).collect();
        (keys(vram), keys(host))
    }

    /// Warm up with experts an earlier run held ([`contents`](Self::contents)), least recently
    /// used first: `host` read into the host tier (when there is one), `vram` brought into VRAM
    /// slots as prefetches (a pass's fills go first, and a pass skips its layer's reads not yet
    /// started). The reads queue behind every pass's. Experts of layers this store does not hold
    /// are skipped. Returns the reads still in flight.
    pub fn warm(&mut self, gpu: &mut Backend, vram: &[(usize, usize)], host: &[(usize, usize)]) -> Result<usize> {
        self.settled = false;
        for &(layer, expert) in host {
            let Some(class) = self.class_of.get(layer).copied().flatten() else { continue };
            let key = (layer as u32, expert as u32);
            if self.classes[class].host_capacity == 0 || self.map.contains_key(&key) || self.host_map.contains_key(&key) {
                continue;
            }
            self.read(gpu, key, false)?;
        }
        for &(layer, expert) in vram {
            if self.class_of.get(layer).copied().flatten().is_some() {
                self.prefetch(gpu, layer, &[expert], true)?;
            }
        }
        self.poll(gpu)
    }

    /// Take the reads that have landed (uploading those a fill waits for), without waiting;
    /// the reads still in flight.
    pub fn poll(&mut self, gpu: &mut Backend) -> Result<usize> {
        self.pump(gpu, false)?;
        Ok(self.reading)
    }
}

impl Drop for DeviceExperts {
    fn drop(&mut self) {
        // Reads still in flight write into the arenas: let them land before the memory goes.
        let mut reading: u64 = self.host.iter().map(|slot| match slot.content {
            Content::Reading(left) => left as u64,
            _ => 0,
        }).sum();
        while reading > 0 && self.done_rx.recv().is_ok() {
            reading -= 1;
        }
        for class in &mut self.classes {
            // Stop the pinner and wait for it; the arenas it handed over but nobody took were
            // never used by a device: unpin each before it is freed.
            if let Some(mut pinner) = class.pinner.take() {
                pinner.stop.store(true, Ordering::Relaxed);
                if let Some(thread) = pinner.thread.take() {
                    let _ = thread.join();
                }
                while let Ok(Ok(mut arena)) = pinner.arenas.try_recv() {
                    // SAFETY: pinned by the pinner, never used by a device.
                    let _ = unsafe { transfer::unpin_host(arena.as_mut_slice().as_mut_ptr()) };
                }
            }
            if !self.settled {
                for arena in std::mem::take(&mut class.arenas) {
                    std::mem::forget(arena);
                }
                continue;
            }
            for arena in &mut class.arenas {
                // SAFETY: pinned before use; settle saw every copy from it finish.
                let _ = unsafe { transfer::unpin_host(arena.as_mut_slice().as_mut_ptr()) };
            }
        }
    }
}
