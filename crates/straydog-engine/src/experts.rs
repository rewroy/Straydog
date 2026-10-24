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
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::gpu::{Backend, Result};
use crate::ops::Buffer;
use crate::storage::{self, ALIGN};
use crate::transfer::{self, Mark, Stream};

/// One drive read: an expert's ~40 pieces go to as many readers at once, so one miss reads at
/// near the drive's rate (1 MiB pieces waited ~10% longer for a token's misses).
const CHUNK: u64 = 256 << 10;
/// Large-row prefill and explicit bulk warm-up; decode/verify keep `CHUNK`.
const BULK_CHUNK: u64 = 1 << 20;
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
    done: Sender<(usize, Chunk, Landed)>,
    slot: usize,
    chunk: Chunk,
    /// Set when the read's expert is no longer wanted (a prefetch the pass did not need): a
    /// chunk not yet started is skipped.
    cancelled: Arc<AtomicBool>,
}

/// A chunk's role bytes, for a fill that uploads each chunk as it lands (see
/// [`DeviceExperts::read`]): `bytes` bytes at `host` in the host slot go to `vram` in the VRAM
/// slot; `head` for a chunk of the first two roles (gate and up).
#[derive(Debug, Clone, Copy)]
struct Chunk {
    host: u64,
    vram: u64,
    bytes: u64,
    head: bool,
}

/// How a chunk's read ended.
enum Landed {
    Read,
    Skipped,
    Failed(String),
}

struct ReadQueue {
    jobs: VecDeque<ReadJob>,
    heads: VecDeque<ReadJob>,
    tails: VecDeque<ReadJob>,
    stopped: bool,
    reading: usize,
    limit: usize,
    head_priority: bool,
    consecutive_heads: usize,
}

impl ReadQueue {
    fn pop(&mut self) -> Option<ReadJob> {
        // Bound head priority too: a sustained head stream must still finish earlier downs.
        if !self.heads.is_empty() && (self.tails.is_empty() || self.consecutive_heads < self.limit) {
            self.consecutive_heads += 1;
            return self.heads.pop_front();
        }
        if let Some(job) = self.tails.pop_front() {
            self.consecutive_heads = 0;
            return Some(job);
        }
        self.consecutive_heads = 0;
        self.jobs.pop_front()
    }
}

type Queue = Arc<(Mutex<ReadQueue>, Condvar)>;

/// A data-read permit. Error, cancellation and panic also return it; verification holds none.
struct ReadPermit(Queue);
impl Drop for ReadPermit {
    fn drop(&mut self) {
        let mut state = self.0.0.lock().unwrap();
        state.reading -= 1;
        drop(state);
        self.0.1.notify_one();
    }
}

/// Threads reading expert chunks, with optional data-I/O admission separate from hashing.
pub struct Reader {
    queue: Queue,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Reader {
    pub fn new(threads: usize) -> Reader {
        Self::bounded(threads, threads.max(1), false)
    }

    /// One process-wide DS expert pool, shared by devices and layer groups: thirty-two workers
    /// share sixteen data-read admissions, returned before verification. No worker consumes a queued
    /// background job before admission, so newly urgent work retains its priority.
    pub fn shared() -> Arc<Reader> {
        static SHARED: OnceLock<Arc<Reader>> = OnceLock::new();
        SHARED.get_or_init(|| Arc::new(Self::bounded(32, 16, true))).clone()
    }

    fn bounded(threads: usize, limit: usize, head_priority: bool) -> Reader {
        let queue: Queue = Arc::new((Mutex::new(ReadQueue {
            jobs: VecDeque::new(), heads: VecDeque::new(), tails: VecDeque::new(), stopped: false,
            reading: 0, limit: limit.max(1).min(threads.max(1)), head_priority, consecutive_heads: 0,
        }), Condvar::new()));
        let workers = (0..threads.max(1)).map(|_| {
            let queue = queue.clone();
            std::thread::spawn(move || loop {
                let (job, permit) = {
                    let mut state = queue.0.lock().unwrap();
                    loop {
                        if state.reading < state.limit {
                            if let Some(job) = state.pop() {
                                state.reading += 1;
                                break (job, ReadPermit(queue.clone()));
                            }
                        }
                        if state.stopped && state.jobs.is_empty() && state.heads.is_empty() && state.tails.is_empty() { return; }
                        state = queue.1.wait(state).unwrap();
                    }
                };
                // A verifier panic fails just this read; the permit is already returned.
                let landed = if job.cancelled.load(Ordering::Relaxed) {
                    drop(permit);
                    Landed::Skipped
                } else {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_read(&job, permit))) {
                        Ok(result) => result.map_or_else(Landed::Failed, |()| Landed::Read),
                        Err(_) => Landed::Failed(format!("the read at {} panicked", job.offset)),
                    }
                };
                let _ = job.done.send((job.slot, job.chunk, landed));
            })
        }).collect();
        Reader { queue, workers }
    }

    /// Pass reads precede prefetches. The shared pool is FIFO across devices' heads, with bounded
    /// priority over tails; legacy pools retain their existing urgent-front ordering.
    fn submit(&self, jobs: Vec<ReadJob>, urgent: bool) {
        if jobs.is_empty() { return; }
        let mut state = self.queue.0.lock().unwrap();
        if urgent && state.head_priority {
            for job in jobs {
                if job.chunk.head { state.heads.push_back(job); } else { state.tails.push_back(job); }
            }
        } else if urgent {
            for job in jobs.into_iter().rev() { state.jobs.push_front(job); }
        } else {
            state.jobs.extend(jobs);
        }
        drop(state);
        self.queue.1.notify_all();
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.queue.0.lock().unwrap().stopped = true;
        self.queue.1.notify_all();
        for worker in self.workers.drain(..) { let _ = worker.join(); }
    }
}

fn run_read(job: &ReadJob, permit: ReadPermit) -> std::result::Result<(), String> {
    // SAFETY: the target is `len` bytes of a host slot reserved for this read (see HostPointer).
    let target = unsafe { std::slice::from_raw_parts_mut(job.target.0, job.len as usize) };
    read_fully(&job.file, job.offset, target).map_err(|exc| format!("expert read at {}: {exc}", job.offset))?;
    drop(permit); // Hashing and proof checks do not consume one of the sixteen data-read admits.
    if let Some((verify, file)) = &job.verifier { verify(*file, job.offset, target)?; }
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
    /// Its waiter uploads each chunk as it lands (an urgent fill's read), rather than the whole
    /// expert once every chunk is in; `head` chunks (gate and up) still to come, and whether a
    /// chunk's copy was queued.
    streamed: bool,
    head: u32,
    copied: bool,
    class: usize,
    /// Its arena in the class, and its offset there.
    arena: usize,
    at: usize,
}

struct VramSlot {
    key: Option<Key>,
    /// The upload (and its stream) after this slot's fill, until known complete.
    ready: Option<(Stream, Mark)>,
    /// That upload's mark once its gate and up are in (down follows): see
    /// [`DeviceExperts::acquire_gate_up`].
    ready_first: Option<(Stream, Mark)>,
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
    /// A slot in a region lent to the store ([`DeviceExperts::lend`]): its bytes there (else its
    /// class's).
    place: Option<Buffer>,
    /// A lent slot its lender has back ([`DeviceExperts::reclaim`]): no expert goes in.
    taken_back: bool,
}

/// A slot of the region lent to the store: where its bytes begin there, and whether the lender
/// has it back ([`DeviceExperts::reclaim`]).
#[derive(Clone, Copy)]
struct LentSlot {
    slot: usize,
    offset: u64,
    out: bool,
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
    /// Free slots, the longest free first: a fill takes the slot whose write-back (an evicted
    /// expert's, to the host tier) had the longest to finish, so its upload seldom waits for one.
    free: VecDeque<usize>,
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
    /// Bytes queued for drive reads (whole blocks), a pass's and a prefetch's alike.
    pub drive_bytes: u64,
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
            drive_bytes: self.drive_bytes + other.drive_bytes,
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

/// An upload whose gate and up are queued and whose down is not yet: the bytes after `at`.
struct Tail {
    slot: usize,
    host: usize,
    stream: Stream,
    at: u64,
}

/// One device's expert slots, and its share of the host tier.
pub struct DeviceExperts {
    /// While a request queues its misses: their downs wait in `tails` until every gate and up is
    /// queued (DS41_HEADS_FIRST=0: each expert whole in turn).
    deferring: bool,
    heads_first: bool,
    tails: Vec<Tail>,
    layout: Arc<Layout>,
    reader: Arc<Reader>,
    classes: Vec<Class>,
    /// Per layer: its class.
    class_of: Vec<Option<usize>>,
    slots: Vec<VramSlot>,
    map: HashMap<Key, usize>,
    host: Vec<HostSlot>,
    host_map: HashMap<Key, usize>,
    done_tx: Sender<(usize, Chunk, Landed)>,
    done_rx: Receiver<(usize, Chunk, Landed)>,
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
    /// Changes to which experts VRAM slots hold and which of them are ready (see
    /// [`vram_generation`](Self::vram_generation)).
    vram_changes: u64,
    /// The slots of the region lent to the store, from its start.
    lent: Vec<LentSlot>,
    /// Slots acquired with only their gate and up waited for ([`acquire_gate_up`](Self::acquire_gate_up)).
    rest: Vec<usize>,
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
                key: None, ready: None, ready_first: None, guards: [None; 2], waiting: None, used: 0, speculative: false, held: false, class: index,
                place: None,
                taken_back: false,
            }));
            classes.push(Class {
                vram_bytes, host_bytes, vram: gpu.buffer(vram_bytes * count as u64)?, first, count, free: (first..first + count).collect(),
                lru: BTreeSet::new(), reserve, host_capacity, per_arena, arenas: Vec::new(), pinner: Some(pinner), hosts: 0,
                host_free: VecDeque::new(), host_lru: BTreeSet::new(),
            });
        }
        let (done_tx, done_rx) = channel();
        Ok(DeviceExperts {
            deferring: false,
            heads_first: std::env::var("DS41_HEADS_FIRST").map_or(true, |value| value != "0"),
            tails: Vec::new(),
            layout, reader, classes, class_of, slots: all_slots, map: HashMap::new(), host: Vec::new(), host_map: HashMap::new(),
            done_tx, done_rx, clock: 0, in_use: Vec::new(), requested: Vec::new(), speculative: HashMap::new(), reading: 0,
            counts: Counts::default(), settled: false, vram_changes: 0, lent: Vec::new(), rest: Vec::new(),
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

    /// The most experts of one layer a pass can hold at once (acquired together, no prefetch
    /// beside them): the smallest class's slots beyond its reserve.
    pub fn pass_limit(&self) -> usize {
        self.classes.iter().map(|class| class.count.saturating_sub(class.reserve)).min().unwrap_or(0)
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
        self.acquire_with(gpu, layer, experts, false)
    }

    /// [`acquire`](Self::acquire) where the compute stream waits only for the fills' gate and up:
    /// their downs may still be landing until [`await_rest`](Self::await_rest), which must come
    /// before anything reads them (a release before it fails).
    pub fn acquire_gate_up(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize]) -> Result<Vec<Buffer>> {
        self.acquire_with(gpu, layer, experts, true)
    }

    /// The compute stream waits for the rest of the fills [`acquire_gate_up`](Self::acquire_gate_up)
    /// took (their downs), the host first for the reads of those still landing.
    pub fn await_rest(&mut self, gpu: &mut Backend) -> Result<()> {
        let rest = std::mem::take(&mut self.rest);
        let started = std::time::Instant::now();
        let result = (|| {
            while rest.iter().any(|&slot| self.slots[slot].waiting.is_some()) {
                self.pump(gpu, true)?;
            }
            self.wait_uploads(gpu, &rest, false)
        })();
        self.counts.store_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn acquire_with(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize], first: bool) -> Result<Vec<Buffer>> {
        self.settled = false;
        let started = std::time::Instant::now();
        let held = self.in_use.len();
        let result = self.acquire_slots(gpu, layer, experts, first);
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
        self.request_with(gpu, layer, experts, CHUNK)
    }

    /// Large-row prefill demand, including its urgent first wave.
    pub fn request_bulk(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize]) -> Result<()> {
        self.request_with(gpu, layer, experts, BULK_CHUNK)
    }

    fn request_with(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize], chunk: u64) -> Result<()> {
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
            // The misses' gates and ups go up first, then their downs: a pass waits for every
            // miss's gate and up before its downs, and one expert's down would hold the next's back.
            self.deferring = true;
            let filled = (|| {
                for &expert in experts {
                    let key = (layer as u32, expert as u32);
                    if !self.map.contains_key(&key) {
                        let slot = self.fill(gpu, key, true, chunk)?;
                        self.hold(slot);
                        self.requested.push(key);
                    }
                }
                Ok(())
            })();
            self.deferring = false;
            let tails = self.upload_tails(gpu);
            filled.and(tails)
        })();
        self.counts.store_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn acquire_slots(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize], first: bool) -> Result<Vec<Buffer>> {
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
        // (an acquire of the gates and ups first takes its misses' downs after all of them, as a
        // request does)
        self.deferring = first;
        let filled = (|| {
            for (index, &expert) in experts.iter().enumerate() {
                if slots[index] == usize::MAX {
                    // (a repeated expert finds the slot its first occurrence filled)
                    let key = (layer as u32, expert as u32);
                    let slot = match self.map.get(&key) {
                        Some(&slot) => slot,
                        None => self.fill(gpu, key, true, CHUNK)?,
                    };
                    self.hold(slot);
                    slots[index] = slot;
                }
            }
            Ok(())
        })();
        self.deferring = false;
        let tails = self.upload_tails(gpu);
        filled.and(tails)?;
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
            while slots.iter().any(|&slot| self.landing(slot, first)) {
                self.pump(gpu, true)?;
            }
            let mut again = false;
            for (index, &expert) in experts.iter().enumerate() {
                let key = (layer as u32, expert as u32);
                if self.slots[slots[index]].key != Some(key) {
                    let slot = match self.map.get(&key) {
                        Some(&slot) => slot,
                        None => self.fill(gpu, key, true, CHUNK)?,
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
        self.wait_uploads(gpu, &slots, first)?;
        if first {
            self.rest.extend(slots.iter().copied().filter(|&slot| self.slots[slot].ready.is_some() || self.slots[slot].waiting.is_some()));
        }
        if slots.iter().any(|&slot| self.slots[slot].taken_back) {
            return Err("an acquire found an expert in a lent slot its lender has back".into());
        }
        Ok(slots.iter().map(|&slot| self.slot_buffer(slot)).collect())
    }

    /// Whether VRAM slot `slot`'s fill still waits for reads: with `first`, for its gate's and up's
    /// (a streamed fill whose chunks of them are all in no longer does).
    fn landing(&self, slot: usize, first: bool) -> bool {
        self.slots[slot].waiting.is_some_and(|host| {
            let entry = &self.host[host];
            !(first && entry.streamed && entry.head == 0 && entry.failed.is_none())
        })
    }

    /// The compute stream waits for these slots' uploads (with `first`, for their gate and up: a
    /// streamed fill's down may still be landing). Each upload stream runs in order: its newest
    /// mark among them covers its others.
    fn wait_uploads(&mut self, gpu: &mut Backend, slots: &[usize], first: bool) -> Result<()> {
        for stream in [Stream::Upload, Stream::LaterUpload] {
            // (an upload known complete has no mark left to wait for)
            let mark_of = |slot: &VramSlot| match first {
                true if slot.waiting.is_some() => slot.ready_first,
                true => slot.ready.and(slot.ready_first),
                false => slot.ready,
            };
            let newest = slots.iter().filter_map(|&slot| mark_of(&self.slots[slot]).filter(|ready| ready.0 == stream)).map(|ready| ready.1).max();
            let Some(mark) = newest else { continue };
            if gpu.mark_done(mark, false)? {
                if !first {
                    for &slot in slots {
                        let entry = &mut self.slots[slot];
                        entry.ready = entry.ready.filter(|&(other, at)| other != stream || at > mark);
                    }
                }
            } else {
                gpu.wait_mark(Stream::Compute, mark)?;
            }
        }
        Ok(())
    }

    /// Start bringing these experts of `layer` in without waiting: from the host tier at once,
    /// and with `drive`, from the drive as reads land. A wrong guess costs a slot and bus time
    /// (and drive time: the scarcest), never a result. Its uploads queue apart from those a pass
    /// waits for.
    pub fn prefetch(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize], drive: bool) -> Result<()> {
        self.prefetch_with(gpu, layer, experts, drive, CHUNK)
    }

    /// A bulk prefill wave or warm-up, not decode/verify prediction.
    pub fn prefetch_bulk(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize], drive: bool) -> Result<()> {
        self.prefetch_with(gpu, layer, experts, drive, BULK_CHUNK)
    }

    fn prefetch_with(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize], drive: bool, chunk: u64) -> Result<()> {
        self.settled = false;
        let class = self.class_of.get(layer).copied().flatten().ok_or_else(|| format!("layer {layer} has no routed experts"))?;
        for &expert in experts {
            let key = (layer as u32, expert as u32);
            if !self.map.contains_key(&key) && (drive || self.host_map.contains_key(&key)) {
                // A guess never takes a slot a pass holds or waits for: with none left, stop.
                if self.classes[class].free.is_empty() && self.victim(class).is_err() {
                    break;
                }
                let slot = self.fill(gpu, key, false, chunk)?;
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
        if !self.rest.is_empty() {
            return Err("experts released before their downs were waited for (await_rest)".into());
        }
        if self.in_use.is_empty() {
            return Ok(());
        }
        let mark = gpu.mark(Stream::Compute)?;
        for slot in std::mem::take(&mut self.in_use) {
            self.slots[slot].guards[0] = Some(mark);
            self.slots[slot].held = false;
        }
        self.requested.clear();
        self.keep_free(gpu, mark)
    }

    /// Take `region` (VRAM its lender has no use for now) as more slots, free: as many as fit,
    /// shared among the classes as their layers share the store (each next slot the class's
    /// furthest below its share), one after another from the region's start. The lender takes
    /// them back from the start ([`reclaim`](Self::reclaim)). [`wave_limit`](Self::wave_limit) and
    /// [`pass_limit`](Self::pass_limit) count only the store's own slots. Returns the slots made.
    pub fn lend(&mut self, region: Buffer) -> Result<usize> {
        if !self.lent.is_empty() {
            return Err("an expert store takes one lent region".into());
        }
        let shares: Vec<u64> = (0..self.classes.len())
            .map(|class| self.class_of.iter().filter(|other| **other == Some(class)).count() as u64).collect();
        let mut taken = vec![0u64; self.classes.len()];
        let mut at = 0u64;
        while let Some(class) = (0..self.classes.len()).filter(|&class| shares[class] > 0)
            .min_by(|&a, &b| (taken[a] * shares[b]).cmp(&(taken[b] * shares[a])).then(a.cmp(&b))) {
            let (bytes, start) = (self.classes[class].vram_bytes, at.next_multiple_of(256));
            if start + bytes > region.bytes {
                break;
            }
            let slot = self.slots.len();
            self.slots.push(VramSlot {
                key: None, ready: None, ready_first: None, guards: [None; 2], waiting: None, used: 0, speculative: false, held: false,
                class, place: Some(region.slice(start, bytes)), taken_back: false,
            });
            self.classes[class].free.push_back(slot);
            self.lent.push(LentSlot { slot, offset: start, out: false });
            taken[class] += 1;
            at = start + bytes;
        }
        Ok(self.lent.len())
    }

    /// Take back the lent slots whose bytes begin within the region's first `bytes`, for work
    /// the compute stream queues next: their experts are written back to the host tier (as an
    /// eviction would), a fill still waiting for the drive leaves its read to the host tier, and
    /// the compute stream waits for every copy into or out of them. Between passes (no slot held).
    pub fn reclaim(&mut self, gpu: &mut Backend, bytes: u64) -> Result<()> {
        let mut waits = Vec::new();
        let mut taken = Vec::new();
        for index in 0..self.lent.len() {
            let LentSlot { slot, offset, out } = self.lent[index];
            if out || offset >= bytes {
                continue;
            }
            if self.slots[slot].held {
                return Err("a lent expert slot is taken back while a pass holds it".into());
            }
            if let Some(host) = self.slots[slot].waiting.take() {
                // the read lands in the host tier, for no slot (after the chunks a streamed fill
                // already copied into it)
                self.host[host].waiter = None;
                if std::mem::take(&mut self.host[host].streamed) && self.host[host].copied {
                    let copies = gpu.mark(Stream::Upload)?;
                    self.host[host].busy = Some(copies);
                    waits.push(copies);
                }
                self.empty_slot(slot);
            } else {
                self.evict(gpu, slot)?;
            }
            let entry = &mut self.slots[slot];
            waits.extend(entry.ready.take().map(|(_, mark)| mark));
            waits.extend(std::mem::take(&mut entry.guards).into_iter().flatten());
            self.classes[entry.class].free.retain(|&free| free != slot);
            entry.taken_back = true;
            self.lent[index].out = true;
            taken.push(slot);
        }
        for list in self.speculative.values_mut() {
            list.retain(|slot| !taken.contains(slot));
        }
        for mark in waits {
            gpu.wait_mark(Stream::Compute, mark)?;
        }
        Ok(())
    }

    /// Give the lent slots taken back ([`reclaim`](Self::reclaim)) to the store again, free: a
    /// fill's upload into one waits for the compute stream to pass `after` (the lender's last use
    /// of its bytes).
    pub fn restore(&mut self, after: Mark) {
        for lent in self.lent.iter_mut().filter(|lent| lent.out) {
            lent.out = false;
            let entry = &mut self.slots[lent.slot];
            entry.taken_back = false;
            entry.guards[0] = Some(after);
            self.classes[entry.class].free.push_back(lent.slot);
        }
    }

    /// The lent slots the store has (not taken back), and all of them.
    pub fn lent_slots(&self) -> (usize, usize) {
        (self.lent.iter().filter(|lent| !lent.out).count(), self.lent.len())
    }

    fn slot_buffer(&self, slot: usize) -> Buffer {
        let entry = &self.slots[slot];
        if let Some(place) = entry.place {
            return place;
        }
        let class = &self.classes[entry.class];
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
        self.vram_changes += 1;
        entry.speculative = false;
        let class = &mut self.classes[entry.class];
        class.lru.remove(&(entry.used, slot));
        class.free.push_back(slot);
        self.map.remove(&key);
        Some(key)
    }

    /// A VRAM slot for `key`, its fill queued (uploaded now, or once its drive reads land).
    fn fill(&mut self, gpu: &mut Backend, key: Key, urgent: bool, chunk: u64) -> Result<usize> {
        let class = self.class_of.get(key.0 as usize).copied().flatten().ok_or_else(|| format!("layer {} has no routed experts", key.0))?;
        let slot = self.free_slot(gpu, class)?;
        self.vram_changes += 1;
        self.map.insert(key, slot);
        self.slots[slot].key = Some(key);
        self.classes[class].lru.insert((self.slots[slot].used, slot));
        self.touch(slot);
        let stream = if urgent { Stream::Upload } else { Stream::LaterUpload };
        let filled = match self.host_map.get(&key).copied() {
            Some(host) if !matches!(self.host[host].content, Content::Reading(_)) => {
                self.counts.host_hits += 1;
                let uploaded = if self.heads_first && self.deferring {
                    self.upload_head(gpu, slot, host, stream).map(|tail| self.tails.push(tail))
                } else {
                    self.upload(gpu, slot, host, stream)
                };
                uploaded.inspect_err(|_| self.discard(host, Some(slot)))
            }
            Some(host) => {
                self.slots[slot].waiting = Some(host);
                self.host[host].waiter = Some(slot);
                Ok(())
            }
            None => {
                self.counts.drive_reads += 1;
                // A pass's fill uploads each chunk as it lands (prefetches, and a layer whose slots
                // keep some columns, upload once every chunk is in).
                let streamed = urgent && !self.layout.columns.keys().any(|&(layer, _)| layer == key.0 as usize);
                let started = if streamed { self.start_stream(slot) } else { Ok(()) };
                started.and_then(|()| self.read(gpu, key, urgent, streamed, chunk)).map(|host| {
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
        let tail = self.upload_head(gpu, slot, host, stream)?;
        self.upload_tail(gpu, tail)
    }

    /// The first part of [`upload`](Self::upload): the gate and up, then the slot's `ready_first`.
    fn upload_head(&mut self, gpu: &mut Backend, slot: usize, host: usize, stream: Stream) -> Result<Tail> {
        let (layer, expert) = self.slots[slot].key.ok_or("an upload into an empty slot")?;
        if self.slots[slot].taken_back {
            return Err("an upload into a lent expert slot its lender has back".into());
        }
        if let Some(error) = &self.host[host].failed {
            return Err(error.clone());
        }
        // The slot's previous expert: its last compute use and any download from it come first.
        for guard in std::mem::take(&mut self.slots[slot].guards).into_iter().flatten() {
            gpu.wait_mark(stream, guard)?;
        }
        let base = self.host_pointer(host);
        let target = self.slot_buffer(slot);
        let mut at = 0;
        match self.host[host].content {
            Content::Read => {
                for span in self.layout.read_plan(layer as usize, expert as usize)?.into_iter().take(2) {
                    // SAFETY: the host slot holds this expert as read, and gives it up in the tail
                    // (only a resident store keeps columns: no CPU reads its host slots).
                    let bytes = unsafe { span.keep_columns(base) };
                    // SAFETY: the host slot holds this expert until the tail's mark completes (busy).
                    unsafe { gpu.transfer(stream, &target, at, base.add(span.data as usize), bytes)? };
                    at += bytes;
                }
            }
            Content::Packed(written) => {
                gpu.wait_mark(stream, written)?;
                at = self.layout.roles(layer as usize)?.iter().take(2).map(|role| role.bytes).sum::<u64>();
                // SAFETY: as above.
                unsafe { gpu.transfer(stream, &target, 0, base, at)? };
            }
            Content::Reading(_) => return Err("an upload of an expert still being read".into()),
        }
        self.slots[slot].ready_first = Some((stream, gpu.mark(stream)?));
        // Out of the host tier's LRU until the tail: no read or write-back takes the slot meanwhile.
        let entry = &self.host[host];
        self.classes[entry.class].host_lru.remove(&(entry.used, host));
        Ok(Tail { slot, host, stream, at })
    }

    /// The rest of [`upload`](Self::upload): the down, then the slot's `ready`; the host tier
    /// gives the expert up.
    fn upload_tail(&mut self, gpu: &mut Backend, tail: Tail) -> Result<()> {
        let Tail { slot, host, stream, mut at } = tail;
        let (layer, expert) = self.slots[slot].key.ok_or("an upload into an empty slot")?;
        let base = self.host_pointer(host);
        let target = self.slot_buffer(slot);
        match self.host[host].content {
            Content::Read => {
                for span in self.layout.read_plan(layer as usize, expert as usize)?.into_iter().skip(2) {
                    // SAFETY: as in upload_head.
                    let bytes = unsafe { span.keep_columns(base) };
                    // SAFETY: the host slot holds this expert until the mark below completes (busy).
                    unsafe { gpu.transfer(stream, &target, at, base.add(span.data as usize), bytes)? };
                    at += bytes;
                }
            }
            Content::Packed(_) => {
                let bytes = self.layout.roles(layer as usize)?.iter().map(|role| role.bytes).sum::<u64>();
                if bytes > at {
                    // SAFETY: as above.
                    unsafe { gpu.transfer(stream, &target, at, base.add(at as usize), bytes - at)? };
                }
            }
            Content::Reading(_) => return Err("an upload of an expert still being read".into()),
        }
        let mark = gpu.mark(stream)?;
        self.uploaded(slot, host, stream, mark);
        Ok(())
    }

    /// Upload the downs [`request`](Self::request) left for after every gate and up; each one
    /// that fails empties its slot, and the first failure returns once all are queued.
    fn upload_tails(&mut self, gpu: &mut Backend) -> Result<()> {
        let mut first_error = None;
        for tail in std::mem::take(&mut self.tails) {
            let (slot, host) = (tail.slot, tail.host);
            if let Err(error) = self.upload_tail(gpu, tail) {
                self.discard(host, Some(slot));
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// VRAM slot `slot` holds host slot `host`'s expert once `stream` passes `mark` (its gate and
    /// up at its `ready_first`); the host tier gives the expert up.
    fn uploaded(&mut self, slot: usize, host: usize, stream: Stream, mark: Mark) {
        self.slots[slot].ready = Some((stream, mark));
        self.vram_changes += 1;
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
    }

    /// A streamed fill of VRAM slot `slot`: no upload of it is known yet.
    fn start_stream(&mut self, slot: usize) -> Result<()> {
        let entry = &mut self.slots[slot];
        if entry.taken_back {
            return Err("an upload into a lent expert slot its lender has back".into());
        }
        entry.ready = None;
        entry.ready_first = None;
        Ok(())
    }

    /// A landed chunk of a streamed fill's read: its role bytes go to the waiting VRAM slot on the
    /// upload stream (the first after the slot's previous expert: its last compute use, a download
    /// from it), and once the gate's and up's chunks are all in, the slot's `ready_first`.
    /// Nothing after a chunk of the read failed (the fill is then dropped: see [`pump`](Self::pump)).
    fn stream_chunk(&mut self, gpu: &mut Backend, host: usize, chunk: Chunk) -> Result<()> {
        let entry = &self.host[host];
        let Some(slot) = entry.waiter.filter(|_| entry.streamed && entry.failed.is_none() && !entry.skipped) else { return Ok(()) };
        if self.slots[slot].taken_back {
            return Err("an upload into a lent expert slot its lender has back".into());
        }
        if !self.host[host].copied {
            self.host[host].copied = true;
            for guard in std::mem::take(&mut self.slots[slot].guards).into_iter().flatten() {
                gpu.wait_mark(Stream::Upload, guard)?;
            }
        }
        if chunk.bytes > 0 {
            let base = self.host_pointer(host);
            let target = self.slot_buffer(slot);
            // SAFETY: the chunk's bytes lie inside the host slot, read and checked; the host slot
            // keeps them until the fill's last mark completes (busy).
            unsafe { gpu.transfer(Stream::Upload, &target, chunk.vram, base.add(chunk.host as usize), chunk.bytes)? };
        }
        if chunk.head {
            self.host[host].head -= 1;
            if self.host[host].head == 0 {
                self.slots[slot].ready_first = Some((Stream::Upload, gpu.mark(Stream::Upload)?));
            }
        }
        Ok(())
    }

    /// Start reading `key` from the drive into a host slot (a staging slot when the host tier
    /// holds nothing). With `streamed`, the VRAM slot the caller then makes its waiter uploads
    /// each chunk's role bytes as the chunk lands (roles back to back: a layer without kept
    /// columns), so that only the last chunk's copy follows the last read.
    fn read(&mut self, gpu: &mut Backend, key: Key, urgent: bool, streamed: bool, chunk_bytes: u64) -> Result<usize> {
        let class = self.class_of[key.0 as usize].ok_or("a read for a layer without routed experts")?;
        let host = self.free_host(gpu, class)?;
        let plan = self.layout.read_plan(key.0 as usize, key.1 as usize)?;
        self.counts.drive_bytes += plan.iter().map(|span| span.len).sum::<u64>();
        let base = self.host_pointer(host);
        let cancel = Arc::new(AtomicBool::new(false));
        let mut jobs = Vec::new();
        let (mut vram, mut head) = (0u64, 0u32);
        for (role, span) in plan.into_iter().enumerate() {
            let handle = self.layout.files[span.file].0.clone();
            let mut done = 0;
            while done < span.len {
                let piece = chunk_bytes.min(span.len - done);
                // the role's bytes in this piece of the span, and where they go in the VRAM slot
                let (from, to) = ((span.at + done).max(span.data), (span.at + done + piece).min(span.data + span.bytes));
                let chunk = Chunk { host: from, vram: vram + from.saturating_sub(span.data), bytes: to.saturating_sub(from), head: role < 2 };
                head += u32::from(chunk.head);
                jobs.push(ReadJob {
                    file: handle.clone(), offset: span.offset + done, len: piece,
                    // SAFETY: inside the host slot (its class's host_bytes covers every role's whole blocks).
                    target: HostPointer(unsafe { base.add((span.at + done) as usize) }),
                    verifier: self.layout.verifier.clone().map(|verify| (verify, span.file)),
                    done: self.done_tx.clone(), slot: host, chunk, cancelled: cancel.clone(),
                });
                done += piece;
            }
            vram += span.bytes;
        }
        let entry = &mut self.host[host];
        entry.key = Some(key);
        entry.content = Content::Reading(jobs.len() as u32);
        entry.cancel = cancel;
        entry.skipped = false;
        entry.streamed = streamed;
        entry.head = head;
        entry.copied = false;
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
        for (host, chunk, result) in landed {
            let entry = &mut self.host[host];
            let read = matches!(result, Landed::Read);
            match result {
                Landed::Read => {}
                Landed::Skipped => entry.skipped = true,
                Landed::Failed(error) => {
                    entry.failed.get_or_insert(error);
                }
            }
            let Content::Reading(left) = entry.content else { continue };
            if read {
                if let Err(error) = self.stream_chunk(gpu, host, chunk) {
                    self.host[host].failed.get_or_insert(error);
                }
            }
            let entry = &mut self.host[host];
            entry.content = if left > 1 { Content::Reading(left - 1) } else { Content::Read };
            if left > 1 {
                continue;
            }
            self.reading -= 1;
            let waiting = entry.waiter;
            let streamed = std::mem::take(&mut entry.streamed);
            if streamed && entry.copied && (entry.skipped || entry.failed.is_some()) {
                if let Some(slot) = waiting {
                    // Chunks already copied into the slot (after its guards), and ops queued once its
                    // gate and up were in: the slot's next fill and the host slot's next read wait
                    // for them.
                    match (gpu.mark(Stream::Upload), gpu.mark(Stream::Compute)) {
                        (Ok(copies), Ok(uses)) => {
                            self.slots[slot].guards = [Some(uses), Some(copies)];
                            self.host[host].busy = Some(copies);
                        }
                        (Err(error), _) | (_, Err(error)) => {
                            first_error.get_or_insert(error);
                        }
                    }
                }
            }
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
                Some(slot) if streamed => {
                    // every chunk is in: the fill completes with the last one's copy
                    match gpu.mark(Stream::Upload) {
                        Ok(mark) => {
                            if self.slots[slot].ready_first.is_none() {
                                self.slots[slot].ready_first = Some((Stream::Upload, mark));
                            }
                            self.uploaded(slot, host, Stream::Upload, mark);
                        }
                        Err(error) => {
                            self.discard(host, Some(slot));
                            first_error.get_or_insert(error);
                        }
                    }
                }
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
        let slot = self.classes[class].free.pop_front().ok_or("no free expert slot after an eviction")?;
        if self.slots[slot].taken_back {
            return Err("a lent expert slot its lender has back came free".into());
        }
        Ok(slot)
    }

    fn victim(&self, class: usize) -> Result<usize> {
        self.classes[class].lru.iter().map(|&(_, slot)| slot)
            .find(|&slot| self.slots[slot].waiting.is_none() && !self.slots[slot].held)
            .ok_or_else(|| "every expert slot of a size is in use by this pass".into())
    }

    /// Keep each class's `reserve` VRAM slots free, writing their experts back to the host tier,
    /// so a miss never waits for a write-back before its upload. The write-backs start once the
    /// compute stream passes `after` (the release's work): queued at once, they ran beside that
    /// work's uploads, sharing the bus (an expert's upload on two R9700s took 250 us instead of
    /// 200) and holding the compute stream's start after an upload back by ~130 us. A fill of a
    /// slot so freed still waits for its write-back (the slot's guard).
    fn keep_free(&mut self, gpu: &mut Backend, after: Mark) -> Result<()> {
        let mut waited = false;
        for class in 0..self.classes.len() {
            while self.classes[class].free.len() < self.classes[class].reserve {
                let victim = self.victim(class)?;
                if !waited {
                    gpu.wait_mark(Stream::Download, after)?;
                    waited = true;
                }
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
            skipped: false, waiter: None, streamed: false, head: 0, copied: false, class, arena, at,
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
        self.warm_with(gpu, vram, host, CHUNK)
    }

    /// Warm-up from disk in bulk; demand reads retain priority.
    pub fn warm_bulk(&mut self, gpu: &mut Backend, vram: &[(usize, usize)], host: &[(usize, usize)]) -> Result<usize> {
        self.warm_with(gpu, vram, host, BULK_CHUNK)
    }

    fn warm_with(&mut self, gpu: &mut Backend, vram: &[(usize, usize)], host: &[(usize, usize)], chunk: u64) -> Result<usize> {
        self.settled = false;
        for &(layer, expert) in host {
            let Some(class) = self.class_of.get(layer).copied().flatten() else { continue };
            let key = (layer as u32, expert as u32);
            if self.classes[class].host_capacity == 0 || self.map.contains_key(&key) || self.host_map.contains_key(&key) {
                continue;
            }
            self.read(gpu, key, false, false, chunk)?;
        }
        for &(layer, expert) in vram {
            if self.class_of.get(layer).copied().flatten().is_some() {
                self.prefetch_with(gpu, layer, &[expert], true, chunk)?;
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

/// Where a decode step's route finds an expert the device does not run (see
/// [`DeviceExperts::host_source`]).
#[derive(Debug, Clone, Copy)]
pub enum HostSource {
    /// Each role's bytes in the host tier (gate, up, down), which stay put until the store next
    /// fills, reads or evicts.
    Ready([*const u8; 3]),
    /// Its drive read into the host tier is in flight ([`DeviceExperts::wait_reads`] lands it).
    Reading,
    /// A VRAM slot holds it (or its fill is in flight): [`DeviceExperts::copy_out`] copies it out.
    Vram,
    /// Neither tier holds it: [`DeviceExperts::fetch_host`] reads it.
    Absent,
}

/// For a decode step that takes VRAM experts by a device table of slot addresses and runs the
/// others on the host CPU from the host tier: the table's view of the store, and moves between the
/// tiers that keep it right.
impl DeviceExperts {
    /// Changes so far to which experts the VRAM slots hold and which of them are ready: a table
    /// built from [`vram_ready`](Self::vram_ready) stays right while this stays the same, apart from
    /// the caller's own [`demote`](Self::demote) and [`promote`](Self::promote).
    pub fn vram_generation(&self) -> u64 {
        self.vram_changes
    }

    /// The bytes of `layer`'s VRAM slots (its size class's).
    pub fn slot_size(&self, layer: usize) -> Option<u64> {
        self.class_of.get(layer).copied().flatten().map(|class| self.classes[class].vram_bytes)
    }

    /// Free VRAM slots of `layer`'s size class.
    pub fn free_slots(&self, layer: usize) -> usize {
        self.class_of.get(layer).copied().flatten().map_or(0, |class| self.classes[class].free.len())
    }

    /// Whether `layer` and `other` take slots of one size class.
    pub fn same_class(&self, layer: usize, other: usize) -> bool {
        let class = |layer: usize| self.class_of.get(layer).copied().flatten();
        class(layer).is_some() && class(layer) == class(other)
    }

    /// The experts VRAM slots hold (or are being filled with), as (layer, expert).
    pub fn vram_keys(&self) -> Vec<(usize, usize)> {
        self.map.keys().map(|&(layer, expert)| (layer as usize, expert as usize)).collect()
    }

    /// The slot address of `layer`'s `expert` when a VRAM slot holds it and its upload has
    /// completed (an op may read it without waiting), else None.
    pub fn vram_ready(&mut self, gpu: &mut Backend, layer: usize, expert: usize) -> Result<Option<u64>> {
        let Some(&slot) = self.map.get(&(layer as u32, expert as u32)) else { return Ok(None) };
        if self.slots[slot].waiting.is_some() {
            return Ok(None);
        }
        if let Some((_, mark)) = self.slots[slot].ready {
            if !gpu.mark_done(mark, false)? {
                return Ok(None);
            }
            self.slots[slot].ready = None;
        }
        Ok(Some(self.slot_buffer(slot).base() as u64))
    }

    /// Mark `layer`'s `expert` (in VRAM) just used, as an acquire would: passes that take slots
    /// evict the least recently used.
    pub fn touch_vram(&mut self, layer: usize, expert: usize) {
        if let Some(&slot) = self.map.get(&(layer as u32, expert as u32)) {
            self.touch(slot);
        }
    }

    /// Where the host finds `layer`'s `expert` (touched as used when the host tier holds it).
    pub fn host_source(&mut self, gpu: &mut Backend, layer: usize, expert: usize) -> Result<HostSource> {
        let key = (layer as u32, expert as u32);
        if self.map.contains_key(&key) {
            return Ok(HostSource::Vram);
        }
        match self.host_map.get(&key) {
            Some(&host) if matches!(self.host[host].content, Content::Reading(_)) => Ok(HostSource::Reading),
            Some(_) => Ok(self.host_roles(gpu, layer, expert)?.map_or(HostSource::Absent, HostSource::Ready)),
            None => Ok(HostSource::Absent),
        }
    }

    /// Start reading `layer`'s `expert` from the drive into the host tier (no VRAM slot): as
    /// urgent as a pass's fills, or (a guess) behind them; once [`wait_reads`](Self::wait_reads)
    /// lands it, the host tier holds it.
    pub fn fetch_host(&mut self, gpu: &mut Backend, layer: usize, expert: usize, urgent: bool) -> Result<()> {
        let key = (layer as u32, expert as u32);
        if self.map.contains_key(&key) || self.host_map.contains_key(&key) {
            return Ok(());
        }
        self.settled = false;
        self.counts.drive_reads += 1;
        if !urgent {
            self.counts.prefetched += 1;
        }
        self.read(gpu, key, urgent, false, CHUNK).map(drop)
    }

    /// Wait for a drive read to land (and take every landed one). A read no pass holds a slot for
    /// that failed drops its expert (counted): its [`host_source`](Self::host_source) is then
    /// Absent again.
    pub fn wait_reads(&mut self, gpu: &mut Backend) -> Result<()> {
        self.pump(gpu, true)
    }

    /// Evict `layer`'s `expert` from its VRAM slot to the host tier (written back when the tier has
    /// room), the slot reused only once the compute stream passes `after` (a point after every op
    /// that may read it). False when no VRAM slot holds it ready (none, or its fill in flight).
    pub fn demote(&mut self, gpu: &mut Backend, layer: usize, expert: usize, after: Mark) -> Result<bool> {
        let Some(&slot) = self.map.get(&(layer as u32, expert as u32)) else { return Ok(false) };
        if self.slots[slot].waiting.is_some() || self.slots[slot].held {
            return Ok(false);
        }
        self.settled = false;
        self.slots[slot].guards[0] = Some(after);
        self.evict(gpu, slot)?;
        Ok(true)
    }

    /// Bring `layer`'s `expert` from the host tier into a free VRAM slot of its size class (none is
    /// evicted for it), uploaded behind the copies passes wait for; the host tier gives it up. Its
    /// slot's address and the upload's mark (an op may read the slot once the compute stream waits
    /// for it), or None when the host tier holds no ready copy or the class no free slot.
    pub fn promote(&mut self, gpu: &mut Backend, layer: usize, expert: usize) -> Result<Option<(u64, Mark)>> {
        let Some(class) = self.class_of.get(layer).copied().flatten() else { return Ok(None) };
        if !self.in_host(layer, expert) || self.classes[class].free.is_empty() {
            return Ok(None);
        }
        self.settled = false;
        let slot = self.fill(gpu, (layer as u32, expert as u32), false, CHUNK)?;
        let (_, mark) = self.slots[slot].ready.ok_or("a promoted expert has no upload")?;
        Ok(Some((self.slot_buffer(slot).base() as u64, mark)))
    }

    /// Copy `layer`'s `expert` out of its VRAM slot to `target` (its roles back to back), waiting
    /// for its fill and the copy: for a host that needs an expert only VRAM holds. False when no
    /// VRAM slot holds it (a fill's failed read drops it).
    ///
    /// # Safety
    /// `target` is page-locked (a [`transfer::pin_host`] range) and holds the layer's slot bytes.
    pub unsafe fn copy_out(&mut self, gpu: &mut Backend, layer: usize, expert: usize, target: *mut u8) -> Result<bool> {
        let key = (layer as u32, expert as u32);
        while self.map.get(&key).is_some_and(|&slot| self.slots[slot].waiting.is_some()) {
            self.pump(gpu, true)?;
        }
        let Some(&slot) = self.map.get(&key) else { return Ok(false) };
        if let Some((_, ready)) = self.slots[slot].ready {
            gpu.wait_mark(Stream::Download, ready)?;
        }
        let bytes: u64 = self.layout.roles(layer)?.iter().map(|role| role.bytes).sum();
        let source = self.slot_buffer(slot);
        // SAFETY: the caller's page-locked target of the slot's bytes; waited for below.
        unsafe { gpu.transfer(Stream::Download, &source, 0, target, bytes)? };
        let mark = gpu.mark(Stream::Download)?;
        self.slots[slot].guards[1] = Some(mark);
        gpu.mark_done(mark, true)?;
        Ok(true)
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

#[cfg(test)]
mod reader_tests {
    use super::*;
    use std::io::Write;
    use std::time::Duration;

    fn file() -> (std::path::PathBuf, Arc<File>) {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!("straydog-reader-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        File::create(&path).unwrap().write_all(&vec![7u8; ALIGN]).unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        (path, file)
    }

    fn job(file: &Arc<File>, target: &mut [u8], done: &Sender<(usize, Chunk, Landed)>, slot: usize,
           head: bool, verifier: Option<Verifier>) -> ReadJob {
        ReadJob { file: file.clone(), offset: 0, len: target.len() as u64, target: HostPointer(target.as_mut_ptr()),
            verifier: verifier.map(|verify| (verify, 0)), done: done.clone(), slot,
            chunk: Chunk { host: 0, vram: 0, bytes: target.len() as u64, head }, cancelled: Arc::new(AtomicBool::new(false)) }
    }

    struct Unblock(Arc<(Mutex<bool>, Condvar)>);
    impl Drop for Unblock {
        fn drop(&mut self) { *self.0.0.lock().unwrap() = true; self.0.1.notify_all(); }
    }

    #[test]
    fn admission_returns_before_verification() {
        let (path, file) = file();
        let mut targets = [vec![0u8; ALIGN], vec![0u8; ALIGN]];
        let reader = Reader::bounded(2, 1, true);
        let blocked = Arc::new((Mutex::new(false), Condvar::new()));
        let unblock = Unblock(blocked.clone());
        let (entered, entries) = channel();
        let verifier: Verifier = Arc::new(move |_, _, data| {
            assert!(data.iter().all(|&byte| byte == 7));
            entered.send(()).unwrap();
            let mut open = blocked.0.lock().unwrap();
            while !*open { open = blocked.1.wait(open).unwrap(); }
            Ok(())
        });
        let (done, results) = channel();
        let jobs = targets.iter_mut().enumerate().map(|(i, target)| job(&file, target, &done, i, true, Some(verifier.clone()))).collect();
        reader.submit(jobs, true);
        // Both workers must reach verification even with just one data-read admission.
        let first = entries.recv_timeout(Duration::from_secs(3));
        let second = entries.recv_timeout(Duration::from_secs(3));
        drop(unblock);
        let completed: Vec<_> = (0..2).map(|_| results.recv_timeout(Duration::from_secs(3))).collect();
        drop(reader);
        assert!(first.is_ok() && second.is_ok(), "verification held the only data-read permit");
        assert!(completed.iter().all(|r| matches!(r, Ok((_, _, Landed::Read)))));
        assert!(targets.iter().flatten().all(|&byte| byte == 7));
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn unadmitted_background_does_not_bypass_urgent_or_starve_tails() {
        let (path, file) = file();
        let mut targets = [vec![0u8; ALIGN], vec![0u8; ALIGN], vec![0u8; ALIGN], vec![0u8; ALIGN]];
        let reader = Reader::bounded(1, 1, true);
        // Represent a data call already occupying the admission. Workers must not dequeue yet.
        reader.queue.0.lock().unwrap().reading = 1;
        let held = ReadPermit(reader.queue.clone());
        let (done, results) = channel();
        let [background, tail, head, another_head] = &mut targets;
        reader.submit(vec![job(&file, background, &done, 0, true, None)], false);
        reader.submit(vec![job(&file, tail, &done, 1, false, None), job(&file, head, &done, 2, true, None),
                           job(&file, another_head, &done, 3, true, None)], true);
        let premature = results.recv_timeout(Duration::from_millis(50));
        drop(held);
        let order: Vec<_> = (0..4).map(|_| results.recv_timeout(Duration::from_secs(3))).collect();
        drop(reader);
        assert!(premature.is_err(), "a data read bypassed the admission bound");
        let slots: Vec<_> = order.into_iter().map(|r| r.unwrap().0).collect();
        assert_eq!(slots, [2, 1, 3, 0], "urgent head first, tail progress, then background");
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn failed_cancelled_and_panicked_reads_return_their_admission() {
        let (path, file) = file();
        let mut targets = [vec![0u8; ALIGN], vec![0u8; ALIGN], vec![0u8; ALIGN], vec![0u8; ALIGN]];
        let reader = Reader::bounded(1, 1, true);
        let (done, results) = channel();
        let mut jobs = Vec::new();
        let verifiers: [Option<Verifier>; 4] = [None, Some(Arc::new(|_, _, _| Err("bad proof".into()))),
            Some(Arc::new(|_, _, _| panic!("expected verifier panic"))), None];
        for (i, (target, verifier)) in targets.iter_mut().zip(verifiers).enumerate() {
            let read = job(&file, target, &done, i, true, verifier);
            if i == 0 { read.cancelled.store(true, Ordering::Relaxed); }
            jobs.push(read);
        }
        reader.submit(jobs, true);
        let completed: Vec<_> = (0..4).map(|_| results.recv_timeout(Duration::from_secs(3))).collect();
        drop(reader);
        let completed: Vec<_> = completed.into_iter().map(|r| r.unwrap().2).collect();
        assert!(matches!(completed[0], Landed::Skipped));
        assert!(matches!(&completed[1], Landed::Failed(message) if message.contains("bad proof")));
        assert!(matches!(&completed[2], Landed::Failed(message) if message.contains("panicked")));
        assert!(matches!(completed[3], Landed::Read));
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
