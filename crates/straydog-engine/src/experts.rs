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

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use crate::gpu::{Backend, Result};
use crate::ops::Buffer;
use crate::storage::{self, ALIGN};
use crate::transfer::{self, Mark, Stream};

/// One drive read: large enough to stream, small enough that one expert's pieces spread over
/// several readers.
const CHUNK: u64 = 1 << 20;
/// Host tier memory is allocated and page-locked in arenas of this size, as it fills.
const ARENA: usize = 1 << 30;

/// A tensor of a layer's experts: expert `e`'s bytes are `stride` bytes at `base + e * stride` of
/// file `file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Role {
    pub file: usize,
    pub base: u64,
    pub stride: u64,
}

/// Checks a drive read of `file` at `offset` (whole aligned blocks) before its first use.
pub type Verifier = Arc<dyn Fn(usize, u64, &[u8]) -> std::result::Result<(), String> + Send + Sync>;

/// The model files and where each layer's experts lie in them.
pub struct Layout {
    /// Per file: a handle, and whether it reads around the page cache (aligned reads only).
    files: Vec<(Arc<File>, bool)>,
    /// Per layer: its roles (gate, up, down...), None for a layer without routed experts.
    layers: Vec<Option<Vec<Role>>>,
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
        }
        Ok(Layout { files, layers, verifier })
    }

    fn roles(&self, layer: usize) -> Result<&[Role]> {
        self.layers.get(layer).and_then(|roles| roles.as_deref()).ok_or_else(|| format!("layer {layer} has no routed experts"))
    }

    /// An expert's bytes in a VRAM slot: its roles one after another.
    pub fn slot_bytes(&self) -> u64 {
        self.layers.iter().flatten().map(|roles| roles.iter().map(|role| role.stride).sum()).max().unwrap_or(0)
    }

    /// An expert's bytes in a host slot: room for either of its layouts there (read: each role's
    /// whole blocks; written back: the VRAM slot's).
    fn host_bytes(&self) -> u64 {
        self.layers.iter().flatten().map(|roles| {
            let read: u64 = roles.iter().map(|role| role.stride.next_multiple_of(ALIGN as u64) + ALIGN as u64).sum();
            read.max(roles.iter().map(|role| role.stride).sum())
        }).max().unwrap_or(0).next_multiple_of(ALIGN as u64)
    }

    /// Where each role's bytes lie in a host slot filled by drive reads.
    fn read_plan(&self, layer: usize, expert: usize) -> Result<Vec<Span>> {
        let mut at = 0u64;
        self.roles(layer)?.iter().map(|role| {
            let start = role.base + expert as u64 * role.stride;
            let offset = start & !(ALIGN as u64 - 1);
            let len = (start + role.stride - offset).next_multiple_of(ALIGN as u64);
            let span = Span { file: role.file, offset, len, at, data: at + (start - offset), bytes: role.stride };
            at += len;
            Ok(span)
        }).collect()
    }
}

/// A role's aligned read: `len` bytes from `offset` of file `file` to `at` in a host slot, where
/// the role's `bytes` start at `data`.
struct Span {
    file: usize,
    offset: u64,
    len: u64,
    at: u64,
    data: u64,
    bytes: u64,
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
    done: Sender<(usize, std::result::Result<(), String>)>,
    slot: usize,
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
                let _ = job.done.send((job.slot, run_read(&job)));
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

type Key = (u32, u32);

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
}

struct VramSlot {
    key: Option<Key>,
    /// The upload mark after this slot's fill, until known complete.
    ready: Option<Mark>,
    /// Marks the slot's bytes must outlive: its last compute use, and a download from it.
    guards: [Option<Mark>; 2],
    /// A fill waiting for the drive: the host slot whose reads it uploads once they land.
    waiting: Option<usize>,
    used: u64,
}

/// Counters since creation.
#[derive(Debug, Default, Clone, Copy)]
pub struct Counts {
    pub hits: u64,
    pub host_hits: u64,
    pub drive_reads: u64,
    pub writebacks: u64,
    pub prefetched: u64,
    /// Seconds the host waited for drive reads (or for a host slot's copy to finish).
    pub wait_seconds: f64,
}

/// One device's expert slots, and its share of the host tier.
pub struct DeviceExperts {
    layout: Arc<Layout>,
    reader: Arc<Reader>,
    vram: Buffer,
    vram_slot_bytes: u64,
    slots: Vec<VramSlot>,
    map: HashMap<Key, usize>,
    host_slot_bytes: u64,
    host_capacity: usize,
    per_arena: usize,
    arenas: Vec<storage::Buffer>,
    host: Vec<HostSlot>,
    host_map: HashMap<Key, usize>,
    done_tx: Sender<(usize, std::result::Result<(), String>)>,
    done_rx: Receiver<(usize, std::result::Result<(), String>)>,
    clock: u64,
    /// Slots acquired since the last release, kept from eviction until then.
    in_use: Vec<usize>,
    /// Free VRAM slots kept ahead of need, their experts written back meanwhile.
    reserve: usize,
    counts: Counts,
    settled: bool,
}

impl DeviceExperts {
    /// `slots` VRAM slots on this device and `host_budget` bytes of host tier.
    pub fn new(gpu: &mut Backend, layout: Arc<Layout>, reader: Arc<Reader>, slots: usize, host_budget: u64)
               -> Result<DeviceExperts> {
        let vram_slot_bytes = layout.slot_bytes().next_multiple_of(256);
        if slots < 16 || vram_slot_bytes == 0 {
            return Err("an expert store needs at least 16 VRAM slots and a layer of experts".into());
        }
        let vram = gpu.buffer(vram_slot_bytes * slots as u64)?;
        let host_slot_bytes = layout.host_bytes();
        let host_capacity = (host_budget / host_slot_bytes) as usize;
        let reserve = (slots / 64).clamp(4, 32);
        // Small tiers (and staging alone) take one arena of just their size.
        let per_arena = ((ARENA as u64 / host_slot_bytes) as usize).clamp(1, host_capacity.max(reserve));
        let (done_tx, done_rx) = channel();
        Ok(DeviceExperts {
            layout, reader, vram, vram_slot_bytes,
            slots: (0..slots).map(|_| VramSlot { key: None, ready: None, guards: [None; 2], waiting: None, used: 0 }).collect(),
            map: HashMap::new(), host_slot_bytes, host_capacity, per_arena, arenas: Vec::new(), host: Vec::new(),
            host_map: HashMap::new(), done_tx, done_rx, clock: 0, in_use: Vec::new(), reserve,
            counts: Counts::default(), settled: false,
        })
    }

    pub fn counts(&self) -> Counts {
        self.counts
    }

    /// VRAM and host slots.
    pub fn capacity(&self) -> (usize, usize) {
        (self.slots.len(), self.host_capacity)
    }

    /// Make these experts of `layer` resident here; the compute stream waits for their copies.
    /// Their slots (each roles one after another) stay until [`release`](Self::release).
    pub fn acquire(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize]) -> Result<Vec<Buffer>> {
        self.settled = false;
        self.pump(gpu, false)?;
        let mut slots = Vec::with_capacity(experts.len());
        for &expert in experts {
            let key = (layer as u32, expert as u32);
            let slot = match self.map.get(&key) {
                Some(&slot) => {
                    self.counts.hits += 1;
                    slot
                }
                None => self.fill(gpu, key, true)?,
            };
            self.clock += 1;
            self.slots[slot].used = self.clock;
            self.in_use.push(slot);
            slots.push(slot);
        }
        // Fills waiting for the drive upload as their reads land.
        while slots.iter().any(|&slot| self.slots[slot].waiting.is_some()) {
            self.pump(gpu, true)?;
        }
        // The upload stream runs in order: its newest mark among these slots covers them all.
        let newest = slots.iter().filter_map(|&slot| self.slots[slot].ready).max();
        if let Some(mark) = newest {
            if gpu.mark_done(mark, false)? {
                for &slot in &slots {
                    self.slots[slot].ready = self.slots[slot].ready.filter(|&ready| ready > mark);
                }
            } else {
                gpu.wait_mark(Stream::Compute, mark)?;
            }
        }
        Ok(slots.iter().map(|&slot| self.vram.slice(slot as u64 * self.vram_slot_bytes, self.vram_slot_bytes)).collect())
    }

    /// Start bringing these experts of `layer` in without waiting: from the host tier at once,
    /// from the drive as reads land. A wrong guess costs a slot and bus time, never a result.
    pub fn prefetch(&mut self, gpu: &mut Backend, layer: usize, experts: &[usize]) -> Result<()> {
        self.settled = false;
        for &expert in experts {
            let key = (layer as u32, expert as u32);
            if !self.map.contains_key(&key) {
                self.fill(gpu, key, false)?;
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
        }
        self.keep_free(gpu)
    }

    /// A VRAM slot for `key`, its fill queued (uploaded now, or once its drive reads land).
    fn fill(&mut self, gpu: &mut Backend, key: Key, urgent: bool) -> Result<usize> {
        let slot = self.free_slot(gpu)?;
        self.map.insert(key, slot);
        self.slots[slot].key = Some(key);
        self.clock += 1;
        self.slots[slot].used = self.clock;
        match self.host_map.get(&key).copied() {
            Some(host) if !matches!(self.host[host].content, Content::Reading(_)) => {
                self.counts.host_hits += 1;
                self.upload(gpu, slot, host)?;
            }
            Some(host) => self.slots[slot].waiting = Some(host),
            None => {
                self.counts.drive_reads += 1;
                let host = self.read(gpu, key, urgent)?;
                self.slots[slot].waiting = Some(host);
            }
        }
        Ok(slot)
    }

    /// Upload host slot `host` into VRAM slot `slot`; the host tier gives the expert up.
    fn upload(&mut self, gpu: &mut Backend, slot: usize, host: usize) -> Result<()> {
        let (layer, expert) = self.slots[slot].key.ok_or("an upload into an empty slot")?;
        if let Some(error) = self.host[host].failed.take() {
            return Err(error);
        }
        // The slot's previous expert: its last compute use and any download from it come first.
        for guard in std::mem::take(&mut self.slots[slot].guards).into_iter().flatten() {
            gpu.wait_mark(Stream::Upload, guard)?;
        }
        let base = self.host_pointer(host);
        let target = self.vram.slice(slot as u64 * self.vram_slot_bytes, self.vram_slot_bytes);
        match self.host[host].content {
            Content::Read => {
                let mut at = 0;
                for span in self.layout.read_plan(layer as usize, expert as usize)? {
                    // SAFETY: the host slot holds this expert until the mark below completes (busy).
                    unsafe { gpu.transfer(Stream::Upload, &target, at, base.add(span.data as usize), span.bytes)? };
                    at += span.bytes;
                }
            }
            Content::Packed(written) => {
                gpu.wait_mark(Stream::Upload, written)?;
                let bytes: u64 = self.layout.roles(layer as usize)?.iter().map(|role| role.stride).sum();
                // SAFETY: as above.
                unsafe { gpu.transfer(Stream::Upload, &target, 0, base, bytes)? };
            }
            Content::Reading(_) => return Err("an upload of an expert still being read".into()),
        }
        let mark = gpu.mark(Stream::Upload)?;
        self.slots[slot].ready = Some(mark);
        self.slots[slot].waiting = None;
        let entry = &mut self.host[host];
        entry.busy = Some(mark);
        if let Some(key) = entry.key.take() {
            self.host_map.remove(&key);
        }
        Ok(())
    }

    /// Start reading `key` from the drive into a host slot (a staging slot when the host tier
    /// holds nothing).
    fn read(&mut self, gpu: &mut Backend, key: Key, urgent: bool) -> Result<usize> {
        let host = self.free_host(gpu)?;
        let plan = self.layout.read_plan(key.0 as usize, key.1 as usize)?;
        let base = self.host_pointer(host);
        let mut jobs = Vec::new();
        for span in plan {
            let handle = self.layout.files[span.file].0.clone();
            let mut done = 0;
            while done < span.len {
                let piece = CHUNK.min(span.len - done);
                jobs.push(ReadJob {
                    file: handle.clone(), offset: span.offset + done, len: piece,
                    // SAFETY: inside the host slot (host_bytes covers every role's whole blocks).
                    target: HostPointer(unsafe { base.add((span.at + done) as usize) }),
                    verifier: self.layout.verifier.clone().map(|verify| (verify, span.file)),
                    done: self.done_tx.clone(), slot: host,
                });
                done += piece;
            }
        }
        let entry = &mut self.host[host];
        entry.key = Some(key);
        entry.content = Content::Reading(jobs.len() as u32);
        self.host_map.insert(key, host);
        self.reader.submit(jobs, urgent);
        Ok(host)
    }

    /// Take finished drive reads; upload the fills waiting for them. With `block`, wait for at
    /// least one read first.
    fn pump(&mut self, gpu: &mut Backend, block: bool) -> Result<()> {
        let mut landed = Vec::new();
        if block {
            let started = std::time::Instant::now();
            let message = self.done_rx.recv().map_err(|_| "the expert reader stopped")?;
            self.counts.wait_seconds += started.elapsed().as_secs_f64();
            landed.push(message);
        }
        landed.extend(self.done_rx.try_iter());
        for (host, result) in landed {
            let entry = &mut self.host[host];
            if let Err(error) = result {
                entry.failed.get_or_insert(error);
            }
            if let Content::Reading(left) = entry.content {
                entry.content = if left > 1 { Content::Reading(left - 1) } else { Content::Read };
                if left == 1 {
                    if let Some(slot) = self.slots.iter().position(|slot| slot.waiting == Some(host)) {
                        self.upload(gpu, slot, host)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// A free VRAM slot: one already free, else the least recently used outside the current pass.
    fn free_slot(&mut self, gpu: &mut Backend) -> Result<usize> {
        if let Some(slot) = self.slots.iter().position(|slot| slot.key.is_none()) {
            return Ok(slot);
        }
        let victim = self.victim()?;
        self.evict(gpu, victim)?;
        Ok(victim)
    }

    fn victim(&self) -> Result<usize> {
        (0..self.slots.len())
            .filter(|&slot| self.slots[slot].waiting.is_none() && !self.in_use.contains(&slot))
            .min_by_key(|&slot| self.slots[slot].used)
            .ok_or_else(|| "every expert slot is in use by this pass".into())
    }

    /// Keep `reserve` VRAM slots free, writing their experts back to the host tier, so a miss
    /// never waits for a write-back before its upload.
    fn keep_free(&mut self, gpu: &mut Backend) -> Result<()> {
        let free = self.slots.iter().filter(|slot| slot.key.is_none()).count();
        for _ in free..self.reserve {
            let victim = self.victim()?;
            self.evict(gpu, victim)?;
        }
        Ok(())
    }

    /// Empty VRAM slot `slot`, writing its expert back to the host tier when that has room.
    fn evict(&mut self, gpu: &mut Backend, slot: usize) -> Result<()> {
        let Some(key) = self.slots[slot].key.take() else { return Ok(()) };
        self.map.remove(&key);
        if self.host_capacity == 0 || self.host_map.contains_key(&key) {
            return Ok(());
        }
        let host = self.free_host(gpu)?;
        let bytes: u64 = self.layout.roles(key.0 as usize)?.iter().map(|role| role.stride).sum();
        if let Some(ready) = self.slots[slot].ready.take() {
            gpu.wait_mark(Stream::Download, ready)?;
        }
        let source = self.vram.slice(slot as u64 * self.vram_slot_bytes, self.vram_slot_bytes);
        // SAFETY: the host slot is reserved for this expert; nothing reads it before the mark.
        unsafe { gpu.transfer(Stream::Download, &source, 0, self.host_pointer(host), bytes)? };
        let mark = gpu.mark(Stream::Download)?;
        self.slots[slot].guards[1] = Some(mark);
        let entry = &mut self.host[host];
        entry.key = Some(key);
        entry.content = Content::Packed(mark);
        self.clock += 1;
        entry.used = self.clock;
        self.host_map.insert(key, host);
        self.counts.writebacks += 1;
        Ok(())
    }

    /// A host slot to fill: a new one while the budget allows, else the least recently used
    /// (after any copy still using it). With no host tier, staging slots grow as reads need.
    fn free_host(&mut self, gpu: &mut Backend) -> Result<usize> {
        let reading = |slot: &HostSlot| matches!(slot.content, Content::Reading(_));
        for host in 0..self.host.len() {
            let entry = &mut self.host[host];
            if entry.key.is_none() && !reading(entry) {
                match entry.busy {
                    None => return Ok(host),
                    Some(mark) if gpu.mark_done(mark, false)? => {
                        entry.busy = None;
                        return Ok(host);
                    }
                    Some(_) => {}
                }
            }
        }
        let waiting: Vec<usize> = self.slots.iter().filter_map(|slot| slot.waiting).collect();
        let candidate = (0..self.host.len())
            .filter(|&host| !reading(&self.host[host]) && !waiting.contains(&host))
            .min_by_key(|&host| (self.host[host].key.is_some(), self.host[host].used));
        if self.host.len() < self.host_capacity.max(self.reserve) || candidate.is_none() {
            return self.grow_host();
        }
        let host = candidate.unwrap();
        if let Some(mark) = self.host[host].busy.take() {
            let started = std::time::Instant::now();
            gpu.mark_done(mark, true)?;
            self.counts.wait_seconds += started.elapsed().as_secs_f64();
        }
        if let Content::Packed(mark) = self.host[host].content {
            gpu.mark_done(mark, true)?;
        }
        if let Some(key) = self.host[host].key.take() {
            self.host_map.remove(&key);
        }
        Ok(host)
    }

    fn grow_host(&mut self) -> Result<usize> {
        if self.host.len() == self.arenas.len() * self.per_arena {
            let mut arena = storage::Buffer::new(self.per_arena * self.host_slot_bytes as usize);
            // SAFETY: the arena lives in self.arenas until drop, which unpins it first.
            unsafe { transfer::pin_host(arena.as_mut_slice().as_mut_ptr(), arena.capacity())? };
            self.arenas.push(arena);
        }
        self.host.push(HostSlot { key: None, content: Content::Read, busy: None, used: 0, failed: None });
        Ok(self.host.len() - 1)
    }

    fn host_pointer(&mut self, host: usize) -> *mut u8 {
        let per_arena = self.per_arena;
        let arena = &mut self.arenas[host / per_arena];
        // SAFETY: slot (host % per_arena) lies inside its arena.
        unsafe { arena.as_mut_slice().as_mut_ptr().add((host % per_arena) * self.host_slot_bytes as usize) }
    }
}

impl DeviceExperts {
    /// Wait for every read and copy in flight; dropping the store frees its host memory only
    /// after this (else it leaks it, which a copy still in flight cannot then corrupt).
    pub fn settle(&mut self, gpu: &mut Backend) -> Result<()> {
        while self.host.iter().any(|slot| matches!(slot.content, Content::Reading(_))) {
            self.pump(gpu, true)?;
        }
        for stream in [Stream::Upload, Stream::Download] {
            let mark = gpu.mark(stream)?;
            gpu.mark_done(mark, true)?;
        }
        self.settled = true;
        Ok(())
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
        if !self.settled {
            for arena in std::mem::take(&mut self.arenas) {
                std::mem::forget(arena);
            }
            return;
        }
        for arena in &mut self.arenas {
            // SAFETY: pinned in grow_host; settle saw every copy from it finish.
            let _ = unsafe { transfer::unpin_host(arena.as_mut_slice().as_mut_ptr()) };
        }
    }
}
