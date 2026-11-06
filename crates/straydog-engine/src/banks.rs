//! The page-locking of a partially resident model's expert banks, which the worker does behind "ready".
//!
//! A bank is a tensor of the mapped model file, so locking it reads it from the drive: 25.9 GiB took 7.5 s of an 18 s start on an RTX 4060 PC.
//! [`Model::new`](crate::model::Model::new) decides what is to be locked and does not lock it ([`Model::bank_locks`](crate::model::Model::bank_locks)
//! hands the plan out as a [`BankLocks`]); a thread of the worker then takes the layers one at a time, off the model's lock, and a request never waits for it.
//!
//! One layer's turn, in the order that keeps the verified-bytes rule (docs/DESIGN.md section 2: only bytes checked against the file's Merkle tree
//! reach the GPU or the CPU's experts):
//!
//! 1. [`BankLocks::lock_next`], on the thread and holding nothing of the model's: page-lock the layer's three banks, then check them against
//!    the file's tree. The check comes after the lock because a locked page cannot change under it, where the pages of an unlocked bank may be
//!    read from the drive again after the load verified them. A layer that cannot be locked, or whose bytes do not match, is unlocked at once
//!    and the thread stops there.
//! 2. [`Model::banks_locked`](crate::model::Model::banks_locked), under the model's lock: only now are the layer's banks registered with the
//!    device (released when it ends) and mapped for the GPU's assist, the one way the GPU reads a bank in place (`sd_hybrid_assist`).
//!
//! Until step 2 the layer is what every layer that is not page-locked is: the CPU's experts read its pages from the mapping and the slots'
//! copies read them as they are. No path that takes a bank's bytes to the GPU, the assist apart, depends on the lock.
//!
//! The locks are an optimisation and the K/V of a conversation is not, so the locks give way to it, in two ways. The locking leaves room: before a
//! layer's turn it looks at the memory available then, and stops when the layer would leave less than the plan's margin and the K/V the context's
//! first 64K tokens still have to grow ([`locking_room`]). And when a K/V growth finds no memory anyway,
//! [`Model::give_back`](crate::model::Model::give_back) takes locked layers back, newest first ([`give_back_count`]), and ends the locking
//! ([`BankLocks`] stops when the model says so, and a layer it locked meanwhile is let go).
//!
//! The pages of the layers not yet locked are in memory only if the page cache holds them, and a request that comes before the locking is done
//! reads the rest by faulting, one fault after another (Linux, a cold page cache: 28 to 31 tok/s with 218,547 page faults against 49.4 tok/s after
//! the locking). [`BankLocks::read_ahead`] hands out the plan's banks as [`Piece`]s in the order the layers are locked, for a thread of the worker to ask the
//! system for at a low priority ([`Piece::ask`], [`Piece::resident`]), a few hundred MB at a time, so that the locking and a request find them in memory.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::host::{Lowered, Priority};
use crate::verity::{self, Verifier};

/// One bank (a layer's gate, up or down experts) as a thread that does not hold the weights sees it: its file's mapping, its place in
/// that, and the file's verity tree if it was opened.
pub(crate) struct Bank {
    map: *const u8,
    map_len: usize,
    offset: u64,
    bytes: u64,
    verifier: Option<Arc<Verifier>>,
}

impl Bank {
    /// A bank at `offset` of `bytes` in the mapping `map` of `map_len` bytes. The mapping must outlive every use of the bank
    /// ([`BankLocks::lock_next`]); the range lies within it ([`Weights::mapped_bank`](crate::weights::Weights::mapped_bank) checks).
    pub(crate) fn new(map: *const u8, map_len: usize, offset: u64, bytes: u64, verifier: Option<Arc<Verifier>>) -> Bank {
        Bank { map, map_len, offset, bytes, verifier }
    }

    /// The bank's bytes.
    ///
    /// # Safety
    /// The mapping must be alive.
    unsafe fn range(&self) -> &[u8] {
        // SAFETY: the caller's; the range was checked to lie in the mapping.
        unsafe { std::slice::from_raw_parts(self.map.add(self.offset as usize), self.bytes as usize) }
    }

    /// Check the bank's bytes against the file's tree, block by block on the threads `priority` allows ([`check_threads`]), which work at that
    /// priority: `Ok(false)` when the file has no tree.
    ///
    /// # Safety
    /// The mapping must be alive, and the bank page-locked (its bytes cannot change under the check).
    unsafe fn verify(&self, priority: Priority) -> Result<bool, String> {
        let Some(verifier) = &self.verifier else { return Ok(false) };
        let threads = check_threads(priority, std::thread::available_parallelism().map_or(8, |count| count.get()));
        // SAFETY: the caller's.
        let map = unsafe { std::slice::from_raw_parts(self.map, self.map_len) };
        verity::verify_mapped_at(map, verifier, self.offset, self.bytes, threads, priority).map(|()| true)
    }
}

/// The threads that check a layer's banks on a machine of `cores`: every core for a locking that nothing waits behind, and at most a quarter of them
/// (one at least) for the locking behind "ready", which leaves the rest to the request's CPU experts and the memory bandwidth they use.
pub(crate) fn check_threads(priority: Priority, cores: usize) -> usize {
    match priority {
        Priority::Normal => cores.max(1),
        Priority::Background => (cores / 4).max(1),
    }
}

/// What a layer's turn works at: [`Lowered::begin_reading`], the CPU and I/O priority of `priority` and, on Windows, the thread's normal memory priority. The turn
/// asks the system for the next layer's pages while it locks this one, and a page that is read in waits in the standby list at the memory priority of the thread that
/// read it: at background mode's 1 it is the first the system takes back for the next page, the next layer's own among them, and the turn after found it gone and read
/// it from the drive again.
fn turn_lowered(priority: Priority) -> Lowered {
    Lowered::begin_reading(priority)
}

/// A layer whose banks were page-locked and checked, and whose locks are the caller's until [`Model::banks_locked`](crate::model::Model::banks_locked)
/// takes them.
pub struct Locked {
    pub layer: usize,
    /// The bytes of its banks.
    pub bytes: u64,
    banks: [Bank; 3],
}

// SAFETY: as [`BankLocks`]'s: the banks address mappings that the holder keeps alive.
unsafe impl Send for Locked {}

impl Locked {
    /// Give up the locks of a layer the model will not take (the engine ended meanwhile).
    ///
    /// # Safety
    /// The weights the plan was made from must still be mapped.
    pub unsafe fn unlock(self) {
        BankLocks::release(&self.banks);
    }
}

/// Why the locking stopped before its layers were done.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    /// The runtime would not page-lock the layer's banks (the commit or the memory is short): the layers that are locked stay so, the rest are
    /// left to the page cache.
    Refused { layer: usize, error: String },
    /// The layer's banks, page-locked, do not match the file's tree: the model's bytes are not those that were pinned.
    Unverified { layer: usize, error: String },
}

impl Stop {
    pub fn message(&self) -> String {
        match self {
            Stop::Refused { layer, error } => format!("registering layer {layer}'s banks failed ({error})"),
            Stop::Unverified { layer, error } => format!("Straydog model bytes do not verify: layer {layer}'s banks ({error})"),
        }
    }
}

/// The layers a start planned to page-lock, in the order they are to be taken, and what it takes to lock them off the model's thread.
pub struct BankLocks {
    layers: VecDeque<(usize, [Bank; 3])>,
    /// The device whose context the locking thread uses (the one that holds the page locks).
    ordinal: usize,
    /// The physical memory the plan keeps free ([`crate::host::left_free`]): a read-ahead stops when less than this is available.
    margin: u64,
    /// Set by the model when it gives locked layers back to a request ([`Model::give_back`](crate::model::Model::give_back)): the plan locks no more.
    halt: Arc<AtomicBool>,
    /// The bytes of K/V that the context's first 64K tokens still have to grow, which the model keeps up as its K/V grows ([`locking_room`]).
    kv_ahead: Arc<AtomicU64>,
}

// SAFETY: the banks hold the address of the weights' mappings, which the thread that uses them keeps alive ([`BankLocks::lock_next`]).
unsafe impl Send for BankLocks {}

impl BankLocks {
    pub(crate) fn new(layers: Vec<(usize, [Bank; 3])>, ordinal: usize, margin: u64) -> BankLocks {
        BankLocks { layers: layers.into(), ordinal, margin, halt: Arc::default(), kv_ahead: Arc::default() }
    }

    /// The plan tied to a model: it stops when `halt` is set, and `kv_ahead` says what K/V is still to grow ([`locking_room`]).
    pub(crate) fn tied_to(mut self, halt: Arc<AtomicBool>, kv_ahead: Arc<AtomicU64>) -> BankLocks {
        (self.halt, self.kv_ahead) = (halt, kv_ahead);
        self
    }

    /// The physical memory the plan keeps free: what the system must have available for a read-ahead of the banks to go on.
    pub fn margin(&self) -> u64 {
        self.margin
    }

    /// Whether the model has ended the locking to give its layers back.
    pub fn halted(&self) -> bool {
        self.halt.load(Ordering::SeqCst)
    }

    /// The layers still to lock (none once the model has ended the locking).
    pub fn remaining(&self) -> usize {
        if self.halted() { 0 } else { self.layers.len() }
    }

    /// The layer [`BankLocks::lock_next`] takes next.
    pub fn next_layer(&self) -> Option<usize> {
        if self.halted() { None } else { self.layers.front().map(|(layer, _)| *layer) }
    }

    /// Whether the next layer's turn would leave the machine too little memory with `available` bytes available now, and if so why
    /// ([`locking_room`]): the plan's margin and the K/V still to grow are kept free. None when it fits, and when no layer is left.
    pub fn room_for_next(&self, available: u64) -> Option<Short> {
        let layer_bytes = self.layers.front().filter(|_| !self.halted()).map(|(_, banks)| banks.iter().map(|bank| bank.bytes).sum::<u64>())?;
        locking_room(available, layer_bytes, self.margin, self.kv_ahead.load(Ordering::Relaxed)).err()
    }


    /// The plan's banks as pieces for a read-ahead, in the order the layers are taken ([`pieces_of`], [`AHEAD_PIECE`]); every byte of the banks still to lock
    /// and no other. Addresses in the weights' mappings, which whoever asks the system for them keeps alive.
    pub fn read_ahead(&self) -> Vec<Piece> {
        if self.halted() {
            return Vec::new();
        }
        let layers: Vec<(usize, [(usize, u64); 3])> = self.layers.iter()
            .map(|(layer, banks)| (*layer, [0, 1, 2].map(|role| (banks[role].map as usize + banks[role].offset as usize, banks[role].bytes))))
            .collect();
        pieces_of(&layers, AHEAD_PIECE)
    }

    /// Page-lock the next layer's banks and check them against the file's tree, holding nothing of the model: None when no layer is left.
    /// The next layer's pages are asked of the drive meanwhile (a hint: nothing is locked or taken), so that reading them overlaps this layer's
    /// check. The thread and the check's threads keep their priority and the check uses every core: for a locking that nothing runs beside (examples and
    /// tools); behind "ready" it is [`BankLocks::lock_next_behind`].
    ///
    /// # Safety
    /// The weights the plan was made from must stay alive, and mapped, for the whole call and until the layer returned is taken or
    /// unlocked: the caller holds a strong reference to the structure the model is in.
    pub unsafe fn lock_next(&mut self) -> Option<Result<Locked, Stop>> {
        // SAFETY: the caller's.
        unsafe { self.lock_at(Priority::Normal) }
    }

    /// [`BankLocks::lock_next`] for the locking behind "ready", which gives way to a request: the calling thread works at a lower priority for the
    /// layer's turn (Windows ends it after, Linux keeps it, [`Lowered`]; the memory priority stays normal on Windows, [`turn_lowered`]), so do the threads
    /// of the check, and there are at most a quarter as many of them as cores ([`check_threads`]). Reading the layer from the drive, registering its pages
    /// and hashing them take the cores and the memory bandwidth that the CPU's experts of a request need.
    ///
    /// # Safety
    /// As [`BankLocks::lock_next`].
    pub unsafe fn lock_next_behind(&mut self) -> Option<Result<Locked, Stop>> {
        // SAFETY: the caller's.
        unsafe { self.lock_at(Priority::Background) }
    }

    /// One layer's turn at `priority`.
    ///
    /// # Safety
    /// As [`BankLocks::lock_next`].
    unsafe fn lock_at(&mut self, priority: Priority) -> Option<Result<Locked, Stop>> {
        if self.halted() {
            return None;
        }
        let (layer, banks) = self.layers.pop_front()?;
        let _lowered = turn_lowered(priority);
        if let Some((_, next)) = self.layers.front() {
            // SAFETY: the caller's.
            let ranges: Vec<(usize, usize)> = next.iter().map(|bank| unsafe { bank.range() }).map(|range| (range.as_ptr() as usize, range.len())).collect();
            let _ = crate::host::prefetch(&ranges);
        }
        let mut locked = 0;
        for bank in &banks {
            // SAFETY: the caller's.
            if let Err(error) = crate::gpu::lock_host_range(self.ordinal, unsafe { bank.range() }) {
                Self::release(&banks[..locked]);
                return Some(Err(Stop::Refused { layer, error }));
            }
            locked += 1;
        }
        // Locked, the banks' pages stay as they are now: with the shards' verity trees, check them (they may have been read again since
        // loading verified their chunks).
        for bank in &banks {
            // SAFETY: the caller's, and the bank is locked above.
            if let Err(error) = unsafe { bank.verify(priority) } {
                Self::release(&banks);
                return Some(Err(Stop::Unverified { layer, error }));
            }
        }
        let bytes = banks.iter().map(|bank| bank.bytes).sum();
        Some(Ok(Locked { layer, bytes, banks }))
    }

    /// Give up the page locks of `banks`, taken by [`BankLocks::lock_next`].
    fn release(banks: &[Bank]) {
        for bank in banks {
            // SAFETY: the mapping is alive for the call that took the locks and for the one that gives them up.
            crate::gpu::unlock_host_range(unsafe { bank.range() });
        }
    }
}

/// The positions of K/V that the locking behind "ready" leaves room for: the context's first 64K tokens (all of it if the context is smaller). A
/// conversation that reaches further is served by the give-back ([`give_back_count`]), which takes the layers locked last back for its K/V; room for
/// the whole of a 262144-token context (12 GiB) would leave a machine with 63 GiB of memory 12 GiB fewer locked layers for the many conversations that
/// never reach it, and the first 64K are where nearly all of them end.
pub const KV_ROOM_POSITIONS: usize = 65536;

/// Why a layer's turn would leave the machine too little memory ([`locking_room`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Short {
    /// The memory available now, the layer's banks, what the plan keeps free, and the K/V still to grow.
    pub available: u64,
    pub layer_bytes: u64,
    pub margin: u64,
    pub kv_ahead: u64,
}

impl Short {
    /// The reason, as the log's line of the locking says it.
    pub fn message(&self) -> String {
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        format!("{:.1} GiB of memory was available, and the next layer's {:.1} GiB would have left less than the {:.1} GiB kept free and the {:.1} GiB \
                 of K/V that the context's first {KV_ROOM_POSITIONS} tokens still have to grow",
                gib(self.available), gib(self.layer_bytes), gib(self.margin), gib(self.kv_ahead))
    }
}

/// Whether a layer of `layer_bytes` can be locked with `available` bytes of memory available now: it must leave the plan's `margin` and the `kv_ahead`
/// bytes of K/V that the context's first 64K tokens still have to grow ([`KV_ROOM_POSITIONS`]) available.
pub fn locking_room(available: u64, layer_bytes: u64, margin: u64, kv_ahead: u64) -> Result<(), Short> {
    if available.saturating_sub(layer_bytes) >= margin.saturating_add(kv_ahead) {
        Ok(())
    } else {
        Err(Short { available, layer_bytes, margin, kv_ahead })
    }
}

/// How many of the locked layers a K/V growth (or another request's allocation) that is `shortfall` bytes short of memory takes back, newest first:
/// `locked` is the layers in the order they were locked, with their bytes. As many of the newest as their bytes cover the shortfall, and at least one
/// where any is locked (the allocation was refused: whatever the system says is available did not give it).
pub fn give_back_count(locked: &[(usize, u64)], shortfall: u64) -> usize {
    let (mut count, mut bytes) = (0, 0u64);
    for &(_, layer_bytes) in locked.iter().rev() {
        if count > 0 && bytes >= shortfall {
            break;
        }
        count += 1;
        bytes = bytes.saturating_add(layer_bytes);
    }
    count
}

/// What a give-back took, for its one line in the log ([`give_back_line`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaveBack {
    /// The layers given back, newest first, and their bytes.
    pub layers: Vec<usize>,
    pub bytes: u64,
    /// The layers that stay locked, and their bytes.
    pub left: usize,
    pub left_bytes: u64,
    /// What asked for it, the bytes it asked for, and the memory available when it did (None: not known).
    pub asked_by: String,
    pub wanted: u64,
    pub available: Option<u64>,
}

/// The one line a give-back leaves in the log.
pub fn give_back_line(gave: &GaveBack) -> String {
    let gib = |bytes: u64| format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64);
    let plural = |count: usize| if count == 1 { "" } else { "s" };
    let layers = gave.layers.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
    let available = gave.available.map_or("not known".to_string(), gib);
    let stay = if gave.left == 1 { "layer stays" } else { "layers stay" };
    format!("Straydog host banks given back: {} layer{} ({layers}), {} of page locks, for {} (it asked for {}, {available} available); {} {stay} locked, {}",
            gave.layers.len(), plural(gave.layers.len()), gib(gave.bytes), gave.asked_by, gib(gave.wanted), gave.left, gib(gave.left_bytes))
}

/// The bytes of one request of a read-ahead: a few hundred MB, so that the read can be followed and stopped between two (the size of a chunk of the pinned
/// files' digests).
pub const AHEAD_PIECE: u64 = 256 << 20;

/// One request of a read-ahead: a range of one bank of a layer of the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    /// The layer's place in the plan: 0 for the layer the locking takes first.
    pub turn: usize,
    pub layer: usize,
    /// The address the range begins at, in the mapping of the bank's file, and its bytes (whole pages but at the ends of the bank).
    pub start: usize,
    pub bytes: u64,
}

impl Piece {
    /// The fraction of the piece's pages that the system is known to hold in memory: Linux, the page cache's; Windows, this process's working set, which
    /// leaves out the pages that wait in the standby list (an earlier prefetch's, or an earlier process's), so it is a lower bound there.
    ///
    /// # Safety
    /// The mapping the piece is of must be alive.
    pub unsafe fn resident(&self) -> f64 {
        // SAFETY: the caller's; the range lies in the mapping (the plan was checked against it).
        let range = unsafe { std::slice::from_raw_parts(self.start as *const u8, self.bytes as usize) };
        if cfg!(windows) { crate::storage::in_working_set(range) } else { crate::storage::resident(range) }
    }

    /// Ask the system to read the piece's pages ([`crate::host::prefetch`]): a hint, nothing is locked or taken.
    ///
    /// # Safety
    /// The mapping the piece is of must be alive.
    pub unsafe fn ask(&self) -> Result<(), String> {
        crate::host::prefetch(&[(self.start, self.bytes as usize)])
    }
}

/// The pieces of a plan's `layers` (each with its three banks as an address and the bytes), in the order the layers are taken and the banks of a layer are
/// locked: each bank is cut at page boundaries into pieces of at most `limit` bytes. A bank of no bytes has no piece.
pub fn pieces_of(layers: &[(usize, [(usize, u64); 3])], limit: u64) -> Vec<Piece> {
    const PAGE: usize = 4096;
    let limit = limit.max(PAGE as u64);
    let mut pieces = Vec::new();
    for (turn, (layer, banks)) in layers.iter().enumerate() {
        for &(start, bytes) in banks {
            let end = start + bytes as usize;
            let mut at = start;
            while at < end {
                // (the next boundary a whole number of pages past `at`, `limit` or less away)
                let next = (at.saturating_add(limit as usize) & !(PAGE - 1)).max((at & !(PAGE - 1)) + PAGE).min(end);
                pieces.push(Piece { turn, layer: *layer, start: at, bytes: (next - at) as u64 });
                at = next;
            }
        }
    }
    pieces
}

/// The layers to page-lock: whole layers from the last, as many as `budget` holds. Reading went through the last layers most recently, so
/// they are the likeliest to be cached.
pub fn layers_within(sizes: &[u64], budget: u64) -> Vec<usize> {
    let (mut layers, mut taken) = (Vec::new(), 0u64);
    for layer in (0..sizes.len()).rev() {
        if taken + sizes[layer] > budget {
            break;
        }
        layers.push(layer);
        taken += sizes[layer];
    }
    layers
}

/// `layers` in the order they are locked: with an earlier run's counts, the layers whose experts the host ran most routes of first (locking
/// a layer lets the GPU take some of its absent routes from the CPU, so those are where it saves most); layers with as many routes keep the
/// order they were given, and without counts all do.
pub fn order_by_host_routes(layers: &mut [usize], host_routes: impl Fn(usize) -> u64) {
    layers.sort_by_key(|&layer| std::cmp::Reverse(host_routes(layer)));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decides: the check behind "ready" has at most a quarter of the cores and at least one, and the load's check, and a locking nothing runs beside,
    /// has every core.
    #[test]
    fn the_check_behind_ready_has_a_quarter_of_the_cores_and_the_load_all_of_them() {
        for cores in [1, 2, 3, 4, 7, 8, 12, 16, 24, 32, 64] {
            let behind = check_threads(Priority::Background, cores);
            assert!(behind >= 1 && behind <= cores.div_ceil(4).max(1), "{behind} threads behind ready on {cores} cores");
            assert_eq!(check_threads(Priority::Normal, cores), cores, "every core for the load's check on {cores} cores");
            if cores >= 2 {
                assert!(behind < cores, "{behind} threads behind ready on {cores} cores are not fewer than the cores");
            }
        }
        assert_eq!((check_threads(Priority::Background, 16), check_threads(Priority::Background, 8), check_threads(Priority::Background, 2)), (4, 2, 1));
        // (this machine's own count: what `Bank::verify` takes)
        let cores = std::thread::available_parallelism().map_or(8, |count| count.get());
        assert!(check_threads(Priority::Background, cores) <= check_threads(Priority::Normal, cores));
        assert_eq!(check_threads(Priority::Normal, 0), 1);
    }

    /// Decides: a layer's turn behind "ready" works in background mode for the CPU and the drive but keeps its thread's normal memory priority on Windows, so
    /// that the pages it asks the system for ahead are not the first the system reuses; the load's turn, and a thread that was not lowered, are unchanged.
    #[cfg(windows)]
    #[test]
    fn a_turn_behind_ready_keeps_its_memory_priority() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadInformation, ThreadMemoryPriority, MEMORY_PRIORITY_INFORMATION};
        fn memory_priority() -> u32 {
            let mut information = MEMORY_PRIORITY_INFORMATION { MemoryPriority: 0 };
            // SAFETY: the calling thread's pseudo handle and an initialized structure of the size passed.
            let read = unsafe {
                GetThreadInformation(GetCurrentThread(), ThreadMemoryPriority, std::ptr::from_mut(&mut information).cast(),
                                     std::mem::size_of::<MEMORY_PRIORITY_INFORMATION>() as u32)
            };
            assert_ne!(read, 0, "{}", std::io::Error::last_os_error());
            information.MemoryPriority
        }
        let (before, behind, load, after) = std::thread::spawn(|| {
            let before = memory_priority();
            let held = turn_lowered(Priority::Background);
            let behind = memory_priority();
            drop(held);
            let load = {
                let _held = turn_lowered(Priority::Normal);
                memory_priority()
            };
            (before, behind, load, memory_priority())
        }).join().unwrap();
        assert_eq!((behind, load, after), (before, before, before), "the memory priority is normal in a turn, and as it was after it");
    }

    #[test]
    fn whole_layers_from_the_last_within_the_budget() {
        let sizes = [10, 10, 10, 10, 10];
        assert_eq!(layers_within(&sizes, 0), Vec::<usize>::new());
        assert_eq!(layers_within(&sizes, 9), Vec::<usize>::new());
        assert_eq!(layers_within(&sizes, 25), vec![4, 3]);
        assert_eq!(layers_within(&sizes, 50), vec![4, 3, 2, 1, 0]);
        assert_eq!(layers_within(&sizes, u64::MAX), vec![4, 3, 2, 1, 0]);
        // a layer that does not fit ends the plan, though a smaller one before it would
        assert_eq!(layers_within(&[1, 1, 30, 1], 5), vec![3]);
    }

    #[test]
    fn counts_put_the_layers_the_host_ran_most_of_first() {
        let mut layers = vec![47, 46, 45, 44];
        order_by_host_routes(&mut layers, |_| 0);
        assert_eq!(layers, vec![47, 46, 45, 44], "no counts: the order given");
        let counts = |layer: usize| match layer {
            45 => 900,
            44 => 700,
            46 => 700,
            _ => 0,
        };
        order_by_host_routes(&mut layers, counts);
        assert_eq!(layers, vec![45, 46, 44, 47], "most first, equal counts in the order given, none last");
    }

    /// Decides: a give-back takes the layers that were locked last, one by one, until their bytes cover what the request is short of, and at least one where
    /// any is locked (a refused allocation that the system's own numbers called affordable); never more than are locked.
    #[test]
    fn a_give_back_takes_the_newest_layers_until_their_bytes_cover_the_shortfall() {
        const GIB: u64 = 1 << 30;
        // the order they were locked in: 40 first, 44 last; 1 GiB each but 42, which is 3
        let locked = [(40, GIB), (41, GIB), (42, 3 * GIB), (43, GIB), (44, GIB)];
        assert_eq!(give_back_count(&[], GIB), 0, "nothing is locked: nothing is given back");
        assert_eq!(give_back_count(&locked, 0), 1, "at least one, though nothing is short by the system's numbers");
        assert_eq!(give_back_count(&locked, 1), 1);
        assert_eq!(give_back_count(&locked, GIB), 1, "44 covers a GiB");
        assert_eq!(give_back_count(&locked, GIB + 1), 2, "44 and 43 cover 2 GiB");
        assert_eq!(give_back_count(&locked, 2 * GIB), 2);
        assert_eq!(give_back_count(&locked, 2 * GIB + 1), 3, "42's 3 GiB come with it whole");
        assert_eq!(give_back_count(&locked, 5 * GIB), 3);
        assert_eq!(give_back_count(&locked, 5 * GIB + 1), 4);
        assert_eq!(give_back_count(&locked, 7 * GIB), 5);
        assert_eq!(give_back_count(&locked, u64::MAX), 5, "all of them, and no more");
        // (the layers it names are the last of the list: the caller takes them from the end)
        let count = give_back_count(&locked, 2 * GIB + 1);
        assert_eq!(locked[locked.len() - count..].iter().map(|layer| layer.0).collect::<Vec<_>>(), vec![42, 43, 44]);
    }

    /// Decides: the line a give-back leaves says the layers (newest first), their bytes, what asked, how much and what was available, and what stays.
    #[test]
    fn the_give_back_line_says_what_was_given_why_and_what_stays() {
        const GIB: u64 = 1 << 30;
        let gave = GaveBack { layers: vec![44, 43], bytes: 3 * GIB / 2, left: 36, left_bytes: 23 * GIB, asked_by: "a K/V growth to 65536 positions".into(),
                              wanted: 4 * GIB, available: Some(GIB * 5 / 2) };
        assert_eq!(give_back_line(&gave),
                   "Straydog host banks given back: 2 layers (44, 43), 1.5 GiB of page locks, for a K/V growth to 65536 positions (it asked for 4.0 GiB, \
                    2.5 GiB available); 36 layers stay locked, 23.0 GiB");
        let last = GaveBack { layers: vec![47], bytes: GIB, left: 1, left_bytes: GIB, available: None, ..gave };
        assert_eq!(give_back_line(&last),
                   "Straydog host banks given back: 1 layer (47), 1.0 GiB of page locks, for a K/V growth to 65536 positions (it asked for 4.0 GiB, \
                    not known available); 1 layer stays locked, 1.0 GiB");
    }

    /// Decides: a layer is locked only where it leaves the plan's margin and the K/V the context's first 64K tokens still have to grow available: the
    /// edge is exact, no K/V left to grow asks the margin alone, and a layer larger than the memory available is never taken (no wrap-around).
    #[test]
    fn a_layer_is_locked_only_where_it_leaves_the_margin_and_the_k_v_still_to_grow() {
        const GIB: u64 = 1 << 30;
        // a machine of 63.6 GiB keeps 8 GiB free; a K/V at 32768 positions has 12 layers x 32768 x 4096 bytes = 1.5 GiB to grow to 65536
        let (margin, kv) = (8 * GIB, 3 * GIB / 2);
        assert!(locking_room(33 * GIB, 7 * GIB / 10, margin, kv).is_ok(), "plenty");
        assert!(locking_room(margin + kv + GIB, GIB, margin, kv).is_ok(), "exactly what is kept free and the K/V leave");
        let short = locking_room(margin + kv + GIB - 1, GIB, margin, kv).unwrap_err();
        assert_eq!(short, Short { available: margin + kv + GIB - 1, layer_bytes: GIB, margin, kv_ahead: kv }, "a byte short");
        assert!(locking_room(margin + GIB, GIB, margin, 0).is_ok(), "no K/V to grow: the margin alone");
        assert!(locking_room(margin + GIB - 1, GIB, margin, 0).is_err());
        assert!(locking_room(GIB / 2, GIB, margin, kv).is_err(), "a layer larger than what is available");
        assert!(locking_room(u64::MAX, GIB, u64::MAX, u64::MAX).is_err(), "nothing wraps");
        let text = short.message();
        for part in ["10.5 GiB of memory was available", "next layer's 1.0 GiB", "8.0 GiB kept free", "1.5 GiB of K/V", "first 65536 tokens"] {
            assert!(text.contains(part), "{part:?} in {text:?}");
        }
    }

    /// Decides: a plan asks the room rule before each layer with the memory available then (here a stand-in), with the K/V still to grow as the model keeps
    /// it up, and says nothing once its model has ended the locking or no layer is left.
    #[test]
    fn a_plan_asks_the_room_rule_of_its_next_layer_with_the_memory_available_now() {
        const GIB: u64 = 1 << 30;
        let path = std::env::temp_dir().join(format!("straydog-room-{}.bin", std::process::id()));
        std::fs::write(&path, vec![7u8; 8 << 20]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: a file of this test's own, which nothing changes while it is mapped.
        let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let bank = |index: u64| Bank::new(map.as_ptr(), map.len(), index << 20, 1 << 20, None);
        let (halt, kv) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(2 * GIB)));
        let locks = BankLocks::new(vec![(47, [bank(0), bank(1), bank(2)]), (46, [bank(3), bank(4), bank(5)])], 0, 3 * GIB).tied_to(halt.clone(), kv.clone());
        let layer = 3 << 20;
        // (3 GiB kept free and 2 GiB of K/V to grow: a layer of 3 MiB needs 5 GiB and 3 MiB)
        assert_eq!(locks.room_for_next(6 * GIB), None);
        assert_eq!(locks.room_for_next(5 * GIB + layer), None);
        assert_eq!(locks.room_for_next(5 * GIB + layer - 1), Some(Short { available: 5 * GIB + layer - 1, layer_bytes: layer, margin: 3 * GIB, kv_ahead: 2 * GIB }));
        // the K/V grew meanwhile: less of it is to come
        kv.store(0, Ordering::Relaxed);
        assert_eq!(locks.room_for_next(3 * GIB + layer), None);
        assert!(locks.room_for_next(3 * GIB + layer - 1).is_some());
        kv.store(2 * GIB, Ordering::Relaxed);
        assert!(locks.room_for_next(3 * GIB + layer).is_some());
        halt.store(true, Ordering::SeqCst);
        assert_eq!(locks.room_for_next(0), None, "the locking is ended: there is no turn to ask for");
        let empty = BankLocks::new(Vec::new(), 0, 3 * GIB);
        assert_eq!(empty.room_for_next(0), None, "no layer is left");
        drop(map);
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// Decides: a plan whose model ended the locking takes no layer, shows none left, and has nothing to read ahead; one that was not ended is as it was.
    #[test]
    fn a_plan_the_model_ended_locks_no_more() {
        let path = std::env::temp_dir().join(format!("straydog-halt-{}.bin", std::process::id()));
        std::fs::write(&path, vec![7u8; 8 << 20]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: a file of this test's own, which nothing changes while it is mapped.
        let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let bank = |index: u64| Bank::new(map.as_ptr(), map.len(), index << 20, 1 << 20, None);
        let halt = Arc::new(AtomicBool::new(false));
        let mut locks = BankLocks::new(vec![(47, [bank(0), bank(1), bank(2)]), (46, [bank(3), bank(4), bank(5)])], 0, 3 << 30)
            .tied_to(halt.clone(), Arc::default());
        assert_eq!((locks.remaining(), locks.next_layer(), locks.halted()), (2, Some(47), false));
        assert_eq!(locks.read_ahead().len(), 6);
        halt.store(true, Ordering::SeqCst);
        assert_eq!((locks.remaining(), locks.next_layer(), locks.halted()), (0, None, true));
        assert!(locks.read_ahead().is_empty());
        // SAFETY: nothing is locked: the plan is ended before it touches a bank.
        assert!(unsafe { locks.lock_next() }.is_none());
        assert!(unsafe { locks.lock_next_behind() }.is_none());
        drop(map);
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// Decides: a plan's banks are cut into pieces in the order of the layers and of the banks in a layer, at page boundaries, none of them over the limit,
    /// together exactly the banks' bytes (a read-ahead asks for no more than the plan), and a bank of no bytes has none.
    #[test]
    fn the_plans_banks_are_cut_into_pieces_in_the_order_they_are_locked() {
        const PAGE: usize = 4096;
        // (layer 47's banks start inside a page, layer 46's gate bank is empty, its down bank a page)
        let layers = [(47, [(0x1000_0000 + 100, 10 * PAGE as u64 + 7), (0x2000_0000, 3 * PAGE as u64), (0x3000_0000 + 4000, 96)]),
                      (46, [(0x4000_0000, 0), (0x5000_0000, 5000), (0x6000_0000, PAGE as u64)])];
        for limit in [4 * PAGE as u64, 3 * PAGE as u64 + 17, 1, u64::MAX] {
            let pieces = pieces_of(&layers, limit);
            // the bytes of each layer's banks, in all
            for (turn, (layer, banks)) in layers.iter().enumerate() {
                let of_layer: u64 = pieces.iter().filter(|piece| piece.turn == turn && piece.layer == *layer).map(|piece| piece.bytes).sum();
                assert_eq!(of_layer, banks.iter().map(|bank| bank.1).sum::<u64>(), "layer {layer} at a limit of {limit}");
            }
            assert!(pieces.iter().all(|piece| piece.bytes > 0 && piece.bytes <= limit.max(PAGE as u64)), "a piece over {limit}: {pieces:?}");
            // the layers in the plan's order, the banks of one in theirs, each bank's pieces joined end to end and cut at page boundaries
            assert!(pieces.windows(2).all(|pair| pair[0].turn <= pair[1].turn));
            let mut expected = Vec::new();
            for (_, banks) in &layers {
                for &(start, bytes) in banks {
                    let mut within: Vec<&Piece> = pieces.iter().filter(|piece| piece.start >= start && piece.start < start + bytes as usize && bytes > 0).collect();
                    within.sort_by_key(|piece| piece.start);
                    assert_eq!(within.first().map(|piece| piece.start), (bytes > 0).then_some(start));
                    for pair in within.windows(2) {
                        assert_eq!(pair[0].start + pair[0].bytes as usize, pair[1].start, "joined end to end");
                        assert_eq!(pair[1].start % PAGE, 0, "cut at a page boundary");
                    }
                    if let Some(last) = within.last() {
                        assert_eq!(last.start + last.bytes as usize, start + bytes as usize);
                    }
                    expected.extend(within.iter().map(|piece| piece.start));
                }
            }
            assert_eq!(pieces.iter().map(|piece| piece.start).collect::<Vec<_>>(), expected, "the order of the layers and of their banks, at a limit of {limit}");
        }
        // 256 MiB pieces of a 640 MiB bank: 256, 256 and 128
        let big = pieces_of(&[(1, [(1 << 30, 640 << 20), (0, 0), (0, 0)])], AHEAD_PIECE);
        assert_eq!(big.iter().map(|piece| piece.bytes >> 20).collect::<Vec<_>>(), vec![256, 256, 128]);
        assert!(pieces_of(&[], AHEAD_PIECE).is_empty());
    }

    /// Decides: a plan's read-ahead is the banks of the layers still to lock in the plan's order, at the addresses of their mappings and nothing else; a piece
    /// of a mapped file that has been touched is resident; and asking the system for it is accepted.
    #[test]
    fn a_plans_read_ahead_is_its_banks_in_order_and_a_touched_piece_is_resident() {
        let path = std::env::temp_dir().join(format!("straydog-ahead-{}.bin", std::process::id()));
        std::fs::write(&path, vec![7u8; 8 << 20]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: a file of this test's own, which nothing changes while it is mapped.
        let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let bank = |index: u64| Bank::new(map.as_ptr(), map.len(), index << 20, 1 << 20, None);
        let mut locks = BankLocks::new(vec![(47, [bank(0), bank(1), bank(2)]), (46, [bank(3), bank(4), bank(5)])], 0, 3 << 30);
        let pieces = locks.read_ahead();
        assert_eq!(pieces.iter().map(|piece| (piece.turn, piece.layer)).collect::<Vec<_>>(), vec![(0, 47), (0, 47), (0, 47), (1, 46), (1, 46), (1, 46)]);
        assert_eq!(pieces.iter().map(|piece| piece.start - map.as_ptr() as usize).collect::<Vec<_>>(), (0..6).map(|index| index << 20).collect::<Vec<_>>());
        assert_eq!(pieces.iter().map(|piece| piece.bytes).sum::<u64>(), 6 << 20, "the banks' bytes, no more");
        assert_eq!(locks.margin(), 3 << 30);
        // what the locking has taken is no longer in the plan
        locks.layers.pop_front();
        assert_eq!(locks.read_ahead().iter().map(|piece| piece.layer).collect::<Vec<_>>(), vec![46, 46, 46]);
        // (SAFETY below: the mapping is alive until the end of the test)
        let piece = pieces[1];
        assert!(unsafe { piece.ask() }.is_ok());
        let range = &map[(1 << 20)..(2 << 20)];
        let touched: u32 = range.iter().step_by(4096).map(|byte| u32::from(*byte)).sum();
        assert_eq!(touched, 7 * 256);
        assert_eq!(unsafe { piece.resident() }, 1.0, "every page of the piece was touched");
        drop(map);
        drop(file);
        let _ = std::fs::remove_file(&path);
    }
}
