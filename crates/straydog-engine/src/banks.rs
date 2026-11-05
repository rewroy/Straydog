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
//! The pages of the layers not yet locked are in memory only if the page cache holds them, and a request that comes before the locking is done
//! reads the rest by faulting, one fault after another (Linux, a cold page cache: 28 to 31 tok/s with 218,547 page faults against 49.4 tok/s after
//! the locking). [`BankLocks::read_ahead`] hands out the plan's banks as [`Piece`]s in the order the layers are locked, for a thread of the worker to ask the
//! system for at a low priority ([`Piece::ask`], [`Piece::resident`]), a few hundred MB at a time, so that the locking and a request find them in memory.

use std::collections::VecDeque;
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
}

// SAFETY: the banks hold the address of the weights' mappings, which the thread that uses them keeps alive ([`BankLocks::lock_next`]).
unsafe impl Send for BankLocks {}

impl BankLocks {
    pub(crate) fn new(layers: Vec<(usize, [Bank; 3])>, ordinal: usize, margin: u64) -> BankLocks {
        BankLocks { layers: layers.into(), ordinal, margin }
    }

    /// The physical memory the plan keeps free: what the system must have available for a read-ahead of the banks to go on.
    pub fn margin(&self) -> u64 {
        self.margin
    }

    /// The layers still to lock.
    pub fn remaining(&self) -> usize {
        self.layers.len()
    }

    /// The layer [`BankLocks::lock_next`] takes next.
    pub fn next_layer(&self) -> Option<usize> {
        self.layers.front().map(|(layer, _)| *layer)
    }

    /// The plan's banks as pieces for a read-ahead, in the order the layers are taken ([`pieces_of`], [`AHEAD_PIECE`]); every byte of the banks still to lock
    /// and no other. Addresses in the weights' mappings, which whoever asks the system for them keeps alive.
    pub fn read_ahead(&self) -> Vec<Piece> {
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
    /// layer's turn (Windows ends it after, Linux keeps it, [`Lowered`]), so do the threads of the check, and there are at most a quarter as many
    /// of them as cores ([`check_threads`]). Reading the layer from the drive, registering its pages and hashing them take the cores and the
    /// memory bandwidth that the CPU's experts of a request need.
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
        let (layer, banks) = self.layers.pop_front()?;
        let _lowered = Lowered::begin(priority);
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
