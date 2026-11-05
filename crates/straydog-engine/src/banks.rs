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

use std::collections::VecDeque;
use std::sync::Arc;

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

    /// Check the bank's bytes against the file's tree, block by block on every core: `Ok(false)` when the file has no tree.
    ///
    /// # Safety
    /// The mapping must be alive, and the bank page-locked (its bytes cannot change under the check).
    unsafe fn verify(&self) -> Result<bool, String> {
        let Some(verifier) = &self.verifier else { return Ok(false) };
        let threads = std::thread::available_parallelism().map_or(8, |count| count.get());
        // SAFETY: the caller's.
        let map = unsafe { std::slice::from_raw_parts(self.map, self.map_len) };
        verity::verify_mapped(map, verifier, self.offset, self.bytes, threads).map(|()| true)
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
}

// SAFETY: the banks hold the address of the weights' mappings, which the thread that uses them keeps alive ([`BankLocks::lock_next`]).
unsafe impl Send for BankLocks {}

impl BankLocks {
    pub(crate) fn new(layers: Vec<(usize, [Bank; 3])>, ordinal: usize) -> BankLocks {
        BankLocks { layers: layers.into(), ordinal }
    }

    /// The layers still to lock.
    pub fn remaining(&self) -> usize {
        self.layers.len()
    }

    /// The layer [`BankLocks::lock_next`] takes next.
    pub fn next_layer(&self) -> Option<usize> {
        self.layers.front().map(|(layer, _)| *layer)
    }

    /// Page-lock the next layer's banks and check them against the file's tree, holding nothing of the model: None when no layer is left.
    /// The next layer's pages are asked of the drive meanwhile (a hint: nothing is locked or taken), so that reading them overlaps this layer's
    /// check.
    ///
    /// # Safety
    /// The weights the plan was made from must stay alive, and mapped, for the whole call and until the layer returned is taken or
    /// unlocked: the caller holds a strong reference to the structure the model is in.
    pub unsafe fn lock_next(&mut self) -> Option<Result<Locked, Stop>> {
        let (layer, banks) = self.layers.pop_front()?;
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
            if let Err(error) = unsafe { bank.verify() } {
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
}
