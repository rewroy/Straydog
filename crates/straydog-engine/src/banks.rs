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
//! A GPU that cannot register host memory read-only (CUDA's `cudaDevAttrHostRegisterReadOnlySupported` 0: two RTX 3090s on Windows 10; or
//! `STRAYDOG_HOST_REGISTER_READ_ONLY=0`, [`lock_mode`]) cannot page-lock the model's read-only mapping in place at all, so a layer's turn is another
//! one ([`LockMode::Copy`]), with the same two steps and the same rule:
//!
//! 1. [`BankLocks::lock_next`] makes ONE page-locked copy of each of the layer's three banks ([`LayerCopy`]: portable, mapped host memory, so every
//!    device reads the one copy), reads the bank's bytes into it from the model's FILE, and checks THE COPY against the file's tree
//!    ([`verity::verify_copy_at`], at the bank's file offset, over the whole blocks around the bank, which the copy holds). A page-locked copy cannot
//!    change after the check, which is what keeps the rule: the bytes the GPU and the CPU's experts read are the bytes that were checked, not the
//!    mapping's, whose pages the system may read again from the drive. A copy that does not match is freed and ends the locking as `Unverified`.
//!
//!    The copy is read from the file and never from the mapping: a copy made from the mapping faults each of the layer's pages into this process's working
//!    set, and the memory then holds the layer twice, in the working set and in the copy, which the runtime allocated outside it (28.2 GiB of copies on a
//!    machine of 62 GiB: 37.5 to 42 GiB resident and 2.6 to 4.7 GiB available, against 23 to 27 GiB resident and 11.8 to 13.2 GiB available with the same
//!    layers registered in place). The file is read as the row tables' rows are ([`crate::rows`]): whole blocks past the page cache straight into the copy
//!    ([`crate::storage::direct`]), in smaller direct reads if the working-set quota refuses that (Windows' error 1453), and once through the page cache if
//!    it refuses those too ([`read_in_ways`]). The mapping's pages of a copied layer are not read at all, so the plan has no read-ahead
//!    ([`BankLocks::read_ahead`]) and a turn asks the system for no pages of the next layer.
//! 2. [`Model::banks_locked`](crate::model::Model::banks_locked), under the model's lock, points the plans' host pointers at the copies
//!    ([`Copies::take_over`], [`HostPointers`]): every reader of a bank's bytes goes through them (the slots' copies, the prompt's streaming, the CPU
//!    experts' gather and routes, the assist's mapping). The model keeps the copies ([`Copies`]) until it gives the layer back or ends.
//!
//! A give-back points the layer's plans back at the mapping first, which waits for everything that reads the copy, and only then lets the copy go
//! ([`Copies::give_back`]). What makes it safe to free the copy is the engine's lock (every CPU read of a bank is a synchronous call made under it, so
//! none runs during a give-back) and the waits in the device's repoint (`sd_hybrid_repoint`: its copy thread, its streams, and the `bank_readers` count
//! that the CPU's reads hold, an error if it does not drain). Nothing else holds a [`LayerCopy`] in production; the tests hold a second reference to the
//! memory to stand for a reader outside the engine's lock.
//!
//! The model's end does not point the plans back: `Model` has no `Drop`, and its fields drop in their order, the devices (which drain their copy thread
//! and destroy their plans) and the CPU engine (whose workers stop) before the [`Copies`], so that nothing can read a copy when it is freed. The order
//! matters: copies freed before the devices that read them would be read after they were freed. The memory is then freed by the runtime
//! (`sd_host_free` sets the device and calls `hipHostFree`), which outlives the device objects.
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
//! (A plan in the copy mode hands out none: its locks do not read the mapping.)

use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::host::{Lowered, Priority};
use crate::storage::{self, ALIGN};
use crate::verity::{self, Verifier};

/// One bank (a layer's gate, up or down experts) as a thread that does not hold the weights sees it: its file's mapping, its place in
/// that, the file's verity tree if it was opened, and the file's handles if the bank is to be copied ([`Bank::reading`]).
pub(crate) struct Bank {
    map: *const u8,
    map_len: usize,
    offset: u64,
    bytes: u64,
    verifier: Option<Arc<Verifier>>,
    file: Option<Arc<BankFile>>,
}

impl Bank {
    /// A bank at `offset` of `bytes` in the mapping `map` of `map_len` bytes. The mapping must outlive every use of the bank
    /// ([`BankLocks::lock_next`]); the range lies within it ([`Weights::mapped_bank`](crate::weights::Weights::mapped_bank) checks).
    pub(crate) fn new(map: *const u8, map_len: usize, offset: u64, bytes: u64, verifier: Option<Arc<Verifier>>) -> Bank {
        Bank { map, map_len, offset, bytes, verifier, file: None }
    }

    /// The bank with the handles of its file, which the copy mode reads its copy through ([`LayerCopy::make`]): the register mode needs none.
    pub(crate) fn reading(mut self, file: Arc<BankFile>) -> Bank {
        self.file = Some(file);
        self
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

/// The handles a model file is read through to copy its banks, made once for the file and shared by every bank in it: one that reads past the page cache
/// ([`storage::direct`], where the system gives one) and one that reads through it, for when the first is refused or the memory is not aligned for it.
pub(crate) struct BankFile {
    direct: Option<File>,
    cached: File,
}

impl BankFile {
    /// The handles of `file`: new ones on the same file ([`storage::direct`], [`storage::reader`]); the weights' own handle, which maps the file, stays what it was.
    pub(crate) fn of(file: &File) -> Result<BankFile, String> {
        let cached = storage::reader(file).map_err(|error| format!("cannot open the model file again to copy its banks: {error}"))?;
        Ok(BankFile { direct: storage::direct(file), cached })
    }
}

/// How a layer's banks are page-locked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockMode {
    /// The model's read-only mapping is registered with the runtime in place, and checked locked.
    Register,
    /// The runtime cannot register a read-only mapping: the banks are copied into page-locked memory, and the copies are checked.
    Copy,
}

/// The environment variable that says the GPUs cannot register host memory read-only (`0`), on any machine whatever the runtime answers: the copy mode
/// where it is not needed, for tests.
pub const READ_ONLY_REGISTRATION_ENV: &str = "STRAYDOG_HOST_REGISTER_READ_ONLY";

/// How the banks are page-locked, given the environment variable's value and each GPU's answer (ordinal, whether it registers host memory read-only; None:
/// the runtime cannot say, taken as it can), and if they are copied, why, for the one line that says so.
pub fn lock_mode(env: Option<&str>, devices: &[(usize, Option<bool>)]) -> (LockMode, Option<String>) {
    if env.is_some_and(|value| value.trim() == "0") {
        return (LockMode::Copy, Some(format!("{READ_ONLY_REGISTRATION_ENV}=0 says to treat the GPUs as unable to register host memory read-only")));
    }
    let refusing: Vec<String> = devices.iter().filter(|(_, answer)| *answer == Some(false)).map(|(ordinal, _)| format!("GPU {ordinal}")).collect();
    match refusing.len() {
        0 => (LockMode::Register, None),
        1 => (LockMode::Copy, Some(format!("{} does not support registering host memory read-only (cudaDevAttrHostRegisterReadOnlySupported 0)", refusing[0]))),
        _ => (LockMode::Copy, Some(format!("{} do not support registering host memory read-only (cudaDevAttrHostRegisterReadOnlySupported 0)", refusing.join(" and ")))),
    }
}

/// Where a bank's copy lives: page-locked host memory every device reads in place.
pub trait HostMemory: Send + Sync {
    /// `bytes` of page-locked memory, or the runtime's reason it would not.
    fn allocate(&self, bytes: u64) -> Result<*mut u8, String>;

    /// Give back what [`HostMemory::allocate`] returned.
    ///
    /// # Safety
    /// `pointer` came from `allocate` of this memory, once, and nothing reads it any more.
    unsafe fn free(&self, pointer: *mut u8);
}

/// Portable, mapped page-locked host memory, allocated or registered under the context of the device `ordinal`.
pub struct PinnedMemory {
    pub ordinal: usize,
}

impl HostMemory for PinnedMemory {
    fn allocate(&self, bytes: u64) -> Result<*mut u8, String> {
        crate::gpu::host_alloc(self.ordinal, bytes)
    }

    unsafe fn free(&self, pointer: *mut u8) {
        // SAFETY: the caller's.
        unsafe { crate::gpu::host_free(self.ordinal, pointer) }
    }
}

/// One bank's copy: an allocation of the whole blocks of the file around the bank, so that the copy can be checked against the file's tree as it is.
struct Part {
    /// The allocation (null until made) and the bytes of the file's blocks it holds, which the check sees: the allocation is these rounded up to whole blocks,
    /// since a direct read ends on a block boundary.
    base: *mut u8,
    bytes: usize,
    /// The file offset of the allocation's first byte (a block boundary), the bank's first byte's place in it, and the bank's bytes.
    start: u64,
    at: usize,
    len: usize,
}

impl Part {
    const NONE: Part = Part { base: std::ptr::null_mut(), bytes: 0, start: 0, at: 0, len: 0 };
}

/// The memory of one layer's three copies, freed with the last [`LayerCopy`] that holds it.
struct CopyBlock {
    memory: Arc<dyn HostMemory>,
    layer: usize,
    parts: [Part; 3],
}

// SAFETY: the parts are page-locked memory that only this block frees, and nothing writes after the copy was made and checked.
unsafe impl Send for CopyBlock {}
unsafe impl Sync for CopyBlock {}

impl Drop for CopyBlock {
    fn drop(&mut self) {
        for part in &self.parts {
            if !part.base.is_null() {
                // SAFETY: allocated by `memory`, once, and freed only here; whoever read it held a `LayerCopy`, which this block outlives.
                unsafe { self.memory.free(part.base) };
            }
        }
    }
}

/// One page-locked copy of a layer's three banks, shared by every device that reads the layer: the model's, which holds it until it gives the layer back
/// or ends (its readers are made safe by the engine's lock and the device's waits, see the module's docs). The memory is freed with its last holder.
pub struct LayerCopy(Arc<CopyBlock>);

impl LayerCopy {
    /// Allocate the copies of `banks` and read the bytes of their file into them, with the whole blocks around each bank ([`read_copy`]): from the file, not
    /// from the mapping, whose pages would join the working set (the module's docs). Nothing is checked yet ([`LayerCopy::verify`]). A bank that has no file
    /// to read ends it before anything is allocated; an allocation that is refused, or a read that fails, frees those made.
    fn make(memory: &Arc<dyn HostMemory>, layer: usize, banks: &[Bank; 3]) -> Result<LayerCopy, String> {
        let mut files = Vec::with_capacity(3);
        for (role, bank) in banks.iter().enumerate() {
            files.push(bank.file.as_deref().ok_or_else(|| format!("a bank with no file to copy from (role {role})"))?);
        }
        let mut block = CopyBlock { memory: memory.clone(), layer, parts: [Part::NONE, Part::NONE, Part::NONE] };
        let profile = cfg!(feature = "cuda") && std::env::var_os("DS41_PROFILE").is_some();
        for (role, (bank, file)) in banks.iter().zip(files).enumerate() {
            if bank.bytes == 0 {
                return Err(format!("a bank of no bytes (role {role})"));
            }
            let (start, end) = verity::covering_blocks(bank.map_len as u64, bank.offset, bank.bytes).ok_or("a bank past its file's end")?;
            let bytes = (end - start) as usize;
            let capacity = bytes.next_multiple_of(ALIGN);
            let started = std::time::Instant::now();
            let base = memory.allocate(capacity as u64)?;
            let allocation = started.elapsed();
            block.parts[role] = Part { base, bytes, start, at: (bank.offset - start) as usize, len: bank.bytes as usize };
            // SAFETY: `base` is a new allocation of `capacity` bytes, which nothing else reads or writes before the copy is made.
            let target = unsafe { std::slice::from_raw_parts_mut(base, capacity) };
            let started = std::time::Instant::now();
            read_copy(file, start, target, bytes).map_err(|error| format!("reading layer {layer}'s bank {role} from its file: {error}"))?;
            if profile {
                eprintln!("Straydog host banks copy: layer {layer}, role {role}, {bytes} bytes, allocation {:.3} s, read {:.3} s",
                          allocation.as_secs_f64(), started.elapsed().as_secs_f64());
            }
        }
        Ok(LayerCopy(Arc::new(block)))
    }

    /// Check the copy's own bytes against the file's tree, bank by bank, as the same blocks the mapping's check takes ([`Bank::verify`]): `Ok(false)` when
    /// the file has no tree.
    fn verify(&self, banks: &[Bank; 3], priority: Priority) -> Result<bool, String> {
        let threads = check_threads(priority, std::thread::available_parallelism().map_or(8, |count| count.get()));
        let mut checked = false;
        for (part, bank) in self.0.parts.iter().zip(banks) {
            let Some(verifier) = &bank.verifier else { continue };
            // SAFETY: the allocation of `bytes` bytes is alive as long as `self`, and written only by `make`.
            let held = unsafe { std::slice::from_raw_parts(part.base, part.bytes) };
            verity::verify_copy_at(held, part.start, verifier, bank.offset, bank.bytes, threads, priority)?;
            checked = true;
        }
        Ok(checked)
    }

    /// The layer the copy is of.
    pub fn layer(&self) -> usize {
        self.0.layer
    }

    /// The bytes of the bank `role` (gate, up, down), in the copy.
    pub fn bank(&self, role: usize) -> &[u8] {
        let part = &self.0.parts[role];
        // SAFETY: `part.at + part.len` lies in the allocation (`make`), which lives as long as `self`.
        unsafe { std::slice::from_raw_parts(part.base.add(part.at), part.len) }
    }

    /// The copy's three banks as the plans read them.
    pub fn pointers(&self) -> [*const u8; 3] {
        [0, 1, 2].map(|role| self.bank(role).as_ptr())
    }

    /// The bytes of the three banks.
    pub fn bytes(&self) -> u64 {
        self.0.parts.iter().map(|part| part.len as u64).sum()
    }

    /// Where the plans read the copy's banks from: each bank's first byte, and the start of the allocation it lies in (the bank begins inside the
    /// whole blocks the allocation holds, and the runtime maps an allocation by its start).
    pub fn banks(&self) -> Banks {
        Banks { start: self.pointers(), base: self.0.parts.each_ref().map(|part| part.base.cast_const()) }
    }
}

/// Where a layer's three banks are read from: each bank's first byte, and the start of the allocation it lies in (the same for the mapping).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Banks {
    pub start: [*const u8; 3],
    pub base: [*const u8; 3],
}

impl Banks {
    /// The three banks as the mapping holds them.
    pub fn of_mapping(mapping: [&[u8]; 3]) -> Banks {
        let start = mapping.map(<[u8]>::as_ptr);
        Banks { start, base: start }
    }
}

/// One region of an [`ExpertTier`]: an expert's bytes of one role, with the whole blocks around them, at `offset` of the tier's allocation.
#[derive(Clone, Copy)]
struct TierRegion {
    /// The file offset of the region's first byte (a block boundary), its place in the allocation, its bytes, and the expert's first byte's
    /// place and its bytes in it.
    start: u64,
    offset: usize,
    bytes: usize,
    at: usize,
    len: usize,
}

/// The memory of a layer's tier, freed with the last [`ExpertTier`] that holds it.
struct TierBlock {
    memory: Arc<dyn HostMemory>,
    layer: usize,
    base: *mut u8,
    capacity: usize,
    /// The tiered experts, in the order chosen, with their regions by role.
    experts: Vec<(usize, [TierRegion; 3])>,
}

// SAFETY: page-locked memory that only this block frees, and nothing writes after the tier was made and checked.
unsafe impl Send for TierBlock {}
unsafe impl Sync for TierBlock {}

impl Drop for TierBlock {
    fn drop(&mut self) {
        if !self.base.is_null() {
            // SAFETY: allocated by `memory`, once, and freed only here; whoever read it held an `ExpertTier`, which this block outlives.
            unsafe { self.memory.free(self.base) };
        }
    }
}

/// A page-locked copy of some of a layer's experts (the most routed ones), each read from the model's file with the whole blocks around it and
/// checked against the file's tree as a [`LayerCopy`] is: the pinned tier the devices read those experts' host bytes from
/// ([`crate::gpu::Backend::set_tier`]), where memory cannot hold the layer's whole banks. Freed with its last holder.
pub struct ExpertTier(Arc<TierBlock>);

impl ExpertTier {
    /// Allocate the tier of `chosen` of the layer's `count` experts and read their bytes from the banks' files: one allocation, each expert's
    /// three regions in it. Nothing is checked yet ([`ExpertTier::verify`]). An allocation that is refused, or a read that fails, frees it.
    pub(crate) fn make(memory: &Arc<dyn HostMemory>, layer: usize, banks: &[Bank; 3], count: usize, chosen: &[usize]) -> Result<ExpertTier, String> {
        if count == 0 || chosen.iter().any(|&expert| expert >= count) {
            return Err(format!("an expert past layer {layer}'s {count}"));
        }
        let mut files = Vec::with_capacity(3);
        for (role, bank) in banks.iter().enumerate() {
            files.push(bank.file.as_deref().ok_or_else(|| format!("a bank with no file to copy from (role {role})"))?);
        }
        let mut experts = Vec::with_capacity(chosen.len());
        let mut capacity = 0usize;
        for &expert in chosen {
            let mut regions = [TierRegion { start: 0, offset: 0, bytes: 0, at: 0, len: 0 }; 3];
            for (role, bank) in banks.iter().enumerate() {
                let per = bank.bytes / count as u64;
                let offset = bank.offset + expert as u64 * per;
                let (start, end) = verity::covering_blocks(bank.map_len as u64, offset, per).ok_or("an expert past its file's end")?;
                let bytes = (end - start) as usize;
                regions[role] = TierRegion { start, offset: capacity, bytes, at: (offset - start) as usize, len: per as usize };
                capacity += bytes.next_multiple_of(ALIGN);
            }
            experts.push((expert, regions));
        }
        if capacity == 0 {
            return Err(format!("no experts to tier in layer {layer}"));
        }
        let base = memory.allocate(capacity as u64)?;
        let block = TierBlock { memory: memory.clone(), layer, base, capacity, experts };
        // The experts' reads on several threads at once (as many as a bank's check takes now, check_threads): one at a time a layer's ~660
        // reads of half a megabyte took as long as the drive's latency each (17 GiB in 25.5 s on one R9700, where locking whole layers took 4-9).
        let threads = check_threads(Priority::Background, std::thread::available_parallelism().map_or(1, |cores| cores.get())).min(block.experts.len()).max(1);
        let address = base as usize;
        let failed = std::sync::Mutex::new(None);
        std::thread::scope(|scope| {
            for part in block.experts.chunks(block.experts.len().div_ceil(threads)) {
                let (files, failed) = (&files, &failed);
                scope.spawn(move || {
                    for (expert, regions) in part {
                        for (role, region) in regions.iter().enumerate() {
                            // SAFETY: within the new allocation, each region written by one thread alone, nothing reading it before the tier is made.
                            let target = unsafe { std::slice::from_raw_parts_mut((address as *mut u8).add(region.offset), region.bytes.next_multiple_of(ALIGN)) };
                            if let Err(error) = read_copy(files[role], region.start, target, region.bytes) {
                                failed.lock().unwrap().get_or_insert(format!("reading layer {layer}'s expert {expert} (role {role}) from its file: {error}"));
                                return;
                            }
                        }
                    }
                });
            }
        });
        if let Some(error) = failed.into_inner().unwrap() {
            return Err(error);
        }
        Ok(ExpertTier(Arc::new(block)))
    }

    /// Check the tier's own bytes against the file's tree, each expert's region as the blocks around it: `Ok(false)` when the files have no tree.
    pub(crate) fn verify(&self, banks: &[Bank; 3], priority: Priority) -> Result<bool, String> {
        if banks.iter().all(|bank| bank.verifier.is_none()) {
            return Ok(false);
        }
        // (the experts' checks on several threads, as many as a bank's check takes now: check_threads)
        let experts = &self.0.experts;
        let threads = check_threads(priority, std::thread::available_parallelism().map_or(1, |cores| cores.get())).min(experts.len()).max(1);
        let address = self.0.base as usize;
        let verifiers = [0, 1, 2].map(|role| banks[role].verifier.as_ref());
        let failed = std::sync::Mutex::new(None);
        std::thread::scope(|scope| {
            for part in experts.chunks(experts.len().div_ceil(threads)) {
                let (failed, verifiers) = (&failed, &verifiers);
                scope.spawn(move || {
                    for (_, regions) in part {
                        for (region, verifier) in regions.iter().zip(verifiers) {
                            let Some(verifier) = verifier else { continue };
                            // SAFETY: inside the allocation, alive as long as `self` and written only by `make`.
                            let held = unsafe { std::slice::from_raw_parts((address as *const u8).add(region.offset), region.bytes) };
                            if let Err(error) = verity::verify_copy_at(held, region.start, verifier, region.start + region.at as u64, region.len as u64, 1, priority) {
                                failed.lock().unwrap().get_or_insert(error);
                                return;
                            }
                        }
                    }
                });
            }
        });
        match failed.into_inner().unwrap() {
            Some(error) => Err(error),
            None => Ok(true),
        }
    }

    /// The layer the tier is of.
    pub fn layer(&self) -> usize {
        self.0.layer
    }

    /// Which of the layer's `count` experts the tier holds.
    pub fn held(&self, count: usize) -> Vec<bool> {
        let mut held = vec![false; count];
        for (expert, _) in &self.0.experts {
            if let Some(slot) = held.get_mut(*expert) {
                *slot = true;
            }
        }
        held
    }

    /// Per expert of the layer's `count`, its gate, up and down bytes in the tier (null where it has none), as the devices take them.
    pub fn pointers(&self, count: usize) -> Vec<[*const u8; 3]> {
        let mut pointers = vec![[std::ptr::null(); 3]; count];
        for (expert, regions) in &self.0.experts {
            // SAFETY: inside the allocation (`make`), which lives as long as `self`.
            pointers[*expert] = regions.map(|region| unsafe { self.0.base.add(region.offset + region.at).cast_const() });
        }
        pointers
    }

    /// The page-locked bytes the tier holds.
    pub fn bytes(&self) -> u64 {
        self.0.capacity as u64
    }
}

/// Read `bytes` bytes from `offset` of the bank's file (a block boundary) into `target`, the copy's allocation: directly where the file has a direct handle and
/// the allocation is aligned for it, else through the page cache ([`read_in_ways`]).
fn read_copy(file: &BankFile, offset: u64, target: &mut [u8], bytes: usize) -> std::io::Result<()> {
    let direct = file.direct.as_ref().filter(|_| offset.is_multiple_of(ALIGN as u64) && (target.as_ptr() as usize).is_multiple_of(ALIGN));
    read_in_ways(direct.is_some(), offset, target, bytes, storage::refused_for_pages, &mut |way, at, span, len| match (way, direct) {
        (Way::Direct, Some(direct)) => storage::read_direct_into(direct, at, span, len),
        _ => storage::read_at(&file.cached, at, &mut span[..len]),
    })
}

/// How a read of a copy's bytes goes: past the page cache, or through it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Way {
    Direct,
    Cached,
}

/// Read `bytes` bytes at `offset` into `target` with `read` (given the way, the file's offset, the part of `target` from there on, and the bytes to read), as
/// the row tables' reads go ([`crate::rows`]): the whole range direct (`direct`: whether that is possible), then the rest of it in direct pieces of a quarter of
/// the range if the whole was refused, then what is left once through the page cache if a piece was refused. Only the working-set quota's refusal (`refused`)
/// is a reason to try the next way: any other error ends the read. Never from the mapping, whose pages the read would bring into the working set.
pub(crate) fn read_in_ways(direct: bool, offset: u64, target: &mut [u8], bytes: usize, refused: fn(&std::io::Error) -> bool,
                read: &mut dyn FnMut(Way, u64, &mut [u8], usize) -> std::io::Result<()>) -> std::io::Result<()> {
    let mut done = 0;
    if direct {
        match read(Way::Direct, offset, target, bytes) {
            Ok(()) => return Ok(()),
            Err(error) if refused(&error) => {}
            Err(error) => return Err(error),
        }
        // (whole blocks; a piece that is no block, or no smaller than the range, is no smaller way)
        let piece = (bytes / 4).next_multiple_of(ALIGN);
        while piece > 0 && piece < bytes && done < bytes {
            let step = piece.min(bytes - done);
            match read(Way::Direct, offset + done as u64, &mut target[done..], step) {
                Ok(()) => done += step,
                Err(error) if refused(&error) => break,
                Err(error) => return Err(error),
            }
        }
        if done == bytes {
            return Ok(());
        }
    }
    read(Way::Cached, offset + done as u64, &mut target[done..], bytes - done)
}

/// A move of plans that failed ([`HostPointers::repoint`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveFailed {
    pub error: String,
    /// Every plan that was moved is back on the pointers it had, so nothing reads `to`. False: some plan may still read it (a device could not be put
    /// back), and `to` must be kept.
    pub restored: bool,
}

/// What the devices do with the host pointers of their plans: the one place a bank's bytes are read from (the slots' copies, the prompt's streaming, the
/// CPU experts' gather, the assist's mapping).
pub trait HostPointers {
    /// Every plan whose gate bank is at `from.start[0]` reads the same banks from `to` from now on, as page-locked copies (`copy`) or as the model's
    /// mapping; whether any plan did. It returns when nothing on the devices reads the old memory any more. On an error it has put the plans it moved
    /// back and says so (`restored`), or says that it could not.
    fn repoint(&mut self, from: Banks, to: Banks, copy: bool) -> Result<bool, MoveFailed>;
}

/// The copies a model holds, a layer's one copy however many devices read the layer.
#[derive(Default)]
pub struct Copies {
    held: BTreeMap<usize, LayerCopy>,
    /// Copies a plan may still read after a move that failed, which nothing frees before the model ends.
    stranded: Vec<LayerCopy>,
}

impl Copies {
    /// Whether the layer's banks are held as copies.
    pub fn holds(&self, layer: usize) -> bool {
        self.held.contains_key(&layer)
    }

    /// The copy of the layer, if there is one.
    pub fn get(&self, layer: usize) -> Option<&LayerCopy> {
        self.held.get(&layer)
    }

    /// The layers held as copies, and their bytes.
    pub fn held(&self) -> (usize, u64) {
        (self.held.len(), self.held.values().map(LayerCopy::bytes).sum())
    }

    /// Take a layer's checked copy: every plan that reads the layer from `mapping` reads it from the copy from now on. A layer is copied once: one that is
    /// held already is refused, and so is a layer no device holds (the copy is freed, nothing reads it). If the move fails and every plan is back on the
    /// mapping the copy is freed at once, since nothing reads it; if some plan may still read it, it is kept until the model ends.
    pub fn take_over(&mut self, plans: &mut dyn HostPointers, mapping: [&[u8]; 3], copy: LayerCopy) -> Result<(), String> {
        let layer = copy.layer();
        if self.held.contains_key(&layer) {
            return Err(format!("layer {layer}'s banks are copied already"));
        }
        if (0..3).any(|role| mapping[role].len() != copy.bank(role).len()) {
            return Err(format!("layer {layer}'s copy is not of the banks of the mapping"));
        }
        match plans.repoint(Banks::of_mapping(mapping), copy.banks(), true) {
            Ok(true) => {
                self.held.insert(layer, copy);
                Ok(())
            }
            Ok(false) => Err(format!("no device reads layer {layer}'s banks")),
            Err(failed) => {
                if !failed.restored {
                    self.stranded.push(copy);
                }
                Err(failed.error)
            }
        }
    }

    /// Give a layer's copy back: its plans read the mapping again, which waits for everything that read the copy, and only then does the model let the copy go
    /// (it is freed with its last holder, the model's). Whether the layer was held. If the move fails the copy stays held, whether the plans are back on
    /// it or some may still read it.
    pub fn give_back(&mut self, plans: &mut dyn HostPointers, layer: usize, mapping: [&[u8]; 3]) -> Result<bool, String> {
        let Some(copy) = self.held.get(&layer) else { return Ok(false) };
        plans.repoint(copy.banks(), Banks::of_mapping(mapping), false).map_err(|failed| failed.error)?;
        self.held.remove(&layer);
        Ok(true)
    }
}

/// A refusal of a layer's turn that came with nothing locked, kept by the model for the plans that follow ([`BankLocks::remembering`]): the runtime said no
/// to the very first layer, so a plan of the layers that are left made after every request would be refused at its first again. It is kept until the
/// memory available has grown by a layer's banks since it came, which is when a plan has something new to try; the refusal is most often a shortage.
#[derive(Clone, Default)]
pub struct Refusal(Arc<Mutex<Option<Refused>>>);

/// What a refusal was: the reason, the memory available when it came, and the bytes of the layer that was refused.
struct Refused {
    why: String,
    available: u64,
    layer_bytes: u64,
}

impl Refusal {
    /// A turn was refused with `taken` layers of its plan taken before it, `available` bytes of memory available then, and the refused layer's banks of
    /// `layer_bytes`.
    fn note(&self, taken: usize, stop: &Stop, available: u64, layer_bytes: u64) {
        if taken == 0 {
            *self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Refused { why: stop.message(), available, layer_bytes });
        }
    }

    /// Why no more is to be planned with `available` bytes of memory available now: a refusal came with nothing locked, and the memory has not grown by a
    /// layer's banks since. `locked` layers locked meanwhile end it (a refusal after some were is a lack of memory that a plan made when memory is free may
    /// get past), and so does that growth, which forgets the refusal: a plan that is refused again keeps its own.
    pub fn blocks_more(&self, locked: usize, available: u64) -> Option<String> {
        if locked > 0 {
            return None;
        }
        let mut held = self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let refused = held.as_ref()?;
        let again = refused.available.saturating_add(refused.layer_bytes);
        if available >= again {
            *held = None;
            return None;
        }
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        Some(format!("{}; planned again once {:.1} GiB of memory is available, a layer's {:.1} GiB more than the {:.1} GiB then",
                     refused.why, gib(again), gib(refused.layer_bytes), gib(refused.available)))
    }
}

/// How a plan registers the mapping's ranges with the runtime: the system's, but for a test's.
#[derive(Clone, Copy)]
struct Registrar {
    lock: fn(usize, &[u8]) -> Result<(), String>,
    unlock: fn(&[u8]),
}

impl Registrar {
    const SYSTEM: Registrar = Registrar { lock: crate::gpu::lock_host_range, unlock: crate::gpu::unlock_host_range };
}

/// Requests running now in this process ([`RequestRunning`]): the locking behind "ready" checks a bank on every core while there are none.
static REQUESTS_RUNNING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A request under way, counted while it lives (the worker holds one for each request it runs): the locking behind "ready" checks its banks on
/// a quarter of the cores meanwhile, and on all of them between requests ([`check_threads`]).
pub struct RequestRunning(());

impl RequestRunning {
    pub fn begin() -> RequestRunning {
        REQUESTS_RUNNING.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        RequestRunning(())
    }
}

impl Drop for RequestRunning {
    fn drop(&mut self) {
        REQUESTS_RUNNING.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// The threads that check a bank on a machine of `cores`, asked as each bank's check begins: every core for a locking that nothing waits behind,
/// and for the locking behind "ready" while no request runs (its threads at the lowered priority still); a quarter of them (one at least) behind
/// "ready" while one does, which leaves the rest to the request's CPU experts and the memory bandwidth they use. With a quarter of the cores
/// throughout, two RTX 3090 on a 20-vCPU host locked 46.8 GiB in 94.0 s after ready, with no request meanwhile; with all of them, 57.4 s.
pub(crate) fn check_threads(priority: Priority, cores: usize) -> usize {
    check_threads_with(priority, cores, REQUESTS_RUNNING.load(std::sync::atomic::Ordering::SeqCst))
}

fn check_threads_with(priority: Priority, cores: usize, running: usize) -> usize {
    match priority {
        Priority::Normal => cores.max(1),
        Priority::Background if running == 0 => cores.max(1),
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

/// One model's layers whose banks are registered in place (the register mode's locks), and the reads under way of each: a layer is marked when
/// its turn has locked and checked its banks, whether the model has taken it yet or not, and the mark goes when they are given up. A layer-order
/// prompt reads a marked layer's banks from the mapping, whose pages the locks keep in memory and checked, instead of reading them from the file
/// again ([`crate::layer_bank::Prefetch`]); so that no read finds them unlocked halfway (a page back from the drive is no checked byte), a read
/// holds the layer ([`Registered::read`]) and giving the locks up waits for it ([`Registered::release`]). The model owns it: another model's
/// marks are never this one's, and they end with it.
#[derive(Default)]
pub struct Registered {
    state: std::sync::Mutex<RegisteredState>,
    released: std::sync::Condvar,
}

struct RegisteredState {
    layers: u64,
    readers: [u32; 64],
}

impl Default for RegisteredState {
    fn default() -> RegisteredState {
        RegisteredState { layers: 0, readers: [0; 64] }
    }
}

impl Registered {
    fn state(&self) -> std::sync::MutexGuard<'_, RegisteredState> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn mark(&self, layer: usize) {
        if layer < 64 {
            self.state().layers |= 1 << layer;
        }
    }

    /// Whether `layer`'s banks are registered in place now.
    pub fn is_registered(&self, layer: usize) -> bool {
        layer < 64 && self.state().layers & (1 << layer) != 0
    }

    /// Hold `layer` registered for a read of its banks from the mapping, while the guard lives: None if it is not registered (read it as any
    /// other layer).
    pub fn read(self: &Arc<Self>, layer: usize) -> Option<RegisteredRead> {
        let mut state = self.state();
        if layer >= 64 || state.layers & (1 << layer) == 0 {
            return None;
        }
        state.readers[layer] += 1;
        Some(RegisteredRead { registered: self.clone(), layer })
    }

    /// `layer`'s registrations are about to be given up: no read takes it from now on, and the reads under way are waited for.
    pub fn release(&self, layer: usize) {
        if layer >= 64 {
            return;
        }
        let mut state = self.state();
        state.layers &= !(1u64 << layer);
        while state.readers[layer] > 0 {
            state = self.released.wait(state).unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

/// A read of a registered layer's banks under way ([`Registered::read`]): the layer stays registered until it drops.
pub struct RegisteredRead {
    registered: Arc<Registered>,
    layer: usize,
}

impl Drop for RegisteredRead {
    fn drop(&mut self) {
        let mut state = self.registered.state();
        state.readers[self.layer] -= 1;
        if state.readers[self.layer] == 0 {
            self.registered.released.notify_all();
        }
    }
}

/// A layer whose banks were page-locked and checked, and whose locks are the caller's until [`Model::banks_locked`](crate::model::Model::banks_locked)
/// takes them.
pub struct Locked {
    pub layer: usize,
    /// The bytes of its banks.
    pub bytes: u64,
    banks: [Bank; 3],
    /// The copy mode: the layer's page-locked copies, checked, that nothing reads yet.
    copy: Option<LayerCopy>,
    /// The tier mode: the layer's chosen experts' page-locked copies, checked, that nothing reads yet.
    tier: Option<ExpertTier>,
    /// The register mode: how the banks were registered, to give the registrations up, and the model's marks of them.
    registrar: Option<Registrar>,
    registered: Option<Arc<Registered>>,
}

// SAFETY: as [`BankLocks`]'s: the banks address mappings that the holder keeps alive.
unsafe impl Send for Locked {}

impl Locked {
    /// Give up the locks of a layer the model will not take (the engine ended meanwhile): the registrations of its banks, or its copies, which no plan reads
    /// yet and which are freed.
    ///
    /// # Safety
    /// The weights the plan was made from must still be mapped.
    pub unsafe fn unlock(self) {
        if let Some(registrar) = self.registrar {
            // (the mark first, and the reads under way waited for: then the locks)
            if let Some(registered) = &self.registered {
                registered.release(self.layer);
            }
            BankLocks::release(&registrar, &self.banks);
        }
    }

    /// Whether the layer was copied, not registered in place.
    pub fn is_copy(&self) -> bool {
        self.copy.is_some()
    }

    /// The layer's copies, for the model to take ([`Copies::take_over`]); none for a layer that was registered in place.
    pub fn take_copy(&mut self) -> Option<LayerCopy> {
        self.copy.take()
    }

    /// The layer's tier of experts (the tier mode, [`BankLocks::tiering`]), for the model to hand to the devices.
    pub fn take_tier(&mut self) -> Option<ExpertTier> {
        self.tier.take()
    }

    /// The address of the layer's gate bank in the mapping, which names its plan on the devices ([`crate::gpu::Backend::set_tier`]).
    pub fn gate_address(&self) -> *const u8 {
        // SAFETY: the address alone, not read.
        unsafe { self.banks[0].map.add(self.banks[0].offset as usize) }
    }
}

/// Why the locking stopped before its layers were done.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    /// The runtime would not page-lock the layer's banks (the commit or the memory is short): the layers that are locked stay so, the rest are
    /// left to the page cache.
    Refused { layer: usize, error: String },
    /// The runtime would not give page-locked memory for the copies of the layer's banks (the commit or the memory is short), or the banks' file could not be
    /// read into it.
    NotCopied { layer: usize, error: String },
    /// The layer's banks, page-locked, do not match the file's tree: the model's bytes are not those that were pinned.
    Unverified { layer: usize, error: String },
}

impl Stop {
    /// Whether the runtime refused the layer (it did not fail the check).
    pub fn is_refusal(&self) -> bool {
        matches!(self, Stop::Refused { .. } | Stop::NotCopied { .. })
    }

    pub fn message(&self) -> String {
        match self {
            Stop::Refused { layer, error } => format!("registering layer {layer}'s banks failed ({error})"),
            Stop::NotCopied { layer, error } => format!("copying layer {layer}'s banks into page-locked memory failed ({error})"),
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
    /// The copy mode: where the copies are made. None: the mapping itself is registered ([`LockMode::Register`]).
    copies: Option<Arc<dyn HostMemory>>,
    /// How the mapping is registered in the register mode, and where the layers registered are marked (the model's, [`Registered`]).
    registrar: Registrar,
    registered: Arc<Registered>,
    /// The layers of this plan handed out so far, and where a refusal is kept for the model ([`Refusal`]).
    taken: usize,
    refusal: Option<Refusal>,
    /// The tier mode ([`BankLocks::tiering`]): each layer's experts to copy (by layer), of `experts` a layer; None: whole layers.
    tier: Option<Arc<Vec<Vec<usize>>>>,
    experts: usize,
}

// SAFETY: the banks hold the address of the weights' mappings, which the thread that uses them keeps alive ([`BankLocks::lock_next`]).
unsafe impl Send for BankLocks {}

impl BankLocks {
    pub(crate) fn new(layers: Vec<(usize, [Bank; 3])>, ordinal: usize, margin: u64) -> BankLocks {
        BankLocks {
            layers: layers.into(), ordinal, margin, halt: Arc::default(), kv_ahead: Arc::default(), copies: None, registrar: Registrar::SYSTEM,
            registered: Arc::default(), taken: 0, refusal: None, tier: None, experts: 0,
        }
    }

    /// The plan in the copy mode ([`LockMode::Copy`]): its turns copy the banks into `memory` and check the copies, and register nothing.
    pub(crate) fn copying(mut self, memory: Arc<dyn HostMemory>) -> BankLocks {
        self.copies = Some(memory);
        self
    }

    /// The plan in the tier mode (plan/20261119-qwen-host-tier.md): each layer's turn copies only the experts `chosen` names for it (of
    /// `experts` a layer, [`tier_within`]) into `memory` and checks them ([`ExpertTier`]); a layer with none chosen is not in the plan.
    pub(crate) fn tiering(mut self, memory: Arc<dyn HostMemory>, chosen: Vec<Vec<usize>>, experts: usize) -> BankLocks {
        self.copies = Some(memory);
        self.tier = Some(Arc::new(chosen));
        self.experts = experts;
        self
    }

    /// The bytes the turn of `layer` with `banks` takes: its chosen experts' in the tier mode, its banks' otherwise.
    fn turn_bytes(&self, layer: usize, banks: &[Bank; 3]) -> u64 {
        let whole: u64 = banks.iter().map(|bank| bank.bytes).sum();
        match &self.tier {
            Some(chosen) if self.experts > 0 => whole / self.experts as u64 * chosen.get(layer).map_or(0, Vec::len) as u64,
            _ => whole,
        }
    }

    /// The plan keeps a refusal that came with nothing locked in `refusal`, for the model's later plans ([`Refusal::blocks_more`]).
    pub(crate) fn remembering(mut self, refusal: Refusal) -> BankLocks {
        self.refusal = Some(refusal);
        self
    }

    /// Tests only: the plan registers with `registrar` in place of the runtime.
    #[cfg(test)]
    fn registering_with(mut self, registrar: Registrar) -> BankLocks {
        self.registrar = registrar;
        self
    }

    /// How the plan page-locks a layer.
    pub fn mode(&self) -> LockMode {
        if self.copies.is_some() { LockMode::Copy } else { LockMode::Register }
    }

    /// The plan tied to a model: it stops when `halt` is set, and `kv_ahead` says what K/V is still to grow ([`locking_room`]).
    /// The plan marks the layers it registers in `registered` (the model's).
    pub(crate) fn marking(mut self, registered: Arc<Registered>) -> BankLocks {
        self.registered = registered;
        self
    }

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
        let layer_bytes = self.layers.front().filter(|_| !self.halted()).map(|(layer, banks)| self.turn_bytes(*layer, banks))?;
        locking_room(available, layer_bytes, self.margin, self.kv_ahead.load(Ordering::Relaxed)).err()
    }


    /// The plan's banks as pieces for a read-ahead, in the order the layers are taken ([`pieces_of`], [`AHEAD_PIECE`]); every byte of the banks still to lock
    /// and no other. Addresses in the weights' mappings, which whoever asks the system for them keeps alive. None in the copy mode: its turns read the
    /// file, not the mapping, so a read-ahead of the mapping's pages would read them for nothing and into the working set.
    pub fn read_ahead(&self) -> Vec<Piece> {
        if self.halted() || self.copies.is_some() {
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
        let layer_bytes = self.turn_bytes(layer, &banks);
        let _lowered = turn_lowered(priority);
        // (the copy mode reads the file: the next layer's pages in the mapping would be read for nothing, into the working set)
        if let Some((_, next)) = self.layers.front().filter(|_| self.copies.is_none()) {
            // SAFETY: the caller's.
            let ranges: Vec<(usize, usize)> = next.iter().map(|bank| unsafe { bank.range() }).map(|range| (range.as_ptr() as usize, range.len())).collect();
            let _ = crate::host::prefetch(&ranges);
        }
        let turn = match (self.copies.clone(), self.tier.clone()) {
            (Some(memory), Some(chosen)) => {
                Self::tier_turn(&memory, layer, banks, chosen.get(layer).map_or(&[][..], Vec::as_slice), self.experts, priority)
            }
            (Some(memory), None) => Self::copy_turn(&memory, layer, banks, priority),
            // SAFETY: the caller's.
            (None, _) => unsafe { self.register_turn(layer, banks, priority) },
        };
        Some(turn.inspect(|_| self.taken += 1).inspect_err(|stop| {
            if let (true, Some(refusal)) = (stop.is_refusal(), &self.refusal) {
                refusal.note(self.taken, stop, crate::host::available_bytes(), layer_bytes);
            }
        }))
    }

    /// The register mode's turn: page-lock the layer's banks in the mapping, then check them locked.
    ///
    /// # Safety
    /// As [`BankLocks::lock_next`].
    unsafe fn register_turn(&self, layer: usize, banks: [Bank; 3], priority: Priority) -> Result<Locked, Stop> {
        let started = std::time::Instant::now();
        let (registrar, ordinal) = (self.registrar, self.ordinal);
        // SAFETY: the caller's.
        let ranges: [&[u8]; 3] = [0, 1, 2].map(|role| unsafe { banks[role].range() });
        // A bank is registered on one thread and checked on many: the next bank is registered while this one is checked (on two RTX 3090,
        // 46.9 GiB, the registering took 27.7 s of a 53.4 s locking and the checks 22.5, one after the other). Locked, a bank's pages stay as they
        // are: with the shards' verity trees, check them (they may have been read again since loading verified their chunks).
        (registrar.lock)(ordinal, ranges[0]).map_err(|error| Stop::Refused { layer, error })?;
        let mut locked = 1;
        let mut registering = 0.0;
        let outcome = std::thread::scope(|scope| {
            for role in 0..3 {
                let next = (role + 1 < 3).then(|| scope.spawn(move || {
                    let began = std::time::Instant::now();
                    ((registrar.lock)(ordinal, ranges[role + 1]), began.elapsed().as_secs_f64())
                }));
                // SAFETY: the caller's, and the bank is locked.
                let checked = unsafe { banks[role].verify(priority) };
                let registered = next.map(|next| next.join().unwrap_or_else(|_| (Err("a bank's registration panicked".to_string()), 0.0)));
                if let Some((Ok(()), seconds)) = &registered {
                    locked += 1;
                    registering += seconds;
                }
                if let Err(error) = checked {
                    return Err(Stop::Unverified { layer, error });
                }
                if let Some((Err(error), _)) = registered {
                    return Err(Stop::Refused { layer, error });
                }
            }
            Ok(())
        });
        if let Err(stop) = outcome {
            Self::release(&self.registrar, &banks[..locked]);
            return Err(stop);
        }
        let registered = registering;
        let bytes = banks.iter().map(|bank| bank.bytes).sum();
        // (as the copy mode's line: where a layer's turn goes, registering its pages or checking them)
        if std::env::var_os("DS41_PROFILE").is_some() {
            eprintln!("Straydog host banks register: layer {layer}, {:.2} GiB, turn {:.3} s, the later banks' registering beside the checks {registered:.3} s, {} threads",
                      bytes as f64 / (1u64 << 30) as f64, started.elapsed().as_secs_f64(),
                      check_threads(priority, std::thread::available_parallelism().map_or(8, |count| count.get())));
        }
        self.registered.mark(layer);
        Ok(Locked { layer, bytes, banks, copy: None, tier: None, registrar: Some(self.registrar), registered: Some(self.registered.clone()) })
    }

    /// The copy mode's turn: read the layer's banks from their file into page-locked memory, then check the copies against the file's tree. The mapping is
    /// registered nowhere and not read, and the copies are read by no one until the model takes them. A copy that does not match is freed as it drops.
    fn copy_turn(memory: &Arc<dyn HostMemory>, layer: usize, banks: [Bank; 3], priority: Priority) -> Result<Locked, Stop> {
        let copy = LayerCopy::make(memory, layer, &banks).map_err(|error| Stop::NotCopied { layer, error })?;
        // Copied, the bytes are the copy's own and cannot change: with the shards' verity trees, check THEM, not the file they came from.
        let started = std::time::Instant::now();
        if let Err(error) = copy.verify(&banks, priority) {
            return Err(Stop::Unverified { layer, error });
        }
        if cfg!(feature = "cuda") && std::env::var_os("DS41_PROFILE").is_some() {
            eprintln!("Straydog host banks check: layer {layer}, verification {:.3} s", started.elapsed().as_secs_f64());
        }
        let bytes = banks.iter().map(|bank| bank.bytes).sum();
        Ok(Locked { layer, bytes, banks, copy: Some(copy), tier: None, registrar: None, registered: None })
    }

    /// The tier mode's turn: copy the layer's `chosen` experts (of `experts`) into page-locked memory and check the copies against the file's tree.
    fn tier_turn(memory: &Arc<dyn HostMemory>, layer: usize, banks: [Bank; 3], chosen: &[usize], experts: usize, priority: Priority)
                 -> Result<Locked, Stop> {
        let tier = ExpertTier::make(memory, layer, &banks, experts, chosen).map_err(|error| Stop::NotCopied { layer, error })?;
        if let Err(error) = tier.verify(&banks, priority) {
            return Err(Stop::Unverified { layer, error });
        }
        Ok(Locked { layer, bytes: tier.bytes(), banks, copy: None, tier: Some(tier), registrar: None, registered: None })
    }

    /// Give up the registrations of `banks`, made by [`BankLocks::lock_next`].
    fn release(registrar: &Registrar, banks: &[Bank]) {
        for bank in banks {
            // SAFETY: the mapping is alive for the call that took the locks and for the one that gives them up.
            (registrar.unlock)(unsafe { bank.range() });
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

/// The layers layer-order passes gave back (`given`, with their bytes, in the order given) to lock again with `left` bytes of commit left now:
/// those in order whose locks leave at least `floor`, what the last pass found with them all locked. A layer that does not fit ends the list.
pub fn relock_within(given: &[(usize, u64)], left: u64, floor: u64) -> Vec<usize> {
    let mut room = left.saturating_sub(floor);
    let mut layers = Vec::new();
    for &(layer, bytes) in given {
        if bytes > room {
            break;
        }
        room -= bytes;
        layers.push(layer);
    }
    layers
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

/// What [`visible_lock_cap`] leaves of the host memory the GPUs see beside the banks and the sequence state: the K/V's growth holds its old
/// and new planes at once for a moment, and a layer-order pass page-locks its rows.
pub const VISIBLE_KEPT: u64 = 2 << 30;

/// The most bank bytes to page-lock on Windows, given what DXGI says the process may reserve of the GPU's system memory group (`reservable`:
/// the NON_LOCAL group's AvailableForReservation, the least of the GPUs') and the page-locked sequence state (`pinned_state`: the host K/V at
/// the context's length and the checkpoints): what is left of it less [`VISIBLE_KEPT`]; no bound where Windows gives no such figure. Page-locked
/// memory is charged to that group 1 to 1. On an RTX 5070 Ti PC with 62 GiB (22.54 GiB reservable, 2.6 GiB of state, UD-IQ2_XS at --vram 5.5
/// and 9, -c 131072), a 99K prompt was read with 17.8 GiB of banks locked (at --vram 9) and with 19.7 GiB (the expert slots not kept across
/// the K/V's growth); with 21.4 GiB, with 19.7 GiB and the slots kept, and with 23.0 GiB (at --vram 9), the GPU's work failed as out of memory
/// with VRAM and the commit to spare, and the GPU faulted. The decode's speed hardly depends on the layers locked there (25 layers or none at
/// --vram 5.5: the same tokens a second).
pub fn visible_lock_cap(reservable: Option<u64>, pinned_state: u64) -> u64 {
    reservable.map_or(u64::MAX, |reservable| reservable.saturating_sub(pinned_state).saturating_sub(VISIBLE_KEPT))
}

/// The bytes of a layer-order pass's rows that may be page-locked beside the banks under `visible_cap` ([`visible_lock_cap`]): what the cap
/// left of the GPUs' view of host memory, [`VISIBLE_KEPT`]; no bound where no cap is in force (u64::MAX). A 99K prompt's one pass page-locks
/// 3.75 GiB of rows, which beside 18.8 GiB of banks and 1.4 GiB of state came to 24 GiB of the 22.5 GiB an RTX 5070 Ti PC with 62 GiB may
/// reserve; within this the pass takes 49,152 rows (three 16,384s, 1.9 GiB) and the prompt is read in passes of that many.
pub fn rows_visible_room(visible_cap: u64) -> u64 {
    if visible_cap == u64::MAX { u64::MAX } else { VISIBLE_KEPT }
}

/// The layers of `planned` (in their order) that fit `room` together, stopping at the first that does not.
pub fn layers_fitting(planned: &[usize], sizes: impl Fn(usize) -> u64, room: u64) -> Vec<usize> {
    let mut taken = 0u64;
    planned.iter().copied().take_while(|&layer| {
        taken = taken.saturating_add(sizes(layer));
        taken <= room
    }).collect()
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

/// The experts of a host tier (plan/20261119-qwen-host-tier.md): where `budget` holds every layer's banks (`sizes`), none (whole layers are
/// locked instead); otherwise, for each layer, the experts to copy into page-locked memory, the most routed of all layers first (`used`:
/// a layer and an expert's routes, an earlier run's counts), each costing `expert_bytes[layer]`, as long as the next one fits. Experts never
/// routed are not taken; equal counts keep the lower layer, then the lower expert. Each layer's list is in expert order.
pub fn tier_within(sizes: &[u64], expert_bytes: &[u64], experts: usize, budget: u64, used: &dyn Fn(usize, usize) -> u64)
                   -> Option<Vec<Vec<usize>>> {
    if sizes.iter().sum::<u64>() <= budget {
        return None;
    }
    let mut ranked: Vec<(u64, usize, usize)> = (0..sizes.len())
        .flat_map(|layer| (0..experts).map(move |expert| (layer, expert)))
        .map(|(layer, expert)| (used(layer, expert), layer, expert))
        .filter(|(count, _, _)| *count > 0)
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut chosen = vec![Vec::new(); sizes.len()];
    let mut taken = 0u64;
    for (_, layer, expert) in ranked {
        let bytes = expert_bytes[layer];
        if taken + bytes > budget {
            break;
        }
        taken += bytes;
        chosen[layer].push(expert);
    }
    for list in &mut chosen {
        list.sort_unstable();
    }
    Some(chosen)
}

/// The parts of a layer's banks (`banks`: gate, up, down, each `count` experts end to end) that hold the experts `held` does not, as ranges
/// (address, bytes) with neighbouring experts' merged: what a read back of a layer with a host tier asks for (the tier's experts are read
/// from its copies, never from these pages).
pub fn untiered_ranges(banks: &[&[u8]; 3], count: usize, held: &[bool]) -> Vec<(usize, usize)> {
    expert_ranges(banks, count, held, false)
}

/// The parts of a layer's banks that hold the experts `held` does: the mapping's pages of a tier's experts, which nothing reads once the tier
/// is made ([`crate::host::leave_working_set`] gives them back).
pub fn tiered_ranges(banks: &[&[u8]; 3], count: usize, held: &[bool]) -> Vec<(usize, usize)> {
    expert_ranges(banks, count, held, true)
}

/// The ranges of `banks` (address, bytes) of the experts whose `held` is `of_held`, neighbouring experts' merged, bank by bank.
fn expert_ranges(banks: &[&[u8]; 3], count: usize, held: &[bool], of_held: bool) -> Vec<(usize, usize)> {
    let is = |expert: usize| held.get(expert).copied().unwrap_or(false) == of_held;
    let mut ranges = Vec::new();
    for bank in banks {
        if bank.is_empty() || count == 0 {
            continue;
        }
        let per = bank.len() / count;
        let mut expert = 0;
        while expert < count {
            if !is(expert) {
                expert += 1;
                continue;
            }
            let first = expert;
            while expert < count && is(expert) {
                expert += 1;
            }
            ranges.push((bank.as_ptr() as usize + first * per, (expert - first) * per));
        }
    }
    ranges
}

/// The layers of a host tier still to make behind the requests (`left`, with their tier's bytes, in the order wanted): in order, as long as the
/// next one fits what the GPUs' view of host memory leaves (`visible`) and the memory `available` leaves beside `margin` and the K/V still to
/// grow (`kv_ahead`, [`locking_room`]), as whole layers behind the requests are planned. A layer that does not fit ends the list.
pub fn tier_more_within(left: &[(usize, u64)], visible: u64, available: u64, margin: u64, kv_ahead: u64) -> Vec<usize> {
    let (mut visible, mut room) = (visible, available);
    let mut layers = Vec::new();
    for &(layer, bytes) in left {
        if bytes > visible || locking_room(room, bytes, margin, kv_ahead).is_err() {
            break;
        }
        visible -= bytes;
        room -= bytes;
        layers.push(layer);
    }
    layers
}

/// The unlocked layers (`layers`, with their banks' bytes, in the order wanted) whose banks a read back takes within `budget` bytes: in order,
/// as long as the next one fits. A layer that does not fit ends the list, so that the likeliest go first and whole.
pub fn read_back_plan(layers: &[(usize, u64)], budget: u64) -> Vec<usize> {
    let mut left = budget;
    let mut planned = Vec::new();
    for &(layer, bytes) in layers {
        if bytes > left {
            break;
        }
        left -= bytes;
        planned.push(layer);
    }
    planned
}

/// How long a read back waits after reading `bytes` in `took` so that it reads at no more than `rate` bytes a second on average: the drive
/// is shared with the reply's own faults.
pub fn read_back_pause(bytes: u64, took: std::time::Duration, rate: u64) -> std::time::Duration {
    if rate == 0 {
        return std::time::Duration::ZERO;
    }
    std::time::Duration::from_secs_f64(bytes as f64 / rate as f64).saturating_sub(took)
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

    /// Decides: the banks' lock leaves the GPUs' view of host memory the sequence state and VISIBLE_KEPT, so that the 62 GiB PC's 22.54 GiB
    /// reservable with 2.6 GiB of state leaves 17.96 GiB to the banks: UD-IQ2_XS's last 22 layers (17.8 GiB, as a 99K prompt was read with)
    /// and not 25 (20.4 GiB); 21 of layers of its average size. No figure, no bound; and the layers given back come back only as far as the
    /// room holds them.
    #[test]
    fn the_lock_leaves_the_gpus_view_of_host_memory_the_sequence_state_and_a_margin() {
        const GIB: u64 = 1 << 30;
        let gib = |value: f64| (value * GIB as f64) as u64;
        // (UD-IQ2_XS: 39.4 GiB of banks over 48 layers, 0.82 GiB a layer on average)
        let sizes = vec![gib(39.4 / 48.0); 48];
        let cap = visible_lock_cap(Some(gib(22.54)), gib(2.58));
        assert_eq!(cap, gib(22.54) - gib(2.58) - VISIBLE_KEPT);
        let layers = layers_within(&sizes, cap);
        assert_eq!(layers.len(), 21, "21 layers of {:.2} GiB fit {:.2} GiB", 39.4 / 48.0, cap as f64 / GIB as f64);
        assert!(layers_within(&sizes, cap).iter().map(|&layer| sizes[layer]).sum::<u64>() <= cap);
        assert_eq!(visible_lock_cap(None, gib(2.58)), u64::MAX);
        assert_eq!(visible_lock_cap(Some(gib(3.0)), gib(2.58)), 0, "less than the state and the margin: nothing");
        let given = [27, 28, 29];
        assert_eq!(layers_fitting(&given, |_| 2 * GIB, 5 * GIB), vec![27, 28]);
        assert_eq!(layers_fitting(&given, |_| 2 * GIB, GIB), Vec::<usize>::new());
        assert_eq!(layers_fitting(&given, |_| 2 * GIB, u64::MAX), vec![27, 28, 29]);
    }

    /// Decides: a read back takes whole layers in the order given while each fits what is left of its budget, and stops at the first that does
    /// not; it waits after each so that its reads average the rate given, and not at all when its read was already that slow.
    #[test]
    fn a_read_back_takes_whole_layers_in_order_within_its_budget_and_its_rate() {
        const GIB: u64 = 1 << 30;
        let layers = [(30, GIB), (12, GIB), (40, 2 * GIB), (7, GIB)];
        assert_eq!(read_back_plan(&layers, 5 * GIB), vec![30, 12, 40, 7]);
        assert_eq!(read_back_plan(&layers, 3 * GIB), vec![30, 12], "40's two GiB do not fit the one left, and 7 waits behind it");
        assert_eq!(read_back_plan(&layers, GIB - 1), Vec::<usize>::new());
        assert_eq!(read_back_plan(&[], 5 * GIB), Vec::<usize>::new());
        let second = std::time::Duration::from_secs(1);
        assert_eq!(read_back_pause(2 * GIB, second / 4, GIB), 2 * second - second / 4);
        assert_eq!(read_back_pause(GIB, 3 * second, GIB), std::time::Duration::ZERO);
        assert_eq!(read_back_pause(GIB, second / 2, 0), std::time::Duration::ZERO);
    }

    /// Decides: the check behind "ready" has at most a quarter of the cores and at least one while a request runs, and every core between
    /// requests, as the load's check and a locking nothing runs beside always have; the count follows the requests as they begin and end.
    #[test]
    fn the_check_behind_ready_has_a_quarter_of_the_cores_while_a_request_runs_and_all_of_them_else() {
        for cores in [1, 2, 3, 4, 7, 8, 12, 16, 24, 32, 64] {
            let behind = check_threads_with(Priority::Background, cores, 1);
            assert!(behind >= 1 && behind <= cores.div_ceil(4).max(1), "{behind} threads behind ready on {cores} cores");
            assert_eq!(check_threads_with(Priority::Background, cores, 0), cores, "every core behind ready with no request on {cores} cores");
            assert_eq!(check_threads_with(Priority::Normal, cores, 1), cores, "every core for the load's check on {cores} cores");
            if cores >= 2 {
                assert!(behind < cores, "{behind} threads behind ready on {cores} cores are not fewer than the cores");
            }
        }
        assert_eq!((check_threads_with(Priority::Background, 16, 2), check_threads_with(Priority::Background, 8, 1),
                    check_threads_with(Priority::Background, 2, 1)), (4, 2, 1));
        // (the count as the worker keeps it: a request begun lowers the check, its end raises it again)
        let cores = 16;
        let before = REQUESTS_RUNNING.load(std::sync::atomic::Ordering::SeqCst);
        {
            let _request = RequestRunning::begin();
            assert_eq!(REQUESTS_RUNNING.load(std::sync::atomic::Ordering::SeqCst), before + 1);
            assert_eq!(check_threads_with(Priority::Background, cores, before + 1), 4);
        }
        assert_eq!(REQUESTS_RUNNING.load(std::sync::atomic::Ordering::SeqCst), before);
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

    /// Decides: the layers passes gave back are locked again in the order given while their locks leave the commit where the last pass found it
    /// with them all locked; one that does not fit ends it, and with the commit at or under that mark none is.
    #[test]
    fn layers_a_pass_gave_back_are_locked_again_within_the_commit_it_found() {
        const GIB: u64 = 1 << 30;
        let given = [(27, GIB), (28, GIB), (29, 2 * GIB)];
        // (the pass found 5 GiB with them all locked: 9 GiB now is room for all four GiB of them)
        assert_eq!(relock_within(&given, 9 * GIB, 5 * GIB), vec![27, 28, 29]);
        assert_eq!(relock_within(&given, 8 * GIB, 5 * GIB), vec![27, 28], "29's two GiB would leave less than the pass found");
        assert_eq!(relock_within(&given, 7 * GIB - 1, 5 * GIB), vec![27]);
        assert_eq!(relock_within(&given, 5 * GIB, 5 * GIB), Vec::<usize>::new(), "other programs took what the pass let go");
        assert_eq!(relock_within(&given, 4 * GIB, 5 * GIB), Vec::<usize>::new());
        assert_eq!(relock_within(&[], 9 * GIB, 5 * GIB), Vec::<usize>::new());
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

    // ---- the copy mode ----

    use std::sync::atomic::AtomicUsize;

    /// A file of 6 MiB and a few bytes, with its tree built, and its mapping. `tamper` changes a byte of the file after the tree was built.
    struct Fixture {
        path: std::path::PathBuf,
        tree: std::path::PathBuf,
        map: Option<memmap2::Mmap>,
        verifier: Option<Arc<Verifier>>,
        /// The handles a copy is read through.
        file: Option<Arc<BankFile>>,
    }

    const SIZE: usize = (6 << 20) + 123;
    /// Three banks of the file: starts inside a block, and the last reaches the file's end (a partial last block).
    const SPANS: [(u64, u64); 3] = [(1000, 1_100_001), (2_000_003, 1_300_001), (SIZE as u64 - 900_001, 900_001)];

    impl Fixture {
        fn new(label: &str, tamper: Option<usize>) -> Fixture {
            let mut data: Vec<u8> = (0..SIZE).map(|at| (at * 37 % 251) as u8).collect();
            let path = std::env::temp_dir().join(format!("straydog-copy-{label}-{}.bin", std::process::id()));
            let tree = path.with_extension("tree");
            std::fs::write(&path, &data).unwrap();
            let digest = verity::build(&std::fs::File::open(&path).unwrap(), &tree, &|_| {}).unwrap();
            if let Some(at) = tamper {
                data[at] ^= 1;
                std::fs::write(&path, &data).unwrap();
            }
            let verifier = Arc::new(Verifier::open(&tree, SIZE as u64, &digest, 4 * verity::BLOCK).unwrap());
            let file = std::fs::File::open(&path).unwrap();
            // SAFETY: a file of this test's own, which nothing changes while it is mapped.
            let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
            let handles = Arc::new(BankFile::of(&file).unwrap());
            Fixture { path, tree, map: Some(map), verifier: Some(verifier), file: Some(handles) }
        }

        fn map(&self) -> &memmap2::Mmap {
            self.map.as_ref().unwrap()
        }

        /// The three banks of [`SPANS`], with the file's tree or without, and the file to copy them from.
        fn banks(&self, tree: bool) -> [Bank; 3] {
            SPANS.map(|(offset, bytes)| {
                Bank::new(self.map().as_ptr(), self.map().len(), offset, bytes, tree.then(|| self.verifier.clone().unwrap())).reading(self.file.clone().unwrap())
            })
        }

        /// The three banks as the mapping holds them.
        fn mapping(&self) -> [&[u8]; 3] {
            SPANS.map(|(offset, bytes)| &self.map()[offset as usize..(offset + bytes) as usize])
        }

        /// Everything that holds the file is dropped by now: remove it.
        fn finish(mut self) {
            self.map.take();
            self.verifier.take();
            self.file.take();
            let _ = std::fs::remove_file(&self.path);
            let _ = std::fs::remove_file(&self.tree);
        }
    }

    /// Host memory from the heap, in place of the runtime's page-locked memory, that counts what it gives and takes back and can refuse.
    #[derive(Default)]
    struct HeapMemory {
        live: Mutex<BTreeMap<usize, std::alloc::Layout>>,
        attempts: AtomicUsize,
        allocated: AtomicUsize,
        freed: AtomicUsize,
        /// The number of the allocation from which this refuses (1 the first; 0 never).
        refuse_from: AtomicUsize,
    }

    impl HostMemory for HeapMemory {
        fn allocate(&self, bytes: u64) -> Result<*mut u8, String> {
            // (a direct read ends on a block boundary, so a copy's allocation reaches one)
            assert!(bytes.is_multiple_of(ALIGN as u64), "a copy's allocation of {bytes} bytes is not whole blocks");
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
            let from = self.refuse_from.load(Ordering::SeqCst);
            if from != 0 && attempt >= from {
                return Err("out of memory".into());
            }
            let layout = std::alloc::Layout::from_size_align(bytes as usize, 4096).unwrap();
            // SAFETY: a layout of a nonzero size.
            let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!pointer.is_null());
            self.live.lock().unwrap().insert(pointer as usize, layout);
            self.allocated.fetch_add(1, Ordering::SeqCst);
            Ok(pointer)
        }

        unsafe fn free(&self, pointer: *mut u8) {
            let layout = self.live.lock().unwrap().remove(&(pointer as usize)).expect("memory is freed once, and only what was allocated");
            // (what a reader that outlived its copy would see)
            // SAFETY: the allocation of this layout, still ours.
            unsafe {
                std::ptr::write_bytes(pointer, 0xDD, layout.size());
                std::alloc::dealloc(pointer, layout);
            }
            self.freed.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl HeapMemory {
        fn live(&self) -> usize {
            self.live.lock().unwrap().len()
        }
        fn allocated(&self) -> usize {
            self.allocated.load(Ordering::SeqCst)
        }
        fn freed(&self) -> usize {
            self.freed.load(Ordering::SeqCst)
        }
        fn refusing_from(attempt: usize) -> Arc<HeapMemory> {
            let memory = HeapMemory::default();
            memory.refuse_from.store(attempt, Ordering::SeqCst);
            Arc::new(memory)
        }
    }

    /// How a fake device group's move fails.
    #[derive(Clone, Copy, PartialEq)]
    enum Failing {
        No,
        /// The move fails, and the plans it moved are back where they were.
        Restored,
        /// The move fails with the first plan moved, and a device that could not be put back: that plan may read the new memory.
        Stuck,
    }

    /// Devices with plans, as the group has them: each plan reads a layer's banks from three addresses. `last` is where the last move pointed them.
    struct Devices {
        plans: Vec<Vec<[*const u8; 3]>>,
        fail: Failing,
        last: Option<Banks>,
    }

    impl HostPointers for Devices {
        fn repoint(&mut self, from: Banks, to: Banks, _copy: bool) -> Result<bool, MoveFailed> {
            self.last = Some(to);
            match self.fail {
                Failing::No => {}
                Failing::Restored => return Err(MoveFailed { error: "a device would not move the plan".into(), restored: true }),
                Failing::Stuck => {
                    if let Some(plan) = self.plans.iter_mut().flatten().find(|plan| plan[0] == from.start[0]) {
                        *plan = to.start;
                    }
                    return Err(MoveFailed { error: "a device would not move the plan, nor be put back".into(), restored: false });
                }
            }
            let mut found = false;
            for plan in self.plans.iter_mut().flatten() {
                if plan[0] == from.start[0] {
                    *plan = to.start;
                    found = true;
                }
            }
            Ok(found)
        }
    }

    fn pointers(banks: [&[u8]; 3]) -> [*const u8; 3] {
        banks.map(<[u8]>::as_ptr)
    }

    /// A copy of layer 7 made by a plan in the copy mode.
    fn copied(fixture: &Fixture, memory: &Arc<HeapMemory>) -> LayerCopy {
        let mut locks = BankLocks::new(vec![(7, fixture.banks(true))], 0, 1 << 30).copying(memory.clone());
        // SAFETY: the fixture keeps the mapping alive.
        let mut locked = unsafe { locks.lock_next() }.expect("a layer").unwrap_or_else(|stop| panic!("{}", stop.message()));
        locked.take_copy().expect("a copy")
    }

    /// Decides: a tiered layer's read back is its banks' parts the tier does not hold, neighbouring experts' merged, in each role.
    #[test]
    fn a_tiered_layers_read_back_leaves_the_tiers_experts_out() {
        let (gate, up, down) = (vec![0u8; 40], vec![0u8; 40], vec![0u8; 80]);
        let banks = [&gate[..], &up[..], &down[..]];
        let held = [true, false, false, true];
        let ranges = untiered_ranges(&banks, 4, &held);
        let at = |bank: &[u8], offset: usize| bank.as_ptr() as usize + offset;
        assert_eq!(ranges, vec![(at(&gate, 10), 20), (at(&up, 10), 20), (at(&down, 20), 40)]);
        assert_eq!(untiered_ranges(&banks, 4, &[false; 4]).len(), 3, "nothing held: the banks whole");
        assert!(untiered_ranges(&banks, 4, &[true; 4]).is_empty(), "all held: nothing to read");
    }

    /// Decides: what leaves the working set once a tier is made is its experts' parts of the mapping and nothing of the others', the
    /// complement of the read back's ranges.
    #[test]
    fn a_tiers_mapping_left_is_its_experts_alone() {
        let (gate, up, down) = (vec![0u8; 40], vec![0u8; 40], vec![0u8; 80]);
        let banks = [&gate[..], &up[..], &down[..]];
        let held = [true, true, false, true];
        let at = |bank: &[u8], offset: usize| bank.as_ptr() as usize + offset;
        assert_eq!(tiered_ranges(&banks, 4, &held),
                   vec![(at(&gate, 0), 20), (at(&gate, 30), 10), (at(&up, 0), 20), (at(&up, 30), 10), (at(&down, 0), 40), (at(&down, 60), 20)]);
        let both: usize = tiered_ranges(&banks, 4, &held).iter().chain(&untiered_ranges(&banks, 4, &held)).map(|(_, bytes)| bytes).sum();
        assert_eq!(both, 160, "the two together cover the banks once");
        assert!(tiered_ranges(&banks, 4, &[false; 4]).is_empty(), "nothing held: nothing leaves");
    }

    /// Decides: a tier's layers left behind by a locking that stopped short are planned behind the requests in order, while the GPUs' view
    /// and the memory available beside what is kept free hold the next; the first that does not fit ends the plan.
    #[test]
    fn a_tier_stopped_short_is_made_on_behind_the_requests_as_room_allows() {
        const GIB: u64 = 1 << 30;
        let left = [(7, GIB), (3, GIB), (12, GIB), (30, GIB)];
        // (8 GiB kept free and 1 GiB of K/V to grow: 11.5 GiB available holds two layers, the third would leave 8.5)
        assert_eq!(tier_more_within(&left, 10 * GIB, 11 * GIB + GIB / 2, 8 * GIB, GIB), vec![7, 3]);
        assert_eq!(tier_more_within(&left, GIB, 30 * GIB, 8 * GIB, GIB), vec![7], "the GPUs' view leaves one");
        assert!(tier_more_within(&left, 10 * GIB, 9 * GIB + GIB / 2, 8 * GIB, GIB).is_empty(), "no room for the first: none");
        assert_eq!(tier_more_within(&left, 10 * GIB, 40 * GIB, 8 * GIB, GIB), vec![7, 3, 12, 30], "room for all");
    }

    /// Decides: a tier only where the budget cannot hold every layer; then the most routed experts of all layers, never-routed ones left out,
    /// stopping at the first that does not fit, each layer's list in expert order.
    #[test]
    fn a_tier_takes_the_most_routed_experts_of_every_layer_where_whole_layers_do_not_fit() {
        let counts = [[5u64, 0, 9, 1], [7, 2, 0, 3]];
        let used = |layer: usize, expert: usize| counts[layer][expert];
        assert_eq!(tier_within(&[40, 40], &[10, 10], 4, 80, &used), None, "every layer fits: whole layers");
        assert_eq!(tier_within(&[40, 40], &[10, 10], 4, 30, &used), Some(vec![vec![0, 2], vec![0]]), "9, 7, 5");
        assert_eq!(tier_within(&[40, 40], &[10, 10], 4, 79, &used), Some(vec![vec![0, 2, 3], vec![0, 1, 3]]), "the routed ones only");
        assert_eq!(tier_within(&[40, 40], &[10, 20], 4, 25, &used), Some(vec![vec![2], vec![]]), "7 does not fit beside 9: stop");
        assert_eq!(tier_within(&[40, 40], &[10, 10], 4, 0, &used), Some(vec![vec![], vec![]]));
    }

    /// Decides: the banks are copied where any GPU says it cannot register host memory read-only, or the variable says `0` whatever they say, and are
    /// registered in place otherwise, also where the runtime cannot say; the reason names the GPUs or the variable.
    #[test]
    fn the_banks_are_copied_where_a_gpu_cannot_register_them_read_only_or_the_variable_says_so() {
        assert_eq!(lock_mode(None, &[(0, Some(true)), (1, Some(true))]), (LockMode::Register, None));
        assert_eq!(lock_mode(None, &[(0, None)]), (LockMode::Register, None), "a runtime that cannot say is taken as able");
        assert_eq!(lock_mode(None, &[]), (LockMode::Register, None));
        let (mode, why) = lock_mode(None, &[(0, Some(false)), (1, Some(false))]);
        assert_eq!(mode, LockMode::Copy);
        assert!(why.as_deref().unwrap().starts_with("GPU 0 and GPU 1 do not support registering host memory read-only"), "{why:?}");
        let (mode, why) = lock_mode(None, &[(0, Some(true)), (1, Some(false))]);
        assert_eq!(mode, LockMode::Copy, "one that cannot is enough: the registrations are portable");
        assert!(why.as_deref().unwrap().starts_with("GPU 1 does not support"), "{why:?}");
        // the variable: on a GPU that supports it, and where the runtime cannot say (HIP)
        for devices in [vec![(0, Some(true))], vec![(0, None)], vec![]] {
            let (mode, why) = lock_mode(Some("0"), &devices);
            assert_eq!(mode, LockMode::Copy, "{devices:?}");
            assert!(why.as_deref().unwrap().starts_with("STRAYDOG_HOST_REGISTER_READ_ONLY=0"), "{why:?}");
        }
        assert_eq!(lock_mode(Some(" 0 "), &[(0, Some(true))]).0, LockMode::Copy);
        // (anything else leaves it to the GPUs)
        for value in ["1", "", "copy"] {
            assert_eq!(lock_mode(Some(value), &[(0, Some(true))]), (LockMode::Register, None), "{value:?}");
            assert_eq!(lock_mode(Some(value), &[(0, Some(false))]).0, LockMode::Copy, "{value:?}");
        }
    }

    /// Decides: a layer's turn in the copy mode makes one copy of each bank, equal to the file's bytes (the mapping's) and apart from them, checks the copies
    /// against the file's tree, registers the mapping nowhere, and frees the copies with their last holder.
    #[test]
    fn a_copy_turn_copies_the_banks_whole_and_checks_the_copies() {
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        fn lock(_: usize, _: &[u8]) -> Result<(), String> {
            CALLS.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn unlock(_: &[u8]) {
            CALLS.fetch_add(1, Ordering::SeqCst);
        }
        let fixture = Fixture::new("turn", None);
        let memory = Arc::new(HeapMemory::default());
        let mut locks = BankLocks::new(vec![(7, fixture.banks(true))], 0, 1 << 30).copying(memory.clone()).registering_with(Registrar { lock, unlock });
        assert_eq!(locks.mode(), LockMode::Copy);
        // SAFETY: the fixture keeps the mapping alive.
        let mut locked = unsafe { locks.lock_next() }.expect("a layer").unwrap_or_else(|stop| panic!("{}", stop.message()));
        assert!(locked.is_copy());
        assert_eq!((locked.layer, locked.bytes), (7, SPANS.iter().map(|span| span.1).sum::<u64>()));
        let copy = locked.take_copy().expect("the copy");
        assert!(!locked.is_copy(), "the copy is the model's now");
        assert_eq!(copy.layer(), 7);
        for (role, mapped) in fixture.mapping().iter().enumerate() {
            assert_eq!(copy.bank(role), *mapped, "bank {role}'s bytes");
            assert_ne!(copy.bank(role).as_ptr(), mapped.as_ptr(), "a copy, not the mapping");
        }
        assert_eq!(copy.bytes(), locked.bytes);
        assert_eq!((memory.allocated(), memory.live(), memory.freed()), (3, 3, 0), "one allocation a bank");
        assert_eq!(CALLS.load(Ordering::SeqCst), 0, "the mapping is registered nowhere");
        // SAFETY: the fixture keeps the mapping alive; a layer whose copy was taken gives nothing up.
        unsafe { locked.unlock() };
        assert_eq!((CALLS.load(Ordering::SeqCst), memory.live()), (0, 3));
        // SAFETY: as above.
        assert!(unsafe { locks.lock_next() }.is_none(), "the plan had one layer");
        drop(copy);
        assert_eq!((memory.live(), memory.freed()), (0, 3), "freed with the last holder");
        drop(locks);
        fixture.finish();
    }

    /// Decides: a file whose bytes do not match its tree, anywhere in a bank, is `Unverified` once copied, frees every copy the turn made, and is
    /// not a refusal that the model remembers; the same bytes untampered pass.
    #[test]
    fn a_file_that_does_not_match_the_tree_is_unverified_and_its_copies_are_freed() {
        for (label, tamper) in [("a", 1500), ("b", 2_500_000), ("c", SIZE - 10), ("d", SPANS[2].0 as usize)] {
            let fixture = Fixture::new(&format!("tampered-{label}"), Some(tamper));
            let memory = Arc::new(HeapMemory::default());
            let refusal = Refusal::default();
            let mut locks = BankLocks::new(vec![(7, fixture.banks(true))], 0, 1 << 30).copying(memory.clone()).remembering(refusal.clone());
            // SAFETY: the fixture keeps the mapping alive.
            let stop = unsafe { locks.lock_next() }.expect("a layer").err().expect("a layer that does not verify");
            assert!(matches!(stop, Stop::Unverified { layer: 7, .. }), "{stop:?} for a byte changed at {tamper}");
            assert!(stop.message().starts_with("Straydog model bytes do not verify: layer 7's banks"), "{}", stop.message());
            assert_eq!((memory.allocated(), memory.live(), memory.freed()), (3, 0, 3), "the copies are freed");
            assert_eq!(refusal.blocks_more(0, 0), None, "a failed check is not a refusal");
            drop(locks);
            fixture.finish();
        }
    }

    /// Decides: the check is of the copy's own bytes: a byte of a copy changed after it was made fails, and restored passes, though the mapping it came
    /// from is intact; a bank of a file without a tree is not checked.
    #[test]
    fn a_copy_changed_after_it_was_made_does_not_pass_the_check() {
        let fixture = Fixture::new("changed", None);
        let memory: Arc<dyn HostMemory> = Arc::new(HeapMemory::default());
        let banks = fixture.banks(true);
        let copy = LayerCopy::make(&memory, 7, &banks).unwrap();
        assert_eq!(copy.verify(&banks, Priority::Normal), Ok(true));
        for role in 0..3 {
            let part = &copy.0.parts[role];
            // (the bank's first byte, its last, and a byte of the blocks around it that the bank does not use)
            for at in [part.at, part.at + part.len - 1, 0] {
                // SAFETY: inside the allocation, which nothing reads meanwhile.
                unsafe { *part.base.add(at) ^= 1 };
                assert!(copy.verify(&banks, Priority::Normal).is_err(), "bank {role}, byte {at} of its blocks changed");
                // SAFETY: as above.
                unsafe { *part.base.add(at) ^= 1 };
                assert_eq!(copy.verify(&banks, Priority::Normal), Ok(true), "restored");
            }
        }
        assert_eq!(copy.verify(&fixture.banks(false), Priority::Normal), Ok(false), "no tree: nothing to check against");
        drop(copy);
        drop(banks);
        fixture.finish();
    }

    /// Decides: a tier holds the chosen experts' bytes as the file has them, checks its own bytes (a byte changed after it was made fails, and
    /// restored passes), points the others at nothing, and is freed with its last holder.
    #[test]
    fn a_tier_holds_its_experts_as_the_file_has_them_and_checks_them() {
        let fixture = Fixture::new("tier", None);
        let memory = Arc::new(HeapMemory::default());
        let held: Arc<dyn HostMemory> = memory.clone();
        let banks = fixture.banks(true);
        const COUNT: usize = 7;
        let tier = ExpertTier::make(&held, 7, &banks, COUNT, &[5, 0, 3]).unwrap();
        assert_eq!(tier.verify(&banks, Priority::Normal), Ok(true));
        let pointers = tier.pointers(COUNT);
        for expert in 0..COUNT {
            for (role, bank) in banks.iter().enumerate() {
                let per = (bank.bytes / COUNT as u64) as usize;
                if [5, 0, 3].contains(&expert) {
                    // SAFETY: the tier holds `per` bytes there; the mapping's bank holds the expert at that stride.
                    let (tiered, mapped) = unsafe { (std::slice::from_raw_parts(pointers[expert][role], per),
                                                     std::slice::from_raw_parts(bank.map.add(bank.offset as usize + expert * per), per)) };
                    assert!(tiered == mapped, "expert {expert}, role {role}");
                } else {
                    assert!(pointers[expert][role].is_null(), "expert {expert} is not tiered");
                }
            }
        }
        let (_, regions) = tier.0.experts[1];
        // SAFETY: the expert's first byte, inside the allocation, which nothing reads meanwhile.
        unsafe { *tier.0.base.add(regions[2].offset + regions[2].at) ^= 1 };
        assert!(tier.verify(&banks, Priority::Normal).is_err(), "a changed byte fails");
        // SAFETY: as above.
        unsafe { *tier.0.base.add(regions[2].offset + regions[2].at) ^= 1 };
        assert_eq!(tier.verify(&banks, Priority::Normal), Ok(true), "restored");
        assert!(ExpertTier::make(&held, 7, &banks, COUNT, &[COUNT]).is_err(), "an expert past the layer's");
        drop(tier);
        assert_eq!((memory.allocated(), memory.live()), (1, 0), "one allocation, freed");
        drop(banks);
        fixture.finish();
    }

    /// Decides: a copy that cannot be made frees the banks already copied and ends the turn as a refusal; the model remembers a refusal that came with nothing
    /// locked (so that no plan is made again after each request), and does not remember one that came after layers were taken.
    #[test]
    fn a_refused_copy_frees_what_was_made_and_the_first_refusal_is_remembered() {
        let fixture = Fixture::new("refused", None);
        // the second bank's allocation is refused
        let memory = HeapMemory::refusing_from(2);
        let refusal = Refusal::default();
        let mut locks = BankLocks::new(vec![(7, fixture.banks(true)), (6, fixture.banks(true))], 0, 1 << 30)
            .copying(memory.clone()).remembering(refusal.clone());
        assert_eq!(refusal.blocks_more(0, 0), None);
        // SAFETY: the fixture keeps the mapping alive.
        let stop = unsafe { locks.lock_next() }.expect("a layer").err().expect("refused");
        assert_eq!(stop, Stop::NotCopied { layer: 7, error: "out of memory".into() });
        assert!(stop.is_refusal());
        assert_eq!(stop.message(), "copying layer 7's banks into page-locked memory failed (out of memory)");
        assert_eq!((memory.allocated(), memory.live(), memory.freed()), (1, 0, 1), "the first bank's copy is freed");
        let blocked = refusal.blocks_more(0, 0).expect("nothing was locked: remembered");
        assert!(blocked.starts_with(&stop.message()) && blocked.contains("planned again once"), "{blocked}");
        assert_eq!(refusal.blocks_more(3, 0), None, "layers locked meanwhile: the memory may be there next time");
        drop(locks);
        // a refusal after a layer was taken is not remembered: the first layer takes allocations 1 to 3, the second's first is the 4th
        let memory = HeapMemory::refusing_from(4);
        let refusal = Refusal::default();
        let mut locks = BankLocks::new(vec![(7, fixture.banks(true)), (6, fixture.banks(true))], 0, 1 << 30)
            .copying(memory.clone()).remembering(refusal.clone());
        // SAFETY: as above.
        let first = unsafe { locks.lock_next() }.expect("a layer").unwrap_or_else(|stop| panic!("{}", stop.message()));
        let stop = unsafe { locks.lock_next() }.expect("a layer").err().expect("refused");
        assert!(matches!(stop, Stop::NotCopied { layer: 6, .. }));
        assert_eq!(refusal.blocks_more(0, 0), None, "a layer was taken before it");
        drop((first, locks));
        assert_eq!(memory.live(), 0);
        fixture.finish();
    }

    /// Decides: a register turn registers a layer's next bank beside the check of the one before, and a check that fails gives up every bank
    /// registered so far, the one registered beside it among them (a byte changed in the first bank: the second was registered meanwhile).
    #[test]
    fn a_failed_check_gives_up_the_bank_registered_beside_it() {
        static RECORD: Mutex<Vec<(&str, usize, usize)>> = Mutex::new(Vec::new());
        fn lock(_: usize, range: &[u8]) -> Result<(), String> {
            RECORD.lock().unwrap().push(("lock", range.as_ptr() as usize, range.len()));
            Ok(())
        }
        fn unlock(range: &[u8]) {
            RECORD.lock().unwrap().push(("unlock", range.as_ptr() as usize, range.len()));
        }
        let fixture = Fixture::new("register-tampered", Some(SPANS[0].0 as usize + 10));
        let base = fixture.map().as_ptr() as usize;
        let ranges: Vec<(usize, usize)> = SPANS.iter().map(|&(offset, bytes)| (base + offset as usize, bytes as usize)).collect();
        let mut locks = BankLocks::new(vec![(7, fixture.banks(true))], 0, 1 << 30).registering_with(Registrar { lock, unlock });
        // SAFETY: the fixture keeps the mapping alive.
        let stop = unsafe { locks.lock_next() }.expect("a layer").err().expect("the first bank does not match its tree");
        assert!(matches!(stop, Stop::Unverified { layer: 7, .. }), "{stop:?}");
        let record = RECORD.lock().unwrap().clone();
        assert_eq!(record.iter().filter(|entry| entry.0 == "lock").map(|entry| (entry.1, entry.2)).collect::<Vec<_>>(), ranges[..2].to_vec(),
                   "the first two banks registered, the third not begun");
        assert_eq!(record.iter().filter(|entry| entry.0 == "unlock").map(|entry| (entry.1, entry.2)).collect::<Vec<_>>(), ranges[..2].to_vec(),
                   "both given up");
        drop(locks);
        fixture.finish();
    }

    /// Decides: where registration is supported the plan is what it was: it registers the three ranges of the mapping, makes no copy, gives the
    /// registrations up for a layer it is told to let go, and a refusal gives up those it made and is remembered.
    #[test]
    fn the_register_mode_registers_the_mapping_and_copies_nothing() {
        static RECORD: Mutex<Vec<(&str, usize, usize)>> = Mutex::new(Vec::new());
        static REFUSE_AT: AtomicUsize = AtomicUsize::new(usize::MAX);
        fn lock(_: usize, range: &[u8]) -> Result<(), String> {
            let mut record = RECORD.lock().unwrap();
            if record.iter().filter(|entry| entry.0 == "lock").count() == REFUSE_AT.load(Ordering::SeqCst) {
                return Err("not supported".into());
            }
            record.push(("lock", range.as_ptr() as usize, range.len()));
            Ok(())
        }
        fn unlock(range: &[u8]) {
            RECORD.lock().unwrap().push(("unlock", range.as_ptr() as usize, range.len()));
        }
        let fixture = Fixture::new("register", None);
        let base = fixture.map().as_ptr() as usize;
        let ranges: Vec<(usize, usize)> = SPANS.iter().map(|&(offset, bytes)| (base + offset as usize, bytes as usize)).collect();
        let mut locks = BankLocks::new(vec![(7, fixture.banks(true)), (6, fixture.banks(true))], 0, 1 << 30).registering_with(Registrar { lock, unlock });
        assert_eq!(locks.mode(), LockMode::Register);
        // SAFETY: the fixture keeps the mapping alive.
        let locked = unsafe { locks.lock_next() }.expect("a layer").unwrap_or_else(|stop| panic!("{}", stop.message()));
        assert!(!locked.is_copy());
        assert_eq!((locked.layer, locked.bytes), (7, SPANS.iter().map(|span| span.1).sum::<u64>()));
        let locks_made: Vec<(&str, usize, usize)> = ranges.iter().map(|&(address, bytes)| ("lock", address, bytes)).collect();
        assert_eq!(*RECORD.lock().unwrap(), locks_made, "the mapping's three ranges, in the order of the banks");
        // SAFETY: the fixture keeps the mapping alive.
        unsafe { locked.unlock() };
        let released: Vec<(&str, usize, usize)> = ranges.iter().map(|&(address, bytes)| ("unlock", address, bytes)).collect();
        assert_eq!(RECORD.lock().unwrap()[3..], released[..], "and given up for a layer the model will not take");
        // the next layer is refused at its second bank: the first is given up, the stop is a refusal, remembered
        RECORD.lock().unwrap().clear();
        REFUSE_AT.store(1, Ordering::SeqCst);
        drop(locks);
        let refusal = Refusal::default();
        let mut locks = BankLocks::new(vec![(6, fixture.banks(true))], 0, 1 << 30).remembering(refusal.clone()).registering_with(Registrar { lock, unlock });
        // SAFETY: as above.
        let stop = unsafe { locks.lock_next() }.expect("a layer").err().expect("refused");
        assert_eq!(stop, Stop::Refused { layer: 6, error: "not supported".into() });
        assert_eq!(*RECORD.lock().unwrap(), vec![("lock", ranges[0].0, ranges[0].1), ("unlock", ranges[0].0, ranges[0].1)]);
        assert!(refusal.blocks_more(0, 0).is_some_and(|blocked| blocked.starts_with(&stop.message())));
        drop(locks);
        fixture.finish();
    }

    /// Decides: two devices that both read a layer read ONE copy of it (the copy is made once and every plan is moved to it), a layer is not copied a
    /// second time, and a layer no device reads takes no copy.
    #[test]
    fn two_devices_that_read_a_layer_read_one_copy_of_it() {
        let fixture = Fixture::new("shared", None);
        let memory = Arc::new(HeapMemory::default());
        let mapping = pointers(fixture.mapping());
        let mut devices = Devices { plans: vec![vec![mapping], vec![mapping, [1 as *const u8; 3]]], fail: Failing::No, last: None };
        let mut copies = Copies::default();
        let copy = copied(&fixture, &memory);
        let at = copy.pointers();
        copies.take_over(&mut devices, fixture.mapping(), copy).unwrap();
        assert_eq!(devices.plans, vec![vec![at], vec![at, [1 as *const u8; 3]]], "both devices read the same copy; another layer's plan is left alone");
        assert_eq!((memory.allocated(), memory.live()), (3, 3), "one copy for both");
        // the plans get each bank's start and the start of the allocation it lies in: the banks begin inside the blocks of the file the copy holds, and
        // the runtime maps an allocation by its start, not an address inside it
        let to = devices.last.expect("the move");
        assert_eq!(to.start, at);
        for (role, &(offset, _)) in SPANS.iter().enumerate() {
            assert_eq!(to.start[role] as usize - to.base[role] as usize, offset as usize % verity::BLOCK, "bank {role}'s place in its allocation");
        }
        assert_ne!(to.start[0], to.base[0], "a bank that does not begin on a block");
        assert_eq!(copies.held(), (1, SPANS.iter().map(|span| span.1).sum::<u64>()));
        assert!(copies.holds(7) && !copies.holds(6));
        // a second copy of the layer is refused, and freed
        let again = copied(&fixture, &memory);
        assert_eq!(memory.live(), 6);
        let error = copies.take_over(&mut devices, fixture.mapping(), again).unwrap_err();
        assert!(error.contains("copied already"), "{error}");
        assert_eq!((memory.live(), memory.freed()), (3, 3), "the second copy is freed");
        assert_eq!(devices.plans[0][0], at, "and moved nothing");
        // a layer that no device reads
        let mut nobody = Devices { plans: vec![vec![[2 as *const u8; 3]]], fail: Failing::No, last: None };
        let mut other = Copies::default();
        let error = other.take_over(&mut nobody, fixture.mapping(), copied(&fixture, &memory)).unwrap_err();
        assert!(error.contains("no device reads layer 7"), "{error}");
        assert_eq!((memory.live(), other.held().0), (3, 0), "freed, not held");
        drop(copies);
        assert_eq!(memory.live(), 0, "the copy goes with the model");
        fixture.finish();
    }

    /// Decides: a give-back points the layer's plans back at the mapping first and then lets the copy go: freed at once where nothing reads it, and not
    /// while a reader that holds a second reference to it (the tests' stand-in for a reader outside the engine's lock; in production the readers are
    /// waited for) reads it intact to the end; after it, the plans are on the mapping, and the give-back's last move was to the mapping itself.
    #[test]
    fn a_give_back_frees_the_copy_only_after_the_last_reader_and_points_the_plans_back() {
        let fixture = Fixture::new("giveback", None);
        let memory = Arc::new(HeapMemory::default());
        let mapping = pointers(fixture.mapping());
        let mut devices = Devices { plans: vec![vec![mapping]], fail: Failing::No, last: None };
        let mut copies = Copies::default();
        // no reader: freed at once
        copies.take_over(&mut devices, fixture.mapping(), copied(&fixture, &memory)).unwrap();
        assert_eq!(copies.give_back(&mut devices, 6, fixture.mapping()), Ok(false), "a layer that is not held");
        assert_eq!(copies.give_back(&mut devices, 7, fixture.mapping()), Ok(true));
        assert_eq!(devices.plans, vec![vec![mapping]], "back on the mapping");
        assert_eq!((memory.live(), memory.freed()), (0, 3));
        assert!(!copies.holds(7));
        // a reader on another thread reads the copy while the give-back runs
        copies.take_over(&mut devices, fixture.mapping(), copied(&fixture, &memory)).unwrap();
        // (a second reference to the memory: what stands for a reader outside the engine's lock, which production has none of)
        let lease = LayerCopy(copies.get(7).unwrap().0.clone());
        assert_eq!(Arc::strong_count(&lease.0), 2);
        let expected: Vec<Vec<u8>> = fixture.mapping().iter().map(|bank| bank.to_vec()).collect();
        let stop = Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::SeqCst) {
                    for (role, bytes) in expected.iter().enumerate() {
                        assert!(lease.bank(role) == &bytes[..], "the copy changed under a reader (bank {role})");
                    }
                    reads += 1;
                }
                reads
            })
        };
        assert_eq!(copies.give_back(&mut devices, 7, fixture.mapping()), Ok(true));
        assert_eq!(devices.plans, vec![vec![mapping]], "the plans are back on the mapping");
        assert_eq!(devices.last, Some(Banks::of_mapping(fixture.mapping())), "and read it from the mapping's own start");
        assert!(!copies.holds(7));
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!((memory.live(), memory.freed()), (3, 3), "the reader still holds the copy: not freed (the first layer's three are)");
        stop.store(true, Ordering::SeqCst);
        assert!(reader.join().unwrap() > 0);
        assert_eq!((memory.live(), memory.freed()), (0, 6), "freed when the reader let go");
        fixture.finish();
    }

    /// Decides: a move that fails frees only what no plan can read. A failed give-back keeps the copy held, whether the plans are back on it or one may
    /// still read it; a take-over that failed with every plan back on the mapping frees the copy at once (its memory is not held until the model ends), and
    /// one that failed with a plan that may read the copy keeps it until the model ends.
    #[test]
    fn a_failed_move_frees_the_copy_only_where_no_plan_can_read_it() {
        let fixture = Fixture::new("failed", None);
        let memory = Arc::new(HeapMemory::default());
        let mapping = pointers(fixture.mapping());
        let mut devices = Devices { plans: vec![vec![mapping]], fail: Failing::No, last: None };
        let mut copies = Copies::default();
        copies.take_over(&mut devices, fixture.mapping(), copied(&fixture, &memory)).unwrap();
        let at = copies.get(7).unwrap().pointers();
        // a give-back that fails keeps the copy, both ways
        for fail in [Failing::Restored, Failing::Stuck] {
            devices.fail = fail;
            assert!(copies.give_back(&mut devices, 7, fixture.mapping()).is_err());
            assert!(copies.holds(7));
            assert_eq!((memory.live(), memory.freed()), (3, 0), "kept");
            devices.plans = vec![vec![at]];
        }
        devices.fail = Failing::No;
        devices.plans = vec![vec![mapping]];
        // a take-over that fails with every plan put back: nothing reads the copy, and it is freed now
        devices.fail = Failing::Restored;
        let mut other = Copies::default();
        let error = other.take_over(&mut devices, fixture.mapping(), copied(&fixture, &memory)).unwrap_err();
        assert!(error.contains("would not move"), "{error}");
        assert_eq!(devices.plans, vec![vec![mapping]], "the plans are where they were");
        assert_eq!((other.held().0, memory.live(), memory.freed()), (0, 3, 3), "not held, and freed at once");
        assert!(other.stranded.is_empty());
        // a take-over that fails with a plan that may read the copy: kept until the model ends
        devices.fail = Failing::Stuck;
        assert!(other.take_over(&mut devices, fixture.mapping(), copied(&fixture, &memory)).is_err());
        assert_ne!(devices.plans, vec![vec![mapping]], "a plan was left on the new pointers");
        assert_eq!((other.held().0, other.stranded.len(), memory.live(), memory.freed()), (0, 1, 6, 3), "not held, and not freed");
        drop(other);
        assert_eq!((memory.live(), memory.freed()), (3, 6), "freed with the model");
        drop(copies);
        assert_eq!(memory.live(), 0);
        fixture.finish();
    }

    /// Decides: a copy is read from the file and never from the mapping: banks whose mapping holds other bytes than the file still make copies of the file's
    /// bytes, which pass the check against the file's tree, and a plan in the copy mode has no read-ahead of the mapping's pages.
    #[test]
    fn a_copy_is_read_from_the_file_not_from_the_mapping() {
        let fixture = Fixture::new("file", None);
        // a mapping of the same length that is not the file's bytes
        let other = vec![0xEEu8; fixture.map().len()];
        let file = fixture.file.clone().unwrap();
        let banks = SPANS.map(|(offset, bytes)| Bank::new(other.as_ptr(), other.len(), offset, bytes, fixture.verifier.clone()).reading(file.clone()));
        let memory = Arc::new(HeapMemory::default());
        let mut locks = BankLocks::new(vec![(7, banks)], 0, 1 << 30).copying(memory.clone());
        assert!(locks.read_ahead().is_empty(), "the mapping's pages are not read ahead where the copy does not read them");
        // SAFETY: the fixture keeps the file alive, and the mapping is not read.
        let mut locked = unsafe { locks.lock_next() }.expect("a layer").unwrap_or_else(|stop| panic!("{}", stop.message()));
        let copy = locked.take_copy().expect("the copy");
        for (role, mapped) in fixture.mapping().iter().enumerate() {
            assert_eq!(copy.bank(role), *mapped, "bank {role} is the file's bytes");
        }
        drop((copy, locked, locks));
        assert_eq!(memory.live(), 0);
        fixture.finish();
    }

    /// Decides: a plan in the register mode reads ahead the pages of its banks as it did, and one in the copy mode none of them.
    #[test]
    fn a_plan_reads_the_mappings_pages_ahead_in_the_register_mode_only() {
        let fixture = Fixture::new("ahead", None);
        let layers = || vec![(7, fixture.banks(true)), (6, fixture.banks(true))];
        let registering = BankLocks::new(layers(), 0, 1 << 30);
        assert_eq!(registering.read_ahead().len(), 6);
        let copying = BankLocks::new(layers(), 0, 1 << 30).copying(Arc::new(HeapMemory::default()));
        assert!(copying.read_ahead().is_empty());
        assert_eq!(copying.remaining(), 2, "its layers are still to lock");
        drop((registering, copying));
        fixture.finish();
    }

    /// Decides: banks that have no file to read are refused as a copy that cannot be made, before anything is allocated: a copy is never made from the mapping.
    #[test]
    fn a_copy_turn_of_banks_without_their_file_is_refused_and_allocates_nothing() {
        let fixture = Fixture::new("nofile", None);
        let banks = SPANS.map(|(offset, bytes)| Bank::new(fixture.map().as_ptr(), fixture.map().len(), offset, bytes, fixture.verifier.clone()));
        let memory = Arc::new(HeapMemory::default());
        let mut locks = BankLocks::new(vec![(7, banks)], 0, 1 << 30).copying(memory.clone());
        // SAFETY: the fixture keeps the mapping alive.
        let stop = unsafe { locks.lock_next() }.expect("a layer").err().expect("refused");
        assert!(matches!(&stop, Stop::NotCopied { layer: 7, error } if error.contains("no file to copy from")), "{stop:?}");
        assert_eq!((memory.allocated(), memory.live()), (0, 0));
        drop(locks);
        fixture.finish();
    }

    /// Host memory whose allocations begin in the middle of a block, which a direct read cannot fill.
    struct Misaligned {
        heap: HeapMemory,
        /// Each allocation's address, and the heap's it lies in.
        starts: Mutex<BTreeMap<usize, usize>>,
    }

    impl HostMemory for Misaligned {
        fn allocate(&self, bytes: u64) -> Result<*mut u8, String> {
            let base = self.heap.allocate(bytes + ALIGN as u64)?;
            // SAFETY: inside the allocation, which is a block longer than asked.
            let start = unsafe { base.add(512) };
            self.starts.lock().unwrap().insert(start as usize, base as usize);
            Ok(start)
        }

        unsafe fn free(&self, pointer: *mut u8) {
            let base = self.starts.lock().unwrap().remove(&(pointer as usize)).expect("freed once");
            // SAFETY: the heap's allocation that `pointer` lies in.
            unsafe { self.heap.free(base as *mut u8) };
        }
    }

    /// Decides: memory that is not aligned for a direct read is filled through the page cache, with the same bytes and the same check.
    #[test]
    fn memory_not_aligned_for_a_direct_read_is_filled_through_the_page_cache() {
        let fixture = Fixture::new("misaligned", None);
        let memory = Arc::new(Misaligned { heap: HeapMemory::default(), starts: Mutex::default() });
        let mut locks = BankLocks::new(vec![(7, fixture.banks(true))], 0, 1 << 30).copying(memory.clone());
        // SAFETY: the fixture keeps the mapping alive.
        let mut locked = unsafe { locks.lock_next() }.expect("a layer").unwrap_or_else(|stop| panic!("{}", stop.message()));
        let copy = locked.take_copy().expect("the copy");
        for (role, mapped) in fixture.mapping().iter().enumerate() {
            assert_eq!(copy.bank(role), *mapped, "bank {role}");
            assert!((copy.0.parts[role].base as usize) % ALIGN != 0, "the allocation is not aligned");
        }
        drop((copy, locked, locks));
        assert_eq!(memory.heap.live(), 0);
        fixture.finish();
    }

    /// Decides: a copy's read goes the way a row's does: the whole range direct, then the range in direct pieces of a quarter if the working-set quota refused
    /// that, then what is left through the page cache if it refused a piece; a read that is not a quota refusal ends it at once, whichever way it came on.
    #[test]
    fn a_copys_read_tries_the_whole_range_then_pieces_then_the_page_cache_on_a_quota_refusal_alone() {
        const AT: u64 = ALIGN as u64;
        let refused = |error: &std::io::Error| error.raw_os_error() == Some(1453);
        let quota = || std::io::Error::from_raw_os_error(1453);
        let bytes = 8 * ALIGN + 100;
        let source: Vec<u8> = (0..bytes).map(|at| (at * 7 % 251) as u8).collect();
        // `fail` is given a call's number, its way, its offset and its bytes, and says what error the call returns, if any
        let run = |direct: bool, bytes: usize, fail: &dyn Fn(usize, Way, u64, usize) -> Option<std::io::Error>| {
            let mut target = vec![0u8; bytes.next_multiple_of(ALIGN)];
            let mut calls: Vec<(Way, u64, usize)> = Vec::new();
            let result = read_in_ways(direct, AT, &mut target, bytes, refused, &mut |way, at, span, len| {
                let number = calls.len();
                calls.push((way, at, len));
                if let Some(error) = fail(number, way, at, len) {
                    return Err(error);
                }
                let from = (at - AT) as usize;
                span[..len].copy_from_slice(&source[from..from + len]);
                Ok(())
            });
            target.truncate(bytes);
            (result, calls, target)
        };
        // (a piece is a quarter of the range, up to whole blocks: three blocks, and the last piece what is left)
        let piece = 3 * ALIGN;
        let pieces = [(Way::Direct, AT, piece), (Way::Direct, AT + piece as u64, piece), (Way::Direct, AT + 2 * piece as u64, bytes - 2 * piece)];
        // the whole range, direct
        let (result, calls, target) = run(true, bytes, &|_, _, _, _| None);
        assert!(result.is_ok());
        assert_eq!(calls, vec![(Way::Direct, AT, bytes)]);
        assert_eq!(target, source);
        // refused: the pieces
        let (result, calls, target) = run(true, bytes, &|number, _, _, _| (number == 0).then(quota));
        assert!(result.is_ok());
        assert_eq!(calls, [vec![(Way::Direct, AT, bytes)], pieces.to_vec()].concat());
        assert_eq!(target, source);
        // a piece refused: what is left once through the page cache, from where the pieces got to
        let (result, calls, target) = run(true, bytes, &|number, _, _, _| (number == 0 || number == 2).then(quota));
        assert!(result.is_ok());
        assert_eq!(calls, vec![(Way::Direct, AT, bytes), pieces[0], pieces[1], (Way::Cached, AT + piece as u64, bytes - piece)]);
        assert_eq!(target, source);
        // the first piece refused: all of it through the page cache
        let (result, calls, target) = run(true, bytes, &|number, _, _, _| (number <= 1).then(quota));
        assert!(result.is_ok());
        assert_eq!(calls, vec![(Way::Direct, AT, bytes), pieces[0], (Way::Cached, AT, bytes)]);
        assert_eq!(target, source);
        // a range that is not more than a block has no smaller way
        let (result, calls, target) = run(true, 100, &|number, _, _, _| (number == 0).then(quota));
        assert!(result.is_ok());
        assert_eq!(calls, vec![(Way::Direct, AT, 100), (Way::Cached, AT, 100)]);
        assert_eq!(target, source[..100]);
        // no direct handle, or memory not aligned for it: the page cache alone
        let (result, calls, target) = run(false, bytes, &|_, _, _, _| None);
        assert!(result.is_ok());
        assert_eq!(calls, vec![(Way::Cached, AT, bytes)]);
        assert_eq!(target, source);
        // any other error ends it where it is, and none is tried again through the page cache
        let other = || std::io::Error::from_raw_os_error(5);
        let (result, calls, _) = run(true, bytes, &|_, _, _, _| Some(other()));
        assert_eq!((result.unwrap_err().raw_os_error(), calls.len()), (Some(5), 1));
        let (result, calls, _) = run(true, bytes, &|number, _, _, _| Some(if number == 0 { quota() } else { other() }));
        assert_eq!((result.unwrap_err().raw_os_error(), calls.len()), (Some(5), 2), "an error of a piece");
        // the page cache's own refusal is the last word
        let (result, calls, _) = run(true, bytes, &|number, way, _, _| (number <= 1 || way == Way::Cached).then(quota));
        assert_eq!(result.unwrap_err().raw_os_error(), Some(1453));
        assert_eq!(calls.last().map(|call| call.0), Some(Way::Cached));
    }

    /// Decides: a refusal that came with nothing locked blocks the next plans until the memory available has grown by the refused layer's banks since it
    /// came, and then it is forgotten; layers locked meanwhile end it; one that came after a layer was taken is not kept.
    #[test]
    fn a_remembered_refusal_ends_when_the_memory_available_has_grown_by_a_layer() {
        const GIB: u64 = 1 << 30;
        let refusal = Refusal::default();
        let stop = Stop::NotCopied { layer: 7, error: "out of memory".into() };
        assert_eq!(refusal.blocks_more(0, 0), None, "nothing refused yet");
        // a layer after one was taken: not kept
        refusal.note(2, &stop, 10 * GIB, GIB);
        assert_eq!(refusal.blocks_more(0, 10 * GIB), None);
        // the first layer, 1 GiB, with 10 GiB available
        refusal.note(0, &stop, 10 * GIB, GIB);
        for available in [0, 10 * GIB, 10 * GIB + GIB - 1] {
            let why = refusal.blocks_more(0, available).unwrap_or_else(|| panic!("{available} available is not a layer's banks more"));
            assert!(why.starts_with("copying layer 7's banks into page-locked memory failed (out of memory); planned again once 11.0 GiB"), "{why}");
        }
        // layers locked meanwhile: nothing blocks, and the refusal stays until memory has grown
        assert_eq!(refusal.blocks_more(1, 0), None);
        assert!(refusal.blocks_more(0, 10 * GIB).is_some(), "still kept");
        // exactly a layer more: planned again, and the refusal is forgotten
        assert_eq!(refusal.blocks_more(0, 11 * GIB), None);
        assert_eq!(refusal.blocks_more(0, 0), None, "forgotten: the next refusal is its own");
        // a refusal again: kept anew, from the memory then
        refusal.note(0, &stop, 3 * GIB, GIB / 2);
        assert!(refusal.blocks_more(0, 3 * GIB + GIB / 2 - 1).is_some());
        assert_eq!(refusal.blocks_more(0, 3 * GIB + GIB / 2), None);
    }
}

#[cfg(test)]
mod registered_tests {
    use super::Registered;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// Decides: a model's marks are its own: a layer one model registered is not another's (a second model in the process, or the model made
    /// again), and a new model has none; a layer is marked until it is released, and one past the mask never is.
    #[test]
    fn a_models_marks_are_its_own_and_last_until_released() {
        let (one, other) = (Arc::new(Registered::default()), Arc::new(Registered::default()));
        one.mark(5);
        assert!(one.is_registered(5) && !other.is_registered(5));
        assert!(one.read(5).is_some() && other.read(5).is_none());
        one.release(5);
        assert!(!one.is_registered(5) && one.read(5).is_none());
        one.mark(70);
        assert!(!one.is_registered(70) && one.read(70).is_none());
        assert!(!Arc::new(Registered::default()).is_registered(5));
    }

    /// Decides: giving a layer's locks up waits for the reads of its banks under way, and no read begins once it has started; the reads of
    /// other layers are not waited for.
    #[test]
    fn giving_the_locks_up_waits_for_the_reads_under_way() {
        let registered = Arc::new(Registered::default());
        registered.mark(3);
        registered.mark(4);
        let read = registered.read(3).unwrap();
        let other = registered.read(4).unwrap();
        let released = Arc::new(AtomicBool::new(false));
        let releasing = {
            let (registered, released) = (registered.clone(), released.clone());
            std::thread::spawn(move || {
                registered.release(3);
                released.store(true, Ordering::SeqCst);
            })
        };
        std::thread::sleep(Duration::from_millis(100));
        assert!(!released.load(Ordering::SeqCst), "the release waits for the read");
        assert!(registered.read(3).is_none(), "no read begins once the release has");
        drop(read);
        releasing.join().unwrap();
        assert!(released.load(Ordering::SeqCst));
        assert!(registered.is_registered(4));
        drop(other);
        registered.release(4);
        assert!(!registered.is_registered(4));
    }
}
