//! The expert store (src/experts.rs) on a GPU: experts of a generated file at unaligned offsets,
//! through few VRAM slots and a small host tier, so that drive reads, host hits of both layouts,
//! write-backs, prefetches and staging-only operation all occur; every acquired slot is read back
//! and compared with the file's bytes, and every drive read with the file.
//!
//! cargo test --release -p straydog-engine --features hip --test experts_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use straydog_engine::experts::{DeviceExperts, Layout, Reader, Role, Verifier};
use straydog_engine::gpu::Backend;

/// The file's byte at `offset`.
fn byte(offset: u64) -> u8 {
    let mut x = offset.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 29;
    (x.wrapping_mul(0xbf58_476d_1ce4_e5b9) >> 56) as u8
}

const EXPERTS: usize = 40;
const STRIDES: [u64; 3] = [70_001, 70_001, 50_003];

/// Layers 0 and 2 have experts (layer 1 none), each role's tensor at an unaligned base.
fn roles() -> Vec<Option<Vec<Role>>> {
    sized_roles(STRIDES, STRIDES)
}

/// As [`roles`], layer 2's experts of other sizes (the store keeps a class of slots per size).
fn sized_roles(first: [u64; 3], third: [u64; 3]) -> Vec<Option<Vec<Role>>> {
    let mut base = 1234u64;
    let mut layer = |strides: [u64; 3]| {
        Some(strides.iter().map(|&stride| {
            let role = Role { file: 0, base, stride, bytes: stride };
            base += stride * EXPERTS as u64 + 777;
            role
        }).collect::<Vec<_>>())
    };
    let first = layer(first);
    let third = layer(third);
    vec![first, None, third]
}

fn expected(roles: &[Role], expert: usize) -> Vec<u8> {
    roles.iter().flat_map(|role| {
        let start = role.base + expert as u64 * role.stride;
        (start..start + role.stride).map(byte)
    }).collect()
}

struct Random(u64);
impl Random {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

fn run(host_budget_slots: u64, steps: usize) -> straydog_engine::experts::Counts {
    run_with(roles(), 16, host_budget_slots, steps)
}

/// A store over a generated file of `layers`' experts: its file (removed by the caller), the
/// GPU, the store, and the count of verified drive reads.
fn open(layers: &[Option<Vec<Role>>], slots: usize, host_budget_slots: u64) -> (std::path::PathBuf, Backend, DeviceExperts, Arc<AtomicU64>) {
    let end = layers.iter().flatten().flatten().map(|role| role.base + role.stride * EXPERTS as u64).max().unwrap();
    // A file of its own per run: tests run in parallel, and some share a capacity.
    static RUNS: AtomicU64 = AtomicU64::new(0);
    let run = RUNS.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("straydog-experts-{}-run{run}", std::process::id()));
    {
        let mut file = File::create(&path).unwrap();
        let bytes: Vec<u8> = (0..end + 333).map(byte).collect();
        file.write_all(&bytes).unwrap();
    }
    let file = File::open(&path).unwrap();
    let reads = Arc::new(AtomicU64::new(0));
    let counted = reads.clone();
    let verifier: Verifier = Arc::new(move |file, offset, data: &[u8]| {
        assert_eq!(file, 0);
        assert!(offset.is_multiple_of(4096) && data.len().is_multiple_of(4096), "a read of partial blocks");
        for (index, &value) in data.iter().enumerate() {
            let at = offset + index as u64;
            if at < end + 333 && value != byte(at) {
                return Err(format!("byte {at} read wrong"));
            }
        }
        counted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
    let layout = Arc::new(Layout::new(&[file], layers.to_vec(), Some(verifier)).unwrap());
    let host_budget = host_budget_slots * (STRIDES.iter().sum::<u64>() + 4 * 4096).next_multiple_of(4096);
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let store = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(4)), slots, host_budget).unwrap();
    (path, gpu, store, reads)
}

/// Read every slot back and compare it with the file's bytes of its expert.
fn check_slots(gpu: &mut Backend, roles: &[Role], experts: &[usize], slots: &[straydog_engine::ops::Buffer], what: &str) {
    for (&expert, slot) in experts.iter().zip(slots) {
        let want = expected(roles, expert);
        let mut got = vec![0u8; want.len()];
        gpu.read_buffer(slot, 0, &mut got).unwrap();
        assert!(got == want, "{what}: expert {expert} differs from the file");
    }
}

fn run_with(layers: Vec<Option<Vec<Role>>>, slots: usize, host_budget_slots: u64, steps: usize) -> straydog_engine::experts::Counts {
    let (path, mut gpu, mut store, reads) = open(&layers, slots, host_budget_slots);
    let mut random = Random(0x1234_5678);
    for step in 0..steps {
        let layer = if step % 2 == 0 { 0 } else { 2 };
        let mut experts: Vec<usize> = (0..1 + random.below(6)).map(|_| random.below(EXPERTS)).collect();
        experts.sort_unstable();
        experts.dedup();
        if step % 3 == 0 {
            let next: Vec<usize> = (0..3).map(|_| random.below(EXPERTS)).collect();
            store.prefetch(&mut gpu, 2 - layer, &next, true).unwrap();
        }
        let slots = store.acquire(&mut gpu, layer, &experts).unwrap();
        check_slots(&mut gpu, layers[layer].as_ref().unwrap(), &experts, &slots, &format!("step {step}: layer {layer}"));
        store.release(&mut gpu).unwrap();
    }
    store.settle(&mut gpu).unwrap();
    let counts = store.counts();
    drop(store);
    std::fs::remove_file(&path).unwrap();
    assert!(reads.load(Ordering::Relaxed) > 0);
    counts
}

#[test]
fn experts_through_a_host_tier() {
    let counts = run(30, 400);
    eprintln!("{counts:?}");
    assert!(counts.drive_reads > 0 && counts.host_hits > 0 && counts.writebacks > 0 && counts.prefetched > 0);
}

/// A release's write-backs wait for the compute stream to pass the release; a fill that takes a
/// slot so freed (the free slots having run out) waits for the write-back before its upload, or
/// it would overwrite the expert being written back. Here the compute stream is kept busy past
/// the release, so the write-backs start well after the next fills are queued.
#[test]
fn a_fill_waits_for_a_write_back_behind_the_pass() {
    let layers = roles();
    let roles = layers[0].clone().unwrap();
    // 16 slots and a reserve of 4; a host tier for every expert
    let (path, mut gpu, mut store, _) = open(&layers, 16, EXPERTS as u64);
    let first: Vec<usize> = (0..12).collect();
    let slots = store.acquire(&mut gpu, 0, &first).unwrap();
    check_slots(&mut gpu, &roles, &first, &slots, "the first twelve");
    store.release(&mut gpu).unwrap();
    // Work on the compute stream (a few ms of GEMVs over a 64 MB matrix), then four more experts:
    // they take the four free slots, and their release writes the four oldest back behind it.
    let (rows, cols) = (4096u64, 4096u64);
    let matrix = gpu.buffer(rows * cols * 4).unwrap();
    gpu.write_buffer(&matrix, 0, &vec![0.001f32; (rows * cols) as usize]).unwrap();
    let (input, output, scratch) = (gpu.buffer(cols * 4).unwrap(), gpu.buffer(rows * 4).unwrap(), gpu.buffer(1 << 20).unwrap());
    for _ in 0..200 {
        gpu.linear(&matrix, 0, rows, cols, 0, input.base(), 0, 1, output.base(), 0, &scratch).unwrap();
    }
    let second: Vec<usize> = (12..16).collect();
    store.acquire(&mut gpu, 0, &second).unwrap();
    store.release(&mut gpu).unwrap();
    // No free slot is left but those four: these fills wait for their write-backs.
    let third: Vec<usize> = (16..20).collect();
    let slots = store.acquire(&mut gpu, 0, &third).unwrap();
    check_slots(&mut gpu, &roles, &third, &slots, "fills of slots being written back");
    store.release(&mut gpu).unwrap();
    // The four written back come from the host tier as they were.
    let back: Vec<usize> = (0..4).collect();
    let slots = store.acquire(&mut gpu, 0, &back).unwrap();
    check_slots(&mut gpu, &roles, &back, &slots, "experts written back");
    store.release(&mut gpu).unwrap();
    store.settle(&mut gpu).unwrap();
    let counts = store.counts();
    assert!(counts.writebacks >= 4 && counts.host_hits >= 4, "{counts:?}");
    drop(store);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn experts_with_staging_only() {
    let counts = run(0, 200);
    eprintln!("{counts:?}");
    assert!(counts.drive_reads > 0 && counts.host_hits == 0 && counts.writebacks == 0);
}

/// A read that fails (here: a block that does not verify) puts nothing in either tier and fails
/// only a pass holding its expert: a failed prefetch no pass took fails nothing (counted), one a
/// pass took fails that pass, as does the pass's own read; the expert is read again on its next
/// use, and the store stays usable.
#[test]
fn a_failed_read_leaves_both_tiers() {
    let layers = roles();
    let end = layers.iter().flatten().flatten().map(|role| role.base + role.stride * EXPERTS as u64).max().unwrap();
    let path = std::env::temp_dir().join(format!("straydog-experts-{}-failing", std::process::id()));
    std::fs::write(&path, (0..end + 333).map(byte).collect::<Vec<u8>>()).unwrap();
    let role = layers[0].as_ref().unwrap()[1];
    // A block in the middle of expert 5's up rows (an edge block is shared with a neighbour).
    let bad = (role.base + 5 * role.stride + role.stride / 2) & !4095;
    let failures = Arc::new(AtomicU64::new(0));
    let left = failures.clone();
    // A failing read takes 100 ms: a pass issued right after a prefetch finds it still in flight.
    let verifier: Verifier = Arc::new(move |_, offset, data: &[u8]| {
        let covers = (offset..offset + data.len() as u64).contains(&bad);
        if covers && left.load(Ordering::Relaxed) > 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            left.fetch_sub(1, Ordering::Relaxed);
            return Err(format!("block {bad} does not verify"));
        }
        Ok(())
    });
    let layout = Arc::new(Layout::new(&[File::open(&path).unwrap()], layers.clone(), Some(verifier)).unwrap());
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut store = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(4)), 16, 1 << 26).unwrap();
    // A prefetch whose read fails, taken by no pass.
    failures.store(1, Ordering::Relaxed);
    store.prefetch(&mut gpu, 0, &[5], true).unwrap();
    store.settle(&mut gpu).unwrap();
    assert_eq!(store.counts().failed_unwaited, 1);
    // A prefetch a pass takes before its read fails, then the pass's own read of it.
    failures.store(1, Ordering::Relaxed);
    store.prefetch(&mut gpu, 0, &[5], true).unwrap();
    assert!(store.acquire(&mut gpu, 0, &[4, 5, 6]).unwrap_err().contains("does not verify"), "the taken prefetch");
    store.release(&mut gpu).unwrap();
    failures.store(1, Ordering::Relaxed);
    assert!(store.acquire(&mut gpu, 0, &[4, 5, 6]).unwrap_err().contains("does not verify"), "the pass's own read");
    store.release(&mut gpu).unwrap();
    let slots = store.acquire(&mut gpu, 0, &[4, 5, 6]).unwrap();
    for (expert, slot) in [4, 5, 6].into_iter().zip(&slots) {
        let want = expected(layers[0].as_ref().unwrap(), expert);
        let mut got = vec![0u8; want.len()];
        gpu.read_buffer(slot, 0, &mut got).unwrap();
        assert!(got == want, "expert {expert} differs after the failed reads");
    }
    store.release(&mut gpu).unwrap();
    store.settle(&mut gpu).unwrap();
    assert_eq!(store.counts().failed_unwaited, 1);
    drop(store);
    std::fs::remove_file(&path).unwrap();
}

/// Experts of two sizes (a class of slots each): every acquired slot still matches the file,
/// through drive reads, host hits, write-backs and prefetches in both classes.
#[test]
fn experts_of_two_sizes() {
    let counts = run_with(sized_roles(STRIDES, [90_001, 30_011, 120_007]), 16, 30, 400);
    eprintln!("{counts:?}");
    assert!(counts.drive_reads > 0 && counts.host_hits > 0 && counts.writebacks > 0 && counts.prefetched > 0);
}

/// From the fewest slots a store takes to more than every expert, with no host tier, a host tier
/// of one slot, and one that holds every expert: each pass gets the file's bytes (checked in
/// `run_with`), and with room for every expert in either tier an expert leaves the drive once
/// (again only when a prefetch of it was cancelled before its read landed).
#[test]
fn experts_at_several_capacities() {
    for (slots, host) in [(16, 0), (16, 1), (16, 7), (40, 7), (40, 100), (100, 0), (100, 100)] {
        let counts = run_with(roles(), slots, host, 300);
        eprintln!("{slots} slots, {host} host slots: {counts:?}");
        let experts = 2 * EXPERTS as u64;
        if slots >= 2 * EXPERTS + 4 || host >= 2 * EXPERTS as u64 {
            assert!(counts.drive_reads <= experts + counts.prefetch_unused, "{slots}/{host}: {counts:?}");
        }
        if slots >= 2 * EXPERTS + 4 {
            assert_eq!(counts.writebacks, 0, "{slots}/{host}: every expert fits in VRAM");
        }
        if host == 0 {
            assert_eq!(counts.host_hits + counts.writebacks, 0, "{slots}/{host}: no host tier");
        }
    }
}

/// A verifier that panics fails its read like a refused block: the pass gets an error, the
/// store stays usable.
#[test]
fn a_panicking_verifier_fails_its_read() {
    let layers = roles();
    let end = layers.iter().flatten().flatten().map(|role| role.base + role.stride * EXPERTS as u64).max().unwrap();
    let path = std::env::temp_dir().join(format!("straydog-experts-{}-panicking", std::process::id()));
    std::fs::write(&path, (0..end + 333).map(byte).collect::<Vec<u8>>()).unwrap();
    let role = layers[0].as_ref().unwrap()[0];
    let bad = (role.base + 7 * role.stride + role.stride / 2) & !4095;
    let left = Arc::new(AtomicU64::new(1));
    let remaining = left.clone();
    let verifier: Verifier = Arc::new(move |_, offset, data: &[u8]| {
        if (offset..offset + data.len() as u64).contains(&bad) && remaining.fetch_sub(1, Ordering::Relaxed) == 1 {
            panic!("a verifier bug");
        }
        Ok(())
    });
    let layout = Arc::new(Layout::new(&[File::open(&path).unwrap()], layers.clone(), Some(verifier)).unwrap());
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut store = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(4)), 16, 1 << 26).unwrap();
    assert!(store.acquire(&mut gpu, 0, &[7]).unwrap_err().contains("panicked"));
    store.release(&mut gpu).unwrap();
    let slots = store.acquire(&mut gpu, 0, &[7, 8]).unwrap();
    for (expert, slot) in [7, 8].into_iter().zip(&slots) {
        let want = expected(layers[0].as_ref().unwrap(), expert);
        let mut got = vec![0u8; want.len()];
        gpu.read_buffer(slot, 0, &mut got).unwrap();
        assert!(got == want, "expert {expert} differs after the panicked read");
    }
    store.release(&mut gpu).unwrap();
    store.settle(&mut gpu).unwrap();
    drop(store);
    std::fs::remove_file(&path).unwrap();
}

/// A prefetch an earlier pass of its layer cancelled, taken by a later pass before its read
/// lands: the pass still gets the expert (read again), with the file's bytes.
#[test]
fn a_cancelled_prefetch_taken_before_it_lands() {
    let layers = roles();
    let end = layers.iter().flatten().flatten().map(|role| role.base + role.stride * EXPERTS as u64).max().unwrap();
    let path = std::env::temp_dir().join(format!("straydog-experts-{}-cancelled", std::process::id()));
    std::fs::write(&path, (0..end + 333).map(byte).collect::<Vec<u8>>()).unwrap();
    // Slow reads on one reader thread: the prefetch's later chunks are still queued when its
    // layer's next pass cancels it.
    let verifier: Verifier = Arc::new(|_, _, _: &[u8]| {
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(())
    });
    let layout = Arc::new(Layout::new(&[File::open(&path).unwrap()], layers.clone(), Some(verifier)).unwrap());
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut store = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(1)), 16, 1 << 26).unwrap();
    // Expert 3 resident first, so the pass that cancels the prefetch waits for no read of its own.
    drop(store.acquire(&mut gpu, 0, &[3]).unwrap());
    store.release(&mut gpu).unwrap();
    store.settle(&mut gpu).unwrap();
    store.prefetch(&mut gpu, 0, &[9], true).unwrap();
    // This pass of layer 0 does not take expert 9: its read is cancelled midway.
    drop(store.acquire(&mut gpu, 0, &[3]).unwrap());
    store.release(&mut gpu).unwrap();
    // The next pass takes it at once, before the cancelled read lands.
    let slots = store.acquire(&mut gpu, 0, &[9, 3]).unwrap();
    for (expert, slot) in [9, 3].into_iter().zip(&slots) {
        let want = expected(layers[0].as_ref().unwrap(), expert);
        let mut got = vec![0u8; want.len()];
        gpu.read_buffer(slot, 0, &mut got).unwrap();
        assert!(got == want, "expert {expert} differs after the cancelled prefetch");
    }
    store.release(&mut gpu).unwrap();
    assert!(store.counts().prefetch_unused >= 1);
    store.settle(&mut gpu).unwrap();
    drop(store);
    std::fs::remove_file(&path).unwrap();
}

/// Page-locks a host buffer until dropped, so a panic still unregisters it.
struct Pinned {
    pointer: *mut u8,
}

impl Pinned {
    fn new(buffer: &mut straydog_engine::storage::Buffer) -> Pinned {
        let pointer = buffer.as_mut_slice().as_mut_ptr();
        // SAFETY: `buffer` outlives the guard, and the guard unpins before `buffer` is dropped.
        unsafe { straydog_engine::transfer::pin_host(pointer, buffer.capacity()).unwrap() };
        Pinned { pointer }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: pinned in `new`. Copies into the range have been waited for by the time it drops,
        // or the test is unwinding and the device is about to go with it.
        unsafe { straydog_engine::transfer::unpin_host(self.pointer).unwrap() };
    }
}

/// A pipelined decode step's use of the store: experts the table does not show found on the host
/// (either layout) or read there from the drive, VRAM experts copied out, experts moved between the
/// tiers (promote, demote); every byte the host or a slot holds is the file's.
#[test]
fn experts_moved_for_a_decode_table() {
    use straydog_engine::experts::HostSource;
    use straydog_engine::transfer::Stream;
    let layers = roles();
    let end = layers.iter().flatten().flatten().map(|role| role.base + role.stride * EXPERTS as u64).max().unwrap();
    let path = std::env::temp_dir().join(format!("straydog-experts-{}-decode", std::process::id()));
    std::fs::write(&path, (0..end + 333).map(byte).collect::<Vec<u8>>()).unwrap();
    let layout = Arc::new(Layout::new(&[File::open(&path).unwrap()], layers.clone(), None).unwrap());
    let host_budget = 24 * (STRIDES.iter().sum::<u64>() + 4 * 4096).next_multiple_of(4096);
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut store = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(4)), 16, host_budget).unwrap();
    let slot_bytes = store.slot_size(0).unwrap() as usize;
    let mut copy = straydog_engine::storage::Buffer::new(slot_bytes);
    let _pinned = Pinned::new(&mut copy);
    let mut random = Random(0x5eed_1234);
    let (mut from_host, mut from_drive, mut from_vram, mut promoted, mut demoted) = (0, 0, 0, 0, 0);
    for step in 0..400 {
        let layer = if step % 2 == 0 { 0 } else { 2 };
        let roles = layers[layer].as_ref().unwrap();
        let expert = random.below(EXPERTS);
        let want = expected(roles, expert);
        let generation = store.vram_generation();
        if store.vram_ready(&mut gpu, layer, expert).unwrap().is_some() || store.in_vram(layer, expert) {
            // SAFETY: page-locked, of a slot's bytes.
            assert!(unsafe { store.copy_out(&mut gpu, layer, expert, copy.as_mut_slice().as_mut_ptr()).unwrap() });
            assert!(copy.as_slice()[..want.len()] == want[..], "step {step}: a VRAM copy differs from the file");
            from_vram += 1;
            // Now and then it goes back to the host tier, its slot reused only after this point.
            if random.below(3) == 0 {
                let after = gpu.mark(Stream::Compute).unwrap();
                assert!(store.demote(&mut gpu, layer, expert, after).unwrap());
                assert!(store.vram_generation() != generation);
                demoted += 1;
            }
            continue;
        }
        let mut source = store.host_source(&mut gpu, layer, expert).unwrap();
        if matches!(source, HostSource::Absent) {
            store.fetch_host(&mut gpu, layer, expert, random.below(2) == 0).unwrap();
            from_drive += 1;
        } else {
            from_host += 1;
        }
        while matches!(source, HostSource::Absent | HostSource::Reading) {
            if matches!(source, HostSource::Reading) {
                store.wait_reads(&mut gpu).unwrap();
            }
            source = store.host_source(&mut gpu, layer, expert).unwrap();
        }
        let HostSource::Ready(pointers) = source else { panic!("step {step}: neither tier nor VRAM holds the expert") };
        let mut at = 0;
        for (role, pointer) in roles.iter().zip(pointers) {
            // SAFETY: the role's bytes in the host tier, untouched until the store's next change.
            let got = unsafe { std::slice::from_raw_parts(pointer, role.bytes as usize) };
            assert!(got == &want[at..at + role.bytes as usize], "step {step}: a host copy differs from the file");
            at += role.bytes as usize;
        }
        // Now and then it moves into a free slot (once its upload lands, the slot holds it).
        if random.below(2) == 0 {
            if let Some((_, mark)) = store.promote(&mut gpu, layer, expert).unwrap() {
                assert!(gpu.mark_done(mark, true).unwrap());
                assert!(store.vram_ready(&mut gpu, layer, expert).unwrap().is_some());
                assert!(!store.in_host(layer, expert));
                promoted += 1;
            }
        }
    }
    store.settle(&mut gpu).unwrap();
    drop(store);
    std::fs::remove_file(&path).unwrap();
    assert!(from_host > 0 && from_drive > 0 && from_vram > 0 && promoted > 0 && demoted > 0,
            "host {from_host}, drive {from_drive}, VRAM {from_vram}, promoted {promoted}, demoted {demoted}");
}

/// A region lent to the store ([`DeviceExperts::lend`]) holds experts like the store's own slots;
/// the lender takes the region's start back between passes (the experts there go to the host
/// tier, the compute stream waiting for their copies) and overwrites it on the compute stream;
/// given back, its slots fill after that work. Every expert reads back as the file has it.
#[test]
fn a_lent_region_holds_experts_and_goes_back_to_its_lender() {
    let layers = roles();
    let roles = layers[0].clone().unwrap();
    // 16 slots and a reserve of 4; a host tier for every expert
    let (path, mut gpu, mut store, _) = open(&layers, 16, EXPERTS as u64);
    let slot_bytes = store.slot_size(0).unwrap();
    let region = gpu.buffer(slot_bytes * 12 + 1000).unwrap();
    assert_eq!(store.lend(region).unwrap(), 12);
    assert_eq!(store.lent_slots(), (12, 12));
    // 24 experts: more than the store's own slots hold beside their reserve
    for batch in [(0..8).collect::<Vec<usize>>(), (8..16).collect(), (16..24).collect()] {
        let slots = store.acquire(&mut gpu, 0, &batch).unwrap();
        check_slots(&mut gpu, &roles, &batch, &slots, "into own and lent slots");
        store.release(&mut gpu).unwrap();
    }
    // The lender takes back the region's first half (six slots) and overwrites it on the compute
    // stream.
    let half = slot_bytes * 6;
    store.reclaim(&mut gpu, half).unwrap();
    assert_eq!(store.lent_slots(), (6, 12));
    let floats = half / 4;
    let (gate, up) = (gpu.buffer(floats * 4).unwrap(), gpu.buffer(floats * 4).unwrap());
    gpu.write_buffer(&gate, 0, &vec![1.5f32; floats as usize]).unwrap();
    gpu.write_buffer(&up, 0, &vec![2.0f32; floats as usize]).unwrap();
    for _ in 0..20 {
        gpu.swiglu(gate.base(), up.base(), 1, floats, 0.0, None, region.base()).unwrap();
    }
    // Every expert, wherever it went, reads back intact (those taken back from the host tier).
    for batch in [(0..8).collect::<Vec<usize>>(), (8..16).collect(), (16..24).collect()] {
        let slots = store.acquire(&mut gpu, 0, &batch).unwrap();
        check_slots(&mut gpu, &roles, &batch, &slots, "after the region's start was taken back");
        for slot in &slots {
            let start = slot.base() as u64;
            assert!(start < region.base() as u64 || start >= region.base() as u64 + half, "an expert in a slot taken back");
        }
        store.release(&mut gpu).unwrap();
    }
    // Given back after more work on its bytes: fills into it wait for that work.
    for _ in 0..20 {
        gpu.swiglu(gate.base(), up.base(), 1, floats, 0.0, None, region.base()).unwrap();
    }
    let after = gpu.mark(straydog_engine::transfer::Stream::Compute).unwrap();
    store.restore(after);
    assert_eq!(store.lent_slots(), (12, 12));
    for batch in [(24..32).collect::<Vec<usize>>(), (32..40).collect(), (0..8).collect()] {
        let slots = store.acquire(&mut gpu, 0, &batch).unwrap();
        check_slots(&mut gpu, &roles, &batch, &slots, "after the region came back");
        store.release(&mut gpu).unwrap();
    }
    store.settle(&mut gpu).unwrap();
    let counts = store.counts();
    assert!(counts.writebacks > 0 && counts.host_hits > 0, "{counts:?}");
    drop(store);
    std::fs::remove_file(&path).unwrap();
}

/// Fills taken by [`DeviceExperts::acquire_gate_up`] read back whole once [`await_rest`] waited for
/// their downs, and a release before that wait fails.
#[test]
fn a_fill_taken_at_its_gate_and_up_is_whole_after_the_rest() {
    let layers = roles();
    let roles = layers[0].clone().unwrap();
    let (path, mut gpu, mut store, _) = open(&layers, 16, EXPERTS as u64);
    let mut random = Random(0x9876_5432);
    for step in 0..60 {
        let mut experts: Vec<usize> = (0..1 + random.below(6)).map(|_| random.below(EXPERTS)).collect();
        experts.sort_unstable();
        experts.dedup();
        let slots = store.acquire_gate_up(&mut gpu, 0, &experts).unwrap();
        if step == 0 {
            assert!(store.release(&mut gpu).is_err(), "a release before the downs were waited for");
        }
        store.await_rest(&mut gpu).unwrap();
        check_slots(&mut gpu, &roles, &experts, &slots, &format!("step {step}"));
        store.release(&mut gpu).unwrap();
    }
    store.settle(&mut gpu).unwrap();
    assert!(store.counts().host_hits > 0, "{:?}", store.counts());
    drop(store);
    std::fs::remove_file(&path).unwrap();
}

/// A request queues every miss's gate and up before any of their downs (so a pass waiting for
/// the gates and ups does not wait behind another expert's down); the slots still hold every
/// byte once the rest is awaited, from the host tier as from the drive.
#[test]
fn a_request_uploads_the_gates_and_ups_before_the_downs() {
    let layers = roles();
    let roles = layers[0].clone().unwrap();
    let (path, mut gpu, mut store, _) = open(&layers, 16, EXPERTS as u64);
    let mut random = Random(0x5eed_1234);
    for step in 0..60 {
        let mut experts: Vec<usize> = (0..2 + random.below(5)).map(|_| random.below(EXPERTS)).collect();
        experts.sort_unstable();
        experts.dedup();
        store.request(&mut gpu, 0, &experts).unwrap();
        let slots = store.acquire_gate_up(&mut gpu, 0, &experts).unwrap();
        store.await_rest(&mut gpu).unwrap();
        check_slots(&mut gpu, &roles, &experts, &slots, &format!("step {step}"));
        store.release(&mut gpu).unwrap();
    }
    store.settle(&mut gpu).unwrap();
    assert!(store.counts().host_hits > 0, "{:?}", store.counts());
    drop(store);
    std::fs::remove_file(&path).unwrap();
}

/// A pass's fill of experts whose roles take several read chunks uploads each chunk as it lands:
/// [`DeviceExperts::acquire_gate_up`] gives the slots once their gate and up are in (read back
/// right then), [`await_rest`] once their downs are too. A chunk that fails fails the pass that
/// waits for it, the down's in await_rest and the gate's or up's in the acquire, and the expert
/// is read again whole on its next use.
#[test]
fn a_streamed_fill_lands_chunk_by_chunk() {
    let layers = sized_roles([600_001, 600_001, 400_003], STRIDES);
    let roles = layers[0].clone().unwrap();
    let end = layers.iter().flatten().flatten().map(|role| role.base + role.stride * EXPERTS as u64).max().unwrap();
    let path = std::env::temp_dir().join(format!("straydog-experts-{}-streamed", std::process::id()));
    std::fs::write(&path, (0..end + 333).map(byte).collect::<Vec<u8>>()).unwrap();
    // The block a failing read covers (none: u64::MAX), and how many more reads of it fail.
    let bad = Arc::new(AtomicU64::new(u64::MAX));
    let failures = Arc::new(AtomicU64::new(0));
    let (block, left) = (bad.clone(), failures.clone());
    // A failing chunk takes 100 ms: the expert's other chunks land long before it.
    let verifier: Verifier = Arc::new(move |_, offset, data: &[u8]| {
        for (index, &value) in data.iter().enumerate() {
            let at = offset + index as u64;
            if at < end + 333 && value != byte(at) {
                return Err(format!("byte {at} read wrong"));
            }
        }
        let covers = (offset..offset + data.len() as u64).contains(&block.load(Ordering::Relaxed));
        if covers && left.load(Ordering::Relaxed) > 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            left.fetch_sub(1, Ordering::Relaxed);
            return Err("a block does not verify".to_string());
        }
        Ok(())
    });
    let layout = Arc::new(Layout::new(&[File::open(&path).unwrap()], layers.clone(), Some(verifier)).unwrap());
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut store = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(4)), 16, 1 << 24).unwrap();
    let head = (roles[0].bytes + roles[1].bytes) as usize;
    let check_heads = |gpu: &mut Backend, experts: &[usize], slots: &[straydog_engine::ops::Buffer], what: &str| {
        for (&expert, slot) in experts.iter().zip(slots) {
            let want = expected(&roles, expert);
            let mut got = vec![0u8; head];
            gpu.read_buffer(slot, 0, &mut got).unwrap();
            assert!(got == want[..head], "{what}: expert {expert}'s gate and up differ from the file");
        }
    };
    let mut random = Random(0x2468_ace0);
    for step in 0..30 {
        let mut experts: Vec<usize> = (0..1 + random.below(5)).map(|_| random.below(EXPERTS)).collect();
        experts.sort_unstable();
        experts.dedup();
        let slots = store.acquire_gate_up(&mut gpu, 0, &experts).unwrap();
        check_heads(&mut gpu, &experts, &slots, &format!("step {step}"));
        store.await_rest(&mut gpu).unwrap();
        check_slots(&mut gpu, &roles, &experts, &slots, &format!("step {step}"));
        store.release(&mut gpu).unwrap();
    }
    // A failing chunk of a down, then of a gate (each in the middle of its rows, of an expert no
    // slot holds now).
    let gone = |store: &DeviceExperts| (0..EXPERTS).find(|&expert| !store.in_vram(0, expert) && !store.in_host(0, expert)).unwrap();
    for (role, what) in [(2, "a down's chunk"), (0, "a gate's chunk")] {
        let expert = gone(&store);
        bad.store((roles[role].base + expert as u64 * roles[role].stride + roles[role].stride / 2) & !4095, Ordering::Relaxed);
        failures.store(1, Ordering::Relaxed);
        if role == 2 {
            let slots = store.acquire_gate_up(&mut gpu, 0, &[expert]).unwrap();
            check_heads(&mut gpu, &[expert], &slots, what);
            assert!(store.await_rest(&mut gpu).unwrap_err().contains("does not verify"), "{what}");
        } else {
            assert!(store.acquire_gate_up(&mut gpu, 0, &[expert]).unwrap_err().contains("does not verify"), "{what}");
        }
        store.release(&mut gpu).unwrap();
        assert!(!store.in_vram(0, expert) && !store.in_host(0, expert), "{what}: the expert stays in neither tier");
        let slots = store.acquire_gate_up(&mut gpu, 0, &[expert]).unwrap();
        store.await_rest(&mut gpu).unwrap();
        check_slots(&mut gpu, &roles, &[expert], &slots, &format!("after {what}"));
        store.release(&mut gpu).unwrap();
    }
    store.settle(&mut gpu).unwrap();
    assert!(store.counts().drive_bytes > 0);
    drop(store);
    std::fs::remove_file(&path).unwrap();
}

/// Bulk uses one read per role here, while decode keeps its 256KiB pieces; both return
/// exactly the same authenticated payload, including every role's streamed head/tail.
#[test]
fn bulk_reads_and_decode_reads_preserve_expert_bytes() {
    let strides = [384 << 10, 384 << 10, 256 << 10];
    let mut base = 0;
    let roles: Vec<Role> = strides.into_iter().map(|stride| {
        let role = Role { file: 0, base, stride, bytes: stride };
        base += stride * EXPERTS as u64;
        role
    }).collect();
    let (path, mut gpu, mut store, reads) = open(&[Some(roles.clone())], 16, 24);
    store.request_bulk(&mut gpu, 0, &[0]).unwrap();
    let bulk = store.acquire(&mut gpu, 0, &[0]).unwrap();
    check_slots(&mut gpu, &roles, &[0], &bulk, "bulk");
    assert_eq!(reads.load(Ordering::Relaxed), 3);
    store.release(&mut gpu).unwrap();
    store.request(&mut gpu, 0, &[1]).unwrap();
    let decode = store.acquire(&mut gpu, 0, &[1]).unwrap();
    check_slots(&mut gpu, &roles, &[1], &decode, "decode");
    assert_eq!(reads.load(Ordering::Relaxed), 8);
    store.release(&mut gpu).unwrap();
    store.settle(&mut gpu).unwrap();
    drop(store);
    std::fs::remove_file(path).unwrap();
}
