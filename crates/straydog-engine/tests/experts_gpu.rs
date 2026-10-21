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

fn run_with(layers: Vec<Option<Vec<Role>>>, slots: usize, host_budget_slots: u64, steps: usize) -> straydog_engine::experts::Counts {
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
    let layout = Arc::new(Layout::new(&[file], layers.clone(), Some(verifier)).unwrap());
    let host_budget = host_budget_slots * (STRIDES.iter().sum::<u64>() + 4 * 4096).next_multiple_of(4096);
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut store = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(4)), slots, host_budget).unwrap();
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
        for (&expert, slot) in experts.iter().zip(&slots) {
            let want = expected(layers[layer].as_ref().unwrap(), expert);
            let mut got = vec![0u8; want.len()];
            gpu.read_buffer(slot, 0, &mut got).unwrap();
            assert!(got == want, "step {step}: layer {layer} expert {expert} differs from the file");
        }
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
