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
    let mut base = 1234u64;
    let mut layer = || {
        Some(STRIDES.iter().map(|&stride| {
            let role = Role { file: 0, base, stride };
            base += stride * EXPERTS as u64 + 777;
            role
        }).collect::<Vec<_>>())
    };
    let first = layer();
    let third = layer();
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
    let layers = roles();
    let end = layers.iter().flatten().flatten().map(|role| role.base + role.stride * EXPERTS as u64).max().unwrap();
    let path = std::env::temp_dir().join(format!("straydog-experts-{}-{host_budget_slots}", std::process::id()));
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
    let mut store = DeviceExperts::new(&mut gpu, layout, Arc::new(Reader::new(4)), 16, host_budget).unwrap();
    let mut random = Random(0x1234_5678);
    for step in 0..steps {
        let layer = if step % 2 == 0 { 0 } else { 2 };
        let mut experts: Vec<usize> = (0..1 + random.below(6)).map(|_| random.below(EXPERTS)).collect();
        experts.sort_unstable();
        experts.dedup();
        if step % 3 == 0 {
            let next: Vec<usize> = (0..3).map(|_| random.below(EXPERTS)).collect();
            store.prefetch(&mut gpu, 2 - layer, &next).unwrap();
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
