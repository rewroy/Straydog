//! A settled expert store gives its host tier back when dropped: the page-locked arenas its
//! pinner made resident are unpinned and freed. Alone in its binary, so that the process's
//! resident size measures this store only.
//!
//! cargo test --release -p straydog-engine --features hip --test experts_drop, under the GPU lock
//! (on Windows, `--features cuda`).
#![cfg(any(feature = "hip", feature = "cuda"))]

use std::fs::File;
use std::sync::Arc;

use straydog_engine::experts::{DeviceExperts, Layout, Reader, Role};
use straydog_engine::gpu::Backend;

const EXPERTS: usize = 40;
const STRIDES: [u64; 3] = [70_001, 70_001, 50_003];

/// Bytes held for the store's arenas.
///
/// Linux reports the resident set. Page-locking puts the arenas into it, and freeing them
/// takes them out, which is what `/proc/self/statm`'s resident field counts.
///
/// Windows reports `PrivateUsage`, the private commit charge. The arenas are private heap
/// blocks (`storage::Buffer`); dropping a settled store frees those blocks, and that charge is
/// the amount released. `WorkingSetSize` also counts file-backed pages, and Windows trims it
/// on its own, so a change there is not the amount this store released.
fn resident() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
        let pages: u64 = statm.split_whitespace().nth(1).unwrap().parse().unwrap();
        // SAFETY: sysconf has no preconditions.
        pages * unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        // SAFETY: the structure's size is set, so the call fills PrivateUsage. The process
        // handle is a pseudo-handle and is not closed.
        let ok = unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters as *mut PROCESS_MEMORY_COUNTERS_EX as *mut PROCESS_MEMORY_COUNTERS,
                counters.cb,
            )
        };
        assert_ne!(ok, 0, "GetProcessMemoryInfo: {}", std::io::Error::last_os_error());
        counters.PrivateUsage as u64
    }
}

/// A store over `layout` with `host_slots` of host tier, through 120 passes that write back and
/// read the host tier, settled; the process's resident bytes with it, then after dropping it.
fn cycle(gpu: &mut Backend, layout: &Arc<Layout>, host_slots: u64) -> (u64, u64) {
    let host_budget = host_slots * (STRIDES.iter().sum::<u64>() + 4 * 4096).next_multiple_of(4096);
    let mut store = DeviceExperts::new(gpu, layout.clone(), Arc::new(Reader::new(4)), 16, host_budget).unwrap();
    for step in 0..120 {
        let experts = [step % EXPERTS, (step * 7 + 3) % EXPERTS];
        drop(store.acquire(gpu, 0, &experts).unwrap());
        store.release(gpu).unwrap();
    }
    store.settle(gpu).unwrap();
    let counts = store.counts();
    // (not writebacks: an upload leaves its copy in the host tier while there is room, and an eviction then writes nothing back)
    assert!(counts.drive_reads > 0 && counts.host_hits > 0, "the host tier was not used: {counts:?}");
    let with = resident();
    drop(store);
    (with, resident())
}

#[test]
fn dropping_a_settled_store_frees_its_host_tier() {
    let mut base = 1234u64;
    let layer: Vec<Role> = STRIDES.iter().map(|&stride| {
        let role = Role { file: 0, base, stride, bytes: stride };
        base += stride * EXPERTS as u64 + 777;
        role
    }).collect();
    let path = std::env::temp_dir().join(format!("straydog-experts-drop-{}", std::process::id()));
    std::fs::write(&path, (0..base + 333).map(|at| (at % 251) as u8).collect::<Vec<u8>>()).unwrap();
    let layout = Arc::new(Layout::new(&[File::open(&path).unwrap()], vec![Some(layer)], None).unwrap());
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    // A small store first: the GPU runtime loads its kernels and copy buffers on first use, and
    // keeps them; what is resident after it is the baseline.
    cycle(&mut gpu, &layout, 30);
    let before = resident();
    // About 400 MiB of host tier: far above what the rest of the process moves.
    let (with, after) = cycle(&mut gpu, &layout, 2000);
    std::fs::remove_file(&path).unwrap();
    let mib = |bytes: u64| bytes as f64 / (1 << 20) as f64;
    eprintln!("resident: {:.0} MiB before, {:.0} MiB with the store, {:.0} MiB after", mib(before), mib(with), mib(after));
    assert!(with >= before + (300 << 20), "the arenas were not resident");
    assert!(after < before + (32 << 20), "the arenas outlived the store");
}
