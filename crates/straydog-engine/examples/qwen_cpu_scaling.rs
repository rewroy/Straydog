//! How the CPU expert engine scales on Qwen3.8's real experts: microseconds a route by threads, with the experts cold (random among
//! four layers' banks, well past the last-level cache) and hot (the same few again and again). Where hot scales and cold does not,
//! memory bandwidth is the limit; where neither does, the scheduling is.
//!
//! STRAYDOG_MODEL=<IQ2_XS directory> cargo run --release --features hip --example qwen_cpu_scaling
//! under the bench lock.

use std::fs::File;
use std::path::PathBuf;

use straydog_gpu::ffi;

const HIDDEN: usize = 2560;
const FFN: usize = 640;
const EXPERTS: usize = 512;
const ROUTES: usize = 2;
const CALLS: usize = 400;

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn main() {
    let root = PathBuf::from(std::env::var_os("STRAYDOG_MODEL").expect("set STRAYDOG_MODEL"));
    let mut paths: Vec<PathBuf> = std::fs::read_dir(root).unwrap().map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "gguf")).collect();
    paths.sort();
    let mut files = Vec::new();
    let mut headers = Vec::new();
    for path in paths {
        let mut file = File::open(path).unwrap();
        headers.push(straydog_gguf::inspect(&mut file).unwrap());
        files.push(file);
    }
    straydog_engine::gpu::load_iq_tables().unwrap();
    let find = |name: &str| headers.iter().enumerate()
        .find_map(|(file, header)| header.tensors.iter().find(|tensor| tensor.name == name).map(|tensor| (file, tensor))).unwrap();
    // Four layers of the commonest format.
    let kind_of = |layer: usize| find(&format!("blk.{layer}.ffn_gate_exps.weight")).1.quant_type;
    let mut counts = std::collections::BTreeMap::new();
    (0..48).for_each(|layer| *counts.entry(kind_of(layer)).or_insert(0) += 1);
    let commonest = counts.into_iter().max_by_key(|(_, count)| *count).unwrap().0;
    // (STRAYDOG_GATE_KIND picks another format's layers, gate and up both of it)
    let wanted = std::env::var("STRAYDOG_GATE_KIND").ok().and_then(|kind| kind.parse().ok()).unwrap_or(commonest);
    let up_of = |layer: usize| find(&format!("blk.{layer}.ffn_up_exps.weight")).1.quant_type;
    let layers: Vec<usize> = (0..48).filter(|layer| kind_of(*layer) == wanted && up_of(*layer) == wanted).take(4).collect();
    // Mapped as a served model's banks are, read through once so the pages are in memory: page size and page walks count.
    // SAFETY: read-only mappings of files nothing changes during this run.
    let maps: Vec<memmap2::Mmap> = files.iter().map(|file| unsafe { memmap2::Mmap::map(file) }.unwrap()).collect();
    let mut banks: Vec<[&[u8]; 3]> = Vec::new();
    let mut kinds = [0i32; 3];
    for layer in &layers {
        banks.push(["gate", "up", "down"].map(|role| {
            let (file, tensor) = find(&format!("blk.{layer}.ffn_{role}_exps.weight"));
            let bytes = &maps[file][tensor.offset as usize..(tensor.offset + tensor.bytes) as usize];
            std::hint::black_box(bytes.iter().step_by(4096).map(|byte| *byte as u64).sum::<u64>());
            bytes
        }));
        for (index, role) in ["gate", "up", "down"].iter().enumerate() {
            kinds[index] = find(&format!("blk.{layer}.ffn_{role}_exps.weight")).1.quant_type as i32;
        }
    }
    let per: [usize; 3] = [0, 1, 2].map(|role| banks[0][role].len() / EXPERTS);
    let cores = std::thread::available_parallelism().map_or(4, |count| count.get());
    println!("kinds {kinds:?}, an expert {} KiB, {} logical CPUs, {} routes a call", (per[0] + per[1] + per[2]) / 1024, cores, ROUTES);
    println!("{:>7} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8}", "threads", "mode", "us/route", "gate+up", "product", "down", "GB/s");
    let mut random = Random(0x5eed);
    let mut sizes: Vec<u32> = [1u32, 2, 3, 4, 6, 8, 12, 16].into_iter().filter(|threads| *threads as usize <= cores).collect();
    sizes.push(0);   // the engine's own choice: its physical cores
    for threads in sizes {
        // SAFETY: a plain value; null on failure.
        let engine = unsafe { ffi::sd_cpu_experts_create(threads, 0) };
        assert!(!engine.is_null());
        for hot in [false, true] {
            let x: Vec<f32> = (0..HIDDEN).map(|_| ((random.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0).collect();
            let mut out = vec![0f32; HIDDEN];
            let weights = [0.5f32; ROUTES];
            let (mut before, mut after) = ([0u64; 7], [0u64; 7]);
            let mut wall = std::time::Duration::ZERO;
            for call in 0..CALLS + 8 {
                if call == 8 {
                    // SAFETY: seven counters.
                    unsafe { ffi::sd_cpu_experts_stats(engine, before.as_mut_ptr()) };
                }
                let (bank, first) = if hot { (0, 0) } else { ((random.next() % 4) as usize, (random.next() % (EXPERTS - 1) as u64) as usize) };
                let slices = |role: usize| -> [*const u8; ROUTES] {
                    [banks[bank][role][first * per[role]..].as_ptr(), banks[bank][role][(first + 1) * per[role]..].as_ptr()]
                };
                let (gate, up, down) = (slices(0), slices(1), slices(2));
                let started = std::time::Instant::now();
                // SAFETY: ROUTES expert slices of the declared shapes.
                let status = unsafe {
                    ffi::sd_cpu_experts_run(engine, kinds.as_ptr(), gate.as_ptr(), up.as_ptr(), down.as_ptr(), weights.as_ptr(),
                                            ROUTES as u32, HIDDEN as u64, FFN as u64, x.as_ptr(), out.as_mut_ptr())
                };
                assert_eq!(status, 0);
                if call >= 8 {
                    wall += started.elapsed();
                }
            }
            // SAFETY: seven counters.
            unsafe { ffi::sd_cpu_experts_stats(engine, after.as_mut_ptr()) };
            let routes = (CALLS * ROUTES) as f64;
            let phase = |index: usize| (after[index] - before[index]) as f64 / 1e3 / routes;
            let us = wall.as_secs_f64() * 1e6 / routes;
            println!("{threads:>7} {:>6} {us:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>8.1}", if hot { "hot" } else { "cold" }, phase(2), phase(3), phase(4),
                     (per[0] + per[1] + per[2]) as f64 / us / 1e3);
        }
        // SAFETY: created above, destroyed once.
        unsafe { ffi::sd_cpu_experts_destroy(engine) };
    }
}
