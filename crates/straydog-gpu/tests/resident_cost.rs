//! One-off microbenchmark of the decode gate and up launch of a partially resident Qwen3.8 layer on CUDA: `resident_expert_launch` as
//! `hybrid_resident_experts` calls it (the generic `resident_expert_gemv`; ten routes of a row, gate and up, and the shared expert's gate and up
//! waves in the one launch), on a plan formed by hand over synthetic bytes of the real sizes, in the engine's slots. It drives `sd_bench_resident`
//! (gpu/kernels.cpp). Not part of any run of the tests:
//!
//!   cargo test --release --features cuda -p straydog-gpu --test resident_cost -- --ignored --nocapture --test-threads=1
//!
//! What it prints: the device and the kernels' registers, shared memory and occupancy; a plain read kernel's GB/s (a reference); then, for two
//! layers of the IQ2_XS bundle (layer 0, routed IQ2_S; layer 1, routed IQ2_XXS) and 1 to 4 rows, the launch's time in four modes (full: routed and the
//! shared expert; routed only; the shared expert only; empty: every route absent, no shared expert) for F, the share of routes that are resident, of
//! 0.71, 1.0 and 0, and O, the share of a row's routes that repeat an expert of the row before it (0.19 of 2 rows, 0.24 of 4: the engine's figures).
//! The medians are of REPS launches after WARMUP, each launch on a different set of experts (a deck of the resident experts, dealt a launch's share at
//! a time, comes round only after far more bytes than the L2 holds), the shared expert's bytes rotating through copies for the same reason.
//! Settings: RESIDENT_COST_DEVICE (the GPU's ordinal), RESIDENT_COST_RESIDENT (how many of the 512 experts are resident, default 384),
//! RESIDENT_COST_ONLY (ExpertGemv::only, 1 the engine's launch beside the staging copy, 0 one launch for all routes; default 1) and
//! RESIDENT_COST_KINDS (gate,up,down GGML kinds of the routed experts and then the shared expert's gate,up,down, for one layer of your own, e.g.
//! 22,22,42,21,23,20).
#![cfg(feature = "cuda")]

use straydog_gpu::{check, device_count, device_info, ffi, Device};

const EXPERTS: usize = 512;
const WARMUP: u32 = 20;
const REPS: u32 = 200;
const SETTLE_REPS: u32 = 3000;
const DEFAULT_RESIDENT: usize = 384;
const FULL: u32 = 0;
const ROUTED: u32 = 1;
const SHARED: u32 = 2;
const EMPTY: u32 = 3;
const ATTRIBUTES: u32 = 4;
const READ: u32 = 5;

/// A layer of the IQ2_XS bundle, from the GGUF's header: gate, up and down of the routed experts, then the shared expert's gate, up and down.
/// Layer 0: IQ2_S (22) gate and up, Q2_0 (42) down; IQ3_S (21) and IQ4_XS (23) shared gate and up, Q4_0 (20) down. Layer 1: IQ2_XXS (16) gate and up
/// (native: no SoA layout), the shared gate Q6_K (14).
const LAYERS: [(&str, [i32; 6]); 2] = [("layer 0", [22, 22, 42, 21, 23, 20]), ("layer 1", [16, 16, 42, 14, 23, 20])];
/// F: the share of routes that are resident.
const RESIDENT_SHARES: [f64; 3] = [0.71, 1.0, 0.0];

/// O by rows: the engine's 0.19 (2 rows) and 0.24 (4), and between them for 3; none for one row.
fn overlap(rows: usize) -> f64 {
    match rows {
        1 => 0.0,
        2 => 0.19,
        3 => 0.215,
        _ => 0.24,
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn shuffle<T>(&mut self, list: &mut [T]) {
        for i in (1..list.len()).rev() {
            let j = self.below(i + 1);
            list.swap(i, j);
        }
    }

    /// `x` rounded down, and up with the probability of its fraction: a mean of `x`.
    fn count(&mut self, x: f64) -> usize {
        x.floor() as usize + usize::from(self.unit() < x.fract())
    }
}

/// Experts dealt one after another from a shuffled order, reshuffled when it runs out.
struct Deck {
    order: Vec<i32>,
    at: usize,
}

impl Deck {
    fn draw(&mut self, rng: &mut Rng, avoid: &[i32]) -> i32 {
        loop {
            if self.at == self.order.len() {
                rng.shuffle(&mut self.order);
                self.at = 0;
            }
            let expert = self.order[self.at];
            self.at += 1;
            if !avoid.contains(&expert) {
                return expert;
            }
        }
    }
}

struct Lists {
    selected: Vec<i32>,
    resident_routes: usize,
    routes: usize,
    overlapped: f64,
    pairs: usize,
}

struct Bench {
    device: Device,
    slot_of: Vec<i32>,
    resident_deck: Deck,
    absent_deck: Deck,
    rng: Rng,
    only: u32,
}

impl Bench {
    /// `resident` of the 512 experts resident, in slots placed at random.
    fn new(device: Device, resident: usize, only: u32) -> Bench {
        assert!((40..=EXPERTS - 40).contains(&resident), "RESIDENT_COST_RESIDENT is between 40 and {}", EXPERTS - 40);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut order: Vec<i32> = (0..EXPERTS as i32).collect();
        rng.shuffle(&mut order);
        let mut slots: Vec<i32> = (0..resident as i32).collect();
        rng.shuffle(&mut slots);
        let mut slot_of = vec![-1i32; EXPERTS];
        for (i, &expert) in order[..resident].iter().enumerate() {
            slot_of[expert as usize] = slots[i];
        }
        let resident_deck = Deck { order: order[..resident].to_vec(), at: 0 };
        let absent_deck = Deck { order: order[resident..].to_vec(), at: 0 };
        Bench { device, slot_of, resident_deck, absent_deck, rng, only }
    }

    /// `reps` launches of `rows` rows of ten routes: each row has about `10 f` resident routes and about `10 o` that repeat an expert of the
    /// row before it (a row's experts are distinct), the rest dealt from the decks, so that a launch's experts are not the last launch's.
    fn lists(&mut self, reps: usize, rows: usize, f: f64, o: f64) -> Lists {
        let mut lists = Lists { selected: Vec::with_capacity(reps * rows * 10), resident_routes: 0, routes: 0, overlapped: 0.0, pairs: 0 };
        for _ in 0..reps {
            let mut previous: Vec<i32> = Vec::new();
            for row in 0..rows {
                let want = self.rng.count(10.0 * f).min(10);
                let mut current: Vec<i32> = Vec::with_capacity(10);
                let (mut resident, mut absent) = (0usize, 0usize);
                if row > 0 {
                    let mut candidates = previous.clone();
                    self.rng.shuffle(&mut candidates);
                    let repeats = self.rng.count(10.0 * o);
                    for expert in candidates {
                        if current.len() >= repeats {
                            break;
                        }
                        if self.slot_of[expert as usize] >= 0 {
                            if resident < want {
                                current.push(expert);
                                resident += 1;
                            }
                        } else if absent < 10 - want {
                            current.push(expert);
                            absent += 1;
                        }
                    }
                }
                let repeated = current.len();
                let mut avoid = previous.clone();
                while resident < want {
                    let expert = self.resident_deck.draw(&mut self.rng, &avoid);
                    avoid.push(expert);
                    current.push(expert);
                    resident += 1;
                }
                while absent < 10 - want {
                    let expert = self.absent_deck.draw(&mut self.rng, &avoid);
                    avoid.push(expert);
                    current.push(expert);
                    absent += 1;
                }
                self.rng.shuffle(&mut current);
                lists.resident_routes += resident;
                lists.routes += 10;
                if row > 0 {
                    lists.overlapped += repeated as f64 / 10.0;
                    lists.pairs += 1;
                }
                lists.selected.extend_from_slice(&current);
                previous = current;
            }
        }
        lists
    }

    fn call(&self, kinds: &[i32; 6], rows: usize, selected: &[i32], mode: u32, warmup: u32, reps: u32) -> [f64; 32] {
        assert_eq!(selected.len(), (warmup + reps) as usize * rows * 10, "the routes of every launch");
        let mut out = [0f64; 32];
        // SAFETY: a live device; `kinds` holds six kinds, `slot_of` one slot an expert, `selected` the routes of `warmup + reps` launches of
        // `rows` rows, `out` the 32 doubles the export writes.
        let status = unsafe {
            ffi::sd_bench_resident(
                self.device.handle(),
                kinds.as_ptr(),
                EXPERTS as u32,
                self.slot_of.as_ptr(),
                rows as u32,
                selected.as_ptr(),
                mode,
                self.only,
                warmup,
                reps,
                out.as_mut_ptr(),
            )
        };
        check(status).unwrap_or_else(|error| panic!("sd_bench_resident mode {mode}, {rows} rows: {error}"));
        out
    }

    /// The device and the kernels (mode 4): no plan, so no routes.
    fn attributes(&self) -> [f64; 32] {
        let mut out = [0f64; 32];
        // SAFETY: a live device; modes 4 and 5 read no plan.
        let status = unsafe {
            ffi::sd_bench_resident(self.device.handle(), std::ptr::null(), 0, std::ptr::null(), 0, std::ptr::null(), ATTRIBUTES, self.only, 0, 1, out.as_mut_ptr())
        };
        check(status).unwrap_or_else(|error| panic!("sd_bench_resident attributes: {error}"));
        out
    }

    /// A plain read kernel over 512 MiB (mode 5).
    fn read(&self) -> [f64; 32] {
        let mut out = [0f64; 32];
        // SAFETY: as in `attributes`.
        let status = unsafe {
            ffi::sd_bench_resident(self.device.handle(), std::ptr::null(), 0, std::ptr::null(), 0, std::ptr::null(), READ, self.only, 10, 50, out.as_mut_ptr())
        };
        check(status).unwrap_or_else(|error| panic!("sd_bench_resident read: {error}"));
        out
    }
}

/// Bytes over microseconds, in GB/s (10^9 bytes).
fn gbs(bytes: f64, micros: f64) -> f64 {
    if micros > 0.0 {
        bytes / micros / 1e3
    } else {
        0.0
    }
}

fn layers() -> Vec<(String, [i32; 6])> {
    match std::env::var("RESIDENT_COST_KINDS") {
        Ok(text) => {
            let list: Vec<i32> = text.split(',').map(|part| part.trim().parse().expect("a GGML kind")).collect();
            assert_eq!(list.len(), 6, "RESIDENT_COST_KINDS is gate,up,down,shared gate,shared up,shared down");
            vec![("custom".to_string(), [list[0], list[1], list[2], list[3], list[4], list[5]])]
        }
        Err(_) => LAYERS.iter().map(|(name, kinds)| (name.to_string(), *kinds)).collect(),
    }
}

struct Line {
    layer: String,
    rows: usize,
    f: f64,
    mode: &'static str,
    out: [f64; 32],
    real_f: f64,
    real_o: f64,
}

#[test]
#[ignore = "a GPU microbenchmark: it is run on purpose (see the top of this file)"]
fn resident_gate_up_costs() {
    let count = device_count().expect("the GPU runtime loads");
    assert!(count > 0, "no GPU");
    let ordinal: usize = std::env::var("RESIDENT_COST_DEVICE").ok().map_or(0, |text| text.parse().expect("a device ordinal"));
    let resident: usize = std::env::var("RESIDENT_COST_RESIDENT").ok().map_or(DEFAULT_RESIDENT, |text| text.parse().expect("a number of experts"));
    let only: u32 = std::env::var("RESIDENT_COST_ONLY").ok().map_or(1, |text| text.parse().expect("0 or 1"));
    let (name, _free, total) = device_info(ordinal).unwrap();
    // (the IQ grids, as the engine's load_iq_tables gives them, before any device: the IQ kinds' row bytes need them)
    const GRIDS: &[u8] = include_bytes!("../../straydog-engine/src/iq_grids.bin");
    const OFFSETS: [usize; 6] = [0, 2048, 6144, 14336, 15360, 17408];
    let grid = |index: usize| &GRIDS[OFFSETS[index]..OFFSETS.get(index + 1).copied().unwrap_or(GRIDS.len())];
    // SAFETY: six grids of the sizes the library expects; it copies them.
    let status = unsafe {
        straydog_gpu::ffi::sd_iq_tables(grid(0).as_ptr(), grid(1).as_ptr(), grid(2).as_ptr(), grid(3).as_ptr(), grid(4).as_ptr(), grid(5).as_ptr())
    };
    assert_eq!(status, 0, "the IQ grids reach the library");
    let device = Device::new(ordinal, 1 << 30).expect("a device with a 1 GiB budget");
    let mut bench = Bench::new(device, resident, only);
    println!("device {ordinal}: {name}, {} MiB; {resident} of {EXPERTS} experts resident, ExpertGemv::only {only}; {REPS} launches after {WARMUP} warm-up", total >> 20);

    let attributes = bench.attributes();
    let sm = attributes[0];
    let warps_an_sm = attributes[2] / 32.0;
    println!(
        "SMs {:.0}, L2 {:.1} MiB, threads an SM {:.0} ({warps_an_sm:.0} warps), registers an SM {:.0}, shared bytes an SM {:.0}, compute capability {:.1}",
        sm,
        attributes[1] / (1u64 << 20) as f64,
        attributes[2],
        attributes[3],
        attributes[4],
        attributes[5] / 10.0
    );
    println!("{:<34} {:>5} {:>13} {:>11} {:>9} {:>9} {:>14} {:>14}", "kernel (256 threads a block)", "regs", "static shared", "local", "max block", "blocks/SM", "warps/SM (pct)", "blocks a wave");
    for (label, at) in [("resident_expert_gemv (CUDA's)", 8), ("resident_expert_gemv_kind<22> (RDNA4's)", 16)] {
        let blocks_an_sm = attributes[at + 4];
        println!(
            "{label:<34} {:>5.0} {:>13.0} {:>11.0} {:>9.0} {:>9.0} {:>8.0} ({:>3.0}%) {:>14.0}",
            attributes[at],
            attributes[at + 1],
            attributes[at + 2],
            attributes[at + 3],
            blocks_an_sm,
            blocks_an_sm * 8.0,
            blocks_an_sm * 8.0 / warps_an_sm * 100.0,
            blocks_an_sm * sm
        );
    }

    // (the GPU's clocks and the driver's first launches settle before the first figure; also the reference read)
    let settle = bench.lists(SETTLE_REPS as usize, 2, 0.71, 0.19);
    let first = layers()[0].1;
    bench.call(&first, 2, &settle.selected, FULL, 0, SETTLE_REPS);
    let read = bench.read();
    println!("reference: a plain read kernel over 512 MiB: {:.1} us median, {:.1} GB/s (least {:.1} us, {:.1} GB/s)", read[0], gbs(read[17], read[0]), read[1], gbs(read[17], read[1]));

    let mut lines: Vec<Line> = Vec::new();
    for (layer, kinds) in layers() {
        for rows in 1..=4usize {
            let o = overlap(rows);
            let mut absent_lists = None;
            for &f in &RESIDENT_SHARES {
                let lists = bench.lists((WARMUP + REPS) as usize, rows, f, o);
                let real_f = lists.resident_routes as f64 / lists.routes as f64;
                let real_o = if lists.pairs > 0 { lists.overlapped / lists.pairs as f64 } else { 0.0 };
                for (mode, label) in [(FULL, "full"), (ROUTED, "routed")] {
                    let out = bench.call(&kinds, rows, &lists.selected, mode, WARMUP, REPS);
                    eprintln!("{layer} rows {rows} F {f:.2} {label}: median {:.1} us", out[0]);
                    lines.push(Line { layer: layer.clone(), rows, f, mode: label, out, real_f, real_o });
                }
                if f == 0.0 {
                    absent_lists = Some((lists.selected, real_f, real_o));
                }
            }
            // (the shared expert alone reads no route; the empty launch has every route absent)
            let (selected, real_f, real_o) = absent_lists.expect("the F = 0 lists");
            for (mode, label) in [(SHARED, "shared"), (EMPTY, "empty")] {
                let out = bench.call(&kinds, rows, &selected, mode, WARMUP, REPS);
                eprintln!("{layer} rows {rows} {label}: median {:.1} us", out[0]);
                lines.push(Line { layer: layer.clone(), rows, f: 0.0, mode: label, out, real_f, real_o });
            }
        }
    }

    println!();
    for (layer, kinds) in layers() {
        let any = lines.iter().find(|line| line.layer == layer).expect("a line").out;
        println!(
            "{layer}: routed gate/up kind {}/{}, down {}; shared gate/up/down {}/{}/{}; routed bytes an expert: gate {:.0} up {:.0} down {:.0}; shared gate {:.0} up {:.0} down {:.0}; slot {:.0}; \
             resident gate/up in SoA layout: {}; {:.0} slots ({:.0} MiB), {:.0} copies of the shared pair rotated",
            kinds[0],
            kinds[1],
            kinds[2],
            kinds[3],
            kinds[4],
            kinds[5],
            any[20],
            any[21],
            any[22],
            any[23],
            any[24],
            any[25],
            any[26],
            if any[30] > 0.0 { "yes" } else { "no" },
            any[27],
            any[29] / (1u64 << 20) as f64,
            any[28]
        );
    }
    println!();
    println!("Times are GPU microseconds of the launch (events around it); blocks = grid x * rows; lanes a row: routed gate/up, shared gate/up;");
    println!("routes res/abs = resident and absent routes a launch (all rows); GB/s all = resident routed gate and up bytes of every row's routes");
    println!("(+ the shared pair, once, where the launch has it) over the median; GB/s uniq = the same with a distinct expert counted once;");
    println!("F, O realised = the lists' resident share of routes and a row's share of routes repeating the row before's.");
    println!();
    println!(
        "{:<8} {:>4} {:>5} {:>5} {:>5} {:<7} {:>6} {:>5} {:>9} {:>8} {:>8} {:>8} {:>8} {:>7} {:>9} {:>9} {:>7} {:>8} {:>8} {:>8}",
        "layer", "rows", "F", "F re", "O re", "mode", "blocks", "waves", "lanes", "median", "p10", "p90", "least", "host", "res/abs", "MB routed", "MB shr", "GB/s all", "GB/s uniq", "us/route"
    );
    for line in &lines {
        let o = &line.out;
        let blocks = o[6] * o[7];
        let shared = o[19];
        let all_bytes = o[17] + shared;
        let unique_bytes = o[18] + shared;
        let routes = o[15];
        println!(
            "{:<8} {:>4} {:>5.2} {:>5.3} {:>5.3} {:<7} {:>6.0} {:>5.0} {:>3.0}/{:.0}/{:.0}/{:.0} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>7.1} {:>4.1}/{:<4.1} {:>9.2} {:>7.2} {:>8.1} {:>9.1} {:>8.2}",
            line.layer,
            line.rows,
            line.f,
            line.real_f,
            line.real_o,
            line.mode,
            blocks,
            o[8],
            o[9],
            o[10],
            o[11],
            o[12],
            o[0],
            o[3],
            o[4],
            o[1],
            o[5],
            routes,
            o[16],
            o[17] / 1e6,
            shared / 1e6,
            gbs(all_bytes, o[0]),
            gbs(unique_bytes, o[0]),
            if routes > 0.0 { o[0] / routes } else { 0.0 }
        );
    }

    // What each launch costs beyond the empty one, by rows and F: the fixed cost, the routed work and the shared expert's.
    println!();
    println!("Summary (median us; empty = the launch's fixed cost; full - empty = the work in it; routed - empty and shared - empty = its parts):");
    println!("{:<8} {:>4} {:>5} {:>8} {:>8} {:>8} {:>8} {:>13} {:>13} {:>13}", "layer", "rows", "F", "full", "routed", "shared", "empty", "full-empty", "routed-empty", "shared-empty");
    for (layer, _) in layers() {
        for rows in 1..=4usize {
            let find = |mode: &str, f: f64| lines.iter().find(|line| line.layer == layer && line.rows == rows && line.mode == mode && (line.f - f).abs() < 1e-9).map(|line| line.out[0]);
            let shared = find("shared", 0.0).unwrap_or(0.0);
            let empty = find("empty", 0.0).unwrap_or(0.0);
            for &f in &RESIDENT_SHARES {
                let full = find("full", f).unwrap_or(0.0);
                let routed = find("routed", f).unwrap_or(0.0);
                println!(
                    "{layer:<8} {rows:>4} {f:>5.2} {full:>8.1} {routed:>8.1} {shared:>8.1} {empty:>8.1} {:>13.1} {:>13.1} {:>13.1}",
                    full - empty,
                    routed - empty,
                    shared - empty
                );
            }
        }
    }
}
