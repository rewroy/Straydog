//! Two GPUs' linked sums (gpu/device_link.hpp), as tensor-parallel Qwen3.5 MoE adds its parts:
//! many sums queued on both devices at once, each with other parts, every result exact and equal
//! on both; the same through recorded graphs replayed with new parts; a sum whose peer never comes
//! fails instead of hanging, and a reset starts the counts over; sums folded into the kernels forming
//! their parts (a projection's GEMV, the routed experts' down) give the bits of the ops they stand
//! in for, also among plain sums in recorded graphs, also when the last block's waves of a folded
//! sum run far apart (as they can on a GPU another process loads). Part links, as tensor-parallel
//! DeepSeek-V4.1 sums its decode steps' outputs (sent as they form, added as the next step reads
//! them), give the bits of sd_op_sum_devices and the ops they stand in for. Skipped without two GPUs.
//!
//! cargo test --release -p straydog-engine --features hip --test link_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;

/// Width of a sum in floats (a decode step's stream; a prefill chunk's is 64 times this).
const WIDTH: usize = 2048;

/// The GPUs for one test at a time: while one test records a graph on its stream, some HIP calls of
/// another thread of this process fail (hipBLASLt's, as a backend opens: "operation not permitted
/// when stream is capturing"), and linked sums of tests run beside one recording went wrong now and
/// then (never one test at a time, nor beside another process's load).
fn gpus() -> std::sync::MutexGuard<'static, ()> {
    static GPUS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn pair() -> Option<(Backend, Backend)> {
    pair_of(64 * WIDTH)
}

fn pair_of(capacity: usize) -> Option<(Backend, Backend)> {
    let (Ok(mut first), Ok(mut second)) = (Backend::new(0, 2 << 30), Backend::new(1, 2 << 30)) else {
        eprintln!("skipped: linked sums need two GPUs");
        return None;
    };
    Backend::link_sums(&mut first, &mut second, capacity as u64).unwrap();
    Some((first, second))
}

/// Parts of `sums` sums of `count` floats: distinct per device, sum and position, exact in F32.
fn parts(device: usize, sums: usize, count: usize, salt: usize) -> Vec<f32> {
    (0..sums * count).map(|i| ((i * 7 + salt * 13 + device * 101) % 100_003) as f32 * if device == 0 { 1.0 } else { -0.25 }).collect()
}

fn read(gpu: &mut Backend, buffer: &Buffer, count: usize) -> Vec<f32> {
    let mut values = vec![0f32; count];
    gpu.read_buffer(buffer, 0, &mut values).unwrap();
    values
}

/// Every result is this device's part plus the other's, the same bits on both devices.
fn check(a: &[f32], b: &[f32], first: &[f32], second: &[f32], base: &[f32]) {
    for i in 0..a.len() {
        let expected = base[i] + (first[i] + second[i]);
        assert_eq!(a[i].to_bits(), expected.to_bits(), "device 0 value {i}: {} against {expected}", a[i]);
        assert_eq!(b[i].to_bits(), expected.to_bits(), "device 1 value {i}: {} against {expected}", b[i]);
    }
}

/// DeepSeek-V4.1's decode sums (an attention's or the experts' output, a row's activations, a
/// verify pass's), thousands queued on both devices without waiting, between work that keeps one
/// device busier than the other at times: every result exact, so no part was read before it landed.
#[test]
fn linked_sums_of_decode_sizes() {
    let _gpus = gpus();
    let Some((mut a, mut b)) = pair_of(64_512) else { return };
    let busy = a.buffer(64 << 20).unwrap();
    for (round, (count, sums)) in [(5120, 1500), (16_128, 600), (64_512, 120), (5120, 1500)].into_iter().enumerate() {
        let (first, second) = (parts(0, sums, count, count + round), parts(1, sums, count, count + round));
        let bytes = (sums * count * 4) as u64;
        let (part_a, part_b, out_a, out_b) = (a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap(), a.buffer(bytes).unwrap(),
                                              b.buffer(bytes).unwrap());
        a.write_buffer(&part_a, 0, &first).unwrap();
        b.write_buffer(&part_b, 0, &second).unwrap();
        for sum in 0..sums {
            let at = sum * count;
            // (every so often device 0 copies 64 MB first, so it falls behind and then catches up)
            if sum % 97 == 0 {
                a.copy(busy.base().cast_const(), busy.at(8 << 20), 32 << 20).unwrap();
            }
            a.link_sum(part_a.at(at), count as u64, out_a.at(at), false).unwrap();
            b.link_sum(part_b.at(at), count as u64, out_b.at(at), false).unwrap();
        }
        let (result_a, result_b) = (read(&mut a, &out_a, sums * count), read(&mut b, &out_b, sums * count));
        assert!(!a.link_failed().unwrap() && !b.link_failed().unwrap());
        check(&result_a, &result_b, &first, &second, &vec![0.0; sums * count]);
    }
}

/// sum_devices reads each other device's part where it lies: a device that writes its part buffer
/// again right after a sum (as a pass's next wave of experts writes its activations) must not
/// change what another device's sum reads. Each round writes new parts into the same buffers, sums
/// them, and the next round overwrites them at once; every round's sums must be its own parts'.
#[test]
fn device_sums_read_parts_before_they_are_written_again() {
    let _gpus = gpus();
    let (Ok(mut a), Ok(mut b)) = (Backend::new(0, 2 << 30), Backend::new(1, 2 << 30)) else {
        eprintln!("skipped: device sums need two GPUs");
        return;
    };
    let (count, rounds) = (WIDTH * 8, 400);
    let (first, second) = (parts(0, rounds, count, 5), parts(1, rounds, count, 6));
    let bytes = (rounds * count * 4) as u64;
    let (source_a, source_b) = (a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap());
    a.write_buffer(&source_a, 0, &first).unwrap();
    b.write_buffer(&source_b, 0, &second).unwrap();
    let (part_a, part_b) = (a.buffer((count * 4) as u64).unwrap(), b.buffer((count * 4) as u64).unwrap());
    let (out_a, out_b) = (a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap());
    let busy = b.buffer(64 << 20).unwrap();
    for round in 0..rounds {
        let at = round * count;
        // (now and then the second device falls behind)
        if round % 7 == 0 {
            b.copy(busy.base().cast_const(), busy.at(8 << 20), 32 << 20).unwrap();
        }
        straydog_engine::ops::devices_settle(&[&a, &b]).unwrap();
        a.copy(source_a.at(at).cast_const(), part_a.base(), (count * 4) as u64).unwrap();
        b.copy(source_b.at(at).cast_const(), part_b.base(), (count * 4) as u64).unwrap();
        straydog_engine::ops::sum_devices(&[&a, &b], &[out_a.at(at), out_b.at(at)], &[part_a.base().cast_const(), part_b.base().cast_const()],
                                          count as u64).unwrap();
    }
    let (result_a, result_b) = (read(&mut a, &out_a, rounds * count), read(&mut b, &out_b, rounds * count));
    check(&result_a, &result_b, &first, &second, &vec![0.0; rounds * count]);
}

#[test]
fn linked_sums() {
    let _gpus = gpus();
    let Some((mut a, mut b)) = pair() else { return };
    // Many sums queued back to back without waiting (each reads an inbox slot the one two before
    // it read), of a decode step's width and of a prefill chunk's, with and without accumulating.
    for (count, sums) in [(WIDTH, 600), (64 * WIDTH, 40)] {
        for accumulate in [false, true] {
            let (first, second) = (parts(0, sums, count, count), parts(1, sums, count, count));
            let base: Vec<f32> = (0..sums * count).map(|i| if accumulate { (i % 977) as f32 } else { 0.0 }).collect();
            let bytes = (sums * count * 4) as u64;
            let (part_a, part_b, out_a, out_b) = (a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap(), a.buffer(bytes).unwrap(),
                                                  b.buffer(bytes).unwrap());
            a.write_buffer(&part_a, 0, &first).unwrap();
            b.write_buffer(&part_b, 0, &second).unwrap();
            a.write_buffer(&out_a, 0, &base).unwrap();
            b.write_buffer(&out_b, 0, &base).unwrap();
            for sum in 0..sums {
                let at = sum * count;
                a.link_sum(part_a.at(at), count as u64, out_a.at(at), accumulate).unwrap();
                b.link_sum(part_b.at(at), count as u64, out_b.at(at), accumulate).unwrap();
            }
            let (result_a, result_b) = (read(&mut a, &out_a, sums * count), read(&mut b, &out_b, sums * count));
            assert!(!a.link_failed().unwrap() && !b.link_failed().unwrap());
            check(&result_a, &result_b, &first, &second, &base);
        }
    }
    // Recorded graphs of eight sums each, replayed with new parts every time: the counts carry on
    // in device memory.
    let sums = 8;
    let bytes = (sums * WIDTH * 4) as u64;
    let (part_a, part_b, out_a, out_b) = (a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap(), a.buffer(bytes).unwrap(),
                                          b.buffer(bytes).unwrap());
    for (gpu, part, out) in [(&mut a, &part_a, &out_a), (&mut b, &part_b, &out_b)] {
        gpu.graph_begin().unwrap();
        for sum in 0..sums {
            gpu.link_sum(part.at(sum * WIDTH), WIDTH as u64, out.at(sum * WIDTH), false).unwrap();
        }
        gpu.graph_end(7, true).unwrap();
    }
    let zero = vec![0f32; sums * WIDTH];
    for replay in 0..200 {
        let (first, second) = (parts(0, sums, WIDTH, replay), parts(1, sums, WIDTH, replay));
        a.write_buffer(&part_a, 0, &first).unwrap();
        b.write_buffer(&part_b, 0, &second).unwrap();
        assert!(a.graph_run(7).unwrap() && b.graph_run(7).unwrap());
        let (result_a, result_b) = (read(&mut a, &out_a, sums * WIDTH), read(&mut b, &out_b, sums * WIDTH));
        check(&result_a, &result_b, &first, &second, &zero);
    }
    assert!(!a.link_failed().unwrap() && !b.link_failed().unwrap());
    // A sum the other device never joins gives up after the await timeout; after a reset both
    // count from zero again and sum as before.
    let started = std::time::Instant::now();
    a.link_sum(part_a.base(), WIDTH as u64, out_a.base(), false).unwrap();
    assert!(a.link_failed().unwrap(), "a lone sum must fail");
    eprintln!("a lone sum failed after {:.1} s", started.elapsed().as_secs_f64());
    a.link_reset().unwrap();
    assert!(!a.link_failed().unwrap() && !b.link_failed().unwrap());
    let (first, second) = (parts(0, 1, WIDTH, 5), parts(1, 1, WIDTH, 5));
    a.write_buffer(&part_a, 0, &first).unwrap();
    b.write_buffer(&part_b, 0, &second).unwrap();
    a.link_sum(part_a.base(), WIDTH as u64, out_a.base(), false).unwrap();
    b.link_sum(part_b.base(), WIDTH as u64, out_b.base(), false).unwrap();
    let (result_a, result_b) = (read(&mut a, &out_a, WIDTH), read(&mut b, &out_b, WIDTH));
    check(&result_a, &result_b, &first, &second, &vec![0f32; WIDTH]);
}

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn floats(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale).collect()
    }
    /// Q8_0 rows of `cols` values: FP16 scales between 2^-7 and 2^-3, either sign, random quants.
    fn q8(&mut self, rows: usize, cols: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows * cols / 32 * 34);
        for _ in 0..rows * cols / 32 {
            let bits = ((self.next() & 1) << 15) | ((8 + self.next() % 5) << 10) | (self.next() & 0x3ff);
            bytes.extend_from_slice(&(bits as u16).to_le_bytes());
            for _ in 0..32 {
                bytes.push(self.next() as u8);
            }
        }
        bytes
    }
}

fn upload(gpu: &mut Backend, values: &[f32]) -> Buffer {
    let buffer = gpu.buffer((values.len() * 4) as u64).unwrap();
    gpu.write_buffer(&buffer, 0, values).unwrap();
    buffer
}

fn bits(gpu: &mut Backend, buffer: &Buffer, count: usize) -> Vec<u32> {
    read(gpu, buffer, count).iter().map(|value| value.to_bits()).collect()
}

const FFN: usize = 256;
const ROW: usize = WIDTH / 32 * 34;
const SLOT: usize = (2 * FFN * ROW + WIDTH * FFN / 32 * 34).next_multiple_of(256);
const OFFSETS: [u64; 3] = [0, (FFN * ROW) as u64, (2 * FFN * ROW) as u64];

/// One device's side of a decode step's two linked sums: a projection's Q8_0 matrix and input, and
/// eight routed experts (ffn 256 of 2048, Q8_0) in slots of a 256-entry table, their routes and
/// weights; the stream (both devices') and the shared expert's part the step starts from; its
/// buffers.
struct Side {
    matrix: Buffer,
    input: Buffer,
    table: Buffer,
    ids: Buffer,
    weights: Buffer,
    x: Buffer,
    scratch: Buffer,
    stream: Vec<f32>,
    shared: Vec<f32>,
    part: Buffer,
    out: Buffer,
    product: Buffer,
    sum: Buffer,
}

fn side(gpu: &mut Backend, random: &mut Random, stream: &[f32]) -> Side {
    let matrix = gpu.upload_bytes(&random.q8(WIDTH, WIDTH)).unwrap();
    let input = upload(gpu, &random.floats(WIDTH, 2.0));
    let mut slots = Vec::new();
    for _ in 0..8 {
        let mut expert = random.q8(2 * FFN, WIDTH);
        expert.extend(random.q8(WIDTH, FFN));
        expert.resize(SLOT, 0);
        slots.extend(expert);
    }
    let slots = gpu.upload_bytes(&slots).unwrap();
    let ids: [i32; 8] = [3, 17, 40, 99, 128, 200, 201, 255];
    let mut bases = vec![0u64; 256];
    for (index, &id) in ids.iter().enumerate() {
        bases[id as usize] = slots.base() as u64 + (index * SLOT) as u64;
    }
    let table = gpu.buffer(256 * 8).unwrap();
    gpu.expert_table(&table, &bases, SLOT as u64).unwrap();
    let id_buffer = gpu.buffer(32).unwrap();
    gpu.write_buffer(&id_buffer, 0, &ids).unwrap();
    let floats = (WIDTH * 4) as u64;
    Side {
        matrix, input, table, ids: id_buffer, weights: upload(gpu, &random.floats(8, 0.5)), x: upload(gpu, &random.floats(WIDTH, 2.0)),
        scratch: gpu.buffer(64 << 20).unwrap(), stream: stream.to_vec(), shared: random.floats(WIDTH, 1.0),
        part: gpu.buffer(floats).unwrap(), out: gpu.buffer(floats).unwrap(), product: gpu.buffer((8 * FFN * 4) as u64).unwrap(),
        sum: gpu.buffer(floats).unwrap(),
    }
}

/// A step's two sums on one device, folded or as the ops.
fn sums(gpu: &mut Backend, side: &Side, folded: bool) {
    let s = side;
    if folded {
        gpu.linear_link_sum(&s.matrix, 8, WIDTH as u64, WIDTH as u64, s.input.base(), s.part.base(), s.sum.base()).unwrap();
        gpu.experts_table_link_sum(&s.table, 256, s.ids.base(), SLOT as u64, OFFSETS, [8; 3], WIDTH as u64, FFN as u64, 8, s.x.base(),
                                   s.weights.base(), s.product.base(), s.out.base(), s.sum.base()).unwrap();
    } else {
        gpu.linear(&s.matrix, 8, WIDTH as u64, WIDTH as u64, 0, s.input.base(), 0, 1, s.part.base(), 0, &s.scratch).unwrap();
        gpu.link_sum(s.part.base(), WIDTH as u64, s.sum.base(), true).unwrap();
        gpu.experts_table(&s.table, 256, s.ids.base(), SLOT as u64, OFFSETS, [8; 3], WIDTH as u64, FFN as u64, 1, 8, s.x.base(),
                          Some(s.weights.base()), 0.0, true, s.product.base(), s.out.base()).unwrap();
        gpu.link_sum(s.out.base(), WIDTH as u64, s.sum.base(), true).unwrap();
    }
}

/// Three steps on both devices from the sides' stream and shared part (the experts accumulate on),
/// each folded or not as `folded` says (in a graph recorded and replayed when `graph`): the stream's
/// bits on each.
fn three_steps(a: &mut Backend, b: &mut Backend, sides: &[Side; 2], folded: [bool; 3], graph: bool) -> (Vec<u32>, Vec<u32>) {
    for (gpu, side) in [(&mut *a, &sides[0]), (&mut *b, &sides[1])] {
        gpu.write_buffer(&side.sum, 0, &side.stream).unwrap();
        gpu.write_buffer(&side.out, 0, &side.shared).unwrap();
    }
    if graph {
        for (gpu, side) in [(&mut *a, &sides[0]), (&mut *b, &sides[1])] {
            gpu.graph_begin().unwrap();
            for fold in folded {
                sums(gpu, side, fold);
            }
            gpu.graph_end(9, true).unwrap();
        }
        assert!(a.graph_run(9).unwrap() && b.graph_run(9).unwrap());
    } else {
        for fold in folded {
            sums(a, &sides[0], fold);
            sums(b, &sides[1], fold);
        }
    }
    let result = (bits(a, &sides[0].sum, WIDTH), bits(b, &sides[1].sum, WIDTH));
    assert!(!a.link_failed().unwrap() && !b.link_failed().unwrap());
    result
}

#[test]
fn folded_sums_match_the_ops() {
    let _gpus = gpus();
    let Some((mut a, mut b)) = pair() else { return };
    let mut random = Random(0x3c6e_f372_fe94_f82b);
    // (both devices hold the same stream; their parts differ)
    let stream = random.floats(WIDTH, 4.0);
    let sides = [side(&mut a, &mut random, &stream), side(&mut b, &mut random, &stream)];
    let expected = three_steps(&mut a, &mut b, &sides, [false; 3], false);
    assert_eq!(expected.0, expected.1, "the devices' streams");
    assert_eq!(three_steps(&mut a, &mut b, &sides, [true; 3], false), expected, "folded");
    assert_eq!(three_steps(&mut a, &mut b, &sides, [true, false, true], false), expected, "folded and plain");
    for replay in 0..20 {
        assert_eq!(three_steps(&mut a, &mut b, &sides, [true, false, true], true), expected, "graph replay {replay}");
    }
}

// ---- Part links (DeepSeek-V4.1's decode sums: parts sent as they form, added as they are read) ----

/// DeepSeek-V4.1's width, and the rows of a part link's sum (a verify pass's).
const DS_WIDTH: usize = 5120;
const DS_ROWS: usize = 4;

fn part_pair() -> Option<(Backend, Backend)> {
    let (Ok(mut first), Ok(mut second)) = (Backend::new(0, 2 << 30), Backend::new(1, 2 << 30)) else {
        eprintln!("skipped: part links need two GPUs");
        return None;
    };
    Backend::link_parts(&mut first, &mut second, (DS_ROWS * DS_WIDTH) as u64).unwrap();
    Some((first, second))
}

/// A device's columns of the experts' output (the down rows it holds): halves.
fn owned(device: usize) -> (usize, usize) {
    (device * DS_WIDTH / 2, DS_WIDTH / 2)
}

/// Random parts with exact zeros of both signs among them; with `columns`, zero (positive) outside
/// them, as a device's part of the experts' output is.
fn signed_parts(random: &mut Random, rows: usize, columns: Option<(usize, usize)>) -> Vec<f32> {
    let mut values = random.floats(rows * DS_WIDTH, 3.0);
    for (i, value) in values.iter_mut().enumerate() {
        match random.next() % 9 {
            0 => *value = -0.0,
            1 => *value = 0.0,
            _ => {}
        }
        if let Some((first, count)) = columns {
            if !(first..first + count).contains(&(i % DS_WIDTH)) {
                *value = 0.0;
            }
        }
    }
    values
}

/// Thousands of sums on both channels, queued on both devices without waiting (one device now and
/// then behind), of one to four rows: the attention's channel sends whole rows, the experts' only
/// the sender's columns. Every result is the two parts' sum as sd_op_sum_devices adds them, to the
/// bit (zeros' signs too), on both devices.
#[test]
fn part_links_add_as_the_device_sum_does() {
    let _gpus = gpus();
    let Some((mut a, mut b)) = part_pair() else { return };
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let sums = 1200;
    let rows_of = |sum: usize| 1 + (sum * 5 + sum / 7) % DS_ROWS;
    let mut first = Vec::new();
    let mut second = Vec::new();
    for sum in 0..sums {
        let rows = rows_of(sum);
        let experts = sum % 2 == 1;
        let mut part = signed_parts(&mut random, rows, experts.then(|| owned(0)));
        part.resize(DS_ROWS * DS_WIDTH, 0.0);
        first.extend(part);
        let mut part = signed_parts(&mut random, rows, experts.then(|| owned(1)));
        part.resize(DS_ROWS * DS_WIDTH, 0.0);
        second.extend(part);
    }
    let stride = DS_ROWS * DS_WIDTH;
    let (part_a, part_b) = (upload(&mut a, &first), upload(&mut b, &second));
    let bytes = (sums * stride * 4) as u64;
    let (out_a, out_b) = (a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap());
    let busy = a.buffer(64 << 20).unwrap();
    for sum in 0..sums {
        let (rows, channel, at) = (rows_of(sum), (sum % 2) as u32, sum * stride);
        if sum % 97 == 0 {
            a.copy(busy.base().cast_const(), busy.at(8 << 20), 32 << 20).unwrap();
        }
        for (device, gpu, part, out) in [(0, &mut a, &part_a, &out_a), (1, &mut b, &part_b, &out_b)] {
            let (columns_first, columns) = if channel == 1 { owned(device) } else { (0, DS_WIDTH) };
            gpu.part_send(channel, part.at(at), rows as u64, DS_WIDTH as u64, columns_first as u64, columns as u64).unwrap();
            gpu.part_receive(channel, part.at(at), (rows * DS_WIDTH) as u64, out.at(at)).unwrap();
        }
    }
    let (result_a, result_b) = (read(&mut a, &out_a, sums * stride), read(&mut b, &out_b, sums * stride));
    assert!(!a.part_failed().unwrap() && !b.part_failed().unwrap());
    for sum in 0..sums {
        for i in sum * stride..sum * stride + rows_of(sum) * DS_WIDTH {
            let expected = (first[i] + second[i]).to_bits();
            assert_eq!(result_a[i].to_bits(), expected, "sum {sum} value {i} on device 0: {} + {}", first[i], second[i]);
            assert_eq!(result_b[i].to_bits(), expected, "sum {sum} value {i} on device 1: {} + {}", first[i], second[i]);
        }
    }
    // A sum the other device never sends gives up after the await timeout; after a reset both count
    // from zero again.
    let started = std::time::Instant::now();
    a.part_send(0, part_a.base(), 1, DS_WIDTH as u64, 0, DS_WIDTH as u64).unwrap();
    a.part_receive(0, part_a.base(), DS_WIDTH as u64, out_a.base()).unwrap();
    assert!(a.part_failed().unwrap(), "a lone sum must fail");
    eprintln!("a lone part sum failed after {:.1} s", started.elapsed().as_secs_f64());
    a.part_reset().unwrap();
    assert!(!a.part_failed().unwrap() && !b.part_failed().unwrap());
    for (gpu, part, out) in [(&mut a, &part_a, &out_a), (&mut b, &part_b, &out_b)] {
        gpu.part_send(0, part.base(), 1, DS_WIDTH as u64, 0, DS_WIDTH as u64).unwrap();
        gpu.part_receive(0, part.base(), DS_WIDTH as u64, out.base()).unwrap();
    }
    let (result_a, result_b) = (read(&mut a, &out_a, DS_WIDTH), read(&mut b, &out_b, DS_WIDTH));
    for i in 0..DS_WIDTH {
        let expected = (first[i] + second[i]).to_bits();
        assert_eq!((result_a[i].to_bits(), result_b[i].to_bits()), (expected, expected), "after the reset, value {i}");
    }
}

const DS_HC: usize = 4;
const DS_MIX: usize = (2 + DS_HC) * DS_HC;
const DS_COLS: usize = 1024;
const DS_FFN: usize = 256;
const DS_EXPERTS: usize = 8;
const DS_TOP_K: usize = 4;

/// One device's side of a DeepSeek-V4.1 decode step's two sums: the attention output projection
/// (Q8_0, DS_COLS columns), the hyper-connection steps' weights and buffers (the same on both
/// devices, as the stream is), eight routed experts (whole gate and up, this device's half of the
/// down rows; Q8_0) and the shared expert's part of the output they add to (zero outside its
/// columns).
struct Step {
    matrix: Buffer,
    input: Buffer,
    scratch: Buffer,
    residual: Buffer,
    post_in: Buffer,
    comb_in: Buffer,
    fn_: Buffer,
    base: Buffer,
    scale: Buffer,
    collapse: Buffer,
    norm: Buffer,
    workspace: Buffer,
    slots: Buffer,
    route_weights: Buffer,
    shared: Vec<f32>,
    // outputs
    parts: [Buffer; 2],
    sums: [Buffer; 2],
    stream: Buffer,
    stream2: Buffer,
    coefficients: [Buffer; 3],
    x: Buffer,
    product: Buffer,
    received: Buffer,
}

fn step_side(gpu: &mut Backend, random: &mut Random, common: &mut Random, device: usize) -> Step {
    let floats = |gpu: &mut Backend, count: usize| gpu.buffer((count * 4) as u64).unwrap();
    let rows = DS_ROWS;
    let mut slots = Vec::new();
    let slot = (2 * DS_FFN * (DS_WIDTH / 32 * 34) + DS_WIDTH / 2 * (DS_FFN / 32 * 34)).next_multiple_of(256);
    for _ in 0..DS_EXPERTS {
        let mut expert = random.q8(2 * DS_FFN, DS_WIDTH);
        expert.extend(random.q8(DS_WIDTH / 2, DS_FFN));
        expert.resize(slot, 0);
        slots.extend(expert);
    }
    let (first, count) = owned(device);
    let shared = signed_parts(random, rows, Some((first, count)));
    Step {
        matrix: gpu.upload_bytes(&random.q8(DS_WIDTH, DS_COLS)).unwrap(),
        input: upload(gpu, &random.floats(rows * DS_COLS, 2.0)),
        scratch: gpu.buffer(64 << 20).unwrap(),
        residual: upload(gpu, &common.floats(rows * DS_HC * DS_WIDTH, 2.0)),
        post_in: upload(gpu, &common.floats(rows * DS_HC, 1.0)),
        comb_in: upload(gpu, &common.floats(rows * DS_HC * DS_HC, 0.5)),
        fn_: upload(gpu, &common.floats(DS_MIX * DS_HC * DS_WIDTH, 0.01)),
        base: upload(gpu, &common.floats(DS_MIX, 0.5)),
        scale: upload(gpu, &common.floats(3, 0.5)),
        collapse: upload(gpu, &common.floats(rows * DS_HC, 1.0)),
        norm: upload(gpu, &common.floats(DS_WIDTH, 1.0)),
        workspace: floats(gpu, 1 << 16),
        slots: gpu.upload_bytes(&slots).unwrap(),
        route_weights: upload(gpu, &common.floats(rows * DS_TOP_K, 0.5)),
        shared,
        parts: [floats(gpu, rows * DS_WIDTH), floats(gpu, rows * DS_WIDTH)],
        sums: [floats(gpu, rows * DS_WIDTH), floats(gpu, rows * DS_WIDTH)],
        stream: floats(gpu, rows * DS_HC * DS_WIDTH),
        stream2: floats(gpu, rows * DS_HC * DS_WIDTH),
        coefficients: [floats(gpu, rows * DS_HC), floats(gpu, rows * DS_HC), floats(gpu, rows * DS_HC * DS_HC)],
        x: floats(gpu, rows * DS_WIDTH),
        product: floats(gpu, rows * DS_TOP_K * DS_FFN),
        received: floats(gpu, rows * DS_WIDTH),
    }
}

/// A step's hyper-connection step from `x` (a sum, or with `link` this device's part of one).
fn step_hc(gpu: &mut Backend, s: &Step, x: &Buffer, rows: usize, link: Option<u32>, stream: &Buffer) {
    let [pre, post, comb] = &s.coefficients;
    gpu.hc_step(Some((x.base().cast_const(), s.residual.base().cast_const(), s.post_in.base().cast_const(), s.comb_in.base().cast_const())),
                stream.base(), Some((s.fn_.base().cast_const(), s.base.base().cast_const(), s.scale.base().cast_const(), pre.base(),
                post.base(), comb.base())), 20, 1e-6, 1e-6, s.collapse.base().cast_const(), s.norm.base().cast_const(), rows as u64,
                DS_HC as u64, DS_WIDTH as u64, &s.workspace, s.x.base(), link).unwrap();
}

/// A decode step of `rows` rows on both devices, linked or with device sums: the attention output
/// projection, the step into the FFN, the experts (gate, up and this device's down rows) added to
/// the shared expert's part, then the step into the next attention (`hc`) or the sum alone.
fn decode_step(a: &mut Backend, b: &mut Backend, sides: &[Step; 2], rows: usize, link: bool, hc: bool, routes: &[u64; 2]) {
    let slot = (2 * DS_FFN * (DS_WIDTH / 32 * 34) + DS_WIDTH / 2 * (DS_FFN / 32 * 34)).next_multiple_of(256) as u64;
    let offsets = [0, (DS_FFN * (DS_WIDTH / 32 * 34)) as u64, (2 * DS_FFN * (DS_WIDTH / 32 * 34)) as u64];
    let mut gpus = [a, b];
    for (gpu, s) in gpus.iter_mut().zip(sides) {
        gpu.write_buffer(&s.parts[1], 0, &s.shared[..rows * DS_WIDTH]).unwrap();
    }
    // the attention's sum
    for (gpu, s) in gpus.iter_mut().zip(sides) {
        if link {
            gpu.linear_send(0, &s.matrix, 8, DS_WIDTH as u64, DS_COLS as u64, s.input.base().cast_const(), rows as u64, s.parts[0].base())
                .unwrap();
            step_hc(gpu, s, &s.parts[0], rows, Some(0), &s.stream2);
        } else {
            gpu.linear(&s.matrix, 8, DS_WIDTH as u64, DS_COLS as u64, 0, s.input.base().cast_const(), 0, rows as u64, s.parts[0].base(), 0,
                       &s.scratch).unwrap();
        }
    }
    if !link {
        straydog_engine::ops::sum_devices(&[&*gpus[0], &*gpus[1]], &[sides[0].sums[0].base(), sides[1].sums[0].base()],
                                          &[sides[0].parts[0].base().cast_const(), sides[1].parts[0].base().cast_const()],
                                          (rows * DS_WIDTH) as u64).unwrap();
        for (gpu, s) in gpus.iter_mut().zip(sides) {
            step_hc(gpu, s, &s.sums[0], rows, None, &s.stream2);
        }
    }
    // the experts' sum
    for (device, (gpu, s)) in gpus.iter_mut().zip(sides).enumerate() {
        let bases: Vec<u64> = (0..rows * DS_TOP_K).map(|r| s.slots.base() as u64 + ((routes[device] as usize + r * 3) % DS_EXPERTS) as u64 * slot)
            .collect();
        let (first, count) = owned(device);
        let split = straydog_engine::ops::ExpertSplit { ffn_first: 0, ffn_rows: DS_FFN as u64, hidden_first: first as u64,
                                                        hidden_rows: count as u64, stages: 3, send: if link { 2 } else { 0 } };
        gpu.experts(Some(&bases), slot, offsets, [8; 3], DS_WIDTH as u64, DS_FFN as u64, rows as u64, DS_TOP_K as u64, s.x.base().cast_const(),
                    0, Some(s.route_weights.base().cast_const()), 0.0, true, s.product.base(), s.parts[1].base(), 0, Some(split)).unwrap();
        if link {
            if hc {
                step_hc(gpu, s, &s.parts[1], rows, Some(1), &s.stream);
            } else {
                gpu.part_receive(1, s.parts[1].base().cast_const(), (rows * DS_WIDTH) as u64, s.received.base()).unwrap();
            }
        }
    }
    if !link {
        straydog_engine::ops::sum_devices(&[&*gpus[0], &*gpus[1]], &[sides[0].sums[1].base(), sides[1].sums[1].base()],
                                          &[sides[0].parts[1].base().cast_const(), sides[1].parts[1].base().cast_const()],
                                          (rows * DS_WIDTH) as u64).unwrap();
        for (gpu, s) in gpus.iter_mut().zip(sides) {
            if hc {
                step_hc(gpu, s, &s.sums[1], rows, None, &s.stream);
            } else {
                gpu.copy(s.sums[1].base().cast_const(), s.received.base(), (rows * DS_WIDTH * 4) as u64).unwrap();
            }
        }
    }
}

/// What a step leaves on a device, as bits: the streams, coefficients and collapsed input of its
/// steps, and the experts' sum as received (without `hc`).
fn step_bits(gpu: &mut Backend, s: &Step, rows: usize, hc: bool) -> Vec<u32> {
    let mut all = Vec::new();
    let last = if hc { (&s.stream, rows * DS_HC * DS_WIDTH) } else { (&s.received, rows * DS_WIDTH) };
    for (buffer, count) in [(&s.stream2, rows * DS_HC * DS_WIDTH), (&s.coefficients[0], rows * DS_HC), (&s.coefficients[1], rows * DS_HC),
                            (&s.coefficients[2], rows * DS_HC * DS_HC), (&s.x, rows * DS_WIDTH), last] {
        all.extend(bits(gpu, buffer, count));
    }
    all
}

/// DeepSeek-V4.1's decode sums over part links (the attention output projection and the routed
/// experts' down sending as they form, the hyper-connection steps and a plain receive adding the
/// other's part as they read it) give the bits of the projection, the experts, sd_op_sum_devices
/// and the steps they stand in for, on both devices, for one to four rows, back to back and in
/// recorded graphs replayed.
#[test]
fn part_link_steps_match_the_device_sums() {
    let _gpus = gpus();
    let Some((mut a, mut b)) = part_pair() else { return };
    // (the stream and the steps' weights the same on both devices)
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    let common = || Random(0x1405_7b7e_f767_814f);
    let sides = [step_side(&mut a, &mut random, &mut common(), 0), step_side(&mut b, &mut random, &mut common(), 1)];
    let mut expected = Vec::new();
    for (index, rows) in [1, 3, 4, 2, 1, 1].into_iter().enumerate() {
        for hc in [true, false] {
            let routes = [index as u64, index as u64 * 5 + 1];
            decode_step(&mut a, &mut b, &sides, rows, false, hc, &routes);
            let plain = (step_bits(&mut a, &sides[0], rows, hc), step_bits(&mut b, &sides[1], rows, hc));
            assert_eq!(plain.0, plain.1, "rows {rows}: the devices' steps");
            for round in 0..3 {
                decode_step(&mut a, &mut b, &sides, rows, true, hc, &routes);
                let linked = (step_bits(&mut a, &sides[0], rows, hc), step_bits(&mut b, &sides[1], rows, hc));
                assert!(linked == plain, "rows {rows}, hc {hc}, round {round}: linked steps differ from the device sums'");
            }
            expected.push((rows, hc, routes, plain));
        }
    }
    // Many steps queued without waiting, then compared against their last.
    for &(rows, hc, routes, ref plain) in &expected {
        for _ in 0..40 {
            decode_step(&mut a, &mut b, &sides, rows, true, hc, &routes);
        }
        assert!((step_bits(&mut a, &sides[0], rows, hc), step_bits(&mut b, &sides[1], rows, hc)) == *plain, "rows {rows}: back to back");
    }
    // The attention's half recorded into graphs (with the experts' send and receive between
    // replays, outside them), replayed: the counts carry on in device memory.
    let rows = 1;
    for (gpu, s) in [(&mut a, &sides[0]), (&mut b, &sides[1])] {
        gpu.graph_begin().unwrap();
        gpu.linear_send(0, &s.matrix, 8, DS_WIDTH as u64, DS_COLS as u64, s.input.base().cast_const(), rows as u64, s.parts[0].base()).unwrap();
        step_hc(gpu, s, &s.parts[0], rows, Some(0), &s.stream2);
        gpu.graph_end(11, true).unwrap();
    }
    let (_, _, routes, plain) = expected.iter().find(|(r, hc, _, _)| *r == rows && *hc).unwrap();
    for replay in 0..30 {
        for (gpu, s) in [(&mut a, &sides[0]), (&mut b, &sides[1])] {
            gpu.write_buffer(&s.parts[1], 0, &s.shared[..rows * DS_WIDTH]).unwrap();
        }
        assert!(a.graph_run(11).unwrap() && b.graph_run(11).unwrap());
        let slot = (2 * DS_FFN * (DS_WIDTH / 32 * 34) + DS_WIDTH / 2 * (DS_FFN / 32 * 34)).next_multiple_of(256) as u64;
        let offsets = [0, (DS_FFN * (DS_WIDTH / 32 * 34)) as u64, (2 * DS_FFN * (DS_WIDTH / 32 * 34)) as u64];
        for (device, (gpu, s)) in [(&mut a, &sides[0]), (&mut b, &sides[1])].into_iter().enumerate() {
            let bases: Vec<u64> = (0..rows * DS_TOP_K)
                .map(|r| s.slots.base() as u64 + ((routes[device] as usize + r * 3) % DS_EXPERTS) as u64 * slot).collect();
            let (first, count) = owned(device);
            let split = straydog_engine::ops::ExpertSplit { ffn_first: 0, ffn_rows: DS_FFN as u64, hidden_first: first as u64,
                                                            hidden_rows: count as u64, stages: 3, send: 2 };
            gpu.experts(Some(&bases), slot, offsets, [8; 3], DS_WIDTH as u64, DS_FFN as u64, rows as u64, DS_TOP_K as u64,
                        s.x.base().cast_const(), 0, Some(s.route_weights.base().cast_const()), 0.0, true, s.product.base(),
                        s.parts[1].base(), 0, Some(split)).unwrap();
            step_hc(gpu, s, &s.parts[1], rows, Some(1), &s.stream);
        }
        assert!((step_bits(&mut a, &sides[0], rows, true), step_bits(&mut b, &sides[1], rows, true)) == *plain, "graph replay {replay}");
    }
    assert!(!a.part_failed().unwrap() && !b.part_failed().unwrap());
}

/// Wrong sums among `rounds` rounds of linked sums on two devices, one now and then behind: per
/// round, 1500 sums of a decode step's width in one kernel each, 300 of a prefill row's in three,
/// 600 folded into a projection's GEMV (each into its own zeroed out), 1500 over a part link, and
/// three-sum steps folded into the kernels forming their parts, plain and in recorded graphs. Each wrong sum of the first
/// three kinds is reported with what its wrong values hold.
fn wrong_linked_sums(a: &mut Backend, b: &mut Backend, rounds: usize) -> usize {
    let busy = a.buffer(64 << 20).unwrap();
    let mut wrong = 0;
    let mut random = Random(0x1234_5678_9abc_def1);
    let (matrix_a, matrix_b) = (a.upload_bytes(&random.q8(WIDTH, WIDTH)).unwrap(), b.upload_bytes(&random.q8(WIDTH, WIDTH)).unwrap());
    for round in 0..rounds {
        for (kind, count, sums) in [("link sum", 5120, 1500), ("3-kernel link sum", 16_128, 300), ("folded GEMV", WIDTH, 600),
                                    ("part link", 5120, 1500)] {
            let folded = kind == "folded GEMV";
            let bytes = (sums * count * 4) as u64;
            let (part_a, part_b, out_a, out_b) = if folded {
                (a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap(), a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap())
            } else {
                let (first, second) = (parts(0, sums, count, round * 2 + 1), parts(1, sums, count, round * 2 + 2));
                (upload(a, &first), upload(b, &second), a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap())
            };
            let inputs = folded.then(|| (upload(a, &random.floats(sums * WIDTH, 2.0)), upload(b, &random.floats(sums * WIDTH, 2.0))));
            for sum in 0..sums {
                let at = sum * count;
                if sum % 97 == 0 {
                    a.copy(busy.base().cast_const(), busy.at(8 << 20), 32 << 20).unwrap();
                }
                for (device, gpu, part, out) in [(0, &mut *a, &part_a, &out_a), (1, &mut *b, &part_b, &out_b)] {
                    match &inputs {
                        _ if kind == "part link" => {
                            gpu.part_send(0, part.at(at).cast_const(), 1, count as u64, 0, count as u64).unwrap();
                            gpu.part_receive(0, part.at(at).cast_const(), count as u64, out.at(at)).unwrap();
                        }
                        Some((input_a, input_b)) => {
                            let (matrix, input) = if device == 0 { (&matrix_a, input_a) } else { (&matrix_b, input_b) };
                            gpu.linear_link_sum(matrix, 8, WIDTH as u64, WIDTH as u64, input.at(at).cast_const(), part.at(at), out.at(at)).unwrap();
                        }
                        None => gpu.link_sum(part.at(at), count as u64, out.at(at), false).unwrap(),
                    }
                }
            }
            let (result_a, result_b) = (read(a, &out_a, sums * count), read(b, &out_b, sums * count));
            let (first, second) = (read(a, &part_a, sums * count), read(b, &part_b, sums * count));
            let failed = if kind == "part link" { a.part_failed().unwrap() || b.part_failed().unwrap() }
                         else { a.link_failed().unwrap() || b.link_failed().unwrap() };
            for (device, result, mine, theirs) in [(0, &result_a, &first, &second), (1, &result_b, &second, &first)] {
                for sum in 0..sums {
                    let mut held = std::collections::BTreeMap::new();
                    let (mut bad, mut low, mut high) = (0, usize::MAX, 0);
                    for j in 0..count {
                        let i = sum * count + j;
                        if result[i].to_bits() == (first[i] + second[i]).to_bits() {
                            continue;
                        }
                        bad += 1;
                        (low, high) = (low.min(j), high.max(j));
                        // the other's part of which sum, near this one
                        let back = (1..=3).find(|&k| sum >= k && result[i].to_bits() == (mine[i] + theirs[i - k * count]).to_bits());
                        *held.entry(back.map_or("other".to_string(), |k| format!("theirs of the sum {k} before"))).or_insert(0) += 1;
                    }
                    if bad > 0 {
                        wrong += 1;
                        eprintln!("round {round}, {kind}, device {device}, sum {sum}: {bad} values wrong (columns {low}..={high}): {held:?}");
                    }
                }
            }
            if failed {
                wrong += 1;
                eprintln!("round {round}, {kind}: a wait gave up");
                if kind == "part link" { a.part_reset().unwrap() } else { a.link_reset().unwrap() }
            }
        }
    }
    // three-sum steps, against the ops they stand in for
    let mut random = Random(0x3c6e_f372_fe94_f82b);
    let stream = random.floats(WIDTH, 4.0);
    let sides = [side(a, &mut random, &stream), side(b, &mut random, &stream)];
    let expected = three_steps(a, b, &sides, [false; 3], false);
    for step in 0..rounds * 40 {
        if three_steps(a, b, &sides, [true, false, true], step % 2 == 1) != expected {
            wrong += 1;
            eprintln!("folded step {step} wrong");
        }
    }
    wrong
}

/// A folded sum's last block adds the slot of its own sum whenever its waves run apart: here the
/// waves past the first wait 50 us before they add (sd_link_skew), as waves of a block can on a
/// GPU another process loads (the last block's first wave stores the next count by then).
#[test]
fn folded_sums_hold_when_waves_run_apart() {
    let _gpus = gpus();
    let Some((mut a, mut b)) = pair_of(64 * WIDTH) else { return };
    Backend::link_parts(&mut a, &mut b, 5120).unwrap();
    for gpu in [&mut a, &mut b] {
        gpu.link_skew(50).unwrap();
    }
    assert_eq!(wrong_linked_sums(&mut a, &mut b, 1), 0, "wrong linked sums with the last blocks' waves apart");
}

/// Linked sums stay exact while another process loads the bus with copies both ways and linked sums
/// of its own (link_traffic_stressor; without the load, STRESS=none). Ignored: it takes the GPUs
/// for a minute or so (ROUNDS rounds, default 3).
#[test]
#[ignore]
fn links_hold_beside_bus_traffic() {
    let _gpus = gpus();
    let Some((mut a, mut b)) = pair_of(64 * WIDTH) else { return };
    Backend::link_parts(&mut a, &mut b, 5120).unwrap();
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|value| value.parse().ok()).unwrap_or(3);
    // The stressor stops when told to, also when this test panics (else it would hold the GPUs).
    struct Stressor(std::path::PathBuf, std::process::Child);
    impl Drop for Stressor {
        fn drop(&mut self) {
            let _ = std::fs::write(&self.0, b"");
            let _ = self.1.wait();
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let stop = std::env::temp_dir().join(format!("straydog-link-stressor-{}", std::process::id()));
    let _ = std::fs::remove_file(&stop);
    let stressor = (std::env::var("STRESS").as_deref() != Ok("none")).then(|| {
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "link_traffic_stressor", "--ignored", "--nocapture"])
            .env("STRESS_SECONDS", "600").env("STRESS_STOP", &stop)
            .spawn().unwrap();
        Stressor(stop.clone(), child)
    });
    std::thread::sleep(std::time::Duration::from_secs(3));
    let wrong = wrong_linked_sums(&mut a, &mut b, rounds);
    drop(stressor);
    assert_eq!(wrong, 0, "wrong linked sums (and failed waits) beside the bus load");
}

/// Development (ignored, links_hold_beside_bus_traffic runs it in another process): STRESS_SECONDS
/// (default 60, or until the file STRESS_STOP names exists) of copies across the bus both ways and
/// within each device, with linked sums on a link of its own between them.
#[test]
#[ignore]
fn link_traffic_stressor() {
    let Some((mut a, mut b)) = pair_of(WIDTH * 8) else { return };
    let seconds: u64 = std::env::var("STRESS_SECONDS").ok().and_then(|value| value.parse().ok()).unwrap_or(60);
    let stop = std::env::var_os("STRESS_STOP").map(std::path::PathBuf::from);
    let bytes = 32u64 << 20;
    let (a1, a2, b1, b2) = (a.buffer(2 * bytes).unwrap(), a.buffer(2 * bytes).unwrap(), b.buffer(2 * bytes).unwrap(), b.buffer(2 * bytes).unwrap());
    let started = std::time::Instant::now();
    let mut rounds = 0u64;
    while started.elapsed().as_secs() < seconds && !stop.as_ref().is_some_and(|path| path.exists()) {
        for _ in 0..8 {
            a.copy_to(a1.base().cast_const(), &mut b, b1.base(), bytes).unwrap();
            b.copy_to(b2.base().cast_const(), &mut a, a2.base(), bytes).unwrap();
            a.copy(a1.at((bytes / 4) as usize).cast_const(), a1.base(), bytes).unwrap();
            b.copy(b1.at((bytes / 4) as usize).cast_const(), b2.base(), bytes).unwrap();
            for sum in 0..50 {
                a.link_sum(a2.at(sum * WIDTH * 4).cast_const(), (WIDTH * 4) as u64, a1.at((64 + sum) * WIDTH * 4), false).unwrap();
                b.link_sum(b2.at(sum * WIDTH * 4).cast_const(), (WIDTH * 4) as u64, b1.at((64 + sum) * WIDTH * 4), false).unwrap();
            }
        }
        let mut one = [0f32; 1];
        a.read_buffer(&a1, 0, &mut one).unwrap();
        b.read_buffer(&b1, 0, &mut one).unwrap();
        rounds += 1;
    }
    eprintln!("stressor: {rounds} rounds");
}
