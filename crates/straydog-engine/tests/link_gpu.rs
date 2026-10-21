//! Two GPUs' linked sums (gpu/device_link.hpp), as tensor-parallel Qwen3.5 MoE adds its parts:
//! many sums queued on both devices at once, each with other parts, every result exact and equal
//! on both; the same through recorded graphs replayed with new parts; a sum whose peer never comes
//! fails instead of hanging, and a reset starts the counts over; sums folded into the kernels forming
//! their parts (a projection's GEMV, the routed experts' down) give the bits of the ops they stand
//! in for, also among plain sums in recorded graphs. Skipped without two GPUs.
//!
//! cargo test --release -p straydog-engine --features hip --test link_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;

/// Width of a sum in floats (a decode step's stream; a prefill chunk's is 64 times this).
const WIDTH: usize = 2048;

fn pair() -> Option<(Backend, Backend)> {
    let (Ok(mut first), Ok(mut second)) = (Backend::new(0, 1 << 30), Backend::new(1, 1 << 30)) else {
        eprintln!("skipped: linked sums need two GPUs");
        return None;
    };
    Backend::link_sums(&mut first, &mut second, (64 * WIDTH) as u64).unwrap();
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

#[test]
fn linked_sums() {
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
