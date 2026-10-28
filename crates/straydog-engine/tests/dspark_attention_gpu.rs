//! DSpark's common noncausal key set and sink denominator against an F64 oracle.
#![cfg(any(feature = "hip", feature = "cuda"))]
use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;

const HEADS: usize = 64;
const DIM: usize = 512;
struct Random(u64);
impl Random {
    fn floats(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| {
            self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        }).collect()
    }
}
fn upload(gpu: &mut Backend, values: &[f32]) -> Buffer {
    let b = gpu.buffer(std::mem::size_of_val(values) as u64).unwrap();
    gpu.write_buffer(&b, 0, values).unwrap(); b
}
fn read(gpu: &mut Backend, b: &Buffer, n: usize) -> Vec<f32> {
    let mut v = vec![0.0; n]; gpu.read_buffer(b, 0, &mut v).unwrap(); v
}
fn filled(gpu: &mut Backend, n: usize, value: f32) -> Buffer { upload(gpu, &vec![value; n]) }
#[allow(clippy::too_many_arguments)]
fn oracle(q: &[f32], ring: &[f32], block: &[f32], sinks: &[f32], heads: usize, slots: usize, anchor: usize, history: usize) -> Vec<f64> {
    let keys: Vec<&[f32]> = (anchor - history..anchor).map(|p| &ring[p % 128 * DIM..(p % 128 + 1) * DIM])
        .chain(block[..slots * DIM].chunks_exact(DIM)).collect();
    q[..slots * heads * DIM].chunks_exact(DIM).enumerate().flat_map(|(row, query)| {
        let sink = sinks[row % heads] as f64;
        let scores: Vec<f64> = keys.iter().map(|key| {
            query.iter().zip(*key).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() / (DIM as f64).sqrt()
        }).collect();
        let top = scores.iter().copied().fold(sink, f64::max);
        let weights: Vec<f64> = scores.iter().map(|x| (x - top).exp()).collect();
        let denominator = weights.iter().sum::<f64>() + (sink - top).exp();
        (0..DIM).map(|d| weights.iter().zip(&keys).map(|(&p, key)| p * key[d] as f64).sum::<f64>() / denominator).collect::<Vec<_>>()
    }).collect()
}
fn check(label: &str, actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    let mut worst = 0f64;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let error = (a as f64 - b).abs() / (1.0 + b.abs());
        assert!(error <= 2e-5, "{label} value {i}: {a} vs {b}, error {error}");
        worst = worst.max(error);
    }
    eprintln!("{label}: {} values, max abs/(1+abs(ref))={worst:e}", actual.len());
}
fn check_constant(label: &str, actual: &[f32], value: f64) { check(label, actual, &vec![value; actual.len()]); }

#[test]
fn b1_to_b5_history_0_37_128_match_independent_cpu() {
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let q = random.floats(5 * HEADS * DIM, 0.3);
    let ring = random.floats(128 * DIM, 0.4);
    let block = random.floats(5 * DIM, 0.6);
    let mut sinks = random.floats(HEADS, 2.0);
    for head in (0..HEADS).step_by(16) { sinks[head] = 30.0; sinks[head + 1] = -30.0; }
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let (q_gpu, ring_gpu, block_gpu, sinks_gpu) = (upload(&mut gpu, &q), upload(&mut gpu, &ring),
        upload(&mut gpu, &block), upload(&mut gpu, &sinks));
    let out = gpu.buffer((q.len() * 4) as u64).unwrap();
    for slots in 1..=5 {
        for (history, anchor) in [(0, 0), (37, 149), (128, 389)] {
            gpu.dspark_attention(q_gpu.base(), &ring_gpu, block_gpu.base(), &sinks_gpu, HEADS as u64, slots as u64,
                anchor as u32, None, history as u64, out.base()).unwrap();
            let expected = oracle(&q, &ring, &block, &sinks, HEADS, slots, anchor, history);
            check(&format!("B={slots} H={history} anchor={anchor}"), &read(&mut gpu, &out, expected.len()), &expected);
        }
    }
    // a device's half of the heads: q [slots,32,512] and its sinks
    for slots in [1, 4] {
        let half = HEADS / 2;
        gpu.dspark_attention(q_gpu.base(), &ring_gpu, block_gpu.base(), &sinks_gpu, half as u64, slots as u64, 149, None, 37,
            out.base()).unwrap();
        let expected = oracle(&q, &ring, &block, &sinks, half, slots, 149, 37);
        check(&format!("B={slots} {half} heads"), &read(&mut gpu, &out, expected.len()), &expected);
    }
    // the anchor position from a device i32: a recorded graph follows it as it moves
    let anchor = gpu.buffer(4).unwrap();
    gpu.write_buffer(&anchor, 0, &[389i32]).unwrap();
    gpu.graph_begin().unwrap();
    gpu.dspark_attention(q_gpu.base(), &ring_gpu, block_gpu.base(), &sinks_gpu, HEADS as u64, 3, 389, Some(anchor.base().cast_const()), 128,
        out.base()).unwrap();
    gpu.graph_end(140, true).unwrap();
    for at in [389usize, 200, 128] {
        gpu.write_buffer(&anchor, 0, &[at as i32]).unwrap();
        assert!(gpu.graph_run(140).unwrap());
        let expected = oracle(&q, &ring, &block, &sinks, HEADS, 3, at, 128);
        check(&format!("B=3 replayed at {at}"), &read(&mut gpu, &out, expected.len()), &expected);
    }
    for (heads, slots, anchor, history) in [(64, 0, 0, 0), (64, 6, 0, 0), (64, 1, 129, 129), (64, 1, 36, 37), (64, 1, u32::MAX, 0),
                                            (0, 1, 0, 0), (65, 1, 0, 0)] {
        assert!(gpu.dspark_attention(q_gpu.base(), &ring_gpu, block_gpu.base(), &sinks_gpu,
            heads, slots, anchor, None, history, out.base()).is_err());
    }
}

#[test]
fn first_capture_future_key_and_sink_mass_are_observable() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let q = filled(&mut gpu, 5 * HEADS * DIM, 0.0);
    let ring = filled(&mut gpu, 128 * DIM, 23.0); // No history: must never be attended.
    let mut block: Vec<f32> = (1..=5).flat_map(|i| vec![i as f32; DIM]).collect();
    let block_gpu = upload(&mut gpu, &block);
    let sinks = upload(&mut gpu, &[0.0; HEADS]);
    let out = gpu.buffer((5 * HEADS * DIM * 4) as u64).unwrap();
    gpu.graph_begin().unwrap();
    gpu.dspark_attention(q.base(), &ring, block_gpu.base(), &sinks, HEADS as u64, 5, 0, None, 0, out.base()).unwrap();
    gpu.graph_end(139, true).unwrap();
    assert!(gpu.graph_run(139).unwrap());
    // Every query, including the first, sees five keys and one sink: (1+2+3+4+5)/6.
    check_constant("captured noncausal block", &read(&mut gpu, &out, 5 * HEADS * DIM), 2.5);
    block[4 * DIM..].fill(11.0);
    gpu.write_buffer(&block_gpu, 0, &block).unwrap();
    assert!(gpu.graph_run(139).unwrap());
    check_constant("future key changed", &read(&mut gpu, &out, 5 * HEADS * DIM), 3.5);
    for (sink, expected) in [(1000.0f32, 0.0f64), (-1000.0, 4.2)] {
        gpu.write_buffer(&sinks, 0, &[sink; HEADS]).unwrap();
        assert!(gpu.graph_run(139).unwrap());
        check_constant(&format!("sink={sink}"), &read(&mut gpu, &out, 5 * HEADS * DIM), expected);
    }
}
