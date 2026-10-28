//! DSpark Markov/confidence head against an independent F64 CPU calculation.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;

struct Random(u64);
impl Random {
    fn float(&mut self, scale: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
    }
    fn floats(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.float(scale)).collect()
    }
    fn bf16(&mut self, n: usize) -> Vec<u16> {
        (0..n).map(|_| (self.float(0.5).to_bits() >> 16) as u16).collect()
    }
}

fn upload<T: Copy>(gpu: &mut Backend, values: &[T]) -> Buffer {
    let b = gpu.buffer(std::mem::size_of_val(values) as u64).unwrap();
    gpu.write_buffer(&b, 0, values).unwrap();
    b
}
fn read<T: Copy + Default>(gpu: &mut Backend, b: &Buffer, n: usize) -> Vec<T> {
    let mut v = vec![T::default(); n];
    gpu.read_buffer(b, 0, &mut v).unwrap();
    v
}
fn bf16(x: u16) -> f64 { f32::from_bits(u32::from(x) << 16) as f64 }

struct Case {
    v: usize, r: usize, w: usize, anchor: u32,
    body: Vec<f32>, hidden: Vec<f32>, w1: Vec<u16>, w2: Vec<u16>, conf: Vec<f32>,
}
struct Expected { logits: Vec<f64>, tokens: Vec<i32>, confidences: Vec<f64> }
impl Case {
    fn oracle(&self) -> Expected {
        let mut out = Expected { logits: Vec::new(), tokens: Vec::new(), confidences: Vec::new() };
        let mut previous = self.anchor as usize;
        for (slot, body) in self.body.chunks_exact(self.v).enumerate() {
            let m: Vec<f64> = self.w1[previous * self.r..(previous + 1) * self.r].iter().map(|&x| bf16(x)).collect();
            let logits: Vec<f64> = self.w2.chunks_exact(self.r).zip(body).map(|(row, &l)| {
                l as f64 + row.iter().zip(&m).map(|(&x, &y)| bf16(x) * y).sum::<f64>()
            }).collect();
            // Strict comparison preserves the lowest index in a tie.
            previous = (0..self.v).fold(0, |best, i| if logits[i] > logits[best] { i } else { best });
            let h = &self.hidden[slot * self.w..(slot + 1) * self.w];
            let dot = h.iter().map(|&x| x as f64).chain(m).zip(&self.conf).map(|(x, &y)| x * y as f64).sum::<f64>();
            out.confidences.push(1.0 / (1.0 + (-dot).exp()));
            out.tokens.push(previous as i32);
            out.logits.extend(logits);
        }
        out
    }
    /// Slot `slot`'s corrected logits with `previous` the token before it.
    fn logits_after(&self, slot: usize, previous: usize) -> Vec<f64> {
        let body = &self.body[slot * self.v..(slot + 1) * self.v];
        let m: Vec<f64> = self.w1[previous * self.r..(previous + 1) * self.r].iter().map(|&x| bf16(x)).collect();
        self.w2.chunks_exact(self.r).zip(body).map(|(row, &l)| l as f64 + row.iter().zip(&m).map(|(&x, &y)| bf16(x) * y).sum::<f64>()).collect()
    }
    fn device(&self, gpu: &mut Backend) -> Device {
        let features = self.w + self.r;
        Device {
            body: upload(gpu, &self.body), hidden: upload(gpu, &self.hidden),
            w1: upload(gpu, &self.w1), w2: upload(gpu, &self.w2), conf: upload(gpu, &self.conf),
            logits: gpu.buffer((self.body.len() * 4) as u64).unwrap(),
            tokens: gpu.buffer(20).unwrap(), confidences: gpu.buffer(20).unwrap(), count: gpu.buffer(4).unwrap(),
            scratch: gpu.buffer((5 * (features * 4).div_ceil(16) * 16 + 512) as u64).unwrap(),
        }
    }
}
struct Device {
    body: Buffer, hidden: Buffer, w1: Buffer, w2: Buffer, conf: Buffer,
    logits: Buffer, tokens: Buffer, confidences: Buffer, count: Buffer, scratch: Buffer,
}
impl Device {
    fn run(&self, gpu: &mut Backend, case: &Case, slots: u64, threshold: f32, minimum: u64) -> Result<(), String> {
        self.run_at(gpu, case, slots, threshold, minimum, None)
    }
    /// With the anchor token from a device i32.
    fn run_at(&self, gpu: &mut Backend, case: &Case, slots: u64, threshold: f32, minimum: u64, anchor: Option<&Buffer>)
              -> Result<(), String> {
        gpu.dspark_head(self.body.base(), None, self.hidden.base(), &self.w1, &self.w2, &self.conf,
            slots, case.v as u64, case.r as u64, case.w as u64, case.anchor, anchor.map(|at| at.base().cast_const()), None, threshold, minimum,
            self.logits.base(), self.tokens.base(), self.confidences.base(), self.count.base(), &self.scratch)
    }
    fn check(&self, gpu: &mut Backend, case: &Case, expected: &Expected, slots: usize, count: u32) {
        assert_eq!(read::<u32>(gpu, &self.count, 1), [count]);
        assert_eq!(read::<i32>(gpu, &self.tokens, slots), expected.tokens[..slots]);
        let confidences = read::<f32>(gpu, &self.confidences, slots);
        for (i, (&actual, &reference)) in confidences.iter().zip(&expected.confidences).enumerate() {
            assert!((actual as f64 - reference).abs() <= 1e-4, "confidence slot {i}: {actual} vs {reference}");
        }
        let logits = read::<f32>(gpu, &self.logits, slots * case.v);
        let mut worst = 0f64;
        for (i, (&actual, &reference)) in logits.iter().zip(&expected.logits).enumerate() {
            let error = (actual as f64 - reference).abs() / (1.0 + reference.abs());
            worst = worst.max(error);
            assert!(error <= 2e-5, "logit {i}: {actual} vs {reference}, scaled error {error}");
        }
        eprintln!("B={slots} V={} R={} W={} count={count}, max abs/(1+abs(ref))={worst:e}, confidence={confidences:?}", case.v, case.r, case.w);
    }
}

#[test]
fn production_shape_b1_to_b5_matches_cpu_and_prefix_limits() {
    let (v, r, w) = (129280, 256, 5120);
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let mut case = Case { v, r, w, anchor: 128799, body: random.floats(5 * v, 0.05),
        hidden: random.floats(5 * w, 0.1), w1: random.bf16(v * r), w2: random.bf16(v * r),
        conf: random.floats(w + r, 0.001) };
    case.conf[0] = 1.0;
    for slot in 0..5 { case.hidden[slot * w] = 3.0 - slot as f32; }
    let expected = case.oracle();
    assert_eq!(expected.confidences.iter().take_while(|&&c| c >= 0.7).count(), 3);
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let device = case.device(&mut gpu);
    // Capture the first invocation, so hidden host initialization cannot be masked by a warm-up.
    gpu.graph_begin().unwrap();
    device.run(&mut gpu, &case, 5, 0.0, 0).unwrap();
    gpu.graph_end(137, true).unwrap();
    assert!(gpu.graph_run(137).unwrap());
    device.check(&mut gpu, &case, &expected, 5, 5);
    // Replay with changed body inputs: every logit, token and confidence must be recomputed.
    let original_body = case.body.clone();
    for row in case.body.chunks_exact_mut(v) { row[17] += 20.0; }
    gpu.write_buffer(&device.body, 0, &case.body).unwrap();
    let changed = case.oracle();
    assert_ne!(changed.tokens, expected.tokens);
    assert!(gpu.graph_run(137).unwrap());
    device.check(&mut gpu, &case, &changed, 5, 5);
    case.body = original_body;
    gpu.write_buffer(&device.body, 0, &case.body).unwrap();
    // The anchor token from a device i32: a recorded graph follows it as it changes.
    let anchor = upload(&mut gpu, &[case.anchor as i32]);
    gpu.graph_begin().unwrap();
    device.run_at(&mut gpu, &case, 5, 0.0, 0, Some(&anchor)).unwrap();
    gpu.graph_end(138, true).unwrap();
    assert!(gpu.graph_run(138).unwrap());
    device.check(&mut gpu, &case, &expected, 5, 5);
    let moved = Case { anchor: 4242, ..case };
    gpu.write_buffer(&anchor, 0, &[moved.anchor as i32]).unwrap();
    let moved_expected = moved.oracle();
    assert!(gpu.graph_run(138).unwrap());
    device.check(&mut gpu, &moved, &moved_expected, 5, 5);
    let case = Case { anchor: 128799, ..moved };
    for slots in 1..=5 {
        device.run(&mut gpu, &case, slots, 0.0, 0).unwrap();
        device.check(&mut gpu, &case, &expected, slots as usize, slots as u32);
    }
    device.run(&mut gpu, &case, 5, 0.7, 0).unwrap();
    device.check(&mut gpu, &case, &expected, 5, 3);
    device.run(&mut gpu, &case, 5, 0.7, 4).unwrap();
    device.check(&mut gpu, &case, &expected, 5, 0);
    device.run(&mut gpu, &case, 2, 0.0, 3).unwrap();
    device.check(&mut gpu, &case, &expected, 2, 0);
}

#[test]
fn ties_odd_vocabulary_cutoff_and_invalid_arguments() {
    let mut case = Case { v: 33, r: 8, w: 32, anchor: 32, body: vec![0.0; 5 * 33], hidden: vec![0.0; 5 * 32],
        w1: vec![0; 33 * 8], w2: vec![0; 33 * 8], conf: vec![0.0; 40] };
    for slot in 0..5 { case.body[slot * 33] = 1.0; case.body[slot * 33 + 32] = 1.0; }
    let expected = case.oracle();
    assert_eq!(expected.tokens, [0; 5]);
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let device = case.device(&mut gpu);
    for (threshold, count) in [(0.5, 5), (0.5001, 0)] {
        device.run(&mut gpu, &case, 5, threshold, 0).unwrap();
        device.check(&mut gpu, &case, &expected, 5, count);
    }
    for minimum in [6, u64::MAX] {
        device.run(&mut gpu, &case, 5, 0.0, minimum).unwrap();
        device.check(&mut gpu, &case, &expected, 5, 0);
    }
    for (slots, threshold, minimum) in [(0, 0.0, 0), (6, 0.0, 0), (1, f32::NAN, 0),
        (1, -0.1, 0), (1, 1.1, 0)] {
        assert!(device.run(&mut gpu, &case, slots, threshold, minimum).is_err());
    }
    case.anchor = 33;
    assert!(device.run(&mut gpu, &case, 1, 0.0, 0).is_err());
    case.anchor = 32;
    // (a slot's features, 16-byte aligned, and the argmax's 512 bytes)
    let one_slot = (case.w + case.r) * 4;
    let short = Device { scratch: device.scratch.slice(0, (one_slot.div_ceil(16) * 16 + 512 - 1) as u64), ..device };
    assert!(short.run(&mut gpu, &case, 1, 0.0, 0).is_err());
    let two = Device { scratch: device.scratch.slice(0, (2 * one_slot.div_ceil(16) * 16 + 512 - 1) as u64), ..short };
    assert!(two.run(&mut gpu, &case, 2, 0.0, 0).is_err());
}

#[test]
fn a_body_in_two_parts_is_the_body_whole() {
    let (v, r, w, split) = (1031, 16, 64, 500);
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    let case = Case { v, r, w, anchor: 7, body: random.floats(5 * v, 1.0), hidden: random.floats(5 * w, 0.1), w1: random.bf16(v * r),
        w2: random.bf16(v * r), conf: random.floats(w + r, 0.05) };
    let expected = case.oracle();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let device = case.device(&mut gpu);
    // the head's columns from `split` on in a buffer of their own, those below in the body's rows (the rest of them garbage)
    let mut front = case.body.clone();
    let mut tail = Vec::new();
    for row in front.chunks_exact_mut(v) {
        tail.extend_from_slice(&row[split..]);
        row[split..].fill(f32::NAN);
    }
    let (front, tail) = (upload(&mut gpu, &front), upload(&mut gpu, &tail));
    for slots in 1..=5u64 {
        gpu.dspark_head(front.base(), Some((tail.base().cast_const(), split as u64)), device.hidden.base(), &device.w1, &device.w2,
                        &device.conf, slots, v as u64, r as u64, w as u64, case.anchor, None, None, 0.0, 0, device.logits.base(),
                        device.tokens.base(), device.confidences.base(), device.count.base(), &device.scratch).unwrap();
        device.check(&mut gpu, &case, &expected, slots as usize, slots as u32);
    }
    for first in [0, v as u64] {
        assert!(gpu.dspark_head(front.base(), Some((tail.base().cast_const(), first)), device.hidden.base(), &device.w1, &device.w2,
                                &device.conf, 1, v as u64, r as u64, w as u64, case.anchor, None, None, 0.0, 0, device.logits.base(),
                                device.tokens.base(), device.confidences.base(), device.count.base(), &device.scratch).is_err());
    }
}

fn softmax(logits: &[f64], inverse_temperature: f64) -> Vec<f64> {
    let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|&l| ((l - peak) * inverse_temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

fn chi_square(counts: &[usize], probabilities: &[f64]) -> f64 {
    let total: usize = counts.iter().sum();
    counts.iter().zip(probabilities).filter(|(_, &p)| p * total as f64 > 5.0)
        .map(|(&c, &p)| (c as f64 - total as f64 * p).powi(2) / (total as f64 * p)).sum()
}

/// With sample_at the head's tokens are draws: slot 0's from the softmax of its corrected logits at the temperature, each later
/// slot's from the softmax of its logits after the token drawn before it (the Markov features follow the draw); a seed gives the same
/// tokens again, another gives others, and an inverse temperature of 0 is the argmax of the greedy head.
#[test]
fn drawn_tokens_follow_the_softmax_of_the_corrected_logits() {
    let (v, r, w, slots) = (12, 8, 32, 3u64);
    let mut random = Random(0x1234_5678_9abc_def1);
    let case = Case { v, r, w, anchor: 5, body: random.floats(5 * v, 1.5), hidden: random.floats(5 * w, 0.1), w1: random.bf16(v * r),
        w2: random.bf16(v * r), conf: random.floats(w + r, 0.05) };
    let expected = case.oracle();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let device = case.device(&mut gpu);
    let sample = gpu.buffer(16).unwrap();
    let run = |gpu: &mut Backend, inverse: f32, seed: u64| -> Vec<i32> {
        gpu.write_buffer(&sample, 0, &[inverse.to_bits(), seed as u32, (seed >> 32) as u32, 0]).unwrap();
        gpu.dspark_head(device.body.base(), None, device.hidden.base(), &device.w1, &device.w2, &device.conf, slots, v as u64, r as u64, w as u64,
                        case.anchor, None, Some(sample.base().cast_const()), 0.0, 0, device.logits.base(), device.tokens.base(),
                        device.confidences.base(), device.count.base(), &device.scratch).unwrap();
        read::<i32>(gpu, &device.tokens, slots as usize)
    };
    // inverse temperature 0: the argmax chain
    assert_eq!(run(&mut gpu, 0.0, 99), expected.tokens[..slots as usize]);
    // the same seed, the same draws; others differ somewhere
    let first = run(&mut gpu, 1.0, 7);
    assert_eq!(run(&mut gpu, 1.0, 7), first);
    assert!((8..40).any(|seed| run(&mut gpu, 1.0, seed) != first));
    for (inverse, trials) in [(1.0f32, 12_000usize), (1.5, 12_000)] {
        let (mut first_counts, mut after) = (vec![0usize; v], vec![vec![0usize; v]; v]);
        for seed in 0..trials {
            let tokens = run(&mut gpu, inverse, 1000 + seed as u64);
            first_counts[tokens[0] as usize] += 1;
            after[tokens[0] as usize][tokens[1] as usize] += 1;
        }
        let p0 = softmax(&case.logits_after(0, case.anchor as usize), f64::from(inverse));
        // (11 degrees of freedom: 31.3 is the 0.001 point)
        let chi = chi_square(&first_counts, &p0);
        assert!(chi < 31.3, "slot 0 at {inverse}: chi-square {chi}, counts {first_counts:?}, expected {p0:?}");
        // slot 1 after the two commonest first tokens
        let mut order: Vec<usize> = (0..v).collect();
        order.sort_by_key(|&t| std::cmp::Reverse(first_counts[t]));
        for &t0 in &order[..2] {
            let p1 = softmax(&case.logits_after(1, t0), f64::from(inverse));
            let chi = chi_square(&after[t0], &p1);
            assert!(chi < 31.3, "slot 1 after {t0} at {inverse}: chi-square {chi}, counts {:?}, expected {p1:?}", after[t0]);
        }
    }
    // the corrected logits the draws were from are those of the drawn chain
    let tokens = run(&mut gpu, 1.0, 31);
    let logits = read::<f32>(&mut gpu, &device.logits, slots as usize * v);
    for slot in 1..slots as usize {
        let reference = case.logits_after(slot, tokens[slot - 1] as usize);
        for (i, (&actual, &want)) in logits[slot * v..(slot + 1) * v].iter().zip(&reference).enumerate() {
            assert!((actual as f64 - want).abs() <= 2e-5 * (1.0 + want.abs()), "slot {slot} logit {i}: {actual} vs {want}");
        }
    }
}
