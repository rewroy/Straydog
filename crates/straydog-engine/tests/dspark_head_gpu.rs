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
    fn device(&self, gpu: &mut Backend) -> Device {
        let features = self.w + self.r;
        Device {
            body: upload(gpu, &self.body), hidden: upload(gpu, &self.hidden),
            w1: upload(gpu, &self.w1), w2: upload(gpu, &self.w2), conf: upload(gpu, &self.conf),
            logits: gpu.buffer((self.body.len() * 4) as u64).unwrap(),
            tokens: gpu.buffer(20).unwrap(), confidences: gpu.buffer(20).unwrap(), count: gpu.buffer(4).unwrap(),
            scratch: gpu.buffer(((features * 4).div_ceil(16) * 16 + 512) as u64).unwrap(),
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
        gpu.dspark_head(self.body.base(), self.hidden.base(), &self.w1, &self.w2, &self.conf,
            slots, case.v as u64, case.r as u64, case.w as u64, case.anchor, anchor.map(|at| at.base().cast_const()), threshold, minimum,
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
    let short = Device { scratch: device.scratch.slice(0, device.scratch.bytes - 1), ..device };
    assert!(short.run(&mut gpu, &case, 1, 0.0, 0).is_err());
}
