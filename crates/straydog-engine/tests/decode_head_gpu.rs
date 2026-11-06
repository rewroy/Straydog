//! A decode step's head in one wait (Backend::decode_head: the HyperConnection mix and the vocabulary projection with no host round trip
//! for the mixed row) against the two calls it replaces (Backend::hyperconnection, then Backend::project_many), on random operands of
//! Qwen3.8's shapes: every logit the same bits, for the Q8 kernels and for the F32 ones, and a vector that is reused from step to step.
//!
//! cargo test --release -p straydog-engine --features hip --test decode_head_gpu (or --features cuda), under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::{Backend, Operand, HIDDEN, VOCABULARY, WIDTH};

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
    /// Random blocks of `kind` (Q8_0: 34 bytes of 32 values, Q6_K: 210 bytes of 256) with a small F16 scale.
    fn blocks(&mut self, kind: u32, values: usize) -> Vec<u8> {
        let (per, bytes, at) = if kind == 8 { (32, 34, 0) } else { (256, 210, 208) };
        let mut out: Vec<u8> = (0..values / per * bytes).map(|_| self.next() as u8).collect();
        for block in out.chunks_exact_mut(bytes) {
            block[at..at + 2].copy_from_slice(&0x2c00u16.to_le_bytes());
        }
        out
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn same_logits(wide_q8: bool, vocabulary: usize) {
    straydog_engine::gpu::load_iq_tables().unwrap();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    if wide_q8 {
        gpu.wide_q8_decode().unwrap();
    }
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    let norm: Vec<u8> = random.floats(WIDTH, 1.0).iter().flat_map(|value| (value + 1.0).to_le_bytes()).collect();
    // (the mix's down and up are Q8_0 on the device, as the engine uploads them)
    let down = random.blocks(8, 320 * WIDTH);
    let up = random.blocks(8, WIDTH * 320);
    let head = random.blocks(14, vocabulary * HIDDEN);
    let operands = [
        Operand::bytes("output_hc_norm.weight", &norm, 0, 1, WIDTH as u64),
        Operand::bytes("output_hc_down.weight:q8", &down, 8, 320, WIDTH as u64),
        Operand::bytes("output_hc_up.weight:q8", &up, 8, WIDTH as u64, 320),
        Operand::bytes("output.weight:None", &head, 14, vocabulary as u64, HIDDEN as u64),
    ];
    let mut fused = Vec::new();
    for step in 0..3 {
        let row = random.floats(WIDTH, 3.0);
        let mixed = gpu.hyperconnection("output_hc", &row, operands[0].clone(), &operands[1..3]).unwrap();
        let separate = gpu.project_many(&[(operands[3].clone(), &mixed[..])]).unwrap().remove(0);
        gpu.decode_head(&operands, &row, &mut fused).unwrap();
        assert_eq!(fused.len(), vocabulary);
        assert_eq!(bits(&fused), bits(&separate), "step {step}: the head's logits (wide Q8 {wide_q8}, {vocabulary} rows)");
    }
    // (a row that is not finite is refused, as the mix refuses it)
    let mut bad = random.floats(WIDTH, 3.0);
    bad[7] = f32::NAN;
    assert!(gpu.decode_head(&operands, &bad, &mut fused).is_err());
    // (and so are logits that are not: a NaN scale on the last row's blocks, found by the device's check, in the last of its blocks of values)
    let mut nan = random.blocks(14, 4096 * HIDDEN);
    let last = nan.len() / 210 - 1;
    nan[last * 210 + 208..last * 210 + 210].copy_from_slice(&0x7e00u16.to_le_bytes());
    let operands_nan = [operands[0].clone(), operands[1].clone(), operands[2].clone(),
                        Operand::bytes("output.weight:nan", &nan, 14, 4096, HIDDEN as u64)];
    let row = random.floats(WIDTH, 3.0);
    let error = gpu.decode_head(&operands_nan, &row, &mut fused).unwrap_err();
    assert!(error.contains("non-finite"), "{error}");
    // (the head works again with the good operands, from the same allocation)
    gpu.decode_head(&operands, &row, &mut fused).unwrap();
    assert_eq!(fused.len(), vocabulary);
}

#[test]
fn a_decode_step_head_in_one_wait_gives_the_logits_of_the_two_calls() {
    // (the whole vocabulary of Qwen3.8, and a part of it that the F32 kernels take as well)
    same_logits(true, VOCABULARY as usize);
    same_logits(true, 40960);
    same_logits(false, 40960);
}
