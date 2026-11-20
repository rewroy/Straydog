//! A long pass's hyper-connection step (op_hc_step_rows_kernel, then op_hc_step_finish_kernel: 32 rows or more, unlinked, from the
//! sublayer's output and with the weights) against the same rows one at a time (op_hc_step_kernel): the same bits of the stream,
//! the coefficients and the collapsed rows. DeepSeek-V4.1's four streams; 256 rows (a prompt pass's batch) and 45 (a block of rows
//! cut short), a width of 5120 (DeepSeek-V4.1 Flash's) and 1000 (a block of columns cut short); F32 and BF16 weights.
//!
//! cargo test --release -p straydog-engine --features hip --test hc_step_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;

const HC: usize = 4;
const MIX: usize = (2 + HC) * HC;

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
}

fn upload(gpu: &mut Backend, values: &[f32]) -> Buffer {
    let buffer = gpu.buffer((values.len() * 4) as u64).unwrap();
    gpu.write_buffer(&buffer, 0, values).unwrap();
    buffer
}

fn read(gpu: &mut Backend, buffer: &Buffer, count: usize) -> Vec<u32> {
    let mut values = vec![0f32; count];
    gpu.read_buffer(buffer, 0, &mut values).unwrap();
    values.iter().map(|value| value.to_bits()).collect()
}

/// The step's outputs: the stream, pre, post, comb and the collapsed rows.
struct Outputs {
    stream: Buffer,
    pre: Buffer,
    post: Buffer,
    comb: Buffer,
    out: Buffer,
    workspace: Buffer,
}

impl Outputs {
    fn new(gpu: &mut Backend, rows: usize, width: usize) -> Outputs {
        Outputs {
            stream: upload(gpu, &vec![0.0; rows * HC * width]),
            pre: upload(gpu, &vec![0.0; rows * HC]),
            post: upload(gpu, &vec![0.0; rows * HC]),
            comb: upload(gpu, &vec![0.0; rows * HC * HC]),
            out: upload(gpu, &vec![0.0; rows * width]),
            workspace: upload(gpu, &vec![0.0; 4 << 20]),
        }
    }
}

fn steps_match(rows: usize, width: usize, bf16: bool) {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    let mut random = Random(0x2545_f491_4f6c_dd1d ^ (rows * width) as u64 ^ u64::from(bf16));
    let x = upload(&mut gpu, &random.floats(rows * width, 2.0));
    let residual = upload(&mut gpu, &random.floats(rows * HC * width, 2.0));
    let post_in = upload(&mut gpu, &random.floats(rows * HC, 1.0));
    let comb_in = upload(&mut gpu, &random.floats(rows * HC * HC, 0.5));
    let weights = random.floats(MIX * HC * width, 0.01);
    let fn_ = if bf16 {
        let halves: Vec<u8> = weights.iter().flat_map(|w| ((w.to_bits() >> 16) as u16).to_le_bytes()).collect();
        gpu.upload_bytes(&halves).unwrap()
    } else {
        upload(&mut gpu, &weights)
    };
    // (BF16 weights tagged: bit 0 of their address, as the engine passes them)
    let fn_at = if bf16 { fn_.base().wrapping_byte_add(1).cast_const() } else { fn_.base().cast_const() };
    let base = upload(&mut gpu, &random.floats(MIX, 0.5));
    let scale = upload(&mut gpu, &random.floats(3, 0.5));
    let collapse = upload(&mut gpu, &random.floats(rows * HC, 1.0));
    let norm = upload(&mut gpu, &random.floats(width, 1.0));
    let (whole, single) = (Outputs::new(&mut gpu, rows, width), Outputs::new(&mut gpu, rows, width));
    let at = |buffer: &Buffer, values: usize| buffer.base().wrapping_byte_add(values * 4);
    // the long pass's kernels: every row at once
    gpu.hc_step(Some((x.base().cast_const(), residual.base().cast_const(), post_in.base().cast_const(), comb_in.base().cast_const())),
                whole.stream.base(), Some((fn_at, base.base().cast_const(), scale.base().cast_const(), whole.pre.base(), whole.post.base(),
                whole.comb.base())), 20, 1e-6, 1e-6, collapse.base().cast_const(), norm.base().cast_const(), rows as u64, HC as u64,
                width as u64, &whole.workspace, whole.out.base(), None).unwrap();
    // a row at a time
    for row in 0..rows {
        gpu.hc_step(Some((at(&x, row * width).cast_const(), at(&residual, row * HC * width).cast_const(), at(&post_in, row * HC).cast_const(),
                          at(&comb_in, row * HC * HC).cast_const())),
                    at(&single.stream, row * HC * width), Some((fn_at, base.base().cast_const(), scale.base().cast_const(),
                    at(&single.pre, row * HC), at(&single.post, row * HC), at(&single.comb, row * HC * HC))), 20, 1e-6, 1e-6,
                    at(&collapse, row * HC).cast_const(), norm.base().cast_const(), 1, HC as u64, width as u64, &single.workspace,
                    at(&single.out, row * width), None).unwrap();
    }
    for (label, a, b, count) in [("stream", &whole.stream, &single.stream, rows * HC * width), ("pre", &whole.pre, &single.pre, rows * HC),
                                 ("post", &whole.post, &single.post, rows * HC), ("comb", &whole.comb, &single.comb, rows * HC * HC),
                                 ("out", &whole.out, &single.out, rows * width)] {
        let (a, b) = (read(&mut gpu, a, count), read(&mut gpu, b, count));
        let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert_eq!(differ, 0, "{rows} rows of {width}, BF16 weights {bf16}: {differ} of {count} {label} values differ");
        assert!(a.iter().any(|&bits| bits != 0), "{label}: all zero");
    }
}

#[test]
fn a_long_pass_step_matches_its_rows_one_at_a_time() {
    for (rows, width) in [(256, 5120), (45, 5120), (256, 1000), (45, 1000)] {
        for bf16 in [false, true] {
            steps_match(rows, width, bf16);
        }
    }
}
