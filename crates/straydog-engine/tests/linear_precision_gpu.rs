//! Per-op GEMM precision, strided/sliced matrices and the GEMV boundary.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::{formats, gpu::Backend, ops::Buffer};

fn round_f16(x: f32) -> f32 {
    let magnitude = x.abs();
    if magnitude >= 65520.0 { return f32::INFINITY.copysign(x); }
    let exponent = if magnitude < 2f32.powi(-14) { -24 } else { ((magnitude.to_bits() >> 23) as i32 - 127) - 10 };
    let quantum = 2f64.powi(exponent);
    ((f64::from(magnitude) / quantum).round_ties_even() * quantum).copysign(f64::from(x)) as f32
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len()).step_by(2).map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap()).collect()
}

fn read(gpu: &mut Backend, buffer: &Buffer, count: usize) -> Vec<f32> {
    let mut values = vec![0f32; count];
    gpu.read_buffer(buffer, 0, &mut values).unwrap();
    values
}

#[test]
fn explicit_precision_preserves_gemv_strides_and_device_defaults() {
    straydog_engine::gpu::load_iq_tables().unwrap();
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    gpu.enable_gemm_checks().unwrap();
    gpu.configure_gemm(2, true).unwrap();
    let golden: serde_json::Value = serde_json::from_str(include_str!("golden/format_blocks.json")).unwrap();
    let (rows, cols, first) = (3usize, 256usize, 1usize);
    let sentinel = -9999.25f32;
    for kind in [0u32, 8, 17, 22] {
        let bytes: Vec<u8> = if kind == 0 {
            (0..(rows + first) * cols).flat_map(|i| (0.2345678f32 + i as f32 * 0.000123).to_le_bytes()).collect()
        } else {
            let blocks: Vec<_> = golden["blocks"].as_array().unwrap().iter()
                .filter(|v| v["kind"].as_u64() == Some(u64::from(kind)))
                .map(|v| hex(v["data"].as_str().unwrap())).collect();
            assert!(!blocks.is_empty(), "missing kind{kind}");
            let (elements, _) = formats::block(kind).unwrap();
            (0..(rows + first) * cols / elements).flat_map(|i| blocks[i % blocks.len()].clone()).collect()
        };
        let mut decoded = Vec::new();
        formats::decode_into(kind, &bytes, &mut decoded).unwrap();
        let weights = gpu.upload_bytes(&bytes).unwrap();
        for steps in [1usize, 4, 5] {
            let (input_stride, output_stride) = (cols + 8, rows + 5);
            let mut x = vec![f32::NAN; steps * input_stride];
            for t in 0..steps {
                x[t * input_stride..t * input_stride + cols].fill(0.0);
                x[t * input_stride + t] = 1.0;
            }
            let input = gpu.buffer((x.len() * 4) as u64).unwrap();
            gpu.write_buffer(&input, 0, &x).unwrap();
            let f32_out = gpu.buffer((steps * output_stride * 4) as u64).unwrap();
            let half_out = gpu.buffer((steps * output_stride * 4) as u64).unwrap();
            let legacy_out = gpu.buffer((steps * output_stride * 4) as u64).unwrap();
            for out in [&f32_out, &half_out, &legacy_out] {
                gpu.write_buffer(out, 0, &vec![sentinel; steps * output_stride]).unwrap();
            }
            let half_input_bytes = (steps * cols * 2).div_ceil(256) * 256;
            // Exactly one expanded row fits: exercise multiple pieces and input/scratch separation.
            let scratch = gpu.buffer((half_input_bytes + cols * 2) as u64).unwrap();
            gpu.linear_gemm(&weights, kind, rows as u64, cols as u64, first as u64, input.base(), input_stride as u64,
                            steps as u64, f32_out.base(), output_stride as u64, &scratch, false).unwrap();
            gpu.linear_gemm(&weights, kind, rows as u64, cols as u64, first as u64, input.base(), input_stride as u64,
                            steps as u64, half_out.base(), output_stride as u64, &scratch, true).unwrap();
            // The explicit F32 call must not change the device's existing FP16 default.
            gpu.linear(&weights, kind, rows as u64, cols as u64, first as u64, input.base(), input_stride as u64,
                       steps as u64, legacy_out.base(), output_stride as u64, &scratch).unwrap();
            let a = read(&mut gpu, &f32_out, steps * output_stride);
            let b = read(&mut gpu, &half_out, steps * output_stride);
            let c = read(&mut gpu, &legacy_out, steps * output_stride);
            assert_eq!(b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                       "device policy changed: kind{kind} steps{steps}");
            for t in 0..steps {
                for row in 0..output_stride {
                    let expected = if row < rows { decoded[(first + row) * cols + t] } else { sentinel };
                    assert_eq!(a[t * output_stride + row].to_bits(), expected.to_bits(), "F32 kind{kind} steps{steps} row{row}");
                    let expected = if row < rows && steps > 4 { round_f16(expected) } else { expected };
                    assert_eq!(b[t * output_stride + row].to_bits(), expected.to_bits(), "FP16 kind{kind} steps{steps} row{row}");
                }
            }
            if steps <= 4 {
                assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            } else {
                if kind == 0 { assert!(a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits())); }
                let too_small = gpu.buffer(half_input_bytes as u64).unwrap();
                assert!(gpu.linear_gemm(&weights, kind, rows as u64, cols as u64, first as u64, input.base(), input_stride as u64,
                                       steps as u64, half_out.base(), output_stride as u64, &too_small, true)
                    .unwrap_err().contains("scratch holds no row"));
                gpu.release_buffer(too_small).unwrap();
            }
            for buffer in [input, f32_out, half_out, legacy_out, scratch] { gpu.release_buffer(buffer).unwrap(); }
        }
        gpu.release_buffer(weights).unwrap();
    }
    let counts = gpu.gemm_checks().unwrap();
    assert_eq!([counts[0], counts[1], counts[2], counts[3], counts[4]], [0; 5], "operand checks: {counts:?}");
    assert!(counts[5] > 0 && counts[6] > 0);
    assert_eq!(f32::from_bits(counts[7] as u32), 1.0, "padding NaNs must not be observed");
}

#[test]
fn cast_checks_detect_finite_input_overflow() {
    let mut gpu = Backend::new(0, 1 << 30).unwrap();
    gpu.enable_gemm_checks().unwrap();
    let weights = gpu.buffer(2 * 256 * 4).unwrap();
    gpu.write_buffer(&weights, 0, &vec![0.25f32; 2 * 256]).unwrap();
    let input = gpu.buffer(5 * 256 * 4).unwrap();
    let mut values = vec![1f32; 5 * 256];
    values[0] = 1e10;
    gpu.write_buffer(&input, 0, &values).unwrap();
    let output = gpu.buffer(5 * 2 * 4).unwrap();
    let scratch = gpu.buffer(1 << 20).unwrap();
    gpu.linear_gemm(&weights, 0, 2, 256, 0, input.base(), 0, 5, output.base(), 0, &scratch, true).unwrap();
    let counts = gpu.gemm_checks().unwrap();
    assert_eq!(counts[0], 0, "the source value was finite");
    assert_eq!(counts[1], 1);
    assert_eq!(counts[2], 1, "one finite activation overflows FP16");
    assert_eq!(counts[4], 0, "finite weights");
}
