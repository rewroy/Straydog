//! The formats' kernels against the host decoders (formats.rs, themselves checked against ggml's
//! definitions in tests/formats.rs), on DeepSeek-V4.1's real tensors: the GPU's linear products
//! (sd_op_linear: GEMV for up to four inputs, an F32 GEMM over expanded rows beyond).
//!
//! Every weight is checked exactly: one-hot inputs make each product return single decoded values,
//! column by column, through the GEMV and through the expansion. Random inputs then check the sums
//! against an F64 reference within the bound of F32 summation alone: n u sum|terms| for n terms
//! (u = 2^-24), the terms being the decoded weights times the inputs. The GEMV sums codes and
//! subtracts biases or minimums per group, terms of up to about three times a weight's size, so
//! its bound takes 4 n u.
//!
//! cargo test --release -p straydog-engine --features hip --test formats_kernels, under the GPU
//! lock, with STRAYDOG_DEEPSEEK set to the directory of the DeepSeek-V4.1-Flash UD-IQ2_XXS shards
//! (the cases read about 60 MB of it).
#![cfg(any(feature = "hip", feature = "cuda"))]

use std::ffi::c_void;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use straydog_engine::formats;
use straydog_gguf::{ModelHeader, TensorInfo};
use straydog_gpu::{ffi, Device};

/// xorshift64*: generated data is reproducible.
struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Roughly normal activations with a few large ones, as real hidden states have.
    fn activation(&mut self) -> f32 {
        let value = (0..4).map(|_| self.uniform()).sum::<f64>() - 2.0;
        (if self.next().is_multiple_of(64) { value * 20.0 } else { value }) as f32
    }
}

fn decode(kind: u32, data: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    formats::decode_into(kind, data, &mut out).unwrap();
    out
}

fn load_tables() {
    straydog_engine::gpu::load_iq_tables().unwrap();
}

/// Each output against F64 sums of the decoded rows: within `factor` n u sum|w x|.
fn check_sums(label: &str, actual: &[f32], weights: &[f32], cols: usize, x: &[f32], factor: f64) {
    for (row, value) in actual.iter().enumerate() {
        let w = &weights[row * cols..(row + 1) * cols];
        let exact: f64 = w.iter().zip(x).map(|(w, x)| f64::from(*w) * f64::from(*x)).sum();
        let magnitude: f64 = w.iter().zip(x).map(|(w, x)| (f64::from(*w) * f64::from(*x)).abs()).sum();
        let bound = factor * cols as f64 * 2f64.powi(-24) * magnitude + 1e-30;
        assert!((f64::from(*value) - exact).abs() <= bound, "{label} row {row}: {value} vs {exact} (bound {bound:e})");
    }
}

/// The DeepSeek-V4.1 shards: tensors by name.
struct Shards {
    files: Vec<File>,
    headers: Vec<ModelHeader>,
}

impl Shards {
    fn open() -> Option<Shards> {
        let Some(root) = std::env::var_os("STRAYDOG_DEEPSEEK") else {
            eprintln!("skipped: set STRAYDOG_DEEPSEEK to the DeepSeek-V4.1-Flash UD-IQ2_XXS directory");
            return None;
        };
        let mut paths: Vec<PathBuf> = std::fs::read_dir(root).unwrap().map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "gguf")).collect();
        paths.sort();
        let mut files = Vec::new();
        let mut headers = Vec::new();
        for path in paths {
            let mut file = File::open(&path).unwrap();
            headers.push(straydog_gguf::inspect(&mut file).unwrap());
            files.push(file);
        }
        Some(Shards { files, headers })
    }

    fn tensor(&self, name: &str) -> (usize, &TensorInfo) {
        self.headers.iter().enumerate()
            .find_map(|(file, header)| header.tensors.iter().find(|tensor| tensor.name == name).map(|tensor| (file, tensor)))
            .unwrap_or_else(|| panic!("no tensor {name}"))
    }

    /// Experts `which` of a [cols, rows, experts] tensor (or the whole 2D tensor), back to back.
    fn experts(&self, name: &str, which: &[usize]) -> (u32, Vec<u8>) {
        let (file, tensor) = self.tensor(name);
        let count = tensor.dimensions.get(2).copied().unwrap_or(1) as usize;
        let per = tensor.bytes as usize / count;
        let mut data = vec![0u8; per * which.len()];
        for (slot, expert) in which.iter().enumerate() {
            self.files[file].read_exact_at(&mut data[slot * per..(slot + 1) * per], tensor.offset + (expert * per) as u64)
                .unwrap();
        }
        (tensor.quant_type, data)
    }

    /// One matrix (an expert's, or a 2D tensor) with its shape: rows, cols.
    fn matrix(&self, name: &str, expert: usize) -> (u32, usize, usize, Vec<u8>) {
        let dims = self.tensor(name).1.dimensions.clone();
        let (kind, data) = self.experts(name, &[expert]);
        (kind, dims[1] as usize, dims[0] as usize, data)
    }
}

/// A device buffer of `bytes`, written from `data` when given.
fn buffer(device: &Device, bytes: usize, data: Option<&[f32]>) -> *mut c_void {
    // SAFETY: a live device; null on failure.
    let pointer = unsafe { ffi::sd_buffer(device.handle(), bytes as u64) };
    assert!(!pointer.is_null(), "{}", straydog_gpu::last_error().0);
    if let Some(data) = data {
        // SAFETY: the buffer holds `bytes`, the data as many.
        straydog_gpu::check(unsafe { ffi::sd_buffer_write(device.handle(), pointer, 0, data.as_ptr().cast(), bytes as u64) })
            .unwrap();
    }
    pointer
}

fn read(device: &Device, pointer: *mut c_void, count: usize) -> Vec<f32> {
    let mut values = vec![0f32; count];
    // SAFETY: the buffer holds `count` floats.
    straydog_gpu::check(unsafe {
        ffi::sd_buffer_read(device.handle(), pointer, 0, values.as_mut_ptr().cast(), count as u64 * 4)
    }).unwrap();
    values
}

/// sd_op_linear over `steps` inputs `input_at` values into `input`, into `output` at `output_at`.
#[allow(clippy::too_many_arguments)]
fn linear(device: &Device, weights: *mut c_void, kind: u32, rows: usize, cols: usize, input: *mut c_void, input_at: usize,
          steps: usize, output: *mut c_void, output_at: usize, scratch: *mut c_void, scratch_bytes: usize) {
    // SAFETY: every range lies in this device's allocations, of the shapes given.
    straydog_gpu::check(unsafe {
        ffi::sd_op_linear(device.handle(), weights, kind as i32, rows as u64, cols as u64, 0,
                          input.cast::<f32>().add(input_at), cols as u64, steps as u64,
                          output.cast::<f32>().add(output_at), rows as u64, scratch.cast(), scratch_bytes as u64)
    }).unwrap();
}

/// Every weight of a matrix through the GEMV (four one-hot inputs at a time) and through the GEMM's
/// expansion (all at once), then random inputs through GEMVs of one to four inputs and the GEMM.
fn check_linear(device: &Device, label: &str, kind: u32, rows: usize, cols: usize, data: &[u8]) {
    let weights = decode(kind, data);
    let matrix = device.upload(data).unwrap();
    let pointer = matrix.pointer();
    let mut identity = vec![0f32; cols * cols];
    for i in 0..cols {
        identity[i * cols + i] = 1.0;
    }
    let input = buffer(device, cols * cols * 4, Some(&identity));
    drop(identity);
    let output = buffer(device, cols * rows * 4, None);
    let scratch_bytes = rows * cols * 4;
    let scratch = buffer(device, scratch_bytes, None);
    let exact = |name: &str, values: &[f32]| {
        for column in 0..cols {
            for row in 0..rows {
                assert_eq!(values[column * rows + row], weights[row * cols + column], "{label} {name}: row {row} col {column}");
            }
        }
    };
    for first in (0..cols).step_by(4) {
        linear(device, pointer, kind, rows, cols, input, first * cols, 4.min(cols - first), output, first * rows, scratch,
               scratch_bytes);
    }
    exact("GEMV", &read(device, output, cols * rows));
    linear(device, pointer, kind, rows, cols, input, 0, cols, output, 0, scratch, scratch_bytes);
    exact("GEMM", &read(device, output, cols * rows));
    let mut random = Random(cols as u64 ^ rows as u64);
    let x: Vec<f32> = (0..40 * cols).map(|_| random.activation()).collect();
    // SAFETY: the input buffer holds cols * cols floats, at least 40 * cols.
    straydog_gpu::check(unsafe { ffi::sd_buffer_write(device.handle(), input, 0, x.as_ptr().cast(), x.len() as u64 * 4) })
        .unwrap();
    for steps in [1, 2, 3, 4, 40] {
        linear(device, pointer, kind, rows, cols, input, 0, steps, output, 0, scratch, scratch_bytes);
        let values = read(device, output, steps * rows);
        for step in 0..steps {
            check_sums(&format!("{label} {steps} inputs, input {step}"), &values[step * rows..(step + 1) * rows], &weights, cols,
                       &x[step * cols..(step + 1) * cols], 4.0);
        }
    }
    for pointer in [input, output, scratch] {
        // SAFETY: allocated above, released once.
        unsafe { ffi::sd_release(device.handle(), pointer) };
    }
    eprintln!("{label}: every weight exact through GEMV and GEMM, sums within bounds");
}

#[test]
fn gpu_linear_products_decode_every_type_of_the_file_exactly() {
    let Some(shards) = Shards::open() else { return };
    load_tables();
    let device = Device::new(0, 2 << 30).unwrap();
    // Every type of the file: the experts' IQ1_S, Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS; the shared
    // experts' Q8_0 and Q5_K; attention's Q6_K and BF16; the router's F32.
    for (name, expert) in [
        ("blk.0.ffn_gate_exps.weight", 3), ("blk.0.ffn_down_exps.weight", 3), ("blk.10.ffn_up_exps.weight", 1),
        ("blk.1.ffn_down_exps.weight", 7), ("blk.25.ffn_gate_exps.weight", 100), ("blk.20.ffn_down_exps.weight", 9),
        ("blk.0.ffn_gate_shexp.weight", 0), ("blk.27.ffn_down_shexp.weight", 0), ("blk.27.attn_kv.weight", 0),
        ("blk.2.attn_compressor_kv.weight", 0), ("blk.0.ffn_gate_inp.weight", 0),
    ] {
        let (kind, rows, cols, data) = shards.matrix(name, expert);
        check_linear(&device, &format!("{name} [{kind}]"), kind, rows, cols, &data);
    }
}
