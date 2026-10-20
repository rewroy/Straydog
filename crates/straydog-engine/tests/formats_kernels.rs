//! The formats' kernels against the host decoders (formats.rs, themselves checked against ggml's
//! definitions in tests/formats.rs): the CPU expert engine's block decoders, row kernels and
//! experts; the GPU's linear products (sd_op_linear: GEMV for up to four inputs, an F32 GEMM over
//! expanded rows beyond); and the GPU's routed expert kernels (sd_expert_rows: the decode GEMV and
//! the RDNA4 grouped WMMA GEMM) at DeepSeek-V4.1's expert shapes. On its real tensors, and on
//! generated blocks (MXFP4, which no file on hand has).
//!
//! Every weight is checked exactly: one-hot inputs make each kernel return single decoded values
//! (F32, or rounded to FP16 for the WMMA GEMM, whose products of FP16 operands are exact in F32).
//! Random inputs then check the sums against an F64 reference within the bound of F32 summation
//! alone: n u sum|terms| for n terms (u = 2^-24), the terms being the decoded weights times the
//! inputs (rounded to FP16 for the WMMA GEMM). The GEMV and CPU kernels sum codes and subtract
//! biases or minimums per group, terms of up to about three times a weight's size, so their bound
//! takes 4 n u.
//!
//! cargo test --release -p straydog-engine --features hip --test formats_kernels, under the GPU
//! lock. The model cases read about 300 MB of DeepSeek-V4.1-Flash UD-IQ2_XXS: set
//! STRAYDOG_DEEPSEEK to its directory.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::ffi::c_void;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use serde_json::Value as Json;
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

/// The FP16 value nearest x (ties to even), as F32.
fn round_f16(x: f32) -> f32 {
    let magnitude = x.abs();
    if magnitude >= 65520.0 {
        return f32::INFINITY.copysign(x);
    }
    let exponent = if magnitude < 2f32.powi(-14) { -24 } else { ((magnitude.to_bits() >> 23) as i32 - 127) - 10 };
    let quantum = 2f64.powi(exponent);
    ((f64::from(magnitude) / quantum).round_ties_even() * quantum).copysign(f64::from(x)) as f32
}

fn decode(kind: u32, data: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    formats::decode_into(kind, data, &mut out).unwrap();
    out
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len()).step_by(2).map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap()).collect()
}

/// A generated block of `kind` with finite scales (MXFP4 exponents keep values in FP16's range).
fn generated_block(kind: u32, random: &mut Random) -> Vec<u8> {
    let (_, size) = formats::block(kind).unwrap();
    let mut block: Vec<u8> = (0..size).map(|_| random.next() as u8).collect();
    let mut half = |block: &mut Vec<u8>, at: usize| {
        let bits = (0x1c00 + (random.next() % 0x1800) as u16) | if random.next().is_multiple_of(2) { 0 } else { 0x8000 };
        block[at..at + 2].copy_from_slice(&bits.to_le_bytes());
    };
    match kind {
        10 => { half(&mut block, 80); half(&mut block, 82); }
        39 => block[0] = 118 + (random.next() % 8) as u8,
        29 => {
            let bits = 0x1c00 + (random.next() % 0x1800) as u16;
            for k in 0..4 {
                block[49 + 2 * k] = (block[49 + 2 * k] & 0x0f) | (((bits >> (4 * k)) & 15) << 4) as u8;
            }
        }
        _ => half(&mut block, 0),
    }
    block
}

/// The CPU expert engine's formats.
const CPU_KINDS: [u32; 12] = [8, 10, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39];

fn load_tables() {
    straydog_engine::gpu::load_iq_tables().unwrap();
}


#[test]
fn cpu_block_decoders_equal_the_host_decoders() {
    load_tables();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/format_blocks.json");
    let golden: Json = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut blocks: Vec<(u32, Vec<u8>)> = golden["blocks"].as_array().unwrap().iter()
        .map(|case| (case["kind"].as_u64().unwrap() as u32, hex(case["data"].as_str().unwrap())))
        .filter(|(kind, _)| CPU_KINDS.contains(kind))
        .collect();
    let mut random = Random(11);
    for kind in CPU_KINDS {
        blocks.extend((0..500).map(|_| (kind, generated_block(kind, &mut random))));
    }
    let wide = std::is_x86_feature_detected!("avx512f");
    for (kind, block) in &blocks {
        let expected = decode(*kind, block);
        for level in [0, i32::from(wide)] {
            let mut out = vec![0f32; expected.len()];
            // SAFETY: a whole block and room for its values.
            let status = unsafe { ffi::sd_cpu_decode(*kind as i32, level, block.as_ptr(), out.as_mut_ptr()) };
            assert_eq!(status, 0, "kind {kind}");
            let bits: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
            assert_eq!(bits, expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "kind {kind} wide {level}");
        }
    }
}

/// Rows of one bank through sd_cpu_rows at an instruction-set level (0: the decoded scalar path).
fn cpu_rows(kind: u32, level: i32, rows: &[u8], count: usize, x: &[f32]) -> Option<Vec<f32>> {
    let mut out = vec![0f32; count];
    // SAFETY: `count` rows of len/count bytes each against x of their width.
    let status = unsafe {
        ffi::sd_cpu_rows(kind as i32, level, rows.as_ptr(), (rows.len() / count) as u64, count as u32, x.as_ptr(),
                         x.len() as u64, out.as_mut_ptr())
    };
    (status == 0).then_some(out)
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


#[test]
fn cpu_row_kernels_decode_every_weight_and_sum_within_f32_bounds() {
    load_tables();
    let mut random = Random(5);
    let levels: Vec<i32> = [(1, "avx2"), (2, "avx512f")].into_iter()
        .filter(|(_, feature)| match *feature { "avx2" => std::is_x86_feature_detected!("avx2"), _ => std::is_x86_feature_detected!("avx512f") })
        .map(|(level, _)| level).collect();
    for kind in CPU_KINDS {
        let (values, _) = formats::block(kind).unwrap();
        let (rows, cols) = (7, values * (2304 / values).min(9));
        let data: Vec<u8> = (0..rows * cols / values).flat_map(|_| generated_block(kind, &mut random)).collect();
        let weights = decode(kind, &data);
        // One-hot inputs: every row returns its decoded weight at that column.
        for col in (0..cols).step_by(if values == 256 { 7 } else { 3 }) {
            let mut x = vec![0f32; cols];
            x[col] = 1.0;
            for &level in std::iter::once(&0).chain(&levels) {
                let out = cpu_rows(kind, level, &data, rows, &x).unwrap();
                for (row, value) in out.iter().enumerate() {
                    assert_eq!(*value, weights[row * cols + col], "kind {kind} level {level} row {row} col {col}");
                }
            }
        }
        let x: Vec<f32> = (0..cols).map(|_| random.activation()).collect();
        let scalar = cpu_rows(kind, 0, &data, rows, &x).unwrap();
        check_sums(&format!("CPU kind {kind}"), &scalar, &weights, cols, &x, 4.0);
        // The fused kernels keep the decoded path's arithmetic: equal bit for bit.
        for &level in &levels {
            let fused = cpu_rows(kind, level, &data, rows, &x).unwrap();
            assert_eq!(fused.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                       scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "kind {kind} level {level}");
        }
    }
}


/// The DeepSeek-V4.1 shards: tensors by name.
struct Shards {
    files: Vec<File>,
    headers: Vec<ModelHeader>,
}

impl Shards {
    fn open() -> Option<Shards> {
        let root = common::model_env("STRAYDOG_DEEPSEEK", "the DeepSeek-V4.1-Flash UD-IQ2_XXS directory")?;
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

#[test]
fn gpu_linear_products_decode_mxfp4_exactly() {
    if straydog_gpu::device_count().unwrap_or(0) == 0 {
        eprintln!("skipped: no GPU");
        return;
    }
    load_tables();
    let device = Device::new(0, 2 << 30).unwrap();
    // No file on hand has MXFP4 tensors: generated blocks, exponents in a trained model's range.
    let mut random = Random(39);
    let (rows, cols) = (2304, 5120);
    let data: Vec<u8> = (0..rows * cols / 32).flat_map(|_| generated_block(39, &mut random)).collect();
    check_linear(&device, "generated [39]", 39, rows, cols, &data);
}

const HIDDEN: usize = 5120;
const FFN: usize = 2304;

/// Gate, up and down banks of a few experts.
struct Banks {
    label: String,
    kinds: [u32; 3],
    experts: usize,
    data: [Vec<u8>; 3],
}

impl Banks {
    fn layer(shards: &Shards, layer: usize, which: &[usize]) -> Banks {
        let roles = ["gate", "up", "down"].map(|role| shards.experts(&format!("blk.{layer}.ffn_{role}_exps.weight"), which));
        let kinds = [roles[0].0, roles[1].0, roles[2].0];
        Banks { label: format!("layer {layer}"), kinds, experts: which.len(), data: roles.map(|role| role.1) }
    }

    fn shared(shards: &Shards, layer: usize) -> Banks {
        let roles = ["gate", "up", "down"].map(|role| shards.experts(&format!("blk.{layer}.ffn_{role}_shexp.weight"), &[0]));
        let kinds = [roles[0].0, roles[1].0, roles[2].0];
        Banks { label: format!("layer {layer} shared"), kinds, experts: 1, data: roles.map(|role| role.1) }
    }

    /// MXFP4 banks of generated blocks (no file on hand has MXFP4 experts).
    fn mxfp4(experts: usize) -> Banks {
        let mut random = Random(39);
        let data = [FFN * HIDDEN, FFN * HIDDEN, HIDDEN * FFN]
            .map(|values| (0..experts * values / 32).flat_map(|_| generated_block(39, &mut random)).collect());
        Banks { label: "MXFP4".into(), kinds: [39; 3], experts, data }
    }

    /// One expert's matrix of a role, decoded (rows x cols, row-major).
    fn decoded(&self, role: usize, expert: usize) -> Vec<f32> {
        let per = self.data[role].len() / self.experts;
        decode(self.kinds[role], &self.data[role][expert * per..(expert + 1) * per])
    }
}


/// sd_expert_rows' outputs.
#[derive(Default)]
struct Rows {
    gate_up: Vec<f32>,
    product: Vec<f32>,
    output: Vec<f32>,
}

/// One sd_expert_rows call: Err when the path is unavailable on this GPU.
#[allow(clippy::too_many_arguments)]
fn expert_rows(device: &Device, pointers: &[*mut c_void; 3], banks: &Banks, steps: usize, top_k: usize, selected: &[i32],
               path: i32, input: Option<&[f32]>, down_input: Option<&[f32]>, want: [bool; 3]) -> Result<Rows, String> {
    let routes = steps * top_k;
    let mut rows = Rows {
        gate_up: vec![0.0; if want[0] { routes * 2 * FFN } else { 0 }],
        product: vec![0.0; if want[1] { routes * FFN } else { 0 }],
        output: vec![0.0; if want[2] { routes * HIDDEN } else { 0 }],
    };
    let pointer = |want: bool, values: &mut Vec<f32>| if want { values.as_mut_ptr() } else { std::ptr::null_mut() };
    let kinds = banks.kinds.map(|kind| kind as i32);
    let gate_up = pointer(want[0], &mut rows.gate_up);
    let product = pointer(want[1], &mut rows.product);
    let output = pointer(want[2], &mut rows.output);
    // SAFETY: the banks are this device's allocations of the declared shapes; every buffer is sized
    // for `routes` rows.
    let status = unsafe {
        ffi::sd_expert_rows(device.handle(), pointers.as_ptr(), kinds.as_ptr(), banks.experts as u64, HIDDEN as u64,
                            FFN as u64, steps as u64, top_k as u64, selected.as_ptr(), path,
                            input.map_or(std::ptr::null(), |input| input.as_ptr()),
                            down_input.map_or(std::ptr::null(), |input| input.as_ptr()), gate_up, product, output,
                            std::ptr::null_mut())
    };
    if status == 0 { Ok(rows) } else { Err(straydog_gpu::last_error().0) }
}

fn identity(size: usize) -> Vec<f32> {
    let mut rows = vec![0f32; size * size];
    for i in 0..size {
        rows[i * size + i] = 1.0;
    }
    rows
}

fn silu_product(gate: f32, up: f32) -> f64 {
    let gate = f64::from(gate);
    gate / (1.0 + (-gate).exp()) * f64::from(up)
}


/// Every check of one set of banks, on both paths.
fn check_banks(device: &Device, banks: &Banks) {
    let allocations: Vec<_> = banks.data.iter().map(|data| device.upload(data).unwrap()).collect();
    let pointers = [allocations[0].pointer(), allocations[1].pointer(), allocations[2].pointer()];
    let decoded: Vec<[Vec<f32>; 3]> = (0..banks.experts).map(|e| [0, 1, 2].map(|role| banks.decoded(role, e))).collect();
    let experts = banks.experts;
    let all: Vec<i32> = (0..experts as i32).collect();
    for path in [0, 1] {
        let name = format!("{} {:?} {}", banks.label, banks.kinds, ["GEMV", "WMMA"][path as usize]);
        let exact = |value: f32| if path == 0 { value } else { round_f16(value) };
        // Every column of every expert: one-hot rows routed to all the experts.
        let selected: Vec<i32> = (0..HIDDEN).flat_map(|_| all.iter().copied()).collect();
        let rows = match expert_rows(device, &pointers, banks, HIDDEN, experts, &selected, path, Some(&identity(HIDDEN)), None,
                                     [true, false, false]) {
            Ok(rows) => rows,
            Err(error) => {
                eprintln!("{name}: skipped ({error})");
                continue;
            }
        };
        for token in 0..HIDDEN {
            for (k, expert) in all.iter().enumerate() {
                let route = &rows.gate_up[(token * experts + k) * 2 * FFN..][..2 * FFN];
                for role in 0..2 {
                    for row in 0..FFN {
                        let wanted = exact(decoded[*expert as usize][role][row * HIDDEN + token]);
                        assert_eq!(route[role * FFN + row], wanted, "{name}: expert {expert} role {role} row {row} col {token}");
                    }
                }
            }
        }
        let selected: Vec<i32> = (0..FFN).flat_map(|_| all.iter().copied()).collect();
        let down_input: Vec<f32> = identity(FFN).chunks(FFN).flat_map(|row| std::iter::repeat_n(row, experts).flatten().copied()).collect();
        let rows = expert_rows(device, &pointers, banks, FFN, experts, &selected, path, None, Some(&down_input), [false, false, true])
            .unwrap();
        for token in 0..FFN {
            for (k, expert) in all.iter().enumerate() {
                let route = &rows.output[(token * experts + k) * HIDDEN..][..HIDDEN];
                for row in 0..HIDDEN {
                    let wanted = exact(decoded[*expert as usize][2][row * FFN + token]);
                    assert_eq!(route[row], wanted, "{name}: expert {expert} down row {row} col {token}");
                }
            }
        }
        // Real-sized activations: sums within F32 summation bounds, the product and the down of it.
        let mut random = Random(17 + path as u64);
        let steps = 40;
        let top_k = experts.min(3);
        let selected: Vec<i32> = (0..steps).flat_map(|_| {
            let mut order = all.clone();
            for i in (1..order.len()).rev() {
                order.swap(i, random.next() as usize % (i + 1));
            }
            order.truncate(top_k);
            order
        }).collect();
        let input: Vec<f32> = (0..steps * HIDDEN).map(|_| random.activation()).collect();
        let rows = expert_rows(device, &pointers, banks, steps, top_k, &selected, path, Some(&input), None, [true, true, true])
            .unwrap();
        let operand = |value: f32| if path == 0 { value } else { round_f16(value) };
        let factor = if path == 0 { 4.0 } else { 1.0 };
        for token in 0..steps {
            let x: Vec<f32> = input[token * HIDDEN..(token + 1) * HIDDEN].iter().map(|v| operand(*v)).collect();
            for k in 0..top_k {
                let route = token * top_k + k;
                let expert = selected[route] as usize;
                for (role, matrix) in decoded[expert][..2].iter().enumerate() {
                    let weights: Vec<f32> = matrix.iter().map(|w| operand(*w)).collect();
                    check_sums(&format!("{name} route {route} role {role}"), &rows.gate_up[(2 * route + role) * FFN..][..FFN],
                               &weights, HIDDEN, &x, factor);
                }
                // SiLU(gate) * up: F32 to a few ulps (GEMV), or that rounded to FP16 (WMMA).
                for j in 0..FFN {
                    let wanted = silu_product(rows.gate_up[2 * route * FFN + j], rows.gate_up[(2 * route + 1) * FFN + j]);
                    let relative = if path == 0 { 1e-6 } else { 2f64.powi(-11) + 2e-6 };
                    let got = f64::from(rows.product[route * FFN + j]);
                    assert!((got - wanted).abs() <= relative * wanted.abs() + 2f64.powi(-25),
                            "{name} route {route} product {j}: {got} vs {wanted}");
                }
                let down: Vec<f32> = decoded[expert][2].iter().map(|w| operand(*w)).collect();
                check_sums(&format!("{name} route {route} down"), &rows.output[route * HIDDEN..][..HIDDEN], &down, FFN,
                           &rows.product[route * FFN..][..FFN], factor);
            }
        }
        eprintln!("{name}: every weight exact, sums within bounds");
    }
}

#[test]
fn gpu_expert_kernels_decode_every_weight_and_sum_within_f32_bounds() {
    let Some(shards) = Shards::open() else { return };
    load_tables();
    let device = Device::new(0, 6 << 30).unwrap();
    // Every expert format of the file: gate/up IQ1_S, IQ2_XXS, IQ2_S; down Q2_K, IQ2_XS, IQ2_XXS,
    // IQ2_S, IQ3_XXS; the shared experts' Q8_0 and Q5_K; and generated MXFP4.
    for (layer, which) in [(0, [3, 381]), (1, [7, 200]), (2, [64, 300]), (10, [1, 255]), (20, [9, 383]), (25, [100, 2])] {
        check_banks(&device, &Banks::layer(&shards, layer, &which));
    }
    check_banks(&device, &Banks::shared(&shards, 0));
    check_banks(&device, &Banks::shared(&shards, 27));
}

#[test]
fn gpu_expert_kernels_decode_generated_mxfp4_exactly() {
    if straydog_gpu::device_count().unwrap_or(0) == 0 {
        eprintln!("skipped: no GPU");
        return;
    }
    load_tables();
    let device = Device::new(0, 6 << 30).unwrap();
    check_banks(&device, &Banks::mxfp4(2));
}


#[test]
fn cpu_experts_on_the_model_equal_the_reference() {
    let Some(shards) = Shards::open() else { return };
    load_tables();
    // SAFETY: plain value; null on failure.
    let engine = unsafe { ffi::sd_cpu_experts_create(0) };
    assert!(!engine.is_null());
    let mut random = Random(23);
    for (layer, which) in [(0, [3, 381]), (1, [7, 200]), (2, [64, 300]), (10, [1, 255]), (20, [9, 383]), (25, [100, 2])] {
        let banks = Banks::layer(&shards, layer, &which);
        let kinds = banks.kinds.map(|kind| kind as i32);
        let per = [0, 1, 2].map(|role| banks.data[role].len() / 2);
        let slices = |role: usize| -> Vec<*const u8> { (0..2).map(|e| banks.data[role][e * per[role]..].as_ptr()).collect() };
        let (gate, up, down) = (slices(0), slices(1), slices(2));
        let weights = [0.625f32, 0.375];
        let x: Vec<f32> = (0..HIDDEN).map(|_| random.activation()).collect();
        let mut out = vec![0f32; HIDDEN];
        // SAFETY: two experts' slices of the declared shapes; x and out hold `hidden` values.
        let status = unsafe {
            ffi::sd_cpu_experts_run(engine, kinds.as_ptr(), gate.as_ptr(), up.as_ptr(), down.as_ptr(), weights.as_ptr(), 2,
                                    HIDDEN as u64, FFN as u64, x.as_ptr(), out.as_mut_ptr())
        };
        assert_eq!(status, 0);
        // The same in F64 from the decoded weights, with the engine's F32 activation in between
        // checked through its bound: each expert's products within 4 n u, then the down sums.
        let mut expected = vec![0f64; HIDDEN];
        let mut bound = vec![0f64; HIDDEN];
        for (e, weight) in weights.iter().enumerate() {
            let [g, u, d] = [0, 1, 2].map(|role| banks.decoded(role, e));
            let sums = |w: &[f32]| -> Vec<(f64, f64)> {
                w.chunks(HIDDEN).map(|row| row.iter().zip(&x).fold((0.0, 0.0), |(s, m), (w, x)| {
                    let t = f64::from(*w) * f64::from(*x);
                    (s + t, m + t.abs())
                })).collect()
            };
            let (gs, us) = (sums(&g), sums(&u));
            let product: Vec<f64> = gs.iter().zip(&us).map(|(g, u)| g.0 / (1.0 + (-g.0).exp()) * u.0).collect();
            // A product's error: its gate's and up's sum errors, carried through SiLU (slope below 1.1).
            let slack: Vec<f64> = gs.iter().zip(&us).map(|(g, u)| {
                let (eg, eu) = (4.0 * HIDDEN as f64 * 2f64.powi(-24) * g.1, 4.0 * HIDDEN as f64 * 2f64.powi(-24) * u.1);
                1.1 * eg * (u.0.abs() + eu) + (g.0.abs() + eg) * eu + 1e-6 * (g.0 * u.0).abs()
            }).collect();
            for (row, w) in d.chunks(FFN).enumerate() {
                let (mut s, mut m, mut carried) = (0.0, 0.0, 0.0);
                for ((w, p), e) in w.iter().zip(&product).zip(&slack) {
                    s += f64::from(*w) * p;
                    m += (f64::from(*w) * p).abs();
                    carried += f64::from(*w).abs() * e;
                }
                expected[row] += f64::from(*weight) * s;
                bound[row] += f64::from(*weight) * (4.0 * FFN as f64 * 2f64.powi(-24) * m + carried) + 1e-30;
            }
        }
        for row in 0..HIDDEN {
            assert!((f64::from(out[row]) - expected[row]).abs() <= bound[row] + 4.0 * 2f64.powi(-24) * expected[row].abs(),
                    "layer {layer} row {row}: {} vs {} (bound {:e})", out[row], expected[row], bound[row]);
        }
        eprintln!("CPU experts, {} {:?}: within bounds", banks.label, banks.kinds);
    }
    // SAFETY: created above, destroyed once.
    unsafe { ffi::sd_cpu_experts_destroy(engine) };
}

