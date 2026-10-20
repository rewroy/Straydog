//! Generic GPU operations over device buffers (docs/DESIGN.md §10.1): shapes are arguments,
//! and an architecture composes them into its layers. Every op is queued on the device's
//! stream; only reading a buffer waits for it.

use std::ffi::c_void;

use straydog_gpu::ffi;

use crate::gpu::{Backend, Result};

/// A device address to read (P) or write (M), into a [`Buffer`].
type P = *const c_void;
type M = *mut c_void;

fn check(status: i32) -> Result<()> {
    straydog_gpu::check(status).map(drop).map_err(|error| error.to_string())
}

/// A device allocation of one backend (freed with the backend).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Buffer {
    pointer: *mut c_void,
    pub bytes: u64,
}

impl Buffer {
    /// The address `floats` F32 values (or 4-byte items) into the buffer.
    pub fn at(&self, floats: usize) -> *mut c_void {
        debug_assert!((floats as u64) * 4 <= self.bytes);
        self.pointer.wrapping_byte_add(floats * 4)
    }

    pub fn base(&self) -> *mut c_void {
        self.pointer
    }
}

impl Backend {
    /// A zeroed device buffer.
    pub fn buffer(&mut self, bytes: u64) -> Result<Buffer> {
        // SAFETY: a live handle.
        let pointer = unsafe { ffi::sd_buffer(self.handle(), bytes) };
        if pointer.is_null() {
            return Err(straydog_gpu::last_error().to_string());
        }
        Ok(Buffer { pointer, bytes })
    }

    /// A device copy of `data` (weights).
    pub fn upload_bytes(&mut self, data: &[u8]) -> Result<Buffer> {
        // SAFETY: a live handle; `data` outlives the call, which copies it.
        let pointer = unsafe { ffi::sd_upload(self.handle(), data.as_ptr().cast(), data.len() as u64) };
        if pointer.is_null() {
            return Err(straydog_gpu::last_error().to_string());
        }
        Ok(Buffer { pointer, bytes: data.len() as u64 })
    }

    pub fn release_buffer(&mut self, buffer: Buffer) -> Result<()> {
        // SAFETY: a live handle; the library refuses pointers it did not allocate.
        check(unsafe { ffi::sd_release(self.handle(), buffer.pointer) })
    }

    pub fn write_buffer<T: Copy>(&mut self, buffer: &Buffer, offset_bytes: u64, values: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(values) as u64;
        // SAFETY: a live handle; the library checks the range lies in its allocation.
        check(unsafe { ffi::sd_buffer_write(self.handle(), buffer.pointer, offset_bytes, values.as_ptr().cast(), bytes) })
    }

    /// Wait for the stream, then copy `out.len()` values from the buffer, `offset` values in.
    pub fn read_buffer<T: Copy>(&mut self, buffer: &Buffer, offset: usize, out: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(out) as u64;
        let at = (offset * std::mem::size_of::<T>()) as u64;
        // SAFETY: a live handle; the library checks the range; `out` holds `bytes`.
        check(unsafe { ffi::sd_buffer_read(self.handle(), buffer.pointer, at, out.as_mut_ptr().cast(), bytes) })
    }

    /// `output[t][r] = W[first + r] . input[t]` over `steps` inputs of `cols` values; strides in
    /// values between consecutive inputs and outputs (0: packed).
    #[allow(clippy::too_many_arguments)]
    pub fn linear(&mut self, weights: &Buffer, kind: u32, rows: u64, cols: u64, first: u64, input: P, input_stride: u64,
                  steps: u64, output: M, output_stride: u64, scratch: &Buffer) -> Result<()> {
        // SAFETY: a live handle; the library checks every range against its allocations.
        check(unsafe {
            ffi::sd_op_linear(self.handle(), weights.pointer, kind as i32, rows, cols, first, input.cast(), input_stride, steps,
                              output.cast(), output_stride, scratch.pointer.cast(), scratch.bytes)
        })
    }

    pub fn rms_norm(&mut self, input: P, weight: &Buffer, rows: u64, width: u64, eps: f32, output: M)
                    -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_rms_norm(self.handle(), input.cast(), weight.pointer.cast(), rows, width, eps, output.cast()) })
    }

    /// Rotate the last `rope_dims` values of each of a row's `vectors` (rows of `vectors * width`).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(&mut self, x: M, rows: u64, vectors: u64, width: u64, rope_dims: u64, positions: &Buffer,
                frequencies: &Buffer, inverse: bool) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_rope(self.handle(), x.cast(), rows, vectors, width, rope_dims, positions.pointer.cast(),
                            frequencies.pointer.cast(), i32::from(inverse))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_coefficients(&mut self, mixes: P, stream: P, rows: u64, hc: u64, width: u64, base: P, scale: P,
                           iterations: u64, eps: f32, norm_eps: f32, pre: M, post: M, comb: M) -> Result<()> {
        // SAFETY: a live handle; the library checks every range against its allocations.
        check(unsafe {
            ffi::sd_op_hc_coefficients(self.handle(), mixes.cast(), stream.cast(), rows, hc, width, base.cast(), scale.cast(),
                                       iterations, eps, norm_eps, pre.cast(), post.cast(), comb.cast())
        })
    }

    pub fn hc_collapse(&mut self, stream: P, pre: P, rows: u64, hc: u64, width: u64, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_hc_collapse(self.handle(), stream.cast(), pre.cast(), rows, hc, width, out.cast()) })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_expand(&mut self, x: P, residual: P, post: P, comb: P, rows: u64, hc: u64, width: u64, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_hc_expand(self.handle(), x.cast(), residual.cast(), post.cast(), comb.cast(), rows, hc, width, out.cast())
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn engram_gate(&mut self, stream: M, kv: P, weight: P, rows: u64, hc: u64, width: u64, eps: f32) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_engram_gate(self.handle(), stream.cast(), kv.cast(), weight.cast(), rows, hc, width, eps) })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn route(&mut self, logits: P, bias: Option<P>, rows: u64, experts: u64, k: u64, gating: i32, normalize: bool,
                 scale: f32, ids: M, weights: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_route(self.handle(), logits.cast(), bias.map_or(std::ptr::null(), |b| b.cast()), rows, experts, k,
                             gating, i32::from(normalize), scale, ids.cast(), weights.cast())
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn swiglu(&mut self, gate: P, up: P, rows: u64, width: u64, clamp: f32, row_weights: Option<P>, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_swiglu(self.handle(), gate.cast(), up.cast(), rows, width, clamp,
                              row_weights.map_or(std::ptr::null(), |w| w.cast()), out.cast())
        })
    }

    pub fn gather(&mut self, x: P, x_rows: u64, rows: P, count: u64, width: u64, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_gather(self.handle(), x.cast(), x_rows, rows.cast(), count, width, out.cast()) })
    }

    pub fn scatter_add(&mut self, x: P, rows: P, count: u64, width: u64, out: M, out_rows: u64) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_scatter_add(self.handle(), x.cast(), rows.cast(), count, width, out.cast(), out_rows) })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attention(&mut self, q: P, raw: P, raw_rows: u64, positions: P, window: u64, compressed: Option<(P, u64)>,
                            selected: Option<(P, u64)>, sinks: P, rows: u64, heads: u64, dim: u64, scale: f32, out: M)
                            -> Result<()> {
        let (compressed, compressed_rows) = compressed.unwrap_or((std::ptr::null(), 0));
        let (selected, selections) = selected.unwrap_or((std::ptr::null(), 0));
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_sparse_attention(self.handle(), q.cast(), raw.cast(), raw_rows, positions.cast(), window,
                                        compressed.cast(), compressed_rows, selected.cast(), selections, sinks.cast(), rows,
                                        heads, dim, scale, out.cast())
        })
    }

    pub fn compress(&mut self, kv: P, gate: Option<P>, groups: u64, ratio: u64, dim: u64, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_compress(self.handle(), kv.cast(), gate.map_or(std::ptr::null(), |g| g.cast()), groups, ratio, dim, out.cast())
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn index_scores(&mut self, q: P, keys: P, weights: P, visible: P, rows: u64, groups: u64, heads: u64, dim: u64,
                        scale: f32, scores: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_index_scores(self.handle(), q.cast(), keys.cast(), weights.cast(), visible.cast(), rows, groups, heads, dim,
                                    scale, scores.cast())
        })
    }

    pub fn top_k(&mut self, scores: P, rows: u64, groups: u64, k: u64, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_top_k(self.handle(), scores.cast(), rows, groups, k, out.cast()) })
    }
}
