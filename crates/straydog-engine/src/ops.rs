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

// A Buffer is a device address and a size, a plain value: nothing reaches the memory through it
// but the ops of the Backend that owns it, which check it against that device's allocations. So it
// may move between threads with its Backend; it is used on the thread that holds its Backend.
unsafe impl Send for Buffer {}

impl Buffer {
    /// The address `floats` F32 values (or 4-byte items) into the buffer.
    pub fn at(&self, floats: usize) -> *mut c_void {
        debug_assert!((floats as u64) * 4 <= self.bytes);
        self.pointer.wrapping_byte_add(floats * 4)
    }

    pub fn base(&self) -> *mut c_void {
        self.pointer
    }

    /// `bytes` of this buffer from `offset` bytes in (for ops that take a buffer, such as weights).
    pub fn slice(&self, offset: u64, bytes: u64) -> Buffer {
        assert!(offset + bytes <= self.bytes, "a buffer slice past its end");
        Buffer { pointer: self.pointer.wrapping_byte_add(offset as usize), bytes }
    }
}

/// A device's rows of each expert for [`Backend::experts`]: gate and up rows [ffn_first,
/// ffn_first + ffn_rows), down rows [hidden_first, hidden_first + hidden_rows); stages 1 (gate/up),
/// 2 (down) or 3 (both).
pub type ExpertSplit = ffi::sd_expert_split;

/// One item of [`Backend::linears`]: `output[t][r] = weights[first_row + r] . input[t]`.
#[derive(Debug, Clone, Copy)]
pub struct Linear {
    pub weights: Buffer,
    pub kind: u32,
    pub rows: u64,
    pub cols: u64,
    pub first_row: u64,
    pub input: P,
    pub output: M,
}

/// Tensor parallel: on every device, `outs[i]` = the sum of every device's `parts[j]` (`count`
/// values each), added in device order, each device's stream waiting for the others' parts. A part
/// stays unchanged until every sum has run.
pub fn sum_devices(devices: &[&Backend], outs: &[M], parts: &[P], count: u64) -> Result<()> {
    if outs.len() != devices.len() || parts.len() != devices.len() {
        return Err("a device sum takes one out and one part per device".into());
    }
    let handles: Vec<*mut c_void> = devices.iter().map(|device| device.handle()).collect();
    let outs: Vec<*mut f32> = outs.iter().map(|out| out.cast()).collect();
    let parts: Vec<*const f32> = parts.iter().map(|part| part.cast()).collect();
    // SAFETY: live handles; the library checks every range against its device's allocations.
    check(unsafe { ffi::sd_op_sum_devices(handles.as_ptr(), outs.as_ptr(), parts.as_ptr(), handles.len() as u64, count) })
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

    /// Start reading `bytes` of the buffer, `offset_bytes` in, behind the work queued so far; the
    /// work queued after it keeps running until [`Backend::read_finish`].
    pub fn read_start(&mut self, buffer: &Buffer, offset_bytes: u64, bytes: u64) -> Result<()> {
        // SAFETY: a live handle; the library checks the range.
        check(unsafe { ffi::sd_buffer_read_start(self.handle(), buffer.pointer, offset_bytes, bytes) })
    }

    /// Wait for the started read and copy its values out (`out` must take exactly its bytes).
    pub fn read_finish<T: Copy>(&mut self, out: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(out) as u64;
        // SAFETY: a live handle; `out` holds `bytes`, which the library checks against the read.
        check(unsafe { ffi::sd_buffer_read_finish(self.handle(), out.as_mut_ptr().cast(), bytes) })
    }

    /// Copy `bytes` from `source` to `target` on this device, queued behind its work.
    pub fn copy(&mut self, source: P, target: M, bytes: u64) -> Result<()> {
        // SAFETY: a live handle; the library checks both ranges against its allocations.
        check(unsafe { ffi::sd_buffer_copy_peer(self.handle(), target.cast(), self.handle(), source.cast(), bytes) })
    }

    /// Copy `bytes` from `source` (this device's) into `target` on `to`, after the work queued here;
    /// `to`'s later work waits for the bytes.
    pub fn copy_to(&mut self, source: P, to: &mut Backend, target: M, bytes: u64) -> Result<()> {
        // SAFETY: two live handles; the library checks both ranges against their devices' allocations.
        check(unsafe { ffi::sd_buffer_copy_peer(to.handle(), target.cast(), self.handle(), source.cast(), bytes) })
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

    /// Record the ops queued from here to [`Backend::graph_end`] into a graph instead of running
    /// them (only ops with no host data: see the C header).
    pub fn graph_begin(&mut self) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_graph_begin(self.handle()) })
    }

    /// Stop recording: with `keep`, store the graph under `key` (replacing any), else drop it.
    pub fn graph_end(&mut self, key: u64, keep: bool) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_graph_end(self.handle(), key, i32::from(keep)) })
    }

    /// Queue the graph stored under `key`; false when there is none.
    pub fn graph_run(&mut self, key: u64) -> Result<bool> {
        // SAFETY: a live handle.
        let state = unsafe { ffi::sd_graph_run(self.handle(), key) };
        check(state.min(0))?;
        Ok(state == 0)
    }

    /// Several linear ops, in one launch where the GEMV takes them all (see the C header).
    pub fn linears(&mut self, items: &[Linear], steps: u64, input_stride: u64, output_stride: u64, scratch: &Buffer) -> Result<()> {
        let items: Vec<ffi::sd_linear> = items.iter().map(|item| ffi::sd_linear {
            weights: item.weights.pointer.cast_const(), kind: item.kind as i32, rows: item.rows, cols: item.cols,
            first_row: item.first_row, input: item.input.cast(), output: item.output.cast(),
        }).collect();
        // SAFETY: a live handle; the library checks every item's ranges against its allocations.
        check(unsafe {
            ffi::sd_op_linears(self.handle(), items.as_ptr(), items.len() as u64, steps, input_stride, output_stride,
                               scratch.pointer.cast(), scratch.bytes)
        })
    }

    pub fn rms_norm(&mut self, input: P, weight: &Buffer, rows: u64, width: u64, eps: f32, output: M)
                    -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_rms_norm(self.handle(), input.cast(), weight.pointer.cast(), rows, width, eps, output.cast()) })
    }

    /// rms_norm, then the last `rope` = (dims, positions, frequencies) values rotated as [`Backend::rope`]
    /// rotates them, each row stored at row `out_rows[row]` of `out` (or in order).
    #[allow(clippy::too_many_arguments)]
    pub fn norm_rope(&mut self, input: P, weight: &Buffer, rows: u64, width: u64, eps: f32, rope: Option<(u64, &Buffer, &Buffer)>,
                     out_rows: Option<P>, out: M, out_count: u64) -> Result<()> {
        let (dims, positions, frequencies) =
            rope.map_or((0, std::ptr::null_mut(), std::ptr::null_mut()), |(dims, p, f)| (dims, p.pointer, f.pointer));
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_norm_rope(self.handle(), input.cast(), weight.pointer.cast(), rows, width, eps, dims, positions.cast(),
                                 frequencies.cast(), out_rows.map_or(std::ptr::null(), |r| r.cast()), out.cast(), out_count)
        })
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

    /// A sublayer boundary in one launch: `expand` = (x, residual, post, comb) expanded into `stream`
    /// (else `stream` is read); `coefficients` = (fn, base, scale, pre, post, comb) from it; and
    /// `out` = rms_norm(stream collapsed by `collapse`) * `norm_weight`.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_step(&mut self, expand: Option<(P, P, P, P)>, stream: M, coefficients: Option<(P, P, P, M, M, M)>, iterations: u64,
                   eps: f32, norm_eps: f32, collapse: P, norm_weight: P, rows: u64, hc: u64, width: u64, workspace: &Buffer,
                   out: M) -> Result<()> {
        let null = std::ptr::null::<c_void>();
        let (x, residual, post_in, comb_in) = expand.unwrap_or((null, null, null, null));
        let mut_null = std::ptr::null_mut::<c_void>();
        let (fn_, base, scale, pre, post, comb) = coefficients.unwrap_or((null, null, null, mut_null, mut_null, mut_null));
        // SAFETY: a live handle; the library checks every range against its allocations.
        check(unsafe {
            ffi::sd_op_hc_step(self.handle(), x.cast(), residual.cast(), post_in.cast(), comb_in.cast(), stream.cast(), fn_.cast(),
                               base.cast(), scale.cast(), iterations, eps, norm_eps, collapse.cast(), norm_weight.cast(), rows, hc,
                               width, workspace.pointer.cast(), workspace.bytes, pre.cast(), post.cast(), comb.cast(), out.cast())
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

    pub fn scatter(&mut self, x: P, rows: P, count: u64, width: u64, out: M, out_rows: u64) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_scatter(self.handle(), x.cast(), rows.cast(), count, width, out.cast(), out_rows) })
    }

    pub fn scatter_add(&mut self, x: P, rows: P, count: u64, width: u64, out: M, out_rows: u64) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_scatter_add(self.handle(), x.cast(), rows.cast(), count, width, out.cast(), out_rows) })
    }

    /// `x` (steps x channels) becomes SiLU of each channel's causal convolution with its `taps`
    /// (channels x kernel) over `history` (kernel - 1 rows, oldest first, left holding the last inputs).
    pub fn conv_silu(&mut self, x: M, history: M, taps: P, channels: u64, kernel: u64, steps: u64) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_conv_silu(self.handle(), x.cast(), history.cast(), taps.cast(), channels, kernel, steps) })
    }

    /// The gated delta rule over `qkv` rows [q | k | v] into `out` (steps x v_heads x dim), `state`
    /// (v_heads x dim x dim, [key][value]) carried (see the C header).
    #[allow(clippy::too_many_arguments)]
    pub fn delta_rule(&mut self, qkv: P, beta: P, alpha: P, dt: P, a: P, state: M, k_heads: u64, v_heads: u64, dim: u64, steps: u64,
                      eps: f32, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_delta_rule(self.handle(), qkv.cast(), beta.cast(), alpha.cast(), dt.cast(), a.cast(), state.cast(), k_heads, v_heads,
                                  dim, steps, eps, out.cast())
        })
    }

    /// rms_norm(x) * weight * SiLU(gate) per row.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_norm(&mut self, x: P, gate: P, weight: &Buffer, rows: u64, width: u64, eps: f32, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_gated_norm(self.handle(), x.cast(), gate.cast(), weight.pointer.cast(), rows, width, eps, out.cast()) })
    }

    /// Rows of [first | second] (2 x width values) into two buffers of rows.
    pub fn split(&mut self, x: P, rows: u64, width: u64, first: M, second: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_split(self.handle(), x.cast(), rows, width, first.cast(), second.cast()) })
    }

    /// Rotate the leading `rope_dims` values of each of a row's `vectors`, pair i = (i, i + rope_dims / 2).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_halves(&mut self, x: M, rows: u64, vectors: u64, width: u64, rope_dims: u64, positions: &Buffer,
                       frequencies: &Buffer) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_rope_halves(self.handle(), x.cast(), rows, vectors, width, rope_dims, positions.pointer.cast(),
                                   frequencies.pointer.cast())
        })
    }

    /// Causal grouped-query attention of `q` (rows x heads x dim) at `positions` over the cache's
    /// keys and values (context x kv_heads x dim), times sigmoid(gate) when given; the keys split
    /// among blocks whose partial results take `scratch` (see the C header).
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&mut self, q: P, keys: P, values: P, context: u64, positions: P, rows: u64, heads: u64, kv_heads: u64, dim: u64,
                     scale: f32, gate: Option<P>, scratch: &Buffer, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_attention(self.handle(), q.cast(), keys.cast(), values.cast(), context, positions.cast(), rows, heads, kv_heads,
                                 dim, scale, gate.map_or(std::ptr::null(), |g| g.cast()), scratch.pointer.cast(), scratch.bytes,
                                 out.cast())
        })
    }

    /// out += x (count values).
    pub fn add(&mut self, x: P, count: u64, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_add(self.handle(), x.cast(), count, out.cast()) })
    }

    /// Each row of x (rows x width) times sigmoid of its scalar.
    pub fn scale_rows(&mut self, x: M, scalars: P, rows: u64, width: u64) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_scale_rows(self.handle(), x.cast(), scalars.cast(), rows, width) })
    }

    /// Experts in slots: route `r` of `rows * top_k` (token-major) runs the expert of the slot at
    /// `bases[r]` (gate, up and down at `offsets` bytes in, of `kinds`; a zero base: a route another
    /// device runs): `product[r]` = its clamped SwiGLU times `weights[r]`, and `out[t]` (+)= the sum
    /// of down . product over the token's routes. With `split`, a device's rows of each expert and
    /// the stages it asks for; without `bases`, this device's last routes again (see the C header).
    #[allow(clippy::too_many_arguments)]
    pub fn experts(&mut self, bases: Option<&[u64]>, slot_bytes: u64, offsets: [u64; 3], kinds: [u32; 3], hidden: u64, ffn: u64,
                   rows: u64, top_k: u64, input: P, input_stride: u64, weights: Option<P>, clamp: f32, accumulate: bool,
                   product: M, out: M, out_stride: u64, split: Option<ExpertSplit>) -> Result<()> {
        if bases.is_some_and(|bases| bases.len() as u64 != rows * top_k) {
            return Err("expert routes do not match the rows".into());
        }
        let kinds = kinds.map(|kind| kind as i32);
        // SAFETY: a live handle; the library checks every slot and range against its allocations.
        check(unsafe {
            ffi::sd_op_experts(self.handle(), bases.map_or(std::ptr::null(), <[u64]>::as_ptr), slot_bytes, offsets.as_ptr(), kinds.as_ptr(), hidden, ffn, rows, top_k,
                               input.cast(), input_stride, weights.map_or(std::ptr::null(), |w| w.cast()), clamp,
                               i32::from(accumulate), product.cast(), out.cast(), out_stride,
                               split.as_ref().map_or(std::ptr::null(), |split| split as *const ffi::sd_expert_split))
        })
    }

    /// Write `bases` (slot addresses, each `slot_bytes`; checked) into the device array `table`, for
    /// [`experts_table`](Self::experts_table).
    pub fn expert_table(&mut self, table: &Buffer, bases: &[u64], slot_bytes: u64) -> Result<()> {
        if bases.len() as u64 * 8 > table.bytes {
            return Err("an expert table past its buffer".into());
        }
        // SAFETY: a live handle; the library checks the table and every slot against its allocations.
        check(unsafe { ffi::sd_expert_table(self.handle(), table.pointer.cast(), bases.as_ptr(), bases.len() as u64, slot_bytes) })
    }

    /// [`experts`](Self::experts) for whole experts, route `r` taking the slot `table[ids[r]]` (`count`
    /// entries; the ids on the device): no host data, so a graph may record it.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_table(&mut self, table: &Buffer, count: u64, ids: P, slot_bytes: u64, offsets: [u64; 3], kinds: [u32; 3], hidden: u64,
                         ffn: u64, rows: u64, top_k: u64, input: P, weights: Option<P>, clamp: f32, accumulate: bool, product: M,
                         out: M) -> Result<()> {
        let kinds = kinds.map(|kind| kind as i32);
        // SAFETY: a live handle; the library checks every range against its allocations (the
        // table's slots were checked as written).
        check(unsafe {
            ffi::sd_op_experts_table(self.handle(), table.pointer.cast(), count, ids.cast(), slot_bytes, offsets.as_ptr(), kinds.as_ptr(),
                                     hidden, ffn, rows, top_k, input.cast(), weights.map_or(std::ptr::null(), |w| w.cast()), clamp,
                                     i32::from(accumulate), product.cast(), out.cast())
        })
    }

    /// Tiled over `scratch` when it holds the scores (see the C header), else one block per head.
    /// `rope` = (dims, frequencies): q's last dims are rotated on the way in, the output's back
    /// (tiled only).
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attention(&mut self, q: P, raw: P, raw_rows: u64, positions: P, window: u64, compressed: Option<(P, u64)>,
                            selected: Option<(P, u64)>, sinks: P, rows: u64, heads: u64, dim: u64, scale: f32,
                            rope: Option<(u64, &Buffer)>, scratch: Option<&Buffer>, out: M) -> Result<()> {
        let (compressed, compressed_rows) = compressed.unwrap_or((std::ptr::null(), 0));
        let (selected, selections) = selected.unwrap_or((std::ptr::null(), 0));
        let (rope_dims, frequencies) = rope.map_or((0, std::ptr::null_mut()), |(dims, buffer)| (dims, buffer.pointer));
        let (scratch, scratch_bytes) = scratch.map_or((std::ptr::null_mut(), 0), |buffer| (buffer.pointer, buffer.bytes));
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_sparse_attention(self.handle(), q.cast(), raw.cast(), raw_rows, positions.cast(), window,
                                        compressed.cast(), compressed_rows, selected.cast(), selections, sinks.cast(), rows,
                                        heads, dim, scale, rope_dims, frequencies.cast(), scratch.cast(), scratch_bytes, out.cast())
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
