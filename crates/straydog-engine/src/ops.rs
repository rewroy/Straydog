//! Generic GPU operations over device buffers (docs/DESIGN.md §10.1): shapes are arguments,
//! and an architecture composes them into its layers. Every op is queued on the device's
//! stream; only reading a buffer waits for it.

use std::ffi::{c_int, c_void};

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

/// A wave of experts in their slots, for the grouped WMMA kernels (`Backend::slot_moe_gate_up` and
/// `slot_moe_down`): expert i's slot at `bases[i]` holds gate and up (`ffn_share` rows of `width`
/// each, `gate_bytes` apart, of `kinds[0]` and `kinds[1]`) then down (`hidden_share` rows of `ffn`,
/// of `kinds[2]`); its routes are the grouped rows `firsts[i]`.. (`counts[i]` of them). `tables`
/// takes each launch's group and tile lists.
pub struct SlotWave<'a> {
    pub bases: &'a [u64],
    pub firsts: &'a [u32],
    pub counts: &'a [u32],
    pub gate_bytes: u64,
    pub kinds: [u32; 3],
    pub width: u64,
    pub ffn_share: u64,
    pub ffn: u64,
    pub hidden_share: u64,
    pub tables: &'a Buffer,
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

/// One row for [`Backend::norm_linears_rope`] to norm and rotate beside its linears: the arguments
/// [`Backend::norm_rope`] takes for it (`rows` 1).
#[derive(Debug, Clone, Copy)]
pub struct NormRope<'a> {
    pub input: P,
    pub weight: &'a Buffer,
    pub width: u64,
    pub eps: f32,
    pub rope: Option<(u64, &'a Buffer, &'a Buffer)>,
    pub out_rows: Option<P>,
    pub out: M,
    pub out_count: u64,
}

/// One of DSpark's three independent target-feature KV projections and K-only rings.
pub struct DsparkInjectStage<'a> {
    pub kv: &'a Buffer,
    pub norm: &'a Buffer,
    pub ring: &'a Buffer,
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

/// Every device's `outs` (`rows` of `full` FP16 values) = each device's `parts` (rows of its `share`
/// FP16 columns) at its columns `firsts`: the devices' disjoint columns of one product, gathered
/// (the peers' read over the peer link, ordered as [`sum_devices`] orders its reads).
pub fn gather_columns(devices: &[&Backend], outs: &[M], parts: &[P], firsts: &[u64], rows: u64, share: u64, full: u64) -> Result<()> {
    if outs.len() != devices.len() || parts.len() != devices.len() || firsts.len() != devices.len() {
        return Err("a column gather takes one out, one part and one first column per device".into());
    }
    let handles: Vec<*mut c_void> = devices.iter().map(|device| device.handle()).collect();
    // SAFETY: live handles; the library checks every range against its device's allocations.
    check(unsafe {
        ffi::sd_op_gather_columns(handles.as_ptr(), outs.as_ptr(), parts.as_ptr(), firsts.as_ptr(), handles.len() as u64, rows, share, full)
    })
}

/// Every device's later work waits until each other device's last [`sum_devices`] has read its
/// part (that sum reads the parts where they lie): before a device writes a summed part's buffer
/// again, as a pass's next wave of experts writes its activations.
pub fn devices_settle(devices: &[&Backend]) -> Result<()> {
    let handles: Vec<*mut c_void> = devices.iter().map(|device| device.handle()).collect();
    // SAFETY: live handles.
    check(unsafe { ffi::sd_devices_settle(handles.as_ptr(), handles.len() as u64) })
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

    /// A device buffer whose contents are undefined until written.
    pub fn buffer_unset(&mut self, bytes: u64) -> Result<Buffer> {
        // SAFETY: a live handle.
        let pointer = unsafe { ffi::sd_allocate(self.handle(), bytes) };
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

    /// DSpark noncausal attention: all queries `[slots,64,512]` see the same history
    /// `[anchor-history,anchor)` from the `[128,512]` ring and every block KV row `[slots,512]`.
    /// Queries and keys already have RoPE applied. Sinks `[64]` enter only the denominator.
    /// Slots 1..5, history <=128 and <=anchor. Scratch: `4*slots*(slots+1)` bytes.
    /// All ranges are disjoint; the first invocation can be captured in a graph.
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_attention(&mut self, q: P, ring: &Buffer, block: P, sinks: &Buffer, slots: u64,
                            anchor: u32, history: u64, out: M, scratch: &Buffer) -> Result<()> {
        // SAFETY: live handle; native code checks shapes and every device allocation range.
        check(unsafe { ffi::sd_op_dspark_attention(self.handle(), q.cast(), ring.pointer.cast(), block.cast(),
            sinks.pointer.cast(), slots, anchor, history, out.cast(), scratch.pointer.cast(), scratch.bytes) })
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

    /// Inject target features `[rows,15360]` through Q8 encoder / RMS and three Q8 KV / RMS /
    /// tail-64 RoPE stages into their `[128,512]` rings. Original positions select ring rows;
    /// the last input wins on collisions. Frequencies are 32 F32 values for theta 10000.
    /// Returns encoder-normalized `[rows,5120]` features in `encoded`. All ranges are disjoint.
    /// Scratch: `4096*rows` bytes, plus at least `align256(30720*rows)+61440` when rows > 4.
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_inject(&mut self, features: P, positions: &Buffer, rows: u64, fc: &Buffer,
                         encoder_norm: &Buffer, stages: &[DsparkInjectStage<'_>; 3], frequencies: &Buffer,
                         encoded: M, scratch: &Buffer) -> Result<()> {
        let stages = stages.each_ref().map(|s| ffi::sd_dspark_inject_stage {
            kv: s.kv.pointer.cast_const(), norm: s.norm.pointer.cast(), ring: s.ring.pointer.cast(),
        });
        // SAFETY: live handle and three host descriptors; native code checks every device range.
        check(unsafe { ffi::sd_op_dspark_inject(self.handle(), features.cast(), positions.pointer.cast(), rows,
            fc.pointer.cast_const(), encoder_norm.pointer.cast(), stages.as_ptr(), frequencies.pointer.cast(),
            encoded.cast(), scratch.pointer.cast(), scratch.bytes) })
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

    /// One linear with an explicit GEMM policy: FP16 inputs/weights with F32 sums/output when
    /// half is true, otherwise F32. Up to four inputs use unchanged GEMV. Device defaults stay fixed.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_gemm(&mut self, weights: &Buffer, kind: u32, rows: u64, cols: u64, first: u64, input: P, input_stride: u64,
                       steps: u64, output: M, output_stride: u64, scratch: &Buffer, half: bool) -> Result<()> {
        // SAFETY: a live handle; the library checks every range against its allocations.
        check(unsafe {
            ffi::sd_op_linear_gemm(self.handle(), weights.pointer, kind as i32, rows, cols, first, input.cast(), input_stride, steps,
                                   output.cast(), output_stride, scratch.pointer.cast(), scratch.bytes, i32::from(half))
        })
    }

    /// DSpark's greedy Markov chain. Inputs/outputs/scratch must be disjoint; scratch holds
    /// `align16(4*(width+rank))+512` bytes. Only the first `count` tokens/confidences are used.
    /// Reference defaults are `threshold=0.0`, `minimum=0` (llama.cpp be83657 common.h:330-333).
    /// No host readback: the first invocation may be captured in a graph.
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_head(&mut self, body: P, hidden: P, w1: &Buffer, w2: &Buffer, conf: &Buffer,
                       slots: u64, vocabulary: u64, rank: u64, width: u64, anchor: u32,
                       threshold: f32, minimum: u64, logits: M, tokens: M, confidences: M,
                       count: M, scratch: &Buffer) -> Result<()> {
        // SAFETY: a live handle; the native op checks shapes and all device allocation ranges.
        check(unsafe {
            ffi::sd_op_dspark_head(self.handle(), body.cast(), hidden.cast(), w1.pointer.cast_const(), w2.pointer.cast_const(),
                conf.pointer.cast(), slots, vocabulary, rank, width, anchor, threshold, minimum, logits.cast(), tokens.cast(),
                confidences.cast(), count.cast(), scratch.pointer.cast(), scratch.bytes)
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

    /// Queue the graph stored under `key` on every one of `gpus` at once (a thread for each past the
    /// first): a launch takes host time for every op it holds, so devices that wait on each other
    /// start together. False when one has none.
    pub fn graph_run_all(gpus: &mut [&mut Backend], key: u64) -> Result<bool> {
        struct Handle(*mut c_void);
        // SAFETY: a device handle is used on one thread at a time: each launch's own thread, while
        // the caller holds every Backend mutably.
        unsafe impl Send for Handle {}
        let run = |handle: Handle| -> Result<bool> {
            // SAFETY: a live handle.
            let state = unsafe { ffi::sd_graph_run(handle.0, key) };
            check(state.min(0))?;
            Ok(state == 0)
        };
        let mut handles = gpus.iter().map(|gpu| Handle(gpu.handle()));
        let Some(first) = handles.next() else { return Ok(true) };
        std::thread::scope(|scope| {
            let others: Vec<_> = handles.map(|handle| scope.spawn(move || run(handle))).collect();
            let mut all = run(first);
            for other in others {
                let ran = other.join().map_err(|_| "a graph launch panicked".to_string())?;
                all = match (all, ran) {
                    (Ok(a), Ok(b)) => Ok(a && b),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                };
            }
            all
        })
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

    /// One vector: `normed` = the RMS norm of `x` (`weight`, `eps`, `width` values), and the
    /// linears over it (every item's input `normed`), in one launch when they share a kind the
    /// GEMV takes; the results are those of [`rms_norm`](Self::rms_norm) then [`linears`](Self::linears).
    pub fn norm_linears(&mut self, x: P, weight: &Buffer, width: u64, eps: f32, normed: M, items: &[Linear]) -> Result<()> {
        let items: Vec<ffi::sd_linear> = items.iter().map(|item| ffi::sd_linear {
            weights: item.weights.pointer.cast_const(), kind: item.kind as i32, rows: item.rows, cols: item.cols,
            first_row: item.first_row, input: item.input.cast(), output: item.output.cast(),
        }).collect();
        // SAFETY: a live handle; the library checks every range against its allocations.
        check(unsafe {
            ffi::sd_op_norm_linears(self.handle(), x.cast(), weight.pointer.cast(), width, eps, normed.cast(), items.as_ptr(),
                                    items.len() as u64)
        })
    }

    /// [`norm_linears`](Self::norm_linears), and `side`'s row normed and rotated as
    /// [`norm_rope`](Self::norm_rope) makes it, in the same launch where it can (a block of its own):
    /// the results of the separate ops.
    #[allow(clippy::too_many_arguments)]
    pub fn norm_linears_rope(&mut self, x: P, weight: &Buffer, width: u64, eps: f32, normed: M, items: &[Linear], side: &NormRope)
                             -> Result<()> {
        let items: Vec<ffi::sd_linear> = items.iter().map(|item| ffi::sd_linear {
            weights: item.weights.pointer.cast_const(), kind: item.kind as i32, rows: item.rows, cols: item.cols,
            first_row: item.first_row, input: item.input.cast(), output: item.output.cast(),
        }).collect();
        let (rope_dims, positions, frequencies) =
            side.rope.map_or((0, std::ptr::null(), std::ptr::null()), |(dims, p, f)| (dims, p.pointer.cast_const().cast(), f.pointer.cast_const().cast()));
        let side = ffi::sd_norm_rope {
            input: side.input.cast(), weight: side.weight.pointer.cast_const().cast(), width: side.width, eps: side.eps, rope_dims,
            positions, frequencies, out_rows: side.out_rows.map_or(std::ptr::null(), |rows| rows.cast()), out: side.out.cast(),
            out_count: side.out_count,
        };
        // SAFETY: a live handle; the library checks every range against its allocations.
        check(unsafe {
            ffi::sd_op_norm_linears_rope(self.handle(), x.cast(), weight.pointer.cast(), width, eps, normed.cast(), items.as_ptr(),
                                         items.len() as u64, &side)
        })
    }

    /// One vector through `items` (every one reading the same input), then the RMS norm of `x` (one of
    /// their outputs) into `normed` and `side` as [`norm_rope`](Self::norm_rope) makes it: the results
    /// of the three ops, in one launch where the GEMV takes it (its last block forms the norm and the
    /// side row).
    #[allow(clippy::too_many_arguments)]
    pub fn linears_norm_rope(&mut self, items: &[Linear], x: P, weight: &Buffer, width: u64, eps: f32, normed: M, side: &NormRope)
                             -> Result<()> {
        let items: Vec<ffi::sd_linear> = items.iter().map(|item| ffi::sd_linear {
            weights: item.weights.pointer.cast_const(), kind: item.kind as i32, rows: item.rows, cols: item.cols,
            first_row: item.first_row, input: item.input.cast(), output: item.output.cast(),
        }).collect();
        let (rope_dims, positions, frequencies) =
            side.rope.map_or((0, std::ptr::null(), std::ptr::null()), |(dims, p, f)| (dims, p.pointer.cast_const().cast(), f.pointer.cast_const().cast()));
        let side = ffi::sd_norm_rope {
            input: side.input.cast(), weight: side.weight.pointer.cast_const().cast(), width: side.width, eps: side.eps, rope_dims,
            positions, frequencies, out_rows: side.out_rows.map_or(std::ptr::null(), |rows| rows.cast()), out: side.out.cast(),
            out_count: side.out_count,
        };
        // SAFETY: a live handle; the library checks every range against its allocations.
        check(unsafe {
            ffi::sd_op_linears_norm_rope(self.handle(), items.as_ptr(), items.len() as u64, x.cast(), weight.pointer.cast(), width, eps,
                                         normed.cast(), &side)
        })
    }

    /// One vector: `out` = W . SwiGLU(gate, up) (`cols` values each; `hidden` takes the SwiGLU
    /// when it is formed apart), each row times sigmoid(`*scale`) when given: the results of
    /// [`swiglu`](Self::swiglu), [`linear`](Self::linear) and [`scale_rows`](Self::scale_rows).
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu_linear(&mut self, weights: &Buffer, kind: u32, rows: u64, cols: u64, gate: P, up: P, scale: Option<P>, hidden: M,
                         out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_swiglu_linear(self.handle(), weights.pointer.cast_const(), kind as i32, rows, cols, gate.cast(), up.cast(),
                                     scale.map_or(std::ptr::null(), |scale| scale.cast()), hidden.cast(), out.cast())
        })
    }

    /// [`swiglu_linear`](Self::swiglu_linear), and meanwhile [`route`](Self::route) of one row of
    /// 256 router `logits` (softmax gating, normalized, no bias, scale 1) into `ids` and
    /// `route_weights`; the same results, in one launch where the GEMV takes it.
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu_linear_route(&mut self, weights: &Buffer, kind: u32, rows: u64, cols: u64, gate: P, up: P, scale: Option<P>,
                               hidden: M, out: M, logits: P, k: u64, ids: M, route_weights: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_swiglu_linear_route(self.handle(), weights.pointer.cast_const(), kind as i32, rows, cols, gate.cast(), up.cast(),
                                           scale.map_or(std::ptr::null(), |scale| scale.cast()), hidden.cast(), out.cast(), logits.cast(),
                                           k, ids.cast(), route_weights.cast())
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

    /// DSpark's target feature of a layer: each row's `hc` copies of `stream` summed in order,
    /// ((x0 + x1) + x2) + x3, over hc; rows from `skip` on, row r into `out` + (r - skip) * `stride` + `offset` floats.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_mean(&mut self, stream: P, rows: u64, hc: u64, width: u64, skip: u64, out: M, stride: u64, offset: u64) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_hc_mean(self.handle(), stream.cast(), rows, hc, width, skip, out.cast(), stride, offset) })
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
    /// `out` = rms_norm(stream collapsed by `collapse`) * `norm_weight`. With `linked` (a part link's
    /// channel, [`link_parts`](Self::link_parts)), x is this device's part of the sublayer's output
    /// and the other device's is added as it is read: the same results as their sum expanded.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_step(&mut self, expand: Option<(P, P, P, P)>, stream: M, coefficients: Option<(P, P, P, M, M, M)>, iterations: u64,
                   eps: f32, norm_eps: f32, collapse: P, norm_weight: P, rows: u64, hc: u64, width: u64, workspace: &Buffer,
                   out: M, linked: Option<u32>) -> Result<()> {
        let null = std::ptr::null::<c_void>();
        let (x, residual, post_in, comb_in) = expand.unwrap_or((null, null, null, null));
        let mut_null = std::ptr::null_mut::<c_void>();
        let (fn_, base, scale, pre, post, comb) = coefficients.unwrap_or((null, null, null, mut_null, mut_null, mut_null));
        // SAFETY: a live handle; the library checks every range against its allocations.
        check(unsafe {
            match linked {
                Some(channel) => ffi::sd_op_hc_step_linked(self.handle(), channel as i32, x.cast(), residual.cast(), post_in.cast(),
                                                           comb_in.cast(), stream.cast(), fn_.cast(), base.cast(), scale.cast(), iterations,
                                                           eps, norm_eps, collapse.cast(), norm_weight.cast(), rows, hc, width,
                                                           workspace.pointer.cast(), workspace.bytes, pre.cast(), post.cast(), comb.cast(),
                                                           out.cast()),
                None => ffi::sd_op_hc_step(self.handle(), x.cast(), residual.cast(), post_in.cast(), comb_in.cast(), stream.cast(),
                                           fn_.cast(), base.cast(), scale.cast(), iterations, eps, norm_eps, collapse.cast(),
                                           norm_weight.cast(), rows, hc, width, workspace.pointer.cast(), workspace.bytes, pre.cast(),
                                           post.cast(), comb.cast(), out.cast()),
            }
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

    /// [`Backend::linear`] of a router (`experts` rows, `steps` contiguous rows of input and logits)
    /// then [`Backend::route`] over its logits, in one launch where the router is F32 or F16 (the
    /// same results as the two).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_route(&mut self, weights: &Buffer, kind: u32, experts: u64, cols: u64, input: P, steps: u64, logits: M,
                        bias: Option<P>, k: u64, gating: i32, normalize: bool, scale: f32, ids: M, route_weights: M,
                        scratch: &Buffer) -> Result<()> {
        // SAFETY: a live handle; the library checks every range against its allocations.
        check(unsafe {
            ffi::sd_op_linear_route(self.handle(), weights.pointer.cast_const(), kind as i32, experts, cols, input.cast(), steps,
                                    logits.cast(), bias.map_or(std::ptr::null(), |b| b.cast()), k, gating, i32::from(normalize),
                                    scale, ids.cast(), route_weights.cast(), scratch.pointer.cast(), scratch.bytes)
        })
    }

    /// [`Backend::linear_route`] in its one launch (an F32 or F16 router of 1-4 rows), the routes also
    /// going straight to this device's route mailbox in page-locked host memory, its flag raised once
    /// every row's are there: [`Backend::route_mailbox_wait`] takes them without a readback.
    /// [`Backend::route_mailbox_clear`] comes before it.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_route_published(&mut self, weights: &Buffer, kind: u32, experts: u64, cols: u64, input: P, steps: u64, logits: M,
                                  bias: Option<P>, k: u64, gating: i32, normalize: bool, scale: f32, ids: M, route_weights: M,
                                  scratch: &Buffer) -> Result<()> {
        // SAFETY: as linear_route.
        check(unsafe {
            ffi::sd_op_linear_route_published(self.handle(), weights.pointer.cast_const(), kind as i32, experts, cols, input.cast(),
                                              steps, logits.cast(), bias.map_or(std::ptr::null(), |b| b.cast()), k, gating,
                                              i32::from(normalize), scale, ids.cast(), route_weights.cast(), scratch.pointer.cast(),
                                              scratch.bytes)
        })
    }

    /// Lower the route mailbox's flag (and make the mailbox), before a published route is queued and
    /// outside any graph being recorded.
    pub fn route_mailbox_clear(&mut self) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_route_mailbox_clear(self.handle()) })
    }

    /// Wait on the host for the published routes and take `out.len()` of them (an error after 60 s,
    /// or when the stream failed).
    pub fn route_mailbox_wait(&mut self, out: &mut [i32]) -> Result<()> {
        // SAFETY: a live handle; `out` holds its count.
        check(unsafe { ffi::sd_route_mailbox_wait(self.handle(), out.as_mut_ptr(), out.len() as u64) })
    }

    /// Zero `buffer` on the compute stream.
    pub fn zero(&mut self, buffer: &Buffer) -> Result<()> {
        // SAFETY: a live handle; the library checks the range lies in its allocations.
        check(unsafe { ffi::sd_op_zero(self.handle(), buffer.pointer, buffer.bytes) })
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

    /// Whether a wave of experts of these kinds and shares takes the grouped WMMA kernels here
    /// (RDNA4, whole tiles and units): see [`SlotWave`].
    pub fn slot_moe_supported(&self, kinds: [u32; 3], width: u64, ffn_share: u64, ffn: u64, hidden_share: u64) -> Result<bool> {
        let kinds = kinds.map(|kind| kind as c_int);
        // SAFETY: a live handle; kinds holds three values.
        let supported = unsafe { ffi::sd_slot_moe_supported(self.handle(), kinds.as_ptr(), width, ffn_share, ffn, hidden_share) };
        if supported < 0 {
            return check(supported).map(|()| false);
        }
        Ok(supported == 1)
    }

    /// A wave's gate and up over the routes' rows (`rows`: each grouped route's row of `input`,
    /// `steps` FP16 rows of the width): clamped SwiGLU times the routes' `weights` into `product`
    /// (FP16 grouped rows of the device's ffn share).
    #[allow(clippy::too_many_arguments)]
    pub fn slot_moe_gate_up(&mut self, wave: &SlotWave<'_>, rows: P, steps: u64, input: P, clamp: f32, weights: P, product: M) -> Result<()> {
        let kinds = wave.kinds.map(|kind| kind as c_int);
        // SAFETY: a live handle; the slices hold `experts` values each; the library checks the
        // buffers' ranges and the slots are the store's.
        check(unsafe {
            ffi::sd_op_slot_moe_gate_up(self.handle(), wave.bases.as_ptr(), wave.firsts.as_ptr(), wave.counts.as_ptr(),
                                        wave.bases.len() as u64, wave.gate_bytes, kinds.as_ptr(), wave.width, wave.ffn_share, wave.ffn,
                                        wave.hidden_share, rows.cast(), steps, input.cast(), clamp, weights.cast(), product.cast(),
                                        wave.tables.pointer, wave.tables.bytes)
        })
    }

    /// A wave's down over its FP16 grouped products (rows of the whole ffn) into `outputs` (F32
    /// grouped rows `out_stride` values apart, the device's hidden share).
    pub fn slot_moe_down(&mut self, wave: &SlotWave<'_>, product: P, outputs: M, out_stride: u64) -> Result<()> {
        let kinds = wave.kinds.map(|kind| kind as c_int);
        // SAFETY: as in slot_moe_gate_up.
        check(unsafe {
            ffi::sd_op_slot_moe_down(self.handle(), wave.bases.as_ptr(), wave.firsts.as_ptr(), wave.counts.as_ptr(),
                                     wave.bases.len() as u64, wave.gate_bytes, kinds.as_ptr(), wave.width, wave.ffn_share, wave.ffn,
                                     wave.hidden_share, product.cast(), outputs.cast(), out_stride, wave.tables.pointer, wave.tables.bytes)
        })
    }

    /// `values` (a multiple of four) of `input` rounded to FP16 into `output`.
    pub fn half_values(&mut self, input: P, values: u64, output: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_half_values(self.handle(), input.cast(), values, output.cast()) })
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

    /// One decode step of a DeltaNet: [`conv_silu`](Self::conv_silu), [`delta_rule`](Self::delta_rule)
    /// and [`gated_norm`](Self::gated_norm) in one launch, the same results (`qkv` keeps its values).
    /// False when not taken (the caller runs the three).
    #[allow(clippy::too_many_arguments)]
    pub fn delta_step(&mut self, qkv: P, history: M, taps: P, kernel: u64, beta: P, alpha: P, dt: P, a: P, state: M, k_heads: u64,
                      v_heads: u64, dim: u64, eps: f32, gate: P, weight: &Buffer, out: M) -> Result<bool> {
        // SAFETY: as above.
        let state_ = unsafe {
            ffi::sd_op_delta_step(self.handle(), qkv.cast(), history.cast(), taps.cast(), kernel, beta.cast(), alpha.cast(), dt.cast(),
                                  a.cast(), state.cast(), k_heads, v_heads, dim, eps, gate.cast(), weight.pointer.cast(), out.cast())
        };
        check(state_.min(0))?;
        Ok(state_ == 0)
    }

    /// rms_norm(x) * weight * SiLU(gate) per row.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_norm(&mut self, x: P, gate: P, weight: &Buffer, rows: u64, width: u64, eps: f32, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_gated_norm(self.handle(), x.cast(), gate.cast(), weight.pointer.cast(), rows, width, eps, out.cast()) })
    }

    /// An attention step's inputs in one launch: [`split`](Self::split) of `q_full` (rows x heads x
    /// [query | gate]) into `q` and `gate`, [`rms_norm`](Self::rms_norm) of the queries and of `k`'s
    /// heads (in place), [`rope_halves`](Self::rope_halves) of both, and [`scatter`](Self::scatter)
    /// of `k` and `v` to rows `positions` of `keys` and `values` (`context` rows each); the same results.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prepare(&mut self, q_full: P, k: M, v: P, rows: u64, heads: u64, kv_heads: u64, dim: u64, q_weight: &Buffer,
                             k_weight: &Buffer, eps: f32, rope_dims: u64, positions: &Buffer, frequencies: &Buffer, q: M, gate: M,
                             keys: M, values: M, context: u64) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_attention_prepare(self.handle(), q_full.cast(), k.cast(), v.cast(), rows, heads, kv_heads, dim,
                                         q_weight.pointer.cast(), k_weight.pointer.cast(), eps, rope_dims, positions.pointer.cast(),
                                         frequencies.pointer.cast(), q.cast(), gate.cast(), keys.cast(), values.cast(), context)
        })
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

    /// Join two devices for [`link_sum`](Self::link_sum), with inboxes of `capacity` floats.
    pub fn link_sums(first: &mut Backend, second: &mut Backend, capacity: u64) -> Result<()> {
        // SAFETY: two live handles.
        check(unsafe { ffi::sd_link_sums(first.handle(), second.handle(), capacity) })
    }

    /// out = this device's part + the linked device's part of the same sum (`count` floats; out may
    /// be the part; with `accumulate`, out += that sum), waiting on the device: a graph may record
    /// it.
    pub fn link_sum(&mut self, part: P, count: u64, out: M, accumulate: bool) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_link_sum(self.handle(), part.cast(), count, out.cast(), i32::from(accumulate)) })
    }

    /// One vector through a matrix into `part`, this device's part of a linked sum then added into
    /// `out` as [`link_sum`](Self::link_sum) adds it (accumulating), in one launch where the GEMV
    /// takes it: the same results as [`linear`](Self::linear) and `link_sum`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_link_sum(&mut self, weights: &Buffer, kind: u32, rows: u64, cols: u64, input: P, part: M, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_linear_link_sum(self.handle(), weights.pointer.cast_const(), kind as i32, rows, cols, input.cast(), part.cast(),
                                       out.cast())
        })
    }

    /// [`experts_table`](Self::experts_table) of one row (clamp 0, accumulating into `part`), then
    /// `part` added into `out` as [`link_sum`](Self::link_sum) adds it, the sum in the down
    /// stage's launch where it takes it: the same results.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_table_link_sum(&mut self, table: &Buffer, count: u64, ids: P, slot_bytes: u64, offsets: [u64; 3], kinds: [u32; 3],
                                  hidden: u64, ffn: u64, top_k: u64, input: P, weights: P, product: M, part: M, out: M) -> Result<()> {
        let kinds = kinds.map(|kind| kind as i32);
        // SAFETY: as experts_table.
        check(unsafe {
            ffi::sd_op_experts_table_link_sum(self.handle(), table.pointer.cast(), count, ids.cast(), slot_bytes, offsets.as_ptr(),
                                              kinds.as_ptr(), hidden, ffn, top_k, input.cast(), weights.cast(), product.cast(), part.cast(),
                                              out.cast())
        })
    }

    /// Queue a copy of the link's failure word into `out` (nonzero after a wait gave up): read with
    /// a pass's results, it tells whether they are void without another wait.
    pub fn link_status(&mut self, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_link_status(self.handle(), out.cast()) })
    }

    /// Wait for both linked devices' streams and start their sums' counts over (after a failed pass).
    pub fn link_reset(&mut self) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_link_reset(self.handle()) })
    }

    /// Development: make the last block's waves past the first of this device's folded linked sums
    /// wait `microseconds` before they add the parts (0: none), to widen the window of races between
    /// them (tests).
    pub fn link_skew(&mut self, microseconds: u64) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_link_skew(self.handle(), microseconds) })
    }

    /// Whether a linked sum's wait gave up (waits for the stream; the sums since are void).
    pub fn link_failed(&mut self) -> Result<bool> {
        // SAFETY: a live handle.
        let state = unsafe { ffi::sd_link_failed(self.handle()) };
        check(state.min(0))?;
        Ok(state == 1)
    }

    /// Join two devices with part links: sums whose producers send this device's part as they form
    /// it ([`linear_send`](Self::linear_send), [`experts`](Self::experts) with a split's `send`) and
    /// whose consumers add the other's as they read it ([`hc_step`](Self::hc_step) linked,
    /// [`part_receive`](Self::part_receive)); three channels, inboxes of `capacity` floats.
    pub fn link_parts(first: &mut Backend, second: &mut Backend, capacity: u64) -> Result<()> {
        // SAFETY: two live handles.
        check(unsafe { ffi::sd_link_parts(first.handle(), second.handle(), capacity) })
    }

    /// [`linear`](Self::linear) of `steps` (at most 4) vectors into `output`, this device's part of
    /// a part link's sum on `channel`, sent to the other device as it forms.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_send(&mut self, channel: u32, weights: &Buffer, kind: u32, rows: u64, cols: u64, input: P, steps: u64, output: M)
                       -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_linear_send(self.handle(), channel as i32, weights.pointer.cast_const(), kind as i32, rows, cols, input.cast(), steps,
                                   output.cast())
        })
    }

    /// Send this device's part of a part link's sum on `channel`, formed before: columns
    /// `first..first + count` of `rows` rows `stride` floats apart.
    pub fn part_send(&mut self, channel: u32, part: P, rows: u64, stride: u64, first: u64, count: u64) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_part_send(self.handle(), channel as i32, part.cast(), rows, stride, first, count) })
    }

    /// out = this device's part + the other's of a part link's sum on `channel` (`count` floats).
    pub fn part_receive(&mut self, channel: u32, part: P, count: u64, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_part_receive(self.handle(), channel as i32, part.cast(), count, out.cast()) })
    }

    /// Queue the part links' failure word into `out` (nonzero after a wait gave up): read with a
    /// pass's results, it tells whether they are void.
    pub fn part_status(&mut self, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_part_status(self.handle(), out.cast()) })
    }

    /// Whether a part link's wait gave up (waits for the stream; the sums since are void).
    pub fn part_failed(&mut self) -> Result<bool> {
        // SAFETY: a live handle.
        let state = unsafe { ffi::sd_part_failed(self.handle()) };
        check(state.min(0))?;
        Ok(state == 1)
    }

    /// Wait for both part-linked devices' streams and start every count over.
    pub fn part_reset(&mut self) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_part_reset(self.handle()) })
    }

    /// Development: queue 64 partial sums of the 32-bit words of `bytes` at `data` (each weighted
    /// by its position) into `partials` (64 u64), to tell a range's contents apart.
    pub fn debug_checksum(&mut self, data: P, bytes: u64, partials: M) -> Result<()> {
        // SAFETY: a live handle; the library checks both ranges against its allocations.
        check(unsafe { ffi::sd_debug_checksum(self.handle(), data.cast(), bytes, partials.cast()) })
    }

    /// out = `count` values of `x` as BF16 (round to nearest, ties to even).
    pub fn to_bf16(&mut self, x: P, count: u64, out: M) -> Result<()> {
        // SAFETY: a live handle; the library checks both ranges against its allocations.
        check(unsafe { ffi::sd_op_to_bf16(self.handle(), x.cast(), count, out.cast()) })
    }

    /// Development: queue a store of the device's wall clock into `stamps[slot]` (u64s).
    pub fn debug_stamp(&mut self, stamps: M, slot: u64) -> Result<()> {
        // SAFETY: a live handle; the library checks the slot against its allocations.
        check(unsafe { ffi::sd_debug_stamp(self.handle(), stamps.cast(), slot) })
    }

    /// Development: the rate of [`debug_stamp`](Self::debug_stamp)'s clock, in kHz.
    pub fn debug_clock_khz(&mut self) -> Result<u64> {
        let mut khz = 0u64;
        // SAFETY: a live handle and a valid output.
        check(unsafe { ffi::sd_debug_clock_khz(self.handle(), &mut khz) })?;
        Ok(khz)
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

    /// The bytes of an entry of an expert table for [`experts_grouped`](Self::experts_grouped).
    pub fn moe_group_bytes() -> u64 {
        // SAFETY: a constant.
        unsafe { ffi::sd_moe_group_bytes() }
    }

    /// Write `groups` (an expert table: an entry per expert of `bases`) from each expert's slot
    /// address (0: an expert another device runs), its gate, up and down at `offsets` bytes in, of
    /// `kinds`; `hidden` and `ffn` the experts' widths.
    pub fn moe_groups(&mut self, groups: &Buffer, bases: &[u64], offsets: [u64; 3], kinds: [u32; 3], hidden: u64, ffn: u64) -> Result<()> {
        if bases.len() as u64 * Self::moe_group_bytes() > groups.bytes {
            return Err("an expert table past its buffer".into());
        }
        let kinds = kinds.map(|kind| kind as i32);
        // SAFETY: a live handle; the library checks the table and every expert's matrices.
        check(unsafe {
            ffi::sd_moe_groups(self.handle(), groups.pointer, bases.as_ptr(), bases.len() as u64, offsets.as_ptr(), kinds.as_ptr(), hidden,
                               ffn)
        })
    }

    /// out += each of `rows` rows' `top_k` routes (`ids`, `weights` on the device) through the
    /// experts of `groups` ([`moe_groups`](Self::moe_groups), `experts` entries) as grouped FP16
    /// GEMMs with F32 sums, routes to experts another device runs left out; `scratch` holds the
    /// grouping and the routes' rows. False when this device does not take them (not RDNA4, or a
    /// shape its tiles do not fit).
    #[allow(clippy::too_many_arguments)]
    pub fn experts_grouped(&mut self, groups: M, experts: u64, ids: P, weights: P, rows: u64, top_k: u64, hidden: u64, ffn: u64,
                           kinds: [u32; 3], input: P, scratch: &Buffer, out: M) -> Result<bool> {
        let kinds = kinds.map(|kind| kind as i32);
        // SAFETY: a live handle; the library checks every range against its allocations.
        let state = unsafe {
            ffi::sd_op_experts_grouped(self.handle(), groups.cast(), experts, ids.cast(), weights.cast(), rows, top_k, hidden, ffn, kinds.as_ptr(),
                                       input.cast(), scratch.pointer, scratch.bytes, out.cast())
        };
        check(state.min(0))?;
        Ok(state == 0)
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

    /// The attention rope table for one pass (`rows` by `rope_dims`), allocated once and kept.
    /// Call it before a captured graph. `rope_dims` 0 reserves nothing.
    pub fn attention_turns_reserve(&mut self, rows: u64, rope_dims: u64) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_attention_turns_reserve(self.handle(), rows, rope_dims) })
    }

    /// The next [`norm_rope`](Self::norm_rope) or [`norm_linears_rope`](Self::norm_linears_rope) of
    /// exactly `rows` (0 clears) writes that table. The next [`sparse_attention`](Self::sparse_attention)
    /// of those rows then does not launch its own rope kernel.
    pub fn attention_turns_arm(&mut self, rows: u64) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_attention_turns_arm(self.handle(), rows) })
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

    /// DeepSeek-V4.1's candidate blocks of each row (its indexer's first level): the `blocks` blocks of
    /// `block` groups whose highest of `scores` (rows x groups, as [`index_select`](Self::index_select)
    /// stores them) is highest, the block of the row's last visible group always among them, in index
    /// order, -1 past the last: `out` rows x blocks (`scratch` holding rows x ceil(groups / block)
    /// floats).
    #[allow(clippy::too_many_arguments)]
    pub fn index_candidates(&mut self, scores: P, visible: P, rows: u64, groups: u64, blocks: u64, block: u64, out: M,
                            scratch: &Buffer) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_index_candidates(self.handle(), scores.cast(), visible.cast(), rows, groups, blocks, block, out.cast(),
                                        scratch.pointer.cast(), scratch.bytes)
        })
    }

    /// [`rope`](Self::rope) of q (`rope` = (dims, positions, frequencies); q itself unchanged),
    /// [`index_scores`](Self::index_scores) and [`top_k`](Self::top_k) of its scores in one launch:
    /// the same scores and indices. With `candidates` = (rows x blocks of
    /// [`index_candidates`](Self::index_candidates), blocks, block), each row's groups scored and
    /// chosen are its candidates' (scores rows x blocks * block, in candidate order).
    #[allow(clippy::too_many_arguments)]
    pub fn index_select(&mut self, q: P, keys: P, weights: P, visible: P, rows: u64, groups: u64, heads: u64, dim: u64, scale: f32,
                        rope: (u64, &Buffer, &Buffer), scores: M, k: u64, out: M, candidates: Option<(P, u64, u64)>) -> Result<()> {
        let (rope_dims, positions, frequencies) = rope;
        let (list, blocks, block) = candidates.unwrap_or((std::ptr::null(), 0, 0));
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_index_select(self.handle(), q.cast(), keys.cast(), weights.cast(), visible.cast(), rows, groups, heads, dim, scale,
                                    rope_dims, positions.pointer.cast_const().cast(), frequencies.pointer.cast_const().cast(),
                                    scores.cast(), k, out.cast(), list.cast(), blocks, block)
        })
    }

    /// [`index_select`](Self::index_select) of one row whose weights it first forms: `weights` = the
    /// `proj` = (weights, kind, cols) matrix (`heads` rows) . `x`, as [`linear`](Self::linear) forms
    /// them, in the same launch where it can.
    #[allow(clippy::too_many_arguments)]
    pub fn index_select_proj(&mut self, proj: (&Buffer, u32, u64), x: P, q: P, keys: P, weights: M, visible: P, groups: u64, heads: u64,
                             dim: u64, scale: f32, rope: (u64, &Buffer, &Buffer), scores: M, k: u64, out: M,
                             candidates: Option<(P, u64, u64)>) -> Result<()> {
        let ((proj_weights, kind, cols), (rope_dims, positions, frequencies)) = (proj, rope);
        let (list, blocks, block) = candidates.unwrap_or((std::ptr::null(), 0, 0));
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_index_select_proj(self.handle(), proj_weights.pointer.cast_const(), kind as i32, cols, x.cast(), q.cast(), keys.cast(),
                                         weights.cast(), visible.cast(), groups, heads, dim, scale, rope_dims,
                                         positions.pointer.cast_const().cast(), frequencies.pointer.cast_const().cast(), scores.cast(), k,
                                         out.cast(), list.cast(), blocks, block)
        })
    }

    pub fn top_k(&mut self, scores: P, rows: u64, groups: u64, k: u64, out: M) -> Result<()> {
        // SAFETY: as above.
        check(unsafe { ffi::sd_op_top_k(self.handle(), scores.cast(), rows, groups, k, out.cast()) })
    }
}
