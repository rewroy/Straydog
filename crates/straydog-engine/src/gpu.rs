//! One GPU: its pinned weights, sequence state and native execution plans.
//!
//! Every weight the engine uses is pinned for the life of the placement (full or partial
//! residency), so nothing is ever evicted; an upload that does not fit the budget is an
//! error. Operands carry their bytes, which are copied only on first use of their key.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_int, c_void};

use indexmap::IndexMap;
use straydog_gpu::ffi;

use crate::formats;

pub type Result<T> = std::result::Result<T, String>;
/// A head's rows: each row's first maximum, its logits and its probability when asked.
pub type Head = (Vec<i32>, Option<Vec<f32>>, Option<Vec<f32>>);

/// The native library pads every VRAM allocation (allocation_tail in kernels.cpp).
pub const ALLOCATION_TAIL: u64 = 64;
const MOE_ROW: u64 = 11 * 4480 + 22;
/// Rows of the peer's batch one pass of expert-parallel help takes (see `@moe:helper`).
const HELP_ROWS: u64 = 2048;
pub const VERIFY_MAX: usize = 4;
pub const VOCABULARY: u64 = 248320;
pub const WIDTH: usize = 10240;
pub const HIDDEN: usize = 2560;

/// The workspace of the fully resident decoder's spans for a prefill capacity.
pub fn decoder_workspace_bytes(capacity: u64) -> u64 {
    (capacity * 15364 + 1024 * 2564 + 8) * 4
}

fn last_error() -> String {
    straydog_gpu::last_error().to_string()
}

fn check(status: c_int) -> Result<c_int> {
    straydog_gpu::check(status).map_err(|error| error.to_string())
}

fn pointer(result: *mut c_void) -> Result<*mut c_void> {
    if result.is_null() { Err(last_error()) } else { Ok(result) }
}

/// An operand's bytes: weights, weights of a GGML kind uploaded decoded to F32 (vectors), or
/// (helper parts of expert parallelism) only its description.
#[derive(Clone)]
pub enum Data<'a> {
    Bytes(&'a [u8]),
    Decode(&'a [u8], u32),
    Described,
}

/// An upload placed before its weights were read: the allocation and the mapped bytes that
/// fill it once their chunk is authenticated (see `crate::load`), decoded to F32 for a kind.
#[derive(Debug, Clone, Copy)]
pub struct Pending {
    pub pointer: *mut c_void,
    pub source: *const u8,
    pub bytes: usize,
    pub decode: Option<u32>,
}

// The source is only an address into the weights' mappings, never read through.
unsafe impl Send for Pending {}

/// A host range page-locked for DMA while this lives; the range must outlive it.
pub struct PageLock(*mut c_void);

// Page locks are process-wide: any thread may release one.
unsafe impl Send for PageLock {}

impl PageLock {
    pub fn new(range: &[u8]) -> Result<PageLock> {
        let start = range.as_ptr().cast_mut().cast();
        // SAFETY: a live range, unregistered (when this drops) before its owner frees it.
        check(unsafe { ffi::sd_host_register(start, range.len() as u64) })?;
        Ok(PageLock(start))
    }
}

impl Drop for PageLock {
    fn drop(&mut self) {
        // SAFETY: registered in `new`.
        unsafe { ffi::sd_host_unregister(self.0) };
    }
}

/// One thread's copies into one device's allocations (see `sd_fill`).
pub struct Filler<'a> {
    handle: *mut c_void,
    device: std::marker::PhantomData<&'a Backend>,
}

// Each filler is used by one thread at a time; fills of distinct fillers may run at once.
unsafe impl Send for Filler<'_> {}

impl Filler<'_> {
    /// Copy page-locked host bytes to `offset` of a pending upload; returns once they landed.
    pub fn fill(&self, upload: &Pending, offset: u64, host: &[u8]) -> Result<()> {
        // SAFETY: a live filler; the library checks the range against the allocation.
        check(unsafe { ffi::sd_fill(self.handle, upload.pointer, offset, host.as_ptr().cast(), host.len() as u64) })
            .map(drop)
    }
}

impl Drop for Filler<'_> {
    fn drop(&mut self) {
        // SAFETY: created by sd_filler_create, destroyed once.
        unsafe { ffi::sd_filler_destroy(self.handle) }
    }
}

#[derive(Clone)]
pub struct Operand<'a> {
    pub key: String,
    pub data: Data<'a>,
    pub kind: u32,
    pub rows: u64,
    pub cols: u64,
}

impl<'a> Operand<'a> {
    pub fn bytes(key: impl Into<String>, data: &'a [u8], kind: u32, rows: u64, cols: u64) -> Operand<'a> {
        Operand { key: key.into(), data: Data::Bytes(data), kind, rows, cols }
    }
}

/// A vector's bytes decoded to `count` finite F32 values.
pub fn decode_vector(bytes: &[u8], kind: u32, count: usize, key: &str) -> Result<Vec<f32>> {
    let mut values = Vec::with_capacity(count);
    formats::decode_into(kind, bytes, &mut values).map_err(|exc| format!("{key}: {exc}"))?;
    if values.len() != count || !values.iter().all(|value| value.is_finite()) {
        return Err("non-finite Straydog model parameters".into());
    }
    Ok(values)
}

fn float_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns as bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

#[derive(Debug, Clone, Copy)]
pub struct Entry {
    pub pointer: *mut c_void,
    pub kind: c_int,
    pub rows: u64,
    pub cols: u64,
}

/// Floats of a sequence state copied out: `height` runs of `width` floats, `pitch` floats apart
/// in the state, from float `offset`; `first` the first position they hold. They lie in `device`
/// when it is given (a VRAM copy, freed by [`Backend::release_positions`]), else in `data`.
pub struct StatePart {
    key: String,
    offset: u64,
    pitch: u64,
    width: u64,
    height: u64,
    first: u64,
    data: Vec<f32>,
    device: Option<crate::ops::Buffer>,
}

/// Floats of an indexer's rolling history after its pooled blocks: its last eight inputs, a ring
/// by position (gpu/sparse_attention.hpp, qsa_history), so that a rewind's rows never overwrite
/// the inputs a block completed after it still pools.
const INDEX_HISTORY: u64 = 8 * 128;

/// Copies of states this small always go to host memory (see `copy_out`).
const HOST_COPY_BYTES: u64 = 256 << 10;

/// VRAM a device keeps free when parked positions go to VRAM instead of host memory.
const PARK_MARGIN: u64 = 512 << 20;

/// One device, its stream and an explicit VRAM budget.
pub struct Backend {
    handle: *mut c_void,
    pub ordinal: usize,
    cache: HashMap<String, Entry>,
    resident: HashSet<String>,
    plans: HashMap<String, *mut c_void>,
    helper_plans: HashMap<String, *mut c_void>,
    decoder_layers: HashSet<usize>,
    states: IndexMap<String, Entry>,
    checkpoints: Vec<*mut c_void>,
    checkpoint_states: Vec<*mut c_void>,
    prefill_capacity: u64,
    slot_bytes: u64,
    registered: Vec<*mut c_void>,
    pub batch_capacity: usize,
    pub logical_capacity: usize,
    pub output_capacity: u64,
    /// The last of two devices that split the layers: it keeps a deferred prompt's MTP pairs
    /// (see [`Backend::mtp_keep`]) when it holds the MTP layer.
    pub keeps_mtp_rows: bool,
    /// One of two devices that split every layer's experts: its help for the peer's batches
    /// runs in a workspace of its own (`@moe:helper`), beside its own batch.
    pub expert_parallel: bool,
    /// Host states page-locking beside the weight upload, created by `finish_model_state`.
    pending_host: Vec<(String, u64, u64)>,
    /// While placing before the weights are read: uploads allocated but not yet filled.
    deferred: Option<Vec<Pending>>,
}

/// Move prompt rows a span left on `from` (layer `from_layer`'s workspace) to `to`'s
/// (`to_layer`'s), device to device.
pub fn prefill_handoff(from: &Backend, to: &Backend, from_layer: usize, to_layer: usize, steps: u64) -> Result<()> {
    // SAFETY: two live devices; the native side checks both workspaces against `steps`.
    check(unsafe { ffi::sd_prefill_handoff(from.handle, to.handle, from_layer as u64, to_layer as u64, steps) })?;
    Ok(())
}

/// The sequence state carried from token to token (DeltaNet, its convolution, PLE history),
/// as opposed to K/V and the indexer, which are indexed by position.
const RECURRENT: [&str; 3] = ["@delta:", "@conv:", "@ple:"];

// The native library orders each device's work on its own stream; the backend moves
// between threads but is never used by two at once.
unsafe impl Send for Backend {}

impl Backend {
    pub fn new(ordinal: usize, budget: u64) -> Result<Backend> {
        if budget < 2 << 20 {
            return Err("Straydog VRAM budget must be at least 2 MiB".into());
        }
        load_iq_tables()?;
        // SAFETY: plain values; null on failure.
        let handle = pointer(unsafe { ffi::sd_create(ordinal as c_int, budget) })?;
        let batch_capacity = if budget >= 256 << 20 { 256 } else { 32 };
        let mut output_capacity = 262144 * if batch_capacity == 256 { 32 } else { 1 };
        if budget >= 2 << 30 {
            output_capacity *= 2;
        }
        Ok(Backend {
            handle,
            ordinal,
            cache: HashMap::new(),
            resident: HashSet::new(),
            plans: HashMap::new(),
            helper_plans: HashMap::new(),
            decoder_layers: HashSet::new(),
            states: IndexMap::new(),
            checkpoints: Vec::new(),
            checkpoint_states: Vec::new(),
            prefill_capacity: 1024,
            slot_bytes: 0,
            registered: Vec::new(),
            batch_capacity,
            logical_capacity: if budget >= 2 << 30 { 4096 } else { 1024 },
            output_capacity,
            keeps_mtp_rows: false,
            expert_parallel: false,
            pending_host: Vec::new(),
            deferred: None,
        })
    }

    pub fn handle(&self) -> *mut c_void {
        self.handle
    }

    /// Allocated bytes and the enforced budget, workspace included.
    pub fn memory(&self) -> Result<(u64, u64)> {
        let (mut used, mut budget) = (0, 0);
        // SAFETY: a live handle and two out-parameters.
        check(unsafe { ffi::sd_memory(self.handle, &mut used, &mut budget) })?;
        Ok((used, budget))
    }

    /// The device's free memory as the runtime reports it (the card's, other programs' included).
    pub fn free_bytes(&self) -> Result<u64> {
        straydog_gpu::device_info(self.ordinal).map(|(_, free, _)| free).map_err(|error| error.to_string())
    }

    /// Enable quality-only observers of FP16 casts once, before graphs are recorded.
    pub fn enable_gemm_checks(&self) -> Result<()> {
        // SAFETY: a live device; native allocation is tracked until this backend drops.
        check(unsafe { ffi::sd_gemm_check_start(self.handle) }).map(drop)
    }

    /// Read cast diagnostics: nonfinite F32/half inputs, overflow, underflow, nonfinite half
    /// weights, input/weight element counts, and maximum finite input magnitude as F32 bits.
    pub fn gemm_checks(&self) -> Result<[u64; 8]> {
        let mut counts = [0u64; 8];
        // SAFETY: a live enabled device and exactly eight output counters.
        check(unsafe { ffi::sd_gemm_check_read(self.handle, counts.as_mut_ptr(), counts.len() as u64) })?;
        Ok(counts)
    }

    pub fn configure_gemm(&self, minimum_batch: u64, half: bool) -> Result<()> {
        if !(2..=257).contains(&minimum_batch) {
            return Err("invalid Straydog GEMM configuration".into());
        }
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_gemm_options(self.handle, minimum_batch, c_int::from(half)) }).map(drop)
    }

    /// Whether a decode row's Q5_K and Q8_0 GEMVs and IQ2 experts' gate/up take the Q8 kernels: their
    /// input quantized per 32 values to INT8 with an F32 scale, the sums exact in INT32 (RDNA4 only;
    /// elsewhere a no-op).
    /// Keep the FP16 expansions of uploaded weights a prompt pass's GEMMs take, in `bytes` of VRAM
    /// (within the budget), for its next batches (see the C header). Once only.
    pub fn keep_expansions(&self, bytes: u64) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_expansions(self.handle, bytes) }).map(drop)
    }

    pub fn configure_decode(&self, q8: bool) -> Result<()> {
        self.decode_options(q8, false)
    }

    /// [`configure_decode`](Self::configure_decode)'s Q8 kernels, and with `half` a decode or verify
    /// pass's index and attention scores through WMMA (FP16 operands, F32 sums; RDNA4 only).
    pub fn decode_options(&self, q8: bool, half: bool) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_decode_options(self.handle, c_int::from(q8) | c_int::from(half) << 1) }).map(drop)
    }

    /// A decode row's Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 GEMVs on the Q8 kernels (INT8 activations, a scale per 32 values),
    /// a list mixing them with BF16 rows in one launch (RDNA4; Qwen3.8's projections). Replaces the other decode options.
    pub fn wide_q8_decode(&self) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_decode_options(self.handle, 4) }).map(drop)
    }

    /// Upload an operand once; later uses of its key reuse the device copy. While uploads are
    /// deferred, mapped bytes are only allocated for (see [`Backend::defer_uploads`]).
    pub fn prepare(&mut self, operand: &Operand) -> Result<Entry> {
        if let Some(entry) = self.cache.get(&operand.key) {
            if (entry.rows, entry.cols) != (operand.rows, operand.cols) {
                return Err("cached Straydog matrix shape mismatch".into());
            }
            return Ok(*entry);
        }
        let size = formats::matrix_bytes(operand.kind, operand.rows as usize, operand.cols as usize)
            .ok_or("unsupported Straydog GPU weight format")?;
        let decoded;
        let (data, decode): (&[u8], Option<u32>) = match &operand.data {
            Data::Bytes(bytes) => (bytes, None),
            Data::Decode(bytes, kind) if self.deferred.is_some() => (bytes, Some(*kind)),
            Data::Decode(bytes, kind) => {
                decoded = decode_vector(bytes, *kind, size / 4, &operand.key)?;
                (float_bytes(&decoded), None)
            }
            Data::Described => return Err("Straydog operand has no data".into()),
        };
        if decode.is_none() && data.len() != size {
            return Err(format!("Straydog GPU weight size mismatch: {}", operand.key));
        }
        let (used, budget) = self.memory()?;
        if used + size as u64 > budget {
            return Err(format!("{} cannot fit in the Straydog VRAM budget", operand.key));
        }
        let device = match (&mut self.deferred, &operand.data) {
            (Some(pending), Data::Bytes(_) | Data::Decode(..)) => {
                // SAFETY: a plain size; the bytes arrive through `fill` or `write`.
                let device = pointer(unsafe { ffi::sd_allocate(self.handle, size as u64) })?;
                pending.push(Pending { pointer: device, source: data.as_ptr(), bytes: data.len(), decode });
                device
            }
            // SAFETY: the bytes are valid for their length; the library copies them.
            _ => pointer(unsafe { ffi::sd_upload(self.handle, data.as_ptr().cast(), data.len() as u64) })?,
        };
        let entry = Entry { pointer: device, kind: operand.kind as c_int, rows: operand.rows, cols: operand.cols };
        self.cache.insert(operand.key.clone(), entry);
        Ok(entry)
    }

    /// Allocate mapped weights instead of uploading them, until [`Backend::take_pending`]:
    /// placement then runs before the weights are read, and reading fills the allocations.
    pub fn defer_uploads(&mut self) {
        self.deferred.get_or_insert_with(Vec::new);
    }

    /// The uploads allocated since [`Backend::defer_uploads`]; later uploads copy at once.
    pub fn take_pending(&mut self) -> Vec<Pending> {
        self.deferred.take().unwrap_or_default()
    }

    /// A filler for one thread's copies into pending uploads.
    pub fn filler(&self) -> Result<Filler<'_>> {
        // SAFETY: a live handle.
        let handle = pointer(unsafe { ffi::sd_filler_create(self.handle) })?;
        Ok(Filler { handle, device: std::marker::PhantomData })
    }

    /// Fill a whole pending upload from caller memory, waiting for it.
    pub fn write(&mut self, upload: &Pending, data: &[u8]) -> Result<()> {
        // SAFETY: the library checks the allocation and its size; the bytes are valid.
        check(unsafe { ffi::sd_write(self.handle, upload.pointer, data.as_ptr().cast(), data.len() as u64) }).map(drop)
    }

    /// Prepare operands and pin them for the life of the placement.
    fn operands(&mut self, operands: &[Operand]) -> Result<(Vec<*mut c_void>, Vec<c_int>)> {
        let mut pointers = Vec::with_capacity(operands.len());
        let mut kinds = Vec::with_capacity(operands.len());
        for operand in operands {
            let entry = self.prepare(operand)?;
            self.resident.insert(operand.key.clone());
            pointers.push(entry.pointer);
            kinds.push(entry.kind);
        }
        Ok((pointers, kinds))
    }

    pub fn pin(&mut self, operands: &[Operand]) -> Result<()> {
        self.operands(operands).map(drop)
    }

    /// A sequence-state allocation, zeroed; in mapped page-locked host memory when `host`.
    fn state(&mut self, key: &str, rows: u64, cols: u64, host: bool) -> Result<Entry> {
        if let Some(entry) = self.states.get(key) {
            if (entry.rows, entry.cols) != (rows, cols) {
                return Err("Straydog state geometry changed".into());
            }
            return Ok(*entry);
        }
        let entry = if host {
            // SAFETY: a live handle; outside the VRAM budget.
            let device = pointer(unsafe { ffi::sd_host_state(self.handle, rows * cols * 4) })?;
            Entry { pointer: device, kind: 0, rows, cols }
        } else {
            let (used, budget) = self.memory()?;
            if used + rows * cols * 4 > budget {
                return Err(format!("{key} cannot fit in the Straydog VRAM budget"));
            }
            // SAFETY: a plain size; the allocation is zeroed on the device, not uploaded.
            let device = pointer(unsafe { ffi::sd_allocate(self.handle, rows * cols * 4) })?;
            // SAFETY: an allocation of this device.
            check(unsafe { ffi::sd_zero(self.handle, device) })?;
            let entry = Entry { pointer: device, kind: 0, rows, cols };
            self.cache.insert(key.to_string(), entry);
            entry
        };
        self.states.insert(key.to_string(), entry);
        Ok(entry)
    }

    fn state_entry(&self, key: &str) -> Result<Entry> {
        self.states.get(key).copied().ok_or_else(|| format!("Straydog state missing: {key}"))
    }

    pub fn host_state_supported(&self) -> Result<bool> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_host_state_supported(self.handle) }).map(|value| value != 0)
    }

    pub fn hybrid_supported(&self) -> Result<bool> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_hybrid_supported(self.handle) }).map(|value| value != 0)
    }

    /// (key, rows, cols, in host memory) for every sequence state of a placement.
    fn state_plan(&self, context: u64, layers: &[usize], prefill: u64, host_attention: bool)
                  -> Result<Vec<(String, u64, u64, bool)>> {
        let mut plan = Vec::new();
        if self.batch_capacity == 256 {
            plan.push(("@moe:workspace".into(), context.min(prefill) * MOE_ROW + 65536, 1, false));
            if self.expert_parallel {
                // Help for batches up to 2048 rows at once, longer ones in pieces.
                plan.push(("@moe:helper".into(), context.min(prefill).min(HELP_ROWS) * MOE_ROW + 65536, 1, false));
            }
            plan.push(("@scratch:qsa".into(), 256 * (context / 4 + 1), 1, false));
        }
        for &layer in layers {
            if layer == 1 && self.output_capacity >= 256 * 33280 {
                plan.push(("@ple:blk.1.".into(), 10, WIDTH as u64, false));
            }
            if layer % 4 != 3 && layer < 48 {
                plan.push((format!("@delta:blk.{layer}."), 48 * 128, 128, false));
                if self.batch_capacity == 256 {
                    plan.push((format!("@conv:blk.{layer}."), 4, WIDTH as u64, false));
                }
            } else {
                plan.push((format!("@qsa:blk.{layer}."), context * 4, 256, host_attention));
                plan.push((format!("@indexer:blk.{layer}."), (context / 4 + 1) * 128 + INDEX_HISTORY, 1, false));
            }
        }
        if self.keeps_mtp_rows && self.batch_capacity == 256 && layers.contains(&48) && !layers.contains(&0) {
            // A batch's rows after the row before it, and the embeddings of the tokens they pair with.
            plan.push(("@mtp:rows".into(), context.min(prefill) + 1, WIDTH as u64, false));
            plan.push(("@mtp:embeddings".into(), context.min(prefill), HIDDEN as u64, false));
        }
        if host_attention && self.batch_capacity == 256 {
            // Up to a sixteenth of the budget: one layer's rows in prefill, per-layer prefix mirrors in decode.
            let rows = context.min(self.memory()?.1 / 16 / 4096);
            if rows >= 256 {
                plan.push(("@kv:stage".into(), rows * 4, 256, false));
            }
        }
        Ok(plan)
    }

    /// Allocate and zero all sequence state of the given layers before advertising capacity.
    /// Host states (K/V in page-locked memory) are only planned: [`Backend::prefetch_host_states`]
    /// starts them on a thread of their own and [`Backend::finish_model_state`] takes them, so
    /// page-locking them overlaps reading the weights.
    pub fn reserve_model_state(&mut self, context: u64, layers: &[usize], prefill: u64, host_attention: bool)
                               -> Result<()> {
        self.prefill_capacity = context.min(prefill);
        for (key, rows, cols, host) in self.state_plan(context, layers, prefill, host_attention)? {
            if host && !self.states.contains_key(&key) {
                self.pending_host.push((key, rows, cols));
            } else {
                self.state(&key, rows, cols, host)?;
            }
        }
        if self.pending_host.is_empty() {
            self.register_kv_stage()?;
        }
        Ok(())
    }

    /// Start page-locking the planned host states. Their thread slows this device's other
    /// calls (some 5 s for 13 GB of K/V), so it starts once placement is done.
    pub fn prefetch_host_states(&self) -> Result<()> {
        let sizes: Vec<u64> = self.pending_host.iter().map(|(_, rows, cols)| rows * cols * 4).collect();
        if !sizes.is_empty() {
            // SAFETY: a live handle; `sizes` outlives the call (the thread copies it).
            check(unsafe { ffi::sd_host_states_prefetch(self.handle, sizes.as_ptr(), sizes.len() as u64) })?;
        }
        Ok(())
    }

    /// Take the host states [`Backend::reserve_model_state`] started, then register the K/V stage.
    pub fn finish_model_state(&mut self) -> Result<()> {
        if self.pending_host.is_empty() {
            return Ok(());
        }
        for (key, rows, cols) in std::mem::take(&mut self.pending_host) {
            self.state(&key, rows, cols, true)?;
        }
        self.register_kv_stage()
    }

    /// The VRAM stage mirrors every registered host state's prefix (see sd_kv_stage).
    fn register_kv_stage(&mut self) -> Result<()> {
        if let Some(stage) = self.states.get("@kv:stage").copied() {
            // SAFETY: the staging allocation made on this device.
            check(unsafe { ffi::sd_kv_stage(self.handle, stage.pointer, stage.rows / 4) })?;
        }
        Ok(())
    }

    /// VRAM a placement would still allocate: its state, then the fully resident decoder workspace.
    pub fn planned_state_bytes(&self, context: u64, layers: &[usize], prefill: u64, host_attention: bool)
                               -> Result<u64> {
        let states: u64 = self
            .state_plan(context, layers, prefill, host_attention)?
            .iter()
            .filter(|(key, _, _, host)| !host && !self.states.contains_key(key))
            .map(|(_, rows, cols, _)| rows * cols * 4 + ALLOCATION_TAIL)
            .sum();
        Ok(states + decoder_workspace_bytes(context.min(prefill)) + ALLOCATION_TAIL)
    }

    /// Whether these weights fit with what is still to be allocated (`reserved`; by default
    /// the decoder workspace), leaving max(512 MiB, 3%) for what the runtime allocates
    /// outside the budget (driver, BLAS, graphs: ~0.9 GiB on an R9700).
    pub fn can_pin(&self, required: u64, reserved: Option<u64>) -> Result<bool> {
        let (used, budget) = self.memory()?;
        let reserved = reserved.unwrap_or(if self.states.contains_key("@scratch:decoder") {
            0
        } else {
            decoder_workspace_bytes(self.prefill_capacity)
        });
        let slack = (512u64 << 20).max(budget * 3 / 100);
        Ok(self.batch_capacity == 256
            && i128::from(required) + i128::from(reserved) <= i128::from(budget) - i128::from(used) - i128::from(slack))
    }

    fn moe_workspace(&mut self, helper: bool) -> Result<(Entry, u64)> {
        if helper && self.expert_parallel {
            let workspace = self.state_entry("@moe:helper")?;
            return Ok((workspace, (workspace.rows - 65536) / MOE_ROW));
        }
        let workspace = match self.states.get("@moe:workspace") {
            Some(entry) => *entry,
            None => self.state("@moe:workspace", 1024 * MOE_ROW + 65536, 1, false)?,
        };
        Ok((workspace, (workspace.rows - 65536) / MOE_ROW))
    }

    /// Pin three packed expert banks (or, under expert parallelism, experts [first, first +
    /// count) of them) with the router and shared expert, and create the layer's native plan.
    /// `helper` registers the part this device evaluates for a layer its peer owns: without
    /// the shared expert, whose three matrices are only described.
    #[allow(clippy::too_many_arguments)]
    pub fn register_resident_moe(&mut self, prefix: &str, operands: &[Operand], experts: u64, first: u64, count: u64,
                                 shared: bool, helper: bool) -> Result<()> {
        let shapes = [(count * 640, 2560), (count * 640, 2560), (count * 2560, 640), (640, 2560), (640, 2560),
                      (2560, 640), (experts, 2560), (1, 2560)];
        let plans = if helper { &self.helper_plans } else { &self.plans };
        if !(10..=512).contains(&experts) || operands.len() != 8 || plans.contains_key(prefix)
            || count == 0 || first + count > experts || helper == shared
            || operands.iter().zip(shapes).any(|(operand, shape)| (operand.rows, operand.cols) != shape)
        {
            return Err("invalid Straydog resident MoE geometry".into());
        }
        let (workspace, capacity) = self.moe_workspace(helper)?;
        let mut pointers = Vec::with_capacity(8);
        let mut kinds = Vec::with_capacity(8);
        for (index, operand) in operands.iter().enumerate() {
            if !shared && (3..6).contains(&index) {
                pointers.push(std::ptr::null_mut());
                kinds.push(operand.kind as c_int);
            } else {
                let entry = self.prepare(operand)?;
                self.resident.insert(operand.key.clone());
                pointers.push(entry.pointer);
                kinds.push(entry.kind);
            }
        }
        // SAFETY: eight operand pointers and kinds on this device; the workspace is this device's.
        let plan = pointer(unsafe {
            if first == 0 && count == experts && shared {
                ffi::sd_resident_moe_create(self.handle, pointers.as_ptr(), kinds.as_ptr(), experts,
                                            workspace.pointer, capacity)
            } else {
                ffi::sd_resident_moe_part(self.handle, pointers.as_ptr(), kinds.as_ptr(), experts, first, count,
                                          c_int::from(shared), workspace.pointer, capacity)
            }
        })?;
        if helper {
            let layer: u64 = prefix.split('.').nth(1).and_then(|layer| layer.parse().ok()).ok_or("invalid prefix")?;
            // SAFETY: the plan was created on this device.
            check(unsafe { ffi::sd_helper_moe(self.handle, layer, plan) })?;
            self.helper_plans.insert(prefix.to_string(), plan);
        } else {
            self.plans.insert(prefix.to_string(), plan);
        }
        Ok(())
    }

    /// Link this device and `peer` for expert parallelism; whether they write each other's VRAM.
    pub fn link_peer(&mut self, peer: &mut Backend, rows: u64) -> Result<bool> {
        // SAFETY: two live handles of distinct devices.
        check(unsafe { ffi::sd_peer_attach(self.handle, peer.handle, rows) })?;
        // SAFETY: a live handle.
        Ok(check(unsafe { ffi::sd_peer_direct(self.handle) })? == 1)
    }

    /// Pin the fixed operands of a decoder layer for GPU-only decode and prefill spans.
    pub fn register_decoder_layer(&mut self, layer: usize, operands: &[Operand], context: u64) -> Result<()> {
        let linear = layer % 4 != 3 && layer < 48;   // layer 48: the MTP draft layer (full attention)
        if layer > 48 || self.decoder_layers.contains(&layer) || operands.len() != if linear { 17 } else { 18 } {
            return Err("invalid Straydog decoder layer".into());
        }
        let prefix = format!("blk.{layer}.");
        let plan = *self.plans.get(&prefix).ok_or("Straydog decoder layer without its MoE plan")?;
        let workspace = self.state("@scratch:decoder", decoder_workspace_bytes(self.prefill_capacity) / 4, 1, false)?;
        let states = if linear {
            [self.state_entry(&format!("@conv:{prefix}"))?.pointer, self.state_entry(&format!("@delta:{prefix}"))?.pointer,
             std::ptr::null_mut()]
        } else {
            [self.state_entry(&format!("@qsa:{prefix}"))?.pointer, self.state_entry(&format!("@indexer:{prefix}"))?.pointer,
             self.state_entry("@scratch:qsa")?.pointer]
        };
        let (pointers, kinds) = self.operands(operands)?;
        // SAFETY: operands, states and workspace all belong to this device.
        check(unsafe {
            ffi::sd_decoder_layer(self.handle, layer as u64, context, pointers.as_ptr(), kinds.as_ptr(),
                                  pointers.len() as u64, states.as_ptr(), workspace.pointer, plan)
        })?;
        self.decoder_layers.insert(layer);
        Ok(())
    }

    fn finite(values: &[f32], what: &str) -> Result<()> {
        if values.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err(format!("Straydog GPU produced non-finite {what}"))
        }
    }

    /// One token through resident layers [begin, end).
    pub fn decode_span(&mut self, begin: usize, end: usize, values: &[f32], position: u64) -> Result<Vec<f32>> {
        if values.len() != WIDTH || begin >= end || end > 49 || position >= 262144 {
            return Err("invalid Straydog decoder span".into());
        }
        let mut output = vec![0f32; WIDTH];
        // SAFETY: one row in and out.
        check(unsafe {
            ffi::sd_decode_span(self.handle, begin as u64, end as u64, position, values.as_ptr(), output.as_mut_ptr())
        })?;
        Self::finite(&output, "decoder state")?;
        Ok(output)
    }

    /// Advance resident prefill rows through [begin, end) (at most four layers). `values`
    /// uploads new rows; `None` continues from this GPU's previous span. The last
    /// `output_rows` rows come back.
    /// Start a prompt in `layer`'s workspace from its embeddings (2560 per row), tiled on the GPU.
    pub fn prefill_embed(&mut self, layer: usize, embeddings: &[f32]) -> Result<()> {
        let steps = embeddings.len() / HIDDEN;
        if embeddings.len() != steps * HIDDEN || !(2..=self.prefill_capacity as usize).contains(&steps)
            || !embeddings.iter().all(|value| value.is_finite())
        {
            return Err("invalid Straydog prefill embeddings".into());
        }
        // SAFETY: steps rows of 2560 floats; the native side checks the workspace.
        check(unsafe { ffi::sd_prefill_embed(self.handle, layer as u64, embeddings.as_ptr(), steps as u64) })?;
        Ok(())
    }

    /// Keep a deferred batch's last-layer rows for the MTP layer after the row before them:
    /// row `previous` of the last batch kept, or `host` (0 and None: no row before).
    pub fn mtp_keep(&mut self, steps: u64, previous: u64, host: Option<&[f32]>) -> Result<()> {
        let rows = self.state_entry("@mtp:rows")?;
        if host.is_some_and(|row| row.len() != WIDTH) || steps + 1 > rows.rows || previous > rows.rows {
            return Err("invalid Straydog MTP rows".into());
        }
        let host = host.map_or(std::ptr::null(), <[f32]>::as_ptr);
        // SAFETY: the kept-rows state of this device; one host row or none.
        check(unsafe { ffi::sd_mtp_keep(self.handle, 47, steps, rows.pointer, previous, host) }).map(drop)
    }

    /// Kept row `index` (the last batch's last is `steps`), read back to the host.
    pub fn mtp_kept_row(&mut self, index: u64) -> Result<Vec<f32>> {
        let rows = self.state_entry("@mtp:rows")?;
        if index >= rows.rows {
            return Err("invalid Straydog MTP row".into());
        }
        let width = WIDTH as u64;
        let mut part = StatePart { key: "@mtp:rows".into(), offset: index * width, pitch: width, width, height: 1, first: 0,
                                   data: vec![0f32; WIDTH], device: None };
        self.state_copy(&mut part, true)?;
        Ok(part.data)
    }

    /// The embeddings the kept rows pair with, staged without waiting for the device.
    pub fn mtp_embed(&mut self, embeddings: &[f32]) -> Result<()> {
        let target = self.state_entry("@mtp:embeddings")?;
        let count = embeddings.len() / HIDDEN;
        if count == 0 || embeddings.len() != count * HIDDEN || count as u64 > target.rows {
            return Err("invalid Straydog MTP embeddings".into());
        }
        // SAFETY: count rows of 2560 floats; the embeddings state holds them.
        check(unsafe { ffi::sd_mtp_embed(self.handle, embeddings.as_ptr(), count as u64, target.pointer) }).map(drop)
    }

    /// `count` MTP pairs from kept row `row` and embedding row `offset`, at `position`.
    pub fn mtp_prompt_kept(&mut self, operands: &[Operand], row: u64, offset: u64, count: u64, position: u64)
                           -> Result<()> {
        let (rows, embeddings) = (self.state_entry("@mtp:rows")?, self.state_entry("@mtp:embeddings")?);
        if count == 0 || row + count > rows.rows || offset + count > embeddings.rows {
            return Err("invalid Straydog MTP prompt rows".into());
        }
        let (pointers, kinds) = self.operands(operands)?;
        // SAFETY: four operands; the kept states hold the rows (checked above and natively).
        check(unsafe {
            ffi::sd_mtp_prompt_kept(self.handle, pointers.as_ptr(), kinds.as_ptr(), rows.pointer, row, embeddings.pointer,
                                    offset, count, position, 48)
        })
        .map(drop)
    }

    /// While on, prefill spans that return no rows end without waiting for the device.
    pub fn prefill_defer(&mut self, on: bool) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_prefill_defer(self.handle, c_int::from(on)) }).map(drop)
    }

    /// Wait for deferred spans, end deferral, and fail if one produced non-finite rows.
    pub fn prefill_settle(&mut self) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_prefill_settle(self.handle) }).map(drop)
    }

    pub fn prefill_layers(&mut self, begin: usize, end: usize, values: Option<&[f32]>, position: u64, steps: u64,
                          output_rows: u64) -> Result<Vec<f32>> {
        if begin >= end || end > 49 || end - begin > 4 || !(2..=self.prefill_capacity).contains(&steps)
            || position > 262144 - steps || output_rows > steps
        {
            return Err("invalid Straydog resident prefill span".into());
        }
        if let Some(values) = values {
            if values.len() as u64 != steps * WIDTH as u64 || !values.iter().all(|value| value.is_finite()) {
                return Err("invalid Straydog resident prefill input".into());
            }
        }
        let mut output = vec![0f32; output_rows as usize * WIDTH];
        let target = if output_rows > 0 { output.as_mut_ptr() } else { std::ptr::null_mut() };
        // SAFETY: input rows (or none) and output rows sized above.
        check(unsafe {
            ffi::sd_prefill_span(self.handle, begin as u64, end as u64, position, steps,
                                 values.map_or(std::ptr::null(), <[f32]>::as_ptr), target, output_rows,
                                 std::ptr::null_mut())
        })?;
        Ok(output)
    }

    fn ple_operands(&mut self, prefix: &str, operands: &[Operand]) -> Result<(Vec<*mut c_void>, Vec<c_int>, Entry)> {
        if operands.len() != 6 || (operands[0].rows, operands[0].cols) != (10240, 2560)
            || (operands[1].rows, operands[1].cols) != (2560, 2560)
        {
            return Err("invalid Straydog PLE geometry".into());
        }
        let state = self.state(&format!("@ple:{prefix}"), 10, WIDTH as u64, false)?;
        let (pointers, kinds) = self.operands(operands)?;
        Ok((pointers, kinds, state))
    }

    /// PLE in place on the rows resident in `layer`'s decoder workspace.
    pub fn prefill_ple(&mut self, prefix: &str, embeddings: &[f32], operands: &[Operand], layer: u64) -> Result<()> {
        let steps = embeddings.len() / HIDDEN;
        if !embeddings.len().is_multiple_of(HIDDEN) || !(2..=4096).contains(&steps) {
            return Err("invalid Straydog PLE geometry".into());
        }
        let (pointers, kinds, state) = self.ple_operands(prefix, operands)?;
        // SAFETY: operands and state on this device; embeddings sized above.
        check(unsafe {
            ffi::sd_prefill_ple(self.handle, pointers.as_ptr(), kinds.as_ptr(), state.pointer, steps as u64,
                                embeddings.as_ptr(), layer)
        })
        .map(drop)
    }

    /// PLE of `values` (rows of 10240) with their n-gram `embeddings`, on this layer's GPU.
    pub fn ple(&mut self, prefix: &str, values: &[f32], embeddings: &[f32], operands: &[Operand]) -> Result<Vec<f32>> {
        let steps = values.len() / WIDTH;
        if !values.len().is_multiple_of(WIDTH) || !(1..=4096).contains(&steps) || embeddings.len() != steps * HIDDEN {
            return Err("invalid Straydog PLE geometry".into());
        }
        let (pointers, kinds, state) = self.ple_operands(prefix, operands)?;
        let mut output = vec![0f32; values.len()];
        for first in (0..steps).step_by(256) {
            let count = (steps - first).min(256);
            // SAFETY: rows [first, first + count) of the three arrays.
            check(unsafe {
                ffi::sd_ple(self.handle, pointers.as_ptr(), kinds.as_ptr(), state.pointer, count as u64,
                            embeddings[first * HIDDEN..].as_ptr(), values[first * WIDTH..].as_ptr(),
                            output[first * WIDTH..].as_mut_ptr())
            })?;
        }
        Self::finite(&output, "PLE state")?;
        Ok(output)
    }

    /// Normalization, the mixing projections and residual weights of a HyperConnection
    /// (the head's: no injection) for rows of 10240; returns the mixed rows of 2560.
    pub fn hyperconnection(&mut self, prefix: &str, values: &[f32], norm: Operand, matrices: &[Operand])
                           -> Result<Vec<f32>> {
        let steps = values.len() / WIDTH;
        if !values.len().is_multiple_of(WIDTH) || !(1..=4096).contains(&steps) || matrices.len() != 2
            || (norm.rows, norm.cols) != (1, WIDTH as u64) || !values.iter().all(|value| value.is_finite())
        {
            return Err("invalid Straydog HyperConnection geometry".into());
        }
        let physical = if self.logical_capacity == 4096 { 1024 } else { 256 };
        let norm = Operand { key: format!("@norm:{prefix}"), ..norm };
        let (mut pointers, mut kinds) = self.operands(&[&[norm], matrices].concat())?;
        pointers.push(std::ptr::null_mut());
        kinds.push(0);
        let mut output = Vec::with_capacity(steps * HIDDEN);
        for first in (0..steps).step_by(physical) {
            let count = (steps - first).min(physical);
            let mut part = vec![0f32; count * HIDDEN];
            // SAFETY: four operand slots (the injection null), rows [first, first + count).
            check(unsafe {
                ffi::sd_hyperconnection(self.handle, pointers.as_ptr(), kinds.as_ptr(), count as u64, 0,
                                        values[first * WIDTH..].as_ptr(), part.as_mut_ptr())
            })?;
            output.extend(part);
        }
        Self::finite(&output, "HyperConnection values")?;
        Ok(output)
    }

    /// Matrix-vector products in one launch: each operand times its input vector.
    pub fn project_many(&mut self, requests: &[(Operand, &[f32])]) -> Result<Vec<Vec<f32>>> {
        let rows_total: u64 = requests.iter().map(|(operand, _)| operand.rows).sum();
        if requests.is_empty() || requests.len() > self.batch_capacity || rows_total > self.output_capacity {
            return Err("Straydog matrix batch exceeds workspace limits".into());
        }
        let mut inputs = Vec::new();
        for (operand, vector) in requests {
            if operand.rows == 0 || operand.rows > 262144 || operand.cols == 0 || operand.cols > 16384
                || vector.len() as u64 != operand.cols || !vector.iter().all(|value| value.is_finite())
            {
                return Err("invalid Straydog input vector".into());
            }
            inputs.extend_from_slice(vector);
        }
        let mut entries = Vec::with_capacity(requests.len());
        for (operand, _) in requests {
            entries.push(self.prepare(operand)?);
        }
        let pointers: Vec<*mut c_void> = entries.iter().map(|entry| entry.pointer).collect();
        let kinds: Vec<c_int> = entries.iter().map(|entry| entry.kind).collect();
        let rows: Vec<u64> = entries.iter().map(|entry| entry.rows).collect();
        let cols: Vec<u64> = entries.iter().map(|entry| entry.cols).collect();
        let mut output = vec![0f32; rows_total as usize];
        // SAFETY: descriptors of `count` entries; inputs and outputs sized from them.
        check(unsafe {
            ffi::sd_matvec_batch(self.handle, pointers.as_ptr().cast(), kinds.as_ptr(), rows.as_ptr(), cols.as_ptr(),
                                 entries.len() as u64, inputs.as_ptr(), inputs.len() as u64, output.as_mut_ptr(),
                                 output.len() as u64)
        })?;
        Self::finite(&output, "values")?;
        let mut results = Vec::with_capacity(entries.len());
        let mut offset = 0;
        for rows in rows {
            results.push(output[offset..offset + rows as usize].to_vec());
            offset += rows as usize;
        }
        Ok(results)
    }

    fn rows(values: &[f32], width: usize) -> Result<usize> {
        let count = values.len() / width;
        if !values.len().is_multiple_of(width) || !(1..=VERIFY_MAX).contains(&count) || !values.iter().all(|value| value.is_finite()) {
            return Err("invalid Straydog speculative rows".into());
        }
        Ok(count)
    }

    /// Up to four consecutive tokens through resident layers [begin, end).
    pub fn verify_span(&mut self, begin: usize, end: usize, values: &[f32], position: u64) -> Result<Vec<f32>> {
        let steps = Self::rows(values, WIDTH)?;
        if begin >= end || end > 49 || position > 262144 - steps as u64 {
            return Err("invalid Straydog verification span".into());
        }
        let mut output = vec![0f32; values.len()];
        // SAFETY: `steps` rows in and out.
        check(unsafe {
            ffi::sd_verify_span(self.handle, begin as u64, end as u64, position, steps as u64, values.as_ptr(),
                                output.as_mut_ptr())
        })?;
        Self::finite(&output, "decoder state")?;
        Ok(output)
    }

    /// Up to four tokens through resident layers [begin, end) with nothing waited for or read back: their rows stay in the
    /// workspace for [`Backend::ple_front`] to continue from.
    pub fn verify_queue(&mut self, begin: usize, end: usize, values: &[f32], position: u64) -> Result<()> {
        let steps = Self::rows(values, WIDTH)?;
        if begin >= end || end > 49 || position > 262144 - steps as u64 {
            return Err("invalid Straydog verification span".into());
        }
        // SAFETY: `steps` rows in.
        check(unsafe { ffi::sd_verify_queue(self.handle, begin as u64, end as u64, position, steps as u64, values.as_ptr()) })
            .map(drop)
    }

    /// The PLE layer over the rows [`Backend::verify_queue`] left in `layer`'s workspace, in one wait: the PLE state is
    /// snapshotted first (for [`Backend::ple_replay`]); returns the first layer's rows (kept on the device too) and the
    /// PLE layer's.
    pub fn ple_front(&mut self, prefix: &str, layer: usize, embeddings: &[f32], operands: &[Operand])
                     -> Result<(Vec<f32>, Vec<f32>)> {
        let steps = Self::rows(embeddings, HIDDEN)?;
        let state = self.state(&format!("@ple:{prefix}"), 10, WIDTH as u64, false)?;
        let snapshot = self.state(&format!("@ple-snapshot:{prefix}"), 10, WIDTH as u64, false)?;
        let kept = self.state(&format!("@ple-rows:{prefix}"), VERIFY_MAX as u64, WIDTH as u64, false)?;
        let (pointers, kinds) = self.operands(operands)?;
        let (mut first, mut hidden) = (vec![0f32; steps * WIDTH], vec![0f32; steps * WIDTH]);
        // SAFETY: `steps` rows of each array; the states are this device's.
        check(unsafe {
            ffi::sd_ple_front(self.handle, pointers.as_ptr(), kinds.as_ptr(), state.pointer, snapshot.pointer, kept.pointer,
                              layer as u64, steps as u64, embeddings.as_ptr(), first.as_mut_ptr(), hidden.as_mut_ptr())
        })?;
        Self::finite(&hidden, "decoder state")?;
        Ok((first, hidden))
    }

    /// After a partial acceptance: the PLE state back as [`Backend::ple_front`] snapshotted it, then the layer over the
    /// first rows' first `kept` rows (queued, not waited for).
    pub fn ple_replay(&mut self, prefix: &str, embeddings: &[f32], operands: &[Operand]) -> Result<()> {
        let steps = Self::rows(embeddings, HIDDEN)?;
        let state = self.state_entry(&format!("@ple:{prefix}"))?;
        let snapshot = self.state_entry(&format!("@ple-snapshot:{prefix}"))?;
        let kept = self.state_entry(&format!("@ple-rows:{prefix}"))?;
        let (pointers, kinds) = self.operands(operands)?;
        // SAFETY: `steps` rows of the embeddings; the states are this device's.
        check(unsafe {
            ffi::sd_ple_replay(self.handle, pointers.as_ptr(), kinds.as_ptr(), state.pointer, snapshot.pointer, kept.pointer,
                               steps as u64, embeddings.as_ptr())
        })
        .map(drop)
    }

    /// Keep the first `kept` tokens of the last verification of this span.
    pub fn verify_commit(&mut self, begin: usize, end: usize, kept: u64) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_verify_commit(self.handle, begin as u64, end as u64, kept) }).map(drop)
    }

    /// The head's mixing and vocabulary rows: each row's first maximum, and when asked its
    /// logits and its probability.
    pub fn output_head(&mut self, operands: &[Operand], values: &[f32], logits: bool, confidence: bool,
                       split: Option<(u64, *mut c_void)>) -> Result<Head> {
        let steps = Self::rows(values, WIDTH)?;
        let vocabulary = operands.get(3).map_or(0, |operand| operand.rows);
        if operands.len() != 4 || operands[3].cols != HIDDEN as u64 || !(1..=VOCABULARY).contains(&vocabulary)
            || split.is_some_and(|(split, _)| split == 0 || split >= vocabulary)
        {
            return Err("invalid Straydog output head operands".into());
        }
        let (pointers, kinds) = self.operands(operands)?;
        let mut tokens = vec![0i32; steps];
        let mut scores = logits.then(|| vec![0f32; steps * vocabulary as usize]);
        let target = scores.as_mut().map_or(std::ptr::null_mut(), |scores| scores.as_mut_ptr());
        let mut top = confidence.then(|| vec![0f32; steps]);
        let probabilities = top.as_mut().map_or(std::ptr::null_mut(), |top| top.as_mut_ptr());
        // SAFETY: four operands; `steps` rows in; tokens and optional logits and probabilities sized above.
        check(unsafe {
            match split {
                None => ffi::sd_output_head(self.handle, pointers.as_ptr(), kinds.as_ptr(), steps as u64, vocabulary,
                                            values.as_ptr(), tokens.as_mut_ptr(), target, probabilities),
                Some((split, peer)) => ffi::sd_output_head_split(self.handle, pointers.as_ptr(), kinds.as_ptr(),
                                                                 steps as u64, vocabulary, split, peer,
                                                                 values.as_ptr(), tokens.as_mut_ptr(), target,
                                                                 probabilities),
            }
        })?;
        Ok((tokens, scores, top))
    }

    /// MTP layer inputs from token embeddings (rows of 2560) and backbone hidden rows.
    pub fn mtp_prologue(&mut self, operands: &[Operand], embeddings: &[f32], hidden: &[f32]) -> Result<Vec<f32>> {
        let steps = hidden.len() / WIDTH;
        if !hidden.len().is_multiple_of(WIDTH) || !(1..=512).contains(&steps) || embeddings.len() != steps * HIDDEN
            || operands.len() != 4
        {
            return Err("invalid Straydog MTP prologue".into());
        }
        let (pointers, kinds) = self.operands(operands)?;
        let mut output = vec![0f32; hidden.len()];
        // SAFETY: `steps` rows of each array.
        check(unsafe {
            ffi::sd_mtp_prologue(self.handle, pointers.as_ptr(), kinds.as_ptr(), steps as u64, embeddings.as_ptr(),
                                 hidden.as_ptr(), output.as_mut_ptr())
        })?;
        Ok(output)
    }

    /// One drafting step on this device: the MTP layer's prologue over `embeddings` and the hidden rows, the layer, and
    /// (with the head's four operands after the prologue's four) the vocabulary head on the layer's last row, in one
    /// call with no wait on the host between them. The hidden rows are `hidden`, or (`None`) the last of the
    /// `previous` rows the layer's last step left in its workspace. The head's token and its probability come back.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_step(&mut self, operands: &[Operand], layer: usize, position: u64, embeddings: &[f32], hidden: Option<&[f32]>,
                      previous: u64, split: Option<(u64, *mut c_void)>) -> Result<Option<(i32, f32)>> {
        let steps = hidden.map_or(1, |hidden| hidden.len() / WIDTH);
        let head = operands.len() == 8;
        let vocabulary = operands.get(7).map_or(0, |operand| operand.rows);
        if (operands.len() != 4 && !head) || hidden.is_some_and(|hidden| !hidden.len().is_multiple_of(WIDTH))
            || !(1..=VERIFY_MAX).contains(&steps) || embeddings.len() != steps * HIDDEN
            || position > 262144 - steps as u64 || hidden.is_none() && !(1..=VERIFY_MAX as u64).contains(&previous)
            || hidden.is_some_and(|hidden| !hidden.iter().all(|value| value.is_finite()))
            || head && (operands[7].cols != HIDDEN as u64 || !(1..=VOCABULARY).contains(&vocabulary)
                        || split.is_some_and(|(split, _)| split == 0 || split >= vocabulary))
        {
            return Err("invalid Straydog drafting step".into());
        }
        let (pointers, kinds) = self.operands(operands)?;
        let (mut token, mut confidence) = (0i32, 0f32);
        let (split, peer) = split.map_or((0, std::ptr::null_mut()), |(split, peer)| (split, peer));
        // SAFETY: four operands of the prologue and, with a head, four of the head; `steps` rows of each array.
        check(unsafe {
            ffi::sd_draft_step(self.handle, pointers.as_ptr(), kinds.as_ptr(), layer as u64, position, steps as u64,
                               embeddings.as_ptr(), hidden.map_or(std::ptr::null(), <[f32]>::as_ptr), previous,
                               if head { pointers[4..].as_ptr() } else { std::ptr::null() },
                               if head { kinds[4..].as_ptr() } else { std::ptr::null() }, vocabulary, split, peer,
                               &mut token, &mut confidence)
        })?;
        if !head {
            return Ok(None);
        }
        Self::finite(&[confidence], "draft probability")?;
        Ok(Some((token, confidence)))
    }

    /// Feed the MTP layer one batch of prompt pairs, the hidden rows coming from `stash`'s
    /// decoder workspace (see `sd_mtp_prompt_rows`).
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_prompt_rows(&mut self, stash: &Backend, operands: &[Operand], stash_layer: usize, first: usize,
                           count: usize, previous: Option<&[f32]>, embeddings: &[f32], position: u64, layer: usize)
                           -> Result<()> {
        if count == 0 || count > 512 || embeddings.len() != count * HIDDEN || operands.len() != 4
            || previous.is_some_and(|row| row.len() != WIDTH)
        {
            return Err("invalid Straydog MTP prompt rows".into());
        }
        let (pointers, kinds) = self.operands(operands)?;
        // SAFETY: `count` embedding rows, one previous row or none; the native side checks both
        // workspaces and the operands.
        check(unsafe {
            ffi::sd_mtp_prompt_rows(self.handle, pointers.as_ptr(), kinds.as_ptr(), stash.handle, stash_layer as u64,
                                    first as u64, count as u64, previous.map_or(std::ptr::null(), <[f32]>::as_ptr),
                                    embeddings.as_ptr(), position, layer as u64)
        })?;
        Ok(())
    }

    /// PLE for up to four rows with the one-row arithmetic (speculative verification).
    pub fn ple_rows(&mut self, prefix: &str, values: &[f32], embeddings: &[f32], operands: &[Operand]) -> Result<Vec<f32>> {
        let steps = Self::rows(values, WIDTH)?;
        if Self::rows(embeddings, HIDDEN)? != steps {
            return Err("invalid Straydog speculative rows".into());
        }
        let state = self.state_entry(&format!("@ple:{prefix}"))?;
        let (pointers, kinds) = self.operands(operands)?;
        let mut output = vec![0f32; values.len()];
        // SAFETY: `steps` rows of each array; the state is this device's.
        check(unsafe {
            ffi::sd_ple_rows(self.handle, pointers.as_ptr(), kinds.as_ptr(), state.pointer, steps as u64,
                             embeddings.as_ptr(), values.as_ptr(), output.as_mut_ptr())
        })?;
        Ok(output)
    }

    /// Copy one state allocation onto another of the same geometry (made on first use).
    pub fn copy_state(&mut self, target: &str, source: &str) -> Result<()> {
        let source = self.state_entry(source)?;
        let target = self.state(target, source.rows, source.cols, false)?;
        // SAFETY: two allocations of this device with the same size.
        check(unsafe { ffi::sd_copy_allocation(self.handle, target.pointer, source.pointer) }).map(drop)
    }

    /// Start a new sequence: zero the recurrent state. K/V and the indexer are read only at
    /// positions already written in the sequence (as a checkpoint restore relies on), so their
    /// gigabytes stay as they are.
    pub fn reset_state(&mut self) -> Result<()> {
        for (key, entry) in &self.states {
            if RECURRENT.iter().any(|prefix| key.starts_with(prefix)) {
                // SAFETY: an allocation of this device.
                check(unsafe { ffi::sd_zero(self.handle, entry.pointer) })?;
            }
        }
        Ok(())
    }

    /// `count` page-locked host copies of the recurrent state (DeltaNet, its convolution, PLE
    /// history); their bytes. K/V and the indexer are indexed by position and need none.
    pub fn reserve_checkpoints(&mut self, count: usize) -> Result<u64> {
        let keys: Vec<&String> =
            self.states.keys().filter(|key| RECURRENT.iter().any(|prefix| key.starts_with(prefix))).collect();
        self.checkpoint_states = keys.iter().map(|key| self.states[*key].pointer).collect();
        let size: u64 = keys.iter().map(|key| self.states[*key].rows * self.states[*key].cols * 4).sum();
        self.checkpoints.clear();
        if !self.checkpoint_states.is_empty() {
            for _ in 0..count {
                // SAFETY: a live handle.
                self.checkpoints.push(pointer(unsafe { ffi::sd_checkpoint_buffer(self.handle, size) })?);
            }
        }
        Ok(size * self.checkpoints.len() as u64)
    }

    /// Bytes [`save_positions`](Self::save_positions) copies per position.
    pub fn position_bytes(&self) -> u64 {
        self.states.keys().map(|key| if key.starts_with("@qsa:") { 4096 } else if key.starts_with("@indexer:") { 128 } else { 0 }).sum()
    }

    /// What a later sequence overwrites from position `from` up to `to`, copied out: each
    /// attention layer's K/V rows (of its four planes), and its indexer's pooled blocks (from the block holding `from`)
    /// with, if `history`, its rolling history. [`restore_positions`](Self::restore_positions)
    /// puts them back.
    pub fn save_positions(&mut self, from: u64, to: u64, history: bool) -> Result<Vec<StatePart>> {
        if from >= to {
            return Ok(Vec::new());
        }
        let mut parts = Vec::new();
        for (key, entry) in &self.states {
            let floats = entry.rows * entry.cols;
            if key.starts_with("@qsa:") {
                // Four planes (each head's keys, then each head's values), a position's 256 values a row.
                let context = entry.rows / 4;
                parts.push(StatePart { key: key.clone(), offset: from * 256, pitch: context * 256, width: (to - from) * 256,
                                       height: 4, first: from, data: Vec::new(), device: None });
            } else if key.starts_with("@indexer:") {
                let pool = (floats - INDEX_HISTORY) / 128;
                let (first, last) = (from / 4, to.div_ceil(4));
                parts.push(StatePart { key: key.clone(), offset: first, pitch: pool, width: last - first, height: 128, first: from,
                                       data: Vec::new(), device: None });
                if history {
                    parts.push(StatePart { key: key.clone(), offset: 128 * pool, pitch: INDEX_HISTORY, width: INDEX_HISTORY, height: 1, first: from,
                                           data: Vec::new(), device: None });
                }
            }
        }
        self.copy_out(parts)
    }

    /// The indexers' rolling pooling history alone (their last eight inputs), copied out as
    /// [`save_positions`](Self::save_positions) copies a range's with `history` and restored the
    /// same way: a parked sequence's, taken when it departs, before a later prompt's prefill
    /// changes it.
    /// `layer` ("blk.48.") keeps one layer's alone.
    pub fn save_history(&mut self, layer: Option<&str>) -> Result<Vec<StatePart>> {
        let mut parts = Vec::new();
        for (key, entry) in &self.states {
            if key.starts_with("@indexer:") && layer.is_none_or(|layer| key.ends_with(layer)) {
                let pool = (entry.rows * entry.cols - INDEX_HISTORY) / 128;
                parts.push(StatePart { key: key.clone(), offset: 128 * pool, pitch: INDEX_HISTORY, width: INDEX_HISTORY, height: 1, first: 0,
                                       data: Vec::new(), device: None });
            }
        }
        self.copy_out(parts)
    }

    /// Copy `parts` out of their states: into VRAM when the budget has room, else host memory.
    fn copy_out(&mut self, mut parts: Vec<StatePart>) -> Result<Vec<StatePart>> {
        if parts.is_empty() {
            return Ok(parts);
        }
        // In VRAM when the budget has room: a copy on the device. The host copy of a 20K-token
        // Qwen3.8 conversation (~0.5 GB a device, into pageable memory, device after device) held
        // the next prompt's first token back by ~0.12 s.
        // A few kilobytes (rolling histories) go to host memory: a VRAM allocation and its
        // release cost more than their copy.
        let floats: u64 = parts.iter().map(|part| part.width * part.height).sum();
        let (used, budget) = self.memory()?;
        let in_vram = floats * 4 > HOST_COPY_BYTES && used + floats * 4 + PARK_MARGIN <= budget;
        let copied = (|| {
            for part in &mut parts {
                let count = part.width * part.height;
                if in_vram {
                    part.device = Some(self.buffer_unset(count * 4)?);
                } else {
                    part.data = vec![0f32; count as usize];
                }
                self.state_copy(part, true)?;
            }
            Ok(())
        })();
        match copied {
            Ok(()) => Ok(parts),
            Err(error) => {
                // (the error is the copy's: a failed release behind it is not reported)
                let _ = self.release_positions(parts);
                Err(error)
            }
        }
    }

    /// Diagnostics: digests of the sequence state that is valid below position `to`, one line a
    /// part ("<state> <part> <sha256>"): each attention layer's K/V rows below it (of its four
    /// planes) in runs of 256 positions and its last 8 rows one by one, each indexer's complete
    /// pooled blocks below it (runs, then its last 4 blocks) and the inputs of its rolling
    /// history from the last block boundary, and each recurrent state. Never the rows above, the
    /// scratch states or the parked copies, nor two kinds of bytes a sequence leaves that
    /// nothing reads before rewriting them:
    /// - the MTP layer's (layer 48's) row `to - 1`: its rows are pairs, a position's hidden
    ///   state with the token after it, and that token is not in the sequence yet. The next
    ///   prompt writes its pairs from the row before it (`first = start - 1` in model.rs
    ///   mtp_prompt :1196, mtp_prompt_resident :1145, mtp_prompt_kept :1171; a one-token
    ///   prompt through them too, :998-1002; inside a pipelined prompt from the row kept on
    ///   the MTP layer's device, kept_hidden), and a decode's first draft catches up from there
    ///   (mtp_position, model.rs:1389), before attending.
    /// - the rolling history's inputs below the last block boundary (another block's, perhaps
    ///   another sequence's): pooling reads it only for positions from the boundary of the
    ///   block being completed up to where the batch starts (qsa_pool, gpu/sparse_attention.hpp
    ///   :53-60; qsa_pool_one, gpu/qsa_decode.hpp:3-9, the block's earlier inputs).
    pub fn state_digest(&mut self, to: u64) -> Result<Vec<String>> {
        use sha2::Digest;
        // (K/V rows and pooled blocks in runs of this many positions, to tell where they differ)
        const RUN: u64 = 256;
        let mut parts = Vec::new();
        for (key, entry) in &self.states {
            let floats = entry.rows * entry.cols;
            let to = if key.ends_with(":blk.48.") { to.saturating_sub(1) } else { to };
            let part = |offset, pitch, width, height| StatePart { key: key.clone(), offset, pitch, width, height, first: 0,
                                                               data: Vec::new(), device: None };
            if key.starts_with("@qsa:") {
                for low in (0..to).step_by(RUN as usize) {
                    let high = (low + RUN).min(to);
                    parts.push((format!("{key} rows {low}..{high}"), part(low * 256, entry.rows / 4 * 256, (high - low) * 256, 4)));
                }
                // (and the last few one by one: where a sequence's writes end)
                for row in to.saturating_sub(8)..to {
                    parts.push((format!("{key} row {row}"), part(row * 256, entry.rows / 4 * 256, 256, 4)));
                }
            } else if key.starts_with("@indexer:") {
                let pool = (floats - INDEX_HISTORY) / 128;
                for low in (0..to / 4).step_by((RUN / 4) as usize) {
                    let high = (low + RUN / 4).min(to / 4);
                    parts.push((format!("{key} blocks {low}..{high}"), part(low, pool, high - low, 128)));
                }
                for block in (to / 4).saturating_sub(4)..to / 4 {
                    parts.push((format!("{key} block {block}"), part(block, pool, 1, 128)));
                }
                for position in to / 4 * 4..to {
                    parts.push((format!("{key} history {position}"), part(128 * pool + position % 8 * 128, 128, 128, 1)));
                }
            } else if RECURRENT.iter().any(|prefix| key.starts_with(prefix)) {
                parts.push((key.clone(), part(0, floats, floats, 1)));
            }
        }
        let mut digests = Vec::with_capacity(parts.len());
        for (label, mut part) in parts {
            part.data = vec![0f32; (part.width * part.height) as usize];
            self.state_copy(&mut part, true)?;
            digests.push(format!("{label} {:x}", sha2::Sha256::digest(bytemuck_bytes(&part.data))));
        }
        Ok(digests)
    }

    /// Put parts [`save_positions`](Self::save_positions) copied out back where they were.
    pub fn restore_positions(&mut self, parts: &mut [StatePart]) -> Result<()> {
        parts.iter_mut().try_for_each(|part| self.state_copy(part, false))
    }

    /// Free the VRAM of parts [`save_positions`](Self::save_positions) copied out (host parts
    /// free themselves).
    pub fn release_positions(&mut self, parts: Vec<StatePart>) -> Result<()> {
        // Every part, a failed one too (the first error returned).
        let mut result = Ok(());
        for buffer in parts.into_iter().filter_map(|part| part.device) {
            let released = self.release_buffer(buffer);
            if result.is_ok() {
                result = released;
            }
        }
        result
    }

    fn state_copy(&mut self, part: &mut StatePart, save: bool) -> Result<()> {
        let entry = self.states.get(&part.key).ok_or("no such Straydog state")?;
        let count = part.width * part.height;
        let copy: *mut f32 = match &part.device {
            Some(buffer) if buffer.bytes == count * 4 => buffer.base().cast(),
            None if part.data.len() as u64 == count => part.data.as_mut_ptr(),
            _ => return Err("a Straydog state part of the wrong size".into()),
        };
        // SAFETY: an allocation of this device and a device or host buffer of width × height
        // floats (the native side checks the runs against the allocation).
        check(unsafe {
            ffi::sd_state_copy(self.handle, entry.pointer, part.offset, part.pitch, part.width, part.height, copy,
                               part.first, c_int::from(save))
        }).map(drop)
    }

    pub fn checkpoint(&mut self, slot: usize, save: bool) -> Result<()> {
        if let Some(buffer) = self.checkpoints.get(slot) {
            // SAFETY: the states and buffer were reserved together on this device.
            check(unsafe {
                ffi::sd_checkpoint_copy(self.handle, c_int::from(save), self.checkpoint_states.as_ptr(),
                                        self.checkpoint_states.len() as u64, *buffer)
            })?;
        }
        Ok(())
    }

    // ---- partial residency ----

    /// Pin a layer's shared expert and router and bind its authenticated host expert banks.
    pub fn register_hybrid_moe(&mut self, prefix: &str, operands: &[Operand], banks: [&[u8]; 3], experts: u64,
                               kinds: [u32; 3]) -> Result<()> {
        let shapes = [(640, 2560), (640, 2560), (2560, 640), (experts, 2560), (1, 2560)];
        let sizes = [formats::matrix_bytes(kinds[0], 640, 2560), formats::matrix_bytes(kinds[1], 640, 2560),
                     formats::matrix_bytes(kinds[2], 2560, 640)];
        if kinds.iter().any(|kind| !formats::CPU_EXPERT_KINDS.contains(kind)) {
            return Err("unsupported Straydog hybrid expert format".into());
        }
        if !(10..=512).contains(&experts) || operands.len() != 5 || self.plans.contains_key(prefix)
            || operands.iter().zip(shapes).any(|(operand, shape)| (operand.rows, operand.cols) != shape)
            || banks.iter().zip(sizes).any(|(bank, size)| size.is_none_or(|size| bank.len() as u64 != experts * size as u64))
        {
            return Err("invalid Straydog hybrid MoE geometry".into());
        }
        let workspace = self.state_entry("@moe:workspace")?;
        let capacity = (workspace.rows - 65536) / MOE_ROW;
        let (pointers, dense_kinds) = self.operands(operands)?;
        let host: Vec<*const u8> = banks.iter().map(|bank| bank.as_ptr()).collect();
        let routed: Vec<c_int> = kinds.iter().map(|kind| *kind as c_int).collect();
        // SAFETY: the banks stay mapped while this device lives (the model drops it first).
        let plan = pointer(unsafe {
            ffi::sd_hybrid_moe_create(self.handle, pointers.as_ptr(), dense_kinds.as_ptr(), routed.as_ptr(), experts,
                                      host.as_ptr(), workspace.pointer, capacity)
        })?;
        self.plans.insert(prefix.to_string(), plan);
        self.slot_bytes = self.slot_bytes.max(expert_bytes(kinds));
        Ok(())
    }

    /// Give the remaining budget to expert cache slots; the cache slot count.
    pub fn create_expert_slots(&mut self, stage: u64, margin: u64) -> Result<u64> {
        let slot = (if self.slot_bytes > 0 { self.slot_bytes } else { expert_bytes([42, 42, 42]) }).div_ceil(256) * 256;
        let (used, budget) = self.memory()?;
        let free = i128::from(budget) - i128::from(used) - i128::from(margin) - i128::from(stage * slot) - 8192 * 4;
        let cache = (free.max(0) as u64) / slot;
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_expert_slots(self.handle, cache, stage, slot) })?;
        Ok(cache)
    }

    /// Delegate absent routed experts to `engine`, which must outlive this device.
    pub fn attach_cpu(&mut self, engine: &CpuExperts) -> Result<()> {
        // SAFETY: a live CPU expert engine; the model drops its devices first.
        check(unsafe { ffi::sd_hybrid_attach_cpu(self.handle, engine.handle()) }).map(drop)
    }

    /// Cache the highest-scoring experts by recent routing; 0 refreshes everything and waits.
    pub fn rebalance(&mut self, max_swaps: u64) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_hybrid_rebalance(self.handle, max_swaps) }).map(drop)
    }

    pub fn enable_assist(&mut self, routes: u32) -> Result<u64> {
        let mut mapped = 0;
        // SAFETY: a live handle and an out-parameter.
        check(unsafe { ffi::sd_hybrid_assist(self.handle, routes, &mut mapped) })?;
        Ok(mapped)
    }

    /// Page-lock a mapped range for direct DMA; false if refused.
    pub fn register_host(&mut self, range: &[u8]) -> bool {
        // SAFETY: a mapped range that outlives the registration (unregistered before unmapping).
        let refused = unsafe { ffi::sd_host_register(range.as_ptr().cast_mut().cast(), range.len() as u64) } != 0;
        if !refused {
            self.registered.push(range.as_ptr().cast_mut().cast());
        }
        !refused
    }

    pub fn hybrid_counts(&self) -> Result<[u64; 8]> {
        let mut values = [0u64; 8];
        let [a, b, c, d, e, f, g, h] = &mut values;
        // SAFETY: four out-parameters each.
        check(unsafe { ffi::sd_hybrid_counts(self.handle, a, b, c, d) })?;
        check(unsafe { ffi::sd_hybrid_timing(self.handle, e, f, g, h) })?;
        Ok(values)
    }

    /// Telemetry: the device, its memory, GEMM and graph counters, partial-residency counters.
    pub fn info(&self) -> Result<serde_json::Value> {
        let (name, free, total) = straydog_gpu::device_info(self.ordinal).map_err(|error| error.to_string())?;
        let pci = straydog_gpu::pci_address(self.ordinal).map_err(|error| error.to_string())?;
        let (used, budget) = self.memory()?;
        let (mut gemms, mut vectors, mut graphs, mut replays, mut plans) = (0, 0, 0, 0, 0);
        // SAFETY: out-parameters.
        check(unsafe { ffi::sd_gemm_counts(self.handle, &mut gemms, &mut vectors) })?;
        check(unsafe { ffi::sd_execution_counts(self.handle, &mut graphs, &mut replays, &mut plans) })?;
        let hybrid = self.hybrid_counts()?;
        let mut info = serde_json::json!({
            "name": name, "pci": pci, "free_bytes": free, "total_bytes": total, "allocated_bytes": used, "budget_bytes": budget,
            "device": self.ordinal, "gemm_calls": gemms, "gemm_vectors": vectors, "cached_graphs": graphs,
            "graph_replays": replays, "resident_moe_plans": plans, "decoder_layers": self.decoder_layers.len(),
        });
        for (key, value) in HYBRID_COUNTERS.iter().zip(hybrid) {
            info[*key] = value.into();
        }
        Ok(info)
    }
}

pub const HYBRID_COUNTERS: [&str; 8] = ["gpu_experts", "cpu_experts", "streamed_experts", "cached_experts", "wait_ns",
                                        "cpu_ns", "swapped_experts", "assisted_experts"];

impl Drop for Backend {
    fn drop(&mut self) {
        // Finish background slot copies before the host banks they read go away.
        // SAFETY: a live handle, destroyed once; page locks are released after the drain.
        unsafe {
            ffi::sd_hybrid_drain(self.handle);
            for range in self.registered.drain(..) {
                ffi::sd_host_unregister(range);
            }
            ffi::sd_destroy(self.handle);
        }
    }
}

/// One routed expert: gate and up (640x2560) plus down (2560x640) of the given kinds.
pub fn expert_bytes(kinds: [u32; 3]) -> u64 {
    (formats::matrix_bytes(kinds[0], 640, 2560).unwrap_or(0) + formats::matrix_bytes(kinds[1], 640, 2560).unwrap_or(0)
        + formats::matrix_bytes(kinds[2], 2560, 640).unwrap_or(0)) as u64
}

/// Give the library the IQ grids once per process, before it opens any GPU or CPU engine.
pub fn load_iq_tables() -> Result<()> {
    static LOADED: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();
    LOADED
        .get_or_init(|| {
            let grids: Vec<&[u8]> = (0..6).map(formats::iq_grid).collect();
            // SAFETY: six grids of the sizes the library expects; it copies them.
            let status = unsafe {
                ffi::sd_iq_tables(grids[0].as_ptr(), grids[1].as_ptr(), grids[2].as_ptr(), grids[3].as_ptr(),
                                  grids[4].as_ptr(), grids[5].as_ptr())
            };
            if status == 0 { Ok(()) } else { Err(format!("Straydog IQ tables: {}", cpu_error())) }
        })
        .clone()
}

pub fn cpu_error() -> String {
    // SAFETY: a thread-local NUL-terminated string.
    unsafe { std::ffi::CStr::from_ptr(ffi::sd_cpu_error()) }.to_string_lossy().into_owned()
}

/// The native CPU expert engine (partially resident models), shared by every device.
pub struct CpuExperts {
    handle: *mut c_void,
}

unsafe impl Send for CpuExperts {}

impl CpuExperts {
    pub fn new(threads: u32) -> Result<CpuExperts> {
        load_iq_tables()?;
        // SAFETY: plain value; null on failure.
        let handle = unsafe { ffi::sd_cpu_experts_create(threads) };
        if handle.is_null() {
            return Err(format!("Straydog CPU experts: {}", cpu_error()));
        }
        Ok(CpuExperts { handle })
    }

    pub fn handle(&self) -> *mut c_void {
        self.handle
    }

    /// out = the sum over the experts of weight x down . (SiLU(gate . x) * (up . x)), each
    /// expert's three matrices (of `kinds`, `inter` x `hidden`, `hidden` x `inter`) in host memory.
    ///
    /// # Safety
    /// Every pointer must address its whole matrix, unchanged for the call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(&self, kinds: [u32; 3], gate: &[*const u8], up: &[*const u8], down: &[*const u8], weights: &[f32], hidden: usize,
                      inter: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        let count = weights.len();
        if gate.len() != count || up.len() != count || down.len() != count || x.len() != hidden || out.len() != hidden {
            return Err("CPU expert operands do not match".into());
        }
        let kinds = kinds.map(|kind| kind as i32);
        // SAFETY: the caller's matrices; the counts and lengths checked above.
        let status = unsafe {
            ffi::sd_cpu_experts_run(self.handle, kinds.as_ptr(), gate.as_ptr(), up.as_ptr(), down.as_ptr(), weights.as_ptr(), count as u32,
                                    hidden as u64, inter as u64, x.as_ptr(), out.as_mut_ptr())
        };
        if status != 0 {
            return Err(format!("Straydog CPU experts: {}", cpu_error()));
        }
        Ok(())
    }
}

impl Drop for CpuExperts {
    fn drop(&mut self) {
        // SAFETY: created by sd_cpu_experts_create, destroyed once after the devices using it.
        unsafe { ffi::sd_cpu_experts_destroy(self.handle) }
    }
}

/// The bytes of F32 values as they lie in memory (little-endian on every supported host).
fn bytemuck_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: an f32 slice reinterpreted as its bytes, same length in bytes and alignment 1.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}
