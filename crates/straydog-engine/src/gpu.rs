//! One GPU: its pinned weights, sequence state and native execution plans.
//!
//! Every weight the engine uses is pinned for the life of the placement (full or partial
//! residency), so nothing is ever evicted; an upload that does not fit the budget is an
//! error. Operands carry their bytes, which are copied only on first use of their key.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_int, c_void};
use std::sync::Arc;

use indexmap::IndexMap;
use straydog_gpu::ffi;

use crate::formats;
use crate::weights::Vector;

pub type Result<T> = std::result::Result<T, String>;

/// The native library pads every VRAM allocation (allocation_tail in kernels.cpp).
pub const ALLOCATION_TAIL: u64 = 64;
const MOE_ROW: u64 = 11 * 4480 + 22;
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

/// An operand's bytes: mapped weights, a decoded vector, or (helper parts of expert
/// parallelism) only its description.
#[derive(Clone)]
pub enum Data<'a> {
    Bytes(&'a [u8]),
    Floats(Arc<Vector>),
    Described,
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

    /// A decoded vector, uploaded as F32 with its own shape.
    pub fn floats(key: impl Into<String>, vector: Arc<Vector>) -> Operand<'a> {
        let (rows, cols) = (vector.rows as u64, vector.cols as u64);
        Operand { key: key.into(), data: Data::Floats(vector), kind: 0, rows, cols }
    }
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

    pub fn configure_gemm(&self, minimum_batch: u64, half: bool) -> Result<()> {
        if !(2..=257).contains(&minimum_batch) {
            return Err("invalid Straydog GEMM configuration".into());
        }
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_gemm_options(self.handle, minimum_batch, c_int::from(half)) }).map(drop)
    }

    /// Upload an operand once; later uses of its key reuse the device copy.
    pub fn prepare(&mut self, operand: &Operand) -> Result<Entry> {
        if let Some(entry) = self.cache.get(&operand.key) {
            if (entry.rows, entry.cols) != (operand.rows, operand.cols) {
                return Err("cached Straydog matrix shape mismatch".into());
            }
            return Ok(*entry);
        }
        let data: &[u8] = match &operand.data {
            Data::Bytes(bytes) => bytes,
            Data::Floats(vector) => float_bytes(&vector.values),
            Data::Described => return Err("Straydog operand has no data".into()),
        };
        let size = formats::matrix_bytes(operand.kind, operand.rows as usize, operand.cols as usize)
            .ok_or("unsupported Straydog GPU weight format")?;
        if data.len() != size {
            return Err(format!("Straydog GPU weight size mismatch: {}", operand.key));
        }
        let (used, budget) = self.memory()?;
        if used + data.len() as u64 > budget {
            return Err(format!("{} cannot fit in the Straydog VRAM budget", operand.key));
        }
        // SAFETY: the bytes are valid for their length; the library copies them.
        let device = pointer(unsafe { ffi::sd_upload(self.handle, data.as_ptr().cast(), data.len() as u64) })?;
        let entry = Entry { pointer: device, kind: operand.kind as c_int, rows: operand.rows, cols: operand.cols };
        self.cache.insert(operand.key.clone(), entry);
        Ok(entry)
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
            let zeros = vec![0u8; (rows * cols * 4) as usize];
            self.prepare(&Operand::bytes(key, &zeros, 0, rows, cols))?
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
                plan.push((format!("@indexer:blk.{layer}."), (context / 4 + 1) * 128 + 512, 1, false));
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
    pub fn reserve_model_state(&mut self, context: u64, layers: &[usize], prefill: u64, host_attention: bool)
                               -> Result<()> {
        self.prefill_capacity = context.min(prefill);
        for (key, rows, cols, host) in self.state_plan(context, layers, prefill, host_attention)? {
            let entry = self.state(&key, rows, cols, host)?;
            if key == "@kv:stage" {
                // SAFETY: the staging allocation just made on this device.
                check(unsafe { ffi::sd_kv_stage(self.handle, entry.pointer, rows / 4) })?;
            }
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

    fn moe_workspace(&mut self) -> Result<(Entry, u64)> {
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
        let (workspace, capacity) = self.moe_workspace()?;
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

    /// Keep the first `kept` tokens of the last verification of this span.
    pub fn verify_commit(&mut self, begin: usize, end: usize, kept: u64) -> Result<()> {
        // SAFETY: a live handle.
        check(unsafe { ffi::sd_verify_commit(self.handle, begin as u64, end as u64, kept) }).map(drop)
    }

    /// The head's mixing and vocabulary rows: each row's first maximum, and its logits when asked.
    pub fn output_head(&mut self, operands: &[Operand], values: &[f32], logits: bool,
                       split: Option<(u64, *mut c_void)>) -> Result<(Vec<i32>, Option<Vec<f32>>)> {
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
        // SAFETY: four operands; `steps` rows in; tokens and optional logits sized above.
        check(unsafe {
            match split {
                None => ffi::sd_output_head(self.handle, pointers.as_ptr(), kinds.as_ptr(), steps as u64, vocabulary,
                                            values.as_ptr(), tokens.as_mut_ptr(), target),
                Some((split, peer)) => ffi::sd_output_head_split(self.handle, pointers.as_ptr(), kinds.as_ptr(),
                                                                 steps as u64, vocabulary, split, peer,
                                                                 values.as_ptr(), tokens.as_mut_ptr(), target),
            }
        })?;
        Ok((tokens, scores))
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
}

impl Drop for CpuExperts {
    fn drop(&mut self) {
        // SAFETY: created by sd_cpu_experts_create, destroyed once after the devices using it.
        unsafe { ffi::sd_cpu_experts_destroy(self.handle) }
    }
}
