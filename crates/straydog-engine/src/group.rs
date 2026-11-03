//! Layer placement across the selected GPUs: one engine, one sequence state.
//!
//! Each device owns a contiguous range of decoder layers, executed in sequence. With two
//! GPUs holding the whole model, every layer's routed experts are split between them
//! instead (expert parallelism): both evaluate their half of each layer's routes at once.
//! With their layers in parts (Group::enable_parts, gpu/qwen_parts.hpp), both run every layer but
//! the MTP layer, each its half of the heads and experts.

use crate::gpu::{Backend, Head, Operand, Result, StatePart, WIDTH};
use crate::weights::layer_of;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitMode {
    Layer,
    Experts,
}

/// Positional state copied out of every device from position `from` (see [`Group::save_positions`]).
pub struct Positions {
    pub from: u64,
    parts: Vec<Vec<StatePart>>,
}

impl Positions {
    /// A saved history's place after the K/V grew to `capacity` positions (see [`Backend::move_history`]).
    pub fn move_history(&mut self, capacity: u64) {
        self.parts.iter_mut().for_each(|parts| Backend::move_history(parts, capacity));
    }
}

pub struct Group {
    pub backends: Vec<Backend>,
    /// First layer owned by each device.
    starts: Vec<usize>,
    baseline: Vec<u64>,
    split_mode: SplitMode,
    linked: bool,
    /// Every layer but the MTP layer split across the two devices (see [`Group::enable_parts`]).
    parts: bool,
    /// The MTP layer split as well: a draft step on both devices.
    mtp_parts: bool,
}

/// Where a layer tensor goes when the layers are in parts: halves (rows or columns of its heads, of the shared expert's width), or
/// whole on both devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Share {
    Halves,
    Both,
}

impl Share {
    /// The share of a layer's tensor `name` (blk.L.suffix).
    pub fn of(name: &str) -> Share {
        const HALVES: [&str; 12] = ["attn_qkv.", "attn_gate.", "ssm_beta.", "ssm_alpha.", "ssm_out.", "ssm_conv1d.", "ssm_dt.", "ssm_a",
                                    "attn_q.", "attn_k.", "attn_v.", "attn_output."];
        let suffix = name.splitn(3, '.').nth(2).unwrap_or("");
        if HALVES.iter().any(|prefix| suffix.starts_with(prefix)) {
            Share::Halves
        } else if suffix.contains("_shexp.weight") && !suffix.starts_with("ffn_gate_inp") {
            // (the shared expert by width: gpu/qwen_parts.hpp, ResidentMoe::half)
            Share::Halves
        } else {
            Share::Both
        }
    }
}

impl Group {
    /// One backend per device, each with its own VRAM budget (`budgets`, in `devices` order).
    pub fn new(devices: &[usize], budgets: &[u64], split_mode: SplitMode) -> Result<Group> {
        let mut distinct = devices.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if devices.is_empty() || devices.len() > 4 || distinct.len() != devices.len() {
            return Err("Straydog requires one to four distinct GPU ordinals".into());
        }
        if budgets.len() != devices.len() {
            return Err("Straydog needs one VRAM budget per GPU".into());
        }
        let backends = devices.iter().zip(budgets).map(|(&device, &budget)| Backend::new(device, budget)).collect::<Result<Vec<_>>>()?;
        // A decode row's projections take INT8 activations (STRAYDOG_Q8=0 keeps them F32)
        if std::env::var("STRAYDOG_Q8").map_or(true, |value| value != "0") {
            for backend in &backends {
                backend.wide_q8_decode()?;
            }
        }
        let baseline = backends.iter().map(|backend| backend.memory().map(|(used, _)| used)).collect::<Result<_>>()?;
        // An even split until the weights are balanced.
        let starts = (0..devices.len()).map(|index| (index * 48).div_ceil(devices.len())).collect();
        let mut group = Group { backends, starts, baseline, split_mode, linked: false, parts: false, mtp_parts: false };
        group.assign_roles()?;
        Ok(group)
    }

    /// With two devices, what each plans for: help in a workspace of its own under expert
    /// parallelism (which needs the GEMM options), and the second's kept MTP rows.
    fn assign_roles(&mut self) -> Result<()> {
        if self.backends.len() == 2 {
            let parallel = self.expert_parallel()?;
            for backend in &mut self.backends {
                backend.expert_parallel = parallel;
            }
            self.backends[1].keeps_mtp_rows = true;
        }
        Ok(())
    }

    /// Whether a fully resident model splits every layer's routed experts across the two
    /// GPUs. It needs the grouped FP16 prefill path on both (RDNA4 or CUDA tensor cores).
    pub fn expert_parallel(&self) -> Result<bool> {
        if self.backends.len() != 2 || self.split_mode != SplitMode::Experts {
            return Ok(false);
        }
        for backend in &self.backends {
            if !backend.hybrid_supported()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Split every layer across the two devices (gpu/qwen_parts.hpp): both run each layer, each its half of the heads and
    /// routed experts. Under expert parallelism of two RDNA4 GPUs that write each other's VRAM; before anything is placed.
    /// `STRAYDOG_PARTS=0` keeps the layers whole, `STRAYDOG_MTP_PARTS=0` the MTP layer (on the second device).
    pub fn enable_parts(&mut self) -> Result<bool> {
        if !self.expert_parallel()? || std::env::var("STRAYDOG_PARTS").is_ok_and(|value| value == "0") {
            return Ok(false);
        }
        for part in 0..self.backends.len() {
            if self.backends[part].set_parts(2, part as u64).is_err() {
                // (a build without them: the layers stay whole)
                self.disable_parts()?;
                return Ok(false);
            }
        }
        self.parts = true;
        self.mtp_parts = std::env::var("STRAYDOG_MTP_PARTS").map_or(true, |value| value != "0");
        for backend in &mut self.backends {
            backend.mtp_parts = self.mtp_parts;
        }
        // (each device then keeps a deferred prompt's MTP rows: both run its pairs)
        self.backends[0].keeps_mtp_rows = self.mtp_parts;
        Ok(true)
    }

    /// The layers whole again (before anything is placed).
    pub fn disable_parts(&mut self) -> Result<()> {
        for backend in &mut self.backends {
            backend.set_parts(1, 0)?;
            backend.mtp_parts = false;
        }
        self.backends[0].keeps_mtp_rows = false;
        self.parts = false;
        self.mtp_parts = false;
        Ok(())
    }

    /// Whether the layers are in parts (see [`Group::enable_parts`]).
    pub fn parts(&self) -> bool {
        self.parts
    }

    /// Whether the MTP layer is in parts as well (see [`Group::enable_parts`]).
    pub fn mtp_parts(&self) -> bool {
        self.mtp_parts
    }

    /// Whether `key`'s tensor belongs to a layer in parts.
    fn parted(&self, key: &str) -> bool {
        self.parts && (layer_of(key) < 48 || (self.mtp_parts && key.starts_with("blk.48.")))
    }

    /// Routed experts [first, first + count) device `index` holds under expert parallelism.
    fn expert_half(index: usize, experts: u64) -> (u64, u64) {
        let half = experts / 2;
        if index == 0 { (0, half) } else { (half, experts - half) }
    }

    /// Split the 48 layers into contiguous ranges with the smallest largest weight share; the
    /// last device also holds `tail` (head, MTP layer). Decided before anything is placed.
    pub fn balance_layers(&mut self, layer_bytes: &[u64], tail: u64) -> Result<()> {
        let count = self.backends.len();
        if layer_bytes.len() != 48 {
            return Err("invalid Straydog layer sizes".into());
        }
        for (backend, baseline) in self.backends.iter().zip(&self.baseline) {
            if backend.memory()?.0 != *baseline {
                return Err("Straydog layers must be balanced before placement".into());
            }
        }
        let mut prefix = vec![0u64];
        for bytes in layer_bytes {
            prefix.push(prefix.last().unwrap() + bytes);
        }
        let budgets = self.backends.iter().map(|backend| backend.memory().map(|(_, budget)| budget.max(1))).collect::<Result<Vec<_>>>()?;
        let mut best: Option<u128> = None;
        let mut starts = self.starts.clone();
        // Every choice of count - 1 split points in lexicographic order; the first minimum of the
        // fullest device (its bytes over its budget) wins.
        #[allow(clippy::too_many_arguments)]
        fn search(first: usize, chosen: &mut Vec<usize>, count: usize, prefix: &[u64], tail: u64, budgets: &[u64],
                  best: &mut Option<u128>, starts: &mut Vec<usize>) {
            if chosen.len() == count - 1 {
                let bounds: Vec<usize> = std::iter::once(0).chain(chosen.iter().copied()).chain([48]).collect();
                let mut loads: Vec<u64> = (0..count).map(|i| prefix[bounds[i + 1]] - prefix[bounds[i]]).collect();
                *loads.last_mut().unwrap() += tail;
                let fullest = loads.iter().zip(budgets).map(|(&load, &budget)| ((load as u128) << 64) / budget as u128).max().unwrap();
                if best.is_none_or(|best| fullest < best) {
                    *best = Some(fullest);
                    *starts = std::iter::once(0).chain(chosen.iter().copied()).collect();
                }
                return;
            }
            for split in first..48 - (count - 2 - chosen.len()) {
                chosen.push(split);
                search(split + 1, chosen, count, prefix, tail, budgets, best, starts);
                chosen.pop();
            }
        }
        search(1, &mut Vec::new(), count, &prefix, tail, &budgets, &mut best, &mut starts);
        self.starts = starts;
        Ok(())
    }

    /// The second device's first layer, with two devices.
    pub fn split(&self) -> Option<usize> {
        (self.backends.len() == 2).then(|| self.starts[1])
    }

    /// Move the boundary between two devices' layers; only before anything is placed.
    pub fn set_split(&mut self, first: usize) -> Result<()> {
        if self.backends.len() != 2 || !(1..48).contains(&first) {
            return Err("invalid Straydog layer split".into());
        }
        for (backend, baseline) in self.backends.iter().zip(&self.baseline) {
            if backend.memory()?.0 != *baseline {
                return Err("Straydog layers must be split before placement".into());
            }
        }
        self.starts = vec![0, first];
        Ok(())
    }

    pub fn batch_capacity(&self) -> usize {
        self.backends.iter().map(|backend| backend.batch_capacity).min().unwrap()
    }

    pub fn output_capacity(&self) -> u64 {
        self.backends.iter().map(|backend| backend.output_capacity).min().unwrap()
    }

    pub fn logical_capacity(&self) -> usize {
        self.backends.iter().map(|backend| backend.logical_capacity).min().unwrap()
    }

    pub fn configure_gemm(&mut self, minimum_batch: u64, half: bool) -> Result<()> {
        self.backends.iter().try_for_each(|backend| backend.configure_gemm(minimum_batch, half))?;
        // (FP16 gives back half of the expanded buffer: nothing has been placed yet, and this is what that looks like now)
        self.baseline = self.backends.iter().map(|backend| backend.memory().map(|memory| memory.0)).collect::<Result<Vec<_>>>()?;
        self.assign_roles()
    }

    /// The device owning a layer's tensors (or a key naming one); the head, embeddings and
    /// the MTP layer go with the last.
    pub fn index(&self, name: &str) -> usize {
        let layer = layer_of(name);
        if layer >= 48 {
            return self.backends.len() - 1;
        }
        self.starts.iter().rposition(|start| layer >= *start).unwrap()
    }

    pub fn owner(&self, layer: usize) -> usize {
        self.index(&format!("blk.{layer}."))
    }

    fn owner_of_span(&self, begin: usize, end: usize) -> Result<usize> {
        let owner = self.owner(begin);
        if (begin..end).any(|layer| self.owner(layer) != owner) {
            return Err("Straydog decoder span crosses devices".into());
        }
        Ok(owner)
    }

    /// Whether these weights fit each device, beside `reserved` VRAM per device (a
    /// placement's planned state) or by default the decoder workspace.
    pub fn can_pin_model(&self, requirements: &[(String, u64)], reserved: Option<&[u64]>) -> Result<bool> {
        let mut groups = vec![0u64; self.backends.len()];
        let parallel = self.expert_parallel()?;
        for (key, size) in requirements {
            let layer = layer_of(key);
            if parallel && (layer < 48 || self.parted(key)) && key.ends_with("_exps.weight") {
                for (index, group) in groups.iter_mut().enumerate() {
                    *group += size * Self::expert_half(index, 512).1 / 512;
                }
                continue;
            }
            if self.parted(key) {
                match Share::of(key) {
                    Share::Halves => groups.iter_mut().for_each(|group| *group += size / 2),
                    Share::Both => groups.iter_mut().for_each(|group| *group += size),
                }
                continue;
            }
            groups[self.index(key)] += size;
            if parallel && key == "output.weight" {
                groups[0] += size / 2;   // the lower vocabulary half the first GPU evaluates
            }
        }
        for (index, (backend, required)) in self.backends.iter().zip(groups).enumerate() {
            if !backend.can_pin(required, reserved.map(|reserved| reserved[index]))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Pin each layer's expert banks on its owner, or half of them on each GPU. The MTP
    /// layer keeps its experts whole unless it is in parts.
    pub fn register_resident_moe(&mut self, prefix: &str, operands: &[Operand], experts: u64) -> Result<()> {
        let owner = self.index(prefix);
        if !self.expert_parallel()? || (layer_of(prefix) >= 48 && !self.parted(prefix)) {
            return self.backends[owner].register_resident_moe(prefix, operands, experts, 0, experts, true, false);
        }
        if !self.linked {
            return Err("Straydog expert parallelism needs linked GPUs (reserve_model_state)".into());
        }
        for index in 0..self.backends.len() {
            let (first, count) = Self::expert_half(index, experts);
            // (parts: each device half the shared expert's width, below; no helper)
            let (shared, helper) = if self.parts { (false, false) } else { (index == owner, index != owner) };
            let mut parts: Vec<Operand> = operands[..3]
                .iter()
                .map(|bank| {
                    let per_expert = match &bank.data {
                        crate::gpu::Data::Bytes(bytes) => bytes.len() as u64 / experts,
                        _ => 0,
                    };
                    let data = match &bank.data {
                        crate::gpu::Data::Bytes(bytes) => crate::gpu::Data::Bytes(
                            &bytes[(first * per_expert) as usize..((first + count) * per_expert) as usize]),
                        other => other.clone(),
                    };
                    Operand { key: format!("{}:{first}+{count}", bank.key), data, kind: bank.kind,
                              rows: bank.rows * count / experts, cols: bank.cols }
                })
                .collect();
            parts.extend(operands[3..].iter().cloned());
            self.backends[index].register_resident_moe(prefix, &parts, experts, first, count, shared, helper)?;
            if self.parts {
                self.backends[index].attach_shared_half(prefix, &Self::shared_half(&operands[3..6], index as u64)?)?;
            }
        }
        Ok(())
    }

    /// Part `part`'s half of a shared expert's width (gate, up, down): rows [320 part, +320) of gate and up, those columns of
    /// down.
    fn shared_half<'a>(operands: &[Operand<'a>], part: u64) -> Result<Vec<Operand<'a>>> {
        let mut half = Vec::with_capacity(3);
        for (role, operand) in operands.iter().enumerate() {
            // (the MTP layer's float matrices quantized as they upload, Model::q8_projection: halves of their own bytes)
            let (bytes, source) = match operand.data {
                crate::gpu::Data::Bytes(bytes) => (bytes, None),
                crate::gpu::Data::Quantize(bytes, kind) => (bytes, Some(kind)),
                _ => return Err("invalid Straydog shared expert".into()),
            };
            let kind = source.unwrap_or(operand.kind);
            let pitch = bytes.len() as u64 / operand.rows;
            let key = format!("{}:half{part}", operand.key);
            if role < 2 {
                if operand.rows != 640 {
                    return Err("invalid Straydog shared expert".into());
                }
                let rows = &bytes[(320 * part * pitch) as usize..(320 * (part + 1) * pitch) as usize];
                let data = source.map_or(crate::gpu::Data::Bytes(rows), |kind| crate::gpu::Data::Quantize(rows, kind));
                half.push(Operand { key, data, kind: operand.kind, rows: 320, cols: operand.cols });
            } else {
                let width = crate::formats::matrix_bytes(kind, 1, 320).ok_or("Straydog shared expert columns do not fill blocks")? as u64;
                if operand.cols != 640 || pitch != 2 * width {
                    return Err("invalid Straydog shared expert".into());
                }
                let columns = crate::gpu::Columns { pitch, first: part * width, width, to_pitch: width };
                let pieces = vec![crate::gpu::Piece { source: bytes, at: 0, columns: Some(columns) }];
                let data = match source {
                    None => crate::gpu::Data::Gather(pieces),
                    Some(kind) => crate::gpu::Data::QuantizeGather(pieces, kind),
                };
                half.push(Operand { key, data, kind: operand.kind, rows: operand.rows, cols: 320 });
            }
        }
        Ok(half)
    }

    pub fn register_decoder_layer(&mut self, layer: usize, operands: &[Operand], context: u64) -> Result<()> {
        let owner = self.owner(layer);
        self.backends[owner].register_decoder_layer(layer, operands, context)
    }

    /// A layer's part on device `part` (its operands: Model's parts), when the layers are in parts.
    pub fn register_decoder_part(&mut self, layer: usize, part: usize, operands: &[Operand], context: u64) -> Result<()> {
        self.backends[part].register_decoder_layer(layer, operands, context)
    }

    /// Pin a layer part's operands on its device ahead of [`Group::register_decoder_part`].
    pub fn pin_part(&mut self, part: usize, operands: &[Operand]) -> Result<()> {
        self.backends[part].pin(operands)
    }

    /// Pin a decoder layer's operands on its owner ahead of [`Group::register_decoder_layer`].
    pub fn pin_layer(&mut self, layer: usize, operands: &[Operand]) -> Result<()> {
        let owner = self.owner(layer);
        self.backends[owner].pin(operands)
    }

    /// The remaining layers partitioned at the device boundaries.
    pub fn decoder_spans(&self, start: usize) -> Vec<(usize, usize)> {
        if self.parts {
            return vec![(start, 48)];
        }
        let mut spans = Vec::new();
        let mut begin = start;
        for layer in start + 1..48 {
            if self.owner(layer) != self.owner(begin) {
                spans.push((begin, layer));
                begin = layer;
            }
        }
        spans.push((begin, 48));
        spans
    }

    /// Whether [begin, end) are layers in parts, run on both devices.
    fn in_parts(&self, begin: usize, end: usize) -> bool {
        self.parts && begin < end && (end <= 48 || (self.mtp_parts && (begin, end) == (48, 49)))
    }

    /// `first` on the first device and `second` on the second at once, from two threads: queueing a pass's span takes the host
    /// up to a millisecond a device (graph launch, K/V mirror copies), which the other device's first sum would otherwise
    /// wait (a pass at 20K took ~1 ms longer queued one device after the other).
    fn on_both<T: Send>(&mut self, first: impl FnOnce(&mut Backend) -> Result<()> + Send,
                        second: impl FnOnce(&mut Backend) -> Result<T> + Send) -> Result<T> {
        let (low, high) = self.backends.split_at_mut(1);
        std::thread::scope(|scope| {
            let queued = scope.spawn(|| first(&mut low[0]));
            let result = second(&mut high[0]);
            queued.join().map_err(|_| "a Straydog queueing thread panicked".to_string())??;
            result
        })
    }

    pub fn decode_span(&mut self, begin: usize, end: usize, values: &[f32], position: u64) -> Result<Vec<f32>> {
        if self.in_parts(begin, end) {
            // A verification of the one row, kept.
            let hidden = self.verify_span(begin, end, values, position)?;
            self.verify_commit(begin, end, 1)?;
            return Ok(hidden);
        }
        let owner = self.owner_of_span(begin, end)?;
        self.backends[owner].decode_span(begin, end, values, position)
    }

    pub fn verify_span(&mut self, begin: usize, end: usize, values: &[f32], position: u64) -> Result<Vec<f32>> {
        if self.in_parts(begin, end) {
            // (the first queued while the second waits: each one's sums wait for the other's parts)
            let hidden = self.on_both(|backend| backend.verify_queue(begin, end, values, position),
                                      |backend| backend.verify_span(begin, end, values, position))?;
            crate::gpu::parts_settle(&self.backends[0], &self.backends[1])?;
            return Ok(hidden);
        }
        let owner = self.owner_of_span(begin, end)?;
        self.backends[owner].verify_span(begin, end, values, position)
    }

    pub fn verify_reserve(&mut self) -> Result<()> {
        self.backends.iter_mut().try_for_each(Backend::verify_reserve)
    }

    pub fn verify_commit(&mut self, begin: usize, end: usize, kept: u64) -> Result<()> {
        if self.in_parts(begin, end) {
            return self.backends.iter_mut().try_for_each(|backend| backend.verify_commit(begin, end, kept));
        }
        let owner = self.owner(begin);
        self.backends[owner].verify_commit(begin, end, kept)
    }

    /// A sequence with images: its decode and verification rows' rope positions, on every device ([`Backend::rope_delta`]).
    pub fn rope_delta(&mut self, delta: u32) -> Result<()> {
        self.backends.iter_mut().try_for_each(|backend| backend.rope_delta(delta))
    }

    /// A sequence with images: the rope positions of the next prefill spans, on every device ([`Backend::rope_rows`]).
    pub fn rope_rows(&mut self, first: u64, axes: &[u32]) -> Result<()> {
        self.backends.iter_mut().try_for_each(|backend| backend.rope_rows(first, axes))
    }

    pub fn prefill_layers(&mut self, begin: usize, end: usize, values: Option<&[f32]>, position: u64, steps: u64,
                          output_rows: u64) -> Result<Vec<f32>> {
        if self.in_parts(begin, end) {
            if !(2..=4096).contains(&steps) {
                return Err("invalid Straydog prefill span of parts".into());
            }
            return crate::gpu::prefill_parts(&self.backends[0], &self.backends[1], begin, end, position, steps, values, output_rows);
        }
        let owner = self.owner_of_span(begin, end)?;
        self.backends[owner].prefill_layers(begin, end, values, position, steps, output_rows)
    }

    pub fn prefill_embed(&mut self, layer: usize, embeddings: &[f32]) -> Result<()> {
        if self.in_parts(layer, layer + 1) {
            return self.backends.iter_mut().try_for_each(|backend| backend.prefill_embed(layer, embeddings));
        }
        let owner = self.owner(layer);
        self.backends[owner].prefill_embed(layer, embeddings)
    }

    /// Whether a prompt's rows can wait on another device than the MTP layer's (the first
    /// layer's) while the MTP layer reads them: two devices, the MTP prologue's and the MTP
    /// layer's being the last.
    pub fn mtp_stash(&self) -> bool {
        let mtp = self.owner(48);
        !self.mtp_parts && mtp == self.backends.len() - 1 && self.owner(0) != mtp
    }

    /// One batch of prompt pairs for the MTP layer, the rows waiting in the first layer's
    /// workspace (see [`Group::mtp_stash`]).
    pub fn mtp_prompt_rows(&mut self, operands: &[Operand], first: usize, count: usize, previous: Option<&[f32]>,
                           embeddings: &[f32], position: u64) -> Result<()> {
        let (mtp, stash) = (self.owner(48), self.owner(0));
        if !self.mtp_stash() {
            return Err("Straydog has no device to keep MTP prompt rows on".into());
        }
        let (low, high) = self.backends.split_at_mut(mtp);
        high[0].mtp_prompt_rows(&low[stash], operands, 0, first, count, previous, embeddings, position, 48)
    }

    /// Move prompt rows from the device of layer `from` to the device of layer `to`, GPU to GPU;
    /// false when both layers are on one device (nothing to move).
    pub fn prefill_handoff(&mut self, from: usize, to: usize, steps: u64) -> Result<bool> {
        // (layers in parts leave their rows on both devices: those of the last layer are already where the first's are,
        // the MTP layer's stash)
        if self.parts && from < 48 && to < 48 {
            return Ok(from == 47 && to == 0);
        }
        let (source, target) = (self.owner(from), self.owner(to));
        if source == target {
            return Ok(false);
        }
        crate::gpu::prefill_handoff(&self.backends[source], &self.backends[target], from, to, steps)?;
        Ok(true)
    }

    /// Whether a deferred prompt's MTP pairs stay on the MTP layer's device (see [`Backend::mtp_keep`]).
    pub fn keeps_mtp_rows(&self) -> bool {
        let mtp = self.owner(48);
        self.backends[mtp].keeps_mtp_rows && self.owner(47) == mtp && self.owner(0) != mtp
    }

    pub fn mtp_keep(&mut self, steps: u64, previous: u64, host: Option<&[f32]>) -> Result<()> {
        if self.mtp_parts {
            return self.backends.iter_mut().try_for_each(|backend| backend.mtp_keep(steps, previous, host));
        }
        let mtp = self.owner(48);
        self.backends[mtp].mtp_keep(steps, previous, host)
    }

    /// Row `index` of the rows the MTP layer's device keeps (see [`Backend::mtp_keep`]).
    pub fn mtp_kept_row(&mut self, index: u64) -> Result<Vec<f32>> {
        let mtp = self.owner(48);
        self.backends[mtp].mtp_kept_row(index)
    }

    pub fn mtp_embed(&mut self, embeddings: &[f32]) -> Result<()> {
        if self.mtp_parts {
            return self.backends.iter_mut().try_for_each(|backend| backend.mtp_embed(embeddings));
        }
        let mtp = self.owner(48);
        self.backends[mtp].mtp_embed(embeddings)
    }

    pub fn mtp_prompt_kept(&mut self, operands: &[Operand], row: u64, offset: u64, count: u64, position: u64)
                           -> Result<()> {
        if self.mtp_parts {
            // (each device's prologue leaves the pairs in its part's workspace; the layer then runs on both)
            for backend in &mut self.backends {
                backend.mtp_prompt_kept(operands, row, offset, count, position)?;
            }
            if count > 1 {
                return self.prefill_layers(48, 49, None, position, count, 0).map(drop);
            }
            self.on_both(|backend| backend.verify_queue_rows(48, 49, None, 1, position),
                         |backend| backend.verify_queue_rows(48, 49, None, 1, position))?;
            return crate::gpu::parts_settle(&self.backends[0], &self.backends[1]);
        }
        let mtp = self.owner(48);
        self.backends[mtp].mtp_prompt_kept(operands, row, offset, count, position)
    }

    /// Deferred prefill on every device (see [`Backend::prefill_defer`]).
    pub fn prefill_defer(&mut self, on: bool) -> Result<()> {
        self.backends.iter_mut().try_for_each(|backend| backend.prefill_defer(on))
    }

    /// Settle every device, the first failure winning.
    pub fn prefill_settle(&mut self) -> Result<()> {
        let mut result = Ok(());
        for backend in &mut self.backends {
            let settled = backend.prefill_settle();
            if result.is_ok() {
                result = settled;
            }
        }
        result
    }

    /// The devices running the PLE layer `prefix`: its owner's, or with the layers in parts both (each has the rows).
    fn ple_devices(&self, prefix: &str) -> std::ops::Range<usize> {
        if self.parts { 0..self.backends.len() } else { let owner = self.index(prefix); owner..owner + 1 }
    }

    pub fn prefill_ple(&mut self, prefix: &str, embeddings: &[f32], operands: &[Operand]) -> Result<()> {
        for index in self.ple_devices(prefix) {
            self.backends[index].prefill_ple(prefix, embeddings, operands, 1)?;
        }
        Ok(())
    }

    pub fn ple(&mut self, prefix: &str, values: &[f32], embeddings: &[f32], operands: &[Operand]) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        for index in self.ple_devices(prefix) {
            output = self.backends[index].ple(prefix, values, embeddings, operands)?;
        }
        Ok(output)
    }

    pub fn ple_rows(&mut self, prefix: &str, values: &[f32], embeddings: &[f32], operands: &[Operand])
                    -> Result<Vec<f32>> {
        let mut output = Vec::new();
        for index in self.ple_devices(prefix) {
            output = self.backends[index].ple_rows(prefix, values, embeddings, operands)?;
        }
        Ok(output)
    }

    /// Whether a verification's first layer and the PLE layer after it can run in sequence on the device with no wait
    /// between them: the same device owns both (or, the layers in parts, both run both).
    pub fn front_chained(&self) -> bool {
        self.parts || self.owner(0) == self.index("blk.1.")
    }

    /// The first layer of a verification queued on its owner, nothing waited for (see [`Group::ple_front`]).
    pub fn verify_queue(&mut self, begin: usize, end: usize, values: &[f32], position: u64) -> Result<()> {
        if self.in_parts(begin, end) {
            return self.on_both(|backend| backend.verify_queue(begin, end, values, position),
                                |backend| backend.verify_queue(begin, end, values, position));
        }
        let owner = self.owner_of_span(begin, end)?;
        self.backends[owner].verify_queue(begin, end, values, position)
    }

    /// The PLE layer over the rows [`Group::verify_queue`] left on its device: the first layer's rows and the PLE layer's.
    pub fn ple_front(&mut self, prefix: &str, embeddings: &[f32], operands: &[Operand]) -> Result<(Vec<f32>, Vec<f32>)> {
        let devices = self.ple_devices(prefix);
        for index in devices.start..devices.end - 1 {
            self.backends[index].ple_front_queue(prefix, 0, embeddings, operands)?;
        }
        self.backends[devices.end - 1].ple_front(prefix, 0, embeddings, operands)
    }

    /// [`Group::ple_front`] queued, nothing read back: the next span continues from its device's workspace.
    pub fn ple_front_queue(&mut self, prefix: &str, embeddings: &[f32], operands: &[Operand]) -> Result<()> {
        for index in self.ple_devices(prefix) {
            self.backends[index].ple_front_queue(prefix, 0, embeddings, operands)?;
        }
        Ok(())
    }

    /// Whether a verification's spans and head can follow each other on the devices with no wait on the host: the
    /// first layer and the PLE layer on one device, the last layer on the head's.
    pub fn chains_verification(&self) -> bool {
        self.front_chained() && (self.parts || self.owner(47) == self.backends.len() - 1)
    }

    /// Span [begin, end) of a chained verification, queued over the rows the span before it (ending at `after`) left: moved
    /// GPU to GPU first when that span ran on another device.
    pub fn verify_chain(&mut self, after: usize, begin: usize, end: usize, steps: usize, position: u64) -> Result<()> {
        if self.in_parts(begin, end) {
            return self.on_both(|backend| backend.verify_queue_rows(begin, end, None, steps, position),
                                |backend| backend.verify_queue_rows(begin, end, None, steps, position));
        }
        self.prefill_handoff(after - 1, begin, steps as u64)?;
        let owner = self.owner_of_span(begin, end)?;
        self.backends[owner].verify_queue_rows(begin, end, None, steps, position)
    }

    /// The head over the rows a chained verification's last span left in layer `layer`'s workspace, those rows handed back
    /// with the head's in its one wait.
    #[allow(clippy::too_many_arguments)]
    pub fn output_head_chained(&mut self, operands: &[Operand], layer: usize, steps: usize, logits: bool, top: usize,
                               confidence: bool) -> Result<(Head, Vec<f32>)> {
        let last = self.backends.len() - 1;
        if self.owner(layer) != last && !self.parts {
            return Err("Straydog chained head is not on the last layer's device".into());
        }
        let lower = self.pin_head(&operands[3])?;
        let head = self.backends[last].output_head_rows(operands, crate::gpu::Rows::Layer(layer), steps, logits, top, confidence, lower)?;
        if self.parts {
            // (the head waited for both devices: a wait for a part that gave up voids the pass)
            crate::gpu::parts_settle(&self.backends[0], &self.backends[1])?;
        }
        Ok(head)
    }

    /// The PLE layer replayed over the first rows [`Group::ple_front`] kept, after a partial acceptance (queued).
    pub fn ple_replay(&mut self, prefix: &str, embeddings: &[f32], operands: &[Operand]) -> Result<()> {
        for index in self.ple_devices(prefix) {
            self.backends[index].ple_replay(prefix, embeddings, operands)?;
        }
        Ok(())
    }

    pub fn copy_state(&mut self, target: &str, source: &str, owner: &str) -> Result<()> {
        for index in self.ple_devices(owner) {
            self.backends[index].copy_state(target, source)?;
        }
        Ok(())
    }

    pub fn hyperconnection(&mut self, prefix: &str, values: &[f32], norm: Operand, matrices: &[Operand])
                           -> Result<Vec<f32>> {
        let owner = self.index(prefix);
        self.backends[owner].hyperconnection(prefix, values, norm, matrices)
    }

    /// Projections on the devices owning their weights, in the callers' order.
    pub fn project_many(&mut self, requests: &[(Operand, &[f32])]) -> Result<Vec<Vec<f32>>> {
        let mut results: Vec<Option<Vec<f32>>> = vec![None; requests.len()];
        for device in 0..self.backends.len() {
            let members: Vec<usize> = (0..requests.len()).filter(|i| self.index(&requests[*i].0.key) == device).collect();
            if members.is_empty() {
                continue;
            }
            let batch: Vec<(Operand, &[f32])> = members.iter().map(|i| requests[*i].clone()).collect();
            for (position, output) in members.into_iter().zip(self.backends[device].project_many(&batch)?) {
                results[position] = Some(output);
            }
        }
        Ok(results.into_iter().map(Option::unwrap).collect())
    }

    /// The vocabulary head and its mixing on the last GPU. In the two-GPU mode the first GPU
    /// evaluates the lower half of the vocabulary rows from its own copy at the same time.
    pub fn output_head(&mut self, operands: &[Operand], values: &[f32], logits: bool, top: usize, confidence: bool)
                       -> Result<Head> {
        let last = self.backends.len() - 1;
        let lower = self.pin_head(&operands[3])?;
        self.backends[last].output_head(operands, values, logits, top, confidence, lower)
    }

    /// Pin the vocabulary head on the last GPU and, in the two-GPU mode, its lower half of the
    /// rows on the first; that half's row count and copy.
    pub fn pin_head(&mut self, head: &Operand) -> Result<Option<(u64, *mut std::ffi::c_void)>> {
        let last = self.backends.len() - 1;
        self.backends[last].pin(std::slice::from_ref(head))?;
        if !(self.expert_parallel()? && self.linked) {
            return Ok(None);
        }
        let split = head.rows / 2;
        let crate::gpu::Data::Bytes(bytes) = &head.data else { return Err("invalid Straydog output head".into()) };
        let per_row = bytes.len() as u64 / head.rows;
        let lower = Operand::bytes(format!("{}:rows0", head.key), &bytes[..(split * per_row) as usize], head.kind,
                                   split, head.cols);
        self.backends[0].pin(std::slice::from_ref(&lower))?;
        Ok(Some((split, self.backends[0].prepare(&lower)?.pointer)))
    }

    /// The MTP layer's K/V and indexer gone with a dropped drafter (see [`Backend::drop_mtp_layer_state`]).
    pub fn drop_mtp_layer_state(&mut self) -> Result<()> {
        self.backends.iter_mut().try_for_each(Backend::drop_mtp_layer_state)
    }

    /// Each device's host K/V planned for `capacity` positions (see [`Backend::start_kv`]).
    pub fn start_kv(&mut self, capacity: u64) -> Result<()> {
        self.backends.iter_mut().try_for_each(|backend| backend.start_kv(capacity))
    }

    /// Each device's host K/V grown to `capacity` positions, `valid` kept (see [`Backend::grow_kv`]).
    /// One device's long prompt: the rows its batches may take (see [`Backend::begin_long_prefill`]), None with more devices than one.
    pub fn begin_long_prefill(&mut self) -> Result<Option<u64>> {
        match self.backends.as_mut_slice() {
            [backend] => backend.begin_long_prefill().map(Some),
            _ => Ok(None),
        }
    }

    pub fn ensure_slots(&mut self) -> Result<()> {
        self.backends.iter_mut().try_for_each(Backend::ensure_slots)
    }

    pub fn end_long_prefill(&mut self) -> Result<()> {
        self.backends.iter_mut().try_for_each(Backend::end_long_prefill)
    }

    pub fn grow_kv(&mut self, capacity: u64, valid: u64) -> Result<()> {
        self.backends.iter_mut().try_for_each(|backend| backend.grow_kv(capacity, valid))
    }

    /// Each device's K/V stage grown into the VRAM its placement left (see [`Backend::grow_kv_stage`]).
    pub fn grow_kv_stages(&mut self, context: u64) -> Result<()> {
        self.backends.iter_mut().try_for_each(|backend| backend.grow_kv_stage(context))
    }

    /// Allocate mapped weights instead of uploading them (see [`Backend::defer_uploads`]).
    pub fn defer_uploads(&mut self) {
        self.backends.iter_mut().for_each(Backend::defer_uploads);
    }

    /// Each device's uploads allocated since [`Group::defer_uploads`].
    pub fn take_pending(&mut self) -> Vec<Vec<crate::gpu::Pending>> {
        self.backends.iter_mut().map(Backend::take_pending).collect()
    }

    pub fn mtp_prologue(&mut self, operands: &[Operand], embeddings: &[f32], hidden: &[f32]) -> Result<Vec<f32>> {
        let last = self.backends.len() - 1;
        self.backends[last].mtp_prologue(operands, embeddings, hidden)
    }

    /// One drafting step on the MTP layer's device, or on both when it is in parts (see [`Backend::draft_step`],
    /// [`Backend::draft_queue`]): the prologue's four operands, then the head's four when a token is wanted. The head's lower
    /// half of the vocabulary runs on the first GPU, as in [`Group::output_head`].
    pub fn draft_step(&mut self, operands: &[Operand], position: u64, embeddings: &[f32], hidden: Option<&[f32]>, previous: u64, top: usize)
                      -> Result<Option<crate::gpu::Draft>> {
        let last = self.backends.len() - 1;
        let lower = if operands.len() == 8 { self.pin_head(&operands[7])? } else { None };
        if self.mtp_parts {
            // The layer's parts queued on both, then the head on the second's rows.
            let prologue = &operands[..4];
            self.on_both(|backend| backend.draft_queue(prologue, 48, position, embeddings, hidden, previous),
                         |backend| backend.draft_queue(prologue, 48, position, embeddings, hidden, previous))?;
            if operands.len() == 8 {
                let steps = hidden.map_or(1, |hidden| hidden.len() / crate::gpu::WIDTH) as u64;
                return self.backends[last].draft_head(&operands[4..], 48, steps - 1, lower, top).map(Some);
            }
            crate::gpu::parts_settle(&self.backends[0], &self.backends[1])?;
            return Ok(None);
        }
        self.backends[last].draft_step(operands, 48, position, embeddings, hidden, previous, lower, top)
    }

    /// A draft's vocabulary on the MTP layer's device when the head is split across the GPUs (see
    /// [`Backend::draft_vocabulary`]); false (nothing done) when it is not.
    pub fn draft_vocabulary(&mut self, head: &Operand, rows: u64, tokens: &[u32], reset: bool) -> Result<bool> {
        if !(self.expert_parallel()? && self.linked) {
            return Ok(false);
        }
        let last = self.backends.len() - 1;
        self.backends[last].draft_vocabulary(head, rows, tokens, reset)?;
        Ok(true)
    }

    /// Pin dense operands, each on the device owning its layer.
    pub fn pin(&mut self, operands: &[Operand]) -> Result<()> {
        for device in 0..self.backends.len() {
            let mine: Vec<Operand> =
                operands.iter().filter(|operand| self.index(&operand.key) == device).cloned().collect();
            if !mine.is_empty() {
                self.backends[device].pin(&mine)?;
            }
        }
        Ok(())
    }

    pub fn register_hybrid_moe(&mut self, prefix: &str, operands: &[Operand], banks: [&[u8]; 3], experts: u64,
                               kinds: [u32; 3]) -> Result<()> {
        let owner = self.index(prefix);
        self.backends[owner].register_hybrid_moe(prefix, operands, banks, experts, kinds)
    }

    pub fn create_expert_slots(&mut self) -> Result<u64> {
        self.backends.iter_mut().map(|backend| backend.create_expert_slots(crate::model::STAGING_SLOTS, crate::model::SLOTS_MARGIN)).sum()
    }

    pub fn hybrid_supported(&self) -> Result<bool> {
        for backend in &self.backends {
            if !backend.hybrid_supported()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn attach_cpu(&mut self, engine: &crate::gpu::CpuExperts) -> Result<()> {
        self.backends.iter_mut().try_for_each(|backend| backend.attach_cpu(engine))
    }

    pub fn rebalance(&mut self, max_swaps: u64) -> Result<()> {
        self.backends.iter_mut().try_for_each(|backend| backend.rebalance(max_swaps))
    }

    pub fn enable_assist(&mut self, routes: u32) -> Result<u64> {
        self.backends.iter_mut().map(|backend| backend.enable_assist(routes)).sum()
    }

    /// Page locks are process-wide; one backend performs them.
    pub fn register_host(&mut self, range: &[u8]) -> bool {
        self.backends[0].register_host(range)
    }

    fn owned_layers(&self, index: usize, drafter: bool) -> Vec<usize> {
        (0..if drafter { 49 } else { 48 }).filter(|layer| self.owner(*layer) == index || self.parted(&format!("blk.{layer}."))).collect()
    }

    /// Reserve each layer's state on its owner (and the MTP layer's) before accepting
    /// requests; under expert parallelism, link the GPUs with inboxes for a prefill batch.
    pub fn reserve_model_state(&mut self, context: u64, prefill: u64, drafter: bool, host_attention: bool) -> Result<()> {
        for index in 0..self.backends.len() {
            let layers = self.owned_layers(index, drafter);
            self.backends[index].reserve_model_state(context, &layers, prefill, host_attention)?;
        }
        if self.expert_parallel()? && !self.linked {
            let (first, second) = self.backends.split_at_mut(1);
            first[0].link_peer(&mut second[0], context.min(prefill))?;
            if self.parts {
                // The parts' sums: a prefill batch's rows.
                Backend::link_sums(&mut first[0], &mut second[0], context.min(prefill) * WIDTH as u64 / 4)?;
            }
            self.linked = true;
        }
        Ok(())
    }

    /// Start page-locking every device's planned host states (see [`Backend::prefetch_host_states`]).
    pub fn prefetch_host_states(&self) -> Result<()> {
        self.backends.iter().try_for_each(Backend::prefetch_host_states)
    }

    /// Take every device's host states once their page-locking (beside reading the weights) is done.
    pub fn finish_model_state(&mut self) -> Result<()> {
        self.backends.iter_mut().try_for_each(Backend::finish_model_state)
    }

    /// Per device, the VRAM a placement would still allocate.
    pub fn planned_state_bytes(&self, context: u64, prefill: u64, drafter: bool, host_attention: bool)
                               -> Result<Vec<u64>> {
        // Expert parallelism links the GPUs with inboxes of two prefill batches each (peer VRAM); the parts' sums with two
        // slots of a batch's rows more.
        let mut inbox = if self.expert_parallel()? { context.min(prefill) * (2560 * 2 + 21) * 4 + 256 } else { 0 };
        if self.parts {
            inbox += context.min(prefill) * 2560 * 2 * 4 + 256;
        }
        (0..self.backends.len())
            .map(|index| {
                let layers = self.owned_layers(index, drafter);
                Ok(self.backends[index].planned_state_bytes(context, &layers, prefill, host_attention)? + inbox)
            })
            .collect()
    }

    pub fn host_state_supported(&self) -> Result<bool> {
        for backend in &self.backends {
            if !backend.host_state_supported()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn reset_state(&mut self) -> Result<()> {
        self.backends.iter_mut().try_for_each(Backend::reset_state)
    }

    pub fn reserve_checkpoints(&mut self, count: usize) -> Result<u64> {
        self.backends.iter_mut().map(|backend| backend.reserve_checkpoints(count)).sum()
    }

    pub fn checkpoint(&mut self, slot: usize, save: bool) -> Result<()> {
        self.backends.iter_mut().try_for_each(|backend| backend.checkpoint(slot, save))
    }

    /// Host bytes [`save_positions`](Self::save_positions) takes per position.
    pub fn position_bytes(&self) -> u64 {
        self.backends.iter().map(Backend::position_bytes).sum()
    }

    /// Every device's positional state over positions `from..to` (see [`Backend::save_positions`]).
    pub fn save_positions(&mut self, from: u64, to: u64, history: bool) -> Result<Positions> {
        let mut parts = Vec::with_capacity(self.backends.len());
        for index in 0..self.backends.len() {
            match self.backends[index].save_positions(from, to, history) {
                Ok(part) => parts.push(part),
                Err(error) => {
                    // The devices before it free what they kept in VRAM.
                    for (backend, saved) in self.backends.iter_mut().zip(parts) {
                        let _ = backend.release_positions(saved);
                    }
                    return Err(error);
                }
            }
        }
        Ok(Positions { from, parts })
    }

    /// Every device's indexer history alone (see [`Backend::save_history`]; one `layer`'s with
    /// it), as positions from `from` (the parked sequence's lowest).
    pub fn save_history(&mut self, from: u64, layer: Option<&str>) -> Result<Positions> {
        let mut parts = Vec::with_capacity(self.backends.len());
        for index in 0..self.backends.len() {
            match self.backends[index].save_history(layer) {
                Ok(part) => parts.push(part),
                Err(error) => {
                    for (backend, saved) in self.backends.iter_mut().zip(parts) {
                        let _ = backend.release_positions(saved);
                    }
                    return Err(error);
                }
            }
        }
        Ok(Positions { from, parts })
    }

    /// Diagnostics: a digest of every device's sequence state valid below `to` (see
    /// [`Backend::state_digest`]).
    pub fn state_digest(&mut self, to: u64) -> Result<Vec<String>> {
        let mut digests = Vec::new();
        for (index, backend) in self.backends.iter_mut().enumerate() {
            digests.extend(backend.state_digest(to)?.into_iter().map(|digest| format!("GPU {index} {digest}")));
        }
        Ok(digests)
    }

    pub fn restore_positions(&mut self, positions: &mut Positions) -> Result<()> {
        self.backends.iter_mut().zip(&mut positions.parts).try_for_each(|(backend, parts)| backend.restore_positions(parts))
    }

    /// Free what [`save_positions`](Self::save_positions) kept in VRAM, once nothing restores it.
    pub fn release_positions(&mut self, positions: Positions) -> Result<()> {
        // Every device's, a failed one too (the first error returned).
        let mut result = Ok(());
        for (backend, parts) in self.backends.iter_mut().zip(positions.parts) {
            let released = backend.release_positions(parts);
            if result.is_ok() {
                result = released;
            }
        }
        result
    }

    /// Allocated bytes and budgets summed over the devices.
    pub fn memory(&self) -> Result<(u64, u64)> {
        let mut total = (0, 0);
        for backend in &self.backends {
            let (used, budget) = backend.memory()?;
            total = (total.0 + used, total.1 + budget);
        }
        Ok(total)
    }

    /// Aggregate telemetry with each device's record.
    pub fn info(&self) -> Result<serde_json::Value> {
        let devices = self.backends.iter().map(Backend::info).collect::<Result<Vec<_>>>()?;
        let names: Vec<&str> = devices.iter().map(|device| device["name"].as_str().unwrap_or("")).collect();
        let mut info = serde_json::json!({"name": names.join(" + "), "device": devices[0]["device"]});
        for key in ["free_bytes", "total_bytes", "allocated_bytes", "budget_bytes", "gemm_calls", "gemm_vectors"]
            .into_iter()
            .chain(crate::gpu::HYBRID_COUNTERS)
        {
            info[key] = devices.iter().map(|device| device[key].as_u64().unwrap_or(0)).sum::<u64>().into();
        }
        info["devices"] = devices.into();
        Ok(info)
    }
}

/// Rows of 10240 in `values`.
pub fn row_count(values: &[f32]) -> usize {
    values.len() / WIDTH
}
