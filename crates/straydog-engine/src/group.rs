//! Layer placement across the selected GPUs: one engine, one sequence state.
//!
//! Each device owns a contiguous range of decoder layers, executed in sequence. With two
//! GPUs holding the whole model, every layer's routed experts are split between them
//! instead (expert parallelism): both evaluate their half of each layer's routes at once.

use crate::gpu::{Backend, Operand, Result, WIDTH};
use crate::weights::layer_of;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitMode {
    Layer,
    Experts,
}

pub struct Group {
    pub backends: Vec<Backend>,
    /// First layer owned by each device.
    starts: Vec<usize>,
    baseline: Vec<u64>,
    split_mode: SplitMode,
    linked: bool,
}

impl Group {
    pub fn new(devices: &[usize], budget: u64, split_mode: SplitMode) -> Result<Group> {
        let mut distinct = devices.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if devices.is_empty() || devices.len() > 4 || distinct.len() != devices.len() {
            return Err("Straydog requires one to four distinct GPU ordinals".into());
        }
        let backends = devices.iter().map(|&device| Backend::new(device, budget)).collect::<Result<Vec<_>>>()?;
        let baseline = backends.iter().map(|backend| backend.memory().map(|(used, _)| used)).collect::<Result<_>>()?;
        // An even split until the weights are balanced.
        let starts = (0..devices.len()).map(|index| (index * 48).div_ceil(devices.len())).collect();
        let mut group = Group { backends, starts, baseline, split_mode, linked: false };
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
        let mut best: Option<u64> = None;
        let mut starts = self.starts.clone();
        // Every choice of count - 1 split points in lexicographic order; the first minimum wins.
        fn search(first: usize, chosen: &mut Vec<usize>, count: usize, prefix: &[u64], tail: u64,
                  best: &mut Option<u64>, starts: &mut Vec<usize>) {
            if chosen.len() == count - 1 {
                let bounds: Vec<usize> = std::iter::once(0).chain(chosen.iter().copied()).chain([48]).collect();
                let mut loads: Vec<u64> = (0..count).map(|i| prefix[bounds[i + 1]] - prefix[bounds[i]]).collect();
                *loads.last_mut().unwrap() += tail;
                let largest = *loads.iter().max().unwrap();
                if best.is_none_or(|best| largest < best) {
                    *best = Some(largest);
                    *starts = std::iter::once(0).chain(chosen.iter().copied()).collect();
                }
                return;
            }
            for split in first..48 - (count - 2 - chosen.len()) {
                chosen.push(split);
                search(split + 1, chosen, count, prefix, tail, best, starts);
                chosen.pop();
            }
        }
        search(1, &mut Vec::new(), count, &prefix, tail, &mut best, &mut starts);
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
            if parallel && layer < 48 && key.ends_with("_exps.weight") {
                for (index, group) in groups.iter_mut().enumerate() {
                    *group += size * Self::expert_half(index, 512).1 / 512;
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
    /// layer keeps its experts whole: it drafts one row at a time.
    pub fn register_resident_moe(&mut self, prefix: &str, operands: &[Operand], experts: u64) -> Result<()> {
        let owner = self.index(prefix);
        if !self.expert_parallel()? || layer_of(prefix) >= 48 {
            return self.backends[owner].register_resident_moe(prefix, operands, experts, 0, experts, true, false);
        }
        if !self.linked {
            return Err("Straydog expert parallelism needs linked GPUs (reserve_model_state)".into());
        }
        for index in 0..self.backends.len() {
            let (first, count) = Self::expert_half(index, experts);
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
            self.backends[index].register_resident_moe(prefix, &parts, experts, first, count, index == owner,
                                                       index != owner)?;
        }
        Ok(())
    }

    pub fn register_decoder_layer(&mut self, layer: usize, operands: &[Operand], context: u64) -> Result<()> {
        let owner = self.owner(layer);
        self.backends[owner].register_decoder_layer(layer, operands, context)
    }

    /// The remaining layers partitioned at the device boundaries.
    pub fn decoder_spans(&self, start: usize) -> Vec<(usize, usize)> {
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

    pub fn decode_span(&mut self, begin: usize, end: usize, values: &[f32], position: u64) -> Result<Vec<f32>> {
        let owner = self.owner_of_span(begin, end)?;
        self.backends[owner].decode_span(begin, end, values, position)
    }

    pub fn verify_span(&mut self, begin: usize, end: usize, values: &[f32], position: u64) -> Result<Vec<f32>> {
        let owner = self.owner_of_span(begin, end)?;
        self.backends[owner].verify_span(begin, end, values, position)
    }

    pub fn verify_commit(&mut self, begin: usize, end: usize, kept: u64) -> Result<()> {
        let owner = self.owner(begin);
        self.backends[owner].verify_commit(begin, end, kept)
    }

    pub fn prefill_layers(&mut self, begin: usize, end: usize, values: Option<&[f32]>, position: u64, steps: u64,
                          output_rows: u64) -> Result<Vec<f32>> {
        let owner = self.owner_of_span(begin, end)?;
        self.backends[owner].prefill_layers(begin, end, values, position, steps, output_rows)
    }

    pub fn prefill_embed(&mut self, layer: usize, embeddings: &[f32]) -> Result<()> {
        let owner = self.owner(layer);
        self.backends[owner].prefill_embed(layer, embeddings)
    }

    /// Whether a prompt's rows can wait on another device than the MTP layer's (the first
    /// layer's) while the MTP layer reads them: two devices, the MTP prologue's and the MTP
    /// layer's being the last.
    pub fn mtp_stash(&self) -> bool {
        let mtp = self.owner(48);
        mtp == self.backends.len() - 1 && self.owner(0) != mtp
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
        let mtp = self.owner(48);
        self.backends[mtp].mtp_keep(steps, previous, host)
    }

    pub fn mtp_embed(&mut self, embeddings: &[f32]) -> Result<()> {
        let mtp = self.owner(48);
        self.backends[mtp].mtp_embed(embeddings)
    }

    pub fn mtp_prompt_kept(&mut self, operands: &[Operand], row: u64, offset: u64, count: u64, position: u64)
                           -> Result<()> {
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

    pub fn prefill_ple(&mut self, prefix: &str, embeddings: &[f32], operands: &[Operand]) -> Result<()> {
        let owner = self.index(prefix);
        self.backends[owner].prefill_ple(prefix, embeddings, operands, 1)
    }

    pub fn ple(&mut self, prefix: &str, values: &[f32], embeddings: &[f32], operands: &[Operand]) -> Result<Vec<f32>> {
        let owner = self.index(prefix);
        self.backends[owner].ple(prefix, values, embeddings, operands)
    }

    pub fn ple_rows(&mut self, prefix: &str, values: &[f32], embeddings: &[f32], operands: &[Operand])
                    -> Result<Vec<f32>> {
        let owner = self.index(prefix);
        self.backends[owner].ple_rows(prefix, values, embeddings, operands)
    }

    pub fn copy_state(&mut self, target: &str, source: &str, owner: &str) -> Result<()> {
        let owner = self.index(owner);
        self.backends[owner].copy_state(target, source)
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
    pub fn output_head(&mut self, operands: &[Operand], values: &[f32], logits: bool)
                       -> Result<(Vec<i32>, Option<Vec<f32>>)> {
        let last = self.backends.len() - 1;
        if !(self.expert_parallel()? && self.linked) {
            return self.backends[last].output_head(operands, values, logits, None);
        }
        let head = &operands[3];
        let split = head.rows / 2;
        let crate::gpu::Data::Bytes(bytes) = &head.data else { return Err("invalid Straydog output head".into()) };
        let per_row = bytes.len() as u64 / head.rows;
        let lower = Operand::bytes(format!("{}:rows0", head.key), &bytes[..(split * per_row) as usize], head.kind,
                                   split, head.cols);
        self.backends[0].pin(std::slice::from_ref(&lower))?;
        let peer = self.backends[0].prepare(&lower)?.pointer;
        self.backends[last].output_head(operands, values, logits, Some((split, peer)))
    }

    pub fn mtp_prologue(&mut self, operands: &[Operand], embeddings: &[f32], hidden: &[f32]) -> Result<Vec<f32>> {
        let last = self.backends.len() - 1;
        self.backends[last].mtp_prologue(operands, embeddings, hidden)
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
        self.backends.iter_mut().map(|backend| backend.create_expert_slots(64, 192 << 20)).sum()
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
        (0..if drafter { 49 } else { 48 }).filter(|layer| self.owner(*layer) == index).collect()
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
            self.linked = true;
        }
        Ok(())
    }

    /// Take every device's host states once their page-locking (beside the weight upload) is done.
    pub fn finish_model_state(&mut self) -> Result<()> {
        self.backends.iter_mut().try_for_each(Backend::finish_model_state)
    }

    /// Per device, the VRAM a placement would still allocate.
    pub fn planned_state_bytes(&self, context: u64, prefill: u64, drafter: bool, host_attention: bool)
                               -> Result<Vec<u64>> {
        // Expert parallelism links the GPUs with inboxes of two prefill batches each (peer VRAM).
        let inbox = if self.expert_parallel()? { context.min(prefill) * (2560 * 2 + 21) * 4 + 256 } else { 0 };
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
