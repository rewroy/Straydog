//! One reusable authenticated host bank, read ahead while the preceding layer computes.
use std::fs::File;
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::experts::{Reader, VerifiedRead, Verifier};
use crate::storage::{self, Buffer, ALIGN};
use crate::weights::Weights;

#[derive(Clone)]
struct Role {
    file: Arc<File>,
    file_index: usize,
    offset: u64,
    length: usize,
    prefix: usize,
    bytes: usize,
    verifier: Verifier,
}

#[derive(Clone)]
struct Plan {
    layer: usize,
    roles: [Role; 3],
    bytes: usize,
}

impl Plan {
    fn new(weights: &Weights, layer: usize) -> Option<Self> {
        let files = weights.files_to_read();
        let roles = ["gate", "up", "down"].map(|role| -> Option<Role> {
            let tensor = weights.tensor(&format!("blk.{layer}.ffn_{role}_exps.weight")).ok()?;
            let verifier = weights.verifier_of(tensor.file)?.clone();
            let file = storage::reader(files.get(tensor.file)?.1).ok()?;
            let prefix = (tensor.offset % ALIGN as u64) as usize;
            let bytes = usize::try_from(tensor.bytes).ok()?;
            let length = prefix.checked_add(bytes)?.checked_add(ALIGN - 1)? / ALIGN * ALIGN;
            // A pinned tree must cover the source tensor. The final rounded block
            // is zero-padded by Reader, then verified like every other block.
            if tensor.offset.checked_add(tensor.bytes)? > files[tensor.file].1.metadata().ok()?.len() { return None; }
            let check: Verifier = Arc::new(move |_, at, data| verifier.verify(at, data));
            Some(Role { file: Arc::new(file), file_index: tensor.file, offset: tensor.offset - prefix as u64,
                        length, prefix, bytes, verifier: check })
        });
        let [Some(gate), Some(up), Some(down)] = roles else { return None };
        let bytes = gate.length.checked_add(up.length)?.checked_add(down.length)?;
        Some(Plan { layer, roles: [gate, up, down], bytes })
    }

    fn read(self, reader: &Reader, mut buffer: Buffer) -> Result<Copy, String> {
        let _trace = crate::layer_trace::Span::new("bank_read", Some(self.layer), 0, 0,
            self.roles.iter().map(|r| r.bytes as u64).sum());
        let mut tail = &mut buffer.as_mut_slice()[..self.bytes];
        let mut ranges = Vec::new();
        for role in &self.roles {
            let (target, remaining) = tail.split_at_mut(role.length);
            tail = remaining;
            ranges.push(VerifiedRead { file: role.file.clone(), file_index: role.file_index, offset: role.offset,
                target, verifier: role.verifier.clone() });
        }
        reader.read_verified(&mut ranges)?;
        drop(ranges);
        Ok(Copy { plan: self, buffer })
    }
}

/// No mutable reference to these bytes is exposed until the synchronous native
/// bank load has finished every CPU gather and H2D copy.
pub(crate) struct Copy {
    plan: Plan,
    buffer: Buffer,
}

impl Copy {
    pub(crate) fn sources(&self) -> [&[u8]; 3] {
        let mut base = 0;
        self.plan.roles.each_ref().map(|role| {
            let first = base + role.prefix;
            base += role.length;
            &self.buffer.as_slice()[first..first + role.bytes]
        })
    }
}

pub(crate) struct Prefetch {
    plans: Vec<Option<Plan>>,
    reader: Arc<Reader>,
    free: Option<Buffer>,
    pending: Option<(usize, JoinHandle<Result<Copy, String>>)>,
    /// The model's layers registered in place, which are read from the mapping instead.
    registered: Arc<crate::banks::Registered>,
}

impl Prefetch {
    /// At most one layer's anonymous copy. Under pressure, or without a pinned
    /// tree, retain the existing authenticated mapping path instead.
    pub(crate) fn new(weights: &Weights, layers: usize, registered: Arc<crate::banks::Registered>) -> Option<Self> {
        let plans: Vec<_> = (0..layers).map(|layer| Plan::new(weights, layer)).collect();
        let bytes = plans.iter().flatten().map(|plan| plan.bytes).max()?;
        if bytes as u64 + (1 << 30) > crate::host::prefetch_available_bytes() { return None; }
        let buffer = Buffer::try_new(bytes)?;
        Some(Self { plans, reader: Reader::shared(), free: Some(buffer), pending: None, registered })
    }

    pub(crate) fn start(&mut self, layer: usize) {
        // (a layer whose banks are registered in place is read from the mapping, whose pages its locks keep: not from the file again)
        if self.pending.is_some() || self.registered.is_registered(layer) { return; }
        let Some(plan) = self.plans.get(layer).cloned().flatten() else { return };
        let Some(buffer) = self.free.take() else { return };
        let reader = self.reader.clone();
        // A scoped copy owns its destination until the reader has drained every
        // job; Drop below joins the task on cancellation or any model error.
        if let Ok(task) = std::thread::Builder::new().name("straydog-layer-bank".into()).spawn(move || plan.read(&reader, buffer)) {
            self.pending = Some((layer, task));
        }
    }

    pub(crate) fn take(&mut self, layer: usize) -> Result<Option<Copy>, String> {
        let Some((wanted, task)) = self.pending.take() else { return Ok(None) };
        let copy = task.join().map_err(|_| "a layer bank reader panicked".to_string())??;
        if wanted != layer { return Err("a copied layer bank arrived out of order".into()); }
        Ok(Some(copy))
    }

    pub(crate) fn recycle(&mut self, copy: Option<Copy>) {
        if let Some(copy) = copy { self.free = Some(copy.buffer); }
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        if let Some((_, task)) = self.pending.take() { let _ = task.join(); }
    }
}
