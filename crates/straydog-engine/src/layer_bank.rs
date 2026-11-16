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
    fn new(weights: &Weights, layer: usize, direct: bool) -> Option<Self> {
        let files = weights.files_to_read();
        let roles = ["gate", "up", "down"].map(|role| -> Option<Role> {
            let tensor = weights.tensor(&format!("blk.{layer}.ffn_{role}_exps.weight")).ok()?;
            let verifier = weights.verifier_of(tensor.file)?.clone();
            // (`direct`: around the page cache, as the experts' own reads are, with their fallback through it where the working-set quota
            // refuses a direct read; else through it: [`BankReads`])
            let opened = files.get(tensor.file)?.1;
            let file = match direct.then(|| storage::direct(opened)).flatten() {
                Some(direct) => direct,
                None => storage::reader(opened).ok()?,
            };
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

/// How a layer-order pass reads the banks it does not hold registered (STRAYDOG_LAYER_BANK_DIRECT): through the page cache (`0`, the
/// default), around it (`1`), or around it only where the memory available could not keep them for the next pass (`auto`). Through it,
/// a host whose memory keeps the banks reads them from memory on the next pass; where it does not keep them (an RTX 5070 Ti PC with 64 GB,
/// UD-IQ2_XS at 9 GiB: 33 GB off the drive for a 32K read at some 3.4 GB/s), each pass reads the drive again, which direct reads with deep
/// queues may read faster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankReads { Cached, Direct, Auto }

fn bank_reads_named(value: Option<&str>) -> BankReads {
    match value.map(str::trim) {
        Some("1") => BankReads::Direct,
        Some("auto") => BankReads::Auto,
        _ => BankReads::Cached,
    }
}

fn bank_reads() -> BankReads {
    static READS: std::sync::OnceLock<BankReads> = std::sync::OnceLock::new();
    *READS.get_or_init(|| bank_reads_named(std::env::var("STRAYDOG_LAYER_BANK_DIRECT").ok().as_deref()))
}

/// Whether a pass reads around the page cache: `reads`, and for `Auto` whether the `unregistered` bytes it reads again each pass exceed
/// the memory `available` to keep them.
fn reads_direct(reads: BankReads, unregistered: u64, available: u64) -> bool {
    match reads {
        BankReads::Cached => false,
        BankReads::Direct => true,
        BankReads::Auto => unregistered > available,
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
        // (the bytes of the layers not registered now, which this pass reads from the file: whether to read them around the page cache)
        let unregistered: u64 = (0..layers).filter(|layer| !registered.is_registered(*layer))
            .filter_map(|layer| Plan::new(weights, layer, false)).map(|plan| plan.bytes as u64).sum();
        let direct = reads_direct(bank_reads(), unregistered, crate::host::prefetch_available_bytes());
        let plans: Vec<_> = (0..layers).map(|layer| Plan::new(weights, layer, direct)).collect();
        let bytes = plans.iter().flatten().map(|plan| plan.bytes).max()?;
        if bytes as u64 + (1 << 30) > crate::host::prefetch_available_bytes() { return None; }
        let buffer = Buffer::try_new(bytes)?;
        Some(Self { plans, reader: Reader::shared(), free: Some(buffer), pending: None, registered })
    }

    pub(crate) fn start(&mut self, layer: usize) {
        // (a layer whose banks are registered in place is read from the mapping, whose pages its locks keep: not from the file again)
        if self.registered.is_registered(layer) { return; }
        self.begin(layer);
    }

    /// Read `layer`'s banks now, registered or not: a layer whose registration was given up after [`Prefetch::start`] passed it by. Its checked
    /// copy, or None where the layer has no plan (it is loaded as before).
    pub(crate) fn read_now(&mut self, layer: usize) -> Result<Option<Copy>, String> {
        self.begin(layer);
        self.take(layer)
    }

    fn begin(&mut self, layer: usize) {
        if self.pending.is_some() { return; }
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

#[cfg(test)]
mod bank_reads_tests {
    use super::{bank_reads_named, reads_direct, BankReads};

    /// Decides: STRAYDOG_LAYER_BANK_DIRECT=1 reads around the page cache, =auto only where the bytes read again each pass exceed the
    /// memory available to keep them, and anything else (unset included) through it, as before.
    #[test]
    fn the_variable_chooses_how_a_pass_reads_its_banks() {
        assert_eq!(bank_reads_named(None), BankReads::Cached);
        assert_eq!(bank_reads_named(Some("0")), BankReads::Cached);
        assert_eq!(bank_reads_named(Some("yes")), BankReads::Cached);
        assert_eq!(bank_reads_named(Some(" 1 ")), BankReads::Direct);
        assert_eq!(bank_reads_named(Some("auto")), BankReads::Auto);
        let gib = 1u64 << 30;
        assert!(!reads_direct(BankReads::Cached, 40 * gib, gib));
        assert!(reads_direct(BankReads::Direct, gib, 40 * gib));
        assert!(reads_direct(BankReads::Auto, 25 * gib, 20 * gib));
        assert!(!reads_direct(BankReads::Auto, 25 * gib, 30 * gib));
    }
}
