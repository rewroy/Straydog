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

/// How a layer-order pass reads the banks it does not hold registered (STRAYDOG_LAYER_BANK_DIRECT): through the page cache (`0`), around
/// it (`1`), or around it only where the memory available could not keep them for the next pass (`auto`, the default). Through it,
/// a host whose memory keeps the banks reads them from memory on the next pass; where it does not keep them (an RTX 5070 Ti PC with 64 GB,
/// UD-IQ2_XS at 9 GiB: 33 GB off the drive for a 32K read at some 3.4 GB/s), each pass reads the drive again, which direct reads with deep
/// queues may read faster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankReads { Cached, Direct, Auto }

fn bank_reads_named(value: Option<&str>) -> BankReads {
    match value.map(str::trim) {
        Some("1") => BankReads::Direct,
        Some("0") => BankReads::Cached,
        _ => BankReads::Auto,
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

/// Whether `leaves` bytes of tree leaf levels fit the read-ahead `room` beside a first copy of `copy` bytes and its margin: none to keep, nothing
/// to fit.
fn leaves_fit(leaves: u64, copy: u64, room: u64) -> bool {
    leaves > 0 && leaves.saturating_add(copy).saturating_add(FIRST_COPIES_MARGIN) <= room
}

/// The layers' reads a pass keeps ahead of the layer it loads at most: each holds one layer's checked copy (some 0.86 GB for UD-IQ2_XS).
/// One read ahead left the drive idle while the GPU computed a layer it held registered, and a layer's read began only once the layer
/// before it had gone up: an RTX 5070 Ti PC (64 GB, UD-IQ2_XS at 9 GiB, 19 of 48 layers registered) read a 32K prompt's 24.4 GB at 1.16 GB/s
/// on average in its 20.5 s, its drive (7 GB/s class) and CPU far from their limits, the GPU waiting for the next layer for 44% of a pass.
const READS_AHEAD: usize = 4;
/// The memory the first two copies keep beside them ([`Prefetch::new`]; more copies keep a GiB).
pub(crate) const FIRST_COPIES_MARGIN: u64 = 1 << 28;

pub(crate) struct Prefetch {
    plans: Vec<Option<Plan>>,
    reader: Arc<Reader>,
    buffers: usize,
    direct: bool,
    /// Buffers no read holds, each of [`Plan::bytes`] of the largest layer.
    free: Vec<Buffer>,
    /// The reads under way, in layer order.
    pending: std::collections::VecDeque<(usize, JoinHandle<Result<Copy, String>>)>,
    /// The next layer a read ahead may start at.
    next: usize,
    /// The model's layers registered in place, which are read from the mapping instead.
    registered: Arc<crate::banks::Registered>,
}

impl Prefetch {
    /// Up to [`READS_AHEAD`] layers' anonymous copies, each only while the memory available (and on Windows the commit left: `read_ahead_room`)
    /// keeps a GiB beside them all (the first two, a quarter of one). Under pressure,
    /// or without a pinned tree, retain the existing authenticated mapping path instead.
    pub(crate) fn new(weights: &Weights, layers: usize, registered: Arc<crate::banks::Registered>) -> Option<Self> {
        // (the bytes of the layers not registered now, which this pass reads from the file: whether to read them around the page cache)
        let unregistered: u64 = (0..layers).filter(|layer| !registered.is_registered(*layer))
            .filter_map(|layer| Plan::new(weights, layer, false)).map(|plan| plan.bytes as u64).sum();
        let direct = reads_direct(bank_reads(), unregistered, crate::host::prefetch_available_bytes());
        let plans: Vec<_> = (0..layers).map(|layer| Plan::new(weights, layer, direct)).collect();
        let bytes = plans.iter().flatten().map(|plan| plan.bytes).max()?;
        let room = crate::host::read_ahead_room();
        // The trees' leaf levels of the files the copies come from, kept whole once, where they fit the room beside a first copy: the copies'
        // checks then take no tree block from the trees' files and no lock between the readers, and the room left goes to the copies. Through
        // the bounded cache instead (4 GiB of a file's bytes), a 2,076-token pass on an RTX 5070 Ti PC that could lock no layer of UD-IQ2_XS took
        // some 80,000 tree blocks again, its readers' checks 162 s summed for 42 GB; with the leaf levels kept (565 MiB) 19 s, the drive the limit.
        let mut files: Vec<usize> = plans.iter().flatten().filter(|plan| !registered.is_registered(plan.layer))
            .flat_map(|plan| plan.roles.iter().map(|role| role.file_index)).collect();
        files.sort_unstable();
        files.dedup();
        // They are kept on a thread of their own, beside the pass's first reads, whose checks take the bounded cache until each level is
        // kept: kept first, they held the first bank's read for 0.32-0.66 s (565 MiB on Linux, read, zeroed and checked), and a PC's pass
        // began with a 0.85 s wait. Their room is taken now, whether or not they are kept in the end.
        let leaves = weights.tree_leaves_to_keep(&files);
        let keeping = leaves_fit(leaves, bytes as u64, room) && {
            let verifiers = weights.tree_verifiers_to_keep(&files);
            std::thread::Builder::new().name("tree-leaves".into()).spawn(move || {
                let mut kept = 0u64;
                for verifier in verifiers {
                    // (a level that does not verify stays unkept: every read is still checked through the bounded cache)
                    kept += verifier.keep_level0(leaves.saturating_sub(kept)).unwrap_or(0);
                }
            }).is_ok()
        };
        // (the thread holds the verifiers it keeps, not the pass: it may outlive a short pass, which does not wait for it)
        let available = room.saturating_sub(if keeping { leaves } else { 0 });
        let mut free = Vec::new();
        // (development: STRAYDOG_LAYER_BANK_BUFFERS=N reads ahead into N buffers at most, to see a pass the memory left one or two)
        let most = std::env::var("STRAYDOG_LAYER_BANK_BUFFERS").ok().and_then(|value| value.trim().parse::<usize>().ok())
            .map_or(READS_AHEAD, |most| most.clamp(1, READS_AHEAD));
        // (two buffers keep a quarter of a GiB beside them, the rest a GiB: with one, a layer's read begins only once the copy before it has
        // gone up, and a 90,000-token prompt on an RTX 5070 Ti PC took 39.1 s where it took 36.8 with two; a 95K prompt served after an 8K
        // and a 32K one got one in three of seven requests, 6 GiB free after them)
        let margin = |buffers: usize| if buffers < 2 { FIRST_COPIES_MARGIN } else { 1u64 << 30 };
        while free.len() < most && (free.len() as u64 + 1) * bytes as u64 + margin(free.len()) <= available {
            match Buffer::try_new(bytes) { Some(buffer) => free.push(buffer), None => break }
        }
        if free.is_empty() { return None; }
        let buffers = free.len();
        Some(Self { plans, reader: Reader::shared(), free, pending: std::collections::VecDeque::new(), next: 0, registered, buffers, direct })
    }

    /// Where the reads stand when `layer` found none of its own (for the request's metrics): the layer, the read at the front, the next layer
    /// to read, and the buffers free.
    pub(crate) fn missed(&self, layer: usize) -> String {
        format!("layer {layer}: reading {:?}, next {}, {} free", self.pending.iter().map(|(at, _)| *at).collect::<Vec<_>>(), self.next, self.free.len())
    }

    /// The buffers the pass reads ahead into (at most [`READS_AHEAD`], as the memory available allowed), and whether it reads around the page
    /// cache: for the request's metrics.
    pub(crate) fn shape(&self) -> (usize, bool) {
        (self.buffers, self.direct)
    }

    /// Reads ahead from `layer` on, as far as the free buffers go: every layer not registered in place now and with a plan, in order
    /// (a layer whose banks are registered in place is read from the mapping, whose pages its locks keep: not from the file again).
    pub(crate) fn start(&mut self, layer: usize) {
        // (a pass's start, `start(0)` with the pass before read through: its reads go from the first layer again)
        if layer < self.next && self.pending.back().is_none_or(|(at, _)| *at < layer) { self.next = layer; }
        self.next = self.next.max(layer);
        while !self.free.is_empty() && self.next < self.plans.len() {
            let at = self.next;
            self.next += 1;
            if self.registered.is_registered(at) || self.plans[at].is_none() { continue; }
            self.begin(at);
        }
    }

    /// Read `layer`'s banks now, registered or not: a layer whose registration was given up after [`Prefetch::start`] passed it by. Its checked
    /// copy, or None where no buffer is free or the layer has no plan (it is loaded as before).
    pub(crate) fn read_now(&mut self, layer: usize) -> Result<Option<Copy>, String> {
        if self.pending.front().is_some_and(|(at, _)| *at == layer) { return self.take(layer); }
        if self.free.is_empty() || self.plans.get(layer).cloned().flatten().is_none() { return Ok(None); }
        let Some(plan) = self.plans[layer].clone() else { return Ok(None) };
        let buffer = self.free.pop().expect("a free buffer");
        let copy = plan.read(&self.reader, buffer)?;
        Ok(Some(copy))
    }

    fn begin(&mut self, layer: usize) {
        let Some(plan) = self.plans.get(layer).cloned().flatten() else { return };
        let Some(buffer) = self.free.pop() else { return };
        let reader = self.reader.clone();
        // A scoped copy owns its destination until the reader has drained every
        // job; Drop below joins the task on cancellation or any model error.
        match std::thread::Builder::new().name("straydog-layer-bank".into()).spawn(move || plan.read(&reader, buffer)) {
            Ok(task) => self.pending.push_back((layer, task)),
            // (no thread: the buffer went with the closure; the layer is loaded as before)
            Err(_) => {}
        }
    }

    /// `layer`'s checked copy if a read ahead holds it, None if none does (registered when the reads went past it). Reads ahead of layers
    /// before it, which the pass no longer loads, are joined and their buffers freed.
    pub(crate) fn take(&mut self, layer: usize) -> Result<Option<Copy>, String> {
        while let Some((at, _)) = self.pending.front() {
            if *at > layer { return Ok(None); }
            let (at, task) = self.pending.pop_front().expect("a read under way");
            let copy = task.join().map_err(|_| "a layer bank reader panicked".to_string())??;
            if at == layer { return Ok(Some(copy)); }
            self.free.push(copy.buffer);
        }
        Ok(None)
    }

    pub(crate) fn recycle(&mut self, copy: Option<Copy>) {
        if let Some(copy) = copy { self.free.push(copy.buffer); }
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        while let Some((_, task)) = self.pending.pop_front() { let _ = task.join(); }
    }
}

#[cfg(test)]
mod bank_reads_tests {
    use super::{bank_reads_named, leaves_fit, reads_direct, BankReads, FIRST_COPIES_MARGIN};

    /// Decides: a pass keeps the trees' leaf levels only where they fit the read-ahead room beside a first copy and its margin, and keeps
    /// none where there are none left to keep.
    #[test]
    fn the_leaf_levels_are_kept_only_beside_a_first_copy() {
        let (mib, copy) = (1u64 << 20, 860u64 << 20);
        assert!(leaves_fit(565 * mib, copy, 565 * mib + copy + FIRST_COPIES_MARGIN));
        assert!(!leaves_fit(565 * mib, copy, 565 * mib + copy + FIRST_COPIES_MARGIN - 1));
        assert!(!leaves_fit(0, copy, u64::MAX));
        assert!(!leaves_fit(u64::MAX, copy, u64::MAX - 1));
    }

    /// Decides: STRAYDOG_LAYER_BANK_DIRECT=1 reads around the page cache, =0 through it, and anything else (unset included) around it
    /// only where the bytes read again each pass exceed the memory available to keep them.
    #[test]
    fn the_variable_chooses_how_a_pass_reads_its_banks() {
        assert_eq!(bank_reads_named(None), BankReads::Auto);
        assert_eq!(bank_reads_named(Some("0")), BankReads::Cached);
        assert_eq!(bank_reads_named(Some("yes")), BankReads::Auto);
        assert_eq!(bank_reads_named(Some(" 1 ")), BankReads::Direct);
        assert_eq!(bank_reads_named(Some("auto")), BankReads::Auto);
        let gib = 1u64 << 30;
        assert!(!reads_direct(BankReads::Cached, 40 * gib, gib));
        assert!(reads_direct(BankReads::Direct, gib, 40 * gib));
        assert!(reads_direct(BankReads::Auto, 25 * gib, 20 * gib));
        assert!(!reads_direct(BankReads::Auto, 25 * gib, 30 * gib));
    }
}
