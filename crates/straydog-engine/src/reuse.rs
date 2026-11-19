//! What a store's eviction costs against the alternatives, measured on the uses it saw (a diagnostic): a window of the
//! decode and verify passes' expert uses, replayed through a least-recently-used store and through the best possible one
//! (Belady's: on a miss with every slot taken, the expert whose next use comes last goes), each class with as many slots
//! as the store gives it. A miss is a fill either way (an expert is computed from a slot), so the best possible store
//! bounds what any eviction could save. Only counts leave a window: its uses are dropped after the replay, and nothing
//! goes to disk.

use std::collections::{BTreeSet, BinaryHeap, HashMap};

/// Uses a window takes before its replay: at ~400 a two-row pass, a few minutes of decoding.
pub const WINDOW: usize = 1 << 20;

type Key = (u32, u32);

/// The misses of `uses` (one class's, in order) through a least-recently-used store of `slots`.
pub fn lru_misses(uses: &[Key], slots: usize) -> u64 {
    if slots == 0 {
        return uses.len() as u64;
    }
    // (an expert is held while it has a time here; the order by last use, oldest first)
    let mut last: HashMap<Key, u64> = HashMap::new();
    let mut order: BTreeSet<(u64, Key)> = BTreeSet::new();
    let mut misses = 0;
    for (time, &key) in uses.iter().enumerate() {
        let time = time as u64;
        match last.insert(key, time) {
            Some(before) => {
                order.remove(&(before, key));
            }
            None => {
                misses += 1;
                if order.len() == slots {
                    let (_, oldest) = order.pop_first().expect("a full store");
                    last.remove(&oldest);
                }
            }
        }
        order.insert((time, key));
    }
    misses
}

/// The misses of `uses` (one class's, in order) through the best possible store of `slots`: on a miss with every slot
/// taken, the held expert whose next use is furthest (or never) goes.
pub fn best_misses(uses: &[Key], slots: usize) -> u64 {
    if slots == 0 {
        return uses.len() as u64;
    }
    let mut next = vec![u64::MAX; uses.len()];
    let mut seen: HashMap<Key, usize> = HashMap::new();
    for (index, &key) in uses.iter().enumerate().rev() {
        if let Some(later) = seen.insert(key, index) {
            next[index] = later as u64;
        }
    }
    // (each held expert's next use; the heap's entries for an expert used again since are stale)
    let mut held: HashMap<Key, u64> = HashMap::new();
    let mut heap: BinaryHeap<(u64, Key)> = BinaryHeap::new();
    let mut misses = 0;
    for (index, &key) in uses.iter().enumerate() {
        if !held.contains_key(&key) {
            misses += 1;
            if held.len() == slots {
                while let Some((at, furthest)) = heap.pop() {
                    if held.get(&furthest) == Some(&at) {
                        held.remove(&furthest);
                        break;
                    }
                }
            }
        }
        held.insert(key, next[index]);
        heap.push((next[index], key));
    }
    misses
}

/// A store's window of uses, by class, and how many of them VRAM did not hold.
pub struct Reuse {
    uses: Vec<Vec<Key>>,
    missed: Vec<u64>,
    taken: usize,
}

impl Reuse {
    pub fn new(classes: usize) -> Reuse {
        Reuse { uses: vec![Vec::new(); classes], missed: vec![0; classes], taken: 0 }
    }

    /// A use of `key` (of `class`) by a pass of a few rows; `hit`: VRAM held it. With the window full, its replay
    /// starts on a thread of its own with the classes' slots and labels (`classes`), and the window starts over.
    pub fn note(&mut self, key: Key, class: usize, hit: bool, classes: impl FnOnce() -> Vec<(usize, String)>) {
        self.uses[class].push(key);
        self.missed[class] += u64::from(!hit);
        self.taken += 1;
        if self.taken < WINDOW {
            return;
        }
        let count = self.uses.len();
        let uses = std::mem::replace(&mut self.uses, vec![Vec::new(); count]);
        let missed = std::mem::replace(&mut self.missed, vec![0; count]);
        self.taken = 0;
        let classes = classes();
        let spawned = std::thread::Builder::new().name("straydog-reuse".into()).spawn(move || {
            eprintln!("{}", replay(&uses, &missed, &classes));
        });
        if let Err(error) = spawned {
            eprintln!("Straydog expert reuse: no replay ({error})");
        }
    }
}

/// Layers as runs ("layers 0-9", "layers 10-14, 20").
pub fn layer_label(layers: &[usize]) -> String {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &layer in layers {
        match runs.last_mut() {
            Some(run) if run.1 + 1 == layer => run.1 = layer,
            _ => runs.push((layer, layer)),
        }
    }
    let text: Vec<String> = runs.iter().map(|&(first, last)| if first == last { first.to_string() } else { format!("{first}-{last}") }).collect();
    format!("layers {}", text.join(", "))
}

/// The log line of a window: its uses, the share VRAM missed, and the shares an LRU store and the best possible one of
/// the same slots would have missed; then each class's.
pub fn replay(uses: &[Vec<Key>], missed: &[u64], classes: &[(usize, String)]) -> String {
    let percent = |part: u64, whole: u64| 100.0 * part as f64 / whole.max(1) as f64;
    let mut total = [0u64; 4];
    let mut parts = Vec::new();
    for (class, ((list, &actual), (slots, label))) in uses.iter().zip(missed).zip(classes).enumerate() {
        if list.is_empty() {
            continue;
        }
        let (lru, best) = (lru_misses(list, *slots), best_misses(list, *slots));
        let used = list.len() as u64;
        for (sum, value) in total.iter_mut().zip([used, actual, lru, best]) {
            *sum += value;
        }
        parts.push(format!("{class} ({label}, {slots} slots) {:.1} / {:.1} / {:.1}%", percent(actual, used), percent(lru, used),
                           percent(best, used)));
    }
    format!("Straydog expert reuse: {} uses of decode and verify passes on the first GPU; VRAM missed {:.1}%, an LRU store of the same \
             slots would miss {:.1}%, the best possible {:.1}%; by class (missed / LRU / best): {}",
            total[0], percent(total[1], total[0]), percent(total[2], total[0]), percent(total[3], total[0]), parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(experts: &[u32]) -> Vec<Key> {
        experts.iter().map(|&expert| (0, expert)).collect()
    }

    #[test]
    fn the_textbook_sequence_misses_as_lru_and_belady_do() {
        // (1 2 3 4 1 2 5 1 2 3 4 5 through three slots: LRU misses 10, the best possible 7)
        let uses = keys(&[1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]);
        assert_eq!(lru_misses(&uses, 3), 10);
        assert_eq!(best_misses(&uses, 3), 7);
        // (four slots: 8 and 6)
        assert_eq!(lru_misses(&uses, 4), 8);
        assert_eq!(best_misses(&uses, 4), 6);
    }

    #[test]
    fn a_store_as_large_as_its_experts_misses_each_once() {
        let uses = keys(&[3, 1, 3, 2, 1, 1, 2, 3]);
        assert_eq!(lru_misses(&uses, 3), 3);
        assert_eq!(best_misses(&uses, 3), 3);
        assert_eq!(lru_misses(&uses, 0), 8);
        assert_eq!(best_misses(&uses, 0), 8);
    }

    #[test]
    fn the_best_possible_store_never_misses_more_than_lru() {
        // (a skewed stream: a few experts often, many seldom)
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let uses: Vec<Key> = (0..20_000).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let draw = (state >> 33) as u32 % 1000;
            (0, if draw < 700 { draw % 40 } else { draw })
        }).collect();
        for slots in [1, 8, 40, 100, 400] {
            assert!(best_misses(&uses, slots) <= lru_misses(&uses, slots), "{slots} slots");
        }
    }

    #[test]
    fn layers_are_labelled_by_runs() {
        assert_eq!(layer_label(&[0, 1, 2, 3]), "layers 0-3");
        assert_eq!(layer_label(&[10, 11, 12, 20, 22, 23]), "layers 10-12, 20, 22-23");
    }

    #[test]
    fn a_window_replays_its_classes_into_one_line() {
        let uses = vec![keys(&[1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]), Vec::new()];
        let line = replay(&uses, &[9, 0], &[(3, "layers 0-9".into()), (5, "layers 10-19".into())]);
        assert!(line.contains("12 uses"), "{line}");
        assert!(line.contains("VRAM missed 75.0%, an LRU store of the same slots would miss 83.3%, the best possible 58.3%"), "{line}");
        assert!(line.contains("0 (layers 0-9, 3 slots) 75.0 / 83.3 / 58.3%"), "{line}");
        assert!(!line.contains("layers 10-19"), "{line}");
    }
}
