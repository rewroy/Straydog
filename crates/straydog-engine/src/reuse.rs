//! What a store's eviction costs against the alternatives, measured on the uses it saw (a diagnostic): a window of the
//! decode and verify passes' expert uses, replayed through a least-recently-used store and through the best possible one
//! (Belady's: on a miss with every slot taken, the expert whose next use comes last goes), each class with as many slots
//! as the store gives it. A miss is a fill either way (an expert is computed from a slot), so the best possible store
//! bounds what any eviction could save. Two stores that could be built are replayed beside them, to tell which would
//! take what: ARC (Megiddo and Modha's: recency and frequency, split by the misses its ghosts see) and LRU-2 (the
//! expert whose last use but one is oldest goes). Only counts leave a window: its uses are dropped after the replay, and
//! nothing goes to disk.

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

/// Experts in order of a time each (oldest first), and each one's time.
#[derive(Default)]
struct Ordered {
    order: BTreeSet<(u64, Key)>,
    at: HashMap<Key, u64>,
}

impl Ordered {
    fn len(&self) -> usize {
        self.at.len()
    }
    fn contains(&self, key: &Key) -> bool {
        self.at.contains_key(key)
    }
    fn push(&mut self, key: Key, time: u64) {
        if let Some(before) = self.at.insert(key, time) {
            self.order.remove(&(before, key));
        }
        self.order.insert((time, key));
    }
    fn remove(&mut self, key: &Key) -> bool {
        match self.at.remove(key) {
            Some(time) => self.order.remove(&(time, *key)),
            None => false,
        }
    }
    fn pop_oldest(&mut self) -> Option<Key> {
        let (_, key) = self.order.pop_first()?;
        self.at.remove(&key);
        Some(key)
    }
}

/// The misses of `uses` through an ARC store of `slots` (T1/T2 the held experts seen once and more than once lately, B1/B2
/// the ghosts of those evicted from each, `target` T1's share, moved toward the list whose ghost was missed).
pub fn arc_misses(uses: &[Key], slots: usize) -> u64 {
    if slots == 0 {
        return uses.len() as u64;
    }
    let (mut t1, mut t2, mut b1, mut b2) = (Ordered::default(), Ordered::default(), Ordered::default(), Ordered::default());
    let mut target = 0f64;
    let mut misses = 0;
    // (an eviction from T1 into B1 or from T2 into B2, the side by `target`)
    let replace = |t1: &mut Ordered, t2: &mut Ordered, b1: &mut Ordered, b2: &mut Ordered, in_b2: bool, target: f64, time: u64| {
        if t1.len() > 0 && ((in_b2 && t1.len() as f64 == target.floor()) || t1.len() as f64 > target) {
            if let Some(old) = t1.pop_oldest() {
                b1.push(old, time);
            }
        } else if let Some(old) = t2.pop_oldest() {
            b2.push(old, time);
        } else if let Some(old) = t1.pop_oldest() {
            b1.push(old, time);
        }
    };
    for (time, &key) in uses.iter().enumerate() {
        let time = time as u64;
        if t1.remove(&key) || t2.contains(&key) {
            t2.push(key, time);
            continue;
        }
        misses += 1;
        if b1.contains(&key) {
            target = (target + (b2.len() as f64 / b1.len() as f64).max(1.0)).min(slots as f64);
            replace(&mut t1, &mut t2, &mut b1, &mut b2, false, target, time);
            b1.remove(&key);
            t2.push(key, time);
            continue;
        }
        if b2.contains(&key) {
            target = (target - (b1.len() as f64 / b2.len() as f64).max(1.0)).max(0.0);
            replace(&mut t1, &mut t2, &mut b1, &mut b2, true, target, time);
            b2.remove(&key);
            t2.push(key, time);
            continue;
        }
        if t1.len() + b1.len() == slots {
            if t1.len() < slots {
                b1.pop_oldest();
                replace(&mut t1, &mut t2, &mut b1, &mut b2, false, target, time);
            } else {
                t1.pop_oldest();
            }
        } else if t1.len() + t2.len() + b1.len() + b2.len() >= slots {
            if t1.len() + t2.len() + b1.len() + b2.len() == 2 * slots {
                b2.pop_oldest();
            }
            if t1.len() + t2.len() >= slots {
                replace(&mut t1, &mut t2, &mut b1, &mut b2, false, target, time);
            }
        }
        t1.push(key, time);
    }
    misses
}

/// The misses of `uses` through an LRU-2 store of `slots`: on a miss with every slot taken, the held expert whose last use
/// but one is oldest goes (one used once lately first, the least recently of those). An expert's uses are remembered
/// after it goes, so one taken again counts its earlier use.
pub fn lru2_misses(uses: &[Key], slots: usize) -> u64 {
    if slots == 0 {
        return uses.len() as u64;
    }
    // (each expert's last two uses, 0 for none: times count from 1)
    let mut seen: HashMap<Key, (u64, u64)> = HashMap::new();
    let mut held: BTreeSet<(u64, u64, Key)> = BTreeSet::new();
    let mut misses = 0;
    for (index, &key) in uses.iter().enumerate() {
        let time = index as u64 + 1;
        let (before, last) = seen.get(&key).copied().unwrap_or((0, 0));
        if !held.remove(&(before, last, key)) {
            misses += 1;
            if held.len() == slots {
                held.pop_first();
            }
        }
        seen.insert(key, (last, time));
        held.insert((last, time, key));
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
    let mut total = [0u64; 6];
    let mut parts = Vec::new();
    for (class, ((list, &actual), (slots, label))) in uses.iter().zip(missed).zip(classes).enumerate() {
        if list.is_empty() {
            continue;
        }
        let (lru, best) = (lru_misses(list, *slots), best_misses(list, *slots));
        let (arc, lru2) = (arc_misses(list, *slots), lru2_misses(list, *slots));
        let used = list.len() as u64;
        for (sum, value) in total.iter_mut().zip([used, actual, lru, best, arc, lru2]) {
            *sum += value;
        }
        parts.push(format!("{class} ({label}, {slots} slots) {:.1} / {:.1} / {:.1} / {:.1} / {:.1}%", percent(actual, used), percent(lru, used),
                           percent(arc, used), percent(lru2, used), percent(best, used)));
    }
    format!("Straydog expert reuse: {} uses of decode and verify passes on the first GPU; VRAM missed {:.1}%, an LRU store of the same \
             slots would miss {:.1}%, ARC {:.1}%, LRU-2 {:.1}%, the best possible {:.1}%; by class (missed / LRU / ARC / LRU-2 / best): {}",
            total[0], percent(total[1], total[0]), percent(total[2], total[0]), percent(total[4], total[0]), percent(total[5], total[0]),
            percent(total[3], total[0]), parts.join(", "))
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
    fn arc_and_lru2_miss_each_expert_once_where_all_fit_and_never_less_than_the_best() {
        let uses = keys(&[3, 1, 3, 2, 1, 1, 2, 3]);
        assert_eq!(arc_misses(&uses, 3), 3);
        assert_eq!(lru2_misses(&uses, 3), 3);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let uses: Vec<Key> = (0..20_000).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let draw = (state >> 33) as u32 % 1000;
            (0, if draw < 700 { draw % 40 } else { draw })
        }).collect();
        for slots in [1, 8, 40, 100, 400] {
            let best = best_misses(&uses, slots);
            assert!(arc_misses(&uses, slots) >= best && lru2_misses(&uses, slots) >= best, "{slots} slots");
        }
    }

    #[test]
    fn a_scan_through_the_store_spares_the_experts_used_often_under_arc_and_lru2() {
        // (a hot set of 30 used over and over, and every 50 uses a pass over 60 experts used once: LRU lets the scans push the hot
        // set out, ARC and LRU-2 keep it)
        let mut uses = Vec::new();
        let mut cold = 1000u32;
        for round in 0..200u32 {
            for hot in 0..30u32 {
                uses.push((0, (hot * 7 + round) % 30));
            }
            if round % 2 == 0 {
                for _ in 0..60 {
                    uses.push((0, cold));
                    cold += 1;
                }
            }
        }
        let (lru, arc, lru2) = (lru_misses(&uses, 50), arc_misses(&uses, 50), lru2_misses(&uses, 50));
        assert!(arc < lru && lru2 < lru, "LRU {lru}, ARC {arc}, LRU-2 {lru2}");
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
        assert!(line.contains("VRAM missed 75.0%, an LRU store of the same slots would miss 83.3%, ARC "), "{line}");
        assert!(line.contains(", the best possible 58.3%; by class (missed / LRU / ARC / LRU-2 / best): 0 (layers 0-9, 3 slots) 75.0 / 83.3 / "), "{line}");
        assert!(!line.contains("layers 10-19"), "{line}");
    }
}
