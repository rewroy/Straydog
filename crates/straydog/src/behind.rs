//! What the worker does behind "ready", and gives way to a request: the warm-up of the GPU code and the page-locking of a partially
//! resident model's expert banks.
//!
//! The app starts the engine when it opens, and the person then types for a while: neither step has to be done before the engine says
//! ready. Both run on a thread of the worker, the warm-up first (it holds the engine for its passes, and stops at the next pass boundary when
//! a request comes), then the locking (it holds the engine only to hand the model what it has locked: a request never waits for a layer's turn,
//! and the turns give way to a request that is generating).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, TryLockError, Weak};
use std::time::{Duration, Instant};

/// Whether the request loop has read a command that takes the engine since the worker began: the warm-up behind "ready" is over at its next
/// pass, or does not begin. A static because the engine is behind the lock the warm-up holds, and the request loop is generic over its
/// engine.
pub(crate) static ARRIVED: AtomicBool = AtomicBool::new(false);

/// Whether the warm-up has yet to end (it is set before "ready" is sent): what takes the engine in the background waits for it, not
/// taking it for a request's.
pub(crate) static WARMING: AtomicBool = AtomicBool::new(false);

/// The state of a warm start that came while the warm-up held the engine, kept for the thread behind "ready" to begin the warm start with when
/// the warm-up is over (as it did when the warm-up came before "ready"), and with the layers it locks first set from it.
static PENDING_WARM: Mutex<Option<serde_json::Value>> = Mutex::new(None);

/// Keep a warm start's `state` for when the warm-up is over, if it is under way: whether it was kept (if not, the warm start begins at once).
pub(crate) fn hold_warm_start(state: &serde_json::Value) -> bool {
    let mut pending = PENDING_WARM.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if !WARMING.load(Ordering::SeqCst) {
        return false;
    }
    *pending = Some(state.clone());
    true
}

/// The warm-up is over: what waits for it goes on. The state of a warm start that came meanwhile, if one did.
pub(crate) fn warm_up_over() -> Option<serde_json::Value> {
    let mut pending = PENDING_WARM.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    WARMING.store(false, Ordering::SeqCst);
    pending.take()
}

/// How the warm-up behind "ready" ended.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WarmedUp {
    /// Both passes ran to their ends.
    Done,
    /// A request came, and the model was reset at the pass boundary it was seen at.
    CutShort,
    /// A request had the engine first.
    Skipped,
    Failed(String),
    /// The engine is gone (the worker ended).
    Gone,
}

/// Run `warm_up` on the engine, which it holds for its passes: it is told how to see a request (`arrived`), and says whether it ran to its end.
pub(crate) fn warm_up_behind<S>(engine: &Weak<Mutex<S>>, arrived: &AtomicBool, warm_up: impl FnOnce(&mut S, &dyn Fn() -> bool) -> Result<bool, String>) -> WarmedUp {
    let Some(alive) = engine.upgrade() else { return WarmedUp::Gone };
    let Ok(mut guard) = alive.lock() else { return WarmedUp::Gone };
    // (a request that took the engine first: the engine is its own, and the GPU code loads with its first pass as after any reset)
    if arrived.load(Ordering::SeqCst) {
        return WarmedUp::Skipped;
    }
    match warm_up(&mut guard, &|| arrived.load(Ordering::SeqCst)) {
        Ok(true) => WarmedUp::Done,
        Ok(false) => WarmedUp::CutShort,
        Err(message) => WarmedUp::Failed(message),
    }
}

/// How the locking gives way to a request between two layers' turns: it looks for a request that has the engine every `poll` and waits while there is
/// one, for `at_most` in all before the next turn, which it then takes whatever the engine is doing (as it did before it gave way at all).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Yielding {
    pub poll: Duration,
    pub at_most: Duration,
}

/// What the worker gives way with. A layer's turn takes 0.1 to 0.5 s of reads and checks, and the gap between an agent's requests (a tool runs) holds
/// one, so a short poll finds the gaps; a locking that waited for the engine to be idle for a time did not, with requests back to back (it was not done
/// 55 s after "ready"). The guard is for a request that never lets go: the locking goes on, a layer in 20 s at the least.
pub(crate) const YIELDING: Yielding = Yielding { poll: Duration::from_millis(5), at_most: Duration::from_secs(20) };

/// How the locking behind "ready" went.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Locking {
    /// The layers taken by the model, and their bytes.
    pub layers: usize,
    pub bytes: u64,
    /// The seconds from the first layer's turn to the end.
    pub seconds: f64,
    /// The requests the engine ran while the locking went on.
    pub requests: u64,
    /// How long the locking waited in all, before layers' turns, for a request that had the engine.
    pub waited: Duration,
    /// A wait reached the guard's time, and a layer's turn was taken with a request still having the engine.
    pub guard_fired: bool,
    /// Why it stopped before its layers were done.
    pub stopped: Option<String>,
    /// The stop was a bank that does not match the file's tree: the model's bytes are not those pinned.
    pub unverified: bool,
    /// The engine ended first.
    pub gone: bool,
    /// The model would take no more of the layers locked (an error of its own), and the rest were let go.
    take_failed: bool,
}

/// Whether a request has the engine.
enum Held {
    Free,
    Busy,
    /// The engine ended (or a request panicked in it).
    Gone,
}

/// Look at the engine's lock without keeping it (nor a strong reference to the engine, so that it can end while the locking waits).
fn held<S>(engine: &Weak<Mutex<S>>) -> Held {
    let Some(alive) = engine.upgrade() else { return Held::Gone };
    let state = match alive.try_lock() {
        Ok(_) => Held::Free,
        Err(TryLockError::WouldBlock) => Held::Busy,
        Err(TryLockError::Poisoned(_)) => Held::Gone,
    };
    state
}

/// Why a layer could not be locked.
pub(crate) struct Stopped {
    pub message: String,
    pub unverified: bool,
}

/// Lock a partially resident model's banks layer by layer, never holding the engine across a layer's turn: `lock` takes the next layer's turn
/// (the page lock and the check against the file's tree, where nothing of the engine is held, with a strong reference to it so that the weights stay
/// mapped) and returns it with its number and bytes. A request never waits for a layer's turn (0.1 to 0.5 s of reads and checks, some seconds for
/// the whole plan), and a turn is not begun while a request has the engine: the locking's reads, hashing and registration take the cores and the
/// memory bandwidth that a generating request's CPU experts need, so before each turn it waits for the request to let go, looking every
/// `pacing.poll`, for `pacing.at_most` in all since the last turn (a request that never lets go does not hold the locking back for good: the next
/// turn is then taken as it was before the locking gave way, and the count begins again after it). The gap between an agent's requests is enough, and
/// is not waited for to last. A turn that has begun is not stopped. The layers locked while a request had the engine are handed to the model
/// together (`take`, under the lock) at the first moment the engine is free after a turn, or when the last is locked, which waits for a request
/// that has it. So requests that follow one another at once, as an agent's do, find the layers registered at the first gap between them.
/// `job` is the plan, made under the engine's lock before a request could take it, with the engine's request count then (None: nothing is to be
/// locked), and `more` says whether it has a layer left (no wait comes before a turn that will not be). `give_up` lets a layer go that the model will not take. `ended` is told how it went, under the lock, when it is over. `report` says it,
/// at once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lock_behind<S, J, L>(engine: &Weak<Mutex<S>>, requests: impl Fn(&S) -> u64, pacing: Yielding, job: Option<(J, u64)>,
                                   more: impl Fn(&J) -> bool,
                                   mut lock: impl FnMut(&mut J) -> Option<Result<(L, usize, u64), Stopped>>,
                                   mut take: impl FnMut(&mut S, L) -> Result<(), String>, give_up: impl Fn(L),
                                   ended: impl FnOnce(&mut S, &Locking), report: impl FnOnce(&Locking)) -> Locking {
    let started = Instant::now();
    let mut done = Locking::default();
    // (nothing planned: nothing to lock)
    let Some((mut job, first)) = job else { return done };
    // The layers locked and checked that the model has not taken yet, with their bytes.
    let mut waiting: Vec<(L, u64)> = Vec::new();
    // Hand what waits to the model, under the lock; once it would take no more, the rest are let go.
    let mut hand_over = |guard: &mut S, waiting: &mut Vec<(L, u64)>, done: &mut Locking| {
        done.requests = requests(guard).saturating_sub(first);
        for (locked, bytes) in waiting.drain(..) {
            if done.take_failed {
                give_up(locked);
            } else {
                match take(guard, locked) {
                    Ok(()) => {
                        done.layers += 1;
                        done.bytes += bytes;
                    }
                    Err(message) => {
                        done.stopped = Some(message);
                        done.take_failed = true;
                    }
                }
            }
        }
    };
    'layers: loop {
        if !more(&job) {
            break;
        }
        // A request that has the engine is waited for to let go (not for the engine to stay idle: a gap is enough), until the guard's time.
        let wait_began = Instant::now();
        let mut blocked = false;
        loop {
            match held(engine) {
                Held::Free => break,
                Held::Gone => {
                    done.gone = true;
                    break 'layers;
                }
                Held::Busy if wait_began.elapsed() > pacing.at_most => {
                    done.guard_fired = true;
                    break;
                }
                Held::Busy => {
                    blocked = true;
                    std::thread::sleep(pacing.poll);
                }
            }
        }
        if blocked {
            done.waited += wait_began.elapsed();
        }
        // (held while a layer's turn runs, so that the weights it reads stay mapped)
        let Some(alive) = engine.upgrade() else { done.gone = true; break };
        match lock(&mut job) {
            None => break,
            Some(Err(stop)) => {
                done.stopped = Some(stop.message);
                done.unverified = stop.unverified;
                break;
            }
            Some(Ok((locked, _, bytes))) => waiting.push((locked, bytes)),
        }
        match alive.try_lock() {
            Ok(mut guard) => hand_over(&mut guard, &mut waiting, &mut done),
            // (a request has the engine: the layers wait for it, and the next layer's turn goes on)
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Poisoned(_)) => {
                done.gone = true;
                break;
            }
        }
        if done.take_failed {
            break;
        }
    }
    // The plan is over (done, stopped, or the engine ended): what waits goes to the model when the engine is free, or is let go.
    if !waiting.is_empty() {
        match engine.upgrade() {
            // (the model's bytes do not match the file's tree: no layer is taken)
            Some(_alive) if done.unverified => waiting.drain(..).for_each(|(locked, _)| give_up(locked)),
            Some(alive) => match alive.lock() {
                Ok(mut guard) => hand_over(&mut guard, &mut waiting, &mut done),
                Err(_) => {
                    done.gone = true;
                    waiting.drain(..).for_each(|(locked, _)| give_up(locked));
                }
            },
            // (the engine is gone: its weights may be unmapped, and the locks go with the process)
            None => done.gone = true,
        }
    }
    done.seconds = started.elapsed().as_secs_f64();
    report(&done);
    if let Some(alive) = engine.upgrade() {
        if let Ok(mut guard) = alive.lock() {
            ended(&mut guard, &done);
        }
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    /// A short poll and a guard that the tests which are not about it do not reach.
    const QUICK: Yielding = Yielding { poll: Duration::from_millis(2), at_most: Duration::from_secs(30) };

    /// An engine's side of the locking: the layers it has taken in order, the requests it has run, and whether it was told the locking ended.
    #[derive(Default)]
    struct Model {
        requests: u64,
        taken: Vec<usize>,
        ended: Option<(usize, Option<String>)>,
    }

    /// A plan of `layers`, of which `refuse` is the first that cannot be locked (a layer past the end: none).
    struct Plan {
        layers: Vec<usize>,
        refuse: usize,
        unverified: bool,
        locked: Arc<Mutex<Vec<usize>>>,
    }

    fn drive(engine: &Arc<Mutex<Model>>, plan: Plan, between: impl Fn(usize) + Send + 'static) -> Locking {
        drive_with(engine, plan, QUICK, between)
    }

    fn drive_with(engine: &Arc<Mutex<Model>>, plan: Plan, pacing: Yielding, between: impl Fn(usize) + Send + 'static) -> Locking {
        let locked_log = plan.locked.clone();
        lock_behind(&Arc::downgrade(engine), |model: &Model| model.requests, pacing, Some((plan, 0)),
                    |plan: &Plan| !plan.layers.is_empty(),
                    |plan: &mut Plan| {
                        if plan.layers.is_empty() {
                            return None;
                        }
                        let layer = plan.layers.remove(0);
                        // (the lock: nothing of the engine is held, so a request may have it now)
                        between(layer);
                        if layer == plan.refuse {
                            return Some(Err(Stopped { message: format!("layer {layer} refused"), unverified: plan.unverified }));
                        }
                        locked_log.lock().unwrap().push(layer);
                        Some(Ok((layer, layer, 100)))
                    },
                    |model: &mut Model, layer| {
                        model.taken.push(layer);
                        Ok(())
                    },
                    |_layer| {},
                    |model: &mut Model, locking| model.ended = Some((locking.layers, locking.stopped.clone())),
                    |_| {})
    }

    fn plan(layers: &[usize], refuse: usize) -> Plan {
        Plan { layers: layers.to_vec(), refuse, unverified: false, locked: Arc::new(Mutex::new(Vec::new())) }
    }

    /// Decides: every layer is locked and then taken, in the order planned, and the model is told when it is over.
    #[test]
    fn the_layers_are_locked_and_taken_in_order() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let locking = drive(&engine, plan(&[47, 46, 45], 99), |_| {});
        assert_eq!((locking.layers, locking.bytes, locking.requests, locking.stopped.clone()), (3, 300, 0, None));
        assert_eq!((locking.waited, locking.guard_fired), (Duration::ZERO, false), "no request had the engine: nothing was waited for");
        let model = engine.lock().unwrap();
        assert_eq!(model.taken, vec![47, 46, 45]);
        assert_eq!(model.ended, Some((3, None)));
    }

    /// Decides: a layer that cannot be locked ends the locking with its reason, and those before it stand; an unverified one says so.
    #[test]
    fn a_layer_that_cannot_be_locked_ends_it_with_the_reason() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let locking = drive(&engine, plan(&[47, 46, 45, 44], 45), |_| {});
        assert_eq!((locking.layers, locking.stopped.as_deref(), locking.unverified), (2, Some("layer 45 refused"), false));
        assert_eq!(engine.lock().unwrap().taken, vec![47, 46]);
        assert_eq!(engine.lock().unwrap().ended, Some((2, Some("layer 45 refused".to_string()))));
        let engine = Arc::new(Mutex::new(Model::default()));
        let mut failing = plan(&[47, 46], 46);
        failing.unverified = true;
        assert!(drive(&engine, failing, |_| {}).unverified);
    }

    /// Decides: a request that holds the engine is not held up by the locking, and the locking waits for it between two layers: no turn is entered
    /// while it generates, the layers after it are locked once it has let go and taken together after them, and the wait is in the result.
    #[test]
    fn a_request_has_the_engine_whatever_the_locking_has_reached() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let requests: Arc<Mutex<Vec<std::thread::JoinHandle<Duration>>>> = Arc::new(Mutex::new(Vec::new()));
        let locked_while_held = Arc::new(Mutex::new(Vec::new()));
        // a request takes the engine as the locking is at layer 46 (nothing of the engine is held there), and holds it for 300 ms
        let between = {
            let (engine, requests, locked_while_held) = (engine.clone(), requests.clone(), locked_while_held.clone());
            move |layer: usize| {
                if layer == 46 {
                    let engine = engine.clone();
                    let (sender, receiver) = std::sync::mpsc::channel();
                    requests.lock().unwrap().push(std::thread::spawn(move || {
                        let started = Instant::now();
                        let mut model = engine.lock().unwrap();
                        sender.send(()).unwrap();
                        model.requests += 1;
                        std::thread::sleep(Duration::from_millis(300));
                        started.elapsed()
                    }));
                    // (the request has the engine before the layer is taken)
                    receiver.recv().unwrap();
                } else if layer < 46 && engine.try_lock().is_err() {
                    // (a turn entered while the request has the engine)
                    locked_while_held.lock().unwrap().push(layer);
                }
            }
        };
        let locking = drive(&engine, plan(&[47, 46, 45, 44], 99), between);
        let held: Vec<Duration> = requests.lock().unwrap().drain(..).map(|request| request.join().unwrap()).collect();
        assert_eq!(held.len(), 1);
        assert!(held[0] >= Duration::from_millis(300) && held[0] < Duration::from_secs(5), "the request held the engine {:?}", held[0]);
        assert_eq!(*locked_while_held.lock().unwrap(), Vec::<usize>::new(), "no turn was entered while the request had the engine");
        assert!(locking.waited >= Duration::from_millis(200) && locking.waited < Duration::from_secs(5), "waited {:?} for a request of 300 ms", locking.waited);
        assert!(!locking.guard_fired);
        assert_eq!((locking.layers, locking.requests, locking.stopped), (4, 1, None));
        assert_eq!(engine.lock().unwrap().taken, vec![47, 46, 45, 44], "the layers locked while the request ran are taken after it, in order");
    }

    /// Runs the locking of layers 47, 46, 45 (the last refused, as `unverified` says) with a request holding the engine from layer 46's turn on for
    /// 200 ms: the layers taken, those given up, and how it went.
    fn refused_during_a_request(unverified: bool) -> (Locking, Vec<usize>, Vec<usize>) {
        let engine = Arc::new(Mutex::new(Model::default()));
        let (given_up, at_46, proceed) = (Arc::new(Mutex::new(Vec::new())), std::sync::mpsc::channel::<()>(), std::sync::mpsc::channel::<()>());
        let locker = {
            let (engine, given_up) = (Arc::downgrade(&engine), given_up.clone());
            let (at_46, proceed) = (at_46.0, Mutex::new(proceed.1));
            std::thread::spawn(move || {
                lock_behind(&engine, |model: &Model| model.requests, QUICK, Some((vec![47, 46, 45], 0)),
                            |layers: &Vec<usize>| !layers.is_empty(),
                            |layers: &mut Vec<usize>| {
                                let layer = layers.remove(0);
                                if layer == 46 {
                                    // (the request takes the engine now)
                                    at_46.send(()).unwrap();
                                    proceed.lock().unwrap().recv().unwrap();
                                }
                                if layer == 45 {
                                    return Some(Err(Stopped { message: "stopped here".into(), unverified }));
                                }
                                Some(Ok((layer, layer, 100)))
                            },
                            |model: &mut Model, layer| {
                                model.taken.push(layer);
                                Ok(())
                            }, move |layer| given_up.lock().unwrap().push(layer), |_: &mut Model, _| {}, |_| {})
            })
        };
        at_46.1.recv().unwrap();
        let held = engine.lock().unwrap();
        proceed.0.send(()).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        drop(held);
        let locking = locker.join().unwrap();
        let taken = engine.lock().unwrap().taken.clone();
        let given_up = given_up.lock().unwrap().clone();
        (locking, taken, given_up)
    }

    /// Decides: a layer that cannot be locked ends the plan, and the layers locked before it that a request kept the engine from are still taken.
    #[test]
    fn the_layers_locked_before_a_refusal_are_taken_after_the_request() {
        let (locking, taken, given_up) = refused_during_a_request(false);
        assert_eq!((locking.layers, locking.stopped.as_deref(), locking.unverified), (2, Some("stopped here"), false));
        assert_eq!(taken, vec![47, 46]);
        assert!(given_up.is_empty());
    }

    /// Decides: layers waiting when a layer's bytes are found not to match the file's tree are let go, not taken, and the plan says so.
    #[test]
    fn nothing_is_taken_once_a_layer_does_not_verify() {
        let (locking, taken, given_up) = refused_during_a_request(true);
        // (layer 47 was taken while the engine was free; 46 was locked while the request had it, and is let go)
        assert!(locking.unverified && locking.layers == 1, "{locking:?}");
        assert_eq!((taken, given_up), (vec![47], vec![46]));
    }

    /// Decides: requests that follow one another with a gap between them, as an agent's do while a tool runs, do not starve the locking: each turn
    /// is entered in a gap (never while a request has the engine), every layer is locked well before the guard's time, and the guard does not fire.
    #[test]
    fn requests_back_to_back_leave_the_gaps_to_the_locking() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let agent = {
            let (engine, stop) = (engine.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    {
                        let mut model = engine.lock().unwrap();
                        model.requests += 1;
                        std::thread::sleep(Duration::from_millis(40));
                    }
                    // (the tool runs)
                    std::thread::sleep(Duration::from_millis(25));
                }
            })
        };
        // (the agent has a request under way before the locking begins)
        while engine.lock().unwrap().requests == 0 {
            std::thread::yield_now();
        }
        let entered_busy = Arc::new(AtomicUsize::new(0));
        let between = {
            let (engine, entered_busy) = (engine.clone(), entered_busy.clone());
            move |_layer: usize| {
                // (a turn takes some milliseconds, more than a gap holds, as the real one takes hundreds)
                if engine.try_lock().is_err() {
                    entered_busy.fetch_add(1, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(12));
            }
        };
        let began = Instant::now();
        let locking = drive(&engine, plan(&[9, 8, 7, 6, 5, 4], 99), between);
        let took = began.elapsed();
        stop.store(true, Ordering::SeqCst);
        agent.join().unwrap();
        assert_eq!((locking.layers, locking.stopped.clone(), locking.guard_fired), (6, None, false));
        assert_eq!(entered_busy.load(Ordering::SeqCst), 0, "a turn was entered while a request had the engine");
        assert!(locking.requests >= 1 && locking.waited > Duration::ZERO, "{locking:?}");
        assert!(took < Duration::from_secs(10), "six layers took {took:?} among requests with gaps of 25 ms");
        assert_eq!(engine.lock().unwrap().taken, vec![9, 8, 7, 6, 5, 4]);
    }

    /// Decides: a request that never lets go does not starve the locking: after the guard's time each layer's turn is taken as it was before the
    /// locking gave way (the guard counts again after each), the layers are handed over together when the request ends, and the result says that the
    /// guard fired and how long the locking waited.
    #[test]
    fn a_request_that_never_lets_go_holds_each_layer_back_for_the_guard_only() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let turns: Arc<Mutex<Vec<Instant>>> = Arc::new(Mutex::new(Vec::new()));
        let guard = Duration::from_millis(120);
        // a request that has the engine from before the first turn until the fourth turn has begun, and a little past
        let (holding, begun) = (std::sync::mpsc::channel::<()>(), Arc::new(AtomicUsize::new(0)));
        let request = {
            let (engine, begun) = (engine.clone(), begun.clone());
            std::thread::spawn(move || {
                let mut model = engine.lock().unwrap();
                model.requests += 1;
                holding.0.send(()).unwrap();
                while begun.load(Ordering::SeqCst) < 4 {
                    std::thread::sleep(Duration::from_millis(2));
                }
                std::thread::sleep(Duration::from_millis(50));
            })
        };
        holding.1.recv().unwrap();
        let between = {
            let (turns, begun) = (turns.clone(), begun.clone());
            move |_layer: usize| {
                turns.lock().unwrap().push(Instant::now());
                begun.fetch_add(1, Ordering::SeqCst);
            }
        };
        let began = Instant::now();
        let locking = drive_with(&engine, plan(&[3, 2, 1, 0], 99), Yielding { poll: Duration::from_millis(2), at_most: guard }, between);
        request.join().unwrap();
        assert!(locking.guard_fired);
        assert_eq!((locking.layers, locking.requests, locking.stopped.clone()), (4, 1, None));
        assert_eq!(engine.lock().unwrap().taken, vec![3, 2, 1, 0], "the layers locked under the request are taken after it, in order");
        let turns = turns.lock().unwrap();
        assert_eq!(turns.len(), 4);
        // the first turn waited the guard's time, and so did each after it: the guard counts from the last turn
        assert!(turns[0].duration_since(began) >= guard, "the first turn was entered {:?} after the start", turns[0].duration_since(began));
        for pair in turns.windows(2) {
            let gap = pair[1].duration_since(pair[0]);
            assert!(gap >= guard && gap < guard * 20, "two turns {gap:?} apart with a guard of {guard:?}");
        }
        assert!(locking.waited >= guard * 4 && locking.waited < guard * 40, "waited {:?} for a guard of {guard:?}", locking.waited);
    }

    /// Decides: a locking that waited for a request that then let go within the guard's time says how long it waited and that the guard did not fire.
    #[test]
    fn the_wait_is_in_the_result_and_the_guard_did_not_fire() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let request = {
            let engine = engine.clone();
            let (sender, receiver) = std::sync::mpsc::channel();
            let request = std::thread::spawn(move || {
                let mut model = engine.lock().unwrap();
                model.requests += 1;
                sender.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(250));
            });
            receiver.recv().unwrap();
            request
        };
        let locking = drive(&engine, plan(&[2, 1], 99), |_| {});
        request.join().unwrap();
        assert!(!locking.guard_fired);
        assert!(locking.waited >= Duration::from_millis(150) && locking.waited < Duration::from_secs(10), "waited {:?}", locking.waited);
        assert_eq!((locking.layers, locking.requests), (2, 1));
    }

    /// Decides: a locking whose engine ended (nobody holds it) is over without taking anything more.
    #[test]
    fn the_engine_ending_ends_the_locking() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let weak = Arc::downgrade(&engine);
        drop(engine);
        let locking = lock_behind(&weak, |model: &Model| model.requests, QUICK, Some(((), 0)), |_: &()| true, |_: &mut ()| Some(Ok(((), 0, 0))), |_: &mut Model, ()| Ok(()), |()| {},
                                  |_: &mut Model, _| panic!("no engine to tell"), |_| {});
        assert!(locking.gone);
    }

    /// Decides: a model with nothing to lock is left alone and told nothing.
    #[test]
    fn nothing_planned_locks_nothing() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let locking = lock_behind(&Arc::downgrade(&engine), |model: &Model| model.requests, QUICK, None::<((), u64)>, |_: &()| true, |_: &mut ()| None::<Result<((), usize, u64), Stopped>>,
                                  |_: &mut Model, ()| Ok(()), |()| {}, |_: &mut Model, _| panic!("nothing was begun"), |_| {});
        assert_eq!((locking.layers, locking.gone), (0, false));
        assert!(engine.lock().unwrap().ended.is_none());
    }

    /// Decides: a warm-up runs on the engine and says whether it ran to its end; one a request cut short says so; and one that a request had
    /// the engine before does not run.
    #[test]
    fn the_warm_up_runs_stops_at_a_request_and_does_not_run_after_one() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let arrived = AtomicBool::new(false);
        assert_eq!(warm_up_behind(&Arc::downgrade(&engine), &arrived, |_, cancelled| Ok(!cancelled())), WarmedUp::Done);
        assert_eq!(warm_up_behind(&Arc::downgrade(&engine), &arrived, |_, cancelled| {
            arrived.store(true, Ordering::SeqCst);
            Ok(!cancelled())
        }), WarmedUp::CutShort);
        assert_eq!(warm_up_behind(&Arc::downgrade(&engine), &arrived, |_, _| panic!("a request came first")), WarmedUp::Skipped);
        arrived.store(false, Ordering::SeqCst);
        assert_eq!(warm_up_behind(&Arc::downgrade(&engine), &arrived, |_, _| Err("the device said no".into())), WarmedUp::Failed("the device said no".into()));
        let weak = Arc::downgrade(&engine);
        drop(engine);
        assert_eq!(warm_up_behind(&weak, &arrived, |_: &mut Model, _| Ok(true)), WarmedUp::Gone);
    }

    /// Decides: a warm start asked for while the warm-up is under way is kept until it is over, and one asked for after begins at once. (The
    /// only test that moves the statics.)
    #[test]
    fn a_warm_start_asked_for_during_the_warm_up_waits_for_it() {
        let state = serde_json::json!({"kind": "routes"});
        WARMING.store(false, Ordering::SeqCst);
        assert!(!hold_warm_start(&state), "no warm-up is under way: the warm start begins at once");
        WARMING.store(true, Ordering::SeqCst);
        assert!(hold_warm_start(&state));
        assert_eq!(warm_up_over(), Some(state.clone()), "the state is handed on once, as the warm-up ends");
        assert!(!WARMING.load(Ordering::SeqCst));
        assert_eq!(warm_up_over(), None);
        assert!(!hold_warm_start(&state), "the warm-up is over: the warm start begins at once");
    }

    /// Decides: a request that comes while the warm-up holds the engine waits for a pass at most: the warm-up sees it and lets go.
    #[test]
    fn a_request_waits_for_a_pass_of_the_warm_up_at_most() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let arrived = Arc::new(AtomicBool::new(false));
        let warming = {
            let (weak, arrived) = (Arc::downgrade(&engine), arrived.clone());
            std::thread::spawn(move || {
                warm_up_behind(&weak, &arrived.clone(), |_, cancelled| {
                    // passes of 5 ms, for as long as nothing has come
                    for _ in 0..2000 {
                        if cancelled() {
                            return Ok(false);
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Ok(true)
                })
            })
        };
        std::thread::sleep(Duration::from_millis(100));
        let asked = Instant::now();
        arrived.store(true, Ordering::SeqCst);
        engine.lock().unwrap().requests += 1;
        let waited = asked.elapsed();
        assert_eq!(warming.join().unwrap(), WarmedUp::CutShort);
        assert!(waited < Duration::from_secs(2), "the request waited {waited:?}");
    }
}
