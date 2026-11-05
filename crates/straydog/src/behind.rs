//! What the worker does behind "ready", and gives way to a request: the warm-up of the GPU code and the page-locking of a partially
//! resident model's expert banks.
//!
//! The app starts the engine when it opens, and the person then types for a while: neither step has to be done before the engine says
//! ready. Both run on a thread of the worker, the warm-up first (it holds the engine for its passes, and stops at the next pass boundary when
//! a request comes), then the locking (it holds the engine only to take what it has locked, and yields to a request between layers).

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
    /// Why it stopped before its layers were done.
    pub stopped: Option<String>,
    /// The stop was a bank that does not match the file's tree: the model's bytes are not those pinned.
    pub unverified: bool,
    /// The engine ended first.
    pub gone: bool,
}

/// Why a layer could not be locked.
pub(crate) struct Stopped {
    pub message: String,
    pub unverified: bool,
}

/// How long the locking waits for a request to leave the engine before it takes its next layer, and for the lock of a layer it holds to be free.
const IDLE_POLL: Duration = Duration::from_millis(10);
const TAKE_POLL: Duration = Duration::from_millis(2);

/// Lock a partially resident model's banks layer by layer, never holding the engine across a layer's turn: `lock` takes the next layer's turn
/// (the page lock and the check against the file's tree, where nothing of the engine is held, with a strong reference to it so that the weights stay
/// mapped) and returns it with its number and bytes; `take` hands it to the model under the lock, where a request that holds the engine is
/// waited for; `give_up` lets a layer go that the model will not take. Between layers it waits until no request holds the engine. `ended` is
/// told how it went, under the lock, when it is over. `report` says it, at once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lock_behind<S, J, L>(engine: &Weak<Mutex<S>>, requests: impl Fn(&S) -> u64, begin: impl FnOnce(&mut S) -> Option<J>,
                                   mut lock: impl FnMut(&mut J) -> Option<Result<(L, usize, u64), Stopped>>,
                                   mut take: impl FnMut(&mut S, L) -> Result<(), String>, give_up: impl Fn(L),
                                   ended: impl FnOnce(&mut S, &Locking), report: impl FnOnce(&Locking)) -> Locking {
    let started = Instant::now();
    let mut done = Locking::default();
    let (mut job, first) = {
        let Some(alive) = engine.upgrade() else { done.gone = true; return done };
        let Ok(mut guard) = alive.lock() else { done.gone = true; return done };
        let Some(job) = begin(&mut guard) else { return done };
        (job, requests(&guard))
    };
    'layers: loop {
        // (a request holds the engine: the next layer waits for it to end)
        let alive = loop {
            let Some(alive) = engine.upgrade() else { done.gone = true; break 'layers };
            let state = match alive.try_lock() {
                Ok(guard) => {
                    drop(guard);
                    Ok(true)
                }
                Err(TryLockError::Poisoned(_)) => Err(()),
                Err(TryLockError::WouldBlock) => Ok(false),
            };
            match state {
                Ok(true) => break alive,
                Ok(false) => {
                    drop(alive);
                    std::thread::sleep(IDLE_POLL);
                }
                Err(()) => {
                    done.gone = true;
                    break 'layers;
                }
            }
        };
        let Some(next) = lock(&mut job) else { break };
        let (locked, _, bytes) = match next {
            Ok(locked) => locked,
            Err(stop) => {
                done.stopped = Some(stop.message);
                done.unverified = stop.unverified;
                break;
            }
        };
        // (a request that came meanwhile has the engine: the layer is taken when it ends)
        let guard = loop {
            match alive.try_lock() {
                Ok(guard) => break Some(guard),
                Err(TryLockError::Poisoned(_)) => break None,
                Err(TryLockError::WouldBlock) => std::thread::sleep(TAKE_POLL),
            }
        };
        let Some(mut guard) = guard else {
            give_up(locked);
            done.gone = true;
            break 'layers;
        };
        done.requests = requests(&guard).saturating_sub(first);
        match take(&mut guard, locked) {
            Ok(()) => {
                done.layers += 1;
                done.bytes += bytes;
            }
            Err(message) => {
                done.stopped = Some(message);
                break;
            }
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
    use std::sync::Arc;

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
        let locked_log = plan.locked.clone();
        lock_behind(&Arc::downgrade(engine), |model: &Model| model.requests, |_| Some(plan),
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

    /// Decides: a request that holds the engine is not held up by the locking, which waits for it to end before it takes its next layer, and
    /// the layer it had locked meanwhile is taken after the request, with the request counted.
    #[test]
    fn a_request_has_the_engine_whatever_the_locking_has_reached() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let requests: Arc<Mutex<Vec<std::thread::JoinHandle<Duration>>>> = Arc::new(Mutex::new(Vec::new()));
        // a request takes the engine as the locking is at layer 46 (nothing of the engine is held there), and holds it for 200 ms
        let between = {
            let (engine, requests) = (engine.clone(), requests.clone());
            move |layer: usize| {
                if layer == 46 {
                    let engine = engine.clone();
                    let (sender, receiver) = std::sync::mpsc::channel();
                    requests.lock().unwrap().push(std::thread::spawn(move || {
                        let started = Instant::now();
                        let mut model = engine.lock().unwrap();
                        sender.send(()).unwrap();
                        model.requests += 1;
                        std::thread::sleep(Duration::from_millis(200));
                        started.elapsed()
                    }));
                    // (the request has the engine before the layer is taken)
                    receiver.recv().unwrap();
                }
            }
        };
        let locking = drive(&engine, plan(&[47, 46, 45, 44], 99), between);
        let held: Vec<Duration> = requests.lock().unwrap().drain(..).map(|request| request.join().unwrap()).collect();
        assert_eq!(held.len(), 1);
        assert!(held[0] >= Duration::from_millis(200) && held[0] < Duration::from_secs(5), "the request held the engine {:?}", held[0]);
        assert_eq!((locking.layers, locking.requests, locking.stopped), (4, 1, None));
        assert_eq!(engine.lock().unwrap().taken, vec![47, 46, 45, 44], "the layer locked while the request ran is taken after it");
    }

    /// Decides: a locking whose engine ended (nobody holds it) is over without taking anything more.
    #[test]
    fn the_engine_ending_ends_the_locking() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let weak = Arc::downgrade(&engine);
        drop(engine);
        let locking = lock_behind(&weak, |model: &Model| model.requests, |_| Some(()), |_: &mut ()| Some(Ok(((), 0, 0))), |_: &mut Model, ()| Ok(()), |()| {},
                                  |_: &mut Model, _| panic!("no engine to tell"), |_| {});
        assert!(locking.gone);
    }

    /// Decides: a model with nothing to lock is left alone and told nothing.
    #[test]
    fn nothing_planned_locks_nothing() {
        let engine = Arc::new(Mutex::new(Model::default()));
        let locking = lock_behind(&Arc::downgrade(&engine), |model: &Model| model.requests, |_| None::<()>, |_: &mut ()| None::<Result<((), usize, u64), Stopped>>,
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
