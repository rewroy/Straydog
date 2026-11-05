//! The state of a system block (the system prompt with its tools), kept for the next start of the engine.
//!
//! The first request of an app that starts the engine carries the same block every time, a few thousand tokens that a small card reads in
//! twenty seconds. The engine saves its state at the end of that block (its root checkpoint). On Windows the server asks the worker for that
//! state once per run, after the reply, seals it for the user ([`crate::prefix_store`]) in `<data dir>/prefix`, and at the next start, after
//! the worker is ready, opens the file that start's settings key and hands the state to the worker in the background: a request that comes
//! first never waits for it, and the worker takes it only while no request has had the engine. Elsewhere none of this is done.
//!
//! Which file a start takes is decided by a key: SHA-256 of the bundle, the engine's version and build, everything the worker's placement
//! decided that changes the state's layout or its numbers, and the tokens the state is held at ([`description`] lists them but the tokens).

#[cfg(windows)]
use std::path::PathBuf;
#[cfg(windows)]
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;

use crate::config::Settings;

/// What a key depends on (but the tokens), as text: every line a fact, in this order.
#[cfg(any(windows, test))]
struct Facts<'a> {
    bundle: String,
    version: &'a str,
    build: &'a str,
    /// The worker's digest of the layout of its states.
    layout: &'a str,
    context: String,
    placement: String,
    devices: String,
    drafts: String,
    numerics: String,
}

#[cfg(any(windows, test))]
impl Facts<'_> {
    fn description(&self) -> String {
        format!("bundle {}\nversion {}\nbuild {}\nlayout {}\ncontext {}\nplacement {}\ndevices {}\ndrafts {}\nnumerics {}\n",
                self.bundle, self.version, self.build, self.layout, self.context, self.placement, self.devices, self.drafts, self.numerics)
    }
}

/// The facts of this start: the settings it was given and what its worker said it placed (`ready`).
#[cfg(any(windows, test))]
fn facts<'a>(settings: &Settings, ready: &'a Value, build: &'a str) -> Option<Facts<'a>> {
    let placement = &ready["placement"];
    let layout = ready["prefix"]["layout"].as_str()?;
    Some(Facts {
        bundle: format!("{}/{}/{}", settings.bundle.model, settings.bundle.provider, settings.bundle.quant),
        version: env!("CARGO_PKG_VERSION"),
        build,
        layout,
        context: ready["context"].to_string(),
        // (what the worker chose, which the settings leave to it: the budget, the batch, where the K/V is, how much is resident)
        placement: format!("resident {} partial {} kv-on-host {} prefill-batch {}", placement["resident"], placement["partial"], placement["host_kv"],
                           placement["prefill_batch_size"]),
        devices: format!("{} split {}", settings.gpu.iter().map(usize::to_string).collect::<Vec<_>>().join(","), placement["split_mode"]),
        drafts: ready["speculative_drafts"].to_string(),
        // (these change the numbers a state holds, not its layout)
        numerics: format!("gemm {:?} min-batch {} expert-floor {:?} engram {:?}", settings.gemm, settings.gemm_min_batch, settings.expert_floor, settings.engram),
    })
}

/// The text a key's prefix is made of ([`crate::prefix_store::key_prefix`]), or None when the worker offers no states (it is not a model that
/// keeps them).
#[cfg(any(windows, test))]
pub fn description(settings: &Settings, ready: &Value, build: &str) -> Option<String> {
    facts(settings, ready, build).map(|facts| facts.description())
}

/// A state saved, opened: what the worker is to be handed.
pub struct Restored {
    #[cfg(windows)]
    opened: crate::prefix_store::Opened,
    /// What reading and opening it took, in seconds.
    pub seconds: f64,
}

impl Restored {
    pub fn state(&self) -> &[u8] {
        #[cfg(windows)]
        {
            self.opened.state()
        }
        #[cfg(not(windows))]
        {
            &[]
        }
    }
}

/// What the server keeps of a run's system block: where it saves the state and how it is keyed, and what the run has done about it.
pub struct Cache {
    #[cfg(windows)]
    on: Option<Arc<On>>,
}

#[cfg(windows)]
struct On {
    settings: Settings,
    directory: PathBuf,
    /// The executable's hash, being made by a thread of its own (the server's start is not held up by it), and once made.
    building: Mutex<Option<std::thread::JoinHandle<Result<[u8; 32], String>>>>,
    build: OnceLock<Result<[u8; 32], String>>,
    state: Mutex<State>,
}

#[cfg(windows)]
#[derive(Default)]
struct State {
    /// The root the last request made: its tokens, and their digest.
    root: Option<(u64, [u8; 32])>,
    /// This run has saved its state, or looked and found it saved, or tried.
    attempted: bool,
}

/// What a start's state is keyed on, and where it is kept.
#[cfg(windows)]
struct Identity {
    store: crate::prefix_store::Store,
    prefix: [u8; 32],
}

/// A state the server is to ask its worker for and save.
pub struct Due {
    /// The tokens the state is held at, and their digest, as the worker says them.
    pub tokens: u64,
    pub digest: String,
    #[cfg(windows)]
    identity: Identity,
    #[cfg(windows)]
    digest_bytes: [u8; 32],
}

#[cfg(windows)]
impl On {
    /// The executable's hash: waited for if it is still being made.
    fn build(&self) -> &Result<[u8; 32], String> {
        self.build.get_or_init(|| match self.building.lock().unwrap().take() {
            Some(thread) => thread.join().unwrap_or_else(|_| Err("its thread failed".into())),
            None => Err("it was not made".into()),
        })
    }

    /// The identity of this start: its store, and the prefix of its keys. None when the worker offers no states, or the executable could not
    /// be read.
    fn identity(&self, ready: &Value) -> Option<Identity> {
        let build = match self.build() {
            Ok(build) => *build,
            Err(message) => {
                tracing::debug!("Straydog prefix: no state kept, the executable cannot be read ({message})");
                return None;
            }
        };
        let text: String = build.iter().map(|byte| format!("{byte:02x}")).collect();
        let description = description(&self.settings, ready, &text)?;
        let bundle = format!("{}-{}-{}", self.settings.bundle.model, self.settings.bundle.provider, self.settings.bundle.quant);
        Some(Identity { store: crate::prefix_store::Store::new(self.directory.clone(), &bundle, &build), prefix: crate::prefix_store::key_prefix(&description) })
    }

    /// The newest state this start may be handed, opened ([`crate::prefix_store::Store::take_newest`]).
    fn open_newest(&self, ready: &Value) -> Option<Restored> {
        let started = std::time::Instant::now();
        let identity = self.identity(ready)?;
        for deleted in identity.store.prune(None) {
            tracing::debug!("Straydog prefix: deleted {}", deleted.display());
        }
        let opened = identity.store.take_newest(&crate::prefix_store::Dpapi, &identity.prefix, straydog_engine::prefix::VERSION)?;
        Some(Restored { opened, seconds: started.elapsed().as_secs_f64() })
    }
}

impl Cache {
    #[cfg(windows)]
    pub fn new(settings: &Settings) -> Cache {
        let on = (settings.prefix_cache && !settings.bundle.is_on_demand()).then(|| {
            Arc::new(On {
                settings: settings.clone(),
                directory: settings.data_dir.join("prefix"),
                building: Mutex::new(Some(std::thread::spawn(crate::prefix_store::build_id))),
                build: OnceLock::new(),
                state: Mutex::new(State::default()),
            })
        });
        Cache { on }
    }

    #[cfg(not(windows))]
    pub fn new(_settings: &Settings) -> Cache {
        Cache {}
    }

    /// Start opening the state this start may be handed (the newest file whose key prefix is this start's), on a thread of its own. The server
    /// hands the result to the worker unless a request has begun by then. The files that no start of this build can use are deleted first.
    pub fn restore(&self, ready: &Value) -> Option<tokio::task::JoinHandle<Option<Restored>>> {
        #[cfg(windows)]
        {
            let (on, ready) = (self.on.clone()?, ready.clone());
            Some(tokio::task::spawn_blocking(move || on.open_newest(&ready)))
        }
        #[cfg(not(windows))]
        {
            let _ = ready;
            None
        }
    }

    /// A request ended with this event: note the root it made, if it made one.
    pub fn note_done(&self, event: &Value) {
        #[cfg(windows)]
        if let Some(on) = &self.on {
            let timings = &event["timings"];
            let digest = timings["root_digest"].as_str().and_then(crate::prefix_store::digest_from_hex);
            if let (Some(tokens), Some(digest)) = (timings["root_checkpoint"].as_u64(), digest) {
                let mut state = on.state.lock().unwrap();
                if !state.attempted {
                    state.root = Some((tokens, digest));
                }
            }
        }
        #[cfg(not(windows))]
        let _ = event;
    }

    /// The state to ask the worker for now, if the last request made a root and this run has not saved a state, and no file holds it already.
    /// At most once a run: whatever comes of it, it is not asked for again.
    pub fn due(&self, ready: &Value) -> Option<Due> {
        #[cfg(windows)]
        {
            let on = self.on.as_ref()?;
            let (tokens, digest) = {
                let mut state = on.state.lock().unwrap();
                if state.attempted {
                    return None;
                }
                state.root.take()?
            };
            let identity = on.identity(ready)?;
            on.state.lock().unwrap().attempted = true;
            if identity.store.has(&identity.prefix, &crate::prefix_store::key_of(&identity.prefix, &digest)) {
                tracing::debug!("Straydog prefix: a file holds this block's state already");
                return None;
            }
            let hex = digest.iter().map(|byte| format!("{byte:02x}")).collect();
            Some(Due { tokens, digest: hex, identity, digest_bytes: digest })
        }
        #[cfg(not(windows))]
        {
            let _ = ready;
            None
        }
    }
}

impl Due {
    /// Seal `state` and write its file, on a thread of its own, then prune the store; the log says how it went.
    pub fn save(self, state: Vec<u8>) -> tokio::task::JoinHandle<()> {
        tokio::task::spawn_blocking(move || {
            #[cfg(windows)]
            {
                let started = std::time::Instant::now();
                let Due { tokens, identity, digest_bytes, .. } = self;
                match identity.store.save(&crate::prefix_store::Dpapi, straydog_engine::prefix::VERSION, &identity.prefix, &digest_bytes, state) {
                    Ok((path, bytes)) => {
                        tracing::info!("Straydog prefix: saved {tokens} tokens ({} MB) in {:.1} s", bytes >> 20, started.elapsed().as_secs_f64());
                        for deleted in identity.store.prune(Some(&path)) {
                            tracing::debug!("Straydog prefix: deleted {}", deleted.display());
                        }
                    }
                    Err(message) => tracing::warn!("Straydog prefix: not saved ({message})"),
                }
            }
            #[cfg(not(windows))]
            drop((self, state));
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn facts_of(change: impl Fn(&mut Value)) -> String {
        let mut ready = json!({"context": 262144, "speculative_drafts": 0, "prefix": {"version": 1, "layout": "aa"},
                               "placement": {"resident": false, "partial": true, "host_kv": true, "prefill_batch_size": 256,
                                             "vram_budgets": [5905580032u64], "split_mode": "experts"}});
        change(&mut ready);
        let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
        let facts = Facts { bundle: bundle.to_string(), version: "0.1.4", build: "b", layout: ready["prefix"]["layout"].as_str().unwrap(),
                            context: ready["context"].to_string(),
                            placement: format!("resident {} partial {} kv-on-host {} prefill-batch {}", ready["placement"]["resident"], ready["placement"]["partial"],
                                               ready["placement"]["host_kv"], ready["placement"]["prefill_batch_size"]),
                            devices: format!("0 split {}", ready["placement"]["split_mode"]), drafts: ready["speculative_drafts"].to_string(),
                            numerics: "gemm Fp16".into() };
        facts.description()
    }

    /// Every fact that changes what a state holds or how it is laid out changes the key's text: the layout, the context, the batch, where the K/V
    /// is, how much is resident, the split, the drafts. And what a start's budget leaves open, the VRAM it was given, does not: a card whose free
    /// memory moves a little between starts keeps its file.
    #[test]
    fn the_key_follows_what_changes_a_state() {
        let base = facts_of(|_| {});
        assert_eq!(base, facts_of(|_| {}));
        for (name, change) in [
            ("layout", Box::new(|ready: &mut Value| ready["prefix"]["layout"] = json!("bb")) as Box<dyn Fn(&mut Value)>),
            ("context", Box::new(|ready| ready["context"] = json!(131072))),
            ("batch", Box::new(|ready| ready["placement"]["prefill_batch_size"] = json!(1024))),
            ("K/V placement", Box::new(|ready| ready["placement"]["host_kv"] = json!(false))),
            ("residency", Box::new(|ready| ready["placement"]["resident"] = json!(true))),
            ("split", Box::new(|ready| ready["placement"]["split_mode"] = json!("layer"))),
            ("speculation", Box::new(|ready| ready["speculative_drafts"] = json!(3))),
        ] {
            assert_ne!(base, facts_of(change), "{name} does not change the key's text");
        }
        assert_eq!(base, facts_of(|ready| ready["placement"]["vram_budgets"] = json!([4_000_000_000u64])), "the budget a start was given is not a fact");
    }
}
