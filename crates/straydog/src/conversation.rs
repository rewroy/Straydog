//! The state of a conversation, kept on the drive for the next start (an option, off by default: `keep_conversation`, `--keep-conversation`).
//!
//! An engine that starts again, as an app's update makes it, has none of the conversation it held: the next request of a long conversation reads
//! all of it again, a few minutes at the speed of a small card. With the option on, the server asks its worker, once, when it stops on purpose (a
//! stop the app asked for, Ctrl+C, Windows ending the session), for the state of the conversation the worker held last: the sequence at the place its
//! last reply began, with the recurrent state, K/V and indexer's blocks of its positions ([`straydog_engine::model::Model::export_conversation`]).
//! The server seals it for the user ([`crate::conversation_store`], DPAPI and AES-256-GCM) in `<data dir>/conversation`, one file for a bundle, written
//! whole at the stop and at no other time: the drive sees one write a stop, and nothing is written while requests run.
//!
//! At the next start the file is opened lazily. The server reads its tokens, a part of their own, and tells the worker; the state stays on the drive.
//! The first request whose prompt begins with those tokens asks for the state (the worker's `conversation_wanted` event), the server reads and opens it
//! and sends it, and the worker continues the request from it. A request that does not begin with them leaves the file as it is, for a later one.
//!
//! Which file a start takes is decided by a key made as the system block's is ([`crate::prefix::description`]): the bundle, the engine's version and
//! build, and everything the worker's placement decided that changes the state's layout or its numbers. A start with another bundle, build or placement
//! never takes it; the file is deleted, as it is when the option is off at the start, or when it does not open.
//!
//! The state is sealed for the Windows account, so this is Windows's alone; elsewhere the option is accepted and keeps nothing.

use std::sync::Arc;

use serde_json::Value;

use crate::config::Settings;

/// What the server keeps of a run's conversation: where the file is and how it is keyed.
#[derive(Clone)]
pub struct Keeper {
    #[cfg(windows)]
    on: Option<Arc<On>>,
}

#[cfg(windows)]
struct On {
    settings: Settings,
    store: crate::conversation_store::Store,
    /// The executable's hash, made by a thread of its own.
    build: Arc<crate::prefix_store::Build>,
}

/// What the state saved at a stop took, for the line in the log.
pub struct Saved {
    pub tokens: usize,
    pub bytes: u64,
}

#[cfg(windows)]
impl On {
    /// The key of this start's file and the layout of the states it may take. None when the worker offers no states (it is not a model that keeps
    /// them), or the executable could not be read.
    fn key(&self, ready: &Value) -> Option<([u8; 32], u32)> {
        let build = match self.build.get() {
            Ok(build) => *build,
            Err(message) => {
                tracing::debug!("Straydog conversation: none kept, the executable cannot be read ({message})");
                return None;
            }
        };
        let text: String = build.iter().map(|byte| format!("{byte:02x}")).collect();
        let description = crate::prefix::description(&self.settings, ready, &text)?;
        Some((crate::conversation_store::key_of(&description), straydog_engine::prefix::VERSION))
    }
}

impl Keeper {
    /// The keeper of this run. With the option off the file an earlier run kept is deleted, at once: nothing of a conversation stays on the drive that
    /// the person has not asked to keep.
    #[cfg(windows)]
    pub fn new(settings: &Settings) -> Keeper {
        let label = format!("{}-{}-{}", settings.bundle.model, settings.bundle.provider, settings.bundle.quant);
        let store = crate::conversation_store::Store::new(&settings.data_dir.join("conversation"), &label);
        if store.begin(settings.keep_conversation) {
            tracing::info!("Straydog conversation: the option is off, deleted {}", store.path().display());
        }
        if !settings.keep_conversation {
            return Keeper { on: None };
        }
        if settings.bundle.is_on_demand() {
            tracing::info!("Straydog conversation: this model keeps none");
            return Keeper { on: None };
        }
        Keeper { on: Some(Arc::new(On { settings: settings.clone(), store, build: crate::prefix_store::shared_build() })) }
    }

    #[cfg(not(windows))]
    pub fn new(settings: &Settings) -> Keeper {
        if settings.keep_conversation {
            tracing::warn!("Straydog conversation: the state can be sealed for the user on Windows only; none is kept");
        }
        Keeper {}
    }

    /// Whether this run keeps the conversation at its stop.
    pub fn is_on(&self) -> bool {
        #[cfg(windows)]
        {
            self.on.is_some()
        }
        #[cfg(not(windows))]
        {
            false
        }
    }

    /// Open the file an earlier run kept as far as a start needs, on a thread of its own: its tokens. A file that does not open, for whatever is wrong
    /// with it, is deleted with one line in the log, and the start goes on as without it.
    pub fn restore(&self, ready: &Value) -> Option<tokio::task::JoinHandle<Option<Vec<u32>>>> {
        #[cfg(windows)]
        {
            let (on, ready) = (self.on.clone()?, ready.clone());
            Some(tokio::task::spawn_blocking(move || {
                let (key, layout) = on.key(&ready)?;
                let started = std::time::Instant::now();
                let tokens = on.store.tokens_or_delete(&crate::prefix_store::Dpapi, &key, layout)?;
                let bytes = std::fs::metadata(on.store.path()).map_or(0, |meta| meta.len());
                tracing::info!("Straydog conversation: {} tokens kept by the last run ({} MB on the drive, opened in {:.2} s); the first request that begins with them goes on from there",
                               tokens.len(), bytes >> 20, started.elapsed().as_secs_f64());
                Some(tokens)
            }))
        }
        #[cfg(not(windows))]
        {
            let _ = ready;
            None
        }
    }

    /// The kept state, read and opened (blocking): for the request that begins with the kept tokens. A file that no longer opens is deleted, as at the
    /// start. The seconds it took.
    pub fn state(&self, ready: &Value) -> Result<(Vec<u8>, f64), String> {
        #[cfg(windows)]
        {
            let on = self.on.as_ref().ok_or("the option is off")?;
            let (key, layout) = on.key(ready).ok_or("this engine keeps no conversation")?;
            let started = std::time::Instant::now();
            let state = on.store.state_or_delete(&crate::prefix_store::Dpapi, &key, layout)?;
            Ok((state, started.elapsed().as_secs_f64()))
        }
        #[cfg(not(windows))]
        {
            let _ = ready;
            Err("the option keeps nothing here".into())
        }
    }

    /// The most bytes a state may take to be kept now: what the drive of the data directory has free less a margin, and the commit that is left
    /// for the copy the server holds while it seals it (the worker's own copy is in the worker's commit, which it asked for first).
    pub fn room(&self) -> u64 {
        #[cfg(windows)]
        {
            let Some(on) = self.on.as_ref() else { return 0 };
            let drive = free_on_drive(on.store.path());
            let memory = straydog_engine::host::commit_left().map_or(u64::MAX, |(_, left)| left.saturating_sub(MEMORY_MARGIN));
            // (a file that is there is replaced when the new one is whole: both are on the drive for a moment)
            drive.saturating_sub(DRIVE_MARGIN).min(memory).min(crate::frames::BODY_LIMIT)
        }
        #[cfg(not(windows))]
        {
            0
        }
    }

    /// Seal `state` (the engine's, which holds its own tokens) and write the file over the one an earlier run kept (blocking); what it took.
    pub fn save(&self, ready: &Value, state: Vec<u8>) -> Result<Saved, String> {
        #[cfg(windows)]
        {
            let on = self.on.as_ref().ok_or("the option is off")?;
            let (key, layout) = on.key(ready).ok_or("this engine keeps no conversation")?;
            let tokens = {
                let parsed = straydog_engine::prefix::parse(&state)?;
                straydog_engine::prefix::tokens_of(parsed.payload, parsed.tokens)
            };
            let count = tokens.len();
            let (_, bytes) = on.store.save(&crate::prefix_store::Dpapi, layout, &key, &tokens, state)?;
            Ok(Saved { tokens: count, bytes })
        }
        #[cfg(not(windows))]
        {
            let _ = (ready, state);
            Err("the option keeps nothing here".into())
        }
    }
}

/// The drive keeps this much free beyond a file: a drive filled to its last byte stops everything else on it.
#[cfg(windows)]
const DRIVE_MARGIN: u64 = 2 << 30;
/// The commit a copy of the state in the server leaves free.
#[cfg(windows)]
const MEMORY_MARGIN: u64 = 1 << 30;

/// The bytes free on the drive that holds `path` (the longest mount point that begins it); the whole of a drive that cannot be found, so that
/// nothing is refused for it.
#[cfg(windows)]
fn free_on_drive(path: &std::path::Path) -> u64 {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let target = path.canonicalize().ok().or_else(|| path.ancestors().find_map(|ancestor| ancestor.canonicalize().ok())).unwrap_or_else(|| path.to_path_buf());
    // (the paths canonicalize to `\\?\C:\...`)
    let plain = |path: &std::path::Path| path.display().to_string().trim_start_matches(r"\\?\").to_lowercase();
    let target = plain(&target);
    disks.list().iter().filter(|disk| target.starts_with(&plain(disk.mount_point()))).max_by_key(|disk| disk.mount_point().as_os_str().len())
        .map_or(u64::MAX, |disk| disk.available_space())
}

/// What the line in the log says of a kept conversation.
pub fn saved_line(saved: &Saved, seconds: f64) -> String {
    format!("Straydog conversation: kept {} tokens ({} MB) in {seconds:.1} s for the next start", saved.tokens, saved.bytes >> 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line the log carries once a conversation is kept says what and how big and how long.
    #[test]
    fn the_line_in_the_log_says_what_was_kept() {
        let line = saved_line(&Saved { tokens: 36_168, bytes: 1_300 << 20 }, 4.26);
        assert_eq!(line, "Straydog conversation: kept 36168 tokens (1300 MB) in 4.3 s for the next start");
    }
}
