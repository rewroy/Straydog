//! The state of a conversation, kept on the drive for the next start (an option, off by default: `keep_conversation`, `--keep-conversation`).
//!
//! An engine that starts again, as an app's update makes it, has none of the conversation it held: the next request of a long conversation reads
//! all of it again, a few minutes at the speed of a small card. With the option on, the server asks its worker, once, when it stops on purpose (a
//! stop the app asked for, Ctrl+C, Windows ending the session), for the state of the conversation the worker held last: the sequence at the place its
//! last reply began, with the recurrent state, K/V and indexer's blocks of its positions ([`straydog_engine::model::Model::export_conversation`]).
//! The server seals it for the user ([`crate::conversation_store`], AES-256-GCM under a key DPAPI or the kernel's keyring keeps) in
//! `<data dir>/conversation`, up to [`KEPT_SLOTS`] files for a bundle (the conversations of the last stops), each written whole at a stop and at
//! no other time: the drive sees one write a stop, and nothing is written while requests run.
//!
//! Which file a stop writes ([`slot_for`]): the one whose conversation the worker's went on from, else an empty one, else the one used longest
//! ago; none when the worker's conversation is all in a kept one already (a health check's, or an app's warm-up of its system prompt, after a start).
//! One file would not do: an app's tests on the same server were the last thing before a stop, and the conversation the person was in was lost
//! with them (a 108K-token test conversation kept on 11-24 stood in for every later one, the person's 64K among them).
//!
//! At the next start the files are opened lazily. The server reads their tokens, a part of their own, and tells the worker; the states stay on the
//! drive. The first request whose prompt begins with one's tokens asks for its state (the worker's `conversation_wanted` event, with the slot), the
//! server reads and opens it and sends it, and the worker continues the request from it. A request that begins with none leaves them as they are.
//!
//! Which file a start takes is decided by a key made as the system block's is, but without the engine's version and build
//! ([`crate::prefix::lasting_description`]): the bundle and everything the worker's placement decided that changes the state's layout or its
//! numbers. A conversation goes on across a deploy; a start with another bundle or placement never takes it, and the file is deleted, as it is
//! when the option is off at the start, or when it does not open. A state whose bytes do not fit the engines is refused at import.
//!
//! The state is sealed for the user: on Windows by DPAPI, on Linux under a key the kernel's keyring holds for the user, which lasts the boot (a
//! state kept before a reboot no longer opens, and is deleted). Elsewhere the option is accepted and keeps nothing.

#[cfg(any(windows, target_os = "linux"))]
use std::sync::Arc;

use serde_json::Value;

use crate::config::Settings;

/// What the server keeps of a run's conversation: where the file is and how it is keyed.
#[derive(Clone)]
pub struct Keeper {
    #[cfg(any(windows, target_os = "linux"))]
    on: Option<Arc<On>>,
}

#[cfg(any(windows, target_os = "linux"))]
struct On {
    settings: Settings,
    /// The files of the slots, the first one the name a single kept conversation had.
    stores: Vec<crate::conversation_store::Store>,
}

/// The most conversations a bundle keeps: the person's, past the tests an app may run on the same server after it (some 1.3 GB at 100K tokens each
/// on DeepSeek-V4.1).
pub const KEPT_SLOTS: usize = 4;

/// A conversation an earlier run kept: its slot, its tokens, and when it was last written or taken.
#[derive(Clone, Debug)]
pub struct Kept {
    pub slot: usize,
    pub tokens: Vec<u32>,
    pub at: std::time::SystemTime,
}

/// The slot a stop writes the worker's conversation (`held`, its tokens) into, of `slots`: the kept one it went on from (the longest), an empty
/// one, else the one written or taken longest ago. None when it adds nothing: all of it is the start of a kept one, as a health check's or an app's
/// warm-up of its system prompt is.
pub fn slot_for(held: &[u32], kept: &[Kept], slots: usize) -> Option<usize> {
    if held.is_empty() || kept.iter().any(|one| one.tokens.starts_with(held)) {
        return None;
    }
    if let Some(one) = kept.iter().filter(|one| held.starts_with(&one.tokens)).max_by_key(|one| one.tokens.len()) {
        return Some(one.slot);
    }
    if let Some(free) = (0..slots).find(|slot| kept.iter().all(|one| one.slot != *slot)) {
        return Some(free);
    }
    kept.iter().min_by_key(|one| one.at).map(|one| one.slot)
}

/// The tokens a state the engine exported is held at.
pub fn state_tokens(state: &[u8]) -> Result<Vec<u32>, String> {
    let parsed = straydog_engine::prefix::parse(state)?;
    Ok(straydog_engine::prefix::tokens_of(parsed.payload, parsed.tokens))
}

/// What the state saved at a stop took, for the line in the log.
pub struct Saved {
    pub tokens: usize,
    pub bytes: u64,
}

#[cfg(any(windows, target_os = "linux"))]
impl On {
    /// The key of this start's file and the layout of the states it may take. None when the worker offers no states (it is not a model that keeps
    /// them), or the executable could not be read.
    fn key(&self, ready: &Value) -> Option<([u8; 32], u32)> {
        let description = crate::prefix::lasting_description(&self.settings, ready)?;
        Some((crate::conversation_store::key_of(&description), straydog_engine::prefix::VERSION))
    }
}

impl Keeper {
    /// The keeper of this run. With the option off the file an earlier run kept is deleted, at once: nothing of a conversation stays on the drive that
    /// the person has not asked to keep.
    #[cfg(any(windows, target_os = "linux"))]
    pub fn new(settings: &Settings) -> Keeper {
        let label = format!("{}-{}-{}", settings.bundle.model, settings.bundle.provider, settings.bundle.quant);
        let directory = settings.data_dir.join("conversation");
        let stores: Vec<_> = (0..KEPT_SLOTS).map(|slot| crate::conversation_store::Store::slot(&directory, &label, slot)).collect();
        for store in &stores {
            if store.begin(settings.keep_conversation) {
                tracing::info!("Straydog conversation: the option is off, deleted {}", store.path().display());
            }
        }
        if !settings.keep_conversation {
            return Keeper { on: None };
        }
        // (where the runtime directory and the home are not the server's to write, as under a service, the keyring's lock goes beside the file)
        #[cfg(target_os = "linux")]
        crate::prefix_store::keyring_lock_beside(&settings.data_dir.join("conversation"));
        Keeper { on: Some(Arc::new(On { settings: settings.clone(), stores })) }
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    pub fn new(settings: &Settings) -> Keeper {
        if settings.keep_conversation {
            tracing::warn!("Straydog conversation: the state can be sealed for the user on Windows and Linux only; none is kept");
        }
        Keeper {}
    }

    /// Whether this run keeps the conversation at its stop.
    pub fn is_on(&self) -> bool {
        #[cfg(any(windows, target_os = "linux"))]
        {
            self.on.is_some()
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        {
            false
        }
    }

    /// Open the files earlier runs kept as far as a start needs, on a thread of its own: their tokens. A file that does not open, for whatever is
    /// wrong with it, is deleted with one line in the log, and the start goes on as without it.
    pub fn restore(&self, ready: &Value) -> Option<tokio::task::JoinHandle<Vec<Kept>>> {
        #[cfg(any(windows, target_os = "linux"))]
        {
            let (on, ready) = (self.on.clone()?, ready.clone());
            Some(tokio::task::spawn_blocking(move || {
                let Some((key, layout)) = on.key(&ready) else { return Vec::new() };
                let mut kept = Vec::new();
                for (slot, store) in on.stores.iter().enumerate() {
                    let started = std::time::Instant::now();
                    let Some(tokens) = store.tokens_or_delete(crate::prefix_store::user_wrap(), &key, layout) else { continue };
                    let meta = std::fs::metadata(store.path()).ok();
                    let bytes = meta.as_ref().map_or(0, |meta| meta.len());
                    let at = meta.and_then(|meta| meta.modified().ok()).unwrap_or(std::time::UNIX_EPOCH);
                    tracing::info!("Straydog conversation: {} tokens kept by an earlier run ({} MB on the drive, slot {slot}, opened in {:.2} s); the first request that begins with them goes on from there",
                                   tokens.len(), bytes >> 20, started.elapsed().as_secs_f64());
                    kept.push(Kept { slot, tokens, at });
                }
                kept
            }))
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        {
            let _ = ready;
            None
        }
    }

    /// The state kept in `slot`, read and opened (blocking): for the request that begins with its tokens. A file that no longer opens is deleted, as
    /// at the start. The seconds it took.
    pub fn state(&self, ready: &Value, slot: usize) -> Result<(Vec<u8>, f64), String> {
        #[cfg(any(windows, target_os = "linux"))]
        {
            let on = self.on.as_ref().ok_or("the option is off")?;
            let store = on.stores.get(slot).ok_or("no such slot")?;
            let (key, layout) = on.key(ready).ok_or("this engine keeps no conversation")?;
            let started = std::time::Instant::now();
            let state = store.state_or_delete(crate::prefix_store::user_wrap(), &key, layout)?;
            Ok((state, started.elapsed().as_secs_f64()))
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        {
            let _ = (ready, slot);
            Err("the option keeps nothing here".into())
        }
    }

    /// The most bytes a state may take to be kept now: what the drive of the data directory has free less a margin, and the commit that is left
    /// for the copy the server holds while it seals it (the worker's own copy is in the worker's commit, which it asked for first).
    pub fn room(&self) -> u64 {
        #[cfg(any(windows, target_os = "linux"))]
        {
            let Some(on) = self.on.as_ref() else { return 0 };
            let drive = free_on_drive(on.stores[0].path());
            let memory = memory_left().map_or(u64::MAX, |left| left.saturating_sub(MEMORY_MARGIN));
            // (a file that is there is replaced when the new one is whole: both are on the drive for a moment)
            drive.saturating_sub(DRIVE_MARGIN).min(memory).min(crate::frames::BODY_LIMIT)
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        {
            0
        }
    }

    /// Seal `state` (the engine's, which holds its own tokens) and write the file of `slot` over what it held (blocking); what it took.
    pub fn save(&self, ready: &Value, slot: usize, state: Vec<u8>) -> Result<Saved, String> {
        #[cfg(any(windows, target_os = "linux"))]
        {
            let on = self.on.as_ref().ok_or("the option is off")?;
            let store = on.stores.get(slot).ok_or("no such slot")?;
            let (key, layout) = on.key(ready).ok_or("this engine keeps no conversation")?;
            let tokens = state_tokens(&state)?;
            let count = tokens.len();
            let (_, bytes) = store.save(crate::prefix_store::user_wrap(), layout, &key, &tokens, state)?;
            Ok(Saved { tokens: count, bytes })
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        {
            let _ = (ready, slot, state);
            Err("the option keeps nothing here".into())
        }
    }
}

/// The drive keeps this much free beyond a file: a drive filled to its last byte stops everything else on it.
#[cfg(any(windows, target_os = "linux"))]
const DRIVE_MARGIN: u64 = 2 << 30;
/// The commit a copy of the state in the server leaves free.
#[cfg(any(windows, target_os = "linux"))]
const MEMORY_MARGIN: u64 = 1 << 30;

/// The memory the server may still take for its copy of a state: the commit left (Windows), the memory available (Linux).
#[cfg(any(windows, target_os = "linux"))]
fn memory_left() -> Option<u64> {
    #[cfg(windows)]
    {
        straydog_engine::host::commit_left().map(|(_, left)| left)
    }
    #[cfg(target_os = "linux")]
    {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        Some(system.available_memory()).filter(|&bytes| bytes > 0)
    }
}

/// The bytes free on the drive that holds `path` (the longest mount point that begins it); the whole of a drive that cannot be found, so that
/// nothing is refused for it.
#[cfg(any(windows, target_os = "linux"))]
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

    fn kept(slot: usize, tokens: &[u32], at_seconds: u64) -> Kept {
        Kept { slot, tokens: tokens.to_vec(), at: std::time::UNIX_EPOCH + std::time::Duration::from_secs(at_seconds) }
    }

    /// Decides: a stop writes the conversation it held into the slot it went on from, an empty slot, or the one used longest ago, and writes nothing
    /// when all of it is in a kept one; so tests that run after the person's conversation take the other slots and leave it kept.
    #[test]
    fn a_stop_keeps_the_persons_conversation_past_later_ones() {
        let (system, person, test) = ([1, 2, 3], [1, 2, 3, 10, 11, 12, 13], [1, 2, 3, 20, 21]);
        assert_eq!(slot_for(&person, &[], 4), Some(0), "nothing kept: the first slot");
        assert_eq!(slot_for(&system, &[kept(0, &person, 10)], 4), None, "a warm-up of the system prompt adds nothing");
        assert_eq!(slot_for(&person[..5], &[kept(0, &person, 10)], 4), None, "nor a conversation that is the start of a kept one");
        assert_eq!(slot_for(&[1, 2, 3, 10, 11, 12, 13, 14], &[kept(2, &test, 30), kept(0, &person, 10)], 4), Some(0), "it went on from the person's");
        assert_eq!(slot_for(&test, &[kept(0, &person, 10)], 4), Some(1), "a test after it takes an empty slot");
        let full = [kept(0, &person, 40), kept(1, &[1, 2, 3, 30], 10), kept(2, &[1, 2, 3, 31], 20), kept(3, &[1, 2, 3, 32], 30)];
        assert_eq!(slot_for(&test, &full, 4), Some(1), "all slots taken: the one used longest ago");
        assert_eq!(slot_for(&[], &full, 4), None);
        // (tonight's order, 11-24/25: a test of 108K kept, the person's 64K, then the team's tests at three stops)
        let mut slots: Vec<Kept> = vec![kept(0, &[1, 2, 3, 99], 0)];
        for (at, held) in [(1, &person[..]), (2, &test[..]), (3, &[1, 2, 3, 40][..]), (4, &[1, 2, 3, 41][..])] {
            let slot = slot_for(held, &slots, 4).unwrap();
            slots.retain(|one| one.slot != slot);
            slots.push(kept(slot, held, at));
        }
        assert!(slots.iter().any(|one| one.tokens == person), "the person's conversation is kept: {slots:?}");
    }

    /// The line the log carries once a conversation is kept says what and how big and how long.
    #[test]
    fn the_line_in_the_log_says_what_was_kept() {
        let line = saved_line(&Saved { tokens: 36_168, bytes: 1_300 << 20 }, 4.26);
        assert_eq!(line, "Straydog conversation: kept 36168 tokens (1300 MB) in 4.3 s for the next start");
    }
}
