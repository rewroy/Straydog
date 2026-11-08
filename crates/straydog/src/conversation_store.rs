//! A conversation's state, sealed on the drive for the next start (Windows): the format of the file and the store that keeps it.
//!
//! The file is the system block's ([`crate::prefix_store`]) with two sealed parts in place of one: the tokens the state is held at, a small part of
//! their own, and the state (K/V and recurrent states, as the engine's export gives them). A random 256-bit key, wrapped with DPAPI for the user and
//! stored in the header, seals both with AES-256-GCM, each under a random 96-bit nonce of its own and with the header and its part's name as the
//! associated data: the file opens only as the user who wrote it, and only if not one byte of it changed. The tokens open without the state being
//! read (a file's tokens are what a start looks at, and the state is read only for a request that begins with them). Nothing in plaintext is ever
//! written, not even a part of a file: the ciphertext goes to a `.part` file, which is synced and then renamed over the old file.
//!
//! One file is kept for a bundle: a new one replaces the old.
//!
//! ```text
//! 0    8   "STRAYCNV"
//! 8    4   FILE_VERSION
//! 12   4   the engine's layout version of the state inside (straydog_engine::prefix::VERSION)
//! 16   32  the key: SHA-256 of everything the state depends on (the bundle, the engine's version and build, the settings that decide its layout)
//! 48   8   the tokens the state is held at, T
//! 56   8   the state's bytes (before sealing), S
//! 64   4   the wrapped key's bytes, W
//! 68   W   the wrapped key (DPAPI)
//! ..   12  the tokens' nonce
//! ..   12  the state's nonce
//! ..       the tokens sealed (T u32, little-endian), and the 16-byte tag after them
//! ..       the state sealed, and the 16-byte tag after it
//! ```

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};

use crate::prefix_store::{hash, Wrap};

/// The format of the files.
const FILE_VERSION: u32 = 1;
const MAGIC: [u8; 8] = *b"STRAYCNV";
const EXTENSION: &str = "sdconv";
const TAG: usize = 16;
const NONCE: usize = 12;
const FIXED: usize = 68;
/// The most a wrapped key takes (DPAPI's blob of 32 bytes is about 200).
const WRAPPED_LIMIT: usize = 4096;
/// The most tokens a file may say: a context of a million tokens is more than any bundle has.
const TOKENS_LIMIT: u64 = 1 << 24;
/// A `.part` file older than this is what a killed write left.
const STALE_PART: Duration = Duration::from_secs(600);

/// The key of a conversation's file: SHA-256 of what a state depends on, `description`'s text ([`crate::prefix::description`]: the bundle, the
/// engine's version and build, every setting that changes the state's layout or its numbers). Not the system block's keys' domain: no file of one
/// kind is the other's.
pub fn key_of(description: &str) -> [u8; 32] {
    hash(&[b"straydog conversation key 1\n", description.as_bytes()])
}

/// What a file's header says, and where its parts are.
struct Head {
    layout: u32,
    key: [u8; 32],
    tokens: u64,
    state: u64,
    wrapped: Vec<u8>,
    tokens_nonce: [u8; NONCE],
    state_nonce: [u8; NONCE],
}

impl Head {
    /// The header as the file holds it; it is the associated data of both parts.
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.length());
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&FILE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.layout.to_le_bytes());
        bytes.extend_from_slice(&self.key);
        bytes.extend_from_slice(&self.tokens.to_le_bytes());
        bytes.extend_from_slice(&self.state.to_le_bytes());
        bytes.extend_from_slice(&(self.wrapped.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.wrapped);
        bytes.extend_from_slice(&self.tokens_nonce);
        bytes.extend_from_slice(&self.state_nonce);
        bytes
    }

    fn length(&self) -> usize {
        FIXED + self.wrapped.len() + 2 * NONCE
    }

    /// The bytes the tokens' part takes in the file, and the state's.
    fn parts(&self) -> (u64, u64) {
        (4 * self.tokens + TAG as u64, self.state + TAG as u64)
    }

    /// The header at the start of `file`, whose length is `file_length`, and the bytes it took; an error when it is not one, or when the file is not
    /// exactly as long as the header says. Only the header is read.
    fn read(file: &mut std::fs::File, file_length: u64) -> Result<(Head, Vec<u8>), String> {
        let mut fixed = vec![0u8; FIXED];
        file.read_exact(&mut fixed).map_err(|_| "it is not a conversation file".to_string())?;
        if fixed[..8] != MAGIC {
            return Err("it is not a conversation file".into());
        }
        let word = |at: usize| u32::from_le_bytes(fixed[at..at + 4].try_into().unwrap());
        let long = |at: usize| u64::from_le_bytes(fixed[at..at + 8].try_into().unwrap());
        if word(8) != FILE_VERSION {
            return Err(format!("its format is version {}", word(8)));
        }
        let (tokens, state, wrapped) = (long(48), long(56), word(64) as usize);
        if wrapped > WRAPPED_LIMIT || tokens > TOKENS_LIMIT {
            return Err("its header says more than a file of this kind holds".into());
        }
        let mut rest = vec![0u8; wrapped + 2 * NONCE];
        file.read_exact(&mut rest).map_err(|_| "its header is short".to_string())?;
        let head = Head {
            layout: word(12),
            key: fixed[16..48].try_into().unwrap(),
            tokens,
            state,
            wrapped: rest[..wrapped].to_vec(),
            tokens_nonce: rest[wrapped..wrapped + NONCE].try_into().unwrap(),
            state_nonce: rest[wrapped + NONCE..].try_into().unwrap(),
        };
        let (tokens_part, state_part) = head.parts();
        let expected = (head.length() as u64).checked_add(tokens_part).and_then(|bytes| bytes.checked_add(state_part));
        if expected != Some(file_length) {
            return Err(format!("its length is {file_length} where its header says {}", expected.map_or("more than a drive holds".to_string(), |bytes| bytes.to_string())));
        }
        fixed.extend_from_slice(&rest);
        Ok((head, fixed))
    }
}

/// What a part is called in the associated data: a part is never taken for the other.
const TOKENS: &[u8] = b"tokens";
const STATE: &[u8] = b"state";

/// The associated data of a part: the header, the part's name, and for the state the tokens' part as the file holds it, so that a state opens only
/// beside the tokens it was sealed with.
fn aad(head: &[u8], part: &[u8], bound: &[u8]) -> Vec<u8> {
    [head, part, bound].concat()
}

/// Seal `buffer` in place under `key` and `nonce`, with `aad`: its tag.
fn seal(key: &[u8], nonce: [u8; NONCE], aad: &[u8], buffer: &mut [u8]) -> Result<[u8; TAG], String> {
    let unbound = UnboundKey::new(&AES_256_GCM, key).map_err(|_| "no AES-256-GCM key".to_string())?;
    let tag = LessSafeKey::new(unbound).seal_in_place_separate_tag(Nonce::assume_unique_for_key(nonce), Aad::from(aad), buffer)
        .map_err(|_| "sealing failed".to_string())?;
    Ok(tag.as_ref().try_into().expect("a 16-byte tag"))
}

/// Open `sealed` (a part and its tag) in place: the part's length.
fn open_part(key: &[u8], nonce: [u8; NONCE], aad: &[u8], sealed: &mut [u8]) -> Result<usize, String> {
    let unbound = UnboundKey::new(&AES_256_GCM, key).map_err(|_| "its wrapped key is not an AES-256-GCM key".to_string())?;
    LessSafeKey::new(unbound).open_in_place(Nonce::assume_unique_for_key(nonce), Aad::from(aad), sealed)
        .map(|plain| plain.len())
        .map_err(|_| "it does not open: a byte of it is not as it was written".to_string())
}

/// The file of one bundle's kept conversation, in a directory.
pub struct Store {
    path: PathBuf,
}

/// A file opened as far as a start needs: its header, checked against what this start is.
struct Opened {
    file: std::fs::File,
    head: Head,
    bytes: Vec<u8>,
    key: Vec<u8>,
}

impl Store {
    /// The store of `bundle` (a label, made fit for a file name) in `directory`.
    pub fn new(directory: &Path, bundle: &str) -> Store {
        let bundle: String = bundle.chars().map(|letter| if letter.is_ascii_alphanumeric() || matches!(letter, '_' | '-') { letter } else { '_' }).collect();
        Store { path: directory.join(format!("{bundle}.{EXTENSION}")) }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn exists(&self) -> bool {
        self.path.is_file()
    }

    /// Delete the file, and a part a killed write left of it: whether anything was deleted.
    pub fn delete(&self) -> bool {
        let _ = std::fs::remove_file(self.part());
        std::fs::remove_file(&self.path).is_ok()
    }

    fn part(&self) -> PathBuf {
        PathBuf::from(format!("{}.part", self.path.display()))
    }

    /// A part a killed write left, once it is old and not before: it may be a write in progress.
    pub fn remove_stale_part(&self) {
        let stale = std::fs::metadata(self.part()).ok().and_then(|meta| meta.modified().ok()).and_then(|time| time.elapsed().ok()).is_some_and(|age| age > STALE_PART);
        if stale {
            let _ = std::fs::remove_file(self.part());
        }
    }

    /// Open the file's header: it must be one this engine's `layout` and this start's `key` made, and its wrapped key must unwrap for this user. Errors
    /// say what was wrong; None when there is no file.
    fn open(&self, wrap: &dyn Wrap, key: &[u8; 32], layout: u32) -> Result<Option<Opened>, String> {
        let mut file = match std::fs::File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.to_string()),
        };
        let length = file.metadata().map_err(|error| error.to_string())?.len();
        let (head, bytes) = Head::read(&mut file, length)?;
        if head.layout != layout {
            return Err(format!("its state is of layout {}, this engine's is {layout}", head.layout));
        }
        if head.key != *key {
            return Err("it is another engine's, build's or settings'".into());
        }
        let unwrapped = wrap.unwrap(&head.wrapped)?;
        Ok(Some(Opened { file, head, bytes, key: unwrapped }))
    }

    /// The tokens the kept state is held at, and nothing else of the file read: the header and the tokens' part. None when there is no file; an error
    /// (the caller deletes the file) when it is not one this start may take or has been changed.
    pub fn open_tokens(&self, wrap: &dyn Wrap, key: &[u8; 32], layout: u32) -> Result<Option<Vec<u32>>, String> {
        let Some(Opened { mut file, head, bytes, mut key }) = self.open(wrap, key, layout)? else { return Ok(None) };
        let opened = (|| {
            let (part, _) = head.parts();
            let mut sealed = vec![0u8; part as usize];
            file.read_exact(&mut sealed).map_err(|error| error.to_string())?;
            let plain = open_part(&key, head.tokens_nonce, &aad(&bytes, TOKENS, &[]), &mut sealed)?;
            if plain as u64 != 4 * head.tokens {
                return Err("its tokens are not the count its header says".to_string());
            }
            Ok(straydog_engine::prefix::tokens_of(&sealed, head.tokens as usize))
        })();
        key.fill(0);
        opened.map(Some)
    }

    /// [`open_tokens`](Self::open_tokens) for a start: a file that does not open, for whatever is wrong with it, is deleted, with one line in the log,
    /// and the start goes on as without it.
    pub fn tokens_or_delete(&self, wrap: &dyn Wrap, key: &[u8; 32], layout: u32) -> Option<Vec<u32>> {
        match self.open_tokens(wrap, key, layout) {
            Ok(tokens) => tokens,
            Err(reason) => {
                self.delete();
                tracing::warn!("Straydog conversation: deleted {} ({reason}); this start goes on without it", self.path.display());
                None
            }
        }
    }

    /// [`open_state`](Self::open_state) for a request: a file that no longer opens is deleted, as at the start.
    pub fn state_or_delete(&self, wrap: &dyn Wrap, key: &[u8; 32], layout: u32) -> Result<Vec<u8>, String> {
        self.open_state(wrap, key, layout).inspect_err(|reason| {
            self.delete();
            tracing::warn!("Straydog conversation: deleted {} ({reason})", self.path.display());
        })
    }

    /// The start of a run: with the option off, the file an earlier run kept is deleted (nothing of a conversation stays on the drive that the person
    /// has not asked to keep); whether it was there. A run that keeps removes only the part a killed write left.
    pub fn begin(&self, keep: bool) -> bool {
        if keep {
            self.remove_stale_part();
            return false;
        }
        self.delete()
    }

    /// The kept state, opened: the tokens' part skipped, the state's read and authenticated whole. As [`open_tokens`](Self::open_tokens) for what a
    /// file may be; a file that is not there is an error here.
    pub fn open_state(&self, wrap: &dyn Wrap, key: &[u8; 32], layout: u32) -> Result<Vec<u8>, String> {
        let Some(Opened { mut file, head, bytes, mut key }) = self.open(wrap, key, layout)? else { return Err("the file is gone".into()) };
        let opened = (|| {
            let (tokens_part, state_part) = head.parts();
            // (the tokens' part, sealed as it is in the file, is bound into the state's associated data)
            let mut tokens_sealed = vec![0u8; tokens_part as usize];
            file.read_exact(&mut tokens_sealed).map_err(|error| error.to_string())?;
            let mut sealed = Vec::new();
            sealed.try_reserve_exact(state_part as usize).map_err(|_| format!("no memory for {} MB of state", state_part >> 20))?;
            let read = (&mut file).take(state_part).read_to_end(&mut sealed).map_err(|error| error.to_string())?;
            if read as u64 != state_part {
                return Err("its state is short".to_string());
            }
            let plain = open_part(&key, head.state_nonce, &aad(&bytes, STATE, &tokens_sealed), &mut sealed)?;
            if plain as u64 != head.state {
                return Err("its state is not the length its header says".to_string());
            }
            sealed.truncate(plain);
            Ok(sealed)
        })();
        key.fill(0);
        opened
    }

    /// Seal `tokens` and `state` (the bytes of the engine's state of layout `layout`, held at those tokens) under `key` and write the file, over the
    /// old one: to a `.part` file, synced, then renamed, so no reader sees less than all of it and nothing in plaintext is written. The path, and the
    /// bytes of the file. The old file stays until the new one is whole.
    pub fn save(&self, wrap: &dyn Wrap, layout: u32, key: &[u8; 32], tokens: &[u32], mut state: Vec<u8>) -> Result<(PathBuf, u64), String> {
        let random = SystemRandom::new();
        let (mut secret, mut tokens_nonce, mut state_nonce) = ([0u8; 32], [0u8; NONCE], [0u8; NONCE]);
        for bytes in [&mut secret[..], &mut tokens_nonce[..], &mut state_nonce[..]] {
            random.fill(bytes).map_err(|_| "no random bytes for a key")?;
        }
        let head = wrap.wrap(&secret).map(|wrapped| Head { layout, key: *key, tokens: tokens.len() as u64, state: state.len() as u64, wrapped, tokens_nonce, state_nonce });
        let sealed = head.and_then(|head| {
            let bytes = head.bytes();
            let mut sealed_tokens = straydog_engine::prefix::token_bytes(tokens);
            let tokens_tag = seal(&secret, tokens_nonce, &aad(&bytes, TOKENS, &[]), &mut sealed_tokens)?;
            let bound = [&sealed_tokens[..], &tokens_tag[..]].concat();
            let state_tag = seal(&secret, state_nonce, &aad(&bytes, STATE, &bound), &mut state)?;
            Ok((bytes, sealed_tokens, tokens_tag, state_tag))
        });
        // (the key is in the header, wrapped, and nowhere else)
        secret.fill(0);
        let (head, sealed_tokens, tokens_tag, state_tag) = sealed?;
        let part = self.part();
        let written = (|| -> std::io::Result<()> {
            if let Some(directory) = self.path.parent() {
                std::fs::create_dir_all(directory)?;
            }
            let mut file = std::fs::File::create(&part)?;
            file.write_all(&head)?;
            file.write_all(&sealed_tokens)?;
            file.write_all(&tokens_tag)?;
            file.write_all(&state)?;
            file.write_all(&state_tag)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&part, &self.path)
        })();
        if let Err(error) = written {
            let _ = std::fs::remove_file(&part);
            return Err(format!("{}: {error}", self.path.display()));
        }
        Ok((self.path.clone(), (head.len() + sealed_tokens.len() + TAG + state.len() + TAG) as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    /// A wrapper that needs no user: the key's bytes flipped.
    struct Flip;

    impl Wrap for Flip {
        fn wrap(&self, key: &[u8]) -> Result<Vec<u8>, String> {
            Ok(key.iter().map(|byte| !byte).collect())
        }

        fn unwrap(&self, blob: &[u8]) -> Result<Vec<u8>, String> {
            Ok(blob.iter().map(|byte| !byte).collect())
        }
    }

    /// A wrapper for a user who is not the one that wrapped: nothing unwraps.
    struct Nobody;

    impl Wrap for Nobody {
        fn wrap(&self, key: &[u8]) -> Result<Vec<u8>, String> {
            Ok(key.to_vec())
        }

        fn unwrap(&self, _: &[u8]) -> Result<Vec<u8>, String> {
            Err("not this user's".into())
        }
    }

    fn directory(name: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!("straydog-conversation-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn state(length: usize) -> Vec<u8> {
        (0..length).map(|index| (index * 31 % 251) as u8).collect()
    }

    fn tokens(count: usize) -> Vec<u32> {
        (0..count as u32).map(|index| 1000 + index * 7 % 5000).collect()
    }

    /// A state and its tokens sealed come back as they were; the file is as long as its header says, and holds none of the state's bytes or the
    /// tokens' in the clear and none of the key.
    #[test]
    fn a_sealed_conversation_opens_as_it_was() {
        let dir = directory("round");
        let store = Store::new(&dir, "Qwen3.8-X");
        let key = key_of("a");
        let (kept, plain) = (tokens(5000), state(100_000));
        assert!(store.open_tokens(&Flip, &key, 1).unwrap().is_none(), "no file, no tokens");
        let (path, bytes) = store.save(&Flip, 1, &key, &kept, plain.clone()).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes);
        assert_eq!(path, store.path());
        let file = std::fs::read(&path).unwrap();
        assert!(!file.windows(64).any(|window| plain.windows(64).any(|clear| clear == window)), "a run of the state is in the file");
        let clear_tokens = straydog_engine::prefix::token_bytes(&kept);
        assert!(!file.windows(64).any(|window| clear_tokens.windows(64).any(|clear| clear == window)), "a run of the tokens is in the file");
        assert_eq!(store.open_tokens(&Flip, &key, 1).unwrap(), Some(kept));
        assert_eq!(store.open_state(&Flip, &key, 1).unwrap(), plain);
        assert!(std::fs::read_dir(&dir).unwrap().flatten().all(|entry| !entry.file_name().to_string_lossy().ends_with(".part")), "no part is left");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The tokens open without the state: a state damaged beyond its first byte leaves the tokens readable (the state is never read for them) and
    /// is refused when it is asked for; and the tokens' own part, read alone, is the few hundred kilobytes they take.
    #[test]
    fn the_tokens_open_without_the_state_part() {
        let dir = directory("tokens");
        let store = Store::new(&dir, "Q");
        let key = key_of("a");
        let kept = tokens(36_000);
        store.save(&Flip, 1, &key, &kept, state(2_000_000)).unwrap();
        let mut file = std::fs::read(store.path()).unwrap();
        let last = file.len() - 1;
        file[last / 2 + 100_000] ^= 0x10;
        std::fs::write(store.path(), &file).unwrap();
        assert_eq!(store.open_tokens(&Flip, &key, 1).unwrap(), Some(kept), "the tokens open with the state damaged");
        assert!(store.open_state(&Flip, &key, 1).unwrap_err().contains("does not open"));
        // the tokens' part is a part of its own: it is T words and a tag, in front of the state's
        let mut opened = std::fs::File::open(store.path()).unwrap();
        let length = opened.metadata().unwrap().len();
        let (head, _) = Head::read(&mut opened, length).unwrap();
        assert_eq!(head.parts().0, 4 * 36_000 + TAG as u64);
        assert!(head.parts().0 < 200_000 && head.parts().1 > 2_000_000);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A byte flipped anywhere in the file, the header's, the wrapped key's, the nonces', the tokens', the state's or the tags', is refused by the
    /// part it belongs to or by both, and so is a file cut short or one with a byte more.
    #[test]
    fn a_flipped_byte_anywhere_is_refused() {
        let dir = directory("flip");
        let store = Store::new(&dir, "Q");
        let key = key_of("a");
        let kept = tokens(300);
        store.save(&Flip, 1, &key, &kept, state(5_000)).unwrap();
        let good = std::fs::read(store.path()).unwrap();
        let head = FIXED + 32 + 2 * NONCE;
        let tokens_end = head + 4 * 300 + TAG;
        assert!(store.open_state(&Flip, &key, 1).is_ok() && store.open_tokens(&Flip, &key, 1).is_ok());
        // (every byte of the header and of the tokens' part with its tag, and every 97th of the state: each a file of its own)
        let places: Vec<usize> = (0..tokens_end).chain((tokens_end..good.len() - TAG).step_by(97)).chain(good.len() - TAG..good.len()).collect();
        for place in places {
            let mut bad = good.clone();
            bad[place] ^= 0x01;
            std::fs::write(store.path(), &bad).unwrap();
            // (the state's part is checked when it is read, and the header's bytes are the associated data of both)
            assert!(store.open_state(&Flip, &key, 1).is_err(), "byte {place} of {} flipped and the state opened", good.len());
            if place < tokens_end {
                assert!(store.open_tokens(&Flip, &key, 1).is_err(), "byte {place} of {} flipped and the tokens opened", good.len());
            }
        }
        std::fs::write(store.path(), &good[..good.len() - 1]).unwrap();
        assert!(store.open_tokens(&Flip, &key, 1).unwrap_err().contains("length"));
        let mut longer = good.clone();
        longer.push(0);
        std::fs::write(store.path(), &longer).unwrap();
        assert!(store.open_state(&Flip, &key, 1).unwrap_err().contains("length"));
        std::fs::write(store.path(), &good).unwrap();
        assert!(store.open_state(&Flip, &key, 1).is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Another bundle's, build's or placement's key is refused (the key is made of the same facts as the system block's), and so is another layout of
    /// the state inside, and a file of another user's: it cannot be unwrapped.
    #[test]
    fn another_builds_placement_or_user_is_refused() {
        let dir = directory("other");
        let store = Store::new(&dir, "Qwen3.8-X");
        let key = key_of("bundle A, build 1");
        store.save(&Flip, 1, &key, &tokens(100), state(1_000)).unwrap();
        assert!(store.open_tokens(&Flip, &key, 1).unwrap().is_some());
        assert!(store.open_tokens(&Flip, &key_of("bundle A, build 2"), 1).unwrap_err().contains("another"));
        assert!(store.open_state(&Flip, &key_of("bundle A, placement 2"), 1).unwrap_err().contains("another"));
        assert!(store.open_tokens(&Flip, &key, 2).unwrap_err().contains("layout"));
        assert!(store.open_state(&Nobody, &key, 1).unwrap_err().contains("not this user's"));
        assert!(store.open_tokens(&Nobody, &key, 1).unwrap_err().contains("not this user's"));
        // the key is made of the facts of the system block's, and is not the system block's key
        assert_ne!(key_of("x"), crate::prefix_store::key_prefix("x"));
        assert_ne!(key_of("x"), key_of("y"));
        // a file of another bundle is not this store's
        let other = Store::new(&dir, "Qwen3.8-Y");
        assert!(other.open_tokens(&Flip, &key, 1).unwrap().is_none());
        // a file of the other kind (the system block's) is not one
        std::fs::write(store.path(), b"STRAYPFX and the rest of it, which is not ours").unwrap();
        assert!(store.open_tokens(&Flip, &key, 1).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// One file is kept for a bundle: a second save replaces the first (whole, never half), leaves no part, and a failed one leaves the first.
    #[test]
    fn a_new_file_replaces_the_old_one() {
        let dir = directory("replace");
        let store = Store::new(&dir, "Q");
        let key = key_of("a");
        store.save(&Flip, 1, &key, &tokens(100), state(1_000)).unwrap();
        let (kept, plain) = (tokens(200), state(3_000));
        store.save(&Flip, 1, &key, &kept, plain.clone()).unwrap();
        assert_eq!(std::fs::read_dir(&dir).unwrap().flatten().count(), 1, "one file a bundle, and no part");
        assert_eq!(store.open_tokens(&Flip, &key, 1).unwrap(), Some(kept));
        assert_eq!(store.open_state(&Flip, &key, 1).unwrap(), plain);
        // a save that cannot wrap its key writes nothing and leaves the old file
        struct Refuses;
        impl Wrap for Refuses {
            fn wrap(&self, _: &[u8]) -> Result<Vec<u8>, String> {
                Err("no DPAPI".into())
            }

            fn unwrap(&self, _: &[u8]) -> Result<Vec<u8>, String> {
                Err("no DPAPI".into())
            }
        }
        let before = std::fs::read(store.path()).unwrap();
        assert_eq!(store.save(&Refuses, 1, &key, &tokens(10), state(10)).unwrap_err(), "no DPAPI");
        assert_eq!(std::fs::read(store.path()).unwrap(), before);
        assert!(store.delete() && !store.exists() && !store.delete());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A start deletes a file that does not open, whatever is wrong with it (a byte changed, another build's or placement's, another layout's, another
    /// user's) and takes the tokens of one that does, leaving it; a request that finds the state damaged deletes the file too; and a run with the option
    /// off deletes the file at its start, where a run that keeps leaves it.
    #[test]
    fn a_file_that_does_not_open_is_deleted_and_a_sound_one_stays() {
        let dir = directory("delete");
        let store = Store::new(&dir, "Q");
        let key = key_of("a");
        let kept = tokens(100);
        let write = || {
            store.save(&Flip, 1, &key, &kept, state(4_000)).unwrap();
        };
        write();
        assert_eq!(store.tokens_or_delete(&Flip, &key, 1), Some(kept.clone()));
        assert!(store.exists(), "a sound file stays, for a request that begins with its tokens");
        assert_eq!(store.tokens_or_delete(&Flip, &key_of("another build"), 1), None);
        assert!(!store.exists(), "another build's or placement's file is deleted");
        write();
        assert_eq!(store.tokens_or_delete(&Flip, &key, 2), None);
        assert!(!store.exists(), "another layout's file is deleted");
        write();
        assert_eq!(store.tokens_or_delete(&Nobody, &key, 1), None);
        assert!(!store.exists(), "another user's file is deleted");
        // a byte changed in the tokens' part, and in the state's (which only a request reads)
        write();
        let good = std::fs::read(store.path()).unwrap();
        let mut bad = good.clone();
        bad[FIXED + 32 + 2 * NONCE + 10] ^= 1;
        std::fs::write(store.path(), &bad).unwrap();
        assert_eq!(store.tokens_or_delete(&Flip, &key, 1), None);
        assert!(!store.exists(), "a damaged file is deleted");
        let mut bad = good.clone();
        let at = bad.len() - 100;
        bad[at] ^= 1;
        std::fs::write(store.path(), &bad).unwrap();
        assert_eq!(store.tokens_or_delete(&Flip, &key, 1), Some(kept.clone()), "the tokens open: the state is not read for them");
        assert!(store.state_or_delete(&Flip, &key, 1).is_err());
        assert!(!store.exists(), "a state found damaged deletes the file");
        assert!(store.state_or_delete(&Flip, &key, 1).unwrap_err().contains("gone"));
        // the option at the start of a run
        write();
        assert!(!store.begin(true) && store.exists(), "a run that keeps leaves the file");
        assert!(store.begin(false), "the option off: the file was there and is gone");
        assert!(!store.exists() && !store.begin(false));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A part a killed write left is removed once it is old, and not before: it may be a write in progress.
    #[test]
    fn a_stale_part_is_removed_and_a_fresh_one_is_not() {
        let dir = directory("part");
        let store = Store::new(&dir, "Q");
        let part = PathBuf::from(format!("{}.part", store.path().display()));
        std::fs::write(&part, b"x").unwrap();
        store.remove_stale_part();
        assert!(part.exists(), "a fresh part may be a write in progress");
        std::fs::OpenOptions::new().write(true).open(&part).unwrap().set_modified(SystemTime::now() - Duration::from_secs(3600)).unwrap();
        store.remove_stale_part();
        assert!(!part.exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// DPAPI seals a whole file for this user and opens it again.
    #[test]
    fn a_file_sealed_with_dpapi_opens() {
        let dir = directory("dpapi");
        let store = Store::new(&dir, "Q");
        let key = key_of("d");
        let (kept, plain) = (tokens(4_000), state(300_000));
        store.save(&crate::prefix_store::Dpapi, 1, &key, &kept, plain.clone()).unwrap();
        assert_eq!(store.open_tokens(&crate::prefix_store::Dpapi, &key, 1).unwrap(), Some(kept));
        assert_eq!(store.open_state(&crate::prefix_store::Dpapi, &key, 1).unwrap(), plain);
        let _ = std::fs::remove_dir_all(dir);
    }
}
