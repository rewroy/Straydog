//! A system block's state, sealed on the drive for the next start (Windows): the format of the files, how a key is made and a store of files
//! kept.
//!
//! A file holds one state under a random 256-bit key. That key is wrapped with DPAPI for the user (`CryptProtectData`, no UI) and stored in the
//! file's header; the state is sealed with AES-256-GCM under it and a random 96-bit nonce, with the header as the associated data, so a file
//! opens only as the user who wrote it, and only if not one byte of it, header or body, changed. Nothing in plaintext is ever written, not even
//! a part of a file: the ciphertext goes to a `.part` file, which is synced and then renamed.
//!
//! ```text
//! 0    8   "STRAYPFX"
//! 8    4   FILE_VERSION
//! 12   4   the engine's layout version of the state inside (straydog_engine::prefix::VERSION)
//! 16   32  the key's prefix: everything it is made of but the tokens
//! 48   32  the digest of the tokens the state is held at
//! 80   32  the key: SHA-256 of the prefix and the tokens' digest
//! 112  8   the state's bytes (before sealing)
//! 120  4   the wrapped key's bytes, W
//! 124  W   the wrapped key (DPAPI)
//! ..   12  the nonce
//! ..       the state sealed, and the 16-byte tag after it
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};
use sha2::{Digest, Sha256};

/// The files a bundle keeps at most.
pub const KEEP: usize = 2;
/// The format of the files.
const FILE_VERSION: u32 = 1;
const MAGIC: [u8; 8] = *b"STRAYPFX";
const EXTENSION: &str = "sdprefix";
const TAG: usize = 16;
const NONCE: usize = 12;
const FIXED: usize = 124;
/// The most a wrapped key takes (DPAPI's blob of 32 bytes is about 200).
const WRAPPED_LIMIT: usize = 4096;
/// A `.part` file older than this is what a killed write left.
const STALE_PART: Duration = Duration::from_secs(600);

/// What keeps the file's key: DPAPI, or in a test something that does not need a user.
pub trait Wrap {
    fn wrap(&self, key: &[u8]) -> Result<Vec<u8>, String>;
    fn unwrap(&self, blob: &[u8]) -> Result<Vec<u8>, String>;
}

/// DPAPI for the current user (no entropy of the caller's: any process of the user may unwrap what it wrapped, nobody else).
pub struct Dpapi;

/// What DPAPI mixes into a wrapped key so that another program's blob is not ours.
const ENTROPY: &[u8] = b"straydog prefix state 1";

impl Dpapi {
    fn call(data: &[u8], protect: bool) -> Result<Vec<u8>, String> {
        use windows_sys::Win32::Foundation::LocalFree;
        use windows_sys::Win32::Security::Cryptography::{CryptProtectData, CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB};
        let blob = |bytes: &[u8]| CRYPT_INTEGER_BLOB { cbData: bytes.len() as u32, pbData: bytes.as_ptr() as *mut u8 };
        let (input, entropy) = (blob(data), blob(ENTROPY));
        let mut output = CRYPT_INTEGER_BLOB { cbData: 0, pbData: std::ptr::null_mut() };
        // SAFETY: the blobs point at slices that outlive the call; the output is the system's, freed below with LocalFree.
        let done = unsafe {
            if protect {
                CryptProtectData(&input, std::ptr::null(), &entropy, std::ptr::null(), std::ptr::null(), CRYPTPROTECT_UI_FORBIDDEN, &mut output)
            } else {
                CryptUnprotectData(&input, std::ptr::null_mut(), &entropy, std::ptr::null(), std::ptr::null(), CRYPTPROTECT_UI_FORBIDDEN, &mut output)
            }
        };
        if done == 0 || output.pbData.is_null() {
            return Err(format!("DPAPI {} failed: {}", if protect { "wrapping" } else { "unwrapping" }, std::io::Error::last_os_error()));
        }
        // SAFETY: the system gave `cbData` bytes at `pbData`; they are copied out, cleared and freed.
        let result = unsafe {
            let bytes = std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec();
            std::ptr::write_bytes(output.pbData, 0, output.cbData as usize);
            LocalFree(output.pbData.cast());
            bytes
        };
        Ok(result)
    }
}

impl Wrap for Dpapi {
    fn wrap(&self, key: &[u8]) -> Result<Vec<u8>, String> {
        Dpapi::call(key, true)
    }

    fn unwrap(&self, blob: &[u8]) -> Result<Vec<u8>, String> {
        Dpapi::call(blob, false)
    }
}

/// SHA-256 of `parts` one after the other.
fn hash(parts: &[&[u8]]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update(part);
    }
    hash.finalize().into()
}

/// The key's prefix: SHA-256 of what a state depends on but the tokens it is held at (the bundle, the engine's version and build, every setting
/// that changes the state's layout or its numbers), `description`'s text.
pub fn key_prefix(description: &str) -> [u8; 32] {
    hash(&[b"straydog prefix key 1\n", description.as_bytes()])
}

/// The key of a state: its prefix and the digest of the tokens it is held at.
pub fn key_of(prefix: &[u8; 32], tokens: &[u8; 32]) -> [u8; 32] {
    hash(&[prefix, tokens])
}

/// Hex of the first `bytes` bytes.
fn short(digest: &[u8], bytes: usize) -> String {
    digest.iter().take(bytes).map(|byte| format!("{byte:02x}")).collect()
}

/// A digest from its hex (64 digits).
pub fn digest_from_hex(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 || !text.is_ascii() {
        return None;
    }
    let mut digest = [0u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * index..2 * index + 2], 16).ok()?;
    }
    Some(digest)
}

/// What a file's header says.
struct Header {
    layout: u32,
    prefix: [u8; 32],
    tokens: [u8; 32],
    key: [u8; 32],
    plain: u64,
    wrapped: Vec<u8>,
    nonce: [u8; NONCE],
}

impl Header {
    /// The header as the file holds it; it is the associated data of the sealing.
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(FIXED + self.wrapped.len() + NONCE);
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&FILE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.layout.to_le_bytes());
        bytes.extend_from_slice(&self.prefix);
        bytes.extend_from_slice(&self.tokens);
        bytes.extend_from_slice(&self.key);
        bytes.extend_from_slice(&self.plain.to_le_bytes());
        bytes.extend_from_slice(&(self.wrapped.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.wrapped);
        bytes.extend_from_slice(&self.nonce);
        bytes
    }

    /// The header at the start of `file`, and the bytes it takes; an error when it is not one, or when the file is not exactly as long as the
    /// header says.
    fn parse(file: &[u8]) -> Result<(Header, usize), String> {
        if file.len() < FIXED || file[..8] != MAGIC {
            return Err("it is not a prefix file".into());
        }
        let word = |at: usize| u32::from_le_bytes(file[at..at + 4].try_into().unwrap());
        if word(8) != FILE_VERSION {
            return Err(format!("its format is version {}", word(8)));
        }
        let array = |at: usize| -> [u8; 32] { file[at..at + 32].try_into().unwrap() };
        let plain = u64::from_le_bytes(file[112..120].try_into().unwrap());
        let wrapped = word(120) as usize;
        if wrapped > WRAPPED_LIMIT {
            return Err("its wrapped key is too long".into());
        }
        let start = FIXED + wrapped + NONCE;
        if file.len() < start {
            return Err("its header is short".into());
        }
        let expected = (start as u64).checked_add(plain).and_then(|bytes| bytes.checked_add(TAG as u64));
        if expected != Some(file.len() as u64) {
            return Err(format!("its length is {} where its header says {}", file.len(), expected.map_or("more than a drive holds".to_string(), |bytes| bytes.to_string())));
        }
        let header = Header {
            layout: word(12),
            prefix: array(16),
            tokens: array(48),
            key: array(80),
            plain,
            wrapped: file[FIXED..FIXED + wrapped].to_vec(),
            nonce: file[FIXED + wrapped..start].try_into().unwrap(),
        };
        Ok((header, start))
    }
}

/// A state's file opened: the file's bytes, in which the state is `start..end`.
pub struct Opened {
    buffer: Vec<u8>,
    start: usize,
    end: usize,
    /// The digest of the tokens it is held at.
    pub tokens: [u8; 32],
}

impl Opened {
    pub fn state(&self) -> &[u8] {
        &self.buffer[self.start..self.end]
    }
}

impl std::fmt::Debug for Opened {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "a state of {} bytes", self.end - self.start)
    }
}

/// Seal `state` for the file of `prefix` and `tokens`: the file's bytes, in order, before the tag: its header and the ciphertext (which takes
/// `state`'s place), and the tag.
fn seal(wrap: &dyn Wrap, layout: u32, prefix: &[u8; 32], tokens: &[u8; 32], state: &mut [u8]) -> Result<(Vec<u8>, [u8; TAG]), String> {
    let random = SystemRandom::new();
    let (mut key, mut nonce) = ([0u8; 32], [0u8; NONCE]);
    random.fill(&mut key).map_err(|_| "no random bytes for a key")?;
    random.fill(&mut nonce).map_err(|_| "no random bytes for a nonce")?;
    let wrapped = wrap.wrap(&key);
    let header = wrapped.map(|wrapped| Header { layout, prefix: *prefix, tokens: *tokens, key: key_of(prefix, tokens), plain: state.len() as u64, wrapped, nonce });
    let sealed = header.and_then(|header| {
        let bytes = header.bytes();
        let unbound = UnboundKey::new(&AES_256_GCM, &key).map_err(|_| "no AES-256-GCM key".to_string())?;
        let tag = LessSafeKey::new(unbound)
            .seal_in_place_separate_tag(Nonce::assume_unique_for_key(nonce), Aad::from(&bytes), state)
            .map_err(|_| "sealing failed".to_string())?;
        Ok((bytes, tag.as_ref().try_into().expect("a 16-byte tag")))
    });
    // (the key is in the header, wrapped, and nowhere else)
    key.fill(0);
    sealed
}

/// Open the file at `path`: its header must be one this engine's `layout` and `prefix` made, its key one the prefix and the tokens give, its
/// wrapped key must unwrap for this user, and the state must open (every byte of the file is authenticated). An error says what was wrong.
pub fn open(path: &Path, wrap: &dyn Wrap, prefix: &[u8; 32], layout: u32) -> Result<Opened, String> {
    let mut file = std::fs::read(path).map_err(|error| error.to_string())?;
    let (header, start) = Header::parse(&file)?;
    if header.layout != layout {
        return Err(format!("its state is of layout {}, this engine's is {layout}", header.layout));
    }
    if header.prefix != *prefix {
        return Err("it is another engine's, build's or settings'".into());
    }
    if header.key != key_of(&header.prefix, &header.tokens) {
        return Err("its key is not its prefix and tokens'".into());
    }
    let mut key = wrap.unwrap(&header.wrapped)?;
    let opened = (|| {
        let unbound = UnboundKey::new(&AES_256_GCM, &key).map_err(|_| "its wrapped key is not an AES-256-GCM key".to_string())?;
        let aad = file[..start].to_vec();
        let (_, sealed) = file.split_at_mut(start);
        LessSafeKey::new(unbound)
            .open_in_place(Nonce::assume_unique_for_key(header.nonce), Aad::from(&aad), sealed)
            .map(|plain| plain.len())
            .map_err(|_| "it does not open: a byte of it is not as it was written".to_string())
    })();
    key.fill(0);
    let plain = opened?;
    if plain as u64 != header.plain {
        return Err("its state is not the length its header says".into());
    }
    Ok(Opened { buffer: file, start, end: start + plain, tokens: header.tokens })
}

/// A file of the store, as its name says.
#[derive(Debug, Clone)]
pub struct Entry {
    pub path: PathBuf,
    build: String,
    prefix: String,
    pub modified: SystemTime,
}

/// The files of one bundle's states, in a directory: `<bundle>.<build>.<prefix>.<key>.sdprefix`, the last three the first 8 bytes of
/// their hashes in hex.
pub struct Store {
    directory: PathBuf,
    bundle: String,
    build: String,
}

impl Store {
    /// The store of `bundle` (a label, made fit for a file name) in `directory`, for the engine build `build` (a hash of the executable).
    pub fn new(directory: PathBuf, bundle: &str, build: &[u8; 32]) -> Store {
        let bundle = bundle.chars().map(|letter| if letter.is_ascii_alphanumeric() || matches!(letter, '_' | '-') { letter } else { '_' }).collect();
        Store { directory, bundle, build: short(build, 8) }
    }

    fn name(&self, prefix: &[u8; 32], key: &[u8; 32]) -> String {
        format!("{}.{}.{}.{}.{EXTENSION}", self.bundle, self.build, short(prefix, 8), short(key, 8))
    }

    /// Whether a file holds the state of `key` (under `prefix`).
    pub fn has(&self, prefix: &[u8; 32], key: &[u8; 32]) -> bool {
        self.directory.join(self.name(prefix, key)).is_file()
    }

    /// This bundle's files, the build and prefix their names carry, newest use first.
    fn entries(&self) -> Vec<Entry> {
        let mut entries: Vec<Entry> = std::fs::read_dir(&self.directory).into_iter().flatten().flatten().filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let stem = name.strip_suffix(&format!(".{EXTENSION}"))?;
            let mut parts = stem.rsplitn(4, '.');
            let (_key, prefix, build, bundle) = (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
            if bundle != self.bundle || build.len() != 16 || prefix.len() != 16 {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some(Entry { path: entry.path(), build: build.to_string(), prefix: prefix.to_string(), modified })
        }).collect();
        entries.sort_by(|one, other| other.modified.cmp(&one.modified));
        entries
    }

    /// The files this build made under `prefix`, most recently used first: what a start may be handed.
    pub fn candidates(&self, prefix: &[u8; 32]) -> Vec<Entry> {
        let wanted = short(prefix, 8);
        self.entries().into_iter().filter(|entry| entry.build == self.build && entry.prefix == wanted).collect()
    }

    /// The newest file this build made under `prefix` that opens, taken for use (it is the newest from now on). A file that does not open, for
    /// whatever is wrong with it, is deleted, with one line in the log, and the next is tried: the start goes on as without it.
    pub fn take_newest(&self, wrap: &dyn Wrap, prefix: &[u8; 32], layout: u32) -> Option<Opened> {
        for entry in self.candidates(prefix) {
            match open(&entry.path, wrap, prefix, layout) {
                Ok(opened) => {
                    self.touch(&entry.path);
                    return Some(opened);
                }
                Err(reason) => {
                    let _ = std::fs::remove_file(&entry.path);
                    tracing::warn!("Straydog prefix: deleted {} ({reason}); this start goes on without it", entry.path.display());
                }
            }
        }
        None
    }

    /// Delete the files of other builds of this bundle (no start of this one can match them), the least recently used beyond [`KEEP`], never
    /// `keep`, and the parts a killed write left. The paths deleted.
    pub fn prune(&self, keep: Option<&Path>) -> Vec<PathBuf> {
        let mut deleted = Vec::new();
        let mut remove = |path: &Path| {
            if std::fs::remove_file(path).is_ok() {
                deleted.push(path.to_path_buf());
            }
        };
        let mut current = Vec::new();
        for entry in self.entries() {
            if entry.build != self.build {
                remove(&entry.path);
            } else {
                current.push(entry);
            }
        }
        // (newest first: the file just written takes one place of KEEP, and of the others the newest take the rest)
        let protected = current.iter().filter(|entry| Some(entry.path.as_path()) == keep).count();
        let mut room = KEEP.saturating_sub(protected);
        for entry in current {
            if Some(entry.path.as_path()) == keep {
                continue;
            }
            if room > 0 {
                room -= 1;
            } else {
                remove(&entry.path);
            }
        }
        let parts = std::fs::read_dir(&self.directory).into_iter().flatten().flatten().filter(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.starts_with(&format!("{}.", self.bundle)) && name.ends_with(&format!(".{EXTENSION}.part"))
                && entry.metadata().ok().and_then(|meta| meta.modified().ok()).and_then(|time| time.elapsed().ok()).is_some_and(|age| age > STALE_PART)
        });
        for part in parts {
            remove(&part.path());
        }
        deleted
    }

    /// Note a use of the file: it is the newest now.
    pub fn touch(&self, path: &Path) {
        let _ = std::fs::OpenOptions::new().write(true).open(path).and_then(|file| file.set_modified(SystemTime::now()));
    }

    /// Seal `state` (the bytes of the engine's state of layout `layout`, held at the tokens whose digest is `tokens`) under `prefix` and write it:
    /// to a `.part` file, synced, then renamed, so no reader sees less than all of it and nothing in plaintext is written. The path, and the
    /// bytes of the file.
    pub fn save(&self, wrap: &dyn Wrap, layout: u32, prefix: &[u8; 32], tokens: &[u8; 32], mut state: Vec<u8>) -> Result<(PathBuf, u64), String> {
        let (header, tag) = seal(wrap, layout, prefix, tokens, &mut state)?;
        let path = self.directory.join(self.name(prefix, &key_of(prefix, tokens)));
        let part = PathBuf::from(format!("{}.part", path.display()));
        let written = (|| -> std::io::Result<()> {
            std::fs::create_dir_all(&self.directory)?;
            let mut file = std::fs::File::create(&part)?;
            file.write_all(&header)?;
            file.write_all(&state)?;
            file.write_all(&tag)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&part, &path)
        })();
        if let Err(error) = written {
            let _ = std::fs::remove_file(&part);
            return Err(format!("{}: {error}", path.display()));
        }
        Ok((path, (header.len() + state.len() + TAG) as u64))
    }
}

/// A hash of the running executable, which is what an engine's build is: states of another build are never read.
pub fn build_id() -> Result<[u8; 32], String> {
    use std::io::Read;
    let path = std::env::current_exe().map_err(|error| error.to_string())?;
    let mut file = std::fs::File::open(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut chunk = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut chunk).map_err(|error| error.to_string())?;
        if read == 0 {
            return Ok(hash.finalize().into());
        }
        hash.update(&chunk[..read]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn directory(name: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!("straydog-prefix-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn state(length: usize) -> Vec<u8> {
        (0..length).map(|index| (index * 31 % 251) as u8).collect()
    }

    fn keys(text: &str) -> ([u8; 32], [u8; 32]) {
        (key_prefix(text), hash(&[text.as_bytes(), b" tokens"]))
    }

    /// A state sealed comes back as it was; the file is as long as its header says, holds none of the state's bytes in the clear and none of the key.
    #[test]
    fn a_sealed_state_opens_as_it_was() {
        let dir = directory("round");
        let store = Store::new(dir.clone(), "Qwen3.8-X", &[1; 32]);
        let (prefix, tokens) = keys("a");
        let plain = state(100_000);
        let (path, bytes) = store.save(&Flip, 1, &prefix, &tokens, plain.clone()).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes);
        assert!(store.has(&prefix, &key_of(&prefix, &tokens)));
        let file = std::fs::read(&path).unwrap();
        assert!(!file.windows(64).any(|window| plain.windows(64).any(|clear| clear == window)), "a run of the state is in the file");
        let opened = open(&path, &Flip, &prefix, 1).unwrap();
        assert_eq!((opened.state(), opened.tokens), (&plain[..], tokens));
        assert!(std::fs::read_dir(&dir).unwrap().flatten().all(|entry| !entry.file_name().to_string_lossy().ends_with(".part")), "no part is left");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A byte flipped anywhere in the file, the header's, the wrapped key's, the nonce's, the state's or the tag's, is refused, and so is a file cut
    /// short or one with a byte more.
    #[test]
    fn a_flipped_byte_anywhere_is_refused() {
        let dir = directory("flip");
        let store = Store::new(dir.clone(), "Q", &[1; 32]);
        let (prefix, tokens) = keys("a");
        let (path, _) = store.save(&Flip, 1, &prefix, &tokens, state(5_000)).unwrap();
        let good = std::fs::read(&path).unwrap();
        assert!(open(&path, &Flip, &prefix, 1).is_ok());
        let header = FIXED + 32 + NONCE;
        // (every byte of the header and the tag, and every 97th of the state: each a file of its own)
        let places: Vec<usize> = (0..header).chain((header..good.len() - TAG).step_by(97)).chain(good.len() - TAG..good.len()).collect();
        for place in places {
            let mut bad = good.clone();
            bad[place] ^= 0x01;
            std::fs::write(&path, &bad).unwrap();
            assert!(open(&path, &Flip, &prefix, 1).is_err(), "byte {place} of {} flipped and the file opened", good.len());
        }
        std::fs::write(&path, &good[..good.len() - 1]).unwrap();
        assert!(open(&path, &Flip, &prefix, 1).unwrap_err().contains("length"));
        let mut longer = good.clone();
        longer.push(0);
        std::fs::write(&path, &longer).unwrap();
        assert!(open(&path, &Flip, &prefix, 1).unwrap_err().contains("length"));
        std::fs::write(&path, &good).unwrap();
        assert!(open(&path, &Flip, &prefix, 1).is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A start takes the newest file that opens; one that does not (a byte flipped, the wrong user's key) is deleted, and the next is tried.
    #[test]
    fn a_file_that_does_not_open_is_deleted_and_the_next_is_taken() {
        let dir = directory("take");
        let store = Store::new(dir.clone(), "Q", &[1; 32]);
        let (prefix, _) = keys("p");
        let (older, newer) = ([1u8; 32], [2u8; 32]);
        let first = store.save(&Flip, 1, &prefix, &older, state(2_000)).unwrap().0;
        let second = store.save(&Flip, 1, &prefix, &newer, state(3_000)).unwrap().0;
        for (path, seconds) in [(&first, 200), (&second, 100)] {
            std::fs::OpenOptions::new().write(true).open(path).unwrap().set_modified(SystemTime::now() - Duration::from_secs(seconds)).unwrap();
        }
        // the newest is damaged in its body: it is deleted, and the older one is taken
        let mut bad = std::fs::read(&second).unwrap();
        let middle = bad.len() / 2;
        bad[middle] ^= 0x40;
        std::fs::write(&second, &bad).unwrap();
        std::fs::OpenOptions::new().write(true).open(&second).unwrap().set_modified(SystemTime::now() - Duration::from_secs(100)).unwrap();
        let taken = store.take_newest(&Flip, &prefix, 1).unwrap();
        assert_eq!((taken.state().len(), taken.tokens), (2_000, older));
        assert!(!second.exists() && first.exists(), "the damaged file is gone and the sound one stays");
        // (a use makes it the newest)
        assert_eq!(store.candidates(&prefix)[0].path, first);
        // a file the wrong user wrapped: a key that does not unwrap deletes it too, and nothing is left to take
        struct Nobody;
        impl Wrap for Nobody {
            fn wrap(&self, key: &[u8]) -> Result<Vec<u8>, String> {
                Ok(key.to_vec())
            }

            fn unwrap(&self, _: &[u8]) -> Result<Vec<u8>, String> {
                Err("not this user's".into())
            }
        }
        assert!(store.take_newest(&Nobody, &prefix, 1).is_none());
        assert!(!first.exists());
        assert!(store.take_newest(&Flip, &prefix, 1).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Another bundle's, build's or settings' key is refused, and so is a file of another layout of the state inside.
    #[test]
    fn another_bundles_or_builds_key_is_refused() {
        let dir = directory("other");
        let store = Store::new(dir.clone(), "Qwen3.8-X", &[1; 32]);
        let (prefix, tokens) = keys("bundle A, build 1");
        let (path, _) = store.save(&Flip, 1, &prefix, &tokens, state(1_000)).unwrap();
        assert!(open(&path, &Flip, &prefix, 1).is_ok());
        let other = key_prefix("bundle A, build 2");
        assert!(open(&path, &Flip, &other, 1).unwrap_err().contains("another"));
        assert!(open(&path, &Flip, &prefix, 2).unwrap_err().contains("layout"));
        // a file's name is of its prefix, its key and its build: the store hands out the files of this one alone
        assert_eq!(store.candidates(&prefix).len(), 1);
        assert!(store.candidates(&other).is_empty());
        let rebuilt = Store::new(dir.clone(), "Qwen3.8-X", &[2; 32]);
        assert!(rebuilt.candidates(&prefix).is_empty(), "another build's file is not offered");
        let sibling = Store::new(dir.clone(), "Qwen3.8-Y", &[1; 32]);
        assert!(sibling.candidates(&prefix).is_empty(), "another bundle's file is not offered");
        // a key that was not made of its own prefix and tokens
        let mut forged = std::fs::read(&path).unwrap();
        forged[80] ^= 1;
        assert!(Header::parse(&forged).is_ok());
        std::fs::write(&path, &forged).unwrap();
        assert!(open(&path, &Flip, &prefix, 1).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A bundle keeps two files: a third written deletes the one used longest ago (never the one just written), a file of another build goes at
    /// once, and what is not a file of the store, or of another bundle, stays.
    #[test]
    fn a_bundle_keeps_two_files_and_drops_the_least_recently_used() {
        let dir = directory("prune");
        let store = Store::new(dir.clone(), "Q", &[1; 32]);
        let other_bundle = Store::new(dir.clone(), "R", &[1; 32]);
        let (prefix, _) = keys("p");
        let tokens: Vec<[u8; 32]> = (0u8..4).map(|index| [index; 32]).collect();
        let set_age = |path: &Path, seconds: u64| {
            let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
            file.set_modified(SystemTime::now() - Duration::from_secs(seconds)).unwrap();
        };
        let first = store.save(&Flip, 1, &prefix, &tokens[0], state(10)).unwrap().0;
        set_age(&first, 300);
        let second = store.save(&Flip, 1, &prefix, &tokens[1], state(10)).unwrap().0;
        set_age(&second, 200);
        assert!(store.prune(Some(&second)).is_empty(), "two files fit");
        // (the first is used again: it is now the newest, and the second the oldest)
        store.touch(&first);
        assert_eq!(store.candidates(&prefix)[0].path, first, "a use makes a file the newest");
        // (a file's time is the system's, which moves in steps of milliseconds: the third is written after one)
        std::thread::sleep(Duration::from_millis(100));
        let third = store.save(&Flip, 1, &prefix, &tokens[2], state(10)).unwrap().0;
        let deleted = store.prune(Some(&third));
        assert_eq!(deleted, [second.clone()], "the least recently used goes");
        assert!(first.is_file() && third.is_file() && !second.is_file());
        // another build's file goes whatever its age, another bundle's and a stranger stay
        let old_build = Store::new(dir.clone(), "Q", &[9; 32]);
        let stale = old_build.save(&Flip, 1, &prefix, &tokens[3], state(10)).unwrap().0;
        let theirs = other_bundle.save(&Flip, 1, &prefix, &tokens[3], state(10)).unwrap().0;
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        assert_eq!(store.prune(None), [stale.clone()]);
        assert!(!stale.is_file() && theirs.is_file() && first.is_file() && third.is_file() && dir.join("notes.txt").is_file());
        // (candidates: newest use first)
        let order: Vec<PathBuf> = store.candidates(&prefix).into_iter().map(|entry| entry.path).collect();
        assert_eq!(order.len(), 2);
        assert_eq!(order[0], third, "the one written last is the newest");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A part a killed write left is removed once it is old, and not before: it may be a write in progress.
    #[test]
    fn a_stale_part_is_removed_and_a_fresh_one_is_not() {
        let dir = directory("part");
        let store = Store::new(dir.clone(), "Q", &[1; 32]);
        let (stale, fresh) = (dir.join("Q.aa.bb.cc.sdprefix.part"), dir.join("Q.dd.ee.ff.sdprefix.part"));
        std::fs::write(&stale, b"x").unwrap();
        std::fs::write(&fresh, b"x").unwrap();
        std::fs::OpenOptions::new().write(true).open(&stale).unwrap().set_modified(SystemTime::now() - Duration::from_secs(3600)).unwrap();
        assert_eq!(store.prune(None), [stale.clone()]);
        assert!(!stale.exists() && fresh.exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// DPAPI wraps a key for this user and gives it back; what it did not wrap is refused.
    #[test]
    fn dpapi_wraps_and_unwraps_for_this_user() {
        let key = state(32);
        let wrapped = Dpapi.wrap(&key).unwrap();
        assert_ne!(wrapped, key);
        assert!(wrapped.len() > 32);
        assert_eq!(Dpapi.unwrap(&wrapped).unwrap(), key);
        let mut bad = wrapped.clone();
        bad[wrapped.len() / 2] ^= 1;
        assert!(Dpapi.unwrap(&bad).is_err());
        assert!(Dpapi.unwrap(&key).is_err());
        // and a whole file under it
        let dir = directory("dpapi");
        let store = Store::new(dir.clone(), "Q", &[1; 32]);
        let (prefix, tokens) = keys("d");
        let plain = state(300_000);
        let (path, _) = store.save(&Dpapi, 1, &prefix, &tokens, plain.clone()).unwrap();
        assert_eq!(open(&path, &Dpapi, &prefix, 1).unwrap().state(), &plain[..]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn digests_read_from_hex() {
        let digest = hash(&[b"x"]);
        let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(digest_from_hex(&hex), Some(digest));
        assert_eq!(digest_from_hex(&hex[..62]), None);
        assert_eq!(digest_from_hex(&hex.replace('0', "g").replace('a', "g")), None);
        assert_eq!(digest_from_hex(&"é".repeat(32)), None);
    }

    #[test]
    fn a_build_is_the_executables_hash() {
        assert_eq!(build_id().unwrap(), build_id().unwrap());
    }
}
