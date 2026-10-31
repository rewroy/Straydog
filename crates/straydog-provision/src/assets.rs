//! A bundle's pinned files: authenticated where they are, reused from a sibling bundle
//! directory, or downloaded (resuming) and authenticated before they are published.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use sha2::{Digest, Sha256};
use straydog_engine::bundle::{self, Asset};
use straydog_engine::digest;
use ureq::Agent;

use crate::{at, invalid, partial_path, Progress};

/// A model download fails after this long without progress (connecting, each read).
pub const TIMEOUT: Duration = Duration::from_secs(180);
const BLOCK: usize = 1 << 20;

/// Every pinned file of `quantization` with the URL it is fetched from: the shards, then
/// the resources.
pub fn entries(quantization: &str) -> Result<Vec<(Asset, String)>, String> {
    let sources = if bundle::is_on_demand(quantization) { bundle::on_demand_sources(quantization)? } else { bundle::sources(quantization)? };
    Ok(sources
        .into_iter()
        .map(|source| {
            let url = format!("https://huggingface.co/{}/resolve/{}/{}", source.repository, source.revision, source.path);
            (source.asset, url)
        })
        .collect())
}

/// Fetch the files of `entries` into `root`, or with `offline` check the files there.
pub fn fetch_with(agent: &Agent, root: &Path, entries: &[(Asset, String)], offline: bool, authenticate_existing: bool,
                  progress: Progress) -> io::Result<()> {
    if offline {
        let files: Vec<(PathBuf, &Asset)> = entries.iter().map(|(asset, _)| (root.join(&asset.name), asset)).collect();
        return check_present(&files, authenticate_existing, progress);
    }
    fs::create_dir_all(root).map_err(at(root))?;
    // Absolute and resolved: the sibling bundle directories are found from here.
    let root = fs::canonicalize(root).map_err(at(root))?;
    let present: Vec<(PathBuf, &Asset)> = entries
        .iter()
        .map(|(asset, _)| (root.join(&asset.name), asset))
        .filter(|(path, _)| path.exists())
        .collect();
    check_present(&present, authenticate_existing, progress)?;
    for (asset, url) in entries {
        let target = root.join(&asset.name);
        let gguf = target.extension().is_some_and(|suffix| suffix == "gguf");
        if target.exists() || (gguf && reuse(&target, asset, progress)?) {
            continue;
        }
        download(agent, &target, asset, url, progress)?;
    }
    Ok(())
}

/// Size, then SHA-256, of a pinned file, the hashing shown as `label`.
pub fn verify_file(path: &Path, asset: &Asset, label: &str, progress: Progress) -> io::Result<()> {
    if fs::metadata(path).map_err(at(path))?.len() != asset.bytes {
        return Err(invalid(format!("size mismatch: {}", asset.name)));
    }
    let file = File::open(path).map_err(at(path))?;
    let done = AtomicU64::new(0);
    progress(label, 0, asset.bytes);
    let count = |bytes: u64| progress(label, done.fetch_add(bytes, Ordering::Relaxed) + bytes, asset.bytes);
    if digest::file_sha256(&file, &count)? != asset.sha256 {
        return Err(invalid(format!("SHA256 mismatch: {}", asset.name)));
    }
    Ok(())
}

/// Sizes of present pinned files and, when `authenticate`, their SHA-256 (two files at a time).
fn check_present(files: &[(PathBuf, &Asset)], authenticate: bool, progress: Progress) -> io::Result<()> {
    for (path, asset) in files {
        if fs::metadata(path).map_err(at(path))?.len() != asset.bytes {
            return Err(invalid(format!("size mismatch: {}", asset.name)));
        }
    }
    if authenticate && !files.is_empty() {
        let opened = files.iter().map(|(path, _)| File::open(path).map_err(at(path))).collect::<io::Result<Vec<_>>>()?;
        let label = format!("authenticating {} present file(s)", files.len());
        let total = files.iter().map(|(_, asset)| asset.bytes).sum();
        let done = AtomicU64::new(0);
        progress(&label, 0, total);
        let count = |bytes: u64| progress(&label, done.fetch_add(bytes, Ordering::Relaxed) + bytes, total);
        let digests = digest::files_sha256(&opened.iter().collect::<Vec<_>>(), &count)?;
        for ((_, asset), digest) in files.iter().zip(digests) {
            if digest != asset.sha256 {
                return Err(invalid(format!("SHA256 mismatch: {}", asset.name)));
            }
        }
    }
    for (_, asset) in files {
        println!("{} {}", if authenticate { "verified" } else { "present" }, asset.name);
    }
    Ok(())
}

/// Link (or copy) an identical pinned file from a sibling bundle directory instead of downloading.
///
/// The bundles share their second shard byte for byte; a candidate is used only after its size
/// and SHA-256 match the pin of `target`.
fn reuse(target: &Path, asset: &Asset, progress: Progress) -> io::Result<bool> {
    let Some(directory) = target.parent() else { return Ok(false) };
    let Some(parent) = directory.parent() else { return Ok(false) };
    // Unreadable directories hold no candidates.
    let mut candidates = Vec::new();
    for sibling in fs::read_dir(parent).into_iter().flatten().flatten() {
        let sibling = sibling.path();
        if sibling.is_dir() {
            for entry in fs::read_dir(&sibling).into_iter().flatten().flatten() {
                if entry.file_name().to_string_lossy().ends_with(".gguf") {
                    candidates.push(entry.path());
                }
            }
        }
    }
    candidates.sort();
    for candidate in candidates {
        if candidate.parent() == Some(directory)
            || fs::symlink_metadata(&candidate).map_err(at(&candidate))?.file_type().is_symlink()
            || fs::metadata(&candidate).map_err(at(&candidate))?.len() != asset.bytes
        {
            continue;
        }
        let name = |path: Option<&Path>| path.and_then(Path::file_name).unwrap_or_default().to_string_lossy().into_owned();
        let label = format!("checking {}/{} for reuse", name(candidate.parent()), name(Some(&candidate)));
        match verify_file(&candidate, asset, &label, progress) {
            Err(error) if error.kind() == io::ErrorKind::InvalidData => continue,
            checked => checked?,
        }
        let partial = partial_path(target);
        match fs::remove_file(&partial) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(at(&partial)(error)),
            _ => {}
        }
        if fs::hard_link(&candidate, &partial).is_err() {
            fs::copy(&candidate, &partial).map_err(at(&partial))?;
        }
        fs::rename(&partial, target).map_err(at(target))?;
        println!("reused {} as {}", candidate.display(), asset.name);
        return Ok(true);
    }
    Ok(false)
}

/// Fetch (or resume) one pinned file, hashing the bytes as they arrive; publish it only if they match.
fn download(agent: &Agent, target: &Path, asset: &Asset, url: &str, progress: Progress) -> io::Result<()> {
    let partial = partial_path(target);
    let partial_name = partial.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let mut offset = match fs::metadata(&partial) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
        Err(error) => return Err(at(&partial)(error)),
    };
    if offset > asset.bytes {
        return Err(invalid(format!("oversized partial asset: {}", partial.display())));
    }
    let mut digest = Sha256::new();
    if offset > 0 {
        hash_partial(&partial, &mut digest, &format!("checking the partial {}", asset.name), offset, progress)?;
    }
    if offset < asset.bytes {
        let mut request = agent.get(url);
        if offset > 0 {
            request = request.header("Range", format!("bytes={offset}-"));
        }
        let response = request.call().map_err(|error| match error {
            // The pinned URLs are HTTPS: only a redirect leads elsewhere.
            ureq::Error::RequireHttpsOnly(_) => invalid("non-HTTPS model redirect"),
            error => error.into_io(),
        })?;
        let status = response.status().as_u16();
        if offset > 0 && status == 200 {
            // The server ignored the range: start over.
            (offset, digest) = (0, Sha256::new());
        }
        if status != 200 && status != 206 {
            return Err(invalid("unexpected model response status"));
        }
        let range = response.headers().get("Content-Range").and_then(|value| value.to_str().ok()).unwrap_or("");
        if status == 206 && !range.starts_with(&format!("bytes {offset}-")) {
            return Err(invalid("model resume range mismatch"));
        }
        let mut file = if offset > 0 {
            OpenOptions::new().append(true).open(&partial)
        } else {
            File::create(&partial)
        }
        .map_err(at(&partial))?;
        let label = format!("downloading {}", asset.name);
        progress(&label, offset, asset.bytes);
        let (mut body, mut buffer, mut written) = (response.into_body().into_reader(), vec![0u8; BLOCK], offset);
        loop {
            let count = read_body(&mut body, &mut buffer)?;
            if count == 0 {
                break;
            }
            written += count as u64;
            if written > asset.bytes {
                return Err(invalid("model download exceeded pinned size"));
            }
            file.write_all(&buffer[..count])?;
            digest.update(&buffer[..count]);
            progress(&label, written, asset.bytes);
        }
    }
    if fs::metadata(&partial).map_err(at(&partial))?.len() != asset.bytes {
        return Err(invalid(format!("size mismatch: {partial_name} (the download ended early; run again to resume)")));
    }
    if format!("{:x}", digest.finalize()) != asset.sha256 {
        fs::remove_file(&partial).map_err(at(&partial))?;
        return Err(invalid(format!("SHA256 mismatch: {} (the download was removed; run again)", asset.name)));
    }
    fs::rename(&partial, target).map_err(at(target))?;
    println!("verified {}", asset.name);
    Ok(())
}

/// Feed `digest` the partial download, the bytes it resumes from. Read sequentially: this
/// precedes a download that takes far longer.
fn hash_partial(partial: &Path, digest: &mut Sha256, label: &str, length: u64, progress: Progress) -> io::Result<()> {
    let mut file = File::open(partial).map_err(at(partial))?;
    let mut buffer = vec![0u8; digest::HASH_CHUNK];
    let mut done = 0;
    progress(label, 0, length);
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            return Ok(());
        }
        digest.update(&buffer[..count]);
        done += count as u64;
        progress(label, done, length);
    }
}

/// A read of a response body. A connection closed before the declared end of the body reads
/// as its end, as the reference's short read did: the caller's size checks then say what is
/// missing, and a download keeps what arrived for the next run to resume from.
pub fn read_body(body: &mut impl Read, buffer: &mut [u8]) -> io::Result<usize> {
    match body.read(buffer) {
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(0),
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http;
    use crate::mock::{self, Scratch, Server};

    fn pinned(name: &str, data: &[u8]) -> Asset {
        Asset { name: name.into(), bytes: data.len() as u64, sha256: format!("{:x}", Sha256::digest(data)) }
    }

    fn quiet(_stage: &str, _done: u64, _total: u64) {}

    fn fetch(server: &Server, root: &Path, assets: &[Asset], authenticate_existing: bool) -> io::Result<()> {
        let entries: Vec<(Asset, String)> =
            assets.iter().map(|asset| (asset.clone(), format!("{}/resolve/{}", server.url, asset.name))).collect();
        fetch_with(&http::plain(TIMEOUT), root, &entries, false, authenticate_existing, &quiet)
    }

    fn offline(root: &Path, asset: &Asset, authenticate: bool) -> io::Result<()> {
        let entries = [(asset.clone(), "https://unused".to_string())];
        fetch_with(&http::plain(TIMEOUT), root, &entries, true, authenticate, &quiet)
    }

    fn failure(result: io::Result<()>) -> String {
        result.expect_err("a failure").to_string()
    }

    #[test]
    fn pinned_files_come_from_fixed_revisions() {
        for quantization in bundle::QUANTIZATIONS {
            let entries = entries(quantization).unwrap();
            assert_eq!(entries.len(), 5);
            assert!(entries.iter().all(|(_, url)| url.starts_with("https://huggingface.co/") && url.contains("/resolve/")
                && !url.contains("/main/")));
            assert!(entries[0].1.ends_with(&format!("/{quantization}/{}", entries[0].0.name)));
        }
        let entries = entries("Q2_0").unwrap();
        assert_eq!(entries[0].1, "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/\
            ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf");
        assert_eq!(entries[2].1, "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/\
            de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer.json");
        assert!(super::entries("Q4_K_M").unwrap_err().contains("unsupported"));
    }

    #[test]
    fn fetch_resumes_and_authenticates_before_publishing() {
        let data: &'static [u8] = b"complete pinned bytes";
        let asset = pinned("model.gguf", data);
        // The Hub redirects to a CDN, which must get the same range.
        let server = Server::start(move |request| match request.path.as_str() {
            "/resolve/model.gguf" => mock::reply(302, &[("Location", "/cdn/model.gguf".into())], b""),
            _ => mock::ranged(data, request.range.as_deref()),
        });
        let scratch = Scratch::new("resume");
        let root = scratch.path().join("bundle");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model.gguf.part"), &data[..5]).unwrap();
        fetch(&server, &root, std::slice::from_ref(&asset), true).unwrap();
        assert_eq!(fs::read(root.join("model.gguf")).unwrap(), data);
        assert!(!root.join("model.gguf.part").exists());
        assert_eq!(server.ranges(), [Some("bytes=5-".to_string()), Some("bytes=5-".to_string())]);
        fetch(&server, &root, std::slice::from_ref(&asset), true).unwrap(); // already present and verified: no request
        assert_eq!(server.requests().len(), 2);
        fs::write(root.join("model.gguf"), vec![b'x'; data.len()]).unwrap();
        assert!(failure(verify_file(&root.join("model.gguf"), &asset, "authenticating", &quiet)).contains("SHA256"));
    }

    #[test]
    fn fetch_server_ignoring_range_restarts_and_bad_range_fails() {
        let data: &'static [u8] = b"test";
        let asset = pinned("model", data);
        let scratch = Scratch::new("ranges");
        let root = scratch.path().join("bundle");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model.part"), b"te").unwrap();
        let ignoring = Server::start(move |_| mock::reply(200, &[], data));
        fetch(&ignoring, &root, std::slice::from_ref(&asset), true).unwrap();
        assert_eq!(fs::read(root.join("model")).unwrap(), data);
        assert_eq!(ignoring.ranges(), [Some("bytes=2-".to_string())]);
        fs::remove_file(root.join("model")).unwrap();
        fs::write(root.join("model.part"), b"te").unwrap();
        let wrong = Server::start(|_| mock::reply(206, &[("Content-Range", "bytes 0-1/4".into())], b"xx"));
        assert!(failure(fetch(&wrong, &root, std::slice::from_ref(&asset), true)).contains("range"));
        assert!(!root.join("model").exists());
    }

    #[test]
    fn present_files_are_left_to_the_worker_unless_authentication_is_asked() {
        let asset = pinned("model", b"pinned");
        let scratch = Scratch::new("present");
        let root = scratch.path();
        fs::write(root.join("model"), b"broken").unwrap(); // right size, wrong bytes
        offline(root, &asset, false).unwrap();
        let server = Server::start(|_| mock::reply(500, &[], b""));
        fetch(&server, root, std::slice::from_ref(&asset), false).unwrap(); // present: no request either
        assert!(server.requests().is_empty());
        assert!(failure(offline(root, &asset, true)).contains("SHA256 mismatch: model"));
        fs::write(root.join("model"), b"short").unwrap();
        assert!(failure(offline(root, &asset, false)).contains("size mismatch: model"));
        fs::remove_file(root.join("model")).unwrap();
        assert!(failure(offline(root, &asset, false)).contains("model"));
    }

    #[test]
    fn download_is_hashed_as_it_arrives_and_a_corrupt_one_is_removed() {
        let data: &'static [u8] = b"complete pinned bytes";
        let asset = pinned("model.gguf", data);
        let scratch = Scratch::new("corrupt");
        let root = scratch.path().join("bundle"); // its own parent: no sibling bundle to reuse a file from
        let corrupt = Server::start(move |_| mock::reply(200, &[], &vec![b'X'; data.len()]));
        assert!(failure(fetch(&corrupt, &root, std::slice::from_ref(&asset), true)).contains("SHA256 mismatch"));
        assert!(!root.join("model.gguf.part").exists() && !root.join("model.gguf").exists());
        let good = Server::start(move |_| mock::reply(200, &[], data));
        fetch(&good, &root, std::slice::from_ref(&asset), true).unwrap(); // starts over rather than resuming the bad bytes
        assert_eq!([corrupt.ranges(), good.ranges()].concat(), [None, None]);
        assert_eq!(fs::read(root.join("model.gguf")).unwrap(), data);
    }

    #[test]
    fn a_download_that_ends_early_is_kept_and_resumed() {
        let data: &'static [u8] = b"complete pinned bytes";
        let asset = pinned("model.gguf", data);
        let scratch = Scratch::new("early");
        let root = scratch.path().join("bundle");
        // The connection closes after 10 of the 21 bytes the response declares.
        let cut = Server::start(move |_| [mock::head(200, &[], data.len()), data[..10].to_vec()].concat());
        let message = failure(fetch(&cut, &root, std::slice::from_ref(&asset), true));
        assert_eq!(message, "size mismatch: model.gguf.part (the download ended early; run again to resume)");
        assert_eq!(fs::read(root.join("model.gguf.part")).unwrap(), &data[..10]);
        let server = Server::start(move |request| mock::ranged(data, request.range.as_deref()));
        fetch(&server, &root, std::slice::from_ref(&asset), true).unwrap();
        assert_eq!(server.ranges(), [Some("bytes=10-".to_string())]);
        assert_eq!(fs::read(root.join("model.gguf")).unwrap(), data);
    }

    #[test]
    fn downloads_and_partials_never_exceed_the_pinned_size() {
        let data: &'static [u8] = b"pinned";
        let asset = pinned("model", data);
        let scratch = Scratch::new("oversized");
        let root = scratch.path().join("bundle");
        let long = Server::start(|_| mock::reply(200, &[], b"pinned and more"));
        assert_eq!(failure(fetch(&long, &root, std::slice::from_ref(&asset), true)), "model download exceeded pinned size");
        fs::write(root.join("model.part"), b"pinned and more").unwrap();
        assert!(failure(fetch(&long, &root, std::slice::from_ref(&asset), true)).starts_with("oversized partial asset"));
        assert_eq!(long.requests().len(), 1);
        let missing = Server::start(|_| mock::reply(404, &[], b""));
        fs::remove_file(root.join("model.part")).unwrap();
        assert!(failure(fetch(&missing, &root, std::slice::from_ref(&asset), true)).contains("404"));
    }

    #[test]
    fn identical_files_of_sibling_bundles_are_reused_once_authenticated() {
        let shared: &'static [u8] = b"the shared second shard";
        let asset = pinned("shared-00002-of-00002.gguf", shared);
        let resource = pinned("tokenizer.json", b"{}");
        let scratch = Scratch::new("reuse");
        let models = scratch.path();
        for (bundle, bytes) in [("IQ2_XS", &b"the shared second shard"[..]), ("BROKEN", &b"the shared second shaRD"[..])] {
            fs::create_dir_all(models.join(bundle)).unwrap();
            fs::write(models.join(bundle).join(&asset.name), bytes).unwrap();
            fs::write(models.join(bundle).join(&resource.name), b"{}").unwrap();
        }
        let server = Server::start(|request| match request.path.as_str() {
            "/resolve/tokenizer.json" => mock::reply(200, &[], b"{}"),
            _ => mock::reply(500, &[], b""),
        });
        let root = models.join("Q2_0");
        fetch(&server, &root, &[asset.clone(), resource], true).unwrap();
        assert_eq!(fs::read(root.join(&asset.name)).unwrap(), shared);
        // Only GGUF files are reused; the corrupt candidate (checked first) was passed over.
        assert_eq!(server.requests().len(), 1);
        assert_eq!(server.requests()[0].path, "/resolve/tokenizer.json");
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let inode = |path: &Path| fs::metadata(path).unwrap().ino();
            assert_eq!(inode(&root.join(&asset.name)), inode(&models.join("IQ2_XS").join(&asset.name)));
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_symbolic_link_is_never_reused() {
        let shared: &'static [u8] = b"the shared second shard";
        let asset = pinned("shared-00002-of-00002.gguf", shared);
        let scratch = Scratch::new("symlink");
        let models = scratch.path();
        fs::write(models.join("elsewhere"), shared).unwrap();
        fs::create_dir_all(models.join("IQ2_XS")).unwrap();
        std::os::unix::fs::symlink(models.join("elsewhere"), models.join("IQ2_XS").join(&asset.name)).unwrap();
        let server = Server::start(move |_| mock::reply(200, &[], shared));
        fetch(&server, &models.join("Q2_0"), &[asset], true).unwrap();
        assert_eq!(server.requests().len(), 1);
        assert!(!fs::symlink_metadata(models.join("Q2_0/shared-00002-of-00002.gguf")).unwrap().file_type().is_symlink());
    }

    #[test]
    fn plain_http_is_refused() {
        let server = Server::start(|_| mock::reply(200, &[], b"pinned"));
        let scratch = Scratch::new("https");
        let asset = pinned("model", b"pinned");
        let url = format!("{}/model", server.url);
        let message = failure(download(&http::agent(TIMEOUT), &scratch.path().join("model"), &asset, &url, &quiet));
        assert_eq!(message, "non-HTTPS model redirect");
        assert!(server.requests().is_empty());
    }

    #[test]
    fn a_silent_connection_times_out() {
        let server = Server::start(|_| {
            std::thread::sleep(Duration::from_secs(5));
            mock::reply(200, &[], b"pinned")
        });
        let scratch = Scratch::new("timeout");
        let asset = pinned("model", b"pinned");
        let started = std::time::Instant::now();
        let url = format!("{}/model", server.url);
        let message = failure(download(&http::plain(Duration::from_millis(300)), &scratch.path().join("model"), &asset,
                                       &url, &quiet));
        assert!(message.contains("timeout"), "{message}");
        assert!(started.elapsed() < Duration::from_secs(3));
    }
}
