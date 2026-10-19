"""Parallel authentication must retain verified descriptors and close every failed batch."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from straydog import digest as straydog_digest
from straydog.weights import _open_authenticated


def _assets(root):
    entries = []
    for name, data in [("first", b"first shard"), ("second", b"second shard")]:
        (root / name).write_bytes(data)
        entries.append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return entries


def test_authentication_hashes_concurrently_and_returns_the_same_open_handles(tmp_path, monkeypatch):
    entries = _assets(tmp_path)
    barrier, original, observed, counted = threading.Barrier(2), straydog_digest.file_sha256, [], []

    def digest(file, progress=None):
        observed.append(file)
        barrier.wait(timeout=5)
        return original(file, progress)

    monkeypatch.setattr(straydog_digest, "file_sha256", digest)
    authenticated = _open_authenticated(tmp_path, entries, counted.append)
    try:
        assert {id(file) for _asset, file, _stat in authenticated} == {id(file) for file in observed}
        assert sum(counted) == sum(asset["bytes"] for asset in entries)   # progress covers every hashed byte
        for asset, file, stat in authenticated:
            file.seek(0)
            assert hashlib.sha256(file.read()).hexdigest() == asset["sha256"]
            assert stat.st_size == asset["bytes"]
    finally:
        for _asset, file, _stat in authenticated:
            file.close()


def test_file_digest_spans_chunks_and_reads_ahead_in_small_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(straydog_digest, "HASH_CHUNK", 4096)
    monkeypatch.setattr(straydog_digest, "HASH_AHEAD", 8192)
    monkeypatch.setattr(straydog_digest, "HINT_STEP", 2048)
    data = bytes(range(256)) * 160 + b"tail"   # several chunks and a partial one
    path = tmp_path / "shard"
    path.write_bytes(data)
    hints = []
    if hasattr(straydog_digest.os, "posix_fadvise"):
        original = straydog_digest.os.posix_fadvise

        def advise(fd, offset, length, advice):
            hints.append((offset, length, advice))
            return original(fd, offset, length, advice)

        monkeypatch.setattr(straydog_digest.os, "posix_fadvise", advise)
    with path.open("rb") as file:
        file.read(10)   # the digest starts at the beginning wherever the handle is
        assert straydog_digest.file_sha256(file) == hashlib.sha256(data).hexdigest()
    if hasattr(straydog_digest.os, "posix_fadvise"):
        # One small WILLNEED per step (the kernel reads at most one device request per call), in order.
        assert hints and hints == [(offset, 2048, straydog_digest.os.POSIX_FADV_WILLNEED)
                                   for offset in range(0, len(hints) * 2048, 2048)]
        assert hints[-1][0] < len(data)


@pytest.mark.parametrize("failure", ["size", "digest", "duplicate", "escape"])
def test_failed_asset_batches_close_all_opened_files(tmp_path, monkeypatch, failure):
    entries, opened, original = _assets(tmp_path), [], Path.open
    if failure == "size":
        entries[1]["bytes"] += 1
    elif failure == "digest":
        entries[1]["sha256"] = "0" * 64
    elif failure == "duplicate":
        entries[1]["name"] = entries[0]["name"]
    else:
        entries[1]["name"] = "../outside"

    def tracked(path, *args, **kwargs):
        file = original(path, *args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(Path, "open", tracked)
    with pytest.raises(ValueError, match="Straydog asset"):
        _open_authenticated(tmp_path, entries)
    assert opened and all(file.closed for file in opened)
