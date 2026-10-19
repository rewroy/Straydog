"""SHA-256 of Straydog's large files at the speed of one core, not of one read-ahead window.

Shared by the worker (which authenticates every file before use) and by provisioning.
Reads only; the files are never written.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

HASH_CHUNK, HASH_AHEAD, HINT_STEP = 8 * 1024**2, 256 * 1024**2, 128 * 1024


def _read_ahead(fd: int, size: int, position: list[int], done: threading.Event) -> None:
    """Request the bytes up to HASH_AHEAD past ``position[0]`` (the hasher's offset) from the device.

    One POSIX_FADV_WILLNEED reads at most the device's largest request (e.g. 256 KiB) and is
    submitted at device speed, so the hints go in small steps on this thread of their own.
    """
    hinted = 0
    try:
        while hinted < size and not done.is_set():
            if hinted >= position[0] + HASH_AHEAD:
                done.wait(0.002)
                continue
            os.posix_fadvise(fd, hinted, HINT_STEP, os.POSIX_FADV_WILLNEED)
            hinted += HINT_STEP
    except OSError:
        logger.debug("Straydog read-ahead hint refused", exc_info=True)


def update_from_file(digest: Any, file, progress: Callable[[int], None] | None = None) -> Any:
    """Feed ``digest`` an open file from its start; where available its next 256 MiB are read ahead.

    A plain sequential read waits on one small kernel read-ahead window at a time (about
    1.2 GB/s from an NVMe that reads 4-5 GB/s with more requests in flight). ``progress``
    receives each chunk's byte count.
    """
    fd = file.fileno()
    buffer = memoryview(bytearray(HASH_CHUNK))
    position, done = [0], threading.Event()
    helper = None
    if hasattr(os, "posix_fadvise"):
        helper = threading.Thread(target=_read_ahead, args=(fd, os.fstat(fd).st_size, position, done),
                                  name="straydog-read-ahead", daemon=True)
        helper.start()
    try:
        file.seek(0)
        while count := file.readinto(buffer):
            digest.update(buffer[:count])
            position[0] += count
            if progress is not None:
                progress(count)
        return digest
    finally:
        done.set()
        if helper is not None:
            helper.join()


def file_sha256(file, progress: Callable[[int], None] | None = None) -> str:
    """Hex SHA-256 of an open file from its start (see update_from_file)."""
    return update_from_file(hashlib.sha256(), file, progress).hexdigest()


def files_sha256(files: list, progress: Callable[[int], None] | None = None) -> list[str]:
    """Hex SHA-256 of each open file, two files at a time (one digest cannot use two cores)."""
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="straydog-auth") as pool:
        return list(pool.map(lambda file: file_sha256(file, progress), files))
