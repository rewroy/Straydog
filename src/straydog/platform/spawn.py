"""Spawn options and console output decoding that work on every platform."""

from __future__ import annotations

import subprocess
import sys
from typing import Any


def no_window_kwargs() -> dict[str, Any]:
    """Keep a console child from opening a visible window on Windows (no-op elsewhere).

    A console program started from a process without a visible console gets a new,
    visible console unless the spawn passes CREATE_NO_WINDOW; redirecting its output
    does not prevent it. Its own children inherit the hidden console.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


# Console tools on Windows write in the OEM code page (cp932 on Japanese Windows), Python
# pipes in the ANSI one; modern tools in UTF-8. UTF-8 is tried strictly first (localized
# text is essentially never valid UTF-8), then the Windows code pages, then lossy UTF-8.
_CONSOLE_FALLBACK_CODECS: tuple[str, ...] = ("oem", "mbcs")


def decode_console_output(data: bytes, *, fallback_codecs: tuple[str, ...] | None = None) -> str:
    """Decode captured subprocess output without mangling localized text."""
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for codec in _CONSOLE_FALLBACK_CODECS if fallback_codecs is None else fallback_codecs:
        try:
            return data.decode(codec)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")
