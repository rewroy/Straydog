"""The minimal environment a contained child starts with."""

from __future__ import annotations

import os
import re
import sys

_SAFE_BASE_VARS = frozenset({"PATH", "HOME", "USER", "LANG", "TZ", "TERM"})
# Windows needs more: without SystemRoot the C runtime and system DLLs fail to load,
# without PATHEXT executables do not resolve, and TEMP/TMP are used everywhere.
# Names are matched case-insensitively there.
_SAFE_WINDOWS_BASE_VARS = frozenset(name.lower() for name in (
    "PATH", "PATHEXT", "SystemRoot", "SystemDrive", "windir", "ComSpec", "TEMP", "TMP", "UserProfile",
    "HomeDrive", "HomePath", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "OS",
    "LOCALAPPDATA", "APPDATA", "LANG",
))
_DANGEROUS = re.compile(r"(?i)(proxy|token|secret|key|password|credential|auth|api_key)")


def sanitized_subprocess_env(allowed_env: dict[str, str] | None = None) -> dict[str, str]:
    """PATH, HOME and a few more from this process, plus ``allowed_env``.

    Proxy, token, secret, key and credential variables are never inherited.
    """
    if sys.platform == "win32":
        env = {k: v for k, v in os.environ.items() if k.lower() in _SAFE_WINDOWS_BASE_VARS and not _DANGEROUS.search(k)}
        env.setdefault("SystemRoot", r"C:\Windows")
        env.setdefault("PATHEXT", ".COM;.EXE;.BAT;.CMD;.PS1")
    else:
        env = {k: v for k, v in os.environ.items() if k in _SAFE_BASE_VARS}
    return env | dict(allowed_env or {})
