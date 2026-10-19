"""One status line for Straydog's long setup steps: downloads, authentication, conversion, startup."""

from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

_SPINNER = "|/-\\"


def _size(count: float) -> str:
    return f"{count / 1e9:.1f} GB" if count >= 1e9 else f"{count / 1e6:.0f} MB"


def _clock(seconds: float) -> str:
    minutes, second = divmod(int(seconds), 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours}:{minute:02d}:{second:02d}" if hours else f"{minute}:{second:02d}"


class ProgressLine:
    """A step's progress: redrawn in place on a terminal, where the elapsed time keeps moving even
    while nothing reports; elsewhere (logs, pipes) a plain line every 15 seconds and at the end.

    ``total`` > 0 shows a percentage (with sizes, rate and time left when ``unit`` is "bytes");
    0 shows the elapsed time only. ``add`` and ``update`` may be called from several threads.
    """

    def __init__(self, label: str, total: int = 0, *, unit: str = "bytes", done: int = 0,
                 stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout
        try:
            self._tty = self._stream.isatty()
        except (AttributeError, ValueError):
            self._tty = False
        self._label, self._total, self._unit = label, total, unit
        self._done = self._first = done
        self._start = self._shown = time.monotonic()
        self._width = 0
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._ticker = None
        self._draw(time.monotonic())
        if self._tty:
            self._ticker = threading.Thread(target=self._tick, name="straydog-progress", daemon=True)
            self._ticker.start()

    def add(self, count: int) -> None:
        with self._lock:
            self._done += count
        self._maybe_draw()

    def update(self, done: int, total: int | None = None) -> None:
        with self._lock:
            self._done = done
            if total is not None:
                self._total = total
        self._maybe_draw()

    def close(self, outcome: str = "done") -> None:
        """End the line with ``outcome`` and the elapsed time; idempotent."""
        if self._closed.is_set():
            return
        self._closed.set()
        if self._ticker is not None:
            self._ticker.join()
        with self._lock:
            text = f"{self._label}: {outcome} ({_clock(time.monotonic() - self._start)})"
            self._write(text, final=True)

    def __enter__(self) -> ProgressLine:
        return self

    def __exit__(self, kind, _value, _traceback) -> None:
        self.close("done" if kind is None else "stopped")

    def _tick(self) -> None:
        while not self._closed.wait(0.5):
            with self._lock:
                self._draw(time.monotonic())

    def _maybe_draw(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._shown >= (0.2 if self._tty else 15) and not self._closed.is_set():
                self._draw(now)

    def _draw(self, now: float) -> None:
        self._shown = now
        elapsed = now - self._start
        if self._total > 0:
            done = min(self._done, self._total)
            parts = [f"{100 * done // self._total:3d}%"]
            if self._unit == "bytes":
                parts.append(f"{_size(done)} / {_size(self._total)}")
                # The first seconds (connections opening, read-ahead filling) would predict nonsense.
                rate = (done - self._first) / elapsed if elapsed >= 3 else 0
                if rate > 0:
                    parts.append(f"{_size(rate)}/s, {_clock((self._total - done) / rate)} left")
            text = f"{self._label}  {'  '.join(parts)}"
        else:
            spinner = _SPINNER[int(elapsed * 2) % len(_SPINNER)] if self._tty else "..."
            text = f"{self._label}  {spinner} {_clock(elapsed)}"
        self._write(text)

    def _write(self, text: str, *, final: bool = False) -> None:
        if self._tty:
            # Pad over the previous text rather than relying on terminal escape codes.
            padding = " " * max(0, self._width - len(text))
            self._stream.write("\r" + text + padding + ("\n" if final else ""))
            self._width = 0 if final else len(text)
        else:
            self._stream.write(text + "\n")
        self._stream.flush()
