"""Straydog's setup progress line: in place on a terminal, plain lines elsewhere."""

from __future__ import annotations

import io

from straydog import progress as straydog_progress
from straydog.progress import ProgressLine


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_progress_outside_a_terminal_prints_plain_lines():
    stream = io.StringIO()
    with ProgressLine("downloading model.gguf", 2_000_000_000, stream=stream) as line:
        line.add(1_000_000_000)   # within 15 s of the first line: not printed
    lines = stream.getvalue().splitlines()
    assert lines[0] == "downloading model.gguf    0%  0 MB / 2.0 GB"
    assert lines[-1] == "downloading model.gguf: done (0:00)"
    assert "\r" not in stream.getvalue()


def test_progress_on_a_terminal_redraws_one_line_with_rate_and_time_left(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(straydog_progress.time, "monotonic", lambda: clock[0])
    stream = Terminal()
    line = ProgressLine("authenticating 2 present file(s)", 4_000_000_000, stream=stream)
    clock[0] += 2
    line.update(500_000_000)   # too early for a rate
    clock[0] += 2
    line.update(1_000_000_000)
    line.close()
    text = stream.getvalue()
    assert text.startswith("\rauthenticating 2 present file(s)    0%  0 MB / 4.0 GB")
    assert "\rauthenticating 2 present file(s)   12%  500 MB / 4.0 GB\r" in text
    progress = "authenticating 2 present file(s)   25%  1.0 GB / 4.0 GB  250 MB/s, 0:12 left"
    assert "\r" + progress in text
    final = "authenticating 2 present file(s): done (0:04)"
    # The last line is padded over the longer progress text, then ends the line.
    assert text.endswith("\r" + final + " " * (len(progress) - len(final)) + "\n") and text.count("\n") == 1


def test_progress_without_a_total_shows_elapsed_time_and_a_failure_outcome():
    stream = io.StringIO()
    line = ProgressLine("Straydog: starting", stream=stream)
    line.close("failed")
    line.close()   # idempotent
    assert stream.getvalue().splitlines() == ["Straydog: starting  ... 0:00", "Straydog: starting: failed (0:00)"]
