"""The runtime owns one isolated worker: startup, bounded jobs, cancellation and failure reports."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from straydog.config import Settings
from straydog.runtime import StraydogRuntime


def _settings(tmp_path, **values):
    """Resolved settings for a worker the test replaces."""
    return Settings(model=str(tmp_path), quant="Q2_0", inspector=str(tmp_path / "inspect"),
                    gpu_library=str(tmp_path / "gpu.so"), batch_size=1024, state_ram=2, draft_max=0, **values)


async def test_runtime_serializes_cancellation_and_bounds_queue():
    runtime = StraydogRuntime(Settings())
    runtime._start = AsyncMock()
    runtime._send = AsyncMock()
    entered = asyncio.Event()

    async def reading(*args):
        entered.set()
        await asyncio.Event().wait()

    runtime._read = reading
    runtime._abort = AsyncMock()
    jobs = [(await runtime.submit({"messages": []}))["request_id"] for _ in range(4)]
    with pytest.raises(ValueError, match="full"):
        await runtime.submit({})
    await entered.wait()
    assert runtime._send.await_count == 1
    await runtime.cancel(jobs[0])
    runtime._abort.assert_awaited_once_with(jobs[0])
    await runtime.close()
    assert not runtime._jobs
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.submit({})


async def test_runtime_terminal_delivery_and_tokenizer():
    runtime = StraydogRuntime(Settings())
    runtime._start = AsyncMock()

    async def sending(frame):
        runtime._last_id = frame["id"]

    runtime._send = sending

    async def reading(*args):
        return {"id": runtime._last_id, "event": "done", "finish_reason": "stop"}

    runtime._read = reading
    request_id = (await runtime.submit({}))["request_id"]
    await runtime._jobs[request_id].task
    assert (await runtime.poll(request_id))["done"]

    async def counting(*args):
        return {"id": runtime._last_id, "event": "count", "tokens": 5}

    runtime._read = counting
    assert await runtime.tokenize("abc") == {"tokens": 5}
    assert not (await runtime.status())["running"]
    assert runtime.snapshot()["active_jobs"] == 0
    await runtime.close()


def test_linux_worker_isolation_has_no_network_or_general_home_mount(tmp_path, monkeypatch):
    from straydog.isolation import isolated_worker_command

    monkeypatch.setattr("straydog.isolation.shutil.which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr("straydog.isolation.sys.platform", "linux")
    args = isolated_worker_command(
        ["python", "-I", "-m", "worker"], model=tmp_path, inspector=tmp_path / "inspect", library=tmp_path / "gpu.so"
    )
    assert "--unshare-all" in args and "--share-net" not in args
    assert str(Path.home()) not in args and "--bind" not in args
    drafter = tmp_path / "mtp"
    drafter.mkdir()
    lock = json.loads((Path(__file__).resolve().parents[1] / "src/straydog/assets.json").read_text())
    (drafter / lock["drafter"]["name"]).write_bytes(b"x")
    args = isolated_worker_command(["python"], model=tmp_path, inspector=tmp_path / "inspect",
                                   library=tmp_path / "gpu.so", drafter=drafter)
    assert str(drafter / lock["drafter"]["name"]) in args and str(drafter) + "/" not in args
    # Windows: the AppContainer launcher receives the same files; no host without containment starts a worker.
    monkeypatch.setattr("straydog.isolation.sys.platform", "win32")
    args = isolated_worker_command(["python", "-m", "worker"], model=tmp_path, inspector=tmp_path / "inspect",
                                   library=tmp_path / "gpu.dll", drafter=drafter)
    assert args[1:4] == ["-I", "-m", "straydog.appcontainer"]
    assert args[args.index("--"):] == ["--", "python", "-m", "worker"]
    files = {args[index + 1] for index, value in enumerate(args) if value == "--file"}
    assert {str(tmp_path / "gpu.dll"), str(drafter / lock["drafter"]["name"])} <= files
    assert args[args.index("--directory") + 1] == str(tmp_path)
    monkeypatch.setattr("straydog.isolation.sys.platform", "darwin")
    with pytest.raises(RuntimeError, match="bubblewrap or Windows AppContainer"):
        isolated_worker_command([], model=tmp_path, inspector=tmp_path, library=tmp_path)


def test_worker_env_names_the_cuda_toolkit_only_on_windows(monkeypatch):
    from straydog import runtime as straydog_runtime

    monkeypatch.setenv("CUDA_PATH", r"C:\CUDA\v12.8")
    monkeypatch.setattr(straydog_runtime.sys, "platform", "win32")
    assert straydog_runtime._worker_env()["CUDA_PATH"] == r"C:\CUDA\v12.8"
    assert straydog_runtime._worker_env()["OPENBLAS_NUM_THREADS"] == "1"
    monkeypatch.setattr(straydog_runtime.sys, "platform", "linux")
    assert "CUDA_PATH" not in straydog_runtime._worker_env()
    monkeypatch.setattr(straydog_runtime.sys, "platform", "win32")
    monkeypatch.delenv("CUDA_PATH")
    assert "CUDA_PATH" not in straydog_runtime._worker_env()


async def test_startup_failure_reports_the_worker_reason(tmp_path, monkeypatch):
    import sys

    reason = "RuntimeError: this Straydog build has no GPU code for gfx1100; build it for this GPU"
    script = f"import sys; sys.stderr.write({('Traceback (most recent call last):' + chr(10) + reason + chr(10))!r})"
    monkeypatch.setattr("straydog.isolation.isolated_worker_command",
                        lambda command, **kwargs: [sys.executable, "-c", script + "; sys.exit(1)"])
    runtime = StraydogRuntime(_settings(tmp_path))
    with pytest.raises(RuntimeError, match="no GPU code for gfx1100"):
        await runtime.status(start=True)
    assert "no GPU code for gfx1100" in runtime.snapshot()["error"]
    await runtime.close()


async def test_startup_failure_reason_in_the_system_code_page_is_readable(tmp_path, monkeypatch):
    # A Windows worker writes its pipe in the ANSI code page: a localized DLL load error must not become mojibake.
    import sys

    reason = "ImportError: DLL load failed while importing _multiarray_umath: アクセスが拒否されました。"
    script = (f"import sys; sys.stderr.buffer.write(b'x' * 9000 + b'\\n' + {(reason + chr(10)).encode('cp932')!r});"
              " sys.stderr.flush(); sys.exit(1)")
    monkeypatch.setattr("straydog.platform.spawn._CONSOLE_FALLBACK_CODECS", ("cp932",))
    monkeypatch.setattr("straydog.isolation.isolated_worker_command",
                        lambda command, **kwargs: [sys.executable, "-c", script])
    runtime = StraydogRuntime(_settings(tmp_path))
    with pytest.raises(RuntimeError, match="アクセスが拒否されました"):
        await runtime.status(start=True)
    await runtime.close()


async def test_runtime_passes_its_host_memory_to_the_isolated_worker(tmp_path, monkeypatch):
    import sys

    spawned = []

    def isolate(command, **kwargs):
        spawned.append(command)
        return [sys.executable, "-c", "import sys; sys.exit(1)"]

    monkeypatch.setattr("straydog.isolation.isolated_worker_command", isolate)
    monkeypatch.setattr("straydog.host.host_memory_bytes", lambda: 48 * 1024**3)
    runtime = StraydogRuntime(_settings(tmp_path))
    with pytest.raises(RuntimeError):
        await runtime.status(start=True)
    await runtime.close()
    command = spawned[0]
    assert command[command.index("--host-memory-bytes") + 1] == str(48 * 1024**3)


async def test_startup_relays_loading_frames_until_ready(tmp_path, monkeypatch):
    import sys

    frames = [
        {"event": "loading", "stage": "authenticate", "done": 0, "total": 10},
        {"event": "loading", "stage": "authenticate", "done": 10, "total": 10},
        {"event": "loading", "stage": "prepare", "done": -3, "total": "many"},
        {"event": "ready", "protocol": 1, "engine": "Straydog", "context": 262144, "model": "Qwen3.8-Flash-Next",
         "quantization": "Q2_0"},
    ]
    script = f"import sys; print({chr(10).join(map(json.dumps, frames))!r}, flush=True); sys.stdin.read()"
    monkeypatch.setattr("straydog.isolation.isolated_worker_command",
                        lambda command, **kwargs: [sys.executable, "-c", script])
    runtime = StraydogRuntime(_settings(tmp_path))
    seen = []
    runtime.on_loading = seen.append
    status = await runtime.status(start=True)
    await runtime.close()
    assert status["ready"]["engine"] == "Straydog"
    # Malformed counts are clamped rather than trusted.
    assert seen == [{"stage": "authenticate", "done": 0, "total": 10},
                    {"stage": "authenticate", "done": 10, "total": 10},
                    {"stage": "prepare", "done": 0, "total": 0}]


async def test_loading_frames_do_not_extend_the_startup_timeout(tmp_path, monkeypatch):
    import sys

    script = ("import json, time\nwhile True:\n    print(json.dumps({'event': 'loading', 'stage': 'prepare'}),"
              " flush=True)\n    time.sleep(0.05)")
    monkeypatch.setattr("straydog.isolation.isolated_worker_command",
                        lambda command, **kwargs: [sys.executable, "-c", script])
    runtime = StraydogRuntime(_settings(tmp_path, startup_timeout=1))
    with pytest.raises(TimeoutError):
        await runtime.status(start=True)
    await runtime.close()


def test_each_finished_generation_is_summarized_in_one_line():
    from straydog.runtime import request_summary

    event = {"event": "done", "usage": {"prompt_tokens": 17199, "completion_tokens": 167, "cached_tokens": 14200},
             "timings": {"resumed_from": "turn", "prefill_tokens_per_second": 1304.2, "ttft_seconds": 2.364,
                         "decode_tokens_per_second": 54.27}}
    assert request_summary(event) == (
        "Straydog request: prompt 17199 tokens (reused 14200 from turn), prefill 1304 tok/s,"
        " first token 2.36 s, 167 tokens at 54.3 tok/s")
    assert request_summary({"event": "done"}) == (
        "Straydog request: prompt - tokens (reused - from -), prefill - tok/s, first token - s, - tokens at - tok/s")


def test_startup_reports_how_the_model_was_placed():
    from straydog.runtime import ready_summary

    ready = {"quantization": "IQ3_S", "context": 262144, "speculative_drafts": 3,
             "placement": {"resident": True, "partial": False, "host_kv": True, "prefill_batch_size": 4096}}
    assert ready_summary(ready) == ("Straydog ready: IQ3_S, context 262144, fully resident, MTP 3, K/V in host memory,"
                                    " prefill batch 4096")
    assert "partially resident, no speculation, K/V in VRAM" in ready_summary(
        {"speculative_drafts": 0, "placement": {"partial": True}})
