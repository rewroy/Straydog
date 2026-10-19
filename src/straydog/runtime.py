"""Ownership of the single isolated Straydog worker and bounded generation jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from straydog.config import MODEL_NAME, Settings
from straydog.platform import process_group
from straydog.platform.spawn import decode_console_output
from straydog.platform.environment import sanitized_subprocess_env

logger = logging.getLogger(__name__)
_FRAME_LIMIT = 16 * 1024 * 1024


def _worker_env() -> dict[str, str]:
    """Variables the worker needs beyond the sanitized base environment."""
    env = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    if sys.platform == "win32" and os.environ.get("CUDA_PATH"):
        # straydog.library resolves a CUDA build's cuBLAS from this toolkit (Windows has no RUNPATH).
        env["CUDA_PATH"] = os.environ["CUDA_PATH"]
    return env


def ready_summary(ready: dict[str, Any]) -> str:
    """How the model was placed: what decides its speed (residency, speculation, K/V place, prefill batch)."""
    placement = ready.get("placement") or {}
    residency = ("fully resident" if placement.get("resident")
                 else "partially resident" if placement.get("partial") else "bounded expert cache")
    drafts = ready.get("speculative_drafts") or 0
    return (
        f"Straydog ready: {ready.get('quantization', '-')}, context {ready.get('context', '-')}, {residency},"
        f" {'MTP ' + str(drafts) if drafts else 'no speculation'},"
        f" K/V in {'host memory' if placement.get('host_kv') else 'VRAM'},"
        f" prefill batch {placement.get('prefill_batch_size', '-')}"
    )


def request_summary(event: dict[str, Any]) -> str:
    """One line per finished generation: what was read, what was reused, and how fast."""
    usage, timings = event.get("usage") or {}, event.get("timings") or {}

    def number(value: Any, digits: int) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"

    return (
        f"Straydog request: prompt {usage.get('prompt_tokens', '-')} tokens"
        f" (reused {usage.get('cached_tokens', '-')} from {timings.get('resumed_from', '-')}),"
        f" prefill {number(timings.get('prefill_tokens_per_second'), 0)} tok/s,"
        f" first token {number(timings.get('ttft_seconds'), 2)} s,"
        f" {usage.get('completion_tokens', '-')} tokens at {number(timings.get('decode_tokens_per_second'), 1)} tok/s"
    )


@dataclass
class _Job:
    events: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    task: asyncio.Task | None = None
    last_poll: float = field(default_factory=time.monotonic)
    done: bool = False


class StraydogRuntime:
    """One worker shared by every request; ``config`` is resolved settings (straydog.config.resolve)."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._stderr: asyncio.Task | None = None
        self._stderr_tail = b""
        self._lock = asyncio.Lock()
        self._jobs: dict[str, _Job] = {}
        self._ready: dict[str, Any] = {}
        self._fatal: str | None = None
        self._closed = False
        # Called with each startup progress frame ({"stage", "done", "total"}) before ``ready``.
        self.on_loading: Callable[[dict[str, Any]], None] | None = None

    async def _send(self, frame: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Straydog worker is unavailable")
        data = json.dumps(frame, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        if len(data) > _FRAME_LIMIT:
            raise ValueError("Straydog request exceeds its IPC frame limit")
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def _read(self, timeout: float = 300) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Straydog worker is unavailable")
        line = await asyncio.wait_for(self._process.stdout.readline(), timeout)
        if not line or len(line) > _FRAME_LIMIT or not line.endswith(b"\n"):
            raise RuntimeError("Straydog worker ended or sent an invalid frame")
        frame = json.loads(line)
        if not isinstance(frame, dict) or len(str(frame.get("content", ""))) > 4096:
            raise RuntimeError("invalid Straydog response")
        return frame

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        while data := await stream.read(4096):
            # Kept as bytes and decoded whole: on Windows the worker writes its pipe in the system
            # code page (a localized OS error such as a DLL load failure), and a chunk can split a character.
            tail = self._stderr_tail + data
            if len(tail) > 8192:   # whole lines only, so a cut never starts inside a character
                tail = tail[-8192:].partition(b"\n")[2]
            self._stderr_tail = tail
            logger.debug("Straydog worker: %s", decode_console_output(data).rstrip())

    async def _start(self) -> None:
        if self._closed:
            raise RuntimeError("Straydog runtime is closed")
        if self._fatal:
            raise RuntimeError(self._fatal)
        if self._process is not None:
            if self._process.returncode is not None:
                self._fatal = "Straydog's worker exited; restart the server to reload it"
                raise RuntimeError(self._fatal)
            return
        cfg = self.config
        if cfg.quant is None or cfg.batch_size is None or cfg.state_ram is None:
            raise ValueError("Straydog settings must be resolved before the worker starts")
        for path in (cfg.model, cfg.inspector, cfg.gpu_library):
            if not path or not Path(path).is_absolute():
                raise ValueError("Straydog model, inspector and GPU library paths must be absolute")
        command = [
            sys.executable,
            "-I",
            "-m",
            "straydog.worker",
            "--model-root",
            cfg.model,
            "--inspector",
            cfg.inspector,
            "--gpu-library",
            cfg.gpu_library,
            "--devices",
            *map(str, cfg.device),
            "--split-mode",
            cfg.split_mode,
            "--context",
            str(cfg.ctx_size),
            "--vram-gib",
            str(cfg.vram),
            "--state-ram-gib",
            str(cfg.state_ram),
            "--prefill-batch-size",
            str(cfg.batch_size),
        ]
        command += ["--matmul-precision", cfg.gemm, "--gemm-min-batch", str(cfg.gemm_min_batch)]
        command += ["--quantization", cfg.quant]
        from straydog.host import host_memory_bytes

        # Isolation hides the worker's cgroup limits from it; host-memory policies (page-locking the
        # expert banks) must see the memory this process tree may actually use.
        memory = host_memory_bytes()
        if memory > 0:
            command += ["--host-memory-bytes", str(memory)]
        drafter = None
        if cfg.speculative:
            drafter = Path(cfg.model_draft)
            if not drafter.is_absolute():
                raise ValueError("Straydog drafter path must be absolute")
            command += ["--drafter-root", str(drafter), "--speculative", str(cfg.speculative)]
        from straydog.isolation import isolated_worker_command

        command = isolated_worker_command(
            command,
            model=Path(cfg.model),
            inspector=Path(cfg.inspector),
            library=Path(cfg.gpu_library),
            drafter=drafter,
            quantization=cfg.quant,
            secrets=[Path(cfg.api_key_file)] if cfg.api_key_file else [],
        )
        logger.info("Straydog starting: %s on GPU %s", cfg.quant, ",".join(map(str, cfg.device)))
        self._stderr_tail = b""
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_FRAME_LIMIT + 1,
                cwd=cfg.model,
                env=sanitized_subprocess_env(_worker_env()),
                **process_group.detached_spawn_kwargs(),
            )
            process_group.register_child_for_tree_kill(self._process)
            self._stderr = asyncio.create_task(self._drain_stderr(self._process.stderr), name="straydog-stderr")
            deadline = time.monotonic() + cfg.startup_timeout
            ready = await self._read(cfg.startup_timeout)
            stage = ""
            while ready.get("event") == "loading":
                progress = {"stage": str(ready.get("stage", ""))[:32]}
                for key in ("done", "total"):
                    value = ready.get(key, 0)
                    progress[key] = value if isinstance(value, int) and value >= 0 else 0
                if progress["stage"] != stage:
                    stage = progress["stage"]
                    logger.info("Straydog loading: %s", stage)
                if self.on_loading is not None:
                    self.on_loading(progress)
                ready = await self._read(max(0.0, deadline - time.monotonic()))
            if (
                ready.get("event") != "ready"
                or ready.get("protocol") != 1
                or ready.get("engine") != "Straydog"
                or ready.get("context") != cfg.ctx_size
                or ready.get("model") != MODEL_NAME
                or ready.get("quantization") != cfg.quant
            ):
                raise RuntimeError("Straydog startup contract mismatch")
            self._ready = ready
            logger.info("%s", ready_summary(ready))
        except BaseException as exc:
            await self._stop_process()
            # The worker's last stderr line names the cause (e.g. a GPU the build has no code for).
            lines = decode_console_output(self._stderr_tail).splitlines()
            reason = next((line.strip() for line in reversed(lines) if line.strip()), "")
            self._fatal = "Straydog failed to start" + (f" ({reason[-500:]})" if reason else "")
            self._fatal += "; see the server log and restart the server"
            logger.error("%s", self._fatal)
            if reason and isinstance(exc, Exception):
                raise RuntimeError(self._fatal) from exc
            raise

    async def status(self, *, start: bool = False) -> dict[str, Any]:
        """Expose observed capacity only after the worker allocates its state."""
        if start and not self._ready:
            async with self._lock:
                await self._start()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Observed state: whether the worker runs, how it placed the model, any fatal error."""
        running = self._process is not None and self._process.returncode is None
        return {
            "quantization": self.config.quant,
            "running": running,
            "ready": self._ready.copy() if running else {},
            "error": self._fatal,
            "active_jobs": len(self._jobs),
        }

    async def tokenize(self, text: str) -> dict[str, Any]:
        """Count tokens with the same authenticated tokenizer used for generation."""
        async with self._lock:
            await self._start()
            request_id = uuid.uuid4().hex
            await self._send({"id": request_id, "command": "tokenize", "text": text})
            result = await self._read()
            if result.get("id") != request_id or result.get("event") != "count":
                raise RuntimeError("Straydog tokenizer contract mismatch")
            return {"tokens": result["tokens"]}

    async def submit(self, body: dict[str, Any]) -> dict[str, str]:
        """Accept a bounded job; this method never executes any generated tool call."""
        if self._closed:
            raise RuntimeError("Straydog runtime is closed")
        if len(self._jobs) >= 4:
            raise ValueError("Straydog generation queue is full")
        if not isinstance(body, dict) or len(json.dumps(body).encode()) > _FRAME_LIMIT - 1024:
            raise ValueError("invalid Straydog generation request")
        request_id = uuid.uuid4().hex
        job = _Job()
        self._jobs[request_id] = job
        job.task = asyncio.create_task(self._run(request_id, job, body), name=f"straydog-{request_id}")
        return {"request_id": request_id}

    async def _run(self, request_id: str, job: _Job, body: dict[str, Any]) -> None:
        watchdog = asyncio.create_task(self._watch(request_id, job), name="straydog-job-watch")
        sent = False
        try:
            async with self._lock:
                try:
                    await self._start()
                    await self._send({"id": request_id, "command": "chat", "body": body})
                    sent = True
                    while True:
                        event = await self._read()
                        if event.get("id") != request_id:
                            raise RuntimeError("Straydog generation request id mismatch")
                        if isinstance(event.get("gpu"), dict):
                            self._ready["gpu"] = event["gpu"]
                        await job.events.put(event)
                        if event.get("event") == "done":
                            logger.info("%s", request_summary(event))
                        if event.get("event") in {"done", "error"}:
                            if event.get("fatal"):
                                self._fatal = "Straydog reported a fatal inference error; restart the server"
                                await self._stop_process()
                            sent = False
                            break
                finally:
                    if sent:
                        await self._abort(request_id)
        except asyncio.CancelledError:
            if not job.events.full():
                job.events.put_nowait({"event": "error", "message": "Straydog generation cancelled"})
        except Exception as exc:
            logger.exception("Straydog request failed")
            if not job.events.full():
                job.events.put_nowait({"event": "error", "message": str(exc)[:500]})
        finally:
            job.done = True
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)

    async def _watch(self, request_id: str, job: _Job) -> None:
        while not job.done:
            await asyncio.sleep(5)
            if time.monotonic() - job.last_poll > 30:
                if job.task is not None:
                    job.task.cancel()
                self._jobs.pop(request_id, None)
                return

    async def _abort(self, request_id: str) -> None:
        try:
            await self._send({"id": request_id, "command": "cancel"})
            async with asyncio.timeout(10):
                while True:
                    event = await self._read(10)
                    if event.get("id") == request_id and event.get("event") in {"done", "error"}:
                        return
        except Exception:
            logger.warning("Straydog cancellation did not settle; stopping owned worker", exc_info=True)
            self._fatal = "Straydog did not acknowledge cancellation; restart the server"
            await self._stop_process()

    async def poll(self, request_id: str) -> dict[str, Any]:
        """Drain a bounded batch; a lost client cannot leave an unbounded transcript."""
        job = self._jobs.get(request_id)
        if job is None:
            raise ValueError("Straydog generation was not found")
        job.last_poll = time.monotonic()
        events = []
        while not job.events.empty() and len(events) < 32:
            events.append(job.events.get_nowait())
        done = job.done and job.events.empty()
        if done:
            self._jobs.pop(request_id, None)
        return {"events": events, "done": done}

    async def cancel(self, request_id: str) -> dict[str, bool]:
        """Cancel queued or running work without replaying it."""
        job = self._jobs.get(request_id)
        if job is not None and job.task is not None:
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
            self._jobs.pop(request_id, None)
        return {"ok": True}

    async def _stop_process(self) -> None:
        proc, self._process = self._process, None
        self._ready = {}
        if proc is not None and proc.returncode is None:
            process_group.kill_tree(proc, signal_name="term")
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except TimeoutError:
                process_group.kill_tree(proc, signal_name="kill")
                await proc.wait()
        if self._stderr is not None:
            # The process has exited: let its last stderr reach the log before stopping the reader.
            await asyncio.wait({self._stderr}, timeout=1)
            self._stderr.cancel()
            await asyncio.gather(self._stderr, return_exceptions=True)
            self._stderr = None

    async def close(self) -> None:
        """Cancel all jobs and release the sole worker at server shutdown."""
        if self._closed:
            return
        self._closed = True
        for request_id in list(self._jobs):
            await self.cancel(request_id)
        await self._stop_process()

