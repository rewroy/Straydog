"""Straydog's private, bounded stdio inference service and owner-facing CLI.

This worker opens no network listener and never executes tools or model code.
The server's runtime serializes requests; cancellation is read independently of
the generation thread. All model bytes are authenticated before use.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

from straydog.bundles import MANIFEST, QUANTIZATIONS
from straydog.devices import GpuGroup
from straydog.model import StraydogModel
from straydog.weights import StraydogWeights

logger = logging.getLogger(__name__)
_FRAME_BYTES = 16 * 1024 * 1024


class _Loading:
    """Startup progress frames before ``ready``: each stage change at once, then a few per second."""

    def __init__(self) -> None:
        self._lock, self._stage, self._sent = threading.Lock(), "", 0.0

    def __call__(self, stage: str, done: int = 0, total: int = 0) -> None:
        now = perf_counter()
        with self._lock:
            if stage == self._stage and now - self._sent < 0.25 and done < total:
                return
            self._stage, self._sent = stage, now
            frame = {"event": "loading", "stage": stage, "done": done, "total": total}
            sys.stdout.buffer.write(json.dumps(frame).encode() + b"\n")
            sys.stdout.buffer.flush()


class StraydogWorker:
    """Own one model, tokenizer and generation thread behind the private pipe."""

    def __init__(self, model: StraydogModel, resources: dict[str, bytes]) -> None:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        self.model = model
        self.tokenizer = Tokenizer.from_str(resources["tokenizer.json"].decode())
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
        environment.globals["raise_exception"] = _template_error
        self.template = environment.from_string(resources["chat_template.jinja"].decode())
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._active: str | None = None

    def _send(self, message: dict[str, Any]) -> None:
        data = json.dumps(message, ensure_ascii=False, allow_nan=False)
        if len(data.encode()) > _FRAME_BYTES:
            raise ValueError("Straydog response exceeds the IPC frame limit")
        with self._write_lock:
            # Frames are UTF-8 on every platform; a Windows pipe's text layer would use the ANSI code page.
            sys.stdout.buffer.write(data.encode() + b"\n")
            sys.stdout.buffer.flush()

    def encode(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        thinking: bool = False,
        effort: str = "medium",
    ) -> list[int]:
        """Render the authenticated official template and encode locally."""
        return self.tokenizer.encode(self.render(messages, tools, thinking=thinking, effort=effort),
                                     add_special_tokens=False).ids

    def render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        thinking: bool = False,
        effort: str = "medium",
    ) -> str:
        """The prompt text the official template renders for a request."""
        if not isinstance(messages, list) or not messages or len(messages) > 4096:
            raise ValueError("invalid Straydog messages")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content", ""), (str, type(None))):
                raise ValueError("Straydog initially supports text messages only")
        messages = [_template_message(message) for message in messages]
        if effort not in {"low", "medium", "xhigh"}:
            raise ValueError("invalid Straydog reasoning effort")
        text = self.template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=thinking,
            reasoning_effort=effort,
        )
        if len(text.encode()) > _FRAME_BYTES:
            raise ValueError("Straydog rendered prompt exceeds the IPC limit")
        return text

    def _generate(self, request: dict[str, Any]) -> None:
        request_id = request["id"]
        terminal: dict[str, Any]
        started = perf_counter()
        first_text = None
        python_profile = None
        try:
            body = request["body"]
            self.model.stats = {} if body.get("profile") else None
            self.model.weights.gpu.stats = {} if body.get("profile") else None
            ids = self.encode(
                body["messages"],
                body.get("tools"),
                thinking=bool(body.get("thinking", False)),
                effort=body.get("reasoning_effort", "medium"),
            )
            encode_seconds = perf_counter() - started
            max_new = body.get("max_tokens", 512)
            if max_new is None:
                max_new = self.model.context - len(ids)
            if isinstance(max_new, bool) or not isinstance(max_new, int):
                raise ValueError("Straydog max_tokens must be an integer")
            keep = body.get("checkpoint", True)
            if not isinstance(keep, bool):
                raise ValueError("Straydog checkpoint must be a boolean")
            generated: list[int] = []
            prefill_profile = None
            emitted = ""
            if body.get("profile_python"):
                import cProfile

                python_profile = cProfile.Profile()
                python_profile.enable()
            for token in self.model.generate(
                ids,
                max_new=max_new,
                temperature=float(body.get("temperature", 0)),
                seed=int(body.get("seed", 0)),
                cancelled=self._cancel.is_set,
                progress=lambda done, total: self._send(
                    {"id": request_id, "event": "progress", "done": done, "total": total}
                ),
                # A prompt ending in a tool result continues the same reply; one ending in a user message starts one.
                turn=body["messages"][-1].get("role") != "tool",
                checkpoint=keep,
            ):
                if prefill_profile is None and self.model.stats is not None:
                    prefill_profile = {"model": dict(self.model.stats), "gpu": dict(self.model.weights.gpu.stats)}
                generated.append(token)
                decoded = self.tokenizer.decode(generated, skip_special_tokens=False).rstrip("\ufffd")
                if not decoded.startswith(emitted):
                    raise RuntimeError("Straydog tokenizer output changed an emitted prefix")
                if decoded != emitted:
                    if first_text is None:
                        first_text = perf_counter() - started
                    self._send({"id": request_id, "event": "delta", "content": decoded[len(emitted) :]})
                    emitted = decoded
            decoded = self.tokenizer.decode(generated, skip_special_tokens=False)
            if decoded.startswith(emitted) and decoded != emitted:
                self._send({"id": request_id, "event": "delta", "content": decoded[len(emitted) :]})
            terminal = {
                "id": request_id,
                "event": "done",
                "finish_reason": "cancelled"
                if self._cancel.is_set()
                else "length"
                if len(generated) == max_new
                else "stop",
                "usage": {
                    "prompt_tokens": len(ids),
                    "completion_tokens": len(generated),
                    "total_tokens": len(ids) + len(generated),
                    "cached_tokens": int(getattr(self.model, "last_metrics", {}).get("cached_tokens") or 0),
                },
                "gpu": self.model.weights.gpu.info(),
                "timings": dict(getattr(self.model, "last_metrics", {}))
                | {
                    "encode_seconds": encode_seconds,
                    "first_text_seconds": first_text,
                    "request_seconds": perf_counter() - started,
                },
            }
            if body.get("profile"):
                terminal["profile"] = {"model": self.model.stats, "gpu": self.model.weights.gpu.stats}
                if prefill_profile is not None:
                    terminal["profile"]["prefill"] = prefill_profile
                    terminal["profile"]["decode"] = {
                        domain: {key: value - prefill_profile[domain].get(key, 0) for key, value in total.items()}
                        for domain, total in (("model", self.model.stats), ("gpu", self.model.weights.gpu.stats))
                    }
        except Exception as exc:
            logger.exception("Straydog generation failed")
            fatal = not isinstance(exc, (ValueError, KeyError, TypeError))
            try:
                self.model.reset()
            except Exception:
                # A failed device (e.g. an illegal memory access) cannot be reset; the request
                # must still end, or the server waits on it until its read timeout.
                logger.exception("Straydog could not reset after a failed generation")
                fatal = True
            terminal = {"id": request_id, "event": "error", "message": str(exc)[:500], "fatal": fatal}
        finally:
            if python_profile is not None:
                python_profile.disable()
        if python_profile is not None:
            import io
            import pstats

            report = io.StringIO()
            pstats.Stats(python_profile, stream=report).sort_stats("tottime").print_stats(35)
            terminal.setdefault("profile", {})["python"] = report.getvalue()
        with self._state_lock:
            self._active = None
        self._send(terminal)

    def serve(self) -> None:
        """Read bounded owner frames; reject concurrent generation."""
        self._send(
            {
                "event": "ready",
                "protocol": 1,
                "engine": "Straydog",
                "model": "Qwen3.8-Flash-Next",
                "quantization": self.model.weights.quantization,
                "context": self.model.context,
                "state_bytes": self.model.state_bytes,
                "speculative_drafts": getattr(self.model, "speculative", 0),
                "placement": {
                    "resident": bool(getattr(self.model, "_resident_moe", False)),
                    "partial": bool(getattr(self.model, "_hybrid", False)),
                    "host_kv": bool(getattr(self.model, "_host_attention", False)),
                    "prefill_batch_size": getattr(self.model, "prefill_batch_size", None),
                },
                "gpu": self.model.weights.gpu.info(),
            }
        )
        try:
            while True:
                line = sys.stdin.buffer.readline(_FRAME_BYTES + 1)
                if not line:
                    break
                if len(line) > _FRAME_BYTES or not line.endswith(b"\n"):
                    raise ValueError("invalid Straydog IPC frame")
                request = json.loads(line)
                if not isinstance(request, dict) or not isinstance(request.get("id"), str):
                    raise ValueError("invalid Straydog request")
                if len(request["id"]) > 64:
                    raise ValueError("invalid Straydog request id")
                command = request.get("command")
                if command == "shutdown":
                    break
                if command == "cancel":
                    if request["id"] == self._active:
                        self._cancel.set()
                    continue
                if command == "tokenize":
                    text = request.get("text")
                    if not isinstance(text, str):
                        raise ValueError("Straydog tokenize needs text")
                    self._send(
                        {
                            "id": request["id"],
                            "event": "count",
                            "tokens": len(self.tokenizer.encode(text, add_special_tokens=False).ids),
                        }
                    )
                    continue
                if command != "chat" or not isinstance(request.get("body"), dict):
                    raise ValueError("unsupported Straydog command")
                with self._state_lock:
                    if self._active is not None:
                        self._send({"id": request["id"], "event": "error", "message": "Straydog is busy"})
                        continue
                    # Join a finished predecessor before reusing the cancellation event.
                    if self._thread is not None:
                        self._thread.join()
                    self._active = request["id"]
                    self._cancel.clear()
                    self._thread = threading.Thread(target=self._generate, args=(request,), name="straydog-generation")
                    self._thread.start()
        finally:
            self._cancel.set()
            if self._thread is not None:
                self._thread.join()


def _template_error(message: str) -> None:
    raise ValueError(message)


def _template_message(message: dict[str, Any]) -> dict[str, Any]:
    """Decode OpenAI-style JSON tool arguments into the mapping the template iterates."""
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return message
    decoded = []
    for call in calls:
        # The template reads ``call.function`` when present, else the call itself.
        wrapped = isinstance(call, dict) and isinstance(call.get("function"), dict)
        function = call["function"] if wrapped else call
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if isinstance(function, dict) and isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise ValueError("Straydog tool-call arguments are not valid JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError("Straydog tool-call arguments must be a JSON object")
            function = {**function, "arguments": arguments}
            call = {**call, "function": function} if wrapped else function
        decoded.append(call)
    return {**message, "tool_calls": decoded}


def main() -> int:
    """Start Straydog from explicit, local model and native-library paths."""
    parser = argparse.ArgumentParser(prog="straydog")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--gpu-library", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--split-mode", choices=["layer", "experts"], default="experts")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--vram-gib", type=float, default=6)
    parser.add_argument("--state-ram-gib", type=float, default=2)
    parser.add_argument("--prefill-batch-size", type=int, default=256)
    parser.add_argument("--matmul-precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--gemm-min-batch", type=int, default=8)
    parser.add_argument("--quantization", choices=QUANTIZATIONS, default="Q2_0")
    parser.add_argument("--drafter-root", type=Path)
    parser.add_argument("--speculative", type=int, default=0)
    parser.add_argument("--host-memory-bytes", type=int, default=0,
                        help="host memory available to the worker (its cgroup limits are invisible inside isolation)")
    args = parser.parse_args()
    loading = _Loading()
    with GpuGroup(
        args.gpu_library.resolve(), devices=args.devices, budget_bytes=int(args.vram_gib * 1024**3),
        split_mode=args.split_mode,
    ) as gpu:
        gpu.configure_gemm(minimum_batch=args.gemm_min_batch, precision=args.matmul_precision)
        weights = StraydogWeights(
            args.model_root, MANIFEST, args.inspector.resolve(), gpu,
            drafter=args.drafter_root if args.speculative else None, quantization=args.quantization,
            host_memory_bytes=args.host_memory_bytes, progress=loading,
        )
        loading("prepare")
        model = StraydogModel(
            weights,
            context=args.context,
            state_budget_bytes=int(args.state_ram_gib * 1024**3),
            prefill_batch_size=args.prefill_batch_size,
            speculative=args.speculative,
        )
        StraydogWorker(model, weights.resources).serve()
        # Not on an error: its traceback still holds views of the mappings (process exit releases them),
        # and a close failure there would replace the cause the server reports.
        weights.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
