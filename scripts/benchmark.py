"""Startup, prefill, first-token and decode figures of single requests, through the isolated worker.

Takes the same options as ``straydog serve`` (and reads the same config file), plus the
cases to run and where to write the report.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from straydog.config import add_options, given_options, load_settings, resolve
from straydog.runtime import StraydogRuntime


def _cases(name: str) -> list[dict]:
    short = {"name": "short", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 32}
    tools = {
        "name": "tools",
        "messages": [{"role": "user", "content": "Use get_weather for Tokyo."}],
        "max_tokens": 64,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
        ],
    }
    long = {
        "name": "long",
        "messages": [{"role": "user", "content": (
            "The reference code is STRAYDOG-7319.\n"
            + "Reference note: clouds move across the sky, and the river flows toward the sea.\n" * 256
            + "What is the reference code at the beginning? Reply with the code only."
        )}],
        "max_tokens": 24,
    }
    generate = {
        "name": "generate",
        "messages": [{"role": "user", "content": (
            "Write a Python function is_prime(n) with a docstring, then explain how it works in a short paragraph."
        )}],
        "max_tokens": 128,
    }
    # A long prompt whose routing reaches most experts (the repeated "long" note reaches few): a fixed,
    # seeded service log, so out-of-RAM runs read expert banks the way varied real prompts do.
    rng = random.Random(7319)
    words = ("cache", "shard", "kernel", "route", "socket", "parser", "budget", "tensor", "window", "ledger",
             "harbor", "violet", "quartz", "meadow", "copper", "thunder", "lantern", "orbit", "falcon", "maple")
    lines = []
    for i in range(120):
        level, component = rng.choice(("INFO", "WARN", "DEBUG", "ERROR")), rng.choice(words)
        text = " ".join(rng.choice(words) for _ in range(rng.randrange(3, 9)))
        stamp = f"{9 + i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}"
        lines.append(f"{stamp} {level} {component}-{rng.randrange(100, 999)} {text}"
                     f" id={rng.getrandbits(32):08x} took {rng.randrange(1, 5000)}ms")
    diverse = {
        "name": "diverse",
        "messages": [{"role": "user", "content": (
            "Service log:\n" + "\n".join(lines) + "\nWhich component logged the most ERROR lines? Answer in one word."
        )}],
        "max_tokens": 24,
    }
    cases = {
        "short": [short], "tools": [tools], "both": [short, tools], "long": [long], "all": [short, tools, long],
        "generate": [generate], "diverse": [diverse], "ssd": [short, tools, long, diverse],
    }
    return cases[name]


def _fingerprints(settings) -> dict:
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src/straydog").glob("*.py"))
    sources += sorted((root / "gpu").glob("*"))
    sources += [Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in sources:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return {
        "engine_source_sha256": digest.hexdigest(),
        "gpu_library_sha256": hashlib.sha256(Path(settings.gpu_library).read_bytes()).hexdigest(),
        "inspector_sha256": hashlib.sha256(Path(settings.inspector).read_bytes()).hexdigest(),
        "model_bundle_sha256": hashlib.sha256((root / "src/straydog/assets.json").read_bytes()).hexdigest(),
    }


async def _measure(args, settings) -> dict:
    runtime = StraydogRuntime(settings)
    report = {"settings": settings.model_dump(mode="json", exclude={"api_key"}),
              "fingerprints": _fingerprints(settings), "cases": []}
    try:
        started = time.perf_counter()
        report["ready"] = (await runtime.status(start=True))["ready"]
        report["startup_seconds"] = time.perf_counter() - started
        print(json.dumps({"startup_seconds": report["startup_seconds"]}), flush=True)
        for case in _cases(args.case):
            for repeat in range(args.repeat):
                request = await runtime.submit(
                    {k: v for k, v in case.items() if k != "name"}
                    | {"profile": True, "profile_python": args.profile_python}
                )
                text, terminal = "", None
                last_progress = time.monotonic()
                while True:
                    batch = await runtime.poll(request["request_id"])
                    for event in batch["events"]:
                        if event["event"] == "delta":
                            text += event["content"]
                        elif event["event"] == "error":
                            raise RuntimeError(event["message"])
                        elif event["event"] == "done":
                            terminal = event
                        elif event["event"] == "progress" and time.monotonic() - last_progress >= 10:
                            print(
                                json.dumps({"name": case["name"], "repeat": repeat, **event}),
                                file=sys.stderr,
                                flush=True,
                            )
                            last_progress = time.monotonic()
                    if batch["done"]:
                        break
                    await asyncio.sleep(0.02)
                if terminal is None:
                    raise RuntimeError("benchmark ended without metrics")
                terminal.pop("id", None)
                result = {"name": case["name"], "repeat": repeat, "content": text, **terminal}
                report["cases"].append(result)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        await runtime.close()
    return report


def main() -> int:
    """Measure startup, template, uncached prefill, TTFT and decode separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser)
    parser.add_argument("--case", choices=["short", "tools", "both", "long", "all", "generate", "diverse", "ssd"],
                        default="both")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-python", action="store_true", help="Diagnostic run; Python profiling adds overhead")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    options, config_file = given_options(args)
    settings, _ = load_settings(options, config_file=config_file)
    report = asyncio.run(_measure(args, resolve(settings)))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
