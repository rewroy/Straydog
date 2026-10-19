"""Golden generations from the Python reference on real GPUs (design phase R3).

For each placement, a scripted sequence of requests that exercises every resume path (the
live sequence, the turn, prior, step and root checkpoints, a one-off request): the prompt
ids, the greedy tokens, the request metrics and a digest of every logits vector the model
returns (the prompt's last, then each decode step's). Rust must reproduce them bit for bit.

    ~/benchlock.sh golden -- uv run python scripts/golden_model.py --model DIR --drafter DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from straydog.bundles import MANIFEST
from straydog.devices import GpuGroup
from straydog.host import host_memory_bytes
from straydog.model import StraydogModel
from straydog.weights import StraydogWeights
from straydog.worker import StraydogWorker

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "crates/straydog-engine/tests/golden/model.json"
PLACEMENTS = [
    {"name": "two-gpu-experts-mtp", "devices": [0, 1], "vram_gib": 30, "context": 32768, "split_mode": "experts",
     "speculative": 3},
    {"name": "two-gpu-layers-mtp", "devices": [0, 1], "vram_gib": 30, "context": 32768, "split_mode": "layer",
     "speculative": 3},
    {"name": "two-gpu-host-kv-mtp", "devices": [0, 1], "vram_gib": 30, "context": 262144, "split_mode": "experts",
     "speculative": 3},
    {"name": "two-gpu-experts", "devices": [0, 1], "vram_gib": 30, "context": 32768, "split_mode": "experts",
     "speculative": 0},
    {"name": "one-gpu-partial", "devices": [0], "vram_gib": 24, "context": 32768, "split_mode": "experts",
     "speculative": 0},
]
SYSTEM = "You are a careful assistant. Reference notes follow.\n" + "".join(
    f"Note {index}: the {['river', 'kernel', 'harbor', 'ledger'][index % 4]} entry {index * 37 % 101} "
    f"belongs to section {index % 13}.\n" for index in range(260))


def conversation(worker, messages, tools=None) -> list[int]:
    return worker.encode(messages, tools, thinking=False, effort="medium")


def requests(worker, generate) -> list[dict]:
    """The scripted requests; each prompt may extend an earlier reply (``generate`` returns its text)."""
    weather = [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    first = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Which section holds note 42?"}]
    script = []

    def run(name, messages, *, tools=None, max_new=24, turn=True, checkpoint=True):
        ids = conversation(worker, messages, tools)
        reply = generate(name, ids, max_new=max_new, turn=turn, checkpoint=checkpoint)
        script.append(reply)
        return reply["text"]

    reply = run("root", first)
    second = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": "And note 7? Be brief."}]
    reply = run("live", second)
    third = second + [{"role": "assistant", "content": reply},
                      {"role": "user", "content": "Use get_weather for 東京."}]
    reply = run("tools", third, tools=weather, max_new=40)
    fourth = third + [{"role": "assistant", "content": reply},
                      {"role": "tool", "content": "晴れ、21°C"}]
    run("step", fourth, tools=weather)
    retry = first + [{"role": "assistant", "content": script[0]["text"]},
                     {"role": "user", "content": "And note 9? Answer in one word."}]
    run("turn", retry)
    run("new-chat", [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "List three notes in section 5."}],
        max_new=48)
    run("one-off", [{"role": "user", "content": "Title this chat in three words: notes and sections."}],
        checkpoint=False, max_new=12)
    run("short", [{"role": "user", "content": "Hi"}], max_new=8)
    return script


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=ROOT / "src/straydog/native/libstraydog_gpu.so")
    parser.add_argument("--quantization", default="IQ3_S")
    parser.add_argument("--only", help="run one placement")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    results = json.loads(args.output.read_text())["placements"] if args.only and args.output.exists() else {}
    for placement in PLACEMENTS:
        if args.only and placement["name"] != args.only:
            continue
        print(f"placement {placement['name']}", flush=True)
        with GpuGroup(args.library.resolve(), devices=placement["devices"],
                      budget_bytes=int(placement["vram_gib"] * 1024**3), split_mode=placement["split_mode"]) as gpu:
            gpu.configure_gemm(minimum_batch=2, precision="fp16")
            weights = StraydogWeights(args.model, MANIFEST, ROOT / ".build/bin/straydog-inspect", gpu,
                                      drafter=args.drafter if placement["speculative"] else None,
                                      quantization=args.quantization, host_memory_bytes=host_memory_bytes())
            try:
                model = StraydogModel(weights, context=placement["context"], state_budget_bytes=13 * 1024**3,
                                      prefill_batch_size=4096, speculative=placement["speculative"])
                worker = StraydogWorker(model, weights.resources)
                digests: list[str] = []
                forward = model._forward_tokens

                def recorded(tokens, *, logits, cancelled=lambda: False):
                    output = forward(tokens, logits=logits, cancelled=cancelled)
                    if logits and output is not None:
                        digests.append(hashlib.sha256(np.ascontiguousarray(output, np.float32).tobytes()).hexdigest())
                    return output

                model._forward_tokens = recorded

                def generate(name, ids, *, max_new, turn, checkpoint):
                    digests.clear()
                    tokens = list(model.generate(ids, max_new=max_new, turn=turn, checkpoint=checkpoint))
                    metrics = {key: model.last_metrics.get(key) for key in (
                        "cached_tokens", "resumed_from", "prefill_tokens", "generated_tokens", "decode_forward_tokens",
                        "speculative_passes", "drafted_tokens", "accepted_tokens")}
                    text = worker.tokenizer.decode(tokens, skip_special_tokens=False)
                    print(f"  {name}: {len(ids)} prompt, {metrics}", flush=True)
                    return {"name": name, "prompt": ids, "max_new": max_new, "turn": turn, "checkpoint": checkpoint,
                            "tokens": tokens, "metrics": metrics, "logits": list(digests),
                            "final_hidden": hashlib.sha256(model._final_hidden.tobytes()).hexdigest(), "text": text}

                script = requests(worker, generate)
                results[placement["name"]] = {
                    "placement": placement,
                    "resident": bool(model._resident_moe), "partial": bool(model._hybrid),
                    "host_kv": bool(model._host_attention), "prefill_batch_size": model.prefill_batch_size,
                    "speculative": model.speculative, "state_bytes": model.state_bytes,
                    "requests": [{key: value for key, value in item.items() if key != "text"} for item in script],
                }
            finally:
                weights.close()
    args.output.write_text(json.dumps({"quantization": args.quantization, "placements": results}, indent=1) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
