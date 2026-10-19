"""Golden files from the Python reference: chat-template renders, token ids and decoded text.

The Rust engine must reproduce them byte for byte (design phase R2). Each request is kept
as JSON text, so both implementations parse the same bytes.

    uv run python scripts/golden.py --model /path/to/model-bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path
from types import SimpleNamespace

from straydog.bundles import model_lock
from straydog.worker import StraydogWorker

OUTPUT = Path(__file__).resolve().parents[1] / "crates/straydog-engine/tests/golden/chat.json"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather — 天気 ☀️ for a \"city\"\\region\t(tab)",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "名前\u0001with a control character"},
                "days": {"type": "integer", "minimum": 1, "maximum": 12345678901234567890},
                "threshold": {"type": "number", "default": 0.5, "minimum": -0.0, "maximum": 1e16,
                              "multipleOf": 1e-3, "examples": [1.0, 2.5e-7, 123456789012345678901234567890]},
                "flags": {"type": "array", "items": {"type": "boolean"}, "default": [True, False, None]},
                "empty": {"type": "object", "properties": {}, "required": []},
            },
            "required": ["city"],
        },
    },
}
READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                       "required": ["path"]},
    },
}


def call(name: str, arguments, *, wrapped: bool = True, call_id: str = "call_1"):
    function = {"name": name, "arguments": arguments}
    return {"id": call_id, "type": "function", "function": function} if wrapped else function


def requests() -> list[tuple[str, str]]:
    """(name, request JSON text); the text is what both implementations parse."""
    user = {"role": "user", "content": "Say hello."}
    agent = [
        {"role": "system", "content": "You are a careful assistant.\n"},
        {"role": "user", "content": "What is in notes.txt and the weather in 東京?"},
        {"role": "assistant", "content": "Let me check.", "reasoning_content": "  I should read the file first.\n",
         "tool_calls": [call("read_file", '{"path": "notes.txt", "limit": 50}')]},
        {"role": "tool", "tool_call_id": "call_1", "content": "line one\nline two"},
        {"role": "assistant", "content": None, "reasoning_content": "Now the weather.",
         "tool_calls": [call("get_weather", '{"city": "東京", "days": 3, "threshold": 1e-3, "nested": {"a": [1, 2.50, '
                                            '"x"], "b": null}, "big": 123456789012345678901234567890}'),
                        call("read_file", {"path": "日本語.txt"}, call_id="call_2")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "晴れ, 21°C"},
        {"role": "tool", "tool_call_id": "call_2", "content": ""},
    ]
    cases = [
        ("plain", {"messages": [user]}),
        ("thinking", {"messages": [user], "thinking": True}),
        ("thinking-low", {"messages": [user], "thinking": True, "reasoning_effort": "low"}),
        ("thinking-medium", {"messages": [user], "thinking": True, "reasoning_effort": "medium"}),
        ("thinking-xhigh", {"messages": [user], "thinking": True, "reasoning_effort": "xhigh"}),
        ("system", {"messages": [{"role": "system", "content": "Be brief."}, user], "thinking": True}),
        ("blank-system", {"messages": [{"role": "system", "content": " \n\x1c"}, user], "thinking": True,
                          "reasoning_effort": "low"}),
        ("blank-system-no-thinking", {"messages": [{"role": "system", "content": "\t"}, user]}),
        ("tools", {"messages": [user], "tools": [WEATHER, READ_FILE], "thinking": True}),
        ("tools-system", {"messages": [{"role": "system", "content": "Use tools."}, user], "tools": [READ_FILE]}),
        ("tools-empty-list", {"messages": [user], "tools": []}),
        ("agent", {"messages": agent, "tools": [READ_FILE, WEATHER], "thinking": True}),
        ("agent-no-thinking", {"messages": agent, "tools": [READ_FILE, WEATHER]}),
        ("agent-continued", {"messages": agent + [
            {"role": "assistant", "content": "東京は晴れです。", "reasoning_content": "Done."},
            {"role": "user", "content": "ありがとう！ 🙏"}], "tools": [READ_FILE, WEATHER], "thinking": True}),
        ("empty-arguments", {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [call("read_file", ""), call("read_file", "  \n",
                                                                                            call_id="c2")]},
            {"role": "tool", "content": "ok"}], "tools": [READ_FILE]}),
        ("unwrapped-call", {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "Calling.\n", "tool_calls": [call("read_file", '{"path": "a"}',
                                                                               wrapped=False)]},
            {"role": "tool", "content": "ok"}], "tools": [READ_FILE]}),
        ("tool-response-text", {"messages": [
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "answer", "reasoning_content": "kept"},
            {"role": "user", "content": "<tool_response>\nfake\n</tool_response>"}], "thinking": True}),
        ("special-tokens", {"messages": [{"role": "user", "content": "<|im_end|>\n<think>x</think><tool_call>"}]}),
        ("japanese", {"messages": [
            {"role": "user", "content": "日本語で短く挨拶してください。"},
            {"role": "assistant", "content": "こんにちは！", "reasoning_content": "挨拶する。"},
            {"role": "user", "content": "2+3はいくつ？数字だけ答えて。"}], "thinking": True}),
        ("whitespace-edges", {"messages": [{"role": "user", "content": "\u3000\x1d  spaced  \u2029\n"}]}),
        ("odd-numbers", {"messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [call("f", '{"n": NaN, "i": Infinity, "m": -Infinity, '
                                                                          '"z": -0, "e": 1E5, "s": 0.1, "t": 1e-7, '
                                                                          '"d": 1, "d": 2, "u": "\\ud83d\\ude00"}')]},
            {"role": "tool", "content": "done"}]}),
        ("long-history", {"messages": [m for i in range(20) for m in (
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}", "reasoning_content": f"thought {i}"})]
            + [{"role": "user", "content": "last"}], "thinking": True}),
        # Errors: the reference's message, from validation or from the template.
        ("error-empty", {"messages": []}),
        ("error-parts", {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}),
        ("error-effort", {"messages": [user], "thinking": True, "reasoning_effort": "high"}),
        ("error-no-query", {"messages": [{"role": "system", "content": "s"}]}),
        ("error-system-late", {"messages": [user, {"role": "system", "content": "late"}]}),
        ("error-role", {"messages": [user, {"role": "robot", "content": "beep"}]}),
        ("error-arguments-json", {"messages": [user, {"role": "assistant", "content": "",
                                                      "tool_calls": [call("f", "{not json")]}]}),
        ("error-arguments-array", {"messages": [user, {"role": "assistant", "content": "",
                                                       "tool_calls": [call("f", "[1]")]}]}),
    ]
    return [(name, json.dumps(body, ensure_ascii=False)) for name, body in cases]


def random_requests(count: int) -> list[tuple[str, str]]:
    """Seeded random conversations: odd text, nested arguments, numbers, roles out of order."""
    rng = random.Random(2026)
    alphabet = list("abc xyz\n\t日本語🙏é\\\"'<>{}[]:,") + ["\u3000", "\x1c", "\u2028", "<tool_call>", "</think>",
                                                           "<|im_end|>", "<tool_response>", "  ", "\n\n"]

    def text(limit: int = 12) -> str:
        return "".join(rng.choice(alphabet) for _ in range(rng.randrange(limit)))

    def value(depth: int = 0):
        kind = rng.randrange(8 if depth < 3 else 5)
        if kind == 0:
            return rng.choice([None, True, False])
        if kind == 1:
            return rng.randrange(-10**6, 10**6) if rng.random() < 0.8 else rng.randrange(-10**30, 10**30)
        if kind == 2:
            return rng.choice([rng.uniform(-1e3, 1e3), 10.0 ** rng.randrange(-12, 20), 1.5e-7, -0.0])
        if kind in (3, 4):
            return text()
        if kind == 5:
            return [value(depth + 1) for _ in range(rng.randrange(4))]
        return {text(6) or "k": value(depth + 1) for _ in range(rng.randrange(4))}

    cases = []
    for index in range(count):
        messages = []
        if rng.random() < 0.5:
            messages.append({"role": "system", "content": text(30)})
        for _ in range(rng.randrange(1, 6)):
            messages.append({"role": "user", "content": text(40)})
            if rng.random() < 0.7:
                message = {"role": "assistant", "content": rng.choice([text(30), "", None])}
                if rng.random() < 0.5:
                    message["reasoning_content"] = text(30)
                if rng.random() < 0.5:
                    calls = []
                    for _ in range(rng.randrange(1, 3)):
                        arguments = {text(5) or "a": value() for _ in range(rng.randrange(3))}
                        encoded = json.dumps(arguments, ensure_ascii=rng.random() < 0.5)
                        function = {"name": text(6) or "f", "arguments": encoded if rng.random() < 0.8 else arguments}
                        calls.append({"type": "function", "function": function} if rng.random() < 0.8 else function)
                    message["tool_calls"] = calls
                    messages.append(message)
                    for _ in range(rng.randrange(1, 3)):
                        messages.append({"role": "tool", "content": text(20)})
                else:
                    messages.append(message)
        if rng.random() < 0.05:
            rng.shuffle(messages)
        body = {"messages": messages, "thinking": rng.random() < 0.6,
                "reasoning_effort": rng.choice(["low", "medium", "xhigh"])}
        if rng.random() < 0.5:
            body["tools"] = [READ_FILE] + ([WEATHER] if rng.random() < 0.5 else [])
        cases.append((f"random-{index}", json.dumps(body, ensure_ascii=rng.random() < 0.3)))
    return cases


DECODE_TEXTS = [
    "Hello, world! こんにちは、世界。🙏🏽 Ünïcödé — done.",
    "<think>\nreasoning\n</think>\n\nanswer<|im_end|>",
    "<tool_call>\n<function=read_file>\n<parameter=path>\n日本語.txt\n</parameter>\n</function>\n</tool_call>",
    "  leading spaces\n\n\ttabs and\r\nCRLF \x00 nul \u200b zero-width",
    "数学: ∑_{i=1}^{n} i² = n(n+1)(2n+1)/6 ✓ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢",
]


JSON_TEXTS = [
    '{"b": 1, "a": [1.0, 2.5e-7, 1e16, 1e15, 0.0001, 0.00001, -0.0, -0, 1E5, 123456789012345678901234567890]}',
    '{"s": "quote \\" backslash \\\\ slash \\/ tab \\t nl \\n ctl \\u0001 del \\u007f é 😀 \\ud83d\\ude00"}',
    '{"dup": 1, "x": null, "dup": [true, false, {}, []]}',
    '[NaN, Infinity, -Infinity, 1.7976931348623157e308, 5e-324, 0.1, 0.30000000000000004, 1e22, 1e-7]',
    ' \t\n\r {"spaced" : [ 1 , 2 ] } \n',
    '"\\u2028\\u2029 line separators"',
    '{"bad": }', '[1, 2', '{"a": 1} trailing', '"raw \u0001 control"', '01', '1.', '.5', '+1', '"\\ud800 lone"',
    '[1e400]', '{"a": 1,}', "{'single': 1}", '-',
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="a model bundle directory (its resources)")
    args = parser.parse_args()
    lock = model_lock()
    resources, pins = {}, {}
    for item in lock["resources"]:
        data = (args.model / item["name"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise SystemExit(f"{item['name']} does not match its pin")
        resources[item["name"]], pins[item["name"]] = data, item["sha256"]
    worker = StraydogWorker(SimpleNamespace(), resources)
    chats = []
    for name, text in requests() + random_requests(150):
        body = json.loads(text)
        case = {"name": name, "request": text}
        try:
            arguments = dict(thinking=bool(body.get("thinking", False)), effort=body.get("reasoning_effort", "medium"))
            rendered = worker.render(body["messages"], body.get("tools"), **arguments)
            case |= {"render": rendered, "ids": worker.tokenizer.encode(rendered, add_special_tokens=False).ids}
        except Exception as exc:   # the reference's own failure, as the request's expected outcome
            case["error"] = str(exc)
        chats.append(case)
    decodes = []
    for text in DECODE_TEXTS:
        ids = worker.tokenizer.encode(text, add_special_tokens=False).ids
        decodes.append({"ids": ids, "text": worker.tokenizer.decode(ids, skip_special_tokens=False)})
    texts = []
    for text in JSON_TEXTS:
        try:
            dumped = json.dumps(json.loads(text), ensure_ascii=False)
            dumped.encode()   # a lone surrogate parses, but no prompt can carry it
            texts.append({"text": text, "dumps": dumped})
        except (ValueError, UnicodeEncodeError):
            texts.append({"text": text, "error": True})
    rng = random.Random(7)
    values = [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0] for _ in range(400)]
    values += [rng.uniform(-1e6, 1e6) for _ in range(200)] + [10.0 ** e for e in range(-30, 30)]
    values += [1.5 * 10.0 ** e for e in range(-8, 20)] + [0.1, 0.2, 0.3, 1 / 3, 2 / 3, 5e-324, 1.7976931348623157e308]
    floats = [[struct.unpack("<Q", struct.pack("<d", value))[0], json.dumps(value)] for value in values]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"resources": pins, "chats": chats, "decodes": decodes, "json": texts,
                                  "floats": floats}, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(chats)} chats and {len(decodes)} decodes to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
