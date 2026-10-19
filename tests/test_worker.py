"""The worker's bounded stdio protocol, template rendering and generation limits."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from straydog.model import StraydogModel
from straydog.weights import StraydogWeights
from straydog.worker import StraydogWorker, main
from straydog.worker import _template_message as straydog_template_message


def worker_with_fake_model():
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    model = SimpleNamespace(
        context=16, history=[], reset=lambda: None, weights=SimpleNamespace(gpu=SimpleNamespace(info=lambda: {}))
    )

    def generate(ids, **kwargs):
        assert kwargs["max_new"] == 14
        yield 1
        yield 2

    model.generate = generate
    resources = {"tokenizer.json": tok.to_str().encode(), "chat_template.jinja": b"{{ messages[0].content }} world"}
    return StraydogWorker(model, resources)


def test_worker_renders_and_streams_utf8_and_clears_busy_before_terminal():
    worker = worker_with_fake_model()
    frames = []

    def send(frame):
        if frame["event"] == "done":
            assert worker._active is None
        frames.append(frame)

    worker._send = send
    assert worker.encode([{"role": "user", "content": "hello"}]) == [1, 2]
    worker._active = "id"
    worker._generate(
        {
            "id": "id",
            "body": {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": None,
                "profile_python": True,
            },
        }
    )
    assert "".join(x.get("content", "") for x in frames) == "hello world"
    assert frames[-1]["usage"]["completion_tokens"] == 2
    assert "function calls" in frames[-1]["profile"]["python"]
    with pytest.raises(ValueError, match="text"):
        worker.encode([{"content": []}])


def test_worker_decodes_json_tool_arguments_for_the_template():
    # The official template iterates ``arguments|items``; agent history carries
    # OpenAI-style JSON strings, which the sandboxed filter rejects.
    worker = worker_with_fake_model()
    worker.template = worker.template.environment.from_string(
        "{% for call in messages[0].tool_calls %}{% for k, v in call.function.arguments|items %}"
        "{{ k }}={{ v }}{% endfor %}{% endfor %}"
    )
    calls = [{"type": "function", "id": "t1", "function": {"name": "read_file", "arguments": '{"hello": "world"}'}}]
    message = {"role": "assistant", "content": None, "tool_calls": calls}
    assert worker.template.render(messages=[straydog_template_message(message)]) == "hello=world"
    assert calls[0]["function"]["arguments"] == '{"hello": "world"}'
    for bad, match in (("{not json", "valid JSON"), ("[1]", "JSON object")):
        bad_calls = [{"function": {"name": "f", "arguments": bad}}]
        with pytest.raises(ValueError, match=match):
            worker.encode([{"role": "assistant", "content": None, "tool_calls": bad_calls}])


def test_worker_frames_are_utf8_whatever_the_pipe_text_encoding(monkeypatch):
    worker = worker_with_fake_model()
    raw = io.BytesIO()
    # A Windows pipe's text layer encodes with the ANSI code page (cp932 on Japanese Windows).
    monkeypatch.setattr("sys.stdout", io.TextIOWrapper(raw, encoding="cp932"))
    worker._send({"id": "id", "event": "delta", "content": "こんにちは"})
    assert json.loads(raw.getvalue().decode("utf-8")) == {"id": "id", "event": "delta", "content": "こんにちは"}


def test_worker_profiles_prefill_and_decode_separately():
    worker, frames = worker_with_fake_model(), []
    worker._send = frames.append

    def generate(ids, **kwargs):
        worker.model.stats["work"] = 2.0
        worker.model.weights.gpu.stats["work"] = 7.0
        yield 1
        worker.model.stats["work"] += 3.0
        worker.model.weights.gpu.stats["work"] += 4.0
        yield 2

    worker.model.generate = generate
    worker._generate({"id": "phase", "body": {"messages": [{"content": "hello"}], "profile": True}})
    profile = frames[-1]["profile"]
    assert profile["prefill"] == {"model": {"work": 2.0}, "gpu": {"work": 7.0}}
    assert profile["decode"] == {"model": {"work": 3.0}, "gpu": {"work": 4.0}}


def test_worker_serve_tokenize_and_bounded_protocol(monkeypatch):
    worker = worker_with_fake_model()
    worker.model.weights = SimpleNamespace(quantization="Q2_0", gpu=SimpleNamespace(info=lambda: {}))
    worker.model.state_bytes = 1
    monkeypatch.setattr(
        "sys.stdin",
        SimpleNamespace(
            buffer=io.BytesIO(
                b'{"id":"a","command":"tokenize","text":"hello world"}\n{"id":"b","command":"shutdown"}\n'
            )
        ),
    )
    frames = []
    worker._send = frames.append
    worker.serve()
    assert frames[0]["event"] == "ready" and frames[1]["tokens"] == 2
    monkeypatch.setattr("sys.argv", ["straydog", "--help"])
    with pytest.raises(SystemExit) as exit_code:
        main()
    assert exit_code.value.code == 0


def test_worker_error_resets_partially_mutated_state():
    worker = worker_with_fake_model()
    worker.model.reset = lambda: worker.model.history.clear()

    def broken(*args, **kwargs):
        worker.model.history.append(1)
        raise RuntimeError("GPU failure")

    worker.model.generate = broken
    frames = []
    worker._send = frames.append
    worker._generate({"id": "id", "body": {"messages": [{"content": "hello"}], "max_tokens": 1}})
    assert not worker.model.history and frames[-1]["fatal"]


def test_worker_ends_the_request_when_the_failed_device_cannot_reset():
    # An illegal memory access poisons the device: reset fails too, yet the server must get a terminal frame.
    worker = worker_with_fake_model()

    def poisoned(*args, **kwargs):
        raise RuntimeError("Straydog GPU: an illegal memory access was encountered")

    worker.model.generate = poisoned
    worker.model.reset = poisoned
    frames = []
    worker._send = frames.append
    worker._generate({"id": "id", "body": {"messages": [{"content": "hello"}], "max_tokens": 1}})
    assert frames[-1]["event"] == "error" and frames[-1]["fatal"] and "illegal memory access" in frames[-1]["message"]
    assert worker._active is None


def test_a_value_error_whose_reset_fails_is_still_fatal():
    worker = worker_with_fake_model()

    def rejected(*args, **kwargs):
        raise ValueError("Straydog prompt exceeds the context")

    def poisoned():
        raise RuntimeError("device lost")

    worker.model.generate, worker.model.reset = rejected, poisoned
    frames = []
    worker._send = frames.append
    worker._generate({"id": "id", "body": {"messages": [{"content": "hello"}], "max_tokens": 1}})
    assert frames[-1]["event"] == "error" and frames[-1]["fatal"]


def test_generation_limits_cancellation_and_prefix_reuse():
    model = object.__new__(StraydogModel)
    model.context, model.history, model._checkpoints = 8, [], {}
    model.weights = SimpleNamespace(check_unchanged=lambda: None)
    model.reset = lambda: model.history.clear()

    def step(token, *, logits=True):
        model.history.append(token)
        return np.array([0, 2, 1], np.float32) if logits else None

    model.step = step
    assert list(model.generate([2, 1], max_new=2)) == [1, 1]
    assert model.history == [2, 1, 1]
    assert model.last_metrics["prefill_tokens"] == 2
    assert model.last_metrics["decode_forward_tokens"] == 1
    assert model.last_metrics["prefill_seconds"] <= model.last_metrics["ttft_seconds"]
    assert model.last_metrics["decode_tokens_per_second"] > 0
    progress = []
    assert list(model.generate([2, 1, 1, 2], max_new=1, progress=lambda n, total: progress.append(n))) == [1]
    assert progress == [4]
    assert list(model.generate([2], max_new=1, cancelled=lambda: True)) == []
    with pytest.raises(ValueError, match="context"):
        list(model.generate([1] * 8, max_new=1))
    with pytest.raises(ValueError, match="temperature"):
        list(model.generate([1], max_new=1, temperature=float("nan")))


def test_weights_slice_geometry_and_change_detection(tmp_path):
    import mmap

    path = tmp_path / "weights"
    path.write_bytes(np.arange(8, dtype="<f4").tobytes())
    weights = object.__new__(StraydogWeights)
    file = path.open("r+b")
    weights._files, weights._maps = [file], [mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)]
    weights._vectors = {}
    weights._projection_cache = {}
    weights._tensors = {"matrix": (0, {"dimensions": [4, 2], "bytes": 32, "offset": 0, "quant_type": 0})}
    stat = path.stat()
    weights._signatures = [(stat.st_size, stat.st_mtime_ns)]
    weights.gpu = SimpleNamespace(
        project=lambda key, factory, rows, cols, vector: factory()[0].view("<f4").reshape(rows, cols) @ vector
    )
    assert weights.shape("matrix") == (4, 2)
    np.testing.assert_equal(weights.row("matrix", 1), [4, 5, 6, 7])
    np.testing.assert_equal(weights.vector("matrix"), np.arange(8).reshape(2, 4))
    np.testing.assert_equal(weights.project("matrix", np.ones(4, np.float32)), [6, 22])
    batches = []

    def many(requests):
        batches.append(len(requests))
        return [weights.gpu.project(*request) for request in requests]

    weights.gpu.project_many = many
    results = weights.project_many([("matrix", np.ones(4, np.float32), None)] * 40)
    assert batches == [32, 8]
    assert all(result.tolist() == [6, 22] for result in results)
    weights._tensors["scalar"] = (0, {"dimensions": [4], "bytes": 16, "offset": 0, "quant_type": 0})
    mixed = weights.project_many(
        [
            ("scalar", np.ones(4, np.float32), None),
            ("matrix", np.ones(4, np.float32), None),
            ("scalar", np.full(4, 2, np.float32), None),
        ]
    )
    assert [result.tolist() for result in mixed] == [[6], [6, 22], [12]]
    assert batches[-1] == 1
    assert weights.project_many([("scalar", np.ones(4, np.float32), None)])[0].tolist() == [6]
    with pytest.raises(ValueError, match="scalar projection"):
        weights.project_many([("scalar", np.ones(3, np.float32), None)])
    weights.check_unchanged()
    file.write(b"changed")
    file.flush()
    import os

    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(RuntimeError, match="changed"):
        weights.check_unchanged()
    with pytest.raises(ValueError, match="range"):
        weights.row("matrix", 2)
    weights.close()


def test_state_capacity_accounts_for_every_owned_persistent_buffer(monkeypatch):
    monkeypatch.setattr(StraydogModel, "_validate_geometry", lambda self: None)
    weights = SimpleNamespace(context_limit=262144)
    model = StraydogModel(weights, context=32)
    actual = model._ple_history.nbytes
    actual += sum(array.nbytes for pair in model._linear.values() for array in pair)
    actual += sum(array.nbytes for group in model._attention.values() for array in group)
    assert model.state_bytes == actual
    with pytest.raises(ValueError, match="RAM budget"):
        StraydogModel(weights, context=32, state_budget_bytes=actual - 1)
    model.reset()
    assert not model.history


def test_worker_loading_frames_send_stage_changes_and_completion_but_thin_the_rest(monkeypatch):
    from straydog.worker import _Loading

    output = io.BytesIO()
    monkeypatch.setattr("sys.stdout", SimpleNamespace(buffer=output))
    loading = _Loading()
    loading("authenticate", 0, 10)
    loading("authenticate", 5, 10)   # a moment later: dropped
    loading("authenticate", 10, 10)  # completion: always sent
    loading("prepare")
    frames = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [(frame["stage"], frame["done"]) for frame in frames] == [
        ("authenticate", 0), ("authenticate", 10), ("prepare", 0)]
    assert all(frame["event"] == "loading" for frame in frames)
