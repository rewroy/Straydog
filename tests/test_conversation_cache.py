"""Straydog resumes from the longest prefix it holds: the live sequence or a conversation checkpoint."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from straydog import model as straydog_model
from straydog.model import StraydogModel

START, END = straydog_model._IM_START, straydog_model._IM_END


def recurrent_model(monkeypatch):
    """A one-token model whose recurrent state (a running hash in host state) depends on every token."""
    monkeypatch.setattr(straydog_model, "_ROOT_MINIMUM", 8)
    model = object.__new__(StraydogModel)
    model.context, model.history, model._checkpoints = 256, [], {}
    model.weights = SimpleNamespace(check_unchanged=lambda: None)
    model.prefill_batch_size, model._gpu_recurrence = 1, False
    model._linear, model._attention = {}, {}
    model._ple_history = np.zeros((10, 10240), np.float32)
    model._final_hidden = None
    model.resets = 0

    def reset():
        model.history.clear()
        model._ple_history.fill(0)
        model._final_hidden = None
        model.resets += 1

    def step(token, *, logits=True):
        model._ple_history[0, 0] = (model._ple_history[0, 0] * 31 + token) % 65521
        model.history.append(token)
        model._final_hidden = np.array([model._ple_history[0, 0]], np.float32)
        return np.array([1, 0], np.float32) if logits else None

    model.reset, model.step = reset, step
    return model


def fresh_state(monkeypatch, tokens):
    """The state after reading ``tokens`` from the start (one generated token is not read back)."""
    model = recurrent_model(monkeypatch)
    for token in tokens:
        model.step(token)
    return model._ple_history[0, 0]


def prompt(system, user):
    # <|im_start|>system ... <|im_end|> <|im_start|>user ... <|im_end|> <|im_start|>assistant (generation prompt)
    return [START, *system, END, START, *user, END, START, 7, 8]


def test_a_new_user_turn_resumes_where_the_last_reply_began(monkeypatch):
    model = recurrent_model(monkeypatch)
    first = prompt(range(10, 30), [41, 42, 43])
    list(model.generate(first, max_new=2))
    assert set(model._checkpoints) == {"root", "turn"}
    turn = model._checkpoints["turn"].position
    assert turn % 4 == 0 and turn <= len(first) - 3 and model._checkpoints["root"].position % 4 == 0
    # The client re-renders that reply differently (its reasoning dropped), then adds a user message.
    second = first + [5, 6, END, START, 44, END, START, 7, 8]
    second[len(first) - 1:len(first) + 1] = [9, 9]
    list(model.generate(second, max_new=1))
    assert model.last_metrics["resumed_from"] == "turn" and model.last_metrics["cached_tokens"] == turn
    assert model.resets == 0
    # Restoring is the same as never having left the checkpoint.
    assert model._ple_history[0, 0] == fresh_state(monkeypatch, second)
    assert "prior" in model._checkpoints and model._checkpoints["prior"].position == turn


def test_a_new_chat_resumes_after_the_shared_system_prompt_and_others_start_over(monkeypatch):
    model = recurrent_model(monkeypatch)
    list(model.generate(prompt(range(10, 30), [41]), max_new=1))
    root = model._checkpoints["root"].position
    other_chat = prompt(range(10, 30), [99, 98, 97])
    list(model.generate(other_chat, max_new=1))
    assert model.last_metrics["resumed_from"] == "root" and model.last_metrics["cached_tokens"] == root
    assert model._ple_history[0, 0] == fresh_state(monkeypatch, other_chat)
    list(model.generate(prompt(range(50, 70), [41]), max_new=1))
    assert model.last_metrics["resumed_from"] == "none" and model.resets == 1


def test_tool_results_keep_their_own_checkpoint_and_one_off_requests_keep_none(monkeypatch):
    model = recurrent_model(monkeypatch)
    first = prompt(range(10, 30), [41])
    list(model.generate(first, max_new=1))
    turn = model._checkpoints["turn"]
    tool = first + [5, END, START, 61, 62, END, START, 7, 8]
    list(model.generate(tool, max_new=1, turn=False))
    assert model.last_metrics["resumed_from"] == "live"
    assert model._checkpoints["turn"] is turn and model._checkpoints["step"].position > turn.position
    slots = {checkpoint.slot for checkpoint in model._checkpoints.values()}
    assert len(slots) == len(model._checkpoints)
    one_off = prompt(range(10, 30), [70, 71, 72, 73, 74])
    list(model.generate(one_off, max_new=1, checkpoint=False))
    assert set(model._checkpoints) == {"root", "turn", "step"} and model._checkpoints["turn"] is turn


def test_a_fully_held_prompt_still_runs_its_last_token(monkeypatch):
    model = recurrent_model(monkeypatch)
    tokens = prompt(range(10, 30), [41])
    list(model.generate(tokens, max_new=1))
    list(model.generate(tokens, max_new=1))   # the same prompt again: resume below its end
    assert model.last_metrics["resumed_from"] == "turn" and model.last_metrics["prefill_tokens"] >= 1
    assert model._ple_history[0, 0] == fresh_state(monkeypatch, tokens)


@pytest.mark.parametrize("edit", [3, 23])
def test_checkpoints_are_used_only_when_the_prompt_matches_their_tokens(monkeypatch, edit):
    model = recurrent_model(monkeypatch)
    tokens = prompt(range(10, 30), [41, 42])
    list(model.generate(tokens, max_new=1))
    edited = list(tokens)
    edited[edit] = 1000   # inside the system prompt, or inside the user message (before the reply began)
    list(model.generate(edited, max_new=1))
    expected = "none" if edit < model._checkpoints["root"].position else "root"
    assert model.last_metrics["resumed_from"] == expected
    assert model._ple_history[0, 0] == fresh_state(monkeypatch, edited)
