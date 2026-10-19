"""The official template's raw output split into reasoning, answer and structured tool calls."""

from __future__ import annotations

import json

from straydog.chat import ReasoningSplitter, ToolCallSplitter

_READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        },
    },
}


def split(deltas, *, thinking=False, tools=None):
    """(reasoning deltas, answer deltas, tool calls) for a stream of raw deltas."""
    reasoning, tool_calls = ReasoningSplitter(thinking=thinking), ToolCallSplitter(tools)
    thoughts, answer = [], []
    for delta in deltas:
        content, thought = reasoning.feed(delta)
        thoughts.append(thought)
        answer.append(tool_calls.feed(content))
    thoughts.append(reasoning.finish())
    late, calls = tool_calls.finish()
    return thoughts, answer + [late], calls


def test_reasoning_before_a_split_close_tag_stays_out_of_the_answer():
    # The template's generation prompt opens <think>, so the stream starts as
    # reasoning with no opening tag; the close tag may be split across deltas.
    thoughts, answer, _ = split(["Let me think", " about it.\n</th", "ink>\n\n", "Hello", " there"], thinking=True)
    assert "".join(thoughts).rstrip() == "Let me think about it."
    assert "".join(answer) == "Hello there"


def test_a_quoted_close_tag_stays_inside_reasoning():
    # Reasoning about the template itself quotes "</think>" mid-line; it decodes to the same
    # token as the real close, which the template only ever writes at a line start.
    deltas = [
        "It renders `\\n</think>\\n\\n` for assistant messages,",
        " and </think> again.\n",
        "</think>\n\nThe cache misses because",
        " reasoning is re-rendered.",
    ]
    thoughts, answer, _ = split(deltas, thinking=True)
    reasoning = "".join(thoughts)
    assert reasoning.endswith("and </think> again.\n") and "for assistant messages" in reasoning
    assert "".join(answer) == "The cache misses because reasoning is re-rendered."


def test_a_held_tail_is_released_when_the_stream_ends_while_reasoning():
    thoughts, answer, _ = split(["cut off at </thi"], thinking=True)
    assert "".join(thoughts) == "cut off at </thi" and not "".join(answer)


def test_native_tool_call_markup_becomes_structured_calls():
    # Prose may precede the call; the opener and parameters can split anywhere.
    deltas = [
        "Let me look.\n\n<tool",
        "_call>\n<function=read_file>\n<parameter=path>\n123\n</param",
        "eter>\n<parameter=limit>\n50\n</parameter>\n</function>\n</tool_call>",
    ]
    _, answer, calls = split(deltas, tools=[_READ_FILE_TOOL])
    assert "".join(answer) == "Let me look.\n\n"
    assert [(call["index"], call["name"]) for call in calls] == [(0, "read_file")]
    assert calls[0]["id"].startswith("call_")
    # A string-typed parameter stays text even when it looks like JSON.
    assert json.loads(calls[0]["arguments"]) == {"path": "123", "limit": 50}


def test_a_quoted_tool_call_opener_streams_as_prose():
    deltas = ["Qwen wraps calls in <tool_call> blocks.\n", "<tool_call>\n<function=read_file>\n</function>\n</tool_call>"]
    _, answer, calls = split(deltas, tools=[_READ_FILE_TOOL])
    assert answer[0] == "Qwen wraps calls in <tool_call> blocks.\n"
    assert [call["name"] for call in calls] == ["read_file"]


def test_markup_that_does_not_parse_is_released_as_text():
    deltas = ["Example: <tool_call><function=read_file>oops</tool_call> done"]
    _, answer, calls = split(deltas, tools=[_READ_FILE_TOOL])
    assert "".join(answer) == deltas[0] and not calls
    strict = "<tool_call>\n<function=read_file><parameter=path>a</parameter></function>\n</tool_call>"
    for text in ("```\n" + strict + "\n```", strict[:-12], strict.replace("</function>", "<parameter=path>b</parameter></function>")):
        _, answer, calls = split([text], tools=[_READ_FILE_TOOL])
        assert not calls and "".join(answer) == text


def test_markup_is_left_alone_without_tools():
    deltas = ["<tool_call>\n<function=read_file>\n</function>\n</tool_call>"]
    _, answer, calls = split(deltas)
    assert "".join(answer) == deltas[0] and not calls
