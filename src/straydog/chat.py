"""The official chat template's output, split as an OpenAI-compatible answer.

Generation is raw template text: reasoning up to ``</think>``, then the answer,
then optional ``<tool_call>`` markup. These splitters stream the reasoning and
answer as separate deltas and turn the markup into structured tool calls.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

_THINK_CLOSE = "</think>"
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_TOOL_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def _split_tag_tail(buf: str, tag: str) -> int:
    """Length of the tail of ``buf`` that could be the start of a split ``tag``."""
    return next((n for n in range(min(len(buf), len(tag) - 1), 0, -1) if tag.startswith(buf[-n:])), 0)


def _line_tag(buf: str, tag: str, *, line_start: bool) -> int:
    """Index of the first ``tag`` in ``buf`` that opens a line, else -1.

    The template only ever writes its control tags at a line start, while a
    model discussing them (say, reasoning about this very template) writes
    them mid-line. Both decode to the same token, so position is the only
    signal. ``line_start`` says whether the text before ``buf`` ended a line.
    """
    at = buf.find(tag)
    while at >= 0:
        if buf[at - 1] == "\n" if at else line_start:
            return at
        at = buf.find(tag, at + 1)
    return -1


class ReasoningSplitter:
    """Split the raw stream into answer and reasoning at the template's close tag.

    With thinking on, the official template ends the prompt inside an open
    ``<think>`` block, so generation starts as reasoning and only ``</think>``
    marks the answer. No opening tag ever reaches the stream for a client to
    find, so the split has to happen here. The template
    closes reasoning as ``\n</think>``; a mid-line ``</think>`` is the model
    quoting the tag and stays reasoning.
    """

    def __init__(self, *, thinking: bool) -> None:
        self._reasoning = thinking
        self._carry = ""
        self._answer_lead = True
        self._line_start = True  # the prompt ends with "<think>\n"

    def feed(self, text: str) -> tuple[str, str]:
        """Return ``(content, reasoning)`` for one delta."""
        if not self._reasoning:
            return self._answer(text), ""
        buf, self._carry = self._carry + text, ""
        end = _line_tag(buf, _THINK_CLOSE, line_start=self._line_start)
        if end >= 0:
            self._reasoning = False
            return self._answer(buf[end + len(_THINK_CLOSE) :]), buf[:end]
        # Hold back a tail that could be the start of a split close tag.
        if hold := _split_tag_tail(buf, _THINK_CLOSE):
            buf, self._carry = buf[:-hold], buf[-hold:]
        if buf:
            self._line_start = buf.endswith("\n")
        return "", buf

    def finish(self) -> str:
        """Release a held tail; the stream ended while still reasoning."""
        carry, self._carry = self._carry, ""
        return carry

    def _answer(self, text: str) -> str:
        # The template separates reasoning from the answer with blank lines.
        if self._answer_lead:
            text = text.lstrip()
            self._answer_lead = not text
        return text


class ToolCallSplitter:
    """Turn the template's native tool-call markup into structured calls.

    Qwen answers a tool turn as optional prose followed by
    ``<tool_call><function=…><parameter=…>…</parameter></function></tool_call>``
    blocks (the template forbids any suffix). The prose streams, the markup is
    held back from the answer and parsed at the end. Only an opener at a line
    start counts (the template's own placement); a mid-line one is prose
    quoting the format. Markup that does not parse is released as text, so the
    client's own recovery still sees it.
    """

    def __init__(self, tools: list[dict[str, Any]] | None) -> None:
        self._string_params: dict[str, set[str]] = {}
        for tool in tools or []:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                continue
            properties = (function.get("parameters") or {}).get("properties") or {}
            self._string_params[function["name"]] = {
                name for name, schema in properties.items() if _is_string_schema(schema)
            }
        self._enabled = bool(tools)
        self._carry = ""
        self._markup: str | None = None
        self._line_start = True

    def feed(self, text: str) -> str:
        """Return the answer text that is safe to show for one delta."""
        if not self._enabled:
            return text
        if self._markup is not None:
            self._markup += text
            return ""
        buf, self._carry = self._carry + text, ""
        start = _line_tag(buf, _TOOL_CALL_OPEN, line_start=self._line_start)
        if start >= 0:
            self._markup = buf[start:]
            return buf[:start]
        if hold := _split_tag_tail(buf, _TOOL_CALL_OPEN):
            buf, self._carry = buf[:-hold], buf[-hold:]
        if buf:
            self._line_start = buf.endswith("\n")
        return buf

    def finish(self) -> tuple[str, list[dict[str, Any]]]:
        """Return ``(late answer text, tool calls)`` once the stream ends."""
        carry, self._carry = self._carry, ""
        if self._markup is None:
            return carry, []
        calls = self._parse(self._markup)
        if calls is None:
            return carry + self._markup, []
        return carry, calls

    def _parse(self, markup: str) -> list[dict[str, Any]] | None:
        blocks = list(_TOOL_CALL_BLOCK_RE.finditer(markup))
        if not blocks or _TOOL_CALL_BLOCK_RE.sub("", markup).strip():
            return None
        calls = []
        for index, block in enumerate(blocks):
            name, body = block[1], block[2]
            parameters = list(_TOOL_PARAMETER_RE.finditer(body))
            if _TOOL_PARAMETER_RE.sub("", body).strip():
                return None
            arguments: dict[str, Any] = {}
            for parameter in parameters:
                key, value = parameter[1], _template_value(parameter[2])
                if key in arguments:
                    return None
                arguments[key] = value if key in self._string_params.get(name, ()) else _json_or_text(value)
            calls.append(
                {
                    "index": index,
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return calls


def _is_string_schema(schema: Any) -> bool:
    kind = schema.get("type") if isinstance(schema, dict) else None
    kinds = set(kind) if isinstance(kind, list) else {kind}
    return "string" in kinds and kinds <= {"string", "null"}


def _template_value(raw: str) -> str:
    """Undo the newline the template writes on each side of a value."""
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    return raw


def _json_or_text(value: str) -> Any:
    # The template writes non-strings with ``tojson``; anything else is text.
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
