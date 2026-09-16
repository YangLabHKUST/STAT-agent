"""Structured output with a provider capability table and a prompted-JSON fallback.

Every role result is validated against a pydantic schema. Native modes (OpenAI
``json_schema``, tool-forced for Anthropic, Gemini ``response_schema``) are used where
the provider supports them; otherwise (Poe and other proxies) the schema is appended to
the prompt and the reply is parsed. On a validation failure exactly one repair call is
made; then :class:`StructuredOutputError` is raised. There is never a fabricated default.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_CAPABILITIES: dict[str, str] = {
    "openai": "json_schema",
    "anthropic": "native",  # LangChain uses forced tool calling
    "google": "native",
    "deepseek": "json_mode",
    "xai": "json_schema",
    "poe": "prompted",
    "moonshot": "prompted",
    "zhipu": "prompted",
    "dashscope": "prompted",
    "custom": "prompted",
}


class StructuredOutputError(ValueError):
    def __init__(self, message: str, raw: str = "", errors: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.raw = raw
        self.errors = errors or []


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)
_FENCE_OPEN_RE = re.compile(r"```(?:json)?[ \t]*\n", re.I)


def _balanced_payloads(text: str, opener: str, start: int = 0) -> Iterator[str]:
    """Every balanced top-level ``{...}`` (or ``[...]``) in ``text`` from ``start``, in order of appearance."""
    closer = "}" if opener == "{" else "]"
    i = text.find(opener, start)
    while i >= 0:
        depth = 0
        in_str = False
        esc = False
        end = -1
        for j in range(i, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end < 0:
            i = text.find(opener, i + 1)  # unbalanced from here: try the next opener
            continue
        yield text[i : end + 1]
        i = text.find(opener, end + 1)


def iter_json_candidates(text: str) -> Iterator[str]:
    """The plausible JSON payloads inside ``text``, most plausible first: a closed fenced block, then the
    balanced objects (then arrays) after an unclosed fence, then those anywhere in the text. A reply that
    explains itself before the JSON, quotes a Python fragment in the explanation, or runs past the block
    without closing the fence still yields its JSON."""
    if text is None:
        raise StructuredOutputError("empty response")
    text = text.strip()
    seen: set[str] = set()
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith(("{", "[")):
            seen.add(candidate)
            yield candidate
    starts = [0]
    mo = _FENCE_OPEN_RE.search(text)
    if mo and not m:
        starts.insert(0, mo.end())
    for start in starts:
        for opener in ("{", "["):
            for candidate in _balanced_payloads(text, opener, start):
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate
    if text.startswith(("{", "[")) and text not in seen:
        yield text


def extract_json(text: str) -> str:
    """Return the most plausible JSON payload inside ``text`` (fenced, or first balanced value)."""
    for candidate in iter_json_candidates(text):
        return candidate
    raise StructuredOutputError("no JSON object or array found in response", raw=text)


def _python_literal(payload: str) -> Any:
    """A dict or list written as a Python literal (single quotes, True/None), or None."""
    try:
        value = ast.literal_eval(payload)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, (dict, list)) else None


def _names_a_field(schema: type[BaseModel], obj: Any) -> bool:
    """A non-empty object is a candidate answer only if it names at least one field of the schema; an
    unrelated fragment quoted in the prose (``{'group': 'x'}``) must not become an all-default answer."""
    if isinstance(obj, dict) and obj:
        fields = getattr(schema, "model_fields", None) or {}
        return not fields or bool(set(obj) & set(fields))
    return True


def parse_json_to(schema: type[T], text: str) -> T:
    """Validate the first candidate payload in ``text`` that matches ``schema``.

    A payload written as a Python literal is accepted as well. When no candidate matches, the error
    describes the first candidate, which is the reply's most plausible payload."""
    first_error: StructuredOutputError | None = None
    for payload in iter_json_candidates(text):
        try:
            obj: Any = json.loads(payload)
        except ValueError as exc:
            obj = _python_literal(payload)
            if obj is None:
                error = StructuredOutputError(f"invalid JSON: {exc}", raw=text)
                first_error = first_error or error
                continue
        if not _names_a_field(schema, obj):
            first_error = first_error or StructuredOutputError(f"response does not match {schema.__name__}: the object names none of its fields", raw=text)
            continue
        try:
            return schema.model_validate(obj)
        except ValidationError as exc:
            first_error = first_error or StructuredOutputError(f"response does not match {schema.__name__}: {exc.errors()[:3]}", raw=text, errors=exc.errors())
    if first_error is None:
        raise StructuredOutputError("no JSON object or array found in response", raw=text)
    raise first_error


def schema_instruction(schema: type[BaseModel]) -> str:
    """Instruction appended to prompted requests."""
    js = json.dumps(schema.model_json_schema(), indent=None, separators=(",", ":"))
    return (
        "\n\nRespond with a single JSON object (no prose, no markdown fences) that matches this JSON schema:\n"
        + js
    )


def repair_instruction(schema: type[BaseModel], raw: str, error: StructuredOutputError) -> str:
    detail = json.dumps(error.errors[:5], default=str) if error.errors else str(error)
    return (
        "Your previous reply could not be parsed as valid JSON for the required schema.\n"
        f"Validation errors: {detail}\n\nPrevious reply:\n{raw[:4000]}\n\n"
        f"Return ONLY the corrected JSON object matching this schema:\n{json.dumps(schema.model_json_schema(), separators=(',', ':'))}"
    )


def native_method_for(provider: str, capability: str) -> str | None:
    """The ``with_structured_output(method=...)`` value for a capability, or None for prompted."""
    if capability == "prompted":
        return None
    if capability == "json_schema":
        return "json_schema"
    if capability == "json_mode":
        return "json_mode"
    return None  # provider default (function calling / native)


def usage_of(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)


__all__ = [
    "DEFAULT_CAPABILITIES",
    "StructuredOutputError",
    "extract_json",
    "parse_json_to",
    "schema_instruction",
    "repair_instruction",
    "native_method_for",
    "usage_of",
]
