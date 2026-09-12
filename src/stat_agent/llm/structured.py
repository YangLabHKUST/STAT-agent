"""Structured output with a provider capability table and a prompted-JSON fallback.

Every role result is validated against a pydantic schema. Native modes (OpenAI
``json_schema``, tool-forced for Anthropic, Gemini ``response_schema``) are used where
the provider supports them; otherwise (Poe and other proxies) the schema is appended to
the prompt and the reply is parsed. On a validation failure exactly one repair call is
made; then :class:`StructuredOutputError` is raised. There is never a fabricated default.
"""

from __future__ import annotations

import json
import re
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


def extract_json(text: str) -> str:
    """Return the most plausible JSON payload inside ``text`` (fenced, or first balanced value)."""
    if text is None:
        raise StructuredOutputError("empty response")
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith(("{", "[")):
            return candidate
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
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
                    return text[start : i + 1]
        # unbalanced: fall through to try the other opener
    if text.startswith(("{", "[")):
        return text
    raise StructuredOutputError("no JSON object or array found in response", raw=text)


def parse_json_to(schema: type[T], text: str) -> T:
    payload = extract_json(text)
    try:
        return schema.model_validate_json(payload)
    except ValidationError as exc:
        raise StructuredOutputError(f"response does not match {schema.__name__}: {exc.errors()[:3]}", raw=text, errors=exc.errors()) from exc
    except ValueError as exc:  # invalid JSON
        raise StructuredOutputError(f"invalid JSON: {exc}", raw=text) from exc


def schema_instruction(schema: type[BaseModel]) -> str:
    """Instruction appended to prompted requests."""
    js = json.dumps(schema.model_json_schema(), indent=None, separators=(",", ":"))
    return (
        "\n\nRespond with a single JSON object (no prose, no markdown fences) that matches this JSON schema:\n"
        + js
    )


def repair_instruction(schema: type[BaseModel], raw: str, error: StructuredOutputError) -> str:
    detail = json.dumps(error.errors[:5]) if error.errors else str(error)
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
