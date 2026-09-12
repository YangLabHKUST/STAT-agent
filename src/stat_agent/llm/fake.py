"""Fixture-backed fake LLM: deterministic replay for tests and benchmarks.

A fixture is keyed by ``(role, prompt_id, sha256(canonical normalised input))`` - never by
the prompt text - so wording edits replay unchanged while semantic changes (new
``prompt_id`` version or different inputs) require re-recording.

Modes (``STAT_FAKE_LLM``): ``replay`` (miss -> :class:`FakeMiss` with the re-record hint),
``record`` (call the real model, then store), ``strict`` (replay; every fixture used at most
once and reported at teardown), ``off``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class FakeMiss(LookupError):
    def __init__(self, role: str, key: str, digest: str):
        self.role, self.key, self.digest = role, key, digest
        super().__init__(
            f"no recorded LLM fixture for role={role} key={key} (input digest: {digest}). "
            f"Run with STAT_FAKE_LLM=record (and real credentials) to record it."
        )


def _normalise(value: Any) -> Any:
    """Canonicalise role inputs: whitespace, floats, paths, and order-insensitive lists."""
    if isinstance(value, str):
        s = re.sub(r"\s+", " ", value).strip()
        s = re.sub(r"(/[\w./-]+/)([\w.-]+\.(h5ad|csv|tsv|tif|tiff|npy|txt|json))", r"<path:\2>", s)
        return s
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items()) if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        items = [_normalise(v) for v in value]
        if all(isinstance(i, (str, int, float)) for i in items):
            return sorted(items, key=str)
        return items
    if hasattr(value, "model_dump"):
        return _normalise(value.model_dump(mode="json"))
    return value


def fixture_key(role: str, prompt_id: str | None, norm_input: Any) -> str:
    payload = json.dumps({"role": role, "prompt_id": prompt_id, "input": _normalise(norm_input)}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class FixtureStore:
    root: Path
    mode: str = field(default_factory=lambda: os.environ.get("STAT_FAKE_LLM", "replay"))
    used: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)

    def path_for(self, role: str, key: str) -> Path:
        return self.root / role / f"{key}.json"

    def lookup(self, role: str, key: str) -> dict[str, Any] | None:
        p = self.path_for(role, key)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        self.used.append(f"{role}/{key}")
        return data

    def store(self, role: str, key: str, record: dict[str, Any]) -> Path:
        p = self.path_for(role, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return p

    def miss(self, role: str, key: str, digest: str) -> FakeMiss:
        self.misses.append(f"{role}/{key}")
        return FakeMiss(role, key, digest)


__all__ = ["FakeMiss", "FixtureStore", "fixture_key", "_normalise"]
