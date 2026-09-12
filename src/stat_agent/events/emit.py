"""Emitters: where events go (a list, a callback, a JSONL file, a LangGraph stream writer)."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

from stat_agent.events.catalogue import EventBase
from stat_agent.events.encoding import dumps


class Emitter(Protocol):
    def emit(self, event: EventBase) -> None: ...


class ListEmitter:
    """Collects events in memory (tests, benchmark recorder)."""

    def __init__(self) -> None:
        self.events: list[EventBase] = []

    def emit(self, event: EventBase) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.type for e in self.events]  # type: ignore[attr-defined]

    def of_type(self, *types: str) -> list[EventBase]:
        return [e for e in self.events if e.type in types]  # type: ignore[attr-defined]


class CallbackEmitter:
    def __init__(self, fn: Callable[[EventBase], None]) -> None:
        self._fn = fn

    def emit(self, event: EventBase) -> None:
        self._fn(event)


class JsonlSink:
    """Append-only JSONL file, one event per line, thread-safe, flushed per event."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: EventBase) -> None:
        line = dumps(event.model_dump(mode="json"))
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read(self) -> list[dict]:
        import json

        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


class MultiEmitter:
    def __init__(self, emitters: Iterable[Emitter]) -> None:
        self._emitters = list(emitters)

    def emit(self, event: EventBase) -> None:
        for e in self._emitters:
            e.emit(event)


class StampingEmitter:
    """Wraps an emitter, filling ``seq``/``turn_id``/``step`` on every event."""

    def __init__(self, inner: Emitter, turn_id: str | None = None) -> None:
        self._inner = inner
        self._seq = 0
        self._lock = threading.Lock()
        self.turn_id = turn_id
        self.step: int | None = None

    def emit(self, event: EventBase) -> None:
        with self._lock:
            self._seq += 1
            event.seq = self._seq
        if event.turn_id is None:
            event.turn_id = self.turn_id
        if event.step is None:
            event.step = self.step
        self._inner.emit(event)

    @property
    def seq(self) -> int:
        return self._seq


__all__ = ["Emitter", "ListEmitter", "CallbackEmitter", "JsonlSink", "MultiEmitter", "StampingEmitter"]
