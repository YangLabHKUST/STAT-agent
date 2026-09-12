"""The per-session ordered event log: global seq, ring buffer, subscribers, JSONL, SSE frames."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from pathlib import Path

from stat_agent.events.catalogue import EventBase, to_wire
from stat_agent.events.encoding import dumps


class EventLog:
    def __init__(self, session_dir: Path | None = None, max_frames: int = 5000, max_bytes: int = 16 * 1024 * 1024) -> None:
        self.session_dir = session_dir
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.seq = 0
        self.generation = 0
        self._buffer: deque[tuple[int, str, str | None, str]] = deque()  # (seq, type, turn_id, json)
        self._bytes = 0
        self._lock = threading.Lock()
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._files: dict[str, Path] = {}

    # ---- publishing -------------------------------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event: EventBase) -> int:
        with self._lock:
            self.seq += 1
            event.seq = self.seq
            frame = dumps(to_wire(event))
            self._buffer.append((self.seq, event.type, event.turn_id, frame))  # type: ignore[attr-defined]
            self._bytes += len(frame)
            while self._buffer and (len(self._buffer) > self.max_frames or self._bytes > self.max_bytes):
                _, _, _, old = self._buffer.popleft()
                self._bytes -= len(old)
            seq = self.seq
            subs = list(self._subscribers)
        self._persist(event.turn_id, frame)  # type: ignore[attr-defined]
        for q in subs:
            self._deliver(q, (seq, event.type, frame))  # type: ignore[attr-defined]
        return seq

    def _deliver(self, q: asyncio.Queue, item) -> None:
        loop = self._loop
        try:
            if loop is not None and loop.is_running() and threading.current_thread() is not threading.main_thread():
                loop.call_soon_threadsafe(self._put, q, item)
            elif loop is not None and loop.is_running():
                try:
                    running = asyncio.get_running_loop()
                except RuntimeError:
                    running = None
                if running is loop:
                    self._put(q, item)
                else:
                    loop.call_soon_threadsafe(self._put, q, item)
            else:
                self._put(q, item)
        except RuntimeError:
            pass

    @staticmethod
    def _put(q: asyncio.Queue, item) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            q.put_nowait(("resync", "resync", None))

    def _persist(self, turn_id: str | None, frame: str) -> None:
        if not self.session_dir:
            return
        path = self._files.get(turn_id or "_session")
        if path is None:
            d = self.session_dir / "events"
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{turn_id or 'session'}.jsonl"
            self._files[turn_id or "_session"] = path
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(frame + "\n")
        except OSError:
            pass

    # ---- reading ----------------------------------------------------------------------------------------
    @property
    def oldest_seq(self) -> int | None:
        with self._lock:
            return self._buffer[0][0] if self._buffer else None

    def replay(self, since: int = 0, turn_id: str | None = None) -> list[tuple[int, str, str]]:
        with self._lock:
            items = [(s, t, f) for s, t, tid, f in self._buffer if s > since and (turn_id is None or tid == turn_id)]
        return items

    def evicted(self, since: int) -> bool:
        oldest = self.oldest_seq
        return since > 0 and oldest is not None and since < oldest - 1

    def turn_frames(self, turn_id: str) -> list[dict]:
        """All frames of a turn: from disk when available, else from the ring buffer."""
        path = self._files.get(turn_id)
        if path and path.exists():
            with path.open(encoding="utf-8") as fh:
                return [json.loads(ln) for ln in fh if ln.strip()]
        return [json.loads(f) for _, _, f in self.replay(0, turn_id)]

    async def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)


__all__ = ["EventLog"]
