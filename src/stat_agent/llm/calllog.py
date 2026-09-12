"""Write-through call logging: one JSON file per LLM call plus the legacy per-turn markdown.

The markdown format (``## LLM Call N: <role>`` / ``### Request Metadata`` / ``### Full
Prompt`` / ``### Response``) is kept byte-compatible with STAT 0.1 so the trace converter
and human readers work on both generations of logs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from stat_agent.events.encoding import dumps


@dataclass
class CallRecord:
    role: str
    model: str
    prompt_id: str | None
    messages: list[dict[str, str]]
    response_text: str
    parsed: Any = None
    mode: str = "text"
    repairs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    turn_id: str | None = None
    step: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class CallLogger:
    """Persist every call immediately (a crash never loses the calls that already happened)."""

    def __init__(self, session_dir: Path | None = None) -> None:
        self.session_dir = Path(session_dir) if session_dir else None
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._index_of: dict[str, int] = {}
        self.records: list[CallRecord] = []
        self.keep_in_memory = True

    def set_session_dir(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)

    def start_turn(self, turn_id: str, user_query: str, turn_index: int | None = None) -> None:
        if not self.session_dir:
            return
        if turn_index is not None:
            self._index_of[turn_id] = turn_index
        path = self._turn_md(turn_id, turn_index)
        with self._lock:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                idx = turn_index if turn_index is not None else len(self._counters) + 1
                path.write_text(
                    f"# Conversation Turn {idx} - {datetime.now():%Y-%m-%d %H:%M:%S}\n\n## User Query\n\n```\n{user_query}\n```\n\n---\n\n",
                    encoding="utf-8",
                )

    def log_user_response(self, turn_id: str, kind: str, question: str, answer: str, turn_index: int | None = None) -> None:
        if not self.session_dir:
            return
        path = self._turn_md(turn_id, turn_index)
        with self._lock, path.open("a", encoding="utf-8") as fh:
            fh.write(f"## User Response ({kind.title()} Clarification)\n\n**Question:**\n```\n{question}\n```\n\n**User Response:**\n```\n{answer}\n```\n\n---\n\n")

    def record(self, rec: CallRecord, turn_index: int | None = None) -> Path | None:
        if self.keep_in_memory:
            self.records.append(rec)
        if not self.session_dir:
            return None
        key = rec.turn_id or "no-turn"
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
            n = self._counters[key]
        calls_dir = self.session_dir / "calls"
        calls_dir.mkdir(parents=True, exist_ok=True)
        json_path = calls_dir / f"{key}_{n:02d}_{rec.role}.json"
        json_path.write_text(dumps(rec.__dict__, indent=2), encoding="utf-8")
        md_path = self._turn_md(rec.turn_id or "no-turn", turn_index)
        prompt_text = "\n\n".join(f"[{m.get('role', 'user')}]\n{m.get('content', '')}" if m.get("role") != "user" else m.get("content", "") for m in rec.messages)
        block = (
            f"## LLM Call {n}: {rec.role}\n\n### Request Metadata\n"
            f"- **Model:** {rec.model}\n- **Timestamp:** {rec.timestamp}\n- **Purpose:** {rec.role}\n"
            f"- **Prompt ID:** {rec.prompt_id or '-'}\n- **Mode:** {rec.mode} (repairs: {rec.repairs})\n"
            f"- **Input tokens:** {rec.input_tokens}\n- **Output tokens:** {rec.output_tokens}\n- **Duration:** {rec.latency_s:.2f}s\n\n"
            f"### Full Prompt\n\n```\n{prompt_text}\n```\n\n### Response\n\n```\n{rec.response_text}\n```\n\n"
            + (f"**Error:** {rec.error}\n\n" if rec.error else "")
            + "---\n\n"
        )
        with self._lock:
            md_path.parent.mkdir(parents=True, exist_ok=True)
            with md_path.open("a", encoding="utf-8") as fh:
                fh.write(block)
        return json_path

    def _turn_md(self, turn_id: str, turn_index: int | None = None) -> Path:
        assert self.session_dir is not None
        if turn_index is None:
            turn_index = self._index_of.get(turn_id)
        name = f"turn_{turn_index:03d}.md" if turn_index is not None else f"turn_{turn_id}.md"
        return self.session_dir / name


__all__ = ["CallRecord", "CallLogger"]
