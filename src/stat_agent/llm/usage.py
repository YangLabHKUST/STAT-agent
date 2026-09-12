"""Per-call usage accounting (no monkey-patching of SDKs)."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class CallUsage:
    role: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    turn_id: str | None = None
    step: int | None = None
    mode: str = "text"
    repairs: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageTotals:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, u: CallUsage) -> None:
        self.llm_calls += 1
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.latency_s += u.latency_s

    def as_dict(self) -> dict[str, float]:
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 3),
        }


class UsageMeter:
    """Thread-safe accumulator: totals overall, per role, per turn."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[CallUsage] = []
        self.total = UsageTotals()
        self.by_role: dict[str, UsageTotals] = {}
        self.by_turn: dict[str, UsageTotals] = {}

    def record(self, usage: CallUsage) -> None:
        with self._lock:
            self.calls.append(usage)
            self.total.add(usage)
            self.by_role.setdefault(usage.role, UsageTotals()).add(usage)
            if usage.turn_id:
                self.by_turn.setdefault(usage.turn_id, UsageTotals()).add(usage)

    def turn_totals(self, turn_id: str) -> UsageTotals:
        return self.by_turn.get(turn_id, UsageTotals())

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()
            self.total = UsageTotals()
            self.by_role.clear()
            self.by_turn.clear()


__all__ = ["CallUsage", "UsageTotals", "UsageMeter"]
