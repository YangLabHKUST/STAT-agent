"""Conversation memory: typed turn records, per-step handoffs, facts and a rolling summary.

Everything here is a plain pydantic model (JSON-serialisable) so it can live in graph
state, be checkpointed, and be persisted to ``memory.json`` next to the session.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class Fact(BaseModel):
    key: str
    value: str
    source_turn: str | None = None
    confidence: float = 1.0


class StepHandoff(BaseModel):
    step_number: int
    description: str = ""
    skill: str | None = None
    mode: str = "guided"  # guided | unguarded | advice | failed | skipped | text
    success: bool = True
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    summary: str = ""
    data_changes: str = ""
    key_findings: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    code_hash: str | None = None
    code: list[str] = Field(default_factory=list)
    complete_query: str | None = None
    response: str = ""

    def brief(self) -> str:
        head = f"Step {self.step_number}"
        if self.skill:
            head += f" [{self.skill}]"
        head += f" ({self.mode}{'' if self.success else ', FAILED'})"
        parts = [head + ": " + (self.summary or self.description)]
        if self.outputs:
            parts.append("  outputs: " + ", ".join(self.outputs))
        if self.key_findings:
            parts.append("  findings: " + "; ".join(self.key_findings[:3]))
        if self.unresolved:
            parts.append("  unresolved: " + "; ".join(self.unresolved[:3]))
        return "\n".join(parts)


class AssumptionRecord(BaseModel):
    assumption_id: str
    kind: str
    question: str
    answer: str
    source: str = "policy_default"
    rationale: str = ""
    step: int | None = None


class TurnRecord(BaseModel):
    turn_id: str
    index: int
    query: str
    status: str = "done"
    response_summary: str = ""
    data_changes: str = ""
    key_findings: list[str] = Field(default_factory=list)
    handoffs: list[StepHandoff] = Field(default_factory=list)
    assumptions: list[AssumptionRecord] = Field(default_factory=list)
    response: str = ""
    ts: float = Field(default_factory=time.time)

    def history_lines(self) -> str:
        """The STAT 0.1 history format: query, assistant summary, execution summary."""
        lines = [f"**Query:** {self.query}"]
        summary = self.response_summary or (self.response[:400] if self.response else "")
        if summary:
            lines.append(f"**Assistant:** {summary}")
        execution = self.data_changes
        findings = "; ".join(self.key_findings[:3])
        if execution or findings:
            lines.append(f"**Execution:** {execution}" + (f" **Findings:** {findings}" if findings else ""))
        if self.assumptions:
            lines.append("**Assumptions:** " + "; ".join(f"{a.question} -> {a.answer}" for a in self.assumptions[:3]))
        return "\n".join(lines)


class ConversationMemory(BaseModel):
    turns: list[TurnRecord] = Field(default_factory=list)
    rolling_summary: str = ""
    summary_covers: int = 0  # number of leading turns folded into rolling_summary
    facts: dict[str, Fact] = Field(default_factory=dict)
    window_turns: int = 4  # verbatim window (the paper's "eight messages" = four turns)

    # ---- updates -------------------------------------------------------------------------------
    def record_turn(self, rec: TurnRecord) -> None:
        self.turns.append(rec)

    def add_fact(self, key: str, value: str, source_turn: str | None = None, confidence: float = 1.0) -> None:
        if not key or value is None:
            return
        self.facts[key] = Fact(key=key, value=str(value), source_turn=source_turn, confidence=confidence)

    @property
    def needs_summary(self) -> bool:
        return len(self.turns) - self.summary_covers > self.window_turns

    def evicted_turns(self) -> list[TurnRecord]:
        cutoff = len(self.turns) - self.window_turns
        return self.turns[self.summary_covers : max(self.summary_covers, cutoff)]

    def apply_summary(self, summary: str, covers: int) -> None:
        self.rolling_summary = summary
        self.summary_covers = covers

    # ---- rendering ---------------------------------------------------------------------------------
    def recent_turns(self, n: int | None = None) -> list[TurnRecord]:
        n = self.window_turns if n is None else n
        return self.turns[-n:] if n else []

    def history_text(self, max_turns: int | None = None, char_budget: int = 12000) -> str:
        """Rolling summary (if any) followed by the recent turns in the STAT 0.1 format."""
        blocks: list[str] = []
        if self.rolling_summary:
            blocks.append(f"**Earlier in this session:** {self.rolling_summary}")
        for t in self.recent_turns(max_turns):
            blocks.append(t.history_lines())
        text = "\n\n".join(blocks)
        if len(text) > char_budget:
            text = "...\n" + text[-char_budget:]
        return text

    def facts_text(self) -> str:
        if not self.facts:
            return ""
        return "\n".join(f"- {f.key}: {f.value}" for f in self.facts.values())

    def handoffs_text(self, n: int = 3) -> str:
        items: list[str] = []
        for t in self.turns[-n:]:
            for h in t.handoffs:
                items.append(h.brief())
        return "\n".join(items)

    def entity_roi_text(self, roi_names: list[str]) -> str:
        """The STAT 0.1 entity context block (ROIs only, which is all it ever rendered)."""
        if not roi_names:
            return ""
        return "**Known ROIs:** " + ", ".join(roi_names)


__all__ = ["Fact", "StepHandoff", "AssumptionRecord", "TurnRecord", "ConversationMemory"]
