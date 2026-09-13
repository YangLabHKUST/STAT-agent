"""Summarizer role: regenerate the rolling conversation summary from evicted turns."""

from __future__ import annotations

from pydantic import BaseModel, Field

from stat_agent.memory.state import TurnRecord
from stat_agent.roles.base import Role


class SummaryOutput(BaseModel):
    summary: str = ""
    facts: list[dict] = Field(default_factory=list)


class SummarizerRole(Role):
    name = "summarizer"
    prompt_name = "summarizer"

    async def summarize(self, previous_summary: str, evicted: list[TurnRecord], *, max_words: int = 300, turn_id: str | None = None) -> SummaryOutput:
        block = "\n\n".join(t.history_lines() for t in evicted) or "(none)"
        variables = {"previous_summary": previous_summary or "(none)", "evicted_turns_block": block, "max_words": int(max_words)}
        fake_input = {"previous": previous_summary, "turns": [t.turn_id for t in evicted]}
        comp = await self.call(variables, schema=SummaryOutput, fake_input=fake_input, turn_id=turn_id)
        return comp.value


__all__ = ["SummarizerRole", "SummaryOutput"]
