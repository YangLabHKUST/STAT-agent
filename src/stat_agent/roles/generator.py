"""Generator role: explanation text plus fenced Python code (the only free-text role)."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel, Field

from stat_agent.core.view import SessionSummary
from stat_agent.memory.context import generator_session_block, history_block, plan_block, skill_guidance_block
from stat_agent.memory.state import ConversationMemory
from stat_agent.roles.base import Role

_INTERPRET_RE = re.compile(r"__INTERPRET__:\s*(.+?)(?:\n|$)")
_CODE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)


class Segment(BaseModel):
    kind: Literal["text", "code"]
    content: str


class Generation(BaseModel):
    segments: list[Segment] = Field(default_factory=list)
    interpret_instruction: str = ""
    raw: str = ""

    @property
    def code_blocks(self) -> list[str]:
        return [s.content for s in self.segments if s.kind == "code"]

    @property
    def has_code(self) -> bool:
        return any(s.kind == "code" for s in self.segments)

    @property
    def text(self) -> str:
        return "\n\n".join(s.content for s in self.segments if s.kind == "text").strip()


def parse_segments(response: str) -> Generation:
    interpret = ""
    m = _INTERPRET_RE.search(response)
    if m:
        interpret = m.group(1).strip()
        response = response[: m.start()] + response[m.end() :]
    segments: list[Segment] = []
    last = 0
    for cm in _CODE_RE.finditer(response):
        text = response[last : cm.start()].strip()
        if text:
            segments.append(Segment(kind="text", content=text))
        code = cm.group(1).strip()
        if code:
            segments.append(Segment(kind="code", content=code))
        last = cm.end()
    tail = response[last:].strip()
    if tail:
        segments.append(Segment(kind="text", content=tail))
    return Generation(segments=segments, interpret_instruction=interpret, raw=response)


class GeneratorRole(Role):
    name = "generator"
    prompt_name = "generator_user"
    system_prompt_name = "generator_system"

    @staticmethod
    def variables(
        query: str,
        summary: SessionSummary | None,
        memory: ConversationMemory | None,
        *,
        roi_names: list[str] = (),
        skill_guidance: str = "",
        state_block: str = "",
        plan: tuple[str, int, int, str] | None = None,
        unguarded_note: str = "",
    ) -> dict:
        has_data = bool(summary and summary.has_data)
        entity = memory.entity_roi_text(list(roi_names)) if memory else ""
        return {
            "plan_block": plan_block(*plan) if plan else "",
            "session_block": generator_session_block(summary, list(roi_names)) if has_data else "",
            "entity_block": entity,
            "has_data": has_data,
            "state_block": state_block,
            "history_block": history_block(memory, "plain"),
            "skill_guidance": (unguarded_note + "\n\n" + skill_guidance).strip() if unguarded_note else skill_guidance,
            "user_query": query,
        }

    async def generate(self, variables: dict, *, fake_input=None, turn_id: str | None = None, step: int | None = None) -> Generation:
        comp = await self.call(variables, fake_input=fake_input, turn_id=turn_id, step=step)
        return parse_segments(comp.text)

    async def generate_streaming(self, variables: dict, *, fake_input=None, turn_id: str | None = None, step: int | None = None) -> AsyncIterator[str | Generation]:
        """Yield text deltas, then the final ``Generation``."""
        msgs = self.messages(variables)
        pieces: list[str] = []
        async for delta in self.llm.stream_text(msgs, role=self.name, prompt_id=self.prompt_id, turn_id=turn_id, step=step, fake_input=fake_input if fake_input is not None else variables):
            pieces.append(delta)
            yield delta
        yield parse_segments("".join(pieces))

    @staticmethod
    def guidance(name: str, body: str, max_chars: int = 25_000) -> str:
        if len(body) > max_chars:
            body = body[:max_chars] + "\n\n[... skill instructions truncated ...]"
        return skill_guidance_block(name, body)


__all__ = ["GeneratorRole", "Generation", "Segment", "parse_segments"]
