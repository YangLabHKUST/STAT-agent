"""Planner role: which slice(s), which steps, or which clarification question."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from stat_agent.core.view import InterfaceState, SessionSummary
from stat_agent.memory.context import (
    clarifications_block,
    history_block,
    interface_block,
    planner_session_block,
)
from stat_agent.memory.state import ConversationMemory
from stat_agent.roles.base import Role

FALLBACK_QUESTION = "Could you please clarify which slice(s) you want to analyze?"


class PlanStep(BaseModel):
    step_number: int
    description: str = ""
    kind: Literal["skill", "general"] = "skill"  # general: no method needed, no method card
    target_slice_ids: list[int] = Field(default_factory=list)
    refined_query: str = ""
    depends_on: list[int] = Field(default_factory=list)

    @field_validator("target_slice_ids", mode="before")
    @classmethod
    def _ints(cls, v):
        return [int(x) for x in (v or [])]


class PlanOutput(BaseModel):
    needs_clarification: bool = False
    clarification_question: str | None = None
    direct_answer: str | None = None
    steps: list[PlanStep] = Field(default_factory=list)
    facts_extracted: list[dict] = Field(default_factory=list)


class PlanResult(BaseModel):
    needs_clarification: bool
    question: str | None = None
    direct_answer: str | None = None  # the whole turn is this answer: no steps run
    preamble: str = ""  # an answer the planner gave *and* planned steps for (a "tell me X, then do Y" turn)
    steps: list[PlanStep] = Field(default_factory=list)
    invalid_slice_ids: list[int] = Field(default_factory=list)
    fallback: bool = False
    raw: str = ""


class PlannerRole(Role):
    name = "planner"
    prompt_name = "planner"

    @staticmethod
    def variables(query: str, summary: SessionSummary, memory: ConversationMemory | None, answers: list[tuple[str, str]], interface: InterfaceState | None = None, catalog: list[tuple[str, str]] | None = None, state_block: str = "") -> dict:
        return {
            "skills_block": skills_block(catalog),
            "session_block": planner_session_block(summary) + interface_block(interface),
            "history_block": history_block(memory, "planner"),
            "state_block": state_block,
            "user_query": query,
            "clarifications_block": clarifications_block(answers),
        }

    async def plan(self, query: str, summary: SessionSummary, *, memory: ConversationMemory | None = None, answers: list[tuple[str, str]] = (), interface: InterfaceState | None = None, turn_id: str | None = None, catalog: list[tuple[str, str]] | None = None, state_block: str = "") -> PlanResult:
        variables = self.variables(query, summary, memory, list(answers), interface, catalog, state_block)
        fake_input = {"query": query, "slices": [s.model_dump(include={"slice_id", "modality", "data_level", "n_obs", "has_celltype"}) for s in summary.slices], "rois": [r.name for r in summary.rois], "answers": list(answers)}
        comp = await self.call(variables, schema=PlanOutput, fake_input=fake_input, turn_id=turn_id)
        out: PlanOutput = comp.value
        return self.validate(out, summary, comp.text)

    @staticmethod
    def validate(out: PlanOutput, summary: SessionSummary, raw: str = "") -> PlanResult:
        answer = (out.direct_answer or "").strip()
        if answer and not out.steps:
            return PlanResult(needs_clarification=False, direct_answer=answer, raw=raw)
        if out.needs_clarification:
            question = (out.clarification_question or "").strip()
            if question and "?" not in question and len(question) > 240:
                # the model answered the user instead of asking (seen with capability questions)
                return PlanResult(needs_clarification=False, direct_answer=question, raw=raw)
            return PlanResult(needs_clarification=True, question=question or FALLBACK_QUESTION, raw=raw)
        steps = [s for s in out.steps if s.refined_query or s.description]
        if not steps:
            return PlanResult(needs_clarification=True, question=FALLBACK_QUESTION, fallback=True, raw=raw)
        invalid = sorted({sid for s in steps for sid in s.target_slice_ids if not summary.has_slice(sid)})
        if invalid:
            return PlanResult(
                needs_clarification=True,
                question=f"The plan referred to slice(s) {invalid}, which are not loaded (available: {summary.slice_ids}). Which slice(s) should I use?",
                invalid_slice_ids=invalid,
                raw=raw,
            )
        for i, s in enumerate(steps, 1):
            s.step_number = i
            if not s.refined_query:
                s.refined_query = s.description
            if not s.target_slice_ids and summary.n_slices == 1:
                s.target_slice_ids = list(summary.slice_ids)
            s.depends_on = [d for d in s.depends_on if 1 <= d < i]
        return PlanResult(needs_clarification=False, steps=steps, preamble=answer, raw=raw)

    @staticmethod
    def skills_block(catalog: list[tuple[str, str]] | None) -> str:
        return skills_block(catalog)

    @staticmethod
    def single_step(query: str, summary: SessionSummary) -> PlanResult:
        """Ablation: no planner - one step over every loaded slice."""
        return PlanResult(needs_clarification=False, steps=[PlanStep(step_number=1, description=query, target_slice_ids=list(summary.slice_ids), refined_query=query)])


def skills_block(catalog: list[tuple[str, str]] | None, max_desc: int = 150) -> str:
    """Catalogue block for the planner (skill slug: what it does)."""
    if not catalog:
        return ""
    lines = ["**Available analysis skills** (slug: what it does):"]
    for slug, desc in catalog:
        d = " ".join(str(desc).split())
        lines.append(f"- {slug}: {d[:max_desc]}{'...' if len(d) > max_desc else ''}")
    return "\n".join(lines)


__all__ = ["PlanStep", "PlanOutput", "PlanResult", "PlannerRole", "FALLBACK_QUESTION"]
