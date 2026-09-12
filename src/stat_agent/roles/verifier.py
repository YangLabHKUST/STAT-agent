"""Verifier role: for each prerequisite, satisfied / ask the user / not obtainable.

A deterministic pre-pass runs first: no prerequisites -> met without an LLM call; a
declared ``consumes`` artefact that is stale or absent on every target slice -> decline
with advice. Only then is the LLM asked, with the same prompt as STAT 0.1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from stat_agent.core.view import SessionSummary
from stat_agent.memory.context import (
    history_block,
    user_responses_block,
    verifier_session_block,
    with_turn_steps,
)
from stat_agent.memory.state import ConversationMemory
from stat_agent.roles.base import Role
from stat_agent.roles.planner import PlanStep
from stat_agent.skills.schema import SkillFrontmatter

VerdictKind = Literal["met", "ask", "decline"]


class VerifierOutput(BaseModel):
    """The STAT 0.1 verifier JSON, unchanged."""

    prerequisites_met: bool = False
    missing_prerequisites: list[str] = Field(default_factory=list)
    can_obtain_by_chat: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    complete_query: str | None = None
    advice: str | None = None


class Verdict(BaseModel):
    kind: VerdictKind
    questions: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    complete_query: str | None = None
    advice: str | None = None
    decided_by: Literal["contract", "llm", "bypass"] = "llm"
    raw: str = ""


ANNOTATE_WORDS = ("annotat", "label", "assign", "classif", "deconvol")
# count layers that only an upstream quantifier can create; any other absent input of a provisional
# card is left to the LLM verifier (raw counts usually live in adata.X, not in a layer)
QUANTIFICATION_LAYERS = {"spliced", "unspliced", "ambiguous"}


class VerifierRole(Role):
    name = "verifier"
    prompt_name = "verifier"

    @staticmethod
    def contract_pass(contract: SkillFrontmatter, step: PlanStep, summary: SessionSummary, stale: list[str]) -> Verdict | None:
        """Deterministic decisions that need no model call (None = ask the LLM)."""
        if not contract.prerequisites and not contract.consumes:
            return Verdict(kind="met", complete_query=step.refined_query, decided_by="contract")
        targets = [summary.slice(sid) for sid in step.target_slice_ids]
        targets = [t for t in targets if t is not None]
        q = (step.refined_query or "").lower()
        # a synthesised (provisional) contract guesses its input keys: only data-level inputs that no
        # analysis can produce (count layers) are hard requirements; other absent keys go to the LLM verifier
        provisional = bool(getattr(contract, "provisional", False))
        sids = [t.slice_id for t in targets]
        missing_layers: list[str] = []
        for art in contract.consumes:
            if art.kind not in ("obs", "obsm", "uns", "var", "layers"):
                continue
            stale_hits = [f"s{t.slice_id}:{art.kind}:{art.key}" for t in targets if f"s{t.slice_id}:{art.kind}:{art.key}" in stale]
            if stale_hits:
                return Verdict(
                    kind="decline",
                    missing=[art.id],
                    advice=f"{art.id} on slice(s) {sids} is stale: an upstream annotation or ROI changed after it was computed. Re-run the step that produced it (or use 'Re-run downstream') before running this analysis.",
                    decided_by="contract",
                )
            present = [t for t in targets if t.has_key(art.kind, art.key)]
            if provisional and (art.kind != "layers" or art.key.lower() not in QUANTIFICATION_LAYERS):
                continue
            if targets and not present and not any(w in q for w in ANNOTATE_WORDS):
                if art.kind == "layers":
                    missing_layers.append(art.key)
                    continue
                producer = "cell type annotation" if art.key == "celltype" else "the step that produces it"
                return Verdict(
                    kind="decline",
                    missing=[art.id],
                    advice=f"This analysis requires {art.id}, which does not exist on slice(s) {sids}. Run {producer} first, then retry.",
                    decided_by="contract",
                )
        if missing_layers:
            keys = ", ".join(f"'{k}'" for k in missing_layers)
            return Verdict(
                kind="decline",
                missing=[f"layers:{k}" for k in missing_layers],
                advice=(
                    f"This analysis needs the count layer(s) {keys} in adata.layers on slice(s) {sids}, and the loaded data has none. "
                    "Such layers come from upstream quantification (they cannot be computed by an analysis step here), so load data that includes them before retrying."
                ),
                decided_by="contract",
            )
        return None

    async def verify(
        self,
        step: PlanStep,
        contract: SkillFrontmatter,
        summary: SessionSummary,
        *,
        answers: list[tuple[str, str]] = (),
        memory: ConversationMemory | None = None,
        stale: list[str] = (),
        skill_name: str | None = None,
        turn_steps: str = "",
        turn_id: str | None = None,
        step_index: int | None = None,
    ) -> Verdict:
        pre = self.contract_pass(contract, step, summary, list(stale))
        if pre is not None:
            return pre
        prereqs = "\n".join(f"  {i}. {p}" for i, p in enumerate(contract.prerequisites, 1))
        variables = {
            "step_description": step.description,
            "refined_query": step.refined_query,
            "target_slices": list(step.target_slice_ids),
            "skill_name": skill_name or contract.display_name,
            "skill_slug": contract.slug,
            "prerequisites_list": prereqs,
            "session_block": verifier_session_block(summary, step.target_slice_ids, stale),
            "user_responses_block": user_responses_block(answers),
            "history_block": with_turn_steps(history_block(memory, "verifier"), turn_steps),
        }
        fake_input = {"skill": contract.slug, "query": step.refined_query, "targets": [{"slice_id": s.slice_id, "has_celltype": s.has_celltype, "obs": sorted(s.obs_columns)[:40]} for s in (summary.slice(i) for i in step.target_slice_ids) if s], "answers": list(answers)}
        comp = await self.call(variables, schema=VerifierOutput, fake_input=fake_input, turn_id=turn_id, step=step_index)
        out: VerifierOutput = comp.value
        if out.prerequisites_met:
            return Verdict(kind="met", complete_query=out.complete_query or step.refined_query, decided_by="llm", raw=comp.text)
        if out.can_obtain_by_chat:
            questions = [q for q in out.clarification_questions if q and q.strip()] or ["Please provide the information needed for this analysis."]
            return Verdict(kind="ask", questions=questions, missing=out.missing_prerequisites, decided_by="llm", raw=comp.text)
        return Verdict(kind="decline", missing=out.missing_prerequisites, advice=out.advice or "The prerequisites for this analysis are not available in the current session.", decided_by="llm", raw=comp.text)


__all__ = ["VerifierRole", "Verdict", "VerifierOutput"]
