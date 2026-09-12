"""Method-card synthesiser: a skill-shaped contract for an unregistered method."""

from __future__ import annotations

from stat_agent.core.view import SessionSummary
from stat_agent.memory.context import verifier_session_block
from stat_agent.roles.base import Role
from stat_agent.roles.matcher import MatcherRole
from stat_agent.skills.schema import MethodCard


class MethodCardRole(Role):
    name = "method_card"
    prompt_name = "method_card"

    async def synthesize(self, request: str, summary: SessionSummary, catalog: list[tuple[str, str]], *, research_notes: str = "", turn_id: str | None = None, step: int | None = None) -> MethodCard:
        variables = {
            "request": request,
            "session_block": verifier_session_block(summary, summary.slice_ids),
            "skills_catalog": MatcherRole.catalog_text(catalog) or "(none)",
            "research_notes": research_notes,
        }
        fake_input = {"request": request, "slices": [s.model_dump(include={"slice_id", "modality", "data_level"}) for s in summary.slices]}
        comp = await self.call(variables, schema=MethodCard, fake_input=fake_input, turn_id=turn_id, step=step)
        card: MethodCard = comp.value
        if research_notes and not card.research_notes:
            card.research_notes = research_notes
        return card


__all__ = ["MethodCardRole"]
