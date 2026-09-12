"""Matcher role: choose up to k skills from the compatible catalog (may choose none)."""

from __future__ import annotations

from pydantic import BaseModel, Field, RootModel

from stat_agent.roles.base import Role


class MatchList(RootModel[list[str]]):
    pass


class MatchResult(BaseModel):
    selected: list[str] = Field(default_factory=list)
    dropped_unknown: list[str] = Field(default_factory=list)
    raw: str = ""

    @property
    def none(self) -> bool:
        return not self.selected


class MatcherRole(Role):
    name = "matcher"
    prompt_name = "matcher"

    @staticmethod
    def catalog_text(catalog: list[tuple[str, str]]) -> str:
        return "\n".join(f"- **{slug}**: {desc}" for slug, desc in catalog)

    async def match(self, request: str, catalog: list[tuple[str, str]], top_k: int = 2, *, turn_id: str | None = None, step: int | None = None) -> MatchResult:
        if not catalog:
            return MatchResult()
        variables = {"request": request, "skills_catalog": self.catalog_text(catalog), "top_k": top_k}
        fake_input = {"request": request, "catalog": sorted(s for s, _ in catalog), "top_k": top_k}
        comp = await self.call(variables, schema=MatchList, fake_input=fake_input, turn_id=turn_id, step=step)
        wanted = [str(x) for x in comp.value.root]
        known = {s for s, _ in catalog}
        selected, dropped = [], []
        for slug in wanted:
            (selected if slug in known else dropped).append(slug)
        return MatchResult(selected=selected[:top_k], dropped_unknown=dropped, raw=comp.text)


__all__ = ["MatcherRole", "MatchResult", "MatchList"]
