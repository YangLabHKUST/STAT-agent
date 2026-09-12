"""Benchmark suite files (YAML/JSON) and the converter from STAT 0.1 ``queries.json``."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ScriptedAnswer(BaseModel):
    kind: str = "any"  # planner | verifier | skill_selection | any
    match: str
    answer: str
    once: bool = False


class ArtifactExpectation(BaseModel):
    kind: str
    key: str
    slice: int | None = None
    n_unique: list[int] | None = None


class Expected(BaseModel):
    behaviour: str = "any"  # execute | clarify | refuse | advise | any
    clarification_kinds_allowed: list[str] = Field(default_factory=lambda: ["planner", "verifier", "skill_selection"])
    skills_allowed: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactExpectation] = Field(default_factory=list)
    verifiers: list[dict[str, Any]] = Field(default_factory=list)
    oor_class: str | None = None
    required_requirement_regex: str | None = None


class Query(BaseModel):
    id: str
    text: str
    dataset: str | None = None
    inputs: list[str] = Field(default_factory=list)
    reference: str | None = None
    image: str | None = None
    category: str = "T11"
    subcategory: str | None = None
    difficulty: str = "clear"
    tags: list[str] = Field(default_factory=list)
    timeout_s: int = 600
    expected: Expected = Field(default_factory=Expected)
    scripted: list[ScriptedAnswer] = Field(default_factory=list)
    scripted_default: str | None = "yes"
    skill_pick: str | int | None = 1
    auto_hints: dict[str, str] = Field(default_factory=dict)
    ground_truth: dict[str, Any] | None = None


class Suite(BaseModel):
    suite_id: str
    version: str = "1"
    description: str = ""
    data_root_env: str = "STAT_BENCH_DATA"
    datasets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    queries: list[Query] = Field(default_factory=list)

    def get(self, qid: str) -> Query:
        for q in self.queries:
            if q.id == qid:
                return q
        raise KeyError(qid)

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.suffix in (".yaml", ".yml") else json.loads(p.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True), encoding="utf-8")
        return p

    def select(self, ids: list[str] | None = None, categories: list[str] | None = None, tags: list[str] | None = None) -> list[Query]:
        out = []
        for q in self.queries:
            if ids and q.id not in ids:
                continue
            if categories and q.category not in categories:
                continue
            if tags and not set(tags) & set(q.tags):
                continue
            out.append(q)
        return out


SPECIAL = {"infeasible": ("refuse", "infeasible"), "out-of-scope": ("refuse", "out_of_scope"), "missing-prereq": ("any", "missing_prereq")}


def convert_legacy_queries(path: str | Path, suite_id: str = "stat40") -> Suite:
    """STAT 0.1 ``queries.json`` -> :class:`Suite` (scripted keyword answers become regex rules)."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = []
    for it in items:
        scripted = [ScriptedAnswer(kind="any", match=rf"(?i)\b{re.escape(k)}\b", answer=v) for k, v in (it.get("scripted_responses") or {}).items()]
        task = it.get("task", "T11")
        behaviour, oor = "any", None
        if task in SPECIAL:
            behaviour, oor = SPECIAL[task]
        inputs = [it["input"]] if isinstance(it.get("input"), str) else list(it.get("input") or [])
        gt = {"file": it["ground_truth"]} if it.get("ground_truth") else None
        queries.append(
            Query(
                id=it["id"],
                text=it["query"],
                dataset=it.get("dataset"),
                inputs=inputs,
                reference=it.get("reference"),
                image=it.get("image"),
                category=task,
                difficulty=it.get("difficulty", "clear"),
                expected=Expected(behaviour=behaviour, oor_class=oor),
                scripted=scripted,
                ground_truth=gt,
            )
        )
    return Suite(suite_id=suite_id, description=f"converted from {Path(path).name}", queries=queries)


__all__ = ["Suite", "Query", "Expected", "ScriptedAnswer", "ArtifactExpectation", "convert_legacy_queries"]
