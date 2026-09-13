"""Clarification policies: what happens when the pipeline would ask the user.

``Interactive`` pauses the turn; ``Auto`` resolves from skill defaults, facts and session
state (never inventing a file path); ``Scripted`` answers from benchmark rules. Every
non-interactive resolution is recorded as an assumption.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

PromptKind = Literal["planner", "verifier", "skill_selection"]


class Interrupt(BaseModel):
    prompt_id: str = Field(default_factory=lambda: "p_" + uuid.uuid4().hex[:8])
    kind: PromptKind
    step: int | None = None
    question: str | None = None
    questions: list[str] = Field(default_factory=list)
    options: list[dict] = Field(default_factory=list)  # {index, label, slug, description}
    skill: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    prompt_id: str | None = None
    kind: PromptKind | None = None
    answer: str | None = None  # free text (planner) or combined verifier text
    option_index: int | None = None  # 1-based (skill selection)
    answers: dict[str, str] | None = None  # verifier: question -> answer
    source: Literal["user", "auto", "scripted", "override"] = "user"
    rationale: str = ""
    unresolvable: list[str] = Field(default_factory=list)

    def as_text(self) -> str:
        if self.answers:
            return ", ".join(f"{q}: {a}" for q, a in self.answers.items()) if len(self.answers) > 1 else next(iter(self.answers.values()))
        if self.option_index is not None:
            return str(self.option_index)
        return self.answer or ""


class ClarificationPolicy:
    name = "interactive"

    def resolve(self, pending: Interrupt, ctx: Any) -> Decision | None:  # noqa: ARG002
        return None


class InteractivePolicy(ClarificationPolicy):
    name = "interactive"


PATH_WORDS = ("path", "file", ".h5ad", "reference dataset", "directory", "upload")
# fact keys that name the data rather than a parameter: they appear in many questions ("... on slice 2")
GENERIC_FACT_KEYS = {"slice", "slices", "slice_id", "dataset", "data", "sample", "section", "sections", "slide", "session"}
STOPWORDS = {"the", "a", "an", "to", "is", "of", "it", "this", "that", "value", "and", "or", "for", "as", "in", "on", "by", "with", "if"}


def _summary_of(ctx: Any):
    s = getattr(ctx, "summary", None)
    return s() if callable(s) else s


class AutoPolicy(ClarificationPolicy):
    """Resolve every clarification without a human; refuse (unresolvable) what cannot be defaulted."""

    name = "auto"

    def __init__(self, hints: dict[str, str] | None = None) -> None:
        self.hints = {k.lower(): str(v) for k, v in (hints or {}).items()}

    def _answer_question(self, q: str, ctx: Any, defaults: dict[str, str], facts: dict[str, str]) -> tuple[str | None, str, str]:
        """(answer, source, rationale); the answer is None when nothing sensible can be assumed.

        Order: the skill's own defaults, explicit hints, a known fact whose key appears as a whole word
        in the question (never a generic key such as "slice": a question about domains on slice 2 is not
        a question about the slice), the species / tissue / column conventions, then a value the
        question itself suggests (a stated default, an example, the first of the options offered), and
        last a number for a counting question. A file path is never assumed unless the user stated it."""
        ql = q.lower()
        for key, val in defaults.items():
            if key.lower() in ql or ql in key.lower():
                return val, "skill_default", f"skill default for '{key}'"
        for key, val in self.hints.items():
            if key in ql:
                return val, "hint", f"hint '{key}'"
        for key, val in facts.items():
            k = key.lower()
            if k not in GENERIC_FACT_KEYS and re.search(rf"\b{re.escape(k)}\b", ql):
                return val, "fact", f"known fact '{key}'"
        if "species" in ql:
            if "human" in ql or "mouse" in ql:
                return ("human" if "human" in ql else "mouse"), "policy_default", "species defaulted to the first option mentioned"
            return "human", "policy_default", "species defaulted to human"
        if "tissue" in ql:
            tissue = self.hints.get("tissue") or facts.get("tissue")
            if tissue:
                return tissue, "fact", "tissue from hints"
            summary = _summary_of(ctx)
            names = [s.tissue_name for s in summary.slices] if summary else []
            if names:
                return names[0], "session_state", "tissue name taken from the loaded slice name"
        if "column" in ql and "celltype" in ql:
            return "celltype", "policy_default", "standard column name"
        if any(w in ql for w in PATH_WORDS):
            return None, "", "a file path cannot be assumed"
        m = re.search(r"\bdefaults?\b(?:\s+(?:to|is|of|value))?\s*:?\s*['\"]?([\w./-]+)", ql)
        if m and m.group(1) not in STOPWORDS:
            return m.group(1), "policy_default", "default value stated in the question"
        m = re.search(r"\be\.g\.?,?\s*['\"]?([\w./-]+)", ql)
        if m and m.group(1) not in STOPWORDS:
            return m.group(1), "policy_default", "example value offered in the question"
        options = re.findall(r"(?:^|[\s(])['\"]([^'\"]{1,40})['\"](?=[\s,?.)]|$)", q)
        if len(options) >= 2 and " or " in ql:
            return options[0], "policy_default", "first of the options offered"
        if "number" in ql or "how many" in ql:
            m = re.search(r"(?<!slice )(?<!slices )(?<!section )\b(\d+)\b", ql)
            if m:
                return m.group(1), "policy_default", "number mentioned in the question"
        return None, "", "no default available"

    def resolve(self, pending: Interrupt, ctx: Any) -> Decision | None:
        defaults = dict(getattr(ctx, "skill_defaults", {}) or {})
        facts = {k: v.value for k, v in getattr(getattr(ctx, "memory", None), "facts", {}).items()} if getattr(ctx, "memory", None) else {}
        if pending.kind == "planner":
            question = pending.question or ""
            if any(w in question.lower() for w in ("slice", "section", "sample", "tissue", "dataset", "which data")):
                summary = _summary_of(ctx)
                ids = list(summary.slice_ids) if summary else []
                answer = f"slice {ids[0]}" if len(ids) == 1 else ("all slices" if ids else "slice 0")
                return Decision(prompt_id=pending.prompt_id, kind="planner", answer=answer, source="auto", rationale="auto mode: use every loaded slice" if len(ids) != 1 else "auto mode: the only loaded slice")
            # any other planner question (parameters, method choices) is answered from defaults and facts or not at all
            val, _src, why = self._answer_question(question, ctx, defaults, facts)
            if val is None:
                return Decision(prompt_id=pending.prompt_id, kind="planner", answer="", source="auto", unresolvable=[question or "clarification"], rationale="auto mode: no default or known fact answers this question")
            return Decision(prompt_id=pending.prompt_id, kind="planner", answer=val, source="auto", rationale=f"auto mode: {why}")
        if pending.kind == "skill_selection":
            return Decision(prompt_id=pending.prompt_id, kind="skill_selection", option_index=1, source="auto", rationale="auto mode: first-ranked skill")
        answers: dict[str, str] = {}
        unresolved: list[str] = []
        rationale: list[str] = []
        for q in pending.questions:
            val, _src, why = self._answer_question(q, ctx, defaults, facts)
            if val is None:
                unresolved.append(q)
            else:
                answers[q] = val
                rationale.append(f"{q} -> {val} ({why})")
        if unresolved:
            return Decision(prompt_id=pending.prompt_id, kind="verifier", answers=answers or None, source="auto", unresolvable=unresolved, rationale="; ".join(rationale))
        return Decision(prompt_id=pending.prompt_id, kind="verifier", answers=answers, source="auto", rationale="; ".join(rationale))


class ScriptedRule(BaseModel):
    kind: str = "any"  # planner | verifier | skill_selection | any
    match: str = ".*"
    answer: str
    once: bool = False


class ScriptedPolicy(ClarificationPolicy):
    """Benchmark answers: ordered regex rules; ``skill_pick`` chooses among options; ``default`` on miss (None -> pause)."""

    name = "scripted"

    def __init__(self, rules: list[ScriptedRule | dict] | None = None, skill_pick: str | int | None = 1, default: str | None = "yes") -> None:
        self.rules = [r if isinstance(r, ScriptedRule) else ScriptedRule(**r) for r in (rules or [])]
        self.skill_pick = skill_pick
        self.default = default
        self._used: set[int] = set()

    def _answer(self, kind: str, text: str) -> str | None:
        for i, r in enumerate(self.rules):
            if r.once and i in self._used:
                continue
            if r.kind not in ("any", kind):
                continue
            if re.search(r.match, text, re.I):
                self._used.add(i)
                return r.answer
        return self.default

    def resolve(self, pending: Interrupt, ctx: Any) -> Decision | None:  # noqa: ARG002
        if pending.kind == "skill_selection":
            idx = 1
            if isinstance(self.skill_pick, int):
                idx = self.skill_pick
            elif isinstance(self.skill_pick, str):
                for o in pending.options:
                    if self.skill_pick.lower() in (o.get("slug") or "").lower() or self.skill_pick.lower() in (o.get("label") or "").lower():
                        idx = int(o.get("index", 1))
                        break
            return Decision(prompt_id=pending.prompt_id, kind="skill_selection", option_index=idx, source="scripted", rationale="scripted skill pick")
        if pending.kind == "planner":
            ans = self._answer("planner", pending.question or "")
            return None if ans is None else Decision(prompt_id=pending.prompt_id, kind="planner", answer=ans, source="scripted")
        answers: dict[str, str] = {}
        for q in pending.questions:
            ans = self._answer("verifier", q)
            if ans is None:
                return None
            answers[q] = ans
        return Decision(prompt_id=pending.prompt_id, kind="verifier", answers=answers, source="scripted")


def make_policy(mode: str, **kwargs: Any) -> ClarificationPolicy:
    if mode == "auto":
        return AutoPolicy(kwargs.get("hints"))
    if mode == "scripted":
        return ScriptedPolicy(kwargs.get("rules"), kwargs.get("skill_pick", 1), kwargs.get("default", "yes"))
    return InteractivePolicy()


__all__ = ["Interrupt", "Decision", "ClarificationPolicy", "InteractivePolicy", "AutoPolicy", "ScriptedPolicy", "ScriptedRule", "make_policy"]
