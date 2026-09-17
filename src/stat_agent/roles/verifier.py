"""Verifier role: for each prerequisite, satisfied / ask the user / not obtainable.

A deterministic pre-pass runs first: no prerequisites -> met without an LLM call; a declared
``consumes`` artefact that is stale -> decline with advice; an absent ``obs`` column is read from a
column that is the same key spelled differently, or from the label column the user names when asked
(the question lists the label columns), and otherwise declined with advice. Only then is the LLM
asked, with the same prompt as STAT 0.1.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from stat_agent.core.view import SessionSummary
from stat_agent.memory.context import (
    earlier_responses_block,
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
    missing_files: list[str] = Field(default_factory=list)  # file prerequisites no file in the session satisfies


# count layers that only an upstream quantifier can create; any other absent input of a provisional
# card is left to the LLM verifier (raw counts usually live in adata.X, not in a layer)
QUANTIFICATION_LAYERS = {"spliced", "unspliced", "ambiguous"}
FILE_WORDS = re.compile(r"\.h5ad\b|\b(?:path|file)\b", re.I)
H5AD_TOKEN = re.compile(r"[\w./~-]+\.h5ad\b")


def required_file_prerequisites(contract: SkillFrontmatter) -> list[str]:
    """The prerequisites only a file can satisfy (a reference .h5ad); one marked optional is not required."""
    return [p for p in contract.prerequisites if FILE_WORDS.search(p) and "optional" not in p.lower()]


def file_in_session(summary: SessionSummary, texts: Iterable[str]) -> bool:
    """A file the session can use exists: the dataset lists extra h5ad files, or a text names an .h5ad that exists."""
    if summary.extra_files:
        return True
    roots = [Path.cwd()] + ([Path(summary.dataset_dir)] if summary.dataset_dir else [])
    for text in texts:
        for token in H5AD_TOKEN.findall(text or ""):
            path = Path(token).expanduser()
            if (path.is_absolute() and path.exists()) or any((root / path).exists() for root in roots):
                return True
    return False


# what an obs key holds, for the question and the advice; any other key is named as it is
KEY_WORDS = {"celltype": ("cell-type labels", "cell type annotation")}
MAX_LISTED_COLUMNS = 8
MAX_LISTED_LABELS = 6


def _name_tokens(name: str) -> list[str]:
    """``predicted_cellType`` -> ``['predicted', 'cell', 'type']``."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def column_name_matches(key: str, column: str) -> bool:
    """The column is ``key`` spelled differently: case, separators, a plural, a prefix or a suffix
    (``cell_type``, ``CellType``, ``celltypes``, ``predicted_celltype``, ``celltype_scanvi`` for ``celltype``)."""
    target = re.sub(r"[^a-z0-9]", "", key.lower())
    if not target or column == key:
        return False
    tokens = _name_tokens(column)
    for i in range(len(tokens)):
        joined = ""
        for token in tokens[i:]:
            joined += token
            if joined in (target, target + "s"):
                return True
            if len(joined) > len(target):
                break
    return False


def columns_named(text: str, columns: Iterable[str]) -> list[str]:
    """The columns a reply names, as whole words, in the order of the listing."""
    return [c for c in columns if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", text or "", re.I)]


def _label_listing(slices: list, key: str) -> str:
    parts = []
    for t in slices:
        cols = sorted((c for c in t.label_columns if c != key), key=lambda c: not column_name_matches(key, c))[:MAX_LISTED_COLUMNS]
        described = ", ".join(
            f"{c} ({t.label_columns[c].n_labels} labels: {', '.join(t.label_columns[c].examples[:MAX_LISTED_LABELS])}{', ...' if t.label_columns[c].n_labels > MAX_LISTED_LABELS else ''})"
            for c in cols
        )
        parts.append(described if len(slices) == 1 else f"slice {t.slice_id}: {described}")
    return "; ".join(parts)


def resolve_contract(
    contract: SkillFrontmatter,
    step: PlanStep,
    summary: SessionSummary,
    stale: Iterable[str] = (),
    answers: Iterable[tuple[str, str]] = (),
) -> tuple[Verdict | None, dict[str, tuple[str, str]]]:
    """The deterministic pre-pass: ``(verdict or None, columns read in place of absent obs keys)``.

    The second item maps ``s<id>:obs:<key>`` to ``(column, reason)``. A verdict of None means the
    LLM verifier decides the rest.
    """
    if not contract.prerequisites and not contract.consumes:
        return Verdict(kind="met", complete_query=step.refined_query, decided_by="contract"), {}
    stale = set(stale)
    targets = [t for t in (summary.slice(sid) for sid in step.target_slice_ids) if t is not None]
    # a synthesised (provisional) contract guesses its input keys: only data-level inputs that no
    # analysis can produce (count layers) are hard requirements; other absent keys go to the LLM verifier
    provisional = bool(getattr(contract, "provisional", False))
    sids = [t.slice_id for t in targets]
    columns: dict[str, tuple[str, str]] = {}
    missing_layers: list[str] = []
    for art in contract.consumes:
        if art.kind not in ("obs", "obsm", "uns", "var", "layers"):
            continue
        if any(f"s{t.slice_id}:{art.id}" in stale for t in targets):
            return Verdict(
                kind="decline",
                missing=[art.id],
                advice=f"{art.id} on slice(s) {sids} is stale: an upstream annotation or ROI changed after it was computed. Re-run the step that produced it (or use 'Re-run downstream') before running this analysis.",
                decided_by="contract",
            ), {}
        if provisional and (art.kind != "layers" or art.key.lower() not in QUANTIFICATION_LAYERS):
            continue
        # a key present on some target slice, or one this skill writes itself, is left to the LLM verifier
        if not targets or any(t.has_key(art.kind, art.key) for t in targets) or any(p.kind == art.kind and p.key == art.key for p in contract.produces):
            continue
        absent = targets
        if art.kind == "layers":
            missing_layers.append(art.key)
            continue
        what, producer = KEY_WORDS.get(art.key, (f"the values of {art.kind}['{art.key}']", "the step that produces it"))
        without = [t.slice_id for t in absent]
        advice = f"This analysis requires {art.id}, which does not exist on slice(s) {without}. Run {producer} first, then retry."
        if art.kind != "obs":
            return Verdict(kind="decline", missing=[art.id], advice=advice, decided_by="contract"), {}
        marker = f"obs['{art.key}']"
        replies = [a for q, a in answers if marker in q]
        unresolved = []
        for t in absent:
            labels = [c for c in t.label_columns if c != art.key]
            if replies:
                named = columns_named(replies[-1], labels)
                if len(named) == 1:
                    columns[f"s{t.slice_id}:{art.id}"] = (named[0], "named by the user")
                    continue
                if len(named) > 1:
                    return Verdict(kind="decline", missing=[art.id], advice=f"This analysis requires {art.id} on slice(s) {without}, and the reply named more than one column ({', '.join(named)}). Name exactly one column to read as {what}, or run {producer} first.", decided_by="contract"), {}
                return Verdict(kind="decline", missing=[art.id], advice=f"This analysis requires {art.id}, which does not exist on slice(s) {without}, and no existing column was named to use in its place. Run {producer} first, then retry.", decided_by="contract"), {}
            same_key = [c for c in labels if column_name_matches(art.key, c)]
            if len(same_key) == 1:
                columns[f"s{t.slice_id}:{art.id}"] = (same_key[0], f"the same key spelled differently from {marker}")
            elif labels:
                unresolved.append(t)
            else:
                return Verdict(kind="decline", missing=[art.id], advice=advice, decided_by="contract"), {}
        if unresolved:
            where = f"slice {unresolved[0].slice_id} does" if len(unresolved) == 1 else f"slices {[t.slice_id for t in unresolved]} do"
            question = f"This analysis needs {what} in {marker}, which {where} not have. Label columns: {_label_listing(unresolved, art.key)}. Reply with the name of the column to read as {what}, or ask for {producer} first."
            return Verdict(kind="ask", questions=[question], missing=[art.id], decided_by="contract"), {}
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
        ), {}
    return None, columns


def column_note(columns: dict[str, tuple[str, str]]) -> str:
    """The instruction that travels with the query when a column stands in for an absent obs key."""
    lines = []
    for ref, (column, reason) in columns.items():
        sid, kind, key = ref.split(":", 2)
        lines.append(f"On slice {sid[1:]} read {kind}['{column}'] wherever the analysis reads {kind}['{key}'] ({reason}); do not create or overwrite {kind}['{key}'].")
    return ("\n\nColumns to use: " + " ".join(lines)) if lines else ""


def column_answers(columns: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    """The same decisions as collected information for the LLM verifier (one question and answer per slice)."""
    out = []
    for ref, (column, reason) in columns.items():
        sid, kind, key = ref.split(":", 2)
        out.append((f"Which column should be read as {kind}['{key}'] on slice {sid[1:]}?", f"{column} ({reason})"))
    return out


class VerifierRole(Role):
    name = "verifier"
    prompt_name = "verifier"

    @staticmethod
    def contract_pass(contract: SkillFrontmatter, step: PlanStep, summary: SessionSummary, stale: list[str], answers: Iterable[tuple[str, str]] = ()) -> Verdict | None:
        """Deterministic decisions that need no model call (None = ask the LLM)."""
        return resolve_contract(contract, step, summary, stale, answers)[0]

    async def verify(
        self,
        step: PlanStep,
        contract: SkillFrontmatter,
        summary: SessionSummary,
        *,
        answers: list[tuple[str, str]] = (),
        earlier_answers: list[tuple[str, str]] = (),
        memory: ConversationMemory | None = None,
        stale: list[str] = (),
        skill_name: str | None = None,
        turn_steps: str = "",
        turn_id: str | None = None,
        step_index: int | None = None,
    ) -> Verdict:
        pre, columns = resolve_contract(contract, step, summary, stale, answers)
        if pre is not None:
            return pre
        note = column_note(columns)
        if note:
            step = step.model_copy(update={"refined_query": step.refined_query + note})
        column_pairs = column_answers(columns)
        # a file cannot be defaulted: once the user has been asked and no file is in the session, the file
        # prerequisite is unmet whatever the model concludes (otherwise it writes an invented path into the query)
        need_files = required_file_prerequisites(contract)
        texts = [step.refined_query, step.description, *(a for _, a in answers), *(a for _, a in earlier_answers)]
        no_file = bool(need_files) and not file_in_session(summary, texts)
        if no_file and answers:
            return Verdict(
                kind="decline",
                missing=need_files,
                missing_files=need_files,
                advice=f"{skill_name or contract.display_name} needs {need_files[0]}, and no such file is in this session or was given; a file cannot be assumed.",
                decided_by="contract",
            )
        prereqs = "\n".join(f"  {i}. {p}" for i, p in enumerate(contract.prerequisites, 1))
        variables = {
            "step_description": step.description,
            "refined_query": step.refined_query,
            "target_slices": list(step.target_slice_ids),
            "skill_name": skill_name or contract.display_name,
            "skill_slug": contract.slug,
            "prerequisites_list": prereqs,
            "session_block": verifier_session_block(summary, step.target_slice_ids, stale),
            "user_responses_block": user_responses_block([*answers, *column_pairs]) + earlier_responses_block(earlier_answers),
            "history_block": with_turn_steps(history_block(memory, "verifier"), turn_steps),
        }
        fake_input = {"skill": contract.slug, "query": step.refined_query, "targets": [{"slice_id": s.slice_id, "has_celltype": s.has_celltype, "obs": sorted(s.obs_columns)[:40]} for s in (summary.slice(i) for i in step.target_slice_ids) if s], "answers": list(answers)}
        comp = await self.call(variables, schema=VerifierOutput, fake_input=fake_input, turn_id=turn_id, step=step_index)
        out: VerifierOutput = comp.value
        if out.prerequisites_met:
            if no_file:  # the model called a file prerequisite met with no file anywhere: ask for it instead
                return Verdict(kind="ask", questions=[f"{p}: no such file is loaded in this session. Please give the path to the file." for p in need_files], missing=need_files, decided_by="contract", raw=comp.text)
            complete = out.complete_query or step.refined_query
            if note and note.strip() not in complete:
                complete += note
            return Verdict(kind="met", complete_query=complete, decided_by="llm", raw=comp.text)
        if out.can_obtain_by_chat:
            questions = [q for q in out.clarification_questions if q and q.strip()] or ["Please provide the information needed for this analysis."]
            return Verdict(kind="ask", questions=questions, missing=out.missing_prerequisites, decided_by="llm", raw=comp.text)
        return Verdict(kind="decline", missing=out.missing_prerequisites, advice=out.advice or "The prerequisites for this analysis are not available in the current session.", decided_by="llm", raw=comp.text, missing_files=need_files if no_file else [])


__all__ = ["VerifierRole", "Verdict", "VerifierOutput", "required_file_prerequisites", "file_in_session", "resolve_contract", "column_name_matches", "columns_named", "column_note", "column_answers"]
