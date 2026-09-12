"""Assisted skill authoring: draft, validate and install a SKILL.md."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from stat_agent.core.view import SessionSummary
from stat_agent.llm import LLMClient
from stat_agent.memory.context import verifier_session_block
from stat_agent.prompts import load_prompt
from stat_agent.skills.lint import lint_file
from stat_agent.skills.schema import parse_skill_text, slugify


async def draft_skill_text(llm: LLMClient, *, method_name: str, query: str, contract: dict[str, Any], code_blocks: list[str], outputs: list[str], summary: SessionSummary, extra_notes: str = "", turn_id: str | None = None) -> str:
    prompt = load_prompt("skill_author")
    text = prompt.render(
        method_name=method_name,
        query=query,
        contract_yaml=yaml.safe_dump(contract, sort_keys=False, allow_unicode=True) if contract else "(none)",
        code_blocks="\n\n".join(f"```python\n{c}\n```" for c in code_blocks) or "(no code)",
        outputs=", ".join(outputs) or "(none observed)",
        session_block=verifier_session_block(summary, summary.slice_ids),
        extra_notes=extra_notes,
    )
    comp = await llm.complete([("user", text)], role="skill_author", prompt_id=prompt.prompt_id, fake_input={"method": method_name, "query": query, "code": code_blocks}, turn_id=turn_id)
    out = comp.text.strip()
    m = re.search(r"```(?:markdown|md)?\s*\n(---\n.*)```", out, re.S)
    if m:
        out = m.group(1).strip()
    if not out.startswith("---"):
        out = out[out.find("---") :] if "---" in out else out
    return out


def install_skill_text(text: str, user_skills_dir: Path, slug: str | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Validate a SKILL.md text and write it to ``<user_skills_dir>/<slug>/SKILL.md``."""
    meta, body, raw = parse_skill_text(text)
    slug = slug or meta.slug or slugify(meta.title)
    if not raw.get("schema_version"):
        text = text.replace("---\n", "---\nschema_version: 2\n", 1)
    target = Path(user_skills_dir) / slug
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists")
    target.mkdir(parents=True, exist_ok=True)
    path = target / "SKILL.md"
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    issues, facts = lint_file(path)
    return {"slug": slug, "path": str(path), "issues": [i.__dict__ for i in issues], "facts": facts, "errors": [str(i) for i in issues if i.level == "error"]}


async def draft_from_handoff(llm: LLMClient, settings, handoff, turn, session, extra_notes: str = "") -> dict[str, Any]:
    summary = SessionSummary.from_session(session)
    method_name = handoff.skill or (handoff.description[:60] if handoff.description else "custom-analysis")
    contract = {}
    if handoff.parameters.get("contract"):
        contract = handoff.parameters["contract"]
    text = await draft_skill_text(llm, method_name=method_name, query=handoff.complete_query or turn.query, contract=contract, code_blocks=list(handoff.code), outputs=list(handoff.outputs), summary=summary, extra_notes=extra_notes, turn_id=turn.turn_id)
    result = install_skill_text(text, settings.user_skills_dir, overwrite=True)
    result["text"] = text
    return result


__all__ = ["draft_skill_text", "install_skill_text", "draft_from_handoff"]
