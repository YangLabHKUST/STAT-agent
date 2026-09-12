"""Scripted responder that answers each role from the prompt text (no network, no fixtures)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from tests.fixtures.sessions import make_session

from stat_agent.agent import Agent
from stat_agent.llm import CallLogger, LLMClient, UsageMeter
from stat_agent.llm.client import content_text
from stat_agent.skills import SkillRegistry

SKILLS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "skills_mini"


def _prompt(messages) -> str:
    return "\n".join(content_text(m) for m in messages)


def default_responder(role: str, messages, schema):
    text = _prompt(messages)
    if role == "planner":
        m = re.search(r'\*\*User Query:\*\* "(.*?)"', text, re.S)
        q = m.group(1) if m else ""
        n_slices = int(re.search(r"Total slices: (\d+)", text).group(1))
        answered = "Previous Clarifications" in text
        if n_slices > 1 and "slice" not in q.lower() and not answered:
            return json.dumps({"needs_clarification": True, "clarification_question": "Which slice would you like to analyze?"})
        target = [1] if (answered and "slice 1" in text.split("Previous Clarifications")[-1]) else ([int(x) for x in re.findall(r"slice (\d+)", q.lower())] or [0])
        if "then" in q.lower():
            first, second = q.split(" then ", 1)
            return json.dumps({"needs_clarification": False, "steps": [
                {"step_number": 1, "description": first, "target_slice_ids": target, "refined_query": first},
                {"step_number": 2, "description": second, "target_slice_ids": target, "refined_query": second, "depends_on": [1]},
            ]})
        return json.dumps({"needs_clarification": False, "steps": [{"step_number": 1, "description": q, "target_slice_ids": target, "refined_query": q}]})
    if role == "matcher":
        req = re.search(r'User Request: "(.*)"', text).group(1).lower()
        catalog = re.findall(r"- \*\*([a-z0-9-]+)\*\*:", text)
        if "ambiguous" in req:
            return json.dumps([s for s in catalog if s in ("mini-annotate", "mini-niche")][:2])
        for key, slug in (("annotat", "mini-annotate"), ("niche", "mini-niche"), ("deconv", "mini-deconv"), ("crash", "mini-crash")):
            if key in req and slug in catalog:
                return json.dumps([slug])
        return "[]"
    if role == "verifier":
        skill = re.search(r"\*\*Selected Skill:\*\* .*?\(([a-z0-9-]+)\)", text).group(1)
        if "Previously Collected Information" in text or skill in ("mini-niche", "mini-crash"):
            return json.dumps({"prerequisites_met": True, "complete_query": "verified query for " + skill})
        if skill == "mini-deconv":
            return json.dumps({"prerequisites_met": False, "missing_prerequisites": ["reference"], "can_obtain_by_chat": True, "clarification_questions": ["Please provide the path to the reference dataset (.h5ad file)"]})
        return json.dumps({"prerequisites_met": False, "missing_prerequisites": ["tissue"], "can_obtain_by_chat": True, "clarification_questions": ["What type of tissue is this?"]})
    if role == "generator":
        q = re.search(r"\*\*User Query:\*\* (.*)", text).group(1).lower()
        if "crash" in q:
            return "I'll try something risky.\n```python\nraise KeyError('boom')\n```"
        if "niche" in q:
            return "I'll detect niches.\n```python\ns = session.get_slice(0)\ns.adata.obs['niche_label'] = 'n1'\nprint('niches done')\n```"
        if "velocity" in q or "no code" in q:
            return "This request cannot be executed with the loaded data."
        return "I'll annotate the cells.\n```python\ns = session.get_slice(0)\ns.adata.obs['celltype'] = 'T cell'\nprint('annotated 600 cells')\n```\n__INTERPRET__: cell type composition"
    if role == "reflector":
        return json.dumps({"error_type": "KeyError", "root_cause": "boom", "fix_strategy": "print instead", "change_magnitude": "small", "fixed_code": "print('fixed on retry')", "confidence": 0.9, "should_retry": True})
    if role == "analyzer":
        return json.dumps({"response_summary": "Annotated cells.", "data_changes": "Added obs celltype.", "interpretation": "T cells dominate." if "cell type composition" in text else None, "key_findings": ["600 T cells"] if "cell type composition" in text else [], "execution_issues": {"has_issues": "**Error:**" in text, "issue_type": "error" if "**Error:**" in text else None, "explanation": "failed" if "**Error:**" in text else None}})
    if role == "method_card":
        return json.dumps({"method_name": "scVelo RNA velocity", "description": "RNA velocity from spliced and unspliced counts", "filter_requirements": {"modalities": ["gene"], "data_levels": ["cell"]}, "prerequisites": ["spliced and unspliced count layers"], "consumes": [{"kind": "layers", "key": "spliced"}], "produces": [{"kind": "obsm", "key": "velocity"}], "outline": ["compute moments", "estimate velocity"], "risks": ["needs unspliced counts"], "confidence": 0.7})
    if role == "summarizer":
        return json.dumps({"summary": "summary", "facts": []})
    return "{}"


@pytest.fixture
def registry():
    reg = SkillRegistry([(SKILLS_MINI, "test")])
    reg.load()
    assert len(reg) == 4 and not reg.broken
    return reg


def make_agent(registry, kind="S1_cell_gene_noCT", responder=default_responder, **kw):
    llm = LLMClient(None, calllog=CallLogger(), meter=UsageMeter(), model_name="scripted", responder=responder)
    session = make_session(kind)
    return Agent(session, llm, registry=registry, **kw)


@pytest.fixture
def make_agent_fixture(registry):
    return lambda **kw: make_agent(registry, **kw)


async def collect(aiter):
    return [ev async for ev in aiter]
