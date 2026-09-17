"""A skill whose prerequisite is a file never runs on an invented path (B60).

The failure this pins down (stat40 Q31, 2026-09-17): no reference file in the session, the matcher went
straight to the reference-based annotation skill, the verifier asked for the path, the user gave no path
("please proceed with a reasonable default"), the second verifier pass called the prerequisite met and wrote
an invented path into the query, and the generated code failed on it.  Now the file prerequisite stays unmet
without a file, and the step switches to a compatible skill that produces the same result without a file;
with a file in the session routing is unchanged, and with no such skill the step ends in advice.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil

import pytest
from tests.graph.conftest import SKILLS_MINI, collect, default_responder, make_agent

from stat_agent.agent.policy import ScriptedPolicy
from stat_agent.core.view import SessionSummary
from stat_agent.roles.verifier import file_in_session, required_file_prerequisites
from stat_agent.skills import SkillRegistry
from stat_agent.skills.schema import SkillFrontmatter

FALLBACK = "Please proceed with a reasonable default and state the assumption you made."

REF_SKILL = """---
schema_version: 2
name: mini-annotate-ref
title: Mini Reference Annotation
slug: mini-annotate-ref
description: Annotate cell types by label transfer from an annotated single-cell reference (test skill).
filter_requirements:
  modalities: [gene]
  data_levels: [cell]
prerequisites:
  - "Annotated single-cell reference dataset path (.h5ad file)"
produces:
  - obs:celltype
default_skill: true
---

# Mini reference annotation

```python
s = session.get_slice(0)
s.adata.obs['celltype'] = 'T cell'
print('annotated from the reference')
```
"""


@pytest.fixture
def registry_with_ref(tmp_path):
    root = tmp_path / "skills"
    shutil.copytree(SKILLS_MINI, root)
    (root / "mini-annotate-ref").mkdir()
    (root / "mini-annotate-ref" / "SKILL.md").write_text(REF_SKILL, encoding="utf-8")
    reg = SkillRegistry([(root, "test")])
    reg.load()
    assert "mini-annotate-ref" in reg.enabled() and not reg.broken
    return reg


def responder(role, messages, schema):
    """The default script, except that the matcher picks the reference skill directly and its verifier asks for
    the file first and calls everything met once any answer exists (the model behaviour seen in Q31)."""
    text = "\n".join(getattr(m, "content", "") for m in messages)
    if role == "matcher" and "mini-annotate-ref" in text:
        req = re.search(r'User Request: "(.*)"', text).group(1).lower()
        if "annotat" in req:
            return json.dumps(["mini-annotate-ref"])
    if role == "verifier" and "(mini-annotate-ref)" in text and "Previously Collected Information" not in text:
        return json.dumps({"prerequisites_met": False, "missing_prerequisites": ["Annotated single-cell reference dataset path (.h5ad file)"], "can_obtain_by_chat": True, "clarification_questions": ["Please provide the path to the annotated reference dataset (.h5ad file)."]})
    if role == "verifier" and "(mini-annotate-ref)" in text:
        return json.dumps({"prerequisites_met": True, "complete_query": "Annotate cell types using the reference at ./reference_mouse_brain.h5ad (assumed default)."})
    return default_responder(role, messages, schema)


def _agent(registry, **kw):
    return make_agent(registry, kind="S1_cell_gene_noCT", responder=responder, mode="interactive", policy=ScriptedPolicy([], skill_pick=1, default=FALLBACK), **kw)


def _calls(agent, role, marker=""):
    return [r for r in agent.llm.calllog.records if r.role == role and marker in "\n".join(m["content"] for m in r.messages)]


def test_no_file_in_session_switches_to_the_skill_that_needs_none(registry_with_ref):
    agent = _agent(registry_with_ref)
    events = asyncio.run(collect(agent.run_turn("annotate cell types in this brain cortex tissue")))
    selected = [e for e in events if e.type == "skill_selected"]
    assert [e.skill for e in selected] == ["mini-annotate-ref", "mini-annotate"]
    assert selected[1].alternatives == ["mini-annotate-ref"]
    declined = [e for e in events if e.type == "verify_complete" and e.skill == "mini-annotate-ref"][-1]
    assert declined.decided_by == "contract" and declined.verdicts[0].status == "not_obtainable"
    # the model was asked once for the reference skill; its second pass (the one that invented a path) never ran
    assert len(_calls(agent, "verifier", "(mini-annotate-ref)")) == 1
    assert agent.last_status == "done" and agent.session.get_slice(0).has_celltype()
    generator_prompts = ["\n".join(m["content"] for m in r.messages) for r in _calls(agent, "generator")]
    assert generator_prompts and not any(".h5ad" in p and "reference_mouse_brain" in p for p in generator_prompts)
    assert all("Mini Cell Type Annotation" in p for p in generator_prompts)


def test_a_file_in_the_session_leaves_routing_unchanged(registry_with_ref):
    agent = _agent(registry_with_ref)
    agent.session.metadata["extra_files"] = ["ref/reference.h5ad"]
    events = asyncio.run(collect(agent.run_turn("annotate cell types in this brain cortex tissue")))
    assert [e.skill for e in events if e.type == "skill_selected"] == ["mini-annotate-ref"]
    assert len(_calls(agent, "verifier", "(mini-annotate-ref)")) == 2  # the model decides the second pass, as before
    assert [e.decided_by for e in events if e.type == "verify_complete"][-1] == "llm"


def test_no_file_and_no_alternative_ends_in_advice_without_code(registry):
    # mini-deconv needs a reference file and no other skill produces deconvolution weights
    agent = make_agent(registry, kind="S2_spot_deconv", mode="interactive", policy=ScriptedPolicy([], skill_pick=1, default=FALLBACK))
    events = asyncio.run(collect(agent.run_turn("deconvolve the spots")))
    assert [e.skill for e in events if e.type == "skill_selected"] == ["mini-deconv"]
    advice = [e for e in events if e.type == "advice"]
    assert advice and "no such file" in advice[-1].message
    assert not [e for e in events if e.type == "code_block_started"]
    assert not _calls(agent, "generator")


def test_file_prerequisite_helpers():
    def meta(prereqs):
        return SkillFrontmatter.model_validate({"name": "x", "slug": "x", "description": "d", "prerequisites": prereqs})

    assert required_file_prerequisites(meta(["Annotated reference scRNA-seq dataset path (.h5ad file)", "Cell type column name"])) == ["Annotated reference scRNA-seq dataset path (.h5ad file)"]
    assert required_file_prerequisites(meta(["Optional: path to a reference scRNA-seq h5ad file with celltype annotations"])) == []
    assert required_file_prerequisites(meta(["Tissue type information (e.g., breast cancer)"])) == []
    assert file_in_session(SessionSummary(extra_files=["ref/r.h5ad"]), [])
    assert not file_in_session(SessionSummary(), ["use ./reference_mouse_brain_that_does_not_exist.h5ad"])


def test_an_existing_path_named_by_the_user_counts(tmp_path):
    ref = tmp_path / "my_reference.h5ad"
    ref.write_bytes(b"")
    assert file_in_session(SessionSummary(), [f"it is at {ref}"])
    assert file_in_session(SessionSummary(dataset_dir=str(tmp_path)), ["the file my_reference.h5ad"])


def test_a_model_that_calls_the_file_met_on_the_first_pass_is_asked_for_the_file(registry_with_ref):
    def eager(role, messages, schema):
        text = "\n".join(getattr(m, "content", "") for m in messages)
        if role == "verifier" and "(mini-annotate-ref)" in text:
            return json.dumps({"prerequisites_met": True, "complete_query": "Annotate using ./made_up_reference.h5ad"})
        return responder(role, messages, schema)

    agent = make_agent(registry_with_ref, kind="S1_cell_gene_noCT", responder=eager, mode="interactive", policy=ScriptedPolicy([], skill_pick=1, default=FALLBACK))
    events = asyncio.run(collect(agent.run_turn("annotate cell types in this brain cortex tissue")))
    asked = [e for e in events if e.type == "prompt" and e.kind == "verifier" and any("no such file is loaded" in q.text for q in e.questions)]
    assert asked, "the met verdict without a file must become a question for the file"
    assert [e.skill for e in events if e.type == "skill_selected"] == ["mini-annotate-ref", "mini-annotate"]
    assert not any("made_up_reference" in "\n".join(m["content"] for m in r.messages) for r in _calls(agent, "generator"))
