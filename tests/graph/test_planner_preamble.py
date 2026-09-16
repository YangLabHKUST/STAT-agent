"""A turn that asks something and then asks for work ("first tell me what this changes, then redo
it") gets both halves: the planner's answer is said before its steps run, not dropped."""

from __future__ import annotations

import asyncio
import json

from tests.fixtures.sessions import make_session
from tests.graph.conftest import collect, default_responder, make_agent

from stat_agent.core.view import SessionSummary
from stat_agent.roles.planner import PlannerRole, PlanOutput, PlanStep

ANSWER = "The counts and the differential expression change; the marker split does not."


def _summary() -> SessionSummary:
    return SessionSummary.from_session(make_session("S1_cell_gene"))


def test_an_answer_given_with_steps_becomes_the_preamble():
    out = PlanOutput(direct_answer=ANSWER, steps=[PlanStep(step_number=1, description="redo the niches", target_slice_ids=[0], refined_query="redo the niches")])
    res = PlannerRole.validate(out, _summary())
    assert not res.needs_clarification and res.direct_answer is None
    assert res.preamble == ANSWER and len(res.steps) == 1


def test_an_answer_given_alone_is_still_the_whole_turn():
    res = PlannerRole.validate(PlanOutput(direct_answer=ANSWER, steps=[]), _summary())
    assert res.direct_answer == ANSWER and not res.steps and res.preamble == ""


def test_a_plan_without_an_answer_has_no_preamble():
    out = PlanOutput(steps=[PlanStep(step_number=1, description="detect niches", target_slice_ids=[0], refined_query="detect niches")])
    assert PlannerRole.validate(out, _summary()).preamble == ""


def _responder_with_preamble(role, messages, schema):
    if role == "planner":
        return json.dumps({
            "needs_clarification": False,
            "direct_answer": ANSWER,
            "steps": [{"step_number": 1, "description": "detect niches", "kind": "skill", "target_slice_ids": [0], "refined_query": "detect niches"}],
        })
    return default_responder(role, messages, schema)


def test_the_preamble_is_said_before_the_steps_run(registry):
    agent = make_agent(registry, kind="S1_cell_gene", responder=_responder_with_preamble)
    events = asyncio.run(collect(agent.run_turn("first tell me what this changes, then redo the niches")))
    kinds = [e.type for e in events]
    texts = [e for e in events if e.type == "text" and e.role == "answer"]
    assert ANSWER in [t.text for t in texts]
    # said before the work: the preamble precedes the plan, the step and the code
    first_text = min(i for i, e in enumerate(events) if e.type == "text" and e.text == ANSWER)
    assert first_text < kinds.index("step_started") < kinds.index("code_block_started")
    assert kinds[-1] == "turn_done" and events[-1].state == "done"
    assert agent.session.get_slice(0).adata.obs["niche_label"].iloc[0] == "n1"  # the steps still ran
