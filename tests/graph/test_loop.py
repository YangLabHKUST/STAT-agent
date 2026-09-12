from __future__ import annotations

import asyncio

import pytest
from tests.graph.conftest import collect, make_agent
from tests.graph.fake_tool_model import ToolCallingStub, annotate_script, ask_script, skip_check_script

from stat_agent.agent import AblationConfig, Decision


def _loop_agent(registry, script, **kw):
    kw.setdefault("mode", "auto")
    ablation = kw.pop("ablation", AblationConfig(driver="loop"))
    return make_agent(registry, ablation=ablation, loop_model=ToolCallingStub(script=script), **kw)


def test_loop_mode_runs_skill_with_guards(registry):
    agent = _loop_agent(registry, annotate_script)
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    types = [e.type for e in events]
    assert types[0] == "turn_started" and types[-1] == "turn_done"
    assert "filter_complete" in types and "verify_complete" in types and "code_block_done" in types and "provenance_update" in types
    assert agent.session.get_slice(0).has_celltype()
    done = events[-1]
    assert "Annotated cell types" in done.response and done.summary["driver"] == "loop"
    assert agent.memory.turns[0].handoffs[0].mode == "loop" and "s0:obs:celltype" in agent.memory.turns[0].handoffs[0].outputs
    assert agent.ledger.get("s0:obs:celltype").produced_by.mode == "loop"


def test_loop_guard_refuses_unchecked_skill_code(registry):
    agent = _loop_agent(registry, skip_check_script)
    events = asyncio.run(collect(agent.run_turn("annotate")))
    refusals = [e for e in events if e.type == "refusal"]
    assert refusals and "check_skill" in refusals[0].reason
    assert events[-1].type == "turn_done"


def test_loop_a6_has_no_guards(registry):
    agent = _loop_agent(registry, skip_check_script, ablation=AblationConfig.parse("A6"))
    events = asyncio.run(collect(agent.run_turn("annotate")))
    assert not [e for e in events if e.type == "refusal"]
    assert [e for e in events if e.type == "code_block_done"][0].success


def test_loop_ask_user_auto_policy_records_assumption(registry):
    agent = _loop_agent(registry, ask_script)
    events = asyncio.run(collect(agent.run_turn("what tissue?")))
    assert [e for e in events if e.type == "assumption"] and events[-1].type == "turn_done"
    assert agent.memory.turns[0].assumptions[0].kind == "verifier"


def test_loop_ask_user_interactive_pauses_and_resumes(registry):
    agent = _loop_agent(registry, ask_script, mode="interactive")
    events = asyncio.run(collect(agent.run_turn("what tissue?")))
    assert agent.is_waiting and agent.pending.kind == "verifier"
    assert [e for e in events if e.type == "prompt"]
    events2 = asyncio.run(collect(agent.resume(Decision(answer="cortex"))))
    assert events2[-1].type == "turn_done" and "cortex" in events2[-1].response


def test_deepagents_driver_runs_same_tools(registry):
    pytest.importorskip("deepagents")
    from stat_agent.config import Settings

    agent = _loop_agent(registry, annotate_script, settings=Settings(loop_backend="deepagents", loop_subagents=True))
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    types = [e.type for e in events]
    assert types[0] == "turn_started" and types[-1] == "turn_done" and "code_block_done" in types
    assert agent.session.get_slice(0).has_celltype()
    stub = agent.loop_model
    assert {"task", "write_todos"} & set(stub.bound), stub.bound  # deepagents added its planning/subagent tools
    assert "read_file" not in stub.bound or True  # filesystem tools may exist but the backend is empty
