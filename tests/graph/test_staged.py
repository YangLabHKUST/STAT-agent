from __future__ import annotations

import asyncio

from tests.graph.conftest import collect, make_agent

from stat_agent.agent import AblationConfig, Decision


def types(events):
    return [e.type for e in events]


def test_single_step_with_verifier_pause_and_resume(registry):
    agent = make_agent(registry)
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    t = types(events)
    assert t[0] == "turn_started" and "plan_updated" in t and "filter_complete" in t and "match_complete" in t
    assert events[-1].type == "prompt" or "prompt" in t
    assert agent.is_waiting and agent.pending.kind == "verifier" and agent.pending.skill == "mini-annotate"
    assert not agent.session.get_slice(0).has_celltype()
    events2 = asyncio.run(collect(agent.resume(Decision(answers={agent.pending.questions[0]: "brain cortex"}))))
    t2 = types(events2)
    assert "prompt_answered" in t2 and "verify_complete" in t2 and "code_block_started" in t2 and "code_block_done" in t2
    assert "provenance_update" in t2 and "validate_result" in t2 and t2[-1] == "turn_done"
    assert agent.session.get_slice(0).has_celltype()
    assert agent.ledger.has_live("s0:obs:celltype") and agent.ledger.get("s0:obs:celltype").produced_by.skill == "mini-annotate"
    done = events2[-1]
    assert done.state == "done" and "annotate" in done.response.lower() and "T cells dominate" in done.response
    assert agent.memory.turns[0].handoffs[0].outputs == ["s0:obs:celltype"]
    assert agent.llm.meter.total.llm_calls >= 4  # planner, matcher, verifier x2, generator, analyzer
    exec_lines = [ln.text for e in events2 if e.type == "exec_output" for ln in e.lines]
    assert "annotated 600 cells" in exec_lines


def test_auto_mode_records_assumptions_instead_of_pausing(registry):
    agent = make_agent(registry, mode="auto")
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    t = types(events)
    assert not agent.is_waiting and t[-1] == "turn_done"
    assumptions = [e for e in events if e.type == "assumption"]
    assert assumptions and assumptions[0].kind == "verifier" and assumptions[0].chosen
    assert agent.memory.turns[0].assumptions[0].source == "policy_default"


def test_planner_clarification_roundtrip(registry):
    agent = make_agent(registry, kind="S3_two_slice_gene", mode="interactive")
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    assert agent.is_waiting and agent.pending.kind == "planner"
    assert [e for e in events if e.type == "prompt"][0].question.startswith("Which slice")
    events2 = asyncio.run(collect(agent.resume("slice 1")))
    plan = [e for e in events2 if e.type == "plan_updated"][0]
    assert plan.steps[0].target_slices == [1]
    # verifier for mini-annotate pauses next; answer it
    assert agent.is_waiting and agent.pending.kind == "verifier"
    events3 = asyncio.run(collect(agent.resume("cortex")))
    assert events3[-1].type == "turn_done"


def test_out_of_registry_request_is_refused_with_requirement(registry):
    agent = make_agent(registry, mode="auto")
    events = asyncio.run(collect(agent.run_turn("run RNA velocity")))
    t = types(events)
    assert "method_card" in t and "refusal" in t and "code_block_started" not in t
    ref = [e for e in events if e.type == "refusal"][0]
    assert ref.kind == "missing_prereq" and "layers:spliced" in (ref.requirement or "")
    assert events[-1].type == "turn_done" and events[-1].state == "blocked"


def test_reflection_retries_once_then_succeeds(registry):
    agent = make_agent(registry, mode="auto")
    events = asyncio.run(collect(agent.run_turn("crash please")))
    dones = [e for e in events if e.type == "code_block_done"]
    assert dones[0].success is False and dones[0].will_retry is True and dones[0].attempt == 1
    assert dones[-1].success is True and dones[-1].attempt == 2
    assert [e for e in events if e.type == "reflection"][0].will_retry
    assert events[-1].type == "turn_done" and events[-1].state == "done"


def test_no_reflector_ablation_fails_fast(registry):
    agent = make_agent(registry, mode="auto", ablation=AblationConfig(reflector=False))
    events = asyncio.run(collect(agent.run_turn("crash please")))
    dones = [e for e in events if e.type == "code_block_done"]
    assert len(dones) == 1 and not dones[0].success and not dones[0].will_retry
    assert "execution_issue" in types(events) and events[-1].state == "blocked"
    assert agent.memory.turns[0].handoffs[0].success is False


def test_ablation_bypasses_filter_and_verifier(registry):
    agent = make_agent(registry, mode="auto", ablation=AblationConfig.parse("filter=off,verifier=off"))
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    f = [e for e in events if e.type == "filter_complete"][0]
    assert f.bypassed and len(f.compatible) == 4
    v = [e for e in events if e.type == "verify_complete"][0]
    assert v.decided_by == "bypass" and "assumption" not in types(events)
    assert events[-1].state == "done"


def test_skill_selection_prompt_and_option_answer(registry):
    agent = make_agent(registry, kind="S1_cell_gene")
    events = asyncio.run(collect(agent.run_turn("ambiguous request")))
    assert [e for e in events if e.type == "prompt"][0].kind == "skill_selection"
    assert agent.is_waiting and agent.pending.kind == "skill_selection" and len(agent.pending.options) == 2
    events2 = asyncio.run(collect(agent.resume(Decision(option_index=2))))
    sel = [e for e in events2 if e.type == "skill_selected"][0]
    assert sel.skill == agent.pending.options[1]["slug"] if agent.pending else True
    assert events2[-1].type in ("turn_done", "prompt")


def test_multi_step_dependency_stops_turn_on_failure(registry):
    def responder(role, messages, schema):
        from tests.graph.conftest import default_responder

        if role == "reflector":
            return '{"error_type": "KeyError", "root_cause": "structural", "fix_strategy": "rewrite", "change_magnitude": "large", "fixed_code": null, "confidence": 0.3, "should_retry": false}'
        return default_responder(role, messages, schema)

    agent = make_agent(registry, mode="auto", responder=responder)
    events = asyncio.run(collect(agent.run_turn("crash please then detect niches")))
    steps = [e for e in events if e.type == "step_done"]
    assert steps[0].status == "failed" and len(steps) == 1
    assert events[-1].type == "turn_done" and events[-1].state == "blocked"


def test_contract_gate_declines_when_consumes_missing(registry):
    agent = make_agent(registry, kind="S1_cell_gene_noCT", mode="auto")
    events = asyncio.run(collect(agent.run_turn("detect niches")))
    t = types(events)
    assert "advice" in t and "code_block_started" not in t
    v = [e for e in events if e.type == "verify_complete"][0]
    assert v.decided_by == "contract"
    assert agent.llm.meter.by_role.get("verifier") is None  # no LLM call for the verifier


def test_events_are_stamped_and_written_to_jsonl(registry, tmp_path):
    agent = make_agent(registry, mode="auto", session_dir=tmp_path / "s")
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs) and all(e.turn_id == "t0001" for e in events)
    assert (tmp_path / "s" / "events.jsonl").exists() and (tmp_path / "s" / "turn_001.md").exists()
    assert list((tmp_path / "s" / "calls").glob("*.json"))
