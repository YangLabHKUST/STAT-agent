from __future__ import annotations

import asyncio
import json

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
    # every event of the step carries its number, also after the resume; the turn-level ones do not
    assert {e.step for e in events2 if e.type in ("prompt_answered", "verify_complete", "code_block_started", "code_block_done", "step_done")} == {1}
    assert done.step is None


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
    second = agent.pending.options[1]["slug"]
    events2 = asyncio.run(collect(agent.resume(Decision(option_index=2))))
    sel = [e for e in events2 if e.type == "skill_selected"][0]
    assert sel.skill == second == "mini-niche"
    assert events2[-1].type == "turn_done"


def test_skill_selection_free_text_must_name_an_option(registry):
    """'no' is not a choice: the prompt is repeated instead of silently picking annotation."""
    agent = make_agent(registry, kind="S1_cell_gene")
    asyncio.run(collect(agent.run_turn("ambiguous request")))
    assert agent.pending.kind == "skill_selection"
    events2 = asyncio.run(collect(agent.resume(Decision(answer="no"))))
    assert not [e for e in events2 if e.type == "skill_selected"]
    assert [e for e in events2 if e.type == "prompt"] and agent.pending.kind == "skill_selection"
    events3 = asyncio.run(collect(agent.resume(Decision(answer="niche"))))
    assert [e for e in events3 if e.type == "skill_selected"][0].skill == "mini-niche"


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


def test_direct_answer_turn_has_no_pipeline(registry):
    agent = make_agent(registry)
    events = asyncio.run(collect(agent.run_turn("What can you do for me?")))
    types_ = [e.type for e in events]
    assert types_[0] == "turn_started" and types_[-1] == "turn_done" and "match_complete" not in types_ and "prompt" not in types_
    assert [e for e in events if e.type == "text"][0].text.startswith("I can annotate")
    assert events[-1].state == "done" and "annotate cell types" in events[-1].response
    assert agent.memory.turns[0].handoffs[0].mode == "text" and agent.llm.meter.total.llm_calls == 1


def test_general_step_runs_without_method_card(registry):
    agent = make_agent(registry)
    events = asyncio.run(collect(agent.run_turn("describe the data")))
    types_ = [e.type for e in events]
    assert "method_card" not in types_ and "refusal" not in types_
    assert "unguarded_mode" in types_ and "code_block_done" in types_ and events[-1].state == "done"
    assert [e for e in events if e.type == "verify_complete"][0].decided_by == "bypass"
    assert [e for e in events if e.type == "step_done"][0].mode == "unguarded"


def test_skill_step_must_produce_its_declared_outputs(registry):
    """A skill step that only inspects (celltype already exists from the load) gets one continuation;
    when that only inspects too, the step is reported as partial and the old artefact is not claimed."""
    from tests.graph.conftest import default_responder

    def responder(role, messages, schema):
        if role == "generator":  # the verifier rewrites the query, so key on the role: every generation only inspects
            return "Just looking.\n```python\nprint(session.get_slice(0).adata.obs['celltype'].value_counts())\n```"
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", responder=responder, mode="auto")
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    vr = [e for e in events if e.type == "validate_result"]
    assert len(vr) == 2 and not vr[-1].passed and "not produced" in vr[-1].failures[0]
    assert not [e for e in events if e.type == "reflection"]
    issues = [e for e in events if e.type == "execution_issue"]
    assert issues and issues[-1].issue_type == "partial_success"
    assert events[-1].type == "turn_done" and events[-1].state == "blocked"
    art = agent.ledger.get("s0:obs:celltype")
    assert art is not None and (art.produced_by is None or art.produced_by.turn_id != "t0001")


def test_cancel_between_stages_stops_before_the_next_model_call(registry):
    from tests.graph.conftest import default_responder, make_agent

    calls: list[str] = []
    box: dict = {}

    def responder(role, messages, schema):
        calls.append(role)
        if role == "matcher":
            box["agent"].cancel()  # the user presses stop while the matcher call is in flight
        return default_responder(role, messages, schema)

    agent = make_agent(registry, mode="auto", responder=responder)
    box["agent"] = agent
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    assert "matcher" in calls and "verifier" not in calls and "generator" not in calls
    assert any(e.type == "turn_cancelled" for e in events)
    assert events[-1].type == "turn_done" and events[-1].state == "cancelled"
    assert agent.memory.turns[-1].status == "cancelled"


def test_cancel_during_code_interrupts_it_and_skips_analysis(registry):
    from tests.graph.conftest import default_responder, make_agent

    calls: list[str] = []

    def responder(role, messages, schema):
        calls.append(role)
        if role == "generator":
            return "Working.\n```python\nimport time\nstat_test_stop()\nfor _ in range(600):\n    time.sleep(0.05)\nprint('never')\n```"
        return default_responder(role, messages, schema)

    agent = make_agent(registry, mode="auto", responder=responder)
    agent.runtime.namespace["stat_test_stop"] = agent.cancel  # the code itself presses stop, then keeps running
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    done = [e for e in events if e.type == "code_block_done"]
    assert done and done[0].success is False and done[0].error.type == "Cancelled"
    assert "analyzer" not in calls and "reflector" not in calls
    assert events[-1].type == "turn_done" and events[-1].state == "cancelled"
    # the interrupted step is closed properly: a handoff, a step_done, a plan card that says cancelled
    step = [e for e in events if e.type == "step_done"]
    assert step and step[0].status == "cancelled" and step[0].mode == "cancelled"
    assert [e for e in events if e.type == "plan_updated"][-1].steps[0].status == "cancelled"
    rec = agent.memory.turns[-1]
    assert rec.status == "cancelled" and rec.handoffs and rec.handoffs[0].success is False and "stopped by the user" in rec.handoffs[0].unresolved[0]
    assert "stopped by the user" in rec.history_lines()


def test_incomplete_step_gets_one_continuation_on_the_live_namespace(registry):
    """The first generation only inspects; the continuation reuses its variables and produces the output."""
    from tests.graph.conftest import _prompt, default_responder

    calls: list[str] = []

    def responder(role, messages, schema):
        if role == "generator":
            calls.append(role)
            if "Continuation of this step" in _prompt(messages):
                return "Continuing.\n```python\ns.adata.obs['celltype'] = 'B cell'\nprint('annotated')\n```"
            return "Just looking.\n```python\ns = session.get_slice(0)\nprint(s.adata.obs['celltype'].value_counts())\n```"
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", responder=responder, mode="auto")
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    assert [e.passed for e in events if e.type == "validate_result"] == [False, True] and calls == ["generator", "generator"]
    assert not [e for e in events if e.type == "reflection"]
    assert events[-1].type == "turn_done" and events[-1].state == "done"
    assert agent.ledger.get("s0:obs:celltype").produced_by.turn_id == "t0001"


def test_card_that_names_no_method_runs_as_general_analysis(registry):
    """A generic workflow gets no contract: the card reports named_method false and the step runs the 0.1 way."""
    from tests.graph.conftest import default_responder

    def responder(role, messages, schema):
        if role == "matcher":
            return "[]"
        if role == "method_card":
            return json.dumps({"method_name": "custom pooled clustering", "description": "d" * 30, "filter_requirements": {"modalities": ["gene"], "data_levels": ["cell"]}, "prerequisites": [], "consumes": [{"kind": "layers", "key": "counts"}], "produces": [], "outline": [], "risks": [], "confidence": 0.8, "named_method": False})
        if role == "generator":
            return "Clustering.\n```python\ns = session.get_slice(0)\ns.adata.obs['my_cluster'] = 'c1'\nprint('done')\n```"
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", responder=responder, mode="auto")
    events = asyncio.run(collect(agent.run_turn("pool the cells and run my custom clustering")))
    assert not [e for e in events if e.type == "refusal"]
    um = [e for e in events if e.type == "unguarded_mode"]
    assert um and "general analysis" in um[0].reason
    assert events[-1].type == "turn_done" and events[-1].state == "done"
    assert "my_cluster" in agent.session.get_slice(0).adata.obs.columns


def test_step_that_writes_other_artefacts_is_noted_not_continued(registry):
    """Declared outputs missing but the step did write something: no continuation, no failure, one log line."""
    from tests.graph.conftest import default_responder

    calls: list[str] = []

    def responder(role, messages, schema):
        if role == "generator":
            calls.append(role)
            return "Working.\n```python\ns = session.get_slice(0)\ns.adata.obs['other_label'] = 'x'\nprint('wrote something else')\n```"
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", responder=responder, mode="auto")
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    vr = [e for e in events if e.type == "validate_result"]
    assert len(vr) == 1 and not vr[0].passed and calls == ["generator"]
    assert [e for e in events if e.type == "step_done"][0].status == "done"
    assert events[-1].type == "turn_done" and events[-1].state == "done"


def test_continuation_gives_the_analyser_both_blocks(registry):
    from tests.graph.conftest import _prompt, default_responder

    seen: dict = {}

    def responder(role, messages, schema):
        if role == "generator":
            if "Continuation of this step" in _prompt(messages):
                return "Continuing.\n```python\ns.adata.obs['celltype'] = 'B cell'\nprint('annotated')\n```"
            return "Just looking.\n```python\ns = session.get_slice(0)\nprint(s.adata.obs['celltype'].value_counts())\n```"
        if role == "analyzer":
            seen["prompt"] = _prompt(messages)
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", responder=responder, mode="auto")
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    assert events[-1].state == "done"
    assert seen["prompt"].count("**Code:**") == 2 and seen["prompt"].count("**Output:**") == 2
    assert seen["prompt"].index("Just looking") < seen["prompt"].index("value_counts") < seen["prompt"].index("annotated")


def test_later_steps_see_earlier_steps_and_the_namespace(registry):
    """Step 2's verifier, generator and analyser are told what step 1 did and which variables it left behind."""
    from tests.graph.conftest import _prompt, default_responder

    seen: dict[str, list[str]] = {"verifier": [], "generator": [], "analyzer": []}

    def responder(role, messages, schema):
        if role in seen:
            seen[role].append(_prompt(messages))
        return default_responder(role, messages, schema)

    agent = make_agent(registry, mode="auto", responder=responder)
    events = asyncio.run(collect(agent.run_turn("annotate cell types then detect niches")))
    assert [e.status for e in events if e.type == "step_done"] == ["done", "done"]
    first, second = seen["generator"]
    assert "**Completed steps in this turn" not in first and "**Python namespace (variables" not in first
    assert "**Completed steps in this turn" in second and "Step 1 [mini-annotate] (done): Annotated cells." in second
    assert "  changes: Added obs celltype." in second and "  variables added: s" in second
    assert "s.adata.obs['celltype'] = 'T cell'" in second  # step 1's code, so names and keys are exact
    assert "**Python namespace (variables" in second and "- s: DataSlice" in second
    assert "**Completed steps in this turn" in seen["verifier"][-1] and "**Completed steps in this turn" in seen["analyzer"][-1]
    assert [e for e in events if e.type == "plan_updated"][-1].steps[1].depends_on == [1]
    handoffs = agent.memory.turns[-1].handoffs
    assert handoffs[0].variables == ["s"] and handoffs[1].variables == []


def test_memory_off_arm_gets_neither_block(registry):
    from tests.graph.conftest import _prompt, default_responder

    seen: list[str] = []

    def responder(role, messages, schema):
        if role == "generator":
            seen.append(_prompt(messages))
        return default_responder(role, messages, schema)

    agent = make_agent(registry, mode="auto", responder=responder, ablation=AblationConfig(memory=False))
    asyncio.run(collect(agent.run_turn("annotate cell types then detect niches")))
    assert len(seen) == 2 and "**Completed steps in this turn" not in seen[1] and "**Python namespace (variables" not in seen[1]


def test_planner_sees_stale_artefacts(registry):
    """A direct answer about earlier results is written with the ledger's STALE marks in view."""
    from tests.graph.conftest import _prompt, default_responder

    seen: list[str] = []

    def responder(role, messages, schema):
        if role == "planner":
            seen.append(_prompt(messages))
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", responder=responder)
    assert agent.ledger.get("s0:obs:celltype") is not None
    agent.ledger.mark_stale("s0:obs:celltype", "the reference was replaced")
    events = asyncio.run(collect(agent.run_turn("what can you do for me?")))
    assert events[-1].type == "turn_done" and "Session Artefacts" in seen[0] and "celltype (STALE)" in seen[0]


def test_result_line_when_the_step_has_no_interpretation(registry):
    """A plain count has no interpretation: the analyser's summary is rendered as the result."""
    from tests.graph.conftest import default_responder

    def responder(role, messages, schema):
        if role == "generator":
            return "I'll count.\n```python\nprint('Cluster 3 contains 385 spots')\n```"
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", mode="auto", responder=responder)
    events = asyncio.run(collect(agent.run_turn("describe the data")))
    assert events[-1].type == "turn_done" and "**Result:** Annotated cells." in events[-1].response


def test_verifier_reask_is_capped(registry):
    """A verifier that keeps asking is stopped after two re-asks; the step blocks with advice."""
    from tests.graph.conftest import default_responder

    calls: list[str] = []

    def responder(role, messages, schema):
        calls.append(role)
        if role == "verifier":
            return json.dumps({"prerequisites_met": False, "missing_prerequisites": ["tissue"], "can_obtain_by_chat": True, "clarification_questions": ["What type of tissue is this?"]})
        return default_responder(role, messages, schema)

    agent = make_agent(registry, mode="auto", responder=responder)
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    assert calls.count("verifier") == 3 and "generator" not in calls
    assert [e for e in events if e.type == "advice"] and [e for e in events if e.type == "step_done"][0].status == "blocked"
    assert events[-1].type == "turn_done" and events[-1].state == "blocked"


def test_planner_reask_is_capped(registry):
    from tests.graph.conftest import default_responder

    calls: list[str] = []

    def responder(role, messages, schema):
        calls.append(role)
        if role == "planner":
            return json.dumps({"needs_clarification": True, "clarification_question": "Which slice would you like to analyze?"})
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S3_two_slice_gene", mode="auto", responder=responder)
    events = asyncio.run(collect(agent.run_turn("annotate")))
    assert calls.count("planner") == 3 and events[-1].type == "turn_done" and events[-1].state == "blocked"
    assert [e for e in events if e.type == "advice"] and agent.memory.turns[-1].status == "blocked"


def test_unguarded_note_is_outside_the_skill_banner(registry):
    from tests.graph.conftest import _prompt, default_responder

    seen: list[str] = []

    def responder(role, messages, schema):
        if role == "generator":
            seen.append(_prompt(messages))
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", mode="auto", responder=responder)
    asyncio.run(collect(agent.run_turn("describe the data")))
    asyncio.run(collect(agent.run_turn("annotate cell types")))
    general, skill = seen
    assert "no registered skill covers this request" in general and "SKILL GUIDANCE PROVIDED" not in general
    assert "SKILL GUIDANCE PROVIDED" in skill and "no registered skill" not in skill
    assert "**Declared outputs of this skill" in skill and "obs:celltype" in skill  # SK3


def test_auto_mode_blocks_on_a_planner_question_it_cannot_answer(registry):
    from tests.graph.conftest import default_responder

    calls: list[str] = []

    def responder(role, messages, schema):
        calls.append(role)
        if role == "planner":
            return json.dumps({"needs_clarification": True, "clarification_question": "Should I use raw counts or the normalised matrix?"})
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", mode="auto", responder=responder)
    events = asyncio.run(collect(agent.run_turn("compare something")))
    assert calls.count("planner") == 1 and events[-1].type == "turn_done" and events[-1].state == "blocked"
    assert not [e for e in events if e.type == "assumption"] and [e for e in events if e.type == "advice"]


def test_skill_defaults_reach_the_auto_policy(registry):
    """mini-annotate declares defaults: tissue; auto mode answers the tissue question from it."""
    agent = make_agent(registry, mode="auto")
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    a = [e for e in events if e.type == "assumption"][0]
    assert a.chosen == "brain cortex" and "skill default" in a.rationale


def test_verifier_sees_answers_from_earlier_turns(registry):
    from tests.graph.conftest import _prompt, default_responder

    seen: list[str] = []

    def responder(role, messages, schema):
        if role == "verifier":
            seen.append(_prompt(messages))
        return default_responder(role, messages, schema)

    agent = make_agent(registry, responder=responder)
    asyncio.run(collect(agent.run_turn("annotate cell types")))
    asyncio.run(collect(agent.resume(Decision(answers={agent.pending.questions[0]: "mouse liver"}))))
    asyncio.run(collect(agent.run_turn("annotate cell types")))
    assert "earlier turns" not in seen[0]
    assert "Answers the user gave in earlier turns" in seen[-1] and "mouse liver" in seen[-1]


def test_results_on_a_derived_object_are_accepted_without_continuation(registry):
    """The skill declares obs:celltype; the code writes it onto a copy: no continuation, no partial verdict."""
    from tests.graph.conftest import default_responder

    calls: list[str] = []

    def responder(role, messages, schema):
        calls.append(role)
        if role == "generator":
            return "Working on a copy.\n```python\ntmp = session.get_slice(0).adata.copy()\ntmp.obs['celltype'] = 'T cell'\nprint('labelled the copy')\n```"
        return default_responder(role, messages, schema)

    agent = make_agent(registry, mode="auto", responder=responder)
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    assert calls.count("generator") == 1
    assert [e for e in events if e.type == "validate_result"][0].passed is False  # the log still says what is missing on the slice
    assert not [e for e in events if e.type == "execution_issue"]
    assert [e for e in events if e.type == "step_done"][0].status == "done" and events[-1].state == "done"


def test_step_that_overwrites_loaded_data_is_flagged(registry):
    """S1_cell_gene ships obs:celltype; the annotation step rewrites it: a warning event and an analyser note."""
    from tests.graph.conftest import _prompt, default_responder

    seen: list[str] = []

    def responder(role, messages, schema):
        if role == "analyzer":
            seen.append(_prompt(messages))
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", mode="auto", responder=responder)
    events = asyncio.run(collect(agent.run_turn("annotate cell types")))
    warn = [e for e in events if e.type == "warning"]
    assert warn and "came with the dataset" in warn[0].message and "s0:obs:celltype" in warn[0].message
    assert "overwrote data that came with the dataset" in seen[-1] and "s0:obs:celltype" in seen[-1]
    assert events[-1].type == "turn_done" and events[-1].state == "done"


def test_analyser_sees_stale_artefacts_of_its_slices(registry):
    from tests.graph.conftest import _prompt, default_responder

    seen: list[str] = []

    def responder(role, messages, schema):
        if role == "analyzer":
            seen.append(_prompt(messages))
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", mode="auto", responder=responder)
    agent.ledger.mark_stale("s0:obs:x", "test edit")
    asyncio.run(collect(agent.run_turn("describe the data")))
    assert "currently marked STALE" in seen[-1] and "s0:obs:x" in seen[-1]


def test_request_naming_a_disabled_skill_gets_advice(registry):
    from tests.graph.conftest import default_responder

    calls: list[str] = []

    def responder(role, messages, schema):
        calls.append(role)
        return default_responder(role, messages, schema)

    registry.disable("mini-niche")
    try:
        agent = make_agent(registry, kind="S1_cell_gene", mode="auto", responder=responder)
        events = asyncio.run(collect(agent.run_turn("detect niches")))
    finally:
        registry.enable("mini-niche")
    advice = [e for e in events if e.type == "advice"]
    assert advice and "switched off" in advice[0].message and advice[0].skill == "mini-niche"
    assert "generator" not in calls and "method_card" not in calls
    assert [e for e in events if e.type == "step_done"][0].status == "blocked"


def test_no_summariser_call_after_a_stop(registry):
    from tests.graph.conftest import default_responder, make_agent

    calls: list[str] = []

    stop_next = {"on": True}

    def responder(role, messages, schema):
        calls.append(role)
        if role == "generator" and stop_next["on"]:
            return "Working.\n```python\nimport time\nstat_test_stop()\nfor _ in range(600):\n    time.sleep(0.05)\n```"
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", mode="auto", responder=responder)
    agent.memory.window_turns = 0  # every finished turn overflows the window
    agent.runtime.namespace["stat_test_stop"] = agent.cancel
    events = asyncio.run(collect(agent.run_turn("describe the data")))
    assert events[-1].state == "cancelled" and "summarizer" not in calls
    stop_next["on"] = False
    asyncio.run(collect(agent.run_turn("describe the data")))
    assert "summarizer" in calls


def test_output_lines_are_batched_in_order(registry):
    from tests.graph.conftest import default_responder

    def responder(role, messages, schema):
        if role == "generator":
            return "Printing.\n```python\nfor i in range(120):\n    print('line', i)\n```"
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", mode="auto", responder=responder)
    events = asyncio.run(collect(agent.run_turn("describe the data")))
    outs = [e for e in events if e.type == "exec_output"]
    lines = [ln.text for e in outs for ln in e.lines]
    assert lines == [f"line {i}" for i in range(120)] and len(outs) <= 6


def test_failed_step_stays_failed_in_the_plan_view(registry):
    """Independent steps: step 1 fails for good, step 2 still runs, and no later plan view repaints step 1 as done."""
    from tests.graph.conftest import default_responder

    def responder(role, messages, schema):
        if role == "planner":
            return json.dumps({"needs_clarification": False, "steps": [
                {"step_number": 1, "description": "crash please", "kind": "skill", "target_slice_ids": [0], "refined_query": "crash please"},
                {"step_number": 2, "description": "detect niches", "kind": "skill", "target_slice_ids": [0], "refined_query": "detect niches"}]})
        if role == "reflector":
            return json.dumps({"error_type": "KeyError", "root_cause": "boom", "fix_strategy": "none", "change_magnitude": "large", "fixed_code": None, "confidence": 0.9, "should_retry": False})
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", responder=responder, mode="auto")
    events = asyncio.run(collect(agent.run_turn("crash please and detect niches")))
    views = [e for e in events if e.type == "plan_updated"]
    first_failure = next(i for i, e in enumerate(views) if any(s.index == 1 and s.status == "failed" for s in e.steps))
    for e in views[first_failure:]:
        assert [s.status for s in e.steps if s.index == 1] == ["failed"], [(s.index, s.status) for s in e.steps]
    assert [s.status for s in views[-1].steps] == ["failed", "done"]
    assert [e for e in events if e.type == "step_done"][-1].status == "done"


def test_writes_on_a_copy_are_not_reported_as_produced(registry):
    """What the code assigned on a copy counts for the output check only; the analyser, the handoff and
    the report see the ledger's list (B39)."""
    from tests.graph.conftest import _prompt, default_responder

    seen: list[str] = []

    def responder(role, messages, schema):
        if role == "generator":
            return "I'll work on a copy.\n```python\ns = session.get_slice(0)\na = s.adata.copy()\na.obs['tmp_clusters'] = 'c1'\ns.adata.uns['copy_note'] = 'kept'\nprint('done')\n```"
        if role == "analyzer":
            seen.append(_prompt(messages))
        return default_responder(role, messages, schema)

    agent = make_agent(registry, kind="S1_cell_gene", mode="auto", responder=responder)
    asyncio.run(collect(agent.run_turn("describe the data")))
    handoff = agent.memory.turns[-1].handoffs[0]
    assert handoff.outputs == ["s0:uns:copy_note"]
    assert "s0:uns:copy_note" in seen[-1] and "s0:obs:tmp_clusters" not in seen[-1]
    assert agent.ledger.get("s0:obs:tmp_clusters") is None


def test_session_llm_config_is_set_for_skill_code(registry):
    """Skill code written for 0.1 reads ``session.llm_config``; the agent fills it from its model spec (B41)."""
    from tests.fixtures.sessions import make_session
    from tests.graph.conftest import default_responder

    from stat_agent.agent import Agent
    from stat_agent.llm import LLMClient
    from stat_agent.llm.models import make_spec

    session = make_session("S1_cell_gene")
    assert session.llm_config is None
    Agent(session, LLMClient(spec=make_spec("openai/gpt-4o", api_key="k"), responder=default_responder), registry=registry)
    cfg = session.llm_config
    assert (cfg["provider"], cfg["model"], cfg["api_key"]) == ("openai", "openai/gpt-4o", "k") and set(cfg) == {"provider", "model", "api_key", "base_url"}
