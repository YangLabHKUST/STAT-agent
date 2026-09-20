"""A turn of several steps reaches the conversation history with every step, not as its first step only."""

from __future__ import annotations

import asyncio
import json

from tests.graph.conftest import collect, default_responder, make_agent

from stat_agent.agent.nodes.finishing import _in_rotation, _shared


def test_every_step_keeps_a_share_of_the_summary():
    first, redo = "relabelled the cells at 50 um; " * 40, "redid the comparison: 703 genes, MMP12 +4.95"
    joined = _shared([first, redo])
    assert len(joined) <= 600 and joined.endswith(redo)  # the short step is whole and the long one took what it left
    assert joined.startswith(first[:500])
    two_long = _shared(["a" * 900, "b" * 900])
    assert len(two_long) == 600 and two_long.count("a") == two_long.count("b") == 299
    assert _shared(["only step " * 100]) == ("only step " * 100)[:600]  # one step: as before
    assert _shared(["", "kept"]) == "kept" and _shared([]) == ""


def test_findings_go_round_the_steps():
    assert _in_rotation([["a1", "a2", "a3"], ["b1", "b2", "b3"]])[:3] == ["a1", "b1", "a2"]
    assert _in_rotation([["a1", "a2", "a3"], [], ["c1"]]) == ["a1", "c1", "a2", "a3"]
    assert _in_rotation([["a1", "a2"]]) == ["a1", "a2"] and _in_rotation([]) == []


def _relabel_then_redo(role, messages, schema):
    """Two general steps; the second one redoes a comparison and reports the numbers that replace the earlier ones."""
    text = "\n".join(str(getattr(m, "content", m)) for m in messages)
    if role == "planner":
        return json.dumps({"needs_clarification": False, "steps": [
            {"step_number": 1, "description": "describe the new labels", "kind": "general", "target_slice_ids": [0], "refined_query": "describe the new labels"},
            {"step_number": 2, "description": "describe the redone comparison", "kind": "general", "target_slice_ids": [0], "refined_query": "describe the redone comparison", "depends_on": [1]},
        ]})
    if role == "generator":
        return "Redoing.\n```python\nprint('redo ran')\n```" if "redone comparison" in text else "Relabelling.\n```python\nprint('relabel ran')\n```"
    if role == "analyzer":
        if "redo ran" in text:
            return json.dumps({"response_summary": "Redid the comparison with the 50 um labels: 703 genes each way.", "data_changes": "uns de_results recomputed.", "key_findings": ["MMP12 log2FC +4.95 at 50 um", "SPP1 +1.96", "SELENOP +0.73"], "execution_issues": {"has_issues": False}})
        return json.dumps({"response_summary": "Relabelled the macrophages at 50 um. " * 30, "data_changes": "obs macrophage_class overwritten. " * 30, "key_findings": ["14,749 TAMs", "7,138 normal", "2,352 moved"], "execution_issues": {"has_issues": False}})
    return default_responder(role, messages, schema)


def test_the_second_step_of_a_turn_is_in_the_history_of_the_next(registry):
    agent = make_agent(registry, responder=_relabel_then_redo)
    events = asyncio.run(collect(agent.run_turn("tighten it, then redo")))
    assert events[-1].type == "turn_done" and events[-1].state == "done"
    turn = agent.memory.turns[-1]
    assert [h.step_number for h in turn.handoffs] == [1, 2]
    history = agent.memory.history_text()  # what the roles of the next turn are given
    assert "703 genes each way" in history and "uns de_results recomputed" in history  # the redo's summary and data changes
    assert "MMP12 log2FC +4.95 at 50 um" in history and "14,749 TAMs" in history  # a finding of each step among the three shown
    assert len(turn.response_summary) <= 600 and len(turn.data_changes) <= 600
