from __future__ import annotations

import json

import numpy as np
import pytest

from stat_agent.events import (
    JsonlSink,
    ListEmitter,
    StampingEmitter,
    catalogue,
    dumps,
    parse_event,
    to_jsonable,
    to_wire,
)
from stat_agent.prompts import available_prompts, load_prompt


def test_every_prompt_loads_and_has_stable_hash():
    names = sorted(available_prompts())
    assert {"planner", "matcher", "verifier", "generator_system", "generator_user", "reflector", "analyzer", "summarizer", "method_card"} <= set(names)
    for name in names:
        p = load_prompt(name)
        assert p.version >= 1 and len(p.sha256) == 64 and p.prompt_id == f"{p.id}/v{p.version}"


def test_planner_renders_verbatim_sections():
    p = load_prompt("planner")
    text = p.render(session_block="- Total slices: 1\n  - Slice 0: t (gene, cell-level)", history_block="", user_query="Annotate celltype", clarifications_block="")
    assert '**User Query:** "Annotate celltype"' in text
    assert '"needs_clarification": true/false' in text
    assert "Example 7 (Single slice - multiple analytical actions, separate steps)" in text


def test_analyzer_conditional_interpretation():
    p = load_prompt("analyzer")
    a = p.render(session_info="", history_block="", user_query="q", assembled_context="ctx", has_interpret=True, interpret_instruction="focus")
    b = p.render(session_info="", history_block="", user_query="q", assembled_context="ctx", has_interpret=False, interpret_instruction="")
    assert "'focus' - Provide biological interpretation" in a and '"key_findings": ["List of 1-3' in a
    assert '"interpretation": null' in b and '"key_findings": []' in b


def test_generator_system_prompt_is_verbatim_and_reflector_context_ok():
    g = load_prompt("generator_system").render()
    assert g.startswith("You are an expert spatial transcriptomics analysis assistant")
    assert "## Data Access API (CRITICAL - NEW SIMPLIFIED ARCHITECTURE)" in g
    r = load_prompt("reflector").render(user_query="q", original_code="x=1", error_msg="KeyError", stdout_tail="", available_rois={}, available_columns=["x"], has_celltype=False)
    assert "{'ROI_1': {'slice_id': '0', 'modality': 'gene', 'n_cells': 3686}}" in r
    assert "Output printed before the failure" not in r


def test_missing_variable_is_an_error():
    from jinja2.exceptions import UndefinedError

    with pytest.raises(UndefinedError):
        load_prompt("matcher").render(request="x")


def test_event_round_trip_and_wire():
    ev = catalogue.PromptEvent(prompt_id="p1", kind="verifier", questions=[{"id": "q1", "text": "Species?"}])
    wire = to_wire(ev)
    assert wire["type"] == "prompt" and wire["questions"][0]["text"] == "Species?"
    back = parse_event(json.loads(dumps(wire)))
    assert isinstance(back, catalogue.PromptEvent) and back.kind == "verifier"
    fig = catalogue.FigureEvent(figure_id="f1", png_b64="AAAA")
    assert "png_b64" not in to_wire(fig)
    assert len(catalogue.event_type_names()) >= 38


def test_safe_encoding():
    payload = {"a": np.float32(1.5), "b": np.array([1, 2]), "nan": float("nan"), "inf": np.inf, "s": {1, 2}, "bytes": b"\x00"}
    out = json.loads(dumps(payload))
    assert out["a"] == 1.5 and out["b"] == [1, 2] and out["nan"] is None and out["inf"] is None and sorted(out["s"]) == [1, 2]
    assert to_jsonable(np.int64(3)) == 3


def test_stamping_emitter_and_jsonl_sink(tmp_path):
    sink = JsonlSink(tmp_path / "events.jsonl")
    lst = ListEmitter()
    stamp = StampingEmitter(inner=lst, turn_id="t1")
    stamp.emit(catalogue.PhaseEvent(phase="planning"))
    stamp.emit(catalogue.TurnDone())
    assert [e.seq for e in lst.events] == [1, 2] and all(e.turn_id == "t1" for e in lst.events)
    for e in lst.events:
        sink.emit(e)
    rows = sink.read()
    assert [r["type"] for r in rows] == ["phase", "turn_done"] and rows[1]["seq"] == 2
