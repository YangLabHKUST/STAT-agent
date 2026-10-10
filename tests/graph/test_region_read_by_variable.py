"""A result computed from regions turns stale when one of them is replaced, however the code reached the region:
through a loop variable, in an earlier block of the step, or through a data object kept from an earlier turn."""

from __future__ import annotations

import asyncio
import json

from tests.graph.conftest import collect, default_responder, make_agent

from stat_agent.events import ListEmitter
from stat_agent.ledger import SessionMutator

BBOX = {"type": "bbox", "x_min": 0, "x_max": 50, "y_min": 0, "y_max": 50}
LARGER = {"type": "bbox", "x_min": 0, "x_max": 80, "y_min": 0, "y_max": 80}
_LOOP = """Composition per region.
```python
counts = {}
for name in ['dcis_duct', 'invasive_front']:
    roi = session.get_roi(name)
    counts[name] = int(roi.adata.n_obs)
session.get_slice(0).adata.uns['roi_cell_counts'] = counts
print(counts)
```"""


def _loops_over_the_region_names(role, messages, schema):
    if role == "planner":
        return json.dumps({"needs_clarification": False, "steps": [{"step_number": 1, "description": "count the cells of each region", "kind": "general", "target_slice_ids": [0], "refined_query": "count the cells of each region"}]})
    return _LOOP if role == "generator" else default_responder(role, messages, schema)


def test_replacing_a_region_read_through_a_loop_variable_marks_its_result_stale(registry):
    agent = make_agent(registry, responder=_loops_over_the_region_names)
    viewer = SessionMutator(agent.session, agent.ledger, emitter=ListEmitter())
    for name in ("dcis_duct", "invasive_front"):
        viewer.create_roi(name, 0, BBOX)
    events = asyncio.run(collect(agent.run_turn("count the cells of each region")))
    assert events[-1].type == "turn_done" and events[-1].state == "done"
    assert agent.ledger.get("s0:uns:roi_cell_counts").inputs == ["roi:dcis_duct", "roi:invasive_front"]
    deleted = viewer.delete_roi("invasive_front")  # the viewer replaces a region by deleting it and drawing the new one
    viewer.create_roi("invasive_front", 0, LARGER)
    assert deleted.delta.entry.stale_now == ["s0:uns:roi_cell_counts"] and agent.ledger.is_stale("s0:uns:roi_cell_counts")
    assert "STALE" in agent.ledger.state_text([0])  # what the next turn's roles are shown


_TWO_BLOCKS = """First the cells of each region.
```python
parts = {name: session.get_roi(name).adata[:, :5].copy() for name in ('dcis_duct', 'invasive_front')}
sizes = {name: part.n_obs for name, part in parts.items()}
print(sizes)
```
Then the comparison, stored on the slice.
```python
gap = float(parts['dcis_duct'].X.mean() - parts['invasive_front'].X.mean())
session.get_slice(0).adata.uns['region_gap'] = {'gap': gap}
print(gap)
```"""


def _two_blocks_then_a_later_turn(role, messages, schema):
    text = "\n".join(str(getattr(m, "content", m)) for m in messages)
    if role == "planner":
        q = "store the mean of the kept cells" if "kept cells" in text else "compare the two regions"
        return json.dumps({"needs_clarification": False, "steps": [{"step_number": 1, "description": q, "kind": "general", "target_slice_ids": [0], "refined_query": q}]})
    if role == "generator":
        if "kept cells" in text:
            return "From the object kept earlier.\n```python\nsession.get_slice(0).adata.uns['kept_mean'] = {'mean': float(parts['invasive_front'].X.mean())}\n```"
        return _TWO_BLOCKS
    return default_responder(role, messages, schema)


def test_a_region_read_in_an_earlier_block_or_turn_is_an_input_of_what_is_written_later(registry):
    agent = make_agent(registry, responder=_two_blocks_then_a_later_turn)
    viewer = SessionMutator(agent.session, agent.ledger, emitter=ListEmitter())
    for name in ("dcis_duct", "invasive_front"):
        viewer.create_roi(name, 0, BBOX)
    events = asyncio.run(collect(agent.run_turn("compare the two regions")))
    assert events[-1].type == "turn_done" and events[-1].state == "done"
    assert len([e for e in events if e.type == "code_block_done" and e.success]) == 2
    assert agent.ledger.get("s0:uns:region_gap").inputs == ["roi:dcis_duct", "roi:invasive_front"]  # written by block 2, read by block 1
    events = asyncio.run(collect(agent.run_turn("store the mean of the kept cells")))
    assert events[-1].type == "turn_done" and events[-1].state == "done"
    assert agent.ledger.get("s0:uns:kept_mean").inputs == ["roi:dcis_duct", "roi:invasive_front"]  # ``parts`` came from both regions
    deleted = viewer.delete_roi("invasive_front")
    assert deleted.delta.entry.stale_now == ["s0:uns:kept_mean", "s0:uns:region_gap"]
