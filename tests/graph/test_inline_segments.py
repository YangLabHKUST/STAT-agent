"""Prose and code events carry their position in the step's generation, so a client can render them in order."""

from __future__ import annotations

import asyncio

from tests.graph.conftest import collect, default_responder, make_agent

from stat_agent.agent.nodes.execution import _png_size

STAGED = (
    "### Stage 1\nCount first.\n```python\nprint('one')\n```\n"
    "### Stage 2\nThen a figure.\n```python\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "plt.figure(figsize=(2, 2))\nplt.plot([0, 1], [0, 1])\n```\nThat is all."
)


def _responder(role, messages, schema):
    if role == "generator":
        return STAGED
    return default_responder(role, messages, schema)


def test_prose_and_code_events_carry_their_generation_order(registry):
    agent = make_agent(registry, responder=_responder)
    events = asyncio.run(collect(agent.run_turn("describe the data in stages")))
    assert events[-1].type == "turn_done" and events[-1].state == "done"
    prose = [e for e in events if e.type == "text" and e.role == "answer"]
    blocks = [e for e in events if e.type == "code_block_started"]
    assert [e.segment_index for e in prose] == [0, 2, 4] and [e.text.split("\n")[0] for e in prose] == ["### Stage 1", "### Stage 2", "That is all."]
    assert [e.segment_index for e in blocks] == [1, 3] and [e.block_id.rsplit("-", 2)[1] for e in blocks] == ["b1", "b2"]
    assert all(e.step_id == blocks[0].step_id for e in prose)  # the client groups by step before ordering by index
    assert all(e.segment_index is None for e in events if e.type == "text" and e.role != "answer")
    figs = [e for e in events if e.type == "figure"]
    assert len(figs) == 1 and figs[0].block_id == blocks[1].block_id and figs[0].width > 0 and figs[0].height > 0  # size read from the PNG


def test_png_size_reads_the_header_and_ignores_other_bytes():
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + (640).to_bytes(4, "big") + (480).to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"
    assert _png_size(header) == (640, 480)
    assert _png_size(b"not a png at all, just bytes that are long enough") == (None, None)
