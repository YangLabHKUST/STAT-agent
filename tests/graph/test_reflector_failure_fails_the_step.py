"""A reflector that gives no usable answer leaves the block failed; the turn finishes instead of dying."""

from __future__ import annotations

import asyncio

from tests.graph.conftest import collect, default_responder, make_agent


def _responder(role, messages, schema):
    if role == "generator":
        return "I'll try something risky.\n```python\nraise KeyError('boom')\n```"
    if role == "reflector":
        return "Let me analyse this. The rename(columns={'group': 'comparison_group'}) did nothing.\n\n```json\n{\"error_type\": \"KeyError\""  # cut off, nothing parseable
    return default_responder(role, messages, schema)


def test_an_unparseable_reflection_fails_the_block_not_the_turn(registry):
    agent = make_agent(registry, responder=_responder)
    events = asyncio.run(collect(agent.run_turn("crash please")))
    done = [e for e in events if e.type == "code_block_done"]
    assert len(done) == 1 and done[0].success is False and done[0].will_retry is True
    refl = [e for e in events if e.type == "reflection"]
    assert len(refl) == 1 and refl[0].will_retry is False and refl[0].summary.startswith("no fix could be obtained")
    assert not any(e.type == "turn_error" for e in events)
    assert events[-1].type == "turn_done" and events[-1].state == "blocked"
