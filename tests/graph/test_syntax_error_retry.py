"""Code that does not parse is retried through the reflector like any other execution error."""

from __future__ import annotations

import asyncio

from tests.graph.conftest import collect, default_responder, make_agent


def _responder(role, messages, schema):
    if role == "generator":
        return "Counting.\n```python\nprint('unterminated\n```"
    return default_responder(role, messages, schema)


def test_a_syntax_error_in_generated_code_is_reflected_and_retried(registry):
    agent = make_agent(registry, responder=_responder)
    events = asyncio.run(collect(agent.run_turn("describe the data")))
    done = [e for e in events if e.type == "code_block_done"]
    assert done[0].success is False and done[0].error.type == "SyntaxError" and done[0].will_retry is True
    assert any(e.type == "reflection" and e.will_retry for e in events)
    assert done[-1].success is True and done[-1].attempt == 2
    assert events[-1].type == "turn_done" and events[-1].state == "done"
