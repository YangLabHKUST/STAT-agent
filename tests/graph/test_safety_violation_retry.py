"""Code the safety check refuses is rewritten once through the reflector; the rewrite is checked again before it runs."""

from __future__ import annotations

import asyncio
import json

from tests.graph.conftest import collect, default_responder, make_agent

_FALLBACK = "Scoring.\n```python\nimport subprocess\nsubprocess.run(['pip', 'install', 'liana'])\nprint('never runs')\n```"


def _generator_writes_an_install_fallback(role, messages, schema):
    if role == "generator":
        return _FALLBACK
    return default_responder(role, messages, schema)


def _reflector_keeps_the_refused_call(role, messages, schema):
    if role == "generator":
        return _FALLBACK
    if role == "reflector":
        return json.dumps({"error_type": "SafetyViolation", "root_cause": "install fallback", "fix_strategy": "shell out differently", "change_magnitude": "small", "fixed_code": "import os\nos.system('pip install liana')", "confidence": 0.9, "should_retry": True})
    return default_responder(role, messages, schema)


def test_a_refused_block_is_reflected_and_the_rewrite_runs(registry):
    agent = make_agent(registry, responder=_generator_writes_an_install_fallback)
    events = asyncio.run(collect(agent.run_turn("describe the data")))
    done = [e for e in events if e.type == "code_block_done"]
    assert done[0].success is False and done[0].error.type == "SafetyViolation" and done[0].will_retry is True
    assert any(e.type == "reflection" and e.will_retry for e in events)
    assert done[-1].success is True and done[-1].attempt == 2
    assert not any("never runs" in (getattr(e, "text", "") or "") for e in events if e.type == "exec_output")  # the refused code did not execute
    assert events[-1].type == "turn_done" and events[-1].state == "done"


def test_a_rewrite_that_still_violates_is_refused_for_good(registry):
    agent = make_agent(registry, responder=_reflector_keeps_the_refused_call)
    events = asyncio.run(collect(agent.run_turn("describe the data")))
    done = [e for e in events if e.type == "code_block_done"]
    assert [d.error.type for d in done] == ["SafetyViolation"] * len(done) and not any(d.success for d in done)
    assert len(done) <= 3 and done[-1].will_retry is False  # the attempts are bounded, and the last refusal is final
    assert not any(e.type == "exec_output" for e in events)  # nothing refused was ever executed
    assert events[-1].type == "turn_done" and events[-1].state != "done"
