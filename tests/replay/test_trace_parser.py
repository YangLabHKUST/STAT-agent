from __future__ import annotations

from pathlib import Path

import pytest

from stat_agent.llm import CallLogger, CallRecord
from stat_bench.trace import parse_session_dir, parse_turn_log, turn_decisions


def test_roundtrip_through_calllogger(tmp_path):
    log = CallLogger(tmp_path)
    log.start_turn("t1", "Annotate celltype on slice 0", turn_index=1)
    log.record(
        CallRecord(role="planner", model="stub", prompt_id="planner/v1", messages=[{"role": "user", "content": 'prompt with ```python\nprint(1)\n```\n**User Query:** "Annotate celltype on slice 0"\n  - Slice 0: mini (gene, cell-level, 600 observations, 40 genes)'}],
                   response_text='{"needs_clarification": false, "steps": [{"step_number": 1, "target_slice_ids": [0], "refined_query": "Annotate"}]}', turn_id="t1", input_tokens=5, output_tokens=2),
        turn_index=1,
    )
    log.log_user_response("t1", "verifier", "Which species?", "human", turn_index=1)
    log.record(CallRecord(role="matcher", model="stub", prompt_id="matcher/v1", messages=[{"role": "user", "content": 'User Request: "Annotate"\n- **celltype-annotation-fast**: d\n- **svg-spatialde**: e\nReturn at most 2 skill slugs'}], response_text='["celltype-annotation-fast"]', turn_id="t1"), turn_index=1)
    trace = parse_turn_log(tmp_path / "turn_001.md")
    assert trace.turn_index == 1 and trace.user_query == "Annotate celltype on slice 0"
    assert [c.role for c in trace.calls] == ["planner", "matcher"]
    assert trace.calls[0].prompt.startswith("prompt with ```python") and "print(1)" in trace.calls[0].prompt
    assert trace.calls[0].metadata["input_tokens"] == 5
    assert trace.user_responses[0].kind == "verifier" and trace.user_responses[0].answer == "human"
    dec = turn_decisions(trace)
    assert dec[0]["decision"]["target_slices"] == [[0]] and dec[0]["inputs"]["slices"][0]["slice_id"] == 0
    assert dec[1]["role"] == "user"
    assert dec[2]["decision"]["top1"] == "celltype-annotation-fast" and dec[2]["inputs"]["top_k"] == 2


REAL_LOGS = Path(__file__).resolve().parents[2] / "logs"


@pytest.mark.skipif(not REAL_LOGS.exists(), reason="historical logs not present")
def test_parse_historical_logs_smoke():
    dirs = sorted(p for p in REAL_LOGS.iterdir() if p.is_dir())[:5]
    n_calls = 0
    for d in dirs:
        for trace in parse_session_dir(d):
            n_calls += len(trace.calls)
            for c in trace.calls:
                assert c.prompt or c.response, f"{trace.path} call {c.index} empty"
            turn_decisions(trace)
    assert n_calls > 0
