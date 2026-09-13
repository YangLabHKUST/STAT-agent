"""Small role and runtime rules: fence parsing, analyser context, reply validation, ledger outcome, seeded names."""

from __future__ import annotations

import pytest
from tests.fixtures.sessions import make_session

from stat_agent.roles.analyzer import BlockResult, assemble_context
from stat_agent.roles.generator import Segment, parse_segments


def test_untagged_python_fence_runs_and_other_fences_stay_prose():
    kinds = lambda text: [(s.kind, s.content) for s in parse_segments(text).segments]  # noqa: E731
    assert kinds("Look.\n```\nx = 1\nprint(x)\n```\n") == [("text", "Look."), ("code", "x = 1\nprint(x)")]
    assert kinds("```Python\ny = 2\n```")[0] == ("code", "y = 2")
    assert kinds("```python3\ny = 2\n```")[0] == ("code", "y = 2")
    assert [k for k, _ in kinds("Run this:\n```bash\nls -la\n```\nthen look")] == ["text"]
    assert [k for k, _ in kinds("```\nhello world, this is prose\n```")] == ["text"]
    assert [k for k, _ in kinds("```\nSTAT\n```")] == ["text"]  # a lone name is not a program
    assert [k for k, _ in kinds("```json\n{\"a\": 1}\n```")] == ["text"]


def test_assemble_context_keeps_head_and_tail_of_long_output():
    out = "\n".join(f"line {i}" for i in range(2000))
    ctx = assemble_context([Segment(kind="code", content="print(1)")], [BlockResult(code="print(1)", stdout=out)], max_output=300, tail=100)
    assert ctx.index("line 0") < ctx.index("(middle of the output omitted") < ctx.index("line 1999")
    short = assemble_context([Segment(kind="code", content="print(1)")], [BlockResult(code="print(1)", stdout="all of it")])
    assert "omitted" not in short and "all of it" in short


def test_reply_validation_rejects_optionless_and_empty_replies():
    from stat_agent.web.errors import ApiError
    from stat_agent.web.turns import Turn, TurnManager

    turn = Turn(turn_id="t1", generation=1, message="m", mode="interactive", driver="staged")
    turn.pending_prompt = {"prompt_id": "p", "kind": "skill_selection", "questions": [], "options": [{"index": 1, "slug": "a"}, {"index": 2, "slug": "b"}]}
    for body in ({"prompt_id": "p", "kind": "option"}, {"prompt_id": "p", "option_index": 0}, {"prompt_id": "p", "kind": "option", "index": "x"}):
        with pytest.raises(ApiError):
            TurnManager._to_decision(turn, body)
    assert TurnManager._to_decision(turn, {"prompt_id": "p", "kind": "option", "index": 2}).option_index == 2
    turn.pending_prompt = {"prompt_id": "p", "kind": "verifier", "questions": [{"id": "q1", "text": "Which tissue?"}], "options": []}
    for body in ({"prompt_id": "p", "answers": {}}, {"prompt_id": "p", "answers": {"q1": ""}}, {"prompt_id": "p", "kind": "prerequisites"}):
        with pytest.raises(ApiError):
            TurnManager._to_decision(turn, body)


def test_failed_run_marks_its_writes_stale():
    from stat_agent.ledger.provenance import ProvenanceLedger, StepRef
    from stat_agent.runtime.snapshot import take_snapshot

    session = make_session("S1_cell_gene")
    ledger = ProvenanceLedger()
    ledger.initialize(take_snapshot(session))
    before = take_snapshot(session)
    session.get_slice(0).adata.obs["half_done"] = 1
    delta = ledger.record_step(before, take_snapshot(session), producer=StepRef(turn_id="t1", step=1, skill="mini-annotate"), writes=["obs:half_done"], target_slice_ids=[0], outcome="failed")
    art = ledger.get("s0:obs:half_done")
    assert art is not None and art.status == "stale" and "failed" in (art.stale_reason or "")
    assert "s0:obs:half_done" in delta.entry.stale_now and "half_done (STALE)" in ledger.state_text([0])
    # a successful re-run of the same column clears it
    before = take_snapshot(session)
    session.get_slice(0).adata.obs["half_done"] = 2
    ledger.record_step(before, take_snapshot(session), producer=StepRef(turn_id="t2", step=1), writes=["obs:half_done"], target_slice_ids=[0])
    assert ledger.get("s0:obs:half_done").status == "fresh"


def test_extra_files_next_to_the_dataset_are_listed(tmp_path):
    from tests.fixtures.sessions import write_dataset

    from stat_agent.core.session import SimpleSession
    from stat_agent.core.view import SessionSummary
    from stat_agent.memory.context import generator_session_block, verifier_session_block

    src = make_session("S1_cell_gene")
    d = write_dataset(src, tmp_path / "ds", image=False)
    (d / "ref").mkdir()
    src.get_slice(0).adata.write_h5ad(d / "ref" / "reference.h5ad")
    session = SimpleSession(name="x")
    assert session.load_dataset(d) == [0]  # the reference is not loaded as a slice
    summary = SessionSummary.from_session(session)
    assert summary.extra_files == ["ref/reference.h5ad"] and summary.dataset_dir == str(d)
    assert f"{d}/ref/reference.h5ad" in verifier_session_block(summary, [0])
    assert "Other data files in the dataset directory" in generator_session_block(summary, [])
    assert "Other data files" not in verifier_session_block(SessionSummary.from_session(src), [0])


def test_matcher_prompt_declines_named_methods_the_skill_does_not_implement():
    from stat_agent.prompts import load_prompt

    p = load_prompt("matcher")
    assert p.version == 2 and "names a specific method, tool or package" in p.body


def test_seeded_names_survive_a_run(tmp_path):
    from stat_agent.runtime.executor import Runtime

    session = make_session("S1_cell_gene")
    rt = Runtime(session=session, snapshot=False, workdir=tmp_path)
    assert rt.run("session = 'oops'\nnp = 5\nkeep = 1").success
    assert rt.namespace["session"] is session and rt.namespace["np"].__name__ == "numpy" and rt.namespace["keep"] == 1
