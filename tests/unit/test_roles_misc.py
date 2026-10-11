"""Small role and runtime rules: fence parsing, analyser context, reply validation, ledger outcome, seeded names."""

from __future__ import annotations

import asyncio

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


def test_a_last_fence_left_open_still_runs():
    """DeepSeek ended a step with its last block and __INTERPRET__ but no closing fence (single-query Q35, Run 4):
    the block was shown as prose and never ran."""
    gen = parse_segments("Finally I'll map it.\n\n```python\nimport numpy as np\nprint(np.pi)\n\n__INTERPRET__: check the factors\n")
    assert [(s.kind, s.content) for s in gen.segments] == [("text", "Finally I'll map it."), ("code", "import numpy as np\nprint(np.pi)")]
    assert gen.interpret_instruction == "check the factors"
    kinds = lambda text: [s.kind for s in parse_segments(text).segments]  # noqa: E731
    assert kinds("```python\nx = 1\n```\nthen\n```python\ny = 2") == ["code", "text", "code"]
    assert kinds("Run:\n```bash\nls -la") == ["text"]  # an open shell fence stays prose


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
    text = ledger.state_text([0])
    assert "s0:obs:half_done" in delta.entry.stale_now and "half_done [int64] (STALE: written by a run that failed) <- mini-annotate, turn t1" in text
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


def test_unchanged_write_is_not_a_change_but_reproduces_a_stale_key():
    """A key the code assigned without changing values (a write to a copy, a library call with a custom
    key) is never an update and never invalidates dependants; it only clears a stale flag on that key."""
    from stat_agent.ledger.provenance import ProvenanceLedger, StepRef
    from stat_agent.runtime.snapshot import take_snapshot

    session = make_session("S1_cell_gene")
    ledger = ProvenanceLedger()
    ledger.initialize(take_snapshot(session))
    # a dependant of celltype
    before = take_snapshot(session)
    session.get_slice(0).adata.obs["niche"] = "n1"
    ledger.record_step(before, take_snapshot(session), producer=StepRef(turn_id="t1", step=1), reads=["obs:celltype"], writes=["obs:niche"], target_slice_ids=[0])
    assert ledger.get("s0:obs:niche").inputs == ["s0:obs:celltype"]
    version = ledger.version
    # code writes celltype on a copy: nothing on the slice changed
    before = take_snapshot(session)
    delta = ledger.record_step(before, take_snapshot(session), producer=StepRef(turn_id="t2", step=1), writes=["obs:celltype"], target_slice_ids=[0])
    assert delta.empty and ledger.version == version and not delta.entry.changed and not delta.entry.stale_now
    assert ledger.get("s0:obs:niche").status == "fresh" and ledger.get("s0:obs:celltype").produced_by is None
    # a stale key reproduced with identical values by a successful run is fresh again and attributed
    ledger.mark_stale("s0:obs:celltype", "test")
    assert ledger.get("s0:obs:niche").status == "stale"
    before = take_snapshot(session)
    delta = ledger.record_step(before, take_snapshot(session), producer=StepRef(turn_id="t3", step=1), writes=["obs:celltype"], target_slice_ids=[0])
    assert "s0:obs:celltype" in delta.entry.stale_cleared and ledger.get("s0:obs:celltype").status == "fresh"
    assert ledger.get("s0:obs:celltype").produced_by.turn_id == "t3" and ledger.get("s0:obs:niche").status == "stale"  # the dependant still needs its own re-run
    # a failed run does not reproduce anything
    ledger.mark_stale("s0:obs:celltype", "test")
    before = take_snapshot(session)
    ledger.record_step(before, take_snapshot(session), producer=StepRef(turn_id="t4", step=1), writes=["obs:celltype"], target_slice_ids=[0], outcome="failed")
    assert ledger.get("s0:obs:celltype").status == "stale"


def test_report_embeds_figures(tmp_path):
    from tests.graph.conftest import SKILLS_MINI, collect, make_agent

    from stat_agent.logging.report import render_report
    from stat_agent.skills import SkillRegistry

    reg = SkillRegistry([(SKILLS_MINI, "test")])
    reg.load()
    agent = make_agent(reg, kind="S1_cell_gene", mode="auto")
    asyncio.run(collect(agent.run_turn("describe the data")))
    agent.figures["t0001-s1-b1-a1-f1"] = b"\x89PNG fake"
    md = render_report(agent, agent.session, "md")
    assert "![figure t0001-s1-b1-a1-f1](figures/t0001-s1-b1-a1-f1.png)" in md
    html = render_report(agent, agent.session, "html")
    assert "data:image/png;base64," in html and "<figure>" in html


def test_event_log_queue_overflow_adds_one_resync_sentinel():
    import asyncio as aio

    from stat_agent.web.events import EventLog

    q: aio.Queue = aio.Queue(maxsize=2)
    for i in range(6):
        EventLog._put(q, (i, "exec_output", "{}"))
    items = list(q._queue)
    assert sum(1 for it in items if it[0] == "resync") == 1 and len(items) == 2


def test_seeded_names_survive_a_run(tmp_path):
    from stat_agent.runtime.executor import Runtime

    session = make_session("S1_cell_gene")
    rt = Runtime(session=session, snapshot=False, workdir=tmp_path)
    assert rt.run("session = 'oops'\nnp = 5\nkeep = 1").success
    assert rt.namespace["session"] is session and rt.namespace["np"].__name__ == "numpy" and rt.namespace["keep"] == 1


def test_auto_policy_answers_verifier_questions_sensibly():
    """Auto mode answers from the skill's defaults, the conventions, a fact named in the question, or a value
    the question itself offers; a fact keyed "slice" never answers a question that merely mentions a slice (B40)."""
    from stat_agent.agent.policy import AutoPolicy

    class _Slice:
        tissue_name = "breast cancer"

    class _Summary:
        slices = [_Slice()]
        slice_ids = [2]

    class _Ctx:
        skill_defaults: dict = {}

        def summary(self):
            return _Summary()

    policy = AutoPolicy()
    facts = {"slice": "Slice 2", "resolution": "0.8"}

    def ask(q, defaults=None):
        return policy._answer_question(q, _Ctx(), defaults or {}, facts)[0]

    assert ask("How many spatial domains should SpaGCN detect on slice 2? (e.g., 7)") == "7"
    assert ask("Please reply with a single integer: your target slice is already fixed as slice 2, I just need the domain count") is None
    assert ask("Which sequencing platform was used, 'Visium' or 'ST'?") == "Visium"
    assert ask("Which Leiden resolution should I use?") == "0.8"
    assert ask("What is the path to the reference h5ad file?") is None
    assert ask("What tissue type is this?") == "breast cancer"
    assert ask("Number of neighbours (default: 15)?") == "15"
    assert ask("How many clusters? I can default to 7 given the 12 annotated cell types") == "7"
    assert ask("Is the user's data or the sample's data normalised?") is None
    assert ask("How many domains?", defaults={"How many domains": "9"}) == "9"
    assert policy._answer_question("Which species is this?", _Ctx(), {}, {"species": "mouse"})[0] == "mouse"  # a stated fact beats the convention
    assert policy._answer_question("What is the path to the reference h5ad file?", _Ctx(), {}, {"reference h5ad": "/data/ref.h5ad"})[0] == "/data/ref.h5ad"  # a path the user gave is not invented


def test_llm_backend_shim_runs_on_the_client():
    """``stat_agent.agent.llm_backend.LLMBackend`` keeps the 0.1 surface skill code expects (B41)."""
    from stat_agent.agent.llm_backend import LLMBackend
    from stat_agent.llm import LLMClient

    seen: list = []

    def responder(role, messages, schema):
        seen.append((role, [(type(m).__name__, m.content) for m in messages]))
        return "Cluster 0: T cells"

    backend = LLMBackend(system_prompt="Respond with JSON.", client=LLMClient(None, model_name="scripted", responder=responder))
    assert asyncio.run(backend.run("name the clusters")) == "Cluster 0: T cells"
    assert seen[0][0] == "skill" and seen[0][1] == [("SystemMessage", "Respond with JSON."), ("HumanMessage", "name the clusters")]
    with pytest.raises(ValueError):
        LLMBackend()


def test_llm_backend_retries_provider_failures_only(monkeypatch):
    """Skill code's model call retries a network or provider failure; any other error reaches the skill at once (B63)."""
    from stat_agent.agent import llm_backend
    from stat_agent.llm import LLMClient

    monkeypatch.setattr(llm_backend, "RETRY_DELAYS_S", (0, 0))

    class APIConnectionError(Exception):
        pass

    calls: list = []

    def backend(fail_first: int, exc: Exception):
        def responder(role, messages, schema):
            calls.append(role)
            if len(calls) <= fail_first:
                raise exc
            return '{"0": "T cells"}'

        return llm_backend.LLMBackend(client=LLMClient(None, model_name="scripted", responder=responder))

    assert asyncio.run(backend(2, APIConnectionError("Connection error.")).run("name the clusters")) == '{"0": "T cells"}' and len(calls) == 3
    calls.clear()
    with pytest.raises(APIConnectionError):
        asyncio.run(backend(99, APIConnectionError("Connection error.")).run("name the clusters"))
    assert len(calls) == 3  # both retries used, then the failure reaches the skill
    calls.clear()
    with pytest.raises(ValueError):
        asyncio.run(backend(99, ValueError("unknown model")).run("name the clusters"))
    assert len(calls) == 1


def test_state_text_names_producers_and_viewer_edits():
    """The artefact block given to the roles names each artefact's producer (skill or code and the
    turn, or the viewer edit) and the recorded stale reason, so a direct answer about what is out of
    date or who produced a result has the record in front of it."""
    from tests.fixtures.sessions import make_session

    from stat_agent.ledger.provenance import ProvenanceLedger, StepRef
    from stat_agent.runtime import take_snapshot

    session = make_session("S1_cell_gene")
    ledger = ProvenanceLedger()
    ledger.initialize(take_snapshot(session))
    before = take_snapshot(session)
    a = session.get_slice(0).adata
    a.obs["score"] = a.obs["celltype"].cat.codes.astype(float)
    ledger.record_step(before, take_snapshot(session), producer=StepRef(turn_id="t3", step=2), reads=["obs:celltype"], writes=["obs:score"], target_slice_ids=[0])
    before = take_snapshot(session)
    a.obs["celltype"] = a.obs["celltype"].cat.rename_categories({"Tumor": "Cancer"})
    ledger.record_external(before, take_snapshot(session), label="celltype_rename")
    text = ledger.state_text([0])
    assert "celltype [labels: B cell, Cancer, Fibroblast, T cell] <- viewer celltype_rename" in text
    assert "score [float64] (STALE: input changed by celltype_rename (s0:obs:celltype)) <- code, turn t3" in text
