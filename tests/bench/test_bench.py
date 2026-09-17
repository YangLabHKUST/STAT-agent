from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from tests.fixtures.sessions import make_session
from tests.graph.conftest import SKILLS_MINI, default_responder

from stat_agent.agent import Agent
from stat_agent.llm import CallLogger, LLMClient, UsageMeter
from stat_bench.corrupt import corrupt_registry
from stat_bench.replay import compare, report
from stat_bench.result import RunResult
from stat_bench.runner import plan_cells, run
from stat_bench.score.sse import detect_sse, safe_handling_class
from stat_bench.session_bench import SessionScript, run_script
from stat_bench.suite import Query, Suite, convert_legacy_queries
from stat_bench.summarize import bootstrap_ci, summarize
from stat_bench.systems.stat import StatSystem

LEGACY = Path(__file__).resolve().parents[2] / "paper" / "benchmarking" / "comparison" / "queries.json"
SUITES = Path(__file__).resolve().parents[2] / "src" / "stat_bench" / "suites"


def _llm(cell=None):
    return LLMClient(None, calllog=CallLogger(), meter=UsageMeter(), model_name="scripted", responder=default_responder)


def test_legacy_conversion_and_suite_roundtrip(tmp_path):
    if not LEGACY.exists():
        pytest.skip("legacy queries.json not present")
    suite = convert_legacy_queries(LEGACY)
    assert len(suite.queries) == 40 and suite.get("Q01").category == "T1" and suite.get("Q40").expected.oor_class == "missing_prereq"
    q1 = suite.get("Q01")
    assert any(s.answer == "mouse" for s in q1.scripted)
    p = suite.save(tmp_path / "stat40.yaml")
    back = Suite.load(p)
    assert [q.id for q in back.queries] == [q.id for q in suite.queries]


def test_run_mini_suite_in_process(tmp_path, registry):
    suite = Suite(suite_id="mini", queries=[
        Query(id="M1", text="annotate cell types", category="T1", scripted=[{"kind": "verifier", "match": "tissue", "answer": "cortex"}]),
        Query(id="M2", text="run RNA velocity", category="out-of-scope", expected={"behaviour": "refuse", "oor_class": "data_impossible"}),
    ])
    cells = plan_cells(suite, system="stat", driver="staged", ablation="A0", policy="scripted", backbone="scripted", seeds=2, out=tmp_path / "res", data_root=None)
    assert len(cells) == 4 and cells[0].seed != cells[1].seed
    system = StatSystem(registry=registry, llm_factory=_llm, session_factory=lambda cell: make_session("S1_cell_gene_noCT"))
    results = run(system, cells, manifest_path=tmp_path / "res" / "manifest.json")
    by = {(r.query_id, r.seed_index): r for r in results}
    assert by[("M1", 0)].success and by[("M1", 0)].exit_status == "ok" and "s0:obs:celltype" in by[("M1", 0)].artifacts_produced
    assert by[("M1", 0)].llm_calls >= 4 and by[("M1", 0)].decisions["selected_skill"] == "mini-annotate"
    assert by[("M2", 0)].refusal["refused"] and by[("M2", 0)].exit_status == "refused" and safe_handling_class(by[("M2", 0)]) == "refusal_named"
    saved = RunResult.load(Path(cells[0].out_dir) / "result.json")
    assert saved.schema_version == 2 and saved.software["stat_version"]
    # resume skips completed cells
    again = run(system, cells)
    assert len(again) == 4
    s = summarize(results)
    g = s["groups"][0]
    assert g["n"] == 4 and 0.4 <= g["success_rate"] <= 0.6 and g["refusal_rate"] == 0.5


def test_bootstrap_ci_and_sse_detectors():
    m, lo, hi = bootstrap_ci([1, 1, 0, 1, 0, 1])
    assert lo <= m <= hi and abs(m - 4 / 6) < 1e-9
    r = RunResult(query_id="x", query_text="deconvolve", task="T6", code="import cell2location\nadata = sc.read_h5ad('/nope/ref.h5ad')", answer="We used RCTD and cell2location.", success=True)
    sse = detect_sse(r, make_session("S1_cell_gene"))
    assert sse["sse1_fabricated_input"]["flag"] and "/nope/ref.h5ad" in sse["sse1_fabricated_input"]["evidence"]
    assert sse["sse2_method_data_mismatch"]["flag"]  # cell2location on cell-level data
    assert sse["sse5_misattribution"]["flag"] and "rctd" in sse["sse5_misattribution"]["evidence"]


def test_corrupt_registry_stays_schema_valid(tmp_path):
    from stat_agent.skills import SkillRegistry

    manifest = corrupt_registry(SKILLS_MINI, tmp_path / "corrupt", severity=2, n=4, seed=1)
    assert manifest and all(m["before"] != m["after"] for m in manifest)
    reg = SkillRegistry([(tmp_path / "corrupt", "corrupt")])
    reg.load()
    assert not reg.broken and len(reg) == 4


def test_replay_compare_and_report():
    old = {"query": "q", "planner": {"needs_clarification": False, "n_steps": 1, "target_slices": [[0]]}, "matched": [["mini-annotate"]], "verdicts": ["met"], "clarifications": []}
    new_same = dict(old)
    new_diff = {**old, "matched": [["mini-niche"]]}
    assert compare(old, new_same)["agree"]
    c = compare(old, new_diff)
    assert not c["agree"] and c["fields"]["matcher_top1"] is False
    rep = report([old], [new_diff])
    assert rep["gate"] == "fail" and rep["per_field"]["matcher_top1"]["rate"] == 0.0
    assert report([old], [new_diff], {"q": "intended"})["gate"] == "pass"


@pytest.mark.parametrize("script_name", ["CR-1_cross_turn_reference.yaml", "CO-1_correction.yaml", "RE-1_recovery.yaml", "OR-1_out_of_registry.yaml"])
def test_session_bench_scripts(script_name, registry):
    script = SessionScript.load(SUITES / "sessionbench" / script_name)

    def responder(role, messages, schema):
        from stat_agent.llm.client import content_text

        text = "\n".join(content_text(m) for m in messages)
        if role == "generator" and "roi_label" in text:
            return "Labelling ROI members.\n```python\nroi = session.get_roi('tumor')\ns = session.get_slice(0)\ns.adata.obs['roi_label'] = 'out'\ns.adata.obs.loc[roi.adata.obs_names, 'roi_label'] = 'in_roi'\nprint('labelled')\n```"
        if role == "matcher" and "roi_label" in text.lower():
            return "[]"
        if role == "reflector" and "unfixable" in text:
            return json.dumps({"error_type": "KeyError", "root_cause": "boom", "fix_strategy": "none", "change_magnitude": "large", "fixed_code": "", "confidence": 0.1, "should_retry": False})
        if role == "method_card" and "roi_label" in text:
            return json.dumps({"method_name": "ROI labelling", "description": "label cells by ROI membership " * 2, "filter_requirements": {}, "prerequisites": [], "consumes": [], "produces": [{"kind": "obs", "key": "roi_label"}], "outline": ["label"], "risks": [], "confidence": 0.9})
        return default_responder(role, messages, schema)

    def factory(script, policy, ablation, sink):
        llm = LLMClient(None, calllog=CallLogger(), meter=UsageMeter(), model_name="scripted", responder=responder)
        return Agent(make_session(script.fixture), llm, registry=registry, mode="interactive", policy=policy, ablation=ablation, sinks=[sink], durable=False)

    result = asyncio.run(run_script(script, factory, mode="scripted"))
    failing = [(t["t"], v) for t in result["turns"] for v in t["verifiers"] if not v["pass"]] + [("final", v) for v in result["final"] if not v["pass"]]
    assert result["all_pass"], failing or [t for t in result["turns"] if t.get("expect_ok") is False]


def test_provider_error_classification():
    from stat_bench.systems.stat import _is_provider_error

    # langchain_openai wraps openai.APIConnectionError as OpenAIConnectionError("Connection error.")
    assert _is_provider_error("Connection error.", "OpenAIConnectionError")
    assert _is_provider_error("Connection error.", None)
    assert _is_provider_error("Error code: 503 - upstream unavailable", "InternalServerError")
    assert not _is_provider_error("KeyError: 'celltype'", "KeyError")

