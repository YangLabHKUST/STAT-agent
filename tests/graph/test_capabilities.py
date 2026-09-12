from __future__ import annotations

import asyncio
import json

from tests.graph.conftest import collect, default_responder, make_agent

from stat_agent.agent import Agent, Decision
from stat_agent.io.session_store import load_session, save_session
from stat_agent.llm import CallLogger, LLMClient, UsageMeter


def _llm(responder=default_responder):
    return LLMClient(None, calllog=CallLogger(), meter=UsageMeter(), model_name="scripted", responder=responder)


def test_rerun_plan_from_stale_artifacts(registry):
    agent = make_agent(registry, mode="auto")
    asyncio.run(collect(agent.run_turn("annotate cell types")))
    asyncio.run(collect(agent.run_turn("detect niches")))
    assert agent.ledger.has_live("s0:obs:niche_label")
    res = agent.mutator.set_column(0, "celltype", "Z")
    assert "s0:obs:niche_label" in res.delta.entry.stale_now
    queries = agent.rerun_plan()
    assert len(queries) == 1 and "niche" in queries[0].lower()
    assert "inputs changed" in agent.rerun_message()
    events = asyncio.run(collect(agent.run_turn(agent.rerun_message())))
    assert events[-1].type == "turn_done" and not agent.ledger.is_stale("s0:obs:niche_label")


def test_durable_checkpoint_resume_after_reload(registry, tmp_path):
    session_dir = tmp_path / "sess"
    agent = make_agent(registry, mode="interactive", session_dir=session_dir)
    asyncio.run(collect(agent.run_turn("annotate cell types")))
    assert agent.is_waiting and (session_dir / "checkpoints.sqlite").exists()
    agent.save_state()
    saved = save_session(agent.session, agent, tmp_path / "saved")
    asyncio.run(agent.aclose())
    # reload everything from disk into a fresh process-equivalent
    session, manifest = load_session(saved)
    assert manifest["slices"][0]["slice_id"] == 0
    agent2 = Agent.load(session, _llm(), session_dir, registry=registry)
    assert agent2.is_waiting and agent2.pending.kind == "verifier" and agent2.turn_index == 1
    events2 = asyncio.run(collect(agent2.resume(Decision(answers={agent2.pending.questions[0]: "cortex"}))))
    assert events2[-1].type == "turn_done" and events2[-1].state == "done"
    assert session.get_slice(0).has_celltype() and agent2.memory.turns[-1].turn_id == "t0001"
    asyncio.run(agent2.aclose())
    state = json.loads((session_dir / "agent_state.json").read_text())
    assert state["turn_index"] == 1


def test_draft_skill_from_unguarded_step(registry, tmp_path, monkeypatch):
    monkeypatch.setenv("STAT_HOME", str(tmp_path / "home"))

    def responder(role, messages, schema):
        if role == "skill_author":
            return (
                "---\nschema_version: 2\nname: rna-velocity-scvelo\ntitle: RNA velocity (scVelo)\nslug: rna-velocity-scvelo\n"
                "description: Estimate RNA velocity from spliced and unspliced counts with scVelo on cell-level data.\n"
                "version: \"1.0\"\nfilter_requirements:\n  modalities: [gene]\n  data_levels: [cell]\nprerequisites:\n  - \"Spliced and unspliced count layers\"\n"
                "consumes:\n  - layers:spliced\nproduces:\n  - obsm:velocity\npostconditions:\n  - artifact: obsm:velocity\n    check: exists\ndependencies:\n  python: [scvelo]\ndefault_skill: true\n---\n\n# RNA velocity\n\n```python\nimport scvelo as scv\ns = session.get_slice(0)  # <-- SET TARGET SLICE\ns.adata.obsm['velocity'] = 1\n```\n"
            )
        if role == "method_card":
            return json.dumps({"method_name": "scVelo RNA velocity", "description": "velocity" * 6, "filter_requirements": {"modalities": ["gene"], "data_levels": ["cell"]}, "prerequisites": [], "consumes": [], "produces": [{"kind": "obsm", "key": "velocity"}], "outline": ["a"], "risks": [], "confidence": 0.6})
        if role == "generator":
            return "Running velocity.\n```python\ns = session.get_slice(0)\ns.adata.obsm['velocity'] = np.zeros((s.n_obs, 2))\nprint('ok')\n```"
        return default_responder(role, messages, schema)

    from stat_agent.config import Settings

    agent = make_agent(registry, kind="S1_cell_gene", mode="auto", responder=responder, settings=Settings.from_env())
    events = asyncio.run(collect(agent.run_turn("run RNA velocity")))
    assert "unguarded_mode" in [e.type for e in events] and events[-1].state == "done"
    assert [e for e in events if e.type == "step_done"][0].mode == "unguarded"
    result = asyncio.run(agent.draft_skill("t0001", 1))
    assert not result["errors"] and result["slug"] == "rna-velocity-scvelo"
    assert (tmp_path / "home" / "skills" / "rna-velocity-scvelo" / "SKILL.md").exists()
    from stat_agent.skills import SkillRegistry

    reg = SkillRegistry.default(user_dir=tmp_path / "home" / "skills")
    assert "rna-velocity-scvelo" in reg and reg.get("rna-velocity-scvelo").origin == "user"
