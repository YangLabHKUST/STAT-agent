"""The consumed-column question end to end (B62): asked once, a named column travels to the generator,
no column means advice without code, and a same-key column needs no question."""

from __future__ import annotations

import asyncio

import pandas as pd
from tests.graph.conftest import _prompt, collect, default_responder, make_agent

from stat_agent.agent.policy import ScriptedPolicy, ScriptedRule

FALLBACK = "Please proceed with a reasonable default and state the assumption you made."
LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]


def _with_layers(agent):
    adata = agent.session.get_slice(0).adata
    adata.obs["layer"] = pd.Categorical([LAYERS[i % 7] for i in range(adata.n_obs)])
    return agent


def _recording():
    seen: dict[str, list[str]] = {"verifier": [], "generator": []}

    def responder(role, messages, schema):
        if role in seen:
            seen[role].append(_prompt(messages))
        return default_responder(role, messages, schema)

    return seen, responder


def test_no_column_named_ends_in_advice_without_code(registry):
    seen, responder = _recording()
    agent = _with_layers(make_agent(registry, responder=responder, mode="interactive", policy=ScriptedPolicy([], default=FALLBACK)))
    events = asyncio.run(collect(agent.run_turn("detect niches")))
    prompts = [e for e in events if e.type == "prompt"]
    assert len(prompts) == 1 and prompts[0].kind == "verifier" and "layer (7 labels" in prompts[0].questions[0].text
    advice = [e for e in events if e.type == "advice"]
    assert advice and "no existing column was named" in advice[-1].message
    assert not [e for e in events if e.type == "code_block_started"] and not seen["generator"] and not seen["verifier"]
    assert [e.decided_by for e in events if e.type == "verify_complete"] == ["contract", "contract"]


def test_a_named_column_is_read_in_place_of_the_key(registry):
    seen, responder = _recording()
    policy = ScriptedPolicy([ScriptedRule(kind="verifier", match=r"obs\['celltype'\]", answer="Use the layer column.")], default=FALLBACK)
    agent = _with_layers(make_agent(registry, responder=responder, mode="interactive", policy=policy))
    events = asyncio.run(collect(agent.run_turn("detect niches")))
    assert [e.status for e in events if e.type == "step_done"] == ["done"]
    assert "read obs['layer'] wherever the analysis reads obs['celltype'] (named by the user)" in seen["generator"][0]
    assert "Q: Which column should be read as obs['celltype'] on slice 0?" in seen["verifier"][0] and "A: layer (named by the user)" in seen["verifier"][0]
    assert "celltype" not in agent.session.get_slice(0).adata.obs.columns


def test_a_same_key_column_needs_no_question(registry):
    seen, responder = _recording()
    agent = make_agent(registry, responder=responder, mode="interactive", policy=ScriptedPolicy([], default=FALLBACK))
    adata = agent.session.get_slice(0).adata
    adata.obs["cell_type"] = pd.Categorical(["T", "B", "Tumor"] * (adata.n_obs // 3))
    events = asyncio.run(collect(agent.run_turn("detect niches")))
    assert not [e for e in events if e.type == "prompt"]
    assert "read obs['cell_type'] wherever the analysis reads obs['celltype'] (the same key spelled differently" in seen["generator"][0]


def test_auto_mode_blocks_instead_of_substituting(registry):
    seen, responder = _recording()
    agent = _with_layers(make_agent(registry, responder=responder, mode="auto"))
    events = asyncio.run(collect(agent.run_turn("detect niches")))
    assert not [e for e in events if e.type == "code_block_started"] and not seen["generator"]
    assert [e for e in events if e.type in ("advice", "refusal")]
    assert events[-1].type == "turn_done" and events[-1].state == "blocked"
