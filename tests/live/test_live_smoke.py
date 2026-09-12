"""Opt-in smoke against real providers: STAT_LIVE=1 STAT_LIVE_MODELS="openai/gpt-4o,anthropic/claude-sonnet-4-6" pytest tests/live."""

from __future__ import annotations

import asyncio
import os

import pytest
from tests.fixtures.sessions import make_session

from stat_agent.agent import Agent
from stat_agent.core import SessionSummary
from stat_agent.llm import LLMClient, make_spec
from stat_agent.roles import PlannerRole

pytestmark = pytest.mark.live
MODELS = [m for m in os.environ.get("STAT_LIVE_MODELS", "").split(",") if m.strip()]


@pytest.mark.parametrize("model_id", MODELS or ["openai/gpt-4o"])
def test_planner_structured_output_and_usage(model_id):
    llm = LLMClient(spec=make_spec(model_id))
    session = make_session("S1_cell_gene_noCT")
    planner = PlannerRole(llm)
    result = asyncio.run(planner.plan("Annotate cell types", SessionSummary.from_session(session), turn_id="live"))
    assert not result.needs_clarification and result.steps[0].target_slice_ids == [0]
    assert llm.meter.total.input_tokens > 0


@pytest.mark.parametrize("model_id", MODELS or ["openai/gpt-4o"])
def test_one_turn_auto_mode(model_id, tmp_path):
    llm = LLMClient(spec=make_spec(model_id))
    agent = Agent(make_session("S1_cell_gene"), llm, mode="auto", session_dir=tmp_path)
    response = asyncio.run(agent.chat("How many cells of each cell type are there on slice 0?"))
    assert response and agent.last_status in ("done", "blocked")
    assert (tmp_path / "events.jsonl").exists()
