from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from stat_agent.llm import (
    CallLogger,
    FakeMiss,
    FixtureStore,
    LLMClient,
    StructuredOutputError,
    UsageMeter,
    extract_json,
)
from stat_agent.llm.client import content_text
from stat_agent.llm.structured import parse_json_to


class Stub(BaseChatModel):
    """Returns canned replies in order and records the prompts it saw."""

    replies: list[str]
    seen: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(messages)
        text = self.replies.pop(0) if self.replies else "{}"
        msg = AIMessage(content=text, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return ChatResult(generations=[ChatGeneration(message=msg)])


class Plan(BaseModel):
    needs_clarification: bool
    steps: list[dict] = []


def _client(replies: list[str], **kw) -> LLMClient:
    stub = Stub(replies=list(replies), seen=[])
    return LLMClient(stub, calllog=CallLogger(), meter=UsageMeter(), model_name="stub", **kw)


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('text before {"a": {"b": [1,2]}} after') == '{"a": {"b": [1,2]}}'
    assert extract_json('here: ["x-y"]') == '["x-y"]'
    with pytest.raises(StructuredOutputError):
        extract_json("no json here")


def test_text_completion_records_usage(tmp_path):
    c = _client(["hello"])
    c.calllog.set_session_dir(tmp_path)
    comp = asyncio.run(c.complete([("system", "s"), ("user", "u")], role="generator", turn_id="t1", prompt_id="generator/v1"))
    assert comp.text == "hello" and comp.mode == "text" and comp.input_tokens == 10
    assert c.meter.total.llm_calls == 1 and c.meter.by_role["generator"].input_tokens == 10
    md = (tmp_path / "turn_t1.md").read_text()
    assert "## LLM Call 1: generator" in md and "### Full Prompt" in md and "### Response" in md
    assert list((tmp_path / "calls").glob("t1_01_generator.json"))


def test_structured_prompted_and_repair():
    c = _client(['Sure! {"needs_clarification": false, "steps": []}'])
    comp = asyncio.run(c.complete([("user", "u")], role="planner", schema=Plan))
    assert comp.value.needs_clarification is False and comp.mode == "prompted" and comp.repairs == 0
    c2 = _client(["not json at all", '{"needs_clarification": true}'])
    comp2 = asyncio.run(c2.complete([("user", "u")], role="planner", schema=Plan))
    assert comp2.value.needs_clarification is True and comp2.repairs == 1 and comp2.mode == "prompted+repair"
    assert comp2.input_tokens == 20
    c3 = _client(["garbage", "still garbage"])
    with pytest.raises(StructuredOutputError):
        asyncio.run(c3.complete([("user", "u")], role="planner", schema=Plan))
    assert c3.calllog.records[-1].error


def test_fixture_record_then_replay(tmp_path):
    store = FixtureStore(root=tmp_path / "llm", mode="record")
    c = _client(['{"needs_clarification": false}'], fixtures=store)
    inp = {"query": "annotate", "slices": [1, 0]}
    comp = asyncio.run(c.complete([("user", "u")], role="planner", schema=Plan, prompt_id="planner/v1", fake_input=inp))
    assert comp.mode == "prompted" and comp.fixture_key
    store2 = FixtureStore(root=tmp_path / "llm", mode="replay")
    c2 = LLMClient(None, fixtures=store2, calllog=CallLogger(), meter=UsageMeter(), model_name="none")
    inp_reordered = {"slices": [0, 1], "query": "annotate  "}
    comp2 = asyncio.run(c2.complete([("user", "different wording")], role="planner", schema=Plan, prompt_id="planner/v1", fake_input=inp_reordered))
    assert comp2.mode == "replay" and comp2.value.needs_clarification is False
    with pytest.raises(FakeMiss):
        asyncio.run(c2.complete([("user", "u")], role="planner", schema=Plan, prompt_id="planner/v2", fake_input=inp))


def test_content_text_handles_blocks():
    assert content_text(AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])) == "ab"
    assert content_text(AIMessage(content="plain")) == "plain"


def test_parse_json_recovers_prose_python_literals_and_unclosed_fences():
    """A reply that explains itself first, quotes a Python fragment, opens a fence it never closes, or
    writes the object as a Python literal still yields its JSON (the reflector reply of CRC-1 t4)."""
    unclosed = "Root cause: rename(columns={'group': 'comparison_group'}) did nothing.\n\n```json\n{\"needs_clarification\": true, \"steps\": []}\n1. [scanpy docs](https://scanpy.readthedocs.io)\n"
    assert parse_json_to(Plan, unclosed).needs_clarification is True
    assert parse_json_to(Plan, "{'needs_clarification': True, 'steps': []}").needs_clarification is True
    assert parse_json_to(Plan, 'e.g. {"x": 1}, then the answer {"needs_clarification": false, "steps": []}').needs_clarification is False
    with pytest.raises(StructuredOutputError):
        parse_json_to(Plan, "prose with {'group': 'x'} and nothing else")
    assert extract_json("see {'a': 1} then {\"b\": 2}") == "{'a': 1}"  # the first candidate, as before


def test_poe_models_stream_without_web_search():
    """Poe's Claude bots may reason for ten minutes before the first word of the reply, and the server cuts a reply
    that has sent nothing for about that long; streamed, the reasoning keeps it alive (B63)."""
    from stat_agent.llm import make_model

    model = make_model("poe/claude-sonnet-4.6", api_key="test-key")
    assert model.streaming and model.stream_usage and model.extra_body == {"web_search": False}
    assert model.stream_chunk_timeout == 600.0  # a pause between chunks is waited out as long as an unstreamed reply was
