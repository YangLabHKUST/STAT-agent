"""LLM access: model factory, client with structured output, usage, logging, fixtures."""

from stat_agent.llm.calllog import CallLogger, CallRecord
from stat_agent.llm.client import Completion, LLMClient
from stat_agent.llm.fake import FakeMiss, FixtureStore, fixture_key
from stat_agent.llm.models import ModelSpec, make_model, make_spec, parse_model_id
from stat_agent.llm.structured import StructuredOutputError, extract_json, parse_json_to
from stat_agent.llm.usage import CallUsage, UsageMeter, UsageTotals

__all__ = [
    "LLMClient",
    "Completion",
    "CallLogger",
    "CallRecord",
    "FixtureStore",
    "FakeMiss",
    "fixture_key",
    "ModelSpec",
    "make_model",
    "make_spec",
    "parse_model_id",
    "StructuredOutputError",
    "extract_json",
    "parse_json_to",
    "CallUsage",
    "UsageMeter",
    "UsageTotals",
]
