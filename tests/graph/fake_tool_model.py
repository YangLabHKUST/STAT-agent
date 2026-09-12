"""A scripted tool-calling chat model for loop-mode tests (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ToolCallingStub(BaseChatModel):
    """``script(messages, bound_tools) -> AIMessage`` decides the next action."""

    script: Callable[[list, list[str]], AIMessage]
    bound: list[str] = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "tool-stub"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
        self.bound = names
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        msg = self.script(messages, self.bound)
        if not getattr(msg, "usage_metadata", None):
            msg.usage_metadata = {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{abs(hash(json.dumps(args, sort_keys=True))) % 10000}"}])


def last_tool_results(messages) -> list[tuple[str, str]]:
    out = []
    for m in messages:
        if isinstance(m, ToolMessage):
            out.append((m.name or "", m.content if isinstance(m.content, str) else str(m.content)))
    return out


def annotate_script(messages, bound):
    """inspect -> list -> check -> run -> finish, adapting to refusals."""
    results = last_tool_results(messages)
    names = [n for n, _ in results]
    if "inspect_session" not in names:
        return call("inspect_session")
    if "list_skills" not in names:
        return call("list_skills", target_slice_ids=[0])
    if "check_skill" in bound and "check_skill" not in names:
        return call("check_skill", slug="mini-annotate", target_slice_ids=[0], request="annotate cell types")
    if "run_code" not in names:
        return call("run_code", code="s = session.get_slice(0)\ns.adata.obs['celltype'] = 'T cell'\nprint('annotated')", skill_slug="mini-annotate")
    if "finish" not in names:
        return call("finish", summary="Annotated cell types on slice 0.", data_changes="obs:celltype added", key_findings=["all cells labelled T cell"])
    return AIMessage(content="Done.")


def skip_check_script(messages, bound):
    """Tries to run skill code without check_skill first (should be refused when guards are on)."""
    results = last_tool_results(messages)
    names = [n for n, _ in results]
    if "run_code" not in names:
        return call("run_code", code="print('hi')", skill_slug="mini-annotate")
    if any(n == "run_code" and c.startswith("Refused") for n, c in results) and "check_skill" not in names and "check_skill" in bound:
        return call("check_skill", slug="mini-annotate", target_slice_ids=[0], request="annotate")
    if "finish" not in names:
        return call("finish", summary="finished")
    return AIMessage(content="Done.")


def ask_script(messages, bound):
    results = last_tool_results(messages)
    names = [n for n, _ in results]
    if "ask_user" not in names:
        return call("ask_user", question="Which tissue is this?", kind="verifier")
    if "finish" not in names:
        answer = [c for n, c in results if n == "ask_user"][-1]
        return call("finish", summary=f"Tissue: {answer}")
    return AIMessage(content="Done.")


__all__ = ["ToolCallingStub", "call", "annotate_script", "skip_check_script", "ask_script", "last_tool_results"]
