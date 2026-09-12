"""Loop mode on ``deepagents`` (``loop`` extra): the same STAT tools and guards, plus subagents.

Differences from :class:`stat_agent.agent.loop.LoopAgentRunner` (plain ``create_agent``):

* ``write_todos`` planning middleware (removed when ``ablation.planner`` is off);
* two isolated subagents: ``method-researcher`` (derives the data requirements of an
  unregistered method, with web lookup when the ``research`` extra is configured) and
  ``skill-author`` (drafts a SKILL.md from executed code);
* an empty in-state filesystem backend, so the built-in file tools cannot leak skill bodies:
  the registry stays the only source of skill text (progressive disclosure).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from stat_agent.agent.loop import LOOP_SYSTEM_PROMPT, GuardChain, StatGuards, build_tools
from stat_agent.events.catalogue import PhaseEvent, TextEvent

logger = logging.getLogger(__name__)

RESEARCHER_PROMPT = """You are a methods researcher for spatial transcriptomics. Given a method name and a digest of the loaded data, produce a concise, factual method card: what the method computes, the exact input data it requires (data level cell/spot, modality, required AnnData keys such as layers/obs/obsm, external files such as references), its typical outputs (AnnData keys), the Python package(s) and their import names, known pitfalls, and whether the loaded data can satisfy the requirements. Say 'unknown' rather than guessing. Use the search/fetch tools when available; otherwise answer from knowledge and mark uncertainty."""

AUTHOR_PROMPT = """You are a skill author for STAT. Given executed Python code, the user's request, the observed outputs and (optionally) a README, write a SKILL.md with YAML frontmatter (name, slug, description, filter_requirements{modalities, data_levels, num_slices}, prerequisites, consumes, produces, dependencies{python}) followed by a body with Overview, Prerequisites, Workflow, a fenced python code block adapted to `session.get_slice(i).adata`, Outputs and Notes. Keep the code faithful to what actually ran."""


def build_subagents(agent) -> list[dict[str, Any]]:
    from stat_agent.extras.research import fetch_url_tool, web_search_tool

    research_tools = [t for t in (web_search_tool(), fetch_url_tool()) if t is not None]
    model = agent.loop_model if agent.loop_model is not None else agent.llm.model
    return [
        {"name": "method-researcher", "description": "Derive the data requirements, outputs and pitfalls of an analysis method that has no registered skill (call before writing code for an unregistered method).", "system_prompt": RESEARCHER_PROMPT, "tools": research_tools, "model": model},
        {"name": "skill-author", "description": "Draft a SKILL.md (frontmatter + body) from code that ran successfully, so the method can be registered for next time.", "system_prompt": AUTHOR_PROMPT, "tools": [], "model": model},
    ]


class DeepAgentRunner:
    """Same contract as LoopAgentRunner.run(); built with ``deepagents.create_deep_agent``."""

    def __init__(self, agent, model, *, max_iterations: int = 16, subagents: bool = True) -> None:
        self.agent = agent
        self.model = model
        self.max_iterations = max_iterations
        self.subagents = subagents

    def _build(self, turn_id: str, ctx, pending_box: dict):
        from deepagents import create_deep_agent
        from deepagents.backends import StateBackend

        guards = GuardChain(ctx.ablation, self.agent.ledger)
        tools = build_tools(self.agent, ctx, guards, turn_id, pending_box)
        if not ctx.ablation.filter and not ctx.ablation.verifier:
            tools = [t for t in tools if t.name != "check_skill"]
        prompt = LOOP_SYSTEM_PROMPT
        if self.subagents:
            prompt += "\n\nFor a method that has no registered skill, delegate to the `method-researcher` subagent first and follow its requirements; after an unregistered method ran successfully, you may delegate to `skill-author` and include its draft in your final summary."
        return create_deep_agent(
            self.model,
            tools=tools,
            system_prompt=prompt,
            middleware=[StatGuards(guards, ctx.emit, ctx.ablation)],
            subagents=build_subagents(self.agent) if self.subagents else None,
            backend=StateBackend(),
            checkpointer=self.agent._checkpointer,
        )

    async def run(self, turn_id: str, message: str | None, ctx, resume: dict | None = None) -> dict:
        emit = ctx.emit
        pending_box: dict = {}
        if resume is not None:
            pending_box["resume_prompt_id"] = resume.get("prompt_id") if isinstance(resume, dict) else None
        graph = self._build(turn_id, ctx, pending_box)
        config = {"configurable": {"thread_id": f"{self.agent.session_id}:{turn_id}"}, "recursion_limit": self.max_iterations * 2 + 8}
        emit(PhaseEvent(phase="running", detail="deep agent loop"))
        if resume is not None:
            result = await graph.ainvoke(Command(resume=resume), config)
        else:
            history = self.agent.memory.history_text() if ctx.ablation.memory else ""
            content = f"Earlier in this session:\n{history}\n\nNew request: {message}" if history else (message or "")
            result = await graph.ainvoke({"messages": [HumanMessage(content=content)]}, config)
        if result.get("__interrupt__"):
            return {"interrupted": True, "pending": pending_box.get("pending"), "assumptions": pending_box.get("assumptions", [])}
        final_text = ""
        for m in reversed(result.get("messages", [])):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                final_text = m.content if isinstance(m.content, str) else str(m.content)
                break
        fin = pending_box.get("finish") or {}
        response = fin.get("summary") or final_text
        if fin.get("key_findings"):
            response += "\n\n**Findings:** " + "; ".join(fin["key_findings"])
        emit(TextEvent(segment_id=f"{turn_id}-loop-final", text=response or "(no summary)", role="answer"))
        return {"interrupted": False, "response": response, "finish": fin, "assumptions": pending_box.get("assumptions", [])}


__all__ = ["DeepAgentRunner", "build_subagents"]
