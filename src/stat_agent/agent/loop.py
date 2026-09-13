"""Loop mode: one ReAct agent over the same skill registry, runtime and guards.

The agent decides everything in a single tool-calling loop (this is ablation arm A6 with
the guards off, and the "optional agentic loop" with them on). It is built on LangChain
1.x ``create_agent`` (which runs on LangGraph) with STAT's guards mounted as middleware;
``deepagents.create_deep_agent`` accepts the same tools and middleware and can be swapped
in through ``extras``.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command, interrupt

from stat_agent.agent.policy import Decision, Interrupt
from stat_agent.core.view import SessionSummary
from stat_agent.events.catalogue import (
    AssumptionEvent,
    CodeBlockDone,
    CodeBlockStarted,
    ErrorView,
    ExecOutput,
    FigureEvent,
    FilterComplete,
    OptionView,
    OutputLine,
    PhaseEvent,
    PromptAnswered,
    PromptEvent,
    QuestionView,
    RefusalEvent,
    RejectedSkill,
    TextEvent,
    VerifyComplete,
)
from stat_agent.ledger.provenance import StepRef
from stat_agent.roles.planner import PlanStep
from stat_agent.skills.filter import filter_skills

logger = logging.getLogger(__name__)

LOOP_SYSTEM_PROMPT = """You are STAT, an expert spatial transcriptomics analysis agent working inside a live Python session.

You have tools to inspect the loaded data, list and read analysis skills (each skill is a vetted workflow with the code to run), check whether a skill applies to the data and what it needs, run Python code against the live `session` object, ask the user a question, and finish with a summary.

Work like a careful analyst: inspect the data first, pick a skill when one fits the request, read it fully, check it before running, adapt its code to the data (explicit slice ids via `session.get_slice(i).adata`), run it, look at the output, and finish with a concise summary of what you did, what changed in the session, and the key findings. State assumptions explicitly. Do not claim results you did not compute."""


class GuardChain:
    """Deterministic guards mounted on the loop's tool calls (also used by the staged graph)."""

    def __init__(self, ablation, ledger) -> None:
        self.ablation = ablation
        self.ledger = ledger
        self.checked: dict[str, dict] = {}  # slug -> verdict for this turn

    def before_tool(self, name: str, args: dict, state: dict | None = None) -> str | None:
        if name == "run_code" and self.ablation.verifier:
            slug = args.get("skill_slug")
            if slug:
                v = self.checked.get(slug)
                if v is None:
                    return f"Refused: call check_skill('{slug}', target_slice_ids) before running skill code."
                if v.get("kind") == "decline":
                    return f"Refused: {slug} cannot run here - {v.get('advice')}"
        if name == "run_code" and self.ablation.provenance:
            from stat_agent.runtime.artifacts import code_access

            acc = code_access(args.get("code", ""))
            stale = set(self.ledger.stale_ids())
            hits = sorted(a for a in stale if any(a.endswith(":" + r) for r in acc.reads))
            if hits and not args.get("confirm_stale"):
                return f"Refused: the code reads stale artefacts {hits} (their inputs changed). Re-run the producing step first or pass confirm_stale=true."
        return None


class StatGuards(AgentMiddleware):
    def __init__(self, guards: GuardChain, emit: Callable, ablation) -> None:
        super().__init__()
        self.guards = guards
        self.emit = emit
        self.ablation = ablation

    def wrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        args = request.tool_call.get("args") or {}
        refusal = self.guards.before_tool(name, args)
        if refusal:
            self.emit(RefusalEvent(kind="other", reason=refusal))
            return ToolMessage(content=refusal, tool_call_id=request.tool_call["id"], name=name)
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        args = request.tool_call.get("args") or {}
        refusal = self.guards.before_tool(name, args)
        if refusal:
            self.emit(RefusalEvent(kind="other", reason=refusal))
            return ToolMessage(content=refusal, tool_call_id=request.tool_call["id"], name=name)
        return await handler(request)


def build_tools(agent, ctx, guards: GuardChain, turn_id: str, pending_box: dict) -> list:
    """The five tool-shaped operations every driver exposes (plus finish)."""
    session, registry, runtime, ledger, roles = agent.session, agent.registry, agent.runtime, agent.ledger, agent.roles
    ctx_emit, policy, ablation = ctx.emit, ctx.policy, ctx.ablation
    run_counter = {"n": 0}

    @tool
    def inspect_session() -> str:
        """Describe the loaded data: slices, modalities, data levels, columns, keys, ROIs, and what is stale."""
        summary = SessionSummary.from_session(session)
        text = summary.describe(verbose=True)
        if ablation.provenance:
            state = ledger.state_text()
            if state:
                text += "\n\nArtefacts (provenance):\n" + state
        return text

    @tool
    def list_skills(target_slice_ids: list[int] | None = None) -> str:
        """List registered analysis skills (slug and description). With target_slice_ids, only skills applicable to those slices are listed."""
        contracts = registry.contracts()
        if target_slice_ids and ablation.filter:
            summary = SessionSummary.from_session(session)
            res = filter_skills(target_slice_ids, summary, {s: m.filter_requirements for s, m in contracts.items()})
            ctx_emit(FilterComplete(compatible=res.compatible, rejected=[RejectedSkill(slug=s, reason=r) for s, r in res.rejected]))
            slugs = res.compatible
            tail = "\n\nNot applicable to the selected slices: " + "; ".join(f"{s} ({r})" for s, r in res.rejected) if res.rejected else ""
        else:
            slugs = sorted(contracts)
            tail = ""
        lines = [f"- {s}: {registry.get(s).description}" for s in slugs]
        return "\n".join(lines) + tail if lines else "No skills registered."

    @tool
    def read_skill(slug: str) -> str:
        """Read the full instructions of a skill (its workflow and code). Call this before running a skill."""
        if slug not in registry:
            return f"Unknown skill '{slug}'. Use list_skills."
        sk = registry.get(slug)
        return f"# {sk.name}\n\nPrerequisites: {'; '.join(sk.meta.prerequisites) or 'none'}\n\n{sk.prompt_instructions()}"

    @tool
    async def check_skill(slug: str, target_slice_ids: list[int], request: str = "") -> str:
        """Check that a skill applies to the target slices and that its prerequisites are met. Returns 'met', questions to ask the user, or advice on what must be done first."""
        if slug not in registry:
            return f"Unknown skill '{slug}'."
        summary = SessionSummary.from_session(session)
        sk = registry.get(slug)
        if ablation.filter:
            from stat_agent.skills.filter import check_compatibility

            targets = [summary.slice(i) for i in target_slice_ids if summary.slice(i)]
            ok, reason = check_compatibility(sk.meta.filter_requirements, targets)
            if not ok:
                guards.checked[slug] = {"kind": "decline", "advice": reason}
                return f"Not applicable: {reason}"
        if not ablation.verifier:
            guards.checked[slug] = {"kind": "met"}
            return "met (verifier disabled)"
        step = PlanStep(step_number=1, description=request or f"run {slug}", target_slice_ids=list(target_slice_ids), refined_query=request or f"run {slug}")
        verdict = await roles.verifier.verify(step, sk.meta, summary, stale=ledger.stale_ids() if ablation.provenance else [], skill_name=sk.name, turn_id=turn_id)
        ctx_emit(VerifyComplete(skill=slug, decided_by=verdict.decided_by, complete_query=verdict.complete_query))
        guards.checked[slug] = {"kind": verdict.kind, "advice": verdict.advice, "questions": verdict.questions}
        if verdict.kind == "met":
            return f"met. Suggested precise request: {verdict.complete_query}"
        if verdict.kind == "ask":
            return "Prerequisites missing; ask the user: " + " | ".join(verdict.questions)
        return f"Cannot run: {verdict.advice}"

    @tool
    async def run_code(code: str, skill_slug: str | None = None, confirm_stale: bool = False) -> str:
        """Run Python code in the live session (variables persist across calls). Pass skill_slug when the code follows a skill."""
        run_counter["n"] += 1
        if run_counter["n"] > getattr(agent.settings, "max_code_runs", 8):
            return "Refused: code-run budget exhausted for this turn. Finish with what you have."
        block_id = f"{turn_id}-loop-b{run_counter['n']}"
        ctx_emit(CodeBlockStarted(block_id=block_id, code=code, attempt=1))
        skill = registry.get(skill_slug) if skill_slug and skill_slug in registry else None
        res = await runtime.arun(code, on_output=lambda s, ln: ctx_emit(ExecOutput(block_id=block_id, lines=[OutputLine(stream=s, text=ln)])), skills=[skill] if skill else [], label=f"turn:{turn_id}")
        figs = []
        for png in res.figures:
            fid = agent.figure_store(png, f"{turn_id}-loop", block_id)
            figs.append(fid)
            ctx_emit(FigureEvent(figure_id=fid, block_id=block_id))
        if ablation.provenance and res.snapshot_before is not None and res.snapshot_after is not None:
            delta = ledger.record_step(res.snapshot_before, res.snapshot_after, producer=StepRef(turn_id=turn_id, step=run_counter["n"], skill=skill_slug, code_hash=res.code_hash, run_id=res.run_id, mode="loop"), reads=res.access.reads if res.access else (), roi_reads=res.access.roi_names if res.access else (), writes=res.access.slice_writes if res.access else (), target_slice_ids=res.access.slice_ids if res.access else ())
            if not delta.empty:
                ctx_emit(delta.to_event(origin="agent"))
        ctx_emit(CodeBlockDone(block_id=block_id, success=res.success, error=None if res.success else ErrorView(type=res.error_type or "Error", message=res.error or ""), duration_ms=int(res.duration_s * 1000), figures=figs))
        if agent.notebook is not None:
            try:
                agent.notebook.append_code_execution(code, res, user_query=f"[loop] {turn_id}", figures_png=res.figures)
            except Exception:  # noqa: BLE001
                pass
        out = res.display()
        if len(out) > 6000:
            out = out[:3000] + "\n... [output truncated] ...\n" + out[-2500:]
        return out

    @tool
    def ask_user(question: str, kind: str = "planner", options: list[str] | None = None) -> str:
        """Ask the user a question when the request is ambiguous or an input is missing. kind: planner (which data/what exactly), verifier (a missing prerequisite), skill_selection (choose among options)."""
        kind = kind if kind in ("planner", "verifier", "skill_selection") else "planner"
        prompt_id = "p_" + hashlib.blake2b(f"{turn_id}|{kind}|{question}".encode(), digest_size=4).hexdigest()
        opts = [{"index": i + 1, "label": o} for i, o in enumerate(options or [])]
        pend = Interrupt(prompt_id=prompt_id, kind=kind, question=question, questions=[question] if kind == "verifier" else [], options=opts)
        resuming = pending_box.get("resume_prompt_id") == prompt_id
        if not resuming:
            ctx_emit(PromptEvent(prompt_id=prompt_id, kind=kind, question=question, questions=[QuestionView(id="q1", text=question)] if kind == "verifier" else [], options=[OptionView(index=o["index"], label=o["label"]) for o in opts]))
        decision = None if resuming else policy.resolve(pend, ctx)
        if decision is None:
            pending_box["pending"] = pend
            value = interrupt(pend.model_dump())
            decision = Decision.model_validate(value) if isinstance(value, dict) else Decision(answer=str(value))
            ctx_emit(PromptAnswered(prompt_id=pend.prompt_id, decision=decision.model_dump(exclude_none=True), source="user"))
            return decision.as_text() or "(no answer)"
        aid = "a_" + uuid.uuid4().hex[:8]
        ctx_emit(AssumptionEvent(assumption_id=aid, kind=pend.kind, question=question, chosen=decision.as_text(), rationale=decision.rationale, source="scripted" if decision.source == "scripted" else "policy_default"))
        ctx_emit(PromptAnswered(prompt_id=pend.prompt_id, decision=decision.model_dump(exclude_none=True), source=decision.source))
        pending_box.setdefault("assumptions", []).append({"assumption_id": aid, "kind": pend.kind, "question": question, "answer": decision.as_text(), "source": decision.source, "rationale": decision.rationale})
        return decision.as_text() or "(no default available; proceed with your best judgement and state it)"

    @tool
    def finish(summary: str, data_changes: str = "", key_findings: list[str] | None = None) -> str:
        """Finish the turn with a summary for the user, the session changes made, and key findings."""
        pending_box["finish"] = {"summary": summary, "data_changes": data_changes, "key_findings": key_findings or []}
        return "finished"

    return [inspect_session, list_skills, read_skill, check_skill, run_code, ask_user, finish]


class LoopAgentRunner:
    """Runs one turn of the loop agent and translates it into STAT events."""

    def __init__(self, agent, model, *, max_iterations: int = 12) -> None:
        self.agent = agent
        self.model = model
        self.max_iterations = max_iterations
        self._graphs: dict[str, Any] = {}

    def _build(self, turn_id: str, ctx, pending_box: dict):
        guards = GuardChain(ctx.ablation, self.agent.ledger)
        tools = build_tools(self.agent, ctx, guards, turn_id, pending_box)
        if not ctx.ablation.filter and not ctx.ablation.verifier:
            tools = [t for t in tools if t.name != "check_skill"]
        middleware = [StatGuards(guards, ctx.emit, ctx.ablation)]
        return create_agent(self.model, tools=tools, system_prompt=LOOP_SYSTEM_PROMPT, middleware=middleware, checkpointer=self.agent._checkpointer)

    async def run(self, turn_id: str, message: str | None, ctx, resume: dict | None = None) -> dict:
        emit = ctx.emit
        pending_box: dict = {}
        if resume is not None:
            pending_box["resume_prompt_id"] = resume.get("prompt_id") if isinstance(resume, dict) else None
        graph = self._build(turn_id, ctx, pending_box)
        config = {"configurable": {"thread_id": f"{self.agent.session_id}:{turn_id}"}, "recursion_limit": self.max_iterations * 2 + 4}
        emit(PhaseEvent(phase="running", detail="agentic loop"))
        if resume is not None:
            result = await graph.ainvoke(Command(resume=resume), config)
        else:
            history = self.agent.memory.history_text() if self.agent.ablation.memory else ""
            content = message or ""
            if history:
                content = f"Earlier in this session:\n{history}\n\nNew request: {message}"
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


__all__ = ["LoopAgentRunner", "GuardChain", "StatGuards", "build_tools", "LOOP_SYSTEM_PROMPT"]
