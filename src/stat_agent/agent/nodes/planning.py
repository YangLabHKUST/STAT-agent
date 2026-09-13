"""plan / ask_planner / select_step nodes."""

from __future__ import annotations

from langgraph.runtime import Runtime as LGRuntime
from langgraph.types import Command

from stat_agent.agent.nodes.common import (
    answer_records,
    cancel_turn,
    cancelled,
    emit_prompt,
    mark_step,
    obtain_decision,
    phase,
)
from stat_agent.agent.policy import Interrupt
from stat_agent.agent.state import Context, TurnState, answers_for, memory_of, pending_interrupt
from stat_agent.core.view import InterfaceState
from stat_agent.events.catalogue import AdviceEvent, PlanStepView, PlanUpdated, StepStarted, TextEvent
from stat_agent.memory.state import StepHandoff


def _plan_views(plan: list[dict], step_index: int, statuses: dict[int, str] | None = None) -> list[PlanStepView]:
    views = []
    for i, s in enumerate(plan):
        status = (statuses or {}).get(i) or ("active" if i == step_index else ("done" if i < step_index else "pending"))
        views.append(PlanStepView(step_id=f"s{i + 1}", index=i + 1, description=s.get("description", ""), status=status, target_slices=list(s.get("target_slice_ids", [])), depends_on=list(s.get("depends_on", [])), skill=s.get("skill")))
    return views


async def plan_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    phase(ctx, "planning")
    summary = ctx.summary()
    query = state["query"]
    interface = InterfaceState.model_validate(state["interface"]) if state.get("interface") else ctx.interface
    if not ctx.ablation.planner:
        result = ctx.roles.planner.single_step(query, summary)
    else:
        result = await ctx.roles.planner.plan(
            query,
            summary,
            memory=memory_of(state, ctx) if ctx.ablation.memory else None,
            answers=answers_for(state, "planner"),
            interface=interface,
            turn_id=state["turn_id"],
            catalog=ctx.registry.catalog(),
            state_block=ctx.ledger.state_text(list(summary.slice_ids)) if ctx.ablation.provenance else "",
        )
    if result.direct_answer:
        answer = result.direct_answer
        ctx.emit(TextEvent(segment_id=f"{state['turn_id']}-answer", text=answer, role="answer"))
        handoff = StepHandoff(step_number=1, description=query, mode="text", success=True, summary=answer[:300], response=answer)
        return Command(update={"plan": [], "step_index": 0, "pending": None, "retries": 0, "handoffs": [handoff.model_dump()]}, goto="finish_turn")
    if result.needs_clarification:
        pending = Interrupt(kind="planner", question=result.question)
        emit_prompt(ctx, pending, None)
        return Command(update={"pending": pending.model_dump()}, goto="ask_planner")
    plan = [s.model_dump() for s in result.steps]
    ctx.emit(PlanUpdated(plan_id=state["turn_id"], reason="created", steps=_plan_views(plan, 0)))
    return Command(update={"plan": plan, "step_index": 0, "pending": None, "retries": 0}, goto="select_step")


MAX_PLANNER_RETRIES = 2


def _block_turn_on_clarification(ctx: Context, state: TurnState, recs: list[dict], assumptions: list[dict], question: str, why: str) -> Command:
    advice = f"I could not settle what to analyse: {why}. Last question: {question} Please restate the request with the slice, the data and the analysis you want."
    ctx.emit(AdviceEvent(message=advice, skill=None, step_id=None, unmet=[question]))
    handoff = StepHandoff(step_number=1, description=state["query"], mode="advice", success=False, summary=advice[:300], response=advice, unresolved=[question])
    return Command(update={"answers": recs, "assumptions": assumptions, "pending": None, "retries": 0, "plan": [], "handoffs": [handoff.model_dump()], "status": "blocked"}, goto="finish_turn")


async def ask_planner_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    pending = pending_interrupt(state)
    if pending is None:
        return Command(goto="plan")
    phase(ctx, "waiting", "planner clarification")
    decision, assumptions = obtain_decision(state, ctx, pending, None)
    recs = answer_records(pending, decision, None, None)
    if decision.unresolvable:
        return _block_turn_on_clarification(ctx, state, recs, assumptions, pending.question or "", "the question cannot be answered automatically")
    retries = state.get("retries", 0) + 1
    if retries > MAX_PLANNER_RETRIES:
        return _block_turn_on_clarification(ctx, state, recs, assumptions, pending.question or "", f"the planner kept asking after {MAX_PLANNER_RETRIES} clarifications")
    return Command(update={"answers": recs, "assumptions": assumptions, "pending": None, "retries": retries}, goto="plan")


async def select_step_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    if cancelled(ctx):
        return cancel_turn(ctx, state)
    plan = state.get("plan") or []
    idx = state.get("step_index", 0)
    if idx >= len(plan):
        return Command(goto="finish_turn")
    step = plan[idx]
    step_no = idx + 1
    step_id = f"{state['turn_id']}-s{step_no}"
    mark_step(ctx, step_no)
    # a step whose dependency failed is skipped explicitly
    failed_steps = {h["step_number"] for h in state.get("handoffs", []) if not h.get("success", True)}
    blocked_by = [d for d in step.get("depends_on", []) if d in failed_steps]
    current = {
        "step_id": step_id,
        "step_number": step_no,
        "target_slice_ids": list(step.get("target_slice_ids", [])),
        "status": "running",
        "mode": "guided",
        "skill": None,
        "contract": None,
        "results": [],
        "figures": [],
        "produced": [],
        "reads": [],
        "text": "",
    }
    ctx.emit(StepStarted(step_id=step_id, index=step_no, total=len(plan), description=step.get("description", ""), target_slices=current["target_slice_ids"]))
    ctx.emit(PlanUpdated(plan_id=state["turn_id"], reason="step_started", steps=_plan_views(plan, idx, {int(k): v for k, v in (state.get("step_statuses") or {}).items()})))
    if blocked_by:
        current.update(status="skipped", mode="skipped", skip_reason=f"depends on failed step(s) {blocked_by}")
        return Command(update={"current": current}, goto="finish_step")
    return Command(update={"current": current, "retries": 0}, goto="filter")


__all__ = ["plan_node", "ask_planner_node", "select_step_node"]
