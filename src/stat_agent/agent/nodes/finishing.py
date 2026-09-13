"""block_step / finish_step / finish_turn nodes."""

from __future__ import annotations

from langgraph.runtime import Runtime as LGRuntime
from langgraph.types import Command

from stat_agent.agent.nodes.common import mark_step, phase
from stat_agent.agent.state import Context, TurnState, current_step, memory_of
from stat_agent.events.catalogue import PlanStepView, PlanUpdated, StepDone, TextEvent, TurnDone
from stat_agent.memory.state import AssumptionRecord, StepHandoff, TurnRecord


async def block_step_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    current = dict(state["current"])
    advice = current.get("advice") or "This step could not be executed."
    ctx.emit(TextEvent(segment_id=f"{current['step_id']}-advice", text=advice, role="note", step_id=current["step_id"]))
    current["text"] = advice
    current["status"] = "blocked"
    return Command(update={"current": current}, goto="finish_step")


async def finish_step_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    current = dict(state.get("current") or {})
    step = current_step(state) or {}
    idx = state.get("step_index", 0)
    plan = state.get("plan") or []
    analysis = current.get("analysis") or {}
    status = current.get("status", "running")
    success = bool(current.get("success", status not in ("blocked", "failed", "skipped", "cancelled")))
    if status == "running":
        status = "done" if success else "failed"
    response_parts = [current.get("text", "")]
    ran_ok = any(r.get("success") for r in current.get("results", []) or [])
    if analysis.get("interpretation"):
        response_parts.append("**Analysis:**\n" + analysis["interpretation"])
    elif ran_ok and analysis.get("response_summary"):
        # a plain count or lookup has no interpretation: the analyser's summary carries the answer
        response_parts.append("**Result:** " + analysis["response_summary"])
    if not success and analysis.get("execution_issues", {}).get("explanation"):
        response_parts.append("**Issue:** " + analysis["execution_issues"]["explanation"])
    response = "\n\n".join(p for p in response_parts if p).strip()
    handoff = StepHandoff(
        step_number=current.get("step_number", idx + 1),
        description=step.get("description", ""),
        skill=current.get("skill") or ((current.get("contract") or {}).get("method_name")),
        mode=status if status in ("skipped", "cancelled") else current.get("mode", "guided"),
        success=success and status == "done",
        inputs=list(current.get("reads", [])),
        outputs=list(current.get("produced", [])),
        summary=analysis.get("response_summary") or (current.get("text", "")[:300]),
        data_changes=analysis.get("data_changes", ""),
        key_findings=list(analysis.get("key_findings", []) or []),
        unresolved=list(current.get("unmet", []) or ([] if success else [analysis.get("execution_issues", {}).get("explanation") or ("stopped by the user" if status == "cancelled" else "step failed")])),
        parameters={"contract": current.get("contract")} if current.get("contract") else {},
        code_hash=None,
        code=[r.get("code", "") for r in current.get("results", []) if r.get("success")],
        variables=list(dict.fromkeys(v for r in current.get("results", []) if r.get("success") for v in r.get("variables", []))),
        complete_query=current.get("complete_query"),
        response=response,
    )
    shown = "done" if handoff.success else (status if status in ("skipped", "blocked", "cancelled") else "failed")
    ctx.emit(StepDone(step_id=current.get("step_id", f"s{idx + 1}"), index=idx + 1, status=shown, mode=handoff.mode, figures=list(current.get("figures", [])), summary=handoff.summary))
    statuses = {int(k): v for k, v in (state.get("step_statuses") or {}).items()}
    statuses[idx] = shown
    next_idx = idx + 1
    # stop the turn when a later step depends on this failed step; the rest is skipped
    later_dependants = [s for s in plan[next_idx:] if handoff.step_number in s.get("depends_on", [])]
    stop = (not handoff.success) and bool(later_dependants)
    if stop or status == "cancelled":
        for j in range(next_idx, len(plan)):
            statuses[j] = "skipped"
    ctx.emit(PlanUpdated(plan_id=state["turn_id"], reason="step_done", steps=[PlanStepView(step_id=f"s{i + 1}", index=i + 1, description=s.get("description", ""), status=statuses.get(i) or ("pending" if i >= next_idx else "done"), target_slices=list(s.get("target_slice_ids", [])), depends_on=list(s.get("depends_on", [])), skill=s.get("skill")) for i, s in enumerate(plan)]))
    figures = list(state.get("figures", [])) + list(current.get("figures", []))
    mark_step(ctx, None)
    update = {"handoffs": [handoff.model_dump()], "figures": figures, "step_index": next_idx, "step_statuses": {str(k): v for k, v in statuses.items()}}
    if status == "cancelled":
        return Command(update={**update, "status": "cancelled"}, goto="finish_turn")
    if stop:
        return Command(update={**update, "status": "blocked"}, goto="finish_turn")
    return Command(update=update, goto="select_step")


async def finish_turn_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    mark_step(ctx, None)
    handoffs = [StepHandoff.model_validate(h) for h in state.get("handoffs", [])]
    status = state.get("status")
    if status in (None, "running"):
        status = "done" if (not handoffs or all(h.success for h in handoffs)) else "blocked"
    response = "\n\n".join(h.response for h in handoffs if h.response).strip()
    memory = memory_of(state, ctx)
    record = TurnRecord(
        turn_id=state["turn_id"],
        index=state.get("turn_index", 0),
        query=state["query"],
        status=status,
        response_summary="; ".join(h.summary for h in handoffs if h.summary)[:600],
        data_changes="; ".join(h.data_changes for h in handoffs if h.data_changes)[:600],
        key_findings=[f for h in handoffs for f in h.key_findings][:6],
        handoffs=handoffs,
        assumptions=[AssumptionRecord.model_validate(a) for a in state.get("assumptions", [])],
        answers=[dict(a) for a in (state.get("answers") or [])],
        response=response,
    )
    if ctx.ablation.memory:
        memory.record_turn(record)
        if memory.needs_summary and status != "cancelled":  # the summary can wait for the next turn; the user is waiting now
            try:
                out = await ctx.roles.summarizer.summarize(memory.rolling_summary, memory.evicted_turns(), turn_id=state["turn_id"])
                memory.apply_summary(out.summary, len(memory.turns) - memory.window_turns)
                for f in out.facts:
                    if isinstance(f, dict) and f.get("key"):
                        memory.add_fact(str(f["key"]), str(f.get("value", "")), state["turn_id"], float(f.get("confidence", 0.8)))
            except Exception:  # noqa: BLE001 - summary is auxiliary
                pass
    phase(ctx, "idle")
    usage = ctx.llm.meter.turn_totals(state["turn_id"]).as_dict()
    ctx.emit(TurnDone(state="done" if status == "done" else ("cancelled" if status == "cancelled" else "blocked"), response=response, figures=list(state.get("figures", [])), usage=usage, ledger_version=ctx.ledger.version, summary={"steps": len(handoffs), "status": status}))
    return {"memory": memory.model_dump(), "status": status, "response": response}


__all__ = ["block_step_node", "finish_step_node", "finish_turn_node"]
