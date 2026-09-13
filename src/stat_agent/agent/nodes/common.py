"""Shared helpers for nodes: emitting, interrupt handling, assumptions."""

from __future__ import annotations

import uuid

from langgraph.types import Command, interrupt

from stat_agent.agent.policy import Decision, Interrupt
from stat_agent.agent.state import Context, TurnState
from stat_agent.events.catalogue import (
    AssumptionEvent,
    OptionView,
    PhaseEvent,
    PromptAnswered,
    PromptEvent,
    QuestionView,
    TurnCancelled,
)


def phase(ctx: Context, name: str, detail: str | None = None) -> None:
    ctx.emit(PhaseEvent(phase=name, detail=detail))  # type: ignore[arg-type]


def emit_prompt(ctx: Context, pending: Interrupt, step_id: str | None) -> None:
    ctx.emit(
        PromptEvent(
            prompt_id=pending.prompt_id,
            kind=pending.kind,
            step_id=step_id,
            question=pending.question,
            questions=[QuestionView(id=f"q{i + 1}", text=q) for i, q in enumerate(pending.questions)],
            options=[OptionView(index=int(o.get("index", i + 1)), label=str(o.get("label", "")), slug=o.get("slug"), description=o.get("description")) for i, o in enumerate(pending.options)],
            skill=pending.skill,
            allow_free_text=pending.kind != "skill_selection",
        )
    )


def obtain_decision(state: TurnState, ctx: Context, pending: Interrupt, step_no: int | None) -> tuple[Decision, list[dict]]:
    """Ask the policy; otherwise interrupt the graph until a decision arrives.

    Returns the decision and the assumption records to append (empty for user answers).
    The node calling this must do nothing but this before it, so re-execution on resume is
    harmless (LangGraph re-runs the node from its start).
    """
    decision = ctx.policy.resolve(pending, ctx)
    assumptions: list[dict] = []
    if decision is None:
        value = interrupt(pending.model_dump())
        decision = Decision.model_validate(value) if isinstance(value, dict) else Decision(answer=str(value))
        decision.prompt_id = decision.prompt_id or pending.prompt_id
        decision.kind = pending.kind
        if decision.source == "auto":
            decision.source = "user"
        ctx.emit(PromptAnswered(prompt_id=pending.prompt_id, decision=decision.model_dump(exclude_none=True), source=decision.source))
        return decision, assumptions
    decision.kind = pending.kind
    if not decision.unresolvable:
        aid = "a_" + uuid.uuid4().hex[:8]
        question = pending.question or " | ".join(pending.questions) or "skill selection"
        rec = {
            "assumption_id": aid,
            "kind": pending.kind,
            "question": question,
            "answer": decision.as_text(),
            "source": "scripted" if decision.source == "scripted" else "policy_default",
            "rationale": decision.rationale,
            "step": step_no,
        }
        assumptions.append(rec)
        ctx.emit(AssumptionEvent(assumption_id=aid, kind=pending.kind, question=question, chosen=decision.as_text(), rationale=decision.rationale, source=rec["source"], step_id=f"{state.get('turn_id')}-s{step_no}" if step_no else None))
    ctx.emit(PromptAnswered(prompt_id=pending.prompt_id, decision=decision.model_dump(exclude_none=True), source=decision.source))
    return decision, assumptions


def answer_records(pending: Interrupt, decision: Decision, step_no: int | None, skill: str | None) -> list[dict]:
    """Turn a decision into Q&A records (one per question; a single free-text reply is
    attached to every question so nothing is lost)."""
    recs: list[dict] = []
    if pending.kind == "verifier":
        if decision.answers:
            for q in pending.questions:
                ans = decision.answers.get(q)
                if ans is None:  # tolerate 'q1'/'q2' ids
                    for k, v in decision.answers.items():
                        if k.lower().strip() in (q.lower().strip(),):
                            ans = v
                    if ans is None and len(decision.answers) == len(pending.questions):
                        ans = list(decision.answers.values())[pending.questions.index(q)]
                recs.append({"kind": "verifier", "question": q, "answer": ans or decision.as_text(), "step": step_no, "skill": skill})
        else:
            for q in pending.questions:
                recs.append({"kind": "verifier", "question": q, "answer": decision.as_text(), "step": step_no, "skill": skill})
    elif pending.kind == "planner":
        recs.append({"kind": "planner", "question": pending.question or "", "answer": decision.as_text(), "step": None, "skill": None})
    else:
        recs.append({"kind": "skill_selection", "question": pending.question or "", "answer": decision.as_text(), "step": step_no, "skill": None})
    return recs


def roi_names(ctx: Context) -> list[str]:
    return list(ctx.session.roi_subsets.keys())


def mark_step(ctx: Context, step_no: int | None) -> None:
    """Stamp later events with the plan step (None between steps); also called on resume."""
    if getattr(ctx, "emitter", None) is not None:
        ctx.emitter.step = step_no


def cancelled(ctx: Context) -> bool:
    return ctx.cancel.is_set()


def cancel_turn(ctx: Context, state: TurnState, *, partial: bool | None = None, update: dict | None = None) -> Command:
    """End the turn now: the user pressed stop. Checked at every stage boundary so at most one
    model call is in flight when the stop lands. A step that was under way is closed through
    ``finish_step`` so it keeps a handoff (its code may have changed the data) and the plan card
    shows it as cancelled; outside a step the turn ends directly."""
    ctx.emit(TurnCancelled(partial=bool(state.get("handoffs")) if partial is None else partial, ledger_version=ctx.ledger.version))
    current = dict((update or {}).get("current") or state.get("current") or {})
    active = bool(current) and current.get("step_number") == state.get("step_index", 0) + 1
    if active:
        current["status"] = "cancelled"
        return Command(update={**(update or {}), "current": current, "status": "cancelled"}, goto="finish_step")
    return Command(update={**(update or {}), "status": "cancelled"}, goto="finish_turn")


__all__ = ["phase", "emit_prompt", "obtain_decision", "answer_records", "roi_names", "cancelled", "cancel_turn"]
