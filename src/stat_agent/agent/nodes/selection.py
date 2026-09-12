"""filter / match / ask_skill / synthesize_card / verify / ask_verifier nodes."""

from __future__ import annotations

from langgraph.runtime import Runtime as LGRuntime
from langgraph.types import Command

from stat_agent.agent.nodes.common import (
    answer_records,
    cancel_turn,
    cancelled,
    emit_prompt,
    obtain_decision,
    phase,
)
from stat_agent.agent.policy import Interrupt
from stat_agent.agent.state import Context, TurnState, answers_for, current_step, memory_of, pending_interrupt
from stat_agent.events.catalogue import (
    AdviceEvent,
    FilterComplete,
    MatchComplete,
    MethodCardEvent,
    RefusalEvent,
    RejectedSkill,
    SkillSelected,
    UnguardedMode,
    VerdictView,
    VerifyComplete,
)
from stat_agent.memory.context import turn_steps_block
from stat_agent.roles.planner import PlanStep
from stat_agent.skills.filter import check_compatibility, filter_skills
from stat_agent.skills.schema import MethodCard, SkillFrontmatter

MAX_SELECTION_RETRIES = 2


def _contract(ctx: Context, current: dict) -> SkillFrontmatter | None:
    if current.get("skill"):
        return ctx.registry.get(current["skill"]).meta
    if current.get("contract"):
        return MethodCard.model_validate(current["contract"])
    return None


async def filter_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    phase(ctx, "filtering")
    current = dict(state["current"])
    summary = ctx.summary()
    contracts = {slug: meta.filter_requirements for slug, meta in ctx.registry.contracts().items()}
    res = filter_skills(current["target_slice_ids"], summary, contracts, bypass=not ctx.ablation.filter)
    ctx.emit(FilterComplete(step_id=current["step_id"], compatible=res.compatible, rejected=[RejectedSkill(slug=s, reason=r) for s, r in res.rejected], bypassed=res.bypassed))
    current["compatible"] = res.compatible
    current["would_reject"] = [list(x) for x in res.would_reject]
    return Command(update={"current": current}, goto="match")


async def match_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    if cancelled(ctx):
        return cancel_turn(ctx, state)
    phase(ctx, "matching")
    current = dict(state["current"])
    step = current_step(state) or {}
    if not ctx.ablation.matcher:
        current["matched"] = []
        ctx.emit(MatchComplete(step_id=current["step_id"], matched=[], reason="matcher disabled"))
        return _no_skill(ctx, current, step)
    catalog = ctx.registry.catalog(current.get("compatible", []))
    if not catalog:
        current["matched"] = []
        ctx.emit(MatchComplete(step_id=current["step_id"], matched=[], reason="no compatible skills"))
        return _no_skill(ctx, current, step)
    res = await ctx.roles.matcher.match(step.get("refined_query", state["query"]), catalog, top_k=ctx.settings.matcher_top_k, turn_id=state["turn_id"], step=current["step_number"])
    current["matched"] = res.selected
    ctx.emit(MatchComplete(step_id=current["step_id"], matched=res.selected))
    if len(res.selected) == 1:
        current["skill"] = res.selected[0]
        ctx.emit(SkillSelected(step_id=current["step_id"], skill=res.selected[0], name=ctx.registry.get(res.selected[0]).name))
        return Command(update={"current": current}, goto="verify")
    if len(res.selected) > 1:
        options = [{"index": i + 1, "label": ctx.registry.get(s).name, "slug": s, "description": ctx.registry.get(s).description} for i, s in enumerate(res.selected)]
        pending = Interrupt(kind="skill_selection", step=current["step_number"], question="Multiple methods match your request. Which one should I use?", options=options)
        emit_prompt(ctx, pending, current["step_id"])
        return Command(update={"current": current, "pending": pending.model_dump()}, goto="ask_skill")
    return _no_skill(ctx, current, step)


def _no_skill(ctx: Context, current: dict, step: dict) -> Command:
    """No registered skill: lightweight ``general`` steps run without a method card (the 0.1 general
    analysis mode); anything that names a method goes through the out-of-registry protocol."""
    if step.get("kind") == "general":
        current.update(mode="unguarded", contract=None)
        ctx.emit(UnguardedMode(step_id=current["step_id"], reason="general analysis: no method named, no skill required"))
        return Command(update={"current": current}, goto="verify")
    return Command(update={"current": current}, goto="synthesize_card")


async def ask_skill_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    current = dict(state["current"])
    pending = pending_interrupt(state)
    if pending is None:
        return Command(goto="verify")
    phase(ctx, "waiting", "skill selection")
    decision, assumptions = obtain_decision(state, ctx, pending, current["step_number"])
    recs = answer_records(pending, decision, current["step_number"], None)
    chosen = None
    text = (decision.as_text() or "").strip().lower()
    if decision.option_index is not None and 1 <= decision.option_index <= len(pending.options):
        chosen = pending.options[decision.option_index - 1]["slug"]
    elif text.isdigit() and 1 <= int(text) <= len(pending.options):
        chosen = pending.options[int(text) - 1]["slug"]
    else:
        for o in pending.options:
            if text and (text == o["slug"].lower() or text in o["label"].lower() or o["slug"].lower() in text):
                chosen = o["slug"]
                break
    if chosen is None:
        retries = state.get("retries", 0) + 1
        if retries > MAX_SELECTION_RETRIES:
            current.update(status="blocked", mode="advice", advice="No valid method was selected.")
            return Command(update={"current": current, "answers": recs, "assumptions": assumptions, "pending": None, "retries": 0}, goto="block_step")
        pending = Interrupt(kind="skill_selection", step=current["step_number"], question=f"'{decision.as_text()}' is not one of the options. Please reply with the option number.", options=pending.options)
        emit_prompt(ctx, pending, current["step_id"])
        return Command(update={"answers": recs, "assumptions": assumptions, "pending": pending.model_dump(), "retries": retries}, goto="ask_skill")
    current["skill"] = chosen
    ctx.emit(SkillSelected(step_id=current["step_id"], skill=chosen, name=ctx.registry.get(chosen).name, alternatives=[o["slug"] for o in pending.options if o["slug"] != chosen]))
    return Command(update={"current": current, "answers": recs, "assumptions": assumptions, "pending": None, "retries": 0}, goto="verify")


async def synthesize_card_node(state: TurnState, runtime: LGRuntime[Context]):
    """Out-of-registry protocol: contract first, then the same gates as a registered skill."""
    ctx = runtime.context
    if cancelled(ctx):
        return cancel_turn(ctx, state)
    current = dict(state["current"])
    step = current_step(state) or {}
    request = step.get("refined_query", state["query"])
    if not ctx.ablation.method_card:
        current.update(mode="unguarded", contract=None)
        ctx.emit(UnguardedMode(step_id=current["step_id"], reason="no registered skill matched; method-card synthesis disabled"))
        return Command(update={"current": current}, goto="generate")
    phase(ctx, "synthesizing", "no registered skill matched")
    summary = ctx.summary()
    card = await ctx.roles.method_card.synthesize(request, summary, ctx.registry.catalog(), turn_id=state["turn_id"], step=current["step_number"])
    ctx.emit(MethodCardEvent(step_id=current["step_id"], card=card.model_dump(mode="json")))
    if not card.named_method or card.confidence < 0.5:
        # a generic workflow, not a method: no contract, no gates beyond the filter; the 0.1 general path
        current.update(mode="unguarded", contract=None)
        ctx.emit(UnguardedMode(step_id=current["step_id"], reason=f"general analysis: the request names no specific method ('{card.method_name}'), no skill required"))
        return Command(update={"current": current}, goto="verify")
    targets = [summary.slice(i) for i in current["target_slice_ids"]]
    ok, reason = check_compatibility(card.filter_requirements, [t for t in targets if t is not None])
    if not ok:
        ctx.emit(RefusalEvent(step_id=current["step_id"], kind="out_of_registry", requirement=reason, reason=f"{card.method_name} is not applicable to the selected data: {reason}"))
        current.update(status="blocked", mode="advice", contract=card.model_dump(mode="json"), advice=f"'{card.method_name}' cannot run on the selected slice(s): {reason}.")
        return Command(update={"current": current}, goto="block_step")
    current.update(mode="unguarded", contract=card.model_dump(mode="json"))
    ctx.emit(UnguardedMode(step_id=current["step_id"], reason=f"unregistered method '{card.method_name}' will run with a synthesised contract; results are provisional"))
    return Command(update={"current": current}, goto="verify")


async def verify_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    if cancelled(ctx):
        return cancel_turn(ctx, state)
    phase(ctx, "verifying")
    current = dict(state["current"])
    step_dict = current_step(state) or {}
    step = PlanStep.model_validate(step_dict)
    contract = _contract(ctx, current)
    slug = current.get("skill") or (contract.slug if contract else None)
    if contract is None or not ctx.ablation.verifier:
        current["complete_query"] = step.refined_query
        ctx.emit(VerifyComplete(step_id=current["step_id"], skill=slug, verdicts=[], decided_by="bypass", complete_query=step.refined_query))
        return Command(update={"current": current}, goto="generate")
    summary = ctx.summary()
    stale = ctx.ledger.stale_ids() if ctx.ablation.provenance else []
    verdict = await ctx.roles.verifier.verify(
        step,
        contract,
        summary,
        answers=answers_for(state, "verifier", skill=slug, step=current["step_number"]),
        memory=memory_of(state, ctx) if ctx.ablation.memory else None,
        turn_steps=turn_steps_block(state.get("handoffs") or []) if ctx.ablation.memory else "",
        stale=stale,
        skill_name=ctx.registry.get(slug).name if current.get("skill") else getattr(contract, "method_name", None),
        turn_id=state["turn_id"],
        step_index=current["step_number"],
    )
    verdicts = [VerdictView(prerequisite=m, status="not_obtainable" if verdict.kind == "decline" else "ask_user", question=None) for m in verdict.missing] + ([VerdictView(prerequisite="(all)", status="satisfied")] if verdict.kind == "met" else [])
    ctx.emit(VerifyComplete(step_id=current["step_id"], skill=slug, verdicts=verdicts, decided_by=verdict.decided_by, complete_query=verdict.complete_query))
    if verdict.kind == "met":
        current["complete_query"] = verdict.complete_query or step.refined_query
        return Command(update={"current": current}, goto="generate")
    if verdict.kind == "ask":
        pending = Interrupt(kind="verifier", step=current["step_number"], questions=verdict.questions, skill=slug)
        emit_prompt(ctx, pending, current["step_id"])
        return Command(update={"current": current, "pending": pending.model_dump()}, goto="ask_verifier")
    if current.get("contract") and not current.get("skill"):
        ctx.emit(RefusalEvent(step_id=current["step_id"], kind="missing_prereq", requirement=verdict.missing[0] if verdict.missing else None, reason=verdict.advice or ""))
    else:
        ctx.emit(AdviceEvent(message=verdict.advice or "", skill=slug, step_id=current["step_id"], unmet=verdict.missing))
    current.update(status="blocked", mode="advice", advice=verdict.advice or "", unmet=verdict.missing)
    return Command(update={"current": current}, goto="block_step")


async def ask_verifier_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    current = dict(state["current"])
    pending = pending_interrupt(state)
    if pending is None:
        return Command(goto="verify")
    phase(ctx, "waiting", "prerequisites")
    decision, assumptions = obtain_decision(state, ctx, pending, current["step_number"])
    if decision.unresolvable:
        req = decision.unresolvable[0]
        ctx.emit(RefusalEvent(step_id=current["step_id"], kind="missing_prereq", requirement=req, reason=f"auto mode cannot supply: {req}"))
        current.update(status="blocked", mode="advice", advice=f"This analysis needs information that cannot be assumed automatically: {req}", unmet=decision.unresolvable)
        return Command(update={"current": current, "assumptions": assumptions, "pending": None}, goto="block_step")
    recs = answer_records(pending, decision, current["step_number"], pending.skill)
    return Command(update={"answers": recs, "assumptions": assumptions, "pending": None}, goto="verify")


__all__ = ["filter_node", "match_node", "ask_skill_node", "synthesize_card_node", "verify_node", "ask_verifier_node"]
