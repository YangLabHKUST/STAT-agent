"""generate / execute (with bounded reflection and postcondition validation) / analyze nodes."""

from __future__ import annotations

import hashlib

from langgraph.runtime import Runtime as LGRuntime
from langgraph.types import Command

from stat_agent.agent.nodes.common import cancelled, phase, roi_names
from stat_agent.agent.state import Context, TurnState, current_step, memory_of
from stat_agent.events.catalogue import (
    CodeBlockDone,
    CodeBlockStarted,
    ErrorView,
    ExecOutput,
    ExecutionIssue,
    FigureEvent,
    OutputLine,
    ReflectionEvent,
    TextDelta,
    TextEvent,
    TurnCancelled,
    ValidateResult,
)
from stat_agent.ledger.provenance import StepRef
from stat_agent.roles.analyzer import BlockResult
from stat_agent.roles.generator import Generation, GeneratorRole
from stat_agent.roles.reflector import retry_allowed, should_attempt_fix
from stat_agent.skills.schema import MethodCard, SkillFrontmatter

UNGUARDED_NOTE = (
    "**NOTE: no registered skill covers this request. The contract below was synthesised for an unregistered "
    "method; follow it carefully, state every assumption in your explanation, and label the result as provisional.**"
)


def _card_guidance(card: MethodCard) -> str:
    lines = [f"**Method card: {card.method_name}** (provisional contract)", card.description, ""]
    if card.prerequisites:
        lines.append("Requirements: " + "; ".join(card.prerequisites))
    if card.outline:
        lines.append("Outline:")
        lines += [f"  {i + 1}. {s}" for i, s in enumerate(card.outline)]
    if card.produces:
        lines.append("Expected outputs: " + ", ".join(a.id for a in card.produces))
    if card.risks:
        lines.append("Caveats to state: " + "; ".join(card.risks))
    if card.research_notes:
        lines.append("Documentation notes: " + card.research_notes[:2000])
    return "\n".join(lines)


async def generate_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    phase(ctx, "generating")
    current = dict(state["current"])
    step = current_step(state) or {}
    plan = state.get("plan") or []
    summary = ctx.summary()
    skill = ctx.registry.get(current["skill"]) if current.get("skill") else None
    guidance = ""
    unguarded_note = ""
    if skill is not None:
        guidance = GeneratorRole.guidance(skill.name, skill.body)
    elif current.get("contract"):
        guidance = _card_guidance(MethodCard.model_validate(current["contract"]))
        unguarded_note = UNGUARDED_NOTE
    elif current.get("mode") == "unguarded":
        unguarded_note = "**NOTE: no registered skill covers this request; proceed with general analysis and state your assumptions.**"
    plan_tuple = (state["query"], current["step_number"], len(plan), step.get("description", "")) if len(plan) > 1 else None
    state_block = ctx.ledger.state_text(current["target_slice_ids"]) if ctx.ablation.provenance else ""
    query = current.get("complete_query") or step.get("refined_query") or state["query"]
    variables = GeneratorRole.variables(
        query,
        summary,
        memory_of(state, ctx) if ctx.ablation.memory else None,
        roi_names=roi_names(ctx),
        skill_guidance=guidance,
        state_block=state_block,
        plan=plan_tuple,
        unguarded_note=unguarded_note,
    )
    fake_input = {"query": query, "skill": current.get("skill"), "targets": current["target_slice_ids"], "step": current["step_number"]}
    gen: Generation | None = None
    if ctx.stream_text:
        seg_id = f"{current['step_id']}-stream"
        async for item in ctx.roles.generator.generate_streaming(variables, fake_input=fake_input, turn_id=state["turn_id"], step=current["step_number"]):
            if isinstance(item, str):
                ctx.emit(TextDelta(segment_id=seg_id, delta=item))
            else:
                gen = item
    else:
        gen = await ctx.roles.generator.generate(variables, fake_input=fake_input, turn_id=state["turn_id"], step=current["step_number"])
    assert gen is not None
    for i, seg in enumerate(gen.segments):
        if seg.kind == "text":
            ctx.emit(TextEvent(segment_id=f"{current['step_id']}-t{i}", text=seg.content, role="answer", step_id=current["step_id"]))
    current["segments"] = [s.model_dump() for s in gen.segments]
    current["interpret"] = gen.interpret_instruction
    current["text"] = gen.text
    current["query_used"] = query
    if not gen.has_code:
        current.update(mode=current.get("mode") if current.get("mode") != "guided" else "text", results=[])
        return Command(update={"current": current}, goto="finish_step")
    return Command(update={"current": current}, goto="execute")


def _contract_of(ctx: Context, current: dict) -> SkillFrontmatter | None:
    if current.get("skill"):
        return ctx.registry.get(current["skill"]).meta
    if current.get("contract"):
        return MethodCard.model_validate(current["contract"])
    return None


def _check_postconditions(ctx: Context, contract: SkillFrontmatter, target_ids: list[int], produced: list[str]) -> list[str]:
    """Evaluate declared postconditions (or existence of declared produces) on the live session."""
    failures: list[str] = []
    checks = list(contract.postconditions)
    if not checks:
        for art in contract.produces:
            if art.kind in ("obs", "obsm", "uns", "var", "layers"):
                checks.append(type(contract.postconditions[0])(artifact=art) if contract.postconditions else None)  # pragma: no cover
    from stat_agent.skills.schema import Postcondition

    if not contract.postconditions:
        checks = [Postcondition(artifact=a) for a in contract.produces if a.kind in ("obs", "obsm", "uns", "var", "layers")]
    for pc in checks:
        art = pc.artifact
        found = False
        for sid in target_ids:
            try:
                adata = ctx.session.get_slice(sid).adata
            except Exception:  # noqa: BLE001
                continue
            space = {"obs": adata.obs, "obsm": adata.obsm, "uns": adata.uns, "var": adata.var, "layers": adata.layers}.get(art.kind)
            if space is None:
                continue
            keys = list(space.columns) if hasattr(space, "columns") else list(space.keys())
            matches = [k for k in keys if art.matches(str(k))]
            if not matches:
                continue
            found = True
            if pc.check == "exists":
                continue
            key = matches[0]
            try:
                if pc.check == "non_null_fraction" and art.kind == "obs":
                    frac = float(adata.obs[key].notna().mean())
                    if frac < pc.params.get("min", 0.9):
                        failures.append(f"{art.id}: non-null fraction {frac:.2f} < {pc.params.get('min', 0.9)}")
                elif pc.check == "n_unique_min" and art.kind == "obs":
                    n = int(adata.obs[key].nunique())
                    if n < pc.params.get("n", 2):
                        failures.append(f"{art.id}: {n} unique values < {pc.params.get('n', 2)}")
                elif pc.check == "rows_match_obs" and art.kind in ("obsm", "layers"):
                    if space[key].shape[0] != adata.n_obs:
                        failures.append(f"{art.id}: {space[key].shape[0]} rows != n_obs {adata.n_obs}")
                elif pc.check == "rows_sum_to_one" and art.kind == "obsm":
                    import numpy as np

                    arr = np.asarray(space[key])
                    if not np.allclose(arr.sum(axis=1), 1.0, atol=pc.params.get("atol", 0.05)):
                        failures.append(f"{art.id}: rows do not sum to one")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{art.id}: postcondition check failed ({exc})")
        if not found:
            failures.append(f"{art.id} was not produced on slice(s) {target_ids}")
    return [f for f in failures if pc.severity == "error" or True]


async def execute_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    current = dict(state["current"])
    step_id = current["step_id"]
    step_no = current["step_number"]
    contract = _contract_of(ctx, current)
    skill = ctx.registry.get(current["skill"]) if current.get("skill") else None
    code_blocks = [s["content"] for s in current.get("segments", []) if s["kind"] == "code"]
    results: list[dict] = []
    produced: list[str] = list(current.get("produced", []))
    reads: set[str] = set(current.get("reads", []))
    figures: list[str] = list(current.get("figures", []))
    max_attempts = ctx.ablation.max_reflect_attempts if ctx.ablation.reflector else 0
    query = current.get("query_used") or state["query"]
    phase(ctx, "running")

    for bi, code in enumerate(code_blocks):
        if cancelled(ctx):
            ctx.emit(TurnCancelled(partial=True, ledger_version=ctx.ledger.version))
            current.update(status="cancelled", results=results)
            return Command(update={"current": current, "status": "cancelled"}, goto="finish_turn")
        attempt = 1
        while True:
            block_id = f"{step_id}-b{bi + 1}-a{attempt}"
            ctx.emit(CodeBlockStarted(block_id=block_id, step_id=step_id, code=code, attempt=attempt))

            def on_output(stream: str, line: str, _bid=block_id) -> None:
                ctx.emit(ExecOutput(block_id=_bid, lines=[OutputLine(stream=stream, text=line)]))

            res = await ctx.runtime.arun(
                code,
                on_output=on_output,
                skills=[skill] if skill else [],
                export_pythonpath=bool(skill and skill.meta.dependencies.subprocess_workers),
                label=f"turn:{state['turn_id']}",
            )
            block_figs: list[str] = []
            for png in res.figures:
                fid = ctx.figure_store(png, step_id, block_id) if ctx.figure_store else f"{block_id}-f{len(block_figs) + 1}"
                block_figs.append(fid)
                ctx.emit(FigureEvent(figure_id=fid, block_id=block_id, step_id=step_id, sha256=hashlib.sha256(png).hexdigest(), png_b64=None))
            figures += block_figs

            # provenance for every run, successful or not (a failing run may still have mutated state)
            if ctx.ablation.provenance and res.snapshot_before is not None and res.snapshot_after is not None:
                delta = ctx.ledger.record_step(
                    res.snapshot_before,
                    res.snapshot_after,
                    producer=StepRef(turn_id=state["turn_id"], step=step_no, skill=current.get("skill") or (contract.slug if contract else None), code_hash=res.code_hash, run_id=res.run_id, mode=current.get("mode", "guided")),
                    reads=res.access.reads if res.access else (),
                    roi_reads=res.access.roi_names if res.access else (),
                    declared_consumes=[a.id for a in contract.consumes] if contract else (),
                    writes=(set(res.access.writes) if res.access else set()) | ({a.id for a in contract.produces if res.success} if contract else set()),
                    target_slice_ids=current["target_slice_ids"],
                )
                if not delta.empty:
                    ctx.emit(delta.to_event(origin="agent"))
                    produced += [a for a in delta.entry.touched if a not in produced]
                if res.access:
                    reads |= {f"{r}" for r in res.access.reads}

            validation_failures: list[str] = []
            if res.success and contract is not None and ctx.ablation.validate_result and (contract.postconditions or contract.produces):
                validation_failures = _check_postconditions(ctx, contract, current["target_slice_ids"], produced)
                ctx.emit(ValidateResult(step_id=step_id, passed=not validation_failures, failures=validation_failures))

            error_text = res.error
            if res.success and validation_failures:
                error_text = "Postcondition check failed: " + "; ".join(validation_failures)
            success = res.success and not validation_failures
            can_reflect = (not success) and attempt <= max_attempts and should_attempt_fix(res.error_type if not res.success else "ValidationError") and not res.refused and not res.timed_out
            ctx.emit(
                CodeBlockDone(
                    block_id=block_id,
                    step_id=step_id,
                    success=success,
                    error=ErrorView(type=res.error_type or "ValidationError", message=error_text or "", traceback_tail=(res.traceback or "")[-1500:] or None) if not success else None,
                    duration_ms=int(res.duration_s * 1000),
                    figures=block_figs,
                    attempt=attempt,
                    will_retry=can_reflect,
                    timed_out=res.timed_out,
                )
            )
            if ctx.notebook is not None:
                try:
                    ctx.notebook.append_code_execution(code, res, user_query=query, figures_png=res.figures)
                except Exception:  # noqa: BLE001
                    pass
            if success:
                results.append(BlockResult(code=code, stdout=res.stdout, success=True).model_dump())
                break
            if not can_reflect:
                results.append(BlockResult(code=code, stdout=res.stdout, error=error_text, success=False).model_dump())
                break
            summary = ctx.summary()
            targets = [summary.slice(i) for i in current["target_slice_ids"]]
            reflection = await ctx.roles.reflector.reflect(
                user_query=query,
                code=code,
                error_msg=(error_text or "") + ("\n" + res.traceback if res.traceback else ""),
                stdout_tail=res.stdout_tail,
                available_rois={r.name: {"slice_id": r.slice_id, "modality": r.modality, "n_cells": r.n_obs} for r in summary.rois},
                available_columns=sorted({c for t in targets if t for c in t.obs_columns}),
                has_celltype=any(t.has_celltype for t in targets if t),
                attempt=attempt,
                turn_id=state["turn_id"],
                step=step_no,
            )
            will_retry = retry_allowed(reflection, attempt, max_attempts)
            ctx.emit(ReflectionEvent(block_id=block_id, attempt=attempt, max_attempts=max_attempts, summary=reflection.root_cause, change_magnitude=reflection.change_magnitude, confidence=reflection.confidence, will_retry=will_retry))
            if not will_retry:
                results.append(BlockResult(code=code, stdout=res.stdout, error=error_text, success=False).model_dump())
                break
            code = reflection.fixed_code or code
            attempt += 1
        if results and not results[-1]["success"]:
            break  # later blocks depend on the failed one

    current.update(results=results, produced=produced, reads=sorted(reads), figures=figures)
    current["success"] = bool(results) and all(r["success"] for r in results)
    return Command(update={"current": current}, goto="analyze")


async def analyze_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    phase(ctx, "analysing")
    current = dict(state["current"])
    from stat_agent.roles.generator import Segment

    segments = [Segment.model_validate(s) for s in current.get("segments", [])]
    results = [BlockResult.model_validate(r) for r in current.get("results", [])]
    produced = list(current.get("produced", []))
    stdout_tail = "\n".join(r.stdout[-1500:] for r in results)
    analysis = None
    if ctx.ablation.analyzer and results:
        try:
            analysis = await ctx.roles.analyzer.analyze(
                user_query=current.get("query_used") or state["query"],
                segments=segments,
                results=results,
                summary=ctx.summary(),
                roi_names=roi_names(ctx),
                memory=memory_of(state, ctx) if ctx.ablation.memory else None,
                interpret_instruction=current.get("interpret", "") or "",
                produced=produced,
                turn_id=state["turn_id"],
                step=current["step_number"],
            )
        except Exception as exc:  # noqa: BLE001 - auxiliary role: degrade explicitly
            analysis = ctx.roles.analyzer.deterministic(results, produced, stdout_tail)
            analysis.execution_issues.explanation = (analysis.execution_issues.explanation or "") + f" (analyzer failed: {exc})"
    if analysis is None:
        analysis = ctx.roles.analyzer.deterministic(results, produced, stdout_tail)
    if analysis.interpretation:
        ctx.emit(TextEvent(segment_id=f"{current['step_id']}-analysis", text=analysis.interpretation, role="analysis", step_id=current["step_id"]))
    if analysis.execution_issues.has_issues:
        ctx.emit(ExecutionIssue(step_id=current["step_id"], issue_type=analysis.execution_issues.issue_type, explanation=analysis.execution_issues.explanation or ""))
    current["analysis"] = analysis.model_dump()
    return Command(update={"current": current}, goto="finish_step")


__all__ = ["generate_node", "execute_node", "analyze_node"]
