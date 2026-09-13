"""generate / execute (with bounded reflection and one continuation for incomplete steps) / analyze nodes."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import threading

from langgraph.runtime import Runtime as LGRuntime
from langgraph.types import Command

from stat_agent.agent.nodes.common import cancel_turn, cancelled, phase, roi_names
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
    ValidateResult,
    WarningEvent,
)
from stat_agent.ledger.provenance import StepRef
from stat_agent.memory.context import namespace_block, turn_steps_block
from stat_agent.roles.analyzer import BlockResult
from stat_agent.roles.generator import Generation, GeneratorRole
from stat_agent.roles.reflector import retry_allowed, should_attempt_fix
from stat_agent.skills.schema import MethodCard, SkillFrontmatter

UNGUARDED_NOTE = (
    "**NOTE: no registered skill covers this request. The method card below is a provisional contract synthesised for an "
    "unregistered method: treat its requirements, outputs and outline as hints, use your own judgement for the implementation, "
    "state every assumption in your explanation, and label the result as provisional. If you implement a different method than "
    "the one the user named, say so in your first sentence and name what you use instead.**"
)
GENERAL_NOTE = (
    "**NOTE: no registered skill covers this request; proceed with general analysis and state your assumptions. If the request "
    "names a method or tool you are not using, say so in your first sentence and name what you use instead.**"
)


class _OutputBatcher:
    """Groups a block's stdout lines into one ``exec_output`` event per ~100 ms or 50 lines: one
    event per line let a verbose block overflow the subscriber queues and storm the page with resyncs."""

    def __init__(self, ctx: Context, block_id: str, max_lines: int = 50, interval: float = 0.1) -> None:
        self.ctx, self.block_id, self.max_lines, self.interval = ctx, block_id, max_lines, interval
        self._buf: list[OutputLine] = []
        self._lock = threading.Lock()

    def add(self, stream: str, line: str) -> None:  # called from the run thread
        with self._lock:
            self._buf.append(OutputLine(stream=stream, text=line))
            full = len(self._buf) >= self.max_lines
        if full:
            self.flush()

    def flush(self) -> None:
        with self._lock:  # emit under the lock so batches leave in order whichever thread flushes
            lines, self._buf = self._buf, []
            if lines:
                self.ctx.emit(ExecOutput(block_id=self.block_id, lines=lines))

    async def pump(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval)
                self.flush()
        finally:
            self.flush()


def _card_guidance(card: MethodCard) -> str:
    lines = [f"**Method card: {card.method_name}** (provisional contract; hints, not instructions)", card.description, ""]
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
    if cancelled(ctx):
        return cancel_turn(ctx, state)
    phase(ctx, "generating")
    current = dict(state["current"])
    step = current_step(state) or {}
    plan = state.get("plan") or []
    summary = ctx.summary()
    skill = ctx.registry.get(current["skill"]) if current.get("skill") else None
    guidance = ""
    unguarded_note = ""
    if skill is not None:
        guidance = GeneratorRole.guidance(skill.name, skill.body, [a.id for a in skill.meta.produces])
    elif current.get("contract"):
        # the card is hints, not a skill body: it is rendered with the note, outside the skill banner
        unguarded_note = UNGUARDED_NOTE + "\n\n" + _card_guidance(MethodCard.model_validate(current["contract"]))
    elif current.get("mode") == "unguarded":
        unguarded_note = GENERAL_NOTE
    plan_tuple = (state["query"], current["step_number"], len(plan), step.get("description", "")) if len(plan) > 1 else None
    state_block = ctx.ledger.state_text(current["target_slice_ids"]) if ctx.ablation.provenance else ""
    query = current.get("complete_query") or step.get("refined_query") or state["query"]
    continuation_note = ""
    if current.get("continuation"):
        continuation_note = (
            "**Continuation of this step.** The code you wrote earlier in this step ran successfully and its variables are still defined"
            " (listed under Python namespace above). The step is not finished: its declared outputs are still missing: "
            + "; ".join(current["continuation"]["missing"])
            + ". Continue from where that code stopped and produce them in this response. Do not repeat work that already ran."
        )
    # what the earlier steps of this turn did and what the live namespace holds (memory-off arm: neither)
    turn_steps = turn_steps_block(state.get("handoffs") or []) if ctx.ablation.memory else ""
    namespace_text = namespace_block(ctx.runtime.describe_namespace()) if ctx.ablation.memory else ""
    variables = GeneratorRole.variables(
        query,
        summary,
        memory_of(state, ctx) if ctx.ablation.memory else None,
        roi_names=roi_names(ctx),
        skill_guidance=guidance,
        state_block=state_block,
        plan=plan_tuple,
        unguarded_note=unguarded_note,
        continuation=continuation_note,
        turn_steps=turn_steps,
        namespace=namespace_text,
    )
    fake_input = {"query": query, "skill": current.get("skill"), "targets": current["target_slice_ids"], "step": current["step_number"]}
    gen: Generation | None = None
    if ctx.stream_text:
        seg_id = f"{current['step_id']}-stream"
        stream = ctx.roles.generator.generate_streaming(variables, fake_input=fake_input, turn_id=state["turn_id"], step=current["step_number"])
        async for item in stream:
            if isinstance(item, str):
                ctx.emit(TextDelta(segment_id=seg_id, delta=item))
                if cancelled(ctx):
                    await stream.aclose()  # drop the in-flight generation
                    break
            else:
                gen = item
    else:
        gen = await ctx.roles.generator.generate(variables, fake_input=fake_input, turn_id=state["turn_id"], step=current["step_number"])
    if cancelled(ctx) or gen is None:
        return cancel_turn(ctx, state)
    for i, seg in enumerate(gen.segments):
        if seg.kind == "text":
            ctx.emit(TextEvent(segment_id=f"{current['step_id']}-t{i}", text=seg.content, role="answer", step_id=current["step_id"]))
    new_segments = [s.model_dump() for s in gen.segments]
    current["segments"] = (list(current.get("segments", [])) + new_segments) if current.get("continuation") else new_segments
    current["interpret"] = gen.interpret_instruction or (current.get("interpret", "") if current.get("continuation") else "")
    current["text"] = (current.get("text", "") + "\n\n" + gen.text).strip() if current.get("continuation") else gen.text
    current["query_used"] = query
    if not gen.has_code:
        if current.get("continuation"):
            # the model explained instead of continuing: the step is reported as incomplete
            current["incomplete"] = list(current.pop("continuation")["missing"])
            current["success"] = False
            return Command(update={"current": current}, goto="analyze")
        current.update(mode=current.get("mode") if current.get("mode") != "guided" else "text", results=[])
        return Command(update={"current": current}, goto="finish_step")
    return Command(update={"current": current}, goto="execute")


def _contract_of(ctx: Context, current: dict) -> SkillFrontmatter | None:
    if current.get("skill"):
        return ctx.registry.get(current["skill"]).meta
    if current.get("contract"):
        return MethodCard.model_validate(current["contract"])
    return None


def _code_writes(res) -> set[str]:
    """Unqualified ``kind:key`` refs the executed code wrote: assignments seen by the AST scan plus
    the outputs of known library calls (``sc.tl.leiden`` -> ``obs:leiden``)."""
    if not res.access:
        return set()
    from stat_agent.skills.lint import LIBRARY_EFFECTS

    out = set(res.access.writes)
    for call in res.access.calls:
        for eff in LIBRARY_EFFECTS.get(call, []):
            if "*" not in eff:
                out.add(eff)
    return out


def _missing_outputs(ctx: Context, contract: SkillFrontmatter, target_ids: list[int], produced: list[str], require_produced: bool = True) -> list[str]:
    """Declared outputs this step did not deliver: absent on every target slice or, with provenance
    on, present only as a leftover from an earlier step (which is what catches a step that only
    inspected the data)."""
    missing: list[str] = []
    for art in contract.produces:
        if art.kind not in ("obs", "obsm", "uns", "var", "layers"):
            continue
        delivered = leftover = False
        for sid in target_ids:
            try:
                adata = ctx.session.get_slice(sid).adata
            except Exception:  # noqa: BLE001
                continue
            space = {"obs": adata.obs, "obsm": adata.obsm, "uns": adata.uns, "var": adata.var, "layers": adata.layers}[art.kind]
            keys = list(space.columns) if hasattr(space, "columns") else list(space.keys())
            matches = [k for k in keys if art.matches(str(k))]
            if not matches:
                continue
            if not require_produced or any(f"s{sid}:{art.kind}:{k}" in produced for k in matches):
                delivered = True
                break
            leftover = True
        if not delivered:
            missing.append(f"{art.id} was not produced on slice(s) {target_ids}" + (" (it exists from an earlier step, but this step did not write it)" if leftover else ""))
    return missing


async def execute_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    current = dict(state["current"])
    step_id = current["step_id"]
    step_no = current["step_number"]
    contract = _contract_of(ctx, current)
    skill = ctx.registry.get(current["skill"]) if current.get("skill") else None
    continuing = bool(current.get("continuation"))
    results: list[dict] = list(current.get("results", [])) if continuing else []
    code_blocks = [s["content"] for s in current.get("segments", []) if s["kind"] == "code"]
    if continuing:
        code_blocks = code_blocks[len(results):]  # segments accumulate across a continuation: run only the new blocks
    wrote_anything = False
    produced: list[str] = list(current.get("produced", []))
    reads: set[str] = set(current.get("reads", []))
    figures: list[str] = list(current.get("figures", []))
    code_writes: set[str] = set(current.get("code_writes") or []) if continuing else set()  # unqualified kind:key refs the code wrote, on any object
    replaced_loaded: list[str] = list(current.get("replaced_loaded") or []) if continuing else []
    max_attempts = ctx.ablation.max_reflect_attempts if ctx.ablation.reflector else 0
    query = current.get("query_used") or state["query"]
    phase(ctx, "running")

    for bi, code in enumerate(code_blocks):
        if cancelled(ctx):
            current.update(status="cancelled", results=results, produced=produced, reads=sorted(reads), figures=figures)
            return cancel_turn(ctx, state, partial=True, update={"current": current})
        attempt = 1
        while True:
            block_id = f"{step_id}-b{bi + 1}-a{attempt}"
            ctx.emit(CodeBlockStarted(block_id=block_id, step_id=step_id, code=code, attempt=attempt))

            batcher = _OutputBatcher(ctx, block_id)
            pump = asyncio.create_task(batcher.pump())
            try:
                res = await ctx.runtime.arun(
                    code,
                    on_output=batcher.add,
                    skills=[skill] if skill else [],
                    export_pythonpath=bool(skill and skill.meta.dependencies.subprocess_workers),
                    label=f"turn:{state['turn_id']}",
                )
            finally:
                pump.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pump
            block_figs: list[str] = []
            for png in res.figures:
                fid = ctx.figure_store(png, step_id, block_id) if ctx.figure_store else f"{block_id}-f{len(block_figs) + 1}"
                block_figs.append(fid)
                ctx.emit(FigureEvent(figure_id=fid, block_id=block_id, step_id=step_id, sha256=hashlib.sha256(png).hexdigest(), png_b64=None))
            figures += block_figs

            step_writes = _code_writes(res)
            code_writes |= step_writes
            # provenance for every run, successful or not (a failing run may still have mutated state)
            if ctx.ablation.provenance and res.snapshot_before is not None and res.snapshot_after is not None:
                delta = ctx.ledger.record_step(
                    res.snapshot_before,
                    res.snapshot_after,
                    producer=StepRef(turn_id=state["turn_id"], step=step_no, skill=current.get("skill") or (contract.slug if contract else None), code_hash=res.code_hash, run_id=res.run_id, mode=current.get("mode", "guided")),
                    reads=res.access.reads if res.access else (),
                    roi_reads=res.access.roi_names if res.access else (),
                    declared_consumes=[a.id for a in contract.consumes] if contract else (),
                    writes=step_writes,
                    target_slice_ids=current["target_slice_ids"],
                    outcome="ok" if res.success else ("cancelled" if res.cancelled else "failed"),
                )
                if not delta.empty:
                    wrote_anything = True
                    ctx.emit(delta.to_event(origin="agent"))
                    produced += [a for a in delta.entry.touched + delta.entry.stale_cleared if a not in produced]  # reproduced keys are re-attributed to this step
                    new_replaced = [a for a in delta.entry.replaced_loaded if a not in replaced_loaded]
                    if new_replaced:
                        replaced_loaded += new_replaced
                        ctx.emit(WarningEvent(message="This step replaced data that came with the dataset: " + ", ".join(new_replaced) + ". The original values are no longer in memory; reload the dataset to get them back.", code="replaced_loaded"))
                if res.access:
                    reads |= {f"{r}" for r in res.access.reads}
            if step_writes and not ctx.ablation.provenance:
                wrote_anything = True

            error_text = res.error
            success = res.success
            can_reflect = (not success) and attempt <= max_attempts and should_attempt_fix(res.error_type) and not res.refused and not res.timed_out
            ctx.emit(
                CodeBlockDone(
                    block_id=block_id,
                    step_id=step_id,
                    success=success,
                    error=ErrorView(type=res.error_type or "Error", message=error_text or "", traceback_tail=(res.traceback or "")[-1500:] or None) if not success else None,
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
                results.append(BlockResult(code=code, stdout=res.stdout, success=True, variables=list(res.namespace_added)).model_dump())
                break
            if cancelled(ctx):
                # the user stopped the run: its provenance is recorded above; no reflection, no analysis
                results.append(BlockResult(code=code, stdout=res.stdout, error=error_text, success=False).model_dump())
                current.update(status="cancelled", results=results, produced=produced, reads=sorted(reads), figures=figures)
                return cancel_turn(ctx, state, partial=True, update={"current": current})
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

    all_ok = bool(results) and all(r["success"] for r in results)
    missing: list[str] = []
    # what the code assigned (on any object, qualified per target slice) counts for the output check, so a
    # step that did its work on a copy or reproduced identical values passes; `produced` stays the ledger's
    # list, which is what the analyser, the handoff and the report see (declared `produces` never count)
    claimed = list(produced) + [f"s{sid}:{w}" for w in sorted(code_writes) for sid in current["target_slice_ids"] if f"s{sid}:{w}" not in produced]
    if not ctx.ablation.provenance:
        produced = claimed  # no ledger to tell the two apart
    if all_ok and contract is not None and ctx.ablation.validate_result and contract.produces:
        missing = _missing_outputs(ctx, contract, current["target_slice_ids"], claimed, require_produced=ctx.ablation.provenance)
        ctx.emit(ValidateResult(step_id=step_id, passed=not missing, failures=missing))
    # the code wrote a declared output somewhere (a pooled or subset object, say): the step did its
    # work on a derived object and is accepted; only an inspection-only step gets a continuation
    declared_hit = bool(contract is not None and any(w.split(":", 1)[0] == a.kind and a.matches(w.split(":", 1)[1]) for a in contract.produces for w in code_writes if ":" in w))
    current.update(results=results, produced=produced, reads=sorted(reads), figures=figures, code_writes=sorted(code_writes), replaced_loaded=replaced_loaded)
    if missing and not continuing and not wrote_anything and not declared_hit and not cancelled(ctx):
        # the code ran but wrote nothing (an inspection-only step): one continuation on the live
        # namespace, never a re-run of what already succeeded
        current["continuation"] = {"missing": missing}
        return Command(update={"current": current}, goto="generate")
    current.pop("continuation", None)
    # a step that wrote something but not its declared outputs (results on a derived object, say) is
    # only noted in the pipeline log; a step that wrote nothing at all is reported as partial
    current["incomplete"] = missing if (missing and not wrote_anything and not declared_hit) else []
    current["success"] = all_ok and not current["incomplete"]
    return Command(update={"current": current}, goto="analyze")


async def analyze_node(state: TurnState, runtime: LGRuntime[Context]):
    ctx = runtime.context
    if cancelled(ctx):
        current = dict(state["current"])
        current["status"] = "cancelled"
        return cancel_turn(ctx, state, partial=True, update={"current": current})
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
                incomplete=list(current.get("incomplete") or []),
                turn_steps=turn_steps_block(state.get("handoffs") or []) if ctx.ablation.memory else "",
                stale=[a for a in ctx.ledger.stale_ids() if a.split(":", 1)[0] in {f"s{i}" for i in current["target_slice_ids"]}] if ctx.ablation.provenance else [],
                replaced_loaded=list(current.get("replaced_loaded") or []),
                turn_id=state["turn_id"],
                step=current["step_number"],
            )
        except Exception as exc:  # noqa: BLE001 - auxiliary role: degrade explicitly
            analysis = ctx.roles.analyzer.deterministic(results, produced, stdout_tail, incomplete=list(current.get("incomplete") or []))
            analysis.execution_issues.explanation = (analysis.execution_issues.explanation or "") + f" (analyzer failed: {exc})"
    if analysis is None:
        analysis = ctx.roles.analyzer.deterministic(results, produced, stdout_tail, incomplete=list(current.get("incomplete") or []))
    if analysis.interpretation:
        ctx.emit(TextEvent(segment_id=f"{current['step_id']}-analysis", text=analysis.interpretation, role="analysis", step_id=current["step_id"]))
    if analysis.execution_issues.has_issues:
        ctx.emit(ExecutionIssue(step_id=current["step_id"], issue_type=analysis.execution_issues.issue_type, explanation=analysis.execution_issues.explanation or ""))
    current["analysis"] = analysis.model_dump()
    return Command(update={"current": current}, goto="finish_step")


__all__ = ["generate_node", "execute_node", "analyze_node"]
