"""STAT v2 as a benchmarked system (in-process)."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from stat_agent.agent import AblationConfig, Agent
from stat_agent.agent.policy import AutoPolicy, InteractivePolicy, ScriptedPolicy, ScriptedRule
from stat_agent.core import SimpleSession
from stat_agent.events import ListEmitter
from stat_agent.llm import CallLogger, LLMClient, UsageMeter, make_spec
from stat_agent.skills import SkillRegistry
from stat_bench.result import RunResult, software_info
from stat_bench.score.sse import detect_sse
from stat_bench.systems.base import Cell


def _seed_everything(seed: int) -> None:
    import os
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001
        pass


def policy_for(cell: Cell):
    q = cell.query
    if cell.policy == "auto":
        return AutoPolicy(q.auto_hints)
    if cell.policy == "interactive":
        return InteractivePolicy()
    rules = [ScriptedRule(kind=s.kind, match=s.match, answer=s.answer, once=s.once) for s in q.scripted]
    return ScriptedPolicy(rules, skill_pick=q.skill_pick, default=q.scripted_default)


class StatSystem:
    name = "stat"

    def __init__(self, registry: SkillRegistry | None = None, llm_factory=None, session_factory=None) -> None:
        self.registry = registry
        self.llm_factory = llm_factory  # (cell) -> LLMClient (tests inject scripted responders)
        self.session_factory = session_factory  # (cell) -> SimpleSession (tests inject fixtures)

    def _session(self, cell: Cell, tmp: Path) -> SimpleSession:
        if self.session_factory is not None:
            return self.session_factory(cell)
        root = Path(cell.data_root or ".")
        ds = tmp / "dataset"
        ds.mkdir(parents=True, exist_ok=True)
        for rel in cell.query.inputs:
            src = root / rel
            if src.is_dir():
                for f in src.iterdir():
                    _copy(f, ds / f.name)
            else:
                _copy(src, ds / src.name)
        if cell.query.reference:
            # a reference is not a slice: the loader reads only the top level, and files in a subdirectory are
            # listed as extra_files for the roles; the query text names it by its bare name, so it is also placed
            # in the run's working directory (StatSystem.run chdirs into cell.out_dir)
            ref = root / cell.query.reference
            (ds / "ref").mkdir(exist_ok=True)
            _copy(ref, ds / "ref" / ref.name)
            _copy(ref, Path(cell.out_dir) / ref.name)
        if cell.query.image:
            img = root / cell.query.image
            _copy(img, ds / img.name)  # he.tif / he_slice_0.tif: matched to slice 0 by the loader
        session = SimpleSession(name=cell.query.id)
        session.load_dataset(ds)
        return session

    async def run(self, cell: Cell) -> RunResult:
        q = cell.query
        _seed_everything(cell.seed)
        tmp = Path(tempfile.mkdtemp(prefix=f"stat_{q.id}_"))
        out = Path(cell.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cwd0 = os.getcwd()
        timeout_s = int(cell.extra.get("timeout_s") or q.timeout_s or 0)
        t0 = time.time()
        res = RunResult(query_id=q.id, query_text=q.text, task=q.category, category=q.category, dataset=q.dataset, difficulty=q.difficulty, has_ground_truth=q.ground_truth is not None, system=self.name, model=cell.backbone, suite_id=cell.suite_id, driver=cell.driver, ablation={"name": cell.ablation}, policy=cell.policy, seed=cell.seed, seed_index=cell.seed_index, timeout_s=timeout_s, software=software_info(), run_id=hashlib.blake2b(f"{cell.suite_id}|{q.id}|{cell.seed_index}".encode(), digest_size=6).hexdigest())
        try:
            os.chdir(out)  # relative paths in the query text and relative writes of generated code resolve in the run directory
            session = self._session(cell, tmp)
            llm = self.llm_factory(cell) if self.llm_factory else LLMClient(spec=make_spec(cell.backbone), calllog=CallLogger(), meter=UsageMeter())
            ablation = AblationConfig.parse(cell.ablation)
            if cell.driver == "loop":
                ablation.driver = "loop"
            sink = ListEmitter()
            agent = Agent(session, llm, registry=self.registry, mode="auto" if cell.policy == "auto" else "interactive", policy=policy_for(cell), ablation=ablation, session_dir=out / "logs", sinks=[sink], durable=False, loop_model=cell.extra.get("loop_model"))
            res.backbone = {"model_id": cell.backbone, "temperature_supported": True}
            events = []

            async def _drain() -> None:
                async for ev in agent.run_turn(q.text):
                    events.append(ev)

            try:
                await asyncio.wait_for(_drain(), timeout=timeout_s or None)
            except TimeoutError:
                agent.cancel()  # stops the graph at its next check and the running code; what exists is recorded
                res.timed_out = True
            turn_errors = [e for e in events if e.type == "turn_error"]
            if res.timed_out:
                res.exit_status = "timeout"
            elif turn_errors and _is_provider_error(turn_errors[-1].message, turn_errors[-1].code):
                res.exit_status = "provider_error"
            # interactive/scripted-miss: a pending prompt ends the run
            elif agent.is_waiting:
                res.exit_status = "clarification_pending"
                res.refusal = {"refused": False, "reason_kind": "clarification_pending", "reason_text": agent.pending.question or " | ".join(agent.pending.questions)}
            else:
                res.exit_status = {"done": "ok", "blocked": "partial", "error": "crash"}.get(agent.last_status or "done", "ok")
            res.response = agent.last_response or ""
            res.answer = res.response
            res.success = res.exit_status == "ok"
            res.wall_s = time.time() - t0
            totals = llm.meter.turn_totals(agent.current_turn_id or "") if agent.current_turn_id else llm.meter.total
            res.prompt_tokens, res.completion_tokens, res.total_tokens, res.llm_calls = totals.input_tokens, totals.output_tokens, totals.total_tokens, totals.llm_calls
            res.usage_calls = [{"role": c.role, "model": c.model, "prompt_tokens": c.input_tokens, "completion_tokens": c.output_tokens, "latency_s": round(c.latency_s, 3), "mode": c.mode, "repairs": c.repairs} for c in llm.meter.calls]
            turn = agent.memory.turns[-1] if agent.memory.turns else None
            if turn:
                res.code = "\n\n# ----\n\n".join(c for h in turn.handoffs for c in h.code)
                res.assumptions = [a.model_dump() for a in turn.assumptions]
                res.artifacts_produced = sorted({o for h in turn.handoffs for o in h.outputs})
            res.clarifications = [{"kind": e.kind, "question": e.question or " | ".join(q.text for q in e.questions)} for e in events if e.type == "prompt"]
            refusals = [e for e in events if e.type == "refusal"]
            if refusals:
                res.refusal = {"refused": True, "reason_kind": refusals[0].kind, "reason_text": refusals[0].reason, "named_requirement": refusals[0].requirement}
                res.exit_status = "refused" if res.exit_status != "ok" else res.exit_status
            cards = [e for e in events if e.type == "method_card"]
            if cards:
                res.method_card = {"used": True, "provisional": True, **cards[0].card}
            res.decisions = _decisions(events)
            res.stdout = "\n".join(ln.text for e in events if e.type == "exec_output" for ln in e.lines)[-20000:]
            res.keys_read_missing = sorted({e.error.message for e in events if e.type == "code_block_done" and e.error and e.error.type == "KeyError"})
            res.provenance_snapshot_hash = agent.ledger.consistency_hash()
            res.events_path = str(out / "logs" / "events.jsonl") if (out / "logs" / "events.jsonl").exists() else None
            if cell.execute and cell.extra.get("save_h5ad", True) and session.has_data:
                try:
                    import anndata as ad

                    slices = [s.adata for s in session.iter_slices()]
                    adata = slices[0] if len(slices) == 1 else ad.concat(slices, label="_slice_id", index_unique="-")
                    path = out / "output.h5ad"
                    adata.write_h5ad(path)
                    res.output_h5ad = str(path)
                except Exception as exc:  # noqa: BLE001
                    res.errors.append({"stage": "save_h5ad", "type": type(exc).__name__, "message": str(exc)})
            res.sse = detect_sse(res, session)
            agent.save_state()
        except Exception as exc:  # noqa: BLE001
            res.exit_status = "provider_error" if any(_is_provider_error(str(exc), c.__name__) for c in type(exc).__mro__) else "crash"
            res.errors.append({"stage": "run", "type": type(exc).__name__, "message": str(exc)[:2000]})
            res.wall_s = time.time() - t0
        finally:
            os.chdir(cwd0)
            shutil.rmtree(tmp, ignore_errors=True)
        res.finished_at = time.time()
        res.save(out)
        return res


def _copy(src: Path, dst: Path) -> None:
    """APFS clone when possible (no disk cost), plain copy otherwise."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode != 0:
        shutil.copy2(src, dst)


_PROVIDER_ERROR_NAMES = ("APIConnectionError", "OpenAIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "ServiceUnavailableError", "ConnectError", "ReadTimeout", "ConnectTimeout", "RemoteProtocolError", "ConnectionError", "TimeoutError", "IncompleteRead", "ChunkedEncodingError")
_PROVIDER_ERROR_TEXT = ("connection error", "connection reset", "connection aborted", "timed out", "temporarily unavailable", "overloaded", "rate limit", "bad gateway", "gateway time", "server error", "empty response", "eof occurred", "error code: 429", "error code: 5", "error code: 408")


def _is_provider_error(message: str, code: str | None = None) -> bool:
    """A network or provider-side failure (retried by the caller, reported as provider_error), not a STAT failure."""
    if code and code in _PROVIDER_ERROR_NAMES:
        return True
    msg = (message or "").lower()
    return any(n.lower() in msg for n in _PROVIDER_ERROR_NAMES) or any(t in msg for t in _PROVIDER_ERROR_TEXT)


def _decisions(events: list) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for e in events:
        if e.type == "plan_updated" and "plan" not in d:
            d["plan"] = {"n_steps": len(e.steps), "target_slices": [s.target_slices for s in e.steps]}
        elif e.type == "filter_complete":
            d["filter"] = {"compatible": e.compatible, "rejected": [r.slug for r in e.rejected], "bypassed": e.bypassed}
        elif e.type == "match_complete":
            d.setdefault("matched", []).append(e.matched)
        elif e.type == "skill_selected":
            d["selected_skill"] = e.skill
        elif e.type == "verify_complete":
            d["verdict"] = {"decided_by": e.decided_by, "n_verdicts": len(e.verdicts)}
        elif e.type == "reflection":
            d["reflect_attempts"] = d.get("reflect_attempts", 0) + 1
    return d


__all__ = ["StatSystem", "policy_for"]
