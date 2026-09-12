"""Run one session script: user turns, GUI actions and verifiers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from stat_agent.agent import AblationConfig, Agent, Decision
from stat_agent.agent.policy import AutoPolicy, InteractivePolicy, ScriptedPolicy, ScriptedRule
from stat_agent.events import ListEmitter
from stat_bench.session_bench.verifiers import VERIFIERS


class Turn(BaseModel):
    t: int
    kind: str = "user"  # user | gui | check
    text: str | None = None
    action: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    scripted: list[dict[str, Any]] = Field(default_factory=list)
    expect: dict[str, Any] = Field(default_factory=dict)
    verifiers: list[dict[str, Any]] = Field(default_factory=list)


class SessionScript(BaseModel):
    script_id: str
    archetype: str = ""
    fixture: str | None = None
    dataset: str | None = None
    description: str = ""
    setup: dict[str, Any] = Field(default_factory=dict)
    turns: list[Turn] = Field(default_factory=list)
    final: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> SessionScript:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class Ctx:
    """What verifiers see: the live agent plus a map from script turn numbers to agent turns."""

    def __init__(self, agent: Agent, events: list) -> None:
        self.agent, self.session, self.ledger, self.events = agent, agent.session, agent.ledger, events
        self.turn_ids: dict[int, str] = {}  # script t -> agent turn_id
        self.turn_status: dict[int, str | None] = {}
        self.turn_records: dict[int, Any] = {}  # script t -> TurnRecord | None

    @property
    def memory(self):
        return self.agent.memory

    def record(self, t: int):
        return self.turn_records.get(t)


def _policy(mode: str, scripted: list[dict[str, Any]]):
    if mode == "auto":
        return AutoPolicy()
    if mode == "interactive":
        return InteractivePolicy()
    return ScriptedPolicy([ScriptedRule(**s) for s in scripted], default="yes")


def _run_verifiers(ctx: Ctx, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for spec in specs:
        spec = dict(spec)
        name = spec.pop("name")
        fn = VERIFIERS.get(name)
        if fn is None:
            out.append({"name": name, "pass": False, "evidence": "unknown verifier"})
            continue
        try:
            ok, evidence = fn(ctx, **spec)
        except Exception as exc:  # noqa: BLE001
            ok, evidence = False, f"{type(exc).__name__}: {exc}"
        out.append({"name": name, "args": spec, "pass": bool(ok), "evidence": evidence})
    return out


async def run_script(script: SessionScript, agent_factory, *, mode: str = "scripted", driver: str = "staged") -> dict[str, Any]:
    """``agent_factory(script, policy, ablation, sink) -> Agent``; returns the session result."""
    sink = ListEmitter()
    ablation = AblationConfig(driver=driver) if driver == "loop" else AblationConfig()
    agent: Agent = agent_factory(script, _policy(mode, []), ablation, sink)
    ctx = Ctx(agent, sink.events)
    results: list[dict[str, Any]] = []
    t0 = time.time()
    for name, sid, definition in [(r["name"], r.get("slice_id", 0), r["geometry"]) for r in script.setup.get("rois", [])]:
        agent.mutator.create_roi(name, sid, definition)
    for turn in script.turns:
        rec: dict[str, Any] = {"t": turn.t, "kind": turn.kind}
        if turn.kind == "gui":
            fn = {"draw_roi": lambda a: agent.mutator.create_roi(a["name"], a.get("slice_id", 0), a["geometry"]), "delete_roi": lambda a: agent.mutator.delete_roi(a["name"]), "rename_roi": lambda a: agent.mutator.rename_roi(a["old"], a["new"]), "edit_annotation": lambda a: agent.mutator.set_column(a.get("slice_id", 0), a.get("column", "celltype"), agent.session.get_slice(a.get("slice_id", 0)).adata.obs[a.get("column", "celltype")].astype(str).replace(a["mapping"]).values if "mapping" in a else a["value"])}[turn.action]
            res = fn(turn.args)
            rec["provenance"] = res.delta.to_v1() if res.delta else None
        elif turn.kind == "user":
            agent.policy = _policy(mode, turn.scripted) if turn.scripted or mode != "scripted" else agent.policy
            n_before = len(sink.events)
            if agent.is_waiting:
                async for _ in agent.resume(Decision(answer=turn.text or "")):
                    pass
            else:
                async for _ in agent.run_turn(turn.text or ""):
                    pass
            evs = sink.events[n_before:]
            rec["status"] = agent.last_status
            ctx.turn_ids[turn.t] = agent.current_turn_id or ""
            ctx.turn_status[turn.t] = agent.last_status
            ctx.turn_records[turn.t] = next((r for r in reversed(agent.memory.turns) if r.turn_id == agent.current_turn_id), None)
            rec["events"] = [e.type for e in evs]
            exp = turn.expect
            if exp.get("behaviour") == "clarify":
                rec["expect_ok"] = any(e.type in ("prompt", "assumption") and (not exp.get("clarification_kind") or e.kind == exp["clarification_kind"]) for e in evs)
            elif exp.get("behaviour") == "refuse":
                rec["expect_ok"] = any(e.type == "refusal" for e in evs)
            elif exp.get("status"):
                rec["expect_ok"] = agent.last_status == exp["status"]
        rec["verifiers"] = _run_verifiers(ctx, turn.verifiers)
        results.append(rec)
    final = _run_verifiers(ctx, script.final.get("verifiers", []))
    all_checks = [v for r in results for v in r["verifiers"]] + final
    expect_fails = [r for r in results if r.get("expect_ok") is False]
    return {
        "script_id": script.script_id,
        "archetype": script.archetype,
        "mode": mode,
        "driver": driver,
        "turns": results,
        "final": final,
        "all_pass": all(v["pass"] for v in all_checks) and not expect_fails,
        "n_pass": sum(v["pass"] for v in all_checks),
        "n_checks": len(all_checks),
        "wall_s": round(time.time() - t0, 2),
        "usage": agent.llm.meter.total.as_dict(),
    }


def save_session_result(result: dict[str, Any], out: str | Path) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return p


__all__ = ["SessionScript", "Turn", "run_script", "save_session_result"]
