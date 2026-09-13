"""Graph state (checkpointed, JSON-friendly) and the run-scoped Context (live objects)."""

from __future__ import annotations

import operator
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from stat_agent.agent.ablation import AblationConfig
from stat_agent.agent.policy import ClarificationPolicy, Interrupt
from stat_agent.config import Settings
from stat_agent.core.view import InterfaceState, SessionSummary
from stat_agent.events.catalogue import EventBase
from stat_agent.llm import LLMClient
from stat_agent.memory.state import ConversationMemory
from stat_agent.roles import (
    AnalyzerRole,
    GeneratorRole,
    MatcherRole,
    MethodCardRole,
    PlannerRole,
    ReflectorRole,
    SummarizerRole,
    VerifierRole,
)


class TurnState(TypedDict, total=False):
    turn_id: str
    turn_index: int
    query: str
    interface: dict | None
    mode: str
    memory: dict
    plan: list[dict]
    step_index: int
    step_statuses: dict  # {step index: done|failed|skipped}, kept across steps for the plan view
    answers: Annotated[list[dict], operator.add]
    handoffs: Annotated[list[dict], operator.add]
    assumptions: Annotated[list[dict], operator.add]
    pending: dict | None
    current: dict
    status: str
    response: str
    figures: list[str]
    error: str | None
    started_at: float
    retries: int


@dataclass
class Roles:
    planner: PlannerRole
    matcher: MatcherRole
    verifier: VerifierRole
    generator: GeneratorRole
    reflector: ReflectorRole
    analyzer: AnalyzerRole
    method_card: MethodCardRole
    summarizer: SummarizerRole

    @classmethod
    def build(cls, llm: LLMClient, versions: dict[str, int] | None = None) -> Roles:
        v = versions or {}
        return cls(
            planner=PlannerRole(llm, v.get("planner")),
            matcher=MatcherRole(llm, v.get("matcher")),
            verifier=VerifierRole(llm, v.get("verifier")),
            generator=GeneratorRole(llm, v.get("generator")),
            reflector=ReflectorRole(llm, v.get("reflector")),
            analyzer=AnalyzerRole(llm, v.get("analyzer")),
            method_card=MethodCardRole(llm, v.get("method_card")),
            summarizer=SummarizerRole(llm, v.get("summarizer")),
        )

    def prompt_ids(self) -> dict[str, str]:
        return {name: getattr(self, name).prompt_id for name in ("planner", "matcher", "verifier", "generator", "reflector", "analyzer", "method_card", "summarizer")}


@dataclass
class Context:
    """Live, non-serialisable objects available to every node (never stored in state)."""

    session: Any
    registry: Any
    runtime: Any
    ledger: Any
    llm: LLMClient
    roles: Roles
    policy: ClarificationPolicy
    ablation: AblationConfig
    emit: Callable[[EventBase], None]
    settings: Settings = field(default_factory=Settings)
    notebook: Any = None
    figure_store: Callable[[bytes, str, str], str] | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    stream_text: bool = False
    interface: InterfaceState | None = None
    skill_defaults: dict[str, str] = field(default_factory=dict)
    memory_obj: ConversationMemory | None = None
    emitter: Any = None  # the stamping emitter behind `emit`, so nodes can set the current plan step

    def summary(self) -> SessionSummary:
        return SessionSummary.from_session(self.session)

    @property
    def summary_cached(self) -> SessionSummary:  # convenience for policies
        return self.summary()

    @property
    def memory(self) -> ConversationMemory | None:
        return self.memory_obj


# ---- helpers used by nodes ------------------------------------------------------------------------------


def memory_of(state: TurnState, ctx: Context) -> ConversationMemory:
    if ctx.memory_obj is None:
        ctx.memory_obj = ConversationMemory.model_validate(state.get("memory") or {})
    return ctx.memory_obj


def answers_for(state: TurnState, kind: str, skill: str | None = None, step: int | None = None) -> list[tuple[str, str]]:
    out = []
    for a in state.get("answers", []) or []:
        if a.get("kind") != kind:
            continue
        if skill is not None and a.get("skill") not in (None, skill):
            continue
        if step is not None and a.get("step") not in (None, step):
            continue
        out.append((a.get("question", ""), a.get("answer", "")))
    return out


def current_step(state: TurnState) -> dict | None:
    plan = state.get("plan") or []
    idx = state.get("step_index", 0)
    return plan[idx] if 0 <= idx < len(plan) else None


def step_id_for(state: TurnState) -> str:
    return f"{state.get('turn_id', 't')}-s{state.get('step_index', 0) + 1}"


def pending_interrupt(state: TurnState) -> Interrupt | None:
    p = state.get("pending")
    return Interrupt.model_validate(p) if p else None


def now() -> float:
    return time.time()


__all__ = ["TurnState", "Context", "Roles", "memory_of", "answers_for", "current_step", "step_id_for", "pending_interrupt", "now"]
