"""The typed event catalogue.

Every event the agent, the runtime, the web layer or a GUI mutation can produce is one of
the pydantic models below, distinguished by ``type``. Events carry a common header
(``seq``, ``turn_id``, ``step``, ``ts``) that the emitter fills in. The catalogue is the
wire contract of the web protocol (v2) and the schema of ``events.jsonl``.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

PROTOCOL_VERSION = 2


class EventBase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    v: int = PROTOCOL_VERSION
    seq: int | None = None
    turn_id: str | None = None
    step: int | None = None
    ts: float = Field(default_factory=time.time)


# ---- transport-level ---------------------------------------------------------------------------


class Hello(EventBase):
    type: Literal["hello"] = "hello"
    protocol: int = PROTOCOL_VERSION
    last_seq: int = 0
    ledger_version: int = 0
    generation: int = 0
    active_turn: dict[str, Any] | None = None


class Ping(EventBase):
    type: Literal["ping"] = "ping"


class Resync(EventBase):
    type: Literal["resync"] = "resync"
    reason: str = "buffer_evicted"
    last_seq: int = 0


class LockState(EventBase):
    type: Literal["lock_state"] = "lock_state"
    locked: bool
    holder: str | None = None
    phase: str | None = None


class SessionReset(EventBase):
    type: Literal["session_reset"] = "session_reset"
    generation: int


# ---- turn lifecycle --------------------------------------------------------------------------------

Phase = Literal[
    "planning",
    "filtering",
    "matching",
    "synthesizing",
    "verifying",
    "generating",
    "running",
    "validating",
    "reflecting",
    "analysing",
    "waiting",
    "idle",
]


class TurnStarted(EventBase):
    type: Literal["turn_started"] = "turn_started"
    message: str = ""
    mode: str = "interactive"
    driver: str = "staged"
    turn_index: int = 0
    action: dict[str, Any] | None = None


class PhaseEvent(EventBase):
    type: Literal["phase"] = "phase"
    phase: Phase
    detail: str | None = None
    elapsed_ms: int | None = None


class PlanStepView(BaseModel):
    step_id: str
    index: int
    description: str
    status: Literal["pending", "active", "done", "skipped", "failed", "blocked"] = "pending"
    skill: str | None = None
    target_slices: list[int] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)


class PlanUpdated(EventBase):
    type: Literal["plan_updated"] = "plan_updated"
    plan_id: str
    reason: Literal["created", "step_started", "step_done", "revised"] = "created"
    steps: list[PlanStepView] = Field(default_factory=list)


class StepStarted(EventBase):
    type: Literal["step_started"] = "step_started"
    step_id: str
    index: int
    total: int
    description: str = ""
    target_slices: list[int] = Field(default_factory=list)


class StepDone(EventBase):
    type: Literal["step_done"] = "step_done"
    step_id: str
    index: int
    status: Literal["done", "failed", "blocked", "skipped"] = "done"
    mode: str | None = None
    figures: list[str] = Field(default_factory=list)
    summary: str | None = None


class RejectedSkill(BaseModel):
    slug: str
    reason: str


class FilterComplete(EventBase):
    type: Literal["filter_complete"] = "filter_complete"
    step_id: str | None = None
    compatible: list[str] = Field(default_factory=list)
    rejected: list[RejectedSkill] = Field(default_factory=list)
    bypassed: bool = False


class MatchComplete(EventBase):
    type: Literal["match_complete"] = "match_complete"
    step_id: str | None = None
    matched: list[str] = Field(default_factory=list)
    reason: str | None = None


class SkillSelected(EventBase):
    type: Literal["skill_selected"] = "skill_selected"
    step_id: str | None = None
    skill: str
    name: str | None = None
    alternatives: list[str] = Field(default_factory=list)


class VerdictView(BaseModel):
    prerequisite: str
    status: Literal["satisfied", "ask_user", "not_obtainable"]
    evidence: str | None = None
    question: str | None = None


class VerifyComplete(EventBase):
    type: Literal["verify_complete"] = "verify_complete"
    step_id: str | None = None
    skill: str | None = None
    verdicts: list[VerdictView] = Field(default_factory=list)
    decided_by: Literal["contract", "llm", "bypass"] = "llm"
    complete_query: str | None = None


# ---- clarification ------------------------------------------------------------------------------------

PromptKind = Literal["planner", "verifier", "skill_selection"]


class QuestionView(BaseModel):
    id: str
    text: str
    hint: str | None = None


class OptionView(BaseModel):
    index: int
    label: str
    slug: str | None = None
    description: str | None = None


class PromptEvent(EventBase):
    type: Literal["prompt"] = "prompt"
    prompt_id: str
    kind: PromptKind
    step_id: str | None = None
    question: str | None = None
    questions: list[QuestionView] = Field(default_factory=list)
    options: list[OptionView] = Field(default_factory=list)
    skill: str | None = None
    allow_free_text: bool = True
    default: str | None = None


class PromptAnswered(EventBase):
    type: Literal["prompt_answered"] = "prompt_answered"
    prompt_id: str
    decision: dict[str, Any] = Field(default_factory=dict)
    source: Literal["user", "auto", "scripted", "override"] = "user"


class AssumptionEvent(EventBase):
    type: Literal["assumption"] = "assumption"
    assumption_id: str
    kind: PromptKind
    step_id: str | None = None
    question: str
    chosen: str
    rationale: str = ""
    source: Literal["skill_default", "session_state", "fact", "policy_default", "scripted"] = "policy_default"
    can_override: bool = True


# ---- execution ------------------------------------------------------------------------------------------


class CodeBlockStarted(EventBase):
    type: Literal["code_block_started"] = "code_block_started"
    block_id: str
    step_id: str | None = None
    code: str
    language: str = "python"
    attempt: int = 1


class OutputLine(BaseModel):
    stream: Literal["stdout", "stderr"] = "stdout"
    text: str


class ExecOutput(EventBase):
    type: Literal["exec_output"] = "exec_output"
    block_id: str
    lines: list[OutputLine] = Field(default_factory=list)


class ErrorView(BaseModel):
    type: str
    message: str
    traceback_tail: str | None = None


class CodeBlockDone(EventBase):
    type: Literal["code_block_done"] = "code_block_done"
    block_id: str
    step_id: str | None = None
    success: bool
    error: ErrorView | None = None
    duration_ms: int = 0
    figures: list[str] = Field(default_factory=list)
    attempt: int = 1
    will_retry: bool = False
    timed_out: bool = False


class FigureEvent(EventBase):
    type: Literal["figure"] = "figure"
    figure_id: str
    block_id: str | None = None
    step_id: str | None = None
    url: str | None = None
    width: int | None = None
    height: int | None = None
    caption: str | None = None
    sha256: str | None = None
    png_b64: str | None = Field(default=None, exclude=True)


class ExecutionIssue(EventBase):
    type: Literal["execution_issue"] = "execution_issue"
    block_id: str | None = None
    step_id: str | None = None
    issue_type: str | None = None
    explanation: str = ""


class ReflectionEvent(EventBase):
    type: Literal["reflection"] = "reflection"
    block_id: str | None = None
    attempt: int
    max_attempts: int
    summary: str = ""
    change_magnitude: str | None = None
    confidence: float | None = None
    will_retry: bool = False


class TextEvent(EventBase):
    type: Literal["text"] = "text"
    segment_id: str
    text: str
    role: Literal["answer", "analysis", "note"] = "answer"
    step_id: str | None = None


class TextDelta(EventBase):
    type: Literal["text_delta"] = "text_delta"
    segment_id: str
    delta: str


# ---- provenance / state -----------------------------------------------------------------------------------


class ArtifactRefView(BaseModel):
    kind: str
    key: str
    slice_id: int | None = None

    @property
    def id(self) -> str:
        prefix = f"s{self.slice_id}:" if self.slice_id is not None else ""
        return f"{prefix}{self.kind}:{self.key}"


class ArtifactChange(BaseModel):
    artifact: ArtifactRefView
    op: Literal["created", "updated", "deleted", "renamed"]
    version: int = 1
    depends_on: list[str] = Field(default_factory=list)


class StateChangeSummary(BaseModel):
    celltypes_updated: list[int] = Field(default_factory=list)
    deconv_weights_updated: list[int] = Field(default_factory=list)
    celltype_colors_updated: list[int] = Field(default_factory=list)
    rois_added: list[str] = Field(default_factory=list)
    rois_deleted: list[str] = Field(default_factory=list)
    rois_renamed: list[list[str]] = Field(default_factory=list)


class ProvenanceUpdate(EventBase):
    type: Literal["provenance_update"] = "provenance_update"
    ledger_version: int
    origin: Literal["agent", "gui", "load", "cancelled_turn", "rerun"] = "agent"
    producer: dict[str, Any] | None = None
    changes: list[ArtifactChange] = Field(default_factory=list)
    stale_now: list[str] = Field(default_factory=list)
    stale_cleared: list[str] = Field(default_factory=list)
    summary: StateChangeSummary = Field(default_factory=StateChangeSummary)


class UsageEvent(EventBase):
    type: Literal["usage"] = "usage"
    role: str | None = None
    step_id: str | None = None
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_ms: int = 0
    cumulative: bool = True
    model: str | None = None


class NotebookCellAdded(EventBase):
    type: Literal["notebook_cell_added"] = "notebook_cell_added"
    cell_index: int
    kind: Literal["code", "markdown"] = "code"
    source_preview: str = ""
    notebook_path: str | None = None


class AdviceEvent(EventBase):
    type: Literal["advice"] = "advice"
    message: str
    skill: str | None = None
    step_id: str | None = None
    unmet: list[str] = Field(default_factory=list)


class WarningEvent(EventBase):
    type: Literal["warning"] = "warning"
    message: str
    code: str | None = None


class ErrorEvent(EventBase):
    type: Literal["error"] = "error"
    code: str = "internal"
    message: str
    detail: str | None = None
    recoverable: bool = True
    stage: str | None = None


class TurnDone(EventBase):
    type: Literal["turn_done"] = "turn_done"
    state: Literal["done", "blocked"] = "done"
    summary: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    ledger_version: int = 0
    message: str | None = None
    response: str | None = None
    figures: list[str] = Field(default_factory=list)


class TurnError(EventBase):
    type: Literal["turn_error"] = "turn_error"
    code: str = "internal"
    message: str
    detail: str | None = None


class TurnCancelled(EventBase):
    type: Literal["turn_cancelled"] = "turn_cancelled"
    partial: bool = False
    exec_still_running: bool = False
    ledger_version: int = 0


# ---- out-of-registry & guards ---------------------------------------------------------------------------------


class MethodCardEvent(EventBase):
    type: Literal["method_card"] = "method_card"
    step_id: str | None = None
    card: dict[str, Any] = Field(default_factory=dict)


class UnguardedMode(EventBase):
    type: Literal["unguarded_mode"] = "unguarded_mode"
    step_id: str | None = None
    reason: str = "no registered skill"


class RefusalEvent(EventBase):
    type: Literal["refusal"] = "refusal"
    step_id: str | None = None
    kind: Literal["out_of_registry", "missing_prereq", "infeasible", "unsafe", "other"] = "missing_prereq"
    requirement: str | None = None
    reason: str = ""


class ValidateResult(EventBase):
    type: Literal["validate_result"] = "validate_result"
    step_id: str | None = None
    passed: bool
    failures: list[str] = Field(default_factory=list)


class SkillContractViolation(EventBase):
    type: Literal["skill_contract_violation"] = "skill_contract_violation"
    step_id: str | None = None
    slug: str
    missing: list[str] = Field(default_factory=list)


class DraftOffer(EventBase):
    type: Literal["draft_offer"] = "draft_offer"
    step_id: str | None = None
    slug: str
    path: str | None = None


Event = Annotated[
    Hello | Ping | Resync | LockState | SessionReset | TurnStarted | PhaseEvent | PlanUpdated | StepStarted | StepDone | FilterComplete | MatchComplete | SkillSelected | VerifyComplete | PromptEvent | PromptAnswered | AssumptionEvent | CodeBlockStarted | ExecOutput | CodeBlockDone | FigureEvent | ExecutionIssue | ReflectionEvent | TextEvent | TextDelta | ProvenanceUpdate | UsageEvent | NotebookCellAdded | AdviceEvent | WarningEvent | ErrorEvent | TurnDone | TurnError | TurnCancelled | MethodCardEvent | UnguardedMode | RefusalEvent | ValidateResult | SkillContractViolation | DraftOffer,
    Field(discriminator="type"),
]

EVENT_ADAPTER: TypeAdapter = TypeAdapter(Event)
TERMINAL_TYPES = frozenset({"turn_done", "turn_error", "turn_cancelled"})
INTERRUPT_TYPES = frozenset({"prompt"})


def parse_event(data: dict[str, Any]) -> EventBase:
    return EVENT_ADAPTER.validate_python(data)


def to_wire(event: EventBase) -> dict[str, Any]:
    """Plain dict for JSON transport (drops fields marked ``exclude``, e.g. figure bytes)."""
    return event.model_dump(mode="json", exclude_none=False)


def event_type_names() -> list[str]:
    return sorted(cls.model_fields["type"].default for cls in Event.__args__[0].__args__)  # type: ignore[attr-defined]


__all__ = [name for name in dir() if name[0].isupper()] + ["parse_event", "to_wire", "event_type_names", "PROTOCOL_VERSION"]
