"""TurnManager: at most one non-terminal turn; drives the agent and publishes its events."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from stat_agent.agent.policy import Decision, make_policy
from stat_agent.events.catalogue import TERMINAL_TYPES, EventBase, TurnCancelled, TurnDone, TurnError
from stat_agent.web.errors import ApiError

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"done", "error", "cancelled"}


@dataclass
class Turn:
    turn_id: str
    generation: int
    message: str
    mode: str
    driver: str = "staged"
    client_turn_id: str | None = None
    state: str = "queued"
    terminal: str | None = None  # terminal event seen; applied to `state` only after the agent finished
    outcome: str | None = None  # turn_done.state (done | blocked | cancelled), error for turn_error; None while the turn runs
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    pending_prompt: dict | None = None
    first_seq: int | None = None
    last_seq: int | None = None
    response: str = ""
    figures: list[str] = field(default_factory=list)
    error: str | None = None
    task: asyncio.Task | None = None
    cancelling: bool = False

    def snapshot(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "generation": self.generation,
            "message": self.message,
            "mode": self.mode,
            "driver": self.driver,
            "state": self.state,
            "outcome": self.outcome,
            "cancelling": self.cancelling,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pending_prompt": self.pending_prompt,
            "pending_prompt_id": (self.pending_prompt or {}).get("prompt_id"),
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "response": self.response,
            "figures": list(self.figures),
            "error": self.error,
        }


class TurnManager:
    def __init__(self, app_state) -> None:
        self.app = app_state
        self.active: Turn | None = None
        self.history: list[Turn] = []

    def reset(self) -> None:
        if self.active and self.active.task and not self.active.task.done():
            self.active.task.cancel()
        self.active = None
        self.history = []

    def active_snapshot(self) -> dict | None:
        return self.active.snapshot() if self.active and self.active.state not in TERMINAL_STATES else None

    def get(self, turn_id: str) -> Turn:
        if self.active and self.active.turn_id == turn_id:
            return self.active
        for t in self.history:
            if t.turn_id == turn_id:
                return t
        raise ApiError("not_found", f"unknown turn {turn_id}", 404)

    # ---- start -----------------------------------------------------------------------------------------
    async def start(self, message: str | None, *, mode: str | None = None, driver: str | None = None, client_turn_id: str | None = None, interface: dict | None = None, action: dict | None = None) -> Turn:
        agent = self.app.require_agent()
        if client_turn_id and self.active and self.active.client_turn_id == client_turn_id:
            return self.active
        if self.active and self.active.state not in TERMINAL_STATES:
            if self.active.cancelling:
                raise ApiError("busy", "the previous turn is still stopping", 409, {"turn_id": self.active.turn_id, "reason": "cancelling"}, {"Retry-After": "2"})
            raise ApiError("turn_active", "a turn is already running; answer its prompt or cancel it first", 409, self.active.snapshot())
        if mode and mode != agent.mode:
            agent.mode = mode
            agent.policy = make_policy(mode)
        if action and action.get("kind") == "rerun_downstream":
            message = message or agent.rerun_message(action.get("artifacts") or None)
        turn = Turn(turn_id=f"t{agent.turn_index + 1:04d}", generation=self.app.generation, message=message or "", mode=agent.mode, driver=driver or agent.ablation.driver, client_turn_id=client_turn_id)
        self.active = turn
        turn.state = "running"
        turn.task = asyncio.create_task(self._run(turn, agent.run_turn(message or "", interface=interface)))
        await asyncio.sleep(0)  # let the first events land before returning
        return turn

    async def reply(self, turn_id: str, decision: dict) -> Turn:
        turn = self.get(turn_id)
        agent = self.app.require_agent()
        if turn.state != "awaiting_input" or not turn.pending_prompt:
            raise ApiError("no_pending_prompt", "this turn is not waiting for input", 409, turn.snapshot())
        pid = decision.get("prompt_id")
        if pid and pid != turn.pending_prompt.get("prompt_id"):
            raise ApiError("prompt_mismatch", "decision is for a different prompt", 409, {"pending_prompt_id": turn.pending_prompt.get("prompt_id")})
        dec = self._to_decision(turn, decision)
        turn.state = "running"
        turn.terminal = None
        turn.pending_prompt = None
        turn.task = asyncio.create_task(self._run(turn, agent.resume(dec)))
        await asyncio.sleep(0)
        return turn

    async def cancel(self, turn_id: str) -> Turn:
        turn = self.get(turn_id)
        agent = self.app.require_agent()
        if turn.state in TERMINAL_STATES:
            return turn
        turn.cancelling = True
        agent.cancel()
        if turn.state == "awaiting_input":
            # the graph is parked at an interrupt and will not run again: close the turn on the stream too
            agent.pending = None
            self._on_event(turn, TurnCancelled(partial=False, ledger_version=agent.ledger.version))
            self._on_event(turn, TurnDone(state="cancelled", ledger_version=agent.ledger.version, summary={"steps": 0, "status": "cancelled"}, response=""))
            turn.state = "cancelled"
            turn.cancelling = False
            turn.finished_at = time.time()
            self._archive(turn)
        return turn

    # ---- internals ---------------------------------------------------------------------------------------
    @staticmethod
    def _to_decision(turn: Turn, d: dict) -> Decision:
        # the documented body ({prompt_id, answer | option_index | answers}) and the viewer's body
        # ({prompt_id, kind, text | index | answers}) are both accepted; an empty reply is rejected
        d = dict(d)
        if d.get("option_index") is not None and d.get("index") is None:
            d["index"] = d["option_index"]
        if d.get("answer") is not None and d.get("text") is None:
            d["text"] = d["answer"]
        kind = d.get("kind") or ("option" if d.get("index") is not None else ("prerequisites" if isinstance(d.get("answers"), dict) else "answer"))
        if kind == "answer" and not (d.get("text") or "").strip():
            raise ApiError("validation", "the reply carries no answer (use answer, option_index or answers)", 400)
        pending_kind = (turn.pending_prompt or {}).get("kind")
        if kind == "assumption_override":
            d = d.get("decision") or {}
            kind = d.get("kind", "answer")
        if kind == "option":
            raw = d.get("index")
            if raw is None or not str(raw).strip().isdigit() or int(raw) < 1:
                raise ApiError("validation", "the reply names no option (option_index starts at 1)", 400)
            return Decision(prompt_id=d.get("prompt_id"), option_index=int(raw), source="override" if d.get("source") == "override" else "user")
        if kind == "prerequisites":
            answers = d.get("answers") or {}
            if not any(str(v or "").strip() for v in answers.values()):
                raise ApiError("validation", "the reply carries no answers", 400)
            questions = (turn.pending_prompt or {}).get("questions") or []
            # map q1/q2 ids back to question text
            mapped = {}
            for k, v in answers.items():
                match = next((q["text"] for q in questions if q.get("id") == k), None)
                mapped[match or k] = v
            return Decision(prompt_id=d.get("prompt_id"), answers=mapped, kind="verifier")
        text = d.get("text") or ""
        if pending_kind == "skill_selection" and text.strip().isdigit():
            return Decision(prompt_id=d.get("prompt_id"), option_index=int(text.strip()))
        return Decision(prompt_id=d.get("prompt_id"), answer=text)

    async def _run(self, turn: Turn, aiter) -> None:
        agent = self.app.require_agent()
        try:
            async for ev in aiter:
                self._on_event(turn, ev)
        except asyncio.CancelledError:
            turn.state = "cancelled"
            turn.finished_at = time.time()
            self._archive(turn)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("turn %s crashed", turn.turn_id)
            self._on_event(turn, TurnError(code=type(exc).__name__, message=str(exc)[:2000], turn_id=turn.turn_id))
        finally:
            if turn.terminal and turn.state != "cancelled":
                turn.state = turn.terminal
                turn.finished_at = turn.finished_at or time.time()
            elif agent.is_waiting and turn.state == "running":
                turn.state = "awaiting_input"
                turn.pending_prompt = self._pending_payload(agent)
            turn.terminal = None
            if turn.state in TERMINAL_STATES:
                self._archive(turn)
            try:
                agent.save_state()
            except Exception:  # noqa: BLE001
                pass

    def _pending_payload(self, agent) -> dict | None:
        p = agent.pending
        if p is None:
            return None
        return {"prompt_id": p.prompt_id, "kind": p.kind, "question": p.question, "questions": [{"id": f"q{i + 1}", "text": q} for i, q in enumerate(p.questions)], "options": p.options, "skill": p.skill}

    def _on_event(self, turn: Turn, ev: EventBase) -> None:
        ev.turn_id = ev.turn_id or turn.turn_id
        seq = self.app.events.publish(ev)
        turn.first_seq = turn.first_seq or seq
        turn.last_seq = seq
        t = ev.type  # type: ignore[attr-defined]
        # states are applied in _run() after the agent has absorbed the result: a client that sees
        # "done" may save the session, and one that sees "awaiting_input" may reply, immediately
        if t == "prompt":
            turn.pending_prompt = {"prompt_id": ev.prompt_id, "kind": ev.kind, "question": ev.question, "questions": [q.model_dump() for q in ev.questions], "options": [o.model_dump() for o in ev.options], "skill": ev.skill}
        elif t == "prompt_answered":
            turn.pending_prompt = None
        elif t == "figure":
            turn.figures.append(ev.figure_id)
        elif t in TERMINAL_TYPES:
            turn.terminal = {"turn_done": "done", "turn_error": "error", "turn_cancelled": "cancelled"}[t]
            turn.outcome = getattr(ev, "state", None) if t == "turn_done" else turn.terminal
            if t == "turn_done" and getattr(ev, "state", None) == "cancelled":
                turn.terminal = "cancelled"
            turn.cancelling = False
            if t == "turn_done":
                turn.response = ev.response or ""
                turn.figures = list(ev.figures) or turn.figures
            elif t == "turn_error":
                turn.error = ev.message

    def _archive(self, turn: Turn) -> None:
        if self.active is turn:
            self.history.append(turn)
            self.active = None
        elif turn not in self.history:
            self.history.append(turn)


__all__ = ["TurnManager", "Turn", "TERMINAL_STATES"]
