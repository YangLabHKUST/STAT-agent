"""Agent facade: run a turn, answer a pending prompt, cancel, and stream typed events."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from stat_agent.agent.ablation import AblationConfig
from stat_agent.agent.graph import build_graph
from stat_agent.agent.policy import ClarificationPolicy, Decision, Interrupt, make_policy
from stat_agent.agent.state import Context, Roles, pending_interrupt
from stat_agent.config import Settings
from stat_agent.core.view import InterfaceState
from stat_agent.events import JsonlSink, MultiEmitter, StampingEmitter
from stat_agent.events.catalogue import EventBase, TurnDone, TurnError, TurnStarted
from stat_agent.ledger import ProvenanceLedger, SessionMutator
from stat_agent.llm import LLMClient, make_spec
from stat_agent.memory.state import ConversationMemory
from stat_agent.runtime import Runtime, take_snapshot
from stat_agent.skills import SkillRegistry

logger = logging.getLogger(__name__)


class AgentBusy(RuntimeError):
    pass


class _QueueEmitter:
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        self._loop, self._queue = loop, queue
        self._ident = threading.get_ident()

    def emit(self, event: EventBase) -> None:
        if threading.get_ident() == self._ident:
            self._queue.put_nowait(event)
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)


class Agent:
    """One agent per session. Drives the staged graph (or the loop agent) turn by turn."""

    def __init__(
        self,
        session: Any,
        llm: LLMClient | str,
        *,
        registry: SkillRegistry | None = None,
        settings: Settings | None = None,
        mode: str = "interactive",
        policy: ClarificationPolicy | None = None,
        ablation: AblationConfig | str | dict | None = None,
        session_dir: str | Path | None = None,
        checkpointer: Any = None,
        stream_text: bool = False,
        prompt_versions: dict[str, int] | None = None,
        sinks: list[Any] = (),
        notebook: Any = None,
        session_id: str | None = None,
        durable: bool = True,
        loop_model: Any = None,
    ) -> None:
        self.session = session
        self.settings = settings or Settings.from_env()
        self.llm = LLMClient(spec=make_spec(llm)) if isinstance(llm, str) else llm
        self.registry = registry or SkillRegistry.from_settings(self.settings)
        self.mode = mode
        self.policy = policy or make_policy(mode)
        self.ablation = ablation if isinstance(ablation, AblationConfig) else AblationConfig.parse(ablation)
        self.session_dir = Path(session_dir) if session_dir else None
        self.stream_text = stream_text
        self.session_id = session_id or getattr(session, "name", "session") + "-" + uuid.uuid4().hex[:6]
        self.roles = Roles.build(self.llm, prompt_versions)
        self.runtime = Runtime(session, timeout_s=self.settings.exec_timeout_s, workdir=self.session_dir)
        self.ledger = ProvenanceLedger()
        self.ledger.initialize(take_snapshot(session))
        self.memory = ConversationMemory()
        self.mutator = SessionMutator(session, self.ledger)
        self.notebook = notebook
        self.figures: dict[str, bytes] = {}
        self.turn_index = 0
        self.current_turn_id: str | None = None
        self.pending: Interrupt | None = None
        self.last_status: str | None = None
        self.last_response: str = ""
        self.cancel_event = threading.Event()
        self._sinks: list[Any] = list(sinks)
        self._running = False
        self._checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(self._checkpointer)
        self.loop_model = loop_model
        self._loop_runner = None
        self.durable = durable and session_dir is not None and checkpointer is None
        self._conn = None
        if self.session_dir:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.llm.calllog.set_session_dir(self.session_dir)
            self._sinks.append(JsonlSink(self.session_dir / "events.jsonl"))
            (self.session_dir / "figures").mkdir(exist_ok=True)

    # ---- properties ----------------------------------------------------------------------------------
    @property
    def is_waiting(self) -> bool:
        return self.pending is not None

    @property
    def busy(self) -> bool:
        return self._running

    def figure_store(self, png: bytes, step_id: str, block_id: str) -> str:
        n = sum(1 for k in self.figures if k.startswith(block_id))
        fid = f"{block_id}-f{n + 1}"
        self.figures[fid] = png
        if self.session_dir:
            try:
                (self.session_dir / "figures" / f"{fid}.png").write_bytes(png)
            except OSError:
                pass
        return fid

    def _context(self, emit, interface: InterfaceState | None) -> Context:
        return Context(
            session=self.session,
            registry=self.registry,
            runtime=self.runtime,
            ledger=self.ledger,
            llm=self.llm,
            roles=self.roles,
            policy=self.policy,
            ablation=self.ablation,
            emit=emit,
            settings=self.settings,
            notebook=self.notebook,
            figure_store=self.figure_store,
            cancel=self.cancel_event,
            stream_text=self.stream_text,
            interface=interface,
        )

    def _config(self, turn_id: str) -> dict:
        return {"configurable": {"thread_id": f"{self.session_id}:{turn_id}"}}

    async def _ensure_graph(self) -> None:
        """Switch to the durable SQLite checkpointer the first time a turn runs inside a loop."""
        if not self.durable or self._conn is not None:
            return
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            self._conn = await aiosqlite.connect(str(self.session_dir / "checkpoints.sqlite"))
            saver = AsyncSqliteSaver(self._conn)
            await saver.setup()
            self._checkpointer = saver
            self.graph = build_graph(saver)
        except Exception as exc:  # noqa: BLE001 - fall back to in-memory checkpoints
            logger.warning("durable checkpointer unavailable (%s); using in-memory checkpoints", exc)
            self.durable = False

    async def aclose(self) -> None:
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    # ---- turns ----------------------------------------------------------------------------------------
    async def run_turn(self, message: str, *, interface: InterfaceState | dict | None = None) -> AsyncIterator[EventBase]:
        if self._running:
            raise AgentBusy("a turn is already running")
        if self.pending is not None:
            raise AgentBusy("a clarification is pending; call resume() or cancel() first")
        self.turn_index += 1
        turn_id = f"t{self.turn_index:04d}"
        self.current_turn_id = turn_id
        self.cancel_event.clear()
        iface = InterfaceState.model_validate(interface) if isinstance(interface, dict) else interface
        state = {
            "turn_id": turn_id,
            "turn_index": self.turn_index,
            "query": message,
            "interface": iface.model_dump() if iface else None,
            "mode": self.mode,
            "memory": self.memory.model_dump(),
            "plan": [],
            "step_index": 0,
            "answers": [],
            "handoffs": [],
            "assumptions": [],
            "pending": None,
            "current": {},
            "status": "running",
            "response": "",
            "figures": [],
            "error": None,
            "started_at": time.time(),
            "retries": 0,
        }
        if self.llm.calllog:
            self.llm.calllog.start_turn(turn_id, message, turn_index=self.turn_index)
        await self._ensure_graph()
        if self.ablation.driver == "loop":
            async for ev in self._drive(turn_id, iface, lambda ctx: self._run_loop(turn_id, message, ctx, None), TurnStarted(message=message, mode=self.mode, driver="loop", turn_index=self.turn_index)):
                yield ev
            return
        async for ev in self._drive(turn_id, iface, lambda ctx: self.graph.ainvoke(state, self._config(turn_id), context=ctx), TurnStarted(message=message, mode=self.mode, driver=self.ablation.driver, turn_index=self.turn_index)):
            yield ev

    async def resume(self, decision: Decision | dict | str) -> AsyncIterator[EventBase]:
        if self.pending is None or self.current_turn_id is None:
            raise AgentBusy("no pending clarification to resume")
        if isinstance(decision, str):
            decision = Decision(answer=decision)
        elif isinstance(decision, dict):
            decision = Decision.model_validate(decision)
        if decision.prompt_id and decision.prompt_id != self.pending.prompt_id:
            raise ValueError(f"decision is for prompt {decision.prompt_id}, pending is {self.pending.prompt_id}")
        decision.prompt_id = self.pending.prompt_id
        decision.kind = self.pending.kind
        if self.llm.calllog:
            self.llm.calllog.log_user_response(self.current_turn_id, self.pending.kind, self.pending.question or " | ".join(self.pending.questions), decision.as_text(), turn_index=self.turn_index)
        turn_id = self.current_turn_id
        payload = decision.model_dump()
        await self._ensure_graph()
        if self.ablation.driver == "loop":
            async for ev in self._drive(turn_id, None, lambda ctx: self._run_loop(turn_id, None, ctx, payload), None):
                yield ev
            return
        async for ev in self._drive(turn_id, None, lambda ctx: self.graph.ainvoke(Command(resume=payload), self._config(turn_id), context=ctx), None):
            yield ev

    async def _run_loop(self, turn_id: str, message: str | None, ctx: Context, resume: dict | None) -> dict:
        """One loop-mode turn (or its resumption); returns a graph-like result dict."""
        from stat_agent.agent.loop import LoopAgentRunner
        from stat_agent.memory.state import AssumptionRecord, StepHandoff, TurnRecord

        model = self.loop_model if self.loop_model is not None else self.llm.model
        if self._loop_runner is None:
            if self.settings.loop_backend == "deepagents":
                from stat_agent.extras.deepagents_driver import DeepAgentRunner

                self._loop_runner = DeepAgentRunner(self, model, subagents=self.settings.loop_subagents)
            else:
                self._loop_runner = LoopAgentRunner(self, model)
        self._loop_message = message if message is not None else getattr(self, "_loop_message", "")
        out = await self._loop_runner.run(turn_id, message, ctx, resume=resume)
        if out.get("interrupted"):
            pend = out.get("pending")
            return {"__interrupt__": [pend], "pending": pend.model_dump() if pend is not None else None, "status": "awaiting_input"}
        fin = out.get("finish") or {}
        ledger_touched = [e for e in self.ledger.log if e.producer and e.producer.turn_id == turn_id]
        outputs = sorted({a for e in ledger_touched for a in e.touched})
        handoff = StepHandoff(step_number=1, description=self._loop_message, skill=None, mode="loop", success=True, outputs=outputs, summary=fin.get("summary") or out.get("response", "")[:300], data_changes=fin.get("data_changes", ""), key_findings=list(fin.get("key_findings", [])), response=out.get("response", ""))
        record = TurnRecord(turn_id=turn_id, index=self.turn_index, query=self._loop_message, status="done", response_summary=handoff.summary[:600], data_changes=handoff.data_changes[:600], key_findings=handoff.key_findings[:6], handoffs=[handoff], assumptions=[AssumptionRecord.model_validate(a) for a in out.get("assumptions", [])], response=out.get("response", ""))
        if self.ablation.memory:
            self.memory.record_turn(record)
        usage = self.llm.meter.turn_totals(turn_id).as_dict()
        ctx.emit(TurnDone(state="done", response=out.get("response", ""), usage=usage, ledger_version=self.ledger.version, summary={"steps": 1, "status": "done", "driver": "loop"}))
        return {"memory": self.memory.model_dump(), "status": "done", "response": out.get("response", "")}

    def cancel(self) -> None:
        self.cancel_event.set()
        self.runtime.cancel()

    async def _drive(self, turn_id: str, interface: InterfaceState | None, factory, first_event: EventBase | None) -> AsyncIterator[EventBase]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        emitter = StampingEmitter(MultiEmitter([_QueueEmitter(loop, queue), *self._sinks]), turn_id=turn_id)
        self.llm.emitter = emitter
        ctx = self._context(emitter.emit, interface)
        self._running = True
        self.pending = None
        if first_event is not None:
            emitter.emit(first_event)
        task = asyncio.create_task(factory(ctx))
        try:
            while True:
                getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    yield getter.result()
                    continue
                getter.cancel()
                break
            while not queue.empty():
                yield queue.get_nowait()
            exc = task.exception()
            if exc is not None:
                logger.exception("turn %s failed", turn_id, exc_info=exc)
                err = TurnError(code=type(exc).__name__, message=str(exc)[:2000])
                emitter.emit(err)
                self.last_status = "error"
                while not queue.empty():
                    yield queue.get_nowait()
                return
            result = task.result()
            self._absorb(result)
            while not queue.empty():
                yield queue.get_nowait()
        finally:
            self._running = False

    def _absorb(self, result: dict) -> None:
        if result.get("__interrupt__"):
            self.pending = pending_interrupt(result)
            self.last_status = "awaiting_input"
            return
        self.pending = None
        if result.get("memory"):
            self.memory = ConversationMemory.model_validate(result["memory"])
        self.last_status = result.get("status", "done")
        self.last_response = result.get("response", "")

    # ---- convenience --------------------------------------------------------------------------------------
    async def chat(self, message: str, *, interface: InterfaceState | dict | None = None) -> str:
        """Run a turn (or answer the pending prompt) and return the response text."""
        events: list[EventBase] = []
        if self.pending is not None:
            async for ev in self.resume(message):
                events.append(ev)
        else:
            async for ev in self.run_turn(message, interface=interface):
                events.append(ev)
        if self.pending is not None:
            p = self.pending
            if p.kind == "skill_selection":
                opts = "\n".join(f"{o.get('index')}. {o.get('label')}" for o in p.options)
                return f"{p.question}\n{opts}"
            return p.question or "\n".join(p.questions)
        return self.last_response

    def chat_sync(self, message: str) -> str:
        return asyncio.run(self.chat(message))

    def save_state(self) -> None:
        if not self.session_dir:
            return
        import json as _json

        self.ledger.save(self.session_dir / "ledger.json")
        (self.session_dir / "memory.json").write_text(self.memory.model_dump_json(indent=1), encoding="utf-8")
        state = {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "current_turn_id": self.current_turn_id,
            "pending": self.pending.model_dump() if self.pending else None,
            "last_status": self.last_status,
            "mode": self.mode,
            "model": self.llm.model_name,
        }
        (self.session_dir / "agent_state.json").write_text(_json.dumps(state, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, session: Any, llm: LLMClient | str, session_dir: str | Path, **kwargs: Any) -> Agent:
        """Rebuild an agent from ``session_dir`` (memory, ledger, pending prompt, durable checkpoints)."""
        import json as _json

        session_dir = Path(session_dir)
        state = _json.loads((session_dir / "agent_state.json").read_text(encoding="utf-8")) if (session_dir / "agent_state.json").exists() else {}
        kwargs.setdefault("session_id", state.get("session_id"))
        kwargs.setdefault("mode", state.get("mode", "interactive"))
        agent = cls(session, llm, session_dir=session_dir, **kwargs)
        if (session_dir / "memory.json").exists():
            agent.memory = ConversationMemory.model_validate_json((session_dir / "memory.json").read_text(encoding="utf-8"))
        if (session_dir / "ledger.json").exists():
            agent.ledger = ProvenanceLedger.load(session_dir / "ledger.json")
            agent.mutator = SessionMutator(session, agent.ledger)
        agent.turn_index = int(state.get("turn_index", len(agent.memory.turns)))
        agent.current_turn_id = state.get("current_turn_id")
        agent.pending = Interrupt.model_validate(state["pending"]) if state.get("pending") else None
        agent.last_status = state.get("last_status")
        return agent

    # ---- provenance-driven actions ------------------------------------------------------------------------
    def rerun_plan(self, artifact_ids: list[str] | None = None) -> list[str]:
        """The queries that produced the stale artefacts, in production order."""
        ids = artifact_ids or self.ledger.stale_ids()
        steps: dict[tuple[str, int], tuple[int, str]] = {}
        for aid in ids:
            art = self.ledger.get(aid)
            if art is None or art.produced_by is None or not art.produced_by.turn_id:
                continue
            ref = art.produced_by
            key = (ref.turn_id, int(ref.step or 1))
            if key in steps:
                continue
            query = None
            for t in self.memory.turns:
                if t.turn_id == ref.turn_id:
                    for h in t.handoffs:
                        if h.step_number == key[1]:
                            query = h.complete_query or h.description or None
                    if query is None:
                        query = t.query
            steps[key] = (art.version, query or f"re-run the step that produced {aid}")
        ordered = sorted(steps.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        return [q for _, (_, q) in ordered]

    def rerun_message(self, artifact_ids: list[str] | None = None) -> str:
        queries = self.rerun_plan(artifact_ids)
        if not queries:
            return "Re-run every analysis whose inputs changed."
        if len(queries) == 1:
            return f"Re-run this analysis because its inputs changed: {queries[0]}"
        listed = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
        return "Re-run these analyses in order because their inputs changed:\n" + listed

    async def draft_skill(self, turn_id: str, step_number: int, extra_notes: str = "") -> dict:
        """Draft a SKILL.md for an unguarded step and write it to the user skills directory."""
        from stat_agent.skills.author import draft_from_handoff

        turn = next((t for t in self.memory.turns if t.turn_id == turn_id), None)
        if turn is None:
            raise ValueError(f"unknown turn {turn_id}")
        handoff = next((h for h in turn.handoffs if h.step_number == step_number), None)
        if handoff is None:
            raise ValueError(f"turn {turn_id} has no step {step_number}")
        return await draft_from_handoff(self.llm, self.settings, handoff, turn, self.session, extra_notes=extra_notes)

    def code_hash(self, code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()[:16]


__all__ = ["Agent", "AgentBusy"]
