"""AppState: the one place that owns the session, the agent, the event log and the turns."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from stat_agent.agent import Agent
from stat_agent.agent.policy import make_policy
from stat_agent.config import Settings
from stat_agent.core import SimpleSession
from stat_agent.events.catalogue import SessionReset
from stat_agent.ledger import SessionMutator
from stat_agent.llm import LLMClient, make_spec
from stat_agent.logging.notebook import NotebookLogger
from stat_agent.skills import SkillRegistry
from stat_agent.web.events import EventLog

logger = logging.getLogger(__name__)


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8889
    jupyter_port: int = 8890
    jupyter_root: Path | None = None  # Jupyter's notebook-dir; notebook links are made relative to it (default: the logs dir)
    debug: bool = False
    demo_mode: bool = False
    lock_timeout_s: float = 2.0
    ping_s: int = 15
    static_dir: Path | None = None
    llm_responder: Any = None  # tests: scripted responder instead of a real model


@dataclass
class AppState:
    settings: Settings
    config: WebConfig
    registry: SkillRegistry
    generation: int = 0
    session: SimpleSession | None = None
    agent: Agent | None = None
    mutator: SessionMutator | None = None
    events: EventLog = field(default_factory=EventLog)
    session_dir: Path | None = None
    llm_config: dict = field(default_factory=dict)
    turns: Any = None  # TurnManager, set by server
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def ready(self) -> bool:
        return self.session is not None and self.agent is not None

    def require_session(self) -> SimpleSession:
        from stat_agent.web.errors import ApiError

        if self.session is None or not self.session.has_data:
            raise ApiError("no_session", "Session not initialized", 400)
        return self.session

    def require_agent(self) -> Agent:
        from stat_agent.web.errors import ApiError

        if self.agent is None:
            raise ApiError("no_agent", "Agent not initialized", 400)
        return self.agent

    def summary_payload(self) -> dict:
        turn = self.turns.active_snapshot() if self.turns else None
        return {
            "success": True,
            "summary": self.session.get_frontend_summary() if self.session else None,
            "agent_active": self.agent is not None,
            "chat_busy": bool(turn and turn.get("state") in ("running", "awaiting_input")),
            "demo_mode": self.config.demo_mode,
            "generation": self.generation,
            "ledger_version": self.agent.ledger.version if self.agent else 0,
            "active_turn": turn,
            "mode": self.agent.mode if self.agent else self.settings.mode,
            "model": self.agent.llm.model_name if self.agent else None,
        }

    # ---- lifecycle ------------------------------------------------------------------------------------------
    def create_session(self, dataset_dir: str | None, *, session_name: str | None = None, provider: str | None = None, model: str | None = None, api_key: str | None = None, base_url: str | None = None, mode: str | None = None, session_dir_to_load: str | None = None) -> None:
        with self._lock:
            # validate first: a mistyped path must not destroy the session the user is working in
            if session_dir_to_load:
                if not (Path(session_dir_to_load) / "manifest.json").exists():
                    raise FileNotFoundError(f"Saved session not found: {session_dir_to_load}")
            elif not dataset_dir or not Path(dataset_dir).is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
            elif not any(Path(dataset_dir).glob("*.h5ad")):
                raise FileNotFoundError(f"No .h5ad files in dataset directory: {dataset_dir}")
            self._teardown()
            saved = None
            if session_dir_to_load:
                from stat_agent.io.session_store import load_session

                session, saved = load_session(session_dir_to_load)
                name = session_name or session.name
                session.name = name
            else:
                name = session_name or Path(dataset_dir).name or "session"
                session = SimpleSession(name=name)
                session.load_dataset(dataset_dir)
            self.session_dir = self.settings.resolved_logs_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}"
            self.session_dir.mkdir(parents=True, exist_ok=True)
            session.workdir = self.session_dir
            model_id = model or self.settings.default_model
            if self.config.llm_responder is not None:
                llm = LLMClient(None, model_name=model_id, responder=self.config.llm_responder)
            else:
                llm = LLMClient(spec=make_spec(model_id, provider=provider, api_key=api_key, base_url=base_url))
            self.llm_config = {"provider": provider, "model": model_id, "base_url": base_url}
            notebook = NotebookLogger(self.session_dir)
            notebook.initialize_notebook(dataset_dir or session_dir_to_load, name)
            self.generation += 1
            self.events.publish(SessionReset(generation=self.generation))  # pages bound to the old log reload instead of listening to a dead log
            self.events = EventLog(self.session_dir)
            self.events.generation = self.generation
            try:
                self.events.bind_loop(asyncio.get_running_loop())
            except RuntimeError:
                pass
            mode = mode or self.settings.mode
            if saved is not None and (Path(session_dir_to_load) / "memory.json").exists():
                agent = Agent.load(session, llm, session_dir_to_load, logs_dir=self.session_dir, registry=self.registry, settings=self.settings, notebook=notebook, stream_text=True)
                agent.mode = mode
                agent.policy = make_policy(mode)
            else:
                agent = Agent(session, llm, registry=self.registry, settings=self.settings, mode=mode, policy=make_policy(mode), session_dir=self.session_dir, notebook=notebook, stream_text=True, session_id=f"{name}-g{self.generation}")
            self.session, self.agent = session, agent
            self.mutator = SessionMutator(session, agent.ledger, emitter=_LogEmitter(self.events), lock_timeout_s=self.config.lock_timeout_s)
            if self.turns is not None:
                self.turns.reset()
            logger.info("session %s created (generation %d)", name, self.generation)

    def reset(self) -> None:
        with self._lock:
            self._teardown()
            self.generation += 1
            self.events.publish(SessionReset(generation=self.generation))
            self.events = EventLog(None)
            self.events.generation = self.generation
            if self.turns is not None:
                self.turns.reset()

    def _teardown(self) -> None:
        if self.agent is not None:
            try:
                self.agent.cancel()
                self.agent.save_state()
            except Exception:  # noqa: BLE001
                pass
        self.session, self.agent, self.mutator = None, None, None


class _LogEmitter:
    def __init__(self, log: EventLog) -> None:
        self._log = log

    def emit(self, event) -> None:
        self._log.publish(event)


__all__ = ["AppState", "WebConfig"]
