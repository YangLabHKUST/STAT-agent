"""Headless chat: REPL or one-shot, printing the typed event stream."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from stat_agent.agent import Agent, Decision
from stat_agent.config import Settings
from stat_agent.core import SimpleSession
from stat_agent.llm import LLMClient, make_spec
from stat_agent.logging.notebook import NotebookLogger


def _print_event(ev, verbose: bool = False) -> None:
    t = ev.type
    if t == "text":
        prefix = "[analysis] " if ev.role == "analysis" else ""
        print(f"\n{prefix}{ev.text}")
    elif t == "text_delta":
        print(ev.delta, end="", flush=True)
    elif t == "exec_output":
        for ln in ev.lines:
            print(("  ! " if ln.stream == "stderr" else "  | ") + ln.text)
    elif t == "code_block_started":
        print(f"  [running code block, attempt {ev.attempt}]")
    elif t == "prompt":
        if ev.options:
            print(f"\n? {ev.question}")
            for o in ev.options:
                print(f"   {o.index}. {o.label}")
        elif ev.questions:
            print("\n? " + "\n? ".join(q.text for q in ev.questions))
        else:
            print(f"\n? {ev.question}")
    elif t == "assumption":
        print(f"  [assumed] {ev.question} -> {ev.chosen} ({ev.rationale})")
    elif t in ("advice", "refusal", "warning", "error", "turn_error"):
        print(f"\n[{t}] {getattr(ev, 'message', None) or getattr(ev, 'reason', '')}")
    elif t == "phase" and verbose:
        print(f"  ({ev.phase})")
    elif t == "provenance_update" and verbose:
        print(f"  [provenance v{ev.ledger_version}] {[c.artifact.id for c in ev.changes]} stale={ev.stale_now}")
    elif t == "turn_done":
        u = ev.usage or {}
        print(f"\n--- done ({ev.state}; {u.get('llm_calls', 0)} calls, {u.get('total_tokens', 0)} tokens) ---")


def build_agent(dataset_dir: str, model: str | None, mode: str | None, session_dir: Path | None = None, **kw) -> Agent:
    settings = Settings.from_env()
    session = SimpleSession(name=Path(dataset_dir).name)
    session.load_dataset(dataset_dir)
    llm = LLMClient(spec=make_spec(model or settings.default_model))
    session_dir = session_dir or settings.resolved_logs_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}"
    notebook = NotebookLogger(session_dir)
    notebook.initialize_notebook(dataset_dir, session.name)
    return Agent(session, llm, settings=settings, mode=mode or settings.mode, session_dir=session_dir, notebook=notebook, **kw)


async def _run(agent: Agent, message: str, verbose: bool) -> None:
    it = agent.resume(Decision(answer=message)) if agent.is_waiting else agent.run_turn(message)
    async for ev in it:
        _print_event(ev, verbose)


def run_chat(args) -> int:
    agent = build_agent(args.dataset_dir, args.model, args.mode)
    print(f"Loaded {agent.session} | model {agent.llm.model_name} | mode {agent.mode} | logs {agent.session_dir}")
    if args.message:
        asyncio.run(_run(agent, args.message, verbose=True))
        agent.save_state()
        return 0
    print("Type a request (empty line or Ctrl-D to quit).")
    while True:
        try:
            line = input("\n> " if not agent.is_waiting else "\n(answer) > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        asyncio.run(_run(agent, line, verbose=False))
        agent.save_state()
    return 0


__all__ = ["run_chat", "build_agent"]
