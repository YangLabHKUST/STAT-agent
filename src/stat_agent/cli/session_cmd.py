"""``stat-agent session ...``: save, load and report."""

from __future__ import annotations

import asyncio
from pathlib import Path


def add_session_subparser(subparsers) -> None:
    p = subparsers.add_parser("session", help="save/load/report sessions")
    sp = p.add_subparsers(dest="session_cmd", required=True)
    r = sp.add_parser("report", help="render a Markdown/HTML report from a session log directory")
    r.add_argument("session_dir")
    r.add_argument("--format", choices=["md", "html"], default="md")
    r.add_argument("--out")
    r.set_defaults(func=cmd_report)
    lo = sp.add_parser("load", help="load a saved session and run one message (headless)")
    lo.add_argument("saved_dir")
    lo.add_argument("--logs", help="session log dir to restore agent state from")
    lo.add_argument("-m", "--message", required=True)
    lo.add_argument("--model")
    lo.set_defaults(func=cmd_load)


def cmd_report(args) -> int:
    from stat_agent.ledger import ProvenanceLedger
    from stat_agent.memory.state import ConversationMemory

    d = Path(args.session_dir)
    memory = ConversationMemory.model_validate_json((d / "memory.json").read_text()) if (d / "memory.json").exists() else ConversationMemory()
    ledger = ProvenanceLedger.load(d / "ledger.json") if (d / "ledger.json").exists() else ProvenanceLedger()

    class _Stub:  # minimal agent-like object for the renderer
        pass

    stub = _Stub()
    stub.memory, stub.ledger, stub.mode = memory, ledger, "n/a"
    stub.llm = type("L", (), {"model_name": "n/a"})()
    session = type("S", (), {"name": d.name, "iter_slices": lambda self: [], "roi_manager": type("R", (), {"rois": {}})()})()
    from stat_agent.logging.report import render_report

    text = render_report(stub, session, fmt=args.format)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_load(args) -> int:
    from stat_agent.agent import Agent
    from stat_agent.config import Settings
    from stat_agent.io.session_store import load_session
    from stat_agent.llm import LLMClient, make_spec

    settings = Settings.from_env()
    session, _ = load_session(args.saved_dir)
    llm = LLMClient(spec=make_spec(args.model or settings.default_model))
    logs = Path(args.logs) if args.logs else Path(args.saved_dir)
    agent = Agent.load(session, llm, logs, settings=settings) if (logs / "agent_state.json").exists() else Agent(session, llm, settings=settings, session_dir=logs)
    print(asyncio.run(agent.chat(args.message)))
    agent.save_state()
    return 0


__all__ = ["add_session_subparser"]
