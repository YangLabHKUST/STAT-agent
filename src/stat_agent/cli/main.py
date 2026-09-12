"""``stat-agent`` command line entry point."""

from __future__ import annotations

import argparse
import sys

from stat_agent import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stat-agent", description="STAT: spatial transcriptomics analytical agent")
    parser.add_argument("--version", action="version", version=f"stat-agent {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    from stat_agent.skills.cli import add_skill_subparser

    add_skill_subparser(sub)

    from stat_agent.cli.session_cmd import add_session_subparser

    add_session_subparser(sub)

    web = sub.add_parser("web", help="start the web interface")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8889)
    web.add_argument("--jupyter-port", type=int, default=8890)
    web.add_argument("--debug", action="store_true")
    web.set_defaults(func=_cmd_web)

    chat = sub.add_parser("chat", help="headless chat with a dataset (REPL or one-shot)")
    chat.add_argument("dataset_dir")
    chat.add_argument("-m", "--message", help="single message (omit for a REPL)")
    chat.add_argument("--model")
    chat.add_argument("--mode", choices=["interactive", "auto"], default=None)
    chat.add_argument("--driver", choices=["staged", "loop"], default=None)
    chat.set_defaults(func=_cmd_chat)
    return parser


def _cmd_web(args) -> int:
    from stat_agent.web.__main__ import run

    return run(host=args.host, port=args.port, jupyter_port=args.jupyter_port, debug=args.debug)


def _cmd_chat(args) -> int:
    from stat_agent.cli.chat import run_chat

    return run_chat(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
