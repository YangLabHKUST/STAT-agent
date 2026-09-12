"""``python -m stat_agent.web`` / ``stat-web``: run the web interface with uvicorn."""

from __future__ import annotations

import argparse
import logging
import os


def run(host: str = "127.0.0.1", port: int = 8889, jupyter_port: int = 8890, debug: bool = False, jupyter_root: str | None = None) -> int:
    import uvicorn

    from stat_agent.web.server import create_app
    from stat_agent.web.state import WebConfig

    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from pathlib import Path

    root = jupyter_root or os.environ.get("STAT_JUPYTER_ROOT")
    app = create_app(WebConfig(host=host, port=port, jupyter_port=jupyter_port, jupyter_root=Path(root) if root else None, debug=debug, demo_mode=os.environ.get("STAT_DEMO_MODE") == "1"))
    uvicorn.run(app, host=host, port=port, workers=1, proxy_headers=True, log_level="debug" if debug else "info")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="STAT web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8889")))
    parser.add_argument("--jupyter-port", type=int, default=8890)
    parser.add_argument("--jupyter-root", help="Jupyter's notebook-dir, for notebook links (default: STAT_JUPYTER_ROOT or the logs dir)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return run(args.host, args.port, args.jupyter_port, args.debug, jupyter_root=args.jupyter_root)


if __name__ == "__main__":
    raise SystemExit(main())
