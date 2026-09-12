"""Run the web app with a scripted LLM and a synthetic dataset (no keys needed).

    python scripts/dev_server_fake.py --port 8899
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

from stat_agent.config import Settings  # noqa: E402
from stat_agent.skills import SkillRegistry  # noqa: E402
from stat_agent.web.server import create_app  # noqa: E402
from stat_agent.web.state import WebConfig  # noqa: E402
from tests.fixtures.sessions import make_session, write_dataset  # noqa: E402
from tests.graph.conftest import SKILLS_MINI, default_responder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    ds = args.dataset or str(write_dataset(make_session("S1_cell_gene_noCT"), Path(tempfile.mkdtemp()) / "ds"))
    os.environ.setdefault("STAT_LOGS_DIR", tempfile.mkdtemp())
    reg = SkillRegistry([(SKILLS_MINI, "test")])
    reg.load()
    app = create_app(WebConfig(llm_responder=default_responder, port=args.port), settings=Settings.from_env(), registry=reg)
    print(f"DATASET={ds}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
