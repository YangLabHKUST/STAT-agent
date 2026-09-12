from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient
from tests.fixtures.sessions import make_session, write_dataset
from tests.graph.conftest import SKILLS_MINI, default_responder

from stat_agent.config import Settings
from stat_agent.skills import SkillRegistry
from stat_agent.web.server import create_app
from stat_agent.web.state import WebConfig


@pytest.fixture
def dataset_dir(tmp_path):
    return write_dataset(make_session("S1_cell_gene_noCT"), tmp_path / "ds")


@pytest.fixture
def client(tmp_path, dataset_dir, monkeypatch):
    monkeypatch.setenv("STAT_LOGS_DIR", str(tmp_path / "logs"))
    reg = SkillRegistry([(SKILLS_MINI, "test")])
    reg.load()
    app = create_app(WebConfig(llm_responder=default_responder, ping_s=1, lock_timeout_s=0.2), settings=Settings.from_env(), registry=reg)
    with TestClient(app) as c:
        yield c


def sse_frames(text: str) -> list[dict]:
    """Parse raw SSE text into [{'event', 'id', 'data'}]."""
    frames = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        ev = {"event": None, "id": None, "data": None}
        for line in block.splitlines():
            if line.startswith("event:"):
                ev["event"] = line[6:].strip()
            elif line.startswith("id:"):
                ev["id"] = int(line[3:].strip())
            elif line.startswith("data:"):
                ev["data"] = json.loads(line[5:].strip() or "{}")
        if ev["event"] or ev["data"]:
            frames.append(ev)
    return frames


def init_session(client, dataset_dir, mode="interactive"):
    r = client.post("/api/session", json={"dataset_dir": str(dataset_dir), "session_name": "t", "model": "scripted/model", "provider": "openai", "mode": mode})
    assert r.status_code == 200, r.text
    return r.json()
