"""``result.json`` v2: a superset of the STAT 0.1 keys, one per (query, system, seed)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from stat_agent import __version__
from stat_agent.events.encoding import dumps


def software_info() -> dict[str, Any]:
    info: dict[str, Any] = {"stat_version": __version__, "python": sys.version.split()[0], "platform": platform.platform()}
    try:
        info["git_sha"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        info["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:  # noqa: BLE001
        info["git_sha"] = None
    for pkg in ("scanpy", "anndata", "squidpy", "langgraph", "langchain"):
        try:
            import importlib.metadata as md

            info[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            info[pkg] = None
    return info


class RunResult(BaseModel):
    schema_version: int = 2
    # v1 keys
    query_id: str
    query_text: str
    task: str
    dataset: str | None = None
    difficulty: str = "clear"
    has_ground_truth: bool = False
    system: str = "stat"
    model: str = ""
    answer: str = ""
    response: str = ""
    code: str = ""
    stdout: str = ""
    stderr: str = ""
    wall_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    success: bool = False
    output_h5ad: str | None = None
    run_id: str = ""
    # v2 keys
    suite_id: str = ""
    category: str = ""
    driver: str = "staged"
    ablation: dict[str, Any] = Field(default_factory=dict)
    policy: str = "scripted"
    backbone: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    seed_index: int = 0
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    timeout_s: int = 600
    timed_out: bool = False
    exit_status: str = "ok"  # ok | partial | failed | refused | clarification_pending | timeout | budget | crash
    usage_calls: list[dict[str, Any]] = Field(default_factory=list)
    events_path: str | None = None
    clarifications: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    refusal: dict[str, Any] = Field(default_factory=dict)
    method_card: dict[str, Any] = Field(default_factory=dict)
    decisions: dict[str, Any] = Field(default_factory=dict)
    artifacts_produced: list[str] = Field(default_factory=list)
    keys_read_missing: list[str] = Field(default_factory=list)
    paths_referenced: list[dict[str, Any]] = Field(default_factory=list)
    provenance_snapshot_hash: str | None = None
    sse: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    software: dict[str, Any] = Field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "result.json.tmp"
        tmp.write_text(dumps(self.model_dump(mode="json"), indent=1), encoding="utf-8")
        tmp.replace(d / "result.json")
        if self.code:
            (d / "code.py").write_text(self.code, encoding="utf-8")
        return d / "result.json"

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def cell_dir(out: Path, system: str, driver: str, ablation: str, policy: str, backbone: str, qid: str, seed_index: int) -> Path:
    slug = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in backbone)
    return out / f"{system}__{driver}__{ablation}__{policy}" / slug / qid / f"seed_{seed_index}"


__all__ = ["RunResult", "software_info", "cell_dir"]
