"""stat-bench run: plan cells, run them (in-process, sequential), skip completed, write a manifest."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from stat_bench.result import RunResult, cell_dir
from stat_bench.suite import Suite
from stat_bench.systems.base import Cell


def derive_seed(suite_id: str, qid: str, system: str, driver: str, ablation: str, policy: str, seed_index: int) -> int:
    h = hashlib.blake2b(f"{suite_id}|{qid}|{system}|{driver}|{ablation}|{policy}|{seed_index}".encode(), digest_size=4).hexdigest()
    return int(h, 16)


def plan_cells(suite: Suite, *, system: str, driver: str, ablation: str, policy: str, backbone: str, seeds: int, out: Path, data_root: str | None, ids=None, categories=None, tags=None, execute: bool = True, extra: dict | None = None) -> list[Cell]:
    cells = []
    for q in suite.select(ids, categories, tags):
        for k in range(seeds):
            cells.append(
                Cell(
                    query=q,
                    suite_id=suite.suite_id,
                    data_root=data_root,
                    out_dir=str(cell_dir(out, system, driver, ablation, policy, backbone, q.id, k)),
                    seed=derive_seed(suite.suite_id, q.id, system, driver, ablation, policy, k),
                    seed_index=k,
                    backbone=backbone,
                    driver=driver,
                    ablation=ablation,
                    policy=policy,
                    execute=execute,
                    extra=dict(extra or {}),
                )
            )
    return cells


async def run_cells(system, cells: list[Cell], *, resume: bool = True, retry_failed: bool = False, manifest_path: Path | None = None, on_result=None) -> list[RunResult]:
    results: list[RunResult] = []
    manifest: dict[str, Any] = {"planned": len(cells), "cells": {}, "started_at": time.time()}
    for cell in cells:
        d = Path(cell.out_dir)
        rjson = d / "result.json"
        key = f"{cell.query.id}/seed_{cell.seed_index}"
        if resume and rjson.exists():
            prev = RunResult.load(rjson)
            if not (retry_failed and prev.exit_status in ("failed", "timeout", "crash")):
                results.append(prev)
                manifest["cells"][key] = {"status": prev.exit_status, "skipped": True}
                continue
        manifest["cells"][key] = {"status": "running"}
        res = await system.run(cell)
        results.append(res)
        manifest["cells"][key] = {"status": res.exit_status, "wall_s": round(res.wall_s, 1)}
        if on_result:
            on_result(res)
        if manifest_path:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    manifest["finished_at"] = time.time()
    if manifest_path:
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return results


def run(system, cells: list[Cell], **kw) -> list[RunResult]:
    return asyncio.run(run_cells(system, cells, **kw))


__all__ = ["plan_cells", "run_cells", "run", "derive_seed"]
