"""Aggregate results with bootstrap confidence intervals."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stat_bench.result import RunResult


def load_results(root: str | Path) -> list[RunResult]:
    return [RunResult.load(p) for p in sorted(Path(root).rglob("result.json"))]


def bootstrap_ci(values: list[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def summarize(results: list[RunResult], by: tuple[str, ...] = ("system", "driver", "policy")) -> dict[str, Any]:
    groups: dict[tuple, list[RunResult]] = defaultdict(list)
    for r in results:
        key = tuple(str(getattr(r, k) if k != "ablation" else r.ablation.get("name", "A0")) for k in by)
        groups[key].append(r)
    table = []
    for key, rs in sorted(groups.items()):
        succ = [1.0 if r.success else 0.0 for r in rs]
        m, lo, hi = bootstrap_ci(succ)
        tok = [r.total_tokens for r in rs]
        sse_any = [1.0 if any((v or {}).get("flag") for k, v in (r.sse or {}).items() if k.startswith("sse")) else 0.0 for r in rs]
        sm, slo, shi = bootstrap_ci(sse_any)
        table.append({
            **dict(zip(by, key)),
            "n": len(rs),
            "n_queries": len({r.query_id for r in rs}),
            "success_rate": round(m, 3), "success_ci": [round(lo, 3), round(hi, 3)],
            "sse_rate": round(sm, 3), "sse_ci": [round(slo, 3), round(shi, 3)],
            "refusal_rate": round(sum(1 for r in rs if r.refusal.get("refused")) / len(rs), 3),
            "mean_tokens": round(float(np.mean(tok)), 0) if tok else 0,
            "mean_wall_s": round(float(np.mean([r.wall_s for r in rs])), 1),
            "by_category": _by_category(rs),
        })
    return {"groups": table}


def _by_category(rs: list[RunResult]) -> dict[str, Any]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rs:
        out[r.category].append(1.0 if r.success else 0.0)
    return {c: {"n": len(v), "success_rate": round(float(np.mean(v)), 3)} for c, v in sorted(out.items())}


def write_summary(results: list[RunResult], out: str | Path, by=("system", "driver", "policy")) -> Path:
    s = summarize(results, by)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=1), encoding="utf-8")
    md = p.with_suffix(".md")
    lines = ["| " + " | ".join(by) + " | n | success (95% CI) | SSE rate | refusals | tokens | wall s |", "|" + "---|" * (len(by) + 6)]
    for g in s["groups"]:
        lines.append("| " + " | ".join(str(g[k]) for k in by) + f" | {g['n']} | {g['success_rate']:.2f} ({g['success_ci'][0]:.2f}-{g['success_ci'][1]:.2f}) | {g['sse_rate']:.2f} | {g['refusal_rate']:.2f} | {int(g['mean_tokens'])} | {g['mean_wall_s']} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


__all__ = ["load_results", "bootstrap_ci", "summarize", "write_summary"]
