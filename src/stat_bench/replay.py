"""Replay report: compare STAT v2 decisions with STAT 0.1 traces (decision level)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stat_bench.trace import parse_session_dir, turn_decisions


def trace_decisions(log_dirs: list[Path]) -> list[dict[str, Any]]:
    """Flatten old traces into per-turn decision records."""
    out = []
    for d in log_dirs:
        for trace in parse_session_dir(d):
            decs = turn_decisions(trace)
            planner = next((x for x in decs if x.get("role") == "planner"), None)
            matcher = [x for x in decs if x.get("role") == "matcher"]
            verifier = [x for x in decs if x.get("role") == "verifier"]
            out.append({
                "path": trace.path,
                "query": trace.user_query,
                "backbone": trace.backbone,
                "planner": planner.get("decision") if planner else None,
                "matched": [m["decision"].get("matched") for m in matcher],
                "verdicts": [v["decision"].get("kind") for v in verifier],
                "clarifications": [x["kind"] for x in decs if x.get("role") == "user"],
            })
    return out


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Decision-level comparison of one old trace with one new run (same query)."""
    fields: dict[str, bool | None] = {}
    op, np_ = old.get("planner") or {}, new.get("planner") or {}
    fields["needs_clarification"] = (op.get("needs_clarification") == np_.get("needs_clarification")) if op and np_ else None
    fields["n_steps"] = (op.get("n_steps") == np_.get("n_steps")) if op and np_ else None
    fields["target_slices"] = (op.get("target_slices") == np_.get("target_slices")) if op and np_ else None
    om, nm = old.get("matched") or [], new.get("matched") or []
    fields["matcher_top1"] = ((om[0] or [None])[0] == (nm[0] or [None])[0]) if om and nm else None
    ov, nv = old.get("verdicts") or [], new.get("verdicts") or []
    fields["verifier_kind"] = (ov[0] == nv[0]) if ov and nv else None
    fields["clarification_sequence"] = (old.get("clarifications") == new.get("clarifications"))
    return {"query": old.get("query"), "fields": fields, "agree": all(v for v in fields.values() if v is not None)}


def report(old_records: list[dict], new_records: list[dict], expected_divergences: dict | None = None) -> dict[str, Any]:
    by_query = {r["query"]: r for r in new_records}
    comps = [compare(o, by_query[o["query"]]) for o in old_records if o["query"] in by_query]
    per_field: dict[str, dict[str, int]] = {}
    for c in comps:
        for f, v in c["fields"].items():
            d = per_field.setdefault(f, {"n": 0, "agree": 0})
            if v is None:
                continue
            d["n"] += 1
            d["agree"] += int(v)
    expl = expected_divergences or {}
    divergences = [c for c in comps if not c["agree"]]
    unexplained = [c for c in divergences if c["query"] not in expl]
    return {"n_compared": len(comps), "per_field": {f: {**d, "rate": round(d["agree"] / d["n"], 3) if d["n"] else None} for f, d in per_field.items()}, "divergences": divergences, "unexplained": len(unexplained), "gate": "pass" if not unexplained else "fail"}


def save_report(rep: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    lines = [f"# Replay report ({rep['n_compared']} traces) - gate: {rep['gate']}", "", "| field | n | agree | rate |", "|---|---|---|---|"]
    for f, d in rep["per_field"].items():
        lines.append(f"| {f} | {d['n']} | {d['agree']} | {d['rate']} |")
    if rep["divergences"]:
        lines += ["", "## Divergences", ""] + [f"- {c['query'][:100]}: " + ", ".join(k for k, v in c["fields"].items() if v is False) for c in rep["divergences"]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["trace_decisions", "compare", "report", "save_report"]
