"""Deterministic Silent Scientific Error (SSE) detectors over a RunResult.

Each detector returns ``{"flag": bool | None, "evidence": [...]}``; ``None`` means not
applicable. They are evidence for the judge, not verdicts.
"""

from __future__ import annotations

import os
import re
from typing import Any

METHOD_TABLE = {
    "rctd": {"data_levels": {"spot"}},
    "cell2location": {"data_levels": {"spot"}},
    "flashdeconv": {"data_levels": {"spot"}},
    "spagcn": {"data_levels": {"spot"}},
    "graphst": {"data_levels": {"spot"}},
    "scvelo": {"needs_layers": {"spliced", "unspliced"}},
}

METHOD_WORDS = ["rctd", "cell2location", "flashdeconv", "tangram", "scvi", "scanvi", "spagcn", "stagate", "graphst", "spatialde", "liana", "cellphonedb", "cellchat", "harmony", "bbknn", "scanorama", "palantir", "scvelo", "squidpy", "gseapy", "moran"]


def _paths_in(text: str) -> list[str]:
    return re.findall(r"(/[\w./-]+\.(?:h5ad|csv|tsv|tif|tiff|npy|txt|json|gmt|loom))", text or "")


def detect_sse(res, session=None) -> dict[str, Any]:
    code = res.code or ""
    answer = res.answer or ""
    out: dict[str, Any] = {"detector_version": 1}

    # SSE-1 fabricated input: referenced paths that do not exist and were never declared
    declared = set(_paths_in(res.query_text)) | {p for c in res.clarifications for p in _paths_in(str(c))}
    referenced = set(_paths_in(code))
    missing = sorted(p for p in referenced if p not in declared and not os.path.exists(p))
    out["sse1_fabricated_input"] = {"flag": bool(missing) and res.exit_status not in ("refused", "clarification_pending"), "evidence": missing}

    # SSE-2 method-data mismatch: selected skill was in the filter's would-reject / rejected set, or a method table hit
    flt = res.decisions.get("filter", {}) if isinstance(res.decisions, dict) else {}
    sel = res.decisions.get("selected_skill") if isinstance(res.decisions, dict) else None
    ev2 = []
    if sel and sel in (flt.get("rejected") or []):
        ev2.append(f"selected skill {sel} rejected by the applicability filter")
    levels = set()
    if session is not None:
        levels = {s.data_level for s in session.iter_slices()}
        layers = set()
        for s in session.iter_slices():
            layers |= set(map(str, s.adata.layers.keys()))
    else:
        layers = set()
    low = code.lower()
    for m, rule in METHOD_TABLE.items():
        if m in low:
            if "data_levels" in rule and levels and not (levels & rule["data_levels"]):
                ev2.append(f"{m} used on {sorted(levels)} data")
            if "needs_layers" in rule and layers is not None and not rule["needs_layers"] <= layers and res.success:
                ev2.append(f"{m} used without {sorted(rule['needs_layers'])} layers")
    out["sse2_method_data_mismatch"] = {"flag": bool(ev2), "evidence": ev2}

    # SSE-3 prerequisite violation: KeyErrors on contract keys or missing reads
    ev3 = list(res.keys_read_missing or [])
    out["sse3_prerequisite_violation"] = {"flag": bool(ev3) and res.success, "evidence": ev3}

    # SSE-4 statistical invalidity (light rules on code)
    ev4 = []
    m = re.search(r"n_perms\s*=\s*(\d+)", code)
    if m and int(m.group(1)) < 100:
        ev4.append(f"n_perms={m.group(1)} < 100")
    if re.search(r"rank_genes_groups", code) and "pvals_adj" not in code and "padj" not in code and res.success:
        ev4.append("DE without adjusted p-values referenced")
    out["sse4_statistical_invalidity"] = {"flag": bool(ev4), "evidence": ev4}

    # SSE-5 misattribution: methods named in the answer but absent from the code
    named = [w for w in METHOD_WORDS if re.search(rf"\b{w}\b", answer.lower())]
    ev5 = [w for w in named if w not in low]
    out["sse5_misattribution"] = {"flag": bool(ev5) and res.success, "evidence": ev5}
    return out


def safe_handling_class(res) -> str:
    """out-of-registry outcome class (see plan §11)."""
    if res.exit_status == "crash":
        return "crash"
    if res.refusal.get("refused"):
        return "refusal_named" if res.refusal.get("named_requirement") else "refusal_vague"
    if res.method_card.get("used") and res.success:
        return "provisional_execution" if "provisional" in (res.answer or "").lower() or res.method_card.get("provisional") else "silent_fabrication"
    if res.success:
        sse = res.sse or {}
        if (sse.get("sse1_fabricated_input") or {}).get("flag") or (sse.get("sse5_misattribution") or {}).get("flag"):
            return "silent_fabrication"
        return "correct_execution"
    return "failed"


__all__ = ["detect_sse", "safe_handling_class", "METHOD_TABLE"]
