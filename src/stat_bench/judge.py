"""LLM judge (ported from the STAT 0.1 protocol): all systems anonymised in one prompt per query.

Requires real model credentials; outputs success / quality / rank per system plus SSE adjudication.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from stat_agent.llm import LLMClient, make_spec
from stat_bench.result import RunResult

JUDGE_SYSTEM = """You are an expert spatial transcriptomics analyst acting as an impartial judge. You will see one analysis request and the outputs of several anonymous systems (A, B, C, ...). Judge each system on (1) task success (yes/no) according to the criteria, (2) quality 1-5 for successful systems, (3) a rank among successful systems (1 = best, ties allowed), and (4) for each listed silent-error flag, whether the evidence confirms it (confirmed/refuted/unclear). Judge only from the evidence shown. Respond with JSON only."""

CRITERIA = {
    "T1": "Produced a celltype/annotation column with biologically meaningful labels (not just numeric cluster IDs)",
    "T2": "Assigned spatial domain labels to cells/spots",
    "T3": "Identified genes with spatial statistics (Moran's I, SpatialDE q-values), not just expression variance",
    "T4": "Identified ligand-receptor interactions using a proper database (CellPhoneDB, CellChat, LIANA)",
    "T5": "Performed spatial co-localization or enrichment analysis with statistical testing",
    "T6": "Estimated cell type proportions per spot using the provided reference",
    "T7": "Identified spatial niches/microenvironments based on local cell type composition",
    "T8": "Identified differentially expressed genes between specified groups with statistical testing",
    "T9": "Performed formal statistical enrichment analysis (not manual gene categorization)",
    "T10": "Loaded ALL input files and applied batch correction (Harmony, BBKNN, scVI)",
    "T11": "Provided a meaningful summary or analysis of the dataset",
    "multi-step": "ALL steps completed - partial completion is judged as failure",
    "infeasible": "Attempted a reasonable alternative approach OR correctly explained the limitation",
    "out-of-scope": "Correctly explained why the task cannot be done with the available data/tools",
    "missing-prereq": "Identified the missing prerequisite OR performed it as a preliminary step",
}


class SystemVerdict(BaseModel):
    system: str
    success: bool
    quality: int | None = None
    rank: int | None = None
    sse: dict[str, str] = Field(default_factory=dict)
    justification: str = ""


class JudgeOutput(BaseModel):
    verdicts: list[SystemVerdict]


def _block(label: str, r: RunResult) -> str:
    sse_flags = {k: v for k, v in (r.sse or {}).items() if isinstance(v, dict) and v.get("flag")}
    return (
        f"### System {label}\n"
        f"Response (truncated):\n{(r.response or '')[:2000]}\n\n"
        f"Code (truncated):\n```python\n{(r.code or '')[:1500]}\n```\n"
        f"stdout (tail):\n{(r.stdout or '')[-500:]}\n"
        f"Artefacts produced: {', '.join(r.artifacts_produced) or 'none'}\n"
        f"Exit status: {r.exit_status}; refusal: {r.refusal or 'none'}; assumptions: {len(r.assumptions)}\n"
        f"Silent-error flags (evidence, not verdicts): {json.dumps(sse_flags) if sse_flags else 'none'}\n"
    )


def build_prompt(query_text: str, category: str, dataset_desc: str, results: list[RunResult], labels: list[str]) -> str:
    blocks = "\n".join(_block(lbl, r) for lbl, r in zip(labels, results))
    return (
        f"## Request\n{query_text}\n\n## Dataset\n{dataset_desc}\n\n## Success criterion ({category})\n{CRITERIA.get(category, CRITERIA['T11'])}\n\n"
        f"## Outputs\n{blocks}\n\n"
        "## Answer format (JSON only)\n"
        '{"verdicts": [{"system": "A", "success": true, "quality": 4, "rank": 1, "sse": {"sse1_fabricated_input": "refuted"}, "justification": "..."}]}\n'
    )


async def judge_query(llm: LLMClient, query_text: str, category: str, dataset_desc: str, results: list[RunResult], seed: int = 0) -> list[dict[str, Any]]:
    order = list(range(len(results)))
    random.Random(seed).shuffle(order)
    labels = [chr(ord("A") + i) for i in range(len(results))]
    shuffled = [results[i] for i in order]
    comp = await llm.complete([("system", JUDGE_SYSTEM), ("user", build_prompt(query_text, category, dataset_desc, shuffled, labels))], role="judge", schema=JudgeOutput)
    out = []
    for v in comp.value.verdicts:
        idx = labels.index(v.system) if v.system in labels else None
        if idx is None:
            continue
        r = shuffled[idx]
        out.append({"system": r.system, "driver": r.driver, "ablation": r.ablation, "run_id": r.run_id, "query_id": r.query_id, **v.model_dump(exclude={"system"})})
    return out


def judge_results(results_root: str | Path, judge_model: str, dataset_descriptions: dict[str, str] | None = None, out: str | Path | None = None, seed: int = 0) -> list[dict[str, Any]]:
    from stat_bench.summarize import load_results

    results = load_results(results_root)
    by_query: dict[str, list[RunResult]] = {}
    for r in results:
        by_query.setdefault(r.query_id, []).append(r)
    llm = LLMClient(spec=make_spec(judge_model))
    verdicts: list[dict[str, Any]] = []

    async def _all():
        for _qid, rs in sorted(by_query.items()):
            desc = (dataset_descriptions or {}).get(rs[0].dataset or "", rs[0].dataset or "")
            verdicts.extend(await judge_query(llm, rs[0].query_text, rs[0].category, desc, rs, seed=seed))

    asyncio.run(_all())
    if out:
        Path(out).write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    return verdicts


__all__ = ["judge_query", "judge_results", "build_prompt", "CRITERIA"]
