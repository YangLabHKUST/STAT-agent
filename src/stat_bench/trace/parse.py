"""Parser for per-turn markdown prompt logs.

Format (identical in STAT 0.1.x ``PromptLogger`` and STAT 2 ``CallLogger``)::

    # Conversation Turn N - <timestamp>
    ## User Query
    ```...```
    ---
    ## LLM Call k: <role>
    ### Request Metadata
    - **Model:** ...
    ### Full Prompt
    ```...(may contain nested fences and '## ' headings)...```
    ### Response
    ```...```
    ---
    ## User Response (<Kind> Clarification)
    **Question:** ```...```  **User Response:** ```...```
    ---
    ## Summary

Sections are delimited by the *top-level markers* only, so nested fences inside prompts
are harmless.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stat_agent.llm.structured import extract_json

ROLE_ALIASES = {
    "query_planner": "planner",
    "semantic_matcher": "matcher",
    "skill_matching": "matcher",
    "skill_verifier": "verifier",
    "main_agent": "generator",
    "analyze_execution_results": "analyzer",
    "error_reflection": "reflector",
    "code_validation": "validator",
}

_CALL_RE = re.compile(r"^## LLM Call (\d+): (\S+)\s*$")
_USER_RESP_RE = re.compile(r"^## User Response \((.+?) Clarification\)\s*$")
_META_RE = re.compile(r"^- \*\*(.+?):\*\* (.*)$")
_TITLE_RE = re.compile(r"^# Conversation Turn (\d+) - (.+)$")


@dataclass
class LLMCallTrace:
    index: int
    call_type: str
    role: str
    metadata: dict[str, Any]
    prompt: str
    response: str

    @property
    def response_json(self) -> Any | None:
        try:
            return json.loads(extract_json(self.response))
        except Exception:  # noqa: BLE001
            return None


@dataclass
class UserResponseTrace:
    kind: str
    question: str
    answer: str


@dataclass
class TurnTrace:
    path: str
    turn_index: int | None
    timestamp: str | None
    user_query: str
    calls: list[LLMCallTrace] = field(default_factory=list)
    user_responses: list[UserResponseTrace] = field(default_factory=list)
    sequence: list[Any] = field(default_factory=list)  # calls and user responses in order
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def backbone(self) -> str | None:
        for c in self.calls:
            if c.metadata.get("model"):
                return str(c.metadata["model"])
        return None

    def calls_of(self, role: str) -> list[LLMCallTrace]:
        return [c for c in self.calls if c.role == role]


def _fenced(lines: list[str], start: int, stop: int) -> str:
    """Text inside the outermost fence between ``lines[start:stop]``."""
    block = lines[start:stop]
    while block and not block[0].strip():
        block.pop(0)
    while block and not block[-1].strip():
        block.pop()
    if block and block[0].startswith("```"):
        block = block[1:]
    if block and block[-1].strip() == "```":
        block = block[:-1]
    return "\n".join(block)


def _is_top_marker(line: str) -> bool:
    return bool(_CALL_RE.match(line) or _USER_RESP_RE.match(line) or line.strip() in ("## Summary", "## User Query"))


def parse_turn_log(path: Path | str) -> TurnTrace:
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    turn_index: int | None = None
    timestamp: str | None = None
    if lines:
        m = _TITLE_RE.match(lines[0])
        if m:
            turn_index, timestamp = int(m.group(1)), m.group(2).strip()

    # Top-level segmentation.
    markers = [i for i, ln in enumerate(lines) if _is_top_marker(ln)]
    markers.append(len(lines))
    trace = TurnTrace(path=str(p), turn_index=turn_index, timestamp=timestamp, user_query="")

    for a, b in zip(markers, markers[1:]):
        head = lines[a].strip()
        body = lines[a + 1 : b]
        # Trim the trailing '---' separator.
        while body and not body[-1].strip():
            body.pop()
        if body and body[-1].strip() == "---":
            body.pop()
        if head == "## User Query":
            trace.user_query = _fenced(body, 0, len(body))
            continue
        if head == "## Summary":
            for ln in body:
                mm = _META_RE.match(ln.strip())
                if mm:
                    trace.summary[mm.group(1).strip().lower().replace(" ", "_")] = mm.group(2).strip()
            continue
        m = _USER_RESP_RE.match(head)
        if m:
            q_idx = next((i for i, ln in enumerate(body) if ln.strip() == "**Question:**"), None)
            a_idx = next((i for i, ln in enumerate(body) if ln.strip() in ("**User Response:**", "**Answer:**")), None)
            question = _fenced(body, q_idx + 1, a_idx if a_idx is not None else len(body)) if q_idx is not None else ""
            answer = _fenced(body, a_idx + 1, len(body)) if a_idx is not None else ""
            ur = UserResponseTrace(kind=m.group(1).strip().lower().replace(" ", "_"), question=question, answer=answer)
            trace.user_responses.append(ur)
            trace.sequence.append(ur)
            continue
        m = _CALL_RE.match(head)
        if m:
            idx, call_type = int(m.group(1)), m.group(2)
            meta: dict[str, Any] = {}
            p_idx = next((i for i, ln in enumerate(body) if ln.strip() == "### Full Prompt"), None)
            r_idx = next((i for i, ln in enumerate(body) if ln.strip() == "### Response"), None)
            meta_end = p_idx if p_idx is not None else (r_idx if r_idx is not None else len(body))
            for ln in body[:meta_end]:
                mm = _META_RE.match(ln.strip())
                if mm:
                    key = mm.group(1).strip().lower().replace(" ", "_")
                    val: Any = mm.group(2).strip()
                    if key in ("input_tokens", "output_tokens"):
                        val = None if val in ("None", "") else int(val)
                    meta[key] = val
            prompt = _fenced(body, p_idx + 1, r_idx if r_idx is not None else len(body)) if p_idx is not None else ""
            response = _fenced(body, r_idx + 1, len(body)) if r_idx is not None else ""
            if p_idx is None or r_idx is None:
                trace.warnings.append(f"call {idx} ({call_type}) missing prompt or response section")
            call = LLMCallTrace(index=idx, call_type=call_type, role=ROLE_ALIASES.get(call_type, call_type), metadata=meta, prompt=prompt, response=response)
            trace.calls.append(call)
            trace.sequence.append(call)
    return trace


def parse_session_dir(directory: Path | str) -> list[TurnTrace]:
    return [parse_turn_log(p) for p in sorted(Path(directory).glob("turn_*.md"))]


# ---- semantic inputs and decisions ----------------------------------------------------------------

_SLICE_LINE_RE = re.compile(r"^\s*- Slice (\d+): (.+?) \((gene|protein), (cell|spot)-level, ([\d,]+) observations", re.M)
_USER_QUERY_RE = re.compile(r'\*\*User Query:\*\* "(.*?)"\s*$', re.S | re.M)
_REQUEST_RE = re.compile(r'^User Request: "(.*)"$', re.M)
_CATALOG_RE = re.compile(r"^- \*\*([a-z0-9-]+)\*\*:", re.M)
_TOPK_RE = re.compile(r"Return at most (\d+) skill slugs")
_SKILL_RE = re.compile(r"\*\*Selected Skill:\*\* (.+?) \(([a-z0-9-]+)\)")
_TARGET_RE = re.compile(r"- Target slices: \[([\d, ]*)\]")


def planner_inputs(call: LLMCallTrace) -> dict[str, Any]:
    slices = [{"slice_id": int(m.group(1)), "tissue": m.group(2), "modality": m.group(3), "data_level": m.group(4), "n_obs": int(m.group(5).replace(",", ""))} for m in _SLICE_LINE_RE.finditer(call.prompt)]
    q = _USER_QUERY_RE.search(call.prompt)
    return {"slices": slices, "query": q.group(1).strip() if q else None}


def planner_decision(call: LLMCallTrace) -> dict[str, Any]:
    data = call.response_json or {}
    steps = data.get("steps") or []
    return {
        "needs_clarification": bool(data.get("needs_clarification", False)) if data else None,
        "n_steps": len(steps),
        "target_slices": [sorted(int(x) for x in (s.get("target_slice_ids") or [])) for s in steps],
        "refined_queries": [s.get("refined_query") for s in steps],
        "parse_ok": bool(data),
    }


def matcher_inputs(call: LLMCallTrace) -> dict[str, Any]:
    r = _REQUEST_RE.search(call.prompt)
    k = _TOPK_RE.search(call.prompt)
    return {"request": r.group(1) if r else None, "catalog": sorted(set(_CATALOG_RE.findall(call.prompt))), "top_k": int(k.group(1)) if k else None}


def matcher_decision(call: LLMCallTrace) -> dict[str, Any]:
    data = call.response_json
    matched = [str(x) for x in data] if isinstance(data, list) else []
    return {"matched": matched, "top1": matched[0] if matched else None, "parse_ok": isinstance(data, list)}


def verifier_inputs(call: LLMCallTrace) -> dict[str, Any]:
    s = _SKILL_RE.search(call.prompt)
    t = _TARGET_RE.search(call.prompt)
    return {"skill": s.group(2) if s else None, "target_slices": [int(x) for x in t.group(1).replace(" ", "").split(",") if x] if t else []}


def verifier_decision(call: LLMCallTrace) -> dict[str, Any]:
    data = call.response_json or {}
    if not data:
        return {"kind": None, "parse_ok": False}
    if data.get("prerequisites_met"):
        kind = "met"
    elif data.get("can_obtain_by_chat"):
        kind = "ask"
    else:
        kind = "decline"
    return {"kind": kind, "n_questions": len(data.get("clarification_questions") or []), "missing": data.get("missing_prerequisites") or [], "parse_ok": True}


def analyzer_decision(call: LLMCallTrace) -> dict[str, Any]:
    data = call.response_json or {}
    issues = data.get("execution_issues") or {}
    return {"has_issues": bool(issues.get("has_issues")) if data else None, "issue_type": issues.get("issue_type"), "parse_ok": bool(data)}


def turn_decisions(trace: TurnTrace) -> list[dict[str, Any]]:
    """Ordered decision records for one turn (role, inputs, decision)."""
    out: list[dict[str, Any]] = []
    for item in trace.sequence:
        if isinstance(item, UserResponseTrace):
            out.append({"role": "user", "kind": item.kind, "question": item.question, "answer": item.answer})
            continue
        c: LLMCallTrace = item
        rec: dict[str, Any] = {"role": c.role, "index": c.index, "model": c.metadata.get("model")}
        if c.role == "planner":
            rec.update(inputs=planner_inputs(c), decision=planner_decision(c))
        elif c.role == "matcher":
            rec.update(inputs=matcher_inputs(c), decision=matcher_decision(c))
        elif c.role == "verifier":
            rec.update(inputs=verifier_inputs(c), decision=verifier_decision(c))
        elif c.role == "analyzer":
            rec.update(inputs={}, decision=analyzer_decision(c))
        elif c.role == "generator":
            n_code = len(re.findall(r"```python", c.response))
            rec.update(inputs={}, decision={"n_code_blocks": n_code, "has_interpret": "__INTERPRET__" in c.response})
        else:
            rec.update(inputs={}, decision={})
        out.append(rec)
    return out


__all__ = [
    "ROLE_ALIASES",
    "LLMCallTrace",
    "UserResponseTrace",
    "TurnTrace",
    "parse_turn_log",
    "parse_session_dir",
    "turn_decisions",
    "planner_decision",
    "matcher_decision",
    "verifier_decision",
    "analyzer_decision",
]
