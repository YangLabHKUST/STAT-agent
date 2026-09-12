"""Analyzer role: summary, data changes, optional interpretation, execution-issue verdict."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from stat_agent.core.view import SessionSummary
from stat_agent.memory.context import analyzer_session_info, history_block
from stat_agent.memory.state import ConversationMemory
from stat_agent.roles.base import Role
from stat_agent.roles.generator import Segment

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean_output(text: str) -> str:
    text = _ANSI_RE.sub("", text or "")
    lines = [ln for ln in text.splitlines() if not re.match(r"^\s*\d+%\|", ln)]
    return "\n".join(lines).strip()


class ExecutionIssues(BaseModel):
    has_issues: bool = False
    issue_type: str | None = None
    explanation: str | None = None


class Analysis(BaseModel):
    response_summary: str = ""
    data_changes: str = ""
    interpretation: str | None = None
    key_findings: list[str] = Field(default_factory=list)
    execution_issues: ExecutionIssues = Field(default_factory=ExecutionIssues)
    analyzer_failed: bool = False
    omitted_changes: list[str] = Field(default_factory=list)


class BlockResult(BaseModel):
    code: str
    stdout: str = ""
    error: str | None = None
    success: bool = True


def assemble_context(segments: list[Segment], results: list[BlockResult], max_output: int = 3000) -> str:
    parts: list[str] = []
    idx = 0
    for seg in segments:
        if seg.kind == "text":
            parts.append(f"**Your explanation:**\n{seg.content}\n")
        else:
            parts.append(f"**Code:**\n```python\n{seg.content}\n```\n")
            if idx < len(results):
                r = results[idx]
                if r.stdout:
                    out = clean_output(r.stdout)
                    if len(out) > max_output:
                        out = out[:max_output] + "\n... (output truncated)"
                    parts.append(f"**Output:**\n{out}\n")
                if r.error:
                    parts.append(f"**Error:**\n{r.error}\n")
                idx += 1
    return "\n".join(parts)


class AnalyzerRole(Role):
    name = "analyzer"
    prompt_name = "analyzer"

    async def analyze(
        self,
        *,
        user_query: str,
        segments: list[Segment],
        results: list[BlockResult],
        summary: SessionSummary | None,
        roi_names: list[str],
        memory: ConversationMemory | None,
        interpret_instruction: str = "",
        produced: list[str] = (),
        turn_id: str | None = None,
        step: int | None = None,
    ) -> Analysis:
        variables = {
            "session_info": analyzer_session_info(summary, roi_names) if summary else "",
            "history_block": history_block(memory, "plain"),
            "user_query": user_query,
            "assembled_context": assemble_context(segments, results),
            "has_interpret": bool(interpret_instruction),
            "interpret_instruction": interpret_instruction,
        }
        fake_input = {"query": user_query, "code": [s.content for s in segments if s.kind == "code"], "outputs": [r.stdout[-500:] for r in results], "errors": [r.error for r in results]}
        comp = await self.call(variables, schema=Analysis, fake_input=fake_input, turn_id=turn_id, step=step)
        a: Analysis = comp.value
        return self.ground(a, results, list(produced))

    @staticmethod
    def ground(a: Analysis, results: list[BlockResult], produced: list[str]) -> Analysis:
        if any(not r.success for r in results):
            if not a.execution_issues.has_issues:
                a.execution_issues = ExecutionIssues(has_issues=True, issue_type="error", explanation=a.execution_issues.explanation or "A code block raised an error.")
        text = (a.data_changes or "") + " " + (a.response_summary or "")
        a.omitted_changes = [p for p in produced if p.split(":")[-1] not in text]
        return a

    @staticmethod
    def deterministic(results: list[BlockResult], produced: list[str], stdout_tail: str) -> Analysis:
        """Ablation / failure fallback: build the handoff from the ledger delta and stdout."""
        ok = all(r.success for r in results)
        changes = ", ".join(produced) if produced else "no session artefacts changed"
        return Analysis(
            response_summary=(stdout_tail[:300].strip() or ("Code executed successfully." if ok else "Code execution failed.")),
            data_changes=f"Artefacts changed: {changes}.",
            execution_issues=ExecutionIssues(has_issues=not ok, issue_type=None if ok else "error", explanation=None if ok else next((r.error for r in results if r.error), None)),
            analyzer_failed=True,
        )


__all__ = ["AnalyzerRole", "Analysis", "ExecutionIssues", "BlockResult", "assemble_context", "clean_output"]
