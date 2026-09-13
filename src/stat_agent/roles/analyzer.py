"""Analyzer role: summary, data changes, optional interpretation, execution-issue verdict."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from stat_agent.core.view import SessionSummary
from stat_agent.memory.context import analyzer_session_info, history_block, with_turn_steps
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


class BlockResult(BaseModel):
    code: str
    stdout: str = ""
    error: str | None = None
    success: bool = True
    variables: list[str] = Field(default_factory=list)  # namespace names this block added


def assemble_context(segments: list[Segment], results: list[BlockResult], max_output: int = 3000, tail: int = 1500) -> str:
    """The per-block text/code/output the analyser reads. Long output keeps its head and its tail:
    the final table or count is what the user asked for and it tends to be printed last."""
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
                    if len(out) > max_output + tail:
                        out = out[:max_output] + "\n... (middle of the output omitted; the data in memory is complete) ...\n" + out[-tail:]
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
        incomplete: list[str] = (),
        turn_steps: str = "",
        stale: list[str] = (),
        replaced_loaded: list[str] = (),
        turn_id: str | None = None,
        step: int | None = None,
    ) -> Analysis:
        context = assemble_context(segments, results)
        if produced:
            context += "\n**Artefacts written this step (from the provenance ledger, the ground truth for data_changes):** " + ", ".join(produced) + "\n"
        if replaced_loaded:
            context += "\n**This step overwrote data that came with the dataset:** " + ", ".join(replaced_loaded) + ". Say so plainly in response_summary and data_changes; the original values are no longer in memory.\n"
        if stale:
            context += "\n**Artefacts on the target slices currently marked STALE (an input changed after they were computed; results derived from them are not valid until recomputed):** " + ", ".join(stale) + ". Never describe a stale artefact or anything built on it as valid or unchanged.\n"
        if incomplete:
            context += "\n**Note:** the code ran without errors, but the step's declared outputs were not produced: " + "; ".join(incomplete) + "\n"
        variables = {
            "session_info": analyzer_session_info(summary, roi_names) if summary else "",
            "history_block": with_turn_steps(history_block(memory, "plain"), turn_steps),
            "user_query": user_query,
            "assembled_context": context,
            "has_interpret": bool(interpret_instruction),
            "interpret_instruction": interpret_instruction,
        }
        fake_input = {"query": user_query, "code": [s.content for s in segments if s.kind == "code"], "outputs": [r.stdout[-500:] for r in results], "errors": [r.error for r in results]}
        comp = await self.call(variables, schema=Analysis, fake_input=fake_input, turn_id=turn_id, step=step)
        a: Analysis = comp.value
        return self.ground(a, results, list(produced), list(incomplete))

    @staticmethod
    def ground(a: Analysis, results: list[BlockResult], produced: list[str], incomplete: list[str] = ()) -> Analysis:
        if any(not r.success for r in results):
            if not a.execution_issues.has_issues:
                a.execution_issues = ExecutionIssues(has_issues=True, issue_type="error", explanation=a.execution_issues.explanation or "A code block raised an error.")
        if incomplete:
            note = "The step's declared outputs were not produced: " + "; ".join(incomplete) + "."
            a.execution_issues = ExecutionIssues(has_issues=True, issue_type="partial_success", explanation=(note + " " + (a.execution_issues.explanation or "")).strip())
        return a

    @staticmethod
    def deterministic(results: list[BlockResult], produced: list[str], stdout_tail: str, incomplete: list[str] = ()) -> Analysis:
        """Ablation / failure fallback: build the handoff from the ledger delta and stdout."""
        ok = all(r.success for r in results)
        changes = ", ".join(produced) if produced else "no session artefacts changed"
        issues = ExecutionIssues(has_issues=not ok, issue_type=None if ok else "error", explanation=None if ok else next((r.error for r in results if r.error), None))
        if ok and incomplete:
            issues = ExecutionIssues(has_issues=True, issue_type="partial_success", explanation="The step's declared outputs were not produced: " + "; ".join(incomplete) + ".")
        return Analysis(
            response_summary=(stdout_tail[:300].strip() or ("Code executed successfully." if ok else "Code execution failed.")),
            data_changes=f"Artefacts changed: {changes}.",
            execution_issues=issues,
            analyzer_failed=True,
        )


__all__ = ["AnalyzerRole", "Analysis", "ExecutionIssues", "BlockResult", "assemble_context", "clean_output"]
