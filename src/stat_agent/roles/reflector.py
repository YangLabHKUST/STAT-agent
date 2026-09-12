"""Reflector role: classify a failure, propose a small fix, or give up."""

from __future__ import annotations

from pydantic import BaseModel

from stat_agent.roles.base import Role

UNFIXABLE = {"MemoryError", "TimeoutError", "Timeout", "PermissionError", "KeyboardInterrupt", "Cancelled", "SafetyViolation", "RuntimeBusy"}


class Reflection(BaseModel):
    error_type: str = ""
    root_cause: str = ""
    fix_strategy: str = ""
    change_magnitude: str = "unknown"
    fixed_code: str | None = None
    confidence: float = 0.0
    should_retry: bool = False


def should_attempt_fix(error_type: str | None) -> bool:
    return (error_type or "") not in UNFIXABLE


def retry_allowed(r: Reflection, attempt: int, max_attempts: int = 2, min_confidence: float = 0.7) -> bool:
    return bool(r.should_retry and r.fixed_code and r.change_magnitude == "small" and r.confidence >= min_confidence and attempt <= max_attempts)


class ReflectorRole(Role):
    name = "reflector"
    prompt_name = "reflector"

    async def reflect(
        self,
        *,
        user_query: str,
        code: str,
        error_msg: str,
        stdout_tail: str = "",
        available_rois: dict | None = None,
        available_columns: list[str] | None = None,
        has_celltype: bool = False,
        attempt: int = 1,
        turn_id: str | None = None,
        step: int | None = None,
    ) -> Reflection:
        variables = {
            "user_query": user_query,
            "original_code": code,
            "error_msg": error_msg,
            "stdout_tail": stdout_tail,
            "available_rois": available_rois or {},
            "available_columns": available_columns or [],
            "has_celltype": has_celltype,
        }
        fake_input = {"query": user_query, "code": code, "error": error_msg, "attempt": attempt}
        comp = await self.call(variables, schema=Reflection, fake_input=fake_input, turn_id=turn_id, step=step)
        r: Reflection = comp.value
        r.change_magnitude = (r.change_magnitude or "unknown").lower()
        if r.change_magnitude == "large" or r.confidence < 0.7:
            r.should_retry = False
        return r


__all__ = ["ReflectorRole", "Reflection", "should_attempt_fix", "retry_allowed", "UNFIXABLE"]
