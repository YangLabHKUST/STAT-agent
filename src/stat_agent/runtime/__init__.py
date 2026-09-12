"""Guarded in-process execution runtime."""

from stat_agent.runtime.artifacts import CodeAccess, code_access
from stat_agent.runtime.executor import RunResult, Runtime, RuntimeBusy
from stat_agent.runtime.safety import SafetyViolation, Violation, check_code, clean_code
from stat_agent.runtime.snapshot import (
    Snapshot,
    SnapshotDiff,
    diff_snapshots,
    parse_artifact_id,
    take_snapshot,
)

__all__ = [
    "Runtime",
    "RunResult",
    "RuntimeBusy",
    "SafetyViolation",
    "Violation",
    "check_code",
    "clean_code",
    "Snapshot",
    "SnapshotDiff",
    "take_snapshot",
    "diff_snapshots",
    "parse_artifact_id",
    "CodeAccess",
    "code_access",
]
