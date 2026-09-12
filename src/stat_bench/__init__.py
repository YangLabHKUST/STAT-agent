"""STAT benchmark harness (stat-bench)."""

from stat_bench.result import RunResult
from stat_bench.suite import Suite, convert_legacy_queries

__all__ = ["RunResult", "Suite", "convert_legacy_queries"]
