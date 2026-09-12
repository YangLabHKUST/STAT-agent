from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from stat_bench.result import RunResult
from stat_bench.suite import Query


@dataclass
class Cell:
    query: Query
    suite_id: str
    data_root: str | None
    out_dir: str
    seed: int
    seed_index: int
    backbone: str
    driver: str = "staged"
    ablation: str = "A0"
    policy: str = "scripted"
    budget: dict[str, Any] = field(default_factory=dict)
    execute: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class System(Protocol):
    name: str

    async def run(self, cell: Cell) -> RunResult: ...
