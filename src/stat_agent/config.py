"""Runtime settings resolved from environment variables (no config file magic)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["interactive", "auto"]
Driver = Literal["staged", "loop"]


class Settings(BaseModel):
    home: Path = Field(default_factory=lambda: Path(os.environ.get("STAT_HOME", Path.home() / ".stat_agent")))
    logs_dir: Path | None = None
    extra_skill_paths: list[Path] = Field(default_factory=list)
    default_model: str = "openai/gpt-4o"
    mode: Mode = "interactive"
    driver: Driver = "staged"
    loop_backend: Literal["langchain", "deepagents"] = "langchain"  # loop-mode host; deepagents adds todo planning + subagents
    loop_subagents: bool = True
    max_code_runs: int = 8  # loop mode: code runs per turn
    exec_timeout_s: float = 1800.0
    max_reflect_attempts: int = 2
    matcher_top_k: int = 2

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        skill_paths = [Path(p) for p in env.get("STAT_SKILL_PATHS", "").split(os.pathsep) if p]
        return cls(
            logs_dir=Path(env["STAT_LOGS_DIR"]) if env.get("STAT_LOGS_DIR") else None,
            extra_skill_paths=skill_paths,
            default_model=env.get("STAT_MODEL") or env.get("SPATIAL_AGENT_MODEL") or "openai/gpt-4o",
            mode=env.get("STAT_MODE", "interactive"),  # type: ignore[arg-type]
            driver=env.get("STAT_DRIVER", "staged"),  # type: ignore[arg-type]
            exec_timeout_s=float(env.get("STAT_EXEC_TIMEOUT", "1800")),
            max_reflect_attempts=int(env.get("STAT_MAX_REFLECT", "2")),
            matcher_top_k=int(env.get("STAT_MATCHER_TOP_K", "2")),
        )

    @property
    def user_skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def extdata_dir(self) -> Path:
        return self.home / "extdata"

    @property
    def resolved_logs_dir(self) -> Path:
        return self.logs_dir or (Path.cwd() / "logs")


__all__ = ["Settings", "Mode", "Driver"]
