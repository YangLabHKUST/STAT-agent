"""Make a skill's vendored helper packages importable for the duration of a run."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List

from stat_agent.skills.registry import Skill


def vendored_packages(skill: Skill) -> List[str]:
    """Top-level importable names shipped inside the skill directory."""
    declared = list(skill.meta.dependencies.vendored)
    if declared:
        return declared
    names = []
    for child in sorted(skill.path.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            names.append(child.name)
        elif child.suffix == ".py" and child.stem not in ("setup",):
            names.append(child.stem)
    return names


def import_paths(skills: Iterable[Skill]) -> List[Path]:
    return [s.path for s in skills if vendored_packages(s)]


@contextmanager
def scoped_sys_path(paths: Iterable[Path], export_pythonpath: bool = False) -> Iterator[None]:
    """Temporarily prepend ``paths`` to ``sys.path`` (and optionally ``PYTHONPATH`` for
    subprocess workers such as Ray). Already-imported modules stay cached, as before."""
    added = [str(p) for p in paths if str(p) not in sys.path]
    for p in reversed(added):
        sys.path.insert(0, p)
    old_env = os.environ.get("PYTHONPATH")
    if export_pythonpath and added:
        os.environ["PYTHONPATH"] = os.pathsep.join(added + ([old_env] if old_env else []))
    try:
        yield
    finally:
        for p in added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass
        if export_pythonpath and added:
            if old_env is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_env


__all__ = ["vendored_packages", "import_paths", "scoped_sys_path"]
