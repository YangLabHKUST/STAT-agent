"""Versioned prompt templates.

Each ``<role>.v<N>.md`` file has a YAML header (id, version, role, description, source) and a
Jinja2 body. Templates are loaded by ``load_prompt("planner")`` (latest version) or
``load_prompt("planner", version=1)``; ``Prompt.render(**vars)`` renders with strict
undefined checking so a missing variable is an error, never silent text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

_HEADER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_ENV = Environment(undefined=StrictUndefined, keep_trailing_newline=True, autoescape=False)


@dataclass(frozen=True)
class Prompt:
    id: str
    version: int
    role: str
    description: str
    source: str
    body: str
    sha256: str
    path: str

    @property
    def prompt_id(self) -> str:
        return f"{self.id}/v{self.version}"

    def render(self, **variables: Any) -> str:
        return _ENV.from_string(self.body).render(**variables)


def _prompt_dir() -> Path:
    return Path(str(resources.files("stat_agent.prompts")))


def available_prompts() -> dict[str, dict[int, Path]]:
    out: dict[str, dict[int, Path]] = {}
    for path in sorted(_prompt_dir().glob("*.v*.md")):
        m = re.match(r"(.+)\.v(\d+)\.md\Z", path.name)
        if not m:
            continue
        out.setdefault(m.group(1), {})[int(m.group(2))] = path
    return out


def load_prompt(name: str, version: int | None = None) -> Prompt:
    versions = available_prompts().get(name)
    if not versions:
        raise FileNotFoundError(f"no prompt template named {name!r} in {_prompt_dir()}")
    version = version or max(versions)
    path = versions[version]
    text = path.read_text(encoding="utf-8")
    m = _HEADER_RE.match(text)
    if not m:
        raise ValueError(f"{path}: missing YAML header")
    header = yaml.safe_load(m.group(1)) or {}
    body = text[m.end() :]
    return Prompt(
        id=str(header.get("id", name)),
        version=int(header.get("version", version)),
        role=str(header.get("role", name)),
        description=str(header.get("description", "")),
        source=str(header.get("source", "")),
        body=body,
        sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        path=str(path),
    )


__all__ = ["Prompt", "load_prompt", "available_prompts"]
