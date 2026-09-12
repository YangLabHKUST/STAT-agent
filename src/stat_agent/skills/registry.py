"""Skill registry: progressive disclosure over one or more skill directories.

Frontmatter is parsed at load time; bodies are read lazily. Later search paths override
earlier ones for the same slug (last wins), so a user directory can shadow a built-in.
Skills whose frontmatter fails validation are recorded in ``registry.broken`` and are
never offered to the matcher.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from stat_agent.skills.schema import SkillFrontmatter, parse_skill_text

logger = logging.getLogger(__name__)


def builtin_skills_path() -> Path:
    return Path(str(resources.files("stat_agent.skills")))


@dataclass
class Skill:
    meta: SkillFrontmatter
    path: Path  # directory containing SKILL.md
    origin: str = "builtin"
    raw_frontmatter: Dict = field(default_factory=dict)
    _body: Optional[str] = field(default=None, repr=False)

    @property
    def slug(self) -> str:
        return self.meta.slug or ""

    @property
    def name(self) -> str:
        return self.meta.display_name

    @property
    def description(self) -> str:
        return self.meta.description

    @property
    def file(self) -> Path:
        return self.path / "SKILL.md"

    @property
    def body(self) -> str:
        if self._body is None:
            _, self._body, _ = parse_skill_text(self.file.read_text(encoding="utf-8"))
        return self._body

    def prompt_instructions(self, max_chars: int = 25_000) -> str:
        body = self.body
        if len(body) > max_chars:
            body = body[:max_chars] + "\n\n[... skill instructions truncated ...]"
        return body

    def __repr__(self) -> str:
        return f"Skill(slug={self.slug!r}, origin={self.origin!r}, path={str(self.path)!r})"


@dataclass
class BrokenSkill:
    path: Path
    error: str
    origin: str = "builtin"


class SkillRegistry:
    """Load and manage skills from a list of directories (each holding ``<slug>/SKILL.md``)."""

    def __init__(self, search_paths: Optional[Sequence[Tuple[Path, str]]] = None):
        self.search_paths: List[Tuple[Path, str]] = [(Path(p), label) for p, label in (search_paths or [])]
        self.skills: Dict[str, Skill] = {}
        self.broken: List[BrokenSkill] = []
        self.disabled: set[str] = set()
        self._seen: set[str] = set()  # slugs whose initial enabled state was applied once
        self._loaded = False

    # ---- construction ----------------------------------------------------------------------
    @classmethod
    def default(cls, extra_paths: Iterable[Path] = (), user_dir: Optional[Path] = None) -> "SkillRegistry":
        paths: List[Tuple[Path, str]] = [(builtin_skills_path(), "builtin")]
        for p in extra_paths:
            paths.append((Path(p), "extra"))
        if user_dir is not None and Path(user_dir).exists():
            paths.append((Path(user_dir), "user"))
        reg = cls(paths)
        reg.load()
        return reg

    @classmethod
    def from_settings(cls, settings) -> "SkillRegistry":
        return cls.default(extra_paths=settings.extra_skill_paths, user_dir=settings.user_skills_dir)

    def load(self) -> "SkillRegistry":
        self.skills.clear()
        self.broken.clear()
        for root, label in self.search_paths:
            if not root.exists():
                logger.warning("skill path does not exist: %s", root)
                continue
            for skill_file in sorted(root.glob("*/SKILL.md")):
                try:
                    meta, _, raw = parse_skill_text(skill_file.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001 - every parse failure is reported, never raised
                    self.broken.append(BrokenSkill(path=skill_file, error=str(exc), origin=label))
                    logger.warning("skipping broken skill %s: %s", skill_file, exc)
                    continue
                skill = Skill(meta=meta, path=skill_file.parent, origin=label, raw_frontmatter=raw)
                if skill.slug in self.skills:
                    logger.info("skill %s from %s overrides %s", skill.slug, label, self.skills[skill.slug].origin)
                self.skills[skill.slug] = skill
        # default_skill: false = off until the user enables it (0.1 semantics); a reload keeps the user's toggles
        for slug, sk in self.skills.items():
            if slug not in self._seen:
                self._seen.add(slug)
                if not sk.meta.default_skill:
                    self.disabled.add(slug)
        self._loaded = True
        logger.info("loaded %d skills (%d broken) from %d path(s)", len(self.skills), len(self.broken), len(self.search_paths))
        return self

    # ---- access -----------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.skills)

    def __contains__(self, slug: object) -> bool:
        return slug in self.skills

    def get(self, slug: str) -> Skill:
        try:
            return self.skills[slug]
        except KeyError:
            raise KeyError(f"unknown skill {slug!r}; available: {sorted(self.skills)}") from None

    def slugs(self) -> List[str]:
        return sorted(self.skills)

    def is_enabled(self, slug: str) -> bool:
        skill = self.skills.get(slug)
        return skill is not None and slug not in self.disabled

    def enabled(self) -> Dict[str, Skill]:
        return {s: sk for s, sk in self.skills.items() if self.is_enabled(s)}

    def enable(self, slug: str) -> None:
        self.get(slug)
        self.disabled.discard(slug)

    def disable(self, slug: str) -> None:
        self.get(slug)
        self.disabled.add(slug)

    def contracts(self, slugs: Optional[Iterable[str]] = None) -> Dict[str, SkillFrontmatter]:
        pool = self.enabled() if slugs is None else {s: self.get(s) for s in slugs}
        return {s: sk.meta for s, sk in pool.items()}

    def catalog(self, slugs: Optional[Iterable[str]] = None) -> List[Tuple[str, str]]:
        """``[(slug, description)]`` sorted by display name, as the matcher prompt lists them."""
        pool = self.enabled() if slugs is None else {s: self.get(s) for s in slugs}
        return [(s, sk.description) for s, sk in sorted(pool.items(), key=lambda kv: kv[1].name.lower())]

    def states(self) -> Dict[str, Dict[str, object]]:
        return {
            s: {
                "name": sk.name,
                "title": sk.name,  # the skills panel reads `title`
                "description": sk.description,
                "enabled": self.is_enabled(s),
                "default_skill": sk.meta.default_skill,
                "origin": sk.origin,
                "filter_requirements": sk.meta.filter_requirements.model_dump(exclude_none=True),
            }
            for s, sk in sorted(self.skills.items())
        }


__all__ = ["Skill", "BrokenSkill", "SkillRegistry", "builtin_skills_path"]
