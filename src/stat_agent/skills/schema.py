"""Skill contract schema (SKILL.md frontmatter), versions 1 and 2.

v1 keys (STAT 0.1): name, title, slug, description, filter_requirements{num_slices,
modalities, data_levels}, prerequisites, default_skill. v2 adds optional, additive keys:
schema_version, version, consumes, produces, postconditions, platforms, dependencies,
defaults, budget. Unknown keys are preserved (``extra='allow'``) and reported by the linter.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MODALITY_TOKENS = {"gene", "protein"}
LEVEL_TOKENS = {"cell", "spot"}
ARTIFACT_KINDS = ("obs", "obsm", "uns", "var", "obsp", "varm", "layers", "roi", "result", "figure")
SLUG_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def looks_like_slug(value: Optional[str]) -> bool:
    return isinstance(value, str) and SLUG_RE.fullmatch(value) is not None


def slugify(value: Optional[str], max_len: int = 64) -> str:
    if not value:
        return ""
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_len].strip("-") if len(slug) > max_len else slug


def _split_tokens(value: str) -> List[str]:
    return [t.strip() for t in str(value).split("/") if t.strip()]


class ArtifactRef(BaseModel):
    """A session artefact: ``obs:celltype``, ``obsm:deconv_weights``, ``roi:tumor`` ..."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[ARTIFACT_KINDS]  # type: ignore[valid-type]
    key: str
    scope: Literal["target", "all"] = "target"

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            if ":" not in data:
                raise ValueError(f"artifact reference must look like 'kind:key', got {data!r}")
            kind, key = data.split(":", 1)
            return {"kind": kind.strip(), "key": key.strip()}
        return data

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.key}"

    def matches(self, key: str) -> bool:
        """Glob-aware key match (``cellphonedb_*``)."""
        import fnmatch

        return fnmatch.fnmatchcase(key, self.key)

    def __str__(self) -> str:
        return self.id


class Postcondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: ArtifactRef
    check: Literal["exists", "non_null_fraction", "n_unique_min", "rows_match_obs", "rows_sum_to_one", "in_range"] = "exists"
    params: Dict[str, float] = Field(default_factory=dict)
    severity: Literal["error", "warn"] = "error"


class Dependencies(BaseModel):
    model_config = ConfigDict(extra="allow")

    python: List[str] = Field(default_factory=list)
    gpu: bool = False
    vendored: List[str] = Field(default_factory=list)
    subprocess_workers: bool = False
    notes: Optional[str] = None


class FilterRequirements(BaseModel):
    """Applicability contract checked deterministically against the target slices."""

    model_config = ConfigDict(extra="forbid")

    num_slices: Optional[int] = None
    min_slices: Optional[int] = None
    modalities: List[str] = Field(default_factory=list)
    data_levels: List[str] = Field(default_factory=list)

    @field_validator("modalities")
    @classmethod
    def _check_modalities(cls, v: List[str]) -> List[str]:
        for item in v:
            bad = [t for t in _split_tokens(item) if t not in MODALITY_TOKENS]
            if bad or not _split_tokens(item):
                raise ValueError(f"invalid modality token(s) {bad or [item]}; allowed: {sorted(MODALITY_TOKENS)} joined by '/'")
        return [str(x) for x in v]

    @field_validator("data_levels")
    @classmethod
    def _check_levels(cls, v: List[str]) -> List[str]:
        for item in v:
            bad = [t for t in _split_tokens(item) if t not in LEVEL_TOKENS]
            if bad or not _split_tokens(item):
                raise ValueError(f"invalid data_level token(s) {bad or [item]}; allowed: {sorted(LEVEL_TOKENS)} joined by '/'")
        return [str(x) for x in v]

    @model_validator(mode="after")
    def _check_lengths(self) -> "FilterRequirements":
        if self.num_slices is not None and self.num_slices < 1:
            raise ValueError("num_slices must be >= 1")
        if self.min_slices is not None and self.min_slices < 1:
            raise ValueError("min_slices must be >= 1")
        for field_name in ("modalities", "data_levels"):
            values = getattr(self, field_name)
            if len(values) > 1:
                if self.num_slices is None:
                    raise ValueError(f"{field_name} lists more than one entry but num_slices is not set (positional lists need num_slices)")
                if len(values) != self.num_slices:
                    raise ValueError(f"{field_name} has {len(values)} entries but num_slices is {self.num_slices}")
        return self

    @property
    def is_empty(self) -> bool:
        return self.num_slices is None and self.min_slices is None and not self.modalities and not self.data_levels

    def describe(self) -> str:
        parts = []
        if self.num_slices is not None:
            parts.append(f"{self.num_slices} slice(s)")
        elif self.min_slices is not None:
            parts.append(f">= {self.min_slices} slices")
        if self.modalities:
            parts.append("modality " + ", ".join(self.modalities))
        if self.data_levels:
            parts.append("level " + ", ".join(self.data_levels))
        return "; ".join(parts) or "any data"


def _parse_list_field(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        if "\n" in value:
            return [ln.strip() for ln in value.split("\n") if ln.strip() and not ln.strip().startswith("#")]
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        return [value.strip()] if value.strip() else []
    return [str(value)]


class SkillFrontmatter(BaseModel):
    """The machine-readable part of a SKILL.md."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_version: int = 1
    name: str
    title: Optional[str] = None
    slug: Optional[str] = None
    description: str
    filter_requirements: FilterRequirements = Field(default_factory=FilterRequirements)
    prerequisites: List[str] = Field(default_factory=list)
    default_skill: bool = True
    # v2 (optional, additive)
    version: str = "1.0"
    consumes: List[ArtifactRef] = Field(default_factory=list)
    produces: List[ArtifactRef] = Field(default_factory=list)
    postconditions: List[Postcondition] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)
    dependencies: Dependencies = Field(default_factory=Dependencies)
    defaults: Dict[str, str] = Field(default_factory=dict)
    budget: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_v1(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("filter_requirements") is None:
            data["filter_requirements"] = {}
        elif not isinstance(data["filter_requirements"], dict):
            raise ValueError("filter_requirements must be a mapping")
        data["prerequisites"] = _parse_list_field(data.get("prerequisites"))
        ds = data.get("default_skill", True)
        if isinstance(ds, str):
            data["default_skill"] = ds.strip().lower() in ("true", "1", "yes")
        if "display_title" in data and not data.get("title"):
            data["title"] = data["display_title"]
        if "platforms" in data:
            data["platforms"] = _parse_list_field(data.get("platforms"))
        return data

    @model_validator(mode="after")
    def _derive_identity(self) -> "SkillFrontmatter":
        if not self.title:
            self.title = self.name
        if not self.slug:
            self.slug = self.name if looks_like_slug(self.name) else slugify(self.title)
        if not looks_like_slug(self.slug):
            raise ValueError(f"slug {self.slug!r} must be lowercase alphanumerics separated by single hyphens")
        if not self.description or not self.description.strip():
            raise ValueError("description is required")
        return self

    @property
    def display_name(self) -> str:
        return self.title or self.name

    @property
    def extra_keys(self) -> List[str]:
        return sorted((self.model_extra or {}).keys())

    def consumes_keys(self, kind: Optional[str] = None) -> List[ArtifactRef]:
        return [a for a in self.consumes if kind is None or a.kind == kind]

    def produces_keys(self, kind: Optional[str] = None) -> List[ArtifactRef]:
        return [a for a in self.produces if kind is None or a.kind == kind]


class MethodCard(SkillFrontmatter):
    """A synthesised contract for an unregistered method; passes through the same gates."""

    method_name: str
    outline: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    provisional: Literal[True] = True
    research_notes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _fill_identity(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("name", slugify(data.get("method_name", "method")) or "method")
            data.setdefault("title", data.get("method_name"))
        return data


def split_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Return ``(frontmatter_dict, body)``; raises ValueError when the header is missing."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing YAML frontmatter (file must start with '---')")
    try:
        closing = lines.index("---", 1)
    except ValueError:
        raise ValueError("unterminated YAML frontmatter") from None
    header = "\n".join(lines[1:closing])
    loaded = yaml.safe_load(header) if header.strip() else {}
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    body = "\n".join(lines[closing + 1 :]).strip()
    return loaded, body


def parse_skill_text(text: str) -> Tuple[SkillFrontmatter, str, Dict[str, Any]]:
    raw, body = split_frontmatter(text)
    return SkillFrontmatter.model_validate(raw), body, raw


__all__ = [
    "ARTIFACT_KINDS",
    "ArtifactRef",
    "Postcondition",
    "Dependencies",
    "FilterRequirements",
    "SkillFrontmatter",
    "MethodCard",
    "split_frontmatter",
    "parse_skill_text",
    "looks_like_slug",
    "slugify",
]
