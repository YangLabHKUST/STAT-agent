"""Deterministic skill filter (the only non-LLM stage of skill selection).

Semantics (unchanged from STAT 0.1, plus ``min_slices``):

* ``num_slices``: omitted -> any count; otherwise the request must target exactly that many;
* ``min_slices``: the request must target at least that many slices;
* ``modalities`` / ``data_levels``: omitted -> no constraint; one element broadcasts to all
  target slices; a list of ``num_slices`` elements is positional; ``a/b`` means "a or b".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Sequence, Tuple

from stat_agent.core.view import SessionSummary, SliceSummary
from stat_agent.skills.schema import FilterRequirements


@dataclass
class FilterResult:
    compatible: List[str]
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    bypassed: bool = False
    would_reject: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def rejected_slugs(self) -> List[str]:
        return [s for s, _ in self.rejected]

    def reason_for(self, slug: str) -> str:
        for s, r in self.rejected + self.would_reject:
            if s == slug:
                return r
        return ""


def _tokens(value: str) -> List[str]:
    return [t.strip() for t in str(value).split("/") if t.strip()]


def check_compatibility(req: FilterRequirements, slices: Sequence[SliceSummary]) -> Tuple[bool, str]:
    """Return ``(compatible, reason)`` for one contract against the ordered target slices."""
    n = len(slices)
    if req.num_slices is not None and n != req.num_slices:
        return False, f"requires {req.num_slices} slice(s), got {n}"
    if req.min_slices is not None and n < req.min_slices:
        return False, f"requires at least {req.min_slices} slice(s), got {n}"

    for field_name, attr in (("modalities", "modality"), ("data_levels", "data_level")):
        required = list(getattr(req, field_name))
        if not required:
            continue
        if len(required) == 1:
            required = required * n
        elif len(required) != n:
            return False, f"{field_name} list length mismatch: requires {len(required)}, got {n} slices"
        for want, s in zip(required, slices):
            actual = getattr(s, attr)
            if actual not in _tokens(want):
                return False, f"slice {s.slice_id} has {attr} '{actual}', requires '{want}'"
    return True, ""


def filter_skills(
    target_slice_ids: Iterable[int],
    summary: SessionSummary,
    contracts: Mapping[str, FilterRequirements],
    bypass: bool = False,
) -> FilterResult:
    """Partition ``contracts`` into compatible and rejected for the target slices.

    Raises ``ValueError`` for an unknown or empty target list (a planner bug must surface,
    not silently produce an empty catalog). With ``bypass=True`` (ablation A1) every skill is
    returned as compatible and the rejections are recorded in ``would_reject`` instead.
    """
    ids = [int(i) for i in target_slice_ids]
    if not ids:
        raise ValueError("filter_skills needs at least one target slice id")
    slices: List[SliceSummary] = []
    for sid in ids:
        s = summary.slice(sid)
        if s is None:
            raise ValueError(f"target slice {sid} is not loaded (available: {summary.slice_ids})")
        slices.append(s)

    compatible: List[str] = []
    rejected: List[Tuple[str, str]] = []
    for slug in sorted(contracts):
        ok, reason = check_compatibility(contracts[slug], slices)
        if ok:
            compatible.append(slug)
        else:
            rejected.append((slug, reason))
    if bypass:
        return FilterResult(compatible=sorted(contracts), rejected=[], bypassed=True, would_reject=rejected)
    return FilterResult(compatible=compatible, rejected=rejected)


__all__ = ["FilterResult", "check_compatibility", "filter_skills"]
