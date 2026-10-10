"""Provenance ledger: what exists in the session, who produced it, and what is stale.

Every artefact (``s0:obs:celltype``, ``s1:obsm:deconv_weights``, ``roi:tumor``, ...) carries
its producer, its inputs and a status. When an artefact changes or is deleted, every
dependant that was not re-produced in the same transaction becomes ``stale``. The ledger
is JSON-serialisable and lives next to the session logs; it never holds AnnData.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from stat_agent.events.catalogue import ArtifactChange, ArtifactRefView, ProvenanceUpdate, StateChangeSummary
from stat_agent.runtime.snapshot import Fingerprint, Snapshot, SnapshotDiff, diff_snapshots, parse_artifact_id


class StepRef(BaseModel):
    turn_id: str | None = None
    step: int | None = None
    skill: str | None = None
    code_hash: str | None = None
    run_id: str | None = None
    mode: str = "guided"
    label: str | None = None


class Artifact(BaseModel):
    id: str
    kind: str
    key: str
    slice_id: int | None = None
    fingerprint: str = ""
    version: int = 1
    produced_by: StepRef | None = None
    inputs: list[str] = Field(default_factory=list)
    status: Literal["fresh", "stale", "deleted"] = "fresh"
    stale_reason: str | None = None
    origin: str = "load"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    shape: list[int] | None = None
    dtype: str | None = None
    categories: list[str] | None = None  # the distinct labels of a string obs column
    columns: list[str] | None = None  # the column names of a table (the width is in `shape`)


MAX_STORED_LABELS = 64
MAX_SHOWN_COLUMNS = 8


def _columns(fp: Fingerprint) -> list[str] | None:
    cols = (fp.extra or {}).get("columns")
    return [str(c)[:40] for c in cols] if cols else None


def _categories(fp: Fingerprint) -> list[str] | None:
    cats = (fp.extra or {}).get("categories")
    if not cats or len(cats) > MAX_STORED_LABELS:  # more than this is an identifier column, not labels
        return None
    return [str(c)[:60] for c in cats]


def artifact_spec(a: Artifact) -> str:
    """What the object is, in a few characters, for the inventory the roles are given: code written
    in a later turn reads this instead of guessing the shape of an artefact it did not create."""
    shape, dtype = a.shape or [], a.dtype or ""
    if a.kind == "roi":
        parts = [p for p in (dtype, f"{shape[0]:,} cells" if shape else "") if p]
        return f" [{', '.join(parts)}]" if parts else ""
    if a.kind == "obs":
        if a.categories:  # the labels themselves, so later code compares against the values present
            shown = ", ".join(c[:24] for c in a.categories[:6])
            more = f", +{len(a.categories) - 6} more" if len(a.categories) > 6 else ""
            return f" [labels: {shown}{more}]"
        return f" [{dtype}]" if dtype else ""
    if dtype == "dict":
        return f" [dict, {shape[0]} entries]" if shape else " [dict]"
    if dtype in ("list", "tuple", "set") and shape:
        return f" [{dtype}, {shape[0]} items]"
    if len(shape) == 2:
        dims = f"{shape[0]:,} × {shape[1]:,}"
        head = f"table {dims}" if dtype == "DataFrame" else (f"{dims} {dtype}" if dtype else dims)
        if a.columns:
            shown = ", ".join(c[:24] for c in a.columns[:MAX_SHOWN_COLUMNS])
            more = f", +{shape[1] - MAX_SHOWN_COLUMNS} more" if shape[1] > MAX_SHOWN_COLUMNS else ""
            return f" [{head}: {shown}{more}]"
        return f" [{head}]"
    if len(shape) == 1 and dtype not in ("dict", "list", "tuple", "set"):
        return f" [{shape[0]:,} values]"
    if dtype in ("str", "int", "float", "bool", "NoneType"):
        return f" [{dtype}]"
    return ""


class LedgerEntry(BaseModel):
    seq: int
    ts: float = Field(default_factory=time.time)
    origin: str
    producer: StepRef | None = None
    added: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    stale_now: list[str] = Field(default_factory=list)
    stale_cleared: list[str] = Field(default_factory=list)
    replaced_loaded: list[str] = Field(default_factory=list)  # artefacts that came with the dataset and were overwritten by this run

    @property
    def touched(self) -> list[str]:
        return self.added + self.changed


class LedgerDelta(BaseModel):
    entry: LedgerEntry
    ledger_version: int
    summary: StateChangeSummary

    @property
    def empty(self) -> bool:
        e = self.entry
        return not (e.added or e.changed or e.removed or e.stale_now or e.stale_cleared)

    def to_event(self, origin: str | None = None, renamed: list[list[str]] | None = None) -> ProvenanceUpdate:
        e = self.entry
        changes: list[ArtifactChange] = []
        for op, ids in (("created", e.added), ("updated", e.changed), ("deleted", e.removed)):
            for aid in ids:
                sid, kind, key = parse_artifact_id(aid)
                changes.append(ArtifactChange(artifact=ArtifactRefView(kind=kind, key=key, slice_id=sid), op=op))
        summary = self.summary.model_copy()
        if renamed:
            summary.rois_renamed = renamed
        return ProvenanceUpdate(
            ledger_version=self.ledger_version,
            origin=origin or e.origin,  # type: ignore[arg-type]
            producer=e.producer.model_dump() if e.producer else None,
            changes=changes,
            stale_now=list(e.stale_now),
            stale_cleared=list(e.stale_cleared),
            summary=summary,
        )

    def to_v1(self) -> dict[str, list]:
        s = self.summary
        return {
            "rois_added": list(s.rois_added),
            "rois_deleted": list(s.rois_deleted),
            "celltypes_updated": list(s.celltypes_updated),
            "celltype_colors_updated": list(s.celltype_colors_updated),
            "deconv_weights_updated": list(s.deconv_weights_updated),
            "artifacts_changed": self.entry.touched,
            "stale": list(self.entry.stale_now),
        }


def summarize_v1(added: Iterable[str], changed: Iterable[str], removed: Iterable[str]) -> StateChangeSummary:
    s = StateChangeSummary()
    for aid in list(added) + list(changed):
        sid, kind, key = parse_artifact_id(aid)
        if kind == "obs" and key == "celltype" and sid is not None:
            s.celltypes_updated.append(sid)
        elif kind == "obsm" and key == "deconv_weights" and sid is not None:
            s.deconv_weights_updated.append(sid)
        elif kind == "uns" and key == "celltype_colors" and sid is not None:
            s.celltype_colors_updated.append(sid)
    for aid in added:
        if aid.startswith("roi:"):
            s.rois_added.append(aid[4:])
    for aid in removed:
        if aid.startswith("roi:"):
            s.rois_deleted.append(aid[4:])
        else:
            sid, kind, key = parse_artifact_id(aid)
            if kind == "obs" and key == "celltype" and sid is not None:
                s.celltypes_updated.append(sid)
    for field_name in ("celltypes_updated", "deconv_weights_updated", "celltype_colors_updated"):
        setattr(s, field_name, sorted(set(getattr(s, field_name))))
    return s


class ProvenanceLedger:
    def __init__(self) -> None:
        self.artifacts: dict[str, Artifact] = {}
        self.log: list[LedgerEntry] = []
        self.version: int = 0

    # ---- queries -------------------------------------------------------------------------------------
    def get(self, aid: str) -> Artifact | None:
        return self.artifacts.get(aid)

    def live(self) -> dict[str, Artifact]:
        return {k: v for k, v in self.artifacts.items() if v.status != "deleted"}

    def has_live(self, aid: str) -> bool:
        a = self.artifacts.get(aid)
        return a is not None and a.status != "deleted"

    def is_stale(self, aid: str) -> bool:
        a = self.artifacts.get(aid)
        return a is not None and a.status == "stale"

    def stale_ids(self) -> list[str]:
        return sorted(k for k, v in self.artifacts.items() if v.status == "stale")

    def live_rois(self) -> set[str]:
        return {v.key for v in self.artifacts.values() if v.kind == "roi" and v.status != "deleted"}

    def dependants(self, aid: str) -> list[str]:
        return sorted(k for k, v in self.artifacts.items() if aid in v.inputs and v.status != "deleted")

    def consistency_hash(self) -> str:
        items = sorted((k, v.version, v.status, v.fingerprint) for k, v in self.artifacts.items())
        return hashlib.sha256(json.dumps(items).encode()).hexdigest()

    # ---- transactions ---------------------------------------------------------------------------------------
    def initialize(self, snapshot: Snapshot, origin: str = "load") -> LedgerDelta:
        """Register everything present at load time as fresh artefacts with no producer."""
        empty = Snapshot(artifacts={})
        return self._apply(diff_snapshots(empty, snapshot), snapshot, origin=origin, producer=None, inputs_for=lambda aid: [])

    def record_step(
        self,
        before: Snapshot,
        after: Snapshot,
        *,
        producer: StepRef,
        reads: Iterable[str] = (),
        roi_reads: Iterable[str] = (),
        any_roi: bool = False,
        declared_consumes: Iterable[str] = (),
        writes: Iterable[str] = (),
        target_slice_ids: Iterable[int] = (),
        origin: str = "agent",
        outcome: str = "ok",
    ) -> LedgerDelta:
        """Record an agent run. ``reads`` are unqualified ``kind:key`` refs from the code
        scan (qualified per target slice); ``roi_reads`` are ROI names, and ``any_roi`` says the code
        read a region the scan could not name, so every region of the target slices is an input (one
        dependency too many costs a recomputation, one too few leaves a stale result looking valid);
        ``declared_consumes`` are unqualified refs from the skill contract; ``writes`` are unqualified refs the code
        assigned on the slice itself (a write on a copy, a subset or a pooled object must not be
        passed: it would clear a stale flag the slice still deserves). Only the snapshot diff
        decides what changed. A key the code assigned whose values
        did not change is not a change (it is most often a write to a copy or a subset, or a
        library call with a custom key): it is never reported as updated and never marks
        dependants stale. The one thing it does is clear a stale flag on that key, so that
        re-running a producer that reproduces identical values counts as re-produced."""
        diff = diff_snapshots(before, after)
        reproduced: list[str] = []
        for ref in set(writes):
            for sid in (list(target_slice_ids) or sorted({a.slice_id for a in before.artifacts.values() if a.slice_id is not None})):
                qid = f"s{sid}:{ref}"
                if qid in after.artifacts and qid not in diff.added and qid not in diff.changed and self.artifacts.get(qid) is not None and self.artifacts[qid].status == "stale":
                    reproduced.append(qid)
        targets = list(target_slice_ids) or sorted({parse_artifact_id(a)[0] for a in diff.touched if parse_artifact_id(a)[0] is not None})
        candidates: set[str] = set()
        for ref in set(reads) | set(declared_consumes):
            for sid in targets:
                qid = f"s{sid}:{ref}"
                if qid in before.artifacts or qid in after.artifacts:
                    candidates.add(qid)
        for name in roi_reads:
            candidates.add(f"roi:{name}")
        if any_roi:
            candidates.update(aid for aid, fp in before.artifacts.items() if fp.kind == "roi" and fp.slice_id in targets)

        def inputs_for(aid: str) -> list[str]:
            return sorted(c for c in candidates if c != aid and c in before.artifacts and c not in diff.touched)

        return self._apply(diff, after, origin=origin, producer=producer, inputs_for=inputs_for, outcome=outcome, reproduced=reproduced)

    def record_external(self, before: Snapshot, after: Snapshot, origin: str = "gui", label: str | None = None) -> LedgerDelta:
        diff = diff_snapshots(before, after)
        producer = StepRef(mode="external", label=label) if label else None
        return self._apply(diff, after, origin=origin, producer=producer, inputs_for=lambda aid: [])

    def mark_stale(self, aid: str, reason: str) -> list[str]:
        """Manually mark an artefact and its dependants stale (returns the newly stale ids)."""
        newly = self._propagate({aid}, reason, exclude=set())
        a = self.artifacts.get(aid)
        if a and a.status == "fresh":
            a.status, a.stale_reason = "stale", reason
            newly.insert(0, aid)
        if newly:
            self.version += 1
            self.log.append(LedgerEntry(seq=self.version, origin="manual", stale_now=newly))
        return newly

    def _apply(self, diff: SnapshotDiff, after: Snapshot, *, origin: str, producer: StepRef | None, inputs_for, outcome: str = "ok", reproduced: Iterable[str] = ()) -> LedgerDelta:
        now = time.time()
        touched = set(diff.touched)
        stale_cleared: list[str] = []
        if outcome == "ok":
            for aid in reproduced:  # re-written with identical values by a successful run: fresh again, attributed to this run
                a = self.artifacts[aid]
                a.status, a.stale_reason, a.updated_at = "fresh", None, now
                a.produced_by, a.origin, a.inputs = producer, origin, inputs_for(aid)
                stale_cleared.append(aid)
        for aid in diff.added:
            fp = after.artifacts[aid]
            existing = self.artifacts.get(aid)
            version = existing.version + 1 if existing else 1
            self.artifacts[aid] = Artifact(
                id=aid, kind=fp.kind, key=fp.key, slice_id=fp.slice_id, fingerprint=fp.digest, version=version,
                produced_by=producer, inputs=inputs_for(aid), status="fresh", origin=origin, created_at=now, updated_at=now,
                shape=list(fp.shape) if fp.shape else None, dtype=fp.dtype, categories=_categories(fp), columns=_columns(fp),
            )
        replaced_loaded = [aid for aid in diff.changed if origin != "load" and aid in self.artifacts and self.artifacts[aid].origin == "load"]
        for aid in diff.changed:
            fp = after.artifacts[aid]
            a = self.artifacts[aid]
            if a.status == "stale":
                stale_cleared.append(aid)
            a.fingerprint, a.version, a.updated_at = fp.digest, a.version + 1, now
            a.status, a.stale_reason = "fresh", None
            a.shape, a.dtype, a.categories, a.columns = (list(fp.shape) if fp.shape else None), fp.dtype, _categories(fp), _columns(fp)
            if producer is not None or origin != "load":
                a.produced_by, a.origin = producer, origin
                a.inputs = inputs_for(aid)
        for aid in diff.removed:
            a = self.artifacts.get(aid)
            if a:
                a.status, a.version, a.updated_at = "deleted", a.version + 1, now
        reason_src = producer.label or (producer.skill or f"turn {producer.turn_id}") if producer else origin
        partial: list[str] = []
        if outcome != "ok":
            # a run that failed or was stopped may have written part of a result: what it touched is
            # recorded (the data really changed) but never trusted as fresh
            reason = "written by a run that was stopped" if outcome == "cancelled" else "written by a run that failed"
            for aid in list(diff.added) + list(diff.changed):
                self.artifacts[aid].status, self.artifacts[aid].stale_reason = "stale", reason
                partial.append(aid)
            stale_cleared = [aid for aid in stale_cleared if aid not in partial]
        stale_now = partial + self._propagate(set(diff.changed) | set(diff.removed), f"input changed by {reason_src}", exclude=touched)
        if not (diff.added or diff.changed or diff.removed or stale_now or stale_cleared):
            # nothing happened: no log entry, no version bump (a cancelled or read-only run)
            entry = LedgerEntry(seq=self.version, ts=now, origin=origin, producer=producer)
            return LedgerDelta(entry=entry, ledger_version=self.version, summary=summarize_v1([], [], []))
        self.version += 1
        entry = LedgerEntry(seq=self.version, ts=now, origin=origin, producer=producer, added=list(diff.added), changed=list(diff.changed), removed=list(diff.removed), stale_now=stale_now, stale_cleared=stale_cleared, replaced_loaded=replaced_loaded)
        self.log.append(entry)
        return LedgerDelta(entry=entry, ledger_version=self.version, summary=summarize_v1(diff.added, diff.changed, diff.removed))

    def _propagate(self, roots: set[str], reason: str, exclude: set[str]) -> list[str]:
        newly: list[str] = []
        queue = deque(roots)
        seen: set[str] = set(roots)
        while queue:
            src = queue.popleft()
            for dep_id, dep in self.artifacts.items():
                if dep_id in seen or dep_id in exclude or dep.status == "deleted":
                    continue
                if src in dep.inputs:
                    seen.add(dep_id)
                    if dep.status == "fresh":
                        dep.status, dep.stale_reason = "stale", f"{reason} ({src})"
                        newly.append(dep_id)
                    queue.append(dep_id)
        return sorted(newly)

    # ---- rendering -------------------------------------------------------------------------------------------
    def state_text(self, slice_ids: Iterable[int] | None = None, max_keys: int = 25) -> str:
        """Compact per-slice inventory for prompts: every artefact with its producer (skill or code
        and the turn, or the viewer edit) and, when stale, the recorded reason, so a role answering
        "what is out of date and why" or "which step produced this" has the record in front of it."""
        ids = set(slice_ids) if slice_ids is not None else None
        by_slice: dict[int | None, dict[str, list[str]]] = {}
        for a in self.artifacts.values():
            if a.status == "deleted" or a.kind in ("X", "var"):
                continue
            if ids is not None and a.slice_id is not None and a.slice_id not in ids:
                continue
            tag = f" (STALE: {a.stale_reason})" if a.status == "stale" and a.stale_reason else (" (STALE)" if a.status == "stale" else "")
            src = ""
            if a.produced_by is not None:
                p = a.produced_by
                who = p.skill or (f"viewer {p.label}" if p.mode == "external" and p.label else p.label or "code")
                src = f" <- {who}" + (f", turn {p.turn_id}" if p.turn_id else "")
            by_slice.setdefault(a.slice_id, {}).setdefault(a.kind, []).append(f"{a.key}{artifact_spec(a)}{tag}{src}")
        lines: list[str] = []
        for sid in sorted(k for k in by_slice if k is not None):
            lines.append(f"Slice {sid}:")
            for kind in ("obs", "obsm", "obsp", "layers", "uns", "roi"):
                keys = by_slice[sid].get(kind)
                if keys:
                    shown = ", ".join(keys[:max_keys]) + (f" ... (+{len(keys) - max_keys})" if len(keys) > max_keys else "")
                    lines.append(f"  {kind}: {shown}")
        rois = by_slice.get(None, {}).get("roi")
        if rois:
            lines.append("ROIs: " + ", ".join(rois))
        return "\n".join(lines)

    # ---- persistence ---------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "artifacts": {k: v.model_dump() for k, v in self.artifacts.items()}, "log": [e.model_dump() for e in self.log]}

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> ProvenanceLedger:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        led = cls()
        led.version = int(data.get("version", 0))
        led.artifacts = {k: Artifact.model_validate(v) for k, v in data.get("artifacts", {}).items()}
        led.log = [LedgerEntry.model_validate(e) for e in data.get("log", [])]
        return led


__all__ = ["StepRef", "Artifact", "LedgerEntry", "LedgerDelta", "ProvenanceLedger", "summarize_v1"]
