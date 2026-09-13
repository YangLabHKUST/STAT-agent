"""SessionMutator: the one door for GUI-side changes (ROI edits, colour/annotation edits).

Every mutation takes the session lock, snapshots before/after, records the change in the
ledger with ``origin='gui'`` and emits a ``provenance_update`` - exactly what an agent run
does, so the canvas and the agent always see the same history.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from stat_agent.core.lock import SessionBusy
from stat_agent.events.catalogue import ProvenanceUpdate
from stat_agent.ledger.provenance import LedgerDelta, ProvenanceLedger
from stat_agent.runtime.snapshot import take_snapshot


class MutationResult:
    def __init__(self) -> None:
        self.delta: LedgerDelta | None = None
        self.event: ProvenanceUpdate | None = None
        self.renamed: list[list[str]] = []
        self.value: Any = None


class SessionMutator:
    def __init__(self, session, ledger: ProvenanceLedger, emitter: Any = None, lock_timeout_s: float = 2.0) -> None:
        self.session = session
        self.ledger = ledger
        self.emitter = emitter
        self.lock_timeout_s = lock_timeout_s

    @contextmanager
    def mutation(self, label: str, origin: str = "gui", timeout: float | None = None) -> Iterator[MutationResult]:
        res = MutationResult()
        with self.session.lock.hold(holder=f"gui:{label}", timeout=self.lock_timeout_s if timeout is None else timeout):
            before = take_snapshot(self.session)
            yield res
            after = take_snapshot(self.session)
            res.delta = self.ledger.record_external(before, after, origin=origin, label=label)
            res.event = res.delta.to_event(origin=origin, renamed=res.renamed or None)  # type: ignore[arg-type]
        if self.emitter is not None and res.event is not None:
            self.emitter.emit(res.event)

    # ---- convenience wrappers used by the web routes and SessionBench ---------------------------------------
    def create_roi(self, name: str, slice_id: int, definition: dict, overwrite: bool = False) -> MutationResult:
        with self.mutation("roi_add") as m:
            m.value = self.session.create_roi(name, slice_id, definition, overwrite=overwrite)
        return m

    def delete_roi(self, name: str) -> MutationResult:
        with self.mutation("roi_delete") as m:
            m.value = self.session.delete_roi(name)
        return m

    def rename_roi(self, old: str, new: str) -> MutationResult:
        with self.mutation("roi_rename") as m:
            m.value = self.session.rename_roi(old, new)
            m.renamed = [[old, new]]
        return m

    def set_celltype_colors(self, slice_id: int, colors: dict) -> MutationResult:
        with self.mutation("celltype_colors") as m:
            s = self.session.get_slice(slice_id)
            merged = dict(s.get_celltype_colors() or {})
            merged.update(colors)
            s.set_celltype_colors(merged)
            if self.session.celltype_colors is not None:
                self.session.celltype_colors.update(colors)
            m.value = merged
        return m

    def rename_celltype(self, slice_id: int, old: str, new: str) -> MutationResult:
        """Rename the cell type ``old`` on one slice. When ``new`` is already a cell type there, the
        cells labelled ``old`` are merged into it (the old label disappears and the target keeps its
        colour). ``old`` must exist: otherwise a KeyError is raised under the lock and nothing is
        changed, recorded or emitted."""
        with self.mutation(f"celltype_rename {old!r} -> {new!r}") as m:  # the label is the recorded reason: it names the change
            s = self.session.get_slice(slice_id)
            col = s.adata.obs["celltype"]
            present = set(s.celltypes) | (set(col.cat.categories) if hasattr(col, "cat") else set())
            if old not in present:
                raise KeyError(old)
            if new != old:
                if hasattr(col, "cat"):
                    if new in col.cat.categories:  # merge: relabel the cells, then drop the old category
                        s.adata.obs["celltype"] = col.where(col != old, new).cat.remove_categories([old])
                    else:
                        s.adata.obs["celltype"] = col.cat.rename_categories({old: new})
                else:
                    s.adata.obs["celltype"] = col.replace({old: new})
                colors = s.get_celltype_colors() or {}
                if old in colors:
                    color = colors.pop(old)
                    colors.setdefault(new, color)  # a merge keeps the target's colour
                    s.set_celltype_colors(colors)
            m.value = new
        return m

    def set_column(self, slice_id: int, column: str, values, label: str = "column_edit") -> MutationResult:
        with self.mutation(label) as m:
            self.session.get_slice(slice_id).adata.obs[column] = values
            m.value = column
        return m


__all__ = ["SessionMutator", "MutationResult", "SessionBusy"]
