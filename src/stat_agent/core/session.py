"""SimpleSession: the single source of truth for loaded spatial data.

Design rules (unchanged from STAT 0.1, tightened here):

* every slice is explicit: ``session.get_slice(slice_id).adata`` - there is **no**
  "current slice" in the session; that is purely interface state;
* ROIs live in exactly one store (``session.roi_manager``); ``session.roi_subsets`` is
  the same mapping, kept for the skills that use it;
* every mutation from the GUI or the agent runtime happens under ``session.lock``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from stat_agent.core.colors import extract_existing_colors, generate_distinct_colors
from stat_agent.core.data_slice import DataSlice
from stat_agent.core.lock import SessionLock
from stat_agent.core.roi_manager import ROI, ROIManager

logger = logging.getLogger(__name__)


def filter_adata_by_roi(adata: ad.AnnData, roi_def: dict[str, Any]) -> ad.AnnData:
    """Return a copy of ``adata`` restricted to observations inside the ROI definition."""
    roi_type = roi_def.get("type", "bbox")
    x = adata.obs["x"].to_numpy(dtype=float)
    y = adata.obs["y"].to_numpy(dtype=float)
    if roi_type == "bbox":
        mask = (x >= roi_def["x_min"]) & (x <= roi_def["x_max"]) & (y >= roi_def["y_min"]) & (y <= roi_def["y_max"])
    elif roi_type == "circle":
        d = np.sqrt((x - roi_def["center_x"]) ** 2 + (y - roi_def["center_y"]) ** 2)
        mask = d <= roi_def["radius"]
    elif roi_type == "polygon":
        from matplotlib.path import Path as MplPath

        vertices = [tuple(v) for v in roi_def["vertices"]]
        mask = MplPath(vertices).contains_points(np.column_stack([x, y]))
    else:
        raise ValueError(f"Unknown ROI type: {roi_type}")
    return adata[np.asarray(mask, dtype=bool)].copy()


class SimpleSession:
    """Session holding slices, ROIs and shared colours.

    Parameters
    ----------
    name : str
        Session name.
    workdir : path, optional
        Directory the runtime may write into (logs, figures, session saves).
    """

    def __init__(self, name: str = "session", workdir: str | Path | None = None):
        self.name = name
        self.slices: dict[int, DataSlice] = {}
        self.roi_manager = ROIManager()
        self.celltype_colors: dict[str, str] | None = None
        self.lock = SessionLock()
        self.workdir: Path | None = Path(workdir) if workdir else None
        self.metadata: dict[str, Any] = {"created_at": datetime.now().isoformat(), "name": name}
        # provider, model, api_key and base_url for skill code that calls a model itself (the 0.1
        # convention: ``session.llm_config`` + ``stat_agent.agent.llm_backend.LLMBackend``); the agent fills it
        self.llm_config: dict[str, Any] | None = None
        logger.info("Created session: %s", name)

    # ---- slices ----------------------------------------------------------------------
    def get_slice(self, slice_id: int) -> DataSlice:
        try:
            return self.slices[int(slice_id)]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Slice {slice_id!r} not found. Available: {self.slice_ids}") from None

    def has_slice(self, slice_id: int) -> bool:
        try:
            return int(slice_id) in self.slices
        except (TypeError, ValueError):
            return False

    def iter_slices(self, modality: str | None = None, data_level: str | None = None) -> Iterator[DataSlice]:
        for sid in self.slice_ids:
            s = self.slices[sid]
            if modality is not None and s.modality != modality:
                continue
            if data_level is not None and s.data_level != data_level:
                continue
            yield s

    def add_slice(self, slice: DataSlice) -> None:
        if slice.slice_id in self.slices:
            logger.warning("Overwriting existing slice %s", slice.slice_id)
        self.slices[slice.slice_id] = slice
        logger.info("Added slice %s (%s, %s)", slice.slice_id, slice.modality, slice.data_level)

    def remove_slice(self, slice_id: int) -> bool:
        if slice_id not in self.slices:
            return False
        del self.slices[slice_id]
        for name in [n for n, r in self.roi_manager.rois.items() if r.slice_id == slice_id]:
            self.roi_manager.remove_roi(name)
        return True

    # ---- loading -----------------------------------------------------------------------
    def load_dataset(self, dataset_dir: str | Path, options=None) -> list[int]:
        """Load every h5ad (+ images) in ``dataset_dir``; returns the loaded slice ids."""
        from stat_agent.io.dataset import load_dataset

        dataset_dir = Path(dataset_dir)
        loaded = load_dataset(dataset_dir, options=options)
        for slice_data in loaded.values():
            self.add_slice(slice_data)
        self.initialize_celltype_colors()
        # h5ad files in subdirectories are not slices (the loader reads the top level only) but they are
        # often what a prerequisite asks for (a reference dataset); remember them for the roles
        extra = sorted(str(p.relative_to(dataset_dir)) for p in dataset_dir.rglob("*.h5ad") if p.parent != dataset_dir and len(p.relative_to(dataset_dir).parts) <= 3)
        self.metadata.update(
            {"dataset_dir": str(dataset_dir), "n_slices": len(self.slices), "slice_ids": self.slice_ids, "extra_files": extra}
        )
        logger.info("Loaded dataset with %d slice(s): %s", len(self.slices), self.slice_ids)
        return sorted(loaded.keys())

    # ---- ROIs --------------------------------------------------------------------------
    @property
    def roi_subsets(self) -> dict[str, ROI]:
        """The ROI store (same mapping as ``roi_manager.rois``)."""
        return self.roi_manager.rois

    def create_roi(self, roi_name: str, slice_id: int, roi_definition: dict[str, Any], overwrite: bool = False) -> ROI:
        """Create an ROI on ``slice_id`` from a geometry definition.

        ``roi_definition``: ``{'type': 'bbox', 'x_min', 'x_max', 'y_min', 'y_max'}``,
        ``{'type': 'circle', 'center_x', 'center_y', 'radius'}`` or
        ``{'type': 'polygon', 'vertices': [(x, y), ...]}``.
        """
        slice_obj = self.get_slice(slice_id)
        roi_type = roi_definition.get("type", "bbox")
        if roi_type == "bbox":
            roi = self.roi_manager.add_bbox_roi(
                name=roi_name,
                min_x=float(roi_definition["x_min"]),
                min_y=float(roi_definition["y_min"]),
                max_x=float(roi_definition["x_max"]),
                max_y=float(roi_definition["y_max"]),
                slice_id=slice_obj.slice_id,
                modality=slice_obj.modality,
                overwrite=overwrite,
            )
        elif roi_type == "circle":
            roi = self.roi_manager.add_circle_roi(
                name=roi_name,
                center_x=float(roi_definition["center_x"]),
                center_y=float(roi_definition["center_y"]),
                radius=float(roi_definition["radius"]),
                slice_id=slice_obj.slice_id,
                modality=slice_obj.modality,
                overwrite=overwrite,
            )
        elif roi_type == "polygon":
            from shapely.geometry import Polygon

            vertices = [tuple(float(c) for c in v) for v in roi_definition["vertices"]]
            if len(vertices) < 3:
                raise ValueError("polygon ROI needs at least 3 vertices")
            roi = self.roi_manager.add_roi(
                name=roi_name,
                geometry=Polygon(vertices),
                slice_id=slice_obj.slice_id,
                modality=slice_obj.modality,
                metadata={"type": "polygon"},
                overwrite=overwrite,
            )
        else:
            raise ValueError(f"Unknown ROI type: {roi_type}")
        roi.adata = filter_adata_by_roi(slice_obj.adata, roi_definition)
        roi.metadata["definition"] = _jsonable_definition(roi_definition)
        logger.info("Created ROI '%s' on slice %s: %d cells/spots", roi_name, slice_id, roi.adata.n_obs)
        return roi

    def get_roi(self, roi_name: str) -> ROI | None:
        return self.roi_manager.rois.get(roi_name)

    def delete_roi(self, roi_name: str) -> bool:
        return self.roi_manager.remove_roi(roi_name)

    def rename_roi(self, old_name: str, new_name: str) -> ROI:
        return self.roi_manager.rename_roi(old_name, new_name)

    def get_roi_data(self, roi_name: str, slice_id: int | None = None) -> ad.AnnData | None:
        roi = self.get_roi(roi_name)
        if roi is None or (slice_id is not None and roi.slice_id != slice_id):
            return None
        return roi.adata

    # ---- colours -------------------------------------------------------------------------
    def initialize_celltype_colors(self) -> dict[str, str]:
        """Give every cell type across all slices a consistent colour.

        Pre-existing colours stored in ``adata.uns['celltype_colors']`` are respected and
        only missing ones are generated (deterministically)."""
        if not self.slices:
            return {}
        all_celltypes: set[str] = set()
        existing: dict[str, str] = dict(self.celltype_colors or {})
        for s in self.slices.values():
            if s.has_celltype():
                all_celltypes.update(s.celltypes)
                for ct, color in extract_existing_colors(s.adata).items():
                    existing.setdefault(ct, color)
        if not all_celltypes:
            return {}
        colors = generate_distinct_colors(all_celltypes, existing=existing)
        self.celltype_colors = colors
        for s in self.slices.values():
            if s.has_celltype():
                s.set_celltype_colors({ct: colors[ct] for ct in s.celltypes if ct in colors})
        logger.info("Initialized colours for %d cell types across %d slices", len(colors), len(self.slices))
        return colors

    _initialize_celltype_colors = initialize_celltype_colors  # backward-compatible alias

    # ---- properties ------------------------------------------------------------------
    @property
    def has_data(self) -> bool:
        return len(self.slices) > 0

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def slice_ids(self) -> list[int]:
        return sorted(self.slices.keys())

    @property
    def modalities(self) -> list[str]:
        return sorted({s.modality for s in self.slices.values()})

    @property
    def data_levels(self) -> list[str]:
        return sorted({s.data_level for s in self.slices.values()})

    @property
    def data_format(self) -> str:
        multi = self.n_slices > 1
        protein = "protein" in self.modalities
        if multi and protein:
            return "multi_slice_multi_omics"
        if multi:
            return "multi_slice"
        if protein:
            return "multi_omics"
        return "single_slice"

    def get_slices_by_modality(self, modality: str) -> list[DataSlice]:
        return list(self.iter_slices(modality=modality))

    def get_slices_by_level(self, data_level: str) -> list[DataSlice]:
        return list(self.iter_slices(data_level=data_level))

    # ---- summaries -----------------------------------------------------------------------
    def get_summary(self) -> dict[str, Any]:
        """Plain-dict summary of slices and ROIs (the typed form is ``SessionSummary``)."""
        return {
            "name": self.name,
            "n_slices": self.n_slices,
            "slice_ids": self.slice_ids,
            "modalities": self.modalities,
            "data_levels": self.data_levels,
            "data_format": self.data_format,
            "slices": [self.slices[sid].get_summary() for sid in self.slice_ids],
            "dataset_dir": self.metadata.get("dataset_dir"),
            "extra_files": list(self.metadata.get("extra_files") or []),
            "n_rois": len(self.roi_manager),
            "rois": [
                {
                    "name": name,
                    "slice_id": roi.slice_id,
                    "modality": roi.modality,
                    "type": roi.type,
                    "n_obs": roi.n_obs,
                    "bounds": list(roi.bounds),
                }
                for name, roi in self.roi_manager.rois.items()
            ],
        }

    def get_frontend_summary(self) -> dict[str, Any]:
        """Summary for the web interface (no interface state: slice selection is client-side)."""
        summary = self.get_summary()
        all_celltypes: set[str] = set()
        for s in self.slices.values():
            all_celltypes.update(s.celltypes)
        summary.update(
            {
                "has_celltype_colors": self.celltype_colors is not None,
                "has_protein": "protein" in self.modalities,
                "available_modalities": self.modalities,
                "available_slices": [
                    {
                        "slice_id": s["slice_id"],
                        "tissue_name": s["tissue_name"],
                        "modality": s["modality"],
                        "data_level": s["data_level"],
                        "data_level_detection": s["data_level_detection"],
                        "image_match": s["image_match"],
                        "n_obs": s["n_obs"],
                        "n_vars": s["n_vars"],
                    }
                    for s in summary["slices"]
                ],
                "celltypes": sorted(all_celltypes),
            }
        )
        return summary

    def __repr__(self) -> str:
        return f"SimpleSession(name='{self.name}', n_slices={self.n_slices}, slice_ids={self.slice_ids})"


def _jsonable_definition(roi_def: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in roi_def.items():
        if k == "vertices":
            out[k] = [[float(c) for c in vertex] for vertex in v]
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


__all__ = ["SimpleSession", "filter_adata_by_roi"]
