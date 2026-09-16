"""Regions of interest (ROIs) in data coordinates.

An ROI is a named shapely geometry pinned to one slice (and therefore one modality),
optionally carrying the AnnData subset of observations inside it. ROIs support
algebra (union / intersection / difference) and JSON (de)serialisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon
from shapely.geometry import box as Box
from shapely.geometry import shape as shape_from_geo
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

Geometry = Polygon | Point


@dataclass
class ROI:
    """A region of interest.

    ``type`` is 'bbox', 'circle' or 'polygon' (free-hand strokes are polygons)."""

    name: str
    geometry: Geometry
    slice_id: int | None = None
    modality: str | None = None
    coordinate_system: str = "global"
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # the slice whose current data the region reads, and the member cells, fixed when the region is drawn
    _source: Any = field(default=None, init=False, repr=False, compare=False)
    _members: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    # a stored subset, only for a region built without a source slice
    _snapshot: ad.AnnData | None = field(default=None, init=False, repr=False, compare=False)

    def bind(self, source: Any, obs_names) -> None:
        """Attach the slice (any object with a current ``adata``) and the names of the member cells."""
        self._source = source
        self._members = np.asarray(list(obs_names), dtype=object)
        self._snapshot = None

    def _member_index(self) -> np.ndarray | None:
        """Positions of the member cells in the source's current data (members no longer present are dropped)."""
        src = getattr(self._source, "adata", None) if self._source is not None else None
        if src is None or self._members is None:
            return None
        idx = src.obs_names.get_indexer(self._members)
        return idx[idx >= 0]

    @property
    def adata(self) -> ad.AnnData | None:
        """The region's cells as they are now: a view of the current slice data, so a column added to the
        slice after the region was drawn is visible through it. Copy it before modifying it in place."""
        idx = self._member_index()
        if idx is None:
            return self._snapshot
        return self._source.adata[idx]

    @adata.setter
    def adata(self, value: ad.AnnData | None) -> None:
        """With a source slice the subset only redefines the membership; without one it is stored as is."""
        if self._source is not None and value is not None:
            self._members = np.asarray(list(value.obs_names), dtype=object)
        else:
            self._snapshot = value

    @property
    def n_obs(self) -> int:
        idx = self._member_index()
        if idx is not None:
            return int(len(idx))
        return int(self._snapshot.n_obs) if self._snapshot is not None else 0

    @property
    def n_vars(self) -> int:
        src = getattr(self._source, "adata", None) if self._source is not None else None
        if src is not None and self._members is not None:
            return int(src.n_vars)
        return int(self._snapshot.n_vars) if self._snapshot is not None else 0

    @property
    def area(self) -> float:
        return float(self.geometry.area)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.geometry.bounds)  # type: ignore[return-value]

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.geometry.centroid
        return (float(c.x), float(c.y))

    @property
    def type(self) -> str:
        return str(self.metadata.get("type", self.geometry.geom_type.lower()))

    @property
    def obs_names(self) -> list[str]:
        idx = self._member_index()
        if idx is not None:
            return [str(n) for n in self._source.adata.obs_names[idx]]
        return [str(n) for n in self._snapshot.obs_names] if self._snapshot is not None else []

    def contains_point(self, x: float, y: float) -> bool:
        return bool(self.geometry.contains(Point(x, y)))

    def to_dict(self) -> dict:
        """Serialise without the AnnData subset (recomputed on load)."""
        return {
            "name": self.name,
            "geometry": self.geometry.__geo_interface__,
            "slice_id": self.slice_id,
            "modality": self.modality,
            "coordinate_system": self.coordinate_system,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "n_obs": self.n_obs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ROI:
        return cls(
            name=data["name"],
            geometry=shape_from_geo(data["geometry"]),
            slice_id=data.get("slice_id"),
            modality=data.get("modality"),
            coordinate_system=data.get("coordinate_system", "global"),
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )

    def __repr__(self) -> str:
        slice_info = f", slice_id={self.slice_id}" if self.slice_id is not None else ""
        modality_info = f", modality='{self.modality}'" if self.modality else ""
        return (
            f"ROI(name='{self.name}', type={self.type}, area={self.area:.2f}, n_obs={self.n_obs}"
            f"{slice_info}{modality_info})"
        )


class ROIManager:
    """Named ROI store with algebra and export helpers."""

    def __init__(self) -> None:
        self.rois: dict[str, ROI] = {}

    # ---- creation --------------------------------------------------------------------
    def add_roi(
        self,
        name: str,
        geometry: Geometry | list[tuple[float, float]],
        slice_id: int | None = None,
        modality: str | None = None,
        coordinate_system: str = "global",
        metadata: dict | None = None,
        overwrite: bool = False,
    ) -> ROI:
        if not name or not isinstance(name, str):
            raise ValueError("ROI name must be a non-empty string")
        if name in self.rois and not overwrite:
            raise ValueError(f"ROI '{name}' already exists. Use overwrite=True to replace.")
        if isinstance(geometry, list):
            geometry = Polygon(geometry)
        roi = ROI(
            name=name,
            geometry=geometry,
            slice_id=slice_id,
            modality=modality,
            coordinate_system=coordinate_system,
            metadata=dict(metadata or {}),
        )
        self.rois[name] = roi
        logger.info("Added %s", roi)
        return roi

    def add_circle_roi(
        self,
        name: str,
        center: tuple[float, float] | None = None,
        radius: float | None = None,
        *,
        center_x: float | None = None,
        center_y: float | None = None,
        coordinate_system: str = "global",
        n_points: int = 64,
        **kwargs,
    ) -> ROI:
        """Circle approximated by an ``n_points`` polygon. Accepts ``center=(x, y)`` or
        ``center_x``/``center_y``."""
        if center is None:
            if center_x is None or center_y is None:
                raise ValueError("add_circle_roi needs center=(x, y) or center_x/center_y")
            center = (float(center_x), float(center_y))
        if radius is None:
            raise ValueError("add_circle_roi needs a radius")
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        coords = list(zip(center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
        return self.add_roi(
            name=name,
            geometry=Polygon(coords),
            coordinate_system=coordinate_system,
            metadata={"type": "circle", "center": [float(center[0]), float(center[1])], "radius": float(radius)},
            **kwargs,
        )

    def add_bbox_roi(
        self,
        name: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        coordinate_system: str = "global",
        **kwargs,
    ) -> ROI:
        return self.add_roi(
            name=name,
            geometry=Box(min_x, min_y, max_x, max_y),
            coordinate_system=coordinate_system,
            metadata={"type": "bbox"},
            **kwargs,
        )

    # ---- access ------------------------------------------------------------------------
    def get_roi(self, name: str) -> ROI:
        if name not in self.rois:
            raise KeyError(f"ROI '{name}' not found. Available: {list(self.rois.keys())}")
        return self.rois[name]

    def remove_roi(self, name: str) -> bool:
        if name in self.rois:
            del self.rois[name]
            logger.info("Removed ROI: %s", name)
            return True
        logger.warning("ROI '%s' not found", name)
        return False

    def rename_roi(self, old_name: str, new_name: str) -> ROI:
        if old_name not in self.rois:
            raise KeyError(f"ROI '{old_name}' not found")
        if not new_name or not isinstance(new_name, str):
            raise ValueError("new ROI name must be a non-empty string")
        if new_name in self.rois and new_name != old_name:
            raise ValueError(f"ROI '{new_name}' already exists")
        roi = self.rois.pop(old_name)
        roi.name = new_name
        self.rois[new_name] = roi
        return roi

    def get_rois_for_slice(self, slice_id: int | None = None, modality: str | None = None) -> dict[str, ROI]:
        out = {}
        for name, roi in self.rois.items():
            if slice_id is not None and roi.slice_id != slice_id:
                continue
            if modality is not None and roi.modality != modality:
                continue
            out[name] = roi
        return out

    # ---- algebra -----------------------------------------------------------------------
    def _pair(self, name1: str, name2: str) -> tuple[ROI, ROI]:
        roi1, roi2 = self.get_roi(name1), self.get_roi(name2)
        if roi1.coordinate_system != roi2.coordinate_system:
            raise ValueError(
                f"ROIs must be in the same coordinate system ({roi1.coordinate_system} vs {roi2.coordinate_system})"
            )
        if roi1.slice_id != roi2.slice_id:
            raise ValueError(f"ROIs must be on the same slice ({roi1.slice_id} vs {roi2.slice_id})")
        return roi1, roi2

    def union(self, name1: str, name2: str, new_name: str) -> ROI:
        roi1, roi2 = self._pair(name1, name2)
        return self.add_roi(
            new_name,
            unary_union([roi1.geometry, roi2.geometry]),
            slice_id=roi1.slice_id,
            modality=roi1.modality,
            coordinate_system=roi1.coordinate_system,
            metadata={"type": "polygon", "operation": "union", "sources": [name1, name2]},
        )

    def intersection(self, name1: str, name2: str, new_name: str) -> ROI:
        roi1, roi2 = self._pair(name1, name2)
        return self.add_roi(
            new_name,
            roi1.geometry.intersection(roi2.geometry),
            slice_id=roi1.slice_id,
            modality=roi1.modality,
            coordinate_system=roi1.coordinate_system,
            metadata={"type": "polygon", "operation": "intersection", "sources": [name1, name2]},
        )

    def difference(self, name1: str, name2: str, new_name: str) -> ROI:
        roi1, roi2 = self._pair(name1, name2)
        return self.add_roi(
            new_name,
            roi1.geometry.difference(roi2.geometry),
            slice_id=roi1.slice_id,
            modality=roi1.modality,
            coordinate_system=roi1.coordinate_system,
            metadata={"type": "polygon", "operation": "difference", "sources": [name1, name2]},
        )

    # ---- export --------------------------------------------------------------------------
    def to_geodataframe(self, coordinate_system: str | None = None) -> GeoDataFrame:
        rois = list(self.rois.values())
        if coordinate_system:
            rois = [r for r in rois if r.coordinate_system == coordinate_system]
        if not rois:
            return gpd.GeoDataFrame()
        return gpd.GeoDataFrame(
            {
                "name": [r.name for r in rois],
                "slice_id": [r.slice_id for r in rois],
                "geometry": [r.geometry for r in rois],
                "coordinate_system": [r.coordinate_system for r in rois],
                "area": [r.area for r in rois],
                "n_obs": [r.n_obs for r in rois],
                "created_at": [r.created_at for r in rois],
            },
            geometry="geometry",
        )

    def clear(self) -> None:
        self.rois.clear()

    def __len__(self) -> int:
        return len(self.rois)

    def __contains__(self, name: object) -> bool:
        return name in self.rois

    def __repr__(self) -> str:
        return f"ROIManager(n_rois={len(self.rois)}, names={list(self.rois.keys())})"


__all__ = ["ROI", "ROIManager"]
