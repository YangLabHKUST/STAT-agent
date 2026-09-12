"""DataSlice: one independent unit of spatial data (one sample, one modality).

Invariant kept from STAT 0.1: every slice owns its own AnnData; ``adata.obs['x']`` and
``adata.obs['y']`` are the spatial coordinates and equal image pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np

from stat_agent.core.colors import extract_existing_colors, generate_distinct_colors

MODALITIES = ("gene", "protein")
DATA_LEVELS = ("cell", "spot")


@dataclass
class DataSlice:
    """A single slice of spatial transcriptomics data.

    Attributes
    ----------
    slice_id : int
        Unique identifier within the session.
    modality : str
        'gene' or 'protein'.
    data_level : str
        'cell' (single-cell resolution) or 'spot' (Visium-like spots).
    adata : AnnData
        Expression matrix; ``obs['x']``/``obs['y']`` are required.
    images : dict
        Named images, e.g. ``{'he': array, 'dapi': array}``.
    metadata : dict
        Free-form metadata (tissue_name, data_level_detection, image_match, ...).
    """

    slice_id: int
    modality: str
    data_level: str
    adata: ad.AnnData
    images: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(f"Invalid modality: {self.modality!r}. Must be one of {MODALITIES}")
        if self.data_level not in DATA_LEVELS:
            raise ValueError(f"Invalid data_level: {self.data_level!r}. Must be one of {DATA_LEVELS}")
        if "x" not in self.adata.obs.columns or "y" not in self.adata.obs.columns:
            raise ValueError("AnnData must have 'x' and 'y' columns in obs")

    # ---- type predicates -------------------------------------------------------------
    @property
    def is_gene(self) -> bool:
        return self.modality == "gene"

    @property
    def is_protein(self) -> bool:
        return self.modality == "protein"

    @property
    def is_cell_level(self) -> bool:
        return self.data_level == "cell"

    @property
    def is_spot_level(self) -> bool:
        return self.data_level == "spot"

    # ---- data access -----------------------------------------------------------------
    @property
    def n_obs(self) -> int:
        return int(self.adata.n_obs)

    @property
    def n_vars(self) -> int:
        return int(self.adata.n_vars)

    @property
    def tissue_name(self) -> str:
        return str(self.metadata.get("tissue_name", f"slice_{self.slice_id}"))

    @property
    def primary_image(self) -> np.ndarray | None:
        return next(iter(self.images.values())) if self.images else None

    @property
    def coordinate_range(self) -> dict[str, tuple]:
        return {
            "x": (float(self.adata.obs["x"].min()), float(self.adata.obs["x"].max())),
            "y": (float(self.adata.obs["y"].min()), float(self.adata.obs["y"].max())),
        }

    # ---- annotation checks -----------------------------------------------------------
    def has_celltype(self) -> bool:
        if "celltype" not in self.adata.obs.columns:
            return False
        return bool(self.adata.obs["celltype"].notna().any())

    @property
    def celltypes(self) -> list[str]:
        """Sorted unique string cell-type labels (empty if none)."""
        if not self.has_celltype():
            return []
        values = self.adata.obs["celltype"].dropna().unique()
        return sorted(str(v) for v in values if isinstance(v, str) or not isinstance(v, float))

    def has_niche_labels(self) -> bool:
        if "niche_label" not in self.adata.obs.columns:
            return False
        return bool(self.adata.obs["niche_label"].notna().any())

    def has_deconv_weights(self) -> bool:
        return "deconv_weights" in self.adata.obsm

    def has_celltype_colors(self) -> bool:
        return "celltype_colors" in self.adata.uns

    def get_celltype_colors(self) -> dict[str, str] | None:
        colors = self.adata.uns.get("celltype_colors")
        return dict(colors) if isinstance(colors, dict) else None

    def set_celltype_colors(self, colors: dict[str, str]) -> None:
        self.adata.uns["celltype_colors"] = dict(colors)

    def ensure_celltype_colors(self, existing: dict[str, str] | None = None) -> dict[str, str]:
        """Guarantee every current cell type has a colour; returns the mapping.

        Existing colours (in ``adata.uns`` in dict or scanpy list format, or passed in
        ``existing``) are kept; only missing ones are generated deterministically.
        """
        if not self.has_celltype():
            return {}
        known = extract_existing_colors(self.adata)
        if existing:
            known = {**existing, **known}
        colors = generate_distinct_colors(self.celltypes, existing=known)
        self.set_celltype_colors(colors)
        return colors

    # ---- summaries -------------------------------------------------------------------
    def get_summary(self) -> dict[str, Any]:
        """JSON-serialisable summary used by the session view and the web layer."""
        obs_columns = [str(c) for c in self.adata.obs.columns]
        summary: dict[str, Any] = {
            "slice_id": self.slice_id,
            "modality": self.modality,
            "data_level": self.data_level,
            "data_level_detection": str(self.metadata.get("data_level_detection", "")),
            "n_obs": self.n_obs,
            "n_vars": self.n_vars,
            "has_celltype": self.has_celltype(),
            "celltypes": self.celltypes,
            "has_celltype_colors": self.has_celltype_colors(),
            "has_deconv_weights": self.has_deconv_weights(),
            "image_count": len(self.images),
            "image_names": list(self.images.keys()),
            "image_match": str(self.metadata.get("image_match", "")),
            "tissue_name": self.tissue_name,
            "obs_columns": obs_columns,
            "obsm_keys": [str(k) for k in self.adata.obsm.keys()],
            "uns_keys": [str(k) for k in self.adata.uns.keys()],
            "var_columns": [str(c) for c in self.adata.var.columns],
            "layers": [str(k) for k in self.adata.layers.keys()],
        }
        if self.is_spot_level:
            if "spot_shape" in self.adata.uns:
                summary["spot_shape"] = str(self.adata.uns["spot_shape"])
            if "spot_diameter" in self.adata.uns:
                try:
                    summary["spot_diameter"] = float(self.adata.uns["spot_diameter"])
                except (TypeError, ValueError):
                    pass
        return summary

    def __repr__(self) -> str:
        return (
            f"DataSlice(slice_id={self.slice_id}, modality='{self.modality}', data_level='{self.data_level}', "
            f"n_obs={self.n_obs}, n_vars={self.n_vars}, tissue='{self.tissue_name}')"
        )


__all__ = ["DataSlice", "MODALITIES", "DATA_LEVELS"]
