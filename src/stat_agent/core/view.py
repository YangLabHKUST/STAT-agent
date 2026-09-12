"""Typed, read-only views of a session for the agent roles and the web layer.

``SessionSummary`` is the *only* form of the session that LLM roles see; it is a plain
pydantic model, so roles can be exercised from JSON fixtures without any AnnData.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field


class SliceSummary(BaseModel):
    slice_id: int
    tissue_name: str = ""
    modality: str
    data_level: str
    data_level_detection: str = ""
    n_obs: int = 0
    n_vars: int = 0
    has_celltype: bool = False
    celltypes: list[str] = Field(default_factory=list)
    has_celltype_colors: bool = False
    has_deconv_weights: bool = False
    image_names: list[str] = Field(default_factory=list)
    image_match: str = ""
    obs_columns: list[str] = Field(default_factory=list)
    obsm_keys: list[str] = Field(default_factory=list)
    uns_keys: list[str] = Field(default_factory=list)
    var_columns: list[str] = Field(default_factory=list)
    layers: list[str] = Field(default_factory=list)
    spot_shape: str | None = None
    spot_diameter: float | None = None

    @property
    def n_celltypes(self) -> int:
        return len(self.celltypes)

    def has_key(self, kind: str, key: str) -> bool:
        pool = {
            "obs": self.obs_columns,
            "obsm": self.obsm_keys,
            "uns": self.uns_keys,
            "var": self.var_columns,
            "layers": self.layers,
        }.get(kind, [])
        return key in pool

    def describe(self, verbose: bool = False) -> str:
        """One-line description in the style the STAT prompts have always used."""
        unit = "genes" if self.modality == "gene" else "proteins"
        desc = (
            f"Slice {self.slice_id}: {self.tissue_name} ({self.modality}, {self.data_level}-level, "
            f"{self.n_obs:,} observations, {self.n_vars:,} {unit}"
        )
        if self.has_celltype:
            desc += f", {self.n_celltypes} cell types"
        else:
            desc += ", NO celltype annotations"
        if self.has_deconv_weights:
            desc += ", deconvolution weights available"
        if self.image_names:
            desc += f", images: {', '.join(self.image_names)}"
        desc += ")"
        if verbose:
            if self.obs_columns:
                cols = self.obs_columns[:15]
                more = f" ... ({len(self.obs_columns)} total)" if len(self.obs_columns) > 15 else ""
                desc += f"\n    obs columns: {', '.join(cols)}{more}"
            if self.obsm_keys:
                desc += f"\n    obsm keys: {', '.join(self.obsm_keys[:10])}"
            if self.uns_keys:
                desc += f"\n    uns keys: {', '.join(self.uns_keys[:15])}"
        return desc


class ROISummary(BaseModel):
    name: str
    slice_id: int | None = None
    modality: str | None = None
    type: str = "polygon"
    n_obs: int = 0
    bounds: tuple[float, float, float, float] | None = None

    def describe(self) -> str:
        return f"'{self.name}' on slice {self.slice_id} ({self.n_obs:,} cells, {self.type})"


class SessionSummary(BaseModel):
    """Typed snapshot of what is loaded. Construct with :meth:`from_session`."""

    name: str = "session"
    n_slices: int = 0
    slice_ids: list[int] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    data_levels: list[str] = Field(default_factory=list)
    data_format: str = "single_slice"
    slices: list[SliceSummary] = Field(default_factory=list)
    rois: list[ROISummary] = Field(default_factory=list)

    @classmethod
    def from_session(cls, session) -> SessionSummary:
        return cls.model_validate(session.get_summary())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSummary:
        return cls.model_validate(data)

    @property
    def has_data(self) -> bool:
        return self.n_slices > 0

    def slice(self, slice_id: int) -> SliceSummary | None:
        for s in self.slices:
            if s.slice_id == slice_id:
                return s
        return None

    def has_slice(self, slice_id: int) -> bool:
        return self.slice(slice_id) is not None

    def rois_on(self, slice_ids: Iterable[int]) -> list[ROISummary]:
        ids = set(slice_ids)
        return [r for r in self.rois if r.slice_id in ids]

    def describe(self, target_slice_ids: Iterable[int] | None = None, verbose: bool = False) -> str:
        """Multi-line text block for prompts (all slices, or only the targets)."""
        ids = set(target_slice_ids) if target_slice_ids is not None else set(self.slice_ids)
        lines = [f"- Total slices: {self.n_slices}"]
        for s in self.slices:
            if s.slice_id in ids:
                lines.append("  - " + s.describe(verbose=verbose))
        rois = self.rois_on(ids)
        if rois:
            lines.append("Available ROIs:" if target_slice_ids is None else "Available ROIs on target slices:")
            for r in rois:
                lines.append("  - " + r.describe())
        return "\n".join(lines)


class InterfaceState(BaseModel):
    """What the user is looking at in the viewer; sent by the browser with each turn."""

    current_slice_id: int | None = None
    current_modality: str | None = None
    viz_mode: str | None = None
    selected_gene: str | None = None
    highlighted_roi: str | None = None
    selected_celltypes: list[str] | None = None

    def describe(self) -> str:
        parts = []
        if self.current_slice_id is not None:
            parts.append(f"viewing slice {self.current_slice_id}")
        if self.current_modality:
            parts.append(f"{self.current_modality} modality")
        if self.viz_mode:
            parts.append(f"{self.viz_mode} view")
        if self.selected_gene:
            parts.append(f"gene {self.selected_gene} displayed")
        if self.highlighted_roi:
            parts.append(f"ROI '{self.highlighted_roi}' highlighted")
        return ", ".join(parts)


__all__ = ["SliceSummary", "ROISummary", "SessionSummary", "InterfaceState"]
