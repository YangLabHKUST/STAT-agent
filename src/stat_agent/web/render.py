"""Rendering helpers for the viewer endpoints (image JPEG, cell overlay, gene expression)."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from stat_agent.core.colors import generate_distinct_colors


def coordinate_extent(adata) -> tuple[int, int]:
    """``(width, height)`` of the coordinate space (cell (x, y) == pixel (x, y))."""
    x = adata.obs["x"].to_numpy(dtype=float)
    y = adata.obs["y"].to_numpy(dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    w = int(np.ceil(x.max())) + 1 if x.size else 1
    h = int(np.ceil(y.max())) + 1 if y.size else 1
    return w, h


def image_payload(slice_obj, quality: int = 75) -> dict:
    image = slice_obj.primary_image
    if image is None:
        # scatter-only slice: the viewer paints its own white background, so no JPEG is built or sent
        w, h = coordinate_extent(slice_obj.adata)
        return {"success": True, "image_data": "", "image_width": w, "image_height": h, "is_scatter": True, "slice_id": slice_obj.slice_id, "modality": slice_obj.modality}
    if image.dtype != np.uint8:
        lo, hi = float(image.min()), float(image.max())
        image = ((image - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(image, dtype=np.uint8)
    pil = Image.fromarray(image) if image.ndim == 3 else Image.fromarray(image, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return {
        "success": True,
        "image_data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "is_scatter": False,
        "slice_id": slice_obj.slice_id,
        "modality": slice_obj.modality,
    }


def overlay_payload(session, slice_obj, selected_celltypes: list[str] | None = None) -> dict:
    adata = slice_obj.adata
    cells = adata.obs
    valid = ~(cells["x"].isna() | cells["y"].isna())
    if not valid.all():
        cells = cells[valid]
    if selected_celltypes and "celltype" in cells.columns:
        cells = cells[cells["celltype"].isin(selected_celltypes)]
    payload: dict = {"x": cells["x"].tolist(), "y": cells["y"].tolist()}
    is_spot = "spot_shape" in adata.uns and "spot_diameter" in adata.uns
    spot_info = None
    if is_spot:
        spot_info = {"spot_shape": str(adata.uns["spot_shape"]), "spot_diameter": float(adata.uns["spot_diameter"]), "has_deconv_weights": bool(adata.uns.get("has_deconv_weights", "deconv_weights" in adata.obsm))}
        if "deconv_weights" in adata.obsm:
            w = adata.obsm["deconv_weights"]
            if hasattr(w, "loc"):
                w = w.loc[cells.index]
                spot_info["deconv_weights"] = {str(c): w[c].tolist() for c in w.columns}
                spot_info["celltype_order"] = [str(c) for c in w.columns]
    has_celltype = "celltype" in adata.obs.columns
    celltypes = None
    colors = None
    if has_celltype:
        payload["celltype"] = [None if (isinstance(v, float) and np.isnan(v)) else str(v) for v in cells["celltype"].tolist()]
        celltypes = slice_obj.celltypes
        existing = slice_obj.get_celltype_colors() or {}
        missing = [ct for ct in celltypes if ct not in existing]
        if missing:
            merged = generate_distinct_colors(celltypes, existing={**(session.celltype_colors or {}), **existing})
            colors = merged  # computed, not written: GETs never mutate the session
        else:
            colors = {ct: existing[ct] for ct in celltypes}
    return {
        "success": True,
        "cells": payload,
        "n_cells": int(len(cells)),
        "total_cells": int(slice_obj.n_obs),
        "has_celltype": bool(has_celltype),
        "celltypes": celltypes,
        "celltype_colors": colors,
        "is_spot_data": bool(is_spot),
        "spot_info": spot_info,
        "slice_id": slice_obj.slice_id,
        "modality": slice_obj.modality,
        "message": None if has_celltype else "No celltype annotations available. You can annotate celltypes using a reference dataset.",
    }


def expression_payload(slice_obj, gene: str) -> dict:
    adata = slice_obj.adata
    if gene not in adata.var_names:
        raise KeyError(gene)
    col = adata[:, gene].X
    expr = np.asarray(col.toarray() if hasattr(col, "toarray") else col).ravel().astype(float)
    finite = np.where(np.isfinite(expr), expr, 0.0)
    return {
        "success": True,
        "gene": gene,
        "cells": {"x": adata.obs["x"].tolist(), "y": adata.obs["y"].tolist(), "expression": finite.tolist()},
        "n_cells": int(adata.n_obs),
        "total_cells": int(slice_obj.n_obs),
        "expression_range": [float(finite.min()) if finite.size else 0.0, float(finite.max()) if finite.size else 0.0],
        "slice_id": slice_obj.slice_id,
    }


def roi_payload(roi) -> dict:
    info = {"name": roi.name, "type": roi.type, "bounds": [float(b) for b in roi.bounds], "slice_id": roi.slice_id, "modality": roi.modality, "n_cells": int(roi.n_obs)}
    if roi.type != "bbox" and hasattr(roi.geometry, "exterior"):
        coords = list(roi.geometry.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        info["vertices"] = [[float(x), float(y)] for x, y in coords]
    if roi.adata is not None and "celltype" in roi.adata.obs.columns:
        info["celltype_distribution"] = {str(k): int(v) for k, v in roi.adata.obs["celltype"].value_counts().items()}
    return info


__all__ = ["image_payload", "overlay_payload", "expression_payload", "roi_payload", "coordinate_extent"]
