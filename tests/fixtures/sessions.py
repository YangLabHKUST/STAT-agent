"""Synthetic, seeded, in-memory sessions for tests (no binary fixtures in git)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from stat_agent.core import DataSlice, SimpleSession

CELLTYPES = ["T cell", "B cell", "Tumor", "Fibroblast"]


def make_adata(
    n_obs: int = 600,
    n_vars: int = 40,
    seed: int = 0,
    celltypes: Iterable[str] | None = CELLTYPES,
    coord_range: float = 100.0,
    spot: bool = False,
    deconv: bool = False,
    prefix: str = "G",
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.5, size=(n_obs, n_vars)).astype(np.float32)
    var_names = [f"{prefix}{i}" for i in range(n_vars)]
    obs = pd.DataFrame(index=[f"obs_{i}" for i in range(n_obs)])
    obs["x"] = rng.uniform(0, coord_range, n_obs)
    obs["y"] = rng.uniform(0, coord_range, n_obs)
    if celltypes:
        cts = list(celltypes)
        obs["celltype"] = pd.Categorical(rng.choice(cts, n_obs))
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    if spot:
        adata.uns["spot_shape"] = "circle"
        adata.uns["spot_diameter"] = 3.0
    if deconv:
        cts = list(celltypes) if celltypes else CELLTYPES
        weights = rng.dirichlet(np.ones(len(cts)), size=n_obs)
        adata.obsm["deconv_weights"] = pd.DataFrame(weights, index=adata.obs_names, columns=cts)
        adata.uns["has_deconv_weights"] = True
    return adata


def make_reference(seed: int = 0, n_obs: int = 200, n_vars: int = 40) -> ad.AnnData:
    return make_adata(n_obs=n_obs, n_vars=n_vars, seed=seed + 100, celltypes=CELLTYPES)


def make_slice(slice_id: int, modality: str = "gene", data_level: str = "cell", image: bool = True, **kw) -> DataSlice:
    adata = make_adata(spot=(data_level == "spot"), **kw)
    images = {"dapi": np.zeros((100, 100), dtype=np.uint8)} if image else {}
    return DataSlice(
        slice_id=slice_id,
        modality=modality,
        data_level=data_level,
        adata=adata,
        images=images,
        metadata={"tissue_name": f"mini_{modality}_{slice_id}", "data_level_detection": "fixture", "image_match": "fixture"},
    )


def make_session(kind: str = "S1_cell_gene", seed: int = 0) -> SimpleSession:
    """Build one of the standard synthetic sessions.

    S1_cell_gene       one gene slice, cell level, with celltype
    S1_cell_gene_noCT  same without a celltype column
    S2_spot_deconv     one gene slice, spot level, with deconvolution weights, no celltype
    S3_two_slice_gene  two gene cell-level slices: slice 0 without, slice 1 with celltype
    S4_gene_protein    gene slice 0 with celltype + protein slice 1 on the same coordinates
    """
    session = SimpleSession(name=kind)
    if kind == "S1_cell_gene":
        session.add_slice(make_slice(0, seed=seed))
    elif kind == "S1_cell_gene_noCT":
        session.add_slice(make_slice(0, seed=seed, celltypes=None))
    elif kind == "S2_spot_deconv":
        session.add_slice(make_slice(0, data_level="spot", seed=seed, n_obs=300, n_vars=60, celltypes=None, deconv=True))
    elif kind == "S3_two_slice_gene":
        session.add_slice(make_slice(0, seed=seed, n_obs=400, celltypes=None))
        session.add_slice(make_slice(1, seed=seed + 1, n_obs=400))
    elif kind == "S4_gene_protein":
        session.add_slice(make_slice(0, seed=seed, n_obs=400))
        session.add_slice(make_slice(1, modality="protein", seed=seed, n_obs=400, n_vars=12, celltypes=None, prefix="P"))
    else:
        raise ValueError(f"unknown fixture kind: {kind}")
    session.initialize_celltype_colors()
    return session


def write_dataset(session: SimpleSession, out_dir: Path, image: bool = True) -> Path:
    """Write the session's slices as an h5ad dataset directory (strict naming)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in session.slice_ids:
        s = session.get_slice(sid)
        suffix = "_protein" if s.modality == "protein" else ""
        prefix = "spot_" if s.data_level == "spot" else ""
        s.adata.write_h5ad(out_dir / f"{prefix}tissue_slice_{sid}{suffix}.h5ad")
        if image and s.modality == "gene":
            np.save(out_dir / f"dapi_slice_{sid}.npy", np.zeros((100, 100), dtype=np.uint8))
    return out_dir


__all__ = ["CELLTYPES", "make_adata", "make_reference", "make_slice", "make_session", "write_dataset"]
