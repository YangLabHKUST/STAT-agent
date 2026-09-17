"""The fast annotation helper never returns cluster numbers as cell types when its model call fails (B63).

stat40 Q01 and Q39 (seed 0) and Q31 (2026-09-17): the labelling call failed, the helper caught the error and
returned ``Cluster_<n>`` as the cell types, and the step was reported as a successful annotation.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scanpy as sc

SKILL_DIR = Path(__file__).resolve().parents[2] / "src" / "stat_agent" / "skills" / "celltype-annotation-GPT"


def _adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, size=(120, 40)).astype(np.float32)
    x[:60, :10] += rng.poisson(8.0, size=(60, 10))
    adata = ad.AnnData(x)
    adata.var_names = [f"g{i}" for i in range(40)]
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_neighbors=10)
    return adata


def test_a_failed_labelling_call_raises_instead_of_returning_cluster_numbers(monkeypatch):
    monkeypatch.syspath_prepend(str(SKILL_DIR))
    from celltype_fast.annotation_clustering import annotate_celltype_clustering

    def unreachable(markers, tissue):
        raise RuntimeError("LLM annotation failed: Connection error.")

    with pytest.raises(RuntimeError, match="Connection error"):
        annotate_celltype_clustering(_adata(), "test tissue", llm_function=unreachable, resolution=1.0, preprocess=False, min_cluster_size=1)

    labels = annotate_celltype_clustering(_adata(), "test tissue", llm_function=lambda markers, tissue: {c: f"type {c}" for c in markers}, resolution=1.0, preprocess=False, min_cluster_size=1)
    assert labels.notna().all() and not labels.astype(str).str.startswith("Cluster_").any()
