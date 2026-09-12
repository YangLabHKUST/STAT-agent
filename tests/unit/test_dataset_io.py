from __future__ import annotations

import numpy as np
from tests.fixtures.sessions import make_adata, make_session, write_dataset

from stat_agent.core import SimpleSession
from stat_agent.io import LoadOptions, detect_data_level, load_dataset, parse_slice_id_from_filename


def test_parse_slice_id():
    assert parse_slice_id_from_filename("tissue.h5ad") == (None, "gene")
    assert parse_slice_id_from_filename("tissue_protein.h5ad") == (None, "protein")
    assert parse_slice_id_from_filename("tissue_slice_3.h5ad") == (3, "gene")
    assert parse_slice_id_from_filename("t_slice_2_protein") == (2, "protein")


def test_detect_data_level_rules():
    small = make_adata(n_obs=50)
    assert detect_data_level(small, "spot_x")[0] == "spot"
    assert detect_data_level(small, "x")[0] == "spot"  # size heuristic
    assert detect_data_level(small, "x", LoadOptions(spot_threshold=10))[0] == "cell"
    with_uns = make_adata(n_obs=50, spot=True)
    assert detect_data_level(with_uns, "x", LoadOptions(spot_threshold=10))[0] == "spot"


def test_write_then_load_dataset_roundtrip(tmp_path):
    src = make_session("S4_gene_protein")
    write_dataset(src, tmp_path / "ds")
    slices = load_dataset(tmp_path / "ds", LoadOptions(spot_threshold=10))
    assert sorted(slices) == [0, 1]
    assert slices[0].modality == "gene" and slices[1].modality == "protein"
    assert slices[0].data_level == "cell"
    assert "dapi" in slices[0].images and slices[0].metadata["image_match"] == "strict"
    assert slices[0].has_celltype()
    session = SimpleSession()
    ids = session.load_dataset(tmp_path / "ds", LoadOptions(spot_threshold=10))
    assert ids == [0, 1] and session.celltype_colors


def test_spot_dataset_gets_virtual_celltype(tmp_path):
    src = make_session("S2_spot_deconv")
    write_dataset(src, tmp_path / "ds", image=False)
    slices = load_dataset(tmp_path / "ds")
    s = slices[0]
    assert s.data_level == "spot" and s.has_deconv_weights()
    assert s.has_celltype()  # dominant weight -> virtual celltype
    assert s.adata.uns["has_deconv_weights"] is True or bool(s.adata.uns["has_deconv_weights"])


def test_non_integer_matrix_recovers_from_layers(tmp_path):
    adata = make_adata(n_obs=30, n_vars=20)
    adata.layers["counts"] = adata.X.copy()
    adata.X = np.log1p(adata.X)
    d = tmp_path / "ds"
    d.mkdir()
    adata.write_h5ad(d / "tissue.h5ad")
    slices = load_dataset(d, LoadOptions(spot_threshold=1))
    X = slices[0].adata.X
    assert np.allclose(X, np.round(X))
