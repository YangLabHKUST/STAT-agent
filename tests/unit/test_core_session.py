from __future__ import annotations

import threading

import numpy as np
import pytest
from tests.fixtures.sessions import make_session

from stat_agent.core import SessionBusy, SessionSummary


def test_get_slice_and_errors():
    s = make_session("S3_two_slice_gene")
    assert s.get_slice(1).slice_id == 1
    assert s.get_slice("0").slice_id == 0
    with pytest.raises(ValueError, match="not found"):
        s.get_slice(7)
    assert s.slice_ids == [0, 1]
    assert s.has_slice(1) and not s.has_slice(9)


def test_iter_slices_filters():
    s = make_session("S4_gene_protein")
    assert [x.slice_id for x in s.iter_slices(modality="gene")] == [0]
    assert [x.slice_id for x in s.iter_slices(modality="protein")] == [1]
    assert s.modalities == ["gene", "protein"]
    assert s.data_format == "multi_slice_multi_omics"


@pytest.mark.parametrize(
    "definition",
    [
        {"type": "bbox", "x_min": 0, "x_max": 50, "y_min": 0, "y_max": 50},
        {"type": "circle", "center_x": 50, "center_y": 50, "radius": 20},
        {"type": "polygon", "vertices": [(0, 0), (60, 0), (60, 60), (0, 60)]},
    ],
)
def test_create_roi_membership_matches_geometry(definition):
    s = make_session("S1_cell_gene")
    roi = s.create_roi("r", 0, definition)
    obs = s.get_slice(0).adata.obs
    x, y = obs["x"].to_numpy(), obs["y"].to_numpy()
    if definition["type"] == "bbox":
        expected = (x >= 0) & (x <= 50) & (y >= 0) & (y <= 50)
    elif definition["type"] == "circle":
        expected = np.sqrt((x - 50) ** 2 + (y - 50) ** 2) <= 20
    else:
        expected = (x >= 0) & (x <= 60) & (y >= 0) & (y <= 60)
    assert set(roi.adata.obs_names) == set(obs.index[expected])
    assert roi.type == definition["type"]
    assert roi.slice_id == 0 and roi.modality == "gene"
    assert roi.metadata["definition"]["type"] == definition["type"]


def test_roi_single_store_and_lifecycle():
    s = make_session("S1_cell_gene")
    s.create_roi("a", 0, {"type": "bbox", "x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10})
    assert s.roi_subsets is s.roi_manager.rois
    assert s.get_roi("a") is s.roi_manager.rois["a"]
    with pytest.raises(ValueError, match="already exists"):
        s.create_roi("a", 0, {"type": "bbox", "x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10})
    s.rename_roi("a", "b")
    assert "a" not in s.roi_subsets and s.get_roi("b").name == "b"
    assert s.delete_roi("b") is True
    assert s.delete_roi("b") is False
    assert s.get_roi("b") is None


def test_create_roi_on_missing_slice_raises():
    s = make_session("S1_cell_gene")
    with pytest.raises(ValueError):
        s.create_roi("x", 3, {"type": "bbox", "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})


def test_summaries_are_typed_and_complete():
    s = make_session("S2_spot_deconv")
    s.create_roi("r", 0, {"type": "bbox", "x_min": 0, "x_max": 50, "y_min": 0, "y_max": 50})
    summary = SessionSummary.from_session(s)
    sl = summary.slice(0)
    assert sl is not None and sl.data_level == "spot" and sl.has_deconv_weights
    assert "x" in sl.obs_columns and "deconv_weights" in sl.obsm_keys
    assert sl.spot_diameter == 3.0
    assert summary.rois[0].name == "r" and summary.rois[0].n_obs == s.get_roi("r").n_obs
    text = summary.describe()
    assert "Slice 0" in text and "Available ROIs" in text
    fe = s.get_frontend_summary()
    for key in ("current_slice_id", "n_cells", "n_genes", "current_modality"):
        assert key not in fe
    assert fe["available_slices"][0]["slice_id"] == 0


def test_celltype_colors_consistent_and_deterministic():
    s = make_session("S3_two_slice_gene")
    colors = s.initialize_celltype_colors()
    assert set(colors) == set(s.get_slice(1).celltypes)
    assert s.get_slice(1).get_celltype_colors() == {ct: colors[ct] for ct in s.get_slice(1).celltypes}
    again = make_session("S3_two_slice_gene").initialize_celltype_colors()
    assert again == colors


def test_lock_timeout_raises_session_busy():
    s = make_session("S1_cell_gene")
    started = threading.Event()
    release = threading.Event()

    def holder():
        with s.lock.hold("worker"):
            started.set()
            release.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    started.wait(5)
    with pytest.raises(SessionBusy) as ei:
        with s.lock.hold("gui", timeout=0.05):
            pass
    assert ei.value.holder == "worker"
    release.set()
    t.join()
    with s.lock.hold("gui", timeout=0.5):
        with s.lock.hold("gui-nested", timeout=0.5):  # re-entrant
            assert s.lock.locked
    assert not s.lock.locked


def test_roi_data_follows_the_slice():
    """A region reads its cells from the current slice: a column written after the region was drawn is
    visible through it, membership stays what was drawn, and cells removed from the slice drop out."""
    s = make_session("S1_cell_gene")
    roi = s.create_roi("box", 0, {"type": "bbox", "x_min": 0, "x_max": 50, "y_min": 0, "y_max": 50})
    n0, members = roi.n_obs, list(roi.obs_names)
    assert n0 > 0 and roi.adata.n_obs == n0 and "later" not in roi.adata.obs.columns
    sl = s.get_slice(0)
    sl.adata.obs["later"] = np.arange(sl.adata.n_obs)
    assert "later" in roi.adata.obs.columns and "later" in s.get_roi_data("box").obs.columns
    assert list(roi.adata.obs_names) == members and roi.n_obs == n0 and roi.n_vars == sl.adata.n_vars
    # a write on the view does not leak into the slice or the membership
    view = roi.adata
    view.obs["scratch"] = 1
    assert "scratch" not in sl.adata.obs.columns and "scratch" not in roi.adata.obs.columns
    # the slice's object is replaced by a subset that drops half of the region: the region follows
    keep = [n for i, n in enumerate(sl.adata.obs_names) if n not in members[: n0 // 2]]
    sl.adata = sl.adata[keep].copy()
    assert roi.n_obs == n0 - n0 // 2 and set(roi.obs_names) == set(members[n0 // 2 :])
    # a region built without a source slice keeps the subset it is given
    from shapely.geometry import box

    from stat_agent.core.roi_manager import ROI

    lone = ROI(name="lone", geometry=box(0, 0, 10, 10))
    assert lone.n_obs == 0 and lone.adata is None
    lone.adata = sl.adata[:3].copy()
    assert lone.n_obs == 3 and lone.obs_names == [str(n) for n in sl.adata.obs_names[:3]]
