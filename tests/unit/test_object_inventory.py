"""What the session tells the roles about its objects: fingerprints that ignore the string/category
distinction, and an artefact inventory that names the type of every artefact."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.fixtures.sessions import make_session

from stat_agent.ledger import ProvenanceLedger
from stat_agent.ledger.provenance import Artifact, StepRef, _categories, artifact_spec
from stat_agent.runtime import Runtime, diff_snapshots, take_snapshot
from stat_agent.runtime.snapshot import Fingerprint, fp_series


@pytest.fixture
def rt(tmp_path):
    return Runtime(make_session("S1_cell_gene"), timeout_s=10, grace_s=1, lock_timeout_s=1, workdir=tmp_path)


def test_a_string_column_and_the_same_column_as_categories_fingerprint_alike():
    labels = ["TAM", "Macrophage", "TAM", "TAM"]
    as_str = pd.Series(labels, dtype=object)
    as_cat = as_str.astype("category")
    assert fp_series(as_str)[0] == fp_series(as_cat)[0]
    assert fp_series(as_str)[1] == "object" and fp_series(as_cat)[1] == "category"  # the true dtype is still reported
    assert fp_series(as_cat)[3]["categories"] == ["Macrophage", "TAM"]


def test_relabelling_is_still_a_change():
    before = pd.Series(["Stromal", "B Cells", "Stromal"], dtype=object).astype("category")
    after = before.cat.rename_categories({"Stromal": "CAF"})
    assert fp_series(before)[0] != fp_series(after)[0]
    # and so is a value moving between existing labels
    moved = pd.Series(["Stromal", "Stromal", "Stromal"], dtype=object).astype("category")
    assert fp_series(before)[0] != fp_series(moved)[0]


def test_a_numeric_column_still_fingerprints_its_dtype():
    assert fp_series(pd.Series([1, 2, 3]))[0] != fp_series(pd.Series([1.0, 2.0, 3.0]))[0]
    assert fp_series(pd.Series([1.0, 2.0, 3.0]))[0] != fp_series(pd.Series([1.0, 2.0, 4.0]))[0]


def test_a_column_of_free_text_does_not_carry_labels():
    free = pd.Series([f"barcode-{i}" for i in range(2500)], dtype=object)
    assert "categories" not in fp_series(free)[3]
    assert fp_series(free)[0] != fp_series(pd.Series([f"barcode-{i}" for i in range(2499)] + ["other"], dtype=object))[0]
    # and a merely long list of values is not shown as labels in the inventory
    long_fp = Fingerprint("s0:obs:id", "obs", "id", 0, "d", (600,), "object", 0, {"categories": [f"b{i}" for i in range(200)]})
    assert _categories(long_fp) is None


def test_saving_an_h5ad_is_not_a_change_of_the_session(rt, tmp_path):
    """anndata turns string columns into categoricals while writing; that is not a value change."""
    adata = rt.session.get_slice(0).adata
    assert rt.run("s = session.get_slice(0)\ns.adata.obs['group'] = ['TAM' if i % 2 else 'Macrophage' for i in range(s.adata.n_obs)]").success
    assert adata.obs["group"].dtype == object
    before = take_snapshot(rt.session)
    assert rt.run(f"session.get_slice(0).adata.write_h5ad(r'{tmp_path / 'slice0.h5ad'}')").success
    assert str(adata.obs["group"].dtype) == "category"  # the write did convert it in place
    assert diff_snapshots(before, take_snapshot(rt.session)).empty


def test_the_inventory_names_what_each_artefact_is(rt):
    before = take_snapshot(rt.session)
    assert rt.run(
        "import pandas as pd\n"
        "s = session.get_slice(0)\n"
        "s.adata.obs['group'] = ['TAM' if i % 2 else 'Macrophage' for i in range(s.adata.n_obs)]\n"
        "s.adata.obs['score'] = 0.5\n"
        "s.adata.uns['enrichment'] = {'TAM': pd.DataFrame({'Term': ['EMT'], 'P': [0.1]}), 'Macrophage': pd.DataFrame({'Term': ['IFN'], 'P': [0.2]})}\n"
        "s.adata.uns['de_results'] = pd.DataFrame({'gene': ['A', 'B', 'C'], 'lfc': [1.0, 2.0, 3.0]})\n"
    ).success
    led = ProvenanceLedger()
    led.initialize(before)
    led.record_step(before, take_snapshot(rt.session), producer=StepRef(turn_id="t0002", step=1), target_slice_ids=[0])
    text = led.state_text([0])
    assert "group [labels: Macrophage, TAM]" in text
    assert "score [float64]" in text
    assert "enrichment [dict, 2 entries]" in text
    assert "de_results [table 3 × 2: gene, lfc]" in text


def test_the_type_note_degrades_quietly():
    assert artifact_spec(Artifact(id="s0:uns:x", kind="uns", key="x")) == ""
    assert artifact_spec(Artifact(id="s0:uns:x", kind="uns", key="x", dtype="dict")) == " [dict]"
    assert artifact_spec(Artifact(id="s0:obsm:emb", kind="obsm", key="emb", shape=[600, 2])) == " [600 × 2]"
    assert artifact_spec(Artifact(id="s0:obsm:emb", kind="obsm", key="emb", shape=[600, 2], dtype="float32")) == " [600 × 2 float32]"
    assert artifact_spec(Artifact(id="s0:uns:names", kind="uns", key="names", shape=[28], dtype="object")) == " [28 values]"
    assert artifact_spec(Artifact(id="roi:goblet", kind="roi", key="goblet", shape=[13273], dtype="polygon")) == " [polygon, 13,273 cells]"
    many = Artifact(id="s0:obs:ct", kind="obs", key="ct", dtype="category", categories=[f"c{i}" for i in range(9)])
    assert artifact_spec(many) == " [labels: c0, c1, c2, c3, c4, c5, +3 more]"


def test_a_region_is_listed_under_its_slice(rt):
    """Regions are filed under their slice_id; the inventory must print them there."""
    adata = rt.session.get_slice(0).adata
    x, y = adata.obs["x"], adata.obs["y"]
    roi = rt.session.create_roi("tumour_nest", 0, {"type": "bbox", "x_min": float(x.min()), "x_max": float(x.median()),
                                                   "y_min": float(y.min()), "y_max": float(y.median())})
    led = ProvenanceLedger()
    led.initialize(take_snapshot(rt.session))
    text = led.state_text([0])
    assert f"roi: tumour_nest [bbox, {roi.n_obs:,} cells]" in text
    assert "Slice 0:" in text and roi.n_obs > 0


def test_a_table_names_its_columns(rt):
    before = take_snapshot(rt.session)
    assert rt.run(
        "import pandas as pd\n"
        "import numpy as np\n"
        "s = session.get_slice(0)\n"
        "s.adata.uns['de_results'] = pd.DataFrame({'names': ['A'], 'scores': [1.0], 'logfoldchanges': [2.0], 'pvals_adj': [0.01]})\n"
        "s.adata.obsm['deconv_weights'] = pd.DataFrame({'Tumor': np.zeros(s.adata.n_obs), 'CAF': np.zeros(s.adata.n_obs)}, index=s.adata.obs_names)\n"
    ).success
    led = ProvenanceLedger()
    led.initialize(before)
    led.record_step(before, take_snapshot(rt.session), producer=StepRef(turn_id="t0003", step=1), target_slice_ids=[0])
    text = led.state_text([0])
    assert "de_results [table 1 × 4: names, scores, logfoldchanges, pvals_adj]" in text
    assert "deconv_weights [600 × 2 float64: Tumor, CAF]" in text


def test_the_column_note_counts_the_real_width():
    wide = Artifact(id="s0:uns:t", kind="uns", key="t", dtype="DataFrame", shape=[10, 12],
                    columns=[f"c{i}" for i in range(12)])
    assert artifact_spec(wide) == " [table 10 × 12: c0, c1, c2, c3, c4, c5, c6, c7, +4 more]"
    # only the first 32 names are stored, so the count comes from the shape, not from the list
    truncated = Artifact(id="s0:uns:t", kind="uns", key="t", dtype="DataFrame", shape=[10, 100],
                         columns=[f"c{i}" for i in range(32)])
    assert truncated.columns is not None and "+92 more" in artifact_spec(truncated)


def test_an_odd_table_like_object_does_not_break_the_snapshot(rt):
    class Odd:
        shape = (2, 2)

        @property
        def columns(self):
            raise RuntimeError("no columns here")

    rt.session.get_slice(0).adata.uns["odd"] = Odd()
    art = take_snapshot(rt.session).artifacts["s0:uns:odd"]
    assert art.extra == {} and art.digest
