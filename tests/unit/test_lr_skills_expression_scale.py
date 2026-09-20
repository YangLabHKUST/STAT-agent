"""The two ligand-receptor skills must score the expression they normalize (B65).

The LIANA skill used to store the raw counts in ``adata.raw``, log-normalize ``adata.X`` and then pass
``use_raw=True``, so LIANA scored the raw counts and the normalization was discarded.  Its count detection
read ``adata.X[:100]`` and required ``max > 20``, which on a targeted panel (the first 100 Xenium cells top
out at 19) classified raw counts as normalized and left ``X`` unnormalized as well.  Both skills now draw
the sample from the whole slice and test only for integer values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[2] / "src" / "stat_agent" / "skills"
LR_SKILLS = ("cell-communication-LIANA", "cell-communication-CellPhoneDB")


def skill_text(slug: str) -> str:
    return (SKILLS / slug / "SKILL.md").read_text()


def test_liana_scores_the_normalized_matrix():
    text = skill_text("cell-communication-LIANA")
    assert "use_raw=False" in text
    assert "use_raw = True" not in text and "use_raw=use_raw" not in text


def test_liana_keeps_no_spatial_graph():
    """B66: the skill built a graph nothing read, and copying it back to the slice broke on any subset."""
    text = skill_text("cell-communication-LIANA")
    assert "spatial_connectivities" not in text
    assert "spatial_neighbors" not in text
    # the only LIANA call is the cluster-level consensus, which takes no connectivity
    assert "li.mt.rank_aggregate" in text and "li.mt.bivariate" not in text


@pytest.mark.parametrize("slug", LR_SKILLS)
def test_count_detection_samples_the_whole_slice(slug: str):
    text = skill_text(slug)
    assert "adata.X[:100]" not in text
    assert "np.random.default_rng(0).choice(adata.n_obs" in text
    # the max > 20 threshold misread targeted panels, whose counts are small
    assert "sample.max() > 20" not in text
