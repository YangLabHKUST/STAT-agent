from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.fixtures.sessions import make_session

from stat_agent.core import SessionSummary
from stat_agent.skills import (
    ArtifactRef,
    FilterRequirements,
    MethodCard,
    SkillFrontmatter,
    filter_skills,
    parse_skill_text,
)
from stat_agent.skills.filter import check_compatibility


def test_v1_frontmatter_derives_identity():
    meta = SkillFrontmatter.model_validate(
        {"name": "Cell Type Deconvolution (RCTD)", "description": "x" * 50, "prerequisites": "a\nb", "default_skill": "false"}
    )
    assert meta.slug == "cell-type-deconvolution-rctd" and meta.title == "Cell Type Deconvolution (RCTD)"
    assert meta.prerequisites == ["a", "b"] and meta.default_skill is False
    meta2 = SkillFrontmatter.model_validate({"name": "svg-spatialde", "description": "y" * 50})
    assert meta2.slug == "svg-spatialde"


def test_v2_fields_and_artifact_refs():
    meta = SkillFrontmatter.model_validate(
        {
            "schema_version": 2,
            "name": "niche-detection",
            "description": "d" * 50,
            "consumes": ["obs:celltype"],
            "produces": [{"kind": "obs", "key": "niche_label"}, "uns:niche_*"],
            "postconditions": [{"artifact": "obs:niche_label", "check": "n_unique_min", "params": {"n": 2}}],
            "dependencies": {"python": ["scanpy"], "gpu": False, "vendored": ["niche_analysis_lib"]},
            "platforms": "Xenium, MERFISH",
            "custom_key": 1,
        }
    )
    assert meta.consumes[0].id == "obs:celltype"
    assert meta.produces[1].matches("niche_scores") and not meta.produces[1].matches("celltype")
    assert meta.platforms == ["Xenium", "MERFISH"]
    assert meta.extra_keys == ["custom_key"]
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate("celltype")


@pytest.mark.parametrize(
    "req",
    [
        {"modalities": ["rna"]},
        {"data_levels": ["cell", "spot"]},  # positional without num_slices
        {"num_slices": 2, "data_levels": ["cell", "spot", "cell"]},
        {"num_slices": 0},
    ],
)
def test_filter_grammar_rejects_bad_contracts(req):
    with pytest.raises(ValidationError):
        FilterRequirements.model_validate(req)


def test_check_compatibility_rules():
    s = SessionSummary.from_session(make_session("S4_gene_protein"))
    gene0, prot1 = s.slice(0), s.slice(1)
    assert check_compatibility(FilterRequirements(), [gene0])[0]
    assert check_compatibility(FilterRequirements(modalities=["gene"]), [gene0])[0]
    ok, reason = check_compatibility(FilterRequirements(modalities=["gene"]), [prot1])
    assert not ok and "modality" in reason
    assert check_compatibility(FilterRequirements(modalities=["gene/protein"]), [gene0, prot1])[0]
    assert check_compatibility(FilterRequirements(num_slices=2, modalities=["gene", "protein"]), [gene0, prot1])[0]
    assert not check_compatibility(FilterRequirements(num_slices=2, modalities=["gene", "protein"]), [prot1, gene0])[0]
    assert not check_compatibility(FilterRequirements(num_slices=1), [gene0, prot1])[0]
    assert check_compatibility(FilterRequirements(min_slices=2), [gene0, prot1])[0]
    assert not check_compatibility(FilterRequirements(min_slices=2), [gene0])[0]
    assert not check_compatibility(FilterRequirements(data_levels=["spot"]), [gene0])[0]


def test_filter_skills_partition_and_bypass():
    summary = SessionSummary.from_session(make_session("S2_spot_deconv"))
    contracts = {
        "deconv": FilterRequirements(modalities=["gene"], data_levels=["spot"]),
        "niche": FilterRequirements(modalities=["gene"], data_levels=["cell"]),
        "anything": FilterRequirements(),
    }
    res = filter_skills([0], summary, contracts)
    assert res.compatible == ["anything", "deconv"] and res.rejected_slugs == ["niche"]
    assert set(res.compatible) | set(res.rejected_slugs) == set(contracts)
    by = filter_skills([0], summary, contracts, bypass=True)
    assert by.bypassed and by.compatible == sorted(contracts) and by.would_reject == res.rejected
    with pytest.raises(ValueError):
        filter_skills([3], summary, contracts)
    with pytest.raises(ValueError):
        filter_skills([], summary, contracts)


def test_parse_skill_text_and_method_card():
    text = "---\nname: x-y\ndescription: " + "z" * 50 + "\nfilter_requirements:\n  data_levels: [cell/spot]\nprerequisites:\n  - a\n---\n\n# X\n\n```python\nprint(1)\n```\n"
    meta, body, raw = parse_skill_text(text)
    assert meta.slug == "x-y" and body.startswith("# X") and raw["name"] == "x-y"
    card = MethodCard.model_validate(
        {"method_name": "scVelo RNA velocity", "description": "v" * 50, "prerequisites": ["spliced and unspliced layers"], "filter_requirements": {"data_levels": ["cell"]}, "confidence": 0.8}
    )
    assert card.provisional is True and card.slug == "scvelo-rna-velocity"
