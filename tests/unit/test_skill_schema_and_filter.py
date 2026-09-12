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


def test_method_card_normalises_single_slice_alternatives():
    from stat_agent.skills.schema import MethodCard

    card = MethodCard.model_validate({"method_name": "X", "description": "d" * 30, "filter_requirements": {"num_slices": 1, "modalities": ["gene"], "data_levels": ["Cell", "spot"]}, "prerequisites": [], "consumes": [], "produces": [], "outline": [], "risks": [], "confidence": 0.5})
    assert card.filter_requirements.data_levels == ["cell/spot"] and card.filter_requirements.modalities == ["gene"]


def test_verifier_prepass_is_advisory_for_provisional_cards():
    from tests.fixtures.sessions import make_session

    from stat_agent.core.view import SessionSummary
    from stat_agent.roles.planner import PlanStep
    from stat_agent.roles.verifier import VerifierRole
    from stat_agent.skills.schema import MethodCard, SkillFrontmatter

    summary = SessionSummary.from_session(make_session("S1_cell_gene"))
    step = PlanStep(step_number=1, description="x", target_slice_ids=[0], refined_query="compute something")
    base = {"description": "d" * 30, "filter_requirements": {}, "prerequisites": ["p"], "produces": []}
    registered = SkillFrontmatter.model_validate({"name": "reg", "slug": "reg", **base, "consumes": ["obs:missing_key"]})
    assert VerifierRole.contract_pass(registered, step, summary, []).kind == "decline"
    card = MethodCard.model_validate({"method_name": "M", **base, "consumes": ["obs:missing_key"], "outline": [], "risks": [], "confidence": 0.5})
    assert VerifierRole.contract_pass(card, step, summary, []) is None
    card_counts = MethodCard.model_validate({"method_name": "M", **base, "consumes": ["layers:counts"], "outline": [], "risks": [], "confidence": 0.5})
    assert VerifierRole.contract_pass(card_counts, step, summary, []) is None  # raw counts live in X; not a hard decline
    assert "annotation" not in VerifierRole.contract_pass(registered, step, summary, []).advice
    card_layers = MethodCard.model_validate({"method_name": "M", **base, "consumes": ["layers:spliced", "layers:unspliced"], "outline": [], "risks": [], "confidence": 0.5})
    verdict = VerifierRole.contract_pass(card_layers, step, summary, [])
    assert verdict.kind == "decline" and verdict.missing == ["layers:spliced", "layers:unspliced"]
    assert "'spliced'" in verdict.advice and "'unspliced'" in verdict.advice and "annotation" not in verdict.advice


def test_non_default_skill_can_be_enabled_and_stays_enabled_across_reloads(tmp_path):
    from stat_agent.skills import SkillRegistry

    d = tmp_path / "skills" / "x-off"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: x-off\ndescription: " + "z" * 50 + "\nfilter_requirements:\n  data_levels: [cell/spot]\nprerequisites:\n  - a\ndefault_skill: false\n---\n\n# X\n\n```python\nprint(1)\n```\n")
    reg = SkillRegistry([(tmp_path / "skills", "t")]).load()
    assert "x-off" in reg and not reg.is_enabled("x-off")
    reg.enable("x-off")
    assert reg.is_enabled("x-off")
    reg.load()
    assert reg.is_enabled("x-off") and reg.states()["x-off"]["enabled"]


def test_ledger_ignores_a_no_change_run():
    from tests.fixtures.sessions import make_session

    from stat_agent.ledger.provenance import ProvenanceLedger, StepRef
    from stat_agent.runtime.snapshot import take_snapshot

    session = make_session("S1_cell_gene")
    ledger = ProvenanceLedger()
    ledger.initialize(take_snapshot(session))
    version, entries = ledger.version, len(ledger.log)
    delta = ledger.record_step(take_snapshot(session), take_snapshot(session), producer=StepRef(turn_id="t1", step=1))
    assert delta.empty and ledger.version == version and len(ledger.log) == entries
