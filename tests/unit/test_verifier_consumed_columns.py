"""An absent obs column a skill consumes (B62).

stat40 Q40, 2026-09-17: "test whether cell types co-localize" on a Visium slice with no cell types; the
planner wrote "using the 'domain' or 'layer' labels", the deterministic "annotate first" decline was skipped
because the refined query contained the word "label", and the model mapped the cortical layers onto
cell types. Now a column counts only when it is the same key spelled differently or the user names it
when asked; the question lists the slice's label columns, and a reply that names none is declined.
"""

from __future__ import annotations

import pandas as pd
from tests.fixtures.sessions import make_session

from stat_agent.agent.policy import AutoPolicy
from stat_agent.core.view import SessionSummary
from stat_agent.roles.planner import PlanStep
from stat_agent.roles.verifier import column_name_matches, column_note, columns_named, resolve_contract
from stat_agent.skills.schema import SkillFrontmatter

FALLBACK = "Please proceed with a reasonable default and state the assumption you made."
LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]


def niche_skill(**extra) -> SkillFrontmatter:
    return SkillFrontmatter.model_validate({"name": "nb", "slug": "nb", "description": "d" * 30, "filter_requirements": {}, "prerequisites": ["Cell type annotations (adata.obs['celltype'])"], "consumes": ["obs:celltype"], "produces": ["uns:nhood"], **extra})


def summary_with(**columns) -> SessionSummary:
    session = make_session("S1_cell_gene_noCT")
    adata = session.get_slice(0).adata
    for name, values in columns.items():
        adata.obs[name] = values
    return SessionSummary.from_session(session)


def layered(n=600) -> dict:
    return {"layer": pd.Categorical([LAYERS[i % 7] for i in range(n)]), "domain": pd.Categorical([f"D{i % 7}" for i in range(n)])}


STEP = PlanStep(step_number=1, description="co-localization", target_slice_ids=[0], refined_query="Test which cell types (using the 'domain' or 'layer' labels) co-localize")


def test_label_columns_in_the_slice_summary():
    n = 600
    summary = summary_with(layer=pd.Categorical([LAYERS[i % 7] for i in range(n)]), barcode=[f"bc{i}" for i in range(n)], sample=["s1"] * n, score=[0.5] * n, region=["a", "b", None] * 200)
    labels = summary.slice(0).label_columns
    assert labels["layer"].n_labels == 7 and labels["layer"].examples == sorted(LAYERS)
    assert labels["region"].n_labels == 2  # missing values are not labels
    assert "barcode" not in labels  # an identifier
    assert "sample" not in labels  # a single value
    assert "score" not in labels and "x" not in labels  # numbers


def test_same_key_spelled_differently():
    for name in ("cell_type", "CellType", "celltypes", "Cell.Types", "predicted_celltype", "celltype_scanvi", "cellType_major"):
        assert column_name_matches("celltype", name), name
    for name in ("celltype", "layer", "domain", "subcelltype", "cell", "type", "leiden"):
        assert not column_name_matches("celltype", name), name


def test_columns_named_in_a_reply():
    cols = ["layer", "domain"]
    assert columns_named("Use the layer column", cols) == ["layer"]
    assert columns_named("adata.obs.domain please", cols) == ["domain"]
    assert columns_named("obs['LAYER']", cols) == ["layer"]
    assert columns_named("annotate cell types first", cols) == []
    assert columns_named("the sublayer_x column", cols) == []
    assert columns_named("layer or domain", cols) == ["layer", "domain"]


def test_absent_column_with_label_columns_asks_once_listing_them():
    verdict, columns = resolve_contract(niche_skill(), STEP, summary_with(**layered()))
    assert verdict.kind == "ask" and verdict.decided_by == "contract" and columns == {}
    q = verdict.questions[0]
    assert "obs['celltype']" in q and "slice 0 does not have" in q
    assert "layer (7 labels: L1, L2, L3, L4, L5, L6, ...)" in q and "domain (7 labels:" in q


def test_reply_naming_no_column_declines_whatever_the_query_says():
    summary = summary_with(**layered())
    asked, _ = resolve_contract(niche_skill(), STEP, summary)
    for reply in (FALLBACK, "annotate cell types first", "celltype"):
        verdict, columns = resolve_contract(niche_skill(), STEP, summary, answers=[(asked.questions[0], reply)])
        assert verdict.kind == "decline" and verdict.decided_by == "contract" and columns == {}, reply
        assert "cell type annotation" in verdict.advice and "no existing column was named" in verdict.advice


def test_reply_naming_one_column_uses_it_and_says_so():
    summary = summary_with(**layered())
    asked, _ = resolve_contract(niche_skill(), STEP, summary)
    verdict, columns = resolve_contract(niche_skill(), STEP, summary, answers=[(asked.questions[0], "use domain")])
    assert verdict is None and columns == {"s0:obs:celltype": ("domain", "named by the user")}
    note = column_note(columns)
    assert "On slice 0 read obs['domain'] wherever the analysis reads obs['celltype'] (named by the user)" in note and "do not create or overwrite obs['celltype']" in note


def test_reply_naming_two_columns_declines():
    summary = summary_with(**layered())
    asked, _ = resolve_contract(niche_skill(), STEP, summary)
    verdict, _ = resolve_contract(niche_skill(), STEP, summary, answers=[(asked.questions[0], "layer or domain, whichever")])
    assert verdict.kind == "decline" and "more than one column (layer, domain)" in verdict.advice


def test_answers_to_other_questions_do_not_count():
    summary = summary_with(**layered())
    verdict, _ = resolve_contract(niche_skill(), STEP, summary, answers=[("What tissue is this?", "use layer")])
    assert verdict.kind == "ask"


def test_same_key_column_is_used_without_asking():
    n = 600
    summary = summary_with(cell_type=pd.Categorical(["T", "B", "Tumor"] * (n // 3)), **layered())
    verdict, columns = resolve_contract(niche_skill(), STEP, summary)
    assert verdict is None and columns == {"s0:obs:celltype": ("cell_type", "the same key spelled differently from obs['celltype']")}


def test_two_same_key_columns_are_asked_about_and_listed_first():
    n = 600
    summary = summary_with(**layered(), cell_type=pd.Categorical(["T", "B", "Tumor"] * (n // 3)), celltype_major=pd.Categorical(["Imm", "Epi"] * (n // 2)))
    verdict, _ = resolve_contract(niche_skill(), STEP, summary)
    q = verdict.questions[0]
    assert verdict.kind == "ask" and q.index("cell_type (") < q.index("layer (") and q.index("celltype_major (") < q.index("layer (")


def test_no_label_columns_declines_at_once():
    verdict, _ = resolve_contract(niche_skill(), STEP, SessionSummary.from_session(make_session("S1_cell_gene_noCT")))
    assert verdict.kind == "decline" and verdict.advice.startswith("This analysis requires obs:celltype, which does not exist on slice(s) [0]. Run cell type annotation first")


def test_a_skill_that_writes_the_key_is_not_asked_and_a_present_key_is_left_to_the_model():
    summary = summary_with(**layered())
    assert resolve_contract(niche_skill(produces=["obs:celltype"]), STEP, summary) == (None, {})
    assert resolve_contract(niche_skill(), STEP, SessionSummary.from_session(make_session("S1_cell_gene"))) == (None, {})


def test_auto_mode_never_turns_the_question_into_a_substitution():
    summary = summary_with(**layered())
    asked, _ = resolve_contract(niche_skill(), STEP, summary)
    answer, _source, _why = AutoPolicy()._answer_question(asked.questions[0], None, {}, {})
    verdict, columns = resolve_contract(niche_skill(), STEP, summary, answers=[(asked.questions[0], answer or "")])
    assert verdict.kind == "decline" and columns == {}
