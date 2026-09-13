"""The blocks a later step receives: the completed steps of this turn and the live namespace."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import pandas as pd

from stat_agent.memory.context import namespace_block, turn_steps_block, with_turn_steps
from stat_agent.memory.state import StepHandoff
from stat_agent.runtime.executor import Runtime, describe_value


def test_turn_steps_block_lists_every_step_and_budgets_code_latest_first():
    h1 = StepHandoff(step_number=1, skill="mini-annotate", summary="Annotated cells.", data_changes="Added obs celltype.", key_findings=["600 T cells"], outputs=["s0:obs:celltype"], variables=["s"], code=["s = session.get_slice(0)\ns.adata.obs['celltype'] = 'T cell'"])
    h2 = StepHandoff(step_number=2, summary="Pooled the spots.", variables=["pooled"], code=["pooled = s.adata.copy()"], mode="unguarded")
    h3 = StepHandoff(step_number=3, description="Broke.", success=False, unresolved=["KeyError"], code=[])
    text = turn_steps_block([h1.model_dump(), h2, h3], code_budget=len("pooled = s.adata.copy()") + 5)
    assert text.startswith("**Completed steps in this turn")
    assert "Step 1 [mini-annotate] (done): Annotated cells." in text
    assert "  changes: Added obs celltype." in text and "  findings: 600 T cells" in text
    assert "  artefacts: s0:obs:celltype" in text and "  variables added: s" in text
    assert "Step 2 (done): Pooled the spots." in text and "```python\npooled = s.adata.copy()\n```" in text
    assert "s.adata.obs['celltype']" not in text  # step 1's code did not fit the budget; the latest step's did
    assert "Step 3 (FAILED): Broke." in text and "  unresolved: KeyError" in text
    assert turn_steps_block([]) == ""
    assert with_turn_steps("", "") == "" and with_turn_steps("h", "") == "h" and with_turn_steps("h", "s") == "h\n\ns"


def test_history_lines_carry_step_details_facts_and_earlier_answers():
    from stat_agent.memory.state import ConversationMemory, TurnRecord

    h = StepHandoff(step_number=1, skill="mini-annotate", summary="Annotated cells.", outputs=["s0:obs:celltype"], variables=["s"], unresolved=[])
    rec = TurnRecord(turn_id="t1", index=1, query="annotate", response_summary="Annotated cells.", handoffs=[h], answers=[{"kind": "verifier", "question": "Which tissue?", "answer": "brain cortex"}])
    text = rec.history_lines()
    assert "**Step 1:** skill mini-annotate; outputs s0:obs:celltype; variables s" in text
    mem = ConversationMemory()
    mem.record_turn(rec)
    mem.add_fact("species", "human", "t1")
    assert mem.history_text().startswith("**Known facts:** species: human")
    assert mem.recent_answers("verifier") == [("Which tissue?", "brain cortex")]
    mem.record_turn(TurnRecord(turn_id="t2", index=2, query="again", answers=[{"kind": "verifier", "question": "Which tissue?", "answer": "liver"}, {"kind": "verifier", "question": "Empty?", "answer": ""}]))
    assert mem.recent_answers("verifier") == [("Which tissue?", "liver")]
    assert mem.recent_answers("verifier", n_turns=1) == [("Which tissue?", "liver")]


def test_describe_value_and_namespace_block():
    adata = ad.AnnData(X=np.zeros((5, 3)), obs=pd.DataFrame({"celltype": list("aabbc"), "cluster": [0, 1, 0, 1, 2]}, index=[f"c{i}" for i in range(5)]))
    adata.obsm["X_pca"] = np.zeros((5, 2))
    assert describe_value(adata) == "AnnData (5 × 3; obs: celltype, cluster; obsm: X_pca)"
    assert describe_value(pd.DataFrame({"a": [1], "b": [2]})) == "DataFrame (1 × 2; columns: a, b)"
    assert describe_value(np.zeros((2, 3))) == "ndarray (shape (2, 3), float64)"
    assert describe_value("x" * 100).startswith("str = 'xxx") and describe_value("x" * 100).endswith("…")
    assert describe_value([1, 2]) == "list (2 items)" and describe_value({"k": 1}) == "dict (1 keys: k)"
    assert describe_value(0.3) == "float = 0.3" and describe_value(lambda: 0) == "function"
    assert describe_value(types.ModuleType("m")) == "module"

    class Odd:
        def __len__(self):
            raise RuntimeError("no")

    assert describe_value(Odd()) == "Odd"
    assert namespace_block([]) == ""
    assert namespace_block([("pooled", "AnnData (5 × 3; obs: cluster)")]) == "**Python namespace (variables from earlier code blocks, steps and turns; they persist and can be reused):**\n- pooled: AnnData (5 × 3; obs: cluster)"


def test_runtime_describe_namespace_skips_seeds_modules_and_private_names(tmp_path):
    rt = Runtime(session=object(), snapshot=False, workdir=tmp_path)
    rt.namespace.update({"sc": types.ModuleType("sc"), "_tmp": 1, "res": 0.3, "labels": [1, 2]})
    assert rt.user_variables() == ["res", "labels"]
    assert rt.describe_namespace() == [("res", "float = 0.3"), ("labels", "list (2 items)")]
    assert rt.describe_namespace(limit=1) == [("labels", "list (2 items)")]


def test_memory_window_facts_and_summary_word_budget():
    """Six verbatim turns, twenty-four facts shown, and a summary word budget that grows with the turns it covers (B43)."""
    from stat_agent.memory.state import ConversationMemory, TurnRecord

    mem = ConversationMemory()
    assert mem.window_turns == 6
    for i in range(1, 9):
        mem.record_turn(TurnRecord(turn_id=f"t{i}", index=i, query=f"query {i}", response_summary=f"did {i}"))
    text = mem.history_text()
    assert "query 3" in text and "query 2" not in text
    assert mem.needs_summary and [t.turn_id for t in mem.evicted_turns()] == ["t1", "t2"]
    assert mem.summary_word_budget() == 150 + 40 * 2
    for _ in range(40):
        mem.record_turn(TurnRecord(turn_id="tx", index=9, query="q"))
    assert mem.summary_word_budget() == 1200
    for k in range(30):
        mem.add_fact(f"k{k}", str(k), "t8")
    text = mem.history_text()
    assert "k6: 6" in text and "k5: 5" not in text


def test_instruction_facts_are_merged_by_wording():
    """The summariser restates every standing instruction each time it runs, usually under a fresh
    key: the same wording updates the existing entry (keeping the first key), a different scope
    stays a separate fact, and other facts are untouched."""
    from stat_agent.memory.state import ConversationMemory

    mem = ConversationMemory()
    mem.add_fact("instruction: slices", "do not touch/modify slice 1 — work on copies", "t8")
    mem.add_fact("instruction: slice1_slice2", "Do not touch or modify slice 1: work on copies.", "t9")
    mem.add_fact("instruction: slice 1 and 2", "do not touch/modify slice 1 or slice 2 themselves — work on copies", "t9")
    mem.add_fact("species", "human", "t9")
    mem.add_fact("species", "mouse", "t10")
    assert [k for k in mem.facts if k.startswith("instruction:")] == ["instruction: slices", "instruction: slice 1 and 2"]
    merged = mem.facts["instruction: slices"]
    assert merged.value == "Do not touch or modify slice 1: work on copies." and merged.source_turn == "t9"
    assert list(mem.facts) == ["instruction: slices", "instruction: slice 1 and 2", "species"] and mem.facts["species"].value == "mouse"
    mem.add_fact("instruction: copies", "do not touch/modify slice 1 - work on copies", "t11")  # restated later: same fact, now the most recent
    assert list(mem.facts) == ["instruction: slice 1 and 2", "species", "instruction: slices"] and mem.facts["instruction: slices"].source_turn == "t11"


def test_planner_session_block_lists_the_cell_type_names():
    """The planner answers directly about viewer edits, so its session block names the current cell
    types of each slice (capped), letting it check a reported rename against the record."""
    from tests.fixtures.sessions import make_session

    from stat_agent.core.view import SessionSummary
    from stat_agent.memory.context import planner_session_block

    summary = SessionSummary.from_session(make_session("S1_cell_gene"))
    text = planner_session_block(summary)
    assert "- Total slices: 1" in text and "    cell types: B cell, Fibroblast, T cell, Tumor" in text
    assert "cell types:" not in planner_session_block(SessionSummary.from_session(make_session("S1_cell_gene_noCT")))
    capped = planner_session_block(summary, celltype_names=2)
    assert "cell types: B cell, Fibroblast (+2 more)" in capped
