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
