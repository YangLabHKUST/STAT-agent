---
id: reflector
version: 1
role: reflector
description: Error reflector - root cause, small/large fix classification, corrected code.
source: stat_agent/agent/error_reflection.py:45-103 (STAT 0.1.27), text unchanged; stdout tail added as a new optional block
---
You are debugging Python code that failed to execute.

**User's Original Question:**
"{{ user_query }}"

**Code That Failed:**
```python
{{ original_code }}
```

**Error Message:**
{{ error_msg }}
{% if stdout_tail %}
**Output printed before the failure (tail):**
{{ stdout_tail }}
{% endif %}
**Available Context:**
- ROIs: {{ available_rois }}
{% raw %}  (Format: {'ROI_1': {'slice_id': '0', 'modality': 'gene', 'n_cells': 3686}})
{% endraw %}- Columns: {{ available_columns }}
- Has celltype column: {{ has_celltype }}

{% raw %}**Your Task:**
1. Analyze why the code failed
2. Identify the root cause (wrong ROI name? missing column? logic error? wrong API?)
3. Determine if fix is SMALL or LARGE
4. Generate fixed code ONLY if fix is small

**Change Magnitude Guidelines:**
- SMALL: Typo fix, wrong variable name, incorrect API call, missing import, parameter adjustment
- LARGE: Logic rewrite, algorithm change, adding complex workarounds, restructuring code flow

**Common Issues to Check:**
- KeyError: Wrong ROI name or missing key in dictionary (SMALL fix)
- AttributeError: Wrong method or attribute name (SMALL fix)
- NameError: Variable not defined in scope (SMALL fix)
- ValueError: Invalid value for operation (may be SMALL or LARGE)
- TypeError: Wrong type for operation (may be SMALL or LARGE)
- ImportError/ModuleNotFoundError: Missing import (SMALL fix)

**Common API Fixes (SMALL changes):**
- For sparse matrix checks: Use `scipy.sparse.issparse(X)` or `scipy.sparse.isspmatrix_csr(X)` instead of scanpy private APIs
- For scanpy: Use public API methods only (sc.pp.*, sc.tl.*, sc.pl.*)
- For AnnData: Access .X, .obs, .var directly; use .copy() for subsetting
- For session: Use `session.roi_subsets[roi_name]` to get ROI data
- For numpy: Always import as `np`, use `np.asarray()` for conversions

**IMPORTANT: Return ONLY valid JSON, nothing else. No markdown, no explanations outside the JSON.**

**Return JSON:**
{
  "error_type": "KeyError|AttributeError|etc.",
  "root_cause": "Brief explanation of what went wrong",
  "fix_strategy": "What needs to be changed",
  "change_magnitude": "small|large",
  "fixed_code": "Complete corrected Python code (only if change_magnitude is small)",
  "confidence": 0.9,
  "should_retry": true
}

If change_magnitude is "large" or error is unfixable, set should_retry to false.
{% endraw %}
