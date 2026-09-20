---
name: differential-expression
title: Differential Gene Expression Analysis
slug: differential-expression
description: Find differentially expressed marker genes between groups using scanpy rank_genes_groups with Wilcoxon test. The comparison target is flexible — between cell types, clusters, spatial domains, slices, ROIs, or any user-defined grouping.

filter_requirements:
  modalities: [gene]

prerequisites:
  - "Understanding of what the user wants to compare (e.g., cell types within a slice, same cell type across slices, cells in different ROIs, etc.)"

default_skill: true
schema_version: 2
version: "2.0"
consumes: []
produces: [uns:rank_genes_groups, uns:de_results]
platforms: [linux, macos, windows]
dependencies:
  python: [scanpy]
---

# Differential Gene Expression Analysis

Find **differentially expressed genes** (DEGs) using `sc.tl.rank_genes_groups`.

**The comparison target is completely flexible.** The agent must understand the user's intent and prepare the data accordingly — the key is to construct an adata with a categorical column that defines the two (or more) groups to compare.

## Core Function

```python
import scanpy as sc

# Prepare adata with a 'group' column that defines the comparison
# Then simply:
sc.tl.rank_genes_groups(adata, groupby='group', method='wilcoxon')
de_df = sc.get.rank_genes_groups_df(adata, group=None)
```

## Comparison Scenarios (for the agent)

The agent should understand the query, select the right cells, and build the comparison adata. Examples:

- **Between cell types in one slice**: Use `groupby='celltype'` directly on that slice's adata.
- **One cell type vs another**: Use `groups=['TypeA', 'TypeB']`, or `groups=['TypeA'], reference='TypeB'` — the second form tests one group, so its result frame has no `group` column (see Outputs).
- **Same cell type across slices**: Subset cells of that type from both slices, concatenate, label by slice origin, then `groupby='slice'`.
- **Between ROIs**: Subset cells from each ROI, concatenate, label by ROI name, then `groupby='roi'`.
- **Tumor vs normal region**: Label cells by region annotation, then `groupby='region'`.
- **Any custom comparison**: The agent creates a categorical column encoding the comparison, then calls `rank_genes_groups`.

**Do not restrict to a fixed workflow.** The agent should flexibly prepare the data based on the user's question.

## Minimal Workflow Template

```python
import scanpy as sc
import pandas as pd
import numpy as np

# Step 1: Prepare adata with comparison groups
# -- The agent constructs this based on the user's query --
# Example: adata with adata.obs['group'] as the comparison column

# Step 2: Normalize if needed
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Step 3: Run DE
sc.tl.rank_genes_groups(adata, groupby='group', method='wilcoxon', n_genes=None, use_raw=False)
de_df = sc.get.rank_genes_groups_df(adata, group=None)

# Step 4: Store results
slice_obj.adata.uns['rank_genes_groups'] = adata.uns['rank_genes_groups']
slice_obj.adata.uns['de_results'] = de_df

# Print the schema, so a later step reads this table by its real column names
# instead of guessing ('gene', 'score', 'log2fc' are NOT columns of this frame)
print(f"uns['de_results'] columns: {list(de_df.columns)}")
print("uns['de_results'] group values: "
      + (f"{sorted(de_df['group'].unique())}" if 'group' in de_df.columns
         else "no 'group' column - this run compared a single group"))
print(f"uns['de_results'] rows: {len(de_df):,}")
```

## Outputs

`uns['de_results']` is the frame `sc.get.rank_genes_groups_df` returns. Its columns are exactly:

| column | what it holds |
|---|---|
| `names` | the gene symbol — **not** `gene` |
| `scores` | the test statistic — **not** `score` |
| `logfoldchanges` | log2 fold change — **not** `log2fc` or `lfc` |
| `pvals` | raw p-value |
| `pvals_adj` | Benjamini-Hochberg adjusted p-value |
| `group` | the value of the `groupby` column this row's comparison is for — **present only when the run covered two or more groups** |

`group` is conditional, and that has bitten real runs. `sc.get.rank_genes_groups_df` returns the five columns
above and **no `group`** when `rank_genes_groups` covered a single group. That includes the two-group
comparison written as `groups=['TypeA'], reference='TypeB'`, which this skill's own scenario list suggests:
one group is tested, so the frame comes back five-wide with the reference implicit. It also happens when the
groups passed reduce to one because a filter emptied the others. Test for it — `if 'group' in df.columns` —
before reading it or grouping by it; do not assume the six-column shape. To get a `group` column for a
two-group comparison, pass both (`groups=['TypeA', 'TypeB']`) instead.

**`method='logreg'` returns a different, narrower frame: `group`, `names`, `scores` only.** No
`logfoldchanges`, no `pvals`, no `pvals_adj`. Any filter on significance or fold change fails on it. Use
`wilcoxon` (or `t-test`) whenever the answer needs p-values or effect sizes, which is nearly always here.

**`n_genes` truncates what you store.** `rank_genes_groups(..., n_genes=100)` keeps the top 100 genes per
group, so `uns['de_results']` is not the full result and a later step that treats it as the whole
transcriptome is reading a pre-cut list. Pass `n_genes=None` for every gene when the table is to be stored
and read again, or state the cut where you print it.

Read it by those names. `group` holds the category values of whatever column was passed as `groupby`, not
labels of your own choosing: if the comparison was run on a column whose values are `dcis_duct` and
`invasive_front`, those are the strings in `group`, and a lookup for `'dcis'` or `'inv'` returns nothing.
Take the values from `de_df['group'].unique()` rather than assuming them.

**If you rename the columns before storing, the table stored is yours, not this one.** A step is free to store
`gene, score, log2fc, pval, pval_adj` instead, and some do — but then no later step can rely on the schema
above. So a step that reads a `de_results` it did not itself write takes the column names from the session
artefact inventory, which lists them for every stored table (`de_results [table 306 × 6: ...]`), or from
`list(df.columns)`, and never from either naming convention by assumption.

`uns['rank_genes_groups']` is scanpy's own structured result, kept for `sc.pl` plotting functions.

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `groupby` | — | Any categorical obs column | Column defining groups to compare |
| `method` | `'wilcoxon'` | `'wilcoxon'`, `'t-test'`, `'logreg'` | Statistical test. `logreg` returns no p-values or fold changes (see Outputs) |
| `groups` | all | List of group names | Specific groups to test. One name here means no `group` column in the result |
| `reference` | `'rest'` | Group name or `'rest'` | Reference group for comparison |
| `n_genes` | `100` | 10-500, or `None` | Top genes per group; `None` keeps every gene. The stored table is cut to this |
