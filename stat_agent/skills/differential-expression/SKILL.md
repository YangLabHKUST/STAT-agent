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
- **One cell type vs another**: Use `groups=['TypeA'], reference='TypeB'`.
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
sc.tl.rank_genes_groups(adata, groupby='group', method='wilcoxon', n_genes=100, use_raw=False)
de_df = sc.get.rank_genes_groups_df(adata, group=None)

# Step 4: Store results
slice_obj.adata.uns['rank_genes_groups'] = adata.uns['rank_genes_groups']
slice_obj.adata.uns['de_results'] = de_df
```

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `groupby` | — | Any categorical obs column | Column defining groups to compare |
| `method` | `'wilcoxon'` | `'wilcoxon'`, `'t-test'`, `'logreg'` | Statistical test |
| `groups` | all | List of group names | Specific groups to test |
| `reference` | `'rest'` | Group name or `'rest'` | Reference group for comparison |
| `n_genes` | `100` | 10-500 | Top genes per group |
