---
name: pathway-ssgsea
title: Per-Cell Pathway Activity Scoring (ssGSEA)
slug: pathway-ssgsea
description: Compute per-cell pathway activity scores using single-sample Gene Set Enrichment Analysis (ssGSEA). Scores each cell/spot for pathway activation, enabling spatial visualization of pathway activity across tissue. Supports MSigDB Hallmark, KEGG, GO, and custom gene sets.

filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [cell/spot]

prerequisites:
  - Pathway or gene set name (e.g., 'HALLMARK_INFLAMMATORY_RESPONSE', 'KEGG_APOPTOSIS'), or a custom gene set dict
  - Species (human or mouse)

default_skill: false
---

# Per-Cell Pathway Activity Scoring (ssGSEA)

Compute **per-cell/spot pathway activity scores** using single-sample Gene Set Enrichment Analysis (ssGSEA). Unlike standard enrichment (which takes a gene list), ssGSEA scores **every cell** for how active a pathway is — enabling **spatial visualization** of pathway activity across the tissue.

**Input**: Gene set name or custom gene set, expression data.

**Output**: Per-cell scores in `adata.obs['ssgsea_{pathway}']` — directly visualizable on the spatial canvas.

---

## Workflow

### Stage 1: Load Data and Gene Sets

```python
import numpy as np
import pandas as pd
import gseapy as gp

print("=" * 60)
print("STAGE 1: Load Data and Gene Sets")
print("=" * 60)

# IMPORTANT: Target slice
slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata.copy()

# IMPORTANT: Species
species = 'human'  # <-- SET SPECIES: 'human' or 'mouse'
organism = 'Human' if species == 'human' else 'Mouse'

# IMPORTANT: Gene set library
# Options: 'MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2023', 'Reactome_2022'
library_name = 'MSigDB_Hallmark_2020'  # <-- SET LIBRARY

# Load gene sets
gene_sets = gp.get_library(library_name, organism=organism)
print(f"  Data: {adata.n_obs} cells/spots, {adata.n_vars} genes")
print(f"  Library: {library_name} ({len(gene_sets)} gene sets)")

# IMPORTANT: Select specific pathways (or use all)
# Set to None to score ALL pathways, or provide a list of names
selected_pathways = None  # <-- e.g., ['HALLMARK_INFLAMMATORY_RESPONSE', 'HALLMARK_HYPOXIA']

if selected_pathways:
    gene_sets = {k: v for k, v in gene_sets.items() if k in selected_pathways}
    print(f"  Selected {len(gene_sets)} pathways")
else:
    # Filter by size for efficiency
    gene_sets = {k: v for k, v in gene_sets.items() if 10 <= len(v) <= 500}
    print(f"  Using {len(gene_sets)} pathways (size 10-500)")
```

### Stage 2: Prepare Expression Matrix

```python
print("\n" + "=" * 60)
print("STAGE 2: Prepare Expression Matrix")
print("=" * 60)

# Normalize for scoring
import scanpy as sc
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Expression matrix: genes × cells (ssGSEA expects genes as rows)
expr_df = adata.to_df().T
print(f"  Expression matrix: {expr_df.shape[0]} genes × {expr_df.shape[1]} cells")

# Check gene overlap
all_gs_genes = set()
for genes in gene_sets.values():
    all_gs_genes.update(genes)
overlap = set(expr_df.index) & all_gs_genes
print(f"  Gene overlap with library: {len(overlap)}")
```

### Stage 3: Run ssGSEA

```python
print("\n" + "=" * 60)
print("STAGE 3: Run ssGSEA")
print("=" * 60)

print(f"  Scoring {len(gene_sets)} pathways across {adata.n_obs} cells...")

ss = gp.ssgsea(
    data=expr_df,
    gene_sets=gene_sets,
    outdir=None,
    no_plot=True,
    verbose=False,
    min_size=5,
)

# Results: pathways × cells
scores = ss.res2d.pivot(index='Term', columns='Name', values='NES')
print(f"  Scored {scores.shape[0]} pathways")
print(f"  Score range: [{scores.values.min():.3f}, {scores.values.max():.3f}]")
```

### Stage 4: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 4: Store Results")
print("=" * 60)

# Store each pathway score as a column in adata.obs for spatial visualization
for pathway in scores.index:
    col_name = f"ssgsea_{pathway[:50]}"  # Truncate long names
    slice_obj.adata.obs[col_name] = scores.loc[pathway, slice_obj.adata.obs_names].values

# Store full score matrix in uns for reference
slice_obj.adata.uns['ssgsea_scores'] = scores
slice_obj.adata.uns['ssgsea_params'] = {
    'library': library_name,
    'species': species,
    'n_pathways': len(scores),
    'pathways': scores.index.tolist(),
}

print(f"  Stored {len(scores)} pathway scores in adata.obs (columns: ssgsea_*)")
print(f"  Full score matrix in adata.uns['ssgsea_scores']")

# Print top variable pathways (most spatially interesting)
pathway_var = scores.var(axis=1).sort_values(ascending=False)
print(f"\nTop variable pathways (highest spatial variation):")
for i, (pathway, var) in enumerate(pathway_var.head(10).items()):
    print(f"  {i+1}. {pathway} (variance={var:.4f})")
```

## Visualization

### Spatial Plot of Pathway Activity

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Pick a pathway to visualize
pathway_col = list(slice_obj.adata.obs.columns[slice_obj.adata.obs.columns.str.startswith('ssgsea_')])[0]

fig, ax = plt.subplots(figsize=(8, 8))
x = slice_obj.adata.obs['x'].values
y = slice_obj.adata.obs['y'].values
scores_val = slice_obj.adata.obs[pathway_col].values

scatter = ax.scatter(x, y, c=scores_val, cmap='RdYlBu_r', s=3, alpha=0.8)
ax.set_title(pathway_col.replace('ssgsea_', ''))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.invert_yaxis()
plt.colorbar(scatter, ax=ax, label='ssGSEA score', shrink=0.7)
plt.tight_layout()
plt.show()
```

---

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `species` | `'human'` | `'human'`, `'mouse'` | Must match gene names |
| `library_name` | `'MSigDB_Hallmark_2020'` | See below | Gene set database |
| `selected_pathways` | `None` (all) | List of pathway names | Specific pathways to score |

**Available libraries**: `MSigDB_Hallmark_2020`, `KEGG_2021_Human`, `Reactome_2022`, `GO_Biological_Process_2023`, `WikiPathways_2024_Human`

## Notes

- ssGSEA scores are **relative** (not p-values). Higher scores indicate stronger pathway activation in that cell.
- Scores are stored in `adata.obs` so they can be visualized spatially — "where is the inflammatory pathway active?"
- For large datasets (>50k cells), consider scoring only selected pathways for speed.
- The most biologically interesting pathways often have the **highest variance** across cells — use this to prioritize visualization.
