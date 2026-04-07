---
name: cnv-inference
title: Expression-based CNV Inference (infercnvpy)
slug: cnv-inference
description: Infer copy number variations (CNVs) from gene expression data using infercnvpy. Identifies tumor subclones by comparing expression patterns against normal reference cells. Useful for cancer spatial transcriptomics to detect genomic alterations and clone structure.

filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [cell/spot]

prerequisites:
  - Cell type annotations in adata.obs['celltype']
  - Identification of normal/reference cell types (e.g., 'Fibroblast', 'Endothelial')
  - Genomic position annotation in adata.var (columns: 'chromosome', 'start', 'end') — if missing, will be added from gene name lookup

default_skill: False
---

# Expression-based CNV Inference (infercnvpy)

Infer **copy number variations** from gene expression by comparing tumor cells against normal reference cells. Identifies CNV patterns, tumor subclones, and genomic alterations — essential for cancer spatial transcriptomics.

**Output**:
- `adata.obsm['X_cnv']`: CNV score matrix (cells x genomic windows)
- `adata.obs['cnv_leiden']`: CNV-based clone clusters

---

## Workflow

### Stage 1: Load and Validate

```python
import numpy as np
import pandas as pd
import scanpy as sc

print("=" * 60)
print("STAGE 1: Load and Validate")
print("=" * 60)

# IMPORTANT: Target slice
slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata.copy()

# Validate celltype
assert 'celltype' in adata.obs.columns, "Need cell type annotations for reference cells"

# IMPORTANT: Reference (normal) cell types
reference_cat = ['Fibroblast', 'Endothelial']  # <-- SET NORMAL CELL TYPES
reference_key = 'celltype'

# Validate reference exists
found = [c for c in reference_cat if c in adata.obs[reference_key].values]
assert len(found) > 0, (
    f"None of {reference_cat} found in celltype. "
    f"Available: {adata.obs[reference_key].unique().tolist()}"
)

n_ref = sum(adata.obs[reference_key].isin(found))
print(f"  Data: {adata.n_obs} cells/spots, {adata.n_vars} genes")
print(f"  Reference cells ({', '.join(found)}): {n_ref}")
print(f"  Non-reference cells: {adata.n_obs - n_ref}")
```

### Stage 2: Run infercnvpy

```python
print("\n" + "=" * 60)
print("STAGE 2: Run CNV Inference")
print("=" * 60)

import infercnvpy as cnv

# Normalize if needed
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Run infercnv
cnv.tl.infercnv(
    adata,
    reference_key=reference_key,
    reference_cat=found,
    window_size=250,
)

print(f"  CNV inference complete")
print(f"  CNV matrix shape: {adata.obsm['X_cnv'].shape}")

# CNV-based dimensionality reduction and clustering
cnv.tl.pca(adata)
cnv.tl.leiden(adata, resolution=0.5)

n_clones = adata.obs['cnv_leiden'].nunique()
print(f"  CNV-based clusters: {n_clones}")
```

### Stage 3: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 3: Store Results")
print("=" * 60)

# Store CNV results
slice_obj.adata.obsm['X_cnv'] = adata.obsm['X_cnv']
slice_obj.adata.obs['cnv_leiden'] = adata.obs['cnv_leiden'].values
slice_obj.adata.uns['cnv_params'] = {
    'method': 'infercnvpy',
    'reference_key': reference_key,
    'reference_cat': found,
    'n_clones': n_clones,
    'window_size': 250,
}

print(f"  Stored X_cnv in adata.obsm ({adata.obsm['X_cnv'].shape})")
print(f"  Stored cnv_leiden in adata.obs ({n_clones} clones)")

# Summary per clone
print(f"\nCNV clone distribution:")
for clone in sorted(adata.obs['cnv_leiden'].unique()):
    n = (adata.obs['cnv_leiden'] == clone).sum()
    print(f"  Clone {clone}: {n} cells")
```

## Visualization

### CNV Heatmap

```python
import matplotlib
matplotlib.use('Agg')
import infercnvpy as cnv

cnv.pl.chromosome_heatmap(adata, groupby='cnv_leiden', show=False)
import matplotlib.pyplot as plt
plt.tight_layout()
plt.show()
```

### Spatial CNV Clones

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
x = slice_obj.adata.obs['x'].values
y = slice_obj.adata.obs['y'].values
clones = slice_obj.adata.obs['cnv_leiden']
categories = sorted(clones.unique())
colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

for i, c in enumerate(categories):
    mask = clones == c
    ax.scatter(x[mask], y[mask], c=[colors[i]], s=3, label=f'Clone {c}', alpha=0.7)

ax.set_title('CNV-based Clones (spatial)')
ax.set_aspect('equal')
ax.invert_yaxis()
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
plt.tight_layout()
plt.show()
```

---

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `reference_cat` | — | List of normal cell types | Baseline for CNV detection |
| `window_size` | 250 | 100-500 | Genomic window size (genes) |
| `resolution` | 0.5 | 0.1-2.0 | CNV clustering resolution |

## Notes

- infercnvpy infers CNVs from expression — not actual genomic data. Results are indicative, not definitive.
- The quality of results depends heavily on choosing good reference (normal) cell types.
- Works best with cancer spatial data where tumor cells have clear CNV patterns.
- Install: `pip install infercnvpy`.
