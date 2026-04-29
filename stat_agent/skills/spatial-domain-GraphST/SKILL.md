---
name: spatial-domain-graphst
title: Spatial Domain Detection (GraphST)
slug: spatial-domain-graphst
description: Identify spatial domains in spot-level data using GraphST (Graph Self-supervised Transformer). Uses contrastive self-supervised learning on spatial transcriptomics graphs to learn domain-specific embeddings. Robust to noise and batch effects.

filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [spot]

prerequisites:
  - "Number of expected spatial domains (optional, default auto-detected)"

default_skill: false
---

# Spatial Domain Detection (GraphST)

Identify **spatial domains** in spot-level data using **GraphST** — a graph self-supervised transformer that uses contrastive learning to learn spatially coherent embeddings.

**Output**: `adata.obs['spatial_domain']` — domain label per spot.

---

## Workflow

### Stage 1: Load and Preprocess

```python
import numpy as np
import pandas as pd
import scanpy as sc

print("=" * 60)
print("STAGE 1: Load and Preprocess")
print("=" * 60)

# IMPORTANT: Target slice
slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata.copy()

# Spatial coordinates
adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()

# Normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)

print(f"  Data: {adata.n_obs} spots, {adata.n_vars} genes")
print(f"  HVGs: {sum(adata.var['highly_variable'])}")
```

### Stage 2: Run GraphST

```python
print("\n" + "=" * 60)
print("STAGE 2: Run GraphST")
print("=" * 60)

from GraphST.GraphST import GraphST as GraphSTModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")

model = GraphSTModel(adata, device=device)
adata = model.train()

print(f"  GraphST training complete")
print(f"  Embedding shape: {adata.obsm['emb'].shape}")
```

### Stage 3: Cluster into Domains

```python
print("\n" + "=" * 60)
print("STAGE 3: Cluster into Spatial Domains")
print("=" * 60)

import GraphST as graphst_pkg

# IMPORTANT: Number of domains
n_domains = 7  # <-- SET NUMBER OF DOMAINS (or None for auto)

if n_domains:
    # Use GraphST's built-in clustering (supports 'mclust' and 'leiden')
    # 'mclust' uses GMM internally; 'leiden' uses binary search for target n_clusters
    try:
        graphst_pkg.clustering(adata, n_clusters=n_domains, key='emb', method='mclust', refinement=True)
        adata.obs['spatial_domain'] = adata.obs['domain']
        print(f"  Clustered into domains (mclust)")
    except Exception as e:
        print(f"  mclust failed ({e}), falling back to leiden")
        graphst_pkg.clustering(adata, n_clusters=n_domains, key='emb', method='leiden', refinement=True)
        adata.obs['spatial_domain'] = adata.obs['domain']
else:
    # Auto: use leiden at default resolution
    sc.pp.neighbors(adata, use_rep='emb', n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5)
    adata.obs['spatial_domain'] = adata.obs['leiden']

adata.obs['spatial_domain'] = pd.Categorical(adata.obs['spatial_domain'].astype(str))
n_domains_found = adata.obs['spatial_domain'].nunique()
print(f"  Found {n_domains_found} domains")

for d in sorted(adata.obs['spatial_domain'].unique()):
    n = (adata.obs['spatial_domain'] == d).sum()
    print(f"  Domain {d}: {n} spots")
```

### Stage 4: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 4: Store Results")
print("=" * 60)

slice_obj.adata.obs['spatial_domain'] = adata.obs['spatial_domain'].values
slice_obj.adata.uns['spatial_domain_params'] = {
    'method': 'GraphST',
    'n_domains': n_domains_found,
    'device': device,
}

print(f"  Stored spatial_domain in adata.obs ({n_domains_found} domains)")
```

## Visualization

### Spatial Domain Map

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
x = slice_obj.adata.obs['x'].values
y = slice_obj.adata.obs['y'].values
domains = slice_obj.adata.obs['spatial_domain']
categories = sorted(domains.unique())
colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

for i, d in enumerate(categories):
    mask = domains == d
    ax.scatter(x[mask], y[mask], c=[colors[i]], s=3, label=f'Domain {d}', alpha=0.7)

ax.set_title('GraphST Spatial Domains')
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
| `n_domains` | 7 | 2-20 or None (auto) | Number of spatial domains |
| `method` | `'mclust'` | `'mclust'`, `'leiden'` | Clustering method on embeddings |
| `refinement` | `True` | True/False | Spatial refinement of domain labels |
| `n_top_genes` | 3000 | 1000-5000 | HVGs for model input |

## Notes

- GraphST uses contrastive self-supervised learning — no labeled data needed.
- Automatically uses GPU if available.
- The built-in `GraphST.clustering()` supports `method='mclust'` (GMM-based, needs target n_clusters) and `method='leiden'` (binary search for target n_clusters). If mclust fails, leiden is used as fallback.
- Install: `pip install GraphST`.
