---
name: spatial-domain-stagate
title: Spatial Domain Detection (STAGATE)
slug: spatial-domain-stagate
description: Identify spatial domains using STAGATE (Spatial-Transcriptomics Graph Attention Auto-Encoder). Uses graph attention networks on spatial neighbor graphs to learn spatially-aware cell embeddings, then clusters them into spatial domains. Works on both cell-level and spot-level data.

filter_requirements:
  num_slices: 1
  modalities: [gene]
  data_levels: [cell/spot]

prerequisites:
  - Number of expected spatial domains (optional, default auto-detected via leiden)

default_skill: false
---

# Spatial Domain Detection (STAGATE)

Identify **spatial domains** using **STAGATE** — a graph attention auto-encoder that learns spatially-aware embeddings by jointly modeling gene expression and spatial neighbor relationships. More expressive than SpaGCN due to attention mechanisms.

**Output**: `adata.obs['spatial_domain']` — domain label per cell/spot.

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
adata = adata[:, adata.var['highly_variable']].copy()

print(f"  Data: {adata.n_obs} cells/spots, {adata.n_vars} HVGs")
```

### Stage 2: Build Spatial Graph and Run STAGATE

```python
print("\n" + "=" * 60)
print("STAGE 2: Run STAGATE")
print("=" * 60)

import STAGATE_pyG as STAGATE

# Build spatial neighbor graph
# Ensure float64 for spatial coordinates
adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float64)

# Build spatial graph using radius cutoff (recommended over k_cutoff for stability)
# Adjust rad_cutoff based on coordinate scale: ~150 for pixel coords, ~50 for normalized
STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
print(f"  Spatial graph built")

# Train STAGATE
adata = STAGATE.train_STAGATE(
    adata,
    hidden_dims=[512, 30],
    n_epochs=1000,
    lr=0.001,
    random_seed=42,
)

print(f"  STAGATE training complete")
print(f"  Embedding shape: {adata.obsm['STAGATE'].shape}")
```

### Stage 3: Cluster into Domains

```python
print("\n" + "=" * 60)
print("STAGE 3: Cluster into Spatial Domains")
print("=" * 60)

# IMPORTANT: Number of domains
n_domains = 7  # <-- SET NUMBER OF DOMAINS (or None for auto)

# Use STAGATE embeddings for clustering
sc.pp.neighbors(adata, use_rep='STAGATE', n_neighbors=15)

if n_domains:
    # GMM clustering on STAGATE embeddings (equivalent to mclust EEE)
    # Use 'tied' covariance for stability; fall back to leiden if GMM fails
    try:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_domains, covariance_type='tied', random_state=42)
        labels = gmm.fit_predict(adata.obsm['STAGATE'])
        adata.obs['spatial_domain'] = pd.Categorical([str(l) for l in labels])
        print(f"  Clustered into {n_domains} domains (GMM tied)")
    except Exception as e:
        print(f"  GMM failed ({e}), falling back to leiden")
        sc.tl.leiden(adata, resolution=0.5)
        adata.obs['spatial_domain'] = adata.obs['leiden']
else:
    sc.tl.leiden(adata, resolution=0.5)
    adata.obs['spatial_domain'] = adata.obs['leiden']
    n_domains = adata.obs['spatial_domain'].nunique()
    print(f"  Clustered into {n_domains} domains (Leiden auto)")

# Summary
for d in sorted(adata.obs['spatial_domain'].unique()):
    n = (adata.obs['spatial_domain'] == d).sum()
    print(f"  Domain {d}: {n} cells/spots")
```

### Stage 4: Store Results

```python
print("\n" + "=" * 60)
print("STAGE 4: Store Results")
print("=" * 60)

slice_obj.adata.obs['spatial_domain'] = adata.obs['spatial_domain'].values
slice_obj.adata.uns['spatial_domain_params'] = {
    'method': 'STAGATE',
    'n_domains': n_domains,
    'hidden_dims': [512, 30],
    'n_epochs': 1000,
}

print(f"  Stored spatial_domain in adata.obs ({n_domains} domains)")
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

ax.set_title('STAGATE Spatial Domains')
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
| `k_cutoff` | 10 | 4-30 | Spatial neighbors |
| `hidden_dims` | [512, 30] | — | Autoencoder hidden dimensions |
| `n_epochs` | 1000 | 200-2000 | Training epochs |

## Notes

- STAGATE uses graph attention to weigh spatial neighbors differently — more expressive than SpaGCN.
- GPU is used automatically if available (via PyTorch).
- The `STAGATE_pyG` package uses PyTorch Geometric. Install: `pip install STAGATE_pyG`.
- If `STAGATE_pyG` is not available, try `pip install stagate-package` which provides the same functionality.
