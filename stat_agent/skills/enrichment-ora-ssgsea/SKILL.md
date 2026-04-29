---
name: enrichment-ora
title: Over-Representation & Pathway Enrichment Analysis (ORA)
slug: enrichment-ora
description: Test whether a gene list is enriched for specific pathways or gene sets using Over-Representation Analysis (Fisher's exact test). Supports KEGG, Reactome, MSigDB Hallmark, WikiPathways, and GO databases via gseapy.

filter_requirements:
  modalities: [gene]

prerequisites:
  - "A gene list (user-provided, or shown how to obtain from data such as DE results, SVG results, or any other gene selection)"
  - Species (human or mouse)

default_skill: False
---

# Over-Representation & Pathway Enrichment Analysis (ORA)

Test whether a gene list is **enriched for specific pathways or gene sets** using Fisher's exact test. Supports **KEGG, Reactome, MSigDB Hallmark, WikiPathways**, and GO databases.

**Input**: A gene list from any source — user-provided, or knowing how to obtain from DE results, from spatially variable genes, or any other gene selection method.

**Output**: Enriched pathways with p-values, odds ratios in `adata.uns['ora_results']`.

## Core Function

```python
import gseapy as gp

enr = gp.enrich(
    gene_list=gene_list,
    gene_sets=gene_sets,
    background=adata.var_names.tolist(),
    outdir=None,
    no_plot=True,
)
results_df = enr.results.sort_values('Adjusted P-value')
```

## Workflow

### Stage 1: Prepare Gene List and Run ORA

```python
import numpy as np
import pandas as pd
import gseapy as gp

# IMPORTANT: Target slice
slice_id = 0  # <-- SET TARGET SLICE
slice_obj = session.get_slice(slice_id)
adata = slice_obj.adata

# IMPORTANT: Species
species = 'human'  # <-- SET SPECIES: 'human' or 'mouse'
organism = 'Human' if species == 'human' else 'Mouse'

# IMPORTANT: Gene list — from user input or derived from data
gene_list = []  # <-- Fill with gene names

background = adata.var_names.tolist()

# IMPORTANT: Choose library
# Options: 'KEGG_2021_Human', 'Reactome_2022', 'MSigDB_Hallmark_2020',
#          'WikiPathways_2024_Human', 'GO_Biological_Process_2023'
library_name = 'KEGG_2021_Human'  # <-- SET LIBRARY

gene_sets = gp.get_library(library_name, organism=organism)
gene_sets_filtered = {
    name: genes for name, genes in gene_sets.items()
    if 10 <= len(genes) <= 500
}

enr = gp.enrich(
    gene_list=gene_list,
    gene_sets=gene_sets_filtered,
    background=background,
    outdir=None,
    no_plot=True,
    verbose=False,
)

results_df = enr.results.sort_values('Adjusted P-value')
fdr_threshold = 0.05
significant = results_df[results_df['Adjusted P-value'] < fdr_threshold]

# Store
slice_obj.adata.uns['ora_results'] = results_df

print(f"Tested {len(results_df)} pathways, Significant (FDR < {fdr_threshold}): {len(significant)}")
for _, row in significant.head(10).iterrows():
    print(f"  {row['Term']}: FDR={row['Adjusted P-value']:.2e}")
```

## Visualization

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = slice_obj.adata.uns['ora_results']
sig = results[results['Adjusted P-value'] < 0.05].head(20).copy()

if len(sig) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(sig) * 0.4)))
    sig['-log10(FDR)'] = -np.log10(sig['Adjusted P-value'].clip(lower=1e-50))
    labels = sig['Term'].apply(lambda x: x[:60] + '...' if len(str(x)) > 60 else x)
    ax.barh(range(len(sig)), sig['-log10(FDR)'].values, color='steelblue')
    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('-log10(FDR)')
    ax.set_title(f'Top Enriched Pathways ({library_name})')
    plt.tight_layout()
    plt.show()
```

## Parameter Guide

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `species` | `'human'` | `'human'`, `'mouse'` | Must match gene name format |
| `library_name` | `'KEGG_2021_Human'` | See above | Gene set database |
| `fdr_threshold` | `0.05` | 0.01-0.25 | Significance cutoff |
