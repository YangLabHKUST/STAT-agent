---
schema_version: 2
name: mini-niche
title: Mini Niche Detection
slug: mini-niche
description: Detect spatial niches from cell type labels (test skill); requires cell type annotations.
filter_requirements:
  modalities: [gene]
  data_levels: [cell]
prerequisites:
  - "Cell type annotations in target slice (adata.obs['celltype'])"
consumes:
  - obs:celltype
produces:
  - obs:niche_label
default_skill: true
---

# Mini niche

```python
s = session.get_slice(0)
s.adata.obs['niche_label'] = 'n1'
```
