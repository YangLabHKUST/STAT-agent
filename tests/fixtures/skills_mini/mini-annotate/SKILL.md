---
schema_version: 2
name: mini-annotate
title: Mini Cell Type Annotation
slug: mini-annotate
description: Annotate cell types on a cell-level gene expression slice using a tiny deterministic rule (test skill).
filter_requirements:
  modalities: [gene]
  data_levels: [cell/spot]
prerequisites:
  - "Tissue type information (e.g., breast cancer, brain cortex)"
produces:
  - obs:celltype
postconditions:
  - artifact: obs:celltype
    check: exists
default_skill: true
---

# Mini annotation

```python
s = session.get_slice(0)
s.adata.obs['celltype'] = 'T cell'
print('annotated')
```
