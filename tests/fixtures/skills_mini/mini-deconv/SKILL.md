---
schema_version: 2
name: mini-deconv
title: Mini Deconvolution
slug: mini-deconv
description: Deconvolve spot-level data with a reference (test skill); spot-level only.
filter_requirements:
  modalities: [gene]
  data_levels: [spot]
prerequisites:
  - "Annotated single-cell reference dataset path (.h5ad file)"
produces:
  - obsm:deconv_weights
default_skill: true
---

# Mini deconv

```python
print('deconv')
```
