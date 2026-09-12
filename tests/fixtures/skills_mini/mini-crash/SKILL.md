---
schema_version: 2
name: mini-crash
title: Mini Crash
slug: mini-crash
description: A skill whose generated code always fails on the first attempt (test skill for reflection and recovery).
filter_requirements:
  modalities: [gene]
prerequisites: []
default_skill: true
---

# Mini crash

```python
raise KeyError('boom')
```
