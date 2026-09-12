# Adding a skill

A skill is one directory under `src/stat_agent/skills/<slug>/` (built-in) or
`~/.stat_agent/skills/<slug>/` (user, takes precedence) containing a `SKILL.md`: YAML frontmatter
that states *when the skill applies and what it needs*, followed by Markdown instructions and
python code blocks that the generator adapts to the session.

## 1. Scaffold

```bash
stat-agent skill new my-method --title "My Method"
```

This writes `~/.stat_agent/skills/my-method/SKILL.md` with a v2 frontmatter template.

## 2. Frontmatter

```yaml
---
name: my-method
title: My Method
slug: my-method
description: One or two sentences the matcher reads to decide whether the skill fits a request.
filter_requirements:          # deterministic applicability contract
  modalities: [gene]          # gene | protein | gene/protein
  data_levels: [cell/spot]    # cell | spot | cell/spot
  min_slices: 2               # or num_slices: 2 with positional modalities/data_levels lists
prerequisites:                # what the verifier must confirm or ask for
  - "Cell type annotations in the target slice (adata.obs['celltype'])"
  - "Species (human or mouse)"
schema_version: 2
version: "1.0"
consumes: [obs:celltype]      # inputs read from the session (kind:key)
produces: [obs:my_label, uns:my_params]
platforms: [linux, macos]
dependencies:
  python: [scanpy, my-package]
  gpu: false
defaults:                     # answers auto mode may assume for a prerequisite
  "Species (human or mouse)": "human"
---
```

`consumes` drives staleness checks (a stale input blocks the step before any code runs) and the
provenance graph. `produces` is checked once the step's code has run (`validate_result`): every
declared output must exist and must have been written by this step, otherwise the generator gets one
continuation on the live namespace and the step is reported as partial if it is still missing. Declare only
outputs that every run of the skill writes onto the target slice; a skill whose outputs depend on the
chosen mode, or that creates a new slice, declares none and the check is skipped. The linter also
checks `produces` against the code. Keys accept globs (`uns:*_nhood_enrichment`).

## 3. Body

Explain the workflow, then give runnable code that uses explicit slice access:

````markdown
## Workflow

```python
slice_id = 0  # the generator substitutes the target slice
adata = session.get_slice(slice_id).adata
# ... analysis ...
adata.obs['my_label'] = labels
print(f"labelled {adata.n_obs} cells")
```
````

Rules that the linter enforces or warns about: python blocks must parse; every declared `produces`
should be written by the body (or by a known library effect such as `sc.tl.leiden`); imports should
be listed under `dependencies.python`; vendored packages next to `SKILL.md` are added to `sys.path`
for the run.

## 4. Validate and try it

```bash
stat-agent skill validate --strict            # every skill in the layered registry
stat-agent skill lint-file path/to/SKILL.md   # one file
stat-agent chat --model openai/gpt-4o --dataset ./dataset
```

The web interface lists user skills next to the built-in ones and can disable any of them.

## 5. Drafting from a successful unguarded run

After the agent executes an unregistered method, the interface offers a draft. The same can be
done from the API: `POST /api/skills/draft {turn_id, step}` writes the draft into the user skills
directory and reloads the registry. Review the frontmatter before relying on it.

A worked example (annotating the STAGATE skill) took about twenty minutes: fifteen for the code
block and five for the contract.
