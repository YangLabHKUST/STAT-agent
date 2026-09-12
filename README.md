<!-- BANNER -->

<div align="center">

# STAT — Spatial Transcriptomics Analytical agenT

Ask in natural language, get a planned, verified, and executed analysis of spatial omics data.

[![PyPI version](https://img.shields.io/pypi/v/stat-agent.svg?color=blue)](https://pypi.org/project/stat-agent/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.05.01.722244-b31b1b.svg)](https://doi.org/10.64898/2026.05.01.722244)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗_Demo-HF_Spaces-yellow.svg)](https://huggingface.co/spaces/CyhVVVV/stat-agent-demo)

</div>


## Table of contents

- [What is new in 2.0](#what-is-new-in-20)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Data format](#data-format)
- [Built-in skills](#built-in-skills)
- [LLM providers](#llm-providers)
- [Benchmarks and reproducibility](#benchmarks-and-reproducibility)
- [Documentation](#documentation)
- [License](#license)

## What is new in 2.0

STAT 2.0 is a rewrite of the agent on the [LangGraph](https://github.com/langchain-ai/langgraph)
runtime. The workflow is the same (plan, filter, match, verify, generate, execute, reflect,
analyse, with planner, verifier and skill-selection clarifications), the implementation is new:

- **Typed staged graph.** Each stage is a node with typed state; clarifications are checkpointed
  interrupts, so a browser reload or a restart resumes at the pending question.
- **Provenance ledger.** Every artefact (`obs` column, `obsm` matrix, ROI, figure) records what
  produced it and what it read. Deleting an ROI or correcting an annotation marks the dependants
  stale; the interface shows them and can re-run them downstream.
- **Out-of-registry requests.** A named method without a skill gets a provisional method card that passes
  the same filter and verifier. Unmet requirements are refused by name before any code runs;
  successful runs can be drafted into a new skill.
- **Auto mode.** Every clarification can be answered by policy (skill defaults, known facts, session
  state) and recorded as an overridable assumption; what cannot be defaulted is refused.
- **Loop mode.** The same tools and guards mounted on a single tool-calling agent (LangChain
  `create_agent`, optionally `deepagents` with subagents) for comparisons and agentic use.
- **Guarded in-process runtime.** AST safety checks, routed output, timeouts, snapshots and figure
  capture; the canvas and the session are two views of one live object.
- **Robust web layer.** Starlette + uvicorn with a single ordered, resumable event stream
  (`Last-Event-ID`), a turn API with typed replies, cooperative cancel and durable sessions.
- **Skills v2.** Frontmatter declares `consumes`, `produces`, `platforms`,
  `dependencies` and `defaults`; a linter checks them against the code. All 26 skills are annotated.
- **`stat-bench`.** Ablation arms, auto versus interactive, repeated seeds, SessionBench scripts,
  frontmatter corruption and silent-error detectors, all as command-line switches.

## Installation

Python 3.11 or newer.

```bash
pip install stat-agent
```

Extras:

```bash
pip install "stat-agent[skills]"     # analysis skill dependencies (squidpy, scvi-tools, torch, ...)
pip install "stat-agent[loop]"       # deepagents-based loop mode with subagents
pip install "stat-agent[research]"   # web lookup for the method researcher (TAVILY_API_KEY)
pip install "stat-agent[dev,bench]"  # tests and benchmark tooling
```

Some skills need packages that are not on PyPI (STAGATE with PyG wheels matching your torch and
CUDA build); see the skill's `SKILL.md` and its `dependencies` block.

## Quick start

### Web interface

```bash
stat-web                    # http://localhost:8889
./start_web.sh              # also starts a Jupyter Lab alongside on 8890
```

1. Enter the path to your dataset directory (or a saved session directory).
2. Pick a provider, model and API key, then *Test Connection*.
3. Load the dataset and ask in the chat panel:
   - *"Annotate cell types using the breast-cancer reference."*
   - *"Find spatially variable genes."*
   - *"Show CD8A expression in slice 1."*
   - *"Run RCTD deconvolution and overlay the dominant cell type."*

The mode selector switches between interactive (questions pause the turn) and auto (questions are
answered by policy and shown as assumptions you can override). The *Changes* drawer lists what each
turn produced, what is stale and why.

### Command line

```bash
export OPENAI_API_KEY=...
stat-agent chat --model openai/gpt-4o --dataset ./dataset          # REPL
stat-agent chat --model openai/gpt-4o --dataset ./dataset -m "annotate cell types" --mode auto
stat-agent session report ./logs/session_x --format html --out report.html
stat-agent skill validate --strict
```

<details>
<summary>A chat turn in the interface</summary>

<p align="center">
  <a href="demo_chat_turn.jpg">
    <img src="demo_chat_turn.jpg" alt="STAT one chat-turn demo" width="500" />
  </a>
</p>

</details>

## Data format

STAT auto-detects your data layout. Place files in a single directory.

**Single-slice**

```text
dataset/
├── tissue.h5ad          # Required: AnnData with x, y in obs
└── he.tif               # Optional: H&E image (pixel coords = cell coords)
```

**Multi-slice**

```text
dataset/
├── tissue_slice_0.h5ad
├── he_slice_0.tif
├── tissue_slice_1.h5ad
└── he_slice_1.tif
```

**Multi-omics (gene + protein)**

```text
dataset/
├── tissue.h5ad          # Gene expression
├── tissue_protein.h5ad  # Protein expression
├── he.tif
└── protein_CD3.tif
```

**Coordinate convention.** Cell coordinates `(x, y)` in `adata.obs` map directly to image pixel
`(x, y)`. No coordinate transformation. Note the array indexing swap: image array `img[y, x]`
corresponds to cell `(x, y)`.

**Required AnnData fields:** `adata.obs['x']`, `adata.obs['y']`, and the expression matrix
`adata.X`. `adata.obs['celltype']` is *optional*; annotation skills populate it.

## Built-in skills

Skills are discovered from `src/stat_agent/skills/<slug>/SKILL.md` and from `~/.stat_agent/skills/`
(user skills take precedence). Each skill carries an applicability contract (modalities, data level,
slice count), prerequisites, declared inputs and outputs, and a templated code body. The catalogue:

<!-- SKILLS-TABLE-START -->
### Cell type annotation

| Skill | Summary |
| --- | --- |
| [Cell Type Annotation with scANVI](src/stat_agent/skills/celltype-annotation-scANVI/SKILL.md) | Annotate cell types in spatial transcriptomics data using scANVI transfer learning from a reference scRNA-seq dataset. |
| [Fast Cell Type Annotation (Clustering + LLM)](src/stat_agent/skills/celltype-annotation-GPT/SKILL.md) | Annotate cell types using unsupervised clustering, marker genes, and LLM-based annotation. |
| [Cell Type Annotation via Spatial Mapping (Tangram)](src/stat_agent/skills/annotation-tangram/SKILL.md) | Map single-cell reference annotations onto spatial transcriptomics data using Tangram deep learning alignment. |

### Spot deconvolution

| Skill | Summary |
| --- | --- |
| [Cell Type Deconvolution (RCTD)](src/stat_agent/skills/celltype-deconvolution-RCTD/SKILL.md) | Perform cell type deconvolution (or annotation on spot) on spatial transcriptomics data (Visium spots) using RCTD with a single-cell refere… |
| [Bayesian Cell Type Deconvolution (Cell2location)](src/stat_agent/skills/deconvolution-cell2location/SKILL.md) | Reference-based Bayesian deconvolution of spot-level spatial transcriptomics using Cell2location. |
| [Fast Spot Deconvolution (FlashDeconv)](src/stat_agent/skills/deconvolution-flashdeconv/SKILL.md) | Ultra-fast reference-based cell type deconvolution for spot-level spatial data using FlashDeconv. |

### Spatial domains

| Skill | Summary |
| --- | --- |
| [Spatial Domain Detection (SpaGCN)](src/stat_agent/skills/spatial-domain-SpaGCN/SKILL.md) | Identify spatial domains in spot-level spatial transcriptomics data using SpaGCN, integrating gene expression, spatial location, and H&E hi… |
| [Spatial Domain Detection (STAGATE)](src/stat_agent/skills/spatial-domain-STAGATE/SKILL.md) | Identify spatial domains using STAGATE (Spatial-Transcriptomics Graph Attention Auto-Encoder). |
| [Spatial Domain Detection (GraphST)](src/stat_agent/skills/spatial-domain-GraphST/SKILL.md) | Identify spatial domains in spot-level data using GraphST (Graph Self-supervised Transformer). |

### Spatial statistics & niches

| Skill | Summary |
| --- | --- |
| [Spatial Statistics Analysis](src/stat_agent/skills/spatial-statistics-squidpy/SKILL.md) | Compute spatial statistics including Moran's I (spatial autocorrelation of genes), Ripley's K (spatial point pattern of cell types), co-occ… |
| [Neighborhood Enrichment Analysis](src/stat_agent/skills/spatial-stats-neighborhood-enrichment/SKILL.md) | Compute neighborhood enrichment z-scores to identify which cell types are spatially co-localized or depleted from each other's neighborhood… |
| [Spatial Niche Detection](src/stat_agent/skills/niche-detection-Harmonics/SKILL.md) | Identify spatial cellular niches using Harmonics hierarchical model. |
| [Spatially Variable Genes (SpatialDE)](src/stat_agent/skills/svg-SpatialDE/SKILL.md) | Identify spatially variable genes using SpatialDE Gaussian process regression. |

### Differential expression & pathway

| Skill | Summary |
| --- | --- |
| [Differential Gene Expression Analysis](src/stat_agent/skills/differential-expression/SKILL.md) | Find differentially expressed marker genes between groups using scanpy rank_genes_groups with Wilcoxon test. |
| [GO Enrichment Analysis](src/stat_agent/skills/pathway-GO-enrichment/SKILL.md) | Find enriched Gene Ontology (GO) terms for a user-provided gene list. |
| [Over-Representation & Pathway Enrichment Analysis (ORA)](src/stat_agent/skills/enrichment-ora-ssgsea/SKILL.md) | Test whether a gene list is enriched for specific pathways or gene sets using Over-Representation Analysis (Fisher's exact test). |
| [Per-Cell Pathway Activity Scoring (ssGSEA)](src/stat_agent/skills/pathway-ssgsea/SKILL.md) | Compute per-cell pathway activity scores using single-sample Gene Set Enrichment Analysis (ssGSEA). |
| [Two-Group Pathway Enrichment Comparison](src/stat_agent/skills/pathway-enrichment-compare/SKILL.md) | Compare pathway / gene-set enrichment between two user-provided gene lists (typically markers of two cell populations, clusters, or conditi… |

### Cell-cell communication

| Skill | Summary |
| --- | --- |
| [Cell-Cell Communication Analysis (LIANA+)](src/stat_agent/skills/cell-communication-LIANA/SKILL.md) | Analyze cell-cell communication using LIANA+ to identify significant ligand-receptor interactions between cell types. |
| [Cell-Cell Communication Analysis (CellPhoneDB)](src/stat_agent/skills/cell-communication-CellPhoneDB/SKILL.md) | Analyze cell-cell communication using CellPhoneDB statistical method to identify significant ligand-receptor interactions between cell type… |

### Multi-slice integration

| Skill | Summary |
| --- | --- |
| [Batch Integration (Harmony)](src/stat_agent/skills/integration-Harmony/SKILL.md) | Integrate multiple spatial transcriptomics slices using Harmony batch correction. |
| [Batch Integration (BBKNN)](src/stat_agent/skills/integration-bbknn/SKILL.md) | Correct batch effects across multiple slices using BBKNN (Batch Balanced K-Nearest Neighbors). |
| [Batch Integration (Scanorama)](src/stat_agent/skills/integration-scanorama/SKILL.md) | Correct batch effects across multiple slices using Scanorama panoramic stitching. |

### Slice alignment & registration

| Skill | Summary |
| --- | --- |
| [Spatial Alignment (STalign)](src/stat_agent/skills/alignment-STalign/SKILL.md) | Align two cell-level spatial transcriptomics slices using STalign. |
| [Slice Registration (PASTE)](src/stat_agent/skills/registration-paste/SKILL.md) | Align multiple spatial transcriptomics slices using PASTE (Probabilistic Alignment of ST Experiments). |

### Trajectory inference

| Skill | Summary |
| --- | --- |
| [Pseudotime Trajectory Analysis (Palantir / DPT)](src/stat_agent/skills/trajectory-palantir-dpt/SKILL.md) | Infer cell developmental trajectories and pseudotime ordering using expression-based methods. |

<!-- SKILLS-TABLE-END -->

**Adding a skill.** See [docs/adding-a-skill.md](docs/adding-a-skill.md): scaffold with
`stat-agent skill new <slug>`, fill in the contract, validate with `stat-agent skill validate --strict`.

## LLM providers

Models are addressed as `provider/model-id` (a bare id infers the provider). In the web interface
pick the provider and type the model id; on the command line export the key and pass `--model`
or set `STAT_MODEL`.

| Provider | Env var | Example |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-5.4`, `openai/gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-opus-4-7`, `anthropic/claude-sonnet-4-6` |
| Google Gemini | `GOOGLE_API_KEY` | `google/gemini-3.1-pro-preview`, `google/gemini-2.5-pro` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-v4-pro` |
| Poe (gateway) | `POE_API_KEY` | `poe/claude-sonnet-4.5`, `poe/gpt-5.5` |
| xAI | `XAI_API_KEY` | `xai/grok-4` |
| Moonshot | `MOONSHOT_API_KEY` | `moonshot/kimi-k2` |
| Zhipu | `ZHIPU_API_KEY` | `zhipu/glm-4.5` |
| DashScope | `DASHSCOPE_API_KEY` | `dashscope/qwen3-max` |
| Any OpenAI-compatible endpoint | `STAT_API_KEY` + `--base-url` | `custom/<model>` |

The staged pipeline uses prompted JSON for structured outputs, so the prompts are identical across
backbones and no tool calling is required. Loop mode needs a model with tool calling.

Other settings: `STAT_MODE` (`interactive` or `auto`), `STAT_DRIVER` (`staged` or `loop`),
`STAT_HOME` (default `~/.stat_agent`, holds user skills and the RCTD reference cache),
`STAT_LOGS_DIR`, `STAT_EXEC_TIMEOUT`, `STAT_SKILL_PATHS`.

## Benchmarks and reproducibility

`stat-bench` (installed with the package) runs query suites through STAT with any ablation arm,
clarification policy, driver, backbone and number of seeds, then scores and summarises the results
with bootstrap confidence intervals. See [docs/benchmarking.md](docs/benchmarking.md). The
analyses and figures of the STAT paper live in `https://github.com/chenyhvvvv/STAT-PaperRepro`.

## Documentation

- [docs/v2-architecture.md](docs/v2-architecture.md): packages, the staged graph, what is deterministic and what the model decides, provenance, loop mode.
- [docs/event-protocol.md](docs/event-protocol.md): the SSE event log, turn API, replies, cancel, errors, legacy aliases.
- [docs/adding-a-skill.md](docs/adding-a-skill.md): the skill contract and the linter.
- [docs/benchmarking.md](docs/benchmarking.md): `stat-bench` and SessionBench.

## License

[BSD-3-Clause](LICENSE) © STAT contributors.
