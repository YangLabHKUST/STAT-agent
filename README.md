<!-- BANNER -->

<div align="center">

# STAT — Spatial Transcriptomics Analytical agenT

Ask in natural language, get a planned, verified, and executed analysis of spatial omics data.

[![PyPI version](https://img.shields.io/pypi/v/stat-agent.svg?color=blue)](https://pypi.org/project/stat-agent/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.05.01.722244-b31b1b.svg)](https://doi.org/10.64898/2026.05.01.722244)
[![Demo](https://img.shields.io/badge/demo-statagent.online-2563eb.svg)](https://statagent.online/)

</div>

STAT is an AI agent for spatial transcriptomics. Ask in plain language, and STAT plans the analysis,
picks a method, checks your data, runs the code, fixes errors and explains the result, next to an
interactive tissue viewer.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Data format](#data-format)
- [Built-in skills](#built-in-skills)
- [LLM providers](#llm-providers)
- [Benchmarks and reproducibility](#benchmarks-and-reproducibility)
- [License](#license)

## Installation

Python 3.11 or newer.

```bash
pip install stat-agent
pip install "git+https://github.com/chenyhvvvv/STAT-agent.git"   # or the latest main from GitHub
```

The analysis skills' dependencies (squidpy, scvi-tools, torch, ...):

```bash
pip install "stat-agent[skills]"
```

Some skills need packages that are not on PyPI (STAGATE with PyG wheels matching your torch and
CUDA build); see the skill's `SKILL.md` and its `dependencies` block.

## Quick start

To try STAT without installing it, open the live demo at [statagent.online](https://statagent.online/).

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
export ANTHROPIC_API_KEY=...
stat-agent chat ./dataset --model anthropic/claude-sonnet-4-6           # REPL
stat-agent chat ./dataset --model anthropic/claude-sonnet-4-6 -m "annotate cell types" --mode auto
stat-agent session report ./logs/session_x --format html --out report.html
stat-agent skill validate --strict
```

<details>
<summary>A chat turn in the interface</summary>

<p align="center">
  <a href="https://raw.githubusercontent.com/chenyhvvvv/STAT-agent/main/demo_chat_turn.jpg">
    <img src="https://raw.githubusercontent.com/chenyhvvvv/STAT-agent/main/demo_chat_turn.jpg" alt="STAT one chat-turn demo" width="500" />
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
| Google Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `google/gemini-3.1-pro-preview`, `google/gemini-2.5-pro` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-flash` |
| Poe (gateway) | `POE_API_KEY` | `poe/claude-sonnet-4.5`, `poe/gpt-5.5` |
| xAI | `XAI_API_KEY` | `xai/grok-4` |
| Moonshot | `MOONSHOT_API_KEY` | `moonshot/kimi-k2` |
| Zhipu | `ZHIPU_API_KEY` | `zhipu/glm-4.5` |
| DashScope | `DASHSCOPE_API_KEY` | `dashscope/qwen3-max` |
| Any OpenAI-compatible endpoint | `STAT_API_KEY` | `custom/<model>`, with the endpoint in the web interface's *Base URL* field |

The staged pipeline uses prompted JSON for structured outputs, so the prompts are identical across
backbones and no tool calling is required. The benchmark runs replayed at statagent.online used
Claude Sonnet 4.6 and DeepSeek V4.1 Flash at maximum reasoning effort.

Other settings: `STAT_MODE` (`interactive` or `auto`), `STAT_HOME` (default `~/.stat_agent`, holds
user skills and the RCTD reference cache), `STAT_LOGS_DIR`, `STAT_EXEC_TIMEOUT`, `STAT_SKILL_PATHS`.

## Benchmarks and reproducibility

[statagent.online](https://statagent.online/) has the records of all STAT runs on the single-turn and
multi-turn queries (the single-query benchmark and SessionBench), each replayable step by step in STAT's
interface, and a download link for the results of the other experiments. The colorectal cancer (CRC)
demo records are in [STAT-PaperRepro](https://github.com/chenyhvvvv/STAT-PaperRepro).

## License

[BSD-3-Clause](LICENSE) © STAT contributors.
