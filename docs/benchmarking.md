# Benchmarking with `stat-bench`

`stat-bench` runs suites of queries through STAT (or other systems), writes one `result.json`
per (query, seed), scores them and aggregates with bootstrap confidence intervals.

```bash
# convert the STAT 0.1 query set into a suite file
stat-bench convert paper/benchmarking/comparison/queries.json --out suites/stat40.yaml

# run the full pipeline (A0) with scripted clarification answers, five seeds
export STAT_BENCH_DATA=/path/to/benchmark/datasets
stat-bench run --suite suites/stat40.yaml --system stat --driver staged --ablation A0 \
  --policy scripted --seeds 5 --backbone poe/claude-sonnet-4.5 --out results/A0

# the single-agent arm (loop driver, guards off)
stat-bench run --suite suites/stat40.yaml --driver loop --ablation A6 --out results/A6

# auto mode instead of scripted answers
stat-bench run --suite suites/stat40.yaml --policy auto --out results/auto

# aggregate
stat-bench summarize results/A0 results/A6 --out results/summary --by system,driver,ablation
```

Completed cells are skipped on re-run (`--retry-failed` re-runs failures). Every `result.json`
records the software versions, driver, ablation, policy, backbone, seed, usage per call, the
clarifications and assumptions, refusals, the method card, the decisions of each role, the
artefacts produced, the provenance hash and the silent-error detector output (`sse1` .. `sse5`).

## Ablation arms

| Arm | Meaning |
| --- | --- |
| A0 | full pipeline |
| A1 .. A5 | planner, filter, matcher, verifier, reflector off |
| A6 | loop driver with guards off (single agent, same tools) |
| A7 | templated analyzer |
| A8 | curation only (no method card) |
| A9 | no method card, no unguarded execution |

`--ablation` also accepts `key=off,...` for custom combinations.

## SessionBench

Session scripts (`src/stat_bench/suites/sessionbench/*.yaml`) interleave user turns, GUI actions
(draw, delete or rename an ROI, edit an annotation) and programmatic verifiers over the live
session, ledger, memory and event stream. Run them from Python with an agent factory:

```python
from stat_bench.session_bench import SessionScript, run_script
result = await run_script(SessionScript.load("CR-1_cross_turn_reference.yaml"), agent_factory, mode="scripted")
```

## Frontmatter corruption

```bash
stat-bench corrupt --skills src/stat_agent/skills --out /tmp/corrupt --severity 2 --n 6
STAT_SKILL_PATHS=/tmp/corrupt stat-bench run ...
```

`manifest.json` lists each corrupted field with its before and after values so containment
(refusal or clarification instead of a wrong execution) can be scored per severity.

## Replay against STAT 0.1 logs

```bash
stat-bench replay --logs paper/benchmarking/logs/* --out results/replay
```

Parses the old `turn_*.md` logs into per-turn decisions (plan, matcher, verifier, clarifications)
and compares them with new runs; divergences listed in an `expected_divergences.json` are
explained, the rest fail the informational gate.

## Judge

`stat-bench judge --results results --judge-model anthropic/claude-opus-4-7 --out verdicts.json`
ports the STAT 0.1 protocol: one prompt per query with all systems anonymised and shuffled,
success, quality and rank per system, and adjudication of the detector flags.
