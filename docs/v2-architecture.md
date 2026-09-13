# STAT 2.0 architecture

STAT 2.0 is a ground-up rewrite of the agent on the LangGraph runtime. The concepts of STAT 0.1
(a planned, filtered, verified, generated and executed analysis with three kinds of clarification)
are unchanged; every component behind them is new.

```
 browser <-- SSE (one ordered event log) --+   +-- POST /api/turns, /reply, /cancel, ROI/colour edits
                                           v   v
 +--------------------- stat_agent.web (Starlette + uvicorn + sse-starlette) --------------------+
 | AppState . EventLog(seq ring buffer + JSONL) . TurnManager . SessionMutator (lock + ledger)   |
 +--------------------------------------------^--------------------------------------------------+
                                              | Agent.run_turn / resume / cancel
 +--------------------------------------------+--------------------------------------------------+
 | stat_agent.agent.Agent (facade)                                                               |
 |   staged driver = LangGraph StateGraph (plan .. finish_turn), interrupts only in ask_* nodes  |
 |   loop driver   = LangChain create_agent / deepagents create_deep_agent + STAT guards         |
 |   shared: Context (session, registry, runtime, ledger, roles, policy, ablation, emit)         |
 +-------+--------+---------+----------+--------+---------+---------+-----------+----------------+
 | roles | skills | ledger  | runtime  | llm    | memory  | core/io | events    | logging        |
 +-------+--------+---------+----------+--------+---------+---------+-----------+----------------+
   stat_bench: runner, suites, SessionBench, SSE detectors, corruption, replay report, judge
```

Principle: **LangGraph owns control flow, state persistence, interrupts and streaming; STAT owns
every decision.** Nodes are thin wrappers that assemble typed inputs, call plain-Python roles and
guards, write typed state and emit typed events. Roles, guards, ledger and runtime never import
the framework, so they are unit-tested without a graph and the same guards mount on the loop agent
as middleware.

## Package map (`src/stat_agent`)

| Package | Contents |
| --- | --- |
| `core/` | `SimpleSession`, `DataSlice`, `ROIManager`, `SessionSummary` (the single typed view of the data), session lock |
| `io/` | dataset loading (`load_dataset`), durable sessions (`save_session`, `load_session`) |
| `skills/` | frontmatter schema v2, layered registry (built-in + `~/.stat_agent/skills`), deterministic filter, linter, author, CLI; `skills/<slug>/SKILL.md` |
| `prompts/` | versioned Jinja templates (`planner.v1.md`, ...) with strict undefined variables |
| `roles/` | planner, matcher, verifier, generator, reflector, analyzer, method card, summarizer: typed input and output, one prompt each |
| `llm/` | `make_model("provider/model")` on LangChain chat models, prompted-JSON structured output with one repair, fixtures, call log, usage meter |
| `runtime/` | guarded in-process executor: AST safety check, routed stdout, timeout, snapshots, figure capture, code access tracing |
| `ledger/` | provenance ledger (artefacts, edges, staleness) and the `SessionMutator` used by GUI edits |
| `memory/` | conversation memory (turn records, step handoffs, facts, rolling summary) and per-role context assembly |
| `agent/` | `TurnState`, `Context`, graph builder, nodes, clarification policies, ablation config, `Agent` facade, loop mode |
| `extras/` | optional deepagents driver with subagents and web research tools |
| `events/` | the typed event catalogue, encoding, emitters and sinks |
| `logging/` | notebook export and Markdown/HTML reports |
| `web/` | Starlette app, event log, turn manager, routes, templates and static assets |
| `cli/` | `stat-agent` command line |

## The staged graph

Nodes: `plan`, `ask_planner`, `select_step`, `filter`, `match`, `ask_skill`, `synthesize_card`,
`verify`, `ask_verifier`, `generate`, `execute` (runs the code, validates the declared outputs and
holds the bounded reflect-and-retry loop), `analyze`, `block_step`, `finish_step`, `finish_turn`.
Routing is decided inside the nodes (`Command(goto=...)`), which keeps every decision in plain
Python and testable. Logical flow:

```
START -> plan --needs clarification--> ask_planner --resume--> plan
           +-> select_step --no steps left--> finish_turn -> END
                 +-> filter -> match --more than one--> ask_skill --resume--> verify
                                   +--none--> synthesize_card --incompatible or unmet--> block_step
                                   +--one---> verify --ask--> ask_verifier --resume--> verify
                                                +--advice--> block_step
                                                +--met-----> generate --no code--> analyze
                                                               +-> execute --ok--> analyze
                                                                     +--error--> (reflect, retry <= 2 inside execute)
                                                                                   +--give up--> analyze (issue) / block_step
               analyze -> finish_step --more steps--> select_step
               block_step -> finish_step --a remaining step depends on it--> finish_turn (blocked)
```

Every interrupt lives in an `ask_*` node that first asks the clarification policy for a decision
and only then calls `interrupt()`, so re-execution on resume is harmless. Execution never happens
inside an interrupt node. A checkpointer (in-memory, or SQLite under the session directory for
durable sessions) persists the turn so a browser reload or a process restart resumes at the
pending prompt.

### Deterministic versus LLM-decided

| Decision | Who decides |
| --- | --- |
| Target slice ids are valid | deterministic post-check of the planner output |
| A skill applies to the target slices | deterministic filter over the frontmatter contract |
| Which of the compatible skills matches the request | matcher (LLM), restricted to the compatible catalogue |
| A declared input is stale or absent | deterministic pre-pass before the verifier LLM call |
| Remaining prerequisites | verifier (LLM) |
| Generated code is safe to run | AST safety check |
| Retry after an error | reflector (LLM) gated by attempt budget, change magnitude and confidence |
| What changed in the session | snapshot diff, never the model's claim |

### Out-of-registry requests

When the matcher returns nothing and the step names a method (the planner's `kind: skill`; custom workflows are `general` and skip all of this), the method-card synthesizer produces a provisional skill contract
for the requested method. The card goes through the same filter and verifier: an incompatible card
or an unobtainable prerequisite becomes a `refusal` event that names the requirement (for example
spliced and unspliced layers for RNA velocity) before any code is generated. A card that passes
runs in unguarded mode, the declared outputs are validated after execution, and the user is offered
a `SKILL.md` draft that can be installed into the user skills directory.

The matcher returns no match for a request that names a method no skill implements, so the
method-card route handles it; the generator is asked to state a substitution in its first sentence.
A request that names a skill which is installed but switched off gets advice to enable it instead of
a provisional re-implementation, and the method card names a method only when the user's own words
name one.

## Clarification policies and ablation arms

`InteractivePolicy` pauses the turn. `AutoPolicy` answers what can be defaulted (skill defaults,
known facts, session state) and refuses what cannot (a file path is never invented); every answer
is recorded as an assumption event that the interface can override. `ScriptedPolicy` answers from
regex rules for benchmarks. `AblationConfig` turns each component off (`A1` .. `A9`) and selects the
driver (`staged` or `loop`).

## Loop mode

`agent/loop.py` exposes the same operations as tools (`inspect_session`, `list_skills`,
`read_skill`, `check_skill`, `run_code`, `ask_user`, `finish`) to a single tool-calling agent built
with LangChain `create_agent`. `StatGuards` middleware refuses skill code that was not checked and
code that reads stale artefacts. `extras/deepagents_driver.py` builds the same agent with
`deepagents` (todo planning, `method-researcher` and `skill-author` subagents, an empty in-state
filesystem so skill bodies are only disclosed through the registry). Select it with
`STAT_DRIVER=loop` and `Settings.loop_backend="deepagents"`.

## Provenance and memory

Artefacts are identified as `s<slice>:<kind>:<key>` (for example `s0:obs:celltype`), `roi:<name>`
and `fig:<id>`. Each executed step records the snapshot diff, the declared and traced inputs, and a
producer reference. Deleting or changing an input marks every dependant stale; re-producing an
artefact clears it. GUI edits go through `SessionMutator` and record the same way, so the canvas
and the agent are two views of one state. The event `provenance_update` carries both the v1
`state_changes` keys and the stale set.

Memory keeps a `TurnRecord` per turn with one `StepHandoff` per step (inputs, outputs, summary,
findings, code), a facts map, a verbatim window of recent turns and a rolling summary regenerated
when the window overflows. Each role receives only the context it needs. Within a turn, the
verifier, generator and analyser of a later step also receive the completed steps so far (summary,
data changes, findings, artefacts, the namespace variables each step added and, within a budget,
its code), and the generator sees a listing of the live Python namespace. That namespace is one
dictionary per session in `Runtime`: variables persist across code blocks, steps and turns, but not
across a session save and reload, so the generator is told to write final results into the AnnData.
Across turns the history carries, per turn, one line per step (skill, artefacts, variables, what
stayed open), the known facts and the clarification answers of the last two turns, which the verifier
sees as answers already given. The planner receives the same artefact block as the generator, with
its STALE marks, so a direct answer about earlier results knows what a later change invalidated.
Artefacts written by a run that failed or was stopped are recorded as stale, never fresh, and a stop
during a step closes the step through `finish_step` so its handoff and plan status survive. Only the
snapshot diff decides what changed: a key the code assigned without changing the slice's values (a
write to a copy or a subset, a library call with a custom key) is never reported as an update and
never marks dependants stale; the one thing it does is clear a stale flag on that key, so a producer
re-run that reproduces identical values counts as re-produced. The analyser is told which artefacts
of its slices are stale and which loaded artefacts the step overwrote, and a `warning` event reports
the overwrite. The report export embeds the figures of every step. The
session summary also lists h5ad files that sit next to the slices without being slices (a reference
dataset), so a prerequisite can be satisfied from them.
