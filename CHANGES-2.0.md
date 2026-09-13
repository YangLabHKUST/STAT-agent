# STAT 2.0 change record

*What changed, what was added, how each part was verified, and what is still untested, from STAT 0.1.27 to 2.0.*

Branch `v2`, commits `924327c` (rewrite) and `4aea6cb` (notebook links). Working tree also holds one uncommitted crash fix (§10, B1). Written 2026-09-12 from the code, the git history and the logs of your local session in `~/Downloads/A0`.

## 1. Read this first

STAT 2.0 keeps the concept, the workflow and the viewer of 0.1 and replaces every component behind them. The staged pipeline (plan, filter, match, verify, generate, execute, reflect, analyse) is now a LangGraph graph whose decisions are plain Python; every artefact the agent or you produce is tracked in a provenance ledger; clarifications can be answered by policy; requests for methods without a skill go through a synthesised contract; the web layer has one ordered, resumable event stream; and a benchmark harness turns every reviewer experiment into a command-line switch.

**How much of this has been exercised, and how.** Every claim in this document carries one of four marks:

<span class="chip t">T</span> covered by the automated suite (111 tests, scripted LLM, no network) · <span class="chip s">S</span> smoke-tested out of process (fake LLM server, curl) · <span class="chip l">L</span> exercised live once with opus 4.8 through Poe on your `bc_demo` dataset · <span class="chip u">U</span> not exercised at all.

**Update after your first test session (same day):** the interface is now exercised in headless Chrome by `scripts/ui_probe.py` (load a dataset through the form, send messages, answer a prerequisite card, reload, resize), and ten real turns were driven through the API with opus 4.8 on your dataset by `scripts/api_probe.py`. Both found bugs; all of them are fixed in the working tree (§10). Until then the interface had only been checked by a syntax check and API tests, which is why your first session hit so many problems.

**Bugs found and fixed so far** (details in §10): B1 crash in the structured-output repair path; B2 method cards with positional lists; B3 no general-analysis path (simple questions went through method cards and could be refused on a guessed key); B4 the planner answering inside its clarification field; B5 a race that let a client save or reply before the agent had absorbed the turn; B6 the dataset-load form never opened the event stream, so nothing rendered in the chat until a page reload; B7 every reload rendered the history twice; B8 raw generator code streamed into the chat bubble; B9 the chat list could not be scrolled to its last message; B10 an inspection-only skill step was accepted as done and credited with outputs it never wrote; B11 loading a saved session polluted the saved directory; B12 the skills panel showed empty rows; B13 the reconnect banner pushed the chat's last lines off-screen; B14 the method-card prompt named the RNA-velocity test case as its example (a prompt leak); B15 the refusal for a missing count layer told you to run cell type annotation; B16 imageless slices were served as multi-megabyte blank JPEGs that the viewer discarded (6 to 32 s per slice on crc_demo); B17 Stop was only honoured before a code block, so it let up to three more model calls run, analysed the stopped run as a failure, labelled it a timeout and ended the turn as done; B18 the ROI list returned a 500 while the agent held the session; B19 the postcondition system failed a successful four-minute RCTD run on an ignored warn-level check and re-ran everything; it is now one rule with a continuation instead of a re-run; B20 the method card was synthesised for generic workflows and could refuse them on a guessed count layer, so it is now reserved for named methods and given to the generator as hints; B21 eight skill contracts declared outputs their code never writes on the slice; B22 the continuation fired for results on derived objects and showed the analyser mismatched code and output; B23 a second `__INTERPRET__` marker leaked into the chat. An independent read-only review (Opus 5, 23 live turns, 50 findings, published separately) followed; its no-decision items are fixed in §10b below. All are in the working tree, uncommitted.

**Live results after the fixes** (`scripts/api_probe.py`, opus 4.8 via Poe, bc_demo, interactive mode with scripted answers): ten of ten turns ended `done` in 461 s and 209k tokens (caveat found later by the independent review, F1: the probe's clarification answers were silently dropped by the reply route, so the prompted turns ran on empty answers; the unprompted turns stand): a capability question (direct answer, one call), "Describe the data" (general path), niche detection (Harmonics on 160,286 cells, 106 s, 3 figures, `obs:niche_label` recorded), four follow-ups on the niches (composition, spatial plot, marker genes with one retry triggered by the output check, enrichment table), an RNA-velocity request (declined naming spliced/unspliced counts), a label rename recorded in provenance, and a memory question answered with the facts of the session.

**Server status.** Servers are restarted from zero on every restart (new session directory, no saved session loaded) unless you ask to preserve history. Since 17:20 on 2026-09-12 two run side by side: **A** on http://127.0.0.1:8889 with `~/Downloads/bc_demo` (three breast cancer slices: 0 and 1 cell-level, 2 spot-level, H&E images), logs in `~/Downloads/A0/bc_demo/stat-web.log` and sessions in `~/Downloads/A0/bc_demo/session_<ts>/`; **B** on http://127.0.0.1:8891 with `~/Downloads/crc_demo/processed` (three spot-level Visium HD colorectal slices, no images), logs in `~/Downloads/A0/crc_processed/`. Earlier today a single server ran `~/Downloads/crc_demo` (top level: three cell-level slices, 311k / 431k / 391k bins, loaded in 11 s). To reproduce one server:

```bash
cd ~/Code/Claude-project/STAT-agent
POE_API_KEY=... STAT_MODEL=poe/claude-opus-4.8 STAT_LOGS_DIR=$HOME/Downloads/A0 STAT_JUPYTER_ROOT=$HOME/Downloads/A0 \
  ~/anaconda3/envs/stat2/bin/stat-web --host 127.0.0.1 --port 8889
curl -X POST http://127.0.0.1:8889/api/session -H 'Content-Type: application/json' \
  -d '{"dataset_dir": "'"$HOME"'/Downloads/crc_demo", "session_name": "crc_demo", "provider": "poe", "model": "poe/claude-opus-4.8", "mode": "interactive"}'
```

Note: the loader reads only the top level of the dataset directory, so the three `adata_P*.h5ad` files are loaded and detected as cell-level (they carry no spot metadata). The prepared spot-level versions with `spot_diameter`/`spot_shape` live in `crc_demo/processed/`; load that directory instead if you want the spot-level skills (deconvolution, SpaGCN domains) offered.

## 2. What did not change

- **Concept and workflow.** Planner, deterministic filter, semantic matcher, verifier, generator with fenced code, bounded reflection (two retries, small fixes, confidence at least 0.7), analyser, three clarification kinds (planner, verifier, skill selection), original query preserved across clarifications.
- **Data model.** `SimpleSession` of `DataSlice`s, each with its own `adata` and images; explicit slice ids; cell (x, y) equals pixel (x, y); ROI manager with bbox, circle, polygon; no spatialdata. The data layer was ported and cleaned, not redesigned (`core/`, `io/`).
- **Skills.** The same 26 `SKILL.md` bodies (RCTD's 264 MB reference cache moved with them). Frontmatter v1 keys still load unchanged; v2 keys are additions.
- **Prompts.** The ten role prompts start from the 0.1 text extracted verbatim into files (`prompts/*.v1.md`); two gained optional blocks (generator: session state; reflector: stdout tail).
- **Viewer.** Three-column layout, canvas engine, ROI tools, legend, gene and proportion modes, slice tabs, logs modal, skill panel, resizable panels: the 0.1 JavaScript for these is untouched. New elements are additive.
- **Log formats.** `turn_NNN.md` per turn, the Jupyter notebook per session, per-call JSON logs.
- **Endpoints.** Every 0.1 endpoint is served as an alias of the new routes (§B).

## 3. By the numbers

| | STAT 0.1.27 | STAT 2.0 |
|---|---|---|
| Agent + data + web backend (Python, no skill bodies) | 17,800 lines in 27 files, one 4,375-line core module | 11,200 lines across 15 packages, largest module 877 lines |
| Prompts | f-strings inside methods, one 284-line system prompt on every call | 10 versioned template files, one prompt per role |
| Frontend | 3,715 + 313 + 1,921 lines | 3,651 + 329 + 1,953 lines; 1,350 lines of `app.js` changed |
| Automated tests | 0 | 111 (96 functions, 18 files), about 14 s, no network |
| Skills | 26 (paper says 27) | 26, all annotated with v2 contracts, 0 lint warnings |
| Event types | 31 ad-hoc dict shapes | 40 typed events with a shared header |
| HTTP routes | 10 documented (Flask) | 55 (Starlette), 20 of them legacy aliases |
| LLM providers | 5 | 9 plus any OpenAI-compatible base URL |
| Benchmark tooling | scripts in the paper repository | `stat-bench` package (1,700 lines) |
| Python | 3.9+ | 3.11+ |

## 4. Module map, 0.1 to 2.0

| 0.1 file (lines) | 2.0 replacement | Note |
|---|---|---|
| `agent/spatial_agent_core.py` (4,375) | `agent/agent.py`, `agent/nodes/execution.py`, `agent/nodes/finishing.py`, `runtime/`, `ledger/` | The core module is gone. Its five duplicated filter-match-verify chains became one graph. |
| `agent/conversation_orchestrator.py` (794), `agent/clarification_context.py` (445) | `agent/graph.py`, `agent/nodes/planning.py`, `agent/nodes/selection.py`, `agent/policy.py`, `agent/state.py` | Clarification state lives in the checkpointed `TurnState`; interrupts replace the three hand-written response handlers. |
| `agent/pipeline_executor.py` (730) | `agent/nodes/selection.py`, `skills/filter.py`, `roles/matcher.py`, `roles/verifier.py` | |
| `agent/query_planner.py` (396) | `roles/planner.py`, `prompts/planner.v1.md` | Adds slice-id validation after the model call. |
| `agent/skill_filter.py` (235) | `skills/filter.py` | Same grammar plus `min_slices`; returns compatible, rejected with reasons, and `would_reject` when bypassed. |
| `agent/skill_verifier.py` (420) | `roles/verifier.py`, `prompts/verifier.v1.md` | Deterministic contract pre-pass before the model call. |
| `agent/skill_registry.py` (747) | `skills/registry.py`, `skills/schema.py`, `skills/loader.py`, `skills/lint.py`, `skills/cli.py`, `skills/author.py` | Layered (built-in + `~/.stat_agent/skills`), schema-validated, broken skills reported instead of crashing. |
| `agent/error_reflection.py` (284) | `roles/reflector.py` | Same gates; unfixable error types skip the model. |
| `agent/executor.py` (277) | `runtime/executor.py`, `runtime/safety.py`, `runtime/snapshot.py`, `runtime/artifacts.py`, `runtime/plots.py` | See §5.6. |
| `agent/memory.py` (1,063) | `memory/state.py`, `memory/context.py`, `roles/summarizer.py` | See §5.8. |
| `agent/llm_backend.py` (1,781), `agent/model_config.py` (328) | `llm/models.py`, `llm/client.py`, `llm/structured.py`, `llm/usage.py`, `llm/calllog.py`, `llm/fake.py` | LangChain chat models; provider quirks reduced to a small table (§5.9). |
| `agent/prompt_logger.py` (307) | `llm/calllog.py` | Write-through per call instead of buffered per turn. |
| `agent/notebook_logger.py` (289) | `logging/notebook.py` | Ported. |
| `agent/tools.py` (484) | `core/view.py` (`SessionSummary`) and the loop-mode tools | The introspection helpers became the typed session view. |
| `core/session.py` (885), `core/data_slice.py` (344), `core/roi_manager.py` (447) | `core/session.py`, `core/data_slice.py`, `core/roi_manager.py`, `core/colors.py`, `core/lock.py`, `core/view.py` | Ported; ROIs have one store; a session lock; a typed summary. |
| `functions/io.py` (739) | `io/dataset.py`, `io/session_store.py` | Loading ported; save/load of whole sessions is new. |
| `web/app.py` (about 2,400) | `web/server.py`, `web/state.py`, `web/events.py`, `web/turns.py`, `web/routes/*.py`, `web/schemas.py`, `web/errors.py`, `web/render.py` | Flask replaced by Starlette + uvicorn + sse-starlette. |
| `web/static/js/app.js` (3,715) | same file, 1,350 lines rewritten | See §5.12. |
| (none) | `events/`, `ledger/`, `extras/`, `stat_bench/`, `tests/` | New. |

## 5. Detailed changes by area

Each item reads: what 0.1 did, what 2.0 does, how it was verified.

### 5.1 Orchestration

- **0.1:** `ConversationOrchestrator` and `SpatialAgent` coordinated a hand-written state machine; clarification routing lived in three handlers keyed on a pending-type string; multi-step plans had no handoff between steps and no failure branch; 40% of the core module was unreachable.
- **2.0:** one LangGraph `StateGraph` with nodes `plan, ask_planner, select_step, filter, match, ask_skill, synthesize_card, verify, ask_verifier, generate, execute, analyze, block_step, finish_step, finish_turn`. Routing is decided inside nodes (`Command(goto=...)`). Interrupts happen only in the three `ask_*` nodes, which first ask the clarification policy and only then pause, so re-execution on resume is harmless. The turn state is checkpointed (in memory, or SQLite under the session directory), so a browser reload or a process restart resumes at the pending question. A failed step blocks the steps that depend on it and the turn ends `blocked` instead of pretending success. Cancel is cooperative: a running code block finishes, its diff is recorded, the turn ends `cancelled`.
- **Verified:** <span class="chip t">T</span> `tests/graph/test_staged.py` (11: single step, planner and verifier clarifications, skill selection, multi-step with dependency, failure containment, reflection retry and give-up, memory off, ablations), `tests/graph/test_capabilities.py` (durable resume after reload, rerun from stale artefacts, draft from an unguarded step); <span class="chip l">L</span> one full staged turn with opus 4.8.

### 5.2 Roles and prompts

- **0.1:** every role call prepended the same 284-line system prompt; JSON was scraped with regexes and silently replaced by fabricated defaults on failure (planner "please clarify", verifier "not met", analyser "executed successfully"); the verifier read a `columns` key the session never produced.
- **2.0:** eight roles (`planner, matcher, verifier, generator, reflector, analyzer, method_card, summarizer`) with typed input and output models and one prompt file each. Structured outputs are prompted JSON validated by pydantic, with exactly one repair call; a second failure raises `StructuredOutputError` and the turn ends with `turn_error` naming the stage. Nothing is fabricated. Prompts are Jinja templates with strict undefined variables and a content hash, so a template typo fails at load, not mid-turn.
- **Verified:** <span class="chip t">T</span> `tests/unit/test_prompts_and_events.py`, `tests/unit/test_llm_client.py` (prompted JSON, repair, fixture replay), graph tests; <span class="chip l">L</span> planner, matcher, verifier, generator and analyser each produced valid output on the first try in your DE turn (5 calls, 17,810 tokens, 39 s).
- **Consequence you will notice:** where 0.1 continued with a fabricated default, 2.0 stops and reports. Some "new bugs" are old silent failures made visible.

### 5.3 Skill system

- **0.1:** frontmatter with `name, title, slug, description, filter_requirements, prerequisites, default_skill`; `num_slices` exact only; 13 skills declared no slice count; nothing declared outputs or dependencies; a malformed file crashed the registry.
- **2.0:** frontmatter v2 adds `schema_version, version, consumes, produces, platforms, dependencies{python, vendored, gpu, subprocess_workers, notes}, defaults`; `min_slices` joins the filter grammar. A linter parses every python block in the body and checks declared outputs against AST writes and known library effects, declared inputs against reads, and imports against dependencies. All 26 skills were annotated (their bodies untouched); `stat-agent skill validate --strict` reports 0 errors, 0 warnings. The registry is layered: built-in plus `~/.stat_agent/skills`, later layers win; broken files are listed, not fatal. Skill bodies are still disclosed only when selected. CLI: `skill validate | list | lint-file | export | new`.
- **Verified:** <span class="chip t">T</span> `tests/contract/test_skill_contracts.py` (every SKILL.md: schema, grammar, produces versus writes), `tests/unit/test_skill_schema_and_filter.py`; <span class="chip u">U</span> no skill body has been executed under 2.0 (see §9).

### 5.4 Verification and clarification

- **0.1:** verifier answers were cleared between steps; several questions were collapsed into one answer; skill choice was matched by substring; every prerequisite was an LLM judgement.
- **2.0:** the verifier first runs a deterministic contract pre-pass: a declared input that is stale (an upstream ROI or annotation changed) or absent on the target slices is declined with a message that names the artefact, before any model call. Then the LLM verifier decides `satisfied | ask_user | not_obtainable` per prerequisite and rewrites the request into a complete query. Answers are typed (`Decision{answer | option_index | answers{question: answer}}`) and carry the prompt id; a mismatched reply is a 409. Three policies resolve prompts: `Interactive` pauses; `Auto` answers from skill defaults, known facts and session state, records each answer as an overridable assumption event, and refuses what it cannot default (a file path is never invented); `Scripted` answers from regex rules for benchmarks.
- **Verified:** <span class="chip t">T</span> graph tests for each kind and each policy, `tests/graph/test_loop.py` for the same in loop mode; <span class="chip l">L</span> the verifier ran live once (contract pre-pass then LLM, "satisfied"); <span class="chip u">U</span> auto-mode assumptions and the "Answer instead" override have not been exercised live in the browser.
- **Watch for:** the pre-pass is strict by design for registered skills; for synthesised method cards it can refuse on a guessed key (§10, B3).

### 5.5 Out-of-registry requests

- **0.1:** when no skill matched, the generator ran in "general analysis mode" without guidance; unknown methods were executed on whatever the model assumed.
- **2.0:** when the matcher returns nothing, the method-card synthesiser writes a provisional skill contract for the named method (requirements, inputs, outputs, outline, risks, confidence). The card passes through the same filter and verifier, so an incompatible card or an unobtainable input becomes a `refusal` event that names the requirement before any code is generated. A card that passes runs in unguarded mode, its declared outputs are validated after execution, and a `draft_offer` lets you turn the executed code into a `SKILL.md` in the user skills directory. Optional: a `method-researcher` subagent with web lookup (deepagents driver, `research` extra).
- **Verified:** <span class="chip t">T</span> RNA-velocity request refused naming spliced/unspliced layers; card path to unguarded execution; draft written and loadable; <span class="chip l">L</span> "Describe the data" ran through this path successfully once and crashed once (§10).
- **Design gap found in your session:** the light general-analysis mode of 0.1 has no equivalent yet, so trivial data questions take the heavy path (§10, B3).

### 5.6 Execution runtime

- **0.1:** bare `exec` in a daemon thread that swapped the global `sys.stdout`; no timeout; the safety check existed but was never called; plots captured in the exec thread; the paper called it an isolated kernel.
- **2.0:** an AST safety check before every run (blocked imports and calls, writes outside the session directory); output routed per thread so concurrent GUI work does not interleave; a wall-clock timeout with an asynchronous interrupt and an explicit `orphaned` state when a C call will not return; snapshots before and after with a diff that is the ground truth of what changed; code-access tracing (which `obs`/`obsm`/`uns` keys and ROIs the code read or wrote); figures captured with the Agg backend; vendored skill packages put on `sys.path` for the run, and on `PYTHONPATH` for skills that spawn workers (RCTD). The namespace holds `session`, `np`, `pd`; code imports what it needs, as the 0.1 prompt already said.
- **Verified:** <span class="chip t">T</span> `tests/executor/test_runtime.py` (12: timeout, ordered output, plots, namespace persistence, safety table, access tracing, snapshots); <span class="chip l">L</span> one live block (1.1 s).

### 5.7 Provenance ledger (replaces state-change detection)

- **0.1:** after a run, five flags were derived by comparing before and after (`celltypes_updated, deconv_weights_updated, celltype_colors_updated, rois_added, rois_deleted`); nothing remembered what depended on what; GUI edits bypassed the agent.
- **2.0:** every artefact has an id (`s0:obs:celltype`, `s1:obsm:deconv_weights`, `roi:tumor`, `fig:...`), a fingerprint, a version and a producer (turn, step, skill, code hash). Each executed step records the snapshot diff plus declared and traced inputs. Deleting or changing an input marks every dependant `stale` with a reason; re-producing an artefact clears it. GUI actions (ROI create, delete, rename; colour edits; cell-type renames; column edits) go through `SessionMutator` and are recorded the same way, so the canvas and the agent are two views of one state. Every change is a `provenance_update` event carrying the five legacy flags plus the stale set; the verifier declines to read stale inputs; "Re-run downstream" builds a synthetic turn over the producers of stale artefacts. The ledger is saved as `ledger.json` per session.
- **Verified:** <span class="chip t">T</span> `tests/property/test_ledger.py` (hypothesis: append-only, staleness monotonic and propagated, re-production clears), ROI CRUD in the web tests, rerun from stale artefacts, SessionBench cross-turn script (ROI drawn, analysis restricted to it, ROI deleted, result marked stale); <span class="chip l">L</span> your DE turn recorded `s0:uns:de_results` and `s0:uns:rank_genes_groups` with the step as producer; <span class="chip u">U</span> stale badges and the Changes drawer in the browser.

### 5.8 Memory

- **0.1:** history was one text blob passed to four prompts; the last 80 messages verbatim; messages over 200 characters were summarised by the model; the summary was frozen once made; findings and modification history were written but never read; user messages were skipped in some prompts.
- **2.0:** one `TurnRecord` per turn with a `StepHandoff` per step (inputs, outputs, summary, findings, parameters, code, complete query, response), a facts map extracted by the planner, a verbatim window of the last four turns and a rolling summary regenerated when the window overflows. Each role receives only the blocks it needs, assembled in `memory/context.py`, in the same formatting the 0.1 prompts expected. Within a turn, later steps also receive the completed steps' handoffs and a listing of the live Python namespace (B24). Memory is saved as `memory.json` and restored with the session; namespace variables are not.
- **Verified:** <span class="chip t">T</span> multi-turn graph tests, memory-off ablation, SessionBench correction script; <span class="chip u">U</span> the summariser has only run with the scripted LLM; long sessions (more than four turns) have not been tried live.
- **Consequence you will notice:** older turns are no longer verbatim; they are summarised plus their handoffs and facts. If a later turn seems to have forgotten a detail, the place to look is `memory.json` and the planner block of `calls/tNNNN_01_planner.json`.

### 5.9 LLM layer and providers

- **0.1:** a 1,781-line backend with hand-written HTTP clients per provider, a retry decorator, a model capability table, a Responses-API path for GPT-5 and special cases for models that refuse a temperature.
- **2.0:** LangChain chat models (`ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`, and `ChatOpenAI` with a base URL for Poe, DeepSeek, xAI, Moonshot, Zhipu, DashScope and any custom endpoint). Models are addressed as `provider/model`; a bare id infers the provider. Reasoning models get the Responses API and no temperature via a short prefix table (`gpt-5, o1, o3, o4, claude-opus-4-7`). Retries are the SDK's (`max_retries=3`). Usage is metered per call, per role and per turn; every call is logged as JSON at call time, and the legacy `turn_NNN.md` is written from the same records. Tests use a scripted responder or recorded fixtures.
- **Verified:** <span class="chip t">T</span> `tests/unit/test_llm_client.py`; <span class="chip l">L</span> Poe with `claude-opus-4.8` (text and prompted JSON); <span class="chip u">U</span> every other provider (a live smoke test exists in `tests/live/test_live_smoke.py`, opt-in with `STAT_LIVE=1`). The 0.1 special cases for Poe bots that force extended thinking and for GPT-5 through Poe are not carried over and may need to return.

### 5.10 Loop mode and deepagents

- **0.1:** none.
- **2.0:** the same operations exposed as tools (`inspect_session, list_skills, read_skill, check_skill, run_code, ask_user, finish`) to a single tool-calling agent built with LangChain `create_agent`. `StatGuards` middleware refuses skill code that was not checked this turn and code that reads stale artefacts. The optional deepagents driver (`pip install "stat-agent[loop]"`, `Settings.loop_backend="deepagents"`) adds todo planning and two isolated subagents (`method-researcher`, `skill-author`) on an empty in-state filesystem so skill bodies are only disclosed through the registry. Ablation arm A6 is this driver with guards off. Select with `STAT_DRIVER=loop` or the mode selector's driver option.
- **Verified:** <span class="chip t">T</span> `tests/graph/test_loop.py` (6, with a scripted tool-calling model: guarded run, refusal before `check_skill`, A6 without guards, auto assumption, interactive pause and resume, deepagents driver); <span class="chip u">U</span> never run with a real model. Expect prompt and tool-description tuning to be needed.

### 5.11 Web backend and protocol

- **0.1:** Flask with module globals; a thread plus queue bridged the agent into a per-request SSE response, which dropped reflection events, synthesised `done`, and died on encoding errors; no reconnection; `current slice` state on the server.
- **2.0:** Starlette app with an `AppState`, an `EventLog` per session (global sequence, 5,000-frame or 16 MB ring buffer, JSONL on disk), a `TurnManager` (`queued, running, awaiting_input, done, error, cancelled`, one non-terminal turn per session) and a `SessionMutator` for GUI edits under the session lock (2 s timeout, then 409 `locked`). The browser holds one `EventSource` on `GET /api/events`; `id:` is the sequence, `Last-Event-ID` replays missed frames, `resync` tells the client to reload the snapshot when the buffer was evicted, `ping` every 15 s. Turns are created with `POST /api/turns` (202), answered with `POST /api/turns/{id}/reply`, cancelled with `POST /api/turns/{id}/cancel`. Data endpoints are per slice and read-only (`/api/slices/{id}/image|cells|genes|expression`); ROI CRUD lives under `/api/rois`; the JSON encoder handles numpy and refuses NaN. Errors are one envelope with codes (`validation, no_session, not_found, turn_active, busy, locked, no_pending_prompt, prompt_mismatch`). Every 0.1 endpoint is an alias.
- **Verified:** <span class="chip t">T</span> `tests/web/test_web_api.py` (9: lifecycle, read-only data endpoints and consistency hash, ROI CRUD with ledger headers, prompt and reply and replay, SSE frames, cancel and reset, legacy aliases and 409s, session save and load, draft endpoint); <span class="chip s">S</span> out-of-process with the fake LLM; <span class="chip l">L</span> session load of 160,286 cells, four live turns; <span class="chip u">U</span> reconnection after a dropped connection, concurrent ROI edits during a turn, cancel during execution, the `once` replay from a real browser.

### 5.12 Frontend

- **0.1:** `app.js` parsed a per-request SSE stream, dropped frames it could not parse, de-duplicated by text prefix, had no reconnection and restored history from a chat log.
- **2.0:** 1,350 of 3,651 lines changed, in three regions: the API client (new methods `getSnapshot, createTurn, replyTurn, cancelTurn, listTurns, getTurnEvents`; ROI creation sends `slice_id`; gene endpoints are per slice), the session entry (`enterApp`, `tryReconnectSession`, `restoreChatHistory` rewritten to use the snapshot and the turn list), and the chat engine (a single `EventSource`, one `handleEvent` reducer used for both live frames and replayed turns, typed replies for the three prompt kinds, cancel, a per-turn status line with phase and elapsed time and tokens, a reconnect banner, staleness information, assumption cards with "Answer instead", a Changes drawer, a mode selector). New elements in `index.html`: `#reconnect-banner, #changes-btn, #changes-drawer, #changes-body, #changes-close, #mode-select, #turn-status, #stale-info`. Untouched: canvas rendering and interaction, ROI drawing, gene search and colour bar, cell-type legend and colours, slice and modality controls, resizable panels, logs modal, skill panel, save-chat, init overlay.
- **Verified:** <span class="chip s">S</span> `scripts/ui_probe.py` drives headless Chrome over the DevTools protocol against the fake-LLM server: dataset load through the form, a turn with a prerequisite card answered in the card, a direct answer, a refused request, a page reload (history restored once, no duplicates), a 1100x700 viewport; it checks console errors and exceptions, that the last message is visible after scrolling, that no raw code is left in bubbles, and saves screenshots; <span class="chip l">L</span> your session; <span class="chip u">U</span> ROI drawing, the Changes drawer, "Answer instead", cancel, reconnect after a dropped connection in a real browser. The planned split into ES modules was deferred; it has no functional effect.

### 5.13 Sessions, logs, notebook, report

- **0.1:** logs per turn buffered in memory and written at the end; the notebook; no way to save or resume a session.
- **2.0:** each dataset load creates `<logs>/session_<timestamp>/` with `events.jsonl` (every event), `calls/tNNNN_NN_<role>.json` (each model call: prompt, raw reply, parsed value, usage), `turn_NNN.md` (legacy format), `ledger.json`, `memory.json`, `agent_state.json`, `checkpoints.sqlite` (the graph state), `figures/`, `analysis.ipynb` (a markdown cell per step and a code cell per block). `POST /api/session/save` writes the slices (`.h5ad`), ROIs, ledger and memory to a directory; `POST /api/session {session_dir}` loads it and re-attaches the conversation, including a pending prompt. Reports: `GET /api/export/report.md|html` and `stat-agent session report`.
- **Verified:** <span class="chip t">T</span> save and load round trip in the web tests and `tests/graph/test_capabilities.py` (resume at a pending prompt after reload); <span class="chip l">L</span> the log directory and the Markdown report for your session; <span class="chip u">U</span> the HTML report in a browser, the notebook in Jupyter (Jupyter was not started, §9).

### 5.14 Command line

`stat-agent chat` (REPL or one-shot, `--mode auto`), `stat-agent web`, `stat-agent session report|load`, `stat-agent skill validate|list|lint-file|export|new`, `stat-web`, `stat-bench convert|run|summarize|replay|corrupt|session|judge`. <span class="chip t">T</span> `tests/unit/test_cli.py`, `tests/bench/test_bench_cli.py`; <span class="chip s">S</span> `skill validate`, `session report`, `stat-bench convert|run --dry-run|corrupt`.

### 5.15 Benchmark harness (`src/stat_bench`)

- Suite files (YAML) with expectations, scripted answers, auto hints and ground truth; a converter from the paper's `queries.json` (all 40 convert). `stat-bench run --system stat --driver staged|loop --ablation A0..A9 --policy scripted|auto|interactive --seeds N --backbone --out`, one `result.json` per query and seed (0.1 keys plus software versions, decisions of each role, clarifications, assumptions, refusal, method card, artefacts, provenance hash, usage per call, silent-error detectors SSE-1 to SSE-5), resume of completed cells, bootstrap confidence intervals in `summarize`. SessionBench: scripted sessions mixing user turns, GUI actions and programmatic verifiers (four archetypes shipped: cross-turn reference, correction, recovery, out-of-registry). Frontmatter corruption at three severities with a manifest; a decision-level replay report against 0.1 logs; the LLM judge ported from the paper protocol.
- **Verified:** <span class="chip t">T</span> `tests/bench/test_bench.py` (conversion, in-process run with resume and summary, detectors, corruption keeps the registry valid, replay report, the four SessionBench scripts pass); <span class="chip u">U</span> no real-backbone benchmark run yet; adapters for the other systems (vanilla, Biomni, SpatialAgent) are not written.

### 5.16 Tests and continuous integration

`tests/{unit, contract, property, executor, graph, web, bench, replay, live}` on synthetic in-memory datasets (`S1_cell_gene, S1_cell_gene_noCT, S2_spot_deconv, S3_two_slice_gene, S4_gene_protein`), a four-skill mini registry, a scripted responder keyed on role and prompt, and a scripted tool-calling model for loop mode. `.github/workflows/ci.yml` runs ruff, strict skill validation and the suite on Python 3.11. Live provider and real-skill tests are opt-in markers.

## 6. Intentional behaviour differences from 0.1

These are by design; they can look like bugs.

| Behaviour | 0.1 | 2.0 | Why |
|---|---|---|---|
| Matcher candidates | top 5 | top 2 (`STAT_MATCHER_TOP_K`) | The paper says 2; fewer skill-selection prompts. |
| Verbatim history | last 80 messages | last 4 turns, older turns as rolling summary plus handoffs and facts | Bounded prompts on any backbone; see §5.8. |
| Invalid model output | fabricated default, turn continues | one repair, then `turn_error` | No silent wrong answers. |
| Failed step in a plan | later steps still ran | dependants are blocked, turn ends `blocked` | No downstream results on missing inputs. |
| Unknown slice id from the planner | used as-is | clarification naming the bad id | |
| No skill matched | general analysis mode | method card, filter, verifier, unguarded run | Requirements checked before code; see §10 B3 for the cost. |
| Stale inputs | not tracked | verifier declines; badge and re-run offered | R2.3. |
| Clarification answers | free text | typed, tied to a prompt id | Replay and benchmarks. |
| Cancel | thread killed | cooperative, diff recorded | No half-written session state. |
| `current slice` | server-side | frontend only, sent as `interface_state` with each turn | Explicit slice ids everywhere. |
| Log writing | at turn end | at call time | Nothing lost on a crash. |

Event names changed as follows: `status/progress` to `phase`; `clarification_needed, prerequisites_needed, skill_selection` to `prompt{kind}`; `planning_complete/plan_created` to `plan_updated`; `step_start` to `step_started`; `execution_output` to `exec_output`; `step_execution_complete` to `step_done`; `state_changes` to `provenance_update` (the five flags are inside `summary`); `matching_complete/skill_matched` to `match_complete/skill_selected`; `verification_complete` to `verify_complete`; `reflection_start/complete` to `reflection`; `agent_text/response/final_response` to `text/text_delta/turn_done`; `no_skill_matched` to `method_card/unguarded_mode`.

## 7. Reviewer concerns and where 2.0 answers them

| Concern | What 2.0 provides |
|---|---|
| R1.1 architecture ablations | `AblationConfig` arms A0 to A9 as switches on the same graph; `stat-bench run --ablation` |
| R1.2 quantified interactivity | Interactive versus auto policies; every auto answer is an assumption event with its source; clarification counts and kinds in `result.json` |
| R2.1 matched single-agent baseline | Loop driver (A6): same tools, same runtime, same registry, guards off |
| R2.2 / R2.9 what is deterministic, unregistered methods | Table in `docs/v2-architecture.md`; method card protocol; refusal names the requirement; frontmatter corruption experiment; SSE detectors |
| R2.3 upstream correction | Provenance ledger, staleness propagation, verifier decline, "Re-run downstream", SessionBench cross-turn and correction scripts |
| R2.4 session-level benchmarks | SessionBench (scripts, GUI actions, verifiers) |
| R2.5 harness comparison | deepagents driver with subagents on the same guards |
| R2.6 repeated runs | `--seeds`, derived seeds per cell, bootstrap CIs |
| R2.8 auto versus interactive | `--policy auto|interactive|scripted` on the same suite |
| R2.m3 "one Markdown file per method" | Contracts (`consumes/produces/dependencies`), linter, `docs/adding-a-skill.md`, `skill new` |

## 8. Verification summary

| Area | Tests | Smoke | Live | Not exercised |
|---|---|---|---|---|
| Staged graph and clarifications | 43 (graph) | yes | 33 turns on 2026-09-13 by the test agent, multi-step and cancel included | |
| Roles and structured output | 13 | yes | 5 calls | repair path live (it crashed once, §10) |
| Skills registry, filter, linter | 11 | yes | filter and matcher on the real registry | executing any real skill body |
| Runtime | 12 | yes | 1 block | timeouts on long skills, RCTD workers |
| Ledger and mutator | 5 + web | yes | 1 step recorded | GUI edits during a turn |
| Memory | via graph | yes | 3 turns | summariser live |
| LLM providers | 5 | | Poe only | 8 providers |
| Loop mode, deepagents | 6 | | | any real model |
| Web backend | 9 | yes | 4 turns, load, report | reconnect, concurrency |
| Frontend | | headless Chrome probe (load, turn, prompt, reload, resize) | your session | ROI drawing, drawer, override, cancel, reconnect |
| Sessions, notebook, report | 3 | yes | log dir, report.md | HTML report, Jupyter |
| Benchmark harness | 7 | yes | | real runs, other systems |

## 9. Known gaps and risks

1. The browser interface has been exercised headless (chat flow, prompts, reload, resize) but not by a person; ROI drawing, the Changes drawer, assumption override and cancel are untested in a browser.
2. No real skill body has been executed under 2.0. Vendored packages (`celltype_fast, celltype_scvi, deconv_rctd, niche_analysis_lib, STalign`) load through `sys.path` scoping that is tested only with the mini skills.
3. Only Poe with `claude-opus-4.8` has been used live. GPT-5 through the Responses API, Anthropic direct, Gemini and the OpenAI-compatible endpoints are untested in 2.0; the 0.1 workarounds for Poe bots with forced thinking are gone.
4. Loop mode and the deepagents driver have only seen a scripted model.
5. The general-analysis path of 0.1 is missing (§10, B3). Until it returns, simple questions may end in a refusal or an extra model call.
6. Durable resume after a server restart is tested in Python, not through the browser.
7. Jupyter Lab was not started (the sandbox refused a token-less server); the notebook tab will not load until you start it (command in the previous message).
8. The 40-query live benchmark that was the gate of the plan has not been run; it needs API keys and the datasets.
9. The demo branch and the Hugging Face Space still serve 0.1.27.
10. The manuscript text (methods, framework statement, skill count, memory description) has not been updated.
11. Large datasets: the cells endpoint returns all 160,286 positions in one JSON payload as 0.1 did; no paging or binary transport was added.

## 10. Bugs found in your session and what to send me

Session directories: `~/Downloads/A0/session_20260912_122258` (turns 1 to 3), `~/Downloads/A0/session_20260912_123025` (turn 1), `~/Downloads/A0/session_20260912_140616` (bc_demo, turns 16 to 20 including the velocity and survival requests) and `~/Downloads/A0/session_20260912_150410` onward (crc_demo, fresh). Server log: `~/Downloads/A0/stat-web.log` (earlier logs rotated to `stat-web.log.<timestamp>`).

<div class="bug high" markdown="1">
**B1. Turn killed by `TypeError: Object of type ValueError is not JSON serializable`** (`session_..._123025`, t0001 "Describ the data").
**Root cause:** when a structured output failed validation, `llm/structured.py: repair_instruction` serialised pydantic's error list with `json.dumps`; custom validators put a `ValueError` object into the error context. The repair call never happened and the exception escaped as `turn_error`.
**Fix:** serialise with `default=str`. **Status: fixed.**
</div>

<div class="bug medium" markdown="1">
**B2. Method card wrote a positional list for a single slice** (`data_levels: [cell, spot]` with `num_slices: 1`).
**Fix:** `MethodCard` normalises single-slice alternatives into the `cell/spot` token before validation, so no repair round trip is needed. **Status: fixed, unit-tested.**
</div>

<div class="bug high" markdown="1">
**B3. Simple requests took the out-of-registry path and could be refused on a guessed key** ("What can you do for me?" ended `blocked` on "requires uns:spatial"; "Describ the data" needed a method card).
**Fix:** the planner (prompt v2) marks each step `kind: skill | general`; `general` steps whose request matches no skill run directly in unguarded mode, as the 0.1 general-analysis mode did; a method card is synthesised only when a method is named. For provisional cards the verifier's deterministic pre-pass only declines on inputs no analysis can produce (count layers); other absent keys go to the LLM verifier. **Status: fixed, tested (`test_general_step_runs_without_method_card`, `test_verifier_prepass_is_advisory_for_provisional_cards`); live: "Describe the data" and four niche follow-ups ran through this path.**
</div>

<div class="bug medium" markdown="1">
**B4. The planner answered a question inside its clarification field.**
**Fix:** the planner output gains `direct_answer` for messages that are not analysis requests (with the skill catalogue in the prompt so the answer is grounded); a long clarification without a question mark is treated as an answer; the prompt says never to decline an analysis that way. **Status: fixed, tested; live: "What can you do for me?" answered in one call, 8 s.**
</div>

<div class="bug high" markdown="1">
**B5. Turn state flipped before the agent had absorbed the result** (found by a flaky web test).
**Root cause:** the turn manager set `done` and `awaiting_input` when the corresponding event arrived, which the agent emits before its generator returns and the memory is updated; a client that saved the session or replied immediately raced it.
**Fix:** terminal and prompt states are stashed on the event and applied after the agent's generator returns. **Status: fixed; the web suite passes repeatedly.**
</div>

<div class="bug high" markdown="1">
**B6. Loading a dataset through the form never opened the event stream.** The init-form handler had its own copy of the entry sequence that lacked `openEventStream()` and `restoreChatHistory()`, so after "Load Dataset" turns ran on the server but nothing rendered in the chat until the page was reloaded. This is the likely source of most of what you saw.
**Fix:** the handler calls `enterApp()`, the single entry path. **Status: fixed; verified headless (events arrive after form load).**
</div>

<div class="bug medium" markdown="1">
**B7. Every page reload rendered the history twice** (the stream was opened from sequence 0, replaying all frames, and then the history was replayed again).
**Fix:** history is restored first, the stream then continues from the last replayed sequence; the reducer also de-duplicates by sequence; a resync forgets the detached bubble before re-rendering. **Status: fixed; verified headless (6 messages before and after reload).**
</div>

<div class="bug medium" markdown="1">
**B8. Raw generator code streamed into the chat bubble.** Streamed tokens were appended as plain text, fences included, until the code block executed.
**Fix:** the stream preview shows prose and replaces an open fence with a "Writing code · N lines" chip; the collapsed Code card appears when the block runs. **Status: fixed.**
</div>

<div class="bug medium" markdown="1">
**B9. The chat list could not be scrolled to its last message.** The messages container is a flex item without `min-height: 0`, so it grew past the panel and its bottom was clipped; auto-scroll also fought the reader.
**Fix:** `min-height: 0` on the panel and the list; auto-scroll follows new content only while the reader is at the bottom, and always after sending. **Status: fixed; verified headless (last message visible after scroll at 1440x900 and 1100x700).**
</div>

<div class="bug high" markdown="1">
**B10. A skill step that only inspected the data was accepted as done, and the ledger credited it with outputs it never wrote** (your two-part immune/tumor query: step 2 matched the DE skill, printed a crosstab, and the turn ended with the comparison never run).
**Root cause:** on every successful run the execute node added the skill's declared `produces` to the ledger's write set, so pre-existing artefacts (here `uns:de_results` from an earlier turn) were re-credited to the step and the output check passed on existence alone.
**Fix:** only what the code wrote (assignments and known library effects) or what changed in the snapshot counts as produced; the output check now requires production by this step, so an inspection-only step fails validation, is reflected on, and ends `blocked` if the retry still produces nothing. The generator is also told to finish a step in one response (plan block and prompt v2). **Status: fixed, tested (`test_skill_step_must_produce_its_declared_outputs`).**
</div>

<div class="bug medium" markdown="1">
**B11. Loading a saved session wrote new events and figures into the saved directory** instead of the new session log directory.
**Fix:** `Agent.load(..., logs_dir=)`; the web layer passes the new log directory. **Status: fixed, tested.**
</div>

<div class="bug medium" markdown="1">
**B12. The skills panel showed empty rows.** The panel reads `title`, the 2.0 endpoint sent only `name`.
**Fix:** `title` added to the skills listing. **Status: fixed, tested.**
</div>

<div class="bug medium" markdown="1">
**B13. The chat could intermittently not be scrolled to its last lines** (the input box and the end of the last message were cut off, and no scrolling revealed them).
**Root cause:** the main layout had a fixed height of `100vh - 44px`, assuming only the 44 px header above it. The reconnect banner sits above the header in normal flow; whenever it appeared (any event-stream hiccup, for example after a server restart) the page became taller than the viewport and, since the body clips overflow, the bottom of the chat column was pushed off-screen.
**Fix:** the app container is a full-height flex column: banner and header take their natural height and the main layout fills the remainder (`flex: 1; min-height: 0`). **Status: fixed; verified headless with the banner forced visible at 1440x900 and 1100x700.**
</div>

<div class="bug high" markdown="1">
**B14. The method-card prompt named the RNA-velocity test case as its example** (a prompt leak: `prompts/method_card.v1.md` illustrated `method_name` with "scVelo RNA velocity" and `prerequisites` with "spliced and unspliced count layers (adata.layers[...])").
**Root cause:** the example was written while the out-of-registry path was being built against that exact acceptance test, so the prompt spoon-fed the expected contract and primed the `layers` kind that the deterministic decline keys on. The two clean velocity refusals observed before this fix (your t0019 and one isolated run) therefore overstated robustness for this case.
**Fix:** the example is now neutral ("a per-cell time-point label in adata.obs['timepoint']"); no prompt file mentions velocity or spliced counts any more; the velocity case stays a held-out test (`test_capabilities.py`, SessionBench OR-1). **Status: fixed; re-run live once after the change (bc_demo, opus 4.8): the card still listed `layers:spliced` and `layers:unspliced`, the pre-pass declined before any code, 3 calls, 29 s.**
</div>

<div class="bug medium" markdown="1">
**B15. The refusal for a missing count layer told you to run cell type annotation** (t0019: "This analysis requires layers:spliced, which does not exist on slice(s) [0]. Run the step that produces it first (e.g. cell type annotation), then retry.").
**Root cause:** the verifier's deterministic pre-pass had one sentence for every absent input, written with `obs` artefacts in mind, and returned on the first missing key, so a second missing layer was never named.
**Fix:** missing layers are collected and refused in one message that names every missing layer and says they come from upstream quantification, not from an analysis step ("This analysis needs the count layer(s) 'spliced', 'unspliced' in adata.layers on slice(s) [0], and the loaded data has none. ..."); the `obs` message names cell type annotation only for `celltype`. **Status: fixed, unit-tested (`test_verifier_prepass_is_advisory_for_provisional_cards`), seen live in the B14 re-run.** Still open and cosmetic: the same sentence appears twice in the chat (refusal card, then the step note), and the method card's own prerequisites are not shown to you.
</div>

<div class="bug high" markdown="1">
**B16. Slices without an image were served as multi-megabyte blank JPEGs that the viewer threw away** (crc_demo, first load: slice 0 took 6.3 s and 7 MB, slice 1 10.6 s and 11.6 MB, slice 2 32 s and 32.6 MB for the image call alone).
**Root cause:** `web/render.py: image_payload` built a white RGB image the size of the coordinate space (24,422 x 36,819 px for slice 0, about 2.7 GB in memory), JPEG-encoded and base64-encoded it, although the viewer's scatter mode explicitly skips decoding it and paints its own white background.
**Fix:** an imageless slice now returns only its coordinate extent with `is_scatter: true` and no image bytes; the viewer accepts that payload. **Status: fixed, unit-tested (`test_imageless_slice_returns_extent_without_a_jpeg`); re-timed live after the restart: 122 bytes and under 10 ms per slice.** The cell overlay (15 to 21 MB of JSON per slice, about 0.5 s on the server) and a gene expression call (17.5 MB, about 1 s) are now the largest transfers; they are inherent to 311k to 431k bins per slice and unchanged from 0.1.
</div>

<div class="bug high" markdown="1">
**B17. Stop was only honoured before a code block** (crc_demo session `session_20260912_151000`: t0001 stopped a 19-minute Moran's I run within a second, then still called the analyser and reported "the computation did not finish" as a failed step; t0002 stopped right after planning, yet the matcher, verifier and generator ran for another 33 s while the interface said "Stopping...", so you pressed Stop twice).
**What was right:** `Agent.cancel` interrupts the running code thread (a 431,004-cell permutation loop stopped at once), the generated code worked on a copy, and provenance recorded no change, so the slice was untouched.
**Root cause:** the stop flag was checked in two places only (start of a step, before each block). After an interrupted block the pipeline went on to the analyser; between the planner and the first block nothing looked at the flag. The runtime labelled the interrupt `Timeout` because the same event served the timeout path, and `finish_turn` mapped a cancelled status to `turn_done {state: "blocked"}`, which the turn manager then recorded as `done`.
**Fix:** one `cancel_turn` helper checked at every stage boundary (match, method card, verify, generate, analyse) and after every block; the generation stream is dropped mid-way on stop; a stopped block ends the turn immediately with its diff recorded and no reflection or analysis; the block is labelled `Cancelled` ("execution stopped by the user"); `turn_done` carries `state: "cancelled"` and the turn state follows it. What remains: a non-streaming model call that is already in flight (planner, matcher, verifier, analyser, 2 to 20 s) still completes before the stop lands. **Status: fixed, tested (`test_cancel_between_stages_stops_before_the_next_model_call`, `test_cancel_during_code_interrupts_it_and_skips_analysis`); needs the server restart to take effect.**
</div>

<div class="bug medium" markdown="1">
**B18. `GET /api/roi/list` returned a 500 with a traceback while the agent held the session lock** (seen at 15:32 during the long run: the interface's ROI refresh hit the lock and the route did not convert `SessionBusy` into the 409 `busy` reply the other data routes use).
**Fix:** the route now answers 409 `busy` like image, cells and expression. **Status: fixed, tested (`test_roi_list_reports_busy_while_the_agent_holds_the_session`).** Note for your multi-slice testing: while a code block runs, every viewer read (image, cells, expression, ROI list) waits up to the lock timeout and then answers busy, so switching slices during a long computation shows a busy message until the block ends or you press Stop.
</div>

<div class="bug high" markdown="1">
**B19. The postcondition system was stricter than its own contract and recovered by re-running everything** (bc_demo t0002 "Please redo the deconvolution": RCTD ran for 3 min 57 s and wrote its results; the skill's `rows_sum_to_one` check, declared `severity: warn`, failed on RCTD's raw weights; the checker ignored severity, marked the block failed, and the reflector's "small fix" restarted the whole workflow, which you had to stop).
**Root cause:** value-level checks (row sums, unique counts, row counts, ranges) encoded expectations the skills did not reliably meet, `severity` was never honoured (a leftover `or True` from the first 2.0 commit), `in_range` was never implemented, the `skill_contract_violation` event was never emitted, the verifier's `contract_conflict` flag was never read, and a failed check took the same path as a Python exception: reflect and re-execute.
**Fix (your decision, 2026-09-12):** the postcondition system is reduced to one rule, checked once after the step's code has run: every declared output must exist and must have been written by this step. A step that ran cleanly but did not deliver its outputs gets one continuation on the live namespace (the generator is told which outputs are missing and which variables are still defined) and, if still incomplete, is reported as partial; a finished block is never re-run. Removed: the `postconditions` frontmatter key and its five check types, `severity`, `in_range`, the `skill_contract_violation` event, `contract_conflict`; the `postconditions` blocks are stripped from all 25 skill files while `produces` stays. The RCTD skill now stores proportions (row-normalised) so its documented output holds. The skill-guidance header no longer says "use it step-by-step"; it asks for all stages in one response. **Status: done, tested (`test_skill_step_must_produce_its_declared_outputs` rewritten, `test_incomplete_step_gets_one_continuation_on_the_live_namespace` added), 26 skills validate with 0 warnings; live check pending on your next deconvolution run.**
</div>

<div class="bug high" markdown="1">
**B20. The method card fired for generic workflows and could refuse them on a guessed input** (crc t0006 "Pool every Goblet spot across slices 0/1/2 ... and run standard scanpy clustering": no skill matched, a card named `layers:counts` as an input, and the deterministic rule declined the step although the raw counts are in `adata.X`; the plan then went on without step 1 and step 3 redid the work).
**Root cause:** the planner marked every analysis-like step as `skill`, so any unmatched request produced a card; the card is a guess, and the "absent count layer" rule (written for spliced/unspliced counts) treated any missing layer as impossible.
**Fix (your decision, option 1):** the planner marks a step `skill` only for a registered skill's analysis or a named algorithm, tool or package; everything else, including custom workflows built from standard operations, is `general` and runs directly as in 0.1. The card role reports `named_method`; a card for a generic workflow sends the step to the general path. The hard decline is limited to the quantification-only layers spliced, unspliced and ambiguous; any other absent input goes to the LLM verifier. The generator receives the card as hints, not instructions. Today's five cards: two correct refusals (velocity, survival), one wrong refusal (this case), two pre-B3 artefacts. **Status: done, tested (`test_card_that_names_no_method_runs_as_general_analysis`, layers case in `test_verifier_prepass_is_advisory_for_provisional_cards`).**
</div>

<div class="bug medium" markdown="1">
**B21. Eight skill contracts declared outputs their code never writes on the target slice** (RCTD `var:Marker`/`var:highly_variable` on the reference copy; STalign's four coordinates on a new slice; SpaGCN `spatial_domain_raw`, Palantir `uns:iroot`, bbknn `X_pca`/`X_umap`, scanorama `X_scanorama` on working copies; the squidpy statistics skill's union of four modes; neighbourhood enrichment's raw table and graph on the copy). The linter accepted them because it sees the key being assigned, not the object.
**Fix:** contracts trimmed to what every run writes onto the slice (RCTD: `obs:celltype`, `obsm:deconv_weights`); mode-dependent and new-slice skills declare nothing and the output check is skipped for them; the linter treats an empty list as information, not a warning. 26 skills validate with 0 warnings. **Status: done.** This is what made B22 fire on a correct RCTD run.
</div>

<div class="bug medium" markdown="1">
**B22. The continuation fired for results that live on derived objects, and then showed the analyser mismatched code and output** (bc t0002: RCTD complete, two over-declared outputs "missing", a pointless continuation, then an "Execution Issue (partial_success)" card claiming prints were missing; crc t0006: DE results on a pooled object, a continuation that crashed, a reflector retry that recomputed everything for 5.6 minutes while the server could not answer).
**Root cause:** the continuation ran whenever a declared output was absent, not only for the inspection-only case it was built for; and after a continuation the step's code segments were replaced while the results accumulated, so the analyser paired the continuation's code with the first block's output.
**Fix:** the continuation runs only when the step wrote nothing at all; a step that wrote other artefacts is noted in the pipeline log and kept successful; segments accumulate across a continuation and only the new blocks run, so the analyser sees every block next to its own output. **Status: done, tested (`test_step_that_writes_other_artefacts_is_noted_not_continued`, `test_continuation_gives_the_analyser_both_blocks`).** Not changed: the plan still continues after a blocked step unless the planner declares a dependency, and the plan panel repaints a failed step as done; listed with the review items.
</div>

<div class="bug low" markdown="1">
**B23. A second `__INTERPRET__` marker leaked into the chat** (three turns today; the model writes one marker per code block in long responses).
**Root cause:** the fence parser removed only the first marker. **Fix:** every marker is removed and the focus texts are joined; the streaming preview hides marker lines too. **Status: done, tested (`test_parse_segments_strips_every_interpret_marker`).**
</div>

<div class="bug high" markdown="1">
**B24. Later steps of a plan did not know what the earlier steps had produced** (crc t0001, 2026-09-13: step 1 pooled 117,845 Goblet spots and clustered them in 80 s with the igraph Leiden backend; step 2's generator prompt contained nothing about step 1, so it searched the namespace for a `leiden` column, missed the `cluster` column step 1 had written, rebuilt the object with the slower leidenalg backend for 9.9 minutes and plotted a different partition, which its own analyser flagged as partial_success; step 3 rebuilt a third time; the server was unresponsive throughout because leidenalg holds the GIL, F9).
**Root cause:** the history block renders only finished turns, and this turn's handoffs were read by finish_turn alone, although the step 1 analyser had written every variable name into its data-changes field. 0.1 appended each step's summary to memory right after the step, so 2.0 had lost that carry. Nothing told the model that the runtime namespace persists, and the runtime's `namespace_added` had no consumer.
**Fix:** the verifier, generator and analyser of every later step receive a "Completed steps in this turn" block (summary, data changes, findings, artefacts, variables added, and the code of the latest steps within a 6,000-character budget); the generator also receives a listing of the live Python namespace with types and keys, and the generator system prompt (v2) states the persistence contract: reuse listed variables, keep intermediates in plain variables, write final results into the AnnData. Handoffs and block results now record the variables each step added; the plan card keeps its dependencies after a step finishes. Memory-off arm: neither block. **Status: done, tested (`test_later_steps_see_earlier_steps_and_the_namespace`, `test_memory_off_arm_gets_neither_block`, `tests/unit/test_memory_context.py`).** Not changed: heavy C++ calls still freeze the server (F9); which Leiden backend the model picks is its own choice.
</div>

<div class="bug high" markdown="1">
**B25. Answers that were not true** (second review and live test, 2026-09-13). (a) After an ROI was deleted the ledger marked the comparison stale but "is the earlier comparison still valid?" was answered "yes": the planner's direct-answer path never saw the ledger. (b) "How many spots were in cluster 3?" got a bubble with no number: the analyser's summary held it and the response builder rendered only the interpretation. (c) When a result table fell past the analyser's 3,000-character output cap it wrote textbook biology, and its prompt forbade flagging the truncation. (d) The analyser never saw the ledger's list of what the step produced, so `data_changes` was an unverified claim. (e) A cancelled turn was later recalled as completed, and a step stopped mid-way left no memory handoff.
**Fix:** the planner receives the Session Artefacts block with STALE marks; a step without an interpretation renders the analyser summary as a Result line; the analyser reads the head and the tail of long output and analyser prompt v2 says to report a missing result rather than infer it; the produced list is given to the analyser as a fact; a stop during a step goes through `finish_step` (handoff with `success=false`, `step_done` and the plan card say cancelled, later steps skipped) and the turn record's history line says the turn was stopped. **Status: done, tested** (`test_planner_sees_stale_artefacts`, `test_result_line_when_the_step_has_no_interpretation`, `test_assemble_context_keeps_head_and_tail_of_long_output`, cancel tests).
</div>

<div class="bug high" markdown="1">
**B26. Paths that reported done on nothing.** A reply in a bare code fence ran no code and the step was done; a verifier or planner that kept asking looped without limit (119 calls in a probe); a reply with an empty answers map or an option reply without an index ran the turn on option 1; a block that crashed after writing half a column left that column fresh and attributed; generated code that assigned `session`, `np` or `pd` poisoned the runtime for the rest of the session; "no" as a skill choice selected annotation.
**Fix:** untagged fences run when their body parses as Python (other languages stay prose); the verifier and planner re-asks are capped at two like the skill-selection prompt, then the step blocks or the turn blocks with advice; content-less replies return 400; artefacts written by a failed or stopped run are recorded stale with the reason; the seeded names are never overwritten; a free-text skill choice must be the slug, the label or a whole word of it. **Status: done, tested** (`test_untagged_python_fence_runs_and_other_fences_stay_prose`, `test_verifier_reask_is_capped`, `test_planner_reask_is_capped`, `test_reply_validation_rejects_optionless_and_empty_replies`, `test_failed_run_marks_its_writes_stale`, `test_seeded_names_survive_a_run`, `test_skill_selection_free_text_must_name_an_option`).
</div>

<div class="bug medium" markdown="1">
**B27. Context across turns and prompt hygiene.** The per-step handoffs and the facts were written to memory and never shown to a model; a prerequisite answered in one turn was asked again in the next; the general-analysis note was rendered under the "SKILL GUIDANCE PROVIDED, follow exactly" banner; the namespace paragraph said "never rebuild"; auto mode answered every planner question with a slice list.
**Fix:** the history carries one line per step (skill, artefacts, variables, unresolved), the known facts and the clarification answers of the last two turns, which the verifier sees as answers already given; the note and a method card render outside the banner; the paragraph says to rebuild what depends on a STALE artefact; auto mode answers only data questions with the slice list and blocks on a question it cannot answer. **Status: done, tested** (`test_history_lines_carry_step_details_facts_and_earlier_answers`, `test_verifier_sees_answers_from_earlier_turns`, `test_unguarded_note_is_outside_the_skill_banner`, `test_auto_mode_blocks_on_a_planner_question_it_cannot_answer`).
</div>

<div class="bug medium" markdown="1">
**B28. Continuation and contracts.** On the crc pooled-Goblet query the marker step wrote its tables onto the pooled object, so nothing changed on the slices, the continuation re-ran the analysis and copied the pooled tables onto all three slices (the ledger then listed six artefacts as created). Four skills declared output unions no single run writes (SpatialDE, Harmony, Palantir, Tangram); the generator was never shown the outputs it is checked against; fifteen skills' declared defaults never reached auto mode; a single-gene Moran's I request became a whole-transcriptome permutation test.
**Fix:** no continuation when the code's traced writes already cover a declared output (the step is accepted and only noted); the four contracts declare what every run writes; the skill guidance ends with the declared outputs; the verifier fills the context's skill defaults; the squidpy skill says to test only the named genes. **Status: done, tested** (`test_results_on_a_derived_object_are_accepted_without_continuation`, `test_unguarded_note_is_outside_the_skill_banner` for the outputs line, `test_skill_defaults_reach_the_auto_policy`).
</div>

<div class="bug medium" markdown="1">
**B29. Frontend wedges and event headers.** Logout followed by a new dataset froze the chat (the duplicate filter kept the old sequence numbers); a 409 from Re-run left the chat disabled; a pending prompt had no Stop and cancelling it published no event; skill-selection cards took no typed text; blocked steps were painted failed and a cancelled step stayed active; the documented `step` header was always null.
**Fix:** logout resets the seen set; a refused re-run resets the busy state and shows the error; Stop stays visible while a prompt is pending and typed text answers every prompt; cancelling a waiting turn publishes `turn_cancelled` and `turn_done`; the plan card has failed, blocked, skipped and cancelled styles; events between `step_started` and `step_done` carry the step number, also after a resume. **Status: done, tested** (web cancel test, `test_single_step_with_verifier_pause_and_resume` step assertions).
</div>

<div class="bug medium" markdown="1">
**B30. Named methods and files next to the data.** "Run CellChat on slice 2" silently ran LIANA (the matcher absorbed the request; the method-card route never fired). Four turns died asking for a reference h5ad that sat in `bc_demo/ref/`, because no role could see the dataset directory.
**Fix:** matcher prompt v2 returns no match for a request naming a method the skill does not implement, and the generator's notes ask it to state a substitution in the first sentence; the session summary lists h5ad files next to the slices with their full paths and the verifier and generator session blocks show them. **Status: done, tested** (`test_matcher_prompt_declines_named_methods_the_skill_does_not_implement`, `test_extra_files_next_to_the_dataset_are_listed`).
</div>

<div class="bug high" markdown="1">
**B31. Phantom changes and contradictory validity answers** (second live-user run, 2026-09-13, t0011/t0017/t0026 and t0017 to t0019). Code that wrote a same-named key on a copy, or ran Leiden with a custom key, made the ledger report the slice's own `obs:leiden` as updated and marked the annotation, DE and GO results stale; three consecutive answers then contradicted each other about validity, because the analyser and summariser never saw the stale set.
**Root cause:** the code scanner cannot see which object a write targets and the library-effects table maps every Leiden call to `obs:leiden`; the "re-written unchanged artefact counts as re-produced" rule turned that into a change with staleness propagation. The analyser had no stale input.
**Fix:** only the snapshot diff decides what changed; an unchanged key the code assigned is never an update and never a staleness root, and only clears a stale flag on that key (a producer re-run with identical values). The analyser receives the stale artefacts of its slices with the instruction never to call them valid. **Status: done, tested** (`test_unchanged_write_is_not_a_change_but_reproduces_a_stale_key`, `test_analyser_sees_stale_artefacts_of_its_slices`).
</div>

<div class="bug high" markdown="1">
**B32. Wrong slice, invented method, silent overwrite** (t0020, t0009, t0039). "Here" and "this slice" were planned on another slice than the interface showed; "annotate those clusters" on spot data made the method-card role invent a named method requiring an API key; in auto mode RCTD replaced the dataset's own cell types and weights without saying so.
**Fix:** planner rule that "here" means the interface slice and that the interface wins when the user says they switched; the card names a method only when the user's own words name one and never invents an API-key requirement; the generator is told never to overwrite data that came with the dataset unless asked; the ledger records which loaded artefacts a run replaced, a `warning` event reports it and the analyser must state it. **Status: done, tested** (`test_step_that_overwrites_loaded_data_is_flagged`; prompt rules).
</div>

<div class="bug medium" markdown="1">
**B33. Disabled skills, plot narration, counts without numbers, figures missing from the report, summariser after Stop** (t0028, t0006/t0021/t0032, t0004/t0012, export, t0031). A request naming cell2location, installed but switched off, ran a 13-minute provisional re-implementation; plot questions came back with textbook narration; a count question's answer summarised the table instead of giving it; the report export had no figures; after Stop the summariser still ran for 43 seconds; a recap credited a skill that was selected but never ran.
**Fix:** a request naming an installed but disabled skill gets advice to enable it; analyser rules that it cannot see figures and that a factual answer must carry the printed values; the report embeds every step's figures (Markdown links, base64 in HTML); no summariser call on a cancelled turn; history lines say "not run" for a skill that did not run. **Status: done, tested** (`test_request_naming_a_disabled_skill_gets_advice`, `test_report_embeds_figures`, `test_no_summariser_call_after_a_stop`).
</div>

<div class="bug high" markdown="1">
**B34. Resync storm and script injection in the page** (interface review U1, U2). A block printing 3,000 lines produced 473 resync frames and 947 extra requests and could lose the transcript; a markdown link in model text and an ROI name with a quote both ran an event handler, and the ROI became undeletable.
**Fix:** output lines are batched (one event per ~100 ms or 50 lines, in order), the event log sends one resync per overflow episode, and the page runs one resync at a time with a 300 ms coalescing delay and resets the Changes drawer state before the replay; the HTML escaper escapes quotes and markdown links are limited to http, https and mailto. **Verified in the browser** with the reviewer's own scripts: 3,000 lines gave 60 output events and no resync; the crafted links render as text; the quoted ROI name renders escaped and deletes. **Status: done, tested** (`test_output_lines_are_batched_in_order`, `test_event_log_queue_overflow_adds_one_resync_sentinel`).
</div>

<div class="bug medium" markdown="1">
**B35. Session lifecycle in the page** (U3, U4, U5, U6). Loading a dataset from one tab left another tab on a dead event log showing a turn as running forever; a mistyped dataset path destroyed the live session before failing; Logout during a turn was refused by the server, ignored by the page, and left the turn running; after a reload mid-turn the chat stayed empty and idle for six seconds.
**Fix:** a new session publishes `session_reset` on the old log before swapping it (the other tab reloads); the dataset directory or saved session is validated before the teardown (404 with the session intact); Logout forces the reset, stops its status timer and clears stale badges; the history and the event stream are restored before the slice data, whose requests wait on the session lock. **Verified in the browser**: the second tab reloads within a second, the bad path leaves the session alive, the server session and turn are gone after Logout, and the chat shows the running turn one second after a reload. **Status: done, tested** (`test_bad_dataset_path_keeps_the_live_session`, `test_new_session_tells_the_old_event_log`).
</div>

<div class="bug low" markdown="1">
**B36. Small interface defects** (U7, U8, U9, M10, notebook link). The status line stuck on "Stopping" after cancelling a waiting turn; the cancelled prompt card kept an enabled Submit; at 1024 by 700 the viewer collapsed to a 158 pixel sliver with a clipped toolbar; a double Enter on an answer sent a second reply and reset the chat state mid-turn; the notebook button opened the Jupyter root.
**Fix:** the status is set before the cancel request; a finished turn disables its open prompt cards; below 1200 pixels the viewer keeps at least 300 pixels and the toolbar wraps; one reply per prompt with the prompt restored only on a validation or mismatch error; the notebook button opens the notebook URL. **Verified in the browser** with the reviewer's scripts (idle status after cancel, no enabled dead card, one reply request on a double Enter, 388 pixel viewer and unclipped toolbar at 1024 by 700). **Status: done.**
</div>

<div class="bug medium" markdown="1">
**B37. Six false "stale" marks after a pooled analysis** (CRC session, turns 6 and 7). Turn 6 clustered the Goblet spots of all three slices in one pooled AnnData and stored the cluster markers on every slice under new keys. Turn 7's pooled macrophage DE then overwrote slice 0's `rank_genes_groups`, and the Changes panel marked the six Goblet artefacts stale with "input changed". The Goblet step had read `rank_genes_groups` only on the in-memory pooled object; the code scan attributed every `X.uns['key']` read to the target slices, so the ledger had recorded slice 0's and slice 1's `rank_genes_groups` as inputs of the Goblet results.
**Fix:** the scan is receiver-aware. A read counts only when its receiver can be a session slice (`session.get_slice(i).adata`, a name bound to one, a slice iterated from the session, a subset or copy of one); a read on an object built by another call (`ad.concat`, `sc.read_h5ad`) or on a namespace variable holding an AnnData that is not a slice is not a slice read; an aliased container (`obs = adata.obs; obs['celltype']`), which the scan used to miss entirely, is now followed too. Writes are unchanged (the snapshot diff decides what changed). **Status: done, tested** (`test_code_access_reads_only_count_on_session_slices`, `test_reads_on_a_pooled_object_are_not_slice_inputs`). A session started before the fix keeps its marks; "Re-run" there recomputes the Goblet markers on the pooled object still in the namespace and clears them.
</div>

<div class="bug high" markdown="1">
**B38. Viewer requests froze the whole server while a block ran** (third review F1). Twelve request handlers took the session lock on the event-loop thread. The runtime holds that lock for the whole of a code block, so every viewer click during an analysis parked the loop for the two-second lock timeout before its 409, requests that take no lock waited behind it, the event stream stalled, and with eight clicks in flight a Stop was not even received for 16 seconds.
**Fix:** the locked bodies of the data, ROI, cell-type, snapshot, consistency and save routes run in a worker thread (`in_thread` / `under_lock` in `web/routes/common.py`); the lock semantics are unchanged, the loop never waits. Dataset loading still runs on the loop behind its overlay. **Status: done, tested** (`test_waiting_for_the_session_lock_does_not_block_other_requests`: three viewer requests wait for their 409s while `/api/meta` answers in milliseconds); live, five viewer requests and `/api/meta` all answered within 0.4 s during a running block.
</div>

<div class="bug high" markdown="1">
**B39. The analyser was told that writes on a copy had hit the slices** (third robustness run F2). The "artefacts written this step (from the provenance ledger)" block, the step handoff and the report listed what the code assigned on any object, qualified per target slice, so a step that clustered a copy was reported as having written PCA, neighbours and a `leiden` column onto the slice. Since B31 the ledger itself ignored those writes, so the chat confessed overwrites the Changes panel denied, and the phantom artefacts reached later recaps, a methods paragraph and the exported report.
**Fix:** `produced` is the ledger's list only (plus keys a re-run reproduced, which the ledger re-attributes to the step); what the code assigned counts for the output check alone (`claimed`), so a step working on a copy or reproducing identical values still passes. With provenance switched off the two lists coincide as before. **Status: done, tested** (`test_writes_on_a_copy_are_not_reported_as_produced`).
</div>

<div class="bug high" markdown="1">
**B40. Auto mode answered every verifier question with the slice** (third robustness run F5). A known fact keyed `slice` matched any question mentioning a slice, so "how many domains on slice 2? (e.g., 7)" and "'Visium' or 'ST'?" were answered "Slice 2" until the re-ask cap blocked the turn; the number rule also picked the slice number.
**Fix:** answer order is the skill's defaults, explicit hints, the species / tissue / column conventions, a fact whose key appears as a whole word in the question (generic keys such as `slice` never answer), then a value the question itself offers (a stated default, an example, the first of the quoted options), and last a number that does not follow "slice". **Status: done, tested** (`test_auto_policy_answers_verifier_questions_sensibly`).
</div>

<div class="bug medium" markdown="1">
**B41. The fast annotation skill could not reach a model** (third robustness run F9). Its vendored code reads `session.llm_config` and imports `stat_agent.agent.llm_backend.LLMBackend`, both 0.1 conventions that 2.0 no longer provided, so "name the groups" failed with `AttributeError` and degraded to `Cluster_0/1/2`.
**Fix:** `SimpleSession.llm_config` (provider, model id, api key, base URL) is filled by the agent from its model spec, and `stat_agent.agent.llm_backend.LLMBackend` keeps the 0.1 surface on the 2.0 client. **Status: done, tested** (`test_session_llm_config_is_set_for_skill_code`, `test_llm_backend_shim_runs_on_the_client`) and verified live with the skill.
</div>

<div class="bug low" markdown="1">
**B42. A meaningless reply to a planner question counted as an answer** (third robustness run F10). "zzzz qwrty" to "which of (a) or (b)?" ran option (a).
**Fix:** a planner rule: a reply that does not answer the question (empty, meaningless, unrelated, a question back) is a reason to ask again within the existing two-re-ask cap; a reply that gives the choice in other words remains an answer. **Status: done, tested** (`test_planner_prompt_re_asks_on_a_non_answer`) and verified live.
</div>

<div class="bug medium" markdown="1">
**B43. Long sessions forgot their first chapter** (third robustness run F4). Anything older than the four-turn window lived only in an eight-sentence rolling summary that was told to describe "what has been done"; after 46 turns it described the current state, and the first clustering's parameters, the cluster names and an early "do not overwrite celltype" instruction were merged away, so recall questions were answered wrongly and the instruction was later violated.
**Fix:** the verbatim window is six turns and the history budget 24,000 characters; the summary's word budget grows with the turns it covers (150 words plus 40 per turn, at most 1,200) instead of a fixed sentence count; the summariser keeps superseded work marked as superseded and the user's standing instructions in their words, and records each instruction as a fact (`instruction: ...`); 24 facts are shown instead of 12. **Status: done, tested** (`test_memory_window_facts_and_summary_word_budget`, `test_summarizer_prompt_scales_with_the_turns_it_covers`); the wording of the summary itself is model-dependent and worth a look in a long live session.
</div>

### 10b. Review items fixed (evening of 2026-09-12)

The independent review's findings are numbered F1 to F50 in its own report. Fixed here, all covered by the suite (134 passed):

<div class="tablewrap" markdown="1">

| finding | what was wrong | fix |
|---|---|---|
| F1 critical | the documented reply body (`answer`, `option_index`, `answers` without `kind`) was accepted with 202 and silently dropped; my probe script used exactly that shape, so its prompted turns ran on empty answers | both bodies are accepted and mapped the same way; a reply with no answer returns 400; the probe sends the viewer's shape; `test_reply_body_aliases_map_to_decisions`, `test_reply_accepts_documented_shapes` |
| F2 critical | a skill's union of mode-dependent outputs forced unrequested heavy work | closed by B21 (contracts trimmed) and B22 (continuation only for inspection-only steps) |
| F4 high, F21 | `snapshot` and `consistency` answered 500 while the agent held the session; `save` serialised without the lock | all three answer 409 `locked`; `test_snapshot_consistency_and_save_report_busy` |
| F5 high | one 409 from a viewer read blanked a slice for good | a busy envelope is retried after 5 s with a "session busy" spinner; the slice is marked loaded only after both reads succeed |
| F12 | Stop during scANVI was reported as a `ValueError` because the library swallows the interrupt | any exception raised after a stop request is labelled `Cancelled` |
| F13 | progress bars never reached the interface (a 7-minute training looked frozen) | the latest carriage-return segment is forwarded at most every 2 s |
| F14 | 11 of 26 skills could never be enabled (`default_skill: false` was ANDed into the enabled state) | 0.1 semantics restored: `default_skill` only sets the initial state, toggles survive a reload; `test_non_default_skill_can_be_enabled_and_stays_enabled_across_reloads` |
| F19 | a client reconnecting with an event id from a previous server life got an empty replay and no `resync` | `since > log.seq` now yields `resync {reason: ahead_of_log}`; `test_events_resync_when_the_client_is_ahead_of_the_log` |
| F36, F37, F38 | the Logs modal's notebook tab was empty; six places rendered the error envelope as `[object Object]`; a `blocked` turn looked like a success | notebook payload accepted as returned; one `errText` helper at every site; a "turn ended early" line for blocked turns |
| F44 | the ledger version moved on every run, even a cancelled one with an empty entry | no entry and no bump when nothing changed; `test_ledger_ignores_a_no_change_run` |
| F45, F47, F50 | dead idempotency logic, an unread analyser field, a test assertion that could not fail | removed |
| (mine) | the plan carried on after a blocked step and repainted it as done | the planner now declares `depends_on` per step; a failed step with a dependant ends the turn and marks the rest skipped; step statuses persist in the turn state and every plan view shows them; `test_failed_step_stays_failed_in_the_plan_view` |

</div>

Still open from the review, waiting for your decision: F3 (loop mode and ablation arms are unreachable from the server: wire them or remove the claim), F9 (an in-process memory guard or a subprocess runtime for heavy blocks; the interpreter-lock starvation you saw belongs here), F10/F22/F26 (provenance fidelity: value-based fingerprints, per-slice write attribution, producer re-attribution on dtype casts), F17 (the planner's fact extraction is dropped), F20 (a saved session does not restore its pending prompt or turn index), the benchmark-scoring set F6/F8/F11/F16/F27 to F31/F33/F48 (to be fixed before any paper numbers), the CLI pair F7/F32, and the smaller F15, F18, F23, F24, F25, F29, F34, F35, F39 to F43, F46, F49.

**Prompt audit (at your request, 2026-09-12).** Apart from B14, no prompt tells the model how to answer a specific request. Soft spots, left as they are: planner Example 8 carries a written-out reply to "What can you do for me?" that the model paraphrases (the skills catalogue block is the legitimate source); the few-shot examples inherited verbatim from 0.1 mention breast cancer, niche detection, ERBB2 and `tumor_region`, which happen to match bc_demo; the verifier skips the absent-input decline when the query contains annotate, label, assign, classify or deconvolve; auto-mode defaults (species human, tissue from the slice name, never an invented path) are emitted as assumption events. The probe script's reply rules and the benchmark suites' auto hints are scripted clarification answers outside the agent and are recorded as assumptions.

**How to report the interface bugs you still see.** For each one, send: what you did (message or click), what you expected, what happened, the turn id from the status line if any, and the matching lines from `events.jsonl` or `stat-web.log`. A screenshot helps for layout problems. I reproduce through the fake-LLM server (`python scripts/dev_server_fake.py`) and the headless probe (`python scripts/ui_probe.py`) where possible, so a bug that does not need a real model is fixed and covered the same day.

## 11. How to debug 2.0

- **A turn did something wrong:** open `calls/tNNNN_NN_<role>.json` for the role that decided it; the file holds the exact prompt, the raw reply and the parsed value. `turn_NNN.md` is the same in the 0.1 layout.
- **The interface shows something odd:** `events.jsonl` is the authoritative sequence the interface consumed; `GET /api/turns/{id}/events?format=json` returns the same for one turn.
- **Data or ROI state looks wrong:** `ledger.json` lists every artefact with its producer and status; `GET /api/debug/consistency` hashes the live session, the ledger and the snapshot so a mismatch is detectable.
- **Reproduce without a model:** `python scripts/dev_server_fake.py --port 8899` serves the interface on a synthetic dataset with the scripted responder; `pytest -q` runs everything in about 14 s.
- **Drive the real server through the API:** `python scripts/api_probe.py --dataset ~/Downloads/bc_demo --model poe/claude-opus-4.8 "Describe the data" "Detect spatial niches"` runs turns, answers prompts from a rule table and prints per-turn decisions, code results, provenance and usage; `--out` keeps a JSON report.
- **Drive the interface headless:** `python scripts/ui_probe.py --base http://127.0.0.1:8899 --dataset <dir> --say "annotate cell types" --reload --narrow` needs Google Chrome; it reports console errors, layout metrics and screenshots per step.
- **Settings:** `STAT_MODEL, STAT_MODE, STAT_DRIVER, STAT_HOME, STAT_LOGS_DIR, STAT_JUPYTER_ROOT, STAT_EXEC_TIMEOUT, STAT_MAX_REFLECT, STAT_MATCHER_TOP_K, STAT_SKILL_PATHS`, plus a provider key.

## Appendix A. Event catalogue

Session: `hello, ping, resync, lock_state, session_reset`. Turn: `turn_started, phase, plan_updated, step_started, step_done, turn_done, turn_error, turn_cancelled`. Selection: `filter_complete, match_complete, skill_selected, verify_complete`. Clarification: `prompt, prompt_answered, assumption`. Execution: `code_block_started, exec_output, code_block_done, figure, execution_issue, reflection, validate_result`. Text: `text, text_delta`. State: `provenance_update, usage, notebook_cell_added`. Advisory: `advice, warning, error`. Out of registry: `method_card, unguarded_mode, refusal, draft_offer`. Header on every event: `v, seq, turn_id, step, ts`.

## Appendix B. Routes

New: `GET /api/meta`, `POST|GET|DELETE /api/session`, `GET /api/session/snapshot`, `POST /api/session/save`, `GET /api/events`, `POST|GET /api/turns`, `GET /api/turns/{id}`, `GET /api/turns/{id}/events`, `POST /api/turns/{id}/reply`, `POST /api/turns/{id}/cancel`, `GET /api/slices/{id}/image|cells|genes|expression`, `PATCH /api/slices/{id}/celltype_colors`, `PATCH /api/slices/{id}/celltypes/{name}`, `POST|GET /api/rois`, `PATCH|DELETE /api/rois/{name}`, `GET /api/figures/{id}.png`, `GET /api/skills`, `POST /api/skills/draft`, `PATCH /api/skills/{slug}`, `GET /api/skills/{slug}/detail`, `GET /api/logs`, `GET /api/logs/turns/{file}`, `GET /api/logs/notebook`, `GET /api/notebook/url`, `GET /api/export/report.{md|html}`, `POST /api/llm/test`, `GET /api/debug/consistency`.

Legacy aliases kept: `/api/init_dataset, /api/session/summary, /api/session/reset, /api/image/data, /api/cells/overlay, /api/genes/list, /api/cells/gene_expression, /api/select_slice, /api/select_modality, /api/roi/add|list|delete|rename, /api/skills/list|toggle, /api/logs/info, /api/logs/turn/{file}, /api/test-llm, /api/chat/history, /api/chat/abort`.

## Appendix C. Skill frontmatter v2

`schema_version, version, consumes[kind:key], produces[kind:key], platforms[], dependencies{python[], vendored[], gpu, subprocess_workers, notes}, defaults{prerequisite: answer}`; filter grammar adds `min_slices`.

## Appendix D. Ablation arms

A0 full · A1 filter off · A2 verifier off · A3 reflector off · A4 planner off · A5 memory off · A6 loop driver, guards off · A7 analyser off · A9 method card off. Custom combinations: `--ablation "filter=off,verifier=off"`.

## Appendix E. Test inventory

`unit/test_core_session.py` 8 · `unit/test_dataset_io.py` 5 · `unit/test_colors.py` 3 · `unit/test_llm_client.py` 5 · `unit/test_prompts_and_events.py` 8 · `unit/test_skill_schema_and_filter.py` 6 · `unit/test_cli.py` 1 · `contract/test_skill_contracts.py` 5 (parametrised over all 26 skills) · `property/test_ledger.py` 3 · `executor/test_runtime.py` 12 · `graph/test_staged.py` 11 · `graph/test_capabilities.py` 3 · `graph/test_loop.py` 6 · `web/test_web_api.py` 9 · `bench/test_bench.py` 6 (SessionBench parametrised over 4 scripts) · `bench/test_bench_cli.py` 1 · `replay/test_trace_parser.py` 2 · `live/test_live_smoke.py` 2 (opt-in).
