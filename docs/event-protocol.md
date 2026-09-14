# Event protocol (v2)

The web layer exposes one ordered event log per session. The browser opens a single
`EventSource` on `GET /api/events`; every frame is a typed event from
`stat_agent.events.catalogue` with the header `{v, seq, turn_id, step, ts}`.

## Framing

* `id:` is the sequence number `seq`; `event:` is the event `type`; `data:` is the JSON body
  (NaN is never emitted; numpy values are converted).
* `hello` is sent on connect with the current `last_seq`, the session generation and the active turn.
* `ping` frames carry no id and are sent every few seconds.
* Reconnecting with `Last-Event-ID` replays the missed frames from the ring buffer; when the
  buffer no longer holds them a `resync` frame tells the client to reload the snapshot
  (`GET /api/session/snapshot`) and continue from `last_seq`.
* `GET /api/events?since=<seq>` replays from a sequence; `once=1` replays and closes (used by tests
  and by `GET /api/turns/{id}/events` for a per-turn view, also available as JSON).

## Turn lifecycle

1. `POST /api/turns {message, mode?, interface_state?}` returns `202 {turn_id}` or `409` when a turn
   is active. `{action: "rerun_downstream", ids: [...]}` re-runs the producers of stale artefacts.
2. Events: `turn_started`, `phase`, `plan_updated`, `step_started`, `filter_complete`,
   `match_complete`, `skill_selected`, `verify_complete`, `text` / `text_delta`,
   `code_block_started`, `exec_output`, `code_block_done`, `figure`, `reflection`,
   `execution_issue`, `provenance_update`, `usage`, `step_done`, `turn_done`.
3. A clarification emits `prompt {prompt_id, kind: planner|verifier|skill_selection, question |
   questions | options}` and the turn state becomes `awaiting_input`. The client answers with
   `POST /api/turns/{id}/reply {prompt_id, answer | option_index | answers}` (the viewer sends `{prompt_id, kind: answer|option|prerequisites, text|index|answers}`; both are accepted; a reply with no answer, an option reply without an index or with index 0, and a prerequisites reply whose answers are all empty return `400`); a mismatched or missing
   prompt returns `409`. Typed text answers every prompt kind (a number picks a skill option). Auto mode answers through its policy: the `prompt` is still
   emitted, followed by `assumption` and `prompt_answered {source: "auto"}`; a planner question that is not about which data to use and has no
   default or known fact blocks the turn with `advice`. A verifier or planner that keeps asking is stopped after two re-asks: the step blocks
   with `advice`, or the turn ends `turn_done {state: "blocked"}`.
4. `POST /api/turns/{id}/cancel` stops the turn cooperatively: a running code block is interrupted
   (`code_block_done` with error type `Cancelled`) and its diff is still recorded; every stage
   boundary checks the stop flag and an in-flight generation stream is dropped, so at most one
   non-streaming model call completes after the stop. No reflection or analysis runs for a stopped
   block. A stop during a step closes that step: `step_done {status: "cancelled"}` and a last `plan_updated`
   (the step `cancelled`, later steps `skipped`), so the step keeps its memory handoff. The turn ends with
   `turn_cancelled` followed by `turn_done {state: "cancelled"}`. Cancelling a turn that is waiting for a reply
   publishes the same two events. Artefacts written by a stopped or failed block are recorded as stale.
5. Failures end with `turn_error` (a stage crashed) or `turn_done {state: "blocked"}` (a step could
   not run and a later step depended on it, or a clarification could not be settled).
6. `text` (role `answer`) and `code_block_started` carry `segment_index`, the position of that prose
   or code block among the step's generated segments (a continuation keeps counting), so a client can
   show prose and code in the order they were written even though the prose is published first and each
   block when it runs; `figure` carries `width` and `height` read from the PNG. Block ids count every block
   of the step, so a continuation never reuses one.
7. Every event carries `step`, the plan step number, from `step_started` to `step_done` (null for
   turn-level events). `plan_updated` step statuses are `pending | active | done | failed | blocked |
   skipped | cancelled`; `step_done.status` uses the same words without `pending` and `active`.
8. `exec_output` carries a batch of lines (one event per ~100 ms or 50 lines, in order), so a
   verbose block cannot overflow a subscriber queue; when a queue does overflow, the log sends one
   `resync` per overflow episode and the page replays from the snapshot once, coalescing bursts.
9. `warning {code: "replaced_loaded"}` is emitted when a step overwrote an artefact that came with
   the dataset (for example `obs:celltype`); the analyser is told the same and says so in its summary.
10. Loading a dataset or a saved session publishes `session_reset` on the previous event log first,
   so a page still attached to it reloads instead of listening to a dead log; a dataset directory
   without `.h5ad` files, or a saved session without `manifest.json`, is refused with `404` before
   the live session is touched. A stopped turn runs no summariser call.

Other events: `method_card`, `unguarded_mode`, `refusal`, `validate_result`,
`draft_offer` (out-of-registry path), `advice`, `warning`, `error`,
`lock_state`, `session_reset`, `notebook_cell_added`.

## Errors

Every error is a JSON envelope `{success: false, error: {code, message}}` with an HTTP status:
`400 validation`, `400 no_session`, `404 not_found`, and the `409` codes `turn_active`, `busy` (the
runtime is executing), `locked` (a GUI mutation could not take the session lock),
`no_pending_prompt` and `prompt_mismatch`. Responses that mutate the session carry the ledger
version in the `X-STAT-Ledger-Version` header.

## Legacy aliases

The 0.1.x endpoints (`/api/init_dataset`, `/api/session/summary`, `/api/image/data`,
`/api/cells/overlay`, `/api/genes/list`, `/api/cells/gene_expression`, `/api/roi/*`,
`/api/skills/list|toggle`, `/api/logs/info`, `/api/chat/history`, `/api/chat/abort`) are served as
thin aliases so existing scripts keep working. `/api/select_slice` and `/api/select_modality` are
no-ops: the backend has no current slice.
