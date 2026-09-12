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
   `POST /api/turns/{id}/reply {prompt_id, answer | option_index | answers}`; a mismatched or missing
   prompt returns `409`. Auto mode emits `assumption` instead and continues.
4. `POST /api/turns/{id}/cancel` is cooperative: a running code block finishes and its diff is still
   recorded; the turn ends with `turn_cancelled`.
5. Failures end with `turn_error` (a stage crashed) or `turn_done {state: "blocked"}` (a step could
   not run and a later step depended on it).

Other events: `method_card`, `unguarded_mode`, `refusal`, `validate_result`,
`skill_contract_violation`, `draft_offer` (out-of-registry path), `advice`, `warning`, `error`,
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
