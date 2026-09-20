"""Session lifecycle, snapshot and meta routes."""

from __future__ import annotations

import hashlib

from starlette.requests import Request
from starlette.routing import Route

from stat_agent import __version__
from stat_agent.core.lock import SessionBusy
from stat_agent.events.catalogue import PROTOCOL_VERSION
from stat_agent.events.encoding import dumps
from stat_agent.web.errors import ApiError
from stat_agent.web.routes.common import in_thread, json, parse_body, state_of, under_lock
from stat_agent.web.routes.data import _busy
from stat_agent.web.schemas import SessionCreate


async def meta(request: Request):
    app = state_of(request)
    return json({"success": True, "protocol": PROTOCOL_VERSION, "app_version": __version__, "demo_mode": app.config.demo_mode, "capabilities": {"ws": False, "stream_text": True}, "defaults": {"mode": app.settings.mode, "driver": app.settings.driver}, "jupyter_port": app.config.jupyter_port, "generation": app.generation})


async def create_session(request: Request):
    app = state_of(request)
    body = await parse_body(request, SessionCreate)
    try:
        app.create_session(body.dataset_dir, session_name=body.session_name, provider=body.provider, model=body.model, api_key=body.api_key, base_url=body.base_url, reasoning_effort=body.reasoning_effort, mode=body.mode, session_dir_to_load=body.session_dir)
    except FileNotFoundError as exc:
        raise ApiError("not_found", str(exc), 404) from exc
    except ValueError as exc:
        raise ApiError("bad_dataset", str(exc), 400) from exc
    return json(app.summary_payload())


async def get_session(request: Request):
    app = state_of(request)
    return json(app.summary_payload())


async def delete_session(request: Request):
    app = state_of(request)
    force = request.query_params.get("force") in ("1", "true")
    turn = app.turns.active if app.turns else None
    if turn and turn.state in ("running", "awaiting_input") and not force:
        raise ApiError("busy", "a turn is active; cancel it or pass force=1", 409, turn.snapshot())
    app.reset()
    return json({"success": True, "generation": app.generation})


async def snapshot(request: Request):
    app = state_of(request)
    session = app.require_session()
    agent = app.require_agent()
    try:
        payload = await under_lock(session, "gui:snapshot", app.config.lock_timeout_s, _snapshot_payload, app, session, agent)
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json(payload)


def _snapshot_payload(app, session, agent) -> dict:
    from stat_agent.web.render import roi_payload

    if True:

        payload = {
            "success": True,
            "summary": session.get_frontend_summary(),
            "rois": [roi_payload(r) for r in session.roi_manager.rois.values()],
            "ledger": {"version": agent.ledger.version, "stale": agent.ledger.stale_ids(), "artifacts": {k: {"version": a.version, "status": a.status, "kind": a.kind, "key": a.key, "slice_id": a.slice_id, "produced_by": a.produced_by.model_dump() if a.produced_by else None} for k, a in agent.ledger.live().items() if a.kind not in ("X", "var")}},
            "active_turn": app.turns.active_snapshot() if app.turns else None,
            "last_seq": app.events.seq,
            "generation": app.generation,
            "mode": agent.mode,
            "assumptions": [a.model_dump() for t in agent.memory.turns for a in t.assumptions],
        }
    return payload


async def consistency(request: Request):
    app = state_of(request)
    session = app.require_session()
    agent = app.require_agent()
    try:
        digest = await in_thread(_consistency_digest, app, session, agent)
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json({"success": True, "ledger_version": agent.ledger.version, "hash": digest, "ledger_hash": agent.ledger.consistency_hash(), "n_artifacts": len(agent.ledger.live())})


def _consistency_digest(app, session, agent) -> str:
    with session.lock.hold("gui:consistency", timeout=app.config.lock_timeout_s):
        parts = []
        for sid in session.slice_ids:
            s = session.get_slice(sid)
            parts.append({"slice_id": sid, "obs": sorted(map(str, s.adata.obs.columns)), "obsm": sorted(map(str, s.adata.obsm.keys())), "uns": sorted(map(str, s.adata.uns.keys())), "n_obs": s.n_obs, "celltypes": s.celltypes, "colors": s.get_celltype_colors()})
        rois = [{"name": r.name, "slice_id": r.slice_id, "n_obs": r.n_obs, "bounds": list(r.bounds)} for r in session.roi_manager.rois.values()]
        canonical = dumps({"slices": parts, "rois": rois, "stale": agent.ledger.stale_ids()}, sort_keys=True)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
    return digest


async def save_session(request: Request):
    app = state_of(request)
    agent = app.require_agent()
    from stat_agent.io.session_store import save_session as _save

    body = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        pass
    target = body.get("dir") if isinstance(body, dict) else None
    try:
        out = await under_lock(app.session, "gui:save", app.config.lock_timeout_s, _save, app.session, agent, target)
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json({"success": True, "dir": str(out)})


def routes() -> list[Route]:
    return [
        Route("/api/meta", meta, methods=["GET"]),
        Route("/api/session", create_session, methods=["POST"]),
        Route("/api/session", get_session, methods=["GET"]),
        Route("/api/session", delete_session, methods=["DELETE"]),
        Route("/api/session/snapshot", snapshot, methods=["GET"]),
        Route("/api/session/save", save_session, methods=["POST"]),
        Route("/api/debug/consistency", consistency, methods=["GET"]),
        # legacy aliases
        Route("/api/init_dataset", create_session, methods=["POST"]),
        Route("/api/session/summary", get_session, methods=["GET"]),
        Route("/api/session/reset", delete_session, methods=["POST"]),
    ]
