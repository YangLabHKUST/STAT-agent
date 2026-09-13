"""Shared route helpers."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from stat_agent.events.encoding import dumps
from stat_agent.web.errors import ApiError

T = TypeVar("T", bound=BaseModel)


def state_of(request: Request):
    return request.app.state.stat


def json(payload: Any, status: int = 200, headers: dict | None = None) -> Response:
    hdrs = dict(headers or {})
    return Response(dumps(payload), status_code=status, media_type="application/json", headers=hdrs)


async def parse_body(request: Request, model: type[T]) -> T:
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        data = {}
    try:
        return model.model_validate(data or {})
    except ValidationError as exc:
        raise ApiError("validation", "invalid request body", 400, exc.errors()) from exc


def slice_from(request: Request, session, param: str = "slice_id"):
    raw = request.query_params.get(param)
    if raw is None or raw == "":
        ids = session.slice_ids
        if not ids:
            raise ApiError("no_session", "no slices loaded", 400)
        return session.get_slice(ids[0])
    try:
        return session.get_slice(int(raw))
    except ValueError as exc:
        raise ApiError("not_found", str(exc), 404) from exc


def ledger_headers(app) -> dict:
    return {"X-STAT-Ledger-Version": str(app.agent.ledger.version if app.agent else 0), "X-STAT-Generation": str(app.generation)}


async def in_thread(fn, *args, **kwargs):
    """Run a body that may wait for the session lock in a worker thread. The runtime holds the lock
    for the whole of a code block, so waiting for it (up to ``lock_timeout_s``) on the event loop
    would stall every other request, the event stream and Stop."""
    return await run_in_threadpool(fn, *args, **kwargs)


async def under_lock(session, holder: str, timeout: float, fn, *args, **kwargs):
    """``fn(*args, **kwargs)`` under the session lock, in a worker thread (see ``in_thread``)."""

    def _locked():
        with session.lock.hold(holder, timeout=timeout):
            return fn(*args, **kwargs)

    return await run_in_threadpool(_locked)


__all__ = ["state_of", "json", "parse_body", "slice_from", "ledger_headers", "in_thread", "under_lock", "JSONResponse"]
