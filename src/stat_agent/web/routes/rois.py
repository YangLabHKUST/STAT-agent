"""ROI routes: every mutation goes through the SessionMutator (lock + ledger + event)."""

from __future__ import annotations

from starlette.requests import Request
from starlette.routing import Route

from stat_agent.core.lock import SessionBusy
from stat_agent.web.errors import ApiError
from stat_agent.web.render import roi_payload
from stat_agent.web.routes.common import in_thread, json, ledger_headers, parse_body, state_of, under_lock
from stat_agent.web.schemas import ROICreate, ROIPatch


def _busy(exc: SessionBusy) -> ApiError:
    return ApiError("locked", f"session is busy ({exc.holder})", 409, {"holder": exc.holder}, {"Retry-After": "2"})


def _definition(body: ROICreate) -> dict:
    p = body.params
    if body.type == "bbox":
        keys = ("min_x", "min_y", "max_x", "max_y")
        if not all(k in p for k in keys):
            raise ApiError("validation", "bbox needs min_x, min_y, max_x, max_y", 400)
        return {"type": "bbox", "x_min": float(p["min_x"]), "x_max": float(p["max_x"]), "y_min": float(p["min_y"]), "y_max": float(p["max_y"])}
    if body.type == "circle":
        return {"type": "circle", "center_x": float(p["center_x"]), "center_y": float(p["center_y"]), "radius": float(p["radius"])}
    verts = p.get("vertices") or []
    if len(verts) < 3:
        raise ApiError("validation", "polygon requires at least 3 vertices", 400)
    return {"type": "polygon", "vertices": [(float(v[0]), float(v[1])) for v in verts]}


async def list_rois(request: Request):
    app = state_of(request)
    session = app.require_session()
    try:
        rois = await under_lock(session, "gui:roi_list", app.config.lock_timeout_s, lambda: [roi_payload(r) for r in session.roi_manager.rois.values()])
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json({"success": True, "rois": rois, "ledger_version": app.agent.ledger.version if app.agent else 0}, headers=ledger_headers(app))


async def create_roi(request: Request):
    app = state_of(request)
    session = app.require_session()
    body = await parse_body(request, ROICreate)
    slice_id = body.slice_id if body.slice_id is not None else (session.slice_ids[0] if session.slice_ids else None)
    if slice_id is None or not session.has_slice(slice_id):
        raise ApiError("not_found", f"slice {slice_id} not found", 404)
    if body.name in session.roi_subsets:
        raise ApiError("conflict", f'ROI "{body.name}" already exists', 409)
    definition = _definition(body)
    try:
        res = await in_thread(app.mutator.create_roi, body.name, slice_id, definition)
    except SessionBusy as exc:
        raise _busy(exc) from exc
    except ValueError as exc:
        raise ApiError("validation", str(exc), 400) from exc
    roi = res.value
    if app.agent and app.agent.notebook is not None:
        try:
            app.agent.notebook.append_roi_creation(body.name, slice_id, definition)
        except Exception:  # noqa: BLE001
            pass
    payload = roi_payload(roi)
    payload.update({"success": True, "roi_name": roi.name, "roi": roi_payload(roi), "provenance_update": res.event.model_dump(mode="json") if res.event else None})
    if "vertices" in definition:
        payload["vertices"] = [list(v) for v in definition["vertices"]]
    return json(payload, headers=ledger_headers(app))


async def delete_roi(request: Request):
    app = state_of(request)
    session = app.require_session()
    name = request.path_params.get("name")
    if name is None:
        body = await request.json()
        name = (body or {}).get("name")
    if not name or name not in session.roi_subsets:
        raise ApiError("not_found", f'ROI "{name}" not found', 404)
    try:
        res = await in_thread(app.mutator.delete_roi, name)
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json({"success": True, "provenance_update": res.event.model_dump(mode="json") if res.event else None}, headers=ledger_headers(app))


async def rename_roi(request: Request):
    app = state_of(request)
    session = app.require_session()
    if "name" in request.path_params:
        old = request.path_params["name"]
        body = await parse_body(request, ROIPatch)
        new = body.new_name
    else:
        data = await request.json()
        old, new = (data or {}).get("old_name"), (data or {}).get("new_name")
    if not old or not new:
        raise ApiError("validation", "old_name and new_name required", 400)
    if old not in session.roi_subsets:
        raise ApiError("not_found", f'ROI "{old}" not found', 404)
    if new in session.roi_subsets:
        raise ApiError("conflict", f'ROI "{new}" already exists', 409)
    try:
        res = await in_thread(app.mutator.rename_roi, old, new)
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json({"success": True, "provenance_update": res.event.model_dump(mode="json") if res.event else None}, headers=ledger_headers(app))


def routes() -> list[Route]:
    return [
        Route("/api/rois", list_rois, methods=["GET"]),
        Route("/api/rois", create_roi, methods=["POST"]),
        Route("/api/rois/{name}", rename_roi, methods=["PATCH"]),
        Route("/api/rois/{name}", delete_roi, methods=["DELETE"]),
        # legacy aliases
        Route("/api/roi/list", list_rois, methods=["GET"]),
        Route("/api/roi/add", create_roi, methods=["POST"]),
        Route("/api/roi/delete", delete_roi, methods=["POST"]),
        Route("/api/roi/rename", rename_roi, methods=["POST"]),
    ]
