"""Viewer data routes (explicit slice ids; GETs never mutate the session)."""

from __future__ import annotations

from starlette.requests import Request
from starlette.routing import Route

from stat_agent.core.lock import SessionBusy
from stat_agent.web.errors import ApiError
from stat_agent.web.render import expression_payload, image_payload, overlay_payload
from stat_agent.web.routes.common import json, ledger_headers, parse_body, slice_from, state_of
from stat_agent.web.schemas import CelltypeRename, ColorPatch


def _busy(exc: SessionBusy) -> ApiError:
    return ApiError("locked", f"session is busy ({exc.holder})", 409, {"holder": exc.holder}, {"Retry-After": "2"})


def _slice(request: Request):
    app = state_of(request)
    session = app.require_session()
    if "slice_id" in request.path_params:
        try:
            return app, session, session.get_slice(int(request.path_params["slice_id"]))
        except ValueError as exc:
            raise ApiError("not_found", str(exc), 404) from exc
    return app, session, slice_from(request, session)


async def image(request: Request):
    app, session, s = _slice(request)
    try:
        with session.lock.hold("gui:image", timeout=app.config.lock_timeout_s):
            payload = image_payload(s)
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json(payload, headers=ledger_headers(app))


async def cells(request: Request):
    app, session, s = _slice(request)
    sel = request.query_params.get("selected_celltypes")
    selected = [x.strip() for x in sel.split(",")] if sel else None
    try:
        with session.lock.hold("gui:cells", timeout=app.config.lock_timeout_s):
            payload = overlay_payload(session, s, selected)
            payload["versions"] = {"celltype": (app.agent.ledger.get(f"s{s.slice_id}:obs:celltype") or type("x", (), {"version": 0})).version if app.agent else 0, "ledger_version": app.agent.ledger.version if app.agent else 0}
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json(payload, headers=ledger_headers(app))


async def genes(request: Request):
    app, session, s = _slice(request)
    genes = [str(g) for g in s.adata.var_names]
    return json({"success": True, "genes": genes, "n_genes": len(genes), "slice_id": s.slice_id}, headers=ledger_headers(app))


async def expression(request: Request):
    app, session, s = _slice(request)
    gene = request.query_params.get("gene")
    if not gene:
        raise ApiError("validation", "gene is required", 400)
    try:
        with session.lock.hold("gui:expression", timeout=app.config.lock_timeout_s):
            payload = expression_payload(s, gene)
    except KeyError:
        raise ApiError("not_found", f'Gene "{gene}" not found in slice {s.slice_id}', 404) from None
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json(payload, headers=ledger_headers(app))


async def patch_colors(request: Request):
    app, session, s = _slice(request)
    body = await parse_body(request, ColorPatch)
    try:
        res = app.mutator.set_celltype_colors(s.slice_id, body.colors)
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json({"success": True, "colors": res.value, "provenance_update": res.event.model_dump(mode="json") if res.event else None}, headers=ledger_headers(app))


async def rename_celltype(request: Request):
    app, session, s = _slice(request)
    body = await parse_body(request, CelltypeRename)
    old = request.path_params["name"]
    if not s.has_celltype():
        raise ApiError("not_found", "slice has no celltype column", 404)
    try:
        res = app.mutator.rename_celltype(s.slice_id, old, body.new_name)
    except SessionBusy as exc:
        raise _busy(exc) from exc
    return json({"success": True, "provenance_update": res.event.model_dump(mode="json") if res.event else None}, headers=ledger_headers(app))


async def legacy_noop(request: Request):
    return json({"success": True, "note": "slice/modality selection is client-side in STAT v2"})


def routes() -> list[Route]:
    return [
        Route("/api/slices/{slice_id:int}/image", image, methods=["GET"]),
        Route("/api/slices/{slice_id:int}/cells", cells, methods=["GET"]),
        Route("/api/slices/{slice_id:int}/genes", genes, methods=["GET"]),
        Route("/api/slices/{slice_id:int}/expression", expression, methods=["GET"]),
        Route("/api/slices/{slice_id:int}/celltype_colors", patch_colors, methods=["PATCH", "POST"]),
        Route("/api/slices/{slice_id:int}/celltypes/{name}", rename_celltype, methods=["PATCH", "POST"]),
        # legacy aliases (slice_id as a query parameter; default = first slice)
        Route("/api/image/data", image, methods=["GET"]),
        Route("/api/cells/overlay", cells, methods=["GET"]),
        Route("/api/genes/list", genes, methods=["GET"]),
        Route("/api/cells/gene_expression", expression, methods=["GET"]),
        Route("/api/select_slice", legacy_noop, methods=["POST"]),
        Route("/api/select_modality", legacy_noop, methods=["POST"]),
    ]
