"""Turn routes and the SSE event streams."""

from __future__ import annotations

import asyncio
import json as _json

from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from starlette.requests import Request
from starlette.routing import Route

from stat_agent.events.catalogue import PROTOCOL_VERSION
from stat_agent.events.encoding import dumps
from stat_agent.web.errors import ApiError
from stat_agent.web.routes.common import json, parse_body, state_of
from stat_agent.web.schemas import ReplyDecision, TurnCreate


def _hello(app, active) -> ServerSentEvent:
    data = {"v": 2, "type": "hello", "protocol": PROTOCOL_VERSION, "last_seq": app.events.seq, "ledger_version": app.agent.ledger.version if app.agent else 0, "generation": app.generation, "active_turn": active}
    return ServerSentEvent(data=dumps(data), event="hello", retry=2000)


async def _stream(request: Request, turn_id: str | None):
    app = state_of(request)
    log = app.events
    since_raw = request.headers.get("last-event-id") or request.query_params.get("since") or "0"
    try:
        since = int(since_raw)
    except ValueError:
        since = 0
    queue = await log.subscribe()
    try:
        log.bind_loop(asyncio.get_running_loop())
        yield _hello(app, app.turns.active_snapshot() if app.turns else None)
        if log.evicted(since) or since > log.seq:
            # evicted from the ring buffer, or a client that reconnects with an id from a previous server life
            yield ServerSentEvent(data=dumps({"v": 2, "type": "resync", "reason": "buffer_evicted" if since <= log.seq else "ahead_of_log", "last_seq": log.seq}), event="resync")
            since = 0
        for seq, etype, frame in log.replay(since, turn_id):
            yield ServerSentEvent(data=frame, event=etype, id=str(seq))
        if request.query_params.get("once") in ("1", "true"):
            return
        if turn_id is not None and app.turns.get(turn_id).state in ("done", "error", "cancelled"):
            return
        while True:
            if await request.is_disconnected():
                return
            try:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            seq, etype, frame = item
            if etype == "resync":
                yield ServerSentEvent(data=dumps({"v": 2, "type": "resync", "reason": "slow_consumer", "last_seq": log.seq}), event="resync")
                continue
            if turn_id is not None:
                try:
                    tid = _json.loads(frame).get("turn_id")
                except Exception:  # noqa: BLE001
                    tid = None
                if tid != turn_id:
                    continue
            yield ServerSentEvent(data=frame, event=etype, id=str(seq))
            if turn_id is not None and etype in ("turn_done", "turn_error", "turn_cancelled"):
                return
    finally:
        log.unsubscribe(queue)


async def events(request: Request):
    app = state_of(request)
    return EventSourceResponse(_stream(request, None), ping=app.config.ping_s, ping_message_factory=lambda: ServerSentEvent(data="{}", event="ping"))


async def turn_events(request: Request):
    app = state_of(request)
    turn_id = request.path_params["turn_id"]
    app.turns.get(turn_id)  # 404 before the stream starts
    accept = request.headers.get("accept", "")
    if "text/event-stream" not in accept and request.query_params.get("format", "") != "sse":
        return json({"success": True, "turn_id": turn_id, "events": app.events.turn_frames(turn_id)})
    return EventSourceResponse(_stream(request, turn_id), ping=app.config.ping_s, ping_message_factory=lambda: ServerSentEvent(data="{}", event="ping"))


async def create_turn(request: Request):
    app = state_of(request)
    app.require_session()
    body = await parse_body(request, TurnCreate)
    turn = await app.turns.start(body.message, mode=body.mode, driver=body.driver, client_turn_id=body.client_turn_id, interface=body.interface_state, action=body.action.model_dump() if body.action else None)
    return json({"success": True, "turn_id": turn.turn_id, "first_seq": turn.first_seq, "state": turn.state}, status=202)


async def list_turns(request: Request):
    app = state_of(request)
    items = [t.snapshot() for t in app.turns.history] + ([app.turns.active.snapshot()] if app.turns.active else [])
    return json({"success": True, "turns": items})


async def get_turn(request: Request):
    app = state_of(request)
    return json({"success": True, "turn": app.turns.get(request.path_params["turn_id"]).snapshot()})


async def reply(request: Request):
    app = state_of(request)
    body = await parse_body(request, ReplyDecision)
    turn = await app.turns.reply(request.path_params["turn_id"], body.model_dump())
    return json({"success": True, "accepted": True, "turn": turn.snapshot()}, status=202)


async def cancel(request: Request):
    app = state_of(request)
    turn = await app.turns.cancel(request.path_params["turn_id"])
    return json({"success": True, "state": turn.state, "cancelling": turn.cancelling}, status=202)


async def history(request: Request):
    """Legacy chat history: assembled from the memory records."""
    app = state_of(request)
    agent = app.require_agent()
    hist = []
    for t in agent.memory.turns:
        hist.append({"role": "user", "content": t.query, "turn_id": t.turn_id})
        hist.append({"role": "assistant", "content": t.response, "turn_id": t.turn_id, "status": t.status})
    return json({"success": True, "history": hist})


async def legacy_abort(request: Request):
    app = state_of(request)
    if app.turns.active:
        await app.turns.cancel(app.turns.active.turn_id)
    return json({"success": True})


def routes() -> list[Route]:
    return [
        Route("/api/events", events, methods=["GET"]),
        Route("/api/turns", create_turn, methods=["POST"]),
        Route("/api/turns", list_turns, methods=["GET"]),
        Route("/api/turns/{turn_id}", get_turn, methods=["GET"]),
        Route("/api/turns/{turn_id}/events", turn_events, methods=["GET"]),
        Route("/api/turns/{turn_id}/reply", reply, methods=["POST"]),
        Route("/api/turns/{turn_id}/cancel", cancel, methods=["POST"]),
        Route("/api/chat/history", history, methods=["GET"]),
        Route("/api/chat/abort", legacy_abort, methods=["POST"]),
    ]


__all__ = ["routes", "ApiError"]
