"""Starlette application factory."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from stat_agent.config import Settings
from stat_agent.skills import SkillRegistry
from stat_agent.web.errors import (
    ApiError,
    api_error_handler,
    error_response,
    unhandled_error_handler,
    validation_error_handler,
)
from stat_agent.web.routes import data, misc, rois, session, turns
from stat_agent.web.state import AppState, WebConfig
from stat_agent.web.turns import TurnManager

logger = logging.getLogger(__name__)
HERE = Path(__file__).parent


class NoCacheStatic(StaticFiles):
    def file_response(self, *args, **kwargs) -> Response:  # type: ignore[override]
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


def create_app(config: WebConfig | None = None, settings: Settings | None = None, registry: SkillRegistry | None = None) -> Starlette:
    config = config or WebConfig()
    settings = settings or Settings.from_env()
    registry = registry or SkillRegistry.from_settings(settings)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    static_dir = config.static_dir or (HERE / "static")

    async def index(request: Request):
        app_state = request.app.state.stat
        return templates.TemplateResponse(request, "index.html", {"protocol": 2, "demo_mode": app_state.config.demo_mode, "jupyter_port": app_state.config.jupyter_port})

    async def not_found_api(request: Request, exc: Exception):
        if request.url.path.startswith("/api/"):
            return error_response("not_found", f"no route for {request.method} {request.url.path}", 404)
        return HTMLResponse("<h1>Not found</h1>", status_code=404)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        import asyncio

        st: AppState = app.state.stat
        st.events.bind_loop(asyncio.get_running_loop())
        yield
        if st.turns and st.turns.active and st.turns.active.task:
            st.turns.active.task.cancel()

    routes = [
        Route("/", index, methods=["GET"]),
        *session.routes(),
        *turns.routes(),
        *data.routes(),
        *rois.routes(),
        *misc.routes(),
    ]
    if static_dir.exists():
        routes.append(Mount("/static", app=NoCacheStatic(directory=str(static_dir)), name="static"))
    middleware = []
    import os

    origins = os.environ.get("STAT_CORS_ORIGINS")
    if origins:
        middleware.append(Middleware(CORSMiddleware, allow_origins=origins.split(","), allow_methods=["*"], allow_headers=["*"]))
    app = Starlette(
        routes=routes,
        middleware=middleware,
        exception_handlers={ApiError: api_error_handler, ValidationError: validation_error_handler, 404: not_found_api, Exception: unhandled_error_handler},
        lifespan=lifespan,
    )
    state = AppState(settings=settings, config=config, registry=registry)
    state.turns = TurnManager(state)
    app.state.stat = state
    return app


__all__ = ["create_app"]
