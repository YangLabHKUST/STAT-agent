"""JSON error envelope for every API error."""

from __future__ import annotations

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from stat_agent.events.encoding import to_jsonable

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, detail: Any = None, headers: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.status, self.detail, self.headers = code, message, status, detail, headers or {}


def error_response(code: str, message: str, status: int = 400, detail: Any = None, headers: dict | None = None) -> JSONResponse:
    body = {"success": False, "error": {"code": code, "message": message}}
    if detail is not None:
        body["error"]["detail"] = to_jsonable(detail)
    # legacy flat field for the old frontend
    body["error_message"] = message
    return JSONResponse(body, status_code=status, headers=headers or {})


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return error_response(exc.code, exc.message, exc.status, exc.detail, exc.headers)


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    errors = getattr(exc, "errors", None)
    return error_response("validation", "invalid request", 400, errors() if callable(errors) else str(exc))


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return error_response("internal", f"{type(exc).__name__}: {exc}", 500)


__all__ = ["ApiError", "error_response", "api_error_handler", "validation_error_handler", "unhandled_error_handler"]
