"""Centralised FastAPI exception handlers."""

from __future__ import annotations

import logfire
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.exceptions import OpenDeepResearchError


def install_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the given app."""

    @app.exception_handler(OpenDeepResearchError)
    async def handle_domain_error(  # type: ignore[override]
        request: Request,
        exc: OpenDeepResearchError,
    ) -> JSONResponse:
        payload = exc.to_payload() | {"path": request.url.path}
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(  # type: ignore[override]
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logfire.exception("Unhandled error", request=str(request.url))
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "Unexpected server error",
                "details": {},
                "path": request.url.path,
            },
        )


__all__ = ["install_error_handlers"]
