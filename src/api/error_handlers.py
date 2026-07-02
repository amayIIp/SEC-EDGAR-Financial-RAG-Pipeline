# src/api/error_handlers.py
# This module registers custom exception handlers for our FastAPI app.
# When an exception is raised (e.g. a database timeout or a missing API key),
# these handlers intercept it and return a standardized, clean JSON response
# with appropriate HTTP status codes, preventing server stack trace leaks.

from __future__ import annotations # Allow self-referencing type annotations.
from fastapi import FastAPI, Request, status # FastAPI web routing.
from fastapi.responses import JSONResponse # HTTP JSON response client.
from src.shared.logging_setup import get_logger # Logger.

log = get_logger(__name__)

def register_error_handlers(app: FastAPI) -> None:
    """
    Registers global exception interceptors on the FastAPI application instance.
    """

    # Intercept standard ValueError exceptions.
    # Typically raised on invalid model providers, missing keys, or query params anomalies.
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        log.warning("api_value_error", path=request.url.path, error=str(exc))
        # Return a 400 Bad Request client error.
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_type": "ValueError",
                "detail": str(exc)
            }
        )

    # Intercept generic system RuntimeErrors.
    # Typically raised when downstream APIs exhaust all retries.
    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        log.error("api_runtime_error", path=request.url.path, error=str(exc), exc_info=True)
        # Return a 500 Internal Server Error.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_type": "RuntimeError",
                "detail": "An internal service error occurred. Downstream APIs might be unreachable."
            }
        )

    # Catch-all handler for any unhandled Python exceptions.
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error("api_unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_type": "UnhandledException",
                "detail": "An unexpected error occurred on the server. Check logs for details."
            }
        )
