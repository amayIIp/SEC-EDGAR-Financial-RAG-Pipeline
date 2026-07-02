from __future__ import annotations 
from fastapi import FastAPI, Request, status 
from fastapi.responses import JSONResponse 
from src.shared.logging_setup import get_logger 
log = get_logger(__name__)
def register_error_handlers(app: FastAPI) -> None:
    """
    Registers global exception interceptors on the FastAPI application instance.
    """
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        log.warning("api_value_error", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_type": "ValueError",
                "detail": str(exc)
            }
        )
    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        log.error("api_runtime_error", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_type": "RuntimeError",
                "detail": "An internal service error occurred. Downstream APIs might be unreachable."
            }
        )
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
