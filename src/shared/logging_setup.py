from __future__ import annotations
import logging        
import sys            
from typing import Any
import structlog      
def configure_logging(log_level: str = "INFO") -> None:
    """
    Call once at application startup (in main.py, run_*.py entry-points, and
    conftest.py for tests) to install the JSON log renderer pipeline.
    Calling this multiple times is safe — structlog.configure() is idempotent.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",   
        stream=sys.stdout,
        level=numeric_level,
    )
    for noisy_logger in [
        "urllib3",           
        "httpx",             
        "opensearchpy",      
        "httpcore",          
        "sentence_transformers",  
        "transformers",      
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer()
            if not _pretty_mode()
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
def _pretty_mode() -> bool:
    """Returns True when the STRUCTLOG_PRETTY env var is set to '1'."""
    import os
    return os.environ.get("STRUCTLOG_PRETTY", "0") == "1"
def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """
    Returns a structlog BoundLogger for the given module name.
    USAGE:
        log = get_logger(__name__)
        log.info("chunk_indexed", chunk_id="abc", tokens=312)
        log.warning("parse_warning", filing="AAPL_10-K_2023.html", reason="no body tag")
        log.error("api_call_failed", provider="cohere", status_code=429)
    """
    return structlog.get_logger(name)
def bind_run_context(run_id: str, query: str | None = None, **extra: Any) -> None:
    """
    Binds key-value pairs to the current async context so they appear in every
    log line emitted during a single pipeline run, without passing them manually
    to every function.
    Call this at the start of each query handler in pipeline_runner.py:
        bind_run_context(run_id=run_id, query=query[:80])
    Clear context after the run completes:
        structlog.contextvars.clear_contextvars()
    """
    context: dict[str, Any] = {"run_id": run_id}
    if query is not None:
        context["query_preview"] = query[:80]
    context.update(extra)
    structlog.contextvars.bind_contextvars(**context)
def clear_run_context() -> None:
    """Clears all context variables bound by bind_run_context()."""
    structlog.contextvars.clear_contextvars()
