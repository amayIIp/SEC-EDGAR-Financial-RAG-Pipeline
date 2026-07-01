# src/shared/logging_setup.py
#
# Configures structured JSON logging for the entire pipeline.
#
# WHY STRUCTURED LOGGING?
# stdlib logging prints human-readable strings like:
#   "2024-01-15 12:34:56 INFO Indexed 500 chunks"
# That is fine for a developer reading a terminal, but useless for:
#   - Grepping across thousands of log lines from a 50-query eval run
#   - Filtering by stage_name or run_id in a log aggregator (Loki, CloudWatch)
#   - Computing p95 latency from log data when profiling is not available
#
# Structlog emits JSON objects instead:
#   {"timestamp": "...", "level": "info", "stage": "bm25_search",
#    "run_id": "abc123", "duration_ms": 47.3, "num_hits": 50}
# Every field is queryable and the format is consistent across every module.
#
# USAGE:
#   from src.shared.logging_setup import get_logger
#   log = get_logger(__name__)
#   log.info("indexed_chunk", chunk_id="abc", token_count=312)

from __future__ import annotations

import logging        # standard-library: underlying log record infrastructure
import sys            # standard-library: stdout/stderr stream references
from typing import Any

import structlog      # third-party: structured logging with JSON output


def configure_logging(log_level: str = "INFO") -> None:
    """
    Call once at application startup (in main.py, run_*.py entry-points, and
    conftest.py for tests) to install the JSON log renderer pipeline.

    Calling this multiple times is safe — structlog.configure() is idempotent.
    """

    # Convert the string level name to the stdlib integer constant.
    # e.g. "INFO" → logging.INFO (20), "DEBUG" → logging.DEBUG (10).
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure the stdlib root logger to accept records at the chosen level
    # and route them to stdout (so Docker / systemd can capture them easily).
    logging.basicConfig(
        format="%(message)s",   # structlog renders the message; stdlib just passes it through
        stream=sys.stdout,
        level=numeric_level,
    )

    # Suppress noisy third-party loggers that would otherwise flood the output.
    for noisy_logger in [
        "urllib3",           # HTTP connection pool chatter from requests
        "httpx",             # async HTTP client verbose output
        "opensearchpy",      # low-level OpenSearch wire-protocol logs
        "httpcore",          # httpx's underlying HTTP/1.1 engine
        "sentence_transformers",  # model-load progress messages
        "transformers",      # HuggingFace tokeniser load messages
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Install structlog's processing pipeline.
    # Each processor in the chain transforms the log event dict before the
    # final renderer serialises it.
    structlog.configure(
        processors=[
            # Merge any context variables bound with structlog.contextvars.bind_contextvars()
            # into every log event on this thread.  This is how run_id propagates
            # automatically to every log line within a single query handler.
            structlog.contextvars.merge_contextvars,

            # Add the log level string ("info", "warning", etc.) to the event dict.
            structlog.stdlib.add_log_level,

            # Add an ISO-8601 UTC timestamp to every log event.
            structlog.processors.TimeStamper(fmt="iso", utc=True),

            # Add the caller's module and line number (useful for debugging).
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),

            # If an exception is being logged (exc_info=True), format the
            # traceback as a single-line string so it stays within the JSON object.
            structlog.processors.ExceptionRenderer(),

            # Render the final event dict as a compact JSON string on one line.
            # In production, pipe this to a log aggregator.
            # In development, set STRUCTLOG_PRETTY=1 to get coloured output instead.
            structlog.processors.JSONRenderer()
            if not _pretty_mode()
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        # Use stdlib logging as the underlying transport so third-party libraries
        # that call logging.getLogger() also get structured output.
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
    # structlog.get_logger() returns a proxy that routes calls through the
    # configured processor chain defined in configure_logging() above.
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
    # structlog.contextvars stores context in a ContextVar that is isolated
    # per asyncio Task, so concurrent requests don't bleed into each other.
    context: dict[str, Any] = {"run_id": run_id}
    if query is not None:
        # Truncate the query to 80 characters so log lines don't overflow.
        context["query_preview"] = query[:80]
    context.update(extra)
    structlog.contextvars.bind_contextvars(**context)


def clear_run_context() -> None:
    """Clears all context variables bound by bind_run_context()."""
    structlog.contextvars.clear_contextvars()
