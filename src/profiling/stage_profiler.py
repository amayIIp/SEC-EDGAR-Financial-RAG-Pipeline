# src/profiling/stage_profiler.py
#
# @profile_stage decorator and ProfilerContext manager.
#
# WHY A CUSTOM PROFILER INSTEAD OF cProfile?
# cProfile measures CPU time summed across all calls — great for optimising
# a hot loop, but useless for answering "what fraction of the 2.3-second
# /query request was spent waiting for the Cohere API to respond?"
# We need wall-clock time per named stage, per query, written to a log that
# can be aggregated across a 50-query eval run.
#
# The design here does three things:
#   1. Records wall-clock start/end with time.perf_counter() (nanosecond precision).
#   2. Emits a structured log line (→ structlog) AND writes a JSON-lines record
#      (→ data/profiles/) so latency can be analysed offline.
#   3. Propagates the run_id from the async context so every stage timing for
#      a single query is groupable in the profile logs.
#
# USAGE — as a decorator:
#   @profile_stage("bm25_search")
#   async def bm25_search(query: str, top_k: int) -> list[BM25Result]:
#       ...
#
# USAGE — as a context manager (for timing a block inside a larger function):
#   async with ProfilerContext("rrf_fusion") as p:
#       fused = rrf_fuse(bm25_hits, vector_hits)
#   # p.duration_ms is available after the block exits

from __future__ import annotations

import functools      # standard-library: wraps() preserves function metadata through decoration
import time           # standard-library: perf_counter() for high-resolution wall-clock timing
import uuid           # standard-library: generates unique run identifiers
from contextlib import asynccontextmanager  # standard-library: async context manager helper
from pathlib import Path                    # standard-library: filesystem path operations
from typing import Any, AsyncGenerator, Callable, TypeVar  # type-hint helpers

import orjson                               # fast JSON serialisation for profile log lines
import structlog                            # structured logging (configured in logging_setup.py)

from src.shared.config import cfg           # pipeline config singleton
from src.shared.models import StageProfile  # typed profiling record

# Logger for this module — every profile event is also emitted as a log line.
log = structlog.get_logger(__name__)

# TypeVar allows the decorator to preserve the return type of the wrapped function.
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Profile log writer
# =============================================================================

def _ensure_profiles_dir() -> Path:
    """Creates data/profiles/ if it doesn't exist yet and returns its Path."""
    profiles_dir = Path(cfg.profiling.profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir


def _write_profile_record(profile: StageProfile) -> None:
    """
    Appends one StageProfile record as a JSON line to the active profile log.

    File name: data/profiles/run_{run_id[:8]}.jsonl
    Each line is a complete JSON object so the file can be streamed line-by-line
    without loading it entirely into memory.
    """
    if not cfg.profiling.enabled:
        # Profiling is disabled in config — skip disk writes to save I/O.
        return

    profiles_dir = _ensure_profiles_dir()
    # Truncate run_id to 8 chars for a readable filename while remaining unique enough.
    log_file = profiles_dir / f"run_{profile.run_id[:8]}.jsonl"

    # orjson serialises Pydantic models natively and is ~3× faster than stdlib json.
    line = orjson.dumps(profile.model_dump()) + b"\n"

    # Open in append+binary mode so concurrent writers from asyncio tasks don't truncate.
    with log_file.open("ab") as fh:
        fh.write(line)


def _get_run_id() -> str:
    """
    Retrieves the run_id from the structlog context if one was bound by
    bind_run_context(), otherwise generates a fresh UUID4.

    This means decorator-wrapped functions automatically inherit the run_id
    of the query that called them — no need to thread it through every signature.
    """
    import structlog.contextvars as ctx
    context = ctx.get_contextvars()
    return context.get("run_id", str(uuid.uuid4()))


# =============================================================================
# Decorator — wraps both sync and async functions
# =============================================================================

def profile_stage(stage_name: str, **extra_fields: Any) -> Callable[[F], F]:
    """
    Decorator that times the execution of a pipeline stage function.

    Works on both regular (sync) and async functions.
    Extra keyword arguments are stored in the StageProfile.extra dict,
    allowing callers to annotate timings with metadata:
        @profile_stage("vector_search", provider="openai")

    Args:
        stage_name:   Human-readable name logged with every timing record.
        extra_fields: Optional static key-value pairs stored in the profile.

    Returns:
        A decorator that wraps the target function with timing instrumentation.
    """
    def decorator(func: F) -> F:
        # asyncio.iscoroutinefunction detects whether the wrapped function is
        # declared with `async def` — we need separate wrappers for each.
        import asyncio

        if asyncio.iscoroutinefunction(func):
            # ── Async wrapper ────────────────────────────────────────────────
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                run_id = _get_run_id()
                started_at = StageProfile.now_iso()

                # Record the wall-clock start time with nanosecond resolution.
                t_start = time.perf_counter()
                try:
                    # Await the original async function, passing all arguments through.
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    # perf_counter() gives seconds; multiply by 1000 for milliseconds.
                    duration_ms = (time.perf_counter() - t_start) * 1_000

                    profile = StageProfile(
                        run_id=run_id,
                        stage_name=stage_name,
                        started_at=started_at,
                        duration_ms=round(duration_ms, 3),
                        extra=extra_fields,
                    )

                    # Emit a structured log line — visible in terminal during dev.
                    log.info(
                        "stage_complete",
                        stage=stage_name,
                        duration_ms=profile.duration_ms,
                        run_id=run_id,
                    )

                    # Write to the JSON-lines profile log for offline analysis.
                    _write_profile_record(profile)

            return async_wrapper  # type: ignore[return-value]

        else:
            # ── Sync wrapper ─────────────────────────────────────────────────
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                run_id = _get_run_id()
                started_at = StageProfile.now_iso()
                t_start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration_ms = (time.perf_counter() - t_start) * 1_000
                    profile = StageProfile(
                        run_id=run_id,
                        stage_name=stage_name,
                        started_at=started_at,
                        duration_ms=round(duration_ms, 3),
                        extra=extra_fields,
                    )
                    log.info(
                        "stage_complete",
                        stage=stage_name,
                        duration_ms=profile.duration_ms,
                        run_id=run_id,
                    )
                    _write_profile_record(profile)

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Context manager — times an inline block of code
# =============================================================================

class ProfilerContext:
    """
    Sync context manager for timing a block of code inside a larger function.

    USAGE:
        with ProfilerContext("rrf_fusion", num_candidates=50) as p:
            fused = rrf_fuse(bm25_hits, vector_hits)
        print(f"RRF took {p.duration_ms:.1f} ms")
    """

    def __init__(self, stage_name: str, **extra_fields: Any) -> None:
        # Name of the code block being timed.
        self.stage_name = stage_name
        # Optional static annotations stored alongside the timing record.
        self.extra_fields = extra_fields
        # Will be set to the measured duration after __exit__ is called.
        self.duration_ms: float = 0.0
        # Internal start time; populated in __enter__.
        self._t_start: float = 0.0

    def __enter__(self) -> "ProfilerContext":
        # Record the wall-clock start the moment we enter the 'with' block.
        self._t_start = time.perf_counter()
        return self  # return self so `as p` gives access to .duration_ms later

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # __exit__ is called whether the block completed normally or raised an exception.
        # We still record the timing even if an exception occurred so partial-run
        # profiles are not silently dropped.
        self.duration_ms = (time.perf_counter() - self._t_start) * 1_000

        profile = StageProfile(
            run_id=_get_run_id(),
            stage_name=self.stage_name,
            started_at=StageProfile.now_iso(),
            duration_ms=round(self.duration_ms, 3),
            extra=self.extra_fields,
        )

        log.info(
            "block_complete",
            stage=self.stage_name,
            duration_ms=profile.duration_ms,
        )

        _write_profile_record(profile)

        # Returning None (implicitly) means we do NOT suppress exceptions —
        # any error raised inside the 'with' block still propagates normally.


@asynccontextmanager
async def async_profiler_context(
    stage_name: str, **extra_fields: Any
) -> AsyncGenerator["_AsyncProfileResult", None]:
    """
    Async context manager for timing a block inside an async function.

    USAGE:
        async with async_profiler_context("context_pack") as p:
            packed = await pack_context(chunks)
        # p.duration_ms available here
    """
    result = _AsyncProfileResult(stage_name, extra_fields)
    result._t_start = time.perf_counter()
    try:
        yield result
    finally:
        result.duration_ms = (time.perf_counter() - result._t_start) * 1_000
        profile = StageProfile(
            run_id=_get_run_id(),
            stage_name=stage_name,
            started_at=StageProfile.now_iso(),
            duration_ms=round(result.duration_ms, 3),
            extra=extra_fields,
        )
        log.info("async_block_complete", stage=stage_name, duration_ms=profile.duration_ms)
        _write_profile_record(profile)


class _AsyncProfileResult:
    """Simple result holder for async_profiler_context."""

    def __init__(self, stage_name: str, extra: dict[str, Any]) -> None:
        self.stage_name = stage_name
        self.extra = extra
        self.duration_ms: float = 0.0
        self._t_start: float = 0.0
