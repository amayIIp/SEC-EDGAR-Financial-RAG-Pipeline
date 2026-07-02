from __future__ import annotations
import functools      
import time           
import uuid           
from contextlib import asynccontextmanager  
from pathlib import Path                    
from typing import Any, AsyncGenerator, Callable, TypeVar  
import orjson                               
import structlog                            
from src.shared.config import cfg           
from src.shared.models import StageProfile  
log = structlog.get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])
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
        return
    profiles_dir = _ensure_profiles_dir()
    log_file = profiles_dir / f"run_{profile.run_id[:8]}.jsonl"
    line = orjson.dumps(profile.model_dump()) + b"\n"
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
        import asyncio
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                run_id = _get_run_id()
                started_at = StageProfile.now_iso()
                t_start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
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
            return async_wrapper  
        else:
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
            return sync_wrapper  
    return decorator
class ProfilerContext:
    """
    Sync context manager for timing a block of code inside a larger function.
    USAGE:
        with ProfilerContext("rrf_fusion", num_candidates=50) as p:
            fused = rrf_fuse(bm25_hits, vector_hits)
        print(f"RRF took {p.duration_ms:.1f} ms")
    """
    def __init__(self, stage_name: str, **extra_fields: Any) -> None:
        self.stage_name = stage_name
        self.extra_fields = extra_fields
        self.duration_ms: float = 0.0
        self._t_start: float = 0.0
    def __enter__(self) -> "ProfilerContext":
        self._t_start = time.perf_counter()
        return self  
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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
