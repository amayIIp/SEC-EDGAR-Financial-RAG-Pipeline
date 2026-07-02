from __future__ import annotations 
import asyncio 
import time 
from unittest.mock import MagicMock, patch 
import numpy as np 
import pytest 
from src.profiling.cache import ExactMatchCache, SemanticCache, QueryCache 
from src.profiling.stage_profiler import profile_stage, ProfilerContext 
def test_exact_match_cache() -> None:
    """
    Verifies that ExactMatchCache saves, retrieves, and expires exact text matches.
    """
    cache = ExactMatchCache(ttl_seconds=0.1)
    assert cache.get("AAPL revenue") is None
    cache.set("AAPL revenue", "Cached AAPL Answer")
    assert cache.get("AAPL revenue") == "Cached AAPL Answer"
    time.sleep(0.15)
    assert cache.get("AAPL revenue") is None
def test_semantic_cache() -> None:
    """
    Verifies that SemanticCache retrieves cached results for semantically similar query vectors.
    """
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=10.0)
    v1 = np.array([1.0, 0.0, 0.0]) 
    v2 = np.array([0.95, 0.05, 0.0]) 
    v3 = np.array([0.0, 1.0, 0.0]) 
    cache.set("Query A", v1, "Answer A")
    hit = cache.get(v2)
    assert hit == "Answer A", "Semantic cache missed a highly similar query vector."
    miss = cache.get(v3)
    assert miss is None, "Semantic cache incorrectly hit on an orthogonal vector."
@pytest.mark.anyio 
async def test_stage_profiler_decorator() -> None:
    """
    Verifies that the @profile_stage decorator tracks execution time and logs profiles.
    """
    @profile_stage("test_stage")
    async def sample_task(val: int) -> int:
        await asyncio.sleep(0.01)
        return val * 2
    result = await sample_task(5)
    assert result == 10
def test_profiler_context_manager() -> None:
    """
    Verifies that ProfilerContext can be used as a synchronous context manager to track block latency.
    """
    with ProfilerContext("test_context") as p:
        time.sleep(0.01)
    assert p.duration_ms > 0.0, "ProfilerContext duration was not recorded."
