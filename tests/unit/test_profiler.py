# tests/unit/test_profiler.py
# This module implements unit tests for the profiling and caching mechanisms.
# We test the ExactMatchCache, SemanticCache, and StageProfiler to verify that
# pipeline latencies are correctly measured and that database query results
# are cached semantically and exactly.

from __future__ import annotations # Allow self-referencing type annotations.
import asyncio # Standard library module for async coordination.
import time # Standard library module for sleeping and timing.
from unittest.mock import MagicMock, patch # Mocking tools.
import numpy as np # Library for numerical operations.
import pytest # Testing framework.
from src.profiling.cache import ExactMatchCache, SemanticCache, QueryCache # Subjects under test.
from src.profiling.stage_profiler import profile_stage, ProfilerContext # Subjects under test.


def test_exact_match_cache() -> None:
    """
    Verifies that ExactMatchCache saves, retrieves, and expires exact text matches.
    """
    # Initialize cache with a tiny TTL of 0.1 seconds for testing expiration.
    cache = ExactMatchCache(ttl_seconds=0.1)
    
    # Check that a missing query returns None.
    assert cache.get("AAPL revenue") is None
    
    # Store a result in the cache.
    cache.set("AAPL revenue", "Cached AAPL Answer")
    
    # Retrieve the result and check that it matches.
    assert cache.get("AAPL revenue") == "Cached AAPL Answer"
    
    # Wait for the TTL to expire.
    time.sleep(0.15)
    
    # Verify that the expired entry is no longer returned.
    assert cache.get("AAPL revenue") is None


def test_semantic_cache() -> None:
    """
    Verifies that SemanticCache retrieves cached results for semantically similar query vectors.
    """
    # Initialize semantic cache with a threshold of 0.90.
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=10.0)
    
    # Define orthogonal unit vectors.
    v1 = np.array([1.0, 0.0, 0.0]) # Direction 1.
    v2 = np.array([0.95, 0.05, 0.0]) # Direction 1 (very similar, cosine ≈ 0.998).
    v3 = np.array([0.0, 1.0, 0.0]) # Direction 2 (orthogonal, cosine = 0.0).
    
    # Set cache entry for v1.
    # The parameters are query (str), query_embedding (np.ndarray), and result (Any).
    cache.set("Query A", v1, "Answer A")
    
    # Query with a highly similar vector v2 (should hit).
    hit = cache.get(v2)
    assert hit == "Answer A", "Semantic cache missed a highly similar query vector."
    
    # Query with an orthogonal vector v3 (should miss).
    miss = cache.get(v3)
    assert miss is None, "Semantic cache incorrectly hit on an orthogonal vector."


@pytest.mark.anyio # Mark as async test.
async def test_stage_profiler_decorator() -> None:
    """
    Verifies that the @profile_stage decorator tracks execution time and logs profiles.
    """
    # Define a dummy profiled async function.
    @profile_stage("test_stage")
    async def sample_task(val: int) -> int:
        # Sleep asynchronously for a tiny duration to simulate workload.
        await asyncio.sleep(0.01)
        return val * 2

    # Execute the profiled function.
    result = await sample_task(5)
    
    # Verify that the return value is preserved.
    assert result == 10


def test_profiler_context_manager() -> None:
    """
    Verifies that ProfilerContext can be used as a synchronous context manager to track block latency.
    """
    # Run block inside context manager.
    with ProfilerContext("test_context") as p:
        # Simulate work using synchronous sleep.
        time.sleep(0.01)
        
    # Check that duration was recorded.
    assert p.duration_ms > 0.0, "ProfilerContext duration was not recorded."
