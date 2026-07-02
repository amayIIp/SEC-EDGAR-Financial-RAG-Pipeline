from __future__ import annotations 
import time 
import pytest 
from src.utils import count_tokens, profile_time, ProfilerContext 
def test_count_tokens() -> None:
    """
    Verifies that count_tokens returns a positive token count for a sample text.
    """
    sample_text = "Apple Inc. announced financial results."
    token_count = count_tokens(sample_text, model_name="gpt-4o-mini")
    assert token_count > 0, "Token count should be positive for non-empty text."
    fallback_count = count_tokens(sample_text, model_name="invalid-model-name-for-fallback-testing")
    assert fallback_count > 0, "Fallback token count should also be positive."
def test_profile_time_decorator() -> None:
    """
    Verifies that the profile_time decorator executes the target function and preserves its output.
    """
    @profile_time
    def mock_operation(x: int, y: int) -> int:
        time.sleep(0.01)
        return x + y
    result = mock_operation(12, 18)
    assert result == 30, "The decorated function failed to return the expected output."
def test_profiler_context_manager() -> None:
    """
    Verifies that ProfilerContext context manager records timing details correctly.
    """
    with ProfilerContext("test_utility_block") as p:
        time.sleep(0.01)
    assert p.start_time is not None, "Context manager start time was not recorded."
