# tests/unit/test_utils.py
# This module implements unit tests for the utility functions in src/utils.py.
# We test token counting, block context profiling, and decorator timing to guarantee
# that basic operational helper utilities are correct.

from __future__ import annotations # Allow self-referencing type annotations.
import time # Standard library module for measuring sleep and execution time.
import pytest # Testing framework.
from src.utils import count_tokens, profile_time, ProfilerContext # Subjects under test.


def test_count_tokens() -> None:
    """
    Verifies that count_tokens returns a positive token count for a sample text.
    """
    sample_text = "Apple Inc. announced financial results."
    
    # Count the tokens using the helper.
    token_count = count_tokens(sample_text, model_name="gpt-4o-mini")
    
    # Assert that token count is greater than 0.
    assert token_count > 0, "Token count should be positive for non-empty text."

    # Test with an unrecognized model name to verify it falls back to cl100k_base.
    fallback_count = count_tokens(sample_text, model_name="invalid-model-name-for-fallback-testing")
    assert fallback_count > 0, "Fallback token count should also be positive."


def test_profile_time_decorator() -> None:
    """
    Verifies that the profile_time decorator executes the target function and preserves its output.
    """
    # Define a sample function wrapped with the decorator.
    @profile_time
    def mock_operation(x: int, y: int) -> int:
        # Simulate minor calculation work.
        time.sleep(0.01)
        return x + y

    # Call the profiled function.
    result = mock_operation(12, 18)
    
    # Check that return value is preserved correctly.
    assert result == 30, "The decorated function failed to return the expected output."


def test_profiler_context_manager() -> None:
    """
    Verifies that ProfilerContext context manager records timing details correctly.
    """
    # Use the context manager to profile a simple block.
    with ProfilerContext("test_utility_block") as p:
        time.sleep(0.01)
        
    # Check that it executed without error.
    assert p.start_time is not None, "Context manager start time was not recorded."
