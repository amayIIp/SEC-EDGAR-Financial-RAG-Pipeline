# tests/unit/test_shared.py
# This module implements unit tests for shared cross-cutting concerns, specifically
# the logging setup and context binding. We verify that loggers are correctly configured,
# that pretty mode detects environmental overrides, and that query run contexts
# propagate variables to logging.

from __future__ import annotations # Allow self-referencing type annotations.
import os # Standard library module for environment variables.
from unittest.mock import patch # Mocking tools.
import pytest # Testing framework.
import structlog # Structured logging library.
from src.shared.logging_setup import (
    _pretty_mode,
    bind_run_context,
    clear_run_context,
    configure_logging,
    get_logger,
) # Subjects under test.


def test_pretty_mode_detection() -> None:
    """
    Verifies that _pretty_mode correctly detects the STRUCTLOG_PRETTY environment variable.
    """
    # Case 1: Environment variable set to "0" or unset.
    with patch.dict(os.environ, {"STRUCTLOG_PRETTY": "0"}):
        assert not _pretty_mode(), "Pretty mode should be False when STRUCTLOG_PRETTY is 0."
        
    # Case 2: Environment variable set to "1".
    with patch.dict(os.environ, {"STRUCTLOG_PRETTY": "1"}):
        assert _pretty_mode(), "Pretty mode should be True when STRUCTLOG_PRETTY is 1."


def test_logging_configuration() -> None:
    """
    Verifies that configure_logging sets up structlog without raising exceptions.
    """
    # Run the configuration function with default level.
    configure_logging("INFO")
    
    # Retrieve a BoundLogger instance.
    logger = get_logger("test_shared_logger")
    
    # Verify that the logger exposes standard logging methods.
    assert hasattr(logger, "info"), "Returned logger lacks info method."
    assert hasattr(logger, "warning"), "Returned logger lacks warning method."


def test_run_context_propagation() -> None:
    """
    Verifies that bind_run_context successfully propagates variables to the contextvars
    and clear_run_context cleans them up.
    """
    # Clear any leftover contexts first.
    clear_run_context()
    
    # Bind contextvars for a mock query run.
    bind_run_context(run_id="test_run_123", query="Sample user question.")
    
    # Retrieve the thread-local context variables.
    context = structlog.contextvars.get_contextvars()
    
    # Assertions.
    assert context.get("run_id") == "test_run_123"
    assert context.get("query_preview") == "Sample user question."
    
    # Clear context.
    clear_run_context()
    
    # Check that contextvars are now empty.
    empty_context = structlog.contextvars.get_contextvars()
    assert "run_id" not in empty_context, "Run context was not cleared."
