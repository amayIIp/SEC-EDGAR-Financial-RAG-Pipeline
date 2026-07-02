from __future__ import annotations 
import os 
from unittest.mock import patch 
import pytest 
import structlog 
from src.shared.logging_setup import (
    _pretty_mode,
    bind_run_context,
    clear_run_context,
    configure_logging,
    get_logger,
) 
def test_pretty_mode_detection() -> None:
    """
    Verifies that _pretty_mode correctly detects the STRUCTLOG_PRETTY environment variable.
    """
    with patch.dict(os.environ, {"STRUCTLOG_PRETTY": "0"}):
        assert not _pretty_mode(), "Pretty mode should be False when STRUCTLOG_PRETTY is 0."
    with patch.dict(os.environ, {"STRUCTLOG_PRETTY": "1"}):
        assert _pretty_mode(), "Pretty mode should be True when STRUCTLOG_PRETTY is 1."
def test_logging_configuration() -> None:
    """
    Verifies that configure_logging sets up structlog without raising exceptions.
    """
    configure_logging("INFO")
    logger = get_logger("test_shared_logger")
    assert hasattr(logger, "info"), "Returned logger lacks info method."
    assert hasattr(logger, "warning"), "Returned logger lacks warning method."
def test_run_context_propagation() -> None:
    """
    Verifies that bind_run_context successfully propagates variables to the contextvars
    and clear_run_context cleans them up.
    """
    clear_run_context()
    bind_run_context(run_id="test_run_123", query="Sample user question.")
    context = structlog.contextvars.get_contextvars()
    assert context.get("run_id") == "test_run_123"
    assert context.get("query_preview") == "Sample user question."
    clear_run_context()
    empty_context = structlog.contextvars.get_contextvars()
    assert "run_id" not in empty_context, "Run context was not cleared."
