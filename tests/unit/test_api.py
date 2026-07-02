# tests/unit/test_api.py
# This module implements unit tests for the FastAPI web endpoints.
# We use FastAPI's TestClient to verify the HTTP endpoints healthcheck,
# ingestion queuing, search queries, and debug route parameters, using
# mocks to decouple from live external API services.

from __future__ import annotations # Allow self-referencing type annotations.

# Pre-set mock API keys in the environment before importing modules to bypass key validation checks.
import os
os.environ["COHERE_API_KEY"] = "mock_cohere_key"
os.environ["OPENAI_API_KEY"] = "mock_openai_key"
os.environ["ANTHROPIC_API_KEY"] = "mock_anthropic_key"

from unittest.mock import AsyncMock, MagicMock, patch # Mocking tools.
from fastapi.testclient import TestClient # FastAPI client for HTTP testing.
import pytest # Testing framework.
from src.api.main import app # The FastAPI application under test.
import src.app # Import src.app so its root entrypoint is covered by tests.


# Create a TestClient wrapper around the FastAPI app.
# Set raise_server_exceptions=False so the custom global error handlers are invoked
# and return HTTP responses instead of propagating the exception to the test framework.
client = TestClient(app, raise_server_exceptions=False)


def test_health_endpoint() -> None:
    """
    Verifies that GET /health returns a 200 OK status code and correct JSON payload.
    """
    # Send a GET request to the healthcheck endpoint.
    response = client.get("/health")
    
    # Check that status code is 200 (Success).
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    # Verify the response JSON data.
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["service"] == "SEC RAG Pipeline"


@patch("src.api.main.runner.ingest") # Mock the ingest background task.
@patch("src.ingestion.run_ingest._resolve_cik") # Mock the CIK lookup logic.
def test_ingest_endpoint(mock_resolve_cik: MagicMock, mock_ingest: MagicMock) -> None:
    """
    Verifies that POST /ingest registers the background task and responds with 202.
    """
    # Mock CIK lookup to return a dummy string.
    mock_resolve_cik.return_value = "0000320193"
    
    # Define request payload.
    payload = {
        "ticker": "AAPL",
        "forms": ["10-K"],
        "limit": 2,
        "embedding_provider": "bge"
    }
    
    # Send POST request to /ingest.
    response = client.post("/ingest", json=payload)
    
    # Verify status code is 202 Accepted.
    assert response.status_code == 202, f"Expected 202, got {response.status_code}"
    # Verify the queued status response message.
    json_data = response.json()
    assert json_data["status"] == "queued"
    assert "AAPL" in json_data["message"]


@pytest.mark.anyio # Enable anyio for async test execution.
@patch("src.api.main.runner.run_pipeline", new_callable=AsyncMock) # Mock the async pipeline run method.
async def test_query_endpoint(mock_run_pipeline: AsyncMock) -> None:
    """
    Verifies that POST /query maps parameters correctly, calls runner, and returns QueryResponse.
    """
    # Mock the return value of the pipeline runner.
    mock_run_pipeline.return_value = {
        "answer": "This is a mock answer citing [1].",
        "citations": [
            {
                "chunk_id": "chunk_1",
                "citation_tag": "[1]",
                "text_snippet": "Relevant corporate content.",
                "ticker": "AAPL",
                "form": "10-K",
                "filing_date": "2023-09-30",
                "section": "Item 7"
            }
        ],
        "latency_breakdown": {
            "retrieval": 12.5,
            "generation": 145.0
        },
        "total_tokens": 150
    }
    
    # Define query request body.
    payload = {
        "query": "What is the R&D expenditure?",
        "mode": "hybrid",
        "top_k": 10,
        "top_n": 3
    }
    
    # Send POST request to /query.
    response = client.post("/query", json=payload)
    
    # Verify status code is 200 OK.
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Verify JSON response contains answer and citations.
    json_data = response.json()
    assert "mock answer" in json_data["answer"]
    assert len(json_data["citations"]) == 1
    assert json_data["citations"][0]["chunk_id"] == "chunk_1"
    assert json_data["total_tokens"] == 150


@pytest.mark.anyio
@patch("src.api.main.runner.run_pipeline", new_callable=AsyncMock)
async def test_debug_endpoint(mock_run_pipeline: AsyncMock) -> None:
    """
    Verifies that POST /debug routes requests and returns a verbose DebugResponse structure.
    """
    # Mock debug response payload containing the raw_state intermediate variables.
    mock_run_pipeline.return_value = {
        "answer": "Debug answer.",
        "latency_breakdown": {"total": 500.0},
        "raw_state": {
            "bm25_hits": [{"id": "b1", "text": "BM25 text"}],
            "vector_hits": [{"id": "v1", "text": "Vector text"}],
            "fused_hits": [{"id": "f1", "text": "Fused text"}],
            "reranked_hits": [{"id": "r1", "text": "Reranked text"}],
            "final_context": [{"id": "c1", "text": "Context text"}]
        }
    }
    
    # Define payload.
    payload = {
        "query": "Verify retrieval details.",
        "mode": "hybrid"
    }
    
    # Send POST request to /debug.
    response = client.post("/debug", json=payload)
    
    # Verify status code is 200 OK.
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Verify JSON data has debug fields.
    json_data = response.json()
    assert json_data["answer"] == "Debug answer."
    assert len(json_data["bm25_hits"]) == 1
    assert json_data["bm25_hits"][0]["text"] == "BM25 text"
    assert len(json_data["final_context"]) == 1


@patch("src.ingestion.run_ingest._resolve_cik") # Mock the CIK lookup logic.
def test_ingest_endpoint_value_error(mock_resolve_cik: MagicMock) -> None:
    """
    Verifies that when CIK resolution fails, a ValueError is raised,
    and the API handles it returning 400 Bad Request.
    """
    # Mock CIK lookup to return None (meaning ticker not found).
    mock_resolve_cik.return_value = None
    
    # Define payload.
    payload = {
        "ticker": "INVALID",
        "forms": ["10-K"]
    }
    
    # Send POST request to /ingest.
    response = client.post("/ingest", json=payload)
    
    # Verify that the value error handler caught the exception and returned 400 Bad Request.
    assert response.status_code == 400
    json_data = response.json()
    assert json_data["error_type"] == "ValueError"


@pytest.mark.anyio
@patch("src.api.main.runner.run_pipeline", new_callable=AsyncMock)
async def test_query_endpoint_runtime_error(mock_run_pipeline: AsyncMock) -> None:
    """
    Verifies that when the pipeline raises a RuntimeError, the API handles it returning 500.
    """
    # Mock the pipeline runner to raise a RuntimeError.
    mock_run_pipeline.side_effect = RuntimeError("OpenAI API key expired")
    
    # Define payload.
    payload = {"query": "Test query"}
    
    # Send POST request.
    response = client.post("/query", json=payload)
    
    # Verify that the runtime error handler returned 500.
    assert response.status_code == 500
    json_data = response.json()
    assert json_data["error_type"] == "RuntimeError"
    assert "internal service error" in json_data["detail"]


@pytest.mark.anyio
@patch("src.api.main.runner.run_pipeline", new_callable=AsyncMock)
async def test_query_endpoint_unhandled_exception(mock_run_pipeline: AsyncMock) -> None:
    """
    Verifies that when the pipeline raises a generic exception, the API handles it returning 500.
    """
    # Mock the pipeline runner to raise a generic Exception.
    mock_run_pipeline.side_effect = Exception("Unexpected database connection crash")
    
    # Define payload.
    payload = {"query": "Test query"}
    
    # Send POST request.
    response = client.post("/query", json=payload)
    
    # Verify that the global exception handler returned 500.
    assert response.status_code == 500
    json_data = response.json()
    assert json_data["error_type"] == "UnhandledException"
    assert "unexpected error" in json_data["detail"]
