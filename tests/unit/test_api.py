from __future__ import annotations 
import os
os.environ["COHERE_API_KEY"] = "mock_cohere_key"
os.environ["OPENAI_API_KEY"] = "mock_openai_key"
os.environ["ANTHROPIC_API_KEY"] = "mock_anthropic_key"
from unittest.mock import AsyncMock, MagicMock, patch 
from fastapi.testclient import TestClient 
import pytest 
from src.api.main import app 
import src.app 
client = TestClient(app, raise_server_exceptions=False)
def test_health_endpoint() -> None:
    """
    Verifies that GET /health returns a 200 OK status code and correct JSON payload.
    """
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["service"] == "SEC RAG Pipeline"
@patch("src.api.main.runner.ingest") 
@patch("src.ingestion.run_ingest._resolve_cik") 
def test_ingest_endpoint(mock_resolve_cik: MagicMock, mock_ingest: MagicMock) -> None:
    """
    Verifies that POST /ingest registers the background task and responds with 202.
    """
    mock_resolve_cik.return_value = "0000320193"
    payload = {
        "ticker": "AAPL",
        "forms": ["10-K"],
        "limit": 2,
        "embedding_provider": "bge"
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 202, f"Expected 202, got {response.status_code}"
    json_data = response.json()
    assert json_data["status"] == "queued"
    assert "AAPL" in json_data["message"]
@pytest.mark.anyio 
@patch("src.api.main.runner.run_pipeline", new_callable=AsyncMock) 
async def test_query_endpoint(mock_run_pipeline: AsyncMock) -> None:
    """
    Verifies that POST /query maps parameters correctly, calls runner, and returns QueryResponse.
    """
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
    payload = {
        "query": "What is the R&D expenditure?",
        "mode": "hybrid",
        "top_k": 10,
        "top_n": 3
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
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
    payload = {
        "query": "Verify retrieval details.",
        "mode": "hybrid"
    }
    response = client.post("/debug", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    json_data = response.json()
    assert json_data["answer"] == "Debug answer."
    assert len(json_data["bm25_hits"]) == 1
    assert json_data["bm25_hits"][0]["text"] == "BM25 text"
    assert len(json_data["final_context"]) == 1
@patch("src.ingestion.run_ingest._resolve_cik") 
def test_ingest_endpoint_value_error(mock_resolve_cik: MagicMock) -> None:
    """
    Verifies that when CIK resolution fails, a ValueError is raised,
    and the API handles it returning 400 Bad Request.
    """
    mock_resolve_cik.return_value = None
    payload = {
        "ticker": "INVALID",
        "forms": ["10-K"]
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 400
    json_data = response.json()
    assert json_data["error_type"] == "ValueError"
@pytest.mark.anyio
@patch("src.api.main.runner.run_pipeline", new_callable=AsyncMock)
async def test_query_endpoint_runtime_error(mock_run_pipeline: AsyncMock) -> None:
    """
    Verifies that when the pipeline raises a RuntimeError, the API handles it returning 500.
    """
    mock_run_pipeline.side_effect = RuntimeError("OpenAI API key expired")
    payload = {"query": "Test query"}
    response = client.post("/query", json=payload)
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
    mock_run_pipeline.side_effect = Exception("Unexpected database connection crash")
    payload = {"query": "Test query"}
    response = client.post("/query", json=payload)
    assert response.status_code == 500
    json_data = response.json()
    assert json_data["error_type"] == "UnhandledException"
    assert "unexpected error" in json_data["detail"]
