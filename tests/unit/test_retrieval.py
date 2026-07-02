from __future__ import annotations 
from unittest.mock import AsyncMock, MagicMock, patch 
import pytest 
from src.retrieval.bm25_search import bm25_search 
from src.retrieval.hybrid_search import hybrid_search 
from src.retrieval.vector_search import vector_search 
from src.shared.models import BM25Result, Chunk, ChunkingStrategy, FilingType, VectorResult 
def _make_mock_chunk(chunk_id: str) -> Chunk:
    """Helper to build a dummy Chunk record for testing."""
    return Chunk(
        chunk_id=chunk_id,
        text="Sample retrieval text content.",
        strategy=ChunkingStrategy.STRUCTURE_AWARE,
        chunk_index=0,
        is_table=False,
        ticker="AAPL",
        cik="0000320193",
        filing_type=FilingType.TEN_K,
        filing_date="2023-09-30",
        fiscal_year=2023,
        accession_number="0000320193-23-000106",
        section="Item 7",
        section_title="MD&A",
        token_count=10
    )
@pytest.mark.anyio 
@patch("src.retrieval.bm25_search.OpenSearch") 
async def test_bm25_search_mocked(mock_opensearch_class: MagicMock) -> None:
    """
    Verifies that bm25_search constructs the expected DSL query, calls search on OpenSearch,
    and returns parsed BM25Result objects.
    """
    mock_client = MagicMock()
    mock_opensearch_class.return_value = mock_client
    mock_hit = {
        "_score": 12.34,
        "_source": _make_mock_chunk("chunk_1").model_dump()
    }
    mock_client.search.return_value = {
        "hits": {
            "hits": [mock_hit]
        }
    }
    filters = {"ticker": "AAPL", "filing_type": "10-K"}
    results = await bm25_search(query="margins", top_k=5, filters=filters)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk_1"
    assert results[0].bm25_score == 12.34
    assert results[0].bm25_rank == 1
    mock_client.search.assert_called_once()
    called_args = mock_client.search.call_args[1]
    assert called_args["index"] == "sec_chunks_v1"
    assert called_args["body"]["query"]["bool"]["must"][0]["match"]["text"] == "margins"
@pytest.mark.anyio
@patch("src.retrieval.vector_search.SECEmbedder") 
@patch("src.retrieval.vector_search.QdrantClient") 
async def test_vector_search_mocked(mock_qdrant_class: MagicMock, mock_embedder_class: MagicMock) -> None:
    """
    Verifies that vector_search calculates the query embedding, calls search on Qdrant,
    and returns parsed VectorResult objects.
    """
    mock_embedder = MagicMock()
    mock_embedder_class.return_value = mock_embedder
    mock_embedder.embed.return_value = [[0.1] * 1536] 
    mock_client = MagicMock()
    mock_qdrant_class.return_value = mock_client
    mock_scored_point = MagicMock()
    mock_scored_point.score = 0.88
    mock_scored_point.payload = _make_mock_chunk("chunk_2").model_dump()
    mock_client.search.return_value = [mock_scored_point]
    filters = {"ticker": "AAPL"}
    results = await vector_search(query="expenses", top_k=5, filters=filters)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk_2"
    assert results[0].vector_score == 0.88
    assert results[0].vector_rank == 1
    mock_client.search.assert_called_once()
    called_args = mock_client.search.call_args[1]
    assert called_args["limit"] == 5
    assert called_args["query_filter"] is not None
@pytest.mark.anyio
@patch("src.retrieval.hybrid_search.bm25_search", new_callable=AsyncMock) 
@patch("src.retrieval.hybrid_search.vector_search", new_callable=AsyncMock) 
async def test_hybrid_search_routing(mock_vector_search: AsyncMock, mock_bm25_search: AsyncMock) -> None:
    """
    Verifies that hybrid_search routes queries to BM25, vector, or hybrid pathways based on parameters.
    """
    chunk_a = _make_mock_chunk("chunk_a")
    chunk_b = _make_mock_chunk("chunk_b")
    mock_bm25_search.return_value = [BM25Result(chunk=chunk_a, bm25_score=10.0, bm25_rank=1)]
    mock_vector_search.return_value = [VectorResult(chunk=chunk_b, vector_score=0.9, vector_rank=1)]
    results_bm25 = await hybrid_search(query="margins", mode="bm25_only")
    assert len(results_bm25) == 1
    assert results_bm25[0].chunk.chunk_id == "chunk_a"
    assert results_bm25[0].bm25_rank == 1
    assert results_bm25[0].vector_rank is None
    results_vector = await hybrid_search(query="margins", mode="vector_only")
    assert len(results_vector) == 1
    assert results_vector[0].chunk.chunk_id == "chunk_b"
    assert results_vector[0].vector_rank == 1
    assert results_vector[0].bm25_rank is None
