from __future__ import annotations 
from unittest.mock import MagicMock, patch 
import pytest 
from src.reranking.bge_reranker import BGEReranker 
from src.reranking.cohere_reranker import CohereReranker 
from src.shared.models import BM25Result, Chunk, ChunkingStrategy, FilingType, FusedResult 
def _make_mock_chunk(chunk_id: str, text: str) -> Chunk:
    """Helper to build a dummy Chunk record for testing."""
    return Chunk(
        chunk_id=chunk_id, 
        text=text, 
        strategy=ChunkingStrategy.NAIVE_FIXED,
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
@patch("src.reranking.cohere_reranker.cohere.Client") 
def test_cohere_reranker_success(mock_cohere_client_class: MagicMock) -> None:
    """
    Verifies that CohereReranker calls the cloud API and maps the response into RankedResult models.
    """
    mock_client = MagicMock()
    mock_cohere_client_class.return_value = mock_client
    mock_response = MagicMock()
    result_a = MagicMock()
    result_a.index = 0 
    result_a.relevance_score = 0.95 
    result_b = MagicMock()
    result_b.index = 1 
    result_b.relevance_score = 0.85 
    mock_response.results = [result_a, result_b]
    mock_client.rerank.return_value = mock_response
    reranker = CohereReranker()
    chunk_a = _make_mock_chunk("chunk_1", "Operating margin increased.")
    chunk_b = _make_mock_chunk("chunk_2", "R&D expenditures grew.")
    candidates = [
        FusedResult(chunk=chunk_a, rrf_score=0.1, bm25_rank=1, vector_rank=2),
        FusedResult(chunk=chunk_b, rrf_score=0.08, bm25_rank=2, vector_rank=3)
    ]
    ranked = reranker.rerank(query="margins", candidates=candidates, top_n=2)
    assert len(ranked) == 2, "Reranked results count mismatch."
    assert ranked[0].chunk.chunk_id == "chunk_1"
    assert ranked[0].rerank_score == 0.95
    assert ranked[0].final_rank == 1
    assert ranked[1].chunk.chunk_id == "chunk_2"
    assert ranked[1].rerank_score == 0.85
    assert ranked[1].final_rank == 2
@patch("src.reranking.cohere_reranker.cohere.Client")
def test_cohere_reranker_failure_fallback(mock_cohere_client_class: MagicMock) -> None:
    """
    Verifies that CohereReranker falls back to original RRF order if the API fails.
    """
    mock_client = MagicMock()
    mock_cohere_client_class.return_value = mock_client
    mock_client.rerank.side_effect = Exception("API rate limit exceeded")
    reranker = CohereReranker()
    chunk_a = _make_mock_chunk("chunk_1", "Text A")
    chunk_b = _make_mock_chunk("chunk_2", "Text B")
    candidates = [
        FusedResult(chunk=chunk_a, rrf_score=0.1, bm25_rank=1, vector_rank=2),
        FusedResult(chunk=chunk_b, rrf_score=0.08, bm25_rank=2, vector_rank=3)
    ]
    ranked = reranker.rerank(query="margins", candidates=candidates, top_n=2)
    assert len(ranked) == 2
    assert ranked[0].chunk.chunk_id == "chunk_1"
    assert ranked[0].rerank_score == 0.0
    assert ranked[0].final_rank == 1
@patch("src.reranking.bge_reranker.CrossEncoder") 
def test_bge_reranker_success(mock_cross_encoder_class: MagicMock) -> None:
    """
    Verifies that BGEReranker runs the cross-encoder locally and returns sorted ranked results.
    """
    mock_model = MagicMock()
    mock_cross_encoder_class.return_value = mock_model
    mock_model.predict.return_value = [0.12, 0.98] 
    reranker = BGEReranker()
    chunk_a = _make_mock_chunk("chunk_1", "Low relevance text.")
    chunk_b = _make_mock_chunk("chunk_2", "High relevance text.")
    candidates = [
        FusedResult(chunk=chunk_a, rrf_score=0.1, bm25_rank=1, vector_rank=2),
        FusedResult(chunk=chunk_b, rrf_score=0.08, bm25_rank=2, vector_rank=3)
    ]
    ranked = reranker.rerank(query="high", candidates=candidates, top_n=2)
    assert len(ranked) == 2
    assert ranked[0].chunk.chunk_id == "chunk_2", "Highest scored chunk was not ranked first."
    assert ranked[0].rerank_score == 0.98
    assert ranked[0].final_rank == 1
    assert ranked[1].chunk.chunk_id == "chunk_1"
    assert ranked[1].rerank_score == 0.12
    assert ranked[1].final_rank == 2
@patch("src.reranking.bge_reranker.CrossEncoder")
def test_bge_reranker_failure_fallback(mock_cross_encoder_class: MagicMock) -> None:
    """
    Verifies that BGEReranker falls back to original RRF order if local prediction fails.
    """
    mock_model = MagicMock()
    mock_cross_encoder_class.return_value = mock_model
    mock_model.predict.side_effect = Exception("CUDA Out Of Memory")
    reranker = BGEReranker()
    chunk_a = _make_mock_chunk("chunk_1", "Text A")
    chunk_b = _make_mock_chunk("chunk_2", "Text B")
    candidates = [
        FusedResult(chunk=chunk_a, rrf_score=0.1, bm25_rank=1, vector_rank=2),
        FusedResult(chunk=chunk_b, rrf_score=0.08, bm25_rank=2, vector_rank=3)
    ]
    ranked = reranker.rerank(query="test", candidates=candidates, top_n=2)
    assert len(ranked) == 2
    assert ranked[0].chunk.chunk_id == "chunk_1"
    assert ranked[0].rerank_score == 0.0
    assert ranked[0].final_rank == 1
