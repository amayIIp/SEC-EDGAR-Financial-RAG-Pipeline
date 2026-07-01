# tests/unit/test_rrf.py
# This module implements unit tests for the Reciprocal Rank Fusion (RRF) algorithm.
# We test edge cases (empty search results, matching ranks, ties) to guarantee
# that the fusion scoring arithmetic is correct and stable.

from __future__ import annotations # Allow self-referencing type annotations.
import pytest # Testing framework.
from src.retrieval.rrf_fusion import rrf_fuse # Subject under test.
from src.shared.models import BM25Result, Chunk, ChunkingStrategy, FilingType, VectorResult # Models.

# Define a shared mock chunk helper function.
def _make_mock_chunk(chunk_id: str) -> Chunk:
    """Helper to build a dummy Chunk record for testing."""
    return Chunk(
        chunk_id=chunk_id,
        text="Sample text",
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

def test_rrf_fuse_empty() -> None:
    """
    Verifies that fusing two empty lists returns an empty list.
    """
    fused = rrf_fuse(bm25_results=[], vector_results=[], k=60)
    assert fused == []

def test_rrf_fuse_single_system() -> None:
    """
    Verifies that when only one search system returns a hit, the RRF score is computed correctly.
    RRF Score = 1 / (60 + Rank)
    For Rank 1, RRF Score = 1 / (60 + 1) = 1/61 ≈ 0.016393
    """
    chunk = _make_mock_chunk("chunk_1")
    bm25_hit = BM25Result(chunk=chunk, bm25_score=10.0, bm25_rank=1)
    
    fused = rrf_fuse(bm25_results=[bm25_hit], vector_results=[], k=60)
    
    assert len(fused) == 1
    assert fused[0].chunk.chunk_id == "chunk_1"
    # Verify calculated RRF score matches the formula.
    assert pytest.approx(fused[0].rrf_score) == 1.0 / 61.0
    assert fused[0].bm25_rank == 1
    assert fused[0].vector_rank is None

def test_rrf_fuse_both_systems() -> None:
    """
    Verifies that when a document matches in both systems, their reciprocal ranks accumulate.
    RRF Score = 1 / (60 + Rank_system_1) + 1 / (60 + Rank_system_2)
    For Rank 1 and Rank 2: 1/61 + 1/62 ≈ 0.016393 + 0.016129 ≈ 0.032522
    """
    chunk = _make_mock_chunk("chunk_1")
    bm25_hit = BM25Result(chunk=chunk, bm25_score=10.0, bm25_rank=1)
    vector_hit = VectorResult(chunk=chunk, vector_score=0.85, vector_rank=2)
    
    fused = rrf_fuse(bm25_results=[bm25_hit], vector_results=[vector_hit], k=60)
    
    assert len(fused) == 1
    assert fused[0].chunk.chunk_id == "chunk_1"
    # Verify score accumulation matches formula.
    expected_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert pytest.approx(fused[0].rrf_score) == expected_score
    assert fused[0].bm25_rank == 1
    assert fused[0].vector_rank == 2
