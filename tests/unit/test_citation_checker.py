# tests/unit/test_citation_checker.py
# This module implements unit tests for the citation checker utility.
# We test typical cases, hallucinated/out-of-bounds citation tags, and empty text
# to ensure our citation validation and context utilization math is correct.

from __future__ import annotations # Allow self-referencing type annotations.
import pytest # Testing framework.
from src.generation.citation_checker import check_citations # Subject under test.
from src.shared.models import Chunk, ChunkingStrategy, FilingType, CitedChunk, RankedResult # Models.

# Define mock chunk builder.
def _make_mock_cited_chunk(idx: int) -> CitedChunk:
    """Helper to build a dummy CitedChunk with sequential citation tag."""
    chunk = Chunk(
        chunk_id=f"chunk_{idx}",
        text=f"Sample text block {idx}",
        strategy=ChunkingStrategy.NAIVE_FIXED,
        chunk_index=idx - 1,
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
    ranked = RankedResult(
        chunk=chunk,
        rerank_score=0.9,
        final_rank=idx
    )
    return CitedChunk(
        ranked_result=ranked,
        citation_tag=f"[{idx}]",
        packed_text=chunk.text,
        packed_token_count=10
    )

def test_check_citations_correct() -> None:
    """
    Verifies that valid citation tags are successfully parsed and ratio is correct.
    """
    # Create 3 dummy chunks.
    chunks = [_make_mock_cited_chunk(i) for i in range(1, 4)]
    
    # Generated answer citing chunks 1 and 3.
    answer = "Apple's revenue grew [1], while its margins remained stable [3]."
    
    cited_tags, ratio = check_citations(answer, chunks)
    
    # Check return values.
    assert cited_tags == ["[1]", "[3]"]
    # 2 out of 3 chunks were cited, so ratio is 2/3 ≈ 0.6666
    assert pytest.approx(ratio) == 2.0 / 3.0

def test_check_citations_hallucinated() -> None:
    """
    Verifies that out-of-bounds citations (hallucinated tags) are ignored.
    """
    chunks = [_make_mock_cited_chunk(i) for i in range(1, 3)] # Only 2 chunks provided.
    
    # Answer cites [1] and [9] (out of bounds).
    answer = "The net sales grew [1], which was unexpected [9]."
    
    cited_tags, ratio = check_citations(answer, chunks)
    
    # Check that [9] was ignored since it exceeds the context count.
    assert cited_tags == ["[1]"]
    # Only 1 out of 2 valid chunks was cited, so ratio is 1/2 = 0.5.
    assert ratio == 0.5

def test_check_citations_none() -> None:
    """
    Verifies that answers without citation brackets return empty results and 0.0 ratio.
    """
    chunks = [_make_mock_cited_chunk(i) for i in range(1, 4)]
    answer = "This is a statement with no citation brackets."
    
    cited_tags, ratio = check_citations(answer, chunks)
    
    assert cited_tags == []
    assert ratio == 0.0
