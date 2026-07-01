# tests/unit/test_generation.py
# This module implements unit tests for context-packing, context compression,
# and citation checking logic. We verify that these components deduplicate corporate data,
# tag citations sequentially, check for hallucinations, and handle mock LLM compression.

from __future__ import annotations # Allow self-referencing type annotations.
from unittest.mock import MagicMock, patch # Mocking tools.
import pytest # Testing framework.
from src.generation.citation_checker import check_citations # Subject under test.
from src.generation.context_compressor import ContextCompressor # Subject under test.
from src.generation.context_packer import compute_jaccard_similarity, pack_context # Subjects under test.
from src.shared.models import Chunk, ChunkingStrategy, FilingType, RankedResult, CitedChunk # Models.


# Create a shared mock helper function.
def _make_mock_ranked_result(chunk_id: str, text: str) -> RankedResult:
    """Helper to build a dummy RankedResult record for testing."""
    chunk = Chunk(
        chunk_id=chunk_id,
        text=text,
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
    return RankedResult(
        chunk=chunk,
        rerank_score=0.9,
        final_rank=1,
        rrf_score=0.08
    )


def test_jaccard_similarity() -> None:
    """
    Verifies that compute_jaccard_similarity correctly measures the overlap of word sets.
    """
    # Test completely identical text.
    similarity_equal = compute_jaccard_similarity("apple banana cherry", "apple banana cherry")
    assert similarity_equal == 1.0, "Identical texts should have 1.0 similarity."
    
    # Test completely disjoint text.
    similarity_disjoint = compute_jaccard_similarity("apple banana", "cherry durian")
    assert similarity_disjoint == 0.0, "Disjoint texts should have 0.0 similarity."
    
    # Test partial overlap (1 shared word out of 3 total words).
    similarity_partial = compute_jaccard_similarity("apple banana", "banana cherry")
    # Union is {apple, banana, cherry} (size 3); Intersection is {banana} (size 1).
    assert similarity_partial == 1.0 / 3.0, "Partial overlap Jaccard score mismatch."


def test_pack_context_dedup() -> None:
    """
    Verifies that pack_context drops near-duplicate chunks and assigns citation tags.
    """
    # Create two identical chunks and one unique chunk.
    # Chunk A and Chunk B are duplicates (identical texts).
    # Chunk C is unique.
    item_a = _make_mock_ranked_result("chunk_a", "Company revenue grew due to cloud services.")
    item_b = _make_mock_ranked_result("chunk_b", "Company revenue grew due to cloud services.")
    item_c = _make_mock_ranked_result("chunk_c", "Risk factors include inflation and interest rate fluctuations.")
    
    candidates = [item_a, item_b, item_c]
    
    # Run context packing.
    cited_chunks = pack_context(candidates)
    
    # Check that only 2 chunks survived (A and C) and B was deduplicated.
    assert len(cited_chunks) == 2, "Expected exactly 2 chunks after deduplication."
    # Verify sequential tag assignments.
    assert cited_chunks[0].citation_tag == "[1]"
    assert cited_chunks[1].citation_tag == "[2]"
    # Verify chunk mappings.
    assert cited_chunks[0].ranked_result.chunk.chunk_id == "chunk_a"
    assert cited_chunks[1].ranked_result.chunk.chunk_id == "chunk_c"


@patch("src.generation.context_compressor.OpenAI") # Mock OpenAI client.
def test_context_compressor(mock_openai_class: MagicMock) -> None:
    """
    Verifies that ContextCompressor requests summarization from OpenAI's API.
    """
    # Initialize mock client.
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock completion return data.
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Compressed text."
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    # Instantiate compressor.
    compressor = ContextCompressor()
    
    # Run compression.
    result = compressor.compress("This is a very long text representing corporate business results.")
    
    # Assertions.
    assert result == "Compressed text.", "Compression result mismatch."
    # Verify mock client call arguments.
    mock_client.chat.completions.create.assert_called_once()


def test_check_citations() -> None:
    """
    Verifies that check_citations parses valid citation tags and evaluates context utilization.
    """
    # Setup mock cited chunks with all required Pydantic properties.
    chunk_1 = CitedChunk(
        ranked_result=_make_mock_ranked_result("a", "text"),
        citation_tag="[1]",
        packed_text="[Filing Source 1] text",
        packed_token_count=10
    )
    chunk_2 = CitedChunk(
        ranked_result=_make_mock_ranked_result("b", "text"),
        citation_tag="[2]",
        packed_text="[Filing Source 2] text",
        packed_token_count=10
    )
    packed = [chunk_1, chunk_2]
    
    # Case 1: All valid citations.
    answer_valid = "According to [1] and [2], margins rose."
    tags, ratio = check_citations(answer_valid, packed)
    assert tags == ["[1]", "[2]"]
    assert ratio == 1.0 # 2 out of 2 used.
    
    # Case 2: Hallucinated citation tag [3] present.
    answer_hallucinated = "Margins rose [1] but inflation is high [3]."
    tags_h, ratio_h = check_citations(answer_hallucinated, packed)
    # Only [1] is valid; [3] should be ignored.
    assert tags_h == ["[1]"]
    assert ratio_h == 0.5 # 1 out of 2 used.
