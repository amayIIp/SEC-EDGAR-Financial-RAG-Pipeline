from __future__ import annotations 
from unittest.mock import MagicMock, patch 
import pytest 
from src.indexing.opensearch_index import OpenSearchIndexManager 
from src.indexing.qdrant_index import QdrantIndexManager 
from src.retrieval.hybrid_search import hybrid_search 
from src.shared.models import Chunk, ChunkingStrategy, FilingType 
pytestmark = pytest.mark.integration
def _make_test_chunks() -> list[Chunk]:
    """
    Helper function to build a list of two distinct test Chunk records.
    """
    return [
        Chunk(
            chunk_id="chunk_a",
            text="Operating margins improved by fifteen percent due to supply chain optimization.",
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
            chunk_index=0,
            is_table=False,
            ticker="MSFT",
            cik="0000789019",
            filing_type=FilingType.TEN_K,
            filing_date="2023-09-30",
            fiscal_year=2023,
            accession_number="0000789019-23-000100",
            section="Item 7",
            section_title="MD&A",
            token_count=15
        ),
        Chunk(
            chunk_id="chunk_b",
            text="Research and development costs for cloud infrastructure exceeded five billion dollars.",
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
            chunk_index=1,
            is_table=False,
            ticker="MSFT",
            cik="0000789019",
            filing_type=FilingType.TEN_K,
            filing_date="2023-09-30",
            fiscal_year=2023,
            accession_number="0000789019-23-000100",
            section="Item 7",
            section_title="MD&A",
            token_count=15
        )
    ]
@pytest.mark.asyncio 
@patch("src.retrieval.vector_search.SECEmbedder") 
async def test_hybrid_search_scenarios(mock_embedder_class: MagicMock) -> None:
    """
    Indexes test chunks and verifies search recall under BM25, vector, and hybrid modes.
    """
    mock_embedder = MagicMock()
    mock_embedder_class.return_value = mock_embedder
    dim = 1536
    vector_a = [0.0] * dim
    vector_a[0] = 1.0 
    vector_b = [0.0] * dim
    vector_b[1] = 1.0 
    mock_embedder.embed.return_value = [vector_a]
    os_manager = OpenSearchIndexManager()
    qd_manager = QdrantIndexManager()
    os_manager.create_index(force=True)
    qd_manager.create_collection(dimension=dim, force=True)
    chunks = _make_test_chunks()
    os_manager.bulk_index_chunks(chunks)
    qd_manager.bulk_upsert_chunks(chunks, [vector_a, vector_b])
    os_manager.client.indices.refresh(index=os_manager.index_name)
    bm25_results = await hybrid_search(
        query="operating margins",
        mode="bm25_only",
        top_k=2
    )
    assert len(bm25_results) >= 1, "BM25 search returned no results."
    assert bm25_results[0].chunk.chunk_id == "chunk_a", "BM25 failed to return chunk_a as top hit."
    assert bm25_results[0].bm25_rank == 1, "Top hit rank is not 1."
    vector_results = await hybrid_search(
        query="semantic query text",
        mode="vector_only",
        top_k=2
    )
    assert len(vector_results) >= 1, "Vector search returned no results."
    assert vector_results[0].chunk.chunk_id == "chunk_a", "Vector search failed to match chunk_a."
    assert vector_results[0].vector_rank == 1, "Top semantic hit rank is not 1."
    hybrid_results = await hybrid_search(
        query="margins",
        mode="hybrid",
        top_k=2
    )
    assert len(hybrid_results) >= 1, "Hybrid search returned no results."
    assert hybrid_results[0].chunk.chunk_id == "chunk_a", "Hybrid search failed to prioritize chunk_a."
