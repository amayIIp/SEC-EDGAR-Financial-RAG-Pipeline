from __future__ import annotations 
import pytest 
from src.indexing.opensearch_index import OpenSearchIndexManager 
from src.indexing.qdrant_index import QdrantIndexManager 
from src.shared.config import cfg 
from src.shared.models import Chunk, ChunkingStrategy, FilingType 
pytestmark = pytest.mark.integration
def _make_test_chunk() -> Chunk:
    """
    Helper function to build a mock Chunk record for integration testing.
    """
    return Chunk(
        chunk_id="test_chunk_1", 
        text="Apple Inc. announced a record-breaking corporate revenue of ninety-nine billion dollars in Q3.", 
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
        token_count=20 
    )
def test_indexing_round_trip() -> None:
    """
    Verifies that a chunk can be indexed into both databases,
    and retrieved directly with matching properties.
    """
    os_manager = OpenSearchIndexManager() 
    qd_manager = QdrantIndexManager() 
    dim = 1536
    os_manager.create_index(force=True)
    qd_manager.create_collection(dimension=dim, force=True)
    chunk = _make_test_chunk() 
    vector = [0.0] * dim
    vector[0] = 1.0 
    indexed_os_count = os_manager.bulk_index_chunks([chunk])
    assert indexed_os_count == 1, "Failed to index the test chunk in OpenSearch."
    os_manager.client.indices.refresh(index=os_manager.index_name)
    qd_manager.bulk_upsert_chunks([chunk], [vector])
    os_response = os_manager.client.search(
        index=os_manager.index_name,
        body={
            "query": {
                "match": {
                    "text": "revenue"
                }
            }
        }
    )
    hits = os_response["hits"]["hits"]
    assert len(hits) == 1, "Failed to retrieve the indexed chunk from OpenSearch."
    assert hits[0]["_id"] == "test_chunk_1", "The retrieved chunk ID does not match 'test_chunk_1'."
    assert hits[0]["_source"]["ticker"] == "AAPL", "Filing ticker metadata was not preserved in OpenSearch."
    qd_hits = qd_manager.client.search(
        collection_name=qd_manager.collection_name,
        query_vector=vector,
        limit=1
    )
    assert len(qd_hits) == 1, "Failed to retrieve the indexed vector from Qdrant."
    payload = qd_hits[0].payload or {}
    assert payload.get("chunk_id") == "test_chunk_1", "The retrieved chunk ID in Qdrant does not match."
    assert payload.get("ticker") == "AAPL", "Filing ticker metadata was not preserved in Qdrant."
