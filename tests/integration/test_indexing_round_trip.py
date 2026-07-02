# tests/integration/test_indexing_round_trip.py
# This module implements integration tests for the indexing round-trip flow.
# We index a mock chunk and its embedding vector into OpenSearch and Qdrant,
# then query both engines directly to verify that the data was stored and
# is retrievable.

from __future__ import annotations # Allow self-referencing type annotations.
import pytest # Testing framework.
from src.indexing.opensearch_index import OpenSearchIndexManager # OpenSearch manager.
from src.indexing.qdrant_index import QdrantIndexManager # Qdrant manager.
from src.shared.config import cfg # Config loader singleton.
from src.shared.models import Chunk, ChunkingStrategy, FilingType # Data models.

# Declare that this is an integration test.
# Integration tests require running database services (OpenSearch, Qdrant).
pytestmark = pytest.mark.integration

def _make_test_chunk() -> Chunk:
    """
    Helper function to build a mock Chunk record for integration testing.
    """
    return Chunk(
        chunk_id="test_chunk_1", # Unique ID for our test chunk.
        text="Apple Inc. announced a record-breaking corporate revenue of ninety-nine billion dollars in Q3.", # Chunk content text.
        strategy=ChunkingStrategy.STRUCTURE_AWARE, # Chunking strategy label.
        chunk_index=0, # Index position.
        is_table=False, # Flag indicating if it represents tabular layout.
        ticker="AAPL", # Stock ticker.
        cik="0000320193", # CIK code.
        filing_type=FilingType.TEN_K, # Document type.
        filing_date="2023-09-30", # Document filing date.
        fiscal_year=2023, # Fiscal year.
        accession_number="0000320193-23-000106", # Accession number.
        section="Item 7", # Section code.
        section_title="MD&A", # Section title.
        token_count=20 # Token length count.
    )

def test_indexing_round_trip() -> None:
    """
    Verifies that a chunk can be indexed into both databases,
    and retrieved directly with matching properties.
    """
    # ── Step 1: Initialize managers ──
    os_manager = OpenSearchIndexManager() # OpenSearch index coordinator.
    qd_manager = QdrantIndexManager() # Qdrant collection coordinator.
    
    # Define vector dimension (match cfg or default to 1536).
    dim = 1536
    
    # ── Step 2: Recreate test indices ──
    # We force recreate the index and collection to ensure a clean state.
    os_manager.create_index(force=True)
    qd_manager.create_collection(dimension=dim, force=True)
    
    # ── Step 3: Create mock chunk and dummy vector ──
    chunk = _make_test_chunk() # Get mock chunk.
    # Generate a simple unit vector with matching dimensions.
    vector = [0.0] * dim
    vector[0] = 1.0 # Set first coordinate to 1 to simulate a normalized embedding.
    
    # ── Step 4: Index into databases ──
    # Upload the chunk to the OpenSearch BM25 keyword index.
    indexed_os_count = os_manager.bulk_index_chunks([chunk])
    assert indexed_os_count == 1, "Failed to index the test chunk in OpenSearch."
    
    # Force a refresh of the OpenSearch index so the document becomes searchable immediately.
    os_manager.client.indices.refresh(index=os_manager.index_name)
    
    # Upload the chunk and vector to the Qdrant dense vector store.
    qd_manager.bulk_upsert_chunks([chunk], [vector])
    
    # ── Step 5: Verify OpenSearch retrieval ──
    # Run a low-level keyword search query directly on OpenSearch client.
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
    
    # Check that we received at least one hit.
    hits = os_response["hits"]["hits"]
    assert len(hits) == 1, "Failed to retrieve the indexed chunk from OpenSearch."
    assert hits[0]["_id"] == "test_chunk_1", "The retrieved chunk ID does not match 'test_chunk_1'."
    assert hits[0]["_source"]["ticker"] == "AAPL", "Filing ticker metadata was not preserved in OpenSearch."

    # ── Step 6: Verify Qdrant retrieval ──
    # Run a low-level vector search directly on the Qdrant client.
    qd_hits = qd_manager.client.search(
        # Pass the collection name.
        collection_name=qd_manager.collection_name,
        # Pass the dummy embedding vector.
        query_vector=vector,
        # Fetch the top-1 closest hit.
        limit=1
    )
    
    # Check that we received at least one point.
    assert len(qd_hits) == 1, "Failed to retrieve the indexed vector from Qdrant."
    payload = qd_hits[0].payload or {}
    assert payload.get("chunk_id") == "test_chunk_1", "The retrieved chunk ID in Qdrant does not match."
    assert payload.get("ticker") == "AAPL", "Filing ticker metadata was not preserved in Qdrant."
