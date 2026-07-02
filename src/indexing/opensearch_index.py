# src/indexing/opensearch_index.py
# This module manages the OpenSearch BM25 keyword index.
# OpenSearch is a distributed search engine built on Apache Lucene.
#
# =========================================================================================
# Advanced Concept: BM25 Lexical Search and Index Mapping
# Lexical search uses the BM25 (Best Matching 25) algorithm, which rates documents based on
# term frequency (how often a keyword appears in a chunk) and inverse document frequency
# (how rare the keyword is across the entire index).
# Mapping is the schema definition in OpenSearch. We map the 'text' field to type 'text'
# so that OpenSearch breaks it down into individual word tokens (analyzed search).
# Other fields like CIK, ticker, and filing_date are mapped as type 'keyword', meaning they
# are indexed as exact strings for fast, non-tokenized metadata filtering (e.g. search only
# filings from ticker "AAPL").
# =========================================================================================

from __future__ import annotations # Allow self-referencing type hints.
from typing import Any, Dict, List # Type helpers.
from opensearchpy import OpenSearch # Official OpenSearch Python client.
from opensearchpy.helpers import bulk # Bulk indexing helper for high-throughput uploads.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import get_logger # Logger setup.
from src.shared.models import Chunk # Shared models.

log = get_logger(__name__)

class OpenSearchIndexManager:
    """
    Manages creation and indexing of SEC chunks into an OpenSearch lexical index.
    """

    def __init__(self) -> None:
        # Establish a connection to the OpenSearch service.
        # We disable basic auth / SSL verification since we disabled the security plugin in docker-compose
        # to simplify local API operations.
        self.client = OpenSearch(
            hosts=[{"host": cfg.opensearch.host, "port": cfg.opensearch.port}],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False
        )
        self.index_name = cfg.opensearch.index_name

    def create_index(self, force: bool = False) -> None:
        """
        Creates the OpenSearch index with BM25 similarity mappings if it does not exist.
        If force=True, deletes the existing index first.
        """
        # Delete index if force deletion is requested.
        if force and self.client.indices.exists(index=self.index_name):
            log.info("deleting_existing_opensearch_index", index=self.index_name)
            self.client.indices.delete(index=self.index_name)

        # Skip creation if index already exists.
        if self.client.indices.exists(index=self.index_name):
            log.info("opensearch_index_exists", index=self.index_name)
            return

        # Define the schema settings and field mappings.
        # We configure BM25 with k1 (term saturation) and b (length normalisation) parameters.
        index_body = {
            "settings": {
                "index": {
                    "number_of_shards": cfg.opensearch.shards,
                    "number_of_replicas": cfg.opensearch.replicas,
                    "similarity": {
                        "custom_bm25": {
                            "type": "BM25",
                            "k1": cfg.opensearch.bm25_k1,
                            "b": cfg.opensearch.bm25_b
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    # The text field uses the custom BM25 similarity configuration and is analyzed.
                    "text": {
                        "type": "text",
                        "similarity": "custom_bm25"
                    },
                    # Metadata fields are indexed as keywords to support exact filtering without tokenization.
                    "chunk_id": {"type": "keyword"},
                    "ticker": {"type": "keyword"},
                    "cik": {"type": "keyword"},
                    "filing_type": {"type": "keyword"},
                    "filing_date": {"type": "keyword"},
                    "fiscal_year": {"type": "integer"},
                    "accession_number": {"type": "keyword"},
                    "section": {"type": "keyword"},
                    "section_title": {"type": "keyword"},
                    "token_count": {"type": "integer"}
                }
            }
        }

        # Issue the create index request.
        self.client.indices.create(index=self.index_name, body=index_body)
        log.info("opensearch_index_created", index=self.index_name)

    def bulk_index_chunks(self, chunks: List[Chunk]) -> int:
        """
        Uploads a list of Chunk objects into OpenSearch using bulk operations.
        Returns the number of successfully indexed documents.
        """
        if not chunks:
            return 0

        # Construct the actions list expected by OpenSearch's bulk helper.
        # Using a generator expression to yield dict actions on the fly saves memory.
        actions = []
        for chunk in chunks:
            # We use the unique chunk_id as the document ID in OpenSearch.
            # This makes our indexing idempotent: re-indexing updates in-place.
            action = {
                "_index": self.index_name,
                "_id": chunk.chunk_id,
                # Convert the Chunk model properties directly to the index source document.
                "_source": chunk.model_dump()
            }
            actions.append(action)

        # Call the bulk helper. This manages network requests, retries, and errors.
        # bulk returns a tuple: (success_count, list_of_errors_if_any).
        success_count, errors = bulk(
            self.client,
            actions,
            chunk_size=cfg.opensearch.bulk_batch_size,
            request_timeout=60
        )
        
        # Log results.
        if errors:
            log.error("opensearch_bulk_index_errors", errors=errors)
        log.info("opensearch_bulk_index_success", count=success_count)
        
        return success_count
