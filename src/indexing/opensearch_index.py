from __future__ import annotations 
from typing import Any, Dict, List 
from opensearchpy import OpenSearch 
from opensearchpy.helpers import bulk 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import Chunk 
log = get_logger(__name__)
class OpenSearchIndexManager:
    """
    Manages creation and indexing of SEC chunks into an OpenSearch lexical index.
    """
    def __init__(self) -> None:
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
        if force and self.client.indices.exists(index=self.index_name):
            log.info("deleting_existing_opensearch_index", index=self.index_name)
            self.client.indices.delete(index=self.index_name)
        if self.client.indices.exists(index=self.index_name):
            log.info("opensearch_index_exists", index=self.index_name)
            return
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
                    "text": {
                        "type": "text",
                        "similarity": "custom_bm25"
                    },
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
        self.client.indices.create(index=self.index_name, body=index_body)
        log.info("opensearch_index_created", index=self.index_name)
    def bulk_index_chunks(self, chunks: List[Chunk]) -> int:
        """
        Uploads a list of Chunk objects into OpenSearch using bulk operations.
        Returns the number of successfully indexed documents.
        """
        if not chunks:
            return 0
        actions = []
        for chunk in chunks:
            action = {
                "_index": self.index_name,
                "_id": chunk.chunk_id,
                "_source": chunk.model_dump()
            }
            actions.append(action)
        success_count, errors = bulk(
            self.client,
            actions,
            chunk_size=cfg.opensearch.bulk_batch_size,
            request_timeout=60
        )
        if errors:
            log.error("opensearch_bulk_index_errors", errors=errors)
        log.info("opensearch_bulk_index_success", count=success_count)
        return success_count
