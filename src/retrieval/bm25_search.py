# src/retrieval/bm25_search.py
# This module implements the asynchronous BM25 lexical keyword search.
# We wrap the synchronous OpenSearch client inside asyncio.to_thread
# to prevent blocking the FastAPI server event loop.

from __future__ import annotations # Allow self-referencing type annotations.
import asyncio # Standard library module for async I/O.
from typing import Any, Dict, List, Optional # Type helpers.
from opensearchpy import OpenSearch # OpenSearch client.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import BM25Result, Chunk # Shared models.

log = get_logger(__name__)

def _sync_bm25_search(
    client: OpenSearch,
    query: str,
    top_k: int,
    filters: Optional[Dict[str, Any]]
) -> List[BM25Result]:
    """
    Synchronous implementation of BM25 search. Runs inside a background thread pool.
    """
    filter_list = []
    
    # If filters are present, translate them into OpenSearch term or range queries.
    if filters:
        # Strategy exact match.
        if filters.get("strategy"):
            filter_list.append({"term": {"strategy": filters["strategy"]}})
        # Ticker exact match.
        if filters.get("ticker"):
            filter_list.append({"term": {"ticker": filters["ticker"].upper()}})
        # Filing type (e.g. 10-K, 10-Q) exact match.
        if filters.get("filing_type"):
            filter_list.append({"term": {"filing_type": filters["filing_type"]}})
        # Date range queries matching YYYY-MM-DD strings.
        if filters.get("date_from") or filters.get("date_to"):
            date_range = {}
            if filters.get("date_from"):
                date_range["gte"] = filters["date_from"]
            if filters.get("date_to"):
                date_range["lte"] = filters["date_to"]
            filter_list.append({"range": {"filing_date": date_range}})

    # Construct the boolean query body.
    # The 'must' clause ensures keyword match; the 'filter' clause narrows the documents
    # without affecting the relevance score.
    query_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": query
                        }
                    }
                ],
                "filter": filter_list
            }
        },
        "size": top_k
    }

    try:
        # Run the search on OpenSearch.
        response = client.search(index=cfg.opensearch.index_name, body=query_body)
        hits = response["hits"]["hits"]
        
        results: List[BM25Result] = []
        # Convert each raw hit dict back into a structured BM25Result.
        for rank, hit in enumerate(hits, start=1):
            source = hit["_source"]
            chunk = Chunk(**source)
            result = BM25Result(
                chunk=chunk,
                bm25_score=float(hit["_score"]),
                bm25_rank=rank
            )
            results.append(result)
            
        return results
    except Exception as exc:
        log.error("opensearch_search_failed", query=query, error=str(exc))
        return []

async def bm25_search(
    query: str,
    top_k: int = 50,
    filters: Optional[Dict[str, Any]] = None
) -> List[BM25Result]:
    """
    Asynchronously queries OpenSearch for keyword BM25 matches.
    Delegates the synchronous network I/O call to a background worker thread.
    """
    # Create the client connection.
    client = OpenSearch(
        hosts=[{"host": cfg.opensearch.host, "port": cfg.opensearch.port}],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False
    )
    
    # Run in thread pool to prevent blocking the event loop.
    return await asyncio.to_thread(
        _sync_bm25_search,
        client=client,
        query=query,
        top_k=top_k,
        filters=filters
    )
