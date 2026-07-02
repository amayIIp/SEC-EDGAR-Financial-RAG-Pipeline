from __future__ import annotations 
import asyncio 
from typing import Any, Dict, List, Optional 
from opensearchpy import OpenSearch 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import BM25Result, Chunk 
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
    if filters:
        if filters.get("strategy"):
            filter_list.append({"term": {"strategy": filters["strategy"]}})
        if filters.get("ticker"):
            filter_list.append({"term": {"ticker": filters["ticker"].upper()}})
        if filters.get("filing_type"):
            filter_list.append({"term": {"filing_type": filters["filing_type"]}})
        if filters.get("date_from") or filters.get("date_to"):
            date_range = {}
            if filters.get("date_from"):
                date_range["gte"] = filters["date_from"]
            if filters.get("date_to"):
                date_range["lte"] = filters["date_to"]
            filter_list.append({"range": {"filing_date": date_range}})
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
        response = client.search(index=cfg.opensearch.index_name, body=query_body)
        hits = response["hits"]["hits"]
        results: List[BM25Result] = []
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
    client = OpenSearch(
        hosts=[{"host": cfg.opensearch.host, "port": cfg.opensearch.port}],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False
    )
    return await asyncio.to_thread(
        _sync_bm25_search,
        client=client,
        query=query,
        top_k=top_k,
        filters=filters
    )
