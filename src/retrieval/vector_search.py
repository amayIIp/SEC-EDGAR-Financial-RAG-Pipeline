from __future__ import annotations 
import asyncio 
from typing import Any, Dict, List, Optional 
from qdrant_client import QdrantClient 
from qdrant_client.http import models as qmodels 
from src.indexing.embedder import SECEmbedder 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import Chunk, VectorResult 
log = get_logger(__name__)
def _sync_vector_search(
    client: QdrantClient,
    query_vector: List[float],
    top_k: int,
    filters: Optional[Dict[str, Any]]
) -> List[VectorResult]:
    """
    Synchronous implementation of Qdrant search. Runs inside a background thread pool.
    """
    conditions = []
    if filters:
        if filters.get("strategy"):
            conditions.append(qmodels.FieldCondition(
                key="strategy",
                match=qmodels.MatchValue(value=filters["strategy"])
            ))
        if filters.get("ticker"):
            conditions.append(qmodels.FieldCondition(
                key="ticker",
                match=qmodels.MatchValue(value=filters["ticker"].upper())
            ))
        if filters.get("filing_type"):
            conditions.append(qmodels.FieldCondition(
                key="filing_type",
                match=qmodels.MatchValue(value=filters["filing_type"])
            ))
        if filters.get("date_from") or filters.get("date_to"):
            conditions.append(qmodels.FieldCondition(
                key="filing_date",
                range=qmodels.Range(
                    gte=filters.get("date_from"),
                    lte=filters.get("date_to")
                )
            ))
    qd_filter = qmodels.Filter(must=conditions) if conditions else None
    try:
        hits = client.search(
            collection_name=cfg.qdrant.collection_name,
            query_vector=query_vector,
            query_filter=qd_filter,
            limit=top_k
        )
        results: List[VectorResult] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            chunk = Chunk(**payload)
            result = VectorResult(
                chunk=chunk,
                vector_score=float(hit.score),
                vector_rank=rank
            )
            results.append(result)
        return results
    except Exception as exc:
        log.error("qdrant_search_failed", error=str(exc))
        return []
async def vector_search(
    query: str,
    top_k: int = 50,
    filters: Optional[Dict[str, Any]] = None,
    precomputed_vector: Optional[List[float]] = None
) -> List[VectorResult]:
    """
    Asynchronously queries Qdrant for semantic similarity matches.
    Computes query vector embedding and runs the search in background threads.
    """
    if precomputed_vector is None:
        embedder = SECEmbedder()
        query_vectors = await asyncio.to_thread(embedder.embed, [query])
        query_vector = query_vectors[0]
    else:
        query_vector = precomputed_vector
    client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
    return await asyncio.to_thread(
        _sync_vector_search,
        client=client,
        query_vector=query_vector,
        top_k=top_k,
        filters=filters
    )
