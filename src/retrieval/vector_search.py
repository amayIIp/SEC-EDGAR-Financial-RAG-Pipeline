# src/retrieval/vector_search.py
# This module implements the asynchronous dense vector similarity search.
# We call our SECEmbedder to compute the query's vector embedding, then query Qdrant.
# To keep the server responsive, we run the Qdrant and Embedder calls in separate threads
# using asyncio.to_thread.

from __future__ import annotations # Allow self-referencing type annotations.
import asyncio # Standard library module for async I/O.
from typing import Any, Dict, List, Optional # Type helpers.
from qdrant_client import QdrantClient # Qdrant client.
from qdrant_client.http import models as qmodels # Qdrant schema models.
from src.indexing.embedder import SECEmbedder # Embeddings generator.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import Chunk, VectorResult # Shared models.

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
    
    # Translate filter constraints into Qdrant match/range models.
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

    # Wrap the conditions inside a Qdrant Filter.
    qd_filter = qmodels.Filter(must=conditions) if conditions else None

    try:
        # Search Qdrant for vectors closest to our query vector under the filters.
        # We specify the search distance configuration from the collection.
        hits = client.search(
            collection_name=cfg.qdrant.collection_name,
            query_vector=query_vector,
            query_filter=qd_filter,
            limit=top_k
        )
        
        results: List[VectorResult] = []
        # Convert Qdrant hit records back to VectorResult models.
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
    # ── Step 1: Compute query embedding if not provided precomputed ──
    if precomputed_vector is None:
        # Run the embedding generation in a separate thread.
        # We initialize a new SECEmbedder instance.
        embedder = SECEmbedder()
        query_vectors = await asyncio.to_thread(embedder.embed, [query])
        query_vector = query_vectors[0]
    else:
        query_vector = precomputed_vector

    # ── Step 2: Query Qdrant ──
    client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
    
    return await asyncio.to_thread(
        _sync_vector_search,
        client=client,
        query_vector=query_vector,
        top_k=top_k,
        filters=filters
    )
