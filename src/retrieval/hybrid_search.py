# src/retrieval/hybrid_search.py
# This module implements the unified hybrid search coordinator.
# It supports three modes: "bm25_only", "vector_only", and "hybrid".
# In "hybrid" mode, it runs both queries concurrently using asyncio.gather,
# then fuses them using Reciprocal Rank Fusion.

from __future__ import annotations # Allow self-referencing type annotations.
import asyncio # Standard library module for async event loop gathering.
from typing import Any, Dict, List, Optional # Type helpers.
from src.retrieval.bm25_search import bm25_search # Async BM25 query.
from src.retrieval.vector_search import vector_search # Async vector query.
from src.retrieval.rrf_fusion import rrf_fuse # RRF rank merger.
from src.shared.config import cfg # Config loader singleton.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import BM25Result, FusedResult, RetrievalMode, VectorResult # Shared models.

log = get_logger(__name__)

async def hybrid_search(
    query: str,
    top_k: int = 50,
    filters: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    depth: Optional[int] = None
) -> List[FusedResult]:
    """
    Executes a search using BM25, vector, or hybrid retrieval.
    Returns a unified list of FusedResult models.
    """
    # Load search mode from arguments, falling back to configuration default.
    search_mode = (mode or cfg.retrieval.mode).lower()
    
    # Load search depth (number of candidates to retrieve from each database).
    search_depth = depth or cfg.retrieval.retrieval_depth
    
    log.info("hybrid_search_triggered", query=query[:60], mode=search_mode, filters=filters)

    # ── Pathway 1: BM25 Only ──
    if search_mode == RetrievalMode.BM25_ONLY.value:
        # Run BM25 search.
        bm25_hits = await bm25_search(query, top_k=top_k, filters=filters)
        
        # Map the results to FusedResult models for consistency.
        fused_hits: List[FusedResult] = []
        for hit in bm25_hits:
            fused = FusedResult(
                chunk=hit.chunk,
                rrf_score=1.0 / (cfg.retrieval.rrf_k + hit.bm25_rank), # Simple reciprocal rank scoring.
                bm25_rank=hit.bm25_rank,
                vector_rank=None
            )
            fused_hits.append(fused)
        return fused_hits

    # ── Pathway 2: Vector Only ──
    elif search_mode == RetrievalMode.VECTOR_ONLY.value:
        # Run vector search.
        vector_hits = await vector_search(query, top_k=top_k, filters=filters)
        
        # Map the results to FusedResult models.
        fused_hits = []
        for hit in vector_hits:
            fused = FusedResult(
                chunk=hit.chunk,
                rrf_score=1.0 / (cfg.retrieval.rrf_k + hit.vector_rank),
                bm25_rank=None,
                vector_rank=hit.vector_rank
            )
            fused_hits.append(fused)
        return fused_hits

    # ── Pathway 3: Hybrid Search (Concurrently executed and fused via RRF) ──
    elif search_mode == RetrievalMode.HYBRID.value:
        # Define tasks for both databases.
        bm25_task = bm25_search(query, top_k=search_depth, filters=filters)
        vector_task = vector_search(query, top_k=search_depth, filters=filters)
        
        # Execute BM25 and Vector queries in parallel using asyncio.gather.
        # This yields two result lists when both tasks finish.
        bm25_res, vector_res = await asyncio.gather(bm25_task, vector_task)
        
        # Fuse the two ranked candidate sets using Reciprocal Rank Fusion.
        fused_hits = rrf_fuse(bm25_res, vector_res)
        
        # Return only the requested top_k items.
        return fused_hits[:top_k]

    else:
        raise ValueError(f"Unknown retrieval mode: {search_mode}")
