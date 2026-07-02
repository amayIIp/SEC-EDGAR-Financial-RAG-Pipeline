from __future__ import annotations 
import asyncio 
from typing import Any, Dict, List, Optional 
from src.retrieval.bm25_search import bm25_search 
from src.retrieval.vector_search import vector_search 
from src.retrieval.rrf_fusion import rrf_fuse 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import BM25Result, FusedResult, RetrievalMode, VectorResult 
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
    search_mode = (mode or cfg.retrieval.mode).lower()
    search_depth = depth or cfg.retrieval.retrieval_depth
    log.info("hybrid_search_triggered", query=query[:60], mode=search_mode, filters=filters)
    if search_mode == RetrievalMode.BM25_ONLY.value:
        bm25_hits = await bm25_search(query, top_k=top_k, filters=filters)
        fused_hits: List[FusedResult] = []
        for hit in bm25_hits:
            fused = FusedResult(
                chunk=hit.chunk,
                rrf_score=1.0 / (cfg.retrieval.rrf_k + hit.bm25_rank), 
                bm25_rank=hit.bm25_rank,
                vector_rank=None
            )
            fused_hits.append(fused)
        return fused_hits
    elif search_mode == RetrievalMode.VECTOR_ONLY.value:
        vector_hits = await vector_search(query, top_k=top_k, filters=filters)
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
    elif search_mode == RetrievalMode.HYBRID.value:
        bm25_task = bm25_search(query, top_k=search_depth, filters=filters)
        vector_task = vector_search(query, top_k=search_depth, filters=filters)
        bm25_res, vector_res = await asyncio.gather(bm25_task, vector_task)
        fused_hits = rrf_fuse(bm25_res, vector_res)
        return fused_hits[:top_k]
    else:
        raise ValueError(f"Unknown retrieval mode: {search_mode}")
