from __future__ import annotations 
from typing import Dict, List, Optional 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import BM25Result, FusedResult, VectorResult 
log = get_logger(__name__)
def rrf_fuse(
    bm25_results: List[BM25Result],
    vector_results: List[VectorResult],
    k: int = None
) -> List[FusedResult]:
    """
    Fuses keyword (BM25) and semantic vector search results using Reciprocal Rank Fusion.
    """
    rrf_k = k if k is not None else cfg.retrieval.rrf_k
    rrf_scores: Dict[str, float] = {}
    bm25_ranks: Dict[str, int] = {}
    vector_ranks: Dict[str, int] = {}
    chunk_lookup: Dict[str, Any] = {}
    for hit in bm25_results:
        chunk_id = hit.chunk.chunk_id
        bm25_ranks[chunk_id] = hit.bm25_rank
        chunk_lookup[chunk_id] = hit.chunk
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + hit.bm25_rank))
    for hit in vector_results:
        chunk_id = hit.chunk.chunk_id
        vector_ranks[chunk_id] = hit.vector_rank
        chunk_lookup[chunk_id] = hit.chunk
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + hit.vector_rank))
    fused_results: List[FusedResult] = []
    for chunk_id, score in rrf_scores.items():
        chunk = chunk_lookup[chunk_id]
        fused = FusedResult(
            chunk=chunk,
            rrf_score=score,
            bm25_rank=bm25_ranks.get(chunk_id),
            vector_rank=vector_ranks.get(chunk_id)
        )
        fused_results.append(fused)
    fused_results.sort(key=lambda x: x.rrf_score, reverse=True)
    log.debug("rrf_fusion_complete", input_bm25=len(bm25_results), input_vector=len(vector_results), fused_total=len(fused_results))
    return fused_results
