# src/retrieval/rrf_fusion.py
# This module implements the Reciprocal Rank Fusion (RRF) algorithm.
#
# =========================================================================================
# Advanced Concept: Reciprocal Rank Fusion (RRF)
# In hybrid search, we receive two ranked lists of matching documents:
# - BM25 keyword match results (scored, e.g. 15.4)
# - Vector similarity results (scored, e.g. 0.85)
# Because these scores represent completely different mathematical scales, we cannot add them.
# RRF solves this by ignoring raw scores entirely. It ranks results based solely on their
# relative rank position (order) in each search index.
# The RRF formula is:
# RRF_Score(doc) = Sum( 1 / (k + Rank_system(doc)) )
# where 'k' is a constant (typically 60) that prevents high-ranked items from dominating
# the score. RRF is robust and requires no tuning of learned weights.
# =========================================================================================

from __future__ import annotations # Allow self-referencing type annotations.
from typing import Dict, List, Optional # Type helpers.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import BM25Result, FusedResult, VectorResult # Shared models.

log = get_logger(__name__)

def rrf_fuse(
    bm25_results: List[BM25Result],
    vector_results: List[VectorResult],
    k: int = None
) -> List[FusedResult]:
    """
    Fuses keyword (BM25) and semantic vector search results using Reciprocal Rank Fusion.
    """
    # Use config value for RRF constant k if not explicitly overridden.
    rrf_k = k if k is not None else cfg.retrieval.rrf_k
    
    # Dictionaries to track rank and score information.
    rrf_scores: Dict[str, float] = {}
    bm25_ranks: Dict[str, int] = {}
    vector_ranks: Dict[str, int] = {}
    chunk_lookup: Dict[str, Any] = {}

    # Step 1: Process BM25 list.
    # Record the rank index (1-based) of each chunk in the BM25 results list.
    for hit in bm25_results:
        chunk_id = hit.chunk.chunk_id
        bm25_ranks[chunk_id] = hit.bm25_rank
        chunk_lookup[chunk_id] = hit.chunk
        
        # Calculate reciprocal rank contribution: 1 / (k + rank).
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + hit.bm25_rank))

    # Step 2: Process Vector search list.
    # Record the rank index (1-based) of each chunk in the vector results list.
    for hit in vector_results:
        chunk_id = hit.chunk.chunk_id
        vector_ranks[chunk_id] = hit.vector_rank
        chunk_lookup[chunk_id] = hit.chunk
        
        # Calculate reciprocal rank contribution and add to accumulator.
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + hit.vector_rank))

    # Step 3: Compile fused result objects.
    fused_results: List[FusedResult] = []
    
    # Loop through each unique chunk found in either list.
    for chunk_id, score in rrf_scores.items():
        chunk = chunk_lookup[chunk_id]
        
        # Build the FusedResult model, attaching original ranks for auditability.
        fused = FusedResult(
            chunk=chunk,
            rrf_score=score,
            bm25_rank=bm25_ranks.get(chunk_id),
            vector_rank=vector_ranks.get(chunk_id)
        )
        fused_results.append(fused)

    # Step 4: Sort all results by their new fused RRF score in descending order (highest first).
    fused_results.sort(key=lambda x: x.rrf_score, reverse=True)

    log.debug("rrf_fusion_complete", input_bm25=len(bm25_results), input_vector=len(vector_results), fused_total=len(fused_results))
    
    # Return the fused list.
    return fused_results
