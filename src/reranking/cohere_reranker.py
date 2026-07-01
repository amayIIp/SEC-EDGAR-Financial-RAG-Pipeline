# src/reranking/cohere_reranker.py
# This module implements the CohereReranker using Cohere's cloud API.
# Cohere Rerank model (rerank-english-v3.0) calculates semantic relevance.

from __future__ import annotations # Allow self-referencing type annotations.
import os # Standard library module to read environment variables.
from typing import List # Type helper.
import cohere # Cohere Python client library.
from src.reranking.base_reranker import BaseReranker # Base interface.
from src.shared.config import cfg # Config loader singleton.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import FusedResult, RankedResult # Shared models.

log = get_logger(__name__)

class CohereReranker(BaseReranker):
    """
    Reranks candidate search hits using Cohere's Rerank API.
    """

    def __init__(self) -> None:
        # Load API key.
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is missing.")
        # Initialize client connection.
        self.client = cohere.Client(api_key)

    def rerank(self, query: str, candidates: List[FusedResult], top_n: int) -> List[RankedResult]:
        """
        Calls Cohere's rerank endpoint and returns the top_n results.
        """
        if not candidates:
            return []

        # Extract the raw text from each chunk to feed to the API.
        documents = [item.chunk.text for item in candidates]
        
        # Determine the number of outputs to request.
        request_limit = min(top_n, len(candidates))

        try:
            # Issue the request.
            # We pass the model name, query, candidate documents, and target count.
            response = self.client.rerank(
                model=cfg.reranking.cohere.model,
                query=query,
                documents=documents,
                top_n=request_limit
            )
            
            ranked_results: List[RankedResult] = []
            # Loop through response results.
            # The API returns objects containing index position and relevance score.
            for idx, result in enumerate(response.results, start=1):
                # Retrieve the original FusedResult record.
                fused = candidates[result.index]
                
                # Build the RankedResult model.
                ranked = RankedResult(
                    chunk=fused.chunk,
                    rerank_score=float(result.relevance_score),
                    final_rank=idx,
                    rrf_score=fused.rrf_score
                )
                ranked_results.append(ranked)
                
            return ranked_results
            
        except Exception as exc:
            log.error("cohere_rerank_failed", error=str(exc))
            # Fall back to returning the original RRF-fused order if the API fails.
            log.warning("cohere_rerank_fallback_to_rrf")
            fallback_results = []
            for idx, item in enumerate(candidates[:top_n], start=1):
                fallback_results.append(RankedResult(
                    chunk=item.chunk,
                    rerank_score=0.0,
                    final_rank=idx,
                    rrf_score=item.rrf_score
                ))
            return fallback_results
