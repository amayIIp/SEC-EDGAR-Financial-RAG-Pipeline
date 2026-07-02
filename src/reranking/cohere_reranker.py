from __future__ import annotations 
import os 
from typing import List 
import cohere 
from src.reranking.base_reranker import BaseReranker 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import FusedResult, RankedResult 
log = get_logger(__name__)
class CohereReranker(BaseReranker):
    """
    Reranks candidate search hits using Cohere's Rerank API.
    """
    def __init__(self) -> None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is missing.")
        self.client = cohere.Client(api_key)
    def rerank(self, query: str, candidates: List[FusedResult], top_n: int) -> List[RankedResult]:
        """
        Calls Cohere's rerank endpoint and returns the top_n results.
        """
        if not candidates:
            return []
        documents = [item.chunk.text for item in candidates]
        request_limit = min(top_n, len(candidates))
        try:
            response = self.client.rerank(
                model=cfg.reranking.cohere.model,
                query=query,
                documents=documents,
                top_n=request_limit
            )
            ranked_results: List[RankedResult] = []
            for idx, result in enumerate(response.results, start=1):
                fused = candidates[result.index]
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
