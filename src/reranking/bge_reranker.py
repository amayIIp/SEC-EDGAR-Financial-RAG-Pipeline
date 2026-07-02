from __future__ import annotations 
from typing import List 
from sentence_transformers import CrossEncoder 
from src.reranking.base_reranker import BaseReranker 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import FusedResult, RankedResult 
log = get_logger(__name__)
class BGEReranker(BaseReranker):
    """
    Reranks search candidate hits locally using a BGE Cross-Encoder model.
    """
    def __init__(self) -> None:
        log.info("loading_local_bge_reranker", model=cfg.reranking.bge.model)
        self.model = CrossEncoder(
            cfg.reranking.bge.model,
            device=cfg.reranking.bge.device if cfg.reranking.bge.device != "auto" else None
        )
        log.info("bge_reranker_loaded_successfully")
    def rerank(self, query: str, candidates: List[FusedResult], top_n: int) -> List[RankedResult]:
        """
        Runs the local Cross-Encoder over query-document pairs, sorts, and returns top_n.
        """
        if not candidates:
            return []
        pairs = [[query, item.chunk.text] for item in candidates]
        try:
            scores = self.model.predict(
                pairs,
                batch_size=cfg.reranking.bge.batch_size,
                show_progress_bar=False
            )
            if hasattr(scores, "tolist"):
                scores_list = scores.tolist()
            else:
                scores_list = list(scores)
            scored_candidates: List[RankedResult] = []
            for idx, item in enumerate(candidates):
                score = float(scores_list[idx])
                ranked = RankedResult(
                    chunk=item.chunk,
                    rerank_score=score,
                    final_rank=0,
                    rrf_score=item.rrf_score
                )
                scored_candidates.append(ranked)
            scored_candidates.sort(key=lambda x: x.rerank_score, reverse=True)
            top_ranked = scored_candidates[:top_n]
            for rank_idx, item in enumerate(top_ranked, start=1):
                item.final_rank = rank_idx
            return top_ranked
        except Exception as exc:
            log.error("bge_rerank_failed", error=str(exc))
            log.warning("bge_rerank_fallback_to_rrf")
            fallback_results = []
            for idx, item in enumerate(candidates[:top_n], start=1):
                fallback_results.append(RankedResult(
                    chunk=item.chunk,
                    rerank_score=0.0,
                    final_rank=idx,
                    rrf_score=item.rrf_score
                ))
            return fallback_results
