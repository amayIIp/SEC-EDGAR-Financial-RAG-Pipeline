# src/reranking/bge_reranker.py
# This module implements the BGEReranker using a local Cross-Encoder model.
# We load BAAI/bge-reranker-v2-m3 locally using SentenceTransformers CrossEncoder.

from __future__ import annotations # Allow self-referencing type annotations.
from typing import List # Type helper.
from sentence_transformers import CrossEncoder # Library to load local CrossEncoder.
from src.reranking.base_reranker import BaseReranker # Base interface.
from src.shared.config import cfg # Config loader singleton.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import FusedResult, RankedResult # Shared models.

log = get_logger(__name__)

class BGEReranker(BaseReranker):
    """
    Reranks search candidate hits locally using a BGE Cross-Encoder model.
    """

    def __init__(self) -> None:
        log.info("loading_local_bge_reranker", model=cfg.reranking.bge.model)
        # Load the CrossEncoder model locally.
        # It automatically detects if GPU is available and handles device configuration.
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

        # Construct pairs: [[query, text_1], [query, text_2], ...]
        pairs = [[query, item.chunk.text] for item in candidates]

        try:
            # Predict relevance scores. The model processes each pair through self-attention layers.
            # We enforce batch size limits to prevent out-of-memory errors on GPU.
            scores = self.model.predict(
                pairs,
                batch_size=cfg.reranking.bge.batch_size,
                show_progress_bar=False
            )
            
            # Convert numpy scores to standard float list.
            if hasattr(scores, "tolist"):
                scores_list = scores.tolist()
            else:
                scores_list = list(scores)

            scored_candidates: List[RankedResult] = []
            
            # Loop through candidates to attach scores.
            for idx, item in enumerate(candidates):
                score = float(scores_list[idx])
                # Note: We temporarily assign final_rank as 0; we will reassign after sorting.
                ranked = RankedResult(
                    chunk=item.chunk,
                    rerank_score=score,
                    final_rank=0,
                    rrf_score=item.rrf_score
                )
                scored_candidates.append(ranked)

            # Sort the candidates by their cross-encoder score in descending order.
            scored_candidates.sort(key=lambda x: x.rerank_score, reverse=True)

            # Slice only the requested top_n results.
            top_ranked = scored_candidates[:top_n]
            
            # Assign the sequential final rank index (1-based) to each element.
            for rank_idx, item in enumerate(top_ranked, start=1):
                item.final_rank = rank_idx

            return top_ranked

        except Exception as exc:
            log.error("bge_rerank_failed", error=str(exc))
            # Fall back to returning the original RRF-fused order if the local inference fails.
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
