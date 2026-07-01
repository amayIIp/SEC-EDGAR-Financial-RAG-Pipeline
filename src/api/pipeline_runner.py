# src/api/pipeline_runner.py
# This module implements the main pipeline runner.
# It orchestrates:
#   1. Query embedding + search (BM25 & Qdrant in parallel).
#   2. Rank Fusion (RRF).
#   3. Reranking (Cohere or BGE Cross-Encoder).
#   4. Prompt context packing (with deduplication).
#   5. LLM Generation.
# We capture exact time differences around each step to feed the latency profiling dashboard.

from __future__ import annotations # Allow self-referencing type annotations.
import time # Standard library module to measure latency.
from typing import Any, Dict, List, Optional # Type helpers.
from src.retrieval.bm25_search import bm25_search # Async BM25 query.
from src.retrieval.vector_search import vector_search # Async vector query.
from src.retrieval.rrf_fusion import rrf_fuse # RRF fusion.
from src.reranking.cohere_reranker import CohereReranker # Cohere API reranker.
from src.reranking.bge_reranker import BGEReranker # Local CrossEncoder reranker.
from src.generation.context_packer import pack_context # Prompt formatter.
from src.generation.llm_client import SECGenerator # LLM client wrapper.
from src.indexing.embedder import SECEmbedder # Embedding generator.
from src.shared.config import cfg # Config loader singleton.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import FusedResult, GenerationOutput, RankedResult, FusedResult # Shared models.

log = get_logger(__name__)

class PipelineRunner:
    """
    Coordinates end-to-end execution of RAG stages, tracking latency profile metrics.
    """

    def __init__(self) -> None:
        # Initialize clients.
        self.generator = SECGenerator()
        self.cohere_reranker = CohereReranker()
        self.bge_reranker = BGEReranker()

    def ingest(self, ticker: str, forms: List[str], limit: int, embedding_provider: str) -> int:
        """
        Runs document download, parsing, chunking, and indexing.
        """
        from src.pipeline import SECRAGPipeline
        pipeline = SECRAGPipeline()
        return pipeline.ingest_ticker_data(
            ticker=ticker,
            forms=forms,
            limit=limit,
            embedding_provider=embedding_provider
        )

    async def run_pipeline(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "hybrid",
        top_k: int = 15,
        top_n: int = 5,
        embedding_provider: Optional[str] = None,
        reranker_provider: Optional[str] = None,
        generator_provider: Optional[str] = None,
        generator_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs the RAG query execution workflow. Returns a dict containing:
        {
          "answer": str,
          "citations": List[dict],
          "latency_breakdown": Dict[str, float],
          "raw_state": Dict[str, Any] (for debugging),
          "total_tokens": int
        }
        """
        timings: Dict[str, float] = {}
        t_start = time.perf_counter()
        
        # Build search filters. Include selected strategy filter.
        search_filters = {}
        if filters:
            if filters.get("ticker"):
                search_filters["ticker"] = filters["ticker"]
            if filters.get("filing_type"):
                search_filters["filing_type"] = filters["filing_type"]
            if filters.get("date_from"):
                search_filters["date_from"] = filters["date_from"]
            if filters.get("date_to"):
                search_filters["date_to"] = filters["date_to"]
                
        # Force matching the active chunking strategy (default to structure_aware for search).
        search_filters["strategy"] = "structure_aware"

        # ── Step 1: Query Embedding ──
        t_step = time.perf_counter()
        embedder = SECEmbedder(provider=embedding_provider)
        query_vectors = embedder.embed([query])
        query_vector = query_vectors[0]
        timings["query_embedding"] = (time.perf_counter() - t_step) * 1000

        # ── Step 2: Parallel Search & RRF Fusion ──
        t_step = time.perf_counter()
        search_mode = mode.lower()
        
        if search_mode == "bm25_only":
            bm25_res = await bm25_search(query, top_k=top_k, filters=search_filters)
            vector_res = []
            fused_hits = []
            for hit in bm25_res:
                fused_hits.append(FusedResult(
                    chunk=hit.chunk,
                    rrf_score=1.0 / (cfg.retrieval.rrf_k + hit.bm25_rank),
                    bm25_rank=hit.bm25_rank,
                    vector_rank=None
                ))
        elif search_mode == "vector_only":
            bm25_res = []
            vector_res = await vector_search(query, top_k=top_k, filters=search_filters, precomputed_vector=query_vector)
            fused_hits = []
            for hit in vector_res:
                fused_hits.append(FusedResult(
                    chunk=hit.chunk,
                    rrf_score=1.0 / (cfg.retrieval.rrf_k + hit.vector_rank),
                    bm25_rank=None,
                    vector_rank=hit.vector_rank
                ))
        else:
            # Concurrently query OpenSearch and Qdrant.
            bm25_task = bm25_search(query, top_k=cfg.retrieval.retrieval_depth, filters=search_filters)
            vector_task = vector_search(query, top_k=cfg.retrieval.retrieval_depth, filters=search_filters, precomputed_vector=query_vector)
            bm25_res, vector_res = await asyncio.gather(bm25_task, vector_task)
            
            # Fuse with Reciprocal Rank Fusion.
            fused_hits = rrf_fuse(bm25_res, vector_res)
            
        timings["hybrid_search"] = (time.perf_counter() - t_step) * 1000

        # ── Step 3: Reranking ──
        t_step = time.perf_counter()
        active_reranker = (reranker_provider or cfg.reranking.provider).lower()
        
        if active_reranker == "cohere":
            ranked_hits = self.cohere_reranker.rerank(query, fused_hits, top_n=top_n)
        elif active_reranker == "bge":
            ranked_hits = self.bge_reranker.rerank(query, fused_hits, top_n=top_n)
        else:
            # Bypass reranking.
            ranked_hits = []
            for idx, hit in enumerate(fused_hits[:top_n], start=1):
                from src.shared.models import RankedResult
                ranked_hits.append(RankedResult(
                    chunk=hit.chunk,
                    rerank_score=hit.rrf_score,
                    final_rank=idx,
                    rrf_score=hit.rrf_score
                ))
                
        timings["reranking"] = (time.perf_counter() - t_step) * 1000

        # ── Step 4: Context Packing ──
        t_step = time.perf_counter()
        packed_context = pack_context(ranked_hits)
        timings["context_packing"] = (time.perf_counter() - t_step) * 1000

        # ── Step 5: Generation ──
        t_step = time.perf_counter()
        gen_out = self.generator.generate(
            query=query,
            chunks=packed_context,
            provider=generator_provider,
            model=generator_model
        )
        timings["generation"] = (time.perf_counter() - t_step) * 1000
        
        timings["total_pipeline"] = (time.perf_counter() - t_start) * 1000

        # Compile citations list for response schema.
        citations = []
        for tag in gen_out.cited_tags:
            # Find the matching CitedChunk in context.
            matched_chunk = next((c for c in packed_context if c.citation_tag == tag), None)
            if matched_chunk:
                citations.append({
                    "chunk_id": matched_chunk.ranked_result.chunk.chunk_id,
                    "citation_tag": tag,
                    "text_snippet": matched_chunk.ranked_result.chunk.text,
                    "ticker": matched_chunk.ranked_result.chunk.ticker,
                    "form": matched_chunk.ranked_result.chunk.filing_type.value,
                    "filing_date": matched_chunk.ranked_result.chunk.filing_date,
                    "section": matched_chunk.ranked_result.chunk.section
                })

        return {
            "answer": gen_out.answer,
            "citations": citations,
            "latency_breakdown": timings,
            "total_tokens": gen_out.total_tokens,
            # Intermediate state dictionary returned for /debug diagnostics.
            "raw_state": {
                "bm25_hits": [h.model_dump() for h in bm25_res],
                "vector_hits": [h.model_dump() for h in vector_res],
                "fused_hits": [h.model_dump() for h in fused_hits[:10]],
                "reranked_hits": [h.model_dump() for h in ranked_hits],
                "final_context": [c.model_dump() for c in packed_context]
            }
        }
