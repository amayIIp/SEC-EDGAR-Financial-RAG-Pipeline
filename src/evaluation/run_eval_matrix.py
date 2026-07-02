# src/evaluation/run_eval_matrix.py
# This script runs the master evaluation matrix over our search and RAG configurations.
# It evaluates combinations of:
#   - Chunking strategy (naive_fixed, structure_aware, semantic)
#   - Search mode (bm25_only, vector_only, hybrid)
#   - Reranking (none, cohere, bge)
#
# To save API cost and latency, generation-level metrics (LLM-judge, RAGAS) are optional
# via the --eval-generation flag. By default, it evaluates retrieval quality (Recall, Precision, MRR).
#
# USAGE:
#   python -m src.evaluation.run_eval_matrix

from __future__ import annotations # Allow self-referencing type annotations.
import asyncio # Standard library module to run async search.
import csv # Standard library module to export results.
import os # Standard library module for file management.
import time # Standard library module to measure latency.
from typing import Any, Dict, List, Optional # Type helpers.
import click # CLI tools.
import pandas as pd # Library for dataframe manipulation and markdown generation.
from rich.console import Console # Pretty terminal output.
from rich.table import Table # Grids.
from src.evaluation.eval_set import load_eval_set # Loader for evaluation set.
from src.evaluation.retrieval_metrics import calculate_mrr, calculate_precision_at_k, calculate_recall_at_k # Search metrics.
from src.evaluation.llm_judge import LLMJudge # Correctness judge.
from src.evaluation.ragas_wrapper import run_ragas_evaluation # RAGAS runner.
from src.retrieval.hybrid_search import hybrid_search # Unified search engine.
from src.reranking.cohere_reranker import CohereReranker # Cohere API reranker.
from src.reranking.bge_reranker import BGEReranker # Local CrossEncoder reranker.
from src.generation.context_packer import pack_context # Prompt context formatter.
from src.generation.llm_client import SECGenerator # LLM client wrapper.
from src.shared.config import cfg # Configuration settings loader.
from src.shared.logging_setup import configure_logging # Logging setup.
from src.shared.models import ChunkingStrategy, RerankerProvider, RetrievalMode # Shared models.

console = Console()

async def evaluate_config(
    strategy: ChunkingStrategy,
    mode: RetrievalMode,
    rerank_provider: RerankerProvider,
    eval_set: List[Any],
    eval_gen: bool,
    generator: SECGenerator,
    judge: LLMJudge,
    cohere_client: Optional[CohereReranker],
    bge_client: Optional[BGEReranker]
) -> Dict[str, Any]:
    """
    Evaluates a single pipeline configuration across all queries in the evaluation set.
    """
    retrieval_latencies = []
    generation_latencies = []
    
    # Lists to aggregate query-level metrics.
    recall_5 = []
    recall_10 = []
    recall_20 = []
    precision_5 = []
    precision_10 = []
    mrr_scores = []
    
    # Lists for generation logs if eval_gen is active.
    gen_records: List[Dict[str, Any]] = []
    judge_scores = []

    # Loop through each query in the evaluation set.
    for query_item in eval_set:
        # Build search filters. Include target strategy to isolate chunks.
        filters = {"strategy": strategy.value}
        if query_item.ticker:
            filters["ticker"] = query_item.ticker
            
        t_start = time.perf_counter()
        
        # Step 1: Run search retrieval.
        fused_hits = await hybrid_search(
            query=query_item.query,
            top_k=20,
            filters=filters,
            mode=mode.value
        )
        
        # Calculate retrieval elapsed time.
        retrieval_latencies.append(time.perf_counter() - t_start)

        # Extract search hit chunk IDs.
        retrieved_ids = [hit.chunk.chunk_id for hit in fused_hits]
        relevant_ids = query_item.relevant_chunk_ids or [] # Fallback if empty.

        # Compute search quality metrics.
        recall_5.append(calculate_recall_at_k(retrieved_ids, relevant_ids, k=5))
        recall_10.append(calculate_recall_at_k(retrieved_ids, relevant_ids, k=10))
        recall_20.append(calculate_recall_at_k(retrieved_ids, relevant_ids, k=20))
        precision_5.append(calculate_precision_at_k(retrieved_ids, relevant_ids, k=5))
        precision_10.append(calculate_precision_at_k(retrieved_ids, relevant_ids, k=10))
        mrr_scores.append(calculate_mrr(retrieved_ids, relevant_ids))

        # Step 2: If generation grading is active, execute RAG pipeline end-to-end.
        if eval_gen:
            # Apply reranking.
            if rerank_provider == RerankerProvider.COHERE:
                ranked = cohere_client.rerank(query_item.query, fused_hits, top_n=5)
            elif rerank_provider == RerankerProvider.BGE:
                ranked = bge_client.rerank(query_item.query, fused_hits, top_n=5)
            else:
                # Map RRF results directly if no reranker is selected.
                ranked = []
                for idx, hit in enumerate(fused_hits[:5], start=1):
                    # We map them using standard schema.
                    from src.shared.models import RankedResult
                    ranked.append(RankedResult(
                        chunk=hit.chunk,
                        rerank_score=hit.rrf_score,
                        final_rank=idx,
                        rrf_score=hit.rrf_score
                    ))

            # Pack context.
            packed_chunks = pack_context(ranked)
            
            t_gen_start = time.perf_counter()
            # Call generation.
            gen_out = generator.generate(query_item.query, packed_chunks)
            generation_latencies.append(time.perf_counter() - t_gen_start)

            # Grade answer correctness.
            judge_res = judge.grade_answer(
                query=query_item.query,
                answer=gen_out.answer,
                expected_answer=query_item.expected_answer
            )
            judge_scores.append(judge_res["score"])

            # Save records for RAGAS.
            gen_records.append({
                "question": query_item.query,
                "answer": gen_out.answer,
                "contexts": [c.packed_text for c in packed_chunks],
                "ground_truth": query_item.expected_answer
            })

    # Step 3: Compute average statistics.
    avg_recall_10 = sum(recall_10) / len(recall_10) if recall_10 else 0.0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_prec_10 = sum(precision_10) / len(precision_10) if precision_10 else 0.0
    
    avg_ret_lat = sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0.0
    avg_gen_lat = sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0.0

    stats = {
        "strategy": strategy.value,
        "mode": mode.value,
        "reranker": rerank_provider.value,
        "recall_at_5": sum(recall_5) / len(recall_5) if recall_5 else 0.0,
        "recall_at_10": avg_recall_10,
        "recall_at_20": sum(recall_20) / len(recall_20) if recall_20 else 0.0,
        "precision_at_5": sum(precision_5) / len(precision_5) if precision_5 else 0.0,
        "precision_at_10": avg_prec_10,
        "mrr": avg_mrr,
        "retrieval_latency_ms": avg_ret_lat * 1000,
        "generation_latency_ms": avg_gen_lat * 1000,
        "total_latency_ms": (avg_ret_lat + avg_gen_lat) * 1000
    }

    # Step 4: Run RAGAS if active.
    if eval_gen and gen_records:
        ragas_scores = run_ragas_evaluation(gen_records)
        if ragas_scores:
            stats.update(ragas_scores)
        stats["llm_judge_score"] = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

    return stats

@click.command()
@click.option("--eval-generation", is_flag=True, default=False, help="Run end-to-end LLM generation and grading.")
@click.option("--log-level", default="WARNING", help="Logging level.")
def main(eval_generation: bool, log_level: str) -> None:
    """
    Executes the master RAG pipeline configuration matrix evaluation.
    """
    configure_logging(log_level)
    
    # Load the evaluation queries.
    eval_set = load_eval_set()
    if not eval_set:
        console.print("[red]Evaluation set is empty. Put queries in data/eval/eval_set.jsonl first.[/red]")
        return
        
    console.print(
        f"\n[bold cyan]Master Evaluation Matrix Runner[/bold cyan]\n"
        f"  Total queries : {len(eval_set)}\n"
        f"  Eval generation: {eval_generation}\n"
    )

    # Initialize generators and judge clients if active.
    generator = SECGenerator() if eval_generation else None
    judge = LLMJudge() if eval_generation else None
    
    # Initialize rerankers.
    cohere_client = CohereReranker()
    bge_client = BGEReranker()

    results: List[Dict[str, Any]] = []

    # Loop through all configured dimensions.
    # Chunking strategies.
    strategies = [ChunkingStrategy.NAIVE_FIXED, ChunkingStrategy.STRUCTURE_AWARE, ChunkingStrategy.SEMANTIC]
    # Retrieval modes.
    modes = [RetrievalMode.BM25_ONLY, RetrievalMode.VECTOR_ONLY, RetrievalMode.HYBRID]
    # Reranking providers.
    rerankers = [RerankerProvider.NONE, RerankerProvider.COHERE, RerankerProvider.BGE]

    total_configs = len(strategies) * len(modes) * len(rerankers)
    config_idx = 1

    for strategy in strategies:
        for mode in modes:
            for reranker in rerankers:
                console.print(
                    f"[{config_idx}/{total_configs}] Evaluating: "
                    f"Strategy={strategy.value} | Mode={mode.value} | Reranker={reranker.value}..."
                )
                
                # Execute evaluation.
                try:
                    # Run the async evaluation loop using a temporary asyncio wrapper.
                    loop = asyncio.get_event_loop()
                    stats = loop.run_until_complete(
                        evaluate_config(
                            strategy=strategy,
                            mode=mode,
                            rerank_provider=reranker,
                            eval_set=eval_set,
                            eval_gen=eval_generation,
                            generator=generator,
                            judge=judge,
                            cohere_client=cohere_client,
                            bge_client=bge_client
                        )
                    )
                    results.append(stats)
                except Exception as exc:
                    console.print(f"  [red]Failed to evaluate config: {exc}[/red]")
                    
                config_idx += 1

    # ── Step 5: Save Results to CSV and Print Markdown Table ──
    results_dir = cfg.evaluation.results_dir
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, "matrix_results.csv")
    
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    
    console.print(f"\n[green]✓ Evaluation matrix completed. Saved results to {csv_file}.[/green]\n")

    # Render results table to terminal.
    # We display key metrics: strategy, mode, reranker, Recall@10, MRR, latency, and LLM-judge (if run).
    table = Table(title="Matrix Evaluation Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Strategy")
    table.add_column("Mode")
    table.add_column("Reranker")
    table.add_column("Recall@10", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Search Latency (ms)", justify="right")
    
    if eval_generation:
        table.add_column("LLM Judge", justify="right")
        table.add_column("Faithfulness", justify="right")

    # Sort results by Recall@10 descending.
    sorted_results = sorted(results, key=lambda x: x["recall_at_10"], reverse=True)

    for res in sorted_results:
        row_cells = [
            res["strategy"],
            res["mode"],
            res["reranker"],
            f"{res['recall_at_10']*100:.1f}%",
            f"{res['mrr']:.4f}",
            f"{res['retrieval_latency_ms']:.1f}"
        ]
        if eval_generation:
            row_cells.append(f"{res.get('llm_judge_score', 0.0):.2f}")
            row_cells.append(f"{res.get('faithfulness', 0.0)*100:.1f}%" if "faithfulness" in res else "-")
            
        table.add_row(*row_cells)

    console.print(table)

if __name__ == "__main__":
    main()
