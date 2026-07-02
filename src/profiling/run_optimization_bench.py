# src/profiling/run_optimization_bench.py
# This script executes A/B benchmark tests comparing our baseline RAG pipeline
# against multiple optimization configurations:
#   1. Query Caching (Exact + Semantic caches using cache.py)
#   2. Reduced Candidate Pool (retrieving 15 items instead of 50 before rerank)
#
# It outputs a consolidated comparison table displaying p50/p95 latency and Recall@10.
#
# USAGE:
#   python -m src.profiling.run_optimization_bench

from __future__ import annotations # Allow self-referencing type annotations.
import asyncio # Standard library module to run async tasks.
import time # Standard library module to measure latency.
from typing import Any, Dict, List # Type helpers.
import click # CLI tools.
import numpy as np # Percentile calculations.
import pandas as pd # Dataframe and markdown formatting.
from rich.console import Console # Formatting.
from rich.table import Table # Grids.
from src.evaluation.eval_set import load_eval_set # Loader for evaluation queries.
from src.evaluation.retrieval_metrics import calculate_recall_at_k # Recall metric.
from src.retrieval.hybrid_search import hybrid_search # Search interface.
from src.profiling.cache import QueryCache # Two-tier cache system.
from src.indexing.embedder import SECEmbedder # Embedder.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import configure_logging # Logging setup.

console = Console()

async def run_baseline_bench(eval_set: List[Any], embedder: SECEmbedder) -> dict[str, Any]:
    """
    Runs the baseline search (depth 50, no cache) and records latencies and Recall@10.
    """
    latencies = []
    recalls = []
    
    for item in eval_set:
        filters = {"strategy": "structure_aware"}
        if item.ticker:
            filters["ticker"] = item.ticker

        t_start = time.perf_counter()
        
        # Execute query.
        fused = await hybrid_search(item.query, top_k=10, filters=filters, depth=50)
        
        # Record time.
        latencies.append(time.perf_counter() - t_start)
        
        retrieved_ids = [hit.chunk.chunk_id for hit in fused]
        relevant_ids = item.relevant_chunk_ids or []
        recalls.append(calculate_recall_at_k(retrieved_ids, relevant_ids, k=10))

    return {
        "optimization": "Baseline (No cache, depth 50)",
        "p50_ms": np.percentile(latencies, 50) * 1000,
        "p95_ms": np.percentile(latencies, 95) * 1000,
        "recall_at_10": np.mean(recalls)
    }

async def run_cached_bench(eval_set: List[Any], embedder: SECEmbedder) -> dict[str, Any]:
    """
    Runs search with two-tier query cache active.
    We run the loop TWICE. The second iteration hits the cache, demonstrating latency delta.
    """
    cache = QueryCache()
    latencies = []
    recalls = []

    # Iteration 1: Populate cache.
    for item in eval_set:
        filters = {"strategy": "structure_aware"}
        if item.ticker:
            filters["ticker"] = item.ticker
        query_vector = embedder.embed([item.query])[0]
        # Query and cache.
        fused = await hybrid_search(item.query, top_k=10, filters=filters)
        cache.set(item.query, np.array(query_vector), fused)

    # Iteration 2: Measure hit latency.
    for item in eval_set:
        t_start = time.perf_counter()
        query_vector = embedder.embed([item.query])[0]
        
        # Check cache first.
        cached = cache.get(item.query, np.array(query_vector))
        if cached is None:
            # Fallback on miss (should not occur in exact-match test).
            cached = await hybrid_search(item.query, top_k=10, filters={"strategy": "structure_aware"})
            
        latencies.append(time.perf_counter() - t_start)
        
        retrieved_ids = [hit.chunk.chunk_id for hit in cached]
        relevant_ids = item.relevant_chunk_ids or []
        recalls.append(calculate_recall_at_k(retrieved_ids, relevant_ids, k=10))

    return {
        "optimization": "Caching Active (Second run hits)",
        "p50_ms": np.percentile(latencies, 50) * 1000,
        "p95_ms": np.percentile(latencies, 95) * 1000,
        "recall_at_10": np.mean(recalls)
    }

async def run_reduced_pool_bench(eval_set: List[Any], embedder: SECEmbedder) -> dict[str, Any]:
    """
    Runs search with a reduced candidate pool depth (e.g. 15 instead of 50) to measure speedup.
    """
    latencies = []
    recalls = []
    
    for item in eval_set:
        filters = {"strategy": "structure_aware"}
        if item.ticker:
            filters["ticker"] = item.ticker

        t_start = time.perf_counter()
        
        # Execute query with depth=15.
        fused = await hybrid_search(item.query, top_k=10, filters=filters, depth=15)
        
        latencies.append(time.perf_counter() - t_start)
        
        retrieved_ids = [hit.chunk.chunk_id for hit in fused]
        relevant_ids = item.relevant_chunk_ids or []
        recalls.append(calculate_recall_at_k(retrieved_ids, relevant_ids, k=10))

    return {
        "optimization": "Reduced Pool Depth (depth 15)",
        "p50_ms": np.percentile(latencies, 50) * 1000,
        "p95_ms": np.percentile(latencies, 95) * 1000,
        "recall_at_10": np.mean(recalls)
    }

async def execute_benchmarks() -> None:
    """
    Runs all A/B tests and prints comparison metrics.
    """
    eval_set = load_eval_set()
    embedder = SECEmbedder()
    
    console.print("[cyan]Running baseline benchmark...[/cyan]")
    baseline = await run_baseline_bench(eval_set, embedder)
    
    console.print("[cyan]Running caching benchmark...[/cyan]")
    cached = await run_cached_bench(eval_set, embedder)
    
    console.print("[cyan]Running reduced pool benchmark...[/cyan]")
    reduced = await run_reduced_pool_bench(eval_set, embedder)

    # Compile results.
    results = [baseline, cached, reduced]
    
    # Render table.
    table = Table(title="Optimization Benchmarks Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Applied Optimization")
    table.add_column("p50 Latency (ms)", justify="right")
    table.add_column("p95 Latency (ms)", justify="right")
    table.add_column("Recall@10", justify="right")
    
    for res in results:
        table.add_row(
            res["optimization"],
            f"{res['p50_ms']:.1f} ms",
            f"{res['p95_ms']:.1f} ms",
            f"{res['recall_at_10']*100:.1f}%"
        )
        
    console.print(table)
    
    # Save comparison to results directory.
    df = pd.DataFrame(results)
    output_path = "data/eval/results/optimization_benchmarks.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"\n[green]✓ Optimization benchmarks saved to {output_path}.[/green]\n")

@click.command()
@click.option("--log-level", default="WARNING", help="Logging level.")
def main(log_level: str) -> None:
    """
    Runs A/B profiling benchmarks for caching and search depth parameters.
    """
    configure_logging(log_level)
    asyncio.run(execute_benchmarks())

if __name__ == "__main__":
    main()
