from __future__ import annotations 
import asyncio 
import time 
from typing import Any, Dict, List 
import click 
import numpy as np 
import pandas as pd 
from rich.console import Console 
from rich.table import Table 
from src.evaluation.eval_set import load_eval_set 
from src.evaluation.retrieval_metrics import calculate_recall_at_k 
from src.retrieval.hybrid_search import hybrid_search 
from src.profiling.cache import QueryCache 
from src.indexing.embedder import SECEmbedder 
from src.shared.config import cfg 
from src.shared.logging_setup import configure_logging 
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
        fused = await hybrid_search(item.query, top_k=10, filters=filters, depth=50)
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
    for item in eval_set:
        filters = {"strategy": "structure_aware"}
        if item.ticker:
            filters["ticker"] = item.ticker
        query_vector = embedder.embed([item.query])[0]
        fused = await hybrid_search(item.query, top_k=10, filters=filters)
        cache.set(item.query, np.array(query_vector), fused)
    for item in eval_set:
        t_start = time.perf_counter()
        query_vector = embedder.embed([item.query])[0]
        cached = cache.get(item.query, np.array(query_vector))
        if cached is None:
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
    results = [baseline, cached, reduced]
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
