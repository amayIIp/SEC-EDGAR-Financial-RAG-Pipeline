from __future__ import annotations 
import json 
import os 
import time 
import click 
import numpy as np 
from rich.console import Console 
from rich.table import Table 
from src.chunking.naive_fixed_chunker import NaiveFixedChunker 
from src.chunking.structure_aware_chunker import StructureAwareChunker 
from src.chunking.semantic_chunker import SemanticChunker 
from src.shared.config import cfg 
from src.shared.logging_setup import configure_logging 
from src.shared.models import Chunk, ChunkingStrategy, ParsedFiling 
console = Console()
def run_chunker_strategy(
    strategy: ChunkingStrategy,
    chunker_instance: Any,
    parsed_filings: List[ParsedFiling],
    output_dir: str
) -> dict[str, Any]:
    """
    Executes a chunking strategy over the parsed filings corpus, writes the output,
    and returns calculated statistics.
    """
    strategy_dir = os.path.join(output_dir, strategy.value)
    os.makedirs(strategy_dir, exist_ok=True)
    output_file = os.path.join(strategy_dir, "chunks.jsonl")
    all_chunks: List[Chunk] = [] 
    latencies: List[float] = [] 
    for filing in parsed_filings:
        start_time = time.perf_counter()
        chunks = chunker_instance.chunk(filing)
        elapsed = time.perf_counter() - start_time
        all_chunks.extend(chunks)
        latencies.append(elapsed)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.model_dump()) + "\n")
    token_counts = [chunk.token_count for chunk in all_chunks]
    num_tables = sum(1 for chunk in all_chunks if chunk.is_table)
    table_pct = (num_tables / len(all_chunks)) * 100 if all_chunks else 0.0
    avg_tokens = np.mean(token_counts) if token_counts else 0.0
    median_tokens = np.median(token_counts) if token_counts else 0.0
    p95_tokens = np.percentile(token_counts, 95) if token_counts else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0
    return {
        "strategy": strategy.value,
        "total_chunks": len(all_chunks),
        "avg_tokens": avg_tokens,
        "median_tokens": median_tokens,
        "p95_tokens": p95_tokens,
        "table_percentage": table_pct,
        "avg_latency_ms": avg_latency * 1000,
        "output_file": output_file
    }
@click.command()
@click.option("--log-level", default="INFO", help="Logging level.")
def main(log_level: str) -> None:
    """
    Runs chunking experiments on parsed SEC filing files.
    """
    configure_logging(log_level)
    parsed_dir = cfg.parsing.parsed_data_dir
    output_dir = cfg.chunking.chunks_dir
    parsed_filings: List[ParsedFiling] = []
    if not os.path.exists(parsed_dir):
        console.print(f"[red]Parsed directory not found at {parsed_dir}. Please run html_parser first.[/red]")
        return
    for root, _, files in os.walk(parsed_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    parsed_filings.append(ParsedFiling(**data))
    if not parsed_filings:
        console.print("[yellow]No parsed filings found to chunk.[/yellow]")
        return
    console.print(f"[green]Loaded {len(parsed_filings)} parsed filings for chunking experiments.[/green]\n")
    chunkers = {
        ChunkingStrategy.NAIVE_FIXED: NaiveFixedChunker(),
        ChunkingStrategy.STRUCTURE_AWARE: StructureAwareChunker(),
        ChunkingStrategy.SEMANTIC: SemanticChunker()
    }
    results = []
    for strategy, chunker in chunkers.items():
        console.print(f"[cyan]Running chunker strategy: {strategy.value}...[/cyan]")
        stats = run_chunker_strategy(strategy, chunker, parsed_filings, output_dir)
        results.append(stats)
    table = Table(title="Chunking Strategy Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Strategy")
    table.add_column("Total Chunks", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Median (p50)", justify="right")
    table.add_column("p95 Tokens", justify="right")
    table.add_column("Table %", justify="right")
    table.add_column("Latency / filing (ms)", justify="right")
    for res in results:
        table.add_row(
            res["strategy"],
            f"{res['total_chunks']:,}",
            f"{res['avg_tokens']:.1f}",
            f"{res['median_tokens']:.1f}",
            f"{res['p95_tokens']:.1f}",
            f"{res['table_percentage']:.1f}%",
            f"{res['avg_latency_ms']:.2f}"
        )
    console.print(table)
    console.print("\n[green]✓ Chunking experiments run completed successfully.[/green]\n")
if __name__ == "__main__":
    main()
