# src/chunking/run_chunk.py
# This script runs our three chunking strategies over all parsed filings,
# saves the resulting chunks to data/chunks/{strategy}/chunks.jsonl,
# and prints summary statistics (counts, tokens, table percentages, latency).
#
# USAGE:
#   python -m src.chunking.run_chunk

from __future__ import annotations # Allow self-referencing type annotations.
import json # Standard library module to read and write JSON data.
import os # Standard library module for OS operations like creating directories.
import time # Standard library module to measure execution time.
import click # Third-party command-line interface library.
import numpy as np # Library for numerical calculations, used for percentiles.
from rich.console import Console # Pretty terminal output library.
from rich.table import Table # Output format grids.
from src.chunking.naive_fixed_chunker import NaiveFixedChunker # Naive fixed tokens chunker.
from src.chunking.structure_aware_chunker import StructureAwareChunker # Structure aware chunker.
from src.chunking.semantic_chunker import SemanticChunker # Semantic similarity-based chunker.
from src.shared.config import cfg # Configuration settings parser.
from src.shared.logging_setup import configure_logging # Setup logging system.
from src.shared.models import Chunk, ChunkingStrategy, ParsedFiling # Shared models.

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
    # Create the directory for this strategy if it doesn't exist yet.
    strategy_dir = os.path.join(output_dir, strategy.value)
    os.makedirs(strategy_dir, exist_ok=True)
    
    # Path to save the JSON lines (jsonl) output file.
    output_file = os.path.join(strategy_dir, "chunks.jsonl")
    
    all_chunks: List[Chunk] = [] # List to hold all generated Chunk objects.
    latencies: List[float] = [] # Track time spent per filing.
    
    # Loop through each parsed filing in the corpus.
    for filing in parsed_filings:
        start_time = time.perf_counter()
        
        # Split the filing using the chunker instance.
        chunks = chunker_instance.chunk(filing)
        
        # Calculate how long chunking took.
        elapsed = time.perf_counter() - start_time
        
        # Save results.
        all_chunks.extend(chunks)
        latencies.append(elapsed)

    # Write the chunks to the output JSONL file.
    with open(output_file, "w", encoding="utf-8") as f:
        # Loop through each chunk.
        for chunk in all_chunks:
            # Serialise Chunk model to JSON string and write as a line.
            f.write(json.dumps(chunk.model_dump()) + "\n")

    # --- Compute Statistics ---
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

# CLI command setup.
@click.command()
@click.option("--log-level", default="INFO", help="Logging level.")
def main(log_level: str) -> None:
    """
    Runs chunking experiments on parsed SEC filing files.
    """
    configure_logging(log_level)
    
    parsed_dir = cfg.parsing.parsed_data_dir
    output_dir = cfg.chunking.chunks_dir
    
    # ── Step 1: Read all parsed JSON filings ──
    parsed_filings: List[ParsedFiling] = []
    
    # Check if the parsed filings directory exists.
    if not os.path.exists(parsed_dir):
        console.print(f"[red]Parsed directory not found at {parsed_dir}. Please run html_parser first.[/red]")
        return
        
    # Traverse directories to find JSON filings.
    for root, _, files in os.walk(parsed_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                # Load the JSON content.
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert raw dict into our structured Pydantic object.
                    parsed_filings.append(ParsedFiling(**data))
                    
    # Log information if no filings are found.
    if not parsed_filings:
        console.print("[yellow]No parsed filings found to chunk.[/yellow]")
        return
        
    console.print(f"[green]Loaded {len(parsed_filings)} parsed filings for chunking experiments.[/green]\n")
    
    # ── Step 2: Initialize our Chunker instances ──
    chunkers = {
        ChunkingStrategy.NAIVE_FIXED: NaiveFixedChunker(),
        ChunkingStrategy.STRUCTURE_AWARE: StructureAwareChunker(),
        ChunkingStrategy.SEMANTIC: SemanticChunker()
    }
    
    results = []
    
    # ── Step 3: Run each chunker strategy and collect stats ──
    for strategy, chunker in chunkers.items():
        console.print(f"[cyan]Running chunker strategy: {strategy.value}...[/cyan]")
        stats = run_chunker_strategy(strategy, chunker, parsed_filings, output_dir)
        results.append(stats)
        
    # ── Step 4: Render results inside a Rich table ──
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
