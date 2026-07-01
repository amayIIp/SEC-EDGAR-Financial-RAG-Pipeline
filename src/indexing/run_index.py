# src/indexing/run_index.py
# This script loads text chunks produced in Phase 2, computes their vector embeddings,
# and indexes them into OpenSearch and Qdrant in batches.
#
# USAGE:
#   python -m src.indexing.run_index --strategy structure_aware --provider openai

from __future__ import annotations # Allow self-referencing type annotations.
import json # Standard library module for JSON deserialisation.
import os # Standard library module to construct file paths.
import time # Standard library module to measure execution time.
import click # Command-line argument parser.
from rich.console import Console # Pretty terminal output.
from src.indexing.embedder import SECEmbedder # Custom embedding generator.
from src.indexing.opensearch_index import OpenSearchIndexManager # OpenSearch BM25 manager.
from src.indexing.qdrant_index import QdrantIndexManager # Qdrant vector manager.
from src.shared.config import cfg # Config loader singleton.
from src.shared.logging_setup import configure_logging # Logging setup.
from src.shared.models import Chunk, ChunkingStrategy, EmbeddingProvider # Shared models.

console = Console()

@click.command()
@click.option(
    "--strategy",
    required=True,
    type=click.Choice([s.value for s in ChunkingStrategy]),
    help="Which chunking strategy output to index."
)
@click.option(
    "--provider",
    default=None,
    type=click.Choice([p.value for p in EmbeddingProvider]),
    help="Embedding provider: 'openai' or 'bge'. Defaults to config."
)
@click.option(
    "--force-reset",
    is_flag=True,
    default=False,
    help="If True, deletes and recreates collections/indices before indexing."
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Logging level."
)
def main(strategy: str, provider: str | None, force_reset: bool, log_level: str) -> None:
    """
    Loads chunked filings from jsonl, embeds them, and uploads them to Qdrant & OpenSearch.
    """
    configure_logging(log_level)
    
    # ── Step 1: Load Chunks from Disk ──
    chunks_file = os.path.join(cfg.chunking.chunks_dir, strategy, "chunks.jsonl")
    if not os.path.exists(chunks_file):
        console.print(f"[red]Chunks file not found at {chunks_file}. Run chunking runner first.[/red]")
        return
        
    console.print(f"[cyan]Loading chunks from {chunks_file}...[/cyan]")
    chunks: List[Chunk] = []
    
    # Read the file line-by-line.
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                # Convert raw JSON string into a structured Chunk object.
                chunks.append(Chunk(**json.loads(line)))
                
    if not chunks:
        console.print("[yellow]No chunks found to index.[/yellow]")
        return
        
    console.print(f"[green]Loaded {len(chunks):,} chunks successfully.[/green]")

    # ── Step 2: Initialize embedding generator and database managers ──
    embed_provider = provider or cfg.embedding.provider
    embedder = SECEmbedder(provider=embed_provider)
    dim = embedder.get_dimension()
    
    # Setup OpenSearch and Qdrant indexes.
    os_manager = OpenSearchIndexManager()
    qd_manager = QdrantIndexManager()
    
    # Initialize the databases (creates indices/collections).
    os_manager.create_index(force=force_reset)
    qd_manager.create_collection(dimension=dim, force=force_reset)

    # ── Step 3: Compute Embeddings and Index Chunks in Batches ──
    # We use Qdrant's configured batch size to limit memory consumption.
    batch_size = cfg.qdrant.upsert_batch_size
    total_indexed = 0
    start_time = time.perf_counter()
    
    console.print(f"[cyan]Embedding and indexing chunks into Qdrant & OpenSearch in batches of {batch_size}...[/cyan]")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_texts = [c.text for c in batch]
        
        # Calculate embeddings for the batch text.
        batch_vectors = embedder.embed(batch_texts)
        
        # Index into OpenSearch (BM25 keyword search).
        os_manager.bulk_index_chunks(batch)
        
        # Index into Qdrant (Dense vector search).
        qd_manager.bulk_upsert_chunks(batch, batch_vectors)
        
        total_indexed += len(batch)
        elapsed = time.perf_counter() - start_time
        throughput = total_indexed / elapsed if elapsed > 0 else 0.0
        
        console.print(
            f"  [dim]Indexed {total_indexed}/{len(chunks)} chunks... "
            f"({throughput:.1f} chunks/sec)[/dim]"
        )

    # Output final summary reports.
    total_time = time.perf_counter() - start_time
    console.print(
        f"\n[bold green]✓ Indexing complete![/bold green]\n"
        f"  Total Chunks : {total_indexed:,}\n"
        f"  Provider     : {embed_provider}\n"
        f"  Vector Dim   : {dim}\n"
        f"  Total Time   : {total_time:.2f} seconds\n"
        f"  Throughput   : {total_indexed / total_time:.1f} chunks/sec\n"
    )

if __name__ == "__main__":
    main()
