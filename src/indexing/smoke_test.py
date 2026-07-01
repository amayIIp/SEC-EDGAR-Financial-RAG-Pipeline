# src/indexing/smoke_test.py
# This script is a diagnostic verification test to confirm that both Qdrant
# (vector search) and OpenSearch (keyword BM25 search) are correctly populated
# and responding to query requests.
#
# USAGE:
#   python -m src.indexing.smoke_test

from __future__ import annotations # Allow self-referencing type annotations.
import os # Standard library module for checking environmental keys.
import click # CLI tools.
from rich.console import Console # Formatting printer.
from rich.table import Table # Output grid formatter.
from opensearchpy import OpenSearch # OpenSearch client.
from qdrant_client import QdrantClient # Qdrant client.
from src.indexing.embedder import SECEmbedder # Embeddings generator.
from src.shared.config import cfg # Config settings loader.

console = Console()

@click.command()
@click.option("--query", default="research and development expense", help="Smoke test query text.")
def main(query: str) -> None:
    """
    Runs a test keyword query against OpenSearch and a vector query against Qdrant,
    printing the top-5 raw results to verify indexing health.
    """
    console.print(f"[bold cyan]Starting Search Infrastructure Smoke Test[/bold cyan]\n")
    console.print(f"Test Query: '{query}'\n")

    # ── Step 1: Smoke Test OpenSearch (BM25) ──
    console.print("[cyan]Testing OpenSearch (BM25 Lexical Index)...[/cyan]")
    
    # Connect to OpenSearch.
    os_client = OpenSearch(
        hosts=[{"host": cfg.opensearch.host, "port": cfg.opensearch.port}],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False
    )
    
    # Check if index exists.
    if not os_client.indices.exists(index=cfg.opensearch.index_name):
        console.print(f"[red]OpenSearch index '{cfg.opensearch.index_name}' does not exist. Run indexing first.[/red]")
        return

    # Build match query body.
    os_body = {
        "query": {
            "match": {
                "text": query
            }
        },
        "size": 5
    }
    
    # Run the query.
    os_response = os_client.search(index=cfg.opensearch.index_name, body=os_body)
    os_hits = os_response["hits"]["hits"]
    
    # Display OpenSearch results.
    os_table = Table(title="OpenSearch BM25 Top-5 Hits", show_header=True, header_style="bold green")
    os_table.add_column("Rank", justify="center")
    os_table.add_column("Chunk ID")
    os_table.add_column("Score", justify="right")
    os_table.add_column("Text Snippet", width=80)
    
    for rank, hit in enumerate(os_hits, start=1):
        source = hit["_source"]
        text_snippet = source.get("text", "")[:120].replace("\n", " ") + "..."
        os_table.add_row(
            str(rank),
            source.get("chunk_id", "?")[:12],
            f"{hit['_score']:.4f}",
            text_snippet
        )
    console.print(os_table)
    console.print()

    # ── Step 2: Smoke Test Qdrant (Vector Store) ──
    console.print("[cyan]Testing Qdrant (Dense Vector Index)...[/cyan]")
    
    # Connect to Qdrant.
    qd_client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
    
    # Check if the collection name is in the list of existing collections returned by Qdrant.
    existing_collections = qd_client.get_collections().collections
    # Check if any collection matches the configured collection name.
    collection_exists = any(c.name == cfg.qdrant.collection_name for c in existing_collections)

    # Check if collection exists.
    if not collection_exists:
        console.print(f"[red]Qdrant collection '{cfg.qdrant.collection_name}' does not exist. Run indexing first.[/red]")
        return
        
    # Initialize embedder to translate the text query into a vector representation.
    embedder = SECEmbedder()
    console.print(f"Generating query embedding vector using '{embedder.provider}'...")
    query_vectors = embedder.embed([query])
    query_vector = query_vectors[0]
    
    # Execute the nearest-neighbor search.
    qd_hits = qd_client.search(
        # Specify the target Qdrant collection.
        collection_name=cfg.qdrant.collection_name,
        # Pass the calculated query vector.
        query_vector=query_vector,
        # Limit the results to top-5 nearest neighbors.
        limit=5
    )
    
    # Display Qdrant results.
    qd_table = Table(title="Qdrant Cosine Similarity Top-5 Hits", show_header=True, header_style="bold blue")
    qd_table.add_column("Rank", justify="center")
    qd_table.add_column("Chunk ID")
    qd_table.add_column("Score (Cosine)", justify="right")
    qd_table.add_column("Text Snippet", width=80)
    
    for rank, hit in enumerate(qd_hits, start=1):
        payload = hit.payload or {}
        text_snippet = payload.get("text", "")[:120].replace("\n", " ") + "..."
        qd_table.add_row(
            str(rank),
            payload.get("chunk_id", "?")[:12],
            f"{hit.score:.4f}",
            text_snippet
        )
    console.print(qd_table)
    console.print("\n[bold green]✓ Search infrastructure smoke test completed successfully.[/bold green]\n")

if __name__ == "__main__":
    main()
