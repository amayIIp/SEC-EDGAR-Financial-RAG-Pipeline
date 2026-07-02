from __future__ import annotations 
import os 
import click 
from rich.console import Console 
from rich.table import Table 
from opensearchpy import OpenSearch 
from qdrant_client import QdrantClient 
from src.indexing.embedder import SECEmbedder 
from src.shared.config import cfg 
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
    console.print("[cyan]Testing OpenSearch (BM25 Lexical Index)...[/cyan]")
    os_client = OpenSearch(
        hosts=[{"host": cfg.opensearch.host, "port": cfg.opensearch.port}],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False
    )
    if not os_client.indices.exists(index=cfg.opensearch.index_name):
        console.print(f"[red]OpenSearch index '{cfg.opensearch.index_name}' does not exist. Run indexing first.[/red]")
        return
    os_body = {
        "query": {
            "match": {
                "text": query
            }
        },
        "size": 5
    }
    os_response = os_client.search(index=cfg.opensearch.index_name, body=os_body)
    os_hits = os_response["hits"]["hits"]
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
    console.print("[cyan]Testing Qdrant (Dense Vector Index)...[/cyan]")
    qd_client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
    existing_collections = qd_client.get_collections().collections
    collection_exists = any(c.name == cfg.qdrant.collection_name for c in existing_collections)
    if not collection_exists:
        console.print(f"[red]Qdrant collection '{cfg.qdrant.collection_name}' does not exist. Run indexing first.[/red]")
        return
    embedder = SECEmbedder()
    console.print(f"Generating query embedding vector using '{embedder.provider}'...")
    query_vectors = embedder.embed([query])
    query_vector = query_vectors[0]
    qd_hits = qd_client.search(
        collection_name=cfg.qdrant.collection_name,
        query_vector=query_vector,
        limit=5
    )
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
