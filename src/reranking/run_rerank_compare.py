# src/reranking/run_rerank_compare.py
# This script executes test queries, runs hybrid search, and compares the
# resulting rankings before and after applying Cohere and BGE rerankers.
# It computes rank churn to quantify how much reranking alters results order.
#
# USAGE:
#   python -m src.reranking.run_rerank_compare

from __future__ import annotations # Allow self-referencing type annotations.
import asyncio # Standard library module to run async tasks.
import click # CLI tools.
from rich.console import Console # Pretty terminal output.
from rich.table import Table # Table grids.
from src.retrieval.hybrid_search import hybrid_search # Hybrid search coordinator.
from src.reranking.cohere_reranker import CohereReranker # Cohere API reranker.
from src.reranking.bge_reranker import BGEReranker # Local CrossEncoder reranker.
from src.shared.config import cfg # Configuration settings loader.
from src.shared.logging_setup import configure_logging # Logging setup.

console = Console()

# Sample test queries to run the comparison.
TEST_QUERIES = [
    "What was Apple's total net revenue for fiscal year 2023?",
    "What is Tesla's primary business activity as described in Item 1?"
]

def calculate_rank_churn(original: List[str], reranked: List[str]) -> int:
    """
    Computes how many items in the reranked top-8 changed index positions
    or are completely new compared to the original top-8.
    """
    churn = 0
    # Loop through each item and its rank position in the reranked list.
    for idx, item_id in enumerate(reranked):
        # If the item was not in the original list or changed its index order,
        # increment the churn counter.
        if item_id not in original or original.index(item_id) != idx:
            churn += 1
    return churn

async def compare_rankings(queries: List[str]) -> None:
    """
    Queries hybrid search and compares post-rerank rankings side-by-side.
    """
    # Initialize the rerankers.
    cohere_reranker = CohereReranker()
    bge_reranker = BGEReranker()
    
    top_n = cfg.reranking.top_n # Default: top 8.

    for query in queries:
        console.print(f"\n[bold yellow]Query: '{query}'[/bold yellow]\n")

        # Step 1: Run hybrid search to pull pre-rerank candidates.
        fused_hits = await hybrid_search(query, top_k=cfg.retrieval.pre_rerank_top_k)
        if not fused_hits:
            console.print("[red]No search results returned. Make sure files are indexed.[/red]")
            continue

        # Extract top fused chunk IDs.
        original_top_8 = [hit.chunk.chunk_id for hit in fused_hits[:top_n]]

        # Step 2: Run Cohere Rerank API.
        cohere_res = cohere_reranker.rerank(query, fused_hits, top_n)
        cohere_top_8 = [hit.chunk.chunk_id for hit in cohere_res]

        # Step 3: Run local BGE Rerank.
        bge_res = bge_reranker.rerank(query, fused_hits, top_n)
        bge_top_8 = [hit.chunk.chunk_id for hit in bge_res]

        # Step 4: Compute rank churn metrics.
        cohere_churn = calculate_rank_churn(original_top_8, cohere_top_8)
        bge_churn = calculate_rank_churn(original_top_8, bge_top_8)

        # Step 5: Render comparison table.
        table = Table(title="Ranking Comparison (Top 8 Chunks)", show_header=True, header_style="bold cyan")
        table.add_column("Pos", justify="center")
        table.add_column("Original RRF Fused (Snippet)")
        table.add_column("Cohere Reranked (Snippet)")
        table.add_column("BGE Reranked (Snippet)")
        
        # Populate table rows.
        for idx in range(top_n):
            orig_text = fused_hits[idx].chunk.text[:40].replace("\n", " ") + "..." if idx < len(fused_hits) else "-"
            cohere_text = cohere_res[idx].chunk.text[:40].replace("\n", " ") + "..." if idx < len(cohere_res) else "-"
            bge_text = bge_res[idx].chunk.text[:40].replace("\n", " ") + "..." if idx < len(bge_res) else "-"
            
            table.add_row(
                str(idx + 1),
                orig_text,
                cohere_text,
                bge_text
            )
            
        console.print(table)
        console.print(f"📈 [bold]Cohere Rank Churn (vs RRF):[/bold] {cohere_churn} / {top_n}")
        console.print(f"🤖 [bold]BGE Rank Churn (vs RRF):[/bold] {bge_churn} / {top_n}")
        console.print("-" * 80)

@click.command()
@click.option("--log-level", default="WARNING", help="Logging level.")
def main(log_level: str) -> None:
    """
    Compares hybrid RRF ranking against Cohere and local BGE reranked orders.
    """
    configure_logging(log_level)
    # Run the comparison loop.
    asyncio.run(compare_rankings(TEST_QUERIES))

if __name__ == "__main__":
    main()
