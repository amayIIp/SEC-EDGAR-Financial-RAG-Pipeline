# src/retrieval/run_search_cli.py
# This script is a command-line utility to query our retrieval engine.
# It takes a query and a search mode, calls hybrid_search, and prints
# the top results in a formatting table showing original and fused ranks.
#
# USAGE:
#   python -m src.retrieval.run_search_cli --query "revenue growth" --mode hybrid

from __future__ import annotations # Allow self-referencing type annotations.
import asyncio # Standard library module to run async tasks.
from typing import Optional # Type helper.
import click # Command-line argument parser.
from rich.console import Console # Pretty terminal output.
from rich.table import Table # Table grids.
from src.retrieval.hybrid_search import hybrid_search # Unified search interface.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import configure_logging # Logging setup.
from src.shared.models import RetrievalMode # Shared models.

console = Console()

async def run_search(query: str, mode: str, ticker: Optional[str]) -> None:
    """
    Executes search and prints results.
    """
    # Construct filters dict if a ticker was specified.
    filters = {"ticker": ticker} if ticker else None
    
    # Run the query.
    results = await hybrid_search(query, top_k=10, filters=filters, mode=mode)
    
    # Render results table.
    table = Table(
        title=f"Retrieval Results (Mode: {mode})",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Rank", justify="center")
    table.add_column("Ticker", justify="center")
    table.add_column("Form", justify="center")
    table.add_column("Section")
    table.add_column("RRF Score", justify="right")
    table.add_column("BM25 Rank", justify="center")
    table.add_column("Vec Rank", justify="center")
    table.add_column("Text Snippet", width=60)
    
    for idx, hit in enumerate(results, start=1):
        # Format columns.
        bm_rank_str = str(hit.bm25_rank) if hit.bm25_rank is not None else "-"
        vc_rank_str = str(hit.vector_rank) if hit.vector_rank is not None else "-"
        text_snippet = hit.chunk.text[:100].replace("\n", " ") + "..."
        
        table.add_row(
            str(idx),
            hit.chunk.ticker,
            hit.chunk.filing_type.value,
            hit.chunk.section,
            f"{hit.rrf_score:.4f}",
            bm_rank_str,
            vc_rank_str,
            text_snippet
        )
        
    console.print(table)

@click.command()
@click.option("--query", required=True, help="Query text string to search.")
@click.option(
    "--mode",
    default=cfg.retrieval.mode,
    type=click.Choice([m.value for m in RetrievalMode]),
    show_default=True,
    help="Search mode: hybrid, bm25_only, or vector_only."
)
@click.option("--ticker", default=None, help="Optional stock ticker filter.")
@click.option("--log-level", default="WARNING", help="Logging level.")
def main(query: str, mode: str, ticker: str | None, log_level: str) -> None:
    """
    Runs a CLI query test against our search indices.
    """
    configure_logging(log_level)
    # Execute the async search routine using asyncio.run.
    asyncio.run(run_search(query, mode, ticker))

if __name__ == "__main__":
    main()
