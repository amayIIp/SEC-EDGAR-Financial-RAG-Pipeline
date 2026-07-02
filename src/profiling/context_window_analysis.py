from __future__ import annotations 
import asyncio 
import click 
from rich.console import Console 
from rich.table import Table 
from src.evaluation.eval_set import load_eval_set 
from src.retrieval.hybrid_search import hybrid_search 
from src.reranking.cohere_reranker import CohereReranker 
from src.generation.context_packer import pack_context 
from src.generation.context_compressor import ContextCompressor 
from src.generation.llm_client import SECGenerator 
from src.shared.config import cfg 
from src.shared.logging_setup import configure_logging 
from src.shared.models import CitedChunk 
console = Console()
async def analyze_query_utilization(
    query_item: Any,
    generator: SECGenerator,
    reranker: CohereReranker,
    compressor: ContextCompressor,
    use_compression: bool
) -> dict[str, Any]:
    """
    Runs search, packs context (with optional compression), calls generation,
    and computes detailed token utilization statistics.
    """
    fused_hits = await hybrid_search(query_item.query, top_k=15, filters={"strategy": "structure_aware"})
    ranked_hits = reranker.rerank(query_item.query, fused_hits, top_n=5)
    packed_chunks = pack_context(ranked_hits)
    if use_compression:
        compressed_chunks = []
        for chunk in packed_chunks:
            compressed_text = compressor.compress(chunk.packed_text)
            c_tokens = len(generator.encoder.encode(compressed_text))
            compressed_chunk = CitedChunk(
                ranked_result=chunk.ranked_result,
                citation_tag=chunk.citation_tag,
                packed_text=compressed_text,
                packed_token_count=c_tokens
            )
            compressed_chunks.append(compressed_chunk)
        active_chunks = compressed_chunks
    else:
        active_chunks = packed_chunks
    total_packed_tokens = sum(c.packed_token_count for c in active_chunks)
    gen_out = generator.generate(query_item.query, active_chunks)
    cited_tokens = 0
    for chunk in active_chunks:
        if chunk.citation_tag in gen_out.cited_tags:
            cited_tokens += chunk.packed_token_count
    util_ratio = cited_tokens / total_packed_tokens if total_packed_tokens > 0 else 0.0
    return {
        "query_id": query_item.id,
        "packed_tokens": total_packed_tokens,
        "cited_tokens": cited_tokens,
        "utilization_ratio": util_ratio,
        "answer_len": len(gen_out.answer)
    }
async def run_analysis() -> None:
    """
    Runs utilization analysis on the seed eval set with compression toggled on and off.
    """
    eval_set = load_eval_set()[:3] 
    generator = SECGenerator()
    reranker = CohereReranker()
    compressor = ContextCompressor()
    console.print("[cyan]Analyzing context utilization without compression (Full Context)...[/cyan]")
    baseline_results = []
    for item in eval_set:
        res = await analyze_query_utilization(item, generator, reranker, compressor, use_compression=False)
        baseline_results.append(res)
    console.print("\n[cyan]Analyzing context utilization with compression (Compressed Context)...[/cyan]")
    compressed_results = []
    for item in eval_set:
        res = await analyze_query_utilization(item, generator, reranker, compressor, use_compression=True)
        compressed_results.append(res)
    table = Table(title="Context Window Utilization Comparison", show_header=True, header_style="bold green")
    table.add_column("Q_ID", justify="center")
    table.add_column("Full: Packed (Tok)", justify="right")
    table.add_column("Full: Cited (Tok)", justify="right")
    table.add_column("Full: Util %", justify="right")
    table.add_column("Comp: Packed (Tok)", justify="right")
    table.add_column("Comp: Cited (Tok)", justify="right")
    table.add_column("Comp: Util %", justify="right")
    for i in range(len(eval_set)):
        base = baseline_results[i]
        comp = compressed_results[i]
        table.add_row(
            base["query_id"],
            f"{base['packed_tokens']:,}",
            f"{base['cited_tokens']:,}",
            f"{base['utilization_ratio']*100:.1f}%",
            f"{comp['packed_tokens']:,}",
            f"{comp['cited_tokens']:,}",
            f"{comp['utilization_ratio']*100:.1f}%"
        )
    console.print(table)
    avg_base_util = sum(r["utilization_ratio"] for r in baseline_results) / len(baseline_results)
    avg_comp_util = sum(r["utilization_ratio"] for r in compressed_results) / len(compressed_results)
    console.print(f"\n📊 [bold]Average Utilization (Full Context):[/bold] {avg_base_util*100:.1f}%")
    console.print(f"📊 [bold]Average Utilization (Compressed Context):[/bold] {avg_comp_util*100:.1f}%")
    console.print("\n[green]✓ Context window utilization analysis completed.[/green]\n")
@click.command()
@click.option("--log-level", default="WARNING", help="Logging level.")
def main(log_level: str) -> None:
    """
    Evaluates context utilization differences with and without text compression.
    """
    configure_logging(log_level)
    asyncio.run(run_analysis())
if __name__ == "__main__":
    main()
