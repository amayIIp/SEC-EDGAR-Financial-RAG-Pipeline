# src/parsing/run_parse.py
#
# CLI entry-point for Phase 1b — batch-parsing all raw HTML filings.
#
# USAGE EXAMPLES:
#   # Parse all downloaded filings in the manifest:
#   python -m src.parsing.run_parse
#
#   # Parse only AAPL filings:
#   python -m src.parsing.run_parse --ticker AAPL
#
#   # Parse only 10-K filings:
#   python -m src.parsing.run_parse --form 10-K
#
#   # Use 8 parallel worker processes (speeds up CPU-bound HTML parsing):
#   python -m src.parsing.run_parse --workers 8

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from src.ingestion.manifest import FilingManifest
from src.parsing.html_parser import parse_filing
from src.shared.config import cfg
from src.shared.logging_setup import configure_logging, get_logger
from src.shared.models import FilingMetadata, FilingType, ParsedFiling

log = get_logger(__name__)
console = Console()


# =============================================================================
# Worker function (called in subprocess by ProcessPoolExecutor)
# =============================================================================

def _parse_one(manifest_row: dict) -> tuple[str, Optional[str], Optional[str]]:
    """
    Parses a single filing from a manifest row dict.

    Returns:
        (accession_number, output_json_path, error_message)
        output_json_path is None on failure; error_message is None on success.

    This function runs in a subprocess so it cannot share in-memory state with
    the parent process.  All inputs come through the manifest_row dict.
    """
    accession = manifest_row["accession_number"]
    html_path = Path(manifest_row["local_path"])
    parsed_dir = Path(cfg.parsing.parsed_data_dir)

    # Construct the output JSON path.
    safe_form = manifest_row["form"].replace("/", "-")
    out_name = f"{safe_form}_{manifest_row['filing_date']}_{accession}.json"
    out_path = parsed_dir / manifest_row["ticker"] / out_name

    # Skip if already parsed (idempotent re-runs).
    if out_path.exists():
        return accession, str(out_path), None

    # Build the FilingMetadata object from the manifest row.
    try:
        metadata = FilingMetadata(
            ticker=manifest_row["ticker"],
            cik=manifest_row["cik"],
            filing_type=FilingType(manifest_row["form"]),
            filing_date=manifest_row["filing_date"],
            fiscal_year=int(manifest_row.get("fiscal_year", 0)),
            accession_number=accession,
            local_path=str(html_path),
        )
    except Exception as exc:
        return accession, None, f"metadata_error: {exc}"

    # Parse the HTML.
    parsed: Optional[ParsedFiling] = None
    try:
        parsed = parse_filing(html_path, metadata)
    except Exception as exc:
        return accession, None, f"parse_exception: {exc}"

    if parsed is None:
        return accession, None, "parse_returned_none"

    # Serialise to JSON and write to disk.
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Pydantic v2's model_dump() serialises all nested models to dicts.
        out_path.write_text(
            json.dumps(parsed.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        return accession, None, f"write_error: {exc}"

    return accession, str(out_path), None


# =============================================================================
# CLI entry-point
# =============================================================================

@click.command()
@click.option("--ticker", default=None, help="Restrict parsing to one ticker symbol.")
@click.option("--form", default=None, help="Restrict parsing to one form type (e.g. 10-K).")
@click.option(
    "--workers",
    default=cfg.parsing.parse_workers,
    show_default=True,
    type=int,
    help="Number of parallel worker processes.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Logging verbosity.",
)
def main(ticker: Optional[str], form: Optional[str], workers: int, log_level: str) -> None:
    """
    Parse all downloaded SEC HTML filings into structured JSON.

    Reads the manifest at data/raw/manifest.csv and processes every 'downloaded'
    entry.  Output JSON files are written to data/parsed/{TICKER}/{FORM}_{DATE}_{ACC}.json.
    """
    configure_logging(log_level)

    manifest = FilingManifest()
    rows = manifest.filter(ticker=ticker, form=form)

    if not rows:
        console.print("[yellow]No matching downloaded filings found in manifest.[/yellow]")
        sys.exit(0)

    console.print(
        f"\n[bold cyan]SEC Filing Parser[/bold cyan]\n"
        f"  Filing(s) to parse : {len(rows)}\n"
        f"  Output dir         : {cfg.parsing.parsed_data_dir}\n"
        f"  Workers            : {workers}\n"
    )

    parsed_count = 0
    skipped_count = 0
    failed: list[tuple[str, str]] = []   # (accession, error_message)

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing filings...", total=len(rows))

        if workers == 1:
            # Single-process path — easier to debug.
            for row in rows:
                accession, out_path, error = _parse_one(row)
                if error:
                    if "skip" in error.lower():
                        skipped_count += 1
                    else:
                        failed.append((accession, error))
                        log.error("parse_failed", accession=accession, error=error)
                else:
                    parsed_count += 1
                progress.advance(task)
        else:
            # Multi-process path for speed on large corpora.
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_parse_one, row): row for row in rows}
                for future in as_completed(futures):
                    try:
                        accession, out_path, error = future.result()
                        if error:
                            failed.append((accession, error))
                        else:
                            parsed_count += 1
                    except Exception as exc:
                        row = futures[future]
                        failed.append((row["accession_number"], str(exc)))
                    progress.advance(task)

    # Print summary.
    summary = Table(title="Parse Summary", border_style="dim")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_row("Filings processed", str(len(rows)))
    summary.add_row("Successfully parsed", f"[green]{parsed_count}[/green]")
    summary.add_row("Skipped (cached)", str(skipped_count))
    summary.add_row(
        "Failures",
        f"[red]{len(failed)}[/red]" if failed else "0",
    )
    console.print(summary)

    if failed:
        console.print("\n[bold red]Failed filings:[/bold red]")
        for acc, err in failed[:20]:
            console.print(f"  {acc}: {err}")
        if len(failed) > 20:
            console.print(f"  ... and {len(failed) - 20} more. Check logs for details.")
        sys.exit(1)

    console.print("\n[bold green]✓ Parsing complete.[/bold green]\n")


if __name__ == "__main__":
    main()
