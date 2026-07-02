# src/ingestion/run_ingest.py
#
# CLI entry-point for Phase 1a — downloading SEC filings.
#
# USAGE EXAMPLES:
#   # Download default ticker list from config.yaml (all configured tickers):
#   python -m src.ingestion.run_ingest
#
#   # Download a specific subset of tickers:
#   python -m src.ingestion.run_ingest --tickers AAPL MSFT NVDA
#
#   # Download only 10-K filings, last 2 years:
#   python -m src.ingestion.run_ingest --forms 10-K --years 2
#
#   # Force re-download even if files already exist:
#   python -m src.ingestion.run_ingest --no-skip-existing
#
#   # Dry-run: show what would be downloaded without hitting EDGAR:
#   python -m src.ingestion.run_ingest --dry-run

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.ingestion.edgar_client import download_filing_html, fetch_filing_index
from src.ingestion.manifest import FilingManifest
from src.shared.config import cfg
from src.shared.logging_setup import configure_logging, get_logger

log = get_logger(__name__)
console = Console()


# =============================================================================
# Core ingestion logic
# =============================================================================

def ingest_ticker(
    ticker: str,
    forms: list[str],
    years: int,
    raw_dir: Path,
    manifest: FilingManifest,
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, int]:
    """
    Downloads all requested filings for a single ticker.

    Returns a counter dict with keys:
        downloaded, skipped, failed
    """
    # Look up the CIK for this ticker from EDGAR's company search.
    # We fetch the submissions JSON first; it contains the CIK in its header.
    from src.ingestion.edgar_client import fetch_submissions

    # Resolve CIK from the ticker — EDGAR's full-text search returns it.
    # The simplest approach: attempt to read it from the submissions endpoint
    # for a ticker string. If the ticker isn't found, log and skip.
    cik = _resolve_cik(ticker)
    if cik is None:
        log.warning("cik_not_found", ticker=ticker)
        console.print(f"  [yellow]⚠ CIK not found for ticker {ticker} — skipping.[/yellow]")
        return {"downloaded": 0, "skipped": 0, "failed": 1}

    # Fetch the list of filing metadata for this CIK.
    try:
        filings = fetch_filing_index(cik=cik, form_types=forms, years_of_history=years)
    except Exception as exc:
        log.error("fetch_filing_index_failed", ticker=ticker, cik=cik, error=str(exc))
        console.print(f"  [red]✗ Failed to fetch filing index for {ticker}: {exc}[/red]")
        return {"downloaded": 0, "skipped": 0, "failed": 1}

    counters = {"downloaded": 0, "skipped": 0, "failed": 0}

    for filing in filings:
        accession = filing["accession_number"]

        # Check manifest for existing successful downloads.
        if skip_existing and manifest.exists(accession):
            log.debug("filing_skipped_exists", accession=accession)
            counters["skipped"] += 1
            continue

        # Construct the local file path for this filing.
        safe_form = filing["form"].replace("/", "-")   # "10-K/A" → "10-K-A"
        filename = f"{safe_form}_{filing['filing_date']}_{accession}.html"
        dest_path = raw_dir / ticker.upper() / filename

        if dry_run:
            console.print(
                f"  [dim][DRY-RUN] Would download: {ticker}/{filename}[/dim]"
            )
            counters["downloaded"] += 1
            continue

        # Download the filing HTML.
        success = download_filing_html(
            cik=cik,
            accession_number=accession,
            primary_doc=filing["primary_doc"],
            dest_path=dest_path,
        )

        if success:
            # Measure the saved file size for the manifest.
            file_size_kb = dest_path.stat().st_size / 1024 if dest_path.exists() else 0.0
            manifest.add_entry(
                ticker=ticker.upper(),
                cik=cik,
                form=filing["form"],
                filing_date=filing["filing_date"],
                fiscal_year=filing["fiscal_year"],
                accession_number=accession,
                primary_doc=filing["primary_doc"],
                local_path=str(dest_path),
                file_size_kb=file_size_kb,
                status="downloaded",
            )
            counters["downloaded"] += 1
        else:
            # Record the failure in the manifest so we know to retry later.
            manifest.add_entry(
                ticker=ticker.upper(),
                cik=cik,
                form=filing["form"],
                filing_date=filing["filing_date"],
                fiscal_year=filing["fiscal_year"],
                accession_number=accession,
                primary_doc=filing["primary_doc"],
                local_path="",
                file_size_kb=0.0,
                status="failed",
            )
            counters["failed"] += 1

    return counters


def _resolve_cik(ticker: str) -> Optional[str]:
    """
    Resolves a stock ticker to a zero-padded 10-digit CIK using EDGAR's
    company tickers JSON endpoint.

    The endpoint at https://www.sec.gov/files/company_tickers.json returns
    a dict of all registered companies with their CIK and ticker symbol.
    We download this once and cache it in memory for the session.
    """
    # Use a module-level cache so we only download the tickers JSON once per run.
    if not hasattr(_resolve_cik, "_cache"):
        _resolve_cik._cache = _build_ticker_cik_map()   # type: ignore[attr-defined]

    return _resolve_cik._cache.get(ticker.upper())   # type: ignore[attr-defined]


def _build_ticker_cik_map() -> dict[str, str]:
    """
    Downloads the SEC's complete company-tickers JSON and builds a dict
    mapping uppercase ticker symbol → zero-padded 10-digit CIK string.

    The JSON structure is:
        { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ... }
    """
    import requests as req

    url = "https://www.sec.gov/files/company_tickers.json"
    log.info("downloading_ticker_map", url=url)

    try:
        resp = req.get(url, headers={"User-Agent": cfg.ingestion.user_agent}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.error("ticker_map_download_failed", error=str(exc))
        return {}

    mapping: dict[str, str] = {}
    for entry in data.values():
        ticker_sym = str(entry.get("ticker", "")).upper()
        cik_int = entry.get("cik_str")
        if ticker_sym and cik_int is not None:
            # Zero-pad to 10 digits to match EDGAR submission URL format.
            mapping[ticker_sym] = str(cik_int).zfill(10)

    log.info("ticker_map_built", num_tickers=len(mapping))
    return mapping


# =============================================================================
# CLI entry-point
# =============================================================================

@click.command()
@click.option(
    "--tickers",
    multiple=True,
    default=None,
    help="Ticker symbols to ingest. Defaults to the list in config.yaml.",
)
@click.option(
    "--forms",
    multiple=True,
    default=None,
    help="SEC form types to download. Defaults to config.yaml form_types.",
)
@click.option(
    "--years",
    default=None,
    type=int,
    help="Years of filing history to retrieve. Defaults to config.yaml years_of_history.",
)
@click.option(
    "--no-skip-existing",
    "no_skip",
    is_flag=True,
    default=False,
    help="Re-download filings even if they already exist in data/raw/.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be downloaded without hitting EDGAR.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Logging verbosity (DEBUG | INFO | WARNING | ERROR).",
)
def main(
    tickers: tuple[str, ...],
    forms: tuple[str, ...],
    years: Optional[int],
    no_skip: bool,
    dry_run: bool,
    log_level: str,
) -> None:
    """
    Download 10-K, 10-Q, and 8-K filings for configured tickers from SEC EDGAR.

    Files are saved to data/raw/{TICKER}/{FORM}_{DATE}_{ACCESSION}.html.
    Progress is tracked in the manifest CSV at data/raw/manifest.csv.
    """
    configure_logging(log_level)

    # Resolve CLI overrides vs config.yaml defaults.
    ticker_list: list[str] = list(tickers) if tickers else cfg.ingestion.tickers
    form_list: list[str] = list(forms) if forms else cfg.ingestion.form_types
    years_int: int = years if years is not None else cfg.ingestion.years_of_history
    skip_existing: bool = not no_skip and cfg.ingestion.skip_existing
    raw_dir = Path(cfg.ingestion.raw_data_dir)

    console.print(
        f"\n[bold cyan]SEC EDGAR Ingestion[/bold cyan]\n"
        f"  Tickers : {', '.join(ticker_list)}\n"
        f"  Forms   : {', '.join(form_list)}\n"
        f"  Years   : {years_int}\n"
        f"  Dest    : {raw_dir}\n"
        f"  Skip existing: {skip_existing}\n"
        f"  Dry-run : {dry_run}\n"
    )

    manifest = FilingManifest()

    # Grand totals across all tickers.
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    # Progress bar with rich columns for a clean terminal display.
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,   # keep the completed bars visible after finish
    ) as progress:
        task = progress.add_task("Ingesting tickers...", total=len(ticker_list))

        for ticker in ticker_list:
            progress.update(task, description=f"[bold blue]{ticker}...")
            counters = ingest_ticker(
                ticker=ticker,
                forms=form_list,
                years=years_int,
                raw_dir=raw_dir,
                manifest=manifest,
                skip_existing=skip_existing,
                dry_run=dry_run,
            )
            total_downloaded += counters["downloaded"]
            total_skipped += counters["skipped"]
            total_failed += counters["failed"]
            progress.advance(task)

    # Print final summary table.
    summary_table = Table(title="Ingestion Summary", border_style="dim", show_header=True)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("Tickers processed", str(len(ticker_list)))
    summary_table.add_row("Filings downloaded", f"[green]{total_downloaded}[/green]")
    summary_table.add_row("Filings skipped (cached)", str(total_skipped))
    summary_table.add_row("Failures", f"[red]{total_failed}[/red]" if total_failed else "0")
    summary_table.add_row("Manifest entries", str(len(manifest)))
    console.print(summary_table)

    if total_failed > 0:
        console.print(
            f"\n[yellow]⚠  {total_failed} filing(s) failed. "
            "Re-run with --no-skip-existing to retry them.[/yellow]"
        )
        sys.exit(1)

    console.print("\n[bold green]✓ Ingestion complete.[/bold green]\n")


if __name__ == "__main__":
    main()
