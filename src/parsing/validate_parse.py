# src/parsing/validate_parse.py
#
# Validation script for Phase 1b — sanity-checks the parsed JSON output.
#
# REPORTS:
#   1. Total filings parsed (JSON files found in data/parsed/).
#   2. Parse failure rate (manifest entries with no corresponding JSON).
#   3. Average / median / p95 sections per filing.
#   4. Average / median / p95 tables per filing.
#   5. Average total content character count per filing.
#   6. Three random parsed examples — printed for manual review.
#
# USAGE:
#   python -m src.parsing.validate_parse
#   python -m src.parsing.validate_parse --ticker AAPL --samples 5

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.ingestion.manifest import FilingManifest
from src.shared.config import cfg
from src.shared.logging_setup import configure_logging

console = Console()


def _load_parsed_json(json_path: Path) -> dict[str, Any] | None:
    """Loads and returns a parsed JSON file, or None if it fails."""
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Failed to load {json_path}: {exc}[/red]")
        return None


@click.command()
@click.option("--ticker", default=None, help="Filter to one ticker.")
@click.option("--samples", default=3, show_default=True, help="Number of random examples to print.")
@click.option("--log-level", default="WARNING", show_default=True)
def main(ticker: str | None, samples: int, log_level: str) -> None:
    """Validate parsed filing JSON output and report corpus statistics."""
    configure_logging(log_level)

    parsed_dir = Path(cfg.parsing.parsed_data_dir)
    if not parsed_dir.exists():
        console.print(f"[red]Parsed directory not found: {parsed_dir}[/red]")
        sys.exit(1)

    # ── 1. Gather all parsed JSON files ──────────────────────────────────────
    if ticker:
        json_files = sorted((parsed_dir / ticker.upper()).glob("*.json"))
    else:
        json_files = sorted(parsed_dir.glob("**/*.json"))

    if not json_files:
        console.print("[yellow]No parsed JSON files found.[/yellow]")
        sys.exit(0)

    console.print(f"\n[bold cyan]Parse Validation[/bold cyan] — {len(json_files)} JSON files\n")

    # ── 2. Compute statistics ─────────────────────────────────────────────────
    section_counts: list[int] = []
    table_counts: list[int] = []
    char_counts: list[int] = []
    tickers_seen: set[str] = set()
    forms_seen: dict[str, int] = {}
    all_parsed: list[dict[str, Any]] = []

    for json_path in json_files:
        data = _load_parsed_json(json_path)
        if data is None:
            continue

        all_parsed.append(data)

        sections = data.get("sections", [])
        section_counts.append(len(sections))
        # Count total tables across all sections.
        n_tables = sum(len(s.get("tables", [])) for s in sections)
        table_counts.append(n_tables)
        char_counts.append(data.get("total_content_chars", 0))

        meta = data.get("metadata", {})
        tickers_seen.add(meta.get("ticker", "?"))
        form = meta.get("filing_type", "?")
        forms_seen[form] = forms_seen.get(form, 0) + 1

    # ── 3. Failure rate ───────────────────────────────────────────────────────
    manifest = FilingManifest()
    downloaded_count = len(manifest.all_downloaded())
    parsed_count = len(all_parsed)
    failure_rate = (
        (downloaded_count - parsed_count) / downloaded_count * 100
        if downloaded_count > 0 else 0.0
    )

    # ── 4. Print statistics table ─────────────────────────────────────────────
    def _pct(arr: list[int], p: int) -> str:
        return f"{float(np.percentile(arr, p)):.1f}" if arr else "N/A"

    stats_table = Table(title="Corpus Statistics", border_style="dim", show_header=True)
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", justify="right")

    stats_table.add_row("Total parsed filings", str(parsed_count))
    stats_table.add_row("Manifest downloaded entries", str(downloaded_count))
    stats_table.add_row(
        "Parse failure rate",
        f"[{'red' if failure_rate > 5 else 'green'}]{failure_rate:.1f}%[/]",
    )
    stats_table.add_row("Unique tickers", str(len(tickers_seen)))
    for form, count in sorted(forms_seen.items()):
        stats_table.add_row(f"  {form} filings", str(count))

    stats_table.add_section()
    stats_table.add_row("Sections/filing — mean", _pct(section_counts, 50))
    stats_table.add_row("Sections/filing — median (p50)", _pct(section_counts, 50))
    stats_table.add_row("Sections/filing — p95", _pct(section_counts, 95))

    stats_table.add_section()
    stats_table.add_row("Tables/filing — mean", f"{float(np.mean(table_counts)):.1f}" if table_counts else "N/A")
    stats_table.add_row("Tables/filing — median (p50)", _pct(table_counts, 50))
    stats_table.add_row("Tables/filing — p95", _pct(table_counts, 95))

    stats_table.add_section()
    stats_table.add_row(
        "Content chars/filing — mean",
        f"{float(np.mean(char_counts)):,.0f}" if char_counts else "N/A",
    )
    stats_table.add_row(
        "Content chars/filing — p50",
        f"{float(np.percentile(char_counts, 50)):,.0f}" if char_counts else "N/A",
    )
    stats_table.add_row(
        "Content chars/filing — p95",
        f"{float(np.percentile(char_counts, 95)):,.0f}" if char_counts else "N/A",
    )

    console.print(stats_table)

    # ── 5. Random sample display ──────────────────────────────────────────────
    if not all_parsed:
        return

    console.print(f"\n[bold]Random Sample — {min(samples, len(all_parsed))} filing(s)[/bold]\n")

    sample = random.sample(all_parsed, min(samples, len(all_parsed)))

    for filing_data in sample:
        meta = filing_data.get("metadata", {})
        sections = filing_data.get("sections", [])

        header = (
            f"[bold]{meta.get('ticker')} | {meta.get('filing_type')} | "
            f"{meta.get('filing_date')}[/bold]  "
            f"({len(sections)} sections, "
            f"{sum(len(s.get('tables',[])) for s in sections)} tables)"
        )

        # Show the first two sections as a preview.
        body_lines: list[str] = []
        for sec in sections[:2]:
            item_label = f"[cyan]{sec['item_id']}[/cyan] {sec.get('title', '')}"
            # Truncate section content to 300 chars for readability.
            preview = sec.get("content", "")[:300].replace("\n", " ")
            if len(sec.get("content", "")) > 300:
                preview += "…"
            body_lines.append(f"{item_label}\n{preview}")

            # Show first table caption if available.
            if sec.get("tables"):
                cap = sec["tables"][0].get("caption", "(no caption)")
                body_lines.append(f"  [dim]↳ Table: {cap[:80]}[/dim]")

        if len(sections) > 2:
            body_lines.append(f"[dim]… {len(sections) - 2} more sections[/dim]")

        console.print(Panel("\n\n".join(body_lines), title=header, border_style="blue"))
        console.print()


if __name__ == "__main__":
    main()
