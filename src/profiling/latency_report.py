from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import click          
import numpy as np   
import orjson        
from rich.console import Console  
from rich.table import Table      
from src.shared.config import cfg
from src.shared.logging_setup import configure_logging, get_logger
log = get_logger(__name__)
console = Console()
def load_profile_records(log_path: Path) -> list[dict[str, Any]]:
    """
    Reads a single JSON-lines profile log file and returns a list of
    raw record dicts.  Skips lines that fail to parse rather than crashing.
    """
    records: list[dict[str, Any]] = []
    for line_num, raw_line in enumerate(log_path.read_bytes().splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            records.append(orjson.loads(raw_line))
        except orjson.JSONDecodeError as exc:
            log.warning(
                "profile_parse_error",
                file=str(log_path),
                line=line_num,
                error=str(exc),
            )
    return records
def compute_stage_percentiles(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Groups all records by stage_name and computes p50 / p95 / p99 / mean
    of duration_ms for each stage.
    Returns a dict like:
        {
          "bm25_search":  {"p50": 42.1, "p95": 98.3, "p99": 210.0, "mean": 51.2, "n": 50},
          "vector_search": {...},
          ...
        }
    """
    stage_durations: dict[str, list[float]] = defaultdict(list)
    for record in records:
        stage = record.get("stage_name", "unknown")
        duration = record.get("duration_ms")
        if duration is not None:
            stage_durations[stage].append(float(duration))
    stats: dict[str, dict[str, float]] = {}
    for stage, durations in stage_durations.items():
        arr = np.array(durations, dtype=np.float64)
        stats[stage] = {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "n": int(len(arr)),
            "total_ms": float(np.sum(arr)),
        }
    return stats
def identify_bottleneck(stats: dict[str, dict[str, float]]) -> str:
    """
    Returns the name of the stage that contributes most to total wall time
    (highest sum of duration_ms across all queries).
    This is the stage that should be optimised first.
    """
    if not stats:
        return "unknown"
    return max(stats, key=lambda s: stats[s]["total_ms"])
def render_table(stats: dict[str, dict[str, float]], title: str) -> None:
    """
    Prints a rich-formatted table of latency percentiles to the terminal.
    Columns: Stage | N | p50 ms | p95 ms | p99 ms | Mean ms | Total ms | % of total
    """
    overall_total = sum(v["total_ms"] for v in stats.values()) or 1.0
    table = Table(
        title=title,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Stage", style="bold white", no_wrap=True)
    table.add_column("N", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("p99 ms", justify="right")
    table.add_column("Mean ms", justify="right")
    table.add_column("Total s", justify="right")
    table.add_column("% of Total", justify="right")
    sorted_stages = sorted(stats.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)
    for stage, s in sorted_stages:
        pct = (s["total_ms"] / overall_total) * 100
        pct_str = f"{pct:.1f}%"
        if pct > 40:
            pct_str = f"[red]{pct_str}[/red]"
        elif pct > 20:
            pct_str = f"[yellow]{pct_str}[/yellow]"
        else:
            pct_str = f"[green]{pct_str}[/green]"
        table.add_row(
            stage,
            str(s["n"]),
            f"{s['p50']:.1f}",
            f"{s['p95']:.1f}",
            f"{s['p99']:.1f}",
            f"{s['mean']:.1f}",
            f"{s['total_ms'] / 1000:.2f}",
            pct_str,
        )
    console.print(table)
    bottleneck = identify_bottleneck(stats)
    console.print(
        f"\n[bold yellow]⚠  Bottleneck:[/bold yellow] "
        f"[bold]{bottleneck}[/bold] accounts for the most total wall time. "
        "Optimise this stage first.\n"
    )
@click.command()
@click.option(
    "--run-id",
    default=None,
    help="First 8 chars of the run_id to load (e.g. 'abc12345'). "
         "Loads data/profiles/run_abc12345.jsonl.",
)
@click.option(
    "--all",
    "load_all",
    is_flag=True,
    default=False,
    help="Aggregate ALL profile log files in the profiles directory.",
)
@click.option(
    "--log-level",
    default="WARNING",
    show_default=True,
    help="Logging verbosity (DEBUG | INFO | WARNING | ERROR).",
)
def main(run_id: str | None, load_all: bool, log_level: str) -> None:
    """
    Compute and display per-stage latency percentiles from profile log files.
    Examples:\n
      python -m src.profiling.latency_report --run-id abc12345\n
      python -m src.profiling.latency_report --all
    """
    configure_logging(log_level)
    profiles_dir = Path(cfg.profiling.profiles_dir)
    if not profiles_dir.exists():
        console.print(
            f"[red]Profiles directory '{profiles_dir}' does not exist.[/red]\n"
            "Run the pipeline with profiling enabled first."
        )
        sys.exit(1)
    if load_all:
        log_files = sorted(profiles_dir.glob("run_*.jsonl"))
        if not log_files:
            console.print("[red]No profile log files found.[/red]")
            sys.exit(1)
        title = f"Latency Report — All Runs ({len(log_files)} files)"
    elif run_id:
        short_id = run_id.replace("run_", "")[:8]
        log_file = profiles_dir / f"run_{short_id}.jsonl"
        if not log_file.exists():
            console.print(f"[red]Profile log not found: {log_file}[/red]")
            sys.exit(1)
        log_files = [log_file]
        title = f"Latency Report — Run {short_id}"
    else:
        all_files = sorted(profiles_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not all_files:
            console.print("[red]No profile log files found.[/red]")
            sys.exit(1)
        log_files = [all_files[-1]]
        title = f"Latency Report — Latest Run ({log_files[0].name})"
    all_records: list[dict[str, Any]] = []
    for lf in log_files:
        all_records.extend(load_profile_records(lf))
    if not all_records:
        console.print("[red]No valid profile records found in selected files.[/red]")
        sys.exit(1)
    console.print(f"\n[bold]Loaded {len(all_records)} profile records.[/bold]\n")
    stats = compute_stage_percentiles(all_records)
    render_table(stats, title)
if __name__ == "__main__":
    main()
