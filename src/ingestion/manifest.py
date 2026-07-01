# src/ingestion/manifest.py
#
# Manifest CSV tracker — records every downloaded filing so that:
#   1. Re-running ingestion skips files that already exist (idempotent).
#   2. The parser knows exactly which raw files exist and their metadata.
#   3. We can audit what we have without scanning the filesystem.
#
# SCHEMA (one row per downloaded filing):
#   ticker, cik, form, filing_date, fiscal_year, accession_number,
#   primary_doc, local_path, downloaded_at, file_size_kb, status
#
# STATUS VALUES:
#   "downloaded" — HTML file exists at local_path.
#   "failed"     — Download was attempted but failed; local_path may be absent.
#   "skipped"    — File already existed; download was not attempted.

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from src.shared.config import cfg
from src.shared.logging_setup import get_logger

log = get_logger(__name__)

# CSV column order — must match the dict keys used in add_entry().
_FIELDNAMES = [
    "ticker",
    "cik",
    "form",
    "filing_date",
    "fiscal_year",
    "accession_number",
    "primary_doc",
    "local_path",
    "downloaded_at",
    "file_size_kb",
    "status",
]


class FilingManifest:
    """
    Reads and writes the filing manifest CSV at cfg.ingestion.manifest_path.
    Provides O(1) existence checks via an in-memory index keyed on accession_number.
    """

    def __init__(self, manifest_path: Optional[str | Path] = None) -> None:
        self._path = Path(manifest_path or cfg.ingestion.manifest_path)
        # In-memory index: accession_number → row dict.
        # Loaded from disk on construction; kept in sync with every write.
        self._index: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Reads the existing CSV into the in-memory index. No-op if file is absent."""
        if not self._path.exists():
            log.info("manifest_not_found", path=str(self._path), action="creating_new")
            return

        with self._path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self._index[row["accession_number"]] = row

        log.info("manifest_loaded", path=str(self._path), entries=len(self._index))

    def _write_all(self) -> None:
        """
        Rewrites the entire CSV from the in-memory index.
        We rewrite rather than append so the file stays consistent after failures.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._index.values())

    def exists(self, accession_number: str) -> bool:
        """Returns True if this accession number has a 'downloaded' entry."""
        entry = self._index.get(accession_number)
        return entry is not None and entry.get("status") == "downloaded"

    def add_entry(
        self,
        *,
        ticker: str,
        cik: str,
        form: str,
        filing_date: str,
        fiscal_year: int,
        accession_number: str,
        primary_doc: str,
        local_path: str,
        file_size_kb: float,
        status: str,
    ) -> None:
        """
        Adds or updates a filing entry in the manifest and flushes to disk.
        If the accession_number already exists, the row is overwritten (e.g.
        status changes from 'failed' to 'downloaded' on a retry).
        """
        self._index[accession_number] = {
            "ticker": ticker,
            "cik": cik,
            "form": form,
            "filing_date": filing_date,
            "fiscal_year": fiscal_year,
            "accession_number": accession_number,
            "primary_doc": primary_doc,
            "local_path": local_path,
            # ISO-8601 timestamp for when this row was written.
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file_size_kb": round(file_size_kb, 2),
            "status": status,
        }
        # Flush after every write — this is not a hot path (100s of rows, not millions).
        self._write_all()
        log.debug(
            "manifest_entry_added",
            accession=accession_number,
            status=status,
        )

    def get_entry(self, accession_number: str) -> Optional[dict[str, Any]]:
        """Returns the manifest row for an accession number, or None."""
        return self._index.get(accession_number)

    def all_downloaded(self) -> list[dict[str, Any]]:
        """Returns every row with status='downloaded', sorted by filing_date desc."""
        rows = [r for r in self._index.values() if r.get("status") == "downloaded"]
        return sorted(rows, key=lambda r: r["filing_date"], reverse=True)

    def filter(
        self,
        ticker: Optional[str] = None,
        form: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Returns downloaded entries matching the optional ticker and form filters.
        Useful for the parser to build a per-ticker, per-form processing queue.
        """
        results = self.all_downloaded()
        if ticker:
            results = [r for r in results if r["ticker"] == ticker.upper()]
        if form:
            results = [r for r in results if r["form"] == form]
        return results

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterates over all entries (any status), sorted by filing_date desc."""
        return iter(
            sorted(self._index.values(), key=lambda r: r["filing_date"], reverse=True)
        )

    def __len__(self) -> int:
        return len(self._index)

    def summary(self) -> dict[str, Any]:
        """Returns a summary dict for logging / CLI status display."""
        statuses: dict[str, int] = {}
        for row in self._index.values():
            s = row.get("status", "unknown")
            statuses[s] = statuses.get(s, 0) + 1
        return {
            "total_entries": len(self._index),
            "by_status": statuses,
            "manifest_path": str(self._path),
        }
