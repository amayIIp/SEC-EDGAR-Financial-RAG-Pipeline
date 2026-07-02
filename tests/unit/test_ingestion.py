from __future__ import annotations 
from pathlib import Path 
from unittest.mock import MagicMock, patch 
import pytest 
import requests 
from src.ingestion.edgar_client import download_filing_html, fetch_filing_index 
from src.ingestion.run_ingest import _resolve_cik 
from src.ingestion.manifest import FilingManifest 
def test_filing_manifest_flow(tmp_path: Path) -> None:
    """
    Verifies that FilingManifest successfully adds entries, saves them to disk,
    and supports O(1) in-memory existence checks.
    """
    manifest_csv = tmp_path / "manifest.csv"
    manifest = FilingManifest(manifest_path=manifest_csv)
    assert not manifest.exists("0000000000-00-000000")
    manifest.add_entry(
        ticker="AAPL",
        cik="0000320193",
        form="10-K",
        filing_date="2023-09-30",
        fiscal_year=2023,
        accession_number="0000320193-23-000106",
        primary_doc="aapl-20230930.htm",
        local_path="data/raw/AAPL/10-K_2023-09-30.html",
        file_size_kb=105.5,
        status="downloaded"
    )
    assert manifest.exists("0000320193-23-000106")
    assert manifest_csv.exists(), "Manifest file was not written to disk."
    new_manifest = FilingManifest(manifest_path=manifest_csv)
    assert new_manifest.exists("0000320193-23-000106"), "Manifest entries did not persist."
@patch("src.ingestion.edgar_client._session.get") 
def test_resolve_cik(mock_get: MagicMock) -> None:
    """
    Verifies that _resolve_cik resolves a ticker into its CIK string using the SEC map.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."}
    }
    mock_get.return_value = mock_response
    cik = _resolve_cik("AAPL")
    assert cik == "0000320193"
    cik_msft = _resolve_cik("MSFT")
    assert cik_msft == "0000789019"
@patch("src.ingestion.edgar_client._session.get")
def test_fetch_filing_index(mock_get: MagicMock) -> None:
    """
    Verifies that fetch_filing_index requests company submissions and parses them.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "cik": "0000320193",
        "entityType": "operating",
        "sic": "3571",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "filings": {
            "recent": {
                "accessionNumber": ["acc-1", "acc-2"],
                "filingDate": ["2023-11-03", "2023-08-04"],
                "reportDate": ["2023-09-30", "2023-07-01"],
                "form": ["10-K", "10-Q"],
                "primaryDocument": ["a.htm", "b.htm"],
                "size": [1000, 2000]
            }
        }
    }
    mock_get.return_value = mock_response
    filings = fetch_filing_index(cik="0000320193", form_types=["10-K", "10-Q"], years_of_history=3)
    assert len(filings) == 2, "Expected exactly 2 filings to be resolved."
    assert filings[0]["accession_number"] == "acc-1"
    assert filings[0]["form"] == "10-K"
    assert filings[0]["fiscal_year"] == 2023
@patch("src.ingestion.edgar_client._session.get")
def test_download_filing_html(mock_get: MagicMock, tmp_path: Path) -> None:
    """
    Verifies that download_filing_html issues document download requests and writes to disk.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body>Financial statement content</body></html>"
    mock_get.return_value = mock_response
    dest = tmp_path / "filing.html"
    success = download_filing_html(
        cik="0000320193",
        accession_number="acc-1",
        primary_doc="a.htm",
        dest_path=dest
    )
    assert success
    assert dest.exists(), "Target filing file was not saved."
    assert "Financial statement content" in dest.read_text(), "Filing text is incomplete."
