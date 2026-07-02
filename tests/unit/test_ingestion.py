# tests/unit/test_ingestion.py
# This module implements unit tests for the ingestion components (FilingManifest and EDGAR Client).
# We verify that our manifest CSV database correctly logs and indexes successfully downloaded
# files, and that the EDGAR client handles submissions fetching and rate limiting.

from __future__ import annotations # Allow self-referencing type annotations.
from pathlib import Path # Path manipulation.
from unittest.mock import MagicMock, patch # Mocking tools.
import pytest # Testing framework.
import requests # Requests client.
from src.ingestion.edgar_client import download_filing_html, fetch_filing_index # Subjects under test.
from src.ingestion.run_ingest import _resolve_cik # Helper to resolve ticker CIK.
from src.ingestion.manifest import FilingManifest # Subject under test.


def test_filing_manifest_flow(tmp_path: Path) -> None:
    """
    Verifies that FilingManifest successfully adds entries, saves them to disk,
    and supports O(1) in-memory existence checks.
    """
    # Define a temporary manifest file path.
    manifest_csv = tmp_path / "manifest.csv"
    
    # Initialize manifest with the temporary file path.
    manifest = FilingManifest(manifest_path=manifest_csv)
    
    # Check that a mock accession number does not exist.
    assert not manifest.exists("0000000000-00-000000")
    
    # Add a mock entry.
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
    
    # Verify that the entry now exists in the manifest database.
    assert manifest.exists("0000320193-23-000106")
    
    # Verify that the file was written to disk.
    assert manifest_csv.exists(), "Manifest file was not written to disk."
    
    # Reload the manifest from disk into a new instance to test persistence.
    new_manifest = FilingManifest(manifest_path=manifest_csv)
    assert new_manifest.exists("0000320193-23-000106"), "Manifest entries did not persist."


@patch("src.ingestion.edgar_client._session.get") # Mock outbound HTTP calls.
def test_resolve_cik(mock_get: MagicMock) -> None:
    """
    Verifies that _resolve_cik resolves a ticker into its CIK string using the SEC map.
    """
    # Create mock response containing the SEC ticker CIK map JSON.
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."}
    }
    mock_get.return_value = mock_response
    
    # Resolve CIK for ticker AAPL.
    cik = _resolve_cik("AAPL")
    # Verify it was padded to 10 characters.
    assert cik == "0000320193"
    
    # Resolve CIK for MSFT.
    cik_msft = _resolve_cik("MSFT")
    assert cik_msft == "0000789019"


@patch("src.ingestion.edgar_client._session.get")
def test_fetch_filing_index(mock_get: MagicMock) -> None:
    """
    Verifies that fetch_filing_index requests company submissions and parses them.
    """
    # Create mock response for CIK submissions JSON.
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
    
    # Execute index fetching.
    filings = fetch_filing_index(cik="0000320193", form_types=["10-K", "10-Q"], years_of_history=3)
    
    # Assertions on returned metadata.
    assert len(filings) == 2, "Expected exactly 2 filings to be resolved."
    assert filings[0]["accession_number"] == "acc-1"
    assert filings[0]["form"] == "10-K"
    assert filings[0]["fiscal_year"] == 2023


@patch("src.ingestion.edgar_client._session.get")
def test_download_filing_html(mock_get: MagicMock, tmp_path: Path) -> None:
    """
    Verifies that download_filing_html issues document download requests and writes to disk.
    """
    # Create mock HTML page response.
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body>Financial statement content</body></html>"
    mock_get.return_value = mock_response
    
    # Define destination path.
    dest = tmp_path / "filing.html"
    
    # Run download.
    success = download_filing_html(
        cik="0000320193",
        accession_number="acc-1",
        primary_doc="a.htm",
        dest_path=dest
    )
    
    # Check that download returned success.
    assert success
    # Check that file was created with correct text.
    assert dest.exists(), "Target filing file was not saved."
    assert "Financial statement content" in dest.read_text(), "Filing text is incomplete."
