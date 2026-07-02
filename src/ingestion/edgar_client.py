from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import requests                         
from tenacity import (                  
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    RetryError,
)
from src.shared.config import cfg
from src.shared.logging_setup import get_logger
log = get_logger(__name__)
_INTER_REQUEST_SLEEP: float = 1.0 / cfg.ingestion.requests_per_second
_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
_session = requests.Session()
_session.headers.update({
    "User-Agent": cfg.ingestion.user_agent,
    "Accept": "application/json, text/html, */*",
})
_last_request_time: float = 0.0
def _rate_limited_get(url: str, **kwargs: Any) -> requests.Response:
    """
    Sends an HTTP GET request to the SEC EDGAR servers, sleeping if necessary
    to stay within the 10 req/s rate limit before each call.
    This is intentionally synchronous. EDGAR's rate limit is per-IP-address,
    not per-connection, so async concurrency would just trigger 429 errors.
    Serialising requests and sleeping between them is the correct approach.
    """
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    sleep_needed = _INTER_REQUEST_SLEEP - elapsed
    if sleep_needed > 0:
        time.sleep(sleep_needed)
    _last_request_time = time.monotonic()
    response = _session.get(url, timeout=30, **kwargs)
    return response
def _is_retryable(exc: BaseException) -> bool:
    """
    Returns True if the exception represents a transient server-side error
    that is worth retrying (5xx responses, connection errors, timeouts).
    Returns False for client-side errors (4xx) that won't improve on retry.
    """
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    reraise=True,   
)
def _get_with_retry(url: str) -> requests.Response:
    """GET with automatic exponential-back-off retry for transient errors."""
    response = _rate_limited_get(url)
    response.raise_for_status()
    return response
def fetch_submissions(cik: str) -> dict[str, Any]:
    """
    Fetches the submissions JSON for a company from the EDGAR Submissions API.
    The submissions JSON contains metadata about the company plus a 'filings'
    sub-object with parallel arrays: form[], filingDate[], accessionNumber[],
    primaryDocument[].  We zip these arrays in fetch_filing_index().
    Args:
        cik: The company's Central Index Key.  Will be zero-padded to 10 digits.
    Returns:
        The parsed JSON dict from EDGAR.
    Raises:
        requests.HTTPError: if the server returns a non-retryable error status.
        RetryError: if all retry attempts are exhausted.
    """
    padded_cik = str(cik).zfill(10)
    url = f"{_SUBMISSIONS_BASE}/CIK{padded_cik}.json"
    log.info("edgar_fetch_submissions", cik=padded_cik, url=url)
    try:
        response = _get_with_retry(url)
        return response.json()
    except RetryError as exc:
        log.error("edgar_submissions_failed", cik=padded_cik, error=str(exc))
        raise
def fetch_filing_index(
    cik: str,
    form_types: list[str],
    years_of_history: int,
) -> list[dict[str, Any]]:
    """
    Returns a list of filing metadata dicts for the specified form types,
    filtered to the last `years_of_history` calendar years.
    Each dict contains:
        cik, ticker (filled by caller), form, filing_date,
        accession_number, primary_doc, fiscal_year.
    Args:
        cik:              Company CIK (zero-padded to 10 digits internally).
        form_types:       List of form type strings to include, e.g. ["10-K", "10-Q"].
        years_of_history: Keep only filings with a filing_date within this many years.
    Returns:
        List of filing dicts sorted by filing_date descending (newest first).
    """
    import datetime
    submissions = fetch_submissions(cik)
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    cutoff_date = datetime.date.today() - datetime.timedelta(days=years_of_history * 365)
    filings: list[dict[str, Any]] = []
    for form, date_str, accession, primary_doc in zip(forms, dates, accessions, primary_docs):
        if form not in form_types:
            continue
        try:
            filing_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            log.warning("edgar_bad_date", date_str=date_str, accession=accession)
            continue
        if filing_date < cutoff_date:
            continue
        filings.append({
            "cik": str(cik).zfill(10),
            "form": form,
            "filing_date": date_str,
            "fiscal_year": filing_date.year,   
            "accession_number": accession,
            "primary_doc": primary_doc,
        })
    filings.sort(key=lambda f: f["filing_date"], reverse=True)
    log.info(
        "edgar_filing_index",
        cik=str(cik).zfill(10),
        total_found=len(filings),
        form_types=form_types,
    )
    return filings
def download_filing_html(
    cik: str,
    accession_number: str,
    primary_doc: str,
    dest_path: Path,
) -> bool:
    """
    Downloads the primary HTML document for a filing and saves it to dest_path.
    The filing document lives at:
        https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_no_dashes}/{primary_doc}
    Args:
        cik:              Zero-padded 10-digit CIK.
        accession_number: Accession number with dashes (e.g. "0000320193-23-000106").
        primary_doc:      Filename of the primary document (e.g. "aapl-20230930.htm").
        dest_path:        Local file path to save the downloaded HTML.
    Returns:
        True if the download succeeded; False if a non-retryable error occurred.
    """
    acc_no_clean = accession_number.replace("-", "")
    url = f"{_ARCHIVES_BASE}/{cik}/{acc_no_clean}/{primary_doc}"
    log.info(
        "edgar_download_start",
        cik=cik,
        accession=accession_number,
        url=url,
        dest=str(dest_path),
    )
    try:
        response = _get_with_retry(url)
    except RetryError as exc:
        log.error(
            "edgar_download_failed",
            cik=cik,
            accession=accession_number,
            error=str(exc),
        )
        return False
    encoding = response.encoding or "utf-8"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(response.text, encoding="utf-8")
    log.info(
        "edgar_download_complete",
        dest=str(dest_path),
        size_kb=round(len(response.content) / 1024, 1),
        encoding=encoding,
    )
    return True
