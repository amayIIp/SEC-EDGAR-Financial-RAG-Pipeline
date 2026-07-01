# src/parsing/html_parser.py
#
# Primary SEC HTML filing parser.
#
# INPUT:  path to a raw HTML filing (data/raw/{ticker}/*.html)
# OUTPUT: ParsedFiling object (see src/shared/models.py)
#
# PIPELINE (in order):
#   1. Read raw HTML from disk.
#   2. Parse with BeautifulSoup + lxml backend.
#   3. Apply boilerplate_filter.clean_soup() to strip noise.
#   4. Detect section boundaries using Item/Part regex on heading elements.
#   5. For each section, collect paragraph text and call table_extractor.
#   6. Assemble ParsedFiling and return.
#
# SECTION BOUNDARY DETECTION STRATEGY:
#   SEC 10-K and 10-Q filings use standardised "Item N" / "Item NA" headings.
#   These headings appear in heading tags (h1–h6), bold <p> tags, or <div>
#   elements styled with font-weight:bold.  We check all three.
#   The regex from config.yaml is applied; matched elements become section starts.
#   Text between two consecutive section starts is that section's content.
#
# ROBUSTNESS:
#   If a filing fails to parse (malformed HTML, network truncation, unexpected
#   structure), we catch the exception, log it with the file path and reason,
#   and return None.  The caller (run_parse.py) increments a failure counter
#   rather than crashing the batch.

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag

from src.parsing.boilerplate_filter import clean_soup, is_boilerplate_paragraph
from src.parsing.table_extractor import extract_all_tables, extract_table
from src.shared.config import cfg
from src.shared.logging_setup import get_logger
from src.shared.models import FilingMetadata, ParsedFiling, ParsedSection

log = get_logger(__name__)

# Compile the section-header regex from config once at module load.
_SECTION_HEADER_RE = re.compile(cfg.parsing.section_header_regex, re.IGNORECASE | re.MULTILINE)

# Tags that may carry section headers in SEC HTML filings.
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# CSS properties that signal a visually-bolded paragraph used as a fake heading.
_BOLD_CSS_HINTS = {"font-weight:bold", "font-weight: bold", "font-weight:700"}


def _is_section_header(tag: Tag) -> bool:
    """
    Returns True if this HTML element is likely an SEC section header
    (e.g. "Item 1A. Risk Factors" or "PART I").

    Detection heuristics (applied in order):
      1. Tag is a semantic heading element (h1–h6).
      2. Tag is a <p> or <div> with inline CSS bold styling.
      3. The element's text matches the Item/Part regex.
    """
    tag_name = tag.name.lower() if tag.name else ""

    # Heuristic 1: real heading tags are almost always section headers if
    # their text matches our Item/Part pattern.
    if tag_name in _HEADING_TAGS:
        return bool(_SECTION_HEADER_RE.search(tag.get_text()))

    # Heuristic 2: <p> or <div> elements styled bold that match the pattern.
    if tag_name in {"p", "div", "span", "b", "strong"}:
        style = tag.get("style", "").lower().replace(" ", "")
        is_bold = (
            any(hint in style for hint in _BOLD_CSS_HINTS)
            or tag_name in {"b", "strong"}
            or bool(tag.find(["b", "strong"]))
        )
        if is_bold:
            return bool(_SECTION_HEADER_RE.search(tag.get_text()))

    return False


def _extract_item_id(text: str) -> str:
    """
    Extracts a normalised item identifier from a heading text.
    e.g. "Item 1A. Risk Factors" → "Item 1A"
         "PART I — FINANCIAL INFORMATION" → "Part I"
    """
    match = _SECTION_HEADER_RE.search(text)
    if not match:
        return ""
    # Group 1 = "item" or "part"; group 2 = "1A", "7", "2.02", etc.
    kind = match.group(1).capitalize()
    number = match.group(2).upper()
    return f"{kind} {number}"


def _extract_title(text: str) -> str:
    """
    Extracts the human-readable title following the item identifier.
    "Item 1A. Risk Factors" → "Risk Factors"
    "PART I" → ""  (no title suffix)
    """
    # Remove the Item/Part prefix and any trailing punctuation.
    cleaned = _SECTION_HEADER_RE.sub("", text).strip(" .\u2014\u2013-—")
    return cleaned[:200]  # cap at 200 chars; headings should not be longer


def _collect_section_content(
    start_tag: Tag,
    end_tag: Optional[Tag],
    soup_body: Tag,
) -> tuple[str, list[dict]]:
    """
    Collects all text and table content between two section header tags
    by walking forward through the document's sibling/child elements.

    Returns:
        (paragraph_text, tables_list)
    """
    paragraphs: list[str] = []
    tables: list[dict] = []

    # We iterate over ALL elements in document order that appear between
    # start_tag and end_tag.  We use a generator that walks the tree linearly.
    collecting = False

    for element in soup_body.descendants:
        if element is start_tag:
            # Start collecting content after the header element itself.
            collecting = True
            continue

        if end_tag is not None and element is end_tag:
            # Stop when we reach the next section header.
            break

        if not collecting:
            continue

        # Skip NavigableString objects that are just whitespace.
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text and len(text) >= cfg.parsing.min_content_length:
                if not is_boilerplate_paragraph(text):
                    paragraphs.append(text)
            continue

        if not isinstance(element, Tag):
            continue

        # Extract table elements — call the structured extractor.
        if element.name == "table":
            table_data = extract_table(element)
            if table_data:
                tables.append(table_data)
            continue

        # For block-level text elements, get the direct text content.
        if element.name in {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            # Skip elements we already identified as section headers.
            if _is_section_header(element):
                continue
            text = element.get_text(separator=" ").strip()
            if text and len(text) >= cfg.parsing.min_content_length:
                if not is_boilerplate_paragraph(text):
                    paragraphs.append(text)

    # Join paragraphs with double newlines to preserve paragraph structure.
    full_text = "\n\n".join(paragraphs)
    return full_text, tables


def parse_filing(html_path: Path, metadata: FilingMetadata) -> Optional[ParsedFiling]:
    """
    Parses one SEC HTML filing into a ParsedFiling object.

    Args:
        html_path: Path to the raw HTML file on disk.
        metadata:  FilingMetadata object with ticker, form, date, etc.

    Returns:
        ParsedFiling on success, None on unrecoverable parse failure.
    """
    log.info("parsing_start", path=str(html_path), ticker=metadata.ticker)

    # ── Step 1: Read raw HTML ────────────────────────────────────────────────
    try:
        raw_html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.error("parse_read_failed", path=str(html_path), error=str(exc))
        return None

    # ── Step 2: Parse with BeautifulSoup + lxml ──────────────────────────────
    try:
        soup = BeautifulSoup(raw_html, cfg.parsing.bs4_parser)
    except Exception as exc:
        log.error("parse_bs4_failed", path=str(html_path), error=str(exc))
        return None

    # ── Step 3: Clean boilerplate ────────────────────────────────────────────
    soup = clean_soup(soup)

    # Work inside <body> if it exists; fall back to the whole document.
    body = soup.body or soup

    # ── Step 4: Find all section header elements in document order ───────────
    # We collect (header_tag, item_id, title) tuples.
    header_elements: list[tuple[Tag, str, str]] = []

    for tag in body.find_all(True):
        if _is_section_header(tag):
            text = tag.get_text(separator=" ").strip()
            item_id = _extract_item_id(text)
            title = _extract_title(text)
            if item_id:
                header_elements.append((tag, item_id, title))

    if not header_elements:
        # Filing has no detectable Item headers — treat it as a single "General" section.
        log.warning("parse_no_sections", path=str(html_path), action="using_full_body")
        full_text = body.get_text(separator="\n").strip()
        tables = extract_all_tables(body)
        sections = [
            ParsedSection(
                item_id="General",
                title="",
                content=full_text,
                tables=tables,
            )
        ]
    else:
        # ── Step 5: Collect content between consecutive headers ───────────────
        sections: list[ParsedSection] = []
        had_warnings = False

        for idx, (header_tag, item_id, title) in enumerate(header_elements):
            # The end boundary is the next section header (or None for the last one).
            end_tag = header_elements[idx + 1][0] if idx + 1 < len(header_elements) else None

            try:
                content, tables = _collect_section_content(header_tag, end_tag, body)
            except Exception as exc:
                log.warning(
                    "section_collection_warning",
                    item_id=item_id,
                    path=str(html_path),
                    error=str(exc),
                )
                content, tables = "", []
                had_warnings = True

            # Skip sections with no extractable content (often duplicate header rows).
            if not content.strip() and not tables:
                continue

            sections.append(
                ParsedSection(
                    item_id=item_id,
                    title=title,
                    content=content,
                    tables=tables,
                )
            )

    # ── Step 6: Assemble ParsedFiling ────────────────────────────────────────
    total_chars = sum(len(s.content) for s in sections)

    filing = ParsedFiling(
        metadata=metadata,
        sections=sections,
        total_content_chars=total_chars,
        had_parse_warnings=had_warnings if header_elements else False,
    )

    log.info(
        "parsing_complete",
        path=str(html_path),
        sections=len(sections),
        tables=sum(len(s.tables) for s in sections),
        total_chars=total_chars,
    )
    return filing
