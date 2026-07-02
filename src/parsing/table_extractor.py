# src/parsing/table_extractor.py
#
# Extracts HTML <table> elements from a BeautifulSoup tree into structured dicts.
#
# WHY STRUCTURED TABLE EXTRACTION MATTERS:
#   If we flatten an income statement table to plain text, we get:
#     "Revenue 383292 365817 Product revenue 298085 316199 Service revenue 85207 ..."
#   The column labels ("FY2023", "FY2022") are lost, so a vector search for
#   "Apple FY2022 revenue" cannot distinguish which number belongs to which year.
#
#   Structured extraction produces:
#     rows = [["", "FY2023", "FY2022"],
#             ["Revenue", "383,292", "365,817"],
#             ["Product revenue", "298,085", "316,199"]]
#   And a flat text rendering with row × column associations preserved:
#     "Revenue: FY2023=383,292 | FY2022=365,817\nProduct revenue: FY2023=298,085 ..."
#
# OUTPUT FORMAT (dict per table):
#   {
#     "caption":  str,           # text of <caption> tag or nearest preceding <p>
#     "headers":  [str, ...],    # first row treated as header
#     "rows":     [[str, ...]],  # remaining rows, each a list of cell texts
#     "raw_text": str,           # BM25-friendly flattened version for embedding
#     "has_numbers": bool,       # True if any cell contains a digit (financial table)
#   }

from __future__ import annotations

import re
from typing import Any, Optional

from bs4 import BeautifulSoup, Tag

from src.shared.logging_setup import get_logger

log = get_logger(__name__)

# Pattern to detect cells that contain numeric financial data.
# Matches digits, commas, parentheses (negative), dollar signs, percent signs.
_NUMERIC_CELL_RE = re.compile(r"[\d,$%()]+")


def _extract_cell_text(cell: Tag) -> str:
    """
    Extracts the plain text from a single <td> or <th> cell.
    Collapses multiple whitespace characters into one space and strips ends.
    """
    # separator=" " adds a space between child elements (e.g. nested <span>s)
    # so "Revenue" and "($M)" in separate spans become "Revenue ($M)" not "Revenue($M)".
    text = cell.get_text(separator=" ")
    # Collapse runs of whitespace (including non-breaking spaces \xa0) into single spaces.
    text = re.sub(r"[\s\xa0]+", " ", text).strip()
    return text


def _get_caption(table_tag: Tag) -> str:
    """
    Finds a descriptive caption for the table by checking three sources in order:
      1. An explicit <caption> child element.
      2. The text of the immediately preceding <p> or heading element.
      3. An empty string if nothing suitable is found.
    """
    # Check for an explicit HTML <caption> element inside the table.
    caption_tag = table_tag.find("caption")
    if caption_tag:
        return caption_tag.get_text(separator=" ").strip()

    # Look backwards from the table in the document tree for the nearest
    # preceding sibling that contains descriptive text.
    for prev in table_tag.previous_siblings:
        if not isinstance(prev, Tag):
            continue
        if prev.name in ("p", "h1", "h2", "h3", "h4", "h5", "h6", "div"):
            text = prev.get_text(separator=" ").strip()
            # Accept as caption only if it is short enough to be a label (≤200 chars)
            # and not just whitespace.
            if text and len(text) <= 200:
                return text
        # Stop searching after two block elements to avoid grabbing unrelated paragraphs.
        break

    return ""


def _rows_to_flat_text(headers: list[str], rows: list[list[str]]) -> str:
    """
    Converts a structured table into a flat text string suitable for BM25 indexing
    and embedding, preserving column-header associations.

    Format:
      Header1 | Header2 | Header3
      Row1Col1: Header1=Row1Col1_val | Header2=Row1Col2_val | ...
      ...
    """
    lines: list[str] = []

    # First line: pipe-separated column headers.
    if headers:
        lines.append(" | ".join(headers))

    for row in rows:
        if not any(cell.strip() for cell in row):
            # Skip entirely empty rows — common in SEC tables used for spacing.
            continue

        # Build "Header=Value" pairs for non-empty cells.
        if headers:
            pairs = [
                f"{headers[col_idx]}={cell}"
                for col_idx, cell in enumerate(row)
                if cell.strip() and col_idx < len(headers)
            ]
            line = " | ".join(pairs) if pairs else " | ".join(row)
        else:
            line = " | ".join(cell for cell in row if cell.strip())

        if line.strip():
            lines.append(line)

    return "\n".join(lines)


def extract_table(table_tag: Tag) -> Optional[dict[str, Any]]:
    """
    Parses a single <table> Tag into a structured dict.

    Returns None if the table has fewer than 2 rows (likely a layout table,
    not a financial data table) or if all cells are empty.
    """
    # Collect all rows from the table, including those in <thead>, <tbody>, <tfoot>.
    all_rows: list[list[str]] = []

    for tr in table_tag.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row = [_extract_cell_text(cell) for cell in cells]
        all_rows.append(row)

    if len(all_rows) < 2:
        # A 0-row or 1-row table is almost certainly a layout/styling table.
        return None

    # Treat the first row as column headers.
    headers = all_rows[0]
    data_rows = all_rows[1:]

    # Detect if any cell contains a number — financial tables always do.
    flat_all = " ".join(cell for row in all_rows for cell in row)
    has_numbers = bool(_NUMERIC_CELL_RE.search(flat_all))

    # Get caption text.
    caption = _get_caption(table_tag)

    # Build the flat text representation for embedding/BM25.
    raw_text = _rows_to_flat_text(headers, data_rows)

    # If the entire text is very short, it is likely a layout/navigation table.
    if len(raw_text.strip()) < 30:
        return None

    return {
        "caption": caption,
        "headers": headers,
        "rows": data_rows,
        "raw_text": raw_text,
        "has_numbers": has_numbers,
        "num_rows": len(data_rows),
        "num_cols": max(len(r) for r in all_rows) if all_rows else 0,
    }


def extract_all_tables(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """
    Finds every <table> in the parse tree and returns a list of structured
    table dicts.  Tables that look like layout/navigation tables are filtered out.
    """
    tables: list[dict[str, Any]] = []

    for table_tag in soup.find_all("table"):
        result = extract_table(table_tag)
        if result is not None:
            tables.append(result)

    log.debug("tables_extracted", total=len(tables))
    return tables
