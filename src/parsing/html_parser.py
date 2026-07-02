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
_SECTION_HEADER_RE = re.compile(cfg.parsing.section_header_regex, re.IGNORECASE | re.MULTILINE)
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
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
    if tag_name in _HEADING_TAGS:
        return bool(_SECTION_HEADER_RE.search(tag.get_text()))
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
    kind = match.group(1).capitalize()
    number = match.group(2).upper()
    return f"{kind} {number}"
def _extract_title(text: str) -> str:
    """
    Extracts the human-readable title following the item identifier.
    "Item 1A. Risk Factors" → "Risk Factors"
    "PART I" → ""  (no title suffix)
    """
    cleaned = _SECTION_HEADER_RE.sub("", text).strip(" .\u2014\u2013-—")
    return cleaned[:200]  
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
    collecting = False
    for element in soup_body.descendants:
        if element is start_tag:
            collecting = True
            continue
        if end_tag is not None and element is end_tag:
            break
        if not collecting:
            continue
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text and len(text) >= cfg.parsing.min_content_length:
                if not is_boilerplate_paragraph(text):
                    paragraphs.append(text)
            continue
        if not isinstance(element, Tag):
            continue
        if element.name == "table":
            table_data = extract_table(element)
            if table_data:
                tables.append(table_data)
            continue
        if element.name in {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            if _is_section_header(element):
                continue
            text = element.get_text(separator=" ").strip()
            if text and len(text) >= cfg.parsing.min_content_length:
                if not is_boilerplate_paragraph(text):
                    paragraphs.append(text)
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
    try:
        raw_html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.error("parse_read_failed", path=str(html_path), error=str(exc))
        return None
    try:
        soup = BeautifulSoup(raw_html, cfg.parsing.bs4_parser)
    except Exception as exc:
        log.error("parse_bs4_failed", path=str(html_path), error=str(exc))
        return None
    soup = clean_soup(soup)
    body = soup.body or soup
    header_elements: list[tuple[Tag, str, str]] = []
    for tag in body.find_all(True):
        if _is_section_header(tag):
            text = tag.get_text(separator=" ").strip()
            item_id = _extract_item_id(text)
            title = _extract_title(text)
            if item_id:
                header_elements.append((tag, item_id, title))
    if not header_elements:
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
        sections: list[ParsedSection] = []
        had_warnings = False
        for idx, (header_tag, item_id, title) in enumerate(header_elements):
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
