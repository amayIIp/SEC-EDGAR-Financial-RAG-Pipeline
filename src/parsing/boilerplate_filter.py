# src/parsing/boilerplate_filter.py
#
# Removes noise from SEC HTML filings before section extraction.
#
# WHAT "BOILERPLATE" MEANS IN SEC FILINGS:
#   SEC HTML filings are rendered documents intended for human readers on
#   EDGAR's web viewer. They contain structural noise not present in the
#   actual financial content:
#
#   1. Table of Contents — navigation links (e.g. "Item 1A……23") that repeat
#      the section titles without any substantive content.
#   2. Page headers / footers — repeated company name, "Form 10-K", fiscal year,
#      and page number printed at the top/bottom of every page.
#   3. XBRL inline tags — <ix:nonNumeric>, <ix:fraction>, <ix:nonFraction>,
#      <ix:header> etc. inserted for machine-readable XBRL data. They wrap
#      real content but their attributes and wrappers pollute text extraction.
#   4. Empty formatting elements — <div>&nbsp;</div>, <p>&#160;</p>, <hr/>,
#      <br/> sequences used for visual spacing with no semantic content.
#   5. External hyperlinks — navigation anchors that add no textual meaning.
#
# STRATEGY:
#   We operate on the BeautifulSoup parse tree BEFORE text extraction.
#   Removing elements from the tree is faster and more reliable than
#   post-processing extracted strings with regex.

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from src.shared.config import cfg
from src.shared.logging_setup import get_logger

log = get_logger(__name__)

# Compile the boilerplate phrase patterns from config once at module load.
# re.IGNORECASE makes matching case-insensitive (e.g. "TABLE OF CONTENTS" or "Table of Contents").
_BOILERPLATE_PATTERNS = [
    re.compile(re.escape(phrase), re.IGNORECASE)
    for phrase in cfg.parsing.boilerplate_prefixes
]

# XBRL inline namespace prefixes used in SEC iXBRL filings.
# We strip the surrounding tag but keep the inner text so numeric values survive.
_XBRL_TAG_PREFIXES = ("ix:", "xbrli:", "xbrldi:", "link:", "xlink:")


def remove_xbrl_tags(soup: BeautifulSoup) -> None:
    """
    Unwraps XBRL inline tags (<ix:nonNumeric>, <ix:fraction>, etc.) in-place.
    "Unwrapping" removes the tag element but keeps its inner content, so
    the text "$29,915" remains even after the <ix:nonNumeric> wrapper is gone.
    """
    if not cfg.parsing.strip_xbrl:
        return

    # Find all tags whose name starts with a known XBRL namespace prefix.
    # We collect them into a list first because modifying the tree during
    # iteration causes BeautifulSoup to skip elements.
    xbrl_tags = [
        tag for tag in soup.find_all(True)  # True = any tag name
        if isinstance(tag.name, str)
        and any(tag.name.lower().startswith(prefix) for prefix in _XBRL_TAG_PREFIXES)
    ]

    for tag in xbrl_tags:
        # unwrap() replaces the tag with its children in the parent's content list.
        tag.unwrap()

    if xbrl_tags:
        log.debug("xbrl_tags_removed", count=len(xbrl_tags))


def remove_html_comments(soup: BeautifulSoup) -> None:
    """
    Removes all HTML comments from the parse tree.
    SEC filings sometimes embed XBRL schema references and tool-generated
    markup inside HTML comments that add noise to extracted text.
    """
    # BeautifulSoup represents HTML comments as Comment objects (subclass of NavigableString).
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


def remove_scripts_and_styles(soup: BeautifulSoup) -> None:
    """
    Removes <script> and <style> tags entirely (including their content).
    These exist for browser-side rendering and contain no financial content.
    """
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()


def remove_empty_elements(soup: BeautifulSoup) -> None:
    """
    Removes block elements that contain only whitespace or non-breaking spaces.
    These are pure visual-spacing elements added by the filing renderer.
    """
    # Non-breaking space characters used as padding in HTML layouts.
    _NBSP_CHARS = {"\xa0", "\u00a0", "&nbsp;", "&#160;"}

    # Only check block-level elements — inline spans may be legitimately empty
    # as CSS hooks and should not be removed.
    for tag in soup.find_all(["p", "div", "span", "td", "th", "li"]):
        text = tag.get_text(separator=" ").strip()
        # Check if the text is empty after stripping non-breaking spaces.
        is_empty = not text or all(c in _NBSP_CHARS or c.isspace() for c in text)
        if is_empty:
            tag.decompose()


def is_boilerplate_paragraph(text: str) -> bool:
    """
    Returns True if the paragraph text matches any boilerplate pattern from config.
    Used to skip TOC entries and page header/footer lines during section assembly.
    """
    stripped = text.strip()
    return any(pattern.search(stripped) for pattern in _BOILERPLATE_PATTERNS)


def remove_toc_section(soup: BeautifulSoup) -> None:
    """
    Attempts to identify and remove the Table of Contents section from the tree.

    Strategy: find the first element whose text is "Table of Contents" (or
    a variant), then remove sibling elements until we hit a text block that
    is clearly a real section header (contains "Item" and substantive text).
    This is heuristic — it works for ~90% of 10-Ks but some edge cases remain.
    """
    toc_pattern = re.compile(r"table\s+of\s+contents", re.IGNORECASE)

    # Locate the TOC heading element.
    toc_heading: Optional[Tag] = None
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "div"]):
        text = tag.get_text(strip=True)
        if toc_pattern.search(text) and len(text) < 60:
            # Short matches only — we don't want to remove a paragraph that
            # merely *mentions* the table of contents in passing.
            toc_heading = tag
            break

    if toc_heading is None:
        return

    # Remove elements following the TOC heading until we hit a real section.
    # We limit removal to 150 siblings to avoid accidentally stripping too much.
    item_pattern = re.compile(r"^\s*(item|part)\s+\d", re.IGNORECASE)
    removed_count = 0
    current = toc_heading.next_sibling

    while current is not None and removed_count < 150:
        next_sibling = current.next_sibling

        if isinstance(current, Tag):
            text = current.get_text(strip=True)
            # Stop when we reach the beginning of actual content.
            # A real Item header followed by more than 80 chars signals content start.
            if item_pattern.match(text) and len(text) > 80:
                break
            current.decompose()
            removed_count += 1
        elif isinstance(current, NavigableString):
            # NavigableString is a plain text node — remove whitespace-only ones.
            if not str(current).strip():
                current.extract()

        current = next_sibling

    # Also remove the TOC heading itself.
    toc_heading.decompose()
    log.debug("toc_removed", siblings_removed=removed_count)


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Applies all boilerplate removal steps to a BeautifulSoup parse tree in-place.
    Returns the same soup object (mutated) for convenient chaining.

    Call this BEFORE section extraction to give the parser a clean tree.
    """
    remove_html_comments(soup)
    remove_scripts_and_styles(soup)
    remove_xbrl_tags(soup)
    remove_toc_section(soup)
    remove_empty_elements(soup)
    return soup
