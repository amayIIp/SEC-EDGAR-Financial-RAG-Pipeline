# tests/unit/test_parser.py
# This module implements unit tests for the HTML parsing, boilerplate filtering,
# and table extraction components. We verify that these utilities clean and structure
# corporate filing pages correctly.

from __future__ import annotations # Allow self-referencing type annotations.
from pathlib import Path # Standard library module for filesystem paths.
from bs4 import BeautifulSoup # HTML parser library.
from src.parsing.boilerplate_filter import (
    clean_soup,
    remove_empty_elements,
    remove_html_comments,
    remove_scripts_and_styles,
    remove_xbrl_tags,
) # Subjects under test.
from src.parsing.html_parser import parse_filing # Subject under test.
from src.parsing.table_extractor import extract_table # Subject under test.
from src.shared.models import FilingMetadata, FilingType # Shared models.


def test_remove_xbrl_tags() -> None:
    """
    Verifies that remove_xbrl_tags unwraps inline XBRL nodes without destroying
    their textual contents.
    """
    # Create HTML source containing an XBRL node.
    html = "<div><ix:nonnumeric contextref='c1'>$10.5 Billion</ix:nonnumeric></div>"
    # Parse with BeautifulSoup.
    soup = BeautifulSoup(html, "lxml")
    
    # Run the unwrapping function.
    remove_xbrl_tags(soup)
    
    # Check that the ix:nonnumeric tag has been unwrapped.
    assert soup.find("ix:nonnumeric") is None, "The XBRL tag was not removed."
    # Check that the inner text content survived intact.
    assert "$10.5 Billion" in soup.get_text(), "The textual content inside the XBRL tag was lost."


def test_remove_html_comments() -> None:
    """
    Verifies that remove_html_comments strips comment nodes from the HTML document.
    """
    # Create HTML with comments.
    html = "<div><!-- This is a comment -->Text content</div>"
    # Parse HTML.
    soup = BeautifulSoup(html, "lxml")
    
    # Remove comments.
    remove_html_comments(soup)
    
    # Verify that the text is clean and the comment is gone.
    assert "This is a comment" not in soup.get_text(), "HTML comments were not extracted."
    assert "Text content" in soup.get_text(), "The regular text content was incorrectly deleted."


def test_remove_scripts_and_styles() -> None:
    """
    Verifies that scripts and styles tags are decomposed along with their inner texts.
    """
    # Create HTML with scripts and style tags.
    html = "<div><style>body {color: red;}</style><script>console.log(5);</script>Content</div>"
    # Parse HTML.
    soup = BeautifulSoup(html, "lxml")
    
    # Run tag removal.
    remove_scripts_and_styles(soup)
    
    # Verify that body style and console statements are completely gone.
    assert "color: red" not in soup.get_text(), "Style tags were not decomposed."
    assert "console.log" not in soup.get_text(), "Script tags were not decomposed."
    assert "Content" in soup.get_text(), "Valid text was incorrectly deleted."


def test_extract_table() -> None:
    """
    Verifies that extract_table converts HTML table tags into clean markdown and grid arrays.
    """
    # Create HTML table source.
    html = """
    <table>
      <caption>Quarterly Profit</caption>
      <tr><th>Year</th><th>Profit</th></tr>
      <tr><td>2023</td><td>$99B</td></tr>
    </table>
    """
    # Parse table soup.
    soup = BeautifulSoup(html, "lxml")
    table_tag = soup.find("table")
    
    # Extract table structure.
    table_data = extract_table(table_tag)
    
    # Verify that data matches.
    assert table_data["headers"] == ["Year", "Profit"], "Extracted table headers do not match."
    # Verify that cells grid was built (excluding the header row).
    assert table_data["rows"] == [["2023", "$99B"]], "Extracted table cell grid does not match."
    # Verify that markdown text contains the headers.
    assert "Year" in table_data["raw_text"], "Markdown did not include headers."
    assert "$99B" in table_data["raw_text"], "Markdown did not include cell values."


def test_parse_filing_end_to_end(tmp_path: Path) -> None:
    """
    Verifies that parse_filing reads an HTML file and produces a structured ParsedFiling.
    """
    # Create dummy HTML file in a temporary folder.
    # The headings include a newline after the item identifier to match the regex pattern
    # which has a trailing newline loaded from the folded YAML block.
    # The text paragraphs are padded to exceed the 80 character min_content_length threshold.
    # The table has 2 rows so it is not skipped as a layout table.
    html_content = """
    <html>
      <body>
        <h1>Item 1.\n Business</h1>
        <p>This is the business overview section detailing core segments. It contains lots of important details about operations, markets, and overall corporate strategy.</p>
        <table>
          <tr><th>Category</th><th>Amount</th></tr>
          <tr><td>Revenue</td><td>$100M</td></tr>
        </table>
        <h2>Item 1A.\n Risk Factors</h2>
        <p>This is the risk factor details section listing market hazards. Investing in our common stock involves a high degree of risk and potential market volatility.</p>
      </body>
    </html>
    """
    # Write the content to a temporary file.
    file_path = tmp_path / "test_filing.html"
    file_path.write_text(html_content, encoding="utf-8")

    # Define metadata.
    metadata = FilingMetadata(
        ticker="GOOGL",
        cik="0001652044",
        filing_type=FilingType.TEN_K,
        filing_date="2023-09-30",
        fiscal_year=2023,
        accession_number="0001652044-23-000100",
        local_path=str(file_path)
    )

    # Execute filing parsing.
    parsed = parse_filing(file_path, metadata)

    # Check that parsing succeeded.
    assert parsed is not None, "ParsedFiling returned None."
    # Check that metadata was attached correctly.
    assert parsed.metadata.ticker == "GOOGL", "Ticker metadata mismatch."
    # Check that sections were extracted (Item 1 and Item 1A).
    assert len(parsed.sections) >= 2, "Expected at least 2 sections to be parsed."
    
    # Validate section details.
    sec_1 = parsed.sections[0]
    assert sec_1.item_id == "Item 1", "First section ID is incorrect."
    assert "business overview" in sec_1.content.lower(), "First section content mismatch."
    
    # Check that table inside the section was extracted.
    assert len(sec_1.tables) == 1, "Expected exactly 1 table in Item 1."
    assert sec_1.tables[0]["rows"] == [["Revenue", "$100M"]], "Table rows mismatch."
