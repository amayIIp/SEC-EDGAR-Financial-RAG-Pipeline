from __future__ import annotations 
from pathlib import Path 
from bs4 import BeautifulSoup 
from src.parsing.boilerplate_filter import (
    clean_soup,
    remove_empty_elements,
    remove_html_comments,
    remove_scripts_and_styles,
    remove_xbrl_tags,
) 
from src.parsing.html_parser import parse_filing 
from src.parsing.table_extractor import extract_table 
from src.shared.models import FilingMetadata, FilingType 
def test_remove_xbrl_tags() -> None:
    """
    Verifies that remove_xbrl_tags unwraps inline XBRL nodes without destroying
    their textual contents.
    """
    html = "<div><ix:nonnumeric contextref='c1'>$10.5 Billion</ix:nonnumeric></div>"
    soup = BeautifulSoup(html, "lxml")
    remove_xbrl_tags(soup)
    assert soup.find("ix:nonnumeric") is None, "The XBRL tag was not removed."
    assert "$10.5 Billion" in soup.get_text(), "The textual content inside the XBRL tag was lost."
def test_remove_html_comments() -> None:
    """
    Verifies that remove_html_comments strips comment nodes from the HTML document.
    """
    html = "<div><!-- This is a comment -->Text content</div>"
    soup = BeautifulSoup(html, "lxml")
    remove_html_comments(soup)
    assert "This is a comment" not in soup.get_text(), "HTML comments were not extracted."
    assert "Text content" in soup.get_text(), "The regular text content was incorrectly deleted."
def test_remove_scripts_and_styles() -> None:
    """
    Verifies that scripts and styles tags are decomposed along with their inner texts.
    """
    html = "<div><style>body {color: red;}</style><script>console.log(5);</script>Content</div>"
    soup = BeautifulSoup(html, "lxml")
    remove_scripts_and_styles(soup)
    assert "color: red" not in soup.get_text(), "Style tags were not decomposed."
    assert "console.log" not in soup.get_text(), "Script tags were not decomposed."
    assert "Content" in soup.get_text(), "Valid text was incorrectly deleted."
def test_extract_table() -> None:
    """
    Verifies that extract_table converts HTML table tags into clean markdown and grid arrays.
    """
    html = """
    <table>
      <caption>Quarterly Profit</caption>
      <tr><th>Year</th><th>Profit</th></tr>
      <tr><td>2023</td><td>$99B</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, "lxml")
    table_tag = soup.find("table")
    table_data = extract_table(table_tag)
    assert table_data["headers"] == ["Year", "Profit"], "Extracted table headers do not match."
    assert table_data["rows"] == [["2023", "$99B"]], "Extracted table cell grid does not match."
    assert "Year" in table_data["raw_text"], "Markdown did not include headers."
    assert "$99B" in table_data["raw_text"], "Markdown did not include cell values."
def test_parse_filing_end_to_end(tmp_path: Path) -> None:
    """
    Verifies that parse_filing reads an HTML file and produces a structured ParsedFiling.
    """
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
    file_path = tmp_path / "test_filing.html"
    file_path.write_text(html_content, encoding="utf-8")
    metadata = FilingMetadata(
        ticker="GOOGL",
        cik="0001652044",
        filing_type=FilingType.TEN_K,
        filing_date="2023-09-30",
        fiscal_year=2023,
        accession_number="0001652044-23-000100",
        local_path=str(file_path)
    )
    parsed = parse_filing(file_path, metadata)
    assert parsed is not None, "ParsedFiling returned None."
    assert parsed.metadata.ticker == "GOOGL", "Ticker metadata mismatch."
    assert len(parsed.sections) >= 2, "Expected at least 2 sections to be parsed."
    sec_1 = parsed.sections[0]
    assert sec_1.item_id == "Item 1", "First section ID is incorrect."
    assert "business overview" in sec_1.content.lower(), "First section content mismatch."
    assert len(sec_1.tables) == 1, "Expected exactly 1 table in Item 1."
    assert sec_1.tables[0]["rows"] == [["Revenue", "$100M"]], "Table rows mismatch."
