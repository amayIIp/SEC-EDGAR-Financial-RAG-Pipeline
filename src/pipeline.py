from __future__ import annotations 
import os 
from pathlib import Path 
from typing import List, Optional 
from src.ingestion.edgar_client import download_filing_html, fetch_filing_index 
from src.ingestion.run_ingest import _resolve_cik 
from src.ingestion.manifest import FilingManifest 
from src.parsing.html_parser import parse_filing 
from src.chunking.structure_aware_chunker import StructureAwareChunker 
from src.indexing.embedder import SECEmbedder 
from src.indexing.opensearch_index import OpenSearchIndexManager 
from src.indexing.qdrant_index import QdrantIndexManager 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import Chunk, FilingMetadata, FilingType 
log = get_logger(__name__)
class SECRAGPipeline:
    """
    Coordinates corporate filings download, parsing, chunking, embedding, and indexing.
    """
    def __init__(self) -> None:
        self.manifest = FilingManifest()
        self.os_manager = OpenSearchIndexManager()
        self.qd_manager = QdrantIndexManager()
    def ingest_ticker_data(
        self,
        ticker: str,
        forms: List[str],
        limit: int,
        embedding_provider: str
    ) -> int:
        """
        Downloads, parses, chunks, embeds, and indexes filings for a given company ticker.
        Returns the count of chunks successfully added to indices.
        """
        ticker_upper = ticker.upper()
        log.info("pipeline_ingest_start", ticker=ticker_upper, forms=forms)
        cik = _resolve_cik(ticker_upper)
        if not cik:
            raise ValueError(f"Could not resolve CIK for ticker: {ticker_upper}")
        filings = fetch_filing_index(cik=cik, form_types=forms, years_of_history=cfg.ingestion.years_of_history)
        target_filings = filings[:limit]
        raw_dir = Path(cfg.ingestion.raw_data_dir)
        parsed_dir = Path(cfg.parsing.parsed_data_dir)
        embedder = SECEmbedder(provider=embedding_provider)
        dim = embedder.get_dimension()
        self.os_manager.create_index()
        self.qd_manager.create_collection(dimension=dim)
        total_chunks_indexed = 0
        for filing in target_filings:
            accession = filing["accession_number"]
            if self.manifest.exists(accession):
                log.info("pipeline_ingest_skip_indexed", accession=accession)
                continue
            safe_form = filing["form"].replace("/", "-")
            filename = f"{safe_form}_{filing['filing_date']}_{accession}.html"
            dest_html_path = raw_dir / ticker_upper / filename
            download_success = download_filing_html(
                cik=cik,
                accession_number=accession,
                primary_doc=filing["primary_doc"],
                dest_path=dest_html_path
            )
            if not download_success:
                log.error("pipeline_ingest_download_failed", accession=accession)
                continue
            file_size_kb = dest_html_path.stat().st_size / 1024 if dest_html_path.exists() else 0.0
            metadata = FilingMetadata(
                ticker=ticker_upper,
                cik=cik,
                filing_type=FilingType(filing["form"]),
                filing_date=filing["filing_date"],
                fiscal_year=filing["fiscal_year"],
                accession_number=accession,
                local_path=str(dest_html_path)
            )
            parsed_filing = parse_filing(dest_html_path, metadata)
            if not parsed_filing:
                log.error("pipeline_ingest_parse_failed", accession=accession)
                continue
            chunker = StructureAwareChunker()
            chunks = chunker.chunk(parsed_filing)
            if not chunks:
                log.warning("pipeline_ingest_no_chunks", accession=accession)
                continue
            chunk_texts = [c.text for c in chunks]
            vectors = embedder.embed(chunk_texts)
            self.os_manager.bulk_index_chunks(chunks)
            self.qd_manager.bulk_upsert_chunks(chunks, vectors)
            self.manifest.add_entry(
                ticker=ticker_upper,
                cik=cik,
                form=filing["form"],
                filing_date=filing["filing_date"],
                fiscal_year=filing["fiscal_year"],
                accession_number=accession,
                primary_doc=filing["primary_doc"],
                local_path=str(dest_html_path),
                file_size_kb=file_size_kb,
                status="downloaded"
            )
            total_chunks_indexed += len(chunks)
            log.info("pipeline_ingest_filing_success", accession=accession, chunks=len(chunks))
        log.info("pipeline_ingest_complete", ticker=ticker_upper, total_indexed=total_chunks_indexed)
        return total_chunks_indexed
