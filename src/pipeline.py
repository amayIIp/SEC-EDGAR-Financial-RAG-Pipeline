# src/pipeline.py
# This module implements the high-level SECRAGPipeline orchestrator.
# It uses our modular sub-packages (ingestion, parsing, chunking, indexing)
# to coordinate batch data ingestion for a company ticker.
# In a production environment, this pipeline is typically scheduled as a cron job
# or triggered via webhook when new filings are released.

from __future__ import annotations # Allow self-referencing type annotations.
import os # Standard library module for file path operations.
from pathlib import Path # Standard library module for filesystem paths.
from typing import List, Optional # Type helpers.
from src.ingestion.edgar_client import download_filing_html, fetch_filing_index # Edgar clients.
from src.ingestion.run_ingest import _resolve_cik # Import CIK resolution helper from the ingestion CLI.
from src.ingestion.manifest import FilingManifest # Manifest database.
from src.parsing.html_parser import parse_filing # HTML document parser.
from src.chunking.structure_aware_chunker import StructureAwareChunker # Structural chunker.
from src.indexing.embedder import SECEmbedder # Embeddings calculator.
from src.indexing.opensearch_index import OpenSearchIndexManager # OpenSearch manager.
from src.indexing.qdrant_index import QdrantIndexManager # Qdrant manager.
from src.shared.config import cfg # Configuration settings loader.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import Chunk, FilingMetadata, FilingType # Shared models.

log = get_logger(__name__)

class SECRAGPipeline:
    """
    Coordinates corporate filings download, parsing, chunking, embedding, and indexing.
    """

    def __init__(self) -> None:
        # Initialize the manifest to keep track of successfully indexed filings.
        self.manifest = FilingManifest()
        # Initialize database managers.
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

        # ── Step 1: Resolve TIK to CIK ──
        cik = _resolve_cik(ticker_upper)
        if not cik:
            raise ValueError(f"Could not resolve CIK for ticker: {ticker_upper}")

        # ── Step 2: Fetch Filing list ──
        # Fetch the available filings matching the requested criteria.
        filings = fetch_filing_index(cik=cik, form_types=forms, years_of_history=cfg.ingestion.years_of_history)
        
        # Limit to the requested count (e.g. latest 3 filings).
        target_filings = filings[:limit]
        
        raw_dir = Path(cfg.ingestion.raw_data_dir)
        parsed_dir = Path(cfg.parsing.parsed_data_dir)
        
        # Initialize embedder to fetch dimension configurations.
        embedder = SECEmbedder(provider=embedding_provider)
        dim = embedder.get_dimension()
        
        # Ensure indices and collections exist in OpenSearch and Qdrant.
        self.os_manager.create_index()
        self.qd_manager.create_collection(dimension=dim)

        total_chunks_indexed = 0

        # Loop through each filing.
        for filing in target_filings:
            accession = filing["accession_number"]
            
            # If the file has already been indexed (exists in the manifest), skip it.
            if self.manifest.exists(accession):
                log.info("pipeline_ingest_skip_indexed", accession=accession)
                continue

            # ── Step 3: Download Filing HTML ──
            safe_form = filing["form"].replace("/", "-")
            filename = f"{safe_form}_{filing['filing_date']}_{accession}.html"
            dest_html_path = raw_dir / ticker_upper / filename
            
            # Download the HTML.
            download_success = download_filing_html(
                cik=cik,
                accession_number=accession,
                primary_doc=filing["primary_doc"],
                dest_path=dest_html_path
            )
            
            if not download_success:
                log.error("pipeline_ingest_download_failed", accession=accession)
                continue

            # Update size for manifest.
            file_size_kb = dest_html_path.stat().st_size / 1024 if dest_html_path.exists() else 0.0

            # ── Step 4: Parse HTML Filing ──
            metadata = FilingMetadata(
                ticker=ticker_upper,
                cik=cik,
                filing_type=FilingType(filing["form"]),
                filing_date=filing["filing_date"],
                fiscal_year=filing["fiscal_year"],
                accession_number=accession,
                local_path=str(dest_html_path)
            )
            
            # Parse HTML into structured sections.
            parsed_filing = parse_filing(dest_html_path, metadata)
            if not parsed_filing:
                log.error("pipeline_ingest_parse_failed", accession=accession)
                continue

            # ── Step 5: Chunk Filing ──
            # We use StructureAwareChunker by default for standard query pipelines.
            chunker = StructureAwareChunker()
            chunks = chunker.chunk(parsed_filing)
            
            if not chunks:
                log.warning("pipeline_ingest_no_chunks", accession=accession)
                continue

            # ── Step 6: Embed Chunks ──
            chunk_texts = [c.text for c in chunks]
            vectors = embedder.embed(chunk_texts)

            # ── Step 7: Index Chunks into OpenSearch & Qdrant ──
            self.os_manager.bulk_index_chunks(chunks)
            self.qd_manager.bulk_upsert_chunks(chunks, vectors)

            # ── Step 8: Update Ingestion Manifest ──
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
