from __future__ import annotations 
from typing import List 
import tiktoken 
from src.chunking.base_chunker import BaseChunker 
from src.shared.config import cfg 
from src.shared.models import Chunk, ChunkingStrategy, ParsedFiling 
class NaiveFixedChunker(BaseChunker):
    """
    Splits a document into fixed-size token chunks with overlapping boundaries,
    ignoring structural headers and tables.
    """
    def __init__(self, chunk_size: int = None, overlap_fraction: float = None) -> None:
        self.chunk_size = chunk_size or cfg.chunking.naive_fixed.chunk_size_tokens
        self.overlap = overlap_fraction if overlap_fraction is not None else cfg.chunking.naive_fixed.overlap_fraction
        self.encoder = tiktoken.get_encoding(cfg.chunking.token_encoding)
    def chunk(self, parsed_filing: ParsedFiling) -> List[Chunk]:
        """
        Splits a ParsedFiling into fixed-size token chunks.
        """
        corpus_elements: List[str] = []
        for section in parsed_filing.sections:
            if section.content.strip():
                corpus_elements.append(section.content)
            for table in section.tables:
                if table.get("raw_text", "").strip():
                    corpus_elements.append(table["raw_text"])
        full_text = "\n\n".join(corpus_elements)
        tokens = self.encoder.encode(full_text)
        overlap_size = int(self.chunk_size * self.overlap)
        stride = self.chunk_size - overlap_size
        if stride <= 0:
            stride = 1
        chunks: List[Chunk] = [] 
        chunk_index = 0 
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunk_id = Chunk.make_id(
                ticker=parsed_filing.metadata.ticker,
                accession_number=parsed_filing.metadata.accession_number,
                strategy=ChunkingStrategy.NAIVE_FIXED.value,
                chunk_index=chunk_index
            )
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                strategy=ChunkingStrategy.NAIVE_FIXED,
                chunk_index=chunk_index,
                is_table=False, 
                ticker=parsed_filing.metadata.ticker,
                cik=parsed_filing.metadata.cik,
                filing_type=parsed_filing.metadata.filing_type,
                filing_date=parsed_filing.metadata.filing_date,
                fiscal_year=parsed_filing.metadata.fiscal_year,
                accession_number=parsed_filing.metadata.accession_number,
                section="General",
                section_title="",
                token_count=len(chunk_tokens)
            )
            chunks.append(chunk)
            chunk_index += 1
            start_idx += stride
            if end_idx >= len(tokens):
                break
        return chunks
