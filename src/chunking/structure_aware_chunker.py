from __future__ import annotations 
from typing import List 
import tiktoken 
from src.chunking.base_chunker import BaseChunker 
from src.shared.config import cfg 
from src.shared.models import Chunk, ChunkingStrategy, ParsedFiling, ParsedSection 
class StructureAwareChunker(BaseChunker):
    """
    Chunks filings along section boundaries, isolates tables, and recursively
    splits text paragraphs using a hierarchy of separators to stay within token limits.
    """
    def __init__(self, max_tokens: int = None, separators: List[str] = None) -> None:
        self.max_tokens = max_tokens or cfg.chunking.structure_aware.max_section_tokens
        self.separators = separators or cfg.chunking.structure_aware.split_separators
        self.encoder = tiktoken.get_encoding(cfg.chunking.token_encoding)
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits text using a list of separators to stay under the max_tokens limit.
        This preserves paragraph and sentence boundaries wherever possible.
        """
        if len(self.encoder.encode(text)) <= self.max_tokens:
            return [text]
        if not separators:
            tokens = self.encoder.encode(text)
            splits = []
            for i in range(0, len(tokens), self.max_tokens):
                chunk_tokens = tokens[i:i + self.max_tokens]
                splits.append(self.encoder.decode(chunk_tokens))
            return splits
        current_sep = separators[0]
        remaining_seps = separators[1:]
        raw_parts = text.split(current_sep)
        chunks: List[str] = [] 
        current_part = "" 
        for part in raw_parts:
            if len(self.encoder.encode(part)) > self.max_tokens:
                if current_part.strip():
                    chunks.append(current_part.strip())
                    current_part = ""
                sub_splits = self._recursive_split(part, remaining_seps)
                chunks.extend(sub_splits)
            elif len(self.encoder.encode(current_part + current_sep + part)) > self.max_tokens:
                if current_part.strip():
                    chunks.append(current_part.strip())
                current_part = part
            else:
                if current_part:
                    current_part += current_sep
                current_part += part
        if current_part.strip():
            chunks.append(current_part.strip())
        return chunks
    def chunk(self, parsed_filing: ParsedFiling) -> List[Chunk]:
        """
        Chunks the parsed filing using structure-aware guidelines.
        """
        chunks: List[Chunk] = [] 
        chunk_index = 0 
        for section in parsed_filing.sections:
            if section.content.strip():
                text_splits = self._recursive_split(section.content, self.separators)
                for split in text_splits:
                    context_prefix = f"[{section.item_id}. {section.title}]\n\n"
                    fused_text = context_prefix + split
                    chunk_id = Chunk.make_id(
                        ticker=parsed_filing.metadata.ticker,
                        accession_number=parsed_filing.metadata.accession_number,
                        strategy=ChunkingStrategy.STRUCTURE_AWARE.value,
                        chunk_index=chunk_index
                    )
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        text=fused_text,
                        strategy=ChunkingStrategy.STRUCTURE_AWARE,
                        chunk_index=chunk_index,
                        is_table=False,
                        ticker=parsed_filing.metadata.ticker,
                        cik=parsed_filing.metadata.cik,
                        filing_type=parsed_filing.metadata.filing_type,
                        filing_date=parsed_filing.metadata.filing_date,
                        fiscal_year=parsed_filing.metadata.fiscal_year,
                        accession_number=parsed_filing.metadata.accession_number,
                        section=section.item_id,
                        section_title=section.title,
                        token_count=len(self.encoder.encode(fused_text))
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            for table in section.tables:
                caption = table.get("caption", "").strip() or "Financial Table"
                table_text = f"[{section.item_id}. {section.title}] - Table: {caption}\n\n{table.get('raw_text', '')}"
                if table.get("raw_text", "").strip():
                    chunk_id = Chunk.make_id(
                        ticker=parsed_filing.metadata.ticker,
                        accession_number=parsed_filing.metadata.accession_number,
                        strategy=ChunkingStrategy.STRUCTURE_AWARE.value,
                        chunk_index=chunk_index
                    )
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        text=table_text,
                        strategy=ChunkingStrategy.STRUCTURE_AWARE,
                        chunk_index=chunk_index,
                        is_table=True, 
                        ticker=parsed_filing.metadata.ticker,
                        cik=parsed_filing.metadata.cik,
                        filing_type=parsed_filing.metadata.filing_type,
                        filing_date=parsed_filing.metadata.filing_date,
                        fiscal_year=parsed_filing.metadata.fiscal_year,
                        accession_number=parsed_filing.metadata.accession_number,
                        section=section.item_id,
                        section_title=section.title,
                        token_count=len(self.encoder.encode(table_text))
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        return chunks
