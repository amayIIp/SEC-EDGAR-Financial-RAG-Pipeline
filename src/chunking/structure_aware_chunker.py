# src/chunking/structure_aware_chunker.py
# This module implements the StructureAwareChunker class.
#
# =========================================================================================
# Advanced Concept: Structure-Aware Chunking
# Standard naive chunking cuts text off at arbitrary token counts, splitting sentences and tables.
# Structure-aware chunking respects the document layout. It works in three steps:
# 1. Section Isolation: We keep text from different SEC Items (e.g., Item 1A vs Item 7) separate.
#    This prevents cross-section context contamination.
# 2. Table Preservation: Tables are indexed as a single chunk whenever possible, because
#    splitting a table makes it impossible to link rows with their respective column headers.
# 3. Context Injection: We prepend the section's name (e.g., "[Item 7. MD&A]") to every chunk,
#    providing critical global context to the LLM during retrieval.
# =========================================================================================

from __future__ import annotations # Allow forward references.
from typing import List # Type helper for lists.
import tiktoken # OpenAI's token counting library.
from src.chunking.base_chunker import BaseChunker # Base abstract class contract.
from src.shared.config import cfg # Config loader singleton.
from src.shared.models import Chunk, ChunkingStrategy, ParsedFiling, ParsedSection # Data models.

class StructureAwareChunker(BaseChunker):
    """
    Chunks filings along section boundaries, isolates tables, and recursively
    splits text paragraphs using a hierarchy of separators to stay within token limits.
    """

    def __init__(self, max_tokens: int = None, separators: List[str] = None) -> None:
        # Load maximum token size per chunk from configuration.
        self.max_tokens = max_tokens or cfg.chunking.structure_aware.max_section_tokens
        
        # Load the split separators in order (e.g., double newlines, single newlines, space).
        self.separators = separators or cfg.chunking.structure_aware.split_separators
        
        # Load the BPE token encoder rules (cl100k_base).
        self.encoder = tiktoken.get_encoding(cfg.chunking.token_encoding)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits text using a list of separators to stay under the max_tokens limit.
        This preserves paragraph and sentence boundaries wherever possible.
        """
        # If the token count of the text is already within the limit, return the text as is.
        if len(self.encoder.encode(text)) <= self.max_tokens:
            return [text]
            
        # If we have run out of separators, fallback to splitting by tokens directly.
        if not separators:
            tokens = self.encoder.encode(text)
            splits = []
            # Slice tokens into groups of max_tokens size.
            for i in range(0, len(tokens), self.max_tokens):
                chunk_tokens = tokens[i:i + self.max_tokens]
                splits.append(self.encoder.decode(chunk_tokens))
            return splits

        # Select the first separator in the list.
        current_sep = separators[0]
        # Get the remaining separators.
        remaining_seps = separators[1:]
        
        # Split the text by the current separator.
        raw_parts = text.split(current_sep)
        
        chunks: List[str] = [] # List to accumulate the split sub-chunks.
        current_part = "" # Accumulates parts before committing to a chunk.
        
        # Loop through each sub-segment from the split.
        for part in raw_parts:
            # If the individual part is larger than the token limit, split it recursively.
            if len(self.encoder.encode(part)) > self.max_tokens:
                # Flush the current accumulated part if it contains text.
                if current_part.strip():
                    chunks.append(current_part.strip())
                    current_part = ""
                # Call recursive split with the remaining separators on the large part.
                sub_splits = self._recursive_split(part, remaining_seps)
                chunks.extend(sub_splits)
            # If adding this part exceeds the token limit, save the current chunk.
            elif len(self.encoder.encode(current_part + current_sep + part)) > self.max_tokens:
                if current_part.strip():
                    chunks.append(current_part.strip())
                current_part = part
            # Otherwise, append the part to our accumulator.
            else:
                if current_part:
                    current_part += current_sep
                current_part += part
                
        # Append any leftover text as the final chunk.
        if current_part.strip():
            chunks.append(current_part.strip())
            
        # Return the list of sub-chunks.
        return chunks

    def chunk(self, parsed_filing: ParsedFiling) -> List[Chunk]:
        """
        Chunks the parsed filing using structure-aware guidelines.
        """
        chunks: List[Chunk] = [] # Final list of Chunk models.
        chunk_index = 0 # Counter for sequential indexing.
        
        # Loop through each section extracted from the HTML.
        for section in parsed_filing.sections:
            # --- 1. Process Section text Content ---
            if section.content.strip():
                # Recursively split the text content to stay within the max_tokens limit.
                text_splits = self._recursive_split(section.content, self.separators)
                
                # Create a Chunk object for each split.
                for split in text_splits:
                    # Prepend section header metadata context to the chunk text.
                    context_prefix = f"[{section.item_id}. {section.title}]\n\n"
                    fused_text = context_prefix + split
                    
                    # Generate deterministic ID.
                    chunk_id = Chunk.make_id(
                        ticker=parsed_filing.metadata.ticker,
                        accession_number=parsed_filing.metadata.accession_number,
                        strategy=ChunkingStrategy.STRUCTURE_AWARE.value,
                        chunk_index=chunk_index
                    )
                    
                    # Create the Chunk model.
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
            
            # --- 2. Process Section Tables ---
            for table in section.tables:
                # Extract caption or use default label.
                caption = table.get("caption", "").strip() or "Financial Table"
                # Formulate the table chunk text, preserving structural columns.
                table_text = f"[{section.item_id}. {section.title}] - Table: {caption}\n\n{table.get('raw_text', '')}"
                
                # Verify that the table contains text and has numbers before indexing.
                if table.get("raw_text", "").strip():
                    # Generate deterministic ID.
                    chunk_id = Chunk.make_id(
                        ticker=parsed_filing.metadata.ticker,
                        accession_number=parsed_filing.metadata.accession_number,
                        strategy=ChunkingStrategy.STRUCTURE_AWARE.value,
                        chunk_index=chunk_index
                    )
                    
                    # Create the Chunk model.
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        text=table_text,
                        strategy=ChunkingStrategy.STRUCTURE_AWARE,
                        chunk_index=chunk_index,
                        is_table=True, # Label this chunk as a table.
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
                    
        # Return the collection of chunks.
        return chunks
