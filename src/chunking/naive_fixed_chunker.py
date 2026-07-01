# src/chunking/naive_fixed_chunker.py
# This module implements the NaiveFixedChunker class.
# Fixed-size token chunking is a baseline strategy in Retrieval-Augmented Generation (RAG).
#
# =========================================================================================
# Advanced Concept: Fixed-Size Token Chunking
# In this approach, we merge the entire document into a single flat string.
# We then use a tokenizer (a tool that breaks text down into sub-word integer codes, or "tokens")
# to divide the document into chunks of a fixed token length (e.g., 512 tokens), with a certain
# percentage of overlap (e.g., 15%) between consecutive chunks.
# The overlap ensures that sentences lying exactly on the chunk boundaries are not cut off
# and lost from context.
# While simple and fast, this strategy is "naive" because it ignores logical boundaries:
# paragraphs, tables, and sections get split down the middle, destroying formatting and logic.
# =========================================================================================

from __future__ import annotations # Allow classes to reference themselves in type hints.
from typing import List # Type helper for lists.
import tiktoken # OpenAI's tokenization tool to count tokens and convert IDs to strings.
from src.chunking.base_chunker import BaseChunker # Base chunker contract interface.
from src.shared.config import cfg # Config settings parser.
from src.shared.models import Chunk, ChunkingStrategy, ParsedFiling # Shared models.

class NaiveFixedChunker(BaseChunker):
    """
    Splits a document into fixed-size token chunks with overlapping boundaries,
    ignoring structural headers and tables.
    """

    def __init__(self, chunk_size: int = None, overlap_fraction: float = None) -> None:
        # Load the configuration settings for naive fixed chunking.
        self.chunk_size = chunk_size or cfg.chunking.naive_fixed.chunk_size_tokens
        # Load overlap fraction. e.g. 0.15 represents 15% overlap.
        self.overlap = overlap_fraction if overlap_fraction is not None else cfg.chunking.naive_fixed.overlap_fraction
        
        # Load the BPE encoder for cl100k_base (standard for GPT-4 and text-embedding-3-small).
        # BPE stands for Byte-Pair Encoding, a sub-word tokenization technique.
        self.encoder = tiktoken.get_encoding(cfg.chunking.token_encoding)

    def chunk(self, parsed_filing: ParsedFiling) -> List[Chunk]:
        """
        Splits a ParsedFiling into fixed-size token chunks.
        """
        # Step 1: Concatenate all section text and flat tables into a single large document string.
        corpus_elements: List[str] = []
        
        # Iterate through every section in the parsed document.
        for section in parsed_filing.sections:
            # Append the main paragraph text of the section.
            if section.content.strip():
                corpus_elements.append(section.content)
            
            # Append the flat text version of all tables in this section.
            for table in section.tables:
                if table.get("raw_text", "").strip():
                    corpus_elements.append(table["raw_text"])

        # Join the text parts with double newlines to simulate paragraphs.
        full_text = "\n\n".join(corpus_elements)

        # Step 2: Convert the entire text string into a list of integer token IDs.
        # This converts characters/words into the internal vocabulary numbers of the model.
        tokens = self.encoder.encode(full_text)
        
        # Calculate the number of overlapping tokens between chunks.
        # e.g., 512 * 0.15 = 76 tokens.
        overlap_size = int(self.chunk_size * self.overlap)
        
        # Calculate the stride (how far forward we move for the next chunk).
        # e.g., 512 - 76 = 436 tokens.
        stride = self.chunk_size - overlap_size
        
        # Ensure stride is at least 1 token to prevent infinite loops.
        if stride <= 0:
            stride = 1

        chunks: List[Chunk] = [] # List to store our final Chunk models.
        chunk_index = 0 # Counter to track the index position of each chunk.
        
        # Step 3: Loop through the tokens list using the stride.
        start_idx = 0
        while start_idx < len(tokens):
            # Extract a slice of tokens up to the maximum chunk size.
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode the integer token IDs back into standard UTF-8 text characters.
            chunk_text = self.encoder.decode(chunk_tokens)
            
            # Generate a deterministic unique ID based on file details and sequence index.
            chunk_id = Chunk.make_id(
                ticker=parsed_filing.metadata.ticker,
                accession_number=parsed_filing.metadata.accession_number,
                strategy=ChunkingStrategy.NAIVE_FIXED.value,
                chunk_index=chunk_index
            )
            
            # Build the Chunk model. Since this is naive chunking, we don't know the exact section,
            # so we label the section as "General" and section_title as empty.
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                strategy=ChunkingStrategy.NAIVE_FIXED,
                chunk_index=chunk_index,
                is_table=False, # We don't isolate tables here.
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
            
            # Add to the list.
            chunks.append(chunk)
            # Increment index counter.
            chunk_index += 1
            
            # Move the start pointer forward by the stride.
            start_idx += stride
            
            # If we reached the end of the tokens list, terminate.
            if end_idx >= len(tokens):
                break

        # Return the collection of chunks.
        return chunks
