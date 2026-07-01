# src/chunking/semantic_chunker.py
# This module implements the SemanticChunker class.
#
# =========================================================================================
# Advanced Concept: Semantic Chunking
# Traditional chunkers split text at arbitrary word/token counts or punctuation.
# Semantic chunking splits text based on meaning.
# 1. Sentence Tokenization: We split a document section into individual sentences.
# 2. Embedding Calculation: We run each sentence through a lightweight deep learning model
#    (all-MiniLM-L6-v2) to get its vector representation.
# 3. Similarity Assessment: We calculate the cosine similarity between consecutive sentence vectors.
#    If the similarity drops below a threshold (e.g. 0.50), it means the topic has changed,
#    so we split the text and start a new chunk.
# 4. Small-Chunk Merging: Chunks that are too small are merged with their neighbors to ensure
#    sufficient context window density.
# =========================================================================================

from __future__ import annotations # Allow self-referencing type hints.
import re # Standard library module for regular expressions.
from typing import List # Type helper.
import numpy as np # Numerical array calculations library.
import tiktoken # OpenAI's token counting library.
from sentence_transformers import SentenceTransformer # Local embedding loader.
from src.chunking.base_chunker import BaseChunker # Base interface.
from src.shared.config import cfg # Configuration settings loader.
from src.shared.models import Chunk, ChunkingStrategy, ParsedFiling, ParsedSection # Shared models.

class SemanticChunker(BaseChunker):
    """
    Splits text sections into sentences, embeds them, and splits where semantic similarity
    between adjacent sentences falls below a threshold.
    """

    def __init__(self, similarity_threshold: float = None, min_tokens: int = None) -> None:
        # Load the similarity threshold below which a chunk boundary is created.
        self.threshold = similarity_threshold if similarity_threshold is not None \
            else cfg.chunking.semantic.similarity_threshold
            
        # Load the minimum token size below which small chunks are merged.
        self.min_tokens = min_tokens if min_tokens is not None \
            else cfg.chunking.semantic.min_chunk_tokens
            
        # Load the maximum token count to cap large merged chunks.
        self.max_tokens = cfg.chunking.semantic.max_chunk_tokens
        
        # Initialize the BPE token encoder (cl100k_base).
        self.encoder = tiktoken.get_encoding(cfg.chunking.token_encoding)
        
        # Load the sentence-transformers model. This is run locally using PyTorch.
        self.model = SentenceTransformer(cfg.chunking.semantic.sentence_model)
        
        # Sentence splitting regular expression. Matches whitespace that follows a period/question mark,
        # but ignores abbreviations like U.S. or Corp.
        self.sentence_regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits a text paragraph into a list of clean, non-empty sentences.
        """
        # Split using the regex pattern.
        raw_sentences = self.sentence_regex.split(text)
        # Filter out empty or whitespace-only elements.
        return [s.strip() for s in raw_sentences if s.strip()]

    def _cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Computes the cosine similarity between two vectors.
        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0
        # Dot product divided by the product of magnitudes.
        return float(np.dot(u, v) / (norm_u * norm_v))

    def _group_sentences_semantically(self, sentences: List[str]) -> List[str]:
        """
        Encodes sentences and groups them by semantic similarity boundaries.
        """
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        # Step 1: Compute embeddings for all sentences.
        # This yields a 2D numpy array of shape (num_sentences, embedding_dimension).
        embeddings = self.model.encode(
            sentences,
            batch_size=cfg.chunking.semantic.sentence_encode_batch_size,
            show_progress_bar=False
        )

        groups: List[str] = [] # Accumulates final paragraph groups.
        current_group: List[str] = [sentences[0]] # Working set of sentences for current chunk.

        # Step 2: Compare similarity between consecutive sentence vectors.
        for i in range(len(sentences) - 1):
            # Calculate cosine similarity.
            similarity = self._cosine_similarity(embeddings[i], embeddings[i+1])
            
            # If similarity drops below threshold, cut and start a new group.
            if similarity < self.threshold:
                # Commit the current group.
                groups.append(" ".join(current_group))
                # Start new working group.
                current_group = [sentences[i+1]]
            else:
                # Otherwise, group them together.
                current_group.append(sentences[i+1])

        # Commit final leftover group.
        if current_group:
            groups.append(" ".join(current_group))

        return groups

    def _merge_small_chunks(self, groups: List[str]) -> List[str]:
        """
        Merges adjacent chunks that are smaller than min_tokens.
        """
        merged_chunks: List[str] = []
        current_chunk = ""

        # Loop through each sentence group.
        for group in groups:
            # If we are starting fresh, initialize the current chunk text.
            if not current_chunk:
                current_chunk = group
                continue

            # Calculate token length of current working chunk.
            curr_tokens = len(self.encoder.encode(current_chunk))
            # Calculate token length of the candidate group.
            group_tokens = len(self.encoder.encode(group))

            # Merge if the current chunk is too small, provided we don't exceed max_tokens.
            if curr_tokens < self.min_tokens and (curr_tokens + group_tokens) <= self.max_tokens:
                current_chunk += " " + group
            else:
                # Save the current chunk and start a new one.
                merged_chunks.append(current_chunk)
                current_chunk = group

        # Save any leftover chunk.
        if current_chunk:
            merged_chunks.append(current_chunk)

        return merged_chunks

    def chunk(self, parsed_filing: ParsedFiling) -> List[Chunk]:
        """
        Chunks the parsed filing using semantic boundary calculations.
        """
        chunks: List[Chunk] = [] # List to hold output Chunk objects.
        chunk_index = 0 # Sequential index counter.

        # Loop through each section of the filing.
        for section in parsed_filing.sections:
            # --- 1. Process Section text Content ---
            if section.content.strip():
                # Split section text into sentences.
                sentences = self._split_into_sentences(section.content)
                # Compute semantic groups based on similarity.
                raw_groups = self._group_sentences_semantically(sentences)
                # Merge small groups to maintain density.
                semantic_splits = self._merge_small_chunks(raw_groups)

                # Convert each semantic split into a Chunk model.
                for split in semantic_splits:
                    # Prepend section metadata context.
                    context_prefix = f"[{section.item_id}. {section.title}]\n\n"
                    fused_text = context_prefix + split
                    
                    # Generate deterministic ID.
                    chunk_id = Chunk.make_id(
                        ticker=parsed_filing.metadata.ticker,
                        accession_number=parsed_filing.metadata.accession_number,
                        strategy=ChunkingStrategy.SEMANTIC.value,
                        chunk_index=chunk_index
                    )

                    # Build Chunk.
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        text=fused_text,
                        strategy=ChunkingStrategy.SEMANTIC,
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
                caption = table.get("caption", "").strip() or "Financial Table"
                table_text = f"[{section.item_id}. {section.title}] - Table: {caption}\n\n{table.get('raw_text', '')}"

                if table.get("raw_text", "").strip():
                    chunk_id = Chunk.make_id(
                        ticker=parsed_filing.metadata.ticker,
                        accession_number=parsed_filing.metadata.accession_number,
                        strategy=ChunkingStrategy.SEMANTIC.value,
                        chunk_index=chunk_index
                    )

                    chunk = Chunk(
                        chunk_id=chunk_id,
                        text=table_text,
                        strategy=ChunkingStrategy.SEMANTIC,
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

        # Return the resulting list.
        return chunks
