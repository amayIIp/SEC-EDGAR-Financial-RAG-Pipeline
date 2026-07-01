# src/generation/context_packer.py
# This module implements the context-packing logic.
#
# =========================================================================================
# Advanced Concept: Context Packing and Deduplication
# When we retrieve the top-N chunks from databases, some might contain near-identical text
# (e.g., repeating paragraphs across quarterly and annual filings, or overlapping segments).
# Pasting duplicate information into the LLM prompt wastes tokens (increasing costs) and
# can confuse the model's reasoning.
# 1. Deduplication: We evaluate the similarity between candidate text chunks.
#    If the Jaccard similarity (word set intersection divided by union) exceeds a threshold,
#    we drop the lower-ranked chunk.
# 2. Context Labeling: We order the remaining unique chunks by relevance, then label them
#    sequentially with citation tags (e.g. "[1]", "[2]"). This allows the LLM to write answers
#    referencing exactly which source block it used, preventing unsourced claims.
# =========================================================================================

from __future__ import annotations # Allow self-referencing type annotations.
from typing import List # Type helper.
import tiktoken # Token counter.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import CitedChunk, RankedResult # Shared models.

log = get_logger(__name__)

def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Computes the word-level Jaccard similarity coefficient between two text strings.
    Jaccard Similarity = size of intersection / size of union.
    """
    # Split text into unique lowercase words.
    words1 = set(text1.lower().split())
    # Split second text.
    words2 = set(text2.lower().split())
    
    # Return 0 if either is empty.
    if not words1 or not words2:
        return 0.0
        
    # Calculate Jaccard similarity.
    intersection_len = len(words1.intersection(words2))
    union_len = len(words1.union(words2))
    return intersection_len / union_len

def pack_context(candidates: List[RankedResult]) -> List[CitedChunk]:
    """
    Deduplicates near-identical chunks, orders them by rerank score,
    and returns a list of CitedChunk objects with sequential citation labels.
    """
    if not candidates:
        return []

    # Get the Jaccard similarity threshold from configuration.
    threshold = cfg.generation.dedup_similarity_threshold
    
    # Initialize the tokenizer to count tokens of the packed context chunks.
    encoder = tiktoken.get_encoding(cfg.chunking.token_encoding)
    
    # Step 1: Filter out near-duplicates.
    # We iterate through candidates in order of relevance (highest score first).
    # We keep a chunk only if its similarity with all already-accepted chunks is below the threshold.
    unique_candidates: List[RankedResult] = []
    
    for candidate in candidates:
        is_duplicate = False
        
        # Compare against previously accepted chunks.
        for accepted in unique_candidates:
            similarity = compute_jaccard_similarity(candidate.chunk.text, accepted.chunk.text)
            if similarity >= threshold:
                is_duplicate = True
                log.debug("context_packer_dedup_match", chunk_id=candidate.chunk.chunk_id, similarity=similarity)
                break
                
        # Keep chunk if not a duplicate.
        if not is_duplicate:
            unique_candidates.append(candidate)

    # Step 2: Build the CitedChunk list.
    cited_chunks: List[CitedChunk] = []
    
    # Loop through unique chunks to assign citation labels.
    for idx, ranked in enumerate(unique_candidates, start=1):
        # Create citation label, e.g. "[1]" or "[2]".
        citation_tag = f"[{idx}]"
        
        # We can format the chunk's text to include its source metadata.
        # This gives the LLM clear visibility into the source.
        source_header = (
            f"Filing Source {idx} | Ticker: {ranked.chunk.ticker} | CIK: {ranked.chunk.cik} | "
            f"Form: {ranked.chunk.filing_type.value} | Date: {ranked.chunk.filing_date} | "
            f"Section: {ranked.chunk.section} ({ranked.chunk.section_title})"
        )
        
        packed_text = f"[{source_header}]\n{ranked.chunk.text}"
        
        # Create the CitedChunk object.
        cited = CitedChunk(
            ranked_result=ranked,
            citation_tag=citation_tag,
            packed_text=packed_text,
            packed_token_count=len(encoder.encode(packed_text))
        )
        cited_chunks.append(cited)

    log.info("context_packing_complete", input_count=len(candidates), packed_count=len(cited_chunks))
    return cited_chunks
