# src/generation/citation_checker.py
# This module implements the citation checker utility.
#
# =========================================================================================
# Advanced Concept: Citation Validation and Context Utilisation
# To prevent LLM hallucinations, we require the model to cite its sources using tags like [1].
# 1. Parsing: We scan the generated text using regex to find all citation brackets.
# 2. Validation: We filter out any hallucinated citations (e.g., if the model cites "[9]"
#    but we only provided 4 context chunks, that citation is invalid).
# 3. Context Utilisation Ratio: We compute the fraction of provided context chunks that
#    were actually referenced. If we feed 8 chunks but the model only cites 2, our context
#    utilisation is 25%. This helps us optimize retrieval depth (Phase 8).
# =========================================================================================

from __future__ import annotations # Allow self-referencing type annotations.
import re # Standard library module for regex.
from typing import List, Tuple # Type helpers.
from src.shared.models import CitedChunk # Shared models.

def check_citations(answer: str, packed_chunks: List[CitedChunk]) -> Tuple[List[str], float]:
    """
    Parses citation brackets from the answer text, filters for valid context tags,
    and returns a tuple of (list of valid cited tags, context utilisation ratio).
    """
    # Find all bracketed digits in the text, e.g. [1], [2], [10].
    # The pattern matches a literal '[' followed by one or more digits, followed by ']'.
    raw_matches = re.findall(r'\[(\d+)\]', answer)
    
    # Map the digit matches to string tags, e.g. "1" -> "[1]".
    # We use a set first to eliminate duplicate citations.
    cited_numbers = set(int(m) for m in raw_matches)
    
    # Establish a set of valid indexes based on the number of packed chunks.
    # e.g., if we have 5 chunks, valid indexes are 1, 2, 3, 4, 5.
    valid_indexes = set(range(1, len(packed_chunks) + 1))
    
    # Filter for numbers that actually correspond to a provided chunk index.
    valid_citations = cited_numbers.intersection(valid_indexes)
    
    # Construct the tag strings, e.g. [1], [3] in sorted order.
    cited_tags = [f"[{num}]" for num in sorted(valid_citations)]
    
    # Compute context utilisation ratio: count of cited chunks / count of total chunks.
    if not packed_chunks:
        ratio = 0.0
    else:
        ratio = len(valid_citations) / len(packed_chunks)
        
    return cited_tags, ratio
