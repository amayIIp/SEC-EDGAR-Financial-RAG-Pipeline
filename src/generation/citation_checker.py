from __future__ import annotations 
import re 
from typing import List, Tuple 
from src.shared.models import CitedChunk 
def check_citations(answer: str, packed_chunks: List[CitedChunk]) -> Tuple[List[str], float]:
    """
    Parses citation brackets from the answer text, filters for valid context tags,
    and returns a tuple of (list of valid cited tags, context utilisation ratio).
    """
    raw_matches = re.findall(r'\[(\d+)\]', answer)
    cited_numbers = set(int(m) for m in raw_matches)
    valid_indexes = set(range(1, len(packed_chunks) + 1))
    valid_citations = cited_numbers.intersection(valid_indexes)
    cited_tags = [f"[{num}]" for num in sorted(valid_citations)]
    if not packed_chunks:
        ratio = 0.0
    else:
        ratio = len(valid_citations) / len(packed_chunks)
    return cited_tags, ratio
