from __future__ import annotations 
from typing import List 
import tiktoken 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import CitedChunk, RankedResult 
log = get_logger(__name__)
def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Computes the word-level Jaccard similarity coefficient between two text strings.
    Jaccard Similarity = size of intersection / size of union.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
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
    threshold = cfg.generation.dedup_similarity_threshold
    encoder = tiktoken.get_encoding(cfg.chunking.token_encoding)
    unique_candidates: List[RankedResult] = []
    for candidate in candidates:
        is_duplicate = False
        for accepted in unique_candidates:
            similarity = compute_jaccard_similarity(candidate.chunk.text, accepted.chunk.text)
            if similarity >= threshold:
                is_duplicate = True
                log.debug("context_packer_dedup_match", chunk_id=candidate.chunk.chunk_id, similarity=similarity)
                break
        if not is_duplicate:
            unique_candidates.append(candidate)
    cited_chunks: List[CitedChunk] = []
    for idx, ranked in enumerate(unique_candidates, start=1):
        citation_tag = f"[{idx}]"
        source_header = (
            f"Filing Source {idx} | Ticker: {ranked.chunk.ticker} | CIK: {ranked.chunk.cik} | "
            f"Form: {ranked.chunk.filing_type.value} | Date: {ranked.chunk.filing_date} | "
            f"Section: {ranked.chunk.section} ({ranked.chunk.section_title})"
        )
        packed_text = f"[{source_header}]\n{ranked.chunk.text}"
        cited = CitedChunk(
            ranked_result=ranked,
            citation_tag=citation_tag,
            packed_text=packed_text,
            packed_token_count=len(encoder.encode(packed_text))
        )
        cited_chunks.append(cited)
    log.info("context_packing_complete", input_count=len(candidates), packed_count=len(cited_chunks))
    return cited_chunks
