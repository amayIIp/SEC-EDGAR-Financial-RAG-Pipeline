# src/evaluation/retrieval_metrics.py
# This module implements standard information retrieval quality metrics.
#
# =========================================================================================
# Advanced Concept: Retrieval Evaluation (Recall, Precision, and MRR)
# A RAG system's answer quality is bounded by its search performance.
# We evaluate the search engine using three metrics against a set of expected chunk IDs:
# 1. Recall@k: Out of all the relevant chunks containing the answer, what percentage did we
#    find in the top-k retrieved list? High recall ensures the LLM has all the facts.
# 2. Precision@k: What percentage of the top-k retrieved chunks are actually relevant?
#    High precision means less filler text is sent to the LLM, saving context space.
# 3. Mean Reciprocal Rank (MRR): Measures where the *first* relevant chunk appears in the list.
#    If the top hit (rank 1) is relevant, the reciprocal rank is 1/1 = 1.0. If the first match
#    appears at rank 4, it is 1/4 = 0.25. MRR is the average across all queries.
# =========================================================================================

from __future__ import annotations # Allow self-referencing type annotations.
from typing import List, Set # Type helpers.

def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes the Recall@k metric.
    """
    # If there are no ground-truth relevant chunks, recall is technically 100%.
    if not relevant_ids:
        return 1.0
        
    # Get the top-k retrieved chunk IDs.
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    # Calculate count of matches in the intersection.
    matches = len(top_k_retrieved.intersection(relevant_set))
    
    # Return matches / total relevant.
    return matches / len(relevant_set)

def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes the Precision@k metric.
    """
    if k <= 0:
        return 0.0
        
    # Get the top-k retrieved chunk IDs.
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    # Calculate matches in intersection.
    matches = len(top_k_retrieved.intersection(relevant_set))
    
    # Return matches / k (size of the retrieved slice).
    return matches / k

def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Computes the Reciprocal Rank (RR) for a single query.
    The caller averages this to find Mean Reciprocal Rank (MRR).
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    
    # Search for the first matching chunk in the retrieved list.
    for idx, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_set:
            # Ranks are 1-indexed, so we add 1 to the loop index.
            # Return reciprocal: 1.0 / rank.
            return 1.0 / (idx + 1)
            
    # Return 0 if no relevant chunk was found.
    return 0.0
