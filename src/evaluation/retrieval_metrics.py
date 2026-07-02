from __future__ import annotations 
from typing import List, Set 
def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes the Recall@k metric.
    """
    if not relevant_ids:
        return 1.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    matches = len(top_k_retrieved.intersection(relevant_set))
    return matches / len(relevant_set)
def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes the Precision@k metric.
    """
    if k <= 0:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    matches = len(top_k_retrieved.intersection(relevant_set))
    return matches / k
def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Computes the Reciprocal Rank (RR) for a single query.
    The caller averages this to find Mean Reciprocal Rank (MRR).
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for idx, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_set:
            return 1.0 / (idx + 1)
    return 0.0
