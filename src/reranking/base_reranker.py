from __future__ import annotations 
from abc import ABC, abstractmethod 
from typing import List 
from src.shared.models import FusedResult, RankedResult 
class BaseReranker(ABC):
    """
    Interface for reranking models that score retrieved candidates against a query.
    """
    @abstractmethod
    def rerank(self, query: str, candidates: List[FusedResult], top_n: int) -> List[RankedResult]:
        """
        Reranks a list of FusedResult search hits.
        Returns a list of RankedResult models containing the top_n items.
        """
        pass
