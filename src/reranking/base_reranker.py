# src/reranking/base_reranker.py
# This module defines the base abstract class for all rerankers.
# Swapping between cloud-based and self-hosted reranking models is seamless
# because they all implement this common interface.

from __future__ import annotations # Allow self-referencing type annotations.
from abc import ABC, abstractmethod # Standard library module to create abstract classes.
from typing import List # Type helper.
from src.shared.models import FusedResult, RankedResult # Shared models.

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
