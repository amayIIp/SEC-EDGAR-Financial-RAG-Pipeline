from __future__ import annotations 
from abc import ABC, abstractmethod 
from typing import List 
from src.shared.models import Chunk, ParsedFiling 
class BaseChunker(ABC):
    """
    Interface for document chunkers that turn a parsed filing into text chunks.
    """
    @abstractmethod
    def chunk(self, parsed_filing: ParsedFiling) -> List[Chunk]:
        """
        Splits a structured ParsedFiling into a list of indexable Chunk objects.
        Must be implemented by subclasses.
        """
        pass
