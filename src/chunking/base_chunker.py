# src/chunking/base_chunker.py
# This module defines the base abstract class for all chunking strategies.
# Abstract base classes (ABC) define a strict contract that all subclasses must implement.
# Swapping chunking strategies is seamless because they all share this common interface.

from __future__ import annotations # Allow classes to reference themselves in type hints.
from abc import ABC, abstractmethod # Standard library tools to create abstract base classes and methods.
from typing import List # Type helper for defining lists of objects.
from src.shared.models import Chunk, ParsedFiling # Import our shared Pydantic model formats.

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
