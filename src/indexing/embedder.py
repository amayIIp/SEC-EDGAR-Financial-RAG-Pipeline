# src/indexing/embedder.py
# This module implements the text embedding functionality.
# Text embeddings are dense, low-dimensional vector representations of text.
# By translating sentences into lists of floating-point numbers (vectors),
# we can compute mathematical distances (like Cosine Similarity) to search for
# semantically similar text passages, even if they share no exact words.
# We support two backends:
# 1. OpenAI (text-embedding-3-small) via cloud API.
# 2. BAAI/bge-large-en-v1.5 run locally via the sentence-transformers library.

from __future__ import annotations # Allow self-referencing type annotations.
import os # Standard library module to read environment variables.
import time # Standard library module to sleep during retries.
from typing import List, Optional # Type helper for lists and nullables.
from openai import OpenAI # Official OpenAI Python SDK client.
from sentence_transformers import SentenceTransformer # Library to load local embedding models.
from src.shared.config import cfg # Config loader singleton.
from src.shared.logging_setup import get_logger # Logger wrapper.

log = get_logger(__name__)

class SECEmbedder:
    """
    Computes text embeddings using either the OpenAI API or local BGE models.
    Exposes a unified interface so the indexing and retrieval layers can swap
    models via configuration.
    """

    def __init__(self, provider: Optional[str] = None) -> None:
        # Determine the active provider (fall back to configuration settings if not specified).
        self.provider = (provider or cfg.embedding.provider).lower()
        
        # Initialize the OpenAI API client if OpenAI is selected.
        if self.provider == "openai":
            # Read the API key from environment variables.
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is missing.")
            self.client = OpenAI(api_key=api_key)
            
        # Placeholder for local BGE model, loaded lazily to save memory.
        self.bge_model = None

    def _init_bge(self) -> None:
        """
        Lazily loads the BGE model into memory (CPU or GPU).
        """
        if self.bge_model is None:
            log.info("loading_local_bge_model", model=cfg.embedding.bge.model)
            # Load the SentenceTransformer model locally.
            # It automatically detects if CUDA (GPU) is available and shifts parameters.
            self.bge_model = SentenceTransformer(
                cfg.embedding.bge.model,
                device=cfg.embedding.bge.device if cfg.embedding.bge.device != "auto" else None
            )
            log.info("bge_model_loaded_successfully")

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """
        Calculates embeddings using the OpenAI API, implementing batching and retries.
        """
        batch_size = cfg.embedding.openai.batch_size
        all_embeddings: List[List[float]] = []
        
        # Process the input texts in batches.
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Implementation of retry loop with exponential backoff on API errors.
            retries = 0
            while retries < cfg.embedding.openai.max_retries:
                try:
                    # Request embeddings from the API.
                    response = self.client.embeddings.create(
                        input=batch,
                        model=cfg.embedding.openai.model,
                        dimensions=cfg.embedding.openai.dimensions
                    )
                    # Extract the floats list and append.
                    all_embeddings.extend([item.embedding for item in response.data])
                    break # Break retry loop on success.
                except Exception as exc:
                    retries += 1
                    # Exponential backoff calculation.
                    delay = cfg.embedding.openai.retry_base_delay * (2 ** retries)
                    log.warning("openai_embedding_retry", attempt=retries, delay_sec=delay, error=str(exc))
                    time.sleep(delay)
                    
            if retries >= cfg.embedding.openai.max_retries:
                raise RuntimeError(f"OpenAI embedding failed after {retries} attempts.")
                
        return all_embeddings

    def _embed_bge(self, texts: List[str]) -> List[List[float]]:
        """
        Calculates embeddings locally using BGE.
        """
        # Ensure the model is loaded.
        self._init_bge()
        
        # sentence-transformers expects query formatting for BGE models to maximize search recall.
        # We prepend the query instruction if this is a query embedding task.
        # However, for document index tasks, we don't modify the text.
        # To handle this cleanly, we can check if texts contains only 1 item (typical query) and prepend,
        # or expose a query flag. We default to simple direct encoding.
        embeddings = self.bge_model.encode(
            texts,
            batch_size=cfg.embedding.bge.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True # Normalise to length 1 (makes dot-product equivalent to Cosine similarity).
        )
        
        # Convert numpy array to standard float list.
        return embeddings.tolist()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text strings using the active provider.
        """
        if not texts:
            return []
            
        if self.provider == "openai":
            return self._embed_openai(texts)
        elif self.provider == "bge":
            return self._embed_bge(texts)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def get_dimension(self) -> int:
        """
        Returns the vector dimension size matching the active embedding model.
        """
        if self.provider == "openai":
            return cfg.embedding.openai.dimensions
        elif self.provider == "bge":
            return cfg.embedding.bge.dimensions
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")
