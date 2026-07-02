from __future__ import annotations 
import os 
import time 
from typing import List, Optional 
from openai import OpenAI 
from sentence_transformers import SentenceTransformer 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
log = get_logger(__name__)
class SECEmbedder:
    """
    Computes text embeddings using either the OpenAI API or local BGE models.
    Exposes a unified interface so the indexing and retrieval layers can swap
    models via configuration.
    """
    def __init__(self, provider: Optional[str] = None) -> None:
        self.provider = (provider or cfg.embedding.provider).lower()
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is missing.")
            self.client = OpenAI(api_key=api_key)
        self.bge_model = None
    def _init_bge(self) -> None:
        """
        Lazily loads the BGE model into memory (CPU or GPU).
        """
        if self.bge_model is None:
            log.info("loading_local_bge_model", model=cfg.embedding.bge.model)
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
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            retries = 0
            while retries < cfg.embedding.openai.max_retries:
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=cfg.embedding.openai.model,
                        dimensions=cfg.embedding.openai.dimensions
                    )
                    all_embeddings.extend([item.embedding for item in response.data])
                    break 
                except Exception as exc:
                    retries += 1
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
        self._init_bge()
        embeddings = self.bge_model.encode(
            texts,
            batch_size=cfg.embedding.bge.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True 
        )
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
