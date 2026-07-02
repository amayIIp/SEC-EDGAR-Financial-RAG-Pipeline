from __future__ import annotations
import hashlib          
import time             
from dataclasses import dataclass, field  
from typing import Any, Optional
import numpy as np      
from src.shared.config import cfg
from src.shared.logging_setup import get_logger
log = get_logger(__name__)
@dataclass
class CacheEntry:
    """One cached query result with metadata for TTL and hit-rate tracking."""
    query: str
    result: Any
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    embedding: Optional[np.ndarray] = None
class ExactMatchCache:
    """
    Simple dict-based exact-match cache with optional TTL expiry.
    Thread-safety: not thread-safe by default. For async FastAPI usage the
    event loop is single-threaded, so concurrent access is safe without locks.
    Add asyncio.Lock if you switch to multi-threaded uvicorn workers.
    """
    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._store: dict[str, CacheEntry] = {}
        self.ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
    @staticmethod
    def _make_key(query: str) -> str:
        """
        Normalises a query string and returns a 16-char hex key.
        Normalisation: lowercase + strip whitespace, so "Apple Revenue " and
        "apple revenue" map to the same cache slot.
        """
        normalised = query.lower().strip()
        return hashlib.sha256(normalised.encode()).hexdigest()[:16]
    def get(self, query: str) -> Optional[Any]:
        """
        Returns the cached result for this query, or None on a miss.
        Evicts the entry if it has expired.
        """
        key = self._make_key(query)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if self.ttl_seconds > 0 and (time.time() - entry.created_at) > self.ttl_seconds:
            del self._store[key]
            self._misses += 1
            log.debug("exact_cache_expired", query=query[:60])
            return None
        entry.hit_count += 1
        self._hits += 1
        log.info("exact_cache_hit", query=query[:60], hit_count=entry.hit_count)
        return entry.result
    def set(self, query: str, result: Any, embedding: Optional[np.ndarray] = None) -> None:
        """Stores a result in the exact-match cache."""
        key = self._make_key(query)
        self._store[key] = CacheEntry(
            query=query,
            result=result,
            embedding=embedding,
        )
        log.debug("exact_cache_set", query=query[:60], total_entries=len(self._store))
    def invalidate(self, query: str) -> None:
        """Removes a specific entry from the cache (e.g. after a corpus update)."""
        key = self._make_key(query)
        self._store.pop(key, None)
    def clear(self) -> None:
        """Removes all entries — useful between eval runs to avoid cross-contamination."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
    @property
    def hit_rate(self) -> float:
        """Fraction of get() calls that returned a cached result."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    def stats(self) -> dict[str, Any]:
        """Returns a diagnostic summary of cache state."""
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }
class SemanticCache:
    """
    Nearest-neighbour cache that returns hits for semantically similar queries
    even when the exact string doesn't match.
    Algorithm:
      1. On get(query_embedding), compute cosine similarity between the query
         embedding and every stored embedding.
      2. If the maximum similarity ≥ threshold, return that entry's result.
      3. On set(), store the embedding alongside the result for future comparisons.
    The max_entries limit prevents unbounded memory growth. When the limit is
    reached, the oldest entries are evicted (FIFO order).
    """
    def __init__(
        self,
        similarity_threshold: float | None = None,
        max_entries: int | None = None,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self.threshold = similarity_threshold if similarity_threshold is not None            else cfg.profiling.semantic_cache_threshold
        self.max_entries = max_entries if max_entries is not None            else cfg.profiling.semantic_cache_max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: list[CacheEntry] = []
        self._hits = 0
        self._misses = 0
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Computes the cosine similarity between two 1-D numpy float arrays.
        Cosine similarity = dot(a, b) / (||a|| × ||b||).
        Range: -1.0 (opposite direction) to 1.0 (identical direction).
        For normalised embeddings (||v|| = 1), this equals the dot product.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    def _evict_expired(self) -> None:
        """Removes entries whose TTL has elapsed."""
        if self.ttl_seconds <= 0:
            return
        now = time.time()
        self._entries = [
            e for e in self._entries
            if (now - e.created_at) <= self.ttl_seconds
        ]
    def get(self, query_embedding: np.ndarray) -> Optional[Any]:
        """
        Returns the cached result whose stored embedding is most similar to
        query_embedding, provided the similarity is ≥ self.threshold.
        Returns None on a miss.
        """
        self._evict_expired()
        if not self._entries:
            self._misses += 1
            return None
        embedding_matrix = np.stack([e.embedding for e in self._entries if e.embedding is not None])
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            self._misses += 1
            return None
        q_unit = query_embedding / q_norm
        row_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        normed_matrix = embedding_matrix / row_norms
        similarities = normed_matrix @ q_unit
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        if best_sim >= self.threshold:
            entry = self._entries[best_idx]
            entry.hit_count += 1
            self._hits += 1
            log.info(
                "semantic_cache_hit",
                similarity=round(best_sim, 4),
                threshold=self.threshold,
                matched_query=entry.query[:60],
            )
            return entry.result
        self._misses += 1
        return None
    def set(self, query: str, query_embedding: np.ndarray, result: Any) -> None:
        """Stores a result with its embedding for future semantic comparisons."""
        self._evict_expired()
        while len(self._entries) >= self.max_entries:
            evicted = self._entries.pop(0)
            log.debug("semantic_cache_evict", evicted_query=evicted.query[:60])
        self._entries.append(
            CacheEntry(query=query, result=result, embedding=query_embedding)
        )
        log.debug(
            "semantic_cache_set",
            query=query[:60],
            total_entries=len(self._entries),
        )
    def clear(self) -> None:
        """Flushes all entries — call between eval run configurations."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "threshold": self.threshold,
        }
class QueryCache:
    """
    Facade that checks exact-match first, then semantic, then returns None.
    Callers only interact with this class — the tier split is invisible.
    USAGE:
        cache = QueryCache()
        result = cache.get(query_text, query_embedding)
        if result is not None:
            return result          # served from cache
        result = await run_pipeline(query_text)
        cache.set(query_text, query_embedding, result)
        return result
    """
    def __init__(self) -> None:
        self.exact = ExactMatchCache()
        self.semantic = SemanticCache()
    def get(self, query: str, query_embedding: np.ndarray) -> Optional[Any]:
        """
        Check exact-match cache first (O(1)), then semantic cache (O(n) but fast
        for n ≤ 1000 with numpy vectorisation).
        """
        exact_result = self.exact.get(query)
        if exact_result is not None:
            return exact_result
        if cfg.profiling.semantic_cache_threshold < 1.0:
            semantic_result = self.semantic.get(query_embedding)
            if semantic_result is not None:
                return semantic_result
        return None
    def set(self, query: str, query_embedding: np.ndarray, result: Any) -> None:
        """Stores in both cache tiers simultaneously."""
        self.exact.set(query, result, embedding=query_embedding)
        self.semantic.set(query, query_embedding, result)
    def clear(self) -> None:
        """Flushes both tiers."""
        self.exact.clear()
        self.semantic.clear()
    def stats(self) -> dict[str, Any]:
        return {
            "exact": self.exact.stats(),
            "semantic": self.semantic.stats(),
        }
