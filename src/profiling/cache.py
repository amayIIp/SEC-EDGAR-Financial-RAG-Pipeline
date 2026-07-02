# src/profiling/cache.py
#
# Two-tier query cache that sits in front of the hybrid_search + rerank pipeline.
#
# TIER 1 — Exact-match cache
#   Key: the raw query string (lowercased, stripped).
#   Hit condition: the exact same string was asked before.
#   Storage: in-process dict (dev) or Redis hash (prod).
#   Use-case: repeated queries in a demo session ("show me this again").
#
# TIER 2 — Semantic cache
#   Key: the query's embedding vector.
#   Hit condition: the new query's embedding has cosine similarity ≥ threshold
#   with a previously cached query's embedding.
#   Storage: in-process list of (embedding, cached_result) pairs.
#   Use-case: paraphrased queries ("Apple's FY23 revenue" vs "Apple revenue 2023").
#   Risk: if threshold is too low, semantically related but factually different
#   queries return wrong cached answers. Default threshold = 0.97 (conservative).
#
# DESIGN DECISION: why not just use Redis for both tiers?
#   The semantic cache requires a nearest-neighbour search over cached embeddings.
#   A small in-process list + numpy cosine is fast enough at ≤1000 entries and
#   adds zero infrastructure dependency for local dev. Swap to Qdrant or FAISS
#   if the cache grows beyond ~10k entries in production.

from __future__ import annotations

import hashlib          # standard-library: SHA-256 key for exact-match cache
import time             # standard-library: TTL expiry timestamps
from dataclasses import dataclass, field  # standard-library: lightweight container
from typing import Any, Optional

import numpy as np      # cosine similarity computation over embedding arrays

from src.shared.config import cfg
from src.shared.logging_setup import get_logger

log = get_logger(__name__)


# =============================================================================
# Cache entry dataclass
# =============================================================================

@dataclass
class CacheEntry:
    """One cached query result with metadata for TTL and hit-rate tracking."""

    # The original query string (exact-match key or the query that generated the vector).
    query: str

    # The cached pipeline result — whatever hybrid_search + rerank returned.
    result: Any

    # Unix timestamp (float) when this entry was created. Used for TTL expiry.
    created_at: float = field(default_factory=time.time)

    # Number of times this cache entry has been returned as a hit.
    hit_count: int = 0

    # The query embedding (numpy array) used for semantic similarity comparison.
    # None if this entry was created by the exact-match tier only.
    embedding: Optional[np.ndarray] = None


# =============================================================================
# Exact-match cache
# =============================================================================

class ExactMatchCache:
    """
    Simple dict-based exact-match cache with optional TTL expiry.

    Thread-safety: not thread-safe by default. For async FastAPI usage the
    event loop is single-threaded, so concurrent access is safe without locks.
    Add asyncio.Lock if you switch to multi-threaded uvicorn workers.
    """

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        # The underlying storage: query_key → CacheEntry.
        self._store: dict[str, CacheEntry] = {}
        # How long (seconds) a cached result stays valid. 0 = never expire.
        self.ttl_seconds = ttl_seconds
        # Track total hits and misses for diagnostics.
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
        # SHA-256 then truncate to 16 chars — collision probability is negligible
        # for a cache of ≤1 million entries.
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

        # Check TTL expiry — evict and return None if the entry is stale.
        if self.ttl_seconds > 0 and (time.time() - entry.created_at) > self.ttl_seconds:
            del self._store[key]
            self._misses += 1
            log.debug("exact_cache_expired", query=query[:60])
            return None

        # Cache hit — increment counters and return the stored result.
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


# =============================================================================
# Semantic cache
# =============================================================================

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
        # Similarity threshold from config — conservative default (0.97) avoids
        # returning wrong answers for related-but-different questions.
        self.threshold = similarity_threshold if similarity_threshold is not None \
            else cfg.profiling.semantic_cache_threshold

        # Maximum number of entries before FIFO eviction kicks in.
        self.max_entries = max_entries if max_entries is not None \
            else cfg.profiling.semantic_cache_max_entries

        self.ttl_seconds = ttl_seconds

        # Ordered list of CacheEntry objects (newest at the end).
        self._entries: list[CacheEntry] = []

        # Diagnostics.
        self._hits = 0
        self._misses = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Computes the cosine similarity between two 1-D numpy float arrays.
        Cosine similarity = dot(a, b) / (||a|| × ||b||).
        Range: -1.0 (opposite direction) to 1.0 (identical direction).
        For normalised embeddings (||v|| = 1), this equals the dot product.
        """
        # np.linalg.norm computes the Euclidean (L2) magnitude of the vector.
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            # A zero vector has no direction — similarity is undefined; return 0.
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

        # Stack all stored embeddings into a 2-D matrix for vectorised dot-product.
        # Shape: (num_entries, embedding_dim).
        embedding_matrix = np.stack([e.embedding for e in self._entries if e.embedding is not None])

        # Compute cosine similarity between the query and every stored embedding.
        # We normalise the query vector once and use matrix multiplication for speed.
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            self._misses += 1
            return None

        q_unit = query_embedding / q_norm

        # Normalise each row of the stored embedding matrix.
        row_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        # Avoid division-by-zero for any all-zero rows (edge case).
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        normed_matrix = embedding_matrix / row_norms

        # Matrix × vector → (num_entries,) array of cosine similarities.
        similarities = normed_matrix @ q_unit

        # Find the entry with the highest similarity.
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
        # Evict expired entries before adding a new one.
        self._evict_expired()

        # Evict the oldest entry if we are at capacity (FIFO).
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


# =============================================================================
# Combined two-tier cache facade
# =============================================================================

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
        # Tier 1: exact-match.
        exact_result = self.exact.get(query)
        if exact_result is not None:
            return exact_result

        # Tier 2: semantic similarity.
        if cfg.profiling.semantic_cache_threshold < 1.0:
            semantic_result = self.semantic.get(query_embedding)
            if semantic_result is not None:
                return semantic_result

        return None

    def set(self, query: str, query_embedding: np.ndarray, result: Any) -> None:
        """Stores in both cache tiers simultaneously."""
        # Pass the embedding to exact cache so it can forward to semantic on promotion.
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
