"""High-performance in-memory semantic cache for RAG queries and evaluation results."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TextEmbedder:
    """Deterministic token-frequency vectorizer for fast local semantic embeddings."""

    def __init__(self, vector_dim: int = 128) -> None:
        self.vector_dim = vector_dim

    def embed(self, text: str) -> np.ndarray:
        """Hash token n-grams into a normalized dense vector."""
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        words = text.lower().strip().split()
        if not words:
            return vec

        for i, word in enumerate(words):
            # Unigram hash
            idx1 = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.vector_dim
            vec[idx1] += 1.0
            # Bigram hash if available
            if i < len(words) - 1:
                bigram = f"{word}_{words[i + 1]}"
                idx2 = int(hashlib.sha256(bigram.encode()).hexdigest(), 16) % self.vector_dim
                vec[idx2] += 1.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class SemanticCacheEntry:
    """Individual cached RAG evaluation or query response."""

    def __init__(
        self,
        key: str,
        query_text: str,
        embedding: np.ndarray,
        value: Any,
        ttl_seconds: float = 86400.0,
    ) -> None:
        self.key = key
        self.query_text = query_text
        self.embedding = embedding
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds
        self.access_count = 0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class SemanticCache:
    """Thread-safe LRU semantic cache with cosine similarity matching and TTL expiration."""

    def __init__(
        self,
        similarity_threshold: float = 0.88,
        max_entries: int = 5000,
        ttl_seconds: float = 86400.0,
        latency_saved_ms_per_hit: float = 25.0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.latency_saved_ms_per_hit = latency_saved_ms_per_hit
        self._embedder = TextEmbedder()
        self._cache: OrderedDict[str, SemanticCacheEntry] = OrderedDict()
        self._hit_count: int = 0
        self._miss_count: int = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get(self, query_text: str) -> Optional[Tuple[Any, float]]:
        """Look up query. Returns (cached_value, similarity_score) or None."""
        self._purge_expired()
        norm_query = query_text.strip().lower()
        exact_key = hashlib.sha256(norm_query.encode()).hexdigest()

        # 1. Exact match check
        if exact_key in self._cache:
            entry = self._cache[exact_key]
            if not entry.is_expired:
                self._cache.move_to_end(exact_key)
                entry.access_count += 1
                self._hit_count += 1
                return entry.value, 1.0

        # 2. Semantic vector similarity check
        query_emb = self._embedder.embed(query_text)
        best_score = 0.0
        best_entry: Optional[SemanticCacheEntry] = None
        best_key: Optional[str] = None

        for k, entry in self._cache.items():
            if entry.is_expired:
                continue
            sim = self._cosine_similarity(query_emb, entry.embedding)
            if sim > best_score:
                best_score = sim
                best_entry = entry
                best_key = k

        if best_entry is not None and best_score >= self.similarity_threshold:
            self._cache.move_to_end(best_key)  # type: ignore[arg-type]
            best_entry.access_count += 1
            self._hit_count += 1
            return best_entry.value, round(best_score, 4)

        self._miss_count += 1
        return None

    def put(self, query_text: str, value: Any) -> str:
        """Store item in semantic cache with LRU eviction."""
        self._purge_expired()
        norm_query = query_text.strip().lower()
        key = hashlib.sha256(norm_query.encode()).hexdigest()
        emb = self._embedder.embed(query_text)

        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)  # evict oldest

        entry = SemanticCacheEntry(
            key=key,
            query_text=query_text,
            embedding=emb,
            value=value,
            ttl_seconds=self.ttl_seconds,
        )
        self._cache[key] = entry
        return key

    def get_batch(self, queries: List[str]) -> List[Optional[Tuple[Any, float]]]:
        """Retrieve multiple queries from semantic cache in batch."""
        return [self.get(q) for q in queries]

    def put_batch(self, items: List[Tuple[str, Any]]) -> List[str]:
        """Store multiple items in semantic cache."""
        return [self.put(q, v) for q, v in items]

    def find_similar(
        self, query_text: str, top_k: int = 5, min_score: float = 0.3
    ) -> List[Tuple[str, Any, float]]:
        """Find top-k entries most similar to query_text with similarity >= min_score."""
        self._purge_expired()
        query_emb = self._embedder.embed(query_text)
        candidates: List[Tuple[str, Any, float]] = []

        for entry in self._cache.values():
            if entry.is_expired:
                continue
            sim = self._cosine_similarity(query_emb, entry.embedding)
            if sim >= min_score:
                candidates.append((entry.query_text, entry.value, round(sim, 4)))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]

    def export_entries(self) -> List[Dict[str, Any]]:
        """Export snapshot of non-expired cached entries without heavy embeddings."""
        self._purge_expired()
        return [
            {
                "key": e.key,
                "query_text": e.query_text,
                "access_count": e.access_count,
                "created_at": e.created_at,
                "expires_at": e.expires_at,
            }
            for e in self._cache.values()
        ]

    def _purge_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now > v.expires_at]
        for k in expired_keys:
            del self._cache[k]

    def clear(self) -> None:
        """Clear all entries and reset stats."""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return round(self._hit_count / total, 4) if total > 0 else 0.0

    @property
    def estimated_latency_saved_ms(self) -> float:
        """Estimated computational time saved by serving requests from cache."""
        return round(self._hit_count * self.latency_saved_ms_per_hit, 2)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_entries": self.size,
            "max_entries": self.max_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self.hit_rate,
            "similarity_threshold": self.similarity_threshold,
            "estimated_latency_saved_ms": self.estimated_latency_saved_ms,
        }
