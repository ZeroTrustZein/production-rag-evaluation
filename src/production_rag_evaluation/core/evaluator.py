"""Top-level RAG evaluator coordinating semantic caching and DAG workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.core.graph import EvaluationGraph
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.models.schemas import (
    EvaluationMetrics,
    RAGQuery,
    RAGResponse,
)


class RAGEvaluator:
    """Production evaluation orchestrator for financial RAG systems."""

    def __init__(
        self,
        cache_similarity_threshold: float = 0.90,
        cache_max_entries: int = 5000,
        cache_ttl_seconds: float = 86400.0,
        numerical_tolerance_pct: float = 0.01,
        enable_cache: bool = True,
    ) -> None:
        self.enable_cache = enable_cache
        self.cache = SemanticCache(
            similarity_threshold=cache_similarity_threshold,
            max_entries=cache_max_entries,
            ttl_seconds=cache_ttl_seconds,
        )
        self.detector = HallucinationDetector(numerical_tolerance_pct=numerical_tolerance_pct)
        self.graph = EvaluationGraph(detector=self.detector)

    def evaluate(
        self, query: RAGQuery, response: RAGResponse, use_cache: Optional[bool] = None
    ) -> EvaluationMetrics:
        """Evaluate a single query and response pair."""
        should_cache = self.enable_cache if use_cache is None else use_cache
        cache_key = f"{query.query_text} || {response.generated_answer}"

        if should_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                val, score = cached_result
                if isinstance(val, EvaluationMetrics):
                    return val

        # Execute evaluation graph
        metrics = self.graph.run(query=query, response=response)

        if should_cache:
            self.cache.put(cache_key, metrics)

        return metrics

    def evaluate_batch(
        self,
        pairs: List[Tuple[RAGQuery, RAGResponse]],
        use_cache: Optional[bool] = None,
    ) -> List[EvaluationMetrics]:
        """Evaluate a batch of query-response pairs."""
        return [self.evaluate(q, r, use_cache=use_cache) for q, r in pairs]

    def clear_cache(self) -> None:
        """Clear semantic cache entries."""
        self.cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        return self.cache.stats()
