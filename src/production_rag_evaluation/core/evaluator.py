"""Top-level RAG evaluator coordinating semantic caching and DAG workflow."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.core.graph import EvaluationGraph
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.models.schemas import (
    BatchEvaluationSummary,
    EvaluationMetrics,
    EvaluationMode,
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
        default_mode: EvaluationMode = EvaluationMode.THOROUGH,
    ) -> None:
        self.enable_cache = enable_cache
        self.default_mode = default_mode
        self.cache = SemanticCache(
            similarity_threshold=cache_similarity_threshold,
            max_entries=cache_max_entries,
            ttl_seconds=cache_ttl_seconds,
        )
        self.detector = HallucinationDetector(numerical_tolerance_pct=numerical_tolerance_pct)
        self.graph = EvaluationGraph(detector=self.detector)

    def evaluate(
        self,
        query: RAGQuery,
        response: RAGResponse,
        use_cache: Optional[bool] = None,
        mode: Optional[EvaluationMode] = None,
    ) -> EvaluationMetrics:
        """Evaluate a single query and response pair."""
        eval_mode = mode or self.default_mode
        should_cache = self.enable_cache if use_cache is None else use_cache
        cache_key = f"[{eval_mode.value}] {query.query_text} || {response.generated_answer}"

        if should_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                val, _score = cached_result
                if isinstance(val, EvaluationMetrics):
                    return val

        # Execute evaluation graph with specified mode
        metrics = self.graph.run(query=query, response=response, mode=eval_mode)

        if should_cache:
            self.cache.put(cache_key, metrics)

        return metrics

    def evaluate_batch(
        self,
        pairs: List[Tuple[RAGQuery, RAGResponse]],
        use_cache: Optional[bool] = None,
        mode: Optional[EvaluationMode] = None,
    ) -> List[EvaluationMetrics]:
        """Evaluate a batch of query-response pairs."""
        return [self.evaluate(q, r, use_cache=use_cache, mode=mode) for q, r in pairs]

    def evaluate_batch_summary(
        self,
        pairs: List[Tuple[RAGQuery, RAGResponse]],
        batch_id: Optional[str] = None,
        use_cache: Optional[bool] = None,
        mode: Optional[EvaluationMode] = None,
    ) -> BatchEvaluationSummary:
        """Evaluate a batch and return aggregated summary metrics."""
        b_id = batch_id or f"batch-{uuid.uuid4().hex[:8]}"
        results = self.evaluate_batch(pairs, use_cache=use_cache, mode=mode)
        total = len(results)

        if total == 0:
            return BatchEvaluationSummary(
                batch_id=b_id,
                total_queries=0,
                passed_queries=0,
                failed_queries=0,
                pass_rate=0.0,
                mean_faithfulness=0.0,
                mean_relevance=0.0,
                mean_latency_ms=0.0,
                p95_latency_ms=0.0,
                results=[],
            )

        passed = sum(1 for m in results if m.overall_pass)
        failed = total - passed
        pass_rate = round(passed / total, 4)
        mean_faith = round(float(np.mean([m.faithfulness_score for m in results])), 4)
        mean_rel = round(float(np.mean([m.answer_relevance_score for m in results])), 4)
        latencies = [m.evaluation_latency_ms for m in results]
        mean_lat = round(float(np.mean(latencies)), 2)
        p95_lat = round(float(np.percentile(latencies, 95)), 2)

        return BatchEvaluationSummary(
            batch_id=b_id,
            total_queries=total,
            passed_queries=passed,
            failed_queries=failed,
            pass_rate=pass_rate,
            mean_faithfulness=mean_faith,
            mean_relevance=mean_rel,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            results=results,
        )

    def clear_cache(self) -> None:
        """Clear semantic cache entries."""
        self.cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        return self.cache.stats()
