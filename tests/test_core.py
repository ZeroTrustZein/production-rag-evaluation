"""Comprehensive unit test suite for core algorithms and primary logic."""

from __future__ import annotations

import time

import numpy as np

from production_rag_evaluation.core.cache import SemanticCache, TextEmbedder
from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.core.graph import EvaluationGraph, EvaluationState
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.models.schemas import (
    Citation,
    ClaimStatus,
    DocumentChunk,
    EvaluationMode,
    HallucinationType,
    NumericalTarget,
    RAGQuery,
    RAGResponse,
    Severity,
)


class TestTextEmbedderAndSemanticCache:
    """Validate vector hashing, cosine similarity, LRU cache and batch operations."""

    def test_text_embedder_deterministic_and_normalized(self) -> None:
        embedder = TextEmbedder(vector_dim=64)
        vec1 = embedder.embed("Apple revenue reached 391 billion dollars")
        vec2 = embedder.embed("Apple revenue reached 391 billion dollars")
        vec_diff = embedder.embed("Semiconductor chip fabrication plants")

        assert isinstance(vec1, np.ndarray)
        assert vec1.shape == (64,)
        np.testing.assert_allclose(vec1, vec2, rtol=1e-5)
        assert abs(float(np.linalg.norm(vec1)) - 1.0) < 1e-4

        # Dot product with different text should be lower than identical text
        sim_same = float(np.dot(vec1, vec2))
        sim_diff = float(np.dot(vec1, vec_diff))
        assert sim_same > sim_diff

    def test_text_embedder_empty_string(self) -> None:
        embedder = TextEmbedder(vector_dim=32)
        vec = embedder.embed("")
        assert vec.shape == (32,)
        assert float(np.linalg.norm(vec)) == 0.0

    def test_cache_batch_get_and_put(self) -> None:
        cache = SemanticCache(similarity_threshold=0.85, max_entries=10)
        items = [
            ("query one", {"score": 1}),
            ("query two", {"score": 2}),
            ("query three", {"score": 3}),
        ]
        keys = cache.put_batch(items)
        assert len(keys) == 3
        assert cache.size == 3

        batch_results = cache.get_batch(["query one", "query two", "missing query"])
        assert len(batch_results) == 3
        assert batch_results[0] is not None
        assert batch_results[0][0] == {"score": 1}
        assert batch_results[1] is not None
        assert batch_results[1][0] == {"score": 2}
        assert batch_results[2] is None

    def test_cache_find_similar(self) -> None:
        cache = SemanticCache(similarity_threshold=0.70)
        cache.put("Apple fiscal 2024 revenue", "val1")
        cache.put("Apple FY24 net sales results", "val2")
        cache.put("Tesla deliveries in Shanghai", "val3")

        results = cache.find_similar("Apple 2024 revenue growth", top_k=2, min_score=0.4)
        assert len(results) >= 1
        assert "Apple" in results[0][0]

    def test_cache_export_entries_and_latency_saved(self) -> None:
        cache = SemanticCache(latency_saved_ms_per_hit=30.0)
        cache.put("Test query", "val")
        cache.get("Test query")  # 1 hit

        exported = cache.export_entries()
        assert len(exported) == 1
        assert exported[0]["query_text"] == "Test query"
        assert exported[0]["access_count"] == 1

        stats = cache.stats()
        assert stats["hit_count"] == 1
        assert stats["estimated_latency_saved_ms"] == 30.0

    def test_cache_ttl_expiration(self) -> None:
        cache = SemanticCache(ttl_seconds=0.01)
        cache.put("quick expire query", "quick val")
        time.sleep(0.02)
        assert cache.get("quick expire query") is None


class TestHallucinationDetectorLogic:
    """Test domain-specific financial distortion, contradiction, and entity/period checks."""

    def test_numerical_extraction_scales(self) -> None:
        detector = HallucinationDetector()
        text = "Revenue was $10.5 billion, margin was 42.5%, profit was 2.1 million, debt was 500k, and market cap was $3.2 trillion."
        nums = detector._extract_numerical_values(text)

        val_map = {raw: val for val, unit, raw in nums}
        assert any("10.5 billion" in k for k in val_map)
        assert any("42.5%" in k for k in val_map)
        # Check scale multipliers
        for val, unit, raw in nums:
            if "billion" in raw:
                assert abs(val - 10.5e9) < 1.0
            elif "trillion" in raw:
                assert abs(val - 3.2e12) < 1.0
            elif "million" in raw:
                assert abs(val - 2.1e6) < 1.0
            elif "500k" in raw:
                assert abs(val - 500e3) < 1.0

    def test_directional_contradiction_negative_context_positive_answer(
        self, sample_chunk_apple: DocumentChunk
    ) -> None:
        detector = HallucinationDetector()
        chunk_down = DocumentChunk(
            chunk_id="c-down",
            text="Operating profit decreased and dropped by 15% due to supply headwinds.",
            ticker="AAPL",
        )
        answer_up = "Operating profit increased, expanded and grew."
        findings = detector.detect_directional_contradictions(answer_up, [chunk_down])
        assert len(findings) > 0
        assert findings[0].finding_type == HallucinationType.CONTRADICTION

    def test_directional_contradiction_positive_context_negative_answer(self) -> None:
        detector = HallucinationDetector()
        chunk_up = DocumentChunk(
            chunk_id="c-up",
            text="Operating profit grew and increased by 20% year over year.",
            ticker="AAPL",
        )
        answer_down = "Operating profit decreased and dropped significantly."
        findings = detector.detect_directional_contradictions(answer_down, [chunk_up])
        assert len(findings) > 0
        assert findings[0].finding_type == HallucinationType.CONTRADICTION

    def test_entity_swap_detection(self, sample_chunk_apple: DocumentChunk) -> None:
        detector = HallucinationDetector()
        # Answer references Microsoft (MSFT) when chunk is AAPL
        answer_with_swap = "MSFT reported net sales of $391.0 billion in fiscal year 2024."
        findings = detector.detect_entity_swaps(answer_with_swap, [sample_chunk_apple])

        assert len(findings) > 0
        swap_finding = findings[0]
        assert swap_finding.finding_type == HallucinationType.ENTITY_SWAP
        assert swap_finding.severity == Severity.HIGH
        assert "MSFT" in swap_finding.claim_text

    def test_entity_swap_clean_when_ticker_matches(self, sample_chunk_apple: DocumentChunk) -> None:
        detector = HallucinationDetector()
        clean_answer = "AAPL reported net sales of $391.0 billion."
        findings = detector.detect_entity_swaps(clean_answer, [sample_chunk_apple])
        assert len(findings) == 0

    def test_temporal_mismatch_detection(self, sample_chunk_apple: DocumentChunk) -> None:
        detector = HallucinationDetector()
        # Context is FY2024 and 2023, answer claims 2019
        answer_with_mismatch = "In 2019, Apple reported net sales of $391.0 billion."
        findings = detector.detect_temporal_mismatches(answer_with_mismatch, [sample_chunk_apple])

        assert len(findings) > 0
        assert findings[0].finding_type == HallucinationType.TEMPORAL_MISMATCH
        assert findings[0].severity == Severity.MEDIUM
        assert "2019" in findings[0].claim_text

    def test_verify_claims_verified_and_refuted(self, sample_chunk_apple: DocumentChunk) -> None:
        detector = HallucinationDetector()
        claims = [
            "Apple reported net sales of $391.0 billion.",  # Verified
            "Apple sales decreased and declined.",  # Refuted by contradiction
            "Space exploration missions were launched.",  # Ungrounded
        ]
        results = detector.verify_claims(claims, [sample_chunk_apple])
        assert len(results) == 3

        assert results[0].status == ClaimStatus.VERIFIED
        assert sample_chunk_apple.chunk_id in results[0].evidence_chunk_ids

        assert results[1].status == ClaimStatus.REFUTED
        assert results[2].status == ClaimStatus.UNGROUNDED

    def test_analyze_aggregates_all_rules(self, sample_chunk_apple: DocumentChunk) -> None:
        detector = HallucinationDetector()
        # Answer with both entity swap and numerical distortion
        dirty_answer = "MSFT reported net sales of $999.0 billion in fiscal year 2024."
        findings = detector.analyze(dirty_answer, [sample_chunk_apple])

        types = {f.finding_type for f in findings}
        assert HallucinationType.ENTITY_SWAP in types
        assert HallucinationType.NUMERICAL_DISTORTION in types


class TestEvaluationGraphDAG:
    """Test DAG nodes, state machine, and structured targets verification."""

    def test_dag_execution_trace_order(self, sample_chunk_apple: DocumentChunk) -> None:
        graph = EvaluationGraph()
        q = RAGQuery(query_id="q-001", query_text="What were Apple's net sales in 2024?", ticker="AAPL")
        r = RAGResponse(
            query_id="q-001",
            generated_answer="Apple reported net sales of $391.0 billion.",
            retrieved_chunks=[sample_chunk_apple],
        )

        metrics = graph.run(q, r)
        trace = metrics.metadata.get("execution_trace", [])
        expected_nodes = [
            "extract_claims",
            "verify_citations",
            "detect_hallucinations",
            "verify_claims",
            "verify_targets",
            "compute_relevance",
            "aggregate_metrics",
        ]
        for node in expected_nodes:
            assert node in trace

    def test_structured_targets_satisfied(self, sample_chunk_apple: DocumentChunk) -> None:
        graph = EvaluationGraph()
        target = NumericalTarget(
            metric_name="net_sales",
            target_value=391.0e9,
            unit="USD",
            tolerance_pct=0.01,
        )
        q = RAGQuery(
            query_id="q-tgt",
            query_text="What was Apple's net sales?",
            ticker="AAPL",
            structured_targets=[target],
        )
        r = RAGResponse(
            query_id="q-tgt",
            generated_answer="Apple reported net sales of $391.0 billion.",
            retrieved_chunks=[sample_chunk_apple],
        )

        metrics = graph.run(q, r)
        target_checks = metrics.metadata.get("target_checks", {})
        assert "net_sales" in target_checks
        assert target_checks["net_sales"]["matched"] is True

    def test_raw_expected_numerical_targets(self, sample_chunk_apple: DocumentChunk) -> None:
        graph = EvaluationGraph()
        q = RAGQuery(
            query_id="q-raw-tgt",
            query_text="What were net sales?",
            ticker="AAPL",
            expected_numerical_targets=[391.0e9],
        )
        r = RAGResponse(
            query_id="q-raw-tgt",
            generated_answer="Apple reported net sales of $391.0 billion.",
            retrieved_chunks=[sample_chunk_apple],
        )
        metrics = graph.run(q, r)
        target_checks = metrics.metadata.get("target_checks", {})
        assert str(391.0e9) in target_checks
        assert target_checks[str(391.0e9)]["matched"] is True

    def test_structured_targets_missed_triggers_finding(self, sample_chunk_apple: DocumentChunk) -> None:
        graph = EvaluationGraph()
        target = NumericalTarget(
            metric_name="operating_cash_flow",
            target_value=120.0e9,
            tolerance_pct=0.01,
        )
        q = RAGQuery(
            query_id="q-tgt-miss",
            query_text="What was operating cash flow?",
            ticker="AAPL",
            structured_targets=[target],
        )
        r = RAGResponse(
            query_id="q-tgt-miss",
            generated_answer="Apple reported net sales of $391.0 billion.",
            retrieved_chunks=[sample_chunk_apple],
        )

        metrics = graph.run(q, r)
        target_checks = metrics.metadata.get("target_checks", {})
        assert target_checks["operating_cash_flow"]["matched"] is False
        assert any("operating_cash_flow" in f.explanation for f in metrics.findings)

    def test_evaluation_modes(self, sample_chunk_apple: DocumentChunk) -> None:
        graph = EvaluationGraph()
        q = RAGQuery(query_id="q-mode", query_text="What were sales?", ticker="AAPL")
        citation = Citation(chunk_id=sample_chunk_apple.chunk_id, quote="$391.0 billion")
        r = RAGResponse(
            query_id="q-mode",
            generated_answer="Apple reported net sales of $391.0 billion.",
            retrieved_chunks=[sample_chunk_apple],
            citations=[citation],
        )

        m_fast = graph.run(q, r, mode=EvaluationMode.FAST)
        assert m_fast.metadata["evaluation_mode"] == "FAST"
        assert m_fast.overall_pass is True

        m_strict = graph.run(q, r, mode=EvaluationMode.FINANCIAL_STRICT)
        assert m_strict.metadata["evaluation_mode"] == "FINANCIAL_STRICT"

        m_audit = graph.run(q, r, mode=EvaluationMode.AUDIT)
        assert m_audit.metadata["evaluation_mode"] == "AUDIT"
        assert len(m_audit.claim_verifications) > 0

    def test_evaluation_state_initialization(self, sample_chunk_apple: DocumentChunk) -> None:
        q = RAGQuery(query_id="q-state", query_text="Query text", ticker="AAPL")
        r = RAGResponse(query_id="q-state", generated_answer="Sample answer")
        state = EvaluationState(query=q, response=r, mode=EvaluationMode.AUDIT)
        assert state.mode == EvaluationMode.AUDIT
        assert state.overall_pass is True
        assert state.claims == []
        assert state.claim_verifications == []


class TestRAGEvaluatorBatchAndCaching:
    """Test top-level orchestrator batch summary and cache mechanics."""

    def test_evaluator_cache_hit_and_stats(self, sample_chunk_apple: DocumentChunk) -> None:
        evaluator = RAGEvaluator(enable_cache=True)
        q = RAGQuery(query_id="q-c", query_text="What was Apple's gross margin?", ticker="AAPL")
        r = RAGResponse(
            query_id="q-c",
            generated_answer="Apple gross margin was 46.2%.",
            retrieved_chunks=[sample_chunk_apple],
        )

        m1 = evaluator.evaluate(q, r)
        stats1 = evaluator.cache_stats()
        assert stats1["hit_count"] == 0

        m2 = evaluator.evaluate(q, r)
        stats2 = evaluator.cache_stats()
        assert stats2["hit_count"] == 1
        assert m2.faithfulness_score == m1.faithfulness_score

    def test_evaluator_evaluate_batch_summary(self, sample_chunk_apple: DocumentChunk) -> None:
        evaluator = RAGEvaluator(enable_cache=False)
        q1 = RAGQuery(query_id="q-1", query_text="What were net sales?", ticker="AAPL")
        r1 = RAGResponse(
            query_id="q-1",
            generated_answer="Net sales were $391.0 billion.",
            retrieved_chunks=[sample_chunk_apple],
        )
        q2 = RAGQuery(query_id="q-2", query_text="What was gross margin?", ticker="AAPL")
        r2 = RAGResponse(
            query_id="q-2",
            generated_answer="Gross margin dropped to 10.0%.",
            retrieved_chunks=[sample_chunk_apple],
        )

        summary = evaluator.evaluate_batch_summary([(q1, r1), (q2, r2)], batch_id="batch-test-01")
        assert summary.batch_id == "batch-test-01"
        assert summary.total_queries == 2
        assert summary.passed_queries >= 1
        assert 0.0 <= summary.pass_rate <= 1.0
        assert summary.mean_latency_ms >= 0.0
        assert len(summary.results) == 2

    def test_evaluator_empty_batch_summary(self) -> None:
        evaluator = RAGEvaluator()
        summary = evaluator.evaluate_batch_summary([])
        assert summary.total_queries == 0
        assert summary.pass_rate == 0.0
        assert summary.results == []
