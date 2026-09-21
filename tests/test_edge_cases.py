"""Edge case and boundary condition test suite.

Covers:
- Zero, NaN, empty inputs, constant return series, division-by-zero guards
- Semantic cache zero vectors, expiration branches, overwrite behavior
- Hallucination detector edge cases (zero values, empty context, short claims)
- Claim verification boundary conditions (ambiguous, ungrounded, refuted)
- Telemetry exception handling in listeners, empty history rollings, SLA latency alerts
"""

import time

import numpy as np

from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.core.graph import EvaluationGraph
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import (
    MonteCarloStressTester,
)
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.models.schemas import (
    ClaimStatus,
    DocumentChunk,
    EvaluationMode,
    HallucinationFinding,
    HallucinationType,
    RAGQuery,
    RAGResponse,
    Severity,
    SystemHealthStatus,
)


class TestCoreEdgeCases:
    """Edge cases in SemanticCache, EvaluationGraph, and Evaluator."""

    def test_cosine_similarity_zero_norm(self):
        cache = SemanticCache()
        zero_vec = np.zeros(64, dtype=np.float32)
        normal_vec = np.ones(64, dtype=np.float32)

        # Norm of zero vector is 0, so similarity must be 0.0
        assert cache._cosine_similarity(zero_vec, normal_vec) == 0.0
        assert cache._cosine_similarity(normal_vec, zero_vec) == 0.0
        assert cache._cosine_similarity(zero_vec, zero_vec) == 0.0

    def test_cache_properties_and_clear(self):
        evaluator = RAGEvaluator()
        evaluator.clear_cache()
        cache = evaluator.cache

        assert cache.hit_count == 0
        assert cache.miss_count == 0
        assert cache.hit_rate == 0.0

        cache.put("Test query", "Answer 1")
        assert cache.size == 1

        # Overwrite existing key
        cache.put("Test query", "Answer 2")
        assert cache.size == 1
        cached = cache.get("Test query")
        assert cached is not None
        val, score = cached
        assert val == "Answer 2"
        assert score == 1.0
        assert cache.hit_count == 1

        cache.clear()
        assert cache.size == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_cache_expiration_handling_in_get_and_find_similar(self):
        cache = SemanticCache(ttl_seconds=0.01)
        cache.put("Immediate expire", "val")
        time.sleep(0.02)

        # In get() and find_similar(), expired entries are filtered
        assert cache.get("Immediate expire") is None
        assert len(cache.find_similar("Immediate expire")) == 0

        # Also test the loop branch where entry.is_expired is true during iteration
        cache.put("Manual expire", "val2")
        # Bypass purge to test loop condition
        cache._purge_expired = lambda: None  # type: ignore[assignment]
        cache._cache[list(cache._cache.keys())[0]].expires_at = time.time() - 10.0
        assert cache.get("Different query that misses exact") is None
        assert len(cache.find_similar("Different query")) == 0

    def test_graph_raw_expected_numerical_targets_missed(self):
        graph = EvaluationGraph()
        query = RAGQuery(
            query_id="q-raw-tgt",
            query_text="What was the EBITDA margin?",
            ticker="XYZ",
            expected_numerical_targets=[25.5, 40.0],
        )
        response = RAGResponse(
            query_id="q-raw-tgt",
            generated_answer="The reported EBITDA margin was 25.5%, while operating margin was 10%.",
            retrieved_chunks=[
                DocumentChunk(
                    chunk_id="chk-1",
                    text="EBITDA margin reached 25.5% and revenue reached $40.0M.",
                    ticker="XYZ",
                )
            ],
        )

        metrics = graph.run(query, response)
        # 40.0 was expected in answer but answer only has 25.5 and 10
        assert "40.0" in metrics.metadata.get("target_checks", {})
        assert metrics.metadata["target_checks"]["40.0"]["matched"] is False
        assert any(
            f.finding_type == HallucinationType.NUMERICAL_DISTORTION
            and "40.0" in f.claim_text
            for f in metrics.findings
        )

    def test_graph_stopwords_only_query_and_empty_retrieved_chunks(self):
        graph = EvaluationGraph()
        # Query with only stopwords
        query = RAGQuery(
            query_id="q-stopwords",
            query_text="What was the in and for?",
            ticker="XYZ",
        )
        # Empty retrieved chunks
        response = RAGResponse(
            query_id="q-stopwords",
            generated_answer="The outcome was positive and satisfactory.",
            retrieved_chunks=[],
        )

        metrics = graph.run(query, response, mode=EvaluationMode.FAST)
        assert metrics.context_precision_score == 0.5
        assert metrics.context_recall_score == 0.5
        assert metrics.answer_relevance_score >= 0.0

    def test_graph_all_evaluation_modes_logic(self):
        graph = EvaluationGraph()
        query = RAGQuery(query_id="q-mode", query_text="What was Apple net profit?", ticker="AAPL")
        chunk = DocumentChunk(chunk_id="c1", text="Apple net profit was $20B.", ticker="AAPL")
        answer = "Apple net profit was $20B."
        response = RAGResponse(query_id="q-mode", generated_answer=answer, retrieved_chunks=[chunk])

        # Test FAST mode pass
        m_fast = graph.run(query, response, mode=EvaluationMode.FAST)
        assert m_fast.overall_pass is True

        # Test AUDIT mode pass
        m_audit = graph.run(query, response, mode=EvaluationMode.AUDIT)
        assert m_audit.overall_pass is True

        # Test THOROUGH mode pass
        m_thorough = graph.run(query, response, mode=EvaluationMode.THOROUGH)
        assert m_thorough.overall_pass is True


class TestHallucinationDetectorEdgeCases:
    """Edge cases in HallucinationDetector logic."""

    def test_zero_matching_in_numerical_distortion(self):
        detector = HallucinationDetector()
        context = [
            DocumentChunk(
                chunk_id="chk-zero",
                text="The net debt for the period was 0.0 dollars or 0 million.",
                ticker="AAPL",
            )
        ]
        answer = "Net debt remained at 0 dollars."
        findings = detector.detect_numerical_distortions(answer, context)
        assert len(findings) == 0

    def test_empty_generated_answer_and_no_numbers(self):
        detector = HallucinationDetector()
        context = [DocumentChunk(chunk_id="chk-1", text="Revenue was $100M.", ticker="AAPL")]
        findings = detector.detect_numerical_distortions("No numbers here at all.", context)
        assert len(findings) == 0

    def test_short_claims_in_detect_ungrounded_claims(self):
        detector = HallucinationDetector()
        context = [DocumentChunk(chunk_id="chk-1", text="Growth was strong.", ticker="AAPL")]
        # Sentence with no tokens of len >= 3
        answer = "It is at 10. We go in."
        findings = detector.detect_ungrounded_claims(answer, context)
        assert len(findings) == 0

    def test_empty_context_in_entity_swaps_and_temporal(self):
        detector = HallucinationDetector()
        findings_entity = detector.detect_entity_swaps("AAPL reported gains.", [])
        assert findings_entity == []

        findings_temporal = detector.detect_temporal_mismatches("In 2024 revenue grew.", [])
        assert findings_temporal == []

    def test_context_without_years_temporal(self):
        detector = HallucinationDetector()
        context = [DocumentChunk(chunk_id="c1", text="Revenue grew significantly over last quarter.", ticker="AAPL")]
        findings = detector.detect_temporal_mismatches("In 2024 it grew.", context)
        assert len(findings) == 0

    def test_verify_claims_empty_input(self):
        detector = HallucinationDetector()
        assert detector.verify_claims([], []) == []

    def test_verify_claims_ambiguous_and_ungrounded_status(self):
        detector = HallucinationDetector()
        context = [
            DocumentChunk(
                chunk_id="chk-fin",
                text="Operating revenue reached eighty billion dollars with strong performance.",
                ticker="AAPL",
            )
        ]
        # Partial overlap (between 0.20 and 0.35) -> AMBIGUOUS
        # "revenue eighty billion" in 10-word claim
        claim_ambiguous = "Overall operating performance was commendable for stakeholders and consumers."
        results = detector.verify_claims([claim_ambiguous], context)
        assert len(results) == 1
        assert results[0].status in (ClaimStatus.AMBIGUOUS, ClaimStatus.UNGROUNDED)

        # Claim with no evidence chunks -> UNGROUNDED
        claim_ungrounded = "Unrelated extraterrestrial observations continued without interruption."
        res_un = detector.verify_claims([claim_ungrounded], context)
        assert res_un[0].status == ClaimStatus.UNGROUNDED

        # Claim with low keyword overlap (between 0.10 and 0.20) -> UNGROUNDED with evidence chunks
        # 1 match out of 7 tokens = ~14.3%
        claim_low_overlap = "Performance requires deliberate perseverance through economic uncertainty and hardship."
        res_low = detector.verify_claims([claim_low_overlap], context)
        assert len(res_low) == 1
        assert res_low[0].status == ClaimStatus.UNGROUNDED
        assert res_low[0].explanation is not None
        assert "insufficient" in res_low[0].explanation

    def test_numerical_extraction_regex_value_error(self):
        from unittest.mock import MagicMock
        detector = HallucinationDetector()
        mock_match = MagicMock()
        mock_match.group.side_effect = lambda idx: "invalid_num" if idx == 1 else ""
        mock_pattern = MagicMock()
        mock_pattern.finditer.return_value = [mock_match]
        detector._num_pattern = mock_pattern

        vals = detector._extract_numerical_values("sample text")
        assert vals == []


class TestFinancialRiskEngineEdgeCases:
    """Edge cases and zero-variance scenarios for FinancialRiskEngine."""

    def test_sortino_ratio_zero_downside_std(self):
        engine = FinancialRiskEngine()
        # All positive returns -> downside is empty or downside_std <= 1e-12
        positive_ret = np.array([0.01, 0.02, 0.015, 0.03])
        sortino = engine.calculate_sortino_ratio(positive_ret, target_return=-0.05)
        assert sortino > 0

        # Micro downside returns producing std <= 1e-12
        micro_downside = np.array([-1e-15, -1e-15])
        assert engine.calculate_sortino_ratio(micro_downside, target_return=0.0) == 0.0

    def test_calmar_ratio_short_series(self):
        engine = FinancialRiskEngine()
        assert engine.calculate_calmar_ratio(np.array([0.01])) == 0.0

    def test_parametric_var_edge_cases(self):
        engine = FinancialRiskEngine()
        # Fewer than 3 returns
        assert engine.calculate_parametric_var(np.array([0.01, 0.02])) == 0.0

        # Constant returns (sigma <= 1e-12)
        constant_ret = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        assert engine.calculate_parametric_var(constant_ret) == 0.0

    def test_beta_alpha_edge_cases(self):
        engine = FinancialRiskEngine()
        ret = np.array([0.01, 0.02, 0.03])
        # Constant benchmark returns (bench_var <= 1e-12)
        flat_bench = np.array([0.005, 0.005, 0.005])
        beta, alpha = engine.calculate_beta_alpha(ret, flat_bench)
        assert beta == 1.0
        assert alpha == 0.0

        # Fewer than 3 returns
        b_short, a_short = engine.calculate_beta_alpha(np.array([0.01]), np.array([0.01]))
        assert b_short == 1.0
        assert a_short == 0.0

    def test_information_ratio_edge_cases(self):
        engine = FinancialRiskEngine()
        # Fewer than 2 returns
        assert engine.calculate_information_ratio(np.array([0.01]), np.array([0.01])) == 0.0

        # Tracking error == 0 with positive excess return
        ret = np.array([0.02, 0.03, 0.04])
        bench = np.array([0.01, 0.02, 0.03])  # constant +0.01 active return
        ir = engine.calculate_information_ratio(ret, bench)
        assert ir > 0.0

        # Tracking error == 0 with negative excess return
        ret_neg = np.array([0.01, 0.02, 0.03])
        bench_pos = np.array([0.02, 0.03, 0.04])  # constant -0.01 active return
        ir_neg = engine.calculate_information_ratio(ret_neg, bench_pos)
        assert ir_neg == 0.0

    def test_omega_ratio_edge_cases(self):
        engine = FinancialRiskEngine()
        # Empty returns
        assert engine.calculate_omega_ratio(np.array([])) == 1.0

        # All positive returns (zero losses)
        all_pos = np.array([0.01, 0.02, 0.03])
        assert engine.calculate_omega_ratio(all_pos, threshold=0.0) == 100.0

        # All zero returns (zero gains, zero losses)
        all_zeros = np.array([0.0, 0.0, 0.0])
        assert engine.calculate_omega_ratio(all_zeros, threshold=0.0) == 1.0

    def test_tail_risk_short_series(self):
        engine = FinancialRiskEngine()
        assert engine.calculate_tail_risk(np.array([0.01, 0.02])) == (0.0, 0.0)

    def test_compute_all_metrics_empty_or_single_return(self):
        engine = FinancialRiskEngine()
        # Falls back to default array
        metrics = engine.compute_all_metrics(np.array([]))
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown is not None


class TestStressTestingAndTelemetryEdgeCases:
    """Edge cases in MonteCarloStressTester and RiskTelemetryCollector."""

    def test_stress_test_short_return_series_fallback(self):
        tester = MonteCarloStressTester(random_seed=42)
        res = tester.run_stress_test(returns=np.array([0.01]))
        assert res.n_simulations == 1000
        assert res.worst_case_drawdown >= 0.0

    def test_hallucination_impact_empty_estimates_fallback(self):
        tester = MonteCarloStressTester(random_seed=42)
        breach_rate = tester.stress_test_hallucination_impact(base_financial_estimates=[])
        assert 0.0 <= breach_rate <= 1.0

    def test_telemetry_empty_history_rolling_metrics(self):
        collector = RiskTelemetryCollector()
        rolling = collector.get_rolling_metrics()
        assert rolling["rolling_faithfulness"] == 1.0
        assert rolling["rolling_relevance"] == 1.0
        assert rolling["rolling_hallucination_rate"] == 0.0
        assert rolling["rolling_latency_ms"] == 0.0

    def test_telemetry_alert_listener_exception_resilience(self):
        collector = RiskTelemetryCollector()

        def buggy_listener(alert):
            raise RuntimeError("Listener crashed")

        collector.add_alert_listener(buggy_listener)

        # Trigger an alert by recording an evaluation with high hallucination rate
        from production_rag_evaluation.models.schemas import EvaluationMetrics
        finding = HallucinationFinding(
            finding_type=HallucinationType.CONTRADICTION,
            severity=Severity.CRITICAL,
            claim_text="Contradiction",
            reference_text="Context",
            confidence=0.9,
            explanation="Explanation",
        )
        m = EvaluationMetrics(
            query_id="q-err",
            faithfulness_score=0.2,
            answer_relevance_score=0.2,
            context_recall_score=0.5,
            context_precision_score=0.5,
            citation_grounding_score=0.5,
            overall_pass=False,
            findings=[finding],
            hallucination_rate=0.5,
            evaluation_latency_ms=3000.0,
        )
        collector.record_evaluation(m)

        # Report generation must not crash even though buggy_listener throws
        report = collector.generate_report(
            max_acceptable_hallucination_rate=0.1,
            max_acceptable_latency_ms=1000.0,
        )
        assert report.health_status == SystemHealthStatus.UNHEALTHY
        assert any("HIGH_LATENCY" in a for a in report.alerts_triggered)

    def test_telemetry_stress_test_failure_triggers_critical_alert(self):
        collector = RiskTelemetryCollector()
        # Returns with severe losses
        crash_returns = np.array([-0.08, -0.09, -0.07, -0.06, -0.05, -0.04])
        report = collector.generate_report(returns=crash_returns)
        if report.monte_carlo_result and not report.monte_carlo_result.stress_pass:
            assert any("STRESS_TEST_FAILED" in a for a in report.alerts_triggered)

    def test_telemetry_drift_insufficient_samples(self):
        collector = RiskTelemetryCollector()
        res = collector.detect_drift(baseline_size=20, test_size=10)
        assert res["drift_detected"] is False
        assert "Insufficient data points" in res["reason"]
