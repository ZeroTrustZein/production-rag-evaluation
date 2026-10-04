"""Comprehensive unit and integration test suite for financial subsystems.

Covers FinancialRiskEngine, MonteCarloStressTester, and RiskTelemetryCollector.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.models.interfaces import (
    FinancialRiskEngineProtocol,
    MonteCarloStressTesterProtocol,
    RiskTelemetryCollectorProtocol,
)
from production_rag_evaluation.models.schemas import (
    AlertSeverity,
    EvaluationMetrics,
    StressScenarioType,
    StressTestConfig,
    SystemHealthStatus,
    TelemetryAlert,
)


class TestFinancialRiskEngineSubsystem:
    """Rigorous evaluation of quantitative risk metrics, edge cases, and portfolio statistics."""

    def test_protocol_conformity(self) -> None:
        engine = FinancialRiskEngine()
        assert isinstance(engine, FinancialRiskEngineProtocol)

    def test_sharpe_ratio_standard_and_edge_cases(self) -> None:
        engine = FinancialRiskEngine(risk_free_rate=0.04)

        # Empty or single observation
        assert engine.calculate_sharpe_ratio(np.array([])) == 0.0
        assert engine.calculate_sharpe_ratio(np.array([0.01])) == 0.0

        # Zero variance / constant returns
        constant_returns = np.array([0.002, 0.002, 0.002, 0.002])
        assert engine.calculate_sharpe_ratio(constant_returns) == 0.0

        # Typical positive returns
        pos_returns = np.array([0.005, 0.003, -0.001, 0.004, 0.002, 0.006])
        sharpe = engine.calculate_sharpe_ratio(pos_returns)
        assert isinstance(sharpe, float)
        assert sharpe > 0.0

        # Severe negative returns
        neg_returns = np.array([-0.02, -0.015, -0.03, -0.01])
        neg_sharpe = engine.calculate_sharpe_ratio(neg_returns)
        assert neg_sharpe < 0.0

    def test_sortino_ratio_downside_volatility(self) -> None:
        engine = FinancialRiskEngine(risk_free_rate=0.02)

        # No negative returns (no downside deviation)
        all_pos = np.array([0.01, 0.02, 0.015, 0.012])
        sortino_high = engine.calculate_sortino_ratio(all_pos, target_return=0.0)
        assert sortino_high > 0.0

        # Mixed returns
        mixed = np.array([0.02, -0.015, 0.01, -0.005, 0.03, -0.02])
        sortino = engine.calculate_sortino_ratio(mixed, target_return=0.0)
        assert isinstance(sortino, float)

        # Edge case: single return
        assert engine.calculate_sortino_ratio(np.array([0.05])) == 0.0

    def test_max_drawdown_trajectories(self) -> None:
        engine = FinancialRiskEngine()

        # Monotonically increasing portfolio: drawdown must be 0.0
        increasing = np.array([0.01, 0.02, 0.015, 0.03])
        assert engine.calculate_max_drawdown(increasing) == 0.0

        # Monotonically decreasing portfolio
        decreasing = np.array([-0.10, -0.10, -0.10])
        # After 3 drops of 10%: 0.9 * 0.9 * 0.9 = 0.729 -> drawdown = 1 - 0.729 = 0.271
        dd = engine.calculate_max_drawdown(decreasing)
        assert 0.270 <= dd <= 0.272

        # Empty array
        assert engine.calculate_max_drawdown(np.array([])) == 0.0

    def test_calmar_ratio_calculation(self) -> None:
        engine = FinancialRiskEngine()

        # Zero drawdown cap
        increasing = np.array([0.005, 0.004, 0.006])
        calmar_capped = engine.calculate_calmar_ratio(increasing)
        assert calmar_capped > 0.0

        # Normal volatility with drawdown
        mixed = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        calmar = engine.calculate_calmar_ratio(mixed)
        assert isinstance(calmar, float)

    def test_var_and_cvar_confidence_levels(self) -> None:
        engine = FinancialRiskEngine()
        np.random.seed(123)
        returns = np.random.normal(0.0005, 0.015, 500)

        var_95, cvar_95 = engine.calculate_var_cvar(returns, confidence_level=0.95)
        var_99, cvar_99 = engine.calculate_var_cvar(returns, confidence_level=0.99)

        assert var_95 > 0.0
        assert cvar_95 >= var_95
        assert var_99 >= var_95
        assert cvar_99 >= var_99

        # Empty returns edge case
        assert engine.calculate_var_cvar(np.array([])) == (0.0, 0.0)

    def test_parametric_var_with_cornish_fisher(self) -> None:
        engine = FinancialRiskEngine()
        np.random.seed(42)
        normal_returns = np.random.normal(0.0002, 0.01, 200)

        # Normal parametric VaR
        p_var_norm = engine.calculate_parametric_var(
            normal_returns, confidence_level=0.95, cornish_fisher=False
        )
        assert p_var_norm > 0.0

        # Skewed returns with Cornish-Fisher expansion
        skewed_returns = np.random.exponential(scale=0.01, size=200) - 0.01
        p_var_cf = engine.calculate_parametric_var(
            skewed_returns, confidence_level=0.95, cornish_fisher=True
        )
        assert p_var_cf > 0.0

    def test_beta_and_alpha_against_benchmark(self) -> None:
        engine = FinancialRiskEngine(risk_free_rate=0.03)
        np.random.seed(99)
        bench = np.random.normal(0.0004, 0.01, 100)

        # Asset perfectly synchronized with benchmark (Beta ~ 1.0, Alpha ~ 0.0)
        asset_sync = bench * 1.0 + np.random.normal(0.0, 0.0001, 100)
        beta, alpha = engine.calculate_beta_alpha(asset_sync, bench)
        assert 0.95 <= beta <= 1.05
        assert abs(alpha) < 0.05

        # Leveraged asset (Beta ~ 2.0)
        asset_2x = bench * 2.0
        beta_2x, _ = engine.calculate_beta_alpha(asset_2x, bench)
        assert 1.95 <= beta_2x <= 2.05

        # Short / insufficient series fallback
        assert engine.calculate_beta_alpha(np.array([0.01]), bench) == (1.0, 0.0)

    def test_information_ratio_and_omega_ratio(self) -> None:
        engine = FinancialRiskEngine()
        np.random.seed(42)
        bench = np.random.normal(0.0002, 0.01, 100)
        asset_alpha = bench + 0.0005  # Consistent excess return

        ir = engine.calculate_information_ratio(asset_alpha, bench)
        assert ir > 0.0

        # Omega ratio
        returns = np.array([0.02, 0.01, -0.005, 0.015, -0.01])
        omega = engine.calculate_omega_ratio(returns, threshold=0.0)
        assert omega > 1.0  # More gains than losses

        # Empty returns fallback
        assert engine.calculate_omega_ratio(np.array([])) == 1.0

    def test_tail_risk_skew_and_kurtosis(self) -> None:
        engine = FinancialRiskEngine()
        np.random.seed(42)
        # Symmetrical normal data
        norm_data = np.random.normal(0.0, 0.01, 1000)
        skew, kurt = engine.calculate_tail_risk(norm_data)
        assert abs(skew) < 0.2
        assert abs(kurt) < 0.5

        # Empty / tiny array
        assert engine.calculate_tail_risk(np.array([0.01])) == (0.0, 0.0)

    def test_compute_all_metrics_full_suite(self) -> None:
        engine = FinancialRiskEngine(risk_free_rate=0.045)
        np.random.seed(42)
        asset = np.random.normal(0.0005, 0.012, 120)
        bench = np.random.normal(0.0003, 0.010, 120)

        metrics = engine.compute_all_metrics(asset, benchmark_returns=bench)

        assert metrics.sharpe_ratio != 0.0
        assert metrics.sortino_ratio != 0.0
        assert 0.0 <= metrics.max_drawdown <= 1.0
        assert metrics.var_95 > 0.0
        assert metrics.cvar_95 >= metrics.var_95
        assert metrics.var_99 is not None and metrics.var_99 >= metrics.var_95
        assert metrics.cvar_99 is not None and metrics.cvar_99 >= metrics.cvar_95
        assert metrics.beta is not None
        assert metrics.alpha is not None
        assert metrics.information_ratio is not None


class TestMonteCarloStressTesterSubsystem:
    """Rigorous evaluation of stochastic trajectory simulation, fat tails, and scenario shocks."""

    def test_protocol_conformity(self) -> None:
        tester = MonteCarloStressTester()
        assert isinstance(tester, MonteCarloStressTesterProtocol)

    def test_gbm_paths_shape_and_determinism(self) -> None:
        tester1 = MonteCarloStressTester(random_seed=123)
        paths1 = tester1.simulate_gbm_paths(
            initial_value=100.0,
            mu=0.08,
            sigma=0.20,
            horizon_days=30,
            n_simulations=50,
        )

        assert paths1.shape == (50, 31)
        # Initial step is always 100.0
        np.testing.assert_allclose(paths1[:, 0], 100.0)

        # Reproducibility with identical seed
        tester2 = MonteCarloStressTester(random_seed=123)
        paths2 = tester2.simulate_gbm_paths(
            initial_value=100.0,
            mu=0.08,
            sigma=0.20,
            horizon_days=30,
            n_simulations=50,
        )
        np.testing.assert_allclose(paths1, paths2)

    def test_fat_tailed_student_t_paths(self) -> None:
        tester = MonteCarloStressTester(random_seed=42)
        paths = tester.simulate_fat_tailed_paths(
            initial_value=50.0,
            mu=0.05,
            sigma=0.25,
            horizon_days=20,
            n_simulations=100,
            degrees_of_freedom=3,
            volatility_shock_multiplier=2.0,
        )
        assert paths.shape == (100, 21)
        assert paths[0, 0] == 50.0
        assert np.all(paths > 0.0)

    def test_stress_test_with_config(self) -> None:
        tester = MonteCarloStressTester(random_seed=42)
        returns = np.array([0.001, -0.002, 0.003, -0.001, 0.002])

        config = StressTestConfig(
            n_simulations=250,
            horizon_days=15,
            volatility_shock=2.0,
            max_allowable_drawdown=0.20,
            initial_value=100.0,
            random_seed=42,
        )
        result = tester.run_stress_test(returns, config=config)

        assert result.n_simulations == 250
        assert result.horizon_steps == 15
        assert result.p5_terminal <= result.p50_terminal <= result.p95_terminal
        assert 0.0 <= result.worst_case_drawdown <= 1.0
        assert 0.0 <= result.breach_probability <= 1.0
        assert result.iterations_data is not None
        assert len(result.iterations_data) > 0

    @pytest.mark.parametrize(
        "scenario",
        [
            StressScenarioType.BASELINE,
            StressScenarioType.VOLATILITY_SPIKE,
            StressScenarioType.LIQUIDITY_CRUNCH,
            StressScenarioType.RATE_SHOCK,
            StressScenarioType.BLACK_SWAN,
            StressScenarioType.HALLUCINATION_CASCADE,
        ],
    )
    def test_all_preset_stress_scenarios(self, scenario: StressScenarioType) -> None:
        tester = MonteCarloStressTester(random_seed=42)
        returns = np.array([0.002, -0.001, 0.004, -0.003, 0.001, 0.002])
        res = tester.run_scenario(returns, scenario_type=scenario)

        assert res.scenario_type == scenario
        assert res.p5_terminal > 0.0
        assert isinstance(res.stress_pass, bool)
        assert isinstance(res.worst_case_drawdown, float)

    def test_hallucination_impact_simulation(self) -> None:
        tester = MonteCarloStressTester(random_seed=42)
        base_targets = [100.0, 105.0, 95.0, 110.0]

        # Zero hallucination rate should result in zero breaches
        clean_rate = tester.stress_test_hallucination_impact(
            base_targets,
            hallucination_probability=0.0,
            n_simulations=100,
        )
        assert clean_rate == 0.0

        # Severe hallucination rate with large distortion
        high_rate = tester.stress_test_hallucination_impact(
            base_targets,
            hallucination_probability=0.80,
            hallucination_distortion_pct=0.40,
            n_simulations=150,
        )
        assert high_rate > 0.0


class TestRiskTelemetryCollectorSubsystem:
    """Rigorous evaluation of operational monitoring, alert thresholds, health state machine, and drift detection."""

    def test_protocol_conformity(self) -> None:
        collector = RiskTelemetryCollector()
        assert isinstance(collector, RiskTelemetryCollectorProtocol)

    def test_empty_collector_defaults(self) -> None:
        collector = RiskTelemetryCollector()
        rep = collector.generate_report()

        assert rep.total_evaluations == 0
        assert rep.avg_faithfulness == 1.0
        assert rep.avg_hallucination_rate == 0.0
        assert rep.health_status == SystemHealthStatus.HEALTHY
        assert len(rep.alerts_triggered) == 0

    def test_record_and_query_evaluations(self) -> None:
        collector = RiskTelemetryCollector()

        # Record 3 metrics: 2 pass, 1 fail
        m_pass1 = EvaluationMetrics(
            faithfulness_score=0.95,
            answer_relevance_score=0.90,
            context_recall_score=0.85,
            context_precision_score=0.90,
            citation_grounding_score=0.95,
            hallucination_rate=0.0,
            overall_pass=True,
        )
        m_pass2 = EvaluationMetrics(
            faithfulness_score=0.88,
            answer_relevance_score=0.85,
            context_recall_score=0.80,
            context_precision_score=0.85,
            citation_grounding_score=0.90,
            hallucination_rate=0.05,
            overall_pass=True,
        )
        m_fail = EvaluationMetrics(
            faithfulness_score=0.40,
            answer_relevance_score=0.45,
            context_recall_score=0.50,
            context_precision_score=0.40,
            citation_grounding_score=0.30,
            hallucination_rate=0.60,
            overall_pass=False,
        )

        collector.record_evaluation(m_pass1)
        collector.record_evaluation(m_pass2)
        collector.record_evaluation(m_fail)

        assert len(collector.get_evaluations()) == 3
        assert len(collector.get_evaluations(passed_only=True)) == 2
        assert len(collector.get_evaluations(passed_only=False)) == 1
        assert len(collector.get_evaluations(limit=2)) == 2

        # Clear history
        collector.clear_history()
        assert len(collector.get_evaluations()) == 0

    def test_rolling_metrics_calculations(self) -> None:
        collector = RiskTelemetryCollector()
        for i in range(10):
            collector.record_evaluation(
                EvaluationMetrics(
                    faithfulness_score=0.80 + i * 0.01,
                    answer_relevance_score=0.85,
                    context_recall_score=0.80,
                    context_precision_score=0.80,
                    citation_grounding_score=0.85,
                    hallucination_rate=0.02,
                    overall_pass=True,
                    evaluation_latency_ms=10.0 + i,
                )
            )

        rolling = collector.get_rolling_metrics(window_size=5)
        assert 0.85 <= rolling["rolling_faithfulness"] <= 0.90
        assert rolling["rolling_latency_ms"] > 10.0

    def test_alert_listeners_callback(self) -> None:
        collector = RiskTelemetryCollector()
        captured_alerts: list[TelemetryAlert] = []

        collector.add_alert_listener(lambda alert: captured_alerts.append(alert))

        # Record high hallucination rate to trigger alert
        collector.record_evaluation(
            EvaluationMetrics(
                faithfulness_score=0.40,
                answer_relevance_score=0.50,
                context_recall_score=0.50,
                context_precision_score=0.50,
                citation_grounding_score=0.40,
                hallucination_rate=0.35,  # Exceeds 0.10 threshold & 0.25 critical threshold
                overall_pass=False,
            )
        )

        rep = collector.generate_report(max_acceptable_hallucination_rate=0.10)
        assert len(captured_alerts) > 0
        assert any(a.severity == AlertSeverity.CRITICAL for a in captured_alerts)
        assert rep.health_status == SystemHealthStatus.UNHEALTHY

    def test_health_status_degraded_on_warning(self) -> None:
        collector = RiskTelemetryCollector()
        collector.record_evaluation(
            EvaluationMetrics(
                faithfulness_score=0.82,
                answer_relevance_score=0.80,
                context_recall_score=0.80,
                context_precision_score=0.80,
                citation_grounding_score=0.80,
                hallucination_rate=0.15,  # Above 0.10 (Warning), below 0.25 (Critical)
                overall_pass=True,
            )
        )

        rep = collector.generate_report(max_acceptable_hallucination_rate=0.10)
        assert rep.health_status == SystemHealthStatus.DEGRADED

    def test_metric_drift_detection(self) -> None:
        collector = RiskTelemetryCollector()

        # Insufficient data points
        drift_res_empty = collector.detect_drift(baseline_size=20, test_size=10)
        assert drift_res_empty["drift_detected"] is False

        # Add 20 high faithfulness records (baseline ~0.95 with variance)
        for i in range(20):
            collector.record_evaluation(
                EvaluationMetrics(
                    faithfulness_score=0.92 + (i % 5) * 0.015,
                    answer_relevance_score=0.90,
                    context_recall_score=0.90,
                    context_precision_score=0.90,
                    citation_grounding_score=0.90,
                    hallucination_rate=0.0,
                    overall_pass=True,
                )
            )

        # Add 10 degraded faithfulness records (recent ~0.60 with variance)
        for i in range(10):
            collector.record_evaluation(
                EvaluationMetrics(
                    faithfulness_score=0.55 + (i % 4) * 0.02,
                    answer_relevance_score=0.70,
                    context_recall_score=0.70,
                    context_precision_score=0.70,
                    citation_grounding_score=0.60,
                    hallucination_rate=0.30,
                    overall_pass=False,
                )
            )

        drift_res = collector.detect_drift(
            metric="faithfulness_score", baseline_size=20, test_size=10
        )
        assert drift_res["drift_detected"] is True
        assert drift_res["p_value"] < 0.01
        assert drift_res["recent_mean"] < drift_res["baseline_mean"]

    def test_export_summary_dictionary(self) -> None:
        collector = RiskTelemetryCollector()
        collector.record_evaluation(
            EvaluationMetrics(
                faithfulness_score=0.90,
                answer_relevance_score=0.85,
                context_recall_score=0.80,
                context_precision_score=0.85,
                citation_grounding_score=0.90,
                hallucination_rate=0.05,
                overall_pass=True,
            )
        )

        summary = collector.export_summary()
        assert summary["total_evaluations"] == 1
        assert summary["avg_faithfulness"] == 0.90
        assert "health_status" in summary
        assert "rolling_metrics" in summary


class TestFeatureStoreSubsystems:
    """Rigorous evaluation of FeatureRegistry, EvaluationFeatureHandler, Compaction, and Drift Detection."""

    @pytest.fixture
    def populated_feature_store(self, tmp_path):
        from production_rag_evaluation.features.extractors import QueryFeatureExtractor
        from production_rag_evaluation.features.store import ParquetFeatureStore
        from production_rag_evaluation.models.schemas import RAGQuery

        store = ParquetFeatureStore(base_path=tmp_path / "fs_subsystem")
        extractor = QueryFeatureExtractor()

        for i in range(15):
            query = RAGQuery(
                query_id=f"q_{i}",
                query_text=f"What was Apple's net margin in fiscal year 2024? Question {i}",
                ticker="AAPL",
            )
            rec = extractor.extract(query)
            store.write_records([rec])

        return store

    def test_feature_registry_schema_registration_and_validation(self) -> None:
        from production_rag_evaluation.features.models import (
            FeatureDefinition,
            FeatureGroup,
            FeatureRecord,
            FeatureSchema,
            FeatureValueType,
        )
        from production_rag_evaluation.features.registry import (
            FeatureRegistry,
            get_default_registry,
        )
        from production_rag_evaluation.models.interfaces import FeatureRegistryProtocol

        registry = get_default_registry()
        assert isinstance(registry, FeatureRegistryProtocol)

        # Check default schemas loaded
        schemas = registry.list_schemas()
        assert FeatureGroup.QUERY in schemas
        assert FeatureGroup.EVALUATION in schemas
        assert FeatureGroup.FINANCIAL in schemas

        # Query schema feature check
        fdef = registry.get_feature_definition(FeatureGroup.QUERY, "query_length")
        assert fdef is not None
        assert fdef.value_type == FeatureValueType.INT

        # Valid record validation
        valid_rec = FeatureRecord(
            entity_id="test_q1",
            feature_group=FeatureGroup.QUERY,
            features={"query_length": 55, "has_question_mark": True},
        )
        val_res = registry.validate_record(valid_rec, strict=False)
        assert val_res.valid is True
        assert len(val_res.errors) == 0

        # Type mismatch validation
        invalid_type_rec = FeatureRecord(
            entity_id="test_q2",
            feature_group=FeatureGroup.QUERY,
            features={"query_length": "not_an_int"},
        )
        val_res_inv = registry.validate_record(invalid_type_rec, strict=True)
        assert val_res_inv.valid is False
        assert any("expected INT" in err for err in val_res_inv.errors)

        # Custom schema registration
        custom_schema = FeatureSchema(
            name="custom_user_features",
            feature_group=FeatureGroup.QUERY,
            version="2.0.0",
            features=[
                FeatureDefinition(
                    name="custom_metric",
                    feature_group=FeatureGroup.QUERY,
                    value_type=FeatureValueType.FLOAT,
                    nullable=False,
                )
            ],
        )
        custom_reg = FeatureRegistry(load_defaults=False)
        custom_reg.register_schema(custom_schema)
        assert custom_reg.get_schema(FeatureGroup.QUERY) is not None

        # Missing required field in strict mode
        rec_missing = FeatureRecord(
            entity_id="test_q3",
            feature_group=FeatureGroup.QUERY,
            features={},
        )
        strict_res = custom_reg.validate_record(rec_missing, strict=True)
        assert strict_res.valid is False
        assert any("Required non-nullable feature" in err for err in strict_res.errors)

    def test_evaluation_feature_handler_subsystem(self, tmp_path) -> None:
        from production_rag_evaluation.features.handlers import EvaluationFeatureHandler
        from production_rag_evaluation.features.pipeline import StreamingFeaturePipeline
        from production_rag_evaluation.features.store import ParquetFeatureStore
        from production_rag_evaluation.finance.metrics import FinancialRiskEngine
        from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
        from production_rag_evaluation.models.interfaces import EvaluationFeatureHandlerProtocol
        from production_rag_evaluation.models.schemas import (
            DocumentChunk,
            DocumentType,
            EvaluationMetrics,
            RAGQuery,
            RAGResponse,
        )

        store = ParquetFeatureStore(base_path=tmp_path / "handler_fs")
        pipeline = StreamingFeaturePipeline(store=store, buffer_size=10, auto_flush=False)
        collector = RiskTelemetryCollector()
        handler = EvaluationFeatureHandler(
            pipeline=pipeline, auto_flush=False, telemetry_collector=collector
        )

        assert isinstance(handler, EvaluationFeatureHandlerProtocol)

        query = RAGQuery(
            query_id="q_101",
            query_text="What was Microsoft cloud revenue growth?",
            ticker="MSFT",
        )
        chunk = DocumentChunk(
            chunk_id="c_101",
            text="Microsoft Cloud revenue was $38.9 billion, up 21% year-over-year.",
            doc_type=DocumentType.SEC_10Q,
            ticker="MSFT",
        )
        response = RAGResponse(
            query_id="q_101",
            generated_answer="Microsoft Cloud revenue grew 21% to $38.9 billion.",
            retrieved_chunks=[chunk],
            latency_ms=120.0,
        )
        metrics = EvaluationMetrics(
            faithfulness_score=1.0,
            answer_relevance_score=0.98,
            context_recall_score=0.95,
            context_precision_score=0.96,
            citation_grounding_score=0.92,
            hallucination_rate=0.0,
            overall_pass=True,
            evaluation_latency_ms=45.0,
        )

        # Ingest single evaluation
        recs = handler.handle_evaluation(
            query=query, response=response, metrics=metrics, flush_immediately=True
        )
        assert len(recs) == 4  # Query, Chunk, Response, Evaluation
        assert store.count_records() == 4
        assert len(collector.get_evaluations()) == 1

        # Ingest financial risk metrics
        risk_engine = FinancialRiskEngine()
        risk_m = risk_engine.compute_all_metrics(
            np.array([0.01, 0.02, -0.01, 0.015, 0.005, -0.002])
        )
        risk_rec = handler.handle_risk_metrics(risk_m, entity_id="risk_101", flush_immediately=True)
        assert risk_rec is not None
        assert store.count_records() == 5

        # Telemetry snapshot
        snap = handler.get_telemetry_snapshot()
        assert "collector_summary" in snap
        assert snap["collector_summary"]["total_evaluations"] == 1

    def test_feature_compaction_and_ttl_pruning(self, tmp_path) -> None:
        from datetime import datetime, timedelta, timezone

        from production_rag_evaluation.features.handlers import FeatureCompactionHandler
        from production_rag_evaluation.features.models import FeatureGroup, FeatureRecord
        from production_rag_evaluation.features.store import ParquetFeatureStore
        from production_rag_evaluation.models.interfaces import FeatureCompactionProtocol

        store = ParquetFeatureStore(base_path=tmp_path / "compact_fs")
        handler = FeatureCompactionHandler(store=store)
        assert isinstance(handler, FeatureCompactionProtocol)

        now = datetime.now(timezone.utc)
        # Write records with varying timestamps: 3 active, 2 expired (older than 2 days)
        recs = [
            FeatureRecord(
                entity_id=f"active_{i}",
                feature_group=FeatureGroup.QUERY,
                features={"query_length": 30 + i},
                timestamp=now - timedelta(hours=i),
            )
            for i in range(3)
        ]
        recs.extend(
            [
                FeatureRecord(
                    entity_id=f"expired_{j}",
                    feature_group=FeatureGroup.QUERY,
                    features={"query_length": 50 + j},
                    timestamp=now - timedelta(days=5 + j),
                )
                for j in range(2)
            ]
        )
        store.write_records(recs)
        assert store.count_records(FeatureGroup.QUERY) == 5

        # Run compaction with 2-day TTL (172800 seconds)
        comp_res = handler.compact_group(
            group=FeatureGroup.QUERY, ttl_seconds=172800.0, compression="snappy"
        )
        assert comp_res.records_compacted == 3
        assert comp_res.expired_records_pruned == 2
        assert store.count_records(FeatureGroup.QUERY) == 3

        # Prune via older_than datetime method
        pruned = handler.prune_expired_records(
            group=FeatureGroup.QUERY, older_than=now + timedelta(days=1)
        )
        # All 3 remaining are older than tomorrow
        assert pruned == 3
        assert store.count_records(FeatureGroup.QUERY) == 0

    def test_dataset_export_handler_formats(self, tmp_path) -> None:
        import json

        from production_rag_evaluation.features.handlers import FeatureExportHandler
        from production_rag_evaluation.features.models import (
            ExportFormat,
            FeatureGroup,
            FeatureRecord,
        )
        from production_rag_evaluation.features.store import ParquetFeatureStore

        store = ParquetFeatureStore(base_path=tmp_path / "export_fs")
        recs = [
            FeatureRecord(
                entity_id=f"entity_{i}",
                feature_group=FeatureGroup.QUERY,
                features={"query_length": 25 + i, "has_question_mark": True},
            )
            for i in range(5)
        ]
        store.write_records(recs)

        exporter = FeatureExportHandler(store=store)
        entity_ids = [f"entity_{i}" for i in range(5)]

        # 1. Export as Dict list
        data_dict = exporter.export_dataset(entity_ids=entity_ids, format=ExportFormat.DICT)
        assert isinstance(data_dict, list)
        assert len(data_dict) == 5
        assert data_dict[0]["entity_id"] == "entity_0"

        # 2. Export as JSONL
        jsonl_path = tmp_path / "export.jsonl"
        exporter.export_dataset(
            entity_ids=entity_ids, output_path=jsonl_path, format=ExportFormat.JSONL
        )
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        assert json.loads(lines[0])["entity_id"] == "entity_0"

        # 3. Export as CSV
        csv_path = tmp_path / "export.csv"
        exporter.export_dataset(
            entity_ids=entity_ids, output_path=csv_path, format=ExportFormat.CSV
        )
        assert csv_path.exists()
        csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(csv_lines) == 6  # header + 5 rows

        # 4. Export as Parquet
        parquet_path = tmp_path / "export.parquet"
        exporter.export_dataset(
            entity_ids=entity_ids, output_path=parquet_path, format=ExportFormat.PARQUET
        )
        assert parquet_path.exists()

    def test_feature_drift_detector_psi_and_alerts(self, populated_feature_store) -> None:
        from production_rag_evaluation.features.drift import FeatureDriftDetector
        from production_rag_evaluation.features.models import FeatureGroup, FeatureRecord
        from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
        from production_rag_evaluation.models.interfaces import FeatureDriftDetectorProtocol

        drift_detector = FeatureDriftDetector(store=populated_feature_store)
        assert isinstance(drift_detector, FeatureDriftDetectorProtocol)

        # Baseline PSI calculation: identical distributions -> PSI = 0.0
        baseline_arr = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
        target_arr_ident = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
        psi_zero = drift_detector.calculate_psi(baseline_arr, target_arr_ident)
        assert psi_zero == 0.0

        # Divergent distribution -> High PSI
        shifted_arr = np.array([120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0])
        psi_high = drift_detector.calculate_psi(baseline_arr, shifted_arr)
        assert psi_high > 0.30

        # Current records with severe shift in query_char_len
        shifted_recs = [
            FeatureRecord(
                entity_id=f"shift_{i}",
                feature_group=FeatureGroup.QUERY,
                features={"query_char_len": 500 + i * 20},
            )
            for i in range(10)
        ]

        report = drift_detector.detect_feature_drift(
            feature_name="query_char_len",
            group=FeatureGroup.QUERY,
            current_records=shifted_recs,
            psi_threshold=0.20,
        )
        assert report.drift_detected is True
        assert report.psi_score > 0.20

        # Alert generation into RiskTelemetryCollector
        collector = RiskTelemetryCollector()
        received = []
        collector.add_alert_listener(lambda a: received.append(a))
        alerts = drift_detector.generate_telemetry_alerts(collector=collector, reports=[report])
        assert len(alerts) == 1
        assert "Feature drift detected" in alerts[0].message
        assert len(received) == 1


class TestMCPSubsystems:
    """Rigorous evaluation of MCP Server, Tool/Resource/Prompt Registries, and Integration Handlers."""

    @pytest.fixture
    def populated_feature_store(self, tmp_path):
        from production_rag_evaluation.features.store import ParquetFeatureStore

        return ParquetFeatureStore(base_path=tmp_path / "mcp_fs_subsystem")

    def test_protocol_conformity(self) -> None:
        from production_rag_evaluation.mcp import (
            MCPIntegrationHandler,
            MCPPromptRegistry,
            MCPResourceRegistry,
            MCPServer,
            MCPSessionHandler,
            MCPToolRegistry,
        )
        from production_rag_evaluation.models.interfaces import (
            MCPIntegrationHandlerProtocol,
            MCPPromptRegistryProtocol,
            MCPResourceRegistryProtocol,
            MCPServerProtocol,
            MCPSessionHandlerProtocol,
            MCPToolRegistryProtocol,
        )

        server = MCPServer()
        assert isinstance(server, MCPServerProtocol)

        tool_reg = MCPToolRegistry()
        assert isinstance(tool_reg, MCPToolRegistryProtocol)

        res_reg = MCPResourceRegistry()
        assert isinstance(res_reg, MCPResourceRegistryProtocol)

        prompt_reg = MCPPromptRegistry()
        assert isinstance(prompt_reg, MCPPromptRegistryProtocol)

        session_mgr = MCPSessionHandler()
        assert isinstance(session_mgr, MCPSessionHandlerProtocol)

        integ = MCPIntegrationHandler()
        assert isinstance(integ, MCPIntegrationHandlerProtocol)

    def test_server_lifecycle_and_handshake(self) -> None:
        from production_rag_evaluation.mcp import (
            MCPConfig,
            create_mcp_server,
        )

        server = create_mcp_server(MCPConfig(server_name="unit-mcp", server_version="1.2.3"))

        # Initialize
        init_res = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 101,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            }
        )
        assert init_res is not None
        assert init_res["id"] == 101
        assert init_res["result"]["serverInfo"]["name"] == "unit-mcp"
        assert init_res["result"]["serverInfo"]["version"] == "1.2.3"
        assert "tools" in init_res["result"]["capabilities"]

        # Notification: initialized
        notif_res = server.handle_request(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
        )
        assert notif_res is None
        assert server._is_initialized is True

        # Ping
        ping_res = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 102,
                "method": "ping",
            }
        )
        assert ping_res is not None
        assert ping_res["result"] == {}

    def test_server_jsonrpc_error_handling(self) -> None:
        import json

        from production_rag_evaluation.mcp import MCPServer

        server = MCPServer()

        # Parse Error (-32700)
        parse_err_raw = server.handle_message("{not-valid-json")
        assert parse_err_raw is not None
        parse_err = json.loads(parse_err_raw)
        assert parse_err["error"]["code"] == -32700

        # Invalid Request (-32600)
        invalid_req = server.handle_request({"not_a_method": True})
        assert invalid_req is not None
        assert invalid_req["error"]["code"] == -32600

        # Method Not Found (-32601)
        missing_method = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 201,
                "method": "unsupported/method",
            }
        )
        assert missing_method is not None
        assert missing_method["error"]["code"] == -32601

        # Invalid Params for tools/call (-32602)
        invalid_params = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 202,
                "method": "tools/call",
                "params": {},
            }
        )
        assert invalid_params is not None
        assert invalid_params["error"]["code"] == -32602

    def test_server_batch_and_notifications(self) -> None:
        import json

        from production_rag_evaluation.mcp import MCPServer

        server = MCPServer()

        # Batch request containing 2 calls and 1 notification
        batch_msg = json.dumps(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ]
        )
        raw_res = server.handle_message(batch_msg)
        assert raw_res is not None
        res_list = json.loads(raw_res)
        assert isinstance(res_list, list)
        assert len(res_list) == 2
        assert res_list[0]["id"] == 1
        assert res_list[1]["id"] == 2

    def test_tool_registry_subsystem(self) -> None:
        from production_rag_evaluation.mcp import (
            MCPTool,
            MCPToolRegistry,
            ToolCallResult,
            ToolParameter,
        )

        registry = MCPToolRegistry()

        # Custom tool with parameters
        custom_tool = MCPTool(
            name="custom_calculator",
            description="Adds two integers",
            parameters=[
                ToolParameter(name="a", type="integer", description="First num", required=True),
                ToolParameter(name="b", type="integer", description="Second num", required=True),
            ],
            handler=lambda args: ToolCallResult(
                content=[{"type": "text", "text": str(args["a"] + args["b"])}]
            ),
        )
        registry.register(custom_tool)

        assert registry.get("custom_calculator") is not None
        listed = registry.list_tools()
        assert any(t["name"] == "custom_calculator" for t in listed)

        # Call success
        res = registry.call("custom_calculator", {"a": 15, "b": 27})
        assert res.isError is False
        assert res.content[0]["text"] == "42"

        # Missing tool
        missing_res = registry.call("non_existent_tool", {})
        assert missing_res.isError is True
        assert "not found" in missing_res.content[0]["text"]

        # Tool without handler
        tool_no_handler = MCPTool(name="no_op", description="Does nothing")
        registry.register(tool_no_handler)
        no_h_res = registry.call("no_op", {})
        assert no_h_res.isError is True
        assert "no execution handler" in no_h_res.content[0]["text"]

        # Handler raising exception
        def failing_handler(args):
            raise RuntimeError("Underlying subsystem failure")

        tool_fail = MCPTool(
            name="failing_tool", description="Always fails", handler=failing_handler
        )
        registry.register(tool_fail)
        fail_res = registry.call("failing_tool", {})
        assert fail_res.isError is True
        assert "Underlying subsystem failure" in fail_res.content[0]["text"]

    def test_resource_registry_subsystem(self) -> None:
        from production_rag_evaluation.mcp import (
            MCPResource,
            MCPResourceRegistry,
            get_default_resource_registry,
        )

        reg = get_default_resource_registry()
        resources = reg.list_resources()
        uris = [r["uri"] for r in resources]
        assert "rag://system/health" in uris
        assert "rag://features/schema" in uris
        assert "rag://finance/scenarios" in uris
        assert "rag://evaluations/summary" in uris

        # Read static and dynamic resources
        health_data = reg.read("rag://system/health")
        assert health_data["contents"][0]["uri"] == "rag://system/health"
        assert "HEALTHY" in health_data["contents"][0]["text"]

        summary_data = reg.read("rag://evaluations/summary")
        assert summary_data["contents"][0]["uri"] == "rag://evaluations/summary"

        # Custom resource and dynamic provider
        custom_reg = MCPResourceRegistry()
        custom_res = MCPResource(
            uri="custom://data/version",
            name="version",
            description="App version",
        )
        custom_reg.register(custom_res, provider=lambda: "v2.0.1")
        resolved = custom_reg.read("custom://data/version")
        assert resolved["contents"][0]["text"] == "v2.0.1"

        # Non-existent resource
        with pytest.raises(KeyError):
            custom_reg.read("custom://missing")

    def test_prompt_registry_subsystem(self) -> None:
        from production_rag_evaluation.mcp import (
            get_default_prompt_registry,
        )

        reg = get_default_prompt_registry()
        prompts = reg.list_prompts()
        names = [p["name"] for p in prompts]
        assert "earnings_call_audit" in names
        assert "sec_10k_verification" in names
        assert "portfolio_risk_audit" in names

        # Render earnings call audit
        rendered = reg.render(
            "earnings_call_audit",
            {
                "company": "NVIDIA",
                "quarter": "Q3-2026",
                "transcript_segment": "Revenue grew 94%.",
                "generated_summary": "Summary of Q3-2026 revenue growth.",
            },
        )
        assert len(rendered["messages"]) == 1
        msg = rendered["messages"][0]
        assert msg["role"] == "user"
        assert "NVIDIA" in msg["content"]["text"]
        assert "Q3-2026" in msg["content"]["text"]

        # Missing required argument
        with pytest.raises(ValueError) as exc:
            reg.render("earnings_call_audit", {"quarter": "Q3-2026"})
        assert "Missing required prompt argument 'company'" in str(exc.value)

        # Unregistered prompt
        with pytest.raises(KeyError):
            reg.render("unknown_prompt", {})

    def test_session_handler_subsystem(self) -> None:
        from production_rag_evaluation.mcp.handlers import MCPSessionHandler

        handler = MCPSessionHandler(auth_token="secret-token-123", session_ttl_seconds=2.0)

        # Unauthorized session
        unauth_session = handler.create_session(
            client_info={"client": "test"}, auth_token="wrong-token"
        )
        assert unauth_session.is_authenticated is False
        assert handler.validate_session(unauth_session.session_id) is False

        # Authorized session
        auth_session = handler.create_session(
            client_info={"client": "valid"}, auth_token="secret-token-123"
        )
        assert auth_session.is_authenticated is True
        assert handler.validate_session(auth_session.session_id) is True

        # Activity touch
        assert auth_session.request_count == 0
        handler.touch_session(auth_session.session_id)
        assert auth_session.request_count == 1

        # Listing active sessions
        active = handler.list_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == auth_session.session_id

        # Close session
        closed = handler.close_session(auth_session.session_id)
        assert closed is True
        assert handler.get_session(auth_session.session_id) is None

    def test_notification_dispatcher_subsystem(self) -> None:
        from production_rag_evaluation.mcp.handlers import MCPNotificationDispatcher

        dispatcher = MCPNotificationDispatcher()
        received_specific: List[Dict[str, Any]] = []
        received_wildcard: List[Dict[str, Any]] = []

        def specific_cb(topic: str, payload: Dict[str, Any]) -> None:
            received_specific.append(payload)

        def wildcard_cb(topic: str, payload: Dict[str, Any]) -> None:
            received_wildcard.append(payload)

        dispatcher.subscribe("alerts", specific_cb)
        dispatcher.subscribe("*", wildcard_cb)

        assert dispatcher.get_subscribers_count("alerts") == 1
        assert dispatcher.get_subscribers_count("*") == 1

        dispatched_count = dispatcher.publish("alerts", {"level": "CRITICAL", "metric": "drift"})
        assert dispatched_count == 2
        assert len(received_specific) == 1
        assert len(received_wildcard) == 1

        # Unsubscribe
        dispatcher.unsubscribe("alerts", specific_cb)
        assert dispatcher.get_subscribers_count("alerts") == 0
        dispatcher.publish("alerts", {"level": "INFO"})
        assert len(received_specific) == 1
        assert len(received_wildcard) == 2

    def test_integration_handler_telemetry_flow(self) -> None:
        from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
        from production_rag_evaluation.mcp.handlers import MCPIntegrationHandler

        collector = RiskTelemetryCollector()
        handler = MCPIntegrationHandler(telemetry_collector=collector)

        # Invoke rag_evaluate
        res = handler.execute_tool(
            "rag_evaluate",
            {
                "query_text": "What was MSFT revenue?",
                "generated_answer": "Microsoft reported revenue of $65.6 billion.",
                "context_chunks": [
                    {
                        "chunk_id": "c1",
                        "content": "Microsoft reported revenue of $65.6 billion.",
                        "document_id": "msft-10q",
                    }
                ],
            },
        )
        assert res.isError is False
        summary = collector.export_summary()
        assert summary["total_evaluations"] == 1
        assert summary["avg_faithfulness"] > 0.0

    def test_integration_handler_feature_store_flow(self, populated_feature_store) -> None:
        from production_rag_evaluation.features.models import FeatureGroup
        from production_rag_evaluation.mcp.handlers import MCPIntegrationHandler

        handler = MCPIntegrationHandler(feature_store=populated_feature_store)

        res = handler.execute_tool(
            "rag_evaluate",
            {
                "query_text": "What were the operating expenses?",
                "generated_answer": "Operating expenses were $14.2 billion.",
                "context_chunks": [
                    {
                        "chunk_id": "c2",
                        "content": "Operating expenses were $14.2 billion for the quarter.",
                        "document_id": "sec-doc",
                    }
                ],
            },
        )
        assert res.isError is False

        # Verify feature store records written
        q_features = populated_feature_store.read_records(group=FeatureGroup.QUERY)
        r_features = populated_feature_store.read_records(group=FeatureGroup.RESPONSE)
        e_features = populated_feature_store.read_records(group=FeatureGroup.EVALUATION)

        assert len(q_features) >= 1
        assert len(r_features) >= 1
        assert len(e_features) >= 1

    def test_integration_handler_execution_stats(self) -> None:
        from production_rag_evaluation.mcp.handlers import MCPIntegrationHandler

        handler = MCPIntegrationHandler()

        handler.execute_tool("rag_risk_metrics", {"returns": [0.01, 0.02, -0.01, 0.03]})
        handler.execute_tool("rag_feature_stats", {})

        stats = handler.get_execution_stats()
        assert stats["total_invocations"] == 2
        assert stats["total_errors"] == 0
        assert "rag_risk_metrics" in stats["stats_by_tool"]
        assert "rag_feature_stats" in stats["stats_by_tool"]
        assert stats["stats_by_tool"]["rag_risk_metrics"]["call_count"] == 1

        # Reset stats
        handler.reset_stats()
        empty_stats = handler.get_execution_stats()
        assert empty_stats["total_invocations"] == 0
        assert len(empty_stats["stats_by_tool"]) == 0

    def test_integration_handler_batch_evaluation(self) -> None:
        from production_rag_evaluation.mcp.handlers import MCPIntegrationHandler

        handler = MCPIntegrationHandler()

        items = [
            {
                "query_text": "What was the EBITDA?",
                "generated_answer": "EBITDA was $5.2 billion.",
                "context_chunks": [
                    {"chunk_id": "c3", "content": "EBITDA was $5.2 billion.", "document_id": "d1"}
                ],
            },
            {
                "query_text": "What was the net margin?",
                "generated_answer": "Net margin was 24%.",
                "context_chunks": [
                    {"chunk_id": "c4", "content": "Net margin reached 24%.", "document_id": "d2"}
                ],
            },
        ]

        batch_results = handler.handle_batch_evaluation(items, mode="FAST")
        assert len(batch_results) == 2
        assert "faithfulness_score" in batch_results[0]
        assert "faithfulness_score" in batch_results[1]
