"""Comprehensive unit and integration test suite for financial subsystems.

Covers FinancialRiskEngine, MonteCarloStressTester, and RiskTelemetryCollector.
"""

from __future__ import annotations

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
