"""Telemetry collection for RAG evaluation performance and risk monitors."""

from __future__ import annotations

import uuid
from typing import List, Optional

import numpy as np

from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.models.schemas import (
    AlertSeverity,
    EvaluationMetrics,
    RiskMetrics,
    SystemHealthStatus,
    TelemetryAlert,
    TelemetryReport,
)


class RiskTelemetryCollector:
    """Collects, aggregates, and alerts on pipeline evaluation quality and financial metrics."""

    def __init__(
        self,
        evaluator: Optional[RAGEvaluator] = None,
        risk_engine: Optional[FinancialRiskEngine] = None,
        stress_tester: Optional[MonteCarloStressTester] = None,
    ) -> None:
        self.evaluator = evaluator or RAGEvaluator()
        self.risk_engine = risk_engine or FinancialRiskEngine()
        self.stress_tester = stress_tester or MonteCarloStressTester()
        self._history: List[EvaluationMetrics] = []

    def record_evaluation(self, metrics: EvaluationMetrics) -> None:
        """Add evaluation outcome to telemetry time-series."""
        self._history.append(metrics)

    def generate_report(
        self,
        returns: Optional[np.ndarray] = None,
        max_acceptable_hallucination_rate: float = 0.10,
        min_acceptable_sharpe: float = 0.80,
    ) -> TelemetryReport:
        """Synthesize unified telemetry report across RAG and risk monitors."""
        total = len(self._history)
        if total == 0:
            avg_faith = 1.0
            avg_hallucination = 0.0
            avg_latency = 0.0
        else:
            avg_faith = float(np.mean([m.faithfulness_score for m in self._history]))
            avg_hallucination = float(np.mean([m.hallucination_rate for m in self._history]))
            avg_latency = float(np.mean([m.evaluation_latency_ms for m in self._history]))

        cache_stats = self.evaluator.cache_stats()
        hit_rate = cache_stats.get("hit_rate", 0.0)

        alerts: List[str] = []
        structured_alerts: List[TelemetryAlert] = []

        if avg_hallucination > max_acceptable_hallucination_rate:
            msg = (
                f"HIGH_HALLUCINATION_RATE: Average hallucination rate {avg_hallucination:.1%} "
                f"exceeds threshold {max_acceptable_hallucination_rate:.1%}"
            )
            alerts.append(msg)
            structured_alerts.append(
                TelemetryAlert(
                    alert_id=f"alert-hal-{uuid.uuid4().hex[:8]}",
                    severity=AlertSeverity.CRITICAL
                    if avg_hallucination > 0.25
                    else AlertSeverity.WARNING,
                    metric_name="avg_hallucination_rate",
                    threshold=max_acceptable_hallucination_rate,
                    observed_value=avg_hallucination,
                    message=msg,
                )
            )

        # Risk metrics evaluation
        risk_met: Optional[RiskMetrics] = None
        if returns is not None and len(returns) > 1:
            risk_met = self.risk_engine.compute_all_metrics(returns)
            if risk_met.sharpe_ratio < min_acceptable_sharpe:
                msg = (
                    f"LOW_SHARPE_RATIO: Portfolio Sharpe {risk_met.sharpe_ratio:.2f} "
                    f"is below risk floor {min_acceptable_sharpe:.2f}"
                )
                alerts.append(msg)
                structured_alerts.append(
                    TelemetryAlert(
                        alert_id=f"alert-sharpe-{uuid.uuid4().hex[:8]}",
                        severity=AlertSeverity.WARNING,
                        metric_name="sharpe_ratio",
                        threshold=min_acceptable_sharpe,
                        observed_value=risk_met.sharpe_ratio,
                        message=msg,
                    )
                )

        # Monte Carlo stress simulation
        mc_result = None
        if returns is not None and len(returns) > 1:
            mc_result = self.stress_tester.run_stress_test(returns, n_simulations=500)
            if not mc_result.stress_pass:
                msg = (
                    f"STRESS_TEST_FAILED: Worst case drawdown {mc_result.worst_case_drawdown:.1%} "
                    "breached limits under stress"
                )
                alerts.append(msg)
                structured_alerts.append(
                    TelemetryAlert(
                        alert_id=f"alert-stress-{uuid.uuid4().hex[:8]}",
                        severity=AlertSeverity.CRITICAL,
                        metric_name="worst_case_drawdown",
                        threshold=0.25,
                        observed_value=mc_result.worst_case_drawdown,
                        message=msg,
                    )
                )

        health = SystemHealthStatus.HEALTHY
        if any(a.severity == AlertSeverity.CRITICAL for a in structured_alerts):
            health = SystemHealthStatus.UNHEALTHY
        elif len(structured_alerts) > 0:
            health = SystemHealthStatus.DEGRADED

        return TelemetryReport(
            total_evaluations=total,
            avg_faithfulness=round(avg_faith, 4),
            avg_hallucination_rate=round(avg_hallucination, 4),
            cache_hit_rate=round(hit_rate, 4),
            avg_latency_ms=round(avg_latency, 2),
            health_status=health,
            risk_metrics=risk_met,
            monte_carlo_result=mc_result,
            alerts_triggered=alerts,
            structured_alerts=structured_alerts,
        )
