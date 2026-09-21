"""Telemetry collection for RAG evaluation performance and risk monitors."""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import scipy.stats as stats

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
        self._alert_listeners: List[Callable[[TelemetryAlert], None]] = []

    def add_alert_listener(self, listener: Callable[[TelemetryAlert], None]) -> None:
        """Register a callback invoked whenever an operational alert triggers."""
        self._alert_listeners.append(listener)

    def record_evaluation(self, metrics: EvaluationMetrics) -> None:
        """Add evaluation outcome to telemetry time-series."""
        self._history.append(metrics)

    def clear_history(self) -> None:
        """Reset historical evaluation telemetry."""
        self._history.clear()

    def get_evaluations(
        self, limit: Optional[int] = None, passed_only: Optional[bool] = None
    ) -> List[EvaluationMetrics]:
        """Query evaluation history with optional limit and pass/fail filtering."""
        records = self._history
        if passed_only is not None:
            records = [r for r in records if r.overall_pass == passed_only]
        if limit is not None:
            records = records[-limit:]
        return list(records)

    def get_rolling_metrics(self, window_size: int = 20) -> Dict[str, float]:
        """Calculate rolling window averages for RAG quality metrics."""
        if not self._history:
            return {
                "rolling_faithfulness": 1.0,
                "rolling_relevance": 1.0,
                "rolling_hallucination_rate": 0.0,
                "rolling_latency_ms": 0.0,
            }

        window = self._history[-window_size:]
        return {
            "rolling_faithfulness": round(float(np.mean([m.faithfulness_score for m in window])), 4),
            "rolling_relevance": round(float(np.mean([m.answer_relevance_score for m in window])), 4),
            "rolling_hallucination_rate": round(float(np.mean([m.hallucination_rate for m in window])), 4),
            "rolling_latency_ms": round(float(np.mean([m.evaluation_latency_ms for m in window])), 2),
        }

    def detect_drift(
        self,
        metric: str = "faithfulness_score",
        baseline_size: int = 20,
        test_size: int = 10,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Detect statistically significant performance degradation using two-sample t-test."""
        total = len(self._history)
        min_required = baseline_size + test_size

        if total < min_required:
            return {
                "drift_detected": False,
                "metric": metric,
                "reason": f"Insufficient data points ({total}/{min_required})",
                "p_value": 1.0,
            }

        all_vals = [getattr(m, metric, 0.0) for m in self._history]
        baseline = all_vals[:baseline_size]
        recent = all_vals[-test_size:]

        res = stats.ttest_ind(baseline, recent, equal_var=False)
        p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
        stat = float(res.statistic) if not np.isnan(res.statistic) else 0.0

        # Drift if statistically significant difference and performance dropped
        drift = bool(p_val < alpha and np.mean(recent) < np.mean(baseline))

        return {
            "drift_detected": drift,
            "metric": metric,
            "p_value": round(p_val, 6),
            "t_statistic": round(stat, 4),
            "baseline_mean": round(float(np.mean(baseline)), 4),
            "recent_mean": round(float(np.mean(recent)), 4),
        }

    def generate_report(
        self,
        returns: Optional[np.ndarray] = None,
        max_acceptable_hallucination_rate: float = 0.10,
        min_acceptable_sharpe: float = 0.80,
        min_acceptable_faithfulness: float = 0.75,
        max_acceptable_latency_ms: float = 2000.0,
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

        def _emit_alert(alert: TelemetryAlert) -> None:
            alerts.append(alert.message)
            structured_alerts.append(alert)
            for listener in self._alert_listeners:
                try:
                    listener(alert)
                except Exception:
                    pass

        # Hallucination alert
        if avg_hallucination > max_acceptable_hallucination_rate:
            msg = (
                f"HIGH_HALLUCINATION_RATE: Average hallucination rate {avg_hallucination:.1%} "
                f"exceeds threshold {max_acceptable_hallucination_rate:.1%}"
            )
            _emit_alert(
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

        # Faithfulness alert
        if avg_faith < min_acceptable_faithfulness and total > 0:
            msg = (
                f"LOW_FAITHFULNESS: Average faithfulness {avg_faith:.1%} "
                f"below threshold {min_acceptable_faithfulness:.1%}"
            )
            _emit_alert(
                TelemetryAlert(
                    alert_id=f"alert-faith-{uuid.uuid4().hex[:8]}",
                    severity=AlertSeverity.CRITICAL
                    if avg_faith < 0.50
                    else AlertSeverity.WARNING,
                    metric_name="avg_faithfulness",
                    threshold=min_acceptable_faithfulness,
                    observed_value=avg_faith,
                    message=msg,
                )
            )

        # Latency alert
        if avg_latency > max_acceptable_latency_ms and total > 0:
            msg = (
                f"HIGH_LATENCY: Average evaluation latency {avg_latency:.1f}ms "
                f"exceeds SLA {max_acceptable_latency_ms:.1f}ms"
            )
            _emit_alert(
                TelemetryAlert(
                    alert_id=f"alert-lat-{uuid.uuid4().hex[:8]}",
                    severity=AlertSeverity.WARNING,
                    metric_name="avg_latency_ms",
                    threshold=max_acceptable_latency_ms,
                    observed_value=avg_latency,
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
                _emit_alert(
                    TelemetryAlert(
                        alert_id=f"alert-sharpe-{uuid.uuid4().hex[:8]}",
                        severity=AlertSeverity.CRITICAL
                        if risk_met.sharpe_ratio < 0.0
                        else AlertSeverity.WARNING,
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
                _emit_alert(
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

    def export_summary(self) -> Dict[str, Any]:
        """Export system metrics snapshot as a plain dictionary."""
        report = self.generate_report()
        rolling = self.get_rolling_metrics()
        return {
            "timestamp": report.timestamp,
            "total_evaluations": report.total_evaluations,
            "avg_faithfulness": report.avg_faithfulness,
            "avg_hallucination_rate": report.avg_hallucination_rate,
            "cache_hit_rate": report.cache_hit_rate,
            "avg_latency_ms": report.avg_latency_ms,
            "health_status": report.health_status.value,
            "alerts_count": len(report.structured_alerts),
            "rolling_metrics": rolling,
        }
