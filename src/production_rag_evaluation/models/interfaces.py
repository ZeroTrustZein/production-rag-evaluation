"""Protocol interfaces and contracts for RAG evaluation, caching, and financial risk engines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    EvaluationMetrics,
    HallucinationFinding,
    MonteCarloStressResult,
    RAGQuery,
    RAGResponse,
    RiskMetrics,
    TelemetryReport,
)


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Contract for RAG evaluation orchestrators."""

    def evaluate(
        self, query: RAGQuery, response: RAGResponse, use_cache: Optional[bool] = None
    ) -> EvaluationMetrics:
        """Evaluate a single query and response pair."""
        ...

    def evaluate_batch(
        self,
        pairs: List[Tuple[RAGQuery, RAGResponse]],
        use_cache: Optional[bool] = None,
    ) -> List[EvaluationMetrics]:
        """Evaluate a batch of query-response pairs."""
        ...

    def clear_cache(self) -> None:
        """Purge all cached results."""
        ...

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache health and hit statistics."""
        ...


@runtime_checkable
class HallucinationDetectorProtocol(Protocol):
    """Contract for domain hallucination detection engines."""

    def analyze(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Run all detection rules against text and context chunks."""
        ...

    def detect_numerical_distortions(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Detect numeric deviations from context figures."""
        ...

    def detect_directional_contradictions(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Detect sentiment / direction contradictions against context."""
        ...

    def detect_ungrounded_claims(
        self,
        generated_answer: str,
        context_chunks: List[DocumentChunk],
        min_overlap_ratio: float = 0.30,
    ) -> List[HallucinationFinding]:
        """Flag claims lacking lexical grounding in context."""
        ...


@runtime_checkable
class FinancialRiskEngineProtocol(Protocol):
    """Contract for financial return analysis and risk indicator computation."""

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """Calculate annualized Sharpe Ratio."""
        ...

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        target_return: float = 0.0,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """Calculate annualized Sortino Ratio focusing on downside risk."""
        ...

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate peak-to-trough maximum drawdown."""
        ...

    def calculate_calmar_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """Calculate Calmar Ratio."""
        ...

    def calculate_var_cvar(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Compute Value-at-Risk and Conditional VaR (Expected Shortfall)."""
        ...

    def compute_all_metrics(
        self, returns: np.ndarray, risk_free_rate: Optional[float] = None
    ) -> RiskMetrics:
        """Compute comprehensive suite of risk indicators."""
        ...


@runtime_checkable
class MonteCarloStressTesterProtocol(Protocol):
    """Contract for stochastic stress testing simulation engines."""

    def simulate_gbm_paths(
        self,
        initial_value: float = 100.0,
        mu: float = 0.08,
        sigma: float = 0.20,
        horizon_days: int = 60,
        n_simulations: int = 1000,
        volatility_shock_multiplier: float = 1.0,
    ) -> np.ndarray:
        """Simulate geometric brownian motion trajectories."""
        ...

    def run_stress_test(
        self,
        returns: np.ndarray,
        n_simulations: int = 1000,
        horizon_days: int = 30,
        volatility_shock: float = 1.5,
        max_allowable_drawdown: float = 0.25,
    ) -> MonteCarloStressResult:
        """Run stress simulation across empirical return distribution."""
        ...

    def stress_test_hallucination_impact(
        self,
        base_financial_estimates: List[float],
        hallucination_probability: float = 0.15,
        hallucination_distortion_pct: float = 0.25,
        n_simulations: int = 500,
    ) -> float:
        """Simulate hallucination breach frequency on forecast aggregates."""
        ...


@runtime_checkable
class SemanticCacheProtocol(Protocol):
    """Contract for fast semantic vector retrieval caches."""

    def get(self, query_text: str) -> Optional[Tuple[Any, float]]:
        """Retrieve entry by exact or semantic match."""
        ...

    def put(self, query_text: str, value: Any) -> str:
        """Store value indexed by query representation."""
        ...

    def clear(self) -> None:
        """Evict all cached entries."""
        ...

    def stats(self) -> Dict[str, Any]:
        """Return cache health telemetry."""
        ...


@runtime_checkable
class RiskTelemetryCollectorProtocol(Protocol):
    """Contract for telemetry gathering and alerting service."""

    def record_evaluation(self, metrics: EvaluationMetrics) -> None:
        """Record evaluation outcome."""
        ...

    def generate_report(
        self,
        returns: Optional[np.ndarray] = None,
        max_acceptable_hallucination_rate: float = 0.10,
        min_acceptable_sharpe: float = 0.80,
    ) -> TelemetryReport:
        """Generate synthesized monitoring report."""
        ...
