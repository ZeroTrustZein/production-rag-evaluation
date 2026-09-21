"""Domain models and schemas for production RAG evaluation and financial risk metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    """Supported financial document categories."""

    EARNINGS_CALL = "EARNINGS_CALL"
    SEC_10K = "SEC_10K"
    SEC_10Q = "SEC_10Q"
    SEC_8K = "SEC_8K"
    PRESS_RELEASE = "PRESS_RELEASE"
    ANALYST_REPORT = "ANALYST_REPORT"
    FINANCIAL_NEWS = "FINANCIAL_NEWS"


class Severity(str, Enum):
    """Severity ratings for detected hallucinations and anomalies."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class HallucinationType(str, Enum):
    """Categorization of RAG hallucination failure modes."""

    NUMERICAL_DISTORTION = "NUMERICAL_DISTORTION"
    UNGROUNDED_CLAIM = "UNGROUNDED_CLAIM"
    CONTRADICTION = "CONTRADICTION"
    ENTITY_SWAP = "ENTITY_SWAP"
    TEMPORAL_MISMATCH = "TEMPORAL_MISMATCH"
    OMISSION_OF_RISK = "OMISSION_OF_RISK"


class RiskProfile(str, Enum):
    """Risk appetite categorizations for portfolio allocation."""

    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    SPECULATIVE = "SPECULATIVE"


class StressScenarioType(str, Enum):
    """Prescribed stress scenario types for Monte Carlo shocks."""

    BASELINE = "BASELINE"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    LIQUIDITY_CRUNCH = "LIQUIDITY_CRUNCH"
    RATE_SHOCK = "RATE_SHOCK"
    BLACK_SWAN = "BLACK_SWAN"
    HALLUCINATION_CASCADE = "HALLUCINATION_CASCADE"


class ReturnFrequency(str, Enum):
    """Frequency of return observations."""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    ANNUAL = "ANNUAL"


class AlertSeverity(str, Enum):
    """Telemetry alert severity tier."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SystemHealthStatus(str, Enum):
    """Operational health status of the evaluation system."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class ClaimStatus(str, Enum):
    """Status of an individual factual claim evaluation."""

    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    UNGROUNDED = "UNGROUNDED"
    AMBIGUOUS = "AMBIGUOUS"


class EvaluationMode(str, Enum):
    """Execution depth mode for RAG evaluation."""

    FAST = "FAST"
    THOROUGH = "THOROUGH"
    AUDIT = "AUDIT"
    FINANCIAL_STRICT = "FINANCIAL_STRICT"


class NumericalTarget(BaseModel):
    """Expected financial figure or key performance indicator."""

    model_config = ConfigDict(frozen=True)

    metric_name: str = Field(..., min_length=1, description="Metric name (e.g. Net Sales, EPS)")
    target_value: float = Field(..., description="Numerical target value")
    unit: Optional[str] = Field(default=None, description="Currency or unit (e.g. USD, %, shares)")
    tolerance_pct: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Acceptable percentage error tolerance"
    )
    is_currency: bool = Field(default=True, description="Whether metric is a monetary amount")


class ClaimVerificationResult(BaseModel):
    """Result of verifying an individual extracted claim against context."""

    claim_id: str = Field(..., min_length=1)
    claim_text: str = Field(..., min_length=1)
    status: ClaimStatus = Field(default=ClaimStatus.UNGROUNDED)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None


class DocumentChunk(BaseModel):
    """Atomic chunk retrieved from financial corpus."""

    chunk_id: str = Field(..., min_length=1, description="Unique chunk identifier")
    text: str = Field(..., min_length=1, description="Raw textual chunk content")
    doc_type: DocumentType = Field(default=DocumentType.SEC_10K)
    ticker: str = Field(..., min_length=1, description="Stock ticker symbol (e.g. AAPL)")
    period: Optional[str] = Field(
        default=None, description="Reporting fiscal period (e.g. Q3 2024)"
    )
    section: Optional[str] = Field(default=None, description="Filing section (e.g. MD&A, Item 7)")
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Grounding citation linking generated statements back to source context."""

    chunk_id: str = Field(..., min_length=1)
    quote: str = Field(..., min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_ticker: Optional[str] = None
    char_offset: Optional[Tuple[int, int]] = None


class RAGQuery(BaseModel):
    """Inbound query evaluated by the benchmark harness."""

    query_id: str = Field(..., min_length=1)
    query_text: str = Field(..., min_length=1)
    ticker: str = Field(..., min_length=1)
    period: Optional[str] = None
    expected_numerical_targets: List[float] = Field(default_factory=list)
    structured_targets: List[NumericalTarget] = Field(default_factory=list)


class RAGResponse(BaseModel):
    """Generated response to evaluate against retrieved context."""

    query_id: str = Field(..., min_length=1)
    generated_answer: str
    retrieved_chunks: List[DocumentChunk] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    model_name: Optional[str] = None


class HallucinationFinding(BaseModel):
    """Detailed diagnosis of a verified or suspected hallucination."""

    finding_type: HallucinationType
    severity: Severity
    claim_text: str = Field(..., min_length=1)
    reference_text: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    explanation: str
    impact_on_risk: Optional[str] = None


class EvaluationMetrics(BaseModel):
    """Holistic multi-dimensional RAG evaluation metrics."""

    faithfulness_score: float = Field(..., ge=0.0, le=1.0)
    answer_relevance_score: float = Field(..., ge=0.0, le=1.0)
    context_recall_score: float = Field(..., ge=0.0, le=1.0)
    context_precision_score: float = Field(..., ge=0.0, le=1.0)
    citation_grounding_score: float = Field(..., ge=0.0, le=1.0)
    hallucination_rate: float = Field(..., ge=0.0, le=1.0)
    findings: List[HallucinationFinding] = Field(default_factory=list)
    claim_verifications: List[ClaimVerificationResult] = Field(default_factory=list)
    overall_pass: bool
    evaluation_latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskMetrics(BaseModel):
    """Quantitative risk metrics computed on financial outcomes / trading returns."""

    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float = Field(..., ge=0.0, le=1.0)
    calmar_ratio: float
    annualized_return: float
    annualized_volatility: float
    downside_deviation: float
    var_95: float = Field(..., description="1-day 95% Value-at-Risk")
    cvar_95: float = Field(..., description="1-day 95% Conditional VaR / Expected Shortfall")
    var_99: Optional[float] = Field(default=None, description="1-day 99% Value-at-Risk")
    cvar_99: Optional[float] = Field(default=None, description="1-day 99% Conditional VaR")
    information_ratio: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None


class StressTestConfig(BaseModel):
    """Configuration parameters for Monte Carlo stress simulation."""

    n_simulations: int = Field(default=1000, ge=10, le=100_000)
    horizon_days: int = Field(default=30, ge=1, le=1260)
    volatility_shock: float = Field(default=1.5, ge=0.1, le=10.0)
    drift_shock: float = Field(default=0.0)
    max_allowable_drawdown: float = Field(default=0.25, ge=0.0, le=1.0)
    confidence_levels: List[float] = Field(default_factory=lambda: [0.05, 0.50, 0.95])
    initial_value: float = Field(default=100.0, gt=0.0)
    random_seed: Optional[int] = 42
    scenario_type: StressScenarioType = Field(default=StressScenarioType.BASELINE)


class MonteCarloStressResult(BaseModel):
    """Monte Carlo stress testing simulation outcome."""

    n_simulations: int
    horizon_steps: int
    p5_terminal: float
    p50_terminal: float
    p95_terminal: float
    mean_terminal: float
    std_terminal: float
    worst_case_drawdown: float
    stress_pass: bool
    scenario_type: StressScenarioType = Field(default=StressScenarioType.BASELINE)
    breach_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    iterations_data: Optional[List[float]] = None


class TelemetryAlert(BaseModel):
    """Structured operational alert triggered by risk or evaluation anomalies."""

    alert_id: str = Field(..., min_length=1)
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    metric_name: str = Field(..., min_length=1)
    threshold: float
    observed_value: float
    message: str = Field(..., min_length=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TelemetryReport(BaseModel):
    """System-level telemetry capturing RAG quality, cache performance, and risk metrics."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_evaluations: int
    avg_faithfulness: float
    avg_hallucination_rate: float
    cache_hit_rate: float
    avg_latency_ms: float
    health_status: SystemHealthStatus = Field(default=SystemHealthStatus.HEALTHY)
    risk_metrics: Optional[RiskMetrics] = None
    monte_carlo_result: Optional[MonteCarloStressResult] = None
    alerts_triggered: List[str] = Field(default_factory=list)
    structured_alerts: List[TelemetryAlert] = Field(default_factory=list)


class BatchEvaluationSummary(BaseModel):
    """Aggregated outcome of a batch evaluation run."""

    batch_id: str = Field(..., min_length=1)
    total_queries: int = Field(..., ge=0)
    passed_queries: int = Field(..., ge=0)
    failed_queries: int = Field(..., ge=0)
    pass_rate: float = Field(..., ge=0.0, le=1.0)
    mean_faithfulness: float = Field(..., ge=0.0, le=1.0)
    mean_relevance: float = Field(..., ge=0.0, le=1.0)
    mean_latency_ms: float = Field(default=0.0, ge=0.0)
    p95_latency_ms: float = Field(default=0.0, ge=0.0)
    results: List[EvaluationMetrics] = Field(default_factory=list)
