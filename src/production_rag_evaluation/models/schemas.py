"""Domain models and schemas for production RAG evaluation and financial risk metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported financial document categories."""

    EARNINGS_CALL = "EARNINGS_CALL"
    SEC_10K = "SEC_10K"
    SEC_10Q = "SEC_10Q"
    PRESS_RELEASE = "PRESS_RELEASE"
    ANALYST_REPORT = "ANALYST_REPORT"


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


class DocumentChunk(BaseModel):
    """Atomic chunk retrieved from financial corpus."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., min_length=1, description="Raw textual chunk content")
    doc_type: DocumentType = Field(default=DocumentType.SEC_10K)
    ticker: str = Field(..., min_length=1, description="Stock ticker symbol (e.g. AAPL)")
    period: Optional[str] = Field(default=None, description="Reporting fiscal period (e.g. Q3 2024)")
    section: Optional[str] = Field(default=None, description="Filing section (e.g. MD&A, Item 7)")
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Grounding citation linking generated statements back to source context."""

    chunk_id: str
    quote: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_ticker: Optional[str] = None


class RAGQuery(BaseModel):
    """Inbound query evaluated by the benchmark harness."""

    query_id: str
    query_text: str = Field(..., min_length=1)
    ticker: str
    period: Optional[str] = None
    expected_numerical_targets: List[float] = Field(default_factory=list)


class RAGResponse(BaseModel):
    """Generated response to evaluate against retrieved context."""

    query_id: str
    generated_answer: str
    retrieved_chunks: List[DocumentChunk] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)
    token_usage: Dict[str, int] = Field(default_factory=dict)


class HallucinationFinding(BaseModel):
    """Detailed diagnosis of a verified or suspected hallucination."""

    finding_type: HallucinationType
    severity: Severity
    claim_text: str
    reference_text: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    explanation: str


class EvaluationMetrics(BaseModel):
    """Holistic multi-dimensional RAG evaluation metrics."""

    faithfulness_score: float = Field(..., ge=0.0, le=1.0)
    answer_relevance_score: float = Field(..., ge=0.0, le=1.0)
    context_recall_score: float = Field(..., ge=0.0, le=1.0)
    context_precision_score: float = Field(..., ge=0.0, le=1.0)
    citation_grounding_score: float = Field(..., ge=0.0, le=1.0)
    hallucination_rate: float = Field(..., ge=0.0, le=1.0)
    findings: List[HallucinationFinding] = Field(default_factory=list)
    overall_pass: bool
    evaluation_latency_ms: float = 0.0


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
    iterations_data: Optional[List[float]] = None


class TelemetryReport(BaseModel):
    """System-level telemetry capturing RAG quality, cache performance, and risk metrics."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_evaluations: int
    avg_faithfulness: float
    avg_hallucination_rate: float
    cache_hit_rate: float
    avg_latency_ms: float
    risk_metrics: Optional[RiskMetrics] = None
    monte_carlo_result: Optional[MonteCarloStressResult] = None
    alerts_triggered: List[str] = Field(default_factory=list)
