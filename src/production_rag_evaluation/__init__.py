"""Production RAG Evaluation and Hallucination Detection Pipeline."""

__version__ = "0.1.0"
__author__ = "Portfolio Automation"

from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.core.graph import EvaluationGraph
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.models.interfaces import (
    EvaluatorProtocol,
    FinancialRiskEngineProtocol,
    HallucinationDetectorProtocol,
    MonteCarloStressTesterProtocol,
    RiskTelemetryCollectorProtocol,
    SemanticCacheProtocol,
)
from production_rag_evaluation.models.schemas import (
    AlertSeverity,
    BatchEvaluationSummary,
    Citation,
    ClaimStatus,
    ClaimVerificationResult,
    DocumentChunk,
    DocumentType,
    EvaluationMetrics,
    EvaluationMode,
    HallucinationFinding,
    HallucinationType,
    MonteCarloStressResult,
    NumericalTarget,
    RAGQuery,
    RAGResponse,
    ReturnFrequency,
    RiskMetrics,
    RiskProfile,
    Severity,
    StressScenarioType,
    StressTestConfig,
    SystemHealthStatus,
    TelemetryAlert,
    TelemetryReport,
)

__all__ = [
    "__version__",
    # Core Engines
    "RAGEvaluator",
    "SemanticCache",
    "EvaluationGraph",
    "HallucinationDetector",
    "FinancialRiskEngine",
    "MonteCarloStressTester",
    "RiskTelemetryCollector",
    # Models & Schemas
    "AlertSeverity",
    "BatchEvaluationSummary",
    "Citation",
    "ClaimStatus",
    "ClaimVerificationResult",
    "DocumentChunk",
    "DocumentType",
    "EvaluationMetrics",
    "EvaluationMode",
    "HallucinationFinding",
    "HallucinationType",
    "MonteCarloStressResult",
    "NumericalTarget",
    "RAGQuery",
    "RAGResponse",
    "ReturnFrequency",
    "RiskMetrics",
    "RiskProfile",
    "Severity",
    "StressScenarioType",
    "StressTestConfig",
    "SystemHealthStatus",
    "TelemetryAlert",
    "TelemetryReport",
    # Protocols
    "EvaluatorProtocol",
    "FinancialRiskEngineProtocol",
    "HallucinationDetectorProtocol",
    "MonteCarloStressTesterProtocol",
    "RiskTelemetryCollectorProtocol",
    "SemanticCacheProtocol",
]
