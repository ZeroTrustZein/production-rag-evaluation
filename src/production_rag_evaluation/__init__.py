"""Production RAG Evaluation and Hallucination Detection Pipeline."""

__version__ = "0.1.0"
__author__ = "Portfolio Automation"

from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.core.graph import EvaluationGraph
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.features.extractors import (
    BaseFeatureExtractor,
    ChunkFeatureExtractor,
    EvaluationFeatureExtractor,
    QueryFeatureExtractor,
    ResponseFeatureExtractor,
)
from production_rag_evaluation.features.models import (
    FeatureDefinition,
    FeatureGroup,
    FeatureRecord,
    FeatureSchema,
    FeatureValueType,
    StreamingBatchStats,
)
from production_rag_evaluation.features.pipeline import StreamingFeaturePipeline
from production_rag_evaluation.features.store import ParquetFeatureStore
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.models.interfaces import (
    EvaluatorProtocol,
    FeatureStoreProtocol,
    FinancialRiskEngineProtocol,
    HallucinationDetectorProtocol,
    MonteCarloStressTesterProtocol,
    RiskTelemetryCollectorProtocol,
    SemanticCacheProtocol,
    StreamingPipelineProtocol,
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
    # Feature Store & Streaming Pipeline
    "BaseFeatureExtractor",
    "ChunkFeatureExtractor",
    "EvaluationFeatureExtractor",
    "FeatureDefinition",
    "FeatureGroup",
    "FeatureRecord",
    "FeatureSchema",
    "FeatureValueType",
    "ParquetFeatureStore",
    "QueryFeatureExtractor",
    "ResponseFeatureExtractor",
    "StreamingBatchStats",
    "StreamingFeaturePipeline",
    # Protocols
    "EvaluatorProtocol",
    "FeatureStoreProtocol",
    "FinancialRiskEngineProtocol",
    "HallucinationDetectorProtocol",
    "MonteCarloStressTesterProtocol",
    "RiskTelemetryCollectorProtocol",
    "SemanticCacheProtocol",
    "StreamingPipelineProtocol",
]
