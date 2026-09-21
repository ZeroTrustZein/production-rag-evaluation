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

__all__ = [
    "__version__",
    "RAGEvaluator",
    "SemanticCache",
    "EvaluationGraph",
    "HallucinationDetector",
    "FinancialRiskEngine",
    "MonteCarloStressTester",
    "RiskTelemetryCollector",
]
