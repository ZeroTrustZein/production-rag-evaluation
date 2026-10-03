"""Tool definitions and registry for MCP client execution."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.features.models import FeatureGroup
from production_rag_evaluation.features.registry import get_default_registry
from production_rag_evaluation.features.store import ParquetFeatureStore
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.mcp.models import MCPTool, ToolCallResult
from production_rag_evaluation.models.schemas import (
    Citation,
    DocumentChunk,
    DocumentType,
    EvaluationMode,
    NumericalTarget,
    RAGQuery,
    RAGResponse,
    StressScenarioType,
    StressTestConfig,
)

logger = logging.getLogger(__name__)


class MCPToolRegistry:
    """Registry managing tool schemas and dispatching invocations."""

    def __init__(self) -> None:
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        """Register a new tool definition."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[MCPTool]:
        """Retrieve tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return all registered tools formatted per MCP tools/list specification."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Dispatch a tool call by name with provided arguments."""
        tool = self.get(name)
        if tool is None:
            return ToolCallResult.text(
                f"Error: Tool '{name}' not found in registry",
                is_error=True,
                metadata={"error_type": "ToolNotFound", "tool_name": name},
            )

        if tool.handler is None:
            return ToolCallResult.text(
                f"Error: Tool '{name}' has no execution handler attached",
                is_error=True,
                metadata={"error_type": "NoHandler", "tool_name": name},
            )

        try:
            res = tool.handler(**arguments)
            if isinstance(res, ToolCallResult):
                return res
            if isinstance(res, str):
                return ToolCallResult.text(res)
            return ToolCallResult.text(json.dumps(res, default=str, indent=2))
        except TypeError as te:
            logger.warning("Invalid arguments passed to tool '%s': %s", name, te)
            return ToolCallResult.text(
                f"Invalid arguments for tool '{name}': {te}",
                is_error=True,
                metadata={"error_type": "InvalidArgument", "details": str(te)},
            )
        except Exception as exc:
            logger.exception("Execution error in tool '%s': %s", name, exc)
            return ToolCallResult.text(
                f"Tool execution failed: {exc}",
                is_error=True,
                metadata={"error_type": exc.__class__.__name__, "details": str(exc)},
            )


# Global singletons for tool performance and caching
_shared_evaluator: Optional[RAGEvaluator] = None
_shared_detector: Optional[HallucinationDetector] = None
_shared_risk_engine: Optional[FinancialRiskEngine] = None
_shared_stress_tester: Optional[MonteCarloStressTester] = None
_shared_semantic_cache: Optional[SemanticCache] = None


def _get_evaluator() -> RAGEvaluator:
    global _shared_evaluator
    if _shared_evaluator is None:
        _shared_evaluator = RAGEvaluator()
    return _shared_evaluator


def _get_detector() -> HallucinationDetector:
    global _shared_detector
    if _shared_detector is None:
        _shared_detector = HallucinationDetector()
    return _shared_detector


def _get_risk_engine() -> FinancialRiskEngine:
    global _shared_risk_engine
    if _shared_risk_engine is None:
        _shared_risk_engine = FinancialRiskEngine()
    return _shared_risk_engine


def _get_stress_tester() -> MonteCarloStressTester:
    global _shared_stress_tester
    if _shared_stress_tester is None:
        _shared_stress_tester = MonteCarloStressTester()
    return _shared_stress_tester


def _get_cache() -> SemanticCache:
    global _shared_semantic_cache
    if _shared_semantic_cache is None:
        _shared_semantic_cache = SemanticCache()
    return _shared_semantic_cache


def handle_rag_evaluate(
    query_text: str,
    generated_answer: str,
    context_chunks: Sequence[str],
    ticker: Optional[str] = None,
    mode: str = "THOROUGH",
    numerical_targets: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end RAG evaluation pipeline."""
    evaluator = _get_evaluator()
    active_ticker = ticker or "DEFAULT"

    chunks = [
        DocumentChunk(
            chunk_id=f"ctx-{idx + 1}",
            text=c,
            doc_type=DocumentType.SEC_10K if ticker else DocumentType.FINANCIAL_NEWS,
            ticker=active_ticker,
        )
        for idx, c in enumerate(context_chunks)
    ]

    targets: List[NumericalTarget] = []
    if numerical_targets:
        for t in numerical_targets:
            targets.append(
                NumericalTarget(
                    metric_name=t.get("metric_name", "target"),
                    expected_value=float(t.get("expected_value", 0.0)),
                    unit=t.get("unit"),
                    tolerance_pct=float(t.get("tolerance_pct", 0.01)),
                )
            )

    rag_query = RAGQuery(
        query_id="mcp-eval-q",
        query_text=query_text,
        ticker=active_ticker,
        numerical_targets=targets,
    )

    citations: List[Citation] = []
    for chunk in chunks:
        if any(word in generated_answer.lower() for word in chunk.text.lower().split()[:5]):
            citations.append(Citation(chunk_id=chunk.chunk_id, quote=chunk.text[:80]))

    rag_resp = RAGResponse(
        query_id="mcp-eval-q",
        generated_answer=generated_answer,
        retrieved_chunks=chunks,
        citations=citations,
    )

    eval_mode = EvaluationMode(mode.upper()) if hasattr(EvaluationMode, mode.upper()) else EvaluationMode.THOROUGH
    metrics = evaluator.evaluate(rag_query, rag_resp, mode=eval_mode)

    overall_score = float(
        np.mean([
            metrics.faithfulness_score,
            metrics.answer_relevance_score,
            metrics.context_precision_score,
            metrics.citation_grounding_score,
        ])
    )

    return {
        "overall_score": round(overall_score, 4),
        "faithfulness_score": round(metrics.faithfulness_score, 4),
        "answer_relevance_score": round(metrics.answer_relevance_score, 4),
        "context_precision_score": round(metrics.context_precision_score, 4),
        "context_recall_score": round(metrics.context_recall_score, 4),
        "citation_grounding_score": round(metrics.citation_grounding_score, 4),
        "hallucination_rate": round(metrics.hallucination_rate, 4),
        "overall_pass": metrics.overall_pass,
        "findings_count": len(metrics.findings),
        "claim_verifications_count": len(metrics.claim_verifications),
        "evaluation_latency_ms": round(metrics.evaluation_latency_ms, 2),
    }


def handle_rag_detect_hallucinations(
    generated_answer: str,
    context_chunks: Sequence[str],
    numerical_tolerance_pct: float = 0.01,
) -> Dict[str, Any]:
    """Audit response text for factual and numerical distortions."""
    detector = _get_detector()
    chunks = [
        DocumentChunk(
            chunk_id=f"ctx-{idx + 1}",
            text=c,
            doc_type=DocumentType.SEC_10K,
            ticker="DEFAULT",
        )
        for idx, c in enumerate(context_chunks)
    ]

    distortions = detector.detect_numerical_distortions(generated_answer, chunks)
    contradictions = detector.detect_directional_contradictions(generated_answer, chunks)
    ungrounded = detector.detect_ungrounded_claims(generated_answer, chunks)

    return {
        "has_hallucinations": bool(distortions or contradictions or any(not u.is_grounded for u in ungrounded)),
        "numerical_distortions": [
            {
                "claim": f.claim_text,
                "severity": f.severity.value,
                "confidence": round(f.confidence, 4),
                "source_evidence": f.source_evidence,
            }
            for f in distortions
        ],
        "contradictions": [
            {
                "claim": f.claim_text,
                "severity": f.severity.value,
                "confidence": round(f.confidence, 4),
                "source_evidence": f.source_evidence,
            }
            for f in contradictions
        ],
        "ungrounded_claims": [
            {
                "claim": u.claim_text,
                "is_grounded": u.is_grounded,
                "confidence": round(u.confidence_score, 4),
                "contradiction": u.is_contradiction,
            }
            for u in ungrounded
            if not u.is_grounded
        ],
        "total_claims_checked": len(ungrounded),
    }


def handle_rag_risk_metrics(
    returns: Sequence[float],
    benchmark_returns: Optional[Sequence[float]] = None,
    risk_free_rate: float = 0.045,
    frequency: str = "DAILY",
) -> Dict[str, Any]:
    """Calculate institutional risk metrics on return series."""
    engine = _get_risk_engine()
    ret_arr = np.array(returns, dtype=np.float64)
    bench_arr = np.array(benchmark_returns, dtype=np.float64) if benchmark_returns else None

    metrics = engine.compute_all_metrics(
        returns=ret_arr,
        risk_free_rate=risk_free_rate,
        benchmark_returns=bench_arr,
    )

    return {
        "sharpe_ratio": round(metrics.sharpe_ratio, 4),
        "sortino_ratio": round(metrics.sortino_ratio, 4),
        "calmar_ratio": round(metrics.calmar_ratio, 4),
        "var_95": round(metrics.var_95, 6),
        "var_99": round(metrics.var_99, 6) if metrics.var_99 is not None else None,
        "cvar_95": round(metrics.cvar_95, 6),
        "cvar_99": round(metrics.cvar_99, 6) if metrics.cvar_99 is not None else None,
        "max_drawdown": round(metrics.max_drawdown, 6),
        "annualized_return": round(metrics.annualized_return, 4),
        "annualized_volatility": round(metrics.annualized_volatility, 4),
        "downside_deviation": round(metrics.downside_deviation, 4),
        "information_ratio": round(metrics.information_ratio, 4) if metrics.information_ratio is not None else None,
        "beta": round(metrics.beta, 4) if metrics.beta is not None else None,
        "alpha": round(metrics.alpha, 4) if metrics.alpha is not None else None,
    }


def handle_rag_stress_test(
    returns: Sequence[float],
    num_simulations: int = 1000,
    horizon_days: int = 30,
    scenario: str = "BASELINE",
    confidence_interval: float = 0.95,
) -> Dict[str, Any]:
    """Execute Monte Carlo forward stress test under simulated market conditions."""
    tester = _get_stress_tester()
    ret_arr = np.array(returns, dtype=np.float64)

    scen_type = (
        StressScenarioType(scenario.upper())
        if hasattr(StressScenarioType, scenario.upper())
        else StressScenarioType.BASELINE
    )
    config = StressTestConfig(
        n_simulations=num_simulations,
        horizon_steps=horizon_days,
        scenario_type=scen_type,
    )

    result = tester.run_stress_test(ret_arr, config=config)
    return {
        "scenario": result.scenario_type.value,
        "n_simulations": result.n_simulations,
        "horizon_steps": result.horizon_steps,
        "mean_terminal": round(result.mean_terminal, 6),
        "std_terminal": round(result.std_terminal, 6),
        "p5_terminal": round(result.p5_terminal, 6),
        "p50_terminal": round(result.p50_terminal, 6),
        "p95_terminal": round(result.p95_terminal, 6),
        "worst_case_drawdown": round(result.worst_case_drawdown, 6),
        "stress_pass": result.stress_pass,
        "breach_probability": round(result.breach_probability, 4),
    }


def handle_rag_feature_stats(
    feature_group: str = "ALL",
    store_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect feature definitions, counts, and Parquet storage metrics."""
    registry = get_default_registry()
    definitions = registry.list_definitions()

    group_filter = feature_group.upper()
    if group_filter != "ALL" and hasattr(FeatureGroup, group_filter):
        definitions = [d for d in definitions if d.group.value == group_filter]

    store = ParquetFeatureStore(base_path=store_path) if store_path else ParquetFeatureStore()
    summary = store.storage_summary()

    return {
        "total_registered_features": len(definitions),
        "feature_definitions": [
            {
                "name": d.name,
                "group": d.group.value,
                "value_type": d.value_type.value,
                "description": d.description,
            }
            for d in definitions[:20]
        ],
        "storage_summary": summary,
    }


def handle_rag_cache_lookup(
    query_text: Optional[str] = None,
    action: str = "GET",
) -> Dict[str, Any]:
    """Interact with semantic evaluation cache."""
    cache = _get_cache()
    act = action.upper()

    if act == "CLEAR":
        cache.clear()
        return {"action": "CLEAR", "success": True, "size": cache.size}

    if act == "STATS":
        return {"action": "STATS", "stats": cache.stats()}

    # GET action
    if not query_text:
        return {"action": "GET", "error": "query_text is required for cache lookup"}

    match = cache.get(query_text)
    if match is None:
        return {"action": "GET", "hit": False, "query_text": query_text}

    value, similarity = match
    return {
        "action": "GET",
        "hit": True,
        "similarity": round(similarity, 4),
        "query_text": query_text,
        "cached_data": value,
    }


def get_default_tool_registry() -> MCPToolRegistry:
    """Build and populate standard tool registry for production-rag-evaluation."""
    reg = MCPToolRegistry()

    reg.register(
        MCPTool(
            name="rag_evaluate",
            description="Evaluate financial RAG response faithfulness, relevance, precision, and hallucination scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "User question or prompt"},
                    "generated_answer": {"type": "string", "description": "Candidate generated LLM answer"},
                    "context_chunks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Retrieved context passages or filings",
                    },
                    "ticker": {"type": "string", "description": "Optional asset or stock ticker (e.g. AAPL)"},
                    "mode": {
                        "type": "string",
                        "enum": ["FAST", "BALANCED", "THOROUGH", "STRICT"],
                        "default": "THOROUGH",
                        "description": "Evaluation rigor mode",
                    },
                    "numerical_targets": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional target values with metric_name, expected_value, tolerance_pct",
                    },
                },
                "required": ["query_text", "generated_answer", "context_chunks"],
            },
            handler=handle_rag_evaluate,
        )
    )

    reg.register(
        MCPTool(
            name="rag_detect_hallucinations",
            description="Audit text against financial ground truth chunks for numerical distortions and contradictions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "generated_answer": {"type": "string", "description": "Generated text under audit"},
                    "context_chunks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Retrieved document text excerpts",
                    },
                    "numerical_tolerance_pct": {
                        "type": "number",
                        "default": 0.01,
                        "description": "Allowed numerical delta percentage",
                    },
                },
                "required": ["generated_answer", "context_chunks"],
            },
            handler=handle_rag_detect_hallucinations,
        )
    )

    reg.register(
        MCPTool(
            name="rag_risk_metrics",
            description="Calculate quantitative risk metrics (Sharpe, Sortino, VaR, CVaR, Max Drawdown) from return series.",
            inputSchema={
                "type": "object",
                "properties": {
                    "returns": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Sequence of period returns (e.g. daily decimals)",
                    },
                    "benchmark_returns": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional benchmark return series",
                    },
                    "risk_free_rate": {
                        "type": "number",
                        "default": 0.045,
                        "description": "Annualized risk-free discount rate",
                    },
                    "frequency": {
                        "type": "string",
                        "enum": ["DAILY", "WEEKLY", "MONTHLY", "ANNUAL"],
                        "default": "DAILY",
                        "description": "Observation interval frequency",
                    },
                },
                "required": ["returns"],
            },
            handler=handle_rag_risk_metrics,
        )
    )

    reg.register(
        MCPTool(
            name="rag_stress_test",
            description="Run Monte Carlo simulation stress tests across historical or synthetic market shock scenarios.",
            inputSchema={
                "type": "object",
                "properties": {
                    "returns": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Baseline returns distribution",
                    },
                    "num_simulations": {
                        "type": "integer",
                        "default": 1000,
                        "description": "Number of stochastic paths",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "default": 30,
                        "description": "Simulation forward step count",
                    },
                    "scenario": {
                        "type": "string",
                        "enum": [
                            "BASELINE",
                            "VOLATILITY_SPIKE",
                            "LIQUIDITY_CRUNCH",
                            "RATE_SHOCK",
                            "BLACK_SWAN",
                            "HALLUCINATION_CASCADE",
                        ],
                        "default": "BASELINE",
                        "description": "Stress scenario profile",
                    },
                    "confidence_interval": {
                        "type": "number",
                        "default": 0.95,
                        "description": "Confidence level for tail risk metrics",
                    },
                },
                "required": ["returns"],
            },
            handler=handle_rag_stress_test,
        )
    )

    reg.register(
        MCPTool(
            name="rag_feature_stats",
            description="Query streaming feature registry definitions and Parquet feature store storage summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "feature_group": {
                        "type": "string",
                        "default": "ALL",
                        "description": "Feature group filter (QUERY, CHUNK, RESPONSE, EVALUATION, FINANCIAL, ALL)",
                    },
                    "store_path": {
                        "type": "string",
                        "description": "Optional custom Parquet directory path",
                    },
                },
            },
            handler=handle_rag_feature_stats,
        )
    )

    reg.register(
        MCPTool(
            name="rag_cache_lookup",
            description="Query, clear, or inspect statistics of the semantic evaluation cache.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "Query text to evaluate against cache",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["GET", "STATS", "CLEAR"],
                        "default": "GET",
                        "description": "Cache operation to execute",
                    },
                },
            },
            handler=handle_rag_cache_lookup,
        )
    )

    return reg
