"""Specialized agents for financial RAG evaluation, auditing, risk analysis, and synthesis."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from production_rag_evaluation.agents.base import BaseAgent
from production_rag_evaluation.agents.models import (
    AgentRole,
    AgentStatus,
    StepExecutionMode,
    WorkflowPlan,
    WorkflowStep,
)
from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    EvaluationMetrics,
    EvaluationMode,
    RAGQuery,
    RAGResponse,
    RiskMetrics,
    Severity,
)


class PlannerAgent(BaseAgent):
    """Planner agent decomposing RAG evaluation requirements into execution DAG."""

    def __init__(self, name: str = "planner-agent") -> None:
        super().__init__(name=name, role=AgentRole.PLANNER)

    def plan(
        self,
        query: RAGQuery,
        response: RAGResponse,
        returns: Optional[np.ndarray] = None,
        evaluation_mode: EvaluationMode = EvaluationMode.THOROUGH,
    ) -> WorkflowPlan:
        """Construct structured workflow plan with dependency DAG."""
        steps: List[WorkflowStep] = []

        # Step 1: Auditor
        step_audit = WorkflowStep(
            name="hallucination_audit",
            agent_role=AgentRole.AUDITOR,
            mode=StepExecutionMode.PARALLEL,
            inputs={"mode": evaluation_mode.value},
        )
        steps.append(step_audit)

        # Step 2: Evaluator
        step_eval = WorkflowStep(
            name="core_rag_evaluation",
            agent_role=AgentRole.EVALUATOR,
            mode=StepExecutionMode.PARALLEL,
            inputs={"mode": evaluation_mode.value},
        )
        steps.append(step_eval)

        # Step 3: Risk Analyst (can run after auditor and evaluator)
        step_risk = WorkflowStep(
            name="financial_risk_analysis",
            agent_role=AgentRole.RISK_ANALYST,
            mode=StepExecutionMode.SEQUENTIAL,
            dependencies=[step_audit.step_id, step_eval.step_id],
            inputs={"has_custom_returns": returns is not None},
        )
        steps.append(step_risk)

        # Step 4: Synthesizer (depends on all upstream steps)
        step_synth = WorkflowStep(
            name="synthesis_and_reporting",
            agent_role=AgentRole.SYNTHESIZER,
            mode=StepExecutionMode.SEQUENTIAL,
            dependencies=[step_audit.step_id, step_eval.step_id, step_risk.step_id],
            inputs={},
        )
        steps.append(step_synth)

        plan = WorkflowPlan(
            query_id=query.query_id,
            ticker=query.ticker,
            steps=steps,
            estimated_duration_ms=150.0,
        )
        return plan

    def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planning step."""
        self.status = AgentStatus.BUSY
        query: RAGQuery = context["query"]
        response: RAGResponse = context["response"]
        returns: Optional[np.ndarray] = context.get("returns")
        plan = self.plan(query, response, returns)
        self.status = AgentStatus.COMPLETED
        return {"plan_id": plan.plan_id, "step_count": len(plan.steps)}


class EvaluatorAgent(BaseAgent):
    """Evaluator agent computing faithfulness, answer relevance, and context recall."""

    def __init__(
        self,
        name: str = "evaluator-agent",
        evaluator: Optional[RAGEvaluator] = None,
    ) -> None:
        super().__init__(name=name, role=AgentRole.EVALUATOR)
        self.evaluator = evaluator or RAGEvaluator()

    def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute core RAG evaluation metrics."""
        self.status = AgentStatus.BUSY
        t0 = time.perf_counter()
        query: RAGQuery = context["query"]
        response: RAGResponse = context["response"]

        metrics: EvaluationMetrics = self.evaluator.evaluate(query, response)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.status = AgentStatus.COMPLETED
        return {
            "metrics": metrics,
            "faithfulness_score": metrics.faithfulness_score,
            "answer_relevance_score": metrics.answer_relevance_score,
            "context_recall_score": metrics.context_recall_score,
            "overall_pass": metrics.overall_pass,
            "execution_time_ms": elapsed_ms,
        }


class AuditorAgent(BaseAgent):
    """Auditor agent specialized in detecting financial numerical distortions and contradictions."""

    def __init__(
        self,
        name: str = "auditor-agent",
        detector: Optional[HallucinationDetector] = None,
    ) -> None:
        super().__init__(name=name, role=AgentRole.AUDITOR)
        self.detector = detector or HallucinationDetector()

    def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Audit response against context for financial hallucinations."""
        self.status = AgentStatus.BUSY
        t0 = time.perf_counter()
        response: RAGResponse = context["response"]
        chunks: List[DocumentChunk] = response.retrieved_chunks

        # Numerical distortions
        num_distortions = self.detector.detect_numerical_distortions(
            response.generated_answer, chunks
        )
        # Contradictions
        contradictions = self.detector.detect_directional_contradictions(
            response.generated_answer, chunks
        )

        all_findings = num_distortions + contradictions
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        severity_counts = {
            Severity.LOW.value: 0,
            Severity.MEDIUM.value: 0,
            Severity.HIGH.value: 0,
            Severity.CRITICAL.value: 0,
        }
        for f in all_findings:
            severity_counts[f.severity.value] += 1

        self.status = AgentStatus.COMPLETED
        return {
            "findings": all_findings,
            "finding_count": len(all_findings),
            "severity_counts": severity_counts,
            "has_critical_findings": severity_counts[Severity.CRITICAL.value] > 0,
            "execution_time_ms": elapsed_ms,
        }


class RiskAnalystAgent(BaseAgent):
    """Risk analyst agent calculating financial portfolio risk ratios and stress tests."""

    def __init__(
        self,
        name: str = "risk-analyst-agent",
        risk_engine: Optional[FinancialRiskEngine] = None,
        stress_tester: Optional[MonteCarloStressTester] = None,
    ) -> None:
        super().__init__(name=name, role=AgentRole.RISK_ANALYST)
        self.risk_engine = risk_engine or FinancialRiskEngine()
        self.stress_tester = stress_tester or MonteCarloStressTester(random_seed=42)

    def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute portfolio risk metrics and stress simulations."""
        self.status = AgentStatus.BUSY
        t0 = time.perf_counter()
        returns: Optional[np.ndarray] = context.get("returns")
        if returns is None or len(returns) < 5:
            # Generate deterministic fallback returns
            rng = np.random.RandomState(42)
            returns = rng.normal(0.0008, 0.015, 60)

        risk_metrics: RiskMetrics = self.risk_engine.compute_all_metrics(returns)

        # If auditor findings exist, run hallucination stress test
        auditor_results = context.get("step_results", {}).get("hallucination_audit", {})
        finding_count = auditor_results.get("finding_count", 0)
        hallucination_prob = min(0.50, 0.05 * finding_count)

        stress_result = self.stress_tester.run_stress_test(
            returns, n_simulations=100, horizon_days=20
        )
        breach_rate = self.stress_tester.stress_test_hallucination_impact(
            base_financial_estimates=[100.0, 105.0, 110.0],
            hallucination_probability=hallucination_prob,
            n_simulations=50,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.status = AgentStatus.COMPLETED
        return {
            "risk_metrics": risk_metrics,
            "sharpe_ratio": risk_metrics.sharpe_ratio,
            "sortino_ratio": risk_metrics.sortino_ratio,
            "max_drawdown": risk_metrics.max_drawdown,
            "var_95": risk_metrics.var_95,
            "worst_case_drawdown": stress_result.worst_case_drawdown,
            "hallucination_breach_rate": breach_rate,
            "execution_time_ms": elapsed_ms,
        }


class SynthesizerAgent(BaseAgent):
    """Synthesizer agent compiling multi-agent outputs into executive decisions and summaries."""

    def __init__(self, name: str = "synthesizer-agent") -> None:
        super().__init__(name=name, role=AgentRole.SYNTHESIZER)

    def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize evaluations into cohesive report and recommendations."""
        self.status = AgentStatus.BUSY
        t0 = time.perf_counter()
        step_results = context.get("step_results", {})

        eval_res = step_results.get("core_rag_evaluation", {})
        audit_res = step_results.get("hallucination_audit", {})
        risk_res = step_results.get("financial_risk_analysis", {})

        faithfulness = eval_res.get("faithfulness_score", 1.0)
        overall_pass = eval_res.get("overall_pass", True)
        finding_count = audit_res.get("finding_count", 0)
        has_critical = audit_res.get("has_critical_findings", False)
        sharpe = risk_res.get("sharpe_ratio", 0.0)

        # Build summary narrative
        status_str = "APPROVED" if (overall_pass and not has_critical) else "FLAGGED"
        summary = (
            f"Financial RAG Evaluation Status: {status_str}. "
            f"Faithfulness Score: {faithfulness:.2%}. "
            f"Hallucination Findings: {finding_count} detected. "
            f"Portfolio Sharpe: {sharpe:.2f}."
        )

        # Build recommendations
        recommendations: List[str] = []
        if not overall_pass:
            recommendations.append("Reject response due to sub-threshold faithfulness.")
        if has_critical:
            recommendations.append("Critical financial distortion detected; manual compliance review required.")
        if finding_count == 0 and overall_pass:
            recommendations.append("Response validated with high confidence; cleared for production dispatch.")
        if not recommendations:
            recommendations.append("Response passed with minor warnings; monitor drift in feature store.")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.status = AgentStatus.COMPLETED
        return {
            "status": status_str,
            "summary": summary,
            "recommendations": recommendations,
            "execution_time_ms": elapsed_ms,
        }
