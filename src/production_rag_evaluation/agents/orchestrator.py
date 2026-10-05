"""Multi-agent swarm orchestrator and DAG workflow execution engine."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np

from production_rag_evaluation.agents.base import (
    AgentRegistry,
    BaseAgent,
    get_default_agent_registry,
)
from production_rag_evaluation.agents.models import (
    AgentRole,
    AgentWorkflowConfig,
    WorkflowExecutionResult,
    WorkflowPlan,
    WorkflowStatus,
    WorkflowStep,
)
from production_rag_evaluation.agents.specialized import (
    AuditorAgent,
    EvaluatorAgent,
    PlannerAgent,
    RiskAnalystAgent,
    SynthesizerAgent,
)
from production_rag_evaluation.models.schemas import (
    EvaluationMetrics,
    EvaluationMode,
    HallucinationFinding,
    RAGQuery,
    RAGResponse,
    RiskMetrics,
)


class AgentSwarmOrchestrator:
    """Orchestrator coordinating multi-agent swarm workflow execution."""

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        config: Optional[AgentWorkflowConfig] = None,
    ) -> None:
        self.registry = registry or get_default_agent_registry()
        self.config = config or AgentWorkflowConfig()
        self._ensure_default_agents()

    def _ensure_default_agents(self) -> None:
        """Register default specialized agents if not present in registry."""
        role_map = {
            AgentRole.PLANNER: PlannerAgent,
            AgentRole.EVALUATOR: EvaluatorAgent,
            AgentRole.AUDITOR: AuditorAgent,
            AgentRole.RISK_ANALYST: RiskAnalystAgent,
            AgentRole.SYNTHESIZER: SynthesizerAgent,
        }
        for role, agent_cls in role_map.items():
            if not self.registry.get_by_role(role):
                self.registry.register(agent_cls())

    def create_plan(
        self,
        query: RAGQuery,
        response: RAGResponse,
        returns: Optional[np.ndarray] = None,
        evaluation_mode: EvaluationMode = EvaluationMode.THOROUGH,
    ) -> WorkflowPlan:
        """Generate structured execution plan using the planner agent."""
        planners = self.registry.get_by_role(AgentRole.PLANNER)
        if not planners:
            planner = PlannerAgent()
            self.registry.register(planner)
        else:
            planner = planners[0]

        if isinstance(planner, PlannerAgent):
            return planner.plan(query, response, returns, evaluation_mode)

        # Fallback manual plan creation
        fallback_planner = PlannerAgent()
        return fallback_planner.plan(query, response, returns, evaluation_mode)

    def _build_execution_order(self, plan: WorkflowPlan) -> List[WorkflowStep]:
        """Resolve step dependencies using DAG topological sort."""
        graph = nx.DiGraph()
        step_by_id = {s.step_id: s for s in plan.steps}

        for step in plan.steps:
            graph.add_node(step.step_id)
            for dep_id in step.dependencies:
                if dep_id in step_by_id:
                    graph.add_edge(dep_id, step.step_id)

        try:
            sorted_step_ids = list(nx.topological_sort(graph))
            return [step_by_id[s_id] for s_id in sorted_step_ids if s_id in step_by_id]
        except nx.NetworkXUnfeasible:
            # Cycle detected; fallback to linear declaration order
            return plan.steps

    def execute_workflow(
        self,
        query: RAGQuery,
        response: RAGResponse,
        returns: Optional[np.ndarray] = None,
        plan: Optional[WorkflowPlan] = None,
    ) -> WorkflowExecutionResult:
        """Execute full multi-agent evaluation workflow end-to-end."""
        t0 = time.perf_counter()
        workflow_id = str(uuid.uuid4())

        if plan is None:
            plan = self.create_plan(query, response, returns)

        execution_order = self._build_execution_order(plan)
        step_results: Dict[str, Dict[str, Any]] = {}
        completed_step_ids: Set[str] = set()

        context: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "query": query,
            "response": response,
            "returns": returns,
            "step_results": step_results,
            "config": self.config,
        }

        overall_status = WorkflowStatus.COMPLETED
        workflow_error: Optional[str] = None

        for step in execution_order:
            # Check dependencies
            missing_deps = [d for d in step.dependencies if d not in completed_step_ids]
            if missing_deps:
                step.status = WorkflowStatus.SKIPPED
                step.error = f"Dependencies unmet: {missing_deps}"
                continue

            agents = self.registry.get_by_role(step.agent_role)
            if not agents:
                step.status = WorkflowStatus.FAILED
                step.error = f"No agent registered for role: {step.agent_role.value}"
                overall_status = WorkflowStatus.FAILED
                workflow_error = step.error
                break

            agent: BaseAgent = agents[0]
            step.status = WorkflowStatus.RUNNING
            step_t0 = time.perf_counter()

            try:
                out = agent.execute(step, context)
                step.outputs = {k: v for k, v in out.items() if not isinstance(v, (np.ndarray,))}
                step.execution_time_ms = (time.perf_counter() - step_t0) * 1000.0
                step.status = WorkflowStatus.COMPLETED

                step_results[step.name] = out
                completed_step_ids.add(step.step_id)

                # Halt condition check
                if self.config.halt_on_high_severity_hallucination and out.get(
                    "has_critical_findings"
                ):
                    overall_status = WorkflowStatus.FAILED
                    workflow_error = "Halting workflow: Critical hallucination detected."
                    break

            except Exception as exc:
                step.status = WorkflowStatus.FAILED
                step.error = str(exc)
                step.execution_time_ms = (time.perf_counter() - step_t0) * 1000.0
                overall_status = WorkflowStatus.FAILED
                workflow_error = f"Step '{step.name}' failed: {exc}"
                break

        total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Consolidate results
        metrics: Optional[EvaluationMetrics] = None
        findings: List[HallucinationFinding] = []
        risk_metrics: Optional[RiskMetrics] = None
        summary: str = ""
        recommendations: List[str] = []

        if "core_rag_evaluation" in step_results:
            metrics = step_results["core_rag_evaluation"].get("metrics")
        if "hallucination_audit" in step_results:
            findings = step_results["hallucination_audit"].get("findings", [])
        if "financial_risk_analysis" in step_results:
            risk_metrics = step_results["financial_risk_analysis"].get("risk_metrics")
        if "synthesis_and_reporting" in step_results:
            summary = step_results["synthesis_and_reporting"].get("summary", "")
            recommendations = step_results["synthesis_and_reporting"].get("recommendations", [])

        return WorkflowExecutionResult(
            workflow_id=workflow_id,
            status=overall_status,
            plan=plan,
            step_results=step_results,
            metrics=metrics,
            findings=findings,
            risk_metrics=risk_metrics,
            summary=summary,
            recommendations=recommendations,
            total_execution_time_ms=total_elapsed_ms,
            error=workflow_error,
        )


def create_default_orchestrator(
    config: Optional[AgentWorkflowConfig] = None,
) -> AgentSwarmOrchestrator:
    """Factory creating an orchestrator loaded with default swarm agents."""
    registry = AgentRegistry()
    registry.register(PlannerAgent())
    registry.register(EvaluatorAgent())
    registry.register(AuditorAgent())
    registry.register(RiskAnalystAgent())
    registry.register(SynthesizerAgent())
    return AgentSwarmOrchestrator(registry=registry, config=config)
