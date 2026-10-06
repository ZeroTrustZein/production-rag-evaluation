"""Multi-agent swarm orchestrator and DAG workflow execution engine."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from production_rag_evaluation.agents.base import (
    AgentRegistry,
    BaseAgent,
    get_default_agent_registry,
)
from production_rag_evaluation.agents.models import (
    AgentMessage,
    AgentRole,
    AgentWorkflowConfig,
    StepExecutionMode,
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
        event_dispatcher: Optional[Any] = None,
        message_broker: Optional[Any] = None,
    ) -> None:
        self.registry = registry or get_default_agent_registry()
        self.config = config or AgentWorkflowConfig()
        self.event_dispatcher = event_dispatcher
        self.message_broker = message_broker
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
        planner: BaseAgent
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

    def _build_execution_waves(self, plan: WorkflowPlan) -> List[List[WorkflowStep]]:
        """Group plan steps into topological dependency generations for concurrent execution."""
        graph = nx.DiGraph()
        step_by_id = {s.step_id: s for s in plan.steps}

        for step in plan.steps:
            graph.add_node(step.step_id)
            for dep_id in step.dependencies:
                if dep_id in step_by_id:
                    graph.add_edge(dep_id, step.step_id)

        try:
            generations = list(nx.topological_generations(graph))
            waves: List[List[WorkflowStep]] = []
            for gen in generations:
                wave_steps = [step_by_id[s_id] for s_id in gen if s_id in step_by_id]
                if wave_steps:
                    waves.append(wave_steps)
            return waves
        except nx.NetworkXUnfeasible:
            # Cycle detected; fall back to each step sequentially
            return [[s] for s in plan.steps]

    def _execute_step_with_retry(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
    ) -> Tuple[WorkflowStep, Optional[Dict[str, Any]], Optional[Exception]]:
        """Execute a single workflow step with optional auto-retry support."""
        agents = self.registry.get_by_role(step.agent_role)
        if not agents:
            step.status = WorkflowStatus.FAILED
            step.error = f"No agent registered for role: {step.agent_role.value}"
            return step, None, RuntimeError(step.error)

        agent: BaseAgent = agents[0]
        max_attempts = (1 + self.config.max_retries) if self.config.auto_retry_failed_steps else 1
        step_t0 = time.perf_counter()
        last_exc: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                step.status = WorkflowStatus.RUNNING
                out = agent.execute(step, context)
                step.outputs = {k: v for k, v in out.items() if not isinstance(v, (np.ndarray,))}
                step.execution_time_ms = (time.perf_counter() - step_t0) * 1000.0
                step.status = WorkflowStatus.COMPLETED
                return step, out, None
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    continue
                step.status = WorkflowStatus.FAILED
                step.error = str(exc)
                step.execution_time_ms = (time.perf_counter() - step_t0) * 1000.0
                return step, None, last_exc

        return step, None, last_exc

    def route_messages(self, messages: List[AgentMessage]) -> None:
        """Route inter-agent messages to recipient agent inboxes."""
        if self.message_broker is not None:
            self.message_broker.route_batch(messages, self.registry)
        else:
            for msg in messages:
                target_agent = self.registry.get(msg.receiver)
                if target_agent is not None:
                    target_agent.receive_message(msg)

    def execute_workflow(
        self,
        query: RAGQuery,
        response: RAGResponse,
        returns: Optional[np.ndarray] = None,
        plan: Optional[WorkflowPlan] = None,
    ) -> WorkflowExecutionResult:
        """Execute full multi-agent evaluation workflow end-to-end using topological waves."""
        t0 = time.perf_counter()
        workflow_id = str(uuid.uuid4())

        if plan is None:
            plan = self.create_plan(query, response, returns)

        # Validate DAG structure
        is_dag_valid, dag_err = plan.validate_dag()
        if not is_dag_valid:
            return WorkflowExecutionResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                plan=plan,
                step_results={},
                error=f"Invalid workflow DAG: {dag_err}",
                total_execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        execution_waves = self._build_execution_waves(plan)
        step_results: Dict[str, Dict[str, Any]] = {}
        completed_step_ids: Set[str] = set()
        messages: List[AgentMessage] = []

        context: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "query": query,
            "response": response,
            "returns": returns,
            "step_results": step_results,
            "config": self.config,
            "messages": messages,
            "registry": self.registry,
        }

        overall_status = WorkflowStatus.COMPLETED
        workflow_error: Optional[str] = None

        for wave in execution_waves:
            # Check timeout
            elapsed_sec = time.perf_counter() - t0
            if elapsed_sec > self.config.timeout_seconds:
                overall_status = WorkflowStatus.FAILED
                workflow_error = f"Workflow timed out after {self.config.timeout_seconds:.1f}s"
                for rem_step in wave:
                    rem_step.status = WorkflowStatus.CANCELLED
                break

            runnable_steps: List[WorkflowStep] = []
            for step in wave:
                # Check dependencies
                missing_deps = [d for d in step.dependencies if d not in completed_step_ids]
                if missing_deps:
                    step.status = WorkflowStatus.SKIPPED
                    step.error = f"Dependencies unmet: {missing_deps}"
                    continue

                # Check conditional mode
                if step.mode == StepExecutionMode.CONDITIONAL:
                    condition_val = step.inputs.get("condition")
                    if condition_val is False or (
                        callable(condition_val) and not condition_val(context)
                    ):
                        step.status = WorkflowStatus.SKIPPED
                        step.outputs = {"skipped_reason": "Conditional check evaluated to false"}
                        continue

                runnable_steps.append(step)

            if not runnable_steps:
                continue

            # Execute wave steps: parallel if concurrency allowed and multiple steps, else sequential
            wave_failed = False
            if len(runnable_steps) > 1 and self.config.concurrency_limit > 1:
                workers = min(len(runnable_steps), self.config.concurrency_limit)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(self._execute_step_with_retry, s, context)
                        for s in runnable_steps
                    ]
                    for fut in futures:
                        s, out, exc = fut.result()
                        if exc is not None or s.status == WorkflowStatus.FAILED:
                            wave_failed = True
                            overall_status = WorkflowStatus.FAILED
                            workflow_error = s.error or str(exc)
                        elif out is not None:
                            step_results[s.name] = out
                            completed_step_ids.add(s.step_id)
            else:
                for s in runnable_steps:
                    s, out, exc = self._execute_step_with_retry(s, context)
                    if exc is not None or s.status == WorkflowStatus.FAILED:
                        wave_failed = True
                        overall_status = WorkflowStatus.FAILED
                        workflow_error = s.error or str(exc)
                        break
                    elif out is not None:
                        step_results[s.name] = out
                        completed_step_ids.add(s.step_id)

            # Route any emitted messages to recipient agents
            if messages:
                self.route_messages(messages)

            if wave_failed:
                break

            # Halt condition check
            if self.config.halt_on_high_severity_hallucination:
                has_critical = any(
                    out.get("has_critical_findings", False) for out in step_results.values()
                )
                if has_critical:
                    overall_status = WorkflowStatus.FAILED
                    workflow_error = "Halting workflow: Critical hallucination detected."
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
    registry: Optional[AgentRegistry] = None,
    event_dispatcher: Optional[Any] = None,
    message_broker: Optional[Any] = None,
) -> AgentSwarmOrchestrator:
    """Factory creating an orchestrator loaded with default swarm agents."""
    if registry is None:
        registry = AgentRegistry()
        registry.register(PlannerAgent())
        registry.register(EvaluatorAgent())
        registry.register(AuditorAgent())
        registry.register(RiskAnalystAgent())
        registry.register(SynthesizerAgent())
    return AgentSwarmOrchestrator(
        registry=registry,
        config=config,
        event_dispatcher=event_dispatcher,
        message_broker=message_broker,
    )
