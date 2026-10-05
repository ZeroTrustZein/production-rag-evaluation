"""Integration handlers, event dispatching, message brokering, and telemetry bridges for agent workflows."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from production_rag_evaluation.agents.base import AgentRegistry
from production_rag_evaluation.agents.models import (
    AgentMessage,
    AgentRole,
    WorkflowExecutionResult,
    WorkflowPlan,
    WorkflowStatus,
)
from production_rag_evaluation.agents.orchestrator import (
    AgentSwarmOrchestrator,
    create_default_orchestrator,
)
from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.features.models import FeatureGroup, FeatureRecord
from production_rag_evaluation.features.store import ParquetFeatureStore
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.models.schemas import (
    AlertSeverity,
    EvaluationMode,
    RAGQuery,
    RAGResponse,
    TelemetryAlert,
)

logger = logging.getLogger(__name__)


class WorkflowEventDispatcher:
    """Publish-subscribe event dispatcher coordinating agent workflow lifecycle events."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = {}

    def subscribe(self, event: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register listener callback for an event or wildcard '*'."""
        if event not in self._listeners:
            self._listeners[event] = []
        if callback not in self._listeners[event]:
            self._listeners[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable[[str, Dict[str, Any]], None]) -> bool:
        """Unregister callback from event."""
        if event in self._listeners and callback in self._listeners[event]:
            self._listeners[event].remove(callback)
            return True
        return False

    def publish(self, event: str, payload: Dict[str, Any]) -> int:
        """Dispatch event payload to exact and wildcard listeners. Returns invoked subscriber count."""
        subscribers: List[Callable[[str, Dict[str, Any]], None]] = []

        if event in self._listeners:
            subscribers.extend(self._listeners[event])
        if "*" in self._listeners and event != "*":
            subscribers.extend(self._listeners["*"])

        delivered = 0
        for sub in subscribers:
            try:
                sub(event, payload)
                delivered += 1
            except Exception as exc:
                logger.error("Error invoking event listener for %s: %s", event, exc)

        return delivered

    def get_subscribers_count(self, event: str) -> int:
        """Count active subscribers registered for an event."""
        return len(self._listeners.get(event, []))

    def clear(self) -> None:
        """Clear all registered event subscribers."""
        self._listeners.clear()


class AgentMessageBroker:
    """Message broker handling inter-agent communication, direct routing, broadcast, and message audit trails."""

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self._history: List[AgentMessage] = []

    def route_message(self, message: AgentMessage, registry: AgentRegistry) -> bool:
        """Route message to recipient inbox or broadcast across swarm. Buffers in audit history."""
        self._history.append(message)
        if len(self._history) > self.max_history:
            self._history.pop(0)

        if message.receiver == "*":
            # Broadcast to all agents except sender
            active_agents = registry.get_active_agents()
            for agent in active_agents:
                if agent.name != message.sender:
                    agent.receive_message(message)
            return True

        target_agent = registry.get(message.receiver)
        if target_agent is not None:
            target_agent.receive_message(message)
            return True

        return False

    def route_batch(self, messages: List[AgentMessage], registry: AgentRegistry) -> int:
        """Route list of messages. Returns count of successfully routed messages."""
        routed = 0
        for msg in messages:
            if self.route_message(msg, registry):
                routed += 1
        return routed

    def get_history(
        self,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
        role: Optional[AgentRole] = None,
    ) -> List[AgentMessage]:
        """Fetch filtered historical message exchange records."""
        results = self._history
        if sender is not None:
            results = [m for m in results if m.sender == sender]
        if receiver is not None:
            results = [m for m in results if m.receiver == receiver]
        if role is not None:
            results = [m for m in results if m.role == role]
        return list(results)

    def clear_history(self) -> None:
        """Clear recorded message history."""
        self._history.clear()

    @property
    def total_messages(self) -> int:
        """Return total count of messages in audit trail."""
        return len(self._history)


class WorkflowTelemetryBridge:
    """Bridge linking agent workflow execution results to telemetry collectors and alerts."""

    def __init__(self, collector: Optional[RiskTelemetryCollector] = None) -> None:
        self.collector = collector or RiskTelemetryCollector()

    def record_workflow_result(
        self,
        result: WorkflowExecutionResult,
        query: Optional[RAGQuery] = None,
    ) -> List[TelemetryAlert]:
        """Convert workflow execution findings and metrics into telemetry metrics and alerts."""
        alerts: List[TelemetryAlert] = []

        if result.status == WorkflowStatus.FAILED:
            alert = TelemetryAlert(
                metric_name="agent_workflow_failure",
                current_value=1.0,
                threshold=0.0,
                severity=AlertSeverity.CRITICAL,
                message=f"Agent workflow {result.workflow_id} failed: {result.error}",
            )
            self.collector.record_alert(alert)
            alerts.append(alert)

        # Check critical hallucination findings
        if result.findings:
            critical_findings = [f for f in result.findings if f.severity.value in ("high", "critical")]
            if critical_findings:
                alert = TelemetryAlert(
                    metric_name="agent_hallucination_critical",
                    current_value=float(len(critical_findings)),
                    threshold=0.0,
                    severity=AlertSeverity.HIGH,
                    message=f"Workflow {result.workflow_id} detected {len(critical_findings)} high/critical hallucinations.",
                )
                self.collector.record_alert(alert)
                alerts.append(alert)

        # Check risk metrics drawdown
        if result.risk_metrics and result.risk_metrics.max_drawdown > 0.20:
            alert = TelemetryAlert(
                metric_name="agent_risk_max_drawdown",
                current_value=float(result.risk_metrics.max_drawdown),
                threshold=0.20,
                severity=AlertSeverity.MEDIUM,
                message=f"Workflow {result.workflow_id} calculated high Max Drawdown: {result.risk_metrics.max_drawdown:.2%}",
            )
            self.collector.record_alert(alert)
            alerts.append(alert)

        return alerts


class AgentWorkflowIntegrationHandler:
    """High-level facade executing multi-agent workflows with caching, telemetry, and feature persistence."""

    def __init__(
        self,
        orchestrator: Optional[AgentSwarmOrchestrator] = None,
        feature_store: Optional[ParquetFeatureStore] = None,
        telemetry_collector: Optional[RiskTelemetryCollector] = None,
        semantic_cache: Optional[SemanticCache] = None,
        event_dispatcher: Optional[WorkflowEventDispatcher] = None,
        message_broker: Optional[AgentMessageBroker] = None,
    ) -> None:
        self.orchestrator = orchestrator or create_default_orchestrator()
        self.feature_store = feature_store
        self.telemetry_bridge = WorkflowTelemetryBridge(collector=telemetry_collector)
        self.semantic_cache = semantic_cache
        self.event_dispatcher = event_dispatcher or WorkflowEventDispatcher()
        self.message_broker = message_broker or AgentMessageBroker()

        self._execution_count: int = 0
        self._error_count: int = 0
        self._total_latency_ms: float = 0.0

    def execute_and_record(
        self,
        query: RAGQuery,
        response: RAGResponse,
        returns: Optional[np.ndarray] = None,
        plan: Optional[WorkflowPlan] = None,
        use_cache: bool = True,
    ) -> WorkflowExecutionResult:
        """Execute multi-agent workflow, check/store cache, record features, and publish events."""
        t0 = time.perf_counter()

        self.event_dispatcher.publish("workflow_started", {
            "query_id": query.query_id,
            "ticker": query.ticker,
        })

        # Cache check
        if use_cache and self.semantic_cache is not None:
            cached_result = self.semantic_cache.get(query.query_text)
            if cached_result is not None:
                self.event_dispatcher.publish("workflow_cache_hit", {
                    "query_id": query.query_id,
                })

        # Execute workflow
        result = self.orchestrator.execute_workflow(
            query=query,
            response=response,
            returns=returns,
            plan=plan,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._execution_count += 1
        self._total_latency_ms += elapsed_ms

        if result.status == WorkflowStatus.FAILED:
            self._error_count += 1
            self.event_dispatcher.publish("workflow_failed", {
                "workflow_id": result.workflow_id,
                "error": result.error,
            })
        else:
            self.event_dispatcher.publish("workflow_completed", {
                "workflow_id": result.workflow_id,
                "status": result.status.value,
                "steps": len(result.step_results),
            })

        # Telemetry alerts
        self.telemetry_bridge.record_workflow_result(result, query=query)

        # Feature store ingestion if attached
        if self.feature_store is not None:
            synth_res = result.step_results.get("synthesis_and_reporting", {})
            comp_score = synth_res.get("composite_health_score", 0.0)

            rec = FeatureRecord(
                entity_id=query.query_id,
                feature_group=FeatureGroup.EVALUATION,
                features={
                    "workflow_id": result.workflow_id,
                    "composite_health_score": float(comp_score),
                    "total_execution_time_ms": float(result.total_execution_time_ms),
                    "finding_count": len(result.findings),
                },
            )
            self.feature_store.write_records([rec])

        # Cache put
        if use_cache and self.semantic_cache is not None and result.metrics is not None:
            self.semantic_cache.put(query.query_text, result.metrics)

        return result

    def handle_batch_workflow(
        self,
        queries_and_responses: List[tuple[RAGQuery, RAGResponse]],
        evaluation_mode: EvaluationMode = EvaluationMode.FAST,
    ) -> List[WorkflowExecutionResult]:
        """Execute workflows for a batch of query-response pairs."""
        results: List[WorkflowExecutionResult] = []
        for q, resp in queries_and_responses:
            plan = self.orchestrator.create_plan(q, resp, evaluation_mode=evaluation_mode)
            res = self.execute_and_record(q, resp, plan=plan)
            results.append(res)
        return results

    def get_execution_stats(self) -> Dict[str, Any]:
        """Return operational execution statistics."""
        avg_lat = (
            self._total_latency_ms / self._execution_count
            if self._execution_count > 0
            else 0.0
        )
        return {
            "total_executions": self._execution_count,
            "total_errors": self._error_count,
            "average_latency_ms": round(avg_lat, 2),
            "cached_items": len(self.semantic_cache) if self.semantic_cache else 0,
            "message_broker_messages": self.message_broker.total_messages,
        }

    def reset_stats(self) -> None:
        """Reset operational statistics."""
        self._execution_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
