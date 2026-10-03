"""Supporting modules, session management, and integration handlers for MCP."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from production_rag_evaluation.features.models import FeatureGroup, FeatureRecord
from production_rag_evaluation.features.store import ParquetFeatureStore
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.mcp.models import (
    JSONRPCError,
    JSONRPCResponse,
    MCPSession,
    ToolCallResult,
    ToolExecutionStats,
)
from production_rag_evaluation.mcp.tools import MCPToolRegistry, get_default_tool_registry
from production_rag_evaluation.models.schemas import EvaluationMetrics

logger = logging.getLogger(__name__)


class MCPSessionHandler:
    """Manages active MCP client sessions, heartbeat activity, and authentication."""

    def __init__(
        self,
        auth_token: Optional[str] = None,
        session_ttl_seconds: float = 3600.0,
    ) -> None:
        self._auth_token = auth_token
        self._session_ttl_seconds = session_ttl_seconds
        self._sessions: Dict[str, MCPSession] = {}

    def create_session(
        self,
        client_info: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None,
    ) -> MCPSession:
        """Create and track an active client session with optional token validation."""
        is_auth = True
        if self._auth_token is not None:
            if auth_token != self._auth_token:
                is_auth = False

        session_id = str(uuid.uuid4())
        session = MCPSession(
            session_id=session_id,
            client_info=client_info or {},
            is_authenticated=is_auth,
            created_at=time.time(),
            last_active_at=time.time(),
            request_count=0,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[MCPSession]:
        """Fetch session metadata by session ID."""
        return self._sessions.get(session_id)

    def validate_session(self, session_id: str) -> bool:
        """Verify whether session exists, is authenticated, and has not expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        if not session.is_authenticated:
            return False
        if (time.time() - session.last_active_at) > self._session_ttl_seconds:
            return False
        return True

    def touch_session(self, session_id: str) -> bool:
        """Record activity on session, updating last_active_at and request count."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        # Reconstruct updated model (frozen=False on MCPSession)
        session.last_active_at = time.time()
        session.request_count += 1
        return True

    def close_session(self, session_id: str) -> bool:
        """Terminate and remove session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_active_sessions(self) -> List[MCPSession]:
        """Return all non-expired active sessions."""
        now = time.time()
        active: List[MCPSession] = []
        for s in self._sessions.values():
            if s.is_authenticated and (now - s.last_active_at) <= self._session_ttl_seconds:
                active.append(s)
        return active


class MCPNotificationDispatcher:
    """Publish-subscribe bus for dispatching MCP events, alerts, and state notifications."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = {}

    def subscribe(self, topic: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register listener callback for a specific notification topic."""
        self._subscribers.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Remove listener callback for a notification topic."""
        if topic in self._subscribers:
            self._subscribers[topic] = [cb for cb in self._subscribers[topic] if cb != callback]

    def publish(self, topic: str, payload: Dict[str, Any]) -> int:
        """Publish payload to all registered topic subscribers. Returns notification count."""
        listeners = self._subscribers.get(topic, [])
        wildcard_listeners = self._subscribers.get("*", [])
        all_listeners = listeners + [l for l in wildcard_listeners if l not in listeners]

        dispatched = 0
        for listener in all_listeners:
            try:
                listener(topic, payload)
                dispatched += 1
            except Exception as exc:
                logger.warning("Notification listener error on topic '%s': %s", topic, exc)

        return dispatched

    def get_subscribers_count(self, topic: str) -> int:
        """Count active subscribers for a topic."""
        return len(self._subscribers.get(topic, []))


class MCPIntegrationHandler:
    """Bridges MCP tool invocations with Risk Telemetry and Parquet Feature Storage."""

    def __init__(
        self,
        tool_registry: Optional[MCPToolRegistry] = None,
        telemetry_collector: Optional[RiskTelemetryCollector] = None,
        feature_store: Optional[ParquetFeatureStore] = None,
        notification_dispatcher: Optional[MCPNotificationDispatcher] = None,
        server: Optional[Any] = None,
    ) -> None:
        self.tool_registry = tool_registry or get_default_tool_registry()
        self.telemetry_collector = telemetry_collector
        self.feature_store = feature_store
        self.notification_dispatcher = notification_dispatcher or MCPNotificationDispatcher()
        self.server = server
        self._stats: Dict[str, Dict[str, Any]] = {}

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Execute a tool while capturing telemetry, logging features, and gathering latency stats."""
        start_t = time.perf_counter()
        result = self.tool_registry.call(tool_name, arguments)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        # Update stats
        current = self._stats.setdefault(
            tool_name,
            {
                "tool_name": tool_name,
                "call_count": 0,
                "error_count": 0,
                "total_latency_ms": 0.0,
                "mean_latency_ms": 0.0,
                "last_executed_at": None,
            },
        )
        current["call_count"] += 1
        if result.isError:
            current["error_count"] += 1
        current["total_latency_ms"] += elapsed_ms
        current["mean_latency_ms"] = current["total_latency_ms"] / current["call_count"]
        current["last_executed_at"] = time.time()

        # Telemetry and feature ingestion if rag_evaluate succeeded
        if tool_name == "rag_evaluate" and not result.isError:
            self._handle_rag_evaluate_post_execution(arguments, result, elapsed_ms)

        # Notify subscribers
        self.notification_dispatcher.publish(
            "tool_executed",
            {
                "tool_name": tool_name,
                "is_error": result.isError,
                "latency_ms": elapsed_ms,
                "arguments": arguments,
            },
        )

        return result

    def _handle_rag_evaluate_post_execution(
        self,
        arguments: Dict[str, Any],
        result: ToolCallResult,
        elapsed_ms: float,
    ) -> None:
        """Extract evaluation results to update telemetry and feature store."""
        try:
            if not result.content:
                return
            raw_text = result.content[0].get("text", "")
            eval_data = json.loads(raw_text)

            # Ingest into RiskTelemetryCollector if present
            if self.telemetry_collector is not None:
                metrics = EvaluationMetrics(
                    faithfulness_score=eval_data.get("faithfulness_score", 0.0),
                    answer_relevance_score=eval_data.get("answer_relevance_score", 0.0),
                    context_recall_score=eval_data.get("context_recall_score", 0.0),
                    context_precision_score=eval_data.get("context_precision_score", 0.0),
                    citation_grounding_score=eval_data.get("citation_grounding_score", 0.0),
                    hallucination_rate=eval_data.get("hallucination_rate", 0.0),
                    overall_pass=eval_data.get("overall_pass", False),
                    evaluation_latency_ms=elapsed_ms,
                )
                self.telemetry_collector.record_evaluation(metrics)

            # Ingest into ParquetFeatureStore if present
            if self.feature_store is not None:
                entity_id = f"mcp_eval_{int(time.time() * 1000)}"
                query_text = arguments.get("query_text", "")
                gen_answer = arguments.get("generated_answer", "")

                records = [
                    FeatureRecord(
                        entity_id=f"q_{entity_id}",
                        feature_group=FeatureGroup.QUERY,
                        features={
                            "query_length": len(query_text),
                            "has_question_mark": "?" in query_text,
                        },
                    ),
                    FeatureRecord(
                        entity_id=f"r_{entity_id}",
                        feature_group=FeatureGroup.RESPONSE,
                        features={
                            "response_length": len(gen_answer),
                            "has_citations": eval_data.get("citation_grounding_score", 0.0) > 0.0,
                        },
                    ),
                    FeatureRecord(
                        entity_id=f"e_{entity_id}",
                        feature_group=FeatureGroup.EVALUATION,
                        features={
                            "faithfulness_score": eval_data.get("faithfulness_score", 0.0),
                            "answer_relevance_score": eval_data.get("answer_relevance_score", 0.0),
                            "overall_pass": eval_data.get("overall_pass", False),
                            "hallucination_rate": eval_data.get("hallucination_rate", 0.0),
                        },
                    ),
                ]
                self.feature_store.write_records(records)
        except Exception as exc:
            logger.warning("Failed to record post-execution telemetry/features: %s", exc)

    def dispatch_jsonrpc(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch JSON-RPC request with telemetry tracking and handler routing."""
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params") or {}

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}
            if not tool_name:
                return JSONRPCResponse(
                    id=req_id,
                    error=JSONRPCError(code=-32602, message="Invalid params: 'name' is required for tools/call"),
                ).model_dump(exclude_none=True)

            res = self.execute_tool(tool_name, tool_args)
            return JSONRPCResponse(
                id=req_id,
                result={
                    "content": res.content,
                    "isError": res.isError,
                    "metadata": res.metadata,
                },
            ).model_dump(exclude_none=True)

        if self.server is not None:
            return self.server.handle_request(request)

        return JSONRPCResponse(
            id=req_id,
            error=JSONRPCError(code=-32601, message=f"Method not found: '{method}'"),
        ).model_dump(exclude_none=True)

    def handle_batch_evaluation(
        self,
        evaluation_items: List[Dict[str, Any]],
        mode: str = "THOROUGH",
    ) -> List[Dict[str, Any]]:
        """Run batch RAG evaluations via MCP, gathering parsed output records."""
        results: List[Dict[str, Any]] = []
        for item in evaluation_items:
            args = {
                "query_text": item.get("query_text", ""),
                "generated_answer": item.get("generated_answer", ""),
                "context_chunks": item.get("context_chunks", []),
                "ticker": item.get("ticker"),
                "mode": mode,
                "numerical_targets": item.get("numerical_targets"),
            }
            call_res = self.execute_tool("rag_evaluate", args)
            if not call_res.isError and call_res.content:
                parsed = json.loads(call_res.content[0].get("text", "{}"))
                results.append(parsed)
            else:
                results.append({"error": True, "details": call_res.metadata})
        return results

    def get_execution_stats(self) -> Dict[str, Any]:
        """Return aggregate execution telemetry and per-tool statistics."""
        total_calls = sum(s["call_count"] for s in self._stats.values())
        total_errors = sum(s["error_count"] for s in self._stats.values())

        typed_stats: Dict[str, Dict[str, Any]] = {}
        for k, v in self._stats.items():
            model = ToolExecutionStats(
                tool_name=v["tool_name"],
                call_count=v["call_count"],
                error_count=v["error_count"],
                total_latency_ms=round(v["total_latency_ms"], 2),
                mean_latency_ms=round(v["mean_latency_ms"], 2),
                last_executed_at=v["last_executed_at"],
            )
            typed_stats[k] = model.model_dump()

        return {
            "total_invocations": total_calls,
            "total_errors": total_errors,
            "stats_by_tool": typed_stats,
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._stats.clear()
