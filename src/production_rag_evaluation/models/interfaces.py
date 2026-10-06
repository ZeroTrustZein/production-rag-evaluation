"""Protocol interfaces and contracts for RAG evaluation, caching, and financial risk engines."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    EvaluationMetrics,
    HallucinationFinding,
    MonteCarloStressResult,
    RAGQuery,
    RAGResponse,
    RiskMetrics,
    TelemetryReport,
)


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Contract for RAG evaluation orchestrators."""

    def evaluate(
        self, query: RAGQuery, response: RAGResponse, use_cache: Optional[bool] = None
    ) -> EvaluationMetrics:
        """Evaluate a single query and response pair."""
        ...

    def evaluate_batch(
        self,
        pairs: List[Tuple[RAGQuery, RAGResponse]],
        use_cache: Optional[bool] = None,
    ) -> List[EvaluationMetrics]:
        """Evaluate a batch of query-response pairs."""
        ...

    def clear_cache(self) -> None:
        """Purge all cached results."""
        ...

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache health and hit statistics."""
        ...


@runtime_checkable
class HallucinationDetectorProtocol(Protocol):
    """Contract for domain hallucination detection engines."""

    def analyze(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Run all detection rules against text and context chunks."""
        ...

    def detect_numerical_distortions(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Detect numeric deviations from context figures."""
        ...

    def detect_directional_contradictions(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Detect sentiment / direction contradictions against context."""
        ...

    def detect_ungrounded_claims(
        self,
        generated_answer: str,
        context_chunks: List[DocumentChunk],
        min_overlap_ratio: float = 0.30,
    ) -> List[HallucinationFinding]:
        """Flag claims lacking lexical grounding in context."""
        ...


@runtime_checkable
class FinancialRiskEngineProtocol(Protocol):
    """Contract for financial return analysis and risk indicator computation."""

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """Calculate annualized Sharpe Ratio."""
        ...

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        target_return: float = 0.0,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """Calculate annualized Sortino Ratio focusing on downside risk."""
        ...

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate peak-to-trough maximum drawdown."""
        ...

    def calculate_calmar_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """Calculate Calmar Ratio."""
        ...

    def calculate_var_cvar(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Compute Value-at-Risk and Conditional VaR (Expected Shortfall)."""
        ...

    def compute_all_metrics(
        self, returns: np.ndarray, risk_free_rate: Optional[float] = None
    ) -> RiskMetrics:
        """Compute comprehensive suite of risk indicators."""
        ...


@runtime_checkable
class MonteCarloStressTesterProtocol(Protocol):
    """Contract for stochastic stress testing simulation engines."""

    def simulate_gbm_paths(
        self,
        initial_value: float = 100.0,
        mu: float = 0.08,
        sigma: float = 0.20,
        horizon_days: int = 60,
        n_simulations: int = 1000,
        volatility_shock_multiplier: float = 1.0,
    ) -> np.ndarray:
        """Simulate geometric brownian motion trajectories."""
        ...

    def run_stress_test(
        self,
        returns: np.ndarray,
        n_simulations: int = 1000,
        horizon_days: int = 30,
        volatility_shock: float = 1.5,
        max_allowable_drawdown: float = 0.25,
    ) -> MonteCarloStressResult:
        """Run stress simulation across empirical return distribution."""
        ...

    def stress_test_hallucination_impact(
        self,
        base_financial_estimates: List[float],
        hallucination_probability: float = 0.15,
        hallucination_distortion_pct: float = 0.25,
        n_simulations: int = 500,
    ) -> float:
        """Simulate hallucination breach frequency on forecast aggregates."""
        ...


@runtime_checkable
class SemanticCacheProtocol(Protocol):
    """Contract for fast semantic vector retrieval caches."""

    def get(self, query_text: str) -> Optional[Tuple[Any, float]]:
        """Retrieve entry by exact or semantic match."""
        ...

    def put(self, query_text: str, value: Any) -> str:
        """Store value indexed by query representation."""
        ...

    def clear(self) -> None:
        """Evict all cached entries."""
        ...

    def stats(self) -> Dict[str, Any]:
        """Return cache health telemetry."""
        ...


@runtime_checkable
class RiskTelemetryCollectorProtocol(Protocol):
    """Contract for telemetry gathering and alerting service."""

    def record_evaluation(self, metrics: EvaluationMetrics) -> None:
        """Record evaluation outcome."""
        ...

    def generate_report(
        self,
        returns: Optional[np.ndarray] = None,
        max_acceptable_hallucination_rate: float = 0.10,
        min_acceptable_sharpe: float = 0.80,
    ) -> TelemetryReport:
        """Generate synthesized monitoring report."""
        ...


@runtime_checkable
class FeatureStoreProtocol(Protocol):
    """Contract for columnar Parquet feature stores."""

    def write_records(
        self,
        records: List[Any],
        compression: str = "snappy",
    ) -> int:
        """Persist a batch of feature records."""
        ...

    def count_records(self, group: Optional[Any] = None) -> int:
        """Count total persisted records."""
        ...

    def storage_summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics on the stored features."""
        ...

    def get_online_features(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
        group: Optional[Any] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Retrieve real-time features for entity IDs."""
        ...


@runtime_checkable
class StreamingPipelineProtocol(Protocol):
    """Contract for streaming feature extraction and ingestion pipelines."""

    def ingest(self, item: Any) -> Optional[Any]:
        """Ingest single item and extract features."""
        ...

    def ingest_batch(self, items: List[Any]) -> List[Any]:
        """Ingest multiple items."""
        ...

    def flush(self) -> Any:
        """Flush buffer to durable storage."""
        ...

    def stats(self) -> Dict[str, Any]:
        """Return operational pipeline statistics."""
        ...


@runtime_checkable
class FeatureRegistryProtocol(Protocol):
    """Contract for feature schema and catalog registries."""

    def register_schema(self, schema: Any) -> None:
        """Register or update a schema specification."""
        ...

    def get_schema(self, group: Any) -> Optional[Any]:
        """Retrieve schema for a feature group."""
        ...

    def list_definitions(self, group: Optional[Any] = None) -> List[Any]:
        """List registered feature definitions."""
        ...

    def validate_record(self, record: Any, strict: bool = False) -> Any:
        """Validate a feature record against registered schema."""
        ...


@runtime_checkable
class EvaluationFeatureHandlerProtocol(Protocol):
    """Contract for evaluation subsystem integration handlers."""

    def handle_evaluation(
        self,
        query: Any,
        response: Any,
        metrics: Any,
        flush_immediately: bool = False,
    ) -> List[Any]:
        """Ingest evaluation pass artifacts into the feature store."""
        ...

    def handle_batch_evaluations(
        self,
        pairs: List[Tuple[Any, Any]],
        metrics_list: List[Any],
        flush_immediately: bool = True,
    ) -> List[Any]:
        """Ingest batch evaluation results into the feature store."""
        ...


@runtime_checkable
class FeatureCompactionProtocol(Protocol):
    """Contract for Parquet storage compaction and retention handlers."""

    def compact_group(
        self,
        group: Any,
        ttl_seconds: Optional[float] = None,
        compression: str = "snappy",
    ) -> Any:
        """Compact a single feature group partition."""
        ...

    def compact_all(
        self,
        ttl_seconds: Optional[float] = None,
        compression: str = "snappy",
    ) -> Dict[Any, Any]:
        """Compact all feature group partitions."""
        ...


@runtime_checkable
class FeatureDriftDetectorProtocol(Protocol):
    """Contract for statistical feature drift detection."""

    def detect_feature_drift(
        self,
        feature_name: str,
        group: Any,
        current_records: Optional[List[Any]] = None,
        psi_threshold: float = 0.25,
    ) -> Any:
        """Detect distribution shift for a specific feature."""
        ...


@runtime_checkable
class MCPServerProtocol(Protocol):
    """Contract for Model Context Protocol (MCP) server dispatcher."""

    def handle_request(self, request_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process an inbound JSON-RPC request."""
        ...

    def run_stdio(self) -> None:
        """Run server loop over standard input/output streams."""
        ...


@runtime_checkable
class MCPToolRegistryProtocol(Protocol):
    """Contract for MCP tool registration and invocation."""

    def register(self, tool: Any) -> None:
        """Register a new tool definition."""
        ...

    def get(self, name: str) -> Optional[Any]:
        """Retrieve tool definition by name."""
        ...

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return all registered tools formatted per MCP specification."""
        ...

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Dispatch a tool call by name with provided arguments."""
        ...


@runtime_checkable
class MCPResourceRegistryProtocol(Protocol):
    """Contract for MCP resource registration and retrieval."""

    def register(
        self,
        resource: Any,
        provider: Optional[Any] = None,
    ) -> None:
        """Register a resource representation and optional dynamic provider."""
        ...

    def get(self, uri: str) -> Optional[Any]:
        """Fetch static resource record by URI."""
        ...

    def list_resources(self) -> List[Dict[str, Any]]:
        """Return resources formatted per MCP specification."""
        ...

    def read(self, uri: str) -> Dict[str, Any]:
        """Resolve resource payload and return MCP contents structure."""
        ...


@runtime_checkable
class MCPPromptRegistryProtocol(Protocol):
    """Contract for MCP prompt template management."""

    def register(self, prompt: Any) -> None:
        """Register a new MCP prompt template."""
        ...

    def get(self, name: str) -> Optional[Any]:
        """Fetch prompt definition by name."""
        ...

    def list_prompts(self) -> List[Dict[str, Any]]:
        """Return prompts formatted per MCP specification."""
        ...

    def render(self, name: str, arguments: Dict[str, str]) -> Dict[str, Any]:
        """Render prompt arguments into MCP messages response."""
        ...


@runtime_checkable
class MCPIntegrationHandlerProtocol(Protocol):
    """Contract for MCP integration pipeline with telemetry and storage."""

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute tool with telemetry tracking."""
        ...

    def dispatch_jsonrpc(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch JSON-RPC request with telemetry tracking."""
        ...

    def get_execution_stats(self) -> Dict[str, Any]:
        """Return aggregate execution telemetry."""
        ...


@runtime_checkable
class MCPSessionHandlerProtocol(Protocol):
    """Contract for MCP client session state and authentication."""

    def create_session(
        self,
        client_info: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None,
    ) -> Any:
        """Create and track an active client session."""
        ...

    def validate_session(self, session_id: str) -> bool:
        """Verify whether session exists and is active."""
        ...

    def close_session(self, session_id: str) -> bool:
        """Terminate an active session."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """Contract for swarm agents."""

    @property
    def name(self) -> str:
        """Agent identifier."""
        ...

    def execute(self, step: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned workflow step."""
        ...


@runtime_checkable
class AgentRegistryProtocol(Protocol):
    """Contract for agent directory and resolution registry."""

    def register(self, agent: Any) -> None:
        """Register an agent."""
        ...

    def get(self, name: str) -> Optional[Any]:
        """Fetch agent by name."""
        ...

    def list_agents(self) -> List[Dict[str, Any]]:
        """List registered agents."""
        ...


@runtime_checkable
class AgentWorkflowOrchestratorProtocol(Protocol):
    """Contract for multi-agent evaluation workflow orchestrator."""

    def create_plan(
        self,
        query: RAGQuery,
        response: RAGResponse,
        returns: Optional[np.ndarray] = None,
        evaluation_mode: Any = ...,
    ) -> Any:
        """Construct multi-agent workflow plan."""
        ...

    def execute_workflow(
        self,
        query: RAGQuery,
        response: RAGResponse,
        returns: Optional[np.ndarray] = None,
        plan: Optional[Any] = None,
    ) -> Any:
        """Execute multi-agent workflow to completion."""
        ...


@runtime_checkable
class WorkflowEventDispatcherProtocol(Protocol):
    """Contract for workflow lifecycle event dispatching and subscription."""

    def subscribe(self, event: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register an event listener callback."""
        ...

    def publish(self, event: str, payload: Dict[str, Any]) -> int:
        """Publish event and invoke matching subscribers."""
        ...


@runtime_checkable
class AgentMessageBrokerProtocol(Protocol):
    """Contract for inter-agent message routing and history tracking."""

    def route_message(self, message: Any, registry: Any) -> bool:
        """Route message to recipient inbox or broadcast across swarm."""
        ...

    def get_history(
        self,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
        role: Optional[Any] = None,
    ) -> List[Any]:
        """Fetch filtered message exchange history."""
        ...


@runtime_checkable
class AgentWorkflowIntegrationHandlerProtocol(Protocol):
    """Contract for multi-agent workflow integration and telemetry recording."""

    def execute_and_record(
        self,
        query: RAGQuery,
        response: RAGResponse,
        returns: Optional[np.ndarray] = None,
        plan: Optional[Any] = None,
        use_cache: bool = True,
    ) -> Any:
        """Execute workflow and record features, telemetry, and metrics."""
        ...

    def get_execution_stats(self) -> Dict[str, Any]:
        """Return execution metrics across invocations."""
        ...
