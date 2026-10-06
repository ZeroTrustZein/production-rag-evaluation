"""Comprehensive test suite for domain models, schemas, and protocol interfaces."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from production_rag_evaluation.agents.base import (
    AgentRegistry,
    BaseAgent,
    get_default_agent_registry,
)
from production_rag_evaluation.agents.handlers import (
    AgentMessageBroker,
    AgentWorkflowIntegrationHandler,
    WorkflowEventDispatcher,
)
from production_rag_evaluation.agents.models import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    AgentWorkflowConfig,
    StepExecutionMode,
    WorkflowExecutionResult,
    WorkflowPlan,
    WorkflowStatus,
    WorkflowStep,
)
from production_rag_evaluation.agents.orchestrator import (
    AgentSwarmOrchestrator,
    create_default_orchestrator,
)
from production_rag_evaluation.agents.specialized import (
    AuditorAgent,
    EvaluatorAgent,
    PlannerAgent,
    RiskAnalystAgent,
    SynthesizerAgent,
)
from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.features.models import (
    FeatureDefinition,
    FeatureGroup,
    FeatureSchema,
    FeatureValueType,
    OnlineFeatureVector,
    SlidingWindowStats,
)
from production_rag_evaluation.features.pipeline import StreamingFeaturePipeline
from production_rag_evaluation.features.store import ParquetFeatureStore
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.mcp.models import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPConfig,
    MCPPrompt,
    MCPPromptArgument,
    MCPPromptMessage,
    MCPProtocolVersion,
    MCPResource,
    MCPSession,
    MCPTool,
    MCPTransport,
    ToolCallResult,
    ToolExecutionStats,
    ToolParameter,
)
from production_rag_evaluation.mcp.prompts import (
    get_default_prompt_registry,
)
from production_rag_evaluation.mcp.resources import (
    get_default_resource_registry,
)
from production_rag_evaluation.mcp.server import (
    MCPServer,
    create_mcp_server,
)
from production_rag_evaluation.mcp.tools import (
    get_default_tool_registry,
)
from production_rag_evaluation.models.interfaces import (
    AgentMessageBrokerProtocol,
    AgentProtocol,
    AgentRegistryProtocol,
    AgentWorkflowIntegrationHandlerProtocol,
    AgentWorkflowOrchestratorProtocol,
    EvaluatorProtocol,
    FeatureStoreProtocol,
    FinancialRiskEngineProtocol,
    HallucinationDetectorProtocol,
    MCPPromptRegistryProtocol,
    MCPResourceRegistryProtocol,
    MCPServerProtocol,
    MCPToolRegistryProtocol,
    MonteCarloStressTesterProtocol,
    RiskTelemetryCollectorProtocol,
    SemanticCacheProtocol,
    StreamingPipelineProtocol,
    WorkflowEventDispatcherProtocol,
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


class TestEnumTypes:
    """Validate all domain enum variants and string behaviors."""

    def test_document_types(self) -> None:
        expected = {
            "EARNINGS_CALL",
            "SEC_10K",
            "SEC_10Q",
            "SEC_8K",
            "PRESS_RELEASE",
            "ANALYST_REPORT",
            "FINANCIAL_NEWS",
        }
        actual = {dt.value for dt in DocumentType}
        assert actual == expected
        assert DocumentType.SEC_10K == "SEC_10K"

    def test_severity_levels(self) -> None:
        assert [s.value for s in Severity] == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_hallucination_types(self) -> None:
        expected = {
            "NUMERICAL_DISTORTION",
            "UNGROUNDED_CLAIM",
            "CONTRADICTION",
            "ENTITY_SWAP",
            "TEMPORAL_MISMATCH",
            "OMISSION_OF_RISK",
        }
        assert {ht.value for ht in HallucinationType} == expected

    def test_risk_profiles(self) -> None:
        assert {p.value for p in RiskProfile} == {
            "CONSERVATIVE",
            "MODERATE",
            "AGGRESSIVE",
            "SPECULATIVE",
        }

    def test_stress_scenarios(self) -> None:
        assert {s.value for s in StressScenarioType} == {
            "BASELINE",
            "VOLATILITY_SPIKE",
            "LIQUIDITY_CRUNCH",
            "RATE_SHOCK",
            "BLACK_SWAN",
            "HALLUCINATION_CASCADE",
        }

    def test_claim_status_and_eval_mode(self) -> None:
        assert ClaimStatus.VERIFIED == "VERIFIED"
        assert EvaluationMode.FINANCIAL_STRICT == "FINANCIAL_STRICT"
        assert ReturnFrequency.DAILY == "DAILY"
        assert AlertSeverity.WARNING == "WARNING"
        assert SystemHealthStatus.HEALTHY == "HEALTHY"

    def test_agent_enums(self) -> None:
        expected_roles = {
            "planner",
            "evaluator",
            "auditor",
            "risk_analyst",
            "synthesizer",
            "coordinator",
        }
        assert {r.value for r in AgentRole} == expected_roles
        assert AgentRole.PLANNER == "planner"

        expected_statuses = {"idle", "busy", "completed", "failed", "waiting"}
        assert {s.value for s in AgentStatus} == expected_statuses
        assert AgentStatus.IDLE == "idle"

        expected_wf_statuses = {
            "pending",
            "running",
            "completed",
            "failed",
            "skipped",
            "cancelled",
        }
        assert {s.value for s in WorkflowStatus} == expected_wf_statuses
        assert WorkflowStatus.PENDING == "pending"

        expected_modes = {"sequential", "parallel", "conditional"}
        assert {m.value for m in StepExecutionMode} == expected_modes
        assert StepExecutionMode.SEQUENTIAL == "sequential"


class TestDomainModels:
    """Validate domain schema construction, serialization, and boundary validations."""

    def test_numerical_target_valid_and_frozen(self) -> None:
        target = NumericalTarget(
            metric_name="Net Sales",
            target_value=391.04,
            unit="USD_BN",
            tolerance_pct=0.005,
            is_currency=True,
        )
        assert target.metric_name == "Net Sales"
        assert target.target_value == 391.04
        assert target.tolerance_pct == 0.005

        # Test frozen immutability
        with pytest.raises(ValidationError):
            target.metric_name = "Operating Margin"  # type: ignore[misc]

    def test_numerical_target_validation(self) -> None:
        with pytest.raises(ValidationError):
            NumericalTarget(metric_name="", target_value=100.0)

        with pytest.raises(ValidationError):
            NumericalTarget(metric_name="Rev", target_value=100.0, tolerance_pct=1.5)

    def test_claim_verification_result(self) -> None:
        res = ClaimVerificationResult(
            claim_id="claim-01",
            claim_text="Revenue was $10M",
            status=ClaimStatus.VERIFIED,
            confidence=0.98,
            evidence_chunk_ids=["chunk-001"],
            explanation="Exact match in 10-K Item 7",
        )
        assert res.claim_id == "claim-01"
        assert res.status == ClaimStatus.VERIFIED
        assert res.confidence == 0.98

        with pytest.raises(ValidationError):
            ClaimVerificationResult(claim_id="c1", claim_text="text", confidence=1.5)

    def test_document_chunk_validation(self) -> None:
        chunk = DocumentChunk(
            chunk_id="chk_101",
            text="Operating income increased 12%",
            doc_type=DocumentType.SEC_10Q,
            ticker="NVDA",
            period="Q2 2025",
            section="MD&A",
            source_url="https://sec.gov/edgar/nvda",
            metadata={"priority": 1},
        )
        assert chunk.ticker == "NVDA"
        assert chunk.doc_type == DocumentType.SEC_10Q
        assert chunk.metadata["priority"] == 1

        with pytest.raises(ValidationError):
            DocumentChunk(chunk_id="", text="valid", ticker="AAPL")

        with pytest.raises(ValidationError):
            DocumentChunk(chunk_id="c1", text="", ticker="AAPL")

        with pytest.raises(ValidationError):
            DocumentChunk(chunk_id="c1", text="valid", ticker="")

    def test_citation_model(self) -> None:
        cit = Citation(
            chunk_id="chk_101",
            quote="net sales reached record high",
            confidence=0.95,
            source_ticker="AAPL",
            char_offset=(10, 42),
        )
        assert cit.chunk_id == "chk_101"
        assert cit.char_offset == (10, 42)

        with pytest.raises(ValidationError):
            Citation(chunk_id="", quote="quote")

        with pytest.raises(ValidationError):
            Citation(chunk_id="chk_1", quote="")

        with pytest.raises(ValidationError):
            Citation(chunk_id="chk_1", quote="quote", confidence=-0.1)

    def test_rag_query_and_response(self) -> None:
        target = NumericalTarget(metric_name="EPS", target_value=1.64)
        query = RAGQuery(
            query_id="q-001",
            query_text="What was diluted EPS in Q3?",
            ticker="AAPL",
            period="Q3 2024",
            expected_numerical_targets=[1.64],
            structured_targets=[target],
        )
        assert len(query.structured_targets) == 1
        assert query.structured_targets[0].metric_name == "EPS"

        response = RAGResponse(
            query_id="q-001",
            generated_answer="Diluted earnings per share was $1.64.",
            latency_ms=142.5,
            token_usage={"prompt_tokens": 1200, "completion_tokens": 50},
            model_name="claude-3-5-sonnet",
        )
        assert response.latency_ms == 142.5
        assert response.model_name == "claude-3-5-sonnet"

    def test_evaluation_metrics_and_batch_summary(self) -> None:
        finding = HallucinationFinding(
            finding_type=HallucinationType.NUMERICAL_DISTORTION,
            severity=Severity.HIGH,
            claim_text="Revenue dropped by 50%",
            reference_text="Revenue grew 5%",
            confidence=0.95,
            explanation="Numerical mismatch",
            impact_on_risk="Severe underestimation of cash flow",
        )
        assert finding.impact_on_risk is not None

        metrics = EvaluationMetrics(
            faithfulness_score=0.85,
            answer_relevance_score=0.90,
            context_recall_score=0.95,
            context_precision_score=0.88,
            citation_grounding_score=0.92,
            hallucination_rate=0.15,
            findings=[finding],
            overall_pass=True,
            evaluation_latency_ms=250.0,
            metadata={"eval_run": "v1.2"},
        )
        assert metrics.overall_pass is True
        assert len(metrics.findings) == 1
        assert metrics.metadata["eval_run"] == "v1.2"

        summary = BatchEvaluationSummary(
            batch_id="batch-2026-09",
            total_queries=1,
            passed_queries=1,
            failed_queries=0,
            pass_rate=1.0,
            mean_faithfulness=0.85,
            mean_relevance=0.90,
            mean_latency_ms=250.0,
            p95_latency_ms=250.0,
            results=[metrics],
        )
        assert summary.pass_rate == 1.0
        assert summary.total_queries == 1

    def test_risk_metrics_schema(self) -> None:
        rm = RiskMetrics(
            sharpe_ratio=1.45,
            sortino_ratio=1.92,
            max_drawdown=0.14,
            calmar_ratio=1.85,
            annualized_return=0.18,
            annualized_volatility=0.12,
            downside_deviation=0.08,
            var_95=0.018,
            cvar_95=0.024,
            var_99=0.028,
            cvar_99=0.035,
            information_ratio=0.85,
            beta=1.05,
            alpha=0.03,
        )
        assert rm.sharpe_ratio == 1.45
        assert rm.var_99 == 0.028
        assert rm.beta == 1.05

        with pytest.raises(ValidationError):
            RiskMetrics(
                sharpe_ratio=1.0,
                sortino_ratio=1.0,
                max_drawdown=1.5,  # le=1.0
                calmar_ratio=1.0,
                annualized_return=0.1,
                annualized_volatility=0.1,
                downside_deviation=0.05,
                var_95=0.02,
                cvar_95=0.03,
            )

    def test_stress_test_config_and_result(self) -> None:
        cfg = StressTestConfig(
            n_simulations=5000,
            horizon_days=60,
            volatility_shock=2.0,
            scenario_type=StressScenarioType.LIQUIDITY_CRUNCH,
        )
        assert cfg.n_simulations == 5000
        assert cfg.scenario_type == StressScenarioType.LIQUIDITY_CRUNCH

        with pytest.raises(ValidationError):
            StressTestConfig(n_simulations=5)  # ge=10

        with pytest.raises(ValidationError):
            StressTestConfig(horizon_days=0)  # ge=1

        res = MonteCarloStressResult(
            n_simulations=5000,
            horizon_steps=60,
            p5_terminal=82.5,
            p50_terminal=103.0,
            p95_terminal=135.0,
            mean_terminal=104.2,
            std_terminal=16.8,
            worst_case_drawdown=0.22,
            stress_pass=True,
            scenario_type=StressScenarioType.LIQUIDITY_CRUNCH,
            breach_probability=0.04,
        )
        assert res.scenario_type == StressScenarioType.LIQUIDITY_CRUNCH
        assert res.breach_probability == 0.04

    def test_telemetry_structured_alert_and_report(self) -> None:
        alert = TelemetryAlert(
            alert_id="alert-001",
            severity=AlertSeverity.CRITICAL,
            metric_name="hallucination_rate",
            threshold=0.10,
            observed_value=0.28,
            message="Critical hallucination spike detected",
        )
        assert alert.severity == AlertSeverity.CRITICAL

        report = TelemetryReport(
            total_evaluations=100,
            avg_faithfulness=0.72,
            avg_hallucination_rate=0.28,
            cache_hit_rate=0.45,
            avg_latency_ms=88.5,
            health_status=SystemHealthStatus.UNHEALTHY,
            alerts_triggered=[alert.message],
            structured_alerts=[alert],
        )
        assert report.health_status == SystemHealthStatus.UNHEALTHY
        assert len(report.structured_alerts) == 1
        assert report.structured_alerts[0].observed_value == 0.28


class TestProtocolConformity:
    """Verify runtime structural subtyping against defined protocols."""

    def test_evaluator_protocol_conformity(self) -> None:
        evaluator = RAGEvaluator()
        assert isinstance(evaluator, EvaluatorProtocol)

    def test_hallucination_detector_protocol_conformity(self) -> None:
        detector = HallucinationDetector()
        assert isinstance(detector, HallucinationDetectorProtocol)

    def test_financial_risk_engine_protocol_conformity(self) -> None:
        engine = FinancialRiskEngine()
        assert isinstance(engine, FinancialRiskEngineProtocol)

    def test_monte_carlo_stress_tester_protocol_conformity(self) -> None:
        tester = MonteCarloStressTester()
        assert isinstance(tester, MonteCarloStressTesterProtocol)

    def test_semantic_cache_protocol_conformity(self) -> None:
        cache = SemanticCache()
        assert isinstance(cache, SemanticCacheProtocol)

    def test_risk_telemetry_collector_protocol_conformity(self) -> None:
        collector = RiskTelemetryCollector()
        assert isinstance(collector, RiskTelemetryCollectorProtocol)

    def test_feature_store_protocol_conformity(self, tmp_path) -> None:
        store = ParquetFeatureStore(base_path=tmp_path / "feat_store")
        assert isinstance(store, FeatureStoreProtocol)

    def test_streaming_pipeline_protocol_conformity(self, tmp_path) -> None:
        store = ParquetFeatureStore(base_path=tmp_path / "feat_store")
        pipeline = StreamingFeaturePipeline(store=store)
        assert isinstance(pipeline, StreamingPipelineProtocol)

    def test_new_subsystem_protocols_conformity(self, tmp_path) -> None:
        from production_rag_evaluation.features.drift import FeatureDriftDetector
        from production_rag_evaluation.features.handlers import (
            EvaluationFeatureHandler,
            FeatureCompactionHandler,
        )
        from production_rag_evaluation.features.registry import get_default_registry
        from production_rag_evaluation.models.interfaces import (
            EvaluationFeatureHandlerProtocol,
            FeatureCompactionProtocol,
            FeatureDriftDetectorProtocol,
            FeatureRegistryProtocol,
        )

        store = ParquetFeatureStore(base_path=tmp_path / "feat_store")
        pipeline = StreamingFeaturePipeline(store=store)

        registry = get_default_registry()
        assert isinstance(registry, FeatureRegistryProtocol)

        handler = EvaluationFeatureHandler(pipeline=pipeline)
        assert isinstance(handler, EvaluationFeatureHandlerProtocol)

        compactor = FeatureCompactionHandler(store=store)
        assert isinstance(compactor, FeatureCompactionProtocol)

        drift = FeatureDriftDetector(store=store)
        assert isinstance(drift, FeatureDriftDetectorProtocol)

    def test_mcp_protocols_conformity(self) -> None:
        server = MCPServer()
        assert isinstance(server, MCPServerProtocol)

        server_created = create_mcp_server()
        assert isinstance(server_created, MCPServerProtocol)

        tools = get_default_tool_registry()
        assert isinstance(tools, MCPToolRegistryProtocol)

        resources = get_default_resource_registry()
        assert isinstance(resources, MCPResourceRegistryProtocol)

        prompts = get_default_prompt_registry()
        assert isinstance(prompts, MCPPromptRegistryProtocol)

        from production_rag_evaluation.mcp.handlers import (
            MCPIntegrationHandler,
            MCPSessionHandler,
        )
        from production_rag_evaluation.models.interfaces import (
            MCPIntegrationHandlerProtocol,
            MCPSessionHandlerProtocol,
        )

        session_mgr = MCPSessionHandler()
        assert isinstance(session_mgr, MCPSessionHandlerProtocol)

        integ = MCPIntegrationHandler()
        assert isinstance(integ, MCPIntegrationHandlerProtocol)

    def test_agent_protocols_conformity(self) -> None:
        for agent_cls in [
            PlannerAgent,
            EvaluatorAgent,
            AuditorAgent,
            RiskAnalystAgent,
            SynthesizerAgent,
        ]:
            assert issubclass(agent_cls, BaseAgent)

        planner = PlannerAgent()
        assert isinstance(planner, AgentProtocol)

        evaluator = EvaluatorAgent()
        assert isinstance(evaluator, AgentProtocol)

        auditor = AuditorAgent()
        assert isinstance(auditor, AgentProtocol)

        risk_analyst = RiskAnalystAgent()
        assert isinstance(risk_analyst, AgentProtocol)

        synthesizer = SynthesizerAgent()
        assert isinstance(synthesizer, AgentProtocol)

        registry = AgentRegistry()
        assert isinstance(registry, AgentRegistryProtocol)

        default_registry = get_default_agent_registry()
        assert isinstance(default_registry, AgentRegistryProtocol)

        orchestrator = AgentSwarmOrchestrator()
        assert isinstance(orchestrator, AgentWorkflowOrchestratorProtocol)

        default_orchestrator = create_default_orchestrator()
        assert isinstance(default_orchestrator, AgentWorkflowOrchestratorProtocol)

        dispatcher = WorkflowEventDispatcher()
        assert isinstance(dispatcher, WorkflowEventDispatcherProtocol)

        broker = AgentMessageBroker()
        assert isinstance(broker, AgentMessageBrokerProtocol)

        handler = AgentWorkflowIntegrationHandler()
        assert isinstance(handler, AgentWorkflowIntegrationHandlerProtocol)


class TestFeatureStoreModels:
    """Validate Feature Definition, Schema, Statistics, and Streaming Batch models."""

    def test_feature_definition_and_schema(self) -> None:
        feat = FeatureDefinition(
            name="query_length",
            feature_group=FeatureGroup.QUERY,
            value_type=FeatureValueType.INT,
            description="Character length of input query",
            nullable=False,
        )
        assert feat.name == "query_length"
        assert feat.value_type == FeatureValueType.INT
        assert feat.nullable is False

        schema = FeatureSchema(
            name="query_schema",
            feature_group=FeatureGroup.QUERY,
            features=[feat],
        )
        assert schema.get_feature("query_length") == feat
        assert schema.get_feature("missing") is None

    def test_sliding_window_and_online_vector(self) -> None:
        stats = SlidingWindowStats(
            window_size=10,
            count=5,
            avg_latency_ms=12.5,
            overall_pass_rate=0.8,
            avg_faithfulness=0.9,
            rolling_hallucination_rate=0.2,
        )
        assert stats.window_size == 10
        assert stats.count == 5

        vec = OnlineFeatureVector(
            entity_id="e1",
            feature_group=FeatureGroup.QUERY,
            features={"char_len": 42},
        )
        assert vec.entity_id == "e1"
        assert vec.features["char_len"] == 42

    def test_feature_registry_list_definitions(self) -> None:
        from production_rag_evaluation.features.registry import get_default_registry

        registry = get_default_registry()
        defs = registry.list_definitions()
        assert len(defs) > 0
        assert all(isinstance(d, FeatureDefinition) for d in defs)

        query_defs = registry.list_definitions(FeatureGroup.QUERY)
        assert len(query_defs) > 0
        assert all(d.feature_group == FeatureGroup.QUERY for d in query_defs)


class TestMCPModels:
    """Validate Model Context Protocol schemas, messages, and configurations."""

    def test_mcp_enums(self) -> None:
        assert MCPProtocolVersion.V2024_11_05.value == "2024-11-05"
        transports = {t.value for t in MCPTransport}
        assert {"stdio", "sse", "http"}.issubset(transports)

    def test_mcp_config(self) -> None:
        cfg = MCPConfig()
        assert cfg.server_name == "production-rag-evaluation-mcp"
        assert cfg.transport == MCPTransport.STDIO
        assert cfg.port == 8000

        with pytest.raises(ValidationError):
            setattr(cfg, "port", 9000)  # frozen model

        custom = MCPConfig(
            server_name="custom-mcp",
            transport=MCPTransport.SSE,
            host="0.0.0.0",
            port=9090,
            auth_token="secret",
        )
        assert custom.server_name == "custom-mcp"
        assert custom.transport == MCPTransport.SSE
        assert custom.port == 9090

    def test_mcp_tool_model(self) -> None:
        tool = MCPTool(
            name="rag_eval",
            description="Evaluate RAG",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Query text",
                    required=True,
                )
            ],
            handler=lambda args: ToolCallResult.text("ok"),
        )
        assert tool.name == "rag_eval"
        assert "query" in tool.inputSchema["properties"]
        assert tool.inputSchema["required"] == ["query"]

        assert tool.handler is not None
        res = tool.handler({"query": "test"})
        assert res.isError is False
        assert res.content[0]["text"] == "ok"

    def test_mcp_resource_model(self) -> None:
        res = MCPResource(
            uri="rag://health",
            name="health",
            description="Health status",
            mimeType="application/json",
            text='{"status": "ok"}',
        )
        assert res.uri == "rag://health"
        assert res.mimeType == "application/json"
        assert res.text == '{"status": "ok"}'

    def test_mcp_prompt_models(self) -> None:
        arg = MCPPromptArgument(
            name="ticker",
            description="Ticker symbol",
            required=True,
        )
        assert arg.name == "ticker"
        assert arg.required is True

        msg = MCPPromptMessage(
            role="user",
            content="Audit ticker {ticker}",
        )
        assert msg.role == "user"

        prompt = MCPPrompt(
            name="audit_template",
            description="Template for audit",
            arguments=[arg],
            template="Audit ticker {ticker}",
        )
        assert prompt.name == "audit_template"
        assert len(prompt.arguments) == 1

    def test_jsonrpc_models(self) -> None:
        req = JSONRPCRequest(
            id=1,
            method="tools/call",
            params={"name": "rag_eval", "arguments": {}},
        )
        assert req.jsonrpc == "2.0"
        assert req.id == 1

        err = JSONRPCError(code=-32601, message="Method not found")
        assert err.code == -32601

        resp = JSONRPCResponse(id=1, result={"ok": True})
        assert resp.result == {"ok": True}
        assert resp.error is None

        err_resp = JSONRPCResponse(id=1, error=err)
        assert err_resp.error is not None
        assert err_resp.error.code == -32601

        session = MCPSession(session_id="sess-1")
        assert session.session_id == "sess-1"
        assert session.is_authenticated is True
        assert session.request_count == 0

        stats = ToolExecutionStats(
            tool_name="rag_evaluate",
            call_count=10,
            error_count=1,
            total_latency_ms=150.0,
            mean_latency_ms=15.0,
        )
        assert stats.tool_name == "rag_evaluate"
        assert stats.call_count == 10
        assert stats.mean_latency_ms == 15.0


class TestAgentModels:
    """Validate multi-agent swarm schemas, workflow plans, and execution models."""

    def test_agent_message_model(self) -> None:
        msg = AgentMessage(
            sender="planner-01",
            receiver="evaluator-01",
            role=AgentRole.PLANNER,
            content="Execute RAG evaluation",
            payload={"query_id": "q-100"},
        )
        assert msg.sender == "planner-01"
        assert msg.receiver == "evaluator-01"
        assert msg.role == AgentRole.PLANNER
        assert msg.content == "Execute RAG evaluation"
        assert msg.payload == {"query_id": "q-100"}
        assert msg.message_id is not None
        assert msg.timestamp is not None

        # Frozen immutability
        with pytest.raises(ValidationError):
            setattr(msg, "content", "Updated content")

    def test_workflow_step_model(self) -> None:
        step = WorkflowStep(
            name="evaluate_grounding",
            agent_role=AgentRole.EVALUATOR,
            mode=StepExecutionMode.PARALLEL,
            dependencies=["step-001"],
            inputs={"threshold": 0.8},
        )
        assert step.name == "evaluate_grounding"
        assert step.agent_role == AgentRole.EVALUATOR
        assert step.mode == StepExecutionMode.PARALLEL
        assert step.dependencies == ["step-001"]
        assert step.status == WorkflowStatus.PENDING
        assert step.inputs["threshold"] == 0.8
        assert step.outputs == {}
        assert step.execution_time_ms == 0.0
        assert step.error is None
        assert step.step_id is not None

        # Mutable update
        step.status = WorkflowStatus.COMPLETED
        step.outputs = {"score": 0.95}
        step.execution_time_ms = 45.2
        assert step.status == WorkflowStatus.COMPLETED
        assert step.outputs["score"] == 0.95

    def test_workflow_plan_model(self) -> None:
        s1 = WorkflowStep(name="plan", agent_role=AgentRole.PLANNER)
        s2 = WorkflowStep(name="eval", agent_role=AgentRole.EVALUATOR)
        plan = WorkflowPlan(
            query_id="q-200",
            ticker="AAPL",
            steps=[s1, s2],
            estimated_duration_ms=120.0,
        )
        assert plan.query_id == "q-200"
        assert plan.ticker == "AAPL"
        assert len(plan.steps) == 2
        assert plan.plan_id is not None
        assert plan.created_at is not None
        assert plan.estimated_duration_ms == 120.0

        assert plan.get_step(s1.step_id) == s1
        assert plan.get_step("non-existent-id") is None

        planner_steps = plan.get_steps_for_role(AgentRole.PLANNER)
        assert planner_steps == [s1]
        auditor_steps = plan.get_steps_for_role(AgentRole.AUDITOR)
        assert auditor_steps == []

    def test_workflow_execution_result_model(self) -> None:
        plan = WorkflowPlan(query_id="q-300", steps=[])
        metrics = EvaluationMetrics(
            faithfulness_score=0.9,
            answer_relevance_score=0.85,
            context_recall_score=0.88,
            context_precision_score=0.92,
            citation_grounding_score=0.91,
            hallucination_rate=0.05,
            overall_pass=True,
        )
        result = WorkflowExecutionResult(
            plan=plan,
            status=WorkflowStatus.COMPLETED,
            metrics=metrics,
            summary="All evaluation steps passed successfully.",
            recommendations=["Monitor latency."],
            total_execution_time_ms=185.5,
        )
        assert result.workflow_id is not None
        assert result.status == WorkflowStatus.COMPLETED
        assert result.plan == plan
        assert result.metrics == metrics
        assert result.summary == "All evaluation steps passed successfully."
        assert len(result.recommendations) == 1
        assert result.total_execution_time_ms == 185.5
        assert result.error is None

    def test_agent_workflow_config_model(self) -> None:
        cfg = AgentWorkflowConfig()
        assert cfg.max_iterations == 10
        assert cfg.timeout_seconds == 60.0
        assert cfg.concurrency_limit == 5
        assert cfg.halt_on_high_severity_hallucination is False
        assert cfg.auto_retry_failed_steps is False

        # Frozen immutability
        with pytest.raises(ValidationError):
            setattr(cfg, "max_iterations", 20)

        # Validation constraints
        with pytest.raises(ValidationError):
            AgentWorkflowConfig(max_iterations=0)  # ge=1

        with pytest.raises(ValidationError):
            AgentWorkflowConfig(max_iterations=51)  # le=50

        with pytest.raises(ValidationError):
            AgentWorkflowConfig(timeout_seconds=0.0)  # gt=0.0

        with pytest.raises(ValidationError):
            AgentWorkflowConfig(concurrency_limit=0)  # ge=1

        with pytest.raises(ValidationError):
            AgentWorkflowConfig(concurrency_limit=21)  # le=20
