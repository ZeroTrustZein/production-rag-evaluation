"""Integration test suite covering end-to-end multi-module pipelines.

Validates end-to-end interactions between:
- RAGEvaluator and EvaluationGraph DAG
- HallucinationDetector and ClaimVerification
- SemanticCache lifecycle and LRU caching
- FinancialRiskEngine and MonteCarloStressTester
- RiskTelemetryCollector, drift detection, and alerting
"""

from typing import List

import numpy as np
import pytest

from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    DocumentType,
    EvaluationMetrics,
    EvaluationMode,
    HallucinationType,
    NumericalTarget,
    RAGQuery,
    RAGResponse,
    StressScenarioType,
    SystemHealthStatus,
    TelemetryAlert,
)


class TestRAGToRiskTelemetryIntegration:
    """Tests end-to-end flow from RAG evaluation to financial risk and telemetry."""

    @pytest.fixture
    def realistic_context(self) -> List[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id="chunk-10k-01",
                text="For fiscal 2024 Apple net sales reached $391.0 billion and services revenue reached $96.0 billion.",
                doc_type=DocumentType.SEC_10K,
                ticker="AAPL",
                period="FY2024",
            ),
        ]

    def test_full_pipeline_accurate_evaluation(self, realistic_context):
        """Accurate RAG response passes strict audit and integrates cleanly with telemetry."""
        evaluator = RAGEvaluator()
        collector = RiskTelemetryCollector(evaluator=evaluator)

        query = RAGQuery(
            query_id="q-aapl-01",
            query_text="What was Apple net sales and services revenue?",
            ticker="AAPL",
            period="FY2024",
            structured_targets=[
                NumericalTarget(metric_name="net_sales", target_value=391.0e9, tolerance_pct=0.02),
                NumericalTarget(metric_name="services", target_value=96.0e9, tolerance_pct=0.02),
            ],
            expected_numerical_targets=[391.0e9, 96.0e9],
        )

        response = RAGResponse(
            query_id="q-aapl-01",
            generated_answer="Apple net sales reached $391.0 billion and services revenue reached $96.0 billion.",
            retrieved_chunks=realistic_context,
            confidence_score=0.98,
        )

        # Run through strict financial mode
        metrics = evaluator.evaluate(query, response, mode=EvaluationMode.FINANCIAL_STRICT)
        assert metrics.overall_pass is True
        assert metrics.faithfulness_score >= 0.85
        assert metrics.answer_relevance_score >= 0.60
        assert len(metrics.findings) == 0

        # Record to telemetry
        collector.record_evaluation(metrics)
        assert len(collector.get_evaluations()) == 1

        # Check telemetry report
        report = collector.generate_report()
        assert report.health_status == SystemHealthStatus.HEALTHY
        assert report.avg_faithfulness >= 0.85
        assert report.avg_hallucination_rate == 0.0

    def test_full_pipeline_hallucinated_evaluation_degrades_system(self, realistic_context):
        """Hallucinated response triggers findings and transitions system health to UNHEALTHY."""
        evaluator = RAGEvaluator()
        collector = RiskTelemetryCollector(evaluator=evaluator)

        received_alerts: List[TelemetryAlert] = []
        collector.add_alert_listener(lambda a: received_alerts.append(a))

        # Query expects AAPL Q3 revenue
        query = RAGQuery(
            query_id="q-aapl-02",
            query_text="What was Apple's Q3 revenue and trajectory?",
            ticker="AAPL",
            period="Q3 2024",
            structured_targets=[
                NumericalTarget(metric_name="revenue", target_value=85.8e9, tolerance_pct=0.01)
            ],
        )

        # Response contains numerical distortion, directional contradiction, and entity swap
        corrupted_response = RAGResponse(
            query_id="q-aapl-02",
            generated_answer=(
                "Microsoft MSFT reported revenue of $142.5 billion, but sales declined sharply "
                "in the hardware segment during 2021."
            ),
            retrieved_chunks=realistic_context,
            confidence_score=0.45,
        )

        metrics = evaluator.evaluate(
            query, corrupted_response, mode=EvaluationMode.FINANCIAL_STRICT
        )
        assert metrics.overall_pass is False
        assert len(metrics.findings) >= 2

        finding_types = {f.finding_type for f in metrics.findings}
        assert (
            HallucinationType.ENTITY_SWAP in finding_types
            or HallucinationType.NUMERICAL_DISTORTION in finding_types
        )

        # Record evaluation multiple times to push rolling average above alert threshold
        for _ in range(5):
            collector.record_evaluation(metrics)

        # Stress returns with high drawdown
        portfolio_returns = np.array([-0.05, -0.04, -0.03, -0.02, 0.01, -0.03])
        report = collector.generate_report(
            returns=portfolio_returns,
            max_acceptable_hallucination_rate=0.05,
            min_acceptable_sharpe=0.50,
        )

        assert report.health_status in (SystemHealthStatus.DEGRADED, SystemHealthStatus.UNHEALTHY)
        assert len(report.structured_alerts) > 0
        assert len(received_alerts) > 0


class TestSemanticCacheAndDAGIntegration:
    """Tests interaction between semantic caching and DAG evaluation execution."""

    def test_cache_hit_bypasses_graph_execution(self):
        evaluator = RAGEvaluator()
        evaluator.clear_cache()

        chunk = DocumentChunk(
            chunk_id="chk-01",
            text="Alphabet Q2 net income rose 28% to $23.6 billion with operating margin of 32%.",
            document_type=DocumentType.SEC_10Q,
            ticker="GOOGL",
            period="Q2 2024",
        )

        query = RAGQuery(
            query_id="q-googl-01",
            query_text="What was Alphabet Q2 net income and operating margin?",
            ticker="GOOGL",
            period="Q2 2024",
        )
        response = RAGResponse(
            query_id="q-googl-01",
            generated_answer="Alphabet reported Q2 net income of $23.6 billion with a 32% operating margin.",
            retrieved_chunks=[chunk],
        )

        # Cold evaluation (cache miss)
        res1 = evaluator.evaluate(query, response, use_cache=True)
        assert evaluator.cache.size == 1
        assert evaluator.cache.miss_count == 1
        assert evaluator.cache.hit_count == 0

        # Identical query text (exact cache hit)
        res2 = evaluator.evaluate(query, response, use_cache=True)
        assert evaluator.cache.hit_count == 1
        assert res2 == res1

        # Semantically identical query text with slight variation
        query_variant = RAGQuery(
            query_id="q-googl-02",
            query_text="What was Alphabet Q2 net income and operating margin?",
            ticker="GOOGL",
            period="Q2 2024",
        )
        res3 = evaluator.evaluate(query_variant, response, use_cache=True)
        assert evaluator.cache.hit_count == 2
        assert res3 == res1

    def test_cache_lru_eviction_under_capacity(self):
        cache = SemanticCache(max_entries=3)
        cache.clear()

        cache.put("Query 1", "Val 1")
        cache.put("Query 2", "Val 2")
        cache.put("Query 3", "Val 3")
        assert cache.size == 3

        # Access Query 1 so it becomes most recently used
        cached = cache.get("Query 1")
        assert cached is not None
        val, _ = cached
        assert val == "Val 1"

        # Add Query 4 -> Query 2 should be evicted (LRU)
        cache.put("Query 4", "Val 4")
        assert cache.size == 3
        assert cache.get("Query 2") is None
        assert cache.get("Query 1") is not None
        assert cache.get("Query 4") is not None


class TestFinancialStressTestingAndRiskIntegration:
    """Tests multi-scenario stress simulations integrated with portfolio risk metrics."""

    def test_all_scenarios_execute_with_expected_drawdown_bounds(self):
        stress_tester = MonteCarloStressTester(random_seed=42)
        base_returns = np.array([0.001, -0.002, 0.003, 0.0005, -0.001, 0.002, -0.0005, 0.0015])

        scenarios = [
            StressScenarioType.BASELINE,
            StressScenarioType.VOLATILITY_SPIKE,
            StressScenarioType.LIQUIDITY_CRUNCH,
            StressScenarioType.RATE_SHOCK,
            StressScenarioType.BLACK_SWAN,
            StressScenarioType.HALLUCINATION_CASCADE,
        ]

        results = {}
        for sc in scenarios:
            res = stress_tester.run_scenario(base_returns, sc)
            assert res.scenario_type == sc
            assert res.n_simulations == 1000
            assert 0.0 <= res.worst_case_drawdown <= 1.0
            assert 0.0 <= res.breach_probability <= 1.0
            results[sc] = res

        # BLACK_SWAN and VOLATILITY_SPIKE should produce heavier drawdowns than BASELINE
        assert (
            results[StressScenarioType.BLACK_SWAN].worst_case_drawdown
            >= results[StressScenarioType.BASELINE].worst_case_drawdown
        )
        assert (
            results[StressScenarioType.VOLATILITY_SPIKE].worst_case_drawdown
            >= results[StressScenarioType.BASELINE].worst_case_drawdown
        )

    def test_risk_engine_cornish_fisher_skewness_impact(self):
        engine = FinancialRiskEngine()

        # Positively skewed series vs negatively skewed series
        rng = np.random.default_rng(101)
        normal_ret = rng.normal(0.001, 0.015, 200)

        # Introduce severe negative tail shocks (crash)
        crash_ret = normal_ret.copy()
        crash_ret[10] = -0.15
        crash_ret[40] = -0.12
        crash_ret[80] = -0.10

        var_normal = engine.calculate_parametric_var(
            normal_ret, confidence_level=0.99, cornish_fisher=True
        )
        var_crash = engine.calculate_parametric_var(
            crash_ret, confidence_level=0.99, cornish_fisher=True
        )

        # Crash distribution must result in a much higher Value at Risk
        assert var_crash > var_normal


class TestDriftDetectionAndAlertingWorkflow:
    """Tests statistical drift detection and alert propagation across shifting distributions."""

    def test_drift_detected_on_quality_degradation(self):
        evaluator = RAGEvaluator()
        collector = RiskTelemetryCollector(evaluator=evaluator)

        # 1. Baseline period: 20 high-quality evaluations with variance
        for i in range(20):
            collector.record_evaluation(
                EvaluationMetrics(
                    faithfulness_score=0.92 + (i % 5) * 0.015,
                    answer_relevance_score=0.90,
                    context_recall_score=0.90,
                    context_precision_score=0.90,
                    citation_grounding_score=0.90,
                    hallucination_rate=0.0,
                    overall_pass=True,
                )
            )

        # Drift check with baseline alone -> no drift
        drift_initial = collector.detect_drift(
            metric="faithfulness_score", baseline_size=15, test_size=5
        )
        assert drift_initial["drift_detected"] is False

        # 2. Degraded period: 10 poor-quality evaluations with variance
        for i in range(10):
            collector.record_evaluation(
                EvaluationMetrics(
                    faithfulness_score=0.55 + (i % 4) * 0.02,
                    answer_relevance_score=0.70,
                    context_recall_score=0.70,
                    context_precision_score=0.70,
                    citation_grounding_score=0.60,
                    hallucination_rate=0.35,
                    overall_pass=False,
                )
            )

        # Drift check now reveals statistically significant drop
        drift_result = collector.detect_drift(
            metric="faithfulness_score",
            baseline_size=20,
            test_size=10,
            alpha=0.05,
        )
        assert drift_result["drift_detected"] is True
        assert drift_result["p_value"] < 0.05
        assert drift_result["recent_mean"] < drift_result["baseline_mean"]


class TestMCPFullIntegrationPipeline:
    """Full integration tests combining MCP Server, FeatureStore, RiskTelemetry, and Evaluator."""

    def test_mcp_evaluation_feature_store_and_telemetry_loop(self, tmp_path) -> None:
        import json

        from production_rag_evaluation.features.models import FeatureGroup
        from production_rag_evaluation.features.store import ParquetFeatureStore
        from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
        from production_rag_evaluation.mcp.handlers import (
            MCPIntegrationHandler,
            MCPNotificationDispatcher,
        )
        from production_rag_evaluation.mcp.server import create_mcp_server

        feature_store = ParquetFeatureStore(base_path=tmp_path / "mcp_integ_fs")
        telemetry_collector = RiskTelemetryCollector()
        dispatcher = MCPNotificationDispatcher()
        received_alerts = []

        dispatcher.subscribe("telemetry/eval", lambda t, p: received_alerts.append(p))

        integration_handler = MCPIntegrationHandler(
            feature_store=feature_store,
            telemetry_collector=telemetry_collector,
            notification_dispatcher=dispatcher,
        )

        server = create_mcp_server()
        # Bind integration handler to server tools
        for tool_name in ["rag_evaluate", "rag_risk_metrics"]:
            t = server.tool_registry.get(tool_name)
            if t:
                server.tool_registry.register(
                    t.model_copy(
                        update={
                            "handler": lambda args, tn=tool_name: integration_handler.execute_tool(
                                tn, args
                            )
                        }
                    )
                )

        # 1. Execute evaluate via server RPC
        resp_eval = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "integ-1",
                "method": "tools/call",
                "params": {
                    "name": "rag_evaluate",
                    "arguments": {
                        "query_text": "What was Amazon AWS operating profit?",
                        "generated_answer": "Amazon AWS operating profit was $10.4 billion in Q3.",
                        "context_chunks": [
                            "Amazon AWS operating profit was $10.4 billion in Q3, up from $7.0 billion a year earlier."
                        ],
                        "ticker": "AMZN",
                    },
                },
            }
        )
        assert resp_eval is not None
        assert resp_eval["result"]["isError"] is False
        eval_metrics = json.loads(resp_eval["result"]["content"][0]["text"])
        assert eval_metrics["faithfulness_score"] >= 0.85
        assert eval_metrics["overall_pass"] is True

        # 2. Verify Feature Store was populated through integration handler
        q_records = feature_store.read_records(group=FeatureGroup.QUERY)
        e_records = feature_store.read_records(group=FeatureGroup.EVALUATION)
        assert len(q_records) >= 1
        assert len(e_records) >= 1
        assert q_records[0].features["has_question_mark"] is True
        assert e_records[0].features["overall_pass"] is True

        # 3. Verify Telemetry collector recorded evaluation
        telem_summary = telemetry_collector.export_summary()
        assert telem_summary["total_evaluations"] == 1
        assert telem_summary["avg_faithfulness"] >= 0.85

        # 4. Read exposed health & summary resources
        res_summary = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "integ-2",
                "method": "resources/read",
                "params": {"uri": "rag://evaluations/summary"},
            }
        )
        assert res_summary is not None
        summary_obj = json.loads(res_summary["result"]["contents"][0]["text"])
        assert "default_thresholds" in summary_obj

        # 5. Render prompt and verify integration consistency
        prompt_rendered = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "integ-3",
                "method": "prompts/get",
                "params": {
                    "name": "sec_10k_verification",
                    "arguments": {
                        "ticker": "AMZN",
                        "fiscal_year": "2024",
                        "filing_excerpt": "AWS operating profit reached $10.4 billion.",
                        "model_claim": "AWS operating profit was $10.4 billion in Q3.",
                    },
                },
            }
        )
        assert prompt_rendered is not None
        assert "AMZN" in prompt_rendered["result"]["messages"][0]["content"]["text"]


class TestAgentSwarmIntegrationWorkflow:
    """Full end-to-end integration tests for multi-agent swarm orchestration and handlers."""

    def test_swarm_end_to_end_lifecycle_with_event_dispatcher_and_messaging(
        self, sample_chunk_apple, sample_returns
    ) -> None:
        from production_rag_evaluation.agents.handlers import (
            AgentMessageBroker,
            WorkflowEventDispatcher,
        )
        from production_rag_evaluation.agents.models import (
            AgentMessage,
            AgentRole,
            WorkflowStatus,
        )
        from production_rag_evaluation.agents.orchestrator import create_default_orchestrator
        from production_rag_evaluation.models.schemas import (
            EvaluationMode,
            RAGQuery,
            RAGResponse,
        )

        dispatcher = WorkflowEventDispatcher()
        broker = AgentMessageBroker()
        events_captured: List[str] = []

        dispatcher.subscribe("workflow_started", lambda ev, payload: events_captured.append(ev))
        dispatcher.subscribe("step_started", lambda ev, payload: events_captured.append(ev))
        dispatcher.subscribe("step_completed", lambda ev, payload: events_captured.append(ev))
        dispatcher.subscribe("workflow_completed", lambda ev, payload: events_captured.append(ev))

        orchestrator = create_default_orchestrator(
            event_dispatcher=dispatcher,
            message_broker=broker,
        )
        from production_rag_evaluation.agents.handlers import AgentWorkflowIntegrationHandler

        handler = AgentWorkflowIntegrationHandler(
            orchestrator=orchestrator,
            event_dispatcher=dispatcher,
            message_broker=broker,
        )

        # Route broker message before execution
        broker.route_message(
            AgentMessage(
                sender="coordinator",
                receiver="planner-agent",
                role=AgentRole.PLANNER,
                content="Use high-priority evaluation",
                payload={"priority": "high"},
            ),
            orchestrator.registry,
        )
        assert broker.total_messages == 1
        assert len(broker.get_history(role=AgentRole.PLANNER)) == 1
        query = RAGQuery(
            query_id="q-integ-swarm-1",
            ticker="AAPL",
            query_text="What were Apple's net sales in fiscal year 2024?",
        )
        response = RAGResponse(
            query_id="q-integ-swarm-1",
            generated_answer="Apple reported net sales of $391.0 billion in fiscal year 2024.",
            retrieved_chunks=[sample_chunk_apple],
        )

        plan = orchestrator.create_plan(
            query=query,
            response=response,
            returns=sample_returns,
            evaluation_mode=EvaluationMode.THOROUGH,
        )
        assert len(plan.steps) >= 3

        result = handler.execute_and_record(
            plan=plan,
            query=query,
            response=response,
            returns=sample_returns,
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert result.metrics is not None
        assert result.metrics.faithfulness_score >= 0.85
        assert result.metrics.overall_pass is True
        assert result.risk_metrics is not None
        assert result.risk_metrics.sharpe_ratio is not None
        assert len(result.summary) > 0
        assert len(result.recommendations) > 0
        assert "workflow_started" in events_captured
        assert "workflow_completed" in events_captured

    def test_swarm_integration_with_feature_store_and_telemetry(
        self, tmp_path, sample_chunk_apple
    ) -> None:
        from production_rag_evaluation.agents.handlers import AgentWorkflowIntegrationHandler
        from production_rag_evaluation.agents.models import WorkflowStatus
        from production_rag_evaluation.features.models import FeatureGroup
        from production_rag_evaluation.features.store import ParquetFeatureStore
        from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
        from production_rag_evaluation.models.schemas import (
            AlertSeverity,
            EvaluationMode,
            RAGQuery,
            RAGResponse,
        )

        store = ParquetFeatureStore(base_path=tmp_path / "swarm_fs")
        collector = RiskTelemetryCollector()
        captured_alerts = []
        collector.add_alert_listener(captured_alerts.append)
        handler = AgentWorkflowIntegrationHandler(
            feature_store=store,
            telemetry_collector=collector,
        )

        # Distorted answer to trigger hallucination and high negative returns for drawdown alert
        bad_returns = np.array([0.01, -0.12, -0.10, -0.08, 0.02, -0.09, -0.05])
        q = RAGQuery(
            query_id="q-integ-distort",
            ticker="AAPL",
            query_text="What was Apple's total net sales in 2024?",
        )
        resp = RAGResponse(
            query_id="q-integ-distort",
            generated_answer="Apple reported net sales of $890.0 billion and gross margin of 12.0%.",
            retrieved_chunks=[sample_chunk_apple],
        )

        plan = handler.orchestrator.create_plan(
            query=q,
            response=resp,
            returns=bad_returns,
            evaluation_mode=EvaluationMode.THOROUGH,
        )
        result = handler.execute_and_record(
            query=q,
            response=resp,
            returns=bad_returns,
            plan=plan,
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.findings) >= 1

        # Check workflow feature was persisted through integration handler
        eval_records = store.read_records(group=FeatureGroup.EVALUATION)
        assert len(eval_records) == 1
        assert eval_records[0].features["finding_count"] >= 1
        assert eval_records[0].features["composite_health_score"] >= 0.0

        # Check telemetry collector summary and alerts
        telem_summary = collector.export_summary()
        assert telem_summary["total_evaluations"] >= 1
        assert len(captured_alerts) >= 1
        assert any(
            a.severity in (AlertSeverity.CRITICAL, AlertSeverity.WARNING) for a in captured_alerts
        )

    def test_swarm_semantic_cache_integration(self, sample_chunk_apple) -> None:
        from production_rag_evaluation.agents.handlers import AgentWorkflowIntegrationHandler
        from production_rag_evaluation.core.cache import SemanticCache
        from production_rag_evaluation.models.schemas import RAGQuery, RAGResponse

        cache = SemanticCache(similarity_threshold=0.88)
        handler = AgentWorkflowIntegrationHandler(semantic_cache=cache)

        q = RAGQuery(
            query_id="q-cache-hit-1",
            ticker="AAPL",
            query_text="What was Apple's total net sales in fiscal year 2024?",
        )
        resp = RAGResponse(
            query_id="q-cache-hit-1",
            generated_answer="Apple reported net sales of $391.0 billion in fiscal year 2024.",
            retrieved_chunks=[sample_chunk_apple],
        )

        # 1. Cold execution fills cache
        res1 = handler.execute_and_record(q, resp, use_cache=True)
        assert res1.status.value == "completed"
        assert len(cache) == 1

        # 2. Warm execution hits cache
        cache_hits: List[dict] = []
        handler.event_dispatcher.subscribe("workflow_cache_hit", lambda e, p: cache_hits.append(p))

        res2 = handler.execute_and_record(q, resp, use_cache=True)
        assert res2.status.value == "completed"
        assert len(cache_hits) == 1
        assert cache_hits[0]["query_id"] == "q-cache-hit-1"

        # 3. Different query executes and creates another cache entry
        q_rephrased = RAGQuery(
            query_id="q-cache-hit-2",
            ticker="AAPL",
            query_text="Summarize Apple's services revenue growth and gross margin trend.",
        )
        res3 = handler.execute_and_record(q_rephrased, resp, use_cache=True)
        assert res3.status.value == "completed"
        assert len(cache_hits) == 1
        assert len(cache) == 2

        # 4. Explicit bypass
        res4 = handler.execute_and_record(q, resp, use_cache=False)
        assert res4.status.value == "completed"
        assert len(cache_hits) == 1  # No new cache hit

    def test_swarm_mcp_rag_swarm_evaluate_tool_integration(
        self, sample_chunk_apple, sample_returns
    ) -> None:
        import json

        from production_rag_evaluation.mcp.server import create_mcp_server

        server = create_mcp_server()

        # Check tool registration
        tools_res = server.handle_request(
            {"jsonrpc": "2.0", "id": "tools-swarm-1", "method": "tools/list", "params": {}}
        )
        assert tools_res is not None
        tool_names = [t["name"] for t in tools_res["result"]["tools"]]
        assert "rag_swarm_evaluate" in tool_names

        # Execute tool via JSON-RPC
        call_res = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "tools-swarm-2",
                "method": "tools/call",
                "params": {
                    "name": "rag_swarm_evaluate",
                    "arguments": {
                        "query_text": "Apple fiscal 2024 net sales?",
                        "generated_answer": "Apple reported net sales of $391.0 billion.",
                        "context_chunks": [sample_chunk_apple.text],
                        "ticker": "AAPL",
                        "returns": sample_returns.tolist()[:30],
                        "mode": "FAST",
                    },
                },
            }
        )
        assert call_res is not None
        assert "result" in call_res
        assert call_res["result"]["isError"] is False
        content_text = call_res["result"]["content"][0]["text"]
        payload = json.loads(content_text)
        assert "workflow_id" in payload
        assert "status" in payload
        assert payload["status"] == "completed"
        assert payload["composite_health_score"] >= 0.0
        assert "summary" in payload
        assert "recommendations" in payload

    def test_swarm_batch_workflow_execution(self, sample_chunk_apple) -> None:
        from production_rag_evaluation.agents.handlers import AgentWorkflowIntegrationHandler
        from production_rag_evaluation.models.schemas import (
            EvaluationMode,
            RAGQuery,
            RAGResponse,
        )

        handler = AgentWorkflowIntegrationHandler()
        pairs = [
            (
                RAGQuery(query_id="batch-1", ticker="AAPL", query_text="Apple sales?"),
                RAGResponse(
                    query_id="batch-1",
                    generated_answer="Apple reported net sales of $391.0 billion.",
                    retrieved_chunks=[sample_chunk_apple],
                ),
            ),
            (
                RAGQuery(query_id="batch-2", ticker="MSFT", query_text="Cloud growth?"),
                RAGResponse(
                    query_id="batch-2",
                    generated_answer="Microsoft Cloud revenue increased by 22% in FY2024.",
                    retrieved_chunks=[sample_chunk_apple],
                ),
            ),
            (
                RAGQuery(query_id="batch-3", ticker="NVDA", query_text="GPU revenue?"),
                RAGResponse(
                    query_id="batch-3",
                    generated_answer="Data center revenue drove gross margin expansion.",
                    retrieved_chunks=[sample_chunk_apple],
                ),
            ),
        ]

        results = handler.handle_batch_workflow(pairs, evaluation_mode=EvaluationMode.FAST)
        assert len(results) == 3
        assert all(r.status.value == "completed" for r in results)

        stats = handler.get_execution_stats()
        assert stats["total_executions"] == 3
        assert stats["total_errors"] == 0
        assert stats["average_latency_ms"] > 0
