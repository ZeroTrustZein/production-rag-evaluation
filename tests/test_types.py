"""Comprehensive test suite for domain models, schemas, and protocol interfaces."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
from production_rag_evaluation.models.interfaces import (
    EvaluatorProtocol,
    FeatureStoreProtocol,
    FinancialRiskEngineProtocol,
    HallucinationDetectorProtocol,
    MonteCarloStressTesterProtocol,
    RiskTelemetryCollectorProtocol,
    SemanticCacheProtocol,
    StreamingPipelineProtocol,
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
