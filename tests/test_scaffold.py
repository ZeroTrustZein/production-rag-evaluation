"""Comprehensive scaffold test suite for production-rag-evaluation."""

from __future__ import annotations

import numpy as np
from click.testing import CliRunner

from production_rag_evaluation import (
    FinancialRiskEngine,
    HallucinationDetector,
    MonteCarloStressTester,
    RAGEvaluator,
    RiskTelemetryCollector,
    SemanticCache,
    __version__,
)
from production_rag_evaluation.cli.main import cli
from production_rag_evaluation.models.schemas import (
    Citation,
    DocumentChunk,
    DocumentType,
    HallucinationType,
    RAGQuery,
    RAGResponse,
    Severity,
)


def test_package_metadata():
    """Verify package version and exposed top-level symbols."""
    assert __version__ == "0.1.0"
    assert RAGEvaluator is not None
    assert SemanticCache is not None
    assert FinancialRiskEngine is not None


def test_schemas_instantiation(sample_chunk_apple: DocumentChunk):
    """Test serialization and field constraints of core domain models."""
    assert sample_chunk_apple.ticker == "AAPL"
    assert sample_chunk_apple.doc_type == DocumentType.SEC_10K
    assert "391.0 billion" in sample_chunk_apple.text

    q = RAGQuery(query_id="q1", query_text="What were Apple's 2024 net sales?", ticker="AAPL")
    assert q.query_id == "q1"

    citation = Citation(chunk_id=sample_chunk_apple.chunk_id, quote="$391.0 billion")
    resp = RAGResponse(
        query_id="q1",
        generated_answer="Apple reported net sales of $391.0 billion.",
        retrieved_chunks=[sample_chunk_apple],
        citations=[citation],
    )
    assert len(resp.retrieved_chunks) == 1
    assert resp.citations[0].quote == "$391.0 billion"


def test_semantic_cache_lifecycle():
    """Test exact hit, semantic similarity match, and LRU eviction."""
    cache = SemanticCache(similarity_threshold=0.80, max_entries=2)

    cache.put("Apple FY2024 revenue growth", {"answer": "Up 2%"})
    cache.put("Microsoft Azure cloud revenue", {"answer": "Up 29%"})

    # Exact match
    res_exact = cache.get("Apple FY2024 revenue growth")
    assert res_exact is not None
    val, sim = res_exact
    assert val == {"answer": "Up 2%"}
    assert sim == 1.0

    # Semantic similarity match
    res_semantic = cache.get("Apple FY2024 revenue growth and performance")
    assert res_semantic is not None
    val, sim = res_semantic
    assert val == {"answer": "Up 2%"}
    assert sim >= 0.80

    # Eviction on capacity
    cache.put("NVIDIA data center GPU sales", {"answer": "Up 154%"})
    assert cache.size == 2
    # Microsoft was least recently used, should be evicted
    assert cache.get("Microsoft Azure cloud revenue") is None

    stats = cache.stats()
    assert stats["hit_count"] >= 2
    assert stats["hit_rate"] > 0.0


def test_hallucination_detector_numerical_distortion(sample_chunk_apple: DocumentChunk):
    """Ensure distorted numerical claims are flagged with high confidence."""
    detector = HallucinationDetector(numerical_tolerance_pct=0.01)

    # Valid answer
    valid_answer = "Apple reported net sales of $391.0 billion and gross margin of 46.2%."
    findings_valid = detector.detect_numerical_distortions(valid_answer, [sample_chunk_apple])
    assert len(findings_valid) == 0

    # Hallucinated answer ($450.0 billion instead of $391.0 billion)
    hallucinated_answer = "Apple reported net sales of $450.0 billion in fiscal year 2024."
    findings_hallucinated = detector.detect_numerical_distortions(hallucinated_answer, [sample_chunk_apple])
    assert len(findings_hallucinated) > 0
    assert findings_hallucinated[0].finding_type == HallucinationType.NUMERICAL_DISTORTION
    assert findings_hallucinated[0].severity == Severity.HIGH


def test_hallucination_detector_contradiction(sample_chunk_apple: DocumentChunk):
    """Ensure directional contradictions are captured."""
    detector = HallucinationDetector()

    # Context says "up 2%", answer claims "sales dropped and declined"
    contradictory_answer = "Apple net sales dropped and declined in fiscal 2024."
    findings = detector.detect_directional_contradictions(contradictory_answer, [sample_chunk_apple])
    assert len(findings) > 0
    assert findings[0].finding_type == HallucinationType.CONTRADICTION
    assert findings[0].severity == Severity.CRITICAL


def test_evaluation_graph_and_rag_evaluator(sample_chunk_apple: DocumentChunk):
    """Test full evaluation pipeline DAG and overall pass/fail scoring."""
    evaluator = RAGEvaluator()

    q = RAGQuery(query_id="q1", query_text="What was Apple's total net sales in 2024?", ticker="AAPL")
    citation = Citation(chunk_id=sample_chunk_apple.chunk_id, quote="$391.0 billion")

    # High quality response
    resp_good = RAGResponse(
        query_id="q1",
        generated_answer="Apple reported net sales of $391.0 billion for fiscal year 2024.",
        retrieved_chunks=[sample_chunk_apple],
        citations=[citation],
    )
    metrics_good = evaluator.evaluate(q, resp_good)
    assert metrics_good.faithfulness_score >= 0.80
    assert metrics_good.overall_pass is True
    assert metrics_good.hallucination_rate == 0.0

    # Distorted response
    resp_bad = RAGResponse(
        query_id="q1",
        generated_answer="Apple reported total net sales of $550.0 billion, which dropped significantly.",
        retrieved_chunks=[sample_chunk_apple],
        citations=[],
    )
    metrics_bad = evaluator.evaluate(q, resp_bad)
    assert metrics_bad.faithfulness_score < 0.80
    assert metrics_bad.overall_pass is False
    assert len(metrics_bad.findings) > 0


def test_financial_risk_engine(sample_returns: np.ndarray):
    """Test Sharpe, Sortino, Max Drawdown, Calmar, and VaR metrics."""
    engine = FinancialRiskEngine(risk_free_rate=0.045)
    metrics = engine.compute_all_metrics(sample_returns)

    assert isinstance(metrics.sharpe_ratio, float)
    assert isinstance(metrics.sortino_ratio, float)
    assert 0.0 <= metrics.max_drawdown <= 1.0
    assert metrics.annualized_volatility > 0.0
    assert metrics.var_95 >= 0.0
    assert metrics.cvar_95 >= metrics.var_95


def test_monte_carlo_stress_tester(sample_returns: np.ndarray):
    """Verify Monte Carlo GBM trajectory simulation and stress results."""
    tester = MonteCarloStressTester(random_seed=42)
    res = tester.run_stress_test(sample_returns, n_simulations=200, horizon_days=20)

    assert res.n_simulations == 200
    assert res.horizon_steps == 20
    assert res.p5_terminal <= res.p50_terminal <= res.p95_terminal
    assert 0.0 <= res.worst_case_drawdown <= 1.0

    # Test hallucination stress impact simulation
    breach_rate = tester.stress_test_hallucination_impact(
        base_financial_estimates=[100.0, 120.0, 110.0],
        hallucination_probability=0.20,
        n_simulations=100,
    )
    assert 0.0 <= breach_rate <= 1.0


def test_telemetry_collector(sample_chunk_apple: DocumentChunk, sample_returns: np.ndarray):
    """Test telemetry record aggregation and risk alert generation."""
    collector = RiskTelemetryCollector()
    q = RAGQuery(query_id="q1", query_text="What was Services revenue?", ticker="AAPL")
    resp = RAGResponse(
        query_id="q1",
        generated_answer="Services revenue reached an all-time record of $96.2 billion.",
        retrieved_chunks=[sample_chunk_apple],
        citations=[Citation(chunk_id=sample_chunk_apple.chunk_id, quote="$96.2 billion")],
    )
    m = collector.evaluator.evaluate(q, resp)
    collector.record_evaluation(m)

    rep = collector.generate_report(returns=sample_returns)
    assert rep.total_evaluations == 1
    assert rep.avg_faithfulness >= 0.80
    assert rep.risk_metrics is not None
    assert rep.monte_carlo_result is not None


def test_cli_subcommands(cli_runner: CliRunner, sample_chunk_apple: DocumentChunk):
    """Test CLI commands run and exit with code 0."""
    # Help
    r_help = cli_runner.invoke(cli, ["--help"])
    assert r_help.exit_code == 0
    assert "Production RAG Evaluation" in r_help.output

    # Evaluate
    r_eval = cli_runner.invoke(
        cli,
        [
            "evaluate",
            "-q", "What were net sales?",
            "-a", "Net sales were $391.0 billion.",
            "-c", sample_chunk_apple.text,
            "--json-out",
        ],
    )
    assert r_eval.exit_code == 0
    assert "faithfulness_score" in r_eval.output

    # Risk metrics
    r_risk = cli_runner.invoke(cli, ["risk-metrics", "--json-out"])
    assert r_risk.exit_code == 0
    assert "sharpe_ratio" in r_risk.output

    # Stress test
    r_stress = cli_runner.invoke(cli, ["stress-test", "-n", "100", "-h", "10", "--json-out"])
    assert r_stress.exit_code == 0
    assert "p5_terminal" in r_stress.output

    # Telemetry
    r_tel = cli_runner.invoke(cli, ["telemetry", "--json-out"])
    assert r_tel.exit_code == 0
    assert "total_evaluations" in r_tel.output
