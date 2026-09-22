"""Comprehensive test suite for CLI interface and command workflows."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from production_rag_evaluation.cli.main import cli
from production_rag_evaluation.models.schemas import DocumentChunk


class TestCLIEvaluateCommands:
    """Validate single query evaluation CLI operations, flags, and formatting."""

    def test_evaluate_basic_rich_output(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk
    ) -> None:
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "-q",
                "What was Apple's total net sales?",
                "-a",
                "Apple reported net sales of $391.0 billion.",
                "-c",
                sample_chunk_apple.text,
                "-t",
                "AAPL",
            ],
        )
        assert result.exit_code == 0
        assert "RAG Evaluation Report" in result.output
        assert "Faithfulness" in result.output
        assert "Overall Pass" in result.output

    def test_evaluate_json_output(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk
    ) -> None:
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "-q",
                "What was Apple's total net sales?",
                "-a",
                "Apple reported net sales of $391.0 billion.",
                "-c",
                sample_chunk_apple.text,
                "--json-out",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "faithfulness_score" in data
        assert "overall_pass" in data
        assert data["overall_pass"] is True

    def test_evaluate_with_modes(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk
    ) -> None:
        for mode in ["FAST", "THOROUGH", "FINANCIAL_STRICT", "AUDIT"]:
            result = cli_runner.invoke(
                cli,
                [
                    "evaluate",
                    "-q",
                    f"What was gross margin in mode {mode}?",
                    "-a",
                    "Gross margin was 46.2%.",
                    "-c",
                    sample_chunk_apple.text,
                    "--mode",
                    mode,
                    "--no-cache",
                    "--json-out",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["metadata"]["evaluation_mode"] == mode

    def test_evaluate_with_numerical_targets(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk
    ) -> None:
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "-q",
                "What were net sales and gross margin?",
                "-a",
                "Net sales were $391.0 billion and gross margin was 46.2%.",
                "-c",
                sample_chunk_apple.text,
                "-T",
                "net_sales=391.0e9",
                "-T",
                "0.462",
                "--json-out",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        checks = data["metadata"].get("target_checks", {})
        assert "net_sales" in checks
        assert checks["net_sales"]["matched"] is True

    def test_evaluate_output_to_file(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk, tmp_path: Path
    ) -> None:
        out_file = tmp_path / "eval_result.json"
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "-q",
                "What was Apple's total net sales?",
                "-a",
                "Apple reported net sales of $391.0 billion.",
                "-c",
                sample_chunk_apple.text,
                "-o",
                str(out_file),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        content = json.loads(out_file.read_text(encoding="utf-8"))
        assert content["faithfulness_score"] > 0.8

    def test_evaluate_read_from_file_prefix(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        ctx_file = tmp_path / "context.txt"
        ctx_file.write_text("Alphabet Inc. revenue for 2024 was $350 billion.", encoding="utf-8")

        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "-q",
                "What was Alphabet revenue?",
                "-a",
                "Alphabet revenue was $350 billion.",
                "-c",
                f"@{ctx_file}",
                "-t",
                "GOOGL",
                "--json-out",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["overall_pass"] is True


class TestCLIBatchEvaluateCommands:
    """Validate batch evaluation CLI operations across JSON and JSONL datasets."""

    def test_batch_evaluate_from_json(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk, tmp_path: Path
    ) -> None:
        dataset = [
            {
                "query": "What were net sales in 2024?",
                "answer": "Net sales were $391.0 billion.",
                "context": sample_chunk_apple.text,
                "ticker": "AAPL",
            },
            {
                "query": "What was gross margin?",
                "answer": "Gross margin dropped significantly to 15%.",
                "context": sample_chunk_apple.text,
                "ticker": "AAPL",
            },
        ]
        input_path = tmp_path / "batch.json"
        input_path.write_text(json.dumps(dataset), encoding="utf-8")

        result = cli_runner.invoke(
            cli,
            ["batch-evaluate", "-i", str(input_path), "--batch-id", "test-batch-001"],
        )
        assert result.exit_code == 0
        assert "Batch Evaluation Results (test-batch-001)" in result.output
        assert "Total Queries:" in result.output
        assert "Pass Rate:" in result.output

    def test_batch_evaluate_from_jsonl_with_output(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk, tmp_path: Path
    ) -> None:
        lines = [
            json.dumps(
                {
                    "query_id": "q-101",
                    "query": "What were net sales?",
                    "answer": "Net sales were $391.0 billion.",
                    "context": sample_chunk_apple.text,
                }
            ),
            json.dumps(
                {
                    "query_id": "q-102",
                    "query": "What was total cost?",
                    "answer": "Total cost was $210 billion.",
                    "context": sample_chunk_apple.text,
                }
            ),
        ]
        input_path = tmp_path / "batch.jsonl"
        input_path.write_text("\n".join(lines), encoding="utf-8")

        out_path = tmp_path / "summary.json"
        result = cli_runner.invoke(
            cli,
            [
                "batch-evaluate",
                "-i",
                str(input_path),
                "-o",
                str(out_path),
                "--json-out",
            ],
        )
        assert result.exit_code == 0
        assert out_path.exists()
        summary = json.loads(out_path.read_text(encoding="utf-8"))
        assert summary["total_queries"] == 2
        assert "pass_rate" in summary


class TestCLIAuditCommands:
    """Validate fine-grained hallucination audit command operations."""

    def test_audit_clean_text(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk
    ) -> None:
        result = cli_runner.invoke(
            cli,
            [
                "audit",
                "-a",
                "Apple reported net sales of $391.0 billion.",
                "-c",
                sample_chunk_apple.text,
                "-t",
                "AAPL",
            ],
        )
        assert result.exit_code == 0
        assert "Audit Clean" in result.output

    def test_audit_detects_distortions_table(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk
    ) -> None:
        result = cli_runner.invoke(
            cli,
            [
                "audit",
                "-a",
                "MSFT reported net sales dropped and declined to $450.0 billion.",
                "-c",
                sample_chunk_apple.text,
                "-t",
                "AAPL",
            ],
        )
        assert result.exit_code == 0
        assert "Financial Hallucination Audit Findings" in result.output
        assert "ENTITY_SWAP" in result.output or "NUMERICAL_DISTORTION" in result.output

    def test_detect_hallucinations_alias_json(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk
    ) -> None:
        result = cli_runner.invoke(
            cli,
            [
                "detect-hallucinations",
                "-a",
                "Apple net sales plummeted to $100 billion.",
                "-c",
                sample_chunk_apple.text,
                "--json-out",
            ],
        )
        assert result.exit_code == 0
        findings = json.loads(result.output)
        assert isinstance(findings, list)
        assert len(findings) > 0


class TestCLIRiskMetricsCommands:
    """Validate financial risk calculation CLI operations."""

    def test_risk_metrics_default_returns(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["risk-metrics"])
        assert result.exit_code == 0
        assert "Quantitative Financial Risk" in result.output
        assert "Sharpe Ratio" in result.output
        assert "Sortino Ratio" in result.output
        assert "Max Drawdown" in result.output

    def test_risk_metrics_custom_series_and_benchmark(self, cli_runner: CliRunner) -> None:
        ret_str = "0.012, -0.008, 0.015, -0.003, 0.009, 0.004, -0.002"
        bench_str = "0.008, -0.005, 0.010, -0.002, 0.006, 0.003, -0.001"
        result = cli_runner.invoke(
            cli,
            [
                "risk-metrics",
                "-r",
                ret_str,
                "-b",
                bench_str,
                "--risk-free",
                "0.05",
                "--json-out",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "sharpe_ratio" in data
        assert "beta" in data
        assert "alpha" in data
        assert data["beta"] is not None


class TestCLIStressTestingCommands:
    """Validate Monte Carlo stress testing CLI operations."""

    def test_stress_test_rich_table_output(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            cli,
            ["stress-test", "-n", "150", "-h", "15", "--shock", "1.8"],
        )
        assert result.exit_code == 0
        assert "Monte Carlo Stress Simulation" in result.output
        assert "5th Percentile Terminal" in result.output
        assert "Worst Case Drawdown" in result.output

    def test_stress_test_preset_scenario_json(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            cli,
            [
                "stress-test",
                "--scenario",
                "LIQUIDITY_CRUNCH",
                "-n",
                "100",
                "-h",
                "10",
                "--seed",
                "42",
                "--json-out",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["scenario_type"] == "LIQUIDITY_CRUNCH"
        assert "worst_case_drawdown" in data
        assert "p5_terminal" in data

    def test_stress_test_fat_tailed_model(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            cli,
            ["stress-test", "--model", "fat-tailed", "-n", "100", "-h", "10", "--json-out"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["scenario_type"] == "BLACK_SWAN"


class TestCLITelemetryAndCacheCommands:
    """Validate telemetry status, drift detection, and semantic cache CLI operations."""

    def test_telemetry_status_only(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["telemetry", "--status-only"])
        assert result.exit_code == 0
        assert result.output.strip() in ["HEALTHY", "DEGRADED", "UNHEALTHY"]

    def test_telemetry_with_drift_check(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            cli,
            ["telemetry", "--drift-metric", "faithfulness_score"],
        )
        assert result.exit_code == 0
        assert "Metric Drift Check" in result.output
        assert "faithfulness_score" in result.output
        assert "p-value" in result.output

    def test_telemetry_with_history_file(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        hist_file = tmp_path / "history.json"
        metrics_record = {
            "faithfulness_score": 0.92,
            "answer_relevance_score": 0.88,
            "context_recall_score": 0.85,
            "context_precision_score": 0.85,
            "citation_grounding_score": 0.90,
            "hallucination_rate": 0.05,
            "overall_pass": True,
        }
        hist_file.write_text(json.dumps([metrics_record]), encoding="utf-8")

        result = cli_runner.invoke(
            cli,
            ["telemetry", "--history-file", str(hist_file), "--json-out"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_evaluations"] == 1
        assert data["avg_faithfulness"] == 0.92

    def test_cache_stats_and_clear(self, cli_runner: CliRunner) -> None:
        res_stats = cli_runner.invoke(cli, ["cache", "stats", "--json-out"])
        assert res_stats.exit_code == 0
        stats = json.loads(res_stats.output)
        assert "total_entries" in stats
        assert "hit_rate" in stats

        res_clear = cli_runner.invoke(cli, ["cache", "clear"])
        assert res_clear.exit_code == 0
        assert "Semantic cache successfully cleared" in res_clear.output
