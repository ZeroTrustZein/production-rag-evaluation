"""Comprehensive CLI coverage and entrypoint tests.

Covers:
- Helper functions parsing errors and file resolution (_read_text_or_file, _parse_float_array, _parse_targets)
- Rich rendering branches for target checks and hallucination findings in evaluate
- Batch evaluate with empty files, list contexts, and mixed targets
- Risk metrics with benchmark comparison table
- Telemetry drift check table with baseline and recent means
- Cache stats rich table output
- __main__.py module entrypoint
"""

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from production_rag_evaluation.cli.main import (
    _parse_float_array,
    _parse_targets,
    _read_text_or_file,
    cli,
)
from production_rag_evaluation.models.schemas import DocumentChunk


class TestCLIHelpersAndEdgeCases:
    """Test helper functions and argument parsing edge cases."""

    def test_read_text_or_file_nonexistent_at_file(self):
        with pytest.raises(click.BadParameter, match="does not exist"):
            _read_text_or_file("@non_existent_file_path_12345.txt")

    def test_read_text_or_file_existing_file_without_at_prefix(self, tmp_path: Path):
        f = tmp_path / "sample.txt"
        f.write_text("file content direct", encoding="utf-8")
        res = _read_text_or_file(str(f))
        assert res == "file content direct"

    def test_parse_float_array_invalid_number(self):
        with pytest.raises(click.BadParameter, match="Invalid floating-point number"):
            _parse_float_array("0.01, not_a_number, 0.05")

    def test_parse_targets_invalid_values(self):
        with pytest.raises(
            click.BadParameter, match="Invalid target numerical value in 'revenue=invalid'"
        ):
            _parse_targets(("revenue=invalid",))

        with pytest.raises(
            click.BadParameter, match="Invalid target numerical value in 'invalid_float'"
        ):
            _parse_targets(("invalid_float",))


class TestCLIRichOutputsAndTables:
    """Test rich console output branches in CLI commands."""

    def test_evaluate_rich_with_targets_matched_and_missed(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk
    ):
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                "-q",
                "What was Apple's net sales and profit?",
                "-a",
                "Apple reported net sales of $391.0 billion.",
                "-c",
                sample_chunk_apple.text,
                "-t",
                "AAPL",
                "-T",
                "net_sales=391000000000",
                "-T",
                "profit=50000000000",
            ],
        )
        assert result.exit_code == 0
        assert "net_sales" in result.output
        assert "profit" in result.output
        assert "MATCHED" in result.output
        assert "MISSED" in result.output

    def test_evaluate_rich_with_hallucination_findings_and_impact(
        self, cli_runner: CliRunner, sample_chunk_apple: DocumentChunk
    ):
        from production_rag_evaluation.models.schemas import (
            EvaluationMetrics,
            HallucinationFinding,
            HallucinationType,
            Severity,
        )

        mock_finding = HallucinationFinding(
            finding_type=HallucinationType.CONTRADICTION,
            severity=Severity.CRITICAL,
            claim_text="Revenue collapsed.",
            reference_text="Revenue grew.",
            confidence=0.95,
            explanation="Asserts decline when source documents growth.",
            impact_on_risk="High portfolio volatility risk.",
        )
        mock_metrics = EvaluationMetrics(
            faithfulness_score=0.4,
            answer_relevance_score=0.5,
            context_recall_score=0.5,
            context_precision_score=0.5,
            citation_grounding_score=0.5,
            hallucination_rate=0.6,
            overall_pass=False,
            findings=[mock_finding],
        )
        with patch(
            "production_rag_evaluation.cli.main.RAGEvaluator.evaluate", return_value=mock_metrics
        ):
            result = cli_runner.invoke(
                cli,
                [
                    "evaluate",
                    "-q",
                    "What was Apple's net sales and trend?",
                    "-a",
                    "Apple reported net sales declined to $10.0 billion.",
                    "-c",
                    sample_chunk_apple.text,
                    "-t",
                    "AAPL",
                ],
            )
            assert result.exit_code == 0
            assert "Hallucination Findings Detected" in result.output
            assert "Financial Risk Impact:" in result.output

    def test_batch_evaluate_empty_input_file_error(self, cli_runner: CliRunner, tmp_path: Path):
        empty_json = tmp_path / "empty.json"
        empty_json.write_text("[]", encoding="utf-8")
        result = cli_runner.invoke(cli, ["batch-evaluate", "-i", str(empty_json)])
        assert result.exit_code != 0
        assert "Input file contains no evaluation items" in result.output

    def test_batch_evaluate_context_list_and_mixed_targets(
        self, cli_runner: CliRunner, tmp_path: Path
    ):
        input_data = [
            {
                "query": "What was net sales?",
                "answer": "Net sales was $391.0 billion.",
                "context": ["Apple FY2024 net sales was $391.0 billion.", "Other detail here."],
                "ticker": "AAPL",
                "period": "FY2024",
                "targets": [
                    {"metric_name": "net_sales", "target_value": 391e9},
                    391e9,
                ],
            }
        ]
        batch_file = tmp_path / "batch_mixed.json"
        batch_file.write_text(json.dumps(input_data), encoding="utf-8")
        out_file = tmp_path / "batch_out.json"

        result = cli_runner.invoke(
            cli,
            ["batch-evaluate", "-i", str(batch_file), "-o", str(out_file), "-m", "FAST"],
        )
        assert result.exit_code == 0
        assert out_file.exists()

    def test_risk_metrics_rich_table_with_benchmark(self, cli_runner: CliRunner):
        series = "0.01, -0.005, 0.02, 0.015, -0.01, 0.025, 0.005"
        benchmark = "0.005, -0.002, 0.01, 0.008, -0.005, 0.015, 0.002"
        result = cli_runner.invoke(
            cli,
            ["risk-metrics", "--returns", series, "--benchmark", benchmark],
        )
        assert result.exit_code == 0
        assert "Benchmark Beta" in result.output
        assert "Jensen's Alpha" in result.output
        assert "Information Ratio" in result.output

    def test_telemetry_rich_table_with_drift_check_and_means(
        self, cli_runner: CliRunner, tmp_path: Path
    ):
        # Create history file with 30 items
        items = []
        for i in range(20):
            items.append(
                {
                    "faithfulness_score": 0.95 + (i % 3) * 0.01,
                    "answer_relevance_score": 0.9,
                    "context_recall_score": 0.9,
                    "context_precision_score": 0.9,
                    "citation_grounding_score": 0.9,
                    "hallucination_rate": 0.0,
                    "overall_pass": True,
                }
            )
        for i in range(10):
            items.append(
                {
                    "faithfulness_score": 0.50 + (i % 3) * 0.01,
                    "answer_relevance_score": 0.5,
                    "context_recall_score": 0.5,
                    "context_precision_score": 0.5,
                    "citation_grounding_score": 0.5,
                    "hallucination_rate": 0.5,
                    "overall_pass": False,
                }
            )

        history_file = tmp_path / "history_drift.json"
        history_file.write_text(json.dumps(items), encoding="utf-8")

        result = cli_runner.invoke(
            cli,
            [
                "telemetry",
                "--history-file",
                str(history_file),
                "--drift-metric",
                "faithfulness_score",
            ],
        )
        assert result.exit_code == 0
        assert "Baseline Mean" in result.output
        assert "Recent Mean" in result.output

    def test_cache_stats_rich_table(self, cli_runner: CliRunner):
        result = cli_runner.invoke(cli, ["cache", "stats"])
        assert result.exit_code == 0
        assert "Cached Entries" in result.output
        assert "Hit Rate" in result.output
        assert "Estimated Latency Saved" in result.output


class TestMainEntryPoint:
    """Test python -m production_rag_evaluation entrypoint."""

    def test_main_module_execution(self):
        test_args = ["production_rag_evaluation", "--help"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("production_rag_evaluation", run_name="__main__")
            assert exc_info.value.code == 0

    def test_feature_store_export_empty_entities_error(self, cli_runner: CliRunner, tmp_path: Path):
        fs_path = tmp_path / "fs"
        out_f = tmp_path / "out.jsonl"
        res = cli_runner.invoke(
            cli,
            [
                "feature-store",
                "export",
                "--path",
                str(fs_path),
                "--entities",
                ", , ",
                "--output",
                str(out_f),
            ],
        )
        assert res.exit_code != 0
        assert "No valid entity IDs provided" in res.output

    def test_feature_store_drift_feature_without_group_error(
        self, cli_runner: CliRunner, tmp_path: Path
    ):
        fs_path = tmp_path / "fs"
        res = cli_runner.invoke(
            cli,
            [
                "feature-store",
                "drift",
                "--path",
                str(fs_path),
                "--feature",
                "query_length",
            ],
        )
        assert res.exit_code != 0
        assert "requires specifying --group" in res.output
