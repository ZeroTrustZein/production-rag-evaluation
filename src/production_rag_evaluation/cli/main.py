"""Command line interface for production RAG evaluation and financial risk metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from production_rag_evaluation.agents.models import (
    AgentWorkflowConfig,
    WorkflowStatus,
)
from production_rag_evaluation.agents.orchestrator import create_default_orchestrator
from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.features.drift import FeatureDriftDetector
from production_rag_evaluation.features.handlers import (
    EvaluationFeatureHandler,
    FeatureCompactionHandler,
    FeatureExportHandler,
)
from production_rag_evaluation.features.models import ExportFormat, FeatureGroup
from production_rag_evaluation.features.registry import FeatureRegistry
from production_rag_evaluation.features.store import ParquetFeatureStore
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.mcp.models import MCPConfig, MCPTransport
from production_rag_evaluation.mcp.prompts import get_default_prompt_registry
from production_rag_evaluation.mcp.resources import get_default_resource_registry
from production_rag_evaluation.mcp.server import create_mcp_server
from production_rag_evaluation.mcp.tools import get_default_tool_registry
from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    DocumentType,
    EvaluationMetrics,
    EvaluationMode,
    NumericalTarget,
    RAGQuery,
    RAGResponse,
    StressScenarioType,
    StressTestConfig,
    SystemHealthStatus,
)

console = Console()

# Singleton cache instance for CLI operations where caching is enabled
_GLOBAL_CACHE = SemanticCache(similarity_threshold=0.90, max_entries=5000)


def _read_text_or_file(value: str) -> str:
    """Read value directly or from file path if prefixed with @ or existing file."""
    if value.startswith("@"):
        path = Path(value[1:])
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        raise click.BadParameter(f"File '{path}' does not exist.")
    path = Path(value)
    if path.is_file() and value.endswith((".txt", ".json", ".md", ".csv", ".jsonl")):
        return path.read_text(encoding="utf-8").strip()
    return value


def _parse_float_array(value: Optional[str]) -> Optional[np.ndarray]:
    """Parse comma-separated decimal values or read from file."""
    if not value:
        return None
    content = _read_text_or_file(value)
    items = [x.strip() for x in content.replace("\n", ",").split(",") if x.strip()]
    try:
        return np.array([float(x) for x in items], dtype=np.float64)
    except ValueError as e:
        raise click.BadParameter(f"Invalid floating-point number in series: {e}") from e


def _parse_targets(targets: Tuple[str, ...]) -> Tuple[List[NumericalTarget], List[float]]:
    """Parse list of target strings into structured targets and raw float targets."""
    structured: List[NumericalTarget] = []
    raw: List[float] = []

    for t in targets:
        t_str = t.strip()
        if "=" in t_str:
            name, val_part = t_str.split("=", 1)
            try:
                val = float(val_part.strip())
                structured.append(NumericalTarget(metric_name=name.strip(), target_value=val))
            except ValueError as e:
                raise click.BadParameter(f"Invalid target numerical value in '{t}': {e}") from e
        else:
            try:
                raw.append(float(t_str))
            except ValueError as e:
                raise click.BadParameter(f"Invalid target numerical value in '{t}': {e}") from e

    return structured, raw


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Production RAG Evaluation & Hallucination Detection Pipeline."""
    pass


# -------------------------------------------------------------------------
# 1. EVALUATE COMMAND (Single Query)
# -------------------------------------------------------------------------
@cli.command(name="evaluate")
@click.option("--query", "-q", required=True, help="Input financial question/query or @file.")
@click.option("--answer", "-a", required=True, help="Generated answer to evaluate or @file.")
@click.option(
    "--context", "-c", required=True, help="Retrieved ground-truth text context or @file."
)
@click.option("--ticker", "-t", default="AAPL", help="Stock ticker symbol.")
@click.option("--period", "-p", default="FY2024", help="Financial reporting period (e.g. Q3 2024).")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["FAST", "THOROUGH", "FINANCIAL_STRICT", "AUDIT"], case_sensitive=False),
    default="FINANCIAL_STRICT",
    help="Evaluation execution mode.",
)
@click.option(
    "--target",
    "-T",
    multiple=True,
    help="Expected numerical target (e.g., 'net_sales=391.0e9' or '391.0e9'). Can be repeated.",
)
@click.option("--no-cache", is_flag=True, default=False, help="Disable semantic evaluation cache.")
@click.option(
    "--output", "-o", type=click.Path(writable=True), help="Path to write evaluation JSON report."
)
@click.option(
    "--feature-store-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional feature store path to stream evaluation features into.",
)
@click.option("--json-out", is_flag=True, help="Output raw JSON results.")
def evaluate_cmd(
    query: str,
    answer: str,
    context: str,
    ticker: str,
    period: str,
    mode: str,
    target: Tuple[str, ...],
    no_cache: bool,
    output: Optional[str],
    feature_store_path: Optional[Path],
    json_out: bool,
) -> None:
    """Evaluate a single RAG query/response against financial context."""
    query_text = _read_text_or_file(query)
    answer_text = _read_text_or_file(answer)
    context_text = _read_text_or_file(context)

    structured_targets, raw_targets = _parse_targets(target)

    eval_mode = EvaluationMode[mode.upper()]
    evaluator = RAGEvaluator(enable_cache=not no_cache)
    if not no_cache:
        evaluator.cache = _GLOBAL_CACHE

    q = RAGQuery(
        query_id="cli-q1",
        query_text=query_text,
        ticker=ticker,
        period=period,
        structured_targets=structured_targets,
        expected_numerical_targets=raw_targets,
    )
    chunk = DocumentChunk(
        chunk_id="cli-c1",
        text=context_text,
        doc_type=DocumentType.SEC_10K,
        ticker=ticker,
        period=period,
    )
    resp = RAGResponse(
        query_id="cli-q1",
        generated_answer=answer_text,
        retrieved_chunks=[chunk],
    )

    metrics = evaluator.evaluate(q, resp, mode=eval_mode)

    if feature_store_path:
        fs = ParquetFeatureStore(base_path=feature_store_path)
        handler = EvaluationFeatureHandler(auto_flush=True)
        handler.pipeline.store = fs
        handler.handle_evaluation(query=q, response=resp, metrics=metrics, flush_immediately=True)

    if output:
        Path(output).write_text(metrics.model_dump_json(indent=2), encoding="utf-8")

    if json_out:
        click.echo(metrics.model_dump_json(indent=2))
        return

    # Rich formatted report
    status_str = (
        "[bold green]PASS[/bold green]" if metrics.overall_pass else "[bold red]FAIL[/bold red]"
    )
    console.print(
        Panel.fit(
            f"[bold]RAG Evaluation Report[/bold] | Ticker: [cyan]{ticker}[/cyan] | Mode: [magenta]{eval_mode.value}[/magenta] | Status: {status_str}",
            border_style="blue",
        )
    )

    table = Table(title="Evaluation Metrics Breakdown")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="bold magenta")
    table.add_column("Status", style="green")

    def status_label(val: float, threshold: float = 0.75) -> str:
        return "[green]PASS[/green]" if val >= threshold else "[red]FAIL[/red]"

    table.add_row(
        "Faithfulness",
        f"{metrics.faithfulness_score:.2%}",
        status_label(metrics.faithfulness_score),
    )
    table.add_row(
        "Answer Relevance",
        f"{metrics.answer_relevance_score:.2%}",
        status_label(metrics.answer_relevance_score, 0.60),
    )
    table.add_row(
        "Context Recall",
        f"{metrics.context_recall_score:.2%}",
        status_label(metrics.context_recall_score, 0.60),
    )
    table.add_row(
        "Context Precision",
        f"{metrics.context_precision_score:.2%}",
        status_label(metrics.context_precision_score, 0.60),
    )
    table.add_row(
        "Citation Grounding",
        f"{metrics.citation_grounding_score:.2%}",
        status_label(metrics.citation_grounding_score, 0.70),
    )
    table.add_row(
        "Hallucination Rate",
        f"{metrics.hallucination_rate:.2%}",
        "[green]LOW[/green]" if metrics.hallucination_rate < 0.2 else "[red]HIGH[/red]",
    )
    table.add_row(
        "Overall Pass",
        str(metrics.overall_pass),
        "[bold green]YES[/bold green]" if metrics.overall_pass else "[bold red]NO[/bold red]",
    )
    table.add_row("Evaluation Latency", f"{metrics.evaluation_latency_ms:.1f} ms", "OK")

    console.print(table)

    # Structured target checks
    target_checks = metrics.metadata.get("target_checks", {})
    if target_checks:
        tgt_table = Table(title="Structured Target Verification")
        tgt_table.add_column("Target Metric / Val", style="cyan")
        tgt_table.add_column("Expected", style="yellow")
        tgt_table.add_column("Found In Answer", style="magenta")
        tgt_table.add_column("Matched", style="bold")

        for key, details in target_checks.items():
            matched = details.get("matched", False)
            match_str = "[green]MATCHED[/green]" if matched else "[red]MISSED[/red]"
            tgt_table.add_row(
                key,
                str(details.get("expected", "")),
                str(details.get("found", "")),
                match_str,
            )
        console.print(tgt_table)

    # Hallucination findings
    if metrics.findings:
        console.print("\n[bold yellow]Hallucination Findings Detected:[/bold yellow]")
        for f in metrics.findings:
            sev_color = "red" if f.severity.value in ("HIGH", "CRITICAL") else "yellow"
            console.print(
                f" - [[{sev_color}]{f.severity.value}[/{sev_color}]] [cyan]{f.finding_type.value}[/cyan]: {f.claim_text}"
            )
            console.print(f"   Reason: {f.explanation}")
            if f.impact_on_risk:
                console.print(f"   Financial Risk Impact: [italic]{f.impact_on_risk}[/italic]")
    else:
        console.print("\n[green]Zero hallucination findings detected.[/green]")


# -------------------------------------------------------------------------
# 2. BATCH EVALUATE COMMAND
# -------------------------------------------------------------------------
@cli.command(name="batch-evaluate")
@click.option(
    "--input",
    "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input JSON or JSONL file with queries.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    help="Path to write batch evaluation JSON summary.",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["FAST", "THOROUGH", "FINANCIAL_STRICT", "AUDIT"], case_sensitive=False),
    default="FINANCIAL_STRICT",
    help="Evaluation execution mode.",
)
@click.option("--batch-id", default=None, help="Custom identifier for the batch run.")
@click.option("--no-cache", is_flag=True, default=False, help="Disable semantic cache for batch.")
@click.option(
    "--feature-store-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional feature store path to stream evaluation features into.",
)
@click.option("--json-out", is_flag=True, help="Output raw JSON summary.")
def batch_evaluate_cmd(
    input_file: str,
    output: Optional[str],
    mode: str,
    batch_id: Optional[str],
    no_cache: bool,
    feature_store_path: Optional[Path],
    json_out: bool,
) -> None:
    """Run batch evaluation across multiple queries and context passages from a file."""
    path = Path(input_file)
    content = path.read_text(encoding="utf-8").strip()

    items: List[Dict[str, Any]] = []
    if path.suffix == ".jsonl" or (not content.startswith("[") and "\n" in content):
        for line in content.splitlines():
            line_str = line.strip()
            if line_str:
                items.append(json.loads(line_str))
    else:
        parsed = json.loads(content)
        items = parsed if isinstance(parsed, list) else parsed.get("items", [])

    if not items:
        raise click.ClickException("Input file contains no evaluation items.")

    pairs: List[Tuple[RAGQuery, RAGResponse]] = []
    for idx, item in enumerate(items, start=1):
        q_text = item.get("query") or item.get("query_text", "")
        a_text = item.get("answer") or item.get("generated_answer", "")
        c_text = item.get("context") or item.get("chunk_text") or item.get("retrieved_chunks", "")
        if isinstance(c_text, list):
            c_text = "\n".join(str(c) for c in c_text)
        ticker = item.get("ticker", "AAPL")
        period = item.get("period", "FY2024")
        q_id = item.get("query_id", f"batch-q{idx}")

        # Parse targets if present
        raw_targets = item.get("targets", [])
        struct_targets: List[NumericalTarget] = []
        num_targets: List[float] = []
        for t in raw_targets:
            if isinstance(t, dict):
                struct_targets.append(NumericalTarget(**t))
            elif isinstance(t, (int, float)):
                num_targets.append(float(t))

        q = RAGQuery(
            query_id=q_id,
            query_text=q_text,
            ticker=ticker,
            period=period,
            structured_targets=struct_targets,
            expected_numerical_targets=num_targets,
        )
        chunk = DocumentChunk(
            chunk_id=f"chk-{q_id}",
            text=str(c_text),
            doc_type=DocumentType.SEC_10K,
            ticker=ticker,
            period=period,
        )
        resp = RAGResponse(
            query_id=q_id,
            generated_answer=a_text,
            retrieved_chunks=[chunk],
        )
        pairs.append((q, resp))

    eval_mode = EvaluationMode[mode.upper()]
    evaluator = RAGEvaluator(enable_cache=not no_cache)
    if not no_cache:
        evaluator.cache = _GLOBAL_CACHE

    summary = evaluator.evaluate_batch_summary(
        pairs, batch_id=batch_id, use_cache=not no_cache, mode=eval_mode
    )

    if feature_store_path:
        fs = ParquetFeatureStore(base_path=feature_store_path)
        handler = EvaluationFeatureHandler(auto_flush=True)
        handler.pipeline.store = fs
        handler.handle_batch_evaluations(
            pairs=pairs,
            metrics_list=summary.results,
            flush_immediately=True,
        )

    if output:
        Path(output).write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    if json_out:
        click.echo(summary.model_dump_json(indent=2))
        return

    # Rich summary display
    table = Table(title=f"Batch Evaluation Results ({summary.batch_id})")
    table.add_column("Query ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Faithfulness", style="magenta")
    table.add_column("Relevance", style="green")
    table.add_column("Findings", style="yellow")
    table.add_column("Latency (ms)", style="dim")

    for i, res in enumerate(summary.results, start=1):
        q_label = pairs[i - 1][0].query_id
        pass_str = "[green]PASS[/green]" if res.overall_pass else "[red]FAIL[/red]"
        table.add_row(
            q_label,
            pass_str,
            f"{res.faithfulness_score:.1%}",
            f"{res.answer_relevance_score:.1%}",
            str(len(res.findings)),
            f"{res.evaluation_latency_ms:.1f}",
        )

    console.print(table)

    summary_panel = Table.grid(padding=1)
    summary_panel.add_column(style="bold cyan")
    summary_panel.add_column(style="bold white")
    summary_panel.add_row("Total Queries:", str(summary.total_queries))
    summary_panel.add_row(
        "Passed / Failed:",
        f"[green]{summary.passed_queries}[/green] / [red]{summary.failed_queries}[/red]",
    )
    summary_panel.add_row("Pass Rate:", f"{summary.pass_rate:.1%}")
    summary_panel.add_row("Mean Faithfulness:", f"{summary.mean_faithfulness:.1%}")
    summary_panel.add_row("Mean Relevance:", f"{summary.mean_relevance:.1%}")
    summary_panel.add_row("Mean Latency:", f"{summary.mean_latency_ms:.1f} ms")
    summary_panel.add_row("P95 Latency:", f"{summary.p95_latency_ms:.1f} ms")

    console.print(
        Panel(summary_panel, title=f"Batch Summary - {summary.batch_id}", border_style="green")
    )


# -------------------------------------------------------------------------
# 3. AUDIT / DETECT-HALLUCINATIONS COMMAND
# -------------------------------------------------------------------------
def _run_audit(answer: str, context: str, ticker: str, tolerance: float, json_out: bool) -> None:
    answer_text = _read_text_or_file(answer)
    context_text = _read_text_or_file(context)

    detector = HallucinationDetector(numerical_tolerance_pct=tolerance)
    chunk = DocumentChunk(
        chunk_id="audit-c1",
        text=context_text,
        ticker=ticker,
        doc_type=DocumentType.SEC_10K,
    )

    findings = detector.analyze(answer_text, [chunk])

    if json_out:
        data = [f.model_dump() for f in findings]
        click.echo(json.dumps(data, indent=2))
        return

    if not findings:
        console.print(
            Panel(
                "[bold green]Audit Clean: No hallucinations or distortions detected.[/bold green]",
                border_style="green",
            )
        )
        return

    table = Table(title=f"Financial Hallucination Audit Findings ({ticker})")
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Severity", style="bold")
    table.add_column("Claim Text", style="magenta")
    table.add_column("Reference Text", style="dim")
    table.add_column("Explanation", style="yellow")

    for f in findings:
        sev_style = "red" if f.severity.value in ("HIGH", "CRITICAL") else "yellow"
        table.add_row(
            f.finding_type.value,
            f"[{sev_style}]{f.severity.value}[/{sev_style}]",
            f.claim_text,
            f.reference_text or "N/A",
            f.explanation,
        )

    console.print(table)


@cli.command(name="audit")
@click.option("--answer", "-a", required=True, help="Generated answer text or @file.")
@click.option(
    "--context", "-c", required=True, help="Ground truth reference context text or @file."
)
@click.option("--ticker", "-t", default="AAPL", help="Stock ticker symbol.")
@click.option("--tolerance", default=0.01, help="Numerical tolerance ratio (e.g. 0.01 for 1%).")
@click.option("--json-out", is_flag=True, help="Output raw JSON results.")
def audit_cmd(answer: str, context: str, ticker: str, tolerance: float, json_out: bool) -> None:
    """Run fine-grained financial hallucination audit on generated text."""
    _run_audit(answer, context, ticker, tolerance, json_out)


@cli.command(name="detect-hallucinations")
@click.option("--answer", "-a", required=True, help="Generated answer text or @file.")
@click.option(
    "--context", "-c", required=True, help="Ground truth reference context text or @file."
)
@click.option("--ticker", "-t", default="AAPL", help="Stock ticker symbol.")
@click.option("--tolerance", default=0.01, help="Numerical tolerance ratio (e.g. 0.01 for 1%).")
@click.option("--json-out", is_flag=True, help="Output raw JSON results.")
def detect_hallucinations_cmd(
    answer: str, context: str, ticker: str, tolerance: float, json_out: bool
) -> None:
    """Detect numerical distortions, contradictions, and entity swaps (alias for audit)."""
    _run_audit(answer, context, ticker, tolerance, json_out)


# -------------------------------------------------------------------------
# 4. RISK METRICS COMMAND
# -------------------------------------------------------------------------
@cli.command(name="risk-metrics")
@click.option(
    "--returns", "-r", default=None, help="Comma-separated decimal daily returns or @file."
)
@click.option("--benchmark", "-b", default=None, help="Comma-separated benchmark returns or @file.")
@click.option("--risk-free", default=0.045, help="Annualized risk-free rate.")
@click.option("--target-return", default=0.0, help="Target minimum acceptable return for Sortino.")
@click.option("--confidence", default=0.95, help="Primary VaR confidence level.")
@click.option("--json-out", is_flag=True, help="Output raw JSON results.")
def risk_metrics_cmd(
    returns: Optional[str],
    benchmark: Optional[str],
    risk_free: float,
    target_return: float,
    confidence: float,
    json_out: bool,
) -> None:
    """Calculate Sharpe, Sortino, Calmar, and Value-at-Risk indicators."""
    arr = _parse_float_array(returns)
    if arr is None or len(arr) == 0:
        # Default representative daily returns
        arr = np.array(
            [0.005, -0.003, 0.008, 0.002, -0.001, 0.006, 0.004, -0.002, 0.003, 0.001],
            dtype=np.float64,
        )

    bench_arr = _parse_float_array(benchmark)

    engine = FinancialRiskEngine(risk_free_rate=risk_free)
    metrics = engine.compute_all_metrics(
        arr,
        risk_free_rate=risk_free,
        benchmark_returns=bench_arr,
    )

    if json_out:
        click.echo(metrics.model_dump_json(indent=2))
        return

    table = Table(title="Quantitative Financial Risk Metrics")
    table.add_column("Indicator", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Annualized Return", f"{metrics.annualized_return:.2%}")
    table.add_row("Annualized Volatility", f"{metrics.annualized_volatility:.2%}")
    table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
    table.add_row("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")
    table.add_row("Max Drawdown", f"{metrics.max_drawdown:.2%}")
    table.add_row("1-Day 95% VaR", f"{metrics.var_95:.2%}")
    table.add_row("1-Day 95% CVaR / ES", f"{metrics.cvar_95:.2%}")
    if metrics.var_99 is not None:
        table.add_row("1-Day 99% VaR", f"{metrics.var_99:.2%}")
    if metrics.cvar_99 is not None:
        table.add_row("1-Day 99% CVaR / ES", f"{metrics.cvar_99:.2%}")

    if metrics.beta is not None and metrics.alpha is not None:
        table.add_row("Benchmark Beta", f"{metrics.beta:.2f}")
        table.add_row("Jensen's Alpha", f"{metrics.alpha:.2%}")
    if metrics.information_ratio is not None:
        table.add_row("Information Ratio", f"{metrics.information_ratio:.2f}")

    console.print(table)


# -------------------------------------------------------------------------
# 5. STRESS TEST COMMAND
# -------------------------------------------------------------------------
@cli.command(name="stress-test")
@click.option("--simulations", "-n", default=1000, help="Number of Monte Carlo paths.")
@click.option("--horizon", "-h", default=30, help="Forecast horizon in trading days.")
@click.option("--shock", "-s", default=1.5, help="Volatility shock multiplier.")
@click.option(
    "--scenario",
    type=click.Choice(
        [
            "BASELINE",
            "VOLATILITY_SPIKE",
            "LIQUIDITY_CRUNCH",
            "RATE_SHOCK",
            "BLACK_SWAN",
            "HALLUCINATION_CASCADE",
        ],
        case_sensitive=False,
    ),
    default="BASELINE",
    help="Macroeconomic stress scenario preset.",
)
@click.option(
    "--model",
    type=click.Choice(["gbm", "fat-tailed"], case_sensitive=False),
    default="gbm",
    help="Stochastic process model (Geometric Brownian Motion or Fat-Tailed Student-t).",
)
@click.option("--initial-value", default=100.0, help="Initial portfolio/asset value.")
@click.option("--max-drawdown", default=0.25, help="Maximum allowable drawdown before breach.")
@click.option("--seed", default=None, type=int, help="Random seed for reproducible trajectories.")
@click.option("--returns", "-r", default=None, help="Empirical returns series or @file.")
@click.option("--json-out", is_flag=True, help="Output raw JSON results.")
def stress_test_cmd(
    simulations: int,
    horizon: int,
    shock: float,
    scenario: str,
    model: str,
    initial_value: float,
    max_drawdown: float,
    seed: Optional[int],
    returns: Optional[str],
    json_out: bool,
) -> None:
    """Run Monte Carlo stress testing on simulated financial trajectories."""
    base_returns = _parse_float_array(returns)
    if base_returns is None or len(base_returns) == 0:
        base_returns = np.array(
            [0.002, -0.004, 0.001, -0.002, 0.005, 0.003, -0.001], dtype=np.float64
        )

    scenario_type = StressScenarioType[scenario.upper()]
    tester = MonteCarloStressTester(random_seed=seed)

    config = StressTestConfig(
        n_simulations=simulations,
        horizon_days=horizon,
        volatility_shock=shock,
        max_allowable_drawdown=max_drawdown,
        initial_value=initial_value,
        scenario_type=scenario_type,
        random_seed=seed,
    )

    if scenario_type != StressScenarioType.BASELINE:
        result = tester.run_scenario(base_returns, scenario_type=scenario_type, config=config)
    elif model.lower() == "fat-tailed":
        result = tester.run_scenario(
            base_returns, scenario_type=StressScenarioType.BLACK_SWAN, config=config
        )
    else:
        result = tester.run_stress_test(
            base_returns,
            n_simulations=simulations,
            horizon_days=horizon,
            volatility_shock=shock,
            max_allowable_drawdown=max_drawdown,
            config=config,
        )

    if json_out:
        click.echo(result.model_dump_json(indent=2))
        return

    table = Table(
        title=f"Monte Carlo Stress Simulation ({simulations} paths, {horizon}d, {scenario_type.value})"
    )
    table.add_column("Parameter / Outcome", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Scenario Type", result.scenario_type.value)
    table.add_row("Initial Portfolio Value", f"${initial_value:.2f}")
    table.add_row("5th Percentile Terminal ($)", f"${result.p5_terminal:.2f}")
    table.add_row("50th Percentile Terminal ($)", f"${result.p50_terminal:.2f}")
    table.add_row("95th Percentile Terminal ($)", f"${result.p95_terminal:.2f}")
    table.add_row("Mean Terminal ($)", f"${result.mean_terminal:.2f}")
    table.add_row("Standard Deviation", f"${result.std_terminal:.2f}")
    table.add_row("Worst Case Drawdown", f"{result.worst_case_drawdown:.2%}")
    table.add_row("Breach Probability", f"{result.breach_probability:.2%}")
    table.add_row(
        "Stress Test Passed", "[green]YES[/green]" if result.stress_pass else "[red]NO[/red]"
    )

    console.print(table)


# -------------------------------------------------------------------------
# 6. TELEMETRY COMMAND
# -------------------------------------------------------------------------
@cli.command(name="telemetry")
@click.option(
    "--returns", "-r", default=None, help="Comma-separated decimal daily returns or @file."
)
@click.option(
    "--history-file",
    type=click.Path(exists=True),
    help="Path to JSON file containing past EvaluationMetrics.",
)
@click.option(
    "--drift-metric",
    default=None,
    help="Detect statistical drift for given metric (e.g. faithfulness_score).",
)
@click.option("--status-only", is_flag=True, help="Print only system health status string.")
@click.option("--json-out", is_flag=True, help="Output raw JSON results.")
def telemetry_cmd(
    returns: Optional[str],
    history_file: Optional[str],
    drift_metric: Optional[str],
    status_only: bool,
    json_out: bool,
) -> None:
    """Show end-to-end telemetry status combining RAG quality and financial risk."""
    collector = RiskTelemetryCollector()

    if history_file:
        data = json.loads(Path(history_file).read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("results", [])
        for item in items:
            collector.record_evaluation(EvaluationMetrics(**item))
    else:
        # Default representative baseline record
        chunk = DocumentChunk(
            chunk_id="c1",
            text="Apple revenue was $85.8 billion, up 5% year-over-year.",
            ticker="AAPL",
        )
        q = RAGQuery(query_id="q1", query_text="What was Apple revenue?", ticker="AAPL")
        resp = RAGResponse(
            query_id="q1",
            generated_answer="Apple reported revenue of $85.8 billion.",
            retrieved_chunks=[chunk],
        )
        m = collector.evaluator.evaluate(q, resp)
        collector.record_evaluation(m)

    ret_arr = _parse_float_array(returns)
    if ret_arr is None and not history_file:
        ret_arr = np.array([0.004, -0.002, 0.003, 0.001, 0.005, -0.001, 0.002], dtype=np.float64)

    rep = collector.generate_report(returns=ret_arr)

    if status_only:
        click.echo(rep.health_status.value)
        return

    if json_out:
        click.echo(rep.model_dump_json(indent=2))
        return

    # Rich formatted report
    health_color = (
        "green"
        if rep.health_status == SystemHealthStatus.HEALTHY
        else "yellow"
        if rep.health_status == SystemHealthStatus.DEGRADED
        else "red"
    )

    console.print(
        Panel.fit(
            f"[bold]System Health Status:[/bold] [{health_color}]{rep.health_status.value}[/{health_color}]",
            border_style=health_color,
        )
    )

    table = Table(title="Production RAG & Financial Telemetry Report")
    table.add_column("Component", style="cyan")
    table.add_column("Telemetry Reading", style="bold green")

    table.add_row("Total Evaluations", str(rep.total_evaluations))
    table.add_row("Average Faithfulness", f"{rep.avg_faithfulness:.2%}")
    table.add_row("Average Hallucination Rate", f"{rep.avg_hallucination_rate:.2%}")
    table.add_row("Cache Hit Rate", f"{rep.cache_hit_rate:.2%}")
    table.add_row("Average Latency", f"{rep.avg_latency_ms:.1f} ms")

    if rep.risk_metrics:
        table.add_row("Portfolio Sharpe Ratio", f"{rep.risk_metrics.sharpe_ratio:.2f}")
        table.add_row("Portfolio Sortino Ratio", f"{rep.risk_metrics.sortino_ratio:.2f}")
        table.add_row("Portfolio Max Drawdown", f"{rep.risk_metrics.max_drawdown:.2%}")

    table.add_row(
        "Active Alerts",
        ", ".join(rep.alerts_triggered) if rep.alerts_triggered else "[green]None[/green]",
    )

    console.print(table)

    # Drift detection check if requested
    if drift_metric:
        drift_res = collector.detect_drift(metric=drift_metric)
        drift_table = Table(title=f"Metric Drift Check: {drift_metric}")
        drift_table.add_column("Property", style="cyan")
        drift_table.add_column("Value", style="magenta")
        drift_table.add_row(
            "Drift Detected",
            "[red]YES[/red]" if drift_res.get("drift_detected") else "[green]NO[/green]",
        )
        drift_table.add_row("p-value", str(drift_res.get("p_value")))
        if "baseline_mean" in drift_res:
            drift_table.add_row("Baseline Mean", f"{drift_res['baseline_mean']:.4f}")
            drift_table.add_row("Recent Mean", f"{drift_res['recent_mean']:.4f}")
        console.print(drift_table)


# -------------------------------------------------------------------------
# 7. CACHE MANAGEMENT COMMAND GROUP
# -------------------------------------------------------------------------
@cli.group(name="cache")
def cache_group() -> None:
    """Manage semantic evaluation cache."""
    pass


@cache_group.command(name="stats")
@click.option("--json-out", is_flag=True, help="Output raw JSON stats.")
def cache_stats_cmd(json_out: bool) -> None:
    """Display semantic cache metrics, capacity, and hit rates."""
    stats = _GLOBAL_CACHE.stats()
    if json_out:
        click.echo(json.dumps(stats, indent=2))
        return

    table = Table(title="Semantic Cache Performance Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Cached Entries", str(stats.get("total_entries", 0)))
    table.add_row("Max Capacity", str(stats.get("max_entries", 0)))
    table.add_row("Hit Count", str(stats.get("hit_count", 0)))
    table.add_row("Miss Count", str(stats.get("miss_count", 0)))
    hit_rate = stats.get("hit_rate", 0.0)
    table.add_row("Hit Rate", f"{hit_rate:.2%}")
    lat_saved = stats.get("estimated_latency_saved_ms", 0.0)
    table.add_row("Estimated Latency Saved", f"{lat_saved:.1f} ms")

    console.print(table)


@cache_group.command(name="clear")
def cache_clear_cmd() -> None:
    """Flush all entries from the semantic cache."""
    _GLOBAL_CACHE.clear()
    console.print("[bold green]Semantic cache successfully cleared.[/bold green]")


# -------------------------------------------------------------------------
# 8. FEATURE STORE COMMAND GROUP
# -------------------------------------------------------------------------
@cli.group(name="feature-store")
def feature_store_group() -> None:
    """Inspect and manage columnar Parquet feature store and drift detection."""
    pass


# Register alias 'features' for feature-store command group
cli.add_command(feature_store_group, name="features")


@feature_store_group.command(name="stats")
@click.option(
    "--path",
    "-p",
    type=click.Path(path_type=Path),
    default=Path("./data/feature_store"),
    help="Path to feature store root directory.",
)
@click.option("--json-out", is_flag=True, help="Output raw JSON summary.")
def feature_store_stats_cmd(path: Path, json_out: bool) -> None:
    """Display feature store storage summary, record counts, and partition sizes."""
    store = ParquetFeatureStore(base_path=path)
    summary = store.storage_summary()

    if json_out:
        click.echo(json.dumps(summary, indent=2))
        return

    table = Table(title="Columnar Feature Store Storage Summary")
    table.add_column("Feature Group", style="cyan")
    table.add_column("Records", style="bold green", justify="right")
    table.add_column("Columns", justify="right")
    table.add_column("Row Groups", justify="right")
    table.add_column("Size (Bytes)", justify="right")

    for group_name, info in summary.get("groups", {}).items():
        table.add_row(
            group_name,
            str(info.get("num_rows", 0)),
            str(info.get("num_columns", 0)),
            str(info.get("num_row_groups", 0)),
            f"{info.get('size_bytes', 0):,}",
        )

    console.print(table)
    console.print(
        f"[bold]Total Records:[/bold] {summary.get('total_records', 0):,} | "
        f"[bold]Total Storage:[/bold] {summary.get('total_bytes', 0):,} bytes | "
        f"[bold]Location:[/bold] {path}"
    )


@feature_store_group.command(name="list-schemas")
@click.option(
    "--group",
    "-g",
    type=click.Choice([g.value for g in FeatureGroup], case_sensitive=False),
    default=None,
    help="Filter by specific feature group.",
)
@click.option("--json-out", is_flag=True, help="Output raw JSON schemas.")
def feature_store_list_schemas_cmd(group: Optional[str], json_out: bool) -> None:
    """List registered feature definitions, value types, and descriptions."""
    registry = FeatureRegistry(load_defaults=True)
    all_schemas = registry.list_schemas()

    selected_group = FeatureGroup(group.upper()) if group else None
    filtered = (
        {selected_group: all_schemas[selected_group]}
        if selected_group and selected_group in all_schemas
        else all_schemas
    )

    if json_out:
        payload = {k.value: v.model_dump(mode="json") for k, v in filtered.items()}
        click.echo(json.dumps(payload, indent=2))
        return

    for grp, schema in filtered.items():
        table = Table(title=f"Schema: {schema.name} ({grp.value}) - v{schema.version}")
        table.add_column("Feature Name", style="cyan")
        table.add_column("Type", style="bold yellow")
        table.add_column("Nullable", justify="center")
        table.add_column("Description", style="italic")

        for feat in schema.features:
            table.add_row(
                feat.name,
                feat.value_type.value,
                "[yellow]Yes[/yellow]" if feat.nullable else "[green]No[/green]",
                feat.description or "-",
            )
    console.print(table)


# -------------------------------------------------------------------------
# 9. MCP SERVER & TOOLS COMMAND GROUP
# -------------------------------------------------------------------------
@cli.group(name="mcp")
def mcp_group() -> None:
    """Manage and inspect Model Context Protocol (MCP) server, tools, and resources."""
    pass


@mcp_group.command(name="serve")
@click.option(
    "--transport",
    "-t",
    type=click.Choice(["stdio", "sse", "http"], case_sensitive=False),
    default="stdio",
    help="MCP transport protocol mode (default: stdio).",
)
@click.option("--host", default="127.0.0.1", help="Host interface to bind network server.")
@click.option("--port", "-p", type=int, default=8000, help="Port for network transport.")
@click.option("--debug", is_flag=True, help="Enable verbose debug logging.")
@click.option("--enable-sessions", is_flag=True, help="Enable session management and tracking.")
def mcp_serve_cmd(
    transport: str,
    host: str,
    port: int,
    debug: bool,
    enable_sessions: bool,
) -> None:
    """Start and run the production MCP server."""
    cfg = MCPConfig(
        transport=MCPTransport(transport.lower()),
        host=host,
        port=port,
        debug=debug,
        log_level="DEBUG" if debug else "INFO",
    )
    server = create_mcp_server(cfg, enable_session_handler=enable_sessions)
    if cfg.transport == MCPTransport.STDIO:
        server.run_stdio()
    else:
        click.echo(
            f"Network transport '{cfg.transport.value}' initialized on {cfg.host}:{cfg.port}. Falling back to stdio dispatcher."
        )
        server.run_stdio()


@mcp_group.command(name="list-tools")
@click.option("--json-out", is_flag=True, help="Output raw JSON tool definitions.")
def mcp_list_tools_cmd(json_out: bool) -> None:
    """List all registered MCP tools and their execution schemas."""
    tool_reg = get_default_tool_registry()
    tools = tool_reg.list_tools()

    if json_out:
        click.echo(json.dumps(tools, indent=2))
        return

    table = Table(title=f"Registered MCP Tools ({len(tools)} available)")
    table.add_column("Tool Name", style="bold cyan")
    table.add_column("Description", style="white")
    table.add_column("Parameters", style="yellow")

    for t in tools:
        props = list(t.get("inputSchema", {}).get("properties", {}).keys())
        table.add_row(
            t.get("name", ""),
            t.get("description", ""),
            ", ".join(props) if props else "-",
        )

    console.print(table)


@mcp_group.command(name="list-resources")
@click.option("--json-out", is_flag=True, help="Output raw JSON resource definitions.")
def mcp_list_resources_cmd(json_out: bool) -> None:
    """List all exposed MCP resources and templates."""
    resource_reg = get_default_resource_registry()
    resources = resource_reg.list_resources()

    if json_out:
        click.echo(json.dumps(resources, indent=2))
        return

    table = Table(title=f"Exposed MCP Resources ({len(resources)} available)")
    table.add_column("URI", style="bold cyan", no_wrap=True)
    table.add_column("Name", style="magenta")
    table.add_column("MIME Type", style="yellow")
    table.add_column("Description", style="white")

    for r in resources:
        table.add_row(
            r.get("uri", ""),
            r.get("name", ""),
            r.get("mimeType", ""),
            r.get("description", "-"),
        )

    console.print(table)


@mcp_group.command(name="list-prompts")
@click.option("--json-out", is_flag=True, help="Output raw JSON prompt definitions.")
def mcp_list_prompts_cmd(json_out: bool) -> None:
    """List all exposed MCP prompt templates and arguments."""
    prompt_reg = get_default_prompt_registry()
    prompts = prompt_reg.list_prompts()

    if json_out:
        click.echo(json.dumps(prompts, indent=2))
        return

    table = Table(title=f"Exposed MCP Prompt Templates ({len(prompts)} available)")
    table.add_column("Prompt Name", style="bold cyan")
    table.add_column("Description", style="white")
    table.add_column("Arguments", style="yellow")

    for p in prompts:
        args = [arg.get("name", "") for arg in p.get("arguments", [])]
        table.add_row(
            p.get("name", ""),
            p.get("description", ""),
            ", ".join(args) if args else "-",
        )

    console.print(table)


@mcp_group.command(name="call-tool")
@click.option("--name", "-n", required=True, help="Tool name to invoke.")
@click.option(
    "--arguments",
    "-a",
    default="{}",
    help="JSON string or @file containing tool parameters.",
)
@click.option("--json-out", is_flag=True, help="Output raw JSON result.")
def mcp_call_tool_cmd(name: str, arguments: str, json_out: bool) -> None:
    """Execute an MCP tool directly from the command line."""
    raw_args = _read_text_or_file(arguments)
    try:
        parsed_args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"Invalid JSON in --arguments: {exc}") from exc

    tool_reg = get_default_tool_registry()
    tool = tool_reg.get(name)
    if not tool:
        raise click.BadParameter(
            f"Unknown MCP tool '{name}'. Run 'rag-eval mcp list-tools' to view available tools."
        )

    result = tool_reg.call(name, parsed_args)

    if json_out:
        click.echo(result.model_dump_json(indent=2))
        return

    status_str = (
        "[bold red]ERROR[/bold red]" if result.isError else "[bold green]SUCCESS[/bold green]"
    )
    table = Table(title=f"MCP Tool Execution: {name} ({status_str})")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Tool", name)
    table.add_row("Status", status_str)
    for idx, c in enumerate(result.content):
        val = c.get("text", "")
        if len(val) > 120:
            val = val[:117] + "..."
        table.add_row(f"Content [{idx}]", val)

    if result.metadata:
        table.add_row("Metadata", json.dumps(result.metadata, indent=2))

    console.print(table)


@feature_store_group.command(name="compact")
@click.option(
    "--path",
    "-p",
    type=click.Path(path_type=Path),
    default=Path("./data/feature_store"),
    help="Path to feature store root directory.",
)
@click.option(
    "--group",
    "-g",
    type=click.Choice([g.value for g in FeatureGroup], case_sensitive=False),
    default=None,
    help="Specific feature group to compact (default: all).",
)
@click.option("--ttl", type=float, default=None, help="Prune records older than TTL (seconds).")
@click.option(
    "--compression",
    type=click.Choice(["snappy", "gzip", "brotli", "lz4", "zstd", "none"], case_sensitive=False),
    default="snappy",
    help="Parquet compression codec.",
)
@click.option("--json-out", is_flag=True, help="Output compaction results as JSON.")
def feature_store_compact_cmd(
    path: Path,
    group: Optional[str],
    ttl: Optional[float],
    compression: str,
    json_out: bool,
) -> None:
    """Compact Parquet partitions, purge duplicates, and prune TTL-expired records."""
    store = ParquetFeatureStore(base_path=path)
    compactor = FeatureCompactionHandler(store=store)

    if group:
        target_group = FeatureGroup(group.upper())
        res = compactor.compact_group(
            group=target_group,
            ttl_seconds=ttl,
            compression=compression.lower(),
        )
        results = {target_group: res}
    else:
        results = compactor.compact_all(ttl_seconds=ttl, compression=compression.lower())

    if json_out:
        out_dict = {k.value: v.model_dump(mode="json") for k, v in results.items()}
        click.echo(json.dumps(out_dict, indent=2))
        return

    table = Table(title="Feature Store Compaction Report")
    table.add_column("Group", style="cyan")
    table.add_column("Records Retained", justify="right", style="bold green")
    table.add_column("Pruned (TTL)", justify="right", style="bold red")
    table.add_column("Bytes Before", justify="right")
    table.add_column("Bytes After", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Duration", justify="right")

    for grp, res in results.items():
        table.add_row(
            grp.value,
            str(res.records_compacted),
            str(res.expired_records_pruned),
            f"{res.bytes_before:,}",
            f"{res.bytes_after:,}",
            f"{res.compression_ratio:.2f}",
            f"{res.duration_ms:.1f} ms",
        )

    console.print(table)


@feature_store_group.command(name="export")
@click.option(
    "--path",
    "-p",
    type=click.Path(path_type=Path),
    default=Path("./data/feature_store"),
    help="Path to feature store root directory.",
)
@click.option(
    "--entities",
    "-e",
    required=True,
    help="Comma-separated entity IDs or path to file containing entity IDs (one per line).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Destination file path for exported matrix.",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["jsonl", "csv", "parquet"], case_sensitive=False),
    default="jsonl",
    help="Export matrix format (default: jsonl).",
)
@click.option(
    "--group",
    "-g",
    type=click.Choice([g.value for g in FeatureGroup], case_sensitive=False),
    multiple=True,
    help="Feature groups to include (default: all).",
)
def feature_store_export_cmd(
    path: Path,
    entities: str,
    output: Path,
    format: str,
    group: Tuple[str, ...],
) -> None:
    """Export aligned multi-group point-in-time feature matrix for entity IDs."""
    store = ParquetFeatureStore(base_path=path)
    exporter = FeatureExportHandler(store=store)

    # Parse entities
    entity_arg = entities.strip()
    candidate_file = Path(entity_arg)
    if candidate_file.is_file():
        entity_ids = [
            line.strip()
            for line in candidate_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        entity_ids = [e.strip() for e in entity_arg.split(",") if e.strip()]

    if not entity_ids:
        raise click.BadParameter("No valid entity IDs provided.")

    target_groups = [FeatureGroup(g.upper()) for g in group] if group else None
    export_format = ExportFormat(format.upper())

    out_res = exporter.export_dataset(
        entity_ids=entity_ids,
        output_path=output,
        format=export_format,
        groups=target_groups,
    )

    console.print(
        f"[bold green]Successfully exported {len(entity_ids)} entities to {out_res}[/bold green]"
    )


@feature_store_group.command(name="drift")
@click.option(
    "--path",
    "-p",
    type=click.Path(path_type=Path),
    default=Path("./data/feature_store"),
    help="Path to feature store root directory.",
)
@click.option(
    "--feature",
    "-f",
    default=None,
    help="Specific feature name to evaluate (default: all numeric).",
)
@click.option(
    "--group",
    "-g",
    type=click.Choice([g.value for g in FeatureGroup], case_sensitive=False),
    default=None,
    help="Feature group to analyze.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.25,
    help="Population Stability Index (PSI) drift alert threshold (default: 0.25).",
)
@click.option("--json-out", is_flag=True, help="Output drift reports as JSON.")
def feature_store_drift_cmd(
    path: Path,
    feature: Optional[str],
    group: Optional[str],
    threshold: float,
    json_out: bool,
) -> None:
    """Analyze feature distribution stability and detect statistical drift using PSI."""
    store = ParquetFeatureStore(base_path=path)
    detector = FeatureDriftDetector(store=store)

    target_group = FeatureGroup(group.upper()) if group else None

    if feature:
        if not target_group:
            raise click.BadParameter(
                "Evaluating a specific feature requires specifying --group / -g."
            )
        reports = [
            detector.detect_feature_drift(
                feature_name=feature,
                group=target_group,
                psi_threshold=threshold,
            )
        ]
    else:
        reports = detector.check_all_features_drift(
            group=target_group,
            psi_threshold=threshold,
        )

    if json_out:
        out_list = [r.model_dump(mode="json") for r in reports]
        click.echo(json.dumps(out_list, indent=2))
        return

    table = Table(title=f"Feature Distribution Drift Analysis (PSI Threshold = {threshold})")
    table.add_column("Feature", style="cyan", no_wrap=True)
    table.add_column("Group", style="magenta")
    table.add_column("PSI Score", justify="right")
    table.add_column("Baseline Mean", justify="right")
    table.add_column("Target Mean", justify="right")
    table.add_column("Drift Detected", justify="center")
    table.add_column("Severity", justify="center")

    for rep in reports:
        drift_str = "[bold red]YES[/bold red]" if rep.drift_detected else "[green]NO[/green]"
        b_mean_str = f"{rep.baseline_mean:.4f}" if rep.baseline_mean is not None else "-"
        t_mean_str = f"{rep.target_mean:.4f}" if rep.target_mean is not None else "-"
        table.add_row(
            rep.feature_name,
            rep.feature_group.value,
            f"{rep.psi_score:.4f}",
            b_mean_str,
            t_mean_str,
            drift_str,
            rep.severity.value,
        )

    console.print(table)


# =====================================================================
# AGENT SWARM WORKFLOW CLI COMMAND GROUP
# =====================================================================


@cli.group(name="swarm")
def swarm_group() -> None:
    """Multi-agent swarm orchestration and DAG workflow evaluation commands."""
    pass


# Register alias 'agent' for swarm command group
cli.add_command(swarm_group, name="agent")


@swarm_group.command(name="list-agents")
@click.option("--json-out", is_flag=True, help="Output agents list as raw JSON.")
def swarm_list_agents_cmd(json_out: bool) -> None:
    """List all registered specialized agents in the swarm registry."""
    orch = create_default_orchestrator()
    agents = orch.registry.list_agents()

    if json_out:
        click.echo(json.dumps(agents, indent=2))
        return

    table = Table(title="Registered Swarm Agents")
    table.add_column("Agent Name", style="cyan", no_wrap=True)
    table.add_column("Role", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Inbox Count", justify="right")
    table.add_column("Last Exec (ms)", justify="right")

    for a in agents:
        table.add_row(
            a.get("name", "unknown"),
            a.get("role", "unknown"),
            a.get("status", "unknown"),
            str(a.get("inbox_count", 0)),
            f"{a.get('last_execution_time_ms', 0.0):.2f}",
        )

    console.print(table)


@swarm_group.command(name="plan")
@click.option("-q", "--query", required=True, help="Financial query text or path to text file.")
@click.option("-a", "--answer", required=True, help="Generated answer text or path to text file.")
@click.option(
    "-c",
    "--context",
    required=True,
    help="Retrieved context snippet, raw text or path to file.",
)
@click.option("-t", "--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL).")
@click.option("-p", "--period", default="FY2024", help="Reporting period (default: FY2024).")
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["fast", "thorough", "stress"], case_sensitive=False),
    default="thorough",
    help="Evaluation mode (default: thorough).",
)
@click.option(
    "-r",
    "--returns",
    default=None,
    help="Comma-separated decimal daily returns for risk agent (or file path).",
)
@click.option("--json-out", is_flag=True, help="Output generated workflow plan as raw JSON.")
def swarm_plan_cmd(
    query: str,
    answer: str,
    context: str,
    ticker: str,
    period: str,
    mode: str,
    returns: Optional[str],
    json_out: bool,
) -> None:
    """Inspect the topological DAG execution plan without executing."""
    query_text = _read_text_or_file(query)
    answer_text = _read_text_or_file(answer)
    context_text = _read_text_or_file(context)
    returns_arr = _parse_float_array(returns)
    eval_mode = EvaluationMode[mode.upper()]

    q = RAGQuery(
        query_id="cli-swarm-q1",
        query_text=query_text,
        ticker=ticker,
        period=period,
    )
    chunk = DocumentChunk(
        chunk_id="cli-swarm-c1",
        text=context_text,
        doc_type=DocumentType.SEC_10K,
        ticker=ticker,
        period=period,
    )
    resp = RAGResponse(
        query_id="cli-swarm-q1",
        generated_answer=answer_text,
        retrieved_chunks=[chunk],
    )

    orch = create_default_orchestrator()
    plan = orch.create_plan(query=q, response=resp, returns=returns_arr, evaluation_mode=eval_mode)

    if json_out:
        click.echo(plan.model_dump_json(indent=2))
        return

    console.print(
        Panel.fit(
            f"[bold]Swarm Workflow DAG Plan[/bold] | Plan ID: [cyan]{plan.plan_id}[/cyan] | Steps: [magenta]{len(plan.steps)}[/magenta]",
            border_style="blue",
        )
    )

    table = Table(title="Topological Execution Steps")
    table.add_column("Step ID", style="dim", no_wrap=True)
    table.add_column("Step Name", style="cyan", no_wrap=True)
    table.add_column("Agent Role", style="magenta")
    table.add_column("Mode", style="yellow")
    table.add_column("Dependencies", style="green")

    for s in plan.steps:
        deps_str = ", ".join(s.dependencies) if s.dependencies else "[dim]None[/dim]"
        table.add_row(
            s.step_id[:8] + "...",
            s.name,
            s.agent_role.value,
            s.mode.value,
            deps_str,
        )

    console.print(table)


@swarm_group.command(name="run")
@click.option("-q", "--query", required=True, help="Financial query text or path to text file.")
@click.option("-a", "--answer", required=True, help="Generated answer text or path to text file.")
@click.option(
    "-c",
    "--context",
    required=True,
    help="Retrieved context snippet, raw text or path to file.",
)
@click.option("-t", "--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL).")
@click.option("-p", "--period", default="FY2024", help="Reporting period (default: FY2024).")
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["fast", "thorough", "stress"], case_sensitive=False),
    default="thorough",
    help="Evaluation mode (default: thorough).",
)
@click.option(
    "-r",
    "--returns",
    default=None,
    help="Comma-separated decimal daily returns for financial risk agent (or file path).",
)
@click.option(
    "--concurrency",
    default=5,
    type=click.IntRange(1, 20),
    help="Maximum parallel execution concurrency limit (default: 5).",
)
@click.option(
    "--halt-on-hallucination",
    is_flag=True,
    help="Halt workflow immediately if critical/high severity hallucination is detected.",
)
@click.option(
    "--auto-retry",
    is_flag=True,
    help="Automatically retry failed agent workflow steps.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Optional file path to write results JSON to.",
)
@click.option("--json-out", is_flag=True, help="Output workflow execution result as raw JSON.")
def swarm_run_cmd(
    query: str,
    answer: str,
    context: str,
    ticker: str,
    period: str,
    mode: str,
    returns: Optional[str],
    concurrency: int,
    halt_on_hallucination: bool,
    auto_retry: bool,
    output: Optional[Path],
    json_out: bool,
) -> None:
    """Execute end-to-end multi-agent swarm evaluation workflow."""
    query_text = _read_text_or_file(query)
    answer_text = _read_text_or_file(answer)
    context_text = _read_text_or_file(context)
    returns_arr = _parse_float_array(returns)
    eval_mode = EvaluationMode[mode.upper()]

    q = RAGQuery(
        query_id="cli-swarm-q1",
        query_text=query_text,
        ticker=ticker,
        period=period,
    )
    chunk = DocumentChunk(
        chunk_id="cli-swarm-c1",
        text=context_text,
        doc_type=DocumentType.SEC_10K,
        ticker=ticker,
        period=period,
    )
    resp = RAGResponse(
        query_id="cli-swarm-q1",
        generated_answer=answer_text,
        retrieved_chunks=[chunk],
    )

    config = AgentWorkflowConfig(
        concurrency_limit=concurrency,
        halt_on_high_severity_hallucination=halt_on_hallucination,
        auto_retry_failed_steps=auto_retry,
    )
    orchestrator = create_default_orchestrator(config=config)
    plan = orchestrator.create_plan(
        query=q,
        response=resp,
        returns=returns_arr,
        evaluation_mode=eval_mode,
    )

    result = orchestrator.execute_workflow(
        query=q,
        response=resp,
        returns=returns_arr,
        plan=plan,
    )

    if output is not None:
        output.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    if json_out:
        click.echo(result.model_dump_json(indent=2))
        return

    status_color = "green" if result.status == WorkflowStatus.COMPLETED else "red"
    console.print(
        Panel.fit(
            f"[bold]Swarm Evaluation Workflow[/bold] | Status: [{status_color}]{result.status.value}[/{status_color}] | Duration: [magenta]{result.total_execution_time_ms:.1f}ms[/magenta]",
            border_style=status_color,
        )
    )

    # Step Execution Summary
    step_table = Table(title="Workflow Step Execution Summary")
    step_table.add_column("Step Name", style="cyan")
    step_table.add_column("Agent Role", style="magenta")
    step_table.add_column("Status", style="bold")
    step_table.add_column("Duration (ms)", justify="right")

    for step in result.plan.steps:
        s_res = result.step_results.get(step.step_id, {})
        s_status = s_res.get("status", step.status.value)
        s_dur = s_res.get("execution_time_ms", step.execution_time_ms)
        color = "green" if s_status == "completed" else "yellow" if s_status == "skipped" else "red"
        step_table.add_row(
            step.name,
            step.agent_role.value,
            f"[{color}]{s_status}[/{color}]",
            f"{s_dur:.1f}",
        )
    console.print(step_table)

    # Core Metrics Breakdown
    if result.metrics:
        m_table = Table(title="Evaluator Agent Metrics")
        m_table.add_column("Metric", style="cyan")
        m_table.add_column("Score", style="bold magenta", justify="right")
        m_table.add_row("Faithfulness", f"{result.metrics.faithfulness_score:.2%}")
        m_table.add_row("Answer Relevance", f"{result.metrics.answer_relevance_score:.2%}")
        m_table.add_row("Context Recall", f"{result.metrics.context_recall_score:.2%}")
        m_table.add_row("Context Precision", f"{result.metrics.context_precision_score:.2%}")
        m_table.add_row("Overall Pass", "[green]PASS[/green]" if result.metrics.overall_pass else "[red]FAIL[/red]")
        console.print(m_table)

    # Hallucination Findings Breakdown
    if result.findings:
        f_table = Table(title=f"Auditor Agent Findings ({len(result.findings)} Detected)")
        f_table.add_column("Type", style="cyan")
        f_table.add_column("Severity", style="bold red")
        f_table.add_column("Claim Text", style="magenta")
        for f in result.findings:
            f_table.add_row(
                f.finding_type.value,
                f.severity.value,
                f.claim_text[:60] + "..." if len(f.claim_text) > 60 else f.claim_text,
            )
        console.print(f_table)

    # Risk Metrics Breakdown
    if result.risk_metrics:
        r_table = Table(title="Risk Analyst Agent Metrics")
        r_table.add_column("Metric", style="cyan")
        r_table.add_column("Value", style="bold magenta", justify="right")
        r_table.add_row("Annualized Volatility", f"{result.risk_metrics.annualized_volatility:.2%}")
        r_table.add_row("Value at Risk (95%)", f"{result.risk_metrics.var_95:.2%}")
        r_table.add_row("Conditional VaR (95%)", f"{result.risk_metrics.cvar_95:.2%}")
        r_table.add_row("Max Drawdown", f"{result.risk_metrics.max_drawdown:.2%}")
        console.print(r_table)

    # Synthesis & Recommendations
    if result.summary:
        console.print(
            Panel(
                f"[bold cyan]Synthesis Summary:[/bold cyan]\n{result.summary}",
                border_style="cyan",
            )
        )
    if result.recommendations:
        rec_text = "\n".join(f"• {r}" for r in result.recommendations)
        console.print(
            Panel(
                f"[bold yellow]Strategic Recommendations:[/bold yellow]\n{rec_text}",
                border_style="yellow",
            )
        )

