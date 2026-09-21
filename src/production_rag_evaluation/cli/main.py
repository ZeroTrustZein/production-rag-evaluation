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

from production_rag_evaluation.core.cache import SemanticCache
from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
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
@click.option("--json-out", is_flag=True, help="Output raw JSON summary.")
def batch_evaluate_cmd(
    input_file: str,
    output: Optional[str],
    mode: str,
    batch_id: Optional[str],
    no_cache: bool,
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
