"""Command line interface for production RAG evaluation and financial risk metrics."""

from __future__ import annotations

from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from production_rag_evaluation.core.evaluator import RAGEvaluator
from production_rag_evaluation.finance.metrics import FinancialRiskEngine
from production_rag_evaluation.finance.stress_testing import MonteCarloStressTester
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    DocumentType,
    RAGQuery,
    RAGResponse,
)

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Production RAG Evaluation & Hallucination Detection Pipeline."""
    pass


@cli.command(name="evaluate")
@click.option("--query", "-q", required=True, help="Input financial question/query.")
@click.option("--answer", "-a", required=True, help="Generated answer to evaluate.")
@click.option("--context", "-c", required=True, help="Retrieved ground-truth text context.")
@click.option("--ticker", "-t", default="AAPL", help="Stock ticker symbol.")
@click.option("--json-out", is_flag=True, help="Output raw JSON results.")
def evaluate_cmd(query: str, answer: str, context: str, ticker: str, json_out: bool) -> None:
    """Evaluate a single RAG query/response against financial context."""
    evaluator = RAGEvaluator()
    q = RAGQuery(query_id="cli-q1", query_text=query, ticker=ticker)
    chunk = DocumentChunk(
        chunk_id="cli-c1",
        text=context,
        doc_type=DocumentType.SEC_10K,
        ticker=ticker,
    )
    resp = RAGResponse(
        query_id="cli-q1",
        generated_answer=answer,
        retrieved_chunks=[chunk],
    )

    metrics = evaluator.evaluate(q, resp)

    if json_out:
        click.echo(metrics.model_dump_json(indent=2))
        return

    table = Table(title=f"RAG Evaluation Report — {ticker}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")
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
        "Hallucination Rate",
        f"{metrics.hallucination_rate:.2%}",
        "[green]LOW[/green]" if metrics.hallucination_rate < 0.2 else "[red]HIGH[/red]",
    )
    table.add_row(
        "Overall Pass",
        str(metrics.overall_pass),
        "[green]YES[/green]" if metrics.overall_pass else "[red]NO[/red]",
    )
    table.add_row("Latency", f"{metrics.evaluation_latency_ms:.1f} ms", "OK")

    console.print(table)

    if metrics.findings:
        console.print("\n[bold yellow]Hallucination Findings:[/bold yellow]")
        for f in metrics.findings:
            console.print(f" - [{f.severity.value}] {f.finding_type.value}: {f.claim_text}")
            console.print(f"   Reason: {f.explanation}")


@cli.command(name="risk-metrics")
@click.option("--returns", "-r", default=None, help="Comma-separated decimal daily returns.")
@click.option("--risk-free", default=0.045, help="Annualized risk-free rate.")
@click.option("--json-out", is_flag=True, help="Output raw JSON results.")
def risk_metrics_cmd(returns: Optional[str], risk_free: float, json_out: bool) -> None:
    """Calculate Sharpe, Sortino, Calmar, and Value-at-Risk indicators."""
    if returns:
        arr = np.array([float(x.strip()) for x in returns.split(",")], dtype=np.float64)
    else:
        # Default representative daily returns
        arr = np.array(
            [0.005, -0.003, 0.008, 0.002, -0.001, 0.006, 0.004, -0.002, 0.003, 0.001],
            dtype=np.float64,
        )

    engine = FinancialRiskEngine(risk_free_rate=risk_free)
    metrics = engine.compute_all_metrics(arr)

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

    console.print(table)


@cli.command(name="stress-test")
@click.option("--simulations", "-n", default=1000, help="Number of Monte Carlo paths.")
@click.option("--horizon", "-h", default=30, help="Forecast horizon in trading days.")
@click.option("--shock", "-s", default=1.5, help="Volatility shock multiplier.")
@click.option("--json-out", is_flag=True, help="Output raw JSON results.")
def stress_test_cmd(simulations: int, horizon: int, shock: float, json_out: bool) -> None:
    """Run Monte Carlo stress testing on simulated financial trajectories."""
    tester = MonteCarloStressTester()
    base_returns = np.array([0.002, -0.004, 0.001, -0.002, 0.005, 0.003, -0.001], dtype=np.float64)
    result = tester.run_stress_test(
        base_returns,
        n_simulations=simulations,
        horizon_days=horizon,
        volatility_shock=shock,
    )

    if json_out:
        click.echo(result.model_dump_json(indent=2))
        return

    table = Table(title=f"Monte Carlo Stress Simulation ({simulations} paths, {horizon}d)")
    table.add_column("Parameter / Outcome", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("5th Percentile Terminal ($)", f"${result.p5_terminal:.2f}")
    table.add_row("50th Percentile Terminal ($)", f"${result.p50_terminal:.2f}")
    table.add_row("95th Percentile Terminal ($)", f"${result.p95_terminal:.2f}")
    table.add_row("Mean Terminal ($)", f"${result.mean_terminal:.2f}")
    table.add_row("Worst Case Drawdown", f"{result.worst_case_drawdown:.2%}")
    table.add_row(
        "Stress Test Passed", "[green]YES[/green]" if result.stress_pass else "[red]NO[/red]"
    )

    console.print(table)


@cli.command(name="telemetry")
@click.option("--json-out", is_flag=True, help="Output raw JSON results.")
def telemetry_cmd(json_out: bool) -> None:
    """Show end-to-end telemetry status combining RAG quality and financial risk."""
    collector = RiskTelemetryCollector()
    # Mock some history
    chunk = DocumentChunk(
        chunk_id="c1", text="Apple revenue was $85.8 billion, up 5% year-over-year.", ticker="AAPL"
    )
    q = RAGQuery(query_id="q1", query_text="What was Apple revenue?", ticker="AAPL")
    resp = RAGResponse(
        query_id="q1",
        generated_answer="Apple reported revenue of $85.8 billion.",
        retrieved_chunks=[chunk],
    )
    m = collector.evaluator.evaluate(q, resp)
    collector.record_evaluation(m)

    returns = np.array([0.004, -0.002, 0.003, 0.001, 0.005, -0.001, 0.002], dtype=np.float64)
    rep = collector.generate_report(returns=returns)

    if json_out:
        click.echo(rep.model_dump_json(indent=2))
        return

    table = Table(title="Production RAG & Financial Telemetry Report")
    table.add_column("Component", style="cyan")
    table.add_column("Telemetry Reading", style="bold green")

    table.add_row("Total Evaluations", str(rep.total_evaluations))
    table.add_row("Average Faithfulness", f"{rep.avg_faithfulness:.2%}")
    table.add_row("Average Hallucination Rate", f"{rep.avg_hallucination_rate:.2%}")
    table.add_row("Cache Hit Rate", f"{rep.cache_hit_rate:.2%}")
    if rep.risk_metrics:
        table.add_row("Portfolio Sharpe Ratio", f"{rep.risk_metrics.sharpe_ratio:.2f}")
        table.add_row("Portfolio Sortino Ratio", f"{rep.risk_metrics.sortino_ratio:.2f}")
        table.add_row("Portfolio Max Drawdown", f"{rep.risk_metrics.max_drawdown:.2%}")
    table.add_row(
        "Active Alerts", ", ".join(rep.alerts_triggered) if rep.alerts_triggered else "None"
    )

    console.print(table)
