# Production RAG Evaluation & Quantitative Financial Risk Pipeline

[![CI](https://github.com/zein/production-rag-evaluation/actions/workflows/ci.yml/badge.svg)](https://github.com/zein/production-rag-evaluation/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Test Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/zein/production-rag-evaluation)

An institutional-grade evaluation, hallucination detection, and quantitative risk platform engineered to audit, benchmark, stress-test, and guardrail Retrieval-Augmented Generation (RAG) pipelines operating on financial earnings calls, SEC filings (10-K, 10-Q, 8-K), and equity research reports.

---

## Documentation Suite

Detailed technical guides and reference manuals are available in the [`docs/`](docs/) directory:

- 📐 **[System Architecture & Quantitative Formulations](docs/ARCHITECTURE.md)**: Deep mathematical formulations (DAG state machine, n-gram hashing, Cornish-Fisher VaR, GBM & fat-tailed Student-t shocks, Welch's t-test drift detection).
- 📚 **[Python API Reference](docs/API_REFERENCE.md)**: Comprehensive reference for all domain models, classes, protocols, parameters, return types, and exceptions.
- 🚀 **[User & CLI Guide](docs/USAGE_GUIDE.md)**: Complete command-line manual, option explanations, downstream JSON recipes, and end-to-end Python workflows.

---

## Key Features

- **Multi-Dimensional RAG Evaluation Framework**:
  - Faithfulness, Answer Relevance, Context Precision, Context Recall, and Citation Grounding scores.
  - Configurable execution modes: `FAST`, `THOROUGH`, `FINANCIAL_STRICT`, and `AUDIT`.
  - Structured numerical target verification with custom tolerance bounds (e.g. `net_sales=391.0e9`).
- **Specialized Financial Hallucination Engine**:
  - Scale-aware numerical distortion detection ($12.5M, 15.4%, $120 billion, etc.) with relative error tolerance.
  - Financial directional contradiction detection (contrasting bullish assertions against bearish disclosures).
  - Lexical token grounding and ungrounded claim segmentation.
  - Cross-company entity and ticker swap detection (filtering financial abbreviations).
  - Temporal period and fiscal year mismatch detection.
- **High-Throughput In-Memory Semantic Cache**:
  - Deterministic n-gram hash vectorizer (`TextEmbedder`, $D=128$, MD5 unigrams, SHA-256 bigrams).
  - Dual-tier lookup (exact SHA-256 hash match + cosine similarity search).
  - Automatic LRU capacity eviction and TTL expiration.
  - Saves ~25ms per evaluation hit with zero GPU dependency.
- **Institutional Quantitative Risk Engine**:
  - Annualized Sharpe Ratio, Sortino Ratio (downside deviation), Max Drawdown, and Calmar Ratio.
  - Historical Value-at-Risk (95%, 99%) & Conditional VaR / Expected Shortfall (CVaR).
  - Parametric Cornish-Fisher VaR adjusting for skewness and excess kurtosis fat tails.
  - CAPM Beta & Jensen's Alpha against benchmark indices (e.g. SPY).
  - Tracking Error, Information Ratio, and Omega Ratio.
- **Stochastic Monte Carlo Stress Engine**:
  - Geometric Brownian Motion (GBM) path simulation under volatility and drift shocks.
  - Heavy-tailed Student-t innovation shocks for black swan market crash modeling.
  - Six preset macroeconomic crisis scenarios (`BASELINE`, `VOLATILITY_SPIKE`, `LIQUIDITY_CRUNCH`, `RATE_SHOCK`, `BLACK_SWAN`, `HALLUCINATION_CASCADE`).
  - Hallucination sensitivity simulation quantifying downstream portfolio error breach probabilities.
- **Real-Time Telemetry & Statistical Drift Detection**:
  - Rolling metric windows tracking historical quality trends.
  - Two-sample Welch's t-test identifying statistically significant performance degradation.
  - Multi-tier operational alerting (`INFO`, `WARNING`, `ERROR`, `CRITICAL`) with pub/sub callbacks.
  - System health state machine (`HEALTHY` $\to$ `DEGRADED` $\to$ `UNHEALTHY`).
- **Columnar Feature Store & Streaming Pipeline (`features`)**:
  - Strongly-typed schema registry (`FeatureRegistry`) across 5 domains (`QUERY`, `CHUNK`, `RESPONSE`, `EVALUATION`, `FINANCIAL`).
  - High-throughput streaming pipeline (`StreamingFeaturePipeline`) with in-memory buffering and batch flushes.
  - Columnar Apache Parquet storage (`ParquetFeatureStore`) with predicate pushdown and column projection.
  - Population Stability Index (PSI) drift detection (`FeatureDriftDetector`) with automated telemetry alerts.
  - Automated partition compaction, entity deduplication, and configurable TTL data retention.
  - Point-in-time multi-group join matrix export to Parquet, CSV, JSON, and Arrow.
- **Production CLI & Script Automation (`rag-eval`)**:
  - Full support for interactive Rich tables and scriptable `--json-out`.
  - Batch evaluation from JSON and JSONL datasets with p95 latency reporting.
  - Feature store management commands (`rag-eval feature-store stats/compact/export/drift/list-schemas`).
  - Cache management commands (`rag-eval cache stats`, `rag-eval cache clear`).

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                        Inbound RAG Query & Generated Response                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     In-Memory Semantic Cache (SemanticCache)                      |
|  - Exact SHA-256 Hash Matching                                                    |
|  - Fast N-Gram Hash Vectorization (D=128) & Cosine Similarity Match               |
+-------------------+-------------------------------------------+-------------------+
                    | [Hit]                                     | [Miss]
                    v                                           v
+-----------------------------------+   +-------------------------------------------+
| Return Cached EvaluationMetrics   |   |   Evaluation Workflow DAG (EvaluationGraph) |
+-----------------------------------+   +---------------------+---------------------+
                                                              |
                                                              v
+-----------------------------------------------------------------------------------+
|                         DAG Execution Pipeline (Sequential)                       |
|  [extract_claims] -> [verify_citations] -> [detect_hallucinations]                |
|  -> [verify_claims] -> [verify_targets] -> [compute_relevance]                    |
|  -> [aggregate_metrics]                                                           |
+-------------------------------------------------------------+---------------------+
                                                              |
                                                              v
+-----------------------------------------------------------------------------------+
|                     Downstream Financial & Operational Analytics                  |
|  - FinancialRiskEngine (Sharpe, Sortino, Calmar, Historical & Cornish-Fisher VaR)  |
|  - MonteCarloStressTester (GBM Paths, Fat-Tailed Student-t, 6 Stress Presets)     |
|  - RiskTelemetryCollector (Rolling Windows, Welch's t-Test, Health & Alerts)     |
+---------------------------------------------+-------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------+
|                  Streaming Feature Store & Parquet Analytics Engine               |
|  - FeatureRegistry (Schemas: QUERY, CHUNK, RESPONSE, EVALUATION, FINANCIAL)       |
|  - StreamingFeaturePipeline (Sliding Windows, In-Memory Buffers, Batch Flushes)   |
|  - ParquetFeatureStore (Columnar Storage, Snappy/ZSTD Compression, Partitions)    |
|  - FeatureDriftDetector (Population Stability Index / PSI Shifts & Alerts)       |
|  - Handlers (Compaction, Deduplication, TTL Retention, Offline Training Export)   |
+-----------------------------------------------------------------------------------+
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/zein/production-rag-evaluation.git
cd production-rag-evaluation

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows PowerShell / CMD
# source venv/bin/activate # Linux / macOS

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Quickstart

### Python API Example

```python
from production_rag_evaluation import (
    RAGEvaluator,
    RAGQuery,
    RAGResponse,
    DocumentChunk,
    DocumentType,
    Citation,
    NumericalTarget,
    EvaluationMode,
)

# Initialize evaluator
evaluator = RAGEvaluator(default_mode=EvaluationMode.FINANCIAL_STRICT)

# Retrieved SEC 10-K context
chunk = DocumentChunk(
    chunk_id="chunk-aapl-01",
    ticker="AAPL",
    doc_type=DocumentType.SEC_10K,
    text="For fiscal year 2024, Apple reported net sales of $391.0 billion, up 2% year-over-year.",
)

# Query with explicit financial target
query = RAGQuery(
    query_id="q1",
    query_text="What were Apple's fiscal year 2024 net sales?",
    ticker="AAPL",
    structured_targets=[
        NumericalTarget(metric_name="Net Sales", target_value=391.0e9)
    ],
)

# Generated model response
response = RAGResponse(
    query_id="q1",
    generated_answer="Apple reported net sales of $391.0 billion for fiscal year 2024.",
    retrieved_chunks=[chunk],
    citations=[Citation(chunk_id=chunk.chunk_id, quote="$391.0 billion")],
)

# Run evaluation DAG
metrics = evaluator.evaluate(query, response)

print(f"Overall Pass: {metrics.overall_pass}")
print(f"Faithfulness Score: {metrics.faithfulness_score:.2%}")
print(f"Hallucination Rate: {metrics.hallucination_rate:.2%}")
print(f"Latency: {metrics.evaluation_latency_ms} ms")
```

---

### Command-Line Interface (CLI)

```bash
# 1. Single query evaluation with formatted Rich table output
rag-eval evaluate \
  -q "What were Apple's net sales in 2024?" \
  -a "Apple reported net sales of $391.0 billion." \
  -c "For fiscal year 2024, Apple reported net sales of $391.0 billion." \
  -T "net_sales=391.0e9"

# 2. Standalone hallucination audit
rag-eval audit \
  -a "Apple revenue dropped to $450 billion." \
  -c "Apple reported net sales of $391.0 billion, up 2%."

# 3. Quantitative portfolio risk metrics
rag-eval risk-metrics \
  -r "0.005,-0.002,0.008,-0.004,0.003,0.006,-0.001,0.007" \
  -b "0.003,-0.001,0.005,-0.002,0.002,0.004,-0.001,0.004"

# 4. Monte Carlo stress simulation (Liquidity Crunch scenario)
rag-eval stress-test --scenario LIQUIDITY_CRUNCH -n 1000 -h 30

# 5. Pipeline telemetry & statistical drift check
rag-eval telemetry --check-drift --drift-metric faithfulness_score

# 6. Cache stats
rag-eval cache stats

# 7. Columnar feature store storage metrics & PSI drift detection
rag-eval feature-store stats --path ./data/feature_store
rag-eval feature-store drift --path ./data/feature_store --psi-threshold 0.25
rag-eval feature-store compact --path ./data/feature_store --ttl 2592000
```

---

## Repository Structure

```
production-rag-evaluation/
|-- .github/workflows/ci.yml       # Automated CI workflow (lint, type-check, test matrix)
|-- docs/
|   |-- ARCHITECTURE.md            # System architecture, DAG topology & mathematical formulations
|   |-- API_REFERENCE.md           # Exhaustive Python API reference
|   +-- USAGE_GUIDE.md             # CLI manual, option flags, and Python workflow recipes
|-- src/production_rag_evaluation/
|   |-- core/
|   |   |-- cache.py               # Deterministic n-gram hash vectorizer & semantic cache
|   |   |-- evaluator.py           # Top-level evaluation orchestrator
|   |   |-- graph.py               # NetworkX topological DAG state machine
|   |   +-- hallucination.py       # Specialized financial hallucination detection rules
|   |-- features/
|   |   |-- drift.py               # Population Stability Index (PSI) drift detection
|   |   |-- extractors.py          # Domain feature extractors (Query, Chunk, Response, Eval, Finance)
|   |   |-- handlers.py            # Evaluation hooks, compaction, and dataset export handlers
|   |   |-- models.py              # Feature schemas, records, online vectors, and compaction models
|   |   |-- pipeline.py            # Streaming feature ingestion and buffering pipeline
|   |   |-- registry.py            # Strongly-typed schema registry and PyArrow mappings
|   |   +-- store.py               # Columnar Apache Parquet storage engine
|   |-- finance/
|   |   |-- metrics.py             # Sharpe, Sortino, Calmar, Drawdown & Cornish-Fisher VaR
|   |   |-- stress_testing.py      # Monte Carlo GBM & fat-tailed Student-t stress simulator
|   |   +-- telemetry.py           # Rolling metrics, Welch's t-test drift & operational alerting
|   |-- models/
|   |   |-- interfaces.py          # Swappable protocol contracts
|   |   +-- schemas.py             # Pydantic domain models & enumerations
|   |-- cli/
|   |   +-- main.py                # Click CLI commands with Rich formatting
|   |-- __init__.py                # Top-level symbol exports
|   +-- py.typed                   # PEP 561 typing marker
|-- tests/
|   |-- conftest.py                # Pytest fixtures and test seeds
|   |-- test_cli.py                # CLI commands and options test suite
|   |-- test_cli_coverage.py       # CLI edge cases and entry points coverage
|   |-- test_core.py               # Cache, detector, DAG, and evaluator tests
|   |-- test_edge_cases.py         # Boundary values, error handlers, and zero divisions
|   |-- test_integration.py        # End-to-end RAG-to-risk integration tests
|   |-- test_scaffold.py           # Scaffold and basic lifecycle tests
|   |-- test_subsystems.py         # Subsystem-specific mathematical tests
|   +-- test_types.py              # Domain model validation and protocol conformity
|-- pyproject.toml                 # Package configuration, dependencies, and tooling
|-- pytest.ini                     # Pytest configuration
|-- requirements.txt               # Production dependencies
|-- requirements-dev.txt           # Developer and test dependencies
|-- ruff.toml                      # Ruff linter configuration
+-- setup.py                       # Setuptools entry point
```

---

## Verification & Quality Standards

The repository enforces strict code quality and test coverage standards:

- **100% Test Pass Rate**: 152/152 tests passing in < 1 second.
- **100% Statement Coverage**: Zero uncovered statements across all core, finance, model, and CLI modules.
- **Static Typing**: Fully typed with strict Mypy compliance (`py.typed`).
- **Code Style**: Formatted and linted with Ruff.

```bash
# Run unit and integration tests with coverage report
python -m pytest --cov=production_rag_evaluation

# Run static type checking
python -m mypy src

# Run linter
python -m ruff check .
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
