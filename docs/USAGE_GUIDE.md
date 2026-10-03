# User & CLI Guide

Practical guide, CLI manual, and Python integration recipes for **`production-rag-evaluation`**.

---

## 1. Installation & Environment Setup

### 1.1 Local Installation
Clone the repository and install in editable mode:

```bash
git clone https://github.com/zein/production-rag-evaluation.git
cd production-rag-evaluation

# Create and activate Python virtual environment (Python 3.10+)
python -m venv venv
venv\Scripts\activate          # Windows PowerShell / CMD
# source venv/bin/activate     # Linux / macOS

# Install package with developer tooling
pip install -e ".[dev]"
```

### 1.2 Verification
Confirm that the package and CLI commands are properly registered:

```bash
rag-eval --help
# Or alias
production-rag-eval --help
```

Run test suite to verify 100% environment compatibility:
```bash
python -m pytest
```

---

## 2. Command-Line Interface (CLI) Manual

`rag-eval` outputs interactive, styled [Rich](https://github.com/Textualize/rich) terminal tables by default and supports `--json-out` across all commands for script piping and CI/CD pipelines.

---

### 2.1 Single Query Evaluation (`evaluate`)

Evaluates an individual question, generated answer, and retrieved context against the multi-factor DAG.

```bash
rag-eval evaluate [OPTIONS]
```

#### Options
- `-q, --query TEXT`: Inbound financial query or file path prefixed with `@` (Required).
- `-a, --answer TEXT`: Generated LLM response text or file path prefixed with `@` (Required).
- `-c, --context TEXT`: Retrieved context text chunk or file path prefixed with `@` (Required).
- `-t, --ticker TEXT`: Stock ticker symbol (Default: `AAPL`).
- `-p, --period TEXT`: Reporting fiscal period (Default: `FY2024`).
- `-m, --mode [FAST|THOROUGH|FINANCIAL_STRICT|AUDIT]`: Evaluation depth mode (Default: `FINANCIAL_STRICT`).
- `-T, --target TEXT`: Expected numerical target (e.g. `net_sales=391.0e9` or `391.0e9`). Can be repeated.
- `--no-cache`: Disables semantic evaluation cache lookup and storage.
- `-o, --output PATH`: Write JSON evaluation metrics to output file.
- `--json-out`: Print raw JSON to stdout instead of Rich tables.

#### Example 1: Basic Evaluation with Inlined Arguments
```bash
rag-eval evaluate \
  -q "What were Apple's fiscal year 2024 total net sales and gross margin?" \
  -a "Apple reported net sales of $391.0 billion for fiscal year 2024, with gross margin of 46.2%." \
  -c "For fiscal year 2024, Apple reported net sales of $391.0 billion, up 2% compared to 2023. Gross margin was 46.2%." \
  -t AAPL \
  -p FY2024
```

#### Example 2: Evaluating with Specific Financial KPI Targets
```bash
rag-eval evaluate \
  -q "What were Apple's 2024 net sales and Services revenue?" \
  -a "Apple recorded net sales of $391.0 billion and Services revenue of $96.2 billion." \
  -c "Apple reported net sales of $391.0 billion. Services revenue reached an all-time record of $96.2 billion." \
  -T "net_sales=391.0e9" \
  -T "services=96.2e9" \
  -m FINANCIAL_STRICT
```

#### Example 3: Loading Long Disclosures Using `@file` Prefix
```bash
rag-eval evaluate \
  -q @"queries/q_revenue.txt" \
  -a @"outputs/model_answer.txt" \
  -c @"data/sec_10k_item7.txt" \
  --output report.json \
  --json-out
```

---

### 2.2 Batch Evaluation (`batch-evaluate`)

Processes a collection of queries from a JSON array or JSONL dataset.

```bash
rag-eval batch-evaluate [OPTIONS]
```

#### Options
- `-i, --input PATH`: Path to input JSON or JSONL file containing evaluation pairs (Required).
- `-o, --output PATH`: Output path for consolidated evaluation report JSON.
- `-m, --mode [FAST|THOROUGH|FINANCIAL_STRICT|AUDIT]`: Evaluation depth mode (Default: `THOROUGH`).
- `--no-cache`: Bypass semantic caching during batch run.
- `--json-out`: Output summary as JSON to stdout.

#### Input Format (`queries.json`)
```json
[
  {
    "query": "What was Microsoft's Azure revenue growth?",
    "answer": "Azure revenue grew 29% in the fiscal quarter.",
    "context": "Server products and cloud services revenue increased 21%, driven by Azure growth of 29%.",
    "ticker": "MSFT",
    "period": "Q3 2024",
    "targets": ["azure_growth=0.29"]
  },
  {
    "query": "What were Alphabet's Google Services revenues?",
    "answer": "Google Services revenue was $70.4 billion, down 5%.",
    "context": "Google Services revenues were $70.4 billion, up 12% year-over-year.",
    "ticker": "GOOGL",
    "period": "Q2 2024"
  }
]
```

#### Running Batch Evaluation
```bash
rag-eval batch-evaluate -i queries.json -o batch_results.json
```

Outputs an aggregated performance summary:
- Total queries processed
- Pass rate (%)
- Mean Faithfulness score
- Mean Relevance score
- Average latency and 95th-percentile (p95) latency

---

### 2.3 Standalone Hallucination Audit (`audit` / `detect-hallucinations`)

Directly checks text against context chunks without running the full DAG pipeline. Ideal for ad-hoc QA by financial analysts.

```bash
rag-eval audit -a "Apple net sales dropped to $450 billion." \
  -c "Apple reported net sales of $391.0 billion, up 2%."
```

Flags detected anomalies:
- Numerical distortions (e.g. `$450B` vs `$391B`).
- Contradictory polarities (`dropped` vs `up 2%`).
- Cross-company ticker swaps.
- Temporal mismatches.

---

### 2.4 Quantitative Risk Metrics (`risk-metrics`)

Computes institutional quantitative risk metrics on portfolio return series.

```bash
rag-eval risk-metrics [OPTIONS]
```

#### Options
- `-r, --returns TEXT`: Comma-separated returns or `@file` path (e.g. `0.01,-0.02,0.015`).
- `-b, --benchmark TEXT`: Benchmark return series for calculating Alpha, Beta, and Information Ratio.
- `--rf FLOAT`: Annualized risk-free rate (Default: `0.045` / 4.5%).
- `--ppy INTEGER`: Trading periods per year (Default: `252` for daily).
- `--json-out`: Print JSON output.

#### Example: Analyzing Portfolio Returns with Benchmark
```bash
rag-eval risk-metrics \
  -r "0.005,-0.002,0.008,-0.004,0.003,0.006,-0.001,0.007" \
  -b "0.003,-0.001,0.005,-0.002,0.002,0.004,-0.001,0.004" \
  --rf 0.045
```

Reports:
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Peak-to-Trough Drawdown
- Annualized Return and Volatility
- 1-Day 95% Historical VaR & CVaR (Expected Shortfall)
- 1-Day 95% Cornish-Fisher Parametric VaR
- CAPM Beta & Alpha against Benchmark
- Information Ratio & Omega Ratio

---

### 2.5 Stochastic Monte Carlo Stress Testing (`stress-test`)

Simulates portfolio return distributions under market shocks or evaluates RAG hallucination error cascades.

```bash
rag-eval stress-test [OPTIONS]
```

#### Options
- `-r, --returns TEXT`: Empirical return series or `@file`.
- `-n, --simulations INTEGER`: Number of simulation trajectories (Default: `1000`).
- `-h, --horizon INTEGER`: Projection horizon in trading days (Default: `30`).
- `-s, --volatility-shock FLOAT`: Volatility multiplier (Default: `1.5`).
- `-d, --drift-shock FLOAT`: Annualized drift shock (Default: `0.0`).
- `--max-drawdown FLOAT`: Max allowable drawdown before failing stress test (Default: `0.25`).
- `--scenario [BASELINE|VOLATILITY_SPIKE|LIQUIDITY_CRUNCH|RATE_SHOCK|BLACK_SWAN|HALLUCINATION_CASCADE]`: Preset scenario.
- `--fat-tailed`: Sample shocks from Student-t distribution instead of Gaussian.
- `--df FLOAT`: Degrees of freedom for Student-t distribution (Default: `5.0`).
- `--json-out`: Print JSON output.

#### Example: Simulating a Black Swan Stress Event
```bash
rag-eval stress-test \
  --scenario BLACK_SWAN \
  -n 2000 \
  -h 60 \
  --fat-tailed \
  --df 4.0
```

Outputs terminal wealth percentiles ($P_5, P_{50}, P_{95}$), worst-case drawdown, and binary compliance status (`stress_pass`).

---

### 2.6 System Telemetry & Drift Monitoring (`telemetry`)

Inspects pipeline operational health, cache efficiency, and tests for statistical drift.

```bash
rag-eval telemetry [OPTIONS]
```

#### Options
- `--history-file PATH`: Path to prior evaluation history JSON.
- `--returns TEXT`: Return series for integrated risk checks.
- `--check-drift`: Execute Welch's two-sample t-test for performance drift.
- `--drift-metric TEXT`: Target metric for drift check (Default: `faithfulness_score`).
- `--drift-alpha FLOAT`: Significance level $\alpha$ (Default: `0.05`).
- `--json-out`: Output report as JSON.

#### Example: Checking Statistical Drift
```bash
rag-eval telemetry \
  --history-file "logs/eval_history.json" \
  --check-drift \
  --drift-metric faithfulness_score
```

---

### 2.7 Semantic Cache Management (`cache`)

Monitor and clear the semantic evaluation cache.

#### Inspect Statistics
```bash
rag-eval cache stats
```
Displays:
- Cached entries count
- Max capacity limit
- Cache hits vs misses
- Overall hit rate percentage
- Estimated execution latency saved in ms

#### Flush Cache
```bash
rag-eval cache clear
```

---

### 2.8 Feature Store Management (`feature-store` / `features`)

The feature store command group inspects, manages, compacts, and exports columnar Parquet partitions and audits statistical feature drift.

#### 1. Inspect Feature Store Storage Metrics (`stats`)
```bash
rag-eval feature-store stats --path ./data/feature_store
```
Displays a Rich table of feature groups (`QUERY`, `CHUNK`, `RESPONSE`, `EVALUATION`, `FINANCIAL`), row counts, column counts, row group counts, and physical byte sizes. Add `--json-out` for automated observability pipelines.

#### 2. List Registered Feature Schemas (`list-schemas`)
```bash
# View all schemas
rag-eval feature-store list-schemas

# Filter to a specific group
rag-eval feature-store list-schemas --group EVALUATION
```
Displays registered features, types (`FLOAT`, `INT`, `STRING`, `BOOLEAN`, `VECTOR`, `JSON`), nullability, and field descriptions.

#### 3. Compact Parquet Partitions & Enforce TTL (`compact`)
```bash
# Compact all groups, deduplicate entity IDs, and prune records older than 30 days (2,592,000s)
rag-eval feature-store compact \
  --path ./data/feature_store \
  --ttl 2592000 \
  --compression snappy

# Compact specific feature group
rag-eval feature-store compact \
  --path ./data/feature_store \
  --group RESPONSE \
  --compression zstd
```

#### 4. Export Aligned Point-in-Time Dataset (`export`)
```bash
# Export from comma-separated entity IDs
rag-eval feature-store export \
  --path ./data/feature_store \
  --entities "entity_001,entity_002,entity_003" \
  --output ./exports/rag_training_set.parquet \
  --format PARQUET

# Export from an entity list file
rag-eval feature-store export \
  --path ./data/feature_store \
  --entities ./entity_ids.txt \
  --output ./exports/rag_training_set.csv \
  --format CSV \
  --group EVALUATION \
  --group FINANCIAL
```

#### 5. Audit Statistical Feature Drift via PSI (`drift`)
```bash
# Audit drift across all groups and features against baseline
rag-eval feature-store drift \
  --path ./data/feature_store \
  --psi-threshold 0.25

# Audit specific feature and group with JSON output
rag-eval feature-store drift \
  --path ./data/feature_store \
  --group EVALUATION \
  --feature faithfulness_score \
  --json-out
```

#### 6. Streaming into Feature Store from Evaluator CLI
The `evaluate` and `batch-evaluate` commands support `--feature-store-path` to stream extracted feature records directly into the columnar store in real time:
```bash
rag-eval evaluate \
  -q "What were Apple's net sales in 2024?" \
  -a "Apple reported net sales of $391.0 billion." \
  -c "For fiscal year 2024, Apple reported net sales of $391.0 billion." \
  --feature-store-path ./data/feature_store
```

---

## 3. Python API Workflows & Recipes

### Workflow 1: End-to-End SEC 10-K RAG Evaluation

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

# 1. Initialize evaluator
evaluator = RAGEvaluator(
    numerical_tolerance_pct=0.01,
    default_mode=EvaluationMode.FINANCIAL_STRICT,
)

# 2. Define ground-truth retrieved chunk from 10-K
chunk = DocumentChunk(
    chunk_id="chunk-aapl-10k-item7",
    text=(
        "Total net sales were $391.035 billion in fiscal 2024, compared to $383.285 billion in 2023. "
        "Products net sales were $294.866 billion and Services net sales were $96.169 billion."
    ),
    doc_type=DocumentType.SEC_10K,
    ticker="AAPL",
    period="FY2024",
    section="Item 7. Management's Discussion and Analysis",
)

# 3. Formulate query and expected KPI targets
query = RAGQuery(
    query_id="q-001",
    query_text="What were Apple's total net sales and Services revenue for fiscal 2024?",
    ticker="AAPL",
    period="FY2024",
    structured_targets=[
        NumericalTarget(metric_name="Total Net Sales", target_value=391.035e9),
        NumericalTarget(metric_name="Services Net Sales", target_value=96.169e9),
    ],
)

# 4. Formulate generated response with grounding citations
response = RAGResponse(
    query_id="q-001",
    generated_answer=(
        "In fiscal 2024, Apple achieved total net sales of $391.035 billion, "
        "with Services net sales reaching $96.169 billion."
    ),
    retrieved_chunks=[chunk],
    citations=[
        Citation(chunk_id=chunk.chunk_id, quote="$391.035 billion"),
        Citation(chunk_id=chunk.chunk_id, quote="$96.169 billion"),
    ],
)

# 5. Run evaluation
metrics = evaluator.evaluate(query, response)

print(f"Overall Pass: {metrics.overall_pass}")
print(f"Faithfulness Score: {metrics.faithfulness_score:.2%}")
print(f"Answer Relevance: {metrics.answer_relevance_score:.2%}")
print(f"Citation Grounding: {metrics.citation_grounding_score:.2%}")
print(f"Hallucination Rate: {metrics.hallucination_rate:.2%}")
print(f"Latency: {metrics.evaluation_latency_ms} ms")
```

---

### Workflow 2: Automated Hallucination Guardrail in CI/CD

```python
import pytest
from production_rag_evaluation import HallucinationDetector, DocumentChunk, Severity

@pytest.fixture
def detector():
    return HallucinationDetector(numerical_tolerance_pct=0.01)

def test_rag_prevents_critical_financial_hallucinations(detector):
    context = [
        DocumentChunk(
            chunk_id="c1",
            ticker="TSLA",
            text="Automotive revenues decreased 3% year-over-year to $21.3 billion.",
        )
    ]
    
    # Generated response hallucinating growth
    hallucinated_answer = "Tesla's automotive revenue grew significantly to $25.0 billion."
    
    findings = detector.analyze(hallucinated_answer, context)
    
    # Must catch contradiction and distortion
    critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
    assert len(critical_findings) > 0, "Failed to flag directional contradiction"
```

---

### Workflow 3: Quant Portfolio Risk & Monte Carlo Stress Testing

```python
import numpy as np
from production_rag_evaluation import (
    FinancialRiskEngine,
    MonteCarloStressTester,
    StressScenarioType,
)

# 100 days of portfolio return history
returns = np.array([0.005, -0.002, 0.008, -0.004, 0.003, 0.006, -0.001, 0.007] * 12)

# Calculate institutional risk metrics
risk_engine = FinancialRiskEngine(risk_free_rate=0.045)
risk_metrics = risk_engine.compute_all_metrics(returns)

print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
print(f"Sortino Ratio: {risk_metrics.sortino_ratio:.2f}")
print(f"Max Drawdown: {risk_metrics.max_drawdown:.2%}")
print(f"1-Day 95% VaR: {risk_metrics.var_95:.4f}")
print(f"1-Day 95% CVaR: {risk_metrics.cvar_95:.4f}")

# Execute Monte Carlo stress testing under Liquidity Crunch
stress_tester = MonteCarloStressTester(random_seed=42)
stress_result = stress_tester.run_stress_scenario(
    returns=returns,
    scenario_type=StressScenarioType.LIQUIDITY_CRUNCH,
    n_simulations=1000,
    horizon_days=30,
)

print(f"Stress Passed: {stress_result.stress_pass}")
print(f"5th Percentile Terminal: {stress_result.p5_terminal}")
print(f"Worst-Case Drawdown: {stress_result.worst_case_drawdown:.2%}")
```

---

### Workflow 4: Real-time Telemetry, Drift Monitoring, and Alert Callbacks

```python
from production_rag_evaluation import (
    RiskTelemetryCollector,
    TelemetryAlert,
    AlertSeverity,
)

collector = RiskTelemetryCollector()

# Register alert listener callback (e.g. for Slack / PagerDuty webhook)
def on_alert(alert: TelemetryAlert):
    print(f"[{alert.severity.value}] {alert.metric_name}: {alert.message}")

collector.add_alert_listener(on_alert)

# Feed incoming evaluations from production streaming pipeline
# ... collector.record_evaluation(metrics) ...

# Periodically test for metric drift (e.g. daily scheduled cron)
drift_result = collector.detect_drift(
    metric="faithfulness_score",
    baseline_size=20,
    test_size=10,
    alpha=0.05,
)

if drift_result.get("drift_detected"):
    print(f"WARNING: Statistically significant drift detected! p-value: {drift_result['p_value']}")
```

---

### Workflow 5: Streaming Feature Ingestion & Online Feature Vector Retrieval

```python
from production_rag_evaluation import RAGEvaluator, RAGQuery, RAGResponse, DocumentChunk
from production_rag_evaluation.features import (
    ParquetFeatureStore,
    StreamingFeaturePipeline,
    EvaluationFeatureHandler,
)

# 1. Initialize store and streaming pipeline
store = ParquetFeatureStore(base_path="./data/feature_store")
pipeline = StreamingFeaturePipeline(store=store, batch_size=50)
handler = EvaluationFeatureHandler(pipeline=pipeline)

# 2. Evaluate RAG interaction
evaluator = RAGEvaluator()
query = "What was Tesla's automotive gross margin?"
response = "Tesla reported automotive gross margin of 18.2% excluding regulatory credits."
chunk = "Automotive gross margin was 18.2% in the quarter."

metrics = evaluator.evaluate(
    query=RAGQuery(text=query, ticker="TSLA"),
    response=RAGResponse(text=response),
    context_chunks=[DocumentChunk(chunk_id="chunk-1", text=chunk)],
)

# 3. Stream features directly through integration handler
vector = handler.on_evaluation_completed(
    metrics=metrics,
    query=query,
    answer=response,
    context=chunk,
    ticker="TSLA",
    period="Q3-2024",
)

# 4. Immediate online serving (<1ms lookup)
print(f"Entity ID: {vector.entity_id}")
print(f"Extracted features count: {len(vector.features)}")
print(f"Faithfulness feature: {vector.features.get('faithfulness_score')}")

# Explicit flush to columnar Parquet
flushed_count = pipeline.flush()
print(f"Flushed {flushed_count} records to Parquet.")
```

---

### Workflow 6: Population Stability Index (PSI) Drift Detection & Telemetry Alerts

```python
from production_rag_evaluation import RiskTelemetryCollector
from production_rag_evaluation.features import (
    ParquetFeatureStore,
    FeatureDriftDetector,
    FeatureGroup,
)

store = ParquetFeatureStore(base_path="./data/feature_store")
detector = FeatureDriftDetector(store=store)
collector = RiskTelemetryCollector()

# Run PSI drift detection across all stored features
reports = detector.detect_all_drift(group=FeatureGroup.EVALUATION, psi_threshold=0.25)

for report in reports:
    status = "DRIFT DETECTED" if report.drift_detected else "STABLE"
    print(f"[{status}] Feature: {report.feature_name} | PSI: {report.psi_score:.4f} (Threshold: {report.threshold})")

# Generate telemetry alerts and register with RiskTelemetryCollector
alerts = detector.generate_telemetry_alerts(collector=collector, reports=reports)
for alert in alerts:
    print(f"Alert Dispatched: [{alert.severity.value}] {alert.message}")
```

---

### Workflow 7: Partition Compaction, Deduplication & TTL Retention

```python
from production_rag_evaluation.features import (
    ParquetFeatureStore,
    FeatureCompactionHandler,
    FeatureGroup,
)

store = ParquetFeatureStore(base_path="./data/feature_store")
compactor = FeatureCompactionHandler(store=store)

# Compact response features: deduplicate entity IDs and prune records older than 14 days
result = compactor.compact_group(
    group=FeatureGroup.RESPONSE,
    ttl_seconds=14 * 86400,
    compression="snappy",
)

print(f"Group: {result.feature_group.value}")
print(f"Original Records: {result.original_records}")
print(f"Retained Records: {result.retained_records}")
print(f"Purged / Deduplicated: {result.purged_records}")
print(f"Compacted File Size: {result.file_size_bytes:,} bytes")
```

---

### Workflow 8: Offline Multi-Group Training Matrix Export

```python
from production_rag_evaluation.features import (
    ParquetFeatureStore,
    FeatureExportHandler,
    FeatureGroup,
    ExportFormat,
)

store = ParquetFeatureStore(base_path="./data/feature_store")
exporter = FeatureExportHandler(store=store)

# Discover entities to export
entity_ids = store.list_entities(FeatureGroup.EVALUATION, limit=500)

# Export point-in-time merged matrix across EVALUATION, FINANCIAL, and QUERY groups
dataset_path = exporter.export_dataset(
    entity_ids=entity_ids,
    output_path="./exports/rag_eval_training.parquet",
    format=ExportFormat.PARQUET,
    groups=[FeatureGroup.EVALUATION, FeatureGroup.FINANCIAL, FeatureGroup.QUERY],
)

print(f"Exported training dataset to: {dataset_path}")
```
