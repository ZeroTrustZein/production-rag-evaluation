# Python API Reference

Comprehensive reference for all public classes, functions, schemas, protocols, and data models in **`production-rag-evaluation`**.

---

## Table of Contents

- [1. Package Root (`production_rag_evaluation`)](#1-package-root-production_rag_evaluation)
- [2. Domain Schemas & Models (`models.schemas`)](#2-domain-schemas--models-modelsschemas)
- [3. Protocol Contracts (`models.interfaces`)](#3-protocol-contracts-modelsinterfaces)
- [4. RAG Evaluator (`core.evaluator`)](#4-rag-evaluator-coreevaluator)
- [5. Evaluation Workflow Graph (`core.graph`)](#5-evaluation-workflow-graph-coregraph)
- [6. Hallucination Detector (`core.hallucination`)](#6-hallucination-detector-corehallucination)
- [7. Semantic Cache (`core.cache`)](#7-semantic-cache-corecache)
- [8. Financial Risk Engine (`finance.metrics`)](#8-financial-risk-engine-financemetrics)
- [9. Monte Carlo Stress Tester (`finance.stress_testing`)](#9-monte-carlo-stress-tester-financestress_testing)
- [10. Risk Telemetry Collector (`finance.telemetry`)](#10-risk-telemetry-collector-financetelemetry)
- [11. CLI Commands & Options (`cli.main`)](#11-cli-commands--options-climain)

---

## 1. Package Root (`production_rag_evaluation`)

Exported symbols available directly from top-level `import production_rag_evaluation` or `from production_rag_evaluation import ...`:

```python
from production_rag_evaluation import (
    # Core Engines
    RAGEvaluator,
    EvaluationGraph,
    HallucinationDetector,
    SemanticCache,
    FinancialRiskEngine,
    MonteCarloStressTester,
    RiskTelemetryCollector,
    # Schemas
    DocumentChunk,
    Citation,
    RAGQuery,
    RAGResponse,
    NumericalTarget,
    EvaluationMetrics,
    RiskMetrics,
    MonteCarloStressResult,
    TelemetryReport,
    TelemetryAlert,
    BatchEvaluationSummary,
    # Enumerations
    DocumentType,
    Severity,
    HallucinationType,
    RiskProfile,
    StressScenarioType,
    EvaluationMode,
    ClaimStatus,
    ReturnFrequency,
    AlertSeverity,
    SystemHealthStatus,
    # Version
    __version__,
)
```

---

## 2. Domain Schemas & Models (`models.schemas`)

Module: `production_rag_evaluation.models.schemas`

### Enumerations

#### `DocumentType(str, Enum)`
Supported financial document categories:
- `EARNINGS_CALL`
- `SEC_10K`
- `SEC_10Q`
- `SEC_8K`
- `PRESS_RELEASE`
- `ANALYST_REPORT`
- `FINANCIAL_NEWS`

#### `Severity(str, Enum)`
Severity classifications for hallucination findings:
- `LOW`: Minor cosmetic distortion or slight ambiguity.
- `MEDIUM`: Moderate ungrounded statement with low risk impact.
- `HIGH`: Factual or numerical distortion impacting an investment metric.
- `CRITICAL`: Direct factual contradiction or inverted polarity in earnings claims.

#### `HallucinationType(str, Enum)`
Failure mode categories:
- `NUMERICAL_DISTORTION`: Extracted figures do not match context within tolerance.
- `UNGROUNDED_CLAIM`: Statement contains keywords or figures unsupported by context.
- `CONTRADICTION`: Claim directly contradicts context polarity (e.g. up vs down).
- `ENTITY_SWAP`: Mentions incorrect companies or tickers (e.g. MSFT instead of AAPL).
- `TEMPORAL_MISMATCH`: Refers to incorrect reporting period or fiscal year.
- `OMISSION_OF_RISK`: Omits crucial risk disclosures stated in source filing.

#### `EvaluationMode(str, Enum)`
Execution depth modes for evaluation:
- `FAST`: Approximate checks, higher tolerance ($\pm 5\%$), fast turnaround.
- `THOROUGH`: Balanced multi-factor analysis ($\pm 1\%$). Default.
- `FINANCIAL_STRICT`: Strict financial figures, zero critical tolerance ($\pm 0.1\%$).
- `AUDIT`: Exhaustive trace verification ($\pm 0.01\%$).

#### `ClaimStatus(str, Enum)`
Status of an individual claim verification:
- `VERIFIED`: Confirmed by retrieved context.
- `REFUTED`: Directly contradicted by retrieved context.
- `UNGROUNDED`: Absent from context, unsupported extrapolation.
- `AMBIGUOUS`: Insufficient context to definitively confirm or deny.

#### `StressScenarioType(str, Enum)`
Preset macroeconomic and market stress scenarios:
- `BASELINE`
- `VOLATILITY_SPIKE`
- `LIQUIDITY_CRUNCH`
- `RATE_SHOCK`
- `BLACK_SWAN`
- `HALLUCINATION_CASCADE`

#### `SystemHealthStatus(str, Enum)`
Pipeline health status:
- `HEALTHY`
- `DEGRADED`
- `UNHEALTHY`

---

### Core Data Models

#### `DocumentChunk`
Represents an individual text chunk retrieved from financial corpus.
- `chunk_id: str`: Unique chunk identifier.
- `text: str`: Raw textual chunk content (min length 1).
- `doc_type: DocumentType = DocumentType.SEC_10K`: Source filing type.
- `ticker: str`: Stock ticker symbol (e.g., `AAPL`).
- `period: Optional[str] = None`: Fiscal period (e.g., `FY2024`, `Q3 2024`).
- `section: Optional[str] = None`: Filing section (e.g., `Item 7. MD&A`).
- `source_url: Optional[str] = None`: Source link or file path.
- `metadata: Dict[str, Any] = {}`: Arbitrary key-value metadata.

#### `Citation`
Grounding citation linking a generated statement back to a source chunk.
- `chunk_id: str`: Target document chunk ID.
- `quote: str`: Excerpt quote string.
- `confidence: float = 1.0`: Match confidence ($0.0 \le c \le 1.0$).
- `source_ticker: Optional[str] = None`: Ticker symbol associated with citation.
- `char_offset: Optional[Tuple[int, int]] = None`: Starting and ending character offset.

#### `NumericalTarget`
Explicit target constraint for verifying financial numbers.
- `metric_name: str`: KPI name (e.g., `"Net Sales"`, `"Gross Margin"`).
- `target_value: float`: Expected numerical value.
- `unit: Optional[str] = None`: Unit or currency (e.g., `"USD"`, `"%"`, `"shares"`).
- `tolerance_pct: float = 0.01`: Allowable percentage deviation ($0.0$ to $1.0$).
- `is_currency: bool = True`: Whether the target represents currency.

#### `RAGQuery`
Inbound query passed to the evaluation harness.
- `query_id: str`: Unique identifier.
- `query_text: str`: Question text.
- `ticker: str`: Subject company ticker.
- `period: Optional[str] = None`: Relevant reporting period.
- `expected_numerical_targets: List[float] = []`: Unnamed target floats.
- `structured_targets: List[NumericalTarget] = []`: Named structured targets.

#### `RAGResponse`
Generated model response and associated retrieval context.
- `query_id: str`: Matched query ID.
- `generated_answer: str`: Response text generated by LLM.
- `retrieved_chunks: List[DocumentChunk] = []`: Context chunks retrieved.
- `citations: List[Citation] = []`: Grounding citations provided in response.
- `latency_ms: float = 0.0`: Model generation latency.
- `token_usage: Dict[str, int] = {}`: Prompt and completion token counts.
- `model_name: Optional[str] = None`: Model identifier string.

#### `HallucinationFinding`
Detailed diagnosis of an identified hallucination.
- `finding_type: HallucinationType`: Category of error.
- `severity: Severity`: Criticality level.
- `claim_text: str`: Specific claim or sentence extracted from answer.
- `reference_text: Optional[str] = None`: Retrieved context passage used as reference.
- `confidence: float = 0.8`: Detection confidence score ($0.0 \le c \le 1.0$).
- `explanation: str`: Human-readable root-cause diagnosis.
- `impact_on_risk: Optional[str] = None`: Downstream risk consequence.

#### `EvaluationMetrics`
Comprehensive multi-dimensional evaluation results for a query-response pair.
- `faithfulness_score: float`: Factuality score ($0.0 \le s \le 1.0$).
- `answer_relevance_score: float`: Relevance of answer to query ($0.0 \le s \le 1.0$).
- `context_recall_score: float`: Measure of retrieved ground truth captured ($0.0 \le s \le 1.0$).
- `context_precision_score: float`: Precision of retrieved chunks ($0.0 \le s \le 1.0$).
- `citation_grounding_score: float`: Ratio of verified citations ($0.0 \le s \le 1.0$).
- `hallucination_rate: float`: Proportion of claims flagged ($0.0 \le r \le 1.0$).
- `findings: List[HallucinationFinding]`: Individual diagnostic findings.
- `claim_verifications: List[ClaimVerificationResult]`: Claim-level statuses.
- `overall_pass: bool`: Binary deployment pass/fail decision.
- `evaluation_latency_ms: float = 0.0`: Duration of evaluation execution in ms.
- `metadata: Dict[str, Any] = {}`: Additional execution metadata.

#### `RiskMetrics`
Portfolio and trading risk metrics calculated on return series.
- `sharpe_ratio: float`: Annualized Sharpe ratio.
- `sortino_ratio: float`: Annualized Sortino ratio.
- `max_drawdown: float`: Maximum peak-to-trough decline ($0.0$ to $1.0$).
- `calmar_ratio: float`: Annualized return over max drawdown.
- `annualized_return: float`: Compounded annualized return.
- `annualized_volatility: float`: Annualized standard deviation of returns.
- `downside_deviation: float`: Annualized downside volatility.
- `var_95: float`: 1-day 95% Historical Value-at-Risk.
- `cvar_95: float`: 1-day 95% Conditional VaR (Expected Shortfall).
- `var_99: Optional[float] = None`: 1-day 99% Value-at-Risk.
- `cvar_99: Optional[float] = None`: 1-day 99% Conditional VaR.
- `beta: Optional[float] = None`: Systematic risk relative to benchmark.
- `alpha: Optional[float] = None`: Jensen's Alpha relative to benchmark.
- `information_ratio: Optional[float] = None`: Active return per unit tracking error.

#### `MonteCarloStressResult`
Outcome of Monte Carlo stress testing simulations.
- `n_simulations: int`: Path count simulated.
- `horizon_steps: int`: Trading days projected.
- `p5_terminal: float`: 5th percentile terminal wealth value.
- `p50_terminal: float`: Median terminal wealth value.
- `p95_terminal: float`: 95th percentile terminal wealth value.
- `mean_terminal: float`: Mean terminal wealth.
- `std_terminal: float`: Standard deviation of terminal wealth.
- `worst_case_drawdown: float`: Deepest drawdown recorded across all paths.
- `stress_pass: bool`: Whether worst-case drawdown stayed within tolerance.
- `scenario_type: StressScenarioType`: Scenario applied.
- `breach_probability: float`: Empirical breach frequency.

#### `TelemetryReport`
Consolidated operational telemetry report.
- `timestamp: str`: ISO-8601 UTC timestamp.
- `total_evaluations: int`: Number of evaluations logged.
- `avg_faithfulness: float`: Historical mean faithfulness.
- `avg_hallucination_rate: float`: Historical mean hallucination rate.
- `cache_hit_rate: float`: Ratio of cache hits.
- `avg_latency_ms: float`: Mean evaluation latency.
- `health_status: SystemHealthStatus`: System operational status.
- `risk_metrics: Optional[RiskMetrics]`: Embedded portfolio risk metrics.
- `monte_carlo_result: Optional[MonteCarloStressResult]`: Stress simulation results.
- `alerts_triggered: List[str]`: String summary of triggered alerts.
- `structured_alerts: List[TelemetryAlert]`: Detailed alert objects.

---

## 3. Protocol Contracts (`models.interfaces`)

Type-checkable Python protocols ensuring loose coupling and swappable components.

- **`EvaluatorProtocol`**: Requires `evaluate`, `evaluate_batch`, `clear_cache`, and `cache_stats`.
- **`HallucinationDetectorProtocol`**: Requires `analyze`, `detect_numerical_distortions`, `detect_directional_contradictions`, and `detect_ungrounded_claims`.
- **`FinancialRiskEngineProtocol`**: Requires `calculate_sharpe_ratio`, `calculate_sortino_ratio`, `calculate_max_drawdown`, `calculate_calmar_ratio`, `calculate_var_cvar`, and `compute_all_metrics`.
- **`MonteCarloStressTesterProtocol`**: Requires `simulate_gbm_paths`, `run_stress_test`, and `stress_test_hallucination_impact`.
- **`SemanticCacheProtocol`**: Requires `get`, `put`, `clear`, and `stats`.
- **`RiskTelemetryCollectorProtocol`**: Requires `record_evaluation`, `generate_report`, and `clear_history`.

---

## 4. RAG Evaluator (`core.evaluator`)

### Class: `RAGEvaluator`

Primary orchestrator coordinating the semantic cache and DAG workflow.

#### Constructor
```python
RAGEvaluator(
    cache_similarity_threshold: float = 0.90,
    cache_max_entries: int = 5000,
    cache_ttl_seconds: float = 86400.0,
    numerical_tolerance_pct: float = 0.01,
    enable_cache: bool = True,
    default_mode: EvaluationMode = EvaluationMode.THOROUGH,
)
```

#### Methods

- **`evaluate(query: RAGQuery, response: RAGResponse, use_cache: Optional[bool] = None, mode: Optional[EvaluationMode] = None) -> EvaluationMetrics`**:
  Evaluates a single query-response pair. Checks semantic cache first if enabled; on cache miss, executes the evaluation DAG and caches the result.

- **`evaluate_batch(pairs: List[Tuple[RAGQuery, RAGResponse]], use_cache: Optional[bool] = None, mode: Optional[EvaluationMode] = None) -> List[EvaluationMetrics]`**:
  Evaluates multiple pairs sequentially, reusing the semantic cache.

- **`evaluate_batch_summary(pairs: List[Tuple[RAGQuery, RAGResponse]], batch_id: Optional[str] = None, use_cache: Optional[bool] = None, mode: Optional[EvaluationMode] = None) -> BatchEvaluationSummary`**:
  Runs batch evaluation and returns aggregated summary statistics including pass rate, mean faithfulness, mean relevance, and p95 latency.

- **`clear_cache() -> None`**:
  Clears all cached entries.

- **`cache_stats() -> Dict[str, Any]`**:
  Returns dictionary of cache health and hit metrics.

---

## 5. Evaluation Workflow Graph (`core.graph`)

### Class: `EvaluationState`
Mutable state container passed across graph nodes:
- `query: RAGQuery`
- `response: RAGResponse`
- `mode: EvaluationMode`
- `claims: List[str]`
- `verified_citations: List[Citation]`
- `hallucinations: List[HallucinationFinding]`
- `claim_verifications: List[ClaimVerificationResult]`
- `target_checks: Dict[str, Any]`
- `faithfulness_score: float`
- `answer_relevance_score: float`
- `context_recall_score: float`
- `context_precision_score: float`
- `citation_grounding_score: float`
- `hallucination_rate: float`
- `overall_pass: bool`
- `execution_trace: List[str]`
- `metrics: Optional[EvaluationMetrics]`

### Class: `EvaluationGraph`
Topological DAG workflow built on `networkx.DiGraph`.

#### Constructor
```python
EvaluationGraph(detector: Optional[HallucinationDetector] = None)
```

#### Methods
- **`run(query: RAGQuery, response: RAGResponse, mode: EvaluationMode = EvaluationMode.THOROUGH) -> EvaluationMetrics`**:
  Initializes `EvaluationState`, traverses the graph in topological order, and constructs the resulting `EvaluationMetrics`.

---

## 6. Hallucination Detector (`core.hallucination`)

### Class: `HallucinationDetector`
Specialized rule-based detector for financial domain anomalies.

#### Constructor
```python
HallucinationDetector(numerical_tolerance_pct: float = 0.01)
```

#### Methods

- **`detect_numerical_distortions(generated_answer: str, context_chunks: List[DocumentChunk]) -> List[HallucinationFinding]`**:
  Extracts and normalizes numbers from generated answers and contexts. Flags deviations outside `numerical_tolerance_pct`.

- **`detect_directional_contradictions(generated_answer: str, context_chunks: List[DocumentChunk]) -> List[HallucinationFinding]`**:
  Detects polarity mismatches (e.g. answer claims revenue grew when context documents a decrease).

- **`detect_ungrounded_claims(generated_answer: str, context_chunks: List[DocumentChunk], min_overlap_ratio: float = 0.30) -> List[HallucinationFinding]`**:
  Segments answer into claims and flags those with less than `min_overlap_ratio` lexical overlap with retrieved context.

- **`detect_entity_swaps(generated_answer: str, context_chunks: List[DocumentChunk]) -> List[HallucinationFinding]`**:
  Extracts stock tickers and flags companies mentioned in the answer that do not exist in the source filing chunks.

- **`detect_temporal_mismatches(generated_answer: str, context_chunks: List[DocumentChunk]) -> List[HallucinationFinding]`**:
  Detects year and period hallucinations where the answer references years absent from the context chunks.

- **`verify_claims(generated_answer: str, context_chunks: List[DocumentChunk]) -> List[ClaimVerificationResult]`**:
  Returns structured claim-by-claim verification statuses (`VERIFIED`, `REFUTED`, `UNGROUNDED`, or `AMBIGUOUS`).

- **`analyze(generated_answer: str, context_chunks: List[DocumentChunk]) -> List[HallucinationFinding]`**:
  Executes all detection rules and returns the aggregated list of findings.

---

## 7. Semantic Cache (`core.cache`)

### Class: `TextEmbedder`
Lightweight CPU-bound n-gram vectorizer ($D=128$).
- **`embed(text: str) -> np.ndarray`**: Returns an L2-normalized 128-dimensional dense vector.

### Class: `SemanticCache`
Thread-safe LRU cache with cosine similarity matching and TTL expiration.

#### Constructor
```python
SemanticCache(
    similarity_threshold: float = 0.88,
    max_entries: int = 5000,
    ttl_seconds: float = 86400.0,
    latency_saved_ms_per_hit: float = 25.0,
)
```

#### Methods
- **`get(query_text: str) -> Optional[Tuple[Any, float]]`**: Retrieves cached value and similarity score, or `None`.
- **`put(query_text: str, value: Any) -> None`**: Inserts or updates an entry. Evicts LRU entry if at capacity.
- **`batch_get(queries: List[str]) -> List[Optional[Tuple[Any, float]]]`**: Batch lookup.
- **`batch_put(items: List[Tuple[str, Any]]) -> None`**: Batch insertion.
- **`find_similar(query_text: str, top_k: int = 3, threshold: Optional[float] = None) -> List[Tuple[str, Any, float]]`**: Returns top-k similar entries above threshold.
- **`clear() -> None`**: Purges all cache entries.
- **`stats() -> Dict[str, Any]`**: Returns hit count, miss count, hit rate, and estimated latency saved.

---

## 8. Financial Risk Engine (`finance.metrics`)

### Class: `FinancialRiskEngine`
Computes quantitative performance and downside risk indicators on portfolio returns.

#### Constructor
```python
FinancialRiskEngine(risk_free_rate: float = 0.045, periods_per_year: int = 252)
```

#### Methods
- **`calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: Optional[float] = None, periods_per_year: Optional[int] = None) -> float`**
- **`calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0.0, risk_free_rate: Optional[float] = None, periods_per_year: Optional[int] = None) -> float`**
- **`calculate_max_drawdown(returns: np.ndarray) -> float`**
- **`calculate_calmar_ratio(returns: np.ndarray, risk_free_rate: Optional[float] = None, periods_per_year: Optional[int] = None) -> float`**
- **`calculate_var_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]`**
- **`calculate_parametric_var(returns: np.ndarray, confidence_level: float = 0.95, cornish_fisher: bool = True) -> float`**
- **`calculate_beta_and_alpha(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray, risk_free_rate: Optional[float] = None, periods_per_year: Optional[int] = None) -> Tuple[float, float]`**
- **`calculate_information_ratio(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray, periods_per_year: Optional[int] = None) -> float`**
- **`calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float`**
- **`calculate_tail_risk_metrics(returns: np.ndarray) -> Tuple[float, float]`**: Returns `(skewness, excess_kurtosis)`.
- **`compute_all_metrics(returns: np.ndarray, risk_free_rate: Optional[float] = None, benchmark_returns: Optional[np.ndarray] = None) -> RiskMetrics`**

---

## 9. Monte Carlo Stress Tester (`finance.stress_testing`)

### Class: `MonteCarloStressTester`
Simulates asset trajectories under market shocks and RAG error injections.

#### Constructor
```python
MonteCarloStressTester(random_seed: Optional[int] = 42)
```

#### Methods
- **`simulate_gbm_paths(initial_value: float = 100.0, mu: float = 0.08, sigma: float = 0.20, horizon_days: int = 60, n_simulations: int = 1000, volatility_shock_multiplier: float = 1.0, drift_shock: float = 0.0) -> np.ndarray`**:
  Returns simulated price paths of shape `(n_simulations, horizon_days + 1)`.

- **`simulate_fat_tailed_paths(initial_value: float = 100.0, mu: float = 0.08, sigma: float = 0.20, horizon_days: int = 60, n_simulations: int = 1000, degrees_of_freedom: float = 5.0, volatility_shock_multiplier: float = 1.0, drift_shock: float = 0.0) -> np.ndarray`**:
  Simulates price paths driven by Student-t innovation shocks.

- **`run_stress_test(returns: np.ndarray, n_simulations: int = 1000, horizon_days: int = 30, volatility_shock: float = 1.5, drift_shock: float = 0.0, max_allowable_drawdown: float = 0.25, degrees_of_freedom: Optional[float] = None) -> MonteCarloStressResult`**:
  Simulates future trajectories based on empirical return parameters under shock conditions.

- **`run_stress_scenario(returns: np.ndarray, scenario_type: StressScenarioType, n_simulations: int = 1000, horizon_days: int = 30) -> MonteCarloStressResult`**:
  Executes a pre-defined macro stress scenario (`BASELINE`, `VOLATILITY_SPIKE`, `LIQUIDITY_CRUNCH`, `RATE_SHOCK`, `BLACK_SWAN`, `HALLUCINATION_CASCADE`).

- **`stress_test_hallucination_impact(base_financial_estimates: List[float], hallucination_probability: float = 0.15, hallucination_distortion_pct: float = 0.25, n_simulations: int = 500) -> float`**:
  Simulates probability that hallucinated outputs induce a $>10\%$ portfolio forecasting breach.

---

## 10. Risk Telemetry Collector (`finance.telemetry`)

### Class: `RiskTelemetryCollector`
Aggregates RAG evaluation outcomes, computes rolling statistics, and monitors statistical drift.

#### Constructor
```python
RiskTelemetryCollector(
    evaluator: Optional[RAGEvaluator] = None,
    risk_engine: Optional[FinancialRiskEngine] = None,
    stress_tester: Optional[MonteCarloStressTester] = None,
)
```

#### Methods
- **`record_evaluation(metrics: EvaluationMetrics) -> None`**: Appends an evaluation record to internal history.
- **`add_alert_listener(listener: Callable[[TelemetryAlert], None]) -> None`**: Registers an alert callback.
- **`clear_history() -> None`**: Clears historical evaluation data.
- **`get_evaluations(limit: Optional[int] = None, passed_only: Optional[bool] = None) -> List[EvaluationMetrics]`**: Queries logged evaluations.
- **`get_rolling_metrics(window_size: int = 20) -> Dict[str, float]`**: Computes rolling averages for faithfulness, relevance, hallucination rate, and latency.
- **`detect_drift(metric: str = "faithfulness_score", baseline_size: int = 20, test_size: int = 10, alpha: float = 0.05) -> Dict[str, Any]`**:
  Executes Welch's two-sample t-test to detect statistically significant metric degradation.
- **`generate_report(returns: Optional[np.ndarray] = None, max_acceptable_hallucination_rate: float = 0.10, min_acceptable_sharpe: float = 0.80) -> TelemetryReport`**:
  Produces a system health report with structured alerts.
- **`export_summary() -> Dict[str, Any]`**: Dumps summary dictionary suitable for JSON serialization.

---

## 11. CLI Commands & Options (`cli.main`)

CLI entry points: `rag-eval` or `production-rag-eval`.

### Subcommands Overview

| Command | Description |
| :--- | :--- |
| `evaluate` | Evaluates a single query-response pair against context. |
| `batch-evaluate` | Processes JSON or JSONL batches of queries and responses. |
| `audit` / `detect-hallucinations` | Audits text against context for financial hallucinations. |
| `risk-metrics` | Computes Sharpe, Sortino, Max Drawdown, and VaR on return series. |
| `stress-test` | Runs stochastic Monte Carlo simulations under market shocks. |
| `telemetry` | Generates a system health and operational telemetry report. |
| `cache stats` | Displays semantic cache capacity, hit counts, and hit rate. |
| `cache clear` | Clears all entries from the semantic evaluation cache. |
| `feature-store stats` | Displays record counts, partitions, and storage byte metrics. |
| `feature-store list-schemas` | Lists registered schemas, feature definitions, and types. |
| `feature-store compact` | Compacts Parquet files, purges duplicates, and prunes TTL. |
| `feature-store export` | Exports aligned multi-group training matrix (Parquet/CSV/Arrow). |
| `feature-store drift` | Computes PSI drift reports and detects distribution shift. |

---

## 12. Feature Store & Streaming Pipeline (`production_rag_evaluation.features`)

High-throughput streaming feature extraction, PyArrow columnar persistence, schema enforcement, compaction, and statistical drift auditing.

### 12.1 Enums & Data Models (`features.models`)

#### `FeatureValueType` (Enum)
Supported data types for engineered features:
- `FLOAT`: 64-bit floating point (`pa.float64()`).
- `INT`: 64-bit signed integer (`pa.int64()`).
- `STRING`: UTF-8 textual string (`pa.string()`).
- `BOOLEAN`: Boolean flag (`pa.bool_()`).
- `VECTOR`: Variable-length 1D float vector (`pa.list_(pa.float64())`).
- `JSON`: Serialized JSON payload string (`pa.string()`).

#### `FeatureGroup` (Enum)
Feature group namespace partitions:
- `QUERY`: Inbound query characteristics.
- `CHUNK`: Retrieved context chunk properties.
- `RESPONSE`: Generated response metrics.
- `EVALUATION`: Evaluator DAG and verification metrics.
- `FINANCIAL`: Domain financial attributes and telemetry.

#### `FeatureDefinition` (Pydantic Model)
- `name: str`: Unique identifier for the feature.
- `value_type: FeatureValueType`: Strongly-typed schema indicator.
- `description: str`: Human-readable semantic documentation.
- `nullable: bool`: Whether `None` values are permitted (default: `True`).
- `default_value: Optional[Any]`: Optional fallback value if extraction is missing.

#### `FeatureSchema` (Pydantic Model)
- `name: str`: Schema name.
- `group: FeatureGroup`: Associated feature group namespace.
- `version: str`: Semantic version string (default: `"1.0.0"`).
- `features: List[FeatureDefinition]`: Ordered feature definitions.

#### `FeatureRecord` (Pydantic Model)
- `entity_id: str`: Unique query or session identifier.
- `feature_group: FeatureGroup`: Group namespace.
- `timestamp: float`: Unix epoch timestamp.
- `values: Dict[str, Any]`: Key-value feature map conforming to the schema.

#### `OnlineFeatureVector` (Pydantic Model)
- `entity_id: str`: Associated entity identifier.
- `features: Dict[str, Any]`: Flattened feature dictionary across all groups.
- `vector: List[float]`: Normalized numerical vector suitable for low-latency scoring.
- `timestamp: float`: Generation timestamp.

#### `FeatureDriftReport` (Pydantic Model)
- `feature_name: str`: Target feature identifier.
- `feature_group: FeatureGroup`: Feature group namespace.
- `psi_score: float`: Population Stability Index score.
- `drift_detected: bool`: `True` if `psi_score >= psi_threshold`.
- `threshold: float`: Alert boundary (default: `0.25`).
- `baseline_mean: float`, `target_mean: float`: Mean shift comparison.
- `baseline_variance: float`, `target_variance: float`: Dispersion comparison.
- `num_baseline_samples: int`, `num_target_samples: int`: Sample sizes.

#### `CompactionResult` (Pydantic Model)
- `feature_group: FeatureGroup`: Compacted namespace.
- `original_records: int`: Record count before compaction.
- `retained_records: int`: Record count after deduplication and TTL pruning.
- `purged_records: int`: Number of deduplicated or expired records dropped.
- `output_file: str`: Resulting Parquet file path.
- `file_size_bytes: int`: Final storage footprint on disk.

#### `ExportFormat` (Enum)
- `PARQUET`, `CSV`, `JSON`, `ARROW`.

---

### 12.2 Columnar Storage Engine (`features.store`)

### Class: `ParquetFeatureStore`
Manages on-disk columnar Apache Parquet files organized by feature group partitions.

#### Constructor
```python
ParquetFeatureStore(
    base_path: Union[str, Path] = "./data/feature_store",
    default_compression: str = "snappy",
)
```

#### Methods
- **`write_features(features: List[FeatureRecord], compression: Optional[str] = None) -> int`**:
  Appends or overwrites feature records to the corresponding group partition. Returns count of written records.
- **`read_features(group: FeatureGroup, entity_ids: Optional[List[str]] = None, start_time: Optional[float] = None, end_time: Optional[float] = None, columns: Optional[List[str]] = None, limit: Optional[int] = None) -> List[FeatureRecord]`**:
  Queries features with optional predicate pushdown and column projection.
- **`get_entity_features(entity_id: str, groups: Optional[List[FeatureGroup]] = None) -> Dict[str, Any]`**:
  Retrieves point-in-time features for an entity ID merged across specified feature groups.
- **`get_feature_vector(entity_id: str, feature_names: List[str], group: FeatureGroup) -> np.ndarray`**:
  Fetches an ordered 1D NumPy array for a specified list of numerical features.
- **`list_entities(group: FeatureGroup, limit: Optional[int] = None) -> List[str]`**:
  Lists distinct entity IDs available in a feature group partition.
- **`storage_summary() -> Dict[str, Any]`**:
  Returns storage breakdown (file size, row count, columns, row groups) across all groups.
- **`delete_group(group: FeatureGroup) -> bool`**:
  Removes storage partition for a feature group.

---

### 12.3 Schema Registry (`features.registry`)

### Class: `FeatureRegistry`
Maintains schemas, type definitions, and Arrow serialization mappings for all feature groups.

#### Methods
- **`register_schema(schema: FeatureSchema) -> None`**: Registers a new or updated schema.
- **`get_schema(group: FeatureGroup) -> Optional[FeatureSchema]`**: Fetches schema for a group.
- **`list_schemas() -> Dict[FeatureGroup, FeatureSchema]`**: Lists all registered schemas.
- **`validate_record(record: FeatureRecord) -> SchemaValidationResult`**: Validates a `FeatureRecord` against registered schema.
- **`to_pyarrow_schema(group: FeatureGroup) -> pa.Schema`**: Generates a strongly-typed `pyarrow.Schema` incorporating metadata columns.

#### Helper
- **`get_default_registry() -> FeatureRegistry`**: Returns a pre-configured registry loaded with standard `production-rag-evaluation` schemas.

---

### 12.4 Feature Extractors (`features.extractors`)

Specialized extractors that derive quantitative and categorical signals from raw RAG artifacts.

- **`QueryFeatureExtractor`**:
  Extracts `token_length`, `financial_entity_count`, `complexity_score`, `query_type`, `sentiment_polarity`.
- **`ChunkFeatureExtractor`**:
  Extracts `token_length`, `numerical_density`, `has_financial_table`, `readability_score`.
- **`ResponseFeatureExtractor`**:
  Extracts `character_length`, `citation_density`, `hedge_word_ratio`, `numerical_claim_count`.
- **`EvaluationFeatureExtractor`**:
  Extracts `faithfulness_score`, `relevance_score`, `context_precision`, `context_recall`, `overall_score`.
- **`FinancialFeatureExtractor`**:
  Extracts `ticker`, `fiscal_period`, `monetary_amount_count`, `metric_reference_count`, `risk_level`.

All extractors inherit from `BaseFeatureExtractor` and expose:
```python
extract(*args, entity_id: str, **kwargs) -> FeatureRecord
```

---

### 12.5 Streaming Pipeline (`features.pipeline`)

### Class: `StreamingFeaturePipeline`
Coordinates streaming ingestion, in-memory buffering, and periodic Parquet flushes.

#### Constructor
```python
StreamingFeaturePipeline(
    store: Optional[ParquetFeatureStore] = None,
    registry: Optional[FeatureRegistry] = None,
    batch_size: int = 100,
    buffer_max_size: int = 10000,
)
```

#### Methods
- **`ingest(records: List[FeatureRecord]) -> int`**: Validates and buffers records, automatically triggering `flush()` when `batch_size` is exceeded.
- **`ingest_query_response(entity_id: str, query: str, response: str, chunk: Optional[str] = None, metrics: Optional[EvaluationMetrics] = None, ticker: Optional[str] = None, period: Optional[str] = None) -> OnlineFeatureVector`**:
  Extracts features across all active domains, updates buffers, and returns an `OnlineFeatureVector`.
- **`flush() -> int`**: Writes all buffered records to the underlying `ParquetFeatureStore`.
- **`get_online_vector(entity_id: str) -> Optional[OnlineFeatureVector]`**: Fetches latest in-memory feature vector for immediate serving.
- **`get_window_stats(group: FeatureGroup) -> SlidingWindowStats`**: Returns streaming statistical distributions for a feature group.

---

### 12.6 Feature Drift Detector (`features.drift`)

### Class: `FeatureDriftDetector`
Computes Population Stability Index (PSI) and monitors statistical drift across Parquet partitions.

#### Methods
- **`calculate_psi(baseline: np.ndarray, target: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4) -> float`** *(static)*:
  Computes PSI between baseline and target numerical samples.
- **`detect_feature_drift(feature_name: str, group: FeatureGroup, current_records: Optional[List[FeatureRecord]] = None, baseline_entity_ids: Optional[List[str]] = None, target_entity_ids: Optional[List[str]] = None, psi_threshold: float = 0.25) -> FeatureDriftReport`**:
  Executes drift detection on an individual named feature.
- **`detect_all_drift(group: Optional[FeatureGroup] = None, current_records: Optional[List[FeatureRecord]] = None, psi_threshold: float = 0.25) -> List[FeatureDriftReport]`**:
  Scans all numeric features in a group (or all groups) and returns drift reports.
- **`generate_telemetry_alerts(collector: Optional[RiskTelemetryCollector] = None, reports: Optional[List[FeatureDriftReport]] = None, psi_threshold: float = 0.25) -> List[TelemetryAlert]`**:
  Converts drift reports exceeding threshold into `TelemetryAlert` notifications and registers them with the collector.

---

### 12.7 Integration Handlers (`features.handlers`)

- **`EvaluationFeatureHandler(pipeline: Optional[StreamingFeaturePipeline] = None)`**:
  Connects `RAGEvaluator` results to feature store extraction automatically.
- **`FeatureCompactionHandler(store: ParquetFeatureStore)`**:
  Performs file consolidation, deduplication by `entity_id`, and TTL eviction.
  - `compact_group(group: FeatureGroup, ttl_seconds: Optional[float] = None, compression: str = "snappy") -> CompactionResult`
  - `compact_all(ttl_seconds: Optional[float] = None, compression: str = "snappy") -> Dict[FeatureGroup, CompactionResult]`
- **`FeatureExportHandler(store: ParquetFeatureStore)`**:
  Generates point-in-time aligned offline datasets across groups for entity lists.
  - `export_dataset(entity_ids: List[str], output_path: Union[str, Path], format: ExportFormat = ExportFormat.PARQUET, groups: Optional[List[FeatureGroup]] = None) -> Path`
