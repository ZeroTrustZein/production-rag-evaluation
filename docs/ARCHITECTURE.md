# System Architecture & Quantitative Methodologies

Comprehensive architectural documentation, mathematical formulations, and engineering designs for **`production-rag-evaluation`**.

---

## 1. High-Level System Architecture

`production-rag-evaluation` is an institutional-grade evaluation, hallucination auditing, and quantitative risk platform built specifically for financial Retrieval-Augmented Generation (RAG) pipelines operating on SEC filings (10-K, 10-Q, 8-K), earnings transcripts, and equity research reports.

The system decouples fast-path semantic caching, topological DAG state machine evaluation, specialized domain hallucination detection, quantitative risk modeling, stochastic Monte Carlo stress testing, and real-time statistical drift telemetry into dedicated, testable subsystems.

```
+-----------------------------------------------------------------------------------+
|                        Inbound RAG Query & Generated Response                     |
|                   (RAGQuery, RAGResponse, Retrieved Context Chunks)               |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     In-Memory Semantic Cache (SemanticCache)                      |
|  - Fast Hash Key Lookups (Exact SHA-256)                                          |
|  - N-Gram Hash Vectorization (TextEmbedder D=128)                                 |
|  - Cosine Similarity Matching (Threshold >= 0.90)                                 |
|  - LRU Eviction & Automatic TTL Expiration                                        |
+-------------------+-------------------------------------------+-------------------+
                    | [Cache Hit]                               | [Cache Miss]
                    v                                           v
+-----------------------------------+   +-------------------------------------------+
| Return Cached EvaluationMetrics   |   |   Evaluation Workflow DAG (EvaluationGraph) |
| (Latency Saved ~25ms per lookup)  |   |   - NetworkX Topological State Machine    |
+-----------------------------------+   +---------------------+---------------------+
                                                              |
                                                              v
+-----------------------------------------------------------------------------------+
|                         DAG Execution Pipeline (Sequential)                       |
|                                                                                   |
|  [extract_claims]      --> Segment generated answer into atomic factual sentences |
|          |                                                                        |
|          v                                                                        |
|  [verify_citations]    --> Map quotes to retrieved chunk text & verify presence   |
|          |                                                                        |
|          v                                                                        |
|  [detect_hallucinations] -> Run HallucinationDetector (numbers, polarity, entities)|
|          |                                                                        |
|          v                                                                        |
|  [verify_claims]       --> Classify claim status (VERIFIED/REFUTED/UNGROUNDED)    |
|          |                                                                        |
|          v                                                                        |
|  [verify_targets]      --> Validate structured NumericalTarget & raw KPIs         |
|          |                                                                        |
|          v                                                                        |
|  [compute_relevance]   --> Lexical & context recall/precision scoring             |
|          |                                                                        |
|          v                                                                        |
|  [aggregate_metrics]   --> Compute overall_pass, hallucination_rate, latency ms   |
+-------------------------------------------------------------+---------------------+
                                                              |
                                                              v
+-----------------------------------------------------------------------------------+
|                      Evaluation Outcome & Financial Risk Engine                   |
|  - EvaluationMetrics (Faithfulness, Relevance, Precision, Recall, Grounding)      |
|  - HallucinationFinding Diagnoses with Severity and Risk Impact                   |
+---------------------------------------------+-------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------+
|                     Downstream Financial & Operational Analytics                  |
|                                                                                   |
|  +---------------------------+  +--------------------------+  +------------------+|
|  | FinancialRiskEngine       |  | MonteCarloStressTester   |  | RiskTelemetry    ||
|  | - Sharpe, Sortino, Calmar |  | - GBM Asset Paths        |  | - Rolling Windows||
|  | - Max Drawdown Duration   |  | - Fat-Tailed Student-t   |  | - Welch's t-Test ||
|  | - Hist & Parametric VaR   |  | - Macro Stress Presets   |  | - Drift Detection||
|  | - Cornish-Fisher CVaR     |  | - Hallucination Cascades |  | - Multi-tier Alert||
|  | - Alpha, Beta, IR, Omega  |  | - Portfolio Breaches     |  | - System Health  ||
|  +---------------------------+  +--------------------------+  +------------------+|
+-----------------------------------------------------------------------------------+
```

---

## 2. In-Memory Semantic Caching Subsystem

Evaluating dense financial texts using graph workflows or neural evaluators introduces non-trivial latency and compute overhead. The semantic cache stores evaluated responses and returns pre-computed metrics for semantically identical or near-identical queries.

### 2.1 Deterministic N-Gram Hash Vectorization (`TextEmbedder`)

Rather than relying on heavy external embedding API calls or unquantized PyTorch models, `TextEmbedder` implements a deterministic, CPU-bound n-gram feature hashing vectorizer:

1. **Token Processing**: Input text is converted to lowercase, stripped, and tokenized by whitespace.
2. **Unigram Hashing**: For each word $w_i$, MD5 hash integer projection is computed into index space $[0, D-1]$ where $D=128$:
   $$j_1 = \text{MD5}(w_i) \pmod D, \quad \mathbf{v}[j_1] \leftarrow \mathbf{v}[j_1] + 1.0$$
3. **Bigram Hashing**: For consecutive token pairs $(w_i, w_{i+1})$, SHA-256 hash projection is computed with enhanced weighting:
   $$j_2 = \text{SHA256}(w_i \mathbin{\Vert} \text{"\_"} \mathbin{\Vert} w_{i+1}) \pmod D, \quad \mathbf{v}[j_2] \leftarrow \mathbf{v}[j_2] + 1.5$$
4. **L2 Unit Normalization**:
   $$\hat{\mathbf{v}} = \begin{cases} \frac{\mathbf{v}}{\|\mathbf{v}\|_2} & \text{if } \|\mathbf{v}\|_2 > 0 \\ \mathbf{0} & \text{otherwise} \end{cases}$$

### 2.2 Dual-Tier Lookup Strategy

- **Tier 1 (Exact Match)**: The query string is normalized and hashed via SHA-256 (`hashlib.sha256(q.strip().lower())`). Direct $O(1)$ dictionary lookup executes first.
- **Tier 2 (Cosine Similarity Search)**: If exact match misses, cosine similarity is evaluated against cached vector embeddings:
  $$S_c(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$$
  If $\max_k S_c(\hat{\mathbf{u}}, \hat{\mathbf{v}}_k) \ge \tau_{\text{sim}}$ (default $\tau_{\text{sim}} = 0.90$), the entry is deemed a hit.

### 2.3 Eviction & Expiration

- **Capacity Management**: Built on `collections.OrderedDict`. When entries exceed `max_entries`, least recently accessed entries are evicted ($O(1)$ LRU).
- **TTL Window**: Entries carry `expires_at = created_at + ttl_seconds` (default 24 hours). Expired items are purged lazily during read/write cycles.

---

## 3. Evaluation Graph DAG State Machine

The core evaluation is orchestrated as a Directed Acyclic Graph (DAG) state machine using `networkx.DiGraph`. State transitions occur via an `EvaluationState` context object passed sequentially through topological stages.

### 3.1 Graph Topology

```
[extract_claims]
       |
       v
[verify_citations]
       |
       v
[detect_hallucinations]
       |
       v
[verify_claims]
       |
       v
[verify_targets]
       |
       v
[compute_relevance]
       |
       v
[aggregate_metrics]
```

### 3.2 Evaluation Modes

The state machine adapts its thresholds and verification stringency based on `EvaluationMode`:

| Evaluation Mode | Target Verification | Tolerance Check | Critical Threshold | Intended Workload |
| :--- | :--- | :--- | :--- | :--- |
| **`FAST`** | Approximate | $\pm 5\%$ | Faithfulness $\ge 0.65$ | Real-time streaming chat, dev testing |
| **`THOROUGH`** | Balanced | $\pm 1\%$ | Faithfulness $\ge 0.75$ | Batch regression, regression gates |
| **`FINANCIAL_STRICT`** | Exact matching | $\pm 0.1\%$ | Faithfulness $\ge 0.85$, Zero Criticals | SEC 10-K extraction, financial advice |
| **`AUDIT`** | Exhaustive trace | $\pm 0.01\%$ | Faithfulness $\ge 0.90$, Zero Tolerated | Regulatory reporting, compliance audit |

### 3.3 Node Responsibilities

1. **`extract_claims`**: Segments the generated answer into discrete, verifiable assertions using sentence delimiters (`.`, `!`, `?`). Short or trivial phrases are filtered out.
2. **`verify_citations`**: Verifies whether cited chunk quotes exist verbatim or normalized inside the retrieved document chunks. Calculates `citation_grounding_score`:
   $$\text{Grounding} = \frac{|\text{Verified Citations}|}{\max(|\text{Provided Citations}|, 1)}$$
3. **`detect_hallucinations`**: Invokes the rule-based `HallucinationDetector` to identify numerical distortions, directional contradictions, ungrounded extrapolations, entity swaps, and temporal mismatches.
4. **`verify_claims`**: Assigns a `ClaimStatus` (`VERIFIED`, `REFUTED`, `UNGROUNDED`, or `AMBIGUOUS`) to each claim candidate.
5. **`verify_targets`**: Validates whether structured `NumericalTarget` constraints (e.g. `net_sales=391.0e9`) or raw float expectations are present in the answer within the specified `tolerance_pct`.
6. **`compute_relevance`**: Measures question-answer token overlap excluding domain stop words, computing `answer_relevance_score`, `context_precision_score`, and `context_recall_score`.
7. **`aggregate_metrics`**: Synthesizes the overall status (`overall_pass`), `hallucination_rate`, and records end-to-end execution latency in milliseconds.

---

## 4. Specialized Financial Hallucination Engine

Financial texts are uniquely susceptible to catastrophic errors that standard NLP metrics (BLEU, ROUGE) miss: a single flipped sign, an erroneous currency unit ($M$ vs $B$), or an incorrect fiscal year can distort an investment thesis.

### 4.1 Numerical Distortion Detection

Extracts values matching regex:
`(?:\$|€|£)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|billion|million|bn|m|k|trillion)?`

- Values are normalized using financial scale multipliers:
  - `k` $\to 10^3$
  - `m` / `million` $\to 10^6$
  - `bn` / `billion` $\to 10^9$
  - `trillion` $\to 10^{12}$
- Relative error between generated figure $g$ and retrieved context figure $c$:
  $$\delta_{\text{rel}} = \frac{|g - c|}{\max(|c|, 10^{-6})}$$
  If $\min_{c \in C} \delta_{\text{rel}} > \tau_{\text{tol}}$ (where $\tau_{\text{tol}} = 0.01$), a `NUMERICAL_DISTORTION` finding of severity `HIGH` is emitted.

### 4.2 Directional Contradiction Detection

Detects polarity inversion between retrieved disclosures and generated statements using curated financial sentiment vocabularies:

- **Positive Directional Set** ($P_{\text{pos}}$): `increased`, `grew`, `growth`, `rose`, `expanded`, `up`, `surpassed`, `outperformed`, `higher`.
- **Negative Directional Set** ($P_{\text{neg}}$): `decreased`, `fell`, `dropped`, `declined`, `contracted`, `down`, `missed`, `lower`, `slumped`.

If context manifests $P_{\text{neg}}$ while generated text exclusively asserts $P_{\text{pos}}$ (or vice versa), a `CONTRADICTION` of severity `CRITICAL` is flagged.

### 4.3 Entity Swap Detection

Captures ticker substitutions (e.g., attributing Microsoft Azure's cloud growth to Apple).
- Extracts uppercase candidate tokens matching `\b[A-Z]{2,5}\b`.
- Filters out non-ticker financial acronyms (`USD`, `EUR`, `GAAP`, `EBITDA`, `EPS`, `SEC`, `CEO`, `CAGR`, `10-K`, `ROIC`, etc.).
- Identifies tickers appearing in generated responses that are completely absent from the source chunk metadata and body text.

### 4.4 Temporal Mismatch Detection

Captures fiscal year hallucinations (e.g. citing FY2024 guidance from FY2022 filings).
- Extracts 4-digit Gregorian years matching `\b(19\d\d|20\d\d)\b`.
- Cross-references years asserted in claims against years present in retrieved contexts. Discrepancies generate a `TEMPORAL_MISMATCH` finding.

---

## 5. Quantitative Financial Risk Methodologies

When RAG evaluations monitor automated trading agents or portfolio reporting systems, evaluation scores feed directly into institutional risk metrics.

### 5.1 Sharpe Ratio

Measures excess return per unit of total risk (standard deviation):

$$\text{Sharpe} = \frac{\bar{R} - R_{f,\text{period}}}{\sigma_R} \times \sqrt{N}$$

Where $N$ is the number of trading periods per year ($N=252$ for daily, $N=52$ for weekly, $N=12$ for monthly), $\bar{R}$ is sample mean return, and $R_{f,\text{period}} = (1 + R_f)^{1/N} - 1$.

### 5.2 Sortino Ratio

Penalizes only harmful downside volatility below a target return $\tau$ (default $\tau = 0.0$):

$$\text{Sortino} = \frac{\bar{R} - R_{f,\text{period}}}{\sigma_{\text{downside}}} \times \sqrt{N}$$

$$\sigma_{\text{downside}} = \sqrt{\frac{1}{K} \sum_{t=1}^{K} \min(0, R_t - \tau)^2}$$

### 5.3 Maximum Drawdown (MDD) & Calmar Ratio

Given compounding wealth trajectory $W_t = \prod_{i=1}^t (1 + R_i)$ with running high-water mark $M_t = \max_{0 \le s \le t} W_s$:

$$\text{Drawdown}_t = \frac{M_t - W_t}{M_t}, \quad \text{MDD} = \max_{t} \text{Drawdown}_t$$

$$\text{Calmar} = \frac{(1 + \bar{R})^N - 1}{\text{MDD}}$$

### 5.4 Value-at-Risk (VaR) & Conditional VaR (Expected Shortfall)

- **Historical VaR**: Empirical quantile of loss distribution at confidence level $\alpha$ (e.g., $\alpha = 0.95$):
  $$\text{VaR}_\alpha = -Q_{1-\alpha}(R)$$
- **Conditional VaR (CVaR / Expected Shortfall)**: Mean expected loss conditional on exceeding VaR threshold:
  $$\text{CVaR}_\alpha = -\mathbb{E}[R \mid R \le -\text{VaR}_\alpha]$$

### 5.5 Parametric VaR with Cornish-Fisher Expansion

Accounts for asymmetry (skewness $S$) and fat tails (excess kurtosis $K$) beyond Gaussian normality:

$$z_{\text{CF}} = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$

$$\text{VaR}_{\text{CF}} = -(\mu + z_{\text{CF}} \cdot \sigma)$$

### 5.6 Capital Asset Pricing Model (CAPM) Alpha & Beta

Evaluates systemic sensitivity and idiosyncratic outperformance relative to a benchmark index (e.g., S&P 500 / SPY):

$$\beta = \frac{\text{Cov}(R_p, R_b)}{\text{Var}(R_b)}, \quad \alpha = (\bar{R}_p - R_{f,\text{period}}) - \beta(\bar{R}_b - R_{f,\text{period}})$$

### 5.7 Information Ratio & Tracking Error

Measures active return per unit of tracking risk:

$$\text{TE} = \sqrt{\frac{1}{T-1} \sum_{t=1}^T \left((R_{p,t} - R_{b,t}) - \overline{R_p - R_b}\right)^2}$$

$$\text{IR} = \frac{\overline{R_p - R_b}}{\text{TE}} \times \sqrt{N}$$

### 5.8 Omega Ratio

Ratio of cumulative probability-weighted gains above target threshold $L$ to cumulative losses below $L$:

$$\Omega(L) = \frac{\int_L^\infty (1 - F(r)) dr}{\int_{-\infty}^L F(r) dr}$$

---

## 6. Stochastic Monte Carlo Stress Engine

Simulates extreme asset trajectories to test whether trading algorithms relying on RAG inputs breach risk limits under adverse conditions.

### 6.1 Geometric Brownian Motion (GBM) Paths

Standard log-normal diffusion with drift $\mu$, volatility $\sigma$, and time increment $\Delta t = \frac{1}{252}$:

$$S_{t+\Delta t} = S_t \exp\left(\left((\mu + \Delta\mu) - \frac{1}{2} (\sigma \cdot M_\sigma)^2\right)\Delta t + (\sigma \cdot M_\sigma) \sqrt{\Delta t} Z_t\right), \quad Z_t \sim \mathcal{N}(0, 1)$$

### 6.2 Fat-Tailed Student-t Innovations

To model black swan kurtosis and sudden liquidity breaks, innovations $Z_t$ can be sampled from a standardized Student-t distribution with $\nu$ degrees of freedom ($\text{Var} = \frac{\nu}{\nu - 2}$):

$$\tilde{Z}_t \sim t(\nu), \quad Z_t = \tilde{Z}_t \sqrt{\frac{\nu - 2}{\nu}}$$

### 6.3 Preset Crisis Scenarios

| Scenario | Volatility Shock ($M_\sigma$) | Drift Shock ($\Delta\mu$) | Max Allowable DD | Economic Analog |
| :--- | :--- | :--- | :--- | :--- |
| **`BASELINE`** | $1.0\times$ | $0.0\%$ | $25\%$ | Normal market regime |
| **`VOLATILITY_SPIKE`** | $2.5\times$ | $-5.0\%$ | $30\%$ | VIX surge, earnings revision shock |
| **`LIQUIDITY_CRUNCH`** | $3.0\times$ | $-15.0\%$ | $20\%$ | Credit spread blowout, repo freeze |
| **`RATE_SHOCK`** | $1.8\times$ | $-10.0\%$ | $25\%$ | Emergency central bank rate hike |
| **`BLACK_SWAN`** | $3.5\times$ | $-30.0\%$ | $35\%$ | 2008 Lehman collapse / March 2020 crash |
| **`HALLUCINATION_CASCADE`** | $2.2\times$ | $-20.0\%$ | $25\%$ | Algorithmic misguidance cascade |

### 6.4 Hallucination Sensitivity Simulation

Evaluates how frequently hallucinations corrupting financial figures induce severe forecasting errors:
- Iterates over $M$ simulations ($N=500$).
- Corrupts baseline figures with probability $p_{\text{hallucination}}$ by distortion factor $\delta_{\text{pct}}$.
- Calculates breach rate where portfolio forecasting error exceeds $10\%$.

---

## 7. Real-Time Telemetry & Drift Detection

Monitors production RAG systems for silent drift caused by model updates, context window degradation, or shifting prompt templates.

### 7.1 Statistical Drift Testing (Welch's Two-Sample t-Test)

Compares a baseline evaluation window ($n_1 \ge 20$) against recent production batches ($n_2 \ge 10$):

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

Degrees of freedom computed via the Welch-Satterthwaite equation:

$$\nu \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}$$

The $p$-value is derived using `scipy.stats.t.sf(abs(t), df) * 2`. When $p < \alpha$ (default $\alpha = 0.05$) and $\bar{X}_2 < \bar{X}_1$, statistically significant quality decay is confirmed.

### 7.2 System Health State Machine

Operational state transitions based on telemetry conditions:

```
                  [No Active Alerts]
                +-------------------+
                |      HEALTHY      |
                +----+---------+----+
                     |         ^
       [Warning Alert]         | [Alerts Cleared]
                     v         |
                +----+---------+----+
                |      DEGRADED     |
                +----+---------+----+
                     |         ^
      [Critical Alert]         | [Critical Resolved]
                     v         |
                +----+---------+----+
                |     UNHEALTHY     |
                +-------------------+
```

- **`HEALTHY`**: Zero active alerts; hallucination rate below threshold, risk metrics within bounds.
- **`DEGRADED`**: Warning-level alerts active (e.g., mild drift, elevated latency).
- **`UNHEALTHY`**: Critical-level alerts active (e.g., hallucination rate spike, stress test failure).

---

## 8. Streaming Feature Pipeline & Columnar Parquet Feature Store Architecture

The feature store subsystem decouples real-time inference telemetry from downstream model retraining, drift auditing, and analytical reporting. It provides a dual-path architecture: a low-latency streaming pipeline for online feature vectors and a columnar Apache Parquet storage engine for point-in-time analytical datasets.

```
+--------------------------------------------------------------------------------------------------+
|                                  INBOUND RAG INFERENCE PIPELINE                                  |
|            [Query Text]           [Retrieved Chunks]           [Generated Response]              |
+------------------+------------------------+-----------------------------+------------------------+
                   |                        |                             |
                   v                        v                             v
+------------------+------------------------+-----------------------------+------------------------+
|                                   SPECIALIZED EXTRACTORS                                         |
|  - QueryFeatureExtractor       - ChunkFeatureExtractor        - ResponseFeatureExtractor         |
|  - FinancialFeatureExtractor   - EvaluationFeatureExtractor                                      |
+-------------------------------------------+------------------------------------------------------+
                                            |
                                            v (FeatureRecord Instances)
+-------------------------------------------+------------------------------------------------------+
|                         STREAMING FEATURE PIPELINE (Real-Time)                                   |
|  - Schema Validation via FeatureRegistry                                                         |
|  - In-Memory Sliding Window Buffering (maxlen=10,000)                                            |
|  - Real-Time Feature Vector Assembly (OnlineFeatureVector)                                       |
+---------------------+----------------------------------------------------+-----------------------+
                      |                                                    |
                      v (Low-Latency Serving)                              v (Batched Flush)
           +-----------------------+                         +---------------------------+
           | Online Feature Vector |                         | ParquetFeatureStore       |
           | O(1) Cache / Lookup   |                         | - Snappy/ZSTD Compression |
           +-----------------------+                         | - Columnar Partitions     |
                                                             | - Predicate Pushdown      |
                                                             +-------------+-------------+
                                                                           |
                                            +------------------------------+------------------------------+
                                            |                                                             |
                                            v                                                             v
                             +------------------------------+                              +------------------------------+
                             |   FEATURE DRIFT DETECTOR     |                              |   FEATURE LIFECYCLE MGMT     |
                             | - Population Stability Index |                              | - Parquet Compaction         |
                             | - Kolmogorov-Smirnov Shift   |                              | - TTL Expiration Pruning     |
                             | - Telemetry Alerts (P2/P1)   |                              | - Point-in-Time Join Export  |
                             +------------------------------+                              +------------------------------+
```

### 8.1 Dual-Path Online/Offline Architecture

1. **Online Path (Low Latency)**:
   - Evaluated queries and contexts immediately yield structured `FeatureRecord` payloads.
   - Stored in an in-memory sliding circular buffer (`collections.deque`, configurable max size).
   - Generates unified `OnlineFeatureVector` representations accessible in $<1\text{ ms}$ for guardrail checks or real-time filtering.

2. **Offline Analytical Path (Columnar Parquet)**:
   - High-throughput streaming batches flush to columnar Apache Parquet files organized by feature group.
   - Column projection allows reading individual metrics (e.g. `feat_faithfulness_score`) across millions of records without scanning full rows.
   - Predicate pushdown filtering by `entity_id` and timestamp ranges minimizes disk I/O.

### 8.2 Feature Groups & Schema Definitions

The feature registry (`FeatureRegistry`) manages strict, strongly-typed schemas across five specialized feature groups:

| Feature Group | Target Domain | Core Engineered Features |
| :--- | :--- | :--- |
| **`QUERY`** | Inbound User Prompts | `token_length`, `financial_entity_count`, `complexity_score`, `query_type`, `sentiment_polarity` |
| **`CHUNK`** | Retrieved Context | `token_length`, `numerical_density`, `has_financial_table`, `readability_score` |
| **`RESPONSE`** | LLM Output | `character_length`, `citation_density`, `hedge_word_ratio`, `numerical_claim_count` |
| **`EVALUATION`** | Multi-Factor DAG | `faithfulness_score`, `relevance_score`, `context_precision`, `context_recall`, `overall_score` |
| **`FINANCIAL`** | Financial Telemetry | `ticker`, `fiscal_period`, `monetary_amount_count`, `metric_reference_count`, `risk_level` |

### 8.3 Columnar Storage Layout & PyArrow Mapping

Each partition persists to a strongly-typed Parquet file where schema definitions are dynamically translated to PyArrow types:

- `FeatureValueType.FLOAT` $\rightarrow$ `pa.float64()`
- `FeatureValueType.INT` $\rightarrow$ `pa.int64()`
- `FeatureValueType.STRING` $\rightarrow$ `pa.string()`
- `FeatureValueType.BOOLEAN` $\rightarrow$ `pa.bool_()`
- `FeatureValueType.VECTOR` $\rightarrow$ `pa.list_(pa.float64())`
- `FeatureValueType.JSON` $\rightarrow$ `pa.string()`

Metadata columns (`entity_id: string`, `timestamp: float64`, `feature_group: string`) are prepended to ensure reproducible historical replay and point-in-time joins.

### 8.4 Statistical Feature Drift Detection via Population Stability Index (PSI)

To monitor continuous degradation of input embeddings, query patterns, and LLM behavior, the `FeatureDriftDetector` computes the **Population Stability Index (PSI)** between baseline reference distributions and active production windows:

$$PSI = \sum_{b=1}^{B} \left( P_b - Q_b \right) \times \ln\left( \frac{P_b + \epsilon}{Q_b + \epsilon} \right)$$

Where:
- $B$: Number of empirical quantiles / bins (default $B = 10$).
- $Q_b$: Proportion of observations from baseline reference distribution falling into bin $b$.
- $P_b$: Proportion of observations from target production distribution falling into bin $b$.
- $\epsilon = 10^{-4}$: Regularization constant preventing division by zero or undefined logarithms.

#### Drift Decision Boundaries
- **$PSI < 0.10$**: **Negligible Shift** — Feature distribution remains stable; normal operation.
- **$0.10 \le PSI < 0.25$**: **Moderate Shift** — Detectable feature divergence; raises `AlertSeverity.WARNING` telemetry alert.
- **$PSI \ge 0.25$**: **Significant Shift** — Substantial structural divergence; raises `AlertSeverity.CRITICAL` alert and flags upstream retrieval or model drift.

In addition to PSI, the detector tracks continuous summary delta metrics:
$$\Delta\mu = \frac{\bar{X}_{\text{target}} - \bar{X}_{\text{baseline}}}{\sigma_{\text{baseline}} + \epsilon}, \quad \Delta\sigma = \frac{s_{\text{target}} - s_{\text{baseline}}}{s_{\text{baseline}} + \epsilon}$$

### 8.5 Compaction, Deduplication & Lifecycle Retention

1. **Storage Compaction**:
   - Repeated streaming appends produce fragmented Parquet files or bloated row groups.
   - `FeatureCompactionHandler` consolidates partitioned tables into contiguous, optimally-sized row groups using configurable compression (`snappy`, `zstd`, `gzip`).
2. **Entity Deduplication**:
   - Groups records by `entity_id` and retains only the most recent timestamp record, eliminating replay duplicates.
3. **TTL Retention Pruning**:
   - Automatically purges historical records older than a configured Time-To-Live threshold ($t_{\text{now}} - t_{\text{record}} > \text{TTL}$), maintaining compliant data governance.
4. **Point-in-Time Multi-Group Join**:
   - `FeatureExportHandler` performs multi-group joins aligned by `entity_id` to produce offline training datasets in Parquet, CSV, JSON, or Arrow formats.
