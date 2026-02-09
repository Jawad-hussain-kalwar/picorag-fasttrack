# PicoRAG — Experimental Design

**Version:** 1.0
**Last Updated:** 2026-02-01

---

## 1. Project Overview

### 1.1 What is PicoRAG?

**PicoRAG** is a lightweight, fully-local Retrieval-Augmented Generation (RAG) framework designed to operate efficiently on resource-constrained personal computers without reliance on cloud infrastructure or external APIs. The system combines Small Language Models (SLMs) with optimized retrieval mechanisms to enable knowledge-grounded text generation on mid-tier hardware.

### 1.2 Purpose and Motivation

Current state-of-the-art RAG systems typically require substantial computational resources, cloud-based infrastructure, and large language models (e.g., LLaMA 3.1 405B, Claude 3.5 Sonnet). This dependency creates significant barriers for small businesses, educational institutions, researchers, and individual users who lack access to expensive hardware or subscription-based API services.

PicoRAG addresses this gap by demonstrating that **effective RAG can be achieved locally on consumer-grade hardware** (≤16GB RAM, mid-tier CPU, ≤4GB VRAM), making advanced AI-powered knowledge retrieval accessible to a broader audience for use cases including:
- Personal knowledge management
- Educational support and research
- Small business operations
- Privacy-sensitive applications requiring offline operation
- Rapid prototyping and experimentation

### 1.3 Scope and Constraints

**Hardware Target:**
- Mid-tier laptop (7th-gen Intel i7 or equivalent, 16GB RAM)
- Single consumer GPU (GTX 1050 Ti 4GB) or CPU-only fallback
- No specialized accelerators or server infrastructure

**System Constraints:**
- Fully local, offline operation (except for explicit online baselines)
- No Docker/containers required
- No external vector database services
- Pip-installable components only
- Text-based RAG (no multimodal or speech systems)

**Primary Focus:**
- General knowledge retrieval and question-answering tasks
- Optimization of existing components (not developing new language models)
- Empirical quantification of performance-resource trade-offs

---

## 2. Research Context

### 2.1 Research Aim

**To develop and evaluate a lightweight Retrieval-Augmented Generation (RAG) framework that operates efficiently on mid-tier personal computers, making RAG technology accessible and practical for resource-constrained environments.**

### 2.2 Research Objectives

**Objective A:** Design and implement a lightweight RAG framework using a locally-stored Small Language Model (SLM) and optimized retrieval mechanisms.

**Objective B:** Explore and compare efficient indexing and retrieval algorithms suitable for operation on mid-tier hardware with limited computational resources.

**Objective C:** Evaluate the framework's effectiveness in terms of response quality, retrieval accuracy, and computational efficiency compared to cloud-based RAG systems.

### 2.3 Research Questions

**RQ1: Architectural Integration and Resource Efficiency**
*What architectural design choices and integration strategies between Small Language Models and retrieval mechanisms yield optimal performance for a locally-deployed RAG framework while maintaining minimal resource consumption?*

- **Measurable aspects:** Framework latency (seconds), memory footprint (GB), throughput (queries/second), system resource utilization (CPU/RAM/VRAM %), integration complexity
- **Focus areas:** SLM-retriever integration patterns, decision-layer integration (selective answering, agentic control), prompt engineering strategies, multi-hop reasoning on-device

**RQ2: Algorithm Combinations and Efficiency-Accuracy Trade-offs**
*How do various combinations of embedding models, indexing structures, and retrieval algorithms affect the efficiency-accuracy trade-off when operating under mid-tier hardware constraints?*

- **Measurable aspects:** Index size (MB), query latency (ms), memory usage (MB), retrieval accuracy (Recall@k, nDCG@k, MRR), generation quality (EM), grounding quality (Faithfulness, Citation metrics)
- **Focus areas:** Embedding model selection, vector search algorithms, hybrid retrieval (BM25 + vector fusion), reranking strategies, retrieval depth (k)

**RQ3: Local vs Cloud Viability Threshold**
*What are the quantifiable performance differences between cloud-based and lightweight local RAG systems across standardized benchmarks, and at what quality threshold does local deployment become practically viable?*

- **Measurable aspects:** Response quality (EM, ROUGE scores), retrieval accuracy (Recall@k, nDCG, MRR), resource efficiency (energy per correct answer, tokens/watt), cost-effectiveness (queries/dollar vs queries/watt), latency profiles (TTFT, p95)
- **Focus areas:** Local SLM vs online LLM generation quality, retrieval parity, total cost of ownership, acceptable quality degradation for local deployment

---

## 3. Experimental Design Overview

### 3.1 Experiment-to-Research Question Mapping

The experimental sequence is designed as a **progressive build-up** where each experiment extends the capabilities tested in prior experiments:

| Experiment | Title | Primary RQs | Objectives | Key Insight |
|------------|-------|-------------|------------|-------------|
| **E1** | Vanilla Local RAG Baseline | RQ1, RQ2 | A, B | Establishes baseline local RAG performance with vector-only retrieval |
| **E2** | Hybrid Retrieval Exploration | RQ2, RQ1 | B, A | Quantifies gains from BM25+Vector hybrid retrieval and embedding choices → identifies "Local-Best" config |
| **E3** | Selective Answering (Decision-Aware RAG) | RQ1, RQ2 | A, C | Tests impact of abstention mechanisms on answer reliability |
| **E4** | Local vs Online Comparison | RQ3 | C | Establishes viability threshold by comparing Local-Best against cloud LLM baseline |
| **E5** | Agentic Multi-Hop RAG | RQ1, RQ2 | A, B | Evaluates advanced agentic control (query reformulation, multi-hop retrieval) on constrained hardware |

### 3.2 How Experiments Fulfill Objectives

**Fulfilling Objective A** (Design and implement lightweight RAG framework):
- **E1** provides the foundational implementation (document processing, embedding, vector retrieval, SLM generation)
- **E3** extends with decision-layer integration (selective answering gate)
- **E5** implements advanced agentic control flow (multi-hop reasoning, query reformulation)

**Fulfilling Objective B** (Explore efficient algorithms):
- **E1** establishes vector-only retrieval baseline with k parameter sweep
- **E2** systematically compares embedding models, BM25 vs Vector vs Hybrid retrieval, optional reranking
- **E5** explores retrieval strategies in agentic context (tool calling, query reformulation)

**Fulfilling Objective C** (Evaluate effectiveness vs cloud systems):
- **E1-E3** establish local system performance envelope
- **E4** directly compares Local-Best configuration against online cloud LLM (Gemini 2.5 Flash)
- All experiments measure quality, accuracy, and efficiency metrics to quantify trade-offs

### 3.3 How Experiments Answer Research Questions

**Answering RQ1** (Architectural integration and resource efficiency):
- **E1** tests basic SLM-retriever integration and measures baseline resource consumption
- **E3** evaluates decision-layer integration (sufficiency gates) and overhead
- **E5** assesses complex agentic controller integration (multi-hop, tool calling) on-device

**Answering RQ2** (Algorithm combinations and trade-offs):
- **E1** provides baseline with single embedding model (all-MiniLM-L6-v2) and vector-only retrieval
- **E2** systematically varies embeddings, retrieval methods (Vector/BM25/Hybrid), reranking to identify optimal combinations
- **E5** tests hybrid retrieval + reformulation in agentic setting

**Answering RQ3** (Local vs cloud viability):
- **E4** directly compares Local-Best (from E2) against cloud baseline (Gemini 2.5 Flash)
- Uses both Mixed (RAG setting) and Oracle (generation ceiling) to isolate retrieval vs generation contributions
- Quantifies quality gap, efficiency gains, and establishes practical viability threshold

---

## 4. Datasets and Evaluation Framework

### 4.1 MIRAGE Dataset

**MIRAGE** (Metric-Intensive Benchmark for Retrieval-Augmented Generation Evaluation) is a specialized RAG evaluation dataset consisting of:
- **7,560 question-answer pairs** across 5 source QA datasets (IfQA, NaturalQA, TriviaQA, DROP, PopQA)
- **37,800 document chunks** forming a compact retrieval pool (~1% of full Wikipedia)
- **Gold answer annotations** (short textual strings) for exact-match and F1 scoring
- **Supporting context annotations** identifying which chunks contain answer-critical information

### 4.2 Evaluation Settings: Mixed vs Oracle

MIRAGE defines three evaluation configurations to isolate different failure modes:

**1. Base (Closed-Book)** — *Not used in our experiments*
LLM generates answers using only internal knowledge, no retrieval. Baseline for model capability.

**2. Oracle (Generation Ceiling)**
- **Setup:** Gold supporting context chunks are injected directly into the prompt, bypassing retrieval
- **Purpose:** Measures the **upper bound of generation quality** when retrieval is perfect
- **Usage in our experiments:** Used in all experiments to establish generation ceiling and isolate retrieval errors

**3. Mixed (Realistic RAG)**
- **Setup:** System must retrieve from a candidate pool containing both correct supporting chunks and distractor chunks
- **Purpose:** Measures **end-to-end RAG performance** including retrieval quality and generation from retrieved contexts
- **Usage in our experiments:** Primary evaluation setting for all experiments

**Comparative Analysis:**
By running both Mixed and Oracle in each experiment, we can **decompose errors**:
- **Retrieval errors:** Performance gap between Mixed and Oracle
- **Generation errors:** Performance gap between Oracle and theoretical perfect answer
- **Noise vulnerability:** Degradation when distractors are present in retrieved contexts

### 4.3 MIRAGE RAG-Specific Metrics

MIRAGE introduces four metrics that **partition all possible outcomes** (sum to 1.0) to measure RAG system adaptability:

**1. Noise Vulnerability (NV)** — *Lower is better*
- **Definition:** Fraction of queries where the system answered correctly in Base (closed-book) but **incorrectly in Mixed** (RAG with noise)
- **Interpretation:** Measures susceptibility to distractor contexts; system is "distracted" by irrelevant information
- **Related to:** Retrieval quality, prompt robustness

**2. Context Acceptability (CA)** — *Higher is better*
- **Definition:** Fraction of queries where the system answered incorrectly in Base but **correctly in Mixed** (RAG with noise)
- **Interpretation:** Measures ability to leverage retrieved context effectively to correct wrong answers
- **Related to:** Retrieval quality, context integration

**3. Context Insensitivity (CI)** — *Lower is better*
- **Definition:** Fraction of queries where the system answered incorrectly in both Base and Oracle
- **Interpretation:** Measures failure to utilize context even when perfect context is provided
- **Related to:** LLM generation capability, prompt design

**4. Context Misinterpretation (CM)** — *Lower is better*
- **Definition:** Fraction of queries where the system answered correctly in Base but **incorrectly in Oracle** (gold context)
- **Interpretation:** Measures hallucination or misinterpretation when gold context contradicts model's internal knowledge
- **Related to:** LLM capability, instruction-following

**Ideal RAG System Profile:**
- High CA (leverages context to improve answers)
- Low NV (robust to noise)
- Low CI (uses context when available)
- Low CM (correctly interprets gold context)

### 4.4 Metrics Taxonomy

Our evaluation employs a multi-dimensional metrics framework:

#### 4.4.1 Retrieval Quality Metrics
- **Recall@k:** Fraction of queries for which at least one gold supporting chunk appears in top-k retrieved chunks
- **nDCG@k (Normalized Discounted Cumulative Gain):** Ranking quality metric that rewards gold chunks appearing earlier in results
- **MRR (Mean Reciprocal Rank):** Average of 1/rank for the first gold chunk

#### 4.4.2 Generation Quality Metrics
- **Exact Match (EM):** Fraction of generated answers that exactly match gold answer strings (after normalization). Primary generation quality metric.
- **Answer Relevance:** Degree to which the generated answer directly addresses the user question, regardless of answer length or phrasing style. This metric captures semantic question-answer alignment and avoids penalizing correct but more verbose generative responses.
- **Semantic Answer Correctness:** Degree to which the generated answer is semantically equivalent to the gold/reference answer, even when wording differs. This metric credits correct paraphrases and meaning-preserving answer variants that strict lexical matching can miss.
- **F1 Score:** Token-level F1 between generated answer and gold answer (harmonic mean of precision/recall). *Reported in E1 only, F1 penalises generative verbosity and is dropped from E2-E5 in favour of EM_loose.*
- **Faithfulness:** Fraction of claims in generated answer that are supported by retrieved contexts. Computed via LLM-as-judge.
- **Groundedness:** Fraction of generated answer content that can be attributed to retrieved contexts. Computed via LLM-as-judge.

#### 4.4.3 Citation Quality Metrics
- **Citation Precision:** Fraction of cited spans that genuinely support the generated answer
- **Citation Recall:** Fraction of supporting spans that were actually cited by the system

#### 4.4.4 Efficiency Metrics
- **TTFT (Time-to-First-Token):** Latency from query submission to first generated token (ms)
- **Latency p50/p95:** Median and 95th percentile end-to-end query latency (ms)
- **Peak RAM/VRAM:** Maximum memory usage during query processing (MB)
- **Tokens In/Out:** Number of tokens processed (context) and generated (answer)
- **Index Build Time:** Offline cost to embed and index document corpus (seconds)

#### 4.4.5 MIRAGE RAG Adaptability Metrics
- **Noise Vulnerability (NV):** Susceptibility to distractors (lower is better)
- **Context Acceptability (CA):** Ability to leverage context (higher is better)
- **Context Insensitivity (CI):** Failure to use context (lower is better)
- **Context Misinterpretation (CM):** Hallucination with gold context (lower is better)

#### 4.4.6 Decision-Aware Metrics (E3, E5)
- **Selective Accuracy:** Fraction of answered queries that are correct (precision of answering)
- **Coverage:** Fraction of queries for which the system provides an answer (recall of answering)
- **AUPRC (Area Under Precision-Recall Curve):** Summary of abstention/decision quality across thresholds, computed on confidence-based answer vs abstain decisions. Chosen because abstention settings are commonly class-imbalanced, where precision-recall analysis is more informative than ROC summaries.
- **ECE (Expected Calibration Error):** Calibration quality of decision confidence. Lower ECE indicates predicted confidence better matches observed empirical outcomes (correctness/answerability frequencies).
- **Success@N:** Fraction of queries answered correctly within ≤N retrieval tool calls (E5)

#### 4.4.7 LLM-as-Judge Reliability Metric (E2-E5)
- **Human-Judge Cohen's Kappa:** Chance-corrected agreement between human spot-check labels and LLM-as-judge labels. This is the single reliability metric used to validate judge-based metrics (Faithfulness, Groundedness, Answer Relevance, Semantic Answer Correctness).

---

## 5. Shared Experimental Configuration

### 5.1 Hardware and Operating Environment
- **Target Hardware:** Mid-tier laptop (7th-gen Intel i7, 16GB RAM, GTX 1050 Ti 4GB VRAM or CPU-only)
- **Operating System:** Windows/Linux/macOS (cross-platform)
- **Deployment:** Fully local, no Docker, no external services

### 5.2 Core Components (Defaults)

**Embedding Model (Baseline):**
- `all-MiniLM-L6-v2`
- Alternative model explored in E2 (one small, recent open embedding)

**Vector Store:**
- **Chroma (embedded mode)** — in-process vector database, pip-installable, no server
- Supports cosine similarity search, metadata filtering, persistence

**Keyword Retriever:**
- **BM25** algorithm for lexical matching (E2, E5)
- Implementation via standard libraries (no Elasticsearch/Whoosh server)

**Hybrid Retrieval:**
- Score fusion or rank interleaving of BM25 + Vector results (E2, E5)
- Optional cross-encoder reranking (MiniLM-based, CPU, E2)

**Generator (Local SLM):**
- **gemma2-7b-it** via Ollama (instruction-tuned, 7B parameters)
- Fixed decoding parameters (temperature, max_tokens, top_p) for reproducibility

**Generator (Online Baseline):**
- **Gemini 2.5 Flash** (E4 only) — cloud API for comparison
- Matched decoding parameters to local SLM for fair comparison

### 5.3 Dataset
- **Primary:** MIRAGE (Mixed + Oracle settings)
- **Evaluation Modes:** Mixed (realistic RAG), Oracle (generation ceiling)

### 5.4 Parameter Sweeps (Where Applicable)
- **Top-k retrieval:** {3, 5, 10}
- **Selective answering threshold (E3):** {low, mid, high}
- **Agentic hops (E5):** ≤2 retrieval rounds

*Note: Chunk size sweep is not applicable — MIRAGE provides pre-chunked documents (37,800 fixed chunks).*

---

## 6. Experiments

### E1 — Vanilla Local RAG Baseline

**Goal:**
Establish a clean, reproducible local-only baseline implementing the core RAG pipeline (document ingestion → parsing → chunking → embedding → vector retrieval → generation). This experiment produces the **numerical anchor** for all subsequent ablations and extensions.

**Research Question Mapping:**
- **RQ1** (Architectural integration): Tests foundational SLM-retriever integration on constrained hardware; measures baseline resource consumption (RAM, VRAM, latency) for local RAG
- **RQ2** (Algorithm combinations): Establishes baseline with single embedding model (all-MiniLM-L6-v2) and vector-only retrieval; sweeps k parameter to understand basic retrieval trade-offs

**Objective Fulfillment:**
- **Objective A:** Implements complete lightweight RAG framework (core pipeline)
- **Objective B:** Establishes baseline retrieval performance with vector-only search

**Experimental Design:**

*Retrieval Configuration:*
- **Vector Store:** Chroma (embedded mode)
- **Retrieval Method:** Vector-only (cosine similarity)
- **Embedding Model:** all-MiniLM-L6-v2
- **Parameter Sweep:** k ∈ {3, 5, 10} (chunk size not applicable — MIRAGE uses pre-chunked documents)

*Generation Configuration:*
- **Local SLM:** gemma2-7b-it (Ollama)
- **Prompt Strategy:** Plain question-answering with instruction to provide citations

**Datasets:**
- **Primary:** MIRAGE Mixed (realistic RAG evaluation)
- **Ceiling:** MIRAGE Oracle (generation upper bound with gold contexts)

**Metrics:**
- *Retrieval:* Recall@k, nDCG@k, MRR
- *Generation:* EM, F1, Faithfulness, Groundedness
- *Efficiency:* TTFT, latency p50/p95, peak RAM/VRAM, index build time
- *MIRAGE:* NV, CA, CI, CM

*Note: Citation Precision/Recall is deferred to E2–E5, where MIRAGE prompts will include citation instructions. E1 uses plain prompts without citation elicitation.*

*Note: F1 is reported for E1 baseline but is known to penalise generative verbosity (see E1 results). EM_loose is the primary generation quality metric. F1 is dropped from E2–E5.*

**Expected Deliverables:**
1. Baseline metrics table (all k configurations)
2. Mixed vs Oracle performance comparison to isolate retrieval errors
3. Resource consumption profile (memory, latency, throughput)

**How to Interpret Results:**
- **Mixed vs Oracle gap:** Large gap indicates retrieval is primary bottleneck; small gap indicates generation limitation
- **MIRAGE metrics:** High NV or CI indicates system weaknesses; CA shows context utilization effectiveness
- **Efficiency profile:** Establishes resource budget for subsequent experiments
- **Parameter sensitivity:** Identifies impact of k on quality/efficiency trade-off

This baseline serves as the **reference point** for evaluating improvements in E2-E5.

---

### E2 — Hybrid Retrieval Exploration

**Goal:**
Systematically quantify accuracy/efficiency trade-offs from varying embedding models, retrieval methods (Vector-only vs BM25 vs Hybrid), and optional reranking under laptop constraints. Identify the **Local-Best** configuration to carry forward into E3-E5.

**Research Question Mapping:**
- **RQ2** (Algorithm combinations): *Primary focus* — compares embedding models, contrasts vector/BM25/hybrid retrieval, tests reranking; measures impact on retrieval accuracy, generation quality, and resource consumption
- **RQ1** (Integration impacts): Tests how hybrid retrieval integration (score fusion, reranking) affects system complexity and resource usage

**Objective Fulfillment:**
- **Objective B:** *Primary* — explores and compares efficient retrieval algorithms (BM25, hybrid, reranking)
- **Objective A:** Selects best local configuration for subsequent experiments

**Experimental Design:**

*Retrieval Configurations (Compared):*
1. **Vector-only** (E1 baseline, for reference)
2. **BM25-only** (keyword-based lexical matching)
3. **Hybrid (BM25 + Vector)** — score fusion or rank interleaving
4. **Hybrid + Reranker** (optional) — MiniLM cross-encoder on top-50 → top-k (CPU)

*Embedding Models (Compared):*
1. **all-MiniLM-L6-v2** (E1 baseline)
2. **One alternative small, recent open embedding** (e.g., newer sentence-transformers model)

*Fixed Components:*
- **Vector Store:** Chroma (embedded mode)
- **Generator:** gemma2-7b-it (Ollama)
- **Parameter Sweep:** Same k sweep as E1 for fair comparison

**Datasets:**
- **Primary:** MIRAGE Mixed
- **Ceiling:** MIRAGE Oracle

**Metrics:**
- *Retrieval:* Recall@k, Precision@k, nDCG@k, MRR (per retrieval method)
- *Generation:* EM, Answer Relevance, Semantic Answer Correctness, Faithfulness, Groundedness
- *Citation:* Citation Precision, Citation Recall (MIRAGE prompts include citation instructions from E2 onward)
- *Judge Reliability:* Human-Judge Cohen's Kappa (from stratified human spot-check sample)
- *Efficiency:* TTFT, latency p95, peak RAM/VRAM, index build time
- *MIRAGE:* NV, CA, CI, CM

**Expected Deliverables:**
1. Comparative table: Vector vs BM25 vs Hybrid vs Hybrid+Rerank (retrieval and generation metrics)
2. Embedding model comparison table
3. Pareto frontier plot: Quality (EM) vs Latency (p95)
4. **Selection of "Local-Best"** configuration (retrieval method + embedding model + parameters)

**How to Interpret Results:**
- **Hybrid vs Vector-only:** Does BM25 fusion improve Recall@k/nDCG? Is latency overhead acceptable?
- **Reranker impact:** Does cross-encoder reranking justify ~2-5x latency increase with quality gains?
- **Embedding model trade-off:** Does alternative embedding improve quality or reduce resource usage?
- **Local-Best selection:** Choose configuration with best quality at acceptable latency (<2s p95) and memory footprint

The **Local-Best** configuration from this experiment becomes the foundation for E3-E5.

#### E2 Local-Best Selection Rubric

**Hard constraints (eliminates config if violated):**
- Recall@k ≥ 0.80 (can't generate correct answers without retrieving gold)

**Weighted composite score (0–1):**

| Metric | Weight | Rationale |
|---|---|---|
| EM_loose (Mixed) | 0.40 | End-to-end answer quality — primary objective |
| nDCG@k | 0.25 | Retrieval ranking quality — gold chunk position matters |
| Citation quality (mean CitP, CitR) | 0.15 | Attribution quality, needed for E3+ |
| MIRAGE CA | 0.10 | Context utilisation — RAG benefit over closed-book |
| 1 − MIRAGE NV | 0.10 | Noise robustness — penalty for distractor vulnerability |

**Composite = Σ(weight × metric)**. All metrics are [0, 1] range. Pick highest composite. Tiebreaker (within 0.02): prefer simpler config (fewer components).

**k selection:** Evaluate composite at each k ∈ {3, 5, 10}. Pick the (config, k) pair with highest composite.

---

### E3 — Selective Answering (Decision-Aware RAG)

**Goal:**
Improve system reliability by implementing a **selective answering gate** that abstains from answering when retrieved context is insufficient. Evaluate whether this decision layer increases **correctness among answered queries** (Selective Accuracy) at the cost of reduced **coverage** (answer rate).

**Research Question Mapping:**
- **RQ1** (Architectural integration): *Primary focus* — tests integration of decision layer (sufficiency gate) with retrieval and generation; measures architectural overhead and resource impact
- **RQ2** (Effect across best retrieval): Evaluates how decision layer interacts with Local-Best retrieval configuration from E2

**Objective Fulfillment:**
- **Objective A:** *Primary* — implements decision layer (selective answering mechanism)
- **Objective C:** Evaluates quality vs efficiency trade-off (higher correctness vs lower coverage)

**Experimental Design:**

*System Configuration:*
- **Base System:** Local-Best pipeline from E2
- **Decision Gate:** Context sufficiency check before generation
  - If max retrieval score < τ (threshold) → return "Not enough evidence in knowledge base"
  - If max retrieval score ≥ τ → proceed with answer generation
- **Threshold Sweep:** τ ∈ {low, mid, high} (e.g., 0.3, 0.5, 0.7)

*Fixed Components:*
- Retrieval: Local-Best from E2
- Generator: gemma2-7b-it
- No additional retrieval hops or external models

**Datasets:**
- **Primary:** MIRAGE Mixed
- **Ceiling:** MIRAGE Oracle

**Metrics:**
- *Decision-Aware:* **Selective Accuracy** (correct among answered), **Coverage** (fraction answered), **AUPRC** (threshold-robust abstention quality), **ECE** (confidence calibration quality)
- *Generation:* EM, Answer Relevance, Semantic Answer Correctness, Faithfulness, Groundedness (for answered queries)
- *Citation:* Citation Precision, Citation Recall
- *Judge Reliability:* Human-Judge Cohen's Kappa (from stratified human spot-check sample)
- *Efficiency:* Latency overhead of decision gate, TTFT, p95
- *MIRAGE:* NV, CA, CI, CM (for answered queries)

**Expected Deliverables:**
1. Accuracy-Coverage curve (plot Selective Accuracy vs Coverage for different τ)
2. Comparison table: E2 (no gate) vs E3 (with gate at optimal τ)
3. Latency overhead analysis

**How to Interpret Results:**
- **Accuracy-Coverage curve:** Upward shift relative to E2 at matched coverage indicates gate improves reliability
- **Optimal threshold:** Identify τ that maximizes Selective Accuracy while maintaining acceptable Coverage (e.g., >70%)
- **False abstentions:** Measure cases where system abstains but Oracle setting succeeds (retrieval failure vs overly conservative gate)
- **Latency overhead:** Gate should add <50ms overhead

**Success Criterion:**
If Selective Accuracy increases by ≥5% at Coverage ≥70% compared to E2 baseline, the decision gate is justified.

---

### E4 — Local vs Online Comparison

**Goal:**
Directly compare the **Local-Best** pipeline (from E2) against a cloud-based RAG baseline using **Gemini 2.5 Flash** as the generator. Establish the **practical viability threshold** by quantifying the quality gap, efficiency gains, and cost trade-offs between local and online deployment.

**Research Question Mapping:**
- **RQ3** (Local vs cloud viability): *Primary and exclusive focus* — quantifies performance differences between local SLM and cloud LLM across quality, efficiency, and cost dimensions; determines at what quality threshold local deployment becomes practically viable

**Objective Fulfillment:**
- **Objective C:** *Primary* — evaluates framework effectiveness compared to cloud-based RAG systems

**Experimental Design:**

*Configurations Compared:*

1. **Local-Best:**
   - Retrieval: Local-Best from E2
   - Generator: gemma2-7b-it (Ollama, local)
   - All processing on-device

2. **Online Baseline:**
   - Retrieval: **Identical** to Local-Best (same retrieved contexts)
   - Generator: Gemini 2.5 Flash (cloud API)
   - Parity controls: matched decoding parameters (temperature, max_tokens, top_p, stop sequences)

*Note:* Online mode is **for comparison only**; core PicoRAG operation remains fully local.

**Datasets:**
- **Primary:** MIRAGE Mixed (realistic RAG evaluation)
- **Ceiling:** MIRAGE Oracle (isolates generator capability; shows Gemini's generation ceiling vs gemma2's)

**Metrics:**
- *Generation Quality:* EM, Answer Relevance, Semantic Answer Correctness, Faithfulness, Groundedness, Citation Precision/Recall
- *Retrieval Quality:* Recall@k, nDCG@k (identical for both, since retrieval is shared)
- *Judge Reliability:* Human-Judge Cohen's Kappa (from stratified human spot-check sample)
- *Efficiency:*
  - **Local:** TTFT, p95 latency, **energy per correct answer** (Joules/correct query)
  - **Online:** TTFT, p95 latency, **cost per correct answer** ($/correct query), **tokens/$**
- *MIRAGE:* NV, CA, CI, CM (compare local vs online adaptation to noise)

**Expected Deliverables:**
1. Side-by-side comparison table: Local-Best vs Online (all metrics)
2. Mixed vs Oracle comparison for both systems (isolates generation gap)
3. Quality-Efficiency-Cost trade-off analysis
4. Viability threshold determination

**How to Interpret Results:**

- **Generation Quality Gap (Mixed setting):**
  - If Local EM ≥ 80% of Online EM → local quality is competitive
  - If Local EM < 70% of Online EM → significant quality degradation

- **Oracle Comparison (Generation Ceiling):**
  - Gemini Oracle vs gemma2 Oracle shows inherent LLM capability gap
  - If Oracle gap is small but Mixed gap is large → retrieval/noise handling is the bottleneck, not LLM

- **Efficiency Comparison:**
  - Local TTFT typically 2-5x slower than Online (acceptable for offline use)
  - Local energy/correct answer should be <10 Wh (feasible on battery)
  - Online cost/correct answer: baseline for ROI calculation

- **MIRAGE Metrics:**
  - Compare NV (noise vulnerability): Does local SLM degrade more with distractors?
  - Compare CA (context acceptability): Does Gemini leverage context more effectively?


**Viability Threshold:**
Local deployment is **practically viable** if:
1. Local EM ≥ 75% of Online EM (acceptable quality degradation)
2. Local latency p95 < 5 seconds (usable interactive experience)
3. Local energy/correct answer < Online cost/correct answer (when amortized over 1000 queries)

---

### E5 — Agentic Multi-Hop RAG

**Goal:**
Evaluate an **agentic multi-hop controller** that autonomously orchestrates retrieval tools, reformulates queries when evidence is weak, and explicitly handles unanswerable queries. Assess whether agentic control (1) improves retrieval coverage via multi-hop reasoning, (2) enhances answer quality, and (3) correctly abstains when knowledge base lacks information — all while maintaining acceptable latency on constrained hardware.

**Research Question Mapping:**
- **RQ1** (Architectural integration): *Primary focus* — tests complex SLM↔retriever↔agentic-controller integration on-device; measures resource overhead of agentic control flow (tool calling, state management, multi-hop)
- **RQ2** (Hybrid & reformulation effects): Tests whether hybrid retrieval + query reformulation in agentic setting improves retrieval effectiveness

**Objective Fulfillment:**
- **Objective A:** *Primary* — implements advanced agentic control flow (multi-hop, tool calling, query reformulation)
- **Objective B:** Explores retrieval strategies in agentic context (tool selection, reformulation)

**Experimental Design:**

*Agentic Controller:*
- **Framework:** LangGraph (finite-state machine or message-passing graph)
- **Agent Policy:**
  1. **Initial Retrieval:** Call hybrid retrieval tool (BM25+Vector from E2)
  2. **Sufficiency Check:** If max score < τ → reformulate query and retry
  3. **Retry (Single):** Reformulated query → call hybrid retrieval again
  4. **Decision:** If retry max score < τ → return "Not in KB"; else generate answer with citations
- **Bounded Hops:** ≤2 retrieval tool calls to meet latency constraints
- **Tools Available:** Vector retriever (Chroma), BM25 retriever, score fusion, query reformulator (SLM-based)

*Base Configuration:*
- Retrieval: Hybrid (BM25+Vector) from E2
- Generator: gemma2-7b-it
- All processing local

**Datasets:**
- **Primary:** MIRAGE Mixed

**Metrics:**
- *Retrieval:* Recall@k, nDCG@k (after final retrieval step)
- *Multi-Hop:* **Success@N** (fraction of queries answered correctly within ≤N tool calls, N ∈ {1, 2})
- *Answerability:*
  - **KB-Coverage Accuracy:** Correctly abstains when KB lacks answer (true negatives)
  - **False-Negative Rate:** Abstains when KB contains answer (over-conservative)
- *Decision-Aware Calibration:* **AUPRC** (threshold-robust abstention quality), **ECE** (confidence calibration quality)
- *Generation:* EM, Answer Relevance, Semantic Answer Correctness, Faithfulness, Groundedness
- *Citation:* Citation Precision, Citation Recall
- *Judge Reliability:* Human-Judge Cohen's Kappa (from stratified human spot-check sample)
- *Efficiency:* Step count (tool calls per query), TTFT, latency p95, peak RAM
- *MIRAGE:* NV, CA, CI, CM

**Expected Deliverables:**
1. Comparison table: E2 (single-shot hybrid) vs E5 (agentic multi-hop)
2. Success@N breakdown (1-hop vs 2-hop success rates)
3. Reformulation effectiveness analysis (Recall improvement after retry)
4. Latency overhead analysis (per-hop cost)
5. Answerability metrics (abstention accuracy)

**How to Interpret Results:**

- **Multi-Hop Benefit:**
  - Compare Success@1 vs Success@2: Does query reformulation + retry improve correctness?
  - Measure Recall@k improvement after reformulation (2nd hop vs 1st hop)

- **Agentic vs Single-Shot (E2):**
  - If E5 EM > E2 EM by ≥3% → agentic control justifies latency overhead
  - If E5 latency p95 > 2× E2 latency → overhead may be prohibitive

- **Answerability Handling:**
  - High KB-Coverage Accuracy (>80%) + Low False-Negative Rate (<15%) → agent correctly identifies unanswerable queries
  - High False-Negative Rate → agent is overly conservative (abstains too often)

- **Efficiency on Constrained Hardware:**
  - Average step count <1.5 → most queries answered in 1 hop (reformulation rarely needed)
  - Average step count >1.8 → agent frequently retries (possible inefficiency)
  - Latency p95 <5s → acceptable for interactive use

**Success Criterion:**
Agentic control is beneficial if:
1. Success@2 > Success@1 by ≥5% (multi-hop improves coverage)
2. E5 EM ≥ E2 EM + 3% (quality improvement)
3. Latency p95 < 5 seconds (usable on constrained hardware)
4. KB-Coverage Accuracy > 75% (reliable abstention)

---

## 7. Reproducibility Framework

### 7.1 Configuration Schema

Each experimental run is defined by a YAML configuration file with the following schema:

```yaml
# Dataset Selection
dataset: mirage_mixed | mirage_oracle

# Embedding Configuration
embed:
  model: all-MiniLM-L6-v2 | <alternative-embedding-model>

# Indexing Configuration
index:
  type: chroma  # Fixed: Chroma embedded mode only
  # chunk_size not applicable — MIRAGE provides pre-chunked documents

# Retrieval Configuration
retrieval:
  method: vector | bm25 | hybrid | hybrid+rerank
  k: 3 | 5 | 10
  reranker: null | miniLM-cross-encoder  # Optional

# Agent Configuration
agent:
  mode: off | selective_gate | multihop
  threshold: 0.3 | 0.5 | 0.7  # For selective_gate, multihop
  max_hops: 2  # For multihop only

# Generator Configuration
generator:
  type: ollama:gemma2-7b-it | online:gemini-2.5-flash
  temperature: 0.7
  max_tokens: 512
  top_p: 0.9

# Reproducibility
seed: <integer>
git_commit: <sha>
notes: <free-text description>
```

### 7.2 Artifact Structure

Each experimental run generates outputs in the following directory structure:

```
/runs/<timestamp>/<config-hash>/
├── config.yaml              # Full configuration for this run
├── metrics.json             # Aggregated metrics with bootstrap CIs
├── samples.csv              # Per-query results (query, answer, gold, metrics)
├── sysinfo.json             # Hardware info (CPU, RAM, GPU, OS)
├── prompts/
│   ├── system_prompt.txt
│   └── few_shot_examples.txt (if applicable)
└── logs/
    ├── retrieval.log        # Retrieval debug logs
    └── generation.log       # Generation debug logs
```

**Config Hash:**
SHA-256 hash of canonical config representation (model + embed + index + chunk + k + retriever + agent + generator + seed), ensuring unique runs are disambiguated.

**Timestamp:**
ISO 8601 format: `YYYY-MM-DD_HH-MM-SS`

### 7.3 Telemetry and Aggregation


**Per-Query Telemetry (samples.csv):**
- Query ID, query text, generated answer, gold answer
- Retrieval: top-k chunk IDs, scores, Recall@k (binary), nDCG@k
- Generation: EM (binary), F1 (E1 only), Answer Relevance, Semantic Answer Correctness, Faithfulness, Groundedness
- Decision/Calibration (E3, E5): abstained (boolean), decision confidence score, calibrated decision probability, decision target label (for AUPRC/ECE computation)
- Citation: Precision, Recall
- Reliability (E2-E5): LLM-as-judge label, human spot-check label (sampled subset), agreement flag
- Latency: retrieval_ms, generation_ms, total_ms
- Resources: RAM snapshot (MB), VRAM snapshot (MB)

**Aggregated Metrics (metrics.json):**
- Mean + 95% bootstrap CI for all metrics
- MIRAGE RAG metrics: NV, CA, CI, CM
- Efficiency: TTFT p50/p95, latency p50/p95, peak RAM/VRAM
- System info: OS, CPU, RAM, GPU, Python version, library versions

### 7.4 Reproducibility Checklist

To reproduce any experimental run:
1. Check out the git commit SHA from `config.yaml`
2. Install dependencies from `requirements.txt` at that commit
3. Load the config file: `python run_experiment.py --config /runs/<timestamp>/<config-hash>/config.yaml`
4. Verify identical `config-hash` in output
5. Compare `metrics.json` (should match within bootstrap CI bounds given seed)

---

## 8. Summary

This experimental design provides a **comprehensive, progressive evaluation** of PicoRAG's lightweight local RAG framework:

1. **E1** establishes the baseline local RAG capability with vector-only retrieval
2. **E2** systematically explores retrieval algorithms and embeddings to identify the Local-Best configuration
3. **E3** extends with decision-aware selective answering to improve reliability
4. **E4** benchmarks Local-Best against cloud LLM baseline to establish viability threshold
5. **E5** demonstrates advanced agentic multi-hop reasoning on constrained hardware

Each experiment is **explicitly mapped** to research questions (RQ1-RQ3) and objectives (A-C), uses the **MIRAGE dataset** (Mixed + Oracle) for standardized evaluation, and reports a **comprehensive metrics suite** spanning retrieval quality, generation quality, citation quality, efficiency, and RAG-specific adaptability.

The **reproducibility framework** ensures all runs are tracked with immutable config hashes, full telemetry, and version control, enabling regression analysis and iterative refinement as the system evolves.

**Expected Outcome:**
This experimental sequence will provide empirical evidence to answer whether **lightweight local RAG on mid-tier hardware** can achieve practical viability (RQ3) through careful architectural integration (RQ1) and algorithm selection (RQ2), ultimately demonstrating that advanced RAG capabilities can be democratized beyond cloud infrastructure.
