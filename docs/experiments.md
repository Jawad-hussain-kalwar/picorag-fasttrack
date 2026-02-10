# PicoRAG — Experimental Design

**Version:** 2.0 (Post-Experiment Revision)
**Last Updated:** 2026-02-10

---

## 1. Project Overview

### 1.1 What is PicoRAG?

**PicoRAG** is a lightweight Retrieval-Augmented Generation (RAG) framework designed for resource-constrained personal computers. The system combines Small Language Models (SLMs) with optimized retrieval mechanisms to enable knowledge-grounded text generation on mid-tier hardware.

*Implementation note:* Local LLM inference (`gemma3:4b-it-qat`) and embedding (`qwen3-embedding:4b`) ran on-device via Ollama throughout E1–E5. Reranking (E2) used `BAAI/bge-reranker-v2-m3` via the FlagEmbedding package, also locally. Only the online comparison pipeline in E4 (`openai/gpt-oss-120b:exacto` for generation, `qwen3-embedding-8b` for retrieval) and the LLM-as-judge evaluations across all experiments were routed through the OpenRouter API. Latency figures for the local pipeline reflect on-device inference times; E4-Online and judge latencies include API round-trip overhead.

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
- Designed for local, offline operation (local inference via Ollama — see §1.1 note)
- No Docker/containers required
- No external vector database services (ChromaDB embedded mode only)
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

- **Measurable aspects:** Response quality (EM), retrieval accuracy (Recall@k, nDCG, MRR), latency profiles (p50, p95), LLM-as-judge quality dimensions (Faithfulness, Groundedness, Semantic Correctness), MIRAGE adaptability (NV, CA, CI, CM)
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
- **E4** directly compares Local-Best configuration against online cloud LLM (`openai/gpt-oss-120b:exacto`)
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
- **E4** directly compares the full local pipeline (Ollama-based) against a full online pipeline (OpenRouter-based) using `openai/gpt-oss-120b:exacto` (120B parameters) and `qwen3-embedding-8b`
- Uses both Mixed (RAG setting) and Oracle (generation ceiling) to quantify the local-vs-online quality gap
- Quantifies quality gap and establishes practical viability threshold

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

**1. Base (Closed-Book)**
LLM generates answers using only internal knowledge, no retrieval. Baseline for model capability.
- **Usage in our experiments:** Used in all experiments to compute MIRAGE RAG adaptability metrics (NV, CA, CI, CM), which require Base results as the closed-book reference point.

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
- **Latency p50/p95:** Median and 95th percentile end-to-end query latency (ms). *Note: Local pipeline latencies are on-device inference times via Ollama. E4-Online and judge latencies include OpenRouter API round-trip overhead.*
- **Peak RAM:** Maximum memory usage during query processing (MB). *Reported in E2 only (453.8 MB).*
- **Index Build Time:** Offline cost to embed and index document corpus (seconds)
- **Average Generation Latency:** Mean per-query generation time (ms)

*Metrics not measured:* TTFT, peak VRAM, tokens in/out, energy per correct answer, and cost per correct answer were planned but not captured in the partial evaluation runs. This is acknowledged as a limitation.

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

#### 4.4.7 LLM-as-Judge Protocol (E1-E5)

Judge-based metrics (Faithfulness, Groundedness, Answer Relevance, Semantic Answer Correctness) are computed via a second LLM acting as an impartial evaluator. Structured output is enforced via tool-calling (`submit_judgment` tool with `tool_choice: auto`) to avoid regex parsing of free-text responses.

**Judge models used (varied across experiments):**
- **E1:** `z-ai/glm-4.5-air`
- **E2, E3:** `z-ai/glm-4.7-flash`
- **E4, E5:** `openai/gpt-oss-120b:exacto`

*Note: Judge model variation across experiments is a construct validity concern. Cross-experiment judge score comparisons should be interpreted cautiously. E4 and E5 share the same judge model and are directly comparable.*

*Planned but not performed:* Human-Judge Cohen's Kappa (stratified spot-check for judge reliability validation) was not conducted in the partial evaluation runs.

---

## 5. Shared Experimental Configuration

### 5.1 Hardware and Operating Environment
- **Target Hardware:** Mid-tier laptop (7th-gen Intel i7, 16GB RAM)
- **Operating System:** Windows (MINGW64_NT-10.0-19045)
- **Inference:** Local LLM inference (`gemma3:4b-it-qat`) and embedding (`qwen3-embedding:4b`) ran on-device via Ollama. ChromaDB (vector store) and BM25 indexing also run locally. The E4 online comparison pipeline and LLM-as-judge calls used OpenRouter API.
- **Deployment:** No Docker, no external vector database services

### 5.2 Core Components (Defaults)

**Embedding Models:**
- **Baseline (E1):** `all-MiniLM-L6-v2` (23M parameters) — ChromaDB's built-in ONNX embedding, runs locally in-process
- **E2–E5 (Local-Best):** `qwen3-embedding:4b` (4B parameters) — runs locally via Ollama

**Vector Store:**
- **ChromaDB 1.4.1 (embedded mode)** — in-process vector database, pip-installable, no server
- Cosine similarity search (HNSW index), metadata filtering, `PersistentClient`

**Keyword Retriever:**
- **BM25** algorithm for lexical matching (E2 comparison, E5 hop-2 fallback)
- Pure Python implementation via `rank_bm25` library (no Elasticsearch/Whoosh server)

**Hybrid Retrieval:**
- **Reciprocal Rank Fusion (RRF)** combining BM25 + Vector results (E2, E5 hop-2)
- **Reranker (E2 only):** `BAAI/bge-reranker-v2-m3` — local cross-encoder reranking via FlagEmbedding package

**Generator (Local SLM):**
- **`gemma3:4b-it-qat`** (4B parameters) — runs locally via Ollama
- Fixed decoding parameters for reproducibility

**Generator (Online Baseline):**
- **`openai/gpt-oss-120b:exacto`** (120B parameters, E4 only) — accessed via OpenRouter API
- Matched decoding parameters to local SLM for fair comparison

### 5.3 Dataset
- **Primary:** MIRAGE (Mixed + Oracle + Base settings)
- **Evaluation Modes:** Mixed (realistic RAG), Oracle (generation ceiling), Base (closed-book, for MIRAGE adaptability metrics)
- **Evaluation Scale:** All experiments used **partial subsets** for rapid iteration:
  - **E1:** 100 answerable queries, 500 chunks indexed
  - **E2–E5:** 100 answerable + 20 unanswerable queries (120 total), 500 questions indexed (2,500 chunks)
  - Full-scale runs (7,560 queries, 37,800 chunks) were not performed

### 5.4 Parameter Sweeps (Where Applicable)
- **Top-k retrieval:** {3, 5, 10} (E1, E2); k=5 fixed for E3–E5 (Local-Best)
- **Selective answering threshold (E3):** τ ∈ {0.75, 0.80, 0.85, 0.90}
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
- **Generator:** `gemma3:4b-it-qat` (4B parameters, via Ollama)
- **Prompt Strategy:** Plain question-answering without citation instructions
- **Judge model:** `z-ai/glm-4.5-air` (via OpenRouter)

**Datasets:**
- **Subset:** 100 answerable queries, 500 chunks indexed
- **Modes:** Base (closed-book), Oracle (gold context), Mixed k ∈ {3, 5, 10}

**Metrics Reported:**
- *Retrieval:* Recall@k, Precision@k, nDCG@k, MRR
- *Generation:* EM_loose, EM_strict, F1 (token-level), Faithfulness, Groundedness
- *Efficiency:* Average generation latency (ms), index build time (s)
- *MIRAGE:* NV, CA, CI, CM

*Note: Citation Precision/Recall is deferred to E2–E5, where MIRAGE prompts include citation instructions. E1 uses plain prompts without citation elicitation.*

*Note: F1 is reported for E1 baseline but is known to penalise generative verbosity (see E1 results). EM_loose is the primary generation quality metric. F1 is dropped from E2–E5 in favour of EM_loose.*

*Note: TTFT, peak RAM/VRAM, and detailed latency percentiles were not captured in the E1 partial run.*

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

*Retrieval Configurations (6 configs compared):*
1. **Vector-only (MiniLM)** — all-MiniLM-L6-v2, E1 baseline reference
2. **BM25-only** — keyword-based lexical matching
3. **Hybrid RRF (MiniLM)** — Reciprocal Rank Fusion of BM25 + MiniLM vector
4. **Hybrid + Reranker** — RRF candidates re-scored by `BAAI/bge-reranker-v2-m3` via FlagEmbedding (local)
5. **Vector-only (Qwen3)** — `qwen3-embedding:4b` via Ollama
6. **Hybrid RRF (Qwen3)** — RRF of BM25 + Qwen3 vector

*Embedding Models (Compared):*
1. **all-MiniLM-L6-v2** (23M parameters, local ONNX via ChromaDB)
2. **qwen3-embedding:4b** (4B parameters, local via Ollama)

*Fixed Components:*
- **Vector Store:** ChromaDB (embedded mode)
- **Generator:** `gemma3:4b-it-qat` (via Ollama)
- **Judge model:** `z-ai/glm-4.7-flash` (via OpenRouter)
- **Parameter Sweep:** k ∈ {3, 5, 10} (same as E1)

**Datasets:**
- **Subset:** 500 questions indexed (2,500 chunks), 100 evaluated
- **Modes:** Base, Oracle, Mixed (all 6 configs × 3 k values)

**Metrics Reported:**
- *Retrieval:* Recall@k, Precision@k, nDCG@k, MRR (per config × k)
- *Generation:* EM_loose, Answer Relevance, Semantic Answer Correctness, Faithfulness, Groundedness
- *Citation:* Citation Precision, Citation Recall (citation instructions added to prompts from E2 onward)
- *Efficiency:* Peak RAM (453.8 MB), index build times (MiniLM 0.06s, BM25 1.15s, Qwen3 0.42s), avg generation latency
- *MIRAGE:* NV, CA, CI, CM (per config × k)

**Deliverables:**
1. Comparative table: all 6 configs × 3 k values (retrieval and generation metrics)
2. Embedding model comparison (MiniLM 23M vs Qwen3 4B)
3. Citation quality comparison across configs
4. **Selection of "Local-Best"** configuration via weighted composite rubric

**Actual Result:** Local-Best selected as **Config 5 (Vector Qwen3, k=5)** — highest EM_loose (0.64) with simpler architecture than Hybrid+Reranker (0.728 vs 0.715 composite, within 0.02 tiebreaker margin → simpler config preferred).

**Key Findings:**
- **Hybrid RRF dilutes** rather than enhances vector signal on semantic QA tasks
- **Qwen3 (4B) embedding outperforms MiniLM (23M)** on ranking quality (nDCG@3: 0.87 vs 0.82)
- **Reranker provides best pure retrieval** (nDCG@3=0.93) but marginal generation gain (+1 EM point vs Qwen3 vector)
- **Top configs cluster at 0.60–0.64 EM** — generation quality plateaus once Recall@k > 0.90

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
- **Base System:** Local-Best pipeline from E2 (Qwen3 vector, k=5)
- **Decision Gate:** Cosine distance threshold before generation
  - If max retrieval similarity < τ → return "Not enough evidence in knowledge base"
  - If max retrieval similarity ≥ τ → proceed with answer generation
- **Threshold Sweep:** τ ∈ {0.75, 0.80, 0.85, 0.90}

*Fixed Components:*
- Retrieval: Local-Best from E2 (Qwen3 vector, k=5)
- Generator: `gemma3:4b-it-qat` (via Ollama)
- Judge model: `z-ai/glm-4.7-flash` (via OpenRouter)
- No additional retrieval hops or external models

**Datasets:**
- **Subset:** 120 queries (100 answerable + 20 unanswerable)
  - Unanswerable queries constructed by excluding gold chunks from index
- **Modes:** Base, Oracle, Mixed (gated)

**Metrics Reported:**
- *Decision-Aware:* **Selective Accuracy**, **Coverage**, **AUPRC**, **ECE** (per threshold)
- *Generation:* EM_loose, Answer Relevance, Semantic Answer Correctness, Faithfulness, Groundedness (answered queries only)
- *Citation:* Citation Precision, Citation Recall
- *Efficiency:* p50 latency (758ms at τ=0.75 — faster than E2 due to skipped generation on abstained queries)
- *MIRAGE:* NV, CA, CI, CM (answered queries only)

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

**Actual Result:** τ=0.75 selected — Coverage 80.8%, Selective Accuracy 64.9% (+0.9% over E2), all 20 unanswerable queries correctly rejected (100% unanswerable detection). Success criterion partially met: coverage criterion met (80.8% > 70%), but selective accuracy gain was modest (+0.9% vs ≥5% target). The primary value was qualitative: the system now correctly identifies unanswerable queries.

---

### E4 — Local vs Online Comparison

**Goal:**
Directly compare the **Local-Best** pipeline (from E2) against a cloud-scale RAG baseline using **`openai/gpt-oss-120b:exacto`** (120B parameters) as the generator. Establish the **practical viability threshold** by quantifying the quality gap between local and online deployment.

**Research Question Mapping:**
- **RQ3** (Local vs cloud viability): *Primary and exclusive focus* — quantifies performance differences between local SLM and cloud LLM across quality, efficiency, and cost dimensions; determines at what quality threshold local deployment becomes practically viable

**Objective Fulfillment:**
- **Objective C:** *Primary* — evaluates framework effectiveness compared to cloud-based RAG systems

**Experimental Design:**

*Configurations Compared:*

1. **Local (Gemma 3 4B):**
   - Retrieval: Local-Best from E2 (`qwen3-embedding:4b` via Ollama, k=5)
   - Generator: `gemma3:4b-it-qat` (4B parameters, via Ollama)
   - Gate: Cosine distance τ=0.75 (from E3)

2. **Online (GPT-oss 120B):**
   - Retrieval: `qwen3-embedding-8b` (via OpenRouter API, k=5)
   - Generator: `openai/gpt-oss-120b:exacto` (120B parameters, via OpenRouter API)
   - Gate: Cosine distance τ=0.75 (same threshold)
   - Parity controls: matched decoding parameters

*Note:* The local pipeline ran entirely on-device via Ollama; the online pipeline ran entirely via OpenRouter API. The two pipelines use different embedding models (4B local vs 8B online) in addition to different generators, so the comparison captures the combined effect of embedding quality and generator capability. Latency differences reflect both model size and local-vs-API overhead.

**Datasets:**
- **Subset:** 120 queries (100 answerable + 20 unanswerable), 2,500 chunks
- **Modes:** Base, Oracle, Mixed (both generators, gated at τ=0.75)
- **Judge model:** `openai/gpt-oss-120b:exacto` (via OpenRouter)

**Metrics Reported:**
- *Generation Quality:* EM_loose, Answer Relevance, Semantic Answer Correctness, Faithfulness, Groundedness
- *Citation:* Citation Precision, Citation Recall
- *Retrieval Quality:* Recall@5, Precision@5, nDCG@5, MRR (per pipeline — embeddings differ)
- *Decision-Aware:* Selective Accuracy, Coverage
- *Latency:* p50, p95, mean (local via Ollama, online via OpenRouter API)
- *MIRAGE:* NV, CA, CI, CM (both generators, answered queries only)

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
  - Online Oracle vs Local Oracle shows inherent LLM capability gap
  - If Oracle gap is small but Mixed gap is large → retrieval/noise handling is the bottleneck, not LLM

- **MIRAGE Metrics:**
  - Compare NV (noise vulnerability): Does local SLM degrade more with distractors?
  - Compare CA (context acceptability): Does the larger model leverage context more effectively?
  - Compare CI (context insensitivity): Is the smaller model less able to use gold context?

**Viability Threshold:**
Local deployment is **practically viable** if:
1. Local EM ≥ 75% of Online EM (acceptable quality degradation)
2. Local latency p95 < 5 seconds (usable interactive experience)

*Note: Criterion 3 (energy/cost comparison) was not measured in the partial run.*

**Actual Result:** Local Mixed EM (0.60) / Online Mixed EM (0.74) = **80.9%** — **PASS** (exceeds 75% threshold). Latency criterion not directly assessable due to API rate-limit inflation, but p50 of 1.5s suggests compliance in single-query scenarios.

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
- **Framework:** Pure Python finite state machine (`src/agent.py`, ~280 lines). No LangGraph, LangChain, or external agent frameworks.
- **Agent Policy (Dual-Gate State Machine):**
  1. **Hop 1:** Qwen3 vector search (k=5)
  2. **Pre-generation gate:** Cosine distance check (τ=0.75)
     - Pass → GENERATE answer
     - Fail ↓
  3. **Reformulate:** LLM rewrites query with different keywords
  4. **Hop 2:** Vector search (reformulated) + BM25 search → RRF hybrid fusion
  5. **Pre-generation gate:** Best confidence across both hops
     - Pass → GENERATE with best hop's context
     - Fail → ABSTAIN
  6. **Post-generation gate:** LLM self-abstention detection + empty/citation-only answer detection
     - If triggered → override to ABSTAIN
- **Bounded Hops:** ≤2 retrieval rounds to meet latency constraints
- **Tools Available:** Qwen3 vector retriever (ChromaDB), BM25 retriever, RRF score fusion, LLM query reformulator

*Base Configuration:*
- Hop 1 retrieval: Qwen3 vector (k=5, from E2 Local-Best, via Ollama)
- Hop 2 retrieval: Hybrid vector+BM25 via RRF (fallback only)
- Generator: `gemma3:4b-it-qat` (via Ollama)
- Judge model: `openai/gpt-oss-120b:exacto` (via OpenRouter)

**Datasets:**
- **Subset:** 120 queries (100 answerable + 20 unanswerable), 2,500 chunks
- **Modes:** Base, Oracle, Agentic Mixed

**Metrics Reported:**
- *Retrieval:* Recall@5, Precision@5, nDCG@5, MRR (after final retrieval step)
- *Multi-Hop:* **Success@N** (N ∈ {1, 2}), hop distribution, avg tool calls per query
- *Answerability:*
  - **KB-Coverage Accuracy:** Correctly abstains when KB lacks answer
  - **False-Negative Rate:** Abstains when KB contains answer
- *Decision-Aware:* **Selective Accuracy**, **Coverage**, **AUPRC**, **ECE**
- *Generation:* EM_loose, Answer Relevance, Semantic Answer Correctness, Faithfulness, Groundedness
- *Citation:* Citation Precision, Citation Recall
- *Efficiency:* Step count (avg 2.42 tool calls), p50 latency (1,231ms mixed), p95 latency (4,664ms)
- *MIRAGE:* NV, CA, CI, CM (answered queries only)

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

**Success Criteria and Actual Results:**

| Criterion | Target | Actual | Verdict |
|-----------|--------|--------|---------|
| Success@2 > Success@1 by ≥5% | +5% | +1% (66%→67%) | **NOT MET** — reformulation recovered only 1 query on compact corpus |
| E5 EM ≥ E2 EM + 3% | +3% | +7.7% (64%→71.7%) | **MET** — driven by dual-gate abstention, not retrieval improvement |
| Latency p95 < 5 seconds | <5s | 4,664ms | **MET** (marginally) |
| KB-Coverage Accuracy > 75% | >75% | 95.0% | **MET** — near-perfect unanswerable detection |

**Key finding:** The agent's primary value is not in multi-hop retrieval recovery (marginal) but in its dual-gate abstention mechanism, which achieves 75.3% selective accuracy (+10.4% over E3) by catching "high-confidence retrieval, low-quality answer" scenarios via post-generation self-abstention.

---

## 7. Reproducibility Framework

### 7.1 Configuration

Each experiment is configured via module-level constants in `src/config.py` and command-line flags on the experiment runners (`run_e1.py` through `run_e5.py`). Key parameters per experiment:

| Parameter | E1 | E2 | E3 | E4 | E5 |
|-----------|----|----|----|----|-----|
| Embedding | MiniLM (local ONNX) | MiniLM + Qwen3 (Ollama) | Qwen3 (Ollama) | Qwen3 (Ollama) / Qwen3-8B (API, online) | Qwen3 (Ollama) |
| Retrieval | Vector | 6 configs | Vector (LB) | Vector (LB) | Vector + Hybrid fallback |
| k | {3, 5, 10} | {3, 5, 10} | 5 | 5 | 5 |
| Gate | None | None | τ ∈ {0.75–0.90} | τ=0.75 | Dual (τ=0.75 + LLM) |
| Generator | gemma3:4b-it-qat (Ollama) | gemma3:4b-it-qat (Ollama) | gemma3:4b-it-qat (Ollama) | gemma3:4b-it-qat (Ollama) + gpt-oss-120b (API) | gemma3:4b-it-qat (Ollama) |
| Judge | glm-4.5-air | glm-4.7-flash | glm-4.7-flash | gpt-oss-120b | gpt-oss-120b |
| Queries | 100 | 100 | 120 | 120 | 120 |
| Chunks | 500 | 2,500 | 2,500 | 2,500 | 2,500 |

*Note: No formal YAML config schema was used. Configuration is code-level, tracked via git commits.*

### 7.2 Artifact Structure

Each experimental run generates outputs in the following directory structure:

```
runs/<experiment>/
├── <timestamp>_partial_100/
│   ├── retrieval_results.json    # Per-query retrieval results (chunk IDs, scores)
│   ├── base_results.json         # Closed-book generation results
│   ├── oracle_results.json       # Gold-context generation results
│   ├── mixed_results.json        # RAG generation results (per k or gated)
│   ├── judge_results.json        # LLM-as-judge evaluation scores
│   └── metrics_summary.json      # Aggregated metrics
```

**Timestamp format:** `YYYY-MM-DD_HH-MM-SS`

### 7.3 Telemetry

**Per-Query Data (in results JSON files):**
- Query ID, query text, generated answer, gold answer(s)
- Retrieval: top-k chunk IDs, cosine distances, Recall@k (binary), nDCG@k
- Generation: EM_loose (binary), EM_strict (binary), F1 (E1 only)
- Decision (E3, E5): abstained (boolean), max confidence score
- Citation: extracted citation indices, precision, recall
- Latency: generation_ms
- Judge scores (separate file): Faithfulness, Groundedness, Answer Relevance, Semantic Correctness

### 7.4 Reproducibility

To reproduce any experimental run:
1. Check out the relevant git commit
2. Install dependencies: `.venv\Scripts\python.exe -m pip install -r requirements.txt`
3. Set `OPENROUTER_API_KEY` in `.env`
4. Run the experiment: `.venv\Scripts\python.exe run_e<N>.py --partial`

*Note: Local results depend on Ollama model versions and hardware. Online results (E4-Online, judges) depend on OpenRouter API availability. Bootstrap confidence intervals were not computed for partial runs.*

---

## 8. Summary

This experimental design provides a **progressive evaluation** of PicoRAG's RAG framework across five experiments:

1. **E1** establishes the baseline with vector-only retrieval (MiniLM, k sweep) → EM=0.72
2. **E2** explores 6 retrieval configurations × 2 embeddings → selects Local-Best (Qwen3 vector, k=5, EM=0.64)
3. **E3** adds cosine distance gating (τ=0.75) → Selective Accuracy=64.9%, 100% unanswerable detection
4. **E4** compares Gemma 4B vs GPT-oss 120B → local achieves 80.9% of online EM (viability threshold passed)
5. **E5** adds agentic dual-gate controller → Selective Accuracy=75.3%, Mixed EM=0.717 (exceeds own Oracle)

Each experiment is mapped to research questions (RQ1-RQ3) and objectives (A-C), uses the **MIRAGE dataset** (Base + Mixed + Oracle) for standardized evaluation, and reports metrics spanning retrieval quality, generation quality, citation quality, decision-aware metrics, and RAG-specific adaptability (NV/CA/CI/CM).

**Key deviations from original design:**
- Partial evaluation subsets (100–120 queries) — full-scale runs not performed
- E4 online pipeline uses different embedding model (`qwen3-embedding-8b`) than local (`qwen3-embedding:4b`), so E4 is not a pure generator comparison
- Judge models varied across experiments (construct validity concern for cross-experiment comparisons)
- Human-Judge Cohen's Kappa not computed
- Energy/cost efficiency metrics not measured
- Bootstrap confidence intervals not computed

**Outcome:**
The experiments demonstrate that a **4B-parameter model with intelligent architecture (dual-gate abstention, conditional hybrid retrieval) achieves quality parity with a 120B model** on trust metrics (faithfulness, groundedness, selective accuracy), with the remaining gap in semantic correctness only. The primary lever for improving local RAG quality is **decision-layer intelligence, not model scale**.
