# Table of Contents

## Abstract
## Acknowledgements
## List of Figures
## List of Tables
## List of Abbreviations

## Chapter 1: Introduction

### 1.1. Problem Statement: The Resource Gap in Neural Information Retrieval

### 1.2. Research Aim and Objectives

### 1.3. Research Questions

   - RQ1: Architectural Integration & Resource Efficiency
   - RQ2: Algorithm Combinations & Efficiency-Accuracy Trade-offs
   - RQ3: Local vs. Cloud Viability Threshold

### 1.4. Scope and Limitations

   - Focus on Text-based RAG
   - Hardware Constraints (Consumer-grade devices)
   - Dataset Scope (MIRAGE)
   - Local Inference via Ollama; OpenRouter for E4-Online and Judge Evaluations

### 1.5. Thesis Contributions

   - The PicoRAG Framework
   - Empirical Analysis of "Architecture vs. Scale"
   - Evaluation of Decision-Aware RAG for Resource-Constrained Deployment

### 1.6. Thesis Organisation

## Chapter 2: Literature Review

### 2.1. Retrieval-Augmented Generation (RAG)

   - Theoretical Foundations
   - Architecture Taxonomy (Naive, Advanced, Modular)

### 2.2. Efficient and On-Device AI

   - Small Language Models (SLMs) vs. Large Language Models (LLMs)
   - Resource-Constrained Inference

### 2.3. Information Retrieval Methodologies

   - Sparse Retrieval (BM25)
   - Dense Vector Retrieval (Bi-Encoders, Embedding Models)
   - Hybrid Fusion Strategies (Reciprocal Rank Fusion)
   - Cross-Encoder Reranking

### 2.4. Decision-Aware and Agentic RAG

   - Selective Answering and Abstention Mechanisms
   - Multi-Hop Retrieval and Query Reformulation

### 2.5. Evaluation of RAG Systems

   - Component-wise vs. End-to-End Evaluation
   - Automated Evaluation Frameworks (RAGAS, ARES)
   - LLM-as-Judge Paradigm
   - The MIRAGE Benchmark and RAG Adaptability Metrics
   - Calibration and Reliability in Generative Systems

## Chapter 3: Methodology (The PicoRAG Framework)

### 3.1. System Architecture Overview

### 3.2. Data Processing Layer

   - Document Ingestion (Pre-chunked MIRAGE Corpus)
   - Embedding Models (all-MiniLM-L6-v2 via ONNX, qwen3-embedding:4b via Ollama)
   - Indexing Strategy (ChromaDB PersistentClient, HNSW with Cosine Similarity)

### 3.3. Retrieval Layer

   - Vector Search (ChromaDB)
   - BM25 Sparse Retrieval (rank_bm25)
   - Hybrid Fusion (Reciprocal Rank Fusion)
   - Local Cross-Encoder Reranking (BAAI/bge-reranker-v2-m3 via FlagEmbedding)

### 3.4. Generation Layer

   - Model Selection (gemma3:4b-it-qat via Ollama)
   - Prompt Engineering (Base, Oracle, Mixed Modes; Conciseness & Citation Instructions)
   - Online Baseline (GPT-oss-120B + qwen3-embedding-8b via OpenRouter for E4 Comparison)

### 3.5. Decision and Control Layer

   - Pre-generation Gate: Cosine Distance Thresholding
   - Post-generation Gate: LLM Self-Abstention Detection
   - Dual-Gate Strategy (E5)

### 3.6. Agentic Multi-Hop Controller

   - Finite State Machine Design (Pure Python, No External Agent Frameworks)
   - Query Reformulation via LLM
   - Conditional Hybrid Retrieval Escalation (Vector → Hybrid on Retry)

### 3.7. Implementation Details

   - Technology Stack (Python 3.10, ChromaDB, Ollama, FlagEmbedding, httpx, OpenRouter API for judges/E4-Online)
   - LLM-as-Judge Implementation (Structured Output via Tool-Calling)
   - Experiment Runner Infrastructure and Checkpointing
   - Reproducibility Protocols (Seeds, Configuration, Artifact Structure)

## Chapter 4: Experimental Design and Evaluation Protocol

### 4.1. Dataset: MIRAGE

   - Composition and Source Domains (7,560 QA Pairs, 37,800 Pre-chunked Documents)
   - Evaluation Modes: Base (Closed-Book), Oracle (Gold Context), Mixed (Realistic RAG)
   - Answerable vs. Unanswerable Query Construction (100 + 20)
   - Partial Evaluation Subset and Rationale

### 4.2. Progressive Experimental Design

   - Experiment-to-Research-Question Mapping
   - **E1:** Vanilla Vector-Only RAG Baseline
   - **E2:** Hybrid Retrieval Exploration (6 Configs × 3 k Values)
   - **E3:** Selective Answering with Cosine Distance Gate (Threshold Sweep)
   - **E4:** Local vs. Online Pipeline Comparison (Gemma 4B + Qwen3-4B via Ollama vs. GPT-oss 120B + Qwen3-8B via OpenRouter)
   - **E5:** Agentic Multi-Hop RAG with Dual-Gate Abstention

### 4.3. Local-Best Selection Rubric

   - Weighted Composite Score (EM 0.40, nDCG 0.25, Citation 0.15, CA 0.10, 1−NV 0.10)
   - Simplicity Tiebreaker

### 4.4. LLM-as-Judge Protocol

   - Judge Models Used (GLM-4.5-Air, GLM-4.7-Flash, GPT-oss-120B)
   - Structured Output Enforcement via Tool-Calling
   - Judge Model Variation across Experiments (Noted as Construct Validity Consideration)

### 4.5. Viability Threshold Criteria

   - Local EM ≥ 75% of Online EM
   - Latency p95 < 5 Seconds
   - Cost-Effectiveness Comparison

### 4.6.  Evaluation Metrics

   #### 4.6. 1. Retrieval Quality

   - Recall@k, Precision@k, nDCG@k, MRR
   
   #### 4.6. 2. Generation Quality
   
   - Exact Match — Loose and Strict (EM_loose, EM_strict)
   - Token-level F1 (E1 Only; Dropped E2+ Due to Generative Verbosity Penalty)
   
   #### 4.6. 3. LLM-as-Judge Metrics
   
   - Faithfulness, Groundedness, Answer Relevance, Semantic Correctness
   
   #### 4.6. 4. Citation Quality
   
   - Citation Precision and Citation Recall
   
   #### 4.6. 5. MIRAGE RAG Adaptability Metrics
   
   - Noise Vulnerability (NV)
   - Context Acceptability (CA)
   - Context Insensitivity (CI)
   - Context Misinterpretation (CM)

   #### 4.6. 6. Decision-Aware Metrics (E3, E5)

   - Selective Accuracy, Coverage
   - AUPRC, ECE

   #### 4.6. 7. Agentic Metrics (E5)

   - Success@N (N ∈ {1, 2})
   - KB-Coverage Accuracy, False-Negative Rate
   - Hop Distribution, Average Tool Calls

   #### 4.6. 8. Latency

   - p50, p95, Mean (On-Device via Ollama for Local; API Round-Trip via OpenRouter for Online/Judge)

## Chapter 5: Results and Analysis

### 5.1. E1 — Vanilla Vector-Only RAG Baseline

   - Retrieval Performance (Recall@k, nDCG@k, MRR across k ∈ {3, 5, 10})
   - Generation Quality across Modes (Base, Oracle, Mixed)
   - MIRAGE Adaptability Profile (NV, CA, CI, CM)
   - Faithfulness and Groundedness (LLM-as-Judge)
   - F1 as a Verbosity Artefact: Rationale for Dropping from E2+
   - EM_strict vs. EM_loose Discrepancy

### 5.2. E2 — Hybrid Retrieval Exploration

   - Retrieval Comparison (6 Configurations: Vector MiniLM, BM25, Hybrid RRF MiniLM, Hybrid+Reranker, Vector Qwen3, Hybrid RRF Qwen3)
   - Generation Quality across Configurations and k Values
   - Citation Quality Analysis
   - MIRAGE Adaptability across Configurations
   - LLM-as-Judge Scores
   - Local-Best Selection: Qwen3 Vector k=5 (Composite Score Analysis)
   - Key Finding: RRF Dilution Effect and Embedding Model Impact

### 5.3. E3 — Selective Answering (Decision-Aware RAG)

   - Threshold Sweep Results (τ ∈ {0.75, 0.80, 0.85, 0.90})
   - Selective Accuracy vs. Coverage Trade-off Curve
   - Unanswerable Query Detection (100% at τ=0.75)
   - Comparison with E2 Ungated Baseline
   - Latency Reduction from Early Abstention

### 5.4. E4 — Local vs. Online Comparison

   - Experimental Setup (Local Ollama Pipeline vs. Online OpenRouter Pipeline)
   - Generation Quality Gap across Modes (Base: 22%, Oracle: 89%, Mixed: 81%)
   - MIRAGE Adaptability Comparison (CI and CA Differences)
   - LLM-as-Judge Score Comparison (Faithfulness, Groundedness, Relevance, Correctness)
   - Citation Quality: Local Slightly Outperforms Online
   - Viability Assessment: 80.9% EM Parity — PASS

### 5.5. E5 — Agentic Multi-Hop RAG

   - Agentic Architecture (Dual-Gate Finite State Machine)
   - Mixed EM Exceeds Oracle EM (Abstention Effect)
   - Multi-Hop Analysis: Success@1 vs. Success@2 (+1 Query Recovery)
   - Selective Accuracy (75.3%) and Coverage (74.2%)
   - KB-Coverage Accuracy (95.0%) and False-Negative Rate (12.0%)
   - Dual-Gate Mechanism: Cosine Gate vs. LLM Self-Abstention Contributions
   - Error Analysis (22 Wrong Answers: Ambiguous Entities, Partial Answers, Genuinely Wrong)
   - Abstention Breakdown (19 True Negatives, 12 False Negatives, 1 False Positive)

### 5.6. Cross-Experiment Comparison

   - End-to-End Generation Quality Progression (E1→E5)
   - Retrieval Performance Stability (E2→E5)
   - MIRAGE Metric Trajectories (NV≈0, CM=0 Throughout; CA Peaks at E5)
   - LLM-as-Judge Quality Progression
   - Citation Quality Progression
   - Selective Answering Evolution (E3 Single Gate → E5 Dual Gate)
   - Latency Profile across Experiments

### 5.7. Efficiency and Latency Analysis

   - Latency Profiles across Architectures (Local On-Device vs. API Round-Trip)
   - Hop Distribution and Tool Call Overhead (E5)
   - Note on Local Ollama vs. Online API Latency Differences

## Chapter 6: Discussion

### 6.1.  Answering the Research Questions

   #### 6.1. 1. RQ1: Architectural Integration and Resource Efficiency

   - Decision-layer intelligence as highest-leverage intervention
   - Post-generation self-abstention as critical innovation
   - Marginal value of multi-hop reformulation on compact corpora

   #### 6.1. 2. RQ2: Algorithm Combinations and Efficiency-Accuracy Trade-offs

   - Embedding quality matters more than retrieval complexity
   - RRF hybrid dilution vs. conditional escalation
   - k=5 as optimal depth for 4B-class generators

   #### 6.1. 3. RQ3: Local vs. Cloud Viability Threshold

   - 80.9% EM parity (E4) → 96.6% with agentic control (E5)
   - Retrieval as an equaliser (Base 4.5× gap → Mixed 1.2× gap)
   - Local exceeds cloud on faithfulness, groundedness, selective accuracy

### 6.2. Architecture Compensates for Scale

   - Interpreting the parity between Agentic 4B and Naive 120B models

### 6.3. The Role of Abstention in Trustworthy RAG

   - Precision-coverage trade-off analysis (E3 vs. E5)

### 6.4. Retrieval vs. Generation as the Bottleneck

   - Retrieval is not the bottleneck (Recall@5 ≥ 0.94)
   - Generation quality and context sensitivity (CI) as limiting factors

### 6.5. Threats to Validity

   #### 6.5.1. Internal Validity

   - E4-Online and judge latencies include API overhead (OpenRouter)
   - Judge model variation across experiments (GLM-4.5, GLM-4.7, GPT-oss-120B)
   - Rate-limiting and retry storms inflating p95/mean latency (E4)

   #### 6.5.2. External Validity

   - Partial evaluation subset (100–120 queries, not full 7,560)
   - Single dataset (MIRAGE) — generalisability to other domains
   - Single generator model (Gemma 3 4B) — results may not hold for other SLMs

   #### 6.5.3. Construct Validity

   - EM_loose as primary metric (substring matching favours verbose answers)
   - F1 penalty for generative models (dropped after E1)
   - E4 uses different embedding models per pipeline (4B local vs 8B online) — not a pure generator comparison

### 6.6. Limitations

   - Single hardware configuration (no cross-device benchmarking)
   - No energy or VRAM measurements
   - No human-judge validation (Cohen's Kappa not computed)
   - Partial runs only — no full 7,560-query evaluation
   - Single 4B model — no comparison across SLM families

## Chapter 7: Conclusion and Future Work

### 7.1. Summary of Contributions

   - Mapping back to Objectives A, B, C

### 7.2. Summary of Findings

### 7.3. Practical Implications

   - When local deployment is sufficient vs. when cloud is justified

### 7.4. Recommendations for Future Research

   - Full-scale evaluation (7,560 queries)
   - Cross-device benchmarking (different hardware configurations)
   - Larger and domain-diverse corpora
   - Human evaluation and judge calibration

## References

## Appendices

### A. Prompt Templates (Base, Oracle, Mixed, E5 Agentic, Judge Rubrics, Reformulation)

### B. Local-Best Selection Composite Score Calculations

### C. Detailed Metric Definitions (Mathematical Formulations of EM, F1, nDCG, AUPRC, ECE, NV/CA/CI/CM)

### D. Additional Results Tables (Full per-config E2 tables, per-threshold E3 tables)

### E. Model Specifications (gemma3:4b-it-qat, qwen3-embedding:4b, all-MiniLM-L6-v2, GPT-oss-120B, qwen3-embedding-8b, BAAI/bge-reranker-v2-m3)

### F. Experiment Configuration Details (Per-experiment: models, k values, thresholds, judge models, worker counts)

### G. Software Artifacts (Repository structure and usage)
