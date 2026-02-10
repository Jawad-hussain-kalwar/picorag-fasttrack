# Table of Contents

**Abstract**
**Acknowledgements**
**List of Figures**
**List of Tables**
**List of Abbreviations**

## Chapter 1: Introduction
1.1. **Problem Statement**: The Resource Gap in Neural Information Retrieval
1.2. **Research Aim and Objectives**
1.3. **Research Questions**
   - RQ1: Architectural Integration & Efficiency
   - RQ2: Algorithm Selection (Hybrid Retrieval & Embeddings)
   - RQ3: Viability Threshold (Local vs. Cloud)
1.4. **Scope and Limitations**
   - Focus on Text-based RAG
   - Hardware Constraints (Consumer-grade devices)
   - Dataset Scope (MIRAGE)
1.5. **Thesis Contributions**
   - The PicoRAG Framework
   - Empirical Analysis of "Architecture vs. Scale"
   - Evaluation of Decision-Aware RAG on Edge Devices
1.6. **Thesis Organization**

## Chapter 2: Literature Review
2.1. **Retrieval-Augmented Generation (RAG)**
   - Theoretical Foundations
   - Architecture Taxonomy (Naive, Advanced, Modular)
2.2. **Efficient and On-Device AI**
   - Small Language Models (SLMs) vs. Large Language Models (LLMs)
   - Quantization and Resource-Constrained Inference
2.3. **Information Retrieval Methodologies**
   - Sparse Retrieval (BM25, TF-IDF)
   - Dense Vector Retrieval (Bi-Encoders, Embeddings)
   - Hybrid Fusion and Reranking Strategies
2.4. **Evaluation of RAG Systems**
   - Component-wise vs. End-to-End Evaluation
   - Automated Evaluation Frameworks (RAGAS, ARES)
   - The MIRAGE Benchmark and Adaptability Metrics
   - Calibration and Reliability in Generative Systems

## Chapter 3: Methodology (The PicoRAG Framework)
3.1. **System Architecture Overview**
3.2. **Data Processing Layer**
   - Ingestion Pipeline
   - Embedding Model Selection (MiniLM vs. Qwen3)
   - Indexing Strategy (ChromaDB)
3.3. **Retrieval Layer**
   - Vector Search Configuration
   - Hybrid Implementation (BM25 + Vector Fusion)
   - Cross-Encoder Reranking
3.4. **Generation Layer**
   - Model Selection (`gemma-3-4b-it`)
   - Prompt Engineering (Conciseness & Citation Instructions)
3.5. **Decision and Control Layer**
   - Context Sufficiency Gating (Cosine Thresholding)
   - Self-Correction Mechanisms (LLM-as-Judge Gates)
   - Agentic Multi-Hop Controller (LangGraph Implementation)
3.6. **Implementation Details**
   - Technology Stack
   - Reproducibility Protocols (Seeding, Config Hashing)

## Chapter 4: Experimental Design and Evaluation Protocol
4.1. **Dataset: MIRAGE**
   - Composition and Domains
   - Evaluation Modes: Mixed (Realistic) vs. Oracle (Ceiling)
   - Robustness Sets and Distractor Analysis
4.2. **Experimental Setup**
   - **Experiment 1 (E1):** Vanilla Local Baseline
   - **Experiment 2 (E2):** Retrieval Optimization (Hybrid/Rerank/Embeddings)
   - **Experiment 3 (E3):** Reliability & Selective Answering
   - **Experiment 4 (E4):** Local vs. Cloud Viability Benchmark
   - **Experiment 5 (E5):** Agentic Multi-Hop Reasoning
4.3. **Evaluation Metrics**
   4.3.1. **Retrieval Quality**
      - Recall@k, Precision@k
      - Mean Reciprocal Rank (MRR)
      - Normalized Discounted Cumulative Gain (nDCG@k)
   4.3.2. **Generation Quality**
      - Exact Match (EM) - Strict vs. Loose
      - LLM-as-Judge Metrics: Faithfulness, Groundedness, Answer Relevance, Semantic Correctness
   4.3.3. **Citation Quality**
      - Citation Precision and Recall
   4.3.4. **Reliability and Calibration**
      - Selective Accuracy vs. Coverage
      - Area Under Precision-Recall Curve (AUPRC)
      - Expected Calibration Error (ECE)
      - KB-Coverage Accuracy and False-Negative Rates
   4.3.5. **MIRAGE Adaptability Metrics**
      - Noise Vulnerability (NV)
      - Context Acceptability (CA)
      - Context Insensitivity (CI)
      - Context Misinterpretation (CM)
   4.3.6. **Efficiency and Resource Usage**
      - Latency (p50, p95)
      - Memory Footprint (RAM/VRAM)
      - Cost Analysis (Local vs. Cloud API)

## Chapter 5: Results and Analysis
5.1. **Phase I: Establishing the Baseline (E1)**
   - Vector-only performance analysis
   - The "Answer Everything" pitfall (High Coverage, Low Reliability)
5.2. **Phase II: Optimizing the Retrieval Engine (E2)**
   - Impact of Embedding Model Size (MiniLM vs. Qwen3)
   - Efficacy of Hybrid Retrieval and Reranking on Edge Hardware
   - Selection of the "Local-Best" Configuration
5.3. **Phase III: Enhancing System Reliability (E3)**
   - The Accuracy-Coverage Trade-off
   - Evaluation of Threshold-based Gating
   - Analysis of Unanswerable Query Detection
5.4. **Phase IV: The Local vs. Cloud Gap (E4)**
   - Comparative Benchmarking: PicoRAG vs. Gemini/GPT-4 class models
   - Decomposing the Error: Retrieval Gap vs. Generation Gap
   - Viability Assessment for Edge Deployment
5.5. **Phase V: Bridging the Gap with Agentic Control (E5)**
   - Impact of Multi-Hop Reasoning on Retrieval Recall
   - Effectiveness of Query Reformulation
   - Dual-Gating (Cosine + LLM) Analysis
   - **Key Finding:** Closing the Performance Gap through Architecture
5.6. **Efficiency and Latency Analysis**
   - Latency Profiles across Architectures
   - Throughput vs. Quality Trade-offs

## Chapter 6: Discussion
6.1. **Architecture Compensates for Scale**
   - Interpreting the parity between Agentic 4B and Naive 120B models
6.2. **The Role of Abstention in Trustworthy RAG**
   - Analysis of E3/E5 selective accuracy gains
6.3. **Retrieval vs. Generation Bottlenecks on Edge Devices**
6.4. **Limitations of the Study**
   - Validity Threats (Internal, External, Construct)

## Chapter 7: Conclusion and Future Work
7.1. **Summary of Findings**
7.2. **Implications for Edge AI Deployment**
7.3. **Recommendations for Future Research**

**References**

**Appendices**
A. **Prompt Templates** (System Prompts, Judge Instructions)
B. **Detailed Metric Definitions** (Mathematical Formulations of AUPRC, ECE, etc.)
C. **Additional Results Tables** (Per-domain breakdown)
D. **Software Artifacts** (Repository structure and usage)
