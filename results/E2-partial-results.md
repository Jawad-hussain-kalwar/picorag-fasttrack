# E2 Partial Results — Hybrid Retrieval Exploration

**Run:** `2026-02-09_09-46-57_partial_100`
**Model:** `google/gemma-3-4b-it`
**Embedding models:** all-MiniLM-L6-v2, Qwen3-Embedding-4B
**Reranker:** Voyage AI rerank-2.5-lite
**Judge model:** `z-ai/glm-4.7-flash`
**Dataset:** MIRAGE subset — 500 questions indexed (2,500 chunks), 100 evaluated
**k values:** {3, 5, 10}

---

## Retrieval Metrics

| Config | Recall@3 | P@3 | nDCG@3 | MRR@3 | Recall@5 | P@5 | nDCG@5 | MRR@5 | Recall@10 | P@10 | nDCG@10 | MRR@10 |
|--------|----------|-----|--------|-------|----------|-----|--------|-------|-----------|------|---------|--------|
| 1. Vector (MiniLM) | 0.89 | 0.56 | 0.82 | 0.80 | 0.96 | 0.44 | 0.83 | 0.81 | 0.98 | 0.34 | 0.82 | 0.81 |
| 2. BM25-only | 0.86 | 0.36 | 0.75 | 0.72 | 0.94 | 0.31 | 0.77 | 0.74 | 1.00 | 0.28 | 0.74 | 0.74 |
| 3. Hybrid RRF (MiniLM) | 0.88 | 0.42 | 0.79 | 0.77 | 0.96 | 0.35 | 0.81 | 0.79 | 1.00 | 0.33 | 0.75 | 0.79 |
| 4. Hybrid + Reranker | **0.97** | **0.54** | **0.93** | **0.92** | **0.99** | **0.41** | **0.92** | **0.93** | **1.00** | **0.33** | **0.86** | **0.93** |
| 5. Vector (Qwen3) | 0.93 | 0.51 | 0.87 | 0.85 | 0.97 | 0.39 | 0.87 | 0.86 | 1.00 | 0.31 | 0.83 | 0.87 |
| 6. Hybrid RRF (Qwen3) | 0.91 | 0.38 | 0.85 | 0.84 | 0.98 | 0.32 | 0.85 | 0.85 | 0.99 | 0.30 | 0.78 | 0.86 |

**Retrieval ranking (by nDCG@3):** Hybrid+Reranker (0.93) > Qwen3 Vector (0.87) > Hybrid Qwen3 (0.85) > MiniLM Vector (0.82) > Hybrid MiniLM (0.79) > BM25 (0.75)

**Indexing time:** MiniLM 0.06s, BM25 1.15s, Qwen3 0.42s (2,500 chunks)

## Generation Metrics (EM_loose)

| Config | k=3 | k=5 | k=10 |
|--------|-----|-----|------|
| Base (closed-book) | 0.05 | — | — |
| Oracle (gold context) | 0.71 | — | — |
| 1. Vector (MiniLM) | 0.59 | 0.58 | 0.57 |
| 2. BM25-only | 0.50 | 0.53 | 0.54 |
| 3. Hybrid RRF (MiniLM) | 0.61 | 0.60 | 0.56 |
| 4. Hybrid + Reranker | 0.63 | 0.61 | **0.63** |
| 5. Vector (Qwen3) | 0.57 | **0.64** | 0.61 |
| 6. Hybrid RRF (Qwen3) | 0.60 | 0.60 | 0.56 |

## Citation Metrics

| Config | CitP k=3 | CitR k=3 | CitP k=5 | CitR k=5 | CitP k=10 | CitR k=10 |
|--------|----------|----------|----------|----------|-----------|-----------|
| 1. Vector (MiniLM) | 0.60 | 0.49 | 0.52 | 0.37 | 0.59 | 0.25 |
| 2. BM25-only | 0.63 | 0.55 | 0.62 | 0.48 | 0.65 | 0.32 |
| 3. Hybrid RRF (MiniLM) | **0.67** | **0.57** | **0.65** | **0.51** | 0.61 | 0.24 |
| 4. Hybrid + Reranker | 0.61 | 0.53 | 0.61 | 0.44 | 0.60 | 0.23 |
| 5. Vector (Qwen3) | 0.58 | 0.51 | 0.63 | 0.47 | 0.60 | 0.26 |
| 6. Hybrid RRF (Qwen3) | 0.67 | **0.63** | 0.59 | 0.47 | 0.57 | 0.27 |

## MIRAGE RAG Adaptability Metrics

| Config | NV k=3 | CA k=3 | CI | CM | NV k=5 | CA k=5 | NV k=10 | CA k=10 |
|--------|--------|--------|------|------|--------|--------|---------|---------|
| 1. Vector (MiniLM) | 0.00 | 0.54 | 0.29 | 0.00 | 0.01 | 0.54 | 0.00 | 0.52 |
| 2. BM25-only | 0.00 | 0.45 | 0.29 | 0.00 | 0.00 | 0.48 | 0.00 | 0.49 |
| 3. Hybrid RRF (MiniLM) | 0.00 | 0.56 | 0.29 | 0.00 | 0.00 | 0.55 | 0.00 | 0.51 |
| 4. Hybrid + Reranker | 0.00 | 0.58 | 0.29 | 0.00 | 0.01 | 0.57 | 0.00 | 0.58 |
| 5. Vector (Qwen3) | 0.01 | 0.53 | 0.29 | 0.00 | 0.01 | **0.60** | 0.01 | 0.57 |
| 6. Hybrid RRF (Qwen3) | 0.01 | 0.56 | 0.29 | 0.00 | 0.00 | 0.55 | 0.00 | 0.51 |

Ideal: NV=low, CA=high, CI=low, CM=low.

## LLM-as-Judge Scores

| Config | Faith k=3 | Ground k=3 | Relev k=3 | Correct k=3 | Faith k=5 | Ground k=5 | Relev k=5 | Correct k=5 | Faith k=10 | Ground k=10 | Relev k=10 | Correct k=10 |
|--------|-----------|------------|-----------|-------------|-----------|------------|-----------|-------------|------------|-------------|------------|--------------|
| 1. Vector (MiniLM) | 0.95 | 0.96 | 0.94 | 0.86 | 0.99 | 0.97 | 0.92 | 0.82 | 0.90 | 0.89 | 0.89 | 0.79 |
| 2. BM25-only | 0.90 | 0.90 | 0.89 | 0.74 | 0.93 | 0.88 | 0.86 | 0.72 | 0.90 | 0.91 | 0.87 | 0.73 |
| 3. Hybrid RRF (MiniLM) | 0.93 | 0.93 | 0.93 | 0.83 | 0.94 | 0.94 | 0.95 | 0.84 | 0.93 | 0.93 | 0.89 | 0.78 |
| 4. Hybrid + Reranker | **0.97** | **0.97** | 0.93 | 0.85 | 0.96 | 0.96 | 0.93 | 0.83 | 0.96 | 0.96 | 0.93 | 0.84 |
| 5. Vector (Qwen3) | 0.95 | 0.95 | **0.95** | 0.83 | **0.97** | **0.97** | 0.94 | **0.86** | 0.95 | 0.95 | 0.93 | 0.83 |
| 6. Hybrid RRF (Qwen3) | 0.95 | 0.95 | 0.94 | 0.84 | 0.95 | 0.94 | 0.93 | 0.82 | 0.91 | 0.92 | 0.88 | 0.77 |

Judge success rate: 81–92 judged per 100 predictions (8–19% null from tool-call parse failures).

## Efficiency

| Metric | Value |
|--------|-------|
| Peak RAM | 453.8 MB |
| MiniLM index | 0.06s |
| BM25 index | 1.15s |
| Qwen3 index | 0.42s |
| Chunks indexed | 2,500 |
| Avg generation latency | 1.1–1.7s per query |
| Total eval questions | 100 |

---

## Interpretation

### Retrieval: Reranker dominates, Qwen3 beats MiniLM

The Hybrid+Reranker config is the clear retrieval winner: Recall@3=0.97, nDCG@3=0.93, MRR=0.92. The Voyage reranker pushes nearly every gold chunk into position 1. At k=10, all configs except Hybrid Qwen3 achieve perfect recall (1.00).

Qwen3-Embedding-4B consistently outperforms MiniLM (23M params) on ranking quality: nDCG@3 of 0.87 vs 0.82 for vector-only, confirming that a 4B-parameter embedding model produces meaningfully better semantic representations on MIRAGE's biomedical/factoid domain.

BM25 alone is the weakest retriever (nDCG@3=0.75) — MIRAGE's questions require semantic understanding that keyword matching cannot provide. At k=10 BM25 does achieve Recall@10=1.00, meaning lexical overlap eventually surfaces the gold chunk but buries it in lower ranks.

### Hybrid RRF: dilution, not enhancement

Both hybrid RRF configs rank between their pure-vector and BM25 components. Hybrid MiniLM (nDCG@3=0.79) is worse than Vector MiniLM (0.82). Hybrid Qwen3 (0.85) is slightly worse than Vector Qwen3 (0.87). RRF averages the strong vector signal with the weaker BM25 signal, diluting gold chunk rankings rather than boosting them. On MIRAGE, where semantic queries dominate, BM25 adds noise to the fusion.

The exception is Hybrid+Reranker, where Voyage's cross-encoder re-scores the RRF candidate set and recovers the ranking. This means reranking compensates for RRF dilution, but at the cost of an extra step per query, that adds slight latency.

### Generation: top configs cluster at 0.60–0.64 EM_loose

The generation ceiling is Oracle=0.71. This is lower than E1's 0.82, reflecting the stricter "Answer concisely" system prompt introduced in E2, which curbs the model's verbosity but may reduce the likelihood of serendipitous exact matches.

Best mixed configs:
- **Qwen3 Vector k=5:** 0.64 EM_loose — the single best generation score
- **Hybrid+Reranker k=3 and k=10:** 0.63
- **Hybrid MiniLM k=3:** 0.61

The top-5 configs span only a 4-point range (0.60–0.64), meaning generation quality is relatively insensitive to retrieval method once recall exceeds ~0.90. The model's ability to extract the correct answer from context plateaus well before perfect retrieval.

BM25 lags at 0.50–0.54, the only config where retrieval quality clearly hurts generation.

### The k sweet spot: k=3 or k=5

For most configs, k=3 and k=5 produce similar or better EM_loose than k=10. More chunks add noise: Vector MiniLM drops from 0.59 (k=3) to 0.57 (k=10), Hybrid MiniLM from 0.61 to 0.56. The main exception is Qwen3 Vector, where k=5 (0.64) beats k=3 (0.57) — Qwen3's higher retrieval precision means the 4th and 5th chunks are more likely gold-relevant.

Hybrid+Reranker is stable across all k values (0.61–0.63), confirming that reranking places gold chunks early regardless of total k.

### Base vs Oracle vs Mixed: RAG adds massive value

Base (0.05) → Oracle (0.71) → best Mixed (0.64). The model cannot answer MIRAGE questions from parametric knowledge alone (5% EM). RAG context closes 83% of the gap to the gold-context ceiling. The 7-point gap between Oracle and best Mixed represents a combination of retrieval misses (~3% at k=5) and distractor confusion.

### MIRAGE metrics: consistent, expected pattern

- **NV ≈ 0.00–0.01:** Near-zero noise vulnerability across all configs. The model never loses a correct closed-book answer after receiving RAG context. This is ideal.
- **CA = 0.45–0.60:** Context acceptability tracks EM_loose. The Qwen3 Vector k=5 achieves the highest CA (0.60), meaning RAG context corrects 60% of queries the model couldn't answer alone.
- **CI = 0.29 (constant):** Context insensitivity reflects the 29% of queries that even oracle context cannot fix. This is higher than E1's 0.18, likely due to the "Answer concisely" constraint preventing the model from explaining its way to a correct answer when the gold label requires specific phrasing.
- **CM = 0.00 (constant):** Zero context misinterpretation. The model never produces a wrong answer specifically because of context that it would have gotten right without context.

The MIRAGE profile {NV≈0, CM=0, CA≈0.55–0.60, CI=0.29} indicates a model that trusts and benefits from retrieval without being corrupted by it.

### Judge scores: faithfulness is high, consistent across configs

Faithfulness ranges 0.90–0.99 across all configs, with the reranker and Qwen3 configs slightly higher (0.95–0.97) than BM25 (0.90). This aligns with retrieval quality: better-ranked chunks give the model cleaner evidence, reducing the need to extrapolate.

Groundedness closely mirrors faithfulness (within 0–2 points for most configs), confirming the model's answers are derived from retrieved chunks rather than parametric knowledge. The one outlier is BM25 k=5 (faith=0.93, ground=0.88), where the 5-point gap suggests the model sometimes adds framing beyond what BM25's lower-quality chunks provide.

Semantic correctness (0.72–0.86) shows wider spread, tracking EM_loose patterns. The judge considers partial correctness, so this metric captures answers that are close but not exact matches.

Answer relevance is uniformly high (0.87–0.95), meaning the model rarely produces off-topic answers regardless of retrieval config.

At higher k, all judge scores dip slightly (faith from ~0.96 to ~0.91), consistent with the distraction effect seen in EM_loose — more chunks don't increase hallucination per se, but they dilute the model's focus.

### Citation behavior: precision stable, recall drops with k

Citation Precision holds at 0.57–0.67 across configs and k values — when the model cites a chunk, it's usually a gold chunk. Citation Recall drops sharply from k=3 (~0.55) to k=10 (~0.25), as the model cannot cite all 10 chunks and must choose a subset. Hybrid RRF configs show slightly better citation precision (0.65–0.67) than pure vector, possibly because the RRF ranking provides more distinct chunks that are easier to attribute.

---

## Local-Best Selection

Per the E2 design, Local-Best is the config carried forward to E3–E5. The selection rubric uses a weighted composite score: **EM_loose (0.40) + nDCG (0.25) + Citation (0.15) + CA (0.10) + (1-NV) (0.10)**, subject to the constraint Recall@k ≥ 0.80.

| Candidate | Recall | EM | nDCG | Cit (mean) | CA | 1-NV | Composite Score |
|-----------|--------|----|------|------------|----|------|-----------------|
| **5_vector_qwen3 k=5** | 0.97 | 0.64 | 0.87 | 0.55 | 0.60 | 0.99 | **0.715** |
| 4_hybrid_rerank_minilm k=3 | 0.97 | 0.63 | 0.93 | 0.57 | 0.58 | 1.00 | **0.728** |
| 4_hybrid_rerank_minilm k=10| 1.00 | 0.63 | 0.86 | 0.42 | 0.58 | 1.00 | 0.687 |
| 3_hybrid_minilm k=3 | 0.88 | 0.61 | 0.79 | 0.62 | 0.56 | 1.00 | 0.690 |

**Winner:** `4_hybrid_rerank_minilm k=3` achieves the highest numeric score (0.728). However, the rubric specifies a tiebreaker for scores within 0.02: **prefer simpler config (fewer components)**.

`5_vector_qwen3` (0.715) is within margin (0.013 diff) and represents a significantly simpler architecture (single embedding model, no secondary reranking step).

**Selected Local-Best: `5_vector_qwen3` at k=5** — chosen for its balance of top-tier generation quality (0.64 EM), high retrieval performance without reranking overhead, and architectural simplicity.

---

## Addressing Research Questions

### RQ1: Integration and Resource Efficiency
*What architectural design choices yield optimal performance while maintaining minimal resource consumption?*

The experiments demonstrate that **component simplicity outperforms complexity** on this dataset. The Hybrid RRF approach added computational overhead (BM25 indexing + fusion) but degraded performance compared to pure vector retrieval due to signal dilution. While adding a Reranker provided the absolute best retrieval metrics (Recall@3=0.97), it did not yield a statistically significant improvement in downstream generation (EM 0.63 vs 0.64 for Qwen3 Vector). Thus, the optimal integration strategy is a **strong single-stage dense retriever** (Qwen3) rather than a multi-stage hybrid pipeline, minimizing both latency and system complexity.

### RQ2: Algorithm Combinations and Trade-offs
*How do combinations of embeddings and retrieval algorithms affect the efficiency-accuracy trade-off?*

1.  **Embedding Impact:** Switching from MiniLM (23M params) to Qwen3 (4B params) provided a massive boost in ranking quality (nDCG@3 +0.05) and generation quality (EM +0.05), with negligible indexing time difference (0.42s vs 0.06s). This suggests that for local RAG, **investing VRAM in a larger embedding model is high-leverage**.
2.  **Retrieval Method:** BM25 is insufficient for MIRAGE's semantic queries. Hybrid RRF is counter-productive without a reranker. The trade-off curve favors **Vector-only retrieval with a capable embedding model** as the "knee in the curve" solution.
3.  **k-depth:** k=5 appears optimal for the 4B generator; k=10 introduces sufficient noise to degrade performance despite perfect recall.

---

## Comparison with E1

| Metric | E1 (100 Qs, 500 chunks) | E2 Best (500 Qs, 2,500 chunks) |
|--------|-------------------------|--------------------------------|
| Oracle EM_loose | 0.82 | 0.71 |
| Best Mixed EM_loose | 0.72 (k=3) | 0.64 (Qwen3 k=5) |
| Best Recall@3 | 0.94 | 0.97 (Reranker) |
| CI | 0.18 | 0.29 |

**Key Differences:**
1.  **Corpus Expansion:** E2 increases the retrieval search space by 5x (2,500 chunks vs 500). Despite this significantly harder task, the **Hybrid+Reranker** config achieved higher Recall@3 (0.97) than E1's baseline (0.94), demonstrating the effectiveness of the advanced retrieval methods.
2.  **Prompt Constraint:** The drop in Oracle generation quality (0.82 → 0.71) corresponds to the addition of the "Answer concisely" instruction. This constraint forces the model to trade verbosity for precision, which exposes a truer measure of the model's ability to extract exact factoid answers without "hedging."

Comparison of absolute values between E1 and E2 is less relevant than E2's internal comparison, which successfully identifies `5_vector_qwen3` as the robust Local-Best configuration for the larger dataset.
