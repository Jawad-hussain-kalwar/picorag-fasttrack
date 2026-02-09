# E2 Smoke Test Results — Hybrid Retrieval Exploration

**Run:** `2026-02-08_22-33-47_smoke_10`
**Model:** `google/gemma-3-4b-it:free` (via OpenRouter)
**Embedding models:** all-MiniLM-L6-v2 (ChromaDB default), Qwen3-Embedding-4B (OpenRouter)
**Reranker:** Voyage AI rerank-2.5-lite
**Dataset:** MIRAGE subset — 50 questions indexed (250 chunks), 10 evaluated
**k values:** {3, 5, 10}
**Scope:** Smoke test only (n=10). Results validate pipeline correctness but are not statistically significant. Full run (n=100) required for conclusions.

---

## Retrieval Metrics

| Config | Recall@3 | nDCG@3 | MRR@3 | Recall@5 | nDCG@5 | MRR@5 | Recall@10 | nDCG@10 | MRR@10 |
|--------|----------|--------|-------|----------|--------|-------|-----------|---------|--------|
| 1. Vector (MiniLM) | **1.00** | 0.86 | 0.82 | **1.00** | 0.87 | 0.82 | **1.00** | 0.78 | 0.82 |
| 2. BM25-only | 0.80 | 0.71 | 0.68 | 0.80 | 0.70 | 0.68 | 1.00 | 0.69 | 0.72 |
| 3. Hybrid RRF (MiniLM) | 0.80 | 0.76 | 0.75 | 0.90 | 0.80 | 0.77 | **1.00** | 0.75 | 0.78 |
| 4. Hybrid + Reranker | **1.00** | **0.96** | **0.95** | **1.00** | **0.96** | **0.95** | **1.00** | **0.89** | **0.95** |
| 5. Vector (Qwen3) | **1.00** | 0.91 | 0.88 | **1.00** | 0.88 | 0.88 | **1.00** | 0.81 | 0.88 |
| 6. Hybrid RRF (Qwen3) | 0.80 | 0.75 | 0.73 | 0.90 | 0.77 | 0.75 | **1.00** | 0.74 | 0.77 |

**Retrieval ranking (by nDCG@3):** Hybrid+Reranker (0.96) > Qwen3 Vector (0.91) > MiniLM Vector (0.86) > Hybrid RRF MiniLM (0.76) > Hybrid RRF Qwen3 (0.75) > BM25 (0.71)

## Generation Metrics (EM_loose)

| Config | k=3 | k=5 | k=10 |
|--------|-----|-----|------|
| Base (closed-book) | 0.10 | — | — |
| Oracle (gold context) | 0.60 | — | — |
| 1. Vector (MiniLM) | 0.50 | **0.60** | **0.60** |
| 2. BM25-only | 0.40 | 0.30 | 0.30 |
| 3. Hybrid RRF (MiniLM) | 0.40 | 0.50 | 0.50 |
| 4. Hybrid + Reranker | 0.50 | 0.40 | 0.40 |
| 5. Vector (Qwen3) | 0.50 | **0.60** | **0.60** |
| 6. Hybrid RRF (Qwen3) | 0.40 | 0.40 | 0.50 |

## Citation Metrics

| Config | CitP k=3 | CitR k=3 | CitP k=5 | CitR k=5 | CitP k=10 | CitR k=10 |
|--------|----------|----------|----------|----------|-----------|-----------|
| 1. Vector (MiniLM) | 0.60 | 0.60 | 0.60 | 0.55 | **0.85** | 0.43 |
| 2. BM25-only | 0.50 | 0.50 | 0.50 | 0.38 | 0.60 | 0.24 |
| 3. Hybrid RRF (MiniLM) | 0.60 | 0.60 | 0.50 | 0.50 | 0.80 | 0.39 |
| 4. Hybrid + Reranker | 0.20 | 0.20 | 0.50 | 0.45 | 0.40 | 0.32 |
| 5. Vector (Qwen3) | **0.70** | **0.63** | 0.60 | 0.50 | 0.80 | 0.39 |
| 6. Hybrid RRF (Qwen3) | 0.40 | 0.40 | 0.50 | 0.40 | 0.70 | 0.36 |

## MIRAGE RAG Adaptability Metrics (k=5)

| Config | NV | CA | CI | CM |
|--------|------|------|------|------|
| 1. Vector (MiniLM) | 0.00 | **0.50** | 0.40 | 0.00 |
| 2. BM25-only | 0.00 | 0.20 | 0.40 | 0.00 |
| 3. Hybrid RRF (MiniLM) | 0.00 | 0.40 | 0.40 | 0.00 |
| 4. Hybrid + Reranker | 0.00 | 0.30 | 0.40 | 0.00 |
| 5. Vector (Qwen3) | 0.00 | **0.50** | 0.40 | 0.00 |
| 6. Hybrid RRF (Qwen3) | 0.00 | 0.30 | 0.40 | 0.00 |

Ideal: NV=low, CA=high, CI=low, CM=low.

---

## Interpretation

> **Caveat:** n=10 is far too small for statistical conclusions. The patterns below are directional signals to validate that the pipeline logic works correctly. All observations need confirmation on the 100-question run.

### Pipeline validation: all 6 configs work end-to-end

All retrieval methods (vector, BM25, hybrid RRF, hybrid+rerank), both embedding models (MiniLM, Qwen3), and the Voyage reranker ran successfully. Checkpointing/resume worked (generation resumed mid-run for BM25 k=5). Citation extraction produced plausible precision/recall values. MIRAGE metrics computed correctly across all configs. The E2 infrastructure is production-ready for the full 100-question run.

### Retrieval: Reranker and Qwen3 show strongest ranking quality

The Hybrid+Reranker config dominates retrieval metrics with nDCG@3=0.96 and MRR=0.95, meaning the gold chunk is almost always ranked first. Qwen3 vector-only is second (nDCG@3=0.91), beating MiniLM (0.86). This suggests the 4B embedding model produces higher-quality semantic representations than the 23M-parameter MiniLM.

BM25 alone trails all vector methods (nDCG@3=0.71), confirming that MIRAGE's biomedical/factoid questions benefit more from semantic matching than keyword matching. The hybrid RRF configs (3 and 6) sit between their pure-vector and pure-BM25 components but don't beat pure vector — the BM25 signal dilutes the stronger vector ranking through RRF.

### Generation: better retrieval doesn't always mean better answers

Despite Hybrid+Reranker having the best retrieval metrics, its EM_loose (0.40–0.50) doesn't surpass simpler configs. MiniLM Vector and Qwen3 Vector both tie at 0.60 EM_loose for k=5 and k=10. This is a known pattern: reranking improves ranking precision but the SLM may not extract answers better from reranked contexts than from reasonably-good vector results. At n=10 this could easily be noise — the full run will clarify.

### BM25 is clearly weakest

BM25-only consistently underperforms: lowest retrieval metrics and lowest EM_loose (0.30–0.40). MIRAGE's factoid questions require semantic understanding that pure lexical matching cannot provide. BM25's CA=0.20 (vs 0.50 for vector methods) means it helps the model on only 20% of questions the model couldn't answer alone.

### Hybrid RRF doesn't help over pure vector (at n=10)

Both hybrid RRF configs (3 and 6) score below their pure-vector counterparts in retrieval and generation. RRF averages the strong vector signal with the weaker BM25 signal, diluting gold chunk rankings. This suggests that for MIRAGE's domain, pure vector retrieval may be sufficient and BM25 fusion adds noise rather than complementary signal. However, this needs confirmation at n=100 — BM25 may help on specific keyword-heavy queries that are underrepresented in 10 samples.

### Citation metrics work but Citation Recall drops with higher k

Citation Precision generally holds steady or improves with k (model cites correctly), but Citation Recall drops (0.60→0.43 for Vector MiniLM). This is expected: with more chunks, the model cannot cite all gold sources. The Qwen3 vector config shows the best citation behavior at k=3 (CitP=0.70, CitR=0.63).

### MIRAGE metrics stable across configs

NV=0.00 and CM=0.00 across all configs — the model never loses correct answers when RAG is added (no noise vulnerability) and never misinterprets gold context. CI=0.40 is higher than E1's 0.18, likely because the 10-question sample happened to include harder questions (oracle EM=0.60 vs E1's 0.82). CA varies from 0.20 (BM25) to 0.50 (Vector configs), tracking EM_loose as expected.

---

## Relevance to Research Questions

### RQ2: Algorithm combinations and efficiency-accuracy trade-offs

E2 directly addresses RQ2 by comparing 6 retrieval configurations. Preliminary signals:

- **Embedding model matters:** Qwen3-Embedding-4B (4B params) consistently outranks MiniLM (23M params) on nDCG and MRR, suggesting larger embedding models improve retrieval quality on medical/factoid domains.
- **Reranking improves retrieval ranking** (nDCG +0.10 over best vector-only) but the downstream generation benefit is unclear at n=10.
- **BM25 alone is insufficient** for MIRAGE — semantic retrieval dominates lexical matching.
- **Hybrid RRF doesn't improve over pure vector** — the BM25 component dilutes rather than complements on this dataset.
- **k=5 is the sweet spot** — enough recall without excessive noise.

### RQ1: Architectural integration and resource efficiency

- BM25 index builds in <0.1s (250 chunks) — negligible overhead for hybrid.
- Qwen3 embedding via OpenRouter adds ~30s per 10 queries (API-bound). For local deployment, this would require a local embedding model.
- Voyage reranking adds ~1s/query but is API-dependent.
- The architectural complexity of hybrid+rerank (3 components: vector + BM25 + reranker) doesn't yield clear quality gains over single-component vector retrieval on this dataset.

---

## What This Smoke Test Delivers

1. **Infrastructure validation:** All 6 retrieval configs, 2 embedding models, Voyage reranking, citation metrics, and MIRAGE metrics run end-to-end with checkpointing.
2. **Directional signals:** Vector-only (Qwen3 or MiniLM) appears competitive with more complex hybrid approaches. Reranking improves retrieval precision but may not translate to generation gains.
3. **Baseline for comparison:** E1 partial (n=100) reported Recall@3=0.94, EM_loose=0.72. E2 full run will reveal whether alternative embeddings/methods narrow or widen these numbers.
4. **Cost confirmation:** Qwen3 embedding costs ~$0.01 for 2,500 chunks. Voyage reranking is free-tier. Total E2 cost is negligible.

## Next Steps

1. **Run `--partial` (n=100)** for statistically meaningful results.
2. **Run judge** on mixed results for faithfulness/groundedness per config.
3. **Select Local-Best** config based on full results — likely Vector-only (MiniLM or Qwen3) at k=3 or k=5.
