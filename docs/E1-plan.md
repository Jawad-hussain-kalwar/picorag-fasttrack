# E1 — Vanilla Local RAG Baseline: Implementation Plan

## Context

PicoRAG has a working interactive pipeline (app.py → src/pipeline.py → src/retrieve.py + src/generate.py) that answers questions from local `.txt` files via ChromaDB + OpenRouter. E1 adds **experiment infrastructure** to evaluate this pipeline against the MIRAGE benchmark.

**Model:** `google/gemma-3-4b-it:free` via OpenRouter (no Ollama).

**MIRAGE data is already local** at `MIRAGE/mirage/`:
- `dataset.json` — 7,560 QA pairs (query, answer[], query_id, source)
- `doc_pool.json` — 37,800 chunks (mapped_id, doc_chunk, doc_name, support: 0|1)
- `oracle.json` — gold context per query_id (single best chunk with support=1)

---

## Why Oracle Mode in E1?

Oracle mode feeds the **gold context chunk** directly to the LLM, bypassing retrieval. Critical because:

1. **Establishes the generation ceiling.** If gemma-3-4b-it scores EM=40% on Oracle, then Mixed can never exceed 40%.
2. **Decomposes errors.** Mixed EM=20%, Oracle EM=40% → retrieval loses half the answers. Mixed EM=20%, Oracle EM=22% → LLM is the bottleneck.
3. **Enables MIRAGE metrics.** NV and CA require per-query Mixed vs Oracle comparison.
4. **Baseline for E2-E5.** Oracle ceiling from E1 shows room for improvement.

## Why Base Mode?

Base sends questions with **NO context** — tests what the LLM already knows. Required for:
- **CI** (Context Insensitivity): wrong in Base AND Oracle → LLM fundamentally can't answer
- **CM** (Context Misinterpretation): correct in Base, wrong in Oracle → context confuses LLM

7,560 API calls, no retrieval. Cheap and essential for complete MIRAGE diagnostics.

---

## Three Evaluation Modes

| Mode | Context Given | Purpose |
|------|--------------|---------|
| **Base** | None | What does the LLM already know? |
| **Oracle** | Gold chunk (perfect) | Generation ceiling — best the LLM can do |
| **Mixed** | Top-k from ChromaDB | Real RAG evaluation |

---

## Metrics

### Retrieval Quality (local, no API calls)
- **Recall@k**: At least one gold chunk in top-k?
- **Precision@k**: Fraction of top-k that are gold
- **nDCG@k**: Ranking quality
- **MRR**: 1/rank of first gold chunk

### Generation Quality (per-query)
- **EM_loose**: Gold answer appears as substring in output (primary metric)
- **EM_strict**: Exact match
- **F1**: Token-level overlap

### MIRAGE RAG Adaptability (all 3 modes required)
- **NV** (Noise Vulnerability): Base correct, Mixed wrong → lower is better
- **CA** (Context Acceptability): Base wrong, Mixed correct → higher is better
- **CI** (Context Insensitivity): Base wrong, Oracle wrong → lower is better
- **CM** (Context Misinterpretation): Base correct, Oracle wrong → lower is better

### Efficiency
- Per-query latency (retrieval_ms, generation_ms)
- Index build time
- Peak RAM

---

## Parameter Sweep

MIRAGE provides fixed pre-chunked documents (37,800 chunks). Chunk size sweep does not apply.
- **k ∈ {3, 5, 10}** (retrieval depth)

---

## Two Run Modes

### Partial (100 questions)
Dev/testing. Deterministic subset (first 100 sorted by query_id) + matching chunks only.

| Mode | Calls |
|------|-------|
| Base | 100 |
| Oracle | 100 |
| Mixed k=3 | 100 |
| Mixed k=5 | 100 |
| Mixed k=10 | 100 |
| **Total** | **500** (~25 min) |

### Full (7,560 questions)
Complete MIRAGE evaluation.

| Mode | Calls |
|------|-------|
| Base | 7,560 |
| Oracle | 7,560 |
| Mixed ×3 k's | 22,680 |
| **Total** | **37,800** (~31.5 hours) |

Both modes checkpoint to JSONL per (mode, k) — resumable after interruption.

---

## CLI

```bash
.venv\Scripts\python.exe run_e1.py --partial              # 100 questions
.venv\Scripts\python.exe run_e1.py --full                  # 7,560 questions
.venv\Scripts\python.exe run_e1.py --full --resume         # resume interrupted
.venv\Scripts\python.exe run_e1.py --partial --phase retrieval  # retrieval only
```

---

## Runner Phases

### Phase A — Retrieval (local, no API calls)
1. Load MIRAGE data (filtered if partial)
2. Index chunks into ChromaDB
3. Query all questions at max(k), slice for each k
4. Compute Recall@k, Precision@k, nDCG@k, MRR
5. Save retrieval results for Phase B

### Phase B — Generation (API calls, checkpointed)
1. Base mode: question → LLM → checkpoint
2. Oracle mode: question + gold chunk → LLM → checkpoint
3. Mixed mode ×3 k's: question + top-k → LLM → checkpoint

Rate-limited (20 req/min). Retries on 429/500/timeout.

### Phase C — Aggregation
1. Compute MIRAGE metrics (NV, CA, CI, CM) per k
2. Aggregate generation metrics per mode
3. Write final JSON artifacts

---

## Output Artifacts

```
runs/e1/<timestamp>_<mode>/
├── config.json
├── retrieval_metrics.json
├── generation_metrics.json
├── mirage_metrics.json
├── sysinfo.json
├── samples/
│   ├── e1_base.jsonl
│   ├── e1_oracle.jsonl
│   ├── e1_mixed_k3.jsonl
│   ├── e1_mixed_k5.jsonl
│   └── e1_mixed_k10.jsonl
└── retrieval/
    ├── retrieved_k3.jsonl
    ├── retrieved_k5.jsonl
    └── retrieved_k10.jsonl
```

---

## Files

### Created
| File | Purpose |
|------|---------|
| `src/mirage_loader.py` | Load MIRAGE JSONs, build lookups, subset selection |
| `src/metrics.py` | EM, F1, retrieval metrics, MIRAGE NV/CA/CI/CM |
| `run_e1.py` | Experiment runner (Phase A/B/C) |

### Modified
| File | Changes |
|------|---------|
| `docs/CLAUDE.md` | Fixed paths (src/), model, no Ollama |
| `src/config.py` | Model → gemma-3-4b-it:free, MIRAGE paths, rate limit |
| `src/retrieve.py` | Added `index_mirage_pool()` |
| `src/generate.py` | Added base/oracle/mixed prompts, `call_openrouter()` with rate limiting |
| `.gitignore` | Added `runs/` |

### Unchanged
app.py, src/ingest.py, src/chunking.py, src/logger.py, test_pipeline.py

---

## Retrieval Results (Partial Run — 100 questions)

First retrieval test completed successfully:

| k | Recall | Precision | nDCG | MRR |
|---|--------|-----------|------|-----|
| 3 | 0.9400 | 0.4267 | 0.8441 | 0.8200 |
| 5 | 0.9800 | 0.3320 | 0.8449 | 0.8290 |
| 10 | 1.0000 | 0.3000 | 0.7780 | 0.8319 |

Index: 500 chunks in 20.6s. Retrieval: 100 queries in 23.0s.
