# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What

PicoRAG FastTrack — lightweight RAG framework for resource-constrained devices (thesis project).
Evaluated against MIRAGE benchmark (7,560 QA pairs, 37,800 chunks).

**Stack:** Python 3.10 | ChromaDB embedded | OpenRouter API | REPL CLI

## CRITICAL — Read Before Doing Anything

- **ALWAYS use `.venv` Python** for running scripts, installing packages, and building anything: `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux/Mac). NEVER use system Python.
- **ALWAYS exclude `.venv/` when searching code.** Use `--glob '!.venv/**'` with Grep, exclude `.venv` path with Glob. Also exclude `MIRAGE/`, `chroma_data/`, `__pycache__/`, `runs/`. Searching without exclusions will drown results in thousands of irrelevant site-packages matches.

## Architecture Constraints — MUST FOLLOW

- **NO PyTorch, NO sentence-transformers** — ChromaDB's built-in ONNX embeddings only (all-MiniLM-L6-v2)
- **NO Docker/containers** — pip-installable only
- **NO cloud vector DBs** — ChromaDB embedded mode with `PersistentClient` only
- **NO OpenAI SDK** — OpenRouter API via `httpx` for LLM calls
- **NO Ollama** — all LLM calls go through OpenRouter API, never local Ollama
- **LLM tier:** 4B-class models from OpenRouter (currently `google/gemma-3-4b-it:free`)
- **Default model:** `google/gemma-3-4b-it:free` — configured in `src/config.py` (`OPENROUTER_MODEL`)
- **Embedding model:** ChromaDB default (all-MiniLM-L6-v2 via ONNX Runtime) — do NOT swap
- **Vector store:** ChromaDB `PersistentClient(path=..., settings=Settings(anonymized_telemetry=False))`

## Commands

**REMINDER: ALWAYS use `.venv\Scripts\python.exe` — never bare `python` or `py`.**

```bash
# Activate venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Install a dependency (ALWAYS through .venv pip)
.venv\Scripts\python.exe -m pip install <package>

# Run tests (real ChromaDB instances, no mocking DB connections)
.venv\Scripts\python.exe -m unittest test_pipeline.py

# Run single test class
.venv\Scripts\python.exe -m unittest test_pipeline.TestChunking

# Run app (REPL mode)
.venv\Scripts\python.exe app.py

# Run app (single query)
.venv\Scripts\python.exe app.py "What is photosynthesis?"

# Run E1 experiment (partial — 100 questions, for dev/testing)
.venv\Scripts\python.exe run_e1.py --partial

# Run E1 experiment (full — 7,560 questions)
.venv\Scripts\python.exe run_e1.py --full

# Resume interrupted E1 run
.venv\Scripts\python.exe run_e1.py --full --resume

# Python version: 3.10.11
```

## Architecture

Source code lives in `src/`. Entry points live at project root.

### Interactive Pipeline
`app.py` → `src/pipeline.py` → `src/retrieve.py` + `src/generate.py`

- **`app.py`** — REPL CLI entry point. Parses args, runs interactive loop or single query.
- **`src/pipeline.py`** — Orchestrator. Calls `ensure_indexed()` → `search()` → `generate_answer()`. Returns `{question, answer, contexts}` dict.
- **`src/config.py`** — All settings as module-level constants. Hand-rolled `.env` loader (no python-dotenv). OpenRouter config, paths, TOP_K.
- **`src/ingest.py`** — Loads `.txt` files from `data/` directory. Returns `list[tuple[path, text]]`.
- **`src/chunking.py`** — `naive_chunk()`: splits text on `\n\n`, generates SHA1 chunk IDs from `source:index:content`.
- **`src/retrieve.py`** — ChromaDB operations: `get_client()`, `get_collection()` (cosine HNSW), `upsert_chunks()`, `ensure_indexed()`, `search()`, `index_mirage_pool()`. ChromaDB handles embedding automatically on `add()` and `query()`.
- **`src/generate.py`** — Builds chat messages and calls OpenRouter `/chat/completions` via `httpx`. Supports base/oracle/mixed prompt modes for MIRAGE evaluation. Requires `OPENROUTER_API_KEY` env var.
- **`src/logger.py`** — Custom `RAGLogger` with colorama-based event styling and `timer()` context manager. Configure via `LOG_LEVEL` and `LOG_COLOR` env vars.

### Experiment Infrastructure
- **`run_e1.py`** — E1 experiment runner. Orchestrates retrieval evaluation + generation evaluation on MIRAGE. Supports `--partial` (100 Qs) and `--full` (7,560 Qs) modes with checkpointing.
- **`src/mirage_loader.py`** — Loads MIRAGE JSON files from `MIRAGE/mirage/` (dataset, doc_pool, oracle). Builds lookups and gold labels.
- **`src/metrics.py`** — Evaluation metrics: EM_loose, EM_strict, F1, Recall@k, Precision@k, nDCG@k, MRR, MIRAGE RAG metrics (NV, CA, CI, CM).

### Tests
- **`test_pipeline.py`** — unittest-based. Tests chunking, ChromaDB persistence (real instances in temp dirs), prompt building, and API error handling.

## Searching the Codebase — IMPORTANT

When using Grep, Glob, or any file search: **ALWAYS exclude `.venv/`, `MIRAGE/`, `chroma_data/`, `runs/`, `__pycache__/`** to avoid noise from site-packages and external code. Example Grep patterns:
- `--glob '!.venv/**'` or `path` scoped to project source files only
- For Glob: search specific patterns like `*.py` in the project root, not recursively into `.venv`

Failing to exclude these dirs will return thousands of irrelevant matches and waste time.

## Code Style

- Python 3.10 — use type hints (builtin generics: `list[str]`, `dict[str, Any]`)
- No classes unless necessary — prefer functions and simple data structures
- Minimal dependencies — every new pip install needs justification
- Config via `src/config.py` constants, secrets via `.env` (never commit `.env`)
- IMPORTANT: No mocking core DB connections in tests — use real ChromaDB instances

## Secrets

`OPENROUTER_API_KEY` must be set in `.env` or environment. The custom loader in `src/config.py` reads `.env` at import time.

## Experiments (E1–E5)

Progressive build-up evaluating retrieval + generation on MIRAGE:
- **E1:** Vanilla vector-only RAG baseline (3 modes: Base/Oracle/Mixed, k ∈ {3,5,10})
- **E2:** Hybrid retrieval (BM25 + vector) → selects "Local-Best"
- **E3:** Selective answering (abstention gate)
- **E4:** Local vs online comparison (Local-Best vs Gemini 2.5 Flash)
- **E5:** Agentic multi-hop RAG with query reformulation

See `docs/experiments.md` for full experimental design, metrics, and YAML config schema.

## MIRAGE Dataset (Local)

MIRAGE data is already cloned locally — **no HuggingFace download needed**:
- **`MIRAGE/mirage/dataset.json`** — 7,560 QA pairs (query, answer[], query_id, source)
- **`MIRAGE/mirage/doc_pool.json`** — 37,800 chunks (mapped_id, doc_chunk, doc_name, support: 0|1)
- **`MIRAGE/mirage/oracle.json`** — gold context per query_id (best chunk with support=1)
- **`MIRAGE/`** — upstream benchmark repo. Reference only — do NOT modify.

## Key Decisions

- **ChromaDB 1.4.1** — pre-built wheel, ONNX embeddings built-in
- **Python 3.10.11** — proven ChromaDB compatibility
- **OpenRouter only** — all LLM calls via OpenRouter API, no Ollama, no local inference
- **Model: `google/gemma-3-4b-it:free`** — 4B-class model, fits 4GB VRAM theoretically; using OpenRouter free tier during dev/eval phase
- **No python-dotenv** — hand-rolled loader in `src/config.py`

## Git

- Branch: `main`
- Commit messages: imperative mood, concise
- Do NOT commit: `.venv/`, `chroma_data/`, `.env`, `__pycache__/`, `runs/`
