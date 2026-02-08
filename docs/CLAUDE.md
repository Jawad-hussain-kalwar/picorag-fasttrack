# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What

PicoRAG FastTrack — lightweight RAG framework for resource-constrained devices (thesis project).
Evaluated against MIRAGE benchmark (7,560 QA pairs, 37,800 chunks).

**Stack:** Python 3.10 | ChromaDB embedded | OpenRouter API | REPL CLI

## CRITICAL — Read Before Doing Anything

- **ALWAYS use `.venv` Python** for running scripts, installing packages, and building anything: `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux/Mac). NEVER use system Python.
- **ALWAYS exclude `.venv/` when searching code.** Use `--glob '!.venv/**'` with Grep, exclude `.venv` path with Glob. Also exclude `MIRAGE/`, `chroma_data/`, `__pycache__/`. Searching without exclusions will drown results in thousands of irrelevant site-packages matches.

## Architecture Constraints — MUST FOLLOW

- **NO PyTorch, NO sentence-transformers** — ChromaDB's built-in ONNX embeddings only (all-MiniLM-L6-v2)
- **NO Docker/containers** — pip-installable only
- **NO cloud vector DBs** — ChromaDB embedded mode with `PersistentClient` only
- **NO OpenAI SDK** — OpenRouter API via `httpx` for LLM calls
- **LLM tier:** 4B-Q8/FP8 or 2B-FP16 class models from OpenRouter
- **Default model:** configured in `config.py` (`OPENROUTER_MODEL`)
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

# Python version: 3.10.11
```

## Architecture

The pipeline flows: `app.py` → `pipeline.py` → `retrieve.py` + `generate.py`

- **`app.py`** — REPL CLI entry point. Parses args, runs interactive loop or single query.
- **`pipeline.py`** — Orchestrator. Calls `ensure_indexed()` → `search()` → `generate_answer()`. Returns `{question, answer, contexts}` dict.
- **`config.py`** — All settings as module-level constants. Hand-rolled `.env` loader (no python-dotenv). OpenRouter config, paths, TOP_K.
- **`ingest.py`** — Loads `.txt` files from `data/` directory. Returns `list[tuple[path, text]]`.
- **`chunking.py`** — `naive_chunk()`: splits text on `\n\n`, generates SHA1 chunk IDs from `source:index:content`.
- **`retrieve.py`** — ChromaDB operations: `get_client()`, `get_collection()` (cosine HNSW), `upsert_chunks()`, `ensure_indexed()`, `search()`. ChromaDB handles embedding automatically on `add()` and `query()`.
- **`generate.py`** — Builds chat messages and calls OpenRouter `/chat/completions` via `httpx`. Requires `OPENROUTER_API_KEY` env var.
- **`logger.py`** — Custom `RAGLogger` with colorama-based event styling and `timer()` context manager. Configure via `LOG_LEVEL` and `LOG_COLOR` env vars.
- **`test_pipeline.py`** — unittest-based. Tests chunking, ChromaDB persistence (real instances in temp dirs), prompt building, and API error handling.

## Searching the Codebase — IMPORTANT

When using Grep, Glob, or any file search: **ALWAYS exclude `.venv/`, `MIRAGE/`, `chroma_data/`, `__pycache__/`** to avoid noise from site-packages and external code. Example Grep patterns:
- `--glob '!.venv/**'` or `path` scoped to project source files only
- For Glob: search specific patterns like `*.py` in the project root, not recursively into `.venv`

Failing to exclude these dirs will return thousands of irrelevant matches and waste time.

## Code Style

- Python 3.10 — use type hints (builtin generics: `list[str]`, `dict[str, Any]`)
- No classes unless necessary — prefer functions and simple data structures
- Minimal dependencies — every new pip install needs justification
- Config via `config.py` constants, secrets via `.env` (never commit `.env`)
- IMPORTANT: No mocking core DB connections in tests — use real ChromaDB instances

## Secrets

`OPENROUTER_API_KEY` must be set in `.env` or environment. The custom loader in `config.py` reads `.env` at import time.

## Experiments (E1–E5)

Progressive build-up evaluating retrieval + generation on MIRAGE:
- **E1:** Vanilla vector-only RAG baseline
- **E2:** Hybrid retrieval (BM25 + vector) → selects "Local-Best"
- **E3:** Selective answering (abstention gate)
- **E4:** Local vs online comparison (Local-Best vs Gemini 2.5 Flash)
- **E5:** Agentic multi-hop RAG with query reformulation

See `experiments.md` for full experimental design, metrics, and YAML config schema.

## Key References

- **`MIRAGE/`** — upstream benchmark repo (cloned). Not our code — reference only.
- **`experiments.md`** — full experimental design document
- **Dataset:** `nlpai-lab/mirage` on HuggingFace

## Key Decisions

- **ChromaDB 1.4.1** — pre-built wheel, ONNX embeddings built-in
- **Python 3.10.11** — proven ChromaDB compatibility
- **OpenRouter over Ollama** — Ollama too resource-heavy for dev phase
- **No python-dotenv** — hand-rolled loader in `config.py`

## Git

- Branch: `main`
- Commit messages: imperative mood, concise
- Do NOT commit: `.venv/`, `chroma_data/`, `.env`, `__pycache__/`
