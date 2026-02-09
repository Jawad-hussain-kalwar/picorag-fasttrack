"""E2 — Hybrid Retrieval Exploration experiment runner.

Compares 6 retrieval configurations across k ∈ {3, 5, 10} on a 500-question
MIRAGE subset (100 questions evaluated).

Usage:
    .venv\\Scripts\\python.exe run_e2.py --partial          # 100 eval questions
    .venv\\Scripts\\python.exe run_e2.py --partial --phase retrieval
    .venv\\Scripts\\python.exe run_e2.py --partial --resume
"""

import argparse
import json
import os
import platform
import threading
import time
import traceback
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx

from src.bm25 import bm25_search, build_bm25_index
from src.config import (
    CHROMA_PERSIST_DIR,
    E2_EVAL_N,
    E2_INDEX_N,
    E2_K_VALUES,
    E2_RERANK_TOP_N,
    E2_RRF_K,
    JUDGE_MODEL,
    JUDGE_RATE_LIMIT,
    MIRAGE_COLLECTION_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBED_MODEL,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_TEMPERATURE,
    OPENROUTER_TIMEOUT_SECONDS,
    RUNS_DIR,
    VOYAGE_API_KEY,
    VOYAGE_RERANK_MODEL,
)
from src.embed_openrouter import (
    index_mirage_with_custom_embeddings,
    search_with_custom_embeddings,
)
from src.generate import (
    build_base_prompt,
    build_mixed_prompt_cited,
    build_oracle_prompt,
    call_openrouter,
)
from src.hybrid import rrf_fuse
from src.metrics import (
    citation_precision,
    citation_recall,
    compute_mirage_metrics,
    em_loose,
    mrr,
    ndcg_at_k,
    parse_citations,
    precision_at_k,
    recall_at_k,
)
from src.mirage_loader import (
    build_gold_lookup,
    load_dataset,
    load_doc_pool,
    load_oracle,
    select_partial_subset,
)
from src.rerank import call_voyage_rerank
from src.retrieve import get_client, get_collection, index_mirage_pool, search


# ---------------------------------------------------------------------------
# Helpers (shared with run_e1.py pattern)
# ---------------------------------------------------------------------------

def _jsonl_path(run_dir: Path, name: str) -> Path:
    return run_dir / "samples" / f"{name}.jsonl"


def _load_checkpoint(path: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                results[record["query_id"]] = record
    return results


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) from a list of values."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


# ---------------------------------------------------------------------------
# Judge constants (Phase D — inline, no import from run_judge.py)
# ---------------------------------------------------------------------------

JUDGE_MAX_TOKENS = 1024
JUDGE_TEMPERATURE = 0.0

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_judgment",
        "description": "Submit evaluation scores for a RAG answer",
        "parameters": {
            "type": "object",
            "properties": {
                "faithfulness": {
                    "type": "number",
                    "description": "Ratio of supported claims to total claims (0.0-1.0)",
                },
                "groundedness": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "0=not grounded, 1=partially, 2=fully grounded",
                },
                "answer_relevance": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "0=irrelevant, 1=partially relevant, 2=fully relevant to the question",
                },
                "semantic_correctness": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "0=incorrect, 1=partially correct, 2=correct compared to gold answers",
                },
            },
            "required": ["faithfulness", "groundedness", "answer_relevance", "semantic_correctness"],
        },
    },
}

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator. You MUST respond by calling the "
    "submit_judgment tool. Do NOT write any text — ONLY call the tool.\n\n"
    "Your job is to evaluate an Answer against Retrieved Contexts and Gold Answers.\n\n"
    "RULES:\n"
    "- For faithfulness/groundedness: ONLY judge based on Retrieved Contexts.\n"
    "- For answer_relevance: judge whether the Answer addresses the Question.\n"
    "- For semantic_correctness: compare the Answer to the Gold Answers.\n"
    "- You MUST use the submit_judgment tool. Do NOT write a text reply."
)

JUDGE_USER_TEMPLATE = """\
Evaluate the following Answer to a Question.

=== QUESTION ===
{question}

=== GOLD ANSWERS ===
{gold_answers_block}

=== RETRIEVED CONTEXTS ===
{contexts_block}

=== ANSWER TO EVALUATE ===
{prediction}

=== EVALUATION INSTRUCTIONS ===

STEP 1 — FAITHFULNESS (float 0.0-1.0):
- Identify every factual claim in the Answer.
- Count how many are supported by the Retrieved Contexts.
- faithfulness = supported_claims / total_claims.
- If no factual claims or Answer is "I don't know"/error/empty: faithfulness = 1.0.

STEP 2 — GROUNDEDNESS (int 0-2):
- 0 = NOT grounded — Answer has substantial info NOT in Retrieved Contexts or contradicts them.
- 1 = PARTIALLY grounded — some info from contexts, some external.
- 2 = FULLY grounded — everything traceable to Retrieved Contexts.

STEP 3 — ANSWER RELEVANCE (int 0-2):
- 0 = IRRELEVANT — Answer does not address the Question at all.
- 1 = PARTIALLY relevant — Answer partially addresses the Question.
- 2 = FULLY relevant — Answer directly and completely addresses the Question.

STEP 4 — SEMANTIC CORRECTNESS (int 0-2):
- Compare the Answer to the Gold Answers.
- 0 = INCORRECT — Answer contradicts or is unrelated to Gold Answers.
- 1 = PARTIALLY correct — Answer overlaps but misses key info or adds errors.
- 2 = CORRECT — Answer conveys the same meaning as one of the Gold Answers.

Call submit_judgment now with all four scores."""


_judge_last_call_time: float = 0.0
_judge_lock = threading.Lock()


def call_judge(
    question: str,
    contexts: list[str],
    prediction: str,
    gold_answers: list[str],
    max_retries: int = 3,
) -> dict:
    """Call the judge model via OpenRouter tool-calling. Returns 4 scores."""
    global _judge_last_call_time

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    contexts_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    gold_answers_block = "\n".join(f"- {a}" for a in gold_answers)

    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        gold_answers_block=gold_answers_block,
        contexts_block=contexts_block,
        prediction=prediction,
    )
    merged_content = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_msg}"

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": merged_content}],
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
        "tools": [SUBMIT_TOOL],
        "tool_choice": "auto",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    min_interval = 60.0 / JUDGE_RATE_LIMIT

    for attempt in range(1, max_retries + 1):
        with _judge_lock:
            elapsed = time.monotonic() - _judge_last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            _judge_last_call_time = time.monotonic()
        try:
            with httpx.Client(timeout=OPENROUTER_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )

            if response.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"    Judge rate limited, waiting {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"    Judge API error {response.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                return _null_judge_result(response.text)

            body = response.json()
            return _parse_judge_response(body)

        except httpx.TimeoutException:
            if attempt < max_retries:
                print(f"    Judge timeout, retrying ({attempt}/{max_retries})")
                time.sleep(2 ** attempt)
                continue
            return _null_judge_result("TIMEOUT")

        except httpx.HTTPError as exc:
            if attempt < max_retries:
                print(f"    Judge HTTP error ({type(exc).__name__}), retrying ({attempt}/{max_retries})")
                time.sleep(2 ** attempt)
                continue
            return _null_judge_result(f"HTTP_ERROR: {type(exc).__name__}: {exc}")

    return _null_judge_result("MAX_RETRIES")


def _null_judge_result(raw: str) -> dict:
    return {
        "faithfulness": None, "groundedness": None,
        "answer_relevance": None, "semantic_correctness": None,
        "raw": raw,
    }


def _parse_judge_response(body: dict) -> dict:
    """Extract 4 judge scores from tool-call response."""
    raw = json.dumps(body, ensure_ascii=False)
    try:
        choices = body.get("choices", [])
        if not choices:
            return _null_judge_result(raw)

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")

        if tool_calls:
            args_str = tool_calls[0]["function"]["arguments"]
            args = json.loads(args_str)

            faith = args.get("faithfulness")
            if faith is not None:
                faith = max(0.0, min(1.0, float(faith)))

            ground = args.get("groundedness")
            if ground is not None:
                ground = max(0, min(2, int(ground)))

            relevance = args.get("answer_relevance")
            if relevance is not None:
                relevance = max(0, min(2, int(relevance)))

            correctness = args.get("semantic_correctness")
            if correctness is not None:
                correctness = max(0, min(2, int(correctness)))

            return {
                "faithfulness": faith,
                "groundedness": ground,
                "answer_relevance": relevance,
                "semantic_correctness": correctness,
                "raw": args_str,
            }

        # Fallback: text instead of tool call
        content = message.get("content", "")
        return _null_judge_result(content)

    except (KeyError, IndexError, json.JSONDecodeError, TypeError, ValueError):
        return _null_judge_result(raw)


# ---------------------------------------------------------------------------
# Retrieval config definitions
# ---------------------------------------------------------------------------

CONFIGS: list[dict[str, str | None]] = [
    {"id": "1_vector_minilm", "label": "Vector-only (MiniLM)", "method": "vector", "embed": "minilm"},
    {"id": "2_bm25", "label": "BM25-only", "method": "bm25", "embed": None},
    {"id": "3_hybrid_minilm", "label": "Hybrid RRF (MiniLM)", "method": "hybrid", "embed": "minilm"},
    {"id": "4_hybrid_rerank_minilm", "label": "Hybrid + Reranker (MiniLM)", "method": "hybrid+rerank", "embed": "minilm"},
    {"id": "5_vector_qwen3", "label": "Vector-only (Qwen3)", "method": "vector", "embed": "qwen3"},
    {"id": "6_hybrid_qwen3", "label": "Hybrid RRF (Qwen3)", "method": "hybrid", "embed": "qwen3"},
]


# ---------------------------------------------------------------------------
# Phase A: Retrieval evaluation (all 6 configs)
# ---------------------------------------------------------------------------

def _run_vector_search_minilm(collection, query: str, n: int) -> dict:
    return search(collection, query, n_results=n)


def _run_vector_search_qwen3(query: str, n: int, collection_name: str) -> dict:
    return search_with_custom_embeddings(collection_name, query, n_results=n)


def run_retrieval_phase(
    eval_dataset: list[dict],
    index_doc_pool: list[dict],
    gold_lookup: dict[str, list[int]],
    run_dir: Path,
    k_values: list[int],
) -> dict[str, dict[int, list[dict]]]:
    """Run retrieval for all 6 configs. Returns {config_id: {k: [per-query results]}}."""
    print(f"\n{'='*60}")
    print("PHASE A: Retrieval Evaluation (6 configs)")
    print(f"{'='*60}")

    max_k = max(k_values)
    fetch_n = max(max_k, E2_RERANK_TOP_N)  # Need top-25 for reranking

    # --- Prepare indices ---

    # 1. MiniLM ChromaDB collection
    collection_name = f"{MIRAGE_COLLECTION_NAME}_e2"
    client = get_client(str(CHROMA_PERSIST_DIR))
    minilm_collection = get_collection(client, collection_name)

    t0 = time.perf_counter()
    indexed = index_mirage_pool(minilm_collection, index_doc_pool)
    minilm_index_time = time.perf_counter() - t0
    print(f"MiniLM ChromaDB: {indexed} chunks indexed in {minilm_index_time:.1f}s")

    # 2. BM25 index
    t0 = time.perf_counter()
    bm25_index, bm25_pool = build_bm25_index(index_doc_pool)
    bm25_index_time = time.perf_counter() - t0
    print(f"BM25: {len(index_doc_pool)} chunks indexed in {bm25_index_time:.1f}s")

    # 3. Qwen3 ChromaDB collection (via OpenRouter embeddings API)
    qwen3_collection_name = f"{MIRAGE_COLLECTION_NAME}_e2_qwen3"
    t0 = time.perf_counter()
    try:
        qwen3_indexed = index_mirage_with_custom_embeddings(
            index_doc_pool, qwen3_collection_name, batch_size=50,
        )
        qwen3_index_time = time.perf_counter() - t0
        print(f"Qwen3 ChromaDB: {qwen3_indexed} chunks indexed in {qwen3_index_time:.1f}s")
        qwen3_available = True
    except RuntimeError as exc:
        qwen3_index_time = time.perf_counter() - t0
        print(f"Qwen3 indexing failed: {exc}")
        print("Configs 5 and 6 will be skipped.")
        qwen3_available = False

    # --- Run retrieval per config ---
    all_results: dict[str, dict[int, list[dict]]] = {}

    for cfg in CONFIGS:
        cfg_id = cfg["id"]
        cfg_label = cfg["label"]
        assert isinstance(cfg_id, str)

        # Skip Qwen3 configs if unavailable
        if cfg["embed"] == "qwen3" and not qwen3_available:
            print(f"\n  Skipping {cfg_label} (Qwen3 unavailable)")
            continue

        # Skip rerank if no Voyage key
        if cfg["method"] == "hybrid+rerank" and not VOYAGE_API_KEY:
            print(f"\n  Skipping {cfg_label} (VOYAGE_API_KEY not set)")
            continue

        # Resume: skip if all k files already exist on disk
        all_k_exist = all(
            (run_dir / "retrieval" / cfg_id / f"retrieved_k{k}.jsonl").exists()
            for k in k_values
        )
        if all_k_exist:
            print(f"\n  Config: {cfg_label} — loading from disk (resume)")
            resumed_results: dict[int, list[dict]] = {}
            for k in k_values:
                ret_path = run_dir / "retrieval" / cfg_id / f"retrieved_k{k}.jsonl"
                records: list[dict] = []
                for line in ret_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        records.append(json.loads(line))
                resumed_results[k] = records
                r_vals = [r["recall"] for r in records]
                print(f"    k={k}: Recall={_mean(r_vals):.4f} ({len(records)} queries loaded)")
            all_results[cfg_id] = resumed_results
            continue

        print(f"\n  Config: {cfg_label}")
        t0 = time.perf_counter()

        raw_results: list[dict] = []
        for i, question in enumerate(eval_dataset):
            qid = question["query_id"]
            query_text = question["query"]

            # Get raw results based on method
            if cfg["method"] == "vector" and cfg["embed"] == "minilm":
                result = _run_vector_search_minilm(minilm_collection, query_text, fetch_n)

            elif cfg["method"] == "bm25":
                result = bm25_search(bm25_index, bm25_pool, query_text, n_results=fetch_n)

            elif cfg["method"] == "hybrid" and cfg["embed"] == "minilm":
                vec_result = _run_vector_search_minilm(minilm_collection, query_text, fetch_n)
                bm25_result = bm25_search(bm25_index, bm25_pool, query_text, n_results=fetch_n)
                result = rrf_fuse(vec_result, bm25_result, k=fetch_n, rrf_k=E2_RRF_K)

            elif cfg["method"] == "hybrid+rerank" and cfg["embed"] == "minilm":
                vec_result = _run_vector_search_minilm(minilm_collection, query_text, fetch_n)
                bm25_result = bm25_search(bm25_index, bm25_pool, query_text, n_results=fetch_n)
                fused = rrf_fuse(vec_result, bm25_result, k=E2_RERANK_TOP_N, rrf_k=E2_RRF_K)
                # Rerank the top-25 fused results
                result = call_voyage_rerank(
                    query=query_text,
                    documents=fused["documents"][0],
                    top_k=fetch_n,
                    metadatas=fused["metadatas"][0],
                )

            elif cfg["method"] == "vector" and cfg["embed"] == "qwen3":
                result = _run_vector_search_qwen3(query_text, fetch_n, qwen3_collection_name)

            elif cfg["method"] == "hybrid" and cfg["embed"] == "qwen3":
                vec_result = _run_vector_search_qwen3(query_text, fetch_n, qwen3_collection_name)
                bm25_result = bm25_search(bm25_index, bm25_pool, query_text, n_results=fetch_n)
                result = rrf_fuse(vec_result, bm25_result, k=fetch_n, rrf_k=E2_RRF_K)

            else:
                raise ValueError(f"Unknown config: {cfg}")

            metadatas = result.get("metadatas", [[]])[0]
            documents = result.get("documents", [[]])[0]
            distances = result.get("distances", [[]])[0]
            supports = [m.get("support", 0) for m in metadatas]

            raw_results.append({
                "query_id": qid,
                "metadatas": metadatas,
                "documents": documents,
                "distances": distances,
                "supports": supports,
            })

            if (i + 1) % 50 == 0 or i + 1 == len(eval_dataset):
                print(f"    Retrieved {i + 1}/{len(eval_dataset)}")

        retrieval_time = time.perf_counter() - t0
        print(f"    Done in {retrieval_time:.1f}s")

        # Compute metrics per k
        cfg_results: dict[int, list[dict]] = {}
        for k in k_values:
            k_results = []
            recalls, precisions, ndcgs, mrrs = [], [], [], []

            for raw, question in zip(raw_results, eval_dataset):
                supports_k = raw["supports"][:k]
                docs_k = raw["documents"][:k]
                dists_k = raw["distances"][:k]
                metas_k = raw["metadatas"][:k]

                r = recall_at_k(supports_k)
                p = precision_at_k(supports_k)
                n = ndcg_at_k(supports_k, k)
                m = mrr(supports_k)

                recalls.append(r)
                precisions.append(p)
                ndcgs.append(n)
                mrrs.append(m)

                k_results.append({
                    "query_id": raw["query_id"],
                    "documents": docs_k,
                    "distances": dists_k,
                    "supports": supports_k,
                    "metadatas": metas_k,
                    "recall": r,
                    "precision": p,
                    "ndcg": n,
                    "mrr": m,
                })

            cfg_results[k] = k_results

            print(f"    k={k}: Recall={_mean(recalls):.4f}  "
                  f"Precision={_mean(precisions):.4f}  "
                  f"nDCG={_mean(ndcgs):.4f}  "
                  f"MRR={_mean(mrrs):.4f}")

            # Save per-query retrieval results
            ret_path = run_dir / "retrieval" / cfg_id / f"retrieved_k{k}.jsonl"
            ret_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ret_path, "w", encoding="utf-8") as f:
                for rec in k_results:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        all_results[cfg_id] = cfg_results

    # Save retrieval metrics summary
    retrieval_summary: dict = {}
    for cfg_id, cfg_results in all_results.items():
        cfg_metrics: dict = {}
        for k in k_values:
            results = cfg_results[k]
            cfg_metrics[f"k={k}"] = {
                f"Recall@{k}": round(_mean([r["recall"] for r in results]), 4),
                f"Precision@{k}": round(_mean([r["precision"] for r in results]), 4),
                f"nDCG@{k}": round(_mean([r["ndcg"] for r in results]), 4),
                f"MRR@{k}": round(_mean([r["mrr"] for r in results]), 4),
            }
        retrieval_summary[cfg_id] = cfg_metrics

    retrieval_summary["_meta"] = {
        "minilm_index_time_s": round(minilm_index_time, 2),
        "bm25_index_time_s": round(bm25_index_time, 2),
        "qwen3_index_time_s": round(qwen3_index_time, 2) if qwen3_available else None,
        "total_chunks_indexed": len(index_doc_pool),
        "total_eval_questions": len(eval_dataset),
        "configs_run": list(all_results.keys()),
    }
    _save_json(run_dir / "retrieval_metrics.json", retrieval_summary)

    del client
    return all_results


# ---------------------------------------------------------------------------
# Phase B: Generation evaluation
# ---------------------------------------------------------------------------

def _run_generation_mode(
    name: str,
    dataset: list[dict],
    oracle: dict[str, dict],
    retrieval_results: list[dict] | None,
    run_dir: Path,
    mode: str,
    use_citations: bool = False,
) -> list[dict]:
    """Run generation for a single mode. Returns per-query results."""
    checkpoint_path = _jsonl_path(run_dir, name)
    completed = _load_checkpoint(checkpoint_path)
    total = len(dataset)
    skipped = len(completed)

    if skipped >= total:
        print(f"  {name}: already complete ({total}/{total})")
        return list(completed.values())

    if skipped > 0:
        print(f"  {name}: resuming from {skipped}/{total}")
    else:
        print(f"  {name}: starting {total} queries")

    results = list(completed.values())
    errors = 0

    for i, question in enumerate(dataset):
        qid = question["query_id"]
        if qid in completed:
            continue

        if mode == "base":
            messages = build_base_prompt(question["query"])
        elif mode == "oracle":
            chunk_text = oracle[qid]["doc_chunk"]
            messages = build_oracle_prompt(question["query"], chunk_text)
        elif mode == "mixed":
            assert retrieval_results is not None
            chunks = retrieval_results[i]["documents"]
            if use_citations:
                messages = build_mixed_prompt_cited(question["query"], chunks)
            else:
                from src.generate import build_mixed_prompt
                messages = build_mixed_prompt(question["query"], chunks)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        t0 = time.perf_counter()
        try:
            prediction = call_openrouter(messages)
        except RuntimeError as exc:
            errors += 1
            prediction = f"ERROR: {exc}"
        gen_ms = int((time.perf_counter() - t0) * 1000)

        answers = question["answer"]
        record = {
            "query_id": qid,
            "query": question["query"],
            "gold_answers": answers,
            "prediction": prediction,
            "em_loose": em_loose(prediction, answers),
            "generation_ms": gen_ms,
            "mode": mode,
        }

        # Citation metrics for mixed mode with citations
        if mode == "mixed" and use_citations and retrieval_results:
            supports = retrieval_results[i].get("supports", [])
            gold_indices = [j + 1 for j, s in enumerate(supports) if s == 1]
            cited = parse_citations(prediction)
            record["cited_indices"] = cited
            record["gold_indices"] = gold_indices
            record["citation_precision"] = citation_precision(cited, gold_indices)
            record["citation_recall"] = citation_recall(cited, gold_indices)

        _append_jsonl(checkpoint_path, record)
        results.append(record)
        completed[qid] = record

        done = len(completed)
        if done % 50 == 0 or done == total:
            em_so_far = _mean([r["em_loose"] for r in results])
            print(f"    {name}: {done}/{total}  EM_loose={em_so_far:.4f}  errors={errors}")

    return results


def run_generation_phase(
    eval_dataset: list[dict],
    oracle: dict[str, dict],
    all_retrieval: dict[str, dict[int, list[dict]]],
    run_dir: Path,
    k_values: list[int],
) -> dict[str, list[dict]]:
    """Run generation for base, oracle, and mixed per config per k (parallel)."""
    print(f"\n{'='*60}")
    print("PHASE B: Generation Evaluation (parallel)")
    print(f"{'='*60}")

    # Build task list: (result_key, name, retrieval_results, mode, use_citations)
    tasks: list[tuple[str, str, list[dict] | None, str, bool]] = []
    tasks.append(("base", "e2_base", None, "base", False))
    tasks.append(("oracle", "e2_oracle", None, "oracle", False))
    for cfg_id, cfg_results in all_retrieval.items():
        for k in k_values:
            gen_name = f"e2_mixed_{cfg_id}_k{k}"
            tasks.append((gen_name, gen_name, cfg_results[k], "mixed", True))

    print(f"  {len(tasks)} generation tasks, 8 workers")

    all_gen: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for key, name, ret_results, mode, use_cit in tasks:
            future = executor.submit(
                _run_generation_mode,
                name, eval_dataset, oracle, ret_results, run_dir, mode, use_cit,
            )
            futures[future] = key

        for future in as_completed(futures):
            key = futures[future]
            try:
                all_gen[key] = future.result()
            except Exception as exc:
                print(f"  ERROR in {key}: {exc}")

    return all_gen


# ---------------------------------------------------------------------------
# Phase C: Aggregation
# ---------------------------------------------------------------------------

def run_aggregation_phase(
    gen_results: dict[str, list[dict]],
    all_retrieval: dict[str, dict[int, list[dict]]],
    k_values: list[int],
    run_dir: Path,
) -> None:
    """Aggregate per-query results into final metrics."""
    print(f"\n{'='*60}")
    print("PHASE C: Aggregation")
    print(f"{'='*60}")

    generation_metrics: dict = {}
    mirage_metrics: dict = {}

    # Generation metrics per mode
    for mode_name, results in gen_results.items():
        em_l = _mean([r["em_loose"] for r in results])
        lat = _mean([r["generation_ms"] for r in results])

        lat_values = [r["generation_ms"] for r in results]
        entry: dict = {
            "EM_loose": round(em_l, 4),
            "avg_generation_ms": round(lat, 1),
            "latency_p50_ms": round(_percentile(lat_values, 50), 1),
            "latency_p95_ms": round(_percentile(lat_values, 95), 1),
            "n_queries": len(results),
        }

        # Citation metrics if available
        cit_p_vals = [r["citation_precision"] for r in results if "citation_precision" in r]
        cit_r_vals = [r["citation_recall"] for r in results if "citation_recall" in r]
        if cit_p_vals:
            entry["citation_precision"] = round(_mean(cit_p_vals), 4)
        if cit_r_vals:
            entry["citation_recall"] = round(_mean(cit_r_vals), 4)

        generation_metrics[mode_name] = entry

        extra = ""
        if cit_p_vals:
            extra = f"  CitP={_mean(cit_p_vals):.4f}  CitR={_mean(cit_r_vals):.4f}"
        print(f"  {mode_name}: EM_loose={em_l:.4f}{extra}")

    _save_json(run_dir / "generation_metrics.json", generation_metrics)

    # MIRAGE metrics per config per k
    base_results = gen_results.get("base", [])
    oracle_results = gen_results.get("oracle", [])
    base_by_qid = {r["query_id"]: r["em_loose"] for r in base_results}
    oracle_by_qid = {r["query_id"]: r["em_loose"] for r in oracle_results}

    for cfg_id in all_retrieval:
        for k in k_values:
            gen_key = f"e2_mixed_{cfg_id}_k{k}"
            mixed_results = gen_results.get(gen_key, [])
            if not mixed_results:
                continue

            base_labels, oracle_labels, mixed_labels = [], [], []
            for r in mixed_results:
                qid = r["query_id"]
                base_labels.append(base_by_qid.get(qid, 0.0))
                oracle_labels.append(oracle_by_qid.get(qid, 0.0))
                mixed_labels.append(r["em_loose"])

            m = compute_mirage_metrics(base_labels, oracle_labels, mixed_labels)
            mirage_key = f"{cfg_id}_k={k}"
            mirage_metrics[mirage_key] = {k_name: round(v, 4) for k_name, v in m.items()}
            print(f"  MIRAGE {mirage_key}: NV={m['NV']:.4f}  CA={m['CA']:.4f}  "
                  f"CI={m['CI']:.4f}  CM={m['CM']:.4f}")

    _save_json(run_dir / "mirage_metrics.json", mirage_metrics)

    # Print comparative retrieval summary table
    print(f"\n{'='*60}")
    print("Retrieval Summary")
    print(f"{'='*60}")
    ret_metrics_path = run_dir / "retrieval_metrics.json"
    if ret_metrics_path.exists():
        ret_summary = json.loads(ret_metrics_path.read_text(encoding="utf-8"))
        for cfg_id in all_retrieval:
            cfg_data = ret_summary.get(cfg_id, {})
            label = next((c["label"] for c in CONFIGS if c["id"] == cfg_id), cfg_id)
            print(f"\n  {label}:")
            for k in k_values:
                kd = cfg_data.get(f"k={k}", {})
                print(f"    k={k}: Recall={kd.get(f'Recall@{k}', 'N/A')}  "
                      f"nDCG={kd.get(f'nDCG@{k}', 'N/A')}  "
                      f"MRR={kd.get(f'MRR@{k}', 'N/A')}")

    # Efficiency block
    efficiency: dict = {}
    if ret_metrics_path.exists():
        ret_summary = json.loads(ret_metrics_path.read_text(encoding="utf-8"))
        meta = ret_summary.get("_meta", {})
        efficiency["minilm_index_time_s"] = meta.get("minilm_index_time_s")
        efficiency["bm25_index_time_s"] = meta.get("bm25_index_time_s")
        efficiency["qwen3_index_time_s"] = meta.get("qwen3_index_time_s")
        efficiency["total_chunks_indexed"] = meta.get("total_chunks_indexed")

    # Peak RAM from tracemalloc
    try:
        _, peak = tracemalloc.get_traced_memory()
        efficiency["peak_ram_mb"] = round(peak / (1024 * 1024), 1)
    except Exception:
        efficiency["peak_ram_mb"] = None

    efficiency["ttft_note"] = "TTFT equals total latency for non-streaming API calls"

    _save_json(run_dir / "efficiency_metrics.json", efficiency)
    print(f"\n  Peak RAM: {efficiency.get('peak_ram_mb', 'N/A')} MB")


# ---------------------------------------------------------------------------
# Phase D: LLM-as-Judge evaluation
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _judge_one_config(cfg_id: str, k: int, run_dir: Path) -> tuple[str, dict] | None:
    """Judge one config/k pair. Returns (judge_key, agg_dict) or None."""
    gen_path = _jsonl_path(run_dir, f"e2_mixed_{cfg_id}_k{k}")
    ret_path = run_dir / "retrieval" / cfg_id / f"retrieved_k{k}.jsonl"

    if not gen_path.exists():
        print(f"  Skipping {cfg_id} k={k}: no generation file")
        return None
    if not ret_path.exists():
        print(f"  Skipping {cfg_id} k={k}: no retrieval file")
        return None

    gen_rows = _load_jsonl(gen_path)
    ret_rows = _load_jsonl(ret_path)
    ret_by_qid = {r["query_id"]: r for r in ret_rows}

    ckpt_path = run_dir / "samples" / f"judge_{cfg_id}_k{k}.jsonl"
    judged = _load_checkpoint(ckpt_path)

    total = len(gen_rows)
    skipped_resume = len(judged)
    if skipped_resume >= total:
        print(f"  {cfg_id} k={k}: already complete ({total}/{total})")
    elif skipped_resume > 0:
        print(f"  {cfg_id} k={k}: resuming from {skipped_resume}/{total}")
    else:
        print(f"  {cfg_id} k={k}: judging {total} predictions")

    errors = 0
    for g in gen_rows:
        qid = g["query_id"]
        if qid in judged:
            continue

        if g["prediction"].startswith("ERROR:"):
            row = {
                "query_id": qid,
                "faithfulness": None, "groundedness": None,
                "answer_relevance": None, "semantic_correctness": None,
                "judge_raw": "SKIPPED_ERROR_PREDICTION",
            }
            _append_jsonl(ckpt_path, row)
            judged[qid] = row
            continue

        try:
            r = ret_by_qid.get(qid)
            contexts = r["documents"] if r else []
            gold_answers = g.get("gold_answers", [])

            result = call_judge(
                question=g["query"],
                contexts=contexts,
                prediction=g["prediction"],
                gold_answers=gold_answers,
            )

            row = {
                "query_id": qid,
                "faithfulness": result["faithfulness"],
                "groundedness": result["groundedness"],
                "answer_relevance": result["answer_relevance"],
                "semantic_correctness": result["semantic_correctness"],
                "judge_raw": result["raw"],
            }
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"    FATAL judge error on qid={qid} ({cfg_id} k={k}): "
                  f"{type(exc).__name__}: {exc}\n{tb}")
            row = {
                "query_id": qid,
                "faithfulness": None, "groundedness": None,
                "answer_relevance": None, "semantic_correctness": None,
                "judge_raw": f"FATAL: {type(exc).__name__}: {exc}",
            }
            errors += 1

        _append_jsonl(ckpt_path, row)
        judged[qid] = row

        if row.get("faithfulness") is None and "FATAL" not in row.get("judge_raw", ""):
            errors += 1

        done = len(judged)
        if done % 10 == 0 or done == total:
            print(f"    {cfg_id} k={k}: {done}/{total} judged ({errors} errors)")

    # Aggregate
    all_rows = list(judged.values())
    faith = [r["faithfulness"] for r in all_rows if r["faithfulness"] is not None]
    ground = [r["groundedness"] for r in all_rows if r["groundedness"] is not None]
    relev = [r["answer_relevance"] for r in all_rows if r["answer_relevance"] is not None]
    correct = [r["semantic_correctness"] for r in all_rows if r["semantic_correctness"] is not None]

    n_judged = len(faith)
    agg = {
        "faithfulness": round(_mean(faith), 4) if faith else None,
        "groundedness": round(_mean(ground) / 2, 4) if ground else None,
        "answer_relevance": round(_mean(relev) / 2, 4) if relev else None,
        "semantic_correctness": round(_mean(correct) / 2, 4) if correct else None,
        "n_judged": n_judged,
        "n_total": len(all_rows),
    }
    judge_key = f"{cfg_id}_k={k}"
    print(f"    {cfg_id} k={k} done — faith={agg['faithfulness']}  "
          f"ground={agg['groundedness']}  relev={agg['answer_relevance']}  "
          f"correct={agg['semantic_correctness']}")
    return (judge_key, agg)


def run_judge_phase(
    all_retrieval: dict[str, dict[int, list[dict]]],
    run_dir: Path,
    k_values: list[int],
) -> dict:
    """Run LLM-as-judge on all mixed-mode generation results (parallel)."""
    print(f"\n{'='*60}")
    print("PHASE D: LLM-as-Judge Evaluation (parallel)")
    print(f"{'='*60}")
    print(f"  Judge model: {JUDGE_MODEL}")
    print(f"  Rate limit: {JUDGE_RATE_LIMIT} RPM")

    judge_tasks = [(cfg_id, k) for cfg_id in all_retrieval for k in k_values]
    print(f"  {len(judge_tasks)} judge tasks, 8 workers")

    judge_metrics: dict = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_judge_one_config, cfg_id, k, run_dir): (cfg_id, k)
            for cfg_id, k in judge_tasks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    key, agg = result
                    judge_metrics[key] = agg
            except Exception as exc:
                cfg_id, k = futures[future]
                tb = traceback.format_exc()
                print(f"  ERROR judging {cfg_id} k={k}: {type(exc).__name__}: {exc}\n{tb}")

    _save_json(run_dir / "judge_metrics.json", judge_metrics)
    return judge_metrics


# ---------------------------------------------------------------------------
# Sysinfo
# ---------------------------------------------------------------------------

def _collect_sysinfo() -> dict:
    import importlib.metadata
    pkgs = {}
    for pkg in ["chromadb", "httpx", "colorama", "rank_bm25"]:
        try:
            pkgs[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pkgs[pkg] = "not installed"
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "packages": pkgs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="E2 — Hybrid Retrieval Exploration")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--partial", action="store_true", help="100 eval questions (dev)")
    group.add_argument("--smoke", type=int, metavar="N", help="Smoke test with N eval questions")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--phase",
        choices=["retrieval", "generation", "judge", "all"],
        default="all",
        help="Run specific phase only (default: all)",
    )
    args = parser.parse_args()

    tracemalloc.start()

    # Load data
    print("Loading MIRAGE dataset...")
    dataset = load_dataset()
    doc_pool = load_doc_pool()
    oracle = load_oracle()

    # Determine subset sizes
    if args.smoke:
        n_index = args.smoke * 5  # 5x eval for retrieval diversity
        n_eval = args.smoke
    else:
        n_index = E2_INDEX_N
        n_eval = E2_EVAL_N

    # Select Q/A subset for indexing
    index_dataset, index_doc_pool, index_oracle = select_partial_subset(
        dataset, doc_pool, oracle, n_index
    )

    # Select first n_eval of those for evaluation
    eval_dataset, _, eval_oracle = select_partial_subset(
        index_dataset, index_doc_pool, index_oracle, n_eval
    )
    # eval uses the full index_doc_pool (all indexed chunks) but only n_eval questions
    eval_oracle = {q["query_id"]: index_oracle[q["query_id"]] for q in eval_dataset if q["query_id"] in index_oracle}

    mode_label = f"smoke_{n_eval}" if args.smoke else f"partial_{E2_EVAL_N}"

    print(f"Mode: {mode_label}")
    print(f"Index questions: {len(index_dataset)} ({len(index_doc_pool)} chunks)")
    print(f"Eval questions: {len(eval_dataset)}")
    print(f"Oracle entries: {len(eval_oracle)}")
    print(f"K values: {E2_K_VALUES}")
    print(f"Model: {OPENROUTER_MODEL}")
    print(f"Embed model: {OPENROUTER_EMBED_MODEL}")
    print(f"Rerank model: {VOYAGE_RERANK_MODEL}")
    print(f"Voyage API key: {'set' if VOYAGE_API_KEY else 'NOT SET'}")

    gold_lookup = build_gold_lookup(index_doc_pool)

    # Setup run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / "e2" / f"{timestamp}_{mode_label}"
    if args.resume:
        e2_dir = RUNS_DIR / "e2"
        if e2_dir.exists():
            candidates = sorted(
                [d for d in e2_dir.iterdir() if d.is_dir() and mode_label in d.name],
                reverse=True,
            )
            if candidates:
                run_dir = candidates[0]
                print(f"Resuming from: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "experiment": "E2",
        "mode": mode_label,
        "model": OPENROUTER_MODEL,
        "embed_model": OPENROUTER_EMBED_MODEL,
        "rerank_model": VOYAGE_RERANK_MODEL,
        "temperature": OPENROUTER_TEMPERATURE,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "k_values": E2_K_VALUES,
        "n_index_questions": len(index_dataset),
        "n_index_chunks": len(index_doc_pool),
        "n_eval_questions": len(eval_dataset),
        "rrf_k": E2_RRF_K,
        "rerank_top_n": E2_RERANK_TOP_N,
        "configs": [c["id"] for c in CONFIGS],
        "timestamp": timestamp,
    }
    _save_json(run_dir / "config.json", config)
    _save_json(run_dir / "sysinfo.json", _collect_sysinfo())

    # Phase A: Retrieval
    all_retrieval: dict[str, dict[int, list[dict]]] = {}
    if args.phase in ("retrieval", "all"):
        all_retrieval = run_retrieval_phase(
            eval_dataset, index_doc_pool, gold_lookup, run_dir, E2_K_VALUES
        )

    # Helper: load retrieval results from disk when not in memory
    def _load_retrieval_from_disk() -> dict[str, dict[int, list[dict]]]:
        loaded: dict[str, dict[int, list[dict]]] = {}
        for cfg in CONFIGS:
            cfg_id = cfg["id"]
            assert isinstance(cfg_id, str)
            cfg_k_results: dict[int, list[dict]] = {}
            all_present = True
            for k in E2_K_VALUES:
                ret_path = run_dir / "retrieval" / cfg_id / f"retrieved_k{k}.jsonl"
                if ret_path.exists():
                    records = []
                    for line in ret_path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            records.append(json.loads(line))
                    cfg_k_results[k] = records
                else:
                    all_present = False
                    break
            if all_present and cfg_k_results:
                loaded[cfg_id] = cfg_k_results
        return loaded

    # Phase B: Generation
    gen_results: dict[str, list[dict]] = {}
    if args.phase in ("generation", "all"):
        if not all_retrieval:
            all_retrieval = _load_retrieval_from_disk()
            if not all_retrieval:
                print("ERROR: No retrieval results found. Run retrieval phase first.")
                return 1

        gen_results = run_generation_phase(
            eval_dataset, eval_oracle, all_retrieval, run_dir, E2_K_VALUES
        )

    # Phase C: Aggregation
    if args.phase == "all" and gen_results:
        run_aggregation_phase(gen_results, all_retrieval, E2_K_VALUES, run_dir)

    # Phase D: Judge
    if args.phase in ("judge", "all"):
        if not all_retrieval:
            all_retrieval = _load_retrieval_from_disk()
            if not all_retrieval:
                print("ERROR: No retrieval results found. Run retrieval phase first.")
                return 1
        run_judge_phase(all_retrieval, run_dir, E2_K_VALUES)

    print(f"\nResults saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
