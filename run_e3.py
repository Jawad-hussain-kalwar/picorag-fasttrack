"""E3 — Selective Answering (Decision-Aware RAG) experiment runner.

Applies a confidence gate on top of the Local-Best retrieval config from E2.
Sweeps thresholds to find optimal accuracy-coverage trade-off.

Usage:
    .venv\\Scripts\\python.exe run_e3.py --partial          # 100 eval questions
    .venv\\Scripts\\python.exe run_e3.py --partial --resume
    .venv\\Scripts\\python.exe run_e3.py --smoke 10
    .venv\\Scripts\\python.exe run_e3.py --smoke 5 --phase judge --resume
"""

import argparse
import json
import platform
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.config import (
    CHROMA_PERSIST_DIR,
    E2_EVAL_N,
    E2_INDEX_N,
    E2_RERANK_TOP_N,
    E2_RRF_K,
    E3_ABSTAIN_MESSAGE,
    E3_LOCAL_BEST_CONFIG,
    E3_LOCAL_BEST_K,
    E3_N_UNANSWERABLE,
    E3_THRESHOLDS,
    JUDGE_MODEL,
    MIRAGE_COLLECTION_NAME,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_TEMPERATURE,
    RUNS_DIR,
)
from src.gate import should_abstain
from src.generate import (
    build_base_prompt,
    build_mixed_prompt_cited,
    build_oracle_prompt,
    call_openrouter,
)
from src.judge import aggregate_judge_scores, call_judge
from src.metrics import (
    auprc,
    citation_precision,
    citation_recall,
    compute_mirage_metrics,
    coverage,
    ece,
    em_loose,
    mrr,
    ndcg_at_k,
    parse_citations,
    precision_at_k,
    recall_at_k,
    selective_accuracy,
)
from src.mirage_loader import (
    load_dataset,
    load_doc_pool,
    load_oracle,
    select_partial_subset,
)
from src.retrieve import get_client, get_collection, index_mirage_pool, search

MAX_GEN_RETRIES = 5
MAX_JUDGE_RETRIES = 5


# ---------------------------------------------------------------------------
# Helpers
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


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Disk-loading helpers (for independent phase execution)
# ---------------------------------------------------------------------------

def _load_retrieval_from_disk(run_dir: Path, k: int) -> list[dict]:
    """Load retrieval results from disk."""
    ret_path = run_dir / "retrieval" / f"retrieved_k{k}.jsonl"
    if not ret_path.exists():
        raise FileNotFoundError(f"Retrieval results not found: {ret_path}")
    rows = _load_jsonl(ret_path)
    print(f"  Loaded {len(rows)} retrieval results from {ret_path}")
    return rows


def _load_generation_from_disk(
    run_dir: Path,
    thresholds: list[float],
) -> dict[str, list[dict]]:
    """Load all generation results from disk (base, oracle, mixed per tau)."""
    gen_results: dict[str, list[dict]] = {}

    # Base
    base_path = _jsonl_path(run_dir, "e3_base")
    if base_path.exists():
        gen_results["base"] = _load_jsonl(base_path)
        print(f"  Loaded {len(gen_results['base'])} base results")

    # Oracle
    oracle_path = _jsonl_path(run_dir, "e3_oracle")
    if oracle_path.exists():
        gen_results["oracle"] = _load_jsonl(oracle_path)
        print(f"  Loaded {len(gen_results['oracle'])} oracle results")

    # Mixed per threshold
    for tau in thresholds:
        tau_label = f"{tau:.2f}".replace(".", "")
        gen_name = f"e3_mixed_tau{tau_label}"
        gen_path = _jsonl_path(run_dir, gen_name)
        if gen_path.exists():
            gen_results[gen_name] = _load_jsonl(gen_path)
            print(f"  Loaded {len(gen_results[gen_name])} {gen_name} results")

    return gen_results


# ---------------------------------------------------------------------------
# E2 reuse: copy retrieval + base + oracle from E2 run
# ---------------------------------------------------------------------------

def _find_latest_e2_run(mode_label: str) -> Path | None:
    """Find the latest E2 run directory matching mode."""
    e2_dir = RUNS_DIR / "e2"
    if not e2_dir.exists():
        return None
    # For smoke tests, look for partial runs (E2 has no smoke mode for matching)
    # For partial, match partial_100
    search_label = "partial_100" if "partial" in mode_label else mode_label
    candidates = sorted(
        [d for d in e2_dir.iterdir() if d.is_dir() and search_label in d.name],
        reverse=True,
    )
    return candidates[0] if candidates else None


def _reuse_e2_results(
    e2_run: Path,
    e3_run: Path,
    n_eval: int,
) -> None:
    """Copy reusable E2 base + oracle generation into E3 run directory.

    Retrieval is NOT copied — it gets re-run for all questions including
    unanswerable ones added for E3.
    """
    # 1. Base generation results
    e2_base = e2_run / "samples" / "e2_base.jsonl"
    e3_base = _jsonl_path(e3_run, "e3_base")
    if e2_base.exists() and not e3_base.exists():
        e3_base.parent.mkdir(parents=True, exist_ok=True)
        rows = _load_jsonl(e2_base)
        rows = rows[:n_eval]
        with open(e3_base, "w", encoding="utf-8") as f:
            for r in rows:
                r["mode"] = "base"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Reused E2 base generation: {len(rows)} queries")

    # 2. Oracle generation results
    e2_oracle = e2_run / "samples" / "e2_oracle.jsonl"
    e3_oracle = _jsonl_path(e3_run, "e3_oracle")
    if e2_oracle.exists() and not e3_oracle.exists():
        e3_oracle.parent.mkdir(parents=True, exist_ok=True)
        rows = _load_jsonl(e2_oracle)
        rows = rows[:n_eval]
        with open(e3_oracle, "w", encoding="utf-8") as f:
            for r in rows:
                r["mode"] = "oracle"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Reused E2 oracle generation: {len(rows)} queries")


# ---------------------------------------------------------------------------
# Unanswerable questions: gold chunks NOT in indexed pool
# ---------------------------------------------------------------------------

def _select_unanswerable_questions(
    full_dataset: list[dict],
    indexed_query_ids: set[str],
    n: int,
) -> list[dict]:
    """Select n questions whose gold chunks are NOT in the indexed pool.

    These questions have valid answers in MIRAGE, but retrieving from the
    indexed subset will only return distractors. Perfect for testing
    the abstention gate.
    """
    import copy
    candidates = [q for q in full_dataset if q["query_id"] not in indexed_query_ids]
    candidates.sort(key=lambda q: q["query_id"])
    selected = []
    for q in candidates[:n]:
        q_copy = copy.deepcopy(q)
        q_copy["unanswerable_in_corpus"] = True
        selected.append(q_copy)
    return selected


# ---------------------------------------------------------------------------
# Retrieval config lookup (mirrors run_e2.py CONFIGS)
# ---------------------------------------------------------------------------

def _get_retrieval_method(config_id: str) -> str:
    """Return the retrieval method string for a given E2 config id."""
    mapping = {
        "1_vector_minilm": "vector",
        "2_bm25": "bm25",
        "3_hybrid_minilm": "hybrid",
        "4_hybrid_rerank_minilm": "hybrid+rerank",
        "5_vector_qwen3": "vector",
        "6_hybrid_qwen3": "hybrid",
    }
    return mapping.get(config_id, "vector")


def _get_gate_method(config_id: str) -> str:
    """Return the gate normalisation method for a config's distance type."""
    method = _get_retrieval_method(config_id)
    if method == "vector":
        return "cosine"
    # BM25, hybrid (RRF), hybrid+rerank all return higher=better scores
    return "higher_better"


# ---------------------------------------------------------------------------
# Phase A: Retrieval (Local-Best config only)
# ---------------------------------------------------------------------------

def run_retrieval_phase(
    eval_dataset: list[dict],
    index_doc_pool: list[dict],
    run_dir: Path,
    k: int,
    config_id: str,
) -> list[dict]:
    """Run retrieval for the Local-Best config. Returns per-query results."""
    print(f"\n{'='*60}")
    print(f"PHASE A: Retrieval ({config_id}, k={k})")
    print(f"{'='*60}")

    # Check if retrieval results already exist
    ret_path = run_dir / "retrieval" / f"retrieved_k{k}.jsonl"
    if ret_path.exists():
        existing = []
        for line in ret_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(json.loads(line))
        if len(existing) >= len(eval_dataset):
            print(f"  Retrieval already complete ({len(existing)} queries)")
            return existing

    method = _get_retrieval_method(config_id)
    fetch_n = max(k, E2_RERANK_TOP_N) if "rerank" in method else k

    # Build indices based on method — reuse E2 ChromaDB collection if it exists
    is_qwen3 = "qwen3" in config_id
    if method in ("vector", "hybrid", "hybrid+rerank"):
        if is_qwen3:
            from src.embed_openrouter import (
                index_mirage_with_custom_embeddings,
                search_with_custom_embeddings,
            )
            qwen3_collection_name = f"{MIRAGE_COLLECTION_NAME}_e2_qwen3"
            t0 = time.perf_counter()
            qwen3_indexed = index_mirage_with_custom_embeddings(
                index_doc_pool, qwen3_collection_name, batch_size=50,
            )
            print(f"  Qwen3 ChromaDB: {qwen3_indexed} chunks indexed in {time.perf_counter()-t0:.1f}s")
        else:
            collection_name = f"{MIRAGE_COLLECTION_NAME}_e2"
            client = get_client(str(CHROMA_PERSIST_DIR))
            collection = get_collection(client, collection_name)
            existing = collection.count()
            if existing >= len(index_doc_pool):
                print(f"  ChromaDB: reusing E2 index ({existing} chunks already indexed)")
            else:
                t0 = time.perf_counter()
                indexed = index_mirage_pool(collection, index_doc_pool)
                print(f"  ChromaDB: {indexed} chunks indexed in {time.perf_counter()-t0:.1f}s")

    if method in ("bm25", "hybrid", "hybrid+rerank"):
        from src.bm25 import bm25_search, build_bm25_index
        t0 = time.perf_counter()
        bm25_index, bm25_pool = build_bm25_index(index_doc_pool)
        print(f"  BM25: {len(index_doc_pool)} chunks indexed in {time.perf_counter()-t0:.1f}s")

    results: list[dict] = []
    t0 = time.perf_counter()

    for i, question in enumerate(eval_dataset):
        qid = question["query_id"]
        query_text = question["query"]

        if method == "vector" and is_qwen3:
            result = search_with_custom_embeddings(
                qwen3_collection_name, query_text, n_results=fetch_n,
            )
        elif method == "vector":
            result = search(collection, query_text, n_results=fetch_n)
        elif method == "bm25":
            result = bm25_search(bm25_index, bm25_pool, query_text, n_results=fetch_n)
        elif method == "hybrid":
            from src.hybrid import rrf_fuse
            vec_result = search(collection, query_text, n_results=fetch_n)
            bm25_result = bm25_search(bm25_index, bm25_pool, query_text, n_results=fetch_n)
            result = rrf_fuse(vec_result, bm25_result, k=fetch_n, rrf_k=E2_RRF_K)
        elif method == "hybrid+rerank":
            from src.hybrid import rrf_fuse
            from src.rerank import call_voyage_rerank
            vec_result = search(collection, query_text, n_results=fetch_n)
            bm25_result = bm25_search(bm25_index, bm25_pool, query_text, n_results=fetch_n)
            fused = rrf_fuse(vec_result, bm25_result, k=E2_RERANK_TOP_N, rrf_k=E2_RRF_K)
            result = call_voyage_rerank(
                query=query_text,
                documents=fused["documents"][0],
                top_k=fetch_n,
                metadatas=fused["metadatas"][0],
            )
        else:
            raise ValueError(f"Unknown method for config {config_id}: {method}")

        metadatas = result.get("metadatas", [[]])[0][:k]
        documents = result.get("documents", [[]])[0][:k]
        distances = result.get("distances", [[]])[0][:k]
        supports = [m.get("support", 0) for m in metadatas]

        rec = {
            "query_id": qid,
            "documents": documents,
            "distances": distances,
            "supports": supports,
            "metadatas": metadatas,
            "recall": recall_at_k(supports),
            "precision": precision_at_k(supports),
            "ndcg": ndcg_at_k(supports, k),
            "mrr": mrr(supports),
        }
        results.append(rec)

        if (i + 1) % 50 == 0 or i + 1 == len(eval_dataset):
            print(f"  Retrieved {i+1}/{len(eval_dataset)}")

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Recall@{k}={_mean([r['recall'] for r in results]):.4f}  "
          f"nDCG@{k}={_mean([r['ndcg'] for r in results]):.4f}  "
          f"MRR={_mean([r['mrr'] for r in results]):.4f}")

    # Save
    ret_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ret_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if method in ("vector", "hybrid", "hybrid+rerank") and not is_qwen3:
        del client

    return results


# ---------------------------------------------------------------------------
# Phase B: Gated Generation
# ---------------------------------------------------------------------------

def _run_generation_mode(
    name: str,
    dataset: list[dict],
    oracle: dict[str, dict],
    retrieval_results: list[dict] | None,
    run_dir: Path,
    mode: str,
    threshold: float | None = None,
    gate_method: str = "cosine",
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

        # Gate decision (mixed mode with threshold only)
        abstained = False
        confidence = 1.0
        if mode == "mixed" and threshold is not None and retrieval_results:
            distances = retrieval_results[i]["distances"]
            abstained, confidence = should_abstain(distances, threshold, method=gate_method)

        if abstained:
            prediction = E3_ABSTAIN_MESSAGE
            gen_ms = 0
        else:
            # Build prompt
            if mode == "base":
                messages = build_base_prompt(question["query"])
            elif mode == "oracle":
                chunk_text = oracle[qid]["doc_chunk"]
                messages = build_oracle_prompt(question["query"], chunk_text)
            elif mode == "mixed":
                assert retrieval_results is not None
                chunks = retrieval_results[i]["documents"]
                messages = build_mixed_prompt_cited(question["query"], chunks)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            t0 = time.perf_counter()
            prediction = None
            for gen_attempt in range(1, MAX_GEN_RETRIES + 1):
                try:
                    prediction = call_openrouter(messages)
                    break
                except RuntimeError as exc:
                    if gen_attempt < MAX_GEN_RETRIES:
                        wait = min(2 ** gen_attempt, 30)
                        print(f"    Gen error (attempt {gen_attempt}/{MAX_GEN_RETRIES}), "
                              f"retrying in {wait}s: {exc}")
                        time.sleep(wait)
                    else:
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
            "abstained": abstained,
            "confidence": round(confidence, 6),
        }

        # Citation metrics for mixed mode (answered only)
        if mode == "mixed" and not abstained and retrieval_results:
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
            n_abstained = sum(1 for r in results if r.get("abstained", False))
            print(f"    {name}: {done}/{total}  EM_loose={em_so_far:.4f}  "
                  f"abstained={n_abstained}  errors={errors}")

    return results


def run_generation_phase(
    eval_dataset: list[dict],
    oracle: dict[str, dict],
    retrieval_results: list[dict],
    run_dir: Path,
    thresholds: list[float],
    gate_method: str,
) -> dict[str, list[dict]]:
    """Run base, oracle, and gated mixed for each threshold (parallel)."""
    print(f"\n{'='*60}")
    print("PHASE B: Gated Generation (parallel)")
    print(f"{'='*60}")

    # Build task list: (result_key, name, ret_results, mode, threshold, gate_method)
    tasks: list[tuple[str, str, list[dict] | None, str, float | None, str]] = []
    tasks.append(("base", "e3_base", None, "base", None, gate_method))
    tasks.append(("oracle", "e3_oracle", None, "oracle", None, gate_method))
    for tau in thresholds:
        tau_label = f"{tau:.2f}".replace(".", "")
        gen_name = f"e3_mixed_tau{tau_label}"
        tasks.append((gen_name, gen_name, retrieval_results, "mixed", tau, gate_method))

    n_workers = min(8, len(tasks))
    print(f"  {len(tasks)} generation tasks, {n_workers} workers")

    all_gen: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for key, name, ret_res, mode, tau_val, gm in tasks:
            future = executor.submit(
                _run_generation_mode,
                name, eval_dataset, oracle, ret_res, run_dir, mode,
                threshold=tau_val, gate_method=gm,
            )
            futures[future] = key

        for future in as_completed(futures):
            key = futures[future]
            try:
                all_gen[key] = future.result()
            except Exception as exc:
                print(f"  ERROR in {key}: {exc}")
                traceback.print_exc()

    return all_gen


# ---------------------------------------------------------------------------
# Phase C: Aggregation
# ---------------------------------------------------------------------------

def run_aggregation_phase(
    gen_results: dict[str, list[dict]],
    retrieval_results: list[dict],
    thresholds: list[float],
    run_dir: Path,
) -> None:
    """Aggregate per-query results into gate + generation + MIRAGE + judge metrics."""
    print(f"\n{'='*60}")
    print("PHASE C: Aggregation")
    print(f"{'='*60}")

    # --- Gate metrics per threshold ---
    gate_metrics: dict = {}
    for tau in thresholds:
        tau_label = f"{tau:.2f}".replace(".", "")
        gen_key = f"e3_mixed_tau{tau_label}"
        results = gen_results.get(gen_key, [])
        if not results:
            continue

        predictions = [r["prediction"] for r in results]
        gold_answers = [r["gold_answers"] for r in results]
        answered_mask = [not r.get("abstained", False) for r in results]
        confidences = [r.get("confidence", 1.0) for r in results]
        em_labels = [int(em_loose(r["prediction"], r["gold_answers"])) for r in results]

        sel_acc = selective_accuracy(predictions, gold_answers, answered_mask)
        cov = coverage(answered_mask)
        # AUPRC: y_true = whether answer is correct, y_scores = confidence
        au = auprc(em_labels, confidences)
        # ECE: calibration of confidence vs correctness
        cal = ece(em_labels, confidences)
        # Overall EM (abstentions count as wrong)
        overall_em = _mean([r["em_loose"] for r in results])

        gate_metrics[f"tau={tau}"] = {
            "selective_accuracy": round(sel_acc, 4),
            "coverage": round(cov, 4),
            "auprc": round(au, 4),
            "ece": round(cal, 4),
            "overall_em_loose": round(overall_em, 4),
            "n_answered": sum(answered_mask),
            "n_abstained": sum(1 for a in answered_mask if not a),
            "n_total": len(results),
        }

        # Citation metrics (answered only)
        cit_p = [r["citation_precision"] for r in results if "citation_precision" in r]
        cit_r = [r["citation_recall"] for r in results if "citation_recall" in r]
        if cit_p:
            gate_metrics[f"tau={tau}"]["citation_precision"] = round(_mean(cit_p), 4)
        if cit_r:
            gate_metrics[f"tau={tau}"]["citation_recall"] = round(_mean(cit_r), 4)

        print(f"  tau={tau}: SelAcc={sel_acc:.4f}  Cov={cov:.4f}  "
              f"AUPRC={au:.4f}  ECE={cal:.4f}  EM={overall_em:.4f}")

    _save_json(run_dir / "gate_metrics.json", gate_metrics)

    # --- Generation metrics ---
    generation_metrics: dict = {}
    for mode_name, results in gen_results.items():
        em_l = _mean([r["em_loose"] for r in results])
        answered = [r for r in results if not r.get("abstained", False)]
        lat = _mean([r["generation_ms"] for r in answered]) if answered else 0.0

        entry: dict = {
            "EM_loose": round(em_l, 4),
            "avg_generation_ms": round(lat, 1),
            "n_queries": len(results),
            "n_answered": len(answered),
        }

        cit_p = [r["citation_precision"] for r in results if "citation_precision" in r]
        cit_r = [r["citation_recall"] for r in results if "citation_recall" in r]
        if cit_p:
            entry["citation_precision"] = round(_mean(cit_p), 4)
        if cit_r:
            entry["citation_recall"] = round(_mean(cit_r), 4)

        generation_metrics[mode_name] = entry
        print(f"  {mode_name}: EM_loose={em_l:.4f}  answered={len(answered)}/{len(results)}")

    _save_json(run_dir / "generation_metrics.json", generation_metrics)

    # --- MIRAGE metrics per threshold (answered queries only) ---
    base_results = gen_results.get("base", [])
    oracle_results = gen_results.get("oracle", [])
    base_by_qid = {r["query_id"]: r["em_loose"] for r in base_results}
    oracle_by_qid = {r["query_id"]: r["em_loose"] for r in oracle_results}

    mirage_metrics: dict = {}
    for tau in thresholds:
        tau_label = f"{tau:.2f}".replace(".", "")
        gen_key = f"e3_mixed_tau{tau_label}"
        mixed_results = gen_results.get(gen_key, [])
        if not mixed_results:
            continue

        # MIRAGE on answered queries only
        answered = [r for r in mixed_results if not r.get("abstained", False)]
        if not answered:
            continue

        base_labels = [base_by_qid.get(r["query_id"], 0.0) for r in answered]
        oracle_labels = [oracle_by_qid.get(r["query_id"], 0.0) for r in answered]
        mixed_labels = [r["em_loose"] for r in answered]

        m = compute_mirage_metrics(base_labels, oracle_labels, mixed_labels)
        mirage_metrics[f"tau={tau}"] = {k: round(v, 4) for k, v in m.items()}
        mirage_metrics[f"tau={tau}"]["n_answered"] = len(answered)

        print(f"  MIRAGE tau={tau}: NV={m['NV']:.4f}  CA={m['CA']:.4f}  "
              f"CI={m['CI']:.4f}  CM={m['CM']:.4f}  (n={len(answered)})")

    _save_json(run_dir / "mirage_metrics.json", mirage_metrics)

    # --- Judge metrics (load from disk if Phase D has run) ---
    judge_all: dict = {}
    for tau in thresholds:
        tau_label = f"{tau:.2f}".replace(".", "")
        judge_path = _jsonl_path(run_dir, f"judge_tau{tau_label}")
        if judge_path.exists():
            rows = _load_jsonl(judge_path)
            agg = aggregate_judge_scores(rows)
            judge_all[f"tau={tau}"] = agg
            print(f"  Judge tau={tau}: faith={agg['faithfulness']}  "
                  f"ground={agg['groundedness']}  relev={agg['answer_relevance']}  "
                  f"correct={agg['semantic_correctness']}  judged={agg['n_judged']}")

    if judge_all:
        _save_json(run_dir / "judge_metrics.json", judge_all)

    # --- Retrieval summary ---
    print(f"\n  Retrieval: Recall@k={_mean([r['recall'] for r in retrieval_results]):.4f}  "
          f"nDCG@k={_mean([r['ndcg'] for r in retrieval_results]):.4f}  "
          f"MRR={_mean([r['mrr'] for r in retrieval_results]):.4f}")


# ---------------------------------------------------------------------------
# Phase D: Judge
# ---------------------------------------------------------------------------


def _judge_single_query(
    tau_label: str,
    qid: str,
    g: dict,
    contexts: list[str],
    judge_path: Path,
) -> dict:
    """Judge a single query. Returns the judge row dict."""
    if g.get("abstained", False):
        return {
            "query_id": qid,
            "faithfulness": None, "groundedness": None,
            "answer_relevance": None, "semantic_correctness": None,
            "judge_raw": "SKIPPED_ABSTAINED",
        }

    if g["prediction"].startswith("ERROR:"):
        return {
            "query_id": qid,
            "faithfulness": None, "groundedness": None,
            "answer_relevance": None, "semantic_correctness": None,
            "judge_raw": "SKIPPED_ERROR_PREDICTION",
        }

    result = call_judge(
        question=g["query"],
        contexts=contexts,
        prediction=g["prediction"],
        gold_answers=g.get("gold_answers"),
    )

    return {
        "query_id": qid,
        "faithfulness": result["faithfulness"],
        "groundedness": result["groundedness"],
        "answer_relevance": result["answer_relevance"],
        "semantic_correctness": result["semantic_correctness"],
        "judge_raw": result["raw"],
    }


def run_judge_phase(
    run_dir: Path,
    thresholds: list[float],
    k: int,
) -> None:
    """Run LLM-as-Judge on mixed-mode predictions, parallelized per-query."""
    print(f"\n{'='*60}")
    print(f"PHASE D: LLM-as-Judge (per-query parallel, {JUDGE_MODEL})")
    print(f"{'='*60}")

    retrieval_results = _load_retrieval_from_disk(run_dir, k)
    ret_by_qid = {r["query_id"]: r for r in retrieval_results}

    # Collect all (tau, qid, gen_row, contexts) tasks across all thresholds
    judge_tasks: list[tuple[str, float, str, dict, list[str], Path]] = []
    for tau in thresholds:
        tau_label = f"{tau:.2f}".replace(".", "")
        gen_name = f"e3_mixed_tau{tau_label}"
        gen_path = _jsonl_path(run_dir, gen_name)
        if not gen_path.exists():
            print(f"  Skipping tau={tau}: generation file not found")
            continue

        judge_path = _jsonl_path(run_dir, f"judge_tau{tau_label}")
        judged = _load_checkpoint(judge_path)
        gen_rows = _load_jsonl(gen_path)

        for g in gen_rows:
            qid = g["query_id"]
            if qid in judged:
                continue
            ret = ret_by_qid.get(qid)
            contexts = ret["documents"] if ret else []
            judge_tasks.append((tau_label, tau, qid, g, contexts, judge_path))

    if not judge_tasks:
        print("  All judge tasks already complete")
        # Still aggregate and save
        judge_all: dict = {}
        for tau in thresholds:
            tau_label = f"{tau:.2f}".replace(".", "")
            judge_path = _jsonl_path(run_dir, f"judge_tau{tau_label}")
            if judge_path.exists():
                rows = _load_jsonl(judge_path)
                agg = aggregate_judge_scores(rows)
                judge_all[f"tau={tau}"] = agg
                print(f"  tau={tau}: faith={agg['faithfulness']}  "
                      f"ground={agg['groundedness']}  judged={agg['n_judged']}")
        if judge_all:
            _save_json(run_dir / "judge_metrics.json", judge_all)
        return

    n_workers = min(12, len(judge_tasks))
    print(f"  {len(judge_tasks)} judge queries, {n_workers} workers")

    completed_count = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for tau_label, tau, qid, g, contexts, judge_path in judge_tasks:
            future = executor.submit(
                _judge_single_query, tau_label, qid, g, contexts, judge_path,
            )
            futures[future] = (tau_label, tau, qid, judge_path)

        for future in as_completed(futures):
            tau_label, tau, qid, judge_path = futures[future]
            try:
                row = future.result()
                _append_jsonl(judge_path, row)
                completed_count += 1
                if row["faithfulness"] is None and "SKIPPED" not in row.get("judge_raw", ""):
                    errors += 1
                if completed_count % 10 == 0 or completed_count == len(judge_tasks):
                    print(f"    Judged {completed_count}/{len(judge_tasks)} ({errors} errors)")
            except Exception as exc:
                print(f"  ERROR judging tau={tau} qid={qid}: {exc}")
                traceback.print_exc()

    # Aggregate per threshold
    judge_all = {}
    for tau in thresholds:
        tau_label = f"{tau:.2f}".replace(".", "")
        judge_path = _jsonl_path(run_dir, f"judge_tau{tau_label}")
        if judge_path.exists():
            rows = _load_jsonl(judge_path)
            agg = aggregate_judge_scores(rows)
            judge_all[f"tau={tau}"] = agg
            print(f"  tau={tau} done — faith={agg['faithfulness']}  "
                  f"ground={agg['groundedness']}  relev={agg['answer_relevance']}  "
                  f"correct={agg['semantic_correctness']}  judged={agg['n_judged']}")

    if judge_all:
        _save_json(run_dir / "judge_metrics.json", judge_all)
        print(f"\n  Judge metrics saved to {run_dir / 'judge_metrics.json'}")


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
    parser = argparse.ArgumentParser(description="E3 — Selective Answering")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--partial", action="store_true", help=f"{E2_EVAL_N} eval questions")
    group.add_argument("--full", action="store_true", help="All 7,560 questions")
    group.add_argument("--smoke", type=int, metavar="N", help="Smoke test with N questions")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--phase",
        choices=["retrieval", "generation", "judge", "aggregation", "all"],
        default="all",
        help="Run only a specific phase (default: all)",
    )
    args = parser.parse_args()

    # Load data
    print("Loading MIRAGE dataset...")
    dataset = load_dataset()
    doc_pool = load_doc_pool()
    oracle = load_oracle()

    # Determine subset sizes
    if args.smoke:
        n_index = args.smoke * 5
        n_eval = args.smoke
    elif args.full:
        n_index = len(dataset)
        n_eval = len(dataset)
    else:
        n_index = E2_INDEX_N
        n_eval = E2_EVAL_N

    # Select subsets
    index_dataset, index_doc_pool, index_oracle = select_partial_subset(
        dataset, doc_pool, oracle, n_index
    )
    eval_dataset, _, eval_oracle = select_partial_subset(
        index_dataset, index_doc_pool, index_oracle, n_eval
    )
    eval_oracle = {
        q["query_id"]: index_oracle[q["query_id"]]
        for q in eval_dataset if q["query_id"] in index_oracle
    }
    n_answerable = len(eval_dataset)

    # Add unanswerable questions (gold chunks NOT in indexed pool)
    n_unanswerable = E3_N_UNANSWERABLE if not args.smoke else min(3, E3_N_UNANSWERABLE)
    indexed_qids = {q["query_id"] for q in index_dataset}
    unanswerable_qs = _select_unanswerable_questions(dataset, indexed_qids, n_unanswerable)
    eval_dataset = eval_dataset + unanswerable_qs
    # Add oracle data for unanswerable questions (they have gold context in MIRAGE)
    for q in unanswerable_qs:
        qid = q["query_id"]
        if qid in oracle:
            eval_oracle[qid] = oracle[qid]

    mode_label = (
        f"smoke_{n_answerable}" if args.smoke
        else "full" if args.full
        else f"partial_{E2_EVAL_N}"
    )

    config_id = E3_LOCAL_BEST_CONFIG
    k = E3_LOCAL_BEST_K
    gate_method = _get_gate_method(config_id)

    print(f"Mode: {mode_label}")
    print(f"Local-Best config: {config_id} (k={k})")
    print(f"Gate method: {gate_method}")
    print(f"Thresholds: {E3_THRESHOLDS}")
    print(f"Index chunks: {len(index_doc_pool)}")
    print(f"Eval questions: {n_answerable} answerable + {len(unanswerable_qs)} unanswerable = {len(eval_dataset)}")
    print(f"Model: {OPENROUTER_MODEL}")
    print(f"Phase: {args.phase}")

    # Setup run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / "e3" / f"{timestamp}_{mode_label}"
    if args.resume:
        e3_dir = RUNS_DIR / "e3"
        if e3_dir.exists():
            candidates = sorted(
                [d for d in e3_dir.iterdir() if d.is_dir() and mode_label in d.name],
                reverse=True,
            )
            if candidates:
                run_dir = candidates[0]
                print(f"Resuming from: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Reuse E2 base + oracle generation (retrieval is re-run for all questions)
    e2_run = _find_latest_e2_run(mode_label)
    if e2_run:
        print(f"\nReusing E2 base+oracle from: {e2_run}")
        _reuse_e2_results(e2_run, run_dir, n_answerable)
    else:
        print("\nNo E2 run found for reuse — will compute from scratch")

    # Save config
    config = {
        "experiment": "E3",
        "mode": mode_label,
        "local_best_config": config_id,
        "local_best_k": k,
        "gate_method": gate_method,
        "thresholds": E3_THRESHOLDS,
        "model": OPENROUTER_MODEL,
        "judge_model": JUDGE_MODEL,
        "temperature": OPENROUTER_TEMPERATURE,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "n_index_chunks": len(index_doc_pool),
        "n_eval_questions": len(eval_dataset),
        "n_answerable": n_answerable,
        "n_unanswerable": len(unanswerable_qs),
        "timestamp": timestamp,
    }
    _save_json(run_dir / "config.json", config)
    _save_json(run_dir / "sysinfo.json", _collect_sysinfo())

    # Phase routing
    phase = args.phase

    if phase in ("retrieval", "all"):
        retrieval_results = run_retrieval_phase(
            eval_dataset, index_doc_pool, run_dir, k, config_id
        )
    else:
        retrieval_results = _load_retrieval_from_disk(run_dir, k)

    if phase in ("generation", "all"):
        gen_results = run_generation_phase(
            eval_dataset, eval_oracle, retrieval_results, run_dir, E3_THRESHOLDS, gate_method
        )
    elif phase in ("aggregation", "judge"):
        gen_results = _load_generation_from_disk(run_dir, E3_THRESHOLDS)
    else:
        gen_results = {}

    if phase in ("judge", "all"):
        run_judge_phase(run_dir, E3_THRESHOLDS, k)

    if phase in ("aggregation", "all"):
        run_aggregation_phase(gen_results, retrieval_results, E3_THRESHOLDS, run_dir)

    print(f"\nResults saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
