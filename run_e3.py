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
        tau_label = f"{tau:.1f}".replace(".", "")
        gen_name = f"e3_mixed_tau{tau_label}"
        gen_path = _jsonl_path(run_dir, gen_name)
        if gen_path.exists():
            gen_results[gen_name] = _load_jsonl(gen_path)
            print(f"  Loaded {len(gen_results[gen_name])} {gen_name} results")

    return gen_results


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
    if method in ("vector", "hybrid", "hybrid+rerank"):
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

        if method == "vector":
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

    if method in ("vector", "hybrid", "hybrid+rerank"):
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
                chunks = retrieval_results[i]["documents"]
                messages = build_mixed_prompt_cited(question["query"], chunks)
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
        tau_label = f"{tau:.1f}".replace(".", "")
        gen_name = f"e3_mixed_tau{tau_label}"
        tasks.append((gen_name, gen_name, retrieval_results, "mixed", tau, gate_method))

    n_workers = min(8, len(tasks))
    print(f"  {len(tasks)} generation tasks, {n_workers} workers")

    all_gen: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for key, name, ret_res, mode, tau, gm in tasks:
            future = executor.submit(
                _run_generation_mode,
                name, eval_dataset, oracle, ret_res, run_dir, mode,
                threshold=tau, gate_method=gm,
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
        tau_label = f"{tau:.1f}".replace(".", "")
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
        tau_label = f"{tau:.1f}".replace(".", "")
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
        tau_label = f"{tau:.1f}".replace(".", "")
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

def _judge_one_threshold(
    tau: float,
    run_dir: Path,
    ret_by_qid: dict[str, dict],
) -> tuple[str, dict] | None:
    """Judge one threshold. Returns (judge_key, agg_dict) or None."""
    tau_label = f"{tau:.1f}".replace(".", "")
    gen_name = f"e3_mixed_tau{tau_label}"
    gen_path = _jsonl_path(run_dir, gen_name)

    if not gen_path.exists():
        print(f"  Skipping tau={tau}: generation file not found")
        return None

    gen_rows = _load_jsonl(gen_path)
    total = len(gen_rows)

    judge_name = f"judge_tau{tau_label}"
    judge_path = _jsonl_path(run_dir, judge_name)
    judged = _load_checkpoint(judge_path)

    if len(judged) >= total:
        print(f"  tau={tau}: already complete ({total}/{total})")
    elif len(judged) > 0:
        print(f"  tau={tau}: resuming from {len(judged)}/{total}")
    else:
        print(f"  tau={tau}: judging {total} predictions")

    skipped = 0
    errors = 0

    for g in gen_rows:
        qid = g["query_id"]
        if qid in judged:
            continue

        if g.get("abstained", False):
            skipped += 1
            row = {
                "query_id": qid,
                "faithfulness": None, "groundedness": None,
                "answer_relevance": None, "semantic_correctness": None,
                "judge_raw": "SKIPPED_ABSTAINED",
            }
            _append_jsonl(judge_path, row)
            judged[qid] = row
            continue

        if g["prediction"].startswith("ERROR:"):
            skipped += 1
            row = {
                "query_id": qid,
                "faithfulness": None, "groundedness": None,
                "answer_relevance": None, "semantic_correctness": None,
                "judge_raw": "SKIPPED_ERROR_PREDICTION",
            }
            _append_jsonl(judge_path, row)
            judged[qid] = row
            continue

        ret = ret_by_qid.get(qid)
        contexts = ret["documents"] if ret else []

        result = call_judge(
            question=g["query"],
            contexts=contexts,
            prediction=g["prediction"],
            gold_answers=g.get("gold_answers"),
        )

        row = {
            "query_id": qid,
            "faithfulness": result["faithfulness"],
            "groundedness": result["groundedness"],
            "answer_relevance": result["answer_relevance"],
            "semantic_correctness": result["semantic_correctness"],
            "judge_raw": result["raw"],
        }
        _append_jsonl(judge_path, row)
        judged[qid] = row

        if result["faithfulness"] is None:
            errors += 1

        done = len(judged)
        if done % 10 == 0 or done == total:
            print(f"    tau={tau}: {done}/{total} judged "
                  f"({errors} parse errors, {skipped} skipped)")

    all_rows = _load_jsonl(judge_path)
    agg = aggregate_judge_scores(all_rows)
    print(f"  tau={tau} done — faith={agg['faithfulness']}  "
          f"ground={agg['groundedness']}  relev={agg['answer_relevance']}  "
          f"correct={agg['semantic_correctness']}  judged={agg['n_judged']}")
    return (f"tau={tau}", agg)


def run_judge_phase(
    run_dir: Path,
    thresholds: list[float],
    k: int,
) -> None:
    """Run LLM-as-Judge on mixed-mode predictions for each threshold (parallel)."""
    print(f"\n{'='*60}")
    print(f"PHASE D: LLM-as-Judge (parallel, {JUDGE_MODEL})")
    print(f"{'='*60}")

    retrieval_results = _load_retrieval_from_disk(run_dir, k)
    ret_by_qid = {r["query_id"]: r for r in retrieval_results}

    n_workers = min(8, len(thresholds))
    print(f"  {len(thresholds)} judge tasks, {n_workers} workers")

    judge_all: dict = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_judge_one_threshold, tau, run_dir, ret_by_qid): tau
            for tau in thresholds
        }
        for future in as_completed(futures):
            tau = futures[future]
            try:
                result = future.result()
                if result:
                    key, agg = result
                    judge_all[key] = agg
            except Exception as exc:
                print(f"  ERROR judging tau={tau}: {exc}")
                traceback.print_exc()

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

    mode_label = (
        f"smoke_{n_eval}" if args.smoke
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
    print(f"Eval questions: {len(eval_dataset)}")
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
