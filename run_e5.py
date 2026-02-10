"""E5 — Agentic Multi-Hop RAG experiment runner.

Adds an agentic controller that reformulates queries and retries retrieval
when evidence is weak.  Builds on E3/E4 Local-Best pipeline.

Usage:
    .venv\\Scripts\\python.exe run_e5.py --partial
    .venv\\Scripts\\python.exe run_e5.py --partial --resume
    .venv\\Scripts\\python.exe run_e5.py --smoke 5
    .venv\\Scripts\\python.exe run_e5.py --partial --phase judge --resume
"""

import argparse
import copy
import json
import platform
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path

from src.config import (
    E2_EVAL_N,
    E2_INDEX_N,
    E3_N_UNANSWERABLE,
    E5_AGENT_WORKERS,
    E5_GATE_THRESHOLD,
    E5_JUDGE_MODEL,
    E5_K,
    MIRAGE_COLLECTION_NAME,
    OPENROUTER_EMBED_MODEL,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_TEMPERATURE,
    RUNS_DIR,
)
from src.generate import (
    build_base_prompt,
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

MAX_GEN_RETRIES = 5
MODES = ["base", "oracle", "agentic_mixed"]


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


def _save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
# Unanswerable questions
# ---------------------------------------------------------------------------

def _select_unanswerable_questions(
    full_dataset: list[dict],
    indexed_query_ids: set[str],
    n: int,
) -> list[dict]:
    candidates = [q for q in full_dataset if q["query_id"] not in indexed_query_ids]
    candidates.sort(key=lambda q: q["query_id"])
    selected = []
    for q in candidates[:n]:
        q_copy = copy.deepcopy(q)
        q_copy["unanswerable_in_corpus"] = True
        selected.append(q_copy)
    return selected


# ---------------------------------------------------------------------------
# E3/E4 reuse — find latest run for base/oracle generation
# ---------------------------------------------------------------------------

def _find_latest_run(experiment: str, mode_label: str) -> Path | None:
    exp_dir = RUNS_DIR / experiment
    if not exp_dir.exists():
        return None
    search_label = "partial_100" if "partial" in mode_label else mode_label
    candidates = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and search_label in d.name],
        reverse=True,
    )
    return candidates[0] if candidates else None


def _reuse_base_oracle(
    run_dir: Path,
    mode_label: str,
    n_eval: int,
) -> None:
    """Try to reuse base/oracle generation from E4 or E3."""
    for exp in ("e4", "e3"):
        prev_run = _find_latest_run(exp, mode_label)
        if not prev_run:
            continue

        # Base
        for src_name in (f"{exp}_local_base", f"{exp}_base"):
            src_path = prev_run / "samples" / f"{src_name}.jsonl"
            if src_path.exists():
                dst_path = _jsonl_path(run_dir, "e5_base")
                if not dst_path.exists():
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    rows = _load_jsonl(src_path)[:n_eval]
                    with open(dst_path, "w", encoding="utf-8") as f:
                        for r in rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print(f"  Reused {exp} base: {len(rows)} queries")
                break

        # Oracle
        for src_name in (f"{exp}_local_oracle", f"{exp}_oracle"):
            src_path = prev_run / "samples" / f"{src_name}.jsonl"
            if src_path.exists():
                dst_path = _jsonl_path(run_dir, "e5_oracle")
                if not dst_path.exists():
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    rows = _load_jsonl(src_path)[:n_eval]
                    with open(dst_path, "w", encoding="utf-8") as f:
                        for r in rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print(f"  Reused {exp} oracle: {len(rows)} queries")
                break


# ---------------------------------------------------------------------------
# Phase A: Indexing (BM25 + Qwen3 ChromaDB)
# ---------------------------------------------------------------------------

def run_indexing_phase(
    index_doc_pool: list[dict],
    collection_name: str,
    embed_model: str,
) -> tuple:
    """Build BM25 index and ensure Qwen3 ChromaDB collection.

    Returns (bm25_index, bm25_pool, embed_fn).
    """
    print(f"\n{'='*60}")
    print("PHASE A: Indexing (BM25 + ChromaDB)")
    print(f"{'='*60}")

    from src.bm25 import build_bm25_index
    from src.embed_openrouter import (
        embed_texts_openrouter,
        index_mirage_with_custom_embeddings,
    )

    # BM25
    t0 = time.perf_counter()
    bm25_index, bm25_pool = build_bm25_index(index_doc_pool)
    print(f"  BM25: {len(index_doc_pool)} chunks in {time.perf_counter()-t0:.1f}s")

    # Qwen3 ChromaDB
    embed_fn = partial(embed_texts_openrouter, model=embed_model, batch_size=25)
    t0 = time.perf_counter()
    n_indexed = index_mirage_with_custom_embeddings(
        index_doc_pool, collection_name, embed_fn=embed_fn, batch_size=25,
    )
    print(f"  ChromaDB: {n_indexed} chunks ({collection_name}) in {time.perf_counter()-t0:.1f}s")

    return bm25_index, bm25_pool, embed_fn


# ---------------------------------------------------------------------------
# Phase B: Generation (base + oracle + agentic-mixed)
# ---------------------------------------------------------------------------

def _run_base_oracle_generation(
    mode: str,
    dataset: list[dict],
    oracle_map: dict[str, dict],
    run_dir: Path,
) -> list[dict]:
    """Run base or oracle generation."""
    task_name = f"e5_{mode}"
    checkpoint_path = _jsonl_path(run_dir, task_name)
    completed = _load_checkpoint(checkpoint_path)
    total = len(dataset)

    if len(completed) >= total:
        print(f"  {task_name}: already complete ({total}/{total})")
        return list(completed.values())

    if completed:
        print(f"  {task_name}: resuming from {len(completed)}/{total}")
    else:
        print(f"  {task_name}: starting {total} queries")

    results = list(completed.values())
    errors = 0

    for i, question in enumerate(dataset):
        qid = question["query_id"]
        if qid in completed:
            continue

        if mode == "base":
            messages = build_base_prompt(question["query"])
        elif mode == "oracle":
            chunk = oracle_map.get(qid, {}).get("doc_chunk", "")
            messages = build_oracle_prompt(question["query"], chunk)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        t0 = time.perf_counter()
        prediction = ""
        for attempt in range(1, MAX_GEN_RETRIES + 1):
            try:
                prediction = call_openrouter(messages)
                if prediction:
                    break
                if attempt < MAX_GEN_RETRIES:
                    time.sleep(1)
            except RuntimeError as exc:
                if attempt < MAX_GEN_RETRIES:
                    time.sleep(min(2 ** attempt, 30))
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
            "em_loose": em_loose(prediction or "", answers),
            "generation_ms": gen_ms,
            "mode": mode,
        }

        _append_jsonl(checkpoint_path, record)
        results.append(record)
        completed[qid] = record

        done = len(completed)
        if done % 10 == 0 or done == total:
            em_so_far = _mean([r["em_loose"] for r in results])
            print(f"    {task_name}: {done}/{total}  EM={em_so_far:.4f}  errors={errors}")

    return results


def _run_agentic_generation(
    dataset: list[dict],
    run_dir: Path,
    collection_name: str,
    embed_fn,
    bm25_index,
    bm25_pool: list[dict],
    n_workers: int,
) -> list[dict]:
    """Run agentic multi-hop generation with parallel workers."""
    from src.agent import run_agent_loop

    task_name = "e5_agentic_mixed"
    checkpoint_path = _jsonl_path(run_dir, task_name)
    completed = _load_checkpoint(checkpoint_path)
    total = len(dataset)

    if len(completed) >= total:
        print(f"  {task_name}: already complete ({total}/{total})")
        return list(completed.values())

    remaining = [q for q in dataset if q["query_id"] not in completed]
    print(f"  {task_name}: {len(remaining)} remaining of {total} ({len(completed)} done)")

    results = list(completed.values())
    errors = 0
    done_count = len(completed)

    def _process_query(question: dict) -> dict:
        return run_agent_loop(
            query_id=question["query_id"],
            query=question["query"],
            gold_answers=question["answer"],
            k=E5_K,
            threshold=E5_GATE_THRESHOLD,
            collection_name=collection_name,
            embed_fn=embed_fn,
            bm25_index=bm25_index,
            bm25_pool=bm25_pool,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_query, q): q["query_id"]
            for q in remaining
        }

        for future in as_completed(futures):
            qid = futures[future]
            try:
                record = future.result()

                # Add unanswerable flag
                q = next(q for q in dataset if q["query_id"] == qid)
                is_unanswerable = q.get("unanswerable_in_corpus", False)
                record["unanswerable"] = is_unanswerable

                # Correct abstention on unanswerable = correct
                if record["abstained"] and is_unanswerable:
                    record["em_loose"] = 1.0

                # Citation metrics
                if not record["abstained"]:
                    gold_indices = [
                        j + 1 for j, s in enumerate(record.get("supports", []))
                        if s == 1
                    ]
                    cited = parse_citations(record.get("prediction", ""))
                    record["cited_indices"] = cited
                    record["gold_indices"] = gold_indices
                    record["citation_precision"] = citation_precision(cited, gold_indices)
                    record["citation_recall"] = citation_recall(cited, gold_indices)

                _append_jsonl(checkpoint_path, record)
                results.append(record)
                done_count += 1

                if done_count % 5 == 0 or done_count == total:
                    em_so_far = _mean([r["em_loose"] for r in results])
                    n_abs = sum(1 for r in results if r.get("abstained", False))
                    n_hop2 = sum(1 for r in results if r.get("hops", 1) >= 2)
                    print(f"    {task_name}: {done_count}/{total}  EM={em_so_far:.4f}  "
                          f"abstained={n_abs}  hop2={n_hop2}  errors={errors}")

            except Exception as exc:
                errors += 1
                print(f"  ERROR {qid}: {exc}")
                traceback.print_exc()

    return results


def run_generation_phase(
    eval_dataset: list[dict],
    oracle_map: dict[str, dict],
    run_dir: Path,
    collection_name: str,
    embed_fn,
    bm25_index,
    bm25_pool: list[dict],
    n_workers: int,
) -> dict[str, list[dict]]:
    """Run all three generation modes."""
    print(f"\n{'='*60}")
    print("PHASE B: Generation (Base + Oracle + Agentic-Mixed)")
    print(f"{'='*60}")

    gen_results: dict[str, list[dict]] = {}

    # Base and oracle can run in parallel (simple LLM calls)
    with ThreadPoolExecutor(max_workers=2) as executor:
        base_future = executor.submit(
            _run_base_oracle_generation, "base", eval_dataset, oracle_map, run_dir,
        )
        oracle_future = executor.submit(
            _run_base_oracle_generation, "oracle", eval_dataset, oracle_map, run_dir,
        )
        gen_results["base"] = base_future.result()
        gen_results["oracle"] = oracle_future.result()

    # Agentic-mixed (uses retrieval infrastructure, runs with its own workers)
    gen_results["agentic_mixed"] = _run_agentic_generation(
        eval_dataset, run_dir, collection_name, embed_fn,
        bm25_index, bm25_pool, n_workers,
    )

    return gen_results


# ---------------------------------------------------------------------------
# Phase C: Judge
# ---------------------------------------------------------------------------

def _judge_single_query(
    qid: str,
    g: dict,
    contexts: list[str],
    judge_model: str,
) -> dict:
    if g.get("abstained", False):
        return {
            "query_id": qid,
            "faithfulness": None, "groundedness": None,
            "answer_relevance": None, "semantic_correctness": None,
            "judge_raw": "SKIPPED_ABSTAINED",
        }

    pred = g.get("prediction", "") or ""
    if pred.startswith("ERROR:") or not pred.strip():
        return {
            "query_id": qid,
            "faithfulness": None, "groundedness": None,
            "answer_relevance": None, "semantic_correctness": None,
            "judge_raw": "SKIPPED_EMPTY_OR_ERROR",
        }

    result = call_judge(
        question=g["query"],
        contexts=contexts,
        prediction=pred,
        gold_answers=g.get("gold_answers"),
        model=judge_model,
    )
    return {
        "query_id": qid,
        "faithfulness": result["faithfulness"],
        "groundedness": result["groundedness"],
        "answer_relevance": result["answer_relevance"],
        "semantic_correctness": result["semantic_correctness"],
        "judge_raw": result["raw"],
    }


def run_judge_phase(run_dir: Path) -> None:
    """Run LLM-as-Judge on agentic-mixed predictions."""
    print(f"\n{'='*60}")
    print(f"PHASE C: LLM-as-Judge ({E5_JUDGE_MODEL})")
    print(f"{'='*60}")

    gen_path = _jsonl_path(run_dir, "e5_agentic_mixed")
    if not gen_path.exists():
        print("  Skipping: agentic-mixed generation not found")
        return

    judge_path = _jsonl_path(run_dir, "judge_agentic")
    judged = _load_checkpoint(judge_path)
    gen_rows = _load_jsonl(gen_path)

    judge_tasks: list[tuple[str, dict, list[str]]] = []
    for g in gen_rows:
        qid = g["query_id"]
        if qid in judged:
            continue
        contexts = g.get("documents", [])
        judge_tasks.append((qid, g, contexts))

    if not judge_tasks:
        print("  All judge tasks complete")
        _save_judge_summary(run_dir)
        return

    n_workers = min(12, len(judge_tasks))
    print(f"  {len(judge_tasks)} queries to judge, {n_workers} workers")

    completed_count = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for qid, g, contexts in judge_tasks:
            future = executor.submit(
                _judge_single_query, qid, g, contexts, E5_JUDGE_MODEL,
            )
            futures[future] = qid

        for future in as_completed(futures):
            qid = futures[future]
            try:
                row = future.result()
                _append_jsonl(judge_path, row)
                completed_count += 1
                if row["faithfulness"] is None and "SKIPPED" not in row.get("judge_raw", ""):
                    errors += 1
                if completed_count % 10 == 0 or completed_count == len(judge_tasks):
                    print(f"    Judged {completed_count}/{len(judge_tasks)} ({errors} errors)")
            except Exception as exc:
                print(f"  ERROR judging {qid}: {exc}")
                traceback.print_exc()

    _save_judge_summary(run_dir)


def _save_judge_summary(run_dir: Path) -> None:
    judge_path = _jsonl_path(run_dir, "judge_agentic")
    if judge_path.exists():
        rows = _load_jsonl(judge_path)
        agg = aggregate_judge_scores(rows)
        _save_json(run_dir / "judge_metrics.json", agg)
        print(f"  Judge: faith={agg['faithfulness']}  ground={agg['groundedness']}  "
              f"relev={agg['answer_relevance']}  correct={agg['semantic_correctness']}  "
              f"judged={agg['n_judged']}")


# ---------------------------------------------------------------------------
# Phase D: Aggregation
# ---------------------------------------------------------------------------

def run_aggregation_phase(
    gen_results: dict[str, list[dict]],
    run_dir: Path,
) -> None:
    """Compute all metrics and save to JSON files."""
    print(f"\n{'='*60}")
    print("PHASE D: Aggregation")
    print(f"{'='*60}")

    k = E5_K
    agentic_results = gen_results.get("agentic_mixed", [])

    # --- Retrieval metrics (from agentic results) ---
    ret_entries = [r for r in agentic_results if r.get("supports")]
    if ret_entries:
        ret_metrics = {
            f"recall@{k}": round(_mean([recall_at_k(r["supports"]) for r in ret_entries]), 4),
            f"precision@{k}": round(_mean([precision_at_k(r["supports"]) for r in ret_entries]), 4),
            f"ndcg@{k}": round(_mean([ndcg_at_k(r["supports"], k) for r in ret_entries]), 4),
            "mrr": round(_mean([mrr(r["supports"]) for r in ret_entries]), 4),
            "n_queries": len(ret_entries),
        }
        _save_json(run_dir / "retrieval_metrics.json", ret_metrics)
        print(f"  Retrieval: Recall@{k}={ret_metrics[f'recall@{k}']}  "
              f"nDCG@{k}={ret_metrics[f'ndcg@{k}']}  MRR={ret_metrics['mrr']}")

    # --- Generation metrics (per mode) ---
    generation_metrics: dict = {}
    for mode_name, results in gen_results.items():
        if not results:
            continue
        answered = [r for r in results if not r.get("abstained", False)]
        em_l = _mean([r["em_loose"] for r in results])
        lat = _mean([r.get("generation_ms", 0) for r in answered]) if answered else 0.0

        entry: dict = {
            "em_loose": round(em_l, 4),
            "avg_generation_ms": round(lat, 1),
            "n_total": len(results),
            "n_answered": len(answered),
            "n_abstained": len(results) - len(answered),
        }

        if answered:
            latencies = sorted([r.get("generation_ms", 0) for r in answered])
            p50_idx = len(latencies) // 2
            p95_idx = int(len(latencies) * 0.95)
            entry["p50_latency_ms"] = latencies[p50_idx]
            entry["p95_latency_ms"] = latencies[min(p95_idx, len(latencies) - 1)]

        # Citation metrics
        cit_p = [r["citation_precision"] for r in results if "citation_precision" in r]
        cit_r = [r["citation_recall"] for r in results if "citation_recall" in r]
        if cit_p:
            entry["citation_precision"] = round(_mean(cit_p), 4)
        if cit_r:
            entry["citation_recall"] = round(_mean(cit_r), 4)

        generation_metrics[mode_name] = entry
        print(f"  {mode_name}: EM={em_l:.4f}  answered={len(answered)}/{len(results)}  lat={lat:.0f}ms")

    _save_json(run_dir / "generation_metrics.json", generation_metrics)

    # --- Agentic metrics (E5-specific) ---
    if agentic_results:
        answerable = [r for r in agentic_results if not r.get("unanswerable", False)]
        unanswerable = [r for r in agentic_results if r.get("unanswerable", False)]

        # Success@N
        s_at_1 = 0
        s_at_2 = 0
        for r in answerable:
            if r["em_loose"] >= 1.0:
                s_at_2 += 1
                if r.get("hops", 1) == 1:
                    s_at_1 += 1

        n_answerable = len(answerable) if answerable else 1

        # KB-Coverage Accuracy (unanswerable correctly abstained)
        true_negatives = sum(1 for r in unanswerable if r.get("abstained", False))
        kb_coverage_acc = true_negatives / len(unanswerable) if unanswerable else 0.0

        # False-Negative Rate (answerable incorrectly abstained)
        false_negatives = sum(1 for r in answerable if r.get("abstained", False))
        fnr = false_negatives / n_answerable

        # Hop distribution
        hop_counts: dict[int, int] = {}
        for r in agentic_results:
            h = r.get("hops", 1)
            hop_counts[h] = hop_counts.get(h, 0) + 1

        # Tool call stats
        tool_calls_list = [r.get("tool_calls", 0) for r in agentic_results]
        total_ms_list = [r.get("total_ms", 0) for r in agentic_results]

        agentic_metrics = {
            "success_at_1": round(s_at_1 / n_answerable, 4),
            "success_at_2": round(s_at_2 / n_answerable, 4),
            "kb_coverage_accuracy": round(kb_coverage_acc, 4),
            "false_negative_rate": round(fnr, 4),
            "hop_distribution": {str(h): c for h, c in sorted(hop_counts.items())},
            "avg_tool_calls": round(_mean(tool_calls_list), 2),
            "avg_total_ms": round(_mean(total_ms_list), 1),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "n_true_negatives": true_negatives,
            "n_false_negatives": false_negatives,
        }

        # Latency percentiles
        if total_ms_list:
            sorted_lat = sorted(total_ms_list)
            agentic_metrics["p50_total_ms"] = sorted_lat[len(sorted_lat) // 2]
            agentic_metrics["p95_total_ms"] = sorted_lat[int(len(sorted_lat) * 0.95)]

        _save_json(run_dir / "agentic_metrics.json", agentic_metrics)
        print(f"  S@1={agentic_metrics['success_at_1']:.4f}  "
              f"S@2={agentic_metrics['success_at_2']:.4f}  "
              f"KB-Cov={kb_coverage_acc:.4f}  FNR={fnr:.4f}")
        print(f"  Hops: {hop_counts}  AvgToolCalls={agentic_metrics['avg_tool_calls']}")

    # --- Gate/decision-aware metrics ---
    if agentic_results:
        predictions = [r["prediction"] for r in agentic_results]
        gold_answers = [r["gold_answers"] for r in agentic_results]
        answered_mask = [not r.get("abstained", False) for r in agentic_results]
        confidences = [r.get("confidence", 1.0) for r in agentic_results]
        em_labels = [int(em_loose(r["prediction"], r["gold_answers"])) for r in agentic_results]

        sel_acc = selective_accuracy(predictions, gold_answers, answered_mask)
        cov = coverage(answered_mask)
        au = auprc(em_labels, confidences)
        cal = ece(em_labels, confidences)

        gate_metrics = {
            "selective_accuracy": round(sel_acc, 4),
            "coverage": round(cov, 4),
            "auprc": round(au, 4),
            "ece": round(cal, 4),
            "threshold": E5_GATE_THRESHOLD,
            "n_answered": sum(answered_mask),
            "n_abstained": sum(1 for a in answered_mask if not a),
            "n_total": len(agentic_results),
        }
        _save_json(run_dir / "gate_metrics.json", gate_metrics)
        print(f"  Gate: SelAcc={sel_acc:.4f}  Cov={cov:.4f}  AUPRC={au:.4f}  ECE={cal:.4f}")

    # --- MIRAGE metrics (answered queries only) ---
    base_results = gen_results.get("base", [])
    oracle_results = gen_results.get("oracle", [])
    if base_results and oracle_results and agentic_results:
        base_by_qid = {r["query_id"]: r["em_loose"] for r in base_results}
        oracle_by_qid = {r["query_id"]: r["em_loose"] for r in oracle_results}

        answered = [r for r in agentic_results if not r.get("abstained", False)]
        if answered:
            base_labels = [base_by_qid.get(r["query_id"], 0.0) for r in answered]
            oracle_labels = [oracle_by_qid.get(r["query_id"], 0.0) for r in answered]
            mixed_labels = [r["em_loose"] for r in answered]

            m = compute_mirage_metrics(base_labels, oracle_labels, mixed_labels)
            mirage_metrics = {mk: round(mv, 4) for mk, mv in m.items()}
            mirage_metrics["n_answered"] = len(answered)
            _save_json(run_dir / "mirage_metrics.json", mirage_metrics)
            print(f"  MIRAGE: NV={m['NV']:.4f}  CA={m['CA']:.4f}  "
                  f"CI={m['CI']:.4f}  CM={m['CM']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="E5 — Agentic Multi-Hop RAG")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--partial", action="store_true", help=f"{E2_EVAL_N} eval questions")
    group.add_argument("--smoke", type=int, metavar="N", help="Smoke test with N questions")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--phase",
        choices=["indexing", "generation", "judge", "aggregation", "all"],
        default="all",
        help="Run only a specific phase (default: all)",
    )
    args = parser.parse_args()

    # Load data
    print("Loading MIRAGE dataset...")
    dataset = load_dataset()
    doc_pool = load_doc_pool()
    oracle = load_oracle()

    # Subset sizes
    if args.smoke:
        n_index = args.smoke * 5
        n_eval = args.smoke
    else:
        n_index = E2_INDEX_N
        n_eval = E2_EVAL_N

    # Select subsets
    index_dataset, index_doc_pool, index_oracle = select_partial_subset(
        dataset, doc_pool, oracle, n_index,
    )
    eval_dataset, _, _ = select_partial_subset(
        index_dataset, index_doc_pool, index_oracle, n_eval,
    )
    eval_oracle: dict[str, dict] = {
        q["query_id"]: index_oracle[q["query_id"]]
        for q in eval_dataset if q["query_id"] in index_oracle
    }
    n_answerable = len(eval_dataset)

    # Add unanswerable questions
    n_unanswerable = E3_N_UNANSWERABLE if not args.smoke else min(2, E3_N_UNANSWERABLE)
    indexed_qids = {q["query_id"] for q in index_dataset}
    unanswerable_qs = _select_unanswerable_questions(dataset, indexed_qids, n_unanswerable)
    eval_dataset = eval_dataset + unanswerable_qs
    for q in unanswerable_qs:
        qid = q["query_id"]
        if qid in oracle:
            eval_oracle[qid] = oracle[qid]

    mode_label = f"smoke_{n_answerable}" if args.smoke else f"partial_{E2_EVAL_N}"
    collection_name = f"{MIRAGE_COLLECTION_NAME}_e2_qwen3"

    print(f"Mode: {mode_label}")
    print(f"K={E5_K}  Threshold={E5_GATE_THRESHOLD}  MaxHops={2}")
    print(f"Model: {OPENROUTER_MODEL}")
    print(f"Embed: {OPENROUTER_EMBED_MODEL}")
    print(f"Judge: {E5_JUDGE_MODEL}")
    print(f"Eval: {n_answerable} answerable + {len(unanswerable_qs)} unanswerable "
          f"= {len(eval_dataset)}")
    print(f"Phase: {args.phase}")

    # Run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / "e5" / f"{timestamp}_{mode_label}"
    if args.resume:
        e5_dir = RUNS_DIR / "e5"
        if e5_dir.exists():
            candidates = sorted(
                [d for d in e5_dir.iterdir() if d.is_dir() and mode_label in d.name],
                reverse=True,
            )
            if candidates:
                run_dir = candidates[0]
                print(f"Resuming from: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Try to reuse base/oracle from previous experiments
    _reuse_base_oracle(run_dir, mode_label, n_answerable)

    # Save config
    config = {
        "experiment": "E5",
        "mode": mode_label,
        "model": OPENROUTER_MODEL,
        "embed_model": OPENROUTER_EMBED_MODEL,
        "judge_model": E5_JUDGE_MODEL,
        "k": E5_K,
        "gate_threshold": E5_GATE_THRESHOLD,
        "max_hops": 2,
        "agent_workers": E5_AGENT_WORKERS,
        "collection_name": collection_name,
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

    bm25_index = None
    bm25_pool: list[dict] = []
    embed_fn = None

    if phase in ("indexing", "all"):
        bm25_index, bm25_pool, embed_fn = run_indexing_phase(
            index_doc_pool, collection_name, OPENROUTER_EMBED_MODEL,
        )
    elif phase in ("generation",):
        # Need indices for agentic generation
        bm25_index, bm25_pool, embed_fn = run_indexing_phase(
            index_doc_pool, collection_name, OPENROUTER_EMBED_MODEL,
        )

    if phase in ("generation", "all"):
        assert embed_fn is not None
        gen_results = run_generation_phase(
            eval_dataset, eval_oracle, run_dir, collection_name,
            embed_fn, bm25_index, bm25_pool, E5_AGENT_WORKERS,
        )
    elif phase in ("aggregation", "judge"):
        gen_results = {}
        for name, key in [("e5_base", "base"), ("e5_oracle", "oracle"),
                          ("e5_agentic_mixed", "agentic_mixed")]:
            path = _jsonl_path(run_dir, name)
            if path.exists():
                gen_results[key] = _load_jsonl(path)
                print(f"  Loaded {len(gen_results[key])} {key} results")
    else:
        gen_results = {}

    if phase in ("judge", "all"):
        run_judge_phase(run_dir)

    if phase in ("aggregation", "all"):
        run_aggregation_phase(gen_results, run_dir)

    print(f"\nResults saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
