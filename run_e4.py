"""E4 — Local vs Online Comparison experiment runner.

Compares the Local-Best pipeline (from E3) against an online cloud-based
generator.  Retrieval is **shared** — only the generator model differs.

Usage:
    .venv\\Scripts\\python.exe run_e4.py --partial
    .venv\\Scripts\\python.exe run_e4.py --partial --resume
    .venv\\Scripts\\python.exe run_e4.py --smoke 10
    .venv\\Scripts\\python.exe run_e4.py --partial --phase judge --resume
"""

import argparse
import copy
from functools import partial
import json
import platform
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.config import (
    E2_EVAL_N,
    E2_INDEX_N,
    E3_ABSTAIN_MESSAGE,
    E3_LOCAL_BEST_CONFIG,
    E3_LOCAL_BEST_K,
    E3_N_UNANSWERABLE,
    E4_JUDGE_MODEL,
    E4_LOCAL_BEST_THRESHOLD,
    E4_ONLINE_EMBED_MODEL,
    E4_ONLINE_MODEL,
    MIRAGE_COLLECTION_NAME,
    OPENROUTER_EMBED_MODEL,
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
    citation_precision,
    citation_recall,
    compute_mirage_metrics,
    coverage,
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
PIPELINES = ["local", "online"]
MODES = ["base", "oracle", "mixed"]


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
    for pkg in ["chromadb", "httpx", "colorama"]:
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
# E3 reuse — copy retrieval + local generation artifacts
# ---------------------------------------------------------------------------

def _find_latest_e3_run(mode_label: str) -> Path | None:
    e3_dir = RUNS_DIR / "e3"
    if not e3_dir.exists():
        return None
    search_label = "partial_100" if "partial" in mode_label else mode_label
    candidates = sorted(
        [d for d in e3_dir.iterdir() if d.is_dir() and search_label in d.name],
        reverse=True,
    )
    return candidates[0] if candidates else None



# ---------------------------------------------------------------------------
# Disk-loading helpers
# ---------------------------------------------------------------------------

def _load_retrieval_from_disk(run_dir: Path, k: int, pipeline: str = "") -> list[dict]:
    suffix = f"_{pipeline}" if pipeline else ""
    ret_path = run_dir / "retrieval" / f"retrieved_k{k}{suffix}.jsonl"
    if not ret_path.exists():
        raise FileNotFoundError(f"Retrieval results not found: {ret_path}")
    rows = _load_jsonl(ret_path)
    print(f"  Loaded {len(rows)} retrieval results")
    return rows


def _load_generation_from_disk(run_dir: Path) -> dict[str, list[dict]]:
    """Load all generation results (local/online x base/oracle/mixed)."""
    gen_results: dict[str, list[dict]] = {}
    for pipeline in PIPELINES:
        for mode in MODES:
            key = f"{pipeline}_{mode}"
            gen_path = _jsonl_path(run_dir, f"e4_{key}")
            if gen_path.exists():
                gen_results[key] = _load_jsonl(gen_path)
                print(f"  Loaded {len(gen_results[key])} {key} results")
    return gen_results


# ---------------------------------------------------------------------------
# Phase A: Shared Retrieval (identical to E3)
# ---------------------------------------------------------------------------

def run_retrieval_phase(
    eval_dataset: list[dict],
    index_doc_pool: list[dict],
    run_dir: Path,
    k: int,
    collection_name: str,
    embed_model: str,
    pipeline_label: str,
) -> list[dict]:
    """Run vector retrieval for a specific pipeline."""
    print(f"\n{'='*60}")
    print(f"PHASE A: Retrieval [{pipeline_label}] ({embed_model}, k={k})")
    print(f"{'='*60}")

    ret_path = run_dir / "retrieval" / f"retrieved_k{k}_{pipeline_label}.jsonl"
    if ret_path.exists():
        existing = _load_jsonl(ret_path)
        if len(existing) >= len(eval_dataset):
            print(f"  Already complete ({len(existing)} queries)")
            return existing

    from src.embed_openrouter import (
        batch_search_with_custom_embeddings,
        embed_texts_openrouter,
        index_mirage_with_custom_embeddings,
    )

    embed_batch = 10 if "8b" in embed_model else 25
    embed_fn = partial(embed_texts_openrouter, model=embed_model, batch_size=embed_batch)
    print(f"  Indexing {len(index_doc_pool)} chunks into '{collection_name}'...", flush=True)
    t0 = time.perf_counter()
    n_indexed = index_mirage_with_custom_embeddings(
        index_doc_pool, collection_name, embed_fn=embed_fn, batch_size=embed_batch,
    )
    print(f"  ChromaDB: {n_indexed} chunks in {time.perf_counter()-t0:.1f}s", flush=True)

    # --- Batch-embed all queries, then retrieve ---
    results: list[dict] = []
    t0 = time.perf_counter()
    queries = [q["query"] for q in eval_dataset]
    print(f"  Batch-embedding {len(queries)} queries...")
    batch_results = batch_search_with_custom_embeddings(
        collection_name, queries, n_results=k, embed_fn=embed_fn,
    )

    for i, question in enumerate(eval_dataset):
        qid = question["query_id"]
        result = batch_results[i]

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

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Recall@{k}={_mean([r['recall'] for r in results]):.4f}  "
          f"nDCG@{k}={_mean([r['ndcg'] for r in results]):.4f}  "
          f"MRR={_mean([r['mrr'] for r in results]):.4f}")

    ret_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ret_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return results


# ---------------------------------------------------------------------------
# Phase B: Generation (local/online x base/oracle/mixed)
# ---------------------------------------------------------------------------

def _run_generation_task(
    pipeline: str,
    model: str,
    mode: str,
    dataset: list[dict],
    oracle_map: dict[str, dict],
    retrieval_results: list[dict] | None,
    run_dir: Path,
    threshold: float,
    gate_method: str,
) -> list[dict]:
    """Run generation for one (pipeline, mode) combination."""
    task_name = f"e4_{pipeline}_{mode}"
    checkpoint_path = _jsonl_path(run_dir, task_name)
    completed = _load_checkpoint(checkpoint_path)
    total = len(dataset)

    if len(completed) >= total:
        print(f"  {task_name}: already complete ({total}/{total})")
        return list(completed.values())

    if completed:
        print(f"  {task_name}: resuming from {len(completed)}/{total}")
    else:
        print(f"  {task_name}: starting {total} queries (model={model})")

    results = list(completed.values())
    errors = 0
    empty_count = 0

    for i, question in enumerate(dataset):
        qid = question["query_id"]
        if qid in completed:
            continue

        # Gate decision (mixed mode only)
        abstained = False
        confidence = 1.0
        if mode == "mixed" and retrieval_results:
            distances = retrieval_results[i]["distances"]
            abstained, confidence = should_abstain(
                distances, threshold, method=gate_method,
            )

        if abstained:
            prediction = E3_ABSTAIN_MESSAGE
            gen_ms = 0
        else:
            # Build prompt per mode
            if mode == "base":
                messages = build_base_prompt(question["query"])
            elif mode == "oracle":
                chunk = oracle_map.get(qid, {}).get("doc_chunk", "")
                messages = build_oracle_prompt(question["query"], chunk)
            elif mode == "mixed":
                assert retrieval_results is not None
                chunks = retrieval_results[i]["documents"]
                messages = build_mixed_prompt_cited(question["query"], chunks)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            t0 = time.perf_counter()
            prediction = ""
            for attempt in range(1, MAX_GEN_RETRIES + 1):
                try:
                    prediction = call_openrouter(messages, model=model)
                    if prediction:  # non-empty → done
                        break
                    # Empty prediction — retry after brief pause
                    if attempt < MAX_GEN_RETRIES:
                        time.sleep(1)
                except RuntimeError as exc:
                    if attempt < MAX_GEN_RETRIES:
                        wait = min(2 ** attempt, 30)
                        time.sleep(wait)
                    else:
                        errors += 1
                        prediction = f"ERROR: {exc}"
            gen_ms = int((time.perf_counter() - t0) * 1000)

            if not prediction and not (prediction or "").startswith("ERROR:"):
                empty_count += 1

        answers = question["answer"]
        is_unanswerable = question.get("unanswerable_in_corpus", False)

        # Correct abstention on unanswerable question = correct decision
        if abstained and is_unanswerable:
            em_score = 1.0
        else:
            em_score = em_loose(prediction or "", answers)

        record: dict = {
            "query_id": qid,
            "query": question["query"],
            "gold_answers": answers,
            "prediction": prediction,
            "em_loose": em_score,
            "generation_ms": gen_ms,
            "pipeline": pipeline,
            "model": model,
            "mode": mode,
            "abstained": abstained,
            "confidence": round(confidence, 6),
            "unanswerable": is_unanswerable,
        }

        # Citation metrics (mixed mode, answered only)
        if mode == "mixed" and not abstained and retrieval_results:
            supports = retrieval_results[i].get("supports", [])
            gold_indices = [j + 1 for j, s in enumerate(supports) if s == 1]
            cited = parse_citations(prediction or "")
            record["cited_indices"] = cited
            record["gold_indices"] = gold_indices
            record["citation_precision"] = citation_precision(cited, gold_indices)
            record["citation_recall"] = citation_recall(cited, gold_indices)

        _append_jsonl(checkpoint_path, record)
        results.append(record)
        completed[qid] = record

        done = len(completed)
        if done % 10 == 0 or done == total:
            em_so_far = _mean([r["em_loose"] for r in results])
            n_abs = sum(1 for r in results if r.get("abstained", False))
            print(f"    {task_name}: {done}/{total}  EM={em_so_far:.4f}  "
                  f"abstained={n_abs}  errors={errors}  empty={empty_count}")

    return results


def run_generation_phase(
    eval_dataset: list[dict],
    oracle_map: dict[str, dict],
    retrieval_by_pipeline: dict[str, list[dict]],
    run_dir: Path,
    gate_method: str,
) -> dict[str, list[dict]]:
    """Run generation for both pipelines x all modes (parallel)."""
    print(f"\n{'='*60}")
    print("PHASE B: Generation (Local vs Online x Base/Oracle/Mixed)")
    print(f"{'='*60}")

    # Build task list: (key, pipeline, model, mode)
    tasks: list[tuple[str, str, str, str]] = []
    for pipeline, model in [("local", OPENROUTER_MODEL), ("online", E4_ONLINE_MODEL)]:
        for mode in MODES:
            tasks.append((f"{pipeline}_{mode}", pipeline, model, mode))

    n_workers = min(6, len(tasks))
    print(f"  {len(tasks)} generation tasks, {n_workers} workers")

    all_gen: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for key, pipeline, model, mode in tasks:
            ret = retrieval_by_pipeline.get(pipeline) if mode == "mixed" else None
            future = executor.submit(
                _run_generation_task,
                pipeline, model, mode, eval_dataset, oracle_map, ret,
                run_dir, E4_LOCAL_BEST_THRESHOLD, gate_method,
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
# Phase C: Judge (mixed-mode predictions, both pipelines)
# ---------------------------------------------------------------------------

def _judge_single_query(
    qid: str,
    g: dict,
    contexts: list[str],
    judge_model: str,
) -> dict:
    """Judge a single query."""
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


def _save_judge_summary(run_dir: Path) -> None:
    """Aggregate and save judge metrics for both pipelines."""
    judge_metrics: dict = {}
    for pipeline in PIPELINES:
        judge_path = _jsonl_path(run_dir, f"judge_{pipeline}")
        if judge_path.exists():
            rows = _load_jsonl(judge_path)
            agg = aggregate_judge_scores(rows)
            judge_metrics[pipeline] = agg
            print(f"  {pipeline}: faith={agg['faithfulness']}  "
                  f"ground={agg['groundedness']}  relev={agg['answer_relevance']}  "
                  f"correct={agg['semantic_correctness']}  judged={agg['n_judged']}")
    if judge_metrics:
        _save_json(run_dir / "judge_metrics.json", judge_metrics)


def run_judge_phase(run_dir: Path, k: int) -> None:
    """Run LLM-as-Judge on mixed-mode predictions for both pipelines."""
    print(f"\n{'='*60}")
    print(f"PHASE C: LLM-as-Judge ({E4_JUDGE_MODEL})")
    print(f"{'='*60}")

    judge_tasks: list[tuple[str, str, dict, list[str], Path]] = []
    for pipeline in PIPELINES:
        gen_path = _jsonl_path(run_dir, f"e4_{pipeline}_mixed")
        if not gen_path.exists():
            print(f"  Skipping {pipeline}: mixed generation not found")
            continue

        try:
            ret_rows = _load_retrieval_from_disk(run_dir, k, pipeline)
        except FileNotFoundError:
            print(f"  Skipping {pipeline}: retrieval not found")
            continue
        ret_by_qid = {r["query_id"]: r for r in ret_rows}

        judge_path = _jsonl_path(run_dir, f"judge_{pipeline}")
        judged = _load_checkpoint(judge_path)
        gen_rows = _load_jsonl(gen_path)

        for g in gen_rows:
            qid = g["query_id"]
            if qid in judged:
                continue
            ret = ret_by_qid.get(qid)
            contexts = ret["documents"] if ret else []
            judge_tasks.append((pipeline, qid, g, contexts, judge_path))

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
        for pipeline, qid, g, contexts, judge_path in judge_tasks:
            future = executor.submit(
                _judge_single_query, qid, g, contexts, E4_JUDGE_MODEL,
            )
            futures[future] = (pipeline, qid, judge_path)

        for future in as_completed(futures):
            pipeline, qid, judge_path = futures[future]
            try:
                row = future.result()
                _append_jsonl(judge_path, row)
                completed_count += 1
                if row["faithfulness"] is None and "SKIPPED" not in row.get("judge_raw", ""):
                    errors += 1
                if completed_count % 10 == 0 or completed_count == len(judge_tasks):
                    print(f"    Judged {completed_count}/{len(judge_tasks)} ({errors} errors)")
            except Exception as exc:
                print(f"  ERROR judging {pipeline}/{qid}: {exc}")
                traceback.print_exc()

    _save_judge_summary(run_dir)


# ---------------------------------------------------------------------------
# Phase D: Aggregation
# ---------------------------------------------------------------------------

def run_aggregation_phase(
    gen_results: dict[str, list[dict]],
    retrieval_by_pipeline: dict[str, list[dict]],
    run_dir: Path,
) -> None:
    """Compute all metrics and save to JSON files."""
    print(f"\n{'='*60}")
    print("PHASE D: Aggregation")
    print(f"{'='*60}")

    k = E3_LOCAL_BEST_K

    # --- Retrieval metrics (per pipeline) ---
    ret_metrics: dict = {}
    for pipeline in PIPELINES:
        retrieval_results = retrieval_by_pipeline.get(pipeline, [])
        if not retrieval_results:
            continue
        ret_metrics[pipeline] = {
            f"recall@{k}": round(_mean([r["recall"] for r in retrieval_results]), 4),
            f"precision@{k}": round(_mean([r["precision"] for r in retrieval_results]), 4),
            f"ndcg@{k}": round(_mean([r["ndcg"] for r in retrieval_results]), 4),
            "mrr": round(_mean([r["mrr"] for r in retrieval_results]), 4),
            "n_queries": len(retrieval_results),
        }
        print(f"  Retrieval [{pipeline}]: Recall@{k}={ret_metrics[pipeline][f'recall@{k}']}  "
              f"nDCG@{k}={ret_metrics[pipeline][f'ndcg@{k}']}  "
              f"MRR={ret_metrics[pipeline]['mrr']}")
    _save_json(run_dir / "retrieval_metrics.json", ret_metrics)

    # --- Per-pipeline generation metrics ---
    generation_metrics: dict = {}
    for pipeline in PIPELINES:
        p_metrics: dict = {}
        for mode in MODES:
            key = f"{pipeline}_{mode}"
            results = gen_results.get(key, [])
            if not results:
                continue

            answered = [r for r in results if not r.get("abstained", False)]
            em_l = _mean([r["em_loose"] for r in results])
            lat = _mean([r["generation_ms"] for r in answered]) if answered else 0.0

            # Selective accuracy & coverage
            sel_acc = selective_accuracy(
                [r["prediction"] for r in results],
                [r["gold_answers"] for r in results],
                [not r.get("abstained", False) for r in results],
            )
            cov = coverage([not r.get("abstained", False) for r in results])

            entry: dict = {
                "em_loose": round(em_l, 4),
                "selective_accuracy": round(sel_acc, 4),
                "coverage": round(cov, 4),
                "avg_generation_ms": round(lat, 1),
                "n_total": len(results),
                "n_answered": len(answered),
                "n_abstained": len(results) - len(answered),
            }

            # Latency percentiles (answered only)
            if answered:
                latencies = sorted([r["generation_ms"] for r in answered])
                p50_idx = len(latencies) // 2
                p95_idx = int(len(latencies) * 0.95)
                entry["p50_latency_ms"] = latencies[p50_idx]
                entry["p95_latency_ms"] = latencies[min(p95_idx, len(latencies) - 1)]

            # Citation metrics (mixed answered only)
            if mode == "mixed":
                cit_p = [r["citation_precision"] for r in results if "citation_precision" in r]
                cit_r = [r["citation_recall"] for r in results if "citation_recall" in r]
                if cit_p:
                    entry["citation_precision"] = round(_mean(cit_p), 4)
                if cit_r:
                    entry["citation_recall"] = round(_mean(cit_r), 4)

            p_metrics[mode] = entry
            print(f"  {pipeline}/{mode}: EM={em_l:.4f}  selAcc={sel_acc:.4f}  cov={cov:.2f}  "
                  f"answered={len(answered)}/{len(results)}  lat={lat:.0f}ms")

        generation_metrics[pipeline] = p_metrics

    _save_json(run_dir / "generation_metrics.json", generation_metrics)

    # --- MIRAGE metrics (per pipeline, answered mixed queries only) ---
    mirage_metrics: dict = {}
    for pipeline in PIPELINES:
        base_results = gen_results.get(f"{pipeline}_base", [])
        oracle_results = gen_results.get(f"{pipeline}_oracle", [])
        mixed_results = gen_results.get(f"{pipeline}_mixed", [])

        if not (base_results and oracle_results and mixed_results):
            continue

        base_by_qid = {r["query_id"]: r["em_loose"] for r in base_results}
        oracle_by_qid = {r["query_id"]: r["em_loose"] for r in oracle_results}

        answered = [r for r in mixed_results if not r.get("abstained", False)]
        if not answered:
            continue

        base_labels = [base_by_qid.get(r["query_id"], 0.0) for r in answered]
        oracle_labels = [oracle_by_qid.get(r["query_id"], 0.0) for r in answered]
        mixed_labels = [r["em_loose"] for r in answered]

        m = compute_mirage_metrics(base_labels, oracle_labels, mixed_labels)
        mirage_metrics[pipeline] = {mk: round(mv, 4) for mk, mv in m.items()}
        mirage_metrics[pipeline]["n_answered"] = len(answered)

        print(f"  MIRAGE {pipeline}: NV={m['NV']:.4f}  CA={m['CA']:.4f}  "
              f"CI={m['CI']:.4f}  CM={m['CM']:.4f}")

    _save_json(run_dir / "mirage_metrics.json", mirage_metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="E4 — Local vs Online Comparison")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--partial", action="store_true", help=f"{E2_EVAL_N} eval questions")
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
    k = E3_LOCAL_BEST_K
    gate_method = "cosine"  # Qwen3 vector distances (lower = more similar)

    print(f"Mode: {mode_label}")
    print(f"Local-Best: {E3_LOCAL_BEST_CONFIG} (k={k})")
    print(f"Gate: {gate_method}, threshold={E4_LOCAL_BEST_THRESHOLD}")
    print(f"Local model: {OPENROUTER_MODEL}")
    print(f"Online model: {E4_ONLINE_MODEL}")
    print(f"Judge model: {E4_JUDGE_MODEL}")
    print(f"Eval: {n_answerable} answerable + {len(unanswerable_qs)} unanswerable "
          f"= {len(eval_dataset)}")
    print(f"Phase: {args.phase}")

    # Run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / "e4" / f"{timestamp}_{mode_label}"
    if args.resume:
        e4_dir = RUNS_DIR / "e4"
        if e4_dir.exists():
            candidates = sorted(
                [d for d in e4_dir.iterdir() if d.is_dir() and mode_label in d.name],
                reverse=True,
            )
            if candidates:
                run_dir = candidates[0]
                print(f"Resuming from: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Reuse E3 local retrieval (Qwen3-4b, same as E3 Local-Best)
    e3_run = _find_latest_e3_run(mode_label)
    if e3_run:
        e3_ret_dir = e3_run / "retrieval"
        if e3_ret_dir.exists():
            for ret_file in e3_ret_dir.glob("retrieved_k*.jsonl"):
                target = run_dir / "retrieval" / f"retrieved_k{k}_local.jsonl"
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    rows = _load_jsonl(ret_file)
                    with open(target, "w", encoding="utf-8") as f:
                        for r in rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print(f"  Reused E3 local retrieval: {len(rows)} queries")
                break  # only need first match

    # Collection names — shared when using the same embed model
    local_collection = f"{MIRAGE_COLLECTION_NAME}_e2_qwen3"
    if OPENROUTER_EMBED_MODEL == E4_ONLINE_EMBED_MODEL:
        online_collection = local_collection  # share retrieval index
    else:
        online_collection = f"{MIRAGE_COLLECTION_NAME}_e4_online"

    # Save config
    config = {
        "experiment": "E4",
        "mode": mode_label,
        "local_model": OPENROUTER_MODEL,
        "local_embed_model": OPENROUTER_EMBED_MODEL,
        "online_model": E4_ONLINE_MODEL,
        "online_embed_model": E4_ONLINE_EMBED_MODEL,
        "judge_model": E4_JUDGE_MODEL,
        "local_best_config": E3_LOCAL_BEST_CONFIG,
        "local_best_k": k,
        "local_collection": local_collection,
        "online_collection": online_collection,
        "gate_method": gate_method,
        "threshold": E4_LOCAL_BEST_THRESHOLD,
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
        # Run local retrieval
        local_retrieval = run_retrieval_phase(
            eval_dataset, index_doc_pool, run_dir, k,
            local_collection, OPENROUTER_EMBED_MODEL, "local",
        )
        if local_collection == online_collection:
            # Shared embeddings — copy local retrieval as online
            online_ret_path = run_dir / "retrieval" / f"retrieved_k{k}_online.jsonl"
            if not online_ret_path.exists():
                online_ret_path.parent.mkdir(parents=True, exist_ok=True)
                with open(online_ret_path, "w", encoding="utf-8") as f:
                    for r in local_retrieval:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"  Shared retrieval: copied local -> online ({len(local_retrieval)} queries)")
            online_retrieval = local_retrieval
        else:
            online_retrieval = run_retrieval_phase(
                eval_dataset, index_doc_pool, run_dir, k,
                online_collection, E4_ONLINE_EMBED_MODEL, "online",
            )
    else:
        local_retrieval = _load_retrieval_from_disk(run_dir, k, "local")
        online_retrieval = _load_retrieval_from_disk(run_dir, k, "online")

    retrieval_by_pipeline = {"local": local_retrieval, "online": online_retrieval}

    if phase in ("generation", "all"):
        gen_results = run_generation_phase(
            eval_dataset, eval_oracle, retrieval_by_pipeline, run_dir, gate_method,
        )
    elif phase in ("aggregation", "judge"):
        gen_results = _load_generation_from_disk(run_dir)
    else:
        gen_results = {}

    if phase in ("judge", "all"):
        run_judge_phase(run_dir, k)

    if phase in ("aggregation", "all"):
        run_aggregation_phase(gen_results, retrieval_by_pipeline, run_dir)

    print(f"\nResults saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
