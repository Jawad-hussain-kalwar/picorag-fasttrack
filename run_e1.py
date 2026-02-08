"""E1 — Vanilla Local RAG Baseline experiment runner.

Usage:
    .venv\\Scripts\\python.exe run_e1.py --partial          # 100 questions (dev)
    .venv\\Scripts\\python.exe run_e1.py --full              # 7,560 questions
    .venv\\Scripts\\python.exe run_e1.py --full --resume     # resume interrupted run
    .venv\\Scripts\\python.exe run_e1.py --partial --phase retrieval  # retrieval only
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import (
    CHROMA_PERSIST_DIR,
    E1_K_VALUES,
    E1_PARTIAL_N,
    MIRAGE_COLLECTION_NAME,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_TEMPERATURE,
    RUNS_DIR,
)
from src.generate import (
    build_base_prompt,
    build_mixed_prompt,
    build_oracle_prompt,
    call_openrouter,
)
from src.metrics import (
    compute_mirage_metrics,
    em_loose,
    em_strict,
    f1_score,
    mrr,
    ndcg_at_k,
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
from src.retrieve import get_client, get_collection, index_mirage_pool, search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonl_path(run_dir: Path, name: str) -> Path:
    return run_dir / "samples" / f"{name}.jsonl"


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """Load completed query_ids from a JSONL checkpoint file."""
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


# ---------------------------------------------------------------------------
# Phase A: Retrieval evaluation (local, no API calls)
# ---------------------------------------------------------------------------

def run_retrieval_phase(
    dataset: list[dict],
    doc_pool: list[dict],
    gold_lookup: dict[str, list[int]],
    run_dir: Path,
    k_values: list[int],
    collection_name: str,
) -> dict[int, list[dict]]:
    """Index chunks, run retrieval for all questions, compute retrieval metrics.

    Returns: {k: [per-query retrieval results]} for use in generation phase.
    """
    print(f"\n{'='*60}")
    print("PHASE A: Retrieval Evaluation")
    print(f"{'='*60}")

    # Index
    client = get_client(str(CHROMA_PERSIST_DIR))
    collection = get_collection(client, collection_name)

    t0 = time.perf_counter()
    indexed = index_mirage_pool(collection, doc_pool)
    index_time = time.perf_counter() - t0
    print(f"Indexed {indexed} chunks in {index_time:.1f}s")

    all_retrieval: dict[int, list[dict]] = {}
    retrieval_metrics: dict[str, dict] = {}

    max_k = max(k_values)

    # Run retrieval once at max_k, then slice for smaller k
    print(f"\nRetrieving top-{max_k} for {len(dataset)} questions...")
    t0 = time.perf_counter()

    raw_results: list[dict] = []
    for i, question in enumerate(dataset):
        qid = question["query_id"]
        result = search(collection, question["query"], n_results=max_k)

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

        if (i + 1) % 500 == 0 or i + 1 == len(dataset):
            print(f"  Retrieved {i + 1}/{len(dataset)}")

    retrieval_time = time.perf_counter() - t0
    print(f"Retrieval done in {retrieval_time:.1f}s")

    # Compute metrics per k
    for k in k_values:
        k_results = []
        recalls, precisions, ndcgs, mrrs = [], [], [], []

        for raw, question in zip(raw_results, dataset):
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

        all_retrieval[k] = k_results

        metrics = {
            f"Recall@{k}": _mean(recalls),
            f"Precision@{k}": _mean(precisions),
            f"nDCG@{k}": _mean(ndcgs),
            f"MRR@{k}": _mean(mrrs),
        }
        retrieval_metrics[f"k={k}"] = metrics
        print(f"\n  k={k}: Recall={metrics[f'Recall@{k}']:.4f}  "
              f"Precision={metrics[f'Precision@{k}']:.4f}  "
              f"nDCG={metrics[f'nDCG@{k}']:.4f}  "
              f"MRR={metrics[f'MRR@{k}']:.4f}")

        # Save per-query retrieval results
        ret_path = run_dir / "retrieval" / f"retrieved_k{k}.jsonl"
        ret_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ret_path, "w", encoding="utf-8") as f:
            for rec in k_results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    retrieval_metrics["_meta"] = {
        "index_time_s": round(index_time, 2),
        "retrieval_time_s": round(retrieval_time, 2),
        "total_chunks_indexed": indexed,
        "total_questions": len(dataset),
    }
    _save_json(run_dir / "retrieval_metrics.json", retrieval_metrics)

    del client
    return all_retrieval


# ---------------------------------------------------------------------------
# Phase B: Generation evaluation (API calls, checkpointed)
# ---------------------------------------------------------------------------

def _run_generation_mode(
    name: str,
    dataset: list[dict],
    oracle: dict[str, dict],
    retrieval_results: list[dict] | None,
    run_dir: Path,
    mode: str,
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

        # Build prompt
        if mode == "base":
            messages = build_base_prompt(question["query"])
        elif mode == "oracle":
            chunk_text = oracle[qid]["doc_chunk"]
            messages = build_oracle_prompt(question["query"], chunk_text)
        elif mode == "mixed":
            chunks = retrieval_results[i]["documents"]
            messages = build_mixed_prompt(question["query"], chunks)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Call LLM
        t0 = time.perf_counter()
        try:
            prediction = call_openrouter(messages)
        except RuntimeError as exc:
            errors += 1
            prediction = f"ERROR: {exc}"
        gen_ms = int((time.perf_counter() - t0) * 1000)

        # Compute metrics
        answers = question["answer"]
        record = {
            "query_id": qid,
            "query": question["query"],
            "gold_answers": answers,
            "prediction": prediction,
            "em_loose": em_loose(prediction, answers),
            "em_strict": em_strict(prediction, answers),
            "f1": f1_score(prediction, answers),
            "generation_ms": gen_ms,
            "mode": mode,
        }

        _append_jsonl(checkpoint_path, record)
        results.append(record)
        completed[qid] = record

        done = len(completed)
        if done % 50 == 0 or done == total:
            em_so_far = _mean([r["em_loose"] for r in results])
            print(f"    {name}: {done}/{total}  EM_loose={em_so_far:.4f}  errors={errors}")

    return results


def run_generation_phase(
    dataset: list[dict],
    oracle: dict[str, dict],
    all_retrieval: dict[int, list[dict]],
    run_dir: Path,
    k_values: list[int],
) -> dict[str, list[dict]]:
    """Run all generation modes. Returns {mode_name: [per-query results]}."""
    print(f"\n{'='*60}")
    print("PHASE B: Generation Evaluation")
    print(f"{'='*60}")

    all_results: dict[str, list[dict]] = {}

    # Base mode
    all_results["base"] = _run_generation_mode(
        "e1_base", dataset, oracle, None, run_dir, "base"
    )

    # Oracle mode
    all_results["oracle"] = _run_generation_mode(
        "e1_oracle", dataset, oracle, None, run_dir, "oracle"
    )

    # Mixed mode per k
    for k in k_values:
        all_results[f"mixed_k{k}"] = _run_generation_mode(
            f"e1_mixed_k{k}", dataset, oracle, all_retrieval[k], run_dir, "mixed"
        )

    return all_results


# ---------------------------------------------------------------------------
# Phase C: Aggregation
# ---------------------------------------------------------------------------

def run_aggregation_phase(
    gen_results: dict[str, list[dict]],
    k_values: list[int],
    run_dir: Path,
) -> None:
    """Aggregate per-query results into final metrics."""
    print(f"\n{'='*60}")
    print("PHASE C: Aggregation")
    print(f"{'='*60}")

    generation_metrics: dict[str, dict] = {}
    mirage_metrics: dict[str, dict] = {}

    # Generation metrics per mode
    for mode_name, results in gen_results.items():
        em_l = _mean([r["em_loose"] for r in results])
        em_s = _mean([r["em_strict"] for r in results])
        f1 = _mean([r["f1"] for r in results])
        lat = _mean([r["generation_ms"] for r in results])

        generation_metrics[mode_name] = {
            "EM_loose": round(em_l, 4),
            "EM_strict": round(em_s, 4),
            "F1": round(f1, 4),
            "avg_generation_ms": round(lat, 1),
            "n_queries": len(results),
        }
        print(f"  {mode_name}: EM_loose={em_l:.4f}  EM_strict={em_s:.4f}  F1={f1:.4f}")

    _save_json(run_dir / "generation_metrics.json", generation_metrics)

    # MIRAGE RAG metrics per k (requires base + oracle + mixed)
    base_results = gen_results["base"]
    oracle_results = gen_results["oracle"]

    # Build lookup by query_id for base and oracle
    base_by_qid = {r["query_id"]: r["em_loose"] for r in base_results}
    oracle_by_qid = {r["query_id"]: r["em_loose"] for r in oracle_results}

    for k in k_values:
        mixed_results = gen_results[f"mixed_k{k}"]
        # Align labels by query_id
        base_labels, oracle_labels, mixed_labels = [], [], []
        for r in mixed_results:
            qid = r["query_id"]
            base_labels.append(base_by_qid.get(qid, 0.0))
            oracle_labels.append(oracle_by_qid.get(qid, 0.0))
            mixed_labels.append(r["em_loose"])

        m = compute_mirage_metrics(base_labels, oracle_labels, mixed_labels)
        mirage_metrics[f"k={k}"] = {k_name: round(v, 4) for k_name, v in m.items()}
        print(f"  MIRAGE k={k}: NV={m['NV']:.4f}  CA={m['CA']:.4f}  "
              f"CI={m['CI']:.4f}  CM={m['CM']:.4f}")

    _save_json(run_dir / "mirage_metrics.json", mirage_metrics)


# ---------------------------------------------------------------------------
# Sysinfo
# ---------------------------------------------------------------------------

def _collect_sysinfo() -> dict:
    import importlib.metadata
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "packages": {
            pkg: importlib.metadata.version(pkg)
            for pkg in ["chromadb", "httpx", "colorama"]
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="E1 — Vanilla Local RAG Baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--partial", action="store_true", help="100 questions (dev)")
    group.add_argument("--full", action="store_true", help="7,560 questions")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--phase",
        choices=["retrieval", "generation", "all"],
        default="all",
        help="Run specific phase only (default: all)",
    )
    args = parser.parse_args()

    # Load data
    print("Loading MIRAGE dataset...")
    dataset = load_dataset()
    doc_pool = load_doc_pool()
    oracle = load_oracle()

    if args.partial:
        dataset, doc_pool, oracle = select_partial_subset(
            dataset, doc_pool, oracle, E1_PARTIAL_N
        )
        mode_label = f"partial_{E1_PARTIAL_N}"
        collection_name = f"{MIRAGE_COLLECTION_NAME}_partial"
    else:
        mode_label = "full"
        collection_name = MIRAGE_COLLECTION_NAME

    print(f"Mode: {mode_label}")
    print(f"Questions: {len(dataset)}")
    print(f"Doc pool chunks: {len(doc_pool)}")
    print(f"Oracle entries: {len(oracle)}")
    print(f"K values: {E1_K_VALUES}")
    print(f"Model: {OPENROUTER_MODEL}")

    gold_lookup = build_gold_lookup(doc_pool)

    # Setup run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / "e1" / f"{timestamp}_{mode_label}"
    if args.resume:
        # Find latest run_dir matching mode
        e1_dir = RUNS_DIR / "e1"
        if e1_dir.exists():
            candidates = sorted(
                [d for d in e1_dir.iterdir() if d.is_dir() and mode_label in d.name],
                reverse=True,
            )
            if candidates:
                run_dir = candidates[0]
                print(f"Resuming from: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "mode": mode_label,
        "model": OPENROUTER_MODEL,
        "temperature": OPENROUTER_TEMPERATURE,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "k_values": E1_K_VALUES,
        "n_questions": len(dataset),
        "n_chunks": len(doc_pool),
        "timestamp": timestamp,
    }
    _save_json(run_dir / "config.json", config)
    _save_json(run_dir / "sysinfo.json", _collect_sysinfo())

    # Phase A
    all_retrieval = {}
    if args.phase in ("retrieval", "all"):
        all_retrieval = run_retrieval_phase(
            dataset, doc_pool, gold_lookup, run_dir, E1_K_VALUES, collection_name
        )

    # Phase B
    gen_results = {}
    if args.phase in ("generation", "all"):
        # If resuming generation-only, load retrieval results from disk
        if not all_retrieval:
            all_retrieval = {}
            for k in E1_K_VALUES:
                ret_path = run_dir / "retrieval" / f"retrieved_k{k}.jsonl"
                if ret_path.exists():
                    records = []
                    for line in ret_path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            records.append(json.loads(line))
                    all_retrieval[k] = records
                else:
                    print(f"ERROR: Missing retrieval results for k={k}. Run retrieval first.")
                    return 1

        gen_results = run_generation_phase(
            dataset, oracle, all_retrieval, run_dir, E1_K_VALUES
        )

    # Phase C
    if args.phase == "all" and gen_results:
        run_aggregation_phase(gen_results, E1_K_VALUES, run_dir)

    print(f"\nResults saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
