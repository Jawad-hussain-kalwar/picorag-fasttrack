import json
from pathlib import Path
from typing import Any

from src.config import MIRAGE_DIR


def load_dataset(mirage_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load 7,560 QA pairs from dataset.json."""
    path = (mirage_dir or MIRAGE_DIR) / "dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_doc_pool(mirage_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load 37,800 document chunks from doc_pool.json."""
    path = (mirage_dir or MIRAGE_DIR) / "doc_pool.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_oracle(mirage_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load oracle.json: query_id → gold context chunk."""
    path = (mirage_dir or MIRAGE_DIR) / "oracle.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_gold_lookup(doc_pool: list[dict]) -> dict[str, list[int]]:
    """Build query_id → list of doc_pool indices where support=1.

    Returns a dict mapping each mapped_id to the list of indices
    in doc_pool that are gold (support=1) chunks for that query.
    """
    lookup: dict[str, list[int]] = {}
    for idx, chunk in enumerate(doc_pool):
        if chunk["support"] == 1:
            mid = chunk["mapped_id"]
            lookup.setdefault(mid, []).append(idx)
    return lookup


def build_pool_index(doc_pool: list[dict]) -> dict[str, list[int]]:
    """Build query_id → list of ALL doc_pool indices for that query.

    Maps each mapped_id to the indices of every chunk (gold + distractor)
    associated with that query.
    """
    lookup: dict[str, list[int]] = {}
    for idx, chunk in enumerate(doc_pool):
        mid = chunk["mapped_id"]
        lookup.setdefault(mid, []).append(idx)
    return lookup


def make_chunk_id(mapped_id: str, pool_index: int) -> str:
    """Generate a unique chunk ID for ChromaDB indexing."""
    return f"{mapped_id}:{pool_index}"


def select_partial_subset(
    dataset: list[dict],
    doc_pool: list[dict],
    oracle: dict[str, dict],
    n: int,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Select a deterministic subset of n questions and their related chunks.

    Sorts by query_id, takes first n, filters doc_pool and oracle to match.
    """
    sorted_ds = sorted(dataset, key=lambda q: q["query_id"])
    subset_ds = sorted_ds[:n]
    query_ids = {q["query_id"] for q in subset_ds}
    subset_pool = [c for c in doc_pool if c["mapped_id"] in query_ids]
    subset_oracle = {qid: oracle[qid] for qid in query_ids if qid in oracle}
    return subset_ds, subset_pool, subset_oracle
