"""Reciprocal Rank Fusion (RRF) for hybrid retrieval."""

from src.config import E2_RRF_K


def rrf_fuse(
    vector_results: dict,
    bm25_results: dict,
    k: int,
    rrf_k: int = E2_RRF_K,
) -> dict:
    """Fuse two result sets using Reciprocal Rank Fusion.

    Both inputs must be in ChromaDB-compatible format:
      {"documents": [[...]], "metadatas": [[...]], "distances": [[...]]}

    Returns fused results in the same format, sorted by RRF score descending,
    trimmed to top-k.
    """
    # Build doc_id -> (rank, metadata, document) from each list.
    # Use pool_index from metadatas as unique doc identifier.
    def _extract(results: dict) -> list[tuple[int, dict, str, float]]:
        """Return list of (pool_index, metadata, document, score)."""
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        return [
            (m["pool_index"], m, d, s)
            for m, d, s in zip(metas, docs, dists)
        ]

    vec_items = _extract(vector_results)
    bm25_items = _extract(bm25_results)

    # Build rank maps (1-indexed)
    vec_rank: dict[int, int] = {
        item[0]: rank for rank, item in enumerate(vec_items, 1)
    }
    bm25_rank: dict[int, int] = {
        item[0]: rank for rank, item in enumerate(bm25_items, 1)
    }

    # Collect all unique pool_indices with their metadata/document
    all_docs: dict[int, tuple[dict, str]] = {}
    for pool_idx, meta, doc, _ in vec_items:
        all_docs[pool_idx] = (meta, doc)
    for pool_idx, meta, doc, _ in bm25_items:
        if pool_idx not in all_docs:
            all_docs[pool_idx] = (meta, doc)

    # Compute RRF scores
    rrf_scores: list[tuple[int, float]] = []
    for pool_idx in all_docs:
        score = 0.0
        if pool_idx in vec_rank:
            score += 1.0 / (rrf_k + vec_rank[pool_idx])
        if pool_idx in bm25_rank:
            score += 1.0 / (rrf_k + bm25_rank[pool_idx])
        rrf_scores.append((pool_idx, score))

    # Sort by RRF score descending, take top-k
    rrf_scores.sort(key=lambda x: x[1], reverse=True)
    top = rrf_scores[:k]

    documents = []
    metadatas = []
    distances = []
    for pool_idx, score in top:
        meta, doc = all_docs[pool_idx]
        documents.append(doc)
        metadatas.append(meta)
        distances.append(score)

    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }
