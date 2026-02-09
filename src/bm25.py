"""BM25 lexical retrieval for MIRAGE doc pool."""

import re
import string

from rank_bm25 import BM25Okapi

from src.logger import get_logger

log = get_logger("pico-rag.bm25")

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = _PUNCT_RE.sub(" ", text.lower())
    return text.split()


def build_bm25_index(
    doc_pool: list[dict],
) -> tuple[BM25Okapi, list[dict]]:
    """Build a BM25 index from MIRAGE doc_pool chunks.

    Returns (bm25_index, doc_pool_ref) where doc_pool_ref is the same
    list passed in (kept for alignment with scored results).
    """
    corpus = [_tokenize(c["doc_chunk"]) for c in doc_pool]
    log.info("Building BM25 index", event="index_start", n_docs=len(doc_pool))
    index = BM25Okapi(corpus)
    log.info("BM25 index built", event="index_done", n_docs=len(doc_pool))
    return index, doc_pool


def bm25_search(
    index: BM25Okapi,
    doc_pool: list[dict],
    query: str,
    n_results: int = 25,
) -> dict:
    """Search BM25 index and return results in ChromaDB-compatible format.

    Returns dict with keys: documents, metadatas, distances (BM25 scores).
    """
    query_tokens = _tokenize(query)
    scores = index.get_scores(query_tokens)

    # Get top-n indices by score (descending)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = ranked[:n_results]

    documents = []
    metadatas = []
    distances = []

    for idx in top_indices:
        chunk = doc_pool[idx]
        documents.append(chunk["doc_chunk"])
        metadatas.append({
            "mapped_id": chunk["mapped_id"],
            "doc_name": chunk["doc_name"],
            "support": chunk["support"],
            "pool_index": idx,
        })
        distances.append(float(scores[idx]))

    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }
