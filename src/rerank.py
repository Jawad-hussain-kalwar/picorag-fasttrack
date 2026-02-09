"""Voyage AI reranking via API."""

import time

import httpx

from src.config import VOYAGE_API_KEY, VOYAGE_RERANK_MODEL
from src.logger import get_logger

log = get_logger("pico-rag.rerank")

VOYAGE_BASE_URL = "https://api.voyageai.com/v1"
VOYAGE_RATE_LIMIT = 10_000  # RPM
VOYAGE_TIMEOUT = 30.0

_last_call_time: float = 0.0


def call_voyage_rerank(
    query: str,
    documents: list[str],
    top_k: int,
    model: str = VOYAGE_RERANK_MODEL,
    metadatas: list[dict] | None = None,
    max_retries: int = 3,
) -> dict:
    """Rerank documents using Voyage AI and return in ChromaDB-compatible format.

    Args:
        query: The search query.
        documents: List of document texts to rerank.
        top_k: Number of top documents to return.
        model: Voyage rerank model name.
        metadatas: Optional parallel list of metadata dicts for each document.
        max_retries: Number of retry attempts.

    Returns dict with keys: documents, metadatas, distances (relevance scores).
    """
    global _last_call_time

    api_key = VOYAGE_API_KEY
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is not set.")

    min_interval = 60.0 / VOYAGE_RATE_LIMIT
    elapsed = time.monotonic() - _last_call_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_k": top_k,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        _last_call_time = time.monotonic()
        try:
            with httpx.Client(timeout=VOYAGE_TIMEOUT) as client:
                response = client.post(
                    f"{VOYAGE_BASE_URL}/rerank",
                    headers=headers,
                    json=payload,
                )

            if response.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                log.warning(f"Voyage rate limited ({model}), waiting {wait}s", event="warning")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    log.warning(
                        f"Voyage API error {response.status_code}, retrying in {wait}s",
                        event="warning",
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Voyage rerank failed ({response.status_code}): {response.text}"
                )

            body = response.json()
            results = body.get("data", [])

            # Results come as [{index, relevance_score}, ...] sorted by relevance
            reranked_docs = []
            reranked_metas = []
            reranked_scores = []

            for item in results[:top_k]:
                idx = item["index"]
                reranked_docs.append(documents[idx])
                if metadatas:
                    reranked_metas.append(metadatas[idx])
                else:
                    reranked_metas.append({})
                reranked_scores.append(float(item["relevance_score"]))

            return {
                "documents": [reranked_docs],
                "metadatas": [reranked_metas],
                "distances": [reranked_scores],
            }

        except httpx.TimeoutException:
            if attempt < max_retries:
                log.warning(f"Voyage timeout, retrying ({attempt}/{max_retries})", event="warning")
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError("Voyage rerank timed out after all retries")

    raise RuntimeError("Voyage rerank failed after all retries")
