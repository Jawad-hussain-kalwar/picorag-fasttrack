"""Custom embeddings via OpenRouter API (e.g. Qwen3-Embedding-4B)."""

from typing import Any, cast

import os
import threading
import time

import chromadb
import httpx
from chromadb.config import Settings

from src.config import (
    CHROMA_PERSIST_DIR,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBED_MODEL,
    OPENROUTER_RATE_LIMIT,
)
from src.logger import get_logger

log = get_logger("pico-rag.embed")

_last_call_time: float = 0.0
_rate_lock = threading.Lock()


def embed_texts_openrouter(
    texts: list[str],
    model: str = OPENROUTER_EMBED_MODEL,
    batch_size: int = 50,
    timeout: float = 300.0,
) -> list[list[float]]:
    """Embed texts via OpenRouter embeddings API.

    Batches requests to respect token/size limits.
    Returns list of embedding vectors (one per input text).
    """
    global _last_call_time

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_embeddings: list[list[float]] = []
    min_interval = 60.0 / OPENROUTER_RATE_LIMIT

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        with _rate_lock:
            elapsed = time.monotonic() - _last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            _last_call_time = time.monotonic()

        payload = {
            "model": model,
            "input": batch,
        }

        max_retries = 6
        batch_done = False
        for attempt in range(1, max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        f"{OPENROUTER_BASE_URL}/embeddings",
                        headers=headers,
                        json=payload,
                    )

                if response.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    print(f"  Embed rate limited ({model}), waiting {wait}s (attempt {attempt}/{max_retries})", flush=True)
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(
                        f"Embed API failed ({response.status_code}): {response.text}"
                    )

                body = response.json()
                data = body.get("data", [])
                # Sort by index to maintain order
                data.sort(key=lambda x: x["index"])
                batch_embs = [item["embedding"] for item in data]
                all_embeddings.extend(batch_embs)
                batch_done = True
                break

            except httpx.TimeoutException:
                if attempt < max_retries:
                    wait = min(2 ** attempt * 3, 60)
                    print(f"  Embed timeout ({model}), retrying ({attempt}/{max_retries}) wait {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError("Embed request timed out after all retries")

        if not batch_done:
            raise RuntimeError(
                f"Embed batch failed after all retries ({model}), "
                f"batch {start//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
            )

        done_count = min(start + batch_size, len(texts))
        if done_count % 100 < batch_size or done_count == len(texts):
            print(f"    Embedded {done_count}/{len(texts)}", flush=True)

    return all_embeddings


def get_custom_collection(
    collection_name: str,
    persist_dir: str | None = None,
) -> chromadb.Collection:
    """Get or create a ChromaDB collection for custom embeddings.

    Uses cosine similarity, same as default MiniLM collection.
    """
    client = chromadb.PersistentClient(
        path=persist_dir or str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_mirage_with_custom_embeddings(
    doc_pool: list[dict],
    collection_name: str,
    embed_fn=None,
    batch_size: int = 100,
    persist_dir: str | None = None,
) -> int:
    """Index MIRAGE doc_pool using custom embeddings into a separate collection.

    If embed_fn is None, uses embed_texts_openrouter.
    Skips if collection already has expected count.
    Returns total chunks indexed.
    """
    if embed_fn is None:
        embed_fn = embed_texts_openrouter

    collection = get_custom_collection(collection_name, persist_dir)

    existing = collection.count()
    if existing >= len(doc_pool):
        log.info(
            "Custom-embedded collection already indexed",
            event="index_done",
            chunks=existing,
        )
        return existing

    # Build set of already-indexed IDs for resume support
    existing_ids: set[str] = set()
    if existing > 0:
        all_stored = collection.get(include=[])
        existing_ids = set(all_stored["ids"])
        log.info(
            f"Resuming: {len(existing_ids)} chunks already indexed, "
            f"{len(doc_pool) - len(existing_ids)} remaining",
            event="index_resume",
        )

    log.info(
        f"Indexing {len(doc_pool)} chunks with custom embeddings",
        event="index_start",
    )

    for start in range(0, len(doc_pool), batch_size):
        batch = doc_pool[start : start + batch_size]
        ids = [f"{c['mapped_id']}:{start + i}" for i, c in enumerate(batch)]

        # Skip batch if all IDs already exist
        if existing_ids and all(id_ in existing_ids for id_ in ids):
            continue

        texts = [c["doc_chunk"] for c in batch]
        metadatas: list[dict[str, Any]] = [
            {
                "mapped_id": c["mapped_id"],
                "doc_name": c["doc_name"],
                "support": c["support"],
                "pool_index": start + i,
            }
            for i, c in enumerate(batch)
        ]

        embeddings = embed_fn(texts)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

        done_count = min(start + batch_size, len(doc_pool))
        if done_count % 250 < batch_size or done_count == len(doc_pool):
            print(f"    Indexed {done_count}/{len(doc_pool)} chunks", flush=True)

    total = collection.count()
    log.info("Custom embedding indexing complete", event="index_done", chunks=total)
    return total


def search_with_custom_embeddings(
    collection_name: str,
    query: str,
    n_results: int,
    embed_fn=None,
    persist_dir: str | None = None,
) -> dict[str, Any]:
    """Search a custom-embedded ChromaDB collection.

    Embeds the query with embed_fn, then queries ChromaDB with query_embeddings.
    Returns results in standard ChromaDB format.
    """
    if embed_fn is None:
        embed_fn = embed_texts_openrouter

    collection = get_custom_collection(collection_name, persist_dir)
    query_embedding = embed_fn([query])[0]

    return cast(dict[str, Any], collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    ))


def batch_search_with_custom_embeddings(
    collection_name: str,
    queries: list[str],
    n_results: int,
    embed_fn=None,
    persist_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Batch search: embed all queries in one call, then query ChromaDB per query.

    Much faster than individual search_with_custom_embeddings calls because
    it avoids N separate embed API round-trips.
    """
    if embed_fn is None:
        embed_fn = embed_texts_openrouter

    collection = get_custom_collection(collection_name, persist_dir)
    all_embeddings = embed_fn(queries)

    results: list[dict[str, Any]] = []
    for emb in all_embeddings:
        result = collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        results.append(cast(dict[str, Any], result))
    return results
