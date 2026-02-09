"""Custom embeddings via OpenRouter API (e.g. Qwen3-Embedding-4B)."""

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
    OPENROUTER_TIMEOUT_SECONDS,
)
from src.logger import get_logger

log = get_logger("pico-rag.embed")

_last_call_time: float = 0.0
_rate_lock = threading.Lock()


def embed_texts_openrouter(
    texts: list[str],
    model: str = OPENROUTER_EMBED_MODEL,
    batch_size: int = 50,
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

        for attempt in range(1, 4):
            try:
                with httpx.Client(timeout=OPENROUTER_TIMEOUT_SECONDS) as client:
                    response = client.post(
                        f"{OPENROUTER_BASE_URL}/embeddings",
                        headers=headers,
                        json=payload,
                    )

                if response.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    log.warning(f"Embed rate limited ({model}), waiting {wait}s", event="warning")
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    if attempt < 3:
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
                break

            except httpx.TimeoutException:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("Embed request timed out after all retries")

        if (start + batch_size) % 500 < batch_size:
            log.info(
                f"Embedded {min(start + batch_size, len(texts))}/{len(texts)}",
                event="info",
            )

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

    log.info(
        f"Indexing {len(doc_pool)} chunks with custom embeddings",
        event="index_start",
    )

    for start in range(0, len(doc_pool), batch_size):
        batch = doc_pool[start : start + batch_size]
        texts = [c["doc_chunk"] for c in batch]
        ids = [f"{c['mapped_id']}:{start + i}" for i, c in enumerate(batch)]
        metadatas = [
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
            metadatas=metadatas,
        )

        if (start + batch_size) % 500 < batch_size:
            log.info(
                f"Indexed {min(start + batch_size, len(doc_pool))}/{len(doc_pool)}",
                event="index_doc",
            )

    total = collection.count()
    log.info("Custom embedding indexing complete", event="index_done", chunks=total)
    return total


def search_with_custom_embeddings(
    collection_name: str,
    query: str,
    n_results: int,
    embed_fn=None,
    persist_dir: str | None = None,
) -> dict:
    """Search a custom-embedded ChromaDB collection.

    Embeds the query with embed_fn, then queries ChromaDB with query_embeddings.
    Returns results in standard ChromaDB format.
    """
    if embed_fn is None:
        embed_fn = embed_texts_openrouter

    collection = get_custom_collection(collection_name, persist_dir)
    query_embedding = embed_fn([query])[0]

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
