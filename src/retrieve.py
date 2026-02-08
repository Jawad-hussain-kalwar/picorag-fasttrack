import chromadb
from chromadb.config import Settings

from src.chunking import naive_chunk
from src.ingest import load_documents_from_dir
from src.logger import get_logger


log = get_logger("pico-rag.retrieve")


def get_client(persist_dir: str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: chromadb.ClientAPI, name: str):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(collection, chunks: list[dict]) -> None:
    if not chunks:
        return
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def ensure_indexed(collection, data_dir: str) -> int:
    log.info("Starting document indexing", event="index_start", data_dir=data_dir)
    total_chunks = 0
    total_docs = 0
    for source, text in load_documents_from_dir(data_dir):
        chunks = naive_chunk(text, source=source)
        upsert_chunks(collection, chunks)
        total_docs += 1
        total_chunks += len(chunks)
        log.debug(
            "Indexed document",
            event="index_doc",
            source=source,
            chunks=len(chunks),
        )
    log.info(
        "Indexing complete",
        event="index_done",
        documents=total_docs,
        chunks=total_chunks,
    )
    return total_chunks


def search(collection, query: str, n_results: int):
    log.info("Running vector search", event="retrieve_start", top_k=n_results)
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def index_mirage_pool(
    collection,
    doc_pool: list[dict],
    batch_size: int = 500,
) -> int:
    """Index MIRAGE doc_pool chunks into ChromaDB.

    Each chunk gets a unique ID of 'mapped_id:pool_index'.
    Skips indexing if collection already has the expected count.
    Returns total chunks indexed.
    """
    existing = collection.count()
    if existing >= len(doc_pool):
        log.info(
            "MIRAGE pool already indexed",
            event="index_done",
            chunks=existing,
        )
        return existing

    log.info(
        "Indexing MIRAGE doc_pool",
        event="index_start",
        total_chunks=len(doc_pool),
    )
    for start in range(0, len(doc_pool), batch_size):
        batch = doc_pool[start : start + batch_size]
        ids = [f"{c['mapped_id']}:{start + i}" for i, c in enumerate(batch)]
        documents = [c["doc_chunk"] for c in batch]
        metadatas = [
            {
                "mapped_id": c["mapped_id"],
                "doc_name": c["doc_name"],
                "support": c["support"],
                "pool_index": start + i,
            }
            for i, c in enumerate(batch)
        ]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        log.debug(
            "Indexed batch",
            event="index_doc",
            batch_start=start,
            batch_size=len(batch),
        )

    total = collection.count()
    log.info("MIRAGE indexing complete", event="index_done", chunks=total)
    return total

