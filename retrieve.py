import chromadb
from chromadb.config import Settings

from chunking import naive_chunk
from ingest import load_documents_from_dir
from logger import get_logger


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
