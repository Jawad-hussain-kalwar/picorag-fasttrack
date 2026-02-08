import chromadb
from chromadb.config import Settings


CHROMA_PERSIST_DIR = "./chroma_data"
COLLECTION_NAME = "documents"


def load_document(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def naive_chunk(text: str) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for i, para in enumerate(paragraphs):
        chunks.append({
            "id": f"chunk_{i}",
            "text": para,
            "metadata": {"chunk_index": i},
        })
    return chunks


def get_client(persist_dir: str = CHROMA_PERSIST_DIR) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: chromadb.ClientAPI, name: str = COLLECTION_NAME):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def index_document(collection, chunks: list[dict], source: str):
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{**c["metadata"], "source": source} for c in chunks],
    )


def search(collection, query: str, n_results: int = 3):
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
