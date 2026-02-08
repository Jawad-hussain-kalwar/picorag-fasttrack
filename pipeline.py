from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, DATA_DIR, TOP_K
from generate import generate_answer
from logger import get_logger
from retrieve import ensure_indexed, get_client, get_collection, search


log = get_logger("pico-rag.pipeline")


def answer_question(question: str) -> dict:
    log.info("Question received", event="user_input", chars=len(question))
    client = get_client(str(CHROMA_PERSIST_DIR))
    collection = get_collection(client, COLLECTION_NAME)
    with log.timer("Retrieval index ready", event="index_done"):
        ensure_indexed(collection, str(DATA_DIR))

    with log.timer("Vector retrieval completed", event="retrieve_done"):
        results = search(collection, query=question, n_results=TOP_K)

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    contexts = []
    for text, metadata, distance in zip(docs, metadatas, distances):
        contexts.append(
            {
                "text": text,
                "metadata": metadata,
                "distance": float(distance),
            }
        )

    if not contexts:
        log.warning("No relevant context found", event="retrieve_empty")
        return {"question": question, "answer": "No relevant context found.", "contexts": []}

    with log.timer("Answer generated", event="llm_response"):
        answer = generate_answer(question, contexts)
    log.info("Pipeline completed successfully", event="success", contexts=len(contexts))
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }
