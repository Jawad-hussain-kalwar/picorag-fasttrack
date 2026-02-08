import shutil
from rag_pipeline import (
    load_document,
    naive_chunk,
    get_client,
    get_collection,
    index_document,
    search,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
)

SAMPLE_DOC = "./data/sample_document.txt"
TEST_QUERIES = [
    "How does light affect the process?",
    "What role does carbon dioxide play?",
    "Which enzymes are involved?",
    "What happens in extreme temperatures?",
    "How does this relate to the global environment?",
]


def cleanup():
    shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)


def test_index_and_search():
    cleanup()

    # --- Load & chunk ---
    text = load_document(SAMPLE_DOC)
    chunks = naive_chunk(text)
    print(f"Loaded document: {len(text)} chars, {len(chunks)} chunks\n")

    # --- Index ---
    client = get_client()
    collection = get_collection(client)
    index_document(collection, chunks, source=SAMPLE_DOC)
    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}'\n")

    # --- Search ---
    print("=" * 60)
    print("VECTOR SEARCH RESULTS")
    print("=" * 60)
    for q in TEST_QUERIES:
        results = search(collection, q, n_results=2)
        print(f"\nQuery: {q}")
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            print(f"  [{i+1}] (distance={dist:.4f}, chunk={meta['chunk_index']})")
            print(f"      {doc[:120]}...")

    # --- Persistence test ---
    print("\n" + "=" * 60)
    print("PERSISTENCE TEST")
    print("=" * 60)
    del collection
    del client

    client2 = get_client()
    collection2 = get_collection(client2)
    count = collection2.count()
    print(f"Reopened collection: {count} chunks found")

    results2 = search(collection2, TEST_QUERIES[0], n_results=1)
    print(f"Query after reopen: '{TEST_QUERIES[0]}'")
    print(f"  Top result: {results2['documents'][0][0][:120]}...")
    print(f"  Distance: {results2['distances'][0][0]:.4f}")

    assert count == len(chunks), f"Expected {len(chunks)} chunks, got {count}"
    assert len(results2["documents"][0]) == 1
    print("\nAll assertions passed.")

    cleanup()
    print("Cleanup done.")


if __name__ == "__main__":
    test_index_and_search()
