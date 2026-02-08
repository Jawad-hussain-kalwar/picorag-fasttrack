import hashlib


def naive_chunk(text: str, source: str) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict] = []
    for idx, para in enumerate(paragraphs):
        raw = f"{source}:{idx}:{para}".encode("utf-8")
        chunk_id = hashlib.sha1(raw).hexdigest()
        chunks.append(
            {
                "id": chunk_id,
                "text": para,
                "metadata": {"source": source, "chunk_index": idx},
            }
        )
    return chunks
