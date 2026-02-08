from pathlib import Path


def load_document(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_documents_from_dir(data_dir: str | Path) -> list[tuple[str, str]]:
    root = Path(data_dir)
    docs: list[tuple[str, str]] = []
    for file_path in sorted(root.glob("*.txt")):
        docs.append((str(file_path), load_document(file_path)))
    return docs
