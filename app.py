import sys

from src.logger import get_logger
from src.pipeline import answer_question


log = get_logger("pico-rag.app")


def main() -> int:
    log.info("PicoRAG CLI starting", event="startup")
    if len(sys.argv) > 1:
        return _run_single_query(" ".join(sys.argv[1:]).strip())
    return _run_repl()


def _run_repl() -> int:
    print("PicoRAG REPL. Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            log.info("REPL interrupted", event="shutdown")
            return 0

        if question.lower() in {"exit", "quit"}:
            log.info("REPL terminated by user", event="shutdown")
            return 0

        if not question:
            log.warning("No question provided", event="warning")
            continue

        _run_single_query(question)


def _run_single_query(question: str) -> int:
    if not question:
        log.warning("No question provided", event="warning")
        return 1

    try:
        result = answer_question(question)
    except Exception as exc:
        log.exception("Request failed", event="error", reason=str(exc))
        print(f"Error: {exc}")
        return 1

    print("\nAnswer:")
    print(result["answer"])
    print("\nRetrieved context:")
    for idx, ctx in enumerate(result["contexts"], start=1):
        source = ctx["metadata"].get("source", "unknown")
        chunk_index = ctx["metadata"].get("chunk_index", -1)
        snippet = ctx["text"][:180].replace("\n", " ")
        print(f"[{idx}] source={source} chunk={chunk_index} distance={ctx['distance']:.4f}")
        print(f"    {snippet}...")

    log.info("PicoRAG CLI completed", event="shutdown", contexts=len(result["contexts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

