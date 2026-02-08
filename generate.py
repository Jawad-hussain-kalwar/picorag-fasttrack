import os

import httpx

from config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_TEMPERATURE,
    OPENROUTER_TIMEOUT_SECONDS,
)
from logger import get_logger


log = get_logger("pico-rag.generate")


def build_prompt(question: str, contexts: list[dict]) -> list[dict]:
    context_lines: list[str] = []
    for idx, ctx in enumerate(contexts, start=1):
        source = ctx["metadata"].get("source", "unknown")
        chunk_idx = ctx["metadata"].get("chunk_index", -1)
        context_lines.append(
            f"[{idx}] Source: {source} | Chunk: {chunk_idx}\n{ctx['text']}"
        )

    context_block = "\n\n".join(context_lines)

    system = (
        "You are a precise RAG assistant. Answer only using the retrieved context. "
        "If context is insufficient, say you do not have enough information."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Provide a concise answer and cite source numbers like [1], [2]."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_answer(question: str, contexts: list[dict]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.error("Missing API key", event="error", env_var="OPENROUTER_API_KEY")
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": build_prompt(question, contexts),
        "temperature": OPENROUTER_TEMPERATURE,
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    log.info(
        "Sending LLM request",
        event="llm_request",
        model=OPENROUTER_MODEL,
        contexts=len(contexts),
        question_chars=len(question),
    )
    with httpx.Client(timeout=OPENROUTER_TIMEOUT_SECONDS) as client:
        response = client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )

    if response.status_code != 200:
        log.error(
            "OpenRouter request failed",
            event="error",
            status_code=response.status_code,
        )
        raise RuntimeError(
            f"OpenRouter request failed ({response.status_code}): {response.text}"
        )

    body = response.json()
    try:
        text = body["choices"][0]["message"]["content"].strip()
        log.info("Received LLM response", event="llm_response", answer_chars=len(text))
        return text
    except (KeyError, IndexError, AttributeError) as exc:
        log.error("Unexpected OpenRouter response format", event="error")
        raise RuntimeError(f"Unexpected OpenRouter response format: {body}") from exc
