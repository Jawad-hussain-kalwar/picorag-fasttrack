import os
import time

import httpx

from src.config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_RATE_LIMIT,
    OPENROUTER_TEMPERATURE,
    OPENROUTER_TIMEOUT_SECONDS,
)
from src.logger import get_logger


log = get_logger("pico-rag.generate")

# Simple rate limiter state
_last_call_time: float = 0.0


def build_prompt(question: str, contexts: list[dict]) -> list[dict]:
    """Build prompt for interactive pipeline (original format)."""
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


# ---------------------------------------------------------------------------
# MIRAGE prompt builders (match MIRAGE's exact format for comparability)
# ---------------------------------------------------------------------------

def build_base_prompt(question: str) -> list[dict]:
    """Base (closed-book): question only, no context."""
    return [
        {"role": "system", "content": "You are a helpful assistant.\n"},
        {"role": "user", "content": f"Question: {question}\n\nAnswer : \n"},
    ]


def build_oracle_prompt(question: str, oracle_chunk: str) -> list[dict]:
    """Oracle: question + gold context chunk."""
    return [
        {"role": "system", "content": "You are a helpful assistant.\n"},
        {
            "role": "user",
            "content": (
                f"Question : {question}\n\n"
                f"Context : {oracle_chunk}\n\n"
                "Answer : "
            ),
        },
    ]


def build_mixed_prompt(question: str, chunks: list[str]) -> list[dict]:
    """Mixed (RAG): question + numbered retrieved chunks."""
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(chunks))
    return [
        {"role": "system", "content": "You are a helpful assistant.\n"},
        {
            "role": "user",
            "content": (
                f"Question : {question}\n\n"
                f"Context : {numbered}\n\n"
                "Answer : "
            ),
        },
    ]


# ---------------------------------------------------------------------------
# OpenRouter API caller with rate limiting and retry
# ---------------------------------------------------------------------------

def call_openrouter(
    messages: list[dict],
    model: str | None = None,
    max_retries: int = 3,
) -> str:
    """Call OpenRouter chat completion with rate limiting and retry.

    Returns the generated text. Raises RuntimeError on persistent failure.
    """
    global _last_call_time

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    model = model or OPENROUTER_MODEL
    min_interval = 60.0 / OPENROUTER_RATE_LIMIT

    # Rate limit: wait if needed
    elapsed = time.monotonic() - _last_call_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": OPENROUTER_TEMPERATURE,
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        _last_call_time = time.monotonic()
        try:
            with httpx.Client(timeout=OPENROUTER_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )

            if response.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                log.warning(
                    f"Rate limited, waiting {wait}s",
                    event="warning",
                    attempt=attempt,
                )
                time.sleep(wait)
                continue

            if response.status_code != 200:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    log.warning(
                        f"API error {response.status_code}, retrying in {wait}s",
                        event="warning",
                        attempt=attempt,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"OpenRouter failed ({response.status_code}): {response.text}"
                )

            body = response.json()
            text = body["choices"][0]["message"]["content"].strip()
            return text

        except httpx.TimeoutException:
            if attempt < max_retries:
                log.warning(
                    f"Timeout, retrying ({attempt}/{max_retries})",
                    event="warning",
                )
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError("OpenRouter request timed out after all retries")

    raise RuntimeError("OpenRouter request failed after all retries")


def generate_answer(question: str, contexts: list[dict]) -> str:
    """Generate answer for interactive pipeline (original function, unchanged API)."""
    messages = build_prompt(question, contexts)
    log.info(
        "Sending LLM request",
        event="llm_request",
        model=OPENROUTER_MODEL,
        contexts=len(contexts),
        question_chars=len(question),
    )
    text = call_openrouter(messages)
    log.info("Received LLM response", event="llm_response", answer_chars=len(text))
    return text

