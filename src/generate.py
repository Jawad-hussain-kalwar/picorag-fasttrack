import os
import threading
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

# Per-model rate limiter state (so local/online calls don't block each other)
_rate_locks: dict[str, threading.Lock] = {}
_last_call_times: dict[str, float] = {}
_init_lock = threading.Lock()


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

_SYSTEM_BASE = (
    "You are a helpful assistant. Give ONLY the answer, nothing else. "
    "No explanations, no full sentences, no parenthetical qualifiers.\n\n"
    "Examples:\n"
    "Q: Who painted the Mona Lisa?\nA: Leonardo da Vinci\n"
    "Q: What year did the Titanic sink?\nA: 1912\n"
)

_SYSTEM_CITED = (
    "You are a helpful assistant. Give ONLY the answer with source citations "
    "in [1], [2] format. No explanations, no full sentences.\n\n"
    "CORRECT citations: [1], [2], [3]\n"
    "WRONG citations: 【1†L1-L3】, 【2】, (1), {1} — NEVER use these formats.\n\n"
    "Examples:\n"
    "Q: Who painted the Mona Lisa?\nA: Leonardo da Vinci [1]\n"
    "Q: What year did the Titanic sink?\nA: 1912 [2], [3]\n"
)


def build_base_prompt(question: str) -> list[dict]:
    """Base (closed-book): question only, no context."""
    return [
        {"role": "system", "content": _SYSTEM_BASE},
        {"role": "user", "content": f"Question: {question}\n\nAnswer concisely in a few words: \n"},
    ]


def build_oracle_prompt(question: str, oracle_chunk: str) -> list[dict]:
    """Oracle: question + gold context chunk."""
    return [
        {"role": "system", "content": _SYSTEM_BASE},
        {
            "role": "user",
            "content": (
                f"Question : {question}\n\n"
                f"Context : {oracle_chunk}\n\n"
                "Answer concisely in a few words: "
            ),
        },
    ]


def build_mixed_prompt(question: str, chunks: list[str]) -> list[dict]:
    """Mixed (RAG): question + numbered retrieved chunks."""
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(chunks))
    return [
        {"role": "system", "content": _SYSTEM_BASE},
        {
            "role": "user",
            "content": (
                f"Question : {question}\n\n"
                f"Context : {numbered}\n\n"
                "Answer concisely in a few words: "
            ),
        },
    ]


def build_mixed_prompt_cited(question: str, chunks: list[str]) -> list[dict]:
    """Mixed (RAG) with citation instructions: question + numbered chunks."""
    numbered = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return [
        {"role": "system", "content": _SYSTEM_CITED},
        {
            "role": "user",
            "content": (
                f"Question : {question}\n\n"
                f"Context :\n{numbered}\n\n"
                "Answer concisely in a few words. "
                "Cite your sources using [1], [2], etc.: "
            ),
        },
    ]


# ---------------------------------------------------------------------------
# E5 prompt builders (hyperoptimised for 4B model)
# ---------------------------------------------------------------------------

_SYSTEM_E5 = (
    "You are a helpful assistant. Answer questions using ONLY the provided context.\n"
    "Give ONLY the answer with citations like [1], [2]. No explanations.\n"
    "If the answer is NOT in the context, say: Not enough evidence in knowledge base\n\n"
    "Examples:\n"
    "Q: Who painted the Mona Lisa?\nA: Leonardo da Vinci [1]\n"
    "Q: What year did the Titanic sink?\nA: 1912 [2], [3]\n"
)


def build_e5_mixed_prompt(question: str, chunks: list[str]) -> list[dict]:
    """E5 hyperoptimised mixed prompt for 4B model with abstention."""
    numbered = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return [
        {"role": "system", "content": _SYSTEM_E5},
        {
            "role": "user",
            "content": (
                f"Question : {question}\n\n"
                f"Context :\n{numbered}\n\n"
                "Answer concisely: "
            ),
        },
    ]


def build_reformulation_prompt(
    original_query: str,
    retrieved_snippets: list[str],
) -> list[dict]:
    """Build prompt asking model to rephrase query with different keywords."""
    snippets_block = "\n".join(
        f"- {s[:200]}" for s in retrieved_snippets[:3]
    )
    return [
        {"role": "system", "content": (
            "You are a search query rewriter. "
            "Given a question and some retrieved snippets that were NOT helpful, "
            "rewrite the question using different keywords, synonyms, or "
            "alternative phrasing to find better results. "
            "Output ONLY the rewritten question, nothing else."
        )},
        {
            "role": "user",
            "content": (
                f"Original question: {original_query}\n\n"
                f"Unhelpful snippets:\n{snippets_block}\n\n"
                "Rewritten question:"
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
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    model = model or OPENROUTER_MODEL
    min_interval = 60.0 / OPENROUTER_RATE_LIMIT

    # Per-model rate limiter setup
    if model not in _rate_locks:
        with _init_lock:
            if model not in _rate_locks:
                _rate_locks[model] = threading.Lock()
                _last_call_times[model] = 0.0

    # Merge system messages into first user message only for models that
    # don't support the system role (Gemma via Google AI Studio).
    if "gemma" in model.lower():
        merged: list[dict] = []
        system_parts: list[str] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                if system_parts and msg["role"] == "user":
                    prefix = "\n".join(system_parts).strip()
                    merged.append({"role": "user", "content": f"{prefix}\n\n{msg['content']}"})
                    system_parts = []
                else:
                    merged.append(msg)
        if not merged:
            merged = [{"role": "user", "content": "\n".join(system_parts)}]
    else:
        merged = messages

    payload = {
        "model": model,
        "messages": merged,
        "temperature": OPENROUTER_TEMPERATURE,
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        with _rate_locks[model]:
            elapsed = time.monotonic() - _last_call_times[model]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            _last_call_times[model] = time.monotonic()
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
                    f"Rate limited ({model}), waiting {wait}s",
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
            if "choices" not in body or not body["choices"]:
                err_msg = body.get("error", {}).get("message", str(body))
                raise RuntimeError(f"OpenRouter returned no choices: {err_msg}")
            text = (body["choices"][0]["message"]["content"] or "").strip()
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

