"""Reusable LLM-as-Judge for Faithfulness, Groundedness, Answer Relevance & Semantic Correctness.

Provides `call_judge()` for single-query evaluation and `aggregate_judge_scores()` for
computing mean metrics over a list of per-query result rows.
"""

import json
import os
import threading
import time

import httpx

from src.config import (
    JUDGE_MODEL,
    JUDGE_RATE_LIMIT,
    OPENROUTER_BASE_URL,
    OPENROUTER_TIMEOUT_SECONDS,
)
from src.logger import get_logger

log = get_logger("pico-rag.judge")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_MAX_TOKENS = 1024
JUDGE_TEMPERATURE = 0.0

SUBMIT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "submit_judgment",
        "description": "Submit evaluation scores for a RAG answer",
        "parameters": {
            "type": "object",
            "properties": {
                "faithfulness": {
                    "type": "number",
                    "description": "Ratio of supported claims to total claims (0.0-1.0)",
                },
                "groundedness": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "0=not grounded, 1=partially grounded, 2=fully grounded",
                },
                "answer_relevance": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "0=irrelevant, 1=partially relevant, 2=fully relevant to the question",
                },
                "semantic_correctness": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "0=incorrect, 1=partially correct, 2=correct compared to gold answers",
                },
            },
            "required": ["faithfulness", "groundedness", "answer_relevance", "semantic_correctness"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are an expert AI Judge evaluating a RAG system's performance. "
    "Your responsibility is to analyze the provided Question, Retrieved Contexts, Gold Answers, and Predicted Answer, "
    "and then score the Answer quality.\n\n"

    "CRITICAL REQUIREMENT: You MUST submit your evaluation ONLY by calling the `submit_judgment` tool. "
    "Do NOT write any natural language text, reasoning, or explanations. "
    "Any text output is a failure. You MUST use the tool.\n\n"

    "You have one tool `submit_judgment` which requires these 4 parameters:\n"
    "  - faithfulness (float 0.0-1.0): Ratio of claims supported by Contexts.\n"
    "  - groundedness (int 0-2): 0=Not grounded, 1=Partial, 2=Fully grounded in Contexts.\n"
    "  - answer_relevance (int 0-2): 0=Irrelevant, 1=Partial, 2=Fully addresses Question.\n"
    "  - semantic_correctness (int 0-2): 0=Incorrect, 1=Partial, 2=Correct vs Gold Answers.\n\n"

    "Analyze the input data and determine these scores. "
    "Then, IMMEDIATELY call `submit_judgment` with your determined values. "
    "Do not add any text before or after the tool call. "
    "CALL THE TOOL NOW."
)

CORRECTION_MSG = (
    "ERROR: You wrote text instead of calling the submit_judgment tool. "
    "This is wrong. You MUST call the submit_judgment function with the "
    "4 required parameters (faithfulness, groundedness, answer_relevance, "
    "semantic_correctness). Do NOT write text. Call the tool NOW."
)

USER_TEMPLATE = """\
Evaluate the following Answer to a Question.

=== QUESTION ===
{question}

=== GOLD ANSWERS ===
{gold_answers_block}

=== RETRIEVED CONTEXTS ===
{contexts_block}

=== ANSWER TO EVALUATE ===
{prediction}

=== EVALUATION INSTRUCTIONS ===

STEP 1 — FAITHFULNESS (float 0.0-1.0):
- Identify every factual claim in the Answer.
- Count how many are supported by the Retrieved Contexts.
- faithfulness = supported_claims / total_claims.
- If no factual claims or Answer is "I don't know"/error/empty: faithfulness = 1.0.

STEP 2 — GROUNDEDNESS (int 0-2):
- 0 = NOT grounded — Answer has substantial info NOT in Retrieved Contexts or contradicts them.
- 1 = PARTIALLY grounded — some info from contexts, some external.
- 2 = FULLY grounded — everything traceable to Retrieved Contexts.

STEP 3 — ANSWER RELEVANCE (int 0-2):
- 0 = IRRELEVANT — Answer does not address the Question at all.
- 1 = PARTIALLY relevant — Answer partially addresses the Question.
- 2 = FULLY relevant — Answer directly and completely addresses the Question.

STEP 4 — SEMANTIC CORRECTNESS (int 0-2):
- Compare the Answer to the Gold Answers.
- 0 = INCORRECT — Answer contradicts or is unrelated to Gold Answers.
- 1 = PARTIALLY correct — Answer overlaps but misses key info or adds errors.
- 2 = CORRECT — Answer conveys the same meaning as one of the Gold Answers.

Call submit_judgment now with all four scores."""


# ---------------------------------------------------------------------------
# Rate-limit state
# ---------------------------------------------------------------------------

_last_call_time: float = 0.0
_rate_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def null_result(raw: str) -> dict:
    """Return a result dict with None scores."""
    return {
        "faithfulness": None,
        "groundedness": None,
        "answer_relevance": None,
        "semantic_correctness": None,
        "raw": raw,
    }


def parse_judge_response(body: dict) -> dict:
    """Extract 4 judge scores from an OpenRouter tool-call response body."""
    raw = json.dumps(body, ensure_ascii=False)
    try:
        choices = body.get("choices", [])
        if not choices:
            err = body.get("error", {}).get("message", str(body))
            log.warning(f"No choices in judge response: {err}", event="warning")
            return null_result(raw)

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")

        if tool_calls:
            args_str = tool_calls[0]["function"]["arguments"]
            args = json.loads(args_str)

            faith = args.get("faithfulness")
            if faith is not None:
                faith = max(0.0, min(1.0, float(faith)))

            ground = args.get("groundedness")
            if ground is not None:
                ground = max(0, min(2, int(ground)))

            relevance = args.get("answer_relevance")
            if relevance is not None:
                relevance = max(0, min(2, int(relevance)))

            correctness = args.get("semantic_correctness")
            if correctness is not None:
                correctness = max(0, min(2, int(correctness)))

            return {
                "faithfulness": faith,
                "groundedness": ground,
                "answer_relevance": relevance,
                "semantic_correctness": correctness,
                "raw": args_str,
            }

        content = message.get("content", "")
        log.warning("Judge did not use tool call, got text response", event="warning")
        return null_result(content)

    except (KeyError, IndexError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning(f"Failed to parse judge response: {exc}", event="warning")
        return null_result(raw)


def _make_judge_request(
    messages: list[dict],
    headers: dict,
    max_http_retries: int = 3,
) -> dict | None:
    """Make one judge API call with HTTP-level retries.

    Returns response body dict on success, None on persistent failure.
    """
    global _last_call_time

    min_interval = 60.0 / JUDGE_RATE_LIMIT

    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
        "tools": [SUBMIT_TOOL],
        "tool_choice": "auto",
    }

    for attempt in range(1, max_http_retries + 1):
        with _rate_lock:
            elapsed = time.monotonic() - _last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
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
                log.warning(f"Rate limited ({JUDGE_MODEL}), waiting {wait}s",
                            event="warning", attempt=attempt)
                time.sleep(wait)
                continue

            if response.status_code != 200:
                if attempt < max_http_retries:
                    wait = 2 ** attempt
                    log.warning(f"API error {response.status_code}, retrying in {wait}s",
                                event="warning", attempt=attempt)
                    time.sleep(wait)
                    continue
                log.error(f"Judge API failed ({response.status_code}): {response.text}",
                          event="error")
                return None

            return response.json()

        except httpx.TimeoutException:
            if attempt < max_http_retries:
                log.warning(f"Timeout, retrying ({attempt}/{max_http_retries})",
                            event="warning")
                time.sleep(2 ** attempt)
                continue
            log.error("Judge request timed out after all retries", event="error")
            return None

    return None


def call_judge(
    question: str,
    contexts: list[str],
    prediction: str,
    gold_answers: list[str] | None = None,
    max_tool_retries: int = 5,
) -> dict:
    """Call the judge model via OpenRouter tool-calling.

    Uses multi-turn conversation retry: if the model responds with text
    instead of a tool call, sends back its response with a correction
    message and retries immediately (no delay).

    Returns dict with faithfulness, groundedness, answer_relevance,
    semantic_correctness, and raw fields.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    contexts_block = "\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(contexts)
    )
    gold_answers_block = "\n".join(f"- {a}" for a in (gold_answers or ["N/A"]))

    user_msg = USER_TEMPLATE.format(
        question=question,
        gold_answers_block=gold_answers_block,
        contexts_block=contexts_block,
        prediction=prediction,
    )

    merged_content = f"{SYSTEM_PROMPT}\n\n{user_msg}"
    messages: list[dict] = [{"role": "user", "content": merged_content}]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for tool_attempt in range(1, max_tool_retries + 1):
        body = _make_judge_request(messages, headers)
        if body is None:
            return null_result("API_FAILED")

        result = parse_judge_response(body)
        if result["faithfulness"] is not None:
            return result

        # Model returned text instead of tool call — add correction
        assistant_text = ""
        try:
            assistant_text = body["choices"][0]["message"].get("content", "") or ""
        except (KeyError, IndexError):
            pass

        if tool_attempt < max_tool_retries:
            messages.append({"role": "assistant", "content": assistant_text or "..."})
            messages.append({"role": "user", "content": CORRECTION_MSG})
            # No delay — retry immediately with conversation context

    return null_result("MAX_TOOL_RETRIES")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_judge_scores(rows: list[dict]) -> dict:
    """Compute mean judge metrics from a list of per-query judge result dicts.

    Expects each row to have faithfulness, groundedness, answer_relevance,
    semantic_correctness keys (may be None for skipped/errored rows).

    Returns dict with mean scores (groundedness/relevance/correctness normalised to 0-1).
    """
    faith = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
    ground = [r["groundedness"] for r in rows if r.get("groundedness") is not None]
    relev = [r["answer_relevance"] for r in rows if r.get("answer_relevance") is not None]
    correct = [r["semantic_correctness"] for r in rows if r.get("semantic_correctness") is not None]

    n_judged = len(faith)

    return {
        "faithfulness": round(_mean(faith), 4) if faith else None,
        "groundedness": round(_mean(ground) / 2, 4) if ground else None,
        "answer_relevance": round(_mean(relev) / 2, 4) if relev else None,
        "semantic_correctness": round(_mean(correct) / 2, 4) if correct else None,
        "n_judged": n_judged,
        "n_total": len(rows),
    }
