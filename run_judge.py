"""LLM-as-Judge for Faithfulness, Groundedness, Answer Relevance & Semantic Correctness.

Supports E1 and E2 experiment outputs.

Usage:
    .venv\\Scripts\\python.exe run_judge.py runs/e1/<run_dir> --k 3 5 10
    .venv\\Scripts\\python.exe run_judge.py runs/e2/<run_dir> --experiment e2 --config 1_vector_minilm --k 3 5 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

SUBMIT_TOOL = {
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
    "You are a strict, impartial evaluator. You MUST respond by calling the "
    "submit_judgment tool. Do NOT write any text — ONLY call the tool.\n\n"
    "Your job is to evaluate an Answer against Retrieved Contexts and Gold Answers.\n\n"
    "RULES:\n"
    "- For faithfulness/groundedness: ONLY judge based on Retrieved Contexts.\n"
    "- For answer_relevance: judge whether the Answer addresses the Question.\n"
    "- For semantic_correctness: compare the Answer to the Gold Answers.\n"
    "- You MUST use the submit_judgment tool. Do NOT write a text reply."
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
# Judge API caller
# ---------------------------------------------------------------------------

_last_call_time: float = 0.0


def call_openrouter_judge(
    question: str,
    contexts: list[str],
    prediction: str,
    gold_answers: list[str] | None = None,
    max_retries: int = 3,
) -> dict:
    """Call the judge model via OpenRouter tool-calling.

    Returns dict with faithfulness, groundedness, answer_relevance, semantic_correctness, raw.
    """
    global _last_call_time

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

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": merged_content}],
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
        "tools": [SUBMIT_TOOL],
        "tool_choice": "auto",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    min_interval = 60.0 / JUDGE_RATE_LIMIT
    elapsed = time.monotonic() - _last_call_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

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
                log.warning(f"Rate limited ({JUDGE_MODEL}), waiting {wait}s", event="warning", attempt=attempt)
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
                log.error(f"Judge API failed ({response.status_code}): {response.text}", event="error")
                return _null_result(response.text)

            body = response.json()
            return _parse_judge_response(body)

        except httpx.TimeoutException:
            if attempt < max_retries:
                log.warning(f"Timeout, retrying ({attempt}/{max_retries})", event="warning")
                time.sleep(2 ** attempt)
                continue
            log.error("Judge request timed out after all retries", event="error")
            return _null_result("TIMEOUT")

    return _null_result("MAX_RETRIES")


def _null_result(raw: str) -> dict:
    return {
        "faithfulness": None, "groundedness": None,
        "answer_relevance": None, "semantic_correctness": None,
        "raw": raw,
    }


def _parse_judge_response(body: dict) -> dict:
    """Extract 4 judge scores from tool-call response."""
    raw = json.dumps(body, ensure_ascii=False)
    try:
        choices = body.get("choices", [])
        if not choices:
            err = body.get("error", {}).get("message", str(body))
            log.warning(f"No choices in judge response: {err}", event="warning")
            return _null_result(raw)

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
        return _null_result(content)

    except (KeyError, IndexError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning(f"Failed to parse judge response: {exc}", event="warning")
        return _null_result(raw)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_and_join(run_dir: Path, k: int, experiment: str = "e1", config_id: str | None = None) -> list[dict]:
    """Load generation + retrieval JSONL and join by query_id.

    Returns list of dicts with keys: query_id, query, prediction, documents, gold_answers.
    """
    if experiment == "e2" and config_id:
        gen_path = run_dir / "samples" / f"e2_mixed_{config_id}_k{k}.jsonl"
        ret_path = run_dir / "retrieval" / config_id / f"retrieved_k{k}.jsonl"
    else:
        gen_path = run_dir / "samples" / f"e1_mixed_k{k}.jsonl"
        ret_path = run_dir / "retrieval" / f"retrieved_k{k}.jsonl"

    if not gen_path.exists():
        raise FileNotFoundError(f"Generation file not found: {gen_path}")
    if not ret_path.exists():
        raise FileNotFoundError(f"Retrieval file not found: {ret_path}")

    gen_rows = load_jsonl(gen_path)
    ret_rows = load_jsonl(ret_path)

    ret_by_qid = {r["query_id"]: r for r in ret_rows}

    joined = []
    for g in gen_rows:
        qid = g["query_id"]
        r = ret_by_qid.get(qid)
        if r is None:
            log.warning(f"No retrieval data for query_id={qid}, skipping", event="warning")
            continue
        joined.append({
            "query_id": qid,
            "query": g["query"],
            "prediction": g["prediction"],
            "documents": r["documents"],
            "gold_answers": g.get("gold_answers", []),
        })

    return joined


# ---------------------------------------------------------------------------
# Judge runner
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_judge_for_k(
    run_dir: Path,
    k: int,
    experiment: str = "e1",
    config_id: str | None = None,
) -> dict:
    """Run the judge on all mixed-mode predictions for a given k.

    Returns aggregated metrics dict.
    """
    joined = load_and_join(run_dir, k, experiment=experiment, config_id=config_id)
    total = len(joined)
    log.info(f"Judging k={k}: {total} queries loaded", event="info")

    # Checkpoint file
    if experiment == "e2" and config_id:
        ckpt_path = run_dir / "samples" / f"judge_{config_id}_k{k}.jsonl"
    else:
        ckpt_path = run_dir / "samples" / f"judge_k{k}.jsonl"

    judged_ids: set[str] = set()

    if ckpt_path.exists():
        existing = load_jsonl(ckpt_path)
        judged_ids = {r["query_id"] for r in existing}
        log.info(f"Resuming: {len(judged_ids)} already judged for k={k}", event="info")

    skipped = 0
    errors = 0

    for item in joined:
        qid = item["query_id"]

        if qid in judged_ids:
            continue

        if item["prediction"].startswith("ERROR:"):
            skipped += 1
            result = {
                "query_id": qid,
                "faithfulness": None, "groundedness": None,
                "answer_relevance": None, "semantic_correctness": None,
                "judge_raw": "SKIPPED_ERROR_PREDICTION",
            }
            with open(ckpt_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            judged_ids.add(qid)
            continue

        result = call_openrouter_judge(
            question=item["query"],
            contexts=item["documents"],
            prediction=item["prediction"],
            gold_answers=item.get("gold_answers"),
        )

        row = {
            "query_id": qid,
            "faithfulness": result["faithfulness"],
            "groundedness": result["groundedness"],
            "answer_relevance": result["answer_relevance"],
            "semantic_correctness": result["semantic_correctness"],
            "judge_raw": result["raw"],
        }

        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        judged_ids.add(qid)

        if result["faithfulness"] is None:
            errors += 1

        done = len(judged_ids)
        if done % 10 == 0 or done == total:
            print(f"  k={k}: {done}/{total} judged ({errors} parse errors, {skipped} skipped)")

    # Aggregate
    all_rows = load_jsonl(ckpt_path)
    faith = [r["faithfulness"] for r in all_rows if r.get("faithfulness") is not None]
    ground = [r["groundedness"] for r in all_rows if r.get("groundedness") is not None]
    relev = [r["answer_relevance"] for r in all_rows if r.get("answer_relevance") is not None]
    correct = [r["semantic_correctness"] for r in all_rows if r.get("semantic_correctness") is not None]

    n_judged = len(faith)

    agg = {
        "faithfulness": round(_mean(faith), 4) if faith else None,
        "groundedness": round(_mean(ground) / 2, 4) if ground else None,
        "answer_relevance": round(_mean(relev) / 2, 4) if relev else None,
        "semantic_correctness": round(_mean(correct) / 2, 4) if correct else None,
        "n_judged": n_judged,
        "n_total": len(all_rows),
    }

    print(f"  k={k} done — faith={agg['faithfulness']}  ground={agg['groundedness']}  "
          f"relev={agg['answer_relevance']}  correct={agg['semantic_correctness']}  "
          f"judged={n_judged}")
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge for E1/E2 Mixed-mode evaluation")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10], help="k values to judge (default: 3 5 10)")
    parser.add_argument("--experiment", choices=["e1", "e2"], default="e1", help="Experiment type (default: e1)")
    parser.add_argument("--config", type=str, nargs="+", help="Config IDs for E2 (e.g. 1_vector_minilm)")
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: run directory not found: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Experiment: {args.experiment}")
    print(f"Run directory: {args.run_dir}")
    print(f"k values: {args.k}")
    if args.config:
        print(f"Config IDs: {args.config}")
    print()

    all_metrics = {}

    if args.experiment == "e2" and args.config:
        for cfg_id in args.config:
            for k in args.k:
                try:
                    agg = run_judge_for_k(args.run_dir, k, experiment="e2", config_id=cfg_id)
                    all_metrics[f"{cfg_id}_k={k}"] = agg
                except FileNotFoundError as e:
                    print(f"  Skipping {cfg_id} k={k}: {e}")
    else:
        for k in args.k:
            try:
                agg = run_judge_for_k(args.run_dir, k, experiment=args.experiment)
                all_metrics[f"k={k}"] = agg
            except FileNotFoundError as e:
                print(f"  Skipping k={k}: {e}")

    if all_metrics:
        metrics_path = args.run_dir / "judge_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nAggregated metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
