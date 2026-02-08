import math
import re
import string


# ---------------------------------------------------------------------------
# Text normalisation (matches MIRAGE evaluation.py conventions)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ---------------------------------------------------------------------------
# Generation quality metrics
# ---------------------------------------------------------------------------

def em_loose(prediction: str, answers: list[str]) -> float:
    """1.0 if any gold answer appears as substring in prediction."""
    pred_norm = _normalize(prediction)
    return 1.0 if any(_normalize(a) in pred_norm for a in answers) else 0.0


def em_strict(prediction: str, answers: list[str]) -> float:
    """1.0 if normalised prediction exactly matches any gold answer."""
    pred_norm = _normalize(prediction)
    return 1.0 if any(_normalize(a) == pred_norm for a in answers) else 0.0


def f1_score(prediction: str, answers: list[str]) -> float:
    """Token-level F1 between prediction and best-matching gold answer."""
    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for ans in answers:
        ans_tokens = _tokenize(ans)
        if not ans_tokens:
            continue
        common = sum(1 for t in pred_tokens if t in ans_tokens)
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(ans_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


# ---------------------------------------------------------------------------
# Retrieval quality metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_supports: list[int]) -> float:
    """Did at least one gold chunk (support=1) appear in retrieved results?

    Binary recall: 1.0 if any retrieved chunk has support=1, else 0.0.
    retrieved_supports: list of support labels (0 or 1) in retrieval order.
    """
    return 1.0 if any(s == 1 for s in retrieved_supports) else 0.0


def precision_at_k(retrieved_supports: list[int]) -> float:
    """Fraction of retrieved chunks that are gold (support=1)."""
    if not retrieved_supports:
        return 0.0
    return sum(1 for s in retrieved_supports if s == 1) / len(retrieved_supports)


def ndcg_at_k(retrieved_supports: list[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k.

    Uses binary relevance (support label). Gold chunks scored as 1, others as 0.
    """
    if not retrieved_supports:
        return 0.0
    # DCG
    dcg = sum(
        rel / math.log2(i + 2)  # i+2 because log2(1)=0, ranks start at 1
        for i, rel in enumerate(retrieved_supports[:k])
    )
    # IDCG: best possible ranking (all golds first)
    n_gold = sum(1 for s in retrieved_supports if s == 1)
    ideal = sorted(retrieved_supports[:k], reverse=True)
    # Pad with remaining golds if needed
    ideal_full = [1] * min(n_gold, k) + [0] * max(0, k - n_gold)
    idcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(ideal_full)
    )
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(retrieved_supports: list[int]) -> float:
    """Mean Reciprocal Rank: 1/rank of first gold chunk.

    Returns 0.0 if no gold chunk in results.
    """
    for i, s in enumerate(retrieved_supports):
        if s == 1:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# MIRAGE RAG adaptability metrics
# ---------------------------------------------------------------------------

def compute_mirage_metrics(
    base_labels: list[float],
    oracle_labels: list[float],
    mixed_labels: list[float],
) -> dict[str, float]:
    """Compute NV, CA, CI, CM from per-query binary correctness labels.

    Each label list is parallel (same index = same query). Values are 0.0 or 1.0.

    NV (Noise Vulnerability): base=1, mixed=0 → distractors hurt
    CA (Context Acceptability): base=0, mixed=1 → RAG context helped
    CI (Context Insensitivity): base=0, oracle=0 → LLM can't use context
    CM (Context Misinterpretation): base=1, oracle=0 → context confuses LLM
    """
    n = len(base_labels)
    if n == 0:
        return {"NV": 0.0, "CA": 0.0, "CI": 0.0, "CM": 0.0}

    nv = sum(1 for b, m in zip(base_labels, mixed_labels) if b == 1.0 and m == 0.0)
    ca = sum(1 for b, m in zip(base_labels, mixed_labels) if b == 0.0 and m == 1.0)
    ci = sum(1 for b, o in zip(base_labels, oracle_labels) if b == 0.0 and o == 0.0)
    cm = sum(1 for b, o in zip(base_labels, oracle_labels) if b == 1.0 and o == 0.0)

    return {
        "NV": nv / n,
        "CA": ca / n,
        "CI": ci / n,
        "CM": cm / n,
    }
