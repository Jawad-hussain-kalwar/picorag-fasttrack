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
    # Pad with remaining golds if needed
    ideal_full = [1] * min(n_gold, k) + [0] * max(0, k - n_gold)
    idcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(ideal_full)
    )
    if idcg == 0:
        return 0.0
    return dcg / idcg


def parse_citations(text: str) -> list[int]:
    """Extract [N] citation references from generated text.

    Returns sorted deduplicated list of 1-indexed citation numbers.
    """
    matches = re.findall(r"\[(\d+)\]", text)
    return sorted(set(int(m) for m in matches))


def citation_precision(cited_indices: list[int], gold_indices: list[int]) -> float:
    """Fraction of cited chunks that are gold (support=1).

    cited_indices: 1-indexed citation numbers from generated text.
    gold_indices: 1-indexed positions of gold chunks in the retrieved list.
    Returns 0.0 if no citations were made.
    """
    if not cited_indices:
        return 0.0
    gold_set = set(gold_indices)
    hits = sum(1 for c in cited_indices if c in gold_set)
    return hits / len(cited_indices)


def citation_recall(cited_indices: list[int], gold_indices: list[int]) -> float:
    """Fraction of gold chunks that were cited.

    Returns 0.0 if there are no gold chunks in the retrieved list.
    """
    if not gold_indices:
        return 0.0
    cited_set = set(cited_indices)
    hits = sum(1 for g in gold_indices if g in cited_set)
    return hits / len(gold_indices)


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


# ---------------------------------------------------------------------------
# Decision-aware metrics (E3, E5)
# ---------------------------------------------------------------------------

def selective_accuracy(
    predictions: list[str],
    gold_answers: list[list[str]],
    answered_mask: list[bool],
) -> float:
    """EM_loose only over answered (non-abstained) queries."""
    correct: float = 0.0
    total = 0
    for pred, golds, answered in zip(predictions, gold_answers, answered_mask):
        if answered:
            total += 1
            correct += em_loose(pred, golds)
    return correct / total if total > 0 else 0.0


def coverage(answered_mask: list[bool]) -> float:
    """Fraction of queries that were answered (not abstained)."""
    if not answered_mask:
        return 0.0
    return sum(answered_mask) / len(answered_mask)


def auprc(y_true: list[int], y_scores: list[float]) -> float:
    """Area Under Precision-Recall Curve (hand-rolled, no sklearn).

    y_true: binary labels (1 = correct/answerable, 0 = incorrect).
    y_scores: confidence scores (higher = more likely correct).
    Uses trapezoidal integration over the PR curve.
    """
    if not y_true or not y_scores:
        return 0.0

    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0

    # Sort by score descending
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])

    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / n_pos
        precisions.append(prec)
        recalls.append(rec)

    # Step-function integration (matches sklearn average_precision_score)
    area = 0.0
    prev_recall = 0.0
    for prec, rec in zip(precisions, recalls):
        area += prec * (rec - prev_recall)
        prev_recall = rec

    return area


def ece(y_true: list[int], y_conf: list[float], n_bins: int = 10) -> float:
    """Expected Calibration Error (hand-rolled, no sklearn).

    y_true: binary labels (1 = correct, 0 = incorrect).
    y_conf: predicted confidence in [0, 1].
    n_bins: number of equal-width bins.
    """
    if not y_true or not y_conf:
        return 0.0

    n = len(y_true)
    bin_width = 1.0 / n_bins
    total_ece = 0.0

    for b in range(n_bins):
        lo = b * bin_width
        hi = lo + bin_width

        indices = [
            i for i in range(n)
            if (y_conf[i] > lo or b == 0) and y_conf[i] <= hi
        ]
        # First bin includes 0.0
        if b == 0:
            indices = [i for i in range(n) if y_conf[i] >= lo and y_conf[i] <= hi]

        if not indices:
            continue

        bin_acc = sum(y_true[i] for i in indices) / len(indices)
        bin_conf = sum(y_conf[i] for i in indices) / len(indices)
        total_ece += len(indices) / n * abs(bin_acc - bin_conf)

    return total_ece
