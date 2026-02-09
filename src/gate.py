"""Confidence gate for selective answering (E3).

Normalises retrieval scores to a [0, 1] confidence and applies a threshold
to decide whether to answer or abstain.
"""


def normalise_cosine_distance(distance: float) -> float:
    """Convert ChromaDB cosine distance to confidence in [0, 1].

    ChromaDB returns cosine *distance* in [0, 2] (distance = 1 - similarity).
    confidence = 1 - distance/2  maps [0, 2] -> [1, 0].
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def should_abstain(
    distances: list[float],
    threshold: float,
    method: str = "cosine",
) -> tuple[bool, float]:
    """Decide whether to abstain based on retrieval distances.

    Uses top-1 distance only (strongest confidence signal).

    Args:
        distances: retrieval distances/scores for top-k results.
        threshold: confidence threshold — abstain if confidence < threshold.
        method: scoring method. "cosine" for ChromaDB cosine distance,
                "higher_better" for BM25/RRF/Voyage scores (already [0,1]).

    Returns:
        (abstain, confidence) tuple.
    """
    if not distances:
        return True, 0.0

    top1 = distances[0]

    if method == "cosine":
        confidence = normalise_cosine_distance(top1)
    elif method == "higher_better":
        # Already in [0, 1] range (e.g. min-max normalised BM25/RRF)
        confidence = max(0.0, min(1.0, top1))
    else:
        raise ValueError(f"Unknown method: {method}")

    return confidence < threshold, confidence
