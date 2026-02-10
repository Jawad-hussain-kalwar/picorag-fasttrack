"""Agentic multi-hop RAG controller (E5).

State machine that reformulates queries and retries retrieval when
the confidence gate rejects hop-1 results.  Uses dual-gate:
  1. Cosine distance threshold (pre-generation)
  2. LLM self-abstention detection (post-generation)

Hop 1: Vector search → gate check → GENERATE or reformulate
Hop 2: Hybrid search (vector + BM25 via RRF) → gate check → GENERATE or ABSTAIN
"""

import re
import time

from src.bm25 import bm25_search
from src.config import (
    E3_ABSTAIN_MESSAGE,
    E5_GATE_THRESHOLD,
    E5_K,
    E5_MAX_HOPS,
    OPENROUTER_MODEL,
)
from src.embed_openrouter import search_with_custom_embeddings
from src.gate import should_abstain
from src.generate import (
    build_e5_mixed_prompt,
    build_reformulation_prompt,
    call_openrouter,
)
from src.hybrid import rrf_fuse
from src.logger import get_logger
from src.metrics import em_loose

log = get_logger("pico-rag.agent")

MAX_GEN_RETRIES = 5

# Phrases the LLM uses to self-abstain (checked case-insensitively)
_ABSTAIN_PHRASES = [
    "not enough evidence",
    "not enough information",
    "cannot be determined",
    "no relevant information",
    "does not contain",
    "do not have enough",
    "cannot answer",
    "unable to answer",
    "not mentioned",
    "no information",
    "insufficient context",
    "insufficient evidence",
]


def _is_self_abstention(prediction: str) -> bool:
    """Check if the LLM's output indicates it chose to abstain."""
    pred_lower = prediction.lower().strip()
    return any(phrase in pred_lower for phrase in _ABSTAIN_PHRASES)


def _clean_prediction(text: str) -> str:
    """Clean Gemma artifacts from generated text.

    Strips stray leading punctuation/brackets the model sometimes emits.
    """
    text = text.strip()
    # Strip leading garbage: }, ], ), *, -, etc. before actual content
    text = re.sub(r'^[\]\}\)\*\-\s:>]+', '', text).strip()
    return text


def _is_empty_or_citation_only(text: str) -> bool:
    """True if prediction has no actual answer text (empty or citations only)."""
    if not text.strip():
        return True
    # Remove all citation markers like [1], [2] and check if anything remains
    stripped = re.sub(r'\[\d+\]', '', text)
    stripped = re.sub(r'[,\s.;:]+', '', stripped)
    return len(stripped) == 0



def reformulate_query(
    query: str,
    snippets: list[str],
    model: str | None = None,
) -> str:
    """Ask LLM to rephrase query. Falls back to original on failure."""
    model = model or OPENROUTER_MODEL
    messages = build_reformulation_prompt(query, snippets)
    try:
        result = call_openrouter(messages, model=model)
        result = result.strip().strip('"').strip("'").strip()
        # Remove leading "Rewritten question:" prefix if model echoes it
        for prefix in ("Rewritten question:", "Rewritten:", "Question:"):
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()
        if not result or len(result) > 200:
            return query
        # Single line only
        result = result.split("\n")[0].strip()
        return result if result else query
    except (RuntimeError, Exception):
        return query


def retrieve_hop(
    query: str,
    k: int,
    hop: int,
    collection_name: str,
    embed_fn,
    bm25_index,
    bm25_pool: list[dict],
) -> tuple[dict, list[float]]:
    """Run retrieval for a single hop.

    Hop 1: vector-only.
    Hop 2: vector + BM25 → RRF hybrid.

    Returns (result_dict, cosine_distances) where cosine_distances
    are from the vector search (used for gating).
    """
    # Always do vector search
    vec_result = search_with_custom_embeddings(
        collection_name, query, n_results=k, embed_fn=embed_fn,
    )
    vec_distances = vec_result.get("distances", [[]])[0][:k]

    if hop == 1:
        return vec_result, vec_distances

    # Hop 2: hybrid (vector + BM25 via RRF)
    bm25_result = bm25_search(bm25_index, bm25_pool, query, n_results=25)
    fused = rrf_fuse(vec_result, bm25_result, k=k)
    return fused, vec_distances


def run_agent_loop(
    query_id: str,
    query: str,
    gold_answers: list[str],
    k: int = E5_K,
    threshold: float = E5_GATE_THRESHOLD,
    collection_name: str = "",
    embed_fn=None,
    bm25_index=None,
    bm25_pool: list[dict] | None = None,
    model: str | None = None,
) -> dict:
    """Run the agentic multi-hop loop for a single query.

    Dual-gate strategy:
      1. Pre-generation: cosine distance threshold rejects obvious misses
      2. Post-generation: if LLM self-abstains, honour that decision

    Returns a dict with all per-query metrics and metadata.
    """
    model = model or OPENROUTER_MODEL
    total_t0 = time.perf_counter()
    tool_calls = 0
    hops_detail: list[dict] = []
    best_confidence = 0.0
    best_hop = 0
    best_result: dict | None = None
    best_distances: list[float] = []
    reformulated_query = ""

    for hop in range(1, E5_MAX_HOPS + 1):
        hop_query = query if hop == 1 else reformulated_query

        # Retrieve
        result, vec_distances = retrieve_hop(
            hop_query, k, hop, collection_name, embed_fn,
            bm25_index, bm25_pool or [],
        )
        tool_calls += 1  # vector search
        if hop == 2:
            tool_calls += 1  # bm25 search

        # Gate check (vector cosine distances)
        abstain, confidence = should_abstain(vec_distances, threshold, method="cosine")

        hop_detail = {
            "hop": hop,
            "query": hop_query,
            "confidence": round(confidence, 6),
            "abstain": abstain,
        }
        hops_detail.append(hop_detail)

        # Track best confidence across hops
        if confidence > best_confidence:
            best_confidence = confidence
            best_hop = hop
            best_result = result
            best_distances = vec_distances

        if not abstain:
            # Gate passed — generate with this hop's context
            break

        if hop < E5_MAX_HOPS:
            # Reformulate for next hop
            snippets = result.get("documents", [[]])[0][:3]
            reformulated_query = reformulate_query(query, snippets, model=model)
            tool_calls += 1  # reformulation LLM call

    # Decision: generate or abstain (pre-generation gate)
    gate_abstain = best_confidence < threshold
    actual_hop = best_hop if not gate_abstain else hop

    if gate_abstain:
        prediction = E3_ABSTAIN_MESSAGE
        gen_ms = 0
        final_abstain = True
    else:
        # Generate answer using best hop's context
        assert best_result is not None
        chunks = best_result.get("documents", [[]])[0][:k]
        messages = build_e5_mixed_prompt(query, chunks)
        gen_t0 = time.perf_counter()
        prediction = ""
        for attempt in range(1, MAX_GEN_RETRIES + 1):
            try:
                prediction = call_openrouter(messages, model=model)
                if prediction:
                    break
                if attempt < MAX_GEN_RETRIES:
                    time.sleep(1)
            except RuntimeError:
                if attempt < MAX_GEN_RETRIES:
                    time.sleep(min(2 ** attempt, 30))
                else:
                    prediction = "ERROR: generation failed"
        gen_ms = int((time.perf_counter() - gen_t0) * 1000)
        tool_calls += 1  # generation LLM call

        # Clean Gemma artifacts
        prediction = _clean_prediction(prediction)

        # Post-generation gate: honour LLM self-abstention or empty output
        if _is_self_abstention(prediction) or _is_empty_or_citation_only(prediction):
            prediction = E3_ABSTAIN_MESSAGE
            final_abstain = True
        else:
            final_abstain = False

    total_ms = int((time.perf_counter() - total_t0) * 1000)

    # Extract supports from best result for retrieval metrics
    use_result = best_result if best_result else result
    metadatas = use_result.get("metadatas", [[]])[0][:k]
    documents = use_result.get("documents", [[]])[0][:k]
    supports = [m.get("support", 0) for m in metadatas]
    distances = best_distances[:k] if best_distances else []

    em_score = em_loose(prediction, gold_answers)

    return {
        "query_id": query_id,
        "query": query,
        "gold_answers": gold_answers,
        "prediction": prediction,
        "em_loose": em_score,
        "hops": actual_hop,
        "tool_calls": tool_calls,
        "reformulated_query": reformulated_query,
        "best_hop": best_hop,
        "documents": documents,
        "distances": distances,
        "supports": supports,
        "confidence": round(best_confidence, 6),
        "abstained": final_abstain,
        "gate_abstain": gate_abstain,
        "llm_abstain": not gate_abstain and final_abstain,
        "generation_ms": gen_ms,
        "total_ms": total_ms,
        "hops_detail": hops_detail,
    }
