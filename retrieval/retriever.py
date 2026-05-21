from __future__ import annotations

from store.vector_store import search, SearchResult
from embed.embedder import embed_one
import config


def retrieve(
    query: str,
    top_k: int = config.TOP_K,
    threshold: float = config.SIMILARITY_THRESHOLD,
    source_filter: str | None = None,
    mmr: bool = True,
) -> list[SearchResult]:
    """Embed a query and return the most relevant chunks.

    Args:
        query:         Natural language question or search string.
        top_k:         Max number of chunks to return.
        threshold:     Minimum cosine similarity (0–1) to include a result.
        source_filter: Optional substring matched against source file paths.
        mmr:           Apply Maximal Marginal Relevance to reduce redundancy
                       when multiple chunks from the same file score highly.
    """
    query_embedding = embed_one(query)

    # Fetch more candidates than needed so MMR has room to diversify
    fetch_k = min(top_k * 3, config.MAX_CONTEXT_CHUNKS * 3) if mmr else top_k

    results = search(
        query_embedding=query_embedding,
        top_k=fetch_k,
        threshold=threshold,
        source_filter=source_filter,
    )

    if mmr and len(results) > top_k:
        results = _mmr(results, top_k)

    return results[:top_k]


def build_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a prompt context block."""
    if not results:
        return ""

    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        source_label = _short_source(r.source)
        parts.append(
            f"[{i}] Source: {source_label} (similarity: {r.similarity:.2f})\n"
            f"{r.content}"
        )

    return "\n\n---\n\n".join(parts)


def _short_source(source: str) -> str:
    """Return just the filename portion of a full path."""
    from pathlib import Path
    return Path(source).name


def _mmr(results: list[SearchResult], k: int) -> list[SearchResult]:
    """Maximal Marginal Relevance: greedily pick chunks that are relevant
    to the query but dissimilar from each other, reducing repetition when
    several chunks from the same file rank highly.

    Uses source-path overlap as a proxy for inter-chunk similarity —
    avoids a full pairwise embedding comparison while still diversifying.
    """
    selected: list[SearchResult] = []
    remaining = list(results)

    # Always take the top result first
    selected.append(remaining.pop(0))

    while remaining and len(selected) < k:
        # Score each candidate: reward relevance, penalise same-source repetition
        best_score = -1.0
        best_idx = 0

        for i, candidate in enumerate(remaining):
            same_source_count = sum(
                1 for s in selected if s.source == candidate.source
            )
            # λ controls relevance vs. diversity trade-off
            lam = 0.6
            score = lam * candidate.similarity - (1 - lam) * same_source_count

            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
