from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from supabase import create_client, Client

import config


@dataclass
class Document:
    source: str
    source_type: str       # 'pdf' | 'markdown' | 'code'
    chunk_index: int
    content: str
    metadata: dict[str, Any]
    embedding: list[float]


@dataclass
class SearchResult:
    id: str
    source: str
    source_type: str
    content: str
    metadata: dict[str, Any]
    similarity: float


_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    return _client


def upsert_documents(docs: list[Document]) -> int:
    """Insert or update chunks. Keyed on (source, chunk_index) so re-ingesting
    a file replaces its old vectors rather than duplicating them."""
    if not docs:
        return 0

    client = _get_client()

    # Fetch existing IDs for this source so we can delete stale tail chunks
    # (e.g. file shrank after re-ingest and now has fewer chunks than before).
    source = docs[0].source
    existing = (
        client.table("documents")
        .select("id, chunk_index")
        .eq("source", source)
        .execute()
    )
    existing_indices = {row["chunk_index"] for row in (existing.data or [])}
    new_indices = {doc.chunk_index for doc in docs}
    stale_indices = existing_indices - new_indices

    if stale_indices:
        client.table("documents").delete().eq("source", source).in_(
            "chunk_index", list(stale_indices)
        ).execute()

    rows = [
        {
            "source":      doc.source,
            "source_type": doc.source_type,
            "chunk_index": doc.chunk_index,
            "content":     doc.content,
            "metadata":    doc.metadata,
            "embedding":   doc.embedding,
        }
        for doc in docs
    ]

    # upsert with on-conflict targeting the unique constraint
    client.table("documents").upsert(
        rows,
        on_conflict="source,chunk_index",
    ).execute()

    return len(rows)


def search(
    query_embedding: list[float],
    top_k: int = config.TOP_K,
    threshold: float = config.SIMILARITY_THRESHOLD,
    source_filter: str | None = None,
) -> list[SearchResult]:
    """Cosine similarity search via the match_documents RPC."""
    client = _get_client()

    params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "match_threshold": threshold,
        "match_count":     top_k,
    }

    response = client.rpc("match_documents", params).execute()
    rows = response.data or []

    if source_filter:
        rows = [r for r in rows if source_filter in r["source"]]

    return [
        SearchResult(
            id=r["id"],
            source=r["source"],
            source_type=r["source_type"],
            content=r["content"],
            metadata=r["metadata"] or {},
            similarity=r["similarity"],
        )
        for r in rows
    ]


def delete_source(source: str) -> int:
    """Remove all chunks for a given source file path."""
    client = _get_client()
    response = (
        client.table("documents")
        .delete()
        .eq("source", source)
        .execute()
    )
    return len(response.data or [])


def list_sources() -> list[dict[str, Any]]:
    """Return one row per unique source with chunk count and type."""
    client = _get_client()
    response = (
        client.table("documents")
        .select("source, source_type, chunk_index")
        .order("source")
        .execute()
    )
    rows = response.data or []

    # Aggregate in Python — avoids needing a GROUP BY RPC
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        src = row["source"]
        if src not in seen:
            seen[src] = {"source": src, "source_type": row["source_type"], "chunks": 0}
        seen[src]["chunks"] += 1

    return list(seen.values())
