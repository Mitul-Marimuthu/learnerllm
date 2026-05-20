from __future__ import annotations

import re
import tiktoken

import config

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries while preserving the delimiter."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def chunk_text(text: str, source: str, extra_metadata: dict | None = None) -> list[dict]:
    """Split text into overlapping token-bounded chunks.

    Tries to break at paragraph → sentence boundaries rather than mid-word.
    Returns a list of dicts ready to be passed to the embedder/store.
    """
    if not text.strip():
        return []

    size    = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP
    meta    = extra_metadata or {}

    # Split into paragraphs first, then sentences within each
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        sentences.extend(_split_sentences(para))
        sentences.append("")  # paragraph boundary marker

    chunks: list[dict] = []
    current_tokens: list[str] = []   # token strings in the current window
    current_text_parts: list[str] = []

    def flush(idx: int) -> None:
        joined = " ".join(p for p in current_text_parts if p)
        if joined.strip():
            chunks.append({
                "source":      source,
                "chunk_index": idx,
                "content":     joined.strip(),
                "metadata":    meta,
            })

    chunk_idx = 0
    for sentence in sentences:
        if not sentence:
            continue
        s_tokens = _enc.encode(sentence)

        # If adding this sentence would exceed the window, flush and roll back
        if _count_tokens(" ".join(current_text_parts)) + len(s_tokens) > size and current_text_parts:
            flush(chunk_idx)
            chunk_idx += 1

            # Keep overlap: walk backwards until we've kept ~overlap tokens
            kept_parts: list[str] = []
            kept_count = 0
            for part in reversed(current_text_parts):
                part_tokens = len(_enc.encode(part))
                if kept_count + part_tokens > overlap:
                    break
                kept_parts.insert(0, part)
                kept_count += part_tokens
            current_text_parts = kept_parts

        current_text_parts.append(sentence)

    # Flush whatever remains
    if current_text_parts:
        flush(chunk_idx)

    return chunks
