from __future__ import annotations

import re
from pathlib import Path

from ingest.chunker import chunk_text


def load_text(path: Path) -> list[dict]:
    """Load a markdown or plain-text file and return chunks.

    Heading text (## Introduction) is extracted and stored in metadata
    so the retriever can surface which section a chunk came from.
    """
    source = str(path)
    content = path.read_text(encoding="utf-8", errors="replace")

    headings = _extract_headings(content)
    metadata = {
        "file_name":   path.name,
        "source_type": "markdown",
        "extension":   path.suffix,
        "headings":    headings[:10],  # top 10 headings for context
    }

    return [
        {**chunk, "source_type": "markdown"}
        for chunk in chunk_text(content, source, extra_metadata=metadata)
    ]


def _extract_headings(text: str) -> list[str]:
    return re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
