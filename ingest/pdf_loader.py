from __future__ import annotations

from pathlib import Path

from ingest.chunker import chunk_text


def load_pdf(path: Path) -> list[dict]:
    """Extract text from a PDF and return chunks.

    Uses pdfplumber for layout-aware extraction (handles multi-column papers
    and slides). Falls back to pypdf if pdfplumber fails on a given page.
    """
    source = str(path)
    full_text_parts: list[str] = []

    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text and text.strip():
                    full_text_parts.append(text)
                else:
                    # Fallback for this page
                    full_text_parts.append(_pypdf_page(path, page_num - 1))
    except Exception:
        # pdfplumber failed entirely — use pypdf for the whole doc
        full_text_parts = [_pypdf_all(path)]

    full_text = "\n\n".join(p for p in full_text_parts if p and p.strip())

    metadata = {
        "file_name":   path.name,
        "source_type": "pdf",
        "pages":       len(full_text_parts),
    }

    return [
        {**chunk, "source_type": "pdf"}
        for chunk in chunk_text(full_text, source, extra_metadata=metadata)
    ]


def _pypdf_page(path: Path, page_index: int) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        if page_index < len(reader.pages):
            return reader.pages[page_index].extract_text() or ""
    except Exception:
        pass
    return ""


def _pypdf_all(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except Exception:
        return ""
