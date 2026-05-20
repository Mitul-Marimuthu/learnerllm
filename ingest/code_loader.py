from __future__ import annotations

from pathlib import Path

import pathspec

import config
from ingest.chunker import chunk_text

# Lines per code chunk (token counting is less meaningful for code)
CODE_CHUNK_LINES = 80
CODE_CHUNK_OVERLAP_LINES = 15

_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".go": "go",
    ".rs": "rust", ".java": "java", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".css": "css", ".html": "html", ".sql": "sql",
    ".sh": "bash", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
}


def load_code_file(path: Path) -> list[dict]:
    """Chunk a single source file by line windows."""
    source = str(path)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = content.splitlines()
    if not lines:
        return []

    lang = _LANGUAGE_MAP.get(path.suffix, "text")
    chunks: list[dict] = []
    chunk_idx = 0

    step = CODE_CHUNK_LINES - CODE_CHUNK_OVERLAP_LINES
    for start in range(0, len(lines), step):
        end = start + CODE_CHUNK_LINES
        window = lines[start:end]
        text = "\n".join(window)
        if not text.strip():
            continue

        chunks.append({
            "source":      source,
            "source_type": "code",
            "chunk_index": chunk_idx,
            "content":     text,
            "metadata": {
                "file_name":  path.name,
                "source_type": "code",
                "language":   lang,
                "line_start": start + 1,
                "line_end":   min(end, len(lines)),
            },
        })
        chunk_idx += 1

    return chunks


def load_code_dir(root: Path) -> list[dict]:
    """Walk a directory and ingest all recognised code files.

    Respects .gitignore if present. Skips binary and oversized files.
    """
    spec = _load_gitignore(root)
    all_chunks: list[dict] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip(path, root, spec):
            continue
        if path.suffix not in config.CODE_EXTENSIONS:
            continue
        if path.stat().st_size > 500_000:  # skip files > 500KB
            continue

        all_chunks.extend(load_code_file(path))

    return all_chunks


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    gitignore = root / ".gitignore"
    if gitignore.exists():
        lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    return None


def _should_skip(path: Path, root: Path, spec: pathspec.PathSpec | None) -> bool:
    # Skip directories in the blocklist
    for part in path.parts:
        if part in config.CODE_SKIP_DIRS:
            return True

    # Skip .gitignore matches
    if spec:
        try:
            rel = path.relative_to(root)
            if spec.match_file(str(rel)):
                return True
        except ValueError:
            pass

    return False
