from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _required(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value


def _path(key: str, default: str) -> Path:
    raw = os.getenv(key, default)
    return Path(raw).expanduser().resolve()


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL: str = _required("SUPABASE_URL")
SUPABASE_KEY: str = _required("SUPABASE_KEY")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str      = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL: str    = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ── Ingestion ─────────────────────────────────────────────────────────────────
WATCH_DIR: Path    = _path("WATCH_DIR", "~/rag-docs")
CHUNK_SIZE: int    = _int("CHUNK_SIZE", 512)      # tokens
CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 64)    # tokens

# Extensions recognised as plain text / markdown
TEXT_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt", ".rst"})

# Extensions recognised as code (language tagged in metadata)
CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".css", ".html", ".sql", ".sh", ".yaml", ".yml", ".toml",
})

# Directories to skip when walking code projects
CODE_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".next", "dist", "build", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "target",  # Rust
})

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int                    = _int("TOP_K", 5)
SIMILARITY_THRESHOLD: float   = _float("SIMILARITY_THRESHOLD", 0.3)

# Maximum chunks fed to the LLM in a single prompt (guards context window)
MAX_CONTEXT_CHUNKS: int = 8

# ── Web server ────────────────────────────────────────────────────────────────
SERVER_HOST: str = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT: int = _int("SERVER_PORT", 8000)
