from __future__ import annotations

import httpx

import config

_EMBED_URL = f"{config.OLLAMA_BASE_URL}/api/embeddings"

# nomic-embed-text produces 768-dim vectors
EMBEDDING_DIM = 768


def embed_one(text: str) -> list[float]:
    """Embed a single string. Raises on Ollama error."""
    response = httpx.post(
        _EMBED_URL,
        json={"model": config.EMBED_MODEL, "prompt": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Embed a list of strings, batched to avoid overwhelming Ollama.
    Returns embeddings in the same order as the input."""
    if not texts:
        return []

    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            results.append(embed_one(text))

    return results


def check_ollama() -> tuple[bool, str]:
    """Verify Ollama is running and the embed model is available."""
    try:
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        # Ollama tags can be "nomic-embed-text:latest" — check prefix
        available = any(m.startswith(config.EMBED_MODEL) for m in models)
        if not available:
            return False, (
                f"Model '{config.EMBED_MODEL}' not found. "
                f"Run: ollama pull {config.EMBED_MODEL}"
            )
        return True, "OK"
    except httpx.ConnectError:
        return False, (
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. "
            "Run: ollama serve"
        )
    except Exception as exc:
        return False, str(exc)
