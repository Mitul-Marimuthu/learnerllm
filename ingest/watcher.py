from __future__ import annotations

import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

import config
from ingest.pdf_loader import load_pdf
from ingest.markdown_loader import load_text
from ingest.code_loader import load_code_file
from embed.embedder import embed_batch
from store.vector_store import upsert_documents, Document
from rich.console import Console

console = Console()


def ingest_file(path: Path) -> int:
    """Ingest a single file. Returns the number of chunks stored."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw_chunks = load_pdf(path)
    elif suffix in config.TEXT_EXTENSIONS:
        raw_chunks = load_text(path)
    elif suffix in config.CODE_EXTENSIONS:
        raw_chunks = load_code_file(path)
    else:
        return 0

    if not raw_chunks:
        return 0

    texts = [c["content"] for c in raw_chunks]
    embeddings = embed_batch(texts)

    docs = [
        Document(
            source=c["source"],
            source_type=c["source_type"],
            chunk_index=c["chunk_index"],
            content=c["content"],
            metadata=c["metadata"],
            embedding=emb,
        )
        for c, emb in zip(raw_chunks, embeddings)
    ]

    return upsert_documents(docs)


class _IngestHandler(FileSystemEventHandler):
    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._handle(Path(event.src_path))

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory:
            self._handle(Path(event.src_path))

    def _handle(self, path: Path) -> None:
        try:
            count = ingest_file(path)
            if count:
                console.print(f"[green]✓[/green] {path.name} — {count} chunks ingested")
        except Exception as exc:
            console.print(f"[red]✗[/red] {path.name} — {exc}")


def watch(directory: Path | None = None) -> None:
    """Block and watch a directory, ingesting files as they appear."""
    watch_dir = directory or config.WATCH_DIR
    watch_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Watching[/bold] {watch_dir}  (Ctrl+C to stop)")

    observer = Observer()
    observer.schedule(_IngestHandler(), str(watch_dir), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
