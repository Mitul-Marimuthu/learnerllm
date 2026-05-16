# Local RAG Study & Research Assistant

A fully local, privacy-first RAG (Retrieval-Augmented Generation) system that acts as your personal research and study assistant. Runs LLM inference and embeddings locally via Ollama, stores vectors in Supabase with pgvector.

---

## Architecture Overview

```
~/rag-docs/              ← drop files here
     │
     ▼
 [Watcher]               ← watches for new/changed files
     │
     ▼
 [Loaders]               ← PDF · Markdown/Text · Code
     │
     ▼
 [Chunker]               ← splits content into overlapping chunks
     │
     ▼
 [Embedder]              ← nomic-embed-text via Ollama (local)
     │
     ▼
 [Supabase pgvector]     ← stores vectors + metadata
     │
     ▼
 [Retriever]             ← cosine similarity search
     │
     ▼
 [LLM Synthesis]         ← qwen2.5:7b via Ollama (local)
     │
     ▼
 [CLI / Web UI]          ← you ask, it answers
```

---

## Stack

| Layer | Tool | Notes |
|---|---|---|
| LLM inference | Ollama + `qwen2.5:7b` | Local, free, M4-optimized |
| Embeddings | Ollama + `nomic-embed-text` | 768-dim, runs locally |
| Vector store | Supabase + pgvector | Free tier, no local DB |
| PDF parsing | `pypdf` + `pdfplumber` | Handles research papers and slides |
| Markdown/text | Native Python | Simple and fast |
| Code ingestion | `tree-sitter` | Language-aware chunking |
| File watching | `watchdog` | Monitors `~/rag-docs` in real-time |
| CLI | `rich` + `prompt_toolkit` | Clean terminal interface |
| Web UI | Single `web_ui.html` | No framework, pure HTML/JS |

---

## Project Structure

```
rag-assistant/
├── README.md
├── requirements.txt
├── .env.example
│
├── config.py                  # Central config — paths, model names, chunk sizes
│
├── ingest/
│   ├── __init__.py
│   ├── pdf_loader.py          # Extracts text from PDFs page-by-page
│   ├── markdown_loader.py     # Loads .md and .txt files
│   ├── code_loader.py         # Walks code folders, respects .gitignore
│   ├── chunker.py             # Overlapping sliding-window chunker
│   └── watcher.py             # watchdog-based folder monitor
│
├── embed/
│   ├── __init__.py
│   └── embedder.py            # Calls Ollama /api/embeddings endpoint
│
├── store/
│   ├── __init__.py
│   └── vector_store.py        # Supabase client: upsert, search, delete
│
├── retrieval/
│   ├── __init__.py
│   └── retriever.py           # Top-k search + optional MMR reranking
│
├── query/
│   ├── __init__.py
│   ├── cli.py                 # Interactive REPL — `python -m query.cli`
│   └── web_ui.html            # Standalone single-file web interface
│
└── main.py                    # Entry points: `ingest`, `watch`, `query`
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- A [Supabase](https://supabase.com) free-tier project

### 2. Pull Ollama models

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Verify both are available:

```bash
ollama list
```

### 3. Supabase setup

In your Supabase project, run this SQL in the **SQL Editor**:

```sql
-- Enable the pgvector extension
create extension if not exists vector;

-- Main documents table
create table documents (
  id          uuid primary key default gen_random_uuid(),
  source      text not null,           -- file path
  source_type text not null,           -- 'pdf' | 'markdown' | 'code'
  chunk_index integer not null,
  content     text not null,
  metadata    jsonb default '{}',
  embedding   vector(768),             -- nomic-embed-text dimension
  created_at  timestamptz default now()
);

-- IVFFlat index for fast approximate nearest-neighbor search
create index on documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Function used by the retriever
create or replace function match_documents(
  query_embedding vector(768),
  match_threshold float,
  match_count     int
)
returns table (
  id          uuid,
  source      text,
  source_type text,
  content     text,
  metadata    jsonb,
  similarity  float
)
language sql stable
as $$
  select
    id, source, source_type, content, metadata,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  where 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

### 4. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env with your Supabase URL and anon key
```

---

## Usage

### Ingest files manually

```bash
# Ingest a single PDF
python main.py ingest ~/Downloads/attention-is-all-you-need.pdf

# Ingest a folder of notes
python main.py ingest ~/notes/

# Ingest a code project
python main.py ingest ~/projects/my-app/ --type code
```

### Watch a folder (auto-ingest on file changes)

```bash
python main.py watch ~/rag-docs/
```

Drop any PDF, Markdown, or code folder into `~/rag-docs/` and it ingests automatically.

### Query — CLI

```bash
python main.py query

# Or pipe a one-shot question
echo "How does auth work in my project?" | python main.py query --once
```

The CLI supports:
- Multi-turn conversation with source citations
- `sources` command to see what's in the vector store
- `clear` to reset conversation history

### Query — Web UI

```bash
# Start the local API server
python main.py serve

# Open web_ui.html in your browser (double-click or file:// URL)
```

---

## Example Queries

| Query | What happens |
|---|---|
| `Explain the method in Attention Is All You Need simply` | Retrieves chunks from that paper, summarizes via qwen2.5 |
| `How does auth work in my project?` | Finds relevant code chunks, explains the flow |
| `Summarize everything I have on transformers` | Aggregates across all sources on that topic |
| `How do I implement what this paper describes in Python?` | Retrieves paper chunks, generates code with context |

---

## Configuration Reference

All tunables live in `config.py`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `WATCH_DIR` | `~/rag-docs` | Folder monitored for new files |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to include a chunk |
| `LLM_MODEL` | `qwen2.5:7b` | Ollama model for answer generation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |

---

## How Each Component Works

### Ingestion pipeline (`ingest/`)

- **`pdf_loader.py`** — Uses `pdfplumber` for layout-aware extraction (handles multi-column papers and slide decks). Falls back to `pypdf` for simple text-layer PDFs.
- **`markdown_loader.py`** — Reads `.md` and `.txt` files, preserves heading structure as metadata.
- **`code_loader.py`** — Walks directories, skips `node_modules`/`.git`/build artifacts, chunks by file with language tag in metadata.
- **`chunker.py`** — Sliding-window chunker with configurable size and overlap. Respects paragraph and sentence boundaries so chunks don't split mid-sentence.
- **`watcher.py`** — Uses `watchdog` to detect `created`/`modified` events and triggers the ingestion pipeline automatically.

### Embedding (`embed/embedder.py`)

Calls Ollama's `/api/embeddings` endpoint with `nomic-embed-text`. Batches requests to avoid hammering the local server. Returns 768-dimensional float vectors.

### Vector store (`store/vector_store.py`)

Wraps Supabase's Python client. On ingest, upserts rows keyed by `(source, chunk_index)` so re-ingesting a changed file updates existing vectors rather than creating duplicates. On query, calls the `match_documents` RPC function.

### Retrieval (`retrieval/retriever.py`)

Embeds the query, calls `match_documents`, returns top-k chunks ranked by cosine similarity. Optionally applies MMR (Maximal Marginal Relevance) to reduce redundancy when multiple chunks from the same source are retrieved.

### LLM synthesis

A system prompt instructs qwen2.5:7b to answer using only the retrieved context, cite sources by filename, and flag when it doesn't have enough information. Conversation history is maintained in-memory for multi-turn CLI sessions.

---

## Limitations & Known Constraints

- **Supabase free tier** has a 500MB database limit and 50K rows. Sufficient for hundreds of papers and projects.
- **nomic-embed-text** produces 768-dim vectors. The IVFFlat index works well from ~1K rows; below that, a sequential scan is faster (pgvector handles this automatically with `lists = 1`).
- **qwen2.5:7b** context window is 32K tokens. The retriever is capped to avoid exceeding this with too many chunks.
- **Code ingestion** does not yet parse ASTs — files are chunked by line count, not by function/class boundaries. This is a planned improvement.
- The **web UI** requires the local API server (`python main.py serve`) to be running. It does not bundle a backend.

---

## Roadmap

- [ ] AST-aware chunking for Python/JS/TS via tree-sitter
- [ ] Hybrid search (BM25 + vector) for keyword-heavy queries
- [ ] Document deduplication fingerprinting
- [ ] Re-ranking with a small cross-encoder model
- [ ] Conversation history persistence across sessions
- [ ] Multi-collection support (separate namespaces per project)

---

## Environment Variables

See `.env.example` for all required and optional variables:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-key

# Optional overrides (defaults shown)
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b
EMBED_MODEL=nomic-embed-text
WATCH_DIR=~/rag-docs
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5
```

---

## License

MIT — use freely, modify openly.
