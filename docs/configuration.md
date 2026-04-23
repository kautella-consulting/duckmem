# Configuration

DuckMem configuration is centralized in the `Settings` class
([duckmem/config.py](../duckmem/config.py)). Values can come from:

1. Environment variables with the `DUCKMEM_` prefix.
2. A `.env` file in the working directory (see [.env.example](../.env.example)).
3. Explicit kwargs to `Settings(...)` and `DuckMem(settings=...)`.

## Environment variables

All variables use the `DUCKMEM_` prefix. Defaults reflect local Ollama usage; override
for cloud providers.

| Variable | Type | Default | Description |
|---|---|---|---|
| `DUCKMEM_DB_PATH` | str | `"duckmem.duckdb"` | Path to the DuckDB database file. |
| `DUCKMEM_EMBED_MODEL` | str | `"ollama/qwen3-embedding:latest"` | LiteLLM embedding model string. |
| `DUCKMEM_EMBED_DIM` | int | `4096` | Embedding vector dimension (must match the model). |
| `DUCKMEM_LLM_MODEL` | str | `"ollama/gpt-oss:20b"` | LiteLLM chat model used for RAG and extraction. |
| `DUCKMEM_CHUNK_STRATEGY` | str | `"markdown"` | One of `fixed`, `markdown`, `sentence`. |
| `DUCKMEM_CHUNK_MAX_CHARS` | int | `1000` | Maximum characters per chunk. |
| `DUCKMEM_CHUNK_OVERLAP` | int | `100` | Character overlap between consecutive chunks. |
| `DUCKMEM_CHUNK_MIN_CHARS` | int | `50` | Minimum characters for a chunk to be indexed. |
| `DUCKMEM_API_HOST` | str | `"127.0.0.1"` | FastAPI bind host. |
| `DUCKMEM_API_PORT` | int | `8000` | FastAPI bind port. |
| `DUCKMEM_BM25_K1` | float | `1.2` | BM25 term-frequency saturation. |
| `DUCKMEM_BM25_B` | float | `0.75` | BM25 length-normalization factor. |
| `DUCKMEM_HNSW_EF_SEARCH` | int | `64` | HNSW search-time candidate list size. |
| `DUCKMEM_HNSW_M` | int | `16` | HNSW graph connectivity. |
| `DUCKMEM_RRF_K` | int | `60` | Reciprocal Rank Fusion constant for hybrid search. |
| `DUCKMEM_DOCTOR_TIMEOUT_SECONDS` | float \| None | `None` | Optional timeout for `doctor` operations. |

## Using `Settings` from Python

```python
from duckmem import Settings, DuckMem

settings = Settings(
    db_path="knowledge.duckdb",
    embed_model="openai/text-embedding-3-small",
    embed_dim=1536,
    llm_model="openai/gpt-4o-mini",
    chunk_strategy="markdown",
    chunk_max_chars=800,
)

mem = DuckMem(settings=settings)
```

`Settings()` with no arguments reads environment variables and `.env`.
`DuckMem(db_path=...)` without a `settings` argument uses environment defaults.

## Provider recipes

### Ollama (local)

```bash
ollama serve
ollama pull qwen3-embedding:latest
ollama pull gpt-oss:20b

export DUCKMEM_EMBED_MODEL="ollama/qwen3-embedding:latest"
export DUCKMEM_EMBED_DIM=4096
export DUCKMEM_LLM_MODEL="ollama/gpt-oss:20b"
```

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
export DUCKMEM_EMBED_MODEL="openai/text-embedding-3-small"
export DUCKMEM_EMBED_DIM=1536
export DUCKMEM_LLM_MODEL="openai/gpt-4o-mini"
```

### Azure OpenAI

```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2024-02-01"
export DUCKMEM_EMBED_MODEL="azure/your-embedding-deployment"
export DUCKMEM_EMBED_DIM=1536
export DUCKMEM_LLM_MODEL="azure/your-chat-deployment"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export DUCKMEM_LLM_MODEL="anthropic/claude-3-5-sonnet-latest"
```

Anthropic does not provide an embedding API; pair with another provider for
`DUCKMEM_EMBED_MODEL`.

### Google (Gemini)

```bash
export GOOGLE_API_KEY="..."
export DUCKMEM_EMBED_MODEL="gemini/text-embedding-004"
export DUCKMEM_EMBED_DIM=768
export DUCKMEM_LLM_MODEL="gemini/gemini-1.5-flash"
```

### Cohere

```bash
export COHERE_API_KEY="..."
export DUCKMEM_EMBED_MODEL="cohere/embed-english-v3.0"
export DUCKMEM_EMBED_DIM=1024
export DUCKMEM_LLM_MODEL="cohere/command-r"
```

Any [LiteLLM-supported provider](https://docs.litellm.ai/docs/providers) works;
set the matching `EMBED_DIM` explicitly.

## Chunking strategies

Strategy is set via `DUCKMEM_CHUNK_STRATEGY` or `Settings(chunk_strategy=...)`.

| Strategy | Behavior |
|---|---|
| `markdown` (default) | Splits on headings, then falls back to paragraph/line boundaries within each section. |
| `sentence` | Splits on sentence boundaries, packing until `chunk_max_chars`. |
| `fixed` | Fixed-size character windows with `chunk_overlap` overlap. |

## Search tuning

- `DUCKMEM_BM25_K1`, `DUCKMEM_BM25_B` tune the lexical (BM25) component.
- `DUCKMEM_HNSW_EF_SEARCH`, `DUCKMEM_HNSW_M` tune the vector (HNSW) component.
- `DUCKMEM_RRF_K` controls how lexical and semantic rankings fuse; higher values
  give lower-ranked results more weight.
