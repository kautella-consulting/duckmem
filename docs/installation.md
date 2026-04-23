# Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Install with uv (Recommended)

```bash
git clone https://github.com/kautella-consulting/duckmem.git
cd duckmem

uv sync

uv sync --group dev
```

## Install with pip

```bash
pip install duckmem
```

Or install from source:

```bash
git clone https://github.com/kautella-consulting/duckmem.git
cd duckmem
pip install -e .
```

## Verify Installation

```bash
# Using uv
uv run python -c "from duckmem import DuckMem; print('DuckMem installed successfully!')"

# Or if installed globally
python -c "from duckmem import DuckMem; print('DuckMem installed successfully!')"
```

## Setting Up Embeddings

DuckMem requires an embedding model for vector search. The default configuration uses Ollama with the `qwen3-embedding:latest` model.

### Option 1: Ollama (Local, Free)

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the embedding model:

```bash
ollama pull qwen3-embedding:latest
```

3. Verify it's working:

```bash
ollama show qwen3-embedding:latest
```

### Option 2: OpenAI

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
export DUCKMEM_EMBED_MODEL="text-embedding-3-small"
export DUCKMEM_EMBED_DIM=1536
```

### Option 3: Other Providers

DuckMem uses LiteLLM, which supports 100+ providers. See [Configuration](configuration.md) for details.

## Setting Up an LLM (for RAG)

For the `ask()` feature, you need a chat model:

### Using Ollama

```bash
ollama pull gpt-oss:20b
export DUCKMEM_LLM_MODEL="ollama/gpt-oss:20b"
```

### Using OpenAI

```bash
export OPENAI_API_KEY="sk-..."
export DUCKMEM_LLM_MODEL="openai/gpt-4o-mini"
```

## Database Extensions

DuckMem automatically installs required DuckDB extensions on first use:

- **fts** - Full-text search with BM25
- **vss** - Vector similarity search with HNSW

No manual installation is required.

## Development Setup

For contributing or development:

```bash
git clone https://github.com/kautella-consulting/duckmem.git
cd duckmem
uv sync --group dev

uv run pytest

uv run ruff check .
uv run ruff format .

uv run pyright
```
