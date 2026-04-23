# DuckMem

A DuckDB-based personal knowledge memory system with hybrid search, knowledge graph, and RAG.

## Features

- **Single-file storage**: All data in one `.duckdb` file
- **Hybrid search**: BM25 lexical + HNSW vector search with RRF fusion
- **Knowledge graph**: Entity extraction, state tracking, graph traversal
- **RAG Q&A**: PydanticAI-powered retrieval-augmented generation
- **Universal LLM support**: LiteLLM for 100+ embedding/LLM providers
- **Multiple interfaces**: Python SDK, FastAPI REST, FastMCP server

## Installation

```bash
# Using uv (recommended)
uv add duckmem

# Or with pip
pip install duckmem
```

## Quick Start

```python
from duckmem import DuckMem

# Create a knowledge base
mem = DuckMem("knowledge.duckdb")

# Add documents
item_id = mem.add(
    "Transformers use attention mechanisms for sequence modeling.",
    title="ML Notes",
)

# Search
results = mem.search("attention mechanism", top_k=5)
for r in results:
    print(f"{r.score:.2f}: {r.chunk.text}")

# RAG Q&A
answer = await mem.ask("What do transformers use?")
print(answer.answer)

# Knowledge graph
mem.add_relation("Transformers", "use", "attention")
state = mem.state("Transformers")
print(state.properties)  # {'use': 'attention'}

# Close when done
mem.close()
```

## Configuration

Configure via environment variables (prefix `DUCKMEM_`):

```bash
export DUCKMEM_EMBED_MODEL="text-embedding-3-small"
export DUCKMEM_LLM_MODEL="ollama/gpt-oss:20b"  # or "openai/gpt-4o-mini" for cloud
export DUCKMEM_CHUNK_STRATEGY="markdown"
```

Or via `.env` file or Python:

```python
from duckmem import Settings, DuckMem

settings = Settings(
    embed_model="text-embedding-3-small",
    llm_model="ollama/gpt-oss:20b",  # or "openai/gpt-4o-mini" for cloud
    chunk_strategy="markdown",
)
mem = DuckMem("knowledge.duckdb", settings=settings)
```

## CLI

```bash
# Create database
duckmem create knowledge.duckdb

# Add content
duckmem add knowledge.duckdb --text "Your content here" --title "Title"

# Search
duckmem search knowledge.duckdb --query "search terms" --top-k 10

# Statistics
duckmem stats knowledge.duckdb

# Maintenance
duckmem verify knowledge.duckdb --deep
duckmem doctor knowledge.duckdb --vacuum --rebuild-fts
```

## FastAPI Server

```bash
duckmem serve knowledge.duckdb --host 127.0.0.1 --port 8000

uv run uvicorn duckmem.api:app --reload
```

API endpoints:
- `POST /items` - Add item
- `GET /search` - Search
- `POST /ask` - RAG Q&A
- `GET /entities/{name}/state` - Entity state
- `GET /stats` - Database stats

See [docs/rest-api.md](docs/rest-api.md) for full request/response schemas.

## MCP Server

For use with Claude Desktop or other MCP clients:

```bash
uv run duckmem-mcp
```

Example Claude Desktop / MCP client config:

```json
{
  "mcpServers": {
    "duckmem": {
      "command": "uv",
      "args": ["run", "duckmem-mcp"],
      "cwd": "/absolute/path/to/duckmem",
      "env": {
        "DUCKMEM_DB_PATH": "/absolute/path/to/knowledge.duckdb"
      }
    }
  }
}
```

See [docs/mcp-server.md](docs/mcp-server.md) for the full tool surface.

## Development

```bash
git clone https://github.com/kautella-consulting/duckmem
cd duckmem
uv sync --group dev

cp .env.example .env

uv run pytest -v

uv run ruff check .
uv run ruff format .
```

## Architecture

```
duckmem/
    core.py          # DuckMem class and core functions
    models.py        # Pydantic models
    config.py        # Settings management
    schema.py        # Database DDL
    inference.py     # LiteLLM embeddings wrapper
    agents.py        # PydanticAI agents for RAG/extraction
    api.py           # FastAPI REST endpoints
    mcp_server.py    # FastMCP server
    cli.py           # Typer CLI
    ingestion/
        chunkers.py  # Text chunking strategies
```

## License

MIT
