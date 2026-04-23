# DuckMem Documentation

**DuckMem** is a DuckDB-based personal knowledge memory system that provides document ingestion, hybrid search, knowledge graph, and RAG (Retrieval-Augmented Generation) capabilities in a single-file database.

## Features

- **Single-file storage** - All data stored in one portable `.duckdb` file
- **Hybrid search** - Combines BM25 lexical search with HNSW vector similarity
- **Knowledge graph** - Entity extraction, state tracking, and graph traversal
- **RAG Q&A** - Ask questions and get AI-generated answers from your knowledge base
- **Universal LLM support** - Works with 100+ providers via LiteLLM
- **Multiple interfaces** - Python SDK, FastAPI REST API, MCP server, CLI

## Quick Start

```python
import asyncio
from duckmem import DuckMem

async def main():
    with DuckMem("knowledge.duckdb") as mem:
        mem.add("Transformers use attention mechanisms.", title="ML Notes")

        for r in mem.search("attention", top_k=5):
            print(f"{r.score:.3f}  {r.chunk.text[:80]}...")

        answer = await mem.ask("What do transformers use?")
        print(answer.answer)

asyncio.run(main())
```

## Documentation

### Getting Started
- [Installation](installation.md) - Setting up DuckMem
- [Configuration](configuration.md) - Environment variables and settings
- [User Guide](guide.md) - Complete usage guide

### Interfaces
- [CLI Reference](cli.md) - Command-line interface
- [REST API](rest-api.md) - FastAPI endpoints
- [MCP Server](mcp-server.md) - Model Context Protocol integration

### Reference
- Python API - see inline docstrings in [duckmem/](../duckmem/) (e.g. `help(duckmem.DuckMem)`).
- REST API - live Swagger UI at `/docs` when the server is running, plus [REST API](rest-api.md).
- MCP tools - see [MCP Server](mcp-server.md).

### Examples
- Runnable scripts live in [examples/](../examples/): basic usage, namespaces,
  knowledge graph, RAG Q&A, sessions, and an end-to-end workflow.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Consumer Layer                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │   CLI   │  │ FastAPI │  │   MCP   │  │  Python SDK     │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘ │
│       └────────────┴────────────┴────────────────┘          │
│                           │                                  │
│                    ┌──────┴──────┐                          │
│                    │   DuckMem   │                          │
│                    └──────┬──────┘                          │
│       ┌──────────────────┼──────────────────┐               │
│       │                  │                  │               │
│  ┌────┴────┐      ┌──────┴──────┐    ┌─────┴─────┐         │
│  │ Search  │      │  Knowledge  │    │   RAG     │         │
│  │ Engine  │      │   Graph     │    │  Engine   │         │
│  └────┬────┘      └──────┬──────┘    └─────┬─────┘         │
│       │                  │                  │               │
│       └──────────────────┴──────────────────┘               │
│                           │                                  │
│                    ┌──────┴──────┐                          │
│                    │   DuckDB    │  ← Single .duckdb file   │
│                    │  ┌───────┐  │                          │
│                    │  │  FTS  │  │  BM25 lexical search     │
│                    │  ├───────┤  │                          │
│                    │  │  VSS  │  │  HNSW vector similarity  │
│                    │  └───────┘  │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## License

MIT License
