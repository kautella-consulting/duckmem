# DuckMem Examples

This directory contains runnable examples demonstrating DuckMem functionality.

## Prerequisites

1. Install DuckMem:
   ```bash
   uv sync
   ```

2. For embedding (required for all examples):
   ```bash
   # Option A: Ollama (local, free)
   ollama pull qwen3-embedding:latest
   ollama serve
   
   # Option B: OpenAI
   export OPENAI_API_KEY="sk-..."
   export DUCKMEM_EMBED_MODEL="text-embedding-3-small"
   export DUCKMEM_EMBED_DIM=1536
   ```

3. For RAG examples (04 and 06):
   ```bash
   # Option A: Ollama (default)
   ollama pull gpt-oss:20b
   
   # Option B: OpenAI
   export OPENAI_API_KEY="sk-..."
   export DUCKMEM_LLM_MODEL="openai/gpt-4o-mini"
   ```

## Running Examples

```bash
# From the duckmem directory
uv run python examples/01_basic_usage.py
uv run python examples/02_documents_and_namespaces.py
uv run python examples/03_knowledge_graph.py
uv run python examples/04_rag_qa.py
uv run python examples/05_sessions.py
uv run python examples/06_complete_workflow.py
```

## Examples

### 01_basic_usage.py
**DuckMem fundamentals**
- Creating a knowledge base
- Adding documents
- Basic search (hybrid, lexical, semantic)
- Viewing statistics
- Verifying integrity

### 02_documents_and_namespaces.py
**Document organization**
- Adding documents with metadata
- Using namespaces for organization
- Listing and filtering items
- Namespace-scoped search
- Pagination

### 03_knowledge_graph.py
**Knowledge graph operations**
- Adding relations (Subject-Predicate-Object)
- Getting entity state
- Graph traversal
- Filtered traversal by relation type
- Building a connected knowledge graph

### 04_rag_qa.py
**RAG (Retrieval-Augmented Generation)**
- Building a knowledge base for Q&A
- Asking questions
- Viewing answer sources and confidence
- Namespace-scoped Q&A

*Note: Requires an LLM (Ollama or API key)*

### 05_sessions.py
**Session tracking**
- Starting and ending sessions
- Logging events (searches, views, insights)
- Session replay
- Session management

### 06_complete_workflow.py
**End-to-end workflow**
- Document ingestion with metadata
- Building a knowledge graph
- Hybrid search
- Session tracking
- RAG Q&A
- Maintenance operations

## Generated Files

Each example creates a `.duckdb` file in the current directory:
- `basic_example.duckdb`
- `documents_example.duckdb`
- `knowledge_graph_example.duckdb`
- `rag_example.duckdb`
- `sessions_example.duckdb`
- `complete_workflow.duckdb`

These can be safely deleted after running the examples.

## Troubleshooting

**"Cannot connect to embedding service"**
- Make sure Ollama is running: `ollama serve`
- Or set OpenAI API key: `export OPENAI_API_KEY="..."`

**"LLM not available" in RAG examples**
- Pull an LLM: `ollama pull gpt-oss:20b`
- Or set: `export DUCKMEM_LLM_MODEL="ollama/gpt-oss:20b"` (default) or `openai/gpt-4o-mini` for cloud

**Import errors**
- Make sure you're running from the duckmem directory
- Install with: `uv sync`
