# MCP Server Reference

DuckMem includes a Model Context Protocol (MCP) server that allows LLMs to directly interact with your knowledge base as a tool.

## Overview

The MCP server exposes DuckMem functionality as tools that AI assistants can call. This enables:

- Adding documents during conversation
- Searching your knowledge base for context
- Asking questions with RAG
- Managing the knowledge graph
- Tracking sessions

## Installing the MCP Server

### Claude Desktop

**Important:** Do not use `mcp install` for DuckMem. It generates a config that runs the server in an isolated environment without DuckMem's dependencies (duckdb, etc.), causing `ModuleNotFoundError`.

Add the server manually to `~/.claude/config.json` (see Integration section below). The config must use `duckmem-mcp` with `cwd` set to the project root so the server runs with all dependencies.

### Cursor

**Recommended:** Use the FastMCP CLI for first-class Cursor integration:

```bash
fastmcp install cursor duckmem/mcp_server.py --project /path/to/duckmem
```

Or add the server manually to `~/.cursor/mcp.json` (see Integration section below).

## Starting the MCP Server

```bash
# Recommended: use the project entry point (includes all dependencies)
uv run duckmem-mcp

# Or with custom database
DUCKMEM_DB_PATH=knowledge.duckdb uv run duckmem-mcp

# Using FastMCP CLI (from project root)
fastmcp run duckmem/mcp_server.py --project .

# Development with auto-reload
fastmcp run duckmem/mcp_server.py --project . --reload

# Using Python directly
uv run python -m duckmem.mcp_server
```

## Integration with AI Assistants

### Claude Desktop

Add to your Claude configuration (`~/.claude/config.json`):

```json
{
  "mcpServers": {
    "duckmem": {
      "command": "uv",
      "args": ["run", "duckmem-mcp"],
      "cwd": "/path/to/duckmem",
      "env": {
        "DUCKMEM_DB_PATH": "/path/to/knowledge.duckdb"
      }
    }
  }
}
```

### Cursor

Add to your Cursor MCP settings (`~/.cursor/mcp.json`), or use `fastmcp install cursor` (see above):

```json
{
  "mcpServers": {
    "duckmem": {
      "command": "uv",
      "args": ["run", "duckmem-mcp"],
      "cwd": "/path/to/duckmem"
    }
  }
}
```

Or use FastMCP CLI with project context:

```json
{
  "mcpServers": {
    "duckmem": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/duckmem", "fastmcp", "run", "duckmem/mcp_server.py"],
      "cwd": "/path/to/duckmem"
    }
  }
}
```

## Available Tools

All tool names use the `memory_` prefix. See
[duckmem/mcp_server.py](../duckmem/mcp_server.py) for authoritative signatures;
the MCP client can also discover them via the protocol.

### Item management

| Tool | Parameters | Purpose |
|---|---|---|
| `memory_add` | `text` (str, required), `title` (str, ""), `uri` (str, ""), `namespace` (str, "default"), `label` (str, "") | Ingest text. Returns `{"item_id": "..."}`. |
| `memory_get` | `item_id` (str, required) | Get a single item by id. |
| `memory_list` | `namespace` (str, ""), `label` (str, ""), `limit` (int, 20) | List items with optional filters. |

### Search

| Tool | Parameters | Purpose |
|---|---|---|
| `memory_search` | `query` (str, required), `mode` ("hybrid" \| "lexical" \| "semantic", "hybrid"), `top_k` (int, 10), `namespace` (str, "") | Returns a compact list of `{chunk_id, item_id, text, title, score}`. |
| `memory_ask` | `question` (str, required), `model` (str, ""), `top_k` (int, 5), `namespace` (str, "") | RAG answer. Returns the full `Answer` JSON (`answer`, `confidence`, `sources`, `context`). |

### Knowledge graph

| Tool | Parameters | Purpose |
|---|---|---|
| `memory_add_relation` | `subject`, `predicate`, `object` (all required), `item_id` (str, "") | Add a Subject-Predicate-Object triple. Returns `{"relation_id": "..."}`. |
| `memory_state` | `entity` (str, required) | Current `{entity, properties}` for an entity. |
| `memory_history` | `entity` (str, required), `predicate` (str, "") | Relation history as `list[Relation]`. Entity must be the subject of the relation. |
| `memory_traverse` | `start` (str, required), `link` (str, ""), `max_hops` (int, 3) | Graph traversal. Returns `TraversalResult` (`entities`, `paths`). |
| `memory_extract` | `item_id` (str, required), `model` (str, "") | LLM-driven relation extraction. Returns `{"relation_ids": [...], "count": N}`. |

### Sessions

| Tool | Parameters | Purpose |
|---|---|---|
| `memory_session_start` | `name` (str, "") | Start a recording session. Returns `{"session_id": "..."}`. |
| `memory_session_end` | none | End the currently-active session. |
| `memory_session_list` | none | List all sessions. |
| `memory_session_replay` | `session_id` (str, required) | Return events for a session. |

There is no `session_log` tool - while a session is active, `memory_add`,
`memory_search`, `memory_ask`, `memory_add_relation`, and `memory_extract` all
auto-log events with their parameters.

### Maintenance

| Tool | Parameters | Purpose |
|---|---|---|
| `memory_stats` | none | `Stats` JSON (`items`, `chunks`, `relations`, `entities`, `sessions`, `file_size_bytes`). |
| `memory_verify` | `deep` (bool, false) | Integrity check. Returns a `VerifyResult`. |
| `memory_doctor` | `vacuum` (bool, false), `rebuild_fts` (bool, false), `rebuild_vec` (bool, false), `timeout_seconds` (float?, null) | Maintenance operations. Returns `dict[str, bool]` of per-op success. |

For large databases, call `memory_doctor` with one flag at a time - combining
vacuum + rebuild_fts + rebuild_vec can exceed typical MCP client timeouts.

### Resources

Two read-only resources are also exposed:

| URI | Purpose |
|---|---|
| `duckmem://stats` | Current database statistics. |
| `duckmem://items` | List of up to 50 recent items. |

---

## Session Events

Operations automatically logged during a session:

- `add` - when adding items
- `search` - when searching
- `ask` - when using RAG Q&A
- `add_relation` - when adding knowledge graph relations
- `extract` - when extracting relations from an item via LLM

Session replay returns these events in chronological order. Replay is an audit
log (read-only); it does not re-execute operations.

---

## Usage Examples

### Research Assistant

```
User: I'm researching transformers. First, search my notes for anything about attention.

[LLM calls memory_search(query="attention mechanisms")]

User: Add this paper summary to my research: "Attention Is All You Need
introduced the transformer architecture, eliminating recurrence in favor of
self-attention..."

[LLM calls memory_add(text=..., title="Attention Is All You Need")]

User: What do my notes say about the advantages of transformers?

[LLM calls memory_ask(question="What are the advantages of transformers?")]
```

### Knowledge Graph Building

```
User: From my document about Python, extract any relationships between concepts.

[LLM calls memory_extract(item_id="item_...")]

User: Show me everything connected to "machine learning" in my knowledge graph.

[LLM calls memory_traverse(start="machine learning", max_hops=2)]
```

### Session Tracking

```
User: Start a new research session called "Transformer Study".

[LLM calls memory_session_start(name="Transformer Study")]

User: [performs various searches and additions - auto-logged]

User: End my session and show me what I did.

[LLM calls memory_session_end, then memory_session_replay(session_id=...)]
```

## Environment Variables

The MCP server respects all `DUCKMEM_*` environment variables:

```bash
export DUCKMEM_DB_PATH="knowledge.duckdb"
export DUCKMEM_EMBED_MODEL="ollama/qwen3-embedding:latest"
export DUCKMEM_LLM_MODEL="ollama/gpt-oss:20b"  # default; use "openai/gpt-4o-mini" for cloud
```

## Troubleshooting

### ModuleNotFoundError: No module named 'duckmem' or 'duckdb'

Your config is using `mcp run` or `fastmcp run` with a file path in an isolated environment without DuckMem's dependencies.

**Fix:** Use `duckmem-mcp` with `cwd` set to the project root, or use `fastmcp run` with `--project`. In `~/.claude/config.json`:

```json
{
  "mcpServers": {
    "duckmem": {
      "command": "uv",
      "args": ["run", "duckmem-mcp"],
      "cwd": "/absolute/path/to/duckmem",
      "env": {
        "DUCKMEM_DB_PATH": "/path/to/knowledge.duckdb"
      }
    }
  }
}
```

Ensure `cwd` is the absolute path to the duckmem project root (where `pyproject.toml` lives).

### Server Not Starting

1. Check that the database path is valid
2. Ensure Ollama is running (if using local models)
3. Verify environment variables are set

### Tools Not Available

1. Restart the MCP server
2. Check AI assistant's MCP configuration
3. Verify the path to duckmem in config

### Slow Responses

1. First query may be slow (loading models)
2. Consider using smaller embedding models
3. Check if Ollama has enough resources
