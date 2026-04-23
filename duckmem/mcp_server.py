"""FastMCP server for DuckMem.

Exposes DuckMem functionality as MCP tools for LLM agents.
Run with: uv run duckmem-mcp, fastmcp run duckmem/mcp_server.py, or python -m duckmem.mcp_server
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# When run by file path (e.g. fastmcp run mcp_server.py), project root may not be in path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastmcp import FastMCP  # noqa: E402

from duckmem.config import get_settings  # noqa: E402
from duckmem.core import DuckMem  # noqa: E402

# Initialize FastMCP server
mcp = FastMCP(
    "DuckMem",
    instructions="Personal Knowledge Memory: ingestion, hybrid search, RAG, knowledge graph.",
)

# Global DuckMem instance (initialized on startup)
_duckmem: DuckMem | None = None

# CLI-provided database path override (set by main()); takes precedence over env/.env
_db_path_override: str | None = None


def get_mem() -> DuckMem:
    """Get or create the DuckMem instance."""
    global _duckmem
    if _duckmem is None:
        settings = get_settings()
        _duckmem = DuckMem(db_path=_db_path_override, settings=settings)
    return _duckmem


def _to_json(obj: Any) -> str:
    """Convert object to JSON string for tool responses."""
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), default=str)
    if isinstance(obj, list):
        items = [o.model_dump() if hasattr(o, "model_dump") else o for o in obj]
        return json.dumps(items, default=str)
    return json.dumps(obj, default=str)


# =============================================================================
# Item Tools
# =============================================================================


@mcp.tool()
def memory_add(
    text: str,
    title: str = "",
    uri: str = "",
    namespace: str = "default",
    label: str = "",
) -> str:
    """Add an item to the knowledge base.

    Ingests text by chunking it, computing embeddings, and storing in the database.
    Returns the generated item ID.

    Args:
        text: Text content to ingest (required)
        title: Optional title for the item
        uri: Optional URI/URL reference
        namespace: Categorization namespace (default: "default")
        label: Optional label for filtering
    """
    mem = get_mem()
    item_id = mem.add(
        text,
        title=title if title else None,
        uri=uri if uri else None,
        namespace=namespace,
        label=label if label else None,
    )
    return json.dumps({"item_id": item_id})


@mcp.tool()
def memory_get(item_id: str) -> str:
    """Get an item by its ID.

    Returns the full item including text, metadata, and timestamps.

    Args:
        item_id: The item ID to retrieve
    """
    mem = get_mem()
    item = mem.get(item_id)
    if not item:
        return json.dumps({"error": "Item not found"})
    return _to_json(item)


@mcp.tool()
def memory_list(
    namespace: str = "",
    label: str = "",
    limit: int = 20,
) -> str:
    """List items in the knowledge base.

    Returns items matching the optional filters, sorted by creation time.

    Args:
        namespace: Filter by namespace (optional)
        label: Filter by label (optional)
        limit: Maximum items to return (default: 20)
    """
    mem = get_mem()
    items = mem.list(
        namespace=namespace if namespace else None,
        label=label if label else None,
        limit=limit,
    )
    return _to_json(items)


# =============================================================================
# Search Tools
# =============================================================================


@mcp.tool()
def memory_search(
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
    namespace: str = "",
) -> str:
    """Search the knowledge base using hybrid BM25 + vector search.

    Combines lexical (BM25) and semantic (vector) search using Reciprocal
    Rank Fusion for optimal relevance.

    Args:
        query: Search query string
        mode: Search mode - "hybrid" (default), "lexical", or "semantic"
        top_k: Maximum results to return (default: 10)
        namespace: Filter by namespace (optional)
    """
    mem = get_mem()
    results = mem.search(
        query,
        mode=mode,  # type: ignore
        top_k=top_k,
        namespace=namespace if namespace else None,
    )
    # Return simplified results for LLM consumption
    simplified = [
        {
            "chunk_id": r.chunk.id,
            "item_id": r.chunk.item_id,
            "text": r.chunk.text,
            "title": r.item.title,
            "score": round(r.score, 4),
        }
        for r in results
    ]
    return json.dumps(simplified)


@mcp.tool()
async def memory_ask(
    question: str,
    model: str = "",
    top_k: int = 5,
    namespace: str = "",
) -> str:
    """Ask a question using RAG (retrieval-augmented generation).

    Searches for relevant context, then uses an LLM to generate an answer
    based on the retrieved information.

    Args:
        question: The question to answer
        model: LLM model override (optional, uses default if empty)
        top_k: Number of context chunks to retrieve (default: 5)
        namespace: Filter search by namespace (optional)
    """
    mem = get_mem()
    answer = await mem.ask(
        question,
        model=model if model else None,
        top_k=top_k,
        namespace=namespace if namespace else None,
    )
    return _to_json(answer)


# =============================================================================
# Knowledge Graph Tools
# =============================================================================


@mcp.tool()
def memory_add_relation(
    subject: str,
    predicate: str,
    object: str,
    item_id: str = "",
) -> str:
    """Add a relation to the knowledge graph.

    Creates a subject-predicate-object triple connecting entities.

    Args:
        subject: Subject entity name
        predicate: Relationship type (e.g., "works_at", "located_in")
        object: Object entity name
        item_id: Optional source item ID
    """
    mem = get_mem()
    relation_id = mem.add_relation(
        subject,
        predicate,
        object,
        item_id=item_id if item_id else None,
    )
    return json.dumps({"relation_id": relation_id})


@mcp.tool()
def memory_state(entity: str) -> str:
    """Get current state of an entity (latest-wins properties).

    Returns the most recent value for each predicate associated with the entity.

    Args:
        entity: Entity name to look up
    """
    mem = get_mem()
    state = mem.state(entity)
    return _to_json(state)


@mcp.tool()
def memory_history(entity: str, predicate: str = "") -> str:
    """Get relation history for an entity.

    Returns all relations where the entity is the subject, in chronological order.
    Relations where the entity is the object (e.g., "X created_by Python") are
    not included.

    Args:
        entity: Entity name
        predicate: Filter by predicate (optional)
    """
    mem = get_mem()
    relations = mem.history(entity, predicate if predicate else None)
    return _to_json(relations)


@mcp.tool()
def memory_traverse(
    start: str,
    link: str = "",
    max_hops: int = 3,
) -> str:
    """Traverse the knowledge graph from a starting entity.

    Follows relations to discover connected entities up to max_hops away.

    Args:
        start: Starting entity name
        link: Filter by predicate (optional)
        max_hops: Maximum traversal depth (default: 3)
    """
    mem = get_mem()
    result = mem.traverse(start, link=link if link else None, max_hops=max_hops)
    return _to_json(result)


@mcp.tool()
async def memory_extract(item_id: str, model: str = "") -> str:
    """Extract relations from an item using LLM.

    Analyzes the item's text and extracts subject-predicate-object relations.

    Args:
        item_id: Item to extract relations from
        model: LLM model override (optional)
    """
    mem = get_mem()
    try:
        relation_ids = await mem.extract(item_id, model=model if model else None)
        return json.dumps({"relation_ids": relation_ids, "count": len(relation_ids)})
    except ValueError as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Session Tools
# =============================================================================


@mcp.tool()
def memory_session_start(name: str = "") -> str:
    """Start a recording session.

    Sessions track all operations for later replay.

    Args:
        name: Optional session name
    """
    mem = get_mem()
    session_id = mem.session_start(name if name else None)
    return json.dumps({"session_id": session_id})


@mcp.tool()
def memory_session_end() -> str:
    """End the current recording session."""
    mem = get_mem()
    mem.session_end()
    return json.dumps({"status": "ended"})


@mcp.tool()
def memory_session_list() -> str:
    """List all recording sessions."""
    mem = get_mem()
    sessions = mem.session_list()
    return _to_json(sessions)


@mcp.tool()
def memory_session_replay(session_id: str) -> str:
    """Replay events from a session.

    Args:
        session_id: Session to replay
    """
    mem = get_mem()
    events = mem.session_replay(session_id)
    return _to_json(events)


# =============================================================================
# Maintenance Tools
# =============================================================================


@mcp.tool()
def memory_stats() -> str:
    """Get database statistics.

    Returns counts of items, chunks, relations, entities, and file size.
    """
    mem = get_mem()
    stats = mem.stats()
    return _to_json(stats)


@mcp.tool()
def memory_verify(deep: bool = False) -> str:
    """Verify database integrity.

    Args:
        deep: If true, also verify checksums (slower)
    """
    mem = get_mem()
    result = mem.verify(deep=deep)
    return _to_json(result)


@mcp.tool()
def memory_doctor(
    vacuum: bool = False,
    rebuild_fts: bool = False,
    rebuild_vec: bool = False,
    timeout_seconds: float | None = None,
) -> str:
    """Run maintenance operations.

    For large databases, call with one operation at a time (vacuum, rebuild_fts,
    or rebuild_vec) to avoid MCP client timeouts. Combining all three can take
    60+ seconds.

    Args:
        vacuum: Compact storage with CHECKPOINT
        rebuild_fts: Rebuild the FTS (full-text search) index
        rebuild_vec: Rebuild the vector (HNSW) index
        timeout_seconds: Max seconds for all ops; None = no limit. When exceeded,
            returns partial results with timeout_hit=True.
    """
    mem = get_mem()
    results = mem.doctor(
        vacuum=vacuum,
        rebuild_fts=rebuild_fts,
        rebuild_vec=rebuild_vec,
        timeout_seconds=timeout_seconds,
    )
    return json.dumps(results)


# =============================================================================
# Resources
# =============================================================================


@mcp.resource("duckmem://stats")
def resource_stats() -> str:
    """Current database statistics."""
    mem = get_mem()
    stats = mem.stats()
    return _to_json(stats)


@mcp.resource("duckmem://items")
def resource_items() -> str:
    """List of recent items."""
    mem = get_mem()
    items = mem.list(limit=50)
    return _to_json(items)


# =============================================================================
# Main Entry Point
# =============================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the MCP server.

    Supports an optional database path as either a positional argument or via
    ``--db-path``. When provided, this takes precedence over ``DUCKMEM_DB_PATH``
    and any value in a ``.env`` file.
    """
    parser = argparse.ArgumentParser(
        prog="duckmem-mcp",
        description="DuckMem MCP server.",
    )
    parser.add_argument(
        "db_path",
        nargs="?",
        default=None,
        help="Path to the DuckDB database file (overrides DUCKMEM_DB_PATH).",
    )
    parser.add_argument(
        "--db-path",
        dest="db_path_opt",
        default=None,
        help="Path to the DuckDB database file (overrides DUCKMEM_DB_PATH).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the MCP server."""
    global _db_path_override
    args = _parse_args(argv)
    db_path = args.db_path_opt or args.db_path
    if db_path:
        _db_path_override = str(Path(db_path).expanduser())
    mcp.run()


if __name__ == "__main__":
    main()
