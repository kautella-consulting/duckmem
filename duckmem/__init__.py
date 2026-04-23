"""DuckMem - A DuckDB-based personal knowledge memory system.

DuckMem provides document ingestion, hybrid BM25+vector search, knowledge graph,
RAG Q&A, and is exposed via FastAPI REST and FastMCP server.

Example:
    >>> from duckmem import DuckMem
    >>> mem = DuckMem("knowledge.duckdb")
    >>> item_id = mem.add("Transformers use attention mechanisms.", title="ML Notes")
    >>> results = mem.search("attention", top_k=5)
    >>> answer = await mem.ask("What do transformers use?")
"""

from duckmem.config import Settings
from duckmem.core import DuckMem
from duckmem.models import (
    Answer,
    Chunk,
    Entity,
    Item,
    Relation,
    SearchResult,
    Session,
    SessionEvent,
)

__version__ = "0.1.0"
__all__ = [
    "DuckMem",
    "Settings",
    "Item",
    "Chunk",
    "Relation",
    "Entity",
    "SearchResult",
    "Answer",
    "Session",
    "SessionEvent",
]
