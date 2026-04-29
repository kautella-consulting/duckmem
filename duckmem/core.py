"""Core DuckMem class and functions.

Provides the main DuckMem class which wraps all functionality, plus
standalone functions for functional programming style usage.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import duckdb
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from duckmem.config import Settings, get_settings
from duckmem.inference import embed, embed_single
from duckmem.ingestion.chunkers import ChunkConfig, TextChunk, chunk_text
from duckmem.models import (
    Answer,
    Chunk,
    EntityState,
    Item,
    Relation,
    SearchResult,
    Session,
    SessionEvent,
    Stats,
    TraversalResult,
    VerifyResult,
)
from duckmem.schema import (
    check_extensions,
    init_fts_index,
    init_hnsw_index,
    init_property_graph,
    init_schema,
)
from duckmem.utils import compute_checksum, generate_uid, timestamp_ms

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


SearchMode = Literal["hybrid", "lexical", "semantic"]

LOCK_MAGIC = b"DUCKMEMLOCK\x01"
LOCK_SALT_BYTES = 16
LOCK_KDF_ITERATIONS = 600_000


# =============================================================================
# Item Ingestion Functions
# =============================================================================


def add_item(
    conn: duckdb.DuckDBPyConnection,
    text: str,
    *,
    title: str | None = None,
    uri: str | None = None,
    namespace: str = "default",
    label: str | None = None,
    metadata: dict | None = None,
    embed_model: str = "text-embedding-3-small",
    chunk_config: ChunkConfig | None = None,
    skip_embedding: bool = False,
) -> str:
    """Add an item to the knowledge base.

    Ingests text by chunking it, computing embeddings, and storing
    in the database. Rebuilds the FTS index after insertion.

    Args:
        conn: DuckDB connection.
        text: Text content to ingest.
        title: Optional title for the item.
        uri: Optional URI reference.
        namespace: Categorization namespace.
        label: Optional label for filtering.
        metadata: Optional metadata dictionary.
        embed_model: LiteLLM model string for embeddings.
        chunk_config: Chunking configuration.
        skip_embedding: If True, skip embedding computation.

    Returns:
        The generated item ID.

    Raises:
        ValueError: If text is empty.

    Example:
        >>> conn = duckdb.connect(":memory:")
        >>> init_schema(conn)
        >>> item_id = add_item(conn, "Transformers use attention.", title="ML")
    """
    if not text.strip():
        raise ValueError("text cannot be empty")

    item_id = generate_uid()
    checksum = compute_checksum(text)
    created_at = timestamp_ms()

    # Insert item
    conn.execute(
        """
        INSERT INTO items (id, title, uri, namespace, label, text, metadata, created_at, checksum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            item_id,
            title,
            uri,
            namespace,
            label,
            text,
            json.dumps(metadata) if metadata else None,
            created_at,
            checksum,
        ],
    )

    # Chunk the text
    if chunk_config is None:
        chunk_config = ChunkConfig()

    text_chunks = chunk_text(text, chunk_config)

    # Fallback: if min_chars filter removed all chunks (e.g. short text),
    # create one chunk with full text so the item is searchable
    if not text_chunks:
        text_chunks = [TextChunk(text=text.strip(), start=0, end=len(text))]

    # Compute embeddings
    if skip_embedding:
        embeddings: list[list[float]] | None = None
    else:
        chunk_texts = [c.text for c in text_chunks]
        embeddings = embed(chunk_texts, model=embed_model) if chunk_texts else None

    # Insert chunks
    for seq, text_chunk in enumerate(text_chunks):
        chunk_id = generate_uid()
        embedding = embeddings[seq] if embeddings else None

        conn.execute(
            """
            INSERT INTO chunks (id, item_id, seq, text, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            [chunk_id, item_id, seq, text_chunk.text, embedding],
        )

    # Rebuild FTS index
    with contextlib.suppress(duckdb.Error):
        init_fts_index(conn)

    return item_id


def get_item(conn: duckdb.DuckDBPyConnection, item_id: str) -> Item | None:
    """Get an item by ID.

    Args:
        conn: DuckDB connection.
        item_id: Item ID to retrieve.

    Returns:
        Item object or None if not found.
    """
    result = conn.execute("SELECT * FROM items WHERE id = ?", [item_id]).fetchone()

    if not result:
        return None

    return Item(
        id=result[0],
        title=result[1],
        uri=result[2],
        namespace=result[3],
        label=result[4],
        text=result[5],
        metadata=json.loads(result[6]) if result[6] else None,
        created_at=result[7],
        checksum=result[8],
    )


def list_items(
    conn: duckdb.DuckDBPyConnection,
    *,
    namespace: str | None = None,
    label: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Item]:
    """List items with optional filtering.

    Args:
        conn: DuckDB connection.
        namespace: Filter by namespace.
        label: Filter by label.
        limit: Maximum items to return.
        offset: Number of items to skip.

    Returns:
        List of Item objects.
    """
    query = "SELECT * FROM items WHERE 1=1"
    params: list = []

    if namespace:
        query += " AND namespace = ?"
        params.append(namespace)

    if label:
        query += " AND label = ?"
        params.append(label)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    results = conn.execute(query, params).fetchall()

    return [
        Item(
            id=r[0],
            title=r[1],
            uri=r[2],
            namespace=r[3],
            label=r[4],
            text=r[5],
            metadata=json.loads(r[6]) if r[6] else None,
            created_at=r[7],
            checksum=r[8],
        )
        for r in results
    ]


# =============================================================================
# Search Functions
# =============================================================================


def search(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    *,
    mode: SearchMode = "hybrid",
    top_k: int = 10,
    namespace: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    embed_model: str = "text-embedding-3-small",
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Search for items matching the query.

    Combines lexical (BM25) and semantic (vector) search using
    Reciprocal Rank Fusion for optimal relevance.

    Args:
        conn: DuckDB connection.
        query: Search query string.
        mode: Search mode - "hybrid", "lexical", or "semantic".
        top_k: Maximum results to return.
        namespace: Filter by namespace.
        start_ts: Filter by minimum timestamp.
        end_ts: Filter by maximum timestamp.
        embed_model: Model for query embedding (semantic/hybrid).
        rrf_k: RRF constant (default 60).

    Returns:
        List of SearchResult objects sorted by relevance.

    Example:
        >>> results = search(conn, "attention mechanism", top_k=5)
        >>> for r in results:
        ...     print(f"{r.score:.2f}: {r.chunk.text[:50]}...")
    """
    lexical_results: dict[str, tuple[int, float]] = {}  # chunk_id -> (rank, score)
    semantic_results: dict[str, tuple[int, float]] = {}

    fetch_k = top_k * 3  # Over-fetch for fusion

    # Lexical search (BM25)
    if mode in ("hybrid", "lexical"):
        lexical_results = _search_lexical(conn, query, fetch_k)

    # Semantic search (vector)
    if mode in ("hybrid", "semantic"):
        query_embedding = embed_single(query, model=embed_model)
        semantic_results = _search_semantic(conn, query_embedding, fetch_k)

    # Combine results using RRF
    combined = _reciprocal_rank_fusion(lexical_results, semantic_results, k=rrf_k)

    # Get top_k chunk IDs
    top_chunk_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_k]

    # Fetch full results
    results: list[SearchResult] = []
    for chunk_id in top_chunk_ids:
        chunk_row = conn.execute("SELECT * FROM chunks WHERE id = ?", [chunk_id]).fetchone()

        if not chunk_row:
            continue

        item = get_item(conn, chunk_row[1])
        if not item:
            continue

        # Apply filters
        if namespace and item.namespace != namespace:
            continue
        if start_ts and item.created_at < start_ts:
            continue
        if end_ts and item.created_at > end_ts:
            continue

        chunk = Chunk(
            id=chunk_row[0],
            item_id=chunk_row[1],
            seq=chunk_row[2],
            text=chunk_row[3],
            embedding=tuple(chunk_row[4]) if chunk_row[4] else None,
        )

        lex_score = lexical_results.get(chunk_id, (999, 0.0))[1]
        sem_score = semantic_results.get(chunk_id, (999, 0.0))[1]

        results.append(
            SearchResult(
                chunk=chunk,
                item=item,
                score=combined[chunk_id],
                lexical_score=lex_score if lex_score > 0 else None,
                semantic_score=sem_score if sem_score > 0 else None,
            )
        )

    return results


def _search_lexical(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    limit: int,
) -> dict[str, tuple[int, float]]:
    """Perform BM25 lexical search.

    Returns:
        Dict mapping chunk_id to (rank, score).
    """
    try:
        results = conn.execute(
            """
            SELECT id, fts_main_chunks.match_bm25(id, ?) AS score
            FROM chunks
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
            """,
            [query, limit],
        ).fetchall()

        return {r[0]: (i, r[1]) for i, r in enumerate(results)}
    except duckdb.Error as e:
        logger.warning("Lexical search failed: %s", e)
        return {}


def _search_semantic(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int,
) -> dict[str, tuple[int, float]]:
    """Perform vector similarity search.

    Returns:
        Dict mapping chunk_id to (rank, similarity_score).
    """
    embed_dim = len(query_embedding)
    try:
        results = conn.execute(
            f"""
            SELECT id, array_cosine_similarity(embedding, ?::FLOAT[{embed_dim}]) AS similarity
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT ?
            """,
            [query_embedding, limit],
        ).fetchall()

        return {r[0]: (i, r[1]) for i, r in enumerate(results)}
    except duckdb.Error as e:
        logger.warning("Semantic search failed (check embed_dim matches model output): %s", e)
        return {}


def _reciprocal_rank_fusion(
    lex_results: dict[str, tuple[int, float]],
    sem_results: dict[str, tuple[int, float]],
    k: int = 60,
) -> dict[str, float]:
    """Combine rankings using Reciprocal Rank Fusion.

    RRF score = 1/(k + rank_lex) + 1/(k + rank_sem)

    Args:
        lex_results: Lexical search results (chunk_id -> (rank, score)).
        sem_results: Semantic search results (chunk_id -> (rank, score)).
        k: RRF constant (default 60).

    Returns:
        Dict mapping chunk_id to combined RRF score.
    """
    all_ids = set(lex_results.keys()) | set(sem_results.keys())
    combined: dict[str, float] = {}

    for chunk_id in all_ids:
        lex_rank = lex_results.get(chunk_id, (999, 0.0))[0]
        sem_rank = sem_results.get(chunk_id, (999, 0.0))[0]

        rrf_score = 1.0 / (k + lex_rank) + 1.0 / (k + sem_rank)
        combined[chunk_id] = rrf_score

    return combined


# =============================================================================
# Knowledge Graph Functions
# =============================================================================


def add_relation(
    conn: duckdb.DuckDBPyConnection,
    subject: str,
    predicate: str,
    obj: str,
    *,
    item_id: str | None = None,
) -> str:
    """Add a relation to the knowledge graph.

    Also ensures subject and object entities exist.

    Args:
        conn: DuckDB connection.
        subject: Subject entity name.
        predicate: Relationship type.
        obj: Object entity name.
        item_id: Optional source item ID.

    Returns:
        The generated relation ID.
    """
    relation_id = generate_uid()
    created_at = timestamp_ms()

    # Ensure entities exist
    _ensure_entity(conn, subject, created_at)
    _ensure_entity(conn, obj, created_at)

    # Insert relation
    conn.execute(
        """
        INSERT INTO relations (id, subject, predicate, object, item_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [relation_id, subject, predicate, obj, item_id, created_at],
    )

    return relation_id


def _ensure_entity(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    timestamp: int,
    kind: str = "unknown",
) -> None:
    """Ensure an entity exists in the database."""
    conn.execute(
        """
        INSERT INTO entities (name, kind, first_seen)
        VALUES (?, ?, ?)
        ON CONFLICT (name) DO NOTHING
        """,
        [name, kind, timestamp],
    )


def get_entity_state(
    conn: duckdb.DuckDBPyConnection,
    entity: str,
) -> EntityState:
    """Get current state of an entity (latest-wins properties).

    Args:
        conn: DuckDB connection.
        entity: Entity name.

    Returns:
        EntityState with current property values.
    """
    results = conn.execute(
        """
        SELECT predicate, object
        FROM relations
        WHERE subject = ?
        ORDER BY created_at DESC
        """,
        [entity],
    ).fetchall()

    # Latest-wins deduplication by predicate
    properties: dict[str, str] = {}
    for predicate, obj in results:
        if predicate not in properties:
            properties[predicate] = obj

    return EntityState(entity=entity, properties=properties)


def get_entity_history(
    conn: duckdb.DuckDBPyConnection,
    entity: str,
    predicate: str | None = None,
) -> list[Relation]:
    """Get full history of relations for an entity.

    Args:
        conn: DuckDB connection.
        entity: Entity name.
        predicate: Optional predicate filter.

    Returns:
        List of Relation objects in chronological order.
    """
    query = """
        SELECT id, subject, predicate, object, item_id, created_at
        FROM relations
        WHERE subject = ?
    """
    params: list = [entity]

    if predicate:
        query += " AND predicate = ?"
        params.append(predicate)

    query += " ORDER BY created_at ASC"

    results = conn.execute(query, params).fetchall()

    return [
        Relation(
            id=r[0],
            subject=r[1],
            predicate=r[2],
            object=r[3],
            item_id=r[4],
            created_at=r[5],
        )
        for r in results
    ]


def traverse_graph(
    conn: duckdb.DuckDBPyConnection,
    start: str,
    *,
    link: str | None = None,
    max_hops: int = 3,
) -> TraversalResult:
    """Traverse the knowledge graph from a starting entity.

    Uses recursive CTE for graph traversal (DuckPGQ fallback available).

    Args:
        conn: DuckDB connection.
        start: Starting entity name.
        link: Optional predicate filter.
        max_hops: Maximum traversal depth.

    Returns:
        TraversalResult with paths and entities found.
    """
    # Build recursive CTE query
    predicate_filter = "AND predicate = ?" if link else ""
    params: list = [start]
    if link:
        params.append(link)
    params.append(max_hops)

    query = f"""
    WITH RECURSIVE paths AS (
        SELECT
            subject AS src,
            object AS dst,
            predicate,
            1 AS depth,
            id AS relation_id,
            item_id,
            created_at
        FROM relations
        WHERE subject = ? {predicate_filter}

        UNION ALL

        SELECT
            p.src,
            r.object,
            r.predicate,
            p.depth + 1,
            r.id,
            r.item_id,
            r.created_at
        FROM paths p
        JOIN relations r ON p.dst = r.subject
        WHERE p.depth < ? {predicate_filter.replace("predicate", "r.predicate")}
    )
    SELECT DISTINCT src, dst, predicate, depth, relation_id, item_id, created_at
    FROM paths
    ORDER BY depth, created_at
    """

    if link:
        params.append(link)  # For the UNION part

    results = conn.execute(query, params).fetchall()

    # Build paths and collect entities
    relations: list[Relation] = []
    entities: set[str] = {start}

    for r in results:
        entities.add(r[0])
        entities.add(r[1])
        relations.append(
            Relation(
                id=r[4],
                subject=r[0],
                predicate=r[2],
                object=r[1],
                item_id=r[5],
                created_at=r[6],
            )
        )

    # Group relations into paths (simplified - returns flat list)
    return TraversalResult(
        paths=[relations] if relations else [],
        entities=sorted(entities),
    )


async def extract_relations_from_item(
    conn: duckdb.DuckDBPyConnection,
    item_id: str,
    *,
    model: str = "ollama/gpt-oss:20b",
) -> list[str]:
    """Extract relations from an item using LLM.

    Args:
        conn: DuckDB connection.
        item_id: Item to extract from.
        model: LLM model for extraction.

    Returns:
        List of created relation IDs.
    """
    from duckmem.agents import extract_relations

    item = get_item(conn, item_id)
    if not item:
        raise ValueError(f"Item {item_id} not found")

    extracted = await extract_relations(item.text, model=model)

    relation_ids: list[str] = []
    for rel in extracted:
        rel_id = add_relation(
            conn,
            rel.subject,
            rel.predicate,
            rel.object,
            item_id=item_id,
        )
        relation_ids.append(rel_id)

    return relation_ids


# =============================================================================
# Session Functions
# =============================================================================


def session_start(
    conn: duckdb.DuckDBPyConnection,
    name: str | None = None,
) -> str:
    """Start a new recording session.

    Args:
        conn: DuckDB connection.
        name: Optional session name.

    Returns:
        Session ID.
    """
    session_id = generate_uid()
    started_at = timestamp_ms()

    conn.execute(
        """
        INSERT INTO sessions (id, name, started_at)
        VALUES (?, ?, ?)
        """,
        [session_id, name, started_at],
    )

    return session_id


def session_end(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
) -> None:
    """End a recording session.

    Args:
        conn: DuckDB connection.
        session_id: Session to end.
    """
    ended_at = timestamp_ms()
    conn.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ?",
        [ended_at, session_id],
    )


def session_log_event(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    kind: str,
    params: dict,
    result_summary: dict,
) -> str:
    """Log an event to a session.

    Args:
        conn: DuckDB connection.
        session_id: Session to log to.
        kind: Event type.
        params: Operation parameters.
        result_summary: Summary of results.

    Returns:
        Event ID.
    """
    event_id = generate_uid()
    timestamp = timestamp_ms()

    conn.execute(
        """
        INSERT INTO session_events (id, session_id, timestamp, kind, params, result_summary)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            event_id,
            session_id,
            timestamp,
            kind,
            json.dumps(params),
            json.dumps(result_summary),
        ],
    )

    return event_id


def session_replay(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
) -> list[SessionEvent]:
    """Replay events from a session.

    Args:
        conn: DuckDB connection.
        session_id: Session to replay.

    Returns:
        List of SessionEvent objects in chronological order.
    """
    results = conn.execute(
        """
        SELECT id, session_id, timestamp, kind, params, result_summary
        FROM session_events
        WHERE session_id = ?
        ORDER BY timestamp ASC
        """,
        [session_id],
    ).fetchall()

    return [
        SessionEvent(
            id=r[0],
            session_id=r[1],
            timestamp=r[2],
            kind=r[3],
            params=json.loads(r[4]) if r[4] else {},
            result_summary=json.loads(r[5]) if r[5] else {},
        )
        for r in results
    ]


def session_list(conn: duckdb.DuckDBPyConnection) -> list[Session]:
    """List all sessions.

    Args:
        conn: DuckDB connection.

    Returns:
        List of Session objects.
    """
    results = conn.execute(
        "SELECT id, name, started_at, ended_at FROM sessions ORDER BY started_at DESC"
    ).fetchall()

    return [
        Session(
            id=r[0],
            name=r[1],
            started_at=r[2],
            ended_at=r[3],
        )
        for r in results
    ]


# =============================================================================
# Maintenance Functions
# =============================================================================


def verify(
    conn: duckdb.DuckDBPyConnection,
    *,
    deep: bool = False,
) -> VerifyResult:
    """Verify database integrity.

    Args:
        conn: DuckDB connection.
        deep: If True, verify checksums.

    Returns:
        VerifyResult with counts and any errors.
    """
    items_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    relations_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    entities_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    errors: list[str] = []
    checksum_ok: bool | None = None

    if deep:
        checksum_ok = True
        items = conn.execute("SELECT id, text, checksum FROM items").fetchall()
        for item_id, text, stored_checksum in items:
            computed = compute_checksum(text)
            if computed != stored_checksum:
                errors.append(f"Checksum mismatch for item {item_id}")
                checksum_ok = False

    return VerifyResult(
        items=items_count,
        chunks=chunks_count,
        relations=relations_count,
        entities=entities_count,
        checksum_ok=checksum_ok,
        errors=errors,
    )


def doctor(
    conn: duckdb.DuckDBPyConnection,
    *,
    vacuum: bool = False,
    rebuild_fts: bool = False,
    rebuild_vec: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, bool]:
    """Perform maintenance operations.

    Args:
        conn: DuckDB connection.
        vacuum: Compact storage with CHECKPOINT.
        rebuild_fts: Rebuild FTS index.
        rebuild_vec: Rebuild HNSW index.
        timeout_seconds: Max seconds for all ops; None = no limit. When exceeded,
            returns partial results with timeout_hit=True.

    Returns:
        Dict of operation -> success status. May include timeout_hit=True.
    """
    results: dict[str, bool] = {}
    start = time.time()

    def _elapsed() -> float:
        return time.time() - start

    def _timeout_exceeded() -> bool:
        return (
            timeout_seconds is not None
            and _elapsed() >= timeout_seconds
        )

    if vacuum:
        if _timeout_exceeded():
            results["timeout_hit"] = True
            return results
        try:
            conn.execute("CHECKPOINT")
            results["vacuum"] = True
        except duckdb.Error:
            results["vacuum"] = False

    if rebuild_fts:
        if _timeout_exceeded():
            results["timeout_hit"] = True
            return results
        try:
            init_fts_index(conn)
            results["rebuild_fts"] = True
        except duckdb.Error:
            results["rebuild_fts"] = False

    if rebuild_vec:
        if _timeout_exceeded():
            results["timeout_hit"] = True
            return results
        try:
            conn.execute("DROP INDEX IF EXISTS idx_chunk_embedding")
            init_hnsw_index(conn)
            results["rebuild_vec"] = True
        except duckdb.Error:
            results["rebuild_vec"] = False

    return results


def stats(conn: duckdb.DuckDBPyConnection, db_path: str | None = None) -> Stats:
    """Get database statistics.

    Args:
        conn: DuckDB connection.
        db_path: Path to database file for size calculation.

    Returns:
        Stats object with counts and file size.
    """
    items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    file_size = 0
    if db_path and os.path.exists(db_path):
        file_size = os.path.getsize(db_path)

    return Stats(
        items=items,
        chunks=chunks,
        relations=relations,
        entities=entities,
        sessions=sessions,
        file_size_bytes=file_size,
    )


# =============================================================================
# Encryption Functions
# =============================================================================


def _derive_key(password: str, salt: bytes, iterations: int = LOCK_KDF_ITERATIONS) -> bytes:
    """Derive a Fernet key from a password using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def _derive_legacy_key(password: str) -> bytes:
    """Derive keys for databases encrypted before salted PBKDF2 was added."""
    hash_bytes = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(hash_bytes)


def lock(src_path: str, dst_path: str, password: str) -> None:
    """Encrypt a DuckDB file.

    Args:
        src_path: Path to source .duckdb file.
        dst_path: Path for encrypted output.
        password: Encryption password.

    Raises:
        FileNotFoundError: If source file doesn't exist.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    salt = os.urandom(LOCK_SALT_BYTES)
    key = _derive_key(password, salt)
    fernet = Fernet(key)

    with open(src_path, "rb") as f:
        data = f.read()

    encrypted = fernet.encrypt(data)
    payload = (
        LOCK_MAGIC
        + LOCK_KDF_ITERATIONS.to_bytes(4, "big")
        + salt
        + encrypted
    )

    with open(dst_path, "wb") as f:
        f.write(payload)


def unlock(src_path: str, dst_path: str, password: str) -> None:
    """Decrypt an encrypted DuckDB file.

    Args:
        src_path: Path to encrypted file.
        dst_path: Path for decrypted output.
        password: Decryption password.

    Raises:
        FileNotFoundError: If source file doesn't exist.
        cryptography.fernet.InvalidToken: If password is wrong.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    with open(src_path, "rb") as f:
        payload = f.read()

    if payload.startswith(LOCK_MAGIC):
        offset = len(LOCK_MAGIC)
        iterations = int.from_bytes(payload[offset : offset + 4], "big")
        offset += 4
        salt = payload[offset : offset + LOCK_SALT_BYTES]
        encrypted = payload[offset + LOCK_SALT_BYTES :]
        if iterations <= 0 or len(salt) != LOCK_SALT_BYTES or not encrypted:
            raise InvalidToken
        key = _derive_key(password, salt, iterations)
    else:
        encrypted = payload
        key = _derive_legacy_key(password)

    fernet = Fernet(key)

    decrypted = fernet.decrypt(encrypted)

    with open(dst_path, "wb") as f:
        f.write(decrypted)


# =============================================================================
# DuckMem Class (Thin Wrapper)
# =============================================================================


class DuckMem:
    """Main DuckMem interface for knowledge base operations.

    A thin wrapper around the functional API that manages database
    connections and provides a convenient object-oriented interface.

    Attributes:
        db_path: Path to the DuckDB database file.
        settings: Configuration settings.
        conn: DuckDB connection.

    Example:
        >>> mem = DuckMem("knowledge.duckdb")
        >>> item_id = mem.add("Transformers use attention.", title="ML")
        >>> results = mem.search("attention")
        >>> answer = await mem.ask("What do transformers use?")
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize DuckMem.

        Args:
            db_path: Path to DuckDB file. Uses settings.db_path if not provided.
            settings: Configuration settings. Uses defaults if not provided.
        """
        self.settings = settings or get_settings()
        self.db_path = str(db_path) if db_path else self.settings.db_path

        self.conn = duckdb.connect(self.db_path)
        self._active_session: str | None = None

        # Initialize schema and extensions
        init_schema(self.conn, embed_dim=self.settings.embed_dim)
        self._extensions = check_extensions(self.conn)

        # Initialize indexes
        with contextlib.suppress(duckdb.Error):
            init_fts_index(self.conn)

        if self._extensions.get("vss"):
            with contextlib.suppress(duckdb.Error):
                init_hnsw_index(self.conn)

        if self._extensions.get("duckpgq"):
            with contextlib.suppress(duckdb.Error):
                init_property_graph(self.conn)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> DuckMem:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()

    # -------------------------------------------------------------------------
    # Item Methods
    # -------------------------------------------------------------------------

    def add(
        self,
        text: str,
        *,
        title: str | None = None,
        uri: str | None = None,
        namespace: str = "default",
        label: str | None = None,
        metadata: dict | None = None,
        skip_embedding: bool = False,
    ) -> str:
        """Add an item to the knowledge base.

        Args:
            text: Text content to ingest.
            title: Optional title.
            uri: Optional URI reference.
            namespace: Categorization namespace.
            label: Optional label for filtering.
            metadata: Optional metadata dictionary.
            skip_embedding: Skip embedding computation.

        Returns:
            The generated item ID.
        """
        chunk_config = ChunkConfig(
            strategy=self.settings.chunk_strategy,
            max_chars=self.settings.chunk_max_chars,
            overlap=self.settings.chunk_overlap,
            min_chars=self.settings.chunk_min_chars,
        )

        item_id = add_item(
            self.conn,
            text,
            title=title,
            uri=uri,
            namespace=namespace,
            label=label,
            metadata=metadata,
            embed_model=self.settings.embed_model,
            chunk_config=chunk_config,
            skip_embedding=skip_embedding,
        )

        # Log to session if active
        if self._active_session:
            session_log_event(
                self.conn,
                self._active_session,
                "add",
                {"title": title, "namespace": namespace},
                {"item_id": item_id},
            )

        return item_id

    def get(self, item_id: str) -> Item | None:
        """Get an item by ID."""
        return get_item(self.conn, item_id)

    def list(
        self,
        *,
        namespace: str | None = None,
        label: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Item]:
        """List items with optional filtering."""
        return list_items(
            self.conn,
            namespace=namespace,
            label=label,
            limit=limit,
            offset=offset,
        )

    # -------------------------------------------------------------------------
    # Search Methods
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        top_k: int = 10,
        namespace: str | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[SearchResult]:
        """Search for items matching the query.

        Args:
            query: Search query string.
            mode: "hybrid", "lexical", or "semantic".
            top_k: Maximum results.
            namespace: Filter by namespace.
            start_ts: Filter by minimum timestamp.
            end_ts: Filter by maximum timestamp.

        Returns:
            List of SearchResult objects.
        """
        results = search(
            self.conn,
            query,
            mode=mode,
            top_k=top_k,
            namespace=namespace,
            start_ts=start_ts,
            end_ts=end_ts,
            embed_model=self.settings.embed_model,
            rrf_k=self.settings.rrf_k,
        )

        if self._active_session:
            session_log_event(
                self.conn,
                self._active_session,
                "search",
                {"query": query, "mode": mode, "top_k": top_k},
                {"count": len(results)},
            )

        return results

    async def ask(
        self,
        question: str,
        *,
        model: str | None = None,
        top_k: int = 5,
        namespace: str | None = None,
    ) -> Answer:
        """Ask a question using RAG.

        Args:
            question: The question to answer.
            model: LLM model (uses settings default if not provided).
            top_k: Number of context chunks to retrieve.
            namespace: Filter search by namespace.

        Returns:
            Answer object with response and sources.
        """
        from duckmem.agents import run_rag_query

        model = model or self.settings.llm_model

        # Search for relevant context
        results = self.search(question, top_k=top_k, namespace=namespace)

        context_chunks = [r.chunk.text for r in results]
        chunk_ids = [r.chunk.id for r in results]

        answer = await run_rag_query(
            question,
            context_chunks,
            model=model,
            chunk_ids=chunk_ids,
        )

        if self._active_session:
            session_log_event(
                self.conn,
                self._active_session,
                "ask",
                {"question": question, "model": model},
                {"confidence": answer.confidence, "sources": len(answer.sources)},
            )

        return answer

    # -------------------------------------------------------------------------
    # Knowledge Graph Methods
    # -------------------------------------------------------------------------

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        item_id: str | None = None,
    ) -> str:
        """Add a relation to the knowledge graph."""
        relation_id = add_relation(
            self.conn, subject, predicate, obj, item_id=item_id
        )

        if self._active_session:
            session_log_event(
                self.conn,
                self._active_session,
                "add_relation",
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "item_id": item_id,
                },
                {"relation_id": relation_id},
            )

        return relation_id

    def state(self, entity: str) -> EntityState:
        """Get current state of an entity."""
        return get_entity_state(self.conn, entity)

    def history(
        self,
        entity: str,
        predicate: str | None = None,
    ) -> list[Relation]:
        """Get relation history for an entity."""
        return get_entity_history(self.conn, entity, predicate)

    def traverse(
        self,
        start: str,
        *,
        link: str | None = None,
        max_hops: int = 3,
    ) -> TraversalResult:
        """Traverse the knowledge graph."""
        return traverse_graph(self.conn, start, link=link, max_hops=max_hops)

    async def extract(
        self,
        item_id: str,
        *,
        model: str | None = None,
    ) -> list[str]:
        """Extract relations from an item using LLM."""
        model = model or self.settings.llm_model
        relation_ids = await extract_relations_from_item(
            self.conn, item_id, model=model
        )

        if self._active_session:
            session_log_event(
                self.conn,
                self._active_session,
                "extract",
                {"item_id": item_id, "model": model},
                {"relation_ids": relation_ids, "count": len(relation_ids)},
            )

        return relation_ids

    # -------------------------------------------------------------------------
    # Session Methods
    # -------------------------------------------------------------------------

    def session_start(self, name: str | None = None) -> str:
        """Start a recording session."""
        session_id = session_start(self.conn, name)
        self._active_session = session_id
        return session_id

    def session_end(self) -> None:
        """End the current recording session."""
        if self._active_session:
            session_end(self.conn, self._active_session)
            self._active_session = None

    def session_replay(self, session_id: str) -> list[SessionEvent]:
        """Replay events from a session."""
        return session_replay(self.conn, session_id)

    def session_list(self) -> list[Session]:
        """List all sessions."""
        return session_list(self.conn)

    # -------------------------------------------------------------------------
    # Maintenance Methods
    # -------------------------------------------------------------------------

    def verify(self, *, deep: bool = False) -> VerifyResult:
        """Verify database integrity."""
        return verify(self.conn, deep=deep)

    def doctor(
        self,
        *,
        vacuum: bool = False,
        rebuild_fts: bool = False,
        rebuild_vec: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict[str, bool]:
        """Perform maintenance operations."""
        effective_timeout = timeout_seconds or self.settings.doctor_timeout_seconds
        return doctor(
            self.conn,
            vacuum=vacuum,
            rebuild_fts=rebuild_fts,
            rebuild_vec=rebuild_vec,
            timeout_seconds=effective_timeout,
        )

    def stats(self) -> Stats:
        """Get database statistics."""
        return stats(self.conn, self.db_path)
