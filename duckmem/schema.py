"""Database schema definitions for DuckMem.

Contains all DDL statements for creating tables, indexes, and the property graph.
The schema is designed for a single-file DuckDB database with FTS, VSS, and
optional DuckPGQ extensions.
"""

import duckdb

# Embedding dimension placeholder - replaced at runtime based on config
EMBED_DIM = 4096

ITEMS_TABLE = """
CREATE TABLE IF NOT EXISTS items (
    id          VARCHAR PRIMARY KEY,
    title       VARCHAR,
    uri         VARCHAR,
    namespace   VARCHAR DEFAULT 'default',
    label       VARCHAR,
    text        VARCHAR NOT NULL,
    metadata    JSON,
    created_at  BIGINT NOT NULL,
    checksum    VARCHAR NOT NULL
)
"""

CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id          VARCHAR PRIMARY KEY,
    item_id     VARCHAR NOT NULL REFERENCES items(id),
    seq         INTEGER NOT NULL,
    text        VARCHAR NOT NULL,
    embedding   FLOAT[{dim}]
)
"""

ENTITIES_TABLE = """
CREATE TABLE IF NOT EXISTS entities (
    name        VARCHAR PRIMARY KEY,
    kind        VARCHAR DEFAULT 'unknown',
    first_seen  BIGINT NOT NULL
)
"""

RELATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS relations (
    id          VARCHAR PRIMARY KEY,
    subject     VARCHAR NOT NULL,
    predicate   VARCHAR NOT NULL,
    object      VARCHAR NOT NULL,
    item_id     VARCHAR REFERENCES items(id),
    created_at  BIGINT NOT NULL
)
"""

SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id          VARCHAR PRIMARY KEY,
    name        VARCHAR,
    started_at  BIGINT NOT NULL,
    ended_at    BIGINT
)
"""

SESSION_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS session_events (
    id              VARCHAR PRIMARY KEY,
    session_id      VARCHAR NOT NULL REFERENCES sessions(id),
    timestamp       BIGINT NOT NULL,
    kind            VARCHAR NOT NULL,
    params          JSON,
    result_summary  JSON
)
"""

TABLES_TABLE = """
CREATE TABLE IF NOT EXISTS extracted_tables (
    id          VARCHAR PRIMARY KEY,
    item_id     VARCHAR REFERENCES items(id),
    headers     JSON,
    rows        JSON,
    created_at  BIGINT NOT NULL
)
"""

# Index definitions
FTS_INDEX = """
PRAGMA create_fts_index(
    'chunks', 'id', 'text',
    stemmer='porter',
    stopwords='english',
    overwrite=1
)
"""

HNSW_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chunk_embedding ON chunks
USING HNSW (embedding)
WITH (metric = 'cosine')
"""

# Property graph definition (for DuckPGQ)
PROPERTY_GRAPH = """
CREATE OR REPLACE PROPERTY GRAPH kg
VERTEX TABLES (
    entities LABEL Entity
)
EDGE TABLES (
    relations
        SOURCE KEY (subject) REFERENCES entities (name)
        DESTINATION KEY (object) REFERENCES entities (name)
        LABEL Rel
)
"""


def init_schema(conn: duckdb.DuckDBPyConnection, embed_dim: int = EMBED_DIM) -> None:
    """Initialize the database schema.

    Creates all tables required for DuckMem operation. This function is
    idempotent - it can be called multiple times without error.

    Args:
        conn: DuckDB connection.
        embed_dim: Embedding vector dimension for the chunks table.

    Example:
        >>> conn = duckdb.connect(":memory:")
        >>> init_schema(conn, embed_dim=384)
        >>> conn.execute("SELECT * FROM items").fetchall()
        []
    """
    conn.execute(ITEMS_TABLE)
    conn.execute(CHUNKS_TABLE.format(dim=embed_dim))
    conn.execute(ENTITIES_TABLE)
    conn.execute(RELATIONS_TABLE)
    conn.execute(SESSIONS_TABLE)
    conn.execute(SESSION_EVENTS_TABLE)
    conn.execute(TABLES_TABLE)


def init_fts_index(conn: duckdb.DuckDBPyConnection) -> None:
    """Initialize or rebuild the FTS (full-text search) index.

    Creates a BM25 index on the chunks table for lexical search.
    This must be called after inserting chunks, as DuckDB FTS does
    not auto-update on INSERT.

    Args:
        conn: DuckDB connection.

    Example:
        >>> conn = duckdb.connect(":memory:")
        >>> init_schema(conn)
        >>> init_fts_index(conn)
    """
    conn.execute(FTS_INDEX)


def init_hnsw_index(conn: duckdb.DuckDBPyConnection) -> None:
    """Initialize the HNSW vector index.

    Creates an HNSW index on chunk embeddings for fast approximate
    nearest neighbor search. Requires the VSS extension.

    Args:
        conn: DuckDB connection.

    Raises:
        duckdb.Error: If VSS extension is not available.

    Example:
        >>> conn = duckdb.connect(":memory:")
        >>> init_schema(conn)
        >>> init_hnsw_index(conn)
    """
    try:
        conn.execute("INSTALL vss; LOAD vss;")
        conn.execute("SET hnsw_enable_experimental_persistence = true;")
        conn.execute(HNSW_INDEX)
    except duckdb.Error:
        pass


def init_property_graph(conn: duckdb.DuckDBPyConnection) -> bool:
    """Initialize the property graph for DuckPGQ queries.

    Creates the property graph linking entities and relations.
    Returns False if DuckPGQ is not available (falls back to CTE).

    Args:
        conn: DuckDB connection.

    Returns:
        True if property graph was created, False if DuckPGQ unavailable.

    Example:
        >>> conn = duckdb.connect(":memory:")
        >>> init_schema(conn)
        >>> has_pgq = init_property_graph(conn)
    """
    try:
        conn.execute("INSTALL duckpgq; LOAD duckpgq;")
        conn.execute(PROPERTY_GRAPH)
        return True
    except duckdb.Error:
        return False


def check_extensions(conn: duckdb.DuckDBPyConnection) -> dict[str, bool]:
    """Check which DuckDB extensions are available.

    Args:
        conn: DuckDB connection.

    Returns:
        Dictionary mapping extension names to availability.

    Example:
        >>> conn = duckdb.connect(":memory:")
        >>> exts = check_extensions(conn)
        >>> "fts" in exts
        True
    """
    extensions = {"fts": False, "vss": False, "duckpgq": False}

    # FTS is always available in recent DuckDB
    try:
        conn.execute("INSTALL fts; LOAD fts;")
        extensions["fts"] = True
    except duckdb.Error:
        pass

    # VSS for vector search
    try:
        conn.execute("INSTALL vss; LOAD vss;")
        extensions["vss"] = True
    except duckdb.Error:
        pass

    # DuckPGQ for graph queries
    try:
        conn.execute("INSTALL duckpgq; LOAD duckpgq;")
        extensions["duckpgq"] = True
    except duckdb.Error:
        pass

    return extensions
