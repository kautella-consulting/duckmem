"""Tests for core DuckMem functionality."""

from unittest.mock import MagicMock, patch

import duckdb
import pytest

from duckmem.core import (
    DuckMem,
    add_item,
    add_relation,
    get_entity_history,
    get_entity_state,
    get_item,
    list_items,
    session_end,
    session_list,
    session_log_event,
    session_replay,
    session_start,
    stats,
    traverse_graph,
    verify,
)
from duckmem.ingestion.chunkers import ChunkConfig


class TestAddItem:
    """Tests for item ingestion."""

    def test_add_item_basic(self, db_conn: duckdb.DuckDBPyConnection, mock_embed: MagicMock):
        """Test basic item addition."""
        item_id = add_item(
            db_conn,
            "This is a test document.",
            title="Test",
            skip_embedding=True,
        )

        assert item_id is not None
        assert len(item_id) == 22

        # Verify item was stored
        item = get_item(db_conn, item_id)
        assert item is not None
        assert item.title == "Test"
        assert "test document" in item.text

    def test_add_item_with_metadata(self, db_conn: duckdb.DuckDBPyConnection):
        """Test item addition with metadata."""
        metadata = {"source": "test", "version": 1}
        item_id = add_item(
            db_conn,
            "Document with metadata.",
            metadata=metadata,
            skip_embedding=True,
        )

        item = get_item(db_conn, item_id)
        assert item is not None
        assert item.metadata == metadata

    def test_add_item_empty_text_raises(self, db_conn: duckdb.DuckDBPyConnection):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            add_item(db_conn, "", skip_embedding=True)

    def test_add_item_creates_chunks(self, db_conn: duckdb.DuckDBPyConnection):
        """Test that chunks are created."""
        text = "First sentence. " * 50  # Long enough to chunk
        item_id = add_item(
            db_conn,
            text,
            chunk_config=ChunkConfig(max_chars=200, overlap=20),
            skip_embedding=True,
        )

        chunks = db_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE item_id = ?", [item_id]
        ).fetchone()
        assert chunks[0] > 1


class TestListItems:
    """Tests for listing items."""

    def test_list_items_empty(self, db_conn: duckdb.DuckDBPyConnection):
        """Test listing when no items exist."""
        items = list_items(db_conn)
        assert items == []

    def test_list_items_with_namespace_filter(self, db_conn: duckdb.DuckDBPyConnection):
        """Test filtering by namespace."""
        add_item(db_conn, "Doc 1", namespace="ns1", skip_embedding=True)
        add_item(db_conn, "Doc 2", namespace="ns2", skip_embedding=True)
        add_item(db_conn, "Doc 3", namespace="ns1", skip_embedding=True)

        ns1_items = list_items(db_conn, namespace="ns1")
        assert len(ns1_items) == 2

        ns2_items = list_items(db_conn, namespace="ns2")
        assert len(ns2_items) == 1


class TestKnowledgeGraph:
    """Tests for knowledge graph operations."""

    def test_add_relation(self, db_conn: duckdb.DuckDBPyConnection):
        """Test adding a relation."""
        rel_id = add_relation(db_conn, "Alice", "works_at", "Acme")

        assert rel_id is not None
        assert len(rel_id) == 22

        # Check entities were created
        entities = db_conn.execute("SELECT name FROM entities").fetchall()
        names = [e[0] for e in entities]
        assert "Alice" in names
        assert "Acme" in names

    def test_get_entity_state(self, db_conn: duckdb.DuckDBPyConnection, sample_relations: list):
        """Test getting entity state."""
        for s, p, o in sample_relations:
            add_relation(db_conn, s, p, o)

        state = get_entity_state(db_conn, "Alice")
        assert state.entity == "Alice"
        assert "works_at" in state.properties
        assert state.properties["works_at"] == "Acme Corp"
        assert "role" in state.properties

    def test_get_entity_history(self, db_conn: duckdb.DuckDBPyConnection, sample_relations: list):
        """Test getting entity history."""
        for s, p, o in sample_relations:
            add_relation(db_conn, s, p, o)

        history = get_entity_history(db_conn, "Alice")
        assert len(history) == 2

        # Filter by predicate
        history_filtered = get_entity_history(db_conn, "Alice", predicate="works_at")
        assert len(history_filtered) == 1

    def test_traverse_graph(self, db_conn: duckdb.DuckDBPyConnection, sample_relations: list):
        """Test graph traversal."""
        for s, p, o in sample_relations:
            add_relation(db_conn, s, p, o)

        result = traverse_graph(db_conn, "Alice", max_hops=2)

        assert "Alice" in result.entities
        assert "Acme Corp" in result.entities
        assert len(result.paths) > 0


class TestSessions:
    """Tests for session recording."""

    def test_session_lifecycle(self, db_conn: duckdb.DuckDBPyConnection):
        """Test session start/end."""
        session_id = session_start(db_conn, "Test Session")
        assert session_id is not None

        session_end(db_conn, session_id)

        sessions = session_list(db_conn)
        assert len(sessions) == 1
        assert sessions[0].name == "Test Session"
        assert sessions[0].ended_at is not None

    def test_session_log_event(self, db_conn: duckdb.DuckDBPyConnection):
        """Test logging events to session."""
        session_id = session_start(db_conn, "Log Test")

        event_id = session_log_event(
            db_conn,
            session_id,
            "add",
            {"title": "Test"},
            {"item_id": "abc123"},
        )

        events = session_replay(db_conn, session_id)
        assert len(events) == 1
        assert events[0].kind == "add"
        assert events[0].params["title"] == "Test"


class TestMaintenance:
    """Tests for maintenance operations."""

    def test_verify_empty_db(self, db_conn: duckdb.DuckDBPyConnection):
        """Test verification on empty database."""
        result = verify(db_conn)

        assert result.items == 0
        assert result.chunks == 0
        assert result.errors == []

    def test_verify_with_items(self, db_conn: duckdb.DuckDBPyConnection):
        """Test verification with items."""
        add_item(db_conn, "Test document 1", skip_embedding=True)
        add_item(db_conn, "Test document 2", skip_embedding=True)

        result = verify(db_conn)
        assert result.items == 2

    def test_verify_deep(self, db_conn: duckdb.DuckDBPyConnection):
        """Test deep verification with checksums."""
        add_item(db_conn, "Test document", skip_embedding=True)

        result = verify(db_conn, deep=True)
        assert result.checksum_ok is True

    def test_stats(self, db_conn: duckdb.DuckDBPyConnection):
        """Test stats collection."""
        add_item(db_conn, "Test document", skip_embedding=True)
        add_relation(db_conn, "A", "rel", "B")

        result = stats(db_conn)
        assert result.items == 1
        assert result.relations == 1
        assert result.entities == 2


class TestDuckMemClass:
    """Tests for the DuckMem wrapper class."""

    def test_context_manager(self, test_settings, mock_embed):
        """Test DuckMem as context manager."""
        with patch("duckmem.inference.litellm"), DuckMem(settings=test_settings) as mem:
            assert mem.conn is not None

    def test_add_and_get(self, duckmem_instance: DuckMem):
        """Test add and get through class interface."""
        item_id = duckmem_instance.add(
            "Test content",
            title="Test Title",
            skip_embedding=True,
        )

        item = duckmem_instance.get(item_id)
        assert item is not None
        assert item.title == "Test Title"

    def test_session_integration(self, duckmem_instance: DuckMem):
        """Test session tracking through class."""
        session_id = duckmem_instance.session_start("Integration Test")

        duckmem_instance.add("Session test", skip_embedding=True)

        duckmem_instance.session_end()

        events = duckmem_instance.session_replay(session_id)
        assert len(events) == 1
        assert events[0].kind == "add"
