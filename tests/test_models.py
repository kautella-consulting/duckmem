"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from duckmem.models import (
    Answer,
    Chunk,
    EntityState,
    Item,
    Relation,
    SearchResult,
    Stats,
)


class TestItemModel:
    """Tests for the Item model."""

    def test_item_creation(self):
        """Test basic item creation."""
        item = Item(
            id="abc123",
            text="Test content",
            checksum="sha256hash",
            created_at=1234567890,
        )
        assert item.id == "abc123"
        assert item.text == "Test content"
        assert item.namespace == "default"  # Default value

    def test_item_with_all_fields(self):
        """Test item with all optional fields."""
        item = Item(
            id="abc123",
            title="Test Title",
            text="Test content",
            uri="https://example.com",
            namespace="custom",
            label="important",
            checksum="sha256hash",
            created_at=1234567890,
            metadata={"key": "value"},
        )
        assert item.title == "Test Title"
        assert item.metadata == {"key": "value"}

    def test_item_is_frozen(self):
        """Test that items are immutable."""
        item = Item(
            id="abc123",
            text="Test",
            checksum="hash",
            created_at=123,
        )
        with pytest.raises(ValidationError):
            item.text = "Modified"  # type: ignore


class TestChunkModel:
    """Tests for the Chunk model."""

    def test_chunk_creation(self):
        """Test chunk creation."""
        chunk = Chunk(
            id="chunk1",
            item_id="item1",
            seq=0,
            text="Chunk text",
        )
        assert chunk.seq == 0
        assert chunk.embedding is None

    def test_chunk_with_embedding(self):
        """Test chunk with embedding."""
        embedding = tuple([0.1] * 384)
        chunk = Chunk(
            id="chunk1",
            item_id="item1",
            seq=0,
            text="Text",
            embedding=embedding,
        )
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384

    def test_chunk_seq_validation(self):
        """Test that seq must be non-negative."""
        with pytest.raises(ValidationError):
            Chunk(id="c", item_id="i", seq=-1, text="t")


class TestRelationModel:
    """Tests for the Relation model."""

    def test_relation_creation(self):
        """Test relation creation."""
        rel = Relation(
            id="rel1",
            subject="Alice",
            predicate="works_at",
            object="Acme",
            created_at=123,
        )
        assert rel.subject == "Alice"
        assert rel.predicate == "works_at"
        assert rel.object == "Acme"

    def test_relation_with_item_id(self):
        """Test relation linked to item."""
        rel = Relation(
            id="rel1",
            subject="Alice",
            predicate="works_at",
            object="Acme",
            item_id="item1",
            created_at=123,
        )
        assert rel.item_id == "item1"


class TestAnswerModel:
    """Tests for the Answer model."""

    def test_answer_creation(self):
        """Test answer creation."""
        answer = Answer(
            answer="The sky is blue.",
            confidence=0.95,
            sources=["chunk1", "chunk2"],
        )
        assert answer.confidence == 0.95
        assert len(answer.sources) == 2

    def test_answer_confidence_validation(self):
        """Test confidence must be 0-1."""
        with pytest.raises(ValidationError):
            Answer(answer="Test", confidence=1.5, sources=[])

        with pytest.raises(ValidationError):
            Answer(answer="Test", confidence=-0.1, sources=[])


class TestSearchResultModel:
    """Tests for the SearchResult model."""

    def test_search_result_creation(self):
        """Test search result creation."""
        chunk = Chunk(id="c1", item_id="i1", seq=0, text="Text")
        item = Item(id="i1", text="Full text", checksum="h", created_at=123)

        result = SearchResult(
            chunk=chunk,
            item=item,
            score=0.85,
            lexical_score=0.7,
            semantic_score=0.9,
        )
        assert result.score == 0.85


class TestEntityStateModel:
    """Tests for the EntityState model."""

    def test_entity_state(self):
        """Test entity state creation."""
        state = EntityState(
            entity="Alice",
            properties={
                "works_at": "Acme",
                "role": "Engineer",
            },
        )
        assert state.entity == "Alice"
        assert state.properties["works_at"] == "Acme"


class TestStatsModel:
    """Tests for the Stats model."""

    def test_stats_creation(self):
        """Test stats creation."""
        stats = Stats(
            items=100,
            chunks=500,
            relations=50,
            entities=30,
            sessions=5,
            file_size_bytes=1024000,
        )
        assert stats.items == 100

    def test_stats_non_negative(self):
        """Test that counts must be non-negative."""
        with pytest.raises(ValidationError):
            Stats(
                items=-1,
                chunks=0,
                relations=0,
                entities=0,
                sessions=0,
                file_size_bytes=0,
            )
