"""Tests for text chunking strategies."""

import pytest

from duckmem.ingestion.chunkers import ChunkConfig, chunk_text


class TestChunkConfig:
    """Tests for ChunkConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkConfig()
        assert config.strategy == "markdown"
        assert config.max_chars == 1000
        assert config.overlap == 100
        assert config.min_chars == 50

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkConfig(
            strategy="fixed",
            max_chars=500,
            overlap=50,
            min_chars=20,
        )
        assert config.strategy == "fixed"
        assert config.max_chars == 500


class TestFixedChunking:
    """Tests for fixed-size chunking strategy."""

    def test_short_text(self):
        """Test text shorter than max_chars."""
        text = "Short text that is long enough to pass minimum."
        config = ChunkConfig(strategy="fixed", max_chars=200, min_chars=10)

        chunks = chunk_text(text, config)
        assert len(chunks) == 1
        assert "Short text" in chunks[0].text

    def test_long_text_creates_multiple_chunks(self):
        """Test that long text is split into multiple chunks."""
        text = "Word " * 100  # 500 chars
        config = ChunkConfig(strategy="fixed", max_chars=100, overlap=20, min_chars=10)

        chunks = chunk_text(text, config)
        assert len(chunks) > 1

    def test_overlap_between_chunks(self):
        """Test that chunks have overlap."""
        text = "Word " * 100  # 500 chars
        config = ChunkConfig(strategy="fixed", max_chars=150, overlap=30, min_chars=10)

        chunks = chunk_text(text, config)
        # With overlap, should have multiple chunks
        assert len(chunks) >= 2


class TestMarkdownChunking:
    """Tests for markdown-aware chunking strategy."""

    def test_split_on_headings(self):
        """Test that markdown headings create chunk boundaries."""
        text = """# Heading 1

Content under heading 1.

## Heading 2

Content under heading 2.

## Heading 3

Content under heading 3."""

        config = ChunkConfig(strategy="markdown", max_chars=1000, min_chars=10)
        chunks = chunk_text(text, config)

        # Should create chunks for each section
        assert len(chunks) >= 2

    def test_large_section_subdivided(self):
        """Test that large sections are subdivided."""
        text = (
            """# Big Section

"""
            + "Content. " * 200
        )  # Large section

        config = ChunkConfig(strategy="markdown", max_chars=200, min_chars=20)
        chunks = chunk_text(text, config)

        # Large section should be split
        assert len(chunks) > 1

    def test_no_headings_falls_back(self):
        """Test that text without headings uses fixed chunking."""
        text = "Plain text without any markdown headings. " * 20
        config = ChunkConfig(strategy="markdown", max_chars=200, min_chars=20)

        chunks = chunk_text(text, config)
        assert len(chunks) > 1


class TestSentenceChunking:
    """Tests for sentence-based chunking strategy."""

    def test_basic_sentences(self):
        """Test splitting on sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        config = ChunkConfig(strategy="sentence", max_chars=1000, min_chars=10)

        chunks = chunk_text(text, config)
        # All sentences should be in one chunk since under max_chars
        assert len(chunks) == 1
        assert "First" in chunks[0].text
        assert "Third" in chunks[0].text

    def test_long_sentences_split(self):
        """Test that many sentences create multiple chunks."""
        text = ". ".join([f"This is sentence number {i}" for i in range(20)]) + "."
        config = ChunkConfig(strategy="sentence", max_chars=100, min_chars=20)

        chunks = chunk_text(text, config)
        assert len(chunks) > 1

    def test_question_marks(self):
        """Test that question marks are treated as sentence ends."""
        text = "What is AI? It is artificial intelligence. How does it work? Through algorithms."
        config = ChunkConfig(strategy="sentence", max_chars=1000, min_chars=10)

        chunks = chunk_text(text, config)
        assert len(chunks) >= 1


class TestChunkTextFunction:
    """Tests for the main chunk_text function."""

    def test_empty_text_raises(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            chunk_text("", ChunkConfig())

    def test_whitespace_only_raises(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            chunk_text("   \n\t  ", ChunkConfig())

    def test_default_config_used(self):
        """Test that default config is used when not provided."""
        text = "Some test content that should be chunked properly."
        chunks = chunk_text(text)  # No config provided

        assert len(chunks) >= 1

    def test_min_chars_filter(self):
        """Test that chunks below min_chars are filtered."""
        text = "A short. B short. C short. D short. E short."
        config = ChunkConfig(strategy="sentence", max_chars=100, min_chars=50)

        chunks = chunk_text(text, config)
        # All returned chunks should meet min_chars
        for chunk in chunks:
            assert len(chunk.text.strip()) >= config.min_chars

    def test_chunk_positions(self):
        """Test that chunk positions are tracked."""
        text = "First part of the text. Second part of the text here."
        config = ChunkConfig(strategy="fixed", max_chars=100, min_chars=10)

        chunks = chunk_text(text, config)
        assert chunks[0].start >= 0
        assert chunks[0].end > chunks[0].start
