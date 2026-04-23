"""Text chunking strategies for document ingestion.

Provides pure functions for splitting text into chunks using different
strategies: fixed-size windows, markdown-aware splitting, and sentence-based.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

ChunkStrategy = Literal["fixed", "markdown", "sentence"]


class ChunkConfig(BaseModel, frozen=True):
    """Configuration for text chunking.

    Attributes:
        strategy: Chunking strategy to use.
        max_chars: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.
        min_chars: Minimum characters for a valid chunk.
    """

    strategy: ChunkStrategy = Field(default="markdown", description="Chunking strategy")
    max_chars: int = Field(default=1000, ge=100, description="Max chars per chunk")
    overlap: int = Field(default=100, ge=0, description="Overlap between chunks")
    min_chars: int = Field(default=50, ge=10, description="Min chars for valid chunk")


class TextChunk(BaseModel, frozen=True):
    """A chunk of text with position information.

    Attributes:
        text: The chunk text content.
        start: Start character position in original text.
        end: End character position in original text.
    """

    text: str = Field(description="Chunk text content")
    start: int = Field(ge=0, description="Start position")
    end: int = Field(ge=0, description="End position")


def chunk_text(text: str, config: ChunkConfig | None = None) -> list[TextChunk]:
    """Split text into chunks using the configured strategy.

    Pure function that takes text and configuration, returning a list
    of chunks without side effects.

    Args:
        text: The text to chunk.
        config: Chunking configuration. Uses defaults if not provided.

    Returns:
        List of TextChunk objects.

    Raises:
        ValueError: If text is empty.

    Example:
        >>> text = "This is a test. Another sentence."
        >>> chunks = chunk_text(text, ChunkConfig(strategy="sentence"))
        >>> len(chunks) >= 1
        True
    """
    if not text.strip():
        raise ValueError("text cannot be empty")

    if config is None:
        config = ChunkConfig()

    strategy_map = {
        "fixed": _chunk_fixed,
        "markdown": _chunk_markdown,
        "sentence": _chunk_sentence,
    }

    chunker = strategy_map[config.strategy]
    chunks = chunker(text, config)

    # Filter out chunks that are too small
    return [c for c in chunks if len(c.text.strip()) >= config.min_chars]


def _chunk_fixed(text: str, config: ChunkConfig) -> list[TextChunk]:
    """Split text into fixed-size chunks with overlap.

    Simple sliding window approach. Good for uniform text without
    clear structure.

    Args:
        text: Text to chunk.
        config: Chunking configuration.

    Returns:
        List of TextChunk objects.
    """
    chunks: list[TextChunk] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + config.max_chars, text_len)

        # Try to break at word boundary
        if end < text_len:
            last_space = text.rfind(" ", start, end)
            if last_space > start + config.max_chars // 2:
                end = last_space

        chunk_text_content = text[start:end].strip()
        if chunk_text_content:
            chunks.append(TextChunk(text=chunk_text_content, start=start, end=end))

        # Move forward, accounting for overlap
        # Ensure we always make progress and start never goes negative
        next_start = end - config.overlap
        if next_start <= start:
            next_start = end
        start = max(0, next_start)

        # If we've reached the end, stop
        if end >= text_len:
            break

    return chunks


def _chunk_markdown(text: str, config: ChunkConfig) -> list[TextChunk]:
    """Split text at markdown headings, then subdivide large sections.

    Respects document structure by splitting at headers first,
    then applying fixed chunking to large sections.

    Args:
        text: Text to chunk (may contain markdown).
        config: Chunking configuration.

    Returns:
        List of TextChunk objects.
    """
    # Pattern to match markdown headings
    heading_pattern = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)

    # Find all heading positions
    headings = list(heading_pattern.finditer(text))

    if not headings:
        # No headings found, fall back to fixed chunking
        return _chunk_fixed(text, config)

    chunks: list[TextChunk] = []

    # Add content before first heading if any
    if headings[0].start() > 0:
        preamble = text[: headings[0].start()].strip()
        if preamble:
            chunks.extend(_subdivide_section(preamble, 0, headings[0].start(), config))

    # Process each section
    for i, heading in enumerate(headings):
        section_start = heading.start()
        section_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)

        section_text = text[section_start:section_end].strip()
        if section_text:
            chunks.extend(_subdivide_section(section_text, section_start, section_end, config))

    return chunks


def _subdivide_section(
    text: str, start_offset: int, end_offset: int, config: ChunkConfig
) -> list[TextChunk]:
    """Subdivide a section if it exceeds max_chars.

    Args:
        text: Section text.
        start_offset: Start position in original document.
        end_offset: End position in original document.
        config: Chunking configuration.

    Returns:
        List of TextChunk objects.
    """
    if len(text) <= config.max_chars:
        return [TextChunk(text=text, start=start_offset, end=end_offset)]

    # Section is too large, split further
    sub_chunks = _chunk_fixed(text, config)

    # Adjust positions to original document
    return [
        TextChunk(
            text=c.text,
            start=start_offset + c.start,
            end=start_offset + c.end,
        )
        for c in sub_chunks
    ]


def _chunk_sentence(text: str, config: ChunkConfig) -> list[TextChunk]:
    """Split text into chunks at sentence boundaries.

    Groups sentences together until reaching max_chars, then starts
    a new chunk. Good for prose text.

    Args:
        text: Text to chunk.
        config: Chunking configuration.

    Returns:
        List of TextChunk objects.
    """
    # Simple sentence splitting pattern
    # Handles . ! ? followed by space or end of string
    sentence_pattern = re.compile(r"(?<=[.!?])\s+|\n\n+")

    sentences = sentence_pattern.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [TextChunk(text=text.strip(), start=0, end=len(text))]

    chunks: list[TextChunk] = []
    current_sentences: list[str] = []
    current_start = 0
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed max_chars, finalize current chunk
        if current_len + sentence_len + 1 > config.max_chars and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_end = text.find(current_sentences[-1], current_start)
            chunk_end += len(current_sentences[-1])

            chunks.append(TextChunk(text=chunk_text, start=current_start, end=chunk_end))

            # Start new chunk with overlap (keep last sentence if it fits)
            if config.overlap > 0 and current_sentences:
                overlap_text = current_sentences[-1]
                if len(overlap_text) <= config.overlap:
                    current_sentences = [overlap_text]
                    current_len = len(overlap_text)
                    current_start = text.find(overlap_text, current_start)
                else:
                    current_sentences = []
                    current_len = 0
                    current_start = text.find(sentence)
            else:
                current_sentences = []
                current_len = 0
                current_start = text.find(sentence)

        current_sentences.append(sentence)
        current_len += sentence_len + 1  # +1 for space

    # Add final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(TextChunk(text=chunk_text, start=current_start, end=len(text)))

    return chunks
