"""Pytest fixtures for DuckMem tests."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from duckmem.config import Settings
from duckmem.core import DuckMem
from duckmem.schema import init_schema


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path."""
    return tmp_path / "test.duckdb"


@pytest.fixture
def db_conn(tmp_path: Path) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Provide a temporary DuckDB connection with schema initialized."""
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    init_schema(conn, embed_dim=4096)
    yield conn
    conn.close()


@pytest.fixture
def mock_embed() -> Generator[MagicMock, None, None]:
    """Mock the embed function to avoid API calls."""
    with patch("duckmem.core.embed") as mock:
        # Return fake 4096-dim embeddings
        mock.return_value = [[0.1] * 4096]
        yield mock


@pytest.fixture
def mock_embed_single() -> Generator[MagicMock, None, None]:
    """Mock the embed_single function."""
    with patch("duckmem.core.embed_single") as mock:
        mock.return_value = [0.1] * 4096
        yield mock


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Provide test settings with temporary database."""
    return Settings(
        db_path=str(tmp_path / "test.duckdb"),
        embed_model="ollama/qwen3-embedding:latest",
        embed_dim=4096,
        chunk_max_chars=500,
        chunk_overlap=50,
    )


@pytest.fixture
def duckmem_instance(
    test_settings: Settings,
    mock_embed: MagicMock,
) -> Generator[DuckMem, None, None]:
    """Provide a DuckMem instance with mocked embeddings."""
    # Also mock embed in the inference module
    with patch("duckmem.inference.litellm") as mock_litellm:
        mock_litellm.embedding.return_value = MagicMock(data=[{"embedding": [0.1] * 4096}])
        mem = DuckMem(settings=test_settings)
        yield mem
        mem.close()


@pytest.fixture
def sample_texts() -> list[str]:
    """Provide sample texts for testing."""
    return [
        "Transformers are a type of neural network architecture.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "BERT is a transformer-based model for natural language processing.",
        "GPT models use transformer decoders for text generation.",
        "The attention mechanism computes weighted sums of values.",
    ]


@pytest.fixture
def sample_relations() -> list[tuple[str, str, str]]:
    """Provide sample relations for testing."""
    return [
        ("Alice", "works_at", "Acme Corp"),
        ("Alice", "role", "Software Engineer"),
        ("Acme Corp", "located_in", "New York"),
        ("Bob", "works_at", "Acme Corp"),
        ("Bob", "reports_to", "Alice"),
    ]
