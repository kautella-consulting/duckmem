"""Configuration management for DuckMem.

Uses Pydantic Settings for type-safe configuration from environment variables
or .env files. All settings can be overridden with DUCKMEM_ prefix.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """DuckMem configuration via environment variables or .env file.

    All settings can be overridden with the DUCKMEM_ prefix in environment
    variables. For example, DUCKMEM_EMBED_MODEL overrides embed_model.

    Attributes:
        db_path: Path to the DuckDB database file.
        embed_model: LiteLLM model string for embeddings.
        embed_dim: Embedding vector dimension (must match model output).
        chunk_strategy: Text chunking strategy.
        chunk_max_chars: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        chunk_min_chars: Minimum characters for a valid chunk.
        llm_model: LiteLLM model string for RAG and extraction (e.g. ollama/gpt-oss:20b).
        api_host: FastAPI server host.
        api_port: FastAPI server port.

    Example:
        >>> settings = Settings()
        >>> settings.embed_model
        'ollama/qwen3-embedding:latest'

        # Override via environment:
        # export DUCKMEM_EMBED_MODEL="openai/text-embedding-3-small"
        >>> import os
        >>> os.environ["DUCKMEM_EMBED_MODEL"] = "openai/text-embedding-3-small"
        >>> Settings().embed_model
        'openai/text-embedding-3-small'
    """

    model_config = SettingsConfigDict(
        env_prefix="DUCKMEM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    db_path: str = "duckmem.duckdb"

    # Embedding configuration
    embed_model: str = "ollama/qwen3-embedding:latest"
    embed_dim: int = 4096

    # Chunking configuration
    chunk_strategy: Literal["fixed", "markdown", "sentence"] = "markdown"
    chunk_max_chars: int = 1000
    chunk_overlap: int = 100
    chunk_min_chars: int = 50

    # LLM configuration (for RAG/extraction) - LiteLLM model string
    llm_model: str = "ollama/gpt-oss:20b"

    # Server configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # BM25 parameters
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # HNSW parameters
    hnsw_ef_search: int = 64
    hnsw_m: int = 16

    # RRF parameter
    rrf_k: int = 60

    # Maintenance
    doctor_timeout_seconds: float | None = None


def get_settings() -> Settings:
    """Get application settings instance.

    Creates a new Settings instance, loading values from environment
    variables and .env file.

    Returns:
        Configured Settings instance.

    Example:
        >>> settings = get_settings()
        >>> settings.db_path
        'duckmem.duckdb'
    """
    return Settings()
