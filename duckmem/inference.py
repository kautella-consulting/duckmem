"""LiteLLM wrapper for embedding operations.

Provides a unified interface to 100+ embedding providers through LiteLLM.
Supports batching, normalization, and error handling.
"""

import litellm
from litellm import EmbeddingResponse


def embed(
    texts: list[str],
    model: str = "ollama/qwen3-embedding:latest",
    *,
    batch_size: int = 64,
    normalize: bool = True,
) -> list[list[float]]:
    """Embed texts using any LiteLLM-supported embedding model.

    Sends texts to the specified embedding model and returns vectors.
    Supports batching for large inputs and optional L2 normalization.

    Args:
        texts: List of texts to embed.
        model: LiteLLM model string (e.g., "text-embedding-3-small",
            "cohere/embed-english-v3.0", "ollama/nomic-embed-text").
        batch_size: Maximum texts per API call for batching.
        normalize: Whether to L2-normalize the resulting vectors.

    Returns:
        List of embedding vectors, one per input text. Each vector
        is a list of floats with dimension determined by the model.

    Raises:
        litellm.exceptions.APIError: If the embedding API call fails.
        ValueError: If texts list is empty.

    Example:
        >>> embeddings = embed(["Hello, world!", "Goodbye!"])
        >>> len(embeddings)
        2
        >>> len(embeddings[0])  # Dimension depends on model
        1536

        # Using a different model
        >>> embeddings = embed(["test"], model="cohere/embed-english-v3.0")
    """
    if not texts:
        raise ValueError("texts list cannot be empty")

    all_embeddings: list[list[float]] = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response: EmbeddingResponse = litellm.embedding(model=model, input=batch)

        for item in response.data:
            embedding = item["embedding"]
            if normalize:
                embedding = _normalize_vector(embedding)
            all_embeddings.append(embedding)

    return all_embeddings


def embed_single(
    text: str,
    model: str = "ollama/qwen3-embedding:latest",
    *,
    normalize: bool = True,
) -> list[float]:
    """Embed a single text string.

    Convenience function for embedding one text without batching.

    Args:
        text: Text to embed.
        model: LiteLLM model string.
        normalize: Whether to L2-normalize the vector.

    Returns:
        Embedding vector as a list of floats.

    Example:
        >>> vec = embed_single("Hello, world!")
        >>> len(vec)
        1536
    """
    return embed([text], model=model, normalize=normalize)[0]


def get_embedding_dimension(model: str = "ollama/qwen3-embedding:latest") -> int:
    """Get the embedding dimension for a model.

    Sends a test embedding to determine the output dimension.
    Results should be cached by the caller for efficiency.

    Args:
        model: LiteLLM model string.

    Returns:
        Integer dimension of the embedding vectors.

    Example:
        >>> dim = get_embedding_dimension("text-embedding-3-small")
        >>> dim
        1536
    """
    test_embedding = embed_single("test", model=model)
    return len(test_embedding)


def _normalize_vector(vec: list[float]) -> list[float]:
    """L2-normalize a vector.

    Args:
        vec: Input vector.

    Returns:
        Normalized vector with unit length.
    """
    magnitude = sum(x * x for x in vec) ** 0.5
    if magnitude == 0:
        return vec
    return [x / magnitude for x in vec]
