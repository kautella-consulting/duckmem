"""Pydantic models for DuckMem data structures.

All models use frozen=True for immutability, following functional programming
principles. Models are used for both internal data representation and API
responses.
"""

from pydantic import BaseModel, Field


class Item(BaseModel, frozen=True):
    """An ingested document in the knowledge base.

    Items are the top-level documents stored in DuckMem. Each item
    is chunked into smaller segments for search and embedding.

    Attributes:
        id: Unique identifier for the item.
        title: Optional human-readable title.
        text: Full text content of the item.
        uri: Optional URI/URL reference.
        namespace: Categorization namespace (default: "default").
        label: Optional label for filtering.
        checksum: SHA-256 hash of the text content.
        created_at: Unix timestamp in milliseconds when created.
        metadata: Optional arbitrary metadata dictionary.
    """

    id: str = Field(description="Unique identifier")
    title: str | None = Field(default=None, description="Human-readable title")
    text: str = Field(description="Full text content")
    uri: str | None = Field(default=None, description="URI/URL reference")
    namespace: str = Field(default="default", description="Categorization namespace")
    label: str | None = Field(default=None, description="Optional label for filtering")
    checksum: str = Field(description="SHA-256 hash of text content")
    created_at: int = Field(description="Unix timestamp in milliseconds")
    metadata: dict | None = Field(default=None, description="Arbitrary metadata")


class Chunk(BaseModel, frozen=True):
    """A text segment with its embedding.

    Chunks are the atomic units of search in DuckMem. Each item is
    split into chunks based on the configured chunking strategy.

    Attributes:
        id: Unique identifier for the chunk.
        item_id: ID of the parent item.
        seq: Sequence number within the item (0-indexed).
        text: Text content of this chunk.
        embedding: Vector embedding as a tuple of floats.
    """

    id: str = Field(description="Unique identifier")
    item_id: str = Field(description="Parent item ID")
    seq: int = Field(ge=0, description="Sequence number within item")
    text: str = Field(description="Chunk text content")
    embedding: tuple[float, ...] | None = Field(default=None, description="Vector embedding")


class Entity(BaseModel, frozen=True):
    """A node in the knowledge graph.

    Entities represent named things (people, organizations, concepts)
    that appear in relations.

    Attributes:
        name: Unique name identifying the entity.
        kind: Entity type (person, organization, concept, etc.).
        first_seen: Timestamp when entity was first extracted.
    """

    name: str = Field(description="Unique entity name")
    kind: str = Field(default="unknown", description="Entity type")
    first_seen: int = Field(description="Timestamp when first seen")


class Relation(BaseModel, frozen=True):
    """A subject-predicate-object fact in the knowledge graph.

    Relations connect entities with typed edges, forming a knowledge
    graph that can be traversed and queried.

    Attributes:
        id: Unique identifier for the relation.
        subject: Subject entity name.
        predicate: Relationship type (e.g., "works_at", "located_in").
        object: Object entity name.
        item_id: ID of the item this relation was extracted from.
        created_at: Timestamp when relation was created.
    """

    id: str = Field(description="Unique identifier")
    subject: str = Field(description="Subject entity name")
    predicate: str = Field(description="Relationship type")
    object: str = Field(description="Object entity name")
    item_id: str | None = Field(default=None, description="Source item ID")
    created_at: int = Field(description="Timestamp when created")


class SearchResult(BaseModel, frozen=True):
    """A search result containing a matched chunk with score.

    Attributes:
        chunk: The matched chunk.
        item: The parent item.
        score: Relevance score (higher is better).
        lexical_score: BM25 lexical search score.
        semantic_score: Vector similarity score.
    """

    chunk: Chunk = Field(description="Matched chunk")
    item: Item = Field(description="Parent item")
    score: float = Field(description="Combined relevance score")
    lexical_score: float | None = Field(default=None, description="BM25 score")
    semantic_score: float | None = Field(default=None, description="Vector similarity")


class Answer(BaseModel, frozen=True):
    """Structured answer from RAG query.

    Attributes:
        answer: The generated answer text.
        confidence: Confidence score (0.0 to 1.0).
        sources: List of source chunk IDs used.
        context: The context text provided to the LLM.
    """

    answer: str = Field(description="Generated answer text")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    sources: list[str] = Field(default_factory=list, description="Source chunk IDs")
    context: str | None = Field(default=None, description="Context provided to LLM")


class ExtractedRelation(BaseModel, frozen=True):
    """A relation extracted by the LLM (before storage).

    Used as the output type for the extraction agent.

    Attributes:
        subject: Subject entity name.
        predicate: Relationship type.
        object: Object entity name.
    """

    subject: str = Field(description="Subject entity")
    predicate: str = Field(description="Relationship type")
    object: str = Field(description="Object entity")


class Session(BaseModel, frozen=True):
    """A recording session for tracking operations.

    Attributes:
        id: Unique session identifier.
        name: Human-readable session name.
        started_at: Timestamp when session started.
        ended_at: Timestamp when session ended (None if active).
    """

    id: str = Field(description="Unique session ID")
    name: str | None = Field(default=None, description="Session name")
    started_at: int = Field(description="Start timestamp")
    ended_at: int | None = Field(default=None, description="End timestamp")


class SessionEvent(BaseModel, frozen=True):
    """An event logged during a session.

    Attributes:
        id: Unique event identifier.
        session_id: Parent session ID.
        timestamp: When the event occurred.
        kind: Event type (put, find, ask, etc.).
        params: Parameters passed to the operation.
        result_summary: Summary of the result.
    """

    id: str = Field(description="Unique event ID")
    session_id: str = Field(description="Parent session ID")
    timestamp: int = Field(description="Event timestamp")
    kind: str = Field(description="Event type")
    params: dict = Field(default_factory=dict, description="Operation parameters")
    result_summary: dict = Field(default_factory=dict, description="Result summary")


class Stats(BaseModel, frozen=True):
    """Database statistics.

    Attributes:
        items: Number of items.
        chunks: Number of chunks.
        relations: Number of relations.
        entities: Number of entities.
        sessions: Number of sessions.
        file_size_bytes: Database file size in bytes.
    """

    items: int = Field(ge=0, description="Number of items")
    chunks: int = Field(ge=0, description="Number of chunks")
    relations: int = Field(ge=0, description="Number of relations")
    entities: int = Field(ge=0, description="Number of entities")
    sessions: int = Field(ge=0, description="Number of sessions")
    file_size_bytes: int = Field(ge=0, description="Database file size")


class EntityState(BaseModel, frozen=True):
    """Current state of an entity (latest-wins properties).

    Attributes:
        entity: The entity name.
        properties: Dictionary of predicate -> latest object value.
    """

    entity: str = Field(description="Entity name")
    properties: dict[str, str] = Field(default_factory=dict, description="Current property values")


class TraversalResult(BaseModel, frozen=True):
    """Result of a graph traversal.

    Attributes:
        paths: List of paths found, each path is a list of relations.
        entities: Set of unique entities encountered.
    """

    paths: list[list[Relation]] = Field(default_factory=list, description="Paths found")
    entities: list[str] = Field(default_factory=list, description="Unique entities")


class VerifyResult(BaseModel, frozen=True):
    """Result of database verification.

    Attributes:
        items: Number of items.
        chunks: Number of chunks.
        relations: Number of relations.
        entities: Number of entities.
        checksum_ok: Whether all checksums are valid (if deep=True).
        errors: List of any errors found.
    """

    items: int = Field(ge=0, description="Number of items")
    chunks: int = Field(ge=0, description="Number of chunks")
    relations: int = Field(ge=0, description="Number of relations")
    entities: int = Field(ge=0, description="Number of entities")
    checksum_ok: bool | None = Field(default=None, description="Checksum verification")
    errors: list[str] = Field(default_factory=list, description="Errors found")
